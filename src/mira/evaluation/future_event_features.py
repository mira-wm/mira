"""Frozen MIRA context features for causal post-context event probes.

This evaluator intentionally lives outside the model implementation.  It observes the final
diffusion-transformer block with a forward hook, so extracting representations does not alter the
released MIRA architecture or checkpoint state.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from mira.data.batch import VideoActionBatch
    from mira.world_model.latent_world_model import LatentWorldModel
    from mira.world_model.multi_wrapper_world_model import MultiWrapperWorldModel


def pool_context_hidden(hidden: Tensor, *, n_players: int) -> Tensor:
    """Pool the last causal context token grid to one feature vector per player.

    Args:
        hidden: Final-block tokens shaped ``(groups, time, tiled_height, width, dim)``.
        n_players: One for an independent single-POV model, otherwise the tiled player count.

    Returns:
        ``(groups * n_players, dim)`` vectors in group-major, player-index order.
    """
    if hidden.ndim != 5:
        raise ValueError(f"expected rank-5 hidden tokens, got {tuple(hidden.shape)}")
    if n_players < 1 or hidden.shape[2] % n_players:
        raise ValueError(
            f"hidden height {hidden.shape[2]} is not divisible by n_players={n_players}"
        )
    last = hidden[:, -1]
    if n_players == 1:
        return last.mean(dim=(1, 2))
    groups, tiled_height, width, dim = last.shape
    by_player = last.reshape(
        groups,
        n_players,
        tiled_height // n_players,
        width,
        dim,
    )
    return by_player.mean(dim=(2, 3)).reshape(groups * n_players, dim)


def _autocast(device: torch.device, dtype: torch.dtype | None):
    if device.type != "cuda" or dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


@torch.no_grad()
def extract_context_features(
    model: LatentWorldModel | MultiWrapperWorldModel,
    batch: VideoActionBatch,
    *,
    context_frames: int,
    autocast_dtype: torch.dtype | None = torch.bfloat16,
) -> Tensor:
    """Extract final-block features using context pixels/actions only.

    The context is ``[0, context_frames)``.  No later video frames or action rows are passed to the
    codec, action encoder, or diffusion transformer.  Clean latents are evaluated at flow time
    ``tau=1`` and the last causal token grid is pooled per player.
    """
    from mira.world_model.latent_world_model import LatentWorldModel
    from mira.world_model.multi_wrapper_world_model import MultiWrapperWorldModel

    if context_frames <= 0:
        raise ValueError("context_frames must be positive")
    if isinstance(model, MultiWrapperWorldModel):
        swm = model.single_world_model
        n_players = model.n_players
    elif isinstance(model, LatentWorldModel):
        swm = model
        n_players = 1
    else:
        raise TypeError(f"unsupported world model type: {type(model)!r}")
    if context_frames > batch.video.shape[1]:
        raise ValueError(
            f"context_frames={context_frames} exceeds batch length {batch.video.shape[1]}"
        )
    if context_frames % swm.temporal_downsampling:
        raise ValueError(
            f"context_frames={context_frames} must divide codec temporal stride "
            f"{swm.temporal_downsampling}"
        )
    if len(batch) % n_players:
        raise ValueError(f"batch rows {len(batch)} are not divisible by n_players={n_players}")

    model.eval()
    context = batch.slice_time(0, context_frames, fps=swm.config.video.fps)
    swm.codec.preprocess_batch(context)
    context = context.to(model.device)

    captured: list[Tensor] = []

    def capture_final_hidden(_module, _inputs, output) -> None:
        sequence = output[0] if isinstance(output, tuple) else output
        captured.append(sequence.detach())

    handle = swm.world_model.transformer[-1].register_forward_hook(capture_final_hidden)
    try:
        with _autocast(model.device, autocast_dtype):
            z_flat = swm.encode_video(context).clone()
            if isinstance(model, MultiWrapperWorldModel):
                rows, latent_frames, height, width, dim = z_flat.shape
                z = (
                    z_flat.reshape(
                        rows // n_players,
                        n_players,
                        latent_frames,
                        height,
                        width,
                        dim,
                    )
                    .permute(0, 2, 1, 3, 4, 5)
                    .reshape(rows // n_players, latent_frames, n_players * height, width, dim)
                )
            else:
                z = z_flat

            action_stride = swm.action_temporal_downsampling
            n_action_steps = (z.shape[1] - 1) * action_stride
            offset = action_stride - 1
            a_flat = swm.action_encoder(
                context.actions.slice_time(offset, offset + n_action_steps)
            )
            a = (
                model._combine_player_actions(a_flat)
                if isinstance(model, MultiWrapperWorldModel)
                else a_flat
            )
            if a.shape[1] != z.shape[1]:
                raise ValueError(
                    f"action/latent context length drifted: actions={a.shape[1]}, latents={z.shape[1]}"
                )

            clean_past = None
            if swm.config.use_clean_past:
                assert swm.bos is not None
                clean_past = torch.cat(
                    [
                        swm.bos[None, None].repeat(z.shape[0], 1, 1, 1, 1),
                        z[:, :-1],
                    ],
                    dim=1,
                )
            tau = torch.ones(
                (z.shape[0], z.shape[1], 1, 1, 1),
                device=z.device,
                dtype=z.dtype,
            )
            swm.world_model(
                z,
                a,
                tau,
                clean_past=clean_past,
                activation_checkpointing=False,
            )
    finally:
        handle.remove()

    if len(captured) != 1:
        raise RuntimeError(f"expected one final-block activation, captured {len(captured)}")
    hidden = captured[0]
    n_register = int(swm.world_model.n_register_tokens)
    if n_register:
        hidden = hidden[:, n_register:]
    return pool_context_hidden(hidden.float(), n_players=n_players).cpu()
