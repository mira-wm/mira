"""Offline autoregressive rollout of the world model, without the final decode.

:func:`rollout` reproduces the denoising loop of ``LatentWorldModel.inference`` (and its multiplayer
counterpart ``MultiWrapperWorldModel.inference``) but stops before decoding latents back to video. It
is the shared core of the latency benchmark (:mod:`scripts.bench_wm_speed`) and the offline-eval
denoise-speed measurement, and it is the determinism anchor the equality harness depends on: a fixed
seed + ``noise_level=0.0`` + a fixed schedule reproduces the generated latents exactly.

The per-step action window starts at ``off = atd - 1`` for both model variants, matching their
training and inference paths. Multiplayer additionally combines the per-player encoded streams via
the model's own ``_combine_player_actions``.

This mirrors each model's ``inference`` method exactly; if those loops change, this helper must follow.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, cast

import torch
from einops import rearrange
from torch import Tensor
from tqdm import tqdm

from mira.data.batch import VideoActionBatch
from mira.world_model.config import WorldModelInferenceConfig

if TYPE_CHECKING:
    from mira.world_model.latent_world_model import LatentWorldModel
    from mira.world_model.multi_wrapper_world_model import MultiWrapperWorldModel

    # Both variants present the surface ``rollout`` reads (config, codec, encode_video,
    # denoise_streaming, action_encoder, device, ...); the wrapper adds tiling + action combination.
    # They share no typed base, so the public type is their union and we cast for multi-only access.
    WorldModelLike = LatentWorldModel | MultiWrapperWorldModel


def _is_multiplayer(model: WorldModelLike) -> bool:
    """Whether ``model`` is the multiplayer wrapper (it alone owns an inner ``single_world_model``)."""
    return hasattr(model, "single_world_model")


def _inner_model(model: WorldModelLike) -> LatentWorldModel:
    """The single-player model that owns the codec / transformer (the wrapper's inner model, or itself)."""
    if _is_multiplayer(model):
        return cast("MultiWrapperWorldModel", model).single_world_model
    return cast("LatentWorldModel", model)


def _encode_window_actions(
    model: WorldModelLike,
    inner: LatentWorldModel,
    batch: VideoActionBatch,
    start: int,
    window_size: int,
) -> Tensor:
    """Encode the action conditioning for the denoise window starting at latent index ``start``.

    Applies the same ``off = atd - 1`` shift as both model variants' inference implementations,
    then routes per-player streams for multiplayer. Returns global ``(b,t_a,d)`` or spatial
    ``(b,t_a,h,w,d)`` conditioning ready for ``denoise_streaming``.
    """
    atd = inner.action_temporal_downsampling
    off = atd - 1
    action_start = start * atd + off
    action_end = (start + window_size - 1) * atd + off
    encoded = inner.action_encoder(batch.actions.slice_time(action_start, action_end)).clone()
    if _is_multiplayer(model):
        return cast("MultiWrapperWorldModel", model)._combine_player_actions(encoded)
    return encoded


@torch.no_grad()
def rollout(
    model: WorldModelLike,
    batch: VideoActionBatch,
    config: WorldModelInferenceConfig | None = None,
    *,
    n_frames: int | None = None,
    progress_bar: bool = False,
) -> Tensor:
    """Autoregressively denoise a clip into latents, skipping the decode step.

    Reproduces ``model.inference`` up to (but not including) ``decode_to_video``. Deterministic given
    the global RNG seed, ``config.noise_level`` and ``config.schedule_type``.

    Args:
        model: A :class:`LatentWorldModel` or :class:`MultiWrapperWorldModel`.
        batch: The bootstrap clip; its first ``n_context_latents`` frames seed the rollout.
        config: Sampling knobs; the model's defaults are used when ``None``.
        n_frames: Cap the number of denoised latent frames (windows). ``None`` rolls the whole clip.
            Used by the latency benchmark to time a fixed number of frames.
        progress_bar: Show a tqdm progress bar over the rollout.

    Returns:
        The generated latents. Single-player: ``(b, t, h, w, c)``. Multiplayer: the vertically tiled
        ``(b, t, p*h, w, c)`` buffer the loop operates on (not split back per player).
    """
    if config is None:
        config = WorldModelInferenceConfig()

    multi = _is_multiplayer(model)
    inner = _inner_model(model)

    inner.codec.preprocess_batch(batch)
    batch = batch.to(model.device)
    z = inner.encode_video(batch).clone()  # clone because of torch.compile
    if multi:
        z = rearrange(z, "(b p) t h w c -> b t (p h) w c", p=cast("MultiWrapperWorldModel", model).n_players)

    z_t = torch.randn_like(z)
    z_t[:, : inner.n_context_latents] = z[:, : inner.n_context_latents]

    window_size = inner.n_context_latents + 1
    n_iters = z.shape[1] - window_size + 1
    if n_frames is not None:
        n_iters = min(n_iters, n_frames)

    streaming_kv_caches = None
    for start in tqdm(range(n_iters), desc="Diffusion rollout", disable=not progress_bar):
        current_a = _encode_window_actions(model, inner, batch, start, window_size)
        z_t[:, start : start + window_size], streaming_kv_caches = inner.denoise_streaming(
            z_t[:, start : start + window_size],
            current_a,
            n_diffusion_steps=config.n_diffusion_steps,
            noise_level=config.noise_level,
            streaming_kv_caches=streaming_kv_caches,
            schedule_type=config.schedule_type,
        )

    return z_t


def measure_rollout_speed(
    model: WorldModelLike,
    batch: VideoActionBatch,
    config: WorldModelInferenceConfig,
    n_frames: int,
) -> dict[str, float]:
    """Time :func:`rollout` over ``n_frames`` latent frames (batch size 1, no decode / metrics).

    This is the per-stream latency that matters for the interactive server, as opposed to end-to-end
    eval throughput (which is dominated by decode and metric computation). The caller is expected to
    size the clip so the rollout produces at least ``n_frames`` windows; the timing divides by
    ``n_frames``. Run a warmup call first so compilation / cudnn autotune does not skew the result.

    Returns:
        ``{"denoise_ms_per_latent_frame": ..., "denoise_latent_fps": ...}``.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    rollout(model, batch, config, n_frames=n_frames)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    s_per_frame = (time.perf_counter() - t0) / n_frames
    return {
        "denoise_ms_per_latent_frame": s_per_frame * 1000,
        "denoise_latent_fps": 1 / s_per_frame,
    }
