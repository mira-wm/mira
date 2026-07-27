"""The latent world model: a frozen codec plus an action-conditioned diffusion transformer.

:class:`LatentWorldModel` encodes video into the frozen codec's latent space, trains a flow-matching
diffusion transformer to predict the next latent frame conditioned on actions, and rolls the model
out autoregressively at inference time with a streaming kv-cache. The codec is loaded frozen from a
checkpoint and is never trained here.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from einops import rearrange
from pydantic import BaseModel, ConfigDict
from torch import Tensor
from tqdm import tqdm

from mira.codec.codec_model import VideoCodec
from mira.data.batch import VideoActionBatch
from mira.ml.config_loading import drop_removed_fields, strip_hydra_targets
from mira.training.checkpoints import resolve_checkpoint
from mira.world_model.actions_config import ActionTensors
from mira.world_model.config import LatentWorldModelConfig, WorldModelInferenceConfig
from mira.world_model.diffusion_transformer import DiffusionTransformer
from mira.world_model.layers.action_encoder import ActionEncoder
from mira.world_model.schedule import build_inference_schedule

logger = logging.getLogger(__name__)

# Fields removed from LatentWorldModelConfig that older release checkpoints still write at their
# no-op value. Tolerated only at that value (see drop_removed_fields): a checkpoint that genuinely
# used the feature carries a different value and must keep failing loudly under extra="forbid".
REMOVED_CONFIG_FIELDS: dict[str, object] = {
    "attention_context": None,
    # RoPE is the only positional encoding; these knobs are not read by the model.
    "positional_encoding": "rope",
    "spatial_positional_encoding": "rope",
}


def _config_dict_from_yaml(config_node: object) -> dict:
    """Resolve a saved architecture config to a plain dict, stripping hydra/removed-field cruft."""
    from omegaconf import OmegaConf  # noqa: PLC0415 -- optional dep, used only here

    raw = OmegaConf.to_container(config_node, resolve=True)  # type: ignore[arg-type]
    assert isinstance(raw, dict), "expected a mapping at model.architecture.config"
    return drop_removed_fields(strip_hydra_targets(raw), REMOVED_CONFIG_FIELDS)  # type: ignore[return-value]


class InferenceOutputs(BaseModel):
    """Outputs of an autoregressive rollout (video-only; the codec carries no audio)."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    output_video: Tensor
    preprocessed_batch: VideoActionBatch
    z_t: Tensor


class LatentWorldModel(nn.Module):
    """Frozen codec + diffusion transformer world model over codec latents."""

    def __init__(self, config: LatentWorldModelConfig) -> None:
        super().__init__()
        self.config = config

        # Load pre-trained tokenizer
        if config.codec_checkpoint is not None:
            resolved_checkpoint = resolve_checkpoint(config.codec_checkpoint)
            # Explicit device="cpu" so the codec doesn't leak onto cuda:0 of every DDP rank
            # before the outer .to(local_rank) places the world model.
            self.codec = VideoCodec.load_from_checkpoint(resolved_checkpoint, device="cpu")
            self.codec.requires_grad_(False)
        else:
            raise NotImplementedError(
                "Must provide a codec checkpoint for LatentWorldModel. Joint training not supported yet."
            )

        self.codec.config.encoder.video.height = self.config.video.height
        self.codec.config.encoder.video.width = self.config.video.width
        self.codec.eval()

        if config.latent_mean_std is not None:
            self.latent_mean, self.latent_std = config.latent_mean_std
        elif self.codec.info_from_checkpoint and "latent_mean_std" in self.codec.info_from_checkpoint:
            self.latent_mean, self.latent_std = self.codec.info_from_checkpoint["latent_mean_std"]
        else:
            raise ValueError(
                "Codec latent mean and std not found in config or checkpoint. "
                "Please provide latent_mean_std in the config."
            )

        self.latent_dim = self.codec.latent_dim
        self.temporal_downsampling = self.codec.temporal_downsampling
        self.spatial_downsampling = self.codec.spatial_downsampling
        self.n_context_latents = config.n_context_frames // self.temporal_downsampling
        self.n_context_frames = config.n_context_frames
        assert config.n_context_frames < config.video.timesteps, (
            f"Number of context frames ({config.n_context_frames}) must be smaller "
            f"than the total window size ({config.video.timesteps})."
        )

        self.world_model = DiffusionTransformer(
            config,
            latent_dim=self.latent_dim,
            temporal_downsampling=self.temporal_downsampling,
            spatial_downsampling=self.spatial_downsampling,
        )

        self.action_encoder = ActionEncoder(
            num_key_presses=len(config.actions.valid_keys),
            dim=config.hidden_dim,
            temporal_downsampling=self.action_temporal_downsampling,
            dropout_prob=config.dropout_action_prob,
            max_mouse_movement=config.max_mouse_movement,
            learned_temporal_pool=config.learned_temporal_pool,
            dropout_action_per_player=config.dropout_action_per_player,
            key_field_names=config.actions.valid_keys,
            subset_drop_prob=config.action_subset_drop_prob,
        )
        # Learned "beginning of sequence" latent used as the past of the first frame.
        # Only created with past-conditioning so older checkpoints still load.
        self.bos = None
        if config.use_clean_past:
            self.bos = nn.Parameter(
                0.02
                * torch.randn(self.world_model.latent_height, self.world_model.latent_width, self.latent_dim)
            )

    def forward(self, batch: VideoActionBatch, *args, **kwargs) -> dict[str, Tensor]:
        self.codec.preprocess_batch(batch)

        with torch.no_grad():
            z = self.encode_video(batch.slice_time(0, self.config.video.timesteps, fps=self.config.video.fps))

        off = self.action_temporal_downsampling - 1
        a = self.action_encoder(batch.actions.slice_time(off, self.n_action_steps + off))
        return self.diffusion_loss(z, a)

    def diffusion_loss(self, z: Tensor, a: Tensor) -> dict[str, Tensor]:
        """Diagonal flow-matching loss given encoded latents ``z`` (b, t, h, w, c) and the action
        embedding ``a``. ``a`` is either global ``(b,t,d)`` conditioning or spatial
        ``(b,t,h,w,d)`` conditioning aligned to ``z``. This is the whole training tail;
        MultiWrapperWorldModel reuses it with a tiled multi-player ``z`` and routed ``a`` so the loss
        logic cannot drift between the two forwards. The bos repeat uses ``z.shape[0]`` so the tiled
        ``z`` (whose batch dim differs from ``len(batch)``) is handled correctly."""
        # The past frames are always clean (un-noised) latents.
        shifted_z = None
        if self.config.use_clean_past:
            assert self.bos is not None
            shifted_z = torch.cat([self.bos[None, None].repeat(z.shape[0], 1, 1, 1, 1), z[:, :-1]], dim=1)

        # Diagonal loss: standard flow matching on the full batch.
        z_t, v, tau = self.prepare_training_inputs(z)
        pred_v = self.world_model(z_t, a, tau, clean_past=shifted_z)

        loss_diffusion = torch.nn.functional.mse_loss(pred_v.float(), v.float())
        outputs: dict[str, Tensor] = {
            "loss_total": loss_diffusion,
            "loss_diffusion": loss_diffusion,
        }

        if self.config.psd_weight > 0:
            # Deterministic PSD: compute the loss every step and add `psd_weight * loss_psd` to the
            # total. Exact gradient (no stochastic skipping), at the cost of always running the
            # three extra forward passes.
            loss_psd = self._compute_psdm_loss(z, a, shifted_z)
            outputs["loss_total"] = loss_diffusion + self.config.psd_weight * loss_psd
            outputs["loss_psd"] = loss_psd.detach()
        elif self.config.psd_loss_prob > 0:
            # Stochastic PSD: compute the (unweighted) loss on a fraction `psd_loss_prob` of steps,
            # skipping its three extra forward passes otherwise. The logged value is
            # importance-weighted by 1 / psd_loss_prob (and 0 on skipped steps) so its expectation
            # over steps equals the true PSD loss.
            loss_psd_logged = torch.zeros((), device=z.device)
            # Sample the keep/skip decision once and broadcast it so every rank takes the same
            # branch on a given step; otherwise some ranks run the three extra PSD forward passes
            # while others skip, and the idle ranks stall at the gradient all-reduce.
            do_psd = torch.rand((), device=z.device) < self.config.psd_loss_prob
            if torch.distributed.is_initialized():
                torch.distributed.broadcast(do_psd, src=0)
            if do_psd:
                loss_psd = self._compute_psdm_loss(z, a, shifted_z)
                outputs["loss_total"] = loss_diffusion + loss_psd
                loss_psd_logged = loss_psd.detach() / self.config.psd_loss_prob

            outputs["loss_psd"] = loss_psd_logged

        return outputs

    def _compute_psdm_loss(self, z: Tensor, a: Tensor, shifted_z: Tensor | None) -> Tensor:
        """PSD-M self-distillation loss: the student's long-range velocity (s -> t) regressed onto
        the midpoint two-hop teacher target (s -> u -> t, stop-grad). Runs three extra forward
        passes."""
        z_s, tau_s, tau_t = self.prepare_psdm_inputs(z)
        tau_u = (tau_s + tau_t) / 2  # midpoint gamma = 0.5

        # Student: long-range velocity from s to t.
        pred_v_st = self.world_model(
            z_s,
            a,
            tau_s,
            tau_delta=tau_t - tau_s,
            clean_past=shifted_z,
            activation_checkpointing=self.config.activation_checkpointing in (True, "psd-only"),
        )

        # Teacher: two-hop target through the midpoint u, with stop-gradient.
        with torch.no_grad():
            pred_v_su = self.world_model(z_s, a, tau_s, tau_delta=tau_u - tau_s, clean_past=shifted_z)
            x_su = z_s + (tau_u - tau_s) * pred_v_su
            pred_v_ut = self.world_model(x_su, a, tau_u, tau_delta=tau_t - tau_u, clean_past=shifted_z)
            v_target_st = 0.5 * pred_v_su + 0.5 * pred_v_ut

        return torch.nn.functional.mse_loss(pred_v_st.float(), v_target_st.float())

    def encode_video(self, batch: VideoActionBatch) -> Tensor:
        _, encoder_output = self.codec.encode(batch.video, trim_video=False)
        # The release codec is the deterministic RAE whose posterior mean equals its sample
        # (z_mean == z), so use_codec_posterior_mean reduces to reading z either way.
        z = encoder_output.z
        z = self.normalize_tokens(z)
        z = rearrange(z, "b t c h w -> b t h w c")
        return z

    def prepare_training_inputs(self, z_1: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        z_0 = torch.randn_like(z_1)
        v = z_1 - z_0

        # Sample the flow-matching time tau per latent frame.
        tau = self.sample_training_tau(z_1)
        z_t = tau * z_1 + (1 - tau) * z_0

        return z_t, v, tau

    def sample_training_tau(self, z_1: Tensor) -> Tensor:
        batch_size, n_latents = z_1.shape[:2]
        device = z_1.device
        tau = torch.rand(batch_size, n_latents, device=device)

        return rearrange(tau, "b t -> b t 1 1 1")

    def prepare_psdm_inputs(self, z_1: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        z_0 = torch.randn_like(z_1)
        tau_s, tau_t = self.sample_upper_triangle_tau(z_1)
        z_s = tau_s * z_1 + (1 - tau_s) * z_0
        return z_s, tau_s, tau_t

    def sample_upper_triangle_tau(self, z_1: Tensor) -> tuple[Tensor, Tensor]:
        """Sample (s, t) uniformly from the upper triangle 0 <= s < t <= 1."""
        batch_size, n_latents = z_1.shape[:2]
        device = z_1.device
        a = torch.rand(batch_size, n_latents, device=device)
        b = torch.rand(batch_size, n_latents, device=device)
        s = rearrange(torch.min(a, b), "b t -> b t 1 1 1")
        t = rearrange(torch.max(a, b), "b t -> b t 1 1 1")
        return s, t

    def denoise_streaming(
        self,
        z_t: Tensor,
        a: Tensor,
        streaming_kv_caches=None,
        n_diffusion_steps=10,
        noise_level: float | None = 0.2,
        schedule_type: str = "linear_quadratic",
    ):
        """Denoise the last latent frame, streaming the context through a kv-cache.

        Args:
            z_t: ``(b t h w c)`` with the last latent being complete gaussian noise.
            a: ``(b t c)`` action conditioning.
            streaming_kv_caches: List of per-layer kv caches, or None to initialise from context.
            n_diffusion_steps: Number of integration steps.
            noise_level: Noise level at which the finished frame is stored in the kv cache (an extra
                forward pass re-noises it to ``tau = 1 - noise_level``). None skips that extra pass
                and caches the frame as seen by the final diffusion step instead.
            schedule_type: Sampling-schedule shape passed to ``build_inference_schedule``.

        Returns:
            ``(z_t, streaming_kv_caches)``.
        """
        batch_size, n_latents = z_t.shape[:2]
        n_context_latents = n_latents - 1
        device = z_t.device
        timesteps = build_inference_schedule(n_diffusion_steps, device, schedule_type)
        delta_ts = timesteps[1:] - timesteps[:-1]

        # Initialise streaming_kv_caches
        if (n_context_latents > 0) and streaming_kv_caches is None:
            context_clean_past = None
            if self.config.use_clean_past:
                assert self.bos is not None
                context_clean_past = torch.cat([self.bos.repeat(batch_size, 1, 1, 1, 1), z_t[:, :-2]], dim=1)
            context_tau = torch.ones((batch_size, n_context_latents, 1, 1, 1), device=device, dtype=z_t.dtype)
            z_t_prefix = z_t[:, :-1]

            _, streaming_kv_caches = self.world_model(
                z_t_prefix,
                a[:, :-1],
                context_tau,
                return_kv=True,
                clean_past=context_clean_past,
            )

        # The past frame is always a clean (un-noised) latent.
        clean_past = z_t[:, -2:-1] if self.config.use_clean_past else None

        last_kv_cache = None
        for step_idx, (timestep, delta_t) in enumerate(zip(timesteps[:-1], delta_ts)):
            tau = timestep * torch.ones((batch_size, 1, 1, 1, 1), device=device, dtype=z_t.dtype)
            # PSD-trained models take the integration-step size as input; None means no delta.
            tau_delta = delta_t * torch.ones_like(tau) if self.config.psd_enabled else None
            # With noise_level=None, reuse the final step's forward to also produce
            # the kv cache instead of running a dedicated call after the loop.
            return_kv = noise_level is None and step_idx == len(delta_ts) - 1

            pred_v = self.world_model(
                z_t[:, -1:],
                a[:, -1:],
                tau,
                tau_delta=tau_delta,
                kv_caches=streaming_kv_caches,
                return_kv=return_kv,
                clean_past=clean_past,
            )
            if return_kv:
                pred_v, last_kv_cache = pred_v

            z_t[:, -1:] += delta_t * pred_v

        if noise_level is not None:
            # update streaming kv cache
            # add noise on last frame
            current_tau = (1 - noise_level) * torch.ones(
                (batch_size, 1, 1, 1, 1), device=device, dtype=z_t.dtype
            )
            current_z_t = current_tau * z_t[:, -1:] + (1 - current_tau) * torch.randn_like(z_t[:, -1:])
            current_z_t = current_z_t.to(dtype=z_t.dtype)
            _, last_kv_cache = self.world_model(
                current_z_t,
                a[:, -1:],
                current_tau,
                kv_caches=streaming_kv_caches,
                return_kv=True,
                clean_past=clean_past,
            )

        assert streaming_kv_caches is not None and last_kv_cache is not None
        n_register_tokens = self.config.n_register_tokens
        for i in range(len(streaming_kv_caches)):
            if streaming_kv_caches[i] is not None:
                k_ctx, v_ctx = streaming_kv_caches[i]
                new_k, new_v = last_kv_cache[i]
                streaming_kv_caches[i] = (
                    torch.cat(
                        [k_ctx[:, :n_register_tokens], k_ctx[:, n_register_tokens + 1 :], new_k],
                        dim=1,
                    ).clone(),
                    torch.cat(
                        [v_ctx[:, :n_register_tokens], v_ctx[:, n_register_tokens + 1 :], new_v],
                        dim=1,
                    ).clone(),
                )

        return z_t, streaming_kv_caches

    def set_inference_context(self, n_context_frames: int) -> None:
        """Override how many frames the autoregressive rollout conditions on (inference-only).

        ``n_context_frames`` is a rollout knob rather than a trained parameter, so an eval can shorten
        it to leave room for the rollout inside shorter clips. It must be a multiple of
        ``temporal_downsampling`` (so ``n_context_latents`` is exact) and smaller than the model window
        ``video.timesteps`` (so the rollout window ``n_context_latents + 1`` stays within the trained
        latent window). Updates ``config.n_context_frames``, ``n_context_frames`` and
        ``n_context_latents`` together so the loader, rollout, and metrics stay consistent.
        """
        if n_context_frames % self.temporal_downsampling != 0:
            raise ValueError(
                f"n_context_frames ({n_context_frames}) must be a multiple of "
                f"temporal_downsampling ({self.temporal_downsampling})."
            )
        if n_context_frames < 1 or n_context_frames >= self.config.video.timesteps:
            raise ValueError(
                f"n_context_frames ({n_context_frames}) must be in [1, video.timesteps="
                f"{self.config.video.timesteps})."
            )
        self.config.n_context_frames = n_context_frames
        self.n_context_frames = n_context_frames
        self.n_context_latents = n_context_frames // self.temporal_downsampling

    def inference(
        self,
        batch: VideoActionBatch,
        config: WorldModelInferenceConfig | None = None,
        progress_bar: bool = True,
    ) -> InferenceOutputs:
        if config is None:
            config = WorldModelInferenceConfig()
        self.codec.preprocess_batch(batch)

        batch = batch.to(self.device)
        z = self.encode_video(batch).clone()  # clone because of torch.compile

        z_t = torch.randn_like(z)
        # initial context
        z_t[:, : self.n_context_latents] = z[:, : self.n_context_latents]

        window_size = self.n_context_latents + 1
        start = 0  # In case the loop does not run
        streaming_kv_caches = None

        # denoise autoregressively one frame at a time
        for start in tqdm(
            range(0, z.shape[1] - window_size + 1),
            desc="Diffusion inference",
            disable=not progress_bar,
        ):
            off = self.action_temporal_downsampling - 1
            action_start = start * self.action_temporal_downsampling + off
            action_end = (start + window_size - 1) * self.action_temporal_downsampling + off
            current_a = self.action_encoder(
                batch.actions.slice_time(action_start, action_end),
            ).clone()  # clone because of torch.compile
            z_t[:, start : start + window_size], streaming_kv_caches = self.denoise_streaming(
                z_t[:, start : start + window_size],
                current_a,
                n_diffusion_steps=config.n_diffusion_steps,
                noise_level=config.noise_level,
                streaming_kv_caches=streaming_kv_caches,
                schedule_type=config.schedule_type,
            )

        n_generated_latents = start + window_size
        output_video = self.decode_to_video(z_t[:, :n_generated_latents])
        return InferenceOutputs(
            output_video=output_video,
            preprocessed_batch=batch,
            z_t=z_t,
        )

    def init_streaming_inference(self, batch: VideoActionBatch):
        """Encode the bootstrap batch into latents for streaming inference."""
        batch = batch.to(self.device)
        self.codec.preprocess_batch(batch)
        return self.encode_video(batch).clone()  # clone because of torch.compile

    def streaming_inference_step(
        self,
        z: Tensor,
        actions_history: ActionTensors,
        streaming_kv_cache=None,
        config: WorldModelInferenceConfig | None = None,
    ):
        if config is None:
            config = WorldModelInferenceConfig()

        # Put noise in the last frame - that's what we want to predict, the rest is context
        z_t = torch.cat([z[:, 1:], torch.randn_like(z[:, :1])], dim=1)

        n_action_steps = (z.shape[1] - 1) * self.action_temporal_downsampling
        # Select the actions that produced the frames this step predicts.
        #
        # Convention: action a_t is the control applied at frame t that leads to
        # frame t+1. Each latent z_k bundles `td` frames, so the actions that
        # generated z_k are a_{td*k-1} .. a_{td*k+td-2}. That window starts one
        # action before the latent's first frame, i.e. at offset (atd - 1) versus a
        # naive "one action per latent" slice (atd == td here, since actions and
        # video share an fps):
        #   atd == 1: offset 0, no shift.
        #   atd == 2: offset 1, shift forward by one action.
        # Getting this offset wrong misaligns actions and frames by one frame.
        #
        # Inference caveat (td > 1 only): the window's last action a_{td*k} is a
        # within-chunk action reacting to a frame we have not displayed yet, so a
        # live player cannot really have issued it. We reuse their currently held
        # control for the whole chunk.
        off = self.action_temporal_downsampling - 1
        current_a = self.action_encoder(
            actions_history.slice_time(-n_action_steps - off, -off if off else None).to(self.device)
        )

        return self.denoise_streaming(
            z_t,
            current_a,
            streaming_kv_caches=streaming_kv_cache,
            n_diffusion_steps=config.n_diffusion_steps,
            noise_level=config.noise_level,
            schedule_type=config.schedule_type,
        )

    @torch.no_grad()
    def visualize(self, outputs: InferenceOutputs) -> dict[str, Tensor]:
        """Render a rollout with the keyboard action HUD for W&B logging (video-only).

        Draws the action overlay on the predicted frames, marks the predicted (non-context) frames
        with a coloured border, and stacks the prediction over the ground truth for a side-by-side
        comparison clip.

        Args:
            outputs: The rollout to visualize (its ``output_video`` and ``preprocessed_batch``).

        Returns:
            ``{"viz_video": ..., "pred_video": ...}`` uint8 tensors of shape ``(B, T, C, H, W)``,
            where ``viz_video`` stacks the HUD-annotated prediction over the ground truth vertically.
        """
        # Imported here (not at module load) to keep the model independent of the training package.
        from mira.training.visualization import (  # noqa: PLC0415
            add_prediction_border,
            video_to_uint8,
            visualize_batch,
        )

        pred_video = outputs.output_video
        preprocessed_batch = outputs.preprocessed_batch

        pred_video = visualize_batch(pred_video, preprocessed_batch.actions, self.actions_per_video_frame)
        output_video_viz = add_prediction_border(pred_video, self.n_context_frames)

        viz_video = torch.cat(
            [
                output_video_viz,  # Prediction first so it's visible if the clip is cropped vertically.
                video_to_uint8(preprocessed_batch.video[:, : output_video_viz.shape[1]].to("cpu")),
            ],
            dim=-2,  # -2 = vertically, -1 = horizontally.
        )
        return {"viz_video": viz_video, "pred_video": pred_video}

    def normalize_tokens(self, z: Tensor) -> Tensor:
        return (z - self.latent_mean) / self.latent_std

    def unnormalize_tokens(self, z: Tensor) -> Tensor:
        return self.latent_std * z + self.latent_mean

    def decode_to_video(self, z: Tensor) -> Tensor:
        z = rearrange(z, "b t h w c -> b t c h w")
        z = self.unnormalize_tokens(z)
        output_video = self.codec.decode(z)

        output_video = output_video * 0.5 + 0.5
        return output_video

    def train(self, mode: bool = True):
        super().train(mode)
        self.codec.eval()

        return self

    CONFIG_FILENAME = "world_model_config.yaml"

    @staticmethod
    def _find_config(checkpoint_path: str | Path) -> Path | None:
        for parent in Path(checkpoint_path).parents:
            candidate = parent / LatentWorldModel.CONFIG_FILENAME
            if candidate.exists():
                return candidate

        return None

    def save_checkpoint(self, checkpoint_path: str | Path, extra_data: dict | None = None) -> None:
        if self._find_config(checkpoint_path) is None:
            raise FileNotFoundError(
                f"Could not find '{LatentWorldModel.CONFIG_FILENAME}' in parent directories "
                f"of {checkpoint_path}. Make sure to save the config file alongside the checkpoint."
            )

        checkpoint = {"state_dict": self.state_dict()}

        if extra_data is not None:
            for key in extra_data.keys():
                if key in checkpoint:
                    raise ValueError(f"Key {key} already exists in checkpoint, cannot overwrite")

            checkpoint.update(extra_data)

        torch.save(checkpoint, checkpoint_path)

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str | torch.device | None = None,
        *,
        codec_checkpoint: str | Path | None = None,
        **kwargs,
    ) -> LatentWorldModel:
        from omegaconf import OmegaConf  # noqa: PLC0415 -- optional dep, used only here

        config_path = LatentWorldModel._find_config(checkpoint_path)

        if config_path is None:
            raise FileNotFoundError(
                f"Could not find '{LatentWorldModel.CONFIG_FILENAME}' in parent directories "
                f"of {checkpoint_path}. Make sure to save the config file alongside the checkpoint."
            )

        # Load via OmegaConf (already a dependency) so the config's ${..} interpolations resolve.
        config_raw = OmegaConf.load(config_path)
        config_dict = _config_dict_from_yaml(config_raw.model.architecture.config)
        if codec_checkpoint is not None:
            config_dict["codec_checkpoint"] = str(Path(codec_checkpoint).resolve())
        config = LatentWorldModelConfig.model_validate(config_dict)
        model = cls(config)
        model.to(device)

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        return model

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def action_temporal_downsampling(self):
        """Actions per LATENT frame. Used by action_encoder and n_action_steps."""
        return self.config.actions.target_fps * self.temporal_downsampling // self.config.video.fps

    @property
    def n_action_steps(self) -> int:
        """Raw action steps feeding the action encoder for a full training sequence: one window
        per latent frame after the first, each ``action_temporal_downsampling`` actions wide.
        Exposed so MultiWrapperWorldModel reuses the exact same action/latent alignment instead of
        recomputing it (which is easy to get wrong at temporal_downsampling > 1)."""
        n_latent_frames = self.config.video.timesteps // self.temporal_downsampling
        return (n_latent_frames - 1) * self.action_temporal_downsampling

    @property
    def actions_per_video_frame(self):
        """Actions per VIDEO frame. Used by visualize_batch (one-to-one with video frames)."""
        return self.config.actions.target_fps // self.config.video.fps
