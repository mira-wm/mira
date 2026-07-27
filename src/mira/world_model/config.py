"""Configuration schemas for the latent world model and its inference.

:class:`LatentWorldModelConfig` is the architecture/training config (validated from the checkpoint's
saved yaml); :class:`WorldModelInferenceConfig` holds the autoregressive-rollout sampling knobs.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from mira.ml.image_config import ImageConfig
from mira.world_model.actions_config import ActionConfig


class LatentWorldModelConfig(BaseModel):
    """Architecture and training configuration of the :class:`LatentWorldModel`.

    The world model is a flow-matching diffusion transformer operating on the frozen codec's latent
    grid. The codec is loaded from ``codec_checkpoint`` and frozen; its latent dimension, temporal
    and spatial downsampling factors are read from it rather than configured here.
    """

    model_config = ConfigDict(extra="forbid")

    actions: ActionConfig
    video: ImageConfig
    codec_checkpoint: str | None = None
    # Optional override of the codec latent mean/std; normally read from the codec checkpoint.
    latent_mean_std: list[float] | None = None

    causal: bool = True
    # Activation checkpointing to bound the memory peak of grad-carrying forward passes.
    # False = off; True = checkpoint every grad-carrying pass; "psd-only" = checkpoint only the PSD
    # student pass (the teacher passes are no_grad and store nothing), leaving the diagonal diffusion
    # pass un-checkpointed at full speed while still bounding the activation peak of PSD steps.
    activation_checkpointing: bool | Literal["psd-only"] = False
    # PSD-M self-distillation: progressive self-distillation with midpoint targets; see "How to
    # build a consistency model" (https://arxiv.org/abs/2505.18825) and the equivalent scheme in
    # "One Step Diffusion via Shortcut Models" (https://arxiv.org/abs/2410.12557).
    # Two mutually-exclusive ways to mix in the PSD loss (both 0 = PSD disabled):
    #   - psd_loss_prob: STOCHASTIC. On a fraction `psd_loss_prob` of steps, add the (unweighted)
    #     PSD loss to the total; the other steps skip its three extra forward passes. Cheaper on
    #     average; the gradient is a stochastic estimate.
    #   - psd_weight: DETERMINISTIC. Compute the PSD loss every step and add `psd_weight * loss_psd`
    #     to the total. Exact gradient, but always pays the three extra forward passes.
    psd_loss_prob: float = 0.0
    psd_weight: float = 0.0
    # If True, condition each denoising step on the clean latents of the previous frames
    # (adds the past_proj and bos parameters). Avoids the train/inference mismatch in diffusion
    # forcing. The default keeps these parameters out of the state dict for strict loading.
    use_clean_past: bool = False

    dropout_action_prob: float = 0.1
    # Symmetric clamp/normalization scale for horizontal/vertical action deltas. Rocket League
    # stores raw mouse dots (default 2048); CounterStrike-1K stores in-game angular degrees.
    max_mouse_movement: float = 2048.0
    # Per-player subset-key action dropout (multiplayer): when on, a dropped (player-)row drops
    # either all keys or just the canonical subset (Q/E/Space/Shift/Ctrl). Default off -> legacy
    # whole-keyboard dropout, so single-player models are unchanged.
    dropout_action_per_player: bool = False
    action_subset_drop_prob: float = 0.5
    learned_temporal_pool: bool = False
    # If True, use the posterior mean from the codec encoder instead of a sampled latent. For the
    # deterministic RAE codec the posterior mean equals the sample, so this reduces to reading z.
    use_codec_posterior_mean: bool = False

    n_register_tokens: int = 0

    patch_size: int = 1
    attention_gating: bool = False
    # If True, apply AdaLN conditioning to the attention sublayers (space/time) as well as the MLP,
    # as in standard DiT. The default applies MLP-only conditioning and keeps the extra parameters
    # out of the state dict for strict loading.
    ada_attn_ln: bool = False

    n_context_frames: int = 39  # only used during inference

    hidden_dim: int = 2048
    n_head: int = 16
    n_kv_head: int | None = None
    n_layers: int = 16
    time_attention_every: int = 1

    @property
    def psd_enabled(self) -> bool:
        """Whether the PSD-M loss is active (via either the stochastic or weighted path)."""
        return self.psd_loss_prob > 0 or self.psd_weight > 0

    @model_validator(mode="after")
    def _check_psd_exclusive(self) -> LatentWorldModelConfig:
        if self.psd_loss_prob > 0 and self.psd_weight > 0:
            raise ValueError("Set at most one of psd_loss_prob (stochastic) and psd_weight (deterministic)")
        return self


class WorldModelInferenceConfig(BaseModel):
    """Sampling knobs for the autoregressive denoising rollout."""

    n_diffusion_steps: int = 10
    # Noise level for the kv-cache update pass; None merges that pass into the
    # last diffusion step (one fewer forward per frame, see denoise_streaming).
    noise_level: float | None = 0.2
    # Sampling-schedule shape: "linear_quadratic" (default) or "linear" (uniform).
    schedule_type: str = "linear_quadratic"
