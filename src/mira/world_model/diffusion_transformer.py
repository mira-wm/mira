"""The diffusion transformer: the action-conditioned flow-matching network over codec latents.

:class:`DiffusionTransformer` patchifies the latent grid, projects it to the hidden dimension, adds
the per-frame action + diffusion-time conditioning, runs a stack of :class:`AdaSTBlock`s with
temporal RoPE and spatial 2D RoPE, and predicts the flow-matching velocity. It supports a streaming
kv-cache for autoregressive inference and optional clean-past conditioning.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.utils.checkpoint
from einops import rearrange
from torch import Tensor

from mira.ml.attention import SelfAttentionConfig
from mira.ml.init import init_weights
from mira.world_model.config import LatentWorldModelConfig
from mira.world_model.layers.rope import RoPE, SpatialRoPE2D
from mira.world_model.layers.timestep_encoder import DiffusionTimeEmbedding
from mira.world_model.layers.transformer import AdaSTBlock


class DiffusionTransformer(nn.Module):
    """Action-conditioned flow-matching transformer over the codec latent grid."""

    def __init__(
        self,
        config: LatentWorldModelConfig,
        latent_dim: int,
        temporal_downsampling: int,
        spatial_downsampling: int,
    ):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size

        self.n_latent_frames = config.video.timesteps // temporal_downsampling
        self.latent_height, self.latent_width = (
            (config.video.height // (spatial_downsampling * self.patch_size)),
            (config.video.width // (spatial_downsampling * self.patch_size)),
        )
        hidden_dim = config.hidden_dim

        self.register_tokens = None
        self.n_register_tokens = config.n_register_tokens
        if self.n_register_tokens > 0:
            self.register_tokens = nn.Parameter(
                0.02 * torch.randn(1, self.n_register_tokens, 1, 1, hidden_dim)
            )

        self.latent_tokens_proj = nn.Linear(latent_dim * self.patch_size**2, hidden_dim)
        # Only created when past-conditioning is enabled so that checkpoints trained
        # without it still load with a strict state dict.
        self.past_proj = None
        if config.use_clean_past:
            self.past_proj = nn.Linear(latent_dim * self.patch_size**2, hidden_dim)

        head_dim = hidden_dim // config.n_head

        # Spatial positional encoding: resolution-independent axial 2D RoPE (enables multiplayer
        # warm-start from a single-player checkpoint via split-screen).
        self.spatial_rope = SpatialRoPE2D(dim=head_dim)
        # Temporal RoPE. fps is the default (10) — a fixed latent-frame-rate convention, kept stable
        # across codecs so a model can warm-start from another temporal stride, not the codec's
        # actual latent rate (video.fps / temporal_downsampling).
        self.rope = RoPE(dim=head_dim)

        self.diffusion_time_embedding = DiffusionTimeEmbedding(dim=hidden_dim)
        # Embedding of the integration-step size tau_delta, used by the PSD-M loss. Only created
        # when PSD is enabled so that checkpoints trained without it still load with a strict
        # state dict.
        self.diffusion_time_embedding_delta = None
        if config.psd_enabled:
            self.diffusion_time_embedding_delta = DiffusionTimeEmbedding(dim=hidden_dim)

        def has_time_attention(i: int, n_layers: int):
            return (i % config.time_attention_every == 0) or (i == (n_layers - 1))

        self.transformer = nn.ModuleList(
            [
                AdaSTBlock(
                    SelfAttentionConfig(
                        embed_dim=hidden_dim,
                        num_heads=config.n_head,
                        num_kv_heads=config.n_kv_head,
                        gating=config.attention_gating,
                    ),
                    cond_dim=hidden_dim,
                    causal=config.causal,
                    time_attention=has_time_attention(i, config.n_layers),
                    ada_attn_ln=config.ada_attn_ln,
                )
                for i in range(config.n_layers)
            ]
        )
        self.head = nn.Linear(hidden_dim, latent_dim * self.patch_size**2)

        self.apply(init_weights)

    def forward(
        self,
        z_t: Tensor,
        a: Tensor,
        tau: Tensor,
        tau_delta: Tensor | None = None,
        return_kv: bool = False,
        # kv_caches is a list of length n_layers, each containing a tuple of (k, v)
        kv_caches=None,
        # Clean latents of the previous frames; optional, requires config.use_clean_past.
        clean_past: Tensor | None = None,
        # Per-call bool override of checkpointing (None = derive from the config).
        activation_checkpointing: bool | None = None,
    ) -> Tensor | tuple[Tensor, list]:
        if activation_checkpointing is None:
            activation_checkpointing = self.config.activation_checkpointing is True
        # Patchify the latent grid into patch_size x patch_size groups, unpatchify at the end
        if self.patch_size > 1:
            z_t = rearrange(
                z_t,
                "b t (h p_h) (w p_w) c -> b t h w (p_h p_w c)",
                p_h=self.patch_size,
                p_w=self.patch_size,
            )
            if clean_past is not None:
                clean_past = rearrange(
                    clean_past,
                    "b t (h p_h) (w p_w) c -> b t h w (p_h p_w c)",
                    p_h=self.patch_size,
                    p_w=self.patch_size,
                )
        b, t, h, w, _ = z_t.shape

        z_t = self.latent_tokens_proj(z_t)
        if clean_past is not None:
            assert self.past_proj is not None, (
                "clean_past was passed but past-conditioning is disabled (config.use_clean_past=False)."
            )
            z_t = z_t + self.past_proj(clean_past)

        if a.ndim == 3:
            a = rearrange(a, "b t c -> b t 1 1 c").repeat(1, 1, h, w, 1)
        elif a.ndim == 5:
            expected = (b, t, h, w, z_t.shape[-1])
            if tuple(a.shape) != expected:
                raise ValueError(
                    "Spatial action conditioning must match the projected latent grid: "
                    f"expected {expected}, got {tuple(a.shape)}"
                )
        else:
            raise ValueError(
                "Action conditioning must have shape (b,t,d) or (b,t,h,w,d), "
                f"got rank {a.ndim} and shape {tuple(a.shape)}"
            )

        tau_emb = self.diffusion_time_embedding(tau)  # (b, t, 1, 1, c)
        if self.diffusion_time_embedding_delta is not None:
            if tau_delta is None:
                tau_delta = torch.zeros_like(tau)
            tau_emb = tau_emb + self.diffusion_time_embedding_delta(tau_delta)
        tau_emb = tau_emb.repeat(1, 1, h, w, 1)

        sequence = z_t  # (b t h w c)
        cond = a + tau_emb  # (b t h w c)

        if (self.register_tokens is not None) and (kv_caches is None):
            register_tokens = self.register_tokens.repeat(b, 1, h, w, 1)
            sequence = torch.cat([register_tokens, sequence], dim=1)

            cond_register_tokens = torch.zeros_like(register_tokens)
            cond = torch.cat([cond_register_tokens, cond], dim=1)

        rope_len = sequence.shape[1]
        if kv_caches is not None:
            rope_len += kv_caches[0][0].shape[1]
        temporal_rotary_emb = self.rope(rope_len, sequence.device)

        spatial_rotary_emb = self.spatial_rope(h, w, sequence.device)

        new_kv_caches = []
        for i, layer in enumerate(self.transformer):
            if self.training and activation_checkpointing:
                assert not return_kv, "Activation checkpointing with return_kv is not supported."
                sequence, _ = torch.utils.checkpoint.checkpoint(  # type: ignore[misc]
                    layer,
                    sequence,
                    cond,
                    temporal_rotary_emb,
                    spatial_rotary_emb,
                    use_reentrant=False,
                )
            else:
                kv_cache_i = kv_caches[i] if kv_caches is not None else None
                sequence, to_cache = layer(
                    sequence,
                    cond,
                    temporal_rotary_emb=temporal_rotary_emb,
                    spatial_rotary_emb=spatial_rotary_emb,
                    return_kv=return_kv,
                    kv_cache=kv_cache_i,
                )
                if return_kv:
                    new_kv_caches.append(to_cache)

        if (self.register_tokens is not None) and (kv_caches is None):
            sequence = sequence[:, self.n_register_tokens :]

        pred_v = self.head(sequence)

        if self.patch_size > 1:
            pred_v = rearrange(
                pred_v,
                "b t h w (p_h p_w c) -> b t (h p_h) (w p_w) c",
                p_h=self.patch_size,
                p_w=self.patch_size,
            )

        if return_kv:
            return pred_v, new_kv_caches
        return pred_v
