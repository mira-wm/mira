"""Pydantic configs for the RAEv2 temporal-downsampling video codec.

The codec is a frozen DINOv3 backbone with a strided-conv bottleneck (:class:`RAEEncoderConfig`)
feeding a ViT video decoder (:class:`ViTDecoderConfig`).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from mira.ml import ImageConfig


class StridedConvBottleneckConfig(BaseModel):
    """kxk strided convolution bottleneck (RAEv2).

    Encoder side: a ``Conv2d`` (``temporal_stride == 1``) or ``Conv3d`` (``temporal_stride > 1``)
    with ``kernel_size == stride``; decoder side: a matching ``ConvTranspose2d`` lifting the latent
    grid back up. ``noise_tau > 0`` adds train-time Gaussian noise ``sigma ~ U(0, noise_tau)`` per
    frame to the latent (the decoder ignores ``noise_tau``). When ``temporal_stride > 1`` the encoder
    downsamples in time by that factor; pair it with ``decoder.patch_size_t == temporal_stride``.
    """

    model_config = ConfigDict(extra="forbid")

    stride: int = 2
    temporal_stride: int = 1
    noise_tau: float = 0.0


class RAEEncoderConfig(BaseModel):
    """Frozen DINO backbone + a strided-conv bottleneck (RAEv2).

    Aggregates intermediate DINOv3 layers (``mean(features at indices) + features[-1]``) when
    ``aggregation_layers`` is set, then projects to the latent through the strided-conv bottleneck.
    """

    model_config = ConfigDict(extra="forbid")

    latent_dim: int
    # Name of the DINO hub variant used as the frozen backbone (e.g. ``dinov3_vitl16``).
    rae_model: str
    video: ImageConfig

    # None = last-layer-only. list = multi-layer-sum: ``mean(features at indices) + features[-1]``.
    # RAEv2's k=7 default is (11, 13, 15, 17, 19, 21, 23).
    aggregation_layers: list[int] | None = None

    bottleneck: StridedConvBottleneckConfig = Field(default_factory=StridedConvBottleneckConfig)

    compile_dino: bool = True


class ViTDecoderConfig(BaseModel):
    """ViT video decoder lifting the RAEv2 latent back to pixels.

    The ``bottleneck`` (a :class:`StridedConvBottleneckConfig`) selects the latent->ViT-input
    projection: a kxk ``ConvTranspose2d`` lifting the latent to the wider ViT grid (e.g. a /32
    latent up to a /16 ViT grid with ``stride=2``).
    """

    model_config = ConfigDict(extra="forbid")

    video: ImageConfig
    out_channels: int = 3
    latent_dim: int = 32
    patch_size: int = 32
    patch_size_t: int = 1
    is_causal: bool = True
    activation_checkpointing: bool = False
    eps: float = 1e-6

    bottleneck: StridedConvBottleneckConfig = Field(default_factory=StridedConvBottleneckConfig)

    vit_width: int = 1024
    vit_depth: int = 24
    vit_num_heads: int = 16
    vit_num_kv_heads: int | None = None
    mlp_dim_multiplier: int = 4
    layerscale_init: float = 1e-4
    rope_theta_spatial: float = 100.0
    rope_theta_temporal: float = 64.0
    qk_norm: Literal["rmsnorm", "layernorm"] = "rmsnorm"


class VideoCodecConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    encoder: RAEEncoderConfig
    decoder: ViTDecoderConfig
