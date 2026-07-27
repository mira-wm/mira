"""RAEv2 encoder: a frozen DINO backbone, layer aggregation, and a strided-conv bottleneck."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from einops import rearrange
from torch import Tensor, nn

from mira.codec.config import RAEEncoderConfig
from mira.codec.dino import DINO_DIM, DinoModel
from mira.ml import init_weights


@dataclass
class RAEEncoderOutputs:
    z: Tensor
    dino_features: tuple[Tensor, ...] | None = None


class RAEEncoder(nn.Module):
    """Frozen DINO backbone -> aggregation -> bottleneck -> latent."""

    def __init__(self, config: RAEEncoderConfig, require_dino_weights: bool = True) -> None:
        """
        require_dino_weights: Can be set to False at inference when we'll load from a pretrained
            checkpoint. That way the deployment environment doesn't need a separate path to valid
            DINO weights.
        """
        super().__init__()
        self.config = config
        dino_dim = DINO_DIM[config.rae_model]

        if config.bottleneck.temporal_stride > 1:
            self.rae_projection = nn.Conv3d(
                dino_dim,
                config.latent_dim,
                kernel_size=(
                    config.bottleneck.temporal_stride,
                    config.bottleneck.stride,
                    config.bottleneck.stride,
                ),
                stride=(
                    config.bottleneck.temporal_stride,
                    config.bottleneck.stride,
                    config.bottleneck.stride,
                ),
                bias=True,
            )
        else:
            self.rae_projection = nn.Conv2d(
                dino_dim,
                config.latent_dim,
                kernel_size=config.bottleneck.stride,
                stride=config.bottleneck.stride,
                bias=True,
            )

        # Initialise the bottleneck projection before building the frozen DINO backbone, so the
        # backbone's pretrained weights are not overwritten by `init_weights`.
        self.apply(init_weights)

        layer_indices = tuple(config.aggregation_layers) if config.aggregation_layers else None
        self.rae_dino = DinoModel(
            config.rae_model,
            last_layer_only=(layer_indices is None),
            layer_indices=layer_indices,
            compile=config.compile_dino,
            require_pretrained=require_dino_weights,
        )

    def get_downsampling_factors(self) -> tuple[int, int]:
        return (
            self.config.bottleneck.temporal_stride,
            self.rae_dino.patch_size * self.config.bottleneck.stride,
        )

    def forward(self, video: Tensor) -> RAEEncoderOutputs:
        # VideoCodec normalizes to [-1, 1]; DinoModel expects [0, 1].
        video = (video + 1) / 2

        with torch.no_grad():
            features = self.rae_dino.dino_forward(video)  # list of (B, T, dino_dim, H, W)

        if self.config.aggregation_layers:
            stacked = torch.stack(features, dim=0)
            agg = stacked.mean(dim=0) + features[-1]
        else:
            agg = features[-1]

        if isinstance(self.rae_projection, nn.Conv3d):
            x = rearrange(agg, "b t c h w -> b c t h w")
            z = self.rae_projection(x)
            z = rearrange(z, "b c t h w -> b t c h w")
        elif isinstance(self.rae_projection, nn.Conv2d):
            b, t = agg.shape[:2]
            x = rearrange(agg, "b t c h w -> (b t) c h w")
            z = self.rae_projection(x)
            z = rearrange(z, "(b t) c h w -> b t c h w", b=b, t=t)
        else:
            raise TypeError(f"Unexpected bottleneck projection: {type(self.rae_projection)}")

        # RAEv2-style noise regulariser (train mode, sigma per frame ~ U(0, noise_tau));
        # like the VAE reparam noise but with a configurable scale.
        if self.training and self.config.bottleneck.noise_tau > 0:
            sigma = (
                torch.rand(z.shape[0], z.shape[1], 1, 1, 1, device=z.device, dtype=z.dtype)
                * self.config.bottleneck.noise_tau
            )
            z = z + sigma * torch.randn_like(z)

        return RAEEncoderOutputs(z=z, dino_features=tuple(features))
