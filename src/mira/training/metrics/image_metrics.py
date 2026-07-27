"""Image- and video-quality metrics: PSNR, LPIPS, SSIM, and DINO Frechet distance.

All metrics accumulate across steps and distributed ranks (via :class:`DistributedMetric` or its own
buffers) so they reduce correctly under ``torchrun`` and run unchanged single-process. The DINO-based
metrics load the DINOv3 backbone through ``torch.hub`` (network / hub cache needed) and the Frechet
distance uses ``scipy`` (imported lazily).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from torchmetrics.functional.image import structural_similarity_index_measure

from mira.codec.dino import (
    DINO_DIM,
    DINO_PATCH_SIZE,
    DINO_REPOSITORY,
    IMAGENET_MEAN,
    IMAGENET_STD,
    resolve_dino_weights,
)

from .distributed_metric import DistributedMetric

logger = logging.getLogger(__name__)


class PSNRMetric(nn.Module):
    """Peak signal-to-noise ratio, accumulated per frame across steps and ranks."""

    def __init__(self, device: str | int | torch.device = "cpu"):
        super().__init__()
        self._metric = DistributedMetric(device=device)

    def reset(self) -> None:
        self._metric.reset()

    def update(self, pred: Tensor, target: Tensor) -> None:
        # pred and target are in [0, 1] and have shape (b, t, c, h, w).
        mse = torch.nn.functional.mse_loss(pred, target, reduction="none").mean(dim=(2, 3, 4))
        psnr = -10 * torch.log10(mse)  # (b, t)
        self._metric.update(psnr)

    def compute_and_reset(self) -> Tensor:
        return self._metric.compute_and_reset()


class DistributedLPIPS(DistributedMetric):
    """Per-frame LPIPS accumulated and averaged across distributed ranks."""

    def __init__(self, device: str | int | torch.device = "cpu"):
        super().__init__(device=device)
        import lpips as lpips_lib  # noqa: PLC0415 -- optional dep, used only here

        self.lpips_fn = lpips_lib.LPIPS(net="alex").to(device)
        self.lpips_fn.requires_grad_(False)

    @torch.no_grad()
    def update(self, gen: Tensor, real: Tensor) -> None:  # type: ignore[override]
        """gen, real: [B, T, C, H, W] in [0, 1]."""
        batch = gen.shape[0]
        # Do it per-video to not OOM on big inputs.
        for b in range(batch):
            scores = self.lpips_fn(gen[b] * 2 - 1, real[b] * 2 - 1)
            super().update(scores)


class DistributedSSIM(DistributedMetric):
    """Per-frame SSIM accumulated and averaged across distributed ranks."""

    @torch.no_grad()
    def update(self, gen: Tensor, real: Tensor) -> None:  # type: ignore[override]
        """gen, real: [B, T, C, H, W] in [0, 1].

        Processes per-video to stay within CUDA 32-bit index limits at high resolution (B*T frames
        concatenated overflows 2^31 elements via torchmetrics' 5-channel cat).
        """
        B, T = gen.shape[:2]
        for b in range(B):
            # structural_similarity_index_measure returns the mean over T frames.
            ssim_mean = structural_similarity_index_measure(gen[b], real[b], data_range=1.0)
            assert isinstance(ssim_mean, Tensor)
            self._sum += ssim_mean.detach().double() * T
            self._n += T


class DinoForMetrics(nn.Module):
    """A frozen DINO backbone exposing last-layer features for the Frechet-distance metric."""

    dino_model: Any  # hub-loaded backbone; typed loosely as its API is dynamic
    mean: Tensor
    std: Tensor

    def __init__(self, model_name: str = "dinov3_vitb16"):
        super().__init__()

        if model_name not in DINO_DIM:
            raise ValueError(f"DINO model {model_name!r} is not supported")
        self.dino_dim = DINO_DIM[model_name]

        logging.getLogger("dinov3").setLevel(logging.WARNING)  # suppress noisy dinov3 logging
        logger.info("Loading DINO metrics model, variant %s", model_name)
        weights = resolve_dino_weights(model_name)
        hub_kwargs: dict[str, Any] = {
            "repo_or_dir": DINO_REPOSITORY[model_name],
            "model": model_name,
            "source": "github",
            "verbose": False,  # Get rid of "Using cache found in ..." message
        }
        if weights is not None:
            self.dino_model = torch.hub.load(**hub_kwargs, weights=str(weights))
        elif model_name.startswith("dinov2_"):
            self.dino_model = torch.hub.load(**hub_kwargs, pretrained=True)
        else:
            raise FileNotFoundError(
                f"DINOv3 metrics weights for {model_name} are unavailable. Set "
                "RS_DINO_WEIGHTS_DIR or select the public dinov2_vitb14 metrics backbone."
            )

        self.register_buffer(
            "mean", torch.tensor(IMAGENET_MEAN, dtype=torch.float)[None, :, None, None], persistent=False
        )
        self.register_buffer(
            "std", torch.tensor(IMAGENET_STD, dtype=torch.float)[None, :, None, None], persistent=False
        )
        self.patch_size = DINO_PATCH_SIZE[model_name]

        self.requires_grad_(False)
        self.eval()

    def image_normalization(self, x: Tensor) -> Tensor:
        return (x - self.mean) / self.std

    @torch.no_grad()
    def dino_forward(self, x: Tensor, max_chunk_size: int | None = None) -> Tensor:
        """Compute DINO features. If max_chunk_size is set, processes frames in chunks to avoid OOM."""
        b, t, _, h, w = x.shape
        x = rearrange(x, "b t c h w -> (b t) c h w")  # x must be in [0, 1]
        x = self.image_normalization(x)
        new_height = self.patch_size * (h // self.patch_size)
        new_width = self.patch_size * (w // self.patch_size)
        if (h, w) != (new_height, new_width):
            x = torch.nn.functional.interpolate(x, (new_height, new_width), mode="bilinear", antialias=True)

        n = x.shape[0]
        if max_chunk_size is not None and n > max_chunk_size:
            features = []
            for i in range(0, n, max_chunk_size):
                feat = self.dino_model.get_intermediate_layers(
                    x[i : i + max_chunk_size], n=1, norm=True, reshape=True
                )[0]
                features.append(feat)
            last_dino_feature = torch.cat(features, dim=0)
        else:
            last_dino_feature = self.dino_model.get_intermediate_layers(x, n=1, norm=True, reshape=True)[0]

        last_dino_feature = rearrange(last_dino_feature, "(b t) c h w -> b t c h w", b=b, t=t)
        return last_dino_feature


def frechet_distance(mean1: Tensor, cov1: Tensor, mean2: Tensor, cov2: Tensor) -> np.floating:
    """Frechet distance between two Gaussians given their means and covariances (returns a scalar)."""
    from scipy.linalg import sqrtm  # noqa: PLC0415 -- optional dep, used only here

    # Convert to numpy because it's not easy to do sqrtm in Torch.
    mean1_np, mean2_np = mean1.cpu().numpy(), mean2.cpu().numpy()
    cov1_np, cov2_np = cov1.cpu().numpy(), cov2.cpu().numpy()

    diff = mean1_np - mean2_np
    cov_prod = sqrtm(cov1_np.dot(cov2_np))
    cov_prod = np.asarray(cov_prod)
    if np.iscomplexobj(cov_prod):
        cov_prod = np.real(cov_prod)

    distance = diff.dot(diff) + np.trace(cov1_np + cov2_np - 2 * cov_prod)
    return distance


class OnlineGaussian(nn.Module):
    """Online estimator of a Gaussian's mean and covariance from sufficient statistics."""

    sum_x: Tensor
    sum_xxT: Tensor
    n: Tensor

    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("sum_x", torch.zeros((dim,), dtype=torch.double))
        self.register_buffer("sum_xxT", torch.zeros((dim, dim), dtype=torch.double))
        self.register_buffer("n", torch.zeros((), dtype=torch.long))

    def reset(self) -> None:
        self.sum_x.zero_()
        self.sum_xxT.zero_()
        self.n.zero_()

    def update(self, x: Tensor) -> None:
        # x must have shape (b, dim).
        x = x.to(dtype=torch.double)
        b = x.shape[0]
        self.n += b
        self.sum_x += x.sum(dim=0)
        self.sum_xxT += x.T @ x

    def all_reduce(self) -> None:
        if not (dist.is_available() and dist.is_initialized()):
            return
        dist.all_reduce(self.sum_x, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.sum_xxT, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.n, op=dist.ReduceOp.SUM)

    def compute(self, unbiased: bool = True, eps: float = 1e-6) -> tuple[Tensor, Tensor]:
        self.all_reduce()
        assert self.n > 1, "Need at least 2 samples to compute statistics."
        n = self.n.to(dtype=torch.double)
        mean = self.sum_x / n

        cov_num = self.sum_xxT - n * torch.outer(mean, mean)
        denom = (n - 1.0) if unbiased else n
        cov = cov_num / denom
        if eps > 0:
            cov = cov + eps * torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
        return mean, cov
