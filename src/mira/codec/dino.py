"""DINO backbone loading for the frozen RAE encoder.

The encoder's frozen feature extractor is a DINOv3 ViT loaded through ``torch.hub`` from the
``facebookresearch/dinov3`` repo. This requires network access (or a populated hub cache) the first
time it runs. Pretrained weights are loaded from a local cache directory when the
``RS_DINO_WEIGHTS_DIR`` environment variable points at one; when no local weights are available and
``require_pretrained=False`` the backbone is built with random weights and restored from a codec
checkpoint instead (the path :meth:`VideoCodec.load_from_checkpoint` takes).

The backbone (:class:`DinoModel`) and the DINO-feature perceptual loss
(:class:`DinoPerceptualLoss`, used for the latent-consistency term) both live here; the loss module
in :mod:`mira.codec.loss` composes them.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch
from einops import rearrange
from torch import Tensor, nn

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# Legacy default used by the standalone DINOv3 evaluation metric.
PATCH_SIZE = 16
DINO_DIM = {
    "dinov2_vitb14": 768,
    "dinov3_vitl16": 1024,
    "dinov3_vitb16": 768,
}
DINO_PATCH_SIZE = {
    "dinov2_vitb14": 14,
    "dinov3_vitl16": 16,
    "dinov3_vitb16": 16,
}
DINO_REPOSITORY = {
    # Pin the hub source so every rank and every training seed executes identical backbone code.
    # The previous floating default branch also made concurrent DDP cache population unsafe.
    "dinov2_vitb14": (
        "facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8"
    ),
    "dinov3_vitl16": "facebookresearch/dinov3",
    "dinov3_vitb16": "facebookresearch/dinov3",
}

# Pretrained DINOv3 weight filenames, as published at
# https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/
DINO_WEIGHT_FILENAMES = {
    "dinov3_vitl16": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    "dinov3_vitb16": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
}


def resolve_dino_weights(dino_model: str) -> Path | None:
    """Return a local DINOv3 weights file for ``dino_model``, or ``None`` if none is configured.

    Set ``RS_DINO_WEIGHTS_DIR`` to a directory holding the pretrained ``.pth`` files (named as in
    :data:`DINO_WEIGHT_FILENAMES`) to load weights from disk; otherwise the backbone is built
    without pretrained weights (and restored from a codec checkpoint, for inference).
    """
    weights_dir = os.environ.get("RS_DINO_WEIGHTS_DIR")
    if not weights_dir:
        return None
    filename = DINO_WEIGHT_FILENAMES.get(dino_model)
    if filename is None:
        return None
    candidate = Path(weights_dir) / filename
    return candidate if candidate.exists() else None


DEFAULT_DINO_LAYERS = {
    "dinov2_vitb14": (2, 5, 8, 11),
    "dinov3_vitl16": (2, 6, 10, 14, 18, 22),
    "dinov3_vitb16": (2, 5, 8, 11),
}


class DinoModel(nn.Module):
    # The hub-loaded (or preloaded) backbone; typed loosely as its API is dynamic.
    dino_model: Any
    mean: Tensor
    std: Tensor

    def __init__(
        self,
        dino_model: str = "dinov3_vitb16",
        preloaded_dino_module: torch.nn.Module | None = None,
        last_layer_only: bool = True,
        layer_indices: tuple[int, ...] | None = None,
        compile: bool = True,
        require_pretrained: bool = True,
    ):
        assert dino_model in DINO_DIM, f"Dino model {dino_model} not supported"
        if last_layer_only and layer_indices is not None:
            raise ValueError("DinoModel: pass either last_layer_only=True OR layer_indices=(...), not both.")
        super().__init__()
        self.dino_model_name = dino_model
        if layer_indices is not None:
            self.layers = layer_indices
        elif last_layer_only:
            self.layers = 1
        else:
            self.layers = DEFAULT_DINO_LAYERS[dino_model]

        self.dino_dim = DINO_DIM[dino_model]
        if preloaded_dino_module:
            self.dino_model = preloaded_dino_module
        else:
            logging.getLogger("dinov3").setLevel(logging.WARNING)  # suppress noisy dinov3 logging
            logger.info(f"Loading DINO model, variant {dino_model}, {compile=}")
            hub_kwargs: dict[str, Any] = {
                "repo_or_dir": DINO_REPOSITORY[dino_model],
                "model": dino_model,
                "source": "github",
                "verbose": False,  # Get rid of "Using cache found in ..." message
            }
            # Load pretrained weights from a local cache dir if RS_DINO_WEIGHTS_DIR provides one;
            # otherwise build without pretrained weights (restored from a codec checkpoint).
            weights_path = resolve_dino_weights(dino_model)
            if weights_path is not None:
                self.dino_model = torch.hub.load(**hub_kwargs, weights=str(weights_path))
            elif require_pretrained and dino_model.startswith("dinov2_"):
                # DINOv2's public hub entry downloads its released checkpoint directly.
                # This is the closest public fallback when gated DINOv3 access is unavailable.
                self.dino_model = torch.hub.load(**hub_kwargs, pretrained=True)
            elif require_pretrained:
                raise FileNotFoundError(
                    f"DINOv3 pretrained weights for {dino_model} not found. Set RS_DINO_WEIGHTS_DIR "
                    f"to a directory containing {DINO_WEIGHT_FILENAMES[dino_model]}. These are "
                    f"required unless the backbone will be restored from a checkpoint. "
                    f"If this is inference, build the codec via VideoCodec.load_from_checkpoint, "
                    f"which sets require_pretrained=False; if this is training, the file must exist."
                )
            else:
                logger.info(
                    "DINO pretrained weights for %s not loaded separately; building with "
                    "pretrained=False (the frozen backbone weights are restored from the "
                    "model checkpoint).",
                    dino_model,
                )
                self.dino_model = torch.hub.load(**hub_kwargs, pretrained=False)
            if compile:
                self.dino_model.get_intermediate_layers = torch.compile(
                    self.dino_model.get_intermediate_layers
                )

        self.register_buffer(
            "mean",
            torch.tensor(IMAGENET_MEAN, dtype=torch.float)[None, :, None, None],
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor(IMAGENET_STD, dtype=torch.float)[None, :, None, None],
            persistent=False,
        )
        self.patch_size = DINO_PATCH_SIZE[dino_model]

        self.requires_grad_(False)
        self.eval()

    def image_normalization(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def dino_forward(self, x: Tensor) -> list[Tensor]:
        b, t, _, h, w = x.shape
        x = rearrange(x, "b t c h w -> (b t) c h w")  # x must be in [0, 1]
        x = self.image_normalization(x)
        new_height = self.patch_size * (h // self.patch_size)
        new_width = self.patch_size * (w // self.patch_size)
        if (h, w) != (new_height, new_width):
            x = torch.nn.functional.interpolate(
                x, (new_height, new_width), mode="bilinear", antialias=True
            )

        dino_features = self.dino_model.get_intermediate_layers(x, n=self.layers, norm=True, reshape=True)  # type: ignore
        dino_features = [
            rearrange(feature, "(b t) c h w -> b t c h w", b=b, t=t) for feature in dino_features
        ]
        return dino_features


class DinoPerceptualLoss(DinoModel):
    """Perceptual loss in DINOv3 feature space, used for the codec's latent-consistency term.

    Compares DINO features of the reconstruction against those of the target, averaged over the
    selected backbone layers. Pass ``preloaded_dino_module=other.dino_model`` to share an
    already-loaded backbone (e.g. the encoder's frozen DINO); ``normalize=True`` selects the
    latent-consistency variant, which L2-normalizes features along the channel dim before the MSE so
    the loss compares feature *directions* rather than magnitudes.
    """

    def __init__(
        self,
        dino_model: str = "dinov3_vitb16",
        preloaded_dino_module: torch.nn.Module | None = None,
        last_layer_only: bool = False,
        layer_indices: tuple[int, ...] | None = None,
        compile: bool = True,
        normalize: bool = False,
    ) -> None:
        super().__init__(
            dino_model=dino_model,
            preloaded_dino_module=preloaded_dino_module,
            last_layer_only=last_layer_only,
            layer_indices=layer_indices,
            compile=compile,
        )
        self.normalize = normalize

    def forward(
        self,
        pred_image: Tensor,
        target_image: Tensor | None = None,
        *,
        target_features: tuple[Tensor, ...] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return ``(loss, last_target_feature)``.

        Provide ``target_features`` to reuse already-computed (detached) target features — the
        latent-consistency term passes the encoder's DINO features here — otherwise the target
        features are computed from ``target_image`` under ``no_grad``.
        """
        pred_features = self.dino_forward(pred_image)
        if target_features is None:
            assert target_image is not None
            with torch.no_grad():
                target_features = tuple(self.dino_forward(target_image))

        layer_terms: list[Tensor] = []
        for p, t in zip(pred_features, target_features):
            if self.normalize:
                p = nn.functional.normalize(p, dim=2, eps=1e-6)
                t = nn.functional.normalize(t, dim=2, eps=1e-6)
            layer_terms.append(nn.functional.mse_loss(p, t))
        return torch.stack(layer_terms).mean(), target_features[-1]
