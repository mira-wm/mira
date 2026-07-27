"""The codec training config composes and instantiates a VideoCodec + CodecLoss via Hydra.

Instantiating the architecture builds the frozen DINOv3 backbone, so this skips gracefully when the
backbone can't be loaded offline.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mira.codec import CodecLoss, CodecLossWeights, VideoCodec

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def _compose(overrides: list[str]):
    pytest.importorskip("hydra")
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(config_name="train_codec", overrides=overrides)


def test_train_codec_config_composes() -> None:
    cfg = _compose(overrides=["run.compile=false"])
    assert cfg.model.architecture._target_ == "mira.codec.VideoCodec"
    assert cfg.model.loss.weights._target_ == "mira.codec.CodecLossWeights"
    # The action vocabulary is interpolated from the dataset (9-key DEFAULT_RL_KEYS).
    assert len(cfg.actions.valid_keys) == 9
    assert cfg.dataset.n_players == 1
    # Loss weights match the release recipe.
    assert cfg.model.loss.weights.loss_mae == 1.0
    assert cfg.model.loss.weights.loss_lpips_perceptual == 1.0
    assert cfg.model.loss.weights.loss_dino_latent_consistency == 1.0
    assert cfg.model.loss.weights.auto_weight is True
    assert cfg.dataloader.prefetch_factor == 2
    assert cfg.dataloader.persistent_workers is False
    assert cfg.dataloader.pin_memory is None


def test_loss_weights_instantiate() -> None:
    from hydra.utils import instantiate

    cfg = _compose(overrides=["run.compile=false"])
    weights = instantiate(cfg.model.loss.weights)
    assert isinstance(weights, CodecLossWeights)
    assert weights.loss_mae == 1.0
    assert weights.auto_weight is True


def test_live_model_config_validates_offline() -> None:
    """The shipped model config validates against the schema without building the DINO backbone.

    Runs offline, so it catches config/schema drift (e.g. a removed field lingering in the YAML)
    that the DINO-gated instantiate test below would otherwise mask behind its skip. The live config
    must validate directly: hydra instantiate does not run the checkpoint-only drop_removed_fields.
    """
    from omegaconf import OmegaConf

    from mira.codec.config import VideoCodecConfig
    from mira.ml.config_loading import strip_hydra_targets

    cfg = _compose(overrides=["run.compile=false"])
    raw = strip_hydra_targets(OmegaConf.to_container(cfg.model.architecture.config, resolve=True))
    config = VideoCodecConfig.model_validate(raw)
    assert config.encoder.latent_dim == config.decoder.latent_dim == 32
    assert config.decoder.patch_size_t == config.encoder.bottleneck.temporal_stride == 2


def test_architecture_instantiates_video_codec() -> None:
    from hydra.utils import instantiate

    cfg = _compose(overrides=["run.compile=false"])
    try:
        model = instantiate(cfg.model.architecture)
    except Exception as exc:  # noqa: BLE001 -- DINOv3 backbone unavailable offline -> skip
        pytest.skip(f"DINOv3 backbone unavailable, skipping: {type(exc).__name__}: {exc}")

    assert isinstance(model, VideoCodec)
    assert model.temporal_downsampling == 2
    # The bound loss is well-formed against the instantiated model.
    loss = CodecLoss(instantiate(cfg.model.loss.weights))
    loss.bind_last_layer(model.decoder.last_layer_weight)
    loss.bind_encoder_dino(model.encoder.rae_dino)
    assert loss.dino_latent_consistency_loss is not None
