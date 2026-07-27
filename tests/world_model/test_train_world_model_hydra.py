"""The world-model training config composes and instantiates a LatentWorldModel + metrics config.

Instantiating the architecture loads the frozen codec, so the model-build test monkeypatches the
codec loader with the stubbed codec (and shrinks the transformer) to run offline.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mira.training.metrics.world_model_metrics import WorldModelMetricsConfig

from .conftest import StubCodec

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"

# Shrink the 1B transformer to a toy size and point at the stub codec so instantiation is cheap.
_TINY_OVERRIDES = [
    "model.architecture.config.codec_checkpoint=stub.pth",
    "model.architecture.config.latent_mean_std=[0.0,1.0]",
    "model.architecture.config.hidden_dim=64",
    "model.architecture.config.n_head=4",
    "model.architecture.config.n_kv_head=2",
    "model.architecture.config.n_layers=2",
    "model.architecture.config.time_attention_every=1",
    "model.architecture.config.n_context_frames=4",
    "model.architecture.config.video.timesteps=8",
    "model.architecture.config.video.height=64",
    "model.architecture.config.video.width=64",
]


def _compose(overrides: list[str]):
    pytest.importorskip("hydra")
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(config_name="train_world_model", overrides=overrides)


def test_train_world_model_config_composes() -> None:
    cfg = _compose(overrides=[])
    assert cfg.model.architecture._target_ == "mira.world_model.latent_world_model.LatentWorldModel"
    assert cfg.model.architecture.config._target_ == "mira.world_model.config.LatentWorldModelConfig"
    # Action vocabulary is interpolated from the dataset (9-key DEFAULT_RL_KEYS).
    assert len(cfg.actions.valid_keys) == 9
    assert cfg.dataset.n_players == 1
    # 1B architecture sizes come from the package-override default.
    assert cfg.model.architecture.config.hidden_dim == 2048
    assert cfg.model.architecture.config.n_head == 16
    assert cfg.model.architecture.config.n_kv_head == 4
    assert cfg.model.architecture.config.n_layers == 16
    assert cfg.model.architecture.config.time_attention_every == 4
    # run.compile is opt-in (default off for reproducibility).
    assert cfg.run.compile is False
    assert cfg.dataloader.prefetch_factor == 2
    assert cfg.dataloader.persistent_workers is False
    assert cfg.dataloader.pin_memory is None
    assert cfg.validation.local_rollout_every is None
    assert cfg.validation.local_rollout_seed == 37
    assert cfg.world_model_metrics._target_.endswith("WorldModelMetricsConfig")


def test_counterstrike_config_exposes_evaluation_group_override() -> None:
    cfg = _compose(overrides=["dataset=counterstrike1k_dust2"])
    assert cfg.dataset.group_mode == "single"
    assert cfg.dataset.validation_group_mode is None


def test_counterstrike_shared_small_config_uses_spatial_action_routing() -> None:
    cfg = _compose(
        overrides=[
            "dataset=counterstrike1k_dust2",
            "model=multi_wrapper_world_model_cs2_small",
        ]
    )

    assert cfg.model.architecture.config.n_players == 10
    assert cfg.model.architecture.config.action_routing == "spatial"


def test_world_model_metrics_config_instantiates() -> None:
    from hydra.utils import instantiate

    cfg = _compose(overrides=[])
    wm_config = instantiate(cfg.world_model_metrics)
    assert isinstance(wm_config, WorldModelMetricsConfig)
    # Shipped 4 s eval defaults: clip_len = 38 + 20*2 = 78 <= the 80-frame chunk (temporal_downsampling=2).
    assert wm_config.n_context_frames == 38
    assert wm_config.num_unrolled_frames == 20
    assert wm_config.drift_metric_frames == 20
    assert wm_config.fdd_slice_frames == 10
    assert wm_config.dino_model == "dinov3_vitb16"
    assert wm_config.inference.schedule_type == "linear"
    assert wm_config.inference.noise_level == 0.0


def test_architecture_instantiates_latent_world_model(monkeypatch) -> None:
    from hydra.utils import instantiate

    import mira.world_model.latent_world_model as lwm

    monkeypatch.setattr(lwm.VideoCodec, "load_from_checkpoint", staticmethod(lambda *a, **k: StubCodec()))
    cfg = _compose(overrides=_TINY_OVERRIDES)
    model = instantiate(cfg.model.architecture)

    assert isinstance(model, lwm.LatentWorldModel)
    assert model.temporal_downsampling == 2
    # Released timing: video.fps=20 + actions.target_fps=20 + codec td=2 => atd=2, 1 action/frame.
    assert model.config.video.fps == 20
    assert model.config.actions.target_fps == 20
    assert model.action_temporal_downsampling == 2
    assert model.actions_per_video_frame == 1
    assert model.action_encoder.temporal_downsampling == 2
    # Trainable params live in the diffusion transformer; the codec is frozen.
    assert any(p.requires_grad for p in model.world_model.parameters())
    assert all(not p.requires_grad for p in model.codec.parameters())
