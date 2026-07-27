from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from mira.inference.loading import load_world_model
from mira.world_model import latent_world_model as latent_module
from mira.world_model.latent_world_model import LatentWorldModel
from mira.world_model.multi_wrapper_world_model import (
    MultiWrapperWorldModel,
    MultiWrapperWorldModelConfig,
)
from tests.world_model.conftest import StubCodec, tiny_config


@pytest.mark.parametrize(
    ("target", "model_cls", "config_key"),
    [
        (
            "mira.world_model.latent_world_model.LatentWorldModel",
            LatentWorldModel,
            "model.architecture.config.codec_checkpoint",
        ),
        (
            "mira.world_model.multi_wrapper_world_model.MultiWrapperWorldModel",
            MultiWrapperWorldModel,
            "model.architecture.config.wm_config.codec_checkpoint",
        ),
    ],
)
def test_load_world_model_relocates_codec_path_in_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    model_cls,
    config_key: str,
) -> None:
    checkpoint = tmp_path / "checkpoint-1" / "checkpoint.pth"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    config = {
        "model": {
            "architecture": {
                "_target_": target,
                "config": (
                    {"wm_config": {"codec_checkpoint": "/old/host/codec/checkpoint.pth"}}
                    if model_cls is MultiWrapperWorldModel
                    else {"codec_checkpoint": "/old/host/codec/checkpoint.pth"}
                ),
            }
        }
    }
    (tmp_path / "world_model_config.yaml").write_text(
        OmegaConf.to_yaml(OmegaConf.create(config)),
        encoding="utf-8",
    )
    relocated_codec = tmp_path / "codec" / "checkpoint.pth"
    relocated_codec.parent.mkdir()
    relocated_codec.write_bytes(b"codec")
    captured = {}
    sentinel = object()

    def fake_load(cls, checkpoint_path, device=None, *, codec_checkpoint=None, **kwargs):
        captured.update(
            checkpoint_path=checkpoint_path,
            device=device,
            codec_checkpoint=codec_checkpoint,
        )
        return sentinel

    monkeypatch.setattr(model_cls, "load_from_checkpoint", classmethod(fake_load))

    model, resolved_config = load_world_model(
        checkpoint,
        device="cpu",
        codec_checkpoint=relocated_codec,
    )

    assert model is sentinel
    assert captured["checkpoint_path"] == checkpoint
    assert captured["codec_checkpoint"] == relocated_codec.resolve()
    assert OmegaConf.select(resolved_config, config_key) == str(relocated_codec.resolve())


@pytest.mark.parametrize("model_cls", [LatentWorldModel, MultiWrapperWorldModel])
def test_checkpoint_loader_uses_relocated_codec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_cls,
) -> None:
    loaded_codec_paths = []

    def load_stub_codec(path, device=None):
        loaded_codec_paths.append(Path(path))
        return StubCodec()

    monkeypatch.setattr(
        latent_module.VideoCodec,
        "load_from_checkpoint",
        staticmethod(load_stub_codec),
    )
    inner_config = tiny_config(codec_checkpoint="/old/host/codec/checkpoint.pth")
    if model_cls is MultiWrapperWorldModel:
        model_config = MultiWrapperWorldModelConfig(n_players=2, wm_config=inner_config)
        source_model = MultiWrapperWorldModel(model_config)
        config_payload = model_config.model_dump(mode="json")
        target = "mira.world_model.multi_wrapper_world_model.MultiWrapperWorldModel"
    else:
        source_model = LatentWorldModel(inner_config)
        config_payload = inner_config.model_dump(mode="json")
        target = "mira.world_model.latent_world_model.LatentWorldModel"

    checkpoint = tmp_path / "checkpoint-1" / "checkpoint.pth"
    checkpoint.parent.mkdir()
    torch.save({"state_dict": source_model.state_dict()}, checkpoint)
    saved_config = {
        "model": {
            "architecture": {
                "_target_": target,
                "config": config_payload,
            }
        }
    }
    (tmp_path / "world_model_config.yaml").write_text(
        OmegaConf.to_yaml(OmegaConf.create(saved_config)),
        encoding="utf-8",
    )
    relocated_codec = tmp_path / "codec" / "checkpoint.pth"
    relocated_codec.parent.mkdir()
    relocated_codec.write_bytes(b"codec")
    loaded_codec_paths.clear()

    reloaded = model_cls.load_from_checkpoint(
        checkpoint,
        device="cpu",
        codec_checkpoint=relocated_codec,
    )

    assert loaded_codec_paths == [relocated_codec.resolve()]
    assert reloaded.config.codec_checkpoint == str(relocated_codec.resolve())
