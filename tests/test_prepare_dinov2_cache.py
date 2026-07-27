from __future__ import annotations

from pathlib import Path

import torch

from scripts.prepare_dinov2_cache import MODEL_NAME, prepare_cache


def test_prepare_cache_warms_pinned_source_and_weights(tmp_path: Path, monkeypatch) -> None:
    checkpoint = tmp_path / "hub" / "checkpoints" / "dinov2.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"weights")
    (tmp_path / "hub" / "pinned_source.py").write_text("source\n", encoding="utf-8")
    calls: list[dict] = []

    class FakeModel:
        def get_intermediate_layers(self) -> None:
            pass

    def fake_load(**kwargs):
        calls.append(kwargs)
        return FakeModel()

    monkeypatch.setattr(torch.hub, "load", fake_load)
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path / "hub"))

    payload = prepare_cache(tmp_path)

    assert payload["status"] == "ready"
    assert payload["repository"].endswith(":7764ea0f912e53c92e82eb78a2a1631e92725fc8")
    assert payload["checkpoint_files"][0]["sha256"]
    assert calls == [
        {
            "repo_or_dir": payload["repository"],
            "model": MODEL_NAME,
            "source": "github",
            "verbose": False,
            "pretrained": True,
        }
    ]
