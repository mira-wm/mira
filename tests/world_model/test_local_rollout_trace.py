from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from mira.training.metrics.world_model_metrics import WorldModelMetricsConfig


def _load_train_script():
    path = Path(__file__).resolve().parents[2] / "scripts" / "train_world_model.py"
    spec = importlib.util.spec_from_file_location("train_world_model_local_rollout_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeModel:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.config = SimpleNamespace(video=SimpleNamespace(fps=8))
        self.action_routing = "spatial"
        self.training = True

    def eval(self):
        self.training = False
        return self

    def train(self):
        self.training = True
        return self

    def inference(self, batch, config, progress_bar):
        assert batch == "batch"
        assert progress_bar is False
        return SimpleNamespace()

    def visualize(self, outputs):
        del outputs
        return {"viz_video": torch.zeros(1, 2, 3, 4, 4, dtype=torch.uint8)}


def test_local_rollout_trace_is_atomic_auditable_and_rng_isolated(
    tmp_path: Path,
    monkeypatch,
) -> None:
    script = _load_train_script()
    monkeypatch.setattr(
        script,
        "get_distributed_settings",
        lambda: SimpleNamespace(is_main_process=True),
    )

    def fake_write(path, video, fps):
        assert path.name == "rollout.tmp.mp4"
        assert video.shape == (2, 3, 4, 4)
        assert fps == 8
        path.write_bytes(b"video")

    monkeypatch.setattr(script, "write_video_ffmpeg", fake_write)
    cfg = OmegaConf.create(
        {
            "run": {"output_dir": str(tmp_path), "deterministic": True},
            "validation": {"local_rollout_seed": 37},
            "dataset": {
                "test_split": "val",
                "group_mode": "synchronized",
                "validation_group_mode": "synchronized",
            },
        }
    )
    metadata = [
        SimpleNamespace(
            match_id="match",
            perspective=0,
            frame_indices=[8, 12],
            round_id="round",
            sample_key="sample",
            source_start_frame=8,
        )
    ]
    model = _FakeModel()
    rng_state = random.getstate()

    script.run_local_rollout_trace(
        cfg,
        model,
        [("batch", metadata)],
        WorldModelMetricsConfig(
            n_context_frames=8,
            num_unrolled_frames=4,
            drift_metric_frames=4,
            fdd_slice_frames=2,
        ),
        iter_num=1000,
    )

    trace_dir = tmp_path / "rollout_traces" / "step-000001000"
    assert (trace_dir / "rollout.mp4").read_bytes() == b"video"
    assert not (trace_dir / "rollout.tmp.mp4").exists()
    payload = json.loads((trace_dir / "metadata.json").read_text())
    assert payload["action_routing"] == "spatial"
    assert payload["samples"][0] == {
        "match_id": "match",
        "perspective": 0,
        "round_id": "round",
        "sample_key": "sample",
        "source_start_frame": 8,
    }
    metrics_row = json.loads((tmp_path / "metrics.jsonl").read_text())
    assert metrics_row["kind"] == "rollout_trace"
    assert metrics_row["step"] == 1000
    assert random.getstate() == rng_state
    assert model.training is True
