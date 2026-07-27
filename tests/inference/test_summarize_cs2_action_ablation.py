"""Tests for paired CS2 action-ablation summaries."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "summarize_cs2_action_ablation.py"
    spec = importlib.util.spec_from_file_location("summarize_cs2_action_ablation_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SUMMARY = _load_module()


def _payload(arm: str, mode: str, seed: int, loss: float) -> dict:
    is_shared = arm == "shared"
    return {
        "checkpoint_sha256": f"{arm}-checkpoint",
        "split": "test",
        "seed": seed,
        "deterministic": True,
        "map_slug": "dust2",
        "group_mode": "synchronized" if is_shared else "single",
        "training_group_mode": "synchronized" if is_shared else "single",
        "n_players": 10 if is_shared else 1,
        "action_mode": mode,
        "window_mode": "midpoint",
        "validation": {"total_raw_pov_rows": 520},
        "results": {"test/loss_total": loss},
    }


def _write_seed(root: Path, seed: int) -> None:
    losses = {"true": 1.0, "batch-shifted": 1.2, "time-shifted": 1.3, "zero": 1.4}
    for arm in SUMMARY.ARMS:
        arm_dir = root / f"seed_{seed}" / arm
        arm_dir.mkdir(parents=True)
        for mode, loss in losses.items():
            (arm_dir / f"{mode}.json").write_text(json.dumps(_payload(arm, mode, seed, loss)))


def test_summarize_action_ablation_reports_paired_degradation(tmp_path) -> None:
    _write_seed(tmp_path, 37)
    _write_seed(tmp_path, 38)

    result = SUMMARY.summarize(tmp_path)

    degradation = result["arms"]["shared"]["test/loss_total"]["time-shifted"]["paired_degradation_vs_true"]
    assert degradation["mean"] == pytest.approx(0.3)
    assert result["contract"]["seeds"] == [37, 38]
    assert result["contract"]["window_mode"] == "midpoint"


def test_summarize_action_ablation_rejects_checkpoint_drift(tmp_path) -> None:
    _write_seed(tmp_path, 37)
    path = tmp_path / "seed_37" / "single" / "zero.json"
    payload = json.loads(path.read_text())
    payload["checkpoint_sha256"] = "different"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="changed checkpoint_sha256"):
        SUMMARY.summarize(tmp_path)


def test_summarize_action_ablation_rejects_arm_row_mismatch(tmp_path) -> None:
    _write_seed(tmp_path, 37)
    for mode in SUMMARY.ACTION_MODES:
        path = tmp_path / "seed_37" / "shared" / f"{mode}.json"
        payload = json.loads(path.read_text())
        payload["validation"]["total_raw_pov_rows"] = 510
        path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="paired validation raw POV rows differ"):
        SUMMARY.summarize(tmp_path)


def test_summarize_action_ablation_rejects_arm_seed_mismatch(tmp_path) -> None:
    _write_seed(tmp_path, 37)
    for mode in SUMMARY.ACTION_MODES:
        path = tmp_path / "seed_37" / "shared" / f"{mode}.json"
        payload = json.loads(path.read_text())
        payload["seed"] = 99
        path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="paired field 'seed' differs"):
        SUMMARY.summarize(tmp_path)


def test_summarize_supports_synchronized_vs_shuffled_training_control(tmp_path) -> None:
    losses = {"true": 1.0, "batch-shifted": 1.2, "time-shifted": 1.3, "zero": 1.4}
    for seed in (37, 38):
        for arm in ("shuffled", "synchronized"):
            arm_dir = tmp_path / f"seed_{seed}" / arm
            arm_dir.mkdir(parents=True)
            for mode, loss in losses.items():
                payload = _payload("shared", mode, seed, loss)
                payload["checkpoint_sha256"] = f"{arm}-checkpoint"
                payload["training_group_mode"] = arm
                payload["action_routing"] = "spatial"
                (arm_dir / f"{mode}.json").write_text(json.dumps(payload))

    result = SUMMARY.summarize(
        tmp_path,
        arm_a_name="shuffled",
        arm_b_name="synchronized",
        arm_a_eval_group_mode="synchronized",
        arm_b_eval_group_mode="synchronized",
        arm_a_n_players=10,
        arm_b_n_players=10,
        arm_a_training_group_mode="shuffled",
        arm_b_training_group_mode="synchronized",
        expected_action_routing="spatial",
    )

    assert result["contract"]["arm_a"] == "shuffled"
    assert result["contract"]["arm_b"] == "synchronized"
    assert result["contract"]["shuffled_checkpoint_sha256"] == "shuffled-checkpoint"
    degradation = result["arms"]["synchronized"]["test/loss_total"]["zero"][
        "paired_degradation_vs_true"
    ]
    assert degradation["mean"] == pytest.approx(0.4)
