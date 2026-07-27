"""Contract and paired-statistics tests for the Dust2 rebuttal evaluator."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_summary_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "summarize_cs2_rebuttal_eval.py"
    spec = importlib.util.spec_from_file_location("summarize_cs2_rebuttal_eval_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SUMMARY = _load_summary_module()


def _payload(arm: str, seed: int, metric: float) -> dict:
    is_shared = arm == "shared"
    batch_size = 1 if is_shared else 10
    n_players = 10 if is_shared else 1
    return {
        "checkpoint_sha256": f"{arm}-checkpoint",
        "split": "test",
        "seed": seed,
        "deterministic": True,
        "map_slug": "dust2",
        "dino_model": "dinov2_vitb14",
        "action_mode": "true",
        "window_mode": "midpoint",
        "group_mode": "synchronized" if is_shared else "single",
        "n_players": n_players,
        "validation": {"total_raw_pov_rows": 520},
        "metrics": {
            "batch_size_groups": batch_size,
            "total_raw_pov_rows": 520,
        },
        "results": {
            "test/loss_total": metric,
            "metrics/psnr": 10.0 - metric,
        },
    }


def _write_pair(root: Path, seed: int, *, single_metric: float, shared_metric: float) -> None:
    seed_dir = root / f"seed_{seed}"
    seed_dir.mkdir()
    (seed_dir / "single.json").write_text(json.dumps(_payload("single", seed, single_metric)))
    (seed_dir / "shared.json").write_text(json.dumps(_payload("shared", seed, shared_metric)))


def test_summarize_reports_paired_improvement(tmp_path) -> None:
    _write_pair(tmp_path, 37, single_metric=2.0, shared_metric=1.0)
    _write_pair(tmp_path, 38, single_metric=4.0, shared_metric=2.0)

    result = SUMMARY.summarize(tmp_path)

    assert result["contract"]["seeds"] == [37, 38]
    assert result["arms"]["single"]["test/loss_total"]["mean"] == 3.0
    assert result["paired"]["test/loss_total"]["shared_minus_single"]["mean"] == -1.5
    assert result["paired"]["test/loss_total"]["shared_improvement"]["mean"] == 1.5
    assert result["paired"]["metrics/psnr"]["shared_improvement"]["mean"] == 1.5


def test_summarize_rejects_unequal_raw_pov_rows(tmp_path) -> None:
    _write_pair(tmp_path, 37, single_metric=2.0, shared_metric=1.0)
    shared_path = tmp_path / "seed_37" / "shared.json"
    shared = json.loads(shared_path.read_text())
    shared["metrics"]["total_raw_pov_rows"] = 510
    shared_path.write_text(json.dumps(shared))

    with pytest.raises(ValueError, match="raw POV rows differ"):
        SUMMARY.summarize(tmp_path)


def test_summarize_supports_synchronized_vs_shuffled_training_control(tmp_path) -> None:
    seed_dir = tmp_path / "seed_37"
    seed_dir.mkdir()
    for name, metric in (("shuffled", 2.0), ("synchronized", 1.0)):
        payload = _payload("shared", 37, metric)
        payload["checkpoint_sha256"] = f"{name}-checkpoint"
        payload["training_group_mode"] = name
        payload["action_routing"] = "spatial"
        (seed_dir / f"{name}.json").write_text(json.dumps(payload))

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

    paired = result["paired"]["test/loss_total"]
    assert paired["synchronized_minus_shuffled"]["mean"] == -1.0
    assert paired["synchronized_improvement"]["mean"] == 1.0
    assert result["contract"]["action_routing"] == "spatial"
