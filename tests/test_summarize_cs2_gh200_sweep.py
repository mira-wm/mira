"""Training-seed aggregation tests for the audited GH200 control."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "summarize_cs2_gh200_sweep.py"
    spec = importlib.util.spec_from_file_location("summarize_cs2_gh200_sweep_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SUMMARY = _load_module()
COMMIT = "1" * 40
MANIFEST = "2" * 64


def _write_seed(root: Path, seed: int, order: tuple[str, str], delta: float) -> None:
    seed_root = root / f"seed_{seed}"
    primary_root = seed_root / "evaluation" / "synchronized_test_seed_sweep"
    action_root = seed_root / "evaluation" / "synchronized_test_action_loss_seed_sweep"
    death_action_root = seed_root / "evaluation" / "synchronized_test_first_death_action_loss_seed_sweep"
    primary_root.mkdir(parents=True)
    action_root.mkdir(parents=True)
    death_action_root.mkdir(parents=True)
    audit = {
        "schema": "mira-cs2-gh200-sync-control-audit-v3",
        "status": "pass",
        "seed": seed,
        "training_commit": COMMIT,
        "train_steps": 10_000,
        "manifest_sha256": MANIFEST,
        "arm_order": list(order),
    }
    (seed_root / "audit.json").write_text(json.dumps(audit), encoding="utf-8")
    primary = {
        "contract": {
            "arm_a": "shuffled",
            "arm_b": "synchronized",
            "split": "test",
            "seeds": [37, 38, 39, 40, 41],
            "window_mode": "midpoint",
            "shuffled_checkpoint_sha256": f"shuffled-{seed}",
            "synchronized_checkpoint_sha256": f"synchronized-{seed}",
        },
        "paired": {
            "metrics/psnr": {
                "direction": "higher",
                "synchronized_minus_shuffled": {"mean": delta},
            }
        },
    }
    (primary_root / "summary.json").write_text(json.dumps(primary), encoding="utf-8")
    arms = {}
    for arm in ("shuffled", "synchronized"):
        arms[arm] = {
            "test/loss_total": {
                mode: {"paired_degradation_vs_true": {"mean": delta + index}}
                for index, mode in enumerate(("batch-shifted", "time-shifted", "zero"))
            }
        }
    action = {
        "contract": {
            "arm_a": "shuffled",
            "arm_b": "synchronized",
            "split": "test",
            "seeds": [37, 38, 39, 40, 41],
            "window_mode": "midpoint",
            "shuffled_checkpoint_sha256": f"shuffled-{seed}",
            "synchronized_checkpoint_sha256": f"synchronized-{seed}",
        },
        "arms": arms,
    }
    (action_root / "summary.json").write_text(json.dumps(action), encoding="utf-8")
    death_action = json.loads(json.dumps(action))
    death_action["contract"]["window_mode"] = "first-death"
    (death_action_root / "summary.json").write_text(
        json.dumps(death_action),
        encoding="utf-8",
    )


def test_summarize_uses_training_seed_as_independent_unit(tmp_path: Path) -> None:
    _write_seed(tmp_path, 28, ("synchronized", "shuffled"), 1.0)
    _write_seed(tmp_path, 29, ("shuffled", "synchronized"), 2.0)
    _write_seed(tmp_path, 30, ("synchronized", "shuffled"), 3.0)

    result = SUMMARY.summarize(tmp_path)

    assert result["schema"] == "mira-cs2-gh200-sync-control-sweep-v3"
    psnr = result["primary"]["metrics/psnr"]["training_seed_summary_of_eval_seed_means"]
    assert psnr["n_training_seeds"] == 3
    assert psnr["mean"] == 2.0
    assert result["independent_unit"] == "training_seed"
    assert result["train_steps"] == 10_000
    assert result["arm_order_counts"]["synchronized,shuffled"] == 2
    paired_action = result["paired_action_sensitivity"]["test/loss_total"]["zero"][
        "training_seed_summary_of_eval_seed_means"
    ]
    assert paired_action["mean"] == 0.0
    paired_death_action = result["paired_first_death_action_sensitivity"]["test/loss_total"]["zero"][
        "training_seed_summary_of_eval_seed_means"
    ]
    assert paired_death_action["mean"] == 0.0


def test_summarize_rejects_uncounterbalanced_order(tmp_path: Path) -> None:
    for seed in (28, 29, 30):
        _write_seed(tmp_path, seed, ("synchronized", "shuffled"), float(seed))

    with pytest.raises(ValueError, match="not counterbalanced"):
        SUMMARY.summarize(tmp_path)


def test_summarize_rejects_fixed_update_budget_drift(tmp_path: Path) -> None:
    _write_seed(tmp_path, 28, ("synchronized", "shuffled"), 1.0)
    _write_seed(tmp_path, 29, ("shuffled", "synchronized"), 2.0)
    _write_seed(tmp_path, 30, ("synchronized", "shuffled"), 3.0)
    audit_path = tmp_path / "seed_30" / "audit.json"
    audit = json.loads(audit_path.read_text())
    audit["train_steps"] = 20_000
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    with pytest.raises(ValueError, match="fixed update budget"):
        SUMMARY.summarize(tmp_path)
