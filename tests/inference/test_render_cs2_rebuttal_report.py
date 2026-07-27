"""Tests for the audit-gated deterministic Dust2 report renderer."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "render_cs2_rebuttal_report.py"
    spec = importlib.util.spec_from_file_location("render_cs2_rebuttal_report_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REPORT = _load_module()


def _stat(mean: float, n: int) -> dict:
    return {"n": n, "mean": mean, "sample_std": 0.1, "min": mean - 0.1, "max": mean + 0.1}


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _action_summary(window_mode: str) -> dict:
    arms = {}
    for arm in ("single", "shared"):
        modes = {"true": _stat(1.0, 5)}
        for index, mode in enumerate(REPORT.ACTION_MODES[1:], start=1):
            modes[mode] = {
                **_stat(1.0 + index / 10, 5),
                "paired_degradation_vs_true": _stat(index / 10, 5),
            }
        arms[arm] = {"test/loss_total": modes}
    return {
        "contract": {
            "split": "test",
            "map_slug": "dust2",
            "deterministic": True,
            "seeds": REPORT.ACTION_SEEDS,
            "action_modes": REPORT.ACTION_MODES,
            "window_mode": window_mode,
            "validation_raw_pov_rows_per_arm_per_seed": 520,
            "single_checkpoint_sha256": "single-sha",
            "shared_checkpoint_sha256": "shared-sha",
        },
        "arms": arms,
    }


def _run_fixture(root: Path) -> None:
    audit = {
        "status": "pass",
        "training_commit": "training-commit",
        "evaluator_commit": "evaluator-commit",
        "dataset_selection_sha256": "selection-sha",
        "stages": {
            stage: {
                "time_limit_step": index * 1000,
                "elapsed_wall_seconds": index * 3600.0,
                "last_logged_processed_frames": index * 160000,
                "checkpoint_sha256": f"{stage}-sha",
            }
            for index, stage in enumerate(("codec", "single", "shared"), start=1)
        },
    }
    primary = {
        "contract": {
            "arm_a": "single",
            "arm_b": "shared",
            "split": "test",
            "map_slug": "dust2",
            "deterministic": True,
            "seeds": REPORT.PRIMARY_SEEDS,
            "validation_raw_pov_rows_per_seed": 520,
            "metrics_raw_pov_rows_per_seed": 520,
            "single_checkpoint_sha256": "single-sha",
            "shared_checkpoint_sha256": "shared-sha",
        },
        "arms": {
            "single": {"test/loss_total": _stat(1.2, 3)},
            "shared": {"test/loss_total": _stat(1.0, 3)},
        },
        "paired": {
            "test/loss_total": {
                "direction": "lower",
                "shared_minus_single": _stat(-0.2, 3),
                "shared_improvement": _stat(0.2, 3),
            }
        },
    }
    _write(root / "audit.json", audit)
    _write(root / REPORT.SUMMARY_PATHS["primary"], primary)
    _write(root / REPORT.SUMMARY_PATHS["midpoint"], _action_summary("midpoint"))
    _write(root / REPORT.SUMMARY_PATHS["first_death"], _action_summary("first-death"))


def test_report_is_deterministic_and_labels_statistical_scope(tmp_path: Path) -> None:
    _run_fixture(tmp_path)

    first = REPORT.render_report(tmp_path)
    second = REPORT.render_report(tmp_path)

    assert first == second
    assert "Audit status | `pass`" in first
    assert "not independent training seeds" in first
    assert "shared − single" in first
    assert "First-death action sensitivity" in first
    assert first.count("`test/loss_total`") == 17


def test_report_rejects_failed_audit(tmp_path: Path) -> None:
    _run_fixture(tmp_path)
    audit_path = tmp_path / "audit.json"
    audit = json.loads(audit_path.read_text())
    audit["status"] = "failed"
    _write(audit_path, audit)

    with pytest.raises(ValueError, match="status=pass"):
        REPORT.render_report(tmp_path)


def test_report_rejects_checkpoint_identity_drift(tmp_path: Path) -> None:
    _run_fixture(tmp_path)
    primary_path = tmp_path / REPORT.SUMMARY_PATHS["primary"]
    primary = json.loads(primary_path.read_text())
    primary["contract"]["shared_checkpoint_sha256"] = "wrong"
    _write(primary_path, primary)

    with pytest.raises(ValueError, match="audited shared checkpoint"):
        REPORT.render_report(tmp_path)


def test_report_can_bind_an_explicit_strict_audit(tmp_path: Path) -> None:
    _run_fixture(tmp_path)
    strict_path = tmp_path / "audit_strict.json"
    strict = json.loads((tmp_path / "audit.json").read_text())
    strict["evaluator_commit"] = "strictly-checked-evaluator"
    _write(strict_path, strict)
    (tmp_path / "audit.json").unlink()

    report = REPORT.render_report(tmp_path, audit_path=strict_path)

    assert "strictly-checked-evaluator" in report
    assert REPORT._sha256(strict_path) in report
