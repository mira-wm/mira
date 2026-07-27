from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_script():
    path = Path(__file__).resolve().parents[2] / "scripts" / "assess_cs2_spatial_routing_preflight.py"
    spec = importlib.util.spec_from_file_location(
        "assess_cs2_spatial_routing_preflight",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _eval_payload(seed: int, mode: str, loss: float) -> dict:
    return {
        "checkpoint_sha256": "checkpoint",
        "split": "val",
        "seed": seed,
        "deterministic": True,
        "map_slug": "dust2",
        "group_mode": "synchronized",
        "training_group_mode": "synchronized",
        "n_players": 10,
        "window_mode": "midpoint",
        "action_mode": mode,
        "validation": {"total_raw_pov_rows": 540},
        "results": {"val/loss_total": loss},
    }


def test_assess_passes_only_when_every_frozen_gate_passes(tmp_path: Path) -> None:
    script = _load_script()
    for seed, degradation in zip(script.EXPECTED_SEEDS, (0.007, 0.006, 0.0055), strict=True):
        _write(tmp_path / f"seed_{seed}" / "true.json", _eval_payload(seed, "true", 0.4))
        _write(
            tmp_path / f"seed_{seed}" / "batch-shifted.json",
            _eval_payload(seed, "batch-shifted", 0.4 + degradation),
        )
    routing = tmp_path / "routing.json"
    _write(
        routing,
        {
            "checkpoint_sha256": "checkpoint",
            "split": "val",
            "map_slug": "dust2",
            "group_mode": "synchronized",
            "n_players": 10,
            "action_routing": "spatial",
            "num_batches": 54,
            "raw_pov_rows": 540,
            "raw_intervention": {"routing_attenuation": {"mean": 0.999}},
        },
    )

    result = script.assess(tmp_path, routing)

    assert result["passed"] is True
    assert result["summary"]["mean_absolute_degradation"] == pytest.approx((0.007 + 0.006 + 0.0055) / 3)


def test_assess_records_failed_gate_without_relaxing_threshold(tmp_path: Path) -> None:
    script = _load_script()
    for seed in script.EXPECTED_SEEDS:
        _write(tmp_path / f"seed_{seed}" / "true.json", _eval_payload(seed, "true", 0.4))
        _write(
            tmp_path / f"seed_{seed}" / "batch-shifted.json",
            _eval_payload(seed, "batch-shifted", 0.401),
        )
    routing = tmp_path / "routing.json"
    _write(
        routing,
        {
            "checkpoint_sha256": "checkpoint",
            "split": "val",
            "map_slug": "dust2",
            "group_mode": "synchronized",
            "n_players": 10,
            "action_routing": "spatial",
            "num_batches": 54,
            "raw_pov_rows": 540,
            "raw_intervention": {"routing_attenuation": {"mean": 0.999}},
        },
    )

    result = script.assess(tmp_path, routing)

    assert result["passed"] is False
    assert result["checks"]["mean_absolute_degradation_at_least_0.005"] is False
