from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "audit_cs2_single_confirmatory_endpoint.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_cs2_single_confirmatory_endpoint_under_test", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


AUDIT = _load_module()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _make_native_evaluation(root: Path, checkpoint_sha256: str) -> None:
    _write_json(
        root / "summary.json",
        {
            "contract": {
                "checkpoint_sha256": checkpoint_sha256,
                "split": "test",
                "map_slug": "dust2",
                "training_seed": 28,
                "evaluation_seeds": list(AUDIT.EXPECTED_EVAL_SEEDS),
                "action_modes": list(AUDIT.EXPECTED_ACTION_MODES),
                "window_modes": list(AUDIT.EXPECTED_WINDOW_MODES),
                "rounds": 69,
                "raw_pov_rows_per_evaluation": 690,
                "inference_unit": "round_id",
            },
            "windows": {"midpoint": {"delta": 0.1}, "first-death": {"delta": 0.2}},
        },
    )
    for window in AUDIT.EXPECTED_WINDOW_MODES:
        for seed in AUDIT.EXPECTED_EVAL_SEEDS:
            for mode in AUDIT.EXPECTED_ACTION_MODES:
                result_path = root / window / f"seed_{seed}" / f"{mode}.json"
                sidecar_path = result_path.with_suffix(".rounds.jsonl")
                sidecar_path.parent.mkdir(parents=True, exist_ok=True)
                records = []
                for index in range(69):
                    round_id = f"round-{index:03d}"
                    donors = [f"round-{(index + 1) % 69:03d}"] if mode == "round-shifted" else []
                    records.append(
                        {
                            "round_id": round_id,
                            "sample_keys": [f"{round_id}-pov-{pov}" for pov in range(10)],
                            "pov_indices": list(range(10)),
                            "action_donor_round_ids": donors,
                            "metrics": {"loss_total": 0.1 + index / 1000},
                        }
                    )
                sidecar_path.write_text(
                    "".join(json.dumps(row) + "\n" for row in records),
                    encoding="utf-8",
                )
                _write_json(
                    result_path,
                    {
                        "dataset_backend": "counterstrike1k",
                        "split": "test",
                        "seed": seed,
                        "deterministic": True,
                        "map_slug": "dust2",
                        "group_mode": "single",
                        "training_group_mode": "single",
                        "n_players": 1,
                        "action_mode": mode,
                        "window_mode": window,
                        "checkpoint_sha256": checkpoint_sha256,
                        "validation": {
                            "total_raw_pov_rows": 690,
                            "num_batches": 69,
                        },
                        "per_batch_records": {
                            "count": 69,
                            "cluster_unit": "round_id",
                            "sha256": _sha256(sidecar_path),
                        },
                    },
                )


def test_verify_native_action_evaluation_accepts_complete_grid(tmp_path: Path) -> None:
    checkpoint_sha256 = "a" * 64
    _make_native_evaluation(tmp_path, checkpoint_sha256)
    audit = AUDIT.verify_native_action_evaluation(tmp_path, checkpoint_sha256)
    assert audit["cell_count"] == 24
    assert audit["round_clusters_per_cell"] == 69


def test_verify_native_action_evaluation_rejects_sidecar_drift(tmp_path: Path) -> None:
    checkpoint_sha256 = "a" * 64
    _make_native_evaluation(tmp_path, checkpoint_sha256)
    sidecar = tmp_path / "midpoint" / "seed_37" / "true.rounds.jsonl"
    sidecar.write_text(sidecar.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="drifted round sidecar"):
        AUDIT.verify_native_action_evaluation(tmp_path, checkpoint_sha256)
