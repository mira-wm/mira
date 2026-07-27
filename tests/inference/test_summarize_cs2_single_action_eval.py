from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def _load_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "summarize_cs2_single_action_eval.py"
    spec = importlib.util.spec_from_file_location("single_action_summary_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SUMMARY = _load_module()


def _write_fixture(root: Path, *, changed_sample: bool = False) -> None:
    for window in SUMMARY.WINDOW_MODES:
        for seed in SUMMARY.EXPECTED_SEEDS:
            for mode_index, mode in enumerate(SUMMARY.ACTION_MODES):
                base = root / window / f"seed_{seed}" / mode
                base.parent.mkdir(parents=True, exist_ok=True)
                records = []
                for round_index in range(SUMMARY.EXPECTED_ROUNDS):
                    records.append(
                        {
                            "round_id": f"round_{round_index:03d}",
                            "sample_keys": [
                                (
                                    "changed"
                                    if changed_sample and mode == "zero" and round_index == 0
                                    else f"round_{round_index:03d}_p{pov}"
                                )
                                for pov in range(10)
                            ],
                            "pov_indices": list(range(10)),
                            "action_donor_round_ids": (
                                [f"round_{(round_index + 1) % SUMMARY.EXPECTED_ROUNDS:03d}"]
                                if mode == "round-shifted"
                                else []
                            ),
                            "metrics": {"loss_total": 1.0 + 0.1 * mode_index},
                        }
                    )
                round_path = base.with_suffix(".rounds.jsonl")
                round_path.write_text(
                    "".join(json.dumps(record) + "\n" for record in records),
                    encoding="utf-8",
                )
                round_sha = hashlib.sha256(round_path.read_bytes()).hexdigest()
                payload = {
                    "checkpoint_sha256": "a" * 64,
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
                    "validation": {"total_raw_pov_rows": 690},
                    "results": {"test/loss_total": 1.0 + 0.1 * mode_index},
                    "per_batch_records": {
                        "count": 69,
                        "cluster_unit": "round_id",
                        "sha256": round_sha,
                    },
                }
                base.with_suffix(".json").write_text(
                    json.dumps(payload) + "\n",
                    encoding="utf-8",
                )


def test_summary_uses_round_clusters_and_paired_degradation(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    payload = SUMMARY.summarize(tmp_path)

    shifted = payload["windows"]["midpoint"]["round-shifted"]
    assert shifted["round_mean_paired_degradation_vs_true"]["mean"] == pytest.approx(0.1)
    assert shifted["positive_round_fraction"] == 1.0
    assert shifted["round_cluster_bootstrap_ci95"]["round_clusters"] == 69
    assert payload["contract"]["raw_pov_rows_per_evaluation"] == 690


def test_summary_rejects_changed_receiver_samples(tmp_path: Path) -> None:
    _write_fixture(tmp_path, changed_sample=True)

    with pytest.raises(ValueError, match="sample_keys changed"):
        SUMMARY.summarize(tmp_path)
