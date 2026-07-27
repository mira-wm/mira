"""World-model-training-seed aggregation tests for the causal event probe."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "summarize_cs2_event_probe_sweep.py"
    spec = importlib.util.spec_from_file_location("summarize_cs2_event_probe_sweep_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SUMMARY = _load_module()
TRAINING_COMMIT = "1" * 40
EVENT_MIRA_COMMIT = "2" * 40
RELEASE_COMMIT = "3" * 40
MANIFEST = "4" * 64
LABELS = "5" * 64
SINGLE = "6" * 64
PROBE_SEEDS = [17, 29, 43]


def _write_seed(root: Path, training_seed: int, delta: float) -> None:
    seed_root = root / f"seed_{training_seed}"
    probes = seed_root / "event_probe" / "probes"
    provenance = seed_root / "event_probe" / "provenance"
    probes.mkdir(parents=True)
    provenance.mkdir(parents=True)
    hashes = {
        "single": SINGLE,
        "synchronized": f"{training_seed:064x}",
        "shuffled": f"{training_seed + 100:064x}",
    }
    audit = {
        "schema": "mira-cs2-gh200-sync-control-audit-v3",
        "status": "pass",
        "seed": training_seed,
        "training_commit": TRAINING_COMMIT,
        "train_steps": 10_000,
        "manifest_sha256": MANIFEST,
        "checkpoints": {arm: {"checkpoint_sha256": hashes[arm]} for arm in ("synchronized", "shuffled")},
    }
    (seed_root / "audit.json").write_text(json.dumps(audit), encoding="utf-8")
    (provenance / "mira_commit.txt").write_text(EVENT_MIRA_COMMIT + "\n", encoding="utf-8")
    (provenance / "release_commit.txt").write_text(RELEASE_COMMIT + "\n", encoding="utf-8")

    results = []
    comparisons = []
    for probe_seed in PROBE_SEEDS:
        for arm in SUMMARY.ARMS:
            offset = delta if arm == "synchronized" else 0.0
            results.append(
                {
                    "arm": arm,
                    "seed": probe_seed,
                    "test": {
                        "macro_ap": 0.4 + offset,
                        "macro_auc": 0.6 + offset,
                        "target_FIRE/ap": 0.3 + offset,
                        "target_FIRE/auc": 0.7 + offset,
                    },
                }
            )
        comparisons.extend(
            [
                {
                    "seed": probe_seed,
                    "left": "synchronized",
                    "right": "shuffled",
                    "observed": {"macro_ap": delta, "macro_auc": delta},
                },
                {
                    "seed": probe_seed,
                    "left": "synchronized",
                    "right": "single",
                    "observed": {"macro_ap": delta, "macro_auc": delta},
                },
            ]
        )
    summary = {
        "schema": "cs1k-causal-future-event-probe-v1",
        "feature_mode": "checkpoint-representations",
        "seeds": PROBE_SEEDS,
        "targets": ["target_FIRE"],
        "excluded_targets": {},
        "labels_sha256": LABELS,
        "bootstrap_samples": 10_000,
        "deterministic_algorithms": True,
        "git_commit": RELEASE_COMMIT,
        "results": results,
        "paired_comparisons": comparisons,
        "embedding_sources": {arm: {"checkpoint_sha256": hashes[arm]} for arm in SUMMARY.ARMS},
    }
    (probes / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_summarize_uses_world_model_seed_as_the_independent_unit(tmp_path: Path) -> None:
    for seed, delta in zip((28, 29, 30), (0.01, 0.02, 0.03), strict=True):
        _write_seed(tmp_path, seed, delta)

    result = SUMMARY.summarize(tmp_path, bootstrap_samples=1000)

    primary = result["paired"]["macro_ap"]["training_seed_summary_of_probe_seed_means"]
    assert result["independent_unit"] == "world_model_training_seed"
    assert result["probe_head_seeds_are_not_independent_replications"]
    assert result["training_seeds"] == [28, 29, 30]
    assert result["probe_head_seeds"] == PROBE_SEEDS
    assert primary["n_training_seeds"] == 3
    assert primary["mean"] == pytest.approx(0.02)
    assert primary["positive_training_seeds"] == 3


def test_summarize_rejects_event_checkpoint_identity_drift(tmp_path: Path) -> None:
    for seed, delta in zip((28, 29, 30), (0.01, 0.02, 0.03), strict=True):
        _write_seed(tmp_path, seed, delta)
    summary_path = tmp_path / "seed_30" / "event_probe" / "probes" / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["embedding_sources"]["synchronized"]["checkpoint_sha256"] = "f" * 64
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="event checkpoint differs"):
        SUMMARY.summarize(tmp_path)


def test_summarize_rejects_probe_head_seed_grid_drift(tmp_path: Path) -> None:
    for seed, delta in zip((28, 29, 30), (0.01, 0.02, 0.03), strict=True):
        _write_seed(tmp_path, seed, delta)
    summary_path = tmp_path / "seed_29" / "event_probe" / "probes" / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["seeds"] = [17, 29]
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="contract drift"):
        SUMMARY.summarize(tmp_path)
