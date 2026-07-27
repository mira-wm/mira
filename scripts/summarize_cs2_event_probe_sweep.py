#!/usr/bin/env python3
"""Aggregate causal MIRA event probes across independent world-model training seeds.

Each ``seed_*`` directory must contain a passing GH200 audit plus
``event_probe/probes/summary.json`` produced by ``run_cs2_frozen_event_probe.sh``. Probe-head seeds
are averaged within each frozen world-model pair; only the world-model training seed is treated as
an independent replication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np

MINIMUM_TRAINING_SEEDS = 3
ARMS = ("single", "synchronized", "shuffled")
PRIMARY_RIGHT_ARM = "shuffled"


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} does not contain a JSON object")
    return payload


def _metric_names(summary: dict[str, Any]) -> list[str]:
    first = summary["results"][0]["test"]
    return sorted(
        name for name in first if name in {"macro_ap", "macro_auc"} or name.endswith(("/ap", "/auc"))
    )


def _metric_seed(label: str, base_seed: int) -> int:
    digest = hashlib.sha256(label.encode()).digest()
    return base_seed ^ int.from_bytes(digest[:8], "little")


def _summary(
    values: list[float],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) < MINIMUM_TRAINING_SEEDS or not np.isfinite(array).all():
        raise ValueError(f"Expected at least {MINIMUM_TRAINING_SEEDS} finite values, got {values}")
    rng = np.random.default_rng(bootstrap_seed)
    draws = array[rng.integers(0, len(array), size=(bootstrap_samples, len(array)))].mean(axis=1)
    return {
        "n_training_seeds": len(values),
        "mean": float(array.mean()),
        "sample_std": float(statistics.stdev(values)),
        "min": float(array.min()),
        "max": float(array.max()),
        "positive_training_seeds": int((array > 0).sum()),
        "training_seed_bootstrap_ci95": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "bootstrap_samples": bootstrap_samples,
    }


def _index_results(summary: dict[str, Any]) -> dict[tuple[int, str], dict[str, float]]:
    indexed: dict[tuple[int, str], dict[str, float]] = {}
    for row in summary["results"]:
        key = (int(row["seed"]), str(row["arm"]))
        if key in indexed:
            raise ValueError(f"Duplicate event-probe result {key}")
        indexed[key] = row["test"]
    return indexed


def _index_comparisons(summary: dict[str, Any]) -> dict[int, dict[str, float]]:
    indexed: dict[int, dict[str, float]] = {}
    for row in summary["paired_comparisons"]:
        if row.get("left") != "synchronized" or row.get("right") != PRIMARY_RIGHT_ARM:
            continue
        seed = int(row["seed"])
        if seed in indexed:
            raise ValueError(f"Duplicate synchronized-minus-{PRIMARY_RIGHT_ARM} comparison seed {seed}")
        indexed[seed] = row["observed"]
    return indexed


def summarize(
    root: Path,
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20_260_726,
) -> dict[str, Any]:
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    seed_roots = sorted(path for path in root.glob("seed_*") if path.is_dir())
    if len(seed_roots) < MINIMUM_TRAINING_SEEDS:
        raise ValueError(
            f"At least {MINIMUM_TRAINING_SEEDS} completed training seeds are required, "
            f"found {len(seed_roots)}"
        )

    audits: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    event_mira_commits: set[str] = set()
    event_release_commits: set[str] = set()
    checkpoint_identities: dict[int, dict[str, str]] = {}
    for seed_root in seed_roots:
        training_seed = int(seed_root.name.removeprefix("seed_"))
        audit = _read(seed_root / "audit.json")
        if audit.get("schema") != "mira-cs2-gh200-sync-control-audit-v3":
            raise ValueError(f"{seed_root}: unexpected child audit schema")
        if audit.get("status") != "pass" or int(audit.get("seed", -1)) != training_seed:
            raise ValueError(f"{seed_root}: child training audit did not pass for its directory seed")

        event_root = seed_root / "event_probe"
        summary = _read(event_root / "probes" / "summary.json")
        if summary.get("schema") != "cs1k-causal-future-event-probe-v1":
            raise ValueError(f"{seed_root}: unexpected event-probe schema")
        if summary.get("feature_mode") != "checkpoint-representations":
            raise ValueError(f"{seed_root}: event probe is not checkpoint-representations")

        checkpoint_identities[training_seed] = {}
        for arm in ("synchronized", "shuffled"):
            event_hash = summary["embedding_sources"][arm]["checkpoint_sha256"]
            audited_hash = audit["checkpoints"][arm]["checkpoint_sha256"]
            if event_hash != audited_hash:
                raise ValueError(f"{seed_root}: {arm} event checkpoint differs from training audit")
            checkpoint_identities[training_seed][arm] = event_hash
        checkpoint_identities[training_seed]["single"] = summary["embedding_sources"]["single"][
            "checkpoint_sha256"
        ]

        event_mira_commit = (
            (event_root / "provenance" / "mira_commit.txt").read_text(encoding="utf-8").strip()
        )
        event_release_commit = (
            (event_root / "provenance" / "release_commit.txt").read_text(encoding="utf-8").strip()
        )
        if summary.get("git_commit") != event_release_commit:
            raise ValueError(f"{seed_root}: event summary release commit differs from provenance")
        event_mira_commits.add(event_mira_commit)
        event_release_commits.add(event_release_commit)
        audits.append(audit)
        summaries.append(summary)

    training_seeds = [int(audit["seed"]) for audit in audits]
    if len(set(training_seeds)) != len(training_seeds):
        raise ValueError(f"Training seeds are not unique: {training_seeds}")
    training_commits = {audit["training_commit"] for audit in audits}
    manifests = {audit["manifest_sha256"] for audit in audits}
    train_steps = {int(audit["train_steps"]) for audit in audits}
    if any(len(values) != 1 for values in (training_commits, manifests, train_steps)):
        raise ValueError("World-model runs drifted in code, data, or fixed update budget")
    if len(event_mira_commits) != 1 or len(event_release_commits) != 1:
        raise ValueError("Event extraction or release code drifted across training seeds")

    first = summaries[0]
    probe_seeds = [int(seed) for seed in first["seeds"]]
    metrics = _metric_names(first)
    labels_sha256 = first["labels_sha256"]
    targets = first["targets"]
    excluded_targets = first["excluded_targets"]
    single_hashes = {summary["embedding_sources"]["single"]["checkpoint_sha256"] for summary in summaries}
    if len(single_hashes) != 1:
        raise ValueError("The contextual single-MIRA checkpoint changed across training seeds")
    checkpoint_pairs = {
        (identity["synchronized"], identity["shuffled"]) for identity in checkpoint_identities.values()
    }
    if len(checkpoint_pairs) != len(training_seeds):
        raise ValueError("A synchronized/cross-round checkpoint pair was reused across training seeds")

    for index, summary in enumerate(summaries):
        contract = (
            summary["feature_mode"],
            [int(seed) for seed in summary["seeds"]],
            summary["targets"],
            summary["excluded_targets"],
            summary["labels_sha256"],
            _metric_names(summary),
            int(summary["bootstrap_samples"]),
            bool(summary["deterministic_algorithms"]),
        )
        expected = (
            "checkpoint-representations",
            probe_seeds,
            targets,
            excluded_targets,
            labels_sha256,
            metrics,
            int(first["bootstrap_samples"]),
            bool(first["deterministic_algorithms"]),
        )
        if contract != expected:
            raise ValueError(f"Event-probe contract drift at training-seed index {index}")

    per_seed_arm_metrics: dict[int, dict[str, dict[str, float]]] = {}
    per_seed_deltas: dict[int, dict[str, float]] = {}
    for training_seed, summary in zip(training_seeds, summaries, strict=True):
        results = _index_results(summary)
        comparisons = _index_comparisons(summary)
        expected_keys = {(seed, arm) for seed in probe_seeds for arm in ARMS}
        if set(results) != expected_keys or set(comparisons) != set(probe_seeds):
            raise ValueError(f"Training seed {training_seed}: incomplete probe-head seed grid")

        per_seed_arm_metrics[training_seed] = {}
        for arm in ARMS:
            per_seed_arm_metrics[training_seed][arm] = {
                metric: statistics.fmean(float(results[(seed, arm)][metric]) for seed in probe_seeds)
                for metric in metrics
            }
        per_seed_deltas[training_seed] = {}
        for metric in metrics:
            head_deltas = [
                float(results[(seed, "synchronized")][metric]) - float(results[(seed, "shuffled")][metric])
                for seed in probe_seeds
            ]
            if metric in {"macro_ap", "macro_auc"}:
                recorded = [float(comparisons[seed][metric]) for seed in probe_seeds]
                if not all(
                    math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)
                    for left, right in zip(head_deltas, recorded, strict=True)
                ):
                    raise ValueError(
                        f"Training seed {training_seed}: paired {metric} does not match arm results"
                    )
            per_seed_deltas[training_seed][metric] = statistics.fmean(head_deltas)

    arms: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        arms[arm] = {}
        for metric in metrics:
            values = [per_seed_arm_metrics[seed][arm][metric] for seed in training_seeds]
            arms[arm][metric] = {
                "training_seed_summary_of_probe_seed_means": _summary(
                    values,
                    bootstrap_samples=bootstrap_samples,
                    bootstrap_seed=_metric_seed(f"arm:{arm}:{metric}", bootstrap_seed),
                ),
                "per_training_seed": dict(zip(map(str, training_seeds), values, strict=True)),
            }

    paired: dict[str, Any] = {}
    for metric in metrics:
        values = [per_seed_deltas[seed][metric] for seed in training_seeds]
        paired[metric] = {
            "definition": "synchronized_minus_cross_round",
            "direction": "higher_is_better",
            "training_seed_summary_of_probe_seed_means": _summary(
                values,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=_metric_seed(f"paired:{metric}", bootstrap_seed),
            ),
            "per_training_seed": dict(zip(map(str, training_seeds), values, strict=True)),
        }

    return {
        "schema": "mira-cs2-causal-event-probe-sweep-v1",
        "status": "pass",
        "independent_unit": "world_model_training_seed",
        "probe_head_seeds_are_not_independent_replications": True,
        "training_seeds": training_seeds,
        "probe_head_seeds": probe_seeds,
        "training_commit": next(iter(training_commits)),
        "event_mira_commit": next(iter(event_mira_commits)),
        "event_release_commit": next(iter(event_release_commits)),
        "manifest_sha256": next(iter(manifests)),
        "labels_sha256": labels_sha256,
        "train_steps": next(iter(train_steps)),
        "targets": targets,
        "excluded_targets": excluded_targets,
        "checkpoint_identities": checkpoint_identities,
        "single_checkpoint_sha256": next(iter(single_hashes)),
        "arms": arms,
        "paired": paired,
        "training_seed_bootstrap_samples": bootstrap_samples,
        "training_seed_bootstrap_seed": bootstrap_seed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_root", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_726)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = summarize(
        args.sweep_root,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    output = args.output or args.sweep_root / "event_probe_sweep_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(output)


if __name__ == "__main__":
    main()
