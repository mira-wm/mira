#!/usr/bin/env python3
"""Aggregate audited Dust2 synchronized-vs-shuffled runs across training seeds."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

MINIMUM_TRAINING_SEEDS = 3


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} does not contain a JSON object")
    return payload


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError(f"Expected finite values, got {values}")
    return {
        "n_training_seeds": len(values),
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _aggregate_action_summaries(
    summaries: list[dict[str, Any]],
    training_seeds: list[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    action: dict[str, Any] = {}
    for arm in ("shuffled", "synchronized"):
        action[arm] = {}
        metric_names = sorted(summaries[0]["arms"][arm])
        for metric in metric_names:
            action[arm][metric] = {}
            for mode in ("batch-shifted", "time-shifted", "zero"):
                degradations = [
                    float(summary["arms"][arm][metric][mode]["paired_degradation_vs_true"]["mean"])
                    for summary in summaries
                ]
                action[arm][metric][mode] = {
                    "training_seed_summary_of_eval_seed_means": _summary(degradations),
                    "per_training_seed": dict(zip(map(str, training_seeds), degradations, strict=True)),
                }

    paired_action_use: dict[str, Any] = {}
    action_metric_names = sorted(summaries[0]["arms"]["shuffled"])
    for metric in action_metric_names:
        paired_action_use[metric] = {}
        for mode in ("batch-shifted", "time-shifted", "zero"):
            differences = []
            for summary in summaries:
                shuffled = float(
                    summary["arms"]["shuffled"][metric][mode]["paired_degradation_vs_true"]["mean"]
                )
                synchronized = float(
                    summary["arms"]["synchronized"][metric][mode]["paired_degradation_vs_true"]["mean"]
                )
                differences.append(synchronized - shuffled)
            paired_action_use[metric][mode] = {
                "interpretation": (
                    "positive means synchronized training increased loss sensitivity "
                    "to this action intervention"
                ),
                "training_seed_summary_of_eval_seed_means": _summary(differences),
                "per_training_seed": dict(zip(map(str, training_seeds), differences, strict=True)),
            }
    return action, paired_action_use


def summarize(root: Path) -> dict[str, Any]:
    seed_roots = sorted(path for path in root.glob("seed_*") if path.is_dir())
    if len(seed_roots) < MINIMUM_TRAINING_SEEDS:
        raise ValueError(
            f"At least {MINIMUM_TRAINING_SEEDS} completed training seeds are required, "
            f"found {len(seed_roots)}"
        )

    audits: list[dict[str, Any]] = []
    primary_summaries: list[dict[str, Any]] = []
    action_summaries: list[dict[str, Any]] = []
    death_action_summaries: list[dict[str, Any]] = []
    for seed_root in seed_roots:
        audit = _read(seed_root / "audit.json")
        if audit.get("schema") != "mira-cs2-gh200-sync-control-audit-v3":
            raise ValueError(f"{seed_root}: unexpected child audit schema")
        if audit.get("status") != "pass":
            raise ValueError(f"{seed_root}: child audit did not pass")
        directory_seed = int(seed_root.name.removeprefix("seed_"))
        if audit.get("seed") != directory_seed:
            raise ValueError(f"{seed_root}: audit seed {audit.get('seed')} does not match directory seed")
        audits.append(audit)
        primary_summaries.append(
            _read(seed_root / "evaluation" / "synchronized_test_seed_sweep" / "summary.json")
        )
        action_summaries.append(
            _read(seed_root / "evaluation" / "synchronized_test_action_loss_seed_sweep" / "summary.json")
        )
        death_action_summaries.append(
            _read(
                seed_root
                / "evaluation"
                / "synchronized_test_first_death_action_loss_seed_sweep"
                / "summary.json"
            )
        )

    training_seeds = [int(audit["seed"]) for audit in audits]
    if len(set(training_seeds)) != len(training_seeds):
        raise ValueError(f"Training seeds are not unique: {training_seeds}")
    commits = {audit["training_commit"] for audit in audits}
    manifests = {audit["manifest_sha256"] for audit in audits}
    train_steps = {audit["train_steps"] for audit in audits}
    if len(commits) != 1 or len(manifests) != 1 or len(train_steps) != 1:
        raise ValueError(
            "Training runs drifted in code, data, or fixed update budget: "
            f"commits={commits}, manifests={manifests}, train_steps={train_steps}"
        )
    checkpoint_pairs = {
        (
            summary["contract"]["shuffled_checkpoint_sha256"],
            summary["contract"]["synchronized_checkpoint_sha256"],
        )
        for summary in primary_summaries
    }
    if len(checkpoint_pairs) != len(training_seeds):
        raise ValueError("A checkpoint pair was reused under more than one training seed")
    arm_orders = [tuple(audit["arm_order"]) for audit in audits]
    first_order = ("shuffled", "synchronized")
    second_order = ("synchronized", "shuffled")
    valid_orders = {first_order, second_order}
    if any(order not in valid_orders for order in arm_orders):
        raise ValueError(f"Invalid arm order found: {arm_orders}")
    order_counts = {order: arm_orders.count(order) for order in valid_orders}
    if min(order_counts.values()) < 1 or abs(order_counts[first_order] - order_counts[second_order]) > 1:
        raise ValueError(f"Arm order is not counterbalanced across training seeds: {order_counts}")

    primary_contract = primary_summaries[0]["contract"]
    action_contract = action_summaries[0]["contract"]
    death_action_contract = death_action_summaries[0]["contract"]
    if action_contract.get("window_mode") != "midpoint":
        raise ValueError("Midpoint action evaluation contract changed window_mode")
    if death_action_contract.get("window_mode") != "first-death":
        raise ValueError("First-death action evaluation contract changed window_mode")
    for index, summary in enumerate(primary_summaries[1:], start=1):
        contract = summary["contract"]
        comparable = {key: value for key, value in contract.items() if not key.endswith("_checkpoint_sha256")}
        expected = {
            key: value for key, value in primary_contract.items() if not key.endswith("_checkpoint_sha256")
        }
        if comparable != expected:
            raise ValueError(f"Primary evaluation contract drift at training seed index {index}")
    for index, summary in enumerate(action_summaries[1:], start=1):
        contract = summary["contract"]
        comparable = {key: value for key, value in contract.items() if not key.endswith("_checkpoint_sha256")}
        expected = {
            key: value for key, value in action_contract.items() if not key.endswith("_checkpoint_sha256")
        }
        if comparable != expected:
            raise ValueError(f"Action evaluation contract drift at training seed index {index}")
    for index, summary in enumerate(death_action_summaries[1:], start=1):
        contract = summary["contract"]
        comparable = {key: value for key, value in contract.items() if not key.endswith("_checkpoint_sha256")}
        expected = {
            key: value
            for key, value in death_action_contract.items()
            if not key.endswith("_checkpoint_sha256")
        }
        if comparable != expected:
            raise ValueError(f"First-death action evaluation contract drift at training seed index {index}")
    for index, (primary_summary, action_summary, death_summary) in enumerate(
        zip(
            primary_summaries,
            action_summaries,
            death_action_summaries,
            strict=True,
        )
    ):
        primary_hashes = (
            primary_summary["contract"]["shuffled_checkpoint_sha256"],
            primary_summary["contract"]["synchronized_checkpoint_sha256"],
        )
        action_hashes = (
            action_summary["contract"]["shuffled_checkpoint_sha256"],
            action_summary["contract"]["synchronized_checkpoint_sha256"],
        )
        death_hashes = (
            death_summary["contract"]["shuffled_checkpoint_sha256"],
            death_summary["contract"]["synchronized_checkpoint_sha256"],
        )
        if not primary_hashes == action_hashes == death_hashes:
            raise ValueError(f"Evaluation checkpoint identity drift at training seed index {index}")

    metric_names = sorted(primary_summaries[0]["paired"])
    primary: dict[str, Any] = {}
    for metric in metric_names:
        deltas = [
            float(summary["paired"][metric]["synchronized_minus_shuffled"]["mean"])
            for summary in primary_summaries
        ]
        primary[metric] = {
            "direction": primary_summaries[0]["paired"][metric]["direction"],
            "training_seed_summary_of_eval_seed_means": _summary(deltas),
            "per_training_seed": dict(zip(map(str, training_seeds), deltas, strict=True)),
        }

    action, paired_action_use = _aggregate_action_summaries(
        action_summaries,
        training_seeds,
    )
    death_action, paired_death_action_use = _aggregate_action_summaries(
        death_action_summaries,
        training_seeds,
    )

    return {
        "schema": "mira-cs2-gh200-sync-control-sweep-v3",
        "status": "pass",
        "independent_unit": "training_seed",
        "training_seeds": training_seeds,
        "training_commit": next(iter(commits)),
        "train_steps": next(iter(train_steps)),
        "manifest_sha256": next(iter(manifests)),
        "arm_order_counts": {
            ",".join(first_order): order_counts[first_order],
            ",".join(second_order): order_counts[second_order],
        },
        "evaluation_seeds_per_training_seed": primary_contract["seeds"],
        "primary": primary,
        "action_sensitivity": action,
        "paired_action_sensitivity": paired_action_use,
        "first_death_action_sensitivity": death_action,
        "paired_first_death_action_sensitivity": paired_death_action_use,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = summarize(args.root)
    output = args.output or args.root / "sweep_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(output)


if __name__ == "__main__":
    main()
