#!/usr/bin/env python
"""Summarize the preregistered, round-paired CS2 single-MIRA action intervention."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

ACTION_MODES = ("true", "round-shifted", "time-shifted", "zero")
WINDOW_MODES = ("midpoint", "first-death")
EXPECTED_SEEDS = (37, 41, 43)
EXPECTED_ROUNDS = 69
EXPECTED_RAW_POV_ROWS = 690
BOOTSTRAP_SEED = 20260725
BOOTSTRAP_REPLICATES = 10_000


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} does not contain a JSON object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError(f"Expected finite values, got {values}")
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _percentile(sorted_values: list[float], probability: float) -> float:
    position = probability * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _cluster_bootstrap(
    round_deltas: dict[str, list[float]],
    *,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, float | int]:
    """Resample rounds; rollout-seed repeats within a round stay in the same cluster."""
    round_ids = sorted(round_deltas)
    if len(round_ids) != EXPECTED_ROUNDS:
        raise ValueError(f"Expected {EXPECTED_ROUNDS} round clusters, got {len(round_ids)}")
    rng = random.Random(seed)
    estimates = []
    for _ in range(replicates):
        sampled = [round_ids[rng.randrange(len(round_ids))] for _ in round_ids]
        estimates.append(statistics.fmean(value for round_id in sampled for value in round_deltas[round_id]))
    estimates.sort()
    return {
        "cluster_unit": "round_id",
        "round_clusters": len(round_ids),
        "replicates": replicates,
        "seed": seed,
        "ci95_low": _percentile(estimates, 0.025),
        "ci95_high": _percentile(estimates, 0.975),
    }


def _read_round_records(path: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != EXPECTED_ROUNDS:
        raise ValueError(f"{path}: expected {EXPECTED_ROUNDS} records, got {len(records)}")
    if not all(isinstance(record, dict) for record in records):
        raise TypeError(f"{path}: every JSONL row must be an object")
    return records


def _validate_result(
    payload: dict[str, Any],
    *,
    path: Path,
    seed: int,
    mode: str,
    window: str,
) -> None:
    expected = {
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
    }
    mismatches = {
        field: {"expected": value, "observed": payload.get(field)}
        for field, value in expected.items()
        if payload.get(field) != value
    }
    if mismatches:
        raise ValueError(f"{path}: result identity mismatch: {mismatches}")
    validation = payload.get("validation", {})
    if validation.get("total_raw_pov_rows") != EXPECTED_RAW_POV_ROWS:
        raise ValueError(f"{path}: expected {EXPECTED_RAW_POV_ROWS} validation POV rows")
    per_batch = payload.get("per_batch_records", {})
    if per_batch.get("count") != EXPECTED_ROUNDS or per_batch.get("cluster_unit") != "round_id":
        raise ValueError(f"{path}: invalid round-record contract: {per_batch}")
    metric = payload.get("results", {}).get("test/loss_total")
    if not isinstance(metric, (int, float)) or not math.isfinite(float(metric)):
        raise ValueError(f"{path}: missing finite test/loss_total")


def summarize(root: Path) -> dict[str, Any]:
    results: dict[str, dict[int, dict[str, dict[str, Any]]]] = {}
    checkpoint_hashes: set[str] = set()
    for window in WINDOW_MODES:
        results[window] = {}
        for seed in EXPECTED_SEEDS:
            mode_records: dict[str, dict[str, Any]] = {}
            for mode in ACTION_MODES:
                base = root / window / f"seed_{seed}" / mode
                result_path = base.with_suffix(".json")
                round_path = base.with_suffix(".rounds.jsonl")
                payload = _read_json(result_path)
                _validate_result(
                    payload,
                    path=result_path,
                    seed=seed,
                    mode=mode,
                    window=window,
                )
                if payload["per_batch_records"]["sha256"] != _sha256(round_path):
                    raise ValueError(f"{round_path}: SHA-256 does not match result sidecar")
                checkpoint_hashes.add(str(payload["checkpoint_sha256"]))
                mode_records[mode] = {
                    "result": payload,
                    "rounds": _read_round_records(round_path),
                }
            results[window][seed] = mode_records

    if len(checkpoint_hashes) != 1:
        raise ValueError(f"Evaluation changed checkpoints: {sorted(checkpoint_hashes)}")

    window_summaries: dict[str, Any] = {}
    for window in WINDOW_MODES:
        true_losses = [
            float(results[window][seed]["true"]["result"]["results"]["test/loss_total"])
            for seed in EXPECTED_SEEDS
        ]
        mode_summaries: dict[str, Any] = {"true": {"loss_total": _summary(true_losses)}}
        for mode in ACTION_MODES[1:]:
            aggregate_losses = [
                float(results[window][seed][mode]["result"]["results"]["test/loss_total"])
                for seed in EXPECTED_SEEDS
            ]
            round_deltas: dict[str, list[float]] = {}
            paired_round_seed_deltas: list[float] = []
            for seed in EXPECTED_SEEDS:
                true_by_round = {
                    record["round_id"]: record for record in results[window][seed]["true"]["rounds"]
                }
                ablated_by_round = {
                    record["round_id"]: record for record in results[window][seed][mode]["rounds"]
                }
                if set(true_by_round) != set(ablated_by_round):
                    raise ValueError(f"{window}/seed_{seed}/{mode}: round IDs changed")
                for round_id, true_record in true_by_round.items():
                    ablated_record = ablated_by_round[round_id]
                    for field in ("sample_keys", "pov_indices"):
                        if true_record[field] != ablated_record[field]:
                            raise ValueError(f"{window}/seed_{seed}/{mode}/{round_id}: {field} changed")
                    if true_record.get("action_donor_round_ids"):
                        raise ValueError(f"{window}/seed_{seed}/true/{round_id}: donor annotated")
                    donor_rounds = ablated_record.get("action_donor_round_ids", [])
                    if mode == "round-shifted" and (len(donor_rounds) != 1 or donor_rounds[0] == round_id):
                        raise ValueError(f"{window}/seed_{seed}/{mode}/{round_id}: invalid donor round")
                    if mode != "round-shifted" and donor_rounds:
                        raise ValueError(f"{window}/seed_{seed}/{mode}/{round_id}: unexpected donor round")
                    true_loss = float(true_record["metrics"]["loss_total"])
                    ablated_loss = float(ablated_record["metrics"]["loss_total"])
                    delta = ablated_loss - true_loss
                    if not math.isfinite(delta):
                        raise ValueError(f"Non-finite paired delta for {round_id}")
                    round_deltas.setdefault(round_id, []).append(delta)
                    paired_round_seed_deltas.append(delta)
            if any(len(values) != len(EXPECTED_SEEDS) for values in round_deltas.values()):
                raise ValueError(f"{window}/{mode}: incomplete rollout-seed repeats by round")
            round_means = [statistics.fmean(values) for values in round_deltas.values()]
            mode_summaries[mode] = {
                "loss_total": _summary(aggregate_losses),
                "aggregate_paired_degradation_vs_true": _summary(
                    [ablated - true for true, ablated in zip(true_losses, aggregate_losses, strict=True)]
                ),
                "round_seed_paired_degradation_vs_true": _summary(paired_round_seed_deltas),
                "round_mean_paired_degradation_vs_true": _summary(round_means),
                "round_cluster_bootstrap_ci95": _cluster_bootstrap(round_deltas),
                "positive_round_fraction": sum(value > 0.0 for value in round_means) / len(round_means),
            }
        window_summaries[window] = mode_summaries

    return {
        "schema": "mira-cs2-single-action-eval-summary-v1",
        "contract": {
            "checkpoint_sha256": next(iter(checkpoint_hashes)),
            "split": "test",
            "map_slug": "dust2",
            "training_seed": 28,
            "evaluation_seeds": list(EXPECTED_SEEDS),
            "action_modes": list(ACTION_MODES),
            "window_modes": list(WINDOW_MODES),
            "rounds": EXPECTED_ROUNDS,
            "raw_pov_rows_per_evaluation": EXPECTED_RAW_POV_ROWS,
            "inference_unit": "round_id",
        },
        "windows": window_summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = summarize(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
