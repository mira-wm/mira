#!/usr/bin/env python3
"""Apply the frozen go/no-go gate to the Dust2 spatial-action G7e preflight.

This is an engineering gate, not a held-out model-quality endpoint. It uses only the release
validation split to check that the spatial router both preserves player-specific conditioning and
learns a measurable preference for correctly aligned actions before scarce GH200 compute is used.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

EXPECTED_SEEDS = (37, 38, 39)
MIN_ABSOLUTE_DEGRADATION = 0.005
MIN_RELATIVE_DEGRADATION = 0.01
MIN_ROUTING_ATTENUATION = 0.95


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} does not contain a JSON object")
    return payload


def assess(eval_root: Path, routing_diagnostic: Path) -> dict[str, Any]:
    rows: list[dict[str, float | int]] = []
    checkpoint_hashes: set[str] = set()
    for seed in EXPECTED_SEEDS:
        true_path = eval_root / f"seed_{seed}" / "true.json"
        shifted_path = eval_root / f"seed_{seed}" / "batch-shifted.json"
        true = _read(true_path)
        shifted = _read(shifted_path)
        for field in (
            "checkpoint_sha256",
            "split",
            "seed",
            "deterministic",
            "map_slug",
            "group_mode",
            "training_group_mode",
            "n_players",
            "window_mode",
        ):
            if true.get(field) != shifted.get(field):
                raise ValueError(
                    f"{shifted_path}: intervention changed {field}: "
                    f"{true.get(field)!r} != {shifted.get(field)!r}"
                )
        expected_contract = {
            "split": "val",
            "seed": seed,
            "deterministic": True,
            "map_slug": "dust2",
            "group_mode": "synchronized",
            "training_group_mode": "synchronized",
            "n_players": 10,
            "window_mode": "midpoint",
        }
        observed = {key: true.get(key) for key in expected_contract}
        if observed != expected_contract:
            raise ValueError(
                f"{true_path}: expected validation-only preflight contract "
                f"{expected_contract}, got {observed}"
            )
        if true.get("action_mode") != "true" or shifted.get("action_mode") != "batch-shifted":
            raise ValueError(f"{true_path.parent}: action mode labels are invalid")
        for payload, source in ((true, true_path), (shifted, shifted_path)):
            if payload["validation"]["total_raw_pov_rows"] != 540:
                raise ValueError(f"{source}: expected all 54 validation rounds / 540 POV rows")

        true_loss = float(true["results"]["val/loss_total"])
        shifted_loss = float(shifted["results"]["val/loss_total"])
        if not math.isfinite(true_loss) or not math.isfinite(shifted_loss):
            raise ValueError(f"{true_path.parent}: action losses are not finite")
        degradation = shifted_loss - true_loss
        rows.append(
            {
                "seed": seed,
                "true_loss": true_loss,
                "batch_shifted_loss": shifted_loss,
                "absolute_degradation": degradation,
                "relative_degradation": degradation / true_loss,
            }
        )
        checkpoint_hashes.add(str(true["checkpoint_sha256"]))

    if len(checkpoint_hashes) != 1:
        raise ValueError(f"Preflight evaluation changed checkpoint across seeds: {checkpoint_hashes}")

    routing = _read(routing_diagnostic)
    routing_contract = {
        "split": routing.get("split"),
        "map_slug": routing.get("map_slug"),
        "group_mode": routing.get("group_mode"),
        "n_players": routing.get("n_players"),
        "action_routing": routing.get("action_routing"),
        "num_batches": routing.get("num_batches"),
        "raw_pov_rows": routing.get("raw_pov_rows"),
    }
    expected_routing_contract = {
        "split": "val",
        "map_slug": "dust2",
        "group_mode": "synchronized",
        "n_players": 10,
        "action_routing": "spatial",
        "num_batches": 54,
        "raw_pov_rows": 540,
    }
    if routing_contract != expected_routing_contract:
        raise ValueError(
            f"{routing_diagnostic}: expected {expected_routing_contract}, got {routing_contract}"
        )
    if routing["checkpoint_sha256"] not in checkpoint_hashes:
        raise ValueError("Routing diagnostic and action-loss sweep used different checkpoints")

    mean_true = statistics.fmean(float(row["true_loss"]) for row in rows)
    mean_degradation = statistics.fmean(float(row["absolute_degradation"]) for row in rows)
    mean_relative = mean_degradation / mean_true
    routing_attenuation = float(routing["raw_intervention"]["routing_attenuation"]["mean"])
    checks = {
        "all_seed_degradations_positive": all(float(row["absolute_degradation"]) > 0 for row in rows),
        "mean_absolute_degradation_at_least_0.005": (mean_degradation >= MIN_ABSOLUTE_DEGRADATION),
        "mean_relative_degradation_at_least_1pct": (mean_relative >= MIN_RELATIVE_DEGRADATION),
        "routing_attenuation_at_least_0.95": (routing_attenuation >= MIN_ROUTING_ATTENUATION),
    }
    return {
        "schema": "mira-cs2-spatial-routing-preflight-gate-v1",
        "scope": "engineering gate on release validation; not a held-out quality endpoint",
        "checkpoint_sha256": next(iter(checkpoint_hashes)),
        "thresholds": {
            "seeds": list(EXPECTED_SEEDS),
            "validation_rounds": 54,
            "validation_raw_pov_rows": 540,
            "min_mean_absolute_batch_shift_degradation": MIN_ABSOLUTE_DEGRADATION,
            "min_mean_relative_batch_shift_degradation": MIN_RELATIVE_DEGRADATION,
            "require_every_seed_positive": True,
            "min_mean_routing_attenuation": MIN_ROUTING_ATTENUATION,
        },
        "per_seed": rows,
        "summary": {
            "mean_true_loss": mean_true,
            "mean_absolute_degradation": mean_degradation,
            "mean_relative_degradation": mean_relative,
            "mean_routing_attenuation": routing_attenuation,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_root", type=Path)
    parser.add_argument("--routing-diagnostic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = assess(args.eval_root, args.routing_diagnostic)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
