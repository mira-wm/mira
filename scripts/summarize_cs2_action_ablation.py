"""Summarize paired true-vs-ablated action validation losses for CS2 checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

ARMS = ("single", "shared")
ACTION_MODES = ("true", "batch-shifted", "time-shifted", "zero")


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"{path} does not contain a JSON object")
    return payload


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


def _validate_mode_pair(true: dict[str, Any], ablated: dict[str, Any], source: Path) -> None:
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
        if true.get(field) != ablated.get(field):
            raise ValueError(
                f"{source}: action ablation changed {field}: "
                f"true={true.get(field)!r}, ablated={ablated.get(field)!r}"
            )
    if not true.get("deterministic"):
        raise ValueError(f"{source}: strict deterministic evaluation was not enabled")
    if true["validation"]["total_raw_pov_rows"] != ablated["validation"]["total_raw_pov_rows"]:
        raise ValueError(f"{source}: action ablation changed the validation row count")
    if set(true["results"]) != set(ablated["results"]):
        raise ValueError(f"{source}: action ablation changed result metric keys")


def _validate_arm_pair(
    arm_a: dict[str, Any],
    arm_b: dict[str, Any],
    source: Path,
    *,
    arm_a_name: str,
    arm_b_name: str,
    arm_a_eval_group_mode: str,
    arm_b_eval_group_mode: str,
    arm_a_n_players: int,
    arm_b_n_players: int,
    arm_a_training_group_mode: str,
    arm_b_training_group_mode: str,
    expected_action_routing: str | None,
) -> None:
    for field in (
        "split",
        "seed",
        "deterministic",
        "map_slug",
        "action_mode",
        "window_mode",
        "action_routing",
    ):
        if arm_a.get(field) != arm_b.get(field):
            raise ValueError(
                f"{source}: paired field {field!r} differs: "
                f"{arm_a_name}={arm_a.get(field)!r}, {arm_b_name}={arm_b.get(field)!r}"
            )
    expected_contracts = (
        (
            arm_a_name,
            arm_a,
            arm_a_eval_group_mode,
            arm_a_training_group_mode,
            arm_a_n_players,
        ),
        (
            arm_b_name,
            arm_b,
            arm_b_eval_group_mode,
            arm_b_training_group_mode,
            arm_b_n_players,
        ),
    )
    for arm, payload, group_mode, training_group_mode, n_players in expected_contracts:
        observed = (
            payload.get("group_mode"),
            payload.get("training_group_mode"),
            payload.get("n_players"),
        )
        expected = (group_mode, training_group_mode, n_players)
        if observed != expected:
            raise ValueError(f"{source}: {arm} arm contract must be {expected}, got {observed}")
        if expected_action_routing is not None and payload.get("action_routing") != expected_action_routing:
            raise ValueError(
                f"{source}: {arm} arm action_routing must be {expected_action_routing}, "
                f"got {payload.get('action_routing')}"
            )
    if not arm_a.get("deterministic"):
        raise ValueError(f"{source}: strict deterministic evaluation was not enabled")
    arm_a_rows = arm_a["validation"]["total_raw_pov_rows"]
    arm_b_rows = arm_b["validation"]["total_raw_pov_rows"]
    if arm_a_rows != arm_b_rows:
        raise ValueError(
            f"{source}: paired validation raw POV rows differ: "
            f"{arm_a_name}={arm_a_rows}, {arm_b_name}={arm_b_rows}"
        )
    if set(arm_a["results"]) != set(arm_b["results"]):
        raise ValueError(f"{source}: {arm_a_name}/{arm_b_name} result metric keys differ")


def summarize(
    root: Path,
    *,
    arm_a_name: str = "single",
    arm_b_name: str = "shared",
    arm_a_eval_group_mode: str = "single",
    arm_b_eval_group_mode: str = "synchronized",
    arm_a_n_players: int = 1,
    arm_b_n_players: int = 10,
    arm_a_training_group_mode: str = "single",
    arm_b_training_group_mode: str = "synchronized",
    expected_action_routing: str | None = None,
) -> dict[str, Any]:
    seed_dirs = sorted(path for path in root.glob("seed_*") if path.is_dir())
    if not seed_dirs:
        raise FileNotFoundError(f"No seed_* directories found in {root}")
    if arm_a_name == arm_b_name:
        raise ValueError("Action-ablation arms must have distinct names")

    payloads: dict[str, dict[str, list[dict[str, Any]]]] = {
        arm: {mode: [] for mode in ACTION_MODES} for arm in (arm_a_name, arm_b_name)
    }
    for seed_dir in seed_dirs:
        true_by_arm: dict[str, dict[str, Any]] = {}
        for arm in (arm_a_name, arm_b_name):
            true = _read(seed_dir / arm / "true.json")
            if true.get("action_mode") != "true":
                raise ValueError(f"{seed_dir}/{arm}: true.json is not action_mode=true")
            true_by_arm[arm] = true
            payloads[arm]["true"].append(true)
            for mode in ACTION_MODES[1:]:
                ablated = _read(seed_dir / arm / f"{mode}.json")
                if ablated.get("action_mode") != mode:
                    raise ValueError(
                        f"{seed_dir}/{arm}/{mode}.json has action_mode={ablated.get('action_mode')!r}"
                    )
                _validate_mode_pair(true, ablated, seed_dir / arm)
                payloads[arm][mode].append(ablated)
        _validate_arm_pair(
            true_by_arm[arm_a_name],
            true_by_arm[arm_b_name],
            seed_dir,
            arm_a_name=arm_a_name,
            arm_b_name=arm_b_name,
            arm_a_eval_group_mode=arm_a_eval_group_mode,
            arm_b_eval_group_mode=arm_b_eval_group_mode,
            arm_a_n_players=arm_a_n_players,
            arm_b_n_players=arm_b_n_players,
            arm_a_training_group_mode=arm_a_training_group_mode,
            arm_b_training_group_mode=arm_b_training_group_mode,
            expected_action_routing=expected_action_routing,
        )

    seeds = [int(payload["seed"]) for payload in payloads[arm_a_name]["true"]]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Evaluation seeds are not unique: {seeds}")

    result_arms: dict[str, Any] = {}
    for arm in (arm_a_name, arm_b_name):
        if (
            len(
                {
                    payload["checkpoint_sha256"]
                    for mode_payloads in payloads[arm].values()
                    for payload in mode_payloads
                }
            )
            != 1
        ):
            raise ValueError(f"{arm} changed checkpoints across action modes/seeds")
        metric_names = sorted(payloads[arm]["true"][0]["results"])
        result_arms[arm] = {}
        for metric in metric_names:
            true_values = [float(payload["results"][metric]) for payload in payloads[arm]["true"]]
            modes: dict[str, Any] = {"true": _summary(true_values)}
            for mode in ACTION_MODES[1:]:
                values = [float(payload["results"][metric]) for payload in payloads[arm][mode]]
                modes[mode] = _summary(values)
                modes[mode]["paired_degradation_vs_true"] = _summary(
                    [
                        ablated_value - true_value
                        for true_value, ablated_value in zip(true_values, values, strict=True)
                    ]
                )
            result_arms[arm][metric] = modes

    first = payloads[arm_a_name]["true"][0]
    return {
        "contract": {
            "arm_a": arm_a_name,
            "arm_b": arm_b_name,
            "split": first["split"],
            "map_slug": first["map_slug"],
            "deterministic": first["deterministic"],
            "seeds": seeds,
            "action_modes": list(ACTION_MODES),
            "window_mode": first["window_mode"],
            "action_routing": first.get("action_routing"),
            "validation_raw_pov_rows_per_arm_per_seed": first["validation"]["total_raw_pov_rows"],
            f"{arm_a_name}_checkpoint_sha256": payloads[arm_a_name]["true"][0][
                "checkpoint_sha256"
            ],
            f"{arm_b_name}_checkpoint_sha256": payloads[arm_b_name]["true"][0][
                "checkpoint_sha256"
            ],
        },
        "arms": result_arms,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--arm-a", default="single")
    parser.add_argument("--arm-b", default="shared")
    parser.add_argument("--arm-a-eval-group-mode", default="single")
    parser.add_argument("--arm-b-eval-group-mode", default="synchronized")
    parser.add_argument("--arm-a-n-players", type=int, default=1)
    parser.add_argument("--arm-b-n-players", type=int, default=10)
    parser.add_argument("--arm-a-training-group-mode", default="single")
    parser.add_argument("--arm-b-training-group-mode", default="synchronized")
    parser.add_argument("--expected-action-routing", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or args.root / "summary.json"
    payload = summarize(
        args.root,
        arm_a_name=args.arm_a,
        arm_b_name=args.arm_b,
        arm_a_eval_group_mode=args.arm_a_eval_group_mode,
        arm_b_eval_group_mode=args.arm_b_eval_group_mode,
        arm_a_n_players=args.arm_a_n_players,
        arm_b_n_players=args.arm_b_n_players,
        arm_a_training_group_mode=args.arm_a_training_group_mode,
        arm_b_training_group_mode=args.arm_b_training_group_mode,
        expected_action_routing=args.expected_action_routing,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(output)


if __name__ == "__main__":
    main()
