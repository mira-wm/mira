"""Validate and summarize paired CounterStrike-1K evaluation runs.

The input directory must contain ``seed_<N>/single.json`` and ``seed_<N>/shared.json`` files
written by :mod:`scripts.eval_world_model_offline`. The summarizer first enforces the experimental
contract (same split, seed, public metric backbone, and number of raw POV rows), then reports
per-arm mean/sample-standard-deviation and paired deltas. Arm names and expected train/eval
grouping can be overridden for the synchronized-vs-shuffled GH200 control.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

LOWER_IS_BETTER_PREFIXES = (
    "val/loss",
    "test/loss",
    "metrics/dino_",
    "metrics/latent_",
    "metrics/frechet_",
    "metrics/lpips",
    "metrics/denoise_ms_",
)
HIGHER_IS_BETTER_PREFIXES = (
    "metrics/psnr",
    "metrics/ssim",
    "metrics/denoise_latent_fps",
    "metrics/denoise_pov_latent_fps",
)


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"{path} does not contain a JSON object")
    return payload


def _direction(metric: str) -> str:
    if metric.startswith(LOWER_IS_BETTER_PREFIXES):
        return "lower"
    if metric.startswith(HIGHER_IS_BETTER_PREFIXES):
        return "higher"
    return "descriptive"


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError(f"Expected finite metric values, got {values}")
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _validate_pair(
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
    arm_a_training_group_mode: str | None,
    arm_b_training_group_mode: str | None,
    expected_action_routing: str | None,
) -> None:
    for field in (
        "split",
        "seed",
        "deterministic",
        "map_slug",
        "dino_model",
        "action_mode",
        "window_mode",
    ):
        if arm_a.get(field) != arm_b.get(field):
            raise ValueError(
                f"{source}: paired field {field!r} differs: "
                f"{arm_a_name}={arm_a.get(field)!r}, {arm_b_name}={arm_b.get(field)!r}"
            )
    arm_contracts = (
        (
            arm_a_name,
            arm_a,
            arm_a_eval_group_mode,
            arm_a_n_players,
            arm_a_training_group_mode,
        ),
        (
            arm_b_name,
            arm_b,
            arm_b_eval_group_mode,
            arm_b_n_players,
            arm_b_training_group_mode,
        ),
    )
    for name, payload, eval_group_mode, n_players, training_group_mode in arm_contracts:
        if payload.get("group_mode") != eval_group_mode or payload.get("n_players") != n_players:
            raise ValueError(
                f"{source}: {name} arm must use group_mode={eval_group_mode} and n_players={n_players}"
            )
        if training_group_mode is not None and payload.get("training_group_mode") != training_group_mode:
            raise ValueError(
                f"{source}: {name} arm training_group_mode must be {training_group_mode}, "
                f"got {payload.get('training_group_mode')}"
            )
        if expected_action_routing is not None and payload.get("action_routing") != expected_action_routing:
            raise ValueError(
                f"{source}: {name} arm action_routing must be {expected_action_routing}, "
                f"got {payload.get('action_routing')}"
            )
    if not arm_a.get("deterministic"):
        raise ValueError(f"{source}: strict deterministic evaluation was not enabled")
    for phase in ("validation", "metrics"):
        arm_a_rows = arm_a[phase]["total_raw_pov_rows"]
        arm_b_rows = arm_b[phase]["total_raw_pov_rows"]
        if arm_a_rows != arm_b_rows:
            raise ValueError(
                f"{source}: {phase} raw POV rows differ: {arm_a_name}={arm_a_rows}, {arm_b_name}={arm_b_rows}"
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
    arm_a_training_group_mode: str | None = None,
    arm_b_training_group_mode: str | None = None,
    expected_action_routing: str | None = None,
) -> dict[str, Any]:
    seed_dirs = sorted(path for path in root.glob("seed_*") if path.is_dir())
    if not seed_dirs:
        raise FileNotFoundError(f"No seed_* directories found in {root}")

    arms: dict[str, list[dict[str, Any]]] = {arm_a_name: [], arm_b_name: []}
    for seed_dir in seed_dirs:
        arm_a = _read(seed_dir / f"{arm_a_name}.json")
        arm_b = _read(seed_dir / f"{arm_b_name}.json")
        _validate_pair(
            arm_a,
            arm_b,
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
        arms[arm_a_name].append(arm_a)
        arms[arm_b_name].append(arm_b)

    seeds = [int(item["seed"]) for item in arms[arm_a_name]]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Evaluation seeds are not unique: {seeds}")
    for arm in arms.values():
        if len({item["checkpoint_sha256"] for item in arm}) != 1:
            raise ValueError("An arm changed checkpoints between evaluation seeds")

    metric_names = sorted(arms[arm_a_name][0]["results"])
    arm_summary: dict[str, Any] = {}
    for arm_name, payloads in arms.items():
        arm_summary[arm_name] = {
            metric: _summary([float(payload["results"][metric]) for payload in payloads])
            for metric in metric_names
        }

    paired: dict[str, Any] = {}
    for metric in metric_names:
        deltas = [
            float(arm_b["results"][metric]) - float(arm_a["results"][metric])
            for arm_a, arm_b in zip(arms[arm_a_name], arms[arm_b_name], strict=True)
        ]
        direction = _direction(metric)
        improvements = (
            [-delta for delta in deltas] if direction == "lower" else deltas if direction == "higher" else []
        )
        paired[metric] = {
            "direction": direction,
            f"{arm_b_name}_minus_{arm_a_name}": _summary(deltas),
        }
        if improvements:
            paired[metric][f"{arm_b_name}_improvement"] = _summary(improvements)

    first_arm_a = arms[arm_a_name][0]
    first_arm_b = arms[arm_b_name][0]
    return {
        "contract": {
            "arm_a": arm_a_name,
            "arm_b": arm_b_name,
            "split": first_arm_a["split"],
            "map_slug": first_arm_a["map_slug"],
            "dino_model": first_arm_a["dino_model"],
            "window_mode": first_arm_a["window_mode"],
            "action_routing": first_arm_a.get("action_routing"),
            "deterministic": first_arm_a["deterministic"],
            "seeds": seeds,
            "validation_raw_pov_rows_per_seed": first_arm_a["validation"]["total_raw_pov_rows"],
            "metrics_raw_pov_rows_per_seed": first_arm_a["metrics"]["total_raw_pov_rows"],
            f"{arm_a_name}_checkpoint_sha256": first_arm_a["checkpoint_sha256"],
            f"{arm_b_name}_checkpoint_sha256": first_arm_b["checkpoint_sha256"],
        },
        "arms": arm_summary,
        "paired": paired,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Directory containing seed_<N>/ result pairs.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: <root>/summary.json).",
    )
    parser.add_argument("--arm-a", default="single")
    parser.add_argument("--arm-b", default="shared")
    parser.add_argument("--arm-a-eval-group-mode", default="single")
    parser.add_argument("--arm-b-eval-group-mode", default="synchronized")
    parser.add_argument("--arm-a-n-players", type=int, default=1)
    parser.add_argument("--arm-b-n-players", type=int, default=10)
    parser.add_argument("--arm-a-training-group-mode", default=None)
    parser.add_argument("--arm-b-training-group-mode", default=None)
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
