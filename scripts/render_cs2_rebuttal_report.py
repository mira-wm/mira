#!/usr/bin/env python3
"""Render an audit-gated, deterministic Markdown report for the Dust2 rebuttal experiment.

This script does not recompute or reinterpret metrics. It verifies that the final run audit passed,
that all summary contracts match the preregistered seed grids, and that every summary is bound to
the audited final checkpoints. Identical input JSON files produce identical report bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

PRIMARY_SEEDS = [37, 38, 39]
ACTION_SEEDS = [37, 38, 39, 40, 41]
ACTION_MODES = ["true", "batch-shifted", "time-shifted", "zero"]
SUMMARY_PATHS = {
    "primary": Path("evaluation/test_seed_sweep/summary.json"),
    "midpoint": Path("evaluation/action_loss_seed_sweep/summary.json"),
    "first_death": Path("evaluation/death_action_loss_seed_sweep/summary.json"),
}


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


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_statistic(value: Any, source: str) -> dict[str, float | int]:
    _require(isinstance(value, dict), f"{source}: expected a statistic object")
    for key in ("n", "mean", "sample_std", "min", "max"):
        _require(key in value, f"{source}: missing {key}")
        _require(
            isinstance(value[key], (int, float)) and math.isfinite(float(value[key])),
            f"{source}: {key} must be finite",
        )
    _require(int(value["n"]) > 0, f"{source}: n must be positive")
    return value


def _mean_sd(value: Any, source: str) -> str:
    statistic = _validate_statistic(value, source)
    return f"{float(statistic['mean']):.6g} ± {float(statistic['sample_std']):.3g}"


def _escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _validate_contracts(
    audit: dict[str, Any],
    primary: dict[str, Any],
    midpoint: dict[str, Any],
    first_death: dict[str, Any],
) -> None:
    _require(audit.get("status") == "pass", "audit.json must have status=pass")
    _require(bool(audit.get("training_commit")), "audit.json is missing training_commit")
    _require(bool(audit.get("evaluator_commit")), "audit.json is missing evaluator_commit")

    primary_contract = primary.get("contract", {})
    _require(primary_contract.get("seeds") == PRIMARY_SEEDS, "primary seed grid drifted")
    _require(primary_contract.get("split") == "test", "primary split must be test")
    _require(primary_contract.get("map_slug") in {"dust2", "de_dust2"}, "primary map must be Dust2")
    _require(primary_contract.get("deterministic") is True, "primary evaluation must be deterministic")
    _require(
        primary_contract.get("validation_raw_pov_rows_per_seed") == 520,
        "primary validation must cover 520 raw POV rows per seed",
    )
    _require(
        primary_contract.get("metrics_raw_pov_rows_per_seed") == 520,
        "primary rollout metrics must cover 520 raw POV rows per seed",
    )

    for label, summary, window in (
        ("midpoint", midpoint, "midpoint"),
        ("first-death", first_death, "first-death"),
    ):
        contract = summary.get("contract", {})
        _require(contract.get("seeds") == ACTION_SEEDS, f"{label} action seed grid drifted")
        _require(contract.get("action_modes") == ACTION_MODES, f"{label} action modes drifted")
        _require(contract.get("window_mode") == window, f"{label} window mode drifted")
        _require(contract.get("split") == "test", f"{label} split must be test")
        _require(contract.get("map_slug") in {"dust2", "de_dust2"}, f"{label} map must be Dust2")
        _require(contract.get("deterministic") is True, f"{label} evaluation must be deterministic")
        _require(
            contract.get("validation_raw_pov_rows_per_arm_per_seed") == 520,
            f"{label} action evaluation must cover 520 raw POV rows per arm and seed",
        )

    stages = audit.get("stages", {})
    for arm in ("single", "shared"):
        audited_hash = stages.get(arm, {}).get("checkpoint_sha256")
        _require(bool(audited_hash), f"audit.json is missing {arm} checkpoint SHA-256")
        _require(
            primary_contract.get(f"{arm}_checkpoint_sha256") == audited_hash,
            f"primary summary is not bound to audited {arm} checkpoint",
        )
        for label, summary in (("midpoint", midpoint), ("first-death", first_death)):
            _require(
                summary.get("contract", {}).get(f"{arm}_checkpoint_sha256") == audited_hash,
                f"{label} summary is not bound to audited {arm} checkpoint",
            )


def _primary_table(summary: dict[str, Any]) -> list[str]:
    contract = summary["contract"]
    arm_a = str(contract["arm_a"])
    arm_b = str(contract["arm_b"])
    delta_key = f"{arm_b}_minus_{arm_a}"
    improvement_key = f"{arm_b}_improvement"
    metric_names = sorted(summary["paired"])
    lines = [
        (
            f"| Metric | Direction | {_escape(arm_a)} mean ± SD | {_escape(arm_b)} mean ± SD | "
            f"{_escape(arm_b)} − {_escape(arm_a)} mean ± SD | "
            "Directional improvement mean ± SD |"
        ),
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metric in metric_names:
        paired = summary["paired"][metric]
        direction = paired["direction"]
        improvement = (
            _mean_sd(paired[improvement_key], f"primary.{metric}.improvement")
            if improvement_key in paired
            else "—"
        )
        lines.append(
            f"| `{_escape(metric)}` | {_escape(direction)} | "
            f"{_mean_sd(summary['arms'][arm_a][metric], f'primary.{arm_a}.{metric}')} | "
            f"{_mean_sd(summary['arms'][arm_b][metric], f'primary.{arm_b}.{metric}')} | "
            f"{_mean_sd(paired[delta_key], f'primary.{metric}.delta')} | {improvement} |"
        )
    return lines


def _action_table(summary: dict[str, Any], label: str) -> list[str]:
    lines = [
        "| Arm | Metric | Action mode | Mean ± SD | Paired degradation vs true mean ± SD |",
        "|---|---|---|---:|---:|",
    ]
    for arm in ("single", "shared"):
        for metric in sorted(summary["arms"][arm]):
            modes = summary["arms"][arm][metric]
            for mode in ACTION_MODES:
                mode_summary = modes[mode]
                degradation = (
                    _mean_sd(
                        mode_summary["paired_degradation_vs_true"],
                        f"{label}.{arm}.{metric}.{mode}.degradation",
                    )
                    if mode != "true"
                    else "—"
                )
                lines.append(
                    f"| {_escape(arm)} | `{_escape(metric)}` | {_escape(mode)} | "
                    f"{_mean_sd(mode_summary, f'{label}.{arm}.{metric}.{mode}')} | {degradation} |"
                )
    return lines


def render_report(run_root: Path, *, audit_path: Path | None = None) -> str:
    inputs = {"audit": audit_path or run_root / "audit.json"}
    inputs.update({name: run_root / relative for name, relative in SUMMARY_PATHS.items()})
    payloads = {name: _read_json(path) for name, path in inputs.items()}
    audit = payloads["audit"]
    primary = payloads["primary"]
    midpoint = payloads["midpoint"]
    first_death = payloads["first_death"]
    _validate_contracts(audit, primary, midpoint, first_death)

    stages = audit["stages"]
    primary_contract = primary["contract"]
    lines = [
        "# MIRA Dust2 fixed-compute pilot report",
        "",
        (
            "This report is rendered only after the fail-closed run audit passes. It reports the "
            "saved summary statistics without selecting or dropping metrics."
        ),
        "",
        (
            "> Statistical scope: repeated rollout/evaluation seeds estimate inference-time "
            "variation for one trained checkpoint per arm. They are not independent training seeds, "
            "so this pilot does not establish training-run uncertainty or a definitive "
            "synchronization effect."
        ),
        "",
        "## Certification",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Audit status | `{_escape(audit['status'])}` |",
        f"| Map / split | `{_escape(primary_contract['map_slug'])}` / `{_escape(primary_contract['split'])}` |",
        f"| Training commit | `{_escape(audit['training_commit'])}` |",
        f"| Evaluator commit | `{_escape(audit['evaluator_commit'])}` |",
        f"| Dataset selection SHA-256 | `{_escape(audit['dataset_selection_sha256'])}` |",
        f"| Primary seeds | `{', '.join(map(str, PRIMARY_SEEDS))}` |",
        f"| Action seeds | `{', '.join(map(str, ACTION_SEEDS))}` |",
        "| Held-out coverage | `52 rounds / 520 POV rows per arm and seed` |",
        "",
        "## Fixed-compute training",
        "",
        "| Stage | Final step | Wall seconds | Last logged frames | Checkpoint SHA-256 |",
        "|---|---:|---:|---:|---|",
    ]
    for stage in ("codec", "single", "shared"):
        item = stages[stage]
        lines.append(
            f"| {_escape(stage)} | {int(item['time_limit_step'])} | "
            f"{float(item['elapsed_wall_seconds']):.3f} | "
            f"{int(item['last_logged_processed_frames'])} | `{_escape(item['checkpoint_sha256'])}` |"
        )
    lines.extend(
        [
            "",
            "## Paired held-out rollout metrics",
            "",
            (
                f"Values are mean ± sample SD across {len(PRIMARY_SEEDS)} deterministic rollout "
                "seeds. Directional improvement is sign-normalized so positive is better."
            ),
            "",
            *_primary_table(primary),
            "",
            "## Midpoint action sensitivity",
            "",
            (
                f"Values are mean ± sample SD across {len(ACTION_SEEDS)} deterministic seeds. "
                "Positive paired degradation means the intervention increased loss versus true "
                "actions."
            ),
            "",
            *_action_table(midpoint, "midpoint"),
            "",
            "## First-death action sensitivity",
            "",
            (
                f"Values are mean ± sample SD across {len(ACTION_SEEDS)} deterministic seeds on "
                "windows centered at each held-out round's first death."
            ),
            "",
            *_action_table(first_death, "first_death"),
            "",
            "## Input artifact hashes",
            "",
            "| Input | SHA-256 |",
            "|---|---|",
        ]
    )
    for name in ("audit", "primary", "midpoint", "first_death"):
        lines.append(f"| {_escape(name)} | `{_sha256(inputs[name])}` |")
    lines.extend(
        [
            "",
            "## Reproduction",
            "",
            "```bash",
            "python scripts/render_cs2_rebuttal_report.py /path/to/audited/run",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument(
        "--audit",
        type=Path,
        help="Strict audit JSON to bind into the report (default: <run_root>/audit.json).",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or args.run_root / "rebuttal_report.md"
    report = render_report(
        args.run_root.resolve(),
        audit_path=args.audit.resolve() if args.audit else None,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(report, encoding="utf-8")
    temporary.replace(output)
    print(output)


if __name__ == "__main__":
    main()
