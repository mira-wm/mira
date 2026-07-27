#!/usr/bin/env python
"""Independently audit the completed CS2 single-MIRA confirmatory endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

EXPECTED_TRAINING_COMMIT = "686d3213c3831dea265952b4ead3a79af6f6fb43"
EXPECTED_EVALUATOR_COMMIT = "4363a9b1789d6a549f94e69d9f1bdcc24ebae4d4"
EXPECTED_MANIFEST_SHA256 = "33abbb623072932431871a612620110c473d4b664c52010e5763c273c6daf10e"
EXPECTED_CHECKPOINT_STEP = 15_000
EXPECTED_SAMPLES = 690
EXPECTED_ROUNDS = 69
EXPECTED_EVAL_SEEDS = (37, 41, 43)
EXPECTED_ACTION_MODES = ("true", "round-shifted", "time-shifted", "zero")
EXPECTED_WINDOW_MODES = ("midpoint", "first-death")
EXPECTED_ARCHIVE_MODES = ("true", "shuffled", "zeros")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"required artifact is absent: {path}") from None
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def assert_finite_tree(value: Any, *, label: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            assert_finite_tree(child, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_finite_tree(child, label=f"{label}[{index}]")
    elif isinstance(value, (int, float)) and not math.isfinite(float(value)):
        raise ValueError(f"non-finite value at {label}: {value}")


def verify_artifacts(metadata_path: Path, records: Any) -> dict[str, Any]:
    if not isinstance(records, dict) or not records:
        raise ValueError(f"artifact records are absent: {metadata_path}")
    audited = {}
    root = metadata_path.parent.resolve()
    for name, record in records.items():
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise TypeError(f"invalid artifact record {name!r}: {metadata_path}")
        path = metadata_path.parent / record["path"]
        if path.parent.resolve() != root:
            raise ValueError(f"artifact escapes its result directory: {path}")
        digest = sha256_file(path)
        if digest != record.get("sha256"):
            raise ValueError(f"artifact SHA-256 mismatch: {path}")
        audited[name] = {
            "path": record["path"],
            "size_bytes": path.stat().st_size,
            "sha256": digest,
            "dtype": record.get("dtype"),
            "shape": record.get("shape"),
        }
    return audited


def read_round_records(path: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != EXPECTED_ROUNDS or not all(isinstance(row, dict) for row in records):
        raise ValueError(f"{path}: expected {EXPECTED_ROUNDS} JSON-object round records")
    round_ids = [str(row.get("round_id")) for row in records]
    if len(set(round_ids)) != EXPECTED_ROUNDS:
        raise ValueError(f"{path}: round records are not unique and complete")
    return records


def verify_native_action_evaluation(root: Path, checkpoint_sha256: str) -> dict[str, Any]:
    summary_path = root / "summary.json"
    summary = load_json(summary_path)
    contract = summary.get("contract", {})
    expected_contract = {
        "checkpoint_sha256": checkpoint_sha256,
        "split": "test",
        "map_slug": "dust2",
        "training_seed": 28,
        "evaluation_seeds": list(EXPECTED_EVAL_SEEDS),
        "action_modes": list(EXPECTED_ACTION_MODES),
        "window_modes": list(EXPECTED_WINDOW_MODES),
        "rounds": EXPECTED_ROUNDS,
        "raw_pov_rows_per_evaluation": EXPECTED_SAMPLES,
        "inference_unit": "round_id",
    }
    if contract != expected_contract:
        raise ValueError(f"native action summary contract mismatch: {contract}")

    cells = {}
    identity_by_window: dict[str, dict[str, tuple[Any, Any]]] = {}
    for window in EXPECTED_WINDOW_MODES:
        identity_by_window[window] = {}
        for seed in EXPECTED_EVAL_SEEDS:
            for mode in EXPECTED_ACTION_MODES:
                result_path = root / window / f"seed_{seed}" / f"{mode}.json"
                sidecar_path = result_path.with_suffix(".rounds.jsonl")
                result = load_json(result_path)
                expected_identity = {
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
                }
                mismatches = {
                    field: {"expected": expected, "actual": result.get(field)}
                    for field, expected in expected_identity.items()
                    if result.get(field) != expected
                }
                if mismatches:
                    raise ValueError(f"{result_path}: identity mismatch: {mismatches}")
                validation = result.get("validation", {})
                if (
                    validation.get("total_raw_pov_rows") != EXPECTED_SAMPLES
                    or validation.get("num_batches") != EXPECTED_ROUNDS
                ):
                    raise ValueError(f"{result_path}: incomplete test coverage")
                per_batch = result.get("per_batch_records", {})
                if (
                    per_batch.get("count") != EXPECTED_ROUNDS
                    or per_batch.get("cluster_unit") != "round_id"
                    or per_batch.get("sha256") != sha256_file(sidecar_path)
                ):
                    raise ValueError(f"{result_path}: invalid or drifted round sidecar")
                records = read_round_records(sidecar_path)
                for row in records:
                    round_id = str(row["round_id"])
                    identity = (row.get("sample_keys"), row.get("pov_indices"))
                    baseline = identity_by_window[window].setdefault(round_id, identity)
                    if identity != baseline:
                        raise ValueError(
                            f"{window}/seed_{seed}/{mode}/{round_id}: sample identity drift"
                        )
                    donors = row.get("action_donor_round_ids", [])
                    if mode == "round-shifted":
                        if len(donors) != 1 or str(donors[0]) == round_id:
                            raise ValueError(
                                f"{window}/seed_{seed}/{mode}/{round_id}: invalid donor"
                            )
                    elif donors:
                        raise ValueError(
                            f"{window}/seed_{seed}/{mode}/{round_id}: unexpected donor"
                        )
                    assert_finite_tree(
                        row.get("metrics"),
                        label=f"{window}.seed_{seed}.{mode}.{round_id}.metrics",
                    )
                cells[f"{window}/seed_{seed}/{mode}"] = {
                    "result_sha256": sha256_file(result_path),
                    "sidecar_sha256": sha256_file(sidecar_path),
                    "rounds": len(records),
                }
    assert_finite_tree(summary.get("windows"), label="native_action_summary.windows")
    return {
        "summary_sha256": sha256_file(summary_path),
        "cells": cells,
        "cell_count": len(cells),
        "round_clusters_per_cell": EXPECTED_ROUNDS,
    }


def verify_probe(probe_summary_path: Path, probe_checkpoint: Path) -> dict[str, Any]:
    summary = load_json(probe_summary_path)
    checkpoint_sha256 = sha256_file(probe_checkpoint)
    if (
        summary.get("status") != "complete"
        or summary.get("checkpoint_sha256") != checkpoint_sha256
        or summary.get("selection_metric") != "validation_macro_average_precision"
        or summary.get("train_examples") != 200_000
        or summary.get("val_examples") != 20_000
    ):
        raise ValueError("temporal action probe contract is invalid")
    features = {}
    for split in ("train", "val"):
        metadata_path = Path(summary[f"{split}_feature_metadata"])
        expected_hash = summary[f"{split}_feature_metadata_sha256"]
        if sha256_file(metadata_path) != expected_hash:
            raise ValueError(f"{split} feature metadata SHA-256 mismatch")
        metadata = load_json(metadata_path)
        expected_windows = 100_000 if split == "train" else 10_000
        if (
            metadata.get("status") != "complete"
            or metadata.get("split") != split
            or metadata.get("num_windows") != expected_windows
        ):
            raise ValueError(f"{split} feature archive contract is invalid")
        features[split] = {
            "metadata_sha256": expected_hash,
            "artifacts": verify_artifacts(metadata_path, metadata.get("artifacts")),
        }
    validation_predictions = Path(summary["validation_predictions"])
    if sha256_file(validation_predictions) != summary["validation_predictions_sha256"]:
        raise ValueError("probe validation-prediction SHA-256 mismatch")
    assert_finite_tree(summary.get("selection_value"), label="probe.selection_value")
    return {
        "summary_sha256": sha256_file(probe_summary_path),
        "checkpoint_sha256": checkpoint_sha256,
        "best_epoch": summary.get("best_epoch"),
        "selection_value": summary.get("selection_value"),
        "features": features,
        "validation_predictions_sha256": summary["validation_predictions_sha256"],
    }


def audit_endpoint(
    run_root: Path,
    *,
    manifest: Path,
    probe_summary: Path,
    probe_checkpoint: Path,
    expected_training_commit: str = EXPECTED_TRAINING_COMMIT,
    expected_evaluator_commit: str = EXPECTED_EVALUATOR_COMMIT,
) -> dict[str, Any]:
    run_root = run_root.resolve()
    manifest_sha256 = sha256_file(manifest)
    if manifest_sha256 != EXPECTED_MANIFEST_SHA256:
        raise ValueError(f"confirmatory manifest SHA-256 drifted: {manifest_sha256}")

    provenance = run_root / "provenance"
    training_commit = (provenance / "code_commit.txt").read_text(encoding="utf-8").strip()
    evaluator_commit = (
        provenance / "post_test_generated_evaluator_commit.txt"
    ).read_text(encoding="utf-8").strip()
    if training_commit != expected_training_commit:
        raise ValueError(f"training commit {training_commit} != {expected_training_commit}")
    if evaluator_commit != expected_evaluator_commit:
        raise ValueError(f"evaluator commit {evaluator_commit} != {expected_evaluator_commit}")
    if (provenance / "code_status.txt").read_text(encoding="utf-8").strip():
        raise ValueError("training checkout was not clean at launch")

    checkpoint = run_root / "single" / "checkpoint-15000" / "checkpoint.pth"
    checkpoint_sha256 = sha256_file(checkpoint)
    expected_checkpoint_sha256 = (provenance / "single_checkpoint.sha256").read_text(
        encoding="utf-8"
    ).split()[0]
    if checkpoint_sha256 != expected_checkpoint_sha256:
        raise ValueError("step-15000 checkpoint SHA-256 drifted")

    pipeline_rows = [
        line.split("\t")
        for line in (run_root / "pipeline_status.tsv").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    completed_stages = {(row[1], row[2]) for row in pipeline_rows if len(row) == 3}
    for stage in ("codec", "single", "action_evaluation", "pipeline"):
        if (stage, "complete") not in completed_stages:
            raise ValueError(f"pipeline stage is incomplete: {stage}")

    native_root = run_root / "evaluation" / "action_conditioning"
    native = verify_native_action_evaluation(native_root, checkpoint_sha256)

    generated_root = run_root / "evaluation" / "generated_action_adherence" / "midpoint"
    generated_summary_path = generated_root / "summary.json"
    generated_summary = load_json(generated_summary_path)
    expected_generated = {
        "status": "complete",
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_step": EXPECTED_CHECKPOINT_STEP,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "window_mode": "midpoint",
        "eval_seeds": [37],
        "action_modes": list(EXPECTED_ARCHIVE_MODES),
        "rounds": EXPECTED_ROUNDS,
        "pov_rows": EXPECTED_SAMPLES,
        "rollout_steps": 8,
    }
    for field, expected in expected_generated.items():
        if generated_summary.get(field) != expected:
            raise ValueError(
                f"generated rollout summary {field}={generated_summary.get(field)!r}, "
                f"expected {expected!r}"
            )

    archive_metadata_path = generated_root / "rollout_archive" / "metadata.json"
    archive = load_json(archive_metadata_path)
    archive_contract = archive.get("contract", {})
    expected_archive = {
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_step": EXPECTED_CHECKPOINT_STEP,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "split": "test",
        "map_slug": "dust2",
        "window_mode": "midpoint",
        "num_samples": EXPECTED_SAMPLES,
        "eval_seeds": [37],
        "action_modes": list(EXPECTED_ARCHIVE_MODES),
        "rollout_steps": 8,
    }
    if archive.get("status") != "complete":
        raise ValueError("generated rollout archive is incomplete")
    for field, expected in expected_archive.items():
        if archive_contract.get(field) != expected:
            raise ValueError(
                f"generated rollout archive {field}={archive_contract.get(field)!r}, "
                f"expected {expected!r}"
            )
    archive_artifacts = verify_artifacts(archive_metadata_path, archive.get("artifacts"))
    archive_metadata_sha256 = sha256_file(archive_metadata_path)

    motion_path = generated_root / "motion_metrics" / "summary.json"
    motion = load_json(motion_path)
    if (
        motion.get("status") != "complete"
        or motion.get("archive_metadata_sha256") != archive_metadata_sha256
        or motion.get("num_samples") != EXPECTED_SAMPLES
        or motion.get("num_rounds") != EXPECTED_ROUNDS
        or motion.get("eval_seeds") != [37]
        or motion.get("action_modes") != list(EXPECTED_ARCHIVE_MODES)
    ):
        raise ValueError("generated rollout motion summary contract is invalid")
    motion_rows = Path(motion["per_sample_metrics"])
    if sha256_file(motion_rows) != motion["per_sample_metrics_sha256"]:
        raise ValueError("generated rollout motion rows SHA-256 mismatch")
    assert_finite_tree(motion.get("means"), label="generated_motion.means")
    assert_finite_tree(motion.get("paired_deltas"), label="generated_motion.paired_deltas")

    probe = verify_probe(probe_summary, probe_checkpoint)
    arr_path = generated_root / "action_recoverability" / "summary.json"
    arr = load_json(arr_path)
    expected_arr = {
        "status": "complete",
        "archive_metadata_sha256": archive_metadata_sha256,
        "sample_plan_sha256": archive_contract["sample_plan_sha256"],
        "probe_checkpoint_sha256": probe["checkpoint_sha256"],
        "num_samples": EXPECTED_SAMPLES,
        "num_complete_segments": 1_380,
        "num_incomplete_segments_excluded": 0,
        "eval_seeds": [37],
        "action_modes": list(EXPECTED_ARCHIVE_MODES),
    }
    for field, expected in expected_arr.items():
        if arr.get(field) != expected:
            raise ValueError(
                f"action recoverability {field}={arr.get(field)!r}, expected {expected!r}"
            )
    bootstrap = arr.get("results", {}).get("bootstrap", {})
    if (
        bootstrap.get("unit") != "round_id"
        or bootstrap.get("round_clusters") != EXPECTED_ROUNDS
        or bootstrap.get("replicates") != 10_000
    ):
        raise ValueError(f"action-recoverability bootstrap contract is invalid: {bootstrap}")
    arr_artifacts = verify_artifacts(arr_path, arr.get("artifacts"))
    assert_finite_tree(arr.get("results", {}).get("primary"), label="mira_arr.primary")

    temporary_files = sorted(
        str(path.relative_to(run_root))
        for root in (native_root, generated_root)
        for path in root.rglob("*")
        if path.is_file() and (path.name.startswith(".") or ".tmp" in path.name)
    )
    if temporary_files:
        raise ValueError(f"accepted endpoint contains temporary files: {temporary_files}")

    recovery_path = provenance / "post_test_recovery.json"
    recovery = load_json(recovery_path)
    if (
        recovery.get("status") != "recovered_complete"
        or recovery.get("checkpoint_sha256") != checkpoint_sha256
        or recovery.get("training_commit") != training_commit
        or recovery.get("evaluator_commit") != evaluator_commit
    ):
        raise ValueError("post-test recovery provenance is invalid")

    return {
        "schema_version": 1,
        "status": "pass",
        "purpose": "post_test_single_mira_artifact_integrity_audit",
        "run_root": str(run_root),
        "training_commit": training_commit,
        "evaluator_commit": evaluator_commit,
        "manifest_sha256": manifest_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_rehashed": True,
        "pipeline_status_sha256": sha256_file(run_root / "pipeline_status.tsv"),
        "native_action_evaluation": native,
        "generated_rollout": {
            "summary_sha256": sha256_file(generated_summary_path),
            "archive_metadata_sha256": archive_metadata_sha256,
            "artifacts": archive_artifacts,
            "motion_summary_sha256": sha256_file(motion_path),
            "motion_rows_sha256": motion["per_sample_metrics_sha256"],
            "arr_summary_sha256": sha256_file(arr_path),
            "arr_artifacts": arr_artifacts,
        },
        "probe": probe,
        "recovery_provenance_sha256": sha256_file(recovery_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--probe-summary", required=True, type=Path)
    parser.add_argument("--probe-checkpoint", required=True, type=Path)
    parser.add_argument("--expected-training-commit", default=EXPECTED_TRAINING_COMMIT)
    parser.add_argument("--expected-evaluator-commit", default=EXPECTED_EVALUATOR_COMMIT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    audit = audit_endpoint(
        args.run_root,
        manifest=args.manifest,
        probe_summary=args.probe_summary,
        probe_checkpoint=args.probe_checkpoint,
        expected_training_commit=args.expected_training_commit,
        expected_evaluator_commit=args.expected_evaluator_commit,
    )
    rendered = json.dumps(audit, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(args.output)
    print(rendered, end="")


if __name__ == "__main__":
    main()
