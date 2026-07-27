#!/usr/bin/env python
"""Publish a checkpoint-free, presigned live manifest for the CS2 single-MIRA run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import mimetypes
import time
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "mira-dust2-live-review-v1"
CONTROL_FRAGMENT_SCHEMA = "mira-dust2-gh200-review-fragment-v1"
SAFE_PROVENANCE = (
    "code_commit.txt",
    "code_status.txt",
    "dataset.json",
    "dataset_files.sha256",
    "environment.txt",
    "environment_files.sha256",
    "gpu_timeseries.csv",
    "nvidia_smi_q.txt",
    "single_checkpoint.sha256",
)
SAFE_SUFFIXES = {".json", ".jsonl", ".mp4", ".tsv", ".txt", ".csv", ".md"}
MAX_TEXT_BYTES = 25 * 1024 * 1024
STAGE_ENDPOINTS = {"codec": 18_000, "single": 15_000}
ACTION_MODES = ("true", "round-shifted", "time-shifted", "zero")
WINDOW_MODES = ("midpoint", "first-death")
EVAL_SEEDS = (37, 41, 43)


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def object_key(prefix: str, relative: str | Path) -> str:
    return str(PurePosixPath(prefix.strip("/")) / PurePosixPath(str(relative).replace("\\", "/")))


def _all_mapping_keys(value: Any) -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            keys.append(str(key))
            keys.extend(_all_mapping_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.extend(_all_mapping_keys(item))
    return keys


def _fragment_object_keys(value: Any) -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"object_key", "metrics_object_key"} and item is not None:
                if not isinstance(item, str):
                    raise TypeError(f"control fragment {key} must be a string")
                keys.append(item)
            else:
                keys.extend(_fragment_object_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.extend(_fragment_object_keys(item))
    return keys


def load_control_fragment(client, bucket: str, prefix: str) -> dict[str, Any] | None:
    """Read and validate the optional private synchronized-control progress fragment."""
    from botocore.exceptions import ClientError

    fragment_key = object_key(prefix, "gh200/fragment.json")
    try:
        response = client.get_object(Bucket=bucket, Key=fragment_key)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise
    body = response["Body"].read()
    if len(body) > 5 * 1024 * 1024:
        raise ValueError("control review fragment exceeds 5 MiB")
    fragment = json.loads(body)
    if not isinstance(fragment, dict):
        raise TypeError("control review fragment must be a JSON object")
    if fragment.get("schema") != CONTROL_FRAGMENT_SCHEMA:
        raise ValueError("control review fragment schema drifted")
    if fragment.get("bucket") != bucket:
        raise ValueError("control review fragment bucket drifted")
    expected_prefix = object_key(prefix, "gh200")
    if fragment.get("object_prefix") != expected_prefix:
        raise ValueError("control review fragment object prefix drifted")
    if any(key == "url" for key in _all_mapping_keys(fragment)):
        raise ValueError("control review fragment must contain object keys, not URLs")
    allowed_prefix = expected_prefix.rstrip("/") + "/"
    forbidden = ("checkpoint.pth", "training_state.pth", "/optimizer")
    if any(
        not key.startswith(allowed_prefix) or any(token in key.lower() for token in forbidden)
        for key in _fragment_object_keys(fragment)
    ):
        raise ValueError("control review fragment contains an unsafe object reference")
    privacy = fragment.get("privacy", {})
    if (
        privacy.get("browser_checkpoint_urls_issued") is not False
        or privacy.get("training_state_uploaded") is not False
    ):
        raise ValueError("control review fragment violates the browser privacy contract")
    return fragment


def merge_control_fragment(
    manifest: dict[str, Any],
    fragment: dict[str, Any],
    *,
    client,
    bucket: str,
    expires_seconds: int,
) -> dict[str, Any]:
    """Overlay live synchronized-control progress while retaining completed endpoint evidence."""
    completed_traces = json.loads(json.dumps(manifest.get("traces", [])))
    completed_artifacts = json.loads(json.dumps(manifest.get("artifacts", [])))
    object_keys = sorted(set(_fragment_object_keys(fragment)))
    urls = {key: signed_url(client, bucket, key, expires_seconds) for key in object_keys}
    traces = json.loads(json.dumps(fragment.get("traces", [])))
    for trace in traces:
        for video in trace.get("videos", []):
            key = video.pop("object_key", None)
            if key:
                video["url"] = urls[key]
        key = trace.pop("metrics_object_key", None)
        trace["metrics_url"] = urls.get(key)
    artifacts = json.loads(json.dumps(fragment.get("artifacts", [])))
    for artifact in artifacts:
        key = artifact.pop("object_key")
        artifact["url"] = urls[key]

    generated = datetime.fromisoformat(str(fragment["generated_at_utc"]).replace("Z", "+00:00"))
    age_seconds = max(0.0, (datetime.now(UTC) - generated).total_seconds())
    experiment = json.loads(json.dumps(fragment["experiment"]))
    active_statuses = {"queued", "training", "evaluating"}
    transport_status = (
        "stale" if experiment.get("status") in active_statuses and age_seconds > 180 else "live"
    )
    experiment["review_transport_status"] = transport_status
    experiment["review_fragment_age_seconds"] = round(age_seconds, 1)
    if transport_status == "stale":
        experiment["status_label"] = (
            f"Review publisher stale for {int(age_seconds // 60)} minutes"
        )

    manifest.update(
        {
            "experiment": experiment,
            "stages": fragment.get("stages", []),
            "traces": completed_traces + traces,
            "dataset": fragment["dataset"],
            "telemetry": fragment["telemetry"],
            "provenance": fragment["provenance"],
            "artifacts": completed_artifacts + artifacts,
            "control_fragment_generated_at_utc": fragment["generated_at_utc"],
        }
    )
    manifest["privacy"] = {
        **manifest["privacy"],
        **fragment["privacy"],
        "artifact_url_expiry_seconds": expires_seconds,
        "bucket_public_access_blocked": True,
    }
    return manifest


def assert_private_bucket(client, bucket: str) -> None:
    block = client.get_public_access_block(Bucket=bucket)["PublicAccessBlockConfiguration"]
    required = (
        "BlockPublicAcls",
        "IgnorePublicAcls",
        "BlockPublicPolicy",
        "RestrictPublicBuckets",
    )
    if not all(block.get(field) is True for field in required):
        raise ValueError(f"S3 bucket {bucket!r} does not block every public access path")


def _safe_tree(root: Path, relative_root: PurePosixPath) -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    if not root.exists():
        return files
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SAFE_SUFFIXES:
            continue
        if path.suffix.lower() != ".mp4" and path.stat().st_size > MAX_TEXT_BYTES:
            continue
        lowered = path.as_posix().lower()
        if "checkpoint" in lowered or "training_state" in lowered or "optimizer" in lowered:
            raise ValueError(f"unsafe review artifact path: {path}")
        files.append((path, str(relative_root / path.relative_to(root))))
    return files


def safe_files(run_root: Path) -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    for path, relative in (
        (run_root / "pipeline_status.tsv", "raw/status/pipeline_status.tsv"),
        (run_root / "single" / "metrics.jsonl", "raw/single/metrics.jsonl"),
        (run_root / "codec" / "metrics.jsonl", "raw/codec/metrics.jsonl"),
    ):
        if path.is_file():
            files.append((path, relative))
    for name in SAFE_PROVENANCE:
        path = run_root / "provenance" / name
        if path.is_file():
            files.append((path, f"raw/provenance/{name}"))
    files.extend(
        _safe_tree(
            run_root / "single" / "rollout_traces",
            PurePosixPath("training-rollouts"),
        )
    )
    files.extend(
        _safe_tree(
            run_root / "evaluation" / "action_conditioning",
            PurePosixPath("action-conditioning"),
        )
    )
    files.extend(
        _safe_tree(
            run_root / "evaluation" / "generated_action_adherence",
            PurePosixPath("generated-action-adherence"),
        )
    )
    return files


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_statuses(path: Path) -> dict[str, str]:
    statuses: dict[str, str] = {}
    if not path.is_file():
        return statuses
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) >= 3:
            statuses[fields[1]] = fields[2]
    return statuses


def stage_manifest(run_root: Path, name: str, statuses: dict[str, str]) -> dict[str, Any]:
    rows = read_jsonl(run_root / name / "metrics.jsonl")
    train_rows = [row for row in rows if row.get("kind") == "train"]
    validation_rows = [row for row in rows if row.get("kind") == "validation"]
    latest = train_rows[-1] if train_rows else {}
    endpoint = STAGE_ENDPOINTS[name]
    latest_step = int(latest.get("step", 0)) if latest else 0
    status = statuses.get(name, "pending")
    progress = 100.0 if status == "complete" else min(100.0, latest_step / endpoint * 100)
    return {
        "name": name,
        "role": "Representation" if name == "codec" else "Single world model",
        "label": "Fresh RAE codec" if name == "codec" else "Single-POV MIRA",
        "status": status,
        "progress_percent": round(progress, 2),
        "latest_step": latest_step if latest else None,
        "endpoint_step": endpoint,
        "latest_train_loss": latest.get("train/loss_total"),
        "processed_frames": int(latest.get("System/n_frames_processed", 0)),
        "latest_throughput_fps": latest.get("System/throughput_fps_total"),
        "validation": [
            {
                "step": int(row["step"]),
                "loss": row.get("test/loss_total", row.get("test/loss_diffusion")),
                "timestamp_utc": row.get("timestamp_utc"),
            }
            for row in validation_rows
        ],
    }


def action_stage(run_root: Path, statuses: dict[str, str]) -> dict[str, Any]:
    root = run_root / "evaluation" / "action_conditioning"
    completed = sum(
        (root / window / f"seed_{seed}" / f"{mode}.json").is_file()
        for window in WINDOW_MODES
        for seed in EVAL_SEEDS
        for mode in ACTION_MODES
    )
    expected = len(WINDOW_MODES) * len(EVAL_SEEDS) * len(ACTION_MODES)
    status = statuses.get("action_evaluation", "pending")
    return {
        "name": "action_evaluation",
        "role": "Conditioning-use endpoint",
        "label": "690-POV paired action evaluation",
        "status": status,
        "progress_percent": round(completed / expected * 100, 2),
        "completed_cells": completed,
        "expected_cells": expected,
    }


def upload_file(client, bucket: str, key: str, path: Path) -> None:
    client.upload_file(
        str(path),
        bucket,
        key,
        ExtraArgs={
            "ContentType": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
            "ServerSideEncryption": "AES256",
            "CacheControl": "private, max-age=86400" if path.suffix == ".mp4" else "no-store",
        },
    )


def signed_url(client, bucket: str, key: str, expires: int) -> str:
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires,
    )


def telemetry(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"samples": 0}
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        "gpu_name": (rows[-1].get(" name", rows[-1].get("name")) if rows else None),
        "samples": len(rows),
        "latest_timestamp": (rows[-1].get("timestamp", rows[-1].get(" timestamp")) if rows else None),
    }


def build_traces(
    run_root: Path,
    *,
    uploaded: dict[str, str],
    urls: dict[str, str],
) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    rollout_root = run_root / "single" / "rollout_traces"
    for step_dir in sorted(rollout_root.glob("step-*")):
        metadata_path = step_dir / "metadata.json"
        video_path = step_dir / "rollout.mp4"
        if not metadata_path.is_file() or not video_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        relative_video = str(PurePosixPath("training-rollouts") / step_dir.name / video_path.name)
        relative_metadata = str(PurePosixPath("training-rollouts") / step_dir.name / metadata_path.name)
        if relative_video not in uploaded:
            continue
        traces.append(
            {
                "id": f"single-{step_dir.name}",
                "arm": "single",
                "kind": "training_rollout",
                "label": f"Single-MIRA validation rollout · step {int(metadata['step']):,}",
                "step": int(metadata["step"]),
                "seed": int(metadata["seed"]),
                "status": "ready",
                "message": "Fixed validation round and RNG; ground truth, reconstruction, and rollout.",
                "videos": [
                    {
                        "label": "fixed validation rollout",
                        "url": urls[uploaded[relative_video]],
                    }
                ],
                "metrics_url": urls.get(uploaded.get(relative_metadata, "")),
            }
        )
    eval_root = run_root / "evaluation" / "action_conditioning"
    for window in WINDOW_MODES:
        for seed in EVAL_SEEDS:
            completed = [
                mode
                for mode in ACTION_MODES
                if (eval_root / window / f"seed_{seed}" / f"{mode}.json").is_file()
            ]
            if not completed:
                continue
            summary_relative = "action-conditioning/summary.json"
            traces.append(
                {
                    "id": f"single-{window}-seed-{seed}",
                    "arm": "single",
                    "kind": f"{window}_action_sensitivity",
                    "label": f"{window.replace('-', ' ')} action sensitivity · seed {seed}",
                    "step": 15_000,
                    "seed": seed,
                    "status": "ready" if len(completed) == len(ACTION_MODES) else "running",
                    "message": (
                        f"Completed {len(completed)}/{len(ACTION_MODES)} modes over "
                        "all 69 untouched test rounds."
                    ),
                    "videos": [],
                    "metrics_url": urls.get(uploaded.get(summary_relative, "")),
                }
            )
    generated_root = run_root / "evaluation" / "generated_action_adherence" / "midpoint"
    review_manifest_path = generated_root / "review_manifest.json"
    if review_manifest_path.is_file():
        review_manifest = json.loads(review_manifest_path.read_text(encoding="utf-8"))
        videos = []
        for item in review_manifest:
            relative = str(PurePosixPath("generated-action-adherence/midpoint") / item["path"])
            key = uploaded.get(relative)
            if key:
                videos.append(
                    {
                        "label": item["sample_key"],
                        "url": urls[key],
                    }
                )
        summary_relative = "generated-action-adherence/midpoint/summary.json"
        traces.append(
            {
                "id": "single-generated-action-adherence-midpoint",
                "arm": "single",
                "kind": "generated_action_adherence",
                "label": "Generated midpoint action counterfactuals",
                "step": 15_000,
                "seed": 37,
                "status": "ready",
                "message": "Ground truth, true, cross-round shuffled, and zero-action columns.",
                "videos": videos,
                "metrics_url": urls.get(uploaded.get(summary_relative, "")),
            }
        )
    return traces


def publish_once(args: argparse.Namespace) -> dict[str, Any]:
    import boto3

    run_root = args.run_root.resolve()
    if not (run_root / "pipeline_status.tsv").is_file():
        raise FileNotFoundError(f"not a started single-MIRA pipeline: {run_root}")
    client = boto3.client("s3", region_name=args.region)
    assert_private_bucket(client, args.bucket)
    files = safe_files(run_root)
    uploaded = {relative: object_key(args.prefix, relative) for _, relative in files}
    for path, relative in files:
        upload_file(client, args.bucket, uploaded[relative], path)
    urls = {key: signed_url(client, args.bucket, key, args.artifact_url_expiry) for key in uploaded.values()}
    statuses = read_statuses(run_root / "pipeline_status.tsv")
    stages = [
        stage_manifest(run_root, "codec", statuses),
        stage_manifest(run_root, "single", statuses),
        action_stage(run_root, statuses),
    ]
    active = next((stage for stage in stages if stage["status"] == "running"), None)
    complete = statuses.get("pipeline") == "complete"
    manifest = {
        "schema": SCHEMA,
        "generated_at_utc": utc_now(),
        "refresh_after_seconds": args.interval_seconds,
        "experiment": {
            "run_id": run_root.name,
            "map": "de_dust2",
            "seed": 28,
            "comparison": "fresh corrected single-POV MIRA action-conditioning baseline",
            "status": "complete" if complete else "running",
            "status_label": ("Complete" if complete else active["label"] if active else "Preparing pipeline"),
            "overline": "DUST2 · SINGLE MIRA · UNTOUCHED 69-ROUND TEST",
            "title": "One world model. Correct actions. Paired counterfactuals.",
            "description": (
                "Fresh codec and single-POV MIRA trained without the confirmatory matches; "
                "fixed validation traces and round-paired action interventions."
            ),
        },
        "stages": stages,
        "traces": build_traces(run_root, uploaded=uploaded, urls=urls),
        "dataset": {
            "aligned_train_pov_hours": 87.08993055555555,
            "train_rounds": 766,
            "val_rounds": 54,
            "test_rounds": 69,
            "test_pov_rows": 690,
            "manifest_sha256": "33abbb623072932431871a612620110c473d4b664c52010e5763c273c6daf10e",
        },
        "telemetry": telemetry(run_root / "provenance" / "gpu_timeseries.csv"),
        "provenance": {
            "training_commit": args.training_commit,
            "evaluator_commit": args.training_commit,
            "strict_deterministic": True,
            "codec_endpoint": 18_000,
            "single_endpoint": 15_000,
        },
        "artifacts": [
            {"label": relative, "url": urls[key]}
            for relative, key in uploaded.items()
            if relative
            in {
                "raw/status/pipeline_status.tsv",
                "raw/provenance/dataset.json",
                "action-conditioning/summary.json",
            }
        ],
        "privacy": {
            "bucket_public_access_blocked": True,
            "object_encryption": "AES256",
            "training_state_uploaded": False,
            "browser_checkpoint_urls_issued": False,
            "artifact_url_expiry_seconds": args.artifact_url_expiry,
        },
    }
    fragment = load_control_fragment(client, args.bucket, args.prefix)
    if fragment is not None:
        manifest = merge_control_fragment(
            manifest,
            fragment,
            client=client,
            bucket=args.bucket,
            expires_seconds=args.artifact_url_expiry,
        )
    payload = json.dumps(manifest, indent=2, sort_keys=True).encode()
    manifest_key = object_key(args.prefix, "manifest.json")
    client.put_object(
        Bucket=args.bucket,
        Key=manifest_key,
        Body=payload,
        ContentType="application/json",
        ServerSideEncryption="AES256",
        CacheControl="no-store",
    )
    result = {
        "published_at_utc": manifest["generated_at_utc"],
        "manifest_s3_uri": f"s3://{args.bucket}/{manifest_key}",
        "manifest_sha256": hashlib.sha256(payload).hexdigest(),
        "uploaded_files": len(files),
        "traces": len(manifest["traces"]),
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--training-commit", required=True)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--artifact-url-expiry", type=int, default=14_400)
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    while True:
        try:
            publish_once(args)
        except Exception as exc:
            print(json.dumps({"published_at_utc": utc_now(), "error": repr(exc)}), flush=True)
            if not args.watch:
                raise
        if not args.watch:
            return
        time.sleep(max(15, args.interval_seconds))


if __name__ == "__main__":
    main()
