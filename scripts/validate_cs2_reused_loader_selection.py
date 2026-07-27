#!/usr/bin/env python
"""Validate prior full GH200 loader evidence plus a cheap current-node smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REQUIRED_GROUP_MODES = {"synchronized", "shuffled"}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_reuse(
    selection_path: Path,
    smoke_path: Path,
    *,
    expected_current_commit: str,
    expected_manifest_sha256: str,
    expected_hostname: str,
) -> dict[str, Any]:
    """Return audited reuse provenance without rewriting historical benchmark facts."""
    selection = _read_json(selection_path)
    _require(
        selection.get("schema") == "mira-cs2-frozen-loader-selection-v1",
        f"{selection_path}: unsupported selection schema",
    )
    _require(selection.get("status") == "pass", f"{selection_path}: selection did not pass")
    _require(
        selection.get("expected_manifest_sha256") == expected_manifest_sha256,
        f"{selection_path}: manifest SHA-256 drifted",
    )
    _require(
        str(selection.get("expected_gpu_substring", "")).lower() == "gh200",
        f"{selection_path}: selection is not GH200 evidence",
    )
    source_inputs = selection.get("benchmark_inputs", [])
    _require(len(source_inputs) >= 3, f"{selection_path}: fewer than three full benchmark repeats")
    _require(
        all(item.get("manifest_sha256") == expected_manifest_sha256 for item in source_inputs),
        f"{selection_path}: a source benchmark used a different manifest",
    )
    source_counts = selection.get("benchmark_host_repeat_counts", {})
    _require(
        isinstance(source_counts, dict)
        and sum(int(value) for value in source_counts.values()) == len(source_inputs)
        and all(int(value) >= 3 for value in source_counts.values()),
        f"{selection_path}: invalid source host repeat counts",
    )
    selected_config = selection.get("selected_config")
    _require(isinstance(selected_config, dict), f"{selection_path}: selected_config is absent")
    _require(
        isinstance(selected_config.get("num_workers"), int)
        and selected_config["num_workers"] > 0,
        f"{selection_path}: selected num_workers is invalid",
    )

    smoke = _read_json(smoke_path)
    _require(
        smoke.get("schema") == "mira-cs2-dataloader-benchmark-v1",
        f"{smoke_path}: unsupported smoke schema",
    )
    _require(smoke.get("status") == "pass", f"{smoke_path}: current-node smoke failed")
    provenance = smoke.get("provenance", {})
    _require(provenance.get("git_clean") is True, f"{smoke_path}: source checkout was dirty")
    _require(
        provenance.get("git_commit") == expected_current_commit,
        f"{smoke_path}: current code commit drifted",
    )
    _require(
        provenance.get("hostname") == expected_hostname,
        f"{smoke_path}: smoke ran on {provenance.get('hostname')}, expected {expected_hostname}",
    )
    _require(
        provenance.get("manifest_sha256") == expected_manifest_sha256,
        f"{smoke_path}: current-node manifest drifted",
    )
    _require(
        "gh200" in str(provenance.get("gpu", "")).lower(),
        f"{smoke_path}: current-node smoke did not use a GH200",
    )
    config = smoke.get("config", {})
    _require(
        set(config.get("group_modes", [])) == REQUIRED_GROUP_MODES,
        f"{smoke_path}: smoke must cover synchronized and shuffled",
    )
    _require(
        config.get("workers") == [selected_config["num_workers"]],
        f"{smoke_path}: smoke did not use the frozen worker count",
    )
    for field in ("prefetch_factor", "persistent_workers", "pin_memory"):
        _require(
            config.get(field) == selected_config.get(field),
            f"{smoke_path}: {field} differs from the frozen selection",
        )
    _require(config.get("warmup_batches") == 1, f"{smoke_path}: expected one warmup batch")
    _require(
        int(config.get("timed_batches", 0)) >= 2,
        f"{smoke_path}: expected at least two timed batches",
    )
    results = smoke.get("results", [])
    _require(len(results) == 2, f"{smoke_path}: expected exactly two smoke cells")
    _require(
        {item.get("group_mode") for item in results} == REQUIRED_GROUP_MODES,
        f"{smoke_path}: smoke result modes drifted",
    )
    _require(
        all(
            item.get("status") == "pass"
            and item.get("num_workers") == selected_config["num_workers"]
            and item.get("transfer_device") in {"cuda", "cuda:0"}
            for item in results
        ),
        f"{smoke_path}: a current-node smoke cell failed",
    )

    return {
        "schema": "mira-cs2-reused-loader-selection-v1",
        "status": "pass",
        "expected_git_commit": expected_current_commit,
        "expected_gpu_substring": "GH200",
        "expected_hostname": expected_hostname,
        "expected_manifest_sha256": expected_manifest_sha256,
        "selected_config": selected_config,
        "source_selection": {
            "path": str(selection_path.resolve()),
            "sha256": _sha256(selection_path),
            "git_commit": selection.get("expected_git_commit"),
            "benchmark_host_repeat_counts": source_counts,
            "benchmark_input_count": len(source_inputs),
        },
        "current_node_smoke": {
            "path": str(smoke_path.resolve()),
            "sha256": _sha256(smoke_path),
            "hostname": provenance.get("hostname"),
            "git_commit": provenance.get("git_commit"),
            "completed_at_utc": smoke.get("completed_at_utc"),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--expected-current-commit", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-hostname", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = validate_reuse(
        args.selection,
        args.smoke,
        expected_current_commit=args.expected_current_commit,
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_hostname=args.expected_hostname,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
