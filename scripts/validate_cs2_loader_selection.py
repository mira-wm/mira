#!/usr/bin/env python3
"""Freeze a target-hardware CS2 loader configuration from repeated exact-contract benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

REQUIRED_GROUP_MODES = ("synchronized", "shuffled")
MINIMUM_REPEATS = 3
MINIMUM_TIMED_BATCHES = 100


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


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def _summary(values: list[float]) -> dict[str, float | int]:
    _require(values and all(math.isfinite(value) and value > 0 for value in values), "invalid throughput")
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def select_num_workers(
    benchmark_paths: list[Path],
    *,
    prefetch_factor: int,
    persistent_workers: bool,
    pin_memory: bool,
) -> tuple[int, dict[str, Any]]:
    """Select one common worker count by a frozen maximin throughput rule.

    A candidate is eligible only when every benchmark repeat contains passing synchronized and
    shuffled cases for the exact queue configuration. The score is the lower of the two
    across-repeat mean input throughputs, so a setting cannot win by accelerating one treatment
    while starving the other. Exact score ties prefer fewer workers.
    """
    candidate_sets: list[set[int]] = []
    payloads = [_read_json(path) for path in benchmark_paths]
    for path, payload in zip(benchmark_paths, payloads, strict=True):
        modes_by_workers: dict[int, set[str]] = {}
        for result in payload.get("results", []):
            workers = result.get("num_workers")
            mode = result.get("group_mode")
            if (
                isinstance(workers, int)
                and workers > 0
                and mode in REQUIRED_GROUP_MODES
                and result.get("prefetch_factor") == prefetch_factor
                and result.get("persistent_workers") is persistent_workers
                and result.get("pin_memory") is pin_memory
                and result.get("status") == "pass"
                and int(result.get("timed_batches", 0)) >= MINIMUM_TIMED_BATCHES
                and result.get("transfer_device") in {"cuda", "cuda:0"}
            ):
                modes_by_workers.setdefault(workers, set()).add(str(mode))
        eligible = {
            workers for workers, modes in modes_by_workers.items() if modes == set(REQUIRED_GROUP_MODES)
        }
        _require(eligible, f"{path}: no eligible common loader candidate")
        candidate_sets.append(eligible)

    common = set.intersection(*candidate_sets)
    _require(common, "benchmark repeats have no common eligible loader candidate")
    candidates: dict[str, Any] = {}
    scored: list[tuple[float, int]] = []
    for workers in sorted(common):
        throughput = {mode: [] for mode in REQUIRED_GROUP_MODES}
        for payload in payloads:
            for result in payload["results"]:
                if (
                    result.get("num_workers") == workers
                    and result.get("group_mode") in REQUIRED_GROUP_MODES
                    and result.get("prefetch_factor") == prefetch_factor
                    and result.get("persistent_workers") is persistent_workers
                    and result.get("pin_memory") is pin_memory
                ):
                    throughput[str(result["group_mode"])].append(float(result["input_frames_per_second"]))
        summaries = {mode: _summary(values) for mode, values in throughput.items()}
        score = min(float(summary["mean"]) for summary in summaries.values())
        candidates[str(workers)] = {
            "score_minimum_group_mode_mean_input_fps": score,
            "throughput_input_frames_per_second": summaries,
        }
        scored.append((score, workers))

    selected_score, selected_workers = max(scored, key=lambda item: (item[0], -item[1]))
    return selected_workers, {
        "rule": (
            "maximize the minimum synchronized/shuffled across-repeat mean input FPS; "
            "prefer fewer workers on an exact tie"
        ),
        "selected_num_workers": selected_workers,
        "selected_score": selected_score,
        "candidates": candidates,
    }


def validate_selection(
    benchmark_paths: list[Path],
    *,
    num_workers: int,
    prefetch_factor: int,
    persistent_workers: bool,
    pin_memory: bool,
    expected_git_commit: str,
    expected_gpu_substring: str = "GH200",
    expected_hostname: str | None = None,
    expected_manifest_sha256: str | None = None,
    expected_host_count: int | None = None,
    minimum_repeats_per_hostname: int = 1,
) -> dict[str, Any]:
    _require(
        len(benchmark_paths) >= MINIMUM_REPEATS,
        f"at least {MINIMUM_REPEATS} independent benchmark repeats are required",
    )
    _require(num_workers > 0, "publication loader selection requires num_workers > 0")
    _require(prefetch_factor > 0, "prefetch_factor must be positive")
    _require(len(expected_git_commit) == 40, "expected_git_commit must be a full Git SHA")
    _require(
        minimum_repeats_per_hostname > 0,
        "minimum_repeats_per_hostname must be positive",
    )
    if expected_manifest_sha256 is not None:
        _require(
            len(expected_manifest_sha256) == 64,
            "expected_manifest_sha256 must be a full SHA-256 digest",
        )
    if expected_host_count is not None:
        _require(expected_host_count > 0, "expected_host_count must be positive")

    resolved_paths = [path.resolve() for path in benchmark_paths]
    _require(
        len(set(resolved_paths)) == len(resolved_paths),
        "benchmark paths must be distinct independent repeats",
    )

    observed_throughput: dict[str, list[float]] = {mode: [] for mode in REQUIRED_GROUP_MODES}
    inputs: list[dict[str, Any]] = []
    input_hashes: set[str] = set()
    completed_timestamps: set[str] = set()
    hostnames: list[str] = []
    for path in benchmark_paths:
        payload = _read_json(path)
        source = str(path)
        digest = _sha256(path)
        _require(
            digest not in input_hashes,
            f"{source}: duplicate benchmark payload; repeats must be independent",
        )
        input_hashes.add(digest)
        completed_at = payload.get("completed_at_utc")
        _require(
            isinstance(completed_at, str) and completed_at,
            f"{source}: missing completed_at_utc",
        )
        _require(
            completed_at not in completed_timestamps,
            f"{source}: duplicate completion timestamp; repeats must be independent",
        )
        completed_timestamps.add(completed_at)
        _require(payload.get("schema") == "mira-cs2-dataloader-benchmark-v1", f"{source}: schema drift")
        _require(payload.get("status") == "pass", f"{source}: benchmark did not pass")
        _require(payload.get("parity", {}).get("status") == "pass", f"{source}: tensor parity failed")

        provenance = payload.get("provenance", {})
        _require(provenance.get("git_clean") is True, f"{source}: benchmark source was dirty")
        _require(
            provenance.get("git_commit") == expected_git_commit,
            f"{source}: benchmark source commit drifted",
        )
        gpu = str(provenance.get("gpu", ""))
        _require(expected_gpu_substring.lower() in gpu.lower(), f"{source}: expected target GPU not found")
        hostname = str(provenance.get("hostname", ""))
        _require(hostname, f"{source}: missing benchmark hostname")
        if expected_hostname is not None:
            _require(
                hostname == expected_hostname,
                f"{source}: benchmark hostname {hostname!r} != {expected_hostname!r}",
            )
        hostnames.append(hostname)
        if expected_manifest_sha256 is not None:
            _require(
                provenance.get("manifest_sha256") == expected_manifest_sha256,
                f"{source}: benchmark manifest digest drifted",
            )

        config = payload.get("config", {})
        expected_contract = {
            "split": "val",
            "map_slug": "dust2",
            "clip_len": 16,
            "target_fps": 8,
            "frame_size": [168, 308],
            "shuffle": True,
        }
        for field, expected in expected_contract.items():
            _require(config.get(field) == expected, f"{source}: {field} must be {expected!r}")
        _require(config.get("transfer_device") in {"cuda", "cuda:0"}, f"{source}: CUDA transfer missing")
        _require(
            set(config.get("group_modes", [])) >= set(REQUIRED_GROUP_MODES),
            f"{source}: synchronized and shuffled benchmark cases are required",
        )

        selected: dict[str, dict[str, Any]] = {}
        for result in payload.get("results", []):
            if (
                result.get("group_mode") in REQUIRED_GROUP_MODES
                and result.get("num_workers") == num_workers
                and result.get("prefetch_factor") == prefetch_factor
                and result.get("persistent_workers") is persistent_workers
                and result.get("pin_memory") is pin_memory
            ):
                selected[str(result["group_mode"])] = result
        _require(
            set(selected) == set(REQUIRED_GROUP_MODES),
            f"{source}: selected loader configuration is missing a required group mode",
        )
        for mode, result in selected.items():
            _require(result.get("status") == "pass", f"{source}: {mode} selected case failed")
            _require(
                int(result.get("timed_batches", 0)) >= MINIMUM_TIMED_BATCHES,
                f"{source}: {mode} requires at least {MINIMUM_TIMED_BATCHES} timed batches",
            )
            _require(
                result.get("transfer_device") in {"cuda", "cuda:0"},
                f"{source}: {mode} selected case omitted CUDA transfer",
            )
            observed_throughput[mode].append(float(result["input_frames_per_second"]))

        inputs.append(
            {
                "path": str(path.resolve()),
                "sha256": digest,
                "gpu": gpu,
                "hostname": hostname,
                "completed_at_utc": completed_at,
                "manifest_sha256": provenance.get("manifest_sha256"),
            }
        )

    hostname_counts = Counter(hostnames)
    if expected_host_count is not None:
        _require(
            len(hostname_counts) == expected_host_count,
            f"expected {expected_host_count} benchmark hosts, found {dict(hostname_counts)}",
        )
    _require(
        all(count >= minimum_repeats_per_hostname for count in hostname_counts.values()),
        (
            f"each benchmark host requires at least {minimum_repeats_per_hostname} "
            f"independent repeats, found {dict(hostname_counts)}"
        ),
    )

    return {
        "schema": "mira-cs2-frozen-loader-selection-v1",
        "status": "pass",
        "expected_git_commit": expected_git_commit,
        "expected_gpu_substring": expected_gpu_substring,
        "expected_hostname": expected_hostname,
        "expected_manifest_sha256": expected_manifest_sha256,
        "benchmark_host_repeat_counts": dict(sorted(hostname_counts.items())),
        "selected_config": {
            "num_workers": num_workers,
            "prefetch_factor": prefetch_factor,
            "persistent_workers": persistent_workers,
            "pin_memory": pin_memory,
        },
        "required_contract": {
            "group_modes": list(REQUIRED_GROUP_MODES),
            "map_slug": "dust2",
            "clip_len": 16,
            "target_fps": 8,
            "frame_size": [168, 308],
            "transfer_device": "cuda",
            "minimum_repeats": MINIMUM_REPEATS,
            "minimum_timed_batches": MINIMUM_TIMED_BATCHES,
            "minimum_repeats_per_hostname": minimum_repeats_per_hostname,
        },
        "throughput_input_frames_per_second": {
            mode: _summary(values) for mode, values in observed_throughput.items()
        },
        "benchmark_inputs": inputs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmarks", type=Path, nargs="+")
    parser.add_argument(
        "--num-workers",
        required=True,
        help="Positive integer, or 'auto' for the frozen maximin selection rule.",
    )
    parser.add_argument("--prefetch-factor", type=int, required=True)
    parser.add_argument("--persistent-workers", type=_parse_bool, required=True)
    parser.add_argument("--pin-memory", type=_parse_bool, required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--expected-gpu-substring", default="GH200")
    parser.add_argument("--expected-hostname")
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--expected-host-count", type=int)
    parser.add_argument("--minimum-repeats-per-hostname", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_workers == "auto":
        num_workers, selection = select_num_workers(
            args.benchmarks,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=args.persistent_workers,
            pin_memory=args.pin_memory,
        )
    else:
        try:
            num_workers = int(args.num_workers)
        except ValueError as exc:
            raise ValueError("--num-workers must be a positive integer or 'auto'") from exc
        selection = {
            "rule": "explicit preregistered worker count",
            "selected_num_workers": num_workers,
        }
    payload = validate_selection(
        args.benchmarks,
        num_workers=num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        pin_memory=args.pin_memory,
        expected_git_commit=args.expected_git_commit,
        expected_gpu_substring=args.expected_gpu_substring,
        expected_hostname=args.expected_hostname,
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_host_count=args.expected_host_count,
        minimum_repeats_per_hostname=args.minimum_repeats_per_hostname,
    )
    payload["selection_evidence"] = selection
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
