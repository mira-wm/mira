"""Reproducible throughput and semantic-parity checks for the CounterStrike-1K loader.

The benchmark intentionally exercises the same model-facing contract as the Dust2 MIRA
experiments: ten POV rows, 16 frames per row at 8 fps, MIRA action aggregation, and either
independent or synchronized grouping.  It does not substitute a video-only microbenchmark for the
real training input pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import torch

from mira.data.batch import VideoActionBatch
from mira.data.counterstrike import CS2_KEYS, CounterStrikeClipMeta
from mira.data.decode import _resize
from mira.data.training_loader import create_loader

GroupMode = Literal["single", "synchronized", "shuffled"]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _metadata_rows(metadata: list[CounterStrikeClipMeta]) -> list[dict[str, Any]]:
    return [
        {
            "sample_key": item.sample_key,
            "match_id": item.match_id,
            "round_id": item.round_id,
            "perspective": item.perspective,
            "source_start_frame": item.source_start_frame,
            "frame_indices": item.frame_indices,
            "group_mode": item.group_mode,
            "window_mode": item.window_mode,
        }
        for item in metadata
    ]


def batch_signature(
    batch: VideoActionBatch,
    metadata: list[CounterStrikeClipMeta],
) -> dict[str, Any]:
    """Return content hashes and model-facing identifiers for one decoded batch."""
    metadata_rows = _metadata_rows(metadata)
    metadata_bytes = json.dumps(metadata_rows, sort_keys=True, separators=(",", ":")).encode()
    return {
        "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "video_sha256": _tensor_sha256(batch.video),
        "key_presses_sha256": _tensor_sha256(batch.actions.key_presses),
        "mouse_movements_sha256": _tensor_sha256(batch.actions.mouse_movements),
        "sample_keys": [row["sample_key"] for row in metadata_rows],
        "source_start_frames": [row["source_start_frame"] for row in metadata_rows],
    }


def assess_decoded_video_parity(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    max_mean_abs: float = 0.0,
    max_abs: int = 0,
    max_fraction_different: float = 0.0,
) -> dict[str, Any]:
    """Compare model-facing decoded pixels against preregistered tolerances.

    The comparison is deliberately separate from throughput: an alternative decoder is not eligible
    for training merely because it is faster.  Both tensors must have the same shape and dtype, and
    all three declared difference limits must hold.
    """
    if reference.shape != candidate.shape:
        return {
            "status": "fail",
            "error": f"shape mismatch: reference={tuple(reference.shape)}, candidate={tuple(candidate.shape)}",
        }
    if reference.dtype != candidate.dtype:
        return {
            "status": "fail",
            "error": f"dtype mismatch: reference={reference.dtype}, candidate={candidate.dtype}",
        }
    if reference.dtype != torch.uint8:
        return {
            "status": "fail",
            "error": f"expected uint8 model-facing video, got {reference.dtype}",
        }

    reference_cpu = reference.detach().cpu().contiguous()
    candidate_cpu = candidate.detach().cpu().contiguous()
    difference = (reference_cpu.to(torch.int16) - candidate_cpu.to(torch.int16)).abs()
    mean_abs = float(difference.float().mean())
    observed_max = int(difference.max())
    fraction_different = float((difference != 0).float().mean())
    thresholds = {
        "max_mean_abs": max_mean_abs,
        "max_abs": max_abs,
        "max_fraction_different": max_fraction_different,
    }
    observed = {
        "mean_abs": mean_abs,
        "max_abs": observed_max,
        "fraction_different": fraction_different,
    }
    limits = {
        "mean_abs": max_mean_abs,
        "max_abs": max_abs,
        "fraction_different": max_fraction_different,
    }
    violations = [name for name, value in observed.items() if value > limits[name]]
    return {
        "status": "pass" if not violations else "fail",
        "shape": list(reference_cpu.shape),
        "dtype": str(reference_cpu.dtype),
        "reference_sha256": _tensor_sha256(reference_cpu),
        "candidate_sha256": _tensor_sha256(candidate_cpu),
        "observed": observed,
        "thresholds": thresholds,
        "violations": violations,
    }


def compare_torchcodec_cpu_cuda(
    data_root: str | Path,
    *,
    index_path: str | Path | None = None,
    split: str = "train",
    map_slug: str = "dust2",
    clip_len: int = 16,
    target_fps: int = 8,
    frame_size: tuple[int, int] = (168, 308),
    seed: int = 28,
    max_mean_abs: float = 0.0,
    max_abs: int = 0,
    max_fraction_different: float = 0.0,
) -> dict[str, Any]:
    """Compare CPU TorchCodec with CUDA TorchCodec on one fixed ten-POV group."""
    if not torch.cuda.is_available():
        return {
            "status": "fail",
            "error": "CUDA is unavailable; cannot evaluate the TorchCodec CUDA candidate",
        }

    root = Path(data_root)
    loader_index = Path(index_path) if index_path is not None else root
    reference_batch, metadata = next(
        iter(
            _loader(
                loader_index,
                split=split,
                map_slug=map_slug,
                group_mode="synchronized",
                clip_len=clip_len,
                target_fps=target_fps,
                frame_size=frame_size,
                num_workers=0,
                prefetch_factor=2,
                persistent_workers=False,
                pin_memory=False,
                shuffle=False,
                infinite=False,
                seed=seed,
            )
        )
    )
    validate_batch_contract(
        reference_batch,
        metadata,
        group_mode="synchronized",
        clip_len=clip_len,
    )

    from torchcodec.decoders import VideoDecoder  # pyright: ignore[reportPrivateImportUsage]

    candidate_rows: list[torch.Tensor] = []
    for item in metadata:
        video_path = root / "videos" / f"{item.sample_key}.mp4"
        if not video_path.is_file():
            video_path = root / "videos" / "360p" / f"{item.sample_key}.mp4"
        if not video_path.is_file():
            raise FileNotFoundError(f"Missing materialized CounterStrike-1K video: {video_path}")
        frames = VideoDecoder(video_path, device="cuda").get_frames_at(item.frame_indices).data
        if frame_size is not None:
            frames = _resize(frames, frame_size)
        candidate_rows.append(frames)
    candidate = torch.stack(candidate_rows)
    torch.cuda.synchronize()

    result = assess_decoded_video_parity(
        reference_batch.video,
        candidate,
        max_mean_abs=max_mean_abs,
        max_abs=max_abs,
        max_fraction_different=max_fraction_different,
    )
    result.update(
        {
            "reference_backend": "torchcodec_cpu",
            "candidate_backend": "torchcodec_cuda",
            "round_id": metadata[0].round_id,
            "sample_keys": [item.sample_key for item in metadata],
            "source_start_frame": metadata[0].source_start_frame,
        }
    )
    return result


def validate_batch_contract(
    batch: VideoActionBatch,
    metadata: list[CounterStrikeClipMeta],
    *,
    group_mode: GroupMode,
    clip_len: int,
    expected_rows: int = 10,
) -> None:
    """Fail closed when a benchmark batch no longer matches the experiment contract."""
    if tuple(batch.video.shape[:2]) != (expected_rows, clip_len):
        raise ValueError(
            f"video contract mismatch: expected ({expected_rows}, {clip_len}, ...), "
            f"got {tuple(batch.video.shape)}"
        )
    if tuple(batch.actions.key_presses.shape) != (expected_rows, clip_len, len(CS2_KEYS)):
        raise ValueError(f"key action contract mismatch: got {tuple(batch.actions.key_presses.shape)}")
    if tuple(batch.actions.mouse_movements.shape) != (expected_rows, clip_len, 2):
        raise ValueError(f"mouse action contract mismatch: got {tuple(batch.actions.mouse_movements.shape)}")
    if len(metadata) != expected_rows:
        raise ValueError(f"metadata contract mismatch: expected {expected_rows}, got {len(metadata)}")
    if {item.group_mode for item in metadata} != {group_mode}:
        raise ValueError(
            f"group-mode metadata mismatch: expected {group_mode!r}, "
            f"got {sorted({item.group_mode for item in metadata})}"
        )

    if group_mode == "synchronized":
        if len({item.round_id for item in metadata}) != 1:
            raise ValueError("synchronized batch contains more than one round")
        if {item.perspective for item in metadata} != set(range(expected_rows)):
            raise ValueError("synchronized batch does not contain POV indices 0..9 exactly once")
        if len({item.source_start_frame for item in metadata}) != 1:
            raise ValueError("synchronized batch does not share one source start")
    elif group_mode == "shuffled":
        if len({item.round_id for item in metadata}) != expected_rows:
            raise ValueError("shuffled batch must contain ten distinct rounds")
        if {item.perspective for item in metadata} != set(range(expected_rows)):
            raise ValueError("shuffled batch does not preserve POV slots 0..9")


def _loader(
    data_root: Path,
    *,
    split: str,
    map_slug: str,
    group_mode: GroupMode,
    window_mode: Literal["midpoint", "first-death"] = "midpoint",
    clip_len: int,
    target_fps: int,
    frame_size: tuple[int, int],
    num_workers: int,
    prefetch_factor: int,
    persistent_workers: bool,
    pin_memory: bool,
    shuffle: bool,
    infinite: bool,
    seed: int,
):
    n_players = 1 if group_mode == "single" else 10
    groups_per_batch = 10 if group_mode == "single" else 1
    return create_loader(
        data_root,
        dataset_backend="counterstrike1k",
        split=split,
        map_slug=map_slug,
        group_mode=group_mode,
        window_mode=window_mode,
        clip_len=clip_len,
        target_fps=target_fps,
        n_players=n_players,
        batch_size=groups_per_batch,
        num_workers=num_workers,
        shuffle=shuffle,
        infinite=infinite,
        shuffle_buffer_size=100,
        seed=seed,
        frame_size=frame_size,
        valid_keys=list(CS2_KEYS),
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )


def verify_single_synchronized_parity(
    data_root: str | Path,
    *,
    split: str = "train",
    map_slug: str = "dust2",
    clip_len: int = 16,
    target_fps: int = 8,
    frame_size: tuple[int, int] = (168, 308),
    seed: int = 28,
) -> dict[str, Any]:
    """Prove both arms decode identical rows/actions for the first fixed round.

    With shuffle disabled, the single loader's first batch contains all ten POVs from the first
    eligible round and must be byte-identical to the synchronized loader's first group.
    """
    root = Path(data_root)
    common = {
        "split": split,
        "map_slug": map_slug,
        # A shared event anchor makes this a tensor-parity test rather than an invalid
        # comparison between the single loader's POV-specific midpoint and the synchronized
        # loader's common-alive-horizon midpoint.
        "window_mode": "first-death",
        "clip_len": clip_len,
        "target_fps": target_fps,
        "frame_size": frame_size,
        "num_workers": 0,
        "prefetch_factor": 2,
        "persistent_workers": False,
        "pin_memory": False,
        "shuffle": False,
        "infinite": False,
        "seed": seed,
    }
    single_batch, single_meta = next(iter(_loader(root, group_mode="single", **common)))
    synced_batch, synced_meta = next(iter(_loader(root, group_mode="synchronized", **common)))
    validate_batch_contract(single_batch, single_meta, group_mode="single", clip_len=clip_len)
    validate_batch_contract(synced_batch, synced_meta, group_mode="synchronized", clip_len=clip_len)

    single_rows = _metadata_rows(single_meta)
    synced_rows = _metadata_rows(synced_meta)
    identity_fields = ("sample_key", "round_id", "perspective", "source_start_frame", "frame_indices")
    for field in identity_fields:
        if [row[field] for row in single_rows] != [row[field] for row in synced_rows]:
            raise ValueError(f"single/synchronized metadata parity failed for {field}")
    if not torch.equal(single_batch.video, synced_batch.video):
        raise ValueError("single/synchronized decoded video tensors differ")
    if not torch.equal(single_batch.actions.key_presses, synced_batch.actions.key_presses):
        raise ValueError("single/synchronized key tensors differ")
    if not torch.equal(single_batch.actions.mouse_movements, synced_batch.actions.mouse_movements):
        raise ValueError("single/synchronized mouse tensors differ")
    if not (
        torch.isnan(single_batch.actions.game_mouse_sensitivity).all()
        and torch.isnan(synced_batch.actions.game_mouse_sensitivity).all()
    ):
        raise ValueError("CounterStrike sensitivity sentinel must remain NaN in both arms")

    return {
        "status": "pass",
        "round_id": single_meta[0].round_id,
        "sample_keys": [item.sample_key for item in single_meta],
        "source_start_frame": single_meta[0].source_start_frame,
        "window_mode": single_meta[0].window_mode,
        "video_sha256": _tensor_sha256(single_batch.video),
        "key_presses_sha256": _tensor_sha256(single_batch.actions.key_presses),
        "mouse_movements_sha256": _tensor_sha256(single_batch.actions.mouse_movements),
    }


def benchmark_case(
    data_root: str | Path,
    *,
    split: str,
    map_slug: str,
    group_mode: GroupMode,
    clip_len: int,
    target_fps: int,
    frame_size: tuple[int, int],
    num_workers: int,
    prefetch_factor: int,
    persistent_workers: bool,
    pin_memory: bool,
    shuffle: bool,
    seed: int,
    warmup_batches: int,
    timed_batches: int,
    transfer_device: str | None,
) -> dict[str, Any]:
    """Measure one real loader configuration after validating every emitted batch."""
    if timed_batches < 1:
        raise ValueError("timed_batches must be positive")
    loader = _loader(
        Path(data_root),
        split=split,
        map_slug=map_slug,
        group_mode=group_mode,
        clip_len=clip_len,
        target_fps=target_fps,
        frame_size=frame_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
        infinite=True,
        seed=seed,
    )
    iterator = iter(loader)
    device = torch.device(transfer_device) if transfer_device is not None else None
    first_signature: dict[str, Any] | None = None

    def consume() -> int:
        nonlocal first_signature
        batch, metadata = next(iterator)
        validate_batch_contract(batch, metadata, group_mode=group_mode, clip_len=clip_len)
        if first_signature is None:
            first_signature = batch_signature(batch, metadata)
        if device is not None:
            batch = batch.to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        return int(batch.video.shape[0] * batch.video.shape[1])

    for _ in range(warmup_batches):
        consume()

    started = time.perf_counter()
    frames = sum(consume() for _ in range(timed_batches))
    elapsed = time.perf_counter() - started
    result = {
        "status": "pass",
        "group_mode": group_mode,
        "num_workers": num_workers,
        "prefetch_factor": prefetch_factor if num_workers > 0 else None,
        "persistent_workers": persistent_workers,
        "pin_memory": pin_memory,
        "shuffle": shuffle,
        "warmup_batches": warmup_batches,
        "timed_batches": timed_batches,
        "rows_per_batch": 10,
        "frames_per_batch": 10 * clip_len,
        "elapsed_seconds": elapsed,
        "batches_per_second": timed_batches / elapsed,
        "rows_per_second": timed_batches * 10 / elapsed,
        "input_frames_per_second": frames / elapsed,
        "transfer_device": str(device) if device is not None else None,
        "first_batch": first_signature,
    }

    return result


def _git_value(args: list[str], cwd: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _command_output(args: list[str]) -> str | None:
    try:
        return subprocess.run(
            args,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_provenance(
    data_root: Path,
    *,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Capture enough environment and source state to audit a benchmark result."""
    project_root = Path(__file__).resolve().parents[3]
    manifest = manifest_path or (
        data_root / "manifest_dust2.parquet"
        if (data_root / "manifest_dust2.parquet").is_file()
        else data_root / "manifest.parquet"
    )
    git_status = _git_value(["status", "--porcelain=v1"], project_root)
    download_manifest = data_root / "hf_download_manifest.tsv"
    return {
        "generated_at_utc": _utc_now(),
        "command": shlex.join(sys.argv),
        "cwd": str(Path.cwd()),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "cpu_count": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
        "torch": torch.__version__,
        "torchcodec": _package_version("torchcodec"),
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "lscpu_json": _command_output(["lscpu", "--json"]),
        "memory_bytes": _command_output(["awk", "/MemTotal/ {print $2 * 1024}", "/proc/meminfo"]),
        "data_mount": _command_output(["findmnt", "--json", "--target", str(data_root)]),
        "block_devices": _command_output(["lsblk", "--json", "--bytes", "--output", "NAME,SIZE,TYPE"]),
        "nvidia_smi": _command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "ffmpeg_version": _command_output(["ffmpeg", "-version"]),
        "git_commit": _git_value(["rev-parse", "HEAD"], project_root),
        "git_status": git_status,
        "git_clean": git_status == "",
        "manifest_path": str(manifest),
        "manifest_sha256": _sha256_file(manifest),
        "hf_download_manifest_path": str(download_manifest) if download_manifest.is_file() else None,
        "hf_download_manifest_sha256": (
            _sha256_file(download_manifest) if download_manifest.is_file() else None
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        help=(
            "Explicit model-facing manifest. It must live directly inside --data-root. "
            "Publication benchmarks should pin the confirmatory manifest rather than rely "
            "on the default filename lookup."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--map-slug", default="dust2")
    parser.add_argument(
        "--group-modes",
        nargs="+",
        choices=["single", "synchronized", "shuffled"],
        default=["single", "synchronized"],
    )
    parser.add_argument("--workers", type=int, nargs="+", default=[0, 4, 8, 12])
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=torch.cuda.is_available(),
    )
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--clip-len", type=int, default=16)
    parser.add_argument("--target-fps", type=int, default=8)
    parser.add_argument("--frame-height", type=int, default=168)
    parser.add_argument("--frame-width", type=int, default=308)
    parser.add_argument("--seed", type=int, default=28)
    parser.add_argument("--warmup-batches", type=int, default=10)
    parser.add_argument("--timed-batches", type=int, default=100)
    parser.add_argument("--transfer-device", default=None)
    parser.add_argument(
        "--compare-torchcodec-cuda",
        action="store_true",
        help="Fail closed unless CUDA TorchCodec matches CPU TorchCodec on a fixed ten-POV group.",
    )
    parser.add_argument("--cuda-parity-max-mean-abs", type=float, default=0.0)
    parser.add_argument("--cuda-parity-max-abs", type=int, default=0)
    parser.add_argument("--cuda-parity-max-fraction-different", type=float, default=0.0)
    parser.add_argument(
        "--skip-parity-check",
        action="store_true",
        help="Skip the fixed first-round single-vs-synchronized tensor equality gate.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if any(workers < 0 for workers in args.workers):
        raise ValueError("worker counts must be non-negative")
    if args.prefetch_factor < 1:
        raise ValueError("prefetch_factor must be positive")

    data_root = args.data_root.resolve()
    manifest_path = args.manifest.resolve() if args.manifest is not None else None
    if manifest_path is not None:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Explicit benchmark manifest not found: {manifest_path}")
        if manifest_path.parent != data_root:
            raise ValueError("--manifest must live directly inside --data-root")
    index_path = manifest_path or data_root
    frame_size = (args.frame_height, args.frame_width)
    payload: dict[str, Any] = {
        "schema": "mira-cs2-dataloader-benchmark-v1",
        "provenance": collect_provenance(data_root, manifest_path=manifest_path),
        "config": {
            "data_root": str(data_root),
            "manifest_path": str(manifest_path) if manifest_path is not None else None,
            "split": args.split,
            "map_slug": args.map_slug,
            "group_modes": args.group_modes,
            "workers": args.workers,
            "prefetch_factor": args.prefetch_factor,
            "persistent_workers": args.persistent_workers,
            "pin_memory": args.pin_memory,
            "shuffle": args.shuffle,
            "clip_len": args.clip_len,
            "target_fps": args.target_fps,
            "frame_size": list(frame_size),
            "seed": args.seed,
            "warmup_batches": args.warmup_batches,
            "timed_batches": args.timed_batches,
            "transfer_device": args.transfer_device,
            "compare_torchcodec_cuda": args.compare_torchcodec_cuda,
            "cuda_parity_max_mean_abs": args.cuda_parity_max_mean_abs,
            "cuda_parity_max_abs": args.cuda_parity_max_abs,
            "cuda_parity_max_fraction_different": args.cuda_parity_max_fraction_different,
        },
        "parity": None,
        "decoder_parity": None,
        "results": [],
    }
    failures = 0
    if not args.skip_parity_check:
        try:
            payload["parity"] = verify_single_synchronized_parity(
                index_path,
                split=args.split,
                map_slug=args.map_slug,
                clip_len=args.clip_len,
                target_fps=args.target_fps,
                frame_size=frame_size,
                seed=args.seed,
            )
        except Exception as exc:  # noqa: BLE001 - preserve failed benchmark evidence
            failures += 1
            payload["parity"] = {
                "status": "fail",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    if args.compare_torchcodec_cuda:
        try:
            payload["decoder_parity"] = compare_torchcodec_cpu_cuda(
                data_root,
                index_path=index_path,
                split=args.split,
                map_slug=args.map_slug,
                clip_len=args.clip_len,
                target_fps=args.target_fps,
                frame_size=frame_size,
                seed=args.seed,
                max_mean_abs=args.cuda_parity_max_mean_abs,
                max_abs=args.cuda_parity_max_abs,
                max_fraction_different=args.cuda_parity_max_fraction_different,
            )
            if payload["decoder_parity"]["status"] != "pass":
                failures += 1
        except Exception as exc:  # noqa: BLE001 - preserve failed candidate evidence
            failures += 1
            payload["decoder_parity"] = {
                "status": "fail",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    for group_mode in args.group_modes:
        for workers in args.workers:
            persistent = bool(args.persistent_workers and workers > 0)
            try:
                result = benchmark_case(
                    index_path,
                    split=args.split,
                    map_slug=args.map_slug,
                    group_mode=group_mode,
                    clip_len=args.clip_len,
                    target_fps=args.target_fps,
                    frame_size=frame_size,
                    num_workers=workers,
                    prefetch_factor=args.prefetch_factor,
                    persistent_workers=persistent,
                    pin_memory=args.pin_memory,
                    shuffle=args.shuffle,
                    seed=args.seed,
                    warmup_batches=args.warmup_batches,
                    timed_batches=args.timed_batches,
                    transfer_device=args.transfer_device,
                )
            except Exception as exc:  # noqa: BLE001 - preserve failed benchmark evidence
                failures += 1
                result = {
                    "status": "fail",
                    "group_mode": group_mode,
                    "num_workers": workers,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            payload["results"].append(result)
            print(json.dumps(result, sort_keys=True), flush=True)

    payload["completed_at_utc"] = _utc_now()
    payload["status"] = "pass" if failures == 0 else "fail"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(f"wrote {args.output}", flush=True)
    return 0 if failures == 0 else 1
