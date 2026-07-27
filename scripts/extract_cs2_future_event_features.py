#!/usr/bin/env python
"""Extract frozen MIRA context representations for leak-free future-event probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from mira.data.training_loader import create_loader
from mira.evaluation.future_event_features import extract_context_features
from mira.inference.loading import load_world_model
from mira.training.checkpoints import resolve_checkpoint
from mira.training.reproducibility import seed_everything
from mira.world_model.latent_world_model import LatentWorldModel
from mira.world_model.multi_wrapper_world_model import MultiWrapperWorldModel


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path):
    columns = [
        "sample_key",
        "match_id",
        "round_id",
        "pov_idx",
        "split",
        "map_slug",
        "round_idx",
        "frames",
        "frame0_tick",
        "frame_tick_stride",
        "fps",
    ]
    available = set(pq.read_schema(path).names)
    return pq.read_table(path, columns=[column for column in columns if column in available]).to_pandas()


def git_output(project_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(project_root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def window_row(meta, source, *, split: str, context_frames: int, target_fps: int) -> dict:
    source_fps = round(float(source["fps"]))
    if source_fps % target_fps:
        raise ValueError(f"source fps {source_fps} is not divisible by target fps {target_fps}")
    source_stride = source_fps // target_fps
    start_frame = int(meta.source_start_frame)
    end_frame = start_frame + context_frames * source_stride
    frame_tick_stride = int(source["frame_tick_stride"])
    frame0_tick = int(source["frame0_tick"])
    start_tick = frame0_tick + start_frame * frame_tick_stride
    end_tick = frame0_tick + end_frame * frame_tick_stride
    return {
        "eval_window_id": f"{meta.round_id}__mira_midpoint",
        "round_id": str(meta.round_id),
        "match_id": str(meta.match_id),
        "round_idx": int(source.get("round_idx", 0)),
        "window_idx": 0,
        "sample_key": str(meta.sample_key),
        "pov_idx": int(meta.perspective),
        "split": split,
        "map_slug": str(source["map_slug"]),
        "team_side": "",
        "phase_bucket": "midpoint",
        "start_tick": start_tick,
        "end_tick": end_tick,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "window_frames": context_frames * source_stride,
        "fps": float(source_fps),
        "frame_tick_stride": frame_tick_stride,
        "alive_only": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--codec-checkpoint",
        type=Path,
        help=(
            "Optional immutable codec relocation. This replaces only the saved absolute codec path "
            "while preserving the world-model checkpoint and weights."
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--context-frames", type=int, default=8)
    parser.add_argument("--future-frames", type=int, default=8)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--max-windows-per-split",
        type=int,
        default=None,
        help="Optional deterministic smoke-test cap; publication extraction leaves this unset.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--autocast-dtype",
        choices=["float32", "bfloat16"],
        default="bfloat16",
    )
    args = parser.parse_args()

    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {args.out}")
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    if args.codec_checkpoint is not None and not args.codec_checkpoint.is_file():
        raise FileNotFoundError(args.codec_checkpoint)
    if args.manifest.resolve().parent != args.dataset_root.resolve():
        raise ValueError("--manifest must live directly inside --dataset-root")
    if args.context_frames <= 0 or args.future_frames <= 0:
        raise ValueError("context and future frame counts must be positive")
    project_root = Path(__file__).resolve().parents[1]
    code_status = git_output(project_root, "status", "--porcelain=v1")
    if code_status:
        raise RuntimeError("publication feature extraction requires a clean MIRA source tree")
    code_commit = git_output(project_root, "rev-parse", "HEAD")

    seed_everything(args.seed, deterministic=True)
    checkpoint = resolve_checkpoint(args.checkpoint).resolve()
    device = torch.device(args.device)
    model, _ = load_world_model(
        checkpoint,
        device=device,
        codec_checkpoint=args.codec_checkpoint,
    )
    if not isinstance(model, (LatentWorldModel, MultiWrapperWorldModel)):
        raise TypeError(f"unsupported checkpoint model type: {type(model)!r}")
    swm = model.single_world_model if isinstance(model, MultiWrapperWorldModel) else model
    total_frames = args.context_frames + args.future_frames
    if total_frames > swm.config.video.timesteps:
        raise ValueError(
            f"requested {total_frames} frames exceeds checkpoint window "
            f"{swm.config.video.timesteps}"
        )
    single_height = int(swm.codec.config.encoder.video.height)
    single_width = int(swm.codec.config.encoder.video.width)
    manifest = read_manifest(args.manifest)
    by_sample = {
        str(row["sample_key"]): row
        for row in manifest.to_dict(orient="records")
    }
    autocast_dtype = torch.bfloat16 if args.autocast_dtype == "bfloat16" else None

    feature_rows: list[np.ndarray] = []
    index_rows: list[dict] = []
    split_counts: dict[str, int] = {}
    for split in args.splits:
        loader = create_loader(
            args.manifest,
            dataset_backend="counterstrike1k",
            split=split,
            map_slug="dust2",
            group_mode="synchronized",
            window_mode="midpoint",
            clip_len=total_frames,
            target_fps=int(swm.config.video.fps),
            n_players=10,
            batch_size=1,
            num_workers=args.num_workers,
            shuffle=False,
            infinite=False,
            shuffle_buffer_size=1,
            seed=args.seed,
            frame_size=(single_height, single_width),
            action_config=swm.config.actions,
            prefetch_factor=args.prefetch_factor,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )
        windows = 0
        for batch, metadata in loader:
            if (
                args.max_windows_per_split is not None
                and windows >= args.max_windows_per_split
            ):
                break
            if len(metadata) != 10:
                raise ValueError(f"{split}: expected one ten-POV group, got {len(metadata)} rows")
            features = extract_context_features(
                model,
                batch,
                context_frames=args.context_frames,
                autocast_dtype=autocast_dtype,
            ).numpy()
            if features.shape[0] != 10:
                raise ValueError(f"{split}: expected ten features, got {features.shape}")
            starts = {int(meta.source_start_frame) for meta in metadata}
            rounds = {str(meta.round_id) for meta in metadata}
            if len(starts) != 1 or len(rounds) != 1:
                raise ValueError(f"{split}: loader emitted a non-synchronized group")
            for feature, meta in zip(features, metadata, strict=True):
                source = by_sample.get(str(meta.sample_key))
                if source is None:
                    raise KeyError(f"sample absent from frozen manifest: {meta.sample_key}")
                index_rows.append(
                    window_row(
                        meta,
                        source,
                        split=split,
                        context_frames=args.context_frames,
                        target_fps=int(swm.config.video.fps),
                    )
                )
                feature_rows.append(feature.astype(np.float32))
            windows += 1
        split_counts[split] = windows

    if not feature_rows:
        raise RuntimeError("feature extraction produced no rows")
    import pandas as pd

    index = pd.DataFrame(index_rows)
    embeddings = np.stack(feature_rows).astype(np.float32)
    args.out.mkdir(parents=True, exist_ok=True)
    index.to_parquet(args.out / "embedding_index.parquet", index=False)
    np.savez_compressed(
        args.out / "embeddings.npz",
        embeddings=embeddings,
        sample_key=index["sample_key"].astype(str).to_numpy(),
        eval_window_id=index["eval_window_id"].astype(str).to_numpy(),
        pov_idx=index["pov_idx"].to_numpy(dtype=np.int16),
        embedding_source_row_id=np.arange(len(index), dtype=np.int64),
    )
    metadata = {
        "schema": "mira-cs2-causal-context-features-v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "dataset_root": str(args.dataset_root.resolve()),
        "code_commit": code_commit,
        "code_status": code_status,
        "model_type": type(model).__name__,
        "n_players": int(model.n_players) if isinstance(model, MultiWrapperWorldModel) else 1,
        "feature_definition": "final causal diffusion-transformer block; tau=1; last context latent; spatial mean per POV",
        "causal_interval": f"video/actions [0,{args.context_frames}); later rows never passed to model",
        "context_frames": args.context_frames,
        "future_frames_reserved_for_labels": args.future_frames,
        "target_fps": int(swm.config.video.fps),
        "feature_rows": len(index),
        "feature_dim": int(embeddings.shape[1]),
        "split_windows": split_counts,
        "max_windows_per_split": args.max_windows_per_split,
        "seed": args.seed,
        "autocast_dtype": args.autocast_dtype,
        "strict_deterministic": True,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "embedding_index_sha256": sha256_file(args.out / "embedding_index.parquet"),
        "embeddings_sha256": sha256_file(args.out / "embeddings.npz"),
    }
    (args.out / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
