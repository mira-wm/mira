#!/usr/bin/env python
"""Export paired single-MIRA rollouts for common RAFT and ARR scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any

# Deterministic CUDA matrix multiplication requires this to be set before
# importing torch. Preserve an explicit caller choice of either valid workspace.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from omegaconf import OmegaConf

from mira.data.batch import VideoActionBatch
from mira.data.training_loader import create_loader
from mira.evaluation.rollout_archive import RolloutArchiveWriter, sha256_file
from mira.world_model.config import WorldModelInferenceConfig

EXPECTED_MANIFEST_SHA256 = "33abbb623072932431871a612620110c473d4b664c52010e5763c273c6daf10e"
ACTION_MODES = ("true", "shuffled", "zeros")
ROLLOUT_STEPS = 8
CONTEXT_FRAMES = 8


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _sha256_json(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _alive_end_frames(manifest: Path) -> dict[str, int]:
    import pyarrow.parquet as pq  # noqa: PLC0415

    table = pq.read_table(manifest, columns=["sample_key", "alive_end_frame"])
    return {
        str(sample_key): int(alive_end)
        for sample_key, alive_end in zip(
            table["sample_key"].to_pylist(),
            table["alive_end_frame"].to_pylist(),
            strict=True,
        )
    }


def _zero_actions(actions):
    result = actions.clone()
    result.key_presses.zero_()
    result.mouse_movements.zero_()
    return result


def _flat_actions(actions: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Return native MIRA order and canonical CS2 release order."""
    keys = actions.key_presses.float()
    mouse = actions.mouse_movements.float()
    native = torch.cat([keys, mouse], dim=-1)  # keys, delta_yaw, delta_pitch
    canonical = torch.cat([keys, mouse[..., 1:2], mouse[..., 0:1]], dim=-1)
    return native, canonical


def _write_review_videos(
    out_dir: Path,
    *,
    ground_truth: torch.Tensor,
    predictions: dict[str, torch.Tensor],
    metadata: list[Any],
    fps: int,
    count: int,
) -> None:
    from mira.training.visualization import write_video_ffmpeg  # noqa: PLC0415

    review_dir = out_dir / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    review_manifest = []
    for index in range(min(count, len(ground_truth))):
        columns = [ground_truth[index]]
        columns.extend(predictions[mode][index] for mode in ACTION_MODES)
        canvas = torch.cat(columns, dim=-1).float().clamp(0, 1).mul(255).round().byte()
        path = review_dir / f"{index:04d}_{metadata[index].sample_key}.mp4"
        write_video_ffmpeg(path, canvas, fps=fps)
        review_manifest.append(
            {
                "path": str(path.relative_to(out_dir)),
                "sample_key": metadata[index].sample_key,
                "round_id": metadata[index].round_id,
                "pov_idx": metadata[index].perspective,
                "columns": ["ground_truth", *ACTION_MODES],
            }
        )
    (out_dir / "review_manifest.json").write_text(
        json.dumps(review_manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[37])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--n-diffusion-steps", type=int, default=10)
    parser.add_argument("--review-videos", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.out_dir.exists():
        raise FileExistsError(f"refusing to overwrite rollout export: {args.out_dir}")
    if sha256_file(args.manifest) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("confirmatory manifest SHA-256 drifted")
    if len(set(args.eval_seeds)) != len(args.eval_seeds) or not args.eval_seeds:
        raise ValueError("evaluation seeds must be non-empty and unique")
    from mira.inference.loading import load_world_model  # noqa: PLC0415
    from mira.training.checkpoints import resolve_checkpoint  # noqa: PLC0415
    from mira.training.reproducibility import seed_everything  # noqa: PLC0415

    checkpoint = resolve_checkpoint(args.checkpoint).resolve()
    config_path = checkpoint.parents[1] / "world_model_config.yaml"
    cfg = OmegaConf.load(config_path)
    if (
        cfg.dataset.get("backend") != "counterstrike1k"
        or cfg.dataset.get("group_mode") != "single"
        or cfg.dataset.get("train_split") != "train"
    ):
        raise ValueError("checkpoint is not a single-POV CounterStrike-1K training run")
    device = torch.device(args.device)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
    model, _ = load_world_model(checkpoint, device=device)
    model = model.eval()
    model.set_inference_context(CONTEXT_FRAMES)

    loader = create_loader(
        index_path=args.manifest,
        dataset_backend="counterstrike1k",
        split="test",
        map_slug="dust2",
        group_mode="single",
        window_mode="midpoint",
        clip_len=CONTEXT_FRAMES + ROLLOUT_STEPS,
        target_fps=model.config.video.fps,
        action_fps=model.config.actions.target_fps,
        n_players=1,
        batch_size=10,
        num_workers=args.num_workers,
        shuffle_buffer_size=100,
        prefetch_factor=2,
        persistent_workers=args.num_workers > 0,
        frame_size=(168, 308),
        valid_keys=list(model.config.actions.valid_keys),
        seed=args.eval_seeds[0],
        shuffle=False,
        exclude_replays=True,
        infinite=True,
    )
    iterator = iter(loader)
    batches = [next(iterator) for _ in range(69)]
    for batch, metadata in batches:
        if (
            len(batch) != 10
            or len({item.round_id for item in metadata}) != 1
            or [item.perspective for item in metadata] != list(range(10))
        ):
            raise ValueError("export loader did not emit one complete POV-ordered round")
    sample_metadata = [item for _batch, metadata in batches for item in metadata]
    if len(sample_metadata) != 690 or len({item.round_id for item in sample_metadata}) != 69:
        raise ValueError("export loader did not cover all 69 test rounds / 690 POV rows")
    alive_end = _alive_end_frames(args.manifest)
    sample_plan = [
        {
            "sample_position": index,
            "dataset_index": index,
            "sample_key": item.sample_key,
            "match_id": item.match_id,
            "round_id": item.round_id,
            "pov_idx": item.perspective,
            "source_frame_indices": item.frame_indices,
            "alive_end_frame": alive_end[item.sample_key],
            "action_donor_sample_key": sample_metadata[((index // 10 + 1) % 69) * 10 + index % 10].sample_key,
            "action_donor_round_id": sample_metadata[((index // 10 + 1) % 69) * 10].round_id,
        }
        for index, item in enumerate(sample_metadata)
    ]
    args.out_dir.mkdir(parents=True)
    sample_plan_path = args.out_dir / "sample_plan.json"
    sample_plan_path.write_text(json.dumps(sample_plan, indent=2) + "\n", encoding="utf-8")

    writer = RolloutArchiveWriter(
        args.out_dir / "rollout_archive",
        num_samples=690,
        eval_seeds=args.eval_seeds,
        action_modes=list(ACTION_MODES),
        rollout_steps=ROLLOUT_STEPS,
        height=168,
        width=308,
        num_model_actions=14,
        num_cs2_actions=14,
    )
    inference = WorldModelInferenceConfig(
        n_diffusion_steps=args.n_diffusion_steps,
        noise_level=0.0,
        schedule_type="linear",
    )
    first_review_predictions: dict[str, torch.Tensor] = {}
    first_review_ground_truth = None
    first_review_metadata = None
    for seed_index, eval_seed in enumerate(args.eval_seeds):
        for batch_index, (receiver, metadata) in enumerate(batches):
            donor = batches[(batch_index + 1) % len(batches)][0]
            actions_by_mode = {
                "true": receiver.actions,
                "shuffled": donor.actions,
                "zeros": _zero_actions(receiver.actions),
            }
            positions = list(range(batch_index * 10, batch_index * 10 + 10))
            valid = np.asarray(
                [
                    [
                        int(frame) < alive_end[item.sample_key]
                        for frame in item.frame_indices[CONTEXT_FRAMES : CONTEXT_FRAMES + ROLLOUT_STEPS]
                    ]
                    for item in metadata
                ],
                dtype=np.bool_,
            )
            for mode_index, mode in enumerate(ACTION_MODES):
                mode_batch = VideoActionBatch(
                    video=receiver.video.clone(),
                    actions=actions_by_mode[mode].clone(),
                )
                paired_seed = eval_seed * 1_000_003 + batch_index * 101
                seed_everything(paired_seed, deterministic=True)
                with torch.no_grad(), _autocast(device):
                    outputs = model.inference(
                        mode_batch,
                        config=inference,
                        progress_bar=False,
                    )
                predicted = outputs.output_video[:, CONTEXT_FRAMES : CONTEXT_FRAMES + ROLLOUT_STEPS]
                ground_truth = outputs.preprocessed_batch.video[
                    :, CONTEXT_FRAMES : CONTEXT_FRAMES + ROLLOUT_STEPS
                ]
                context_last = outputs.preprocessed_batch.video[:, CONTEXT_FRAMES - 1]
                action_slice = outputs.preprocessed_batch.actions.slice_time(
                    CONTEXT_FRAMES - 1,
                    CONTEXT_FRAMES - 1 + ROLLOUT_STEPS,
                )
                native_actions, canonical_actions = _flat_actions(action_slice)
                writer.write_batch(
                    seed_index=seed_index,
                    mode_index=mode_index,
                    sample_positions=positions,
                    predictions=predicted,
                    ground_truth=ground_truth,
                    context_last=context_last,
                    conditioning_actions_model=native_actions,
                    conditioning_actions_cs2=canonical_actions,
                    valid_steps=valid,
                )
                if seed_index == 0 and batch_index == 0:
                    first_review_predictions[mode] = predicted.detach().cpu()
                    first_review_ground_truth = ground_truth.detach().cpu()
                    first_review_metadata = metadata

    checkpoint_sha256 = sha256_file(checkpoint)
    writer.finalize(
        contract={
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_step": 15_000,
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "split": "test",
            "map_slug": "dust2",
            "window_mode": "midpoint",
            "target_fps": 8,
            "sample_plan_sha256": sha256_file(sample_plan_path),
            "action_plan_sha256": _sha256_json(
                [
                    {
                        "target": item["sample_key"],
                        "donor": item["action_donor_sample_key"],
                    }
                    for item in sample_plan
                ]
            ),
            "transition_action_indices": "7..14 condition predicted frames 8..15",
            "model_action_schema": [
                *list(model.config.actions.valid_keys),
                "delta_yaw",
                "delta_pitch",
            ],
            "cs2_action_schema": [
                *list(model.config.actions.valid_keys),
                "delta_pitch",
                "delta_yaw",
            ],
            "alive_mask": "source_frame < alive_end_frame",
            "inference": inference.model_dump(),
        }
    )
    if (
        first_review_ground_truth is None
        or first_review_metadata is None
        or set(first_review_predictions) != set(ACTION_MODES)
    ):
        raise ValueError("review rollout capture is incomplete")
    _write_review_videos(
        args.out_dir,
        ground_truth=first_review_ground_truth,
        predictions=first_review_predictions,
        metadata=first_review_metadata,
        fps=8,
        count=args.review_videos,
    )
    summary = {
        "schema": "mira-cs2-single-rollout-archive-v1",
        "status": "complete",
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_step": 15_000,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "window_mode": "midpoint",
        "eval_seeds": args.eval_seeds,
        "action_modes": list(ACTION_MODES),
        "rounds": 69,
        "pov_rows": 690,
        "rollout_steps": ROLLOUT_STEPS,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
