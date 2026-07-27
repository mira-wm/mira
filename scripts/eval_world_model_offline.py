"""Run the world-model training-time evals offline from a saved checkpoint.

This reproduces, standalone, the two evaluations that ``train_world_model.py`` runs periodically:

  1. Validation loss   (mirrors ``run_validation``): diffusion loss on the test set.
  2. World-model metrics (mirrors the metrics loop): DINO / latent drift + Frechet DINO/Inception
     curves + per-frame PSNR/LPIPS/SSIM, with optional rollout viz clips saved to disk.

The dataset / model config is read from the ``world_model_config.yaml`` saved alongside the
checkpoint, so the eval matches the training run; how the metrics eval is run comes from
``configs/eval_world_model.yaml``. Single-process (no torchrun); video-only (no audio metrics).

The checkpoint may be a local path or a W&B run -- anything ``resolve_checkpoint`` accepts. Usage::

    python scripts/eval_world_model_offline.py /path/to/checkpoint-1000/checkpoint.pth
    python scripts/eval_world_model_offline.py <wandb-url> --num-samples 512 --skip-validation
    python scripts/eval_world_model_offline.py <ckpt> --viz 8 --output-dir out --no-compile
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import time
from collections import defaultdict
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

# This must be set before importing Torch for strict deterministic CUDA matmul.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from mira.data.batch import VideoActionBatch
from mira.inference.rollout import measure_rollout_speed
from mira.training.metrics.distributed_metric import DistributedMetric
from mira.training.metrics.world_model_metrics import WorldModelMetricsConfig
from mira.world_model.config import WorldModelInferenceConfig

if TYPE_CHECKING:
    # Single- and multi-player models are duck-typed under LatentWorldModel (the wrapper exposes the
    # same forward/inference/visualize/decode_to_video surface); n_players is read with getattr.
    from mira.world_model.latent_world_model import LatentWorldModel

logger = logging.getLogger(__name__)

CONFIG_FILENAME = "world_model_config.yaml"
EVAL_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "eval_world_model.yaml"
SPEED_BENCH_FRAMES = 32
ActionMode = Literal[
    "true",
    "zero",
    "batch-shifted",
    "round-shifted",
    "time-shifted",
]


def _autocast(device: torch.device | int | str):
    """bfloat16 autocast on CUDA, a no-op elsewhere (so the eval pieces run on CPU too)."""
    from contextlib import nullcontext

    if torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def load_run_config(checkpoint_path: Path):
    """Load the training run's config saved alongside the checkpoint (two dirs above the .pth)."""
    from omegaconf import OmegaConf  # noqa: PLC0415

    config_path = checkpoint_path.parents[1] / CONFIG_FILENAME
    if not config_path.exists():
        raise FileNotFoundError(
            f"Could not find {CONFIG_FILENAME} at {config_path}. It is expected two directories "
            "above the checkpoint (the training output dir)."
        )
    return OmegaConf.load(config_path)


def load_eval_metrics_config(
    *,
    num_samples: int | None = None,
    no_compile: bool = False,
    inference: WorldModelInferenceConfig | None = None,
) -> WorldModelMetricsConfig:
    """Build the eval :class:`WorldModelMetricsConfig` from ``configs/eval_world_model.yaml``.

    CLI overrides (sample count, compile, inference knobs) are applied on top.
    """
    from hydra.utils import instantiate  # noqa: PLC0415
    from omegaconf import OmegaConf  # noqa: PLC0415

    raw = OmegaConf.load(EVAL_CONFIG)
    config: WorldModelMetricsConfig = instantiate(raw.world_model_metrics)
    if num_samples is not None:
        config.num_samples = num_samples
    if no_compile:
        config.compile = False
    if inference is not None:
        config.inference = inference
    return config


def run_validation_loss(
    model: "LatentWorldModel",
    val_iter: Iterator[tuple[VideoActionBatch, Any]],
    device: torch.device | int | str,
    n_batches: int,
    *,
    per_batch_records: list[dict[str, Any]] | None = None,
) -> dict[str, float]:
    """Mirror ``train_world_model.run_validation``: average the forward diffusion loss on the test set."""
    import tqdm  # noqa: PLC0415

    model.eval()
    metric_trackers: dict[str, DistributedMetric] = defaultdict(lambda: DistributedMetric(device=device))
    t1 = time.time()
    for batch_index in tqdm.trange(max(1, n_batches), desc="Validation loss"):
        batch, metadata = next(val_iter)
        batch = batch.to(device)
        with torch.no_grad(), _autocast(device):
            batch_metrics = model(batch)
            for k, v in batch_metrics.items():
                metric_trackers[k].update(v)
        if per_batch_records is not None:
            round_ids = {str(getattr(item, "round_id", "")) for item in metadata}
            if len(round_ids) != 1 or "" in round_ids:
                raise ValueError("per-batch CS2 records require exactly one non-empty round_id")
            donor_round_ids = sorted(
                {
                    str(getattr(item, "action_donor_round_id", ""))
                    for item in metadata
                    if getattr(item, "action_donor_round_id", None) is not None
                }
            )
            per_batch_records.append(
                {
                    "batch_index": batch_index,
                    "round_id": next(iter(round_ids)),
                    "sample_keys": [str(getattr(item, "sample_key", "")) for item in metadata],
                    "pov_indices": [int(getattr(item, "perspective")) for item in metadata],
                    "action_donor_round_ids": donor_round_ids,
                    "action_donor_sample_keys": [
                        str(getattr(item, "action_donor_sample_key"))
                        for item in metadata
                        if getattr(item, "action_donor_sample_key", None) is not None
                    ],
                    "metrics": {
                        key: float(value.detach().float().item()) for key, value in batch_metrics.items()
                    },
                }
            )

    metrics = {k: tracker.compute_and_reset().item() for k, tracker in metric_trackers.items()}
    logger.info("Validation loss took %.1fs over %d batches", time.time() - t1, max(1, n_batches))
    return metrics


def save_viz_videos(
    model: "LatentWorldModel",
    inference_outputs,
    metadata: list,
    output_dir: Path,
    sample_idx: int,
) -> None:
    """Render one rollout to disk as a predicted-over-ground-truth grid (video-only, no audio)."""
    from mira.training.visualization import (  # noqa: PLC0415
        draw_text_on_first_frame,
        videos_to_grid,
        write_video_ffmpeg,
    )

    n_players = getattr(model, "n_players", 1)
    captions = [f"{m.match_id}:{m.perspective}" for m in metadata]
    grouped = [" + ".join(captions[i : i + n_players]) for i in range(0, len(captions), n_players)]

    viz_video = model.visualize(inference_outputs)["viz_video"]
    viz_video = draw_text_on_first_frame(viz_video, grouped)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_video_ffmpeg(
        output_dir / f"viz_{sample_idx:03d}.mp4",
        videos_to_grid(viz_video.cpu()),
        fps=model.config.video.fps,
    )


def run_world_model_metrics(
    model: "LatentWorldModel",
    metrics_iter: Iterator[tuple[VideoActionBatch, list]],
    device: torch.device | int | str,
    *,
    wm_metrics_config: WorldModelMetricsConfig,
    num_eval_batches: int,
    num_viz: int = 0,
    output_dir: Path | None = None,
    compile_models: bool = False,
) -> dict[str, float]:
    """Mirror the training metrics loop: unroll, score, optionally save viz, then aggregate."""
    import tqdm  # noqa: PLC0415

    from mira.training.metrics.world_model_metrics import WorldModelMetrics  # noqa: PLC0415

    if compile_models and not getattr(model, "_offline_eval_compiled", False):
        model.world_model.compile()
        model.decode_to_video = torch.compile(model.decode_to_video)
        # Guard so the sweep doesn't re-wrap decode on every call; setattr sidesteps nn.Module's
        # typed __setattr__ (which only accepts Tensor/Module values).
        setattr(model, "_offline_eval_compiled", True)

    wm_metrics = WorldModelMetrics(config=wm_metrics_config, iter_dataloader=metrics_iter, device=device)
    model.eval()
    t_start = time.time()
    for batch_idx in tqdm.trange(num_eval_batches, desc="World model metrics"):
        inference_outputs, metadata = wm_metrics.process_batch(model)
        if batch_idx < num_viz and output_dir is not None:
            save_viz_videos(model, inference_outputs, metadata, output_dir, batch_idx)

    scalar_result, _frechet_curves = wm_metrics.compute()
    elapsed = time.time() - t_start
    logger.info("World model metrics took %.1fs over %d batches", elapsed, num_eval_batches)
    if num_viz > 0 and output_dir is not None:
        logger.info("Saved %d viz clip(s) to %s", min(num_viz, num_eval_batches), output_dir)
    return {k: float(v) for k, v in scalar_result.items()}


def measure_denoise_speed(
    model: "LatentWorldModel",
    batch: VideoActionBatch,
    config: WorldModelInferenceConfig | None = None,
    n_frames: int = SPEED_BENCH_FRAMES,
) -> dict[str, float]:
    """Time the pure denoising rollout (no decode / metrics) after a short warmup."""
    config = config or WorldModelInferenceConfig()
    with torch.no_grad(), _autocast(model.device):
        # rollout() preprocesses its input in place. Separate clones keep the warmup from dividing
        # the timed input by 255 a second time.
        measure_rollout_speed(model, batch.clone(), config, n_frames=min(4, n_frames))  # warmup
        return measure_rollout_speed(model, batch.clone(), config, n_frames=n_frames)


def _frame_size(cfg) -> tuple[int, int] | None:
    fs = cfg.dataset.get("frame_size")
    return tuple(fs) if fs is not None else None  # type: ignore[return-value]


def _apply_action_mode(batch: VideoActionBatch, mode: ActionMode) -> VideoActionBatch:
    """Return a batch with a deterministic action intervention and unchanged video."""
    if mode == "true":
        return batch

    actions = batch.actions.clone()
    if mode == "zero":
        actions.key_presses.zero_()
        actions.mouse_movements.zero_()
    elif mode == "batch-shifted":
        if len(batch) < 2:
            raise ValueError("batch-shifted actions require at least two raw POV rows per batch")
        actions.key_presses = actions.key_presses.roll(shifts=1, dims=0)
        actions.mouse_movements = actions.mouse_movements.roll(shifts=1, dims=0)
        actions.game_mouse_sensitivity = actions.game_mouse_sensitivity.roll(shifts=1, dims=0)
    elif mode == "time-shifted":
        shift = max(1, actions.n_steps // 2)
        actions.key_presses = actions.key_presses.roll(shifts=shift, dims=1)
        actions.mouse_movements = actions.mouse_movements.roll(shifts=shift, dims=1)
    elif mode == "round-shifted":
        raise ValueError("round-shifted actions require metadata-aware _action_mode_iter")
    else:
        raise ValueError(f"Unsupported action mode {mode!r}")
    return VideoActionBatch(video=batch.video, actions=actions)


def _action_mode_iter(loader, mode: ActionMode):
    if mode == "round-shifted":
        receiver_batch, receiver_metadata = next(loader)
        for donor_batch, donor_metadata in loader:
            if len(receiver_batch) != len(donor_batch):
                raise ValueError("round-shifted receiver and donor batches differ in size")
            receiver_rounds = {str(getattr(item, "round_id", "")) for item in receiver_metadata}
            donor_rounds = {str(getattr(item, "round_id", "")) for item in donor_metadata}
            if (
                len(receiver_rounds) != 1
                or len(donor_rounds) != 1
                or "" in receiver_rounds
                or "" in donor_rounds
                or receiver_rounds == donor_rounds
            ):
                raise ValueError("round-shifted batches must contain two different complete rounds")
            receiver_povs = [int(getattr(item, "perspective")) for item in receiver_metadata]
            donor_povs = [int(getattr(item, "perspective")) for item in donor_metadata]
            if receiver_povs != donor_povs:
                raise ValueError("round-shifted donor must preserve POV-slot ordering")
            annotated_metadata = []
            for receiver, donor in zip(
                receiver_metadata,
                donor_metadata,
                strict=True,
            ):
                annotated = copy.copy(receiver)
                setattr(
                    annotated,
                    "action_donor_round_id",
                    str(getattr(donor, "round_id")),
                )
                setattr(
                    annotated,
                    "action_donor_sample_key",
                    str(getattr(donor, "sample_key")),
                )
                annotated_metadata.append(annotated)
            yield (
                VideoActionBatch(
                    video=receiver_batch.video,
                    actions=donor_batch.actions.clone(),
                ),
                annotated_metadata,
            )
            receiver_batch, receiver_metadata = donor_batch, donor_metadata
        return
    for batch, metadata in loader:
        yield _apply_action_mode(batch, mode), metadata


def _build_loader(
    cfg,
    model: "LatentWorldModel",
    *,
    split: str,
    group_mode: str | None,
    clip_len: int,
    batch_size: int,
    seed: int,
    window_mode: str = "midpoint",
):
    """Build a held-out eval loader from the checkpoint's dataset config (fixed seed, no replays)."""
    from mira.data.training_loader import create_loader  # noqa: PLC0415

    n_players = getattr(model, "n_players", 1)
    return create_loader(
        index_path=cfg.dataset.test_index,
        dataset_backend=cfg.dataset.get("backend", "rocket_science"),
        split=split,
        map_slug=cfg.dataset.get("map_slug"),
        group_mode=group_mode or cfg.dataset.get("group_mode"),
        window_mode=window_mode,
        clip_len=clip_len,
        target_fps=model.config.video.fps,
        action_fps=model.config.actions.target_fps,
        n_players=n_players,
        batch_size=batch_size,
        num_workers=cfg.dataloader.num_workers,
        shuffle_buffer_size=cfg.dataloader.shuffle_buffer_size,
        prefetch_factor=cfg.dataloader.get("prefetch_factor", 2),
        persistent_workers=cfg.dataloader.get("persistent_workers", False),
        pin_memory=cfg.dataloader.get("pin_memory"),
        frame_size=_frame_size(cfg),
        valid_keys=list(model.config.actions.valid_keys),
        seed=seed,
        shuffle=False,
        exclude_replays=True,
        infinite=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("checkpoint", type=str, help="Checkpoint path or W&B run URL.")
    parser.add_argument("--num-samples", type=int, default=None, help="World-model-metrics samples.")
    parser.add_argument("--val-n-samples", type=int, default=None, help="Validation-loss samples.")
    parser.add_argument("--skip-validation", action="store_true", help="Skip the validation-loss eval.")
    parser.add_argument("--skip-metrics", action="store_true", help="Skip the world-model-metrics eval.")
    parser.add_argument("--viz", type=int, default=0, metavar="N", help="Render N rollout viz clips.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write viz clips (default: <checkpoint dir>/offline_eval).",
    )
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile.")
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default=None,
        help="Override the held-out manifest split (default: training config test_split).",
    )
    parser.add_argument(
        "--dino-model",
        default=None,
        help="DINO metrics backbone, e.g. public dinov2_vitb14.",
    )
    parser.add_argument(
        "--group-mode",
        choices=["single", "synchronized", "shuffled"],
        default=None,
        help=(
            "Override eval grouping (e.g. evaluate a cross-round-grouped model on "
            "synchronized POVs; 'shuffled' is the backward-compatible internal key)."
        ),
    )
    parser.add_argument(
        "--window-mode",
        choices=["midpoint", "first-death"],
        default="midpoint",
        help="CounterStrike-1K held-out window selection; event modes keep all ten POVs synchronized.",
    )
    parser.add_argument(
        "--action-mode",
        choices=[
            "true",
            "zero",
            "batch-shifted",
            "round-shifted",
            "time-shifted",
        ],
        default="true",
        help="Deterministic action intervention; videos and rollout RNG stay fixed.",
    )
    parser.add_argument("--seed", type=int, default=37, help="Base deterministic eval seed.")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Require deterministic Torch kernels (also set CUBLAS_WORKSPACE_CONFIG=:4096:8).",
    )
    parser.add_argument("--n-context-frames", type=int, default=None)
    parser.add_argument("--num-unrolled-frames", type=int, default=None)
    parser.add_argument("--drift-metric-frames", type=int, default=None)
    parser.add_argument("--fdd-slice-frames", type=int, default=None)
    parser.add_argument(
        "--per-device-batch-size",
        type=int,
        default=None,
        help="Evaluation group batch size (the loader expands each group by n_players).",
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        default=None,
        help="Write scalar results and evaluation settings to this JSON file.",
    )
    parser.add_argument(
        "--per-batch-jsonl",
        type=Path,
        default=None,
        help=(
            "Write one validation-loss record per complete CS2 round. "
            "Requires each evaluation batch to contain one round."
        ),
    )
    parser.add_argument(
        "--n-diffusion-steps", type=int, default=None, help="Override rollout diffusion steps."
    )
    parser.add_argument("--schedule-type", type=str, default=None, choices=["linear", "linear_quadratic"])
    parser.add_argument(
        "--noise-level",
        type=lambda s: None if s.lower() == "none" else float(s),
        default=argparse.SUPPRESS,
        help="kv-cache noise level; 'none' merges the cache update into the last diffusion step.",
    )
    return parser.parse_args()


def _exact_num_batches(label: str, n_samples: int, batch_size: int) -> int:
    """Convert an exact requested sample count to batches without silently dropping samples."""
    if n_samples < 1:
        raise ValueError(f"{label} must be >= 1, got {n_samples}")
    if batch_size < 1:
        raise ValueError(f"{label} batch size must be >= 1, got {batch_size}")
    quotient, remainder = divmod(n_samples, batch_size)
    if remainder:
        raise ValueError(
            f"{label}={n_samples} must be divisible by batch_size={batch_size}; "
            "choose exact counts so arms evaluate the same number of examples"
        )
    return quotient


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inference_overrides(
    args_dict: dict[str, Any], base: WorldModelInferenceConfig
) -> WorldModelInferenceConfig:
    """Apply the CLI rollout knobs onto the eval config's inference defaults (only those provided)."""
    updates: dict[str, Any] = {}
    if args_dict.get("n_diffusion_steps") is not None:
        updates["n_diffusion_steps"] = args_dict["n_diffusion_steps"]
    if args_dict.get("schedule_type") is not None:
        updates["schedule_type"] = args_dict["schedule_type"]
    if "noise_level" in args_dict:  # argparse.SUPPRESS: present only when the flag was passed
        updates["noise_level"] = args_dict["noise_level"]
    return base.model_copy(update=updates) if updates else base


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    args_dict = vars(args)  # so `in args` works for the SUPPRESS-defaulted noise_level

    from mira.inference.loading import load_world_model  # noqa: PLC0415
    from mira.training.checkpoints import resolve_checkpoint  # noqa: PLC0415
    from mira.training.reproducibility import seed_everything  # noqa: PLC0415

    seed_everything(args.seed, deterministic=args.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = resolve_checkpoint(args.checkpoint).resolve()
    cfg = load_run_config(checkpoint)
    eval_split = args.split or cfg.dataset.get("test_split", "test")
    eval_group_mode = args.group_mode or cfg.dataset.get("group_mode")
    output_dir = args.output_dir or (checkpoint.parent / "offline_eval")

    model, _ = load_world_model(checkpoint, device=device)
    model = model.eval()
    logger.info("Loaded world model from %s on %s", checkpoint, device)

    eval_config = load_eval_metrics_config(
        num_samples=args.num_samples,
        no_compile=args.no_compile,
    )
    eval_config.inference = _inference_overrides(args_dict, eval_config.inference)
    if args.dino_model is not None:
        eval_config.dino_model = args.dino_model
    if args.per_device_batch_size is not None:
        eval_config.per_device_batch_size = args.per_device_batch_size
    for argument, field in (
        ("n_context_frames", "n_context_frames"),
        ("num_unrolled_frames", "num_unrolled_frames"),
        ("drift_metric_frames", "drift_metric_frames"),
        ("fdd_slice_frames", "fdd_slice_frames"),
    ):
        if (value := getattr(args, argument)) is not None:
            setattr(eval_config, field, value)
    compile_models = (not args.no_compile) and bool(cfg.run.get("compile"))

    results: dict[str, float] = {}
    per_batch_records: list[dict[str, Any]] | None = [] if args.per_batch_jsonl is not None else None
    val_num_batches = 0
    metric_num_batches = 0

    if not args.skip_validation:
        batch_size = cfg.validation.batch_size or cfg.run.batch_size
        n_samples = args.val_n_samples if args.val_n_samples is not None else cfg.validation.val_n_samples
        val_num_batches = _exact_num_batches("val_n_samples", n_samples, batch_size)
        seed_everything(args.seed, deterministic=args.deterministic)
        val_loader = _build_loader(
            cfg,
            model,
            split=eval_split,
            group_mode=eval_group_mode,
            clip_len=model.config.video.timesteps,
            batch_size=batch_size,
            seed=args.seed,
            window_mode=args.window_mode,
        )
        results |= {
            f"{eval_split}/{k}": v
            for k, v in run_validation_loss(
                model,
                _action_mode_iter(iter(val_loader), args.action_mode),
                device,
                n_batches=val_num_batches,
                per_batch_records=per_batch_records,
            ).items()
        }

    if not args.skip_metrics:
        # Apply the eval's context override before deriving the clip length, so the loader, the
        # rollout, and the metric indexing all agree on n_context_frames (see set_inference_context).
        if eval_config.n_context_frames is not None:
            model.set_inference_context(eval_config.n_context_frames)
        stride = eval_config.eval_temporal_downsampling or model.temporal_downsampling
        metric_num_batches = _exact_num_batches(
            "num_samples", eval_config.num_samples, eval_config.per_device_batch_size
        )
        seed_everything(args.seed + 1, deterministic=args.deterministic)
        metrics_loader = _build_loader(
            cfg,
            model,
            split=eval_split,
            group_mode=eval_group_mode,
            clip_len=model.config.n_context_frames + eval_config.num_unrolled_frames * stride,
            batch_size=eval_config.per_device_batch_size,
            seed=args.seed + 1,
            window_mode=args.window_mode,
        )
        results |= {
            f"metrics/{k}": v
            for k, v in run_world_model_metrics(
                model,
                _action_mode_iter(iter(metrics_loader), args.action_mode),
                device,
                wm_metrics_config=eval_config,
                num_eval_batches=metric_num_batches,
                num_viz=args.viz,
                output_dir=output_dir,
                compile_models=compile_models,
            ).items()
        }
        # Pure denoising speed at the same group batch size as the metric pass (no decode/metrics).
        try:
            seed_everything(args.seed + 2, deterministic=args.deterministic)
            speed_batch = _build_loader(
                cfg,
                model,
                split=eval_split,
                group_mode=eval_group_mode,
                clip_len=model.config.n_context_frames + SPEED_BENCH_FRAMES * model.temporal_downsampling,
                batch_size=eval_config.per_device_batch_size,
                seed=args.seed + 2,
                window_mode=args.window_mode,
            )
            batch, _ = next(_action_mode_iter(iter(speed_batch), args.action_mode))
            batch = batch.to(device)
            speed_results = measure_denoise_speed(model, batch, eval_config.inference)
            speed_results["denoise_raw_pov_rows"] = float(len(batch))
            speed_results["denoise_pov_latent_fps"] = speed_results["denoise_latent_fps"] * len(batch)
            results |= {f"metrics/{k}": v for k, v in speed_results.items()}
        except Exception:
            logger.exception("Denoise-speed measurement failed; skipping it.")

    logger.info("%s", "=" * 60)
    logger.info("Offline eval results:")
    for k, v in results.items():
        logger.info("  %s: %.4f", k, v)
    per_batch_sha256 = None
    if args.per_batch_jsonl is not None:
        if per_batch_records is None or len(per_batch_records) != val_num_batches:
            raise ValueError("per-batch validation records do not match the evaluated batch count")
        args.per_batch_jsonl.parent.mkdir(parents=True, exist_ok=True)
        temporary_records = args.per_batch_jsonl.with_name(f".{args.per_batch_jsonl.name}.tmp")
        with temporary_records.open("w", encoding="utf-8") as handle:
            for record in per_batch_records:
                handle.write(
                    json.dumps(
                        {
                            **record,
                            "seed": args.seed,
                            "action_mode": args.action_mode,
                            "window_mode": args.window_mode,
                            "split": eval_split,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        temporary_records.replace(args.per_batch_jsonl)
        per_batch_sha256 = _sha256(args.per_batch_jsonl)
    if args.results_json is not None:
        args.results_json.parent.mkdir(parents=True, exist_ok=True)
        n_players = getattr(model, "n_players", 1)
        payload = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "split": eval_split,
            "seed": args.seed,
            "deterministic": args.deterministic,
            "dataset_backend": cfg.dataset.get("backend", "rocket_science"),
            "map_slug": cfg.dataset.get("map_slug"),
            "group_mode": eval_group_mode,
            "training_group_mode": cfg.dataset.get("group_mode"),
            "action_routing": getattr(model, "action_routing", None),
            "action_mode": args.action_mode,
            "window_mode": args.window_mode,
            "n_players": n_players,
            "dino_model": eval_config.dino_model,
            "world_model_metrics": eval_config.model_dump(),
            "validation": {
                "batch_size_groups": cfg.validation.batch_size or cfg.run.batch_size,
                "num_batches": val_num_batches,
                "requested_samples": (
                    args.val_n_samples if args.val_n_samples is not None else cfg.validation.val_n_samples
                ),
                "raw_pov_rows_per_batch": (cfg.validation.batch_size or cfg.run.batch_size) * n_players,
                "total_raw_pov_rows": val_num_batches
                * (cfg.validation.batch_size or cfg.run.batch_size)
                * n_players,
            },
            "metrics": {
                "batch_size_groups": eval_config.per_device_batch_size,
                "num_batches": metric_num_batches,
                "requested_samples": eval_config.num_samples,
                "raw_pov_rows_per_batch": eval_config.per_device_batch_size * n_players,
                "total_raw_pov_rows": metric_num_batches * eval_config.per_device_batch_size * n_players,
            },
            "results": results,
            "per_batch_records": (
                None
                if args.per_batch_jsonl is None
                else {
                    "path": str(args.per_batch_jsonl),
                    "sha256": per_batch_sha256,
                    "count": len(per_batch_records or []),
                    "cluster_unit": "round_id",
                }
            ),
        }
        temporary = args.results_json.with_suffix(args.results_json.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(args.results_json)


if __name__ == "__main__":
    main()
