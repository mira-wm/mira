"""Train the latent world model (frozen codec + action-conditioned diffusion transformer).

A single-GPU-safe trainer: flow-matching diffusion loss on codec latents, periodic validation loss,
and an optional world-model metrics eval (DINO/latent drift + Frechet curves) with rollout
visualizations logged to W&B. Video-only. Data is read
through :func:`mira.data.training_loader.create_loader`; the codec is loaded frozen from
its checkpoint by the model itself. Launch with Hydra::

    python scripts/train_world_model.py \\
        model.architecture.config.codec_checkpoint=/path/to/codec_ckpt \\
        dataset.train_index=/path/to/train dataset.test_index=/path/to/test

Multi-GPU runs go through ``torchrun`` (the distributed setup no-ops for a single process).
``run.compile`` is opt-in (default off) for reproducibility; when enabled it compiles the codec
encoder, the inner diffusion transformer, and the codec decoder *separately* — the model forward has
data-dependent control flow, so compiling the whole module would graph-break every step.
"""

from __future__ import annotations

import contextlib
import logging
import random
import time
from collections import defaultdict
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
import tqdm
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from rich.logging import RichHandler
from torch.nn.parallel import DistributedDataParallel

from mira.data.training_loader import ClipMeta, create_loader
from mira.training.checkpoint_manager import CheckpointManager
from mira.training.distributed import get_distributed_settings, set_up_distributed
from mira.training.lr_schedule import WarmupConstantCosineDecayLR
from mira.training.metrics.distributed_metric import DistributedMetric
from mira.training.metrics.world_model_metrics import (
    WorldModelMetrics,
    WorldModelMetricsConfig,
    build_frechet_curve_plots,
)
from mira.training.tracker import TrainingTracker, display_execution_time, periodic_event
from mira.training.visualization import (
    VideoForWandb,
    draw_text_on_first_frame,
    videos_for_wandb,
)
from mira.world_model.latent_world_model import InferenceOutputs, LatentWorldModel

logging.basicConfig(format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])

logger = logging.getLogger(__name__)


def _autocast(device: int | str | torch.device):
    """bfloat16 autocast on CUDA, a no-op elsewhere (so the trainer runs on CPU too)."""
    if torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _frame_size(cfg: DictConfig) -> tuple[int, int] | None:
    fs = cfg.dataset.get("frame_size")
    return tuple(fs) if fs is not None else None  # type: ignore[return-value]


@hydra.main(version_base=None, config_path="../configs", config_name="train_world_model")
def train(cfg: DictConfig) -> None:
    distributed_settings = set_up_distributed()
    is_main_process = distributed_settings.is_main_process
    device = distributed_settings.device

    torch.manual_seed(cfg.run.seed + distributed_settings.rank)
    logging.getLogger().setLevel(logging.INFO if is_main_process else logging.ERROR)

    logger.info("[magenta]" + "=" * 60 + "[/magenta]")
    logger.info("[magenta]Training configuration:[/magenta]")
    logger.info(OmegaConf.to_yaml(cfg))

    if is_main_process:
        _init_wandb(cfg)

    raw_model: LatentWorldModel = instantiate(cfg.model.architecture)
    raw_model.train().to(device)
    is_distributed = dist.is_available() and dist.is_initialized()
    model = DistributedDataParallel(raw_model, device_ids=[device]) if is_distributed else raw_model

    if cfg.run.get("compile"):
        # Compile the codec encoder, the inner diffusion transformer, and the codec decoder
        # separately. The model forward has data-dependent control flow, so compiling the whole
        # module would graph-break every step.
        raw_model.codec.encode = torch.compile(raw_model.codec.encode)
        raw_model.world_model.compile()
        raw_model.decode_to_video = torch.compile(raw_model.decode_to_video)

    if is_main_process:
        n_codec_params = sum(p.numel() for p in raw_model.codec.parameters())
        n_total_params = sum(p.numel() for p in raw_model.parameters())
        logger.info(
            f"Initialized latent world model with {(n_total_params - n_codec_params) / 1e6:.2f}M "
            f"parameters (+{n_codec_params / 1e6:.2f}M in the frozen codec)."
        )

    wm_metrics_config: WorldModelMetricsConfig = instantiate(cfg.world_model_metrics)
    train_loader, val_loader, metrics_loader = _create_dataloaders(cfg, wm_metrics_config, raw_model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        # `lr` needs to be a tensor to compile the backward pass.
        **{**cfg.optim.optimizer, "lr": torch.tensor(cfg.optim.optimizer.lr)},
    )
    lr_scheduler = WarmupConstantCosineDecayLR(optimizer, **cfg.optim.scheduler)

    @torch.compile(disable=not cfg.run.get("compile"))
    def optimizer_step() -> None:
        optimizer.step()
        lr_scheduler.step()

    iter_train_loader = iter(train_loader)
    with display_execution_time("Warming up dataloader", print_output=is_main_process):
        # First batch takes longer; warm it up before the timed loop.
        next(iter_train_loader)

    training_tracker = TrainingTracker(
        world_size=distributed_settings.world_size, device=device, total_steps=int(cfg.run.steps)
    )
    checkpoint_manager = CheckpointManager(
        raw_model,
        checkpoint_dir=cfg.run.output_dir,
        save_every=cfg.run.checkpoint_every,
        keep_recent=cfg.run.checkpoint_keep_recent,
        keep_permanent_every=cfg.run.checkpoint_keep_permanent_every,
        total_steps=int(cfg.run.steps),
        model_ema_decay=cfg.optim.model_ema_decay,
    )
    checkpoint_manager.register({"optimizer": optimizer, "lr_scheduler": lr_scheduler})

    start_step = _resume(cfg, checkpoint_manager)

    # Built lazily on the first metrics eval: constructing it loads the DINO/Inception backbones and
    # pytorch_fid. If those are unavailable, downstream metrics are skipped for the rest of training.
    wm_metrics: WorldModelMetrics | None = None
    wm_metrics_disabled = False

    losses: dict[str, torch.Tensor] = {}
    iter_num = start_step - 1  # well-defined for the final save below if the loop never runs
    for iter_num in range(start_step, int(cfg.run.steps)):
        step_start_time = time.monotonic()

        batch, _ = next(iter_train_loader)
        batch = batch.to(device)

        optimizer.zero_grad(set_to_none=True)
        with _autocast(device):
            losses = model(batch)

        losses["loss_total"].backward()
        if cfg.optim.max_grad_norm:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.max_grad_norm)
        optimizer_step()
        checkpoint_manager.model_ema.step()

        training_tracker.on_batch_processed(batch, losses)

        early_logging_steps = 10
        if periodic_event(iter_num, cfg.run.log_every, cfg.run.steps) or iter_num < early_logging_steps:
            # All ranks must call get_stats() so the inner all_reduce completes.
            stats = training_tracker.get_stats(step=iter_num)
            stats["train/learning_rate"] = optimizer.param_groups[0]["lr"]
            if cfg.optim.max_grad_norm:
                stats["train/grad_norm"] = grad_norm.item()
            if is_main_process:
                stats["System/step_ms"] = (time.monotonic() - step_start_time) * 1000
                logger.info(f"Step {iter_num}: total loss {stats['train/loss_total']:.4f}")
                _wandb_log(stats, step=iter_num)

        if periodic_event(
            iter_num, cfg.validation.val_every, cfg.run.steps, include_0=cfg.validation.val_first
        ):
            with checkpoint_manager.model_ema.average_parameters():
                # Fresh iterator each eval so the same fixed subsample is scored every time.
                run_validation(cfg, device, raw_model, iter(val_loader), iter_num)

        if periodic_event(iter_num, cfg.run.checkpoint_every, cfg.run.steps, include_0=False):
            if is_main_process:
                checkpoint_manager.maybe_save_checkpoint(iter_num, extra_data=_extra_data(iter_num, losses))
            if is_distributed:
                dist.barrier()

        if not wm_metrics_disabled and periodic_event(
            iter_num, cfg.validation.downstream_val_every, cfg.run.steps, include_0=False
        ):
            if wm_metrics is None:
                try:
                    wm_metrics = WorldModelMetrics(wm_metrics_config, iter(metrics_loader), device)
                except Exception as exc:  # noqa: BLE001 -- any dep/weight failure disables the eval
                    logger.warning(
                        "World-model metrics unavailable (%s: %s); skipping the downstream eval for "
                        "the rest of training. Install `mira[eval]` and make the DINO/"
                        "Inception weights reachable to enable it.",
                        type(exc).__name__,
                        exc,
                    )
                    wm_metrics_disabled = True
            if wm_metrics is not None:
                with checkpoint_manager.model_ema.average_parameters():
                    run_world_model_metrics(cfg, wm_metrics, metrics_loader, raw_model, iter_num)

        if is_distributed:
            dist.barrier()

    if is_main_process and iter_num >= start_step:  # skip when resuming an already-finished run
        checkpoint_manager.maybe_save_checkpoint(
            iter_num, extra_data=_extra_data(iter_num, losses), final=True
        )
    logger.info("Done training")
    if is_distributed:
        dist.destroy_process_group()


def run_validation(
    cfg: DictConfig,
    device: torch.device | int | str,
    model: LatentWorldModel,
    iter_val_loader: Iterator[tuple],
    iter_num: int,
) -> None:
    """Average the world-model forward loss over a fixed validation subsample. Runs on all ranks."""
    t1 = time.time()
    distributed_settings = get_distributed_settings()
    is_distributed = dist.is_available() and dist.is_initialized()
    model.eval()
    world_size = distributed_settings.world_size
    val_batch_size = cfg.validation.batch_size or cfg.run.batch_size
    n_batches = cfg.validation.val_n_samples // (val_batch_size * world_size)

    metric_trackers: dict[str, DistributedMetric] = defaultdict(lambda: DistributedMetric(device=device))
    for _ in tqdm.trange(
        max(1, n_batches), disable=not distributed_settings.is_main_process, desc="Running validation"
    ):
        batch, _ = next(iter_val_loader)
        batch = batch.to(device)
        with torch.no_grad(), _autocast(device):
            for k, v in model(batch).items():
                metric_trackers[k].update(v)

    metrics = {k: tracker.compute_and_reset().item() for k, tracker in metric_trackers.items()}
    if distributed_settings.is_main_process:
        logger.info(f"Validation took {time.time() - t1:.2f}s")
        logger.info(
            f"Validation at step {iter_num}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        )
        _wandb_log({f"test/{k}": v for k, v in metrics.items()}, step=iter_num)

    model.train()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if is_distributed:
        dist.barrier()


def run_world_model_metrics(
    cfg: DictConfig,
    wm_metrics: WorldModelMetrics,
    metrics_loader,
    model: LatentWorldModel,
    iter_num: int,
) -> None:
    """Unroll the world model on a held-out subsample, log metrics + rollout viz. Runs on all ranks."""
    distributed_settings = get_distributed_settings()
    is_main_process = distributed_settings.is_main_process
    is_distributed = dist.is_available() and dist.is_initialized()
    config = wm_metrics.config

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model.eval()

    num_eval_batches = max(
        1, config.num_samples // (config.per_device_batch_size * distributed_settings.world_size)
    )
    logger.info(f"Running world model metrics at step {iter_num} for {num_eval_batches} batches")

    # Fresh iterator each eval so the same fixed subsample is scored every time.
    wm_metrics.iter_dataloader = iter(metrics_loader)

    # Visualize a random subset of the eval samples, reusing the videos generated for metrics (no
    # extra inference). Re-sampled each eval so the logged clips vary; rank 0 only (it does logging).
    viz_batch_indices: set[int] = set()
    if is_main_process:
        viz_batch_indices = set(
            random.sample(range(num_eval_batches), min(config.num_viz_samples, num_eval_batches))
        )
    viz_samples: list[dict] = []

    t_start = time.time()
    for batch_idx in tqdm.trange(
        num_eval_batches, disable=not is_main_process, desc="Evaluating world model metrics"
    ):
        inference_outputs, metadata = wm_metrics.process_batch(model)
        if batch_idx in viz_batch_indices:
            viz_samples.append(_render_viz_sample(model, inference_outputs, metadata))
        if is_distributed:
            dist.barrier()

    if is_main_process and viz_samples:
        _log_viz_videos(viz_samples, fps=model.config.video.fps, iter_num=iter_num)

    metric_values, frechet_curves = wm_metrics.compute()

    if is_main_process:
        for k, v in metric_values.items():
            logger.info(f"  {k}: {float(v):.4f}")
        viz_plots = build_frechet_curve_plots(frechet_curves, config.fdd_slice_frames)
        _wandb_log(
            {f"metrics/{k}": float(v) for k, v in metric_values.items()}
            | viz_plots
            | {"metrics/time_min": (time.time() - t_start) / 60},
            step=iter_num,
        )

    model.train()
    if is_distributed:
        dist.barrier()


def _render_viz_sample(
    model: LatentWorldModel, inference_outputs: InferenceOutputs, metadata: list[ClipMeta]
) -> dict:
    """Render one already-generated rollout into a CPU viz tensor (rank 0, metrics loop)."""
    n_players = getattr(model, "n_players", 1)
    captions = [f"{m.match_id}:{m.perspective}" for m in metadata]
    grouped = [" + ".join(captions[i : i + n_players]) for i in range(0, len(captions), n_players)]

    viz_video = model.visualize(inference_outputs)["viz_video"]
    viz_video = draw_text_on_first_frame(viz_video, grouped)
    return {"viz_video": viz_video.cpu(), "captions": captions}


def _log_viz_videos(viz_samples: list[dict], fps: float, iter_num: int) -> None:
    """Concatenate per-sample rollout viz tensors and log them to W&B. Rank 0 only."""
    caption = ", ".join(cid for s in viz_samples for cid in s["captions"])
    video = torch.cat([s["viz_video"] for s in viz_samples], dim=0)
    with videos_for_wandb(
        {"videos/viz_video": VideoForWandb(video=video, caption=caption)}, fps=fps
    ) as wandb_videos:
        _wandb_log(dict(wandb_videos), step=iter_num)


def _create_dataloaders(cfg: DictConfig, wm_metrics_config: WorldModelMetricsConfig, model: LatentWorldModel):
    """Build the (train, val, metrics) dataloaders. The val/metrics loaders use fixed seeds so the
    same held-out subsample is scored every eval."""
    n_players = getattr(model, "n_players", 1)
    dataset_n_players = cfg.dataset.get("n_players", 1)
    if dataset_n_players != n_players:
        raise ValueError(
            f"dataset n_players ({dataset_n_players}) != model n_players ({n_players}); the "
            "match-grouped dataset must provide exactly as many perspectives as the model expects."
        )

    # Apply the eval's context override before deriving the metrics clip length, so the loader, the
    # rollout, and the metric indexing all agree on n_context_frames (see set_inference_context).
    if wm_metrics_config.n_context_frames is not None:
        model.set_inference_context(wm_metrics_config.n_context_frames)

    common = dict(
        target_fps=model.config.video.fps,
        # Actions are sampled at their own rate, decoupled from the frame rate. The released default
        # has both at 20fps (one action per video frame), so this is a no-op; setting actions.target_fps
        # above video.fps turns on the knob (e.g. 20fps frames + 40fps actions => 2 actions/frame).
        action_fps=model.config.actions.target_fps,
        n_players=n_players,
        num_workers=cfg.dataloader.num_workers,
        shuffle_buffer_size=cfg.dataloader.shuffle_buffer_size,
        frame_size=_frame_size(cfg),
        valid_keys=list(model.config.actions.valid_keys),
        infinite=True,
    )
    stride = wm_metrics_config.eval_temporal_downsampling or model.temporal_downsampling

    train_loader = create_loader(
        index_path=cfg.dataset.train_index,
        clip_len=model.config.video.timesteps,
        batch_size=cfg.run.batch_size,
        seed=cfg.run.seed,
        exclude_replays=cfg.dataset.exclude_replays,
        **common,
    )
    val_loader = create_loader(
        index_path=cfg.dataset.test_index,
        clip_len=model.config.video.timesteps,
        batch_size=cfg.validation.batch_size or cfg.run.batch_size,
        seed=37,
        exclude_replays=True,
        **common,
    )
    metrics_loader = create_loader(
        index_path=cfg.dataset.test_index,
        clip_len=model.config.n_context_frames + wm_metrics_config.num_unrolled_frames * stride,
        batch_size=wm_metrics_config.per_device_batch_size,
        seed=38,
        exclude_replays=True,
        **common,
    )
    return train_loader, val_loader, metrics_loader


def _extra_data(iter_num: int, losses: dict[str, torch.Tensor]) -> dict:
    extra_data = {k: v.item() for k, v in losses.items() if v.numel() == 1}
    extra_data["iter_num"] = iter_num
    return extra_data


def _init_wandb(cfg: DictConfig) -> None:
    import wandb  # noqa: PLC0415 -- optional dep, used only on the main process

    timestamp = datetime.now().strftime("%y%m%d-%H%M")
    cfg.wandb.name = f"{cfg.wandb.name}-{timestamp}"
    if cfg.wandb.group is not None:
        cfg.wandb.group = f"{cfg.wandb.group}-{timestamp}"

    Path(cfg.run.output_dir).mkdir(parents=True, exist_ok=True)
    wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        group=cfg.wandb.group,
        config=OmegaConf.to_container(cfg, resolve=True),
        dir=cfg.run.output_dir,
        mode=cfg.wandb.mode,
    )
    # The checkpoint loader reads world_model_config.yaml from a parent dir of the checkpoint.
    OmegaConf.save(cfg, Path(cfg.run.output_dir) / LatentWorldModel.CONFIG_FILENAME, resolve=True)


def _wandb_log(stats: dict, step: int) -> None:
    import wandb  # noqa: PLC0415 -- optional dep, used only on the main process

    if wandb.run is not None:
        wandb.log(stats, step=step)


def _resume(cfg: DictConfig, checkpoint_manager: CheckpointManager) -> int:
    """Resume from continue_from / finetune_from / an auto-discovered checkpoint, else start at 0."""
    continue_from = cfg.run.get("continue_from")
    finetune_from = cfg.run.get("finetune_from")
    if continue_from and finetune_from:
        raise ValueError("Set at most one of run.continue_from and run.finetune_from")

    if continue_from:
        return checkpoint_manager.continue_from(continue_from)
    if finetune_from:
        checkpoint_manager.finetune_from(finetune_from)
        return 0
    if checkpoint_manager.latest_checkpoint is not None:
        logger.info(f"Auto-resuming from existing checkpoint in {cfg.run.output_dir}")
        return checkpoint_manager.continue_from(checkpoint_manager.latest_checkpoint)
    return 0


if __name__ == "__main__":
    train()
