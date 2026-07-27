"""Train the RAEv2 temporal-downsampling video codec.

A single-GPU-safe trainer: reconstruction losses only (L1 + LPIPS + DINO latent consistency),
video-only. Data is read through :func:`mira.data.training_loader.create_loader`. Launch
with Hydra:

    python scripts/train_codec.py dataset.train_index=/path/to/train dataset.test_index=/path/to/test

Multi-GPU runs go through ``torchrun`` (the distributed setup no-ops for a single process).
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
from collections import defaultdict
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
import tqdm
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from rich.logging import RichHandler
from torch.nn.parallel import DistributedDataParallel

from mira.codec import CodecLoss, VideoCodec, VideoCodecOutputs
from mira.codec.viz import (
    visualize_latent_correlation,
    visualize_latent_std,
    visualize_side_by_side,
)
from mira.data.training_loader import ClipMeta, create_loader
from mira.training.checkpoint_manager import CheckpointManager
from mira.training.distributed import (
    broadcast_main_process_bool,
    get_distributed_settings,
    set_up_distributed,
)
from mira.training.ema import DistributedEMA
from mira.training.lr_schedule import WarmupConstantCosineDecayLR
from mira.training.metrics.distributed_metric import DistributedMetric
from mira.training.reproducibility import seed_everything
from mira.training.tracker import TrainingTracker, display_execution_time, periodic_event
from mira.training.visualization import (
    VideoForWandb,
    draw_text_on_first_frame,
    videos_for_wandb,
)

logging.basicConfig(format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])

logger = logging.getLogger(__name__)


def _jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().item() if value.numel() == 1 else value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _log_jsonl(cfg: DictConfig, record: dict) -> None:
    """Append a dependency-free local metrics record even when W&B is disabled/offline."""
    path = Path(cfg.run.output_dir) / "metrics.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: _jsonable(value) for key, value in record.items()}
    payload["timestamp_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def _autocast(device: int | str | torch.device):
    """bfloat16 autocast on CUDA, a no-op elsewhere (so the trainer runs on CPU too)."""
    if torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _frame_size(cfg: DictConfig) -> tuple[int, int] | None:
    fs = cfg.dataset.get("frame_size")
    return tuple(fs) if fs is not None else None  # type: ignore[return-value]


@hydra.main(version_base=None, config_path="../configs", config_name="train_codec")
def train(cfg: DictConfig) -> None:
    distributed_settings = set_up_distributed()
    is_main_process = distributed_settings.is_main_process
    device = distributed_settings.device

    seed_everything(
        cfg.run.seed + distributed_settings.rank,
        deterministic=bool(cfg.run.get("deterministic", False)),
    )
    logging.getLogger().setLevel(logging.INFO if is_main_process else logging.ERROR)

    logger.info("[magenta]" + "=" * 60 + "[/magenta]")
    logger.info("[magenta]Training configuration:[/magenta]")
    logger.info(OmegaConf.to_yaml(cfg))

    if is_main_process:
        _init_wandb(cfg)

    # Build the codec and (optionally) wrap it for distributed data parallelism.
    raw_model: VideoCodec = instantiate(cfg.model.architecture)
    raw_model.train().to(device)
    is_distributed = dist.is_available() and dist.is_initialized()
    model = DistributedDataParallel(raw_model, device_ids=[device]) if is_distributed else raw_model

    if cfg.run.get("compile"):
        model.compile()

    if is_main_process:
        n_params = sum(p.numel() for p in raw_model.parameters()) / 1e6
        logger.info(f"Initialized codec model with {n_params:.2f}M parameters")

    loss = CodecLoss(instantiate(cfg.model.loss.weights))
    loss.to(device)
    if loss.weights.auto_weight:
        loss.bind_last_layer(raw_model.decoder.last_layer_weight)
    if loss.weights.loss_dino_latent_consistency > 0:
        loss.bind_encoder_dino(raw_model.encoder.rae_dino)

    train_loader, val_loader = _create_dataloaders(cfg, model=raw_model)

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
    ema_latent_mean = DistributedEMA(decay=cfg.run.latents_ema_decay, device=device)
    ema_latent_std = DistributedEMA(decay=cfg.run.latents_ema_decay, initial_value=1.0, device=device)
    checkpoint_manager.register(
        {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "ema_latent_mean": ema_latent_mean,
            "ema_latent_std": ema_latent_std,
        }
    )

    start_step = _resume(cfg, checkpoint_manager, ema_latent_mean, ema_latent_std)

    losses: dict[str, torch.Tensor] = {}
    # Start the timed budget only after every rank has loaded state and reached the same boundary.
    if is_distributed:
        dist.barrier()
    training_start_time = time.monotonic()
    max_duration_hours = cfg.run.get("max_duration_hours")
    iter_num = start_step - 1  # so the final save below is well-defined even if the loop never runs
    for iter_num in range(start_step, int(cfg.run.steps)):
        step_start_time = time.monotonic()

        batch, _ = next(iter_train_loader)
        batch = batch.to(device)

        optimizer.zero_grad(set_to_none=True)
        with _autocast(device):
            model_outputs = model(batch)
            # The losses compute DINO embeddings, so keep autocast active here too.
            losses = loss(model_outputs, global_step=iter_num)

        with torch.no_grad():
            z = model_outputs.z.float()  # from bfloat16 to float32
            ema_latent_mean.update(z)
            ema_latent_std.update(z.std(keepdim=True))

        losses["loss_total"].backward()
        optimizer_step()
        checkpoint_manager.model_ema.step()

        training_tracker.on_batch_processed(batch, losses)

        early_logging_steps = 10
        if periodic_event(iter_num, cfg.run.log_every, cfg.run.steps) or iter_num < early_logging_steps:
            # All ranks must call get_stats()/compute() so the inner all_reduce completes.
            stats = training_tracker.get_stats(step=iter_num)
            stats["train/learning_rate"] = optimizer.param_groups[0]["lr"]
            stats["train/latent_mean"] = ema_latent_mean.compute()
            stats["train/latent_std"] = ema_latent_std.compute()
            if is_main_process:
                stats["System/step_ms"] = (time.monotonic() - step_start_time) * 1000
                stats["System/elapsed_wall_seconds"] = time.monotonic() - training_start_time
                stats |= {f"grad_norm/{k}": v.item() for k, v in loss.backward_metrics.items()}
                logger.info(f"Step {iter_num}: total loss {stats['train/loss_total']:.4f}")
                _wandb_log(stats, step=iter_num)
                _log_jsonl(cfg, {"kind": "train", "step": iter_num, **stats})

        if periodic_event(
            iter_num, cfg.validation.val_every, cfg.run.steps, include_0=cfg.validation.val_first
        ):
            with checkpoint_manager.model_ema.average_parameters():
                run_validation(cfg, device, raw_model, iter(val_loader), loss, iter_num)

        if periodic_event(iter_num, cfg.run.checkpoint_every, cfg.run.steps, include_0=False):
            if is_main_process:
                checkpoint_manager.maybe_save_checkpoint(
                    iter_num,
                    extra_data=_extra_data(iter_num, losses, ema_latent_mean, ema_latent_std),
                )
            if is_distributed:
                dist.barrier()

        if is_distributed:
            dist.barrier()

        local_time_limit_reached = (
            max_duration_hours is not None
            and time.monotonic() - training_start_time >= float(max_duration_hours) * 3600
        )
        time_limit_reached = (
            broadcast_main_process_bool(local_time_limit_reached, device=device)
            if max_duration_hours is not None
            else False
        )
        if time_limit_reached:
            if is_main_process:
                logger.info(
                    "Reached run.max_duration_hours=%.3f after step %d; saving final checkpoint.",
                    float(max_duration_hours),
                    iter_num,
                )
                _log_jsonl(
                    cfg,
                    {
                        "kind": "time_limit",
                        "step": iter_num,
                        "elapsed_wall_seconds": time.monotonic() - training_start_time,
                    },
                )
            break

    if is_main_process and iter_num >= start_step:  # skip when resuming an already-finished run
        checkpoint_manager.maybe_save_checkpoint(
            iter_num, extra_data=_extra_data(iter_num, losses, ema_latent_mean, ema_latent_std), final=True
        )
    logger.info("Done training")
    if is_distributed:
        # Keep peers alive until rank 0 atomically publishes the final model and training state.
        dist.barrier()
        dist.destroy_process_group()


def _extra_data(
    iter_num: int,
    losses: dict[str, torch.Tensor],
    ema_latent_mean: DistributedEMA,
    ema_latent_std: DistributedEMA,
) -> dict:
    extra_data = {k: v.item() for k, v in losses.items()}
    extra_data["iter_num"] = iter_num
    extra_data["latent_mean_std"] = [ema_latent_mean.value, ema_latent_std.value]
    return extra_data


def _init_wandb(cfg: DictConfig) -> None:
    import wandb  # noqa: PLC0415 -- optional dep, used only on the main process

    if cfg.wandb.remove_old_runs:
        try:
            api = wandb.Api()
            runs = api.runs(
                f"{cfg.wandb.entity}/{cfg.wandb.project}", filters={"display_name": cfg.wandb.name}
            )
            for run in runs:
                logger.info(f"Deleting existing run: {run.id}")
                run.delete()
        except Exception:  # noqa: BLE001 -- skip if the project does not exist yet
            pass
    else:
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
    # The codec checkpoint loader reads codec_config.yaml from a parent dir of the checkpoint.
    OmegaConf.save(cfg, Path(cfg.run.output_dir) / VideoCodec.CONFIG_FILENAME, resolve=True)


def _wandb_log(stats: dict, step: int) -> None:
    import wandb  # noqa: PLC0415 -- optional dep, used only on the main process

    if wandb.run is not None:
        wandb.log(stats, step=step)


def _resume(
    cfg: DictConfig,
    checkpoint_manager: CheckpointManager,
    ema_latent_mean: DistributedEMA,
    ema_latent_std: DistributedEMA,
) -> int:
    """Resume from continue_from / finetune_from / an auto-discovered checkpoint, else start at 0."""
    continue_from = cfg.run.get("continue_from")
    finetune_from = cfg.run.get("finetune_from")
    if continue_from and finetune_from:
        raise ValueError("Set at most one of run.continue_from and run.finetune_from")

    if continue_from:
        return checkpoint_manager.continue_from(continue_from)
    if finetune_from:
        init_extra_data = checkpoint_manager.finetune_from(finetune_from)
        if "latent_mean_std" in init_extra_data:
            init_mean, init_std = init_extra_data["latent_mean_std"]
            ema_latent_mean._ema.fill_(float(init_mean))
            ema_latent_std._ema.fill_(float(init_std))
            logger.info(f"Seeded latent EMA from finetune checkpoint: mean={init_mean}, std={init_std}")
        return 0
    if checkpoint_manager.latest_checkpoint is not None:
        logger.info(f"Auto-resuming from existing checkpoint in {cfg.run.output_dir}")
        return checkpoint_manager.continue_from(checkpoint_manager.latest_checkpoint)
    return 0


def _create_dataloaders(cfg: DictConfig, model: VideoCodec):
    common = dict(
        dataset_backend=cfg.dataset.get("backend", "rocket_science"),
        map_slug=cfg.dataset.get("map_slug"),
        group_mode=cfg.dataset.get("group_mode"),
        clip_len=model.config.encoder.video.timesteps,
        target_fps=model.config.encoder.video.fps,
        n_players=cfg.dataset.n_players,
        num_workers=cfg.dataloader.num_workers,
        shuffle_buffer_size=cfg.dataloader.shuffle_buffer_size,
        prefetch_factor=cfg.dataloader.get("prefetch_factor", 2),
        persistent_workers=cfg.dataloader.get("persistent_workers", False),
        pin_memory=cfg.dataloader.get("pin_memory"),
        frame_size=_frame_size(cfg),
        valid_keys=list(cfg.actions.valid_keys),
        infinite=True,
    )
    train_loader = create_loader(
        index_path=cfg.dataset.train_index,
        split=cfg.dataset.get("train_split", "train"),
        batch_size=cfg.run.batch_size,
        seed=cfg.run.seed,
        exclude_replays=cfg.dataset.exclude_replays,
        shuffle=True,
        **common,
    )
    val_loader = create_loader(
        index_path=cfg.dataset.test_index,
        split=cfg.dataset.get("test_split", "test"),
        batch_size=cfg.validation.batch_size or cfg.run.batch_size,
        seed=37,
        exclude_replays=True,
        shuffle=False,
        **common,
    )
    return train_loader, val_loader


def run_validation(
    cfg: DictConfig,
    device: torch.device | int | str,
    model: VideoCodec,
    iter_val_loader: Iterator[tuple],
    loss: CodecLoss,
    iter_num: int,
) -> None:
    t1 = time.time()
    distributed_settings = get_distributed_settings()
    is_distributed = dist.is_available() and dist.is_initialized()
    model.eval()
    world_size = distributed_settings.world_size
    val_batch_size = cfg.validation.batch_size or cfg.run.batch_size
    n_batches = cfg.validation.val_n_samples // (val_batch_size * world_size)
    fps = model.config.encoder.video.fps

    if distributed_settings.is_main_process:
        _visualize(model, iter_val_loader, device, iter_num, fps)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if is_distributed:
        dist.barrier()

    metric_trackers: dict[str, DistributedMetric] = defaultdict(lambda: DistributedMetric(device=device))
    for _ in tqdm.trange(
        max(1, n_batches), disable=not distributed_settings.is_main_process, desc="Running validation"
    ):
        batch, _ = next(iter_val_loader)
        batch = batch.to(device)
        with torch.no_grad(), _autocast(device):
            model_outputs: VideoCodecOutputs = model(batch)
            for k, v in loss(model_outputs).items():
                metric_trackers[k].update(v)

    metrics = {k: tracker.compute_and_reset().item() for k, tracker in metric_trackers.items()}
    if distributed_settings.is_main_process:
        logger.info(f"Validation took {time.time() - t1:.2f}s")
        logger.info(
            f"Validation at step {iter_num}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        )
        _wandb_log({f"test/{k}": v for k, v in metrics.items()}, step=iter_num)
        _log_jsonl(
            cfg,
            {
                "kind": "validation",
                "step": iter_num,
                "duration_seconds": time.time() - t1,
                **{f"test/{k}": v for k, v in metrics.items()},
            },
        )

    model.train()
    if is_distributed:
        dist.barrier()


def _visualize(
    model: VideoCodec,
    iter_val_loader: Iterator[tuple],
    device: torch.device | int | str,
    iter_num: int,
    fps: int,
) -> None:
    """Log a reconstruction side-by-side video and latent diagnostics to W&B (rank 0 only)."""
    batch, metadata = next(iter_val_loader)
    batch = batch.to(device)
    with torch.no_grad(), _autocast(device):
        model_outputs: VideoCodecOutputs = model(batch)

    meta: list[ClipMeta] = metadata
    captions = [f"{m.match_id}:{m.perspective}" for m in meta]
    caption = ", ".join(captions)

    viz_video = visualize_side_by_side(model_outputs)["viz_video"]
    viz_video = draw_text_on_first_frame(viz_video, captions)

    with videos_for_wandb(
        {"videos/reconstruction": VideoForWandb(video=viz_video, caption=caption)}, fps=fps
    ) as wandb_videos:
        _wandb_log(dict(wandb_videos), step=iter_num)

    _wandb_log(
        {
            "viz/latent_correlation_patch": visualize_latent_correlation(model_outputs, "patch", caption),
            "viz/latent_correlation_channels": visualize_latent_correlation(
                model_outputs, "channels", caption
            ),
            "viz/latent_correlation_time": visualize_latent_correlation(model_outputs, "time", caption),
            "viz/latent_std_patch": visualize_latent_std(model_outputs, axis="patch"),
            "viz/latent_std_channels": visualize_latent_std(model_outputs, axis="channels"),
        },
        step=iter_num,
    )


if __name__ == "__main__":
    train()
