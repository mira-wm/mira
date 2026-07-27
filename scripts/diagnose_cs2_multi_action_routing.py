#!/usr/bin/env python3
"""Quantify how much a multiplayer checkpoint's action router preserves cross-POV changes.

The diagnostic keeps each synchronized video batch fixed, cyclically shifts actions across player
rows, and compares the action tensors at two points:

1. after player identity + projection, before routing; and
2. after the wrapper's configured routing (legacy global mean or spatial POV bands).

An attenuation ratio near zero means the routing operation erases most of the player-specific
conditioning change before it reaches the diffusion transformer. This is an architectural
diagnostic, not a model-quality endpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from einops import rearrange

from mira.data.training_loader import create_loader
from mira.inference.loading import load_world_model
from mira.training.checkpoints import resolve_checkpoint
from mira.training.reproducibility import seed_everything
from mira.world_model.multi_wrapper_world_model import MultiWrapperWorldModel


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(command: str) -> str:
    result = subprocess.run(
        ["git", *command.split()],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _pair_stats(reference: torch.Tensor, intervention: torch.Tensor) -> dict[str, float]:
    reference = reference.detach().float()
    intervention = intervention.detach().float()
    delta = intervention - reference
    reference_rms = torch.sqrt(torch.mean(reference.square())).item()
    intervention_rms = torch.sqrt(torch.mean(intervention.square())).item()
    delta_rms = torch.sqrt(torch.mean(delta.square())).item()
    cosine = torch.nn.functional.cosine_similarity(
        reference.reshape(1, -1),
        intervention.reshape(1, -1),
    ).item()
    return {
        "reference_rms": reference_rms,
        "intervention_rms": intervention_rms,
        "delta_rms": delta_rms,
        "relative_delta_rms": delta_rms / max(reference_rms, torch.finfo(torch.float32).eps),
        "cosine_similarity": cosine,
    }


def summarize_rows(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    if not rows:
        raise ValueError("At least one diagnostic row is required")
    summary: dict[str, dict[str, float]] = {}
    for key in rows[0]:
        values = [row[key] for row in rows]
        summary[key] = {
            "mean": statistics.fmean(values),
            "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    return summary


def _frame_size(cfg) -> tuple[int, int] | None:
    value = cfg.dataset.get("frame_size")
    return tuple(value) if value is not None else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="Multiplayer checkpoint path or W&B reference.")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--map-slug", default="dust2")
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--num-batches", type=int, default=52)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None, help="Default: cuda when available, otherwise cpu.")
    parser.add_argument(
        "--action-routing",
        choices=["global_mean", "spatial"],
        default=None,
        help="Override the saved checkpoint's router for this representation-only diagnostic.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_batches < 1:
        raise ValueError("--num-batches must be >= 1")

    seed_everything(args.seed, deterministic=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = resolve_checkpoint(args.checkpoint).resolve()
    model, cfg = load_world_model(checkpoint, device=device)
    if not isinstance(model, MultiWrapperWorldModel):
        raise TypeError("This diagnostic requires a MultiWrapperWorldModel checkpoint")
    model.eval()
    checkpoint_action_routing = model.action_routing
    if args.action_routing is not None:
        model.action_routing = args.action_routing

    loader = create_loader(
        index_path=cfg.dataset.test_index,
        dataset_backend=cfg.dataset.get("backend", "rocket_science"),
        split=args.split,
        map_slug=args.map_slug,
        group_mode="synchronized",
        window_mode="midpoint",
        clip_len=model.config.video.timesteps,
        target_fps=model.config.video.fps,
        action_fps=model.config.actions.target_fps,
        n_players=model.n_players,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False,
        infinite=True,
        shuffle_buffer_size=1,
        seed=args.seed,
        exclude_replays=True,
        frame_size=_frame_size(cfg),
        valid_keys=list(model.config.actions.valid_keys),
        pin_memory=False,
        persistent_workers=False,
    )

    projected_rows: list[dict[str, float]] = []
    routed_rows: list[dict[str, float]] = []
    raw_rows: list[dict[str, float]] = []
    iterator = iter(loader)
    swm = model.single_world_model
    off = swm.action_temporal_downsampling - 1

    with torch.no_grad():
        for _ in range(args.num_batches):
            batch, _metadata = next(iterator)
            batch = batch.to(device)
            true_actions = batch.actions.slice_time(off, swm.n_action_steps + off)
            shifted_actions = true_actions.clone()
            shifted_actions.key_presses = shifted_actions.key_presses.roll(1, dims=0)
            shifted_actions.mouse_movements = shifted_actions.mouse_movements.roll(1, dims=0)
            shifted_actions.game_mouse_sensitivity = shifted_actions.game_mouse_sensitivity.roll(1, dims=0)

            encoded_true = swm.action_encoder(true_actions)
            encoded_shifted = swm.action_encoder(shifted_actions)
            projected_true = model._project_player_actions(encoded_true)
            projected_shifted = model._project_player_actions(encoded_shifted)
            routed_true = model._combine_player_actions(encoded_true)
            routed_shifted = model._combine_player_actions(encoded_shifted)

            projected = _pair_stats(projected_true, projected_shifted)
            routed = _pair_stats(routed_true, routed_shifted)
            projected_rows.append(projected)
            routed_rows.append(routed)
            raw_rows.append(
                {
                    "keyboard_changed_fraction": (true_actions.key_presses != shifted_actions.key_presses)
                    .float()
                    .mean()
                    .item(),
                    "mouse_delta_rms": torch.sqrt(
                        torch.mean(
                            (
                                true_actions.mouse_movements.float() - shifted_actions.mouse_movements.float()
                            ).square()
                        )
                    ).item(),
                    "routing_attenuation": routed["delta_rms"]
                    / max(projected["delta_rms"], torch.finfo(torch.float32).eps),
                }
            )

    config_path = checkpoint.parents[1] / "world_model_config.yaml"
    payload: dict[str, Any] = {
        "schema": "mira-cs2-multi-action-routing-diagnostic-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": sys.argv,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_config_sha256": _sha256(config_path),
        "code_commit": _git("rev-parse HEAD"),
        "code_status": _git("status --porcelain=v1"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "seed": args.seed,
        "split": args.split,
        "map_slug": args.map_slug,
        "group_mode": "synchronized",
        "num_batches": args.num_batches,
        "raw_pov_rows": args.num_batches * model.n_players,
        "n_players": model.n_players,
        "checkpoint_action_routing": checkpoint_action_routing,
        "action_routing": model.action_routing,
        "projected_shape": list(rearrange(projected_true, "b p t d -> (b p) t d").shape),
        "routed_shape": list(routed_true.shape),
        "raw_intervention": summarize_rows(raw_rows),
        "projected_player_actions": summarize_rows(projected_rows),
        "routed_actions": summarize_rows(routed_rows),
        "interpretation": (
            "routing_attenuation compares routed cross-POV delta RMS with projected per-player "
            "delta RMS; lower values mean more player-specific conditioning was erased by routing"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
