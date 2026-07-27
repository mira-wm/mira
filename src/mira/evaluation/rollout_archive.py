"""Atomic, model-agnostic rollout archives for downstream dynamics metrics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ARRAY_SPECS = {
    "predictions_uint8": "eval_seed,action_mode,sample,time,height,width,channel",
    "ground_truth_uint8": "sample,time,height,width,channel",
    "context_last_uint8": "sample,height,width,channel",
    "conditioning_actions_model_float32": "action_mode,sample,time,model_action",
    "conditioning_actions_cs2_float32": "action_mode,sample,time,cs2_action",
    "valid_steps_bool": "sample,time",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frames_to_uint8(frames: torch.Tensor) -> np.ndarray:
    """Convert ``[..., C, H, W]`` frames in ``[0, 1]`` to uint8 NHWC."""
    if frames.ndim < 4 or frames.shape[-3] != 3:
        raise ValueError(f"expected [..., 3, H, W] frames, got {tuple(frames.shape)}")
    leading = frames.shape[:-3]
    height, width = frames.shape[-2:]
    flat = (
        frames.detach()
        .float()
        .cpu()
        .clamp(0, 1)
        .mul(255)
        .round()
        .byte()
        .reshape(-1, 3, height, width)
        .permute(0, 2, 3, 1)
        .numpy()
    )
    return flat.reshape(*leading, height, width, 3)


class RolloutArchiveWriter:
    """Incrementally write one paired evaluation into atomic ``.npy`` arrays."""

    def __init__(
        self,
        out_dir: Path,
        *,
        num_samples: int,
        eval_seeds: list[int],
        action_modes: list[str],
        rollout_steps: int,
        height: int,
        width: int,
        num_model_actions: int,
        num_cs2_actions: int,
    ) -> None:
        dimensions = (num_samples, len(eval_seeds), len(action_modes), rollout_steps)
        if min(dimensions) <= 0:
            raise ValueError("rollout archive dimensions must be positive")
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.eval_seeds = list(eval_seeds)
        self.action_modes = list(action_modes)
        self.num_samples = num_samples
        self.rollout_steps = rollout_steps
        self.height = height
        self.width = width
        self.num_model_actions = num_model_actions
        self.num_cs2_actions = num_cs2_actions
        self._final = {name: out_dir / f"{name}.npy" for name in ARRAY_SPECS}
        self._temporary = {name: out_dir / f".{name}.tmp.npy" for name in ARRAY_SPECS}
        occupied = [
            path
            for path in [out_dir / "metadata.json", *self._final.values(), *self._temporary.values()]
            if path.exists()
        ]
        if occupied:
            raise FileExistsError(f"rollout archive destination is not empty: {occupied[0]}")

        s, m, n, t = len(eval_seeds), len(action_modes), num_samples, rollout_steps
        h, w = height, width
        self.predictions = np.lib.format.open_memmap(
            self._temporary["predictions_uint8"],
            mode="w+",
            dtype=np.uint8,
            shape=(s, m, n, t, h, w, 3),
        )
        self.ground_truth = np.lib.format.open_memmap(
            self._temporary["ground_truth_uint8"],
            mode="w+",
            dtype=np.uint8,
            shape=(n, t, h, w, 3),
        )
        self.context_last = np.lib.format.open_memmap(
            self._temporary["context_last_uint8"],
            mode="w+",
            dtype=np.uint8,
            shape=(n, h, w, 3),
        )
        self.conditioning_actions_model = np.lib.format.open_memmap(
            self._temporary["conditioning_actions_model_float32"],
            mode="w+",
            dtype=np.float32,
            shape=(m, n, t, num_model_actions),
        )
        self.conditioning_actions_cs2 = np.lib.format.open_memmap(
            self._temporary["conditioning_actions_cs2_float32"],
            mode="w+",
            dtype=np.float32,
            shape=(m, n, t, num_cs2_actions),
        )
        self.valid_steps = np.lib.format.open_memmap(
            self._temporary["valid_steps_bool"],
            mode="w+",
            dtype=np.bool_,
            shape=(n, t),
        )
        self._base_written = np.zeros(n, dtype=np.bool_)
        self._actions_written = np.zeros((m, n), dtype=np.bool_)
        self._predictions_written = np.zeros((s, m, n), dtype=np.bool_)

    def write_batch(
        self,
        *,
        seed_index: int,
        mode_index: int,
        sample_positions: list[int],
        predictions: torch.Tensor,
        ground_truth: torch.Tensor,
        context_last: torch.Tensor,
        conditioning_actions_model: torch.Tensor,
        conditioning_actions_cs2: torch.Tensor,
        valid_steps: np.ndarray,
    ) -> None:
        positions = np.asarray(sample_positions, dtype=np.int64)
        if positions.ndim != 1 or not len(positions) or len(np.unique(positions)) != len(positions):
            raise ValueError("sample positions must be a unique one-dimensional list")
        if np.any(positions < 0) or np.any(positions >= self.num_samples):
            raise IndexError("sample position is outside the archive")
        b, t, h, w = len(positions), self.rollout_steps, self.height, self.width
        if tuple(predictions.shape) != (b, t, 3, h, w):
            raise ValueError("prediction shape differs from rollout archive")
        if tuple(ground_truth.shape) != (b, t, 3, h, w):
            raise ValueError("ground-truth shape differs from rollout archive")
        if tuple(context_last.shape) != (b, 3, h, w):
            raise ValueError("context shape differs from rollout archive")
        if tuple(conditioning_actions_model.shape) != (b, t, self.num_model_actions):
            raise ValueError("model-action shape differs from rollout archive")
        if tuple(conditioning_actions_cs2.shape) != (b, t, self.num_cs2_actions):
            raise ValueError("CS2-action shape differs from rollout archive")
        valid = np.asarray(valid_steps, dtype=np.bool_)
        if valid.shape != (b, t):
            raise ValueError("valid-step shape differs from rollout archive")
        if self._predictions_written[seed_index, mode_index, positions].any():
            raise ValueError("attempted to overwrite archived predictions")

        prediction_uint8 = frames_to_uint8(predictions)
        ground_truth_uint8 = frames_to_uint8(ground_truth)
        context_uint8 = frames_to_uint8(context_last)
        model_actions = conditioning_actions_model.detach().float().cpu().numpy()
        cs2_actions = conditioning_actions_cs2.detach().float().cpu().numpy()
        self.predictions[seed_index, mode_index, positions] = prediction_uint8
        self._predictions_written[seed_index, mode_index, positions] = True

        already = self._base_written[positions]
        if already.any():
            prior = positions[already]
            if (
                not np.array_equal(self.ground_truth[prior], ground_truth_uint8[already])
                or not np.array_equal(self.context_last[prior], context_uint8[already])
                or not np.array_equal(self.valid_steps[prior], valid[already])
            ):
                raise ValueError("receiver video or validity changed across paired modes")
        if (~already).any():
            new = positions[~already]
            self.ground_truth[new] = ground_truth_uint8[~already]
            self.context_last[new] = context_uint8[~already]
            self.valid_steps[new] = valid[~already]
            self._base_written[new] = True

        action_already = self._actions_written[mode_index, positions]
        if action_already.any():
            prior = positions[action_already]
            if not np.array_equal(
                self.conditioning_actions_model[mode_index, prior],
                model_actions[action_already],
            ) or not np.array_equal(
                self.conditioning_actions_cs2[mode_index, prior],
                cs2_actions[action_already],
            ):
                raise ValueError("conditioning actions changed across evaluation seeds")
        if (~action_already).any():
            new = positions[~action_already]
            self.conditioning_actions_model[mode_index, new] = model_actions[~action_already]
            self.conditioning_actions_cs2[mode_index, new] = cs2_actions[~action_already]
            self._actions_written[mode_index, new] = True

    def finalize(self, *, contract: dict) -> Path:
        if not self._base_written.all():
            raise ValueError("rollout archive is missing receiver videos")
        if not self._actions_written.all():
            raise ValueError("rollout archive is missing conditioned action streams")
        if not self._predictions_written.all():
            raise ValueError("rollout archive is missing predictions")
        arrays = (
            self.predictions,
            self.ground_truth,
            self.context_last,
            self.conditioning_actions_model,
            self.conditioning_actions_cs2,
            self.valid_steps,
        )
        for array in arrays:
            array.flush()
        del arrays
        del self.predictions
        del self.ground_truth
        del self.context_last
        del self.conditioning_actions_model
        del self.conditioning_actions_cs2
        del self.valid_steps

        artifacts = {}
        for name, final_path in self._final.items():
            self._temporary[name].replace(final_path)
            array = np.load(final_path, mmap_mode="r")
            artifacts[name] = {
                "path": final_path.name,
                "sha256": sha256_file(final_path),
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "axes": ARRAY_SPECS[name],
            }
        metadata = {
            "schema_version": 1,
            "status": "complete",
            "contract": {
                **contract,
                "num_samples": self.num_samples,
                "eval_seeds": self.eval_seeds,
                "action_modes": self.action_modes,
                "rollout_steps": self.rollout_steps,
                "height": self.height,
                "width": self.width,
                "num_model_actions": self.num_model_actions,
                "num_cs2_actions": self.num_cs2_actions,
                "pixel_range": "uint8_[0,255]",
            },
            "artifacts": artifacts,
        }
        metadata_path = self.out_dir / "metadata.json"
        temporary_metadata = self.out_dir / ".metadata.tmp.json"
        temporary_metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        temporary_metadata.replace(metadata_path)
        return metadata_path
