"""A PyTorch ``DataLoader`` over the Rocket League dataset, yielding model-ready batches.

This adapts :meth:`mira.data.RocketScienceDataset.iter_clips` (the streaming read path) into
the ``(VideoActionBatch, list[ClipMeta])`` batches the codec and world model consume. It does not
re-implement any reading, decoding, or action parsing — those live in :mod:`mira.data`.

Multiplayer: ``iter_clips(perspective="all")`` yields one clip per ``(match, chunk)`` carrying all
``P`` perspectives as ``(P, T, C, H, W)``, ordered by ``player_id``. The perspectives are flattened
into the batch dimension. With ``n_players > 1`` the ``n_players`` perspectives of a match are kept
**contiguous and player_id-ordered** so a downstream wrapper can ``rearrange("(b p) ... -> b p ...")``;
with ``n_players == 1`` each perspective is an independent row. The loader's ``batch_size`` is
therefore multiplied by ``n_players`` so a batch holds ``batch_size`` whole groups.
"""

from __future__ import annotations

import copy
import random
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Any, Literal

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from mira.world_model.actions_config import ActionConfig, ActionTensors, stack_action_tensors

from .actions import DEFAULT_RL_KEYS, KeyVocab
from .batch import VideoActionBatch
from .counterstrike import (
    CS2_KEYS,
    CS2_SOURCE_FPS,
    CounterStrike1KIterable,
    CounterStrikeWindowMode,
)
from .dataset import RocketScienceDataset


@dataclass
class ClipMeta:
    """Per-row provenance for one perspective of one clip (not read by the models).

    Attributes:
        match_id: The match this clip belongs to.
        perspective: Index of this perspective within the match's player_id-ordered perspectives.
        player_id: The player_id of this perspective.
        clip_id: Per-match running clip index (shared across the match's perspectives).
        chunk_idx: Position of the source chunk this clip was taken from.
        frame_indices: Chunk-local source-frame indices of the clip's steps.
    """

    match_id: str
    perspective: int
    player_id: int
    clip_id: int
    chunk_idx: int
    frame_indices: list[int]


def _rank_and_world_size() -> tuple[int, int]:
    """Distributed (rank, world_size), or (0, 1) when not running under ``torch.distributed``."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


class _VideoActionIterable(IterableDataset):
    """Streams per-perspective ``{"video", "actions", "metadata"}`` samples from the dataset.

    Sharding mirrors the source loader: shards are split across distributed ranks and then across
    DataLoader workers, so each ``(rank, worker)`` reads a disjoint set of shards. A group-level
    shuffle buffer mixes whole ``n_players`` groups while keeping each group's perspectives
    contiguous. With ``infinite=True`` the stream loops forever (re-shuffling shard order each epoch).
    """

    def __init__(
        self,
        index_path: str | Path,
        action_config: ActionConfig,
        *,
        clip_len: int,
        target_fps: int,
        n_players: int,
        exclude_replays: bool,
        frame_size: tuple[int, int] | None,
        shuffle: bool,
        infinite: bool,
        shuffle_buffer_size: int,
        seed: int,
        action_fps: int | None = None,
    ):
        if n_players < 1:
            raise ValueError(f"n_players must be >= 1, got {n_players}")
        self.index_path = Path(index_path)
        self.action_config = action_config
        self.vocab = KeyVocab(tuple(action_config.valid_keys))
        self.clip_len = clip_len
        self.target_fps = target_fps
        self.action_fps = action_fps
        self.n_players = n_players
        self.exclude_replays = exclude_replays
        self.frame_size = frame_size
        self.shuffle = shuffle
        self.infinite = infinite
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed

    def _my_shards(self, dataset: RocketScienceDataset) -> list[str]:
        """The shards this (rank, worker) is responsible for, split rank-first then worker-second."""
        rank, world_size = _rank_and_world_size()
        info = get_worker_info()
        worker_id, num_workers = (info.id, info.num_workers) if info is not None else (0, 1)

        all_shards = sorted({e.shard for e in dataset.index.entries})
        return all_shards[rank::world_size][worker_id::num_workers]

    def _plan_perspective(self, clip, p: int) -> dict[str, Any]:
        """A lightweight buffered element for one perspective: an undecoded-clip reference plus the
        perspective index. No pixels — decoding is deferred to ``_decode_sample`` at the yield point."""
        return {"clip": clip, "p": p}

    def _decode_sample(self, plan: dict[str, Any]) -> dict[str, Any]:
        """Decode one buffered plan into a model-ready per-perspective sample (drops the clip ref)."""
        clip, p = plan["clip"], plan["p"]

        actions = ActionTensors(config=self.action_config, batch_size=1)
        actions.key_presses = clip.actions[p].unsqueeze(0).to(torch.int32)  # (1, T, n_keys)
        n_steps = actions.key_presses.shape[1]
        actions.mouse_movements = torch.zeros((1, n_steps, 2), dtype=torch.float32)
        # Keyboard-only data has no mouse sensitivity; NaN is the encoder's "unknown" signal.
        actions.game_mouse_sensitivity = torch.full((1,), float("nan"), dtype=torch.float32)

        return {
            "video": clip.decode_perspective(p, self.frame_size),  # (T, C, H, W) uint8
            "actions": actions,
            "metadata": ClipMeta(
                match_id=clip.match_id,
                perspective=p,
                player_id=clip.player_ids[p],
                clip_id=clip.clip_id,
                chunk_idx=clip.chunk_idx,
                frame_indices=list(clip.frame_indices),
            ),
        }

    def _shard_view(self, dataset: RocketScienceDataset, shard: str) -> RocketScienceDataset:
        """A shard-scoped view of ``dataset``: same loaded match data, index restricted to one shard.

        ``iter_clips`` derives the shards it streams from ``index.entries``; the per-shard reading
        uses ``matches``/``_chunk_pos``. The view is a shallow copy whose ``index`` carries only this
        shard's entries, so the shared dataset (and its fully populated ``matches``) is left intact.
        """
        view = copy.copy(dataset)
        view.index = dataset.index.model_copy(update={"entries": self._entries_by_shard[shard]})
        return view

    def _groups(self, dataset: RocketScienceDataset, shard: str) -> Iterator[list[dict[str, Any]]]:
        """Yield groups of ``n_players`` contiguous per-perspective samples from one shard's clips."""
        dataset = self._shard_view(dataset, shard)
        for clip in dataset.iter_clips(
            clip_len=self.clip_len,
            target_fps=self.target_fps,
            exclude_replays=self.exclude_replays,
            decode=False,
            perspective="all",
            frame_size=self.frame_size,
            seed=self.seed,
            action_fps=self.action_fps,
            carry_video=True,
        ):
            assert clip.video_bytes is not None, "iter_clips must carry video for the training loader"
            p_data = len(clip.perspectives)
            if p_data % self.n_players != 0:
                raise ValueError(
                    f"clip has {p_data} perspectives, not divisible by n_players={self.n_players}"
                )
            for g in range(0, p_data, self.n_players):
                yield [self._plan_perspective(clip, p) for p in range(g, g + self.n_players)]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        rank, _ = _rank_and_world_size()
        info = get_worker_info()
        worker_id = info.id if info is not None else 0

        dataset = RocketScienceDataset.from_local(self.index_path, vocab=self.vocab)
        # Group entries by shard once so a single shard can be streamed at a time (for shard-order
        # shuffling); _shard_view scopes iter_clips to one shard's entries.
        self._entries_by_shard: dict[str, list] = {}
        for e in dataset.index.entries:
            self._entries_by_shard.setdefault(e.shard, []).append(e)

        my_shards = self._my_shards(dataset)
        if not my_shards:
            return  # this (rank, worker) has no shards; finish cleanly rather than hang

        rng = random.Random(self.seed + rank * 1024 + worker_id)
        buffer: list[list[dict[str, Any]]] = []

        def drained(group: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
            # Decode a group's perspectives here, on the yield path, so decode stays parallelized
            # across workers; emit them contiguously so the DataLoader batches them together.
            for plan in group:
                yield self._decode_sample(plan)

        for _epoch in count() if self.infinite else range(1):
            shard_order = list(my_shards)
            if self.shuffle:
                rng.shuffle(shard_order)
            for shard in shard_order:
                for group in self._groups(dataset, shard):
                    if not self.shuffle:
                        yield from drained(group)
                        continue
                    buffer.append(group)
                    if len(buffer) >= self.shuffle_buffer_size:
                        yield from drained(buffer.pop(rng.randrange(len(buffer))))

        if self.shuffle:  # drain whatever remains (only reached when not infinite)
            rng.shuffle(buffer)
            for group in buffer:
                yield from drained(group)


def _collate(samples: list[dict[str, Any]]) -> tuple[VideoActionBatch, list[ClipMeta]]:
    """Stack per-perspective samples into one ``VideoActionBatch`` plus its per-row metadata."""
    video = torch.stack([s["video"] for s in samples], dim=0)  # (B, T, C, H, W) uint8
    actions = stack_action_tensors([s["actions"] for s in samples])
    metadata = [s["metadata"] for s in samples]
    return VideoActionBatch(video=video, actions=actions), metadata


def create_loader(
    index_path: str | Path,
    *,
    dataset_backend: Literal["rocket_science", "counterstrike1k"] = "rocket_science",
    split: str = "train",
    map_slug: str | None = None,
    group_mode: Literal["single", "synchronized", "shuffled"] | None = None,
    window_mode: CounterStrikeWindowMode = "midpoint",
    clip_len: int = 16,
    target_fps: int = 10,
    n_players: int = 1,
    batch_size: int = 4,
    num_workers: int = 0,
    shuffle: bool = True,
    infinite: bool = True,
    shuffle_buffer_size: int = 100,
    seed: int = 2025,
    exclude_replays: bool = False,
    frame_size: tuple[int, int] | None = None,
    valid_keys: list[str] | None = None,
    source_fps: int = 20,
    action_fps: int | None = None,
    action_config: ActionConfig | None = None,
    prefetch_factor: int = 2,
    pin_memory: bool | None = None,
    persistent_workers: bool = False,
) -> DataLoader:
    """Build a ``DataLoader`` yielding ``(VideoActionBatch, list[ClipMeta])`` from a dataset index.

    The train/val split is selected by which index is passed (the dataset ships separate ``train/``
    and ``test/`` indices via ``from_hub(split=...)`` / ``from_local(dir)``); there is no random
    split here. Set ``exclude_replays=True`` for evaluation.

    Args:
        index_path: Path to a dataset directory or its ``index.json``.
        window_mode: CounterStrike-1K window selection. ``midpoint`` is the fixed held-out default;
            ``first-death`` centers each eligible synchronized round on its first death event.
        clip_len: Number of time steps per clip.
        target_fps: Frame rate the clips (and actions) are downsampled to.
        n_players: Perspectives grouped contiguously per row-block. ``1`` treats every perspective
            as an independent row; ``>1`` keeps that many perspectives of a match contiguous.
        batch_size: Number of groups per batch; the loader uses ``batch_size * n_players`` rows.
        num_workers: DataLoader worker processes.
        shuffle: Shuffle shard order and groups (via a group-level shuffle buffer).
        infinite: Loop the stream forever (typical for training).
        shuffle_buffer_size: Number of groups buffered before one is emitted at random.
        seed: Base RNG seed (offset per rank/worker).
        exclude_replays: Drop clips overlapping a goal-replay span.
        frame_size: Optional ``(H, W)`` to resize decoded frames to (native size if ``None``).
        valid_keys: Key vocabulary; defaults to ``DEFAULT_RL_KEYS``. Ignored if ``action_config`` is
            given.
        source_fps: Nominal recording fps stored in the built ``ActionConfig`` (RL recordings are
            ~20fps). This is metadata only: the authoritative action downsampling is done inside the
            dataset using each match's measured fps, so the per-match factor may differ. Ignored if
            ``action_config`` is given.
        action_fps: Action sample rate, decoupled from the frame ``target_fps``. ``None`` (default)
            keeps one action step per frame, the released default. When set to a multiple of
            ``target_fps`` (e.g. ``2 * target_fps``) each clip yields ``action_fps // target_fps``
            action steps per video frame, so ``key_presses`` has ``action_fps // target_fps × T``
            steps. The built ``ActionConfig.target_fps`` follows ``action_fps`` so the stored action
            rate matches the emitted steps.
        action_config: Explicit ``ActionConfig``; built from ``valid_keys`` + fps when ``None``.
        prefetch_factor: Per-worker prefetch (only used when ``num_workers > 0``).
        pin_memory: Pin host memory; defaults to whether CUDA is available.
        persistent_workers: Keep worker processes alive across iterator recreation. Requires
            ``num_workers > 0``.

    Returns:
        A ``DataLoader`` over the dataset.
    """
    if persistent_workers and num_workers == 0:
        raise ValueError("persistent_workers=True requires num_workers > 0")

    if dataset_backend == "counterstrike1k":
        valid_keys = list(CS2_KEYS) if valid_keys is None else valid_keys
        source_fps = CS2_SOURCE_FPS

    if action_config is None:
        action_config = ActionConfig(
            valid_keys=list(valid_keys) if valid_keys is not None else list(DEFAULT_RL_KEYS),
            source_fps=source_fps,
            # The stored action rate follows action_fps when decoupled, else the frame rate.
            target_fps=action_fps if action_fps is not None else target_fps,
        )

    if dataset_backend == "counterstrike1k":
        dataset = CounterStrike1KIterable(
            index_path,
            action_config,
            split=split,
            map_slug=map_slug,
            group_mode=group_mode or ("single" if n_players == 1 else "synchronized"),
            window_mode=window_mode,
            clip_len=clip_len,
            target_fps=target_fps,
            n_players=n_players,
            frame_size=frame_size,
            shuffle=shuffle,
            infinite=infinite,
            shuffle_buffer_size=shuffle_buffer_size,
            seed=seed,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size * n_players,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            pin_memory=torch.cuda.is_available() if pin_memory is None else pin_memory,
            persistent_workers=persistent_workers,
            collate_fn=_collate,
        )
    if dataset_backend != "rocket_science":
        raise ValueError(f"Unknown dataset_backend={dataset_backend!r}")
    if window_mode != "midpoint":
        raise ValueError("window_mode is supported only by dataset_backend='counterstrike1k'")

    # Fail loudly up front if no clip in the dataset can satisfy `clip_len`: otherwise every match is
    # skipped as "too long for its chunks" and, with `infinite=True`, the stream loops over an empty
    # epoch forever (a silent hang). Some-but-not-all fitting is fine -- the short matches are skipped
    # while streaming. This reads only index metadata (no video), so it is cheap to check once here.
    probe = RocketScienceDataset.from_local(index_path, vocab=KeyVocab(tuple(action_config.valid_keys)))
    longest = probe.max_clip_frames(target_fps)
    if clip_len > longest:
        raise ValueError(
            f"Requested clip_len={clip_len} @ {target_fps}fps but the longest clip that fits any "
            f"chunk in {index_path} is {longest} frames. Lower clip_len (for the world-model eval: "
            f"world_model_metrics.num_unrolled_frames / n_context_frames) or use longer-chunked data."
        )

    dataset = _VideoActionIterable(
        index_path,
        action_config,
        clip_len=clip_len,
        target_fps=target_fps,
        n_players=n_players,
        exclude_replays=exclude_replays,
        frame_size=frame_size,
        shuffle=shuffle,
        infinite=infinite,
        shuffle_buffer_size=shuffle_buffer_size,
        seed=seed,
        action_fps=action_fps,
    )

    return DataLoader(
        dataset,
        # A batch holds the n_players aligned perspectives of `batch_size` groups, concatenated along
        # the batch dim; a downstream wrapper chunks them back apart.
        batch_size=batch_size * n_players,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=torch.cuda.is_available() if pin_memory is None else pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=_collate,
    )
