"""CounterStrike-1K materialized-video loader for MIRA training.

CounterStrike-1K stores one 360p MP4 and one packed action stream per POV.  The
``manifest.parquet``/``manifest_dust2.parquet`` rows share ``round_id`` and
``frame0_tick`` across the ten POVs, so equal source-frame indices are
synchronized.  This module deliberately supports three experimental units:

``single``
    One independently modelled POV per group.
``synchronized``
    Ten time-aligned POVs from the same round.
``shuffled``
    Ten POVs from different rounds, preserving the synchronized model's exact
    architecture, row/token count and action count while destroying cross-POV
    synchronization.  This is the matched-information rebuttal control.

The source records target-frame-aligned actions at 32 fps: action row ``i``
describes the transition into video frame ``i``.  For an emitted observation
at source frame ``t``, the following transition therefore aggregates rows
``t+1`` through ``t+4`` at the frozen 8 fps target rate.  Button states are
OR-reduced and angular deltas are summed over that interval.  Mouse
conditioning is ``(delta_yaw, delta_pitch)``: horizontal then vertical, with
the signs and degree units in the released action stream preserved.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from mira.world_model.actions_config import ActionConfig, ActionTensors

from .decode import decode_frames

CS2_KEYS = (
    "FORWARD",
    "BACK",
    "LEFT",
    "RIGHT",
    "JUMP",
    "DUCK",
    "WALK",
    "FIRE",
    "RIGHTCLICK",
    "RELOAD",
    "INSPECT",
    "USE",
)
CS2_SOURCE_FPS = 32
CS2_ACTION_TARGET_OFFSET = 1
CS2_ACTION_DTYPE = np.dtype(
    [
        ("tick", "<u4"),
        ("delta_pitch", "<f4"),
        ("delta_yaw", "<f4"),
        ("buttons", "<u2"),
    ]
)
GroupMode = Literal["single", "synchronized", "shuffled"]
CounterStrikeWindowMode = Literal["midpoint", "first-death"]


@dataclass(frozen=True)
class _ManifestRow:
    sample_key: str
    match_id: str
    round_id: str
    pov_idx: int
    frames: int
    frame0_tick: int
    alive_end_frame: int


@dataclass
class CounterStrikeClipMeta:
    """Provenance for one emitted POV row."""

    match_id: str
    perspective: int
    player_id: int
    clip_id: int
    chunk_idx: int
    frame_indices: list[int]
    round_id: str
    sample_key: str
    source_start_frame: int
    alive_end_frame: int
    group_mode: str
    window_mode: str


def _rank_and_world_size() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def _resolve_manifest(index_path: str | Path) -> tuple[Path, Path]:
    path = Path(index_path)
    if path.is_file():
        return path.parent, path

    preferred = path / "manifest_dust2.parquet"
    fallback = path / "manifest.parquet"
    if preferred.exists():
        return path, preferred
    if fallback.exists():
        return path, fallback
    raise FileNotFoundError(
        f"No manifest_dust2.parquet or manifest.parquet under CounterStrike-1K root {path}"
    )


def _read_rounds(
    index_path: str | Path,
    *,
    split: str,
    map_slug: str | None,
) -> tuple[Path, list[list[_ManifestRow]]]:
    """Read and validate complete, ten-POV round groups from the Parquet manifest."""
    try:
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "The CounterStrike-1K backend needs pyarrow; install the MIRA training dependencies."
        ) from exc

    root, manifest = _resolve_manifest(index_path)
    table = pq.read_table(
        manifest,
        columns=[
            "sample_key",
            "match_id",
            "round_id",
            "pov_idx",
            "split",
            "map_slug",
            "frames",
            "frame0_tick",
            "alive_end_frame",
            "fps",
        ],
    )
    mask = pc.equal(table["split"], split)  # pyright: ignore[reportAttributeAccessIssue]
    if map_slug is not None:
        mask = pc.and_(  # pyright: ignore[reportAttributeAccessIssue]
            mask,
            pc.equal(  # pyright: ignore[reportAttributeAccessIssue]
                table["map_slug"], map_slug
            ),
        )
    table = table.filter(mask)
    if table.num_rows == 0:
        raise ValueError(f"No CounterStrike-1K rows for split={split!r}, map_slug={map_slug!r}")

    fps_values = set(table["fps"].to_pylist())
    if fps_values != {float(CS2_SOURCE_FPS)}:
        raise ValueError(f"Expected CounterStrike-1K actions/video at 32 fps, found {fps_values}")

    by_round: dict[str, list[_ManifestRow]] = {}
    for raw in table.to_pylist():
        row = _ManifestRow(
            sample_key=str(raw["sample_key"]),
            match_id=str(raw["match_id"]),
            round_id=str(raw["round_id"]),
            pov_idx=int(raw["pov_idx"]),
            frames=int(raw["frames"]),
            frame0_tick=int(raw["frame0_tick"]),
            alive_end_frame=int(raw["alive_end_frame"]),
        )
        by_round.setdefault(row.round_id, []).append(row)

    rounds: list[list[_ManifestRow]] = []
    for round_id in sorted(by_round):
        rows = sorted(by_round[round_id], key=lambda row: row.pov_idx)
        if [row.pov_idx for row in rows] != list(range(10)):
            # The synchronization experiment needs full ten-player rounds.  Incomplete
            # groups are skipped rather than silently changing the number of players.
            continue
        frame0_ticks = {row.frame0_tick for row in rows}
        if len(frame0_ticks) != 1:
            raise ValueError(f"POVs in {round_id} do not share frame0_tick: {frame0_ticks}")
        rounds.append(rows)

    if not rounds:
        raise ValueError(
            f"No complete ten-POV CounterStrike-1K rounds for split={split!r}, map_slug={map_slug!r}"
        )
    return root, rounds


class CounterStrike1KIterable(IterableDataset):
    """Yield model-ready rows from materialized CounterStrike-1K MP4/action files."""

    def __init__(
        self,
        index_path: str | Path,
        action_config: ActionConfig,
        *,
        split: str,
        map_slug: str | None,
        group_mode: GroupMode,
        window_mode: CounterStrikeWindowMode,
        clip_len: int,
        target_fps: int,
        n_players: int,
        frame_size: tuple[int, int] | None,
        shuffle: bool,
        infinite: bool,
        shuffle_buffer_size: int,
        seed: int,
    ) -> None:
        if target_fps > CS2_SOURCE_FPS or CS2_SOURCE_FPS % target_fps:
            raise ValueError(f"CounterStrike-1K requires target_fps to divide 32, got {target_fps}")
        if action_config.target_fps != target_fps:
            raise ValueError(
                "CounterStrike-1K emits one aggregated action per video frame; "
                f"action target_fps={action_config.target_fps} must equal video target_fps={target_fps}"
            )
        if tuple(action_config.valid_keys) != CS2_KEYS:
            raise ValueError(
                "CounterStrike-1K action vocabulary/order must be exactly "
                f"{list(CS2_KEYS)}, got {action_config.valid_keys}"
            )
        expected_players = 1 if group_mode == "single" else 10
        if n_players != expected_players:
            raise ValueError(
                f"group_mode={group_mode!r} requires n_players={expected_players}, got {n_players}"
            )
        if shuffle_buffer_size < 1:
            raise ValueError("shuffle_buffer_size must be >= 1")

        self.root, self.rounds = _read_rounds(index_path, split=split, map_slug=map_slug)
        if group_mode == "shuffled" and len(self.rounds) < n_players:
            raise ValueError(f"Shuffled control needs at least {n_players} rounds, found {len(self.rounds)}")
        if window_mode != "midpoint" and shuffle:
            raise ValueError("Event-centered CounterStrike-1K windows require shuffle=False")
        if window_mode != "midpoint" and group_mode == "shuffled":
            raise ValueError("Event-centered windows are defined only for real synchronized rounds")

        self.action_config = action_config
        self.group_mode = group_mode
        self.window_mode = window_mode
        self.clip_len = clip_len
        self.target_fps = target_fps
        self.source_stride = CS2_SOURCE_FPS // target_fps
        self.required_source_frames = clip_len * self.source_stride
        self.n_players = n_players
        self.frame_size = frame_size
        self.shuffle = shuffle
        self.infinite = infinite
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed

        if self.group_mode == "single" and self.window_mode == "midpoint":
            longest_common = max(min(row.frames, row.alive_end_frame) for rows in self.rounds for row in rows)
        else:
            longest_common = max(
                min(
                    (row.frames if self.window_mode != "midpoint" else min(row.frames, row.alive_end_frame))
                    for row in rows
                )
                for rows in self.rounds
            )
        required_with_action_target = self.required_source_frames + CS2_ACTION_TARGET_OFFSET
        if required_with_action_target > longest_common:
            raise ValueError(
                f"Requested {clip_len} frames @ {target_fps} fps needs "
                f"{required_with_action_target} target-aligned source rows, but "
                "the longest complete synchronized "
                f"round has {longest_common}"
            )

    def _my_round_indices(self) -> list[int]:
        rank, world_size = _rank_and_world_size()
        info = get_worker_info()
        worker_id, num_workers = (info.id, info.num_workers) if info is not None else (0, 1)
        return list(range(len(self.rounds)))[rank::world_size][worker_id::num_workers]

    def _start(self, max_frames: int, rng: random.Random) -> int:
        max_start = max_frames - self.required_source_frames - CS2_ACTION_TARGET_OFFSET
        if max_start < 0:
            raise ValueError("Internal error: attempted to sample a clip from a short POV")
        # A fixed midpoint makes validation repeatable even though its iterator is infinite.
        return rng.randint(0, max_start) if self.shuffle else max_start // 2

    def _event_start(self, rows: list[_ManifestRow], max_frames: int) -> int | None:
        """Center a synchronized source window on the first preregistered round event."""
        if self.window_mode != "first-death":
            raise ValueError(f"Unsupported CounterStrike-1K window_mode={self.window_mode!r}")
        events_path = self.root / "events" / f"{rows[0].sample_key}.events.json"
        if not events_path.is_file():
            raise FileNotFoundError(f"Missing CounterStrike-1K events: {events_path}")
        raw = json.loads(events_path.read_text())
        events = raw.get("events", []) if isinstance(raw, dict) else raw
        anchors = sorted(
            int(event["frame_idx"])
            for event in events
            if event.get("type") == "player_death"
            and "frame_idx" in event
            and 0 <= int(event["frame_idx"]) < max_frames
        )
        if not anchors:
            return None
        max_start = max_frames - self.required_source_frames - CS2_ACTION_TARGET_OFFSET
        return min(max(anchors[0] - self.required_source_frames // 2, 0), max_start)

    def _plans_for_round(
        self,
        round_idx: int,
        *,
        epoch: int,
        rng: random.Random,
    ) -> Iterator[list[tuple[_ManifestRow, int]]]:
        rows = self.rounds[round_idx]
        if self.group_mode == "single":
            if self.window_mode != "midpoint":
                common_frames = min(row.frames for row in rows)
                if common_frames < self.required_source_frames + CS2_ACTION_TARGET_OFFSET:
                    return
                start = self._event_start(rows, common_frames)
                if start is None:
                    return
                for row in rows:
                    yield [(row, start)]
                return
            for row in rows:
                alive_frames = min(row.frames, row.alive_end_frame)
                if alive_frames < self.required_source_frames + CS2_ACTION_TARGET_OFFSET:
                    continue
                yield [(row, self._start(alive_frames, rng))]
            return

        if self.group_mode == "synchronized":
            common_frames = min(
                (row.frames if self.window_mode != "midpoint" else min(row.frames, row.alive_end_frame))
                for row in rows
            )
            if common_frames < self.required_source_frames + CS2_ACTION_TARGET_OFFSET:
                return
            shared_start = (
                self._start(common_frames, rng)
                if self.window_mode == "midpoint"
                else self._event_start(rows, common_frames)
            )
            if shared_start is None:
                return
            yield [(row, shared_start) for row in rows]
            return

        # POV slot p comes from a different round.  Rotating the offset each epoch
        # avoids one fixed cross-round pairing while remaining deterministic by seed.
        control_order = list(range(len(self.rounds)))
        random.Random(self.seed + 100_003 + epoch).shuffle(control_order)
        plans: list[tuple[_ManifestRow, int]] = []
        for pov_idx in range(self.n_players):
            source_round_idx = control_order[(round_idx + pov_idx) % len(self.rounds)]
            row = self.rounds[source_round_idx][pov_idx]
            alive_frames = min(row.frames, row.alive_end_frame)
            if alive_frames < self.required_source_frames + CS2_ACTION_TARGET_OFFSET:
                return
            plans.append((row, self._start(alive_frames, rng)))
        yield plans

    def _decode_sample(
        self,
        row: _ManifestRow,
        start: int,
        *,
        clip_id: int,
    ) -> dict:
        source_indices = start + np.arange(self.clip_len, dtype=np.int64) * self.source_stride
        video_path = self.root / "videos" / f"{row.sample_key}.mp4"
        if not video_path.exists():
            # The official materialization helper places resized videos under videos/360p.
            video_path = self.root / "videos" / "360p" / f"{row.sample_key}.mp4"
        actions_path = self.root / "actions" / f"{row.sample_key}.actions.bin"
        if not video_path.exists():
            raise FileNotFoundError(f"Missing materialized CounterStrike-1K video: {video_path}")
        if not actions_path.exists():
            raise FileNotFoundError(f"Missing CounterStrike-1K actions: {actions_path}")

        raw = np.fromfile(actions_path, dtype=CS2_ACTION_DTYPE)
        action_start = start + CS2_ACTION_TARGET_OFFSET
        end = action_start + self.required_source_frames
        if end > len(raw):
            raise ValueError(
                f"{actions_path} has {len(raw)} action records, but target-aligned "
                f"rows [{action_start}:{end}] were requested"
            )
        windows = raw[action_start:end].reshape(
            self.clip_len,
            self.source_stride,
        )
        button_masks = np.bitwise_or.reduce(windows["buttons"], axis=1)
        bit_indices = np.arange(len(CS2_KEYS), dtype=np.uint16)
        keys = ((button_masks[:, None] >> bit_indices[None, :]) & 1).astype(np.int32)
        mouse = np.stack(
            [
                windows["delta_yaw"].sum(axis=1),
                windows["delta_pitch"].sum(axis=1),
            ],
            axis=-1,
        ).astype(np.float32)

        actions = ActionTensors(config=self.action_config, batch_size=1)
        actions.key_presses = torch.from_numpy(keys).unsqueeze(0)
        actions.mouse_movements = torch.from_numpy(mouse).unsqueeze(0)
        # Angular deltas are already expressed in in-game degrees.  There is no
        # meaningful raw-device sensitivity to multiply in.
        actions.game_mouse_sensitivity = torch.full((1,), float("nan"), dtype=torch.float32)

        return {
            "video": decode_frames(video_path, source_indices.tolist(), self.frame_size),
            "actions": actions,
            "metadata": CounterStrikeClipMeta(
                match_id=row.match_id,
                perspective=row.pov_idx,
                player_id=row.pov_idx,
                clip_id=clip_id,
                chunk_idx=0,
                frame_indices=source_indices.tolist(),
                round_id=row.round_id,
                sample_key=row.sample_key,
                source_start_frame=start,
                alive_end_frame=row.alive_end_frame,
                group_mode=self.group_mode,
                window_mode=self.window_mode,
            ),
        }

    def __iter__(self) -> Iterator[dict]:
        rank, _ = _rank_and_world_size()
        info = get_worker_info()
        worker_id = info.id if info is not None else 0
        my_round_indices = self._my_round_indices()
        if not my_round_indices:
            return

        rng = random.Random(self.seed + rank * 1024 + worker_id)
        buffer: list[list[tuple[_ManifestRow, int]]] = []
        clip_id = 0

        def drain(group: list[tuple[_ManifestRow, int]]) -> Iterator[dict]:
            nonlocal clip_id
            for row, start in group:
                yield self._decode_sample(row, start, clip_id=clip_id)
            clip_id += 1

        epochs = count() if self.infinite else range(1)
        for epoch in epochs:
            round_order = list(my_round_indices)
            if self.shuffle:
                rng.shuffle(round_order)
            for round_idx in round_order:
                for group in self._plans_for_round(round_idx, epoch=epoch, rng=rng):
                    if not self.shuffle:
                        yield from drain(group)
                        continue
                    buffer.append(group)
                    if len(buffer) >= self.shuffle_buffer_size:
                        yield from drain(buffer.pop(rng.randrange(len(buffer))))

        if self.shuffle:
            rng.shuffle(buffer)
            for group in buffer:
                yield from drain(group)
