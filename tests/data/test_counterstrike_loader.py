import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from mira.data.counterstrike import CS2_ACTION_DTYPE, CS2_KEYS
from mira.data.counterstrike_benchmark import (
    assess_decoded_video_parity,
    batch_signature,
    collect_provenance,
    validate_batch_contract,
    verify_single_synchronized_parity,
)
from mira.data.training_loader import create_loader


def _fixture(root: Path) -> Path:
    (root / "videos" / "360p").mkdir(parents=True)
    (root / "actions").mkdir()
    (root / "events").mkdir()
    manifest = []
    for round_idx in range(12):
        match_id = f"{round_idx:012d}"
        round_id = f"match_{match_id}__r001"
        for pov_idx in range(10):
            sample_key = f"{round_id}__p{pov_idx:02d}"
            manifest.append(
                {
                    "sample_key": sample_key,
                    "match_id": match_id,
                    "round_id": round_id,
                    "pov_idx": pov_idx,
                    "split": "train",
                    "map_slug": "dust2",
                    "frames": 32,
                    "frame0_tick": 100,
                    "alive_end_frame": 32,
                    "fps": 32.0,
                }
            )
            (root / "videos" / "360p" / f"{sample_key}.mp4").touch()
            actions = np.zeros(32, dtype=CS2_ACTION_DTYPE)
            actions["delta_yaw"] = 1.0
            actions["delta_pitch"] = -0.5
            actions["buttons"][::4] = 1 << 0
            actions["buttons"][1::4] = 1 << 1
            actions.tofile(root / "actions" / f"{sample_key}.actions.bin")
            (root / "events" / f"{sample_key}.events.json").write_text(
                json.dumps({"events": [{"type": "player_death", "frame_idx": 10}]})
            )
    pq.write_table(pa.Table.from_pylist(manifest), root / "manifest_dust2.parquet")
    return root


def _loader(root: Path, *, mode: str, n_players: int, window_mode: str = "midpoint"):
    return create_loader(
        root,
        dataset_backend="counterstrike1k",
        split="train",
        map_slug="dust2",
        group_mode=mode,
        window_mode=window_mode,
        clip_len=2,
        target_fps=8,
        n_players=n_players,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        infinite=False,
        frame_size=(16, 28),
        valid_keys=list(CS2_KEYS),
        pin_memory=False,
    )


def test_counterstrike_action_aggregation_and_video_subdir(tmp_path, monkeypatch) -> None:
    root = _fixture(tmp_path)
    monkeypatch.setattr(
        "mira.data.counterstrike.decode_frames",
        lambda _path, indices, frame_size: torch.zeros(len(indices), 3, *frame_size, dtype=torch.uint8),
    )

    batch, metadata = next(iter(_loader(root, mode="single", n_players=1)))

    assert batch.video.shape == (1, 2, 3, 16, 28)
    assert batch.actions.key_presses.shape == (1, 2, len(CS2_KEYS))
    assert batch.actions.key_presses[0, :, :2].tolist() == [[1, 1], [1, 1]]
    assert batch.actions.mouse_movements[0].tolist() == [[4.0, -2.0], [4.0, -2.0]]
    assert torch.isnan(batch.actions.game_mouse_sensitivity).all()
    assert metadata[0].sample_key.endswith("__p00")
    assert metadata[0].source_start_frame == 11
    assert metadata[0].frame_indices == [11, 15]
    assert metadata[0].alive_end_frame == 32


def test_midpoint_window_stays_inside_each_pov_alive_interval(
    tmp_path,
    monkeypatch,
) -> None:
    root = _fixture(tmp_path)
    manifest_path = root / "manifest_dust2.parquet"
    table = pq.read_table(manifest_path)
    rows = table.to_pylist()
    rows[0]["alive_end_frame"] = 20
    pq.write_table(pa.Table.from_pylist(rows), manifest_path)
    monkeypatch.setattr(
        "mira.data.counterstrike.decode_frames",
        lambda _path, indices, frame_size: torch.zeros(
            len(indices),
            3,
            *frame_size,
            dtype=torch.uint8,
        ),
    )

    _batch, metadata = next(iter(_loader(root, mode="single", n_players=1)))

    assert metadata[0].sample_key.endswith("__p00")
    assert metadata[0].source_start_frame == 5
    assert max(metadata[0].frame_indices) < metadata[0].alive_end_frame


def test_counterstrike_actions_start_one_row_after_observation(
    tmp_path,
    monkeypatch,
) -> None:
    root = _fixture(tmp_path)
    sample_key = "match_000000000000__r001__p00"
    action_path = root / "actions" / f"{sample_key}.actions.bin"
    actions = np.zeros(32, dtype=CS2_ACTION_DTYPE)
    # The first-death fixture starts at observation frame 6. Row 6 is poison:
    # target-aligned reduction must use rows 7..10, never the observation row.
    actions[6]["buttons"] = 1 << 11
    actions[6]["delta_yaw"] = 1000.0
    actions[7]["buttons"] = 1 << 0
    actions[10]["buttons"] = 1 << 7
    actions[7:11]["delta_yaw"] = 1.0
    actions[7:11]["delta_pitch"] = -0.5
    actions.tofile(action_path)
    monkeypatch.setattr(
        "mira.data.counterstrike.decode_frames",
        lambda _path, indices, frame_size: torch.zeros(
            len(indices),
            3,
            *frame_size,
            dtype=torch.uint8,
        ),
    )

    batch, metadata = next(
        iter(
            _loader(
                root,
                mode="single",
                n_players=1,
                window_mode="first-death",
            )
        )
    )

    assert metadata[0].source_start_frame == 6
    assert batch.actions.key_presses[0, 0, 0] == 1
    assert batch.actions.key_presses[0, 0, 7] == 1
    assert batch.actions.key_presses[0, 0, 11] == 0
    assert batch.actions.mouse_movements[0, 0].tolist() == [4.0, -2.0]


def test_synchronized_and_shuffled_controls(tmp_path, monkeypatch) -> None:
    root = _fixture(tmp_path)
    monkeypatch.setattr(
        "mira.data.counterstrike.decode_frames",
        lambda _path, indices, frame_size: torch.zeros(len(indices), 3, *frame_size, dtype=torch.uint8),
    )

    synced, synced_meta = next(iter(_loader(root, mode="synchronized", n_players=10)))
    shuffled, shuffled_meta = next(iter(_loader(root, mode="shuffled", n_players=10)))

    assert synced.video.shape[0] == shuffled.video.shape[0] == 10
    assert len({meta.round_id for meta in synced_meta}) == 1
    assert len({meta.source_start_frame for meta in synced_meta}) == 1
    assert len({meta.round_id for meta in shuffled_meta}) == 10
    assert len({meta.match_id for meta in shuffled_meta}) == 10


def test_first_death_windows_are_centered_and_paired_across_arms(tmp_path, monkeypatch) -> None:
    root = _fixture(tmp_path)
    monkeypatch.setattr(
        "mira.data.counterstrike.decode_frames",
        lambda _path, indices, frame_size: torch.zeros(len(indices), 3, *frame_size, dtype=torch.uint8),
    )

    _single, single_meta = next(iter(_loader(root, mode="single", n_players=1, window_mode="first-death")))
    _shared, shared_meta = next(
        iter(_loader(root, mode="synchronized", n_players=10, window_mode="first-death"))
    )

    # Two target frames at 8 fps require eight source frames; frame 10 is centered at start 6.
    assert single_meta[0].source_start_frame == 6
    assert {meta.source_start_frame for meta in shared_meta} == {6}
    assert single_meta[0].round_id == shared_meta[0].round_id
    assert {meta.window_mode for meta in shared_meta} == {"first-death"}


def test_single_synchronized_tensor_parity_gate(tmp_path, monkeypatch) -> None:
    root = _fixture(tmp_path)

    def fake_decode(path, indices, frame_size):
        pov_idx = int(Path(path).stem.rsplit("__p", maxsplit=1)[1])
        return torch.full(
            (len(indices), 3, *frame_size),
            pov_idx,
            dtype=torch.uint8,
        )

    monkeypatch.setattr("mira.data.counterstrike.decode_frames", fake_decode)
    result = verify_single_synchronized_parity(
        root,
        clip_len=2,
        target_fps=8,
        frame_size=(16, 28),
    )

    assert result["status"] == "pass"
    assert len(result["sample_keys"]) == 10
    assert len(result["video_sha256"]) == 64
    assert result["source_start_frame"] == 6
    assert result["window_mode"] == "first-death"


def test_benchmark_provenance_pins_explicit_manifest(tmp_path) -> None:
    root = _fixture(tmp_path)
    explicit = root / "manifest_dust2_confirmatory_spatial_v1.parquet"
    explicit.write_bytes((root / "manifest_dust2.parquet").read_bytes())

    provenance = collect_provenance(root, manifest_path=explicit)

    assert provenance["manifest_path"] == str(explicit)
    assert len(provenance["manifest_sha256"]) == 64


def test_benchmark_contract_and_signature_fail_closed(tmp_path, monkeypatch) -> None:
    root = _fixture(tmp_path)
    monkeypatch.setattr(
        "mira.data.counterstrike.decode_frames",
        lambda _path, indices, frame_size: torch.zeros(len(indices), 3, *frame_size, dtype=torch.uint8),
    )
    batch, metadata = next(iter(_loader(root, mode="synchronized", n_players=10)))

    validate_batch_contract(batch, metadata, group_mode="synchronized", clip_len=2)
    signature = batch_signature(batch, metadata)
    assert len(signature["metadata_sha256"]) == 64
    assert len(signature["sample_keys"]) == 10

    metadata[0].source_start_frame += 1
    with pytest.raises(ValueError, match="one source start"):
        validate_batch_contract(batch, metadata, group_mode="synchronized", clip_len=2)


def test_decoded_video_parity_is_exact_by_default() -> None:
    reference = torch.arange(48, dtype=torch.uint8).reshape(1, 1, 3, 4, 4)

    exact = assess_decoded_video_parity(reference, reference.clone())
    assert exact["status"] == "pass"
    assert exact["observed"] == {
        "mean_abs": 0.0,
        "max_abs": 0,
        "fraction_different": 0.0,
    }
    assert exact["reference_sha256"] == exact["candidate_sha256"]

    changed = reference.clone()
    changed[..., 0, 0] += 1
    mismatch = assess_decoded_video_parity(reference, changed)
    assert mismatch["status"] == "fail"
    assert set(mismatch["violations"]) == {
        "mean_abs",
        "max_abs",
        "fraction_different",
    }


def test_decoded_video_parity_uses_declared_tolerances() -> None:
    reference = torch.zeros((1, 1, 3, 2, 2), dtype=torch.uint8)
    candidate = reference.clone()
    candidate[..., 0, 0] = 2

    result = assess_decoded_video_parity(
        reference,
        candidate,
        max_mean_abs=0.5,
        max_abs=2,
        max_fraction_different=0.25,
    )
    assert result["status"] == "pass"
    assert result["thresholds"] == {
        "max_mean_abs": 0.5,
        "max_abs": 2,
        "max_fraction_different": 0.25,
    }


def test_decoded_video_parity_rejects_shape_and_dtype_changes() -> None:
    reference = torch.zeros((1, 1, 3, 2, 2), dtype=torch.uint8)

    shape = assess_decoded_video_parity(reference, reference[:, :, :, :, :1])
    assert shape["status"] == "fail"
    assert "shape mismatch" in shape["error"]

    dtype = assess_decoded_video_parity(reference, reference.float())
    assert dtype["status"] == "fail"
    assert "dtype mismatch" in dtype["error"]


def test_persistent_worker_loader_configuration(tmp_path) -> None:
    root = _fixture(tmp_path)
    loader = create_loader(
        root,
        dataset_backend="counterstrike1k",
        split="train",
        map_slug="dust2",
        group_mode="single",
        clip_len=2,
        target_fps=8,
        n_players=1,
        batch_size=10,
        num_workers=1,
        shuffle=False,
        infinite=False,
        frame_size=(16, 28),
        valid_keys=list(CS2_KEYS),
        prefetch_factor=3,
        pin_memory=False,
        persistent_workers=True,
    )

    assert loader.num_workers == 1
    assert loader.prefetch_factor == 3
    assert loader.persistent_workers is True

    with pytest.raises(ValueError, match="requires num_workers"):
        create_loader(
            root,
            dataset_backend="counterstrike1k",
            split="train",
            map_slug="dust2",
            group_mode="single",
            clip_len=2,
            target_fps=8,
            n_players=1,
            num_workers=0,
            persistent_workers=True,
        )
