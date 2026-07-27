"""Tests for freezing a GH200 loader configuration from exact-contract benchmarks."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "validate_cs2_loader_selection.py"
    spec = importlib.util.spec_from_file_location("validate_cs2_loader_selection_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VALIDATE = _load_module()
COMMIT = "1" * 40
MANIFEST = "2" * 64


def _benchmark(
    path: Path,
    *,
    repeat: int,
    include_shuffled: bool = True,
    clean: bool = True,
    hostname: str | None = None,
) -> None:
    modes = ["synchronized", "shuffled"] if include_shuffled else ["synchronized"]
    payload = {
        "schema": "mira-cs2-dataloader-benchmark-v1",
        "status": "pass",
        "completed_at_utc": f"2026-07-25T00:00:{repeat:02d}+00:00",
        "parity": {"status": "pass"},
        "provenance": {
            "git_clean": clean,
            "git_commit": COMMIT,
            "gpu": "NVIDIA GH200 480GB",
            "hostname": hostname or f"node-{repeat}",
            "manifest_sha256": MANIFEST,
        },
        "config": {
            "split": "val",
            "map_slug": "dust2",
            "group_modes": modes,
            "clip_len": 16,
            "target_fps": 8,
            "frame_size": [168, 308],
            "transfer_device": "cuda",
            "shuffle": True,
        },
        "results": [
            {
                "status": "pass",
                "group_mode": mode,
                "num_workers": 8,
                "prefetch_factor": 2,
                "persistent_workers": True,
                "pin_memory": True,
                "timed_batches": 100,
                "transfer_device": "cuda",
                "input_frames_per_second": 1000.0 + repeat,
            }
            for mode in modes
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _repeats(root: Path) -> list[Path]:
    paths = [root / f"repeat_{index}.json" for index in range(3)]
    for index, path in enumerate(paths):
        _benchmark(path, repeat=index)
    return paths


def test_freezes_repeated_target_hardware_selection(tmp_path: Path) -> None:
    result = VALIDATE.validate_selection(
        _repeats(tmp_path),
        num_workers=8,
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=True,
        expected_git_commit=COMMIT,
        expected_manifest_sha256=MANIFEST,
    )

    assert result["status"] == "pass"
    assert result["selected_config"]["num_workers"] == 8
    assert result["throughput_input_frames_per_second"]["synchronized"]["n"] == 3
    assert result["throughput_input_frames_per_second"]["shuffled"]["mean"] == 1001.0
    assert len(result["benchmark_inputs"]) == 3
    assert result["expected_manifest_sha256"] == MANIFEST


def test_rejects_fewer_than_three_repeats(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least 3"):
        VALIDATE.validate_selection(
            _repeats(tmp_path)[:2],
            num_workers=8,
            prefetch_factor=2,
            persistent_workers=True,
            pin_memory=True,
            expected_git_commit=COMMIT,
        )


def test_rejects_missing_shuffled_contract(tmp_path: Path) -> None:
    paths = _repeats(tmp_path)
    _benchmark(paths[1], repeat=1, include_shuffled=False)

    with pytest.raises(ValueError, match="synchronized and shuffled"):
        VALIDATE.validate_selection(
            paths,
            num_workers=8,
            prefetch_factor=2,
            persistent_workers=True,
            pin_memory=True,
            expected_git_commit=COMMIT,
        )


def test_rejects_dirty_benchmark_source(tmp_path: Path) -> None:
    paths = _repeats(tmp_path)
    _benchmark(paths[2], repeat=2, clean=False)

    with pytest.raises(ValueError, match="source was dirty"):
        VALIDATE.validate_selection(
            paths,
            num_workers=8,
            prefetch_factor=2,
            persistent_workers=True,
            pin_memory=True,
            expected_git_commit=COMMIT,
        )


def test_rejects_duplicate_benchmark_evidence(tmp_path: Path) -> None:
    paths = _repeats(tmp_path)
    copied = tmp_path / "copied.json"
    copied.write_bytes(paths[0].read_bytes())

    with pytest.raises(ValueError, match="duplicate benchmark payload"):
        VALIDATE.validate_selection(
            [paths[0], paths[1], copied],
            num_workers=8,
            prefetch_factor=2,
            persistent_workers=True,
            pin_memory=True,
            expected_git_commit=COMMIT,
        )


def test_binds_repeats_to_current_hostname(tmp_path: Path) -> None:
    paths = [tmp_path / f"repeat_{index}.json" for index in range(3)]
    for index, path in enumerate(paths):
        _benchmark(path, repeat=index, hostname="nid00001")

    result = VALIDATE.validate_selection(
        paths,
        num_workers=8,
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=True,
        expected_git_commit=COMMIT,
        expected_hostname="nid00001",
        expected_host_count=1,
        minimum_repeats_per_hostname=3,
    )
    assert result["benchmark_host_repeat_counts"] == {"nid00001": 3}

    with pytest.raises(ValueError, match="benchmark hostname"):
        VALIDATE.validate_selection(
            paths,
            num_workers=8,
            prefetch_factor=2,
            persistent_workers=True,
            pin_memory=True,
            expected_git_commit=COMMIT,
            expected_hostname="nid00002",
        )


def test_auto_selection_uses_common_maximin_throughput(tmp_path: Path) -> None:
    paths = _repeats(tmp_path)
    for path in paths:
        payload = json.loads(path.read_text())
        for workers, synchronized_fps, shuffled_fps in (
            (4, 1200.0, 1190.0),
            (12, 1500.0, 900.0),
        ):
            for mode, throughput in (
                ("synchronized", synchronized_fps),
                ("shuffled", shuffled_fps),
            ):
                payload["results"].append(
                    {
                        "status": "pass",
                        "group_mode": mode,
                        "num_workers": workers,
                        "prefetch_factor": 2,
                        "persistent_workers": True,
                        "pin_memory": True,
                        "timed_batches": 100,
                        "transfer_device": "cuda",
                        "input_frames_per_second": throughput,
                    }
                )
        path.write_text(json.dumps(payload), encoding="utf-8")

    selected, evidence = VALIDATE.select_num_workers(
        paths,
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=True,
    )

    assert selected == 4
    assert evidence["candidates"]["4"]["score_minimum_group_mode_mean_input_fps"] == 1190.0
    assert evidence["candidates"]["12"]["score_minimum_group_mode_mean_input_fps"] == 900.0
