from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.validate_cs2_reused_loader_selection import validate_reuse

MANIFEST_SHA = "3" * 64
OLD_COMMIT = "4" * 40
NEW_COMMIT = "5" * 40


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _selection() -> dict:
    inputs = [
        {
            "hostname": "nid-old",
            "manifest_sha256": MANIFEST_SHA,
            "gpu": "NVIDIA GH200 120GB",
        }
        for _ in range(3)
    ]
    return {
        "schema": "mira-cs2-frozen-loader-selection-v1",
        "status": "pass",
        "expected_git_commit": OLD_COMMIT,
        "expected_gpu_substring": "GH200",
        "expected_hostname": None,
        "expected_manifest_sha256": MANIFEST_SHA,
        "benchmark_host_repeat_counts": {"nid-old": 3},
        "benchmark_inputs": inputs,
        "selected_config": {
            "num_workers": 12,
            "prefetch_factor": 2,
            "persistent_workers": True,
            "pin_memory": True,
        },
    }


def _smoke() -> dict:
    return {
        "schema": "mira-cs2-dataloader-benchmark-v1",
        "status": "pass",
        "completed_at_utc": "2026-07-27T00:00:00Z",
        "provenance": {
            "git_clean": True,
            "git_commit": NEW_COMMIT,
            "hostname": "nid-new",
            "manifest_sha256": MANIFEST_SHA,
            "gpu": "NVIDIA GH200 120GB",
        },
        "config": {
            "group_modes": ["synchronized", "shuffled"],
            "workers": [12],
            "prefetch_factor": 2,
            "persistent_workers": True,
            "pin_memory": True,
            "warmup_batches": 1,
            "timed_batches": 2,
        },
        "results": [
            {
                "group_mode": mode,
                "status": "pass",
                "num_workers": 12,
                "transfer_device": "cuda",
            }
            for mode in ("synchronized", "shuffled")
        ],
    }


def test_validate_reuse_preserves_old_evidence_and_requires_current_smoke(tmp_path: Path) -> None:
    selection = _write(tmp_path / "selection.json", _selection())
    smoke = _write(tmp_path / "smoke.json", _smoke())

    payload = validate_reuse(
        selection,
        smoke,
        expected_current_commit=NEW_COMMIT,
        expected_manifest_sha256=MANIFEST_SHA,
        expected_hostname="nid-new",
    )

    assert payload["status"] == "pass"
    assert payload["selected_config"]["num_workers"] == 12
    assert payload["source_selection"]["git_commit"] == OLD_COMMIT
    assert payload["current_node_smoke"]["git_commit"] == NEW_COMMIT


def test_validate_reuse_rejects_current_node_drift(tmp_path: Path) -> None:
    selection = _write(tmp_path / "selection.json", _selection())
    smoke_payload = _smoke()
    smoke_payload["provenance"]["hostname"] = "wrong-node"
    smoke = _write(tmp_path / "smoke.json", smoke_payload)

    with pytest.raises(ValueError, match="expected nid-new"):
        validate_reuse(
            selection,
            smoke,
            expected_current_commit=NEW_COMMIT,
            expected_manifest_sha256=MANIFEST_SHA,
            expected_hostname="nid-new",
        )
