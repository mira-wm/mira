from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "audit_cs2_rebuttal_run.py"
    spec = importlib.util.spec_from_file_location("audit_cs2_rebuttal_run_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


AUDIT = _load_module()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def test_dataset_audit_accepts_pinned_dust2_selection(tmp_path: Path) -> None:
    splits = {
        split: {
            **values,
            "pov_hours_full": 1.0,
            "shared_timeline_hours": values["aligned_pov_hours"] / 10,
        }
        for split, values in AUDIT.EXPECTED_SPLITS.items()
    }
    _write_json(
        tmp_path / "provenance" / "dataset.json",
        {
            "map_slug": "dust2",
            "full_manifest_sha256": AUDIT.EXPECTED_FULL_MANIFEST_SHA256,
            "selection_sha256": AUDIT.EXPECTED_SELECTION_SHA256,
            "source_shards": 116,
            "verification": {"samples": 9410, "payload_files": 47050},
            "statistics": {
                "match_overlap": {"test:train": [], "test:val": [], "train:val": []},
                "splits": splits,
            },
        },
    )
    auditor = AUDIT.Auditor(tmp_path)
    auditor.audit_dataset()
    assert auditor.checks["dataset.train.pov_rows"] == 8350


def test_dataset_audit_rejects_selection_drift(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "provenance" / "dataset.json",
        {
            "map_slug": "dust2",
            "full_manifest_sha256": AUDIT.EXPECTED_FULL_MANIFEST_SHA256,
            "selection_sha256": "0" * 64,
        },
    )
    with pytest.raises(ValueError, match="dataset.selection_sha256"):
        AUDIT.Auditor(tmp_path).audit_dataset()


def test_pipeline_status_requires_exact_completed_sequence(tmp_path: Path) -> None:
    lines = [
        f"2026-07-24T{hour:02d}:00:00Z\t{stage}\t{state}"
        for hour, (stage, state) in enumerate(AUDIT.EXPECTED_PIPELINE_STATES)
    ]
    (tmp_path / "pipeline_status.tsv").write_text("\n".join(lines) + "\n")
    times = AUDIT.Auditor(tmp_path).audit_pipeline_status()
    assert times["pipeline.complete"].hour == 6


def test_pipeline_status_rejects_missing_shared_completion(tmp_path: Path) -> None:
    rows = AUDIT.EXPECTED_PIPELINE_STATES[:-2] + [("pipeline", "complete")]
    lines = [f"2026-07-24T{hour:02d}:00:00Z\t{stage}\t{state}" for hour, (stage, state) in enumerate(rows)]
    (tmp_path / "pipeline_status.tsv").write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="pipeline.states"):
        AUDIT.Auditor(tmp_path).audit_pipeline_status()


def test_finite_results_rejects_nan(tmp_path: Path) -> None:
    source = tmp_path / "result.json"
    with pytest.raises(ValueError, match="non-finite"):
        AUDIT._finite_results({"results": {"loss": float("nan")}}, source)


def _result_identity_payload() -> dict:
    return {
        "dataset_backend": "counterstrike1k",
        "split": "test",
        "map_slug": "dust2",
        "deterministic": True,
        "seed": 37,
        "n_players": 10,
        "group_mode": "synchronized",
        "training_group_mode": "synchronized",
        "action_mode": "batch-shifted",
        "window_mode": "first-death",
        "checkpoint_sha256": "b" * 64,
    }


def test_result_identity_binds_file_to_arm_seed_action_and_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "evaluation" / "seed_37" / "shared" / "batch-shifted.json"
    auditor = AUDIT.Auditor(tmp_path)
    auditor._audit_result_identity(
        check_prefix="event",
        path=path,
        payload=_result_identity_payload(),
        arm="shared",
        seed=37,
        action_mode="batch-shifted",
        window_mode="first-death",
        checkpoint_hashes={"single": "a" * 64, "shared": "b" * 64},
    )
    assert auditor.checks[f"event.{path.relative_to(tmp_path)}.checkpoint_sha256"] == {
        "expected": "b" * 64,
        "observed": "b" * 64,
    }


@pytest.mark.parametrize(
    ("field", "incorrect"),
    [
        ("seed", 38),
        ("action_mode", "true"),
        ("checkpoint_sha256", "c" * 64),
        ("group_mode", "shuffled"),
    ],
)
def test_result_identity_rejects_mislabeled_or_copied_result(
    tmp_path: Path,
    field: str,
    incorrect: Any,
) -> None:
    payload = _result_identity_payload()
    payload[field] = incorrect
    path = tmp_path / "evaluation" / "seed_37" / "shared" / "batch-shifted.json"

    with pytest.raises(ValueError, match=field):
        AUDIT.Auditor(tmp_path)._audit_result_identity(
            check_prefix="event",
            path=path,
            payload=payload,
            arm="shared",
            seed=37,
            action_mode="batch-shifted",
            window_mode="first-death",
            checkpoint_hashes={"single": "a" * 64, "shared": "b" * 64},
        )


def test_dataloader_contract_normalizes_legacy_defaults(tmp_path: Path) -> None:
    auditor = AUDIT.Auditor(tmp_path)
    legacy = {"dataloader": {"num_workers": 4, "shuffle_buffer_size": 100}}
    explicit = {
        "dataloader": {
            "num_workers": 4,
            "shuffle_buffer_size": 100,
            "prefetch_factor": 2,
            "persistent_workers": False,
            "pin_memory": None,
        }
    }

    assert auditor._dataloader_config(legacy) == auditor._dataloader_config(explicit)


def test_require_records_success_evidence_without_failure_label(tmp_path: Path) -> None:
    auditor = AUDIT.Auditor(tmp_path)
    optimizer = {"optimizer": {"lr": 1e-4}, "scheduler": {"warmup_steps": 500}}
    evidence = {"single": optimizer, "shared": optimizer}

    auditor.require("arms.identical_optimizer", evidence["single"] == evidence["shared"], evidence)

    assert auditor.checks["arms.identical_optimizer"] == evidence


def test_gpu_telemetry_parses_nvidia_smi_csv(tmp_path: Path) -> None:
    source = tmp_path / "provenance" / "gpu_timeseries.csv"
    source.parent.mkdir()
    source.write_text(
        "timestamp, index, name, utilization.gpu [%], utilization.memory [%], "
        "memory.used [MiB], memory.total [MiB], power.draw [W], temperature.gpu\n"
        "2026/07/24 12:00:05.000, 0, GPU, 100 %, 10 %, 5000 MiB, 96000 MiB, 200 W, 40\n"
        "2026/07/24 12:00:10.000, 0, GPU, 50 %, 5 %, 6000 MiB, 96000 MiB, 150 W, 39\n"
    )
    utc = timezone.utc
    times = {
        "single.running": datetime(2026, 7, 24, 11, 0, tzinfo=utc),
        "shared.running": datetime(2026, 7, 24, 12, 0, tzinfo=utc),
        "pipeline.complete": datetime(2026, 7, 24, 13, 0, tzinfo=utc),
    }
    result = AUDIT.Auditor(tmp_path).audit_gpu_telemetry(times)
    assert result["shared_peak_memory_mib"] == 6000
    assert result["single_trace_is_partial"] is True
