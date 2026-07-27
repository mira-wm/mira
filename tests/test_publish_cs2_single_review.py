from __future__ import annotations

import importlib.util
from datetime import UTC, datetime
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "publish_cs2_single_review.py"
    spec = importlib.util.spec_from_file_location("single_review_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PUBLISH = _load_module()


def test_safe_files_include_rollouts_but_never_checkpoints(tmp_path: Path) -> None:
    (tmp_path / "pipeline_status.tsv").write_text("time\tcodec\trunning\n")
    trace = tmp_path / "single" / "rollout_traces" / "step-000001000"
    trace.mkdir(parents=True)
    (trace / "rollout.mp4").write_bytes(b"video")
    (trace / "metadata.json").write_text("{}")
    checkpoint = tmp_path / "single" / "checkpoint-1000"
    checkpoint.mkdir()
    (checkpoint / "checkpoint.pth").write_bytes(b"secret")

    files = PUBLISH.safe_files(tmp_path)
    relative = {item[1] for item in files}

    assert "training-rollouts/step-000001000/rollout.mp4" in relative
    assert "training-rollouts/step-000001000/metadata.json" in relative
    assert not any("checkpoint" in item for item in relative)


def test_stage_progress_uses_frozen_step_endpoint(tmp_path: Path) -> None:
    single = tmp_path / "single"
    single.mkdir()
    (single / "metrics.jsonl").write_text('{"kind":"train","step":7500,"train/loss_total":0.4}\n')

    stage = PUBLISH.stage_manifest(tmp_path, "single", {"single": "running"})

    assert stage["endpoint_step"] == 15_000
    assert stage["progress_percent"] == 50.0


class _SigningClient:
    def generate_presigned_url(self, method, *, Params, ExpiresIn):
        assert method == "get_object"
        return f"https://signed.invalid/{Params['Key']}?expires={ExpiresIn}"


def test_control_fragment_merge_preserves_completed_endpoint_evidence() -> None:
    generated = datetime.now(UTC).isoformat(timespec="seconds")
    manifest = {
        "experiment": {"run_id": "single"},
        "stages": [{"name": "single", "status": "complete"}],
        "traces": [{"id": "single-generated", "videos": [{"url": "https://single.invalid"}]}],
        "dataset": {"test_rounds": 69},
        "telemetry": {"samples": 1},
        "provenance": {"training_commit": "single"},
        "artifacts": [{"label": "single-summary", "url": "https://single.invalid/summary"}],
        "privacy": {
            "training_state_uploaded": False,
            "browser_checkpoint_urls_issued": False,
        },
    }
    prefix = "mira-dust2/run/gh200"
    fragment = {
        "schema": PUBLISH.CONTROL_FRAGMENT_SCHEMA,
        "generated_at_utc": generated,
        "bucket": "private-bucket",
        "object_prefix": prefix,
        "experiment": {
            "run_id": "sync-control",
            "status": "training",
            "status_label": "Synchronized arm",
        },
        "stages": [{"name": "synchronized", "status": "running"}],
        "traces": [
            {
                "id": "sync-step-1000",
                "videos": [{"label": "rollout", "object_key": f"{prefix}/rollout.mp4"}],
                "metrics_object_key": f"{prefix}/metrics.json",
            }
        ],
        "dataset": {"test_rounds": 69, "grouping": "synchronized-vs-shuffled"},
        "telemetry": {"samples": 2},
        "provenance": {"training_commit": "control"},
        "artifacts": [{"label": "status", "object_key": f"{prefix}/status.tsv"}],
        "privacy": {
            "training_state_uploaded": False,
            "browser_checkpoint_urls_issued": False,
        },
    }

    merged = PUBLISH.merge_control_fragment(
        manifest,
        fragment,
        client=_SigningClient(),
        bucket="private-bucket",
        expires_seconds=600,
    )

    assert merged["experiment"]["run_id"] == "sync-control"
    assert merged["experiment"]["review_transport_status"] == "live"
    assert [trace["id"] for trace in merged["traces"]] == [
        "single-generated",
        "sync-step-1000",
    ]
    assert merged["traces"][1]["videos"][0]["url"].startswith("https://signed.invalid/")
    assert "object_key" not in merged["traces"][1]["videos"][0]
    assert merged["traces"][1]["metrics_url"].startswith("https://signed.invalid/")
    assert [artifact["label"] for artifact in merged["artifacts"]] == [
        "single-summary",
        "status",
    ]
