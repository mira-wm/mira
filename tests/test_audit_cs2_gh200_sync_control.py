"""Focused fail-closed checks for the GH200 Dust2 auditor."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "audit_cs2_gh200_sync_control.py"
    spec = importlib.util.spec_from_file_location("audit_cs2_gh200_sync_control_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


AUDIT = _load_module()
COMMIT = "1" * 40


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _split_provenance(path: Path, manifest_hash: str) -> None:
    payload = {
        "source_manifest_sha256": AUDIT.EXPECTED_SOURCE_MANIFEST_SHA256,
        "output_manifest_sha256": manifest_hash,
        "confirmatory_selection_sha256": AUDIT.EXPECTED_SELECTION_SHA256,
        "map_slug": "dust2",
        "statistics": {
            "match_overlap": {"train:test": []},
            "splits": {
                split: expected | {"aligned_pov_hours": 1.0}
                for split, expected in AUDIT.EXPECTED_SPLITS.items()
            },
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _auditor(tmp_path: Path, monkeypatch) -> AUDIT.Auditor:
    manifest = tmp_path / "manifest.parquet"
    manifest.write_bytes(b"frozen manifest")
    monkeypatch.setattr(AUDIT, "EXPECTED_MANIFEST_SHA256", _sha256(manifest))
    split_provenance = tmp_path / "split.json"
    _split_provenance(split_provenance, _sha256(manifest))
    return AUDIT.Auditor(
        root=tmp_path / "seed_28",
        manifest=manifest,
        split_provenance=split_provenance,
        train_steps=10_000,
        arm_hours=12.0,
        expected_training_commit=COMMIT,
    )


def test_audit_dataset_accepts_frozen_disjoint_split(tmp_path: Path, monkeypatch) -> None:
    auditor = _auditor(tmp_path, monkeypatch)
    auditor.audit_dataset()

    assert auditor.checks["dataset.test.rounds"] == 69
    assert auditor.checks["dataset.pilot_test.rounds"] == 52


def test_audit_nodes_requires_one_full_four_gh200_node(tmp_path: Path, monkeypatch) -> None:
    auditor = _auditor(tmp_path, monkeypatch)
    loader = {
        "status": "pass",
        "expected_git_commit": COMMIT,
        "expected_gpu_substring": "GH200",
        "expected_manifest_sha256": AUDIT.EXPECTED_MANIFEST_SHA256,
        "selected_config": {
            "num_workers": 8,
            "prefetch_factor": 2,
            "persistent_workers": True,
            "pin_memory": True,
        },
    }
    launcher = {
        "seed": "28",
        "train_steps": "10000",
        "arm_hours": "12.0",
        "arm_order": "synchronized,shuffled",
        "nnodes": "1",
        "nproc_per_node": "4",
        "master_addr": "nid00000",
        "master_port": "29500",
        "scheduler": "slurm",
        "slurm_job_id": "12345",
        "slurm_job_nodelist": "nid00000",
        "slurm_cpus_per_task": "288",
        "logical_cpus_visible": "288",
        "manifest_sha256": AUDIT.EXPECTED_MANIFEST_SHA256,
        "split_provenance_sha256": _sha256(auditor.split_provenance),
        "action_routing": "spatial",
        "dataloader_workers": "8",
        "dataloader_prefetch_factor": "2",
        "dataloader_persistent_workers": "true",
        "dataloader_pin_memory": "true",
    }
    hostname = "nid00000"
    global_loader = loader | {
        "schema": "mira-cs2-frozen-loader-selection-v1",
        "expected_hostname": None,
        "benchmark_host_repeat_counts": {hostname: 3},
        "benchmark_inputs": [
            {
                "hostname": hostname,
                "manifest_sha256": AUDIT.EXPECTED_MANIFEST_SHA256,
            }
            for _ in range(3)
        ],
        "selection_evidence": {
            "rule": AUDIT.GLOBAL_LOADER_SELECTION_RULE,
            "selected_num_workers": 8,
        },
    }
    global_loader_text = json.dumps(global_loader, sort_keys=True)
    global_loader_sha256 = hashlib.sha256(global_loader_text.encode()).hexdigest()
    node = auditor.root / "provenance" / "node_0"
    node.mkdir(parents=True)
    (node / "code_commit.txt").write_text(COMMIT + "\n", encoding="utf-8")
    (node / "code_status.txt").write_text("", encoding="utf-8")
    (node / "code.patch").write_text("", encoding="utf-8")
    (node / "nvidia_smi_q.txt").write_text(
        "Product Name : NVIDIA GH200\n" * 4,
        encoding="utf-8",
    )
    (node / "gpu_timeseries.csv").write_text("header\nsample\n", encoding="utf-8")
    node_loader = loader | {
        "expected_hostname": hostname,
        "benchmark_host_repeat_counts": {hostname: 3},
        "benchmark_inputs": [
            {
                "hostname": hostname,
                "manifest_sha256": AUDIT.EXPECTED_MANIFEST_SHA256,
            }
            for _ in range(3)
        ],
    }
    node_loader_text = json.dumps(node_loader)
    node_launcher = launcher | {
        "node_rank": "0",
        "hostname": hostname,
        "visible_gpu_count": "4",
        "visible_gpu_name": "NVIDIA GH200 480GB",
        "visible_gpu_names": "|".join(["NVIDIA GH200 480GB"] * 4),
        "slurm_procid": "0",
        "slurm_nodeid": "0",
        "slurm_localid": "0",
        "global_loader_selection_sha256": global_loader_sha256,
        "frozen_loader_selection_sha256": hashlib.sha256(node_loader_text.encode()).hexdigest(),
    }
    (node / "launcher.env").write_text(
        "".join(f"{key}={value}\n" for key, value in node_launcher.items()),
        encoding="utf-8",
    )
    (node / "frozen_loader_selection.json").write_text(
        node_loader_text,
        encoding="utf-8",
    )
    (node / "global_loader_selection.json").write_text(
        global_loader_text,
        encoding="utf-8",
    )

    commit, selected, seed, order = auditor.audit_nodes()

    assert commit == COMMIT
    assert selected["num_workers"] == 8
    assert seed == 28
    assert order == ("synchronized", "shuffled")

    mismatched_global = global_loader | {
        "selected_config": global_loader["selected_config"] | {"num_workers": 12},
    }
    mismatched_text = json.dumps(mismatched_global, sort_keys=True)
    (node / "global_loader_selection.json").write_text(mismatched_text)
    node_launcher_text = (node / "launcher.env").read_text()
    node_launcher_text = node_launcher_text.replace(
        f"global_loader_selection_sha256={global_loader_sha256}",
        f"global_loader_selection_sha256={hashlib.sha256(mismatched_text.encode()).hexdigest()}",
    )
    (node / "launcher.env").write_text(node_launcher_text)
    with pytest.raises(ValueError, match="node_0.global_loader_selected_config"):
        auditor.audit_nodes()

    (node / "global_loader_selection.json").write_text(global_loader_text)
    (node / "launcher.env").write_text(
        node_launcher_text.replace(
            f"global_loader_selection_sha256={hashlib.sha256(mismatched_text.encode()).hexdigest()}",
            f"global_loader_selection_sha256={global_loader_sha256}",
        )
    )
    (node / "gpu_timeseries.csv").write_text("header\n", encoding="utf-8")
    with pytest.raises(ValueError, match="node_0.telemetry"):
        auditor.audit_nodes()


def test_audit_nodes_accepts_reused_selection_with_current_smoke(
    tmp_path: Path, monkeypatch
) -> None:
    auditor = _auditor(tmp_path, monkeypatch)
    hostname = "nid-new"
    selected = {
        "num_workers": 12,
        "prefetch_factor": 2,
        "persistent_workers": True,
        "pin_memory": True,
    }
    global_loader = {
        "schema": "mira-cs2-frozen-loader-selection-v1",
        "status": "pass",
        "expected_git_commit": "2" * 40,
        "expected_gpu_substring": "GH200",
        "expected_hostname": None,
        "expected_manifest_sha256": AUDIT.EXPECTED_MANIFEST_SHA256,
        "benchmark_host_repeat_counts": {"nid-old": 3},
        "benchmark_inputs": [
            {
                "hostname": "nid-old",
                "manifest_sha256": AUDIT.EXPECTED_MANIFEST_SHA256,
            }
            for _ in range(3)
        ],
        "selected_config": selected,
        "selection_evidence": {
            "rule": AUDIT.GLOBAL_LOADER_SELECTION_RULE,
            "selected_num_workers": 12,
        },
    }
    global_text = json.dumps(global_loader, sort_keys=True)
    global_hash = hashlib.sha256(global_text.encode()).hexdigest()
    smoke = {
        "schema": "mira-cs2-dataloader-benchmark-v1",
        "status": "pass",
        "provenance": {
            "git_clean": True,
            "git_commit": COMMIT,
            "hostname": hostname,
            "manifest_sha256": AUDIT.EXPECTED_MANIFEST_SHA256,
            "gpu": "NVIDIA GH200 120GB",
        },
        "results": [
            {"group_mode": mode, "status": "pass"}
            for mode in ("synchronized", "shuffled")
        ],
    }
    smoke_text = json.dumps(smoke, sort_keys=True)
    smoke_hash = hashlib.sha256(smoke_text.encode()).hexdigest()
    node_loader = {
        "schema": "mira-cs2-reused-loader-selection-v1",
        "status": "pass",
        "expected_git_commit": COMMIT,
        "expected_gpu_substring": "GH200",
        "expected_hostname": hostname,
        "expected_manifest_sha256": AUDIT.EXPECTED_MANIFEST_SHA256,
        "selected_config": selected,
        "source_selection": {
            "sha256": global_hash,
            "git_commit": "2" * 40,
            "benchmark_host_repeat_counts": {"nid-old": 3},
            "benchmark_input_count": 3,
        },
        "current_node_smoke": {"sha256": smoke_hash},
    }
    node_loader_text = json.dumps(node_loader, sort_keys=True)
    node = auditor.root / "provenance" / "node_0"
    node.mkdir(parents=True)
    (node / "code_commit.txt").write_text(COMMIT + "\n", encoding="utf-8")
    (node / "code_status.txt").write_text("", encoding="utf-8")
    (node / "code.patch").write_text("", encoding="utf-8")
    (node / "nvidia_smi_q.txt").write_text(
        "Product Name : NVIDIA GH200\n" * 4,
        encoding="utf-8",
    )
    (node / "gpu_timeseries.csv").write_text("header\nsample\n", encoding="utf-8")
    (node / "global_loader_selection.json").write_text(global_text, encoding="utf-8")
    (node / "loader_smoke.json").write_text(smoke_text, encoding="utf-8")
    (node / "frozen_loader_selection.json").write_text(node_loader_text, encoding="utf-8")
    launcher = {
        "seed": "28",
        "train_steps": "10000",
        "arm_hours": "12.0",
        "arm_order": "synchronized,shuffled",
        "nnodes": "1",
        "nproc_per_node": "4",
        "node_rank": "0",
        "hostname": hostname,
        "master_addr": hostname,
        "master_port": "29500",
        "scheduler": "slurm",
        "slurm_job_id": "12345",
        "slurm_job_nodelist": hostname,
        "slurm_procid": "0",
        "slurm_nodeid": "0",
        "slurm_localid": "0",
        "slurm_cpus_per_task": "288",
        "logical_cpus_visible": "288",
        "visible_gpu_count": "4",
        "visible_gpu_name": "NVIDIA GH200 120GB",
        "visible_gpu_names": "|".join(["NVIDIA GH200 120GB"] * 4),
        "manifest_sha256": AUDIT.EXPECTED_MANIFEST_SHA256,
        "split_provenance_sha256": _sha256(auditor.split_provenance),
        "action_routing": "spatial",
        "dataloader_workers": "12",
        "dataloader_prefetch_factor": "2",
        "dataloader_persistent_workers": "true",
        "dataloader_pin_memory": "true",
        "loader_selection_mode": "reused",
        "loader_smoke_sha256": smoke_hash,
        "global_loader_selection_sha256": global_hash,
        "frozen_loader_selection_sha256": hashlib.sha256(
            node_loader_text.encode()
        ).hexdigest(),
    }
    (node / "launcher.env").write_text(
        "".join(f"{key}={value}\n" for key, value in launcher.items()),
        encoding="utf-8",
    )

    commit, observed, seed, order = auditor.audit_nodes()

    assert commit == COMMIT
    assert observed == selected
    assert seed == 28
    assert order == ("synchronized", "shuffled")


def _write_fixed_step_training_arm(
    auditor: AUDIT.Auditor,
    arm: str,
    loader: dict,
) -> None:
    stage = auditor.root / arm
    stage.mkdir(parents=True)
    config = {
        "run": {
            "seed": 28,
            "steps": auditor.train_steps,
            "batch_size": 1,
            "deterministic": True,
            "max_duration_hours": auditor.arm_hours,
            "continue_from": None,
        },
        "dataset": {
            "map_slug": "dust2",
            "n_players": 10,
            "group_mode": arm,
            "train_index": str(auditor.manifest),
            "test_index": str(auditor.manifest),
            "train_split": "train",
            "test_split": "val",
            "validation_group_mode": "synchronized",
        },
        "model": {
            "architecture": {
                "config": {
                    "action_routing": "spatial",
                }
            }
        },
        "dataloader": loader,
        "validation": {
            "val_first": True,
            "val_every": 1000,
            "val_n_samples": 40,
            "local_rollout_every": 1000,
            "local_rollout_seed": 37,
        },
        "optim": {"optimizer": {"lr": 1e-4}},
    }
    (stage / "world_model_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    rows = [
        {
            "kind": "train",
            "step": 0,
            "System/n_frames_processed": AUDIT.GLOBAL_FRAMES_PER_STEP,
        },
        {"kind": "validation", "step": 0},
        {"kind": "rollout_trace", "step": 0},
    ]
    (stage / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    trace = stage / "rollout_traces" / "step-000000000"
    trace.mkdir(parents=True)
    (trace / "metadata.json").write_text(
        json.dumps({"seed": 37, "split": "val", "action_routing": "spatial"}),
        encoding="utf-8",
    )
    (trace / "rollout.mp4").write_bytes(b"video")
    final_step = auditor.train_steps - 1
    checkpoint = stage / f"checkpoint-{final_step}" / "checkpoint.pth"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(f"{arm}-checkpoint".encode())
    termination = {
        "schema": "mira-cs2-fixed-step-termination-v1",
        "arm": arm,
        "termination": "fixed_step_complete",
        "optimizer_updates": auditor.train_steps,
        "final_step": final_step,
        "launcher_elapsed_wall_seconds": 1234,
        "max_duration_hours": auditor.arm_hours,
        "checkpoint": str(checkpoint.resolve()),
    }
    (stage / "training_termination.json").write_text(json.dumps(termination), encoding="utf-8")


def test_training_audit_requires_fixed_matched_updates_and_no_time_limit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    auditor = _auditor(tmp_path, monkeypatch)
    loader = {
        "num_workers": 8,
        "prefetch_factor": 2,
        "persistent_workers": True,
        "pin_memory": True,
    }
    for arm in AUDIT.ARMS:
        _write_fixed_step_training_arm(auditor, arm, loader)

    result = auditor.audit_training(seed=28, loader_config=loader)

    expected_frames = auditor.train_steps * AUDIT.GLOBAL_FRAMES_PER_STEP
    assert result["synchronized"]["optimizer_updates"] == auditor.train_steps
    assert result["shuffled"]["processed_frames"] == expected_frames

    metrics = auditor.root / "shuffled" / "metrics.jsonl"
    with metrics.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"kind": "time_limit", "step": 123}) + "\n")
    with pytest.raises(ValueError, match="shuffled.time_limit_count"):
        auditor.audit_training(seed=28, loader_config=loader)


def test_audits_complete_first_death_action_grid(tmp_path: Path, monkeypatch) -> None:
    auditor = _auditor(tmp_path, monkeypatch)
    root = auditor.root / "evaluation" / "synchronized_test_first_death_action_loss_seed_sweep"
    provenance = root / "provenance"
    provenance.mkdir(parents=True)
    (provenance / "evaluator_code_commit.txt").write_text(COMMIT + "\n")
    (provenance / "evaluator_code_status.txt").write_text("")
    (provenance / "evaluator_code.patch").write_text("")
    (provenance / "window_mode.txt").write_text("first-death\n")
    checkpoint_hashes = {
        "shuffled": "a" * 64,
        "synchronized": "b" * 64,
    }
    summary = {
        "contract": {
            "arm_a": "shuffled",
            "arm_b": "synchronized",
            "split": "test",
            "map_slug": "dust2",
            "deterministic": True,
            "seeds": list(AUDIT.EVAL_SEEDS),
            "action_modes": list(AUDIT.ACTION_MODES),
            "window_mode": "first-death",
            "action_routing": "spatial",
            "validation_raw_pov_rows_per_arm_per_seed": 690,
            "shuffled_checkpoint_sha256": checkpoint_hashes["shuffled"],
            "synchronized_checkpoint_sha256": checkpoint_hashes["synchronized"],
        }
    }
    (root / "summary.json").write_text(json.dumps(summary))
    for seed in AUDIT.EVAL_SEEDS:
        for arm in AUDIT.ARMS:
            arm_root = root / f"seed_{seed}" / arm
            arm_root.mkdir(parents=True)
            for mode in AUDIT.ACTION_MODES:
                payload = {
                    "split": "test",
                    "seed": seed,
                    "deterministic": True,
                    "map_slug": "dust2",
                    "group_mode": "synchronized",
                    "training_group_mode": arm,
                    "n_players": 10,
                    "action_routing": "spatial",
                    "window_mode": "first-death",
                    "action_mode": mode,
                    "checkpoint_sha256": checkpoint_hashes[arm],
                    "validation": {"total_raw_pov_rows": 690},
                    "results": {"test/loss_total": 0.5},
                }
                (arm_root / f"{mode}.json").write_text(json.dumps(payload))

    result = auditor._audit_action_evaluation(
        root,
        label="death_action",
        window_mode="first-death",
        evaluator_commit=COMMIT,
        checkpoint_hashes=checkpoint_hashes,
    )

    assert result.endswith("summary.json")
    (provenance / "window_mode.txt").write_text("midpoint\n")
    with pytest.raises(ValueError, match="death_action.provenance.window_mode"):
        auditor._audit_action_evaluation(
                root,
                label="death_action",
                window_mode="first-death",
                evaluator_commit=COMMIT,
                checkpoint_hashes=checkpoint_hashes,
            )
