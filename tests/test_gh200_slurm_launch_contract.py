"""Static and fail-closed checks for the full-node Clariden GH200 entrypoints."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_gh200_shell_entrypoints_parse() -> None:
    scripts = [
        ROOT / "scripts" / "run_cs2_gh200_loader_preflight.sh",
        ROOT / "scripts" / "run_cs2_gh200_slurm_seed.sh",
        ROOT / "scripts" / "run_cs2_gh200_sync_control.sh",
        ROOT / "scripts" / "run_cs2_gh200_sync_control_eval.sh",
        ROOT / "scripts" / "run_cs2_g7e_sync_control_preflight.sh",
        ROOT / "scripts" / "run_cs2_frozen_event_probe_slurm_seed.sh",
        ROOT / "scripts" / "run_cs2_gh200_sweep_finalize.sh",
        ROOT / "scripts" / "submit_cs2_gh200_sweep.sh",
        ROOT / "scripts" / "setup_cs2_clariden_uenv.sh",
        ROOT / "scripts" / "prepare_and_submit_cs2_clariden_sweep.sh",
        ROOT / "scripts" / "run_cs2_clariden_dataset_stage.sh",
        ROOT / "scripts" / "submit_cs2_clariden_dataset_stage.sh",
        ROOT / "scripts" / "stage_cs2_frozen_endpoints.sh",
    ]
    subprocess.run(["bash", "-n", *map(str, scripts)], check=True)


def test_publication_launcher_rejects_non_slurm_invocation() -> None:
    env = {
        **os.environ,
        "CS1K_DATASET_DIR": "/nonexistent/data",
        "CS1K_MANIFEST_PATH": "/nonexistent/manifest.parquet",
        "CS1K_CONFIRMATORY_SPLIT_PROVENANCE": "/nonexistent/split.json",
        "CS1K_CODEC_CHECKPOINT": "/nonexistent/codec.pth",
        "CS1K_OUTPUT_ROOT": "/nonexistent/output",
        "CS1K_TRAIN_STEPS": "10000",
        "CS1K_ARM_HOURS": "12",
        "CS1K_DATALOADER_WORKERS": "8",
        "CS1K_DATALOADER_PREFETCH_FACTOR": "2",
        "CS1K_DATALOADER_PERSISTENT_WORKERS": "true",
        "CS1K_DATALOADER_PIN_MEMORY": "true",
        "CS1K_LOADER_BENCHMARK_ROOT": "/nonexistent/benchmarks",
        "CS1K_GLOBAL_LOADER_SELECTION": "/nonexistent/global-selection.json",
        "NODE_RANK": "0",
        "MASTER_ADDR": "nid00000",
    }
    for key in tuple(env):
        if key.startswith("SLURM_"):
            env.pop(key)
    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / "run_cs2_gh200_sync_control.sh")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Slurm allocation" in result.stderr


def test_slurm_orchestrator_requests_exact_one_node_four_gpu_topology() -> None:
    text = (ROOT / "scripts" / "run_cs2_gh200_slurm_seed.sh").read_text()
    preflight = (ROOT / "scripts" / "run_cs2_gh200_loader_preflight.sh").read_text()
    assert "--nodes=1" in text
    assert "--ntasks-per-node=1" in text
    assert "--gpus-per-task=4" in text
    assert "--cpus-per-task=288" in text
    assert "export NNODES=1" in text
    assert "export NPROC_PER_NODE=4" in text
    assert "import torch.distributed.run" in text
    assert "prepare_dinov2_cache.py" in text
    assert "CS1K_REUSE_LOADER_SELECTION" in text
    assert "validate_cs2_reused_loader_selection.py" in text
    assert "--warmup-batches 1" in text
    assert "--timed-batches 2" in text
    assert "--num-workers auto" in text
    assert "--expected-host-count 1" in text
    assert "--nodes=4" not in text
    assert '${#visible_gpu_names[@]} -ne 4' in preflight
    assert "visible_gpu_count=${#visible_gpu_names[@]}" in preflight
    assert "run_cs2_gh200_sync_control_eval.sh" in text
    assert "CS1K_TRAIN_STEPS" in text


def test_publication_launcher_uses_fixed_steps_and_external_hydra_output() -> None:
    text = (ROOT / "scripts" / "run_cs2_gh200_sync_control.sh").read_text()
    assert 'torchrun_command=("$python_bin" -m torch.distributed.run)' in text
    assert 'run.steps="$train_steps"' in text
    assert "run.steps=100000000" not in text
    assert 'hydra.run.dir="$experiment_root/hydra/$arm/node_$node_rank"' in text
    assert '"kind": "time_limit"' in text


def test_g7e_preflight_uses_the_same_fixed_step_endpoint_contract() -> None:
    text = (ROOT / "scripts" / "run_cs2_g7e_sync_control_preflight.sh").read_text()
    assert 'run.steps="$train_steps"' in text
    assert "run.steps=100000000" not in text
    assert "checkpoint-$((train_steps - 1))/checkpoint.pth" in text
    assert 'hydra.run.dir="$output_root/hydra/$arm"' in text


def test_complete_sweep_submitter_builds_fail_closed_dependency_dag() -> None:
    text = (ROOT / "scripts" / "submit_cs2_gh200_sweep.sh").read_text()
    assert 'CS1K_TRAINING_SEEDS:-"28 29 30"' in text
    assert '--nodes=1' in text
    assert '--gpus-per-node=4' in text
    assert '--cpus-per-task=288' in text
    assert '--exclusive' in text
    assert '--nodes=4' not in text
    assert '--dependency="afterok:$training_job_id"' in text
    assert '--dependency="afterok:$event_dependency"' in text
    assert "run_cs2_frozen_event_probe_slurm_seed.sh" in text
    assert "run_cs2_gh200_sweep_finalize.sh" in text
    assert "submission_manifest.json" in text
    assert "CS1K_EXPECTED_CODEC_CHECKPOINT_SHA256" in text
    assert "codec_checkpoint_sha256" in text
    assert "loader_selection_reuse" in text


def test_posttrain_recovery_reuses_checkpoints_without_training_or_preflight() -> None:
    worker = (ROOT / "scripts" / "run_cs2_gh200_posttrain_eval_slurm_seed.sh").read_text()
    submit = (ROOT / "scripts" / "submit_cs2_gh200_posttrain_recovery.sh").read_text()
    prepare = (
        ROOT / "scripts" / "prepare_and_submit_cs2_clariden_posttrain_recovery.sh"
    ).read_text()

    assert "checkpoint-$final_step/checkpoint.pth" in worker
    assert "training_termination.json" in worker
    assert "run_cs2_gh200_sync_control_eval.sh" in worker
    assert "run_cs2_gh200_sync_control.sh" not in worker
    assert "run_cs2_gh200_loader_preflight.sh" not in worker
    assert "CS1K_EXPECTED_TRAINING_COMMIT" in worker
    assert "CS1K_EXPECTED_MIRA_COMMIT" in worker
    assert '--dependency="afterok:$eval_job_id"' in submit
    assert '--dependency="afterok:$event_dependency"' in submit
    assert "posttrain_recovery_manifest.json" in submit
    assert "run_cs2_gh200_posttrain_eval_slurm_seed.sh" in submit
    assert "run_cs2_frozen_event_probe_slurm_seed.sh" in submit
    assert "setup_cs2_clariden_uenv.sh" not in prepare


def test_dependent_event_job_uses_exact_fixed_step_checkpoints() -> None:
    text = (ROOT / "scripts" / "run_cs2_frozen_event_probe_slurm_seed.sh").read_text()
    event = (ROOT / "scripts" / "run_cs2_frozen_event_probe.sh").read_text()
    assert "checkpoint_index=$((train_steps - 1))" in text
    assert "synchronized/checkpoint-$checkpoint_index/checkpoint.pth" in text
    assert "shuffled/checkpoint-$checkpoint_index/checkpoint.pth" in text
    assert "mira-cs2-gh200-sync-control-audit-v3" in text
    assert "CS1K_EXPECTED_SINGLE_CHECKPOINT_SHA256" in text
    assert "CS1K_EXPECTED_CODEC_CHECKPOINT_SHA256" in text
    assert 'export TORCH_HOME="$torch_home_root/event_$SLURM_JOB_ID"' in text
    assert "prepare_dinov2_cache.py" in text
    assert '--codec-checkpoint "$codec_checkpoint"' in event


def test_event_only_recovery_reuses_evaluation_and_archives_failed_roots() -> None:
    submit = (ROOT / "scripts" / "submit_cs2_gh200_event_recovery.sh").read_text()
    prepare = (
        ROOT / "scripts" / "prepare_and_submit_cs2_clariden_event_recovery.sh"
    ).read_text()

    assert "synchronized_test_seed_sweep/summary.json" in submit
    assert "synchronized_test_action_loss_seed_sweep/summary.json" in submit
    assert "synchronized_test_first_death_action_loss_seed_sweep/summary.json" in submit
    assert "CS1K_EVENT_RECOVERY_SEEDS" in submit
    assert "CS1K_EVENT_RECOVERY_MANIFEST" in submit
    assert 'mv "$output_root/seed_$seed/event_probe" "$archive_path"' in submit
    assert "event_recovery_manifest.json" in submit
    assert "run_cs2_frozen_event_probe_slurm_seed.sh" in submit
    assert "run_cs2_gh200_sweep_finalize.sh" in submit
    assert '--dependency="afterok:$event_dependency"' in submit
    assert "run_cs2_gh200_posttrain_eval_slurm_seed.sh" not in submit
    assert "run_cs2_gh200_sync_control.sh" not in submit
    assert "run_cs2_gh200_loader_preflight.sh" not in submit
    assert "setup_cs2_clariden_uenv.sh" not in prepare


def test_submitted_jobs_pin_source_commits_until_execution() -> None:
    training = (ROOT / "scripts" / "run_cs2_gh200_slurm_seed.sh").read_text()
    event = (ROOT / "scripts" / "run_cs2_frozen_event_probe.sh").read_text()
    submit = (ROOT / "scripts" / "submit_cs2_gh200_sweep.sh").read_text()
    assert "CS1K_EXPECTED_MIRA_COMMIT" in training
    assert "CS1K_EXPECTED_MIRA_COMMIT" in event
    assert "CS1K_EXPECTED_RELEASE_COMMIT" in event
    assert "CS1K_EXPECTED_MIRA_COMMIT=$(git" in submit
    assert "CS1K_EXPECTED_RELEASE_COMMIT=$(git" in submit


def test_clariden_wrapper_uses_pinned_arm64_pytorch_uenv() -> None:
    setup = (ROOT / "scripts" / "setup_cs2_clariden_uenv.sh").read_text()
    prepare = (ROOT / "scripts" / "prepare_and_submit_cs2_clariden_sweep.sh").read_text()
    config = (ROOT / "scripts" / "clariden_sync_sweep.env.example").read_text()
    for text in (setup, prepare, config):
        assert "pytorch/v2.8.0:v1" in text
    assert "7dd6092b40a76a262b633c591d8edb6ce0a86c11" in setup
    assert "--branch v0.7.0" in setup
    assert "--no-build-isolation" in setup
    assert '--editable "$torchcodec_source"' in setup
    assert "https://download.pytorch.org/whl/cpu" not in setup
    assert '"torchcodec_install": "source-editable"' in setup
    assert "22dad05a3f2fd6c242a56e55e1eb2af61ed42385" in setup
    assert "--system-site-packages" in setup
    assert "platform.machine()" in setup
    assert "uenv run --view=default" in prepare
    assert "--uenv=$uenv_label:/user-environment --view=default" in prepare
    assert "set -a\nsource" in prepare
    assert 'CS1K_GH200_PARTITION="normal"' in config


def test_login_node_submitter_does_not_execute_uenv_python() -> None:
    text = (ROOT / "scripts" / "submit_cs2_gh200_sweep.sh").read_text()
    assert "CS1K_SUBMIT_PYTHON" in text
    assert '"$submit_python" - \\' in text
    assert '[[ ! -x "$path" && ! -L "$path" ]]' in text
    assert '"$output_root/submission_manifest.json"' in text
    assert "Refusing to reuse an already submitted or finalized sweep" in text
    assert "strict=True" not in text
    assert "Submitted job counts do not match the frozen seed list" in text


def test_clariden_dataset_stage_is_pinned_and_fail_closed() -> None:
    download = (ROOT / "scripts" / "download_cs2_dust2_subset.py").read_text()
    worker = (ROOT / "scripts" / "run_cs2_clariden_dataset_stage.sh").read_text()
    submit = (ROOT / "scripts" / "submit_cs2_clariden_dataset_stage.sh").read_text()
    assert "5a105b6e470407769d17fd14d73ef44e21b61b9a" in download
    assert "509e628617aa2ff2af3e848cf1aec89592a4c94b" in download
    assert "EXPECTED_DUST2_SAMPLES = 9_410" in download
    assert "EXPECTED_DUST2_SHARDS = 116" in download
    assert "snapshot_download(" in download
    assert "--splits train val test pilot_test" in worker
    assert "cs2_train.scripts.materialize_dust2_subset" in worker
    assert "materialization_provenance.json" in worker
    assert "33abbb623072932431871a612620110c473d4b664c52010e5763c273c6daf10e" in worker
    assert "--job-name=mira-cs2-dataset-stage" in submit
    assert "dataset_stage_complete.json" in submit


def test_frozen_endpoint_stager_pins_bundle_and_each_extracted_file() -> None:
    text = (ROOT / "scripts" / "stage_cs2_frozen_endpoints.sh").read_text()
    assert "198b53912948a53fd248f68cd7219d0526ab0965ab717eb59e18911c4a3664e2" in text
    assert "3c286c59b74cd141e72af69cde1a0a005142d8d2b472c789cdf2a39a140c4b7a" in text
    assert "3dbd8f0e43dbe833a5f36370d75f6306c7aa036dfcd3edba767ab138232fa047" in text
    assert "Refusing to overwrite a conflicting destination" in text


def _initialize_clean_repo(path: Path) -> None:
    path.mkdir()
    (path / "tracked.txt").write_text("frozen\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "add", "tracked.txt"], cwd=path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "frozen",
        ],
        cwd=path,
        check=True,
    )


def _event_recovery_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    project = tmp_path / "mira"
    release = tmp_path / "release"
    fake_bin = tmp_path / "bin"
    _initialize_clean_repo(project)
    _initialize_clean_repo(release)
    fake_bin.mkdir()

    codec = tmp_path / "codec.pth"
    single = tmp_path / "single.pth"
    codec.write_bytes(b"codec")
    single.write_bytes(b"single")
    output_root = tmp_path / "output"
    for seed in (28, 29, 30):
        seed_root = output_root / f"seed_{seed}"
        (seed_root / "evaluation" / "synchronized_test_seed_sweep").mkdir(parents=True)
        (seed_root / "evaluation" / "synchronized_test_action_loss_seed_sweep").mkdir()
        (seed_root / "evaluation" / "synchronized_test_first_death_action_loss_seed_sweep").mkdir()
        (seed_root / "audit.json").write_text(
            json.dumps(
                {
                    "schema": "mira-cs2-gh200-sync-control-audit-v3",
                    "status": "pass",
                    "seed": seed,
                    "train_steps": 10_000,
                }
            ),
            encoding="utf-8",
        )
        for path in (
            seed_root / "evaluation" / "synchronized_test_seed_sweep" / "summary.json",
            seed_root / "evaluation" / "synchronized_test_action_loss_seed_sweep" / "summary.json",
            seed_root
            / "evaluation"
            / "synchronized_test_first_death_action_loss_seed_sweep"
            / "summary.json",
        ):
            path.write_text("{}\n", encoding="utf-8")
        event_root = seed_root / "event_probe"
        event_root.mkdir()
        (event_root / "status.tsv").write_text(
            "2026-07-27T09:00:00Z\tfeatures_single\trunning\n",
            encoding="utf-8",
        )

    sbatch_log = tmp_path / "sbatch.log"
    sbatch_counter = tmp_path / "sbatch.counter"
    fake_sbatch = fake_bin / "sbatch"
    fake_sbatch.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
counter=${FAKE_SBATCH_COUNTER:?}
log=${FAKE_SBATCH_LOG:?}
if [[ -f "$counter" ]]; then value=$(<"$counter"); else value=1000; fi
value=$((value + 1))
printf '%s\n' "$value" >"$counter"
printf '%s\n' "$*" >>"$log"
printf '%s\n' "$value"
""",
        encoding="utf-8",
    )
    fake_sbatch.chmod(0o755)
    fake_sha256sum = fake_bin / "sha256sum"
    fake_sha256sum.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
case "$1" in
  *codec.pth) digest=3c286c59b74cd141e72af69cde1a0a005142d8d2b472c789cdf2a39a140c4b7a ;;
  *single.pth) digest=3dbd8f0e43dbe833a5f36370d75f6306c7aa036dfcd3edba767ab138232fa047 ;;
  *) exit 2 ;;
esac
printf '%s  %s\n' "$digest" "$1"
""",
        encoding="utf-8",
    )
    fake_sha256sum.chmod(0o755)

    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAKE_SBATCH_COUNTER": str(sbatch_counter),
        "FAKE_SBATCH_LOG": str(sbatch_log),
        "MIRA_PROJECT_DIR": str(project),
        "MIRA_PYTHON": sys.executable,
        "CS1K_SUBMIT_PYTHON": sys.executable,
        "CS1K_RELEASE_DIR": str(release),
        "CS1K_RELEASE_PYTHON": sys.executable,
        "CS1K_OUTPUT_ROOT": str(output_root),
        "CS1K_CODEC_CHECKPOINT": str(codec),
        "CS1K_SINGLE_CHECKPOINT": str(single),
        "CS1K_TRAIN_STEPS": "10000",
        "CS1K_TRAINING_SEEDS": "28 29 30",
        "CS1K_SLURM_ACCOUNT": "test-account",
        "CS1K_GH200_PARTITION": "test-gh200",
        "CS1K_EVENT_RECOVERY_TAG": "test-recovery",
    }
    return output_root, sbatch_log, env


def test_event_only_recovery_submits_three_events_and_one_finalizer(tmp_path: Path) -> None:
    output_root, sbatch_log, env = _event_recovery_fixture(tmp_path)
    subprocess.run(
        ["bash", str(ROOT / "scripts" / "submit_cs2_gh200_event_recovery.sh")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    calls = sbatch_log.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 4
    assert all("--dependency=" not in call for call in calls[:3])
    assert "afterok:1001:1002:1003" in calls[3]
    for seed in (28, 29, 30):
        seed_root = output_root / f"seed_{seed}"
        assert not (seed_root / "event_probe").exists()
        assert (seed_root / "event_probe_failed_test-recovery").is_dir()

    manifest = json.loads((output_root / "event_recovery_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "mira-cs2-event-recovery-submission-v2"
    assert manifest["training_reused"] is True
    assert manifest["evaluation_reused"] is True
    assert manifest["codec_path_relocated"] is True
    assert manifest["recovery_seeds"] == [28, 29, 30]
    assert manifest["event_jobs"] == [1001, 1002, 1003]
    assert manifest["finalize_job"] == 1004


def test_event_only_recovery_retries_failed_subset_and_reuses_complete_seed(
    tmp_path: Path,
) -> None:
    output_root, sbatch_log, env = _event_recovery_fixture(tmp_path)
    seed_28_event_root = output_root / "seed_28" / "event_probe"
    (seed_28_event_root / "status.tsv").write_text(
        "2026-07-27T17:00:00Z\tpipeline\tcomplete\n",
        encoding="utf-8",
    )
    retry_manifest = output_root / "event_recovery_manifest_retry.json"
    env.update(
        {
            "CS1K_EVENT_RECOVERY_SEEDS": "29 30",
            "CS1K_EVENT_RECOVERY_TAG": "retry",
            "CS1K_EVENT_RECOVERY_MANIFEST": str(retry_manifest),
        }
    )

    subprocess.run(
        ["bash", str(ROOT / "scripts" / "submit_cs2_gh200_event_recovery.sh")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    calls = sbatch_log.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 3
    assert "CS1K_SEED=29" in calls[0]
    assert "CS1K_SEED=30" in calls[1]
    assert "afterok:1001:1002" in calls[2]
    assert seed_28_event_root.is_dir()
    assert (output_root / "seed_29" / "event_probe_failed_retry").is_dir()
    assert (output_root / "seed_30" / "event_probe_failed_retry").is_dir()

    manifest = json.loads(retry_manifest.read_text(encoding="utf-8"))
    assert manifest["schema"] == "mira-cs2-event-recovery-submission-v2"
    assert manifest["recovery_seeds"] == [29, 30]
    assert manifest["event_jobs"] == [1001, 1002]
    assert manifest["finalize_job"] == 1003


def test_complete_sweep_submitter_submits_three_seed_dependency_dag(tmp_path: Path) -> None:
    project = tmp_path / "mira"
    release = tmp_path / "release"
    dataset = tmp_path / "dataset"
    fake_bin = tmp_path / "bin"
    _initialize_clean_repo(project)
    _initialize_clean_repo(release)
    dataset.mkdir()
    fake_bin.mkdir()

    manifest = dataset / "manifest.parquet"
    provenance = dataset / "provenance.json"
    codec = tmp_path / "codec.pth"
    single = tmp_path / "single.pth"
    for path in (manifest, provenance, codec, single):
        path.write_bytes(b"fixture")

    sbatch_log = tmp_path / "sbatch.log"
    sbatch_counter = tmp_path / "sbatch.counter"
    fake_sbatch = fake_bin / "sbatch"
    fake_sbatch.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
counter=${FAKE_SBATCH_COUNTER:?}
log=${FAKE_SBATCH_LOG:?}
if [[ -f "$counter" ]]; then
  value=$(<"$counter")
else
  value=1000
fi
value=$((value + 1))
printf '%s\n' "$value" >"$counter"
printf '%s\n' "$*" >>"$log"
printf '%s\n' "$value"
""",
        encoding="utf-8",
    )
    fake_sbatch.chmod(0o755)
    fake_sha256sum = fake_bin / "sha256sum"
    fake_sha256sum.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
case "$1" in
  *manifest.parquet) digest=33abbb623072932431871a612620110c473d4b664c52010e5763c273c6daf10e ;;
  *codec.pth) digest=3c286c59b74cd141e72af69cde1a0a005142d8d2b472c789cdf2a39a140c4b7a ;;
  *single.pth) digest=3dbd8f0e43dbe833a5f36370d75f6306c7aa036dfcd3edba767ab138232fa047 ;;
  *) exit 2 ;;
esac
printf '%s  %s\n' "$digest" "$1"
""",
        encoding="utf-8",
    )
    fake_sha256sum.chmod(0o755)

    output_root = tmp_path / "output"
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAKE_SBATCH_COUNTER": str(sbatch_counter),
        "FAKE_SBATCH_LOG": str(sbatch_log),
        "MIRA_PROJECT_DIR": str(project),
        "MIRA_PYTHON": sys.executable,
        "CS1K_RELEASE_DIR": str(release),
        "CS1K_RELEASE_PYTHON": sys.executable,
        "CS1K_DATASET_DIR": str(dataset),
        "CS1K_MANIFEST_PATH": str(manifest),
        "CS1K_CONFIRMATORY_SPLIT_PROVENANCE": str(provenance),
        "CS1K_CODEC_CHECKPOINT": str(codec),
        "CS1K_SINGLE_CHECKPOINT": str(single),
        "CS1K_OUTPUT_ROOT": str(output_root),
        "CS1K_TRAIN_STEPS": "10000",
        "CS1K_ARM_HOURS": "4",
        "CS1K_SLURM_ACCOUNT": "test-account",
        "CS1K_GH200_PARTITION": "test-gh200",
    }
    subprocess.run(
        ["bash", str(ROOT / "scripts" / "submit_cs2_gh200_sweep.sh")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    calls = sbatch_log.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 7
    assert all("--nodes=1" in call for call in calls[:3])
    assert all("--gpus-per-node=4" in call for call in calls[:3])
    assert all("--nodes=4" not in call for call in calls[:3])
    assert all(f"--chdir={output_root}" in call for call in calls)
    assert all(f"--output={output_root}/logs/" in call for call in calls)
    assert [f"CS1K_SEED={seed}" in calls[index] for index, seed in enumerate((28, 29, 30))] == [
        True,
        True,
        True,
    ]
    assert "afterok:1001" in calls[3]
    assert "afterok:1002" in calls[4]
    assert "afterok:1003" in calls[5]
    assert "afterok:1004:1005:1006" in calls[6]

    submission = json.loads((output_root / "submission_manifest.json").read_text(encoding="utf-8"))
    assert submission["schema"] == "mira-cs2-clariden-submission-v2"
    assert submission["slurm"]["log_root"] == str(output_root / "logs")
    assert submission["contract"]["training_seeds"] == [28, 29, 30]
    assert submission["contract"]["training_topology"] == {
        "nodes_per_seed_job": 1,
        "gh200s_per_node": 4,
        "ddp_processes_per_node": 4,
        "cpus_per_launcher_task": 288,
        "world_size": 4,
    }
    assert (
        submission["contract"]["codec_checkpoint_sha256"]
        == "3c286c59b74cd141e72af69cde1a0a005142d8d2b472c789cdf2a39a140c4b7a"
    )
    assert submission["slurm"]["finalize_job"] == "1007"
