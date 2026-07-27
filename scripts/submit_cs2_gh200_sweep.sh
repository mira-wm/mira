#!/usr/bin/env bash
# Submit the complete three-seed Dust2 synchronization experiment as one Slurm dependency DAG.
#
# Usage:
#   bash scripts/submit_cs2_gh200_sweep.sh /path/to/clariden_sync_sweep.env
#
# This is a control-plane script: run it once on a Slurm login node. It submits three independent
# full-node jobs, each using all four GH200s on one Clariden node, a one-GPU event-probe job after
# each successful child audit, and a final fail-closed aggregation job after all probes succeed.
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [environment-file]" >&2
  exit 2
fi
if [[ $# -eq 1 ]]; then
  environment_file=$1
  if [[ ! -f "$environment_file" ]]; then
    echo "Environment file not found: $environment_file" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "$environment_file"
fi

project_dir=${MIRA_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
python_bin=${MIRA_PYTHON:-$project_dir/.pixi/envs/default/bin/python}
submit_python=${CS1K_SUBMIT_PYTHON:-python3}
release_dir=${CS1K_RELEASE_DIR:?Set the pinned CounterStrike-1K checkout}
release_python=${CS1K_RELEASE_PYTHON:?Set the CounterStrike-1K Python}
dataset_dir=${CS1K_DATASET_DIR:?Set the materialized Dust2 dataset root}
manifest_path=${CS1K_MANIFEST_PATH:?Set the frozen confirmatory manifest}
split_provenance=${CS1K_CONFIRMATORY_SPLIT_PROVENANCE:?Set split provenance}
codec_checkpoint=${CS1K_CODEC_CHECKPOINT:?Set the frozen codec checkpoint}
single_checkpoint=${CS1K_SINGLE_CHECKPOINT:?Set the frozen single-MIRA checkpoint}
output_root=${CS1K_OUTPUT_ROOT:?Set a new shared sweep output root}
train_steps=${CS1K_TRAIN_STEPS:?Set the fixed optimizer-update count}
arm_hours=${CS1K_ARM_HOURS:?Set the fail-closed per-arm hour cap}
account=${CS1K_SLURM_ACCOUNT:?Set the Clariden Slurm account}
partition=${CS1K_GH200_PARTITION:?Set the Clariden GH200 partition}
training_time=${CS1K_TRAINING_TIME:-12:00:00}
event_time=${CS1K_EVENT_TIME:-04:00:00}
finalize_time=${CS1K_FINALIZE_TIME:-00:30:00}
training_seed_text=${CS1K_TRAINING_SEEDS:-"28 29 30"}
expected_manifest_sha256=33abbb623072932431871a612620110c473d4b664c52010e5763c273c6daf10e
expected_codec_sha256=${CS1K_EXPECTED_CODEC_CHECKPOINT_SHA256:-3c286c59b74cd141e72af69cde1a0a005142d8d2b472c789cdf2a39a140c4b7a}
expected_single_sha256=${CS1K_EXPECTED_SINGLE_CHECKPOINT_SHA256:-3dbd8f0e43dbe833a5f36370d75f6306c7aa036dfcd3edba767ab138232fa047}
reuse_loader_selection=${CS1K_REUSE_LOADER_SELECTION:-}

for command_name in git realpath sbatch sha256sum "$submit_python"; do
  if ! command -v "$command_name" >/dev/null; then
    echo "Required command not found: $command_name" >&2
    exit 1
  fi
done
for directory in "$project_dir" "$release_dir" "$dataset_dir"; do
  if [[ ! -d "$directory" ]]; then
    echo "Required directory is absent: $directory" >&2
    exit 1
  fi
done
for path in \
  "$manifest_path" \
  "$split_provenance" \
  "$codec_checkpoint" \
  "$single_checkpoint"; do
  if [[ ! -e "$path" ]]; then
    echo "Required input is absent: $path" >&2
    exit 1
  fi
done
# A uenv-layered venv may point into /user-environment and therefore appear as a broken symlink on
# the login node. The interpreters are executed only after Slurm mounts the pinned uenv.
for path in "$python_bin" "$release_python"; do
  if [[ ! -x "$path" && ! -L "$path" ]]; then
    echo "Configured runtime Python is absent: $path" >&2
    exit 1
  fi
done
if [[ -n "$(git -C "$project_dir" status --porcelain=v1)" ]]; then
  echo "MIRA checkout must be clean before submission" >&2
  exit 1
fi
if [[ -n "$(git -C "$release_dir" status --porcelain=v1)" ]]; then
  echo "CounterStrike-1K checkout must be clean before submission" >&2
  exit 1
fi
if [[ "$(sha256sum "$manifest_path" | cut -d' ' -f1)" != "$expected_manifest_sha256" ]]; then
  echo "Confirmatory manifest SHA-256 drifted" >&2
  exit 1
fi
if [[ "$(sha256sum "$codec_checkpoint" | cut -d' ' -f1)" != "$expected_codec_sha256" ]]; then
  echo "Codec checkpoint SHA-256 drifted" >&2
  exit 1
fi
if [[ "$(sha256sum "$single_checkpoint" | cut -d' ' -f1)" != "$expected_single_sha256" ]]; then
  echo "Single-MIRA checkpoint SHA-256 drifted" >&2
  exit 1
fi
reuse_loader_selection_sha256=
if [[ -n "$reuse_loader_selection" ]]; then
  if [[ ! -s "$reuse_loader_selection" ]]; then
    echo "Reusable loader selection is absent: $reuse_loader_selection" >&2
    exit 1
  fi
  reuse_loader_selection=$(realpath "$reuse_loader_selection")
  reuse_loader_selection_sha256=$(sha256sum "$reuse_loader_selection" | cut -d' ' -f1)
  export CS1K_REUSE_LOADER_SELECTION="$reuse_loader_selection"
fi
if ! [[ "$train_steps" =~ ^[1-9][0-9]*$ ]]; then
  echo "CS1K_TRAIN_STEPS must be a positive integer" >&2
  exit 1
fi

read -r -a training_seeds <<<"$training_seed_text"
if [[ "${training_seeds[*]}" != "28 29 30" ]]; then
  echo "Publication sweep requires exactly CS1K_TRAINING_SEEDS='28 29 30'" >&2
  exit 1
fi
for seed in "${training_seeds[@]}"; do
  if [[ -e "$output_root/seed_$seed" ]]; then
    echo "Refusing to reuse existing seed output: $output_root/seed_$seed" >&2
    exit 1
  fi
done
for path in \
  "$output_root/submission_manifest.json" \
  "$output_root/sweep_summary.json" \
  "$output_root/event_probe_sweep_summary.json"; do
  if [[ -e "$path" ]]; then
    echo "Refusing to reuse an already submitted or finalized sweep: $path" >&2
    exit 1
  fi
done

export MIRA_PROJECT_DIR="$project_dir"
export MIRA_PYTHON="$python_bin"
export CS1K_RELEASE_DIR="$release_dir"
export CS1K_RELEASE_PYTHON="$release_python"
export CS1K_DATASET_DIR="$dataset_dir"
export CS1K_MANIFEST_PATH="$manifest_path"
export CS1K_CONFIRMATORY_SPLIT_PROVENANCE="$split_provenance"
export CS1K_CODEC_CHECKPOINT="$codec_checkpoint"
export CS1K_SINGLE_CHECKPOINT="$single_checkpoint"
export CS1K_OUTPUT_ROOT="$output_root"
export CS1K_TRAIN_STEPS="$train_steps"
export CS1K_ARM_HOURS="$arm_hours"
export CS1K_EVENT_FEATURE_SEED=${CS1K_EVENT_FEATURE_SEED:-37}
export CS1K_EVENT_PROBE_SEEDS=${CS1K_EVENT_PROBE_SEEDS:-"17 29 43"}
export CS1K_EVENT_BOOTSTRAP_SAMPLES=${CS1K_EVENT_BOOTSTRAP_SAMPLES:-10000}
export CS1K_EXPECTED_CODEC_CHECKPOINT_SHA256="$expected_codec_sha256"
export CS1K_EXPECTED_SINGLE_CHECKPOINT_SHA256="$expected_single_sha256"
export CS1K_EXPECTED_MIRA_COMMIT
CS1K_EXPECTED_MIRA_COMMIT=$(git -C "$project_dir" rev-parse HEAD)
export CS1K_EXPECTED_RELEASE_COMMIT
CS1K_EXPECTED_RELEASE_COMMIT=$(git -C "$release_dir" rev-parse HEAD)

read -r -a training_extra_args <<<"${CS1K_TRAIN_SBATCH_ARGS:-}"
read -r -a auxiliary_extra_args <<<"${CS1K_AUX_SBATCH_ARGS:-}"

parse_job_id() {
  local raw=$1
  local job_id=${raw%%;*}
  if ! [[ "$job_id" =~ ^[0-9]+$ ]]; then
    echo "Could not parse sbatch job id from: $raw" >&2
    return 1
  fi
  printf '%s\n' "$job_id"
}

log_root=$output_root/logs
mkdir -p "$log_root"
training_job_ids=()
event_job_ids=()
for seed in "${training_seeds[@]}"; do
  raw_job_id=$(sbatch \
    --parsable \
    --account="$account" \
    --partition="$partition" \
    --job-name="mira-cs2-train-$seed" \
    --nodes=1 \
    --ntasks-per-node=1 \
    --gpus-per-node=4 \
    --cpus-per-task=288 \
    --exclusive \
    --hint=nomultithread \
    --chdir="$output_root" \
    --output="$log_root/training-seed-$seed-%j.log" \
    --time="$training_time" \
    --export=ALL,CS1K_SEED="$seed" \
    "${training_extra_args[@]}" \
    "$project_dir/scripts/run_cs2_gh200_slurm_seed.sh")
  training_job_ids+=("$(parse_job_id "$raw_job_id")")
done

for index in "${!training_seeds[@]}"; do
  seed=${training_seeds[$index]}
  training_job_id=${training_job_ids[$index]}
  raw_job_id=$(sbatch \
    --parsable \
    --account="$account" \
    --partition="$partition" \
    --job-name="mira-cs2-events-$seed" \
    --nodes=1 \
    --ntasks=1 \
    --gpus-per-node=1 \
    --chdir="$output_root" \
    --output="$log_root/event-seed-$seed-%j.log" \
    --time="$event_time" \
    --dependency="afterok:$training_job_id" \
    --export=ALL,CS1K_SEED="$seed" \
    "${auxiliary_extra_args[@]}" \
    "$project_dir/scripts/run_cs2_frozen_event_probe_slurm_seed.sh")
  event_job_ids+=("$(parse_job_id "$raw_job_id")")
done

event_dependency=$(IFS=:; printf '%s' "${event_job_ids[*]}")
raw_finalize_job_id=$(sbatch \
  --parsable \
  --account="$account" \
  --partition="$partition" \
  --job-name=mira-cs2-finalize \
  --nodes=1 \
  --ntasks=1 \
  --gpus-per-node=1 \
  --chdir="$output_root" \
  --output="$log_root/finalize-%j.log" \
  --time="$finalize_time" \
  --dependency="afterok:$event_dependency" \
  --export=ALL \
  "${auxiliary_extra_args[@]}" \
  "$project_dir/scripts/run_cs2_gh200_sweep_finalize.sh")
finalize_job_id=$(parse_job_id "$raw_finalize_job_id")

submission_manifest=$output_root/submission_manifest.json
"$submit_python" - \
  "$submission_manifest" \
  "$account" \
  "$partition" \
  "$CS1K_EXPECTED_MIRA_COMMIT" \
  "$CS1K_EXPECTED_RELEASE_COMMIT" \
  "$manifest_path" \
  "$expected_manifest_sha256" \
  "$codec_checkpoint" \
  "$expected_codec_sha256" \
  "$single_checkpoint" \
  "$expected_single_sha256" \
  "$reuse_loader_selection" \
  "$reuse_loader_selection_sha256" \
  "$train_steps" \
  "$arm_hours" \
  "$finalize_job_id" \
  "${training_seeds[*]}" \
  "${training_job_ids[*]}" \
  "${event_job_ids[*]}" <<'PY'
import json
import sys
from pathlib import Path

(
    output,
    account,
    partition,
    mira_commit,
    release_commit,
    manifest_path,
    manifest_sha256,
    codec_checkpoint,
    codec_checkpoint_sha256,
    single_checkpoint,
    single_checkpoint_sha256,
    reuse_loader_selection,
    reuse_loader_selection_sha256,
    train_steps,
    arm_hours,
    finalize_job_id,
    seed_text,
    training_job_text,
    event_job_text,
) = sys.argv[1:]
seeds = [int(value) for value in seed_text.split()]
training_jobs = training_job_text.split()
event_jobs = event_job_text.split()
if len(training_jobs) != len(seeds) or len(event_jobs) != len(seeds):
    raise SystemExit("Submitted job counts do not match the frozen seed list")
payload = {
    "schema": "mira-cs2-clariden-submission-v2",
    "status": "submitted",
    "slurm": {
        "account": account,
        "partition": partition,
        "training_jobs": dict(zip(map(str, seeds), training_jobs)),
        "event_jobs": dict(zip(map(str, seeds), event_jobs)),
        "finalize_job": finalize_job_id,
        "log_root": str(Path(output).parent / "logs"),
    },
    "contract": {
        "training_seeds": seeds,
        "train_steps_per_arm": int(train_steps),
        "arm_hours_safety_cap": float(arm_hours),
        "training_topology": {
            "nodes_per_seed_job": 1,
            "gh200s_per_node": 4,
            "ddp_processes_per_node": 4,
            "cpus_per_launcher_task": 288,
            "world_size": 4,
        },
        "mira_commit": mira_commit,
        "counterstrike_1k_commit": release_commit,
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_sha256,
        "codec_checkpoint_path": codec_checkpoint,
        "codec_checkpoint_sha256": codec_checkpoint_sha256,
        "single_checkpoint_path": single_checkpoint,
        "single_checkpoint_sha256": single_checkpoint_sha256,
        "loader_selection_reuse": (
            {
                "path": reuse_loader_selection,
                "sha256": reuse_loader_selection_sha256,
                "current_node_smoke_required": True,
            }
            if reuse_loader_selection
            else None
        ),
        "final_results": ["sweep_summary.json", "event_probe_sweep_summary.json"],
    },
}
path = Path(output)
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(path)
PY

printf 'Submitted complete Dust2 synchronization sweep.\n'
for index in "${!training_seeds[@]}"; do
  printf '  seed %s: training %s -> event %s\n' \
    "${training_seeds[$index]}" "${training_job_ids[$index]}" "${event_job_ids[$index]}"
done
printf '  final aggregation: %s\n' "$finalize_job_id"
printf '  submission manifest: %s\n' "$submission_manifest"
