#!/usr/bin/env bash
# Submit evaluation, event probes, and final aggregation for completed training checkpoints.
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [environment-file]" >&2
  exit 2
fi
if [[ $# -eq 1 ]]; then
  if [[ ! -f "$1" ]]; then
    echo "Environment file not found: $1" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "$1"
fi

project_dir=${MIRA_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
python_bin=${MIRA_PYTHON:-$project_dir/.pixi/envs/default/bin/python}
submit_python=${CS1K_SUBMIT_PYTHON:-python3}
release_dir=${CS1K_RELEASE_DIR:?Set the pinned CounterStrike-1K checkout}
release_python=${CS1K_RELEASE_PYTHON:?Set the CounterStrike-1K Python}
output_root=${CS1K_OUTPUT_ROOT:?Set the existing completed training root}
train_steps=${CS1K_TRAIN_STEPS:?Set the fixed optimizer-update count}
account=${CS1K_SLURM_ACCOUNT:?Set the Clariden Slurm account}
partition=${CS1K_GH200_PARTITION:?Set the Clariden GH200 partition}
event_time=${CS1K_EVENT_TIME:-04:00:00}
finalize_time=${CS1K_FINALIZE_TIME:-00:30:00}
training_seed_text=${CS1K_TRAINING_SEEDS:-"28 29 30"}
recovery_manifest=$output_root/posttrain_recovery_manifest.json

for command_name in git sbatch sha256sum "$submit_python"; do
  if ! command -v "$command_name" >/dev/null; then
    echo "Required command not found: $command_name" >&2
    exit 1
  fi
done
for directory in "$project_dir" "$release_dir" "$output_root"; do
  if [[ ! -d "$directory" ]]; then
    echo "Required directory is absent: $directory" >&2
    exit 1
  fi
done
for path in "$python_bin" "$release_python"; do
  if [[ ! -x "$path" && ! -L "$path" ]]; then
    echo "Configured runtime Python is absent: $path" >&2
    exit 1
  fi
done
for repo in "$project_dir" "$release_dir"; do
  if [[ -n "$(git -C "$repo" status --porcelain=v1)" ]]; then
    echo "Post-training recovery requires a clean checkout: $repo" >&2
    exit 1
  fi
done
if [[ -e "$recovery_manifest" ]]; then
  echo "Post-training recovery was already submitted: $recovery_manifest" >&2
  exit 1
fi
for path in "$output_root/sweep_summary.json" "$output_root/event_probe_sweep_summary.json"; do
  if [[ -e "$path" ]]; then
    echo "Sweep is already finalized: $path" >&2
    exit 1
  fi
done

read -r -a training_seeds <<<"$training_seed_text"
if [[ "${training_seeds[*]}" != "28 29 30" ]]; then
  echo "Publication recovery requires exactly CS1K_TRAINING_SEEDS='28 29 30'" >&2
  exit 1
fi
if ! [[ "$train_steps" =~ ^[1-9][0-9]*$ ]]; then
  echo "CS1K_TRAIN_STEPS must be a positive integer" >&2
  exit 1
fi
final_step=$((train_steps - 1))
training_commit=
for seed in "${training_seeds[@]}"; do
  seed_root=$output_root/seed_$seed
  recorded_commit=$(
    <"$seed_root/provenance/node_0/code_commit.txt" tr -d '[:space:]'
  )
  if [[ -z "$training_commit" ]]; then
    training_commit=$recorded_commit
  elif [[ "$recorded_commit" != "$training_commit" ]]; then
    echo "Training commits differ across completed seeds" >&2
    exit 1
  fi
  for arm in synchronized shuffled; do
    checkpoint=$seed_root/$arm/checkpoint-$final_step/checkpoint.pth
    termination=$seed_root/$arm/training_termination.json
    for path in "$checkpoint" "$termination"; do
      if [[ ! -s "$path" ]]; then
        echo "Completed training artifact is absent: $path" >&2
        exit 1
      fi
    done
  done
  if [[ -s "$seed_root/audit.json" ]]; then
    echo "Seed $seed is already evaluated and audited" >&2
    exit 1
  fi
  if [[ -e "$seed_root/event_probe" ]]; then
    echo "Seed $seed already has an event-probe root" >&2
    exit 1
  fi
done
if [[ -n "${CS1K_EXPECTED_TRAINING_COMMIT:-}" ]] &&
  [[ "$CS1K_EXPECTED_TRAINING_COMMIT" != "$training_commit" ]]; then
  echo "Configured training commit does not match completed artifacts" >&2
  exit 1
fi

export MIRA_PROJECT_DIR="$project_dir"
export MIRA_PYTHON="$python_bin"
export CS1K_RELEASE_DIR="$release_dir"
export CS1K_RELEASE_PYTHON="$release_python"
export CS1K_EXPECTED_TRAINING_COMMIT="$training_commit"
export CS1K_EXPECTED_MIRA_COMMIT
CS1K_EXPECTED_MIRA_COMMIT=$(git -C "$project_dir" rev-parse HEAD)
export CS1K_EXPECTED_RELEASE_COMMIT
CS1K_EXPECTED_RELEASE_COMMIT=$(git -C "$release_dir" rev-parse HEAD)

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
eval_job_ids=()
event_job_ids=()
for seed in "${training_seeds[@]}"; do
  raw_job_id=$(sbatch \
    --parsable \
    --account="$account" \
    --partition="$partition" \
    --job-name="mira-cs2-eval-$seed" \
    --nodes=1 \
    --ntasks=1 \
    --gpus-per-node=1 \
    --cpus-per-task=72 \
    --hint=nomultithread \
    --chdir="$output_root" \
    --output="$log_root/recovery-eval-seed-$seed-%j.log" \
    --time="$event_time" \
    --export=ALL,CS1K_SEED="$seed" \
    "${auxiliary_extra_args[@]}" \
    "$project_dir/scripts/run_cs2_gh200_posttrain_eval_slurm_seed.sh")
  eval_job_ids+=("$(parse_job_id "$raw_job_id")")
done

for index in "${!training_seeds[@]}"; do
  seed=${training_seeds[$index]}
  eval_job_id=${eval_job_ids[$index]}
  raw_job_id=$(sbatch \
    --parsable \
    --account="$account" \
    --partition="$partition" \
    --job-name="mira-cs2-events-$seed" \
    --nodes=1 \
    --ntasks=1 \
    --gpus-per-node=1 \
    --chdir="$output_root" \
    --output="$log_root/recovery-event-seed-$seed-%j.log" \
    --time="$event_time" \
    --dependency="afterok:$eval_job_id" \
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
  --output="$log_root/recovery-finalize-%j.log" \
  --time="$finalize_time" \
  --dependency="afterok:$event_dependency" \
  --export=ALL \
  "${auxiliary_extra_args[@]}" \
  "$project_dir/scripts/run_cs2_gh200_sweep_finalize.sh")
finalize_job_id=$(parse_job_id "$raw_finalize_job_id")

"$submit_python" - \
  "$recovery_manifest" \
  "$training_commit" \
  "$CS1K_EXPECTED_MIRA_COMMIT" \
  "$finalize_job_id" \
  "${eval_job_ids[*]}" \
  "${event_job_ids[*]}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    output,
    training_commit,
    evaluator_commit,
    finalize_job_id,
    eval_job_ids,
    event_job_ids,
) = sys.argv[1:]
payload = {
    "schema": "mira-cs2-posttrain-recovery-submission-v1",
    "submitted_at": datetime.now(timezone.utc).isoformat(),
    "training_reused": True,
    "training_commit": training_commit,
    "evaluator_commit": evaluator_commit,
    "evaluation_jobs": [int(value) for value in eval_job_ids.split()],
    "event_jobs": [int(value) for value in event_job_ids.split()],
    "finalize_job": int(finalize_job_id),
}
Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

printf 'Submitted checkpoint-only post-training recovery.\n'
for index in "${!training_seeds[@]}"; do
  printf '  seed %s: evaluation %s -> event %s\n' \
    "${training_seeds[$index]}" "${eval_job_ids[$index]}" "${event_job_ids[$index]}"
done
printf '  final aggregation: %s\n' "$finalize_job_id"
printf '  recovery manifest: %s\n' "$recovery_manifest"
