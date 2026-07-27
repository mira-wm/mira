#!/usr/bin/env bash
# Re-run only deterministic evaluation/audit for an already completed Clariden training seed.
set -euo pipefail

project_dir=${MIRA_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
python_bin=${MIRA_PYTHON:-$project_dir/.pixi/envs/default/bin/python}
output_root=${CS1K_OUTPUT_ROOT:?Set the existing completed sweep root}
train_steps=${CS1K_TRAIN_STEPS:?Set the fixed optimizer-update count}
seed=${CS1K_SEED:?Set the completed training seed}
expected_training_commit=${CS1K_EXPECTED_TRAINING_COMMIT:?Set the immutable training commit}
expected_evaluator_commit=${CS1K_EXPECTED_MIRA_COMMIT:?Set the recovery evaluator commit}

: "${SLURM_JOB_ID:?Run this recovery through sbatch}"
if ! [[ "$seed" =~ ^[0-9]+$ ]]; then
  echo "CS1K_SEED must be a non-negative integer" >&2
  exit 1
fi
if ! [[ "$train_steps" =~ ^[1-9][0-9]*$ ]]; then
  echo "CS1K_TRAIN_STEPS must be a positive integer" >&2
  exit 1
fi

cd "$project_dir"
export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
if [[ -n "$(git status --porcelain=v1)" ]]; then
  echo "Post-training recovery requires a clean MIRA source tree" >&2
  exit 1
fi
if [[ "$(git rev-parse HEAD)" != "$expected_evaluator_commit" ]]; then
  echo "Recovery evaluator commit drifted after submission" >&2
  exit 1
fi

training_root=$output_root/seed_$seed
final_step=$((train_steps - 1))
for arm in synchronized shuffled; do
  checkpoint=$training_root/$arm/checkpoint-$final_step/checkpoint.pth
  termination=$training_root/$arm/training_termination.json
  for path in "$checkpoint" "$termination"; do
    if [[ ! -s "$path" ]]; then
      echo "Completed training artifact is absent: $path" >&2
      exit 1
    fi
  done
done
if [[ -s "$training_root/audit.json" ]]; then
  echo "Seed $seed is already evaluated and audited" >&2
  exit 1
fi

recorded_training_commit=$(
  <"$training_root/provenance/node_0/code_commit.txt" tr -d '[:space:]'
)
if [[ "$recorded_training_commit" != "$expected_training_commit" ]]; then
  echo "Seed $seed training commit does not match the frozen recovery commit" >&2
  exit 1
fi

# The evaluator is single-process, but using a job-private prepared cache also makes retries
# independent of mutable per-user Torch Hub state.
export TORCH_HOME="$output_root/torch_hub/posttrain_eval_$SLURM_JOB_ID"
mkdir -p "$TORCH_HOME"
"$python_bin" scripts/prepare_dinov2_cache.py \
  --torch-home "$TORCH_HOME" \
  --output "$TORCH_HOME/dinov2_cache_ready.json"

export CS1K_TRAINING_ROOT="$training_root"
export CS1K_EXPECTED_TRAINING_COMMIT="$expected_training_commit"
exec "$project_dir/scripts/run_cs2_gh200_sync_control_eval.sh"
