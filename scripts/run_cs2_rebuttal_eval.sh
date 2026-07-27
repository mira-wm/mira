#!/usr/bin/env bash
# Paired, deterministic Dust2 evaluation for the preliminary single/shared MIRA comparison.
#
# The complete held-out test split has 52 ten-POV rounds. Each seed evaluates the exact same 520
# POV clips in both arms: the single model uses batch=10 independent rows and the shared model uses
# batch=1 synchronized ten-POV group. Three stochastic rollout seeds are kept separate and then
# summarized as paired shared-minus-single deltas.
set -euo pipefail

project_dir=${MIRA_PROJECT_DIR:-/home/ubuntu/projects/mira}
run_root=${CS1K_RUN_ROOT:?Set CS1K_RUN_ROOT to the completed training pipeline directory}
python_bin=${MIRA_PYTHON:-$project_dir/.pixi/envs/default/bin/python}
eval_root=${CS1K_EVAL_ROOT:-$run_root/evaluation/test_seed_sweep}
eval_seeds=${CS1K_EVAL_SEEDS:-"37 38 39"}
test_rounds=${CS1K_TEST_ROUNDS:-52}

cd "$project_dir"
export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB_MODE=disabled

latest_checkpoint() {
  local arm=$1
  find "$run_root/$arm" -path '*/checkpoint-*/checkpoint.pth' -print0 |
    sort -zV | tail -z -n 1 | tr -d '\0'
}

single_checkpoint=$(latest_checkpoint single)
shared_checkpoint=$(latest_checkpoint shared)
if [[ -z "$single_checkpoint" || -z "$shared_checkpoint" ]]; then
  echo "Both final single and shared checkpoints are required under $run_root" >&2
  exit 1
fi

mkdir -p "$eval_root/provenance"
printf '%s\n' "$single_checkpoint" >"$eval_root/provenance/single_checkpoint.txt"
printf '%s\n' "$shared_checkpoint" >"$eval_root/provenance/shared_checkpoint.txt"
sha256sum "$single_checkpoint" "$shared_checkpoint" >"$eval_root/provenance/checkpoints.sha256"
git rev-parse HEAD >"$eval_root/provenance/evaluator_code_commit.txt"
git status --porcelain=v1 >"$eval_root/provenance/evaluator_code_status.txt"
git diff --binary >"$eval_root/provenance/evaluator_code.patch"
printf '%s\n' "$eval_seeds" >"$eval_root/provenance/seeds.txt"
printf '%s\n' "$test_rounds" >"$eval_root/provenance/test_rounds.txt"

for seed in $eval_seeds; do
  seed_dir=$eval_root/seed_$seed
  mkdir -p "$seed_dir"

  "$python_bin" scripts/eval_world_model_offline.py "$single_checkpoint" \
    --split test \
    --seed "$seed" \
    --deterministic \
    --dino-model dinov2_vitb14 \
    --n-context-frames 8 \
    --num-unrolled-frames 4 \
    --drift-metric-frames 4 \
    --fdd-slice-frames 2 \
    --per-device-batch-size 10 \
    --num-samples "$((test_rounds * 10))" \
    --val-n-samples "$((test_rounds * 10))" \
    --no-compile \
    --viz 2 \
    --output-dir "$seed_dir/single_viz" \
    --results-json "$seed_dir/single.json" \
    2>&1 | tee "$seed_dir/single.log"

  "$python_bin" scripts/eval_world_model_offline.py "$shared_checkpoint" \
    --split test \
    --seed "$seed" \
    --deterministic \
    --dino-model dinov2_vitb14 \
    --n-context-frames 8 \
    --num-unrolled-frames 4 \
    --drift-metric-frames 4 \
    --fdd-slice-frames 2 \
    --per-device-batch-size 1 \
    --num-samples "$test_rounds" \
    --val-n-samples "$test_rounds" \
    --no-compile \
    --viz 2 \
    --output-dir "$seed_dir/shared_viz" \
    --results-json "$seed_dir/shared.json" \
    2>&1 | tee "$seed_dir/shared.log"
done

"$python_bin" scripts/summarize_cs2_rebuttal_eval.py "$eval_root"
