#!/usr/bin/env bash
# Action-sensitivity diagnostic on two-second windows centered at each test round's first death.
#
# The 52-round Dust2 test split has an in-range player_death event in every round. Both arms see
# the same synchronized source frames: 520 raw POV rows per action mode and seed. This measures
# event-centered held-out diffusion sensitivity, not generated-event classification accuracy.
set -euo pipefail

project_dir=${MIRA_PROJECT_DIR:-$PWD}
run_root=${CS1K_RUN_ROOT:?Set CS1K_RUN_ROOT to the completed single/shared pipeline}
python_bin=${MIRA_PYTHON:-$project_dir/.pixi/envs/default/bin/python}
eval_root=${CS1K_DEATH_EVAL_ROOT:-$run_root/evaluation/death_action_loss_seed_sweep}
eval_seeds=${CS1K_EVAL_SEEDS:-"37 38 39 40 41"}
test_rounds=${CS1K_TEST_ROUNDS:-52}
action_modes=${CS1K_ACTION_MODES:-"true batch-shifted time-shifted zero"}

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
  echo "Final single and shared checkpoints are required under $run_root" >&2
  exit 1
fi

mkdir -p "$eval_root/provenance"
sha256sum "$single_checkpoint" "$shared_checkpoint" >"$eval_root/provenance/checkpoints.sha256"
git rev-parse HEAD >"$eval_root/provenance/evaluator_code_commit.txt"
git status --porcelain=v1 >"$eval_root/provenance/evaluator_code_status.txt"
git diff --binary >"$eval_root/provenance/evaluator_code.patch"
printf '%s\n' "$eval_seeds" >"$eval_root/provenance/seeds.txt"
printf '%s\n' "$action_modes" >"$eval_root/provenance/action_modes.txt"
printf '%s\n' "$test_rounds" >"$eval_root/provenance/test_rounds.txt"
printf '%s\n' "first-death" >"$eval_root/provenance/window_mode.txt"

for seed in $eval_seeds; do
  for arm in single shared; do
    if [[ "$arm" == single ]]; then
      checkpoint=$single_checkpoint
      val_samples=$((test_rounds * 10))
      eval_batch_size=10
    else
      checkpoint=$shared_checkpoint
      val_samples=$test_rounds
      eval_batch_size=1
    fi
    arm_dir=$eval_root/seed_$seed/$arm
    mkdir -p "$arm_dir"
    for action_mode in $action_modes; do
      "$python_bin" scripts/eval_world_model_offline.py "$checkpoint" \
        --split test \
        --window-mode first-death \
        --action-mode "$action_mode" \
        --seed "$seed" \
        --deterministic \
        --dino-model dinov2_vitb14 \
        --per-device-batch-size "$eval_batch_size" \
        --val-n-samples "$val_samples" \
        --skip-metrics \
        --no-compile \
        --results-json "$arm_dir/$action_mode.json" \
        2>&1 | tee "$arm_dir/$action_mode.log"
    done
  done
done

"$python_bin" scripts/summarize_cs2_action_ablation.py "$eval_root"
