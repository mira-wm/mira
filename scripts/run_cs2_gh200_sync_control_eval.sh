#!/usr/bin/env bash
# Deterministic held-out evaluation for the GH200 synchronized-vs-shuffled training control.
#
# Both ten-player models are evaluated on the same synchronized test groups. The only difference
# between arms is their training grouping, captured as training_group_mode in each result JSON.
set -euo pipefail

project_dir=${MIRA_PROJECT_DIR:-$PWD}
training_root=${CS1K_TRAINING_ROOT:?Set the seed-specific GH200 training directory}
dataset_dir=${CS1K_DATASET_DIR:?Set the CounterStrike-1K materialization directory}
manifest_path=${CS1K_MANIFEST_PATH:?Set the frozen confirmatory manifest path}
split_provenance=${CS1K_CONFIRMATORY_SPLIT_PROVENANCE:?Set the frozen split provenance}
python_bin=${MIRA_PYTHON:-$project_dir/.pixi/envs/default/bin/python}
eval_root=${CS1K_EVAL_ROOT:-$training_root/evaluation/synchronized_test_seed_sweep}
action_eval_root=${CS1K_ACTION_EVAL_ROOT:-$training_root/evaluation/synchronized_test_action_loss_seed_sweep}
death_action_eval_root=${CS1K_DEATH_ACTION_EVAL_ROOT:-$training_root/evaluation/synchronized_test_first_death_action_loss_seed_sweep}
eval_seeds=${CS1K_EVAL_SEEDS:-"37 38 39 40 41"}
action_modes=${CS1K_ACTION_MODES:-"true batch-shifted time-shifted zero"}
expected_training_commit=${CS1K_EXPECTED_TRAINING_COMMIT:-}

cd "$project_dir"
export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB_MODE=disabled
evaluator_commit=$(git rev-parse HEAD)
if [[ -z "$expected_training_commit" ]]; then
  expected_training_commit=$evaluator_commit
fi

if [[ -n "$(git status --porcelain=v1)" ]]; then
  echo "GH200 publication evaluation requires a clean source tree" >&2
  exit 1
fi
if [[ ! -f "$manifest_path" || ! -f "$split_provenance" ]]; then
  echo "Frozen confirmatory manifest and provenance are required" >&2
  exit 1
fi
if [[ "$(dirname "$(realpath "$manifest_path")")" != "$(realpath "$dataset_dir")" ]]; then
  echo "CS1K_MANIFEST_PATH must live directly inside CS1K_DATASET_DIR" >&2
  exit 1
fi

test_rounds=$(
  "$python_bin" - "$split_provenance" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(payload["statistics"]["splits"]["test"]["rounds"])
PY
)
if [[ -n "${CS1K_TEST_ROUNDS:-}" && "$CS1K_TEST_ROUNDS" != "$test_rounds" ]]; then
  echo "CS1K_TEST_ROUNDS=$CS1K_TEST_ROUNDS disagrees with frozen split count $test_rounds" >&2
  exit 1
fi

latest_checkpoint() {
  local arm=$1
  find "$training_root/$arm" -path '*/checkpoint-*/checkpoint.pth' -print0 |
    sort -zV | tail -z -n 1 | tr -d '\0'
}

shuffled_checkpoint=$(latest_checkpoint shuffled)
synchronized_checkpoint=$(latest_checkpoint synchronized)
if [[ -z "$shuffled_checkpoint" || -z "$synchronized_checkpoint" ]]; then
  echo "Final shuffled and synchronized checkpoints are required under $training_root" >&2
  exit 1
fi

mkdir -p "$eval_root/provenance"
"$python_bin" scripts/prepare_cs2_confirmatory_split.py \
  --source-manifest "$dataset_dir/manifest.parquet" \
  --output-manifest "$manifest_path" \
  --provenance-output "$split_provenance" \
  --verify-only \
  >"$eval_root/provenance/confirmatory_split_verification.json"
printf '%s\n' "$shuffled_checkpoint" >"$eval_root/provenance/shuffled_checkpoint.txt"
printf '%s\n' "$synchronized_checkpoint" >"$eval_root/provenance/synchronized_checkpoint.txt"
sha256sum "$shuffled_checkpoint" "$synchronized_checkpoint" \
  >"$eval_root/provenance/checkpoints.sha256"
git rev-parse HEAD >"$eval_root/provenance/evaluator_code_commit.txt"
git status --porcelain=v1 >"$eval_root/provenance/evaluator_code_status.txt"
git diff --binary >"$eval_root/provenance/evaluator_code.patch"
printf '%s\n' "$eval_seeds" >"$eval_root/provenance/seeds.txt"
printf '%s\n' "$test_rounds" >"$eval_root/provenance/test_rounds.txt"
sha256sum "$manifest_path" "$split_provenance" \
  >"$eval_root/provenance/confirmatory_split_files.sha256"

"$python_bin" - "$manifest_path" "$shuffled_checkpoint" "$synchronized_checkpoint" \
  >"$eval_root/provenance/checkpoint_config_verification.json" <<'PY'
import json
import sys
from pathlib import Path

import yaml

manifest = str(Path(sys.argv[1]).resolve())
result = {}
for arm, checkpoint in zip(("shuffled", "synchronized"), sys.argv[2:], strict=True):
    config_path = Path(checkpoint).resolve().parents[1] / "world_model_config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    observed_manifest = str(Path(config["dataset"]["test_index"]).resolve())
    observed_group = config["dataset"]["group_mode"]
    observed_routing = config["model"]["architecture"]["config"].get(
        "action_routing", "global_mean"
    )
    if observed_manifest != manifest:
        raise ValueError(f"{arm} checkpoint points at {observed_manifest}, expected {manifest}")
    if observed_group != arm:
        raise ValueError(f"{arm} checkpoint records group_mode={observed_group}")
    if observed_routing != "spatial":
        raise ValueError(f"{arm} checkpoint records action_routing={observed_routing}")
    result[arm] = {
        "config": str(config_path),
        "manifest": observed_manifest,
        "training_group_mode": observed_group,
        "action_routing": observed_routing,
    }
print(json.dumps({"verified": True, "checkpoints": result}, indent=2, sort_keys=True))
PY

write_action_provenance() {
  local root=$1
  local window_mode=$2
  mkdir -p "$root/provenance"
  printf '%s\n' "$shuffled_checkpoint" >"$root/provenance/shuffled_checkpoint.txt"
  printf '%s\n' "$synchronized_checkpoint" >"$root/provenance/synchronized_checkpoint.txt"
  sha256sum "$shuffled_checkpoint" "$synchronized_checkpoint" \
    >"$root/provenance/checkpoints.sha256"
  git rev-parse HEAD >"$root/provenance/evaluator_code_commit.txt"
  git status --porcelain=v1 >"$root/provenance/evaluator_code_status.txt"
  git diff --binary >"$root/provenance/evaluator_code.patch"
  printf '%s\n' "$eval_seeds" >"$root/provenance/seeds.txt"
  printf '%s\n' "$action_modes" >"$root/provenance/action_modes.txt"
  printf '%s\n' "$test_rounds" >"$root/provenance/test_rounds.txt"
  printf '%s\n' "$window_mode" >"$root/provenance/window_mode.txt"
  sha256sum "$manifest_path" "$split_provenance" \
    >"$root/provenance/confirmatory_split_files.sha256"
}
write_action_provenance "$action_eval_root" midpoint
write_action_provenance "$death_action_eval_root" first-death

for seed in $eval_seeds; do
  seed_dir=$eval_root/seed_$seed
  mkdir -p "$seed_dir"

  for arm in shuffled synchronized; do
    if [[ "$arm" == shuffled ]]; then
      checkpoint=$shuffled_checkpoint
    else
      checkpoint=$synchronized_checkpoint
    fi
    "$python_bin" scripts/eval_world_model_offline.py "$checkpoint" \
      --split test \
      --group-mode synchronized \
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
      --output-dir "$seed_dir/${arm}_viz" \
      --results-json "$seed_dir/$arm.json" \
      2>&1 | tee "$seed_dir/$arm.log"

    action_arm_dir=$action_eval_root/seed_$seed/$arm
    mkdir -p "$action_arm_dir"
    for action_mode in $action_modes; do
      "$python_bin" scripts/eval_world_model_offline.py "$checkpoint" \
        --split test \
        --group-mode synchronized \
        --action-mode "$action_mode" \
        --seed "$seed" \
        --deterministic \
        --dino-model dinov2_vitb14 \
        --val-n-samples "$test_rounds" \
        --skip-metrics \
        --no-compile \
        --results-json "$action_arm_dir/$action_mode.json" \
        2>&1 | tee "$action_arm_dir/$action_mode.log"
    done

    death_action_arm_dir=$death_action_eval_root/seed_$seed/$arm
    mkdir -p "$death_action_arm_dir"
    for action_mode in $action_modes; do
      "$python_bin" scripts/eval_world_model_offline.py "$checkpoint" \
        --split test \
        --group-mode synchronized \
        --window-mode first-death \
        --action-mode "$action_mode" \
        --seed "$seed" \
        --deterministic \
        --dino-model dinov2_vitb14 \
        --val-n-samples "$test_rounds" \
        --skip-metrics \
        --no-compile \
        --results-json "$death_action_arm_dir/$action_mode.json" \
        2>&1 | tee "$death_action_arm_dir/$action_mode.log"
    done
  done
done

"$python_bin" scripts/summarize_cs2_rebuttal_eval.py "$eval_root" \
  --arm-a shuffled \
  --arm-b synchronized \
  --arm-a-eval-group-mode synchronized \
  --arm-b-eval-group-mode synchronized \
  --arm-a-n-players 10 \
  --arm-b-n-players 10 \
  --arm-a-training-group-mode shuffled \
  --arm-b-training-group-mode synchronized \
  --expected-action-routing spatial

"$python_bin" scripts/summarize_cs2_action_ablation.py "$action_eval_root" \
  --arm-a shuffled \
  --arm-b synchronized \
  --arm-a-eval-group-mode synchronized \
  --arm-b-eval-group-mode synchronized \
  --arm-a-n-players 10 \
  --arm-b-n-players 10 \
  --arm-a-training-group-mode shuffled \
  --arm-b-training-group-mode synchronized \
  --expected-action-routing spatial

"$python_bin" scripts/summarize_cs2_action_ablation.py "$death_action_eval_root" \
  --arm-a shuffled \
  --arm-b synchronized \
  --arm-a-eval-group-mode synchronized \
  --arm-b-eval-group-mode synchronized \
  --arm-a-n-players 10 \
  --arm-b-n-players 10 \
  --arm-a-training-group-mode shuffled \
  --arm-b-training-group-mode synchronized \
  --expected-action-routing spatial

read -r train_steps arm_hours < <(
  "$python_bin" - "$synchronized_checkpoint" <<'PY'
import sys
from pathlib import Path

import yaml

checkpoint = Path(sys.argv[1]).resolve()
config = yaml.safe_load((checkpoint.parents[1] / "world_model_config.yaml").read_text())
print(config["run"]["steps"], config["run"]["max_duration_hours"])
PY
)
"$python_bin" scripts/audit_cs2_gh200_sync_control.py "$training_root" \
  --manifest "$manifest_path" \
  --split-provenance "$split_provenance" \
  --train-steps "$train_steps" \
  --arm-hours "$arm_hours" \
  --expected-training-commit "$expected_training_commit" \
  --expected-evaluator-commit "$evaluator_commit"
