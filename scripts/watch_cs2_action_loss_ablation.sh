#!/usr/bin/env bash
# Launch the midpoint action diagnostic only after the paired rollout evaluation succeeds.
set -euo pipefail

run_root=${1:?Usage: watch_cs2_action_loss_ablation.sh RUN_ROOT EVAL_PROJECT PYTHON}
eval_project=${2:?}
python_bin=${3:?}
status_file=$run_root/action_ablation_status.tsv
pipeline_pid=$(cat "$run_root/pipeline.pid")
primary_watcher_pid=$(cat "$run_root/post_pipeline_watcher.pid")

printf '%s\twatching\tpipeline_pid=%s primary_watcher_pid=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$pipeline_pid" "$primary_watcher_pid" >>"$status_file"
while kill -0 "$pipeline_pid" 2>/dev/null; do
  sleep 60
done
while kill -0 "$primary_watcher_pid" 2>/dev/null; do
  sleep 60
done

primary_summary=$run_root/evaluation/test_seed_sweep/summary.json
if [[ ! -s "$primary_summary" ]]; then
  printf '%s\tblocked\tprimary_evaluation_incomplete\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >>"$status_file"
  exit 1
fi

action_root=$run_root/evaluation/action_loss_seed_sweep
mkdir -p "$action_root"
if ! mkdir "$action_root/.launch_lock" 2>/dev/null; then
  printf '%s\tskipped\taction_evaluation_already_claimed\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$status_file"
  exit 0
fi

printf '%s\taction_evaluation\trunning\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$status_file"
set +e
MIRA_PROJECT_DIR="$eval_project" \
MIRA_PYTHON="$python_bin" \
CS1K_RUN_ROOT="$run_root" \
CS1K_ACTION_EVAL_ROOT="$action_root" \
  "$eval_project/scripts/run_cs2_action_loss_ablation.sh"
result=$?
set -e

if [[ $result -eq 0 ]]; then
  state=complete
else
  state=failed_exit_$result
fi
printf '%s\taction_evaluation\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$state" \
  >>"$status_file"
exit "$result"
