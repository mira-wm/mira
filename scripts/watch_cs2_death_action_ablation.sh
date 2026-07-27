#!/usr/bin/env bash
# Launch the death-centered action diagnostic only after the midpoint diagnostic succeeds.
set -uo pipefail

run_root=$1
eval_project=$2
python_bin=$3
status_file=$run_root/death_action_ablation_status.tsv
action_watcher_pid=$(cat "$run_root/action_ablation_watcher.pid")

printf '%s\twatching\taction_watcher_pid=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$action_watcher_pid" >>"$status_file"
while kill -0 "$action_watcher_pid" 2>/dev/null; do
  sleep 60
done

if ! tail -n 1 "$run_root/action_ablation_status.tsv" |
  grep -q $'\taction_evaluation\tcomplete$'; then
  printf '%s\tblocked\tmidpoint_action_evaluation_incomplete\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$status_file"
  exit 1
fi

eval_root=$run_root/evaluation/death_action_loss_seed_sweep
mkdir -p "$eval_root"
if ! mkdir "$eval_root/.launch_lock" 2>/dev/null; then
  printf '%s\tskipped\tdeath_evaluation_already_claimed\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$status_file"
  exit 0
fi

printf '%s\tdeath_action_evaluation\trunning\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$status_file"
set +e
MIRA_PROJECT_DIR="$eval_project" \
MIRA_PYTHON="$python_bin" \
CS1K_RUN_ROOT="$run_root" \
CS1K_DEATH_EVAL_ROOT="$eval_root" \
  "$eval_project/scripts/run_cs2_death_action_ablation.sh"
result=$?
set -e

if [[ $result -eq 0 ]]; then
  state=complete
else
  state=failed_exit_$result
fi
printf '%s\tdeath_action_evaluation\t%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$state" >>"$status_file"
exit "$result"
