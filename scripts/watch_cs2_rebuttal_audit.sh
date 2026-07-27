#!/usr/bin/env bash
# Wait for the automatic CS2 pilot evaluations, stop GPU telemetry, and certify the full run.
set -euo pipefail

run_root=${1:?Usage: watch_cs2_rebuttal_audit.sh RUN_ROOT PROJECT_DIR PYTHON TRAIN_COMMIT EVAL_COMMIT}
project_dir=${2:?}
python_bin=${3:?}
training_commit=${4:?}
evaluator_commit=${5:?}
status_file=$run_root/final_audit_status.tsv
death_watcher_pid=$(cat "$run_root/death_action_ablation_watcher.pid")

printf '%s\twatching\tdeath_watcher_pid=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$death_watcher_pid" >>"$status_file"
while kill -0 "$death_watcher_pid" 2>/dev/null; do
  sleep 60
done

if ! tail -n 1 "$run_root/death_action_ablation_status.tsv" |
  grep -q $'\tdeath_action_evaluation\tcomplete$'; then
  printf '%s\tblocked\tdeath_action_evaluation_incomplete\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$status_file"
  exit 1
fi

gpu_pid_file=$run_root/provenance/gpu_timeseries.pid
if [[ -s "$gpu_pid_file" ]]; then
  gpu_pid=$(cat "$gpu_pid_file")
  gpu_command=$(ps -p "$gpu_pid" -o comm= 2>/dev/null || true)
  if [[ "$gpu_command" == nvidia-smi ]] && kill -0 "$gpu_pid" 2>/dev/null; then
    kill "$gpu_pid"
    for _ in 1 2 3 4 5; do
      kill -0 "$gpu_pid" 2>/dev/null || break
      sleep 1
    done
  fi
fi

printf '%s\taudit\trunning\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$status_file"
cd "$project_dir"
export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1

set +e
"$python_bin" scripts/audit_cs2_rebuttal_run.py "$run_root" \
  --expected-training-commit "$training_commit" \
  --expected-evaluator-commit "$evaluator_commit" \
  --output "$run_root/audit.json" \
  >"$run_root/final_audit.log" 2>&1
result=$?
set -e

if [[ $result -eq 0 ]]; then
  state=complete
else
  state=failed_exit_$result
fi
printf '%s\taudit\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$state" >>"$status_file"
exit "$result"
