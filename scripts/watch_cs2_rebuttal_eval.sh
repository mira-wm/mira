#!/usr/bin/env bash
# Launch the paired all-test-round evaluation only after the training pipeline succeeds.
set -euo pipefail

run_root=${1:?Usage: watch_cs2_rebuttal_eval.sh RUN_ROOT EVAL_PROJECT PYTHON}
eval_project=${2:?}
python_bin=${3:?}
status_file=$run_root/post_pipeline_status.tsv
eval_root=$run_root/evaluation/test_seed_sweep
pipeline_pid=$(cat "$run_root/pipeline.pid")

printf '%s\twatching\tpipeline_pid=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$pipeline_pid" \
  >>"$status_file"
while kill -0 "$pipeline_pid" 2>/dev/null; do
  sleep 60
done

if ! tail -n 1 "$run_root/pipeline_status.tsv" | grep -q $'\tpipeline\tcomplete$'; then
  printf '%s\tblocked\tpipeline_not_complete\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >>"$status_file"
  exit 1
fi

mkdir -p "$eval_root"
if ! mkdir "$eval_root/.launch_lock" 2>/dev/null; then
  printf '%s\tskipped\tevaluation_already_claimed\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >>"$status_file"
  exit 0
fi

printf '%s\tevaluation\trunning\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$status_file"
set +e
MIRA_PROJECT_DIR="$eval_project" \
MIRA_PYTHON="$python_bin" \
CS1K_RUN_ROOT="$run_root" \
CS1K_EVAL_ROOT="$eval_root" \
  "$eval_project/scripts/run_cs2_rebuttal_eval.sh"
result=$?
set -e

if [[ $result -eq 0 ]]; then
  state=complete
else
  state=failed_exit_$result
fi
printf '%s\tevaluation\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$state" >>"$status_file"
exit "$result"
