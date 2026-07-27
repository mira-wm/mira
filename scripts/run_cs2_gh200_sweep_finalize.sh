#!/usr/bin/env bash
# Fail-closed final aggregation after every training audit and event-probe job succeeds.
set -euo pipefail

project_dir=${MIRA_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
python_bin=${MIRA_PYTHON:-$project_dir/.pixi/envs/default/bin/python}
output_root=${CS1K_OUTPUT_ROOT:?Set the shared sweep output root}
expected_mira_commit=${CS1K_EXPECTED_MIRA_COMMIT:-}

: "${SLURM_JOB_ID:?Run final aggregation through the submitted Slurm dependency job}"
cd "$project_dir"
if [[ -n "$(git status --porcelain=v1)" ]]; then
  echo "Final publication aggregation requires a clean MIRA source tree" >&2
  exit 1
fi
if [[ -n "$expected_mira_commit" ]] &&
  [[ "$(git rev-parse HEAD)" != "$expected_mira_commit" ]]; then
  echo "MIRA commit drifted after sweep submission" >&2
  exit 1
fi

"$python_bin" "$project_dir/scripts/summarize_cs2_gh200_sweep.py" "$output_root"
"$python_bin" "$project_dir/scripts/summarize_cs2_event_probe_sweep.py" "$output_root"

printf 'Final audited results:\n'
printf '  %s\n' "$output_root/sweep_summary.json"
printf '  %s\n' "$output_root/event_probe_sweep_summary.json"
