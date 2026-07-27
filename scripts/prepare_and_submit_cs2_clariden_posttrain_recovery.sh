#!/usr/bin/env bash
# Submit checkpoint-only Clariden recovery without reinstalling or rerunning training/preflight.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /path/to/clariden_sync_sweep.env" >&2
  exit 2
fi
environment_file=$1
if [[ ! -f "$environment_file" ]]; then
  echo "Environment file not found: $environment_file" >&2
  exit 1
fi
# shellcheck disable=SC1090
set -a
source "$environment_file"
set +a

project_dir=${MIRA_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
venv_dir=${CS1K_CLARIDEN_VENV:?Set the persistent Clariden venv path}
uenv_label=${CS1K_CLARIDEN_UENV:-pytorch/v2.8.0:v1}

for command_name in bash git sbatch uenv; do
  if ! command -v "$command_name" >/dev/null; then
    echo "Required Clariden login-node command not found: $command_name" >&2
    exit 1
  fi
done
if uenv status 2>/dev/null | grep -q '^uenv '; then
  echo "Run this command from a clean login shell, not inside an active uenv" >&2
  exit 1
fi

export MIRA_PROJECT_DIR="$project_dir"
export MIRA_PYTHON="$venv_dir/bin/python"
export CS1K_RELEASE_PYTHON="$venv_dir/bin/python"
export CS1K_SUBMIT_PYTHON=${CS1K_SUBMIT_PYTHON:-python3}
export CS1K_GH200_PARTITION=${CS1K_GH200_PARTITION:-normal}
export CS1K_AUX_SBATCH_ARGS=${CS1K_AUX_SBATCH_ARGS:---uenv=$uenv_label:/user-environment --view=default}

exec bash "$project_dir/scripts/submit_cs2_gh200_posttrain_recovery.sh"
