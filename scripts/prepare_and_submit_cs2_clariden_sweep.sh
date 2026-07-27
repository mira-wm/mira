#!/usr/bin/env bash
# One login-node command to prepare the pinned Clariden ARM64 runtime and submit the full sweep.
#
# Usage:
#   bash scripts/prepare_and_submit_cs2_clariden_sweep.sh /path/to/clariden_sync_sweep.env
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
release_dir=${CS1K_RELEASE_DIR:?Set the pinned CounterStrike-1K checkout}
venv_dir=${CS1K_CLARIDEN_VENV:?Set the persistent Clariden venv path}
uenv_label=${CS1K_CLARIDEN_UENV:-pytorch/v2.8.0:v1}

for command_name in bash git python3 sbatch sha256sum uenv; do
  if ! command -v "$command_name" >/dev/null; then
    echo "Required Clariden login-node command not found: $command_name" >&2
    exit 1
  fi
done
if uenv status 2>/dev/null | grep -q '^uenv '; then
  echo "Run this command from a clean login shell, not inside an active uenv" >&2
  exit 1
fi
for repo in "$project_dir" "$release_dir"; do
  if [[ ! -d "$repo/.git" ]]; then
    echo "Pinned checkout not found: $repo" >&2
    exit 1
  fi
done

export MIRA_PROJECT_DIR="$project_dir"
export CS1K_RELEASE_DIR="$release_dir"
export CS1K_CLARIDEN_VENV="$venv_dir"
export CS1K_CLARIDEN_UENV="$uenv_label"

uenv run --view=default "$uenv_label" -- \
  bash "$project_dir/scripts/setup_cs2_clariden_uenv.sh"

export MIRA_PYTHON="$venv_dir/bin/python"
export CS1K_RELEASE_PYTHON="$venv_dir/bin/python"
export CS1K_SUBMIT_PYTHON=${CS1K_SUBMIT_PYTHON:-python3}
export CS1K_GH200_PARTITION=${CS1K_GH200_PARTITION:-normal}
export CS1K_TRAIN_SBATCH_ARGS=${CS1K_TRAIN_SBATCH_ARGS:---uenv=$uenv_label:/user-environment --view=default}
export CS1K_AUX_SBATCH_ARGS=${CS1K_AUX_SBATCH_ARGS:---uenv=$uenv_label:/user-environment --view=default}

exec bash "$project_dir/scripts/submit_cs2_gh200_sweep.sh"
