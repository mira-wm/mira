#!/usr/bin/env bash
# Submit the pinned 360p Dust2 download and materialization as one Clariden batch job.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /path/to/clariden_sync_sweep.env" >&2
  exit 2
fi
environment_file=$(realpath "$1")
if [[ ! -f "$environment_file" ]]; then
  echo "Environment file not found: $environment_file" >&2
  exit 1
fi
# shellcheck disable=SC1090
set -a
source "$environment_file"
set +a

project_dir=${MIRA_PROJECT_DIR:?Set the pinned MIRA checkout}
data_root=${CS1K_DATASET_DIR:?Set the Dust2 materialization root}
account=${CS1K_SLURM_ACCOUNT:?Set the Clariden Slurm account}
partition=${CS1K_GH200_PARTITION:-normal}
uenv_label=${CS1K_CLARIDEN_UENV:-pytorch/v2.8.0:v1}
stage_time=${CS1K_DATASET_STAGE_TIME:-12:00:00}

for command_name in realpath sbatch squeue; do
  if ! command -v "$command_name" >/dev/null; then
    echo "Required Clariden command not found: $command_name" >&2
    exit 1
  fi
done
if [[ -n "$(squeue -h -u "$USER" -n mira-cs2-dataset-stage)" ]]; then
  echo "A mira-cs2-dataset-stage job is already queued or running" >&2
  exit 1
fi
if [[ -f "$data_root/dataset_stage_complete.json" ]]; then
  echo "Dust2 dataset staging is already complete: $data_root/dataset_stage_complete.json"
  exit 0
fi

mkdir -p "$data_root/logs"
job_id=$(sbatch \
  --parsable \
  --account="$account" \
  --partition="$partition" \
  --job-name=mira-cs2-dataset-stage \
  --nodes=1 \
  --ntasks=1 \
  --time="$stage_time" \
  --uenv="$uenv_label:/user-environment" \
  --view=default \
  --output="$data_root/logs/dataset-stage-%j.log" \
  --export=ALL \
  "$project_dir/scripts/run_cs2_clariden_dataset_stage.sh" \
  "$environment_file")
job_id=${job_id%%;*}
if ! [[ "$job_id" =~ ^[0-9]+$ ]]; then
  echo "Could not parse staging job id: $job_id" >&2
  exit 1
fi

printf 'Submitted Dust2 dataset staging job %s\n' "$job_id"
printf 'Monitor: squeue -j %s\n' "$job_id"
printf 'Log: %s/logs/dataset-stage-%s.log\n' "$data_root" "$job_id"
