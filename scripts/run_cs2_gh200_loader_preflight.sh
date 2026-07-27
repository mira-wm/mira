#!/usr/bin/env bash
# Run repeated exact-contract loader benchmarks on one representative GH200 of the allocated node.
#
# Invoke once before the four-local-rank training step. Results are written to
# BENCHMARK_ROOT/<hostname>/ so the launcher can prove that the selected configuration came from
# three independent repeats on the exact Clariden node used for training.
set -euo pipefail

project_dir=${MIRA_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
python_bin=${MIRA_PYTHON:-$project_dir/.pixi/envs/default/bin/python}
dataset_dir=${CS1K_DATASET_DIR:?Set CS1K_DATASET_DIR}
manifest_path=${CS1K_MANIFEST_PATH:?Set the frozen confirmatory manifest}
benchmark_root=${CS1K_LOADER_BENCHMARK_ROOT:?Set CS1K_LOADER_BENCHMARK_ROOT}
worker_candidates=${CS1K_DATALOADER_WORKER_CANDIDATES:-"4 8 12"}
prefetch_factor=${CS1K_DATALOADER_PREFETCH_FACTOR:-2}
persistent_workers=${CS1K_DATALOADER_PERSISTENT_WORKERS:-true}
pin_memory=${CS1K_DATALOADER_PIN_MEMORY:-true}
repeat_count=${CS1K_LOADER_BENCHMARK_REPEATS:-3}
warmup_batches=${CS1K_LOADER_BENCHMARK_WARMUP_BATCHES:-10}
timed_batches=${CS1K_LOADER_BENCHMARK_TIMED_BATCHES:-100}
expected_manifest_sha256=${CS1K_EXPECTED_MANIFEST_SHA256:-33abbb623072932431871a612620110c473d4b664c52010e5763c273c6daf10e}
node_hostname=$(hostname)

if [[ -z ${SLURM_JOB_ID:-} || -z ${SLURM_PROCID:-} || -z ${SLURM_NODEID:-} ]]; then
  echo "GH200 publication loader evidence must run inside a Slurm step" >&2
  exit 1
fi
if ! [[ "$repeat_count" =~ ^[0-9]+$ ]] || (( repeat_count < 3 )); then
  echo "CS1K_LOADER_BENCHMARK_REPEATS must be an integer >= 3" >&2
  exit 1
fi
if ! [[ "$prefetch_factor" =~ ^[0-9]+$ ]] || (( prefetch_factor < 1 )); then
  echo "CS1K_DATALOADER_PREFETCH_FACTOR must be a positive integer" >&2
  exit 1
fi
if [[ "$persistent_workers" != true && "$persistent_workers" != false ]]; then
  echo "CS1K_DATALOADER_PERSISTENT_WORKERS must be true or false" >&2
  exit 1
fi
if [[ "$pin_memory" != true && "$pin_memory" != false ]]; then
  echo "CS1K_DATALOADER_PIN_MEMORY must be true or false" >&2
  exit 1
fi

read -r -a workers <<<"$worker_candidates"
if [[ ${#workers[@]} -lt 2 ]]; then
  echo "Provide at least two positive worker candidates for the target-node comparison" >&2
  exit 1
fi
for worker_count in "${workers[@]}"; do
  if ! [[ "$worker_count" =~ ^[0-9]+$ ]] || (( worker_count < 1 )); then
    echo "Worker candidates must be positive integers, found $worker_count" >&2
    exit 1
  fi
done

cd "$project_dir"
export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
if [[ -n "$(git status --porcelain=v1)" ]]; then
  echo "GH200 loader benchmarks require a clean source tree" >&2
  exit 1
fi
if [[ ! -f "$manifest_path" ]]; then
  echo "Confirmatory manifest not found: $manifest_path" >&2
  exit 1
fi
if [[ "$(dirname "$(realpath "$manifest_path")")" != "$(realpath "$dataset_dir")" ]]; then
  echo "CS1K_MANIFEST_PATH must live directly inside CS1K_DATASET_DIR" >&2
  exit 1
fi
observed_manifest_sha256=$(sha256sum "$manifest_path" | cut -d' ' -f1)
if [[ "$observed_manifest_sha256" != "$expected_manifest_sha256" ]]; then
  echo "Confirmatory manifest SHA-256 drifted: $observed_manifest_sha256" >&2
  exit 1
fi

mapfile -t visible_gpu_names < <(
  nvidia-smi --query-gpu=name --format=csv,noheader | sed 's/[[:space:]]*$//'
)
if [[ ${#visible_gpu_names[@]} -ne 4 ]]; then
  echo "Expected the allocated node's four scheduler-visible GH200s, found: ${visible_gpu_names[*]:-none}" >&2
  exit 1
fi
for gpu_name in "${visible_gpu_names[@]}"; do
  if [[ "${gpu_name,,}" != *gh200* ]]; then
    echo "Expected four GH200s, found: ${visible_gpu_names[*]}" >&2
    exit 1
  fi
done
visible_gpu_names_joined=$(IFS='|'; printf '%s' "${visible_gpu_names[*]}")

node_root=$benchmark_root/$node_hostname
mkdir -p "$node_root"
printf '%s\n' \
  "schema=mira-cs2-gh200-loader-preflight-v1" \
  "hostname=$node_hostname" \
  "slurm_job_id=$SLURM_JOB_ID" \
  "slurm_procid=$SLURM_PROCID" \
  "slurm_nodeid=$SLURM_NODEID" \
  "visible_gpu_count=${#visible_gpu_names[@]}" \
  "visible_gpu_names=$visible_gpu_names_joined" \
  "git_commit=$(git rev-parse HEAD)" \
  "manifest_path=$(realpath "$manifest_path")" \
  "manifest_sha256=$observed_manifest_sha256" \
  "worker_candidates=${workers[*]}" \
  "prefetch_factor=$prefetch_factor" \
  "persistent_workers=$persistent_workers" \
  "pin_memory=$pin_memory" \
  "repeat_count=$repeat_count" \
  "warmup_batches=$warmup_batches" \
  "timed_batches=$timed_batches" \
  >"$node_root/benchmark_contract.env"

persistent_flag=--no-persistent-workers
if [[ "$persistent_workers" == true ]]; then
  persistent_flag=--persistent-workers
fi
pin_flag=--no-pin-memory
if [[ "$pin_memory" == true ]]; then
  pin_flag=--pin-memory
fi

for ((repeat = 0; repeat < repeat_count; repeat++)); do
  output=$node_root/repeat_$(printf '%02d' "$repeat").json
  log=$node_root/repeat_$(printf '%02d' "$repeat").log
  if [[ -e "$output" || -e "$log" ]]; then
    echo "Refusing to overwrite existing independent repeat: $output or $log" >&2
    exit 1
  fi
  "$python_bin" scripts/bench_cs2_dataloader.py \
    --data-root "$dataset_dir" \
    --manifest "$manifest_path" \
    --output "$output" \
    --split val \
    --map-slug dust2 \
    --group-modes synchronized shuffled \
    --workers "${workers[@]}" \
    --prefetch-factor "$prefetch_factor" \
    "$persistent_flag" \
    "$pin_flag" \
    --shuffle \
    --clip-len 16 \
    --target-fps 8 \
    --frame-height 168 \
    --frame-width 308 \
    --seed 28 \
    --warmup-batches "$warmup_batches" \
    --timed-batches "$timed_batches" \
    --transfer-device cuda \
    2>&1 | tee "$log"
done
