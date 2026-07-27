#!/usr/bin/env bash
# Slurm-native orchestration for one audited full-node Clariden GH200 training seed.
#
# Submit this file with site-specific account/partition/time flags, for example:
#   sbatch --nodes=1 --ntasks-per-node=1 --gpus-per-node=4 ... \
#     scripts/run_cs2_gh200_slurm_seed.sh
#
# Required CS1K_* paths, fixed optimizer-update count, and fail-closed wall-clock cap must be
# exported with --export or by the batch environment. The script benchmarks the allocated node,
# freezes one deterministic loader configuration, trains both arms with four local DDP processes,
# evaluates the untouched test, and audits.
set -euo pipefail

project_dir=${MIRA_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
python_bin=${MIRA_PYTHON:-$project_dir/.pixi/envs/default/bin/python}
dataset_dir=${CS1K_DATASET_DIR:?Set CS1K_DATASET_DIR}
manifest_path=${CS1K_MANIFEST_PATH:?Set CS1K_MANIFEST_PATH}
split_provenance=${CS1K_CONFIRMATORY_SPLIT_PROVENANCE:?Set split provenance}
codec_checkpoint=${CS1K_CODEC_CHECKPOINT:?Set CS1K_CODEC_CHECKPOINT}
output_root=${CS1K_OUTPUT_ROOT:?Set CS1K_OUTPUT_ROOT}
train_steps=${CS1K_TRAIN_STEPS:?Set CS1K_TRAIN_STEPS}
arm_hours=${CS1K_ARM_HOURS:?Set CS1K_ARM_HOURS as a safety cap}
seed=${CS1K_SEED:-28}
prefetch_factor=${CS1K_DATALOADER_PREFETCH_FACTOR:-2}
persistent_workers=${CS1K_DATALOADER_PERSISTENT_WORKERS:-true}
pin_memory=${CS1K_DATALOADER_PIN_MEMORY:-true}
expected_manifest_sha256=33abbb623072932431871a612620110c473d4b664c52010e5763c273c6daf10e
expected_codec_sha256=${CS1K_EXPECTED_CODEC_CHECKPOINT_SHA256:-3c286c59b74cd141e72af69cde1a0a005142d8d2b472c789cdf2a39a140c4b7a}
expected_mira_commit=${CS1K_EXPECTED_MIRA_COMMIT:-}
reuse_loader_selection=${CS1K_REUSE_LOADER_SELECTION:-}

: "${SLURM_JOB_ID:?Run through sbatch or inside a one-node, four-GH200 Slurm allocation}"
: "${SLURM_JOB_NODELIST:?SLURM_JOB_NODELIST is required}"
slurm_nodes=${SLURM_NNODES:-${SLURM_JOB_NUM_NODES:-}}
if [[ "$slurm_nodes" != 1 ]]; then
  echo "The publication control requires exactly one Slurm node, found ${slurm_nodes:-unset}" >&2
  exit 1
fi
if ! [[ "$seed" =~ ^[0-9]+$ ]]; then
  echo "CS1K_SEED must be a non-negative integer" >&2
  exit 1
fi
if ! [[ "$train_steps" =~ ^[1-9][0-9]*$ ]]; then
  echo "CS1K_TRAIN_STEPS must be a positive integer" >&2
  exit 1
fi
for command_name in scontrol srun sha256sum; do
  if ! command -v "$command_name" >/dev/null; then
    echo "Required command not found: $command_name" >&2
    exit 1
  fi
done

cd "$project_dir"
export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
if [[ -n "$(git status --porcelain=v1)" ]]; then
  echo "GH200 publication runs require a clean source tree" >&2
  exit 1
fi
if [[ -n "$expected_mira_commit" ]] &&
  [[ "$(git rev-parse HEAD)" != "$expected_mira_commit" ]]; then
  echo "MIRA commit drifted after sweep submission" >&2
  exit 1
fi
for path in "$manifest_path" "$split_provenance" "$codec_checkpoint"; do
  if [[ ! -f "$path" ]]; then
    echo "Required frozen input not found: $path" >&2
    exit 1
  fi
done
if [[ "$(sha256sum "$manifest_path" | cut -d' ' -f1)" != "$expected_manifest_sha256" ]]; then
  echo "Confirmatory manifest SHA-256 drifted" >&2
  exit 1
fi
if [[ "$(sha256sum "$codec_checkpoint" | cut -d' ' -f1)" != "$expected_codec_sha256" ]]; then
  echo "Codec checkpoint SHA-256 drifted" >&2
  exit 1
fi
"$python_bin" -c 'import torch; import torch.distributed.run'

# Torch Hub is not concurrency-safe while downloading/extracting a repository. Populate a
# job-private cache once, before either the loader gate or four DDP ranks start, and then let every
# rank consume the completed cache read-only.
torch_home_root=${CS1K_TORCH_HOME_ROOT:-$output_root/torch_hub}
export TORCH_HOME="$torch_home_root/slurm_$SLURM_JOB_ID"
mkdir -p "$TORCH_HOME"
"$python_bin" scripts/prepare_dinov2_cache.py \
  --torch-home "$TORCH_HOME" \
  --output "$TORCH_HOME/dinov2_cache_ready.json"

mapfile -t allocated_hosts < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
if [[ ${#allocated_hosts[@]} -ne 1 ]]; then
  echo "Expected one hostname from Slurm, found ${allocated_hosts[*]:-none}" >&2
  exit 1
fi
master_addr=${allocated_hosts[0]}
master_port=${MASTER_PORT:-29500}

read -r -a single_gpu_args <<<"${CS1K_SRUN_SINGLE_GPU_ARGS:---gpus-per-task=1}"
read -r -a training_gpu_args <<<"${CS1K_SRUN_TRAINING_GPU_ARGS:---gpus-per-task=4}"
loader_srun=(
  --nodes=1
  --ntasks=1
  --ntasks-per-node=1
  --cpus-per-task=72
  --exclusive
  --kill-on-bad-exit=1
  "${single_gpu_args[@]}"
)
training_srun=(
  --nodes=1
  --ntasks=1
  --ntasks-per-node=1
  --cpus-per-task=288
  --exclusive
  --kill-on-bad-exit=1
  "${training_gpu_args[@]}"
)

benchmark_root=$output_root/loader_benchmarks/slurm_$SLURM_JOB_ID
selection_path=$benchmark_root/frozen_loader_selection_global.json
if [[ -e "$benchmark_root" ]]; then
  echo "Refusing to overwrite existing Slurm loader preflight: $benchmark_root" >&2
  exit 1
fi
mkdir -p "$benchmark_root"

export MIRA_PROJECT_DIR="$project_dir"
export MIRA_PYTHON="$python_bin"
export CS1K_DATASET_DIR="$dataset_dir"
export CS1K_MANIFEST_PATH="$manifest_path"
export CS1K_LOADER_BENCHMARK_ROOT="$benchmark_root"
export CS1K_DATALOADER_PREFETCH_FACTOR="$prefetch_factor"
export CS1K_DATALOADER_PERSISTENT_WORKERS="$persistent_workers"
export CS1K_DATALOADER_PIN_MEMORY="$pin_memory"
export CS1K_EXPECTED_MANIFEST_SHA256="$expected_manifest_sha256"

if [[ -n "$reuse_loader_selection" ]]; then
  if [[ ! -s "$reuse_loader_selection" ]]; then
    echo "Reusable loader selection not found: $reuse_loader_selection" >&2
    exit 1
  fi
  cp "$reuse_loader_selection" "$selection_path"
else
  srun "${loader_srun[@]}" "$project_dir/scripts/run_cs2_gh200_loader_preflight.sh"

  mapfile -d '' -t benchmark_paths < <(
    find "$benchmark_root" -mindepth 2 -maxdepth 2 -type f -name 'repeat_*.json' -print0 | sort -z
  )
  minimum_expected=${CS1K_LOADER_BENCHMARK_REPEATS:-3}
  if [[ ${#benchmark_paths[@]} -lt "$minimum_expected" ]]; then
    echo "Expected at least $minimum_expected node-local benchmark JSONs, found ${#benchmark_paths[@]}" >&2
    exit 1
  fi
  "$python_bin" scripts/validate_cs2_loader_selection.py "${benchmark_paths[@]}" \
    --num-workers auto \
    --prefetch-factor "$prefetch_factor" \
    --persistent-workers "$persistent_workers" \
    --pin-memory "$pin_memory" \
    --expected-git-commit "$(git rev-parse HEAD)" \
    --expected-gpu-substring GH200 \
    --expected-manifest-sha256 "$expected_manifest_sha256" \
    --expected-host-count 1 \
    --minimum-repeats-per-hostname 3 \
    --output "$selection_path"
fi

read -r selected_workers selected_prefetch selected_persistent selected_pin < <(
  "$python_bin" - "$selection_path" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1], encoding="utf-8"))["selected_config"]
print(
    config["num_workers"],
    config["prefetch_factor"],
    str(config["persistent_workers"]).lower(),
    str(config["pin_memory"]).lower(),
)
PY
)
export CS1K_DATALOADER_WORKERS="$selected_workers"
export CS1K_DATALOADER_PREFETCH_FACTOR="$selected_prefetch"
export CS1K_DATALOADER_PERSISTENT_WORKERS="$selected_persistent"
export CS1K_DATALOADER_PIN_MEMORY="$selected_pin"
export CS1K_GLOBAL_LOADER_SELECTION="$selection_path"
if [[ -n "$reuse_loader_selection" ]]; then
  persistent_flag=--no-persistent-workers
  if [[ "$selected_persistent" == true ]]; then
    persistent_flag=--persistent-workers
  fi
  pin_flag=--no-pin-memory
  if [[ "$selected_pin" == true ]]; then
    pin_flag=--pin-memory
  fi
  smoke_path=$benchmark_root/current_node_smoke.json
  srun "${loader_srun[@]}" "$python_bin" scripts/bench_cs2_dataloader.py \
    --data-root "$dataset_dir" \
    --manifest "$manifest_path" \
    --output "$smoke_path" \
    --split val \
    --map-slug dust2 \
    --group-modes synchronized shuffled \
    --workers "$selected_workers" \
    --prefetch-factor "$selected_prefetch" \
    "$persistent_flag" \
    "$pin_flag" \
    --shuffle \
    --clip-len 16 \
    --target-fps 8 \
    --frame-height 168 \
    --frame-width 308 \
    --seed 28 \
    --warmup-batches 1 \
    --timed-batches 2 \
    --transfer-device cuda \
    --skip-parity-check
  "$python_bin" scripts/validate_cs2_reused_loader_selection.py \
    --selection "$selection_path" \
    --smoke "$smoke_path" \
    --expected-current-commit "$(git rev-parse HEAD)" \
    --expected-manifest-sha256 "$expected_manifest_sha256" \
    --expected-hostname "$master_addr" \
    --output "$benchmark_root/reuse_validation.json"
  export CS1K_LOADER_SELECTION_MODE=reused
  export CS1K_LOADER_SMOKE_JSON="$smoke_path"
else
  export CS1K_LOADER_SELECTION_MODE=measured
  unset CS1K_LOADER_SMOKE_JSON
fi
export CS1K_CONFIRMATORY_SPLIT_PROVENANCE="$split_provenance"
export CS1K_CODEC_CHECKPOINT="$codec_checkpoint"
export CS1K_EXPECTED_CODEC_CHECKPOINT_SHA256="$expected_codec_sha256"
export CS1K_OUTPUT_ROOT="$output_root"
export CS1K_TRAIN_STEPS="$train_steps"
export CS1K_ARM_HOURS="$arm_hours"
export CS1K_SEED="$seed"
export CS1K_REQUIRE_SLURM=true
export NNODES=1
export NPROC_PER_NODE=4
export NODE_RANK=0
export MASTER_ADDR="$master_addr"
export MASTER_PORT="$master_port"

srun "${training_srun[@]}" "$project_dir/scripts/run_cs2_gh200_sync_control.sh"

training_root=$output_root/seed_$seed
export CS1K_TRAINING_ROOT="$training_root"
srun \
  --nodes=1 \
  --ntasks=1 \
  --ntasks-per-node=1 \
  --cpus-per-task=72 \
  --exclusive \
  "${single_gpu_args[@]}" \
  "$project_dir/scripts/run_cs2_gh200_sync_control_eval.sh"

test -s "$training_root/audit.json"
printf 'completed and audited %s\n' "$training_root"
