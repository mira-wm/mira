#!/usr/bin/env bash
# One-node, four-GH200 synchronized-vs-cross-round-grouped matched-information ablation.
#
# The publication path launches this script once with all four local GH200s visible; torchrun
# starts one DDP process per GPU. Each arm uses the
# identical ten-player architecture, global batch, action/video volume, seed, GPU topology, and
# optimizer-update budget; only whether the ten POVs come from the same synchronized round changes.
# The wall-clock setting is a fail-closed safety cap, not the compute-matching variable.
set -euo pipefail

project_dir=${MIRA_PROJECT_DIR:-$PWD}
dataset_dir=${CS1K_DATASET_DIR:?Set CS1K_DATASET_DIR on every node}
manifest_path=${CS1K_MANIFEST_PATH:?Set the frozen confirmatory manifest path on every node}
split_provenance=${CS1K_CONFIRMATORY_SPLIT_PROVENANCE:?Set the frozen split provenance on every node}
codec_checkpoint=${CS1K_CODEC_CHECKPOINT:?Set CS1K_CODEC_CHECKPOINT on every node}
expected_codec_sha256=${CS1K_EXPECTED_CODEC_CHECKPOINT_SHA256:-3c286c59b74cd141e72af69cde1a0a005142d8d2b472c789cdf2a39a140c4b7a}
output_root=${CS1K_OUTPUT_ROOT:?Set CS1K_OUTPUT_ROOT to a shared result directory}
train_steps=${CS1K_TRAIN_STEPS:?Set the identical optimizer-update count for both arms}
arm_hours=${CS1K_ARM_HOURS:?Set the per-arm fail-closed wall-clock cap}
seed=${CS1K_SEED:-28}
if ! [[ "$train_steps" =~ ^[1-9][0-9]*$ ]]; then
  echo "CS1K_TRAIN_STEPS must be a positive integer" >&2
  exit 1
fi
if ! [[ "$seed" =~ ^[0-9]+$ ]]; then
  echo "CS1K_SEED must be a non-negative integer" >&2
  exit 1
fi
if (( seed % 2 == 0 )); then
  default_arm_order=synchronized,shuffled
else
  default_arm_order=shuffled,synchronized
fi
arm_order=${CS1K_ARM_ORDER:-$default_arm_order}

nnodes=${NNODES:-1}
node_rank=${NODE_RANK:-${SLURM_PROCID:-}}
nproc_per_node=${NPROC_PER_NODE:-4}
master_addr=${MASTER_ADDR:?Set MASTER_ADDR to the rank-0 hostname or IP}
master_port=${MASTER_PORT:-29500}
dataloader_workers=${CS1K_DATALOADER_WORKERS:?Freeze workers from the GH200 loader benchmark}
dataloader_prefetch_factor=${CS1K_DATALOADER_PREFETCH_FACTOR:?Freeze prefetch from the GH200 benchmark}
dataloader_persistent_workers=${CS1K_DATALOADER_PERSISTENT_WORKERS:?Freeze persistence from the benchmark}
dataloader_pin_memory=${CS1K_DATALOADER_PIN_MEMORY:?Freeze pin-memory from the GH200 benchmark}
loader_benchmark_root=${CS1K_LOADER_BENCHMARK_ROOT:-}
loader_benchmark_jsons=${CS1K_LOADER_BENCHMARK_JSONS:-}
global_loader_selection=${CS1K_GLOBAL_LOADER_SELECTION:?Set the frozen loader selection}
loader_selection_mode=${CS1K_LOADER_SELECTION_MODE:-measured}
loader_smoke_json=${CS1K_LOADER_SMOKE_JSON:-}
require_slurm=${CS1K_REQUIRE_SLURM:-true}
node_hostname=$(hostname)

if [[ "$nnodes" != 1 || "$nproc_per_node" != 4 ]]; then
  echo "The preregistered topology is one Clariden node with four local GH200 DDP processes" >&2
  exit 1
fi
if [[ "$node_rank" != 0 ]]; then
  echo "NODE_RANK must be 0 for the single-node launcher" >&2
  exit 1
fi
if [[ "$require_slurm" != true && "$require_slurm" != false ]]; then
  echo "CS1K_REQUIRE_SLURM must be true or false" >&2
  exit 1
fi
if [[ "$require_slurm" == true ]]; then
  : "${SLURM_JOB_ID:?Publication GH200 runs require a Slurm allocation}"
  : "${SLURM_JOB_NODELIST:?SLURM_JOB_NODELIST is required}"
  : "${SLURM_PROCID:?SLURM_PROCID is required}"
  : "${SLURM_NODEID:?SLURM_NODEID is required}"
  slurm_nodes=${SLURM_NNODES:-${SLURM_JOB_NUM_NODES:-}}
  if [[ "$slurm_nodes" != 1 ]]; then
    echo "Slurm allocation must contain exactly one node, found ${slurm_nodes:-unset}" >&2
    exit 1
  fi
  if [[ "$SLURM_PROCID" != 0 || "$SLURM_NODEID" != 0 || "${SLURM_LOCALID:-0}" != 0 ]]; then
    echo "One four-GPU launcher task is required: procid=$SLURM_PROCID nodeid=$SLURM_NODEID localid=${SLURM_LOCALID:-unset}" >&2
    exit 1
  fi
fi

python_bin=${MIRA_PYTHON:-$project_dir/.pixi/envs/default/bin/python}
if [[ -n ${TORCHRUN_BIN:-} ]]; then
  torchrun_command=("$TORCHRUN_BIN")
else
  # Clariden's uenv exposes PyTorch through system site-packages, so the layered venv can import
  # torch without owning a bin/torchrun console script.
  torchrun_command=("$python_bin" -m torch.distributed.run)
fi

cd "$project_dir"
export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export WANDB_MODE=disabled

code_commit=$(git rev-parse HEAD)
if [[ -n "$(git status --porcelain=v1)" ]]; then
  echo "GH200 publication runs require a clean source tree" >&2
  exit 1
fi
if [[ ! -f "$codec_checkpoint" ]]; then
  echo "Codec checkpoint not found: $codec_checkpoint" >&2
  exit 1
fi
if [[ "$(sha256sum "$codec_checkpoint" | cut -d' ' -f1)" != "$expected_codec_sha256" ]]; then
  echo "Codec checkpoint SHA-256 drifted on $node_hostname" >&2
  exit 1
fi
if [[ ! -f "$manifest_path" ]]; then
  echo "Confirmatory manifest not found: $manifest_path" >&2
  exit 1
fi
if [[ ! -f "$split_provenance" ]]; then
  echo "Confirmatory split provenance not found: $split_provenance" >&2
  exit 1
fi
if [[ ! -f "$global_loader_selection" ]]; then
  echo "Frozen loader selection not found: $global_loader_selection" >&2
  exit 1
fi
mapfile -t visible_gpu_names < <(
  nvidia-smi --query-gpu=name --format=csv,noheader | sed 's/[[:space:]]*$//'
)
if [[ ${#visible_gpu_names[@]} -ne 4 ]]; then
  echo "Exactly four scheduler-visible GPUs are required, found ${#visible_gpu_names[@]}" >&2
  exit 1
fi
for gpu_name in "${visible_gpu_names[@]}"; do
  if [[ "${gpu_name,,}" != *gh200* ]]; then
    echo "Expected four GH200s, found ${visible_gpu_names[*]}" >&2
    exit 1
  fi
done
visible_gpu_names_joined=$(IFS='|'; printf '%s' "${visible_gpu_names[*]}")
if [[ "$(dirname "$(realpath "$manifest_path")")" != "$(realpath "$dataset_dir")" ]]; then
  echo "CS1K_MANIFEST_PATH must live directly inside CS1K_DATASET_DIR" >&2
  exit 1
fi
IFS=, read -r -a arms <<<"$arm_order"
if [[ ${#arms[@]} -ne 2 ]]; then
  echo "CS1K_ARM_ORDER must contain exactly two comma-separated arms" >&2
  exit 1
fi
for arm in "${arms[@]}"; do
  if [[ "$arm" != synchronized && "$arm" != shuffled ]]; then
    echo "Unsupported arm $arm; expected synchronized or shuffled" >&2
    exit 1
  fi
done
if [[ "${arms[0]}" == "${arms[1]}" ]]; then
  echo "CS1K_ARM_ORDER must contain synchronized and shuffled once each" >&2
  exit 1
fi

experiment_root=$output_root/seed_$seed
node_provenance=$experiment_root/provenance/node_$node_rank
mkdir -p "$node_provenance"
cp "$global_loader_selection" "$node_provenance/global_loader_selection.json"
"$python_bin" scripts/prepare_cs2_confirmatory_split.py \
  --source-manifest "$dataset_dir/manifest.parquet" \
  --output-manifest "$manifest_path" \
  --provenance-output "$split_provenance" \
  --verify-only \
  >"$node_provenance/confirmatory_split_verification.json"
if [[ "$loader_selection_mode" == measured ]]; then
  if [[ -n "$loader_benchmark_root" && -n "$loader_benchmark_jsons" ]]; then
    echo "Set only one of CS1K_LOADER_BENCHMARK_ROOT or CS1K_LOADER_BENCHMARK_JSONS" >&2
    exit 1
  fi
  if [[ -n "$loader_benchmark_root" ]]; then
    benchmark_node_root=$loader_benchmark_root/$node_hostname
    if [[ ! -d "$benchmark_node_root" ]]; then
      echo "Node-local loader benchmark directory not found: $benchmark_node_root" >&2
      exit 1
    fi
    mapfile -d '' -t loader_benchmark_paths < <(
      find "$benchmark_node_root" -maxdepth 1 -type f -name '*.json' -print0 | sort -z
    )
  elif [[ -n "$loader_benchmark_jsons" ]]; then
    loader_benchmark_jsons=${loader_benchmark_jsons//\{hostname\}/$node_hostname}
    read -r -a loader_benchmark_paths <<<"$loader_benchmark_jsons"
  else
    echo "Provide node-local GH200 evidence through CS1K_LOADER_BENCHMARK_ROOT or JSONS" >&2
    exit 1
  fi
  if [[ ${#loader_benchmark_paths[@]} -lt 3 ]]; then
    echo "At least three node-local loader benchmark JSONs are required" >&2
    exit 1
  fi
  for benchmark_path in "${loader_benchmark_paths[@]}"; do
    if [[ ! -f "$benchmark_path" ]]; then
      echo "Loader benchmark JSON not found: $benchmark_path" >&2
      exit 1
    fi
  done
  "$python_bin" scripts/validate_cs2_loader_selection.py "${loader_benchmark_paths[@]}" \
    --num-workers "$dataloader_workers" \
    --prefetch-factor "$dataloader_prefetch_factor" \
    --persistent-workers "$dataloader_persistent_workers" \
    --pin-memory "$dataloader_pin_memory" \
    --expected-git-commit "$code_commit" \
    --expected-gpu-substring GH200 \
    --expected-hostname "$node_hostname" \
    --expected-host-count 1 \
    --minimum-repeats-per-hostname 3 \
    --expected-manifest-sha256 "$(sha256sum "$manifest_path" | cut -d' ' -f1)" \
    --output "$node_provenance/frozen_loader_selection.json"
elif [[ "$loader_selection_mode" == reused ]]; then
  if [[ ! -s "$loader_smoke_json" ]]; then
    echo "Reused loader selection requires CS1K_LOADER_SMOKE_JSON" >&2
    exit 1
  fi
  cp "$loader_smoke_json" "$node_provenance/loader_smoke.json"
  "$python_bin" scripts/validate_cs2_reused_loader_selection.py \
    --selection "$node_provenance/global_loader_selection.json" \
    --smoke "$node_provenance/loader_smoke.json" \
    --expected-current-commit "$code_commit" \
    --expected-manifest-sha256 "$(sha256sum "$manifest_path" | cut -d' ' -f1)" \
    --expected-hostname "$node_hostname" \
    --output "$node_provenance/frozen_loader_selection.json"
else
  echo "CS1K_LOADER_SELECTION_MODE must be measured or reused" >&2
  exit 1
fi
"$python_bin" scripts/prepare_counterstrike1k.py \
  --data-root "$dataset_dir" \
  --manifest "$manifest_path" \
  --map-slug dust2 \
  --provenance-output "$node_provenance/dataset.json"
git rev-parse HEAD >"$node_provenance/code_commit.txt"
git status --porcelain=v1 >"$node_provenance/code_status.txt"
git diff --binary >"$node_provenance/code.patch"
sha256sum "$codec_checkpoint" >"$node_provenance/codec_checkpoint.sha256"
sha256sum "$manifest_path" "$split_provenance" \
  >"$node_provenance/confirmatory_split_files.sha256"
sha256sum pixi.lock pyproject.toml >"$node_provenance/environment_files.sha256"
"$python_bin" - <<'PY' >"$node_provenance/installed_packages.txt"
from importlib.metadata import distributions

packages = sorted(
    (distribution.metadata["Name"], distribution.version)
    for distribution in distributions()
    if distribution.metadata["Name"]
)
for name, version in packages:
    print(f"{name}=={version}")
PY
nvidia-smi -q >"$node_provenance/nvidia_smi_q.txt"
uname -a >"$node_provenance/uname.txt"
loader_smoke_sha256=
if [[ -f "$node_provenance/loader_smoke.json" ]]; then
  loader_smoke_sha256=$(sha256sum "$node_provenance/loader_smoke.json" | cut -d' ' -f1)
fi
printf '%s\n' \
  "seed=$seed" \
  "train_steps=$train_steps" \
  "arm_hours=$arm_hours" \
  "arm_order=$arm_order" \
  "nnodes=$nnodes" \
  "nproc_per_node=$nproc_per_node" \
  "node_rank=$node_rank" \
  "hostname=$node_hostname" \
  "visible_gpu_count=${#visible_gpu_names[@]}" \
  "visible_gpu_name=${visible_gpu_names[0]}" \
  "visible_gpu_names=$visible_gpu_names_joined" \
  "master_addr=$master_addr" \
  "master_port=$master_port" \
  "scheduler=$([[ -n ${SLURM_JOB_ID:-} ]] && printf slurm || printf manual)" \
  "slurm_job_id=${SLURM_JOB_ID:-}" \
  "slurm_job_nodelist=${SLURM_JOB_NODELIST:-}" \
  "slurm_procid=${SLURM_PROCID:-}" \
  "slurm_nodeid=${SLURM_NODEID:-}" \
  "slurm_localid=${SLURM_LOCALID:-}" \
  "slurm_cpus_per_task=${SLURM_CPUS_PER_TASK:-}" \
  "logical_cpus_visible=$(nproc)" \
  "manifest_path=$manifest_path" \
  "manifest_sha256=$(sha256sum "$manifest_path" | cut -d' ' -f1)" \
  "split_provenance=$split_provenance" \
  "split_provenance_sha256=$(sha256sum "$split_provenance" | cut -d' ' -f1)" \
  "action_routing=spatial" \
  "dataloader_workers=$dataloader_workers" \
  "dataloader_prefetch_factor=$dataloader_prefetch_factor" \
  "dataloader_persistent_workers=$dataloader_persistent_workers" \
  "dataloader_pin_memory=$dataloader_pin_memory" \
  "loader_selection_mode=$loader_selection_mode" \
  "loader_smoke_sha256=$loader_smoke_sha256" \
  "global_loader_selection_sha256=$(sha256sum "$node_provenance/global_loader_selection.json" | cut -d' ' -f1)" \
  "frozen_loader_selection_sha256=$(sha256sum "$node_provenance/frozen_loader_selection.json" | cut -d' ' -f1)" \
  >"$node_provenance/launcher.env"

telemetry_pid=
stop_telemetry() {
  if [[ -n "$telemetry_pid" ]]; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
}
trap stop_telemetry EXIT
nvidia-smi \
  --query-gpu=timestamp,name,uuid,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
  --format=csv \
  --loop=5 \
  >"$node_provenance/gpu_timeseries.csv" \
  2>"$node_provenance/gpu_timeseries.log" &
telemetry_pid=$!

torchrun_args=(
  --nnodes "$nnodes"
  --nproc-per-node "$nproc_per_node"
  --node-rank "$node_rank"
  --master-addr "$master_addr"
  --master-port "$master_port"
)

write_status() {
  if [[ "$node_rank" == 0 ]]; then
    printf '%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" \
      >>"$experiment_root/status.tsv"
  fi
}

for arm in "${arms[@]}"; do
  write_status "$arm" running
  arm_started_epoch=$(date +%s)
  arm_started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  "${torchrun_command[@]}" "${torchrun_args[@]}" scripts/train_world_model.py \
    model=multi_wrapper_world_model_cs2_small \
    dataset=counterstrike1k_dust2 \
    dataset.train_index="$manifest_path" \
    dataset.test_index="$manifest_path" \
    dataset.n_players=10 \
    dataset.group_mode="$arm" \
    dataset.validation_group_mode=synchronized \
    model.architecture.config.action_routing=spatial \
    model.architecture.config.wm_config.codec_checkpoint="$codec_checkpoint" \
    run.seed="$seed" \
    run.steps="$train_steps" \
    run.batch_size=1 \
    run.deterministic=true \
    run.compile=false \
    run.max_duration_hours="$arm_hours" \
    run.log_every=25 \
    run.checkpoint_every=1000 \
    run.checkpoint_keep_recent=10 \
    run.output_dir="$experiment_root/$arm" \
    dataloader.num_workers="$dataloader_workers" \
    dataloader.shuffle_buffer_size=100 \
    dataloader.prefetch_factor="$dataloader_prefetch_factor" \
    dataloader.persistent_workers="$dataloader_persistent_workers" \
    dataloader.pin_memory="$dataloader_pin_memory" \
    validation.val_first=true \
    validation.val_every=1000 \
    validation.val_n_samples=40 \
    validation.local_rollout_every=1000 \
    validation.local_rollout_seed=37 \
    validation.downstream_val_every=100000000 \
    optim.scheduler.warmup_steps=500 \
    optim.model_ema_decay=0.999 \
    world_model_metrics.n_context_frames=8 \
    world_model_metrics.num_unrolled_frames=4 \
    world_model_metrics.drift_metric_frames=4 \
    world_model_metrics.fdd_slice_frames=2 \
    world_model_metrics.dino_model=dinov2_vitb14 \
    world_model_metrics.num_samples=40 \
    world_model_metrics.per_device_batch_size=1 \
    world_model_metrics.num_viz_samples=2 \
    wandb.mode=disabled \
    hydra.run.dir="$experiment_root/hydra/$arm/node_$node_rank"
  final_step=$((train_steps - 1))
  final_checkpoint=$experiment_root/$arm/checkpoint-$final_step/checkpoint.pth
  if [[ ! -s "$final_checkpoint" ]]; then
    echo "Arm $arm did not reach the required $train_steps optimizer updates: $final_checkpoint missing" >&2
    exit 1
  fi
  if grep -q '"kind": "time_limit"' "$experiment_root/$arm/metrics.jsonl"; then
    echo "Arm $arm hit the safety wall-clock cap before the fixed-step endpoint" >&2
    exit 1
  fi
  if [[ "$node_rank" == 0 ]]; then
    arm_ended_epoch=$(date +%s)
    arm_ended_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    "$python_bin" - \
      "$experiment_root/$arm/training_termination.json.tmp" \
      "$arm" \
      "$train_steps" \
      "$final_step" \
      "$arm_started_utc" \
      "$arm_ended_utc" \
      "$((arm_ended_epoch - arm_started_epoch))" \
      "$arm_hours" \
      "$final_checkpoint" <<'PY'
import json
import sys
from pathlib import Path

(
    output,
    arm,
    train_steps,
    final_step,
    started_utc,
    ended_utc,
    elapsed_seconds,
    max_duration_hours,
    checkpoint,
) = sys.argv[1:]
payload = {
    "schema": "mira-cs2-fixed-step-termination-v1",
    "arm": arm,
    "termination": "fixed_step_complete",
    "optimizer_updates": int(train_steps),
    "final_step": int(final_step),
    "launcher_elapsed_wall_seconds": int(elapsed_seconds),
    "max_duration_hours": float(max_duration_hours),
    "started_utc": started_utc,
    "ended_utc": ended_utc,
    "checkpoint": str(Path(checkpoint).resolve()),
}
Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
    mv \
      "$experiment_root/$arm/training_termination.json.tmp" \
      "$experiment_root/$arm/training_termination.json"
  fi
  write_status "$arm" complete
done

write_status ablation complete
