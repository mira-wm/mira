#!/usr/bin/env bash
# One-hour G7e engineering preflight for the post-pilot spatial player-action router.
#
# This intentionally touches only the release validation split. It is a frozen go/no-go check that
# the corrected architecture preserves and learns player-aligned conditioning before GH200 use; it
# is not a synchronization-vs-shuffled comparison and not a held-out model-quality result.
set -euo pipefail

project_dir=${MIRA_PROJECT_DIR:-$PWD}
dataset_dir=${CS1K_DATASET_DIR:?Set the CounterStrike-1K materialization directory}
manifest_path=${CS1K_MANIFEST_PATH:?Set the frozen confirmatory manifest path}
split_provenance=${CS1K_CONFIRMATORY_SPLIT_PROVENANCE:?Set the frozen split provenance}
codec_checkpoint=${CS1K_CODEC_CHECKPOINT:?Set the already frozen pilot codec checkpoint}
run_root=${CS1K_RUN_ROOT:?Set a new output directory for the spatial-routing preflight}
python_bin=${MIRA_PYTHON:-$project_dir/.pixi/envs/default/bin/python}
dataloader_workers=${CS1K_DATALOADER_WORKERS:-4}
dataloader_prefetch_factor=${CS1K_DATALOADER_PREFETCH_FACTOR:-2}
dataloader_persistent_workers=${CS1K_DATALOADER_PERSISTENT_WORKERS:-true}
dataloader_pin_memory=${CS1K_DATALOADER_PIN_MEMORY:-true}
train_seed=28
preflight_hours=1.0
eval_seeds="37 38 39"
val_rounds=54

cd "$project_dir"
export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE=disabled
export CUBLAS_WORKSPACE_CONFIG=:4096:8

if [[ -n "$(git status --porcelain=v1)" ]]; then
  echo "The spatial-routing preflight requires a clean source tree" >&2
  exit 1
fi
if [[ ! -f "$codec_checkpoint" || ! -f "$manifest_path" || ! -f "$split_provenance" ]]; then
  echo "Codec, frozen confirmatory manifest, and split provenance are required" >&2
  exit 1
fi
if [[ "$(dirname "$(realpath "$manifest_path")")" != "$(realpath "$dataset_dir")" ]]; then
  echo "CS1K_MANIFEST_PATH must live directly inside CS1K_DATASET_DIR" >&2
  exit 1
fi
if ! nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | grep -q 'RTX PRO 6000'; then
  echo "The preregistered preflight hardware is one RTX PRO 6000 (G7e)" >&2
  exit 1
fi
if [[ -e "$run_root" ]]; then
  echo "Refusing to reuse existing preflight output: $run_root" >&2
  exit 1
fi

mkdir -p "$run_root/provenance"
status_file=$run_root/status.tsv
write_status() {
  printf '%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" >>"$status_file"
}

"$python_bin" scripts/prepare_cs2_confirmatory_split.py \
  --source-manifest "$dataset_dir/manifest.parquet" \
  --output-manifest "$manifest_path" \
  --provenance-output "$split_provenance" \
  --verify-only \
  >"$run_root/provenance/confirmatory_split_verification.json"
"$python_bin" scripts/prepare_counterstrike1k.py \
  --data-root "$dataset_dir" \
  --manifest "$manifest_path" \
  --map-slug dust2 \
  --provenance-output "$run_root/provenance/dataset.json"
git rev-parse HEAD >"$run_root/provenance/code_commit.txt"
git status --porcelain=v1 >"$run_root/provenance/code_status.txt"
git diff --binary >"$run_root/provenance/code.patch"
sha256sum "$codec_checkpoint" >"$run_root/provenance/codec_checkpoint.sha256"
sha256sum "$manifest_path" "$split_provenance" \
  >"$run_root/provenance/confirmatory_split_files.sha256"
sha256sum pixi.lock pyproject.toml >"$run_root/provenance/environment_files.sha256"
"$python_bin" - <<'PY' >"$run_root/provenance/installed_packages.txt"
from importlib.metadata import distributions

packages = sorted(
    (distribution.metadata["Name"], distribution.version)
    for distribution in distributions()
    if distribution.metadata["Name"]
)
for name, version in packages:
    print(f"{name}=={version}")
PY
nvidia-smi -q >"$run_root/provenance/nvidia_smi_q.txt"
uname -a >"$run_root/provenance/uname.txt"
printf '%s\n' \
  "purpose=validation-only spatial-routing engineering gate" \
  "train_seed=$train_seed" \
  "preflight_hours=$preflight_hours" \
  "eval_seeds=$eval_seeds" \
  "val_rounds=$val_rounds" \
  "action_routing=spatial" \
  "dataloader_workers=$dataloader_workers" \
  "dataloader_prefetch_factor=$dataloader_prefetch_factor" \
  "dataloader_persistent_workers=$dataloader_persistent_workers" \
  "dataloader_pin_memory=$dataloader_pin_memory" \
  >"$run_root/provenance/contract.env"

telemetry_file=$run_root/provenance/gpu_timeseries.csv
nvidia-smi \
  --query-gpu=timestamp,name,memory.used,utilization.gpu,power.draw \
  --format=csv \
  >"$telemetry_file"
(
  while sleep 5; do
    nvidia-smi \
      --query-gpu=timestamp,name,memory.used,utilization.gpu,power.draw \
      --format=csv,noheader \
      >>"$telemetry_file"
  done
) &
telemetry_pid=$!
printf '%s\n' "$telemetry_pid" >"$run_root/provenance/gpu_telemetry.pid"
stop_telemetry() {
  if kill -0 "$telemetry_pid" 2>/dev/null; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
}
trap stop_telemetry EXIT

write_status training running
"$python_bin" scripts/train_world_model.py \
  model=multi_wrapper_world_model_cs2_small \
  dataset=counterstrike1k_dust2 \
  dataset.train_index="$manifest_path" \
  dataset.test_index="$manifest_path" \
  dataset.n_players=10 \
  dataset.group_mode=synchronized \
  dataset.validation_group_mode=synchronized \
  model.architecture.config.action_routing=spatial \
  model.architecture.config.wm_config.codec_checkpoint="$codec_checkpoint" \
  run.seed="$train_seed" \
  run.steps=100000000 \
  run.batch_size=1 \
  run.deterministic=true \
  run.compile=false \
  run.max_duration_hours="$preflight_hours" \
  run.log_every=25 \
  run.checkpoint_every=1000 \
  run.checkpoint_keep_recent=10 \
  run.output_dir="$run_root/model" \
  dataloader.num_workers="$dataloader_workers" \
  dataloader.shuffle_buffer_size=100 \
  dataloader.prefetch_factor="$dataloader_prefetch_factor" \
  dataloader.persistent_workers="$dataloader_persistent_workers" \
  dataloader.pin_memory="$dataloader_pin_memory" \
  validation.val_first=true \
  validation.val_every=1000 \
  validation.val_n_samples="$val_rounds" \
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
  world_model_metrics.num_samples="$val_rounds" \
  world_model_metrics.per_device_batch_size=1 \
  world_model_metrics.num_viz_samples=2 \
  wandb.mode=disabled
write_status training complete

checkpoint=$(
  find "$run_root/model" -path '*/checkpoint-*/checkpoint.pth' -print0 |
    sort -zV | tail -z -n 1 | tr -d '\0'
)
if [[ -z "$checkpoint" ]]; then
  write_status training missing_checkpoint
  exit 1
fi
printf '%s\n' "$checkpoint" >"$run_root/provenance/checkpoint.txt"
sha256sum "$checkpoint" >"$run_root/provenance/checkpoint.sha256"

mkdir -p "$run_root/evaluation"
write_status routing_diagnostic running
"$python_bin" scripts/diagnose_cs2_multi_action_routing.py "$checkpoint" \
  --split val \
  --seed 37 \
  --num-batches "$val_rounds" \
  --num-workers "$dataloader_workers" \
  --output "$run_root/evaluation/routing.json" \
  2>&1 | tee "$run_root/evaluation/routing.log"
write_status routing_diagnostic complete

write_status action_loss running
for seed in $eval_seeds; do
  seed_dir=$run_root/evaluation/action_loss/seed_$seed
  mkdir -p "$seed_dir"
  for mode in true batch-shifted; do
    "$python_bin" scripts/eval_world_model_offline.py "$checkpoint" \
      --split val \
      --group-mode synchronized \
      --action-mode "$mode" \
      --seed "$seed" \
      --deterministic \
      --dino-model dinov2_vitb14 \
      --per-device-batch-size 1 \
      --val-n-samples "$val_rounds" \
      --skip-metrics \
      --no-compile \
      --results-json "$seed_dir/$mode.json" \
      2>&1 | tee "$seed_dir/$mode.log"
  done
done
write_status action_loss complete

write_status gate running
if "$python_bin" scripts/assess_cs2_spatial_routing_preflight.py \
  "$run_root/evaluation/action_loss" \
  --routing-diagnostic "$run_root/evaluation/routing.json" \
  --output "$run_root/evaluation/preflight_gate.json" \
  2>&1 | tee "$run_root/evaluation/preflight_gate.log"; then
  write_status gate passed
else
  write_status gate failed
  exit 2
fi
write_status preflight complete
