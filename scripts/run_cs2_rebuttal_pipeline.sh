#!/usr/bin/env bash
# G7e preliminary CounterStrike-1K rebuttal pipeline.
#
# One shared codec is trained first and excluded from the single/shared comparison. The world-model
# runs then receive equal 5.5-hour wall-clock budgets on the same GPU and process ten POV rows per
# optimizer step (single batch=10; shared batch=1 group x 10 POVs). Local metrics.jsonl files record
# loss, held-out loss, elapsed time, processed frames and throughput independently of W&B.
set -euo pipefail

project_dir=${MIRA_PROJECT_DIR:-/home/ubuntu/projects/mira}
dataset_dir=${CS1K_DATASET_DIR:-/home/ubuntu/projects/cs2_clean/data/cs1k-360p}
run_root=${CS1K_RUN_ROOT:-/home/ubuntu/projects/mira_runs/cs2_rebuttal/20260724_g7e2}
python_bin=${MIRA_PYTHON:-$project_dir/.pixi/envs/default/bin/python}
dataloader_workers=${CS1K_DATALOADER_WORKERS:-4}
dataloader_prefetch_factor=${CS1K_DATALOADER_PREFETCH_FACTOR:-2}
dataloader_persistent_workers=${CS1K_DATALOADER_PERSISTENT_WORKERS:-true}

mkdir -p "$run_root"
cd "$project_dir"
export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE=disabled
export CUBLAS_WORKSPACE_CONFIG=:4096:8

mkdir -p "$run_root/provenance"
"$python_bin" scripts/prepare_counterstrike1k.py \
  --data-root "$dataset_dir" \
  --map-slug dust2 \
  --provenance-output "$run_root/provenance/dataset.json"
git rev-parse HEAD >"$run_root/provenance/code_commit.txt"
git status --porcelain=v1 >"$run_root/provenance/code_status.txt"
git diff --binary >"$run_root/provenance/code.patch"
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

write_status() {
  local stage=$1
  local state=$2
  printf '%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$stage" "$state" >>"$run_root/pipeline_status.tsv"
}

write_status codec running
"$python_bin" scripts/train_codec.py \
  model=raev2_codec_cs2_fast \
  dataset=counterstrike1k_dust2 \
  dataset.train_index="$dataset_dir" \
  run.steps=1000000 \
  run.batch_size=4 \
  run.deterministic=true \
  run.compile=false \
  run.max_duration_hours=1.0 \
  run.log_every=25 \
  run.checkpoint_every=5000 \
  run.checkpoint_keep_recent=3 \
  run.output_dir="$run_root/codec" \
  dataloader.num_workers="$dataloader_workers" \
  dataloader.shuffle_buffer_size=100 \
  dataloader.prefetch_factor="$dataloader_prefetch_factor" \
  dataloader.persistent_workers="$dataloader_persistent_workers" \
  validation.val_first=true \
  validation.val_every=2000 \
  validation.val_n_samples=64 \
  optim.scheduler.warmup_steps=500 \
  optim.scheduler.decay_steps=0 \
  wandb.mode=disabled
write_status codec complete

codec_checkpoint=$(find "$run_root/codec" -path '*/checkpoint-*/checkpoint.pth' -print0 |
  sort -zV | tail -z -n 1 | tr -d '\0')
if [[ -z "$codec_checkpoint" ]]; then
  write_status codec missing_checkpoint
  exit 1
fi
printf '%s\n' "$codec_checkpoint" >"$run_root/codec_checkpoint.txt"

write_status single running
"$python_bin" scripts/train_world_model.py \
  model=latent_world_model_cs2_small \
  dataset=counterstrike1k_dust2 \
  dataset.train_index="$dataset_dir" \
  model.architecture.config.codec_checkpoint="$codec_checkpoint" \
  run.steps=1000000 \
  run.batch_size=10 \
  run.deterministic=true \
  run.compile=false \
  run.max_duration_hours=5.5 \
  run.log_every=25 \
  run.checkpoint_every=1000 \
  run.checkpoint_keep_recent=10 \
  run.output_dir="$run_root/single" \
  dataloader.num_workers="$dataloader_workers" \
  dataloader.shuffle_buffer_size=100 \
  dataloader.prefetch_factor="$dataloader_prefetch_factor" \
  dataloader.persistent_workers="$dataloader_persistent_workers" \
  validation.val_first=true \
  validation.val_every=1000 \
  validation.val_n_samples=100 \
  validation.downstream_val_every=1000000 \
  optim.scheduler.warmup_steps=500 \
  optim.model_ema_decay=0.999 \
  world_model_metrics.n_context_frames=8 \
  world_model_metrics.num_unrolled_frames=4 \
  world_model_metrics.drift_metric_frames=4 \
  world_model_metrics.fdd_slice_frames=2 \
  world_model_metrics.dino_model=dinov2_vitb14 \
  world_model_metrics.num_samples=100 \
  world_model_metrics.per_device_batch_size=10 \
  world_model_metrics.num_viz_samples=2 \
  wandb.mode=disabled
write_status single complete

write_status shared running
"$python_bin" scripts/train_world_model.py \
  model=multi_wrapper_world_model_cs2_small \
  dataset=counterstrike1k_dust2 \
  dataset.train_index="$dataset_dir" \
  dataset.n_players=10 \
  dataset.group_mode=synchronized \
  model.architecture.config.wm_config.codec_checkpoint="$codec_checkpoint" \
  run.steps=1000000 \
  run.batch_size=1 \
  run.deterministic=true \
  run.compile=false \
  run.max_duration_hours=5.5 \
  run.log_every=25 \
  run.checkpoint_every=1000 \
  run.checkpoint_keep_recent=10 \
  run.output_dir="$run_root/shared" \
  dataloader.num_workers="$dataloader_workers" \
  dataloader.shuffle_buffer_size=100 \
  dataloader.prefetch_factor="$dataloader_prefetch_factor" \
  dataloader.persistent_workers="$dataloader_persistent_workers" \
  validation.val_first=true \
  validation.val_every=1000 \
  validation.val_n_samples=10 \
  validation.downstream_val_every=1000000 \
  optim.scheduler.warmup_steps=500 \
  optim.model_ema_decay=0.999 \
  world_model_metrics.n_context_frames=8 \
  world_model_metrics.num_unrolled_frames=4 \
  world_model_metrics.drift_metric_frames=4 \
  world_model_metrics.fdd_slice_frames=2 \
  world_model_metrics.dino_model=dinov2_vitb14 \
  world_model_metrics.num_samples=10 \
  world_model_metrics.per_device_batch_size=1 \
  world_model_metrics.num_viz_samples=2 \
  wandb.mode=disabled
write_status shared complete
write_status pipeline complete
