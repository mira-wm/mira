#!/usr/bin/env bash
# Fresh, leak-free CounterStrike-1K Dust2 single-MIRA confirmatory baseline.
#
# The launcher trains a new codec and world model on the frozen 36-match training split, selects
# exact step endpoints without looking at test output, then runs paired action interventions on all
# 69 untouched test rounds. It refuses dirty source, existing output, manifest drift, and checkpoint
# drift. Set MIRA_REVIEW_S3_URI to mirror fixed validation traces during training.
set -euo pipefail

project_dir=${MIRA_PROJECT_DIR:-/home/ubuntu/projects/mira_cs1k_confirmatory_single_v1}
dataset_dir=${CS1K_DATASET_DIR:-/home/ubuntu/projects/cs2_clean/data/cs1k-360p}
manifest=${CS1K_CONFIRMATORY_MANIFEST:-$dataset_dir/manifest_dust2_confirmatory_spatial_v1.parquet}
manifest_provenance=${CS1K_CONFIRMATORY_PROVENANCE:-$dataset_dir/manifest_dust2_confirmatory_spatial_v1.provenance.json}
source_manifest=${CS1K_SOURCE_MANIFEST:-$dataset_dir/manifest.parquet}
run_root=${CS1K_RUN_ROOT:?Set CS1K_RUN_ROOT to a new, nonexistent output directory}
python_bin=${MIRA_PYTHON:-/home/ubuntu/projects/mira/.pixi/envs/default/bin/python}
expected_commit=${MIRA_EXPECTED_COMMIT:?Set MIRA_EXPECTED_COMMIT to the reviewed public commit}
review_s3_uri=${MIRA_REVIEW_S3_URI:-}
dataloader_workers=${CS1K_DATALOADER_WORKERS:-4}
dataloader_prefetch_factor=${CS1K_DATALOADER_PREFETCH_FACTOR:-2}
dataloader_persistent_workers=${CS1K_DATALOADER_PERSISTENT_WORKERS:-true}

expected_manifest_sha256=33abbb623072932431871a612620110c473d4b664c52010e5763c273c6daf10e
expected_manifest_provenance_sha256=3f6419f9414576c88773874c8009c0814e827186c8782831cc373a27be70c2ef
expected_source_manifest_sha256=e6d1199595327ccbed7a79760266f1c30fdcf23b717232705c9b11bb6d8707d3
codec_endpoint=18000
single_endpoint=15000
eval_seeds="37 41 43"
action_modes="true round-shifted time-shifted zero"
window_modes="midpoint first-death"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ ! -e "$run_root" ]] || fail "run root already exists: $run_root"
[[ -x "$python_bin" ]] || fail "Python environment is unavailable: $python_bin"
[[ -f "$manifest" ]] || fail "confirmatory manifest is unavailable: $manifest"
[[ -f "$manifest_provenance" ]] || fail "manifest provenance is unavailable: $manifest_provenance"
[[ -f "$source_manifest" ]] || fail "source manifest is unavailable: $source_manifest"

cd "$project_dir"
observed_commit=$(git rev-parse HEAD)
[[ "$observed_commit" == "$expected_commit" ]] ||
  fail "source commit is $observed_commit, expected $expected_commit"
[[ -z "$(git status --porcelain=v1)" ]] || fail "source checkout is not clean"
[[ "$(sha256sum "$manifest" | awk '{print $1}')" == "$expected_manifest_sha256" ]] ||
  fail "confirmatory manifest SHA-256 drifted"
[[ "$(sha256sum "$manifest_provenance" | awk '{print $1}')" == "$expected_manifest_provenance_sha256" ]] ||
  fail "confirmatory provenance SHA-256 drifted"
[[ "$(sha256sum "$source_manifest" | awk '{print $1}')" == "$expected_source_manifest_sha256" ]] ||
  fail "source manifest SHA-256 drifted"

mkdir -p "$run_root/provenance"
export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE=disabled
export CUBLAS_WORKSPACE_CONFIG=:4096:8

printf '%s\n' "$observed_commit" >"$run_root/provenance/code_commit.txt"
git status --porcelain=v1 >"$run_root/provenance/code_status.txt"
git diff --binary >"$run_root/provenance/code.patch"
printf '%s\n' "$expected_commit" >"$run_root/provenance/expected_code_commit.txt"
printf '%s\n' "$manifest" >"$run_root/provenance/manifest_path.txt"
printf '%s\n' "$manifest_provenance" >"$run_root/provenance/manifest_provenance_path.txt"
sha256sum "$manifest" "$manifest_provenance" "$source_manifest" \
  >"$run_root/provenance/dataset_files.sha256"
sha256sum pixi.lock pyproject.toml >"$run_root/provenance/environment_files.sha256"
{
  printf 'CUBLAS_WORKSPACE_CONFIG=%s\n' ':4096:8'
  printf 'CS1K_DATALOADER_WORKERS=%s\n' "$dataloader_workers"
  printf 'CS1K_DATALOADER_PREFETCH_FACTOR=%s\n' "$dataloader_prefetch_factor"
  printf 'CS1K_DATALOADER_PERSISTENT_WORKERS=%s\n' "$dataloader_persistent_workers"
  printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-}"
  printf 'MIRA_PYTHON=%s\n' "$python_bin"
  printf 'PYTHONHASHSEED=%s\n' "${PYTHONHASHSEED:-}"
} >"$run_root/provenance/environment.txt"
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

"$python_bin" scripts/prepare_cs2_confirmatory_split.py \
  --source-manifest "$source_manifest" \
  --output-manifest "$manifest" \
  --provenance-output "$manifest_provenance" \
  --verify-only \
  >"$run_root/provenance/confirmatory_split_verification.json"
"$python_bin" scripts/prepare_counterstrike1k.py \
  --data-root "$dataset_dir" \
  --manifest "$manifest" \
  --map-slug dust2 \
  --splits train val test pilot_test \
  --provenance-output "$run_root/provenance/dataset.json" \
  >"$run_root/provenance/materialization_verification.json"

write_status() {
  local stage=$1
  local state=$2
  printf '%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$stage" "$state" \
    >>"$run_root/pipeline_status.tsv"
}

telemetry_pid=
review_pid=
cleanup_background() {
  if [[ -n "$review_pid" ]] && kill -0 "$review_pid" 2>/dev/null; then
    kill "$review_pid" 2>/dev/null || true
    wait "$review_pid" 2>/dev/null || true
  fi
  if [[ -n "$telemetry_pid" ]] && kill -0 "$telemetry_pid" 2>/dev/null; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
}
trap cleanup_background EXIT

nvidia-smi \
  --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
  --format=csv -l 5 >"$run_root/provenance/gpu_timeseries.csv" &
telemetry_pid=$!

if [[ -n "$review_s3_uri" ]]; then
  command -v aws >/dev/null || fail "aws CLI is required when MIRA_REVIEW_S3_URI is set"
  (
    while true; do
      if [[ -d "$run_root/single/rollout_traces" ]]; then
        aws s3 sync "$run_root/single/rollout_traces" \
          "$review_s3_uri/training-rollouts" --only-show-errors
      fi
      sleep 30
    done
  ) >"$run_root/provenance/review_sync.log" 2>&1 &
  review_pid=$!
  printf '%s\n' "$review_s3_uri" >"$run_root/provenance/review_s3_uri.txt"
fi

write_status codec running
"$python_bin" scripts/train_codec.py \
  model=raev2_codec_cs2_fast \
  dataset=counterstrike1k_dust2 \
  dataset.train_index="$manifest" \
  dataset.test_index="$manifest" \
  dataset.train_split=train \
  dataset.test_split=val \
  run.seed=28 \
  run.steps="$((codec_endpoint + 1))" \
  run.batch_size=4 \
  run.deterministic=true \
  run.compile=false \
  run.max_duration_hours=null \
  run.log_every=25 \
  run.checkpoint_every=6000 \
  run.checkpoint_keep_recent=3 \
  run.output_dir="$run_root/codec" \
  dataloader.num_workers="$dataloader_workers" \
  dataloader.shuffle_buffer_size=100 \
  dataloader.prefetch_factor="$dataloader_prefetch_factor" \
  dataloader.persistent_workers="$dataloader_persistent_workers" \
  validation.val_first=true \
  validation.val_every=6000 \
  validation.val_n_samples=64 \
  optim.scheduler.warmup_steps=500 \
  optim.scheduler.decay_steps=0 \
  wandb.mode=disabled \
  2>&1 | tee "$run_root/codec.log"

codec_checkpoint=$run_root/codec/checkpoint-$codec_endpoint/checkpoint.pth
[[ -f "$codec_checkpoint" ]] || fail "exact codec endpoint is absent: $codec_checkpoint"
write_status codec complete
printf '%s\n' "$codec_checkpoint" >"$run_root/codec_checkpoint.txt"
sha256sum "$codec_checkpoint" >"$run_root/provenance/codec_checkpoint.sha256"

write_status single running
"$python_bin" scripts/train_world_model.py \
  model=latent_world_model_cs2_small \
  dataset=counterstrike1k_dust2 \
  dataset.train_index="$manifest" \
  dataset.test_index="$manifest" \
  dataset.train_split=train \
  dataset.test_split=val \
  dataset.n_players=1 \
  dataset.group_mode=single \
  dataset.validation_group_mode=single \
  model.architecture.config.codec_checkpoint="$codec_checkpoint" \
  run.seed=28 \
  run.steps="$((single_endpoint + 1))" \
  run.batch_size=10 \
  run.deterministic=true \
  run.compile=false \
  run.max_duration_hours=null \
  run.log_every=25 \
  run.checkpoint_every=3000 \
  run.checkpoint_keep_recent=5 \
  run.output_dir="$run_root/single" \
  dataloader.num_workers="$dataloader_workers" \
  dataloader.shuffle_buffer_size=100 \
  dataloader.prefetch_factor="$dataloader_prefetch_factor" \
  dataloader.persistent_workers="$dataloader_persistent_workers" \
  validation.val_first=true \
  validation.val_every=1000 \
  validation.val_n_samples=100 \
  validation.local_rollout_every=1000 \
  validation.local_rollout_seed=37 \
  validation.downstream_val_every=1000000 \
  optim.scheduler.warmup_steps=500 \
  optim.scheduler.decay_steps=0 \
  optim.model_ema_decay=0.999 \
  world_model_metrics.n_context_frames=8 \
  world_model_metrics.num_unrolled_frames=4 \
  world_model_metrics.drift_metric_frames=4 \
  world_model_metrics.fdd_slice_frames=2 \
  world_model_metrics.dino_model=dinov2_vitb14 \
  world_model_metrics.num_samples=10 \
  world_model_metrics.per_device_batch_size=10 \
  world_model_metrics.num_viz_samples=1 \
  wandb.mode=disabled \
  2>&1 | tee "$run_root/single.log"

single_checkpoint=$run_root/single/checkpoint-$single_endpoint/checkpoint.pth
[[ -f "$single_checkpoint" ]] || fail "exact single endpoint is absent: $single_checkpoint"
write_status single complete
printf '%s\n' "$single_checkpoint" >"$run_root/single_checkpoint.txt"
sha256sum "$single_checkpoint" >"$run_root/provenance/single_checkpoint.sha256"

if [[ -n "$review_s3_uri" ]]; then
  aws s3 sync "$run_root/single/rollout_traces" \
    "$review_s3_uri/training-rollouts" --only-show-errors
fi

eval_root=$run_root/evaluation/action_conditioning
write_status action_evaluation running
for window in $window_modes; do
  for seed in $eval_seeds; do
    output_dir=$eval_root/$window/seed_$seed
    mkdir -p "$output_dir"
    for mode in $action_modes; do
      "$python_bin" scripts/eval_world_model_offline.py "$single_checkpoint" \
        --split test \
        --group-mode single \
        --window-mode "$window" \
        --action-mode "$mode" \
        --seed "$seed" \
        --deterministic \
        --val-n-samples 690 \
        --skip-metrics \
        --no-compile \
        --results-json "$output_dir/$mode.json" \
        --per-batch-jsonl "$output_dir/$mode.rounds.jsonl" \
        2>&1 | tee "$output_dir/$mode.log"
    done
  done
done

"$python_bin" scripts/summarize_cs2_single_action_eval.py "$eval_root" \
  --output "$eval_root/summary.json" \
  >"$eval_root/summary.stdout.json"
write_status action_evaluation complete

if [[ -n "$review_s3_uri" ]]; then
  aws s3 sync "$eval_root" "$review_s3_uri/action-conditioning" --only-show-errors
fi

write_status pipeline complete
cleanup_background
trap - EXIT
