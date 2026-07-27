#!/usr/bin/env bash
# Causal future-event probe over frozen single/synchronized/shuffled MIRA representations.
#
# The synchronized and shuffled checkpoints receive identical synchronized contexts.  Feature
# extraction stops before the labelled horizon and never passes future pixels/actions to MIRA.
# This launcher changes neither MIRA checkpoint weights nor model source.
set -euo pipefail

mira_dir=${MIRA_PROJECT_DIR:-$PWD}
release_dir=${CS1K_RELEASE_DIR:?Set CS1K_RELEASE_DIR to the public CounterStrike-1K checkout}
dataset_dir=${CS1K_DATASET_DIR:?Set CS1K_DATASET_DIR}
manifest_path=${CS1K_MANIFEST_PATH:?Set the frozen confirmatory manifest}
single_checkpoint=${CS1K_SINGLE_CHECKPOINT:?Set the frozen single-MIRA checkpoint}
codec_checkpoint=${CS1K_CODEC_CHECKPOINT:?Set the frozen codec checkpoint}
control_root=${CS1K_CONTROL_ROOT:?Set the synchronized-vs-shuffled training root}
output_root=${CS1K_EVENT_OUTPUT_ROOT:?Set a new event-probe output root}
mira_python=${MIRA_PYTHON:-$mira_dir/.pixi/envs/default/bin/python}
release_python=${CS1K_RELEASE_PYTHON:-$release_dir/.venv/bin/python}
explicit_synchronized_checkpoint=${CS1K_SYNCHRONIZED_CHECKPOINT:-}
explicit_shuffled_checkpoint=${CS1K_SHUFFLED_CHECKPOINT:-}
feature_seed=${CS1K_EVENT_FEATURE_SEED:-37}
probe_seeds=${CS1K_EVENT_PROBE_SEEDS:-"17 29 43"}
bootstrap_samples=${CS1K_EVENT_BOOTSTRAP_SAMPLES:-10000}
max_windows_per_split=${CS1K_EVENT_MAX_WINDOWS_PER_SPLIT:-}
expected_mira_commit=${CS1K_EXPECTED_MIRA_COMMIT:-}
expected_release_commit=${CS1K_EXPECTED_RELEASE_COMMIT:-}

cd "$mira_dir"
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB_MODE=disabled

for repo in "$mira_dir" "$release_dir"; do
  if [[ -n "$(git -C "$repo" status --porcelain=v1)" ]]; then
    echo "Event-probe publication requires a clean source tree: $repo" >&2
    exit 1
  fi
done
if [[ -n "$expected_mira_commit" ]] &&
  [[ "$(git -C "$mira_dir" rev-parse HEAD)" != "$expected_mira_commit" ]]; then
  echo "MIRA commit drifted after sweep submission" >&2
  exit 1
fi
if [[ -n "$expected_release_commit" ]] &&
  [[ "$(git -C "$release_dir" rev-parse HEAD)" != "$expected_release_commit" ]]; then
  echo "CounterStrike-1K commit drifted after sweep submission" >&2
  exit 1
fi
for path in "$manifest_path" "$single_checkpoint" "$codec_checkpoint"; do
  if [[ ! -f "$path" ]]; then
    echo "Required event-probe input is absent: $path" >&2
    exit 1
  fi
done
if [[ "$(dirname "$(realpath "$manifest_path")")" != "$(realpath "$dataset_dir")" ]]; then
  echo "CS1K_MANIFEST_PATH must live directly inside CS1K_DATASET_DIR" >&2
  exit 1
fi
if [[ -e "$output_root" ]]; then
  echo "Refusing to resume or overwrite existing event-probe root: $output_root" >&2
  exit 1
fi

latest_checkpoint() {
  local arm=$1
  find "$control_root/$arm" -path '*/checkpoint-*/checkpoint.pth' -print0 |
    sort -zV | tail -z -n 1 | tr -d '\0'
}
if { [[ -n "$explicit_synchronized_checkpoint" ]] && [[ -z "$explicit_shuffled_checkpoint" ]]; } ||
  { [[ -z "$explicit_synchronized_checkpoint" ]] && [[ -n "$explicit_shuffled_checkpoint" ]]; }; then
  echo "Set both CS1K_SYNCHRONIZED_CHECKPOINT and CS1K_SHUFFLED_CHECKPOINT, or neither" >&2
  exit 1
fi
if [[ -n "$explicit_synchronized_checkpoint" ]]; then
  synchronized_checkpoint=$(realpath "$explicit_synchronized_checkpoint")
  shuffled_checkpoint=$(realpath "$explicit_shuffled_checkpoint")
else
  shuffled_checkpoint=$(latest_checkpoint shuffled)
  synchronized_checkpoint=$(latest_checkpoint synchronized)
fi
if [[ -z "$shuffled_checkpoint" || -z "$synchronized_checkpoint" ]]; then
  echo "Frozen shuffled and synchronized checkpoints are required under $control_root" >&2
  exit 1
fi
for checkpoint in "$synchronized_checkpoint" "$shuffled_checkpoint"; do
  if [[ ! -f "$checkpoint" ]]; then
    echo "Frozen control checkpoint not found: $checkpoint" >&2
    exit 1
  fi
done

mkdir -p "$output_root/provenance"
status_file=$output_root/status.tsv
write_status() {
  printf '%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" >>"$status_file"
}
git -C "$mira_dir" rev-parse HEAD >"$output_root/provenance/mira_commit.txt"
git -C "$release_dir" rev-parse HEAD >"$output_root/provenance/release_commit.txt"
git -C "$mira_dir" status --porcelain=v1 >"$output_root/provenance/mira_status.txt"
git -C "$release_dir" status --porcelain=v1 >"$output_root/provenance/release_status.txt"
printf '%s\n' \
  "$codec_checkpoint" \
  "$single_checkpoint" \
  "$synchronized_checkpoint" \
  "$shuffled_checkpoint" \
  >"$output_root/provenance/checkpoints.txt"
sha256sum \
  "$codec_checkpoint" \
  "$single_checkpoint" \
  "$synchronized_checkpoint" \
  "$shuffled_checkpoint" \
  >"$output_root/provenance/checkpoints.sha256"
sha256sum "$manifest_path" >"$output_root/provenance/manifest.sha256"
printf '%s\n' \
  "feature_seed=$feature_seed" \
  "probe_seeds=$probe_seeds" \
  "bootstrap_samples=$bootstrap_samples" \
  "context_frames=8" \
  "future_label_horizon_frames=8" \
  "future_pixels_or_actions_passed_to_model=false" \
  "primary_comparison=synchronized_minus_shuffled" \
  >"$output_root/provenance/contract.env"

max_args=()
if [[ -n "$max_windows_per_split" ]]; then
  max_args=(--max-windows-per-split "$max_windows_per_split")
fi

for arm in single synchronized shuffled; do
  case "$arm" in
    single) checkpoint=$single_checkpoint ;;
    synchronized) checkpoint=$synchronized_checkpoint ;;
    shuffled) checkpoint=$shuffled_checkpoint ;;
  esac
  write_status "features_$arm" running
  PYTHONPATH="$mira_dir/src" "$mira_python" \
    "$mira_dir/scripts/extract_cs2_future_event_features.py" \
    "$checkpoint" \
    --dataset-root "$dataset_dir" \
    --manifest "$manifest_path" \
    --codec-checkpoint "$codec_checkpoint" \
    --out "$output_root/features/$arm" \
    --context-frames 8 \
    --future-frames 8 \
    --seed "$feature_seed" \
    --num-workers 4 \
    --prefetch-factor 2 \
    --device cuda \
    --autocast-dtype bfloat16 \
    "${max_args[@]}"
  write_status "features_$arm" complete
done

"$release_python" - \
  "$output_root/features/single/embedding_index.parquet" \
  "$output_root/features/synchronized/embedding_index.parquet" \
  "$output_root/features/shuffled/embedding_index.parquet" \
  >"$output_root/provenance/feature_index_verification.json" <<'PY'
import json
import sys

import pandas as pd

columns = ["eval_window_id", "sample_key", "pov_idx", "split", "start_frame", "end_frame"]
tables = [pd.read_parquet(path)[columns].reset_index(drop=True) for path in sys.argv[1:]]
if not tables[0].equals(tables[1]) or not tables[0].equals(tables[2]):
    raise ValueError("single/synchronized/shuffled feature indices are not identical")
print(json.dumps({
    "verified": True,
    "rows": len(tables[0]),
    "windows": int(tables[0]["eval_window_id"].nunique()),
    "splits": sorted(tables[0]["split"].astype(str).unique().tolist()),
}, indent=2, sort_keys=True))
PY

write_status labels running
PYTHONPATH="$release_dir" "$release_python" -m cs2_release.future_events.labels \
  --root "$dataset_dir" \
  --shard-root "$dataset_dir" \
  --resolution 360p \
  --windows "$output_root/features/synchronized/embedding_index.parquet" \
  --horizon-seconds 1.0 \
  --out "$output_root/labels"
write_status labels complete

read -r -a probe_seed_args <<<"$probe_seeds"
write_status probes running
PYTHONPATH="$release_dir" "$mira_python" -m cs2_release.future_events.train \
  --labels "$output_root/labels/future_event_labels.parquet" \
  --checkpoint-embeddings "$output_root/features" \
  --seeds "${probe_seed_args[@]}" \
  --bootstrap-samples "$bootstrap_samples" \
  --device cuda \
  --out "$output_root/probes"
write_status probes complete
write_status pipeline complete
echo "$output_root"
