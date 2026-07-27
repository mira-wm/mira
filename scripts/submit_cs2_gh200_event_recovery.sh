#!/usr/bin/env bash
# Submit only failed event probes plus final aggregation after successful post-training evaluation.
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [environment-file]" >&2
  exit 2
fi
if [[ $# -eq 1 ]]; then
  if [[ ! -f "$1" ]]; then
    echo "Environment file not found: $1" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "$1"
fi

project_dir=${MIRA_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
python_bin=${MIRA_PYTHON:-$project_dir/.pixi/envs/default/bin/python}
submit_python=${CS1K_SUBMIT_PYTHON:-python3}
release_dir=${CS1K_RELEASE_DIR:?Set the pinned CounterStrike-1K checkout}
release_python=${CS1K_RELEASE_PYTHON:?Set the CounterStrike-1K Python}
output_root=${CS1K_OUTPUT_ROOT:?Set the completed and evaluated training root}
codec_checkpoint=${CS1K_CODEC_CHECKPOINT:?Set the relocated frozen codec checkpoint}
single_checkpoint=${CS1K_SINGLE_CHECKPOINT:?Set the frozen single-MIRA checkpoint}
train_steps=${CS1K_TRAIN_STEPS:?Set the fixed optimizer-update count}
account=${CS1K_SLURM_ACCOUNT:?Set the Clariden Slurm account}
partition=${CS1K_GH200_PARTITION:?Set the Clariden GH200 partition}
event_time=${CS1K_EVENT_TIME:-04:00:00}
finalize_time=${CS1K_FINALIZE_TIME:-00:30:00}
training_seed_text=${CS1K_TRAINING_SEEDS:-"28 29 30"}
recovery_seed_text=${CS1K_EVENT_RECOVERY_SEEDS:-$training_seed_text}
expected_codec_sha256=${CS1K_EXPECTED_CODEC_CHECKPOINT_SHA256:-3c286c59b74cd141e72af69cde1a0a005142d8d2b472c789cdf2a39a140c4b7a}
expected_single_sha256=${CS1K_EXPECTED_SINGLE_CHECKPOINT_SHA256:-3dbd8f0e43dbe833a5f36370d75f6306c7aa036dfcd3edba767ab138232fa047}
recovery_manifest=${CS1K_EVENT_RECOVERY_MANIFEST:-$output_root/event_recovery_manifest.json}
archive_tag=${CS1K_EVENT_RECOVERY_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}

for command_name in date git mv sbatch sha256sum "$submit_python"; do
  if ! command -v "$command_name" >/dev/null; then
    echo "Required command not found: $command_name" >&2
    exit 1
  fi
done
for directory in "$project_dir" "$release_dir" "$output_root"; do
  if [[ ! -d "$directory" ]]; then
    echo "Required directory is absent: $directory" >&2
    exit 1
  fi
done
for path in "$python_bin" "$release_python"; do
  if [[ ! -x "$path" && ! -L "$path" ]]; then
    echo "Configured runtime Python is absent: $path" >&2
    exit 1
  fi
done
for repo in "$project_dir" "$release_dir"; do
  if [[ -n "$(git -C "$repo" status --porcelain=v1)" ]]; then
    echo "Event recovery requires a clean checkout: $repo" >&2
    exit 1
  fi
done
if [[ ! "$archive_tag" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "CS1K_EVENT_RECOVERY_TAG contains unsafe characters" >&2
  exit 1
fi
if [[ -e "$recovery_manifest" ]]; then
  echo "Event recovery was already submitted: $recovery_manifest" >&2
  exit 1
fi
for path in "$output_root/sweep_summary.json" "$output_root/event_probe_sweep_summary.json"; do
  if [[ -e "$path" ]]; then
    echo "Sweep is already finalized: $path" >&2
    exit 1
  fi
done
if [[ ! -f "$codec_checkpoint" ]] ||
  [[ "$(sha256sum "$codec_checkpoint" | cut -d' ' -f1)" != "$expected_codec_sha256" ]]; then
  echo "Relocated codec checkpoint is absent or has the wrong SHA-256" >&2
  exit 1
fi
if [[ ! -f "$single_checkpoint" ]] ||
  [[ "$(sha256sum "$single_checkpoint" | cut -d' ' -f1)" != "$expected_single_sha256" ]]; then
  echo "Single-MIRA checkpoint is absent or has the wrong SHA-256" >&2
  exit 1
fi

read -r -a training_seeds <<<"$training_seed_text"
if [[ "${training_seeds[*]}" != "28 29 30" ]]; then
  echo "Publication recovery requires exactly CS1K_TRAINING_SEEDS='28 29 30'" >&2
  exit 1
fi
read -r -a recovery_seeds <<<"$recovery_seed_text"
if [[ ${#recovery_seeds[@]} -eq 0 ]]; then
  echo "CS1K_EVENT_RECOVERY_SEEDS must select at least one failed seed" >&2
  exit 1
fi
declare -A recovery_seed_set=()
for seed in "${recovery_seeds[@]}"; do
  if [[ "$seed" != "28" && "$seed" != "29" && "$seed" != "30" ]]; then
    echo "Event recovery seed is outside the frozen training seeds: $seed" >&2
    exit 1
  fi
  if [[ -n "${recovery_seed_set[$seed]:-}" ]]; then
    echo "Duplicate event recovery seed: $seed" >&2
    exit 1
  fi
  recovery_seed_set[$seed]=1
done
if ! [[ "$train_steps" =~ ^[1-9][0-9]*$ ]]; then
  echo "CS1K_TRAIN_STEPS must be a positive integer" >&2
  exit 1
fi

for seed in "${training_seeds[@]}"; do
  seed_root=$output_root/seed_$seed
  required_paths=(
    "$seed_root/audit.json"
    "$seed_root/evaluation/synchronized_test_seed_sweep/summary.json"
    "$seed_root/evaluation/synchronized_test_action_loss_seed_sweep/summary.json"
    "$seed_root/evaluation/synchronized_test_first_death_action_loss_seed_sweep/summary.json"
  )
  for path in "${required_paths[@]}"; do
    if [[ ! -s "$path" ]]; then
      echo "Successful post-training evaluation artifact is absent: $path" >&2
      exit 1
    fi
  done
  "$submit_python" - "$seed_root/audit.json" "$seed" "$train_steps" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    audit = json.load(handle)
if audit.get("schema") != "mira-cs2-gh200-sync-control-audit-v3":
    raise SystemExit("Unexpected child audit schema")
if audit.get("status") != "pass":
    raise SystemExit("Child audit did not pass")
if int(audit.get("seed", -1)) != int(sys.argv[2]):
    raise SystemExit("Child audit seed mismatch")
if int(audit.get("train_steps", -1)) != int(sys.argv[3]):
    raise SystemExit("Child audit update-budget mismatch")
PY

  event_root=$seed_root/event_probe
  event_complete=false
  if [[ -f "$event_root/status.tsv" ]] &&
    grep -q $'\tpipeline\tcomplete$' "$event_root/status.tsv"; then
    event_complete=true
  fi
  if [[ -n "${recovery_seed_set[$seed]:-}" ]]; then
    if [[ "$event_complete" == true ]]; then
      echo "Requested recovery seed $seed already has a complete event probe" >&2
      exit 1
    fi
  elif [[ "$event_complete" != true ]]; then
    echo "Unselected seed $seed does not have a complete event probe" >&2
    exit 1
  fi
done

archive_paths=()
for seed in "${recovery_seeds[@]}"; do
  seed_root=$output_root/seed_$seed
  event_root=$seed_root/event_probe
  archive_path=$seed_root/event_probe_failed_$archive_tag
  if [[ -e "$archive_path" ]]; then
    echo "Event archive destination already exists: $archive_path" >&2
    exit 1
  fi
  if [[ -e "$event_root" ]]; then
    archive_paths+=("$archive_path")
  else
    archive_paths+=("")
  fi
done

# Preserve every failed partial root under a unique name. The publication path is recreated only
# by a fresh successful job; no failed artifact is deleted or silently resumed.
for index in "${!recovery_seeds[@]}"; do
  seed=${recovery_seeds[$index]}
  archive_path=${archive_paths[$index]}
  if [[ -n "$archive_path" ]]; then
    mv "$output_root/seed_$seed/event_probe" "$archive_path"
  fi
done

export MIRA_PROJECT_DIR="$project_dir"
export MIRA_PYTHON="$python_bin"
export CS1K_RELEASE_DIR="$release_dir"
export CS1K_RELEASE_PYTHON="$release_python"
export CS1K_EXPECTED_MIRA_COMMIT
CS1K_EXPECTED_MIRA_COMMIT=$(git -C "$project_dir" rev-parse HEAD)
export CS1K_EXPECTED_RELEASE_COMMIT
CS1K_EXPECTED_RELEASE_COMMIT=$(git -C "$release_dir" rev-parse HEAD)
export CS1K_EXPECTED_CODEC_CHECKPOINT_SHA256="$expected_codec_sha256"
export CS1K_EXPECTED_SINGLE_CHECKPOINT_SHA256="$expected_single_sha256"

read -r -a auxiliary_extra_args <<<"${CS1K_AUX_SBATCH_ARGS:-}"
parse_job_id() {
  local raw=$1
  local job_id=${raw%%;*}
  if ! [[ "$job_id" =~ ^[0-9]+$ ]]; then
    echo "Could not parse sbatch job id from: $raw" >&2
    return 1
  fi
  printf '%s\n' "$job_id"
}

log_root=$output_root/logs
mkdir -p "$log_root"
event_job_ids=()
for seed in "${recovery_seeds[@]}"; do
  raw_job_id=$(sbatch \
    --parsable \
    --account="$account" \
    --partition="$partition" \
    --job-name="mira-cs2-events-$seed" \
    --nodes=1 \
    --ntasks=1 \
    --gpus-per-node=1 \
    --chdir="$output_root" \
    --output="$log_root/event-recovery-seed-$seed-%j.log" \
    --time="$event_time" \
    --export=ALL,CS1K_SEED="$seed" \
    "${auxiliary_extra_args[@]}" \
    "$project_dir/scripts/run_cs2_frozen_event_probe_slurm_seed.sh")
  event_job_ids+=("$(parse_job_id "$raw_job_id")")
done

event_dependency=$(IFS=:; printf '%s' "${event_job_ids[*]}")
raw_finalize_job_id=$(sbatch \
  --parsable \
  --account="$account" \
  --partition="$partition" \
  --job-name=mira-cs2-finalize \
  --nodes=1 \
  --ntasks=1 \
  --gpus-per-node=1 \
  --chdir="$output_root" \
  --output="$log_root/event-recovery-finalize-%j.log" \
  --time="$finalize_time" \
  --dependency="afterok:$event_dependency" \
  --export=ALL \
  "${auxiliary_extra_args[@]}" \
  "$project_dir/scripts/run_cs2_gh200_sweep_finalize.sh")
finalize_job_id=$(parse_job_id "$raw_finalize_job_id")

"$submit_python" - \
  "$recovery_manifest" \
  "$archive_tag" \
  "$CS1K_EXPECTED_MIRA_COMMIT" \
  "$finalize_job_id" \
  "${recovery_seeds[*]}" \
  "${event_job_ids[*]}" \
  "${archive_paths[*]}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    output,
    archive_tag,
    evaluator_commit,
    finalize_job_id,
    recovery_seeds,
    event_job_ids,
    archive_paths,
) = sys.argv[1:]
payload = {
    "schema": "mira-cs2-event-recovery-submission-v2",
    "submitted_at": datetime.now(timezone.utc).isoformat(),
    "training_reused": True,
    "evaluation_reused": True,
    "codec_path_relocated": True,
    "evaluator_commit": evaluator_commit,
    "archive_tag": archive_tag,
    "archived_failed_event_roots": archive_paths.split(),
    "recovery_seeds": [int(value) for value in recovery_seeds.split()],
    "event_jobs": [int(value) for value in event_job_ids.split()],
    "finalize_job": int(finalize_job_id),
}
Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

printf 'Submitted event-only recovery.\n'
for index in "${!recovery_seeds[@]}"; do
  printf '  seed %s: event %s\n' "${recovery_seeds[$index]}" "${event_job_ids[$index]}"
done
printf '  final aggregation: %s\n' "$finalize_job_id"
printf '  recovery manifest: %s\n' "$recovery_manifest"
