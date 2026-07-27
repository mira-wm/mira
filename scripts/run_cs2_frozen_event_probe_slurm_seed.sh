#!/usr/bin/env bash
# One dependent Slurm job for the frozen causal event probe of one training seed.
#
# The submitter pins the exact MIRA and CounterStrike-1K commits and passes an explicit
# checkpoint-<steps-1> from both synchronized and cross-round training arms.
set -euo pipefail

project_dir=${MIRA_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
python_bin=${MIRA_PYTHON:-$project_dir/.pixi/envs/default/bin/python}
output_root=${CS1K_OUTPUT_ROOT:?Set the shared sweep output root}
train_steps=${CS1K_TRAIN_STEPS:?Set the fixed optimizer-update count}
single_checkpoint=${CS1K_SINGLE_CHECKPOINT:?Set the frozen single-MIRA checkpoint}
codec_checkpoint=${CS1K_CODEC_CHECKPOINT:?Set the frozen codec checkpoint}
seed=${CS1K_SEED:?Set the world-model training seed}
expected_single_sha256=${CS1K_EXPECTED_SINGLE_CHECKPOINT_SHA256:-3dbd8f0e43dbe833a5f36370d75f6306c7aa036dfcd3edba767ab138232fa047}
expected_codec_sha256=${CS1K_EXPECTED_CODEC_CHECKPOINT_SHA256:-3c286c59b74cd141e72af69cde1a0a005142d8d2b472c789cdf2a39a140c4b7a}

: "${SLURM_JOB_ID:?Run this event probe through sbatch}"
if ! [[ "$seed" =~ ^[0-9]+$ ]]; then
  echo "CS1K_SEED must be a non-negative integer" >&2
  exit 1
fi
if ! [[ "$train_steps" =~ ^[1-9][0-9]*$ ]]; then
  echo "CS1K_TRAIN_STEPS must be a positive integer" >&2
  exit 1
fi
for command_name in git sha256sum; do
  if ! command -v "$command_name" >/dev/null; then
    echo "Required command not found: $command_name" >&2
    exit 1
  fi
done
if [[ ! -x "$python_bin" ]]; then
  echo "MIRA Python is not executable: $python_bin" >&2
  exit 1
fi
for path in "$single_checkpoint" "$codec_checkpoint"; do
  if [[ ! -f "$path" ]]; then
    echo "Frozen event-probe input is absent: $path" >&2
    exit 1
  fi
done
if [[ "$(sha256sum "$single_checkpoint" | cut -d' ' -f1)" != "$expected_single_sha256" ]]; then
  echo "Single-MIRA checkpoint SHA-256 drifted" >&2
  exit 1
fi
if [[ "$(sha256sum "$codec_checkpoint" | cut -d' ' -f1)" != "$expected_codec_sha256" ]]; then
  echo "Codec checkpoint SHA-256 drifted" >&2
  exit 1
fi

checkpoint_index=$((train_steps - 1))
control_root=$output_root/seed_$seed
audit_path=$control_root/audit.json
event_root=$control_root/event_probe
synchronized_checkpoint=$control_root/synchronized/checkpoint-$checkpoint_index/checkpoint.pth
cross_round_checkpoint=$control_root/shuffled/checkpoint-$checkpoint_index/checkpoint.pth

for path in "$audit_path" "$synchronized_checkpoint" "$cross_round_checkpoint"; do
  if [[ ! -s "$path" ]]; then
    echo "Required audited training artifact is absent: $path" >&2
    exit 1
  fi
done
if [[ -e "$event_root" ]]; then
  echo "Refusing to resume or overwrite event-probe root: $event_root" >&2
  exit 1
fi

"$python_bin" - "$audit_path" "$seed" "$train_steps" <<'PY'
import json
import sys

audit = json.load(open(sys.argv[1], encoding="utf-8"))
expected_seed = int(sys.argv[2])
expected_steps = int(sys.argv[3])
if audit.get("schema") != "mira-cs2-gh200-sync-control-audit-v3":
    raise SystemExit("Unexpected child audit schema")
if audit.get("status") != "pass":
    raise SystemExit("Child training audit did not pass")
if int(audit.get("seed", -1)) != expected_seed:
    raise SystemExit("Child audit seed does not match the requested seed")
if int(audit.get("train_steps", -1)) != expected_steps:
    raise SystemExit("Child audit update budget does not match the requested budget")
PY

# Torch Hub removes and re-extracts the pinned repository while resolving a GitHub revision.
# Concurrent event jobs must therefore never share the mutable per-user cache. Warm one
# job-private cache before the three sequential feature-extraction processes consume it.
torch_home_root=${CS1K_TORCH_HOME_ROOT:-$output_root/torch_hub}
export TORCH_HOME="$torch_home_root/event_$SLURM_JOB_ID"
mkdir -p "$TORCH_HOME"
PYTHONPATH="$project_dir/src" "$python_bin" "$project_dir/scripts/prepare_dinov2_cache.py" \
  --torch-home "$TORCH_HOME" \
  --output "$TORCH_HOME/dinov2_cache_ready.json"

export CS1K_CONTROL_ROOT="$control_root"
export CS1K_SYNCHRONIZED_CHECKPOINT="$synchronized_checkpoint"
# The historical internal key is "shuffled"; this checkpoint is the cross-round-grouped arm.
export CS1K_SHUFFLED_CHECKPOINT="$cross_round_checkpoint"
export CS1K_EVENT_OUTPUT_ROOT="$event_root"

exec "$project_dir/scripts/run_cs2_frozen_event_probe.sh"
