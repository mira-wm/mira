#!/usr/bin/env bash
# Build the MIRA + CounterStrike-1K Python layer on Clariden's pinned PyTorch 2.8 GH200 uenv.
#
# Run through:
#   uenv run --view=default pytorch/v2.8.0:v1 -- \
#     bash scripts/setup_cs2_clariden_uenv.sh
set -euo pipefail

project_dir=${MIRA_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
release_dir=${CS1K_RELEASE_DIR:?Set the pinned CounterStrike-1K checkout}
venv_dir=${CS1K_CLARIDEN_VENV:?Set the persistent Clariden venv path}
uenv_label=${CS1K_CLARIDEN_UENV:-pytorch/v2.8.0:v1}
loader_commit=22dad05a3f2fd6c242a56e55e1eb2af61ed42385
torchcodec_commit=7dd6092b40a76a262b633c591d8edb6ce0a86c11

unset PYTHONPATH
export PYTHONUSERBASE
PYTHONUSERBASE=$(dirname "$(dirname "$(command -v python)")")

for command_name in cmake ffmpeg git pkg-config python sha256sum; do
  if ! command -v "$command_name" >/dev/null; then
    echo "Required uenv command not found: $command_name" >&2
    exit 1
  fi
done
for repo in "$project_dir" "$release_dir"; do
  if [[ ! -d "$repo/.git" ]]; then
    echo "Pinned Git checkout not found: $repo" >&2
    exit 1
  fi
  if [[ -n "$(git -C "$repo" status --porcelain=v1)" ]]; then
    echo "Clariden environment setup requires a clean checkout: $repo" >&2
    exit 1
  fi
done

python - <<'PY'
import platform
import sys

import torch

if platform.machine() not in {"aarch64", "arm64"}:
    raise SystemExit(f"Expected a Clariden ARM64 runtime, found {platform.machine()}")
if not torch.__version__.split("+", 1)[0].startswith("2.8."):
    raise SystemExit(f"Expected the PyTorch 2.8 uenv, found torch {torch.__version__}")
if sys.version_info[:2] > (3, 13):
    raise SystemExit(f"TorchCodec 0.7 supports Python through 3.13, found {sys.version}")
PY

if [[ ! -d "$venv_dir" ]]; then
  python -m venv --system-site-packages "$venv_dir"
fi
venv_python=$venv_dir/bin/python
if [[ ! -x "$venv_python" ]]; then
  echo "Clariden venv Python is not executable inside the uenv: $venv_python" >&2
  exit 1
fi

"$venv_python" -m pip install \
  --disable-pip-version-check \
  --upgrade \
  "pip==26.1.2" \
  "setuptools==79.0.1" \
  "wheel==0.47.0"

# TorchCodec 0.7 is the release matched to PyTorch 2.8, but it did not publish a Linux ARM64
# wheel. Build the exact upstream release commit against the uenv's FFmpeg shared libraries.
# Keep the source checkout beside the persistent venv so the editable installation remains valid.
if ! pkg-config --exists libavcodec libavformat libavutil libswresample libswscale; then
  echo "The PyTorch uenv does not expose the FFmpeg development libraries" >&2
  exit 1
fi
torchcodec_source=$venv_dir/cs1k_sources/torchcodec-$torchcodec_commit
if [[ ! -d "$torchcodec_source/.git" ]]; then
  mkdir -p "$(dirname "$torchcodec_source")"
  git clone \
    --branch v0.7.0 \
    --depth 1 \
    https://github.com/meta-pytorch/torchcodec.git \
    "$torchcodec_source"
fi
if [[ "$(git -C "$torchcodec_source" rev-parse HEAD)" != "$torchcodec_commit" ]]; then
  echo "Pinned TorchCodec source commit drifted: $torchcodec_source" >&2
  exit 1
fi
export CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL:-8}
export TORCHCODEC_CMAKE_BUILD_DIR=$venv_dir/cs1k_build/torchcodec-$torchcodec_commit
export TORCHCODEC_DISABLE_COMPILE_WARNING_AS_ERROR=ON
BUILD_VERSION=0.7.0 "$venv_python" -m pip install \
  --disable-pip-version-check \
  --no-build-isolation \
  --no-deps \
  --editable "$torchcodec_source"
"$venv_python" -m pip install \
  --disable-pip-version-check \
  "$project_dir[viz,hf,train,eval,dev]"
"$venv_python" -m pip install \
  --disable-pip-version-check \
  "counterstrike1k @ git+https://github.com/AnirudhhRamesh/counterstrike1k.git@$loader_commit" \
  pandas
"$venv_python" -m pip install \
  --disable-pip-version-check \
  --no-deps \
  "$release_dir"

provenance_dir=$venv_dir/cs1k_provenance
mkdir -p "$provenance_dir"
"$venv_python" -m pip freeze --all >"$provenance_dir/pip_freeze.txt"
git -C "$project_dir" rev-parse HEAD >"$provenance_dir/mira_commit.txt"
git -C "$release_dir" rev-parse HEAD >"$provenance_dir/counterstrike_1k_commit.txt"
git -C "$torchcodec_source" rev-parse HEAD >"$provenance_dir/torchcodec_commit.txt"
git -C "$torchcodec_source" status --porcelain=v1 >"$provenance_dir/torchcodec_source_status.txt"
printf '%s\n' "$uenv_label" >"$provenance_dir/uenv_label.txt"
ffmpeg -version >"$provenance_dir/ffmpeg_version.txt"

"$venv_python" - \
  "$provenance_dir/environment.json" \
  "$uenv_label" \
  "$loader_commit" \
  "$torchcodec_commit" <<'PY'
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import pyarrow
import torch
import torchcodec

loader_distribution = importlib.metadata.distribution("counterstrike1k")
loader_direct_url = json.loads(loader_distribution.read_text("direct_url.json") or "{}")
loader_commit = (
    loader_direct_url.get("vcs_info", {}).get("commit_id")
    or loader_direct_url.get("archive_info", {}).get("hash")
)
payload = {
    "schema": "mira-cs2-clariden-uenv-v1",
    "uenv": sys.argv[2],
    "machine": platform.machine(),
    "python": sys.version,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "torchcodec": importlib.metadata.version("torchcodec"),
    "torchcodec_commit": sys.argv[4],
    "torchcodec_install": "source-editable",
    "pyarrow": pyarrow.__version__,
    "counterstrike1k": loader_distribution.version,
    "counterstrike1k_commit": loader_commit,
}
if not payload["torch"].split("+", 1)[0].startswith("2.8."):
    raise SystemExit(f"Unexpected torch version after install: {payload['torch']}")
if payload["torchcodec"] != "0.7.0":
    raise SystemExit(f"Unexpected torchcodec version after install: {payload['torchcodec']}")
if payload["torchcodec_commit"] != "7dd6092b40a76a262b633c591d8edb6ce0a86c11":
    raise SystemExit(f"Unexpected torchcodec source commit: {payload['torchcodec_commit']}")
if loader_commit != sys.argv[3]:
    raise SystemExit(
        f"Unexpected counterstrike1k commit after install: {loader_commit}; expected {sys.argv[3]}"
    )
path = Path(sys.argv[1])
temporary = path.with_suffix(".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(path)
PY

printf 'Clariden runtime ready:\n'
printf '  uenv: %s\n' "$uenv_label"
printf '  python: %s\n' "$venv_python"
printf '  provenance: %s\n' "$provenance_dir"
