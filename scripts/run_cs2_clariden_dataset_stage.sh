#!/usr/bin/env bash
# Download, freeze, materialize, and verify the publication Dust2 dataset inside one Slurm job.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /path/to/clariden_sync_sweep.env" >&2
  exit 2
fi
: "${SLURM_JOB_ID:?Run this staging worker through sbatch}"
environment_file=$1
if [[ ! -f "$environment_file" ]]; then
  echo "Environment file not found: $environment_file" >&2
  exit 1
fi
# shellcheck disable=SC1090
set -a
source "$environment_file"
set +a

project_dir=${MIRA_PROJECT_DIR:?Set the pinned MIRA checkout}
release_dir=${CS1K_RELEASE_DIR:?Set the pinned CounterStrike-1K checkout}
data_root=${CS1K_DATASET_DIR:?Set the Dust2 materialization root}
manifest_path=${CS1K_MANIFEST_PATH:?Set the frozen confirmatory manifest}
split_provenance=${CS1K_CONFIRMATORY_SPLIT_PROVENANCE:?Set split provenance}
venv_dir=${CS1K_CLARIDEN_VENV:?Set the persistent Clariden venv path}

mkdir -p "$data_root"
bash "$project_dir/scripts/setup_cs2_clariden_uenv.sh"
python_bin=$venv_dir/bin/python

"$python_bin" "$project_dir/scripts/download_cs2_dust2_subset.py" \
  --data-root "$data_root" \
  --workers "${CS1K_DATASET_DOWNLOAD_WORKERS:-4}"

"$python_bin" "$project_dir/scripts/prepare_cs2_confirmatory_split.py" \
  --source-manifest "$data_root/manifest.parquet" \
  --output-manifest "$manifest_path" \
  --provenance-output "$split_provenance"

index_root=$data_root/.stage_index
mkdir -p "$index_root"
cp "$manifest_path" "$index_root/$(basename "$manifest_path")"
cp "$data_root/sample_index_360p.parquet" "$index_root/sample_index_360p.parquet"

PYTHONPATH="$release_dir" "$python_bin" -m cs2_train.scripts.materialize_dust2_subset \
  --source-root "$index_root" \
  --shard-root "$data_root" \
  --output-root "$data_root" \
  --manifest-name "$(basename "$manifest_path")" \
  --sample-index-name sample_index_360p.parquet \
  --resolution 360p \
  --map-slug dust2 \
  --splits train val test pilot_test \
  --workers "${CS1K_DATASET_MATERIALIZE_WORKERS:-8}"

"$python_bin" "$project_dir/scripts/prepare_counterstrike1k.py" \
  --data-root "$data_root" \
  --manifest "$manifest_path" \
  --map-slug dust2 \
  --splits train val test pilot_test \
  --provenance-output "$data_root/dust2_materialization.provenance.json"

expected_manifest_sha256=33abbb623072932431871a612620110c473d4b664c52010e5763c273c6daf10e
if [[ "$(sha256sum "$manifest_path" | cut -d' ' -f1)" != "$expected_manifest_sha256" ]]; then
  echo "Generated confirmatory manifest SHA-256 drifted" >&2
  exit 1
fi

"$python_bin" - \
  "$data_root/dataset_stage_complete.json" \
  "$data_root/hf_dust2_download_manifest.json" \
  "$data_root/materialization_provenance.json" \
  "$data_root/dust2_materialization.provenance.json" \
  "$manifest_path" \
  "$split_provenance" <<'PY'
import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


(
    output,
    download_path,
    hash_materialization_path,
    materialization_path,
    manifest_path,
    split_path,
) = map(Path, sys.argv[1:])
download = json.loads(download_path.read_text(encoding="utf-8"))
hash_materialization = json.loads(hash_materialization_path.read_text(encoding="utf-8"))
materialization = json.loads(materialization_path.read_text(encoding="utf-8"))
split = json.loads(split_path.read_text(encoding="utf-8"))
if download.get("shards") != 116 or download.get("samples") != 9410:
    raise SystemExit("Downloaded Dust2 source cardinality drifted")
if hash_materialization.get("samples") != 9410:
    raise SystemExit("Hash-verified materialization cardinality drifted")
if materialization.get("verification") != {"samples": 9410, "payload_files": 47050}:
    raise SystemExit("Dust2 materialization cardinality drifted")
if split.get("confirmatory_selection_sha256") != (
    "056e60b7bb4435e212f43e6a2e4c2ea0f5c1265978180588ff59689a2fbabb0f"
):
    raise SystemExit("Confirmatory split semantic digest drifted")
payload = {
    "schema": "mira-cs2-clariden-dataset-stage-v1",
    "status": "complete",
    "source_shards": download["shards"],
    "samples": download["samples"],
    "payload_files": materialization["verification"]["payload_files"],
    "manifest": str(manifest_path),
    "manifest_sha256": sha256(manifest_path),
    "split_provenance": str(split_path),
    "download_manifest": str(download_path),
    "hash_materialization_provenance": str(hash_materialization_path),
    "dataset_verification_provenance": str(materialization_path),
}
temporary = output.with_suffix(".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(output)
print(json.dumps(payload, indent=2, sort_keys=True))
PY
