#!/usr/bin/env bash
# Download and verify the three-match Dust2 validation subset used for isolated loader benchmarks.
set -euo pipefail

data_root=${1:?Usage: stage_cs2_loader_benchmark_val.sh DATA_ROOT PROVENANCE_JSON}
provenance_output=${2:?}
project_dir=${MIRA_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
python_bin=${MIRA_PYTHON:-$project_dir/.pixi/envs/default/bin/python}

repository=https://huggingface.co/datasets/ArnieRamesh/CounterStrike-1K-360-wds
revision=509e628617aa2ff2af3e848cf1aec89592a4c94b
records=(
  "manifest.parquet|2631325|e6d1199595327ccbed7a79760266f1c30fdcf23b717232705c9b11bb6d8707d3"
  "shards-360p/v12/match_252d9e682aeb/shard-0000.tar|1515284480|38a969ad9e7b1ead8549efde29600737874adff6727817e89a4c0914ef9e04dd"
  "shards-360p/v12/match_252d9e682aeb/shard-0001.tar|1377474560|cd52b9cec9fa15ddda4b2b49499416e2e32a4a403caf66ad75f9d6ad8b8fb6cf"
  "shards-360p/v12/match_461ec1e93eef/shard-0000.tar|1698580480|5e1d833de9d1c27a876d65ecd630b13c6098a0b7941efe866fd4847d1fb9389b"
  "shards-360p/v12/match_461ec1e93eef/shard-0001.tar|511600640|2ef268a8392f9a5741d608935fea48c2c19de08f9c8608be286c6852259b7149"
  "shards-360p/v12/match_79d0198760c8/shard-0000.tar|1732454400|68627404f3b81e8d44eba38631cee8ff841d55a4eeb8f5948a3c3c5666b1ed12"
  "shards-360p/v12/match_79d0198760c8/shard-0001.tar|1205288960|c621f96f4376315488480b153210e3ef81100c819e244e90043017bfdd86fc18"
)

mkdir -p "$data_root"
for record in "${records[@]}"; do
  IFS='|' read -r relative expected_bytes expected_sha256 <<<"$record"
  destination=$data_root/$relative
  temporary=$destination.partial
  mkdir -p "$(dirname "$destination")"

  if [[ -f "$destination" ]] &&
    [[ $(stat -c '%s' "$destination") == "$expected_bytes" ]] &&
    [[ $(sha256sum "$destination" | cut -d ' ' -f 1) == "$expected_sha256" ]]; then
    printf 'verified existing %s\n' "$relative"
    continue
  fi

  curl \
    --fail \
    --location \
    --retry 8 \
    --retry-all-errors \
    --connect-timeout 30 \
    --continue-at - \
    --output "$temporary" \
    "$repository/resolve/$revision/$relative"

  observed_bytes=$(stat -c '%s' "$temporary")
  observed_sha256=$(sha256sum "$temporary" | cut -d ' ' -f 1)
  if [[ "$observed_bytes" != "$expected_bytes" || "$observed_sha256" != "$expected_sha256" ]]; then
    printf 'verification failed for %s: bytes=%s sha256=%s\n' \
      "$relative" "$observed_bytes" "$observed_sha256" >&2
    exit 1
  fi
  mv -f "$temporary" "$destination"
  printf 'downloaded and verified %s\n' "$relative"
done

download_manifest=$data_root/hf_download_manifest.tsv
temporary_manifest=$download_manifest.tmp
{
  printf 'repository\trevision\trelative_path\tbytes\tsha256\n'
  for record in "${records[@]}"; do
    IFS='|' read -r relative expected_bytes expected_sha256 <<<"$record"
    printf '%s\t%s\t%s\t%s\t%s\n' \
      "$repository" "$revision" "$relative" "$expected_bytes" "$expected_sha256"
  done
} >"$temporary_manifest"
mv -f "$temporary_manifest" "$download_manifest"

"$python_bin" "$project_dir/scripts/prepare_counterstrike1k.py" \
  --data-root "$data_root" \
  --map-slug dust2 \
  --splits val \
  --extract \
  --provenance-output "$provenance_output"
