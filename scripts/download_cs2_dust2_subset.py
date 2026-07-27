#!/usr/bin/env python3
"""Download the frozen CounterStrike-1K 360p Dust2 source shards from Hugging Face.

Only shards containing a Dust2 sample in the pinned release manifest are downloaded. The
materialization and confirmatory-split steps are intentionally separate so their existing
fail-closed verifiers remain the publication contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, snapshot_download

METADATA_REPO = "ArnieRamesh/CounterStrike-1K"
METADATA_REVISION = "5a105b6e470407769d17fd14d73ef44e21b61b9a"
WDS_REPO = "ArnieRamesh/CounterStrike-1K-360-wds"
WDS_REVISION = "509e628617aa2ff2af3e848cf1aec89592a4c94b"
EXPECTED_MANIFEST_SHA256 = "e6d1199595327ccbed7a79760266f1c30fdcf23b717232705c9b11bb6d8707d3"
EXPECTED_SAMPLE_INDEX_SHA256 = "6b833c23a919d0323611a946897b284159ce100de8191ceeba883dea25984bee"
EXPECTED_DUST2_SAMPLES = 9_410
EXPECTED_DUST2_MATCHES = 45
EXPECTED_DUST2_SHARDS = 116


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_metadata(data_root: Path) -> tuple[Path, Path]:
    paths = []
    for filename in ("manifest.parquet", "sample_index_360p.parquet"):
        paths.append(
            Path(
                hf_hub_download(
                    repo_id=METADATA_REPO,
                    filename=filename,
                    repo_type="dataset",
                    revision=METADATA_REVISION,
                    local_dir=data_root,
                )
            )
        )
    manifest_path, sample_index_path = paths
    observed = {
        manifest_path.name: sha256_file(manifest_path),
        sample_index_path.name: sha256_file(sample_index_path),
    }
    expected = {
        manifest_path.name: EXPECTED_MANIFEST_SHA256,
        sample_index_path.name: EXPECTED_SAMPLE_INDEX_SHA256,
    }
    if observed != expected:
        raise ValueError(f"Pinned CounterStrike-1K metadata hashes drifted: {observed}")
    return manifest_path, sample_index_path


def _dust2_shards(manifest_path: Path, sample_index_path: Path) -> tuple[list[str], dict]:
    manifest = pq.read_table(manifest_path)
    dust2 = manifest.filter(pc.equal(manifest["map_slug"], "dust2"))
    sample_keys = dust2["sample_key"]
    match_ids = {str(value) for value in dust2["match_id"].to_pylist()}
    if dust2.num_rows != EXPECTED_DUST2_SAMPLES or len(match_ids) != EXPECTED_DUST2_MATCHES:
        raise ValueError(
            "Pinned Dust2 selection drifted: "
            f"samples={dust2.num_rows}, matches={len(match_ids)}"
        )

    sample_index = pq.read_table(
        sample_index_path,
        columns=["sample_key", "member_suffix", "shard_path"],
    )
    indexed = sample_index.filter(pc.is_in(sample_index["sample_key"], value_set=sample_keys))
    mp4_rows = indexed.filter(pc.equal(indexed["member_suffix"], "mp4"))
    shard_paths = sorted({str(value) for value in mp4_rows["shard_path"].to_pylist()})
    if len(shard_paths) != EXPECTED_DUST2_SHARDS:
        raise ValueError(f"Pinned Dust2 shard count drifted: {len(shard_paths)}")
    expected_prefixes = {
        f"shards-360p/v12/match_{match_id}/"
        for match_id in match_ids
    }
    unexpected = [
        path
        for path in shard_paths
        if not any(path.startswith(prefix) for prefix in expected_prefixes)
    ]
    if unexpected:
        raise ValueError(f"Dust2 index resolved unexpected shard paths: {unexpected[:3]}")
    return shard_paths, {
        "samples": dust2.num_rows,
        "matches": len(match_ids),
        "match_ids": sorted(match_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 8:
        raise ValueError("--workers must be between 1 and 8")

    data_root = args.data_root.resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    manifest_path, sample_index_path = _download_metadata(data_root)
    shard_paths, selection = _dust2_shards(manifest_path, sample_index_path)

    snapshot_download(
        repo_id=WDS_REPO,
        repo_type="dataset",
        revision=WDS_REVISION,
        allow_patterns=shard_paths,
        local_dir=data_root,
        max_workers=args.workers,
    )
    missing = [path for path in shard_paths if not (data_root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} downloaded shards are absent; first: {missing[0]}")
    total_bytes = sum((data_root / path).stat().st_size for path in shard_paths)

    payload = {
        "schema": "mira-cs2-hf-dust2-download-v1",
        "metadata_repository": METADATA_REPO,
        "metadata_revision": METADATA_REVISION,
        "webdataset_repository": WDS_REPO,
        "webdataset_revision": WDS_REVISION,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "sample_index_360p_sha256": EXPECTED_SAMPLE_INDEX_SHA256,
        **selection,
        "shards": len(shard_paths),
        "downloaded_bytes": total_bytes,
        "shard_relative_paths": shard_paths,
    }
    output = data_root / "hf_dust2_download_manifest.json"
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({key: value for key, value in payload.items() if key != "shard_relative_paths"}, indent=2))


if __name__ == "__main__":
    main()
