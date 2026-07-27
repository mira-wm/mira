#!/usr/bin/env python
"""Materialize and verify one map from CounterStrike-1K WebDataset shards.

The script treats the full ``manifest.parquet`` as the source of truth, selects one ``map_slug``,
derives the exact match-shard list, optionally extracts the five released payloads per sample, and
emits a canonical provenance report. Re-running without ``--extract`` is a cheap integrity check of
an existing materialization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

ACTION_RECORD_BYTES = 14


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _destination(root: Path, member_name: str) -> Path | None:
    name = Path(member_name).name
    if name.endswith(".mp4"):
        return root / "videos" / "360p" / name
    if name.endswith(".actions.bin"):
        return root / "actions" / name
    if name.endswith(".state.bin"):
        return root / "state" / name
    if name.endswith(".events.json"):
        return root / "events" / name
    if name.endswith(".json"):
        return root / "metadata" / name
    return None


def _extract(root: Path, shards: list[Path]) -> dict[str, int]:
    counts = {"shards": 0, "members": 0, "written": 0, "existing": 0}
    missing = [path for path in shards if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} source shards are missing; first: {missing[0]}")

    for shard_index, shard in enumerate(shards, start=1):
        print(f"[{shard_index}/{len(shards)}] {shard}", flush=True)
        counts["shards"] += 1
        with tarfile.open(shard, "r") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                destination = _destination(root, member.name)
                if destination is None:
                    continue
                counts["members"] += 1
                if destination.is_file() and destination.stat().st_size == member.size:
                    counts["existing"] += 1
                    continue
                source = archive.extractfile(member)
                if source is None:
                    raise OSError(f"Could not read {member.name} from {shard}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(destination.suffix + ".tmp")
                with source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
                temporary.replace(destination)
                counts["written"] += 1
    return counts


def _verify_materialization(root: Path, rows: list[dict]) -> dict[str, int]:
    missing: list[str] = []
    bad_actions: list[str] = []
    for row in rows:
        key = str(row["sample_key"])
        paths = (
            root / "videos" / "360p" / f"{key}.mp4",
            root / "actions" / f"{key}.actions.bin",
            root / "state" / f"{key}.state.bin",
            root / "events" / f"{key}.events.json",
            root / "metadata" / f"{key}.json",
        )
        missing.extend(str(path) for path in paths if not path.is_file())
        action_path = paths[1]
        expected_action_bytes = int(row["frames"]) * ACTION_RECORD_BYTES
        if action_path.is_file() and action_path.stat().st_size != expected_action_bytes:
            bad_actions.append(f"{action_path}: {action_path.stat().st_size} != {expected_action_bytes}")

    if missing:
        raise FileNotFoundError(f"{len(missing)} materialized payloads are missing; first: {missing[0]}")
    if bad_actions:
        raise ValueError(f"{len(bad_actions)} action files have the wrong size; first: {bad_actions[0]}")
    return {"samples": len(rows), "payload_files": len(rows) * 5}


def _dataset_statistics(rows: list[dict]) -> dict:
    split_rows: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        split_rows[str(row["split"])].append(row)

    split_matches: dict[str, set[str]] = {}
    result: dict[str, dict] = {}
    for split, selected in sorted(split_rows.items()):
        rounds: dict[str, list[dict]] = defaultdict(list)
        for row in selected:
            rounds[str(row["round_id"])].append(row)
        complete = {key: value for key, value in rounds.items() if len(value) == 10}
        if len(complete) != len(rounds):
            raise ValueError(f"{split} contains incomplete ten-POV rounds")
        split_matches[split] = {str(row["match_id"]) for row in selected}
        result[split] = {
            "matches": len(split_matches[split]),
            "rounds": len(rounds),
            "pov_rows": len(selected),
            "pov_hours_full": sum(float(row["frames"]) / float(row["fps"]) for row in selected) / 3600,
            "shared_timeline_hours": sum(
                min(int(row["frames"]) for row in group) / float(group[0]["fps"])
                for group in complete.values()
            )
            / 3600,
        }
        result[split]["aligned_pov_hours"] = result[split]["shared_timeline_hours"] * 10

    split_names = sorted(split_matches)
    overlap = {
        f"{left}:{right}": sorted(split_matches[left] & split_matches[right])
        for index, left in enumerate(split_names)
        for right in split_names[index + 1 :]
    }
    if any(overlap.values()):
        raise ValueError(f"Match leakage across splits: {overlap}")
    return {"splits": result, "match_overlap": overlap}


def _select_table(table, *, map_slug: str, splits: list[str] | None):
    """Select and canonically order a map, optionally restricted to named release splits."""
    mask = pc.equal(  # pyright: ignore[reportAttributeAccessIssue]
        table["map_slug"], map_slug
    )
    normalized_splits = sorted(set(splits or []))
    if normalized_splits:
        split_mask = pc.equal(  # pyright: ignore[reportAttributeAccessIssue]
            table["split"], normalized_splits[0]
        )
        for split in normalized_splits[1:]:
            split_mask = pc.or_(  # pyright: ignore[reportAttributeAccessIssue]
                split_mask,
                pc.equal(  # pyright: ignore[reportAttributeAccessIssue]
                    table["split"], split
                ),
            )
        mask = pc.and_(mask, split_mask)  # pyright: ignore[reportAttributeAccessIssue]
    selected = table.filter(mask)
    return selected.take(
        pc.sort_indices(  # pyright: ignore[reportAttributeAccessIssue]
            selected, sort_keys=[("sample_key", "ascending")]
        )
    )


def _resolve_manifest(data_root: Path, manifest: Path | None) -> Path:
    selected = manifest or (data_root / "manifest.parquet")
    if selected.parent.resolve() != data_root.resolve():
        raise ValueError("--manifest must live directly inside --data-root so payload paths stay bound")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional manifest override, e.g. a frozen confirmatory split within data-root.",
    )
    parser.add_argument("--map-slug", default="dust2")
    parser.add_argument(
        "--splits",
        nargs="+",
        help="Optional release splits to materialize (for example: --splits val).",
    )
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--provenance-output", type=Path, required=True)
    args = parser.parse_args()

    full_manifest = _resolve_manifest(args.data_root, args.manifest)
    table = pq.read_table(full_manifest)
    subset = _select_table(table, map_slug=args.map_slug, splits=args.splits)
    rows = subset.to_pylist()
    if not rows:
        raise ValueError(
            f"No rows with map_slug={args.map_slug!r}, splits={args.splits!r} in {full_manifest}"
        )

    match_ids = sorted({str(row["match_id"]) for row in rows})
    shards = [
        shard
        for match_id in match_ids
        for shard in sorted(
            (args.data_root / "shards-360p" / "v12" / f"match_{match_id}").glob("shard-*.tar")
        )
    ]
    if not shards:
        raise FileNotFoundError("No source WebDataset shards found for the selected matches")

    extraction = _extract(args.data_root, shards) if args.extract else None
    verification = _verify_materialization(args.data_root, rows)
    statistics = _dataset_statistics(rows)

    canonical_rows = "\n".join(
        "\t".join(
            (
                str(row["split"]),
                str(row["match_id"]),
                str(row["round_id"]),
                str(row["pov_idx"]),
                str(row["sample_key"]),
                str(row["frames"]),
            )
        )
        for row in rows
    )
    provenance = {
        "map_slug": args.map_slug,
        "splits_filter": sorted(set(args.splits or [])) or None,
        "action_contract": {
            "source_fps": 32,
            "target_fps": 8,
            "target_frame_offset": 1,
            "source_rows_per_target_step": 4,
            "interval": "observation t uses target-aligned rows t+1..t+4",
            "buttons": "bitwise_or",
            "mouse": "sum(delta_yaw,delta_pitch)",
            "normalization": "none before the native MIRA action encoder",
        },
        "full_manifest": str(full_manifest),
        "full_manifest_sha256": _sha256(full_manifest),
        "selection_sha256": hashlib.sha256(canonical_rows.encode()).hexdigest(),
        "source_shards": len(shards),
        "source_shard_relative_paths": [str(path.relative_to(args.data_root)) for path in shards],
        "verification": verification,
        "statistics": statistics,
        "extraction": extraction,
    }
    args.provenance_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.provenance_output.with_suffix(args.provenance_output.suffix + ".tmp")
    temporary.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.provenance_output)
    print(json.dumps(provenance["statistics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
