#!/usr/bin/env python3
"""Freeze a fresh match-atomic Dust2 test split after the G7e architecture diagnosis.

The completed G7e pilot already consumed the release test split. Corrective spatial-action routing
was chosen after that result, so a subsequent synchronized-vs-shuffled experiment must not present
the same test matches as untouched confirmatory evidence.

This script:

* keeps the release validation matches as ``val``;
* relabels the release test matches as ``pilot_test``;
* selects a fixed number of release-train matches by salted SHA-256 order and relabels them ``test``;
* leaves all remaining release-train matches as ``train``; and
* writes a canonical Dust2-only Parquet manifest plus complete provenance.

Selection depends only on the source match identifier and the versioned salt, never on video,
actions, events, duration, or any metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_SALT = "cs1k-dust2-spatial-routing-confirmatory-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(command: str) -> str:
    result = subprocess.run(
        ["git", *command.split()],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def select_confirmatory_test_matches(
    train_match_ids: set[str],
    *,
    count: int,
    salt: str,
) -> list[str]:
    if count < 1:
        raise ValueError("Confirmatory test match count must be >= 1")
    if count >= len(train_match_ids):
        raise ValueError(
            f"Confirmatory test match count {count} must be smaller than "
            f"the {len(train_match_ids)} release-train matches"
        )
    return sorted(
        train_match_ids,
        key=lambda match_id: (
            hashlib.sha256(f"{salt}\0{match_id}".encode()).hexdigest(),
            match_id,
        ),
    )[:count]


def relabel_rows(
    rows: list[dict[str, Any]],
    *,
    map_slug: str,
    confirmatory_test_matches: set[str],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    source_split_by_match: dict[str, str] = {}
    output_split_by_match: dict[str, str] = {}
    for source in rows:
        if str(source["map_slug"]) != map_slug:
            continue
        row = dict(source)
        match_id = str(row["match_id"])
        source_split = str(row["split"])
        previous_source = source_split_by_match.setdefault(match_id, source_split)
        if previous_source != source_split:
            raise ValueError(f"Source match {match_id} crosses release splits")

        if source_split == "train":
            output_split = "test" if match_id in confirmatory_test_matches else "train"
        elif source_split == "val":
            output_split = "val"
        elif source_split == "test":
            output_split = "pilot_test"
        else:
            raise ValueError(f"Unexpected source split {source_split!r} for match {match_id}")
        previous_output = output_split_by_match.setdefault(match_id, output_split)
        if previous_output != output_split:
            raise ValueError(f"Output match {match_id} crosses confirmatory splits")
        row["split"] = output_split
        selected.append(row)

    if not selected:
        raise ValueError(f"No rows found for map_slug={map_slug!r}")
    missing = confirmatory_test_matches - {str(row["match_id"]) for row in selected if row["split"] == "test"}
    if missing:
        raise ValueError(
            f"Selected confirmatory test matches are absent from the map rows: {sorted(missing)}"
        )
    return sorted(selected, key=lambda row: str(row["sample_key"]))


def _statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    split_rows: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        split_rows.setdefault(str(row["split"]), []).append(row)
    result: dict[str, Any] = {}
    match_sets: dict[str, set[str]] = {}
    for split, selected in sorted(split_rows.items()):
        matches = {str(row["match_id"]) for row in selected}
        rounds: dict[str, list[dict[str, Any]]] = {}
        for row in selected:
            rounds.setdefault(str(row["round_id"]), []).append(row)
        match_sets[split] = matches
        shared_timeline_hours = (
            sum(
                min(int(row["frames"]) for row in group) / float(group[0]["fps"]) for group in rounds.values()
            )
            / 3600
        )
        result[split] = {
            "matches": len(matches),
            "rounds": len(rounds),
            "pov_rows": len(selected),
            "pov_hours_full": sum(float(row["frames"]) / float(row["fps"]) for row in selected) / 3600,
            "shared_timeline_hours": shared_timeline_hours,
            "aligned_pov_hours": shared_timeline_hours * 10,
        }
    overlap = {
        f"{left}:{right}": sorted(match_sets[left] & match_sets[right])
        for index, left in enumerate(sorted(match_sets))
        for right in sorted(match_sets)[index + 1 :]
    }
    if any(overlap.values()):
        raise ValueError(f"Match leakage across confirmatory splits: {overlap}")
    round_counts = Counter(str(row["round_id"]) for row in rows)
    incomplete = [round_id for round_id, count in round_counts.items() if count != 10]
    if incomplete:
        raise ValueError(
            f"Confirmatory manifest contains {len(incomplete)} incomplete rounds; first: {incomplete[0]}"
        )
    return {"splits": result, "match_overlap": overlap}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--provenance-output", type=Path, required=True)
    parser.add_argument("--map-slug", default="dust2")
    parser.add_argument("--test-matches", type=int, default=3)
    parser.add_argument("--salt", default=DEFAULT_SALT)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Recompute the frozen split and fail unless the existing manifest/provenance match.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_manifest.resolve() == args.source_manifest.resolve():
        raise ValueError("Refusing to overwrite the source manifest")

    import pyarrow as pa
    import pyarrow.parquet as pq

    source_table = pq.read_table(args.source_manifest)
    source_rows = source_table.to_pylist()
    train_match_ids = {
        str(row["match_id"])
        for row in source_rows
        if str(row["map_slug"]) == args.map_slug and str(row["split"]) == "train"
    }
    selected_test = select_confirmatory_test_matches(
        train_match_ids,
        count=args.test_matches,
        salt=args.salt,
    )
    rows = relabel_rows(
        source_rows,
        map_slug=args.map_slug,
        confirmatory_test_matches=set(selected_test),
    )
    statistics = _statistics(rows)

    canonical = "\n".join(
        "\t".join(
            (
                str(row["split"]),
                str(row["match_id"]),
                str(row["round_id"]),
                str(row["pov_idx"]),
                str(row["sample_key"]),
            )
        )
        for row in rows
    )
    selection_sha256 = hashlib.sha256(canonical.encode()).hexdigest()
    table = pa.Table.from_pylist(rows, schema=source_table.schema)

    if args.verify_only:
        if not args.output_manifest.is_file():
            raise FileNotFoundError(f"Confirmatory manifest is absent: {args.output_manifest}")
        if not args.provenance_output.is_file():
            raise FileNotFoundError(f"Confirmatory provenance is absent: {args.provenance_output}")
        existing_table = pq.read_table(args.output_manifest)
        if not existing_table.equals(table):
            raise ValueError("Existing confirmatory manifest is not the recomputed canonical table")
        existing = json.loads(args.provenance_output.read_text(encoding="utf-8"))
        expected = {
            "schema": "mira-cs2-confirmatory-split-v1",
            "source_manifest_sha256": _sha256(args.source_manifest),
            "output_manifest_sha256": _sha256(args.output_manifest),
            "map_slug": args.map_slug,
            "selection_salt": args.salt,
            "confirmatory_test_matches": selected_test,
            "confirmatory_selection_sha256": selection_sha256,
            "statistics": statistics,
        }
        mismatches = {
            key: {"expected": value, "observed": existing.get(key)}
            for key, value in expected.items()
            if existing.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Confirmatory provenance mismatch: {mismatches}")
        print(
            json.dumps(
                {
                    "verified": True,
                    "manifest": str(args.output_manifest.resolve()),
                    **expected,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = args.output_manifest.with_suffix(args.output_manifest.suffix + ".tmp")
    pq.write_table(table, temporary_manifest)
    temporary_manifest.replace(args.output_manifest)

    payload = {
        "schema": "mira-cs2-confirmatory-split-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": sys.argv,
        "code_commit": _git("rev-parse HEAD"),
        "code_status": _git("status --porcelain=v1"),
        "python": platform.python_version(),
        "pyarrow": pa.__version__,
        "source_manifest": str(args.source_manifest.resolve()),
        "source_manifest_sha256": _sha256(args.source_manifest),
        "output_manifest": str(args.output_manifest.resolve()),
        "output_manifest_sha256": _sha256(args.output_manifest),
        "map_slug": args.map_slug,
        "selection_method": "ascending salted SHA-256 of source release-train match_id",
        "selection_salt": args.salt,
        "confirmatory_test_matches": selected_test,
        "confirmatory_selection_sha256": selection_sha256,
        "statistics": statistics,
    }
    args.provenance_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_provenance = args.provenance_output.with_suffix(args.provenance_output.suffix + ".tmp")
    temporary_provenance.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_provenance.replace(args.provenance_output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
