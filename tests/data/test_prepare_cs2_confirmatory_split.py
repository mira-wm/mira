from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_script():
    path = Path(__file__).resolve().parents[2] / "scripts" / "prepare_cs2_confirmatory_split.py"
    spec = importlib.util.spec_from_file_location("prepare_cs2_confirmatory_split", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows() -> list[dict]:
    rows = []
    for split, matches in (
        ("train", ("m0", "m1", "m2", "m3")),
        ("val", ("mv",)),
        ("test", ("mt",)),
    ):
        for match_id in matches:
            for pov_idx in range(10):
                rows.append(
                    {
                        "map_slug": "dust2",
                        "split": split,
                        "match_id": match_id,
                        "round_id": f"{match_id}-r0",
                        "pov_idx": pov_idx,
                        "sample_key": f"{match_id}-p{pov_idx}",
                        "frames": 320,
                        "fps": 32.0,
                    }
                )
    return rows


def test_selection_is_reproducible_and_hash_only() -> None:
    script = _load_script()
    matches = {"m0", "m1", "m2", "m3"}

    selected = script.select_confirmatory_test_matches(matches, count=2, salt="frozen-v1")
    selected_reordered = script.select_confirmatory_test_matches(
        set(matches),
        count=2,
        salt="frozen-v1",
    )

    assert selected == selected_reordered
    assert len(selected) == 2


def test_relabel_preserves_match_atomicity_and_quarantines_pilot_test() -> None:
    script = _load_script()
    selected_test = {"m1"}

    output = script.relabel_rows(
        _rows(),
        map_slug="dust2",
        confirmatory_test_matches=selected_test,
    )

    by_match = {}
    for row in output:
        by_match.setdefault(row["match_id"], set()).add(row["split"])
    assert by_match["m1"] == {"test"}
    assert by_match["mt"] == {"pilot_test"}
    assert by_match["mv"] == {"val"}
    assert by_match["m0"] == {"train"}
    assert all(len(splits) == 1 for splits in by_match.values())
    assert script._statistics(output)["match_overlap"] == {
        "pilot_test:test": [],
        "pilot_test:train": [],
        "pilot_test:val": [],
        "test:train": [],
        "test:val": [],
        "train:val": [],
    }
    assert script._statistics(output)["splits"]["test"] == {
        "matches": 1,
        "rounds": 1,
        "pov_rows": 10,
        "pov_hours_full": pytest.approx(10 / 360),
        "shared_timeline_hours": pytest.approx(1 / 360),
        "aligned_pov_hours": pytest.approx(10 / 360),
    }


def test_selection_rejects_empty_or_exhaustive_holdout() -> None:
    script = _load_script()
    with pytest.raises(ValueError, match="must be >= 1"):
        script.select_confirmatory_test_matches({"m0", "m1"}, count=0, salt="frozen-v1")
    with pytest.raises(ValueError, match="must be smaller"):
        script.select_confirmatory_test_matches({"m0", "m1"}, count=2, salt="frozen-v1")
