from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_counterstrike1k.py"
    spec = importlib.util.spec_from_file_location("prepare_counterstrike1k_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PREPARE = _load_module()


def test_select_table_filters_split_and_orders_sample_keys() -> None:
    table = pa.table(
        {
            "sample_key": ["dust_val_b", "other_val", "dust_train", "dust_val_a"],
            "map_slug": ["dust2", "mirage", "dust2", "dust2"],
            "split": ["val", "val", "train", "val"],
        }
    )

    selected = PREPARE._select_table(table, map_slug="dust2", splits=["val"])

    assert selected["sample_key"].to_pylist() == ["dust_val_a", "dust_val_b"]


def test_select_table_without_split_filter_preserves_all_map_splits() -> None:
    table = pa.table(
        {
            "sample_key": ["dust_val", "other_val", "dust_train"],
            "map_slug": ["dust2", "mirage", "dust2"],
            "split": ["val", "val", "train"],
        }
    )

    selected = PREPARE._select_table(table, map_slug="dust2", splits=None)

    assert selected["sample_key"].to_pylist() == ["dust_train", "dust_val"]


def test_manifest_override_must_remain_bound_to_data_root(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    default = PREPARE._resolve_manifest(data_root, None)
    override = PREPARE._resolve_manifest(data_root, data_root / "confirmatory.parquet")

    assert default == data_root / "manifest.parquet"
    assert override == data_root / "confirmatory.parquet"
    with pytest.raises(ValueError, match="must live directly"):
        PREPARE._resolve_manifest(data_root, tmp_path / "elsewhere.parquet")
