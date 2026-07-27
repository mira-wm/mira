from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_script():
    path = Path(__file__).resolve().parents[2] / "scripts" / "diagnose_cs2_multi_action_routing.py"
    spec = importlib.util.spec_from_file_location("diagnose_cs2_multi_action_routing", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summarize_rows_reports_sample_variation() -> None:
    script = _load_script()

    result = script.summarize_rows(
        [
            {"attenuation": 0.1, "delta": 1.0},
            {"attenuation": 0.3, "delta": 3.0},
        ]
    )

    assert result["attenuation"]["mean"] == pytest.approx(0.2)
    assert result["attenuation"]["sample_std"] == pytest.approx(2**0.5 / 10)
    assert result["delta"] == {
        "mean": 2.0,
        "sample_std": pytest.approx(2**0.5),
        "min": 1.0,
        "max": 3.0,
    }


def test_summarize_rows_rejects_empty_input() -> None:
    script = _load_script()

    with pytest.raises(ValueError, match="At least one"):
        script.summarize_rows([])
