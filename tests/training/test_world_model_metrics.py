"""Offline-safe world-model metric math: SlicedFrechetMetric, the config defaults, and the curve
plots. The full DINO/Inception-driven metrics smoke lives in tests/world_model/.
"""

from __future__ import annotations

import math
from typing import cast

import pytest
import torch

from mira.training.metrics.frechet import SlicedFrechetMetric
from mira.training.metrics.image_metrics import DinoForMetrics, OnlineGaussian
from mira.training.metrics.world_model_metrics import (
    WorldModelMetricsConfig,
    _dino_feature_drift,
    _generated_video_at_latent_rate,
    build_frechet_curve_plots,
)


def test_sliced_frechet_is_finite_and_returns_one_value_per_slice() -> None:
    torch.manual_seed(0)
    dim, num_slices = 16, 3
    metric = SlicedFrechetMetric(dim, num_slices)
    for s in range(num_slices):
        target = torch.randn(64, dim)
        pred = torch.randn(64, dim) + 0.7  # shifted mean -> non-trivial distance
        metric.update(s, target, pred)

    aggregate, curve = metric.compute()
    assert math.isfinite(aggregate) and aggregate > 0
    assert len(curve) == num_slices
    assert all(math.isfinite(v) and v >= 0 for v in curve)


def test_sliced_frechet_near_zero_for_identical_distributions() -> None:
    torch.manual_seed(1)
    metric = SlicedFrechetMetric(dim=16, num_slices=2)
    for s in range(2):
        x = torch.randn(128, 16)
        metric.update(s, x, x)

    aggregate, curve = metric.compute()
    assert aggregate < 1e-3
    assert all(v < 1e-3 for v in curve)


def test_sliced_frechet_reset_clears_state() -> None:
    metric = SlicedFrechetMetric(dim=4, num_slices=1)
    metric.update(0, torch.randn(8, 4), torch.randn(8, 4) + 1.0)
    metric.reset()
    for g in (*metric.target, *metric.pred):
        assert int(cast(OnlineGaussian, g).n.item()) == 0


def test_world_model_metrics_config_defaults() -> None:
    config = WorldModelMetricsConfig(num_unrolled_frames=120, drift_metric_frames=20)
    assert config.fdd_slice_frames == 20
    assert config.eval_temporal_downsampling is None
    assert config.dino_model == "dinov3_vitb16"
    # The inference rollout config defaults are carried through.
    assert config.inference.n_diffusion_steps == 10
    assert config.inference.schedule_type == "linear_quadratic"


def test_generated_reconstruction_region_is_aligned_at_latent_rate() -> None:
    video = torch.arange(10).reshape(1, 10, 1, 1, 1)

    generated = _generated_video_at_latent_rate(video, n_context_frames=4, temporal_stride=2)

    assert generated.flatten().tolist() == [4, 6, 8]


def test_dino_cosine_drift_uses_embedding_channel_axis() -> None:
    # Target is a different positive scale at each spatial position, so every C-vector is
    # collinear (zero channel-wise cosine drift). Treating W as the vector axis, as the old
    # implementation did, incorrectly produces non-zero drift.
    predicted = torch.tensor([[[[[1.0, 3.0]], [[2.0, 4.0]]]]])
    target = torch.tensor([[[[[2.0, 9.0]], [[4.0, 12.0]]]]])

    cosine_drift, l2_drift = _dino_feature_drift(predicted, target)

    assert cosine_drift.item() == pytest.approx(0.0, abs=1e-6)
    assert l2_drift.item() > 0


def test_dino_metrics_public_v2_loading(monkeypatch) -> None:
    class FakeDino(torch.nn.Module):
        def get_intermediate_layers(self, x, **_kwargs):
            return [torch.zeros((len(x), 768, 2, 3))]

    calls: list[dict] = []

    def fake_hub_load(**kwargs):
        calls.append(kwargs)
        return FakeDino()

    monkeypatch.setattr(torch.hub, "load", fake_hub_load)
    model = DinoForMetrics(model_name="dinov2_vitb14")

    assert model.dino_dim == 768
    assert model.patch_size == 14
    assert calls == [
        {
            "repo_or_dir": (
                "facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8"
            ),
            "model": "dinov2_vitb14",
            "source": "github",
            "verbose": False,
            "pretrained": True,
        }
    ]


def test_build_frechet_curve_plots_one_plot_per_curve() -> None:
    pytest.importorskip("plotly")
    pytest.importorskip("wandb")

    curves = {"fdd_at": [3.0, 2.0, 1.0], "fid_at": [0.5, 0.4, 0.3]}
    plots = build_frechet_curve_plots(curves, slice_frames=20)
    assert set(plots) == {"viz/fdd_at", "viz/fid_at"}
