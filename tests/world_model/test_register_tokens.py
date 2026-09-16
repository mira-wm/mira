"""Temporal and spatial register tokens: output shape, streaming equivalence, and cache layout."""

from __future__ import annotations

import pytest
import torch

from mira.world_model.diffusion_transformer import DiffusionTransformer

from .conftest import LATENT_DIM, SPATIAL_DOWNSAMPLING, TEMPORAL_DOWNSAMPLING, tiny_config


def _build(**overrides) -> DiffusionTransformer:
    torch.manual_seed(0)
    config = tiny_config(use_clean_past=False, **overrides)
    return DiffusionTransformer(config, LATENT_DIM, TEMPORAL_DOWNSAMPLING, SPATIAL_DOWNSAMPLING).eval()


def _inputs(model: DiffusionTransformer, t: int):
    b, h, w = 2, model.latent_height, model.latent_width
    z = torch.randn(b, t, h, w, LATENT_DIM, dtype=torch.float64)
    a = torch.randn(b, t, model.config.hidden_dim, dtype=torch.float64)
    tau = torch.rand(b, t, 1, 1, 1, dtype=torch.float64)
    return z, a, tau


@pytest.mark.parametrize("n_register_tokens", [0, 1, 2])
@pytest.mark.parametrize("spatial_register", [False, True])
def test_register_tokens_streaming_matches_full_forward(
    n_register_tokens: int, spatial_register: bool
) -> None:
    model = _build(n_register_tokens=n_register_tokens, spatial_register=spatial_register).double()
    assert (model.register_tokens is not None) == (n_register_tokens > 0)
    assert (model.spatial_register_token is not None) == spatial_register
    t = 5
    z, a, tau = _inputs(model, t)

    with torch.no_grad():
        full = model(z, a, tau)
        # Register frames and the register row must be removed from the prediction.
        assert full.shape == z.shape

        _, cache = model(z[:, :-1], a[:, :-1], tau[:, :-1], return_kv=True)
        # Temporal registers sit at the front of the cache; the spatial register adds one grid row.
        b, h, w = z.shape[0], model.latent_height, model.latent_width
        assert cache[0][0].shape[:2] == (b * (h + int(spatial_register)) * w, t - 1 + n_register_tokens)

        step = model(z[:, -1:], a[:, -1:], tau[:, -1:], kv_caches=cache)
    torch.testing.assert_close(step, full[:, -1:], rtol=0, atol=1e-9)


def test_spatial_register_token_affects_output() -> None:
    model = _build(spatial_register=True).double()
    z, a, tau = _inputs(model, 3)
    with torch.no_grad():
        before = model(z, a, tau)
        model.spatial_register_token.add_(torch.randn_like(model.spatial_register_token))
        after = model(z, a, tau)
    assert not torch.allclose(before, after)


def test_register_kv_is_input_independent() -> None:
    """Temporal registers only see themselves, so their cached K/V must not depend on the video."""
    model = _build(n_register_tokens=2, spatial_register=True).double()
    with torch.no_grad():
        _, cache_a = model(*_inputs(model, 3), return_kv=True)
        _, cache_b = model(*_inputs(model, 3), return_kv=True)
    for (k_a, v_a), (k_b, v_b) in zip(cache_a, cache_b):
        torch.testing.assert_close(k_a[:, :2], k_b[:, :2])
        torch.testing.assert_close(v_a[:, :2], v_b[:, :2])
