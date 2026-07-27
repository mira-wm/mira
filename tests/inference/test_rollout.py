"""Tests for the offline autoregressive rollout helper.

All tests use the stubbed codec (see tests/world_model/conftest) so they run offline without any
checkpoint. The headline guarantee is determinism: a fixed seed + ``noise_level=0.0`` reproduces the
generated latents exactly (this is what the equality harness depends on). The rollout is also
shown to match each model's own ``inference`` latents bit-for-bit, since it mirrors that loop minus
the final decode.
"""

from __future__ import annotations

import torch
from einops import rearrange

from mira.inference.rollout import measure_rollout_speed, rollout
from mira.world_model.config import WorldModelInferenceConfig
from mira.world_model.multi_wrapper_world_model import (
    MultiWrapperWorldModel,
    MultiWrapperWorldModelConfig,
)
from tests.world_model.conftest import StubCodec, tiny_config

from .conftest import build_multi_wrapper, build_world_model, make_batch

# noise_level=0.0 + a fixed schedule is the determinism contract; n_diffusion_steps kept small.
DETERMINISTIC = WorldModelInferenceConfig(n_diffusion_steps=3, noise_level=0.0, schedule_type="linear")


def _build_multi_wrapper_td2(monkeypatch, n_players: int) -> MultiWrapperWorldModel:
    import mira.world_model.latent_world_model as lwm

    monkeypatch.setattr(
        lwm.VideoCodec, "load_from_checkpoint", staticmethod(lambda *args, **kwargs: StubCodec())
    )
    model = MultiWrapperWorldModel(MultiWrapperWorldModelConfig(n_players=n_players, wm_config=tiny_config()))
    return model.eval()


def test_rollout_single_is_deterministic(monkeypatch) -> None:
    model = build_world_model(monkeypatch)
    batch = make_batch(batch_size=1)

    torch.manual_seed(0)
    z1 = rollout(model, batch.clone(), DETERMINISTIC)
    torch.manual_seed(0)
    z2 = rollout(model, batch.clone(), DETERMINISTIC)

    assert torch.equal(z1, z2)
    assert torch.isfinite(z1).all()


def test_rollout_multi_is_deterministic(monkeypatch) -> None:
    n_players = 4
    model = build_multi_wrapper(monkeypatch, n_players=n_players)
    batch = make_batch(batch_size=n_players)  # one match: n_players contiguous rows.

    torch.manual_seed(0)
    z1 = rollout(model, batch.clone(), DETERMINISTIC)
    torch.manual_seed(0)
    z2 = rollout(model, batch.clone(), DETERMINISTIC)

    assert torch.equal(z1, z2)
    # The multiplayer rollout returns the vertically tiled buffer ``(b, t, p*h, w, c)``.
    assert z1.shape[2] == n_players * (z1.shape[2] // n_players)


def test_rollout_different_seed_changes_latents(monkeypatch) -> None:
    """Sanity check that the rollout actually consumes the RNG (a fixed result would also be 'equal')."""
    model = build_world_model(monkeypatch)
    batch = make_batch(batch_size=1)

    torch.manual_seed(0)
    z0 = rollout(model, batch.clone(), DETERMINISTIC)
    torch.manual_seed(1)
    z1 = rollout(model, batch.clone(), DETERMINISTIC)

    assert not torch.equal(z0, z1)


def test_rollout_matches_single_inference_latents(monkeypatch) -> None:
    """The rollout is ``LatentWorldModel.inference`` minus decode, so the latents match exactly."""
    model = build_world_model(monkeypatch)
    batch = make_batch(batch_size=2)

    torch.manual_seed(7)
    z_rollout = rollout(model, batch.clone(), DETERMINISTIC)
    torch.manual_seed(7)
    inference_out = model.inference(batch.clone(), config=DETERMINISTIC, progress_bar=False)

    assert torch.equal(z_rollout, inference_out.z_t)


def test_rollout_matches_multi_inference_latents(monkeypatch) -> None:
    """Same fidelity check for the multiplayer wrapper (its inference returns split-per-player z_t)."""
    n_players = 4
    model = build_multi_wrapper(monkeypatch, n_players=n_players)
    batch = make_batch(batch_size=n_players)

    torch.manual_seed(7)
    z_rollout = rollout(model, batch.clone(), DETERMINISTIC)  # tiled (b, t, p*h, w, c)
    torch.manual_seed(7)
    inference_out = model.inference(batch.clone(), config=DETERMINISTIC, progress_bar=False)

    # inference returns split-per-player latents; re-tile them to compare with the rollout buffer.
    z_inference_tiled = rearrange(inference_out.z_t, "(b p) t h w c -> b t (p h) w c", p=n_players)
    assert torch.equal(z_rollout, z_inference_tiled)


def test_rollout_matches_multi_inference_latents_at_td2(monkeypatch) -> None:
    """Pin the CS2 path where action temporal downsampling is 2 and the offset is non-zero."""
    n_players = 2
    model = _build_multi_wrapper_td2(monkeypatch, n_players=n_players)
    assert model.temporal_downsampling == 2
    assert model.single_world_model.action_temporal_downsampling == 2
    batch = make_batch(batch_size=n_players)

    torch.manual_seed(11)
    z_rollout = rollout(model, batch.clone(), DETERMINISTIC)
    torch.manual_seed(11)
    inference_out = model.inference(batch.clone(), config=DETERMINISTIC, progress_bar=False)

    z_inference_tiled = rearrange(inference_out.z_t, "(b p) t h w c -> b t (p h) w c", p=n_players)
    assert torch.equal(z_rollout, z_inference_tiled)


def test_rollout_n_frames_caps_iterations(monkeypatch) -> None:
    """``n_frames`` caps the number of denoised windows; the seeded prefix matches an uncapped run."""
    model = build_world_model(monkeypatch)
    batch = make_batch(batch_size=1)

    torch.manual_seed(3)
    z_full = rollout(model, batch.clone(), DETERMINISTIC)
    torch.manual_seed(3)
    z_capped = rollout(model, batch.clone(), DETERMINISTIC, n_frames=1)

    window = model.n_context_latents + 1
    # Only the first window is denoised when n_frames=1; that prefix equals the full run's.
    assert torch.equal(z_capped[:, :window], z_full[:, :window])


def test_measure_rollout_speed_returns_finite(monkeypatch) -> None:
    model = build_world_model(monkeypatch)
    batch = make_batch(batch_size=1)

    result = measure_rollout_speed(model, batch, DETERMINISTIC, n_frames=2)
    assert set(result) == {"denoise_ms_per_latent_frame", "denoise_latent_fps"}
    assert result["denoise_ms_per_latent_frame"] > 0
    assert result["denoise_latent_fps"] > 0
