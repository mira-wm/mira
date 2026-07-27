"""Tests for the multiplayer wrapper: tiling, action combination, warm-start, and the tiled rollout.

All tests use the stubbed codec (see conftest) so they run offline without any checkpoint. The
fidelity invariant under test is that the loader emits ``n_players`` perspectives of a match
contiguously and player-id-ordered as consecutive batch rows, and the wrapper's
``rearrange("(b p) ... -> b p ...")`` lines those rows up with the vertical tiling.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from einops import rearrange

from mira.world_model.actions_config import ActionConfig
from mira.world_model.config import WorldModelInferenceConfig
from mira.world_model.multi_wrapper_world_model import (
    MultiWrapperWorldModel,
    MultiWrapperWorldModelConfig,
)

from .conftest import (
    KEYS,
    LATENT_DIM,
    SPATIAL_DOWNSAMPLING,
    VIDEO_FPS,
    VIDEO_HEIGHT,
    VIDEO_TIMESTEPS,
    VIDEO_WIDTH,
    StubCodec,
    build_world_model,
    make_batch,
    tiny_config,
)


def _td1_codec(*args, **kwargs) -> StubCodec:
    """A :class:`StubCodec` with ``temporal_downsampling == 1``.

    The existing wrapper tests build their inner codec at td==1; the td>1 path is covered by the
    td==2 fixture (the stub codec otherwise defaults to td==2). The single-player baseline that
    warm-starts in still uses td==2.
    """
    codec = StubCodec()
    codec.temporal_downsampling = 1
    return codec


def build_multi_wrapper(
    monkeypatch,
    n_players: int = 2,
    action_routing: str = "global_mean",
    **wm_overrides,
) -> MultiWrapperWorldModel:
    """Build a :class:`MultiWrapperWorldModel` whose inner codec is a td==1 :class:`StubCodec`."""
    import mira.world_model.latent_world_model as lwm

    monkeypatch.setattr(lwm.VideoCodec, "load_from_checkpoint", staticmethod(_td1_codec))
    config = MultiWrapperWorldModelConfig(
        n_players=n_players,
        action_routing=action_routing,
        wm_config=tiny_config(**wm_overrides),
    )
    model = MultiWrapperWorldModel(config)
    model.eval()
    return model


def build_multi_wrapper_td2(
    monkeypatch,
    n_players: int = 2,
    action_routing: str = "global_mean",
    **wm_overrides,
) -> MultiWrapperWorldModel:
    """Build a :class:`MultiWrapperWorldModel` on a td==2 :class:`StubCodec`.

    Unlike :func:`build_multi_wrapper` (which forces td==1 via ``_td1_codec``), this uses the stub's
    default ``TEMPORAL_DOWNSAMPLING == 2`` so the wrapper exercises the temporal_downsampling > 1 path
    that the action-offset fix targets.
    """
    import mira.world_model.latent_world_model as lwm

    monkeypatch.setattr(lwm.VideoCodec, "load_from_checkpoint", staticmethod(lambda *a, **k: StubCodec()))
    config = MultiWrapperWorldModelConfig(
        n_players=n_players,
        action_routing=action_routing,
        wm_config=tiny_config(**wm_overrides),
    )
    model = MultiWrapperWorldModel(config)
    model.eval()
    return model


def test_inner_model_built_at_tiled_height_codec_keeps_single(monkeypatch) -> None:
    """The inner world model sees the tiled height; the codec keeps the single-player height."""
    n_players = 4
    wrapper = build_multi_wrapper(monkeypatch, n_players=n_players)

    assert wrapper.n_players == n_players
    # The inner LatentWorldModel's config height is multiplied by n_players...
    assert wrapper.config.video.height == VIDEO_HEIGHT * n_players
    assert wrapper.world_model.latent_height == (VIDEO_HEIGHT * n_players) // SPATIAL_DOWNSAMPLING
    # ...but the codec resolution is reset to the original single-player height (it runs per player).
    assert wrapper.codec.config.encoder.video.height == VIDEO_HEIGHT


def test_proxies_n_context_latents(monkeypatch) -> None:
    """The wrapper proxies n_context_latents to the inner model (read by the offline metrics)."""
    wrapper = build_multi_wrapper(monkeypatch, n_players=2)
    assert wrapper.n_context_latents == wrapper.single_world_model.n_context_latents


def test_tile_split_round_trip() -> None:
    """4 player rows -> tiled frame -> split is the identity, and player ``pl`` lands in band ``pl``.

    This is the exact ``rearrange`` the wrapper uses; the round-trip and per-band check pin the
    loader's contiguous, player-id-ordered row layout to the vertical tiling.
    """
    p, b, t, h, w, c = 4, 2, 5, 4, 6, LATENT_DIM
    z_flat = torch.randn(b * p, t, h, w, c)

    tiled = rearrange(z_flat, "(b p) t h w c -> b t (p h) w c", p=p)
    assert tiled.shape == (b, t, p * h, w, c)

    split = rearrange(tiled, "b t (p h) w c -> (b p) t h w c", p=p)
    assert torch.equal(split, z_flat)

    # Row g*p + pl (group g, player pl) must occupy height band [pl*h : (pl+1)*h].
    for g in range(b):
        for pl in range(p):
            assert torch.equal(tiled[g, :, pl * h : (pl + 1) * h], z_flat[g * p + pl])


def test_combine_player_actions_shape(monkeypatch) -> None:
    """Legacy global routing collapses the player axis: ``(b*p,t,d) -> (b,t,d)``."""
    n_players = 4
    wrapper = build_multi_wrapper(monkeypatch, n_players=n_players)
    b, t = 2, 7
    d = wrapper.single_world_model.action_encoder.dim

    a_flat = torch.randn(b * n_players, t, d)
    a = wrapper._combine_player_actions(a_flat)

    assert a.shape == (b, t, d)
    assert torch.isfinite(a).all()


def test_spatial_player_actions_align_with_latent_height_bands(monkeypatch) -> None:
    """Spatial routing copies player ``p``'s projected actions only into player ``p``'s POV band."""
    n_players = 4
    wrapper = build_multi_wrapper(
        monkeypatch,
        n_players=n_players,
        action_routing="spatial",
    )
    b, t = 2, 7
    d = wrapper.single_world_model.action_encoder.dim
    a_flat = torch.randn(b * n_players, t, d)

    projected = wrapper._project_player_actions(a_flat)
    routed = wrapper._combine_player_actions(a_flat)

    player_height = wrapper.world_model.latent_height // n_players
    latent_width = wrapper.world_model.latent_width
    assert routed.shape == (
        b,
        t,
        wrapper.world_model.latent_height,
        latent_width,
        d,
    )
    for player in range(n_players):
        band = routed[:, :, player * player_height : (player + 1) * player_height]
        expected = projected[:, player].unsqueeze(2).unsqueeze(3).expand_as(band)
        assert torch.equal(band, expected)


def test_spatial_player_actions_preserve_cross_pov_permutation(monkeypatch) -> None:
    """Permuting player actions changes their routed POV bands instead of collapsing them by mean."""
    n_players = 2
    wrapper = build_multi_wrapper(
        monkeypatch,
        n_players=n_players,
        action_routing="spatial",
    )
    t = 3
    d = wrapper.single_world_model.action_encoder.dim
    a_flat = torch.stack(
        [
            torch.full((t, d), -1.0),
            torch.full((t, d), 1.0),
        ]
    )
    shifted = a_flat.roll(1, dims=0)

    routed = wrapper._combine_player_actions(a_flat)
    routed_shifted = wrapper._combine_player_actions(shifted)

    assert not torch.equal(routed, routed_shifted)
    player_height = wrapper.world_model.latent_height // n_players
    assert not torch.equal(
        routed[:, :, :player_height],
        routed_shifted[:, :, :player_height],
    )


def test_spatial_forward_loss_backpropagates(monkeypatch) -> None:
    """The complete spatial-conditioning path is finite and reaches the player-routing parameters."""
    wrapper = build_multi_wrapper(
        monkeypatch,
        n_players=2,
        action_routing="spatial",
    )
    wrapper.train()
    batch = make_batch(batch_size=4)

    loss = wrapper(batch)["loss_total"]
    loss.backward()

    assert torch.isfinite(loss)
    assert wrapper.player_embedding.grad is not None
    assert wrapper.player_action_projection[1].weight.grad is not None


def test_forward_four_players_finite_loss(monkeypatch) -> None:
    """A 4-player forward on a loader-shaped batch (B*4 contiguous rows) yields a finite loss.

    Exercises the optional action_fps knob: with td==1 and 20fps actions + 10fps frames atd =
    20*1//10 = 2 (2 action steps per video frame). The released default is 1:1 (atd == td == 1).
    """
    n_players = 4
    wrapper = build_multi_wrapper(
        monkeypatch,
        n_players=n_players,
        actions=ActionConfig(valid_keys=KEYS, source_fps=20, target_fps=2 * VIDEO_FPS),
    )
    assert wrapper.single_world_model.temporal_downsampling == 1
    assert wrapper.single_world_model.action_temporal_downsampling == 2

    # 2 groups * 4 players = 8 contiguous rows; 2 action steps per video frame.
    batch = make_batch(batch_size=2 * n_players, n_frames=VIDEO_TIMESTEPS, n_actions=2 * VIDEO_TIMESTEPS)
    outputs = wrapper(batch)

    assert set(outputs) == {"loss_total", "loss_diffusion"}
    for key, value in outputs.items():
        assert value.shape == (), f"{key} should be a scalar"
        assert torch.isfinite(value), f"{key} should be finite, got {value}"


def test_forward_loss_backpropagates(monkeypatch) -> None:
    """Gradients reach the world model, action encoder, and the new player params; not the codec."""
    wrapper = build_multi_wrapper(monkeypatch, n_players=2)
    wrapper.train()
    batch = make_batch(batch_size=4)  # 2 groups * 2 players
    wrapper(batch)["loss_total"].backward()

    assert any(p.grad is not None for p in wrapper.world_model.parameters())
    assert wrapper.player_embedding.grad is not None
    assert all(not p.requires_grad for p in wrapper.codec.parameters())


def test_inference_rollout_tiled_shapes(monkeypatch) -> None:
    """The tiled rollout returns a tiled video ``(B, T, C, P*H, W)`` and split per-player latents."""
    n_players = 2
    wrapper = build_multi_wrapper(monkeypatch, n_players=n_players)
    batch = make_batch(batch_size=2 * n_players)  # 2 groups * 2 players
    config = WorldModelInferenceConfig(n_diffusion_steps=2)

    outputs = wrapper.inference(batch, config=config, progress_bar=False)

    # The multiplayer model runs at td==1, so there are VIDEO_TIMESTEPS latent frames.
    assert wrapper.single_world_model.temporal_downsampling == 1
    n_latent_frames = VIDEO_TIMESTEPS
    h_lat = VIDEO_HEIGHT // SPATIAL_DOWNSAMPLING
    w_lat = VIDEO_WIDTH // SPATIAL_DOWNSAMPLING
    # z_t is split back to per-player latents (b*p rows).
    assert outputs.z_t.shape == (2 * n_players, n_latent_frames, h_lat, w_lat, LATENT_DIM)
    assert torch.isfinite(outputs.z_t).all()

    window_size = wrapper.single_world_model.n_context_latents + 1
    n_generated_latents = (n_latent_frames - window_size) + window_size
    # output_video is tiled along height: P*H (td==1, so one video frame per latent frame).
    assert outputs.output_video.shape == (
        2,
        n_generated_latents,
        3,
        n_players * VIDEO_HEIGHT,
        VIDEO_WIDTH,
    )
    assert torch.isfinite(outputs.output_video).all()


def test_warm_start_from_single_player_checkpoint(monkeypatch) -> None:
    """A single-player checkpoint warm-starts: matched params load, exempt params keep their init.

    The single-player baseline runs at td==2 (atd=2 here); the multiplayer wrapper runs at td==1
    (atd=1). Loading with ``strict=True`` (the default) succeeds, which proves the inner transformer
    params are resolution- and td-independent (RoPE). The params that genuinely differ are exempt:
    ``bos`` (tiled-resolution shape), the action temporal pools (their input dim is ``atd*dim``, so
    td==2 -> td==1 changes the shape), and the new per-player params.
    """
    single = build_world_model(monkeypatch)  # single-player baseline: td==2 stub
    wrapper = build_multi_wrapper(monkeypatch, n_players=2)  # multiplayer: td==1, inner height *2

    assert single.temporal_downsampling == 2 and wrapper.single_world_model.temporal_downsampling == 1

    single_state_dict = single.state_dict()
    assert not any(k.startswith("single_world_model.") for k in single_state_dict)

    wrapper_bos, single_bos = wrapper.single_world_model.bos, single.bos
    assert wrapper_bos is not None and single_bos is not None
    # Differs in shape (tiled vs single resolution), so the checkpoint cannot supply it.
    assert wrapper_bos.shape != single_bos.shape

    # The action temporal pools differ in shape (atd=1 vs atd=2 input), so they cannot load either.
    wrapper_pool = wrapper.single_world_model.action_encoder.mouse_temporal_pool
    assert wrapper_pool.weight.shape != single.action_encoder.mouse_temporal_pool.weight.shape

    # Snapshot the exempt params before the load; they must stay at the wrapper's random init.
    bos_before = wrapper_bos.detach().clone()
    player_embedding_before = wrapper.player_embedding.detach().clone()
    pool_before = wrapper_pool.weight.detach().clone()
    # A matched, resolution-independent transformer param that must take the checkpoint's value.
    matched_key = next(iter(dict(wrapper.single_world_model.world_model.named_parameters())))

    wrapper.load_state_dict(single_state_dict)  # strict=True: must not raise.

    wrapper_wm = dict(wrapper.single_world_model.world_model.named_parameters())
    single_wm = dict(single.world_model.named_parameters())
    assert torch.equal(wrapper_wm[matched_key], single_wm[matched_key]), "matched param did not load"

    # Exempt params kept their init (the checkpoint had no matching entry / a shape mismatch).
    assert torch.equal(wrapper_bos, bos_before)
    assert torch.equal(wrapper.player_embedding, player_embedding_before)
    assert torch.equal(wrapper_pool.weight, pool_before)


def test_forward_action_offset_multi_matches_single_td2(monkeypatch) -> None:
    """At td=2 the multi forward slices actions at ``off = atd-1`` with length
    ``(timesteps//td - 1)*atd``, aligning identically to the single-player forward for the same batch.
    """
    n_players = 2
    wrapper = build_multi_wrapper_td2(monkeypatch, n_players=n_players)
    swm = wrapper.single_world_model
    atd = swm.action_temporal_downsampling
    assert swm.temporal_downsampling == 2 and atd == 2

    batch = make_batch(batch_size=2 * n_players)

    def record_forward_slice(model, run) -> list[float]:
        """Swap ``model``'s action encoder for a recorder, run ``run``, return the slice it saw."""
        recorded: list[list[float]] = []

        class Recorder(torch.nn.Module):
            def forward(self, actions):
                idx = actions.mouse_movements[:, :, 0]
                recorded.append(idx[0].tolist())
                b, n_in = idx.shape
                return torch.zeros(b, n_in // atd + 1, model.config.hidden_dim)

        model.action_encoder = Recorder()  # type: ignore[assignment]
        run()
        return recorded[-1]

    multi_slice = record_forward_slice(swm, lambda: wrapper(batch))

    single = build_world_model(monkeypatch)  # single-player baseline on the same td==2 stub codec.
    assert single.temporal_downsampling == 2 and single.action_temporal_downsampling == atd
    single_slice = record_forward_slice(single, lambda: single(batch))

    off = atd - 1
    expected_len = (VIDEO_TIMESTEPS // 2 - 1) * atd
    assert off == 1 and expected_len == 6
    # Starts at off = atd-1, contiguous, and the expected //td-aware length.
    assert multi_slice == list(range(off, off + expected_len))
    # ...and byte-for-byte the window the single-player forward selects for the same batch.
    assert multi_slice == single_slice


def test_inference_action_offset_multi_matches_single_td2(monkeypatch) -> None:
    """At td=2 the multi rollout selects each action window at offset ``atd-1``, aligning identically
    to the single-player rollout.
    """
    n_players = 2
    wrapper = build_multi_wrapper_td2(monkeypatch, n_players=n_players)
    swm = wrapper.single_world_model
    atd = swm.action_temporal_downsampling
    off = atd - 1
    assert swm.temporal_downsampling == 2 and atd == 2 and off == 1

    def record_inference_starts(model, run) -> list[float]:
        """Swap ``model``'s action encoder for a recorder, run ``run``, return each slice's start."""
        recorded: list[list[float]] = []

        class Recorder(torch.nn.Module):
            def forward(self, actions):
                idx = actions.mouse_movements[:, :, 0]
                recorded.append(idx[0].tolist())
                b, n_in = idx.shape
                return torch.zeros(b, n_in // atd + 1, model.config.hidden_dim)

        model.action_encoder = Recorder()  # type: ignore[assignment]
        run()
        return [slice_idx[0] for slice_idx in recorded]

    batch = make_batch(batch_size=2 * n_players)
    cfg = WorldModelInferenceConfig(n_diffusion_steps=2)
    multi_starts = record_inference_starts(
        swm, lambda: wrapper.inference(batch, config=cfg, progress_bar=False)
    )

    n_latent_frames = VIDEO_TIMESTEPS // 2
    window_size = swm.n_context_latents + 1
    expected = [start * atd + off for start in range(n_latent_frames - window_size + 1)]
    assert multi_starts == expected

    single = build_world_model(monkeypatch)  # single-player rollout is already offset atd-1.
    single_starts = record_inference_starts(
        single, lambda: single.inference(batch, config=cfg, progress_bar=False)
    )
    assert multi_starts == single_starts


def test_td2_wrapper_forward_and_inference_run(monkeypatch) -> None:
    """A td=2 wrapper constructs (no ``assert td==1``) and both forward and inference run."""
    n_players = 2
    wrapper = build_multi_wrapper_td2(monkeypatch, n_players=n_players)
    assert wrapper.single_world_model.temporal_downsampling == 2
    batch = make_batch(batch_size=2 * n_players)

    outputs = wrapper(batch)
    assert set(outputs) == {"loss_total", "loss_diffusion"}
    assert torch.isfinite(outputs["loss_total"])

    config = WorldModelInferenceConfig(n_diffusion_steps=2)
    inference_outputs = wrapper.inference(batch, config=config, progress_bar=False)
    assert torch.isfinite(inference_outputs.z_t).all()
    assert torch.isfinite(inference_outputs.output_video).all()


_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"

# Shrink the inner 1B transformer to a toy size and point at the stub codec so instantiation is cheap.
_TINY_WM_OVERRIDES = [
    "model=multi_wrapper_world_model",
    "dataset.n_players=4",
    "model.architecture.config.wm_config.codec_checkpoint=stub.pth",
    "model.architecture.config.wm_config.latent_mean_std=[0.0,1.0]",
    "model.architecture.config.wm_config.hidden_dim=64",
    "model.architecture.config.wm_config.n_head=4",
    "model.architecture.config.wm_config.n_kv_head=2",
    "model.architecture.config.wm_config.n_layers=2",
    "model.architecture.config.wm_config.time_attention_every=1",
    "model.architecture.config.wm_config.n_context_frames=4",
    "model.architecture.config.wm_config.video.timesteps=8",
    "model.architecture.config.wm_config.video.height=64",
    "model.architecture.config.wm_config.video.width=64",
]


def test_multi_config_composes_and_instantiates_wrapper(monkeypatch) -> None:
    """The ``model=multi_wrapper_world_model`` config composes and instantiates the wrapper."""
    pytest.importorskip("hydra")
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    import mira.world_model.latent_world_model as lwm

    monkeypatch.setattr(lwm.VideoCodec, "load_from_checkpoint", staticmethod(_td1_codec))
    with initialize_config_dir(version_base=None, config_dir=str(_CONFIG_DIR)):
        cfg = compose(config_name="train_world_model", overrides=_TINY_WM_OVERRIDES)

    assert cfg.model.architecture._target_.endswith("MultiWrapperWorldModel")
    assert cfg.model.architecture.config._target_.endswith("MultiWrapperWorldModelConfig")
    assert cfg.model.architecture.config.n_players == 4
    assert cfg.dataset.n_players == 4

    model = instantiate(cfg.model.architecture)
    assert isinstance(model, MultiWrapperWorldModel)
    assert model.n_players == 4
    # The multiplayer model runs at td==1; with actions.target_fps=20 + video.fps=20 => atd=1
    # (one action per video frame, the released default).
    assert model.temporal_downsampling == 1
    assert model.single_world_model.action_temporal_downsampling == 1
    # The codec keeps the single-player height; the inner transformer sees the tiled height.
    assert model.codec.config.encoder.video.height == 64
    assert model.world_model.latent_height == (64 * 4) // SPATIAL_DOWNSAMPLING


def _load_train_script():
    """Import scripts/train_world_model.py as a standalone module (it is not an installed package)."""
    path = Path(__file__).resolve().parents[2] / "scripts" / "train_world_model.py"
    spec = importlib.util.spec_from_file_location("train_world_model_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_n_players_consistency_check_raises_on_mismatch() -> None:
    """The trainer rejects a dataset whose ``n_players`` differs from the model's."""
    from omegaconf import OmegaConf

    train_script = _load_train_script()
    cfg = OmegaConf.create({"dataset": {"n_players": 2}})
    model = SimpleNamespace(n_players=4)

    with pytest.raises(ValueError, match="n_players"):
        # The check fires before any dataloader is built, so the unused args are safe as None.
        train_script._create_dataloaders(cfg, None, model)


def test_training_and_evaluation_group_modes_are_separate(monkeypatch) -> None:
    """A shuffled control must still use synchronized held-out groups for both eval loaders."""
    from omegaconf import OmegaConf

    train_script = _load_train_script()
    calls = []
    monkeypatch.setattr(
        train_script,
        "create_loader",
        lambda **kwargs: calls.append(kwargs) or object(),
    )
    cfg = OmegaConf.create(
        {
            "dataset": {
                "backend": "counterstrike1k",
                "train_index": "/data",
                "test_index": "/data",
                "train_split": "train",
                "test_split": "test",
                "map_slug": "dust2",
                "n_players": 10,
                "group_mode": "shuffled",
                "validation_group_mode": "synchronized",
                "exclude_replays": False,
                "frame_size": [168, 308],
            },
            "dataloader": {
                "num_workers": 4,
                "shuffle_buffer_size": 100,
                "prefetch_factor": 2,
                "persistent_workers": True,
                "pin_memory": True,
            },
            "run": {"batch_size": 1, "seed": 28},
            "validation": {"batch_size": None},
        }
    )
    model = SimpleNamespace(
        n_players=10,
        temporal_downsampling=2,
        config=SimpleNamespace(
            video=SimpleNamespace(fps=8, timesteps=16),
            actions=SimpleNamespace(target_fps=8, valid_keys=["w"]),
            n_context_frames=8,
        ),
        set_inference_context=lambda _: None,
    )
    metrics = SimpleNamespace(
        n_context_frames=8,
        num_unrolled_frames=4,
        eval_temporal_downsampling=None,
        per_device_batch_size=1,
    )

    train_script._create_dataloaders(cfg, metrics, model)

    assert [call["group_mode"] for call in calls] == [
        "shuffled",
        "synchronized",
        "synchronized",
    ]


# -- Multiplayer warm-start wiring -------------------------------------------------------------------

SOURCE_REPO = os.environ.get("RS_SOURCE_REPO")
REF_CKPT = os.environ.get("RS_REF_CKPT")


@pytest.mark.skipif(
    not (SOURCE_REPO and REF_CKPT),
    reason="set RS_SOURCE_REPO and RS_REF_CKPT to run the multi-wrapper warm-start wiring test",
)
def test_multi_wrapper_equality_harness_wires_up() -> None:
    """With a released single-player checkpoint, the multiplayer wrapper warm-starts from it.

    Asserts only that the warm-start mechanism wires up: a single-player state dict loads into the
    4-player wrapper without error. The full output-equality comparison lives in ``tests/equality``.
    """
    from mira.world_model import LatentWorldModel

    assert SOURCE_REPO is not None and REF_CKPT is not None
    assert Path(SOURCE_REPO).exists(), f"RS_SOURCE_REPO does not exist: {SOURCE_REPO}"

    single = LatentWorldModel.load_from_checkpoint(REF_CKPT, device="cpu").eval()
    config = MultiWrapperWorldModelConfig(n_players=4, wm_config=single.config.model_copy(deep=True))
    wrapper = MultiWrapperWorldModel(config)
    # Warm-start from the single-player checkpoint must load without error.
    wrapper.load_state_dict(single.state_dict())
    assert wrapper.n_players == 4
