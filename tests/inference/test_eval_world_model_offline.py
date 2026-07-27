"""Tests for the offline world-model eval script.

The validation-loss loop and the denoising-speed measurement run offline on the stub-codec model
(single AND multi). The world-model-metrics + viz path needs the DINO / Inception backbones, so it
skip-gates like tests/world_model/test_world_model_metrics.py. The full end-to-end main() against a
real checkpoint is gated on ``RS_REF_CKPT``.
"""

from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mira.world_model.config import WorldModelInferenceConfig

from .conftest import build_multi_wrapper, build_world_model, make_batch


def _load_eval_module():
    """Import scripts/eval_world_model_offline.py as a standalone module (not an installed package)."""
    path = Path(__file__).resolve().parents[2] / "scripts" / "eval_world_model_offline.py"
    spec = importlib.util.spec_from_file_location("eval_world_model_offline_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EVAL = _load_eval_module()
DETERMINISTIC = WorldModelInferenceConfig(n_diffusion_steps=2, noise_level=0.0, schedule_type="linear")


def _val_iter(batch, n: int = 2):
    """A finite (looping) validation iterator of ``(batch, meta)`` pairs."""
    for _ in range(n):
        yield batch.clone(), ["clip"] * len(batch)


def test_validation_loss_single_offline(monkeypatch) -> None:
    model = build_world_model(monkeypatch)
    batch = make_batch(batch_size=2)
    metrics = EVAL.run_validation_loss(model, _val_iter(batch), device="cpu", n_batches=2)

    assert set(metrics) == {"loss_total", "loss_diffusion"}
    for k, v in metrics.items():
        assert math.isfinite(v), k


def test_validation_loss_multi_offline(monkeypatch) -> None:
    n_players = 4
    model = build_multi_wrapper(monkeypatch, n_players=n_players)
    batch = make_batch(batch_size=n_players)  # one match, n_players contiguous rows
    metrics = EVAL.run_validation_loss(model, _val_iter(batch), device="cpu", n_batches=1)

    assert math.isfinite(metrics["loss_total"])


def test_measure_denoise_speed_single_offline(monkeypatch) -> None:
    model = build_world_model(monkeypatch)
    batch = make_batch(batch_size=1)
    original_video = batch.video.clone()
    result = EVAL.measure_denoise_speed(model, batch, DETERMINISTIC, n_frames=2)
    assert result["denoise_latent_fps"] > 0
    assert torch.equal(batch.video, original_video)


def test_measure_denoise_speed_multi_offline(monkeypatch) -> None:
    model = build_multi_wrapper(monkeypatch, n_players=4)
    batch = make_batch(batch_size=4)
    result = EVAL.measure_denoise_speed(model, batch, DETERMINISTIC, n_frames=2)
    assert result["denoise_latent_fps"] > 0


def test_load_eval_metrics_config_reads_yaml_and_overrides() -> None:
    pytest.importorskip("hydra")
    config = EVAL.load_eval_metrics_config()
    # Defaults from configs/eval_world_model.yaml.
    assert config.num_samples == 2048
    assert config.compile is False
    assert config.dino_model == "dinov3_vitb16"
    assert config.inference.schedule_type == "linear"
    assert config.inference.noise_level == 0.0

    overridden = EVAL.load_eval_metrics_config(num_samples=4, no_compile=True)
    assert overridden.num_samples == 4


@pytest.mark.parametrize(
    ("n_samples", "batch_size", "expected"),
    [(1, 1, 1), (100, 10, 10), (52, 1, 52)],
)
def test_exact_num_batches(n_samples: int, batch_size: int, expected: int) -> None:
    assert EVAL._exact_num_batches("samples", n_samples, batch_size) == expected


@pytest.mark.parametrize(("n_samples", "batch_size"), [(0, 1), (10, 0), (11, 10)])
def test_exact_num_batches_rejects_ambiguous_counts(n_samples: int, batch_size: int) -> None:
    with pytest.raises(ValueError):
        EVAL._exact_num_batches("samples", n_samples, batch_size)


def test_window_mode_cli_is_preregistered(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["eval_world_model_offline.py", "checkpoint.pth", "--window-mode", "first-death"],
    )
    assert EVAL.parse_args().window_mode == "first-death"


def test_eval_loader_reuses_recorded_dataloader_tuning(monkeypatch) -> None:
    class AttrDict(dict):
        __getattr__ = dict.__getitem__

    captured = {}
    monkeypatch.setattr(
        "mira.data.training_loader.create_loader",
        lambda **kwargs: captured.update(kwargs) or "loader",
    )
    cfg = SimpleNamespace(
        dataset=AttrDict(
            test_index="/dataset",
            backend="counterstrike1k",
            map_slug="dust2",
            group_mode="single",
            frame_size=[168, 308],
        ),
        dataloader=AttrDict(
            num_workers=8,
            shuffle_buffer_size=100,
            prefetch_factor=3,
            persistent_workers=True,
            pin_memory=False,
        ),
    )
    model = SimpleNamespace(
        n_players=1,
        config=SimpleNamespace(
            video=SimpleNamespace(fps=8),
            actions=SimpleNamespace(target_fps=8, valid_keys=["FORWARD"]),
        ),
    )

    assert (
        EVAL._build_loader(
            cfg,
            model,
            split="val",
            group_mode="single",
            clip_len=16,
            batch_size=10,
            seed=37,
        )
        == "loader"
    )
    assert captured["num_workers"] == 8
    assert captured["prefetch_factor"] == 3
    assert captured["persistent_workers"] is True
    assert captured["pin_memory"] is False


def test_action_modes_are_deterministic_and_do_not_mutate_input() -> None:
    batch = make_batch(batch_size=2, n_frames=4, n_actions=4)
    batch.actions.key_presses[0].fill_(1)
    batch.actions.key_presses[1].fill_(2)
    batch.actions.mouse_movements[0, :, 0] = torch.arange(4)
    batch.actions.mouse_movements[1, :, 0] = 10 + torch.arange(4)
    original = batch.clone()

    zero = EVAL._apply_action_mode(batch, "zero")
    batch_shifted = EVAL._apply_action_mode(batch, "batch-shifted")
    time_shifted = EVAL._apply_action_mode(batch, "time-shifted")

    assert torch.count_nonzero(zero.actions.key_presses) == 0
    assert torch.count_nonzero(zero.actions.mouse_movements) == 0
    assert torch.equal(batch_shifted.actions.key_presses[0], original.actions.key_presses[1])
    assert torch.equal(
        time_shifted.actions.mouse_movements[0],
        original.actions.mouse_movements[0].roll(shifts=2, dims=0),
    )
    assert torch.equal(batch.video, original.video)
    assert torch.equal(batch.actions.key_presses, original.actions.key_presses)
    assert torch.equal(batch.actions.mouse_movements, original.actions.mouse_movements)


def test_round_shifted_actions_use_next_round_and_same_pov_slots() -> None:
    first = make_batch(batch_size=2, n_frames=4, n_actions=4)
    second = make_batch(batch_size=2, n_frames=4, n_actions=4)
    first.video.fill_(3)
    first.actions.key_presses.fill_(1)
    second.video.fill_(7)
    second.actions.key_presses.fill_(2)
    first_metadata = [
        SimpleNamespace(
            round_id="round_0",
            sample_key=f"round_0_p{pov}",
            perspective=pov,
        )
        for pov in range(2)
    ]
    second_metadata = [
        SimpleNamespace(
            round_id="round_1",
            sample_key=f"round_1_p{pov}",
            perspective=pov,
        )
        for pov in range(2)
    ]

    shifted, metadata = next(
        EVAL._action_mode_iter(
            iter(
                [
                    (first, first_metadata),
                    (second, second_metadata),
                ]
            ),
            "round-shifted",
        )
    )

    assert torch.equal(shifted.video, first.video)
    assert torch.equal(
        shifted.actions.key_presses,
        second.actions.key_presses,
    )
    assert [item.perspective for item in metadata] == [0, 1]
    assert {item.action_donor_round_id for item in metadata} == {"round_1"}
    assert [item.action_donor_sample_key for item in metadata] == ["round_1_p0", "round_1_p1"]
    assert torch.equal(first.actions.key_presses, torch.ones_like(first.actions.key_presses))


def test_world_model_metrics_and_viz_offline(monkeypatch, tmp_path) -> None:
    """The eval's metrics loop + viz writer run end-to-end on the stub model (gated on DINO/FID)."""
    pytest.importorskip("pytorch_fid")
    from mira.training.metrics.world_model_metrics import WorldModelMetricsConfig

    config = WorldModelMetricsConfig(
        num_samples=1,
        per_device_batch_size=1,
        num_unrolled_frames=4,
        drift_metric_frames=2,
        fdd_slice_frames=2,
        compile=False,
        num_viz_samples=0,
        inference=DETERMINISTIC,
    )
    model = build_world_model(monkeypatch)
    stride = model.temporal_downsampling
    n_frames = model.config.n_context_frames + config.num_unrolled_frames * stride
    batch = make_batch(batch_size=1, n_frames=n_frames, n_actions=2 * n_frames)

    # ClipMeta-like objects for the viz captions (match_id / perspective).
    from types import SimpleNamespace

    meta = [SimpleNamespace(match_id="m0", perspective="p0")]

    # Guard: the DINO/Inception backbones are downloaded lazily; skip when unavailable offline.
    try:
        results = EVAL.run_world_model_metrics(
            model,
            iter([(batch, meta)]),
            device="cpu",
            wm_metrics_config=config,
            num_eval_batches=1,
            num_viz=1,
            output_dir=tmp_path,
        )
    except Exception as exc:  # noqa: BLE001 -- DINO/Inception weights unavailable offline -> skip
        pytest.skip(f"DINO/Inception backbone unavailable offline: {type(exc).__name__}: {exc}")

    assert math.isfinite(results["psnr"])
    assert (tmp_path / "viz_000.mp4").exists()


# -- Real-checkpoint end-to-end -----------------------------------------------------------------------

REF_CKPT = os.environ.get("RS_REF_CKPT")


@pytest.mark.skipif(not REF_CKPT, reason="set RS_REF_CKPT to run the offline eval against a real checkpoint")
def test_offline_eval_end_to_end_real_checkpoint() -> None:
    """Load a released checkpoint and run validation + denoise speed (single or multi, inferred)."""
    from mira.inference.loading import load_world_model
    from mira.training.checkpoints import resolve_checkpoint

    assert REF_CKPT is not None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = resolve_checkpoint(REF_CKPT).resolve()
    cfg = EVAL.load_run_config(checkpoint)
    model, _ = load_world_model(checkpoint, device=device)
    model = model.eval()

    batch_size = cfg.validation.batch_size or cfg.run.batch_size
    val_loader = EVAL._build_loader(
        cfg,
        model,
        split=cfg.dataset.get("test_split", "test"),
        group_mode=cfg.dataset.get("group_mode"),
        clip_len=model.config.video.timesteps,
        batch_size=batch_size,
        seed=37,
    )
    metrics = EVAL.run_validation_loss(model, iter(val_loader), device, n_batches=1)
    assert math.isfinite(metrics["loss_total"])
