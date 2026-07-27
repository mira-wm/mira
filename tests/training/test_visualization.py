"""Video helpers: uint8 conversion, grid layout, prediction border, and (gated) ffmpeg writing."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
import torch

from mira.training.visualization import (
    add_prediction_border,
    draw_text_on_first_frame,
    video_to_uint8,
    videos_to_grid,
    write_video_ffmpeg,
)


def test_video_to_uint8_scales_floats() -> None:
    out = video_to_uint8(torch.tensor([[0.0, 0.5, 1.0]]))
    assert out.dtype == torch.uint8
    assert out.tolist() == [[0, 127, 255]]  # 0.5 * 255 = 127.5, truncated to 127
    # uint8 input is returned unchanged.
    u = torch.zeros(2, 2, dtype=torch.uint8)
    assert torch.equal(video_to_uint8(u), u)


def test_videos_to_grid_tiles_batch() -> None:
    # 4 videos of (T=2, C=3, H=8, W=8) -> a 2x2 grid: (T, C, 16, 16).
    video = torch.randint(0, 256, (4, 2, 3, 8, 8), dtype=torch.uint8)
    grid = videos_to_grid(video)
    assert grid.shape == (2, 3, 16, 16)


def test_add_prediction_border_marks_later_frames() -> None:
    video = torch.zeros(1, 4, 3, 10, 10, dtype=torch.uint8)
    out = add_prediction_border(video, context=2, color=(255, 0, 0), border=2)
    # Frames before `context` are untouched; frames from `context` on get a red border.
    assert out[0, 0].sum() == 0
    assert out[0, 2, 0, 0, 0] == 255  # red channel set on the border
    assert out[0, 2, 1, 0, 0] == 0


def test_draw_text_falls_back_when_pillow_lacks_freetype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable_truetype(*_args, **_kwargs):
        raise ImportError("cannot import name '_imagingft' from 'PIL'")

    monkeypatch.setattr("mira.training.visualization.ImageFont.truetype", unavailable_truetype)
    video = torch.zeros((1, 2, 3, 64, 128), dtype=torch.uint8)
    output = draw_text_on_first_frame(video, ["fallback"])

    assert output.shape == video.shape
    assert output[0, 0].count_nonzero() > 0
    assert output[0, 1].count_nonzero() == 0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_write_video_ffmpeg_produces_a_file() -> None:
    video = torch.randint(0, 256, (4, 3, 16, 16), dtype=torch.uint8)
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "clip.mp4"
        write_video_ffmpeg(out, video, fps=10)
        assert out.is_file() and out.stat().st_size > 0


def test_write_video_ffmpeg_falls_back_when_libx264_options_are_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **_kwargs) -> subprocess.CompletedProcess:
        calls.append(cmd)
        if len(calls) == 1:
            return subprocess.CompletedProcess(
                cmd,
                8,
                stderr=b"Unrecognized option 'preset'.\nError splitting the argument list: Option not found",
            )
        return subprocess.CompletedProcess(cmd, 0, stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    write_video_ffmpeg(
        "rollout.mp4",
        torch.zeros((2, 3, 16, 16), dtype=torch.uint8),
    )

    assert len(calls) == 2
    assert "libx264" in calls[0]
    assert "-preset" in calls[0]
    assert "mpeg4" in calls[1]
    assert "-preset" not in calls[1]
    assert "-crf" not in calls[1]
