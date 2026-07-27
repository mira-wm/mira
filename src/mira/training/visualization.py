"""Video helpers for logging codec/world-model reconstructions to Weights & Biases.

These encode batches of ``(B, T, C, H, W)`` videos to mp4 (via a system ``ffmpeg``) and arrange them
into a grid for W&B. The pipeline is video-only, so no audio track is muxed.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ConfigDict
from torch import Tensor

if TYPE_CHECKING:
    from mira.world_model.actions_config import ActionTensors


def embeddings_pca_to_rgb(
    embeddings: Tensor,
    output_height: int,
    output_width: int,
    n_fit_frames: int | None = None,
) -> Tensor:
    """Project high-dimensional embeddings to RGB via PCA for visualization.

    Args:
        embeddings: Tensor of shape (B, T, C, H, W).
        output_height: Target height for the output RGB images.
        output_width: Target width for the output RGB images.
        n_fit_frames: Number of frames to use when fitting PCA. None means all frames. Reducing this
            speeds up PCA fitting for high-dimensional features (e.g. raw DINO).

    Returns:
        Tensor of shape (B, T, 3, output_height, output_width), dtype uint8.
    """
    from sklearn.decomposition import PCA  # type: ignore[import-not-found]  # noqa: PLC0415 -- optional dep

    b, t, _c, h, w = embeddings.shape

    projected_batches = []
    for i in range(b):
        feats = embeddings[i]  # T, C, H, W
        feats_flat = rearrange(feats, "t c h w -> (t h w) c").float().cpu().numpy()

        if n_fit_frames is None or n_fit_frames >= t:
            fit_flat = feats_flat
        else:
            # Evenly sample n_fit_frames across the sequence for a representative fit.
            indices = torch.linspace(0, t - 1, n_fit_frames).long()
            fit_flat = rearrange(feats[indices], "t c h w -> (t h w) c").float().cpu().numpy()

        pca = PCA(n_components=3, whiten=True)
        pca.fit(fit_flat)

        projected = torch.from_numpy(pca.transform(feats_flat))
        projected = rearrange(projected, "(t h w) c -> t c h w", t=t, h=h, w=w)
        projected_batches.append(projected)

    projected_images = torch.stack(projected_batches, dim=0)  # B, T, 3, H, W

    # Sigmoid color enhancement to spread the PCA components across the RGB range.
    projected_images = torch.nn.functional.sigmoid(projected_images.mul(2.0))

    # Upsample to output resolution.
    projected_images = rearrange(projected_images, "b t c h w -> (b t) c h w")
    projected_images = F.interpolate(
        projected_images, size=(output_height, output_width), mode="bilinear", align_corners=False
    )
    projected_images = rearrange(projected_images, "(b t) c h w -> b t c h w", b=b, t=t)

    return (255.0 * projected_images).clamp(0, 255).to(torch.uint8)


def video_to_uint8(video: Tensor) -> Tensor:
    """Convert a floating-point video in [0, 1] to uint8 in [0, 255] (no-op if already uint8)."""
    if video.dtype != torch.uint8:
        if not torch.is_floating_point(video):
            raise ValueError(f"Expected uint8 or floating point video tensor, got dtype {video.dtype}")

        # Cast to float32 first so that rounding in reduced-precision dtypes (bfloat16 / float16)
        # does not push values outside [0, 255].
        video = video.float()
        video = torch.clamp(video * 255.0, 0, 255).to(torch.uint8)
    return video


def draw_text_on_first_frame(video: Tensor, texts: list[str]) -> Tensor:
    """Draw text labels on the first frame of each batch item.

    Args:
        video: (B, T, C, H, W) uint8 tensor.
        texts: list of length B with text to draw on each item's first frame.
    """
    video = video.clone()
    for i, text in enumerate(texts):
        frame = video[i, 0]  # (C, H, W)
        img = Image.fromarray(rearrange(frame.cpu().numpy(), "c h w -> h w c"))
        draw = ImageDraw.Draw(img, "RGBA")
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except (ImportError, OSError):
            bitmap_font_loader = getattr(ImageFont, "load_default_imagefont", None)
            if bitmap_font_loader is not None:
                font = bitmap_font_loader()
            else:
                try:
                    font = ImageFont.load_default()
                except (ImportError, OSError):
                    # Text is diagnostic metadata. Preserve the rollout itself on Pillow builds
                    # that expose neither FreeType nor the bundled bitmap font.
                    continue
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        padding = 4
        draw.rectangle([4, 4, 4 + text_w + 2 * padding, 4 + text_h + 2 * padding], fill=(0, 0, 0, 180))
        draw.text((4 + padding, 4 + padding), text, fill="white", font=font)
        video[i, 0] = torch.from_numpy(rearrange(np.array(img)[..., :3], "h w c -> c h w"))
    return video


def write_video_ffmpeg(
    filename: str | Path,
    video_tensor: Tensor,
    fps: float = 10,
    video_codec: str = "libx264",
    crf: int = 23,
    preset: str = "medium",
) -> None:
    """Write a ``(T, C, H, W)`` video to ``filename`` using ffmpeg (video-only, no audio)."""
    video_np = video_to_uint8(video_tensor).permute(0, 2, 3, 1).detach().cpu().numpy()
    _, H, W, _ = video_np.shape

    v_args = [
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{W}x{H}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(fps),
        "-i",
        "-",
    ]
    preferred_out_args = [
        "-vcodec",
        video_codec,
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
    ]

    cmd = ["ffmpeg", "-y", *v_args, *preferred_out_args, str(filename)]
    res = subprocess.run(
        cmd,
        input=video_np.tobytes(),
        capture_output=True,
        timeout=120,
        check=False,
    )

    if not res.returncode:
        return

    preferred_error = res.stderr.decode(errors="replace")
    codec_unavailable = (
        video_codec == "libx264"
        and (
            "Unrecognized option 'preset'" in preferred_error
            or "Option not found" in preferred_error
            or "Unknown encoder 'libx264'" in preferred_error
        )
    )
    if not codec_unavailable:
        raise RuntimeError(f"ffmpeg failed ({res.returncode}):\n{preferred_error}")

    # Some HPC FFmpeg builds intentionally omit libx264 and therefore do not recognize its
    # private ``-preset``/``-crf`` options. MPEG-4 Part 2 is built into FFmpeg, works in an MP4
    # container, and is sufficient for diagnostic rollout traces.
    fallback_out_args = [
        "-vcodec",
        "mpeg4",
        "-q:v",
        "5",
        "-pix_fmt",
        "yuv420p",
    ]
    fallback_cmd = ["ffmpeg", "-y", *v_args, *fallback_out_args, str(filename)]
    fallback_res = subprocess.run(
        fallback_cmd,
        input=video_np.tobytes(),
        capture_output=True,
        timeout=120,
        check=False,
    )
    if fallback_res.returncode:
        fallback_error = fallback_res.stderr.decode(errors="replace")
        raise RuntimeError(
            f"ffmpeg preferred codec failed ({res.returncode}):\n{preferred_error}\n"
            f"ffmpeg MPEG-4 fallback failed ({fallback_res.returncode}):\n{fallback_error}"
        )


class VideoForWandb(BaseModel):
    """A video (with optional caption) to be logged to Weights & Biases."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    video: Tensor
    caption: str | None = None


@contextmanager
def videos_for_wandb(videos: dict[str, VideoForWandb], fps: float = 10) -> Iterator[dict]:
    """Encode video tensors to temp mp4 files and yield ``wandb.Video`` objects.

    Creating ``wandb.Video`` from a video Tensor directly can hang in some cases and crash training.
    Writing the video to a temporary file first and uploading that works reliably.
    """
    import wandb  # noqa: PLC0415 -- optional dep, used only here

    tmp_files: list[tempfile._TemporaryFileWrapper] = []
    try:
        wandb_videos: dict[str, wandb.Video] = {}
        for key, value in videos.items():
            video_tensor = videos_to_grid(value.video)

            f = tempfile.NamedTemporaryFile(suffix=".mp4", delete=True)
            tmp_files.append(f)
            write_video_ffmpeg(f.name, video_tensor.cpu(), fps=fps)
            wandb_videos[key] = wandb.Video(f.name, format="mp4", caption=value.caption)

        yield wandb_videos
    finally:
        for f in tmp_files:
            f.close()


def videos_to_grid(video: Tensor) -> Tensor:
    """Arrange a batch of videos into a square-ish grid (wandb-style layout)."""
    if video.ndim < 4:
        raise ValueError("Video must be at least 4 dimensions: time, channels, height, width")
    if video.ndim == 4:
        video = video.unsqueeze(0)
    b, t, c, h, w = video.shape

    if video.dtype != torch.uint8:
        logging.warning("Converting video data to uint8")
        video = video_to_uint8(video)

    def is_power2(num: int) -> bool:
        return num != 0 and ((num & (num - 1)) == 0)

    # Pad to nearest power of 2, all at once.
    if not is_power2(video.shape[0]):
        len_addition = int(2 ** video.shape[0].bit_length() - video.shape[0])
        video = torch.cat(
            (video, torch.zeros(len_addition, t, c, h, w, dtype=video.dtype, device=video.device)),
            dim=0,
        )

    n_rows = 2 ** ((b.bit_length() - 1) // 2)
    video = rearrange(video, "(n_rows n_cols) t c h w -> t c (n_rows h) (n_cols w)", n_rows=n_rows)

    return video


def visualize_batch(videos: Tensor, actions: ActionTensors, action_temporal_downsampling: int = 1) -> Tensor:
    """Overlay a per-item keyboard HUD onto a batch of videos.

    Args:
        videos: ``(B, T, C, H, W)`` videos (uint8, or floating point in ``[0, 1]``).
        actions: Per-row :class:`ActionTensors`; only ``key_presses`` are drawn (RL is keyboard-only).
        action_temporal_downsampling: Number of action steps per video frame. When ``> 1`` the action
            stream is OR-pooled within each window so it lines up one-to-one with the video frames.

    Returns:
        ``(B, T, C, H, W)`` uint8 videos with the keyboard overlay drawn in the top-right corner.
    """
    if action_temporal_downsampling > 1:
        d = action_temporal_downsampling
        n = actions.key_presses.shape[1] // d * d  # trim to a multiple of d
        actions = actions.slice_time(0, n)
        # OR the key presses across each window so a key held for any sub-step shows as pressed.
        actions.key_presses = actions.key_presses.unflatten(1, (-1, d)).any(dim=2).int()
    assert videos.shape[:2] == actions.key_presses.shape[:2], (
        f"Mismatch: {videos.shape[:2]=}, {actions.key_presses.shape[:2]=}"
    )
    visualized_videos = [
        visualize_sample(videos[i], actions.slice_batch(i, i + 1)) for i in range(videos.shape[0])
    ]
    return torch.stack(visualized_videos, dim=0)


def visualize_sample(video: Tensor, actions: ActionTensors) -> Tensor:
    """Overlay a keyboard HUD onto a single ``(T, C, H, W)`` video (``actions`` has batch size 1)."""
    video = video.clone().to("cpu")

    if video.dtype != torch.uint8:
        video = torch.clamp(video, 0, 1)
        video = (video * 255).to(torch.uint8)

    overlay_height = video.shape[2] // 4
    keyboard_overlay = draw_keyboard_video(
        actions.key_presses[0].to("cpu"),
        valid_keys=actions.config.valid_keys,
        width=int(overlay_height * 2.5),
        height=overlay_height,
        fade_frames=1,
    )
    return overlay_video(video, keyboard_overlay, corner="top-right", xpad=10, ypad=10, opacity=0.5)


def draw_keyboard_video(
    key_presses: Tensor,
    valid_keys: list[str],
    width: int = 800,
    height: int = 300,
    background_color: tuple[int, int, int] = (0, 0, 0),
    key_color: tuple[int, int, int] = (50, 50, 50),
    pressed_color: tuple[int, int, int] = (100, 200, 255),
    outline_width: int = 1,
    fade_frames: int = 5,
) -> Tensor:
    """Render keyboard key presses over time with a fade-out highlight.

    Draws the keyboard layout (:data:`LAYOUT`) and highlights keys when pressed; the highlight fades
    out over ``fade_frames`` after release. Keys not in ``valid_keys`` are skipped, so any vocabulary
    that is a subset of the layout renders in its natural position.

    Args:
        key_presses: ``(T, num_keys)`` tensor of 0/1 press flags, columns aligned to ``valid_keys``.
        valid_keys: Key names for the ``key_presses`` columns.
        width: Frame width in pixels.
        height: Frame height in pixels.
        background_color: RGB background colour.
        key_color: RGB fill for an unpressed key.
        pressed_color: RGB fill for a freshly pressed key.
        outline_width: Key outline width in pixels.
        fade_frames: Number of frames over which the highlight fades after release.

    Returns:
        ``(T, C, H, W)`` uint8 video of the keyboard.
    """
    n_steps = key_presses.shape[0]
    key_presses_np = key_presses.cpu().numpy()
    num_keys = len(valid_keys)

    # Track the last frame each key was pressed for the fade-out effect.
    last_pressed_frame = np.full(num_keys, -fade_frames - 1, dtype=np.int32)

    max_row_width_units = max(sum(width_mult for _, width_mult in row) for row in LAYOUT)
    num_rows = len(LAYOUT)

    padding = 0.05
    gap_ratio = 0.02
    available_width = width * (1 - 2 * padding)
    available_height = height * (1 - 2 * padding)
    key_height = available_height / (num_rows + gap_ratio * (num_rows - 1))
    gap_height = key_height * gap_ratio
    key_width_unit = available_width / (max_row_width_units + gap_ratio * (max_row_width_units - 1))
    gap_width = key_width_unit * gap_ratio

    # Build key positions from the layout.
    key_positions: dict[str, dict[str, float]] = {}
    y_pos = height * padding
    for row in LAYOUT:
        x_pos = width * padding
        for key_name, width_mult in row:
            key_width = key_width_unit * width_mult
            key_positions[key_name] = {"x": x_pos, "y": y_pos, "width": key_width, "height": key_height}
            x_pos += key_width + gap_width
        y_pos += key_height + gap_height

    key_to_idx = {key: idx for idx, key in enumerate(valid_keys)}

    frames = []
    for t in range(n_steps):
        for key_idx in range(num_keys):
            if key_presses_np[t, key_idx] > 0:
                last_pressed_frame[key_idx] = t

        img = Image.new("RGB", (width, height), background_color)
        draw = ImageDraw.Draw(img)

        for key_name, pos in key_positions.items():
            if key_name not in key_to_idx:
                continue
            x, y = int(pos["x"]), int(pos["y"])
            w, h = int(pos["width"]), int(pos["height"])

            frames_since_press = t - last_pressed_frame[key_to_idx[key_name]]
            alpha = 1.0 - max(0, min(1, frames_since_press / fade_frames))
            faded_color = tuple(
                int(bg * (1 - alpha) + pressed * alpha) for bg, pressed in zip(key_color, pressed_color)
            )
            draw.rectangle([x, y, x + w, y + h], fill=faded_color, width=outline_width)

            text = DISPLAY_NAMES.get(key_name, key_name)
            text_bbox = draw.textbbox((0, 0), text)
            text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            draw.text((x + (w - text_w) // 2, y + (h - text_h) // 2), text, fill=(255, 255, 255))

        frames.append(torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1))

    return torch.stack(frames, dim=0)


def overlay_video(
    main_video: Tensor,
    overlay: Tensor,
    xpad: int = 0,
    ypad: int = 0,
    corner: Literal["top-left", "top-right", "bottom-left", "bottom-right"] = "top-left",
    opacity: float = 0.5,
) -> Tensor:
    """Alpha-blend a smaller ``(T, C, h, w)`` overlay onto ``(T, C, H, W)`` at a chosen corner.

    Args:
        main_video: The base video tensor ``(T, C, H, W)``.
        overlay: The video to overlay ``(T, C, h, w)``.
        xpad: Horizontal offset from the chosen corner.
        ypad: Vertical offset from the chosen corner.
        corner: One of ``"top-left"``, ``"top-right"``, ``"bottom-left"``, ``"bottom-right"``.
        opacity: Blending factor (``1.0`` is fully opaque).

    Returns:
        A new tensor with the overlay applied.
    """
    res = main_video.clone()
    _, _, H, W = main_video.shape
    _, _, h, w = overlay.shape

    if corner == "top-left":
        start_x, start_y = xpad, ypad
    elif corner == "top-right":
        start_x, start_y = W - w - xpad, ypad
    elif corner == "bottom-left":
        start_x, start_y = xpad, H - h - ypad
    else:  # bottom-right
        start_x, start_y = W - w - xpad, H - h - ypad

    if opacity >= 1.0:
        res[:, :, start_y : start_y + h, start_x : start_x + w] = overlay
    else:
        region = res[:, :, start_y : start_y + h, start_x : start_x + w]
        blended = (1.0 - opacity) * region.float() + opacity * overlay.float()
        res[:, :, start_y : start_y + h, start_x : start_x + w] = blended.to(res.dtype)

    return res


SPACING_FACTOR = 0.2

# Generic keyboard layout: each row is a list of ``[key_name, width_multiplier]`` pairs. Keys not in
# the active vocabulary are skipped at draw time, so the same layout serves any subset of keys.
LAYOUT = [
    [
        ["D1", 1],
        ["D2", 1],
        ["D3", 1],
        ["D4", 1],
        ["D5", 1],
        ["LButton", 1.5],
        ["RButton", 1.5],
    ],
    [
        ["Q", 1 - SPACING_FACTOR],
        ["W", 1],
        ["E", 1],
        ["R", 1],
        ["T", 1],
    ],
    [
        ["A", 1 + SPACING_FACTOR],
        ["S", 1],
        ["D", 1],
        ["F", 1],
        ["G", 1],
        ["H", 1],
    ],
    [
        ["LShiftKey", 2 + SPACING_FACTOR],
        ["Z", 1],
        ["X", 1],
        ["C", 1],
        ["V", 1],
        ["B", 1],
    ],
    [
        ["LControlKey", 1.25],
        ["LWin", 1.25 + SPACING_FACTOR],
        ["Space", 3],
    ],
]

# Short display labels for keys whose internal name is unwieldy.
DISPLAY_NAMES = {
    "LShiftKey": "Shift",
    "RShiftKey": "Shift",
    "LControlKey": "Ctrl",
    "RControlKey": "Ctrl",
    "LWin": "Win",
    "D0": "0",
    "D1": "1",
    "D2": "2",
    "D3": "3",
    "D4": "4",
    "D5": "5",
    "D6": "6",
    "D7": "7",
    "D8": "8",
    "D9": "9",
    "LButton": "LMB",
    "RButton": "RMB",
}


def add_prediction_border(
    video: Tensor,
    context: int,
    color: tuple = (128, 0, 32),
    border: int = 4,
    clone: bool = True,
) -> Tensor:
    """Draw a coloured border around frames at index ``context`` onward (marks predicted frames)."""
    video = video_to_uint8(video)

    video_output = video.clone() if clone else video

    c = rearrange(torch.tensor(color, device=video_output.device), "c -> c 1 1")
    video_output[:, context:, :, :border, :] = c
    video_output[:, context:, :, -border:, :] = c
    video_output[:, context:, :, :, :border] = c
    video_output[:, context:, :, :, -border:] = c

    return video_output
