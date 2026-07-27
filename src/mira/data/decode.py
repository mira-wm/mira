"""Video decoding via torchcodec, from raw in-tar bytes or a materialized MP4 path.

Decode from mp4 bytes (no temp files); fetch exactly the requested frame indices. torch/torchcodec
are imported lazily so the torch-free parts of the package (schema, events, clip enumeration) work
without them installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def decode_frames(
    video_bytes: bytes | str | Path,
    frame_indices: list[int],
    frame_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Decode the given frame indices into a uint8 tensor of shape (T, C, H, W).

    If `frame_size=(H, W)` is given, frames are resampled to it with antialiased bilinear
    interpolation (typically a downscale). The source is decoded full-res first, then resized.
    """
    from torchcodec.decoders import VideoDecoder  # pyright: ignore[reportPrivateImportUsage]

    decoder = VideoDecoder(video_bytes, device="cpu")
    frames = decoder.get_frames_at(frame_indices).data  # (T, C, H, W) uint8
    if frame_size is not None:
        frames = _resize(frames, frame_size)
    return frames


def _resize(frames: torch.Tensor, frame_size: tuple[int, int]) -> torch.Tensor:
    """Resample (T, C, H, W) uint8 frames to `frame_size=(H, W)`, returning uint8.

    No-op when already at the target size. Antialiased bilinear interpolation runs in float;
    the result is rounded and clamped back into the valid uint8 range (antialiasing can ring
    slightly past [0, 255]).
    """
    import torch
    import torch.nn.functional as F

    if tuple(frames.shape[-2:]) == tuple(frame_size):
        return frames
    resized = F.interpolate(
        frames.float(), size=frame_size, mode="bilinear", align_corners=False, antialias=True
    )
    return resized.round().clamp_(0, 255).to(torch.uint8)
