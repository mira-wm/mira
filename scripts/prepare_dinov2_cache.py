#!/usr/bin/env python
"""Populate one job-private, read-only-after-warmup DINOv2 Torch Hub cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

from mira.codec.dino import DINO_REPOSITORY

MODEL_NAME = "dinov2_vitb14"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_cache(torch_home: Path) -> dict[str, Any]:
    """Warm source and pretrained weights once before any DDP worker is launched."""
    torch_home = torch_home.resolve()
    torch_home.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(torch_home)

    model = torch.hub.load(
        repo_or_dir=DINO_REPOSITORY[MODEL_NAME],
        model=MODEL_NAME,
        source="github",
        verbose=False,
        pretrained=True,
    )
    if not hasattr(model, "get_intermediate_layers"):
        raise TypeError("DINOv2 hub model does not expose get_intermediate_layers")
    del model

    hub_dir = Path(torch.hub.get_dir()).resolve()
    cached_files = sorted(path for path in hub_dir.rglob("*") if path.is_file())
    if not cached_files:
        raise RuntimeError(f"DINOv2 warmup produced no cache files under {hub_dir}")
    checkpoint_files = sorted((hub_dir / "checkpoints").glob("*"))
    if not checkpoint_files:
        raise RuntimeError(f"DINOv2 pretrained weights are absent under {hub_dir / 'checkpoints'}")

    return {
        "schema": "mira-dinov2-job-cache-v1",
        "status": "ready",
        "model": MODEL_NAME,
        "repository": DINO_REPOSITORY[MODEL_NAME],
        "torch": torch.__version__,
        "torch_home": str(torch_home),
        "hub_dir": str(hub_dir),
        "checkpoint_files": [
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in checkpoint_files
            if path.is_file()
        ],
        "cached_file_count": len(cached_files),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch-home", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = prepare_cache(args.torch_home)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
