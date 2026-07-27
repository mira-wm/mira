"""Small reproducibility helpers shared by the training entry points."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy and Torch; optionally request deterministic CUDA kernels.

    Deterministic mode is strict: an operation without a deterministic implementation raises
    instead of silently changing the reproducibility contract. The launch environment should also
    set ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` before importing Torch when deterministic CUDA execution
    is requested.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
