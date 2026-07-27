"""Distributed-training setup, single-GPU safe.

``get_distributed_settings`` reports ``(device, rank, world_size)``, defaulting to the single-process
``(0, 0, 1)`` when ``torch.distributed`` is not initialized. ``set_up_distributed`` initializes the
process group from the ``torchrun`` environment when present, and otherwise no-ops so the
trainers and tests run without ``torchrun``.
"""

from __future__ import annotations

import os

import torch
from pydantic import BaseModel


class DistributedSettings(BaseModel):
    device: int
    rank: int
    world_size: int

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def get_distributed_settings() -> DistributedSettings:
    """Current ``(device, rank, world_size)``; ``(0, 0, 1)`` when not running distributed."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        rank = 0
        world_size = 1

    if torch.accelerator.device_count() == 0:
        # CPU-only machine - still useful for tests.
        device = 0
    else:
        assert rank < torch.accelerator.device_count() * world_size
        device = rank % torch.accelerator.device_count()

    return DistributedSettings(device=device, rank=rank, world_size=world_size)


def set_up_distributed() -> DistributedSettings:
    """Initialize the process group from the ``torchrun`` env, or no-op for single-process runs.

    When ``LOCAL_RANK`` is set (i.e. launched under ``torchrun``) and an accelerator is available,
    this selects the local device and initializes the default process group. Without ``LOCAL_RANK``,
    or with no accelerator, it leaves ``torch.distributed`` uninitialized and just reports the
    single-process settings, so scripts run unchanged on a single GPU or on CPU.
    """
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None or torch.accelerator.device_count() == 0:
        return get_distributed_settings()

    torch.accelerator.set_device_index(int(local_rank))
    backend = torch.distributed.get_default_backend_for_device(
        torch.accelerator.current_accelerator(),  # type: ignore[arg-type]
    )
    torch.distributed.init_process_group(backend, device_id=int(local_rank))

    return get_distributed_settings()


def broadcast_main_process_bool(
    value: bool,
    *,
    device: int | str | torch.device,
) -> bool:
    """Broadcast rank 0's boolean decision to every rank.

    Wall-clock termination must be a collective decision. If each rank checks its own clock, one
    rank can leave the training loop while peers enter the next DDP collective and hang forever.
    Rank 0 is the authoritative timer so every process executes the same final optimizer step.
    """
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return value

    rank = torch.distributed.get_rank()
    decision = torch.tensor(
        int(value) if rank == 0 else 0,
        dtype=torch.int32,
        device=device,
    )
    torch.distributed.broadcast(decision, src=0)
    return bool(decision.item())
