"""distributed.py is single-process safe: it reports (0, 1) and never touches an uninit process group."""

from __future__ import annotations

import pytest
import torch
import torch.multiprocessing as mp

from mira.training.distributed import (
    broadcast_main_process_bool,
    get_distributed_settings,
    set_up_distributed,
)


def _broadcast_non_main_true_worker(
    rank: int,
    world_size: int,
    init_method: str,
    result_queue,
) -> None:
    torch.distributed.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    try:
        # Rank 1 requests a stop, but rank 0 is authoritative and broadcasts False.
        result = broadcast_main_process_bool(rank == 1, device="cpu")
        result_queue.put((rank, result))
    finally:
        torch.distributed.destroy_process_group()


def test_get_distributed_settings_single_process() -> None:
    settings = get_distributed_settings()
    assert settings.rank == 0
    assert settings.world_size == 1
    assert settings.is_main_process is True


def test_set_up_distributed_no_torchrun_is_a_noop(monkeypatch) -> None:
    # No LOCAL_RANK in the environment => no process group is initialized.
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    settings = set_up_distributed()
    assert settings.world_size == 1
    assert settings.rank == 0
    assert not torch.distributed.is_initialized()


def test_broadcast_main_process_bool_is_noop_without_process_group() -> None:
    assert broadcast_main_process_bool(True, device="cpu") is True
    assert broadcast_main_process_bool(False, device="cpu") is False


def test_broadcast_main_process_bool_uses_only_rank_zero_decision(monkeypatch) -> None:
    observed: dict[str, int] = {}

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)

    def fake_broadcast(decision: torch.Tensor, *, src: int) -> None:
        observed["initial"] = int(decision.item())
        observed["src"] = src
        decision.fill_(1)

    monkeypatch.setattr(torch.distributed, "broadcast", fake_broadcast)

    assert broadcast_main_process_bool(False, device="cpu") is True
    assert observed == {"initial": 0, "src": 0}


@pytest.mark.skipif(
    not torch.distributed.is_gloo_available(),
    reason="Gloo is required for the two-rank CPU collective smoke test",
)
def test_broadcast_main_process_bool_two_rank_gloo(tmp_path) -> None:
    context = mp.get_context("spawn")
    result_queue = context.SimpleQueue()
    init_method = f"file://{tmp_path / 'process-group-init'}"

    mp.spawn(
        _broadcast_non_main_true_worker,
        args=(2, init_method, result_queue),
        nprocs=2,
        join=True,
    )

    results = dict(result_queue.get() for _ in range(2))
    assert results == {0: False, 1: False}
