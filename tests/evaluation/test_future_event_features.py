from __future__ import annotations

import pytest
import torch

from mira.evaluation.future_event_features import pool_context_hidden


def test_pool_context_hidden_uses_last_token_and_preserves_player_order() -> None:
    hidden = torch.zeros(2, 3, 8, 2, 4)
    for group in range(2):
        for player in range(4):
            hidden[group, -1, player * 2 : (player + 1) * 2] = 10 * group + player

    pooled = pool_context_hidden(hidden, n_players=4)

    assert pooled.shape == (8, 4)
    assert torch.equal(
        pooled[:, 0],
        torch.tensor([0, 1, 2, 3, 10, 11, 12, 13], dtype=torch.float32),
    )


def test_pool_context_hidden_single_model() -> None:
    hidden = torch.arange(2 * 3 * 2 * 2 * 1, dtype=torch.float32).reshape(2, 3, 2, 2, 1)

    pooled = pool_context_hidden(hidden, n_players=1)

    assert pooled.shape == (2, 1)
    assert torch.allclose(pooled[:, 0], hidden[:, -1, :, :, 0].mean(dim=(1, 2)))


def test_pool_context_hidden_rejects_misaligned_height() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        pool_context_hidden(torch.zeros(1, 2, 7, 1, 3), n_players=4)
