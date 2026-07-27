from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from mira.evaluation.rollout_archive import RolloutArchiveWriter


def test_rollout_archive_is_atomic_and_hashes_complete_arrays(tmp_path) -> None:
    writer = RolloutArchiveWriter(
        tmp_path,
        num_samples=2,
        eval_seeds=[37],
        action_modes=["true", "shuffled"],
        rollout_steps=2,
        height=3,
        width=4,
        num_model_actions=2,
        num_cs2_actions=3,
    )
    for mode_index in range(2):
        writer.write_batch(
            seed_index=0,
            mode_index=mode_index,
            sample_positions=[0, 1],
            predictions=torch.full((2, 2, 3, 3, 4), 0.25 * mode_index),
            ground_truth=torch.zeros(2, 2, 3, 3, 4),
            context_last=torch.zeros(2, 3, 3, 4),
            conditioning_actions_model=torch.full((2, 2, 2), float(mode_index)),
            conditioning_actions_cs2=torch.full((2, 2, 3), float(mode_index)),
            valid_steps=np.ones((2, 2), dtype=np.bool_),
        )

    metadata_path = writer.finalize(contract={"sample_plan_sha256": "a" * 64})

    metadata = json.loads(metadata_path.read_text())
    assert metadata["status"] == "complete"
    assert metadata["contract"]["action_modes"] == ["true", "shuffled"]
    predictions = np.load(tmp_path / "predictions_uint8.npy")
    assert predictions.shape == (1, 2, 2, 2, 3, 4, 3)
    assert predictions[0, 1].mean() == 64
    assert all(len(item["sha256"]) == 64 for item in metadata["artifacts"].values())


def test_rollout_archive_rejects_incomplete_finalize(tmp_path) -> None:
    writer = RolloutArchiveWriter(
        tmp_path,
        num_samples=1,
        eval_seeds=[37],
        action_modes=["true"],
        rollout_steps=1,
        height=2,
        width=2,
        num_model_actions=1,
        num_cs2_actions=1,
    )

    with pytest.raises(ValueError, match="receiver videos"):
        writer.finalize(contract={})
