"""The action encoder: embeds keyboard + mouse actions into per-latent-frame conditioning tokens.

The encoder embeds each key with its own learned embedding, embeds the (mostly zero) mouse deltas and
the (often NaN) mouse sensitivity, temporally pools to the latent frame rate, and prepends a learned
initial-action token so the first latent frame has a conditioning token. The NaN-sensitivity path
(``nan_to_num`` + masking to a learned token) is what makes the keyboard-only Rocket League data
work: all-NaN sensitivity is the expected "no mouse" signal.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from mira.ml.init import init_weights
from mira.world_model.actions_config import ActionTensors
from mira.world_model.schedule import symlog_normalize

logger = logging.getLogger(__name__)


class ActionEncoder(torch.nn.Module):
    # Per-player subset-key dropout: when a (player-)row is dropped we may drop only this subset
    # of keys (canonical names) instead of the whole keyboard. Inactive unless
    # `dropout_action_per_player=True` and `key_field_names` are provided.
    DEFAULT_SUBSET_KEYS = ("Q", "E", "Space", "LShiftKey", "LControlKey")

    def __init__(
        self,
        num_key_presses: int,
        dim: int,
        temporal_downsampling: int,
        dropout_prob: float = 0.0,
        max_mouse_movement: float = 2048.0,
        learned_temporal_pool: bool = True,
        dropout_action_per_player: bool = False,
        key_field_names: list[str] | None = None,
        subset_keys: tuple[str, ...] = DEFAULT_SUBSET_KEYS,
        subset_drop_prob: float = 0.5,
    ):
        super().__init__()
        self.dim = dim
        self.temporal_downsampling = temporal_downsampling
        self.dropout_prob = dropout_prob
        self.max_mouse_movement = max_mouse_movement
        self.dropout_action_per_player = dropout_action_per_player
        self.subset_drop_prob = subset_drop_prob

        mouse_dim = dim // 2
        keyboard_dim = dim - mouse_dim
        self.mouse_mlp = nn.Linear(2, mouse_dim)
        self.mouse_sensitivity_mlp = nn.Linear(1, mouse_dim)
        self.mouse_sensitivity_dropout_token = nn.Parameter(0.02 * torch.randn(1, 1, mouse_dim))

        # closest power of 2
        keyboard_split_dim = 2 ** math.floor(math.log2(keyboard_dim / num_key_presses))
        keyboard_remaining_dim = keyboard_dim - num_key_presses * keyboard_split_dim

        if keyboard_remaining_dim > 0:
            self.register_buffer(
                "keyboard_zero_vector",
                torch.zeros((1, 1, keyboard_remaining_dim)),
                persistent=False,
            )
        else:
            self.keyboard_zero_vector = None

        self.keyboard_embedding_dict = nn.ModuleDict()
        for k in range(num_key_presses):
            self.keyboard_embedding_dict[str(k)] = nn.Embedding(2, keyboard_split_dim)

        self.keyboard_mlp = nn.Linear(keyboard_dim, keyboard_dim)

        # Per-player subset-key dropout (additive; default off -> behaviour unchanged). When active,
        # a dropped (player-)row has either all keys or just `subset_keys` replaced by a learned
        # per-key token at the embedding level. These params are warm-start-exempt
        # (MultiWrapperWorldModel._WARMSTART_EXEMPT) so single-player checkpoints still load.
        self.key_dropout_embed = None
        if dropout_action_per_player and dropout_prob > 0:
            subset_key_mask = torch.zeros(num_key_presses, dtype=torch.bool)
            if key_field_names is not None:
                name_to_idx = {name: i for i, name in enumerate(key_field_names)}
                for name in subset_keys:
                    if name in name_to_idx:
                        subset_key_mask[name_to_idx[name]] = True
                missing = [n for n in subset_keys if n not in name_to_idx]
                if missing:
                    logger.warning(
                        "Subset-key dropout: keys %s absent from valid_keys; using present subset %s.",
                        missing,
                        [n for n in subset_keys if n in name_to_idx],
                    )
            else:
                subset_key_mask[:] = True  # no names -> subset dropout degrades to whole-keyboard
            self.register_buffer("subset_key_mask", subset_key_mask, persistent=False)
            self.key_dropout_embed = nn.Parameter(0.02 * torch.randn(num_key_presses, keyboard_split_dim))

        self.learned_temporal_pool = learned_temporal_pool
        if learned_temporal_pool:
            self.mouse_temporal_pool = nn.Linear(temporal_downsampling * mouse_dim, mouse_dim)
            self.keyboard_temporal_pool = nn.Linear(temporal_downsampling * keyboard_dim, keyboard_dim)

        self.joint_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

        self.mouse_dropout_token, self.keyboard_dropout_token = None, None
        if dropout_prob > 0:
            self.mouse_dropout_token = nn.Parameter(0.02 * torch.randn(1, 1, mouse_dim))
            # In per-player mode the whole-keyboard dropout token is replaced by the per-key
            # key_dropout_embed, so keyboard_dropout_token would never be referenced -> an unused
            # parameter that makes DDP abort. Only create it in the legacy whole-keyboard mode.
            if not dropout_action_per_player:
                self.keyboard_dropout_token = nn.Parameter(0.02 * torch.randn(1, 1, keyboard_dim))

        # initial action token
        self.initial_action_token = nn.Parameter(0.02 * torch.randn(1, 1, dim))

        self.apply(init_weights)

    def _sample_drop_masks(self, batch_size: int, device):
        """Per-(player-)row action dropout. Each row is dropped w.p. ``dropout_prob``; a dropped row
        then (w.p. ``subset_drop_prob``) either drops **all** its actions (all keys + mouse) or only
        the **subset** keys (Q/E/Space/Shift/Ctrl) -- keeping WASD *and* the mouse. Since
        MultiWrapper feeds a (b*n_players) batch, per-row == per-player.

        Returns ``(key_drop_mask (b, num_keys) bool, mouse_drop_mask (b,) bool)``."""
        row_drop = torch.rand(batch_size, device=device) < self.dropout_prob
        drop_subset = torch.rand(batch_size, device=device) < self.subset_drop_prob
        assert isinstance(self.subset_key_mask, Tensor)  # registered whenever key_dropout_embed is
        subset = self.subset_key_mask.view(1, -1).expand(batch_size, -1)
        key_mask = torch.where(drop_subset.unsqueeze(-1), subset, torch.ones_like(subset))
        key_drop_mask = key_mask & row_drop.unsqueeze(-1)
        mouse_drop_mask = row_drop & ~drop_subset  # mouse dropped only on the "drop all" branch
        return key_drop_mask, mouse_drop_mask

    def forward(self, actions: ActionTensors, drop_mask: Tensor | None = None) -> Tensor:
        mouse_movements = actions.mouse_movements
        batch_size, n_actions, _ = mouse_movements.shape
        device = mouse_movements.device

        # we record raw mouse deltas Δx and Δy, their unit is in dots through the formula
        # Δx = physical_mouse_displacement_in_inches * DPI , where DPI is mouse-dependent.
        # This translates to an in-game movement (Δx, Δy) * game_mouse_sensitivity
        mouse_movements = mouse_movements.clamp(-self.max_mouse_movement, self.max_mouse_movement)
        delta_xy_normalized = symlog_normalize(mouse_movements, scale=1.0, max_value=self.max_mouse_movement)
        mouse_embed = self.mouse_mlp(delta_xy_normalized)

        # mouse sensitivity
        mouse_sensitivity = rearrange(actions.game_mouse_sensitivity, "b -> b 1 1")
        mouse_sensitivity_mask = torch.isnan(mouse_sensitivity)
        mouse_sensitivity = torch.nan_to_num(mouse_sensitivity, nan=1.0)
        mouse_sensitivity_embed = self.mouse_sensitivity_mlp(mouse_sensitivity)
        mouse_sensitivity_embed = torch.where(
            mouse_sensitivity_mask, self.mouse_sensitivity_dropout_token, mouse_sensitivity_embed
        )

        mouse_embed = mouse_embed + mouse_sensitivity_embed

        key_presses = actions.key_presses

        # Per-(player-)row action dropout. key_drop_mask -> per-key keyboard dropout (below);
        # mouse_drop_mask -> drop the mouse only on the "drop all actions" branch (further below).
        key_drop_mask, mouse_drop_mask = None, None
        if self.key_dropout_embed is not None:
            if self.training:
                key_drop_mask, mouse_drop_mask = self._sample_drop_masks(batch_size, device)
            elif drop_mask is not None:
                # (b, num_keys) per-key control, or (b,) -> drop all that row's actions (keys+mouse).
                if drop_mask.dim() == 2:
                    key_drop_mask = drop_mask
                else:
                    key_drop_mask = drop_mask.unsqueeze(-1).expand(-1, key_presses.shape[-1])
                    mouse_drop_mask = drop_mask

        keyboard_embed_list = []
        for k in range(key_presses.shape[-1]):
            embed_k = self.keyboard_embedding_dict[str(k)](key_presses[:, :, k])
            if key_drop_mask is not None:
                assert self.key_dropout_embed is not None
                token = self.key_dropout_embed[k].to(embed_k.dtype)
                embed_k = torch.where(key_drop_mask[:, k].view(-1, 1, 1), token, embed_k)
            keyboard_embed_list.append(embed_k)

        if self.keyboard_zero_vector is not None:
            keyboard_embed_list.append(self.keyboard_zero_vector.expand(batch_size, n_actions, -1))

        keyboard_embed = torch.cat(keyboard_embed_list, dim=-1)
        keyboard_embed = self.keyboard_mlp(keyboard_embed)

        # Temporally downsample
        mouse_embed = mouse_embed.unflatten(dim=1, sizes=(-1, self.temporal_downsampling))
        keyboard_embed = keyboard_embed.unflatten(dim=1, sizes=(-1, self.temporal_downsampling))
        if self.learned_temporal_pool:
            mouse_embed = self.mouse_temporal_pool(mouse_embed.flatten(2))
            keyboard_embed = self.keyboard_temporal_pool(keyboard_embed.flatten(2))
        else:
            mouse_embed = mouse_embed.mean(dim=2)
            keyboard_embed = keyboard_embed.mean(dim=2)

        if self.dropout_action_per_player:
            # Per-player mode: the keyboard was already dropped per-key at the embedding level
            # above; here we drop the mouse *coordinated* with that decision -- only on the
            # "drop all actions" branch (mouse_drop_mask), so a "subset" drop keeps mouse + WASD.
            if mouse_drop_mask is not None:
                assert self.mouse_dropout_token is not None
                mouse_dropout_token = self.mouse_dropout_token.to(mouse_embed.dtype)
                mouse_embed = torch.where(mouse_drop_mask.view(-1, 1, 1), mouse_dropout_token, mouse_embed)
        elif self.training and self.dropout_prob > 0:
            # Legacy mode: independent whole-mouse and whole-keyboard dropout.
            assert self.mouse_dropout_token is not None and self.keyboard_dropout_token is not None
            drop_mouse_prob = (torch.rand((batch_size,), device=device) < self.dropout_prob).view(-1, 1, 1)
            mouse_dropout_token = self.mouse_dropout_token.to(mouse_embed.dtype)
            mouse_embed = torch.where(drop_mouse_prob, mouse_dropout_token, mouse_embed)

            drop_keyboard_prob = (torch.rand((batch_size,), device=device) < self.dropout_prob).view(-1, 1, 1)
            keyboard_dropout_token = self.keyboard_dropout_token.to(keyboard_embed.dtype)
            keyboard_embed = torch.where(drop_keyboard_prob, keyboard_dropout_token, keyboard_embed)

        actions_embed = torch.cat((mouse_embed, keyboard_embed), dim=-1)
        actions_embed = self.joint_mlp(actions_embed)

        # append initial action token
        initial_action_token = self.initial_action_token.expand(batch_size, -1, -1)
        actions_embed = torch.cat([initial_action_token, actions_embed], dim=1)
        return actions_embed
