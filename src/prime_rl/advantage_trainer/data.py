"""Data adapter for the Advantage Trainer (Phase 7b).

Converts an `AdvantageTrainingSample` from the wire format into the tensors
the training step needs. The tensors are aligned to the full input sequence
(prompt + completion) so the value-head outputs from `value_forward` line up
1:1 with the targets.

Per Phase 6's design (and the splay logic in compute.py), `v_targets` and
`q_plus_targets` on `AdvantageTrainingSample` have length `len(completion_ids)`
with zeros at completion-mask=False positions. The loss mask filters those
out. This adapter just shifts them onto the full input by prepending zeros
for prompt positions.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from prime_rl.transport.types import AdvantageTrainingSample


@dataclass
class PreparedAdvantageInputs:
    """Tensors ready to feed value_forward + value_regression_loss_fn."""

    input_ids: Tensor                  # [1, prompt_len + completion_len]
    v_targets: Tensor                  # [prompt_len + completion_len]
    q_plus_targets: Tensor | None      # [prompt_len + completion_len] or None
    loss_mask: Tensor                  # [prompt_len + completion_len], bool


def prepare_advantage_sample(
    sample: AdvantageTrainingSample,
    *,
    device: torch.device | str = "cpu",
) -> PreparedAdvantageInputs:
    """Convert one AdvantageTrainingSample into forward-ready tensors.

    The output tensors are length `len(prompt_ids) + len(completion_ids)`,
    matching what `value_forward` consumes. Prompt positions get target=0
    and loss_mask=False so they don't contribute to the loss.

    Args:
        sample: AdvantageTrainingSample produced by the Advantage Server.
            `v_targets` is required; `q_plus_targets` is optional (None for PPO).
            Both have length `len(completion_ids)` per the Phase 6 contract.
        device: where to allocate the tensors.

    Returns:
        PreparedAdvantageInputs with the four aligned tensors.
    """
    if sample.v_targets is None:
        raise ValueError("AdvantageTrainingSample.v_targets must be populated.")

    prompt_len = len(sample.prompt_ids)
    completion_len = len(sample.completion_ids)

    if len(sample.v_targets) != completion_len:
        raise ValueError(
            f"v_targets length ({len(sample.v_targets)}) doesn't match "
            f"completion_ids length ({completion_len})."
        )
    if sample.q_plus_targets is not None and len(sample.q_plus_targets) != completion_len:
        raise ValueError(
            f"q_plus_targets length ({len(sample.q_plus_targets)}) doesn't match "
            f"completion_ids length ({completion_len})."
        )

    input_ids = torch.tensor(
        [list(sample.prompt_ids) + list(sample.completion_ids)],
        dtype=torch.long,
        device=device,
    )

    # v_targets aligned to full input: zeros at prompt positions, sample's
    # values at completion positions.
    v_targets_full = torch.zeros(prompt_len + completion_len, dtype=torch.float32, device=device)
    v_targets_full[prompt_len:] = torch.tensor(sample.v_targets, dtype=torch.float32, device=device)

    if sample.q_plus_targets is not None:
        q_plus_targets_full = torch.zeros(
            prompt_len + completion_len, dtype=torch.float32, device=device
        )
        q_plus_targets_full[prompt_len:] = torch.tensor(
            sample.q_plus_targets, dtype=torch.float32, device=device
        )
    else:
        q_plus_targets_full = None

    # Loss mask: prompt positions are False; completion positions are
    # `completion_mask` (True for assistant-generated, False for bridge tokens).
    loss_mask_full = torch.zeros(prompt_len + completion_len, dtype=torch.bool, device=device)
    loss_mask_full[prompt_len:] = torch.tensor(
        list(sample.completion_mask), dtype=torch.bool, device=device
    )

    return PreparedAdvantageInputs(
        input_ids=input_ids,
        v_targets=v_targets_full,
        q_plus_targets=q_plus_targets_full,
        loss_mask=loss_mask_full,
    )
