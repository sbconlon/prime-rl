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

from prime_rl.advantage_server.action_advantage import tokenize_action
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
# ---------------------------------------------------------------------------
# Phase 10 lever 2: batched data adapter for micro-batched training.
# ---------------------------------------------------------------------------


@dataclass
class BatchedAdvantageInputs:
    """Tensors ready to feed value_forward + value_regression_loss_fn_batched.

    Right-padded across K samples to S_max = max length within the chunk.
    Padded positions have loss_mask=False so they don't contribute to the
    loss; their predictions are still computed (wasted compute) but
    discarded.

    The right-padding scheme keeps the model's attention correctness
    intact without needing a per-row length mask: every valid position
    i < lens[b] attends only to positions j < i < lens[b], which are all
    valid by definition. Padded positions waste compute attending to
    earlier positions (valid or pad) but their outputs go to loss_mask=False
    bins. Strict-causal alone (in the flex kernel for Q+) and standard
    causal (in HF for V/V_target) both naturally restrict valid positions
    to attend only to other valid positions under right-padding.
    """

    input_ids: Tensor                  # [K, S_max] long
    v_targets: Tensor                  # [K, S_max] float
    q_plus_targets: Tensor | None      # [K, S_max] float, or None for PPO
    loss_mask: Tensor                  # [K, S_max] bool
    lens: Tensor                       # [K] long, original per-sample lengths


def prepare_batched_advantage_samples(
    samples: list[AdvantageTrainingSample],
    *,
    device: torch.device | str = "cpu",
    pad_token_id: int = 0,
) -> BatchedAdvantageInputs:
    """Pack K AdvantageTrainingSamples into [K, S_max] tensors via right-padding.

    Same per-sample math as prepare_advantage_sample, just packed. Pads
    input_ids with `pad_token_id`, targets with 0.0, loss_mask with False.

    Args:
        samples: K samples. Empty list -> empty tensors (K=0).
        device: target device for the returned tensors.
        pad_token_id: token ID to use for padded positions. Default 0
            (Qwen3 reserves 0 for an unused/pad slot; embedding lookup is
            well-defined regardless). The contribution to V/Q+ predictions
            at padded positions is irrelevant -- they're masked out by
            loss_mask.

    Returns:
        BatchedAdvantageInputs with [K, S_max]-shaped tensors plus a [K]
        lens tensor capturing each row's true length.
    """
    K = len(samples)
    if K == 0:
        return BatchedAdvantageInputs(
            input_ids=torch.empty((0, 0), dtype=torch.long, device=device),
            v_targets=torch.empty((0, 0), dtype=torch.float32, device=device),
            q_plus_targets=None,
            loss_mask=torch.empty((0, 0), dtype=torch.bool, device=device),
            lens=torch.empty((0,), dtype=torch.long, device=device),
        )

    # Validate per-sample invariants once; same checks as prepare_advantage_sample.
    for s in samples:
        if s.v_targets is None:
            raise ValueError("AdvantageTrainingSample.v_targets must be populated.")
        comp_len = len(s.completion_ids)
        if len(s.v_targets) != comp_len:
            raise ValueError(
                f"v_targets length ({len(s.v_targets)}) doesn't match "
                f"completion_ids length ({comp_len})."
            )
        if s.q_plus_targets is not None and len(s.q_plus_targets) != comp_len:
            raise ValueError(
                f"q_plus_targets length ({len(s.q_plus_targets)}) doesn't match "
                f"completion_ids length ({comp_len})."
            )

    # Determine S_max from the K samples.
    lens_list = [len(s.prompt_ids) + len(s.completion_ids) for s in samples]
    S_max = max(lens_list)

    # Is q_plus tracked for ANY sample? Either all-or-none per batch normally,
    # but defensive: only emit q_plus_targets when at least one sample has them.
    any_q_plus = any(s.q_plus_targets is not None for s in samples)

    input_ids = torch.full(
        (K, S_max), pad_token_id, dtype=torch.long, device=device
    )
    v_targets = torch.zeros((K, S_max), dtype=torch.float32, device=device)
    q_plus_targets = (
        torch.zeros((K, S_max), dtype=torch.float32, device=device) if any_q_plus else None
    )
    loss_mask = torch.zeros((K, S_max), dtype=torch.bool, device=device)
    lens = torch.tensor(lens_list, dtype=torch.long, device=device)

    for k, s in enumerate(samples):
        p_len = len(s.prompt_ids)
        c_len = len(s.completion_ids)
        total_len = p_len + c_len

        input_ids[k, :total_len] = torch.tensor(
            list(s.prompt_ids) + list(s.completion_ids), dtype=torch.long, device=device,
        )
        v_targets[k, p_len:total_len] = torch.tensor(
            s.v_targets, dtype=torch.float32, device=device,
        )
        if any_q_plus and s.q_plus_targets is not None:
            q_plus_targets[k, p_len:total_len] = torch.tensor(  # type: ignore[index]
                s.q_plus_targets, dtype=torch.float32, device=device,
            )
        loss_mask[k, p_len:total_len] = torch.tensor(
            list(s.completion_mask), dtype=torch.bool, device=device,
        )

    return BatchedAdvantageInputs(
        input_ids=input_ids,
        v_targets=v_targets,
        q_plus_targets=q_plus_targets,
        loss_mask=loss_mask,
        lens=lens,
    )


# ---------------------------------------------------------------------------
# Action-level ARM: per-decision-point data prep (Phase 7)
# ---------------------------------------------------------------------------


@dataclass
class PreparedActionAdvantageInputs:
    """Per-chunk tensors for the action-level forward/loss.

    V side (one all-positions forward per sample, then gather at boundaries):
        trajectory_input_ids [K, S_max] right-padded prompt+completion.
        dp_sample_idx [D]    which sample (row) each decision point belongs to.
        dp_v_pos [D]         full-input index prompt_len + response_start - 1 (the
                             o-boundary, the token before the response begins).
    Q+ side (evaluated per decision point -- right-padding obs would break
    forward_q_plus_action's cat, so o/a stay as ragged python lists; batching is
    Phase 8):
        q_plus_obs_ids  list[list[int]]  o_k = prompt + completion[:response_start]
        q_plus_action_ids list[list[int]]  tokenize_action(a*_k) (terminator incl.)
    Targets + per-sample-equal weighting:
        v_targets [D], q_plus_targets [D]
        weights [D]          1/(S * c_s) so per-dp weighted sum == per-sample mean
                             then mean across the S samples that have decision points.
        n_samples K
    """

    trajectory_input_ids: Tensor
    dp_sample_idx: Tensor
    dp_v_pos: Tensor
    q_plus_obs_ids: list[list[int]]
    q_plus_action_ids: list[list[int]]
    v_targets: Tensor
    q_plus_targets: Tensor
    weights: Tensor
    n_samples: int


def prepare_action_advantage_samples(
    samples: list[AdvantageTrainingSample],
    tokenizer,
    *,
    device: torch.device | str = "cpu",
    pad_token_id: int = 0,
) -> PreparedActionAdvantageInputs:
    """Build the action-level forward/loss tensors from decision_point_targets.

    The trainer consumes AdvantageTrainingSample alone (no TrainingSample pairing),
    so o is sliced from the sample's own ids via response_start and a* is tokenized
    from executed_action text (the shared tokenize_action helper, matching the
    AdvServer Q+ builder and the rollout-side pi_hat).
    """
    K = len(samples)
    lens_list = [len(s.prompt_ids) + len(s.completion_ids) for s in samples]
    S_max = max(lens_list) if lens_list else 0

    trajectory_input_ids = torch.full((K, S_max), pad_token_id, dtype=torch.long, device=device)
    for k, s in enumerate(samples):
        ids = list(s.prompt_ids) + list(s.completion_ids)
        trajectory_input_ids[k, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)

    dp_sample_idx: list[int] = []
    dp_v_pos: list[int] = []
    q_plus_obs_ids: list[list[int]] = []
    q_plus_action_ids: list[list[int]] = []
    v_targets: list[float] = []
    q_plus_targets: list[float] = []
    per_sample_counts = [0] * K

    for k, s in enumerate(samples):
        dpts = s.decision_point_targets
        if not dpts:
            continue
        prompt_len = len(s.prompt_ids)
        for dpt in dpts:
            dp_sample_idx.append(k)
            dp_v_pos.append(prompt_len + dpt.response_start - 1)
            q_plus_obs_ids.append(
                list(s.prompt_ids) + list(s.completion_ids[: dpt.response_start])
            )
            q_plus_action_ids.append(tokenize_action(tokenizer, dpt.executed_action))
            v_targets.append(float(dpt.v_target))
            q_plus_targets.append(float(dpt.q_plus_target))
            per_sample_counts[k] += 1

    n_present = sum(1 for c in per_sample_counts if c > 0)
    weights = [
        1.0 / (n_present * per_sample_counts[k]) if n_present > 0 else 0.0
        for k in dp_sample_idx
    ]

    return PreparedActionAdvantageInputs(
        trajectory_input_ids=trajectory_input_ids,
        dp_sample_idx=torch.tensor(dp_sample_idx, dtype=torch.long, device=device),
        dp_v_pos=torch.tensor(dp_v_pos, dtype=torch.long, device=device),
        q_plus_obs_ids=q_plus_obs_ids,
        q_plus_action_ids=q_plus_action_ids,
        v_targets=torch.tensor(v_targets, dtype=torch.float32, device=device),
        q_plus_targets=torch.tensor(q_plus_targets, dtype=torch.float32, device=device),
        weights=torch.tensor(weights, dtype=torch.float32, device=device),
        n_samples=K,
    )
