"""Per-token advantage computation for PPO and ARM.

This module is the home of the per-token advantage functions that the
Advantage Server's compute layer dispatches to. Phase 2 introduces PPO's
GAE; Phase 3 adds ARM's regret-matching as a sibling.

The functions are pure: they consume a per-rollout view of the trajectory
plus pre-computed per-token V/V_target/Q+ tensors and return per-token
outputs. The Advantage Server (Phase 6) and its compute layer (Phase 6.5)
own the tensor production; this module is purely arithmetic and is not
imported by `advantage.py` or wired into `setup_advantage_fn`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

from prime_rl.transport.types import TrainingSample

# NOTE on the sample type: Phase 6 will introduce the
# LLMTrainingSample/AdvantageTrainingSample split. Until then `TrainingSample`
# is the canonical sample type and is what callers pass here.


@dataclass
class PpoAdvantageInputs:
    """Inputs for PPO's GAE advantage computation, per rollout.

    The advantage function operates on the joint trajectory view of one
    rollout's interleaved samples. Per-token V/V_target tensors are
    pre-computed by the Advantage Server's compute layer (Phase 6.5)
    before this function is called.

    Lengths:
        Let `N_active = sum(sum(s.completion_mask) for s in samples)` —
        the total number of completion-mask=True positions across all
        samples. `v_all` and `v_target_all` are length-`N_active` 1D
        tensors carrying the V predictions at exactly those positions,
        in sample-then-position order. Mask=False positions inside
        `completion_ids` (e.g. injected next-turn prompts when a
        rollout was fragmented by `interleave_rollout`) carry no V
        prediction and receive zero advantage in the output.
    """

    samples: list[TrainingSample]
    episodic_reward: float
    is_terminal: bool
    v_all: Tensor
    v_target_all: Tensor
    gamma: float = 0.99
    lam: float = 0.95


@dataclass
class ArmAdvantageInputs:
    """Inputs for ARM's regret-matching advantage computation, per rollout.

    Like PpoAdvantageInputs, the function operates on the joint trajectory
    view of one rollout's interleaved samples. All per-token tensors are
    pre-computed by the Advantage Server's compute layer (Phase 6.5)
    before this function is called -- including the K candidate Q+
    evaluations at every completion-mask=True position.

    Lengths:
        Let `N_active = sum(sum(s.completion_mask) for s in samples)`.
        `v_all`, `v_target_all`, `q_plus_sampled_all` are length
        `N_active`. `q_plus_candidates` is shape `[N_active, K]` where K
        is fixed (32 per Phase 5's commitment) and each row's K values
        match the order of `samples[i].completion_top_k_token_ids[k]`
        (Phase 5). The function does not need the token IDs themselves
        -- it indexes the Q+ values directly.
    """

    samples: list[TrainingSample]
    episodic_reward: float
    is_terminal: bool
    v_all: Tensor
    v_target_all: Tensor
    q_plus_sampled_all: Tensor      # [N_active] -- Q+(o_k, sampled_action_k)
    q_plus_candidates: Tensor       # [N_active, K] -- Q+ at all K candidates
    gamma: float = 0.99


@dataclass
class PerTokenAdvantageOutputs:
    """Outputs of any per-token advantage computation, per rollout.

    Shapes mirror the input sample list at the *completion_ids* granularity
    (NOT mask=True only): `advantages[i]` and `v_targets[i]` have length
    matching `samples[i].completion_ids`. Positions inside `completion_ids`
    where `completion_mask` is False receive 0.0 (no policy-gradient
    contribution; loss_mask handles the trainer-side masking).

    `q_plus_targets` is shared with ARM (Phase 3) and is `None` for PPO.
    """

    advantages: list[list[float]]
    v_targets: list[list[float]]
    q_plus_targets: list[list[float]] | None = None


PpoAdvantageFn = Callable[[PpoAdvantageInputs], PerTokenAdvantageOutputs]
ArmAdvantageFn = Callable[[ArmAdvantageInputs], PerTokenAdvantageOutputs]


# ---------------------------------------------------------------------------
# Internal helpers (shared between PPO and ARM)
# ---------------------------------------------------------------------------


def _active_indices_per_sample(samples: list[TrainingSample]) -> list[list[int]]:
    """Per-sample lists of completion_ids indices where completion_mask is True."""
    return [[j for j, m in enumerate(s.completion_mask) if m] for s in samples]


def _splay_active_to_completion(
    samples: list[TrainingSample],
    active_idx_per_sample: list[list[int]],
    flat_active: Tensor,
) -> list[list[float]]:
    """Splay a length-N_active 1D tensor back into per-sample completion-aligned lists.

    Mask=False positions in `completion_ids` get 0.0; mask=True positions get
    the corresponding value from `flat_active`.
    """
    out: list[list[float]] = []
    cursor = 0
    for s, active_idx in zip(samples, active_idx_per_sample):
        n = len(active_idx)
        per_sample = [0.0] * len(s.completion_ids)
        for local, full in enumerate(active_idx):
            per_sample[full] = float(flat_active[cursor + local])
        out.append(per_sample)
        cursor += n
    return out


# ---------------------------------------------------------------------------
# PPO: GAE
# ---------------------------------------------------------------------------


def ppo_gae_advantage_fn(inputs: PpoAdvantageInputs) -> PerTokenAdvantageOutputs:
    """Token-level GAE for PPO, run jointly over a fragmented rollout.

    Recursion (standard GAE):
        delta_k = r_k + gamma * V_target(o_{k+1}) * not_done_k - V(o_k)
        A_k     = delta_k + gamma * lam * not_done_k * A_{k+1}
        v_target_k = A_k + V(o_k)

    Reward / done derivation from rollout-level metadata:
        r_k    = episodic_reward at the LAST mask=True position of the LAST
                 sample iff is_terminal; 0.0 elsewhere.
        done_k = True at that same position iff is_terminal; False elsewhere.

    Boundary at the trajectory's final position:
        - Terminal: V_target(o_{N+1}) is gated to zero by not_done=0 -- the
          recursion stops cleanly.
        - Truncated: not_done remains 1 at the final position, and we
          bootstrap using V_target at the final position itself
          (v_target_all[-1]) since no post-trajectory state is available.
          This is a deliberate simplification documented in the phase doc
          section 6 (`test_gae_truncation_no_terminal_reward`).

    Output is per-completion-token (NOT per-mask=True): mask=False positions
    receive 0.0 advantage / 0.0 v_target.
    """
    samples = inputs.samples
    if not samples:
        return PerTokenAdvantageOutputs(advantages=[], v_targets=[])

    active_idx_per_sample = _active_indices_per_sample(samples)
    n_active = sum(len(a) for a in active_idx_per_sample)

    if n_active == 0:
        return PerTokenAdvantageOutputs(
            advantages=[[0.0] * len(s.completion_ids) for s in samples],
            v_targets=[[0.0] * len(s.completion_ids) for s in samples],
        )

    v_all = inputs.v_all
    v_target_all = inputs.v_target_all
    if tuple(v_all.shape) != (n_active,):
        raise ValueError(
            f"v_all shape {tuple(v_all.shape)} != ({n_active},) "
            f"(expected sum of completion_mask=True across samples)"
        )
    if tuple(v_target_all.shape) != (n_active,):
        raise ValueError(
            f"v_target_all shape {tuple(v_target_all.shape)} != ({n_active},)"
        )

    gamma = float(inputs.gamma)
    lam = float(inputs.lam)
    dtype = v_all.dtype

    rewards = torch.zeros(n_active, dtype=dtype)
    not_done = torch.ones(n_active, dtype=dtype)
    if inputs.is_terminal:
        rewards[-1] = inputs.episodic_reward
        not_done[-1] = 0.0

    v_target_next = torch.zeros(n_active, dtype=dtype)
    if n_active > 1:
        v_target_next[:-1] = v_target_all[1:]
    v_target_next[-1] = (
        torch.zeros((), dtype=dtype) if inputs.is_terminal else v_target_all[-1]
    )

    deltas = rewards + gamma * v_target_next * not_done - v_all

    advantages_active = torch.zeros(n_active, dtype=dtype)
    running = torch.zeros((), dtype=dtype)
    for k in range(n_active - 1, -1, -1):
        running = deltas[k] + gamma * lam * not_done[k] * running
        advantages_active[k] = running

    v_targets_active = advantages_active + v_all

    return PerTokenAdvantageOutputs(
        advantages=_splay_active_to_completion(samples, active_idx_per_sample, advantages_active),
        v_targets=_splay_active_to_completion(samples, active_idx_per_sample, v_targets_active),
        q_plus_targets=None,
    )


# ---------------------------------------------------------------------------
# ARM: regret matching
# ---------------------------------------------------------------------------


def arm_regret_matching_advantage_fn(inputs: ArmAdvantageInputs) -> PerTokenAdvantageOutputs:
    """Token-level regret-matching advantage for ARM, run per rollout.

    For each completion-mask=True position k with K candidate actions S_k:
        clipped_a    = max(0, Q+(o_k, sampled_action_k))
        clipped_S    = max(0, Q+(o_k, a')) for a' in S_k
        denom        = sum(clipped_S)
        A(o_k, a_k)  = clipped_a / denom - 1/K     if denom > 0
                       0                            otherwise (cold-start branch)

    The cold-start branch fires when every candidate's Q+ is non-positive
    (the "all zero" or "all negative" cases), corresponding to a uniform
    policy. Matches ARM's expected behavior during the first few iterations
    when Q+ has not yet learned.

    The CFR+ recurrence target for the next iteration's Q+ regression is:
        q_plus_target_k = max(0, Q+(o_k, a_k) + A(o_k, a_k))

    The clip-to-zero is what distinguishes CFR+ from vanilla CFR.

    The v_target is the n-step (Monte-Carlo) discounted return bootstrapped
    using V_target at the trajectory boundary. With V_target substituted
    throughout PPO's GAE-derived v_target at lam=1, the formula collapses
    to a simple backward recursion:

        running = 0 if is_terminal else v_target_all[-1]
        for k from N-1 down to 0:
            running = r_k + gamma * running
            v_target[k] = running

    This depends only on `v_target_all` (not `v_all`) -- ARM regresses Q+
    against a self-referential target that includes Q+'s prior output,
    requiring the slow-copy stabilization that V_target provides.

    Output is per-completion-token (NOT per-mask=True): mask=False positions
    receive 0.0 for all three outputs.
    """
    samples = inputs.samples
    if not samples:
        return PerTokenAdvantageOutputs(advantages=[], v_targets=[], q_plus_targets=[])

    active_idx_per_sample = _active_indices_per_sample(samples)
    n_active = sum(len(a) for a in active_idx_per_sample)

    if n_active == 0:
        return PerTokenAdvantageOutputs(
            advantages=[[0.0] * len(s.completion_ids) for s in samples],
            v_targets=[[0.0] * len(s.completion_ids) for s in samples],
            q_plus_targets=[[0.0] * len(s.completion_ids) for s in samples],
        )

    v_target_all = inputs.v_target_all
    q_plus_sampled = inputs.q_plus_sampled_all
    q_plus_candidates = inputs.q_plus_candidates

    if tuple(inputs.v_all.shape) != (n_active,):
        raise ValueError(
            f"v_all shape {tuple(inputs.v_all.shape)} != ({n_active},)"
        )
    if tuple(v_target_all.shape) != (n_active,):
        raise ValueError(
            f"v_target_all shape {tuple(v_target_all.shape)} != ({n_active},)"
        )
    if tuple(q_plus_sampled.shape) != (n_active,):
        raise ValueError(
            f"q_plus_sampled_all shape {tuple(q_plus_sampled.shape)} != ({n_active},)"
        )
    if q_plus_candidates.dim() != 2 or q_plus_candidates.shape[0] != n_active:
        raise ValueError(
            f"q_plus_candidates shape {tuple(q_plus_candidates.shape)} expected "
            f"({n_active}, K)"
        )

    K = q_plus_candidates.shape[1]
    gamma = float(inputs.gamma)
    dtype = inputs.v_all.dtype

    # Regret-matching per active position. Tensorized; the cold-start branch
    # is handled by leaving advantages at zero where denom == 0.
    clipped_candidates = torch.clamp(q_plus_candidates, min=0.0)  # [N_active, K]
    denom = clipped_candidates.sum(dim=1)                          # [N_active]
    clipped_sampled = torch.clamp(q_plus_sampled, min=0.0)         # [N_active]

    advantages_active = torch.zeros(n_active, dtype=dtype)
    nondegenerate = denom > 0
    if nondegenerate.any():
        advantages_active[nondegenerate] = (
            clipped_sampled[nondegenerate] / denom[nondegenerate] - 1.0 / K
        )

    # CFR+ recurrence target.
    q_plus_targets_active = torch.clamp(q_plus_sampled + advantages_active, min=0.0)

    # n-step return / Monte-Carlo v_target bootstrapped via V_target.
    rewards = torch.zeros(n_active, dtype=dtype)
    if inputs.is_terminal:
        rewards[-1] = inputs.episodic_reward

    v_targets_active = torch.zeros(n_active, dtype=dtype)
    running = (
        torch.zeros((), dtype=dtype)
        if inputs.is_terminal
        else v_target_all[-1].to(dtype=dtype)
    )
    for k in range(n_active - 1, -1, -1):
        running = rewards[k] + gamma * running
        v_targets_active[k] = running

    return PerTokenAdvantageOutputs(
        advantages=_splay_active_to_completion(samples, active_idx_per_sample, advantages_active),
        v_targets=_splay_active_to_completion(samples, active_idx_per_sample, v_targets_active),
        q_plus_targets=_splay_active_to_completion(samples, active_idx_per_sample, q_plus_targets_active),
    )
