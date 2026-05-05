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
        Let `N_active = sum(sum(s.completion_mask) for s in samples)` --
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
    evaluations at every completion-mask=True position AND the previous
    iteration's V/Q+ at the sampled action (needed for the CFR+
    accumulation term phi).

    Lengths:
        Let `N_active = sum(sum(s.completion_mask) for s in samples)`.
        `v_all`, `v_target_all`, `v_prev_all`, `q_plus_sampled_all`, and
        `q_plus_prev_sampled_all` are all length `N_active`.
        `q_plus_candidates` is shape `[N_active, K]` where K is fixed
        (32 per Phase 5's commitment) and each row's K values match the
        order of `samples[i].completion_top_k_token_ids[k]` (Phase 5).

    Iteration semantics:
        `v_all` and `q_plus_*_all` (without `_prev`) are CURRENT iteration
        network outputs (theta_t, omega_t).
        `v_prev_all` and `q_plus_prev_sampled_all` are PREVIOUS iteration
        network outputs (theta_{t-1}, omega_{t-1}). They participate only
        in the phi (CFR+ accumulation) term of q_plus_target; advantages
        and v_target use only the current iteration's tensors.
        `v_target_all` is the slow-Polyak-copy of V (phi parameter), used
        only for n-step return bootstrapping.
    """

    samples: list[TrainingSample]
    episodic_reward: float
    is_terminal: bool
    # current iteration
    v_all: Tensor                           # [N_active] -- V(o_k; theta_t)
    q_plus_sampled_all: Tensor              # [N_active] -- Q+(o_k, a_k; omega_t)
    q_plus_candidates: Tensor               # [N_active, K] -- Q+(o_k, a; omega_t) for K candidates
    # slow target (Polyak-averaged)
    v_target_all: Tensor                    # [N_active] -- V_target(o_k; phi)
    # previous iteration (for CFR+ accumulation)
    v_prev_all: Tensor                      # [N_active] -- V(o_k; theta_{t-1})
    q_plus_prev_sampled_all: Tensor         # [N_active] -- Q+(o_k, a_k; omega_{t-1})
    gamma: float = 0.99
    n_step: int = 5                         # n-step return horizon


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


def _n_step_return(
    rewards: Tensor,
    v_target_all: Tensor,
    n_step: int,
    gamma: float,
    is_terminal: bool,
) -> Tensor:
    """Compute g_k = sum_{j=k}^{k+n-1} gamma^(j-k) * r_j + gamma^h * V_target(o_{k+h}).

    `h = min(n_step, N - k)` is the effective horizon for position k. When
    `k + n_step <= N - 1`, the bootstrap is `v_target_all[k+n_step]` (an
    interior V_target estimate). When `k + n_step >= N`, the bootstrap is
    the trajectory-boundary value: 0 for terminal rollouts, `v_target_all[-1]`
    for truncated rollouts (no post-trajectory V_target estimate available).
    """
    n = rewards.shape[0]
    dtype = rewards.dtype
    boundary = (
        torch.zeros((), dtype=dtype)
        if is_terminal
        else v_target_all[-1].to(dtype=dtype)
    )
    g = torch.zeros(n, dtype=dtype)
    for k in range(n):
        horizon = min(k + n_step, n)
        # Discounted reward sum from index k to horizon - 1
        for j in range(k, horizon):
            g[k] = g[k] + (gamma ** (j - k)) * rewards[j]
        # Bootstrap at index `horizon`
        bootstrap = v_target_all[horizon] if horizon < n else boundary
        g[k] = g[k] + (gamma ** (horizon - k)) * bootstrap
    return g


def arm_regret_matching_advantage_fn(inputs: ArmAdvantageInputs) -> PerTokenAdvantageOutputs:
    """Token-level CFR+-style regret-matching advantage for ARM.

    For each completion-mask=True position k, with current-iter networks
    (theta_t, omega_t), previous-iter networks (theta_{t-1}, omega_{t-1}),
    and Polyak-target network (phi):

        # 1. n-step return using V_target for stable bootstrapping
        g_k = sum_{k'=k}^{k+n-1} gamma^(k'-k) * r_{k'}
              + gamma^n * V_target(o_{k+n}; phi)

        # 2. Previous iteration's clipped advantage (CFR+ accumulation)
        phi_k = max(0, Q_plus(o_k, a_k; omega_{t-1}) - V(o_k; theta_{t-1}))

        # 3. Regression targets for the Advantage Trainer (Phase 7)
        q_plus_target_k = phi_k + g_k    # historical clip + fresh n-step return
        v_target_k      = g_k

        # 4. Current advantage for regret matching (raw, value-relative)
        A_plus = Q_plus(o_k, a_k; omega_t) - V(o_k; theta_t)

        # 5. Normalize over the K-candidate action set (centered policy advantage)
        q_vals = [max(0, Q_plus(o_k, a; omega_t) - V(o_k; theta_t)) for a in S_k]
        total  = sum(q_vals)
        adv = max(0, A_plus) / total - 1/K       if total >= 1e-8
        adv = 0.0                                 otherwise (cold-start)

    Why phi uses *previous* iteration's networks: that's what makes Q+
    target a CFR+ recurrence (target = historical clipped advantage +
    fresh return), which is what gives ARM its O(sqrt(T)) regret bound.
    Without the phi term Q+ would just be regressing against the n-step
    return and would have no memory of past iterations.

    Why advantage uses *current* iteration's V baseline: A_plus is the
    quantity the policy update consumes. It must be computed against the
    networks the policy is being optimized against (current iter), not
    against historical networks.

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

    v_all = inputs.v_all
    v_target_all = inputs.v_target_all
    v_prev_all = inputs.v_prev_all
    q_plus_sampled = inputs.q_plus_sampled_all
    q_plus_prev_sampled = inputs.q_plus_prev_sampled_all
    q_plus_candidates = inputs.q_plus_candidates

    for name, t in [
        ("v_all", v_all),
        ("v_target_all", v_target_all),
        ("v_prev_all", v_prev_all),
        ("q_plus_sampled_all", q_plus_sampled),
        ("q_plus_prev_sampled_all", q_plus_prev_sampled),
    ]:
        if tuple(t.shape) != (n_active,):
            raise ValueError(f"{name} shape {tuple(t.shape)} != ({n_active},)")
    if q_plus_candidates.dim() != 2 or q_plus_candidates.shape[0] != n_active:
        raise ValueError(
            f"q_plus_candidates shape {tuple(q_plus_candidates.shape)} expected "
            f"({n_active}, K)"
        )
    if inputs.n_step < 1:
        raise ValueError(f"n_step must be >= 1, got {inputs.n_step}")

    K = q_plus_candidates.shape[1]
    gamma = float(inputs.gamma)
    n_step = int(inputs.n_step)
    dtype = v_all.dtype

    # ----- 1. n-step return g_k using V_target for bootstrap -----
    rewards = torch.zeros(n_active, dtype=dtype)
    if inputs.is_terminal:
        rewards[-1] = inputs.episodic_reward
    g = _n_step_return(rewards, v_target_all, n_step, gamma, inputs.is_terminal)

    # ----- 2. CFR+ accumulation term phi_k from previous iteration's networks -----
    phi = torch.clamp(q_plus_prev_sampled - v_prev_all, min=0.0)

    # ----- 3. Regression targets -----
    q_plus_targets_active = phi + g
    v_targets_active = g

    # ----- 4 & 5. Regret matching using current-iter V baseline -----
    a_plus = q_plus_sampled - v_all                                # [N_active]
    q_vals = torch.clamp(q_plus_candidates - v_all.unsqueeze(1), min=0.0)  # [N_active, K]
    total = q_vals.sum(dim=1)                                      # [N_active]

    advantages_active = torch.zeros(n_active, dtype=dtype)
    nondegenerate = total >= 1e-8
    if nondegenerate.any():
        advantages_active[nondegenerate] = (
            torch.clamp(a_plus[nondegenerate], min=0.0) / total[nondegenerate]
            - 1.0 / K
        )

    return PerTokenAdvantageOutputs(
        advantages=_splay_active_to_completion(samples, active_idx_per_sample, advantages_active),
        v_targets=_splay_active_to_completion(samples, active_idx_per_sample, v_targets_active),
        q_plus_targets=_splay_active_to_completion(samples, active_idx_per_sample, q_plus_targets_active),
    )
