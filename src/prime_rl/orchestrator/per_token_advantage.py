"""Per-token advantage computation for PPO and ARM.

This module is the home of the per-token advantage functions that the
Advantage Server's compute layer dispatches to. Phase 2 introduces PPO's
GAE; Phase 3 will add ARM's regret-matching.

The functions are pure: they consume a per-rollout view of the trajectory
plus pre-computed per-token V/V_target tensors and return per-token outputs.
The Advantage Server (Phase 6) and its compute layer (Phase 6.5) own the
V/V_target tensor production; this module is purely arithmetic and is not
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
        - Terminal: V_target(o_{N+1}) is gated to zero by not_done=0 — the
          recursion stops cleanly.
        - Truncated: not_done remains 1 at the final position, and we
          bootstrap using V_target at the final position itself
          (v_target_all[-1]) since no post-trajectory state is available.
          This is a deliberate simplification documented in the phase doc
          §6 (`test_gae_truncation_no_terminal_reward`).

    Output is per-completion-token (NOT per-mask=True): mask=False positions
    receive 0.0 advantage / 0.0 v_target.
    """
    samples = inputs.samples
    if not samples:
        return PerTokenAdvantageOutputs(advantages=[], v_targets=[])

    # Identify mask=True positions per sample. GAE runs only on these; mask=False
    # positions (e.g. injected next-turn prompts inside completion_ids) get zero
    # advantage in the output.
    active_idx_per_sample: list[list[int]] = [
        [j for j, m in enumerate(s.completion_mask) if m] for s in samples
    ]
    n_active_per_sample = [len(a) for a in active_idx_per_sample]
    n_active = sum(n_active_per_sample)

    if n_active == 0:
        # All masked out — no policy-gradient signal. Return zero advantages
        # at every completion_ids position to keep the output shape contract.
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

    # Per-active-position rewards and not_done.
    rewards = torch.zeros(n_active, dtype=dtype)
    not_done = torch.ones(n_active, dtype=dtype)
    if inputs.is_terminal:
        rewards[-1] = inputs.episodic_reward
        not_done[-1] = 0.0

    # Shift-by-one V_target lookup. At the trajectory boundary:
    #   terminal  -> 0 (gated by not_done=0)
    #   truncated -> v_target_all[-1] (bootstrap from the last estimate)
    v_target_next = torch.zeros(n_active, dtype=dtype)
    if n_active > 1:
        v_target_next[:-1] = v_target_all[1:]
    v_target_next[-1] = (
        torch.zeros((), dtype=dtype) if inputs.is_terminal else v_target_all[-1]
    )

    deltas = rewards + gamma * v_target_next * not_done - v_all

    # Backward GAE recursion.
    advantages_active = torch.zeros(n_active, dtype=dtype)
    running = torch.zeros((), dtype=dtype)
    for k in range(n_active - 1, -1, -1):
        running = deltas[k] + gamma * lam * not_done[k] * running
        advantages_active[k] = running

    v_targets_active = advantages_active + v_all

    # Splay active positions back into per-completion_ids lists; mask=False
    # positions get 0.0.
    advantages_per_sample: list[list[float]] = []
    v_targets_per_sample: list[list[float]] = []
    cursor = 0
    for s, active_idx in zip(samples, active_idx_per_sample):
        n = len(active_idx)
        adv_list = [0.0] * len(s.completion_ids)
        vt_list = [0.0] * len(s.completion_ids)
        for local, full in enumerate(active_idx):
            adv_list[full] = float(advantages_active[cursor + local])
            vt_list[full] = float(v_targets_active[cursor + local])
        advantages_per_sample.append(adv_list)
        v_targets_per_sample.append(vt_list)
        cursor += n

    return PerTokenAdvantageOutputs(
        advantages=advantages_per_sample,
        v_targets=v_targets_per_sample,
        q_plus_targets=None,
    )
