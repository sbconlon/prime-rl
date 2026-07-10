"""Warm-start target assembly (Phase 1).

Converts a rollout's TrainingSamples (already carrying decision_points) plus the
episode's terminal reward into warm-start AdvantageTrainingSamples whose
decision_point_targets carry the Monte-Carlo return as BOTH the V and Q+ target
(v_target = q_plus_target = g_k). Reuses n_step_return_action_level for the return
and the existing wire types; the only new logic is the assembly.
"""

from __future__ import annotations

from prime_rl.advantage_server.action_advantage import n_step_return_action_level
from prime_rl.transport.types import (
    AdvantageTrainingSample,
    DecisionPointTarget,
    TrainingSample,
)


def build_warmstart_samples(
    samples: list[TrainingSample],
    episodic_reward: float,
    is_terminal: bool,
    *,
    gamma: float = 1.0,
    n_step: int | None = None,
) -> list[AdvantageTrainingSample]:
    """One rollout's TrainingSamples -> warm-start AdvantageTrainingSamples.

    `samples` is the full ordered list interleave_rollout produced for ONE rollout
    (len > 1 when truncation broke the extension property). Every sample must carry
    decision_points. The Monte-Carlo return g_k is computed over the JOINT episode
    decision-point sequence, so the terminal reward propagates across sample splits;
    it is assigned to both heads (v_target = q_plus_target = g_k). The default
    (gamma=1, n_step=None) is full MC: g_k = the terminal reward at every decision
    point.
    """
    for s in samples:
        if s.decision_points is None:
            raise ValueError(
                "build_warmstart_samples requires action-level samples; found a "
                "TrainingSample with decision_points=None."
            )

    # Owning-sample index for each decision point, in joint episode order.
    dp_owner = [
        sample_idx for sample_idx, s in enumerate(samples) for _ in s.decision_points
    ]
    total_dps = len(dp_owner)
    if total_dps == 0:
        raise ValueError("build_warmstart_samples: rollout has no decision points.")

    rewards = [0.0] * total_dps
    if is_terminal:
        rewards[-1] = float(episodic_reward)
    n = n_step if n_step is not None else total_dps
    g = n_step_return_action_level(rewards, [0.0] * total_dps, gamma, n)

    # Split the joint returns back onto their owning samples, in order.
    per_sample_returns: list[list[float]] = [[] for _ in samples]
    for sample_idx, g_k in zip(dp_owner, g):
        per_sample_returns[sample_idx].append(g_k)

    out: list[AdvantageTrainingSample] = []
    for s, returns in zip(samples, per_sample_returns):
        targets = [
            DecisionPointTarget(
                response_start=dp.response_start,
                executed_action=dp.admissible_actions[dp.executed_action_idx],
                v_target=g_k,
                q_plus_target=g_k,
            )
            for dp, g_k in zip(s.decision_points, returns)
        ]
        out.append(
            AdvantageTrainingSample(
                prompt_ids=s.prompt_ids,
                prompt_mask=s.prompt_mask,
                completion_ids=s.completion_ids,
                completion_mask=s.completion_mask,
                decision_point_targets=targets,
            )
        )
    return out
