"""Action-level ARM advantage + regression-target math (pure functions).

The Advantage Server owns advantage computation (D-1), so the per-decision-point
math lives here next to compute.py. These functions re-grain the token-level
`log_regret_ratio` logic (orchestrator/per_token_advantage.py, left untouched as
dead code on this branch) from K candidate tokens to the admissible action set.

Phase 2 defines and unit-tests these; Phase 6 wires them into compute.py. The
end-to-end composition (executed in Phases 4/6, not here):

    # Phase 4 (verifiers env), per decision point:
    cond_j         = conditional_renorm(action_logprobs_j)   # for each block j
    star_per_block = [cond_j[executed_idx] for j in blocks]
    pi_hat_star    = rbmc_marginal_loo(star_per_block, exclude_idx=j_star)

    # Phase 6 (AdvServer compute.py), per rollout:
    g              = n_step_return_action_level(rewards, v_target, gamma, n_step)
    for k:
        rm         = pi_rm(q_plus[k], v[k])
        adv[k]     = action_advantage(rm[executed_idx[k]], pi_hat_star[k])
        vtg[k]     = v_target_from_return(g[k])
        qtg[k]     = q_plus_target(q_plus[k][executed_idx[k]], v[k], g[k])

Diary §6-§7 is the source of truth for the math.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def pi_rm(q_plus: list[float], v: float) -> list[float]:
    """Regret-matching distribution π_RM over A(o).

    r_a = max(0, q_plus[a] - v); if Σ r_a > 0 then π_RM[a] = r_a / Σ r_a, else
    uniform 1/|A| (cold start / all-nonpositive regret). Sums to 1.
    """
    n = len(q_plus)
    if n == 0:
        raise ValueError("q_plus must be non-empty")
    regrets = [max(0.0, q - v) for q in q_plus]
    total = sum(regrets)
    if total > 0.0:
        return [r / total for r in regrets]
    return [1.0 / n] * n


def action_advantage(pi_rm_star: float, pi_hat_star: float, floor: float = 1e-6) -> float:
    """A(a*) = log(clamp(pi_rm_star, floor)) - log(clamp(pi_hat_star, floor)).

    Flooring both probabilities before the log bounds the advantage (avoids ±inf
    when pi_rm_star == 0 or pi_hat_star ~ 0); with floor=1e-6 the magnitude is
    bounded at ~log(1/1e-6) ≈ 13.8. The advantage itself is not separately
    clamped -- the probability floor bounds it. Matches the log_regret_ratio
    convention. The fixed point A=0 at pi_rm_star == pi_hat_star is preserved.
    """
    return math.log(max(pi_rm_star, floor)) - math.log(max(pi_hat_star, floor))


def n_step_return_action_level(
    rewards: list[float],
    v_target: list[float],
    gamma: float,
    n_step: int,
) -> list[float]:
    """n-step return g_k per decision point (turn), V_target-bootstrapped.

        g_k = Σ_{i=0}^{h-1} gamma^i * rewards[k+i] + gamma^h * v_target[k+h]
        where h = min(n_step, K - k).

    When k + n_step < K the bootstrap is the interior estimate v_target[k+n_step].
    When k + n_step >= K the horizon reaches the terminal decision point and there
    is no bootstrap past the end (boundary 0) -- correct for ALFWorld's terminal-
    only reward (the episode's final value already sits in rewards[-1]). With
    gamma=1 and a terminal-only reward, n_step >= K makes every g_k the terminal
    reward (Monte Carlo).

    This takes the *full rollout's* decision-point sequence (all K turns), not a
    single truncated sample's -- the terminal reward must propagate across the
    whole episode (see phase doc §8). The Phase 6 caller assembles it.
    """
    if n_step < 1:
        raise ValueError(f"n_step must be >= 1, got {n_step}")
    K = len(rewards)
    if len(v_target) != K:
        raise ValueError(f"rewards (len {K}) and v_target (len {len(v_target)}) must match")
    g = [0.0] * K
    for k in range(K):
        horizon = min(k + n_step, K)
        acc = 0.0
        for j in range(k, horizon):
            acc += (gamma ** (j - k)) * rewards[j]
        if horizon < K:
            acc += (gamma ** (horizon - k)) * v_target[horizon]
        # else: horizon == K -> terminal boundary, no bootstrap (boundary value 0)
        g[k] = acc
    return g


def v_target_from_return(g_k: float) -> float:
    """v_target = g_k (identity; named for symmetry / call-site clarity)."""
    return g_k


def q_plus_target(q_plus_star: float, v: float, g_k: float) -> float:
    """Q+ regression target: max(0, q_plus_star - v) + g_k.

    The CFR+ accumulation's 'previous weights' are supplied implicitly by
    pipeline lag (the AdvServer's forward lags the trainer's update), so
    q_plus_star / v here are the AdvServer's current forward -- no separate
    prev-weight inputs. max(0, q+_star - v) is the carried-forward clipped
    regret; g_k is the fresh n-step return.
    """
    return max(0.0, q_plus_star - v) + g_k


@dataclass
class ActionAdvantageInputs:
    """Per-rollout inputs to the AdvServer advantage computation.

    All per-decision-point lists are indexed by decision point k = 0..K-1 and
    must have equal length K; q_plus[k] is indexed by admissible action a.
    """

    q_plus: list[list[float]]  # q_plus[k][a] = Q+(o_k, a) over A(o_k)
    v: list[float]  # v[k] = V(o_k)
    executed_idx: list[int]  # index of a*_k within A(o_k)
    pi_hat_star: list[float]  # π̂(a*_k|o_k), shipped from rollout (Phase 4)
    rewards: list[float]  # per-turn reward (zero except terminal)
    v_target: list[float]  # V_target(o_k) for bootstrapping g_k
    gamma: float = 1.0
    n_step: int = 1  # tunable; default a placeholder, set in config (DQ2.3)

    def __post_init__(self) -> None:
        k = len(self.q_plus)
        for name, seq in (
            ("v", self.v),
            ("executed_idx", self.executed_idx),
            ("pi_hat_star", self.pi_hat_star),
            ("rewards", self.rewards),
            ("v_target", self.v_target),
        ):
            if len(seq) != k:
                raise ValueError(
                    f"ActionAdvantageInputs.{name} has length {len(seq)}, expected {k} "
                    "(all per-decision-point lists must equal len(q_plus))"
                )
        for j, (row, idx) in enumerate(zip(self.q_plus, self.executed_idx)):
            if not 0 <= idx < len(row):
                raise ValueError(
                    f"executed_idx[{j}]={idx} out of range for |A(o_{j})|={len(row)}"
                )


@dataclass
class ActionAdvantageOutputs:
    """Per-rollout outputs: one value per decision point k = 0..K-1."""

    advantage: list[float]  # A(a*_k) -- Phase 6 broadcasts across the response span
    v_target_out: list[float]  # V regression target per decision point (= g_k)
    q_plus_target_out: list[float]  # Q+ regression target per decision point


def action_advantage_fn(inputs: ActionAdvantageInputs) -> ActionAdvantageOutputs:
    """Per-rollout composition: the action-level analog of
    arm_regret_matching_advantage_fn. The n-step return runs once over the whole
    decision-point sequence (so the terminal reward propagates across truncation
    splits -- the caller assembles the full sequence); per decision point k:
    pi_RM over A(o_k), advantage log pi_RM(a*) - log pi_hat(a*), and the V/Q+
    regression targets. Pure -- the math is Phase 2; this just wires it.
    """
    g = n_step_return_action_level(inputs.rewards, inputs.v_target, inputs.gamma, inputs.n_step)
    advantage: list[float] = []
    v_target_out: list[float] = []
    q_plus_target_out: list[float] = []
    for k in range(len(inputs.v)):
        e = inputs.executed_idx[k]
        rm = pi_rm(inputs.q_plus[k], inputs.v[k])
        advantage.append(action_advantage(rm[e], inputs.pi_hat_star[k]))
        v_target_out.append(v_target_from_return(g[k]))
        q_plus_target_out.append(q_plus_target(inputs.q_plus[k][e], inputs.v[k], g[k]))
    return ActionAdvantageOutputs(
        advantage=advantage,
        v_target_out=v_target_out,
        q_plus_target_out=q_plus_target_out,
    )


def tokenize_action(tokenizer, action_text: str) -> list[int]:
    """Shared tokenization of an admissible action for Q+ append-and-read.

    Encodes "<action>{action_text}</action>" with the closing terminator (so
    prefix-overlapping actions like "go to cabinet 1" vs "12" are distinguished)
    and no special tokens. This is the single helper the AdvServer Q+ (Phase 6)
    and the AdvTrainer Q+ (Phase 7) both call, so a* is the *same* token sequence
    on both sides. (Phase 4's rollout-side pi_hat scores the same "<action>...
    </action>" span; the live equality of all three is the Phase 9 gate, DQ5.5.)
    """
    return tokenizer.encode("<action>" + action_text + "</action>", add_special_tokens=False)
