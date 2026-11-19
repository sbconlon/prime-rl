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


# ---------------------------------------------------------------------------
# ARM redesign (plan/arm-redesign, Phase 1): anchored regret matching.
#
# Vocabulary: the *instantaneous regret* of an action is its plain action value
# minus V; the *stored regret* is what Q+ - V holds. The functions below read the
# stored regret.
# ---------------------------------------------------------------------------

ADVANTAGE_FORMS = ("log_ratio", "difference")
TARGET_FORMS = ("sum", "average")


def cleared_regrets(q_plus: list[float], v: float, margin: float = 0.0) -> list[float]:
    """R(a) = max(0, Q+(o,a) - V(o) - margin): the stored regret that clears the
    noise margin. Continuous in every input; zero for every action whose gap is
    at or below the margin."""
    if len(q_plus) == 0:
        raise ValueError("q_plus must be non-empty")
    if margin < 0.0:
        raise ValueError(f"margin must be >= 0, got {margin}")
    return [max(0.0, q - v - margin) for q in q_plus]


def difference_advantage(regret_star: float, total: float, pi_hat_star: float, anchor: float) -> float:
    """A(a*) = target(a*) - pi_hat(a*), with the anchored target

        target(a) = (R(a) + anchor * pi_hat(a)) / (S + anchor),   S = sum_b R(b).

    The current policy holds `anchor` votes, split the way it splits its
    probability; each action adds its cleared regret as extra votes. Algebraically

        A(a*) = (R(a*) - S * pi_hat(a*)) / (S + anchor).

    Properties (pinned by tests): exactly 0.0 when S == 0 (no cleared regret means
    no target, so no fallback distribution exists); |A| <= S / (S + anchor) < 1;
    A == 0 for every action exactly when pi_hat == R / S; continuous in R, S and
    pi_hat because the divisor never falls below `anchor`. pi_hat is used as
    scored (not renormalised), so probability mass that leaks off the admissible
    actions puts every action with cleared regret below its target.
    """
    if anchor <= 0.0:
        raise ValueError(f"anchor must be > 0 for the difference advantage, got {anchor}")
    if total <= 0.0:
        return 0.0
    return (regret_star - total * pi_hat_star) / (total + anchor)


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


def q_plus_target(
    q_plus_star: float, v: float, g_k: float, phi_decay: float = 1.0, form: str = "sum"
) -> float:
    """Q+ regression target.

    form="sum" (default, the original rule): phi_decay * max(0, q_plus_star - v) + g_k.

    form="average" (plan/arm-redesign D-5): phi_decay * max(q_plus_star, v)
    + (1 - phi_decay) * g_k. In stored-regret terms (x = Q+ - V) this is
    x <- phi * max(0, x) + (1 - phi) * (g - V): the store holds the *average* of
    regret, so for a steady instantaneous regret r > 0 it settles at r, not at
    r / (1 - phi). That is the unit the warm start is in (its labels are plain
    returns), so thresholds on the stored regret keep their meaning through a
    run, and phi_decay becomes a pure memory-length knob. It is the "sum" rule
    divided by 1 / (1 - phi); negative regret does not accumulate in either.
    phi_decay = 0 gives the plain return.

    Notes on the "sum" form follow.

    The CFR+ accumulation's 'previous weights' are supplied implicitly by
    pipeline lag (the AdvServer's forward lags the trainer's update), so
    q_plus_star / v here are the AdvServer's current forward -- no separate
    prev-weight inputs. max(0, q+_star - v) is the carried-forward clipped
    regret; g_k is the fresh n-step return.

    phi_decay (arm_phi_decay) bounds the regret accumulation: the self-feeding
    recurrence q_next = phi_decay * max(0, q_prev - v) + g is a contraction for
    phi_decay < 1 (bounded fixed point (g - phi_decay*v)/(1 - phi_decay)), but at
    phi_decay = 1 with gamma=1 and g > v it grows by (g - v) every step and the
    target diverges -- the run-001 collapse (q_tgt_abs 0.30 -> 1.37, advantages
    spiking, discrimination cratering). Mirrors the token-level convention
    (per_token_advantage: phi = arm_phi_decay * clamp(q+ - v, min=0)). phi_decay=1.0
    is the previous (unbounded) behavior.
    """
    if form == "average":
        return phi_decay * max(q_plus_star, v) + (1.0 - phi_decay) * g_k
    if form != "sum":
        raise ValueError(f"unknown Q+ target form {form!r}; expected one of {TARGET_FORMS}")
    return phi_decay * max(0.0, q_plus_star - v) + g_k


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
    phi_decay: float = 1.0  # arm_phi_decay: CFR+ regret-accumulation decay (<1 bounds it)
    # plan/arm-redesign Phase 1. Defaults reproduce the original rule exactly.
    advantage_form: str = "log_ratio"  # "log_ratio" | "difference" (anchored regret matching)
    margin: float = 0.0  # delta: stored regret at or below it counts as zero (difference form)
    anchor: float = 0.0  # kappa: votes held by the current policy (difference form, must be > 0)
    target_form: str = "sum"  # "sum" | "average" (see q_plus_target)

    def __post_init__(self) -> None:
        if self.advantage_form not in ADVANTAGE_FORMS:
            raise ValueError(
                f"unknown advantage_form {self.advantage_form!r}; expected one of {ADVANTAGE_FORMS}"
            )
        if self.target_form not in TARGET_FORMS:
            raise ValueError(f"unknown target_form {self.target_form!r}; expected one of {TARGET_FORMS}")
        if self.margin < 0.0:
            raise ValueError(f"margin must be >= 0, got {self.margin}")
        if self.advantage_form == "difference" and self.anchor <= 0.0:
            raise ValueError(f"the difference advantage needs anchor > 0, got {self.anchor}")
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
    # training-probe (Phase 2, D-8): intermediates the loop already computes,
    # kept for the diagnostics view. Additive; existing callers ignore them.
    pi_rm: list[list[float]] | None = None  # pi_rm[k][a] = π_RM(a | o_k) over A(o_k)
    g: list[float] | None = None  # g[k] = n-step return at decision point k
    # plan/arm-redesign Phase 1: regret[k][a] = cleared regret R(a | o_k). Set only
    # under the difference form (None under the original rule, so its outputs are
    # unchanged). Under that form pi_rm[k] is R / S, and all zeros when S == 0.
    regret: list[list[float]] | None = None


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
    pi_rm_rows: list[list[float]] = []
    difference = inputs.advantage_form == "difference"
    regret_rows: list[list[float]] | None = [] if difference else None
    for k in range(len(inputs.v)):
        e = inputs.executed_idx[k]
        if difference:
            # Anchored regret matching: no fallback distribution. With no cleared
            # regret the target is the policy itself and the advantage is 0.
            r = cleared_regrets(inputs.q_plus[k], inputs.v[k], inputs.margin)
            total = sum(r)
            rm = [x / total for x in r] if total > 0.0 else [0.0] * len(r)
            regret_rows.append(r)
            advantage.append(difference_advantage(r[e], total, inputs.pi_hat_star[k], inputs.anchor))
        else:
            rm = pi_rm(inputs.q_plus[k], inputs.v[k])
            advantage.append(action_advantage(rm[e], inputs.pi_hat_star[k]))
        pi_rm_rows.append(rm)
        v_target_out.append(v_target_from_return(g[k]))
        q_plus_target_out.append(
            q_plus_target(inputs.q_plus[k][e], inputs.v[k], g[k], inputs.phi_decay, inputs.target_form)
        )
    return ActionAdvantageOutputs(
        advantage=advantage,
        v_target_out=v_target_out,
        q_plus_target_out=q_plus_target_out,
        pi_rm=pi_rm_rows,
        g=list(g),
        regret=regret_rows,
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
