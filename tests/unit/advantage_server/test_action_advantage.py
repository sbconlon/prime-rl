"""Phase 2 (action-level ARM) -- pure advantage + regression-target math.

Synthetic inputs only; no GPU/env/vLLM. Pins the numerics that make the
algorithm correct: regret matching over A(o), the log-ratio fixed point, the
probability floor, and the n-step return terminal/bootstrap boundary.
"""

from __future__ import annotations

import math

import pytest

from prime_rl.advantage_server.action_advantage import (
    ActionAdvantageInputs,
    ActionAdvantageOutputs,
    action_advantage,
    action_advantage_fn,
    n_step_return_action_level,
    pi_rm,
    q_plus_target,
    tokenize_action,
    v_target_from_return,
)


# --------------------------------------------------------------------------- #
# pi_rm
# --------------------------------------------------------------------------- #


def test_pi_rm_all_positive():
    # q+ - v = [3, 1] -> regrets [3, 1] -> [0.75, 0.25]
    out = pi_rm([5.0, 3.0], v=2.0)
    assert out == pytest.approx([0.75, 0.25])
    assert sum(out) == pytest.approx(1.0)


def test_pi_rm_all_nonpositive_uniform():
    # every q+ - v <= 0 -> uniform 1/|A|
    out = pi_rm([1.0, 0.0, -2.0], v=3.0)
    assert out == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert sum(out) == pytest.approx(1.0)


def test_pi_rm_mixed():
    # q+ - v = [2, -1, 6] -> regrets [2, 0, 6] -> [0.25, 0, 0.75]
    out = pi_rm([5.0, 2.0, 9.0], v=3.0)
    assert out == pytest.approx([0.25, 0.0, 0.75])
    assert sum(out) == pytest.approx(1.0)


def test_pi_rm_empty_raises():
    with pytest.raises(ValueError):
        pi_rm([], v=0.0)


# --------------------------------------------------------------------------- #
# action_advantage (log-ratio + floor)
# --------------------------------------------------------------------------- #


def test_action_advantage_fixed_point():
    """pi_rm_star == pi_hat_star -> 0.0. The property that makes the formula sound
    (pi_RM is a stable fixed point, not mode-seeking)."""
    assert action_advantage(0.4, 0.4) == pytest.approx(0.0)


def test_action_advantage_clamps_zero_pi_rm():
    a = action_advantage(0.0, 0.5)
    assert math.isfinite(a)
    assert a < 0.0
    # log(1e-6) - log(0.5)
    assert a == pytest.approx(math.log(1e-6) - math.log(0.5))


def test_action_advantage_clamps_tiny_pi_hat():
    a = action_advantage(0.5, 0.0)
    assert math.isfinite(a)
    assert a > 0.0
    assert a == pytest.approx(math.log(0.5) - math.log(1e-6))


def test_action_advantage_monotone():
    """For fixed pi_hat, higher pi_RM -> higher advantage."""
    lo = action_advantage(0.2, 0.3)
    hi = action_advantage(0.6, 0.3)
    assert hi > lo


def test_action_advantage_log_ratio_value():
    a = action_advantage(0.6, 0.2)
    assert a == pytest.approx(math.log(0.6) - math.log(0.2))


# --------------------------------------------------------------------------- #
# n_step_return_action_level
# --------------------------------------------------------------------------- #


def test_n_step_return_gamma1_terminal():
    """gamma=1, reward only at the terminal turn, n_step >= K -> every g_k equals
    the terminal reward (Monte Carlo, no bootstrap)."""
    rewards = [0.0, 0.0, 0.0, 1.0]
    v_target = [9.0, 9.0, 9.0, 9.0]  # must be ignored (no bootstrap reached)
    g = n_step_return_action_level(rewards, v_target, gamma=1.0, n_step=10)
    assert g == pytest.approx([1.0, 1.0, 1.0, 1.0])


def test_n_step_return_bootstrap():
    """Small n_step bootstraps from v_target[k+n]. gamma=1, n_step=1:
    g_k = rewards[k] + v_target[k+1] for interior k; terminal k = rewards[-1]."""
    rewards = [0.0, 0.0, 1.0]
    v_target = [0.5, 0.7, 0.9]
    g = n_step_return_action_level(rewards, v_target, gamma=1.0, n_step=1)
    # k=0: 0 + v_target[1]=0.7 ; k=1: 0 + v_target[2]=0.9 ; k=2: terminal -> 1.0
    assert g == pytest.approx([0.7, 0.9, 1.0])


def test_n_step_return_gamma_discount():
    """gamma < 1 discounts both the reward sum and the bootstrap."""
    rewards = [1.0, 2.0, 0.0]
    v_target = [0.0, 0.0, 5.0]
    g = n_step_return_action_level(rewards, v_target, gamma=0.5, n_step=2)
    # k=0: r0 + 0.5*r1 + 0.5^2 * v_target[2] = 1 + 1 + 0.25*5 = 3.25
    # k=1: r1 + 0.5*r2 + (horizon=3=K -> no bootstrap) = 2 + 0 = 2.0
    # k=2: r2 = 0.0
    assert g == pytest.approx([3.25, 2.0, 0.0])


def test_n_step_return_terminal_no_overrun():
    """Bootstrap horizon past the terminal clamps to Monte Carlo -- no OOB read.
    A poisoned v_target sentinel would surface as a wrong value if read."""
    rewards = [0.0, 0.0, 1.0]
    v_target = [0.0, 0.0, 0.0]
    # n_step=5 > K=3: every horizon hits K, no bootstrap, no index error
    g = n_step_return_action_level(rewards, v_target, gamma=1.0, n_step=5)
    assert g == pytest.approx([1.0, 1.0, 1.0])


def test_n_step_return_bad_n_step_raises():
    with pytest.raises(ValueError):
        n_step_return_action_level([0.0], [0.0], gamma=1.0, n_step=0)


def test_n_step_return_length_mismatch_raises():
    with pytest.raises(ValueError):
        n_step_return_action_level([0.0, 1.0], [0.0], gamma=1.0, n_step=1)


# --------------------------------------------------------------------------- #
# value-target helpers
# --------------------------------------------------------------------------- #


def test_v_target_from_return():
    assert v_target_from_return(0.42) == 0.42


def test_q_plus_target_positive_regret():
    # max(0, q+_star - v) + g_k = max(0, 5 - 2) + 1.0 = 4.0
    assert q_plus_target(5.0, 2.0, 1.0) == pytest.approx(4.0)


def test_q_plus_target_negative_regret_branch():
    # q+_star - v < 0 -> clipped to 0 -> result == g_k
    assert q_plus_target(1.0, 3.0, 0.5) == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# arm_phi_decay: CFR+ regret-accumulation bound (run-001 collapse fix)
# --------------------------------------------------------------------------- #


def test_q_plus_target_phi_decay_default_is_unbounded_behavior():
    """Default phi_decay=1.0 is byte-identical to the pre-fix behavior (no
    regression for any existing target)."""
    assert q_plus_target(5.0, 2.0, 1.0) == pytest.approx(max(0.0, 5.0 - 2.0) + 1.0)
    assert q_plus_target(5.0, 2.0, 1.0, 1.0) == q_plus_target(5.0, 2.0, 1.0)


def test_q_plus_target_phi_decay_scales_only_clipped_regret():
    # phi_decay multiplies the clipped regret term, NOT g_k.
    assert q_plus_target(5.0, 2.0, 1.0, phi_decay=0.5) == pytest.approx(0.5 * 3.0 + 1.0)  # 2.5
    # negative-regret branch: clipped to 0, so phi_decay is irrelevant -> g_k.
    assert q_plus_target(1.0, 3.0, 0.7, phi_decay=0.5) == pytest.approx(0.7)


def test_q_plus_target_self_feeding_diverges_at_phi_1_bounded_below_1():
    """The exact run-001 runaway, reproduced and bounded. Self-feeding recurrence
    q_next = phi_decay*max(0, q_prev - v) + g with constant v, g (g > v)."""
    v, g = 0.0, 1.0  # g > v, so phi=1 grows by (g - v) = 1 every step
    # phi_decay = 1.0 -> diverges monotonically (the collapse).
    q = 0.0
    seq = []
    for _ in range(50):
        q = q_plus_target(q, v, g, phi_decay=1.0)
        seq.append(q)
    assert seq[-1] > seq[-2] > seq[10]  # strictly growing
    assert seq[-1] > 40  # ~ grows by 1/step from 0

    # phi_decay = 0.9 -> contraction to the fixed point (g - phi*v)/(1 - phi) = 10.
    q = 0.0
    for _ in range(500):
        q = q_plus_target(q, v, g, phi_decay=0.9)
    assert q == pytest.approx(g / (1 - 0.9), abs=1e-3)  # 10.0


def test_action_advantage_fn_threads_phi_decay():
    """phi_decay set on ActionAdvantageInputs reaches q_plus_target_out."""
    inp = ActionAdvantageInputs(
        q_plus=[[5.0, 3.0]],
        v=[2.0],
        executed_idx=[0],
        pi_hat_star=[0.5],
        rewards=[1.0],
        v_target=[0.0],
        gamma=1.0,
        n_step=10,  # MC -> g_0 = terminal reward 1.0
        phi_decay=0.5,
    )
    out = action_advantage_fn(inp)
    # g_0 = 1.0; q+*-v = 5-2 = 3; target = 0.5*3 + 1 = 2.5
    assert out.q_plus_target_out[0] == pytest.approx(2.5)
    # default (phi_decay=1.0) would be 3 + 1 = 4.0
    inp_default = ActionAdvantageInputs(
        q_plus=[[5.0, 3.0]], v=[2.0], executed_idx=[0], pi_hat_star=[0.5],
        rewards=[1.0], v_target=[0.0], gamma=1.0, n_step=10,
    )
    assert action_advantage_fn(inp_default).q_plus_target_out[0] == pytest.approx(4.0)


# --------------------------------------------------------------------------- #
# dataclasses
# --------------------------------------------------------------------------- #


def test_inputs_outputs_dataclasses():
    inp = ActionAdvantageInputs(
        q_plus=[[1.0, 2.0], [0.0, 3.0]],
        v=[0.5, 1.0],
        executed_idx=[1, 0],
        pi_hat_star=[0.4, 0.6],
        rewards=[0.0, 1.0],
        v_target=[0.2, 0.9],
        gamma=1.0,
        n_step=2,
    )
    assert len(inp.q_plus) == 2
    out = ActionAdvantageOutputs(
        advantage=[0.1, -0.2],
        v_target_out=[0.2, 0.9],
        q_plus_target_out=[0.5, 1.0],
    )
    assert out.advantage == [0.1, -0.2]


def test_inputs_length_invariant_raises():
    with pytest.raises(ValueError):
        ActionAdvantageInputs(
            q_plus=[[1.0, 2.0], [0.0, 3.0]],
            v=[0.5],  # wrong length
            executed_idx=[1, 0],
            pi_hat_star=[0.4, 0.6],
            rewards=[0.0, 1.0],
            v_target=[0.2, 0.9],
        )


def test_inputs_executed_idx_out_of_range_raises():
    with pytest.raises(ValueError):
        ActionAdvantageInputs(
            q_plus=[[1.0, 2.0]],
            v=[0.5],
            executed_idx=[5],  # out of range for |A|=2
            pi_hat_star=[0.4],
            rewards=[1.0],
            v_target=[0.2],
        )


# --------------------------------------------------------------------------- #
# action_advantage_fn (Phase 6 composition)
# --------------------------------------------------------------------------- #


def test_composition_matches_pieces():
    inp = ActionAdvantageInputs(
        q_plus=[[5.0, 3.0], [2.0, 9.0]],  # k=0 regrets vs v=2 -> [3,1]; k=1 vs v=3 -> [0,6]
        v=[2.0, 3.0],
        executed_idx=[0, 1],
        pi_hat_star=[0.5, 0.5],
        rewards=[0.0, 1.0],
        v_target=[0.2, 0.9],
        gamma=1.0,
        n_step=10,  # MC: every g_k == terminal reward 1.0
    )
    out = action_advantage_fn(inp)
    # hand-composed
    g = n_step_return_action_level(inp.rewards, inp.v_target, inp.gamma, inp.n_step)
    for k in range(2):
        rm = pi_rm(inp.q_plus[k], inp.v[k])
        e = inp.executed_idx[k]
        assert out.advantage[k] == pytest.approx(action_advantage(rm[e], inp.pi_hat_star[k]))
        assert out.v_target_out[k] == pytest.approx(v_target_from_return(g[k]))
        assert out.q_plus_target_out[k] == pytest.approx(
            q_plus_target(inp.q_plus[k][e], inp.v[k], g[k])
        )


def test_composition_preserves_fixed_point():
    """If pi_RM(a*) == pi_hat(a*), the composed advantage is 0."""
    # k=0: q_plus [4,2] vs v=2 -> regrets [2,0] -> pi_RM = [1.0, 0.0]; executed idx 0
    # -> pi_RM(a*) = 1.0; set pi_hat_star = 1.0 -> advantage 0.
    inp = ActionAdvantageInputs(
        q_plus=[[4.0, 2.0]],
        v=[2.0],
        executed_idx=[0],
        pi_hat_star=[1.0],
        rewards=[1.0],
        v_target=[0.0],
        gamma=1.0,
        n_step=1,
    )
    out = action_advantage_fn(inp)
    assert out.advantage[0] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# tokenize_action (shared Q+/AdvTrainer helper)
# --------------------------------------------------------------------------- #


class _FakeTok:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def test_tokenize_action_wraps_with_tags_and_terminator():
    tok = _FakeTok()
    ids = tokenize_action(tok, "look")
    assert ids == [ord(c) for c in "<action>look</action>"]


def test_tokenize_action_deterministic_same_text():
    """The three-site consistency invariant (DQ5.5): identical text -> identical ids."""
    tok = _FakeTok()
    assert tokenize_action(tok, "go to cabinet 1") == tokenize_action(tok, "go to cabinet 1")
    assert tokenize_action(tok, "go to cabinet 1") != tokenize_action(tok, "go to cabinet 12")
