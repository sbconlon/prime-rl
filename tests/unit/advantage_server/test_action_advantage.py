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
    n_step_return_action_level,
    pi_rm,
    q_plus_target,
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
