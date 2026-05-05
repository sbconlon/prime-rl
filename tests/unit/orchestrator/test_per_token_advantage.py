"""Phase 2 — PPO GAE advantage tests.

Pure-arithmetic tests of `ppo_gae_advantage_fn`. No GPU, no model loading,
no orchestrator wiring — V/V_target tensors are constructed deterministically
in each test so the GAE math can be hand-verified.

Reference recursion (mirrored from per_token_advantage.py):
    delta_k = r_k + gamma * V_target(o_{k+1}) * not_done_k - V(o_k)
    A_k     = delta_k + gamma * lam * not_done_k * A_{k+1}
    v_target_k = A_k + V(o_k)
"""

from __future__ import annotations

import torch

from prime_rl.orchestrator.per_token_advantage import (
    PerTokenAdvantageOutputs,
    PpoAdvantageInputs,
    ppo_gae_advantage_fn,
)
from prime_rl.transport.types import TrainingSample


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sample(completion_len: int, prompt_len: int = 1) -> TrainingSample:
    """Construct a minimal TrainingSample with completion_mask all True."""
    return TrainingSample(
        prompt_ids=[0] * prompt_len,
        prompt_mask=[False] * prompt_len,
        completion_ids=list(range(completion_len)),
        completion_mask=[True] * completion_len,
        completion_logprobs=[0.0] * completion_len,
        completion_temperatures=[1.0] * completion_len,
    )


def _gae_reference(
    v: torch.Tensor,
    v_target: torch.Tensor,
    rewards: torch.Tensor,
    not_done: torch.Tensor,
    gamma: float,
    lam: float,
    bootstrap_last: float,
) -> torch.Tensor:
    """Independent reference implementation for cross-checking.

    `bootstrap_last` is the V_target value used at the trajectory boundary
    (post-last position). Terminal -> 0; truncated -> v_target[-1].
    """
    n = v.shape[0]
    v_target_next = torch.zeros(n, dtype=v.dtype)
    if n > 1:
        v_target_next[:-1] = v_target[1:]
    v_target_next[-1] = bootstrap_last
    deltas = rewards + gamma * v_target_next * not_done - v
    a = torch.zeros(n, dtype=v.dtype)
    running = 0.0
    for k in range(n - 1, -1, -1):
        running = float(deltas[k]) + gamma * lam * float(not_done[k]) * running
        a[k] = running
    return a


# ---------------------------------------------------------------------------
# GAE arithmetic correctness
# ---------------------------------------------------------------------------


def test_gae_single_sample_constant_v():
    """Single sample with constant V; advantages match the hand-derived recursion."""
    sample = _make_sample(completion_len=5)
    v = torch.full((5,), 0.5)
    inputs = PpoAdvantageInputs(
        samples=[sample],
        episodic_reward=1.0,
        is_terminal=True,
        v_all=v.clone(),
        v_target_all=v.clone(),
        gamma=0.99,
        lam=0.95,
    )
    out = ppo_gae_advantage_fn(inputs)

    # Hand-computed reference: rewards = [0,0,0,0,1], not_done = [1,1,1,1,0]
    rewards = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0])
    not_done = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0])
    expected = _gae_reference(v, v, rewards, not_done, gamma=0.99, lam=0.95, bootstrap_last=0.0)

    got = torch.tensor(out.advantages[0])
    assert torch.allclose(got, expected, atol=1e-6)


def test_gae_gamma_zero_reduces_to_td_error():
    """At gamma=0, GAE collapses to delta_k = r_k - V(o_k)."""
    sample = _make_sample(completion_len=3)
    v = torch.tensor([0.1, 0.2, 0.3])
    inputs = PpoAdvantageInputs(
        samples=[sample],
        episodic_reward=1.0,
        is_terminal=True,
        v_all=v.clone(),
        v_target_all=v.clone(),
        gamma=0.0,
        lam=0.95,
    )
    out = ppo_gae_advantage_fn(inputs)

    # delta_k = r_k - V(o_k) since gamma=0 zeros the bootstrap; gamma also
    # zeros the recursion's future-contribution term, so A_k = delta_k.
    expected = torch.tensor([0.0 - 0.1, 0.0 - 0.2, 1.0 - 0.3])
    got = torch.tensor(out.advantages[0])
    assert torch.allclose(got, expected, atol=1e-6)


def test_gae_lambda_zero_reduces_to_td_error():
    """At lambda=0, GAE reduces to one-step TD error delta_k (no propagation)."""
    sample = _make_sample(completion_len=3)
    v = torch.tensor([0.1, 0.2, 0.3])
    inputs = PpoAdvantageInputs(
        samples=[sample],
        episodic_reward=1.0,
        is_terminal=True,
        v_all=v.clone(),
        v_target_all=v.clone(),
        gamma=0.99,
        lam=0.0,
    )
    out = ppo_gae_advantage_fn(inputs)

    # rewards = [0,0,1], not_done = [1,1,0]; v_target_next = [0.2,0.3,0]
    # deltas = [0+0.99*0.2-0.1, 0+0.99*0.3-0.2, 1+0-0.3] = [0.098, 0.097, 0.7]
    # A_k = delta_k (no propagation at lambda=0)
    expected = torch.tensor([0.098, 0.097, 0.7])
    got = torch.tensor(out.advantages[0])
    assert torch.allclose(got, expected, atol=1e-6)


def test_gae_lambda_one_reduces_to_monte_carlo_minus_v():
    """At lambda=1 with V_target=V, GAE = (sum gamma^(k'-k) r_k') - V(o_k)."""
    sample = _make_sample(completion_len=3)
    v = torch.tensor([0.1, 0.2, 0.3])
    gamma = 0.99
    inputs = PpoAdvantageInputs(
        samples=[sample],
        episodic_reward=1.0,
        is_terminal=True,
        v_all=v.clone(),
        v_target_all=v.clone(),
        gamma=gamma,
        lam=1.0,
    )
    out = ppo_gae_advantage_fn(inputs)

    # MC return at each k (only r_2 = 1.0 is non-zero):
    # G_2 = 1.0; G_1 = gamma * 1.0; G_0 = gamma^2 * 1.0
    g2 = 1.0
    g1 = gamma * 1.0
    g0 = gamma * gamma * 1.0
    expected = torch.tensor([g0 - 0.1, g1 - 0.2, g2 - 0.3])
    got = torch.tensor(out.advantages[0])
    assert torch.allclose(got, expected, atol=1e-6)


def test_gae_terminal_bootstrap_zero():
    """At the final position when terminal, bootstrap term is gated by not_done=0.

    Concretely: the advantage at the last position equals r_{N-1} - V(o_{N-1})
    regardless of v_target_all[-1]. Verify by changing v_target_all[-1]
    drastically and asserting the last advantage doesn't move.
    """
    sample = _make_sample(completion_len=2)
    v = torch.tensor([0.5, 0.7])

    a_low = ppo_gae_advantage_fn(
        PpoAdvantageInputs(
            samples=[sample],
            episodic_reward=1.0,
            is_terminal=True,
            v_all=v.clone(),
            v_target_all=torch.tensor([0.5, 0.7]),
            gamma=0.99,
            lam=0.95,
        )
    ).advantages[0]
    a_high = ppo_gae_advantage_fn(
        PpoAdvantageInputs(
            samples=[sample],
            episodic_reward=1.0,
            is_terminal=True,
            v_all=v.clone(),
            v_target_all=torch.tensor([0.5, 999.0]),  # last V_target wildly different
            gamma=0.99,
            lam=0.95,
        )
    ).advantages[0]

    # Last position is gated; identical between the two runs.
    assert abs(a_low[-1] - a_high[-1]) < 1e-6
    # And explicitly: last advantage = episodic_reward - V(o_{-1}).
    assert abs(a_low[-1] - (1.0 - 0.7)) < 1e-6


def test_gae_truncation_no_terminal_reward():
    """Truncated rollout: rewards/dones all zero/False; final-position bootstrap via V_target."""
    sample = _make_sample(completion_len=2)
    v = torch.tensor([0.5, 0.7])
    v_target = torch.tensor([0.5, 0.7])
    gamma = 0.99
    lam = 0.95

    out = ppo_gae_advantage_fn(
        PpoAdvantageInputs(
            samples=[sample],
            episodic_reward=0.0,  # ignored: is_terminal=False zeros the rewards vector
            is_terminal=False,
            v_all=v.clone(),
            v_target_all=v_target.clone(),
            gamma=gamma,
            lam=lam,
        )
    )

    # Truncation derivation: rewards=[0,0], not_done=[1,1], v_target_next=[0.7, 0.7]
    # deltas[k=0] = 0 + 0.99*0.7*1 - 0.5 = 0.193
    # deltas[k=1] = 0 + 0.99*0.7*1 - 0.7 = -0.007  (bootstrap from V_target at last pos)
    # A_1 = -0.007
    # A_0 = 0.193 + 0.99*0.95*1*(-0.007)
    delta_last = gamma * v_target[-1].item() * 1.0 - v[-1].item()
    delta_prev = gamma * v_target[1].item() * 1.0 - v[0].item()
    a_last_expected = delta_last
    a_prev_expected = delta_prev + gamma * lam * 1.0 * a_last_expected

    got = torch.tensor(out.advantages[0])
    assert torch.allclose(
        got, torch.tensor([a_prev_expected, a_last_expected]), atol=1e-6
    )

    # And: truncation doesn't generate a phantom episodic reward — changing
    # episodic_reward should not change advantages when is_terminal=False.
    out_with_phantom = ppo_gae_advantage_fn(
        PpoAdvantageInputs(
            samples=[sample],
            episodic_reward=999.0,  # ignored
            is_terminal=False,
            v_all=v.clone(),
            v_target_all=v_target.clone(),
            gamma=gamma,
            lam=lam,
        )
    )
    assert out_with_phantom.advantages == out.advantages


# ---------------------------------------------------------------------------
# Joint recursion across fragmented samples
# ---------------------------------------------------------------------------


def test_gae_two_samples_joint_recursion():
    """Two samples representing one fragmented rollout; recursion runs jointly."""
    s0 = _make_sample(completion_len=2)
    s1 = _make_sample(completion_len=3)
    v = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])  # length 5 = 2 + 3
    inputs = PpoAdvantageInputs(
        samples=[s0, s1],
        episodic_reward=1.0,
        is_terminal=True,
        v_all=v.clone(),
        v_target_all=v.clone(),
        gamma=0.99,
        lam=0.95,
    )
    out = ppo_gae_advantage_fn(inputs)

    # Reference: GAE over the joint length-5 trajectory.
    rewards = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0])
    not_done = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0])
    a_joint = _gae_reference(v, v, rewards, not_done, gamma=0.99, lam=0.95, bootstrap_last=0.0)

    got_flat = torch.tensor(out.advantages[0] + out.advantages[1])
    assert torch.allclose(got_flat, a_joint, atol=1e-6)
    # Per-sample lengths.
    assert len(out.advantages[0]) == 2
    assert len(out.advantages[1]) == 3


def test_gae_joint_recursion_differs_from_independent():
    """Joint computation must differ from per-sample-independent computation."""
    s0 = _make_sample(completion_len=2)
    s1 = _make_sample(completion_len=3)
    v = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])

    # Joint
    out_joint = ppo_gae_advantage_fn(
        PpoAdvantageInputs(
            samples=[s0, s1],
            episodic_reward=1.0,
            is_terminal=True,
            v_all=v.clone(),
            v_target_all=v.clone(),
        )
    )
    a_joint_s0 = out_joint.advantages[0]

    # Independent: pretend each sample is its own complete terminal trajectory.
    out_indep_s0 = ppo_gae_advantage_fn(
        PpoAdvantageInputs(
            samples=[s0],
            episodic_reward=1.0,
            is_terminal=True,
            v_all=v[:2].clone(),
            v_target_all=v[:2].clone(),
        )
    )
    a_indep_s0 = out_indep_s0.advantages[0]

    # Sample 0's advantages must differ between the two computations — the
    # joint version sees the future tokens of sample 1; the independent version
    # treats sample 0 as ending at its own last position.
    assert any(abs(a - b) > 1e-4 for a, b in zip(a_joint_s0, a_indep_s0))


def test_split_back_into_per_sample_lists():
    """Output per-sample shape matches input completion_ids lengths."""
    s0 = _make_sample(completion_len=10)
    s1 = _make_sample(completion_len=5)
    v = torch.full((15,), 0.0)
    out = ppo_gae_advantage_fn(
        PpoAdvantageInputs(
            samples=[s0, s1],
            episodic_reward=1.0,
            is_terminal=True,
            v_all=v.clone(),
            v_target_all=v.clone(),
        )
    )
    assert len(out.advantages) == 2
    assert len(out.advantages[0]) == 10
    assert len(out.advantages[1]) == 5
    assert len(out.v_targets) == 2
    assert len(out.v_targets[0]) == 10
    assert len(out.v_targets[1]) == 5


# ---------------------------------------------------------------------------
# v_target correctness
# ---------------------------------------------------------------------------


def test_v_target_matches_advantage_plus_v():
    """v_target_k = A_k + V(o_k) — the GAE-derived V regression target identity."""
    sample = _make_sample(completion_len=4)
    torch.manual_seed(0)
    v = torch.rand(4)
    v_target = torch.rand(4)
    out = ppo_gae_advantage_fn(
        PpoAdvantageInputs(
            samples=[sample],
            episodic_reward=0.7,
            is_terminal=True,
            v_all=v.clone(),
            v_target_all=v_target.clone(),
        )
    )
    a = torch.tensor(out.advantages[0])
    vt = torch.tensor(out.v_targets[0])
    assert torch.allclose(vt, a + v, atol=1e-6)


def test_v_target_length_matches_completion_length():
    """len(v_targets[i]) == len(samples[i].completion_ids)."""
    s0 = _make_sample(completion_len=7)
    s1 = _make_sample(completion_len=3)
    v = torch.zeros(10)
    out = ppo_gae_advantage_fn(
        PpoAdvantageInputs(
            samples=[s0, s1],
            episodic_reward=1.0,
            is_terminal=True,
            v_all=v.clone(),
            v_target_all=v.clone(),
        )
    )
    assert len(out.v_targets[0]) == len(s0.completion_ids)
    assert len(out.v_targets[1]) == len(s1.completion_ids)


# ---------------------------------------------------------------------------
# Output shape and invariants
# ---------------------------------------------------------------------------


def test_advantages_length_matches_completion_length():
    """len(advantages[i]) == len(samples[i].completion_ids)."""
    s0 = _make_sample(completion_len=8)
    s1 = _make_sample(completion_len=2)
    v = torch.zeros(10)
    out = ppo_gae_advantage_fn(
        PpoAdvantageInputs(
            samples=[s0, s1],
            episodic_reward=1.0,
            is_terminal=True,
            v_all=v.clone(),
            v_target_all=v.clone(),
        )
    )
    assert len(out.advantages[0]) == len(s0.completion_ids)
    assert len(out.advantages[1]) == len(s1.completion_ids)


def test_q_plus_targets_is_none_for_ppo():
    """PPO's GAE function returns q_plus_targets=None; ARM (Phase 3) populates it."""
    sample = _make_sample(completion_len=3)
    v = torch.zeros(3)
    out = ppo_gae_advantage_fn(
        PpoAdvantageInputs(
            samples=[sample],
            episodic_reward=1.0,
            is_terminal=True,
            v_all=v.clone(),
            v_target_all=v.clone(),
        )
    )
    assert isinstance(out, PerTokenAdvantageOutputs)
    assert out.q_plus_targets is None


def test_deterministic_inputs_yield_deterministic_output():
    """Same inputs -> bit-identical outputs (no hidden internal state)."""
    sample = _make_sample(completion_len=6)
    torch.manual_seed(42)
    v = torch.rand(6)
    v_target = torch.rand(6)

    def run() -> PerTokenAdvantageOutputs:
        return ppo_gae_advantage_fn(
            PpoAdvantageInputs(
                samples=[sample],
                episodic_reward=0.5,
                is_terminal=True,
                v_all=v.clone(),
                v_target_all=v_target.clone(),
                gamma=0.97,
                lam=0.9,
            )
        )

    out_a = run()
    out_b = run()
    assert out_a.advantages == out_b.advantages
    assert out_a.v_targets == out_b.v_targets


# ===========================================================================
# Phase 3 -- ARM regret-matching advantage tests
# ===========================================================================

from prime_rl.orchestrator.per_token_advantage import (
    ArmAdvantageInputs,
    arm_regret_matching_advantage_fn,
)


def _arm_inputs(
    samples: list[TrainingSample],
    *,
    episodic_reward: float = 1.0,
    is_terminal: bool = True,
    v_all: torch.Tensor | None = None,
    v_target_all: torch.Tensor | None = None,
    q_plus_sampled_all: torch.Tensor | None = None,
    q_plus_candidates: torch.Tensor | None = None,
    gamma: float = 0.99,
    K: int = 4,
) -> ArmAdvantageInputs:
    """Helper: ArmAdvantageInputs with sensible zero defaults for unused tensors."""
    n_active = sum(sum(s.completion_mask) for s in samples)
    return ArmAdvantageInputs(
        samples=samples,
        episodic_reward=episodic_reward,
        is_terminal=is_terminal,
        v_all=torch.zeros(n_active) if v_all is None else v_all,
        v_target_all=torch.zeros(n_active) if v_target_all is None else v_target_all,
        q_plus_sampled_all=(
            torch.zeros(n_active) if q_plus_sampled_all is None else q_plus_sampled_all
        ),
        q_plus_candidates=(
            torch.zeros(n_active, K) if q_plus_candidates is None else q_plus_candidates
        ),
        gamma=gamma,
    )


# ---------------------------------------------------------------------------
# Regret-matching arithmetic correctness
# ---------------------------------------------------------------------------


def test_arm_single_position_single_positive_q_plus():
    """K=4, only the sampled action has positive Q+ -> A = Q+/Q+ - 1/K = 1 - 1/4 = 0.75."""
    sample = _make_sample(completion_len=1)
    out = arm_regret_matching_advantage_fn(
        _arm_inputs(
            [sample],
            q_plus_sampled_all=torch.tensor([2.0]),
            q_plus_candidates=torch.tensor([[2.0, 0.0, 0.0, 0.0]]),
            K=4,
        )
    )
    assert abs(out.advantages[0][0] - 0.75) < 1e-6


def test_arm_single_position_uniform_positive_q_plus():
    """K=4, all candidates equal -> A = 1/K - 1/K = 0 (uniform regret-matching policy)."""
    sample = _make_sample(completion_len=1)
    out = arm_regret_matching_advantage_fn(
        _arm_inputs(
            [sample],
            q_plus_sampled_all=torch.tensor([1.0]),
            q_plus_candidates=torch.tensor([[1.0, 1.0, 1.0, 1.0]]),
            K=4,
        )
    )
    assert abs(out.advantages[0][0] - 0.0) < 1e-6


def test_arm_single_position_mixed_positive_negative_q_plus():
    """K=4, q_plus_candidates=[2, -1, 1, 3]: clipped=[2,0,1,3], denom=6, A=2/6-1/4."""
    sample = _make_sample(completion_len=1)
    out = arm_regret_matching_advantage_fn(
        _arm_inputs(
            [sample],
            q_plus_sampled_all=torch.tensor([2.0]),
            q_plus_candidates=torch.tensor([[2.0, -1.0, 1.0, 3.0]]),
            K=4,
        )
    )
    expected = 2.0 / 6.0 - 1.0 / 4.0
    assert abs(out.advantages[0][0] - expected) < 1e-6


# ---------------------------------------------------------------------------
# Cold-start degenerate branch
# ---------------------------------------------------------------------------


def test_arm_cold_start_all_zero_q_plus():
    """All Q+ zero at every position -> A = 0 (no division by zero)."""
    sample = _make_sample(completion_len=3)
    out = arm_regret_matching_advantage_fn(
        _arm_inputs(
            [sample],
            q_plus_sampled_all=torch.zeros(3),
            q_plus_candidates=torch.zeros(3, 4),
            K=4,
        )
    )
    assert all(abs(a) < 1e-9 for a in out.advantages[0])


def test_arm_cold_start_all_negative_q_plus():
    """All Q+ non-positive (clipped to zero) -> degenerate branch -> A = 0."""
    sample = _make_sample(completion_len=1)
    out = arm_regret_matching_advantage_fn(
        _arm_inputs(
            [sample],
            q_plus_sampled_all=torch.tensor([-1.0]),
            q_plus_candidates=torch.tensor([[-1.0, -2.0, -0.5, -3.0]]),
            K=4,
        )
    )
    assert abs(out.advantages[0][0] - 0.0) < 1e-9


# ---------------------------------------------------------------------------
# Bounds and invariants
# ---------------------------------------------------------------------------


def test_arm_advantage_bounded():
    """For random Q+ inputs at K=32, every advantage is in [-1/K, 1 - 1/K]."""
    K = 32
    n_pos = 5
    sample = _make_sample(completion_len=n_pos)
    torch.manual_seed(123)
    for trial in range(5):
        # Mix positive and negative Q+ values across candidates.
        q_plus_candidates = (torch.rand(n_pos, K) - 0.5) * 4.0
        # Sampled-action Q+ matches one of the candidates per position.
        sampled_idx = torch.randint(0, K, (n_pos,))
        q_plus_sampled = torch.stack(
            [q_plus_candidates[k, sampled_idx[k]] for k in range(n_pos)]
        )
        out = arm_regret_matching_advantage_fn(
            _arm_inputs(
                [sample],
                q_plus_sampled_all=q_plus_sampled,
                q_plus_candidates=q_plus_candidates,
                K=K,
            )
        )
        for a in out.advantages[0]:
            assert -1.0 / K - 1e-6 <= a <= 1.0 - 1.0 / K + 1e-6, (
                f"trial={trial}: advantage {a} out of [-1/K, 1-1/K]"
            )


def test_arm_advantage_sum_to_zero_across_action_set():
    """Sum of advantages over the K candidates (each treated as sampled) is ~0."""
    K = 32
    sample = _make_sample(completion_len=1)
    torch.manual_seed(7)
    q_plus_candidates_row = (torch.rand(K) - 0.3) * 5.0  # mix of positive/negative

    advantages_per_candidate = []
    for i in range(K):
        out = arm_regret_matching_advantage_fn(
            _arm_inputs(
                [sample],
                q_plus_sampled_all=q_plus_candidates_row[i : i + 1].clone(),
                q_plus_candidates=q_plus_candidates_row.unsqueeze(0).clone(),
                K=K,
            )
        )
        advantages_per_candidate.append(out.advantages[0][0])

    assert abs(sum(advantages_per_candidate)) < 1e-5


# ---------------------------------------------------------------------------
# q_plus_target correctness
# ---------------------------------------------------------------------------


def test_arm_q_plus_target_cfr_plus_recurrence():
    """q_plus_target = max(0, Q+_sampled + A). Hand-construct Q+_sampled=0.5 -> A=0.3 -> 0.8."""
    # Construct candidates s.t. with K=2 and q_plus_sampled=0.5 we get A=0.3:
    #   A = 0.5 / denom - 1/2 = 0.3 -> denom = 0.5/0.8 = 0.625
    #   candidates = [0.5, 0.125] -> denom = 0.625, sampled = 0.5 -> A = 0.8 - 0.5 = 0.3
    sample = _make_sample(completion_len=1)
    out = arm_regret_matching_advantage_fn(
        _arm_inputs(
            [sample],
            q_plus_sampled_all=torch.tensor([0.5]),
            q_plus_candidates=torch.tensor([[0.5, 0.125]]),
            K=2,
        )
    )
    # Sanity-check the construction first.
    assert abs(out.advantages[0][0] - 0.3) < 1e-6
    # The CFR+ target.
    assert abs(out.q_plus_targets[0][0] - 0.8) < 1e-6


def test_arm_q_plus_target_clipped_when_negative():
    """q_plus_target = max(0, 0.1 + (-0.3)) = 0 -- the CFR+ clip distinguishes from raw CFR."""
    # K=2 with q_plus_sampled=0.1 and A=-0.3:
    #   -0.3 = 0.1/denom - 1/2 -> denom = 0.5
    #   candidates = [0.1, 0.4] -> denom = 0.5, A = 0.2 - 0.5 = -0.3
    sample = _make_sample(completion_len=1)
    out = arm_regret_matching_advantage_fn(
        _arm_inputs(
            [sample],
            q_plus_sampled_all=torch.tensor([0.1]),
            q_plus_candidates=torch.tensor([[0.1, 0.4]]),
            K=2,
        )
    )
    assert abs(out.advantages[0][0] - (-0.3)) < 1e-6
    assert abs(out.q_plus_targets[0][0] - 0.0) < 1e-6


def test_arm_q_plus_targets_length_matches_completion_length():
    """len(q_plus_targets[i]) == len(samples[i].completion_ids) for every sample."""
    s0 = _make_sample(completion_len=8)
    s1 = _make_sample(completion_len=2)
    out = arm_regret_matching_advantage_fn(_arm_inputs([s0, s1], K=4))
    assert out.q_plus_targets is not None
    assert len(out.q_plus_targets[0]) == len(s0.completion_ids)
    assert len(out.q_plus_targets[1]) == len(s1.completion_ids)


# ---------------------------------------------------------------------------
# v_target correctness
# ---------------------------------------------------------------------------


def test_arm_v_target_uses_v_target_all_not_v_all():
    """Truncated rollout: changing v_all does not change v_targets (only v_target_all does)."""
    sample = _make_sample(completion_len=3)

    out_a = arm_regret_matching_advantage_fn(
        _arm_inputs(
            [sample],
            episodic_reward=0.0,
            is_terminal=False,  # bootstrap from v_target_all[-1]
            v_all=torch.tensor([0.1, 0.2, 0.3]),
            v_target_all=torch.tensor([0.5, 0.5, 0.5]),
            K=4,
        )
    )
    out_b = arm_regret_matching_advantage_fn(
        _arm_inputs(
            [sample],
            episodic_reward=0.0,
            is_terminal=False,
            v_all=torch.tensor([100.0, 200.0, 300.0]),  # wildly different
            v_target_all=torch.tensor([0.5, 0.5, 0.5]),  # same as run a
            K=4,
        )
    )
    assert out_a.v_targets == out_b.v_targets


def test_arm_v_target_matches_n_step_return():
    """Terminal rollout: v_target[k] = gamma^(N-1-k) * episodic_reward."""
    sample = _make_sample(completion_len=4)
    gamma = 0.99
    out = arm_regret_matching_advantage_fn(
        _arm_inputs([sample], episodic_reward=1.0, is_terminal=True, gamma=gamma, K=4)
    )
    expected = [gamma**3, gamma**2, gamma**1, 1.0]
    for e, g in zip(expected, out.v_targets[0]):
        assert abs(e - g) < 1e-6


# ---------------------------------------------------------------------------
# Joint recursion across fragmented samples
# ---------------------------------------------------------------------------


def test_arm_two_samples_joint_v_target_recursion():
    """Two samples, terminal: v_target propagates correctly across the boundary."""
    s0 = _make_sample(completion_len=2)
    s1 = _make_sample(completion_len=3)
    gamma = 0.99
    out = arm_regret_matching_advantage_fn(
        _arm_inputs([s0, s1], episodic_reward=1.0, is_terminal=True, gamma=gamma, K=4)
    )
    expected_joint = [gamma**4, gamma**3, gamma**2, gamma**1, 1.0]
    got_flat = out.v_targets[0] + out.v_targets[1]
    for e, g in zip(expected_joint, got_flat):
        assert abs(e - g) < 1e-6
    assert len(out.v_targets[0]) == 2
    assert len(out.v_targets[1]) == 3


# ---------------------------------------------------------------------------
# Output shape and determinism
# ---------------------------------------------------------------------------


def test_arm_advantages_length_matches_completion_length():
    """len(advantages[i]) == len(samples[i].completion_ids)."""
    s0 = _make_sample(completion_len=7)
    s1 = _make_sample(completion_len=4)
    out = arm_regret_matching_advantage_fn(_arm_inputs([s0, s1], K=4))
    assert len(out.advantages[0]) == len(s0.completion_ids)
    assert len(out.advantages[1]) == len(s1.completion_ids)


def test_arm_q_plus_targets_populated():
    """ARM populates q_plus_targets (distinguishes from PPO's None)."""
    sample = _make_sample(completion_len=3)
    out = arm_regret_matching_advantage_fn(_arm_inputs([sample], K=4))
    assert out.q_plus_targets is not None
    assert len(out.q_plus_targets) == 1


def test_arm_deterministic_inputs_yield_deterministic_output():
    """Same inputs -> bit-identical outputs (no hidden internal state)."""
    K = 8
    sample = _make_sample(completion_len=5)
    torch.manual_seed(99)
    q_cand = (torch.rand(5, K) - 0.4) * 3.0
    q_sampled = q_cand[:, 0].clone()
    v_target = torch.rand(5)

    def run() -> PerTokenAdvantageOutputs:
        return arm_regret_matching_advantage_fn(
            ArmAdvantageInputs(
                samples=[sample],
                episodic_reward=0.7,
                is_terminal=True,
                v_all=torch.zeros(5),
                v_target_all=v_target.clone(),
                q_plus_sampled_all=q_sampled.clone(),
                q_plus_candidates=q_cand.clone(),
                gamma=0.97,
            )
        )

    out_a = run()
    out_b = run()
    assert out_a.advantages == out_b.advantages
    assert out_a.v_targets == out_b.v_targets
    assert out_a.q_plus_targets == out_b.q_plus_targets
