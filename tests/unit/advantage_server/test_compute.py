"""Phase 6a -- compute_advantages_and_targets_ppo / _arm tests.

Exercises the Advantage Server's compute path on a tiny Qwen2 ValueNetworkBackbone
(no real model download). With zero-initialized adapters and value heads, the
iteration-0 regime is deterministic and predictable, which makes assertions
tight.

Tests use a 2-layer Qwen2 with hidden_size=64, runs in <2s on CPU.
"""

from __future__ import annotations

import math

import pytest
import torch
from transformers import Qwen2Config, Qwen2Model

from prime_rl.advantage_server.compute import (
    compute_advantages_and_targets,
    compute_advantages_and_targets_arm,
    compute_advantages_and_targets_ppo,
)
from prime_rl.configs.trainer import LoRAConfig
from prime_rl.orchestrator.value_networks import ValueNetworkBackbone
from prime_rl.transport.types import AdvantageTrainingSample, TrainingSample


def _tiny_qwen2_config() -> Qwen2Config:
    return Qwen2Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
    )


@pytest.fixture
def tiny_backbone() -> ValueNetworkBackbone:
    torch.manual_seed(0)
    base = Qwen2Model(_tiny_qwen2_config())
    return ValueNetworkBackbone(
        base, lora_config=LoRAConfig(rank=8, alpha=16.0, dropout=0.0), polyak_tau=0.005
    )


def _make_sample(
    completion_len: int,
    *,
    completion_mask: list[bool] | None = None,
    completion_top_k_token_ids: list[list[int]] | None = None,
    prompt_ids: list[int] | None = None,
) -> TrainingSample:
    if prompt_ids is None:
        prompt_ids = [1, 2]
    if completion_mask is None:
        completion_mask = [True] * completion_len
    completion_ids = list(range(10, 10 + completion_len))
    return TrainingSample(
        prompt_ids=prompt_ids,
        prompt_mask=[False] * len(prompt_ids),
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        completion_logprobs=[0.0] * completion_len,
        completion_temperatures=[1.0] * completion_len,
        completion_top_k_token_ids=completion_top_k_token_ids,
    )


# ---------------------------------------------------------------------------
# PPO compute
# ---------------------------------------------------------------------------


def test_compute_ppo_iteration_zero_returns_zero_advantages(
    tiny_backbone: ValueNetworkBackbone,
):
    """At iteration 0 (zero-init backbone), V outputs 0 everywhere; PPO GAE
    with V=V_target=0 gives delta_k = r_k. Only the last position carries
    the episodic reward, so advantages = (gamma*lam)^(N-1-k) * episodic_reward.
    """
    sample = _make_sample(completion_len=3)
    paired = compute_advantages_and_targets_ppo(
        samples=[sample],
        episodic_reward=1.0,
        is_terminal=True,
        backbone=tiny_backbone,
        gamma=0.99,
        lam=0.95,
    )
    assert len(paired) == 1
    new_llm_sample, adv_sample = paired[0]
    # Per-token advantages = (gamma*lam)^(N-1-k) * episodic_reward at mask=True positions.
    gamma, lam = 0.99, 0.95
    expected = [(gamma * lam) ** (2 - k) * 1.0 for k in range(3)]
    for got, exp in zip(new_llm_sample.advantages, expected):
        assert abs(got - exp) < 1e-5, f"advantage {got} != expected {exp}"
    # v_targets = A + V = A + 0 = A (same shape).
    for got, exp in zip(adv_sample.v_targets, expected):
        assert abs(got - exp) < 1e-5
    # PPO has no q_plus_targets.
    assert adv_sample.q_plus_targets is None


def test_compute_ppo_paired_output_carries_prompt_completion_verbatim(
    tiny_backbone: ValueNetworkBackbone,
):
    """The paired LLMTrainingSample/AdvantageTrainingSample must share
    prompt/completion structure verbatim with the input sample."""
    sample = _make_sample(completion_len=2)
    paired = compute_advantages_and_targets_ppo(
        samples=[sample],
        episodic_reward=1.0,
        is_terminal=True,
        backbone=tiny_backbone,
    )
    new_llm_sample, adv_sample = paired[0]
    assert new_llm_sample.prompt_ids == sample.prompt_ids
    assert new_llm_sample.prompt_mask == sample.prompt_mask
    assert new_llm_sample.completion_ids == sample.completion_ids
    assert new_llm_sample.completion_mask == sample.completion_mask
    assert adv_sample.prompt_ids == sample.prompt_ids
    assert adv_sample.prompt_mask == sample.prompt_mask
    assert adv_sample.completion_ids == sample.completion_ids
    assert adv_sample.completion_mask == sample.completion_mask
    # advantages now populated; was None on input.
    assert new_llm_sample.advantages is not None
    assert len(new_llm_sample.advantages) == len(sample.completion_ids)


def test_compute_ppo_truncated_rollout_no_terminal_reward(
    tiny_backbone: ValueNetworkBackbone,
):
    """Truncated rollouts: no episodic reward; advantages should be near-zero
    with a zero-init backbone (V=V_target=0 everywhere -> delta=0 -> A=0).
    Final-position bootstrap from V_target[-1]=0 keeps the recursion at zero.
    """
    sample = _make_sample(completion_len=3)
    paired = compute_advantages_and_targets_ppo(
        samples=[sample],
        episodic_reward=0.0,
        is_terminal=False,
        backbone=tiny_backbone,
    )
    new_llm_sample, adv_sample = paired[0]
    for a in new_llm_sample.advantages:
        assert abs(a) < 1e-6
    for vt in adv_sample.v_targets:
        assert abs(vt) < 1e-6


def test_compute_ppo_two_samples_joint_recursion(
    tiny_backbone: ValueNetworkBackbone,
):
    """Two-sample fragmented rollout: GAE recursion runs across the boundary.

    With zero-init backbone + episodic_reward at the last position of the last
    sample, the joint advantages match the closed form (gamma*lam)^(N-1-k) * r,
    where N is the total active positions across BOTH samples. If the recursion
    were independent per-sample, sample 0's advantages would all be zero (no
    terminal reward) and only sample 1's would be non-zero.
    """
    s0 = _make_sample(completion_len=2)
    s1 = _make_sample(completion_len=3, prompt_ids=[3, 4])
    paired = compute_advantages_and_targets_ppo(
        samples=[s0, s1],
        episodic_reward=1.0,
        is_terminal=True,
        backbone=tiny_backbone,
        gamma=0.99,
        lam=0.95,
    )
    flat = [a for _, _ in [paired[0]] for a in paired[0][0].advantages] + [
        a for a in paired[1][0].advantages
    ]
    # Joint length 5; expected = (gamma*lam)^(4-k) * 1.0 for k in 0..4
    gamma, lam = 0.99, 0.95
    expected = [(gamma * lam) ** (4 - k) for k in range(5)]
    for got, exp in zip(flat, expected):
        assert abs(got - exp) < 1e-5
    # If we'd computed independently, sample 0 would have all zeros (no terminal).
    # Confirm the joint computation gives non-zero for sample 0's positions.
    assert all(abs(a) > 1e-3 for a in paired[0][0].advantages)


# ---------------------------------------------------------------------------
# ARM compute
# ---------------------------------------------------------------------------


def test_compute_arm_iteration_zero_triggers_cold_start(
    tiny_backbone: ValueNetworkBackbone,
):
    """At iteration 0: Q+ outputs 0 everywhere, so the regret-matching denominator
    is 0, the cold-start branch fires, and advantages are all zero.

    q_plus_targets follow the CFR+ formula phi + g; with phi=0 (Q+=V=0) and
    g=n-step return, q_plus_targets equal the n-step return (= v_targets).
    """
    K = 4
    completion_len = 3
    sample = _make_sample(
        completion_len=completion_len,
        completion_top_k_token_ids=[
            [10 + j for j in range(K)] for j in range(completion_len)
        ],
    )
    paired = compute_advantages_and_targets_arm(
        samples=[sample],
        episodic_reward=1.0,
        is_terminal=True,
        backbone=tiny_backbone,
        gamma=0.99,
        n_step=5,
    )
    new_llm_sample, adv_sample = paired[0]
    # Cold-start: advantages all zero.
    for a in new_llm_sample.advantages:
        assert abs(a) < 1e-6
    # ARM has q_plus_targets populated.
    assert adv_sample.q_plus_targets is not None
    # phi = max(0, 0 - 0) = 0, so q_plus_target = phi + g = g (the n-step return).
    # With n_step >= N_active = 3, g_k = gamma^(N-1-k) * episodic_reward (terminal).
    gamma = 0.99
    expected_g = [gamma ** (2 - k) for k in range(3)]
    for got, exp in zip(adv_sample.q_plus_targets, expected_g):
        assert abs(got - exp) < 1e-5
    # v_targets equal g_k (same n-step return).
    for got, exp in zip(adv_sample.v_targets, expected_g):
        assert abs(got - exp) < 1e-5


def test_compute_arm_requires_completion_top_k_token_ids(
    tiny_backbone: ValueNetworkBackbone,
):
    """ARM raises a clear error when the sample lacks completion_top_k_token_ids
    (Phase 5 toggle was off)."""
    sample = _make_sample(completion_len=3, completion_top_k_token_ids=None)
    with pytest.raises(ValueError, match="completion_top_k_token_ids"):
        compute_advantages_and_targets_arm(
            samples=[sample],
            episodic_reward=1.0,
            is_terminal=True,
            backbone=tiny_backbone,
        )


# ---------------------------------------------------------------------------
# Dispatch helper
# ---------------------------------------------------------------------------


def test_dispatch_routes_to_correct_algorithm(tiny_backbone: ValueNetworkBackbone):
    """compute_advantages_and_targets dispatches based on `algorithm`."""
    K = 4
    sample = _make_sample(
        completion_len=2,
        completion_top_k_token_ids=[[10 + j for j in range(K)] for _ in range(2)],
    )
    out_ppo = compute_advantages_and_targets(
        samples=[sample],
        episodic_reward=1.0,
        is_terminal=True,
        algorithm="ppo",
        backbone=tiny_backbone,
    )
    out_arm = compute_advantages_and_targets(
        samples=[sample],
        episodic_reward=1.0,
        is_terminal=True,
        algorithm="arm",
        backbone=tiny_backbone,
    )
    # PPO produces no q_plus_targets, ARM does.
    assert out_ppo[0][1].q_plus_targets is None
    assert out_arm[0][1].q_plus_targets is not None


def test_dispatch_unknown_algorithm_raises(tiny_backbone: ValueNetworkBackbone):
    sample = _make_sample(completion_len=1)
    with pytest.raises(ValueError, match="Unsupported algorithm"):
        compute_advantages_and_targets(
            samples=[sample],
            episodic_reward=1.0,
            is_terminal=True,
            algorithm="grpo",  # type: ignore[arg-type]
            backbone=tiny_backbone,
        )
