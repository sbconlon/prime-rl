import json
from pathlib import Path

import pytest
import torch

from prime_rl.configs.orchestrator import CustomAdvantageConfig, DefaultAdvantageConfig
from prime_rl.orchestrator.advantage import (
    AdvantageInputs,
    AdvantageOutputs,
    compute_advantages,
    default_advantage_fn,
    setup_advantage_fn,
)


def test_default_advantage_fn_simple_mean():
    inputs = AdvantageInputs(
        rewards=torch.tensor([[1.0, 0.5, 0.8], [0.2, 0.9, 0.1]]),
        completion_lengths=torch.tensor([[10, 12, 8], [15, 11, 9]]),
    )
    result = default_advantage_fn(inputs)

    assert result.advantages.shape == (2, 3)
    # Check that mean is subtracted per row
    assert torch.allclose(result.advantages.mean(dim=1), torch.zeros(2), atol=1e-6)


def test_default_advantage_fn_gr3_length_shaping():
    inputs = AdvantageInputs(
        rewards=torch.tensor([[1.0, 0.5, 0.8]]),
        completion_lengths=torch.tensor([[10, 20, 10]]),
    )

    result = default_advantage_fn(inputs, length_shaping_alpha=0.33)

    expected = torch.tensor([[0.20915856, -0.25799648, 0.04883792]])
    assert torch.allclose(result.advantages, expected, atol=1e-6)
    assert torch.allclose(result.advantages.mean(dim=1), torch.zeros(1), atol=1e-6)


def test_compute_advantages_with_config():
    rewards = [1.0, 0.5, 0.8, 0.2, 0.9, 0.1]
    lengths = [10, 12, 8, 15, 11, 9]

    result = compute_advantages(rewards, lengths, samples_per_problem=3, advantage_config=DefaultAdvantageConfig())

    assert len(result) == 6
    # First 3 should sum to ~0 (mean subtracted)
    assert abs(sum(result[:3])) < 1e-5
    # Last 3 should sum to ~0
    assert abs(sum(result[3:])) < 1e-5


def test_compute_advantages_without_config():
    rewards = [1.0, 0.5, 0.8]
    lengths = [10, 12, 8]

    result = compute_advantages(rewards, lengths, samples_per_problem=3, advantage_config=None)

    # Without config, returns raw rewards
    assert result == rewards


def test_setup_advantage_fn_with_custom_config():
    config = CustomAdvantageConfig(
        import_path="tests.unit.orchestrator.test_advantage._dummy_custom_advantage",
        kwargs={"scale": 2.0},
    )
    advantage_fn = setup_advantage_fn(config)

    inputs = AdvantageInputs(
        rewards=torch.tensor([[1.0, 0.5, 0.8]]),
        completion_lengths=torch.tensor([[10, 12, 8]]),
    )

    result = advantage_fn(inputs)
    assert isinstance(result, AdvantageOutputs)
    # Dummy just multiplies rewards by scale
    assert torch.allclose(result.advantages, torch.tensor([[2.0, 1.0, 1.6]]))


def _dummy_custom_advantage(inputs: AdvantageInputs, scale: float = 1.0) -> AdvantageOutputs:
    """A simple custom advantage for testing."""
    return AdvantageOutputs(advantages=inputs.rewards * scale)


# ---------------------------------------------------------------------------
# Phase 0 - GRPO regression harness
# ---------------------------------------------------------------------------
# Layer 1 (golden-master, bit-level) and Layer 2 (property, semantic invariants)
# tests. See plan/phases/phase-00-grpo-lockdown.md and plan/stubbing-strategy.md
# section 6 for the harness design. Layer 3 is the existing
# tests/integration/test_rl.py canary - not modified here.
# ---------------------------------------------------------------------------

_FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> tuple[list[float], list[int], int, torch.Tensor]:
    """Load (rewards_flat, lengths_flat, samples_per_problem, expected_tensor)."""
    inputs = json.loads((_FIXTURES_DIR / f"{name}.json").read_text())
    samples_per_problem: int = inputs["samples_per_problem"]
    rewards_flat = [r for row in inputs["rewards"] for r in row]
    lengths_flat = [length for row in inputs["completion_lengths"] for length in row]
    expected = torch.load(_FIXTURES_DIR / f"{name}.pt", weights_only=True)
    return rewards_flat, lengths_flat, samples_per_problem, expected


# --- Layer 1: golden-master (bit-level) ------------------------------------


def test_grpo_golden_master_no_length_shaping():
    rewards_flat, lengths_flat, samples_per_problem, expected = _load_fixture("grpo_golden_master")
    config = DefaultAdvantageConfig(length_shaping_alpha=None)
    flat = compute_advantages(rewards_flat, lengths_flat, samples_per_problem, config)
    result = torch.tensor(flat, dtype=torch.float32).view(-1, samples_per_problem)
    assert torch.allclose(result, expected, atol=1e-7)


def test_grpo_golden_master_gr3():
    rewards_flat, lengths_flat, samples_per_problem, expected = _load_fixture("grpo_golden_master_gr3")
    config = DefaultAdvantageConfig(length_shaping_alpha=0.33)
    flat = compute_advantages(rewards_flat, lengths_flat, samples_per_problem, config)
    result = torch.tensor(flat, dtype=torch.float32).view(-1, samples_per_problem)
    assert torch.allclose(result, expected, atol=1e-7)


# --- Layer 2: property tests (semantic invariants) -------------------------


@pytest.mark.parametrize(
    "g,s,seed",
    [
        (2, 2, 0),
        (2, 8, 1),
        (4, 4, 2),
        (8, 2, 3),
        (8, 8, 4),
    ],
)
def test_grpo_advantages_sum_to_zero_per_group(g: int, s: int, seed: int):
    """For any [G, S] rewards tensor, GRPO advantages sum to ~0 along the S axis."""
    torch.manual_seed(seed)
    rewards = torch.rand(g, s)
    inputs = AdvantageInputs(
        rewards=rewards,
        completion_lengths=torch.full((g, s), 100, dtype=torch.long),
    )
    result = default_advantage_fn(inputs, length_shaping_alpha=None)
    assert result.advantages.sum(dim=1).abs().max().item() < 1e-5


def test_grpo_advantage_ordering_matches_reward_ordering():
    """For each problem group, argmax/argmin of advantages match those of rewards."""
    torch.manual_seed(42)
    rewards = torch.rand(6, 8)
    inputs = AdvantageInputs(
        rewards=rewards,
        completion_lengths=torch.full((6, 8), 100, dtype=torch.long),
    )
    result = default_advantage_fn(inputs, length_shaping_alpha=None)
    for g in range(rewards.shape[0]):
        assert result.advantages[g].argmax().item() == rewards[g].argmax().item()
        assert result.advantages[g].argmin().item() == rewards[g].argmin().item()


def test_grpo_uniform_rewards_give_zero_advantages():
    """Every reward in a problem equal -> all advantages exactly zero."""
    inputs = AdvantageInputs(
        rewards=torch.tensor([[0.5, 0.5, 0.5], [0.7, 0.7, 0.7]]),
        completion_lengths=torch.full((2, 3), 100, dtype=torch.long),
    )
    result = default_advantage_fn(inputs, length_shaping_alpha=None)
    assert result.advantages.abs().max().item() < 1e-7


@pytest.mark.parametrize(
    "a,b",
    [
        (0.0, 1.0),
        (1.0, 0.0),
        (0.3, 0.7),
        (0.5, 0.5),
        (-0.2, 0.8),
    ],
)
def test_grpo_two_sample_closed_form(a: float, b: float):
    """For [[a, b]] rewards, advantages must equal [[(a-b)/2, (b-a)/2]] exactly."""
    inputs = AdvantageInputs(
        rewards=torch.tensor([[a, b]]),
        completion_lengths=torch.tensor([[100, 100]]),
    )
    result = default_advantage_fn(inputs, length_shaping_alpha=None)
    expected = torch.tensor([[(a - b) / 2.0, (b - a) / 2.0]])
    assert torch.allclose(result.advantages, expected, atol=1e-7)
