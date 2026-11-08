"""training-probe Phase 2 -- Advantage Server diagnostics (compute layer).

Tiny CPU Qwen2 backbone + char tokenizer (as test_compute_action.py). Pins:
action_advantage_fn's additive pi_rm / g outputs leave the existing outputs
untouched; the diagnostics align with the per-decision-point records and the
server's own advantage formula; the flag-off return is the plain paired list;
PPO / no-decision-point paths yield empty diagnostics under the flag.
"""

from __future__ import annotations

import math

import pytest
import torch
from transformers import Qwen2Config, Qwen2Model

from prime_rl.advantage_server.action_advantage import (
    ActionAdvantageInputs,
    action_advantage,
    action_advantage_fn,
    n_step_return_action_level,
    pi_rm,
    q_plus_target,
)
from prime_rl.advantage_server.compute import (
    _build_diagnostics_action,
    _build_per_decision_point_value_tensors,
    compute_advantages_and_targets,
    compute_advantages_and_targets_arm_action,
)
from prime_rl.configs.trainer import LoRAConfig
from prime_rl.orchestrator.advantage_server_client import DecisionPointDiagnostics
from prime_rl.orchestrator.value_networks import ValueNetworkBackbone
from prime_rl.transport.types import DecisionPoint, TrainingSample


class _CharTok:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 256 for c in text]


def _tiny_qwen2_config() -> Qwen2Config:
    return Qwen2Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=512,
    )


@pytest.fixture
def backbone() -> ValueNetworkBackbone:
    torch.manual_seed(0)
    base = Qwen2Model(_tiny_qwen2_config())
    lora = LoRAConfig(rank=8, alpha=16.0, dropout=0.0)
    bb = ValueNetworkBackbone(base, lora_config=lora, polyak_tau=0.005)
    # Non-zero heads so Q+ / V are not all zero (pi_RM not trivially uniform).
    torch.manual_seed(1)
    for head in (bb.v_head, bb.q_plus_head, bb.v_target_head):
        head.linear.weight.data.normal_()
        head.linear.bias.data.normal_()
    return bb


def _two_sample_rollout() -> list[TrainingSample]:
    """Three decision points across two samples with |A| = 2, 3, 2 (a* appended
    in the second: 'open' is not in the raw set but is in the union)."""
    s0 = TrainingSample(
        prompt_ids=[1, 2, 3],
        prompt_mask=[False, False, False],
        completion_ids=[10, 11, 12, 13, 20, 21],
        completion_mask=[True, True, False, False, True, True],
        completion_logprobs=[0.0] * 6,
        completion_temperatures=[1.0] * 6,
        decision_points=[
            DecisionPoint(0, 2, ["go", "look"], 0, pi_hat=0.5),
            DecisionPoint(4, 6, ["take", "look", "open"], 2, pi_hat=0.05),
        ],
    )
    s1 = TrainingSample(
        prompt_ids=[100, 101],
        prompt_mask=[False, False],
        completion_ids=[30, 31, 32],
        completion_mask=[True, True, True],
        completion_logprobs=[0.0] * 3,
        completion_temperatures=[1.0] * 3,
        decision_points=[DecisionPoint(0, 3, ["put", "look"], 1, pi_hat=0.9)],
    )
    return [s0, s1]


# ---------------------------------------------------------------------------
# action_advantage_fn: additive outputs
# ---------------------------------------------------------------------------


def test_outputs_backward_compatible():
    inp = ActionAdvantageInputs(
        q_plus=[[5.0, 3.0], [2.0, 9.0, 4.0], [1.0, 1.0]],
        v=[2.0, 3.0, 5.0],  # k=2: all regrets <= 0 -> uniform pi_RM
        executed_idx=[0, 1, 1],
        pi_hat_star=[0.5, 0.2, 0.7],
        rewards=[0.0, 0.0, 1.0],
        v_target=[0.2, 0.9, 0.4],
        gamma=0.9,
        n_step=2,
        phi_decay=0.8,
    )
    out = action_advantage_fn(inp)
    g = n_step_return_action_level(inp.rewards, inp.v_target, inp.gamma, inp.n_step)
    assert out.g == pytest.approx(g)
    assert out.pi_rm is not None and len(out.pi_rm) == 3
    for k in range(3):
        rm = pi_rm(inp.q_plus[k], inp.v[k])
        e = inp.executed_idx[k]
        assert out.pi_rm[k] == pytest.approx(rm)
        assert sum(out.pi_rm[k]) == pytest.approx(1.0)
        assert len(out.pi_rm[k]) == len(inp.q_plus[k])
        # the three pre-existing outputs are unchanged by the additive fields
        assert out.advantage[k] == pytest.approx(action_advantage(rm[e], inp.pi_hat_star[k]))
        assert out.v_target_out[k] == pytest.approx(g[k])
        assert out.q_plus_target_out[k] == pytest.approx(
            q_plus_target(inp.q_plus[k][e], inp.v[k], g[k], 0.8)
        )
    assert out.pi_rm[2] == pytest.approx([0.5, 0.5])


# ---------------------------------------------------------------------------
# Diagnostics assembly against the real records
# ---------------------------------------------------------------------------


def test_diagnostics_align_with_records(backbone):
    tok = _CharTok()
    samples = _two_sample_rollout()
    records = _build_per_decision_point_value_tensors(samples, backbone, tok)
    result = compute_advantages_and_targets_arm_action(
        samples, episodic_reward=1.0, is_terminal=True, backbone=backbone,
        tokenizer=tok, gamma=1.0, n_step=10, arm_phi_decay=0.5, return_diagnostics=True,
    )
    assert isinstance(result, tuple) and len(result) == 2
    paired, diags = result
    assert len(paired) == 2
    assert len(diags) == len(records) == 3
    expected_idx = [(0, 0), (0, 4), (1, 0)]
    for k, (d, r) in enumerate(zip(diags, records)):
        assert isinstance(d, DecisionPointDiagnostics)
        assert (d.sample_idx, d.response_start) == expected_idx[k]
        dp = samples[d.sample_idx].decision_points[[0, 1, 0][k]]
        n_a = len(dp.admissible_actions)
        assert len(d.q_plus) == len(d.pi_rm) == n_a
        assert d.q_plus == pytest.approx(r["q_plus"])
        assert d.v == pytest.approx(r["v"])
        assert d.v_target == pytest.approx(r["v_target"])
        assert sum(d.pi_rm) == pytest.approx(1.0, abs=1e-9)
        assert d.pi_rm == pytest.approx(pi_rm(r["q_plus"], r["v"]))
        # server's own A(a*) with the 1e-6 floor
        e = dp.executed_action_idx
        expected_adv = math.log(max(d.pi_rm[e], 1e-6)) - math.log(max(dp.pi_hat, 1e-6))
        assert d.adv_star == pytest.approx(expected_adv, abs=1e-9)
        # MC return with gamma=1, n_step >= K -> terminal reward everywhere
        assert d.g_k == pytest.approx(1.0)
        assert d.q_plus_target_star == pytest.approx(0.5 * max(0.0, d.q_plus[e] - d.v) + d.g_k)
    # the diagnostics agree with what the paired outputs carry
    llm0, adv0 = paired[0]
    assert llm0.advantages[0] == pytest.approx(diags[0].adv_star)
    assert llm0.advantages[4] == pytest.approx(diags[1].adv_star)
    assert adv0.decision_point_targets[1].q_plus_target == pytest.approx(diags[1].q_plus_target_star)
    assert adv0.decision_point_targets[1].v_target == pytest.approx(diags[1].g_k)
    assert paired[1][1].decision_point_targets[0].v_target == pytest.approx(diags[2].g_k)


def test_flag_off_returns_old_shape(backbone):
    tok = _CharTok()
    samples = _two_sample_rollout()
    plain = compute_advantages_and_targets_arm_action(
        samples, episodic_reward=1.0, is_terminal=True, backbone=backbone,
        tokenizer=tok, gamma=1.0, n_step=10,
    )
    paired, _diags = compute_advantages_and_targets_arm_action(
        samples, episodic_reward=1.0, is_terminal=True, backbone=backbone,
        tokenizer=tok, gamma=1.0, n_step=10, return_diagnostics=True,
    )
    assert isinstance(plain, list) and len(plain) == 2
    for (l1, a1), (l2, a2) in zip(plain, paired):
        assert l1.advantages == pytest.approx(l2.advantages)
        assert a1.decision_point_targets is not None
        for t1, t2 in zip(a1.decision_point_targets, a2.decision_point_targets):
            assert t1.v_target == pytest.approx(t2.v_target)
            assert t1.q_plus_target == pytest.approx(t2.q_plus_target)
            assert t1.executed_action == t2.executed_action


def test_missing_pi_hat_yields_zero_adv_star_record(backbone):
    tok = _CharTok()
    sample = TrainingSample(
        prompt_ids=[1, 2],
        prompt_mask=[False, False],
        completion_ids=[10, 11],
        completion_mask=[True, True],
        completion_logprobs=[0.0, 0.0],
        completion_temperatures=[1.0, 1.0],
        decision_points=[DecisionPoint(0, 2, ["go", "look"], 0, pi_hat=None)],
    )
    paired, diags = compute_advantages_and_targets_arm_action(
        [sample], episodic_reward=1.0, is_terminal=True, backbone=backbone,
        tokenizer=tok, gamma=1.0, n_step=10, return_diagnostics=True,
    )
    assert len(diags) == 1
    assert diags[0].adv_star == 0.0
    assert len(diags[0].q_plus) == 2 and sum(diags[0].pi_rm) == pytest.approx(1.0)
    assert diags[0].g_k == pytest.approx(1.0)
    assert all(a == 0.0 for a in paired[0][0].advantages)


def test_no_decision_points_with_flag_returns_empty_diagnostics(backbone):
    tok = _CharTok()
    sample = TrainingSample(
        prompt_ids=[1, 2],
        prompt_mask=[False, False],
        completion_ids=[10, 11],
        completion_mask=[True, True],
        completion_logprobs=[0.0, 0.0],
        completion_temperatures=[1.0, 1.0],
    )
    paired, diags = compute_advantages_and_targets_arm_action(
        [sample], episodic_reward=1.0, is_terminal=True, backbone=backbone,
        tokenizer=tok, return_diagnostics=True,
    )
    assert diags == []
    assert len(paired) == 1 and paired[0][0].advantages == [0.0, 0.0]
    plain = compute_advantages_and_targets_arm_action(
        [sample], episodic_reward=1.0, is_terminal=True, backbone=backbone, tokenizer=tok,
    )
    assert isinstance(plain, list)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def test_ppo_with_flag_returns_empty_diagnostics(backbone):
    sample = TrainingSample(
        prompt_ids=[1, 2],
        prompt_mask=[False, False],
        completion_ids=[10, 11],
        completion_mask=[True, True],
        completion_logprobs=[0.0, 0.0],
        completion_temperatures=[1.0, 1.0],
    )
    result = compute_advantages_and_targets(
        samples=[sample], episodic_reward=1.0, is_terminal=True, algorithm="ppo",
        backbone=backbone, return_diagnostics=True,
    )
    paired, diags = result
    assert diags == []
    assert len(paired) == 1 and paired[0][1].q_plus_targets is None
    plain = compute_advantages_and_targets(
        samples=[sample], episodic_reward=1.0, is_terminal=True, algorithm="ppo", backbone=backbone,
    )
    assert isinstance(plain, list) and len(plain) == 1


def test_dispatch_arm_threads_flag(backbone):
    tok = _CharTok()
    samples = _two_sample_rollout()
    paired, diags = compute_advantages_and_targets(
        samples=samples, episodic_reward=1.0, is_terminal=True, algorithm="arm",
        backbone=backbone, tokenizer=tok, n_step=10, return_diagnostics=True,
    )
    assert len(diags) == 3 and len(paired) == 2
    plain = compute_advantages_and_targets(
        samples=samples, episodic_reward=1.0, is_terminal=True, algorithm="arm",
        backbone=backbone, tokenizer=tok, n_step=10,
    )
    assert isinstance(plain, list)


# ---------------------------------------------------------------------------
# _build_diagnostics_action guards
# ---------------------------------------------------------------------------


def test_build_diagnostics_requires_pi_rm_and_lengths():
    from prime_rl.advantage_server.action_advantage import ActionAdvantageOutputs

    records = [
        {"sample_idx": 0, "response_start": 0, "q_plus": [0.1, 0.2], "v": 0.0, "v_target": 0.0},
    ]
    legacy = ActionAdvantageOutputs(advantage=[0.0], v_target_out=[0.0], q_plus_target_out=[0.0])
    with pytest.raises(ValueError):
        _build_diagnostics_action(records, legacy)
    bad_width = ActionAdvantageOutputs(
        advantage=[0.0], v_target_out=[0.0], q_plus_target_out=[0.0], pi_rm=[[1.0]], g=[0.0]
    )
    with pytest.raises(ValueError):
        _build_diagnostics_action(records, bad_width)
    ok = ActionAdvantageOutputs(
        advantage=[0.3], v_target_out=[1.0], q_plus_target_out=[1.2], pi_rm=[[0.25, 0.75]], g=[1.0]
    )
    diags = _build_diagnostics_action(records, ok)
    assert diags == [
        DecisionPointDiagnostics(
            sample_idx=0, response_start=0, q_plus=[0.1, 0.2], v=0.0, v_target=0.0,
            pi_rm=[0.25, 0.75], adv_star=0.3, g_k=1.0, q_plus_target_star=1.2,
        )
    ]
