"""Phase 6 (action-level ARM) -- Advantage Server compute integration.

Tiny CPU Qwen2 backbone + a char tokenizer. Validates the V-at-o-boundary read,
Q+-over-admissible (bare o, no reasoning), per-rollout assembly across split
samples (terminal-reward propagation), the per-token advantage broadcast, the
per-decision-point target population, and dispatcher routing.
"""

from __future__ import annotations

import pytest
import torch
from transformers import Qwen2Config, Qwen2Model

from prime_rl.advantage_server.action_advantage import tokenize_action
from prime_rl.advantage_server.compute import (
    _build_per_decision_point_value_tensors,
    compute_advantages_and_targets,
    compute_advantages_and_targets_arm_action,
)
from prime_rl.configs.trainer import LoRAConfig
from prime_rl.orchestrator.value_networks import ValueNetworkBackbone
from prime_rl.transport.types import DecisionPoint, TrainingSample


class _CharTok:
    """Deterministic char tokenizer; ids stay < the tiny model's vocab (256)."""

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
    return ValueNetworkBackbone(base, lora_config=lora, polyak_tau=0.005)


def _nonzero_heads(backbone: ValueNetworkBackbone) -> None:
    torch.manual_seed(1)
    for head in (backbone.v_head, backbone.q_plus_head, backbone.v_target_head):
        head.linear.weight.data.normal_()
        head.linear.bias.data.normal_()


def _merged_two_turn_sample() -> TrainingSample:
    # prompt [1,2,3]; turn0 resp [10,11] (start 0,end 2); bridge [12,13] (mask F);
    # turn1 resp [20,21] (start 4,end 6).
    return TrainingSample(
        prompt_ids=[1, 2, 3],
        prompt_mask=[False, False, False],
        completion_ids=[10, 11, 12, 13, 20, 21],
        completion_mask=[True, True, False, False, True, True],
        completion_logprobs=[0.0] * 6,
        completion_temperatures=[1.0] * 6,
        decision_points=[
            DecisionPoint(0, 2, ["go", "look"], 0, pi_hat=0.5),
            DecisionPoint(4, 6, ["take", "look"], 1, pi_hat=0.3),
        ],
    )


def test_v_read_at_o_boundary(backbone):
    _nonzero_heads(backbone)
    tok = _CharTok()
    sample = _merged_two_turn_sample()
    records = _build_per_decision_point_value_tensors([sample], backbone, tok)
    assert len(records) == 2
    with torch.no_grad():
        for r in records:
            o_ids = torch.tensor(
                [sample.prompt_ids + sample.completion_ids[: r["response_start"]]], dtype=torch.long
            )
            ref_v = float(backbone.forward_v(o_ids)[0])
        # dp1 explicitly: o = prompt + completion[:4]
        o1 = torch.tensor([sample.prompt_ids + sample.completion_ids[:4]], dtype=torch.long)
        ref_v1 = float(backbone.forward_v(o1)[0])
    assert records[1]["v"] == pytest.approx(ref_v1, abs=1e-4)


def test_q_plus_over_admissible(backbone):
    _nonzero_heads(backbone)
    tok = _CharTok()
    sample = _merged_two_turn_sample()
    records = _build_per_decision_point_value_tensors([sample], backbone, tok)
    r0 = records[0]
    assert len(r0["q_plus"]) == 2  # one per admissible action
    with torch.no_grad():
        o0 = torch.tensor([sample.prompt_ids], dtype=torch.long)  # response_start 0 -> bare prompt
        for i, a in enumerate(["go", "look"]):
            a_ids = torch.tensor([tokenize_action(tok, a)], dtype=torch.long)
            ref = float(backbone.forward_q_plus_action(o0, a_ids)[0])
            assert r0["q_plus"][i] == pytest.approx(ref, abs=1e-4)


def test_per_rollout_assembly_across_samples(backbone):
    """Decision points split across two samples form one episode sequence; the
    terminal reward propagates across the boundary (MC return -> every g_k == 1.0)."""
    tok = _CharTok()
    s0 = TrainingSample(
        prompt_ids=[1, 2],
        prompt_mask=[False, False],
        completion_ids=[10, 11],
        completion_mask=[True, True],
        completion_logprobs=[0.0, 0.0],
        completion_temperatures=[1.0, 1.0],
        decision_points=[DecisionPoint(0, 2, ["go", "look"], 0, pi_hat=0.5)],
    )
    s1 = TrainingSample(
        prompt_ids=[100, 101, 102],
        prompt_mask=[False, False, False],
        completion_ids=[30, 31],
        completion_mask=[True, True],
        completion_logprobs=[0.0, 0.0],
        completion_temperatures=[1.0, 1.0],
        decision_points=[DecisionPoint(0, 2, ["take", "look"], 1, pi_hat=0.3)],
    )
    paired = compute_advantages_and_targets_arm_action(
        [s0, s1], episodic_reward=1.0, is_terminal=True, backbone=backbone,
        tokenizer=tok, gamma=1.0, n_step=10,
    )
    assert len(paired) == 2
    # Both decision points (across the split) get g_k == 1.0 (terminal reward MC).
    dpt0 = paired[0][1].decision_point_targets
    dpt1 = paired[1][1].decision_point_targets
    assert dpt0 is not None and len(dpt0) == 1 and dpt0[0].v_target == pytest.approx(1.0)
    assert dpt1 is not None and len(dpt1) == 1 and dpt1[0].v_target == pytest.approx(1.0)


def test_advantage_broadcast(backbone):
    tok = _CharTok()
    sample = _merged_two_turn_sample()
    paired = compute_advantages_and_targets_arm_action(
        [sample], episodic_reward=1.0, is_terminal=True, backbone=backbone,
        tokenizer=tok, gamma=1.0, n_step=10,
    )
    llm, _adv = paired[0]
    advs = llm.advantages
    assert advs is not None and len(advs) == 6
    # dp0 broadcasts across [0,2); dp1 across [4,6); bridge [2,4) stays 0.
    assert advs[0] == advs[1]
    assert advs[4] == advs[5]
    assert advs[2] == 0.0 and advs[3] == 0.0


def test_decision_point_targets_populated(backbone):
    tok = _CharTok()
    sample = _merged_two_turn_sample()
    paired = compute_advantages_and_targets_arm_action(
        [sample], episodic_reward=1.0, is_terminal=True, backbone=backbone,
        tokenizer=tok, gamma=1.0, n_step=10,
    )
    _llm, adv = paired[0]
    assert adv.v_targets is None and adv.q_plus_targets is None  # not per-token for action-level
    assert adv.decision_point_targets is not None
    assert len(adv.decision_point_targets) == 2
    assert adv.decision_point_targets[0].executed_action == "go"
    assert adv.decision_point_targets[1].executed_action == "look"
    assert adv.decision_point_targets[0].response_start == 0
    assert adv.decision_point_targets[1].response_start == 4


def test_dispatch_arm_routes_action(backbone):
    tok = _CharTok()
    sample = _merged_two_turn_sample()
    paired = compute_advantages_and_targets(
        samples=[sample], episodic_reward=1.0, is_terminal=True, algorithm="arm",
        backbone=backbone, tokenizer=tok, n_step=10,
    )
    # The action path is identified by per-decision-point targets.
    assert paired[0][1].decision_point_targets is not None


def test_dispatch_arm_without_tokenizer_raises(backbone):
    sample = _merged_two_turn_sample()
    with pytest.raises(ValueError):
        compute_advantages_and_targets(
            samples=[sample], episodic_reward=1.0, is_terminal=True, algorithm="arm",
            backbone=backbone, tokenizer=None, n_step=10,
        )


def test_missing_pi_hat_zeros_advantage(backbone):
    """A decision point with pi_hat=None (e.g. error turn) gets advantage 0 but
    still produces value targets."""
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
    paired = compute_advantages_and_targets_arm_action(
        [sample], episodic_reward=1.0, is_terminal=True, backbone=backbone,
        tokenizer=tok, gamma=1.0, n_step=10,
    )
    llm, adv = paired[0]
    assert all(a == 0.0 for a in (llm.advantages or []))
    assert adv.decision_point_targets is not None
    assert adv.decision_point_targets[0].v_target == pytest.approx(1.0)
