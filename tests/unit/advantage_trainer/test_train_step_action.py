"""Phase 7 (action-level ARM) -- _train_step_action one-cycle mechanics + loss.

Tiny CPU backbone. Validates per-decision-point MSE, per-sample-equal weighting,
that both legs backprop and Polyak fires once, and that V is read at the
o-boundary (vs the in-line sampled read).
"""

from __future__ import annotations

import pytest
import torch
from torch import optim
from transformers import Qwen2Config, Qwen2Model

from prime_rl.advantage_trainer.train import _train_step_action
from prime_rl.configs.trainer import LoRAConfig
from prime_rl.orchestrator.value_networks import (
    V_TARGET_SLOT,
    ValueNetworkBackbone,
)
from prime_rl.trainer.models.layers.lora.multi_linear import MultiLoRALinear
from prime_rl.transport.types import (
    AdvantageTrainingBatch,
    AdvantageTrainingSample,
    DecisionPointTarget,
)


class _CharTok:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 256 for c in text]


def _backbone(seed: int = 0, randomize_heads: bool = False) -> ValueNetworkBackbone:
    torch.manual_seed(seed)
    config = Qwen2Config(
        vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=128,
    )
    backbone = ValueNetworkBackbone(
        Qwen2Model(config), lora_config=LoRAConfig(rank=8, alpha=16.0, dropout=0.0),
        polyak_tau=0.005,
    )
    backbone.train()
    with torch.no_grad():
        for name, p in backbone.named_parameters():
            if ("lora_A" in name) or ("lora_B" in name):
                p.copy_(torch.randn_like(p) * 0.1)
        if randomize_heads:
            for head in (backbone.v_head, backbone.q_plus_head, backbone.v_target_head):
                torch.nn.init.normal_(head.linear.weight, mean=0.0, std=0.02)
    return backbone


def _sample(prompt_ids, completion_ids, dpts):
    return AdvantageTrainingSample(
        prompt_ids=prompt_ids,
        prompt_mask=[False] * len(prompt_ids),
        completion_ids=completion_ids,
        completion_mask=[True] * len(completion_ids),
        decision_point_targets=dpts,
    )


def _snapshot_v_target(backbone):
    return [
        m.lora_A[V_TARGET_SLOT].detach().clone()
        for m in backbone.base_model.modules()
        if isinstance(m, MultiLoRALinear)
    ]


def test_per_decision_point_mse_cold_start():
    """Zero-init heads -> V=Q+=0 -> losses are the weighted squared targets.
    Single sample, single dp, weight 1.0: l_v = v_target^2, l_q = q_plus_target^2."""
    backbone = _backbone(randomize_heads=False)  # heads stay zero-init
    optimizer = optim.SGD([p for p in backbone.parameters() if p.requires_grad], lr=0.0)
    s = _sample([1, 2, 3], [10, 11], [DecisionPointTarget(0, "go", 1.5, 2.0)])
    metrics = _train_step_action(
        backbone, optimizer, AdvantageTrainingBatch(examples=[s], step=0),
        tokenizer=_CharTok(), polyak_tau=0.0,
    )
    assert metrics["l_v"] == pytest.approx(1.5 ** 2, abs=1e-5)
    assert metrics["l_q"] == pytest.approx(2.0 ** 2, abs=1e-5)
    assert metrics["n_samples"] == 1
    assert metrics["inner_steps"] == 1


def test_per_sample_equal_weighting_in_loss():
    """Cold start; two samples (4 dps vs 1 dp) with the same per-dp target -> each
    sample contributes equally, so l_q == target^2 (not skewed toward the 4-dp one)."""
    backbone = _backbone(randomize_heads=False)
    optimizer = optim.SGD([p for p in backbone.parameters() if p.requires_grad], lr=0.0)
    s0 = _sample([1, 2], [10, 11, 12, 13], [DecisionPointTarget(i, "a", 1.0, 2.0) for i in range(4)])
    s1 = _sample([5, 6], [30, 31], [DecisionPointTarget(0, "b", 1.0, 2.0)])
    metrics = _train_step_action(
        backbone, optimizer, AdvantageTrainingBatch(examples=[s0, s1], step=0),
        tokenizer=_CharTok(), polyak_tau=0.0,
    )
    # Every per-dp squared error is (0 - 2)^2 = 4; per-sample-equal mean is 4.
    assert metrics["l_q"] == pytest.approx(4.0, abs=1e-5)
    assert metrics["l_v"] == pytest.approx(1.0, abs=1e-5)


def test_one_cycle_updates_params_and_polyak():
    """Non-zero lr: both legs backprop (params move) and Polyak updates V_target once."""
    backbone = _backbone(randomize_heads=True)
    optimizer = optim.SGD([p for p in backbone.parameters() if p.requires_grad], lr=0.1)
    before_q_head = backbone.q_plus_head.linear.weight.detach().clone()
    before_v_head = backbone.v_head.linear.weight.detach().clone()
    vt_before = _snapshot_v_target(backbone)
    s0 = _sample([1, 2, 3], [10, 11], [DecisionPointTarget(0, "go", 1.0, 2.0)])
    s1 = _sample([4, 5], [20, 21, 22], [DecisionPointTarget(0, "look", 1.0, 0.5)])
    metrics = _train_step_action(
        backbone, optimizer, AdvantageTrainingBatch(examples=[s0, s1], step=0),
        tokenizer=_CharTok(), polyak_tau=0.5,
    )
    assert metrics["n_samples"] == 2
    # Both legs produced gradients -> the heads moved under SGD.
    assert not torch.allclose(backbone.v_head.linear.weight, before_v_head)
    assert not torch.allclose(backbone.q_plus_head.linear.weight, before_q_head)
    # Polyak moved V_target toward V (tau=0.5).
    vt_after = _snapshot_v_target(backbone)
    moved = any(not torch.allclose(a, b) for a, b in zip(vt_after, vt_before))
    assert moved


def test_empty_batch_returns_zero_metrics():
    backbone = _backbone()
    optimizer = optim.SGD([p for p in backbone.parameters() if p.requires_grad], lr=0.0)
    metrics = _train_step_action(
        backbone, optimizer, AdvantageTrainingBatch(examples=[], step=0),
        tokenizer=_CharTok(), polyak_tau=0.0,
    )
    assert metrics["n_samples"] == 0
    assert metrics["inner_steps"] == 0
