"""Branched dataset Phase 0: per-leg weights inside value_regression_backward.

A target with v_weight = 0 (a branch root) contributes nothing to the V leg:
changing its v_target leaves l_v and the V-head gradient unchanged, while the
same change with v_weight = 1 moves both. Q+ is unaffected either way.
"""

from __future__ import annotations

import pytest
import torch

from prime_rl.advantage_trainer.shared import value_regression_backward
from prime_rl.transport.types import AdvantageTrainingSample, DecisionPointTarget
from tests.unit.advantage_trainer.test_train_step_action import _backbone, _CharTok


def _sample(dpts):
    return AdvantageTrainingSample(
        prompt_ids=[1, 2, 3],
        prompt_mask=[False] * 3,
        completion_ids=[10, 11, 12, 13, 14, 15],
        completion_mask=[True] * 6,
        decision_point_targets=dpts,
    )


def _run(root_v_target: float, root_v_weight: float):
    backbone = _backbone(seed=0, randomize_heads=True)
    chunk = [
        _sample(
            [
                DecisionPointTarget(0, "go", 1.0, 1.0),
                DecisionPointTarget(3, "take", root_v_target, 0.5, v_weight=root_v_weight),
            ]
        )
    ]
    l_v, l_q = value_regression_backward(backbone, chunk, _CharTok(), chunk_scale=1.0, device="cpu")
    grad = backbone.v_head.linear.weight.grad.detach().clone()
    return l_v, l_q, grad


def test_masked_root_ignores_its_v_target():
    l_v_a, l_q_a, g_a = _run(root_v_target=0.0, root_v_weight=0.0)
    l_v_b, l_q_b, g_b = _run(root_v_target=50.0, root_v_weight=0.0)
    assert l_v_a == pytest.approx(l_v_b)
    assert torch.allclose(g_a, g_b)
    assert l_q_a == pytest.approx(l_q_b)


def test_unmasked_root_uses_its_v_target():
    l_v_a, _, g_a = _run(root_v_target=0.0, root_v_weight=1.0)
    l_v_b, _, g_b = _run(root_v_target=50.0, root_v_weight=1.0)
    assert l_v_b > l_v_a
    assert not torch.allclose(g_a, g_b)


def test_default_weight_equals_explicit_one():
    l_v_a, l_q_a, g_a = _run(root_v_target=0.0, root_v_weight=1.0)
    backbone = _backbone(seed=0, randomize_heads=True)
    chunk = [_sample([DecisionPointTarget(0, "go", 1.0, 1.0), DecisionPointTarget(3, "take", 0.0, 0.5)])]
    l_v_b, l_q_b = value_regression_backward(backbone, chunk, _CharTok(), chunk_scale=1.0, device="cpu")
    assert l_v_a == pytest.approx(l_v_b)
    assert l_q_a == pytest.approx(l_q_b)
    assert torch.allclose(g_a, backbone.v_head.linear.weight.grad)
