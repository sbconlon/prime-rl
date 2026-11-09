"""Branched dataset Phase 0: per-leg weights out of prepare_action_advantage_samples.

v_weights / q_weights equal the per-sample-equal weights when every target has
default weights (parity with the pre-change trainer), and are zeroed exactly where
a target masks a leg (a branch root: v_weight = 0, q_weight = 1).
"""

from __future__ import annotations

import pytest

from prime_rl.advantage_trainer.data import prepare_action_advantage_samples
from prime_rl.transport.types import AdvantageTrainingSample, DecisionPointTarget


class _CharTok:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 256 for c in text]


def _sample(prompt_ids, completion_ids, dpts):
    return AdvantageTrainingSample(
        prompt_ids=prompt_ids,
        prompt_mask=[False] * len(prompt_ids),
        completion_ids=completion_ids,
        completion_mask=[True] * len(completion_ids),
        decision_point_targets=dpts,
    )


def test_default_weights_parity():
    s0 = _sample([1, 2], [10, 11, 12, 13], [DecisionPointTarget(0, "a", 1.0, 1.0), DecisionPointTarget(2, "b", 1.0, 1.0)])
    s1 = _sample([1, 2], [10, 11], [DecisionPointTarget(0, "c", 0.0, 0.0)])
    prep = prepare_action_advantage_samples([s0, s1], _CharTok())
    assert prep.v_weights.tolist() == pytest.approx(prep.weights.tolist())
    assert prep.q_weights.tolist() == pytest.approx(prep.weights.tolist())
    # per-sample-equal: 0.25, 0.25, 0.5
    assert prep.weights.tolist() == pytest.approx([0.25, 0.25, 0.5])


def test_root_masks_v_leg_only():
    root = DecisionPointTarget(2, "b", 1.0, 1.0, v_weight=0.0, q_weight=1.0)
    s0 = _sample([1, 2], [10, 11, 12, 13], [DecisionPointTarget(0, "a", 1.0, 1.0), root])
    prep = prepare_action_advantage_samples([s0], _CharTok())
    assert prep.weights.tolist() == pytest.approx([0.5, 0.5])
    assert prep.v_weights.tolist() == pytest.approx([0.5, 0.0])
    assert prep.q_weights.tolist() == pytest.approx([0.5, 0.5])


def test_q_weight_scales_q_leg_only():
    t = DecisionPointTarget(0, "a", 1.0, 1.0, v_weight=1.0, q_weight=2.0)
    prep = prepare_action_advantage_samples([_sample([1], [10, 11], [t])], _CharTok())
    assert prep.v_weights.tolist() == pytest.approx([1.0])
    assert prep.q_weights.tolist() == pytest.approx([2.0])
