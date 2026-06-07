"""Phase 7 (action-level ARM) -- prepare_action_advantage_samples.

Validates the trajectory ids, the o-boundary V positions, the per-decision-point
o/a token lists, the targets, and the per-sample-equal weights -- all built from
AdvantageTrainingSample.decision_point_targets alone.
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


def test_prepare_decision_points_single_sample():
    s = _sample(
        [1, 2, 3],
        [10, 11, 12, 13, 20, 21],
        [
            DecisionPointTarget(0, "go", 1.0, 0.5),
            DecisionPointTarget(4, "look", 1.0, 0.2),
        ],
    )
    tok = _CharTok()
    prep = prepare_action_advantage_samples([s], tok)
    assert prep.n_samples == 1
    assert prep.trajectory_input_ids.tolist() == [[1, 2, 3, 10, 11, 12, 13, 20, 21]]
    # o-boundary: prompt_len(3) + response_start - 1 -> 2 and 6
    assert prep.dp_v_pos.tolist() == [2, 6]
    assert prep.dp_sample_idx.tolist() == [0, 0]
    # o = prompt + completion[:response_start]
    assert prep.q_plus_obs_ids[0] == [1, 2, 3]
    assert prep.q_plus_obs_ids[1] == [1, 2, 3, 10, 11, 12, 13]
    assert prep.q_plus_action_ids[0] == tok.encode("<action>go</action>")
    assert prep.v_targets.tolist() == pytest.approx([1.0, 1.0])
    assert prep.q_plus_targets.tolist() == pytest.approx([0.5, 0.2])


def test_prepare_per_sample_equal_weights():
    """A 4-decision-point sample and a 1-decision-point sample each get total
    weight 0.5 (per-sample-equal: 1/(S * c_s) per dp)."""
    s0 = _sample(
        [1, 2],
        [10, 11, 12, 13],
        [DecisionPointTarget(i, "a", 1.0, 0.1) for i in range(4)],
    )
    s1 = _sample([5, 6], [30, 31], [DecisionPointTarget(0, "b", 1.0, 0.2)])
    prep = prepare_action_advantage_samples([s0, s1], _CharTok())
    w = prep.weights.tolist()
    idx = prep.dp_sample_idx.tolist()
    w_s0 = sum(wi for wi, si in zip(w, idx) if si == 0)
    w_s1 = sum(wi for wi, si in zip(w, idx) if si == 1)
    assert w_s0 == pytest.approx(0.5)
    assert w_s1 == pytest.approx(0.5)
    # Within s0 each of the 4 dps weighs 1/(2*4) = 0.125.
    assert all(wi == pytest.approx(0.125) for wi, si in zip(w, idx) if si == 0)


def test_prepare_skips_samples_without_targets():
    s0 = _sample([1, 2], [10, 11], [DecisionPointTarget(0, "go", 1.0, 0.5)])
    s1 = AdvantageTrainingSample(
        prompt_ids=[5, 6],
        prompt_mask=[False, False],
        completion_ids=[30, 31],
        completion_mask=[True, True],
        decision_point_targets=None,
    )
    prep = prepare_action_advantage_samples([s0, s1], _CharTok())
    assert prep.n_samples == 2  # K counts all chunk samples
    assert prep.dp_sample_idx.tolist() == [0]  # only s0 contributes a decision point
    # s1 still occupies row 1 of the trajectory tensor (right-padded).
    assert prep.trajectory_input_ids.shape[0] == 2
