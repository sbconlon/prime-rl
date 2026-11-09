"""Branched dataset Phase 3: build_warmstart_samples with label_from_dp / root_dp
(plan branched-dataset D-10). Defaults reproduce the single-action output
byte-for-byte; the pre-root prefix gets no targets; the root masks V only; the
joint return still crosses a truncation split.
"""

import msgspec
import pytest

from prime_rl.advantage_server.action_advantage import n_step_return_action_level
from prime_rl.transport.types import (
    AdvantageTrainingSample,
    DecisionPoint,
    DecisionPointTarget,
    TrainingSample,
)
from prime_rl.value_warmstart.targets import build_warmstart_samples


def _dp(response_start, response_end, admissible, executed_idx):
    return DecisionPoint(
        response_start=response_start,
        response_end=response_end,
        admissible_actions=list(admissible),
        executed_action_idx=executed_idx,
    )


def _sample(decision_points):
    clen = max((dp.response_end for dp in decision_points), default=0)
    return TrainingSample(
        prompt_ids=[1, 2],
        prompt_mask=[False, False],
        completion_ids=list(range(10, 10 + clen)),
        completion_mask=[True] * clen,
        completion_logprobs=[0.0] * clen,
        completion_temperatures=[1.0] * clen,
        decision_points=decision_points,
    )


def _legacy_build(samples, episodic_reward, is_terminal, *, gamma=1.0, n_step=None):
    """The pre-Phase-3 function body, kept verbatim as the parity oracle."""
    dp_owner = [i for i, s in enumerate(samples) for _ in s.decision_points]
    total = len(dp_owner)
    rewards = [0.0] * total
    if is_terminal:
        rewards[-1] = float(episodic_reward)
    n = n_step if n_step is not None else total
    g = n_step_return_action_level(rewards, [0.0] * total, gamma, n)
    per_sample = [[] for _ in samples]
    for i, g_k in zip(dp_owner, g):
        per_sample[i].append(g_k)
    out = []
    for s, returns in zip(samples, per_sample):
        targets = [
            DecisionPointTarget(
                response_start=dp.response_start,
                executed_action=dp.admissible_actions[dp.executed_action_idx],
                v_target=g_k,
                q_plus_target=g_k,
            )
            for dp, g_k in zip(s.decision_points, returns)
        ]
        out.append(
            AdvantageTrainingSample(
                prompt_ids=s.prompt_ids,
                prompt_mask=s.prompt_mask,
                completion_ids=s.completion_ids,
                completion_mask=s.completion_mask,
                decision_point_targets=targets,
            )
        )
    return out


def _three_dps():
    return [_dp(0, 2, ["a", "b"], 0), _dp(2, 4, ["a", "b"], 1), _dp(4, 6, ["a", "b"], 0)]


@pytest.mark.parametrize(
    "samples,reward,terminal,kw",
    [
        ([_sample(_three_dps())], 1.0, True, {}),
        ([_sample(_three_dps())], 0.0, False, {}),
        ([_sample(_three_dps()), _sample([_dp(0, 2, ["a", "b"], 1)])], 1.0, True, {}),
        ([_sample(_three_dps())], 1.0, True, {"gamma": 0.9}),
        ([_sample(_three_dps())], 1.0, True, {"n_step": 1}),
    ],
)
def test_defaults_reproduce_old_targets(samples, reward, terminal, kw):
    new = build_warmstart_samples(samples, episodic_reward=reward, is_terminal=terminal, **kw)
    old = _legacy_build(samples, reward, terminal, **kw)
    assert msgspec.msgpack.encode(new) == msgspec.msgpack.encode(old)


def test_label_from_drops_prefix():
    dps = [_dp(0, 2, ["a", "b"], 0), _dp(2, 4, ["a", "b"], 1), _dp(4, 6, ["a", "b"], 0), _dp(6, 8, ["a", "b"], 1), _dp(8, 10, ["a", "b"], 0)]
    out = build_warmstart_samples([_sample(dps)], episodic_reward=1.0, is_terminal=True, label_from_dp=2)
    targets = out[0].decision_point_targets
    assert [t.response_start for t in targets] == [4, 6, 8]
    assert all(t.v_target == 1.0 and t.q_plus_target == 1.0 for t in targets)
    assert all(t.v_weight == 1.0 and t.q_weight == 1.0 for t in targets)


def test_root_v_weight_zero():
    dps = _three_dps()
    out = build_warmstart_samples(
        [_sample(dps)], episodic_reward=1.0, is_terminal=True, label_from_dp=1, root_dp=1
    )
    targets = out[0].decision_point_targets
    assert [t.response_start for t in targets] == [2, 4]
    assert targets[0].v_weight == 0.0
    assert targets[0].q_weight == 1.0
    assert targets[1].v_weight == 1.0
    assert targets[1].q_weight == 1.0
    # root_v_weight is a knob
    out2 = build_warmstart_samples(
        [_sample(dps)], episodic_reward=1.0, is_terminal=True, label_from_dp=1, root_dp=1, root_v_weight=0.25
    )
    assert out2[0].decision_point_targets[0].v_weight == 0.25


def test_joint_return_across_split_with_label_from():
    # dps [0,1,2] then [3,4]; the root is joint dp 3 (second sample). The terminal
    # reward must still reach the labelled dps; the first sample emits [].
    s0 = _sample(_three_dps())
    s1 = _sample([_dp(0, 2, ["a", "b"], 1), _dp(2, 4, ["a", "b"], 0)])
    out = build_warmstart_samples([s0, s1], episodic_reward=1.0, is_terminal=True, label_from_dp=3, root_dp=3)
    assert len(out) == 2
    assert out[0].decision_point_targets == []
    assert out[0].prompt_ids == s0.prompt_ids
    t = out[1].decision_point_targets
    assert len(t) == 2
    assert all(x.v_target == 1.0 and x.q_plus_target == 1.0 for x in t)
    assert t[0].v_weight == 0.0 and t[1].v_weight == 1.0


def test_label_from_all_dropped_still_one_output_per_sample():
    out = build_warmstart_samples([_sample(_three_dps())], episodic_reward=1.0, is_terminal=True, label_from_dp=3)
    assert len(out) == 1
    assert out[0].decision_point_targets == []


def test_invalid_indices_raise():
    s = _sample(_three_dps())
    with pytest.raises(ValueError):
        build_warmstart_samples([s], episodic_reward=1.0, is_terminal=True, label_from_dp=4)
    with pytest.raises(ValueError):
        build_warmstart_samples([s], episodic_reward=1.0, is_terminal=True, label_from_dp=2, root_dp=1)
    with pytest.raises(ValueError):
        build_warmstart_samples([s], episodic_reward=1.0, is_terminal=True, root_dp=3)
