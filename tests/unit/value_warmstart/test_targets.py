import msgspec
import pytest

from prime_rl.advantage_server.action_advantage import (
    ActionAdvantageInputs,
    action_advantage_fn,
)
from prime_rl.transport.types import (
    AdvantageTrainingSample,
    DecisionPoint,
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


def _sample(decision_points, completion_len=None):
    """A synthetic action-level TrainingSample. completion_ids/masks are sized to
    cover the decision points' response spans; prompt is 2 tokens."""
    if completion_len is None:
        completion_len = max((dp.response_end for dp in decision_points), default=0)
    return TrainingSample(
        prompt_ids=[1, 2],
        prompt_mask=[False, False],
        completion_ids=list(range(10, 10 + completion_len)),
        completion_mask=[True] * completion_len,
        completion_logprobs=[0.0] * completion_len,
        completion_temperatures=[1.0] * completion_len,
        decision_points=decision_points,
    )


def test_single_sample_mc_success():
    dps = [
        _dp(0, 2, ["go to cabinet 1", "look"], 0),
        _dp(2, 5, ["open cabinet 1", "look"], 1),
        _dp(5, 8, ["take mug 1", "look"], 0),
    ]
    out = build_warmstart_samples([_sample(dps)], episodic_reward=1.0, is_terminal=True)
    assert len(out) == 1
    targets = out[0].decision_point_targets
    assert len(targets) == 3
    for t, dp in zip(targets, dps):
        assert t.v_target == 1.0
        assert t.q_plus_target == 1.0
        assert t.response_start == dp.response_start


def test_single_sample_mc_failure():
    dps = [_dp(0, 2, ["a", "b"], 0), _dp(2, 4, ["a", "b"], 1)]
    out = build_warmstart_samples([_sample(dps)], episodic_reward=0.0, is_terminal=False)
    for t in out[0].decision_point_targets:
        assert t.v_target == 0.0
        assert t.q_plus_target == 0.0


def test_v_target_equals_q_plus_target():
    dps = [_dp(0, 2, ["a", "b"], 1), _dp(2, 4, ["a", "b", "c"], 2)]
    out = build_warmstart_samples([_sample(dps)], episodic_reward=1.0, is_terminal=True)
    for t in out[0].decision_point_targets:
        assert t.v_target == t.q_plus_target


def test_executed_action_is_text_from_admissible():
    # Executed index points to an action appended by the union invariant (last).
    dps = [_dp(0, 2, ["go to cabinet 1", "look", "take apple"], 2)]
    out = build_warmstart_samples([_sample(dps)], episodic_reward=1.0, is_terminal=True)
    assert out[0].decision_point_targets[0].executed_action == "take apple"


def test_ids_and_masks_copied():
    dps = [_dp(0, 3, ["a", "b"], 0)]
    s = _sample(dps)
    out = build_warmstart_samples([s], episodic_reward=1.0, is_terminal=True)
    assert out[0].prompt_ids == s.prompt_ids
    assert out[0].prompt_mask == s.prompt_mask
    assert out[0].completion_ids == s.completion_ids
    assert out[0].completion_mask == s.completion_mask
    assert out[0].v_targets is None
    assert out[0].q_plus_targets is None


def test_joint_return_across_truncation_split():
    # One rollout split into two samples: dps [0,1,2] then [3,4]; success at the
    # true terminal. EVERY dp in BOTH samples must get g_k = 1.0.
    s0 = _sample([_dp(0, 2, ["a", "b"], 0), _dp(2, 4, ["a", "b"], 1), _dp(4, 6, ["a", "b"], 0)])
    s1 = _sample([_dp(0, 2, ["a", "b"], 1), _dp(2, 4, ["a", "b"], 0)])
    out = build_warmstart_samples([s0, s1], episodic_reward=1.0, is_terminal=True)
    assert len(out) == 2
    all_targets = out[0].decision_point_targets + out[1].decision_point_targets
    assert len(all_targets) == 5
    assert all(t.v_target == 1.0 and t.q_plus_target == 1.0 for t in all_targets)


def test_one_output_per_input_sample():
    s0 = _sample([_dp(0, 2, ["a", "b"], 0)])
    s1 = _sample([_dp(0, 2, ["a", "b"], 1), _dp(2, 4, ["a", "b"], 0)])
    out = build_warmstart_samples([s0, s1], episodic_reward=1.0, is_terminal=True)
    assert len(out) == 2
    assert len(out[0].decision_point_targets) == 1
    assert len(out[1].decision_point_targets) == 2


def test_gamma_lt_one_discounts():
    # gamma=0.9, terminal reward at the last of K=3 dps, full-MC horizon (n_step=None):
    # g_k = 0.9 ** (K - 1 - k).
    dps = [_dp(0, 2, ["a", "b"], 0), _dp(2, 4, ["a", "b"], 0), _dp(4, 6, ["a", "b"], 0)]
    out = build_warmstart_samples(
        [_sample(dps)], episodic_reward=1.0, is_terminal=True, gamma=0.9
    )
    targets = out[0].decision_point_targets
    assert targets[0].v_target == pytest.approx(0.9**2)
    assert targets[1].v_target == pytest.approx(0.9**1)
    assert targets[2].v_target == pytest.approx(0.9**0)


def test_raises_on_none_decision_points():
    s = TrainingSample(
        prompt_ids=[1],
        prompt_mask=[False],
        completion_ids=[10, 11],
        completion_mask=[True, True],
        completion_logprobs=[0.0, 0.0],
        completion_temperatures=[1.0, 1.0],
    )
    with pytest.raises(ValueError):
        build_warmstart_samples([s], episodic_reward=1.0, is_terminal=True)


def test_raises_on_no_decision_points():
    s = _sample([])
    with pytest.raises(ValueError):
        build_warmstart_samples([s], episodic_reward=1.0, is_terminal=True)


def test_roundtrip_msgspec():
    dps = [_dp(0, 2, ["a", "b"], 0), _dp(2, 4, ["a", "b"], 1)]
    out = build_warmstart_samples([_sample(dps)], episodic_reward=1.0, is_terminal=True)
    blob = msgspec.msgpack.encode(out[0])
    decoded = msgspec.msgpack.decode(blob, type=AdvantageTrainingSample)
    assert decoded.decision_point_targets[0].executed_action == "a"
    assert decoded.decision_point_targets[1].v_target == 1.0
    assert decoded.prompt_ids == out[0].prompt_ids


def test_matches_online_zero_net():
    # Our targets must equal what the online action_advantage_fn produces with a
    # zero-init net (q_plus=v=v_target=0), gamma=1, n_step>=K, phi_decay=1.
    admissible = [["go", "look"], ["open", "look", "take"], ["take", "look"]]
    executed = [0, 2, 0]
    dps = [
        _dp(0, 2, admissible[0], executed[0]),
        _dp(2, 4, admissible[1], executed[1]),
        _dp(4, 6, admissible[2], executed[2]),
    ]
    out = build_warmstart_samples([_sample(dps)], episodic_reward=1.0, is_terminal=True)
    ours = [(t.v_target, t.q_plus_target) for t in out[0].decision_point_targets]

    K = len(dps)
    inputs = ActionAdvantageInputs(
        q_plus=[[0.0] * len(a) for a in admissible],
        v=[0.0] * K,
        executed_idx=list(executed),
        pi_hat_star=[1.0] * K,
        rewards=[0.0, 0.0, 1.0],
        v_target=[0.0] * K,
        gamma=1.0,
        n_step=K,
        phi_decay=1.0,
    )
    online = action_advantage_fn(inputs)
    for (v_ours, q_ours), v_online, q_online in zip(
        ours, online.v_target_out, online.q_plus_target_out
    ):
        assert v_ours == pytest.approx(v_online)
        assert q_ours == pytest.approx(q_online)
