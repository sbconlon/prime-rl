"""Phase 1 (action-level ARM) -- transport-layer roundtrip tests for
DecisionPoint and the TrainingSample.decision_points field.

GRPO/PPO/token-ARM never populate decision_points; omit_defaults keeps it off
the wire so those paths are byte-for-byte unchanged.
"""

from __future__ import annotations

import msgspec

from prime_rl.transport.types import DecisionPoint, TrainingSample


def _make_minimal_sample(**overrides) -> TrainingSample:
    defaults = dict(
        prompt_ids=[1, 2],
        prompt_mask=[False, False],
        completion_ids=[10, 20, 30],
        completion_mask=[True, True, True],
        completion_logprobs=[-0.1, -0.2, -0.3],
        completion_temperatures=[1.0, 1.0, 1.0],
    )
    defaults.update(overrides)
    return TrainingSample(**defaults)


def test_decision_point_roundtrip():
    """A DecisionPoint with pi_hat populated round-trips bit-equal."""
    dp = DecisionPoint(
        response_start=0,
        response_end=5,
        admissible_actions=["go to cabinet 1", "look"],
        executed_action_idx=0,
        pi_hat=0.42,
    )
    encoded = msgspec.msgpack.encode(dp)
    decoded = msgspec.msgpack.decode(encoded, type=DecisionPoint)
    assert decoded.response_start == 0
    assert decoded.response_end == 5
    assert decoded.admissible_actions == ["go to cabinet 1", "look"]
    assert decoded.executed_action_idx == 0
    assert decoded.pi_hat == 0.42


def test_decision_point_roundtrip_pi_hat_none():
    """Phase 1 default: pi_hat=None (reserved for Phase 4) survives the round-trip."""
    dp = DecisionPoint(
        response_start=3,
        response_end=8,
        admissible_actions=["look", "go north"],
        executed_action_idx=1,
    )
    encoded = msgspec.msgpack.encode(dp)
    decoded = msgspec.msgpack.decode(encoded, type=DecisionPoint)
    assert decoded.pi_hat is None
    assert decoded.executed_action_idx == 1


def test_training_sample_with_decision_points_roundtrip():
    """A TrainingSample carrying a decision_points list survives encode/decode."""
    sample = _make_minimal_sample(
        decision_points=[
            DecisionPoint(
                response_start=0,
                response_end=3,
                admissible_actions=["a", "b"],
                executed_action_idx=0,
            ),
            DecisionPoint(
                response_start=3,
                response_end=3,
                admissible_actions=["c"],
                executed_action_idx=0,
                pi_hat=0.1,
            ),
        ]
    )
    encoded = msgspec.msgpack.encode(sample)
    decoded = msgspec.msgpack.decode(encoded, type=TrainingSample)
    assert decoded.decision_points is not None
    assert len(decoded.decision_points) == 2
    assert decoded.decision_points[0].admissible_actions == ["a", "b"]
    assert decoded.decision_points[0].pi_hat is None
    assert decoded.decision_points[1].pi_hat == 0.1


def test_training_sample_none_decision_points_default():
    """Default is None; the GRPO construction path is unchanged."""
    sample = _make_minimal_sample()
    assert sample.decision_points is None
    decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(sample), type=TrainingSample)
    assert decoded.decision_points is None


def test_training_sample_omit_defaults_keeps_grpo_wire_minimal():
    """omit_defaults=True means a None-decision_points sample never puts the field
    name on the wire (zero overhead for GRPO/PPO/token-ARM)."""
    encoded = msgspec.msgpack.encode(_make_minimal_sample())
    assert b"decision_points" not in encoded
