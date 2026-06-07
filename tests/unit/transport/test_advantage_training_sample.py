"""Phase 6a -- transport-layer roundtrip tests for AdvantageTrainingSample
and AdvantageTrainingBatch.
"""

from __future__ import annotations

import msgspec

from prime_rl.transport.types import (
    AdvantageTrainingBatch,
    AdvantageTrainingSample,
    DecisionPointTarget,
)


def _make_minimal_adv_sample(
    completion_len: int = 3,
    *,
    v_targets: list[float] | None = None,
    q_plus_targets: list[float] | None = None,
) -> AdvantageTrainingSample:
    return AdvantageTrainingSample(
        prompt_ids=[1, 2],
        prompt_mask=[False, False],
        completion_ids=list(range(10, 10 + completion_len)),
        completion_mask=[True] * completion_len,
        v_targets=v_targets,
        q_plus_targets=q_plus_targets,
    )


def test_advantage_training_sample_round_trip():
    """Populated v_targets + q_plus_targets round-trip bit-equal."""
    sample = _make_minimal_adv_sample(
        completion_len=3,
        v_targets=[0.1, 0.2, 0.3],
        q_plus_targets=[0.5, 0.6, 0.7],
    )
    encoded = msgspec.msgpack.encode(sample)
    decoded = msgspec.msgpack.decode(encoded, type=AdvantageTrainingSample)
    assert decoded.prompt_ids == sample.prompt_ids
    assert decoded.prompt_mask == sample.prompt_mask
    assert decoded.completion_ids == sample.completion_ids
    assert decoded.completion_mask == sample.completion_mask
    assert decoded.v_targets == sample.v_targets
    assert decoded.q_plus_targets == sample.q_plus_targets


def test_advantage_training_sample_q_plus_none_round_trip():
    """PPO path: q_plus_targets=None survives the round-trip."""
    sample = _make_minimal_adv_sample(
        completion_len=2, v_targets=[0.5, 0.6], q_plus_targets=None
    )
    encoded = msgspec.msgpack.encode(sample)
    decoded = msgspec.msgpack.decode(encoded, type=AdvantageTrainingSample)
    assert decoded.v_targets == [0.5, 0.6]
    assert decoded.q_plus_targets is None


def test_advantage_training_sample_omit_defaults_for_none_targets():
    """omit_defaults=True keeps the wire format minimal when targets are None."""
    sample = _make_minimal_adv_sample(completion_len=3, v_targets=None, q_plus_targets=None)
    encoded = msgspec.msgpack.encode(sample)
    # The field names should NOT appear in the encoded bytes when both are None
    # (omit_defaults strips defaults from the wire format).
    assert b"v_targets" not in encoded
    assert b"q_plus_targets" not in encoded


def test_advantage_training_batch_round_trip():
    """AdvantageTrainingBatch round-trips bit-equal."""
    samples = [
        _make_minimal_adv_sample(completion_len=2, v_targets=[0.1, 0.2], q_plus_targets=[0.3, 0.4]),
        _make_minimal_adv_sample(completion_len=3, v_targets=[0.5, 0.6, 0.7], q_plus_targets=None),
    ]
    batch = AdvantageTrainingBatch(examples=samples, step=42, run_idx=7)
    encoded = msgspec.msgpack.encode(batch)
    decoded = msgspec.msgpack.decode(encoded, type=AdvantageTrainingBatch)
    assert decoded.step == 42
    assert decoded.run_idx == 7
    assert len(decoded.examples) == 2
    assert decoded.examples[0].v_targets == [0.1, 0.2]
    assert decoded.examples[1].q_plus_targets is None


def test_advantage_training_batch_run_idx_omit_defaults():
    """run_idx default (None) is omitted from the wire format."""
    batch = AdvantageTrainingBatch(examples=[_make_minimal_adv_sample()], step=1)
    encoded = msgspec.msgpack.encode(batch)
    assert b"run_idx" not in encoded


# --------------------------------------------------------------------------- #
# Action-level ARM: decision_point_targets (Phase 6 contract change)
# --------------------------------------------------------------------------- #


def test_decision_point_target_round_trip():
    dpt = DecisionPointTarget(
        response_start=4, executed_action="go to cabinet 1", v_target=1.0, q_plus_target=0.3
    )
    decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(dpt), type=DecisionPointTarget)
    assert decoded.response_start == 4
    assert decoded.executed_action == "go to cabinet 1"
    assert decoded.v_target == 1.0
    assert decoded.q_plus_target == 0.3


def test_advantage_sample_with_decision_point_targets_round_trip():
    sample = _make_minimal_adv_sample(completion_len=2)
    sample.decision_point_targets = [
        DecisionPointTarget(0, "look", 1.0, 0.5),
        DecisionPointTarget(2, "go north", 1.0, 0.2),
    ]
    decoded = msgspec.msgpack.decode(
        msgspec.msgpack.encode(sample), type=AdvantageTrainingSample
    )
    assert decoded.decision_point_targets is not None
    assert len(decoded.decision_point_targets) == 2
    assert decoded.decision_point_targets[1].executed_action == "go north"


def test_advantage_sample_decision_point_targets_default_none_omitted():
    sample = _make_minimal_adv_sample(v_targets=[0.1, 0.2, 0.3])
    assert sample.decision_point_targets is None
    encoded = msgspec.msgpack.encode(sample)
    assert b"decision_point_targets" not in encoded
