"""Branched dataset Phase 0: DecisionPointTarget per-leg weights (D-10).

The two new fields default to 1.0, round-trip, and a frame encoded by the
pre-change 4-field struct still decodes (weights 1.0).
"""

import msgspec

from prime_rl.transport.types import DecisionPointTarget


def test_weights_default_to_one():
    t = DecisionPointTarget(response_start=3, executed_action="go", v_target=1.0, q_plus_target=1.0)
    assert t.v_weight == 1.0
    assert t.q_weight == 1.0


def test_weights_roundtrip():
    t = DecisionPointTarget(3, "go", 1.0, 0.5, v_weight=0.0, q_weight=1.0)
    blob = msgspec.msgpack.encode(t)
    back = msgspec.msgpack.decode(blob, type=DecisionPointTarget)
    assert back == t
    assert back.v_weight == 0.0


def test_legacy_four_field_frame_decodes():
    # What the pre-change struct put on the wire: a 4-element array.
    legacy = msgspec.msgpack.encode([3, "go", 1.0, 0.5])
    back = msgspec.msgpack.decode(legacy, type=DecisionPointTarget)
    assert back.response_start == 3
    assert back.executed_action == "go"
    assert back.v_weight == 1.0
    assert back.q_weight == 1.0
