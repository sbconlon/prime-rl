"""branched-dataset Phase 4 -- the gate metrics a probe round emits (master doc
§8: G-B spread, G-C argmax-Q+ == oracle, split by verb class).

Oracle turns count only with status expert / look_fallback and an oracle action
inside the admissible list (the oracle is evaluation-only, D-6).
"""

from __future__ import annotations

import statistics

import pytest

from prime_rl.orchestrator.probe import gate_metrics, probe_metrics

ADM = ["go to desk 1", "take pen 1", "look"]


def _turn(q_plus, exec_idx, oracle_status, oracle_action, adm=ADM):
    return {
        "admissible": list(adm),
        "executed": {"text": adm[exec_idx], "idx": exec_idx},
        "oracle": {"status": oracle_status, "action": oracle_action},
        "pi_hat": None,
        "Z": [],
        "pi_hat_renorm": False,
        "values": {
            "q_plus": list(q_plus),
            "pi_rm": [1 / len(adm)] * len(adm),
            "adv_star": 0.0,
            "adv_star_valid": False,
        },
    }


TURNS = [
    # argmax "take pen 1": executed hit, oracle (expert, class take) hit
    _turn([0.1, 0.5, 0.2], 1, "expert", "take pen 1"),
    # argmax "go to desk 1": executed miss, oracle (look_fallback, class look) miss
    _turn([0.3, 0.2, 0.1], 2, "look_fallback", "look"),
    # flat row: argmax is the first index -> executed hit; oracle status random -> excluded
    _turn([0.2, 0.2, 0.2], 0, "random", "look"),
    # oracle action not admissible -> excluded from the oracle rate; executed hit
    _turn([0.0, 0.0, 0.6], 2, "expert", "open drawer 1"),
]


def test_spread_median_and_rates():
    m = gate_metrics(TURNS)
    stds = [statistics.pstdev(t["values"]["q_plus"]) for t in TURNS]
    assert m["probe/q_plus_std_median"] == pytest.approx(statistics.median(stds))
    assert m["probe/argmax_q_is_exec_rate"] == pytest.approx(0.75)
    assert m["probe/argmax_q_is_oracle_rate"] == pytest.approx(0.5)
    assert m["probe/argmax_q_is_oracle_rate/take"] == pytest.approx(1.0)
    assert m["probe/argmax_q_is_oracle_rate/look"] == pytest.approx(0.0)
    assert "probe/argmax_q_is_oracle_rate/open" not in m


def test_no_oracle_turns_omits_oracle_rate():
    turns = [_turn([0.1, 0.5, 0.2], 1, "off", None), _turn([0.3, 0.2, 0.1], 0, "random", "look")]
    m = gate_metrics(turns)
    assert "probe/argmax_q_is_oracle_rate" not in m
    assert m["probe/argmax_q_is_exec_rate"] == pytest.approx(1.0)
    assert m["probe/q_plus_std_median"] > 0.0


def test_empty_turns():
    assert gate_metrics([]) == {}


def test_probe_metrics_carries_gate_metrics():
    records = [{"turns": TURNS, "num_turns": len(TURNS), "reward": 1.0, "is_truncated": False}]
    m = probe_metrics(records, [], env_name="alfworld-probe", num_skipped=0, probe_time=1.0)
    assert m["probe/q_plus_std_median"] == pytest.approx(
        statistics.median(statistics.pstdev(t["values"]["q_plus"]) for t in TURNS)
    )
    assert m["probe/argmax_q_is_oracle_rate"] == pytest.approx(0.5)
    # the pre-existing metrics are untouched
    assert m["probe/q_plus_spread_mean"] == pytest.approx((0.4 + 0.2 + 0.0 + 0.6) / 4)
    assert m["probe/oracle_agreement"] == pytest.approx(0.5)
