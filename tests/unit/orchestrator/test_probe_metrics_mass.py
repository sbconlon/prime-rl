"""D-13: probe_metrics reports the admissible mass exp(Z) and the pi_hat convention."""

from __future__ import annotations

import math

import pytest

from prime_rl.orchestrator.probe import probe_metrics


def _turn(pi_hat, Z, renorm, adv_star=0.1, valid=True):
    n = 3
    return {
        "executed": {"text": "look"},
        "oracle": {"action": "look", "status": "expert"},
        "pi_hat": pi_hat,
        "Z": Z,
        "pi_hat_renorm": renorm,
        "values": {
            "q_plus": [0.1, 0.2, 0.3],
            "pi_rm": [1 / n] * n,
            "adv_star": adv_star,
            "adv_star_valid": valid,
        },
    }


def _record(turns):
    return {"turns": turns, "num_turns": len(turns), "reward": 1.0, "is_truncated": False}


def test_admissible_mass_metrics_and_flag():
    turns = [
        _turn([0.2, 0.1, 0.1], [math.log(0.4), math.log(0.8)], False),
        _turn([0.3, 0.3, 0.1], [math.log(0.7), math.log(0.2)], False),
        _turn(None, [], False, adv_star=0.0, valid=False),  # scoring failure: no Z contribution
    ]
    m = probe_metrics(
        [_record(turns)], completion_lens=[], env_name="alfworld-probe", num_skipped=0,
        probe_time=1.0, emit_eval_metrics=False,
    )
    assert m["probe/admissible_mass_mean"] == pytest.approx((0.4 + 0.8 + 0.7 + 0.2) / 4)
    assert m["probe/admissible_mass_min"] == pytest.approx(0.2)
    assert m["probe/admissible_mass_frac_below_half"] == pytest.approx(2 / 4)
    assert m["probe/pi_hat_renorm"] == 0.0
    assert m["probe/pi_hat_missing_frac"] == pytest.approx(1 / 3)


def test_flag_default_true_and_mixed():
    renormed = [_turn([0.5, 0.3, 0.2], [0.0, -0.1], True)]
    m = probe_metrics(
        [_record(renormed)], completion_lens=[], env_name="e", num_skipped=0, probe_time=0.0,
        emit_eval_metrics=False,
    )
    assert m["probe/pi_hat_renorm"] == 1.0
    assert m["probe/admissible_mass_mean"] == pytest.approx((1.0 + math.exp(-0.1)) / 2)
    legacy = _turn([0.5, 0.3, 0.2], [0.0], True)
    legacy.pop("pi_hat_renorm")  # records written before the key existed default to True
    mixed = [legacy, _turn([0.5, 0.3, 0.2], [0.0], False)]
    m2 = probe_metrics(
        [_record(mixed)], completion_lens=[], env_name="e", num_skipped=0, probe_time=0.0,
        emit_eval_metrics=False,
    )
    assert m2["probe/pi_hat_renorm"] == 0.5


def test_no_scored_turns_no_mass_metric():
    m = probe_metrics(
        [_record([_turn(None, [], False, adv_star=0.0, valid=False)])], completion_lens=[],
        env_name="e", num_skipped=0, probe_time=0.0, emit_eval_metrics=False,
    )
    assert "probe/admissible_mass_mean" not in m
    assert m["probe/pi_hat_renorm"] == 0.0
