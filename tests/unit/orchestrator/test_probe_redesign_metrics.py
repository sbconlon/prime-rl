"""plan/arm-redesign Phase 2 -- the per-property probe metrics and the
ARM_DIAG extras.

Fabricated records only (the check against the stored probe round lives in the
thesis folder: advantage_redesign_2026-11-18/metric_parity_check.py).
"""

from __future__ import annotations

import asyncio
import math

import pytest

from prime_rl.advantage_server.action_advantage import ActionAdvantageInputs, action_advantage_fn
from prime_rl.advantage_server.compute import _log_advantage_magnitudes_action, arm_diag_extras
from prime_rl.orchestrator.probe import (
    DEAD_TURN_LAST,
    dead_turns,
    probe_metrics,
    redesign_metrics,
    redesign_summary_line,
)
from tests.unit.orchestrator.test_probe_assemble import _assemble, load_fixture

ADM = ["go to desk 1", "take box 1", "look", "open drawer 1"]


def _turn(k, *, adv_star=0.0, e=0, pi_rm=None, q=None, v=0.0, regret=None, valid=True, blocks=None,
          reasoning="<think>The box may be on the desk.</think>"):
    values = {
        "v": v,
        "q_plus": q if q is not None else [0.0] * len(ADM),
        "pi_rm": pi_rm if pi_rm is not None else [0.0] * len(ADM),
        "adv_star": adv_star,
        "adv_star_valid": valid,
    }
    if regret is not None:
        values["regret"] = regret
    return {
        "turn": k,
        "admissible": list(ADM),
        "executed": {"idx": e, "text": ADM[e], "reasoning": reasoning},
        "blocks": blocks if blocks is not None else [],
        "values": values,
    }


def _record(turns, *, won=False, stop="max_turns_reached"):
    return {"won": won, "stop_condition": stop, "num_turns": len(turns), "turns": turns}


# --------------------------------------------------------------------------- #
# dead turns
# --------------------------------------------------------------------------- #


def test_dead_turns_definition():
    turns = [_turn(k) for k in range(15)]
    out_of_turns = _record(turns)
    assert [t["turn"] for t in dead_turns(out_of_turns)] == [12, 13, 14]
    assert DEAD_TURN_LAST == 3
    assert dead_turns(_record(turns, won=True, stop=None)) == []          # a won game
    assert dead_turns(_record(turns[:6], won=False, stop=None)) == []     # lost, but not out of turns
    assert [t["turn"] for t in dead_turns(out_of_turns, last=5)] == [10, 11, 12, 13, 14]
    # a game shorter than the window: every turn counts
    assert len(dead_turns(_record([_turn(0), _turn(1)]))) == 2


def test_dead_turn_push_frac():
    advs = [0.0] * 12 + [0.0, -0.04, -0.60]  # of the three dead turns, one is pushed
    lost = _record([_turn(k, adv_star=a) for k, a in enumerate(advs)])
    won = _record([_turn(k, adv_star=-0.9) for k in range(4)], won=True, stop=None)
    m = redesign_metrics([lost, won])
    assert m["probe/dead_turns"] == 3.0
    assert m["probe/dead_turn_push_frac"] == pytest.approx(1 / 3)
    # an invalid advantage (pi_hat missing) is not a dead-turn reading
    advs_turns = [_turn(k, adv_star=a) for k, a in enumerate(advs)]
    advs_turns[14]["values"]["adv_star_valid"] = False
    m2 = redesign_metrics([_record(advs_turns)])
    assert m2["probe/dead_turns"] == 2.0 and m2["probe/dead_turn_push_frac"] == 0.0
    # no game out of turns: the count is zero and the fraction is absent
    m3 = redesign_metrics([won])
    assert m3["probe/dead_turns"] == 0.0 and "probe/dead_turn_push_frac" not in m3


# --------------------------------------------------------------------------- #
# agreement with pi_RM
# --------------------------------------------------------------------------- #


def test_pi_rm_agreement_difference_form():
    turns = [
        # cleared regret on action 1; the policy played action 1 -> weight 0.75
        _turn(0, e=1, pi_rm=[0.0, 0.75, 0.25, 0.0], regret=[0.0, 0.3, 0.1, 0.0]),
        # cleared regret elsewhere; the policy played action 0 -> weight 0
        _turn(1, e=0, pi_rm=[0.0, 1.0, 0.0, 0.0], regret=[0.0, 0.2, 0.0, 0.0]),
        # nothing clears the margin: pi_RM is not defined, the turn is skipped
        _turn(2, e=0, pi_rm=[0.0, 0.0, 0.0, 0.0], regret=[0.0, 0.0, 0.0, 0.0]),
        _turn(3, e=2, pi_rm=[0.0, 0.0, 0.0, 0.0], regret=[0.0, 0.0, 0.0, 0.0]),
    ]
    m = redesign_metrics([_record(turns, won=True, stop=None)])
    assert m["probe/pi_rm_defined_frac"] == pytest.approx(0.5)
    assert m["probe/pi_rm_agreement"] == pytest.approx((0.75 + 0.0) / 2)


def test_pi_rm_agreement_original_rule():
    turns = [
        _turn(0, e=0, pi_rm=[0.6, 0.4, 0.0, 0.0], q=[0.7, 0.6, 0.2, 0.1], v=0.4),
        # no Q+ above V: the uniform fallback does not count as defined
        _turn(1, e=0, pi_rm=[0.25] * 4, q=[0.3, 0.2, 0.1, 0.0], v=0.4),
    ]
    m = redesign_metrics([_record(turns, won=True, stop=None)])
    assert m["probe/pi_rm_defined_frac"] == pytest.approx(0.5)
    assert m["probe/pi_rm_agreement"] == pytest.approx(0.6)
    # nowhere defined: the share is zero and the agreement is absent
    m2 = redesign_metrics([_record([turns[1]], won=True, stop=None)])
    assert m2["probe/pi_rm_defined_frac"] == 0.0 and "probe/pi_rm_agreement" not in m2


# --------------------------------------------------------------------------- #
# monitors
# --------------------------------------------------------------------------- #


def test_reason_action_agreement():
    blocks = [
        {"reasoning": "<think>go</think>", "action": "go to desk 1", "cond": [0.9, 0.05, 0.03, 0.02]},
        {"reasoning": "<think>take</think>", "action": "take box 1", "cond": [0.3, 0.5, 0.1, 0.1]},
        # an action outside the admissible list is skipped
        {"reasoning": "<think>?</think>", "action": "fly away", "cond": [0.25] * 4},
    ]
    m = redesign_metrics([_record([_turn(0, blocks=blocks)], won=True, stop=None)])
    assert m["probe/reason_action_agreement"] == pytest.approx((0.9 + 0.5) / 2)
    assert "probe/reason_action_agreement" not in redesign_metrics([_record([_turn(0)], won=True, stop=None)])


def test_prefill_phrase_frac():
    blocks = [
        {"reasoning": "<think>I will now take box 1 because it is here.</think>", "action": "take box 1",
         "cond": [0.1, 0.8, 0.05, 0.05]},
        {"reasoning": "<think>The desk is the likely place.</think>", "action": "go to desk 1",
         "cond": [0.9, 0.05, 0.03, 0.02]},
    ]
    turns = [
        _turn(0, blocks=blocks, reasoning="  <think> I will now go to desk 1.</think>"),
        _turn(1, reasoning="I will now look"),
        _turn(2, reasoning="<think>Maybe I will now look.</think>"),  # not at the opening
    ]
    m = redesign_metrics([_record(turns, won=True, stop=None)])
    # 5 pieces of reasoning (3 executed + 2 blocks), 3 open with the phrase
    assert m["probe/prefill_phrase_frac"] == pytest.approx(3 / 5)


def test_redesign_metrics_empty():
    assert redesign_metrics([]) == {}
    assert redesign_metrics([_record([])]) == {}


def test_probe_metrics_keeps_existing_keys_and_adds_new():
    record, _, _ = asyncio.run(_assemble(load_fixture()))
    m = probe_metrics([record], [10], env_name="unit", num_skipped=0, probe_time=1.0)
    for key in (
        "probe/num_rollouts", "probe/num_skipped", "probe/time", "probe/num_turns_mean",
        "probe/q_plus_spread_mean", "probe/q_plus_std_median", "probe/pi_rm_entropy_mean",
        "probe/admissible_mass_mean", "probe/pi_hat_renorm", "probe/pi_hat_missing_frac",
        "probe/adv_star_mean", "probe/adv_star_absmax", "eval/unit/avg@1",
    ):
        assert key in m
    for key in ("probe/dead_turns", "probe/pi_rm_defined_frac", "probe/reason_action_agreement",
                "probe/prefill_phrase_frac"):
        assert key in m
    assert m["probe/dead_turns"] == 0.0  # the fixture game is won
    # the fixture's value networks are at zero: no Q+ above V, so pi_RM is nowhere defined
    assert m["probe/pi_rm_defined_frac"] == 0.0 and "probe/pi_rm_agreement" not in m


def test_redesign_summary_line():
    line = redesign_summary_line(
        10,
        {"probe/dead_turns": 42.0, "probe/dead_turn_push_frac": 5 / 42, "probe/pi_rm_agreement": 0.0341,
         "probe/pi_rm_defined_frac": 151 / 273, "probe/admissible_mass_mean": 0.9597},
    )
    assert line.startswith("PROBE_REDESIGN ckpt=10 dead_turns=42.0000 dead_turn_push_frac=0.1190 ")
    assert "pi_rm_defined_frac=0.5531 pi_rm_agreement=0.0341 admissible_mass_mean=0.9597" in line
    # keys the round did not produce are printed as na, so the line always has the same fields
    assert "q_plus_std_median=na" in line and line.endswith("prefill_phrase_frac=na")


# --------------------------------------------------------------------------- #
# ARM_DIAG extras (Advantage Server)
# --------------------------------------------------------------------------- #


def _out_and_records(form: str):
    q_plus = [[0.90, 0.20, 0.45], [0.30, 0.35], [0.50, 0.10]]
    v = [0.40, 0.40, 0.60]
    kwargs = dict(advantage_form="difference", margin=0.10, anchor=0.3) if form == "difference" else {}
    inp = ActionAdvantageInputs(
        q_plus=q_plus, v=v, executed_idx=[1, 0, 0], pi_hat_star=[0.8, 0.6, 0.4],
        rewards=[0.0, 0.0, 1.0], v_target=[0.0, 0.0, 0.0], gamma=1.0, n_step=15, phi_decay=0.9, **kwargs,
    )
    records = [{"v": vi, "q_plus": qi} for vi, qi in zip(v, q_plus)]
    return action_advantage_fn(inp), records


def test_v_bias_and_extras_difference_form():
    out, records = _out_and_records("difference")
    extras = arm_diag_extras(out, records)
    # g = 1 at every decision point (terminal reward, horizon >= K)
    assert extras["v_bias"] == pytest.approx(((1 - 0.40) + (1 - 0.40) + (1 - 0.60)) / 3)
    # cleared regret: [0.40, 0, 0], [0, 0], [0, 0] -> mean total 0.40 / 3
    assert extras["s_mean"] == pytest.approx(0.40 / 3)
    # the last two decision points have no cleared regret: exactly zero advantages
    assert extras["adv_zero_frac"] == pytest.approx(2 / 3)


def test_v_bias_and_extras_original_rule():
    out, records = _out_and_records("log_ratio")
    extras = arm_diag_extras(out, records)
    assert extras["v_bias"] == pytest.approx(((1 - 0.40) + (1 - 0.40) + (1 - 0.60)) / 3)
    # positive regret summed: (0.50 + 0.05), 0, 0
    assert extras["s_mean"] == pytest.approx((0.50 + 0.05) / 3)
    assert extras["adv_zero_frac"] == 0.0


def test_arm_diag_extras_without_records_and_logging():
    out, records = _out_and_records("difference")
    extras = arm_diag_extras(out, None)
    assert set(extras) == {"adv_zero_frac"}
    # the log call accepts both signatures and never raises
    _log_advantage_magnitudes_action(out)
    _log_advantage_magnitudes_action(out, records)
    assert math.isfinite(arm_diag_extras(out, records)["v_bias"])
