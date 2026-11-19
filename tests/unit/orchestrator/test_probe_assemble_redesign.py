"""plan/arm-redesign Phase 1 -- assemble_probe_rollout under the difference
advantage form (anchored regret matching).

The diagnostics carry the cleared regrets and the anchor; the probe assembles
the per-action advantage (R(a) - S * pi_hat(a)) / (S + anchor) from them and
cross-checks the executed action against the server's adv_star. Reuses the
two-turn fixture of test_probe_assemble.py.
"""

from __future__ import annotations

import asyncio

import pytest

from prime_rl.orchestrator.advantage_server_client import DecisionPointDiagnostics
from prime_rl.orchestrator.probe import ProbeAssemblyError, flatten_decision_points, probe_metrics
from tests.unit.orchestrator._stubs import _zero_diagnostics_for_samples
from tests.unit.orchestrator.test_probe_assemble import VALUES_KEYS, _assemble, load_fixture

ANCHOR = 0.3


def _difference_factory(*, amount: float = 0.3, anchor: float | None = ANCHOR, adv_shift: float = 0.0):
    """Diagnostics with `amount` of cleared regret on the action after the
    executed one (none when amount == 0) and the matching adv_star."""

    def factory(samples, episodic_reward, is_terminal, algorithm):
        base = _zero_diagnostics_for_samples(samples)
        flat = flatten_decision_points(samples)
        out = []
        for d, (_, dp) in zip(base, flat):
            n = len(d.q_plus)
            e = dp.executed_action_idx
            regret = [0.0] * n
            regret[(e + 1) % n] = amount
            total = sum(regret)
            kappa = anchor if anchor is not None else ANCHOR
            adv_star = (regret[e] - total * dp.pi_hat) / (total + kappa) if total > 0 else 0.0
            out.append(
                DecisionPointDiagnostics(
                    sample_idx=d.sample_idx,
                    response_start=d.response_start,
                    q_plus=d.q_plus,
                    v=d.v,
                    v_target=d.v_target,
                    pi_rm=[x / total for x in regret] if total > 0 else [0.0] * n,
                    adv_star=adv_star + adv_shift,
                    g_k=d.g_k,
                    q_plus_target_star=d.q_plus_target_star,
                    regret=regret,
                    anchor=anchor,
                )
            )
        return out

    return factory


def test_probe_advantage_follows_difference_form():
    record, _, diag = asyncio.run(_assemble(load_fixture(), diagnostics_factory=_difference_factory()))
    for turn, d in zip(record["turns"], diag.diagnostics):
        values = turn["values"]
        assert set(values) == VALUES_KEYS | {"regret", "advantage_form", "anchor"}
        assert values["advantage_form"] == "difference" and values["anchor"] == pytest.approx(ANCHOR)
        assert values["regret"] == pytest.approx(d.regret)
        total = sum(d.regret)
        expected = [(r - total * ph) / (total + ANCHOR) for r, ph in zip(d.regret, turn["pi_hat"])]
        assert values["adv"] == pytest.approx(expected)
        e = turn["executed"]["idx"]
        assert values["adv"][e] == pytest.approx(values["adv_star"], abs=1e-6)
        assert values["adv_star_valid"] is True
        assert all(abs(a) < 1.0 for a in values["adv"])
        # the executed action has no cleared regret and policy mass: pushed down
        assert values["adv"][e] < 0.0
        # the action that holds the cleared regret is pushed up
        assert values["adv"][(e + 1) % len(values["adv"])] > 0.0


def test_probe_no_cleared_regret_is_all_zero():
    record, _, _ = asyncio.run(_assemble(load_fixture(), diagnostics_factory=_difference_factory(amount=0.0)))
    for turn in record["turns"]:
        values = turn["values"]
        n = len(turn["admissible"])
        assert values["adv"] == [0.0] * n and values["adv_star"] == 0.0
        assert values["pi_rm"] == [0.0] * n and values["regret"] == [0.0] * n
    # the round summary still computes on all-zero pi_rm rows (no log of zero)
    metrics = probe_metrics([record], [10], env_name="unit", num_skipped=0, probe_time=0.0)
    assert metrics["probe/pi_rm_entropy_mean"] == 0.0


def test_probe_rejects_inconsistent_difference_diagnostics():
    with pytest.raises(ProbeAssemblyError, match="adv"):
        asyncio.run(_assemble(load_fixture(), diagnostics_factory=_difference_factory(adv_shift=0.01)))
    with pytest.raises(ProbeAssemblyError, match="anchor"):
        asyncio.run(_assemble(load_fixture(), diagnostics_factory=_difference_factory(anchor=None)))


def test_probe_default_form_record_has_no_new_keys():
    record, _, _ = asyncio.run(_assemble(load_fixture()))
    for turn in record["turns"]:
        assert set(turn["values"]) == VALUES_KEYS
