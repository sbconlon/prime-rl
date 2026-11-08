"""training-probe Phase 3 -- assemble_probe_rollout (pure join) and probe_metrics.

The fixture is a two-turn ALFWorld probe rollout with tokens, ``extras["probe"]``
and the decision-point extras; diagnostics come from FakeAdvantageServer
(iteration-0 regime: Q+ = V = 0, uniform pi_RM, adv_star = log(1/|A|) - log pi_hat).
"""

from __future__ import annotations

import asyncio
import copy
import json
import math
from pathlib import Path

import pytest
import verifiers as vf

from prime_rl.orchestrator.advantage_server_client import DecisionPointDiagnostics
from prime_rl.orchestrator.probe import (
    ADV_FLOOR,
    ProbeAssemblyError,
    assemble_probe_rollout,
    flatten_decision_points,
    probe_metrics,
)
from prime_rl.orchestrator.trajectories import interleave_rollout
from tests.unit.orchestrator._stubs import FakeAdvantageServer, _zero_diagnostics_for_samples

FIXTURE = Path(__file__).parent / "fixtures" / "probe_rollout_fixture.json"
SAMPLING = {"exec_temperature": 0.0, "block_temperature": 0.8, "block_min_p": 0.1, "m": 2}

TOP_KEYS = {
    "schema_version", "run", "round", "probe_id", "game_id", "task_type", "split", "role", "won",
    "reward", "num_turns", "is_truncated", "stop_condition", "context_evictions", "oracle_mode",
    "sampling", "turns",
}
TURN_KEYS = {
    "turn", "observation", "window", "admissible", "executed", "blocks", "pi_hat", "pi_hat_star", "Z",
    "raw_logps", "values", "oracle", "env_response",
}
VALUES_KEYS = {"v", "v_target", "q_plus", "pi_rm", "adv", "adv_star", "adv_star_valid", "g_k", "q_plus_target_star"}


def load_fixture() -> vf.RolloutOutput:
    with FIXTURE.open(encoding="utf-8") as fh:
        return vf.RolloutOutput(json.load(fh))


async def _assemble(rollout, *, diagnostics_factory=None, ckpt_step=10, orchestrator_step=12):
    samples = interleave_rollout(rollout)
    assert samples is not None
    server = FakeAdvantageServer(diagnostics_factory=diagnostics_factory, initial_weight_step=7)
    diag = await server.compute_with_diagnostics(samples, float(rollout["reward"]), True, "arm")
    record = assemble_probe_rollout(
        rollout,
        samples,
        diag,
        ckpt_step=ckpt_step,
        orchestrator_step=orchestrator_step,
        run="unit-run",
        sampling=SAMPLING,
        oracle_mode="handcoded",
        timestamp="2026-11-07T00:00:00+00:00",
    )
    return record, samples, diag


def _diag_with(records_fn):
    """diagnostics_factory wrapper: build the fake's zero diagnostics, then let
    ``records_fn`` perturb the list."""

    def factory(samples, episodic_reward, is_terminal, algorithm):
        return records_fn(_zero_diagnostics_for_samples(samples))

    return factory


def _replace(d: DecisionPointDiagnostics, **changes) -> DecisionPointDiagnostics:
    fields = dict(
        sample_idx=d.sample_idx, response_start=d.response_start, q_plus=d.q_plus, v=d.v,
        v_target=d.v_target, pi_rm=d.pi_rm, adv_star=d.adv_star, g_k=d.g_k,
        q_plus_target_star=d.q_plus_target_star,
    )
    fields.update(changes)
    return DecisionPointDiagnostics(**fields)


def test_assemble_matches_schema():
    rollout = load_fixture()
    record, samples, diag = asyncio.run(_assemble(rollout))
    assert set(record) == TOP_KEYS
    assert record["schema_version"] == 1
    assert record["round"] == {
        "ckpt_step": 10,
        "weight_step": 7,
        "orchestrator_step": 12,
        "timestamp": "2026-11-07T00:00:00+00:00",
    }
    assert (record["probe_id"], record["task_type"], record["split"], record["role"]) == (
        "seen-look-bite", "look_at_obj_in_light", "valid_seen", "bite"
    )
    assert record["game_id"].startswith("valid_seen/look_at_obj_in_light-Box")
    assert record["won"] is True and record["reward"] == 1.0
    assert record["num_turns"] == 2 == len(record["turns"])
    assert record["is_truncated"] is False and record["stop_condition"] is None
    assert record["context_evictions"] == 0 and record["oracle_mode"] == "handcoded"
    assert record["sampling"] == SAMPLING
    for k, turn in enumerate(record["turns"]):
        assert set(turn) == TURN_KEYS
        assert turn["turn"] == k
        assert set(turn["values"]) == VALUES_KEYS
        n = len(turn["admissible"])
        assert len(turn["pi_hat"]) == n
        assert len(turn["values"]["q_plus"]) == len(turn["values"]["pi_rm"]) == len(turn["values"]["adv"]) == n
        assert all(len(b["cond"]) == n for b in turn["blocks"])
        assert len(turn["Z"]) == len(turn["blocks"]) == 2
        assert turn["admissible"][turn["executed"]["idx"]] == turn["executed"]["text"]
        assert turn["pi_hat_star"] == turn["pi_hat"][turn["executed"]["idx"]]
        assert turn["oracle"]["status"] == "expert"
    assert record["turns"][0]["executed"]["text"] == "go to desk 1"
    assert record["turns"][1]["executed"]["text"] == "take box 1 from desk 1"
    assert record["turns"][0]["window"]["labels"] == ["system", "task", "obs0"]
    # the two turns merged into one sample: decision points flatten in order
    flat = flatten_decision_points(samples)
    assert len(samples) == 1 and len(flat) == 2
    assert [dp.response_start for _, dp in flat] == [d.response_start for d in diag.diagnostics]


def test_adv_vector_and_star_consistency():
    rollout = load_fixture()
    record, _, diag = asyncio.run(_assemble(rollout))
    for turn, d in zip(record["turns"], diag.diagnostics):
        n = len(turn["admissible"])
        e = turn["executed"]["idx"]
        expected = [math.log(max(1.0 / n, ADV_FLOOR)) - math.log(max(p, ADV_FLOOR)) for p in turn["pi_hat"]]
        assert turn["values"]["adv"] == pytest.approx(expected)
        assert turn["values"]["adv"][e] == pytest.approx(turn["values"]["adv_star"], abs=1e-6)
        assert turn["values"]["adv_star"] == pytest.approx(d.adv_star)
        assert turn["values"]["adv_star_valid"] is True

    # a turn whose scoring failed: pi_hat null -> adv zeros, adv_star_valid False
    degraded = load_fixture()
    probe = degraded["trajectory"][1]["extras"]["probe"]
    probe["blocks"], probe["pi_hat_marginal"], probe["Z"] = [], None, []
    del degraded["trajectory"][1]["extras"]["pi_hat"]
    record, _, diag = asyncio.run(_assemble(degraded))
    t1 = record["turns"][1]
    assert t1["pi_hat"] is None and t1["pi_hat_star"] is None and t1["blocks"] == []
    assert t1["values"]["adv"] == [0.0, 0.0]
    assert t1["values"]["adv_star"] == 0.0 == diag.diagnostics[1].adv_star
    assert t1["values"]["adv_star_valid"] is False
    assert record["turns"][0]["values"]["adv_star_valid"] is True  # the other turn is untouched


def test_misaligned_vectors_raise():
    rollout = load_fixture()

    # q_plus length disagrees with the admissible set
    def short_q(diags):
        diags[0] = _replace(diags[0], q_plus=diags[0].q_plus[:-1])
        return diags

    with pytest.raises(ProbeAssemblyError, match="q_plus"):
        asyncio.run(_assemble(rollout, diagnostics_factory=_diag_with(short_q)))

    # one diagnostics record missing
    with pytest.raises(ProbeAssemblyError, match="count mismatch"):
        asyncio.run(_assemble(rollout, diagnostics_factory=_diag_with(lambda diags: diags[:-1])))

    # admissible_union order perturbed relative to the decision point
    perturbed = load_fixture()
    probe = perturbed["trajectory"][0]["extras"]["probe"]
    probe["admissible_union"] = list(reversed(probe["admissible_union"]))
    with pytest.raises(ProbeAssemblyError, match="admissible_union"):
        asyncio.run(_assemble(perturbed))

    # the server's adv_star disagrees with the recomputed adv[a*]
    def wrong_star(diags):
        diags[1] = _replace(diags[1], adv_star=diags[1].adv_star + 0.01)
        return diags

    with pytest.raises(ProbeAssemblyError, match="adv_star"):
        asyncio.run(_assemble(rollout, diagnostics_factory=_diag_with(wrong_star)))

    # diagnostics record pointing at the wrong decision point
    def wrong_start(diags):
        diags[1] = _replace(diags[1], response_start=diags[1].response_start + 1)
        return diags

    with pytest.raises(ProbeAssemblyError, match="decision point"):
        asyncio.run(_assemble(rollout, diagnostics_factory=_diag_with(wrong_start)))

    # extras["probe"] missing on a decision turn
    no_probe = load_fixture()
    del no_probe["trajectory"][0]["extras"]["probe"]
    with pytest.raises(ProbeAssemblyError, match="probe"):
        asyncio.run(_assemble(no_probe))


def test_env_response_and_observation_threading():
    rollout = load_fixture()
    record, _, _ = asyncio.run(_assemble(rollout))
    t0, t1 = record["turns"]
    assert t0["observation"] == rollout["trajectory"][0]["prompt"][-1]["content"]
    assert t1["observation"] == rollout["trajectory"][1]["prompt"][-1]["content"]
    assert t0["env_response"] == rollout["trajectory"][1]["prompt"][-1]["content"]
    assert t1["env_response"] == "You pick up the box 1 from the desk 1."

    # no final env response (max_turns hit without a game response): None, not a crash
    cut = load_fixture()
    del cut["final_env_response"]
    record, _, _ = asyncio.run(_assemble(cut))
    assert record["turns"][1]["env_response"] is None


def test_probe_metrics_from_records():
    rollout = load_fixture()
    won, _, _ = asyncio.run(_assemble(rollout))
    lost = copy.deepcopy(won)
    lost.update(probe_id="seen-look-control", role="control", reward=0.0, won=False, is_truncated=True)
    lost["turns"][1]["oracle"]["status"] = "random"
    lost["turns"][1]["oracle"]["action"] = "look"
    m = probe_metrics([won, lost], [20, 30], env_name="alfworld-probe", num_skipped=1, probe_time=2.5)
    assert m["eval/alfworld-probe/avg@1"] == 0.5
    assert m["eval/alfworld-probe/valid_seen/avg@1"] == 0.5
    assert m["eval/alfworld-probe/look_at_obj_in_light/avg@1"] == 0.5
    assert m["eval/alfworld-probe/bite/avg@1"] == 1.0 and m["eval/alfworld-probe/control/avg@1"] == 0.0
    assert m["eval/alfworld-probe/completion_len/mean"] == 25.0
    assert m["eval/alfworld-probe/is_truncated/mean"] == 0.5
    assert m["eval/alfworld-probe/time"] == 2.5
    assert m["probe/num_rollouts"] == 2 and m["probe/num_skipped"] == 1
    assert m["probe/oracle_status/expert"] == 0.75 and m["probe/oracle_status/random"] == 0.25
    assert m["probe/oracle_agreement"] == 1.0  # every expert turn executed the oracle's action
    assert m["probe/q_plus_spread_mean"] == 0.0  # iteration-0 regime: flat Q+
    assert m["probe/pi_rm_entropy_mean"] == pytest.approx((math.log(3) + math.log(2)) / 2)
    assert m["probe/pi_hat_missing_frac"] == 0.0
    assert "probe/adv_star_mean" in m and "probe/adv_star_absmax" in m
    assert probe_metrics([], [], env_name="x", num_skipped=3, probe_time=1.0)["probe/num_skipped"] == 3
    # eval namespace can be switched off (eval configured separately)
    m2 = probe_metrics([won], [1], env_name="alfworld-probe", num_skipped=0, probe_time=1.0, emit_eval_metrics=False)
    assert not any(k.startswith("eval/") for k in m2)
