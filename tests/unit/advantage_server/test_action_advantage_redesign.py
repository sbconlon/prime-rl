"""plan/arm-redesign Phase 1 -- anchored regret matching (difference advantage)
and the averaged Q+ target.

Pins what holds by the formulas: the default settings reproduce the original
rule; the difference advantage is bounded, continuous across the margin, exactly
zero with no cleared regret (no fallback distribution) and zero for every action
exactly at pi_hat == R / S; the averaged target settles at one times the
instantaneous regret. Then that the four settings reach the math through the
compute path and the endpoint, and that the diagnostics carry what the
orchestrator needs.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import msgspec
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from prime_rl.advantage_server.action_advantage import (
    ActionAdvantageInputs,
    action_advantage,
    action_advantage_fn,
    cleared_regrets,
    difference_advantage,
    n_step_return_action_level,
    pi_rm,
    q_plus_target,
)
from prime_rl.advantage_server.compute import (
    _build_per_decision_point_value_tensors,
    compute_advantages_and_targets_arm_action,
)
from prime_rl.configs.advantage_server import AdvantageServerConfig
from prime_rl.orchestrator.advantage_server_client import (
    ComputeAdvantagesRequest,
    ComputeAdvantagesResponse,
    DecisionPointDiagnostics,
)
from tests.unit.advantage_server.test_compute_action_diagnostics import (  # noqa: F401  (fixture)
    _CharTok,
    _two_sample_rollout,
    backbone,
)
from tests.unit.advantage_server.test_server_endpoint import _build_tiny_app, _CharTokenizer
from tests.unit.advantage_server.test_server_endpoint_diagnostics import _arm_sample

MARGIN = 0.10
ANCHOR = 0.3
# Ten turns cut from the stored probe records of the pps gate round, with the
# advantage the analysis scripts computed for them (an independent stdlib
# implementation). Written by the thesis-side parity_check.py, which checks all
# 273 turns.
PROBE_TURNS = Path(__file__).parent / "fixtures" / "probe_turns_pps_771426.json"


def _adv_all(q_plus, v, pi_hat, margin=MARGIN, anchor=ANCHOR):
    r = cleared_regrets(q_plus, v, margin)
    s = sum(r)
    return [difference_advantage(ri, s, p, anchor) for ri, p in zip(r, pi_hat)], r, s


# --------------------------------------------------------------------------- #
# defaults reproduce the original rule
# --------------------------------------------------------------------------- #


def test_defaults_reproduce_log_ratio():
    inp = ActionAdvantageInputs(
        q_plus=[[5.0, 3.0], [2.0, 9.0, 4.0], [1.0, 1.0]],
        v=[2.0, 3.0, 5.0],
        executed_idx=[0, 1, 1],
        pi_hat_star=[0.5, 0.2, 0.7],
        rewards=[0.0, 0.0, 1.0],
        v_target=[0.2, 0.9, 0.4],
        gamma=0.9,
        n_step=2,
        phi_decay=0.8,
    )
    assert (inp.advantage_form, inp.margin, inp.anchor, inp.target_form) == ("log_ratio", 0.0, 0.0, "sum")
    out = action_advantage_fn(inp)
    g = n_step_return_action_level(inp.rewards, inp.v_target, inp.gamma, inp.n_step)
    assert out.regret is None
    for k in range(3):
        rm = pi_rm(inp.q_plus[k], inp.v[k])
        e = inp.executed_idx[k]
        assert out.pi_rm[k] == rm
        assert out.advantage[k] == action_advantage(rm[e], inp.pi_hat_star[k])
        assert out.q_plus_target_out[k] == inp.phi_decay * max(0.0, inp.q_plus[k][e] - inp.v[k]) + g[k]
        assert out.v_target_out[k] == g[k]


def test_q_plus_target_default_form_is_sum():
    assert q_plus_target(0.9, 0.4, 1.0, 0.9) == 0.9 * 0.5 + 1.0
    assert q_plus_target(0.9, 0.4, 1.0, 0.9, "sum") == q_plus_target(0.9, 0.4, 1.0, 0.9)
    with pytest.raises(ValueError):
        q_plus_target(0.9, 0.4, 1.0, 0.9, "mean")


# --------------------------------------------------------------------------- #
# the difference advantage
# --------------------------------------------------------------------------- #


def test_cleared_regrets():
    assert cleared_regrets([0.9, 0.55, 0.5, 0.2], 0.4, 0.10) == pytest.approx([0.4, 0.05, 0.0, 0.0])
    assert cleared_regrets([0.5, 0.3], 0.4) == pytest.approx([0.1, 0.0])
    with pytest.raises(ValueError):
        cleared_regrets([], 0.0)
    with pytest.raises(ValueError):
        cleared_regrets([1.0], 0.0, -0.1)


def test_difference_bounded():
    rng = random.Random(0)
    for _ in range(2000):
        n = rng.randint(2, 30)
        v = rng.random()
        q = [rng.uniform(-0.5, 1.5) for _ in range(n)]
        raw = [rng.random() for _ in range(n)]
        z = sum(raw) / rng.uniform(0.5, 1.0)  # pi_hat sums to at most 1 (mass may leak)
        pi_hat = [x / z for x in raw]
        adv, _, s = _adv_all(q, v, pi_hat)
        bound = s / (s + ANCHOR)
        for a in adv:
            assert abs(a) < 1.0
            assert abs(a) <= bound + 1e-12


def test_difference_zero_without_cleared_regret():
    # every gap at or below the margin: exactly zero, and no fallback distribution
    inp = ActionAdvantageInputs(
        q_plus=[[0.45, 0.50, 0.30, 0.10]],
        v=[0.40],
        executed_idx=[2],
        pi_hat_star=[0.9],
        rewards=[1.0],
        v_target=[0.0],
        n_step=15,
        advantage_form="difference",
        margin=MARGIN,
        anchor=ANCHOR,
    )
    out = action_advantage_fn(inp)
    assert out.advantage == [0.0]
    assert out.regret == [[0.0, 0.0, 0.0, 0.0]]
    assert out.pi_rm == [[0.0, 0.0, 0.0, 0.0]]
    assert difference_advantage(0.0, 0.0, 0.3, ANCHOR) == 0.0


def test_difference_zero_at_fixed_point():
    q, v = [0.80, 0.60, 0.45, 0.10], 0.40
    r = cleared_regrets(q, v, MARGIN)
    s = sum(r)
    at_target = [x / s for x in r]
    adv, _, _ = _adv_all(q, v, at_target)
    assert adv == pytest.approx([0.0] * 4, abs=1e-12)
    # below its target share -> positive; above -> negative
    low = list(at_target)
    low[0] -= 0.15
    assert _adv_all(q, v, low)[0][0] > 0.0
    high = list(at_target)
    high[0] += 0.15
    assert _adv_all(q, v, high)[0][0] < 0.0
    # an action with no cleared regret and some policy mass is pushed down
    off = [at_target[0] - 0.1, at_target[1], 0.1, 0.0]
    assert _adv_all(q, v, off)[0][2] < 0.0


def test_difference_continuous_at_margin():
    v, step = 0.40, 1e-4
    pi_hat = [0.90, 0.06, 0.04]
    prev = None
    worst = 0.0
    for i in range(-200, 201):
        q0 = v + MARGIN + i * step  # sweeps action 0's gap across the margin
        adv, _, _ = _adv_all([q0, 0.75, 0.30], v, pi_hat)
        if prev is not None:
            worst = max(worst, max(abs(a - b) for a, b in zip(adv, prev)))
        prev = adv
    assert worst < 20 * step  # a jump of the order of the step: no cliff


def test_scale_is_read():
    # the diary's worked example: same pattern of gaps, ten times smaller
    v = 0.30
    pi_hat = [0.02, 0.03, 0.90, 0.05]
    gaps = [0.40, 0.06, -0.004, -0.20]
    adv, r, s = _adv_all([v + g for g in gaps], v, pi_hat)
    assert r == pytest.approx([0.30, 0.0, 0.0, 0.0])
    assert s == pytest.approx(0.30)
    assert adv[2] == pytest.approx(-0.45)
    assert adv[0] == pytest.approx(0.49)
    small, _, _ = _adv_all([v + g / 10 for g in gaps], v, pi_hat)
    assert small == [0.0, 0.0, 0.0, 0.0]
    # the original rule cannot tell the two states apart
    for scale in (1.0, 0.1):
        rm = pi_rm([v + g * scale for g in gaps], v)
        assert action_advantage(rm[2], pi_hat[2]) == pytest.approx(math.log(1e-6) - math.log(0.9))


def test_leak_pushes_valid_actions_up():
    q, v = [0.80, 0.60, 0.45, 0.10], 0.40
    r = cleared_regrets(q, v, MARGIN)
    s = sum(r)
    leaked = [0.9 * x / s for x in r]  # at the target's shape, 10 % of the mass gone
    adv, _, _ = _adv_all(q, v, leaked)
    for a, ri in zip(adv, r):
        if ri > 0:
            assert a > 0.0
        else:
            assert a == 0.0


def test_difference_needs_positive_anchor():
    with pytest.raises(ValueError):
        difference_advantage(0.1, 0.2, 0.5, 0.0)
    kwargs = dict(
        q_plus=[[0.9, 0.2]], v=[0.4], executed_idx=[0], pi_hat_star=[0.5], rewards=[1.0], v_target=[0.0]
    )
    with pytest.raises(ValueError):
        ActionAdvantageInputs(**kwargs, advantage_form="difference", anchor=0.0)
    with pytest.raises(ValueError):
        ActionAdvantageInputs(**kwargs, advantage_form="ratio")
    with pytest.raises(ValueError):
        ActionAdvantageInputs(**kwargs, target_form="mean")
    with pytest.raises(ValueError):
        ActionAdvantageInputs(**kwargs, margin=-0.1)


def test_fn_difference_outputs():
    inp = ActionAdvantageInputs(
        q_plus=[[0.90, 0.20, 0.45], [0.30, 0.35]],
        v=[0.40, 0.40],
        executed_idx=[1, 0],
        pi_hat_star=[0.8, 0.6],
        rewards=[0.0, 1.0],
        v_target=[0.0, 0.0],
        gamma=1.0,
        n_step=15,
        phi_decay=0.9,
        advantage_form="difference",
        margin=MARGIN,
        anchor=ANCHOR,
        target_form="average",
    )
    out = action_advantage_fn(inp)
    assert out.g == [1.0, 1.0]  # horizon >= K, gamma = 1: the terminal reward everywhere
    assert out.regret[0] == pytest.approx([0.40, 0.0, 0.0])
    assert out.pi_rm[0] == pytest.approx([1.0, 0.0, 0.0])
    assert out.advantage[0] == pytest.approx((0.0 - 0.40 * 0.8) / (0.40 + ANCHOR))
    assert out.regret[1] == [0.0, 0.0] and out.pi_rm[1] == [0.0, 0.0] and out.advantage[1] == 0.0
    # averaged target: phi * max(Q+, V) + (1 - phi) * g on the executed action
    assert out.q_plus_target_out[0] == pytest.approx(0.9 * 0.40 + 0.1 * 1.0)  # Q+ 0.20 < V 0.40
    assert out.q_plus_target_out[1] == pytest.approx(0.9 * 0.40 + 0.1 * 1.0)  # Q+ 0.30 < V 0.40
    assert out.v_target_out == [1.0, 1.0]


def test_parity_with_stored_probe_turns():
    fixture = json.loads(PROBE_TURNS.read_text(encoding="utf-8"))
    turns = fixture["turns"]
    assert len(turns) == 10
    got = []
    for t in turns:
        out = action_advantage_fn(
            ActionAdvantageInputs(
                q_plus=[t["q_plus"]],
                v=[t["v"]],
                executed_idx=[t["executed_idx"]],
                pi_hat_star=[t["pi_hat_star"]],
                rewards=[0.0],
                v_target=[0.0],
                n_step=15,
                advantage_form="difference",
                margin=fixture["margin"],
                anchor=fixture["anchor"],
            )
        )
        assert out.advantage[0] == pytest.approx(t["expected_adv"], abs=1e-9)
        got.append(out.advantage[0])
    # the cut spans the range the rule produces on real turns: exact zeros and
    # strong negative pushes, nothing outside (-1, 1)
    assert any(a == 0.0 for a in got) and min(got) < -0.8 and max(abs(a) for a in got) < 1.0


# --------------------------------------------------------------------------- #
# the averaged Q+ target
# --------------------------------------------------------------------------- #


def _settle(form: str, inst: float, phi: float = 0.9, steps: int = 400) -> float:
    v, x = 0.40, 0.0
    for _ in range(steps):
        x = q_plus_target(v + x, v, v + inst, phi, form) - v
    return x


def test_average_target_fixed_point():
    assert _settle("average", +0.05) == pytest.approx(0.05)
    assert _settle("sum", +0.05) == pytest.approx(0.50)  # 1 / (1 - 0.9) times the inflow
    assert _settle("average", +0.05, phi=0.97, steps=2000) == pytest.approx(0.05)
    # negative regret does not accumulate in either form
    assert _settle("average", -0.05) == pytest.approx(-0.005)
    assert _settle("sum", -0.05) == pytest.approx(-0.05)


def test_average_target_first_updates():
    # the leaky average climbs toward the inflow: 0.005, 0.0095, ...
    v, x, seen = 0.40, 0.0, []
    for _ in range(3):
        x = q_plus_target(v + x, v, v + 0.05, 0.9, "average") - v
        seen.append(x)
    assert seen == pytest.approx([0.005, 0.0095, 0.01355])


def test_average_target_bounds():
    rng = random.Random(1)
    for _ in range(2000):
        q, v, g, phi = rng.random(), rng.random(), float(rng.random() < 0.5), rng.random()
        t = q_plus_target(q, v, g, phi, "average")
        assert -1e-12 <= t <= 1.0 + 1e-12
        assert min(g, max(q, v)) - 1e-12 <= t <= max(g, max(q, v)) + 1e-12


def test_phi_zero_is_plain_return():
    assert q_plus_target(0.9, 0.4, 0.0, 0.0, "average") == 0.0
    assert q_plus_target(0.1, 0.4, 1.0, 0.0, "average") == 1.0


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #


def test_config_defaults_and_validator():
    cfg = AdvantageServerConfig(model_name="m")
    assert cfg.arm_action_advantage_form == "log_ratio"
    assert cfg.arm_margin == 0.0 and cfg.arm_anchor == 0.0 and cfg.arm_target_form == "sum"
    ok = AdvantageServerConfig(
        model_name="m",
        arm_action_advantage_form="difference",
        arm_margin=0.10,
        arm_anchor=0.3,
        arm_target_form="average",
    )
    assert ok.arm_anchor == 0.3
    with pytest.raises(ValidationError, match="arm_anchor"):
        AdvantageServerConfig(model_name="m", arm_action_advantage_form="difference")
    with pytest.raises(ValidationError):
        AdvantageServerConfig(model_name="m", arm_margin=-0.1)
    with pytest.raises(ValidationError):
        AdvantageServerConfig(model_name="m", arm_target_form="mean")


# --------------------------------------------------------------------------- #
# the settings reach the math (compute path, tiny CPU backbone)
# --------------------------------------------------------------------------- #


def test_settings_reach_the_math(backbone):  # noqa: F811
    samples = _two_sample_rollout()
    # Lower V so that Q+ sits clearly above it (with the fixture's random heads Q+
    # is about -3 to -7 and V about 0 to 8, so every gap is under the margin;
    # that case is covered by the cold-start endpoint test below).
    backbone.v_head.linear.bias.data -= 20.0
    paired, diags = compute_advantages_and_targets_arm_action(
        samples, 1.0, True, backbone, _CharTok(),
        gamma=1.0, n_step=15, arm_phi_decay=0.9, return_diagnostics=True,
        arm_action_advantage_form="difference", arm_margin=0.05, arm_anchor=ANCHOR,
        arm_target_form="average",
    )
    records = _build_per_decision_point_value_tensors(samples, backbone, _CharTok())
    assert len(diags) == len(records) == 3
    some_regret = False
    for d, r in zip(diags, records):
        reg = cleared_regrets(r["q_plus"], r["v"], 0.05)
        total = sum(reg)
        e = r["executed_idx"]
        some_regret = some_regret or total > 0
        assert d.regret == pytest.approx(reg, abs=1e-6)
        assert d.anchor == pytest.approx(ANCHOR)
        assert d.pi_rm == pytest.approx([x / total for x in reg] if total > 0 else [0.0] * len(reg), abs=1e-6)
        assert d.adv_star == pytest.approx(difference_advantage(reg[e], total, r["pi_hat"], ANCHOR), abs=1e-6)
        assert abs(d.adv_star) < 1.0
        assert d.g_k == pytest.approx(1.0)
        assert d.q_plus_target_star == pytest.approx(
            q_plus_target(r["q_plus"][e], r["v"], 1.0, 0.9, "average"), abs=1e-6
        )
    assert some_regret  # the fixture's non-zero heads give at least one cleared regret
    # the broadcast advantage on the LLM sample is the same number
    first = diags[0]
    assert paired[0][0].advantages[first.response_start] == pytest.approx(first.adv_star, abs=1e-6)


def test_default_form_diagnostics_unchanged(backbone):  # noqa: F811
    samples = _two_sample_rollout()
    _, diags = compute_advantages_and_targets_arm_action(
        samples, 1.0, True, backbone, _CharTok(), gamma=1.0, n_step=5, arm_phi_decay=0.9,
        return_diagnostics=True,
    )
    assert all(d.regret is None and d.anchor is None for d in diags)
    for d in diags:
        assert sum(d.pi_rm) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# wire and endpoint
# --------------------------------------------------------------------------- #


class _OldDiagnostics(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    """DecisionPointDiagnostics before the redesign (nine fields)."""

    sample_idx: int
    response_start: int
    q_plus: list[float]
    v: float
    v_target: float
    pi_rm: list[float]
    adv_star: float
    g_k: float
    q_plus_target_star: float


def test_diagnostics_wire_compat():
    old = _OldDiagnostics(0, 4, [0.1, 0.2], 0.3, 0.3, [0.5, 0.5], -0.2, 1.0, 0.9)
    decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(old), type=DecisionPointDiagnostics)
    assert decoded.regret is None and decoded.anchor is None and decoded.q_plus == [0.1, 0.2]
    new = DecisionPointDiagnostics(0, 4, [0.1, 0.2], 0.3, 0.3, [1.0, 0.0], -0.2, 1.0, 0.9, [0.05, 0.0], 0.3)
    again = msgspec.msgpack.decode(msgspec.msgpack.encode(new), type=DecisionPointDiagnostics)
    assert again.regret == [0.05, 0.0] and again.anchor == 0.3
    # an older reader drops the two trailing fields
    as_old = msgspec.msgpack.decode(msgspec.msgpack.encode(new), type=_OldDiagnostics)
    assert as_old.pi_rm == [1.0, 0.0]


def test_endpoint_difference_form_cold_start(monkeypatch: pytest.MonkeyPatch):
    """Zero-initialised heads: no regret clears the margin, so the advantage is
    exactly zero and no uniform fallback appears (the original rule returns
    log(1/3) - log(0.4) on the same request)."""
    monkeypatch.setattr("transformers.AutoTokenizer", _CharTokenizer)
    app = _build_tiny_app(monkeypatch)
    app.state.arm_action_advantage_form = "difference"
    app.state.arm_margin = MARGIN
    app.state.arm_anchor = ANCHOR
    app.state.arm_target_form = "average"
    app.state.arm_phi_decay = 0.9
    app.state.n_step = 15
    body = msgspec.msgpack.encode(
        ComputeAdvantagesRequest(
            samples=[_arm_sample()], episodic_reward=1.0, is_terminal=True, algorithm="arm",
            return_diagnostics=True,
        )
    )
    with TestClient(app) as client:
        r = client.post(
            "/compute_advantages_and_targets", content=body, headers={"Content-Type": "application/x-msgpack"}
        )
    assert r.status_code == 200
    decoded = msgspec.msgpack.decode(r.content, type=ComputeAdvantagesResponse)
    diag = decoded.diagnostics[0]
    assert diag.regret == [0.0, 0.0, 0.0] and diag.anchor == pytest.approx(ANCHOR)
    assert diag.pi_rm == [0.0, 0.0, 0.0]
    assert diag.adv_star == 0.0
    assert decoded.paired_samples[0].llm_sample.advantages == [0.0, 0.0, 0.0]
    # averaged target with Q+ = V = 0 and g = 1: (1 - 0.9) * 1
    assert diag.q_plus_target_star == pytest.approx(0.1)
    assert decoded.paired_samples[0].advantage_sample.decision_point_targets[0].q_plus_target == pytest.approx(0.1)
