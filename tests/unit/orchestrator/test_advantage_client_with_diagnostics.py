"""training-probe Phase 2 -- AdvantageServerClient.compute_with_diagnostics and the
FakeAdvantageServer counterpart.

The real client is driven against the tiny FastAPI app through an in-process
ASGI transport (startup run by TestClient); the fake is exercised directly.
"""

from __future__ import annotations

import asyncio
import math

import httpx
import pytest
from fastapi.testclient import TestClient

from prime_rl.configs.advantage_server import AdvantageServerClientConfig
from prime_rl.orchestrator.advantage_server_client import (
    AdvantageServerClient,
    DecisionPointDiagnostics,
    DiagnosticsResult,
)
from prime_rl.transport.types import DecisionPoint, TrainingSample
from tests.unit.advantage_server.test_server_endpoint import _build_tiny_app, _CharTokenizer
from tests.unit.orchestrator._stubs import FakeAdvantageServer


def _arm_sample(pi_hat=0.4) -> TrainingSample:
    return TrainingSample(
        prompt_ids=[1, 2],
        prompt_mask=[False, False],
        completion_ids=[10, 11, 12],
        completion_mask=[True, True, True],
        completion_logprobs=[0.0] * 3,
        completion_temperatures=[1.0] * 3,
        decision_points=[DecisionPoint(0, 3, ["go", "look", "take"], 1, pi_hat=pi_hat)],
    )


def _asgi_client(app) -> AdvantageServerClient:
    client = AdvantageServerClient(
        AdvantageServerClientConfig(base_url="http://adv.test", request_timeout=60.0)
    )
    asyncio.run(client._client.aclose())
    client._client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://adv.test"
    )
    return client


def test_compute_with_diagnostics_roundtrip(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("transformers.AutoTokenizer", _CharTokenizer)
    app = _build_tiny_app(monkeypatch)
    sample = _arm_sample()
    with TestClient(app):  # runs the startup hook (backbone + tokenizer)
        client = _asgi_client(app)
        try:
            result = asyncio.run(
                client.compute_with_diagnostics(
                    samples=[sample], episodic_reward=1.0, is_terminal=True, algorithm="arm"
                )
            )
            plain = asyncio.run(
                client.compute_advantages_and_targets(
                    samples=[sample], episodic_reward=1.0, is_terminal=True, algorithm="arm"
                )
            )
        finally:
            asyncio.run(client.aclose())

    assert isinstance(result, DiagnosticsResult)
    assert result.weight_step == 0
    assert len(result.paired) == 1 and len(result.diagnostics) == 1
    d = result.diagnostics[0]
    assert isinstance(d, DecisionPointDiagnostics)
    assert len(d.q_plus) == len(d.pi_rm) == 3
    assert d.adv_star == pytest.approx(math.log(1 / 3) - math.log(0.4), abs=1e-6)
    # the existing method is untouched and agrees with the paired half
    llm, adv = plain[0]
    assert llm.advantages == pytest.approx(result.paired[0][0].advantages)
    assert adv.decision_point_targets[0].v_target == pytest.approx(
        result.paired[0][1].decision_point_targets[0].v_target
    )


def test_fake_compute_with_diagnostics():
    fake = FakeAdvantageServer(initial_weight_step=2)
    s0 = _arm_sample(pi_hat=0.4)
    s1 = TrainingSample(
        prompt_ids=[5],
        prompt_mask=[False],
        completion_ids=[20, 21],
        completion_mask=[True, True],
        completion_logprobs=[0.0, 0.0],
        completion_temperatures=[1.0, 1.0],
        decision_points=[
            DecisionPoint(0, 1, ["a", "b"], 0, pi_hat=None),
            DecisionPoint(1, 2, ["x"], 0, pi_hat=1.0),
        ],
    )
    result = asyncio.run(
        fake.compute_with_diagnostics(
            samples=[s0, s1], episodic_reward=1.0, is_terminal=True, algorithm="arm"
        )
    )
    assert isinstance(result, DiagnosticsResult)
    assert result.weight_step == 2
    assert len(result.paired) == 2
    assert [(d.sample_idx, d.response_start) for d in result.diagnostics] == [(0, 0), (1, 0), (1, 1)]
    d0, d1, d2 = result.diagnostics
    assert d0.pi_rm == pytest.approx([1 / 3] * 3) and d0.q_plus == [0.0, 0.0, 0.0]
    assert d0.adv_star == pytest.approx(math.log(1 / 3) - math.log(0.4))
    assert d1.adv_star == 0.0  # pi_hat None -> placeholder zero, as the server does
    assert d2.pi_rm == [1.0] and d2.adv_star == pytest.approx(0.0)
    # the call is recorded with the flag
    assert fake.calls[-1]["return_diagnostics"] is True
    assert fake.calls[-1]["weight_step_at_call"] == 2
    # PPO -> no diagnostics
    ppo = asyncio.run(
        fake.compute_with_diagnostics(
            samples=[s0], episodic_reward=0.0, is_terminal=True, algorithm="ppo"
        )
    )
    assert ppo.diagnostics == []


def test_fake_diagnostics_factory_override():
    canned = [
        DecisionPointDiagnostics(
            sample_idx=0, response_start=0, q_plus=[0.2, 0.1, 0.0], v=0.05, v_target=0.1,
            pi_rm=[0.6, 0.4, 0.0], adv_star=0.1, g_k=1.0, q_plus_target_star=1.15,
        )
    ]
    fake = FakeAdvantageServer(diagnostics_factory=lambda *a, **k: canned)
    result = asyncio.run(
        fake.compute_with_diagnostics(
            samples=[_arm_sample()], episodic_reward=1.0, is_terminal=True, algorithm="arm"
        )
    )
    assert result.diagnostics is canned


def test_both_clients_expose_the_protocol_method():
    assert callable(getattr(AdvantageServerClient, "compute_with_diagnostics"))
    assert callable(getattr(FakeAdvantageServer, "compute_with_diagnostics"))
