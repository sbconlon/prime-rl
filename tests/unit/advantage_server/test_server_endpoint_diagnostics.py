"""training-probe Phase 2 -- /compute_advantages_and_targets diagnostics via TestClient.

Reuses the tiny-app harness from test_server_endpoint.py. Pins: with the flag
the response carries diagnostics and the serving weight_step (which advances
after /update_weights); without the flag the response is the pre-probe one
plus two trailing nils and decodes with the legacy one-field struct; PPO with
the flag yields empty diagnostics.
"""

from __future__ import annotations

import io
import math

import msgspec
import pytest
import torch
from fastapi.testclient import TestClient

from prime_rl.orchestrator.advantage_server_client import (
    ComputeAdvantagesRequest,
    ComputeAdvantagesResponse,
    PairedSample,
)
from prime_rl.transport.types import DecisionPoint, TrainingSample
from tests.unit.advantage_server.test_server_endpoint import (
    _build_tiny_app,
    _CharTokenizer,
    _make_sample,
)

_HEADERS = {"Content-Type": "application/x-msgpack"}


class _LegacyResponse(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    """The pre-probe one-field response (what an older client decodes with)."""

    paired_samples: list[PairedSample]


def _arm_sample() -> TrainingSample:
    return TrainingSample(
        prompt_ids=[1, 2],
        prompt_mask=[False, False],
        completion_ids=[10, 11, 12],
        completion_mask=[True, True, True],
        completion_logprobs=[0.0] * 3,
        completion_temperatures=[1.0] * 3,
        decision_points=[DecisionPoint(0, 3, ["go", "look", "take"], 1, pi_hat=0.4)],
    )


def _post_update_weights(client: TestClient, app) -> None:
    backbone = app.state.backbone
    trainable_state = {
        name: param.detach().cpu()
        for name, param in backbone.named_parameters()
        if param.requires_grad
    }
    buf = io.BytesIO()
    torch.save(trainable_state, buf)
    r = client.post("/update_weights", content=buf.getvalue())
    assert r.status_code == 200


def test_endpoint_attaches_diagnostics_and_weight_step(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("transformers.AutoTokenizer", _CharTokenizer)
    app = _build_tiny_app(monkeypatch)
    sample = _arm_sample()
    body = msgspec.msgpack.encode(
        ComputeAdvantagesRequest(
            samples=[sample], episodic_reward=1.0, is_terminal=True, algorithm="arm",
            return_diagnostics=True,
        )
    )
    with TestClient(app) as client:
        r1 = client.post("/compute_advantages_and_targets", content=body, headers=_HEADERS)
        assert r1.status_code == 200
        d1 = msgspec.msgpack.decode(r1.content, type=ComputeAdvantagesResponse)
        assert d1.weight_step == 0 == app.state.weight_step

        _post_update_weights(client, app)
        r2 = client.post("/compute_advantages_and_targets", content=body, headers=_HEADERS)
        d2 = msgspec.msgpack.decode(r2.content, type=ComputeAdvantagesResponse)
        assert d2.weight_step == 1 == app.state.weight_step

    assert d1.diagnostics is not None and len(d1.diagnostics) == 1
    diag = d1.diagnostics[0]
    assert diag.sample_idx == 0 and diag.response_start == 0
    assert len(diag.q_plus) == len(diag.pi_rm) == 3
    assert sum(diag.pi_rm) == pytest.approx(1.0, abs=1e-9)
    # cold start: zero-init Q+ head -> uniform pi_RM -> adv = log(1/3) - log(0.4)
    assert diag.pi_rm == pytest.approx([1 / 3] * 3)
    assert diag.adv_star == pytest.approx(math.log(1 / 3) - math.log(0.4), abs=1e-6)
    assert diag.g_k == pytest.approx(1.0)  # terminal reward, n_step=5 >= K=1
    # the paired outputs carry the same numbers
    paired = d1.paired_samples[0]
    assert paired.llm_sample.advantages[0] == pytest.approx(diag.adv_star, abs=1e-6)
    assert paired.advantage_sample.decision_point_targets[0].v_target == pytest.approx(diag.g_k)
    assert paired.advantage_sample.decision_point_targets[0].q_plus_target == pytest.approx(
        diag.q_plus_target_star
    )
    # the second call (same weights, just re-loaded) gives the same diagnostics
    assert d2.diagnostics[0].q_plus == pytest.approx(diag.q_plus, abs=1e-6)


def test_endpoint_without_flag_unchanged(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("transformers.AutoTokenizer", _CharTokenizer)
    app = _build_tiny_app(monkeypatch)
    sample = _arm_sample()

    class _LegacyRequest(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
        samples: list[TrainingSample]
        episodic_reward: float
        is_terminal: bool
        algorithm: str

    legacy_body = msgspec.msgpack.encode(
        _LegacyRequest(samples=[sample], episodic_reward=1.0, is_terminal=True, algorithm="arm")
    )
    assert len(msgspec.msgpack.decode(legacy_body)) == 4  # the pre-probe 4-element array

    with TestClient(app) as client:
        r = client.post("/compute_advantages_and_targets", content=legacy_body, headers=_HEADERS)
        assert r.status_code == 200

    decoded = msgspec.msgpack.decode(r.content, type=ComputeAdvantagesResponse)
    assert decoded.diagnostics is None
    assert decoded.weight_step is None
    # On the wire: the pre-probe one-element array plus two trailing nils.
    raw = msgspec.msgpack.decode(r.content)
    assert isinstance(raw, list) and len(raw) == 3
    assert raw[1] is None and raw[2] is None
    legacy = msgspec.msgpack.decode(r.content, type=_LegacyResponse)
    assert len(legacy.paired_samples) == 1
    assert legacy.paired_samples[0].llm_sample.advantages == pytest.approx(
        decoded.paired_samples[0].llm_sample.advantages
    )
    assert legacy.paired_samples[0].advantage_sample.decision_point_targets is not None


def test_endpoint_ppo_with_flag_empty_diagnostics(monkeypatch: pytest.MonkeyPatch):
    app = _build_tiny_app(monkeypatch)
    sample = _make_sample(completion_len=2)
    body = msgspec.msgpack.encode(
        ComputeAdvantagesRequest(
            samples=[sample], episodic_reward=1.0, is_terminal=True, algorithm="ppo",
            return_diagnostics=True,
        )
    )
    with TestClient(app) as client:
        r = client.post("/compute_advantages_and_targets", content=body, headers=_HEADERS)
        assert r.status_code == 200
    decoded = msgspec.msgpack.decode(r.content, type=ComputeAdvantagesResponse)
    assert decoded.diagnostics == []
    assert decoded.weight_step == 0
    assert len(decoded.paired_samples) == 1
