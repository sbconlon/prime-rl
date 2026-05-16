"""Phase 6b -- Advantage Server HTTP endpoint tests via FastAPI's TestClient.

In-process tests: TestClient invokes the FastAPI app's startup hook (lazy
backbone load) and then routes HTTP requests through the app's ASGI
interface without spinning up uvicorn or a subprocess. Suitable for the
laptop-CPU run; the model is a tiny 2-layer Qwen2 (no real download).

For real-subprocess + GPU tests, see Phase 6's manual e2e smoke (deferred
to cluster).
"""

from __future__ import annotations

import msgspec
import pytest
import torch
from fastapi.testclient import TestClient
from transformers import Qwen2Config, Qwen2Model

from prime_rl.advantage_server.server import create_app
from prime_rl.configs.advantage_server import AdvantageServerConfig
from prime_rl.configs.trainer import LoRAConfig
from prime_rl.orchestrator.advantage_server_client import (
    ComputeAdvantagesRequest,
    ComputeAdvantagesResponse,
)
from prime_rl.orchestrator.value_networks import ValueNetworkBackbone
from prime_rl.transport.types import TrainingSample


def _tiny_qwen2_config() -> Qwen2Config:
    return Qwen2Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
    )


def _build_tiny_app(monkeypatch: pytest.MonkeyPatch):
    """Build the FastAPI app with the tiny Qwen2 backbone preloaded.

    The production startup hook calls AutoModel.from_pretrained which would
    download a HuggingFace model. We monkeypatch the loader to return a tiny
    config-built model (same approach as test_value_networks.py).
    """
    from prime_rl.advantage_server import server as server_module

    class _TinyAutoModel:
        @staticmethod
        def from_pretrained(name: str, **kwargs):  # noqa: ARG004 -- accept kwargs (dtype=)
            # Server passes `dtype=` since the 2026-05-11 dtype-on-load fix;
            # accept and ignore -- the tiny test model stays in fp32.
            torch.manual_seed(0)
            return Qwen2Model(_tiny_qwen2_config())

    monkeypatch.setattr(server_module, "AutoModel", _TinyAutoModel)

    config = AdvantageServerConfig(
        model_name="ignored-by-monkeypatch",
        lora=LoRAConfig(rank=8, alpha=16.0, dropout=0.0),
        polyak_tau=0.005,
        host="127.0.0.1",
        port=8200,
        n_step=5,
    )
    return create_app(config)


def _make_sample(
    completion_len: int,
    *,
    completion_top_k_token_ids: list[list[int]] | None = None,
) -> TrainingSample:
    return TrainingSample(
        prompt_ids=[1, 2],
        prompt_mask=[False, False],
        completion_ids=list(range(10, 10 + completion_len)),
        completion_mask=[True] * completion_len,
        completion_logprobs=[0.0] * completion_len,
        completion_temperatures=[1.0] * completion_len,
        completion_top_k_token_ids=completion_top_k_token_ids,
    )


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


def test_health_endpoint_returns_200_after_startup(monkeypatch: pytest.MonkeyPatch):
    """/health is 200 OK once the lifespan startup hook has loaded the backbone."""
    app = _build_tiny_app(monkeypatch)
    with TestClient(app) as client:
        # TestClient's context manager runs the startup hook before yielding.
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Compute endpoint round-trip
# ---------------------------------------------------------------------------


def test_compute_endpoint_round_trip_ppo(monkeypatch: pytest.MonkeyPatch):
    """POST /compute_advantages_and_targets with a PPO request returns paired outputs.

    Iteration-0 regime with zero-init backbone: V=V_target=0 everywhere; PPO GAE
    gives advantages = (gamma*lam)^(N-1-k) * episodic_reward at mask=True positions.
    """
    app = _build_tiny_app(monkeypatch)
    sample = _make_sample(completion_len=2)
    request = ComputeAdvantagesRequest(
        samples=[sample], episodic_reward=1.0, is_terminal=True, algorithm="ppo"
    )
    body = msgspec.msgpack.encode(request)

    with TestClient(app) as client:
        response = client.post(
            "/compute_advantages_and_targets",
            content=body,
            headers={"Content-Type": "application/x-msgpack"},
        )
        assert response.status_code == 200
        decoded = msgspec.msgpack.decode(response.content, type=ComputeAdvantagesResponse)

    assert len(decoded.paired_samples) == 1
    paired = decoded.paired_samples[0]
    # Token shapes mirror the input.
    assert paired.llm_sample.completion_ids == sample.completion_ids
    assert paired.advantage_sample.completion_ids == sample.completion_ids
    # PPO -> q_plus_targets None.
    assert paired.advantage_sample.q_plus_targets is None
    # Advantages populated; non-zero (closed-form (gamma*lam)^(N-1-k) * 1.0)
    assert paired.llm_sample.advantages is not None
    assert len(paired.llm_sample.advantages) == 2
    assert paired.llm_sample.advantages[-1] == pytest.approx(1.0, abs=1e-5)


def test_compute_endpoint_round_trip_arm(monkeypatch: pytest.MonkeyPatch):
    """POST /compute_advantages_and_targets with an ARM request returns paired outputs.

    Iteration-0 regime: cold-start branch fires (advantages all zero);
    q_plus_targets follow phi + g = 0 + g = n-step return.
    """
    K = 4
    app = _build_tiny_app(monkeypatch)
    sample = _make_sample(
        completion_len=2,
        completion_top_k_token_ids=[[10 + j for j in range(K)] for _ in range(2)],
    )
    request = ComputeAdvantagesRequest(
        samples=[sample], episodic_reward=1.0, is_terminal=True, algorithm="arm"
    )
    body = msgspec.msgpack.encode(request)

    with TestClient(app) as client:
        response = client.post(
            "/compute_advantages_and_targets",
            content=body,
            headers={"Content-Type": "application/x-msgpack"},
        )
        assert response.status_code == 200
        decoded = msgspec.msgpack.decode(response.content, type=ComputeAdvantagesResponse)

    paired = decoded.paired_samples[0]
    assert paired.advantage_sample.q_plus_targets is not None
    # Cold-start: advantages all zero.
    for a in paired.llm_sample.advantages or []:
        assert abs(a) < 1e-6


def test_compute_endpoint_multi_sample_request(monkeypatch: pytest.MonkeyPatch):
    """Multi-sample request returns paired outputs in matching order."""
    app = _build_tiny_app(monkeypatch)
    s0 = _make_sample(completion_len=2)
    s1 = _make_sample(completion_len=3)
    request = ComputeAdvantagesRequest(
        samples=[s0, s1], episodic_reward=1.0, is_terminal=True, algorithm="ppo"
    )
    body = msgspec.msgpack.encode(request)

    with TestClient(app) as client:
        response = client.post(
            "/compute_advantages_and_targets",
            content=body,
            headers={"Content-Type": "application/x-msgpack"},
        )
        assert response.status_code == 200
        decoded = msgspec.msgpack.decode(response.content, type=ComputeAdvantagesResponse)

    assert len(decoded.paired_samples) == 2
    assert decoded.paired_samples[0].llm_sample.completion_ids == s0.completion_ids
    assert decoded.paired_samples[1].llm_sample.completion_ids == s1.completion_ids


def test_compute_endpoint_serial_consistency(monkeypatch: pytest.MonkeyPatch):
    """Same request sent twice returns identical responses (no random state)."""
    app = _build_tiny_app(monkeypatch)
    sample = _make_sample(completion_len=2)
    request = ComputeAdvantagesRequest(
        samples=[sample], episodic_reward=1.0, is_terminal=True, algorithm="ppo"
    )
    body = msgspec.msgpack.encode(request)

    with TestClient(app) as client:
        r1 = client.post(
            "/compute_advantages_and_targets",
            content=body,
            headers={"Content-Type": "application/x-msgpack"},
        )
        r2 = client.post(
            "/compute_advantages_and_targets",
            content=body,
            headers={"Content-Type": "application/x-msgpack"},
        )
    assert r1.status_code == r2.status_code == 200
    assert r1.content == r2.content


# ---------------------------------------------------------------------------
# Weight status endpoint (AdvTrainer<->orchestrator lockstep barrier)
# ---------------------------------------------------------------------------


def test_weight_status_initial_value_is_zero(monkeypatch: pytest.MonkeyPatch):
    """/weight_status reports 0 before any /update_weights POST.

    This is the initial state the orchestrator's step-0 wait_for_weight_step
    short-circuits against: AdvServer.weight_step=0 satisfies min_step=0 with
    no AdvTrainer broadcast required (the zero-init backbone is the correct
    bootstrap state).
    """
    app = _build_tiny_app(monkeypatch)
    with TestClient(app) as client:
        response = client.get("/weight_status")
        assert response.status_code == 200
        assert response.json() == {"weight_step": 0}


def test_weight_status_increments_after_update_weights(monkeypatch: pytest.MonkeyPatch):
    """Each successful /update_weights POST increments weight_step by 1.

    Simulates the AdvTrainer broadcast: serialize the backbone's own state_dict
    (trivially-valid bytes), POST it to /update_weights, observe weight_step
    advance. Two POSTs -> weight_step goes 0 -> 1 -> 2. This is the signal the
    orchestrator's wait_for_weight_step polls; pre-increment means the
    orchestrator can rely on weight_step as an "applied" not "received" counter.
    """
    import io
    import torch

    app = _build_tiny_app(monkeypatch)
    with TestClient(app) as client:
        # Sanity: starts at 0.
        assert client.get("/weight_status").json() == {"weight_step": 0}

        # Build a trivially-valid LoRA+heads state_dict by reading the live
        # backbone's trainable params. POSTing this back is a no-op
        # mathematically but exercises the full /update_weights path.
        backbone = app.state.backbone
        trainable_state = {
            name: param.detach().cpu()
            for name, param in backbone.named_parameters()
            if param.requires_grad
        }
        buf = io.BytesIO()
        torch.save(trainable_state, buf)
        body = buf.getvalue()

        r1 = client.post(
            "/update_weights",
            content=body,
            headers={"Content-Type": "application/octet-stream"},
        )
        assert r1.status_code == 200
        assert client.get("/weight_status").json() == {"weight_step": 1}

        r2 = client.post(
            "/update_weights",
            content=body,
            headers={"Content-Type": "application/octet-stream"},
        )
        assert r2.status_code == 200
        assert client.get("/weight_status").json() == {"weight_step": 2}


def test_weight_status_independent_of_compute_calls(monkeypatch: pytest.MonkeyPatch):
    """/compute_advantages_and_targets does NOT modify weight_step.

    The two endpoints are independent: compute is read-only on weight_step;
    only /update_weights advances the counter. Without this guarantee the
    orchestrator's barrier could be accidentally satisfied by issuing
    compute calls instead of by the AdvTrainer's broadcast.
    """
    app = _build_tiny_app(monkeypatch)
    sample = _make_sample(completion_len=2)
    request = ComputeAdvantagesRequest(
        samples=[sample], episodic_reward=1.0, is_terminal=True, algorithm="ppo"
    )
    body = msgspec.msgpack.encode(request)

    with TestClient(app) as client:
        assert client.get("/weight_status").json() == {"weight_step": 0}
        for _ in range(3):
            r = client.post(
                "/compute_advantages_and_targets",
                content=body,
                headers={"Content-Type": "application/x-msgpack"},
            )
            assert r.status_code == 200
        # No /update_weights -> weight_step unchanged.
        assert client.get("/weight_status").json() == {"weight_step": 0}
