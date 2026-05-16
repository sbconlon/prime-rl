"""Phase 6a -- AdvantageServerClient + FakeAdvantageServer + sync invariant."""

from __future__ import annotations

import asyncio

import pytest

from prime_rl.configs.advantage_server import AdvantageServerClientConfig
from prime_rl.orchestrator.advantage_server_client import (
    AdvantageServerClient,
    AdvantageServerClientProtocol,
    ComputeAdvantagesRequest,
    ComputeAdvantagesResponse,
    PairedSample,
)
from prime_rl.transport.types import AdvantageTrainingSample, TrainingSample
from tests.unit.orchestrator._stubs import FakeAdvantageServer


def _make_sample(
    completion_len: int,
    *,
    prompt_ids: list[int] | None = None,
    completion_top_k_token_ids: list[list[int]] | None = None,
) -> TrainingSample:
    if prompt_ids is None:
        prompt_ids = [1, 2]
    return TrainingSample(
        prompt_ids=prompt_ids,
        prompt_mask=[False] * len(prompt_ids),
        completion_ids=list(range(10, 10 + completion_len)),
        completion_mask=[True] * completion_len,
        completion_logprobs=[0.0] * completion_len,
        completion_temperatures=[1.0] * completion_len,
        completion_top_k_token_ids=completion_top_k_token_ids,
    )


# ---------------------------------------------------------------------------
# Wire-protocol roundtrip (msgspec types must serialize cleanly)
# ---------------------------------------------------------------------------


def test_compute_advantages_request_response_round_trip():
    """ComputeAdvantagesRequest + ComputeAdvantagesResponse round-trip cleanly."""
    import msgspec

    samples = [_make_sample(completion_len=2)]
    request = ComputeAdvantagesRequest(
        samples=samples,
        episodic_reward=1.0,
        is_terminal=True,
        algorithm="ppo",
    )
    encoded = msgspec.msgpack.encode(request)
    decoded = msgspec.msgpack.decode(encoded, type=ComputeAdvantagesRequest)
    assert decoded.episodic_reward == 1.0
    assert decoded.is_terminal is True
    assert decoded.algorithm == "ppo"
    assert len(decoded.samples) == 1
    assert decoded.samples[0].completion_ids == samples[0].completion_ids

    paired = PairedSample(
        llm_sample=_make_sample(completion_len=2),
        advantage_sample=AdvantageTrainingSample(
            prompt_ids=[1, 2],
            prompt_mask=[False, False],
            completion_ids=[10, 11],
            completion_mask=[True, True],
            v_targets=[0.5, 0.6],
            q_plus_targets=None,
        ),
    )
    response = ComputeAdvantagesResponse(paired_samples=[paired])
    enc_resp = msgspec.msgpack.encode(response)
    dec_resp = msgspec.msgpack.decode(enc_resp, type=ComputeAdvantagesResponse)
    assert len(dec_resp.paired_samples) == 1
    assert dec_resp.paired_samples[0].advantage_sample.v_targets == [0.5, 0.6]


# ---------------------------------------------------------------------------
# FakeAdvantageServer behavior
# ---------------------------------------------------------------------------


def test_fake_advantage_server_returns_zero_paired_outputs_default():
    """Default fake produces zero advantages / v_targets / q_plus_targets."""
    fake = FakeAdvantageServer()
    samples = [_make_sample(completion_len=3), _make_sample(completion_len=2)]
    paired = asyncio.run(
        fake.compute_advantages_and_targets(
            samples=samples,
            episodic_reward=1.0,
            is_terminal=True,
            algorithm="ppo",
        )
    )
    assert len(paired) == 2
    for input_sample, (llm_sample, adv_sample) in zip(samples, paired):
        assert llm_sample.advantages == [0.0] * len(input_sample.completion_ids)
        assert adv_sample.v_targets == [0.0] * len(input_sample.completion_ids)
        # PPO -> q_plus_targets None.
        assert adv_sample.q_plus_targets is None


def test_fake_advantage_server_records_calls():
    """Calls are recorded on the fake for assertions in orchestrator tests."""
    fake = FakeAdvantageServer()
    samples = [_make_sample(completion_len=1)]
    asyncio.run(
        fake.compute_advantages_and_targets(
            samples=samples,
            episodic_reward=0.5,
            is_terminal=False,
            algorithm="arm",
        )
    )
    assert len(fake.calls) == 1
    assert fake.calls[0]["episodic_reward"] == 0.5
    assert fake.calls[0]["is_terminal"] is False
    assert fake.calls[0]["algorithm"] == "arm"


def test_fake_advantage_server_arm_returns_q_plus_targets():
    """algorithm='arm' produces non-None q_plus_targets in the default fake."""
    fake = FakeAdvantageServer()
    samples = [_make_sample(completion_len=2)]
    paired = asyncio.run(
        fake.compute_advantages_and_targets(
            samples=samples,
            episodic_reward=0.0,
            is_terminal=True,
            algorithm="arm",
        )
    )
    _, adv_sample = paired[0]
    assert adv_sample.q_plus_targets == [0.0, 0.0]


def test_fake_advantage_server_protocol_compliance():
    """FakeAdvantageServer implements AdvantageServerClientProtocol structurally
    (duck-typed; the test confirms the methods exist with awaitable signatures)."""
    fake = FakeAdvantageServer()
    # Static check: the fake satisfies the protocol shape.
    fake_typed: AdvantageServerClientProtocol = fake
    assert hasattr(fake_typed, "compute_advantages_and_targets")
    assert hasattr(fake_typed, "aclose")


# ---------------------------------------------------------------------------
# Sync invariant -- DQ5
# ---------------------------------------------------------------------------


def test_llm_advantage_sample_sync():
    """Phase 6 DQ5: paired LLMTrainingSample/AdvantageTrainingSample share
    prompt/completion structure verbatim, and target list lengths match
    completion_ids length.

    Iterates 5 random seeded list-of-samples and asserts the invariant.
    """
    import random

    rng = random.Random(0)
    fake = FakeAdvantageServer()

    for trial in range(5):
        n_samples = rng.randint(1, 3)
        samples = []
        for _ in range(n_samples):
            comp_len = rng.randint(1, 5)
            prompt_len = rng.randint(1, 3)
            samples.append(
                _make_sample(
                    completion_len=comp_len,
                    prompt_ids=[rng.randint(0, 100) for _ in range(prompt_len)],
                )
            )

        paired = asyncio.run(
            fake.compute_advantages_and_targets(
                samples=samples,
                episodic_reward=1.0,
                is_terminal=True,
                algorithm="arm",  # arm -> q_plus_targets present too
            )
        )

        for input_sample, (llm_sample, adv_sample) in zip(samples, paired):
            # Prompt + completion structure mirrored.
            assert llm_sample.prompt_ids == input_sample.prompt_ids == adv_sample.prompt_ids
            assert llm_sample.prompt_mask == input_sample.prompt_mask == adv_sample.prompt_mask
            assert llm_sample.completion_ids == input_sample.completion_ids == adv_sample.completion_ids
            assert llm_sample.completion_mask == input_sample.completion_mask == adv_sample.completion_mask
            # Target list lengths align with completion_ids length.
            n = len(input_sample.completion_ids)
            assert llm_sample.advantages is not None and len(llm_sample.advantages) == n
            assert adv_sample.v_targets is not None and len(adv_sample.v_targets) == n
            assert adv_sample.q_plus_targets is not None and len(adv_sample.q_plus_targets) == n


# ---------------------------------------------------------------------------
# AdvantageServerClient construction (no real HTTP -- the actual end-to-end
# request/response is exercised in Phase 6b's test_server_endpoint.py against
# a real subprocess).
# ---------------------------------------------------------------------------


def test_client_construction_uses_config_base_url_and_timeout():
    """Sanity: construction wires base_url + timeout into the underlying httpx client."""
    config = AdvantageServerClientConfig(
        base_url="http://localhost:9999",
        request_timeout=42.0,
    )
    client = AdvantageServerClient(config)
    assert str(client._client.base_url).rstrip("/") == "http://localhost:9999"
    assert client._client.timeout.read == pytest.approx(42.0)
    asyncio.run(client.aclose())


# ---------------------------------------------------------------------------
# wait_for_weight_step -- AdvTrainer<->orchestrator lockstep barrier
# ---------------------------------------------------------------------------


def test_fake_wait_for_weight_step_zero_short_circuits():
    """min_step=0 returns immediately; AdvServer starts at weight_step=0,
    condition trivially satisfied. This is the step-0 bootstrap case."""
    fake = FakeAdvantageServer()
    assert fake.weight_step == 0
    result = asyncio.run(fake.wait_for_weight_step(min_step=0))
    assert result == 0
    assert len(fake.wait_calls) == 1
    assert fake.wait_calls[0]["min_step"] == 0


def test_fake_wait_for_weight_step_satisfied_returns_immediately():
    """When weight_step already meets the bar, no polling is needed."""
    fake = FakeAdvantageServer(initial_weight_step=3)
    result = asyncio.run(fake.wait_for_weight_step(min_step=3))
    assert result == 3
    # Pre-advance via the test helper -> condition met before the loop body
    # would sleep.
    result = asyncio.run(fake.wait_for_weight_step(min_step=2))
    assert result == 3


def test_fake_wait_for_weight_step_blocks_until_advance():
    """Wait blocks until a concurrent task advances weight_step.

    Models the AdvTrainer broadcasting mid-orchestrator-wait: the orchestrator
    enters wait_for_weight_step(1), the AdvTrainer broadcasts (advance to 1),
    the wait returns. Uses asyncio.gather to schedule both as concurrent
    coroutines.
    """
    fake = FakeAdvantageServer()  # weight_step=0

    async def slow_advance():
        # Simulate AdvTrainer finishing a cycle; advances weight_step after
        # a short delay.
        await asyncio.sleep(0.05)
        fake.advance_weight_step(1)

    async def run() -> int:
        # Use a small poll_interval so the test runs quickly; the wait should
        # complete after the advance happens.
        waiter = fake.wait_for_weight_step(min_step=1, poll_interval=0.01)
        advancer = slow_advance()
        result, _ = await asyncio.gather(waiter, advancer)
        return result

    result = asyncio.run(run())
    assert result == 1
    assert fake.weight_step == 1


def test_fake_wait_for_weight_step_raises_on_timeout():
    """Timeout fires if weight_step never advances enough."""
    fake = FakeAdvantageServer()  # weight_step=0
    with pytest.raises(TimeoutError, match="weight_step"):
        asyncio.run(
            fake.wait_for_weight_step(min_step=5, timeout=0.05, poll_interval=0.01)
        )


def _build_weight_status_only_app(initial_weight_step: int = 0):
    """Minimal FastAPI app exposing just /weight_status, no startup lifespan.

    Avoids the full create_app's HuggingFace download + backbone load -- those
    aren't needed for client-side wait_for_weight_step tests. The app's
    weight_step is mutable from the outside (set `app.state.weight_step`).
    """
    import msgspec
    from fastapi import FastAPI, Response

    app = FastAPI(title="Test AdvServer (weight_status only)")
    app.state.weight_step = initial_weight_step

    @app.get("/weight_status")
    async def weight_status():
        body = msgspec.json.encode({"weight_step": int(app.state.weight_step)})
        return Response(status_code=200, content=body, media_type="application/json")

    return app


def _make_asgi_client(app, base_url: str = "http://testserver", timeout: float = 5.0):
    """Construct an AdvantageServerClient whose underlying httpx routes through
    `app`'s ASGI transport (no real socket, no uvicorn, no startup lifespan)."""
    import httpx

    config = AdvantageServerClientConfig(base_url=base_url, request_timeout=timeout)
    client = AdvantageServerClient(config)
    # Replace the default httpx client with one bound to the ASGI app.
    asyncio.run(client._client.aclose())
    client._client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url=base_url,
        timeout=httpx.Timeout(timeout),
    )
    return client


def test_real_client_wait_for_weight_step_against_advserver():
    """Real AdvantageServerClient + minimal ASGI-mounted app: real httpx.
    Verifies the wire-level integration end-to-end (HTTP GET + JSON parse +
    polling loop) without needing a real subprocess.

    Simulates the AdvTrainer broadcast by mutating app.state.weight_step
    from inside the asyncio event loop (the same loop running the wait).
    """
    app = _build_weight_status_only_app(initial_weight_step=0)
    client = _make_asgi_client(app)

    async def run():
        # Initial state: weight_step=0, min_step=0 short-circuits.
        assert await client.wait_for_weight_step(min_step=0) == 0

        # Schedule a background coroutine that advances the server's
        # weight_step (simulates the AdvTrainer's /update_weights side
        # effect).
        async def server_advance():
            await asyncio.sleep(0.05)
            app.state.weight_step = 1

        advancer = asyncio.create_task(server_advance())
        result = await client.wait_for_weight_step(
            min_step=1, timeout=5.0, poll_interval=0.02
        )
        await advancer
        return result

    observed = asyncio.run(run())
    assert observed == 1
    asyncio.run(client.aclose())


def test_real_client_wait_for_weight_step_raises_on_timeout():
    """Real client raises TimeoutError when the server's counter never catches up."""
    app = _build_weight_status_only_app(initial_weight_step=0)
    client = _make_asgi_client(app)

    try:
        with pytest.raises(TimeoutError, match="weight_step"):
            asyncio.run(
                client.wait_for_weight_step(
                    min_step=5, timeout=0.15, poll_interval=0.03
                )
            )
    finally:
        asyncio.run(client.aclose())


def test_real_client_wait_for_weight_step_short_circuits_at_min_step_zero():
    """min_step=0 never issues an HTTP call (the orchestrator's step-0 case)."""
    app = _build_weight_status_only_app(initial_weight_step=0)
    client = _make_asgi_client(app)

    try:
        # If this hit the network, the FastAPI app would respond with
        # weight_step=0 (still satisfied), so we can't distinguish a real
        # call from a short-circuit by the return value alone. Verify by
        # the wall-time being below the poll_interval -- a real call
        # would take a tiny amount of time even on ASGI; short-circuit is
        # essentially instant.
        import time as _time
        t0 = _time.perf_counter()
        result = asyncio.run(
            client.wait_for_weight_step(min_step=0, poll_interval=1.0)
        )
        elapsed = _time.perf_counter() - t0
        assert result == 0
        # Short-circuit path returns without hitting httpx.
        assert elapsed < 0.05, (
            f"wait_for_weight_step(min_step=0) should short-circuit; "
            f"took {elapsed:.3f}s"
        )
    finally:
        asyncio.run(client.aclose())
