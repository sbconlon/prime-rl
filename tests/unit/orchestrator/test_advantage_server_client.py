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
