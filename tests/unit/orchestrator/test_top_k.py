"""Phase 5 -- top-K helpers + sampling-args toggle + msgspec roundtrip.

Tests organized in four sections:
    1. substitute_sampled_into_top_k pure helper
    2. get_sampling_args toggle behavior
    3. TrainingSample msgspec roundtrip with the new field
    4. (FakeInferenceServer fixture sanity check)

Routing tests for make_sample / extend_sample live in test_top_k_routing.py.
"""

from __future__ import annotations

import msgspec
import pytest

from prime_rl.configs.orchestrator import SamplingConfig
from prime_rl.orchestrator.top_k import substitute_sampled_into_top_k
from prime_rl.orchestrator.utils import get_sampling_args
from prime_rl.transport.types import TrainingSample
from tests.unit.orchestrator._stubs import FakeInferenceServer


# ---------------------------------------------------------------------------
# 1. substitute_sampled_into_top_k pure helper
# ---------------------------------------------------------------------------


def test_substitute_sampled_already_in_top_k_unchanged():
    """When the sampled token is already in the top-K, return it unchanged."""
    out = substitute_sampled_into_top_k([10, 20, 30, 40], sampled_id=20)
    assert out == [10, 20, 30, 40]
    assert len(out) == 4
    assert 20 in out


def test_substitute_sampled_at_front():
    """Sampled token at index 0 -- still unchanged."""
    out = substitute_sampled_into_top_k([99, 1, 2, 3], sampled_id=99)
    assert out == [99, 1, 2, 3]


def test_substitute_sampled_absent_replaces_last_slot():
    """When the sampled token is NOT in top-K, the last slot is overwritten."""
    out = substitute_sampled_into_top_k([10, 20, 30, 40], sampled_id=999)
    assert out == [10, 20, 30, 999]
    assert len(out) == 4
    assert 999 in out
    # The first K-1 entries are preserved.
    assert out[:-1] == [10, 20, 30]


def test_substitute_k_equals_one():
    """Edge case: K=1 means the only slot is the sampled token."""
    out_already_in = substitute_sampled_into_top_k([42], sampled_id=42)
    assert out_already_in == [42]
    out_replaced = substitute_sampled_into_top_k([99], sampled_id=42)
    assert out_replaced == [42]


def test_substitute_idempotent():
    """Calling substitution twice with the same sampled_id is a no-op."""
    once = substitute_sampled_into_top_k([10, 20, 30], sampled_id=99)
    twice = substitute_sampled_into_top_k(once, sampled_id=99)
    assert once == twice


def test_substitute_does_not_mutate_input():
    """The helper returns a new list; the caller's input is unchanged."""
    original = [10, 20, 30, 40]
    snapshot = list(original)
    _ = substitute_sampled_into_top_k(original, sampled_id=999)
    assert original == snapshot


def test_substitute_empty_input_raises():
    """Empty top-K is an invalid input."""
    with pytest.raises(ValueError):
        substitute_sampled_into_top_k([], sampled_id=42)


# ---------------------------------------------------------------------------
# 2. get_sampling_args toggle behavior
# ---------------------------------------------------------------------------


def test_get_sampling_args_default_omits_top_logprobs():
    """Default SamplingConfig: extra_body must NOT carry top_logprobs.

    Top-level `logprobs: True` (chosen-token logprob) stays as is. The
    Phase 5 toggle adds `extra_body["top_logprobs"] = K` only when on.
    """
    config = SamplingConfig(max_tokens=128)
    args = get_sampling_args(config, temperature=1.0, is_vllm=True)
    assert args["logprobs"] is True  # OpenAI-bool form for chosen-token logprob (existing behavior)
    extra = args.get("extra_body", {})
    assert "logprobs" not in extra
    assert "top_logprobs" not in extra


def test_get_sampling_args_does_not_leak_internal_fields_to_top_level():
    """Regression for Phase 5 hotfix: prime-rl-internal SamplingConfig fields
    (return_top_k_token_ids, top_k_action_set_size) must NOT appear at the top
    level of sampling_args, since the orchestrator forwards sampling_args as
    kwargs to AsyncCompletions.create(...) which raises TypeError on unknown
    arguments. Same goes for the toggle being on -- still no top-level leak.
    """
    for toggle in (False, True):
        config = SamplingConfig(
            max_tokens=128,
            return_top_k_token_ids=toggle,
            top_k_action_set_size=8,
        )
        args = get_sampling_args(config, temperature=1.0, is_vllm=True)
        assert "return_top_k_token_ids" not in args, (
            f"return_top_k_token_ids leaked into sampling_args (toggle={toggle})"
        )
        assert "top_k_action_set_size" not in args, (
            f"top_k_action_set_size leaked into sampling_args (toggle={toggle})"
        )


def test_get_sampling_args_with_toggle_on_adds_extra_body_top_logprobs_K():
    """Toggle on -> OpenAI-standard `logprobs=True` + `top_logprobs=K`.

    vLLM 0.17 enforces the OpenAI schema strictly (rejects integer
    `logprobs`); the bool/int split is the supported path. vLLM accepts
    `top_logprobs > 20`, so K=32 stays viable.
    """
    config = SamplingConfig(
        max_tokens=128,
        return_top_k_token_ids=True,
        top_k_action_set_size=32,
    )
    args = get_sampling_args(config, temperature=1.0, is_vllm=True)
    assert args["extra_body"]["logprobs"] is True
    assert args["extra_body"]["top_logprobs"] == 32


def test_get_sampling_args_toggle_respects_custom_K():
    """Custom K propagates via top_logprobs."""
    config = SamplingConfig(
        max_tokens=128,
        return_top_k_token_ids=True,
        top_k_action_set_size=8,
    )
    args = get_sampling_args(config, temperature=1.0, is_vllm=True)
    assert args["extra_body"]["logprobs"] is True
    assert args["extra_body"]["top_logprobs"] == 8


# ---------------------------------------------------------------------------
# 3. TrainingSample msgspec roundtrip
# ---------------------------------------------------------------------------


def _make_minimal_sample(**overrides) -> TrainingSample:
    defaults = dict(
        prompt_ids=[1, 2],
        prompt_mask=[False, False],
        completion_ids=[10, 20, 30],
        completion_mask=[True, True, True],
        completion_logprobs=[-0.1, -0.2, -0.3],
        completion_temperatures=[1.0, 1.0, 1.0],
    )
    defaults.update(overrides)
    return TrainingSample(**defaults)


def test_top_k_round_trip_populated():
    """A sample with populated completion_top_k_token_ids round-trips bit-equal."""
    sample = _make_minimal_sample(
        completion_top_k_token_ids=[[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]],
    )
    encoded = msgspec.msgpack.encode(sample)
    decoded = msgspec.msgpack.decode(encoded, type=TrainingSample)
    assert decoded.completion_top_k_token_ids == sample.completion_top_k_token_ids


def test_top_k_round_trip_none():
    """A sample with None top-K (the GRPO path) deserializes as None."""
    sample = _make_minimal_sample()  # default completion_top_k_token_ids = None
    encoded = msgspec.msgpack.encode(sample)
    decoded = msgspec.msgpack.decode(encoded, type=TrainingSample)
    assert decoded.completion_top_k_token_ids is None


def test_top_k_omit_defaults_keeps_grpo_wire_minimal():
    """omit_defaults=True on TrainingSample means None top-K adds zero wire bytes
    relative to a sample that doesn't even define the field. Sanity check that
    the encoded size with None equals the encoded size from before the field
    existed (i.e., no overhead for GRPO)."""
    sample_with_none = _make_minimal_sample()
    # Construct the same struct shape but bypass the new field via raw msgpack:
    # if omit_defaults works, the encoding of a None-top-K sample must not include
    # the field name.
    encoded = msgspec.msgpack.encode(sample_with_none)
    # The field name "completion_top_k_token_ids" should NOT appear in the encoded bytes.
    assert b"completion_top_k_token_ids" not in encoded


# ---------------------------------------------------------------------------
# 4. FakeInferenceServer fixture sanity check
# ---------------------------------------------------------------------------


def test_fake_inference_server_synthesizes_correct_shape():
    """FakeInferenceServer's top-K synthesis: K consecutive integers from sampled."""
    server = FakeInferenceServer(top_k_action_set_size=4)
    completion_ids = [42, 100, 7]
    top_k = server.synthesize_top_k_for_completion(completion_ids)
    assert top_k == [
        [42, 43, 44, 45],
        [100, 101, 102, 103],
        [7, 8, 9, 10],
    ]
    # Invariant: sampled token always present in its row.
    for sampled, row in zip(completion_ids, top_k):
        assert sampled in row


def test_fake_inference_server_make_tokens_dict_with_top_k():
    """The convenience builder produces tokens with top-K populated."""
    server = FakeInferenceServer(top_k_action_set_size=2)
    tokens = server.make_tokens_dict(completion_ids=[5, 6], with_top_k=True)
    assert tokens["completion_top_k_token_ids"] == [[5, 6], [6, 7]]


def test_fake_inference_server_make_tokens_dict_without_top_k():
    """with_top_k=False omits the field (mirrors the GRPO path)."""
    server = FakeInferenceServer(top_k_action_set_size=2)
    tokens = server.make_tokens_dict(completion_ids=[5, 6], with_top_k=False)
    assert "completion_top_k_token_ids" not in tokens
