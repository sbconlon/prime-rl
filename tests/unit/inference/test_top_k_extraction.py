"""Phase 5b -- verifiers-side top-K extraction helper.

Tests the `_extract_top_k_token_ids_from_logprobs_content` helper from
verifiers' openai_chat_completions_client. Mocks the ChatCompletion shape
to avoid spinning up a real vLLM.

Two access modes are tested:
- Pydantic-object form: logprobs.content is a list of objects with .token,
  .logprob, .top_logprobs (each entry having .token_id added by vLLM's
  processed_logprobs extension).
- Dict form: logprobs.content is a list of dicts.

Helper signature:
    _extract_top_k_token_ids_from_logprobs_content(
        logprobs_content, is_dict
    ) -> list[list[int]] | None
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from verifiers.clients.openai_chat_completions_client import (
    _extract_top_k_token_ids_from_logprobs_content,
)


# ---------------------------------------------------------------------------
# Pydantic-object-form mocks (matches OpenAI Python SDK's response shape
# augmented with vLLM's `.token_id` extension).
# ---------------------------------------------------------------------------


@dataclass
class _FakeTopLogprob:
    token: str
    logprob: float
    token_id: int  # vLLM extension


@dataclass
class _FakeTokenLogprob:
    token: str
    logprob: float
    top_logprobs: list[_FakeTopLogprob]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_extract_returns_per_position_ids_in_order():
    """A 3-position completion with K=4 alternatives at each position
    yields a 3x4 list of token IDs in the alternatives' order.
    """
    content = [
        _FakeTokenLogprob(
            token="a",
            logprob=-0.1,
            top_logprobs=[
                _FakeTopLogprob(token="a", logprob=-0.1, token_id=10),
                _FakeTopLogprob(token="b", logprob=-1.0, token_id=11),
                _FakeTopLogprob(token="c", logprob=-2.0, token_id=12),
                _FakeTopLogprob(token="d", logprob=-3.0, token_id=13),
            ],
        ),
        _FakeTokenLogprob(
            token="x",
            logprob=-0.2,
            top_logprobs=[
                _FakeTopLogprob(token="x", logprob=-0.2, token_id=20),
                _FakeTopLogprob(token="y", logprob=-1.5, token_id=21),
                _FakeTopLogprob(token="z", logprob=-2.5, token_id=22),
                _FakeTopLogprob(token="w", logprob=-3.5, token_id=23),
            ],
        ),
        _FakeTokenLogprob(
            token="p",
            logprob=-0.05,
            top_logprobs=[
                _FakeTopLogprob(token="p", logprob=-0.05, token_id=30),
                _FakeTopLogprob(token="q", logprob=-0.6, token_id=31),
                _FakeTopLogprob(token="r", logprob=-1.7, token_id=32),
                _FakeTopLogprob(token="s", logprob=-3.2, token_id=33),
            ],
        ),
    ]
    out = _extract_top_k_token_ids_from_logprobs_content(content, is_dict=False)
    assert out == [
        [10, 11, 12, 13],
        [20, 21, 22, 23],
        [30, 31, 32, 33],
    ]


def test_extract_returns_none_when_top_logprobs_empty():
    """If alternatives are empty (top-K wasn't requested), return None so the
    upstream path treats top-K as unavailable.
    """
    content = [
        _FakeTokenLogprob(token="a", logprob=-0.1, top_logprobs=[]),
        _FakeTokenLogprob(token="x", logprob=-0.2, top_logprobs=[]),
    ]
    assert _extract_top_k_token_ids_from_logprobs_content(content, is_dict=False) is None


def test_extract_returns_none_when_token_id_missing():
    """If vLLM's response shape lacks `.token_id` on alternatives, return None."""
    @dataclass
    class _NoIdTopLogprob:
        token: str
        logprob: float

    content = [
        _FakeTokenLogprob(
            token="a",
            logprob=-0.1,
            top_logprobs=[_NoIdTopLogprob(token="a", logprob=-0.1)],  # type: ignore[list-item]
        ),
    ]
    assert _extract_top_k_token_ids_from_logprobs_content(content, is_dict=False) is None


def test_extract_returns_none_for_empty_content():
    """An empty logprobs content list returns None (nothing to extract)."""
    assert _extract_top_k_token_ids_from_logprobs_content([], is_dict=False) is None


def test_extract_handles_dict_form():
    """When the response is accessed via raw JSON (dict form), the helper
    reads `top_logprobs` and `token_id` from dict keys instead of attrs.
    """
    content = [
        {
            "token": "a",
            "logprob": -0.1,
            "top_logprobs": [
                {"token": "a", "logprob": -0.1, "token_id": 100},
                {"token": "b", "logprob": -1.0, "token_id": 101},
            ],
        },
        {
            "token": "x",
            "logprob": -0.2,
            "top_logprobs": [
                {"token": "x", "logprob": -0.2, "token_id": 200},
                {"token": "y", "logprob": -1.5, "token_id": 201},
            ],
        },
    ]
    out = _extract_top_k_token_ids_from_logprobs_content(content, is_dict=True)
    assert out == [[100, 101], [200, 201]]


def test_extract_dict_form_returns_none_when_token_id_missing():
    """Dict-form: missing token_id key bails out same as object-form."""
    content = [
        {
            "token": "a",
            "logprob": -0.1,
            "top_logprobs": [
                {"token": "a", "logprob": -0.1},  # no token_id
            ],
        },
    ]
    assert _extract_top_k_token_ids_from_logprobs_content(content, is_dict=True) is None


# ---------------------------------------------------------------------------
# Substitution applied at orchestrator-side (Phase 5a helper composes with
# Phase 5b extraction). Tests that make_sample-side substitution is the right
# place to enforce sampled-in-K when verifiers' extraction does NOT enforce it.
# ---------------------------------------------------------------------------


def test_extracted_top_k_does_not_substitute_at_verifiers_side():
    """Verifiers' helper returns vLLM's raw top-K -- it does NOT guarantee
    the sampled token is present. That guarantee is Phase 5a's
    `substitute_sampled_into_top_k` (called on the prime-rl orchestrator
    side in make_sample / extend_sample).

    This test pins the contract: when the sampled token is the lowest-prob
    alternative and falls outside vLLM's top-K, verifiers' extraction
    returns the raw K candidates (no substitution).
    """
    content = [
        _FakeTokenLogprob(
            token="b",
            logprob=-5.0,  # sampled at low probability
            top_logprobs=[
                # The actually-sampled token "b" (id=11) is NOT among these top-K.
                _FakeTopLogprob(token="a", logprob=-0.1, token_id=10),
                _FakeTopLogprob(token="c", logprob=-1.0, token_id=12),
                _FakeTopLogprob(token="d", logprob=-2.0, token_id=13),
                _FakeTopLogprob(token="e", logprob=-3.0, token_id=14),
            ],
        ),
    ]
    out = _extract_top_k_token_ids_from_logprobs_content(content, is_dict=False)
    assert out == [[10, 12, 13, 14]]
    # Sampled id 11 is not yet here -- it's prime-rl's substitute_sampled_into_top_k
    # that will add it.
    assert 11 not in out[0]
