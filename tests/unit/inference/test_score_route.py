"""Phase 9 perf -- /v1/score route: the span-sum helper + request model.

The custom /v1/score route teacher-force-scores token-id prompts and returns ONLY
each prompt's summed action-span logprob (one float/prompt), avoiding the
full-prompt prompt_logprobs payload that was the cache-invariant throughput wall.
The engine-driving create_score is validated live on the cluster; here we pin the
pure span summation and the request model.
"""

from __future__ import annotations

import pytest

from prime_rl.inference.vllm.serving_chat_with_tokens import (
    ScoreRequest,
    _sum_span_prompt_logprobs,
)


class _LP:
    """Stand-in for vLLM's Logprob (engine-level prompt_logprobs values)."""

    def __init__(self, logprob: float):
        self.logprob = logprob


def test_sum_span_by_token_id():
    # prompt token-ids; engine-level prompt_logprobs are dicts keyed by INT token-id.
    token_ids = [10, 20, 30, 40]
    prompt_logprobs = [
        None,  # position 0 has no logprob
        {20: _LP(-0.5), 999: _LP(-0.01)},  # actual token 20 (not the top one)
        {30: _LP(-0.25)},
        {40: _LP(-5.0)},  # outside the span -> ignored
    ]
    total = _sum_span_prompt_logprobs(prompt_logprobs, token_ids, start=1, end=3)
    assert total == pytest.approx(-0.75)


def test_sum_span_skips_none_and_missing():
    """Cached/missing positions (None entry, or token absent) contribute 0."""
    token_ids = [1, 2, 3]
    prompt_logprobs = [None, None, {3: _LP(-1.0)}]  # pos 1 cached -> None
    # span [1,3): pos1 None -> 0, pos2 -> -1.0
    assert _sum_span_prompt_logprobs(prompt_logprobs, token_ids, 1, 3) == pytest.approx(-1.0)


def test_sum_span_clamps_end_to_length():
    token_ids = [1, 2]
    prompt_logprobs = [None, {2: _LP(-0.3)}]
    # end past the list length -> clamped, no IndexError
    assert _sum_span_prompt_logprobs(prompt_logprobs, token_ids, 1, 99) == pytest.approx(-0.3)


def test_sum_span_empty_prompt_logprobs():
    assert _sum_span_prompt_logprobs(None, [1, 2], 0, 2) == 0.0
    assert _sum_span_prompt_logprobs([], [1, 2], 0, 2) == 0.0


def test_score_request_model():
    req = ScoreRequest(
        model="qwen",
        prompts=[[1, 2, 3], [1, 2, 4]],
        spans=[[2, 3], [2, 3]],
        temperature=1.0,
    )
    assert req.model == "qwen"
    assert req.prompts == [[1, 2, 3], [1, 2, 4]]
    assert req.spans == [[2, 3], [2, 3]]
    assert req.top_p == 1.0  # default
