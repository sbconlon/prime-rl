"""Test fakes colocated with orchestrator-side tests.

Per stubbing-strategy section 4: shared fakes used across multiple test files
live in `_stubs.py` modules colocated with their consumers. Fakes for use
in a single test file stay inline.
"""

from __future__ import annotations

from typing import Any


def make_fake_tokens_dict(
    *,
    completion_ids: list[int],
    prompt_ids: list[int] | None = None,
    completion_mask: list[bool] | None = None,
    completion_logprobs: list[float] | None = None,
    completion_top_k_token_ids: list[list[int]] | None = None,
    routed_experts: list[list[list[int]]] | None = None,
) -> dict[str, Any]:
    """Build a `tokens` dict shaped like what verifiers' parse_response_tokens
    produces, suitable for calls to make_sample / extend_sample in tests.

    `prompt_ids` defaults to a single-token bridge `[0]`. `completion_mask`
    defaults to all-True (every completion token is an assistant generation).
    `completion_logprobs` defaults to all-zero. `completion_top_k_token_ids`
    is None unless explicitly provided.
    """
    if prompt_ids is None:
        prompt_ids = [0]
    if completion_mask is None:
        completion_mask = [True] * len(completion_ids)
    if completion_logprobs is None:
        completion_logprobs = [0.0] * len(completion_ids)

    tokens: dict[str, Any] = {
        "prompt_ids": prompt_ids,
        "prompt_mask": [False] * len(prompt_ids),
        "completion_ids": completion_ids,
        "completion_mask": completion_mask,
        "completion_logprobs": completion_logprobs,
        "routed_experts": routed_experts,
    }
    if completion_top_k_token_ids is not None:
        tokens["completion_top_k_token_ids"] = completion_top_k_token_ids
    return tokens


class FakeInferenceServer:
    """Per stubbing-strategy section 4: deterministic stand-in for the vLLM
    inference server in orchestrator-side tests.

    For Phase 5: synthesizes per-completion-token top-K candidates as
    consecutive integers starting from each sampled token's ID
    (e.g. sampled=42, K=4 -> top-K=[42, 43, 44, 45]). Not realistic, but
    testable: orchestrator-side tests assert the field is copied verbatim
    by make_sample / extend_sample, which is what we want to verify.

    Method signatures intentionally mirror the bits of the real inference
    path that orchestrator-side tests need; nothing more.
    """

    def __init__(self, top_k_action_set_size: int = 32):
        self.top_k_action_set_size = top_k_action_set_size

    def synthesize_top_k_for_completion(
        self, completion_ids: list[int]
    ) -> list[list[int]]:
        """Phase 5 fake top-K: K consecutive integers starting from each
        sampled token's ID. Always satisfies the invariant
        `sampled_id in top_k_token_ids[i]`.
        """
        K = self.top_k_action_set_size
        return [[sampled + j for j in range(K)] for sampled in completion_ids]

    def make_tokens_dict(
        self,
        *,
        completion_ids: list[int],
        prompt_ids: list[int] | None = None,
        completion_mask: list[bool] | None = None,
        with_top_k: bool = True,
    ) -> dict[str, Any]:
        """Convenience: build a tokens dict with synthetic top-K populated
        (or omitted, when with_top_k is False).
        """
        completion_top_k_token_ids = (
            self.synthesize_top_k_for_completion(completion_ids)
            if with_top_k
            else None
        )
        return make_fake_tokens_dict(
            completion_ids=completion_ids,
            prompt_ids=prompt_ids,
            completion_mask=completion_mask,
            completion_top_k_token_ids=completion_top_k_token_ids,
        )
