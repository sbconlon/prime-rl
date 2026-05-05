"""Phase 5 -- make_sample / extend_sample routing of completion_top_k_token_ids.

Tests cover both Phase 5a (compact layout, bridge-skip, field copy)
and Phase 5b (substitution applied so each row contains the sampled token).
"""

from __future__ import annotations

from prime_rl.orchestrator.top_k import substitute_sampled_into_top_k
from prime_rl.transport.types import TrainingSample
from tests.unit.orchestrator._stubs import (
    FakeInferenceServer,
    make_fake_tokens_dict,
)


def _build_sample_with_top_k(
    completion_ids: list[int],
    completion_mask: list[bool] | None = None,
    completion_top_k_token_ids: list[list[int]] | None = None,
) -> TrainingSample:
    """Mirror what make_sample produces."""
    if completion_mask is None:
        completion_mask = [True] * len(completion_ids)
    return TrainingSample(
        prompt_ids=[0],
        prompt_mask=[False],
        completion_ids=list(completion_ids),
        completion_mask=list(completion_mask),
        completion_logprobs=[0.0] * len(completion_ids),
        completion_temperatures=[1.0] * len(completion_ids),
        completion_top_k_token_ids=(
            [list(row) for row in completion_top_k_token_ids]
            if completion_top_k_token_ids is not None
            else None
        ),
    )


def _apply_make_sample_top_k_logic(
    raw_top_k: list[list[int]] | None, completion_ids: list[int]
) -> list[list[int]] | None:
    """Mirror the Phase 5b substitution-in-make_sample logic."""
    if raw_top_k is None:
        return None
    return [
        substitute_sampled_into_top_k(list(row), int(sampled_id))
        for row, sampled_id in zip(raw_top_k, completion_ids)
    ]


# ---------------------------------------------------------------------------
# make_sample-side: top-K present / absent / substitution applied
# ---------------------------------------------------------------------------


def test_top_k_present_in_tokens_lands_in_sample():
    """Single turn, all mask=True: top-K outer length equals completion_ids length."""
    server = FakeInferenceServer(top_k_action_set_size=4)
    completion_ids = [42, 100]
    tokens = server.make_tokens_dict(completion_ids=completion_ids, with_top_k=True)

    raw_top_k = tokens["completion_top_k_token_ids"]
    final_top_k = _apply_make_sample_top_k_logic(raw_top_k, tokens["completion_ids"])
    sample = _build_sample_with_top_k(
        completion_ids=tokens["completion_ids"],
        completion_top_k_token_ids=final_top_k,
    )

    # FakeInferenceServer's synthesized top-K already contains the sampled
    # token, so substitution is a no-op.
    assert sample.completion_top_k_token_ids == [
        [42, 43, 44, 45],
        [100, 101, 102, 103],
    ]
    assert len(sample.completion_top_k_token_ids) == len(sample.completion_ids)


def test_top_k_absent_in_tokens_yields_none_field_on_sample():
    """GRPO path: tokens dict has no top-K -> sample.completion_top_k_token_ids is None."""
    tokens = make_fake_tokens_dict(
        completion_ids=[42, 100], completion_top_k_token_ids=None
    )
    assert "completion_top_k_token_ids" not in tokens

    final_top_k = _apply_make_sample_top_k_logic(
        tokens.get("completion_top_k_token_ids"), tokens["completion_ids"]
    )
    sample = _build_sample_with_top_k(
        completion_ids=tokens["completion_ids"],
        completion_top_k_token_ids=final_top_k,
    )
    assert sample.completion_top_k_token_ids is None


def test_make_sample_applies_substitution_when_sampled_outside_top_k():
    """Phase 5b: when the raw top-K from vLLM doesn't contain the sampled token,
    make_sample's substitution logic adds it (replacing the last slot)."""
    completion_ids = [42, 99]
    # Raw top-K: vLLM returned candidates that don't include the sampled tokens
    # at either position (rare but legal under temperature > 0 sampling).
    raw_top_k = [
        [10, 20, 30, 40],   # sampled = 42, not in here
        [50, 60, 70, 80],   # sampled = 99, not in here
    ]
    final_top_k = _apply_make_sample_top_k_logic(raw_top_k, completion_ids)
    assert final_top_k == [
        [10, 20, 30, 42],   # last slot replaced with 42 (the sampled token)
        [50, 60, 70, 99],   # last slot replaced with 99
    ]
    # Invariant: every row contains its sampled token.
    for row, sampled in zip(final_top_k, completion_ids):
        assert sampled in row


def test_make_sample_substitution_is_noop_when_sampled_already_in_top_k():
    """When the sampled token is already in the raw top-K, substitution is a no-op."""
    completion_ids = [42, 99]
    raw_top_k = [
        [42, 10, 20, 30],   # sampled = 42, already first
        [50, 60, 99, 70],   # sampled = 99, already in middle
    ]
    final_top_k = _apply_make_sample_top_k_logic(raw_top_k, completion_ids)
    assert final_top_k == raw_top_k


# ---------------------------------------------------------------------------
# extend_sample-side: multi-turn no-bridge / multi-turn with-bridge
# ---------------------------------------------------------------------------


def test_extend_sample_concatenates_top_k_no_bridge():
    """Multi-turn with no bridge tokens: rows concatenate."""
    server = FakeInferenceServer(top_k_action_set_size=3)
    turn_one = [10, 20]
    turn_two = [30, 40, 50]

    sample = _build_sample_with_top_k(
        completion_ids=turn_one,
        completion_top_k_token_ids=server.synthesize_top_k_for_completion(turn_one),
    )

    new_top_k = server.synthesize_top_k_for_completion(turn_two)
    sample.completion_ids.extend(turn_two)
    sample.completion_mask.extend([True] * len(turn_two))
    sample.completion_logprobs.extend([0.0] * len(turn_two))
    sample.completion_temperatures.extend([1.0] * len(turn_two))
    if new_top_k is not None:
        substituted = [
            substitute_sampled_into_top_k(list(row), int(sampled_id))
            for row, sampled_id in zip(new_top_k, turn_two)
        ]
        if sample.completion_top_k_token_ids is not None:
            sample.completion_top_k_token_ids.extend(substituted)
        else:
            sample.completion_top_k_token_ids = substituted

    assert sample.completion_top_k_token_ids == [
        [10, 11, 12], [20, 21, 22],
        [30, 31, 32], [40, 41, 42], [50, 51, 52],
    ]
    assert len(sample.completion_top_k_token_ids) == sum(sample.completion_mask)


def test_extend_sample_skips_bridge_tokens_under_compact_layout():
    """Multi-turn with bridge tokens: top-K outer length == sum(completion_mask)."""
    server = FakeInferenceServer(top_k_action_set_size=3)
    turn_one_completion = [10, 20]
    bridge_prompt_ids = [97, 98, 99]
    turn_two_completion = [30, 40, 50]

    sample = _build_sample_with_top_k(
        completion_ids=turn_one_completion,
        completion_top_k_token_ids=server.synthesize_top_k_for_completion(turn_one_completion),
    )

    sample.completion_ids.extend(bridge_prompt_ids)
    sample.completion_mask.extend([False] * len(bridge_prompt_ids))
    sample.completion_logprobs.extend([0.0] * len(bridge_prompt_ids))
    sample.completion_temperatures.extend([1.0] * len(bridge_prompt_ids))
    # NO top-K extension for bridge -- compact layout.

    new_top_k = server.synthesize_top_k_for_completion(turn_two_completion)
    sample.completion_ids.extend(turn_two_completion)
    sample.completion_mask.extend([True] * len(turn_two_completion))
    sample.completion_logprobs.extend([0.0] * len(turn_two_completion))
    sample.completion_temperatures.extend([1.0] * len(turn_two_completion))
    if new_top_k is not None:
        substituted = [
            substitute_sampled_into_top_k(list(row), int(sampled_id))
            for row, sampled_id in zip(new_top_k, turn_two_completion)
        ]
        if sample.completion_top_k_token_ids is not None:
            sample.completion_top_k_token_ids.extend(substituted)
        else:
            sample.completion_top_k_token_ids = substituted

    assert len(sample.completion_ids) == 8
    assert sum(sample.completion_mask) == 5
    assert len(sample.completion_top_k_token_ids) == sum(sample.completion_mask)
    assert sample.completion_top_k_token_ids == [
        [10, 11, 12], [20, 21, 22],
        [30, 31, 32], [40, 41, 42], [50, 51, 52],
    ]


def test_extend_sample_substitution_applied_per_appended_turn():
    """Phase 5b: extend_sample's appended rows have substitution applied independently
    of make_sample's already-substituted rows for earlier turns."""
    # Turn 1: top-K already contains its sampled tokens (no substitution needed).
    sample = _build_sample_with_top_k(
        completion_ids=[10, 20],
        completion_top_k_token_ids=[[10, 0, 0, 0], [20, 0, 0, 0]],
    )
    # Turn 2: raw top-K does NOT contain sampled tokens (substitution needed).
    turn_two_completion_ids = [30, 40]
    raw_new_top_k = [
        [1, 2, 3, 4],   # sampled = 30, not present
        [5, 6, 7, 8],   # sampled = 40, not present
    ]

    sample.completion_ids.extend(turn_two_completion_ids)
    sample.completion_mask.extend([True] * len(turn_two_completion_ids))
    substituted_new = [
        substitute_sampled_into_top_k(list(row), int(sampled_id))
        for row, sampled_id in zip(raw_new_top_k, turn_two_completion_ids)
    ]
    sample.completion_top_k_token_ids.extend(substituted_new)

    # Turn 1 unchanged; turn 2 has sampled token in last slot.
    assert sample.completion_top_k_token_ids == [
        [10, 0, 0, 0], [20, 0, 0, 0],
        [1, 2, 3, 30], [5, 6, 7, 40],
    ]
