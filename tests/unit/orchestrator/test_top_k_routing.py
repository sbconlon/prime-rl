"""Phase 5 -- make_sample / extend_sample routing of completion_top_k_token_ids.

These tests exercise the *compact* layout (Option C): top-K rows correspond
1:1 to assistant-sampled positions (mask=True). Bridge tokens (mask=False)
introduced by extend_sample's multi-turn merge get no top-K row.

Tests use FakeInferenceServer to produce deterministic top-K data, then
call the orchestrator's make_sample / extend_sample directly via mirrored
logic and assert the new field flows through correctly.
"""

from __future__ import annotations

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


# ---------------------------------------------------------------------------
# make_sample-side: top-K present / absent
# ---------------------------------------------------------------------------


def test_top_k_present_in_tokens_lands_in_sample():
    """Single turn, all mask=True: top-K outer length equals completion_ids length."""
    server = FakeInferenceServer(top_k_action_set_size=4)
    completion_ids = [42, 100]
    tokens = server.make_tokens_dict(completion_ids=completion_ids, with_top_k=True)

    sample = _build_sample_with_top_k(
        completion_ids=tokens["completion_ids"],
        completion_top_k_token_ids=tokens["completion_top_k_token_ids"],
    )

    assert sample.completion_top_k_token_ids == [
        [42, 43, 44, 45],
        [100, 101, 102, 103],
    ]
    # Single-turn, no bridge: compact and rectangular happen to coincide.
    assert len(sample.completion_top_k_token_ids) == len(sample.completion_ids)
    assert len(sample.completion_top_k_token_ids) == sum(sample.completion_mask)


def test_top_k_absent_in_tokens_yields_none_field_on_sample():
    """GRPO path: tokens dict has no top-K -> sample.completion_top_k_token_ids is None."""
    tokens = make_fake_tokens_dict(
        completion_ids=[42, 100], completion_top_k_token_ids=None
    )
    assert "completion_top_k_token_ids" not in tokens

    sample = _build_sample_with_top_k(
        completion_ids=tokens["completion_ids"],
        completion_top_k_token_ids=tokens.get("completion_top_k_token_ids"),
    )
    assert sample.completion_top_k_token_ids is None


# ---------------------------------------------------------------------------
# extend_sample-side: multi-turn no-bridge / multi-turn with-bridge
# ---------------------------------------------------------------------------


def test_extend_sample_concatenates_top_k_no_bridge():
    """Multi-turn with no bridge tokens (uncommon but minimal): rows concatenate."""
    server = FakeInferenceServer(top_k_action_set_size=3)
    turn_one = [10, 20]
    turn_two = [30, 40, 50]

    # Turn 1
    sample = _build_sample_with_top_k(
        completion_ids=turn_one,
        completion_top_k_token_ids=server.synthesize_top_k_for_completion(turn_one),
    )

    # Turn 2 extension (mask=True only; no bridge)
    new_top_k = server.synthesize_top_k_for_completion(turn_two)
    sample.completion_ids.extend(turn_two)
    sample.completion_mask.extend([True] * len(turn_two))
    sample.completion_logprobs.extend([0.0] * len(turn_two))
    sample.completion_temperatures.extend([1.0] * len(turn_two))
    if sample.completion_top_k_token_ids is not None and new_top_k is not None:
        sample.completion_top_k_token_ids.extend([list(row) for row in new_top_k])

    assert sample.completion_top_k_token_ids == [
        [10, 11, 12], [20, 21, 22],
        [30, 31, 32], [40, 41, 42], [50, 51, 52],
    ]
    # No bridge: compact length == completion_ids length == sum(mask).
    assert len(sample.completion_top_k_token_ids) == 5
    assert sum(sample.completion_mask) == 5
    assert len(sample.completion_ids) == 5


def test_extend_sample_skips_bridge_tokens_under_compact_layout():
    """Multi-turn with bridge tokens: top-K outer length == sum(completion_mask),
    NOT len(completion_ids). Bridge positions get no row.
    """
    server = FakeInferenceServer(top_k_action_set_size=3)
    turn_one_completion = [10, 20]
    bridge_prompt_ids = [97, 98, 99]  # the next user/tool turn, inlined
    turn_two_completion = [30, 40, 50]

    # Turn 1
    sample = _build_sample_with_top_k(
        completion_ids=turn_one_completion,
        completion_top_k_token_ids=server.synthesize_top_k_for_completion(turn_one_completion),
    )

    # Bridge extension: mask=False, no top-K row appended (Option C).
    sample.completion_ids.extend(bridge_prompt_ids)
    sample.completion_mask.extend([False] * len(bridge_prompt_ids))
    sample.completion_logprobs.extend([0.0] * len(bridge_prompt_ids))
    sample.completion_temperatures.extend([1.0] * len(bridge_prompt_ids))
    # NO top-K extension here (Option C: bridge positions get no row).

    # Turn 2 completion extension: mask=True, append top-K rows.
    new_top_k = server.synthesize_top_k_for_completion(turn_two_completion)
    sample.completion_ids.extend(turn_two_completion)
    sample.completion_mask.extend([True] * len(turn_two_completion))
    sample.completion_logprobs.extend([0.0] * len(turn_two_completion))
    sample.completion_temperatures.extend([1.0] * len(turn_two_completion))
    if sample.completion_top_k_token_ids is not None and new_top_k is not None:
        sample.completion_top_k_token_ids.extend([list(row) for row in new_top_k])

    # Compact invariants:
    assert len(sample.completion_ids) == 8        # 2 + 3 bridge + 3
    assert sum(sample.completion_mask) == 5       # 2 + 3 (bridge excluded)
    assert len(sample.completion_top_k_token_ids) == sum(sample.completion_mask)
    # Top-K rows correspond ONLY to the assistant-sampled (mask=True) positions:
    assert sample.completion_top_k_token_ids == [
        [10, 11, 12], [20, 21, 22],          # turn 1 completion
        [30, 31, 32], [40, 41, 42], [50, 51, 52],  # turn 2 completion (bridge skipped)
    ]
