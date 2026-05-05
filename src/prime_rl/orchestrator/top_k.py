"""Top-K action set helpers for ARM (Phase 5).

Phase 3 / 5 commitment: the K-element action set at every position must
contain the actually-sampled token. vLLM's natural top-K is "highest-
probability candidates," and at temperature > 0 the sampled token can
occasionally fall outside the top-K (the "rare tail-sample case").

`substitute_sampled_into_top_k` enforces the invariant by swapping the
last (lowest-probability) candidate out for the sampled token in those
rare cases. The function is pure -- given (top-K candidates from vLLM,
sampled token), it returns a length-K list that always contains the
sampled token.
"""

from __future__ import annotations


def substitute_sampled_into_top_k(
    top_k_ids: list[int],
    sampled_id: int,
) -> list[int]:
    """Return a length-K list of token IDs that always contains `sampled_id`.

    If `sampled_id` is already in `top_k_ids`, returns `top_k_ids` unchanged.
    Otherwise, replaces the last entry of `top_k_ids` with `sampled_id`.
    The output preserves the input ordering of all surviving candidates.

    Phase doc 14 option C: keeps |S| = K constant (matching Phase 3's
    K=32 commitment cleanly), keeps the wire cost minimal, and avoids the
    variable-bound issue of letting |S| float.

    Args:
        top_k_ids: vLLM's top-K candidates at this position (length K, K >= 1).
        sampled_id: the token actually sampled by the policy.

    Returns:
        A new list of length len(top_k_ids) where sampled_id is guaranteed
        to be present.

    Raises:
        ValueError: if top_k_ids is empty.
    """
    if not top_k_ids:
        raise ValueError("top_k_ids must be non-empty")
    if sampled_id in top_k_ids:
        return list(top_k_ids)
    return list(top_k_ids[:-1]) + [sampled_id]
