"""Admissible-action union invariant for action-level ARM (Phase 1).

The action-level analog of ``substitute_sampled_into_top_k`` (top_k.py): the
ALFWorld env provides the admissible action set the model saw at choice time
(text), and the executed action a* (text, parsed from the ``<action>`` tag).
The model can emit an action that is not in the admissible set (the env's
``admissible_action_rate`` is < 1), so we guarantee a* is present before any
downstream math runs over the set. With a* guaranteed in the set, π_RM(a*|o),
Q+(o,a*), and π̂(a*|o) are all defined with no special-case branch, and the
value network learns inadmissible actions are low-value -- a format/validity
failure becomes ordinary negative advantage signal rather than an exception.

This lives on the prime-rl side (not the verifiers env) so the union logic is
not duplicated across repos: the env writes the *raw* set + executed-action
text into the step extras, and interleave_rollout applies this helper while
building TrainingSample.decision_points -- exactly mirroring how
substitute_sampled_into_top_k is applied in interleave_rollout for tokens.
"""

from __future__ import annotations


def substitute_executed_into_admissible(
    admissible: list[str],
    executed: str,
) -> tuple[list[str], int]:
    """Guarantee the executed action is in the admissible set; return (set, index).

    If ``executed`` is already present, returns ``admissible`` (copied) unchanged
    and the index of its first occurrence. Otherwise appends ``executed`` and
    returns index == len(admissible) (the pre-append length).

    The returned list is always a fresh copy -- the caller's input is never
    mutated. Duplicate entries in ``admissible`` are preserved; the returned
    index points at the first match (documented behavior).
    """
    out = list(admissible)
    try:
        idx = out.index(executed)
    except ValueError:
        idx = len(out)
        out.append(executed)
    return out, idx
