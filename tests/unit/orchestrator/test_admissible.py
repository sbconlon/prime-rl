"""Phase 1 (action-level ARM) -- substitute_executed_into_admissible union helper.

The action-level analog of substitute_sampled_into_top_k: guarantees the
executed action a* is present in the admissible set so π_RM / Q+ / π̂ are all
total over the set, and returns its index.
"""

from __future__ import annotations

from prime_rl.orchestrator.admissible import substitute_executed_into_admissible


def test_executed_already_present():
    out, idx = substitute_executed_into_admissible(["a", "b", "c"], "b")
    assert out == ["a", "b", "c"]
    assert idx == 1


def test_executed_absent_appended():
    out, idx = substitute_executed_into_admissible(["a", "b"], "c")
    assert out == ["a", "b", "c"]
    assert idx == 2


def test_executed_absent_empty_set():
    """Defensive: never an empty action set downstream -- the executed action
    becomes the sole admissible action."""
    out, idx = substitute_executed_into_admissible([], "look")
    assert out == ["look"]
    assert idx == 0


def test_executed_already_present_at_front():
    out, idx = substitute_executed_into_admissible(["look", "go north"], "look")
    assert out == ["look", "go north"]
    assert idx == 0


def test_duplicate_admissible_entries():
    """If the env ever yields duplicate commands, the index points at the first
    match (documented behavior)."""
    out, idx = substitute_executed_into_admissible(["look", "go", "look"], "look")
    assert out == ["look", "go", "look"]
    assert idx == 0


def test_does_not_mutate_input():
    original = ["a", "b"]
    snapshot = list(original)
    out, _ = substitute_executed_into_admissible(original, "c")
    assert original == snapshot
    assert out is not original


def test_multi_token_action_strings():
    """Actions are full command strings, including prefixes that share words."""
    admissible = ["go to cabinet 1", "go to cabinet 12", "open cabinet 1"]
    out, idx = substitute_executed_into_admissible(admissible, "go to cabinet 12")
    assert idx == 1
    assert out == admissible
