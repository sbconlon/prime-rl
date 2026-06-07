"""Phase 1 (action-level ARM) -- interleave_rollout decision-point construction.

These complement test_trajectories.py (general interleaving) and mirror its
synthetic vf.RolloutOutput pattern, adding the action-level `extras` keys the
ALFWorld env writes (admissible_actions, executed_action). The owning sample's
`decision_points` carries one DecisionPoint per turn, with response spans in the
sample's completion-id space and the union invariant applied.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import verifiers as vf

from prime_rl.orchestrator.trajectories import interleave_rollout


def _step(prompt_ids, completion_ids, *, extras, completion_logprobs=None):
    n = len(completion_ids)
    return vf.TrajectoryStep(
        prompt=[{"role": "user", "content": "U"}],
        completion=[{"role": "assistant", "content": "A"}],
        response=MagicMock(),
        tokens=vf.TrajectoryStepTokens(
            prompt_ids=list(prompt_ids),
            prompt_mask=[0] * len(prompt_ids),
            completion_ids=list(completion_ids),
            completion_mask=[1] * n,
            completion_logprobs=completion_logprobs or [-0.1] * n,
            overlong_prompt=False,
            is_truncated=False,
        ),
        reward=None,
        advantage=None,
        is_truncated=False,
        trajectory_id="1",
        extras=extras,
    )


def _output(steps, *, error=None):
    return vf.RolloutOutput(
        example_id=0,
        trajectory=steps,
        sampling_args={"temperature": 1.0},
        error=error,
    )


def test_decision_points_single_turn():
    output = _output(
        [
            _step(
                [1, 2],
                [3, 4],
                extras={
                    "admissible_actions": ["go to cabinet 1", "look"],
                    "executed_action": "go to cabinet 1",
                },
            )
        ]
    )
    samples = interleave_rollout(output)
    assert samples is not None and len(samples) == 1
    dps = samples[0].decision_points
    assert dps is not None and len(dps) == 1
    dp = dps[0]
    assert dp.response_start == 0
    assert dp.response_end == 2  # whole completion span [3, 4]
    assert dp.admissible_actions == ["go to cabinet 1", "look"]
    assert dp.executed_action_idx == 0
    assert dp.pi_hat is None


def test_decision_points_inadmissible_action_unioned():
    """An executed action absent from the admissible set is appended (union)."""
    output = _output(
        [
            _step(
                [1, 2],
                [3, 4],
                extras={
                    "admissible_actions": ["look", "go north"],
                    "executed_action": "take apple",
                },
            )
        ]
    )
    dp = interleave_rollout(output)[0].decision_points[0]
    assert dp.admissible_actions == ["look", "go north", "take apple"]
    assert dp.executed_action_idx == 2


def test_decision_points_merged_turns():
    """Three turns merged into one sample: each DP's response span lands on that
    turn's mask=True completion, accounting for the mask=False bridge tokens."""
    output = _output(
        [
            _step([1, 2], [3, 4], extras={"admissible_actions": ["a0", "x"], "executed_action": "a0"}),
            _step([1, 2, 3, 4, 5, 6], [7, 8], extras={"admissible_actions": ["a1", "y"], "executed_action": "a1"}),
            _step(
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                [11, 12],
                extras={"admissible_actions": ["a2", "z"], "executed_action": "a2"},
            ),
        ]
    )
    samples = interleave_rollout(output)
    assert len(samples) == 1
    sample = samples[0]
    # completion_ids: [3,4] + bridge[5,6] + [7,8] + bridge[9,10] + [11,12]
    assert sample.completion_ids == [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    dps = sample.decision_points
    assert dps is not None and len(dps) == 3
    # turn 0 -> [0, 2); turn 1 -> [4, 6); turn 2 -> [8, 10)
    assert (dps[0].response_start, dps[0].response_end) == (0, 2)
    assert (dps[1].response_start, dps[1].response_end) == (4, 6)
    assert (dps[2].response_start, dps[2].response_end) == (8, 10)
    assert sample.completion_ids[dps[1].response_start : dps[1].response_end] == [7, 8]
    assert sample.completion_ids[dps[2].response_start : dps[2].response_end] == [11, 12]
    assert [dp.admissible_actions[dp.executed_action_idx] for dp in dps] == ["a0", "a1", "a2"]


def test_decision_points_o_prefix_property():
    """For each merged-turn DP, prompt_ids + completion_ids[:response_start] equals
    the actual context that turn's prompt was built from (the observation o)."""
    step1_prompt = [1, 2, 3, 4, 5, 6]
    step2_prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    output = _output(
        [
            _step([1, 2], [3, 4], extras={"admissible_actions": ["a0"], "executed_action": "a0"}),
            _step(step1_prompt, [7, 8], extras={"admissible_actions": ["a1"], "executed_action": "a1"}),
            _step(step2_prompt, [11, 12], extras={"admissible_actions": ["a2"], "executed_action": "a2"}),
        ]
    )
    sample = interleave_rollout(output)[0]
    expected_prefixes = [[1, 2], step1_prompt, step2_prompt]
    for dp, expected_o in zip(sample.decision_points, expected_prefixes):
        o = sample.prompt_ids + sample.completion_ids[: dp.response_start]
        assert o == expected_o


def test_decision_points_multi_sample():
    """When the extension property breaks, decision points partition across the
    resulting samples, each indexed relative to its own completion_ids."""
    output = _output(
        [
            _step([1, 2], [3, 4], extras={"admissible_actions": ["a0"], "executed_action": "a0"}),
            # different prefix -> new sample
            _step([10, 20, 30], [5, 6], extras={"admissible_actions": ["a1"], "executed_action": "a1"}),
        ]
    )
    samples = interleave_rollout(output)
    assert len(samples) == 2
    assert len(samples[0].decision_points) == 1
    assert (samples[0].decision_points[0].response_start, samples[0].decision_points[0].response_end) == (0, 2)
    assert samples[0].decision_points[0].admissible_actions[0] == "a0"
    assert len(samples[1].decision_points) == 1
    assert (samples[1].decision_points[0].response_start, samples[1].decision_points[0].response_end) == (0, 2)
    assert samples[1].decision_points[0].admissible_actions[0] == "a1"


def test_decision_points_none_without_extras():
    """The GRPO/PPO path: steps with empty extras -> decision_points is None and
    the rest of the sample is unaffected."""
    output = _output(
        [
            _step([1, 2], [3, 4], extras={}),
            _step([1, 2, 3, 4, 5, 6], [7, 8], extras={}),
        ]
    )
    samples = interleave_rollout(output)
    assert len(samples) == 1
    assert samples[0].decision_points is None
    assert samples[0].completion_ids == [3, 4, 5, 6, 7, 8]


def test_decision_points_error_rollout():
    """Error rollout: completion_mask is all-False (inert, no gradient) but the
    decision points are still built (the turns happened)."""
    output = _output(
        [
            _step(
                [1, 2],
                [3, 4],
                extras={"admissible_actions": ["a0", "x"], "executed_action": "a0"},
            )
        ],
        error="boom",
    )
    samples = interleave_rollout(output)
    assert len(samples) == 1
    assert samples[0].completion_mask == [False, False]
    dps = samples[0].decision_points
    assert dps is not None and len(dps) == 1
    assert (dps[0].response_start, dps[0].response_end) == (0, 2)
