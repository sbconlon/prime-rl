from prime_rl.transport.types import DecisionPoint, TrainingSample
from prime_rl.value_warmstart.collect import keep_output
from prime_rl.value_warmstart.dataset_io import RecordWriter, WarmStartRecord, read_records


def _sample():
    return TrainingSample(
        prompt_ids=[1, 2],
        prompt_mask=[False, False],
        completion_ids=[10, 11, 12],
        completion_mask=[True, True, True],
        completion_logprobs=[0.0, 0.0, 0.0],
        completion_temperatures=[1.0, 1.0, 1.0],
        decision_points=[
            DecisionPoint(
                response_start=0,
                response_end=3,
                admissible_actions=["go", "look"],
                executed_action_idx=0,
            )
        ],
    )


def test_keep_output_includes_failure():
    assert keep_output({"error": None, "completion": [{}], "reward": 0.0}) is True


def test_keep_output_excludes_error():
    assert keep_output({"error": "boom", "completion": [{}], "reward": 0.0}) is False


def test_keep_output_excludes_empty_completion():
    assert keep_output({"error": None, "completion": None, "reward": 1.0}) is False
    assert keep_output({"error": None, "completion": [], "reward": 1.0}) is False


def test_dataset_roundtrip(tmp_path):
    path = tmp_path / "warmstart.msgpack"
    records = [
        WarmStartRecord(samples=[_sample()], reward=1.0, is_truncated=False),
        WarmStartRecord(samples=[_sample(), _sample()], reward=0.0, is_truncated=True),
    ]
    with RecordWriter(path) as writer:
        for r in records:
            writer.write(r)

    read_back = list(read_records(path))
    assert len(read_back) == 2
    assert read_back[0].reward == 1.0
    assert read_back[0].is_truncated is False
    assert len(read_back[1].samples) == 2
    assert read_back[1].is_truncated is True
    assert read_back[0].samples[0].decision_points[0].admissible_actions == ["go", "look"]


def test_is_terminal_from_truncation():
    r = WarmStartRecord(samples=[_sample()], reward=1.0, is_truncated=False)
    assert (not r.is_truncated) is True  # is_terminal = not is_truncated
