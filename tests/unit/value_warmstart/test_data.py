from prime_rl.transport.types import DecisionPoint, TrainingSample
from prime_rl.value_warmstart.data import WarmStartDataset
from prime_rl.value_warmstart.dataset_io import RecordWriter, WarmStartRecord


def _sample(dp_specs):
    dps = [
        DecisionPoint(
            response_start=rs, response_end=re, admissible_actions=list(adm), executed_action_idx=ei
        )
        for (rs, re, adm, ei) in dp_specs
    ]
    clen = max(re for _, re, _, _ in dp_specs)
    return TrainingSample(
        prompt_ids=[1, 2],
        prompt_mask=[False, False],
        completion_ids=list(range(10, 10 + clen)),
        completion_mask=[True] * clen,
        completion_logprobs=[0.0] * clen,
        completion_temperatures=[1.0] * clen,
        decision_points=dps,
    )


def _write(path, records):
    with RecordWriter(path) as w:
        for r in records:
            w.write(r)


def test_dataset_applies_mc_targets(tmp_path):
    path = tmp_path / "ds.msgpack"
    _write(
        path,
        [
            # success: is_truncated=False -> is_terminal=True -> g_k = reward = 1.0
            WarmStartRecord(
                samples=[_sample([(0, 2, ["a", "b"], 0), (2, 4, ["a", "b"], 1)])],
                reward=1.0,
                is_truncated=False,
            ),
            # failure: reward 0 -> g_k = 0.0
            WarmStartRecord(samples=[_sample([(0, 2, ["a", "b"], 0)])], reward=0.0, is_truncated=True),
        ],
    )
    ds = WarmStartDataset(path)
    assert len(ds) == 2  # 2 records, one sample each
    assert len(ds[0].decision_point_targets) == 2
    assert all(t.v_target == 1.0 and t.q_plus_target == 1.0 for t in ds[0].decision_point_targets)
    assert ds[1].decision_point_targets[0].v_target == 0.0


def test_dataset_executed_action_text(tmp_path):
    path = tmp_path / "ds.msgpack"
    _write(
        path,
        [WarmStartRecord(samples=[_sample([(0, 2, ["go to cabinet 1", "look"], 1)])], reward=1.0, is_truncated=False)],
    )
    ds = WarmStartDataset(path)
    assert ds[0].decision_point_targets[0].executed_action == "look"


def test_dataset_flattens_multi_sample_record(tmp_path):
    path = tmp_path / "ds.msgpack"
    _write(
        path,
        [
            WarmStartRecord(
                samples=[_sample([(0, 2, ["a", "b"], 0)]), _sample([(0, 2, ["a", "b"], 1)])],
                reward=1.0,
                is_truncated=False,
            )
        ],
    )
    ds = WarmStartDataset(path)
    assert len(ds) == 2  # one record, two samples -> two AdvantageTrainingSamples
