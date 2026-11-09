"""Branched dataset Phase 0: BranchRecord IO (branch-schema.md §3).

Round trip of every field; a BranchRecord file still reads as WarmStartRecords
(trailing fields ignored); a legacy WarmStartRecord file reads as trunk-like
BranchRecords; append mode keeps earlier frames.
"""

from prime_rl.transport.types import DecisionPoint, TrainingSample
from prime_rl.value_warmstart.dataset_io import (
    BranchRecord,
    RecordWriter,
    WarmStartRecord,
    read_branch_records,
    read_records,
)


def _sample():
    dp = DecisionPoint(response_start=0, response_end=2, admissible_actions=["a", "b"], executed_action_idx=0)
    return TrainingSample(
        prompt_ids=[1, 2],
        prompt_mask=[False, False],
        completion_ids=[10, 11],
        completion_mask=[True, True],
        completion_logprobs=[0.0, 0.0],
        completion_temperatures=[1.0, 1.0],
        decision_points=[dp],
    )


def _branch(**kw):
    base = dict(
        samples=[_sample()],
        reward=1.0,
        is_truncated=False,
        label_from_dp=3,
        root_dp=3,
        root_forced=True,
        game_id="g1",
        trunk_id="g1-t0",
        branch_id="g1-t0-b2",
        root_turn=3,
        action_source="tail",
        block_source="elicited",
        p_accept=0.71,
        allocation_stage=2,
        cont_len=5,
        holdout=True,
    )
    base.update(kw)
    return BranchRecord(**base)


def test_branch_record_roundtrip(tmp_path):
    path = tmp_path / "b.msgpack"
    recs = [_branch(), _branch(branch_id=None, root_dp=None, root_forced=False, label_from_dp=0)]
    with RecordWriter(path) as w:
        for r in recs:
            w.write(r)
    back = list(read_branch_records(path))
    assert back == recs


def test_branch_file_reads_as_warmstart_records(tmp_path):
    path = tmp_path / "b.msgpack"
    with RecordWriter(path) as w:
        w.write(_branch())
    (ws,) = list(read_records(path))
    assert isinstance(ws, WarmStartRecord)
    assert ws.reward == 1.0
    assert ws.is_truncated is False
    assert ws.samples[0].decision_points[0].admissible_actions == ["a", "b"]


def test_legacy_file_reads_as_trunk_branch_records(tmp_path):
    path = tmp_path / "w.msgpack"
    with RecordWriter(path) as w:
        w.write(WarmStartRecord(samples=[_sample()], reward=0.0, is_truncated=True))
    (br,) = list(read_branch_records(path))
    assert br.label_from_dp == 0
    assert br.root_dp is None
    assert br.root_forced is False
    assert br.is_truncated is True


def test_append_mode_keeps_frames(tmp_path):
    path = tmp_path / "b.msgpack"
    with RecordWriter(path) as w:
        w.write(_branch(branch_id="first"))
    with RecordWriter(path, append=True) as w:
        w.write(_branch(branch_id="second"))
    ids = [r.branch_id for r in read_branch_records(path)]
    assert ids == ["first", "second"]
