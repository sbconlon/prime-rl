"""Branched dataset Phase 3: BranchedDataset reads BranchRecords (and legacy
WarmStartRecords), applies label_from_dp / root_dp, tags provenance, excludes
held-out games, and drops samples without targets.
"""

from prime_rl.transport.types import DecisionPoint, TrainingSample
from prime_rl.value_warmstart.data import BranchedDataset, WarmStartDataset
from prime_rl.value_warmstart.dataset_io import BranchRecord, RecordWriter, WarmStartRecord


def _sample(n_dps, prompt=(1, 2)):
    dps = [
        DecisionPoint(response_start=2 * k, response_end=2 * k + 2, admissible_actions=["a", "b"], executed_action_idx=k % 2)
        for k in range(n_dps)
    ]
    clen = 2 * n_dps
    return TrainingSample(
        prompt_ids=list(prompt),
        prompt_mask=[False] * len(prompt),
        completion_ids=list(range(10, 10 + clen)),
        completion_mask=[True] * clen,
        completion_logprobs=[0.0] * clen,
        completion_temperatures=[1.0] * clen,
        decision_points=dps,
    )


def _trunk(game, reward=1.0, n_dps=4, holdout=False):
    return BranchRecord(
        samples=[_sample(n_dps)], reward=reward, is_truncated=False,
        label_from_dp=0, root_dp=None, root_forced=False,
        game_id=game, trunk_id=f"{game}-t0", holdout=holdout,
    )


def _branch(game, t, reward, n_dps=4, split=False, holdout=False, bid="b"):
    samples = [_sample(t), _sample(n_dps - t)] if split else [_sample(n_dps)]
    return BranchRecord(
        samples=samples, reward=reward, is_truncated=False,
        label_from_dp=t, root_dp=t, root_forced=True,
        game_id=game, trunk_id=f"{game}-t0", branch_id=f"{game}-t0-{bid}",
        root_turn=t, action_source="own", block_source="sampled", p_accept=0.8,
        cont_len=n_dps - t - 1, holdout=holdout,
    )


def _write(path, records):
    with RecordWriter(path) as w:
        for r in records:
            w.write(r)


def test_reads_branch_records_with_label_rules(tmp_path):
    path = tmp_path / "b.msgpack"
    _write(path, [_trunk("g1"), _branch("g1", 2, 0.0)])
    ds = BranchedDataset(path)
    assert len(ds) == 2
    assert ds.provenance == ["trunk", "branch"]
    trunk, branch = ds[0], ds[1]
    assert len(trunk.decision_point_targets) == 4
    assert all(t.v_weight == 1.0 for t in trunk.decision_point_targets)
    assert [t.response_start for t in branch.decision_point_targets] == [4, 6]
    assert branch.decision_point_targets[0].v_weight == 0.0
    assert branch.decision_point_targets[1].v_weight == 1.0
    assert all(t.v_target == 0.0 for t in branch.decision_point_targets)
    assert ds.n_root_targets == 1


def test_split_branch_drops_empty_prefix_sample(tmp_path):
    path = tmp_path / "b.msgpack"
    _write(path, [_branch("g1", 2, 1.0, split=True)])
    ds = BranchedDataset(path)
    # the prefix sample (dps 0,1) has no targets and is dropped; the second keeps 2
    assert len(ds) == 1
    assert ds.n_dropped_empty == 1
    t = ds[0].decision_point_targets
    assert [x.response_start for x in t] == [0, 2]
    assert t[0].v_weight == 0.0


def test_reads_legacy_warmstart_file(tmp_path):
    path = tmp_path / "w.msgpack"
    _write(path, [WarmStartRecord(samples=[_sample(3)], reward=1.0, is_truncated=False)])
    ds = BranchedDataset(path)
    assert len(ds) == 1
    assert ds.provenance == ["trunk"]
    assert len(ds[0].decision_point_targets) == 3
    # and WarmStartDataset still reads it the same way
    assert len(WarmStartDataset(path)[0].decision_point_targets) == 3


def test_holdout_excluded_by_flag_and_ids(tmp_path):
    path = tmp_path / "b.msgpack"
    _write(
        path,
        [
            _trunk("g1"), _branch("g1", 1, 1.0),
            _trunk("g2", holdout=True), _branch("g2", 1, 0.0, holdout=True),
            _trunk("g3"), _branch("g3", 1, 0.0),
        ],
    )
    ds = BranchedDataset(path, holdout_ids={"g3"})
    assert len(ds) == 2
    assert ds.holdout_game_ids == {"g2", "g3"}
    assert ds.n_records == 6 and ds.n_holdout_records == 4
    ds_all = BranchedDataset(path, holdout_ids={"g3"}, include_holdout=True)
    assert len(ds_all) == 6
    assert ds_all.holdout_game_ids == {"g2", "g3"}
