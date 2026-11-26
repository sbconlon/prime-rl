"""merge_collects: several collect folders into one."""

import json

import pytest

from prime_rl.transport.types import TrainingSample
from prime_rl.value_warmstart.branch_records import CollectStats
from prime_rl.value_warmstart.dataset_io import BranchRecord, RecordWriter, read_branch_records
from prime_rl.value_warmstart.merge_collects import add_stats, merge_collects, read_sources
from prime_rl.value_warmstart.push_eval import rows_by_source


def _sample():
    return TrainingSample(
        prompt_ids=[1],
        prompt_mask=[False],
        completion_ids=[2],
        completion_mask=[True],
        completion_logprobs=[0.0],
        completion_temperatures=[1.0],
    )


def _rec(game, branch=None):
    return BranchRecord(
        samples=[_sample()],
        reward=0.0,
        is_truncated=False,
        game_id=game,
        trunk_id=f"{game}-trunk",
        branch_id=branch,
        root_forced=branch is not None,
    )


def _collect(root, name, games, done, cut_tail=False, stats=None):
    """A collect folder: two records per game, a node file and a done marker for the finished ones."""
    d = root / name
    (d / "done").mkdir(parents=True)
    with RecordWriter(d / "records.msgpack") as w:
        for g in games:
            w.write(_rec(g))
            w.write(_rec(g, branch=f"{g}-trunk-t0-b0"))
    if cut_tail:
        with open(d / "records.msgpack", "ab") as f:
            f.write(b"\x40\x00\x00\x00half a frame")
    for g in games:
        node = d / "nodes" / g / f"{g}-trunk"
        node.mkdir(parents=True)
        (node / "t0.json").write_text(json.dumps({"game": g}))
    for g in done:
        (d / "done" / f"{g}.json").write_text(json.dumps({"game_id": g}))
    if stats is not None:
        stats.save(d / "collect_stats.json")
    return d


def test_merge_takes_finished_games_only(tmp_path):
    a = _collect(
        tmp_path,
        "heldout",
        ["g1", "g2"],
        done=["g1", "g2"],
        stats=CollectStats(games_done=2, branches_run=8, branches_skipped={"leak": 1}),
    )
    b = _collect(
        tmp_path,
        "valid_seen",
        ["g3", "g4"],
        done=["g3"],
        cut_tail=True,
        stats=CollectStats(games_done=1, branches_run=4, branches_skipped={"leak": 2, "weak": 1}),
    )
    out = tmp_path / "gold"
    report = merge_collects([a, b], out)
    assert report["games"] == 3 and report["records"] == 6
    assert report["sources"]["valid_seen"]["records_left_out"] == 2
    back = list(read_branch_records(out / "records.msgpack"))
    assert [r.game_id for r in back] == ["g1", "g1", "g2", "g2", "g3", "g3"]
    assert sorted(p.stem for p in (out / "done").glob("*.json")) == ["g1", "g2", "g3"]
    assert (out / "nodes" / "g3" / "g3-trunk" / "t0.json").exists()
    assert not (out / "nodes" / "g4").exists()
    stats = json.loads((out / "collect_stats.json").read_text())
    assert stats["games_done"] == 3 and stats["branches_run"] == 12
    assert stats["branches_skipped"] == {"leak": 3, "weak": 1}
    assert read_sources(out) == {"g1": "heldout", "g2": "heldout", "g3": "valid_seen"}
    assert read_sources(tmp_path) == {}


def test_merge_refuses_a_game_in_two_inputs_and_an_existing_output(tmp_path):
    a = _collect(tmp_path, "a", ["g1"], done=["g1"])
    b = _collect(tmp_path, "b", ["g1"], done=["g1"])
    with pytest.raises(ValueError, match="g1 is in both 'a' and 'b'"):
        merge_collects([a, b], tmp_path / "gold")
    c = _collect(tmp_path, "c", ["g2"], done=["g2"])
    merge_collects([a, c], tmp_path / "gold2")
    with pytest.raises(FileExistsError):
        merge_collects([a, c], tmp_path / "gold2")
    with pytest.raises(ValueError, match="names must differ"):
        merge_collects([a, c], tmp_path / "gold3", names=["x", "x"])
    with pytest.raises(FileNotFoundError):
        merge_collects([tmp_path / "nothing"], tmp_path / "gold4")


def test_names_and_stats_sum(tmp_path):
    a = _collect(tmp_path, "part1", ["g1"], done=["g1"])
    b = _collect(tmp_path, "part2", ["g2"], done=["g2"])
    report = merge_collects([a, b], tmp_path / "gold", names=["train held-out", "valid unseen"])
    assert set(report["sources"]) == {"train held-out", "valid unseen"}
    total = add_stats([CollectStats(roots=2, admissible_mass_at_roots_sum=1.5), CollectStats(roots=3)])
    assert total.roots == 5 and total.admissible_mass_at_roots_sum == 1.5


def test_rows_split_by_source():
    rows = [{"game_id": "g1"}, {"game_id": "g2"}, {"game_id": "g9"}]
    assert rows_by_source(rows, {}) == {}
    parts = rows_by_source(rows, {"g1": "heldout", "g2": "valid_seen"})
    assert {k: [r["game_id"] for r in v] for k, v in parts.items()} == {
        "heldout": ["g1"],
        "valid_seen": ["g2"],
        "other": ["g9"],
    }
