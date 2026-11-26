"""Packed records: the xz copy stands in for the plain file, losslessly."""

import json
import lzma

import pytest

from prime_rl.transport.types import TrainingSample
from prime_rl.value_warmstart import branch_collect as bc
from prime_rl.value_warmstart import pack_records as pr
from prime_rl.value_warmstart.dataset_io import (
    BranchRecord,
    RecordWriter,
    open_records,
    packed_path,
    read_branch_records,
    read_records,
    records_exist,
)
from prime_rl.value_warmstart.merge_collects import merge_collects
from prime_rl.value_warmstart.pack_records import pack_records


def _sample(i):
    return TrainingSample(
        prompt_ids=[1, i],
        prompt_mask=[False, False],
        completion_ids=[2],
        completion_mask=[True],
        completion_logprobs=[0.0],
        completion_temperatures=[1.0],
    )


def _write(path, games=("g1", "g2"), per_game=3):
    path.parent.mkdir(parents=True, exist_ok=True)
    with RecordWriter(path) as w:
        for g in games:
            for i in range(per_game):
                w.write(
                    BranchRecord(
                        samples=[_sample(i)],
                        reward=float(i % 2),
                        is_truncated=False,
                        game_id=g,
                        trunk_id=f"{g}-trunk",
                        branch_id=f"{g}-trunk-t0-b{i}" if i else None,
                        root_forced=bool(i),
                    )
                )
    return path


def test_pack_and_replace_reads_back_the_same(tmp_path):
    path = _write(tmp_path / "records.msgpack")
    raw = path.read_bytes()
    before = list(read_branch_records(path))
    r = pack_records(path, replace=True)
    assert r["status"] == "packed" and r["replaced"] and r["plain_bytes"] == len(raw)
    assert not path.exists() and packed_path(path).exists()
    assert records_exist(path)
    assert lzma.open(packed_path(path)).read() == raw
    assert list(read_branch_records(path)) == before
    assert [x.reward for x in read_records(path)] == [x.reward for x in before]
    # the packed file can be named directly too
    assert list(read_branch_records(packed_path(path))) == before
    digest, name = (tmp_path / "records.msgpack.sha256").read_text().split()
    assert digest == r["sha256"] and name == "records.msgpack"
    assert pack_records(path, replace=True)["status"] == "already packed"


def test_pack_without_replace_keeps_the_plain_file(tmp_path):
    path = _write(tmp_path / "records.msgpack")
    raw = path.read_bytes()
    r = pack_records(path)
    assert r["status"] == "packed" and not r["replaced"]
    assert path.read_bytes() == raw and packed_path(path).exists()
    # readers still take the plain file
    with open_records(path) as f:
        assert f.read() == raw
    # a second call checks the packed copy again, then removes the plain file
    r2 = pack_records(path, replace=True)
    assert r2["status"] == "verified" and r2["replaced"] and r2["sha256"] == r["sha256"]
    assert not path.exists()
    assert lzma.open(packed_path(path)).read() == raw


def test_a_stale_packed_copy_is_never_trusted(tmp_path):
    path = _write(tmp_path / "records.msgpack")
    pack_records(path)
    with RecordWriter(path, append=True) as w:
        w.write(BranchRecord(samples=[_sample(9)], reward=1.0, is_truncated=False, game_id="g3"))
    with pytest.raises(ValueError, match="does not read back identical"):
        pack_records(path, replace=True)
    assert path.exists()


def test_a_file_written_to_while_packing_is_kept(tmp_path, monkeypatch):
    path = _write(tmp_path / "records.msgpack")
    real = pr._pack_to

    def pack_then_append(src, dst, preset):
        out = real(src, dst, preset)
        with open(src, "ab") as f:
            f.write(b"\x01\x00\x00\x00x")
        return out

    monkeypatch.setattr(pr, "_pack_to", pack_then_append)
    with pytest.raises(RuntimeError, match="changed while it was packed"):
        pack_records(path, replace=True)
    assert path.exists() and not packed_path(path).exists()
    assert not (tmp_path / "records.msgpack.xz.tmp").exists()


def test_missing_records(tmp_path):
    with pytest.raises(FileNotFoundError):
        pack_records(tmp_path / "records.msgpack")
    with pytest.raises(FileNotFoundError, match="no packed copy"):
        list(read_branch_records(tmp_path / "records.msgpack"))
    assert not records_exist(tmp_path / "records.msgpack")


def test_a_collect_refuses_a_packed_folder(tmp_path):
    bc.check_not_packed(tmp_path)  # nothing there: fine
    path = _write(tmp_path / "records.msgpack")
    bc.check_not_packed(tmp_path)  # plain records: fine (a resume)
    pack_records(path, replace=True)
    with pytest.raises(RuntimeError, match="records were packed"):
        bc.check_not_packed(tmp_path)


def test_merge_reads_packed_inputs(tmp_path):
    for name, games in (("a", ["g1", "g2"]), ("b", ["g3"])):
        d = tmp_path / name
        _write(d / "records.msgpack", games=games)
        (d / "done").mkdir()
        for g in games:
            (d / "done" / f"{g}.json").write_text(json.dumps({"game_id": g}))
    pack_records(tmp_path / "a" / "records.msgpack", replace=True)
    report = merge_collects([tmp_path / "a", tmp_path / "b"], tmp_path / "out")
    assert report["games"] == 3 and report["records"] == 9
    merged = list(read_branch_records(tmp_path / "out" / "records.msgpack"))
    assert [r.game_id for r in merged] == ["g1"] * 3 + ["g2"] * 3 + ["g3"] * 3
