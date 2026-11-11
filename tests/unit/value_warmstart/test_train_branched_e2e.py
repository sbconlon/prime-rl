"""Branched dataset Phase 3, Layer 2: one epoch of warmstart-train on a tiny
branched fixture with the tiny CPU backbone writes value_state.pt, holdout.json
and summary.json with the provenance split; the single-action kind still trains.
"""

import json

import torch

from prime_rl.configs.trainer import LoRAConfig
from prime_rl.configs.value_warmstart import WarmStartDataConfig, WarmStartTrainConfig
from prime_rl.orchestrator.value_networks import ValueNetworkConfig
from prime_rl.transport.types import DecisionPoint, TrainingSample
from prime_rl.utils.pathing import get_ckpt_dir, get_step_path
from prime_rl.value_warmstart.dataset_io import BranchRecord, RecordWriter, WarmStartRecord
from prime_rl.value_warmstart.train import train
from tests.unit.advantage_trainer.test_train_step_action import _backbone


class _CharTok:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 256 for c in text]


def _sample(n_dps):
    dps = [
        DecisionPoint(response_start=2 * k, response_end=2 * k + 2, admissible_actions=["go", "look"], executed_action_idx=k % 2)
        for k in range(n_dps)
    ]
    clen = 2 * n_dps
    return TrainingSample(
        prompt_ids=[3, 4, 5], prompt_mask=[False] * 3,
        completion_ids=[(20 + i) % 256 for i in range(clen)], completion_mask=[True] * clen,
        completion_logprobs=[0.0] * clen, completion_temperatures=[1.0] * clen,
        decision_points=dps,
    )


def _records():
    out = []
    for g, reward in (("g1", 1.0), ("g2", 0.0), ("g3", 1.0)):
        out.append(BranchRecord(samples=[_sample(3)], reward=reward, is_truncated=False, game_id=g, trunk_id=f"{g}-t0", holdout=(g == "g3")))
        out.append(BranchRecord(samples=[_sample(3)], reward=1.0 - reward, is_truncated=False, label_from_dp=1, root_dp=1, root_forced=True, game_id=g, trunk_id=f"{g}-t0", branch_id=f"{g}-b1", root_turn=1, block_source="sampled", action_source="own", holdout=(g == "g3")))
    return out


def _config(tmp_path, path, kind, max_epochs=1, resume=True, batch_size=2, ckpt_every_steps=100):
    return WarmStartTrainConfig(
        model=ValueNetworkConfig(base_model_name="tiny", lora=LoRAConfig(rank=8, alpha=16.0, dropout=0.0), polyak_tau=0.005),
        data=WarmStartDataConfig(path=path, kind=kind, batch_size=batch_size, inner_batch_size=1),
        max_epochs=max_epochs,
        resume=resume,
        ckpt_every_steps=ckpt_every_steps,
        output_dir=tmp_path / "out",
    )


def test_train_branched_one_epoch(tmp_path):
    path = tmp_path / "branched.msgpack"
    with RecordWriter(path) as w:
        for r in _records():
            w.write(r)
    config = _config(tmp_path, path, "branched")
    summary = train(config, backbone=_backbone(), tokenizer=_CharTok())

    ckpt = get_step_path(get_ckpt_dir(config.output_dir), 0) / "trainer" / "value_state.pt"
    assert ckpt.exists()
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert "trainable_state_dict" in payload

    holdout = json.loads((config.output_dir / "holdout.json").read_text())
    assert holdout == {"game_ids": ["g3"]}

    written = json.loads((config.output_dir / "summary.json").read_text())
    assert written["dataset_kind"] == "branched"
    assert written["n_samples"] == 4  # g1, g2 trunk + branch; g3 held out
    assert written["n_samples_by_provenance"] == {"trunk": 2, "branch": 2}
    assert written["n_root_targets"] == 2
    final = written["final"]
    assert final["epoch"] == 0 and final["steps"] == 2
    by = final["by_provenance"]
    assert by["trunk"]["n_samples"] == 2 and by["branch"]["n_samples"] == 2
    assert abs(by["trunk"]["l_v"] + by["branch"]["l_v"] - final["l_v"]) < 1e-6
    assert abs(by["trunk"]["l_q"] + by["branch"]["l_q"] - final["l_q"]) < 1e-6
    assert summary["final"] == final


def test_train_warmstart_kind_unchanged(tmp_path):
    path = tmp_path / "ws.msgpack"
    with RecordWriter(path) as w:
        w.write(WarmStartRecord(samples=[_sample(2)], reward=1.0, is_truncated=False))
        w.write(WarmStartRecord(samples=[_sample(2)], reward=0.0, is_truncated=True))
    config = _config(tmp_path, path, "warmstart")
    summary = train(config, backbone=_backbone(), tokenizer=_CharTok())
    assert summary["n_samples_by_provenance"] == {"trunk": 2, "branch": 0}
    assert json.loads((config.output_dir / "holdout.json").read_text()) == {"game_ids": []}
    assert summary["final"]["by_provenance"]["branch"]["n_samples"] == 0


def _trainable(backbone):
    return {k: v.detach().clone() for k, v in backbone.state_dict().items()
            if any(k.startswith(n) for n, p in backbone.named_parameters() if p.requires_grad)}


def test_resume_continues_epochs_and_restores_weights(tmp_path):
    path = tmp_path / "branched.msgpack"
    with RecordWriter(path) as w:
        for r in _records():
            w.write(r)
    tok = _CharTok()

    # job 1: one epoch, killed by the walltime afterwards (simulated by returning)
    bb1 = _backbone()
    s1 = train(_config(tmp_path, path, "branched", max_epochs=3), backbone=bb1, tokenizer=tok)
    assert [e["epoch"] for e in s1["epochs"]] == [0, 1, 2]  # max_epochs honoured when nothing is resumed
    # simulate a kill after epoch 0: rebuild the output dir with one epoch only
    import shutil
    shutil.rmtree(tmp_path / "out")
    bb1 = _backbone()
    s1 = train(_config(tmp_path, path, "branched", max_epochs=1), backbone=bb1, tokenizer=tok)
    resume_ckpt = tmp_path / "out" / "resume" / "checkpoints" / "step_2" / "trainer" / "value_state.pt"
    assert resume_ckpt.exists()
    payload = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
    assert payload["step"] == 2 and "optimizer_state_dict" in payload  # 2 batches per epoch
    w1 = _trainable(bb1)

    # job 2: same command with max_epochs=2 -> loads epoch-1 weights, trains epoch 1 only
    bb2 = _backbone()
    s2 = train(_config(tmp_path, path, "branched", max_epochs=2), backbone=bb2, tokenizer=tok)
    assert [e["epoch"] for e in s2["epochs"]] == [0, 1]
    assert s2["final"]["epoch"] == 1 and s2["epochs"][0] == s1["epochs"][0]
    assert (tmp_path / "out" / "resume" / "checkpoints" / "step_4").exists()
    assert not (tmp_path / "out" / "resume" / "checkpoints" / "step_2").exists()  # keep_last=1
    w2 = _trainable(bb2)
    assert any(not torch.equal(w1[k], w2[k]) for k in w1)  # it trained further
    # the resumed run started from job 1's weights: a fresh backbone loaded from the
    # step-1 checkpoint matches bb1, not bb2's untrained init
    written = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert [e["epoch"] for e in written["epochs"]] == [0, 1]

    # job 3: nothing left to do
    bb3 = _backbone()
    s3 = train(_config(tmp_path, path, "branched", max_epochs=2), backbone=bb3, tokenizer=tok)
    assert [e["epoch"] for e in s3["epochs"]] == [0, 1] and s3["final"]["epoch"] == 1
    assert torch.equal(_trainable(bb3)[next(iter(w2))], w2[next(iter(w2))])  # loaded, not trained

    # resume=False starts over
    bb4 = _backbone()
    s4 = train(_config(tmp_path, path, "branched", max_epochs=1, resume=False), backbone=bb4, tokenizer=tok)
    assert [e["epoch"] for e in s4["epochs"]] == [0]


def _tagged_records():
    """Four training samples whose first prompt id is their dataset index."""
    def sample(tag, n_dps=3):
        s = _sample(n_dps)
        s.prompt_ids = [tag, 4, 5]
        return s

    out = []
    tag = 0
    for g in ("g1", "g2"):
        out.append(BranchRecord(samples=[sample(tag)], reward=1.0, is_truncated=False, game_id=g, trunk_id=f"{g}-t0"))
        tag += 1
        out.append(BranchRecord(samples=[sample(tag)], reward=0.0, is_truncated=False, label_from_dp=1, root_dp=1,
                                root_forced=True, game_id=g, trunk_id=f"{g}-t0", branch_id=f"{g}-b1", root_turn=1,
                                block_source="sampled", action_source="own"))
        tag += 1
    return out


def test_resume_inside_an_epoch_replays_and_skips_no_batch(tmp_path, monkeypatch):
    import pytest

    import prime_rl.value_warmstart.train as tr

    path = tmp_path / "tagged.msgpack"
    with RecordWriter(path) as w:
        for r in _tagged_records():
            w.write(r)
    tok = _CharTok()
    real = tr.value_regression_backward
    seen: list[int] = []
    state = {"calls": 0, "die_at": 3}

    def spy(backbone, group, tokenizer, chunk_scale, device):
        state["calls"] += 1
        if state["die_at"] is not None and state["calls"] == state["die_at"]:
            raise KeyboardInterrupt  # the walltime kill, before this batch is trained
        seen.append(group[0].prompt_ids[0])
        return real(backbone, group, tokenizer, chunk_scale, device)

    monkeypatch.setattr(tr, "value_regression_backward", spy)
    cfg = lambda: _config(tmp_path, path, "branched", max_epochs=2, batch_size=1, ckpt_every_steps=1)  # noqa: E731

    # job 1: killed while starting the third batch of epoch 0
    with pytest.raises(KeyboardInterrupt):
        train(cfg(), backbone=_backbone(), tokenizer=tok)
    order0 = [b[0] for b in tr._shuffled_batches(4, 1, 0)]
    order1 = [b[0] for b in tr._shuffled_batches(4, 1, 1)]
    assert seen == order0[:2]
    progress = json.loads((tmp_path / "out" / "resume" / "progress.json").read_text())
    assert (progress["global_step"], progress["epoch"], progress["batch_in_epoch"], progress["n_batches"]) == (2, 0, 2, 2)
    assert (tmp_path / "out" / "resume" / "checkpoints" / "step_2").exists()
    # the weights-only file exists mid-epoch (contrast_eval can read it between jobs)
    assert (get_step_path(get_ckpt_dir(tmp_path / "out"), 0) / "trainer" / "value_state.pt").exists()
    assert not (tmp_path / "out" / "summary.json").exists()  # no epoch finished yet

    # job 2: same command, runs to the end
    state["die_at"] = None
    summary = train(cfg(), backbone=_backbone(), tokenizer=tok)
    assert seen == order0 + order1  # every batch exactly once, in the original order
    assert sorted(seen[:4]) == [0, 1, 2, 3] and sorted(seen[4:]) == [0, 1, 2, 3]
    assert [e["epoch"] for e in summary["epochs"]] == [0, 1]
    assert summary["epochs"][0]["steps"] == 4  # the interrupted epoch's sums were restored
    assert (tmp_path / "out" / "resume" / "checkpoints" / "step_8").exists()

    # a different batch shape must not silently resume
    with pytest.raises(ValueError):
        train(_config(tmp_path, path, "branched", max_epochs=3, batch_size=2, ckpt_every_steps=1),
              backbone=_backbone(), tokenizer=tok)
