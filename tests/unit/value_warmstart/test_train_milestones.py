"""Warm-start trainer: milestone checkpoints with in-training contrast scoring,
learning-rate decay, separate optimizer groups for the value heads, the
free-baseline loss, and the gradient-coherence readout."""

import json

import pytest
import torch

from prime_rl.advantage_trainer.shared import MasterWeightAdamW, setup_value_optimizer
from prime_rl.transport.types import DecisionPoint, TrainingSample
from prime_rl.utils.pathing import get_ckpt_dir, get_step_path
from prime_rl.value_warmstart.contrast_eval import Pair, score_pairs
from prime_rl.value_warmstart.data import BranchedDataset
from prime_rl.value_warmstart.dataset_io import BranchRecord, RecordWriter, WarmStartRecord
from prime_rl.value_warmstart.train import constant_baselines, lr_scale, train
from tests.unit.advantage_trainer.test_train_step_action import _backbone
from tests.unit.value_warmstart.test_train_branched_e2e import (
    _CharTok,
    _config,
    _records,
    _sample,
    _tagged_records,
)


def _sample_with_actions(idxs):
    dps = [
        DecisionPoint(response_start=2 * k, response_end=2 * k + 2, admissible_actions=["go", "look"],
                      executed_action_idx=i)
        for k, i in enumerate(idxs)
    ]
    clen = 2 * len(idxs)
    return TrainingSample(
        prompt_ids=[9, 4, 5], prompt_mask=[False] * 3,
        completion_ids=[(20 + i) % 256 for i in range(clen)], completion_mask=[True] * clen,
        completion_logprobs=[0.0] * clen, completion_temperatures=[1.0] * clen,
        decision_points=dps,
    )


def _holdout_contrast_records():
    """A held-out game with one contrast root: at turn 1 the trunk took 'look'
    (return 1) and a branch was forced to 'go' (return 0)."""
    trunk = BranchRecord(samples=[_sample_with_actions([0, 1, 0])], reward=1.0, is_truncated=False,
                         game_id="h1", trunk_id="h1-t0", holdout=True)
    branch = BranchRecord(samples=[_sample_with_actions([0, 0, 1])], reward=0.0, is_truncated=False,
                          label_from_dp=1, root_dp=1, root_forced=True, game_id="h1", trunk_id="h1-t0",
                          branch_id="h1-b1", root_turn=1, block_source="elicited", action_source="tail",
                          p_accept=0.9, holdout=True)
    return [trunk, branch]


def _write(path, records):
    with RecordWriter(path) as w:
        for r in records:
            w.write(r)
    return path


def _weights(path):
    return torch.load(path, map_location="cpu", weights_only=False)["trainable_state_dict"]


# ---------------------------------------------------------------- milestones


def test_milestones_keep_weights_and_score_contrast(tmp_path):
    path = _write(tmp_path / "d.msgpack", _tagged_records() + _holdout_contrast_records())
    bb = _backbone()
    # 4 training samples, batch 1 -> 4 steps per epoch, 8 in total; a checkpoint every step
    cfg = _config(tmp_path, path, "branched", max_epochs=2, batch_size=1, ckpt_every_steps=1,
                  milestone_every_steps=3)
    summary = train(cfg, backbone=bb, tokenizer=_CharTok())
    out = tmp_path / "out"

    kept = sorted(p.name for p in (out / "milestones" / "checkpoints").iterdir())
    assert kept == ["step_3", "step_6", "step_8"]  # multiples of 3, and the final step
    assert sorted(p.name for p in (out / "resume" / "checkpoints").iterdir()) == ["step_8"]  # rolling

    rows = [json.loads(line) for line in (out / "milestones" / "curve.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == [3, 6, 8]
    for r in rows:
        assert r["n_pairs"] == 1 and r["n_roots"] == 1 and r["accuracy"] in (0.0, 1.0)
        assert {"mean_margin", "n_ties", "accuracy_no_ties", "running_l_v", "running_l_q"} <= set(r)
        payload = json.loads((out / "gate" / f"contrast_step{r['step']}.json").read_text())
        assert payload["step"] == r["step"] and len(payload["pairs"]) == 1
        assert (payload["pairs"][0]["a_i"], payload["pairs"][0]["a_j"]) == ("go", "look")
        assert (payload["pairs"][0]["g_i"], payload["pairs"][0]["g_j"]) == (0.0, 1.0)
    assert bb.training  # scoring switches to eval mode and back

    final = _weights(get_step_path(get_ckpt_dir(out), 0) / "trainer" / "value_state.pt")
    m3 = _weights(out / "milestones" / "checkpoints" / "step_3" / "trainer" / "value_state.pt")
    m8 = _weights(out / "milestones" / "checkpoints" / "step_8" / "trainer" / "value_state.pt")
    assert all(torch.equal(m8[k], final[k]) for k in final)  # the last milestone is the final state
    assert any(not torch.equal(m3[k], m8[k]) for k in m8)  # an earlier one is a different state
    assert summary["constant_baseline"]["l_q"]["loss"] > 0.0


def test_milestones_across_a_kill_and_resume(tmp_path, monkeypatch):
    import prime_rl.value_warmstart.train as tr

    path = _write(tmp_path / "d.msgpack", _tagged_records() + _holdout_contrast_records())
    real = tr.value_regression_backward
    state = {"calls": 0, "die_at": 6}

    def spy(backbone, group, tokenizer, chunk_scale, device):
        state["calls"] += 1
        if state["die_at"] is not None and state["calls"] == state["die_at"]:
            raise KeyboardInterrupt  # the walltime kill, before the sixth batch is trained
        return real(backbone, group, tokenizer, chunk_scale, device)

    monkeypatch.setattr(tr, "value_regression_backward", spy)
    cfg = lambda: _config(tmp_path, path, "branched", max_epochs=2, batch_size=1, ckpt_every_steps=1,  # noqa: E731
                          milestone_every_steps=3)
    with pytest.raises(KeyboardInterrupt):
        train(cfg(), backbone=_backbone(), tokenizer=_CharTok())
    curve = tmp_path / "out" / "milestones" / "curve.jsonl"
    assert [json.loads(line)["step"] for line in curve.read_text().splitlines()] == [3]

    state["die_at"] = None
    train(cfg(), backbone=_backbone(), tokenizer=_CharTok())  # resumes at step 5
    assert [json.loads(line)["step"] for line in curve.read_text().splitlines()] == [3, 6, 8]
    kept = sorted(p.name for p in (tmp_path / "out" / "milestones" / "checkpoints").iterdir())
    assert kept == ["step_3", "step_6", "step_8"]


def test_milestones_on_the_single_action_kind_keep_weights_only(tmp_path):
    path = tmp_path / "ws.msgpack"
    with RecordWriter(path) as w:
        w.write(WarmStartRecord(samples=[_sample(2)], reward=1.0, is_truncated=False))
        w.write(WarmStartRecord(samples=[_sample(2)], reward=0.0, is_truncated=True))
    cfg = _config(tmp_path, path, "warmstart", max_epochs=2, milestone_every_steps=1)
    train(cfg, backbone=_backbone(), tokenizer=_CharTok())
    out = tmp_path / "out"
    assert sorted(p.name for p in (out / "milestones" / "checkpoints").iterdir()) == ["step_1", "step_2"]
    rows = [json.loads(line) for line in (out / "milestones" / "curve.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == [1, 2] and all("accuracy" not in r for r in rows)
    assert not (out / "gate").exists()


def test_no_milestones_by_default(tmp_path):
    path = _write(tmp_path / "d.msgpack", _tagged_records() + _holdout_contrast_records())
    train(_config(tmp_path, path, "branched", batch_size=1, ckpt_every_steps=1), backbone=_backbone(),
          tokenizer=_CharTok())
    assert not (tmp_path / "out" / "milestones").exists() and not (tmp_path / "out" / "gate").exists()


# ---------------------------------------------------------------- learning-rate schedule


def test_lr_scale_warmup_only():
    assert [lr_scale(s, 4, 100) for s in range(6)] == [0.25, 0.5, 0.75, 1.0, 1.0, 1.0]
    assert lr_scale(0, 0, 100) == 1.0 and lr_scale(99, 0, 100) == 1.0


def test_lr_scale_linear_and_cosine():
    assert [round(lr_scale(s, 0, 11, "linear", 0.0), 6) for s in (0, 5, 10)] == [1.0, 0.5, 0.0]
    cos = [lr_scale(s, 0, 11, "cosine", 0.1) for s in range(11)]
    assert cos[0] == 1.0 and abs(cos[-1] - 0.1) < 1e-12 and abs(cos[5] - 0.55) < 1e-12
    assert all(a >= b for a, b in zip(cos, cos[1:]))  # never increases
    # warmup, then decay: the peak is the first step after the warmup
    both = [lr_scale(s, 4, 8, "cosine", 0.5) for s in range(8)]
    assert [round(x, 6) for x in both] == [0.25, 0.5, 0.75, 1.0, 1.0, 0.875, 0.625, 0.5]
    # a schedule no longer than its warmup has nothing to decay over
    assert lr_scale(2, 4, 4, "cosine", 0.1) == 0.75 and lr_scale(3, 4, 5, "linear", 0.1) == 1.0


# ---------------------------------------------------------------- head groups


def test_head_lr_gives_the_heads_their_own_groups():
    bb = _backbone()
    legacy = setup_value_optimizer(bb, 1e-3, 2e-3, fp32_master=True)
    assert [g["lr"] for g in legacy.param_groups] == [1e-3, 2e-3, 0.0]
    split = setup_value_optimizer(bb, 1e-3, 2e-3, fp32_master=True, head_lr=1e-5)
    assert isinstance(split, MasterWeightAdamW)
    assert [g["lr"] for g in split.param_groups] == [1e-3, 2e-3, 0.0, 1e-5, 1e-5]
    n_legacy = [len(g["params"]) for g in legacy.param_groups]
    n_split = [len(g["params"]) for g in split.param_groups]
    assert n_split[3] == n_split[4] == 2  # weight + bias of one linear head each
    assert n_split[:3] == [n_legacy[0] - 2, n_legacy[1] - 2, n_legacy[2]]  # V_target keeps its head
    assert len(setup_value_optimizer(bb, 1e-3, 2e-3, head_lr=1e-5).param_groups) == 5  # plain AdamW too


def _spies(monkeypatch):
    import prime_rl.value_warmstart.train as tr

    captured: dict = {}
    real_setup = tr.setup_value_optimizer

    def setup_spy(backbone, v_lr, q_lr, **kw):
        captured["opt"] = real_setup(backbone, v_lr, q_lr, **kw)
        return captured["opt"]

    lrs: list[tuple] = []
    real_legs = tr.value_regression_backward
    state = {"calls": 0, "die_at": None}

    def legs_spy(backbone, group, tokenizer, chunk_scale, device):
        state["calls"] += 1
        if state["die_at"] is not None and state["calls"] == state["die_at"]:
            raise KeyboardInterrupt
        lrs.append(tuple(g["lr"] for g in captured["opt"].param_groups))
        return real_legs(backbone, group, tokenizer, chunk_scale, device)

    clips: list[float] = []
    real_clip = torch.nn.utils.clip_grad_norm_

    def clip_spy(params, max_norm, *a, **kw):
        clips.append(float(max_norm))
        return real_clip(params, max_norm, *a, **kw)

    monkeypatch.setattr(tr, "setup_value_optimizer", setup_spy)
    monkeypatch.setattr(tr, "value_regression_backward", legs_spy)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip_spy)
    return captured, lrs, clips, state


SCHEDULE = [0.25, 0.5, 0.75, 1.0, 1.0, 0.875, 0.625, 0.5]  # warmup 4, cosine to 0.5 over 8 steps


def _head_cfg(tmp_path, path):
    return _config(tmp_path, path, "branched", max_epochs=2, batch_size=1, ckpt_every_steps=1,
                   warmup_steps=4, max_grad_norm=1.0, v_learning_rate=4e-4, q_plus_learning_rate=8e-4,
                   head_learning_rate=1e-5, lr_decay="cosine", lr_final_scale=0.5)


def _check_schedule(lrs):
    assert len(lrs) == 8 and all(len(row) == 5 for row in lrs)
    for row, s in zip(lrs, SCHEDULE):
        v, q, vt, hv, hq = row
        assert abs(v - 4e-4 * s) < 1e-15 and abs(q - 8e-4 * s) < 1e-15 and vt == 0.0
        assert abs(hv - 1e-5 * s) < 1e-15 and abs(hq - 1e-5 * s) < 1e-15


def test_head_groups_follow_the_schedule_and_are_clipped_separately(tmp_path, monkeypatch):
    path = _write(tmp_path / "d.msgpack", _tagged_records())
    captured, lrs, clips, _ = _spies(monkeypatch)
    bb = _backbone()
    head_before = bb.v_head.linear.weight.detach().clone()
    train(_head_cfg(tmp_path, path), backbone=bb, tokenizer=_CharTok())
    _check_schedule(lrs)
    assert clips == [1.0] * 32  # V LoRA, Q+ LoRA, V head, Q+ head: once each per step
    assert not torch.equal(bb.v_head.linear.weight.detach(), head_before)  # the head trained
    # V_target (head included) was seeded from V at the last checkpoint
    assert torch.equal(bb.v_target_head.linear.weight, bb.v_head.linear.weight)
    assert len(captured["opt"].coherence()) == 5


def test_head_groups_survive_a_kill_and_resume(tmp_path, monkeypatch):
    path = _write(tmp_path / "d.msgpack", _tagged_records())
    _, lrs, _, state = _spies(monkeypatch)
    state["die_at"] = 4  # killed before the fourth batch: 3 steps done, checkpoint at step 3
    with pytest.raises(KeyboardInterrupt):
        train(_head_cfg(tmp_path, path), backbone=_backbone(), tokenizer=_CharTok())
    assert len(lrs) == 3
    state["die_at"] = None
    train(_head_cfg(tmp_path, path), backbone=_backbone(), tokenizer=_CharTok())  # 5-group state reloads
    _check_schedule(lrs)  # every step exactly once, each at its scheduled rate


def test_head_groups_can_be_switched_on_at_a_resume(tmp_path, monkeypatch):
    """The saved optimizer state has three groups, the resumed run five: the
    checkpoint manager keeps the weights and restarts the Adam moments (it logs
    a warning), and training continues from the saved step."""
    path = _write(tmp_path / "d.msgpack", _tagged_records())
    bb1 = _backbone()
    train(_config(tmp_path, path, "branched", max_epochs=1, batch_size=1, ckpt_every_steps=1),
          backbone=bb1, tokenizer=_CharTok())
    captured, lrs, _, _ = _spies(monkeypatch)
    summary = train(_config(tmp_path, path, "branched", max_epochs=2, batch_size=1, ckpt_every_steps=1,
                            head_learning_rate=1e-5), backbone=_backbone(), tokenizer=_CharTok())
    assert [e["epoch"] for e in summary["epochs"]] == [0, 1]
    assert len(lrs) == 4 and all(len(row) == 5 and row[3] == 1e-5 for row in lrs)  # epoch 1 only
    assert len(captured["opt"].param_groups) == 5


# ---------------------------------------------------------------- coherence


def _param(values):
    return torch.nn.Parameter(torch.tensor(values, dtype=torch.float32).to(torch.bfloat16))


def test_coherence_tells_signal_from_sign_flipping():
    steady, flipping, idle = _param([0.5] * 4), _param([0.5] * 4), _param([0.5] * 4)
    opt = MasterWeightAdamW([
        {"params": [steady], "lr": 1e-3},
        {"params": [flipping], "lr": 1e-3},
        {"params": [idle], "lr": 0.0},
    ])
    for step in range(200):
        steady.grad = torch.full_like(steady, 0.3)
        flipping.grad = torch.full_like(flipping, 0.3 if step % 2 == 0 else -0.3)
        idle.grad = torch.zeros_like(idle)
        opt.step()
    coh = opt.coherence()
    assert abs(coh[0] - 1.0) < 0.02  # a gradient that keeps its sign: full-size updates
    assert coh[1] < 0.1  # a gradient that flips every step: the updates cancel
    assert coh[2] == 0.0  # a frozen group has no state


# ---------------------------------------------------------------- free baseline


def test_constant_baselines_by_hand(tmp_path):
    # g1: trunk return 1 (3 targets), branch return 0 (root + 1 turn; the root has no V target)
    # g2: trunk return 0, branch return 1; g3 is held out
    path = _write(tmp_path / "b.msgpack", _records())
    base = constant_baselines(BranchedDataset(path))
    assert abs(base["l_v"]["constant"] - 0.5) < 1e-9 and abs(base["l_v"]["loss"] - 0.1875) < 1e-9
    assert abs(base["l_v"]["weight"] - 0.75) < 1e-9  # two samples carry half their weight on V
    assert abs(base["l_q"]["constant"] - 0.5) < 1e-9 and abs(base["l_q"]["loss"] - 0.25) < 1e-9
    assert abs(base["l_q"]["weight"] - 1.0) < 1e-9


def test_constant_baseline_uses_the_trainers_weighting(tmp_path):
    """A network that predicts 0 everywhere (zero-initialised heads, a learning
    rate too small to move them) must log the loss the baseline formula gives
    for the constant 0: loss(best constant) + constant^2 x weight."""
    path = _write(tmp_path / "b.msgpack", _records())
    cfg = _config(tmp_path, path, "branched", v_learning_rate=1e-12, q_plus_learning_rate=1e-12)
    summary = train(cfg, backbone=_backbone(), tokenizer=_CharTok())
    base = summary["constant_baseline"]
    for leg in ("l_v", "l_q"):
        zero_loss = base[leg]["loss"] + base[leg]["constant"] ** 2 * base[leg]["weight"]
        assert abs(summary["final"][leg] - zero_loss) < 1e-4
    assert abs(summary["final"]["l_v"] - 0.375) < 1e-4 and abs(summary["final"]["l_q"] - 0.5) < 1e-4


# ---------------------------------------------------------------- contrast payload


def test_score_pairs_reports_ties_and_margin():
    def pair(tag, g_i, g_j):
        return Pair(game_id="g", trunk_id=f"t{tag}", root_turn=0, a_i="a", a_j="b", g_i=g_i, g_j=g_j,
                    o_ids=[tag])

    scores = {(1, "a"): 0.6, (1, "b"): 0.4, (2, "a"): 0.5, (2, "b"): 0.5, (3, "a"): 0.7, (3, "b"): 0.3}
    payload = score_pairs([pair(1, 1.0, 0.0), pair(2, 0.0, 1.0), pair(3, 0.0, 1.0)],
                          lambda o_ids, action: scores[(o_ids[0], action)])
    assert payload["n_pairs"] == 3 and abs(payload["accuracy"] - 1 / 3) < 1e-9
    assert payload["n_ties"] == 1 and abs(payload["accuracy_no_ties"] - 0.5) < 1e-9
    assert abs(payload["mean_margin"] - (0.2 + 0.0 - 0.4) / 3) < 1e-9
