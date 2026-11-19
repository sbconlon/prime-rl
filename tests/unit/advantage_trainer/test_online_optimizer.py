"""Online Advantage Trainer optimizer (arm-redesign Phase 3).

The optimizer fixes of the offline warm-start trainer, switched on in the online
trainer through four opt-in settings: fp32_master (float32 master weights and
AdamW state), head_learning_rate (the value heads in their own groups),
max_grad_norm (each group clipped separately) and warmup_cycles, plus two
health readings in the cycle log line (gradient norms, update coherence).
At the defaults the optimizer, its updates and the log line are unchanged.

Two levels: the step functions on the tiny CPU backbone, and the real train()
entrypoint fed through a filesystem transport dir (_online).
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen2Config, Qwen2Model

import prime_rl.advantage_trainer.train as tr
from prime_rl.advantage_trainer.ckpt import AdvantageCheckpointManager, load_warm_start
from prime_rl.advantage_trainer.data import (
    prepare_action_advantage_samples,
    prepare_batched_advantage_samples,
)
from prime_rl.advantage_trainer.shared import (
    MasterWeightAdamW,
    setup_value_optimizer,
    value_regression_backward,
)
from prime_rl.configs.advantage_trainer import AdvantageTrainerConfig
from prime_rl.configs.trainer import CheckpointConfig, LoRAConfig
from prime_rl.orchestrator.value_networks import ValueNetworkConfig
from prime_rl.transport.advantage_batch import FileSystemAdvantageTrainingBatchSender
from prime_rl.transport.types import AdvantageTrainingBatch, DecisionPointTarget
from prime_rl.utils.logger import reset_logger
from prime_rl.value_warmstart.train import lr_scale
from tests.unit.advantage_trainer.test_jin_inner_loop import _make_batch
from tests.unit.advantage_trainer.test_train_step_action import _backbone, _CharTok, _sample

BF16 = torch.bfloat16
ANSI = re.compile(r"\x1b\[[0-9;]*m")
NUM = r"[-+0-9.e]+"


# ---------------------------------------------------------------- fixtures


def _action_batch() -> AdvantageTrainingBatch:
    s0 = _sample(
        [1, 2, 3], [10, 11, 12, 13],
        [DecisionPointTarget(0, "go", 1.0, 2.0), DecisionPointTarget(2, "look", 0.5, 0.25)],
    )
    s1 = _sample([4, 5], [20, 21, 22], [DecisionPointTarget(1, "take", 0.0, 1.0)])
    return AdvantageTrainingBatch(examples=[s0, s1], step=0)


def _one(i: int) -> AdvantageTrainingBatch:
    """Cycle i's batch for the train() runs: a single sample, so the cycle's
    unseeded shuffle cannot change the result."""
    s = _sample(
        [1 + i, 2, 3], [10, 11, 12, 13],
        [DecisionPointTarget(0, "go", 1.0, 2.0 - 0.25 * i), DecisionPointTarget(2, "look", 0.5, 0.25)],
    )
    return AdvantageTrainingBatch(examples=[s], step=i)


def _trainable(backbone) -> dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in backbone.named_parameters() if p.requires_grad}


def _inner(opt):
    """The torch optimizer that holds the AdamW state."""
    return opt._opt if isinstance(opt, MasterWeightAdamW) else opt


def _second_moments(opt, groups=(0, 1)) -> torch.Tensor:
    """Every second-moment entry of the given groups, as one vector (dtype kept)."""
    inner = _inner(opt)
    parts = [
        inner.state[p]["exp_avg_sq"].detach().clone().flatten()
        for g in groups
        for p in inner.param_groups[g]["params"]
        if p in inner.state
    ]
    return torch.cat(parts)


def _config(out: Path, transport: Path | None = None, **kw) -> AdvantageTrainerConfig:
    kw.setdefault("algorithm", "arm")
    return AdvantageTrainerConfig(
        model=ValueNetworkConfig(
            base_model_name="tiny", lora=LoRAConfig(rank=8, alpha=16.0, dropout=0.0), polyak_tau=0.005
        ),
        output_dir=out,
        transport_input_dir=transport,
        **kw,
    )


def _warm_state(tmp_path: Path) -> str:
    """A value_state.pt holding the trainable weights of _backbone(randomize_heads=True)."""
    mgr = AdvantageCheckpointManager(tmp_path / "warm", CheckpointConfig(skip_optimizer=True))
    mgr.save(0, _backbone(randomize_heads=True), None)
    return str(mgr.get_ckpt_path(0) / "value_state.pt")


def _saved(out: Path, step: int) -> dict:
    path = out / "checkpoints" / f"step_{step}" / "trainer" / "value_state.pt"
    return torch.load(path, map_location="cpu", weights_only=False)


def _tiny_base():
    # Same seed and construction order as _backbone(seed=0): the frozen base of a
    # train() run equals the fixture's, so the two levels can be compared exactly.
    torch.manual_seed(0)
    return Qwen2Model(
        Qwen2Config(
            vocab_size=256, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128,
        )
    )


def _online(monkeypatch, out: Path, batches, transport_name: str = "transport", first_step: int = 0, **settings):
    """Run the real train() entrypoint on the tiny CPU backbone. The batches are
    written to a filesystem transport dir as steps first_step..first_step+n-1,
    the way the orchestrator files them (a resumed trainer reads on from its
    checkpoint step, so a resumed run passes that step), and the model /
    tokenizer loaders are replaced. Returns the optimizer, the rates at every
    legs call and the cycle log lines. max_steps must not ask for more cycles
    than there are batches."""
    transport = out / transport_name
    sender = FileSystemAdvantageTrainingBatchSender(transport)
    for i, b in enumerate(batches):
        sender.send(AdvantageTrainingBatch(examples=b.examples, step=first_step + i))
    config = _config(out, transport, **settings)

    seen = SimpleNamespace(opt=None, backbone=None, lrs=[], lines=[], config=config)
    real_setup, real_legs, real_line = tr.setup_value_optimizer, tr.value_regression_backward, tr._cycle_log_line

    def setup_spy(backbone, v_lr, q_lr, **kw):
        seen.backbone = backbone
        seen.opt = real_setup(backbone, v_lr, q_lr, **kw)
        return seen.opt

    def legs_spy(backbone, chunk, tokenizer, chunk_scale, device):
        seen.lrs.append(tuple(g["lr"] for g in seen.opt.param_groups))
        return real_legs(backbone, chunk, tokenizer, chunk_scale, device)

    def line_spy(step, metrics):
        seen.lines.append(real_line(step, metrics))
        return seen.lines[-1]

    with monkeypatch.context() as mp:
        mp.setattr(tr, "AutoModel", SimpleNamespace(from_pretrained=lambda *a, **k: _tiny_base()))
        mp.setattr(tr, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda *a, **k: _CharTok()))
        mp.setattr(torch.cuda, "is_available", lambda: False)  # a unit test never touches a GPU
        mp.setattr(tr, "setup_value_optimizer", setup_spy)
        mp.setattr(tr, "value_regression_backward", legs_spy)
        mp.setattr(tr, "_cycle_log_line", line_spy)
        try:
            tr.train(config)
        finally:
            reset_logger()
    return seen


def _approx_rows(rows):
    return [pytest.approx(row, rel=1e-12, abs=0.0) for row in rows]


# ---------------------------------------------------------------- defaults


def _legacy_action_cycle(bb, opt, batch, tok, tau, n_epochs, mb_size, generator):
    """The action-level cycle as it was before the Phase-3 settings, frozen here
    as the reference: zero_grad, the shared legs per sample, a bare
    optimizer.step(), Polyak once after the inner loop."""
    device = next(bb.parameters()).device
    bb.train()
    n = len(batch.examples)
    for _ in range(n_epochs):
        shuffled = [batch.examples[i] for i in torch.randperm(n, generator=generator).tolist()]
        for start in range(0, n, mb_size):
            mb = shuffled[start : start + mb_size]
            opt.zero_grad()
            for sample in mb:  # inner_batch_size = 1
                value_regression_backward(bb, [sample], tok, 1.0 / float(len(mb)), device)
            opt.step()
    bb.polyak_update_v_target(tau=tau)


def _legacy_token_cycle(bb, opt, batch, tau, generator):
    """The token-level cycle as it was, for one epoch with the whole batch as
    one minibatch in one chunk."""
    bb.train()
    n = len(batch.examples)
    shuffled = [batch.examples[i] for i in torch.randperm(n, generator=generator).tolist()]
    prepared = prepare_batched_advantage_samples(shuffled, device="cpu")
    mask_f = prepared.loss_mask.to(dtype=prepared.v_targets.dtype)
    valid_count = prepared.loss_mask.sum(dim=1).clamp(min=1)
    opt.zero_grad()
    v_pred, _ = bb.forward_v_all_positions(prepared.input_ids)
    l_v = (((v_pred - prepared.v_targets) ** 2 * mask_f).sum(dim=1) / valid_count).mean()
    (l_v * 1.0).backward()
    q_pred, _ = bb.forward_q_plus_sampled_all_positions(prepared.input_ids)
    l_q = (((q_pred - prepared.q_plus_targets) ** 2 * mask_f).sum(dim=1) / valid_count).mean()
    (l_q * 1.0).backward()
    opt.step()
    bb.polyak_update_v_target(tau=tau)


def _legacy_line(step: int, metrics: dict) -> str:
    """The cycle log line as it was (the f-string of train(), verbatim)."""
    return (
        f"step={step} inner_steps={metrics['inner_steps']} "
        f"l_v_first_epoch={metrics['l_v_first_epoch']:.4f} "
        f"l_v_last_epoch={metrics['l_v_last_epoch']:.4f} "
        f"l_v_mean={metrics['l_v']:.4f} "
        f"l_q_first_epoch={metrics['l_q_first_epoch']:.4f} "
        f"l_q_last_epoch={metrics['l_q_last_epoch']:.4f} "
        f"l_q_mean={metrics['l_q']:.4f} "
        f"q_tgt_abs={metrics['q_plus_target_abs_mean']:.4f} "
        f"n_samples={metrics['n_samples']}"
    )


def _layout(backbone, opt) -> list[tuple[list[str], dict]]:
    """Per group: the parameter names and every hyperparameter."""
    names = {id(p): n for n, p in backbone.named_parameters()}
    return [
        ([names[id(p)] for p in g["params"]], {k: v for k, v in g.items() if k != "params"})
        for g in opt.param_groups
    ]


def _assert_same_state(bb_a, opt_a, bb_b, opt_b):
    a, b = _trainable(bb_a), _trainable(bb_b)
    assert a.keys() == b.keys()
    for name in a:
        assert torch.equal(a[name], b[name]), name
    sa, sb = _inner(opt_a).state_dict()["state"], _inner(opt_b).state_dict()["state"]
    assert sa.keys() == sb.keys() and len(sa) > 0
    for idx in sa:
        assert sa[idx].keys() == sb[idx].keys()
        for key in sa[idx]:
            assert torch.equal(torch.as_tensor(sa[idx][key]), torch.as_tensor(sb[idx][key])), (idx, key)


def test_defaults_unchanged(tmp_path):
    cfg = _config(tmp_path)
    assert (cfg.fp32_master, cfg.head_learning_rate, cfg.max_grad_norm, cfg.warmup_cycles) == (False, None, None, 0)
    tok = _CharTok()

    # the optimizer: plain AdamW, the three slot groups, torch's default hyperparameters
    bb_new, bb_old = _backbone(randomize_heads=True), _backbone(randomize_heads=True)
    opt_new, base_lrs = tr._setup_optimizer(bb_new, cfg)
    opt_old = setup_value_optimizer(bb_old, cfg.learning_rate, cfg.learning_rate)  # the call as it was
    assert type(opt_new) is torch.optim.AdamW and type(opt_old) is torch.optim.AdamW
    assert base_lrs == [cfg.learning_rate, cfg.learning_rate, 0.0]
    assert _layout(bb_new, opt_new) == _layout(bb_old, opt_old)
    assert len(opt_new.param_groups) == 3
    heads = [n for names, _ in _layout(bb_new, opt_new) for n in names if "head" in n]
    assert len(heads) == 6  # every head still sits in its slot's group

    # one action-level cycle through the new path == the cycle as it was, bit for bit
    tr._set_learning_rates(opt_new, base_lrs, 0, cfg.warmup_cycles)
    assert _layout(bb_new, opt_new) == _layout(bb_old, opt_old)  # the rates are untouched
    metrics = tr._train_step_action(
        bb_new, opt_new, _action_batch(), tok, polyak_tau=0.5, n_epochs=2, minibatch_size=1,
        inner_batch_size=1, generator=torch.Generator().manual_seed(7), max_grad_norm=cfg.max_grad_norm,
    )
    _legacy_action_cycle(bb_old, opt_old, _action_batch(), tok, 0.5, 2, 1, torch.Generator().manual_seed(7))
    _assert_same_state(bb_new, opt_new, bb_old, opt_old)
    assert metrics["inner_steps"] == 4
    assert "gnorm" not in metrics  # nothing is measured on the default path

    # the log line is the old one, byte for byte
    assert tr._cycle_log_line(3, metrics) == _legacy_line(3, metrics)

    # the token-level cycle (PPO / legacy ARM path) likewise
    bb_new, bb_old = _backbone(randomize_heads=True), _backbone(randomize_heads=True)
    opt_new, base_lrs = tr._setup_optimizer(bb_new, cfg)
    opt_old = setup_value_optimizer(bb_old, cfg.learning_rate, cfg.learning_rate)
    tr._set_learning_rates(opt_new, base_lrs, 0, cfg.warmup_cycles)
    batch = _make_batch(3)
    metrics = tr._train_step(
        bb_new, opt_new, batch, "arm", polyak_tau=0.5, n_epochs=1, minibatch_size=None,
        inner_batch_size=3, generator=torch.Generator().manual_seed(7), max_grad_norm=cfg.max_grad_norm,
    )
    _legacy_token_cycle(bb_old, opt_old, batch, 0.5, torch.Generator().manual_seed(7))
    _assert_same_state(bb_new, opt_new, bb_old, opt_old)
    assert "gnorm" not in metrics

    # the per-slot rate overrides keep their meaning
    _, base_lrs = tr._setup_optimizer(_backbone(), _config(tmp_path, v_learning_rate=2e-4))
    assert base_lrs == [2e-4, cfg.learning_rate, 0.0]


def test_defaults_unchanged_through_train(tmp_path, monkeypatch):
    """The real loop at the defaults: plain AdamW, constant rates, the old log
    line, and the weights of the cycle as it was."""
    warm = _warm_state(tmp_path)
    run = _online(monkeypatch, tmp_path / "run", [_one(0)], max_steps=1, warm_start_path=warm,
                  n_epochs=2, polyak_tau=0.5, ckpt=CheckpointConfig(interval=1))
    assert type(run.opt) is torch.optim.AdamW
    assert run.lrs == [(1e-3, 1e-3, 0.0)] * 2
    assert all(key not in run.lines[0] for key in ("gnorm=", "coh=", "lr="))
    assert re.fullmatch(
        rf"step=0 inner_steps=2 l_v_first_epoch={NUM} l_v_last_epoch={NUM} l_v_mean={NUM} "
        rf"l_q_first_epoch={NUM} l_q_last_epoch={NUM} l_q_mean={NUM} q_tgt_abs={NUM} n_samples=1",
        run.lines[0],
    )

    ref = _backbone(randomize_heads=True)
    opt = setup_value_optimizer(ref, 1e-3, 1e-3)
    _legacy_action_cycle(ref, opt, _one(0), _CharTok(), 0.5, 2, 1, torch.Generator().manual_seed(0))
    final = _saved(tmp_path / "run", 1)["trainable_state_dict"]
    want = _trainable(ref)
    assert final.keys() == want.keys()
    assert all(torch.equal(final[n], want[n]) for n in want)


# ---------------------------------------------------------------- float32 state on a bfloat16 model


def _fitted(bb, sample, tok):
    """`sample` with every target replaced by the network's current prediction:
    a batch the networks already fit, so its gradients are exactly zero."""
    prep = prepare_action_advantage_samples([sample], tok, device="cpu")
    with torch.no_grad():
        v_all, _ = bb.forward_v_all_positions(prep.trajectory_input_ids)
        v = v_all[prep.dp_sample_idx, prep.dp_v_pos]
        q = [
            bb.forward_q_plus_action(torch.tensor([o], dtype=torch.long), torch.tensor([a], dtype=torch.long))[0]
            for o, a in zip(prep.q_plus_obs_ids, prep.q_plus_action_ids)
        ]
    targets = [
        DecisionPointTarget(t.response_start, t.executed_action, float(v[i]), float(q[i]))
        for i, t in enumerate(sample.decision_point_targets)
    ]
    return _sample(sample.prompt_ids, sample.completion_ids, targets)


def test_second_moment_decays_online():
    """A bfloat16 model, as on the GPU. A spike, then a quiet step (a batch the
    networks fit exactly): with fp32_master the second moment falls by the
    factor beta2 = 0.999; with the default optimizer not one entry falls."""
    tok = _CharTok()
    spike = _sample(
        [1, 2, 3], [10, 11, 12, 13],
        [DecisionPointTarget(0, "go", 100.0, -100.0), DecisionPointTarget(2, "look", -50.0, 80.0)],
    )

    def run(fp32_master):
        bb = _backbone(randomize_heads=True).to(BF16)
        opt = setup_value_optimizer(bb, 1e-4, 1e-4, fp32_master=fp32_master)
        tr._train_step_action(bb, opt, AdvantageTrainingBatch(examples=[spike], step=0), tok, polyak_tau=0.0)
        v_spike = _second_moments(opt)
        quiet = AdvantageTrainingBatch(examples=[_fitted(bb, spike, tok)], step=1)
        metrics = tr._train_step_action(bb, opt, quiet, tok, polyak_tau=0.0)
        assert all(p.dtype == BF16 for p in bb.parameters())  # the model stays in bfloat16
        return v_spike, _second_moments(opt), metrics

    try:
        v_spike, v_quiet, metrics = run(fp32_master=True)
        old_spike, old_quiet, _ = run(fp32_master=False)
    except RuntimeError as exc:  # a torch build without the bfloat16 CPU kernels the tiny model needs
        if "bfloat16" in str(exc).lower() or "not implemented" in str(exc).lower():
            pytest.skip(f"bfloat16 CPU kernel missing: {exc}")
        raise

    assert metrics["l_v"] == 0.0 and metrics["l_q"] == 0.0  # the quiet batch is fit exactly
    assert v_spike.dtype == torch.float32 and v_quiet.dtype == torch.float32
    hit = v_spike > 0
    assert int(hit.sum()) > 1000  # the spike reached the LoRA and the heads of both slots
    assert torch.allclose(v_quiet, 0.999 * v_spike, rtol=1e-6, atol=0.0)
    assert bool((v_quiet[hit] < v_spike[hit]).all())

    # the default optimizer on the same model: bfloat16 state, the decay rounds away
    assert old_spike.dtype == BF16 and old_quiet.dtype == BF16
    assert int((old_spike > 0).sum()) > 1000
    assert int((old_quiet.float() < old_spike.float()).sum()) == 0
    assert torch.equal(old_quiet, old_spike)


# ---------------------------------------------------------------- per-group clipping


def _lora_deltas(spike_cycle: int | None, head_lr: float | None, fp32_master: bool):
    """Three cycles of one optimizer step each at clip 1.0. In cycle spike_cycle
    (None: never) every head gradient is a hundred times its size: one bad batch.
    Returns how far the V and the Q+ LoRA moved, and each cycle's gradient norms."""
    bb = _backbone(randomize_heads=True)
    opt = setup_value_optimizer(bb, 1e-3, 1e-3, fp32_master=fp32_master, head_lr=head_lr)
    scale = {"now": 1.0}
    for head in (bb.v_head, bb.q_plus_head):
        for p in head.parameters():
            p.register_hook(lambda g: g * scale["now"])
    lora = [n for n, p in bb.named_parameters() if p.requires_grad and n.endswith((".0", ".1")) and ".lora_" in n]
    before = _trainable(bb)
    norms = []
    for cycle in range(3):
        scale["now"] = 100.0 if cycle == spike_cycle else 1.0
        metrics = tr._train_step_action(
            bb, opt, _action_batch(), _CharTok(), polyak_tau=0.0, n_epochs=1, minibatch_size=None,
            inner_batch_size=1, generator=torch.Generator().manual_seed(3), max_grad_norm=1.0,
        )
        assert metrics["inner_steps"] == 1
        norms.append(metrics["gnorm"])
    after = _trainable(bb)
    return torch.cat([(after[n] - before[n]).flatten() for n in lora]), norms


@pytest.mark.parametrize("fp32_master", [True, False])
def test_groups_clipped_separately_online(fp32_master):
    # The heads sit in their own groups at a rate too small to move them, so the
    # only way a head gradient could reach the LoRA's update is through the clip.
    frozen = 1e-30
    calm, gn_calm = _lora_deltas(None, frozen, fp32_master)
    spiky, gn_spiky = _lora_deltas(1, frozen, fp32_master)
    assert float(calm.abs().sum()) > 0.0
    assert all(len(row) == 5 and row[2] == 0.0 for row in gn_calm + gn_spiky)  # V_target: never clipped
    # the spike: head gradients a hundred times their size, over a hundred times the LoRA's
    assert gn_spiky[1][3] == pytest.approx(100.0 * gn_calm[1][3], rel=1e-4)
    assert gn_spiky[1][4] == pytest.approx(100.0 * gn_calm[1][4], rel=1e-4)
    assert gn_spiky[1][3] > 100.0 * gn_spiky[1][0] and gn_spiky[1][4] > 100.0 * gn_spiky[1][1]
    assert gn_calm[1][3] > 1.0 and gn_calm[1][4] > 1.0  # the heads are clipped with and without the spike
    # the LoRA's norms and its update are untouched by the heads' clip, in every cycle
    assert [row[:2] for row in gn_spiky] == [row[:2] for row in gn_calm]
    assert torch.equal(spiky, calm)

    # The contrast, today's layout (the heads inside their slot's group): the same
    # bad batch shrinks the LoRA's gradient through the shared clip, and Adam,
    # which normalises by the gradient's recent size, moves the LoRA elsewhere.
    calm_shared, gn = _lora_deltas(None, None, fp32_master)
    spiky_shared, _ = _lora_deltas(1, None, fp32_master)
    assert all(len(row) == 3 for row in gn)
    rel = float((spiky_shared - calm_shared).norm() / calm_shared.norm())
    assert rel > 0.1, rel


# ---------------------------------------------------------------- the slow copy of V


def _v_source(name: str) -> str:
    if name.startswith("v_target_head."):
        return name.replace("v_target_head.", "v_head.", 1)
    assert name.endswith(".2")
    return name[:-1] + "0"


@pytest.mark.parametrize("fp32_master", [True, False])
@pytest.mark.parametrize("path", ["action", "token"])
def test_v_target_moves_only_by_polyak(path, fp32_master):
    """Two cycles: after each, V_target (LoRA slot 2 and its head) is exactly the
    Polyak blend of its value before the cycle and V after it. The multi-slot
    LoRA hands slot 2 zero-valued gradients and the head gets none; a write-back
    of the optimizer's stale master copy would show in the second cycle."""
    tau = 0.5
    bb = _backbone(randomize_heads=True)
    opt = setup_value_optimizer(bb, 1e-2, 1e-2, fp32_master=fp32_master, head_lr=1e-3)
    assert isinstance(opt, MasterWeightAdamW) == fp32_master
    params = dict(bb.named_parameters())
    slow = [n for n in params if (".lora_" in n and n.endswith(".2")) or n.startswith("v_target_head.")]
    assert sum(".lora_" in n for n in slow) >= 2 and sum("head" in n for n in slow) == 2
    generator = torch.Generator().manual_seed(11)

    for cycle in range(2):
        before = {n: params[n].detach().clone() for n in slow}
        if path == "action":
            metrics = tr._train_step_action(
                bb, opt, _action_batch(), _CharTok(), polyak_tau=tau, n_epochs=2, minibatch_size=1,
                generator=generator, max_grad_norm=1.0,
            )
        else:
            metrics = tr._train_step(
                bb, opt, _make_batch(3), "arm", polyak_tau=tau, n_epochs=2, minibatch_size=2,
                generator=generator, max_grad_norm=1.0,
            )
        assert metrics["inner_steps"] == 4 and metrics["gnorm"][2] == 0.0
        for n in slow:
            blend = before[n].clone().mul_(1.0 - tau).add_(params[_v_source(n)].detach(), alpha=tau)
            assert torch.equal(params[n].detach(), blend), (cycle, n)
            assert not torch.equal(params[n].detach(), before[n]), (cycle, n)  # Polyak did move it
        # the two gradient conditions the optimizer has to leave alone
        assert all(params[n].grad is not None and not bool(params[n].grad.any()) for n in slow if ".lora_" in n)
        assert all(params[n].grad is None for n in slow if "head" in n)
        if fp32_master:
            # the optimizer's own copy of the slow weights is stale from the first Polyak update on
            assert any(not torch.equal(m, p) for p, m in zip(opt._model_params[2], opt._masters[2]))


# ---------------------------------------------------------------- warmup


def test_warmup_scales_rates(tmp_path, monkeypatch):
    # the scale: the offline trainer's lr_scale without decay
    assert [tr._warmup_scale(c, 10) for c in (0, 4, 9, 10)] == [0.1, 0.5, 1.0, 1.0]
    for warmup in (0, 1, 4, 10):
        assert [tr._warmup_scale(c, warmup) for c in range(30)] == [lr_scale(c, warmup, 1000) for c in range(30)]

    # applied to every group; V_target stays at exactly 0
    opt = setup_value_optimizer(_backbone(), 4e-4, 8e-4, fp32_master=True, head_lr=1e-5)
    base = [4e-4, 8e-4, 0.0, 1e-5, 1e-5]
    for cycle, scale in ((0, 0.1), (4, 0.5), (9, 1.0), (10, 1.0), (500, 1.0)):
        tr._set_learning_rates(opt, base, cycle, 10)
        assert [g["lr"] for g in opt.param_groups] == _approx_rows([[b * scale for b in base]])[0]
        assert opt.param_groups[2]["lr"] == 0.0

    # the real loop: 11 cycles, one legs call each
    run = _online(
        monkeypatch, tmp_path / "run", [_one(i) for i in range(11)], max_steps=11,
        warm_start_path=_warm_state(tmp_path), fp32_master=True, v_learning_rate=4e-4,
        q_plus_learning_rate=8e-4, head_learning_rate=1e-5, warmup_cycles=10, n_epochs=1,
    )
    assert len(run.lrs) == 11
    want = [[b * min(1.0, (c + 1) / 10) for b in base] for c in range(11)]
    assert [list(row) for row in run.lrs] == _approx_rows(want)
    for cycle, scale in ((0, 0.1), (4, 0.5), (9, 1.0), (10, 1.0)):
        assert run.lrs[cycle][0] == pytest.approx(4e-4 * scale, rel=1e-12)
        assert run.lrs[cycle][3] == pytest.approx(1e-5 * scale, rel=1e-12)
    assert all(row[2] == 0.0 for row in run.lrs)
    assert "lr=[4.00e-05,8.00e-05,1.00e-06,1.00e-06]" in run.lines[0]  # the line shows the cycle's rates
    assert "lr=[4.00e-04,8.00e-04,1.00e-05,1.00e-05]" in run.lines[10]


@pytest.mark.parametrize("fp32_master", [False, True])
def test_rates_follow_the_config_on_resume(tmp_path, monkeypatch, fp32_master):
    """A rate changed between jobs takes effect on resume: the checkpoint holds
    the old one, the config wins (for the default optimizer too)."""
    out = tmp_path / "run"
    common = dict(warm_start_path=_warm_state(tmp_path), fp32_master=fp32_master, n_epochs=1)
    first = _online(monkeypatch, out, [_one(0)], max_steps=1, ckpt=CheckpointConfig(interval=1),
                    learning_rate=1e-3, **common)
    assert first.lrs == [(1e-3, 1e-3, 0.0)]
    assert [g["lr"] for g in _saved(out, 1)["optimizer_state_dict"]["param_groups"]] == [1e-3, 1e-3, 0.0]

    second = _online(monkeypatch, out, [_one(1)], first_step=1, max_steps=2,
                     ckpt=CheckpointConfig(interval=1, resume_step=-1), learning_rate=1e-3,
                     q_plus_learning_rate=2e-4, **common)
    assert second.lrs == [(1e-3, 2e-4, 0.0)]
    assert _saved(out, 2)["step"] == 2


# ---------------------------------------------------------------- checkpoints and warm start


SETTINGS = dict(fp32_master=True, head_learning_rate=1e-4, max_grad_norm=1.0, learning_rate=1e-3,
                n_epochs=2, polyak_tau=0.5)


def test_warm_start_is_not_overwritten_by_stale_masters(tmp_path, monkeypatch):
    """The optimizer is built before the warm start is loaded, so its float32
    masters hold the pre-load weights. train() must sync them: the first cycle
    then equals the same cycle run on the warm weights directly."""
    run = _online(monkeypatch, tmp_path / "run", [_one(0)], max_steps=1, warm_start_path=_warm_state(tmp_path),
                  ckpt=CheckpointConfig(interval=1), **SETTINGS)
    assert isinstance(run.opt, MasterWeightAdamW)
    final = _saved(tmp_path / "run", 1)["trainable_state_dict"]

    ref = _backbone(randomize_heads=True)
    start = _trainable(ref)
    opt = setup_value_optimizer(ref, 1e-3, 1e-3, fp32_master=True, head_lr=1e-4)
    tr._train_step_action(ref, opt, _one(0), _CharTok(), polyak_tau=0.5, n_epochs=2, max_grad_norm=1.0)
    want = _trainable(ref)
    assert final.keys() == want.keys()
    for name in want:
        assert torch.equal(final[name], want[name]), name
    assert any(not torch.equal(want[n], start[n]) for n in want if n.endswith(".lora_A.0"))  # it trained


def test_checkpoint_round_trip_online(tmp_path, monkeypatch):
    """Save, load, one more cycle == an uninterrupted run: weights, AdamW state
    and rates (the warmup follows the saved cycle count)."""
    settings = dict(SETTINGS, warm_start_path=_warm_state(tmp_path), warmup_cycles=4)
    batches = [_one(0), _one(1), _one(2)]
    # three cycles in one job
    _online(monkeypatch, tmp_path / "full", batches, max_steps=3, ckpt=CheckpointConfig(interval=1), **settings)
    # the same run killed after two cycles ...
    cut = _online(monkeypatch, tmp_path / "cut", batches[:2], max_steps=2, ckpt=CheckpointConfig(interval=1),
                  **settings)
    mid = _saved(tmp_path / "cut", 2)
    assert mid["step"] == 2 and "optimizer_state_dict" in mid
    assert len(mid["optimizer_state_dict"]["param_groups"]) == 5
    states = [s for s in mid["optimizer_state_dict"]["state"].values() if "exp_avg_sq" in s]
    assert states and all(s["exp_avg_sq"].dtype == torch.float32 for s in states)
    # ... and resumed for the third
    resumed = _online(monkeypatch, tmp_path / "cut", batches[2:], first_step=2, max_steps=3,
                      ckpt=CheckpointConfig(interval=1, resume_step=-1), **settings)
    assert len(cut.lrs) == 4 and len(resumed.lrs) == 2  # two optimizer steps per cycle; one cycle after the resume
    assert [list(row) for row in resumed.lrs] == _approx_rows([[0.75e-3, 0.75e-3, 0.0, 0.75e-4, 0.75e-4]] * 2)
    assert resumed.lines[0].startswith("step=2 ")

    a, b = _saved(tmp_path / "full", 3), _saved(tmp_path / "cut", 3)
    assert a["step"] == b["step"] == 3
    assert a["trainable_state_dict"].keys() == b["trainable_state_dict"].keys()
    for name in a["trainable_state_dict"]:
        assert torch.equal(a["trainable_state_dict"][name], b["trainable_state_dict"][name]), name
    sa, sb = a["optimizer_state_dict"]["state"], b["optimizer_state_dict"]["state"]
    assert sa.keys() == sb.keys() and len(sa) > 0
    for idx in sa:
        for key in ("step", "exp_avg", "exp_avg_sq"):
            assert torch.equal(torch.as_tensor(sa[idx][key]), torch.as_tensor(sb[idx][key])), (idx, key)
    assert [g["lr"] for g in a["optimizer_state_dict"]["param_groups"]] == [
        g["lr"] for g in b["optimizer_state_dict"]["param_groups"]
    ]
    # the resumed cycle changed the weights (the comparison is not between two no-ops)
    assert any(
        not torch.equal(mid["trainable_state_dict"][n], b["trainable_state_dict"][n])
        for n in b["trainable_state_dict"]
    )


def test_group_layout_can_change_at_a_resume_online(tmp_path, monkeypatch):
    """A checkpoint of the default optimizer (three groups) resumed with head
    groups (five): the checkpoint manager keeps the weights and restarts the
    Adam moments (it logs a warning). The resumed cycle then equals a fresh
    master-weight optimizer run on the checkpoint's weights."""
    out = tmp_path / "run"
    warm = _warm_state(tmp_path)
    _online(monkeypatch, out, [_one(0)], max_steps=1, warm_start_path=warm, learning_rate=1e-3, n_epochs=2,
            polyak_tau=0.5, ckpt=CheckpointConfig(interval=1))
    assert len(_saved(out, 1)["optimizer_state_dict"]["param_groups"]) == 3
    resumed = _online(monkeypatch, out, [_one(1)], first_step=1, max_steps=2,
                      warm_start_path=warm, ckpt=CheckpointConfig(interval=1, resume_step=-1), **SETTINGS)
    assert isinstance(resumed.opt, MasterWeightAdamW) and len(resumed.opt.param_groups) == 5
    assert resumed.lines[0].startswith("step=1 ")

    ref = _backbone(randomize_heads=True)
    load_warm_start(ref, out / "checkpoints" / "step_1" / "trainer" / "value_state.pt")
    opt = setup_value_optimizer(ref, 1e-3, 1e-3, fp32_master=True, head_lr=1e-4)
    tr._train_step_action(ref, opt, _one(1), _CharTok(), polyak_tau=0.5, n_epochs=2, max_grad_norm=1.0)
    final, want = _saved(out, 2)["trainable_state_dict"], _trainable(ref)
    for name in want:
        assert torch.equal(final[name], want[name]), name
    assert len(_saved(out, 2)["optimizer_state_dict"]["param_groups"]) == 5


def test_token_level_path_gets_the_settings_through_train(tmp_path, monkeypatch):
    """The PPO branch of train() (_train_step): master weights, head groups and
    the clip reach it, and the readings reach the line."""
    batch = _make_batch(1, with_q_plus=False)
    settings = dict(SETTINGS, max_grad_norm=0.1)
    run = _online(monkeypatch, tmp_path / "run", [batch], max_steps=1, algorithm="ppo",
                  warm_start_path=_warm_state(tmp_path), ckpt=CheckpointConfig(interval=1), **settings)
    assert isinstance(run.opt, MasterWeightAdamW)
    assert re.search(rf" gnorm=\[{NUM},{NUM},{NUM},{NUM}\] coh=\[{NUM},{NUM},{NUM},{NUM}\] lr=\[", run.lines[0])

    ref = _backbone(randomize_heads=True)
    opt = setup_value_optimizer(ref, 1e-3, 1e-3, fp32_master=True, head_lr=1e-4)
    metrics = tr._train_step(ref, opt, batch, "ppo", polyak_tau=0.5, n_epochs=2, max_grad_norm=0.1)
    assert metrics["gnorm"][0] > 0.1 and metrics["gnorm"][3] > 0.1  # the clip is active
    assert metrics["gnorm"][1] == 0.0 and metrics["gnorm"][4] == 0.0  # PPO: no Q+ leg
    final, want = _saved(tmp_path / "run", 1)["trainable_state_dict"], _trainable(ref)
    for name in want:
        assert torch.equal(final[name], want[name]), name


def test_checkpoint_round_trip_online_bfloat16_state_loads_cast_up(tmp_path):
    """A checkpoint written by the default optimizer on a bfloat16 model holds
    bfloat16 AdamW state. Loaded into the master-weight optimizer it is cast up
    to float32, value for value, and from then on it can decay."""
    tok = _CharTok()
    mgr = AdvantageCheckpointManager(tmp_path, CheckpointConfig())
    batch = _action_batch()
    try:
        old_bb = _backbone(randomize_heads=True).to(BF16)
        old_opt = setup_value_optimizer(old_bb, 1e-4, 1e-4)
        tr._train_step_action(old_bb, old_opt, batch, tok, polyak_tau=0.5, n_epochs=2)
        mgr.save(1, old_bb, old_opt)

        bb = _backbone(seed=5, randomize_heads=True).to(BF16)  # other weights, overwritten by the load
        opt = setup_value_optimizer(bb, 1e-4, 1e-4, fp32_master=True)
        assert mgr.load(1, bb, opt) == 1
        opt.sync_from_params()
        loaded = {
            i: {k: torch.as_tensor(v).clone() for k, v in s.items()}
            for i, s in opt.state_dict()["state"].items()
        }
        v_loaded = _second_moments(opt)
        loaded_weights = _trainable(bb)
        masters_follow = all(
            torch.equal(m.detach(), p.detach().to(torch.float32))
            for ps, ms in zip(opt._model_params, opt._masters)
            for p, m in zip(ps, ms)
        )
        tr._train_step_action(bb, opt, batch, tok, polyak_tau=0.5, n_epochs=2, max_grad_norm=1.0)
    except RuntimeError as exc:
        if "bfloat16" in str(exc).lower() or "not implemented" in str(exc).lower():
            pytest.skip(f"bfloat16 CPU kernel missing: {exc}")
        raise

    saved = _saved(tmp_path, 1)["optimizer_state_dict"]["state"]
    assert loaded.keys() == saved.keys() and len(saved) > 0
    n_moments = 0
    for idx, state in saved.items():
        for key in ("exp_avg", "exp_avg_sq"):
            assert state[key].dtype == BF16
            assert loaded[idx][key].dtype == torch.float32
            assert torch.equal(loaded[idx][key], state[key].to(torch.float32)), (idx, key)
            n_moments += 1
        assert float(loaded[idx]["step"]) == float(state["step"]) == 2.0
    assert n_moments > 0
    # weights: the model holds the checkpoint's bfloat16 values, the masters their float32 copies
    for name, tensor in _saved(tmp_path, 1)["trainable_state_dict"].items():
        assert tensor.dtype == BF16
        assert torch.equal(loaded_weights[name], tensor), name
    assert masters_follow
    for ps, ms in zip(opt._model_params, opt._masters):
        for p, m in zip(ps, ms):
            assert p.dtype == BF16 and m.dtype == torch.float32
    # two more steps in float32: second moments now fall where the gradient got smaller
    v_after = _second_moments(opt)
    assert v_after.dtype == torch.float32
    assert int((v_after < v_loaded).sum()) > 0


# ---------------------------------------------------------------- health readings


HEALTH = re.compile(r" gnorm=\[(?P<gnorm>[^\]]*)\] coh=\[(?P<coh>[^\]]*)\] lr=\[(?P<lr>[^\]]*)\]$")


def test_health_readings_logged(tmp_path, monkeypatch):
    warm = _warm_state(tmp_path)
    run = _online(monkeypatch, tmp_path / "on", [_one(0), _one(1)], max_steps=2, warm_start_path=warm,
                  fp32_master=True, head_learning_rate=1e-4, max_grad_norm=1.0, learning_rate=1e-3, n_epochs=3)
    log = ANSI.sub("", (tmp_path / "on" / "logs" / "advantage_trainer.log").read_text())
    assert len(run.lines) == 2
    for step, line in enumerate(run.lines):
        assert line in log  # it reached the log file
        assert line.startswith(f"step={step} inner_steps=3 ")
        m = HEALTH.search(line)
        assert m is not None, line
        gnorm = [float(x) for x in m["gnorm"].split(",")]
        coh = [float(x) for x in m["coh"].split(",")]
        lrs = [float(x) for x in m["lr"].split(",")]
        # four groups: V, Q+, V head, Q+ head (the frozen V_target group is left out)
        assert len(gnorm) == len(coh) == len(lrs) == 4
        assert all(g > 0.0 for g in gnorm)
        assert gnorm[3] > 1.0  # above the clip of 1.0: the reading is taken before clipping
        assert all(0.0 < c < 1.5 for c in coh)
        assert lrs == [1e-3, 1e-3, 1e-4, 1e-4]
        # the old part of the line is unchanged in front of the readings
        assert re.fullmatch(
            rf"step={step} inner_steps=3 l_v_first_epoch={NUM} l_v_last_epoch={NUM} l_v_mean={NUM} "
            rf"l_q_first_epoch={NUM} l_q_last_epoch={NUM} l_q_mean={NUM} q_tgt_abs={NUM} n_samples=1",
            line[: m.start()],
        )
    # the last line's coherence is the optimizer's own reading
    full = run.opt.coherence()
    assert len(full) == 5 and full[2] == 0.0
    assert coh == [float(f"{c:.3f}") for i, c in enumerate(full) if i != 2]

    # the default optimizer with clipping only: norms and rates, no coherence, three groups
    clip = _online(monkeypatch, tmp_path / "clip", [_one(0)], max_steps=1, warm_start_path=warm,
                   max_grad_norm=1.0, learning_rate=1e-3, n_epochs=3)
    assert type(clip.opt) is torch.optim.AdamW
    m = re.search(rf" gnorm=\[({NUM}),({NUM})\] lr=\[1.00e-03,1.00e-03\]$", clip.lines[0])
    assert m is not None and "coh=" not in clip.lines[0], clip.lines[0]
    assert float(m[1]) > 1.0 and float(m[2]) > 1.0  # before clipping here too

    # the defaults: nothing is appended
    off = _online(monkeypatch, tmp_path / "off", [_one(0)], max_steps=1, warm_start_path=warm,
                  learning_rate=1e-3, n_epochs=3)
    assert all(key not in off.lines[0] for key in ("gnorm=", "coh=", "lr="))


def test_cycle_log_line_formats_the_readings():
    metrics = {
        "inner_steps": 4, "l_v_first_epoch": 0.5, "l_v_last_epoch": 0.25, "l_v": 0.375,
        "l_q_first_epoch": 1.0, "l_q_last_epoch": 0.5, "l_q": 0.75, "q_plus_target_abs_mean": 0.3,
        "n_samples": 2,
    }
    old = _legacy_line(7, metrics)
    assert tr._cycle_log_line(7, metrics) == old
    metrics.update(gnorm=[0.52, 1.25, 0.0, 30.5, 2.0], coh=[0.181, 0.2, 0.0, 0.05, 0.049],
                   lr=[1e-4, 1e-4, 0.0, 1e-5, 1e-5])
    assert tr._cycle_log_line(7, metrics) == (
        old + " gnorm=[0.52,1.25,30.5,2] coh=[0.181,0.200,0.050,0.049] lr=[1.00e-04,1.00e-04,1.00e-05,1.00e-05]"
    )
    metrics = dict(metrics, gnorm=[0.52, 1.25, 0.0], lr=[1e-4, 2e-4, 0.0])
    del metrics["coh"]
    assert tr._cycle_log_line(7, metrics) == old + " gnorm=[0.52,1.25] lr=[1.00e-04,2.00e-04]"
