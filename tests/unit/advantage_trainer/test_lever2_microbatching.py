"""Lever 2 parity tests: micro-batched _train_step gives the same gradients
and loss as the per-sample (inner_batch_size=1) path.

Three tests:

  (1) Vectorized batched loss equals per-sample loss aggregated across K.
      Pure math check on value_regression_loss_fn_batched vs
      value_regression_loss_fn.
  (2) prepare_batched_advantage_samples produces the same per-row tensors
      as prepare_advantage_sample (just padded).
  (3) Gradient equivalence: _train_step(..., inner_batch_size=K) vs
      _train_step(..., inner_batch_size=1) produces matching LoRA gradients.
"""

from __future__ import annotations

import copy

import pytest
import torch
from torch import optim
from transformers import Qwen2Config, Qwen2Model

from prime_rl.advantage_trainer.data import (
    prepare_advantage_sample,
    prepare_batched_advantage_samples,
)
from prime_rl.advantage_trainer.forward import value_forward
from prime_rl.advantage_trainer.loss import (
    BatchedValueLossInputs,
    ValueLossInputs,
    value_regression_loss_fn,
    value_regression_loss_fn_batched,
)
from prime_rl.advantage_trainer.train import _train_step
from prime_rl.configs.trainer import LoRAConfig
from prime_rl.orchestrator.value_networks import (
    Q_PLUS_SLOT,
    ValueNetworkBackbone,
)
from prime_rl.transport.types import (
    AdvantageTrainingBatch,
    AdvantageTrainingSample,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _tiny_backbone() -> ValueNetworkBackbone:
    """Same tiny backbone as test_routing_polyak_step."""
    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
    )
    backbone = ValueNetworkBackbone(
        Qwen2Model(config),
        lora_config=LoRAConfig(rank=8, alpha=16.0, dropout=0.0),
        polyak_tau=0.005,
    )
    backbone.train()
    with torch.no_grad():
        for name, p in backbone.named_parameters():
            if ("lora_A" in name) or ("lora_B" in name):
                p.copy_(torch.randn_like(p) * 0.1)
        for head in (backbone.v_head, backbone.q_plus_head, backbone.v_target_head):
            torch.nn.init.normal_(head.linear.weight, mean=0.0, std=0.02)
    return backbone


def _make_sample(prompt_len: int, completion_len: int, seed: int) -> AdvantageTrainingSample:
    """Synthetic AdvantageTrainingSample with deterministic targets + mask."""
    rng = torch.Generator().manual_seed(seed)
    prompt_ids = torch.randint(1, 200, (prompt_len,), generator=rng).tolist()
    completion_ids = torch.randint(1, 200, (completion_len,), generator=rng).tolist()
    # Loss mask: True for ~80% of completion positions.
    mask = (torch.rand(completion_len, generator=rng) < 0.8).tolist()
    v_targets = (torch.randn(completion_len, generator=rng) * 0.3).tolist()
    q_targets = (torch.randn(completion_len, generator=rng) * 0.3).tolist()
    return AdvantageTrainingSample(
        prompt_ids=prompt_ids,
        prompt_mask=[False] * prompt_len,
        completion_ids=completion_ids,
        completion_mask=mask,
        v_targets=v_targets,
        q_plus_targets=q_targets,
    )


# ---------------------------------------------------------------------------
# 1. Batched loss matches per-sample loss aggregated across K
# ---------------------------------------------------------------------------


def test_batched_loss_matches_per_sample_mean():
    """value_regression_loss_fn_batched returns the per-sample mean (mean over
    samples of mean-over-valid-positions). Compare against the equivalent
    Python-loop computation using value_regression_loss_fn."""
    torch.manual_seed(42)
    K, S = 4, 10

    v_pred = torch.randn(K, S)
    v_tgt = torch.randn(K, S)
    q_pred = torch.randn(K, S)
    q_tgt = torch.randn(K, S)
    mask = torch.rand(K, S) < 0.7  # bool [K, S]
    # Make sure every row has at least one valid position
    mask[:, 0] = True

    # Batched path
    batched_out = value_regression_loss_fn_batched(
        BatchedValueLossInputs(
            v_predictions=v_pred,
            v_targets=v_tgt,
            q_plus_predictions=q_pred,
            q_plus_targets=q_tgt,
            loss_mask=mask,
            algorithm="arm",
        )
    )

    # Per-sample loop reference
    per_sample_losses = []
    per_sample_l_vs = []
    per_sample_l_qs = []
    for k in range(K):
        out = value_regression_loss_fn(
            ValueLossInputs(
                v_predictions=v_pred[k],
                v_targets=v_tgt[k],
                q_plus_predictions=q_pred[k],
                q_plus_targets=q_tgt[k],
                loss_mask=mask[k],
                algorithm="arm",
            )
        )
        per_sample_losses.append(out.loss)
        per_sample_l_vs.append(out.metrics["l_v"])
        per_sample_l_qs.append(out.metrics["l_q"])

    ref_loss = torch.stack(per_sample_losses).mean()
    ref_l_v = torch.stack(per_sample_l_vs).mean()
    ref_l_q = torch.stack(per_sample_l_qs).mean()

    assert torch.allclose(batched_out.loss, ref_loss, atol=1e-6), (
        f"batched loss {batched_out.loss.item()} != per-sample mean {ref_loss.item()}"
    )
    assert torch.allclose(batched_out.metrics["l_v"], ref_l_v, atol=1e-6)
    assert torch.allclose(batched_out.metrics["l_q"], ref_l_q, atol=1e-6)


# ---------------------------------------------------------------------------
# 2. Batched adapter row-wise matches per-sample adapter
# ---------------------------------------------------------------------------


def test_batched_adapter_matches_per_sample():
    """prepare_batched_advantage_samples produces the same per-row content
    as prepare_advantage_sample, just right-padded to chunk-max-S."""
    samples = [
        _make_sample(prompt_len=4, completion_len=6, seed=1),
        _make_sample(prompt_len=3, completion_len=8, seed=2),
        _make_sample(prompt_len=5, completion_len=5, seed=3),
    ]

    batched = prepare_batched_advantage_samples(samples)
    K = len(samples)
    assert batched.input_ids.shape[0] == K
    S_max = batched.input_ids.shape[1]

    for k, s in enumerate(samples):
        ps = prepare_advantage_sample(s)
        L_k = batched.lens[k].item()
        # Valid portion matches
        assert torch.equal(batched.input_ids[k, :L_k], ps.input_ids[0, :L_k])
        assert torch.allclose(batched.v_targets[k, :L_k], ps.v_targets[:L_k])
        assert batched.q_plus_targets is not None and ps.q_plus_targets is not None
        assert torch.allclose(batched.q_plus_targets[k, :L_k], ps.q_plus_targets[:L_k])
        assert torch.equal(batched.loss_mask[k, :L_k], ps.loss_mask[:L_k])
        # Padded portion is zeros / False
        if S_max > L_k:
            assert (batched.input_ids[k, L_k:] == 0).all()
            assert (batched.v_targets[k, L_k:] == 0.0).all()
            assert (batched.q_plus_targets[k, L_k:] == 0.0).all()
            assert not batched.loss_mask[k, L_k:].any()


# ---------------------------------------------------------------------------
# 3. Gradient equivalence: batched _train_step == per-sample _train_step
# ---------------------------------------------------------------------------


def _grad_snapshot(backbone: ValueNetworkBackbone) -> dict[str, torch.Tensor]:
    return {
        name: p.grad.detach().clone()
        for name, p in backbone.named_parameters()
        if p.grad is not None
    }


def test_batched_train_step_gradients_match_per_sample():
    """Running _train_step at inner_batch_size=K should produce LoRA
    gradients that match (within tight tolerance) running it at
    inner_batch_size=1. Both branches should converge to mean-of-per-sample
    semantics on the same batch.

    This pins the memory-chunking invariant: at fixed (n_epochs=1,
    minibatch_size=batch_size), inner_batch_size only controls how the
    minibatch's gradient is computed (one chunk vs. K accumulated chunks)
    -- it must not change the final gradient up to float ordering.
    """
    samples = [
        _make_sample(prompt_len=4, completion_len=6, seed=10),
        _make_sample(prompt_len=5, completion_len=4, seed=11),
        _make_sample(prompt_len=3, completion_len=7, seed=12),
        _make_sample(prompt_len=4, completion_len=5, seed=13),
    ]
    batch = AdvantageTrainingBatch(examples=samples, step=0)

    # Per-sample path
    backbone_a = _tiny_backbone()
    optimizer_a = optim.SGD(
        [p for p in backbone_a.parameters() if p.requires_grad], lr=0.0  # lr=0: we only inspect grads
    )
    # Run one step; the LR=0 SGD will compute grads but not move params.
    # Pin n_epochs=1, minibatch_size=None (full batch) so this remains a
    # pure memory-chunking parity test -- not a Jin-loop test.
    metrics_a = _train_step(
        backbone=backbone_a,
        optimizer=optimizer_a,
        batch=batch,
        algorithm="arm",
        polyak_tau=0.0,  # disable polyak so V_target doesn't drift
        n_epochs=1,
        minibatch_size=None,
        inner_batch_size=1,
    )
    grads_a = _grad_snapshot(backbone_a)

    # Batched path: same backbone init via same seed
    backbone_b = _tiny_backbone()
    optimizer_b = optim.SGD(
        [p for p in backbone_b.parameters() if p.requires_grad], lr=0.0
    )
    metrics_b = _train_step(
        backbone=backbone_b,
        optimizer=optimizer_b,
        batch=batch,
        algorithm="arm",
        polyak_tau=0.0,
        n_epochs=1,
        minibatch_size=None,
        inner_batch_size=4,  # one chunk of 4 samples
    )
    grads_b = _grad_snapshot(backbone_b)

    # Loss scalars match (within tolerance for floating-point ordering).
    assert abs(metrics_a["loss"] - metrics_b["loss"]) < 1e-5, (
        f"loss diverged: per-sample={metrics_a['loss']:.6e} batched={metrics_b['loss']:.6e}"
    )
    assert abs(metrics_a["l_v"] - metrics_b["l_v"]) < 1e-5
    assert abs(metrics_a["l_q"] - metrics_b["l_q"]) < 1e-5

    # LoRA gradients match between paths (the structural correctness gate).
    assert set(grads_a.keys()) == set(grads_b.keys())
    grad_tol = 5e-5
    for name in grads_a:
        ga, gb = grads_a[name], grads_b[name]
        if "lora" not in name:
            continue
        max_abs = (ga - gb).abs().max().item()
        assert max_abs < grad_tol, (
            f"gradient diverged for {name}: max_abs={max_abs:.3e} (tol {grad_tol})"
        )


def test_batched_train_step_handles_fractional_last_chunk():
    """inner_batch_size doesn't evenly divide n_samples: last chunk has K<inner_batch_size.
    Per-sample-mean weighting must still be correct."""
    # 5 samples with inner_batch_size=2 -> chunks of sizes [2, 2, 1]
    samples = [_make_sample(prompt_len=3, completion_len=5, seed=20 + i) for i in range(5)]
    batch = AdvantageTrainingBatch(examples=samples, step=0)

    backbone_a = _tiny_backbone()
    optimizer_a = optim.SGD([p for p in backbone_a.parameters() if p.requires_grad], lr=0.0)
    metrics_a = _train_step(
        backbone=backbone_a, optimizer=optimizer_a, batch=batch,
        algorithm="arm", polyak_tau=0.0,
        n_epochs=1, minibatch_size=None, inner_batch_size=1,
    )
    grads_a = _grad_snapshot(backbone_a)

    backbone_b = _tiny_backbone()
    optimizer_b = optim.SGD([p for p in backbone_b.parameters() if p.requires_grad], lr=0.0)
    metrics_b = _train_step(
        backbone=backbone_b, optimizer=optimizer_b, batch=batch,
        algorithm="arm", polyak_tau=0.0,
        n_epochs=1, minibatch_size=None, inner_batch_size=2,
    )
    grads_b = _grad_snapshot(backbone_b)

    assert abs(metrics_a["loss"] - metrics_b["loss"]) < 1e-5
    grad_tol = 5e-5
    for name in grads_a:
        if "lora" not in name:
            continue
        max_abs = (grads_a[name] - grads_b[name]).abs().max().item()
        assert max_abs < grad_tol, f"{name}: {max_abs:.3e}"
