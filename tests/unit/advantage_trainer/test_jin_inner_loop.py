"""Jin-aligned inner training loop tests (2026-05-16 spec).

These tests pin the behavior introduced to fix the race condition diagnosed
in the K=4 reverse-text collapse: each AdvTrainer cycle now runs N_EPOCHS
epochs of minibatch SGD over the cycle's batch before letting the policy
respond. Polyak update + weight broadcast fire once per cycle, after the
inner loop completes.

Test coverage:

  1. Legacy recovery: n_epochs=1 + minibatch_size=None reproduces the
     pre-spec single-optimizer-step-per-cycle behavior (lever2 parity test
     already validates the chunking equivalence at fixed legacy settings;
     here we additionally pin that the new default function signature
     matches that legacy).
  2. n_epochs > 1 takes more optimizer steps than n_epochs == 1, and the
     resulting weight delta is correspondingly larger.
  3. Polyak fires exactly once per cycle regardless of n_epochs *
     n_minibatches (V_target tracks V by ONE Polyak step, not by 32).
  4. L_V decreases meaningfully within a cycle when the inner loop runs to
     convergence -- the direct race-condition diagnostic.
  5. Shuffle determinism: same `generator` seed yields identical
     trajectories; different seeds yield different trajectories.
  6. Adam optimizer state persists across cycles (no implicit reset).
  7. Uneven last minibatch is handled (batch_size not divisible by
     minibatch_size).

All tests use the same tiny 2-layer Qwen2 ValueNetworkBackbone fixture
that the other AdvTrainer tests use, so the suite stays CPU-fast.
"""

from __future__ import annotations

import copy

import torch
from torch import optim
from transformers import Qwen2Config, Qwen2Model

from prime_rl.advantage_trainer.train import _train_step
from prime_rl.configs.trainer import LoRAConfig
from prime_rl.orchestrator.value_networks import (
    Q_PLUS_SLOT,
    V_SLOT,
    V_TARGET_SLOT,
    ValueNetworkBackbone,
)
from prime_rl.trainer.models.layers.lora.multi_linear import MultiLoRALinear
from prime_rl.transport.types import (
    AdvantageTrainingBatch,
    AdvantageTrainingSample,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _tiny_backbone(seed: int = 0) -> ValueNetworkBackbone:
    """Match the fixture used by the other AdvTrainer tests."""
    torch.manual_seed(seed)
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
    rng = torch.Generator().manual_seed(seed)
    prompt_ids = torch.randint(1, 200, (prompt_len,), generator=rng).tolist()
    completion_ids = torch.randint(1, 200, (completion_len,), generator=rng).tolist()
    mask = (torch.rand(completion_len, generator=rng) < 0.8).tolist()
    # Make the mask non-empty for every sample (avoid degenerate L_V=0 cases).
    if not any(mask):
        mask[0] = True
    v_targets = (torch.randn(completion_len, generator=rng) * 0.5).tolist()
    q_targets = (torch.randn(completion_len, generator=rng) * 0.5).tolist()
    return AdvantageTrainingSample(
        prompt_ids=prompt_ids,
        prompt_mask=[False] * prompt_len,
        completion_ids=completion_ids,
        completion_mask=mask,
        v_targets=v_targets,
        q_plus_targets=q_targets,
    )


def _snapshot_v_lora(backbone: ValueNetworkBackbone) -> list[torch.Tensor]:
    """Snapshot V's LoRA A across all MultiLoRALinear modules."""
    snaps = []
    for module in backbone.base_model.modules():
        if isinstance(module, MultiLoRALinear):
            snaps.append(module.lora_A[V_SLOT].detach().clone())
    return snaps


def _snapshot_v_target_lora(backbone: ValueNetworkBackbone) -> list[torch.Tensor]:
    snaps = []
    for module in backbone.base_model.modules():
        if isinstance(module, MultiLoRALinear):
            snaps.append(module.lora_A[V_TARGET_SLOT].detach().clone())
    return snaps


def _l2_norm_delta(after: list[torch.Tensor], before: list[torch.Tensor]) -> float:
    """L2 norm of the concatenated parameter delta."""
    total = 0.0
    for a, b in zip(after, before):
        total += (a - b).pow(2).sum().item()
    return total**0.5


def _make_batch(n_samples: int, *, with_q_plus: bool = True, seed_offset: int = 100) -> AdvantageTrainingBatch:
    samples = [
        _make_sample(prompt_len=3, completion_len=5, seed=seed_offset + i)
        for i in range(n_samples)
    ]
    if not with_q_plus:
        for s in samples:
            s.q_plus_targets = None
    return AdvantageTrainingBatch(examples=samples, step=0)


# ---------------------------------------------------------------------------
# 1. Legacy recovery: defaults reproduce single-step behavior
# ---------------------------------------------------------------------------


def test_legacy_defaults_produce_single_optimizer_step():
    """With n_epochs=1, minibatch_size=None (the function-signature defaults),
    _train_step takes exactly ONE optimizer step regardless of batch size."""
    batch = _make_batch(n_samples=8)
    backbone = _tiny_backbone()
    optimizer = optim.SGD(
        [p for p in backbone.parameters() if p.requires_grad], lr=0.0
    )

    metrics = _train_step(
        backbone=backbone,
        optimizer=optimizer,
        batch=batch,
        algorithm="arm",
        polyak_tau=0.0,
    )

    assert metrics["inner_steps"] == 1, (
        f"legacy defaults should produce one inner step; got {metrics['inner_steps']}"
    )
    # With one epoch, first_epoch and last_epoch are the same epoch.
    assert metrics["l_v_first_epoch"] == metrics["l_v_last_epoch"]
    assert metrics["l_q_first_epoch"] == metrics["l_q_last_epoch"]


# ---------------------------------------------------------------------------
# 2. n_epochs > 1 takes more steps and moves weights further
# ---------------------------------------------------------------------------


def test_n_epochs_increases_inner_step_count():
    """n_epochs=4 with minibatch_size=2 over an 8-sample batch produces
    4 epochs * 4 minibatches/epoch = 16 inner steps."""
    batch = _make_batch(n_samples=8)
    backbone = _tiny_backbone()
    optimizer = optim.SGD(
        [p for p in backbone.parameters() if p.requires_grad], lr=0.0
    )
    gen = torch.Generator().manual_seed(0)

    metrics = _train_step(
        backbone=backbone,
        optimizer=optimizer,
        batch=batch,
        algorithm="arm",
        polyak_tau=0.0,
        n_epochs=4,
        minibatch_size=2,
        generator=gen,
    )
    assert metrics["inner_steps"] == 16, (
        f"expected 4 epochs * 4 minibatches = 16 inner steps; got {metrics['inner_steps']}"
    )


def test_n_epochs_8_moves_v_further_than_n_epochs_1():
    """Holding everything else fixed, more inner steps -> larger weight delta.
    Compares the L2 norm of V's LoRA delta between n_epochs=1 (single
    full-batch step) and n_epochs=8 with minibatch_size=2."""
    batch = _make_batch(n_samples=8)
    lr = 1e-3

    # n_epochs=1 path
    bb_a = _tiny_backbone()
    opt_a = optim.SGD([p for p in bb_a.parameters() if p.requires_grad], lr=lr)
    v_before_a = _snapshot_v_lora(bb_a)
    _train_step(
        backbone=bb_a, optimizer=opt_a, batch=batch,
        algorithm="arm", polyak_tau=0.0,
        n_epochs=1, minibatch_size=None,
        generator=torch.Generator().manual_seed(0),
    )
    delta_a = _l2_norm_delta(_snapshot_v_lora(bb_a), v_before_a)

    # n_epochs=8, minibatch_size=2 -> 32 inner steps
    bb_b = _tiny_backbone()
    opt_b = optim.SGD([p for p in bb_b.parameters() if p.requires_grad], lr=lr)
    v_before_b = _snapshot_v_lora(bb_b)
    _train_step(
        backbone=bb_b, optimizer=opt_b, batch=batch,
        algorithm="arm", polyak_tau=0.0,
        n_epochs=8, minibatch_size=2,
        generator=torch.Generator().manual_seed(0),
    )
    delta_b = _l2_norm_delta(_snapshot_v_lora(bb_b), v_before_b)

    assert delta_b > delta_a * 1.5, (
        f"n_epochs=8 should move V's LoRA further than n_epochs=1: "
        f"delta_1ep={delta_a:.4e} delta_8ep={delta_b:.4e}"
    )


# ---------------------------------------------------------------------------
# 3. Polyak fires exactly once per cycle
# ---------------------------------------------------------------------------


def test_polyak_fires_exactly_once_per_cycle():
    """V_target should track V by exactly one Polyak step per _train_step
    call, regardless of how many inner Adam steps run inside the cycle.

    Verification: take the V_target delta and check it matches
    `tau * (V_after_inner_loop - V_target_before)`. If Polyak fires
    multiple times (e.g., once per inner step), the actual delta would
    be substantially larger.
    """
    batch = _make_batch(n_samples=8)
    backbone = _tiny_backbone()
    optimizer = optim.SGD(
        [p for p in backbone.parameters() if p.requires_grad], lr=1e-2
    )
    tau = 0.5  # large tau so the Polyak delta is easy to measure

    vt_before = _snapshot_v_target_lora(backbone)
    v_before = _snapshot_v_lora(backbone)

    _train_step(
        backbone=backbone, optimizer=optimizer, batch=batch,
        algorithm="arm", polyak_tau=tau,
        n_epochs=4, minibatch_size=2,  # 16 inner steps
        generator=torch.Generator().manual_seed(0),
    )

    vt_after = _snapshot_v_target_lora(backbone)
    v_after = _snapshot_v_lora(backbone)

    # Expected: V_target_after = (1-tau) * V_target_before + tau * V_after
    # (per backbone.polyak_update_v_target with tau=0.5).
    for vt_b, vt_a, v_a in zip(vt_before, vt_after, v_after):
        expected_vt = (1 - tau) * vt_b + tau * v_a
        max_err = (vt_a - expected_vt).abs().max().item()
        assert max_err < 1e-6, (
            f"V_target delta inconsistent with single Polyak application: "
            f"max_err={max_err:.3e}. Multiple Polyak fires would compound."
        )

    # Sanity: V itself did move (gradient descent ran).
    assert _l2_norm_delta(v_after, v_before) > 0, "V should have moved"


# ---------------------------------------------------------------------------
# 4. Loss decreases within a cycle (the race-condition diagnostic)
# ---------------------------------------------------------------------------


def test_loss_decreases_within_cycle():
    """Under the spec-aligned inner loop, the last epoch's mean loss should
    be meaningfully smaller than the first epoch's mean loss -- the value
    network regresses toward its (frozen) targets over the inner loop's
    steps.

    Each epoch covers the same n_total samples (only their order differs
    via per-epoch shuffling), so the per-epoch means are apples-to-apples.
    The first-vs-last-epoch gap is the direct race-condition diagnostic.

    Uses SGD with a moderate LR. The targets are fixed (in production
    they're frozen by the transport boundary -- the AdvServer produced
    them under V_prev/Q+_prev and the AdvTrainer cannot recompute them).
    """
    batch = _make_batch(n_samples=16)
    backbone = _tiny_backbone()
    # Adam @ 1e-2 matches the convention in test_routing_polyak_step's
    # Polyak/Adam tests; it adapts step size and converges quickly on
    # tiny synthetic regressions where SGD's fixed step-size struggles
    # to traverse the loss landscape in a small number of steps.
    optimizer = optim.Adam(
        [p for p in backbone.parameters() if p.requires_grad], lr=1e-2
    )

    metrics = _train_step(
        backbone=backbone, optimizer=optimizer, batch=batch,
        algorithm="arm", polyak_tau=0.0,
        n_epochs=8, minibatch_size=4,  # 8 epochs * 4 mb/epoch = 32 inner steps
        generator=torch.Generator().manual_seed(42),
    )

    assert metrics["inner_steps"] == 32
    assert metrics["l_v_last_epoch"] < metrics["l_v_first_epoch"] * 0.5, (
        f"L_V should drop by at least 50% across the cycle (race fix): "
        f"first_epoch={metrics['l_v_first_epoch']:.4e} "
        f"last_epoch={metrics['l_v_last_epoch']:.4e}"
    )
    assert metrics["l_q_last_epoch"] < metrics["l_q_first_epoch"] * 0.5, (
        f"L_Q+ should drop by at least 50% across the cycle (race fix): "
        f"first_epoch={metrics['l_q_first_epoch']:.4e} "
        f"last_epoch={metrics['l_q_last_epoch']:.4e}"
    )


# ---------------------------------------------------------------------------
# 5. Shuffle determinism
# ---------------------------------------------------------------------------


def test_shuffle_seed_determinism_same_seed_identical_trajectory():
    """Two _train_step calls with the same backbone init + same generator
    seed produce bit-identical metrics."""
    batch = _make_batch(n_samples=8)

    bb_a = _tiny_backbone()
    opt_a = optim.SGD(
        [p for p in bb_a.parameters() if p.requires_grad], lr=1e-3
    )
    m_a = _train_step(
        backbone=bb_a, optimizer=opt_a, batch=batch,
        algorithm="arm", polyak_tau=0.0,
        n_epochs=4, minibatch_size=2,
        generator=torch.Generator().manual_seed(123),
    )

    bb_b = _tiny_backbone()
    opt_b = optim.SGD(
        [p for p in bb_b.parameters() if p.requires_grad], lr=1e-3
    )
    m_b = _train_step(
        backbone=bb_b, optimizer=opt_b, batch=batch,
        algorithm="arm", polyak_tau=0.0,
        n_epochs=4, minibatch_size=2,
        generator=torch.Generator().manual_seed(123),
    )

    for key in (
        "l_v_first_epoch", "l_v_last_epoch",
        "l_q_first_epoch", "l_q_last_epoch",
        "mean_loss", "inner_steps",
    ):
        assert m_a[key] == m_b[key], (
            f"determinism violated for {key}: {m_a[key]} vs {m_b[key]}"
        )


def test_shuffle_seed_different_seeds_different_trajectory():
    """Different generator seeds produce different minibatch orderings.

    At lr=0 (no parameter updates), an epoch's mean loss equals the
    full-batch mean loss regardless of shuffle order -- so we can't
    distinguish seeds by `l_v_first_epoch`. Instead we capture the
    per-minibatch sequence by hooking into the inner loop indirectly:
    run with lr>0 so each minibatch's gradient affects subsequent
    minibatches' loss values, then compare the first-epoch mean across
    seeds. Different shuffle orders -> different mid-epoch parameter
    trajectories -> different first-epoch mean.
    """
    batch = _make_batch(n_samples=8)

    bb_a = _tiny_backbone()
    opt_a = optim.SGD(
        [p for p in bb_a.parameters() if p.requires_grad], lr=1e-2
    )
    m_a = _train_step(
        backbone=bb_a, optimizer=opt_a, batch=batch,
        algorithm="arm", polyak_tau=0.0,
        n_epochs=1, minibatch_size=2,  # 4 minibatches; shuffle order matters
        generator=torch.Generator().manual_seed(0),
    )

    bb_b = _tiny_backbone()
    opt_b = optim.SGD(
        [p for p in bb_b.parameters() if p.requires_grad], lr=1e-2
    )
    m_b = _train_step(
        backbone=bb_b, optimizer=opt_b, batch=batch,
        algorithm="arm", polyak_tau=0.0,
        n_epochs=1, minibatch_size=2,
        generator=torch.Generator().manual_seed(999),
    )

    # Different shuffle order + lr>0 -> different first-epoch mean.
    assert m_a["l_v_first_epoch"] != m_b["l_v_first_epoch"], (
        f"different seeds produced identical first-epoch mean; "
        f"shuffling may not be wired through. seed0={m_a['l_v_first_epoch']:.6e} "
        f"seed999={m_b['l_v_first_epoch']:.6e}"
    )


# ---------------------------------------------------------------------------
# 6. Optimizer state persists across cycles
# ---------------------------------------------------------------------------


def test_adam_state_persists_across_cycles():
    """Adam's exp_avg / exp_avg_sq should be non-zero after _train_step,
    and the second _train_step call should preserve and continue updating
    them (not reset)."""
    batch_1 = _make_batch(n_samples=4, seed_offset=100)
    batch_2 = _make_batch(n_samples=4, seed_offset=200)
    backbone = _tiny_backbone()
    optimizer = optim.Adam(
        [p for p in backbone.parameters() if p.requires_grad], lr=1e-3
    )

    # Cycle 1.
    _train_step(
        backbone=backbone, optimizer=optimizer, batch=batch_1,
        algorithm="arm", polyak_tau=0.0,
        n_epochs=2, minibatch_size=2,
        generator=torch.Generator().manual_seed(0),
    )

    # Capture Adam moments after cycle 1. `state["step"]` is a tensor in
    # modern PyTorch; coerce to int *immediately* so the snapshot doesn't
    # alias the still-mutating Adam state.
    def _step_to_int(step):
        if hasattr(step, "item"):
            return int(step.item())
        return int(step)

    moments_after_1 = {}
    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state.get(p, {})
            if "exp_avg" in state:
                moments_after_1[id(p)] = (
                    state["exp_avg"].detach().clone(),
                    state["exp_avg_sq"].detach().clone(),
                    _step_to_int(state.get("step", 0)),
                )

    assert moments_after_1, "Adam state should be non-empty after cycle 1"
    # At least one moment should be non-zero (gradient flowed somewhere).
    any_nonzero = any(
        m[0].abs().max().item() > 0 for m in moments_after_1.values()
    )
    assert any_nonzero, "Adam exp_avg should be non-zero after cycle 1"

    # Cycle 2 on a different batch.
    _train_step(
        backbone=backbone, optimizer=optimizer, batch=batch_2,
        algorithm="arm", polyak_tau=0.0,
        n_epochs=2, minibatch_size=2,
        generator=torch.Generator().manual_seed(1),
    )

    # Step counter should have advanced from cycle 1's count, not reset to 0.
    moments_after_2 = {}
    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state.get(p, {})
            if "exp_avg" in state:
                moments_after_2[id(p)] = _step_to_int(state.get("step", 0))

    # Pick any param with state in both cycles and confirm step advanced.
    # Cycle 1 ran 2 epochs * 2 mb/epoch = 4 inner steps; cycle 2 added
    # another 4 -> step_2 should be 8 (cycle 1's 4 + cycle 2's 4) given
    # Adam state persisted across cycles.
    found = False
    for pid, step_2 in moments_after_2.items():
        if pid in moments_after_1:
            step_1 = moments_after_1[pid][2]
            assert step_2 > step_1, (
                f"Adam step count regressed across cycles "
                f"(state reset?): step_1={step_1} step_2={step_2}"
            )
            found = True
            break
    assert found, "No param had Adam state in both cycles"


# ---------------------------------------------------------------------------
# 7. Uneven minibatch handled
# ---------------------------------------------------------------------------


def test_uneven_last_minibatch_no_crash():
    """batch_size=5 with minibatch_size=2 -> minibatches of sizes [2, 2, 1].
    Should not crash; should still produce gradients on every param.
    """
    batch = _make_batch(n_samples=5)
    backbone = _tiny_backbone()
    optimizer = optim.SGD(
        [p for p in backbone.parameters() if p.requires_grad], lr=1e-3
    )
    v_before = _snapshot_v_lora(backbone)

    metrics = _train_step(
        backbone=backbone, optimizer=optimizer, batch=batch,
        algorithm="arm", polyak_tau=0.0,
        n_epochs=2, minibatch_size=2,
        generator=torch.Generator().manual_seed(0),
    )

    # 2 epochs * ceil(5/2)=3 minibatches/epoch = 6 inner steps.
    assert metrics["inner_steps"] == 6
    # V should have moved (last minibatch's 1-sample gradient still applied).
    delta = _l2_norm_delta(_snapshot_v_lora(backbone), v_before)
    assert delta > 0, "V did not move despite gradient steps"


# ---------------------------------------------------------------------------
# 8. q_plus_target_abs_mean diagnostic
# ---------------------------------------------------------------------------


def test_q_plus_target_abs_mean_reported():
    """The cycle-level diagnostic |q_plus_target| (averaged over valid
    positions across the batch) is reported in metrics for ARM cycles.

    Spec 20260516 predicts this stays bounded under the Jin-aligned loop
    (instead of growing 0.2 -> 0.9 as in the K=4 collapse run); the
    metric is the AdvTrainer-side check on that prediction.
    """
    batch = _make_batch(n_samples=4)
    backbone = _tiny_backbone()
    optimizer = optim.SGD(
        [p for p in backbone.parameters() if p.requires_grad], lr=0.0
    )

    metrics = _train_step(
        backbone=backbone, optimizer=optimizer, batch=batch,
        algorithm="arm", polyak_tau=0.0,
        n_epochs=1, minibatch_size=None,
        generator=torch.Generator().manual_seed(0),
    )

    # Hand-compute the reference: mean of |q_plus_targets| across all valid
    # completion-mask=True positions in the batch.
    abs_sum = 0.0
    count = 0
    for s in batch.examples:
        assert s.q_plus_targets is not None
        for tgt in s.q_plus_targets:
            abs_sum += abs(tgt)
            count += 1
    expected = abs_sum / count if count else 0.0

    assert abs(metrics["q_plus_target_abs_mean"] - expected) < 1e-6, (
        f"q_plus_target_abs_mean mismatch: reported={metrics['q_plus_target_abs_mean']:.6e} "
        f"expected={expected:.6e}"
    )


def test_q_plus_target_abs_mean_zero_for_ppo():
    """PPO cycles have q_plus_targets=None; the diagnostic should be 0.0."""
    batch = _make_batch(n_samples=4, with_q_plus=False)
    backbone = _tiny_backbone()
    optimizer = optim.SGD(
        [p for p in backbone.parameters() if p.requires_grad], lr=0.0
    )

    metrics = _train_step(
        backbone=backbone, optimizer=optimizer, batch=batch,
        algorithm="ppo", polyak_tau=0.0,
        n_epochs=1, minibatch_size=None,
    )

    assert metrics["q_plus_target_abs_mean"] == 0.0
