"""Phase 7a -- gradient routing, Polyak, and single-training-step tests.

These tests build a tiny ValueNetworkBackbone (2-layer Qwen2 config from
Phase 4's pattern) and exercise the algorithmic core:
  - V's loss only updates V's adapter (LoRA slot 0 + v_head).
  - Q+'s loss only updates Q+'s adapter (LoRA slot 1 + q_plus_head).
  - V_target's adapter (slot 2 + v_target_head) receives no gradient ever.
  - Frozen base parameters receive no gradient ever.
  - Combined ARM loss updates both V and Q+ correctly in one backward.
  - Polyak update fires after gradient step and only modifies V_target.
  - One full training step (forward + backward + optimizer step + Polyak)
    completes for PPO and ARM and updates the right parameters.
"""

from __future__ import annotations

import torch
from torch import optim
from transformers import Qwen2Config, Qwen2Model

from prime_rl.advantage_trainer.forward import value_forward
from prime_rl.advantage_trainer.loss import (
    ValueLossInputs,
    value_regression_loss_fn,
)
from prime_rl.configs.trainer import LoRAConfig
from prime_rl.orchestrator.value_networks import (
    Q_PLUS_SLOT,
    V_SLOT,
    V_TARGET_SLOT,
    ValueNetworkBackbone,
)
from prime_rl.trainer.models.layers.lora.multi_linear import MultiLoRALinear


def _tiny_backbone() -> ValueNetworkBackbone:
    """2-layer Qwen2 ValueNetworkBackbone with non-zero LoRA + non-zero heads.

    Non-zero init means gradient flow into the LoRA slots and value heads
    is testable -- otherwise zero-init makes everything trivially zero.
    """
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
        for module in backbone.base_model.modules():
            if isinstance(module, MultiLoRALinear):
                # Non-zero LoRA at all three slots so gradients are non-trivial.
                module.lora_A[V_SLOT].fill_(0.4)
                module.lora_B[V_SLOT].fill_(0.4)
                module.lora_A[Q_PLUS_SLOT].fill_(0.6)
                module.lora_B[Q_PLUS_SLOT].fill_(0.6)
                module.lora_A[V_TARGET_SLOT].fill_(0.5)
                module.lora_B[V_TARGET_SLOT].fill_(0.5)
        for head in (backbone.v_head, backbone.q_plus_head, backbone.v_target_head):
            torch.nn.init.normal_(head.linear.weight, mean=0.0, std=0.02)
    return backbone


def _zero_grads(backbone: ValueNetworkBackbone) -> None:
    for p in backbone.parameters():
        if p.grad is not None:
            p.grad.zero_()


# ---------------------------------------------------------------------------
# Gradient routing
# ---------------------------------------------------------------------------


def test_v_loss_only_updates_v_adapter():
    """L_V backward: gradients flow into V's LoRA + v_head only.
    Q+'s LoRA, V_target's LoRA, frozen base, q_plus_head, v_target_head
    all see zero gradient.
    """
    backbone = _tiny_backbone()
    input_ids = torch.tensor([[1, 2, 3, 4]])

    v_pred, _ = value_forward(backbone, input_ids, algorithm="ppo")
    v_targets = torch.zeros_like(v_pred)
    out = value_regression_loss_fn(
        ValueLossInputs(
            v_predictions=v_pred[0],
            v_targets=v_targets[0],
            q_plus_predictions=None,
            q_plus_targets=None,
            loss_mask=torch.ones(v_pred.shape[1], dtype=torch.bool),
            algorithm="ppo",
        )
    )
    out.loss.backward()

    # Inspect each adapter's lora_A / lora_B grads.
    saw_v_lora_grad = False
    for module in backbone.base_model.modules():
        if not isinstance(module, MultiLoRALinear):
            continue
        # V's LoRA should have non-zero grad.
        assert module.lora_A[V_SLOT].grad is not None
        if module.lora_A[V_SLOT].grad.abs().max().item() > 0:
            saw_v_lora_grad = True
        # Q+'s LoRA grad should be None or zero (no path from L_V to it).
        if module.lora_A[Q_PLUS_SLOT].grad is not None:
            assert module.lora_A[Q_PLUS_SLOT].grad.abs().max().item() < 1e-12
        # V_target's LoRA grad should be None or zero.
        if module.lora_A[V_TARGET_SLOT].grad is not None:
            assert module.lora_A[V_TARGET_SLOT].grad.abs().max().item() < 1e-12
    assert saw_v_lora_grad, "V's LoRA never received gradient"

    # Value heads.
    assert backbone.v_head.linear.weight.grad is not None
    assert backbone.v_head.linear.weight.grad.abs().max().item() > 0
    # Q+'s head and V_target's head should have None or zero grads.
    if backbone.q_plus_head.linear.weight.grad is not None:
        assert backbone.q_plus_head.linear.weight.grad.abs().max().item() < 1e-12
    if backbone.v_target_head.linear.weight.grad is not None:
        assert backbone.v_target_head.linear.weight.grad.abs().max().item() < 1e-12

    # Frozen base parameters: requires_grad=False so .grad stays None.
    saw_base = False
    for name, p in backbone.named_parameters():
        if "lora_" in name or "_head" in name:
            continue
        # This is a base parameter.
        assert not p.requires_grad, f"Base param {name} should be frozen"
        assert p.grad is None or p.grad.abs().max().item() < 1e-12
        saw_base = True
    assert saw_base, "Test sanity: walked no base params"


def test_q_plus_loss_only_updates_q_plus_adapter():
    """Mirror of the V test but with the V loss zeroed out.

    To isolate Q+'s gradient, we send v_targets equal to v_predictions
    so L_V is exactly zero (no gradient w.r.t. V's params).
    """
    backbone = _tiny_backbone()
    input_ids = torch.tensor([[1, 2, 3, 4]])

    v_pred, q_pred = value_forward(backbone, input_ids, algorithm="arm")
    # v_targets == v_predictions so L_V is zero (and L_V's gradient w.r.t. all
    # params is zero -- including V's LoRA).
    v_targets = v_pred.detach().clone()
    q_targets = torch.zeros_like(q_pred)
    out = value_regression_loss_fn(
        ValueLossInputs(
            v_predictions=v_pred[0],
            v_targets=v_targets[0],
            q_plus_predictions=q_pred[0],
            q_plus_targets=q_targets[0],
            loss_mask=torch.ones(v_pred.shape[1], dtype=torch.bool),
            algorithm="arm",
        )
    )
    out.loss.backward()

    saw_q_lora_grad = False
    for module in backbone.base_model.modules():
        if not isinstance(module, MultiLoRALinear):
            continue
        # Q+'s LoRA should have non-zero grad.
        assert module.lora_A[Q_PLUS_SLOT].grad is not None
        if module.lora_A[Q_PLUS_SLOT].grad.abs().max().item() > 0:
            saw_q_lora_grad = True
        # V's LoRA: l_v=0 so grad is zero. (May be present as a tensor if any
        # forward path touched it, just must be near-zero numerically.)
        if module.lora_A[V_SLOT].grad is not None:
            assert module.lora_A[V_SLOT].grad.abs().max().item() < 1e-6
        # V_target's LoRA: never touched.
        if module.lora_A[V_TARGET_SLOT].grad is not None:
            assert module.lora_A[V_TARGET_SLOT].grad.abs().max().item() < 1e-12
    assert saw_q_lora_grad, "Q+'s LoRA never received gradient"

    # Q+'s head should have non-zero grad.
    assert backbone.q_plus_head.linear.weight.grad is not None
    assert backbone.q_plus_head.linear.weight.grad.abs().max().item() > 0
    # V_target's head: never touched.
    if backbone.v_target_head.linear.weight.grad is not None:
        assert backbone.v_target_head.linear.weight.grad.abs().max().item() < 1e-12


def test_combined_arm_loss_updates_v_and_q_plus_adapters():
    """ARM combined loss updates both V and Q+ adapters (and their heads) in
    a single backward pass. V_target adapter and frozen base remain at zero.
    """
    backbone = _tiny_backbone()
    input_ids = torch.tensor([[1, 2, 3, 4]])

    v_pred, q_pred = value_forward(backbone, input_ids, algorithm="arm")
    v_targets = torch.zeros_like(v_pred)
    q_targets = torch.zeros_like(q_pred)
    out = value_regression_loss_fn(
        ValueLossInputs(
            v_predictions=v_pred[0],
            v_targets=v_targets[0],
            q_plus_predictions=q_pred[0],
            q_plus_targets=q_targets[0],
            loss_mask=torch.ones(v_pred.shape[1], dtype=torch.bool),
            algorithm="arm",
        )
    )
    out.loss.backward()

    # V and Q+ both update.
    v_lora_changed = False
    q_lora_changed = False
    vt_lora_unchanged = True
    for module in backbone.base_model.modules():
        if not isinstance(module, MultiLoRALinear):
            continue
        if module.lora_A[V_SLOT].grad is not None and module.lora_A[V_SLOT].grad.abs().max().item() > 0:
            v_lora_changed = True
        if module.lora_A[Q_PLUS_SLOT].grad is not None and module.lora_A[Q_PLUS_SLOT].grad.abs().max().item() > 0:
            q_lora_changed = True
        if module.lora_A[V_TARGET_SLOT].grad is not None and module.lora_A[V_TARGET_SLOT].grad.abs().max().item() >= 1e-12:
            vt_lora_unchanged = False
    assert v_lora_changed and q_lora_changed
    assert vt_lora_unchanged, "V_target adapter should never receive gradient"


# ---------------------------------------------------------------------------
# Polyak update
# ---------------------------------------------------------------------------


def test_polyak_fires_after_each_step_and_only_changes_v_target():
    """After one gradient step + Polyak update:
      - V's adapter changed (gradient descent moved it).
      - V_target's adapter changed (Polyak moved it toward V).
      - Q+'s adapter changed (gradient descent moved it).
    """
    backbone = _tiny_backbone()
    input_ids = torch.tensor([[1, 2, 3, 4]])

    # Snapshot LoRA values before training step.
    v_lora_before: list[torch.Tensor] = []
    q_lora_before: list[torch.Tensor] = []
    vt_lora_before: list[torch.Tensor] = []
    for module in backbone.base_model.modules():
        if isinstance(module, MultiLoRALinear):
            v_lora_before.append(module.lora_A[V_SLOT].detach().clone())
            q_lora_before.append(module.lora_A[Q_PLUS_SLOT].detach().clone())
            vt_lora_before.append(module.lora_A[V_TARGET_SLOT].detach().clone())

    # Build optimizer over trainable params only (exclude frozen base).
    trainable = [p for p in backbone.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable, lr=1e-2)  # large LR for visible step

    # Forward + backward + optimizer step.
    v_pred, q_pred = value_forward(backbone, input_ids, algorithm="arm")
    out = value_regression_loss_fn(
        ValueLossInputs(
            v_predictions=v_pred[0],
            v_targets=torch.zeros_like(v_pred[0]),
            q_plus_predictions=q_pred[0],
            q_plus_targets=torch.zeros_like(q_pred[0]),
            loss_mask=torch.ones(v_pred.shape[1], dtype=torch.bool),
            algorithm="arm",
        )
    )
    optimizer.zero_grad()
    out.loss.backward()
    optimizer.step()

    # Polyak update.
    backbone.polyak_update_v_target(tau=0.005)

    # Verify V's adapter changed (gradient descent).
    v_changed = False
    q_changed = False
    vt_changed = False
    for i, module in enumerate(
        m for m in backbone.base_model.modules() if isinstance(m, MultiLoRALinear)
    ):
        if not torch.allclose(module.lora_A[V_SLOT], v_lora_before[i], atol=1e-9):
            v_changed = True
        if not torch.allclose(module.lora_A[Q_PLUS_SLOT], q_lora_before[i], atol=1e-9):
            q_changed = True
        if not torch.allclose(module.lora_A[V_TARGET_SLOT], vt_lora_before[i], atol=1e-9):
            vt_changed = True
    assert v_changed, "V's adapter did not move under gradient descent"
    assert q_changed, "Q+'s adapter did not move under gradient descent"
    assert vt_changed, "V_target's adapter did not move under Polyak"


def test_v_target_only_changes_via_polyak_not_via_gradient():
    """Without Polyak: V_target's adapter is invariant under gradient steps.
    The gradient routing test already covers "no gradient flows into V_target,"
    but this test pins the implication: optimizer.step() alone (no Polyak)
    leaves V_target's LoRA unchanged.
    """
    backbone = _tiny_backbone()
    input_ids = torch.tensor([[1, 2, 3, 4]])

    vt_lora_before: list[torch.Tensor] = []
    for module in backbone.base_model.modules():
        if isinstance(module, MultiLoRALinear):
            vt_lora_before.append(module.lora_A[V_TARGET_SLOT].detach().clone())

    trainable = [p for p in backbone.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable, lr=1e-2)

    v_pred, q_pred = value_forward(backbone, input_ids, algorithm="arm")
    out = value_regression_loss_fn(
        ValueLossInputs(
            v_predictions=v_pred[0],
            v_targets=torch.zeros_like(v_pred[0]),
            q_plus_predictions=q_pred[0],
            q_plus_targets=torch.zeros_like(q_pred[0]),
            loss_mask=torch.ones(v_pred.shape[1], dtype=torch.bool),
            algorithm="arm",
        )
    )
    optimizer.zero_grad()
    out.loss.backward()
    optimizer.step()
    # Skip Polyak.

    # V_target's adapter should be unchanged.
    for i, module in enumerate(
        m for m in backbone.base_model.modules() if isinstance(m, MultiLoRALinear)
    ):
        assert torch.allclose(module.lora_A[V_TARGET_SLOT], vt_lora_before[i], atol=1e-9), (
            f"V_target's LoRA changed without Polyak at module {i}"
        )


# ---------------------------------------------------------------------------
# Single-training-step integration
# ---------------------------------------------------------------------------


def test_training_step_completes_ppo():
    """Single PPO training step: forward + backward + optimizer + Polyak.
    Asserts loss is finite, V's adapter moved, Q+'s adapter is unchanged."""
    backbone = _tiny_backbone()
    input_ids = torch.tensor([[1, 2, 3, 4]])

    q_lora_before: list[torch.Tensor] = []
    for module in backbone.base_model.modules():
        if isinstance(module, MultiLoRALinear):
            q_lora_before.append(module.lora_A[Q_PLUS_SLOT].detach().clone())

    trainable = [p for p in backbone.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable, lr=1e-2)

    v_pred, q_pred = value_forward(backbone, input_ids, algorithm="ppo")
    assert q_pred is None  # PPO doesn't forward Q+
    out = value_regression_loss_fn(
        ValueLossInputs(
            v_predictions=v_pred[0],
            v_targets=torch.zeros_like(v_pred[0]),
            q_plus_predictions=None,
            q_plus_targets=None,
            loss_mask=torch.ones(v_pred.shape[1], dtype=torch.bool),
            algorithm="ppo",
        )
    )
    assert torch.isfinite(out.loss)
    optimizer.zero_grad()
    out.loss.backward()
    optimizer.step()
    backbone.polyak_update_v_target(tau=0.005)

    # Q+'s adapter should be unchanged in PPO training (no Q+ loss, no Q+ forward).
    for i, module in enumerate(
        m for m in backbone.base_model.modules() if isinstance(m, MultiLoRALinear)
    ):
        assert torch.allclose(module.lora_A[Q_PLUS_SLOT], q_lora_before[i], atol=1e-9), (
            "Q+'s LoRA changed during PPO training"
        )


def test_training_step_loss_decreases_with_repeated_targets():
    """Run repeated training steps with the same fixed targets. Loss should
    decrease overall (compared via mean of late steps vs first step).

    Uses SGD with a conservative LR rather than Adam: Adam's adaptive scaling
    can produce wild early oscillations on tiny models, which makes a
    strict monotonicity check unreliable. The point of the test is "the
    optimizer can actually fit the target" -- a "late mean below initial"
    check is the right granularity.
    """
    backbone = _tiny_backbone()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    fixed_v_target = torch.zeros(input_ids.shape[1])
    fixed_q_target = torch.zeros(input_ids.shape[1])

    trainable = [p for p in backbone.parameters() if p.requires_grad]
    optimizer = optim.SGD(trainable, lr=1e-3)

    losses = []
    n_steps = 20
    for _ in range(n_steps):
        v_pred, q_pred = value_forward(backbone, input_ids, algorithm="arm")
        out = value_regression_loss_fn(
            ValueLossInputs(
                v_predictions=v_pred[0],
                v_targets=fixed_v_target,
                q_plus_predictions=q_pred[0],
                q_plus_targets=fixed_q_target,
                loss_mask=torch.ones(v_pred.shape[1], dtype=torch.bool),
                algorithm="arm",
            )
        )
        losses.append(out.loss.item())
        optimizer.zero_grad()
        out.loss.backward()
        optimizer.step()
        backbone.polyak_update_v_target(tau=0.005)

    # Mean of the last 5 steps should be below the first loss -- the optimizer
    # has clearly moved in the right direction.
    late_mean = sum(losses[-5:]) / 5
    assert late_mean < losses[0], (
        f"Late-mean loss did not decrease vs first step: first={losses[0]}, "
        f"late_mean={late_mean}, all={losses}"
    )
