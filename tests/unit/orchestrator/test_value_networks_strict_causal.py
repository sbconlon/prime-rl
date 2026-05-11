"""Unit tests for plan 4.1: slot-dependent 4D strict-causal mask.

The mask is *not* tested against an algorithmic oracle (no golden
implementation exists for "strict-causal multi-layer Q+ forward" —
that\'s exactly the algorithmic change). Instead these tests verify:

  1. The mask\'s shape and per-cell values are constructed correctly.
  2. Position 0 self-attends (no NaN propagation).
  3. The mask is applied only on Q+ slot forwards; V and V_target
     forwards remain on HF\'s default self-inclusive causal path.
  4. The Q+ strict-causal forward differs from the standard-causal
     forward (sanity: the mask actually changes the output).
  5. Backward through the masked Q+ forward produces finite, non-zero
     gradients to Q+\'s LoRA-slot-1 parameters (HF SDPA backward works
     with the 4D mask -- the Q5 question from plan review).

Uses the tiny 2-layer Qwen2 fixture from the existing
test_value_networks.py for CPU-runnable tests.
"""

from __future__ import annotations

import pytest
import torch
from transformers import Qwen2Config, Qwen2Model

from prime_rl.configs.trainer import LoRAConfig
from prime_rl.orchestrator.value_networks import (
    Q_PLUS_SLOT,
    V_SLOT,
    V_TARGET_SLOT,
    ValueNetworkBackbone,
)


# ---------------------------------------------------------------------------
# Fixtures (mirrors test_value_networks.py\'s tiny config)
# ---------------------------------------------------------------------------


def _tiny_qwen2_config() -> Qwen2Config:
    return Qwen2Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
    )


@pytest.fixture
def tiny_backbone() -> ValueNetworkBackbone:
    torch.manual_seed(0)
    base = Qwen2Model(_tiny_qwen2_config())
    lora = LoRAConfig(rank=8, alpha=16.0, dropout=0.0)
    backbone = ValueNetworkBackbone(base, lora_config=lora, polyak_tau=0.005)
    # Break zero-init so Q+ produces non-trivial outputs.
    torch.manual_seed(1)
    with torch.no_grad():
        for name, p in backbone.named_parameters():
            if ("lora_A" in name) or ("lora_B" in name):
                p.copy_(torch.randn_like(p) * 0.1)
        for head in (backbone.v_head, backbone.q_plus_head, backbone.v_target_head):
            torch.nn.init.normal_(head.linear.weight, std=0.02)
    return backbone


# ---------------------------------------------------------------------------
# 1. Mask construction
# ---------------------------------------------------------------------------


def test_strict_causal_mask_shape(tiny_backbone: ValueNetworkBackbone):
    """Mask has shape [1, 1, seq_len, seq_len]."""
    mask = tiny_backbone._build_strict_causal_mask(seq_len=8, device=torch.device("cpu"))
    assert mask.shape == (1, 1, 8, 8)


def test_strict_causal_mask_pattern(tiny_backbone: ValueNetworkBackbone):
    """Specific cells: 0 for j < i, -inf for j >= i, except [0, 0] = 0."""
    mask = tiny_backbone._build_strict_causal_mask(seq_len=4, device=torch.device("cpu"))
    m = mask[0, 0]  # [4, 4]
    sentinel = torch.finfo(m.dtype).min

    # Position 0 self-attend: only [0, 0] is 0, rest of row 0 is -inf.
    assert m[0, 0].item() == 0.0
    assert m[0, 1].item() == sentinel
    assert m[0, 2].item() == sentinel
    assert m[0, 3].item() == sentinel

    # Position 1: [1, 0] = 0 (attend to 0), [1, 1..] = -inf (no self, no future).
    assert m[1, 0].item() == 0.0
    assert m[1, 1].item() == sentinel
    assert m[1, 2].item() == sentinel

    # Position 2: [2, 0..1] = 0, [2, 2..] = -inf.
    assert m[2, 0].item() == 0.0
    assert m[2, 1].item() == 0.0
    assert m[2, 2].item() == sentinel
    assert m[2, 3].item() == sentinel

    # Position 3: [3, 0..2] = 0, [3, 3] = -inf.
    assert m[3, 0].item() == 0.0
    assert m[3, 1].item() == 0.0
    assert m[3, 2].item() == 0.0
    assert m[3, 3].item() == sentinel


def test_strict_causal_mask_dtype_matches_model(tiny_backbone: ValueNetworkBackbone):
    """Mask dtype matches the base model\'s parameter dtype."""
    param_dtype = next(tiny_backbone.base_model.parameters()).dtype
    mask = tiny_backbone._build_strict_causal_mask(seq_len=8, device=torch.device("cpu"))
    assert mask.dtype == param_dtype


# ---------------------------------------------------------------------------
# 2. Position 0 self-attend: no NaN
# ---------------------------------------------------------------------------


def test_qplus_forward_position_0_no_nan(tiny_backbone: ValueNetworkBackbone):
    """forward_q_plus_sampled_all_positions must not produce NaN at position 0
    even though strict-causal would otherwise mask the entire row."""
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    q_values, _ = tiny_backbone.forward_q_plus_sampled_all_positions(input_ids)
    assert torch.isfinite(q_values).all(), (
        f"NaN/Inf in Q+ output: {q_values}"
    )


# ---------------------------------------------------------------------------
# 3. Mask is applied only on Q+ slot
# ---------------------------------------------------------------------------


def test_qplus_forward_differs_from_unmasked(tiny_backbone: ValueNetworkBackbone):
    """Sanity: the strict-causal mask actually changes the Q+ output vs the
    HF-default self-inclusive forward. If outputs were identical, either the
    mask wasn\'t applied or it was a no-op."""
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)

    # With the strict-causal mask (current implementation).
    q_strict, _ = tiny_backbone.forward_q_plus_sampled_all_positions(input_ids)

    # Without any mask (would be standard self-inclusive causal).
    # Bypass _forward_all_positions by routing manually + calling base_model.
    from prime_rl.orchestrator.value_networks import _adapter_routing
    with _adapter_routing(Q_PLUS_SLOT, input_ids.numel()):
        output_default = tiny_backbone.base_model(input_ids, use_cache=True)
    q_default = tiny_backbone.q_plus_head(output_default.last_hidden_state)

    # Should differ across the board for positions > 0 (position 0 is
    # self-attend in both; small numerical differences are fine to consider
    # equivalent there).
    for pos in range(1, q_strict.shape[1]):
        diff = (q_strict[0, pos] - q_default[0, pos]).abs().item()
        assert diff > 1e-6, (
            f"Q+ at position {pos} is identical between strict-causal and default; "
            f"the mask did not change the output (diff={diff})"
        )


def test_v_and_vtarget_unaffected_by_mask_change(tiny_backbone: ValueNetworkBackbone):
    """V and V_target forwards must not use the strict-causal mask. Verify by
    calling forward_v_all_positions / forward_v_target_all_positions and
    confirming the output matches a manually-no-mask call through the same
    slots."""
    from prime_rl.orchestrator.value_networks import _adapter_routing
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)

    for slot, fwd, head in [
        (V_SLOT, tiny_backbone.forward_v_all_positions, tiny_backbone.v_head),
        (V_TARGET_SLOT, tiny_backbone.forward_v_target_all_positions, tiny_backbone.v_target_head),
    ]:
        produced, _ = fwd(input_ids)
        # Manually run the no-mask forward through the same slot.
        with _adapter_routing(slot, input_ids.numel()):
            ref_out = tiny_backbone.base_model(input_ids, use_cache=True)
        reference = head(ref_out.last_hidden_state)
        assert torch.allclose(produced, reference, atol=1e-6), (
            f"slot {slot} forward differs from no-mask reference -- the mask leaked into V/V_target path"
        )


# ---------------------------------------------------------------------------
# 4. Backward through the masked path (the Q5 question from plan review)
# ---------------------------------------------------------------------------


def test_qplus_strict_causal_backward_produces_finite_gradients(
    tiny_backbone: ValueNetworkBackbone,
):
    """Backward through the masked Q+ forward must produce finite, non-zero
    gradients to Q+\'s LoRA-slot-1 parameters. This is the Q5 concern from
    the plan review (does HF SDPA backward work with a custom 4D mask?)."""
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)

    # Synthetic q_plus_target (scalar regression target per position).
    q_target = torch.randn(1, 8)

    # Forward + loss.
    q_pred, _ = tiny_backbone.forward_q_plus_sampled_all_positions(input_ids)
    # Loss mask: positions 1..7 (exclude position 0 since it\'s degenerate).
    loss_mask = torch.tensor([[False, True, True, True, True, True, True, True]])
    loss = (((q_pred - q_target) ** 2)[loss_mask]).mean()
    loss.backward()

    # Locate Q+ slot LoRA params: any param with "lora_A" or "lora_B" name
    # whose index == Q_PLUS_SLOT (1).
    found_q_plus_grad = False
    for name, p in tiny_backbone.named_parameters():
        # The adapter index is the FINAL segment of the param name (lora_A is
        # an nn.ParameterList, so its entries are indexed: e.g.
        # "base_model.layers.0.self_attn.q_proj.lora_A.1" is the Q+-slot
        # adapter of q_proj in layer 0. Match endswith, not substring.
        is_lora_A = name.endswith(f'.lora_A.{Q_PLUS_SLOT}')
        is_lora_B = name.endswith(f'.lora_B.{Q_PLUS_SLOT}')
        if not (is_lora_A or is_lora_B):
            continue
        # This is a Q+ LoRA param.
        found_q_plus_grad = True
        assert p.grad is not None, f"{name} has no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has non-finite gradient"
        assert (p.grad != 0).any(), f"{name} has all-zero gradient"

    assert found_q_plus_grad, "no Q+ LoRA params were found -- test setup wrong"
