"""Spike 5.1: MultiLoRALinear under rank-2 input + empty groups.

Tests the assumption that `MultiLoRALinear.forward` produces correct outputs
for rank-2 input shape `[B, hidden]` under single-slot routing
`LORA_NUM_TOKENS = [0, B, 0]` -- the exact pattern the new
forward_q_plus_candidates_batched method will use.

Three assertions, in order of escalating strictness:

  (1) Rank-2 input `[B, hidden]` produces the same output as rank-3 input
      `[B, 1, hidden]` (squeezed) under identical routing. Confirms the
      `.view(-1, in)` reshape preserves semantics.

  (2) Routing to a different slot produces a different output. Confirms the
      slot is actually being respected, not silently defaulting to slot 0
      (or all slots being summed).

  (3) `use_grouped_mm=True` produces the same output as `use_grouped_mm=False`
      (the for-loop fallback) when the active group has nonzero rows and the
      others are empty. Confirms `torch._grouped_mm` handles empty groups
      correctly.

Run:
    uv run python scripts/canary/spike_5_1_multilora_rank2.py

Expects: all checks PASS. Run time ~30s on any GPU; no model load.
"""

from __future__ import annotations

import sys

import torch
from torch import nn

# ---------------------------------------------------------------------------
# Setup: import + initialize the LoRA globals before constructing the module.
# ---------------------------------------------------------------------------

from prime_rl.trainer.models.layers.lora import base as lora_base
from prime_rl.trainer.models.layers.lora.base import (
    set_lora_num_tokens,
    set_multilora_scaling,
)
from prime_rl.trainer.models.layers.lora.multi_linear import MultiLoRALinear


N_ADAPTERS = 3
HIDDEN_SIZE = 1024
B = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32  # spike runs in FP32 for clean parity checks
print(f"device={device} dtype={dtype}")


def init_lora_globals(n_adapters: int = N_ADAPTERS, scaling: float = 2.0) -> None:
    """Initialize LORA_NUM_TOKENS and SCALING_FACTORS to shape [n_adapters]."""
    lora_base.LORA_NUM_TOKENS = None
    lora_base.SCALING_FACTORS = None
    set_lora_num_tokens(torch.zeros(n_adapters, dtype=torch.long), reset_reference=True)
    set_multilora_scaling(
        torch.full((n_adapters,), float(scaling)), reset_reference=True
    )


def build_module(use_grouped_mm: bool = True) -> MultiLoRALinear:
    """Build a fresh MultiLoRALinear wrapping a synthetic nn.Linear."""
    base = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False).to(device, dtype)
    # Reset base weights to a deterministic value so the parity tests aren't
    # confounded by RNG drift between module instances.
    with torch.no_grad():
        torch.manual_seed(0)
        base.weight.copy_(torch.randn_like(base.weight) * 0.02)
    return MultiLoRALinear(
        base_layer=base,
        rank=8,
        n_adapters=N_ADAPTERS,
        alpha=16.0,
        dropout=0.0,
        use_grouped_mm=use_grouped_mm,
    )


def copy_lora_weights(src: MultiLoRALinear, dst: MultiLoRALinear) -> None:
    """Copy lora_A and lora_B parameters from src to dst."""
    with torch.no_grad():
        for i in range(N_ADAPTERS):
            dst.lora_A[i].copy_(src.lora_A[i])
            dst.lora_B[i].copy_(src.lora_B[i])
        dst.base_layer.weight.copy_(src.base_layer.weight)


def report(label: str, ok: bool, max_abs_diff: float, atol: float) -> None:
    status = "[32mPASS[0m" if ok else "[31mFAIL[0m"
    print(f"  {status}: {label}  (max_abs_diff={max_abs_diff:.3e}, atol={atol:.0e})")


def main() -> int:
    init_lora_globals()

    # Build one module + non-trivial LoRA weights, share across tests.
    mod = build_module(use_grouped_mm=True)
    with torch.no_grad():
        torch.manual_seed(1)
        for i in range(N_ADAPTERS):
            mod.lora_A[i].copy_(torch.randn_like(mod.lora_A[i]) * 0.1)
            mod.lora_B[i].copy_(torch.randn_like(mod.lora_B[i]) * 0.1)

    print()
    print("===== Spike 5.1: MultiLoRALinear rank-2 + empty groups =====")
    print()

    all_pass = True

    # -----------------------------------------------------------------------
    # Check 1: rank-2 [B, H] vs rank-3 [B, 1, H] (squeezed) parity
    # -----------------------------------------------------------------------
    print("--- Check 1: rank-2 vs rank-3 parity (Q+ slot only) ---")
    torch.manual_seed(2)
    x_2d = torch.randn(B, HIDDEN_SIZE, device=device, dtype=dtype)
    x_3d = x_2d.view(B, 1, HIDDEN_SIZE)  # logically identical

    # Route all B tokens to slot 1 (Q+ analogue): LORA_NUM_TOKENS = [0, B, 0].
    set_lora_num_tokens(torch.tensor([0, B, 0], dtype=torch.long))
    y_2d = mod(x_2d)
    y_3d = mod(x_3d).squeeze(1)

    diff_1 = (y_2d - y_3d).abs().max().item()
    ok_1 = diff_1 < 1e-5
    report("rank-2 == rank-3 under [0, B, 0] routing", ok_1, diff_1, 1e-5)
    all_pass &= ok_1

    # -----------------------------------------------------------------------
    # Check 2: routing actually selects the slot.
    # -----------------------------------------------------------------------
    print()
    print("--- Check 2: slot routing changes output ---")
    set_lora_num_tokens(torch.tensor([0, B, 0], dtype=torch.long))
    y_q_plus = mod(x_2d)
    set_lora_num_tokens(torch.tensor([B, 0, 0], dtype=torch.long))
    y_v = mod(x_2d)
    set_lora_num_tokens(torch.tensor([0, 0, B], dtype=torch.long))
    y_v_target = mod(x_2d)

    diff_qv = (y_q_plus - y_v).abs().max().item()
    diff_qvt = (y_q_plus - y_v_target).abs().max().item()
    diff_vvt = (y_v - y_v_target).abs().max().item()

    ok_2 = diff_qv > 1e-3 and diff_qvt > 1e-3 and diff_vvt > 1e-3
    print(f"      Q+ vs V    diff: {diff_qv:.3e}")
    print(f"      Q+ vs V_t  diff: {diff_qvt:.3e}")
    print(f"      V  vs V_t  diff: {diff_vvt:.3e}")
    report("each slot produces a distinct output (>1e-3)", ok_2, min(diff_qv, diff_qvt, diff_vvt), 1e-3)
    all_pass &= ok_2

    # -----------------------------------------------------------------------
    # Check 3: grouped_mm vs for-loop fallback parity at [0, B, 0]
    # -----------------------------------------------------------------------
    print()
    print("--- Check 3: grouped_mm vs for-loop parity (empty groups) ---")
    mod_loop = build_module(use_grouped_mm=False)
    copy_lora_weights(mod, mod_loop)

    set_lora_num_tokens(torch.tensor([0, B, 0], dtype=torch.long))
    y_grouped = mod(x_2d)
    y_loop = mod_loop(x_2d)

    diff_3 = (y_grouped - y_loop).abs().max().item()
    ok_3 = diff_3 < 1e-4
    report("grouped_mm == for-loop under [0, B, 0]", ok_3, diff_3, 1e-4)
    all_pass &= ok_3

    # -----------------------------------------------------------------------
    # Check 4 (bonus): mixed-slot routing [B/2, B/2, 0]
    # -----------------------------------------------------------------------
    print()
    print("--- Check 4: mixed-slot routing parity (grouped vs loop) ---")
    set_lora_num_tokens(torch.tensor([B // 2, B // 2, 0], dtype=torch.long))
    y_mixed_grouped = mod(x_2d)
    y_mixed_loop = mod_loop(x_2d)

    diff_4 = (y_mixed_grouped - y_mixed_loop).abs().max().item()
    ok_4 = diff_4 < 1e-4
    report("grouped_mm == for-loop under [B/2, B/2, 0]", ok_4, diff_4, 1e-4)
    all_pass &= ok_4

    # -----------------------------------------------------------------------
    # Check 5 (sanity): different slots in mixed routing give different
    # halves of the output.
    # -----------------------------------------------------------------------
    print()
    print("--- Check 5: mixed routing first half != second half ---")
    set_lora_num_tokens(torch.tensor([B // 2, B // 2, 0], dtype=torch.long))
    y_mixed = mod(x_2d)
    set_lora_num_tokens(torch.tensor([B, 0, 0], dtype=torch.long))
    y_all_v = mod(x_2d)
    set_lora_num_tokens(torch.tensor([0, B, 0], dtype=torch.long))
    y_all_qplus = mod(x_2d)

    # First half of y_mixed should match first half of y_all_v;
    # second half should match second half of y_all_qplus.
    diff_first  = (y_mixed[: B // 2] - y_all_v[: B // 2]).abs().max().item()
    diff_second = (y_mixed[B // 2:] - y_all_qplus[B // 2:]).abs().max().item()
    ok_5 = diff_first < 1e-5 and diff_second < 1e-5
    print(f"      first half  vs all-V    diff: {diff_first:.3e}")
    print(f"      second half vs all-Q+   diff: {diff_second:.3e}")
    report("each slot\'s rows match its single-slot output", ok_5, max(diff_first, diff_second), 1e-5)
    all_pass &= ok_5

    # -----------------------------------------------------------------------
    print()
    if all_pass:
        print("\033[32m===== Spike 5.1 PASS =====\033[0m")
        return 0
    print("\033[31m===== Spike 5.1 FAIL =====\033[0m")
    return 1


if __name__ == "__main__":
    sys.exit(main())
