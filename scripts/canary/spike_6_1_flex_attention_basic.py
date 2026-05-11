"""Spike 6.1: FlexAttention basic capability.

Validates PyTorch's FlexAttention can express strict-causal attention
correctly and produces forward + backward outputs that match
`F.scaled_dot_product_attention` with the equivalent 4D additive mask.

Used as a derisking gate before implementing the FlexAttention-based
strict-causal Q+ all-positions forward (lever 1 of the AdvTrainer perf
pass, 2026-05-11). If this spike fails, we abort before writing any
real kernel code.

Four checks (all must PASS):

  (1) torch.nn.attention.flex_attention import succeeds + CUDA available.
  (2) Forward parity: flex_attention output matches SDPA Math output
      within BF16 tolerance at Qwen3-0.6B-shaped Q/K/V (no GQA in
      this spike -- 6.2 covers GQA composition).
  (3) Backward parity: gradients w.r.t. Q, K, V match SDPA Math within
      tolerance.
  (4) Forward speed: flex_attention >= 1.5x faster than SDPA Math at
      S=2048. (Realistic floor; we expect 3-5x in practice.)

Run:
    uv run python scripts/canary/spike_6_1_flex_attention_basic.py
"""

from __future__ import annotations

import sys
import time

import torch
import torch.nn.functional as F


if not torch.cuda.is_available():
    print("ERROR: spike requires CUDA; no GPU available.", file=sys.stderr)
    sys.exit(1)

print("=" * 72)
print("Spike 6.1: FlexAttention basic capability")
print("=" * 72)
print(f"  torch.__version__       = {torch.__version__}")
print(f"  CUDA device             = {torch.cuda.get_device_name(0)}")

# ---- Check 1: import ----
try:
    from torch.nn.attention.flex_attention import (
        create_block_mask,
        flex_attention,
    )
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError as e:
    print(f"\nFAIL [1]: FlexAttention import failed: {e}")
    print("  Need PyTorch >= 2.5 with FlexAttention enabled.")
    sys.exit(1)
print("\nPASS [1]: torch.nn.attention.flex_attention imported")

# Compile the kernel ahead of time so first-call cost doesn't pollute timing.
flex_attention_c = torch.compile(flex_attention, dynamic=False)

B, H, S, D = 1, 16, 2048, 64
dtype = torch.bfloat16
device = torch.device("cuda")
print(f"\nInput shapes: B={B}, H={H}, S={S}, D={D}, dtype={dtype}")

torch.manual_seed(0)
Q = torch.randn(B, H, S, D, dtype=dtype, device=device)
K = torch.randn(B, H, S, D, dtype=dtype, device=device)
V = torch.randn(B, H, S, D, dtype=dtype, device=device)


def strict_causal_mask_mod(b, h, q_idx, kv_idx):
    return (kv_idx < q_idx) | ((q_idx == 0) & (kv_idx == 0))


print("\nBuilding FlexAttention block_mask (one-time JIT)...")
t0 = time.perf_counter()
block_mask = create_block_mask(
    strict_causal_mask_mod,
    B=None, H=None,
    Q_LEN=S, KV_LEN=S,
    device=device,
)
torch.cuda.synchronize()
print(f"  block_mask built in {(time.perf_counter() - t0)*1000:.1f} ms")

# Equivalent 4D additive mask for SDPA.
i_idx = torch.arange(S, device=device).unsqueeze(1)
j_idx = torch.arange(S, device=device).unsqueeze(0)
keep = (j_idx < i_idx) | ((i_idx == 0) & (j_idx == 0))
sentinel = torch.finfo(dtype).min
sdpa_mask = torch.where(
    keep,
    torch.zeros((), dtype=dtype, device=device),
    torch.full((), sentinel, dtype=dtype, device=device),
).unsqueeze(0).unsqueeze(0)

# ---- Check 2: Forward parity ----
print("\n[2] Forward parity check...")

for _ in range(3):
    _ = flex_attention_c(Q, K, V, block_mask=block_mask)
torch.cuda.synchronize()

out_flex = flex_attention_c(Q, K, V, block_mask=block_mask)
with sdpa_kernel(SDPBackend.MATH):
    out_sdpa = F.scaled_dot_product_attention(Q, K, V, attn_mask=sdpa_mask)
torch.cuda.synchronize()

atol = 5e-3
rtol = 1e-2
abs_diff = (out_flex - out_sdpa).abs()
max_abs = abs_diff.max().item()
max_rel = (abs_diff / (out_sdpa.abs() + 1e-6)).max().item()
print(f"  out_flex.shape = {tuple(out_flex.shape)}, dtype={out_flex.dtype}")
print(f"  max_abs_diff   = {max_abs:.4e}  (tol atol={atol})")
print(f"  max_rel_diff   = {max_rel:.4e}  (tol rtol={rtol})")
forward_pass = (max_abs < atol) or (max_rel < rtol)
print(f"  {'PASS' if forward_pass else 'FAIL'} [2]: forward outputs match")

# ---- Check 3: Backward parity ----
print("\n[3] Backward parity check...")

Qf = Q.detach().clone().requires_grad_(True)
Kf = K.detach().clone().requires_grad_(True)
Vf = V.detach().clone().requires_grad_(True)
Qs = Q.detach().clone().requires_grad_(True)
Ks = K.detach().clone().requires_grad_(True)
Vs = V.detach().clone().requires_grad_(True)

target = torch.randn_like(Q)

out_f = flex_attention_c(Qf, Kf, Vf, block_mask=block_mask)
(out_f * target).sum().backward()

with sdpa_kernel(SDPBackend.MATH):
    out_s = F.scaled_dot_product_attention(Qs, Ks, Vs, attn_mask=sdpa_mask)
    (out_s * target).sum().backward()

backward_pass = True
for name, gf, gs in [("dQ", Qf.grad, Qs.grad), ("dK", Kf.grad, Ks.grad), ("dV", Vf.grad, Vs.grad)]:
    if gf is None or gs is None:
        print(f"  FAIL [3]: gradient missing for {name}")
        backward_pass = False
        continue
    if not torch.isfinite(gf).all():
        print(f"  FAIL [3]: non-finite gradient in flex {name}")
        backward_pass = False
        continue
    abs_d = (gf - gs).abs()
    mab = abs_d.max().item()
    mre = (abs_d / (gs.abs() + 1e-6)).max().item()
    ok = (mab < atol) or (mre < rtol)
    backward_pass &= ok
    print(f"  {name}: max_abs={mab:.4e}, max_rel={mre:.4e}  {'OK' if ok else 'BAD'}")
print(f"  {'PASS' if backward_pass else 'FAIL'} [3]: backward gradients match")

# ---- Check 4: Forward speed ----
print("\n[4] Forward speed (20 iters, post-warmup)...")

for _ in range(5):
    _ = flex_attention_c(Q, K, V, block_mask=block_mask)
torch.cuda.synchronize()

n_iter = 20
t0 = time.perf_counter()
for _ in range(n_iter):
    _ = flex_attention_c(Q, K, V, block_mask=block_mask)
torch.cuda.synchronize()
flex_ms = (time.perf_counter() - t0) / n_iter * 1000

with sdpa_kernel(SDPBackend.MATH):
    for _ in range(5):
        _ = F.scaled_dot_product_attention(Q, K, V, attn_mask=sdpa_mask)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = F.scaled_dot_product_attention(Q, K, V, attn_mask=sdpa_mask)
    torch.cuda.synchronize()
    sdpa_ms = (time.perf_counter() - t0) / n_iter * 1000

speedup = sdpa_ms / flex_ms
print(f"  flex_attention  = {flex_ms:.2f} ms/call")
print(f"  sdpa Math       = {sdpa_ms:.2f} ms/call")
print(f"  speedup         = {speedup:.2f}x")
speed_pass = speedup > 1.5
print(f"  {'PASS' if speed_pass else 'FAIL'} [4]: FlexAttention >= 1.5x faster")

print("\n" + "=" * 72)
all_pass = forward_pass and backward_pass and speed_pass
if all_pass:
    print("Spike 6.1: ALL CHECKS PASS  -- FlexAttention is viable.")
    print("Proceed to Spike 6.2 (GQA + RoPE composition with a real Qwen3 layer).")
else:
    print("Spike 6.1: FAILED. Do not proceed to implementation.")
print("=" * 72)
sys.exit(0 if all_pass else 1)
