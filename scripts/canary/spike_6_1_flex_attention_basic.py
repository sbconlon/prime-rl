"""Spike 6.1: FlexAttention basic capability.

Validates PyTorch's FlexAttention can express strict-causal attention
correctly and produces forward + backward outputs that match
`F.scaled_dot_product_attention` with the equivalent 4D additive mask.

Derisking gate before implementing the FlexAttention-based strict-causal
Q+ all-positions forward (lever 1 of the AdvTrainer perf pass).

Strategy: separate is-the-math-right from is-BF16-noise-acceptable.

  - FP32 parity: tight tolerance (atol=1e-5). The correctness gate.
  - BF16 sanity: finite + within BF16 accumulation noise budget (atol=3e-2).
  - BF16 speed:  FlexAttention >= 1.5x faster than SDPA Math at S=2048.

Five checks (all must PASS):

  (1) FlexAttention import + CUDA available.
  (2) FP32 forward parity (tight, the correctness gate).
  (3) FP32 backward parity for dQ/dK/dV.
  (4) BF16 forward sanity (finite + within BF16 noise budget).
  (5) BF16 forward speed >= 1.5x SDPA Math at S=2048.

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
    sys.exit(1)
print("\nPASS [1]: torch.nn.attention.flex_attention imported")

flex_attention_c = torch.compile(flex_attention, dynamic=False)

B, H, S, D = 1, 16, 2048, 64
device = torch.device("cuda")
print(f"\nShapes: B={B}, H={H}, S={S}, D={D}")


def strict_causal_mask_mod(b, h, q_idx, kv_idx):
    return (kv_idx < q_idx) | ((q_idx == 0) & (kv_idx == 0))


print("\nBuilding FlexAttention block_mask (one-time JIT)...")
t0 = time.perf_counter()
block_mask = create_block_mask(
    strict_causal_mask_mod,
    B=None, H=None,  # broadcast across batch + heads
    Q_LEN=S, KV_LEN=S,
    device=device,
)
torch.cuda.synchronize()
print(f"  block_mask built in {(time.perf_counter() - t0)*1000:.1f} ms")


def build_sdpa_mask(dtype):
    i_idx = torch.arange(S, device=device).unsqueeze(1)
    j_idx = torch.arange(S, device=device).unsqueeze(0)
    keep = (j_idx < i_idx) | ((i_idx == 0) & (j_idx == 0))
    sentinel = torch.finfo(dtype).min
    return torch.where(
        keep,
        torch.zeros((), dtype=dtype, device=device),
        torch.full((), sentinel, dtype=dtype, device=device),
    ).unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# Check 2: FP32 forward parity (correctness gate)
# ---------------------------------------------------------------------------

print("\n[2] FP32 forward parity (correctness gate)...")
torch.manual_seed(0)
Q32 = torch.randn(B, H, S, D, dtype=torch.float32, device=device)
K32 = torch.randn(B, H, S, D, dtype=torch.float32, device=device)
V32 = torch.randn(B, H, S, D, dtype=torch.float32, device=device)
sdpa_mask_fp32 = build_sdpa_mask(torch.float32)

for _ in range(3):
    _ = flex_attention_c(Q32, K32, V32, block_mask=block_mask)
torch.cuda.synchronize()

out_flex_fp32 = flex_attention_c(Q32, K32, V32, block_mask=block_mask)
with sdpa_kernel(SDPBackend.MATH):
    out_sdpa_fp32 = F.scaled_dot_product_attention(Q32, K32, V32, attn_mask=sdpa_mask_fp32)
torch.cuda.synchronize()

abs_diff = (out_flex_fp32 - out_sdpa_fp32).abs()
max_abs = abs_diff.max().item()
mean_abs = abs_diff.mean().item()
tol_fp32 = 1e-5
print(f"  max_abs_diff  = {max_abs:.3e}")
print(f"  mean_abs_diff = {mean_abs:.3e}")
print(f"  out range     = [{out_sdpa_fp32.abs().min().item():.3e}, {out_sdpa_fp32.abs().max().item():.3e}]")
fp32_fwd_pass = max_abs < tol_fp32
print(f"  {'PASS' if fp32_fwd_pass else 'FAIL'} [2]: FP32 forward within atol={tol_fp32}")

# ---------------------------------------------------------------------------
# Check 3: FP32 backward parity
# ---------------------------------------------------------------------------

print("\n[3] FP32 backward parity...")
Qf = Q32.detach().clone().requires_grad_(True)
Kf = K32.detach().clone().requires_grad_(True)
Vf = V32.detach().clone().requires_grad_(True)
Qs = Q32.detach().clone().requires_grad_(True)
Ks = K32.detach().clone().requires_grad_(True)
Vs = V32.detach().clone().requires_grad_(True)
target = torch.randn_like(Q32)

out_f = flex_attention_c(Qf, Kf, Vf, block_mask=block_mask)
(out_f * target).sum().backward()
with sdpa_kernel(SDPBackend.MATH):
    out_s = F.scaled_dot_product_attention(Qs, Ks, Vs, attn_mask=sdpa_mask_fp32)
    (out_s * target).sum().backward()

bwd_tol = 1e-4
fp32_bwd_pass = True
for name, gf, gs in [("dQ", Qf.grad, Qs.grad), ("dK", Kf.grad, Ks.grad), ("dV", Vf.grad, Vs.grad)]:
    if gf is None or gs is None:
        print(f"  FAIL: missing grad for {name}")
        fp32_bwd_pass = False
        continue
    if not torch.isfinite(gf).all():
        print(f"  FAIL: non-finite grad for {name}")
        fp32_bwd_pass = False
        continue
    mab = (gf - gs).abs().max().item()
    ok = mab < bwd_tol
    fp32_bwd_pass = fp32_bwd_pass and ok
    print(f"  {name}: max_abs={mab:.3e}  {'OK' if ok else 'BAD'}")
print(f"  {'PASS' if fp32_bwd_pass else 'FAIL'} [3]: FP32 backward within atol={bwd_tol}")

# ---------------------------------------------------------------------------
# Check 4: BF16 forward sanity (finite + within BF16 noise budget)
# ---------------------------------------------------------------------------

print("\n[4] BF16 forward sanity (atol=3e-2, BF16 accumulation noise)...")
Q16 = Q32.to(torch.bfloat16)
K16 = K32.to(torch.bfloat16)
V16 = V32.to(torch.bfloat16)
sdpa_mask_bf16 = build_sdpa_mask(torch.bfloat16)

for _ in range(3):
    _ = flex_attention_c(Q16, K16, V16, block_mask=block_mask)
torch.cuda.synchronize()

out_flex_bf16 = flex_attention_c(Q16, K16, V16, block_mask=block_mask)
with sdpa_kernel(SDPBackend.MATH):
    out_sdpa_bf16 = F.scaled_dot_product_attention(Q16, K16, V16, attn_mask=sdpa_mask_bf16)
torch.cuda.synchronize()

bf16_abs = (out_flex_bf16.float() - out_sdpa_bf16.float()).abs()
bf16_max_abs = bf16_abs.max().item()
bf16_mean_abs = bf16_abs.mean().item()
bf16_finite = torch.isfinite(out_flex_bf16).all().item()
bf16_tol = 3e-2
print(f"  max_abs_diff  = {bf16_max_abs:.3e}")
print(f"  mean_abs_diff = {bf16_mean_abs:.3e}")
print(f"  isfinite      = {bf16_finite}")
bf16_pass = bf16_finite and (bf16_max_abs < bf16_tol)
print(f"  {'PASS' if bf16_pass else 'FAIL'} [4]: BF16 forward within atol={bf16_tol}")

# ---------------------------------------------------------------------------
# Check 5: BF16 forward speed
# ---------------------------------------------------------------------------

print("\n[5] BF16 forward speed (20 iters, post-warmup)...")
for _ in range(5):
    _ = flex_attention_c(Q16, K16, V16, block_mask=block_mask)
torch.cuda.synchronize()

n_iter = 20
t0 = time.perf_counter()
for _ in range(n_iter):
    _ = flex_attention_c(Q16, K16, V16, block_mask=block_mask)
torch.cuda.synchronize()
flex_ms = (time.perf_counter() - t0) / n_iter * 1000

with sdpa_kernel(SDPBackend.MATH):
    for _ in range(5):
        _ = F.scaled_dot_product_attention(Q16, K16, V16, attn_mask=sdpa_mask_bf16)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = F.scaled_dot_product_attention(Q16, K16, V16, attn_mask=sdpa_mask_bf16)
    torch.cuda.synchronize()
    sdpa_ms = (time.perf_counter() - t0) / n_iter * 1000

speedup = sdpa_ms / flex_ms
print(f"  flex_attention  = {flex_ms:.3f} ms/call")
print(f"  sdpa Math       = {sdpa_ms:.3f} ms/call")
print(f"  speedup         = {speedup:.2f}x")
speed_pass = speedup > 1.5
print(f"  {'PASS' if speed_pass else 'FAIL'} [5]: FlexAttention >= 1.5x faster")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("\n" + "=" * 72)
all_pass = fp32_fwd_pass and fp32_bwd_pass and bf16_pass and speed_pass
if all_pass:
    print("Spike 6.1: ALL CHECKS PASS  -- FlexAttention is viable.")
    print("Proceed to Spike 6.2 (GQA + RoPE composition with a real Qwen3 layer).")
else:
    print("Spike 6.1: FAILED.")
print("=" * 72)
sys.exit(0 if all_pass else 1)
