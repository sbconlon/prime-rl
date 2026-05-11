"""Spike 5.2: flash_attn_with_kvcache cross-attention against FP32 oracle.

Tests the assumption that `flash_attn_with_kvcache(q, k=None, v=None, ...,
cache_batch_idx=zeros(B), causal=True)` produces cross-attention semantics
(no self-attention from the query's own K/V) with a shared physical cache
across all rows.

Five assertions:

  (1) Outputs match FP32 Python oracle within tolerance.
  (2) k_cache is NOT mutated by the call (byte-equal before/after).
  (3) v_cache is NOT mutated by the call.
  (4) cache_seqlens is NOT mutated by the call.
  (5) Different rows with the same Q but different cache_seqlens produce
      different outputs (sanity check that the per-row cutoff is respected).

Run:
    uv run python scripts/canary/spike_5_2_flash_kvcache_oracle.py

Expects: all checks PASS. Run time ~10s on any post-Ampere GPU; no model load.
"""

from __future__ import annotations

import sys

import torch

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

if not torch.cuda.is_available():
    print("ERROR: flash_attn requires CUDA; no GPU available.", file=sys.stderr)
    sys.exit(1)

try:
    from flash_attn import flash_attn_with_kvcache
except ImportError as e:
    print(f"ERROR: flash_attn not installed: {e}", file=sys.stderr)
    sys.exit(1)


# Shape parameters approximate Qwen2.5-1.5B (the ALFWorld scale target).
# n_q_heads = 16, n_kv_heads = 2 (GQA ratio 8), head_dim = 128.
B = 64                  # number of (position, candidate) rows
S_CACHE = 256           # cache seqlen
N_Q_HEADS = 16
N_KV_HEADS = 2
HEAD_DIM = 128

device = torch.device("cuda")
dtype = torch.bfloat16

print(f"device={device} dtype={dtype}")
print(f"B={B} S_CACHE={S_CACHE} n_q={N_Q_HEADS} n_kv={N_KV_HEADS} d={HEAD_DIM}")


def gqa_expand(kv: torch.Tensor, n_q_heads: int) -> torch.Tensor:
    """Expand a [..., n_kv_heads, head_dim] tensor to [..., n_q_heads, head_dim]
    by repeating each KV head to match GQA grouping (head h_q reads head
    h_q // (n_q_heads // n_kv_heads) of the KV)."""
    n_kv = kv.shape[-2]
    repeat = n_q_heads // n_kv
    assert n_q_heads % n_kv == 0, f"n_q ({n_q_heads}) must be divisible by n_kv ({n_kv})"
    return kv.repeat_interleave(repeat, dim=-2)


def fp32_oracle(
    q: torch.Tensor,                   # [B, 1, n_q, d]  bf16
    k_cache: torch.Tensor,             # [1, S, n_kv, d] bf16
    v_cache: torch.Tensor,             # [1, S, n_kv, d] bf16
    cache_seqlens: torch.Tensor,       # [B]             int32
) -> torch.Tensor:
    """FP32 Python attention reference. Returns [B, 1, n_q, d] in bf16."""
    n_q_heads = q.shape[-2]
    head_dim = q.shape[-1]
    scale = head_dim ** -0.5

    out_rows: list[torch.Tensor] = []
    for b in range(q.shape[0]):
        seqlen = int(cache_seqlens[b].item())
        # All rows index cache batch 0 (shared cache).
        K_b = k_cache[0, :seqlen, :, :].to(torch.float32)      # [seqlen, n_kv, d]
        V_b = v_cache[0, :seqlen, :, :].to(torch.float32)      # [seqlen, n_kv, d]
        K_b = gqa_expand(K_b, n_q_heads)                       # [seqlen, n_q, d]
        V_b = gqa_expand(V_b, n_q_heads)                       # [seqlen, n_q, d]
        Q_b = q[b, 0, :, :].to(torch.float32)                  # [n_q, d]

        # scores[h, s] = (Q_b[h] dot K_b[s, h]) * scale
        # Q_b: [n_q, d]; K_b: [seqlen, n_q, d] -> transpose to [n_q, seqlen, d] for batched matmul.
        K_b_t = K_b.transpose(0, 1)                            # [n_q, seqlen, d]
        V_b_t = V_b.transpose(0, 1)                            # [n_q, seqlen, d]
        scores = (Q_b.unsqueeze(1) @ K_b_t.transpose(-2, -1)).squeeze(1) * scale  # [n_q, seqlen]
        attn = torch.softmax(scores, dim=-1)                   # [n_q, seqlen]
        out_b = (attn.unsqueeze(1) @ V_b_t).squeeze(1)         # [n_q, d]
        out_rows.append(out_b)

    return torch.stack(out_rows, dim=0).unsqueeze(1).to(dtype)  # [B, 1, n_q, d]


def report(label: str, ok: bool, detail: str = "") -> None:
    status = "\033[32mPASS\033[0m" if ok else "\033[31mFAIL\033[0m"
    print(f"  {status}: {label}  {detail}")


def main() -> int:
    torch.manual_seed(42)

    # Per-row cache cutoffs. Pick a mix so we exercise different prefix lengths.
    cache_seqlens_list = [32, 64, 128, 256] * (B // 4)
    cache_seqlens = torch.tensor(cache_seqlens_list, dtype=torch.int32, device=device)

    # All rows share the single physical cache slot at index 0.
    cache_batch_idx = torch.zeros(B, dtype=torch.int32, device=device)

    # Synthetic Q + cache. Small std so softmax doesn\'t saturate.
    q = torch.randn(B, 1, N_Q_HEADS, HEAD_DIM, device=device, dtype=dtype) * 0.1
    k_cache = torch.randn(1, S_CACHE, N_KV_HEADS, HEAD_DIM, device=device, dtype=dtype) * 0.1
    v_cache = torch.randn(1, S_CACHE, N_KV_HEADS, HEAD_DIM, device=device, dtype=dtype) * 0.1

    print()
    print("===== Spike 5.2: flash_attn_with_kvcache vs FP32 oracle =====")
    print()

    # Snapshot for mutation checks.
    k_cache_before = k_cache.detach().clone()
    v_cache_before = v_cache.detach().clone()
    cache_seqlens_before = cache_seqlens.detach().clone()

    # -------------------------------------------------------------------
    # The call under test.
    # -------------------------------------------------------------------
    out_fa = flash_attn_with_kvcache(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        k=None,
        v=None,
        cache_seqlens=cache_seqlens,
        cache_batch_idx=cache_batch_idx,
        causal=True,
    )  # expected: [B, 1, n_q, d]

    all_pass = True

    # -------------------------------------------------------------------
    # Check 1: output shape
    # -------------------------------------------------------------------
    print("--- Check 1: output shape ---")
    expected_shape = (B, 1, N_Q_HEADS, HEAD_DIM)
    shape_ok = out_fa.shape == expected_shape
    report("output shape == [B, 1, n_q, d]", shape_ok, f"got {tuple(out_fa.shape)}")
    all_pass &= shape_ok

    # -------------------------------------------------------------------
    # Checks 2-4: cache + cache_seqlens are NOT mutated
    # -------------------------------------------------------------------
    print()
    print("--- Checks 2-4: no mutation of inputs ---")

    k_ok = torch.equal(k_cache, k_cache_before)
    v_ok = torch.equal(v_cache, v_cache_before)
    seq_ok = torch.equal(cache_seqlens, cache_seqlens_before)

    if not k_ok:
        diff_idx = (k_cache != k_cache_before).nonzero(as_tuple=False)
        report("k_cache unchanged",
               k_ok,
               f"mutated at {diff_idx.shape[0]} positions (first: {diff_idx[0].tolist() if len(diff_idx) else None})")
    else:
        report("k_cache unchanged", k_ok, "byte-equal")
    all_pass &= k_ok

    if not v_ok:
        diff_idx = (v_cache != v_cache_before).nonzero(as_tuple=False)
        report("v_cache unchanged",
               v_ok,
               f"mutated at {diff_idx.shape[0]} positions")
    else:
        report("v_cache unchanged", v_ok, "byte-equal")
    all_pass &= v_ok

    if not seq_ok:
        report("cache_seqlens unchanged",
               seq_ok,
               f"before={cache_seqlens_before.tolist()[:8]}... after={cache_seqlens.tolist()[:8]}...")
    else:
        report("cache_seqlens unchanged", seq_ok, "byte-equal")
    all_pass &= seq_ok

    # -------------------------------------------------------------------
    # Check 5: outputs match FP32 oracle
    # -------------------------------------------------------------------
    print()
    print("--- Check 5: outputs match FP32 oracle ---")
    out_oracle = fp32_oracle(q, k_cache, v_cache, cache_seqlens)

    abs_diff = (out_fa.to(torch.float32) - out_oracle.to(torch.float32)).abs()
    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()

    # bf16 has ~3 decimal digits of precision; per-row over ~32-256 keys
    # gives some accumulated rounding. 5e-3 absolute, 1e-2 relative is the
    # plan\'s pass criterion.
    out_mag = out_oracle.to(torch.float32).abs().mean().item()
    rel = max_abs / max(out_mag, 1e-6)

    parity_ok = max_abs < 5e-3 and rel < 5e-2
    report(
        "FA bf16 ≈ FP32 oracle",
        parity_ok,
        f"max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} rel={rel:.3e} out_mag={out_mag:.3e}",
    )
    all_pass &= parity_ok

    # Per-row diagnostic when failing.
    if not parity_ok:
        per_row_max = abs_diff.flatten(1).max(dim=1).values
        worst = per_row_max.argmax().item()
        print(f"      worst row: {worst}  cache_seqlens[{worst}]={cache_seqlens[worst].item()}  max_diff={per_row_max[worst].item():.3e}")
        per_seqlen = {}
        for b in range(B):
            sl = int(cache_seqlens[b].item())
            per_seqlen.setdefault(sl, []).append(per_row_max[b].item())
        for sl, diffs in sorted(per_seqlen.items()):
            print(f"      cache_seqlens={sl}: max_diff_over_rows={max(diffs):.3e}, n_rows={len(diffs)}")

    # -------------------------------------------------------------------
    # Check 6: per-row cache_seqlens is actually respected
    # (sanity: row b=0 with seqlen=32 should differ from row b=3 with seqlen=256
    # even given identical Q)
    # -------------------------------------------------------------------
    print()
    print("--- Check 6: per-row cache_seqlens respected ---")
    q_same = torch.randn(2, 1, N_Q_HEADS, HEAD_DIM, device=device, dtype=dtype) * 0.1
    q_same[1] = q_same[0]   # identical Q rows
    cs_diff = torch.tensor([32, 256], dtype=torch.int32, device=device)
    cbi_diff = torch.zeros(2, dtype=torch.int32, device=device)

    out_diff = flash_attn_with_kvcache(
        q=q_same,
        k_cache=k_cache,
        v_cache=v_cache,
        cache_seqlens=cs_diff,
        cache_batch_idx=cbi_diff,
        causal=True,
    )
    row_diff = (out_diff[0] - out_diff[1]).abs().max().item()
    ok_6 = row_diff > 1e-3
    report("identical Q with different cache_seqlens → different outputs", ok_6, f"max_diff={row_diff:.3e}")
    all_pass &= ok_6

    # -------------------------------------------------------------------
    print()
    if all_pass:
        print("\033[32m===== Spike 5.2 PASS =====\033[0m")
        return 0
    print("\033[31m===== Spike 5.2 FAIL =====\033[0m")
    return 1


if __name__ == "__main__":
    sys.exit(main())
