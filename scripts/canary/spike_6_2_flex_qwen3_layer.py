"""Spike 6.2: FlexAttention + GQA + RoPE composition on a real Qwen3 layer.

Builds a tiny Qwen3 decoder layer and verifies that manually replicating
Qwen3's attention path with flex_attention (strict-causal mask_mod) matches
the result of running HF's stock Qwen3Attention.forward with our 4D
additive strict-causal mask.

Four checks (all must PASS):

  (1) FP32 forward parity at a real Qwen3 attention layer with GQA + RoPE
      + q_norm + k_norm + strict-causal mask (atol=1e-5).
  (2) BF16 forward sanity (atol=3e-2, finite).
  (3) Output shape + dtype sanity.
  (4) Padded sequences via mask_mod with per-row length: outputs at valid
      (non-padded) positions match the unpadded run within FlexAttention's
      block-reorder noise budget (atol=1e-5 in FP32). Needed for lever 2.

Run:
    uv run python scripts/canary/spike_6_2_flex_qwen3_layer.py
"""

from __future__ import annotations

import sys

import torch
from transformers import Qwen3Config, Qwen3Model


if not torch.cuda.is_available():
    print("ERROR: spike requires CUDA.", file=sys.stderr)
    sys.exit(1)

print("=" * 72)
print("Spike 6.2: FlexAttention + GQA + RoPE on a real Qwen3 layer")
print("=" * 72)

from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from torch.nn.attention import SDPBackend, sdpa_kernel

flex_attention_c = torch.compile(flex_attention, dynamic=False)


def tiny_qwen3_config() -> Qwen3Config:
    return Qwen3Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=256,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
    )


device = torch.device("cuda")


def build_fp32_model():
    """Build a tiny Qwen3Model in FP32 with deterministic weights."""
    cfg = tiny_qwen3_config()
    torch.manual_seed(0)
    model = Qwen3Model(cfg).to(device=device, dtype=torch.float32).eval()
    # Break zero-init so attention is non-trivial.
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() >= 2:
                p.normal_(mean=0.0, std=0.02)
    return model


def strict_causal_mask_mod(b, h, q_idx, kv_idx):
    return (kv_idx < q_idx) | ((q_idx == 0) & (kv_idx == 0))


def build_sdpa_4d_mask(S, dtype):
    i_idx = torch.arange(S, device=device).unsqueeze(1)
    j_idx = torch.arange(S, device=device).unsqueeze(0)
    keep = (j_idx < i_idx) | ((i_idx == 0) & (j_idx == 0))
    sentinel = torch.finfo(dtype).min
    return torch.where(
        keep,
        torch.zeros((), dtype=dtype, device=device),
        torch.full((), sentinel, dtype=dtype, device=device),
    ).unsqueeze(0).unsqueeze(0)


def apply_rotary(q, k, cos, sin):
    """RoPE: matches HF's apply_rotary_pos_emb_qwen3.

    q, k: [B, H, S, D]
    cos, sin: [B, S, D]
    """
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    from transformers.models.qwen3.modeling_qwen3 import rotate_half
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


def manual_attn_via_flex(attn, rotary_emb, h, position_ids, mask_mod):
    """Replicate one Qwen3Attention layer using flex_attention. Returns
    post-o_proj output. Takes rotary_emb explicitly (no globals)."""
    B, S, _ = h.shape
    n_q = attn.q_proj.out_features // attn.head_dim
    n_kv = attn.k_proj.out_features // attn.head_dim
    d = attn.head_dim

    Q = attn.q_proj(h).view(B, S, n_q, d).transpose(1, 2)
    K = attn.k_proj(h).view(B, S, n_kv, d).transpose(1, 2)
    V = attn.v_proj(h).view(B, S, n_kv, d).transpose(1, 2)

    if hasattr(attn, "q_norm"):
        Q = attn.q_norm(Q)
    if hasattr(attn, "k_norm"):
        K = attn.k_norm(K)

    cos, sin = rotary_emb(h, position_ids)
    Q, K = apply_rotary(Q, K, cos, sin)

    n_rep = n_q // n_kv
    K = K.repeat_interleave(n_rep, dim=1)
    V = V.repeat_interleave(n_rep, dim=1)

    block_mask = create_block_mask(
        mask_mod, B=None, H=None, Q_LEN=S, KV_LEN=S, device=device,
    )

    attn_out = flex_attention_c(Q, K, V, block_mask=block_mask)
    attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, n_q * d)
    return attn.o_proj(attn_out)


# ---------------------------------------------------------------------------
# Setup: one FP32 model that lives for the whole spike
# ---------------------------------------------------------------------------

S = 128
B = 1
print(f"\nShapes: B={B}, S={S}, GQA=4q/2kv, head_dim=16, hidden=64")

model_fp32 = build_fp32_model()
attn_fp32 = model_fp32.layers[0].self_attn

torch.manual_seed(1)
h32 = torch.randn(B, S, model_fp32.config.hidden_size, dtype=torch.float32, device=device)
position_ids = torch.arange(S, dtype=torch.long, device=device).unsqueeze(0)

# ---------------------------------------------------------------------------
# Check 1: FP32 forward parity
# ---------------------------------------------------------------------------

print("\n[1] FP32 forward parity (real Qwen3 layer)...")
sdpa_mask_fp32 = build_sdpa_4d_mask(S, torch.float32)

cos, sin = model_fp32.rotary_emb(h32, position_ids)
with sdpa_kernel(SDPBackend.MATH):
    ref_out, _ = attn_fp32(
        hidden_states=h32,
        position_embeddings=(cos, sin),
        attention_mask=sdpa_mask_fp32,
    )

test_out = manual_attn_via_flex(
    attn_fp32, model_fp32.rotary_emb, h32, position_ids, strict_causal_mask_mod,
)

abs_diff = (test_out - ref_out).abs()
max_abs = abs_diff.max().item()
mean_abs = abs_diff.mean().item()
print(f"  test_out.shape = {tuple(test_out.shape)}, dtype={test_out.dtype}")
print(f"  max_abs_diff   = {max_abs:.3e}")
print(f"  mean_abs_diff  = {mean_abs:.3e}")
print(f"  out range      = [{ref_out.abs().min().item():.3e}, {ref_out.abs().max().item():.3e}]")
fp32_pass = max_abs < 1e-5
print(f"  {'PASS' if fp32_pass else 'FAIL'} [1]: FP32 parity at atol=1e-5")

# ---------------------------------------------------------------------------
# Check 2: BF16 forward sanity. Build a fresh BF16 model from the same
# initial RNG seed so its weights are FP32-then-cast-to-BF16 of the same
# values as model_fp32 -- we do NOT round-trip those back to FP32 anywhere.
# ---------------------------------------------------------------------------

print("\n[2] BF16 forward sanity...")
model_bf16 = build_fp32_model().to(torch.bfloat16)
attn_bf16 = model_bf16.layers[0].self_attn

h16 = h32.to(torch.bfloat16)
sdpa_mask_bf16 = build_sdpa_4d_mask(S, torch.bfloat16)
cos_bf16, sin_bf16 = model_bf16.rotary_emb(h16, position_ids)

with sdpa_kernel(SDPBackend.MATH):
    ref_out_bf16, _ = attn_bf16(
        hidden_states=h16,
        position_embeddings=(cos_bf16, sin_bf16),
        attention_mask=sdpa_mask_bf16,
    )

test_out_bf16 = manual_attn_via_flex(
    attn_bf16, model_bf16.rotary_emb, h16, position_ids, strict_causal_mask_mod,
)

bf16_abs = (test_out_bf16.float() - ref_out_bf16.float()).abs()
bf16_max = bf16_abs.max().item()
bf16_mean = bf16_abs.mean().item()
bf16_finite = torch.isfinite(test_out_bf16).all().item()
print(f"  max_abs_diff   = {bf16_max:.3e}")
print(f"  mean_abs_diff  = {bf16_mean:.3e}")
print(f"  isfinite       = {bf16_finite}")
bf16_pass = bf16_finite and bf16_max < 3e-2
print(f"  {'PASS' if bf16_pass else 'FAIL'} [2]: BF16 within atol=3e-2 + finite")

# ---------------------------------------------------------------------------
# Check 3: Output shape + dtype sanity
# ---------------------------------------------------------------------------

print("\n[3] Output shape + dtype sanity...")
shape_ok = test_out.shape == (B, S, model_fp32.config.hidden_size)
dtype_ok = test_out_bf16.dtype == torch.bfloat16
print(f"  test_out.shape      = {tuple(test_out.shape)}  ({'OK' if shape_ok else 'BAD'})")
print(f"  test_out_bf16.dtype = {test_out_bf16.dtype}  ({'OK' if dtype_ok else 'BAD'})")
shape_pass = shape_ok and dtype_ok
print(f"  {'PASS' if shape_pass else 'FAIL'} [3]")

# ---------------------------------------------------------------------------
# Check 4: Padded sequences via per-row length mask_mod (lever 2 dep).
# Reuse model_fp32 (no BF16 round-trip).
# ---------------------------------------------------------------------------

print("\n[4] Padded-sequence mask_mod (lever 2 dependency)...")
PAD = 16
S_padded = S + PAD
lens = torch.tensor([S], dtype=torch.long, device=device)  # all rows valid up to S


def strict_causal_lens_mask_mod(b, h, q_idx, kv_idx):
    in_sample = (q_idx < lens[b]) & (kv_idx < lens[b])
    sc = (kv_idx < q_idx) | ((q_idx == 0) & (kv_idx == 0))
    return in_sample & sc


h_padded = torch.cat(
    [h32, torch.zeros(B, PAD, model_fp32.config.hidden_size, device=device, dtype=torch.float32)],
    dim=1,
)
position_ids_padded = torch.arange(S_padded, dtype=torch.long, device=device).unsqueeze(0)

test_out_padded = manual_attn_via_flex(
    attn_fp32, model_fp32.rotary_emb, h_padded, position_ids_padded, strict_causal_lens_mask_mod,
)
valid_diff = (test_out_padded[:, :S] - test_out).abs()
valid_max = valid_diff.max().item()
valid_mean = valid_diff.mean().item()
print(f"  test_out_padded.shape   = {tuple(test_out_padded.shape)}")
print(f"  max_abs_diff (valid)    = {valid_max:.3e}")
print(f"  mean_abs_diff (valid)   = {valid_mean:.3e}")
# Block tiling at S=128 vs S=144 may reorder FP32 accumulations slightly;
# allow up to atol=1e-5 (still well below BF16 noise).
pad_tol = 1e-5
pad_pass = valid_max < pad_tol
print(f"  {'PASS' if pad_pass else 'FAIL'} [4]: padded == unpadded at valid positions (atol={pad_tol})")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("\n" + "=" * 72)
all_pass = fp32_pass and bf16_pass and shape_pass and pad_pass
if all_pass:
    print("Spike 6.2: ALL CHECKS PASS  -- FlexAttention composes correctly with Qwen3.")
    print("Green light to implement forward_q_plus_sampled_all_positions_flex_kernel.")
else:
    print("Spike 6.2: FAILED.")
print("=" * 72)
sys.exit(0 if all_pass else 1)
