"""Spike 6.2: FlexAttention + GQA + RoPE composition on a real Qwen3 layer.

Builds a tiny Qwen3 decoder layer and verifies that manually replicating
Qwen3's attention path with flex_attention (strict-causal mask_mod) matches
the result of running HF's stock Qwen3Attention.forward with our 4D
additive strict-causal mask.

This is the integration test that catches GQA bugs, RoPE bugs, and
q_norm/k_norm misordering before we commit to a full kernel
implementation. If 6.1 was "does FlexAttention work at all?", this is
"does it compose correctly with all the Qwen3-specific bits?".

Four checks (all must PASS):

  (1) FP32 forward parity at a real Qwen3 attention layer with GQA + RoPE
      + q_norm + k_norm + strict-causal mask (atol=1e-5).
  (2) BF16 forward sanity (atol=3e-2).
  (3) Output shape and finiteness.
  (4) Padded sequences via mask_mod with per-row length: outputs at padded
      positions are unchanged when we extend the sequence with zero tokens
      and pad the mask accordingly. (Needed for lever 2 micro-batching.)

Run:
    uv run python scripts/canary/spike_6_2_flex_qwen3_layer.py
"""

from __future__ import annotations

import sys
import time

import torch
import torch.nn.functional as F
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


# Tiny Qwen3 config with the same 2:1 GQA ratio as Qwen3-0.6B.
def tiny_qwen3_config() -> Qwen3Config:
    return Qwen3Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
    )


device = torch.device("cuda")
torch.manual_seed(0)


def build_model_layer(dtype):
    """Build a tiny Qwen3Model and return (model, layer0_attention)."""
    cfg = tiny_qwen3_config()
    model = Qwen3Model(cfg).to(device=device, dtype=dtype).eval()
    # Break zero-init so attention is non-trivial.
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() >= 2:
                p.normal_(mean=0.0, std=0.02)
    return model, model.layers[0].self_attn


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
    ).unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]


def apply_rotary(q, k, cos, sin):
    """Reference RoPE: matches HF's apply_rotary_pos_emb_qwen3 modulo shape.

    q, k: [B, H, S, D]
    cos, sin: [B, S, D]
    """
    # Following HF's pattern.
    cos = cos.unsqueeze(1)  # [B, 1, S, D]
    sin = sin.unsqueeze(1)
    from transformers.models.qwen3.modeling_qwen3 import rotate_half
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


def manual_attn_via_flex(attn, h, position_ids, mask_mod, dtype):
    """Replicate one Qwen3Attention layer using flex_attention for the
    attention call. h: [B, S, hidden]. Returns post-o_proj output."""
    B, S, hidden = h.shape
    n_q = attn.q_proj.out_features // attn.head_dim
    n_kv = attn.k_proj.out_features // attn.head_dim
    d = attn.head_dim

    Q = attn.q_proj(h).view(B, S, n_q, d).transpose(1, 2)  # [B, n_q, S, d]
    K = attn.k_proj(h).view(B, S, n_kv, d).transpose(1, 2)  # [B, n_kv, S, d]
    V = attn.v_proj(h).view(B, S, n_kv, d).transpose(1, 2)  # [B, n_kv, S, d]

    if hasattr(attn, "q_norm"):
        Q = attn.q_norm(Q)
    if hasattr(attn, "k_norm"):
        K = attn.k_norm(K)

    # RoPE: rotary_emb returns cos, sin shape [B, S, head_dim].
    # Use the module's rotary_emb (lives on model.rotary_emb, not on the attn).
    rotary_emb = model.rotary_emb
    cos, sin = rotary_emb(h, position_ids)
    Q, K = apply_rotary(Q, K, cos, sin)

    # GQA: repeat K, V to match Q heads count.
    n_rep = n_q // n_kv
    K = K.repeat_interleave(n_rep, dim=1)
    V = V.repeat_interleave(n_rep, dim=1)

    # Build the block mask once per S.
    block_mask = create_block_mask(
        mask_mod, B=None, H=None, Q_LEN=S, KV_LEN=S, device=device,
    )

    attn_out = flex_attention_c(Q, K, V, block_mask=block_mask)
    # attn_out: [B, n_q, S, d] -> [B, S, n_q*d]
    attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, n_q * d)
    return attn.o_proj(attn_out)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

S = 128
B = 1
print(f"\nShapes: B={B}, S={S}, GQA=4q/2kv, head_dim=16, hidden=64")

# ---------------------------------------------------------------------------
# Check 1: FP32 forward parity
# ---------------------------------------------------------------------------

print("\n[1] FP32 forward parity (real Qwen3 layer)...")
model, attn = build_model_layer(torch.float32)

torch.manual_seed(1)
h32 = torch.randn(B, S, model.config.hidden_size, dtype=torch.float32, device=device)
position_ids = torch.arange(S, dtype=torch.long, device=device).unsqueeze(0)
sdpa_mask_fp32 = build_sdpa_4d_mask(S, torch.float32)

# Path A: HF's self_attn.forward with a 4D strict-causal mask.
# Need to get position_embeddings (cos, sin) externally.
cos, sin = model.rotary_emb(h32, position_ids)
with sdpa_kernel(SDPBackend.MATH):
    ref_out, _ = attn(
        hidden_states=h32,
        position_embeddings=(cos, sin),
        attention_mask=sdpa_mask_fp32,
    )

# Path B: manual FlexAttention.
test_out = manual_attn_via_flex(attn, h32, position_ids, strict_causal_mask_mod, torch.float32)

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
# Check 2: BF16 forward sanity
# ---------------------------------------------------------------------------

print("\n[2] BF16 forward sanity...")
model_bf16, attn_bf16 = build_model_layer(torch.bfloat16)
# Copy weights from the FP32 model so we're comparing apples to apples.
with torch.no_grad():
    for p_dst, p_src in zip(model_bf16.parameters(), model.parameters()):
        p_dst.copy_(p_src.to(torch.bfloat16))

h16 = h32.to(torch.bfloat16)
sdpa_mask_bf16 = build_sdpa_4d_mask(S, torch.bfloat16)
cos_bf16, sin_bf16 = model_bf16.rotary_emb(h16, position_ids)

with sdpa_kernel(SDPBackend.MATH):
    ref_out_bf16, _ = attn_bf16(
        hidden_states=h16,
        position_embeddings=(cos_bf16, sin_bf16),
        attention_mask=sdpa_mask_bf16,
    )

# Manual flex path -- swap the module global since manual_attn_via_flex uses it.
model = model_bf16
test_out_bf16 = manual_attn_via_flex(attn_bf16, h16, position_ids, strict_causal_mask_mod, torch.bfloat16)

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
# Check 3: Output shape sanity
# ---------------------------------------------------------------------------

print("\n[3] Output shape + dtype sanity...")
shape_ok = test_out.shape == (B, S, model_bf16.config.hidden_size)
dtype_ok = test_out_bf16.dtype == torch.bfloat16
print(f"  test_out.shape   = {tuple(test_out.shape)}  ({'OK' if shape_ok else 'BAD'})")
print(f"  test_out_bf16.dtype = {test_out_bf16.dtype}  ({'OK' if dtype_ok else 'BAD'})")
shape_pass = shape_ok and dtype_ok
print(f"  {'PASS' if shape_pass else 'FAIL'} [3]")

# ---------------------------------------------------------------------------
# Check 4: Padded sequences via per-row length mask_mod (relevant to lever 2)
# ---------------------------------------------------------------------------

print("\n[4] Padded-sequence mask_mod (lever 2 dependency)...")
# Extend the sequence by padding zeros at the end; build a mask_mod that
# combines strict-causal with per-row length capping. Compare outputs at
# the valid (non-padded) positions vs the original short-sequence outputs.
PAD = 16
S_padded = S + PAD
true_len = S  # all rows have the same length here (B=1)
lens = torch.tensor([true_len], dtype=torch.long, device=device)


def strict_causal_lens_mask_mod(b, h, q_idx, kv_idx):
    in_sample = (q_idx < lens[b]) & (kv_idx < lens[b])
    sc = (kv_idx < q_idx) | ((q_idx == 0) & (kv_idx == 0))
    return in_sample & sc


h_padded = torch.cat([h32, torch.zeros(B, PAD, model.config.hidden_size, device=device, dtype=torch.float32)], dim=1)
position_ids_padded = torch.arange(S_padded, dtype=torch.long, device=device).unsqueeze(0)
# Reload the FP32 model for this check.
model = build_model_layer(torch.float32)[0]
with torch.no_grad():
    for p_dst, p_src in zip(model.parameters(), model_bf16.parameters()):
        p_dst.copy_(p_src.to(torch.float32))
attn = model.layers[0].self_attn

test_out_padded = manual_attn_via_flex(
    attn, h_padded, position_ids_padded, strict_causal_lens_mask_mod, torch.float32,
)
# Compare the non-padded portion against the original FP32 test_out.
valid_diff = (test_out_padded[:, :S] - test_out).abs()
valid_max = valid_diff.max().item()
print(f"  test_out_padded.shape   = {tuple(test_out_padded.shape)}")
print(f"  max_abs_diff (valid)    = {valid_max:.3e}")
pad_pass = valid_max < 1e-5
print(f"  {'PASS' if pad_pass else 'FAIL'} [4]: padded == unpadded at valid positions (atol=1e-5)")

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
