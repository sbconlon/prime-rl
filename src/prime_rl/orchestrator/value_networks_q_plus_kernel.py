"""Custom batched K-candidate Q+ forward kernel (plan sections 4.2-4.4).

Replaces the per-position loop in `_build_per_token_value_tensors` (which
calls `forward_q_plus_candidates_at_position` 128 times per request) with a
single batched forward over `B = N_active * K` (position, candidate) pairs.

All B rows share the same physical Q+ prefix cache (produced once per
request by `forward_q_plus_sampled_all_positions`) via
`flash_attn_with_kvcache(cache_batch_idx=zeros(B))`. The candidate\'s own K/V
do not enter attention (passed `k=None, v=None`), giving cross-attention
semantics: Q_cand attends to prefix K/V only -- matching the strict-causal
training-time forward from section 4.1.

The kernel reuses HF Qwen3\'s individual submodules directly (embed_tokens,
input_layernorm, q_proj/k_proj/v_proj/q_norm/o_proj, post_attention_layernorm,
mlp, final norm). Only the attention call is replaced. This preserves the
MultiLoRALinear wrapping that the orchestrator applied to the projections
during ValueNetworkBackbone construction -- so Q+ slot routing works
automatically when the kernel wraps the layer loop in
`_adapter_routing(Q_PLUS_SLOT, B)`.

CPU testing path: when flash_attn is unavailable (no GPU, or pre-Ampere
hardware), the kernel falls back to a FP32 Python attention reference. The
Python path is correct but slow; production always uses flash_attn.

Architecture-specific: this kernel targets HuggingFace Qwen3
(transformers.models.qwen3). Adaptation to Qwen2.5 (no q_norm/k_norm) or
other architectures would require replicating their respective per-layer
forward sequences.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

try:
    from flash_attn import flash_attn_with_kvcache as _flash_attn_with_kvcache
    _HAS_FLASH_ATTN = True
except ImportError:
    _flash_attn_with_kvcache = None  # type: ignore[assignment]
    _HAS_FLASH_ATTN = False


from transformers.models.qwen3.modeling_qwen3 import rotate_half

if TYPE_CHECKING:
    from transformers import DynamicCache
    from prime_rl.orchestrator.value_networks import ValueNetworkBackbone


# ---------------------------------------------------------------------------
# RoPE for a Q-only single-token-per-row batch.
# ---------------------------------------------------------------------------


def _apply_rotary_to_q_only(q: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply RoPE to Q only.

    K side comes from the prefix cache which was already rotated when the
    cache was built by `forward_q_plus_sampled_all_positions` (at training
    time, HF\'s standard rotary path rotates K before storing in cache).

    Args:
      q: [B, n_q_heads, 1, head_dim] -- the queries to rotate.
      cos, sin: [B, 1, head_dim] -- per-row RoPE coefficients at each
        candidate\'s absolute position.

    Returns: rotated q, same shape as input.
    """
    cos = cos.unsqueeze(1)                  # [B, 1, 1, head_dim]
    sin = sin.unsqueeze(1)
    return q * cos + rotate_half(q) * sin


# ---------------------------------------------------------------------------
# Attention calls: flash_attn (production) + Python reference (CPU tests).
# ---------------------------------------------------------------------------


def _attention_flash(
    q_fa: Tensor,           # [B, 1, n_q, head_dim] (FA layout)
    k_cache_fa: Tensor,     # [1, S_cache, n_kv, head_dim]
    v_cache_fa: Tensor,
    cache_seqlens: Tensor,  # [B] int32
    cache_batch_idx: Tensor,  # [B] int32 (all zeros for shared cache)
) -> Tensor:
    """flash_attn_with_kvcache call. Returns [B, 1, n_q, head_dim]."""
    assert _flash_attn_with_kvcache is not None, "flash_attn not available"
    return _flash_attn_with_kvcache(
        q=q_fa,
        k_cache=k_cache_fa,
        v_cache=v_cache_fa,
        k=None,
        v=None,
        cache_seqlens=cache_seqlens,
        cache_batch_idx=cache_batch_idx,
        causal=True,
    )


def _attention_python(
    q_fa: Tensor,           # [B, 1, n_q, head_dim]
    k_cache_fa: Tensor,     # [1, S_cache, n_kv, head_dim]
    v_cache_fa: Tensor,
    cache_seqlens: Tensor,
    cache_batch_idx: Tensor,
    *,
    scaling: float,
) -> Tensor:
    """FP32 Python attention reference. Mirrors flash_attn_with_kvcache with
    k=None, v=None, cache_batch_idx supported. Used in CPU tests."""
    B = q_fa.shape[0]
    n_q = q_fa.shape[2]
    head_dim = q_fa.shape[3]
    n_kv = k_cache_fa.shape[2]
    assert n_q % n_kv == 0
    repeat = n_q // n_kv

    out_rows: list[Tensor] = []
    for b in range(B):
        seqlen = int(cache_seqlens[b].item())
        cache_b = int(cache_batch_idx[b].item())
        K_b = k_cache_fa[cache_b, :seqlen, :, :].to(torch.float32)  # [s, n_kv, d]
        V_b = v_cache_fa[cache_b, :seqlen, :, :].to(torch.float32)
        # GQA expansion.
        K_b = K_b.repeat_interleave(repeat, dim=-2)                # [s, n_q, d]
        V_b = V_b.repeat_interleave(repeat, dim=-2)
        Q_b = q_fa[b, 0, :, :].to(torch.float32)                   # [n_q, d]
        # [n_q, d] x [n_q, d, s] -> [n_q, s]
        K_b_t = K_b.transpose(0, 1)                                # [n_q, s, d]
        V_b_t = V_b.transpose(0, 1)                                # [n_q, s, d]
        scores = (Q_b.unsqueeze(1) @ K_b_t.transpose(-2, -1)).squeeze(1) * scaling
        attn = torch.softmax(scores, dim=-1)                       # [n_q, s]
        out_b = (attn.unsqueeze(1) @ V_b_t).squeeze(1)             # [n_q, d]
        out_rows.append(out_b)
    return torch.stack(out_rows, dim=0).unsqueeze(1).to(q_fa.dtype)


# ---------------------------------------------------------------------------
# The kernel.
# ---------------------------------------------------------------------------


def forward_q_plus_candidates_batched_kernel(
    backbone: "ValueNetworkBackbone",
    prefix_cache: "DynamicCache",
    candidate_token_ids: Tensor,      # [N_active, K] int64
    candidate_positions: Tensor,      # [N_active] integer dtype
    *,
    use_flash_attn: bool | None = None,
) -> Tensor:
    """Run a single batched Q+ forward over all (N_active, K) candidates.

    Args:
      backbone: ValueNetworkBackbone whose base_model is a HF Qwen3 model
        wrapped with MultiLoRALinear at q/k/v/o_proj + gate/up/down_proj.
      prefix_cache: DynamicCache built by
        `backbone.forward_q_plus_sampled_all_positions(full_input_ids)`.
        Shared across all rows via `cache_batch_idx=zeros(B)`.
      candidate_token_ids: [N_active, K] int64. Row j is the K candidate
        token IDs at active completion position j (Phase 5 compact layout).
      candidate_positions: [N_active]. Row j is the absolute trajectory
        position of the candidate: prompt_len + j_local in the full input.
        Must be >= 1 (position 0 is degenerate under strict-causal; we never
        query Q+ there).
      use_flash_attn: force flash-attn path or the FP32 Python fallback.
        Default: auto -- use flash-attn iff available and tensors are on CUDA.

    Returns: [N_active, K] Q+ values (after q_plus_head).
    """
    from prime_rl.orchestrator.value_networks import Q_PLUS_SLOT, _adapter_routing

    assert candidate_token_ids.dim() == 2, (
        f"candidate_token_ids must be [N_active, K], got {tuple(candidate_token_ids.shape)}"
    )
    assert candidate_positions.dim() == 1, (
        f"candidate_positions must be [N_active], got {tuple(candidate_positions.shape)}"
    )
    N_active, K = candidate_token_ids.shape
    assert candidate_positions.shape[0] == N_active, (
        f"candidate_positions length {candidate_positions.shape[0]} != "
        f"N_active {N_active}"
    )
    assert (candidate_positions >= 1).all(), (
        "candidate_positions must be >= 1. Position 0 of Q+ is degenerate "
        "under strict-causal (the candidate would attend to no prefix) and "
        "is never queried by the orchestrator. See plan section 4.2 / Q6."
    )

    if N_active == 0:
        return torch.zeros((0, K), dtype=torch.float32, device=candidate_token_ids.device)

    B = N_active * K
    device = candidate_token_ids.device

    # Flatten to [B] for the model forward; reshape at the end.
    cand_flat = candidate_token_ids.reshape(B).to(dtype=torch.long, device=device)
    pos_flat = (
        candidate_positions.to(torch.int32)
        .unsqueeze(-1)
        .expand(N_active, K)
        .reshape(B)
        .contiguous()
    )
    cache_seqlens = pos_flat.clone()  # strict-causal: row attends to [0, position)
    cache_batch_idx = torch.zeros(B, dtype=torch.int32, device=device)

    # Resolve attention path.
    if use_flash_attn is None:
        use_flash_attn = _HAS_FLASH_ATTN and device.type == "cuda"
    if use_flash_attn and not _HAS_FLASH_ATTN:
        raise RuntimeError("use_flash_attn=True but flash_attn is not installed")

    # Sanity: this kernel doesn\'t handle sliding-window attention. Qwen3-0.6B
    # has full attention on all layers; bail loudly if a future variant turns
    # sliding window on (we\'d need to plumb window_size into flash-attn).
    for layer in backbone.base_model.layers:
        sw = getattr(layer.self_attn, "sliding_window", None)
        assert sw is None, (
            f"layer {layer.self_attn.layer_idx} has sliding_window={sw}; "
            "the K-candidate kernel currently only supports full attention. "
            "Plumb window_size into flash_attn_with_kvcache if needed."
        )

    # Embed.
    h = backbone.base_model.embed_tokens(cand_flat).unsqueeze(1)  # [B, 1, hidden]

    # RoPE cos/sin per row. position_ids shape [B, 1]; rotary_emb returns
    # (cos, sin) shape [B, 1, head_dim].
    rotary_emb = backbone.base_model.rotary_emb
    pos_2d = pos_flat.to(torch.long).unsqueeze(-1)  # [B, 1]
    cos, sin = rotary_emb(h, pos_2d)

    # Layer chain, all routed through Q+ slot (LORA_NUM_TOKENS = [0, B, 0]).
    with _adapter_routing(Q_PLUS_SLOT, B):
        for layer in backbone.base_model.layers:
            attn = layer.self_attn
            head_dim = attn.head_dim
            n_q = attn.q_proj.out_features // head_dim
            n_kv = attn.k_proj.out_features // head_dim

            # ---- Attention block ----
            residual = h
            h_norm = layer.input_layernorm(h)  # [B, 1, hidden]

            # Projections (LoRA-wrapped). k_proj and v_proj are computed
            # because the LoRA module can\'t be skipped, but their outputs
            # are discarded (the attention call uses k=None, v=None).
            Q = attn.q_proj(h_norm)                          # [B, 1, n_q * d]
            _K = attn.k_proj(h_norm)                         # discarded
            _V = attn.v_proj(h_norm)                         # discarded
            del _K, _V

            # Q-norm + RoPE. q_norm is Qwen3-specific (applied per head_dim
            # before RoPE); Qwen2/Qwen2.5 don't have it. Detect and skip.
            Q = Q.view(B, 1, n_q, head_dim)                  # [B, 1, n_q, d]
            if hasattr(attn, "q_norm"):
                Q = attn.q_norm(Q)
            Q = Q.transpose(1, 2)                            # [B, n_q, 1, d]
            Q = _apply_rotary_to_q_only(Q, cos, sin)

            # Convert to FA layout [B, 1, n_q, d].
            q_fa = Q.transpose(1, 2).contiguous()

            # Pull cache in FA layout. HF DynamicCache: [1, n_kv, S, d];
            # FA wants [1, S, n_kv, d].
            cache_layer = prefix_cache.layers[attn.layer_idx]
            k_cache_fa = cache_layer.keys.transpose(1, 2).contiguous()
            v_cache_fa = cache_layer.values.transpose(1, 2).contiguous()

            if use_flash_attn:
                attn_out = _attention_flash(
                    q_fa, k_cache_fa, v_cache_fa, cache_seqlens, cache_batch_idx
                )
            else:
                attn_out = _attention_python(
                    q_fa, k_cache_fa, v_cache_fa, cache_seqlens, cache_batch_idx,
                    scaling=attn.scaling,
                )
            # attn_out: [B, 1, n_q, d]
            attn_out = attn_out.reshape(B, 1, n_q * head_dim)
            attn_out = attn.o_proj(attn_out)                 # [B, 1, hidden]
            h = residual + attn_out

            # ---- MLP block ----
            residual2 = h
            h_norm2 = layer.post_attention_layernorm(h)
            mlp_out = layer.mlp(h_norm2)                     # gate/up/down LoRA-wrapped
            h = residual2 + mlp_out

    # Final norm.
    h = backbone.base_model.norm(h)                          # [B, 1, hidden]

    # Q+ head (returns scalar per row).
    q_plus = backbone.q_plus_head(h[:, 0, :])                # [B]
    return q_plus.view(N_active, K)
