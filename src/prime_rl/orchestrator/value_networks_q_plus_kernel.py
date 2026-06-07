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

_LOGGED_PATH = False  # set to True after first kernel call logs the chosen path


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


# flash_attn_with_kvcache launches a CUDA grid whose batch dim is capped at
# 65535 (cudaGridDim.y/z). At ALFWorld scale (N_active=8000, K=32) the flat
# batch B = 256k blows past that. Chunk B internally; the layer chain stays
# unchunked because projections/MLP on B=256k are fine memory-wise on A100s.
_FLASH_MAX_BATCH = 32768


def _attention_flash(
    q_fa: Tensor,           # [B, 1, n_q, head_dim] (FA layout)
    k_cache_fa: Tensor,     # [1, S_cache, n_kv, head_dim]
    v_cache_fa: Tensor,
    cache_seqlens: Tensor,  # [B] int32
    cache_batch_idx: Tensor,  # [B] int32 (all zeros for shared cache)
) -> Tensor:
    """flash_attn_with_kvcache call. Returns [B, 1, n_q, head_dim].

    Chunks along B when B > _FLASH_MAX_BATCH to keep flash_attn's grid
    dimension under the CUDA 65535 cap.
    """
    assert _flash_attn_with_kvcache is not None, "flash_attn not available"
    B = q_fa.shape[0]
    if B <= _FLASH_MAX_BATCH:
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
    outs: list[Tensor] = []
    for start in range(0, B, _FLASH_MAX_BATCH):
        end = min(start + _FLASH_MAX_BATCH, B)
        outs.append(
            _flash_attn_with_kvcache(
                q=q_fa[start:end],
                k_cache=k_cache_fa,
                v_cache=v_cache_fa,
                k=None,
                v=None,
                cache_seqlens=cache_seqlens[start:end],
                cache_batch_idx=cache_batch_idx[start:end],
                causal=True,
            )
        )
    return torch.cat(outs, dim=0)


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

    # One-shot log so we can confirm which attention path is being used at
    # runtime. Set _LOGGED at module scope after first call.
    global _LOGGED_PATH
    if not _LOGGED_PATH:
        import logging
        logging.getLogger("prime_rl.advantage_server").warning(
            f"forward_q_plus_candidates_batched: use_flash_attn={use_flash_attn}, "
            f"_HAS_FLASH_ATTN={_HAS_FLASH_ATTN}, device={device}, N_active={N_active}, K={K}"
        )
        _LOGGED_PATH = True

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
# ---------------------------------------------------------------------------
# FlexAttention all-positions Q+ kernel (lever 1 of the AdvTrainer perf pass).
# ---------------------------------------------------------------------------
#
# Replaces the SDPA Math backend path used by `_forward_all_positions` when
# slot == Q_PLUS_SLOT. The 4D additive strict-causal mask that path uses
# kicks SDPA off Flash backend onto Math, which materializes the full S x S
# attention probabilities at every layer and saves them for backward --
# that's ~3.6 GB/sample at S=2048 in BF16 across 28 layers, which OOMs the
# AdvTrainer at batch sizes typical of the canary.
#
# FlexAttention JIT-compiles a Triton kernel that respects strict-causal
# via a mask_mod function without materializing the score matrix. The
# strict-causal mask_mod is:
#
#     M[i, j] = 0       if j <  i  (attend)
#     M[i, j] = -inf    if j >= i  (mask out)
#     M[0, 0] = 0       (position-0 self-attend; avoids NaN softmax row)
#
# Spike 6.2 validated FP32 forward parity = 7.45e-9 (essentially zero)
# against HF's stock self_attn with the equivalent 4D additive mask.

try:
    from torch.nn.attention.flex_attention import (
        create_block_mask,
        flex_attention,
    )
    _HAS_FLEX_ATTN = True
    # dynamic=True is REQUIRED: the AdvTrainer per-sample backward feeds
    # variable-length sequences, one sample at a time. With dynamic=False,
    # torch.compile generates a new kernel per distinct seq_len; after 8
    # recompiles dynamo hits its config.recompile_limit and silently falls
    # back to the eager unfused flex_attention path -- which materializes
    # the SxS score matrix at every layer, exactly the path we built this
    # kernel to escape. Once that fallback is cached, the perf win is
    # gone for the rest of the run. dynamic=True generates one kernel that
    # handles all shapes; marginally slower per call than a shape-specialized
    # kernel, but no recompilation cascade.
    _flex_attention_compiled = torch.compile(flex_attention, dynamic=True)
except ImportError:
    create_block_mask = None  # type: ignore[assignment]
    flex_attention = None  # type: ignore[assignment]
    _HAS_FLEX_ATTN = False
    _flex_attention_compiled = None  # type: ignore[assignment]


# Cache strict-causal block masks by (seq_len, device_str). FlexAttention's
# block_mask is a heavy object (precomputed sparsity pattern); reusing it
# across calls is essential. AdvTrainer training batches all share the
# same seq_len after packing, so the cache hit rate is ~100%.
_BLOCK_MASK_CACHE: dict[tuple[int, str], object] = {}


def _strict_causal_mask_mod(b, h, q_idx, kv_idx):
    # Position-0 self-attend handled explicitly to avoid a fully-masked
    # softmax row at q_idx=0 (would produce NaN).
    return (kv_idx < q_idx) | ((q_idx == 0) & (kv_idx == 0))


def _get_strict_causal_block_mask(seq_len: int, device: torch.device):
    """Build or fetch a cached strict-causal FlexAttention block_mask."""
    key = (seq_len, str(device))
    cached = _BLOCK_MASK_CACHE.get(key)
    if cached is None:
        cached = create_block_mask(
            _strict_causal_mask_mod,
            B=None, H=None,
            Q_LEN=seq_len, KV_LEN=seq_len,
            device=device,
        )
        _BLOCK_MASK_CACHE[key] = cached
    return cached


def _apply_rotary_qk(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """Apply RoPE to both Q and K. Mirrors HF's apply_rotary_pos_emb_qwen3.

    q, k: [B, H, S, D]
    cos, sin: [B, S, D]
    """
    cos = cos.unsqueeze(1)  # [B, 1, S, D]
    sin = sin.unsqueeze(1)
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


def forward_q_plus_sampled_all_positions_flex_kernel(
    backbone: "ValueNetworkBackbone",
    input_ids: Tensor,
) -> tuple[Tensor, "DynamicCache"]:
    """All-positions Q+ forward via FlexAttention with strict-causal mask_mod.

    Memory-efficient and ~40x faster than the SDPA Math fallback at S=2048
    in BF16 (spike 6.1 measurement on A100 MIG slice).

    Returns the same (q_plus_seq [B, S], cache) contract as the SDPA path
    so the AdvSrv K-candidate kernel can consume the cache unchanged.
    Cache layout: HF's [B, n_kv, S, head_dim] per layer.
    """
    from prime_rl.orchestrator.value_networks import Q_PLUS_SLOT, _adapter_routing
    from transformers import DynamicCache

    assert _HAS_FLEX_ATTN, "FlexAttention not available (needs PyTorch >= 2.5)"
    assert input_ids.dim() == 2, f"input_ids must be [B, S], got {tuple(input_ids.shape)}"
    B, S = input_ids.shape
    device = input_ids.device

    block_mask = _get_strict_causal_block_mask(S, device)

    # Embed + RoPE coefficients (RoPE shared across layers).
    h = backbone.base_model.embed_tokens(input_ids)  # [B, S, hidden]
    position_ids = torch.arange(S, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)
    cos, sin = backbone.base_model.rotary_emb(h, position_ids)

    cache = DynamicCache()

    # All layer forwards route through Q+ slot.
    with _adapter_routing(Q_PLUS_SLOT, B * S):
        for layer_idx, layer in enumerate(backbone.base_model.layers):
            attn = layer.self_attn
            d = attn.head_dim
            n_q = attn.q_proj.out_features // d
            n_kv = attn.k_proj.out_features // d

            # ---- Attention block ----
            residual = h
            h_norm = layer.input_layernorm(h)

            Q = attn.q_proj(h_norm).view(B, S, n_q, d).transpose(1, 2)     # [B, n_q,  S, d]
            K = attn.k_proj(h_norm).view(B, S, n_kv, d).transpose(1, 2)    # [B, n_kv, S, d]
            V = attn.v_proj(h_norm).view(B, S, n_kv, d).transpose(1, 2)    # [B, n_kv, S, d]

            if hasattr(attn, "q_norm"):
                Q = attn.q_norm(Q)
            if hasattr(attn, "k_norm"):
                K = attn.k_norm(K)

            Q, K = _apply_rotary_qk(Q, K, cos, sin)

            # Cache contract: store K/V pre-GQA-expansion in HF's
            # [B, n_kv, S, d] layout so the K-candidate kernel
            # (forward_q_plus_candidates_batched) can read them unchanged.
            # Make K/V contiguous before cache store -- the K-candidate kernel calls
            # .contiguous() on cache reads; if our stored tensors are strided views
            # (from the transpose above) that contiguous() must materialize.
            cache.update(K.contiguous(), V.contiguous(), layer_idx)

            # GQA expansion for the attention computation.
            n_rep = n_q // n_kv
            K_full = K.repeat_interleave(n_rep, dim=1)                     # [B, n_q, S, d]
            V_full = V.repeat_interleave(n_rep, dim=1)

            attn_out = _flex_attention_compiled(Q, K_full, V_full, block_mask=block_mask)
            # attn_out: [B, n_q, S, d] -> [B, S, n_q*d]
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, n_q * d)
            attn_out = attn.o_proj(attn_out)

            h = residual + attn_out

            # ---- MLP block ----
            residual2 = h
            h = layer.post_attention_layernorm(h)
            h = residual2 + layer.mlp(h)

    h = backbone.base_model.norm(h)
    q_plus_seq = backbone.q_plus_head(h)  # [B, S]
    return q_plus_seq, cache


# ---------------------------------------------------------------------------
# Action-level ARM: batched multi-token Q+ over A(o) off a shared o-cache
# (Phase 8 -- the standard-causal continuation kernel; DQ8.3).
# ---------------------------------------------------------------------------
#
# Contrast with forward_q_plus_candidates_batched_kernel above: that one is
# single-token cross-attention (k=None, strict-causal) -- a token-level
# artifact. This one is MULTI-TOKEN, STANDARD-causal: each action writes its
# own K/V (k=action_K, v=action_V) and each action token attends to o + earlier
# action tokens + itself; Q+ is read at the action terminal. The o-prefill is
# computed once (by the caller) into a [1, ...] cache and shared across the |A|
# actions by tiling it into the per-row k_cache (flash appends each row's action
# K/V to its own tiled slot -- cache_batch_idx=zeros cannot be used here because
# appending writes per-row and would collide on a single shared slot). The
# saving vs the naive path is the o-PREFILL: one o forward + one batched action
# continuation, not |A| full [o, a] forwards.
#
# CPU testing: when flash_attn is unavailable / pre-Ampere, an FP32 Python
# attention reference computes the identical math (parity-gated against the naive
# forward_q_plus_action oracle). Production on Ampere uses flash_attn.


def _action_attention_python(
    Q: Tensor,       # [A, n_q, A_max, d]
    K: Tensor,       # [A, n_kv, A_max, d]
    V: Tensor,       # [A, n_kv, A_max, d]
    o_K: Tensor,     # [1, n_kv, o_len, d]  (HF cache layout; rotated at prefill)
    o_V: Tensor,     # [1, n_kv, o_len, d]
    o_len: int,
    *,
    scaling: float,
) -> Tensor:
    """FP32 Python reference for the standard-causal action continuation.

    Each action token (query i, absolute position o_len+i) attends to the full o
    prefix (o_len keys) plus action keys 0..i (causal, self-inclusive). Returns
    [A, n_q, A_max, d]. Padded action positions (i >= true length) are computed
    but ignored by the caller's terminal read; under the causal mask they never
    influence valid (<= terminal) positions.
    """
    A, n_q, A_max, d = Q.shape
    n_kv = K.shape[1]
    repeat = n_q // n_kv
    L = o_len + A_max
    qpos = (o_len + torch.arange(A_max, device=Q.device)).view(1, A_max, 1)
    kpos = torch.arange(L, device=Q.device).view(1, 1, L)
    masked = kpos > qpos  # True where the key is in the future -> masked out
    out_rows: list[Tensor] = []
    for r in range(A):
        Kr = torch.cat([o_K[0], K[r]], dim=1).to(torch.float32)  # [n_kv, L, d]
        Vr = torch.cat([o_V[0], V[r]], dim=1).to(torch.float32)
        Kr = Kr.repeat_interleave(repeat, dim=0)                 # [n_q, L, d]
        Vr = Vr.repeat_interleave(repeat, dim=0)
        Qr = Q[r].to(torch.float32)                              # [n_q, A_max, d]
        scores = (Qr @ Kr.transpose(-2, -1)) * scaling          # [n_q, A_max, L]
        scores = scores.masked_fill(masked, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out_rows.append(attn @ Vr)                               # [n_q, A_max, d]
    return torch.stack(out_rows, dim=0).to(Q.dtype)             # [A, n_q, A_max, d]


def _action_attention_flash(
    Q: Tensor,       # [A, n_q, A_max, d]
    K: Tensor,       # [A, n_kv, A_max, d]
    V: Tensor,       # [A, n_kv, A_max, d]
    o_K: Tensor,     # [1, n_kv, o_len, d]
    o_V: Tensor,     # [1, n_kv, o_len, d]
    o_len: int,
) -> Tensor:
    """flash_attn_with_kvcache standard-causal continuation. Returns [A, n_q, A_max, d].

    The shared o-cache is tiled across the A rows into per-row k/v caches sized
    [A, o_len + A_max, n_kv, d]; flash appends each row's action K/V at
    cache_seqlens=o_len and computes causal attention (GQA handled internally).
    """
    assert _flash_attn_with_kvcache is not None, "flash_attn not available"
    A, n_q, A_max, d = Q.shape
    n_kv = K.shape[1]
    q_fa = Q.transpose(1, 2).contiguous()  # [A, A_max, n_q, d]
    k_fa = K.transpose(1, 2).contiguous()  # [A, A_max, n_kv, d]
    v_fa = V.transpose(1, 2).contiguous()
    o_K_fa = o_K.transpose(1, 2)           # [1, o_len, n_kv, d]
    o_V_fa = o_V.transpose(1, 2)
    k_cache = q_fa.new_zeros((A, o_len + A_max, n_kv, d))
    v_cache = q_fa.new_zeros((A, o_len + A_max, n_kv, d))
    k_cache[:, :o_len] = o_K_fa.expand(A, o_len, n_kv, d)
    v_cache[:, :o_len] = o_V_fa.expand(A, o_len, n_kv, d)
    cache_seqlens = torch.full((A,), o_len, dtype=torch.int32, device=Q.device)
    out = _flash_attn_with_kvcache(
        q=q_fa,
        k_cache=k_cache,
        v_cache=v_cache,
        k=k_fa,
        v=v_fa,
        cache_seqlens=cache_seqlens,
        causal=True,
    )  # [A, A_max, n_q, d]
    return out.transpose(1, 2)  # [A, n_q, A_max, d]


def forward_q_plus_action_batched_kernel(
    backbone: "ValueNetworkBackbone",
    o_cache: "DynamicCache",
    action_ids: Tensor,      # [A, A_max] int64, right-padded admissible-action suffixes
    action_lens: Tensor,     # [A] true lengths (read Q+ at each terminal)
    *,
    use_flash_attn: bool | None = None,
) -> Tensor:
    """Batched Q+(o, a) over the admissible set off a single shared o-prefill cache.

    Args:
      backbone: ValueNetworkBackbone (HF Qwen2/Qwen3 base wrapped with MultiLoRA).
      o_cache: DynamicCache from ONE standard-causal prefill of o under the Q+
        adapter (batch dim 1). Shared (tiled) across the A actions.
      action_ids: [A, A_max] right-padded action token suffixes (terminator incl.).
      action_lens: [A] true (unpadded) action lengths.
      use_flash_attn: force flash / Python fallback. Default: flash iff available
        and on CUDA.

    Returns: [A] Q+(o, a) for each admissible action.
    """
    from prime_rl.orchestrator.value_networks import Q_PLUS_SLOT, _adapter_routing

    A, A_max = action_ids.shape
    device = action_ids.device
    if A == 0:
        return torch.zeros((0,), dtype=torch.float32, device=device)

    o_len = o_cache.layers[0].keys.shape[-2]  # HF layout [1, n_kv, o_len, d]

    if use_flash_attn is None:
        use_flash_attn = _HAS_FLASH_ATTN and device.type == "cuda"
    if use_flash_attn and not _HAS_FLASH_ATTN:
        raise RuntimeError("use_flash_attn=True but flash_attn is not installed")

    # Sliding-window guard (mirrors the candidate kernel).
    for layer in backbone.base_model.layers:
        sw = getattr(layer.self_attn, "sliding_window", None)
        assert sw is None, (
            f"layer {layer.self_attn.layer_idx} has sliding_window={sw}; the action "
            "Q+ kernel only supports full attention."
        )

    h = backbone.base_model.embed_tokens(action_ids)  # [A, A_max, hidden]
    position_ids = (o_len + torch.arange(A_max, device=device)).unsqueeze(0).expand(A, -1)
    cos, sin = backbone.base_model.rotary_emb(h, position_ids)  # [A, A_max, d]

    with _adapter_routing(Q_PLUS_SLOT, A * A_max):
        for layer in backbone.base_model.layers:
            attn = layer.self_attn
            d = attn.head_dim
            n_q = attn.q_proj.out_features // d
            n_kv = attn.k_proj.out_features // d

            residual = h
            h_norm = layer.input_layernorm(h)
            Q = attn.q_proj(h_norm).view(A, A_max, n_q, d).transpose(1, 2)    # [A, n_q,  A_max, d]
            K = attn.k_proj(h_norm).view(A, A_max, n_kv, d).transpose(1, 2)   # [A, n_kv, A_max, d]
            V = attn.v_proj(h_norm).view(A, A_max, n_kv, d).transpose(1, 2)
            if hasattr(attn, "q_norm"):
                Q = attn.q_norm(Q)
            if hasattr(attn, "k_norm"):
                K = attn.k_norm(K)
            Q, K = _apply_rotary_qk(Q, K, cos, sin)

            cache_layer = o_cache.layers[attn.layer_idx]
            o_K = cache_layer.keys      # [1, n_kv, o_len, d]
            o_V = cache_layer.values
            if use_flash_attn:
                attn_out = _action_attention_flash(Q, K, V, o_K, o_V, o_len)
            else:
                attn_out = _action_attention_python(
                    Q, K, V, o_K, o_V, o_len, scaling=attn.scaling
                )
            # [A, n_q, A_max, d] -> [A, A_max, n_q*d]
            attn_out = attn_out.transpose(1, 2).reshape(A, A_max, n_q * d)
            attn_out = attn.o_proj(attn_out)
            h = residual + attn_out

            residual2 = h
            h = layer.post_attention_layernorm(h)
            h = residual2 + layer.mlp(h)

    h = backbone.base_model.norm(h)  # [A, A_max, hidden]
    term = (action_lens.to(device) - 1).clamp(min=0).long()
    h_term = h[torch.arange(A, device=device), term]  # [A, hidden]
    return backbone.q_plus_head(h_term)  # [A]
