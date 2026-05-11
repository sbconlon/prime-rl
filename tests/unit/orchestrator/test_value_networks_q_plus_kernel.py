"""Unit tests for forward_q_plus_candidates_batched (plan sections 4.2-4.4).

CPU-runnable via the FP32 Python attention fallback inside the kernel.
The flash_attn path (use_flash_attn=True) is verified by spike 5.2 on the
cluster; here we test the layer-chain math + cache sharing + per-row RoPE
+ LoRA routing + position-0 assertion + shape contract.

Strategy:
  - Run `forward_q_plus_sampled_all_positions` over the full trajectory
    once. This produces the prefix_cache that all candidate rows will share.
  - For each (j, c) pair, run the SAME `forward_q_plus_sampled_all_positions`
    over a substituted trajectory where completion_ids[j] is replaced by
    candidate c. Read the Q+ value at position prompt_len + j. This is the
    cross-attention oracle.
  - Compare against `forward_q_plus_candidates_batched`\'s output at row (j, c).
"""

from __future__ import annotations

import pytest
import torch
from transformers import Qwen3Config, Qwen3Model

from prime_rl.configs.trainer import LoRAConfig
from prime_rl.orchestrator.value_networks import (
    Q_PLUS_SLOT,
    ValueNetworkBackbone,
)


# ---------------------------------------------------------------------------
# Fixtures: tiny 2-layer Qwen3 (mirrors test_value_networks.py but for Qwen3
# specifically -- the kernel imports rotate_half from Qwen3 modeling).
# ---------------------------------------------------------------------------


def _tiny_qwen3_config() -> Qwen3Config:
    """Smallest Qwen3 config that satisfies MultiLoRALinear (divisible-by-8)
    and Qwen3Attention (head_dim must be even for RoPE)."""
    return Qwen3Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
    )


@pytest.fixture
def tiny_qwen3_backbone() -> ValueNetworkBackbone:
    torch.manual_seed(0)
    base = Qwen3Model(_tiny_qwen3_config())
    lora = LoRAConfig(rank=8, alpha=16.0, dropout=0.0)
    backbone = ValueNetworkBackbone(base, lora_config=lora, polyak_tau=0.005)
    # Break zero-init so outputs are non-trivial.
    torch.manual_seed(1)
    with torch.no_grad():
        for name, p in backbone.named_parameters():
            if ("lora_A" in name) or ("lora_B" in name):
                p.copy_(torch.randn_like(p) * 0.1)
        for head in (backbone.v_head, backbone.q_plus_head, backbone.v_target_head):
            torch.nn.init.normal_(head.linear.weight, std=0.05)
    backbone.eval()
    return backbone


# ---------------------------------------------------------------------------
# 1. Shape and basic contract
# ---------------------------------------------------------------------------


def test_kernel_output_shape(tiny_qwen3_backbone: ValueNetworkBackbone):
    """Output is [N_active, K]."""
    torch.manual_seed(2)
    prompt_len = 4
    completion_len = 6
    full_input = torch.randint(1, 256, (1, prompt_len + completion_len), dtype=torch.long)
    _, prefix_cache = tiny_qwen3_backbone.forward_q_plus_sampled_all_positions(full_input)

    N_active = 3
    K = 4
    candidate_token_ids = torch.randint(1, 256, (N_active, K), dtype=torch.long)
    candidate_positions = torch.tensor(
        [prompt_len, prompt_len + 1, prompt_len + 2], dtype=torch.int32
    )

    out = tiny_qwen3_backbone.forward_q_plus_candidates_batched(
        prefix_cache=prefix_cache,
        candidate_token_ids=candidate_token_ids,
        candidate_positions=candidate_positions,
        use_flash_attn=False,  # CPU test path
    )
    assert out.shape == (N_active, K)


def test_kernel_rejects_position_zero(tiny_qwen3_backbone: ValueNetworkBackbone):
    """candidate_positions must all be >= 1 (Q6)."""
    full_input = torch.randint(1, 256, (1, 8), dtype=torch.long)
    _, prefix_cache = tiny_qwen3_backbone.forward_q_plus_sampled_all_positions(full_input)

    candidate_token_ids = torch.tensor([[10, 20]], dtype=torch.long)
    bad_positions = torch.tensor([0], dtype=torch.int32)
    with pytest.raises(AssertionError, match="candidate_positions must be >= 1"):
        tiny_qwen3_backbone.forward_q_plus_candidates_batched(
            prefix_cache=prefix_cache,
            candidate_token_ids=candidate_token_ids,
            candidate_positions=bad_positions,
            use_flash_attn=False,
        )


def test_kernel_handles_empty_input(tiny_qwen3_backbone: ValueNetworkBackbone):
    """N_active == 0 returns an empty [0, K] tensor (no degenerate forward)."""
    full_input = torch.randint(1, 256, (1, 8), dtype=torch.long)
    _, prefix_cache = tiny_qwen3_backbone.forward_q_plus_sampled_all_positions(full_input)
    K = 4
    out = tiny_qwen3_backbone.forward_q_plus_candidates_batched(
        prefix_cache=prefix_cache,
        candidate_token_ids=torch.empty((0, K), dtype=torch.long),
        candidate_positions=torch.empty((0,), dtype=torch.int32),
        use_flash_attn=False,
    )
    assert out.shape == (0, K)


# ---------------------------------------------------------------------------
# 2. Parity vs forward_q_plus_sampled_all_positions oracle
# ---------------------------------------------------------------------------


def test_kernel_parity_vs_all_positions_oracle(
    tiny_qwen3_backbone: ValueNetworkBackbone,
):
    """Batched kernel output matches per-(j, c) oracle output.

    Oracle for one row (j, c):
        substituted_input = prompt + completion with completion[j] replaced by c
        q_plus_seq, _ = forward_q_plus_sampled_all_positions(substituted_input)
        oracle_q[j, c] = q_plus_seq at position (prompt_len + j)

    Both paths use the same strict-causal mask (from §4.1), so the Q+ value
    at position p of the substituted-input forward represents
    Q+(prefix=tokens[:p], action=tokens[p]) -- which is exactly what the
    batched kernel computes for (j, c) when prefix_cache is the unsubstituted
    trajectory\'s cache.

    Tolerances: 5e-4 absolute for FP32; if you bump model dtype this should
    relax to ~1e-2 for BF16.
    """
    torch.manual_seed(3)
    prompt_len = 3
    completion_len = 5
    seq_len = prompt_len + completion_len
    full_input = torch.randint(1, 256, (1, seq_len), dtype=torch.long)

    # Prefix cache from the unsubstituted trajectory.
    _, prefix_cache = tiny_qwen3_backbone.forward_q_plus_sampled_all_positions(full_input)

    # Candidates: at active positions [prompt_len, prompt_len+1, prompt_len+2],
    # try K=3 candidates per position.
    K = 3
    active_local_j = [0, 1, 2]  # indices into the completion
    candidate_token_ids = torch.randint(
        1, 256, (len(active_local_j), K), dtype=torch.long
    )
    candidate_positions = torch.tensor(
        [prompt_len + j for j in active_local_j], dtype=torch.int32
    )

    # Batched kernel.
    batched_out = tiny_qwen3_backbone.forward_q_plus_candidates_batched(
        prefix_cache=prefix_cache,
        candidate_token_ids=candidate_token_ids,
        candidate_positions=candidate_positions,
        use_flash_attn=False,
    )  # [N_active, K]

    # Per-(j, c) oracle.
    oracle_out = torch.zeros_like(batched_out)
    for i, j_local in enumerate(active_local_j):
        full_pos = prompt_len + j_local
        for c_idx in range(K):
            cand = candidate_token_ids[i, c_idx].item()
            sub_input = full_input.clone()
            sub_input[0, full_pos] = cand
            q_seq, _ = tiny_qwen3_backbone.forward_q_plus_sampled_all_positions(sub_input)
            # q_seq: [1, seq_len]; read at position full_pos.
            oracle_out[i, c_idx] = q_seq[0, full_pos]

    abs_diff = (batched_out - oracle_out).abs().max().item()
    assert abs_diff < 5e-4, (
        f"Batched kernel diverges from per-(j, c) oracle: "
        f"max_abs_diff={abs_diff:.3e}\n"
        f"batched=\n{batched_out}\noracle=\n{oracle_out}"
    )


# ---------------------------------------------------------------------------
# 3. RoPE / cache-sharing sanity: different positions produce different outputs
# ---------------------------------------------------------------------------


def test_kernel_different_positions_yield_different_outputs(
    tiny_qwen3_backbone: ValueNetworkBackbone,
):
    """Two rows with the SAME candidate token but DIFFERENT trajectory
    positions must produce different Q+ values. Catches:
      - RoPE not being applied per-row (would make same-token rows equal)
      - cache_seqlens not being respected (would attend to the same prefix
        regardless of position)
    """
    torch.manual_seed(4)
    full_input = torch.randint(1, 256, (1, 10), dtype=torch.long)
    _, prefix_cache = tiny_qwen3_backbone.forward_q_plus_sampled_all_positions(full_input)

    # Same candidate at two different positions.
    same_token = torch.tensor([[42], [42]], dtype=torch.long)   # [N_active=2, K=1]
    positions = torch.tensor([3, 7], dtype=torch.int32)         # [2]

    out = tiny_qwen3_backbone.forward_q_plus_candidates_batched(
        prefix_cache=prefix_cache,
        candidate_token_ids=same_token,
        candidate_positions=positions,
        use_flash_attn=False,
    )
    diff = (out[0, 0] - out[1, 0]).abs().item()
    assert diff > 1e-3, (
        f"Same token at different positions produced near-identical Q+ "
        f"({out[0,0].item()} vs {out[1,0].item()}, diff={diff:.3e}). "
        "RoPE or cache_seqlens isn\'t per-row."
    )
# ---------------------------------------------------------------------------
# FlexAttention all-positions Q+ kernel tests (lever 1).
# ---------------------------------------------------------------------------
#
# These three tests exercise forward_q_plus_sampled_all_positions_flex_kernel
# directly (bypassing the auto-detect dispatch since local CI is CPU-only).
# The kernel itself requires CUDA + PyTorch >= 2.5 (FlexAttention), so the
# tests are skipped when those aren't available.
#
# The dispatch path (use_flex_attn parameter on
# forward_q_plus_sampled_all_positions) is covered by the existing
# value_networks tests, which exercise the SDPA Math fallback on CPU.


@pytest.fixture
def cuda_tiny_qwen3_backbone() -> ValueNetworkBackbone:
    """CUDA + BF16 backbone, mirrors AdvSrv/AdvTrainer production config."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for FlexAttention tests")
    try:
        from torch.nn.attention.flex_attention import flex_attention  # noqa: F401
    except ImportError:
        pytest.skip("FlexAttention not available (needs PyTorch >= 2.5)")

    torch.manual_seed(0)
    base = Qwen3Model(_tiny_qwen3_config()).to(device="cuda", dtype=torch.bfloat16)
    lora = LoRAConfig(rank=8, alpha=16.0, dropout=0.0)
    backbone = ValueNetworkBackbone(base, lora_config=lora, polyak_tau=0.005)
    backbone = backbone.to(device="cuda", dtype=torch.bfloat16)
    # Break zero-init.
    torch.manual_seed(1)
    with torch.no_grad():
        for name, p in backbone.named_parameters():
            if ("lora_A" in name) or ("lora_B" in name):
                p.copy_(torch.randn_like(p) * 0.1)
        for head in (backbone.v_head, backbone.q_plus_head, backbone.v_target_head):
            torch.nn.init.normal_(head.linear.weight, std=0.05)
    backbone.eval()
    return backbone


def test_flex_forward_parity_vs_sdpa_math(
    cuda_tiny_qwen3_backbone: ValueNetworkBackbone,
):
    """Flex kernel's Q+ output matches the SDPA Math path within BF16 noise.

    Per spike 6.2 the FP32 parity is essentially zero (atol < 1e-8); in BF16
    we allow atol=3e-2 for accumulation noise. This is the canonical
    correctness gate: if it ever regresses we have a real math bug.
    """
    torch.manual_seed(2)
    input_ids = torch.randint(
        1, 256, (1, 32), dtype=torch.long, device="cuda"
    )

    flex_q, _ = cuda_tiny_qwen3_backbone.forward_q_plus_sampled_all_positions(
        input_ids, use_flex_attn=True,
    )
    sdpa_q, _ = cuda_tiny_qwen3_backbone.forward_q_plus_sampled_all_positions(
        input_ids, use_flex_attn=False,
    )

    abs_diff = (flex_q.float() - sdpa_q.float()).abs()
    max_abs = abs_diff.max().item()
    assert torch.isfinite(flex_q).all(), "flex Q+ produced non-finite values"
    assert max_abs < 3e-2, (
        f"flex vs SDPA Math Q+ output diverged: max_abs={max_abs:.3e} "
        f"(BF16 noise budget = 3e-2)"
    )


def test_flex_backward_parity_vs_sdpa_math(
    cuda_tiny_qwen3_backbone: ValueNetworkBackbone,
):
    """Gradients w.r.t. Q+ LoRA params match between flex and SDPA Math paths.

    Confirms that FlexAttention's backward is correct end-to-end through
    the full Qwen3 layer chain + LoRA routing + q_plus_head, not just the
    raw attention call covered by spike 6.1.
    """
    torch.manual_seed(3)
    input_ids = torch.randint(
        1, 256, (1, 32), dtype=torch.long, device="cuda"
    )
    target = torch.randn(1, 32, dtype=torch.bfloat16, device="cuda")

    # Q+ LoRA slot (1) params: collect references for grad inspection.
    def q_plus_lora_params(backbone):
        return [
            (name, p) for name, p in backbone.named_parameters()
            if (("lora_A" in name) or ("lora_B" in name))
            and name.endswith(f".{Q_PLUS_SLOT}")
        ]

    # Flex path backward.
    backbone = cuda_tiny_qwen3_backbone
    for _, p in q_plus_lora_params(backbone):
        p.grad = None
    backbone.train()
    flex_q, _ = backbone.forward_q_plus_sampled_all_positions(
        input_ids, use_flex_attn=True,
    )
    ((flex_q - target) ** 2).mean().backward()
    flex_grads = {name: p.grad.detach().clone() for name, p in q_plus_lora_params(backbone)}

    # SDPA Math path backward.
    for _, p in q_plus_lora_params(backbone):
        p.grad = None
    sdpa_q, _ = backbone.forward_q_plus_sampled_all_positions(
        input_ids, use_flex_attn=False,
    )
    ((sdpa_q - target) ** 2).mean().backward()
    sdpa_grads = {name: p.grad.detach().clone() for name, p in q_plus_lora_params(backbone)}

    backbone.eval()

    assert flex_grads.keys() == sdpa_grads.keys() and len(flex_grads) > 0
    grad_tol = 5e-2  # BF16 backward accumulates more aggressively than forward
    for name in flex_grads:
        gf = flex_grads[name]
        gs = sdpa_grads[name]
        assert torch.isfinite(gf).all(), f"non-finite flex grad: {name}"
        mab = (gf - gs).float().abs().max().item()
        assert mab < grad_tol, (
            f"grad diverged for {name}: max_abs={mab:.3e} (tol {grad_tol})"
        )


def test_flex_cache_contract_matches_sdpa_math(
    cuda_tiny_qwen3_backbone: ValueNetworkBackbone,
):
    """Flex kernel populates DynamicCache identically (within BF16 noise) to
    the SDPA Math path. The AdvSrv K-candidate kernel reads this cache, so
    a layout/shape/dtype mismatch would silently break inference.

    Compares per-layer K/V tensors. Cache layout must be [B, n_kv, S, d]
    in BF16, matching what HF's standard forward produces.
    """
    torch.manual_seed(4)
    input_ids = torch.randint(
        1, 256, (1, 32), dtype=torch.long, device="cuda"
    )

    _, flex_cache = cuda_tiny_qwen3_backbone.forward_q_plus_sampled_all_positions(
        input_ids, use_flex_attn=True,
    )
    _, sdpa_cache = cuda_tiny_qwen3_backbone.forward_q_plus_sampled_all_positions(
        input_ids, use_flex_attn=False,
    )

    assert len(flex_cache.layers) == len(sdpa_cache.layers), (
        f"layer count mismatch: flex={len(flex_cache.layers)} "
        f"sdpa={len(sdpa_cache.layers)}"
    )

    cache_tol = 5e-2  # BF16 noise; cache K, V go through Q+ LoRA path
    for i, (fl, sd) in enumerate(zip(flex_cache.layers, sdpa_cache.layers)):
        assert fl.keys.shape == sd.keys.shape, (
            f"layer {i} keys shape: flex={fl.keys.shape} sdpa={sd.keys.shape}"
        )
        assert fl.values.shape == sd.values.shape, (
            f"layer {i} values shape: flex={fl.values.shape} sdpa={sd.values.shape}"
        )
        assert fl.keys.dtype == sd.keys.dtype, (
            f"layer {i} keys dtype: flex={fl.keys.dtype} sdpa={sd.keys.dtype}"
        )

        k_diff = (fl.keys.float() - sd.keys.float()).abs().max().item()
        v_diff = (fl.values.float() - sd.values.float()).abs().max().item()
        assert k_diff < cache_tol, (
            f"layer {i} K cache diverged: max_abs={k_diff:.3e} (tol {cache_tol})"
        )
        assert v_diff < cache_tol, (
            f"layer {i} V cache diverged: max_abs={v_diff:.3e} (tol {cache_tol})"
        )
