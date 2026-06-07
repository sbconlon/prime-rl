"""Phase 8 (action-level ARM) -- shared-o-cache Q+ optimization parity.

forward_q_plus_action_shared_o prefills o once and continues the |A| actions off
the shared KV-cache. It must equal the naive forward_q_plus_action per-action loop
(the oracle) to tight tolerance -- KV-cache continuation is exact. Runs on the
tiny CPU model (the flash_attn_with_kvcache kernel is the Phase 9 / Ampere step).
"""

from __future__ import annotations

import pytest
import torch
from transformers import Qwen2Config, Qwen2Model

from prime_rl.advantage_server.action_advantage import tokenize_action
from prime_rl.advantage_server.compute import _build_per_decision_point_value_tensors
from prime_rl.configs.trainer import LoRAConfig
from prime_rl.orchestrator.value_networks import ValueNetworkBackbone
from prime_rl.transport.types import DecisionPoint, TrainingSample


def _tiny_qwen2_config() -> Qwen2Config:
    return Qwen2Config(
        vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=512,
    )


def _backbone() -> ValueNetworkBackbone:
    torch.manual_seed(0)
    base = Qwen2Model(_tiny_qwen2_config())
    lora = LoRAConfig(rank=8, alpha=16.0, dropout=0.0)
    bb = ValueNetworkBackbone(base, lora_config=lora, polyak_tau=0.005)
    # Non-zero Q+ head + LoRA so the reads are non-trivial (else all-zero parity
    # is uninformative).
    with torch.no_grad():
        for name, p in bb.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                p.copy_(torch.randn_like(p) * 0.1)
        bb.q_plus_head.linear.weight.data.normal_()
        bb.q_plus_head.linear.bias.data.normal_()
    return bb


class _CharTok:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 256 for c in text]


def test_shared_o_matches_naive_oracle():
    bb = _backbone()
    torch.manual_seed(2)
    o = torch.randint(0, 256, (1, 7))
    actions = [
        torch.tensor([5, 6, 7]),
        torch.tensor([10, 11]),
        torch.tensor([20, 21, 22, 23]),
    ]
    with torch.no_grad():
        shared = bb.forward_q_plus_action_shared_o(o, actions)
        naive = torch.stack([bb.forward_q_plus_action(o, a.unsqueeze(0))[0] for a in actions])
    assert torch.allclose(shared, naive, atol=1e-4), f"{shared} vs {naive}"


def test_shared_o_ragged_actions_terminal_reads():
    """Differing action lengths each read Q+ at their own terminal (not contaminated
    by other actions)."""
    bb = _backbone()
    torch.manual_seed(3)
    o = torch.randint(0, 256, (1, 5))
    a_short = torch.tensor([9])
    a_long = torch.tensor([9, 8, 7, 6])
    with torch.no_grad():
        shared = bb.forward_q_plus_action_shared_o(o, [a_short, a_long])
        naive_short = bb.forward_q_plus_action(o, a_short.unsqueeze(0))[0]
        naive_long = bb.forward_q_plus_action(o, a_long.unsqueeze(0))[0]
    assert torch.allclose(shared[0], naive_short, atol=1e-4)
    assert torch.allclose(shared[1], naive_long, atol=1e-4)


def test_shared_o_empty_actions():
    bb = _backbone()
    o = torch.randint(0, 256, (1, 4))
    with torch.no_grad():
        out = bb.forward_q_plus_action_shared_o(o, [])
    assert out.numel() == 0


def test_kernel_matches_naive_oracle():
    """forward_q_plus_action_batched (the flash continuation kernel; FP32 Python
    fallback on this pre-Ampere CPU run) equals the naive per-action loop."""
    bb = _backbone()
    torch.manual_seed(5)
    o = torch.randint(0, 256, (1, 6))
    actions = [torch.tensor([5, 6, 7]), torch.tensor([10, 11]), torch.tensor([20, 21, 22, 23])]
    with torch.no_grad():
        kernel = bb.forward_q_plus_action_batched(o, actions, use_flash_attn=False)
        naive = torch.stack([bb.forward_q_plus_action(o, a.unsqueeze(0))[0] for a in actions])
    assert torch.allclose(kernel, naive, atol=1e-4), f"{kernel} vs {naive}"


def test_kernel_matches_shared_o():
    """The two o-sharing paths (flash kernel fallback and HF continuation) agree."""
    bb = _backbone()
    torch.manual_seed(6)
    o = torch.randint(0, 256, (1, 5))
    actions = [torch.tensor([9]), torch.tensor([9, 8, 7, 6]), torch.tensor([3, 4])]
    with torch.no_grad():
        kernel = bb.forward_q_plus_action_batched(o, actions, use_flash_attn=False)
        shared = bb.forward_q_plus_action_shared_o(o, actions)
    assert torch.allclose(kernel, shared, atol=1e-4)


@pytest.mark.gpu
def test_flash_kernel_matches_naive_on_gpu():
    """CLUSTER (Ampere) parity: the REAL flash_attn path of the batched kernel
    equals the naive per-action loop. Skipped off-GPU. bf16 tolerance (flash_attn
    requires fp16/bf16). Run on an A100 with:  uv run pytest \
    tests/unit/orchestrator/test_q_plus_action_shared_o.py -m gpu -v"""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("flash_attn needs Ampere+ (sm_80); this GPU is pre-Ampere")
    bb = _backbone().to(device="cuda", dtype=torch.bfloat16)
    o = torch.randint(0, 256, (1, 6), device="cuda")
    actions = [
        torch.tensor([5, 6, 7], device="cuda"),
        torch.tensor([10, 11], device="cuda"),
        torch.tensor([20, 21, 22, 23], device="cuda"),
    ]
    with torch.no_grad():
        flash = bb.forward_q_plus_action_batched(o, actions, use_flash_attn=True)
        naive = torch.stack([bb.forward_q_plus_action(o, a.unsqueeze(0))[0] for a in actions])
    assert torch.allclose(flash.float(), naive.float(), atol=5e-2), f"{flash} vs {naive}"


@pytest.mark.parametrize("mode", ["shared_o", "kernel"])
def test_builder_optimized_matches_naive(mode):
    """_build_per_decision_point_value_tensors in each optimized mode equals the
    naive-loop oracle for the q_plus records."""
    bb = _backbone()
    tok = _CharTok()
    sample = TrainingSample(
        prompt_ids=[1, 2, 3],
        prompt_mask=[False, False, False],
        completion_ids=[10, 11, 12, 13, 20, 21],
        completion_mask=[True, True, False, False, True, True],
        completion_logprobs=[0.0] * 6,
        completion_temperatures=[1.0] * 6,
        decision_points=[
            DecisionPoint(0, 2, ["go", "look", "take apple"], 0, pi_hat=0.5),
            DecisionPoint(4, 6, ["take", "look"], 1, pi_hat=0.3),
        ],
    )
    opt = _build_per_decision_point_value_tensors([sample], bb, tok, q_plus_mode=mode)
    naive = _build_per_decision_point_value_tensors([sample], bb, tok, q_plus_mode="naive")
    assert len(opt) == len(naive) == 2
    for ro, rn in zip(opt, naive):
        assert len(ro["q_plus"]) == len(rn["q_plus"])
        for qo, qn in zip(ro["q_plus"], rn["q_plus"]):
            assert abs(qo - qn) < 1e-4
        assert abs(ro["v"] - rn["v"]) < 1e-6
