"""Phase 5 (action-level ARM) -- forward_q_plus_action: multi-token append-and-read.

Tiny CPU Qwen2 model. Validates the read position (action terminal), standard-causal
self-inclusion (and difference from the strict-causal mask the token-level path uses),
prefix/terminal distinctness, the cold-start zero, and ragged padded-batch reads.
"""

from __future__ import annotations

import pytest
import torch
from transformers import Qwen2Config, Qwen2Model

from prime_rl.configs.trainer import LoRAConfig
from prime_rl.orchestrator.value_networks import (
    Q_PLUS_SLOT,
    ValueNetworkBackbone,
    _adapter_routing,
)


def _tiny_qwen2_config() -> Qwen2Config:
    return Qwen2Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
    )


@pytest.fixture
def backbone() -> ValueNetworkBackbone:
    torch.manual_seed(0)
    base_model = Qwen2Model(_tiny_qwen2_config())
    lora = LoRAConfig(rank=8, alpha=16.0, dropout=0.0)
    return ValueNetworkBackbone(base_model, lora_config=lora, polyak_tau=0.005)


def _nonzero_q_plus_head(backbone: ValueNetworkBackbone) -> None:
    """Q+ head is zero-initialized (cold start). Randomize it so reads reflect the
    underlying hidden state (otherwise every read is 0 and masks/positions can't be
    distinguished)."""
    torch.manual_seed(1)
    backbone.q_plus_head.linear.weight.data.normal_()
    backbone.q_plus_head.linear.bias.data.normal_()


def _o_a(o_len=5, a_len=3):
    torch.manual_seed(2)
    o = torch.randint(0, 256, (1, o_len))
    a = torch.randint(0, 256, (1, a_len))
    return o, a


def _standard_causal_hidden(backbone, full_ids):
    with torch.no_grad():
        with _adapter_routing(Q_PLUS_SLOT, full_ids.numel()):
            out = backbone.base_model(full_ids)
    return out.last_hidden_state


def test_action_q_plus_read_position(backbone):
    _nonzero_q_plus_head(backbone)
    o, a = _o_a(o_len=5, a_len=3)
    with torch.no_grad():
        got = backbone.forward_q_plus_action(o, a)
        full = torch.cat([o, a], dim=-1)
        hidden = _standard_causal_hidden(backbone, full)
        # Read at the action terminal: index o_len + a_len - 1 == -1 here.
        ref_terminal = backbone.q_plus_head(hidden[:, o.shape[1] + a.shape[1] - 1, :])
        ref_o_end = backbone.q_plus_head(hidden[:, o.shape[1] - 1, :])
    assert torch.allclose(got, ref_terminal, atol=1e-5)
    assert not torch.allclose(got, ref_o_end)  # NOT read at o's end


def test_action_q_plus_standard_causal(backbone):
    _nonzero_q_plus_head(backbone)
    o, a = _o_a()
    full = torch.cat([o, a], dim=-1)
    with torch.no_grad():
        got = backbone.forward_q_plus_action(o, a)
        # Reference: standard causal (self-inclusive), no mask.
        std = backbone.q_plus_head(_standard_causal_hidden(backbone, full)[:, -1, :])
        # The strict-causal path (token-level dead code) excludes the token's own
        # K/V; its terminal read must differ.
        strict = backbone.q_plus_head(backbone._forward_base(full, Q_PLUS_SLOT)[:, -1, :])
    assert torch.allclose(got, std, atol=1e-5)  # standard causal
    assert not torch.allclose(got, strict)  # and NOT the strict mask


def test_action_q_plus_prefix_actions_distinct(backbone):
    """Actions whose token-ids overlap as a prefix read at different terminal
    positions, so Q+ distinguishes them (relies on the caller including the
    terminator in action_ids)."""
    _nonzero_q_plus_head(backbone)
    torch.manual_seed(3)
    o = torch.randint(0, 256, (1, 4))
    a1 = torch.tensor([[5, 6]])
    a2 = torch.tensor([[5, 6, 7]])
    with torch.no_grad():
        q1 = backbone.forward_q_plus_action(o, a1)
        q2 = backbone.forward_q_plus_action(o, a2)
    assert not torch.allclose(q1, q2)


def test_action_q_plus_cold_start_zero(backbone):
    """Zero-init q_plus_head (cold start) -> Q+(o,a) == 0 for any action ->
    pi_RM uniform at iteration 0."""
    o, a = _o_a()
    with torch.no_grad():
        got = backbone.forward_q_plus_action(o, a)
    assert torch.allclose(got, torch.zeros_like(got))


def test_action_q_plus_ragged_batch(backbone):
    """A padded batch of differing action lengths reads each row at its true
    terminal index; right padding after the terminal is ignored (causal)."""
    _nonzero_q_plus_head(backbone)
    torch.manual_seed(4)
    o = torch.randint(0, 256, (2, 4))
    # Row 0 action length 2, row 1 action length 3; pad row 0 to width 3.
    pad = 0
    action_ids = torch.tensor([[11, 12, pad], [21, 22, 23]])
    action_lengths = torch.tensor([2, 3])
    with torch.no_grad():
        batched = backbone.forward_q_plus_action(o, action_ids, action_lengths=action_lengths)
        # Per-row single forwards (true, unpadded actions) as the oracle.
        row0 = backbone.forward_q_plus_action(o[:1], torch.tensor([[11, 12]]))
        row1 = backbone.forward_q_plus_action(o[1:], torch.tensor([[21, 22, 23]]))
    assert torch.allclose(batched[0], row0[0], atol=1e-5)
    assert torch.allclose(batched[1], row1[0], atol=1e-5)
