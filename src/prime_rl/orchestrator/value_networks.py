"""ARM value-network architecture: V, Q+, V_target as 3 LoRA adapter slots
on a shared frozen base.

Each value network shares the same SFT'd Qwen2.5-1.5B base. The base is
frozen; all task-specific learning happens through the LoRA adapter slots
and the per-task value heads. V (slot 0) and Q+ (slot 1) are trained by
the Advantage Trainer (Phase 7); V_target (slot 2) is a Polyak-averaged
snapshot of V's adapter slot, updated via `polyak_update_v_target`.

This module is structural-only: it defines the artifact, the forward
methods, and the Polyak update. No production code imports it yet --
Phase 6 wires it into the Advantage Server.

The naive forward methods here (`forward_v`, `forward_q_plus`,
`forward_v_target`) take one full forward pass each per call and read the
value head at a single token position. Phase 6.5 layers optimized
all-positions / batched-K-candidates variants on top without modifying
this baseline.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn as nn
from torch import Tensor

from prime_rl.configs.trainer import LoRAConfig
from prime_rl.trainer.lora import (
    _find_target_modules,
    _get_module_by_name,
    _set_module_by_name,
)
from prime_rl.trainer.models.layers.lora.base import (
    LORA_NUM_TOKENS,
    SCALING_FACTORS,
    set_lora_num_tokens,
    set_multilora_scaling,
)
from prime_rl.trainer.models.layers.lora.multi_linear import MultiLoRALinear

# Adapter slot assignments. Three slots on a shared MultiLoRALinear instance
# at every target module (q_proj, k_proj, ..., down_proj) per the phase
# doc DQ1 commitment.
V_SLOT: int = 0
Q_PLUS_SLOT: int = 1
V_TARGET_SLOT: int = 2
N_ADAPTERS: int = 3


@dataclass
class ValueNetworkConfig:
    """Configuration for the V/Q+/V_target value networks.

    `base_model_name` is consumed by `ValueNetworkBackbone.from_pretrained`;
    when constructing directly with an already-loaded model, it is unused.
    """

    base_model_name: str
    lora: LoRAConfig
    polyak_tau: float = 0.005


class ValueHead(nn.Module):
    """Replaces the LM head: hidden_dim -> 1 scalar value.

    Zero-initialized weight and bias so that V/Q+/V_target output exactly
    0.0 before any training. This makes Phase 3's cold-start degenerate
    branch trigger cleanly at iteration 0 (all Q+ values non-positive ->
    A = 0 for every position).
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 1, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.linear(hidden).squeeze(-1)


def _ensure_lora_globals(scaling: float) -> None:
    """Ensure prime-rl's global LORA_NUM_TOKENS / SCALING_FACTORS are
    initialized with shape (N_ADAPTERS,). Required because
    MultiLoRALinear.__init__ calls `get_lora_num_tokens()`.

    Subsequent forwards must use in-place updates (`set_lora_num_tokens`
    with default `reset_reference=False`); replacing the tensor would
    orphan the MultiLoRALinear modules that captured the prior reference.
    """
    from prime_rl.trainer.models.layers.lora import base as _lora_base

    needs_reset = (
        _lora_base.LORA_NUM_TOKENS is None
        or _lora_base.LORA_NUM_TOKENS.shape != (N_ADAPTERS,)
    )
    if needs_reset:
        set_lora_num_tokens(
            torch.zeros(N_ADAPTERS, dtype=torch.long), reset_reference=True
        )
        set_multilora_scaling(
            torch.full((N_ADAPTERS,), float(scaling)), reset_reference=True
        )


@contextmanager
def _adapter_routing(slot: int, n_tokens: int) -> Iterator[None]:
    """Context manager: route all `n_tokens` tokens to `slot` for the with-block.

    Updates the global `LORA_NUM_TOKENS` in-place so wrapped MultiLoRALinear
    modules pick up the new routing immediately. Does not reset on exit --
    the next forward writes its own routing.
    """
    routing = torch.zeros(N_ADAPTERS, dtype=torch.long)
    routing[slot] = n_tokens
    set_lora_num_tokens(routing)
    yield


class ValueNetworkBackbone(nn.Module):
    """SFT'd base + 3-slot MultiLoRALinear (V/Q+/V_target) + 3 value heads.

    Constructor takes an already-loaded HuggingFace base model. Use
    `from_pretrained(config)` to load by name.

    The base is frozen; only LoRA adapters and value heads are trainable.
    """

    def __init__(
        self,
        base_model: nn.Module,
        lora_config: LoRAConfig,
        polyak_tau: float = 0.005,
    ):
        super().__init__()

        # MultiLoRALinear's __init__ reads from the global LoRA state, so it
        # must be initialized with the right shape (N_ADAPTERS=3) before we
        # start wrapping any layers.
        _ensure_lora_globals(scaling=lora_config.alpha / lora_config.rank)

        self.base_model = base_model
        self.lora_config = lora_config
        self.polyak_tau = polyak_tau

        # Wrap target Linear modules with MultiLoRALinear(n_adapters=3).
        target_module_names = _find_target_modules(
            base_model, lora_config.target_modules
        )
        if not target_module_names:
            raise ValueError(
                "No target modules found in base model. Check that "
                f"lora_config.target_modules ({lora_config.target_modules}) "
                "match the base model's module naming (Qwen2 uses q_proj, "
                "k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)."
            )
        for name in target_module_names:
            base_layer = _get_module_by_name(base_model, name)
            if isinstance(base_layer, nn.Linear):
                multi = MultiLoRALinear(
                    base_layer=base_layer,
                    rank=lora_config.rank,
                    n_adapters=N_ADAPTERS,
                    alpha=lora_config.alpha,
                    dropout=lora_config.dropout,
                )
                _set_module_by_name(base_model, name, multi)
            # GroupedExperts (MoE) modules are out of scope -- Qwen2.5-1.5B
            # is dense. Phase 6 may revisit if a MoE base is ever used.

        # Freeze base, leave LoRA + heads trainable.
        # We replicate the existing `freeze_all_except_lora_and_specified`
        # logic locally because that function depends on a LoRAConfig that
        # carries `modules_to_save`, which is the orchestrator-side concept.
        for name, p in self.base_model.named_parameters():
            is_lora = "lora_A" in name or "lora_B" in name
            p.requires_grad = is_lora

        # Three value heads, one per adapter slot. Read at the appropriate
        # token position for each forward (last position for V/V_target,
        # appended-action position for Q+).
        hidden_size = self._infer_hidden_size(base_model)
        self.v_head = ValueHead(hidden_size)
        self.q_plus_head = ValueHead(hidden_size)
        self.v_target_head = ValueHead(hidden_size)

    @staticmethod
    def _infer_hidden_size(base_model: nn.Module) -> int:
        """Pull hidden_size from the model's HF config attribute."""
        if hasattr(base_model, "config") and hasattr(base_model.config, "hidden_size"):
            return int(base_model.config.hidden_size)
        raise ValueError(
            "Could not infer hidden_size from base_model.config; "
            "pass a HuggingFace PreTrainedModel."
        )

    @classmethod
    def from_pretrained(cls, config: ValueNetworkConfig) -> "ValueNetworkBackbone":
        """Load the base model by name and wrap with V/Q+/V_target adapters."""
        from transformers import AutoModel

        base_model = AutoModel.from_pretrained(config.base_model_name)
        return cls(
            base_model=base_model,
            lora_config=config.lora,
            polyak_tau=config.polyak_tau,
        )

    # -------------------------------------------------------------------
    # Forward methods (naive: one full forward per call, read at one position)
    # -------------------------------------------------------------------

    def _forward_base(self, input_ids: Tensor, slot: int) -> Tensor:
        """Run base model with all tokens routed to `slot`. Returns last_hidden_state."""
        batch, seq = input_ids.shape
        with _adapter_routing(slot, batch * seq):
            output = self.base_model(input_ids)
        # Both AutoModel.from_pretrained's BaseModelOutputWithPast and direct
        # Qwen2Model output expose `last_hidden_state`.
        return output.last_hidden_state  # [batch, seq, hidden]

    def forward_v(self, input_ids: Tensor) -> Tensor:
        """V(o_k): scalar per batch element, read at the last position of input_ids.

        Returns shape [batch].
        """
        hidden = self._forward_base(input_ids, V_SLOT)
        return self.v_head(hidden[:, -1, :])

    def forward_q_plus(self, observation_ids: Tensor, action_id: int) -> Tensor:
        """Q+(o_k, a_k): scalar per batch element.

        Appends `action_id` to `observation_ids`, runs base through Q+ slot,
        reads q_plus_head at the appended action position. Returns shape [batch].

        For batched observations all batch elements share the same action_id.
        Per-batch action_ids would be a thin wrapper on top of this.
        """
        batch = observation_ids.shape[0]
        action_tensor = torch.full(
            (batch, 1),
            action_id,
            dtype=observation_ids.dtype,
            device=observation_ids.device,
        )
        full_ids = torch.cat([observation_ids, action_tensor], dim=-1)
        hidden = self._forward_base(full_ids, Q_PLUS_SLOT)
        # The appended action token is at the last position (index -1).
        return self.q_plus_head(hidden[:, -1, :])

    def forward_v_target(self, input_ids: Tensor) -> Tensor:
        """V_target(o_k): scalar per batch element, read at the last position.

        Identical to forward_v but routed through the V_target adapter slot.
        Returns shape [batch].
        """
        hidden = self._forward_base(input_ids, V_TARGET_SLOT)
        return self.v_target_head(hidden[:, -1, :])

    # -------------------------------------------------------------------
    # Polyak update
    # -------------------------------------------------------------------

    def polyak_update_v_target(self, tau: float | None = None) -> None:
        """Update V_target parameters toward V via Polyak averaging.

            V_target_lora_A <- (1 - tau) * V_target_lora_A + tau * V_lora_A
            V_target_lora_B <- (1 - tau) * V_target_lora_B + tau * V_lora_B
            v_target_head   <- (1 - tau) * v_target_head   + tau * v_head

        Every MultiLoRALinear's slot V_TARGET_SLOT is updated from its slot
        V_SLOT, plus the dedicated value heads.
        """
        tau_value = float(tau if tau is not None else self.polyak_tau)
        with torch.no_grad():
            for module in self.base_model.modules():
                if isinstance(module, MultiLoRALinear):
                    module.lora_A[V_TARGET_SLOT].mul_(1.0 - tau_value).add_(
                        module.lora_A[V_SLOT], alpha=tau_value
                    )
                    module.lora_B[V_TARGET_SLOT].mul_(1.0 - tau_value).add_(
                        module.lora_B[V_SLOT], alpha=tau_value
                    )
            for src_p, dst_p in zip(
                self.v_head.parameters(), self.v_target_head.parameters()
            ):
                dst_p.mul_(1.0 - tau_value).add_(src_p, alpha=tau_value)

    # -------------------------------------------------------------------
    # Verification utility (called by tests)
    # -------------------------------------------------------------------

    def freeze_base_weights(self) -> None:
        """Verify base parameters are frozen and LoRA/heads are trainable.

        Asserts on every parameter in the model. Used by tests as a
        belt-and-suspenders check; production gradient masks come from
        `__init__`'s freezing logic.
        """
        for name, p in self.named_parameters():
            is_lora = "lora_A" in name or "lora_B" in name
            is_head = (
                name.startswith("v_head.")
                or name.startswith("q_plus_head.")
                or name.startswith("v_target_head.")
            )
            if is_lora or is_head:
                assert p.requires_grad, (
                    f"LoRA/head parameter '{name}' should require grad"
                )
            else:
                assert not p.requires_grad, (
                    f"Base parameter '{name}' should not require grad"
                )
