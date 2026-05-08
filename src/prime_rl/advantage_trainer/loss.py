"""Value regression loss for the Advantage Trainer.

The Advantage Trainer regresses V (and optionally Q+) against the targets
produced by the Advantage Server. For PPO it's just MSE on V; for ARM it
adds a parallel MSE term on Q+ at the sampled action.

The combined loss `L = L_V + L_Q+` is run through a single backward pass.
This is mathematically equivalent to running `L_V.backward()` and
`L_Q+.backward()` separately because:
  - V's parameters are LoRA slot 0 + v_head
  - Q+'s parameters are LoRA slot 1 + q_plus_head
  - These sets are disjoint
  - The forward graph for L_V doesn't depend on Q+'s parameters and
    vice versa, so autograd produces the same gradients either way.
The combined backward is mechanically cheaper (one traversal of the
network's computation graph instead of two).

Loss masking: only completion-mask=True positions contribute. Prompt
tokens and bridge tokens (mask=False inside completion_ids during
fragmented multi-turn rollouts) get zero contribution.

Per Phase 7 doc Â§8 DQ4: combined loss with single backward, gated for
PPO via `algorithm` field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor


@dataclass
class ValueLossInputs:
    """Inputs for one sample's value regression loss.

    Tensors are shape [seq] over the input sequence (prompt + completion);
    the loss_mask isolates the positions where the loss should be computed.
    """

    v_predictions: Tensor                  # [seq] -- V output at each token
    v_targets: Tensor                      # [seq] -- V regression target at each token
    q_plus_predictions: Tensor | None      # [seq] -- Q+ output at each token (ARM); None for PPO
    q_plus_targets: Tensor | None          # [seq] -- Q+ regression target at each token; None for PPO
    loss_mask: Tensor                      # [seq] -- bool, True where loss applies
    algorithm: Literal["ppo", "arm"]


@dataclass
class ValueLossOutputs:
    """Outputs of value_regression_loss_fn."""

    loss: Tensor                           # scalar
    metrics: dict[str, Tensor] = field(default_factory=dict)


def value_regression_loss_fn(inputs: ValueLossInputs) -> ValueLossOutputs:
    """Combined V + Q+ regression loss for one sample.

    Returns L = L_V + L_Q+ where each term is MSE on completion-mask=True
    positions only. For PPO (q_plus_targets is None), L_Q+ is zero.

    Mathematical note: V's params and Q+'s params are disjoint LoRA slots
    plus disjoint value heads, so the combined backward through L produces
    the same gradients as separate backwards through L_V and L_Q+. Single
    backward is cheaper.
    """
    mask = inputs.loss_mask
    device = inputs.v_predictions.device
    dtype = inputs.v_predictions.dtype

    # ---- L_V: MSE between V predictions and V targets at mask=True positions
    v_diff = inputs.v_predictions - inputs.v_targets
    if mask.any():
        l_v = (v_diff[mask] ** 2).mean()
    else:
        # Empty mask: gradient-safe zero. Returning a 0-tensor that participates
        # in autograd (rather than a literal 0.0) keeps the loss tensor's
        # graph alive even when no positions contribute.
        l_v = (v_diff * 0.0).sum()

    # ---- L_Q+: same structure for Q+, only for ARM
    if inputs.algorithm == "arm":
        if inputs.q_plus_predictions is None or inputs.q_plus_targets is None:
            raise ValueError(
                "ARM requires both q_plus_predictions and q_plus_targets to be provided."
            )
        q_diff = inputs.q_plus_predictions - inputs.q_plus_targets
        if mask.any():
            l_q = (q_diff[mask] ** 2).mean()
        else:
            l_q = (q_diff * 0.0).sum()
    else:
        # PPO: gradient-safe zero on the right device/dtype. We deliberately
        # don't multiply v_predictions here -- L_Q+ at PPO has zero gradient
        # w.r.t. EVERY parameter, so a fresh zero scalar (not graph-connected)
        # is correct.
        l_q = torch.zeros((), device=device, dtype=dtype)

    loss = l_v + l_q

    # Metrics: detached (no autograd tracking on the metrics dict). Useful
    # for monitoring, not for backprop.
    metrics: dict[str, Tensor] = {
        "l_v": l_v.detach(),
        "l_q": l_q.detach(),
    }
    if mask.any():
        metrics["v_pred_mean"] = inputs.v_predictions[mask].detach().mean()
        metrics["v_target_mean"] = inputs.v_targets[mask].detach().mean()
        if inputs.algorithm == "arm" and inputs.q_plus_predictions is not None:
            assert inputs.q_plus_targets is not None  # for type checkers
            metrics["q_plus_pred_mean"] = inputs.q_plus_predictions[mask].detach().mean()
            metrics["q_plus_target_mean"] = inputs.q_plus_targets[mask].detach().mean()

    return ValueLossOutputs(loss=loss, metrics=metrics)
