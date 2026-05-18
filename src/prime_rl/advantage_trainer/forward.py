"""Forward step for the Advantage Trainer.

This is the wrapper that produces V and Q+ predictions from a packed
input sequence using the Phase 6.5 optimized all-positions forwards on
ValueNetworkBackbone. The output tensors retain their autograd graphs so
the loss's backward pass can compute gradients w.r.t. the LoRA adapters
and value heads.

The Advantage Trainer's forward step is far simpler than the Advantage
Server's compute layer: the Trainer regresses against fixed scalar
targets, so it only needs Q+ at the SAMPLED action (already encoded in
input_ids), never at K=32 candidates. Two forward passes per training
step's batch (V and Q+); for PPO the Q+ forward is skipped.
"""

from __future__ import annotations

from typing import Literal

from torch import Tensor

from prime_rl.advantage_trainer._prof import prof
from prime_rl.orchestrator.value_networks import ValueNetworkBackbone


def value_forward(
    backbone: ValueNetworkBackbone,
    input_ids: Tensor,
    algorithm: Literal["ppo", "arm"],
) -> tuple[Tensor, Tensor | None]:
    """Run the Advantage Trainer's forward pass.

    Args:
        backbone: ValueNetworkBackbone (V/V_target/Q+ as 3 LoRA slots).
        input_ids: [batch, seq] packed token sequence (prompt + completion).
        algorithm: "ppo" returns only V predictions; "arm" returns both
            V and Q+_sampled predictions.

    Returns:
        v_predictions: [batch, seq] V output at every token position.
        q_plus_predictions: [batch, seq] Q+ at the sampled action at every
            token position (ARM only); None for PPO.

    Notes:
        - The KV cache returned by the all-positions forwards is discarded
          here -- the Trainer doesn't need it (no candidate evaluation).
        - V_target is NOT forwarded. V_target receives no gradient training;
          its weights track V via Polyak averaging only.
        - The forwards are NOT wrapped in torch.no_grad() -- the Trainer
          NEEDS the autograd graph for the subsequent loss.backward() call.
    """
    B, S = input_ids.shape
    with prof("advtrainer.forward.v", sync_cuda=True, B=B, S=S):
        v_predictions, _ = backbone.forward_v_all_positions(input_ids)

    if algorithm == "arm":
        with prof("advtrainer.forward.q_plus", sync_cuda=True, B=B, S=S):
            q_plus_predictions, _ = backbone.forward_q_plus_sampled_all_positions(input_ids)
    else:
        q_plus_predictions = None

    return v_predictions, q_plus_predictions
