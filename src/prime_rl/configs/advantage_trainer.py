"""Configuration for the Advantage Trainer (Phase 7).

The Advantage Trainer process consumes AdvantageTrainingBatch from the
orchestrator over ZMQ/filesystem transport, runs gradient updates on the
value network's V and Q+ LoRA adapters and value heads via MSE regression
against the targets the Advantage Server produced, performs Polyak averaging
to update V_target's adapter slot, and broadcasts the updated weights back
to the Advantage Server.

This config mirrors `prime_rl.configs.trainer.TrainerConfig` where applicable;
differs in the model (value backbone, not policy LM), the loss (value
regression, not policy gradient), and the broadcast target (Advantage Server,
not vLLM).

Phase 7a scope: only the fields the algorithmic core (value_regression_loss_fn,
value_forward, Polyak update, single-step training tests) needs. The full
process plumbing (transport, weight broadcast, train.py entrypoint) is Phase 7b.
"""

from typing import Annotated, Literal

from pydantic import Field

from prime_rl.orchestrator.value_networks import ValueNetworkConfig
from prime_rl.utils.config import BaseConfig


class AdvantageTrainerConfig(BaseConfig):
    """Configures the Advantage Trainer process.

    See `phase-07-advantage-trainer.md` Â§5 for the full intended field set.
    Phase 7a delivers the algorithmic core; Phase 7b will extend with
    optim/scheduler/transport/broadcast/log/wandb/ckpt/heartbeat fields
    mirroring the LLM Trainer's TrainerConfig.
    """

    # Value network architecture (V/V_target/Q+ as 3 LoRA slots on a frozen base).
    model: Annotated[
        ValueNetworkConfig,
        Field(description="Value network backbone (frozen base + 3 LoRA slots + 3 heads)."),
    ]

    # Algorithm selector. Determines whether Q+ training is active.
    # PPO: only V is trained (q_plus_targets are None on incoming batches).
    # ARM: both V and Q+ are trained (q_plus_targets populated).
    algorithm: Annotated[
        Literal["ppo", "arm"],
        Field(description="Algorithm. PPO trains only V; ARM trains V and Q+."),
    ]

    # V_target Polyak averaging rate. Applied after every gradient step:
    #   V_target â (1-Ï) V_target + Ï V
    # Default 0.005 matches DDPG/SAC/TD3 conventions and is the value the
    # Phase 4 ValueNetworkBackbone defaults to.
    polyak_tau: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description="Polyak averaging rate for V_target update. 0 freezes V_target; 1 hard-copies V each step.",
        ),
    ] = 0.005

    # Number of optimizer steps before the trainer exits. The launch script
    # (Phase 7b) coordinates this with the LLM Trainer's max_steps.
    max_steps: Annotated[
        int,
        Field(ge=1, description="Number of training steps before the trainer exits."),
    ] = 1000

    # Per-step optimizer learning rate. The full OptimConfig and SchedulerConfig
    # from prime_rl.configs.trainer get plumbed in Phase 7b; for Phase 7a the
    # algorithmic-core tests construct their own optimizers directly.
    learning_rate: Annotated[
        float,
        Field(gt=0.0, description="Learning rate for the value-network LoRA adapters and value heads."),
    ] = 1e-3
