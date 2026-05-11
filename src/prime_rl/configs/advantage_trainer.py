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

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from prime_rl.configs.shared import FileSystemTransportConfig, LogConfig, TransportConfig
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

    # Phase 10 lever 2: how many samples per forward+backward chunk inside
    # a training step. Larger = better GPU utilization (matmul efficiency),
    # but more activation memory per chunk. Default 8 fits comfortably on
    # a 40GB MIG slice at canary-shape sequences. Set to 1 to recover the
    # pre-lever-2 per-sample backward behavior (useful for debugging or
    # parity checks against earlier runs).
    inner_batch_size: Annotated[
        int,
        Field(
            ge=1,
            description=(
                "Samples per forward+backward chunk inside a training step. "
                "1 = per-sample backward (pre-lever-2); 8 = default; larger "
                "amortizes Python overhead but grows activation memory."
            ),
        ),
    ] = 8

    # Where to write logs / checkpoints. Mirrors trainer.output_dir.
    output_dir: Annotated[
        Path,
        Field(description="Directory to write outputs to (logs, checkpoints, broadcast staging)."),
    ] = Path("outputs/advantage_trainer")

    # Orchestrator -> Advantage Trainer transport. Default to filesystem (debug
    # / single-node); production typically uses ZMQ.
    transport: Annotated[
        TransportConfig,
        Field(description="Transport for receiving AdvantageTrainingBatch from the orchestrator."),
    ] = FileSystemTransportConfig()

    # Phase 7c: directory the orchestrator writes AdvantageTrainingBatch
    # files into (filesystem transport only). The launch script wires this
    # to <orchestrator.output_dir>/advantage_trainer_transport so sender
    # and receiver agree on layout. Required when transport.type ==
    # "filesystem"; ignored when transport.type == "zmq".
    transport_input_dir: Annotated[
        Path | None,
        Field(
            description=(
                "Directory to read AdvantageTrainingBatch files from "
                "(filesystem transport only). Set by the launch script "
                "to match the orchestrator's sender output_dir."
            )
        ),
    ] = None

    # Log level + structured-log toggle.
    log: Annotated[
        LogConfig,
        Field(description="Logging configuration."),
    ] = LogConfig()

    # URL of the Advantage Server (Phase 7c). When set, the Trainer POSTs
    # serialized LoRA + value-head weights to {url}/update_weights after each
    # gradient step. When None, the Trainer trains in isolation (useful for
    # standalone tests / smoke runs without a Server).
    advantage_server_url: Annotated[
        str | None,
        Field(description="Base URL of the Advantage Server (Phase 7c weight broadcast target). None to disable broadcast."),
    ] = None

    # Phase 7d / future: optim, scheduler, ckpt, wandb, heartbeat,
    # metrics_server. For 7c, the algorithmic-core-via-train.py uses AdamW
    # with `learning_rate` directly and stubs ckpt.
