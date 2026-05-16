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
        Field(
            gt=0.0,
            description=(
                "Learning rate for the value-network LoRA adapters and value heads. "
                "Used as the fallback when v_learning_rate / q_plus_learning_rate "
                "are unset (the single-LR / pre-decoupling regime)."
            ),
        ),
    ] = 1e-3

    v_learning_rate: Annotated[
        float | None,
        Field(
            default=None,
            gt=0.0,
            description=(
                "Override learning rate for V\'s LoRA slot + v_head. If None "
                "(default), falls back to `learning_rate`. Set this to use a "
                "different LR for V than for Q+."
            ),
        ),
    ] = None

    q_plus_learning_rate: Annotated[
        float | None,
        Field(
            default=None,
            gt=0.0,
            description=(
                "Override learning rate for Q+\'s LoRA slot + q_plus_head. If None "
                "(default), falls back to `learning_rate`. Set this to use a "
                "different LR for Q+ than for V."
            ),
        ),
    ] = None

    # Phase 10 lever 2: how many samples per forward+backward chunk inside
    # ONE minibatch (memory knob). Used purely for activation-memory control.
    # Defaults to 8, comfortable on a 40GB MIG slice at canary-shape sequences.
    # When `minibatch_size <= inner_batch_size` (canary defaults), the
    # minibatch is forwarded in a single chunk and grad accumulation
    # collapses to a no-op.
    #
    # NOT to be confused with `minibatch_size` (SGD knob): inner_batch_size
    # controls how a single optimizer step's gradient is computed (one
    # chunk vs. K accumulated chunks both producing the same final
    # gradient up to floating-point ordering); minibatch_size controls how
    # often `optimizer.step()` fires within a cycle.
    inner_batch_size: Annotated[
        int,
        Field(
            ge=1,
            description=(
                "Activation-memory chunk size INSIDE one minibatch. "
                "When >= minibatch_size the minibatch is forwarded in one "
                "chunk; smaller values use gradient accumulation across "
                "multiple chunks before each optimizer step. Memory-only "
                "knob; does not affect SGD semantics (the per-minibatch "
                "mean-of-per-sample-means is recovered regardless)."
            ),
        ),
    ] = 8

    # ---- Jin-aligned inner training loop (2026-05-16 spec) ---------------
    #
    # Per-cycle structure:
    #   1 cycle  = n_epochs epochs over the cycle's batch
    #   1 epoch  = ceil(batch_size / minibatch_size) minibatches (reshuffled)
    #   1 mb     = 1 optimizer.step() + grad accumulation over ceil(MB/IB) chunks
    # Polyak update + weight broadcast fire ONCE per cycle, after the inner
    # loop completes. Targets are frozen for the cycle (structurally
    # enforced by the transport boundary: AdvServer produced them under
    # V_prev / Q+_prev and they cannot be recomputed without those weights).
    #
    # At canary defaults (batch_size=128, minibatch_size=32, n_epochs=8):
    # 4 minibatches/epoch x 8 epochs = 32 inner Adam steps per cycle. Matches
    # Jin's intent of running the value-network regression to (near-)
    # convergence within each cycle before letting the policy respond,
    # closing the race condition diagnosed in 20260516.

    n_epochs: Annotated[
        int,
        Field(
            ge=1,
            description=(
                "Number of passes over the cycle's batch per AdvTrainer step. "
                "Jin's effective epoch count is ~7.7 (3000 grad steps x mb=32 "
                "/ 12,500 transitions). 8 is the spec default; ablate up to "
                "16 if needed. Past that, risk overfitting LoRA adapters to a "
                "single cycle's batch."
            ),
        ),
    ] = 8

    minibatch_size: Annotated[
        int,
        Field(
            ge=1,
            description=(
                "Samples per optimizer step (Jin's MB_SIZE). When less than "
                "the cycle's batch size, drives genuine minibatch SGD: each "
                "epoch shuffles the batch and the optimizer takes one step "
                "per minibatch. When >= batch size, each epoch is a single "
                "full-batch step. Spec default 32; combines with n_epochs=8 "
                "and batch_size=128 to give 32 inner steps per cycle."
            ),
        ),
    ] = 32

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
