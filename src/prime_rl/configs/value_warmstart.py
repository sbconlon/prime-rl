"""Configs for the value-network warm-start: `warmstart-collect` (data
generation) and `warmstart-train` (offline value regression)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import Field

from prime_rl.configs.orchestrator import EnvConfig, EvalSamplingConfig
from prime_rl.configs.shared import ClientConfig, LogConfig
from prime_rl.orchestrator.value_networks import ValueNetworkConfig
from prime_rl.configs.trainer import CheckpointConfig
from prime_rl.utils.config import BaseConfig


class CollectConfig(BaseConfig):
    """Generate warm-start data: run the fixed SFT policy over the ALFWorld train
    games and dump rollouts. See phase-02-data-generation.md."""

    model_name: Annotated[
        str,
        Field(description="SFT checkpoint path (served on vLLM; also the env tokenizer)."),
    ]
    env: Annotated[
        EnvConfig,
        Field(
            description=(
                "ALFWorld env. Set args.split='train', args.max_context_tokens=2048, "
                "args.num_reasoning_blocks=1 (m-blocks/pi_hat off). Reuse the RL run's "
                "train [env] args."
            ),
        ),
    ]
    client: Annotated[
        ClientConfig,
        Field(description="Client to the already-running vLLM inference server (same type as the RL rollout client)."),
    ]
    sampling: Annotated[
        EvalSamplingConfig,
        Field(description="Rollout sampling args (match the RL rollout temperature)."),
    ] = EvalSamplingConfig()
    rollouts_per_example: Annotated[
        int,
        Field(ge=1, description="Rollouts generated per game."),
    ] = 16
    num_examples: Annotated[
        int | None,
        Field(default=None, description="Number of games. None => the full env dataset."),
    ] = None
    num_workers: Annotated[
        int,
        Field(ge=1, description="Env-server workers."),
    ] = 8
    max_retries: Annotated[
        int,
        Field(ge=0, description="Per-rollout retries."),
    ] = 3
    output_path: Annotated[
        Path,
        Field(description="Destination framed-msgpack dataset file."),
    ]
    seed: Annotated[
        int,
        Field(description="Dataset seed."),
    ] = 0
    log: Annotated[
        LogConfig,
        Field(description="Logging configuration."),
    ] = LogConfig()


class WarmStartDataConfig(BaseConfig):
    """Where the Phase-2 dataset lives + batch shape for the warm-start loop."""

    path: Annotated[
        Path,
        Field(description="Framed-msgpack dataset produced by warmstart-collect."),
    ]
    batch_size: Annotated[
        int,
        Field(ge=1, description="AdvantageTrainingSamples per optimizer step."),
    ] = 32
    inner_batch_size: Annotated[
        int,
        Field(ge=1, description="Activation-memory chunk size inside one batch (grad accumulates)."),
    ] = 8


class WarmStartTrainConfig(BaseConfig):
    """Offline value regression: pretrain V and Q+ on MC returns, emit value_state.pt.
    See phase-03-training-pipeline.md."""

    model: Annotated[
        ValueNetworkConfig,
        Field(description="Value network backbone (frozen SFT base + 3 LoRA slots + 3 heads)."),
    ]
    data: Annotated[
        WarmStartDataConfig,
        Field(description="Dataset location + batch shape."),
    ]
    v_learning_rate: Annotated[
        float,
        Field(gt=0.0, description="Learning rate for V's LoRA slot + v_head."),
    ] = 1e-3
    q_plus_learning_rate: Annotated[
        float,
        Field(gt=0.0, description="Learning rate for Q+'s LoRA slot + q_plus_head."),
    ] = 1e-3
    max_epochs: Annotated[
        int,
        Field(ge=1, description="Epochs over the fixed dataset (watch the loss plateau)."),
    ]
    output_dir: Annotated[
        Path,
        Field(description="Output dir; value_state.pt is written under checkpoints/step_0/trainer/."),
    ] = Path("outputs/value_warmstart")
    ckpt: Annotated[
        CheckpointConfig,
        Field(description="Checkpoint config. Set skip_optimizer=True for a weights-only artifact."),
    ]
    seed: Annotated[
        int,
        Field(description="Shuffle seed (offset per epoch)."),
    ] = 0
    log: Annotated[
        LogConfig,
        Field(description="Logging configuration."),
    ] = LogConfig()
