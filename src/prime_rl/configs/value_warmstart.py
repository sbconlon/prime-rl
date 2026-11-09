"""Configs for the value-network warm-start: `warmstart-collect` (data
generation) and `warmstart-train` (offline value regression)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from prime_rl.configs.orchestrator import EnvConfig, EvalSamplingConfig
from prime_rl.configs.shared import ClientConfig, LogConfig
from prime_rl.orchestrator.value_networks import ValueNetworkConfig
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
    max_concurrent: Annotated[
        int,
        Field(
            ge=1,
            description=(
                "Max rollouts in flight at once (mirrors the RL orchestrator's "
                "max_inflight_rollouts). The old evaluate()/gather path fanned out ALL "
                "rollouts uncapped, swamping the env's single _TW_PARSE_LOCK and making "
                "the per-rollout timeout meaningless (it timed dispatch-queue wait, not "
                "work). Bounding concurrency keeps the lock uncontended and the timeout real."
            ),
        ),
    ] = 16
    rollout_timeout_s: Annotated[
        float,
        Field(
            gt=0.0,
            description=(
                "Per-rollout wall-clock timeout, measured from when the rollout ACQUIRES a "
                "concurrency slot (real work time, since concurrency is capped). A rollout "
                "that exceeds it -- e.g. wedged behind a held _TW_PARSE_LOCK -- is dropped "
                "and its slot reclaimed, so the collect proceeds instead of hanging. Set "
                "well above a healthy multi-turn episode."
            ),
        ),
    ] = 600.0
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
    kind: Annotated[
        Literal["warmstart", "branched"],
        Field(
            description=(
                "Dataset kind: 'warmstart' = single-action WarmStartRecords (warmstart-collect); "
                "'branched' = BranchRecords (branch-collect; plan branched-dataset Phase 3)."
            )
        ),
    ] = "warmstart"
    holdout_ids_path: Annotated[
        Path | None,
        Field(
            description=(
                "JSON list of game ids to hold out (branched only), in addition to records "
                "flagged holdout by the collect. Excluded ids are written to output_dir/holdout.json."
            )
        ),
    ] = None
    include_holdout: Annotated[
        bool,
        Field(description="Train on held-out records too (debug only; defeats the gate's G-A)."),
    ] = False
    min_p_accept: Annotated[
        float | None,
        Field(
            ge=0.0,
            le=1.0,
            description=(
                "Branched only: drop branch records whose forced root was accepted with "
                "p(action | o, reasoning) below this (BranchRecord.p_accept; D-17). None keeps all."
            ),
        ),
    ] = None


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
    seed: Annotated[
        int,
        Field(description="Shuffle seed (offset per epoch)."),
    ] = 0
    log: Annotated[
        LogConfig,
        Field(description="Logging configuration."),
    ] = LogConfig()


ELICIT_DEFAULT = (
    "Your next action has been decided: {action}. Write your reasoning for this "
    "step in your usual format and end with exactly that action."
)

LEAK_DEFAULT = [
    r"\bdecided\b",
    r"\btold\b",
    r"\binstruct",
    r"\bgiven action\b",
    r"\basked to\b",
    r"\bchosen for me\b",
    r"\bmust take\b",
    r"\bhas been decided\b",
]


class BranchCollectConfig(CollectConfig):
    """Generate the branched (Job-B) warm-start dataset: trunks from the frozen SFT
    policy, branches at selected roots with a forced, congruent action, labels for
    every (o, a) leg. See plan/branched-dataset (master doc §2, D-4, D-8, D-9)."""

    # --- roots (D-9) ---
    roots_per_trunk: Annotated[
        int,
        Field(ge=1, description="B: roots per trunk under the stratified allocation."),
    ] = 6
    allocation: Annotated[
        Literal["stratified", "two_stage"],
        Field(description="Root allocation: stratified with early tilt, or two-stage hindsight."),
    ] = "stratified"
    early_tilt: Annotated[
        float,
        Field(ge=0.0, le=1.0, description="Share of roots placed in the first third of the trunk."),
    ] = 0.5
    stage1_roots: Annotated[
        int,
        Field(ge=1, description="B1: candidate roots probed with one branch (two_stage)."),
    ] = 10
    stage2_roots: Annotated[
        int,
        Field(ge=0, description="B2: pivotal roots that receive the full fan (two_stage)."),
    ] = 4
    # --- actions at a root (D-7, D-8) ---
    branches_per_root: Annotated[
        int,
        Field(ge=1, description="K: actions branched per root."),
    ] = 4
    n_uniform: Annotated[
        int,
        Field(ge=0, description="n_unif: uniform draws over the admissible tail per root."),
    ] = 2
    node_blocks: Annotated[
        int,
        Field(ge=2, description="m: sampled reasoning blocks at a root (plus one greedy)."),
    ] = 8
    # --- synthesis (D-4) ---
    tau: Annotated[
        float,
        Field(gt=0.0, lt=1.0, description="Acceptance threshold on the normalized row value at the action."),
    ] = 0.5
    theta: Annotated[
        float | None,
        Field(
            ge=0.0,
            lt=1.0,
            description=(
                "Weak acceptance (D-17): when no candidate reaches tau, the best of the sampled blocks "
                "and the elicited attempts is taken if its row value at the action is >= theta. "
                "None = tau (strong acceptance only). The value lands in BranchRecord.p_accept, so "
                "data.min_p_accept can tighten it at train time without recollecting."
            ),
        ),
    ] = None
    n_elicit: Annotated[
        int,
        Field(ge=0, description="Elicitation attempts per action without a congruent sampled block."),
    ] = 3
    elicit_prompt: Annotated[
        str,
        Field(description="Instruction appended as a user message; {action} is substituted. Never stored."),
    ] = ELICIT_DEFAULT
    leak_patterns: Annotated[
        list[str],
        Field(description="Case-insensitive regexes; a block matching any is rejected as a leak."),
    ] = LEAK_DEFAULT
    # --- budget guards / IO ---
    max_branches_per_game: Annotated[
        int,
        Field(ge=1, description="Hard cap on branch episodes per game."),
    ] = 32
    games: Annotated[
        int | None,
        Field(description="Number of train games to process (None = env dataset size)."),
    ] = None
    holdout_games: Annotated[
        int,
        Field(ge=0, description="Last N games are tagged held-out (for the gate's G-A)."),
    ] = 50
    output_dir: Annotated[
        Path,
        Field(description="Run directory: records.msgpack, nodes/, done/, collect_stats.json."),
    ] = Path("outputs/branch_collect")
    resume: Annotated[
        bool,
        Field(description="Skip games with a done marker in output_dir/done/."),
    ] = True

    @model_validator(mode="after")
    def _validate_branching(self) -> "BranchCollectConfig":
        if self.n_uniform > self.branches_per_root:
            raise ValueError("n_uniform must be <= branches_per_root")
        if self.stage2_roots > self.stage1_roots:
            raise ValueError("stage2_roots must be <= stage1_roots")
        if self.theta is not None and self.theta > self.tau:
            raise ValueError("theta must be <= tau")
        return self
