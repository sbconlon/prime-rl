import msgspec


# Action-level ARM: per-decision-point metadata (one ALFWorld decision per turn).
class DecisionPoint(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    """Per-turn metadata for action-level ARM. One per ALFWorld decision point.

    Attached to the TrainingSample whose completion_ids contain this turn's
    response (a sample may contain several turns when interleave_rollout merges
    them under the extension property). None of these fields exist for GRPO/PPO
    or for token-level ARM.
    """

    # Response span in *completion-id space* of the owning sample.
    # The observation is o = prompt_ids + completion_ids[:response_start]; the
    # advantage A(a*) broadcasts across completion_ids[response_start:response_end].
    response_start: int
    response_end: int
    # Admissible action TEXT. Includes the executed action by the union invariant
    # (DQ1.4) -- substitute_executed_into_admissible is applied in interleave_rollout.
    admissible_actions: list[str]
    # Index of the executed action a* within admissible_actions.
    executed_action_idx: int
    # π̂(a*|o), the RBMC marginal. Reserved here; populated in Phase 4. None until then.
    pi_hat: float | None = None


# Orchestrator -> Packer
class TrainingSample(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    """A single training example."""

    prompt_ids: list[int]
    prompt_mask: list[bool]
    completion_ids: list[int]
    completion_mask: list[bool]
    completion_logprobs: list[float]
    completion_temperatures: list[float]  # Per-token temperatures used during generation
    teacher_logprobs: list[float] | None = None
    # Per-token advantages over the completion. Invariant when non-None:
    # len(advantages) == len(completion_ids). For GRPO this is the scalar
    # advantage broadcast across tokens; PPO/ARM populate genuine per-token
    # values. Prompt-position zeros are filled in by the packer, not stored
    # here.
    advantages: list[float] | None = None
    reward: float | None = None

    # Multimodal fields (Qwen3-VL) — pixel_values stored as raw float32 bytes for efficient serialization
    pixel_values: bytes | None = None
    pixel_values_shape: list[int] | None = None  # [num_patches, patch_dim]
    # image_grid_thw: grid dimensions [num_images, 3] where each entry is [temporal, height, width]
    image_grid_thw: list[list[int]] | None = None

    routed_experts: list[list[list[int]]] | None = None  # [seq_len, layers, topk]

    # Phase 5: per-assistant-sampled-token top-K candidate token IDs for ARM
    # regret matching. *Compact layout*: when non-None, outer length equals
    # sum(completion_mask) (one row per mask=True position), NOT len(completion_ids).
    # Bridge tokens that appear inside completion_ids in multi-turn fragmented
    # rollouts (mask=False positions) are NOT represented here -- they were
    # never sampled by the policy. Inner length is top_k_action_set_size, and
    # the sampled token at the i-th mask=True position is guaranteed present
    # in completion_top_k_token_ids[i] (substitute_sampled_into_top_k upholds
    # this). None for GRPO and PPO.
    completion_top_k_token_ids: list[list[int]] | None = None

    # Action-level ARM: per-decision-point metadata. None for GRPO/PPO and for
    # token-level ARM. When non-None, each entry's [response_start, response_end)
    # indexes this sample's completion_ids. Built by interleave_rollout from the
    # ALFWorld env's step extras. Kept off the wire for non-ARM by omit_defaults.
    decision_points: list[DecisionPoint] | None = None


class TrainingBatch(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    """A batch of training examples with metadata for transport."""

    examples: list[TrainingSample]
    step: int
    run_idx: int | None = None


# Phase 6: Advantage Server -> Orchestrator -> Advantage Trainer.
class AdvantageTrainingSample(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    """A single advantage training example, paired 1-1 with a TrainingSample.

    Mirrors TrainingSample's prompt/completion structure exactly so the
    Advantage Trainer (Phase 7) can reuse the same packer / loss-function
    infrastructure as the LLM Trainer, just with different per-token
    regression targets.

    The per-token target lists (`v_targets`, `q_plus_targets`) follow Phase 1's
    `advantages` length convention: when non-None, len matches len(completion_ids),
    with zeros at mask=False positions (the splay applied by the advantage
    functions in `prime_rl.orchestrator.per_token_advantage`). `q_plus_targets`
    is non-None only for ARM; PPO leaves it None.
    """

    prompt_ids: list[int]
    prompt_mask: list[bool]
    completion_ids: list[int]
    completion_mask: list[bool]
    v_targets: list[float] | None = None
    q_plus_targets: list[float] | None = None


class AdvantageTrainingBatch(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    """A batch of advantage training examples for transport to the Advantage Trainer."""

    examples: list[AdvantageTrainingSample]
    step: int
    run_idx: int | None = None


# Packer -> Trainer
class MicroBatch(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    """A micro batch of data for training."""

    input_ids: list[int]
    loss_mask: list[bool]
    advantages: list[float]
    inference_logprobs: list[float]
    position_ids: list[int]
    temperatures: list[float]  # Per-token temperatures used during generation
    teacher_logprobs: list[float] | None = None
    lora_num_tokens: list[int] | None = None
    routed_experts: list[list[list[int]]] | None = None

    # Multimodal fields (Qwen3-VL) — pixel_values stored as raw float32 bytes for efficient serialization
    pixel_values: bytes | None = None
    pixel_values_shape: list[int] | None = None  # [num_patches, patch_dim]
    # image_grid_thw: grid dimensions [num_images, 3] where each entry is [temporal, height, width]
    image_grid_thw: list[list[int]] | None = None
