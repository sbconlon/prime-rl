import msgspec


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


class TrainingBatch(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    """A batch of training examples with metadata for transport."""

    examples: list[TrainingSample]
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
