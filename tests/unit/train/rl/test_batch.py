"""Phase 1: prepare_sample per-token advantages contract.

Tests the invariant introduced when TrainingSample.advantage (scalar) was
lifted to TrainingSample.advantages (per-token list of length
len(completion_ids)).
"""

from prime_rl.trainer.batch import prepare_sample
from prime_rl.transport.types import TrainingSample


def test_prepare_sample_advantages_mapping():
    """Per-token advantages map to completion positions; prompt positions are zero."""
    sample = TrainingSample(
        prompt_ids=[1, 2],
        prompt_mask=[False, False],
        completion_ids=[10, 20, 30],
        completion_mask=[True, True, True],
        completion_logprobs=[-0.1, -0.2, -0.3],
        completion_temperatures=[1.0, 1.0, 1.0],
        advantages=[0.5, -0.3, 0.8],
    )

    micro_batch = prepare_sample(sample, seq_len=10)

    assert micro_batch.advantages == [0.0, 0.0, 0.5, -0.3, 0.8]
    assert len(micro_batch.advantages) == len(micro_batch.input_ids)


def test_prepare_sample_advantages_none_defaults_to_zeros():
    """Phase 1 decision: advantages=None produces an all-zero MicroBatch.advantages.

    Rationale (see plan/phases/phase-01-data-contract.md §10): a TrainingSample whose
    advantages have not been assigned yet should not crash the packer. Zero advantages
    produce zero gradient at completion positions, matching the prior "absent advantage
    -> no signal" semantics without leaking None into the tensor pipeline.
    """
    sample = TrainingSample(
        prompt_ids=[1, 2],
        prompt_mask=[False, False],
        completion_ids=[10, 20, 30],
        completion_mask=[True, True, True],
        completion_logprobs=[-0.1, -0.2, -0.3],
        completion_temperatures=[1.0, 1.0, 1.0],
        advantages=None,
    )

    micro_batch = prepare_sample(sample, seq_len=10)

    assert micro_batch.advantages == [0.0, 0.0, 0.0, 0.0, 0.0]
    assert len(micro_batch.advantages) == len(micro_batch.input_ids)
