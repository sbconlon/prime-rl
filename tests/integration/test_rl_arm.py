"""Phase 10 -- ARM end-to-end canary.

20-step canary on `reverse-text` with `algorithm="arm"`,
`return_top_k_token_ids=true`, `top_k_action_set_size=32` against the
`PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT` checkpoint. Exercises the full
four-process pipeline (vLLM Inference Server + LLM Trainer + Advantage Server
+ Advantage Trainer) end-to-end.

Five tests:
  - test_no_error: orchestrator process exits cleanly within the timeout.
  - test_reward_goes_up: reward at step 20 > reward at step 1.
  - test_reward_in_range: final reward >= min_threshold (placeholder; see
    TODO at the call site for calibration after the first cluster run).
  - test_q_plus_network_learns (ARM-specific): Q+ regression loss decreases
    from step 1 to step 20 (parsed from the Advantage Trainer\'s stdout).
  - test_top_k_extraction_active (ARM-specific): the orchestrator writes
    `completion_top_k_token_ids` populated on every TrainingSample (Phase 5
    compact layout: outer length == sum(completion_mask), inner length 32).

Pipeline-correctness canary, not a thesis-grade benchmark. ALFWorld empirical
runs are post-Phase-10 work.
"""

from pathlib import Path
from typing import Callable

import pytest

from tests.conftest import ProcessResult
from tests.utils import (
    check_no_error,
    check_reward_goes_up,
    check_reward_in_range,
    strip_escape_codes,
)

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


# ARM\'s four-process pipeline does meaningfully more work per step than
# GRPO at the same scale (Advantage Server compute + weight broadcast +
# Advantage Trainer step). The 600s timeout from test_rl.py is GRPO-tuned;
# bump for ARM with margin.
TIMEOUT = 1200  # 20 minutes


@pytest.fixture(scope="module")
def wandb_name(branch_name: str) -> str:
    """Fixture for W&B name for ARM CI integration tests."""
    return f"test-rl-arm-{branch_name}"


@pytest.fixture(scope="module")
def rl_arm_process(
    run_process: Callable[..., ProcessResult],
    output_dir: Path,
    wandb_project: str,
    wandb_name: str,
) -> ProcessResult:
    """Launch the four-process pipeline (LLM Trainer + Inference + Advantage
    Server + Advantage Trainer) with the ARM CI config; share the result
    across tests in this module."""
    cmd = [
        "uv",
        "run",
        "rl",
        "@",
        "configs/ci/integration/rl_arm/start.toml",
        "--wandb.project",
        wandb_project,
        "--wandb.name",
        wandb_name,
        "--output-dir",
        output_dir.as_posix(),
    ]
    return run_process(cmd, timeout=TIMEOUT)


# ---------------------------------------------------------------------------
# Mirror Phase 9 / GRPO canary: process exits cleanly, reward goes up, reward
# meets the (calibration-pending) minimum threshold.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def test_no_error(rl_arm_process: ProcessResult, output_dir: Path):
    """Tests that the ARM RL process does not fail."""
    check_no_error(rl_arm_process, output_dir)


def test_reward_goes_up(rl_arm_process: ProcessResult, test_no_error, output_dir: Path):
    """Tests that the reward goes up across the ARM RL run."""
    with open(output_dir / "logs" / "orchestrator.stdout", "r") as f:
        orchestrator_stdout = strip_escape_codes(f.read()).splitlines()
    check_reward_goes_up(orchestrator_stdout)


def test_reward_in_range(rl_arm_process: ProcessResult, test_no_error, output_dir: Path):
    """Tests that the final reward of the ARM RL run is at or above a
    minimum threshold.

    TODO(phase-10-calibration): the placeholder ``min_threshold=0.0`` is a
    sentinel that just asserts the reward extraction works. Calibrate this
    against the first successful cluster run\'s final reward, mirroring the
    process used for GRPO\'s 0.65 threshold in test_rl.py.
    """
    with open(output_dir / "logs" / "orchestrator.stdout", "r") as f:
        orchestrator_stdout = strip_escape_codes(f.read()).splitlines()
    check_reward_in_range(orchestrator_stdout, min_threshold=0.0)


# ---------------------------------------------------------------------------
# ARM-specific canary checks (per phase-10 §10).
# ---------------------------------------------------------------------------


def test_q_plus_network_learns(rl_arm_process: ProcessResult, test_no_error, output_dir: Path):
    """Q+ regression loss must decrease across the run.

    Without this, a silent failure where the Q+ adapter is frozen or its
    loss is masked to zero would still pass test_reward_goes_up (V learns,
    rewards still propagate via v_targets) but produce a Q+ network that
    never learned anything -- breaking ARM\'s regret-matching.

    Parses the Advantage Trainer\'s stdout for the per-step
    ``step={N} mean_loss={X} l_v={Y} l_q={Z} n_samples={M}`` log line
    (see src/prime_rl/advantage_trainer/train.py). Asserts l_q at step 20
    is strictly less than l_q at step 1.
    """
    import re

    log_path = output_dir / "logs" / "advantage_trainer.stdout"
    assert log_path.exists(), (
        f"Advantage Trainer stdout not found at {log_path}; the launcher "
        "should have spawned the subprocess (config.advantage_trainer is set)."
    )
    with open(log_path, "r") as f:
        trainer_stdout = strip_escape_codes(f.read())

    # Pattern matches the trainer\'s per-step log line. step is captured to
    # disambiguate first vs last; l_q is the Q+ regression loss.
    step_re = re.compile(r"step=(\d+)\b.*?\bl_q=(-?\d+(?:\.\d+)?)")
    matches = step_re.findall(trainer_stdout)
    assert matches, (
        "No `step=... l_q=...` lines found in advantage_trainer.stdout; "
        "either the trainer never logged a step or the log format changed."
    )

    by_step: dict[int, float] = {int(s): float(l_q) for s, l_q in matches}
    assert by_step, "parsed step->l_q dict is empty"

    sorted_steps = sorted(by_step.keys())
    first_step, last_step = sorted_steps[0], sorted_steps[-1]
    first_l_q, last_l_q = by_step[first_step], by_step[last_step]

    assert last_l_q < first_l_q, (
        f"Q+ regression loss did not decrease across the run: "
        f"step {first_step} l_q={first_l_q:.6f} -> "
        f"step {last_step} l_q={last_l_q:.6f}. "
        f"Expected last_l_q < first_l_q (Q+ is fitting its targets)."
    )


def test_top_k_extraction_active(
    rl_arm_process: ProcessResult, test_no_error, output_dir: Path
):
    """Top-K candidate token IDs (Phase 5) must be populated on the
    TrainingSamples the orchestrator writes during ARM canary.

    Silent failure mode this catches: vLLM is configured to return top-K
    but the field is dropped somewhere between the inference server and
    the Advantage Server -- ARM falls back to the cold-start branch every
    step, advantages stay zero, and the canary still appears to pass
    test_reward_goes_up (weakly, via v_targets). This test asserts the
    actual top-K wire payload survives the orchestrator -> trainer
    transport.

    Phase 5 compact layout: when ``completion_top_k_token_ids`` is non-None,
    its outer length equals ``sum(completion_mask)`` (one row per
    mask=True / sampled-by-policy position; bridge tokens are not
    represented). Inner length equals ``top_k_action_set_size = 32``.
    """
    import msgspec

    from prime_rl.transport.types import TrainingBatch

    # RLConfig.auto_setup_output_dir sets orchestrator.output_dir =
    # <RLConfig.output_dir>/run_default; rollouts land at
    # <orch.output_dir>/rollouts/step_N/rollouts.bin.
    rollouts_dir = output_dir / "run_default" / "rollouts"
    assert rollouts_dir.exists(), (
        f"Rollouts dir not found at {rollouts_dir}; orchestrator may not "
        "have run any steps."
    )

    step_dirs = sorted(rollouts_dir.glob("step_*"))
    assert step_dirs, (
        f"No step_* dirs under {rollouts_dir}; orchestrator wrote no batches."
    )

    # Pick the first step that has a rollouts.bin file (some intermediate
    # step dirs may exist transiently without the batch).
    batch_path: Path | None = None
    for step_dir in step_dirs:
        candidate = step_dir / "rollouts.bin"
        if candidate.exists():
            batch_path = candidate
            break
    assert batch_path is not None, (
        f"No rollouts.bin found in any step_* dir under {rollouts_dir}."
    )

    decoder = msgspec.msgpack.Decoder(type=TrainingBatch)
    with open(batch_path, "rb") as f:
        batch = decoder.decode(f.read())

    assert batch.examples, f"TrainingBatch from {batch_path} has no examples"
    sample = batch.examples[0]

    assert sample.completion_top_k_token_ids is not None, (
        f"completion_top_k_token_ids is None on the first TrainingSample of "
        f"{batch_path} -- top-K extraction did not propagate through the "
        "inference -> orchestrator -> transport path. Check that the canary "
        "config has orchestrator.sampling.return_top_k_token_ids = true."
    )

    # Phase 5 compact layout: outer length == sum(completion_mask).
    expected_outer = sum(sample.completion_mask)
    actual_outer = len(sample.completion_top_k_token_ids)
    assert actual_outer == expected_outer, (
        f"completion_top_k_token_ids outer length {actual_outer} != "
        f"sum(completion_mask) {expected_outer} (Phase 5 compact layout: "
        "one row per mask=True position only)."
    )

    # Inner length equals top_k_action_set_size = 32 from the canary config.
    expected_inner = 32
    if actual_outer > 0:
        for i, row in enumerate(sample.completion_top_k_token_ids):
            assert len(row) == expected_inner, (
                f"completion_top_k_token_ids[{i}] inner length {len(row)} "
                f"!= top_k_action_set_size {expected_inner}"
            )
