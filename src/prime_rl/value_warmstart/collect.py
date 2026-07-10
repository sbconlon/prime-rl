"""Offline warm-start data generation (Phase 2).

Runs the fixed SFT policy over the ALFWorld train games and dumps the rollouts
(interleaved TrainingSamples + reward + is_truncated) as the warm-start dataset.
Reuses the orchestrator's rollout building blocks -- `vf_utils.evaluate` is the
generation driver, `interleave_rollout` builds the decision_points -- so there is
no new rollout engine. See phase-02-data-generation.md.
"""

from __future__ import annotations

import asyncio

import verifiers as vf

from prime_rl.configs.value_warmstart import CollectConfig
from prime_rl.orchestrator.eval_utils import get_eval_sampling_args
from prime_rl.orchestrator.trajectories import interleave_rollout
from prime_rl.orchestrator.vf_utils import evaluate, setup_env_client, spawn_env_server
from prime_rl.utils.client import setup_inference_pool
from prime_rl.utils.config import cli
from prime_rl.utils.logger import setup_logger
from prime_rl.value_warmstart.dataset_io import RecordWriter, WarmStartRecord


def keep_output(output: vf.RolloutOutput) -> bool:
    """Include successes AND task-failures (reward 0 and 1) for calibration;
    exclude infrastructure-error rollouts and empty completions."""
    return output.get("error") is None and bool(output.get("completion"))


async def collect(config: CollectConfig) -> None:
    logger = setup_logger(config.log.level)
    logger.info(
        f"Warm-start collect: env={config.env.id} "
        f"split={config.env.args.get('split')} "
        f"num_reasoning_blocks={config.env.args.get('num_reasoning_blocks')}"
    )

    env = vf.load_environment(config.env.id, **config.env.args)
    env_name = config.env.name or config.env.id

    address, _process = spawn_env_server(
        env_id=config.env.id,
        env_args=config.env.args,
        extra_env_kwargs=config.env.extra_env_kwargs,
        num_workers=config.num_workers,
    )
    env.env_client = setup_env_client(address=address, name=env_name)

    # EnvGroup only to resolve the dataset size when num_examples is unset.
    env_group = vf.EnvGroup(
        envs=[env], env_names=[env_name], map_kwargs=dict(writer_batch_size=1)
    )
    dataset = env_group.get_dataset(seed=config.seed)
    num_examples = config.num_examples if config.num_examples is not None else len(dataset)
    logger.info(f"Collecting {config.rollouts_per_example} rollouts x {num_examples} games")

    inference_pool = await setup_inference_pool(config.client, model_name=config.model_name)
    await inference_pool.wait_for_ready(config.model_name)

    outputs = await evaluate(
        env=env,
        model_name=config.model_name,
        sampling_args=get_eval_sampling_args(config.sampling),
        num_examples=num_examples,
        rollouts_per_example=config.rollouts_per_example,
        get_client=inference_pool.get_next_client,
        max_retries=config.max_retries,
    )
    logger.info(f"Generated {len(outputs)} rollouts")

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    kept = skipped = positives = 0
    with RecordWriter(config.output_path) as writer:
        for output in outputs:
            if not keep_output(output):
                skipped += 1
                continue
            samples = interleave_rollout(output)
            if not samples:
                skipped += 1
                continue
            reward = float(output["reward"])
            writer.write(
                WarmStartRecord(
                    samples=samples,
                    reward=reward,
                    is_truncated=bool(output.get("is_truncated", False)),
                )
            )
            kept += 1
            positives += reward > 0.0

    pos_fraction = positives / kept if kept else 0.0
    logger.success(
        f"Wrote {kept} records to {config.output_path} "
        f"(skipped {skipped}; positive-label fraction {pos_fraction:.3f})"
    )
    await inference_pool.stop()


def main() -> None:
    """Entry point: `uv run warmstart-collect @ collect.toml`."""
    asyncio.run(collect(cli(CollectConfig)))


if __name__ == "__main__":
    main()
