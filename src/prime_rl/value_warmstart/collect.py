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
from prime_rl.orchestrator.vf_utils import run_rollout, setup_env_client, spawn_env_server
from prime_rl.utils.client import setup_inference_pool
from prime_rl.utils.config import cli
from prime_rl.utils.logger import setup_logger
from prime_rl.value_warmstart.dataset_io import RecordWriter, WarmStartRecord


def keep_output(output: vf.RolloutOutput) -> bool:
    """Include successes AND task-failures (reward 0 and 1) for calibration;
    exclude infrastructure-error rollouts and empty completions."""
    return output.get("error") is None and bool(output.get("completion"))


async def _run_bounded_collect(
    *,
    env: vf.Environment,
    inputs: list[dict],
    get_client,
    model_name: str,
    sampling_args: dict,
    max_retries: int,
    max_concurrent: int,
    rollout_timeout_s: float,
    writer,
    logger,
) -> dict[str, int]:
    """Bounded, fill-as-completed rollout driver that mirrors the RL Scheduler's
    discipline, replacing the old evaluate()/generate() all-or-nothing gather.

    Why: the orchestrator never wedges on ALFWorld because it caps in-flight
    rollouts (max_inflight_rollouts) and advances on whatever completes, leaving a
    stuck rollout behind. The old collect fanned out ALL rollouts via one
    asyncio.gather with no cap -- 2000 concurrent resets serialized through the
    single _TW_PARSE_LOCK (GPU-starved), and any single wedged rollout stalled the
    whole gather. This driver:
      * caps concurrency at `max_concurrent` (a Semaphore) so the lock is never swamped;
      * starts the per-rollout timeout only AFTER a slot is acquired, so it times real
        work, not queue wait -- a wedged rollout is dropped and its slot reclaimed;
      * writes each kept record incrementally, so a walltime/OOM kill keeps partial data
        (the old path wrote nothing until the entire gather returned).
    """
    sem = asyncio.Semaphore(max_concurrent)
    write_lock = asyncio.Lock()
    total = len(inputs)
    stats = {"done": 0, "kept": 0, "skipped": 0, "positives": 0, "timed_out": 0, "errored": 0}

    async def _one(idx: int, example: dict) -> None:
        output = None
        async with sem:  # bounded concurrency; timeout below is measured from here
            client = await get_client()
            try:
                output = await asyncio.wait_for(
                    run_rollout(
                        env=env,
                        client=client,
                        example=example,
                        model_name=model_name,
                        sampling_args=sampling_args,
                        max_retries=max_retries,
                    ),
                    timeout=rollout_timeout_s,
                )
            except asyncio.TimeoutError:
                stats["timed_out"] += 1
                logger.warning(
                    f"Rollout {idx} exceeded {rollout_timeout_s:.0f}s (wedged, or blocked "
                    f"behind a held _TW_PARSE_LOCK -- see env_worker_*.log [lock] lines); "
                    f"dropped, slot reclaimed."
                )
            except Exception as e:  # infra error on this rollout: drop it, keep going
                stats["errored"] += 1
                logger.warning(f"Rollout {idx} errored: {e}; dropped.")
        # File I/O outside the slot so the concurrency budget tracks generation, not disk.
        stats["done"] += 1
        if output is not None and keep_output(output):
            samples = interleave_rollout(output)
            if samples:
                reward = float(output["reward"])
                async with write_lock:
                    writer.write(
                        WarmStartRecord(
                            samples=samples,
                            reward=reward,
                            is_truncated=bool(output.get("is_truncated", False)),
                        )
                    )
                stats["kept"] += 1
                stats["positives"] += int(reward > 0.0)
            else:
                stats["skipped"] += 1
        elif output is not None:
            stats["skipped"] += 1
        if stats["done"] % 50 == 0 or stats["done"] == total:
            logger.info(
                f"collect progress {stats['done']}/{total} | kept={stats['kept']} "
                f"skipped={stats['skipped']} timed_out={stats['timed_out']} "
                f"errored={stats['errored']}"
            )

    await asyncio.gather(*[_one(i, ex) for i, ex in enumerate(inputs)])
    return stats


async def collect(config: CollectConfig) -> None:
    logger = setup_logger(config.log.level)
    logger.info(
        f"Warm-start collect: env={config.env.id} "
        f"split={config.env.args.get('split')} "
        f"num_reasoning_blocks={config.env.args.get('num_reasoning_blocks')}"
    )

    env = vf.load_environment(config.env.id, **config.env.args)
    env_name = config.env.name or config.env.id

    # Route the env-server WORKER logs to files. spawn_env_server forces
    # console_logging=False, so without a log_dir the workers' stdlib logs (incl.
    # the [lock] wedge-diagnosis lines in alfworld_env._logged_parse_lock) go
    # NOWHERE -- which is why prior hangs were invisible. With log_dir set, each
    # worker writes <output_dir>/env_worker_<id>.log at INFO: tail/grep those to
    # see exactly which game holds _TW_PARSE_LOCK and never releases it.
    log_dir = str(config.output_path.parent)
    address, env_server_process = spawn_env_server(
        env_id=config.env.id,
        env_args=config.env.args,
        extra_env_kwargs=config.env.extra_env_kwargs,
        num_workers=config.num_workers,
        log_level="INFO",
        log_dir=log_dir,
    )
    logger.info(f"Env-server worker logs -> {log_dir}/env_worker_<id>.log")
    try:
        env.env_client = setup_env_client(address=address, name=env_name)

        # EnvGroup only to resolve the dataset size when num_examples is unset.
        env_group = vf.EnvGroup(
            envs=[env], env_names=[env_name], map_kwargs=dict(writer_batch_size=1)
        )
        dataset = env_group.get_dataset(seed=config.seed)
        num_examples = config.num_examples if config.num_examples is not None else len(dataset)
        logger.info(f"Collecting {config.rollouts_per_example} rollouts x {num_examples} games")

        # ACTION-LEVEL: interleave_rollout needs per-step token IDs (trajectories.py:329
        # "Missing rollout tokens" -> the sample is dropped -> 0 records). A plain chat
        # client returns text only, so use the token-in-token-out client, exactly as the RL
        # orchestrator does under use_token_client=true (orchestrator.py:121).
        inference_pool = await setup_inference_pool(
            config.client, model_name=config.model_name, client_type="openai_chat_completions_token"
        )
        await inference_pool.wait_for_ready(config.model_name)

        # Bounded, fill-as-completed generation (mirrors the RL Scheduler) in place of
        # the old evaluate()/gather. _get_eval_inputs already repeats each of the
        # num_examples games rollouts_per_example times -> a flat list of per-rollout
        # input dicts, which we drive through a concurrency-capped pool.
        inputs = env._get_eval_inputs(num_examples, config.rollouts_per_example)
        logger.info(
            f"Driving {len(inputs)} rollouts | max_concurrent={config.max_concurrent} "
            f"rollout_timeout={config.rollout_timeout_s:.0f}s | incremental write"
        )

        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        with RecordWriter(config.output_path) as writer:
            stats = await _run_bounded_collect(
                env=env,
                inputs=inputs,
                get_client=inference_pool.get_next_client,
                model_name=config.model_name,
                sampling_args=get_eval_sampling_args(config.sampling),
                max_retries=config.max_retries,
                max_concurrent=config.max_concurrent,
                rollout_timeout_s=config.rollout_timeout_s,
                writer=writer,
                logger=logger,
            )

        kept = stats["kept"]
        pos_fraction = stats["positives"] / kept if kept else 0.0
        logger.success(
            f"Wrote {kept} records to {config.output_path} "
            f"(skipped {stats['skipped']}, timed_out {stats['timed_out']}, "
            f"errored {stats['errored']}; positive-label fraction {pos_fraction:.3f})"
        )
        await inference_pool.stop()
    finally:
        # The env server is a NON-DAEMON mp.Process (daemon=False; it spawns its own
        # worker subprocesses). Python's multiprocessing atexit handler joins non-daemon
        # children on interpreter exit, and this one is an infinite serve loop that never
        # returns -> collect HANGS after writing (fatal in a non-interactive sbatch, where
        # there is no Ctrl-C). Terminate it here on both success and error so collect exits.
        env_server_process.terminate()
        env_server_process.join(timeout=15)
        if env_server_process.is_alive():
            env_server_process.kill()
            env_server_process.join(timeout=5)


def main() -> None:
    """Entry point: `uv run warmstart-collect @ collect.toml`."""
    asyncio.run(collect(cli(CollectConfig)))


if __name__ == "__main__":
    main()
