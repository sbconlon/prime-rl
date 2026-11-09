"""Branched (Job-B) warm-start data generation: `branch-collect`.

Per game (plan branched-dataset, master doc §2; phase-02 §5):

1. one on-policy TRUNK episode of the frozen SFT policy (ordinary rollout);
2. ROOTS along the trunk (stratified with an early tilt, or the two-stage
   hindsight allocation);
3. at each root: score the node (m sampled + 1 greedy blocks and their
   teacher-forced rows, on an in-process env instance), pick K actions (own
   candidates then uniform tail draws), synthesize a congruent reasoning block
   per action (sampled-block reuse, else elicitation, both verified);
4. each accepted (root, action) becomes a BRANCH episode: a seeded rollout
   through the env server (engine replayed to the root, forced turn injected,
   on-policy continuation), spliced onto the trunk prefix + forced step and
   interleaved like any rollout;
5. BranchRecords (trunk: labels everywhere; branch: labels from the root, V
   masked at the root) are written incrementally; node JSON and running stats
   feed the HUD and the gate.

Reuses collect.py's bounded-concurrency discipline, env-server setup, and the
token inference client. Resumable per game (done/<game_id>.json).
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

import verifiers as vf

from prime_rl.configs.value_warmstart import BranchCollectConfig
from prime_rl.orchestrator.eval_utils import get_eval_sampling_args
from prime_rl.orchestrator.trajectories import interleave_rollout
from prime_rl.orchestrator.vf_utils import run_rollout, setup_env_client, spawn_env_server
from prime_rl.utils.client import setup_inference_pool
from prime_rl.utils.config import cli
from prime_rl.utils.logger import setup_logger
from prime_rl.value_warmstart.allocation import pivotal_roots, stratified_roots
from prime_rl.value_warmstart.branch_records import (
    CollectStats,
    branch_example,
    branch_record,
    game_id_from_file,
    make_seed,
    node_json,
    splice_output,
    step_admissible,
    step_executed_action,
    trunk_id_for,
    trunk_record,
    verb_class,
    write_node_json,
)
from prime_rl.value_warmstart.collect import keep_output
from prime_rl.value_warmstart.dataset_io import RecordWriter


def _pick_actions(own, admissible, a_taken, k, n_unif, rng):
    """D-7/D-8 selection rule, implemented once in the verifiers env package
    (environments/alfworld/branching.py); resolved lazily so tests can stub it."""
    try:
        from branching import pick_actions  # installed env package layout
    except ImportError:
        from environments.alfworld.branching import pick_actions  # in-repo layout
    return pick_actions(own, admissible, a_taken, k, n_unif, rng)


def greedy_sampling_args(sampling_args: dict) -> dict:
    """The block-0 (greedy) request: T = 0, min_p off, everything else as the policy."""
    out = dict(sampling_args)
    out["temperature"] = 0.0
    extra = dict(out.get("extra_body") or {})
    extra["min_p"] = 0.0
    out["extra_body"] = extra
    return out


@dataclass
class _Ctx:
    config: BranchCollectConfig
    env: Any  # env-server-backed instance (episodes)
    node_env: Any  # in-process instance (scoring, synthesis, forced steps)
    get_client: Callable[[], Awaitable[Any]]
    model_name: str
    sampling_args: dict
    block_args: dict
    greedy_args: dict
    writer: Any
    out_dir: Path
    logger: Any
    stats: CollectStats
    n_games: int
    sem: asyncio.Semaphore
    write_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    stats_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def nodes_dir(self) -> Path:
        return self.out_dir / "nodes"

    @property
    def done_dir(self) -> Path:
        return self.out_dir / "done"

    @property
    def stats_path(self) -> Path:
        return self.out_dir / "collect_stats.json"

    async def episode(self, example: dict, label: str) -> dict | None:
        """One bounded rollout (trunk or branch); None when dropped."""
        async with self.sem:
            client = await self.get_client()
            try:
                return await asyncio.wait_for(
                    run_rollout(
                        env=self.env,
                        client=client,
                        example=example,
                        model_name=self.model_name,
                        sampling_args=self.sampling_args,
                        max_retries=self.config.max_retries,
                    ),
                    timeout=self.config.rollout_timeout_s,
                )
            except asyncio.TimeoutError:
                self.logger.warning(f"{label}: exceeded {self.config.rollout_timeout_s:.0f}s; dropped")
            except Exception as e:  # noqa: BLE001 -- infra error on one episode
                self.logger.warning(f"{label}: errored ({e!r}); dropped")
            return None

    async def write(self, record) -> None:
        async with self.write_lock:
            self.writer.write(record)

    async def bump(self, fn) -> None:
        async with self.stats_lock:
            fn(self.stats)
            self.stats.save(self.stats_path)


# ---------------------------------------------------------------------------
# Per-game pipeline
# ---------------------------------------------------------------------------


async def _score(ctx: _Ctx, step: dict, admissible: list[str]):
    client = await ctx.get_client()
    return await ctx.node_env.score_node(
        client,
        ctx.model_name,
        step["prompt"],
        admissible,
        ctx.config.node_blocks,
        ctx.block_args,
        ctx.greedy_args,
    )


async def _run_branch(
    ctx: _Ctx,
    *,
    example: dict,
    trunk_out: dict,
    steps: list,
    t: int,
    action: str,
    source: str,
    syn: Any,
    game_id: str,
    trunk_id: str,
    branch_id: str,
    stage: int | None,
    holdout: bool,
) -> float | None:
    admissible = step_admissible(steps[t])
    forced = ctx.node_env.build_forced_step(steps[t], syn, action, branch_id, admissible)
    seed = make_seed(trunk_out, t, action, syn.completion_text, trunk_id, branch_id)
    ex = branch_example(example, trunk_out, seed)
    bout = await ctx.episode(ex, branch_id)
    # A branch whose forced action ended the game returns an EMPTY continuation
    # (no completion); that is a valid record (cont_len 0), so unlike the trunk
    # only an error disqualifies it. A replay-invariant failure is a skip, not a drop.
    if bout is None or bout.get("error") is not None:
        if bout is not None and "replay invariant" in str(bout.get("error") or ""):
            ctx.logger.warning(f"{branch_id}: replay invariant failed; skipped")
            await ctx.bump(lambda s: (setattr(s, "replay_failures", s.replay_failures + 1),
                                      CollectStats._bump(s.branches_skipped, "replay")))
            return None
        await ctx.bump(lambda s: setattr(s, "branches_dropped", s.branches_dropped + 1))
        return None
    spliced = splice_output(bout, steps[:t], forced)
    samples = interleave_rollout(spliced)
    if not samples:
        await ctx.bump(lambda s: setattr(s, "branches_dropped", s.branches_dropped + 1))
        return None
    g = float(bout["reward"])
    rec = branch_record(
        samples,
        bout,
        game_id=game_id,
        trunk_id=trunk_id,
        branch_id=branch_id,
        root_turn=t,
        action_source=source,
        block_source=syn.source,
        p_accept=syn.p_accept,
        allocation_stage=stage,
        holdout=holdout,
    )
    await ctx.write(rec)
    vc = verb_class(action)

    def _upd(s: CollectStats) -> None:
        s.branches_run += 1
        s.branch_positives += int(g > 0.0)
        s.generations_est += len(bout.get("trajectory") or [])
        CollectStats._bump(s.root_labels_by_verb_class, vc)

    await ctx.bump(_upd)
    return g


async def _process_root(
    ctx: _Ctx,
    *,
    example: dict,
    trunk_out: dict,
    G: float,
    t: int,
    k_fan: int,
    stage: int | None,
    exclude: set[str],
    node_cache: dict,
    game_id: str,
    trunk_id: str,
    holdout: bool,
    rng: random.Random,
    budget: list[int],
    pivotal: bool | None = None,
) -> list[float | None]:
    """Score (or reuse) the node at t, pick k_fan actions, synthesize, branch."""
    cfg = ctx.config
    steps = trunk_out["trajectory"]
    step = steps[t]
    admissible = step_admissible(step)
    a_taken = step_executed_action(step)
    timing = {"score_s": 0.0, "elicit_s": 0.0}

    if t not in node_cache:
        t0 = time.monotonic()
        node_cache[t] = await _score(ctx, step, admissible)
        timing["score_s"] = time.monotonic() - t0
        masses = [float(m) for m in getattr(node_cache[t], "masses", [])]

        def _upd_root(s: CollectStats) -> None:
            s.roots += 1
            s.generations_est += cfg.node_blocks + 1
            if masses:
                s.admissible_mass_at_roots_sum += sum(masses) / len(masses)
                s.admissible_mass_at_roots_n += 1

        await ctx.bump(_upd_root)
    node = node_cache[t]

    # Stage-1 probe branches use the policy's own first choice (D-9); the tail
    # is reached only when no own candidate survives the filters.
    n_unif = 0 if stage == 1 else min(cfg.n_uniform, k_fan + len(exclude))
    picks = _pick_actions(node.own, admissible, a_taken, k_fan + len(exclude), n_unif, rng)
    picks = [(a, src) for a, src in picks if a not in exclude][:k_fan]

    chosen_rows: list[dict] = []
    tasks = []
    for k, (a, src) in enumerate(picks):
        if budget[0] >= cfg.max_branches_per_game:
            await ctx.bump(lambda s: CollectStats._bump(s.branches_skipped, "budget"))
            chosen_rows.append(
                {"action": a, "source": src, "block_source": "none", "block_idx": None, "reasoning": None,
                 "p_accept": None, "attempts": 0, "reject_reasons": ["budget"], "branch_id": None, "g": None}
            )
            continue
        await ctx.bump(lambda s, src=src: CollectStats._bump(s.attempts_by_action_source, src))
        t1 = time.monotonic()
        client = await ctx.get_client()
        syn = await ctx.node_env.synthesize_block(
            client, ctx.model_name, step["prompt"], admissible, a, node, ctx.block_args
        )
        timing["elicit_s"] += time.monotonic() - t1
        for r in getattr(syn, "reject_reasons", []) or []:
            await ctx.bump(lambda s, r=r: CollectStats._bump(s.reject_reasons, r))
        # Phase 1 contract: failure is a Synthesis with source == "none" (None tolerated).
        if syn is None or getattr(syn, "source", "none") == "none":
            await ctx.bump(lambda s: CollectStats._bump(s.branches_skipped, "no_congruent_block"))
            ctx.logger.info(f"{trunk_id} t{t}: no congruent block for {a!r} ({src}); skipped")
            chosen_rows.append(
                {"action": a, "source": src, "block_source": "none", "block_idx": None, "reasoning": None,
                 "p_accept": None, "attempts": getattr(syn, "attempts", cfg.n_elicit) if syn else cfg.n_elicit,
                 "reject_reasons": list(getattr(syn, "reject_reasons", []) or []) if syn else ["exhausted"],
                 "branch_id": None, "g": None}
            )
            continue

        def _upd_acc(s: CollectStats, src=src, syn=syn) -> None:
            CollectStats._bump(s.accept_by_action_source, src)
            CollectStats._bump(s.accept_by_block_source, syn.source)
            s.generations_est += int(getattr(syn, "attempts", 0) or 0)

        await ctx.bump(_upd_acc)
        budget[0] += 1
        branch_id = f"{trunk_id}-t{t}-b{k}"
        row = {
            "action": a, "source": src, "block_source": syn.source,
            "block_idx": getattr(syn, "block_idx", None), "reasoning": syn.reasoning,
            "p_accept": syn.p_accept, "attempts": getattr(syn, "attempts", 0),
            "reject_reasons": list(getattr(syn, "reject_reasons", []) or []),
            "branch_id": branch_id, "g": None,
        }
        chosen_rows.append(row)
        tasks.append(
            (
                row,
                _run_branch(
                    ctx, example=example, trunk_out=trunk_out, steps=steps, t=t, action=a, source=src,
                    syn=syn, game_id=game_id, trunk_id=trunk_id, branch_id=branch_id, stage=stage,
                    holdout=holdout,
                ),
            )
        )

    results = await asyncio.gather(*[c for _, c in tasks]) if tasks else []
    for (row, _), g in zip(tasks, results):
        row["g"] = g

    contrast = any(g is not None and g != G for g in results)
    if contrast:
        await ctx.bump(lambda s: setattr(s, "roots_with_contrast", s.roots_with_contrast + 1))

    payload = node_json(
        game_id=game_id, trunk_id=trunk_id, turn=t, admissible=admissible, a_taken=a_taken,
        trunk_return=G, node=node, chosen=chosen_rows, allocation=cfg.allocation, stage=stage,
        pivotal=pivotal, replay_ok=all(g is not None for g in results) if results else True,
        timing=timing,
    )
    write_node_json(ctx.nodes_dir, game_id, trunk_id, t, payload)
    return list(results)


async def collect_game(ctx: _Ctx, idx: int, example: dict) -> None:
    cfg = ctx.config
    game_file = str((example.get("info") or {}).get("game_file", f"game{idx}"))
    game_id = game_id_from_file(game_file)
    marker = ctx.done_dir / f"{game_id}.json"
    if cfg.resume and marker.exists():
        await ctx.bump(lambda s: setattr(s, "games_skipped_resume", s.games_skipped_resume + 1))
        return
    holdout = idx >= ctx.n_games - cfg.holdout_games
    trunk_id = trunk_id_for(game_id)
    rng = random.Random(cfg.seed * 1_000_003 + idx)

    # 1. trunk
    out = await ctx.episode(example, trunk_id)
    if out is None or not keep_output(out):
        await ctx.bump(lambda s: setattr(s, "trunks_dropped", s.trunks_dropped + 1))
        return
    samples = interleave_rollout(out)
    if not samples:
        await ctx.bump(lambda s: setattr(s, "trunks_dropped", s.trunks_dropped + 1))
        return
    G = float(out["reward"])
    await ctx.write(trunk_record(samples, out, game_id=game_id, trunk_id=trunk_id, holdout=holdout))
    steps = out["trajectory"]
    T = len(steps)

    def _upd_trunk(s: CollectStats) -> None:
        s.trunks += 1
        s.trunk_positives += int(G > 0.0)
        s.generations_est += T

    await ctx.bump(_upd_trunk)

    node_cache: dict = {}
    budget = [0]
    common = dict(
        example=example, trunk_out=out, G=G, node_cache=node_cache, game_id=game_id,
        trunk_id=trunk_id, holdout=holdout, rng=rng, budget=budget,
    )

    # 2./3. roots and branches
    if cfg.allocation == "stratified":
        for t in stratified_roots(T, cfg.roots_per_trunk, cfg.early_tilt):
            await _process_root(ctx, t=t, k_fan=cfg.branches_per_root, stage=None, exclude=set(), **common)
    else:
        cands = stratified_roots(T, cfg.stage1_roots, cfg.early_tilt)
        stage1: dict[int, float | None] = {}
        probe_action: dict[int, str] = {}
        for t in cands:
            res = await _process_root(ctx, t=t, k_fan=1, stage=1, exclude=set(), **common)
            stage1[t] = res[0] if res else None
            # the probe action is the first chosen row of that node
            node_file = ctx.nodes_dir / game_id / trunk_id / f"t{t}.json"
            try:
                rows = json.loads(node_file.read_text(encoding="utf-8"))["chosen"]
                probe_action[t] = rows[0]["action"] if rows else ""
            except Exception:  # noqa: BLE001
                probe_action[t] = ""
        await ctx.bump(lambda s: setattr(s, "stage1_roots", s.stage1_roots + len(cands)))
        piv = pivotal_roots(stage1, G, cfg.stage2_roots)
        for t in piv:
            await _process_root(
                ctx, t=t, k_fan=max(0, cfg.branches_per_root - 1), stage=2,
                exclude={probe_action.get(t, "")} - {""}, pivotal=True, **common,
            )
        await ctx.bump(lambda s: setattr(s, "stage2_roots", s.stage2_roots + len(piv)))

    # 4. done marker
    ctx.done_dir.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps({"game_id": game_id, "trunk_id": trunk_id, "T": T, "G": G, "branches": budget[0], "holdout": holdout}),
        encoding="utf-8",
    )
    await ctx.bump(lambda s: setattr(s, "games_done", s.games_done + 1))
    ctx.logger.info(f"game {idx + 1}/{ctx.n_games} {game_id}: T={T} G={G:.0f} branches={budget[0]}")


async def run_branch_collect(ctx: _Ctx, inputs: list[dict]) -> CollectStats:
    """Drive every game with a bounded number of games in flight (episodes are
    bounded separately by ctx.sem)."""
    games_sem = asyncio.Semaphore(max(1, ctx.config.max_concurrent // 4))

    async def _one(i: int, ex: dict) -> None:
        async with games_sem:
            try:
                await collect_game(ctx, i, ex)
            except Exception as e:  # noqa: BLE001 -- one bad game must not end the collect
                ctx.logger.warning(f"game {i} failed: {e!r}")

    await asyncio.gather(*[_one(i, ex) for i, ex in enumerate(inputs)])
    return ctx.stats


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def _branch_env_args(config: BranchCollectConfig) -> dict:
    args = dict(config.env.args)
    args.update(
        branch_mode=True,
        tau=config.tau,
        n_elicit=config.n_elicit,
        elicit_prompt=config.elicit_prompt,
        leak_patterns=list(config.leak_patterns),
    )
    args.setdefault("num_reasoning_blocks", 1)
    return args


async def branch_collect(config: BranchCollectConfig) -> None:
    logger = setup_logger(config.log.level)
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env_args = _branch_env_args(config)
    logger.info(
        f"Branch collect: env={config.env.id} split={env_args.get('split')} allocation={config.allocation} "
        f"B={config.roots_per_trunk} K={config.branches_per_root} n_unif={config.n_uniform} m={config.node_blocks} "
        f"tau={config.tau} n_elicit={config.n_elicit} -> {out_dir}"
    )

    env = vf.load_environment(config.env.id, **env_args)
    node_env = vf.load_environment(config.env.id, **env_args)
    env_name = config.env.name or config.env.id

    address, env_server_process = spawn_env_server(
        env_id=config.env.id,
        env_args=env_args,
        extra_env_kwargs=config.env.extra_env_kwargs,
        num_workers=config.num_workers,
        log_level="INFO",
        log_dir=str(out_dir),
    )
    logger.info(f"Env-server worker logs -> {out_dir}/env_worker_<id>.log")
    try:
        env.env_client = setup_env_client(address=address, name=env_name)
        env_group = vf.EnvGroup(envs=[env], env_names=[env_name], map_kwargs=dict(writer_batch_size=1))
        dataset = env_group.get_dataset(seed=config.seed)
        n_games = config.games if config.games is not None else len(dataset)
        inputs = env._get_eval_inputs(n_games, 1)
        logger.info(f"{len(inputs)} games | max_concurrent={config.max_concurrent} | holdout={config.holdout_games}")

        inference_pool = await setup_inference_pool(
            config.client, model_name=config.model_name, client_type="openai_chat_completions_token"
        )
        await inference_pool.wait_for_ready(config.model_name)

        sampling_args = get_eval_sampling_args(config.sampling)
        stats_path = out_dir / "collect_stats.json"
        stats = CollectStats.load(stats_path) if (config.resume and stats_path.exists()) else CollectStats()
        records_path = out_dir / "records.msgpack"
        with RecordWriter(records_path, append=config.resume and records_path.exists()) as writer:
            ctx = _Ctx(
                config=config, env=env, node_env=node_env, get_client=inference_pool.get_next_client,
                model_name=config.model_name, sampling_args=sampling_args, block_args=sampling_args,
                greedy_args=greedy_sampling_args(sampling_args), writer=writer, out_dir=out_dir,
                logger=logger, stats=stats, n_games=len(inputs), sem=asyncio.Semaphore(config.max_concurrent),
            )
            stats = await run_branch_collect(ctx, inputs)
        stats.save(stats_path)
        logger.success(
            f"branch-collect done: games={stats.games_done} trunks={stats.trunks} roots={stats.roots} "
            f"contrast_roots={stats.roots_with_contrast} branches={stats.branches_run} "
            f"dropped={stats.branches_dropped} skipped={stats.branches_skipped} -> {records_path}"
        )
        await inference_pool.stop()
    finally:
        env_server_process.terminate()
        env_server_process.join(timeout=15)
        if env_server_process.is_alive():
            env_server_process.kill()
            env_server_process.join(timeout=5)


def main() -> None:
    asyncio.run(branch_collect(cli(BranchCollectConfig)))


if __name__ == "__main__":
    main()
