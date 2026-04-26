import bisect
import hashlib
import itertools
import json
import random
from collections import defaultdict, deque
from functools import partial
from pathlib import Path
from typing import cast

import verifiers as vf
from datasets import Dataset
from verifiers.utils.save_utils import make_serializable

from prime_rl.configs.orchestrator import BufferConfig
from prime_rl.utils.logger import get_logger
from prime_rl.utils.utils import format_num, mean, mean_normalize


class Buffer:
    """A buffer for storing rollouts and metadata."""

    POOLS = ["easy", "normal", "hard"]

    def __init__(
        self,
        dataset: Dataset,
        env_names: list[str],
        buffer_config: BufferConfig,
    ):
        self.dataset = dataset
        self.env_names = env_names
        self.config = buffer_config
        self.logger = get_logger()

        if self.config.seed is not None:
            random.seed(self.config.seed)

        # Basic assertions
        assert "example_id" in self.dataset.column_names, "The dataset must contain a `example_id` column."
        assert "prompt" in self.dataset.column_names, "The dataset must contain a `prompt` column."
        assert "task" in self.dataset.column_names, "The dataset must contain a `task` column."
        assert len(self.dataset) > 0, "The dataset must contain at least one example."
        assert isinstance(self.dataset["example_id"][0], int), "The `example_id` column must be of type int."
        assert len(set(self.dataset["example_id"])) == len(self.dataset), "The `example_id` column must be unique."
        assert set(self.dataset["task"]) == set(self.env_names), "The `task` column must contain all environment names."

        # Initialize example buffer (env_name -> (example_id -> example))
        self.example_buffer: dict[str, dict[int, dict]] = defaultdict(dict)
        for example in map(partial(cast, dict), self.dataset):
            self.example_buffer[example["task"]][example["example_id"]] = example
        assert len(self.example_buffer) == len(self.env_names)
        self.logger.debug(
            f"Initialized buffer with {format_num(len(self.dataset), precision=0)} example(s) in {len(self.env_names)} environment(s)"
        )

        if self.config.env_ratios is not None:
            # Convert ratios to probabilities
            env_ratio = mean_normalize(self.config.env_ratios)
            self.env_probs = {env_name: ratio for env_name, ratio in zip(self.env_names, env_ratio)}
            self.logger.debug(
                f"Sampling buffer according to provided environment ratios ({', '.join(f'{k}={v:.2f}' for k, v in self.env_probs.items())})"
            )
        else:
            # Count examples per environment to sample according to natural env distribution
            env_counts = [len(self.example_buffer[env_name]) for env_name in self.env_names]
            env_ratio = mean_normalize(env_counts)
            self.env_probs = {env_name: ratio for env_name, ratio in zip(self.env_names, env_ratio)}
            self.logger.debug(
                f"Sampling buffer according to natural environment distribution ({', '.join(f'{k}={v:.2f}' for k, v in self.env_probs.items())})"
            )

        # Initialize buffers for easy/ hard examples
        self.easy_examples: list[dict] = []
        self.hard_examples: list[dict] = []

        # Initialize rollout buffer (flat list of rollouts)
        self.rollout_buffer: list[vf.RolloutOutput] = []

        self.reset_step_metrics()

        # Curriculum state ----------------------------------------------------
        # Stage index. Always present (0 if curriculum disabled). Updated by
        # maybe_advance_stage() based on cumulative step boundaries.
        self.stage: int = 0

        # True iff curriculum is configured AND we have not yet advanced past
        # the last scheduled stage. Set False on post-curriculum handoff.
        self.curriculum_active: bool = bool(self.config.curriculum)

        # Cache of (examples_list, weights) per env, valid for current stage only.
        # Invalidated on stage advance.
        self._curriculum_weights_cache: dict[str, tuple[list[dict], list[float]]] = {}

        # Rolling solve-rate window for adaptive-trigger logging.
        self._stage_solve_rate_window: deque = deque(maxlen=self.config.adaptive_window_size)
        self._adaptive_already_logged_for_stage: bool = False

        if self.curriculum_active:
            self.logger.info(
                f"Curriculum enabled: {len(self.config.curriculum)} stages with "
                f"durations {self.config.curriculum} steps "
                f"(adaptive logging threshold={self.config.adaptive_advance_threshold}, "
                f"window={self.config.adaptive_window_size})"
            )
            # Safety: warn loudly if config requested curriculum but no rows have weights.
            # (Without curriculum_weights on rows, sampling silently falls back to uniform.)
            has_weights = any(
                "curriculum_weights" in ex
                for env_examples in self.example_buffer.values()
                for ex in env_examples.values()
            )
            if not has_weights:
                self.logger.warning(
                    "config.curriculum is set but NO rows in the dataset have a "
                    "`curriculum_weights` field. Buffer will silently fall back to "
                    "uniform sampling. Did you forget to pass `curriculum=...` to "
                    "load_environment for the relevant env(s)?"
                )

    # ------------------------------------------------------------------------
    # Curriculum stage management
    # ------------------------------------------------------------------------

    def maybe_advance_stage(self, step: int) -> bool:
        """Advance curriculum stage if `step` has crossed a stage boundary.

        Called by the orchestrator at the top of each step loop iteration.
        Computes the new stage from cumulative step boundaries derived from
        config.curriculum. On a transition, invalidates the per-env weight
        cache and resets adaptive trigger state for the new stage.

        When the new stage exceeds the configured number of stages (post-
        curriculum), curriculum_active is set False and online_difficulty_filtering
        is enabled for the steady-state handoff.

        Returns True if a transition just fired (for logging / metrics).
        """
        if not self.config.curriculum:
            return False

        cumulative_boundaries = list(itertools.accumulate(self.config.curriculum))
        new_stage = bisect.bisect_right(cumulative_boundaries, step)
        if new_stage == self.stage:
            return False

        old_stage = self.stage
        self.stage = new_stage

        # Invalidate caches and reset adaptive state for the new stage
        self._curriculum_weights_cache.clear()
        self._stage_solve_rate_window.clear()
        self._adaptive_already_logged_for_stage = False

        if new_stage >= len(self.config.curriculum):
            self.curriculum_active = False
            self.config.online_difficulty_filtering = True
            self.logger.info(
                f"Curriculum complete at step {step} (stage {old_stage} -> "
                f"post-curriculum). Enabled online_difficulty_filtering for handoff."
            )
        else:
            self.logger.info(
                f"Curriculum advanced: stage {old_stage} -> {new_stage} at step {step}"
            )
        return True

    def _get_curriculum_weighted_examples(
        self, env_name: str
    ) -> tuple[list[dict], list[float]]:
        """Return (examples_list, weights) for an env at the current stage.

        Cached per stage (cache cleared on stage advance). Falls back to
        uniform weights if every row's stage weight resolves to zero.
        """
        if env_name in self._curriculum_weights_cache:
            return self._curriculum_weights_cache[env_name]

        examples = list(self.example_buffer[env_name].values())
        weights: list[float] = []
        for ex in examples:
            ex_weights = ex.get("curriculum_weights")
            if ex_weights is None or self.stage >= len(ex_weights):
                weights.append(1.0)
            else:
                w = float(ex_weights[self.stage])
                weights.append(max(0.0, w))

        if sum(weights) <= 0.0:
            self.logger.warning(
                f"All curriculum weights at stage {self.stage} for env "
                f"'{env_name}' are zero. Falling back to uniform sampling."
            )
            weights = [1.0] * len(examples)

        self._curriculum_weights_cache[env_name] = (examples, weights)
        return examples, weights

    def _maybe_log_adaptive_trigger(self) -> None:
        """Log a 'would-have-advanced' event once per stage if the rolling
        solve rate exceeds the configured threshold. Diagnostic only — does
        not change sampling behavior.
        """
        if self._adaptive_already_logged_for_stage:
            return
        # Require the window to be at least half-full before evaluating, to
        # avoid spurious triggers from a few lucky early rollout groups.
        max_size = self._stage_solve_rate_window.maxlen
        if max_size is None or len(self._stage_solve_rate_window) < max(1, max_size // 2):
            return
        rolling_rate = sum(self._stage_solve_rate_window) / len(self._stage_solve_rate_window)
        if rolling_rate >= self.config.adaptive_advance_threshold:
            self.logger.info(
                f"[curriculum-adaptive] Stage {self.stage} rolling solve rate "
                f"{rolling_rate:.3f} >= threshold {self.config.adaptive_advance_threshold} "
                f"(window={len(self._stage_solve_rate_window)}). "
                f"Would-have-advanced if adaptive trigger were enabled."
            )
            self._adaptive_already_logged_for_stage = True

    def get_example_hash(self, example: dict) -> str:
        """Returns a hash of the example based on hash keys."""
        hash_keys = [key for key in self.config.hash_keys if key in example]
        assert hash_keys, "No hashable keys found in example."
        return hashlib.sha256(json.dumps([example[key] for key in hash_keys]).encode()).hexdigest()

    def save(self, path: Path) -> None:
        """Saves pool assignments and rollout buffer."""
        path.mkdir(parents=True, exist_ok=True)

        def write_jsonl(lst: list, path: Path) -> None:
            with open(path, "w") as f:
                for item in lst:
                    f.write(json.dumps(item, default=make_serializable) + "\n")

        write_jsonl(self.easy_examples, path / "easy_examples.jsonl")
        write_jsonl(self.hard_examples, path / "hard_examples.jsonl")
        write_jsonl(self.rollout_buffer, path / "rollout_buffer.jsonl")

    def load(self, path: Path) -> None:
        """Loads pool assignments and rollouts."""

        def read_jsonl(path: Path) -> list[dict]:
            with open(path, "r") as f:
                return [json.loads(line) for line in f]

        saved_easy_examples = read_jsonl(path / "easy_examples.jsonl")
        saved_hard_examples = read_jsonl(path / "hard_examples.jsonl")
        saved_rollout_buffer = cast(list[vf.RolloutOutput], read_jsonl(path / "rollout_buffer.jsonl"))

        if any(saved_easy_examples) or any(saved_hard_examples) or any(saved_rollout_buffer):
            # Build hash lookup for example buffer (env -> (example_hash -> example_id))
            example_hash_lookup = defaultdict(dict)
            all_hashes = set()
            for env in self.example_buffer:
                for example_id, example in self.example_buffer[env].items():
                    example_hash = self.get_example_hash(example)
                    if example_hash in all_hashes:
                        self.logger.warning(
                            f"Duplicate example hash found based on hash_keys={self.config.hash_keys}. Overwriting with latest example. This may cause unexpected behavior when resuming the buffer."
                        )
                    example_hash_lookup[env][example_hash] = example_id
                    all_hashes.add(example_hash)

            def move_saved_pool(saved_examples: list[dict], target_pool: list[dict]) -> int:
                """Moves saved examples to the target pool from example buffer based on hash lookup."""
                num_moved = 0
                for example in saved_examples:
                    example_hash = self.get_example_hash(example)
                    for env in example_hash_lookup:
                        if example_hash in example_hash_lookup[env]:
                            example_id = example_hash_lookup[env][example_hash]
                            example = self.example_buffer[env].pop(example_id, None)
                            if example is not None:
                                target_pool.append(example)
                                num_moved += 1
                                break
                return num_moved

            if any(saved_easy_examples):
                num_moved = move_saved_pool(saved_easy_examples, self.easy_examples)
                self.logger.debug(
                    f"Loaded {num_moved}/{len(saved_easy_examples)} example(s) to easy pool from checkpoint."
                )
                if num_moved != len(saved_easy_examples):
                    num_not_moved = len(saved_easy_examples) - num_moved
                    self.logger.warning(
                        f"Could not move {num_not_moved} example(s) from checkpoint to easy pool. This usually means you resumed with an env mix that does not contain all previous examples."
                    )

            if any(saved_hard_examples):
                num_moved = move_saved_pool(saved_hard_examples, self.hard_examples)
                self.logger.debug(
                    f"Moved {num_moved}/{len(saved_hard_examples)} example(s) to hard pool from checkpoint."
                )
                if num_moved != len(saved_hard_examples):
                    num_not_moved = len(saved_hard_examples) - num_moved
                    self.logger.warning(
                        f"Could not move {num_not_moved} example(s) from checkpoint to hard pool. This usually means you resumed with an env mix that does not contain all previous examples."
                    )

            if any(saved_rollout_buffer):
                # Extend rollout buffer, but only include rollouts for which the example still exists in the example buffer
                valid_saved_rollouts = [
                    rollout for rollout in saved_rollout_buffer if rollout["task"] in self.env_names
                ]
                self.rollout_buffer.extend(valid_saved_rollouts)
                self.logger.debug(f"Loaded {len(valid_saved_rollouts)} rollout(s) from checkpoint.")

            # Load rollouts, filtering out removed environments and problems
            def convert_examples_to_normal(examples: list[dict], fraction: float) -> int:
                """Moves a fraction of examples from the given pool back to normal."""
                if fraction <= 0.0 or not examples:
                    return 0
                num_moved = round(len(examples) * fraction)
                if num_moved <= 0:
                    return 0
                for _ in range(num_moved):
                    example = random.choice(examples)
                    env_name = example["task"]
                    example_id = example["example_id"]
                    examples.remove(example)
                    self.example_buffer[env_name][example_id] = example
                return num_moved

            num_easy_examples = len(self.easy_examples)
            num_moved = convert_examples_to_normal(self.easy_examples, self.config.easy_fraction)
            self.logger.debug(f"Converted {num_moved}/{num_easy_examples} example(s) back to normal from easy pool.")
            num_hard_examples = len(self.hard_examples)
            num_moved = convert_examples_to_normal(self.hard_examples, self.config.hard_fraction)
            self.logger.debug(f"Converted {num_moved}/{num_hard_examples} example(s) back to normal from hard pool.")
        else:
            self.logger.debug("No easy/ hard examples or rollouts found in checkpoint")

    def sample_examples(self, n: int) -> list[dict]:
        """Samples n examples from the buffer, respecting env ratios.

        When `curriculum_active` is True, the within-env sampling step uses
        per-row weights from `row["curriculum_weights"][self.stage]` (cached
        per stage). Otherwise within-env sampling is uniform random (legacy
        behavior). Env-level sampling (via env_probs) is unchanged in either
        mode.
        """

        non_empty_envs = [env for env, examples in self.example_buffer.items() if examples]

        if not non_empty_envs:
            raise ValueError("No environments left with examples.")

        non_empty_env_probs = [self.env_probs[env] for env in non_empty_envs]
        sampled_examples = []
        for sampled_env in random.choices(non_empty_envs, weights=non_empty_env_probs, k=n):
            if self.curriculum_active:
                examples, weights = self._get_curriculum_weighted_examples(sampled_env)
                sampled_example = random.choices(examples, weights=weights, k=1)[0]
            else:
                sampled_example = random.choice(list(self.example_buffer[sampled_env].values()))
            sampled_examples.append(sampled_example)

        return sampled_examples

    def update(self, rollouts: list[vf.RolloutOutput]):
        """Updates the buffer state with completed rollouts."""

        rollouts_by_example = defaultdict(list)
        for rollout in rollouts:
            rollouts_by_example[rollout["example_id"]].append(rollout)

        for example_id, example_rollouts in rollouts_by_example.items():
            avg_reward = mean([r["reward"] for r in example_rollouts])
            env_name = example_rollouts[0]["task"]

            # Curriculum: track rolling solve rate at current stage and log
            # adaptive-trigger events. Diagnostic only.
            if self.curriculum_active:
                self._stage_solve_rate_window.append(float(avg_reward))
                self._maybe_log_adaptive_trigger()

            if self.config.easy_threshold is not None and avg_reward >= self.config.easy_threshold:
                pool = "easy"
            elif self.config.hard_threshold is not None and avg_reward <= self.config.hard_threshold:
                pool = "hard"
            else:
                pool = "normal"

            if pool != "normal" and example_id in self.example_buffer[env_name]:
                example = self.example_buffer[env_name].pop(example_id)
                target_pool = self.easy_examples if pool == "easy" else self.hard_examples
                target_pool.append(example)

            self.num_examples_per_step[env_name][pool] += 1
            if self.config.online_difficulty_filtering:
                if avg_reward == 0.0:
                    self.num_rollouts_per_step[env_name]["hard"] += len(example_rollouts)
                    continue
                elif avg_reward == 1.0:
                    self.num_rollouts_per_step[env_name]["easy"] += len(example_rollouts)
                    continue

            self.num_rollouts_per_step[env_name]["normal"] += len(example_rollouts)
            self.rollout_buffer.extend(example_rollouts)

    def sample_rollouts(self, n: int) -> list[vf.RolloutOutput]:
        """Samples the latest n rollouts from the buffer."""
        n = min(n, len(self.rollout_buffer))
        sampled_rollouts = self.rollout_buffer[-n:]
        self.rollout_buffer = self.rollout_buffer[:-n]
        return sampled_rollouts

    def reset_step_metrics(self) -> None:
        """Reset per-step metrics (called after get_metrics)."""
        zero_per_pool = lambda: {p: 0 for p in self.POOLS}
        # num examples per env per step per pool (env_name -> (pool -> num_examples))
        self.num_examples_per_step = {env: zero_per_pool() for env in self.env_names}
        # num rollouts per env per step per pool (env_name -> (pool -> num_rollouts))
        self.num_rollouts_per_step = {env: zero_per_pool() for env in self.env_names}

    def get_metrics(self) -> dict[str, float]:
        """Returns the buffer metrics for the current step."""

        metrics = {}
        easy_examples_per_env = defaultdict(int)
        hard_examples_per_env = defaultdict(int)
        for example in self.easy_examples:
            easy_examples_per_env[example["task"]] += 1
        for example in self.hard_examples:
            hard_examples_per_env[example["task"]] += 1

        # sum over envs (e.g. log globally)
        num_examples_per_step_per_pool = {
            pool: sum(self.num_examples_per_step[env][pool] for env in self.env_names) for pool in self.POOLS
        }
        num_rollouts_per_step_per_pool = {
            pool: sum(self.num_rollouts_per_step[env][pool] for env in self.env_names) for pool in self.POOLS
        }
        num_examples_per_step = sum(num_examples_per_step_per_pool.values())
        num_rollouts_per_step = sum(num_rollouts_per_step_per_pool.values())

        for pool in ["easy", "hard"]:
            if num_examples_per_step:
                metrics[f"evicted_examples/{pool}"] = num_examples_per_step_per_pool[pool] / num_examples_per_step
            if num_rollouts_per_step:
                metrics[f"filtered_rollouts/{pool}"] = num_rollouts_per_step_per_pool[pool] / num_rollouts_per_step

        total_normal = sum(len(self.example_buffer[env]) for env in self.env_names)
        pool_counts = [len(self.easy_examples), total_normal, len(self.hard_examples)]
        pool_ratios = mean_normalize(pool_counts)
        for pool, pool_ratio in zip(self.POOLS, pool_ratios):
            metrics[f"pool/{pool}"] = pool_ratio

        for env in self.env_names:
            env_num_examples_per_step_per_pool = self.num_examples_per_step[env]
            env_num_rollouts_per_step_per_pool = self.num_rollouts_per_step[env]
            env_num_examples_per_step = sum(env_num_examples_per_step_per_pool.values())
            env_num_rollouts_per_step = sum(env_num_rollouts_per_step_per_pool.values())

            for pool in ["easy", "hard"]:
                if env_num_examples_per_step:
                    metrics[f"evicted_examples/{env}/{pool}"] = (
                        env_num_examples_per_step_per_pool[pool] / env_num_examples_per_step
                    )
                if env_num_rollouts_per_step:
                    metrics[f"filtered_rollouts/{env}/{pool}"] = (
                        env_num_rollouts_per_step_per_pool[pool] / env_num_rollouts_per_step
                    )

            env_pool_counts = [
                easy_examples_per_env[env],
                len(self.example_buffer[env]),
                hard_examples_per_env[env],
            ]
            env_pool_ratios = mean_normalize(env_pool_counts)
            for pool, pool_ratio in zip(self.POOLS, env_pool_ratios):
                metrics[f"pool/{env}/{pool}"] = pool_ratio

        self.reset_step_metrics()

        return metrics
