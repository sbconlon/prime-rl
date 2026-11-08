"""Training-time probe: the orchestrator hook (training-probe plan, Phase 3).

A third sibling of the eval / val blocks. Every ``interval`` policy checkpoint
steps (gated with eval's ``compute_eval_ckpt_step``, D-11) the orchestrator:

1. rolls out the probe set once each on the live policy, **in-process** (no env
   server, D-7), with the probe env in capture mode: greedy executed response
   plus m sampled reasoning blocks per turn and ``extras["probe"]`` per step
   (Phase 1);
2. interleaves each rollout exactly as the training path does;
3. makes one read-only ``compute_with_diagnostics`` call per rollout to the
   Advantage Server (Phase 2). Nothing is sent to either trainer;
4. joins turn k of the trajectory with diagnostics record k, computes the
   per-action advantage ``adv[a] = log pi_RM(a) - log pi_hat(a)`` and writes one
   JSON per rollout under ``<output_dir>/<output_subdir>/ckpt_<step>/`` plus a
   line in ``index.jsonl`` (probe-schema.md sections 3 and 4, D-9);
5. logs ``eval/<env_name>/...`` (the probe replaces the online eval for
   ALFWorld, D-2) and ``probe/...`` metrics.

``assemble_probe_rollout`` is a pure function so the join is testable without
a server; ``ProbeRunner`` owns the I/O. The training block, scheduler, buffer
and transports are untouched: probe rollouts never reach them.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Awaitable, Callable

import verifiers as vf

from prime_rl.configs.orchestrator import ProbeConfig
from prime_rl.orchestrator.advantage_server_client import (
    AdvantageServerClientProtocol,
    DiagnosticsResult,
)
from prime_rl.orchestrator.eval_utils import get_eval_sampling_args
from prime_rl.orchestrator.trajectories import interleave_rollout, pretokenize_rollout_trajectory
from prime_rl.orchestrator.vf_utils import generate, get_completion_len
from prime_rl.transport.types import DecisionPoint, TrainingSample
from prime_rl.utils.logger import get_logger

PROBE_SCHEMA_VERSION = 1
# action_advantage's probability floor (advantage_server/action_advantage.py); the
# per-action vector reproduces the server's A(a*) exactly at the executed index.
ADV_FLOOR = 1e-6
ADV_STAR_TOL = 1e-6
# State keys the record needs beyond RolloutOutput's required fields; vf_utils.generate
# copies them from the env state into the output.
PROBE_STATE_COLUMNS = ["final_env_response", "context_evictions"]
_ORACLE_OFF = {"action": None, "status": "off", "plan": None, "steps_left": None}


class ProbeAssemblyError(ValueError):
    """The rollout, its samples and the server's diagnostics do not line up.

    The runner logs and skips the rollout rather than writing a misaligned file.
    """


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _content(message: Any) -> str:
    """Text of a chat message (dict or object). Multimodal parts are joined on
    their text."""
    if message is None:
        return ""
    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if "text" in part:
                    parts.append(str(part.get("text") or ""))
            else:
                text = getattr(part, "text", None)
                if text is not None:
                    parts.append(str(text))
        return "".join(parts)
    return str(content)


def flatten_decision_points(samples: list[TrainingSample]) -> list[tuple[int, DecisionPoint]]:
    """``(sample_idx, DecisionPoint)`` in sample order, then decision-point order:
    the order the server's diagnostics records use."""
    out: list[tuple[int, DecisionPoint]] = []
    for s_idx, sample in enumerate(samples):
        for dp in sample.decision_points or []:
            out.append((s_idx, dp))
    return out


def _turn_steps(trajectory: list) -> list[dict]:
    """Trajectory steps that produce a DecisionPoint: extras carry both
    ``admissible_actions`` and ``executed_action`` (the rule in
    ``trajectories._build_decision_point``). Error turns without them are not
    probe turns."""
    out = []
    for step in trajectory:
        extras = step.get("extras") or {}
        if extras.get("admissible_actions") is not None and extras.get("executed_action") is not None:
            out.append(step)
    return out


def probe_sampling_summary(config: ProbeConfig) -> dict[str, Any]:
    """The record's ``sampling`` block: the executed-response temperature from
    ``probe.sampling``, the m-block regime from the probe env's args."""
    args = config.env.args
    exec_t = config.sampling.temperature
    return {
        "exec_temperature": float(exec_t if exec_t is not None else 0.0),
        "block_temperature": float(args.get("probe_block_temperature", 0.8)),
        "block_min_p": float(args.get("probe_block_min_p", 0.1)),
        "m": int(args.get("num_reasoning_blocks", 1)),
    }


def assemble_probe_rollout(
    rollout: vf.RolloutOutput,
    samples: list[TrainingSample],
    diag: DiagnosticsResult,
    *,
    ckpt_step: int,
    orchestrator_step: int,
    run: str,
    sampling: dict[str, Any],
    oracle_mode: str = "off",
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Join one probe rollout with its diagnostics into the on-disk record
    (probe-schema.md section 3). Pure: no I/O.

    Turn k is the k-th trajectory step that produced a DecisionPoint; it is
    joined with the k-th flattened decision point and the k-th diagnostics
    record. Every alignment the schema promises is asserted here
    (``ProbeAssemblyError``): equal counts, ``admissible_union ==
    DecisionPoint.admissible_actions``, matching executed index, matching
    ``(sample_idx, response_start)``, vector lengths, and ``adv[a*] ==
    adv_star`` (the server's value) whenever ``pi_hat`` was set. A turn whose
    scoring failed (``pi_hat_marginal`` null) gets ``adv`` of zeros and
    ``adv_star_valid=False``; the server's placeholder ``adv_star`` (0.0) is
    recorded as is.
    """
    trajectory = rollout.get("trajectory") or []
    steps = _turn_steps(trajectory)
    flat = flatten_decision_points(samples)
    diags = list(diag.diagnostics)
    if not (len(steps) == len(flat) == len(diags)):
        raise ProbeAssemblyError(
            f"count mismatch: probe turns={len(steps)} decision points={len(flat)} "
            f"diagnostics={len(diags)} (trajectory steps={len(trajectory)})"
        )

    turns: list[dict[str, Any]] = []
    for k, (step, (s_idx, dp), d) in enumerate(zip(steps, flat, diags)):
        extras = step.get("extras") or {}
        probe = extras.get("probe")
        if not isinstance(probe, dict):
            raise ProbeAssemblyError(f"turn {k}: extras['probe'] missing (probe env not in capture mode?)")
        admissible = [str(a) for a in (probe.get("admissible_union") or [])]
        n = len(admissible)
        if n == 0 or admissible != list(dp.admissible_actions):
            raise ProbeAssemblyError(f"turn {k}: admissible_union != DecisionPoint.admissible_actions")
        e = int(probe.get("executed_idx", -1))
        if e != dp.executed_action_idx or not 0 <= e < n:
            raise ProbeAssemblyError(
                f"turn {k}: executed_idx {e} != DecisionPoint.executed_action_idx {dp.executed_action_idx}"
            )
        if int(d.sample_idx) != s_idx or int(d.response_start) != dp.response_start:
            raise ProbeAssemblyError(
                f"turn {k}: diagnostics record ({d.sample_idx}, {d.response_start}) != "
                f"decision point ({s_idx}, {dp.response_start})"
            )
        q_plus = [float(x) for x in d.q_plus]
        pi_rm = [float(x) for x in d.pi_rm]
        if len(q_plus) != n or len(pi_rm) != n:
            raise ProbeAssemblyError(f"turn {k}: |q_plus|={len(q_plus)} |pi_rm|={len(pi_rm)} != |admissible|={n}")

        pi_hat_raw = probe.get("pi_hat_marginal")
        if pi_hat_raw is not None:
            pi_hat = [float(x) for x in pi_hat_raw]
            if len(pi_hat) != n:
                raise ProbeAssemblyError(f"turn {k}: |pi_hat_marginal|={len(pi_hat)} != |admissible|={n}")
            adv = [math.log(max(pr, ADV_FLOOR)) - math.log(max(ph, ADV_FLOOR)) for pr, ph in zip(pi_rm, pi_hat)]
            if abs(adv[e] - float(d.adv_star)) > ADV_STAR_TOL:
                raise ProbeAssemblyError(
                    f"turn {k}: adv[a*]={adv[e]:.8f} != server adv_star={float(d.adv_star):.8f}"
                )
            pi_hat_star: float | None = pi_hat[e]
            adv_star_valid = True
        else:
            pi_hat = None
            pi_hat_star = None
            adv = [0.0] * n
            adv_star_valid = False

        blocks = list(probe.get("blocks") or [])
        for j, block in enumerate(blocks):
            if len(block.get("cond") or []) != n:
                raise ProbeAssemblyError(f"turn {k}: |blocks[{j}].cond| != |admissible|={n}")
        Z = list(probe.get("Z") or [])
        if len(Z) != len(blocks):
            raise ProbeAssemblyError(f"turn {k}: |Z|={len(Z)} != |blocks|={len(blocks)}")

        exec_block = probe.get("exec_block") or {}
        observation = _content(step["prompt"][-1]) if step.get("prompt") else ""
        if k + 1 < len(steps):
            nxt = steps[k + 1]
            env_response: str | None = _content(nxt["prompt"][-1]) if nxt.get("prompt") else None
        else:
            final = rollout.get("final_env_response")
            if isinstance(final, list):
                env_response = _content(final[0]) if final else None
            else:
                env_response = _content(final) if final else None

        turns.append(
            {
                "turn": k,
                "observation": observation,
                "window": dict(probe.get("window") or {"labels": [], "token_count": None, "evictions_so_far": 0}),
                "admissible": admissible,
                "executed": {
                    "text": admissible[e],
                    "idx": e,
                    "reasoning": str(exec_block.get("reasoning") or ""),
                    "completion": str(exec_block.get("completion") or ""),
                    "parsed_ok": bool(probe.get("parsed_ok", True)),
                },
                "blocks": blocks,
                "pi_hat": pi_hat,
                "pi_hat_star": pi_hat_star,
                "Z": Z,
                "raw_logps": probe.get("raw_logps"),
                # D-13: which pi_hat convention the env used for this turn (True =
                # softmax over the admissible union, False = raw sequence probabilities).
                "pi_hat_renorm": bool(probe.get("pi_hat_renorm", True)),
                "values": {
                    "v": float(d.v),
                    "v_target": float(d.v_target),
                    "q_plus": q_plus,
                    "pi_rm": pi_rm,
                    "adv": adv,
                    "adv_star": float(d.adv_star),
                    "adv_star_valid": adv_star_valid,
                    "g_k": float(d.g_k),
                    "q_plus_target_star": float(d.q_plus_target_star),
                },
                "oracle": dict(probe.get("oracle") or _ORACLE_OFF),
                "env_response": env_response,
            }
        )

    info = rollout.get("info") or {}
    reward = float(rollout.get("reward") or 0.0)
    metrics = rollout.get("metrics") or {}
    evictions = metrics.get("context_evictions")
    if evictions is None:
        evictions = rollout.get("context_evictions") or 0
    return {
        "schema_version": PROBE_SCHEMA_VERSION,
        "run": run,
        "round": {
            "ckpt_step": int(ckpt_step),
            "weight_step": int(diag.weight_step),
            "orchestrator_step": int(orchestrator_step),
            "timestamp": timestamp or _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        },
        "probe_id": str(info.get("probe_id", rollout.get("example_id"))),
        "game_id": str(info.get("game_id", "")),
        "task_type": str(info.get("task_type", "")),
        "split": str(info.get("split", "")),
        "role": str(info.get("role", "")),
        "won": reward == 1.0,
        "reward": reward,
        "num_turns": len(turns),
        "is_truncated": bool(rollout.get("is_truncated", False)),
        "stop_condition": rollout.get("stop_condition"),
        "context_evictions": int(evictions or 0),
        "oracle_mode": oracle_mode,
        "sampling": dict(sampling),
        "turns": turns,
    }


def probe_metrics(
    records: list[dict[str, Any]],
    completion_lens: list[int],
    *,
    env_name: str,
    num_skipped: int,
    probe_time: float,
    emit_eval_metrics: bool = True,
) -> dict[str, float]:
    """Round metrics (Phase 3 section 5.5). ``eval/<env_name>/...`` follows eval's
    naming so dashboards carry over; ``probe/...`` are the probe's own.

    ``probe/oracle_agreement`` is the fraction of turns with an expert oracle
    answer on which the greedy action equals it (absent when no turn has one).
    """
    m: dict[str, float] = {}
    n = len(records)
    m["probe/num_rollouts"] = float(n)
    m["probe/num_skipped"] = float(num_skipped)
    m["probe/time"] = float(probe_time)
    if n == 0:
        return m

    def _mean(xs) -> float:
        xs = list(xs)
        return float(sum(xs) / len(xs)) if xs else 0.0

    if emit_eval_metrics:
        m[f"eval/{env_name}/avg@1"] = _mean(r["reward"] for r in records)
        for key in ("split", "task_type", "role"):
            groups: dict[str, list[float]] = defaultdict(list)
            for r in records:
                groups[str(r.get(key) or "")].append(float(r["reward"]))
            for g, vals in sorted(groups.items()):
                if g:
                    m[f"eval/{env_name}/{g}/avg@1"] = _mean(vals)
        if completion_lens:
            m[f"eval/{env_name}/completion_len/mean"] = _mean(completion_lens)
        m[f"eval/{env_name}/is_truncated/mean"] = _mean(float(r["is_truncated"]) for r in records)
        m[f"eval/{env_name}/time"] = float(probe_time)

    turns = [t for r in records for t in r["turns"]]
    m["probe/num_turns_mean"] = _mean(r["num_turns"] for r in records)
    if not turns:
        return m
    statuses = Counter(str((t.get("oracle") or {}).get("status") or "off") for t in turns)
    for status, count in sorted(statuses.items()):
        m[f"probe/oracle_status/{status}"] = count / len(turns)
    expert = [t for t in turns if (t.get("oracle") or {}).get("status") == "expert"]
    if expert:
        m["probe/oracle_agreement"] = _mean(
            float(t["executed"]["text"] == (t["oracle"] or {}).get("action")) for t in expert
        )
    m["probe/q_plus_spread_mean"] = _mean(max(t["values"]["q_plus"]) - min(t["values"]["q_plus"]) for t in turns)
    m["probe/pi_rm_entropy_mean"] = _mean(
        -sum(p * math.log(p) for p in t["values"]["pi_rm"] if p > 0.0) for t in turns
    )
    # D-13: the admissible mass exp(Z_j) per scored block -- the normalizer pi_hat
    # drops when pi_hat_renorm is False; its size is the open question the probe
    # is meant to answer, so it is a round metric either way.
    masses = [
        math.exp(float(z)) for t in turns if t.get("pi_hat") is not None for z in (t.get("Z") or [])
    ]
    if masses:
        m["probe/admissible_mass_mean"] = _mean(masses)
        m["probe/admissible_mass_min"] = float(min(masses))
        m["probe/admissible_mass_frac_below_half"] = _mean(float(x < 0.5) for x in masses)
    flags = {bool(t.get("pi_hat_renorm", True)) for t in turns}
    m["probe/pi_hat_renorm"] = 1.0 if flags == {True} else 0.0 if flags == {False} else 0.5
    valid = [t["values"]["adv_star"] for t in turns if t["values"]["adv_star_valid"]]
    m["probe/pi_hat_missing_frac"] = 1.0 - len(valid) / len(turns)
    if valid:
        m["probe/adv_star_mean"] = _mean(valid)
        m["probe/adv_star_absmax"] = float(max(abs(x) for x in valid))
    return m


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


class ProbeWriter:
    """Writes ``<output_dir>/<subdir>/ckpt_<ckpt_step>/<probe_id>.json`` (pretty
    printed, UTF-8, temp file + rename) and appends one line per record to
    ``<subdir>/index.jsonl`` (section 4). Re-running a round for the same
    ``ckpt_step`` overwrites the files and appends new index lines; consumers
    keep the last line per ``(ckpt_step, probe_id)``."""

    def __init__(self, output_dir: Path, subdir: str = "probes"):
        self.output_dir = Path(output_dir)
        self.subdir = subdir
        self.root = self.output_dir / subdir

    @staticmethod
    def _safe_id(probe_id: str) -> str:
        return str(probe_id).replace("/", "_").replace("\\", "_")

    def write(self, record: dict[str, Any]) -> Path:
        ckpt = int(record["round"]["ckpt_step"])
        pid = self._safe_id(record["probe_id"])
        rel = f"{self.subdir}/ckpt_{ckpt}/{pid}.json"
        path = self.output_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{pid}.json.tmp")
        tmp.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
        line = {
            "ckpt_step": ckpt,
            "weight_step": int(record["round"]["weight_step"]),
            "orchestrator_step": int(record["round"]["orchestrator_step"]),
            "probe_id": record["probe_id"],
            "game_id": record["game_id"],
            "split": record["split"],
            "task_type": record["task_type"],
            "role": record["role"],
            "won": bool(record["won"]),
            "num_turns": int(record["num_turns"]),
            "path": rel,
        }
        with (self.root / "index.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
        return path


class ProbeRunner:
    """One probe round per call: generate, interleave, diagnostics, assemble,
    write, log. Built once at orchestrator start; ``run_round`` is awaited
    inside the loop's ``checkpoint_ready`` pause window and once more at the
    end of training."""

    def __init__(
        self,
        config: ProbeConfig,
        env: vf.Environment,
        model_name: str | Callable[[], str],
        get_client: Callable[[], Awaitable[vf.ClientConfig]],
        adv_client: AdvantageServerClientProtocol,
        output_dir: Path,
        run_name: str,
        tokenizer,
        monitor,
        processor=None,
    ):
        self.config = config
        self.env = env
        self.model_name = model_name
        self.get_client = get_client
        self.adv_client = adv_client
        self.run_name = run_name
        self.tokenizer = tokenizer
        self.processor = processor
        self.monitor = monitor
        self.env_name = config.env.resolved_name
        self.oracle_mode = str(config.env.args.get("oracle", "off"))
        self.sampling_summary = probe_sampling_summary(config)
        self.sampling_args = get_eval_sampling_args(config.sampling)
        # interleave_rollout reads sampling_args["temperature"]; the executed response is
        # greedy by default (ProbeSamplingConfig), so None falls back to 0.0 here.
        self.sampling_args.setdefault("temperature", 0.0)
        self.writer = ProbeWriter(output_dir, config.output_subdir)

    def _resolve_model_name(self) -> str:
        return self.model_name() if callable(self.model_name) else self.model_name

    async def run_round(self, ckpt_step: int, orchestrator_step: int) -> dict[str, float]:
        logger = get_logger()
        start = time.perf_counter()
        examples = self.env.get_dataset().to_list()
        logger.info(f"Probe round for checkpoint step {ckpt_step}: {len(examples)} game(s) on {self.env_name}")
        rollouts = await generate(
            env=self.env,
            model_name=self._resolve_model_name(),
            examples=examples,
            rollouts_per_example=1,
            sampling_args=self.sampling_args,
            get_client=self.get_client,
            max_retries=self.config.max_retries,
            state_columns=PROBE_STATE_COLUMNS,
            pbar_description="Generating rollouts (probe)",
        )

        records: list[dict[str, Any]] = []
        completion_lens: list[int] = []
        num_skipped = 0
        for rollout in rollouts:
            probe_id = (rollout.get("info") or {}).get("probe_id", rollout.get("example_id"))
            try:
                pretokenize_rollout_trajectory(rollout, self.tokenizer, processor=self.processor)
                samples = interleave_rollout(rollout)
                if not samples:
                    raise ProbeAssemblyError("no training samples (empty trajectory or missing tokens)")
                diag = await self.adv_client.compute_with_diagnostics(
                    samples,
                    float(rollout["reward"]),
                    not bool(rollout.get("is_truncated", False)),
                    "arm",
                )
                record = assemble_probe_rollout(
                    rollout,
                    samples,
                    diag,
                    ckpt_step=ckpt_step,
                    orchestrator_step=orchestrator_step,
                    run=self.run_name,
                    sampling=self.sampling_summary,
                    oracle_mode=self.oracle_mode,
                )
            except ProbeAssemblyError as exc:
                num_skipped += 1
                logger.warning(f"Probe rollout {probe_id!r} skipped (misaligned): {exc}")
                continue
            except Exception as exc:  # noqa: BLE001 -- one bad rollout must not end the round
                num_skipped += 1
                logger.error(f"Probe rollout {probe_id!r} skipped (error): {exc!r}")
                continue
            self.writer.write(record)
            records.append(record)
            try:
                completion_lens.append(get_completion_len(rollout))
            except Exception:  # noqa: BLE001 -- metric only
                pass

        probe_time = time.perf_counter() - start
        metrics = probe_metrics(
            records,
            completion_lens,
            env_name=self.env_name,
            num_skipped=num_skipped,
            probe_time=probe_time,
            emit_eval_metrics=self.config.emit_eval_metrics,
        )
        metrics["progress/ckpt_step"] = ckpt_step
        metrics["step"] = orchestrator_step
        self.monitor.log(metrics, step=orchestrator_step)
        avg = metrics.get(f"eval/{self.env_name}/avg@1")
        logger.success(
            f"Probe round for checkpoint step {ckpt_step} done in {probe_time:.1f}s: "
            f"{len(records)} record(s) written, {num_skipped} skipped"
            + (f", Avg@1={avg:.4f}" if avg is not None else "")
        )
        return metrics
