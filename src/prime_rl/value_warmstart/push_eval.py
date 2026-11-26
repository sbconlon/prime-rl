"""Push-direction score of a value_state.pt on a gold test set (plan
bulletproof-warm-start, the "ruler").

A gold *state* is a root of a branch collect run with `all_actions` and
`replays`: a context o_t on the starting policy's own play where every admissible
action, the played one included, was forced and played out several times. An
action's win rate over its replays is the truth; an action is judged against the
PLAYED action by comparing the two win rates.

Each state is sorted into one of three kinds:

* fine     -- no labelled action has a win rate above the played one's by more
              than `fine_gap`, and enough of the admissible set is labelled;
* beaten   -- some action's win rate exceeds the played one's by `clear_gap` or
              more ("clearly better");
* unclear  -- neither; left out of the score.

For a value network the advantage rule is applied exactly as the Advantage Server
applies it (cleared regrets with a margin, anchored target, difference form), with
the policy's probabilities taken from the node file the collect wrote. Then:

* harm  -- fine states where the played action is pushed down;
* catch -- beaten states where the played action is pushed down;
* lift  -- beaten states where the most-lifted action is clearly better.

Pure functions first (records -> states -> truth -> score); the CLI loads the
networks and writes gate/push.json. `truth_only` skips the networks and reports the
kinds and the label noise, which is what the pilot needs.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Callable, Iterable

import torch
from pydantic import Field

from prime_rl.advantage_server.action_advantage import cleared_regrets, difference_advantage, tokenize_action
from prime_rl.advantage_trainer.ckpt import load_warm_start
from prime_rl.configs.shared import LogConfig
from prime_rl.orchestrator.value_networks import ValueNetworkBackbone, ValueNetworkConfig
from prime_rl.utils.config import BaseConfig, cli
from prime_rl.utils.logger import setup_logger
from prime_rl.value_warmstart.contrast_eval import _episode_return, _joint_dp
from prime_rl.value_warmstart.dataset_io import BranchRecord, read_branch_records
from prime_rl.value_warmstart.merge_collects import read_sources

KINDS = ("fine", "beaten", "unclear")


class PushEvalConfig(BaseConfig):
    """Score a value_state.pt on a gold test set, or read the set's truth alone."""

    dataset: Annotated[Path, Field(description="records.msgpack of an all_actions collect.")]
    nodes_dir: Annotated[
        Path | None,
        Field(description="The collect's nodes/ directory (policy probabilities). None = next to the dataset."),
    ] = None
    value_state: Annotated[Path | None, Field(description="value_state.pt to score. Not needed with truth_only.")] = (
        None
    )
    model: Annotated[
        ValueNetworkConfig | None,
        Field(description="Value backbone config (same base model + LoRA as training)."),
    ] = None
    truth_only: Annotated[
        bool,
        Field(description="Skip the networks: report the kinds of state and the label noise (the pilot's read-out)."),
    ] = False
    holdout_only: Annotated[bool, Field(description="Use only records flagged holdout.")] = False
    include_trunk: Annotated[
        bool,
        Field(description="Count the trunk's own outcome as one more replay of the played action."),
    ] = True
    # --- the advantage rule, as the run sets it ---
    margin: Annotated[float, Field(ge=0.0, description="delta: regret at or below it counts as zero.")] = 0.10
    anchor: Annotated[float, Field(gt=0.0, description="kappa: votes held by the current policy.")] = 0.3
    eps: Annotated[float, Field(ge=0.0, description="An advantage larger than this in size is a push.")] = 0.05
    # --- the truth ---
    min_replays: Annotated[int, Field(ge=1, description="Replays an action needs before its win rate counts.")] = 2
    clear_gap: Annotated[
        float,
        Field(gt=0.0, le=1.0, description="Win-rate lead over the played action that makes an action clearly better."),
    ] = 0.5
    fine_gap: Annotated[
        float,
        Field(ge=0.0, lt=1.0, description="A state is fine only if no labelled action leads the played one by more."),
    ] = 0.0
    min_coverage: Annotated[
        float,
        Field(ge=0.0, le=1.0, description="Share of the admissible set that must be labelled for a state to be fine."),
    ] = 0.8
    # --- the pass line (plan section 3) ---
    harm_max: Annotated[float, Field(ge=0.0, le=1.0)] = 0.10
    catch_min: Annotated[float, Field(ge=0.0, le=1.0)] = 0.80
    lift_min: Annotated[float, Field(ge=0.0, le=1.0)] = 0.80
    bootstrap: Annotated[int, Field(ge=0, description="Resamples of games for the 95 % intervals (0 = none).")] = 2000
    seed: Annotated[int, Field(description="Seed of the resampling.")] = 0
    out: Annotated[Path, Field(description="Output JSON path.")] = Path("gate/push.json")
    log: Annotated[LogConfig, Field(description="Logging configuration.")] = LogConfig()


@dataclass
class GoldState:
    """One root with the outcomes of its forced actions."""

    game_id: str
    trunk_id: str
    turn: int
    o_ids: list[int]
    admissible: list[str]
    a_taken: str
    returns: dict[str, list[float]] = field(default_factory=dict)  # action -> one return per replay


@dataclass
class Truth:
    kind: str  # "fine" | "beaten" | "unclear"
    rate_taken: float | None
    rates: dict[str, float]  # labelled actions only (>= min_replays), the played one included
    better: list[str]  # clearly better than the played action
    coverage: float  # labelled share of the admissible set


# ---------------------------------------------------------------------------
# records -> states
# ---------------------------------------------------------------------------


def gold_states(
    records: Iterable[BranchRecord],
    *,
    holdout_only: bool = False,
    include_trunk: bool = True,
) -> list[GoldState]:
    """Group forced branches by root (trunk_id, root_turn). The context, the
    admissible set and the played action come from the trunk's own decision point
    at that turn. Deterministic order: by (game_id, trunk_id, turn)."""
    trunks: dict[str, BranchRecord] = {}
    by_root: dict[tuple[str, int], list[BranchRecord]] = defaultdict(list)
    for r in records:
        if holdout_only and not r.holdout:
            continue
        if not r.root_forced:
            trunks[r.trunk_id] = r
        elif r.root_dp is not None and r.root_turn is not None:
            by_root[(r.trunk_id, int(r.root_turn))].append(r)

    states: list[GoldState] = []
    for (trunk_id, turn), branches in by_root.items():
        trunk = trunks.get(trunk_id)
        found = _joint_dp(trunk, turn) if trunk is not None else None
        if found is None:
            continue
        s, dp = found
        admissible = [str(a) for a in dp.admissible_actions]
        state = GoldState(
            game_id=trunk.game_id,
            trunk_id=trunk_id,
            turn=turn,
            o_ids=list(s.prompt_ids) + list(s.completion_ids[: dp.response_start]),
            admissible=admissible,
            a_taken=admissible[dp.executed_action_idx],
        )
        for b in branches:
            fb = _joint_dp(b, b.root_dp)
            if fb is None:
                continue
            _, bdp = fb
            action = str(bdp.admissible_actions[bdp.executed_action_idx])
            state.returns.setdefault(action, []).append(_episode_return(b))
        if include_trunk:
            state.returns.setdefault(state.a_taken, []).append(_episode_return(trunk))
        states.append(state)
    states.sort(key=lambda st: (st.game_id, st.trunk_id, st.turn))
    return states


# ---------------------------------------------------------------------------
# truth
# ---------------------------------------------------------------------------


def truth_of(
    state: GoldState,
    *,
    min_replays: int = 2,
    clear_gap: float = 0.5,
    fine_gap: float = 0.0,
    min_coverage: float = 0.8,
) -> Truth:
    """Sort a state by comparing every labelled action's win rate with the played
    action's. Without enough replays of the played action nothing can be said."""
    rates = {a: sum(g) / len(g) for a, g in state.returns.items() if len(g) >= min_replays and a in state.admissible}
    coverage = len(rates) / len(state.admissible) if state.admissible else 0.0
    if state.a_taken not in rates:
        return Truth("unclear", None, rates, [], coverage)
    rt = rates[state.a_taken]
    leads = {a: r - rt for a, r in rates.items() if a != state.a_taken}
    better = sorted(a for a, d in leads.items() if d >= clear_gap - 1e-12)
    if better:
        return Truth("beaten", rt, rates, better, coverage)
    if coverage >= min_coverage - 1e-12 and all(d <= fine_gap + 1e-12 for d in leads.values()):
        return Truth("fine", rt, rates, [], coverage)
    return Truth("unclear", rt, rates, [], coverage)


def label_noise(states: list[GoldState], *, min_replays: int = 2) -> dict:
    """How often replays of the same (state, action) disagree: the pilot's question.

    Per (state, action) with at least `min_replays` returns: `mixed` when its
    returns are not all equal. `flip_rate` estimates the chance that two replays
    of the same action end differently, pooled over those actions."""
    n_actions = mixed = pairs = pair_diff = 0
    hist: dict[str, int] = defaultdict(int)
    for st in states:
        for g in st.returns.values():
            n = len(g)
            if n < min_replays:
                continue
            n_actions += 1
            wins = sum(1 for x in g if x > 0.0)
            hist[f"{wins}/{n}"] += 1
            if 0 < wins < n:
                mixed += 1
            if n >= 2:
                pairs += n * (n - 1) // 2
                pair_diff += wins * (n - wins)
    return {
        "actions_with_replays": n_actions,
        "mixed_share": (mixed / n_actions) if n_actions else None,
        "flip_rate": (pair_diff / pairs) if pairs else None,
        "wins_of_replays": dict(sorted(hist.items())),
    }


# ---------------------------------------------------------------------------
# the rule
# ---------------------------------------------------------------------------


def rule_advantages(q_plus: list[float], v: float, pi_hat: list[float], *, margin: float, anchor: float) -> list[float]:
    """The difference-form advantage of every action, as the Advantage Server computes it."""
    regret = cleared_regrets(q_plus, v, margin)
    total = sum(regret)
    return [difference_advantage(regret[i], total, pi_hat[i], anchor) for i in range(len(q_plus))]


def pi_hat_from_node(node: dict, admissible: list[str]) -> list[float] | None:
    """The policy's probability of each admissible action at a node, not
    renormalised: the mean over the SAMPLED blocks (block 0 is the greedy one) of
    the block's normalized row times its admissible mass. None when the node file
    does not cover the admissible set."""
    node_adm = [str(a) for a in node.get("admissible") or []]
    blocks = node.get("blocks") or []
    sampled = blocks[1:] if len(blocks) > 1 else blocks
    if not sampled or not node_adm:
        return None
    index = {a: i for i, a in enumerate(node_adm)}
    if any(a not in index for a in admissible):
        return None
    out = [0.0] * len(admissible)
    for b in sampled:
        mass = b.get("mass")
        scale = float(mass) if mass is not None else 1.0
        row = b["row"]
        for j, a in enumerate(admissible):
            out[j] += float(row[index[a]]) * scale
    return [x / len(sampled) for x in out]


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------


def score_states(
    states: list[GoldState],
    value_fn: Callable[[list[int], list[str]], tuple[float, list[float]]],
    pi_hat_fn: Callable[[GoldState], list[float] | None],
    *,
    margin: float = 0.10,
    anchor: float = 0.3,
    eps: float = 0.05,
    min_replays: int = 2,
    clear_gap: float = 0.5,
    fine_gap: float = 0.0,
    min_coverage: float = 0.8,
) -> list[dict]:
    """One row per state: its truth, the networks' values, the rule's advantages,
    and the three verdicts (None where the verdict does not apply)."""
    rows: list[dict] = []
    for st in states:
        truth = truth_of(st, min_replays=min_replays, clear_gap=clear_gap, fine_gap=fine_gap, min_coverage=min_coverage)
        row: dict = {
            "game_id": st.game_id,
            "trunk_id": st.trunk_id,
            "turn": st.turn,
            "a_taken": st.a_taken,
            "n_admissible": len(st.admissible),
            "kind": truth.kind,
            "rate_taken": truth.rate_taken,
            "rates": truth.rates,
            "better": truth.better,
            "coverage": truth.coverage,
            "replays": {a: len(g) for a, g in st.returns.items()},
        }
        pi_hat = pi_hat_fn(st)
        if pi_hat is None:
            row.update(scored=False, reason="no policy probabilities for this state")
            rows.append(row)
            continue
        v, q_plus = value_fn(st.o_ids, st.admissible)
        adv = rule_advantages(q_plus, v, pi_hat, margin=margin, anchor=anchor)
        i_taken = st.admissible.index(st.a_taken)
        top = max(range(len(adv)), key=lambda i: adv[i])
        lifted = st.admissible[top] if adv[top] > eps else None
        pushed_down = adv[i_taken] < -eps
        row.update(
            scored=True,
            v=v,
            q_plus=q_plus,
            pi_hat=pi_hat,
            adv=adv,
            adv_taken=adv[i_taken],
            pushed_down=pushed_down,
            lifted=lifted,
            harm=pushed_down if truth.kind == "fine" else None,
            catch=pushed_down if truth.kind == "beaten" else None,
            lift=(lifted in truth.better) if truth.kind == "beaten" else None,
            lift_weak=(lifted is not None and truth.rates.get(lifted, -1.0) > (truth.rate_taken or 0.0))
            if truth.kind == "beaten"
            else None,
        )
        rows.append(row)
    return rows


def rows_by_source(rows: list[dict], source_of: dict[str, str]) -> dict[str, list[dict]]:
    """The scored rows split by the source of their game (a merged gold set's
    sources.json). {} when the set has no sources; a game outside the map is "other"."""
    if not source_of:
        return {}
    out: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        out[source_of.get(r["game_id"], "other")].append(r)
    return dict(out)


def _rate(rows: list[dict], key: str) -> tuple[float | None, int]:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return ((sum(1 for x in vals if x) / len(vals)) if vals else None), len(vals)


def summarize(
    rows: list[dict],
    *,
    harm_max: float = 0.10,
    catch_min: float = 0.80,
    lift_min: float = 0.80,
    bootstrap: int = 2000,
    seed: int = 0,
) -> dict:
    """The three numbers with their counts, a 95 % interval from resampling games,
    and the verdict against the pass line (the interval has to clear it)."""
    scored = [r for r in rows if r.get("scored")]
    out: dict = {
        "states": len(rows),
        "scored": len(scored),
        "kinds": {k: sum(1 for r in rows if r["kind"] == k) for k in KINDS},
        "games": len({r["game_id"] for r in rows}),
    }
    by_game: dict[str, list[dict]] = defaultdict(list)
    for r in scored:
        by_game[r["game_id"]].append(r)
    games = sorted(by_game)
    rng = random.Random(seed)
    draws: dict[str, list[float]] = {"harm": [], "catch": [], "lift": []}
    for _ in range(bootstrap if len(games) > 1 else 0):
        sample = [r for g in (rng.choice(games) for _ in games) for r in by_game[g]]
        for key in draws:
            rate, _ = _rate(sample, key)
            if rate is not None:
                draws[key].append(rate)

    def interval(key: str) -> list[float] | None:
        xs = sorted(draws[key])
        if len(xs) < 40:
            return None
        return [xs[int(0.025 * (len(xs) - 1))], xs[int(round(0.975 * (len(xs) - 1)))]]

    lines = {"harm": ("max", harm_max), "catch": ("min", catch_min), "lift": ("min", lift_min)}
    verdicts = {}
    for key, (side, line) in lines.items():
        rate, n = _rate(scored, key)
        ci = interval(key)
        if rate is None:
            verdict = "NOT AVAILABLE"
        elif ci is None:
            verdict = "NO INTERVAL"
        elif side == "max":
            verdict = "PASS" if ci[1] <= line else "FAIL"
        else:
            verdict = "PASS" if ci[0] >= line else "FAIL"
        out[key] = {"rate": rate, "n": n, "interval95": ci, "line": line, "side": side, "verdict": verdict}
        verdicts[key] = verdict
    lift_weak, _ = _rate(scored, "lift_weak")
    out["lift_weak"] = lift_weak
    helpful = sum(1 for r in scored if r.get("catch"))
    harmful = sum(1 for r in scored if r.get("harm"))
    out["helpful_pushes"] = helpful
    out["harmful_pushes"] = harmful
    out["helpful_to_harmful"] = (helpful / harmful) if harmful else None
    out["passed"] = all(v == "PASS" for v in verdicts.values())
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_value_fn(
    backbone: ValueNetworkBackbone, tokenizer, device
) -> Callable[[list[int], list[str]], tuple[float, list[float]]]:
    """(V(o), [Q+(o, a) for a in admissible]), eval mode, no grad. On a GPU the
    actions continue off one shared prefill of o, as in the Advantage Server."""
    backbone.eval()

    @torch.no_grad()
    def value_fn(o_ids: list[int], admissible: list[str]) -> tuple[float, list[float]]:
        o = torch.tensor([list(o_ids)], dtype=torch.long, device=device)
        v = float(backbone.forward_v(o)[0].float().item())
        if device.type == "cuda":
            acts = [torch.tensor(tokenize_action(tokenizer, a), dtype=torch.long, device=device) for a in admissible]
            q = [float(x) for x in backbone.forward_q_plus_action_shared_o(o, acts).float()]
        else:
            q = []
            for a in admissible:
                a_ids = torch.tensor([tokenize_action(tokenizer, a)], dtype=torch.long, device=device)
                q.append(float(backbone.forward_q_plus_action(o, a_ids)[0].float().item()))
        return v, q

    return value_fn


def node_pi_hat_fn(nodes_dir: Path) -> Callable[[GoldState], list[float] | None]:
    """Policy probabilities from the collect's node files: nodes/<game>/<trunk>/t<turn>.json."""

    def fn(state: GoldState) -> list[float] | None:
        path = Path(nodes_dir) / state.game_id / state.trunk_id / f"t{state.turn}.json"
        try:
            node = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return pi_hat_from_node(node, state.admissible)

    return fn


def push_eval(config: PushEvalConfig) -> dict:
    logger = setup_logger(config.log.level)
    states = gold_states(
        read_branch_records(config.dataset), holdout_only=config.holdout_only, include_trunk=config.include_trunk
    )
    if not states:
        raise ValueError(f"No gold states in {config.dataset} (holdout_only={config.holdout_only}).")
    truth_kw = dict(
        min_replays=config.min_replays,
        clear_gap=config.clear_gap,
        fine_gap=config.fine_gap,
        min_coverage=config.min_coverage,
    )
    payload: dict = {
        "schema": 1,
        "dataset": str(config.dataset),
        "settings": {
            "margin": config.margin,
            "anchor": config.anchor,
            "eps": config.eps,
            "include_trunk": config.include_trunk,
            **truth_kw,
        },
        "label_noise": label_noise(states, min_replays=config.min_replays),
    }
    if config.truth_only:
        kinds = [truth_of(st, **truth_kw).kind for st in states]
        payload["truth"] = {
            "states": len(states),
            "games": len({st.game_id for st in states}),
            "kinds": {k: kinds.count(k) for k in KINDS},
            "actions_per_state": sum(len(st.admissible) for st in states) / len(states),
            "labelled_per_state": sum(len(st.returns) for st in states) / len(states),
        }
        logger.success(f"truth only: {payload['truth']} | label noise: {payload['label_noise']}")
    else:
        if config.value_state is None or config.model is None:
            raise ValueError("value_state and [model] are required unless truth_only = true")
        from transformers import AutoModel, AutoTokenizer

        if torch.cuda.is_available():
            device, dtype = torch.device("cuda"), torch.bfloat16
        else:
            device, dtype = torch.device("cpu"), torch.float32
        logger.info(f"Loading value backbone from {config.model.base_model_name} ({device}, {dtype})")
        base = AutoModel.from_pretrained(config.model.base_model_name, dtype=dtype)
        backbone = ValueNetworkBackbone(base, lora_config=config.model.lora, polyak_tau=config.model.polyak_tau).to(
            device=device, dtype=dtype
        )
        load_warm_start(backbone, config.value_state)
        tokenizer = AutoTokenizer.from_pretrained(config.model.base_model_name)
        nodes_dir = config.nodes_dir if config.nodes_dir is not None else Path(config.dataset).parent / "nodes"
        logger.info(f"Scoring {len(states)} gold states (policy probabilities from {nodes_dir})")
        rows = score_states(
            states,
            build_value_fn(backbone, tokenizer, device),
            node_pi_hat_fn(nodes_dir),
            margin=config.margin,
            anchor=config.anchor,
            eps=config.eps,
            **truth_kw,
        )
        payload["value_state"] = str(config.value_state)
        payload["summary"] = summarize(
            rows,
            harm_max=config.harm_max,
            catch_min=config.catch_min,
            lift_min=config.lift_min,
            bootstrap=config.bootstrap,
            seed=config.seed,
        )
        payload["states"] = rows
        by_source = rows_by_source(rows, read_sources(Path(config.dataset).parent))
        if by_source:
            payload["by_source"] = {
                name: summarize(
                    part,
                    harm_max=config.harm_max,
                    catch_min=config.catch_min,
                    lift_min=config.lift_min,
                    bootstrap=config.bootstrap,
                    seed=config.seed,
                )
                for name, part in by_source.items()
            }
        s = payload["summary"]

        def show(key: str) -> str:
            d = s[key]
            if d["rate"] is None:
                return f"{key} n/a"
            ci = d["interval95"]
            band = f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""
            return f"{key} {d['rate']:.3f}{band} (n={d['n']}, {d['verdict']})"

        logger.success(
            f"push score: {show('harm')} | {show('catch')} | {show('lift')} | kinds {s['kinds']} | "
            f"passed={s['passed']} -> {config.out}"
        )
    Path(config.out).parent.mkdir(parents=True, exist_ok=True)
    with open(config.out, "w") as f:
        json.dump(payload, f, indent=2)
    return payload


def main() -> None:
    """Entry point: `python -m prime_rl.value_warmstart.push_eval @ push.toml`."""
    push_eval(cli(PushEvalConfig))


if __name__ == "__main__":
    main()
