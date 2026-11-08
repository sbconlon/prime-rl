"""training-probe Phase 3 -- ProbeRunner round: generate (faked) -> interleave ->
diagnostics (FakeAdvantageServer) -> assemble -> write -> log.

Nothing here touches the training-batch or advantage-batch senders: the runner
has no handle on them by construction (test_nothing_sent_to_trainers pins that).
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
from pathlib import Path

import pytest
import verifiers as vf

import prime_rl.orchestrator.probe as probe_mod
from prime_rl.configs.orchestrator import ProbeConfig, ProbeEnvConfig
from prime_rl.orchestrator.probe import ProbeRunner, ProbeWriter
from tests.unit.orchestrator._stubs import FakeAdvantageServer, _zero_diagnostics_for_samples

FIXTURE = Path(__file__).parent / "fixtures" / "probe_rollout_fixture.json"


def _rollouts(n: int = 2) -> list[vf.RolloutOutput]:
    with FIXTURE.open(encoding="utf-8") as fh:
        base = json.load(fh)
    out = []
    for i in range(n):
        r = copy.deepcopy(base)
        r["example_id"] = i
        r["info"]["probe_id"] = f"probe-{i}"
        r["info"]["role"] = "bite" if i % 2 == 0 else "control"
        r["reward"] = 1.0 if i % 2 == 0 else 0.0
        out.append(vf.RolloutOutput(r))
    return out


class _Dataset:
    def __init__(self, rows):
        self._rows = rows

    def to_list(self):
        return list(self._rows)


class _FakeProbeEnv:
    """In-process probe env stand-in: only get_dataset() is used by the runner;
    generation is faked at the module boundary (probe_mod.generate)."""

    def __init__(self, n: int):
        self.rows = [
            {"example_id": i, "task": "alfworld-probe", "prompt": [], "info": {"probe_id": f"probe-{i}"}}
            for i in range(n)
        ]

    def get_dataset(self, n: int = -1, seed=None):
        return _Dataset(self.rows)


class _Monitor:
    def __init__(self):
        self.calls: list[tuple[dict, int]] = []

    def log(self, metrics, step):
        self.calls.append((dict(metrics), step))


def _config() -> ProbeConfig:
    return ProbeConfig(
        env=ProbeEnvConfig(
            id="alfworld-env",
            name="alfworld-probe",
            args={
                "probe_set": "alfworld-24-v1",
                "probe_capture": True,
                "num_reasoning_blocks": 2,
                "oracle": "handcoded",
                "probe_block_temperature": 0.8,
                "probe_block_min_p": 0.1,
            },
        ),
        interval=10,
    )


def _runner(tmp_path: Path, rollouts, monkeypatch, *, server=None, monitor=None, run_name="unit-run"):
    captured = {}

    async def fake_generate(**kwargs):
        captured.update(kwargs)
        return [copy.deepcopy(r) for r in rollouts]

    monkeypatch.setattr(probe_mod, "generate", fake_generate)

    async def get_client():
        return object()

    server = server or FakeAdvantageServer(initial_weight_step=3)
    monitor = monitor or _Monitor()
    runner = ProbeRunner(
        config=_config(),
        env=_FakeProbeEnv(len(rollouts)),
        model_name=lambda: "policy-lora",
        get_client=get_client,
        adv_client=server,
        output_dir=tmp_path / "run",
        run_name=run_name,
        tokenizer=None,  # steps carry tokens; pretokenize is a no-op
        monitor=monitor,
    )
    return runner, server, monitor, captured


def test_round_writes_files_and_index(tmp_path, monkeypatch):
    runner, server, monitor, captured = _runner(tmp_path, _rollouts(2), monkeypatch)
    metrics = asyncio.run(runner.run_round(ckpt_step=0, orchestrator_step=0))

    # generate was called the way eval calls it: the probe set, one rollout each, greedy
    assert captured["rollouts_per_example"] == 1
    assert captured["model_name"] == "policy-lora"
    assert captured["sampling_args"]["temperature"] == 0.0
    assert set(captured["state_columns"]) >= {"final_env_response", "context_evictions"}
    assert len(captured["examples"]) == 2

    probes = tmp_path / "run" / "probes"
    files = sorted(p.name for p in (probes / "ckpt_0").glob("*.json"))
    assert files == ["probe-0.json", "probe-1.json"]
    lines = [json.loads(l) for l in (probes / "index.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [l["probe_id"] for l in lines] == ["probe-0", "probe-1"]
    assert lines[0]["path"] == "probes/ckpt_0/probe-0.json"
    assert {l["ckpt_step"] for l in lines} == {0} and {l["weight_step"] for l in lines} == {3}
    record = json.loads((probes / "ckpt_0" / "probe-0.json").read_text(encoding="utf-8"))
    assert record["run"] == "unit-run" and record["round"]["weight_step"] == 3
    assert record["num_turns"] == 2 and record["sampling"]["m"] == 2 and record["oracle_mode"] == "handcoded"

    # metrics: eval namespace (D-2) + probe namespace, logged once at the orchestrator step
    assert metrics["eval/alfworld-probe/avg@1"] == 0.5
    assert metrics["eval/alfworld-probe/bite/avg@1"] == 1.0
    assert metrics["probe/num_rollouts"] == 2 and metrics["probe/num_skipped"] == 0
    assert metrics["progress/ckpt_step"] == 0 and metrics["step"] == 0
    assert len(monitor.calls) == 1 and monitor.calls[0][1] == 0
    assert monitor.calls[0][0]["eval/alfworld-probe/avg@1"] == 0.5

    # every server call was a diagnostics call
    assert len(server.calls) == 2 and all(c.get("return_diagnostics") for c in server.calls)
    assert all(c["algorithm"] == "arm" and c["is_terminal"] for c in server.calls)


def test_round_overwrites_and_appends_on_rerun(tmp_path, monkeypatch):
    """Re-running a round for the same ckpt_step (resume without skip) overwrites the
    files and appends a second index line; consumers keep the last."""
    runner, _, _, _ = _runner(tmp_path, _rollouts(1), monkeypatch)
    asyncio.run(runner.run_round(ckpt_step=10, orchestrator_step=11))
    asyncio.run(runner.run_round(ckpt_step=10, orchestrator_step=12))
    probes = tmp_path / "run" / "probes"
    assert sorted(p.name for p in (probes / "ckpt_10").glob("*.json")) == ["probe-0.json"]
    lines = [json.loads(l) for l in (probes / "index.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [l["orchestrator_step"] for l in lines] == [11, 12]
    assert json.loads((probes / "ckpt_10" / "probe-0.json").read_text())["round"]["orchestrator_step"] == 12
    assert not list((probes / "ckpt_10").glob(".*.tmp"))


def test_round_skips_bad_rollout(tmp_path, monkeypatch):
    """One rollout's diagnostics are misaligned: it is skipped and counted, the other
    is written, no exception escapes."""
    calls = {"n": 0}

    def factory(samples, episodic_reward, is_terminal, algorithm):
        diags = _zero_diagnostics_for_samples(samples)
        calls["n"] += 1
        if calls["n"] == 2:  # second rollout: drop a record
            return diags[:-1]
        return diags

    server = FakeAdvantageServer(diagnostics_factory=factory)
    runner, _, monitor, _ = _runner(tmp_path, _rollouts(2), monkeypatch, server=server)
    metrics = asyncio.run(runner.run_round(ckpt_step=0, orchestrator_step=0))
    files = sorted(p.name for p in (tmp_path / "run" / "probes" / "ckpt_0").glob("*.json"))
    assert files == ["probe-0.json"]
    assert metrics["probe/num_skipped"] == 1 and metrics["probe/num_rollouts"] == 1
    assert metrics["eval/alfworld-probe/avg@1"] == 1.0


def test_round_survives_server_error(tmp_path, monkeypatch):
    class _Boom(FakeAdvantageServer):
        async def compute_with_diagnostics(self, samples, episodic_reward, is_terminal, algorithm):
            raise RuntimeError("server down")

    runner, _, monitor, _ = _runner(tmp_path, _rollouts(2), monkeypatch, server=_Boom())
    metrics = asyncio.run(runner.run_round(ckpt_step=0, orchestrator_step=0))
    assert metrics["probe/num_skipped"] == 2 and metrics["probe/num_rollouts"] == 0
    assert "eval/alfworld-probe/avg@1" not in metrics
    assert len(monitor.calls) == 1  # the round still logs (num_skipped is the signal)


def test_nothing_sent_to_trainers():
    """The runner cannot reach the training-batch or advantage-batch senders: its
    constructor takes neither, and its only outbound dependency besides the
    inference client is the diagnostics method of the Advantage Server client."""
    params = set(inspect.signature(ProbeRunner.__init__).parameters)
    assert not {p for p in params if "sender" in p or "batch" in p}
    src = inspect.getsource(probe_mod)
    assert "training_batch_sender" not in src and "advantage_trainer_sender" not in src
    assert "compute_advantages_and_targets(" not in src  # only compute_with_diagnostics


def test_writer_sanitizes_probe_id(tmp_path):
    writer = ProbeWriter(tmp_path / "run", "probes")
    record = {
        "round": {"ckpt_step": 5, "weight_step": 4, "orchestrator_step": 6},
        "probe_id": "a/b", "game_id": "g", "split": "s", "task_type": "t", "role": "bite",
        "won": False, "num_turns": 1,
    }
    path = writer.write(record)
    assert path == tmp_path / "run" / "probes" / "ckpt_5" / "a_b.json"
    line = json.loads((tmp_path / "run" / "probes" / "index.jsonl").read_text())
    assert line["path"] == "probes/ckpt_5/a_b.json" and line["probe_id"] == "a/b"
