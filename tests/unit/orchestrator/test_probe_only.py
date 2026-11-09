"""branched-dataset Phase 4 -- probe.only: one probe round at checkpoint step 0
against the value networks the Advantage Server loaded, then exit.

Reuses the ProbeRunner harness (faked generate, FakeAdvantageServer); the
orchestrator's early-exit path calls run_probe_only and tears down. The launcher
skips the trainer and Advantage Trainer processes in this mode.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.entrypoints.rl import is_probe_only
from prime_rl.orchestrator.probe import run_probe_only
from tests.unit.orchestrator.test_probe_runner import _rollouts, _runner


def _probe_dict(**extra) -> dict:
    args = {"num_reasoning_blocks": 2, "probe_capture": True, "probe_set": "alfworld-24-v1"}
    probe = {"env": {"id": "alfworld-env", "name": "alfworld-probe", "args": args}}
    probe.update(extra)
    return probe


def test_only_flag_defaults_false_and_parses():
    assert OrchestratorConfig(probe=_probe_dict()).probe.only is False
    assert OrchestratorConfig(probe=_probe_dict(only=True)).probe.only is True


def test_run_probe_only_writes_round_zero_and_metrics(tmp_path, monkeypatch):
    runner, server, monitor, captured = _runner(tmp_path, _rollouts(2), monkeypatch)
    metrics = asyncio.run(run_probe_only(runner, output_dir=tmp_path / "run"))

    probes = tmp_path / "run" / "probes"
    assert sorted(p.name for p in (probes / "ckpt_0").glob("*.json")) == ["probe-0.json", "probe-1.json"]
    lines = [json.loads(l) for l in (probes / "index.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {l["ckpt_step"] for l in lines} == {0}
    # the round ran at ckpt 0 / orchestrator step 0, once
    assert metrics["progress/ckpt_step"] == 0 and metrics["step"] == 0
    assert len(monitor.calls) == 1
    # every server call was a read-only diagnostics call
    assert len(server.calls) == 2 and all(c.get("return_diagnostics") for c in server.calls)
    # the gate reads this file next to the records
    saved = json.loads((probes / "probe_only_metrics.json").read_text(encoding="utf-8"))
    assert saved["probe/num_rollouts"] == 2.0
    assert saved["progress/ckpt_step"] == 0
    # the runner has no handle on either trainer transport
    assert not any("sender" in a for a in vars(runner))


def test_is_probe_only():
    assert is_probe_only(SimpleNamespace(orchestrator=SimpleNamespace(probe=None))) is False
    assert is_probe_only(SimpleNamespace(orchestrator=SimpleNamespace(probe=SimpleNamespace(only=False)))) is False
    assert is_probe_only(SimpleNamespace(orchestrator=SimpleNamespace(probe=SimpleNamespace(only=True)))) is True
