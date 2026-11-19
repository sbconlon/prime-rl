"""The two run A configs of the ARM redesign (KB plan/arm-redesign, Phase 5).

Run A is the new advantage and target rule, the per-property metrics and the
optimizer fixes in the online Advantage Trainer, with no branching. These tests
pin what the configs say, so that a later edit cannot quietly turn a setting off:

* both load through the launcher's own config class;
* run A holds every redesign setting (plan D-3, D-5, D-6, D-9, D-10);
* run A differs from the config it was generated from only in the listed places;
* the three-step trial differs from run A only in its length, names and the
  milestone interval.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from prime_rl.configs.rl import RLConfig
from prime_rl.utils.config import cli

CONFIGS = Path(__file__).resolve().parents[2] / "configs" / "alfworld"
RUN_A = "arm_action_1.5b_ctx2048_pps_redesign_a.toml"
TRIAL = "arm_action_1.5b_ctx2048_pps_redesign_a_trial.toml"
PARENT = "arm_action_1.5b_ctx2048_pps_warm_branched_probe.toml"
V3 = "/home/3312841/branched/pps/value_nets_v3/checkpoints/step_0/trainer/value_state.pt"


def _toml(name: str) -> dict:
    return tomllib.loads((CONFIGS / name).read_text(encoding="utf-8"))


def _load(name: str) -> RLConfig:
    return cli(RLConfig, args=["@", (CONFIGS / name).as_posix()])


@pytest.mark.parametrize("name", [RUN_A, TRIAL])
def test_config_loads_with_the_redesign_settings(name):
    cfg = _load(name)

    server = cfg.advantage_server
    assert server.arm_action_advantage_form == "difference"
    assert server.arm_margin == pytest.approx(0.10)
    assert server.arm_anchor == pytest.approx(0.3)
    assert server.arm_target_form == "average"
    assert server.arm_phi_decay == pytest.approx(0.9)
    assert server.gamma == 1.0

    trainer = cfg.advantage_trainer
    assert trainer.fp32_master is True
    assert trainer.learning_rate == pytest.approx(1e-4)
    assert trainer.head_learning_rate == pytest.approx(1e-5)
    assert trainer.max_grad_norm == pytest.approx(1.0)
    assert trainer.warmup_cycles == 10
    assert trainer.milestone_every_steps is not None

    # both value processes start from the same file, and it is value_nets_v3
    assert server.warm_start_path == trainer.warm_start_path == V3


@pytest.mark.parametrize("name", [RUN_A, TRIAL])
def test_the_return_is_the_real_outcome(name):
    """n_step covers a whole episode, so g never bootstraps from V (plan D-6)."""
    raw = _toml(name)
    max_turns = raw["orchestrator"]["env"][0]["args"]["max_turns"]
    assert raw["advantage_server"]["n_step"] == max_turns == 15
    assert raw["orchestrator"]["probe"]["env"]["args"]["max_turns"] == max_turns


@pytest.mark.parametrize("name", [RUN_A, TRIAL])
def test_run_a_has_no_branching_and_probes_the_pps_set(name):
    raw = _toml(name)
    orch = raw["orchestrator"]
    # 16 trunks per step, 4 per game
    assert orch["batch_size"] == 16
    assert orch["rollouts_per_example"] == 4
    assert orch["probe"]["interval"] == 10
    assert orch["probe"]["env"]["args"]["probe_set"] == "alfworld-pps-24-v1"
    # the pick_and_place_simple stage lasts longer than the run
    assert orch["buffer"]["curriculum"][0] >= raw["max_steps"]
    assert orch["env"][0]["args"]["curriculum"][0] == {"pick_and_place_simple": 1.0, "default": 0.0}


def test_run_a_length_and_milestones():
    raw = _toml(RUN_A)
    assert raw["max_steps"] == raw["advantage_trainer"]["max_steps"] == 100
    assert raw["advantage_trainer"]["milestone_every_steps"] == 10
    assert raw["output_dir"] == "/home/3312841/runs/arm-redesign-a-001"
    assert raw["advantage_trainer"]["output_dir"] == raw["output_dir"] + "/advantage_trainer"


def test_run_a_differs_from_its_parent_only_in_the_listed_settings():
    parent, run_a = _toml(PARENT), _toml(RUN_A)

    def strip(cfg: dict) -> dict:
        cfg = json.loads(json.dumps(cfg))
        cfg.pop("max_steps")
        cfg.pop("output_dir")
        cfg["orchestrator"]["model"]["lora"].pop("name")
        cfg["orchestrator"]["probe"].pop("interval")
        cfg["orchestrator"]["probe"]["env"]["args"].pop("probe_set")
        server = cfg["advantage_server"]
        server.pop("warm_start_path")
        for key in ("arm_action_advantage_form", "arm_margin", "arm_anchor", "arm_target_form", "n_step"):
            server.pop(key, None)
        trainer = cfg["advantage_trainer"]
        for key in ("max_steps", "output_dir", "warm_start_path", "learning_rate"):
            trainer.pop(key)
        for key in ("head_learning_rate", "max_grad_norm", "warmup_cycles", "fp32_master", "milestone_every_steps"):
            trainer.pop(key, None)
        return cfg

    assert strip(run_a) == strip(parent)
    # the parent ran the old rule: none of the redesign keys were set there
    assert "arm_action_advantage_form" not in parent["advantage_server"]
    assert "fp32_master" not in parent["advantage_trainer"]
    assert parent["advantage_trainer"]["learning_rate"] == pytest.approx(5e-4)


def test_trial_differs_from_run_a_only_in_length_names_and_milestone():
    run_a, trial = _toml(RUN_A), _toml(TRIAL)

    assert trial["max_steps"] == trial["advantage_trainer"]["max_steps"] == 3
    # one milestone inside three steps, so the scoring launcher has a file to read
    assert trial["advantage_trainer"]["milestone_every_steps"] == 2
    assert trial["output_dir"] == "/home/3312841/runs/arm-redesign-a-trial"
    assert trial["output_dir"] != run_a["output_dir"]
    assert trial["advantage_trainer"]["output_dir"] == trial["output_dir"] + "/advantage_trainer"
    assert trial["orchestrator"]["model"]["lora"]["name"] != run_a["orchestrator"]["model"]["lora"]["name"]

    def strip(cfg: dict) -> dict:
        cfg = json.loads(json.dumps(cfg))
        cfg.pop("max_steps")
        cfg.pop("output_dir")
        cfg["orchestrator"]["model"]["lora"].pop("name")
        for key in ("max_steps", "output_dir", "milestone_every_steps"):
            cfg["advantage_trainer"].pop(key)
        return cfg

    assert strip(trial) == strip(run_a)
