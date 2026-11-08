"""training-probe Phase 0 -- ProbeConfig on OrchestratorConfig.

The probe is opt-in: absent by default, and when present it must describe an
env that can actually capture (m >= 2 reasoning blocks, probe_capture=true).
"""

import pytest
import tomli_w
from pydantic import ValidationError

from prime_rl.configs.orchestrator import OrchestratorConfig, ProbeConfig, ProbeEnvConfig, ProbeSamplingConfig
from prime_rl.utils.config import cli


def _probe_dict(**overrides) -> dict:
    args = {"num_reasoning_blocks": 2, "probe_capture": True, "probe_set": "alfworld-24-v1"}
    args.update(overrides.pop("args", {}))
    probe = {"env": {"id": "alfworld-env", "name": "alfworld-probe", "args": args}}
    probe.update(overrides)
    return probe


def test_probe_absent_by_default():
    assert OrchestratorConfig().probe is None


def test_probe_config_parses_with_defaults():
    cfg = OrchestratorConfig(probe=_probe_dict())
    assert isinstance(cfg.probe, ProbeConfig)
    assert isinstance(cfg.probe.env, ProbeEnvConfig)
    assert isinstance(cfg.probe.sampling, ProbeSamplingConfig)
    assert cfg.probe.env.resolved_name == "alfworld-probe"
    assert cfg.probe.interval == 10
    assert cfg.probe.probe_base_model is True
    assert cfg.probe.skip_probe_on_resume is True
    assert cfg.probe.output_subdir == "probes"
    assert cfg.probe.max_retries == 0
    assert cfg.probe.emit_eval_metrics is True
    # executed response is greedy by default (D-4)
    assert cfg.probe.sampling.temperature == 0.0


def test_probe_sampling_is_eval_sampling_shape():
    cfg = OrchestratorConfig(probe=_probe_dict(sampling={"max_tokens": 512, "min_p": 0.1}))
    assert cfg.probe.sampling.max_tokens == 512
    assert cfg.probe.sampling.min_p == 0.1
    assert cfg.probe.sampling.temperature == 0.0


@pytest.mark.parametrize("m", [None, 1, 0, True, "2"])
def test_probe_rejects_fewer_than_two_blocks(m):
    args = {} if m is None else {"num_reasoning_blocks": m}
    if m is None:
        probe = _probe_dict()
        del probe["env"]["args"]["num_reasoning_blocks"]
    else:
        probe = _probe_dict(args=args)
    with pytest.raises(ValidationError, match="num_reasoning_blocks"):
        OrchestratorConfig(probe=probe)


def test_probe_rejects_missing_capture_flag():
    probe = _probe_dict()
    del probe["env"]["args"]["probe_capture"]
    with pytest.raises(ValidationError, match="probe_capture"):
        OrchestratorConfig(probe=probe)
    with pytest.raises(ValidationError, match="probe_capture"):
        OrchestratorConfig(probe=_probe_dict(args={"probe_capture": False}))


def test_probe_requires_env_block():
    with pytest.raises(ValidationError):
        OrchestratorConfig(probe={"interval": 5})


def test_probe_name_must_not_collide_with_eval_env():
    eval_cfg = {"env": [{"id": "alfworld-env", "name": "alfworld-probe"}]}
    with pytest.raises(ValidationError, match="collides"):
        OrchestratorConfig(eval=eval_cfg, probe=_probe_dict())
    # distinct names coexist (not the ALFWorld setup, but allowed)
    eval_cfg = {"env": [{"id": "alfworld-env", "name": "alfworld-valid-seen"}]}
    cfg = OrchestratorConfig(eval=eval_cfg, probe=_probe_dict())
    assert cfg.eval is not None and cfg.probe is not None


def test_bench_mode_disables_probe():
    cfg = OrchestratorConfig(bench=True, probe=_probe_dict())
    assert cfg.probe is None
    assert cfg.eval is None


def test_probe_toml_round_trip(tmp_path):
    toml = {
        "probe": {
            "interval": 10,
            "probe_base_model": True,
            "sampling": {"temperature": 0.0, "max_tokens": 512},
            "env": {
                "id": "alfworld-env",
                "name": "alfworld-probe",
                "args": {
                    "probe_set": "alfworld-24-v1",
                    "num_reasoning_blocks": 2,
                    "probe_capture": True,
                    "probe_block_temperature": 0.8,
                    "probe_block_min_p": 0.1,
                    "oracle": "handcoded",
                },
            },
        }
    }
    path = tmp_path / "orch.toml"
    path.write_text(tomli_w.dumps(toml))
    cfg = cli(OrchestratorConfig, args=["@", path.as_posix()])
    assert cfg.probe is not None
    assert cfg.probe.env.args["probe_set"] == "alfworld-24-v1"
    assert cfg.probe.env.args["oracle"] == "handcoded"
    assert cfg.probe.sampling.max_tokens == 512
