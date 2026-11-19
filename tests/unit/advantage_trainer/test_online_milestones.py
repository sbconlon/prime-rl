"""Online Advantage Trainer value milestones (arm-redesign D-9).

``milestone_every_steps`` keeps a permanent weights-only copy of the value
networks every N cycles, in the file format ``warm_start_path`` and
``contrast_eval`` read, so a run's value networks can be scored on the held-out
contrast pairs afterwards. Opt-in: by default nothing is written.

Runs the real train() entrypoint on the tiny CPU backbone (harness of
test_online_optimizer.py).
"""

from __future__ import annotations

from pathlib import Path

import torch

from prime_rl.advantage_trainer.ckpt import load_warm_start
from prime_rl.configs.advantage_trainer import AdvantageTrainerConfig
from tests.unit.advantage_trainer.test_online_optimizer import _config, _one, _online, _trainable
from tests.unit.advantage_trainer.test_train_step_action import _backbone


def _milestone(out: Path, step: int) -> Path:
    return out / "milestones" / "checkpoints" / f"step_{step}" / "trainer" / "value_state.pt"


def test_no_milestones_by_default(tmp_path, monkeypatch):
    assert AdvantageTrainerConfig.model_fields["milestone_every_steps"].default is None
    _online(monkeypatch, tmp_path, [_one(i) for i in range(2)], max_steps=2)
    assert not (tmp_path / "milestones").exists()


def test_milestones_are_weights_only_and_loadable(tmp_path, monkeypatch):
    seen = _online(
        monkeypatch, tmp_path, [_one(i) for i in range(4)], max_steps=4, milestone_every_steps=2
    )
    kept = sorted(p.name for p in (tmp_path / "milestones" / "checkpoints").iterdir())
    assert kept == ["step_2", "step_4"]  # every second cycle, the final step included

    payloads = {s: torch.load(_milestone(tmp_path, s), map_location="cpu", weights_only=False) for s in (2, 4)}
    for step, payload in payloads.items():
        assert payload["step"] == step
        assert "trainable_state_dict" in payload
        assert "optimizer_state_dict" not in payload  # weights only

    # The final milestone holds the trainer's final weights, and loads through the
    # same reader the warm start and contrast_eval use.
    final = _trainable(seen.backbone)
    fresh = _backbone()
    load_warm_start(fresh, str(_milestone(tmp_path, 4)))
    loaded = _trainable(fresh)
    assert set(loaded) == set(final)
    for name, value in final.items():
        assert torch.equal(loaded[name].to(value.dtype), value), name

    # Training went on after step 2: that milestone is a different set of weights.
    two = payloads[2]["trainable_state_dict"]
    four = payloads[4]["trainable_state_dict"]
    assert any(not torch.equal(two[k], four[k]) for k in four)


def test_milestones_work_with_the_master_weight_optimizer(tmp_path, monkeypatch):
    seen = _online(
        monkeypatch, tmp_path, [_one(i) for i in range(2)], max_steps=2, milestone_every_steps=1,
        fp32_master=True, head_learning_rate=1e-5, max_grad_norm=1.0,
    )
    assert _milestone(tmp_path, 1).is_file() and _milestone(tmp_path, 2).is_file()
    final = _trainable(seen.backbone)
    saved = torch.load(_milestone(tmp_path, 2), map_location="cpu", weights_only=False)["trainable_state_dict"]
    # the milestone stores the model's parameters (what the Advantage Server serves),
    # not the optimizer's float32 masters
    for name, value in final.items():
        assert torch.equal(saved[name].to(value.dtype), value.cpu()), name


def test_milestone_setting_is_validated(tmp_path):
    cfg = _config(tmp_path, milestone_every_steps=10)
    assert cfg.milestone_every_steps == 10
    try:
        _config(tmp_path, milestone_every_steps=0)
    except Exception as exc:  # pydantic ValidationError
        assert "milestone_every_steps" in str(exc)
    else:
        raise AssertionError("milestone_every_steps = 0 must be rejected")
