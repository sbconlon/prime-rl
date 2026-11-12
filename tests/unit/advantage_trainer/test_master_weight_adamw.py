"""MasterWeightAdamW: float32 master weights and state for low-precision params.

The bug it fixes, measured on job 755044: with bfloat16 parameters AdamW's
second moment never decreased (0 of 18.5M entries over 100 steps), because the
beta2 = 0.999 decay is below bfloat16's resolution.
"""

import pytest
import torch

from prime_rl.advantage_trainer.shared import MasterWeightAdamW

BF16 = torch.bfloat16


def _param(values, dtype=BF16):
    return torch.nn.Parameter(torch.tensor(values, dtype=torch.float32).to(dtype))


def _v(opt: MasterWeightAdamW, group: int = 0, idx: int = 0) -> torch.Tensor:
    master = opt.param_groups[group]["params"][idx]
    return opt.state_dict()["state"][0 if (group, idx) == (0, 0) else idx]["exp_avg_sq"].clone() \
        if False else opt._opt.state[master]["exp_avg_sq"].clone()


def test_state_and_masters_are_float32_and_params_follow():
    p = _param([0.05, -0.02, 0.5])
    opt = MasterWeightAdamW([{"params": [p], "lr": 1e-3}])
    p.grad = torch.ones_like(p)
    norms = opt.step()
    assert norms == [pytest.approx(3 ** 0.5, rel=1e-3)]
    master = opt.param_groups[0]["params"][0]
    assert master.dtype == torch.float32 and p.dtype == BF16
    state = opt._opt.state[master]
    assert state["exp_avg"].dtype == torch.float32 and state["exp_avg_sq"].dtype == torch.float32
    assert torch.equal(p.detach(), master.detach().to(BF16))  # the model sees the rounded master


def test_second_moment_decays_after_a_spike():
    p = _param([0.05] * 4)
    opt = MasterWeightAdamW([{"params": [p], "lr": 1e-4}])
    p.grad = torch.full_like(p, 100.0)  # the spike
    opt.step()
    v_spike = _v(opt)
    for _ in range(700):  # about one half-life of beta2 = 0.999
        p.grad = torch.full_like(p, 0.01)
        opt.step()
    v_later = _v(opt)
    assert torch.all(v_later < 0.6 * v_spike)  # it forgets the spike
    assert torch.all(v_later > 0.3 * v_spike)


def test_plain_bfloat16_adamw_cannot_forget_a_spike():
    """Documents the failure mode the wrapper exists for."""
    p = _param([0.05] * 4)
    try:
        opt = torch.optim.AdamW([p], lr=1e-4)
        p.grad = torch.full_like(p, 100.0)
        opt.step()
        v_spike = opt.state[p]["exp_avg_sq"].float().clone()
        for _ in range(700):
            p.grad = torch.full_like(p, 0.01)
            opt.step()
        v_later = opt.state[p]["exp_avg_sq"].float()
    except RuntimeError as exc:  # a torch build without bfloat16 CPU kernels for AdamW
        pytest.skip(f"bfloat16 AdamW not runnable on this CPU build: {exc}")
    assert opt.state[p]["exp_avg_sq"].dtype == BF16
    assert torch.equal(v_later, v_spike)  # not one bit of decay in 700 steps


def test_updates_below_bfloat16_resolution_accumulate():
    # |w| = 0.5: bfloat16 spacing is 2^-8 ~ 0.0039, an Adam step of 1e-4 is 2.5 % of it
    p = _param([0.5])
    opt = MasterWeightAdamW([{"params": [p], "lr": 1e-4}])
    start = p.detach().float().clone()
    for _ in range(100):
        p.grad = torch.ones_like(p)
        opt.step()
    master = opt.param_groups[0]["params"][0]
    assert master.item() == pytest.approx(0.5 - 100 * 1e-4, abs=2e-4)
    assert p.detach().float().item() < start.item()  # the rounded weight moved

    # the same updates applied directly in bfloat16 never move the weight
    q = torch.tensor([0.5]).to(BF16)
    for _ in range(100):
        q = q - torch.tensor([1e-4]).to(BF16)
    assert q.float().item() == 0.5


def test_per_group_clipping_and_norms():
    a, b = _param([0.0, 0.0]), _param([0.0])
    opt = MasterWeightAdamW([{"params": [a], "lr": 1e-3}, {"params": [b], "lr": 1e-3}])
    a.grad = torch.tensor([3.0, 4.0]).to(BF16)
    b.grad = torch.tensor([0.5]).to(BF16)
    norms = opt.step(max_grad_norm=1.0)
    assert norms == [pytest.approx(5.0), pytest.approx(0.5)]
    ga = opt.param_groups[0]["params"][0].grad
    gb = opt.param_groups[1]["params"][0].grad
    assert ga.norm().item() == pytest.approx(1.0, rel=1e-4)  # clipped
    assert gb.norm().item() == pytest.approx(0.5, rel=1e-4)  # untouched: its own budget


def test_parameters_without_gradient_are_never_overwritten():
    trained, target = _param([0.1]), _param([0.1])
    opt = MasterWeightAdamW([{"params": [trained], "lr": 1e-2}, {"params": [target], "lr": 0.0}])
    with torch.no_grad():
        target.copy_(torch.tensor([0.75]).to(BF16))  # a Polyak update on the model
    trained.grad = torch.ones_like(trained)
    norms = opt.step()
    assert norms[1] == 0.0
    assert target.detach().float().item() == 0.75  # not reset to the stale master
    assert trained.detach().float().item() < 0.1


def test_zero_rate_group_with_zero_gradients_is_never_overwritten():
    """The real backbone: V_target gets zero-valued (not None) gradients."""
    trained, target = _param([0.1]), _param([0.1])
    opt = MasterWeightAdamW([{"params": [trained], "lr": 1e-2}, {"params": [target], "lr": 0.0}])
    for value in (0.75, 0.25):
        with torch.no_grad():
            target.copy_(torch.tensor([value]).to(BF16))  # Polyak update on the model
        trained.grad = torch.ones_like(trained)
        target.grad = torch.zeros_like(target)
        norms = opt.step(max_grad_norm=1.0)
        assert norms[1] == 0.0
        assert target.detach().float().item() == value


def test_sync_from_params_and_zero_grad():
    p = _param([0.1, 0.2])
    opt = MasterWeightAdamW([{"params": [p], "lr": 1e-3}])
    with torch.no_grad():
        p.copy_(torch.tensor([0.5, -0.5]).to(BF16))  # weights loaded from a checkpoint
    opt.sync_from_params()
    assert torch.equal(opt.param_groups[0]["params"][0].detach(), torch.tensor([0.5, -0.5]))
    p.grad = torch.ones_like(p)
    opt.zero_grad()
    assert p.grad is None


def test_state_dict_roundtrip_and_loading_bfloat16_state():
    p = _param([0.05, -0.02])
    opt = MasterWeightAdamW([{"params": [p], "lr": 1e-3}])
    for _ in range(3):
        p.grad = torch.ones_like(p)
        opt.step()
    sd = opt.state_dict()

    q = _param([0.05, -0.02])
    opt2 = MasterWeightAdamW([{"params": [q], "lr": 1e-3}])
    opt2.load_state_dict(sd)
    m2 = opt2.param_groups[0]["params"][0]
    assert torch.equal(opt2._opt.state[m2]["exp_avg_sq"], opt._opt.state[opt.param_groups[0]["params"][0]]["exp_avg_sq"])
    assert opt2._opt.state[m2]["step"] == 3

    # a checkpoint written by the old bfloat16 optimizer: state tensors are cast to float32
    old = {"state": {0: {"step": torch.tensor(5.0), "exp_avg": torch.ones(2).to(BF16),
                         "exp_avg_sq": torch.full((2,), 4.0).to(BF16)}},
           "param_groups": sd["param_groups"]}
    opt3 = MasterWeightAdamW([{"params": [_param([0.05, -0.02])], "lr": 1e-3}])
    opt3.load_state_dict(old)
    m3 = opt3.param_groups[0]["params"][0]
    assert opt3._opt.state[m3]["exp_avg_sq"].dtype == torch.float32
    assert torch.equal(opt3._opt.state[m3]["exp_avg_sq"], torch.full((2,), 4.0))
