"""The LLM (policy) trainer's optimizer over bfloat16 parameters (arm-redesign Phase 3).

Documents a situation; fixes nothing. The policy trainer builds plain AdamW
(trainer/optim.py, type "adamw") over parameters created in
model.optimization_dtype, which every thesis ARM config sets to bfloat16. The
optimizer state then lives in bfloat16 too, and the second-moment decay
(beta2 = 0.999, a 0.1 % change per step) is below bfloat16's resolution: it
rounds away, so a gradient spike lowers that weight's learning rate for good.
The same fault was measured on the value networks (job 755044) and fixed there
by MasterWeightAdamW; see tests/unit/advantage_trainer/test_master_weight_adamw.py.

Whether real policy checkpoints show it is measured separately (thesis script
measure_policy_optimizer_state.py); this file is the CPU check of the dtype
and of the arithmetic.
"""

import pytest
import torch
from torch import nn

from prime_rl.configs.trainer import AdamWConfig
from prime_rl.trainer.optim import _create_optimizer

BF16 = torch.bfloat16


def _policy_optimizer(param: nn.Parameter) -> torch.optim.Optimizer:
    """The optimizer the policy trainer builds for a LoRA parameter: the default
    [trainer.optim] (AdamW, betas 0.9 / 0.999). The "adamw" branch never reads
    parallel_dims."""
    return _create_optimizer(AdamWConfig(), [("lora_A", param)], parallel_dims=None)


def _spike_then_quiet(dtype: torch.dtype, quiet_steps: int):
    """One step with gradient 100, then quiet steps with gradient 0.01. Returns
    the optimizer, the parameter, and the second moment after the spike, after
    one quiet step and after all of them (as float32 copies)."""
    p = nn.Parameter(torch.full((4,), 0.05, dtype=torch.float32).to(dtype))
    opt = _policy_optimizer(p)
    p.grad = torch.full_like(p, 100.0)
    opt.step()
    v_spike = opt.state[p]["exp_avg_sq"].float().clone()
    v_one = None
    for i in range(quiet_steps):
        p.grad = torch.full_like(p, 0.01)
        opt.step()
        if i == 0:
            v_one = opt.state[p]["exp_avg_sq"].float().clone()
    return opt, p, v_spike, v_one, opt.state[p]["exp_avg_sq"].float().clone()


def test_policy_adamw_state_is_bfloat16_and_does_not_decay():
    try:
        opt, p, v_spike, v_one, v_later = _spike_then_quiet(BF16, quiet_steps=700)
    except RuntimeError as exc:  # a torch build without bfloat16 CPU kernels for AdamW
        pytest.skip(f"bfloat16 AdamW not runnable on this CPU build: {exc}")
    assert type(opt) is torch.optim.AdamW
    assert opt.defaults["betas"] == (0.9, 0.999)
    state = opt.state[p]
    assert p.dtype == BF16
    assert state["exp_avg"].dtype == BF16 and state["exp_avg_sq"].dtype == BF16  # the state follows the parameter
    assert bool((v_spike > 0).all())
    assert torch.equal(v_one, v_spike)  # one quiet step: the 0.1 % decay rounds away
    assert torch.equal(v_later, v_spike)  # 700 of them (one half-life of beta2): not one bit of decay


def test_policy_adamw_decays_in_float32():
    """The control: the same optimizer over a float32 parameter forgets the spike."""
    opt, p, v_spike, v_one, v_later = _spike_then_quiet(torch.float32, quiet_steps=700)
    assert opt.state[p]["exp_avg_sq"].dtype == torch.float32
    assert torch.allclose(v_one, 0.999 * v_spike, rtol=1e-5)  # the quiet gradient adds 1e-7 to a value of 10
    assert bool((v_one < v_spike).all())
    assert bool((v_later < 0.6 * v_spike).all()) and bool((v_later > 0.3 * v_spike).all())
