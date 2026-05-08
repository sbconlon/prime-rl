"""Phase 8 -- LLM-Trainer loss verification with per-token advantages.

These tests confirm that ``default_loss_fn`` (the shared IPO loss for GRPO,
PPO, and ARM) consumes the per-token ``advantages`` tensor correctly,
without algorithm-specific branching, and produces finite values + sensible
gradients across the regimes the three algorithms produce:

  - all-zero advantages (cold-start ARM/PPO with zero-init value networks)
  - uniform advantages (GRPO\'s scalar broadcast after Phase 1\'s contract
    evolution)
  - varying advantages (post-training PPO GAE / ARM regret matching)

CPU-only (no ``pytest.mark.gpu``). Tiny tensors so the laptop budget stays
well under the per-test second-budget guideline.

See plan/phases/phase-08-llm-loss-verification.md for the full design.
"""

from __future__ import annotations

import torch

from prime_rl.configs.trainer import DefaultLossConfig
from prime_rl.trainer.rl.loss import LossInputs, default_loss_fn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_inputs(
    *,
    seq_len: int = 8,
    advantages: torch.Tensor | None = None,
    trainer_logprobs: torch.Tensor | None = None,
    inference_logprobs: torch.Tensor | None = None,
    teacher_logprobs: torch.Tensor | None = None,
    loss_mask: torch.Tensor | None = None,
    requires_grad: bool = False,
) -> LossInputs:
    """Build a single-sample LossInputs with sensible defaults.

    Defaults: small non-zero logprob differences (so the IPO importance
    ratio + KL terms are non-trivial), full loss_mask=True, no teacher.
    Override individual fields to exercise specific regimes.
    """
    if trainer_logprobs is None:
        trainer_logprobs = torch.full((seq_len,), -0.10, dtype=torch.float32)
    if inference_logprobs is None:
        inference_logprobs = torch.full((seq_len,), -0.15, dtype=torch.float32)
    if advantages is None:
        advantages = torch.zeros(seq_len, dtype=torch.float32)
    if loss_mask is None:
        loss_mask = torch.ones(seq_len, dtype=torch.bool)

    if requires_grad:
        trainer_logprobs = trainer_logprobs.clone().detach().requires_grad_(True)
        # advantages don\'t need grad themselves -- the gradient we care
        # about flows back to trainer_logprobs.

    return LossInputs(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        teacher_logprobs=teacher_logprobs,
        advantages=advantages,
        loss_mask=loss_mask,
    )


def _default_loss_config() -> DefaultLossConfig:
    """A loss config that mirrors the production GRPO baseline\'s shape:
    nonzero ipo_mask thresholds + adv_tau=1 + a small kl_tau."""
    return DefaultLossConfig(
        ipo_mask_high=10.0,
        ipo_mask_low=10.0,
    )


# ---------------------------------------------------------------------------
# Loss correctness with per-token advantages (Layer 1 -- §10)
# ---------------------------------------------------------------------------


def test_loss_finite_with_zero_advantages():
    """Cold-start ARM/PPO regime: advantages are all zero. The pg_loss term
    contributes nothing; only the KL term has weight. The loss must still be
    finite and non-NaN -- in particular, no divide-by-zero or implicit log of
    zero anywhere along the path."""
    seq_len = 8
    inputs = _make_inputs(
        seq_len=seq_len,
        advantages=torch.zeros(seq_len, dtype=torch.float32),
    )
    out = default_loss_fn(inputs, _default_loss_config())

    assert torch.isfinite(out.loss).all(), f"loss not finite: {out.loss}"
    assert not torch.isnan(out.loss), f"loss is NaN: {out.loss}"
    # All metrics must also be finite (empty-mask metrics fall back to 0 via _safe_mean).
    for name, value in out.metrics.items():
        assert torch.isfinite(value).all(), f"metric {name} not finite: {value}"


def test_loss_finite_with_uniform_advantages():
    """GRPO scalar-broadcast regime after Phase 1\'s contract evolution:
    advantages = [c, c, ..., c]. This is what ``prepare_sample`` produces
    when given a TrainingSample whose advantages were computed by GRPO.
    The loss must be finite. It must also equal the loss computed when the
    same scalar c is broadcast manually -- a sanity-check that the per-token
    code path is broadcast-equivalent for the uniform case (the ``pg_loss``
    expression ``keep_mask * advantages * importance_ratio`` is element-wise
    multiplication, so the uniform case must reduce to the scalar case)."""
    seq_len = 8
    c = 0.5
    uniform_advantages = torch.full((seq_len,), c, dtype=torch.float32)
    scalar_broadcast_advantages = torch.zeros(seq_len, dtype=torch.float32) + c

    out_uniform = default_loss_fn(
        _make_inputs(seq_len=seq_len, advantages=uniform_advantages),
        _default_loss_config(),
    )
    out_broadcast = default_loss_fn(
        _make_inputs(seq_len=seq_len, advantages=scalar_broadcast_advantages),
        _default_loss_config(),
    )

    assert torch.isfinite(out_uniform.loss).all()
    assert torch.allclose(out_uniform.loss, out_broadcast.loss, atol=0, rtol=0), (
        "uniform per-token and manual scalar-broadcast must produce bit-equal "
        "loss (same advantages tensor, same code path)"
    )


def test_loss_finite_with_varying_advantages():
    """Post-training PPO/ARM regime: advantages vary across positions. Both
    loss and gradients (w.r.t. trainer_logprobs) must be finite. This is the
    most common ARM/PPO production case and the one Phase 1\'s contract
    change was made to support."""
    seq_len = 8
    torch.manual_seed(0)
    varying_advantages = torch.randn(seq_len, dtype=torch.float32)

    inputs = _make_inputs(
        seq_len=seq_len,
        advantages=varying_advantages,
        requires_grad=True,
    )
    out = default_loss_fn(inputs, _default_loss_config())

    assert torch.isfinite(out.loss).all()
    out.loss.backward()
    assert inputs.trainer_logprobs.grad is not None
    assert torch.isfinite(inputs.trainer_logprobs.grad).all(), (
        "gradient w.r.t. trainer_logprobs must be finite for varying advantages"
    )


def test_loss_gradient_proportional_to_advantage_magnitude():
    """The pg_loss term scales linearly with advantages; the KL term doesn\'t.
    Setting kl_tau=0 isolates the pg_loss path so we can verify advantages
    flow through to the gradient with the expected linearity.

    Mechanically: pg_loss = keep_mask * (adv_tau * advantages) * importance_ratio.
    With kl_tau=0 + no teacher: loss = -pg_loss.sum(). doubling advantages
    must double the gradient w.r.t. trainer_logprobs (within float tol).
    """
    seq_len = 8
    base_adv = torch.full((seq_len,), 0.5, dtype=torch.float32)

    cfg = DefaultLossConfig(
        ipo_mask_high=10.0,
        ipo_mask_low=10.0,
        kl_tau=0.0,  # isolate pg_loss
    )

    def grad_for(advantages: torch.Tensor) -> torch.Tensor:
        inputs = _make_inputs(
            seq_len=seq_len, advantages=advantages, requires_grad=True
        )
        out = default_loss_fn(inputs, cfg)
        out.loss.backward()
        assert inputs.trainer_logprobs.grad is not None
        return inputs.trainer_logprobs.grad.detach().clone()

    grad_1x = grad_for(base_adv)
    grad_2x = grad_for(base_adv * 2.0)

    assert torch.isfinite(grad_1x).all()
    assert torch.isfinite(grad_2x).all()
    # The gradient must scale linearly with advantage magnitude.
    assert torch.allclose(grad_2x, 2.0 * grad_1x, atol=1e-5, rtol=1e-5), (
        f"gradient at 2x advantage should be 2x gradient at 1x; "
        f"max abs diff = {(grad_2x - 2.0 * grad_1x).abs().max().item()}"
    )
