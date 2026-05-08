"""Phase 7a -- value_regression_loss_fn correctness tests (CPU).

The loss is pure tensor arithmetic, no model forward needed. These tests
pin the closed-form behavior:
  - Predictions matching targets -> loss is zero.
  - General case -> loss matches hand-computed MSE on masked positions.
  - PPO -> Q+ contribution is zero.
  - ARM -> both contributions present.
  - Loss masking excludes prompt and bridge positions.
  - Combined-backward-equals-separate-backwards (the math claim from Phase 7 DQ4).
"""

from __future__ import annotations

import torch

from prime_rl.advantage_trainer.loss import (
    ValueLossInputs,
    ValueLossOutputs,
    value_regression_loss_fn,
)


# ---------------------------------------------------------------------------
# Closed-form correctness
# ---------------------------------------------------------------------------


def test_loss_zero_when_predictions_match_targets():
    """V_pred == V_target and Q+_pred == Q+_target -> loss = 0."""
    seq = 5
    v = torch.zeros(seq)
    q = torch.zeros(seq)
    out = value_regression_loss_fn(
        ValueLossInputs(
            v_predictions=v.clone(),
            v_targets=v.clone(),
            q_plus_predictions=q.clone(),
            q_plus_targets=q.clone(),
            loss_mask=torch.ones(seq, dtype=torch.bool),
            algorithm="arm",
        )
    )
    assert isinstance(out, ValueLossOutputs)
    assert out.loss.item() == 0.0
    assert out.metrics["l_v"].item() == 0.0
    assert out.metrics["l_q"].item() == 0.0


def test_loss_matches_hand_computed_mse_arm():
    """ARM with known prediction/target arrays -> loss equals the hand-computed
    MSE summed across V and Q+ at masked positions."""
    v_pred = torch.tensor([0.1, 0.2, 0.3, 0.4])
    v_targets = torch.tensor([0.5, 0.5, 0.5, 0.5])
    q_pred = torch.tensor([1.0, 1.5, 2.0, 2.5])
    q_targets = torch.tensor([1.0, 1.0, 1.0, 1.0])
    mask = torch.tensor([False, True, True, True])  # exclude position 0

    out = value_regression_loss_fn(
        ValueLossInputs(
            v_predictions=v_pred,
            v_targets=v_targets,
            q_plus_predictions=q_pred,
            q_plus_targets=q_targets,
            loss_mask=mask,
            algorithm="arm",
        )
    )
    # Hand-computed:
    #   v_diff at masked positions (1,2,3) = [-0.3, -0.2, -0.1]
    #   l_v = mean((0.3^2 + 0.2^2 + 0.1^2) / 3) = (0.09 + 0.04 + 0.01) / 3 = 0.14/3
    expected_l_v = (0.09 + 0.04 + 0.01) / 3.0
    #   q_diff at masked positions (1,2,3) = [0.5, 1.0, 1.5]
    #   l_q = (0.25 + 1.0 + 2.25) / 3 = 3.5/3
    expected_l_q = (0.25 + 1.0 + 2.25) / 3.0
    expected_loss = expected_l_v + expected_l_q

    assert abs(out.metrics["l_v"].item() - expected_l_v) < 1e-7
    assert abs(out.metrics["l_q"].item() - expected_l_q) < 1e-7
    assert abs(out.loss.item() - expected_loss) < 1e-7


def test_ppo_loss_ignores_q_plus():
    """PPO: q_plus_predictions and q_plus_targets are None; only L_V contributes."""
    v_pred = torch.tensor([0.1, 0.2, 0.3])
    v_targets = torch.tensor([0.0, 0.0, 0.0])

    out = value_regression_loss_fn(
        ValueLossInputs(
            v_predictions=v_pred,
            v_targets=v_targets,
            q_plus_predictions=None,
            q_plus_targets=None,
            loss_mask=torch.ones(3, dtype=torch.bool),
            algorithm="ppo",
        )
    )
    # l_v = (0.01 + 0.04 + 0.09) / 3 = 0.14/3
    expected_l_v = (0.01 + 0.04 + 0.09) / 3.0
    assert abs(out.metrics["l_v"].item() - expected_l_v) < 1e-7
    # l_q == 0 for PPO
    assert out.metrics["l_q"].item() == 0.0
    assert abs(out.loss.item() - expected_l_v) < 1e-7


def test_arm_with_both_terms_nonzero():
    """ARM: both l_v and l_q are non-zero when neither matches its target."""
    v_pred = torch.tensor([0.5, 1.5])
    v_targets = torch.tensor([0.0, 0.0])
    q_pred = torch.tensor([0.5, 1.5])
    q_targets = torch.tensor([0.0, 0.0])
    out = value_regression_loss_fn(
        ValueLossInputs(
            v_predictions=v_pred,
            v_targets=v_targets,
            q_plus_predictions=q_pred,
            q_plus_targets=q_targets,
            loss_mask=torch.ones(2, dtype=torch.bool),
            algorithm="arm",
        )
    )
    assert out.metrics["l_v"].item() > 0
    assert out.metrics["l_q"].item() > 0
    assert abs(out.loss.item() - (out.metrics["l_v"].item() + out.metrics["l_q"].item())) < 1e-7


def test_loss_mask_excludes_prompt_and_bridge_positions():
    """Loss mask = [F, F, T, T, F] -> only positions 2 and 3 contribute."""
    v_pred = torch.tensor([10.0, 10.0, 0.5, 0.5, 10.0])  # positions 0,1,4 excluded
    v_targets = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
    mask = torch.tensor([False, False, True, True, False])
    out = value_regression_loss_fn(
        ValueLossInputs(
            v_predictions=v_pred,
            v_targets=v_targets,
            q_plus_predictions=None,
            q_plus_targets=None,
            loss_mask=mask,
            algorithm="ppo",
        )
    )
    # Excluded positions have huge errors but should be ignored.
    # Only positions 2 and 3 contribute: each has diff 0.5 -> squared 0.25.
    # mean(0.25, 0.25) = 0.25
    expected = 0.25
    assert abs(out.loss.item() - expected) < 1e-7


def test_loss_handles_empty_mask():
    """All-False mask -> loss is gradient-safe zero (doesn't NaN, doesn't crash)."""
    v_pred = torch.tensor([0.1, 0.2, 0.3])
    v_targets = torch.tensor([0.0, 0.0, 0.0])
    out = value_regression_loss_fn(
        ValueLossInputs(
            v_predictions=v_pred,
            v_targets=v_targets,
            q_plus_predictions=None,
            q_plus_targets=None,
            loss_mask=torch.zeros(3, dtype=torch.bool),
            algorithm="ppo",
        )
    )
    assert out.loss.item() == 0.0
    # Loss should still be a valid tensor with the autograd graph alive
    # (so backward through it would just produce zero grads, not crash).
    assert isinstance(out.loss, torch.Tensor)


# ---------------------------------------------------------------------------
# Math claim: combined-backward-equals-separate-backwards (DQ4)
# ---------------------------------------------------------------------------


def test_combined_backward_equals_separate_backwards():
    """Combined L = L_V + L_Q+ with one backward gives the same gradients as
    L_V.backward() then L_Q+.backward() separately, when V's and Q+'s
    parameter sets are disjoint.

    This codifies the math claim that justifies our single-backward design.
    """
    # Two disjoint parameter sets: theta_V (a 3-vector) and theta_Q (a 3-vector).
    seq = 4
    mask = torch.ones(seq, dtype=torch.bool)

    # Build predictions as a function of disjoint parameters.
    theta_v = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
    theta_q = torch.tensor([1.0, -0.5, 0.7], requires_grad=True)

    # v_predictions depends only on theta_v: simple linear projection of theta_v
    # to a [seq] tensor (stack 3 elements then pad).
    def make_v_preds(theta):
        # 4 outputs: theta[0], theta[1], theta[2], theta.sum()
        return torch.stack([theta[0], theta[1], theta[2], theta.sum()])

    def make_q_preds(theta):
        return torch.stack([theta[0] * 2, theta[1] * 0.5, theta[2] - 1.0, theta.sum() + 1.0])

    v_targets = torch.tensor([0.5, 0.5, 0.5, 0.5])
    q_targets = torch.tensor([0.0, 0.0, 0.0, 0.0])

    # ---- Case 1: combined backward
    v_pred = make_v_preds(theta_v)
    q_pred = make_q_preds(theta_q)
    out = value_regression_loss_fn(
        ValueLossInputs(
            v_predictions=v_pred,
            v_targets=v_targets,
            q_plus_predictions=q_pred,
            q_plus_targets=q_targets,
            loss_mask=mask,
            algorithm="arm",
        )
    )
    out.loss.backward()
    grad_combined_v = theta_v.grad.clone()
    grad_combined_q = theta_q.grad.clone()
    theta_v.grad.zero_()
    theta_q.grad.zero_()

    # ---- Case 2: separate backwards
    v_pred = make_v_preds(theta_v)
    q_pred = make_q_preds(theta_q)
    out = value_regression_loss_fn(
        ValueLossInputs(
            v_predictions=v_pred,
            v_targets=v_targets,
            q_plus_predictions=q_pred,
            q_plus_targets=q_targets,
            loss_mask=mask,
            algorithm="arm",
        )
    )
    # Backward through l_v alone, then through l_q alone.
    out.metrics["l_v"]  # detached; can't backward through this. We need to
    # recompute l_v and l_q from a fresh forward to get their graphs.
    v_pred = make_v_preds(theta_v)
    q_pred = make_q_preds(theta_q)
    v_diff = v_pred - v_targets
    q_diff = q_pred - q_targets
    l_v = (v_diff[mask] ** 2).mean()
    l_q = (q_diff[mask] ** 2).mean()
    l_v.backward(retain_graph=True)
    l_q.backward()
    grad_separate_v = theta_v.grad.clone()
    grad_separate_q = theta_q.grad.clone()

    # Combined and separate should produce bit-identical gradients (within
    # float tolerance from the order of ops).
    assert torch.allclose(grad_combined_v, grad_separate_v, atol=1e-7)
    assert torch.allclose(grad_combined_q, grad_separate_q, atol=1e-7)
