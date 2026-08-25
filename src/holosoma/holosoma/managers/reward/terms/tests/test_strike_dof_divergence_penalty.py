"""Unit tests for ``StrikeDofDivergencePenalty`` (managers/reward/terms/wbt.py) -- the LINEAR,
NON-SATURATING per-joint strike-phase divergence price added 2026-08-24.

It exists because its Gaussian sibling ``MotionStrikeDofPosErrorExp`` gives up on exactly the
joints that have drifted furthest (measured gradient 0.06-0.21 on 54-64 deg joints vs ~1.7 on a
typical 20 deg one). The tests below pin the three properties that make this term able to do the
job the Gaussian cannot, plus the deadband/no-op guarantees:

  1. gradient stays CONSTANT however far a joint drifts (the Gaussian's collapses)
  2. below ``threshold`` it is exactly zero -- it composes with, rather than re-prices, the regime
     the Gaussian already handles well
  3. one blown-up joint is never damped by well-tracked neighbours
  4. strike-gated, and covers the sagittal kick chain the other two strike terms omit

``_get_motion_command_and_assert_type`` is patched out, same isolation discipline as
``test_motion_strike_dof_pos_error_exp.py``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import holosoma.managers.reward.terms.wbt as wbt

_DOF = ["left_hip_pitch_joint", "left_knee_joint", "waist_yaw_joint", "left_elbow_joint"]


class _FakeCfg:
    def __init__(self, **params):
        self.params = params
        self.weight = -1.0


class _FakeEnv:
    def __init__(self, dof_names=_DOF, device="cpu"):
        self.device = device
        self.simulator = SimpleNamespace(dof_names=list(dof_names))
        self.command_manager = SimpleNamespace(get_state=lambda name: None)


def _make(threshold=0.35, dof_names=None, env=None):
    env = env or _FakeEnv()
    return wbt.StrikeDofDivergencePenalty(_FakeCfg(threshold=threshold, dof_names=dof_names), env), env


def _mc(ref, actual, in_strike):
    return SimpleNamespace(joint_pos=ref, robot_joint_pos=actual, in_strike_phase=in_strike)


def _call(term, env, ref, actual, in_strike):
    with patch.object(wbt, "_get_motion_command_and_assert_type", return_value=_mc(ref, actual, in_strike)):
        return term(env)


# ------------------------------------------------------------------ deadband
def test_error_below_threshold_is_exactly_zero():
    """Composes with the Gaussian instead of double-pricing the regime it already handles."""
    term, env = _make(threshold=0.35)
    ref = torch.zeros(1, 4)
    actual = torch.full((1, 4), 0.30)  # every joint inside the deadband
    out = _call(term, env, ref, actual, torch.tensor([True]))
    assert torch.allclose(out, torch.zeros(1))


def test_exact_match_is_zero():
    term, env = _make()
    q = torch.zeros(2, 4)
    assert torch.allclose(_call(term, env, q.clone(), q.clone(), torch.tensor([True, True])), torch.zeros(2))


def test_only_the_excess_over_threshold_is_priced():
    term, env = _make(threshold=0.35)
    ref = torch.zeros(1, 4)
    actual = torch.tensor([[0.95, 0.0, 0.0, 0.0]])  # one joint 0.60 rad past the deadband
    out = _call(term, env, ref, actual, torch.tensor([True]))
    assert torch.allclose(out, torch.tensor([0.60 / 4]), atol=1e-6)


def test_sign_is_symmetric_in_error_direction():
    term, env = _make(threshold=0.35)
    ref = torch.zeros(1, 4)
    pos = _call(term, env, ref, torch.tensor([[0.95, 0.0, 0.0, 0.0]]), torch.tensor([True]))
    neg = _call(term, env, ref, torch.tensor([[-0.95, 0.0, 0.0, 0.0]]), torch.tensor([True]))
    assert torch.allclose(pos, neg)


# ------------------------------------------------- the property the Gaussian lacks
def test_gradient_is_constant_however_far_the_joint_drifts():
    """THE point of this term. The Gaussian's gradient collapses past ~2 sigma; this one does not,
    so a badly-diverged joint keeps getting pulled back."""
    term, env = _make(threshold=0.35)
    grads = []
    for offset in (0.5, 1.0, 2.0, 5.0):
        actual = torch.tensor([[offset, 0.0, 0.0, 0.0]], requires_grad=True)
        out = _call(term, env, torch.zeros(1, 4), actual, torch.tensor([True]))
        out.sum().backward()
        grads.append(actual.grad[0, 0].item())
    assert all(abs(g - grads[0]) < 1e-6 for g in grads), f"gradient not constant: {grads}"
    assert grads[0] > 0


def test_gaussian_sibling_collapses_where_this_term_does_not():
    """Regression guard on the motivating comparison, computed explicitly on the same input."""
    sigma = 0.5
    near, far = 0.35, 1.12  # ~20 deg (typical) and ~64 deg (skill012 left_elbow)
    g_near = (2 * near / sigma**2) * torch.exp(torch.tensor(-(near**2) / sigma**2))
    g_far = (2 * far / sigma**2) * torch.exp(torch.tensor(-(far**2) / sigma**2))
    assert g_far < 0.1 * g_near, "Gaussian should collapse on the far-diverged joint"
    term, env = _make(threshold=0.35)
    lin = []
    for off in (near, far):
        a = torch.tensor([[off, 0.0, 0.0, 0.0]], requires_grad=True)
        _call(term, env, torch.zeros(1, 4), a, torch.tensor([True])).sum().backward()
        lin.append(a.grad[0, 0].item())
    assert abs(lin[1] - lin[0]) < 1e-6, "linear term must NOT collapse on the far joint"


def test_one_blown_up_joint_is_not_damped_by_good_neighbours():
    term, env = _make(threshold=0.35)
    ref = torch.zeros(2, 4)
    lonely = torch.tensor([[1.35, 0.0, 0.0, 0.0]])          # one joint 1.0 rad past deadband
    company = torch.tensor([[1.35, 0.30, 0.30, 0.30]])      # same joint, neighbours inside deadband
    a = _call(term, env, ref[:1], lonely, torch.tensor([True]))
    b = _call(term, env, ref[:1], company, torch.tensor([True]))
    assert torch.allclose(a, b), "well-tracked neighbours must not dilute the offender"


# ------------------------------------------------------------------ gating / masking
def test_zero_outside_strike_phase():
    term, env = _make(threshold=0.35)
    ref = torch.zeros(2, 4)
    actual = torch.full((2, 4), 1.5)
    out = _call(term, env, ref, actual, torch.tensor([True, False]))
    assert out[0] > 0 and out[1] == 0.0


def test_default_dof_names_covers_every_joint_including_the_sagittal_chain():
    """The coverage gap this term closes: the two exp strike terms omit hip_pitch/knee/ankle_pitch
    on both legs between them. dof_names=None must price all of them."""
    term, env = _make(threshold=0.0, dof_names=None)
    assert term.dof_indexes is None
    ref = torch.zeros(1, 4)
    for j in range(4):
        actual = torch.zeros(1, 4)
        actual[0, j] = 1.0
        out = _call(term, env, ref, actual, torch.tensor([True]))
        assert out.item() > 0, f"joint index {j} ({_DOF[j]}) was not priced"


def test_dof_subset_ignores_unlisted_joints():
    term, env = _make(threshold=0.0, dof_names=["waist_yaw_joint"])
    ref = torch.zeros(1, 4)
    off_target = torch.tensor([[3.0, 0.0, 0.0, 0.0]])  # left_hip_pitch, not in the subset
    assert _call(term, env, ref, off_target, torch.tensor([True])).item() == 0.0
    on_target = torch.tensor([[0.0, 0.0, 3.0, 0.0]])   # waist_yaw
    assert _call(term, env, ref, on_target, torch.tensor([True])).item() > 0


# ------------------------------------------------------------------ construction
def test_bad_joint_name_raises():
    with pytest.raises(AssertionError):
        _make(dof_names=["not_a_real_joint"])


def test_negative_threshold_rejected():
    with pytest.raises(AssertionError):
        _make(threshold=-0.1)


def test_reset_is_a_noop_and_callable():
    term, _ = _make()
    term.reset(torch.tensor([0]))
