"""Unit tests for ``MotionStrikeDofPosErrorExp`` (managers/reward/terms/wbt.py) -- the per-joint,
strike-phase-only DOF tracking term. Formula: ``mean_j(exp(-error_j^2/sigma^2))``, per-joint exp
THEN mean, deliberately NOT ``exp(-mean_j(error_j^2)/sigma^2)`` like the 6 Cartesian siblings above
it in ``wbt.py`` -- see the class's own docstring and the regression-guard test below.

``_get_motion_command_and_assert_type`` is patched out for every ``__call__`` test below
(isolating this term's own logic from that helper's ``isinstance(motion_command, MotionCommand)``
check, which a plain ``SimpleNamespace`` fake can never satisfy) -- same isolation discipline
``test_kick_scale_wrappers.py`` and ``test_shooting_strike_gate.py`` already use for their own
out-of-scope dependencies.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import holosoma.managers.reward.terms.wbt as wbt
from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg
from holosoma.managers.reward.manager import RewardManager

_DOF_NAMES = ["hip_pitch", "waist_yaw", "left_shoulder_pitch", "left_elbow", "right_wrist_roll"]
_MASK = ["waist_yaw", "left_shoulder_pitch", "left_elbow"]  # 3 of the 5, skips hip_pitch/right_wrist_roll


class _FakeCfg:
    def __init__(self, **params):
        self.params = params
        self.weight = 1.0


class _FakeEnv:
    def __init__(self, dof_names=_DOF_NAMES, device="cpu"):
        self.device = device
        self.simulator = SimpleNamespace(dof_names=list(dof_names))
        # unused directly -- _get_motion_command_and_assert_type is patched in every __call__ test
        self.command_manager = SimpleNamespace(get_state=lambda name: None)


def _make(dof_names=_MASK, sigma=0.5, env=None):
    env = env or _FakeEnv()
    return wbt.MotionStrikeDofPosErrorExp(_FakeCfg(dof_names=dof_names, sigma=sigma), env), env


def _fake_motion_command(joint_pos, robot_joint_pos, in_strike_phase):
    return SimpleNamespace(joint_pos=joint_pos, robot_joint_pos=robot_joint_pos, in_strike_phase=in_strike_phase)


# ---------------------------------------------------------------- construction
def test_bad_joint_name_raises():
    with pytest.raises(AssertionError):
        _make(dof_names=["not_a_real_joint"])


def test_reset_is_a_noop_and_callable():
    term, _ = _make()
    term.reset(torch.tensor([0]))  # must not raise; RewardTermBase.reset is abstract, must exist


# ---------------------------------------------------------------- formula + masking
def test_exact_match_gives_reward_one():
    term, env = _make()
    n, d = 2, len(_DOF_NAMES)
    q = torch.zeros(n, d)
    mc = _fake_motion_command(q.clone(), q.clone(), in_strike_phase=torch.tensor([True, True]))
    with (
        patch.object(wbt, "_get_motion_command_and_assert_type", return_value=mc),
        patch.object(wbt, "motion_tracking_reward_scale", return_value=torch.ones(n)),
    ):
        out = term(env)
    assert torch.allclose(out, torch.ones(n))


def test_formula_matches_exp_of_squared_error_at_concrete_offsets():
    """sigma=0.5, single masked joint offset by 0.3 rad -> exp(-0.09/0.25); by sigma itself
    (0.5 rad) -> exp(-1)."""
    term, env = _make(dof_names=["waist_yaw"], sigma=0.5)  # 1-joint mask, isolates the kernel
    d = len(_DOF_NAMES)
    ref = torch.zeros(2, d)
    actual = torch.zeros(2, d)
    waist_idx = _DOF_NAMES.index("waist_yaw")
    actual[0, waist_idx] = 0.3
    actual[1, waist_idx] = 0.5
    mc = _fake_motion_command(ref, actual, in_strike_phase=torch.tensor([True, True]))
    with (
        patch.object(wbt, "_get_motion_command_and_assert_type", return_value=mc),
        patch.object(wbt, "motion_tracking_reward_scale", return_value=torch.ones(2)),
    ):
        out = term(env)
    expected = torch.tensor([math.exp(-(0.3**2) / 0.25), math.exp(-1.0)])
    assert torch.allclose(out, expected, atol=1e-6)


def test_one_badly_diverged_joint_scores_higher_than_exp_of_mean_error_would():
    """The core regression guard. 3 masked joints, 2 exact, 1 badly off (1.2 rad). Computes BOTH
    formulas explicitly and asserts mean-after-exp > exp-of-mean-error -- guaranteed by Jensen's
    inequality (exp is convex) whenever cross-joint error has nonzero variance, so this isn't an
    incidental numeric coincidence."""
    term, env = _make(dof_names=_MASK, sigma=0.5)
    d = len(_DOF_NAMES)
    ref = torch.zeros(1, d)
    actual = torch.zeros(1, d)
    actual[0, _DOF_NAMES.index("left_elbow")] = 1.2  # 1 of the 3 masked joints blows up
    mc = _fake_motion_command(ref, actual, in_strike_phase=torch.tensor([True]))
    with (
        patch.object(wbt, "_get_motion_command_and_assert_type", return_value=mc),
        patch.object(wbt, "motion_tracking_reward_scale", return_value=torch.ones(1)),
    ):
        out = term(env)

    errors_sq = torch.tensor([0.0, 0.0, 1.2**2])
    mean_after_exp = torch.exp(-errors_sq / 0.5**2).mean()
    exp_of_mean_error = torch.exp(-errors_sq.mean() / 0.5**2)
    assert mean_after_exp.item() > exp_of_mean_error.item()  # sanity: the two formulas do differ
    assert torch.isclose(out[0], mean_after_exp, atol=1e-6)
    assert out[0] > exp_of_mean_error + 0.3, f"got {out[0]}, exp-of-mean would be {exp_of_mean_error}"


def test_unmasked_joints_are_ignored_even_if_wildly_diverged():
    term, env = _make(dof_names=_MASK, sigma=0.5)
    d = len(_DOF_NAMES)
    ref = torch.zeros(1, d)
    actual = torch.zeros(1, d)
    actual[0, _DOF_NAMES.index("hip_pitch")] = 3.0  # outside the mask, huge
    actual[0, _DOF_NAMES.index("right_wrist_roll")] = -2.5  # outside the mask, huge
    mc = _fake_motion_command(ref, actual, in_strike_phase=torch.tensor([True]))
    with (
        patch.object(wbt, "_get_motion_command_and_assert_type", return_value=mc),
        patch.object(wbt, "motion_tracking_reward_scale", return_value=torch.ones(1)),
    ):
        out = term(env)
    assert torch.allclose(out, torch.ones(1)), "unmasked joints must not affect the reward at all"


# ---------------------------------------------------------------- phase gate
def test_zero_outside_strike_phase():
    term, env = _make()
    d = len(_DOF_NAMES)
    actual = torch.ones(2, d)  # nonzero error -- would be nonzero reward if not gated
    mc = _fake_motion_command(torch.zeros(2, d), actual, in_strike_phase=torch.tensor([False, False]))
    with (
        patch.object(wbt, "_get_motion_command_and_assert_type", return_value=mc),
        patch.object(wbt, "motion_tracking_reward_scale", return_value=torch.ones(2)),
    ):
        out = term(env)
    assert torch.allclose(out, torch.zeros(2))


def test_mixed_phase_gated_per_env():
    term, env = _make()
    d = len(_DOF_NAMES)
    mc = _fake_motion_command(torch.zeros(2, d), torch.zeros(2, d), in_strike_phase=torch.tensor([True, False]))
    with (
        patch.object(wbt, "_get_motion_command_and_assert_type", return_value=mc),
        patch.object(wbt, "motion_tracking_reward_scale", return_value=torch.ones(2)),
    ):
        out = term(env)
    assert out[0].item() == 1.0
    assert out[1].item() == 0.0


# ---------------------------------------------------------------- scale composition
def test_motion_tracking_reward_scale_multiplies_in():
    term, env = _make()
    d = len(_DOF_NAMES)
    mc = _fake_motion_command(torch.zeros(1, d), torch.zeros(1, d), in_strike_phase=torch.tensor([True]))
    with (
        patch.object(wbt, "_get_motion_command_and_assert_type", return_value=mc),
        patch.object(wbt, "motion_tracking_reward_scale", return_value=torch.tensor([0.4])) as mock_scale,
    ):
        out = term(env)
    mock_scale.assert_called_once_with(env)
    assert torch.allclose(out, torch.tensor([0.4]))


# ---------------------------------------------------------------- true no-op guard
def test_weight_zero_is_never_instantiated_by_reward_manager():
    """Pins RewardManager's own weight==0.0 skip (managers/reward/manager.py:60-62), the
    guarantee this term's entire "ships as a true no-op" design leans on: at weight=0.0 the class
    must never be instantiated (so a bad dof_names entry, or any other constructor-time cost,
    literally cannot fire), never appear in _term_names, and never get an episode-sum logging key.
    A minimal env (no simulator.dof_names needed) proves this -- if the manager DID try to
    construct the term, this test would fail with an AttributeError instead of passing cleanly."""
    cfg = RewardManagerCfg(
        terms={
            "motion_strike_dof_pos_error_exp": RewardTermCfg(
                func="holosoma.managers.reward.terms.wbt:MotionStrikeDofPosErrorExp",
                params={"sigma": 0.5, "dof_names": ["waist_yaw_joint"]},
                weight=0.0,
                task_mode="kick",
            )
        }
    )
    env = SimpleNamespace(num_envs=4, logger=None)  # deliberately NO .simulator -- would AttributeError if touched
    manager = RewardManager(cfg, env, device="cpu")
    assert "motion_strike_dof_pos_error_exp" not in manager._term_names
    assert "motion_strike_dof_pos_error_exp" not in manager._term_instances
    assert "motion_strike_dof_pos_error_exp" not in manager._episode_sums
