"""Unit tests for ``motion_global_feet_lin_vel`` (managers/reward/terms/wbt.py) -- the 7th
motion-tracking term, 2026-08-05, ported from RoboNaldo (arXiv:2606.11092)'s
``motion_global_feet_linear_velocity_error_exp``. Verified against the real RoboNaldo source to be
formula-identical to ``motion_global_body_lin_vel`` (this project's existing port of RoboNaldo's
``motion_global_body_linear_velocity_error_exp``), restricted to a feet-only body subset -- these
tests focus on the part that's NEW relative to that sibling: index resolution against
``motion_cfg.body_names_to_track`` (a different index space than ``env.simulator.body_names``,
which every OTHER body-indexed term in this file resolves against instead).

``_get_motion_command_and_assert_type`` is patched out for every test (same isolation discipline
``test_motion_strike_dof_pos_error_exp.py`` already uses), since a plain ``SimpleNamespace`` fake
can never satisfy that helper's ``isinstance(motion_command, MotionCommand)`` check.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import patch

import torch

import holosoma.managers.reward.terms.wbt as wbt

_TRACKED_NAMES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
]
_LEFT_FOOT_IDX = _TRACKED_NAMES.index("left_ankle_roll_link")
_RIGHT_FOOT_IDX = _TRACKED_NAMES.index("right_ankle_roll_link")


def _fake_motion_command(body_lin_vel_w, robot_body_lin_vel_w, body_names_to_track=_TRACKED_NAMES):
    motion_cfg = SimpleNamespace(body_names_to_track=list(body_names_to_track))
    return SimpleNamespace(
        motion_cfg=motion_cfg,
        body_lin_vel_w=body_lin_vel_w,
        robot_body_lin_vel_w=robot_body_lin_vel_w,
    )


def test_exact_match_gives_reward_one():
    n, b = 2, len(_TRACKED_NAMES)
    v = torch.zeros(n, b, 3)
    mc = _fake_motion_command(v.clone(), v.clone())
    with patch.object(wbt, "_get_motion_command_and_assert_type", return_value=mc):
        out = wbt.motion_global_feet_lin_vel(SimpleNamespace(), sigma=1.0)
    assert torch.allclose(out, torch.ones(n))


def test_formula_matches_exp_of_squared_error_at_concrete_offset():
    n, b = 1, len(_TRACKED_NAMES)
    ref = torch.zeros(n, b, 3)
    actual = torch.zeros(n, b, 3)
    actual[0, _LEFT_FOOT_IDX, 0] = 0.5  # 0.5 m/s x-velocity error on the left foot only
    mc = _fake_motion_command(ref, actual)
    with patch.object(wbt, "_get_motion_command_and_assert_type", return_value=mc):
        out = wbt.motion_global_feet_lin_vel(SimpleNamespace(), sigma=1.0)
    # error = mean over 2 tracked feet of sum-of-squares: (0.5^2 + 0)/2 = 0.125
    expected = math.exp(-0.125 / 1.0**2)
    assert torch.isclose(out[0], torch.tensor(expected), atol=1e-6)


def test_only_feet_indexes_are_read_not_the_full_tracked_body_set():
    """A huge velocity error on a NON-foot tracked body (e.g. pelvis) must not affect the output
    at all -- confirms indexing is restricted to body_names, not the whole body_names_to_track
    tensor."""
    n, b = 1, len(_TRACKED_NAMES)
    ref = torch.zeros(n, b, 3)
    actual = torch.zeros(n, b, 3)
    actual[0, _TRACKED_NAMES.index("pelvis"), :] = 10.0
    actual[0, _TRACKED_NAMES.index("torso_link"), :] = 10.0
    mc = _fake_motion_command(ref, actual)
    with patch.object(wbt, "_get_motion_command_and_assert_type", return_value=mc):
        out = wbt.motion_global_feet_lin_vel(SimpleNamespace(), sigma=1.0)
    assert torch.allclose(out, torch.ones(n))


def test_custom_body_names_override_the_default():
    n, b = 1, len(_TRACKED_NAMES)
    ref = torch.zeros(n, b, 3)
    actual = torch.zeros(n, b, 3)
    actual[0, _TRACKED_NAMES.index("left_knee_link"), 0] = 1.0
    mc = _fake_motion_command(ref, actual)
    with patch.object(wbt, "_get_motion_command_and_assert_type", return_value=mc):
        default_out = wbt.motion_global_feet_lin_vel(SimpleNamespace(), sigma=1.0)
        custom_out = wbt.motion_global_feet_lin_vel(SimpleNamespace(), sigma=1.0, body_names=["left_knee_link"])
    assert default_out[0].item() == 1.0, "default (feet-only) must be unaffected by a knee-only error"
    assert custom_out[0].item() < 1.0, "custom body_names=[left_knee_link] must pick up the same error"


def test_both_feet_independent_contribution():
    n, b = 1, len(_TRACKED_NAMES)
    ref = torch.zeros(n, b, 3)
    actual = torch.zeros(n, b, 3)
    actual[0, _LEFT_FOOT_IDX, 0] = 0.3
    actual[0, _RIGHT_FOOT_IDX, 1] = 0.4
    mc = _fake_motion_command(ref, actual)
    with patch.object(wbt, "_get_motion_command_and_assert_type", return_value=mc):
        out = wbt.motion_global_feet_lin_vel(SimpleNamespace(), sigma=1.0)
    # error = mean([0.3^2, 0.4^2]) = mean([0.09, 0.16]) = 0.125
    expected = math.exp(-0.125 / 1.0**2)
    assert torch.isclose(out[0], torch.tensor(expected), atol=1e-6)


def test_per_env_independent():
    n, b = 2, len(_TRACKED_NAMES)
    ref = torch.zeros(n, b, 3)
    actual = torch.zeros(n, b, 3)
    actual[1, _LEFT_FOOT_IDX, 0] = 5.0  # only env 1 has error
    mc = _fake_motion_command(ref, actual)
    with patch.object(wbt, "_get_motion_command_and_assert_type", return_value=mc):
        out = wbt.motion_global_feet_lin_vel(SimpleNamespace(), sigma=1.0)
    assert out[0].item() == 1.0
    assert out[1].item() < 1.0
