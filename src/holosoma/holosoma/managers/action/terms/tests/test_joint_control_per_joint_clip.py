"""Unit tests for ``JointPositionActionTerm._configure_per_joint_action_clip``
(managers/action/terms/joint_control.py) -- the resolution logic behind
``RobotControlConfig.per_joint_action_clip`` (2026-08-05, ported from RoboNaldo, arXiv:2606.11092).

Tested in isolation: the method only ever reads ``env.dof_names`` /
``env.robot_config.control.per_joint_action_clip`` and mutates
``self._action_clip_low``/``self._action_clip_high``, so it's called unbound against a lightweight
``SimpleNamespace`` stand-in for ``self`` -- no full robot config / PD gains setup needed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import holosoma.managers.action.terms.joint_control as jc

_DOF_NAMES = [
    "left_hip_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "waist_yaw_joint",
    "left_wrist_roll_joint",
]


class _FakeControl:
    def __init__(self, per_joint_action_clip):
        self.per_joint_action_clip = per_joint_action_clip


class _FakeRobotConfig:
    def __init__(self, per_joint_action_clip):
        self.control = _FakeControl(per_joint_action_clip)


class _FakeEnv:
    def __init__(self, per_joint_action_clip, dof_names=_DOF_NAMES):
        self.robot_config = _FakeRobotConfig(per_joint_action_clip)
        self.dof_names = list(dof_names)


def _configure(per_joint_action_clip, scalar_clip=100.0, dof_names=_DOF_NAMES):
    n = len(dof_names)
    fake_self = SimpleNamespace(
        _action_clip_low=torch.full((n,), -scalar_clip),
        _action_clip_high=torch.full((n,), scalar_clip),
    )
    env = _FakeEnv(per_joint_action_clip, dof_names)
    jc.JointPositionActionTerm._configure_per_joint_action_clip(fake_self, env)
    return fake_self._action_clip_low, fake_self._action_clip_high


def test_none_is_a_byte_identical_no_op():
    low, high = _configure(None, scalar_clip=100.0)
    assert torch.equal(low, torch.full((len(_DOF_NAMES),), -100.0))
    assert torch.equal(high, torch.full((len(_DOF_NAMES),), 100.0))


def test_empty_dict_is_a_byte_identical_no_op():
    low, high = _configure({}, scalar_clip=100.0)
    assert torch.equal(low, torch.full((len(_DOF_NAMES),), -100.0))
    assert torch.equal(high, torch.full((len(_DOF_NAMES),), 100.0))


def test_matched_joint_type_overrides_both_matching_dof_names():
    """"ankle_roll" must match BOTH left_ankle_roll_joint and right_ankle_roll_joint -- same
    substring-matching convention as stiffness/damping."""
    low, high = _configure({"ankle_roll": (-0.6, 0.6)}, scalar_clip=100.0)
    left_idx = _DOF_NAMES.index("left_ankle_roll_joint")
    right_idx = _DOF_NAMES.index("right_ankle_roll_joint")
    assert low[left_idx].item() == pytest.approx(-0.6) and high[left_idx].item() == pytest.approx(0.6)
    assert low[right_idx].item() == pytest.approx(-0.6) and high[right_idx].item() == pytest.approx(0.6)


def test_unmatched_joints_keep_the_scalar_bound():
    low, high = _configure({"ankle_roll": (-0.6, 0.6)}, scalar_clip=100.0)
    hip_idx = _DOF_NAMES.index("left_hip_pitch_joint")
    assert low[hip_idx].item() == -100.0 and high[hip_idx].item() == 100.0


def test_multiple_joint_types_each_apply_independently():
    low, high = _configure(
        {"ankle_roll": (-0.6, 0.6), "waist_roll": (-1.2, 1.2), "waist_pitch": (-1.2, 1.2)},
        scalar_clip=100.0,
    )
    assert low[_DOF_NAMES.index("left_ankle_roll_joint")].item() == pytest.approx(-0.6)
    assert low[_DOF_NAMES.index("waist_roll_joint")].item() == pytest.approx(-1.2)
    assert low[_DOF_NAMES.index("waist_pitch_joint")].item() == pytest.approx(-1.2)
    # waist_yaw is a DIFFERENT joint type, matches neither "waist_roll" nor "waist_pitch" as a
    # substring -- must stay at the scalar bound.
    assert low[_DOF_NAMES.index("waist_yaw_joint")].item() == -100.0
    assert high[_DOF_NAMES.index("waist_yaw_joint")].item() == 100.0


def test_asymmetric_bounds_are_respected():
    """Bounds need not be symmetric around zero -- the method must not assume that."""
    low, high = _configure({"left_wrist_roll": (-0.1, 0.3)}, scalar_clip=100.0)
    idx = _DOF_NAMES.index("left_wrist_roll_joint")
    assert low[idx].item() == pytest.approx(-0.1)
    assert high[idx].item() == pytest.approx(0.3)
