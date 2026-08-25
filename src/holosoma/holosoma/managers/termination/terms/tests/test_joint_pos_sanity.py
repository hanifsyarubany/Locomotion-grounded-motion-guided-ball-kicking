"""Unit tests for ``joint_pos_sanity_exceeded`` (managers/termination/terms/locomotion.py,
2026-08-10) -- an absolute physical-sanity check (NaN/Inf or an absurd magnitude in
``env.simulator.dof_pos``), deliberately distinct from ``BadTracking``'s reference-relative
check. See MultiSkillConfig.joint_pos_sanity_check_enabled's own docstring for the live incident
this was built from.

Isolated via a minimal fake exposing only ``simulator.dof_pos`` -- the only attribute the
function reads.
"""

from __future__ import annotations

import torch

from holosoma.managers.termination.terms.locomotion import joint_pos_sanity_exceeded


class _FakeSimulator:
    def __init__(self, dof_pos: torch.Tensor):
        self.dof_pos = dof_pos


class _FakeEnv:
    def __init__(self, dof_pos: torch.Tensor):
        self.simulator = _FakeSimulator(dof_pos)


def test_normal_joint_angles_never_trigger():
    dof_pos = torch.randn(8, 29) * 0.5  # well within any sane range
    env = _FakeEnv(dof_pos)
    result = joint_pos_sanity_exceeded(env, joint_pos_sanity_threshold=20.0)
    assert not result.any()
    assert result.shape == (8,)


def test_nan_triggers_even_though_a_naive_threshold_comparison_would_miss_it():
    """The exact failure mode this term exists for: `nan > threshold` is False under IEEE 754,
    so a magnitude-only check would silently pass a NaN-corrupted env. isfinite() must catch it."""
    dof_pos = torch.zeros(4, 29)
    dof_pos[1, 5] = float("nan")
    env = _FakeEnv(dof_pos)

    result = joint_pos_sanity_exceeded(env, joint_pos_sanity_threshold=20.0)

    assert result.tolist() == [False, True, False, False]


def test_inf_triggers():
    dof_pos = torch.zeros(4, 29)
    dof_pos[2, 0] = float("inf")
    dof_pos[3, 10] = float("-inf")
    env = _FakeEnv(dof_pos)

    result = joint_pos_sanity_exceeded(env, joint_pos_sanity_threshold=20.0)

    assert result.tolist() == [False, False, True, True]


def test_absurd_but_finite_magnitude_triggers():
    """The live-observed failure: a genuine (non-NaN, non-Inf) numerical explosion, e.g. a
    solver-divergence value in the millions -- not just literal NaN/Inf."""
    dof_pos = torch.zeros(3, 29)
    dof_pos[1, 3] = 2.36e8  # matches the magnitude of the real observed incident
    env = _FakeEnv(dof_pos)

    result = joint_pos_sanity_exceeded(env, joint_pos_sanity_threshold=20.0)

    assert result.tolist() == [False, True, False]


def test_value_exactly_at_threshold_does_not_trigger_strictly_greater_than():
    dof_pos = torch.zeros(2, 29)
    dof_pos[0, 0] = 20.0  # exactly at threshold
    dof_pos[1, 0] = 20.0001  # just past it
    env = _FakeEnv(dof_pos)

    result = joint_pos_sanity_exceeded(env, joint_pos_sanity_threshold=20.0)

    assert result.tolist() == [False, True]


def test_negative_magnitude_beyond_threshold_also_triggers():
    dof_pos = torch.zeros(2, 29)
    dof_pos[1, 0] = -25.0
    env = _FakeEnv(dof_pos)

    result = joint_pos_sanity_exceeded(env, joint_pos_sanity_threshold=20.0)

    assert result.tolist() == [False, True]


def test_custom_threshold_is_respected():
    dof_pos = torch.zeros(2, 29)
    dof_pos[1, 0] = 3.0
    env = _FakeEnv(dof_pos)

    # 3.0 exceeds a tight threshold...
    assert joint_pos_sanity_exceeded(env, joint_pos_sanity_threshold=2.0).tolist() == [False, True]
    # ...but not the default-scale one.
    assert joint_pos_sanity_exceeded(env, joint_pos_sanity_threshold=20.0).tolist() == [False, False]


def test_multiple_envs_independent():
    dof_pos = torch.zeros(5, 29)
    dof_pos[0, 0] = float("nan")
    dof_pos[2, 15] = 1e9
    env = _FakeEnv(dof_pos)

    result = joint_pos_sanity_exceeded(env, joint_pos_sanity_threshold=20.0)

    assert result.tolist() == [True, False, True, False, False]
