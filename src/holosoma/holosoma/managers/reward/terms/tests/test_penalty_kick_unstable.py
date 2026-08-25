"""Unit tests for ``penalty_kick_unstable`` (managers/reward/terms/locomotion.py) -- 2026-08-05,
ported from RoboNaldo (arXiv:2606.11092)'s ``unstable_penalty``. Reuses the exact
``_FakeMotionCommand``/``_FakeCommandManager``/``_FakeEnv`` fakes from
test_kick_recovery_gate.py (same file, same production expression for ``in_kicking_phase``, so
these tests exercise the real ``_kick_recovery_gate`` this term composes with, not a
reimplementation of it).
"""

from __future__ import annotations

import torch

from holosoma.managers.reward.terms import locomotion as loco_terms
from holosoma.managers.reward.terms.tests.test_kick_recovery_gate import (
    _FakeEnv,
    _FakeMotionCommand,
)


def _env_with_root_states(motion_command, num_envs, root_states):
    env = _FakeEnv(motion_command, num_envs)
    env.simulator = type("Sim", (), {"robot_root_states": root_states})()
    return env


def _root_states(lin_vel_xy, ang_vel_xyz, num_envs):
    """[num_envs, 13]: [0:3] pos (unused), [3:7] quat (unused), [7:10] lin_vel, [10:13] ang_vel."""
    rs = torch.zeros(num_envs, 13)
    rs[:, 7:9] = torch.tensor(lin_vel_xy)
    rs[:, 10:13] = torch.tensor(ang_vel_xyz)
    return rs


def test_zero_during_swing_regardless_of_velocity():
    """A fast-moving base during swing (locomotion-approach + strike) must NOT be penalized --
    the whole point of gating this to post-kick recovery/hold only."""
    n = 1
    mc = _FakeMotionCommand(
        time_steps=torch.tensor([50]), motion_ids=torch.tensor([0]), stand_start_idx=torch.tensor([200])
    )
    root_states = _root_states([[3.0, 3.0]], [[5.0, 5.0, 5.0]], n)
    env = _env_with_root_states(mc, n, root_states)
    out = loco_terms.penalty_kick_unstable(env)
    assert out.item() == 0.0


def test_nonzero_well_past_grace_period():
    """Well past stand_start_idx + grace_steps, the gate is at full strength (1.0) -- output must
    match the raw formula exactly."""
    n = 1
    mc = _FakeMotionCommand(
        time_steps=torch.tensor([300]), motion_ids=torch.tensor([0]), stand_start_idx=torch.tensor([200])
    )
    root_states = _root_states([[1.0, 2.0]], [[0.0, 0.0, 3.0]], n)
    env = _env_with_root_states(mc, n, root_states)
    out = loco_terms.penalty_kick_unstable(env, ang_vel_weight=0.5, grace_steps=50.0)
    expected_lin = 1.0**2 + 2.0**2  # 5.0
    expected_ang = 0.5 * (3.0**2)  # 4.5
    assert torch.isclose(out, torch.tensor(expected_lin + expected_ang), atol=1e-5)


def test_ramps_linearly_within_grace_window():
    """Exactly at the midpoint of the grace window, the gate multiplier must be 0.5 -- confirms
    composition with _kick_recovery_gate's own linear ramp, not a hard step."""
    n = 1
    mc = _FakeMotionCommand(
        time_steps=torch.tensor([225]),  # 25 steps past stand_start_idx=200, grace_steps=50 -> ramp=0.5
        motion_ids=torch.tensor([0]),
        stand_start_idx=torch.tensor([200]),
    )
    root_states = _root_states([[2.0, 0.0]], [[0.0, 0.0, 0.0]], n)
    env = _env_with_root_states(mc, n, root_states)
    out = loco_terms.penalty_kick_unstable(env, grace_steps=50.0)
    raw_magnitude = 2.0**2  # 4.0
    assert torch.isclose(out, torch.tensor(raw_magnitude * 0.5), atol=1e-5)


def test_ang_vel_weight_scales_only_the_angular_term():
    n = 1
    mc = _FakeMotionCommand(
        time_steps=torch.tensor([300]), motion_ids=torch.tensor([0]), stand_start_idx=torch.tensor([200])
    )
    root_states = _root_states([[0.0, 0.0]], [[0.0, 0.0, 2.0]], n)
    env = _env_with_root_states(mc, n, root_states)
    out_default = loco_terms.penalty_kick_unstable(env, ang_vel_weight=0.5)
    out_double = loco_terms.penalty_kick_unstable(env, ang_vel_weight=1.0)
    assert torch.isclose(out_default, torch.tensor(0.5 * 4.0), atol=1e-5)
    assert torch.isclose(out_double, torch.tensor(1.0 * 4.0), atol=1e-5)


def test_zero_velocity_gives_zero_regardless_of_phase():
    n = 1
    mc = _FakeMotionCommand(
        time_steps=torch.tensor([300]), motion_ids=torch.tensor([0]), stand_start_idx=torch.tensor([200])
    )
    root_states = _root_states([[0.0, 0.0]], [[0.0, 0.0, 0.0]], n)
    env = _env_with_root_states(mc, n, root_states)
    out = loco_terms.penalty_kick_unstable(env)
    assert out.item() == 0.0


def test_per_env_independent_mixed_phase():
    """Two envs, one deep in recovery (full gate) and one still swinging (gate=0), same nonzero
    velocity -- only the recovering env's penalty should be nonzero."""
    n = 2
    mc = _FakeMotionCommand(
        time_steps=torch.tensor([300, 50]), motion_ids=torch.tensor([0, 0]), stand_start_idx=torch.tensor([200])
    )
    root_states = _root_states([[2.0, 0.0], [2.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], n)
    env = _env_with_root_states(mc, n, root_states)
    out = loco_terms.penalty_kick_unstable(env)
    assert out[0].item() > 0.0
    assert out[1].item() == 0.0
