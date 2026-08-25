"""Unit tests for ``kick_aim_command`` (2026-08-22 azimuth-aim refactor) -- the observation term
that replaces ``target_pos_b`` in the ``kick_target_pos_b`` obs slot. See
``SkillConfig.kick_aim_enabled``'s own docstring for the full mechanism.

The two properties that must hold, and that these tests pin:
  1. Any env whose assigned skill is NOT kick_aim_enabled (including the mechanism being entirely
     absent, e.g. no ball in the scene) falls through to ``target_pos_b`` verbatim -- bit-identical
     to before this function existed.
  2. Any env whose assigned skill IS kick_aim_enabled reads ``[theta/theta_ref, 0.0]`` -- bounded,
     constant per attempt (not recomputed from robot/ball position at all), independent of the
     fake env's robot_xy/yaw here (proving it does NOT fall through to the world-frame transform).

Uses the same lightweight fake env as test_target_pos_b_distance_compression.py, extended with the
kick_aim-specific attributes.
"""

from __future__ import annotations

import math

import torch

from holosoma.managers.observation.terms.unified import kick_aim_command


class _FakeMotionCommand:
    def __init__(
        self,
        target_xy_w: torch.Tensor | None,
        kick_aim_enabled_per_motion: torch.Tensor | None = None,
        motion_ids: torch.Tensor | None = None,
        kick_aim_theta: torch.Tensor | None = None,
        kick_aim_theta_ref_deg: float = 45.0,
    ):
        if target_xy_w is not None:
            self.target_xy_w = target_xy_w
        if kick_aim_enabled_per_motion is not None:
            self.kick_aim_enabled_per_motion = kick_aim_enabled_per_motion
            self.motion_ids = motion_ids
            self.kick_aim_theta = kick_aim_theta
            self.kick_aim_theta_ref_deg = kick_aim_theta_ref_deg


class _FakeCommandManager:
    def __init__(self, motion_command):
        self._motion_command = motion_command

    def get_state(self, name):
        assert name == "motion_command"
        return self._motion_command


class _FakeSimulator:
    def __init__(self, robot_root_states):
        self.robot_root_states = robot_root_states


class _FakeEnv:
    def __init__(self, robot_xy: torch.Tensor, target_xy: torch.Tensor | None, motion_command_kwargs=None, yaw: float = 0.0):
        n = robot_xy.shape[0]
        root = torch.zeros(n, 13)
        root[:, :2] = robot_xy
        self.simulator = _FakeSimulator(root)
        self.base_quat = torch.tensor([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]).repeat(n, 1)
        self.command_manager = _FakeCommandManager(_FakeMotionCommand(target_xy, **(motion_command_kwargs or {})))
        self.num_envs = n
        self.device = "cpu"


def test_mechanism_absent_falls_through_to_target_pos_b():
    """No kick_aim_enabled_per_motion attribute at all (e.g. an older checkpoint's config, or a
    locomotion-only debug config) -- must reduce to target_pos_b's own raw-offset behavior."""
    robot_xy = torch.tensor([[0.0, 0.0]])
    target_xy = torch.tensor([[3.0, 4.0]])
    env = _FakeEnv(robot_xy, target_xy)
    result = kick_aim_command(env)
    assert torch.allclose(result, torch.tensor([[3.0, 4.0]]), atol=1e-5)


def test_non_aim_enabled_env_falls_through_to_target_pos_b():
    """Mechanism present, but THIS env's assigned skill has kick_aim_enabled=False -- must be
    bit-identical to target_pos_b, not the theta command."""
    robot_xy = torch.tensor([[0.0, 0.0]])
    target_xy = torch.tensor([[3.0, 4.0]])
    env = _FakeEnv(
        robot_xy, target_xy,
        motion_command_kwargs=dict(
            kick_aim_enabled_per_motion=torch.tensor([False]),
            motion_ids=torch.tensor([0]),
            kick_aim_theta=torch.tensor([20.0]),  # nonzero -- must be IGNORED for this env
            kick_aim_theta_ref_deg=45.0,
        ),
    )
    result = kick_aim_command(env)
    assert torch.allclose(result, torch.tensor([[3.0, 4.0]]), atol=1e-5)


def test_aim_enabled_env_reads_normalized_theta_not_world_frame():
    """kick_aim_enabled=True: must read theta/theta_ref in dim 0, 0.0 in dim 1 -- and must NOT
    depend on robot_xy/target_xy at all (proving it doesn't fall through to the world transform).
    Checked at two different robot positions with the SAME theta -- result must be identical."""
    target_xy = torch.tensor([[3.0, 4.0]])
    common_kwargs = dict(
        kick_aim_enabled_per_motion=torch.tensor([True]),
        motion_ids=torch.tensor([0]),
        kick_aim_theta=torch.tensor([22.5]),
        kick_aim_theta_ref_deg=45.0,
    )
    env_a = _FakeEnv(torch.tensor([[0.0, 0.0]]), target_xy, motion_command_kwargs=common_kwargs)
    env_b = _FakeEnv(torch.tensor([[50.0, -30.0]]), target_xy, motion_command_kwargs=common_kwargs, yaw=1.2)

    result_a = kick_aim_command(env_a)
    result_b = kick_aim_command(env_b)

    expected = torch.tensor([[22.5 / 45.0, 0.0]])
    assert torch.allclose(result_a, expected, atol=1e-5)
    assert torch.allclose(result_b, expected, atol=1e-5)
    assert torch.allclose(result_a, result_b)


def test_aim_command_bounded_at_theta_max_equal_to_theta_ref():
    """The config-load-time invariant (theta_max <= theta_ref) means the normalized command must
    never exceed [-1, 1] -- checked at the boundary."""
    target_xy = torch.tensor([[3.0, 4.0]])
    for theta, expected_dim0 in ((45.0, 1.0), (-45.0, -1.0), (0.0, 0.0)):
        env = _FakeEnv(
            torch.tensor([[0.0, 0.0]]), target_xy,
            motion_command_kwargs=dict(
                kick_aim_enabled_per_motion=torch.tensor([True]),
                motion_ids=torch.tensor([0]),
                kick_aim_theta=torch.tensor([theta]),
                kick_aim_theta_ref_deg=45.0,
            ),
        )
        result = kick_aim_command(env)
        assert abs(result[0, 0].item() - expected_dim0) < 1e-5
        assert result[0, 1].item() == 0.0


def test_mixed_batch_selects_per_env():
    """Two envs in the same batch, one aim-enabled one not -- each must get its own correct
    value, not whichever the batch mostly is."""
    robot_xy = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    target_xy = torch.tensor([[3.0, 4.0], [1.0, 0.0]])
    env = _FakeEnv(
        robot_xy, target_xy,
        motion_command_kwargs=dict(
            kick_aim_enabled_per_motion=torch.tensor([True, False]),
            motion_ids=torch.tensor([0, 1]),
            kick_aim_theta=torch.tensor([9.0, 999.0]),  # env 1's theta must be irrelevant/ignored
            kick_aim_theta_ref_deg=45.0,
        ),
    )
    result = kick_aim_command(env)
    assert torch.allclose(result[0], torch.tensor([9.0 / 45.0, 0.0]), atol=1e-5)
    assert torch.allclose(result[1], torch.tensor([1.0, 0.0]), atol=1e-5)  # target_pos_b(env1)
