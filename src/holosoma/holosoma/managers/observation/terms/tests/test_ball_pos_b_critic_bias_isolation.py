"""Unit tests proving the actor/critic bias-isolation fix for kick_ball_pos_b
(managers/observation/terms/unified.py): ``ball_pos_b`` (actor) applies the per-episode
perception bias; ``ball_pos_b_ground_truth`` (critic) does not, given identical underlying
simulator state.

Before this fix, both actor_obs and critic_obs called the same ``ball_pos_b``, so the bias --
unlike noise/delay, which ObservationManager correctly excludes for the critic via
``group.enable_noise=False`` -- leaked into the critic's supposedly-privileged ground truth.

Uses a lightweight fake env exposing exactly what ``_ball_pos_b_raw`` reads (simulator root
states, base_quat, cached ball-actor indices) so this exercises the REAL production functions,
not a reimplementation of their math.
"""

from __future__ import annotations

import math

import torch

from holosoma.managers.observation.terms.unified import ball_pos_b, ball_pos_b_ground_truth


class _FakeSimulator:
    def __init__(self, robot_root_states: torch.Tensor, all_root_states: torch.Tensor):
        self.robot_root_states = robot_root_states
        self.all_root_states = all_root_states


class _FakeEnv:
    def __init__(self, robot_pos: torch.Tensor, ball_pos: torch.Tensor, base_quat: torch.Tensor):
        n = robot_pos.shape[0]
        robot_root_states = torch.zeros(n, 13)
        robot_root_states[:, :3] = robot_pos
        self.simulator = _FakeSimulator(robot_root_states, ball_pos)
        self.base_quat = base_quat
        # Bypass get_actor_indices/simulator lookup -- _ball_actor_indices reads this cache
        # directly if already set, matching production's own lazy-cache pattern.
        self._ball_obs_actor_indices = torch.arange(n)


_IDENTITY_XYZW = torch.tensor([0.0, 0.0, 0.0, 1.0])


def _yaw_quat_xyzw(yaw_rad: float) -> torch.Tensor:
    return torch.tensor([0.0, 0.0, math.sin(yaw_rad / 2), math.cos(yaw_rad / 2)])


def test_ground_truth_matches_ball_pos_b_when_no_bias_set():
    """With no bias configured (the common case -- env._ball_obs_bias never set), both functions
    must return the identical value: the fix must not change behavior when bias is absent."""
    robot_pos = torch.tensor([[1.0, 2.0, 0.8]])
    ball_pos = torch.tensor([[3.0, 2.0, 0.11]])
    env = _FakeEnv(robot_pos, ball_pos, _IDENTITY_XYZW.unsqueeze(0))

    actor_val = ball_pos_b(env)
    critic_val = ball_pos_b_ground_truth(env)
    assert torch.allclose(actor_val, critic_val, atol=1e-6)


def test_ground_truth_matches_ball_pos_b_when_bias_is_exactly_zero():
    """A zero bias tensor (the reset hook's own no-op case, e.g. observation_bias=0.0 in the
    yaml) must also produce identical values -- zero is a real, common configuration, not just
    the absent-attribute case above."""
    robot_pos = torch.tensor([[0.0, 0.0, 0.8]])
    ball_pos = torch.tensor([[2.0, -0.5, 0.11]])
    env = _FakeEnv(robot_pos, ball_pos, _IDENTITY_XYZW.unsqueeze(0))
    env._ball_obs_bias = torch.zeros(1, 3)

    actor_val = ball_pos_b(env)
    critic_val = ball_pos_b_ground_truth(env)
    assert torch.allclose(actor_val, critic_val, atol=1e-6)


def test_nonzero_bias_affects_only_actor_not_critic():
    """THE decisive check: with a real, nonzero bias set, ball_pos_b (actor) must reflect it,
    and ball_pos_b_ground_truth (critic) must be completely unaffected -- exactly reproducing
    the pre-bias ground-truth relative position."""
    robot_pos = torch.tensor([[0.0, 0.0, 0.8]])
    ball_pos = torch.tensor([[2.0, -0.5, 0.11]])
    env = _FakeEnv(robot_pos, ball_pos, _IDENTITY_XYZW.unsqueeze(0))
    bias = torch.tensor([[0.05, -0.03, 0.0]])
    env._ball_obs_bias = bias

    ground_truth_expected = ball_pos_b_ground_truth(env)  # no bias applied here regardless
    actor_val = ball_pos_b(env)

    assert torch.allclose(actor_val, ground_truth_expected + bias, atol=1e-6)
    assert not torch.allclose(actor_val, ground_truth_expected, atol=1e-6), "bias should be visibly nonzero"
    # Re-confirm critic stays exactly the raw value, unaffected by the SAME bias env now carries.
    critic_val = ball_pos_b_ground_truth(env)
    assert torch.allclose(critic_val, ground_truth_expected, atol=1e-6)


def test_ground_truth_respects_heading_frame_rotation_identically_to_actor_transform():
    """Both functions must apply the SAME heading-frame rotation (not just the same translation)
    -- verified with a non-trivial yaw, bias-free, so they should match exactly here too."""
    robot_pos = torch.tensor([[1.0, 1.0, 0.8]])
    ball_pos = torch.tensor([[2.0, 1.0, 0.11]])  # directly "east" of the robot in world frame
    quat = _yaw_quat_xyzw(math.pi / 2).unsqueeze(0)  # robot facing +90deg (world +y)
    env = _FakeEnv(robot_pos, ball_pos, quat)

    actor_val = ball_pos_b(env)
    critic_val = ball_pos_b_ground_truth(env)
    assert torch.allclose(actor_val, critic_val, atol=1e-6)
    # sanity: world-east ball, robot facing world+y -> ball reads as "to the robot's right"
    # (negative local y), not "in front" (local x) -- proves the rotation is genuinely applied,
    # not just translation matching by coincidence.
    assert critic_val[0, 0].abs().item() < 1e-4  # ~0 forward component
    assert critic_val[0, 1].item() < -0.5  # clearly lateral
