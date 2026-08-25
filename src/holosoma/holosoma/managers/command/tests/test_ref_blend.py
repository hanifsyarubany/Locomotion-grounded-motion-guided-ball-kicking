"""Unit tests for MotionCommand's reference-side blend mechanism (2026-08-13, locomotion->kick
handoff plan increment 3 -- pre_kick_reference_blend_steps; see capture_ref_blend's own docstring
in wbt.py for the full mechanism, and
https://claude.ai/code/artifact/53c1da51-d841-4979-8bf8-efd5ea652e06 for the design this
implements): _ref_blend_ratio, _apply_ref_blend, capture_ref_blend.

Same "verified true no-op" discipline as test_ref_anchor.py:
1. At the default (_ref_blend_active=False), every accessor takes the exact pre-existing code
   path -- the SAME tensor object, since capture_ref_blend is only ever called when
   pre_kick_reference_blend_steps > 0.0.
2. Once active, the blend is mathematically correct: ratio 0 at the capture tick (pure captured
   value), linear growth to 1.0 at the window boundary, clamped at 1.0 past it, and an env that
   was never captured (sentinel -1) always reads ratio 1.0 (pure clip value) regardless of any
   other env's state -- no cross-env leakage.

Isolated via a lightweight fake object providing the handful of attributes/properties these
methods actually read, and borrowing the REAL, unbound methods from MotionCommand.
"""

from __future__ import annotations

import torch

from holosoma.managers.command.terms.wbt import MotionCommand


class _FakeSimulator:
    class _Scene:
        def __init__(self, env_origins: torch.Tensor):
            self.env_origins = env_origins

    def __init__(
        self,
        env_origins: torch.Tensor,
        rigid_body_pos: torch.Tensor,
        rigid_body_rot: torch.Tensor,
        rigid_body_vel: torch.Tensor,
        rigid_body_ang_vel: torch.Tensor,
    ):
        self.scene = self._Scene(env_origins)
        self._rigid_body_pos = rigid_body_pos
        self._rigid_body_rot = rigid_body_rot
        self._rigid_body_vel = rigid_body_vel
        self._rigid_body_ang_vel = rigid_body_ang_vel


class _FakeEnv:
    def __init__(self, simulator: _FakeSimulator, episode_length_buf: torch.Tensor):
        self.simulator = simulator
        self.episode_length_buf = episode_length_buf


class _FakeMotionCommand:
    """Duck-typed stand-in for a real MotionCommand: provides just enough state for the real
    _ref_blend_ratio/_apply_ref_blend/capture_ref_blend methods, borrowed unbound below, to run
    unmodified."""

    _ref_blend_ratio = MotionCommand._ref_blend_ratio
    _apply_ref_blend = MotionCommand._apply_ref_blend
    capture_ref_blend = MotionCommand.capture_ref_blend
    robot_body_pos_w = MotionCommand.robot_body_pos_w
    robot_body_quat_w = MotionCommand.robot_body_quat_w
    robot_body_lin_vel_w = MotionCommand.robot_body_lin_vel_w
    robot_body_ang_vel_w = MotionCommand.robot_body_ang_vel_w

    def __init__(self, env: _FakeEnv, num_envs: int, num_bodies: int, active: bool = False):
        self._env = env
        self.tracked_body_indexes = torch.arange(num_bodies)
        self._ref_blend_captured_pos = torch.zeros(num_envs, num_bodies, 3)
        self._ref_blend_captured_quat = torch.zeros(num_envs, num_bodies, 4)
        self._ref_blend_captured_quat[:, :, 3] = 1.0
        self._ref_blend_captured_lin_vel = torch.zeros(num_envs, num_bodies, 3)
        self._ref_blend_captured_ang_vel = torch.zeros(num_envs, num_bodies, 3)
        self._ref_blend_start_step = torch.full((num_envs,), -1, dtype=torch.long)
        self._ref_blend_window_steps = torch.zeros(num_envs)
        self._ref_blend_active = active


def _make_env(num_envs: int, num_bodies: int, episode_length_buf: list[int]) -> _FakeEnv:
    sim = _FakeSimulator(
        env_origins=torch.zeros(num_envs, 3),
        rigid_body_pos=torch.zeros(num_envs, num_bodies, 3),
        rigid_body_rot=torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(num_envs, num_bodies, 1),
        rigid_body_vel=torch.zeros(num_envs, num_bodies, 3),
        rigid_body_ang_vel=torch.zeros(num_envs, num_bodies, 3),
    )
    return _FakeEnv(sim, torch.tensor(episode_length_buf, dtype=torch.long))


# ============================================================================================
# No-op guarantee: _ref_blend_active == False
# ============================================================================================


def test_apply_ref_blend_is_exact_same_tensor_object_when_inactive():
    env = _make_env(2, 3, [10, 20])
    fake = _FakeMotionCommand(env, num_envs=2, num_bodies=3)
    clip_val = torch.randn(2, 3, 3)
    out = fake._apply_ref_blend(clip_val, fake._ref_blend_captured_pos, is_quat=False)
    assert out is clip_val


def test_apply_ref_blend_quat_is_exact_same_tensor_object_when_inactive():
    env = _make_env(2, 3, [10, 20])
    fake = _FakeMotionCommand(env, num_envs=2, num_bodies=3)
    clip_quat = torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(2, 3, 1)
    out = fake._apply_ref_blend(clip_quat, fake._ref_blend_captured_quat, is_quat=True)
    assert out is clip_quat


def test_capture_ref_blend_on_empty_env_ids_is_a_no_op():
    env = _make_env(2, 3, [10, 20])
    fake = _FakeMotionCommand(env, num_envs=2, num_bodies=3)
    fake.capture_ref_blend(torch.zeros(0, dtype=torch.long), torch.zeros(0))
    assert fake._ref_blend_active is False
    assert torch.equal(fake._ref_blend_start_step, torch.tensor([-1, -1]))


# ============================================================================================
# _ref_blend_ratio
# ============================================================================================


def test_ratio_is_one_for_an_env_never_captured():
    env = _make_env(1, 2, [500])
    fake = _FakeMotionCommand(env, num_envs=1, num_bodies=2, active=True)
    ratio = fake._ref_blend_ratio()
    assert torch.allclose(ratio, torch.tensor([1.0]))


def test_ratio_is_zero_at_the_exact_capture_tick():
    env = _make_env(1, 2, [50])
    fake = _FakeMotionCommand(env, num_envs=1, num_bodies=2)
    fake.capture_ref_blend(torch.tensor([0]), torch.tensor([10.0]))
    ratio = fake._ref_blend_ratio()
    assert torch.allclose(ratio, torch.tensor([0.0]))


def test_ratio_grows_linearly_within_the_window():
    env = _make_env(1, 2, [50])
    fake = _FakeMotionCommand(env, num_envs=1, num_bodies=2)
    fake.capture_ref_blend(torch.tensor([0]), torch.tensor([10.0]))
    env.episode_length_buf[:] = 53  # 3 of 10 ticks elapsed
    ratio = fake._ref_blend_ratio()
    assert torch.allclose(ratio, torch.tensor([0.3]))


def test_ratio_clamped_to_one_past_the_window():
    env = _make_env(1, 2, [50])
    fake = _FakeMotionCommand(env, num_envs=1, num_bodies=2)
    fake.capture_ref_blend(torch.tensor([0]), torch.tensor([10.0]))
    env.episode_length_buf[:] = 200
    ratio = fake._ref_blend_ratio()
    assert torch.allclose(ratio, torch.tensor([1.0]))


def test_ratio_per_env_independent_no_leakage():
    """Two envs, only one captured -- the other must read exactly 1.0 regardless."""
    env = _make_env(2, 2, [50, 50])
    fake = _FakeMotionCommand(env, num_envs=2, num_bodies=2)
    fake.capture_ref_blend(torch.tensor([0]), torch.tensor([10.0]))
    env.episode_length_buf[:] = torch.tensor([55, 999])  # env0: 5/10 elapsed; env1: never captured
    ratio = fake._ref_blend_ratio()
    assert torch.allclose(ratio, torch.tensor([0.5, 1.0]))


# ============================================================================================
# capture_ref_blend: snapshots the robot's OWN live pose
# ============================================================================================


def test_capture_ref_blend_snapshots_live_robot_state():
    env = _make_env(1, 2, [50])
    env.simulator._rigid_body_pos[:] = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    env.simulator._rigid_body_vel[:] = torch.tensor([[0.1, 0.0, 0.0], [0.0, 0.2, 0.0]])
    fake = _FakeMotionCommand(env, num_envs=1, num_bodies=2)

    fake.capture_ref_blend(torch.tensor([0]), torch.tensor([15.0]))

    assert fake._ref_blend_active is True
    assert int(fake._ref_blend_start_step[0]) == 50
    assert float(fake._ref_blend_window_steps[0]) == 15.0
    assert torch.allclose(fake._ref_blend_captured_pos[0], env.simulator._rigid_body_pos[0])
    assert torch.allclose(fake._ref_blend_captured_lin_vel[0], env.simulator._rigid_body_vel[0])


# ============================================================================================
# _apply_ref_blend: the actual interpolation math
# ============================================================================================


def test_apply_ref_blend_pos_returns_captured_value_at_ratio_zero():
    env = _make_env(1, 1, [50])
    fake = _FakeMotionCommand(env, num_envs=1, num_bodies=1)
    fake.capture_ref_blend(torch.tensor([0]), torch.tensor([10.0]))
    fake._ref_blend_captured_pos[0, 0] = torch.tensor([1.0, 2.0, 3.0])  # override the (zero) live snapshot

    clip_val = torch.tensor([[[9.0, 9.0, 9.0]]])
    out = fake._apply_ref_blend(clip_val, fake._ref_blend_captured_pos, is_quat=False)
    assert torch.allclose(out[0, 0], torch.tensor([1.0, 2.0, 3.0]), atol=1e-5)


def test_apply_ref_blend_pos_linearly_interpolates_mid_window():
    env = _make_env(1, 1, [0])
    fake = _FakeMotionCommand(env, num_envs=1, num_bodies=1)
    fake.capture_ref_blend(torch.tensor([0]), torch.tensor([10.0]))
    fake._ref_blend_captured_pos[0, 0] = torch.tensor([0.0, 0.0, 0.0])
    env.episode_length_buf[:] = 4  # ratio = 0.4

    clip_val = torch.tensor([[[10.0, 0.0, 0.0]]])
    out = fake._apply_ref_blend(clip_val, fake._ref_blend_captured_pos, is_quat=False)
    assert torch.allclose(out[0, 0], torch.tensor([4.0, 0.0, 0.0]), atol=1e-5)


def test_apply_ref_blend_pos_returns_clip_value_past_the_window():
    env = _make_env(1, 1, [0])
    fake = _FakeMotionCommand(env, num_envs=1, num_bodies=1)
    fake.capture_ref_blend(torch.tensor([0]), torch.tensor([10.0]))
    fake._ref_blend_captured_pos[0, 0] = torch.tensor([1.0, 1.0, 1.0])
    env.episode_length_buf[:] = 999

    clip_val = torch.tensor([[[9.0, 9.0, 9.0]]])
    out = fake._apply_ref_blend(clip_val, fake._ref_blend_captured_pos, is_quat=False)
    assert torch.allclose(out[0, 0], torch.tensor([9.0, 9.0, 9.0]), atol=1e-5)


def test_apply_ref_blend_quat_returns_captured_quat_at_ratio_zero():
    env = _make_env(1, 1, [50])
    fake = _FakeMotionCommand(env, num_envs=1, num_bodies=1)
    fake.capture_ref_blend(torch.tensor([0]), torch.tensor([10.0]))
    import math

    captured = torch.tensor([0.0, 0.0, math.sin(0.4), math.cos(0.4)])
    fake._ref_blend_captured_quat[0, 0] = captured

    clip_quat = torch.tensor([[[0.0, 0.0, math.sin(1.2), math.cos(1.2)]]])
    out = fake._apply_ref_blend(clip_quat, fake._ref_blend_captured_quat, is_quat=True)
    assert torch.allclose(out[0, 0].abs(), captured.abs(), atol=1e-5)


def test_apply_ref_blend_quat_returns_clip_quat_past_the_window():
    env = _make_env(1, 1, [0])
    fake = _FakeMotionCommand(env, num_envs=1, num_bodies=1)
    fake.capture_ref_blend(torch.tensor([0]), torch.tensor([10.0]))
    env.episode_length_buf[:] = 999
    import math

    clip_quat = torch.tensor([[[0.0, 0.0, math.sin(1.2), math.cos(1.2)]]])
    out = fake._apply_ref_blend(clip_quat, fake._ref_blend_captured_quat, is_quat=True)
    assert torch.allclose(out[0, 0].abs(), clip_quat[0, 0].abs(), atol=1e-5)


def test_apply_ref_blend_multi_body_ratio_broadcasts_across_bodies():
    """One env, 3 tracked bodies -- the SAME per-env ratio must apply to every body uniformly."""
    env = _make_env(1, 3, [0])
    fake = _FakeMotionCommand(env, num_envs=1, num_bodies=3)
    fake.capture_ref_blend(torch.tensor([0]), torch.tensor([10.0]))
    fake._ref_blend_captured_pos[0] = torch.zeros(3, 3)
    env.episode_length_buf[:] = 5  # ratio = 0.5

    clip_val = torch.tensor([[[2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 6.0]]])
    out = fake._apply_ref_blend(clip_val, fake._ref_blend_captured_pos, is_quat=False)
    assert torch.allclose(out[0], torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]), atol=1e-5)
