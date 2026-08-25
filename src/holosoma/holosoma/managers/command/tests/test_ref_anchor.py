"""Unit tests for MotionCommand's mid-episode kick-entry re-anchor mechanism (2026-08-13,
locomotion->kick handoff plan increment 1 -- see enter_at_frame's own docstring in wbt.py for the
full mechanism, and https://claude.ai/code/artifact/53c1da51-d841-4979-8bf8-efd5ea652e06 for the
design this implements): _apply_ref_anchor_pos/_rot/_vec, _ref_anchor_broadcast, and
enter_at_frame.

Two claims to prove, matching this project's own "verified true no-op" discipline (see
test_local_xy_to_world.py / test_rsi_critical_frame_oversampling.py for the established
fake-object + production-method convention this file follows):

1. At the default (_ref_anchor_active=False), every accessor takes the exact pre-existing code
   path -- not just numerically equal, the SAME tensor object, since no caller exists yet for
   enter_at_frame anywhere in this codebase.
2. Once anchored, the transform is mathematically correct: the entry frame maps EXACTLY onto the
   robot's actual pose, later frames reflect the clip's own subsequent relative motion rotated by
   the yaw delta, and Z passes through as a plain offset (never rotated) because delta_quat is
   yaw-only by construction.

Isolated via a lightweight fake object providing the handful of attributes/properties these
methods actually read, and borrowing the REAL, unbound methods from MotionCommand -- so these
tests exercise the actual production implementation, not a hand-rolled reimplementation of it.
"""

from __future__ import annotations

import math

import torch

from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.utils.rotations import quat_apply, quat_inverse, quat_mul, yaw_quat


def _yaw_quat_xyzw(yaw: float) -> torch.Tensor:
    return torch.tensor([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])


_IDENTITY_XYZW = torch.tensor([0.0, 0.0, 0.0, 1.0])


class _FakeMotion:
    """Provides body_pos_w/body_quat_w as plain (T, num_bodies, 3)/(T, num_bodies, 4) tensors,
    matching MotionLoader's real storage convention -- body index 0 is the root/pelvis, matching
    enter_at_frame's own hardcoded index-0 read."""

    def __init__(self, body_pos_w: torch.Tensor, body_quat_w: torch.Tensor):
        self.body_pos_w = body_pos_w
        self.body_quat_w = body_quat_w


class _FakeSimulator:
    class _Scene:
        def __init__(self, env_origins: torch.Tensor):
            self.env_origins = env_origins

    def __init__(
        self,
        env_origins: torch.Tensor,
        robot_root_states: torch.Tensor,
        rigid_body_pos: torch.Tensor | None = None,
        rigid_body_rot: torch.Tensor | None = None,
    ):
        self.scene = self._Scene(env_origins)
        self.robot_root_states = robot_root_states
        # Only needed for tests exercising robot_ref_pos_w/robot_ref_quat_w (ref_body_index !=
        # 0 -- see enter_at_frame's own docstring for why this distinction is load-bearing).
        # Defaults to a single-body robot at the root state, for tests that don't care.
        self._rigid_body_pos = rigid_body_pos if rigid_body_pos is not None else robot_root_states[:, None, :3]
        self._rigid_body_rot = rigid_body_rot if rigid_body_rot is not None else robot_root_states[:, None, 3:7]


class _FakeEnv:
    def __init__(self, simulator: _FakeSimulator):
        self.simulator = simulator


class _FakeMotionCommand:
    """Duck-typed stand-in for a real MotionCommand: provides just enough state (motion,
    time_steps, _env, and the anchor buffers init_buffers() would create) for the real
    _apply_ref_anchor_*/enter_at_frame methods, borrowed unbound below, to run unmodified."""

    _ref_anchor_broadcast = MotionCommand._ref_anchor_broadcast
    _apply_ref_anchor_pos = MotionCommand._apply_ref_anchor_pos
    _apply_ref_anchor_rot = MotionCommand._apply_ref_anchor_rot
    _apply_ref_anchor_vec = MotionCommand._apply_ref_anchor_vec
    enter_at_frame = MotionCommand.enter_at_frame
    robot_root_pos_w = MotionCommand.robot_root_pos_w
    robot_root_quat_w = MotionCommand.robot_root_quat_w
    robot_ref_pos_w = MotionCommand.robot_ref_pos_w
    robot_ref_quat_w = MotionCommand.robot_ref_quat_w

    def __init__(
        self, motion: _FakeMotion, env: _FakeEnv, num_envs: int, active: bool = False, ref_body_index: int = 0
    ):
        self.motion = motion
        self._env = env
        self.ref_body_index = ref_body_index
        self.time_steps = torch.zeros(num_envs, dtype=torch.long)
        self._ref_anchor_delta_quat = _IDENTITY_XYZW.unsqueeze(0).repeat(num_envs, 1).clone()
        self._ref_anchor_delta_pos = torch.zeros(num_envs, 3)
        self._ref_anchor_active = active


# ============================================================================================
# No-op guarantee: _ref_anchor_active == False
# ============================================================================================


def test_apply_ref_anchor_pos_is_exact_same_tensor_object_when_inactive():
    fake = _FakeMotionCommand(_FakeMotion(torch.zeros(1, 1, 3), torch.zeros(1, 1, 4)), _FakeEnv(None), num_envs=2)
    raw = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = fake._apply_ref_anchor_pos(raw)
    assert out is raw


def test_apply_ref_anchor_rot_is_exact_same_tensor_object_when_inactive():
    fake = _FakeMotionCommand(_FakeMotion(torch.zeros(1, 1, 3), torch.zeros(1, 1, 4)), _FakeEnv(None), num_envs=2)
    raw = _IDENTITY_XYZW.unsqueeze(0).repeat(2, 1)
    out = fake._apply_ref_anchor_rot(raw)
    assert out is raw


def test_apply_ref_anchor_vec_is_exact_same_tensor_object_when_inactive():
    fake = _FakeMotionCommand(_FakeMotion(torch.zeros(1, 1, 3), torch.zeros(1, 1, 4)), _FakeEnv(None), num_envs=2)
    raw = torch.tensor([[0.5, -0.2, 0.0], [0.0, 0.0, 1.0]])
    out = fake._apply_ref_anchor_vec(raw)
    assert out is raw


def test_apply_ref_anchor_pos_no_op_holds_for_per_body_batched_shape_too():
    """(num_envs, num_bodies, 3) shape -- body_pos_w's actual shape, not just root-level (N,3)."""
    fake = _FakeMotionCommand(_FakeMotion(torch.zeros(1, 1, 3), torch.zeros(1, 1, 4)), _FakeEnv(None), num_envs=2)
    raw = torch.randn(2, 17, 3)
    out = fake._apply_ref_anchor_pos(raw)
    assert out is raw


# ============================================================================================
# _ref_anchor_broadcast
# ============================================================================================


def test_broadcast_2d_passthrough_unchanged():
    fake = _FakeMotionCommand(_FakeMotion(torch.zeros(1, 1, 3), torch.zeros(1, 1, 4)), _FakeEnv(None), num_envs=3)
    per_env = torch.arange(12).reshape(3, 4).float()
    shape_ref = torch.zeros(3, 3)
    out = fake._ref_anchor_broadcast(per_env, shape_ref)
    assert out is per_env


def test_broadcast_3d_repeats_across_body_dim():
    fake = _FakeMotionCommand(_FakeMotion(torch.zeros(1, 1, 3), torch.zeros(1, 1, 4)), _FakeEnv(None), num_envs=2)
    per_env = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    shape_ref = torch.zeros(2, 5, 3)  # 5 bodies
    out = fake._ref_anchor_broadcast(per_env, shape_ref)
    assert out.shape == (2, 5, 4)
    assert torch.equal(out[:, 0], per_env)
    assert torch.equal(out[:, 4], per_env)  # every body slot gets the SAME per-env transform


# ============================================================================================
# Correctness once active: applying an identity transform must still be numerically a no-op
# (proves the arithmetic itself is exact for identity, distinct from the object-identity tests
# above which only prove the short-circuit branch)
# ============================================================================================


def test_active_but_identity_anchor_is_numerically_a_no_op():
    fake = _FakeMotionCommand(
        _FakeMotion(torch.zeros(1, 1, 3), torch.zeros(1, 1, 4)), _FakeEnv(None), num_envs=2, active=True
    )
    raw_pos = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert torch.allclose(fake._apply_ref_anchor_pos(raw_pos), raw_pos, atol=1e-6)

    raw_quat = torch.stack([_yaw_quat_xyzw(0.7), _yaw_quat_xyzw(-1.2)])
    assert torch.allclose(fake._apply_ref_anchor_rot(raw_quat), raw_quat, atol=1e-6)

    raw_vec = torch.tensor([[0.3, -0.1, 0.0], [0.0, 0.0, 0.5]])
    assert torch.allclose(fake._apply_ref_anchor_vec(raw_vec), raw_vec, atol=1e-6)


# ============================================================================================
# enter_at_frame: the actual math
# ============================================================================================


def _make_env(env_origins: torch.Tensor, robot_root_pos: torch.Tensor, robot_root_quat: torch.Tensor) -> _FakeEnv:
    robot_root_states = torch.zeros(env_origins.shape[0], 13)
    robot_root_states[:, :3] = robot_root_pos
    robot_root_states[:, 3:7] = robot_root_quat
    return _FakeEnv(_FakeSimulator(env_origins, robot_root_states))


def test_enter_at_frame_sets_active_true():
    motion = _FakeMotion(torch.zeros(5, 1, 3), _IDENTITY_XYZW.repeat(5, 1).unsqueeze(1))
    env = _make_env(torch.zeros(1, 3), torch.zeros(1, 3), _IDENTITY_XYZW.unsqueeze(0))
    fake = _FakeMotionCommand(motion, env, num_envs=1)
    assert fake._ref_anchor_active is False
    fake.enter_at_frame(torch.tensor([0]), torch.tensor([2]))
    assert fake._ref_anchor_active is True


def test_enter_at_frame_empty_env_ids_is_a_no_op():
    motion = _FakeMotion(torch.zeros(5, 1, 3), _IDENTITY_XYZW.repeat(5, 1).unsqueeze(1))
    env = _make_env(torch.zeros(1, 3), torch.zeros(1, 3), _IDENTITY_XYZW.unsqueeze(0))
    fake = _FakeMotionCommand(motion, env, num_envs=1)
    fake.enter_at_frame(torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long))
    assert fake._ref_anchor_active is False
    assert torch.equal(fake.time_steps, torch.zeros(1, dtype=torch.long))


def test_entry_frame_maps_exactly_onto_robots_actual_pose():
    """The one property the whole mechanism exists to guarantee: at the entry tick itself,
    the transformed reference position/orientation must equal the robot's ACTUAL pose exactly,
    regardless of how different the clip's own raw captured pose was at that frame."""
    T = 10
    clip_pos = torch.zeros(T, 1, 3)
    clip_pos[:, 0, 0] = torch.linspace(0.0, 2.0, T)  # clip's root walks along its own +x
    clip_quat = _yaw_quat_xyzw(0.3).view(1, 1, 4).repeat(T, 1, 1)  # clip facing 0.3 rad, constant
    motion = _FakeMotion(clip_pos, clip_quat)

    entry_frame = 4
    # robot is nowhere near the clip's frame-4 position/heading -- 7m away, facing the opposite way
    robot_pos = torch.tensor([[7.0, -3.0, 0.05]])
    robot_quat = _yaw_quat_xyzw(math.pi + 0.3).unsqueeze(0)
    env = _make_env(torch.zeros(1, 3), robot_pos, robot_quat)
    fake = _FakeMotionCommand(motion, env, num_envs=1)

    fake.enter_at_frame(torch.tensor([0]), torch.tensor([entry_frame]))
    fake.time_steps[:] = entry_frame

    transformed_pos = fake._apply_ref_anchor_pos(
        motion.body_pos_w[fake.time_steps] + env.simulator.scene.env_origins[:, None, :]
    )
    transformed_quat = fake._apply_ref_anchor_rot(motion.body_quat_w[fake.time_steps])

    assert torch.allclose(transformed_pos[:, 0], robot_pos, atol=1e-5)
    assert torch.allclose(transformed_quat[:, 0].abs(), robot_quat.abs(), atol=1e-5)


def test_later_frame_reflects_clips_own_relative_motion_rotated_by_yaw_delta():
    """Past the entry tick, the transformed trajectory must follow the CLIP's own subsequent
    displacement (not just sit frozen at the entry point), rotated into the robot's actual
    heading -- e.g. the clip walking +1m along its own local +x, entered at a 90deg yaw offset
    from the robot, must show up as +1m along the ROBOT's local +x (which is world -y here)."""
    T = 3
    clip_pos = torch.zeros(T, 1, 3)
    clip_pos[1, 0, 0] = 1.0  # clip moves 1m in world +x between frame 0 and frame 1, in ITS OWN
    #                          zero-yaw frame -- i.e. clip's own local +x
    clip_quat = _IDENTITY_XYZW.view(1, 1, 4).repeat(T, 1, 1)  # clip facing world +x throughout
    motion = _FakeMotion(clip_pos, clip_quat)

    # robot enters at frame 0, at the world origin, facing world -y (90deg clockwise from clip)
    robot_pos = torch.zeros(1, 3)
    robot_quat = _yaw_quat_xyzw(-math.pi / 2).unsqueeze(0)
    env = _make_env(torch.zeros(1, 3), robot_pos, robot_quat)
    fake = _FakeMotionCommand(motion, env, num_envs=1)
    fake.enter_at_frame(torch.tensor([0]), torch.tensor([0]))

    frame1_pos = fake._apply_ref_anchor_pos(motion.body_pos_w[1:2] + env.simulator.scene.env_origins[:, None, :])
    # clip's own +1m local-x displacement, rotated -90deg into the robot's frame, should land at
    # world (0, -1, 0) -- i.e. the robot's own "forward" direction, not the clip's original +x.
    assert torch.allclose(frame1_pos[0, 0], torch.tensor([0.0, -1.0, 0.0]), atol=1e-5)


def test_z_passes_through_as_plain_offset_never_rotated():
    """delta_quat is yaw-only by construction -- Z must reduce to a simple height offset (robot's
    actual entry height minus the clip's captured height), never mixed with X/Y by the rotation,
    even when the yaw delta is a large, non-trivial angle."""
    T = 2
    clip_pos = torch.zeros(T, 1, 3)
    clip_pos[0, 0, 2] = 0.9  # clip's own captured height at the entry frame
    clip_quat = _IDENTITY_XYZW.view(1, 1, 4).repeat(T, 1, 1)
    motion = _FakeMotion(clip_pos, clip_quat)

    robot_pos = torch.tensor([[3.0, -1.0, 0.75]])  # robot's actual height differs from the clip's
    robot_quat = _yaw_quat_xyzw(2.1).unsqueeze(0)  # large, non-trivial yaw delta
    env = _make_env(torch.zeros(1, 3), robot_pos, robot_quat)
    fake = _FakeMotionCommand(motion, env, num_envs=1)
    fake.enter_at_frame(torch.tensor([0]), torch.tensor([0]))

    transformed = fake._apply_ref_anchor_pos(motion.body_pos_w[0:1] + env.simulator.scene.env_origins[:, None, :])
    assert torch.allclose(transformed[0, 0, 2], torch.tensor(0.75), atol=1e-5)


def test_delta_quat_is_always_pure_yaw_even_if_clip_or_robot_quat_had_roll_pitch():
    """yaw_quat() strips roll/pitch before it ever reaches the stored anchor -- confirms via the
    stored buffer directly, independent of the position/rotation tests above."""
    motion = _FakeMotion(torch.zeros(1, 1, 3), _IDENTITY_XYZW.view(1, 1, 4))
    # robot quat with real roll/pitch content, not just yaw
    robot_quat = quat_mul(
        quat_mul(_yaw_quat_xyzw(0.4).unsqueeze(0), torch.tensor([[0.1, 0.0, 0.0, 0.995]]), w_last=True),
        torch.tensor([[0.0, 0.15, 0.0, 0.989]]),
        w_last=True,
    )
    robot_quat = robot_quat / robot_quat.norm(dim=-1, keepdim=True)
    env = _make_env(torch.zeros(1, 3), torch.zeros(1, 3), robot_quat)
    fake = _FakeMotionCommand(motion, env, num_envs=1)
    fake.enter_at_frame(torch.tensor([0]), torch.tensor([0]))

    stored = fake._ref_anchor_delta_quat[0]
    recomputed_yaw_only = yaw_quat(stored.unsqueeze(0), w_last=True)[0]
    assert torch.allclose(stored, recomputed_yaw_only, atol=1e-6)


def test_anchor_uses_ref_body_index_not_body_zero():
    """Regression test for a real bug caught by live IsaacSim verification (2026-08-13): this
    project's g1 config sets body_name_ref=["torso_link"], so ref_pos_w/ref_quat_w -- what
    motion_global_ref_position_error_exp/_orientation_error_exp and body_pos_relative_w's own
    self-correcting computation actually read -- are built from ref_body_index, NOT body 0
    (pelvis/root). An earlier implementation anchored to body 0 and left a real ~5cm residual
    (the pelvis-to-torso offset) at the entry tick. Two bodies with DELIBERATELY DIFFERENT poses
    here -- if enter_at_frame used body 0, this test would catch it exactly as the live probe did."""
    T = 2
    clip_pos = torch.zeros(T, 2, 3)  # 2 bodies: index 0 (root), index 1 (ref body)
    clip_pos[0, 0] = torch.tensor([0.0, 0.0, 0.78])  # root/pelvis, frame 0
    clip_pos[0, 1] = torch.tensor([0.1, 0.0, 1.20])  # ref body (torso), frame 0 -- offset from root
    clip_quat = _IDENTITY_XYZW.view(1, 1, 4).repeat(T, 2, 1)
    motion = _FakeMotion(clip_pos, clip_quat)

    # robot's ACTUAL current torso (ref body) pose -- what the anchor must land on exactly
    robot_ref_pos = torch.tensor([[5.0, -2.0, 1.15]])
    robot_ref_quat = _yaw_quat_xyzw(1.0).unsqueeze(0)
    # robot's ACTUAL current root pose -- DELIBERATELY inconsistent with a rigid pelvis->torso
    # offset, so a body-0-anchored implementation would land somewhere clearly different
    robot_root_pos = torch.tensor([[5.0, -2.0, 0.70]])
    robot_root_quat = _IDENTITY_XYZW.unsqueeze(0)

    robot_root_states = torch.zeros(1, 13)
    robot_root_states[:, :3] = robot_root_pos
    robot_root_states[:, 3:7] = robot_root_quat
    rigid_body_pos = torch.stack([robot_root_pos, robot_ref_pos], dim=1)  # (1, 2, 3)
    rigid_body_rot = torch.stack([robot_root_quat, robot_ref_quat], dim=1)  # (1, 2, 4)
    sim = _FakeSimulator(torch.zeros(1, 3), robot_root_states, rigid_body_pos, rigid_body_rot)
    env = _FakeEnv(sim)

    fake = _FakeMotionCommand(motion, env, num_envs=1, ref_body_index=1)
    fake.enter_at_frame(torch.tensor([0]), torch.tensor([0]))

    transformed_ref_pos = fake._apply_ref_anchor_pos(
        motion.body_pos_w[fake.time_steps, fake.ref_body_index] + sim.scene.env_origins
    )
    assert torch.allclose(transformed_ref_pos, robot_ref_pos, atol=1e-5), (
        "must land on the robot's ACTUAL torso (ref_body_index) pose, not its root pose",
        transformed_ref_pos,
        robot_ref_pos,
    )


def test_env_origins_cancel_out_of_the_final_transform():
    """Algebraic claim from enter_at_frame's own docstring: whether env_origins is included in
    what gets rotated or not, it cancels between the entry-frame and later-frame reads, so the
    same (env_origins=0) and (env_origins=nonzero) setups must produce IDENTICAL relative
    trajectories once anchored -- only the absolute entry point differs, and that's already
    pinned to the robot's actual (env-origin-inclusive) position by construction."""
    T = 3
    clip_pos = torch.zeros(T, 1, 3)
    clip_pos[1, 0, 0] = 0.5
    clip_pos[2, 0, 1] = 0.3
    clip_quat = _yaw_quat_xyzw(0.2).view(1, 1, 4).repeat(T, 1, 1)
    motion = _FakeMotion(clip_pos, clip_quat)
    robot_pos = torch.tensor([[1.0, 1.0, 0.78]])
    robot_quat = _yaw_quat_xyzw(1.1).unsqueeze(0)

    env_a = _make_env(torch.zeros(1, 3), robot_pos, robot_quat)
    fake_a = _FakeMotionCommand(motion, env_a, num_envs=1)
    fake_a.enter_at_frame(torch.tensor([0]), torch.tensor([0]))

    env_b = _make_env(torch.tensor([[50.0, -20.0, 0.0]]), robot_pos, robot_quat)
    fake_b = _FakeMotionCommand(motion, env_b, num_envs=1)
    fake_b.enter_at_frame(torch.tensor([0]), torch.tensor([0]))

    for t in range(T):
        pos_a = fake_a._apply_ref_anchor_pos(clip_pos[t : t + 1] + env_a.simulator.scene.env_origins[:, None, :])
        pos_b = fake_b._apply_ref_anchor_pos(clip_pos[t : t + 1] + env_b.simulator.scene.env_origins[:, None, :])
        assert torch.allclose(pos_a, pos_b, atol=1e-5), (t, pos_a, pos_b)
