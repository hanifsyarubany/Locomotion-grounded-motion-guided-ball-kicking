"""Unit tests for MotionCommand.local_xy_to_world: converts a ROBOT-LOCAL (forward, lateral)
offset -- e.g. SkillConfig.x/y -- to world (x, y), anchored to the robot's ACTUAL spawn position
(not just env_origin) and rotated by its ACTUAL (yaw-only) spawn heading (not assumed to be
world +X).

This is the fix for a real, empirically-verified bug: the robot's kick-mode spawn pose comes
directly from the reference clip's own captured pelvis position/orientation (essentially
arbitrary per clip), while the OLD ball-placement code added (x, y) as a fixed world-axis offset
from env_origin alone -- so "x meters forward" was only ever true for a clip captured at exactly
zero yaw with zero pelvis offset. Live verification found real production clips landing 2-3m off
from their configured position, including one case where the ball ended up effectively behind the
robot instead of 3.87m in front.

Two real bugs were caught and fixed during a live re-verification of the FIRST attempt at this
fix (both covered below, not just asserted from memory):
1. root_quat_w/robot_quat_w is XYZW, not wxyz -- a stale comment elsewhere in wbt.py mislabels it;
   the actual MotionLoader conversion (`# Change to xyzw`) is the source of truth. The first
   attempt used w_last=False throughout and was simply wrong on every non-trivial rotation.
2. During a REAL env.reset(), the robot's simulated pose is NOT root_pos_w/root_quat_w directly --
   NoiseToInitialPoseConfig applies real, substantial per-reset orientation noise
   (overall_noise_scale=1.0, root_rot=[0.1,0.1,0.2] rad) on top before writing to the simulator.
   Placing the ball relative to the pre-noise heading landed it relative to a heading the robot
   doesn't actually end up at. Fixed via explicit robot_pos_w/robot_quat_w override params, which
   reset()'s own call site now passes (the post-noise target_root_pos/target_root_rot), while
   replay.py/BallPositionWindow (which set the robot's displayed pose with no noise at all -- see
   WholeBodyTrackingManager.step_visualize_motion) correctly keep using the defaults.

Isolated via a lightweight fake object (not a real MotionCommand instance -- constructing one
needs a live env) that provides root_pos_w/root_quat_w as plain instance attributes and borrows
the real, unbound local_xy_to_world method -- so these tests exercise the actual production
implementation, not a hand-rolled reimplementation of it.
"""

import math

import torch

from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.utils.rotations import quat_rotate_inverse, yaw_quat


class _FakeMotionCommand:
    """Provides root_pos_w/root_quat_w as plain (num_envs, 3)/(num_envs, 4) XYZW attributes,
    matching MotionCommand's ACTUAL storage convention (see module docstring) -- and borrows the
    REAL local_xy_to_world implementation so tests exercise production code, not a
    reimplementation of it."""

    def __init__(self, root_pos_w: torch.Tensor, root_quat_w: torch.Tensor):
        self.root_pos_w = root_pos_w
        self.root_quat_w = root_quat_w
        self.device = "cpu"

    local_xy_to_world = MotionCommand.local_xy_to_world


def _yaw_quat_xyzw(yaw_rad: float) -> torch.Tensor:
    """A pure-yaw quaternion in XYZW convention (x, y, z, w)."""
    return torch.tensor([0.0, 0.0, math.sin(yaw_rad / 2), math.cos(yaw_rad / 2)])


_IDENTITY_XYZW = torch.tensor([0.0, 0.0, 0.0, 1.0])


def test_reduces_to_plain_add_at_identity_heading_and_zero_position():
    """MuJoCo/RoboJuDo's own convention: robot at world origin, identity quat (facing +x) --
    should be bit-identical to the old plain env_origin add."""
    fake = _FakeMotionCommand(root_pos_w=torch.zeros(1, 3), root_quat_w=_IDENTITY_XYZW.unsqueeze(0))
    local_xy = torch.tensor([[2.84, -0.46]])
    world_xy = fake.local_xy_to_world(local_xy, torch.tensor([0]))
    assert torch.allclose(world_xy, local_xy, atol=1e-5)


def test_anchors_to_robot_actual_position_not_just_env_origin():
    """Robot spawned away from origin (e.g. env_origin + the clip's own captured pelvis offset,
    like the real (2.87, 0.92) measured for a production clip) -- with identity heading, the
    world result should be robot_pos + local_xy, not local_xy alone."""
    robot_pos = torch.tensor([[2.87, 0.92, 0.81]])
    fake = _FakeMotionCommand(root_pos_w=robot_pos, root_quat_w=_IDENTITY_XYZW.unsqueeze(0))
    local_xy = torch.tensor([[2.84, -0.46]])
    world_xy = fake.local_xy_to_world(local_xy, torch.tensor([0]))
    assert torch.allclose(world_xy, robot_pos[:, :2] + local_xy, atol=1e-5)


def test_rotates_by_robot_actual_yaw_heading():
    """Robot facing +90deg yaw (rotated away from +x) -- "2.84m forward" should land somewhere
    OTHER than the naive (2.84, -0.46) world offset, proving the rotation is actually applied."""
    fake = _FakeMotionCommand(root_pos_w=torch.zeros(1, 3), root_quat_w=_yaw_quat_xyzw(math.pi / 2).unsqueeze(0))
    local_xy = torch.tensor([[2.84, -0.46]])
    world_xy = fake.local_xy_to_world(local_xy, torch.tensor([0]))
    assert not torch.allclose(world_xy, local_xy, atol=1e-3)
    # magnitude (distance from robot) must be preserved by a pure rotation
    assert torch.allclose(world_xy.norm(), local_xy.norm(), atol=1e-4)


def test_only_yaw_applies_pitch_and_roll_are_ignored():
    """A clip captured with significant pitch/roll (e.g. mid-kick torso lean) but zero yaw must
    still reduce to a plain translation -- only heading (yaw) should affect ball placement, never
    pitch/roll, or the ball could end up rotated into the ground/air."""
    half = math.pi / 6 / 2
    pitch_quat = torch.tensor([[0.0, math.sin(half), 0.0, math.cos(half)]])  # xyzw, pitch (y-axis) only
    fake = _FakeMotionCommand(root_pos_w=torch.zeros(1, 3), root_quat_w=pitch_quat)
    local_xy = torch.tensor([[2.84, -0.46]])
    world_xy = fake.local_xy_to_world(local_xy, torch.tensor([0]))
    assert torch.allclose(world_xy, local_xy, atol=1e-5)


def test_is_the_exact_mathematical_inverse_of_the_ball_pos_b_observation_transform():
    """The decisive correctness property: for ANY robot position/heading, transforming a local
    offset to world and then back via the SAME heading-frame transform observations use
    (managers/observation/terms/unified.py::ball_pos_b, itself XYZW/w_last=True) must recover the
    original local offset exactly. This is what guarantees "x meters forward" is what the POLICY
    actually observes, for every clip's arbitrary captured orientation, not just a
    coincidentally-aligned one."""
    torch.manual_seed(0)
    for _ in range(20):
        robot_pos = torch.rand(1, 3) * 10 - 5
        q = torch.rand(1, 4) * 2 - 1  # random full 3D orientation (pitch/roll/yaw all nonzero), xyzw
        q = q / q.norm(dim=-1, keepdim=True)
        fake = _FakeMotionCommand(root_pos_w=robot_pos, root_quat_w=q)

        local_xy = torch.rand(1, 2) * 6 - 3
        world_xy = fake.local_xy_to_world(local_xy, torch.tensor([0]))

        # invert exactly as ball_pos_b does: quat_rotate_inverse(yaw_quat(quat, w_last=True), rel, w_last=True)
        rel_w = torch.cat([world_xy - robot_pos[:, :2], torch.zeros(1, 1)], dim=-1)
        recovered = quat_rotate_inverse(yaw_quat(q, w_last=True), rel_w, w_last=True)
        assert torch.allclose(recovered[:, :2], local_xy, atol=1e-4), (recovered, local_xy)


def test_batched_multiple_envs_independent():
    """Different envs with different robot positions/headings must each get their OWN correct
    world placement, not accidentally sharing/broadcasting one env's transform onto another."""
    root_pos_w = torch.tensor([[0.0, 0.0, 0.0], [10.0, 5.0, 0.0]])
    root_quat_w = torch.cat([_IDENTITY_XYZW.unsqueeze(0), _yaw_quat_xyzw(math.pi / 2).unsqueeze(0)], dim=0)
    fake = _FakeMotionCommand(root_pos_w=root_pos_w, root_quat_w=root_quat_w)

    local_xy = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    world_xy = fake.local_xy_to_world(local_xy, torch.tensor([0, 1]))

    assert torch.allclose(world_xy[0], torch.tensor([1.0, 0.0]), atol=1e-5)
    # env 1: rotated 90deg + offset by (10,5) -- must NOT equal env 0's result
    assert not torch.allclose(world_xy[1], world_xy[0] + torch.tensor([10.0, 5.0]), atol=1e-3)


def test_explicit_robot_pose_override_takes_precedence_over_root_pos_w_default():
    """reset()'s real call site passes the robot's ACTUAL, post-noise simulated pose explicitly
    -- must be used INSTEAD of root_pos_w/root_quat_w (the pre-noise clip pose), not ignored."""
    fake = _FakeMotionCommand(
        root_pos_w=torch.zeros(1, 3), root_quat_w=_IDENTITY_XYZW.unsqueeze(0)  # pre-noise: at origin, facing +x
    )
    local_xy = torch.tensor([[2.84, -0.46]])

    # no override -> uses root_pos_w/root_quat_w (pre-noise) defaults
    default_world_xy = fake.local_xy_to_world(local_xy, torch.tensor([0]))
    assert torch.allclose(default_world_xy, local_xy, atol=1e-5)

    # override with a DIFFERENT (post-noise) pose -- result must reflect the override, not the default
    override_pos = torch.tensor([[5.0, 5.0, 0.0]])
    override_quat = _yaw_quat_xyzw(math.pi).unsqueeze(0)  # 180deg -- facing -x
    overridden_world_xy = fake.local_xy_to_world(
        local_xy, torch.tensor([0]), robot_pos_w=override_pos, robot_quat_w=override_quat
    )
    assert not torch.allclose(overridden_world_xy, default_world_xy, atol=1e-3)
    # facing -x, "2.84 forward" should end up in -x direction from (5,5)
    assert overridden_world_xy[0, 0].item() < override_pos[0, 0].item()
