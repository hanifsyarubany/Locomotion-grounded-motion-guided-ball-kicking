"""Unit tests for increment 4's ball-geometry entry-search features (2026-08-13,
mid_episode_kick_entry_ball_fixed -- MotionCommand._live_entry_features's optional 2 extra
columns, D2b closing the "ball doesn't move at deploy" training-scaffold gap). See
_build_entry_search_table's own docstring for the full rationale.

Isolated via a lightweight fake object providing the handful of attributes/properties
_live_entry_features actually reads, and borrowing the REAL, unbound method from MotionCommand --
same pattern as test_ref_anchor.py / test_ref_blend.py.
"""

from __future__ import annotations

import math

import torch

from holosoma.managers.command.terms.wbt import MotionCommand


def _yaw_quat_xyzw(yaw: float) -> torch.Tensor:
    return torch.tensor([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])


_IDENTITY_XYZW = torch.tensor([0.0, 0.0, 0.0, 1.0])


class _FakeSimulator:
    def __init__(
        self,
        rigid_body_pos: torch.Tensor,
        rigid_body_rot: torch.Tensor,
        rigid_body_vel: torch.Tensor,
        rigid_body_ang_vel: torch.Tensor,
        all_root_states: torch.Tensor,
    ):
        self._rigid_body_pos = rigid_body_pos
        self._rigid_body_rot = rigid_body_rot
        self._rigid_body_vel = rigid_body_vel
        self._rigid_body_ang_vel = rigid_body_ang_vel
        self.all_root_states = all_root_states


class _FakeEnv:
    def __init__(self, simulator: _FakeSimulator):
        self.simulator = simulator


class _FakeMotionCommand:
    """Duck-typed stand-in for a real MotionCommand: provides just enough state for the real
    _live_entry_features method, borrowed unbound below, to run unmodified. Body layout: index 0
    = root, index 1 = ref body (torso), indices 2/3 = left/right ankle."""

    _live_entry_features = MotionCommand._live_entry_features
    live_ball_pos_w = MotionCommand.live_ball_pos_w

    def __init__(
        self,
        env: _FakeEnv,
        *,
        ref_body_index: int = 1,
        left_ankle_idx: int = 2,
        right_ankle_idx: int = 3,
        use_ball_geometry: bool = False,
        ball_indices_in_simulator: torch.Tensor | None = None,
    ):
        self._env = env
        self.ref_body_index = ref_body_index
        self._entry_search_left_ankle_idx = left_ankle_idx
        self._entry_search_right_ankle_idx = right_ankle_idx
        self._entry_search_use_ball_geometry = use_ball_geometry
        self.ball_indices_in_simulator = (
            ball_indices_in_simulator
            if ball_indices_in_simulator is not None
            else torch.arange(env.simulator.all_root_states.shape[0])
        )


def _make_env(
    num_envs: int,
    num_bodies: int,
    *,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    ball_pos: torch.Tensor,
) -> _FakeEnv:
    rigid_body_pos = torch.zeros(num_envs, num_bodies, 3)
    rigid_body_pos[:, 0, :] = root_pos
    rigid_body_rot = _IDENTITY_XYZW.repeat(num_envs, num_bodies, 1).clone()
    rigid_body_rot[:, 0, :] = root_quat
    rigid_body_vel = torch.zeros(num_envs, num_bodies, 3)
    rigid_body_ang_vel = torch.zeros(num_envs, num_bodies, 3)
    all_root_states = torch.zeros(num_envs, 13)
    all_root_states[:, :3] = ball_pos
    sim = _FakeSimulator(rigid_body_pos, rigid_body_rot, rigid_body_vel, rigid_body_ang_vel, all_root_states)
    return _FakeEnv(sim)


def test_ball_geometry_columns_absent_when_disabled():
    env = _make_env(1, 4, root_pos=torch.zeros(1, 3), root_quat=_IDENTITY_XYZW.unsqueeze(0), ball_pos=torch.tensor([[2.0, 1.0, 0.1]]))
    fake = _FakeMotionCommand(env, use_ball_geometry=False)
    feats = fake._live_entry_features(torch.tensor([0]))
    assert feats.shape == (1, 5)


def test_ball_geometry_columns_present_when_enabled():
    env = _make_env(1, 4, root_pos=torch.zeros(1, 3), root_quat=_IDENTITY_XYZW.unsqueeze(0), ball_pos=torch.tensor([[2.0, 1.0, 0.1]]))
    fake = _FakeMotionCommand(env, use_ball_geometry=True)
    feats = fake._live_entry_features(torch.tensor([0]))
    assert feats.shape == (1, 7)


def test_ball_geometry_offset_identity_heading_matches_world_delta():
    """Root at the world origin, facing world +x (identity yaw): the local (forward, lateral)
    offset to the ball must equal its raw world (x, y) delta from the root, unrotated."""
    env = _make_env(
        1, 4, root_pos=torch.zeros(1, 3), root_quat=_IDENTITY_XYZW.unsqueeze(0), ball_pos=torch.tensor([[2.0, 1.0, 0.1]])
    )
    fake = _FakeMotionCommand(env, use_ball_geometry=True)
    feats = fake._live_entry_features(torch.tensor([0]))
    assert torch.allclose(feats[0, 5:7], torch.tensor([2.0, 1.0]), atol=1e-5)


def test_ball_geometry_offset_rotates_with_root_heading():
    """Root facing world -y (90deg clockwise from +x): a ball 1m along the root's own forward
    direction shows up at world (0, -1, *) -- the offset must report back as (1, 0) in the root's
    own local frame, not the raw world delta."""
    root_pos = torch.zeros(1, 3)
    root_quat = _yaw_quat_xyzw(-math.pi / 2).unsqueeze(0)
    ball_pos = torch.tensor([[0.0, -1.0, 0.1]])
    env = _make_env(1, 4, root_pos=root_pos, root_quat=root_quat, ball_pos=ball_pos)
    fake = _FakeMotionCommand(env, use_ball_geometry=True)
    feats = fake._live_entry_features(torch.tensor([0]))
    assert torch.allclose(feats[0, 5:7], torch.tensor([1.0, 0.0]), atol=1e-5)


def test_ball_geometry_offset_anchored_to_root_not_ref_body():
    """Root and ref body (torso) at DELIBERATELY different poses -- the ball offset must be
    computed from the ROOT (body 0), matching ball_rel_at_frame's own root-anchored convention,
    NOT ref_body_index (torso) the way the 5 gait features are."""
    rigid_body_pos = torch.zeros(1, 4, 3)
    rigid_body_pos[0, 0] = torch.tensor([5.0, 5.0, 0.7])  # root, far from the origin
    rigid_body_pos[0, 1] = torch.tensor([0.0, 0.0, 1.2])  # ref body (torso) -- deliberately elsewhere
    rigid_body_rot = _IDENTITY_XYZW.repeat(1, 4, 1).clone()
    ball_pos = torch.tensor([[6.0, 5.0, 0.1]])  # 1m along root's local +x
    all_root_states = torch.zeros(1, 13)
    all_root_states[:, :3] = ball_pos
    sim = _FakeSimulator(rigid_body_pos, rigid_body_rot, torch.zeros(1, 4, 3), torch.zeros(1, 4, 3), all_root_states)
    env = _FakeEnv(sim)
    fake = _FakeMotionCommand(env, ref_body_index=1, use_ball_geometry=True)
    feats = fake._live_entry_features(torch.tensor([0]))
    assert torch.allclose(feats[0, 5:7], torch.tensor([1.0, 0.0]), atol=1e-5)


def test_ball_geometry_per_env_independent():
    root_pos = torch.zeros(2, 3)
    root_quat = _IDENTITY_XYZW.unsqueeze(0).repeat(2, 1)
    ball_pos = torch.tensor([[1.0, 0.0, 0.1], [0.0, 3.0, 0.1]])
    env = _make_env(2, 4, root_pos=root_pos, root_quat=root_quat, ball_pos=ball_pos)
    fake = _FakeMotionCommand(env, use_ball_geometry=True)
    feats = fake._live_entry_features(torch.tensor([0, 1]))
    assert torch.allclose(feats[:, 5:7], torch.tensor([[1.0, 0.0], [0.0, 3.0]]), atol=1e-5)
