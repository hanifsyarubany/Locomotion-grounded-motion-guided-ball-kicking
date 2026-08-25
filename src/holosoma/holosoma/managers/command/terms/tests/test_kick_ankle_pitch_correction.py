"""Unit tests for the self-calibrating ankle-pitch correction, using a small synthetic
kinematic tree (not the real 51-body G1 MJCF) so these run fast and don't depend on training
data being present -- test_motion_fk.py already covers the FK port's fidelity against the real
asset; these tests cover the correction/bisection/edge-ramp logic in isolation.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from holosoma.managers.command.terms.kick_ankle_pitch_correction import (
    _angular_velocity_xyzw,
    _edge_ramp,
    correct_kick_foot_ankle_pitch,
    solve_ankle_pitch_correction,
)
from holosoma.utils.motion_fk import DEFAULT_MJCF_RELATIVE_PATH, KinematicTree, forward_kinematics, pitch_signal
from holosoma.utils.path import resolve_data_file_path


class _FakeTree:
    """Two-body chain: world(0) -> root(1, no joint, matches KinematicTree's own convention of
    supplying root pose externally) -> foot(2, single hinge about local Y, axis matching the
    real ankle_pitch_joint's own "0 1 0" convention) -- just enough structure for
    forward_kinematics to exercise the same code path a real tree would, with a trivial,
    hand-checkable geometry."""

    def __init__(self):
        self.body_names = ["world", "root", "foot"]
        self.parent_idx = [-1, 0, 1]
        self.local_pos = [torch.zeros(3), torch.zeros(3), torch.zeros(3)]
        self.local_quat = [torch.tensor([0.0, 0.0, 0.0, 1.0])] * 3
        self.joint_names = [None, None, "foot_pitch_joint"]
        self.joint_axis = [None, None, torch.tensor([0.0, 1.0, 0.0])]

    @property
    def num_bodies(self) -> int:
        return len(self.body_names)


def test_solve_ankle_pitch_correction_leaves_already_good_frames_untouched():
    """A frame whose pitch is already at/above target must get delta=0 -- the correction should
    never touch frames that don't need it, preserving whatever's already authored correctly."""
    tree = _FakeTree()
    num_frames = 3
    root_pos = torch.zeros(num_frames, 3)
    root_quat_xyzw = torch.tensor([[0.0, 0.0, 0.0, 1.0]] * num_frames)
    dof_pos = torch.zeros(num_frames, 1)
    # frame 0: dof=+60deg -> pitch well above any reasonable target already.
    dof_pos[0, 0] = math.radians(60.0)
    joint_col_index = [-1, -1, 0]

    delta = solve_ankle_pitch_correction(
        tree, root_pos, root_quat_xyzw, dof_pos, joint_col_index,
        ankle_dof_col=0, kick_foot_body_idx=2, window_start=0, window_end=num_frames,
        target_pitch_deg=10.0, edge_taper_frames=0,
    )
    assert delta[0].item() == 0.0, "already-good frame must not be touched"


def test_solve_ankle_pitch_correction_reaches_target_for_deficient_frame():
    """A frame that starts toe-up must be corrected up to (approximately) the target pitch --
    verifies the bisection actually converges, not just that it changes something."""
    tree = _FakeTree()
    num_frames = 1
    root_pos = torch.zeros(num_frames, 3)
    root_quat_xyzw = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    dof_pos = torch.zeros(num_frames, 1)
    dof_pos[0, 0] = math.radians(-40.0)  # toe-up
    joint_col_index = [-1, -1, 0]

    target_deg = 10.0
    delta = solve_ankle_pitch_correction(
        tree, root_pos, root_quat_xyzw, dof_pos, joint_col_index,
        ankle_dof_col=0, kick_foot_body_idx=2, window_start=0, window_end=num_frames,
        target_pitch_deg=target_deg, edge_taper_frames=0,
    )
    corrected_dof = dof_pos.clone()
    corrected_dof[:, 0] += delta
    _, body_quat = forward_kinematics(tree, root_pos, root_quat_xyzw, corrected_dof, joint_col_index)
    achieved_signal = pitch_signal(body_quat[:, 2])
    target_signal = math.sin(math.radians(target_deg))
    assert torch.allclose(achieved_signal, torch.full((1,), target_signal), atol=1e-4)


def test_solve_ankle_pitch_correction_zero_outside_window():
    """Frames outside [window_start, window_end) must get exactly zero correction regardless of
    how deficient their pitch is -- the correction must not leak outside the strike window."""
    tree = _FakeTree()
    num_frames = 5
    root_pos = torch.zeros(num_frames, 3)
    root_quat_xyzw = torch.tensor([[0.0, 0.0, 0.0, 1.0]] * num_frames)
    dof_pos = torch.full((num_frames, 1), math.radians(-40.0))  # every frame badly toe-up
    joint_col_index = [-1, -1, 0]

    delta = solve_ankle_pitch_correction(
        tree, root_pos, root_quat_xyzw, dof_pos, joint_col_index,
        ankle_dof_col=0, kick_foot_body_idx=2, window_start=2, window_end=4,
        target_pitch_deg=10.0, edge_taper_frames=0,
    )
    assert delta[0].item() == 0.0
    assert delta[1].item() == 0.0
    assert delta[4].item() == 0.0
    assert delta[2].item() > 0.0
    assert delta[3].item() > 0.0


def test_solve_ankle_pitch_correction_respects_joint_max_rad():
    """A frame that's ALREADY close to the joint's own physical ceiling must not get pushed past
    it, even if the target pitch would otherwise require more delta than the joint can supply --
    regression test for a live bug this session: an unclamped solve pushed a real frame's ankle
    DOF to 95.8deg against G1's real ~+30deg physical limit."""
    tree = _FakeTree()
    num_frames = 1
    root_pos = torch.zeros(num_frames, 3)
    root_quat_xyzw = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    dof_pos = torch.zeros(num_frames, 1)
    dof_pos[0, 0] = math.radians(25.0)  # already close to a hypothetical +30deg ceiling
    joint_col_index = [-1, -1, 0]
    joint_max_rad = math.radians(30.0)

    # target_pitch_deg=60 is deliberately unreachable from dof=25deg while staying <= 30deg.
    delta = solve_ankle_pitch_correction(
        tree, root_pos, root_quat_xyzw, dof_pos, joint_col_index,
        ankle_dof_col=0, kick_foot_body_idx=2, window_start=0, window_end=num_frames,
        target_pitch_deg=60.0, edge_taper_frames=0, joint_max_rad=joint_max_rad,
    )
    resulting_dof = dof_pos[0, 0].item() + delta[0].item()
    assert resulting_dof <= joint_max_rad + 1e-4, (
        f"resulting dof {math.degrees(resulting_dof):.1f}deg exceeds the joint's own "
        f"{math.degrees(joint_max_rad):.1f}deg physical limit"
    )
    assert math.isclose(resulting_dof, joint_max_rad, abs_tol=1e-3), (
        "an unreachable target should settle at exactly the joint's own ceiling, not short of it"
    )


def test_solve_ankle_pitch_correction_reachable_target_ignores_joint_max_rad():
    """When the target IS reachable within the joint's own range, the joint_max_rad clamp must
    not interfere with the normal bisection result (only kicks in when actually needed)."""
    tree = _FakeTree()
    num_frames = 1
    root_pos = torch.zeros(num_frames, 3)
    root_quat_xyzw = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    dof_pos = torch.zeros(num_frames, 1)
    dof_pos[0, 0] = math.radians(-20.0)
    joint_col_index = [-1, -1, 0]

    target_deg = 10.0
    delta = solve_ankle_pitch_correction(
        tree, root_pos, root_quat_xyzw, dof_pos, joint_col_index,
        ankle_dof_col=0, kick_foot_body_idx=2, window_start=0, window_end=num_frames,
        target_pitch_deg=target_deg, edge_taper_frames=0, joint_max_rad=math.radians(30.0),
    )
    corrected_dof = dof_pos.clone()
    corrected_dof[:, 0] += delta
    _, body_quat = forward_kinematics(tree, root_pos, root_quat_xyzw, corrected_dof, joint_col_index)
    achieved_signal = pitch_signal(body_quat[:, 2])
    target_signal = math.sin(math.radians(target_deg))
    assert torch.allclose(achieved_signal, torch.full((1,), target_signal), atol=1e-4)


def test_edge_ramp_shape():
    """[0..1..0] raised-cosine shape: starts at (near-)0, reaches exactly 1.0 in the interior,
    ends at (near-)0, monotonically increasing then decreasing -- no overshoot beyond [0,1]."""
    ramp = _edge_ramp(n_total=10, taper=3, device=torch.device("cpu"))
    assert ramp.shape == (10,)
    assert torch.isclose(ramp[5], torch.tensor(1.0))  # deep interior, fully active
    assert (ramp >= 0.0).all() and (ramp <= 1.0).all()
    assert ramp[0] < ramp[1] < ramp[2]  # monotonic ramp-in
    assert ramp[-1] < ramp[-2] < ramp[-3]  # monotonic ramp-out


def _load_real_g1_tree() -> KinematicTree:
    mjcf_path = resolve_data_file_path(DEFAULT_MJCF_RELATIVE_PATH)
    if not Path(mjcf_path).exists():
        pytest.skip(f"MJCF not present in this checkout ({mjcf_path})")
    return KinematicTree(mjcf_path)


def test_correct_kick_foot_ankle_pitch_updates_body_quat_consistently():
    """End-to-end against the REAL G1 tree (name resolution -- "right_ankle_pitch_joint"/
    "right_ankle_roll_link" -- is coupled to the real G1 naming convention, so this needs the
    real MJCF, unlike solve_ankle_pitch_correction's pure-algorithm tests above): after
    correction, body_quat_w for the kick foot must reflect the CORRECTED joint_pos, not the
    original -- catches a forgotten FK-regeneration step (the whole reason this module exists
    over a bare joint_pos edit), and joint_vel/body_*_vel_w must actually change too."""
    tree = _load_real_g1_tree()
    num_frames = 20
    num_dofs = 29
    joint_pos = torch.zeros(num_frames, num_dofs)
    right_ankle_pitch_col = 10  # G1_29DOF_JOINT_NAMES.index("right_ankle_pitch_joint")
    joint_pos[:, right_ankle_pitch_col] = math.radians(-40.0)  # toe-up throughout

    from holosoma.managers.command.terms.kick_ankle_pitch_correction import G1_29DOF_JOINT_NAMES

    joint_col_index = [
        G1_29DOF_JOINT_NAMES.index(name) if name is not None else -1 for name in tree.joint_names
    ]
    root_pos = torch.zeros(num_frames, 3)
    root_quat_xyzw = torch.tensor([[0.0, 0.0, 0.0, 1.0]] * num_frames)
    body_pos_w, body_quat_w = forward_kinematics(tree, root_pos, root_quat_xyzw, joint_pos, joint_col_index)
    joint_vel = torch.zeros(num_frames, num_dofs)
    body_lin_vel_w = torch.zeros_like(body_pos_w)
    body_ang_vel_w = torch.zeros_like(body_pos_w)

    right_ankle_roll_idx = tree.body_names.index("right_ankle_roll_link")
    pitch_before = pitch_signal(body_quat_w[:, right_ankle_roll_idx]).clone()

    correct_kick_foot_ankle_pitch(
        joint_pos=joint_pos,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        joint_vel=joint_vel,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
        dt=1.0 / 50.0,
        kick_foot="right",
        strike_start_idx=5,
        stand_start_idx=15,
        tree=tree,
    )

    pitch_after = pitch_signal(body_quat_w[:, right_ankle_roll_idx])
    assert (pitch_after[7:13] > pitch_before[7:13] + 0.1).all(), "deep-interior window frames must improve substantially"
    assert torch.allclose(pitch_after[0], pitch_before[0], atol=1e-5), "frame outside window must be untouched"
    assert not torch.allclose(joint_vel, torch.zeros_like(joint_vel)), "joint_vel must be recomputed, not left at zero"
    assert not torch.allclose(body_ang_vel_w, torch.zeros_like(body_ang_vel_w)), "body_ang_vel_w must be recomputed"

    # Regression guard for the live joint-limit bug this session: the corrected DOF value must
    # never exceed the real MJCF's own right_ankle_pitch_joint range, even though this test's
    # -40deg starting point + a +10deg target would need an unreachable ~68deg delta if unclamped.
    ankle_range = tree.joint_range[tree.joint_names.index("right_ankle_pitch_joint")]
    assert ankle_range is not None, "test assumes the real MJCF specifies range=... directly"
    joint_pos_after = joint_pos[7:13, right_ankle_pitch_col]
    assert (joint_pos_after <= ankle_range[1] + 1e-4).all(), (
        f"corrected DOF value(s) {joint_pos_after.tolist()} exceed the joint's own "
        f"{math.degrees(ankle_range[1]):.1f}deg physical limit"
    )


def test_angular_velocity_matches_known_constant_rotation_rate():
    """Sanity check the hand-derived central-difference angular-velocity formula against a case
    with a known-exact answer: a body rotating about world +Z at a constant rate omega_true
    should read back omega_true (within numerical tolerance) at every interior frame."""
    from holosoma.utils.rotations import quat_from_angle_axis

    dt = 0.01
    omega_true = 2.0  # rad/s about +Z
    num_frames = 20
    t = torch.arange(num_frames, dtype=torch.float32) * dt
    angle = omega_true * t
    axis = torch.zeros(num_frames, 3)
    axis[:, 2] = 1.0
    quat_seq = quat_from_angle_axis(angle, axis, w_last=True).unsqueeze(1)  # (T, 1, 4) -- one "body"

    omega = _angular_velocity_xyzw(quat_seq, dt)
    interior = omega[2:-2, 0]  # avoid the repeated-endpoint frames
    expected = torch.zeros_like(interior)
    expected[:, 2] = omega_true
    assert torch.allclose(interior, expected, atol=1e-2), f"expected ~{omega_true} rad/s about Z, got {interior}"
