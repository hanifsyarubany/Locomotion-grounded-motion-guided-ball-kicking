"""Verifies the ported FK utility against the ACTUAL npz it needs to correctly reproduce, before
trusting it for the ankle-pitch correction (kick_ankle_pitch_correction.py) built on top of it.
Skips (not fails) if the training data / MJCF assets aren't present in this environment -- this
project's data lives in a sibling checkout, not something every test environment necessarily has.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from holosoma.utils.motion_fk import DEFAULT_MJCF_RELATIVE_PATH, KinematicTree, forward_kinematics, pitch_signal
from holosoma.utils.path import resolve_data_file_path

_NPZ_PATH = (
    Path(__file__).resolve().parents[7]
    / "locomotion_and_motion_tracking/video2robot/data/video_012/npz_holosoma/robot_motion_track_1.npz"
)

pytestmark = pytest.mark.skipif(
    not _NPZ_PATH.exists(), reason=f"training data not present in this checkout ({_NPZ_PATH})"
)


def _load_mjcf_tree() -> KinematicTree:
    mjcf_path = resolve_data_file_path(DEFAULT_MJCF_RELATIVE_PATH)
    if not Path(mjcf_path).exists():
        pytest.skip(f"MJCF not present in this checkout ({mjcf_path})")
    return KinematicTree(mjcf_path)


def test_forward_kinematics_matches_stored_npz_data():
    """The whole ankle-pitch correction depends on this FK port producing EXACTLY what
    pkl_to_npz_holosoma.py's own numpy FK produced when it generated this npz -- if this doesn't
    match, nothing built on top of it can be trusted."""
    npz = np.load(_NPZ_PATH, allow_pickle=True)
    joint_pos = npz["joint_pos"]  # (T, 36) = root_pos(3) + root_quat_wxyz(4) + dof_pos(29)
    body_pos_w_stored = npz["body_pos_w"]
    body_quat_w_stored_wxyz = npz["body_quat_w"]
    body_quat_w_stored_xyzw = body_quat_w_stored_wxyz[:, :, [1, 2, 3, 0]]

    tree = _load_mjcf_tree()

    G1_29DOF_JOINT_NAMES = [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    ]
    joint_col_index = [
        G1_29DOF_JOINT_NAMES.index(name) if name is not None else -1 for name in tree.joint_names
    ]

    root_pos = torch.tensor(joint_pos[:, 0:3], dtype=torch.float32)
    root_quat_wxyz = joint_pos[:, 3:7]
    root_quat_xyzw = torch.tensor(root_quat_wxyz[:, [1, 2, 3, 0]], dtype=torch.float32)
    dof_pos = torch.tensor(joint_pos[:, 7:36], dtype=torch.float32)

    body_pos_fk, body_quat_fk = forward_kinematics(tree, root_pos, root_quat_xyzw, dof_pos, joint_col_index)

    pos_err = (body_pos_fk.numpy() - body_pos_w_stored).__abs__()
    assert pos_err.max() < 1e-3, f"position mismatch, max error {pos_err.max()}"

    dot = np.sum(body_quat_fk.numpy() * body_quat_w_stored_xyzw, axis=-1)
    ang_err_deg = np.degrees(np.arccos(np.clip(np.abs(dot), 0.0, 1.0)) * 2.0)
    assert ang_err_deg.max() < 0.5, f"orientation mismatch, max error {ang_err_deg.max()} deg"


def test_pitch_signal_zero_at_world_horizontal():
    """pitch_signal's own zero-point contract (used throughout this session's live diagnostics):
    identity orientation (local +x aligned with world +x, horizontal) must read exactly 0."""
    identity_xyzw = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    assert torch.allclose(pitch_signal(identity_xyzw), torch.zeros(1), atol=1e-6)


def test_pitch_signal_sign_convention_toe_down_positive():
    """+90deg about local Y (xyzw) rotates local +x to world (0,0,-1) -- toe pointing straight
    down (plantarflexed) -- pitch_signal must read +1.0, matching foot_strike_pitch's own
    convention (positive = toe-down = good)."""
    import math

    q = torch.tensor([[0.0, math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4)]])
    assert torch.allclose(pitch_signal(q), torch.ones(1), atol=1e-6)
