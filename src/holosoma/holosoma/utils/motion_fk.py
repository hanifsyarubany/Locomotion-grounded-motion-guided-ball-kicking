"""Minimal forward-kinematics utility for G1 motion clips (2026-08-08), built to support an
in-memory kick-foot ankle-pitch correction applied at motion-load time (see
managers/command/terms/wbt.py's MotionCommand.setup()) -- NOT a hand-edit of any .npz asset.

Ported from video2robot/scripts/pkl_to_npz_holosoma.py's own KinematicTree/forward_kinematics
(the exact code that originally computed the training clips' body_pos_w/body_quat_w from
joint_pos), reimplemented in torch/xyzw to reuse holosoma.utils.rotations' own
quat_mul/quat_rotate/quat_from_angle_axis rather than re-deriving quaternion math a second time.
Verified bit-exact against that script's own output (see tests/test_motion_fk.py) before being
trusted for anything new.

Why this needs to exist at all: MotionLoader._load_data_from_motion_npz loads joint_pos and
body_pos_w/body_quat_w as independently-stored arrays -- editing joint_pos alone (e.g. the
ankle-pitch correction this module exists for) does NOT automatically update body_pos_w/
body_quat_w, which is what motion_relative_body_position/orientation_error_exp and every shooting
term actually read. Any edit to joint_pos must re-run FK through this same chain to stay
self-consistent, or the reward/tracking terms will see stale, decoupled body poses.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import torch

from holosoma.utils.rotations import quat_from_angle_axis, quat_mul, quat_rotate

# Relative to this project's own CWD at training launch time (matches the exact convention
# motion_npz fields in configs/*.yaml already use, e.g.
# "../../locomotion_and_motion_tracking/video2robot/data/video_012/npz_holosoma/...") -- points at
# the SAME MJCF pkl_to_npz_holosoma.py used to generate every training clip's body_pos_w/
# body_quat_w in the first place, so a KinematicTree built from it reproduces that npz data
# exactly (verified in tests/test_motion_fk.py), not a possibly-different copy of the G1 model.
DEFAULT_MJCF_RELATIVE_PATH = (
    "../../locomotion_and_motion_tracking/holosoma/src/holosoma_retargeting/"
    "holosoma_retargeting/models/g1/g1_29dof.xml"
)

_IDENTITY_QUAT_XYZW = [0.0, 0.0, 0.0, 1.0]


class KinematicTree:
    """Minimal MJCF kinematic-chain reader: root body + single-axis hinge/fixed children.

    World (index 0, fixed at origin) is prepended in front of the MJCF's own root body (e.g.
    "pelvis"), whose pose is supplied externally per frame -- same convention as the
    pkl_to_npz_holosoma.py original this was ported from."""

    def __init__(self, xml_path: str | Path, device: str = "cpu"):
        self.device = device
        tree = ET.parse(xml_path)
        worldbody = tree.getroot().find("worldbody")
        if worldbody is None:
            raise ValueError(f"No <worldbody> in {xml_path}")
        root_body = worldbody.find("body")
        if root_body is None:
            raise ValueError(f"No root <body> under <worldbody> in {xml_path}")

        self.body_names: list[str] = ["world"]
        self.parent_idx: list[int] = [-1]
        self.local_pos: list[torch.Tensor] = [torch.zeros(3, device=device)]
        self.local_quat: list[torch.Tensor] = [torch.tensor(_IDENTITY_QUAT_XYZW, device=device)]
        self.joint_names: list[str | None] = [None]
        self.joint_axis: list[torch.Tensor | None] = [None]
        # (min, max) rad, or None if this <joint> has no range="..." attribute (unlimited, or a
        # class-default this minimal parser doesn't resolve -- see module docstring's caveat).
        self.joint_range: list[tuple[float, float] | None] = [None]

        self._add_body(root_body, parent_index=0, is_root=True)

    def _add_body(self, el: ET.Element, parent_index: int, is_root: bool = False) -> None:
        name = el.get("name")
        pos = torch.tensor([float(x) for x in el.get("pos", "0 0 0").split()], device=self.device)
        quat_wxyz = [float(x) for x in el.get("quat", "1 0 0 0").split()]  # MJCF quat attr is wxyz
        quat_xyzw = torch.tensor(
            [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], device=self.device
        )

        joint_name = None
        joint_axis = None
        joint_range = None
        if not is_root:
            joints = el.findall("joint")
            if len(joints) == 1:
                joint_name = joints[0].get("name")
                joint_axis = torch.tensor(
                    [float(x) for x in joints[0].get("axis", "0 0 1").split()], device=self.device
                )
                range_attr = joints[0].get("range")
                if range_attr is not None:
                    lo, hi = (float(x) for x in range_attr.split())
                    joint_range = (lo, hi)
            elif len(joints) > 1:
                raise ValueError(
                    f"Unsupported multi-joint body '{name}' (only single-axis hinge joints supported)"
                )
            # len == 0 -> fixed body, no dof

        self.body_names.append(name)
        self.parent_idx.append(parent_index)
        self.local_pos.append(pos)
        self.local_quat.append(quat_xyzw)
        self.joint_names.append(joint_name)
        self.joint_axis.append(joint_axis)
        self.joint_range.append(joint_range)

        idx = len(self.body_names) - 1
        for child in el.findall("body"):
            self._add_body(child, parent_index=idx)

    @property
    def num_bodies(self) -> int:
        return len(self.body_names)


def forward_kinematics(
    tree: KinematicTree,
    root_pos: torch.Tensor,
    root_quat_xyzw: torch.Tensor,
    dof_pos: torch.Tensor,
    joint_col_index: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """root_pos: (T,3), root_quat_xyzw: (T,4), dof_pos: (T, num_dofs) -- all frames computed
    independently (FK has no cross-frame dependency), so any (root_pos, root_quat_xyzw, dof_pos)
    triple sharing a leading T works, including a sliced window rather than a whole clip.

    Returns (body_pos: (T, num_bodies, 3), body_quat_xyzw: (T, num_bodies, 4)), in EXACTLY
    tree.body_names' order -- for a tree built from the same MJCF pkl_to_npz_holosoma.py used,
    this order matches this project's training .npz files' own body_names/body_pos_w/body_quat_w
    storage order (verified in tests/test_motion_fk.py), so the output can be written directly
    into MotionLoader._body_pos_w/_body_quat_w without any further reindexing."""
    num_frames = root_pos.shape[0]
    device = root_pos.device
    body_pos: list[torch.Tensor] = [torch.zeros(1), ] * tree.num_bodies
    body_quat: list[torch.Tensor] = [torch.zeros(1), ] * tree.num_bodies
    body_pos[0] = torch.zeros(num_frames, 3, device=device)
    body_quat[0] = torch.tensor(_IDENTITY_QUAT_XYZW, device=device).expand(num_frames, 4)
    body_pos[1] = root_pos
    body_quat[1] = root_quat_xyzw

    identity_quat = torch.tensor(_IDENTITY_QUAT_XYZW, device=device).expand(num_frames, 4)
    for j in range(2, tree.num_bodies):
        parent = tree.parent_idx[j]
        parent_pos, parent_quat = body_pos[parent], body_quat[parent]

        joint_axis_j = tree.joint_axis[j]
        if joint_axis_j is not None:
            dof_val = dof_pos[:, joint_col_index[j]]
            joint_quat = quat_from_angle_axis(dof_val, joint_axis_j.expand(num_frames, 3), w_last=True)
        else:
            joint_quat = identity_quat

        local_pos_j = tree.local_pos[j].expand(num_frames, 3)
        local_quat_j = tree.local_quat[j].expand(num_frames, 4)

        world_trans = quat_rotate(parent_quat, local_pos_j, w_last=True)
        body_pos[j] = parent_pos + world_trans
        body_quat[j] = quat_mul(quat_mul(parent_quat, local_quat_j, w_last=True), joint_quat, w_last=True)

    return torch.stack(body_pos, dim=1), torch.stack(body_quat, dim=1)


def pitch_signal(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """-[R(q) @ local +x].z -- exactly managers/reward/terms/shooting.py:foot_strike_pitch's own
    kernel (positive = toe-down/plantarflexed, negative = toe-up/dorsiflexed), factored out here
    so the correction solver (below) and that reward term measure the identical quantity."""
    v = torch.zeros(quat_xyzw.shape[:-1] + (3,), device=quat_xyzw.device, dtype=quat_xyzw.dtype)
    v[..., 0] = 1.0
    toe_dir_world = quat_rotate(quat_xyzw.reshape(-1, 4), v.reshape(-1, 3), w_last=True).reshape(quat_xyzw.shape[:-1] + (3,))
    return -toe_dir_world[..., 2]
