"""Self-calibrating kick-foot ankle-pitch correction, applied in-memory to a loaded motion clip's
joint_pos/body_pos_w/body_quat_w (2026-08-08) -- NOT a hand-edit of any .npz asset, and NOT gated
behind a new yaml field (user-requested: no skills.yaml changes needed to activate).

Why: a live diagnostic this session found the kick clip's own authored ankle trajectory is
genuinely toe-up (dorsiflexed, sole presented to the ball) exactly where real contact happens --
median -19.7deg, only 9.1% of contacts landing with the reference already toe-down at that exact
frame (see diagnose_contact_timing_vs_clip.py's findings). This makes foot_strike_pitch's own
reward unwinnable (it asks for absolute toe-down while motion tracking pulls, at 4x the weight,
toward matching this same toe-up reference) and plausibly contributes to weak/sole-first strikes.

Design: rather than a hand-tuned "+40deg" magic constant (fragile, clip-specific, would need
re-tuning per clip), this SELF-CALIBRATES per motion: for each frame in [strike_start_idx,
stand_start_idx), numerically solves (bisection, via the verified FK port in utils/motion_fk.py)
for the MINIMAL kick-foot ankle_pitch DOF delta needed to bring that frame's world-frame pitch up
to TARGET_MIN_PITCH_DEG. Frames already at/above target get delta=0 -- untouched. This generalizes
to any clip (a clip that's already fine gets ~zero correction) without a per-clip tuned constant,
and preserves whatever's already good in the authored trajectory instead of overwriting it.

After solving, body_pos_w/body_quat_w and all velocity arrays are regenerated for the WHOLE
motion (not just the corrected window) via the same FK/finite-difference machinery, so nothing
reading Cartesian pose or velocity ever sees a joint_pos edit that hasn't been propagated --
see MotionCommand.setup()'s call site for exactly where this runs (after strike_start_idx/
stand_start_idx are finalized, so index math is done once, post-prepend-splice, correctly)."""

from __future__ import annotations

import math

import torch
from loguru import logger

from holosoma.utils.motion_fk import KinematicTree, forward_kinematics, pitch_signal

# Matches video2robot/scripts/pkl_to_npz_holosoma.py's own G1_29DOF_JOINT_NAMES constant exactly
# -- the fixed joint-column convention every training .npz produced by that pipeline uses (its own
# comment: "matches ... holosoma's g1_29dof.xml joint order, which is also ... RobotRetargeter's"
# dof_pos column order"). This is the RAW npz storage ordering (MotionLoader._joint_pos, before
# the .joint_pos property's own robot_joint_names reindexing), not this project's own dof_names
# order -- both exist and are NOT interchangeable, see MotionLoader._get_index_of_a_in_b.
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

# No yaml field for these -- user-requested this ship as an automatic, self-calibrating
# correction, not an opt-in knob. Values are deliberately conservative (see each comment).
TARGET_MIN_PITCH_DEG = 10.0  # a genuine plantarflexed margin, not just "not toe-up" (0deg)
EDGE_TAPER_FRAMES = 5  # smooth ramp in/out at the correction window's own boundaries
BISECTION_ITERS = 40  # 2^-40 rad precision, comically overkill but the cost is negligible
BISECTION_MAX_DELTA_RAD = math.radians(90.0)  # generous upper bound; actual deltas measured ~30-50deg


def _edge_ramp(n_total: int, taper: int, device: torch.device) -> torch.Tensor:
    """[0..1..0]-shaped weight over n_total frames: raised-cosine ramp up over the first `taper`
    frames, flat 1.0 in the interior, raised-cosine ramp down over the last `taper` -- avoids a
    hard step in the correction (and therefore in the recomputed joint_vel) at the window edges."""
    w = torch.ones(n_total, device=device)
    taper = min(taper, n_total // 2)
    if taper > 0:
        t = torch.linspace(0.0, math.pi / 2.0, taper, device=device)
        ramp_in = torch.sin(t) ** 2  # smooth 0 -> 1
        w[:taper] = ramp_in
        w[n_total - taper :] = ramp_in.flip(0)
    return w


def solve_ankle_pitch_correction(
    tree: KinematicTree,
    root_pos: torch.Tensor,
    root_quat_xyzw: torch.Tensor,
    dof_pos: torch.Tensor,
    joint_col_index: list[int],
    ankle_dof_col: int,
    kick_foot_body_idx: int,
    window_start: int,
    window_end: int,
    target_pitch_deg: float = TARGET_MIN_PITCH_DEG,
    edge_taper_frames: int = EDGE_TAPER_FRAMES,
    num_iters: int = BISECTION_ITERS,
    joint_max_rad: float | None = None,
) -> torch.Tensor:
    """Returns a (T,) delta to add to dof_pos[:, ankle_dof_col] -- zero outside
    [window_start, window_end), self-calibrated (via bisection) to bring each in-window frame's
    world-frame pitch_signal up to target_pitch_deg, tapered to zero at the window's own edges.
    T = dof_pos.shape[0] (the caller's full per-motion frame count); vectorized across all T
    frames at once (FK has no cross-frame dependency), not a per-frame Python loop.

    joint_max_rad: the ankle_pitch joint's own physical upper limit (e.g. ~+0.5236 rad / +30deg
    for G1) -- caps the bisection's search per-frame at (joint_max_rad - that frame's OWN current
    dof value), which varies per frame. Without this, the bisection solves the target in the
    PURE KINEMATIC sense only, and a frame that's already sitting close to the joint's physical
    ceiling can get a "solution" the real joint could never reach (measured live: an unclamped
    solve pushed one frame's dof to 95.8deg against a ~+30deg physical limit) -- a target that's
    unreachable given how close to the ceiling this frame already starts settles for however close
    the joint's own limit can get it, not an unconstrained (and physically meaningless) delta."""
    num_frames = dof_pos.shape[0]
    device = dof_pos.device
    target = math.sin(math.radians(target_pitch_deg))

    def signal_at(delta: torch.Tensor) -> torch.Tensor:
        dof = dof_pos.clone()
        dof[:, ankle_dof_col] = dof[:, ankle_dof_col] + delta
        _, body_quat = forward_kinematics(tree, root_pos, root_quat_xyzw, dof, joint_col_index)
        return pitch_signal(body_quat[:, kick_foot_body_idx])

    lo = torch.zeros(num_frames, device=device)
    hi = torch.full((num_frames,), BISECTION_MAX_DELTA_RAD, device=device)
    if joint_max_rad is not None:
        per_frame_headroom = (joint_max_rad - dof_pos[:, ankle_dof_col]).clamp(min=0.0)
        hi = torch.minimum(hi, per_frame_headroom)

    current = signal_at(lo)
    needs_correction = current < target
    if needs_correction.any():
        for _ in range(num_iters):
            mid = (lo + hi) / 2.0
            below = signal_at(mid) < target
            lo = torch.where(below, mid, lo)
            hi = torch.where(below, hi, mid)

    # hi is now either the (bisection-converged) delta that reaches target, or -- if even the
    # joint's own physical ceiling (hi's initial value, when clamped above) falls short of
    # target -- the ceiling itself, since `below` stays True through every iteration and hi never
    # moves off its starting value. Either way it's the best physically-achievable delta.
    solved_delta = torch.where(needs_correction, hi, torch.zeros(num_frames, device=device))

    ramp = torch.zeros(num_frames, device=device)
    win_len = window_end - window_start
    if win_len > 0:
        ramp[window_start:window_end] = _edge_ramp(win_len, edge_taper_frames, device)

    return solved_delta * ramp


def correct_kick_foot_ankle_pitch(
    joint_pos: torch.Tensor,
    body_pos_w: torch.Tensor,
    body_quat_w: torch.Tensor,
    joint_vel: torch.Tensor,
    body_lin_vel_w: torch.Tensor,
    body_ang_vel_w: torch.Tensor,
    dt: float,
    kick_foot: str,
    strike_start_idx: int,
    stand_start_idx: int,
    tree: KinematicTree,
) -> None:
    """Mutates joint_pos/body_pos_w/body_quat_w/joint_vel/body_lin_vel_w/body_ang_vel_w IN PLACE
    for ONE motion's own frame range (caller passes exactly that slice, e.g.
    self.motion._joint_pos[motion_start_idx:motion_end_idx] -- a view, so in-place writes here
    propagate back into the underlying (Multi)MotionLoader storage without the caller needing to
    write anything back itself).

    kick_foot: "left" or "right". strike_start_idx/stand_start_idx: THIS MOTION'S OWN-RELATIVE
    frame indices into the slice already passed in (i.e. if the caller sliced
    [motion_start_idx:motion_end_idx], these must already be motion_start_idx-subtracted -- see
    the call site in wbt.py for the exact conversion)."""
    device = joint_pos.device
    ankle_dof_col = G1_29DOF_JOINT_NAMES.index(f"{kick_foot}_ankle_pitch_joint")
    kick_foot_body_idx = tree.body_names.index(f"{kick_foot}_ankle_roll_link")
    ankle_joint_name = f"{kick_foot}_ankle_pitch_joint"
    ankle_tree_idx = tree.joint_names.index(ankle_joint_name)
    ankle_range = tree.joint_range[ankle_tree_idx]
    joint_max_rad = ankle_range[1] if ankle_range is not None else None
    if joint_max_rad is None:
        logger.warning(
            f"kick_ankle_pitch_correction: {ankle_joint_name} has no range=\"...\" in the MJCF -- "
            "correction will run UNCONSTRAINED by any physical joint limit. Verify this is intended."
        )
    joint_col_index = [
        G1_29DOF_JOINT_NAMES.index(name) if name is not None else -1 for name in tree.joint_names
    ]

    root_pos = body_pos_w[:, 1]
    root_quat_xyzw = body_quat_w[:, 1]

    delta = solve_ankle_pitch_correction(
        tree, root_pos, root_quat_xyzw, joint_pos, joint_col_index,
        ankle_dof_col, kick_foot_body_idx, strike_start_idx, stand_start_idx,
        joint_max_rad=joint_max_rad,
    )
    if not torch.any(delta != 0.0):
        return

    max_delta_deg = math.degrees(delta.abs().max().item())
    logger.info(
        f"kick_ankle_pitch_correction: {kick_foot} ankle_pitch corrected, max delta "
        f"{max_delta_deg:.1f}deg over frames [{strike_start_idx}, {stand_start_idx})"
    )

    joint_pos[:, ankle_dof_col] = joint_pos[:, ankle_dof_col] + delta

    new_body_pos_w, new_body_quat_w = forward_kinematics(
        tree, root_pos, root_quat_xyzw, joint_pos, joint_col_index
    )
    body_pos_w.copy_(new_body_pos_w)
    body_quat_w.copy_(new_body_quat_w)

    # Recompute velocities via the same central-difference scheme
    # pkl_to_npz_holosoma.py's own convert_pkl_to_npz used (np.gradient / so3_derivative) --
    # editing joint_pos without this would leave joint_vel/body_*_vel_w stale and inconsistent
    # with the new positions, which motion_global_body_lin_vel/ang_vel and the feet-velocity
    # tracking terms read directly.
    new_joint_vel = torch.gradient(joint_pos, spacing=dt, dim=0)[0]
    joint_vel.copy_(new_joint_vel)
    new_body_lin_vel_w = torch.gradient(body_pos_w, spacing=dt, dim=0)[0]
    body_lin_vel_w.copy_(new_body_lin_vel_w)
    body_ang_vel_w.copy_(_angular_velocity_xyzw(body_quat_w, dt))


def _angular_velocity_xyzw(quat_seq_xyzw: torch.Tensor, dt: float) -> torch.Tensor:
    """Central-difference world-frame angular velocity from a (T, B, 4) xyzw quat sequence --
    same central-difference convention as pkl_to_npz_holosoma.py's so3_derivative_wxyz, ported to
    xyzw/torch. Endpoints repeat the nearest interior value (matches np.gradient's own edge
    behavior used by the original script for the linear velocities).

    quat_to_angle_axis returns (angle, axis*angle) -- NOT (angle, unit_axis) despite the name; the
    second element is already the full axis-angle VECTOR (direction=axis, magnitude=angle), so
    dividing IT (not the first, scalar element) by 2*dt gives the angular velocity vector directly."""
    from holosoma.utils.rotations import quat_conjugate, quat_mul, quat_to_angle_axis

    q_prev = quat_seq_xyzw[:-2]
    q_next = quat_seq_xyzw[2:]
    vec_shape = q_prev.shape[:-1] + (3,)  # (T-2, B, 3)
    q_rel = quat_mul(q_next.reshape(-1, 4), quat_conjugate(q_prev.reshape(-1, 4), w_last=True), w_last=True)
    axis_angle_vec = quat_to_angle_axis(q_rel)[1]  # ((T-2)*B, 3), magnitude=angle
    omega = axis_angle_vec.reshape(vec_shape) / (2.0 * dt)
    return torch.cat([omega[:1], omega, omega[-1:]], dim=0)
