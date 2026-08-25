"""In-memory smoothing of a loaded motion clip's LEADING velocity frames (2026-08-11) -- NOT a
hand-edit of any .npz asset. Opt-in, default off (0 frames = exact no-op).

Why: the prepend (``enable_default_pose_prepend``, 1.0s of interpolation from the static default
standing pose into the clip's own frame 0 -- see MotionCommand._maybe_add_default_pose_transition)
hands off into whatever the clip's opening frames actually contain, and for a video2robot-extracted
clip that boundary is where pose-estimation noise concentrates. Measured on video_012, the only
clip this project currently trains on::

    |joint_vel| per frame:  frame0=18.97  frame1=15.84  frame2=6.46  frame3=0.74
                            median=2.50   p95=6.73   p99=9.77   max=18.97 (at frame 0)

Frame 0 is the single highest-velocity frame of all 250 -- 7.6x the clip's own median, higher even
than the actual kick strike -- decaying to below-median by frame 3. The dominant contributors are
arm joints (right_shoulder_yaw +10.4 rad/s, right_wrist_roll, right_wrist_yaw) while the pelvis
root linear velocity at that same frame is exactly [0,0,0]. Arms whipping at 10+ rad/s about a
motionless pelvis is not plausible captured human motion; it is the signature of pose-estimation
noise at a video boundary.

This is NOT a corrupt velocity CHANNEL that could be fixed by recomputing it: finite-differencing
the stored ``joint_pos`` reproduces the stored ``joint_vel`` closely at the head (13.19 vs 14.26 at
frame 0) and everywhere else (4.29 vs 4.32 at frame 50, 5.77 vs 5.81 at frame 120), so the stored
POSITION data itself jumps over those opening frames. The velocity channel is faithfully reporting
a jump that is really there in the source data.

IMPORTANT, established by live measurement rather than assumed -- the npz's own velocity arrays are
NOT what training ends up seeing. ``correct_kick_foot_ankle_pitch`` (enabled by default per skill)
unconditionally REGENERATES every velocity array for the whole motion from positions
(``torch.gradient`` over joint_pos/body_pos_w plus an so3 derivative for body_ang_vel_w), after the
prepend/append splices. So the live post-setup buffer for video_012 reads, by absolute frame::

    prepend [0,50): 1.4562 flat   (= the gradient of the linearly-interpolated prepend positions)
    frame 50: 9.67   frame 51: 15.84   frame 52: 6.46   frame 53: 0.74   (raw clip frames 0..3)

The junction is therefore a ~6.6x step (1.46 -> 9.67) into a spike that peaks at 15.84 one frame
later. Two consequences that dictate this module's call site: (1) the regeneration already
half-smooths frame 50 itself (its central difference now straddles the prepend boundary, 9.67 vs
the ~14 a one-sided difference on raw data gives), and (2) any velocity-only edit applied BEFORE
that regeneration is silently erased -- an earlier version of this feature ran before the prepend
loop and was verified to be an exact no-op, producing byte-identical buffers on and off. Hence
MotionCommand._maybe_smooth_motion_head_velocities runs LAST for each motion, and offsets past the
prepend to land on real captured frames.

Measured effect at 3 frames on video_012: frame 50 9.67 -> 0.00, frame 51 15.84 -> 3.96, frame 52
6.46 -> 4.85, frame 53 unchanged at 0.74. A small residual step remains at the junction itself
(prepend 1.46 -> 0.00) -- far smaller than the 1.46 -> 9.67 it replaces, but not zero.

Design: ramp the three velocity channels (``joint_vel``, ``body_lin_vel_w``, ``body_ang_vel_w``)
from zero at the clip's frame 0 up to their full authored values by frame ``n_frames``, using the
same raised-cosine (``sin^2``) shape ``kick_ankle_pitch_correction._edge_ramp`` already uses for
its own window edges -- so the ramp itself introduces no hard step at either end. Rationale for
ramping to ZERO specifically (rather than to some clip-derived "calm" value): the clip is being
preceded by a static, zero-velocity default standing pose, so beginning at rest is what the
prepend's own premise already implies. That makes the prepend->clip junction genuinely smooth in
the velocity channel -- the prepend lerps velocity toward a frame-0 value that is now itself zero,
instead of accelerating for a full second toward the noisiest frame in the clip.

DELIBERATELY LEAVES POSITIONS UNTOUCHED. Rewriting ``joint_pos``/``body_pos_w``/``body_quat_w`` at
the head would be editing captured motion content, and the position jump at the head is far less
punishing than the velocity one: only 3 of the 7 motion-tracking reward terms are velocity terms
(motion_global_body_lin_vel / motion_global_body_ang_vel / motion_global_feet_lin_vel), and a
~0.26 rad L2 position step between consecutive frames is a modest tracking target next to an
18.97 rad/s velocity one. The consequence, stated plainly rather than left implicit: within the
smoothed window the reference is kinematically inconsistent (``d(joint_pos)/dt`` no longer equals
the smoothed ``joint_vel``). That inconsistency ALREADY exists across the prepend itself, whose
positions lerp at a constant implied rate while its velocity channel ramps independently -- this
does not introduce a new class of problem, it narrows an existing one. If the position jump turns
out to matter on its own, trimming the clip's first frames at the source is the cleaner fix and
should be done there, not here.

NOT yet validated by a training run.
"""

from __future__ import annotations

import math

import torch


def head_velocity_ramp(n_frames: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Raised-cosine ``sin^2`` ramp of length ``n_frames``: 0.0 at index 0, rising to (but not
    reaching) 1.0 at index ``n_frames`` -- so index ``n_frames`` itself, the first UNSMOOTHED
    frame, is where full authored velocity resumes. Derivative is zero at both ends, so neither
    the start of the ramp nor the handoff back to authored data introduces a kink.

    Same shape as ``kick_ankle_pitch_correction._edge_ramp``'s own ramp-in half, deliberately --
    this project already uses that curve for exactly this "don't put a hard step in a velocity
    channel" purpose.
    """
    if n_frames <= 0:
        return torch.ones(0, device=device, dtype=dtype)
    idx = torch.arange(n_frames, device=device, dtype=dtype)
    return torch.sin(idx / float(n_frames) * (math.pi / 2.0)) ** 2


def smooth_motion_head_velocities(
    *,
    joint_vel: torch.Tensor,
    body_lin_vel_w: torch.Tensor,
    body_ang_vel_w: torch.Tensor,
    n_frames: int,
) -> None:
    """Scale the first ``n_frames`` frames of each velocity channel by ``head_velocity_ramp``,
    IN PLACE (these are live views into MotionLoader's own concatenated buffers -- same in-place
    convention ``correct_kick_foot_ankle_pitch`` uses).

    ``n_frames <= 0`` returns immediately without touching anything -- the exact no-op default.
    ``n_frames`` larger than the available frame count is clamped to that count, so a
    mis-specified value can never index past the end of a short clip.

    Every tensor is expected to be a slice covering ONE motion's own frame span, already sliced by
    the caller (``[motion_start:motion_end]``), so index 0 here is genuinely that motion's own
    first authored frame -- NOT the concatenated buffer's frame 0, and NOT anything inside a
    previously-spliced prepend.
    """
    if n_frames <= 0:
        return

    n_available = int(joint_vel.shape[0])
    n = min(int(n_frames), n_available)
    if n <= 0:
        return

    ramp = head_velocity_ramp(n, device=joint_vel.device, dtype=joint_vel.dtype)

    joint_vel[:n] *= ramp.view(n, *([1] * (joint_vel.dim() - 1)))
    body_lin_vel_w[:n] *= ramp.to(body_lin_vel_w.dtype).view(n, *([1] * (body_lin_vel_w.dim() - 1)))
    body_ang_vel_w[:n] *= ramp.to(body_ang_vel_w.dtype).view(n, *([1] * (body_ang_vel_w.dim() - 1)))
