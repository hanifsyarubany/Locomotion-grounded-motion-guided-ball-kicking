#!/usr/bin/env python3
"""Headless scan: for one onnx checkpoint, teleport the robot DIRECTLY into a pose sampled from
the AUTHORED reference clip (idealised retargeted mocap kinematics -- no contact forces, no
tracking error), start it in ordinary LOCOMOTION mode (never triggers a kick), and report whether
it recovers to a stable stance without falling. This is the sim2sim analogue of training's
``kick_state_init_prob`` reset-seeding mechanism (``UnifiedManager._maybe_kick_state_init``,
``MotionCommand.sample_authored_clip_frames`` / ``teleport_to_frames`` in
``managers/command/terms/wbt.py``), and the "reference-clip" half of the train x test matrix Table
XI(b) reports -- see that mechanism's own docstring for the full training-time rationale.

Deliberately the SIMPLER sibling of mujoco_kick_loco_flip_scan.py / mujoco_loco_to_kick_handoff_scan.py:
no settle phase, no "[TRIGGER_KICK]"/"[RETURN_TO_LOCO]" scripted-command dance. UnifiedLocoKickPolicy.
reset() already sets ``self.task_mode = _TASK_LOCOMOTION`` directly (confirmed by reading
robojudo/policy/unified_loco_kick_policy.py:421-422), so a fresh reset already puts the policy in
exactly the mode this scan needs -- the pose injected below is what would ordinarily be a
mid-episode disturbance; nothing else about the trial needs to look different from an ordinary
locomotion episode.

DOF ORDERING (verified against the actual files, not assumed -- 2026-09-09):
  * The clip npz's 29 hinge joint_names match this project's MuJoCo scene's own kinematic-tree
    <joint> declaration order (g1_29dof.xml) name-for-name, index-for-index. No permutation.
  * MotionLoader's own comment (managers/command/terms/wbt.py) confirms the npz's root quaternion
    is stored wxyz -- MuJoCo's own free-joint qpos convention -- so no xyzw<->wxyz conversion is
    needed either.
  * Column layout (confirmed by reading MotionLoader.__init__, not inferred): joint_pos[:, 0:3] =
    root xyz, [:, 3:7] = root quat wxyz, [:, 7:] = 29 hinge angles in MuJoCo's own order.
    joint_vel[:, 0:6] = root linvel+angvel, [:, 6:] = 29 hinge velocities. This script reimplements
    ONLY that stripping (a few lines), not the rest of MotionLoader -- the training-side class pulls
    in torch/IsaacLab dependencies this scan's ``robojudo`` conda env does not have.

FRAME WINDOW: ``sample_authored_clip_frames`` (the mechanism this scan evaluates) draws uniformly
from ``motion_start_idx .. pre_recovery_motion_end_idx`` -- the WHOLE authored clip, not just the
strike sub-window. ``pre_recovery_motion_end_idx`` is training-internal and not independently
confirmed to equal any one field in configs/skill/skill_*.yaml; ``--frame-max`` is a REQUIRED CLI
arg for exactly this reason -- the caller should pass that skill's own ``stand_start_frame`` as the
best available proxy (the point at which the synthetic recovery/hold tail begins) rather than this
script silently guessing. ``--frame-min`` defaults to 0 (this project's clips are single-motion
files with no other documented leading boundary).

RECENTRING: ``teleport_to_frames`` adds each IsaacLab env's own ``env_origins`` to the clip's root
position (a batched-scene offset with no MuJoCo equivalent here). This scan's clip frames carry the
ORIGINAL capture session's absolute world xy (e.g. ~(2.9, -0.2) for skill_011), which has no
relationship to this scene's own spawn point. ``--recentre-xy`` (default on) zeroes root x/y while
keeping the clip's own z (height) and full orientation -- landing the robot at this scene's nominal
"default_stand" origin (0, 0, *) instead of wherever the original capture happened to be. Height and
orientation are exactly the injected disturbance under test and are never touched.

BALL: parked far away (30, 30), same constant and rationale as
mujoco_loco_to_kick_handoff_scan.py's ``_BALL_PARK_XY`` -- an untested ball sitting in the robot's
landing/recovery footprint would contaminate the fall measurement with an unrelated trip, and this
scan has no shooting objective to satisfy by keeping it nominal.

Per-trial output: "RESULT <step> <trial> <frame> <fall_step_or_-1> <min_z> <recovery_step_or_-1>".
fall_step is ticks since injection (0 = the very first post-teleport tick), or -1 if FALL_Z was
never crossed during the hold window. recovery_step (see RECOVERY_MIN_HEIGHT/
RECOVERY_CONSECUTIVE_STEPS above) is the first tick of the first sustained-recovery streak, or -1
if none completed within the hold window -- can be -1 even for a trial that never fell (settling
into a sub-0.70m crouch that neither falls nor recovers). Two summary lines: "SUMMARY_REFCLIP
<step> <num_alive>/<num_trials> <rate>" and "SUMMARY_RECOVERY_STEPS <step>
<num_recovered>/<num_trials> <recovered_rate> <mean_recovery_steps_or_NA>".

Usage:
    /workspaces/isaaclab_arena/submodules/workspaces/conda_env/robojudo/bin/python \\
        mujoco_ref_clip_recovery_scan.py --onnx-path /path/to/model.onnx --step-label 300000 \\
        --motion-npz /path/to/robot_motion_track_1.npz --frame-max 215 \\
        --num-trials 32 --seed 0 --kick-aim-enabled
"""
from __future__ import annotations

import argparse
import os
import sys

ROBOJUDO_REPO = "/workspaces/isaaclab_arena/submodules/workspaces/humanoid_deployment/RoboJuDo"
FALL_Z = 0.4  # same physical-fall threshold as every sibling scan, for direct comparability

# Same constant, same rationale, as mujoco_loco_to_kick_handoff_scan.py's _BALL_PARK_XY: far enough
# that nothing this scan does can bump it, at the same nominal resting height every sibling scan's
# ball uses.
_BALL_PARK_XY = (30.0, 30.0)
_BALL_REST_Z = 0.11

# "Recovered" = sustained AT/ABOVE this height for this many consecutive ticks. Same constants,
# same rationale, as mujoco_kick_loco_flip_scan.py's own RECOVERY_MIN_HEIGHT/
# RECOVERY_CONSECUTIVE_STEPS -- mirrors training's base_height_below_threshold_sustained_post_
# flip_graced (min_height=0.70, consecutive_steps=10) so both scans measure "recovered" the same
# way, and the ref-clip vs live-donor columns stay comparable.
RECOVERY_MIN_HEIGHT = 0.70
RECOVERY_CONSECUTIVE_STEPS = 10

sys.path.insert(0, ROBOJUDO_REPO)
# This file's OWN directory -- same cross-fork-contamination guard every sibling scan uses.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mujoco_kick_rollout_worker import SCENE_WITH_BALL, _install_ball_observation_patch  # noqa: E402

# Only used when --kick-aim-enabled; this scan never fires a kick, so the value is inert (see
# mujoco_kick_loco_flip_scan.py's own identically-unused constant for the same reasoning).
_KICK_AIM_THETA_REF_DEG_UNUSED = 45.0


def _load_clip(motion_npz: str) -> tuple["np.ndarray", "np.ndarray", int]:  # noqa: F821
    """Minimal, numpy-only reimplementation of MotionLoader's own root-stripping (managers/command
    /terms/wbt.py) -- NOT importing that class directly since it pulls in torch/IsaacLab, which
    this scan's ``robojudo`` conda env does not have. Returns (joint_pos[:, 7:], joint_vel[:, 6:]
    with the root columns re-attached as columns [0:7]/[0:6] respectively -- i.e. the SAME 36/35
    column layout the raw npz already uses, so callers slice it exactly the way MotionLoader's own
    comment documents (see this module's own DOF ORDERING note) -- and num_frames."""
    import numpy as np

    data = np.load(motion_npz, allow_pickle=True)
    joint_pos = data["joint_pos"]
    joint_vel = data["joint_vel"]
    joint_names = data["joint_names"].tolist()
    if joint_pos.shape[1] != len(joint_names) + 7:
        raise ValueError(
            f"{motion_npz}: joint_pos has {joint_pos.shape[1]} columns, expected "
            f"{len(joint_names)} joints + 7 root DOFs. This script assumes the same 'Holosoma "
            f"format' MotionLoader documents (managers/command/terms/wbt.py) -- verify before use."
        )
    return joint_pos, joint_vel, joint_pos.shape[0]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx-path", required=True)
    parser.add_argument("--motion-npz", required=True, help="Retargeted clip npz, e.g. robot_motion_track_1.npz.")
    parser.add_argument("--step-label", required=True)
    parser.add_argument("--skill-id", type=int, default=0)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0, help="Frame-sampling RNG seed; trial i draws from seed+i.")
    parser.add_argument("--frame-min", type=int, default=0, help="Inclusive lower bound, GLOBAL frame index.")
    parser.add_argument(
        "--frame-max", type=int, required=True,
        help="Exclusive upper bound, GLOBAL frame index. Pass this skill's own stand_start_frame "
        "(configs/skill/skill_XXX.yaml) -- see this module's own FRAME WINDOW note for why this "
        "is required rather than defaulted.",
    )
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument(
        "--hold-s", type=float, default=5.0,
        help="How long to keep stepping after injection before judging alive/fallen. Same default "
        "as mujoco_kick_loco_flip_scan.py's --post-flip-hold-s, for direct comparability.",
    )
    parser.add_argument(
        "--recentre-xy", action=argparse.BooleanOptionalAction, default=True,
        help="Zero the clip's root x/y before injecting (default on) -- see this module's own "
        "RECENTRING note. Height and orientation are never touched either way.",
    )
    parser.add_argument(
        "--kick-aim-enabled", action="store_true",
        help="Pass this for a checkpoint trained with kick_aim_enabled=True (every skill in this "
        "project, as of 2026-08-22) -- same rationale as every sibling scan's own flag, even "
        "though this scan never fires a kick: _install_ball_observation_patch still needs it.",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    import mujoco
    import numpy as np

    import robojudo.config.g1  # noqa: F401
    import robojudo.pipeline  # noqa: F401
    from robojudo.config import cfg_registry

    joint_pos, joint_vel, num_frames = _load_clip(args.motion_npz)
    if not (0 <= args.frame_min < args.frame_max <= num_frames):
        raise ValueError(
            f"--frame-min {args.frame_min} / --frame-max {args.frame_max} out of range for "
            f"{args.motion_npz} ({num_frames} frames)."
        )

    cfg = cfg_registry.get("g1_unified_loco_kick")()
    cfg.policy.onnx_path = args.onnx_path
    cfg.env.xml = SCENE_WITH_BALL
    pl = getattr(robojudo.pipeline, cfg.pipeline_type)(cfg=cfg)
    env = pl.env
    env.viewer.is_alive = False
    inner = pl.policy.policy
    inner._update_velocity_command = lambda cd, ball_pos_b=None: None

    ball_jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_freejoint")
    assert ball_jid != -1, "ball_freejoint not found in compiled model -- is SCENE_WITH_BALL correct?"
    ball_qpos_addr = int(env.model.jnt_qposadr[ball_jid])
    ball_qvel_addr = int(env.model.jnt_dofadr[ball_jid])

    # Resolved by name, not assumed to be qpos[0:7]/qvel[0:6] -- same defensive pattern the ball
    # address above already uses, and scene_g1_29dof_with_ball.xml's own comment documents that
    # the robot's freejoint is compiled first, but resolving it explicitly costs nothing and
    # removes the assumption entirely.
    robot_jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
    assert robot_jid != -1, "floating_base_joint not found in compiled model -- scene changed?"
    robot_qpos_addr = int(env.model.jnt_qposadr[robot_jid])
    robot_qvel_addr = int(env.model.jnt_dofadr[robot_jid])

    if args.kick_aim_enabled and inner.get_skill_target_xy(args.skill_id) is None:
        raise ValueError(
            "--kick-aim-enabled requires this checkpoint's ONNX to carry skill_target_xy metadata "
            "-- get_skill_target_xy returned None for this skill_id."
        )

    hold_steps = int(args.hold_s * args.fps)
    rng = np.random.default_rng(args.seed)

    try:
        num_alive = 0
        num_recovered = 0
        sum_recovery_steps = 0
        for trial in range(args.num_trials):
            mujoco.mj_resetDataKeyframe(env.model, env.data, 0)
            inner.reset()  # sets self.task_mode = _TASK_LOCOMOTION (unified_loco_kick_policy.py:422)
            inner._update_velocity_command = lambda cd, ball_pos_b=None: None

            # Ball: parked, at rest -- see this module's own BALL note.
            env.data.qpos[ball_qpos_addr] = _BALL_PARK_XY[0]
            env.data.qpos[ball_qpos_addr + 1] = _BALL_PARK_XY[1]
            env.data.qpos[ball_qpos_addr + 2] = _BALL_REST_Z
            env.data.qvel[ball_qvel_addr : ball_qvel_addr + 6] = 0.0

            # Injection: overwrite the robot's own root + 29 hinge qpos/qvel with a clip frame's
            # recorded state. See this module's own DOF ORDERING note for why these slices need no
            # permutation and no wxyz<->xyzw conversion.
            frame = int(args.frame_min + rng.integers(0, args.frame_max - args.frame_min))
            root_pos = joint_pos[frame, 0:3].copy()
            root_quat = joint_pos[frame, 3:7].copy()  # wxyz
            hinge_pos = joint_pos[frame, 7:]
            root_vel = joint_vel[frame, 0:6].copy()
            hinge_vel = joint_vel[frame, 6:]
            if args.recentre_xy:
                root_pos[0] = 0.0
                root_pos[1] = 0.0

            env.data.qpos[robot_qpos_addr : robot_qpos_addr + 3] = root_pos
            env.data.qpos[robot_qpos_addr + 3 : robot_qpos_addr + 7] = root_quat
            env.data.qpos[robot_qpos_addr + 7 : robot_qpos_addr + 7 + hinge_pos.shape[0]] = hinge_pos
            env.data.qvel[robot_qvel_addr : robot_qvel_addr + 6] = root_vel
            env.data.qvel[robot_qvel_addr + 6 : robot_qvel_addr + 6 + hinge_vel.shape[0]] = hinge_vel

            _install_ball_observation_patch(
                env, inner, mujoco, np, ball_qpos_addr, args.skill_id,
                kick_aim_theta_deg=(0.0 if args.kick_aim_enabled else None),
                kick_aim_theta_ref_deg=_KICK_AIM_THETA_REF_DEG_UNUSED,
            )

            mujoco.mj_forward(env.model, env.data)
            env.update()

            def step_zero_vel() -> None:
                inner.lin_vel_command = np.zeros(2)
                inner.ang_vel_command = 0.0
                pl.step()

            # No settle phase (deliberately -- see this module's own docstring): the injected pose
            # IS the disturbance under test, so measurement starts on the very next tick.
            fall_step = -1
            min_z = float("inf")
            # Same streak-tracking as mujoco_kick_loco_flip_scan.py's post_flip_recovery_step --
            # see that module's own comment for exactly what "streak start, not streak confirm"
            # means and why.
            recovery_step = -1
            _streak_start = None
            for i in range(hold_steps):
                step_zero_vel()
                z = float(env.base_pos[2])
                if z < min_z:
                    min_z = z
                if fall_step == -1 and z < FALL_Z:
                    fall_step = i
                if recovery_step == -1:
                    if z >= RECOVERY_MIN_HEIGHT:
                        if _streak_start is None:
                            _streak_start = i
                        elif i - _streak_start + 1 >= RECOVERY_CONSECUTIVE_STEPS:
                            recovery_step = _streak_start
                    else:
                        _streak_start = None

            # A streak found BEFORE a later fall is not a recovery -- see
            # mujoco_kick_loco_flip_scan.py's identical gate (post_flip_recovery_step) for why,
            # caught by the same empirical smoke test that flagged it there.
            if fall_step != -1:
                recovery_step = -1

            if min_z == float("inf"):  # hold_steps == 0
                min_z = float("nan")
            if fall_step == -1:
                num_alive += 1
            if recovery_step != -1:
                num_recovered += 1
                sum_recovery_steps += recovery_step

            print(
                f"RESULT {args.step_label} {trial} {frame} {fall_step} {min_z:.4f} {recovery_step}",
                flush=True,
            )

        alive_rate = num_alive / args.num_trials
        print(f"SUMMARY_REFCLIP {args.step_label} {num_alive}/{args.num_trials} {alive_rate:.4f}", flush=True)
        # Denominator is num_trials (there is no pre-injection population to exclude here, unlike
        # the flip scan's num_reached_flip) -- same "don't let the mean's own subset hide a bad
        # rate" reasoning as that scan's own SUMMARY_RECOVERY_STEPS line.
        recovered_rate = num_recovered / args.num_trials
        mean_recovery_steps = (sum_recovery_steps / num_recovered) if num_recovered > 0 else None
        mrs = f"{mean_recovery_steps:.4f}" if mean_recovery_steps is not None else "NA"
        print(
            f"SUMMARY_RECOVERY_STEPS {args.step_label} {num_recovered}/{args.num_trials} "
            f"{recovered_rate:.4f} {mrs}",
            flush=True,
        )
        return 0
    finally:
        pl.env.shutdown()


def main() -> int:
    return run(_parse_args())


if __name__ == "__main__":
    sys.exit(main())
