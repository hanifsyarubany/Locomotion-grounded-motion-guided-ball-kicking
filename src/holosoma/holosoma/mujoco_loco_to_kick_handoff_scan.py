#!/usr/bin/env python3
"""Headless scan: for one onnx checkpoint, run the same RoboJuDo `g1_unified_loco_kick` MuJoCo
pipeline as mujoco_kick_survival_scan.py/mujoco_kick_loco_flip_scan.py, but drive RANDOM
locomotion (random lin_vel_x/lin_vel_y/ang_vel_yaw, held for a randomized 2-3s window) before
forcing a SUDDEN flip into kick mode -- the reverse direction of mujoco_kick_loco_flip_scan.py,
and the sim2sim analogue of training's mid_episode_kick_entry_prob (Stage D's locomotion->kick
handoff, see MultiSkillConfig.mid_episode_kick_entry_prob's own docstring for the training-time
mechanism). Reports the FALL RATE for this handoff -- see
FastSACConfig.mujoco_loco_to_kick_handoff_every_n_saves's own docstring for the full motivation.

MECHANISM. Same "[TRIGGER_KICK]" scripted-command channel every sibling scan already uses
(UnifiedLocoKickPolicy._trigger_kick via post_step_callback) -- this script's only new piece is
WHERE the ball gets placed, because unlike every sibling scan (which stays put at the world
origin the whole trial), the robot here has just walked an unpredictable distance in an
unpredictable direction. The ball is placed relative to the robot's ACTUAL pose AT THE FLIP
INSTANT, not a fixed world position -- the same "robot-spawn-anchored, yaw-rotated" transform
training's own `WholeBodyTrackingManager.place_ball_at_entry`/`local_xy_to_world`
(managers/command/terms/wbt.py) already establishes as the single source of truth for exactly
this operation (a mid-episode kick entry's ball placement, live robot pose, not a teleported one).
Reimplemented here in numpy/scipy via RoboJuDo's own `calc_heading_quat_np`/`my_quat_rotate_np`
(robojudo/utils/util_func.py) -- the same forward-rotation counterpart of
`quat_rotate_inverse_np`, which `_install_ball_observation_patch` already uses for the inverse
(world->local) direction.

WHY THE BALL IS PARKED FAR AWAY DURING THE WALK: a ball sitting at its usual ~1.3m-forward nominal
spot would get bumped by a robot walking in a random direction, contaminating the handoff-specific
fall measurement with an unrelated "robot tripped over the ball mid-walk" failure mode. It is
teleported to its real, robot-relative spawn only at the exact tick the flip fires (same tick as
the "[TRIGGER_KICK]" command), with its velocity explicitly zeroed at that teleport (a "parked and
settled" ball should already be at rest, but a MuJoCo teleport should never trust incidental
residual velocity to already be exactly zero).

WHY --kick-aim-enabled IS REQUIRED (not just recommended, unlike mujoco_kick_loco_flip_scan.py):
`_install_ball_observation_patch`'s NON-aim-mode path computes the observed TARGET as a fixed
world-frame point (get_skill_target_xy + ball_x_shift) with no robot-pose transform at all -- correct
only when the robot is known to be at the origin, which is exactly the assumption a random walk
breaks. aim_mode sidesteps this entirely (obs[157:159] is always the theta-normalized, world-frame-
independent [0, 0] this scan feeds), so it is the only mode this script's ball-placement design
supports. Every skill in this project trains with kick_aim_enabled=True (2026-08-22 azimuth-aim
refactor) -- see this file's own --kick-aim-enabled flag for the hard error if omitted.

WHY A TRIAL THAT FALLS DURING THE WALK IS EXCLUDED FROM THE FALL-RATE (AND HIT-RATE) DENOMINATOR:
same "exclude the degenerate case" pattern mujoco_kick_loco_flip_scan.py already established for
its own pre-flip-fail exclusion (itself mirroring kick_direction_success_rate's num_hit
denominator) -- a trial that already toppled from ordinary random-velocity locomotion (a
locomotion robustness failure, not a handoff failure) never actually tested the handoff. Reported
separately as `pre_handoff_fail_rate` so a checkpoint whose locomotion is itself fragile under
aggressive commands doesn't silently inflate or deflate either handoff-specific number. Both
fall_rate and hit_rate share this SAME denominator (num_reached_handoff) -- one coherent
"trials that got a fair test of the handoff" population, rather than each metric quietly defining
its own.

BALL-HIT DETECTION (2026-08-30, added alongside fall-rate): same N trials, no extra rollout cost --
mirrors mujoco_kick_survival_scan.py's own `_ball_foot_contact_now` exactly (real MuJoCo
geom-geom ball<->foot contact, not an approximation), checked only during the POST-FLIP hold
window -- during the walk the ball is parked 30m away (see above), so contact there is physically
impossible and not worth checking. fall_step and hit_step are tracked independently over that same
window (a trial can hit the ball and still fall afterward, or vice versa) -- not mutually
exclusive outcomes.

Per-trial output: "RESULT <step> <trial> <lin_vel_x> <lin_vel_y> <ang_vel_yaw> <loco_steps>
<pre_handoff_fall_step_or_-1> <post_handoff_fall_step_or_-1> <hit_step_or_-1> <min_z>". Three
summary lines: "SUMMARY_PREHANDOFFFAIL <step> <num_pre_handoff_fail>/<num_trials> <rate>" (always
defined), "SUMMARY_LOCOTOKICKFALL <step> <num_fell>/<num_reached_handoff> <rate_or_NA>", and
"SUMMARY_HIT <step> <num_hit>/<num_reached_handoff> <rate_or_NA>" (the latter two "NA" when
num_reached_handoff is 0 -- every trial fell during the walk itself, nothing to measure the
handoff against).

Usage:
    /workspaces/isaaclab_arena/submodules/workspaces/conda_env/robojudo/bin/python \\
        mujoco_loco_to_kick_handoff_scan.py --onnx-path /path/to/model_0005000.onnx \\
        --step-label 5000 --num-trials 32 --seed 0 --skill-id 0 --kick-aim-enabled
"""
from __future__ import annotations

import argparse
import os
import sys

ROBOJUDO_REPO = "/workspaces/isaaclab_arena/submodules/workspaces/humanoid_deployment/RoboJuDo"
FALL_Z = 0.4  # same physical-fall threshold as the sibling scans, for direct comparability

# Where the ball parks during the random walk -- far enough from any reachable walk radius (up to
# ~1.0 m/s * 3.0 s = 3.0 m at the g1 locomotion command's own default range, see
# --lin-vel-x-range/--lin-vel-y-range's own defaults) that the robot can never bump it mid-walk,
# and at the same nominal resting height every sibling scan's ball uses.
_BALL_PARK_XY = (30.0, 30.0)
_BALL_REST_Z = 0.11

sys.path.insert(0, ROBOJUDO_REPO)
# This file's OWN directory -- same cross-fork-contamination guard as every sibling scan.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mujoco_kick_rollout_worker import SCENE_WITH_BALL, _install_ball_observation_patch  # noqa: E402

# Only used when --kick-aim-enabled and theta is always 0.0 (centered) in this scan -- 0.0 /
# anything is 0.0, so this constant's actual value never affects the injected observation. Kept
# internal rather than exposed as a knob with no effect (same rationale as
# mujoco_kick_loco_flip_scan.py's own identical constant).
_KICK_AIM_THETA_REF_DEG_UNUSED = 45.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx-path", required=True)
    parser.add_argument("--settle-s", type=float, default=1.5)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--step-label", required=True)
    parser.add_argument(
        "--skill-id", type=int, default=0,
        help="Which of the ONNX's embedded motion skills to kick -- same convention as every "
        "sibling scan's own --skill-id.",
    )
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0, help="Per-trial command/duration RNG seed.")
    parser.add_argument(
        "--loco-duration-min-s", type=float, default=2.0,
        help="Inclusive lower bound for the randomized random-locomotion window before the flip.",
    )
    parser.add_argument(
        "--loco-duration-max-s", type=float, default=3.0,
        help="Inclusive upper bound for the randomized random-locomotion window before the flip.",
    )
    parser.add_argument(
        "--lin-vel-x-range", type=float, nargs=2, default=(-1.0, 1.0), metavar=("MIN", "MAX"),
        help="Uniform range each trial draws ONE forward velocity command from, held fixed for "
        "the whole walk window. Defaults match this project's own g1 locomotion training range "
        "(config_values/loco/g1/command.py's command_ranges) -- in-distribution, not invented.",
    )
    parser.add_argument("--lin-vel-y-range", type=float, nargs=2, default=(-1.0, 1.0), metavar=("MIN", "MAX"))
    parser.add_argument("--ang-vel-yaw-range", type=float, nargs=2, default=(-1.0, 1.0), metavar=("MIN", "MAX"))
    parser.add_argument(
        "--post-flip-hold-s", type=float, default=8.0,
        help="How long to keep stepping after the flip before judging the trial -- same default "
        "as mujoco_kick_survival_scan.py's own --hold-s (this measures the SAME kind of "
        "let-the-kick-play-out-to-completion window, just from a walking start).",
    )
    parser.add_argument(
        "--kick-aim-enabled", action="store_true",
        help="REQUIRED (not just recommended) for this scan -- see this file's own module "
        "docstring for why the non-aim-mode target-position path is incorrect once the robot has "
        "moved from the origin.",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    if not args.kick_aim_enabled:
        raise ValueError(
            "--kick-aim-enabled is required for mujoco_loco_to_kick_handoff_scan.py -- the "
            "non-aim-mode observed-target path is a fixed world-frame point with no robot-pose "
            "transform, which is only correct when the robot is at the origin. A random walk "
            "breaks that assumption by construction. See this module's own docstring."
        )

    import mujoco
    import numpy as np

    import robojudo.config.g1  # noqa: F401
    import robojudo.pipeline  # noqa: F401
    from robojudo.config import cfg_registry
    from robojudo.utils.util_func import calc_heading_quat_np, my_quat_rotate_np

    cfg = cfg_registry.get("g1_unified_loco_kick")()
    cfg.policy.onnx_path = args.onnx_path
    cfg.env.xml = SCENE_WITH_BALL
    pl = getattr(robojudo.pipeline, cfg.pipeline_type)(cfg=cfg)
    env = pl.env
    env.viewer.is_alive = False
    inner = pl.policy.policy

    ball_jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_freejoint")
    assert ball_jid != -1, "ball_freejoint not found in compiled model -- is SCENE_WITH_BALL correct?"
    ball_qpos_addr = int(env.model.jnt_qposadr[ball_jid])
    ball_qvel_addr = int(env.model.jnt_dofadr[ball_jid])

    # Ball CONTACT HIT detection -- same geoms/mechanism as mujoco_kick_survival_scan.py's own
    # _ball_foot_contact_now (see this module's own docstring). Resolved ONCE here (not per-tick):
    # XML/geom naming is fixed for the compiled model's lifetime.
    ball_geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
    assert ball_geom_id != -1, "ball_geom not found in compiled model -- is SCENE_WITH_BALL correct?"
    foot_geom_ids = {
        gid
        for gid in range(env.model.ngeom)
        if (name := mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, gid)) is not None
        and name.endswith("_collision")
        and (name.startswith("left_foot") or name.startswith("right_foot"))
    }
    assert foot_geom_ids, "no left_foot*_collision/right_foot*_collision geoms found -- model changed?"

    def _ball_foot_contact_now() -> bool:
        contacts = env.data.contact
        for c in range(env.data.ncon):
            g1, g2 = int(contacts.geom1[c]), int(contacts.geom2[c])
            if (g1 == ball_geom_id and g2 in foot_geom_ids) or (g2 == ball_geom_id and g1 in foot_geom_ids):
                return True
        return False

    nominal_ball_xy = inner.get_skill_ball_xy(args.skill_id)
    if nominal_ball_xy is None:
        from mujoco_kick_rollout_worker import BALL_WORLD_POS

        nominal_ball_xy = (BALL_WORLD_POS[0], BALL_WORLD_POS[1])
    if inner.get_skill_target_xy(args.skill_id) is None:
        raise ValueError(
            "--kick-aim-enabled requires this checkpoint's ONNX to carry skill_target_xy metadata "
            "-- get_skill_target_xy returned None for this skill_id."
        )

    settle_steps = int(args.settle_s * args.fps)
    post_flip_hold_steps = int(args.post_flip_hold_s * args.fps)
    trigger_cmd = "[TRIGGER_KICK]" if args.skill_id == 0 else f"[TRIGGER_KICK:{args.skill_id}]"
    rng = np.random.default_rng(args.seed)

    try:
        num_pre_handoff_fail = 0
        num_reached_handoff = 0
        num_post_handoff_fall = 0
        num_hit = 0
        for trial in range(args.num_trials):
            mujoco.mj_resetDataKeyframe(env.model, env.data, 0)
            inner.reset()

            # Park the ball out of the walk's reach -- see this module's own docstring for why.
            env.data.qpos[ball_qpos_addr] = _BALL_PARK_XY[0]
            env.data.qpos[ball_qpos_addr + 1] = _BALL_PARK_XY[1]
            env.data.qpos[ball_qpos_addr + 2] = _BALL_REST_Z
            env.data.qvel[ball_qvel_addr : ball_qvel_addr + 6] = 0.0

            _install_ball_observation_patch(
                env, inner, mujoco, np, ball_qpos_addr, args.skill_id,
                kick_aim_theta_deg=0.0,
                kick_aim_theta_ref_deg=_KICK_AIM_THETA_REF_DEG_UNUSED,
            )

            mujoco.mj_forward(env.model, env.data)
            env.update()

            def step_zero_vel() -> None:
                inner.lin_vel_command = np.zeros(2)
                inner.ang_vel_command = 0.0
                pl.step()

            for _ in range(settle_steps):
                step_zero_vel()

            lin_vel_x = float(rng.uniform(*args.lin_vel_x_range))
            lin_vel_y = float(rng.uniform(*args.lin_vel_y_range))
            ang_vel_yaw = float(rng.uniform(*args.ang_vel_yaw_range))
            loco_duration_s = float(rng.uniform(args.loco_duration_min_s, args.loco_duration_max_s))
            loco_steps = int(loco_duration_s * args.fps)

            def step_random_vel() -> None:
                inner.lin_vel_command = np.array([lin_vel_x, lin_vel_y])
                inner.ang_vel_command = ang_vel_yaw
                pl.step()

            z_series = []
            pre_handoff_fall_step = -1
            for i in range(loco_steps):
                step_random_vel()
                z = float(env.base_pos[2])
                z_series.append(z)
                if pre_handoff_fall_step == -1 and z < FALL_Z:
                    pre_handoff_fall_step = i

            # Flip instant: teleport the ball relative to the robot's ACTUAL pose right now (same
            # transform as training's own local_xy_to_world -- see this module's own docstring),
            # then trigger the kick. Robot velocity command reset to zero first -- kick mode
            # tracks a reference clip, not a velocity command, same as every sibling scan holding
            # zero-vel throughout its own kick phase.
            robot_pos_w = env.base_pos.copy()
            robot_quat_xyzw = env.base_quat.copy()
            heading_quat = calc_heading_quat_np(robot_quat_xyzw)
            local_xyz = np.array([nominal_ball_xy[0], nominal_ball_xy[1], 0.0])
            ball_world_xy = my_quat_rotate_np(heading_quat, local_xyz)[:2] + robot_pos_w[:2]
            env.data.qpos[ball_qpos_addr] = ball_world_xy[0]
            env.data.qpos[ball_qpos_addr + 1] = ball_world_xy[1]
            env.data.qpos[ball_qpos_addr + 2] = _BALL_REST_Z
            env.data.qvel[ball_qvel_addr : ball_qvel_addr + 6] = 0.0
            mujoco.mj_forward(env.model, env.data)

            inner.lin_vel_command = np.zeros(2)
            inner.ang_vel_command = 0.0
            env.update()
            inner.get_observation(env.get_data(), {})
            inner.post_step_callback([trigger_cmd])

            post_handoff_fall_step = -1
            hit_step = -1
            for i in range(post_flip_hold_steps):
                step_zero_vel()
                z = float(env.base_pos[2])
                z_series.append(z)
                if post_handoff_fall_step == -1 and z < FALL_Z:
                    post_handoff_fall_step = i
                if hit_step == -1 and _ball_foot_contact_now():
                    hit_step = i

            min_z = min(z_series) if z_series else float("nan")
            if pre_handoff_fall_step != -1:
                num_pre_handoff_fail += 1
            else:
                num_reached_handoff += 1
                if post_handoff_fall_step != -1:
                    num_post_handoff_fall += 1
                if hit_step != -1:
                    num_hit += 1

            print(
                f"RESULT {args.step_label} {trial} {lin_vel_x:.4f} {lin_vel_y:.4f} "
                f"{ang_vel_yaw:.4f} {loco_steps} {pre_handoff_fall_step} {post_handoff_fall_step} "
                f"{hit_step} {min_z:.4f}",
                flush=True,
            )

        pre_handoff_fail_rate = num_pre_handoff_fail / args.num_trials
        print(
            f"SUMMARY_PREHANDOFFFAIL {args.step_label} {num_pre_handoff_fail}/{args.num_trials} "
            f"{pre_handoff_fail_rate:.4f}",
            flush=True,
        )
        if num_reached_handoff > 0:
            fall_rate = num_post_handoff_fall / num_reached_handoff
            print(
                f"SUMMARY_LOCOTOKICKFALL {args.step_label} {num_post_handoff_fall}/{num_reached_handoff} "
                f"{fall_rate:.4f}",
                flush=True,
            )
            hit_rate = num_hit / num_reached_handoff
            print(
                f"SUMMARY_HIT {args.step_label} {num_hit}/{num_reached_handoff} {hit_rate:.4f}",
                flush=True,
            )
        else:
            print(f"SUMMARY_LOCOTOKICKFALL {args.step_label} 0/0 NA", flush=True)
            print(f"SUMMARY_HIT {args.step_label} 0/0 NA", flush=True)
        return 0
    finally:
        pl.env.shutdown()


def main() -> int:
    return run(_parse_args())


if __name__ == "__main__":
    sys.exit(main())
