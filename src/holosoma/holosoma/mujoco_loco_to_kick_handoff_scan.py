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
Optionally jittered in that same robot-relative frame by --ball-pos-randomization-x/-y (default
0.0, i.e. exact nominal placement, unchanged unless passed) -- see that flag's own help text for
why the jitter has to be applied in the robot's frame here rather than world frame the way
mujoco_kick_survival_scan.py's identically-named flag does.
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

TRANSITION METRICS (--track-transition-metrics, 2026-09-02, opt-in/off by default): fall_rate/
hit_rate only see an outright topple (FALL_Z=0.40m) or a made contact -- neither can tell "smooth
handoff" from "upright but drifting/jittering through it", which is exactly the failure mode this
scan exists to catch that kick_survival's settled-standstill start cannot. Three quantities,
computed ONLY over trials that reached the handoff (same num_reached_handoff population as
fall_rate/hit_rate), each split into an EARLY window (the first --transition-window-steps ticks
after the flip -- this project's checkpoints all use a 1.0s/50-tick default-pose PREPEND ramp
there, see sim2sim_eval.py's own get_strike_window_ticks docstring for that derivation) and a LATE
window (the last --transition-window-steps ticks of the hold, once any transient has settled) --
the difference between the two is the transition's own signature, not a property of the checkpoint
in general:
  - tracking error: mean per-tick ||env.dof_pos - inner.motion_command_t[:num_dofs]|| -- distance
    from the reference the policy is actually supposed to be tracking at that instant.
  - jerk: mean per-tick ||action_t - action_{t-1}|| (inner.last_action) -- action-space smoothness.
  - drift: SINGLE scalar (no early/late split -- it inherently describes the ramp itself), the
    change in the robot-relative ball offset from the flip instant (by construction exactly
    nominal_ball_xy) to the end of the early window -- reusing quat_rotate_inverse_np, the exact
    inverse of this file's own robot-relative ball-placement transform above.
Five more summary lines when enabled: "SUMMARY_TRACKING_ERROR_EARLY/_LATE <step> <value_or_NA>",
"SUMMARY_JERK_EARLY/_LATE <step> <value_or_NA>", "SUMMARY_DRIFT <step> <value_or_NA>" -- "NA" under
the same num_reached_handoff==0 condition as SUMMARY_LOCOTOKICKFALL/SUMMARY_HIT.

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

# Normalization reference for the injected aim command: obs[157:159] = [theta / ref, 0.0].
# 2026-09-12: this WAS a genuinely inert constant back when this scan always fed theta = 0.0 (0 /
# anything is 0), and was named _UNUSED accordingly. --kick-aim-theta-max-deg now sweeps theta, so
# the reference divides a NONZERO value and its magnitude matters. 45.0 is MultiSkillConfig/
# BallConfig.kick_aim_theta_ref_deg's own default (held fixed across curriculum changes) and is the
# same value mujoco_kick_survival_scan.py feeds -- the two scans MUST agree on it, or the same
# commanded theta means a different observation in each and their success rates stop being
# comparable, which is the entire point of measuring this row.
_KICK_AIM_THETA_REF_DEG_DEFAULT = 45.0


def _trial_mean(series: list[float], window: int) -> float | None:
    """Mean of this trial's own EARLY window (first `window` entries) -- the per-trial counterpart
    of the cross-trial SUMMARY_*_EARLY aggregate, for the RESULT line only. None when the series is
    empty (transition tracking off, or a zero-length hold)."""
    early = series[: min(window, len(series))]
    return float(sum(early) / len(early)) if early else None


def _fmt(value: float | None) -> str:
    """RESULT-line field formatter -- "nan" (not an empty field) for a missing value, so the
    column count stays fixed and the line remains whitespace-splittable."""
    return "nan" if value is None else f"{value:.4f}"


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
        "--ball-pos-randomization-x", type=float, default=0.0,
        help="Uniform +/- half-range (meters) each trial jitters the ball's ROBOT-RELATIVE spawn "
        "offset at the flip instant, applied to local_xyz BEFORE the heading rotation -- same "
        "half-range convention as mujoco_kick_survival_scan.py's own flag of the same name, "
        "just expressed in the robot's own frame instead of world frame since this scan's robot "
        "isn't anchored at the origin. In the degenerate case where the robot IS at the origin "
        "with zero heading (mujoco_kick_survival_scan.py's every trial), the two are identical. "
        "Default 0.0 -- no jitter, exact nominal_ball_xy every trial, unchanged from before this "
        "flag existed. Pass the checkpoint's own BallConfig.position_randomization[0] to match "
        "the settled-state (kick_survival) row for a fair Table VIII comparison.",
    )
    parser.add_argument("--ball-pos-randomization-y", type=float, default=0.0)
    parser.add_argument(
        "--kick-aim-enabled", action="store_true",
        help="REQUIRED (not just recommended) for this scan -- see this file's own module "
        "docstring for why the non-aim-mode target-position path is incorrect once the robot has "
        "moved from the origin.",
    )
    parser.add_argument(
        "--track-transition-metrics", action="store_true",
        help="Additionally report tracking-error/jerk (early vs late window) and drift -- see "
        "this file's own module docstring TRANSITION METRICS section. Off by default: adds no "
        "extra rollout cost either way, just extra per-tick bookkeeping during the hold window.",
    )
    parser.add_argument(
        "--transition-window-steps", type=int, default=50,
        help="Ticks per early/late window for --track-transition-metrics (default 50 = this "
        "project's own 1.0s default-pose prepend ramp duration at 50Hz -- override for a "
        "checkpoint with a different default_pose_prepend_duration_s).",
    )
    # ---- Aim + success scoring (2026-09-12) -------------------------------------------------
    # This scan historically reported only survival and CONTACT. Contact saturates (near 100% on
    # most skills), so it cannot show the tradeoff this row exists to test: entering the kick
    # while carrying real momentum may still strike the ball but aim it worse. These four flags
    # add the same success measurement mujoco_kick_survival_scan.py already reports, so the
    # settled and in-motion rows of the paper's handoff table can be read side by side.
    parser.add_argument(
        "--kick-aim-theta-max-deg", type=float, default=0.0,
        help="Sample this trial's commanded aim offset theta uniformly from [-X, +X] degrees, "
        "instead of the fixed 0.0 this scan used to hardcode. Pass the SAME value the companion "
        "kick_survival sweep uses (15.0 in this project) -- at theta=0 the aim task is the easy, "
        "centered one, so scoring success here at 0.0 against a settled row swept over +/-15 "
        "would flatter this row for a reason that has nothing to do with the handoff. "
        "0.0 (default) reproduces the previous fixed-theta behavior BIT-FOR-BIT: the theta draw "
        "is skipped entirely rather than drawn-and-ignored, so the RNG stream feeding the walk "
        "commands is untouched and every previously-recorded run replays identically.",
    )
    parser.add_argument(
        "--kick-aim-theta-ref-deg", type=float, default=_KICK_AIM_THETA_REF_DEG_DEFAULT,
        help="Normalization reference for the injected aim command (obs[157:159] = "
        "[theta/ref, 0]). Must match the checkpoint's own kick_aim_theta_ref_deg AND the value "
        "every sibling scan feeds -- see that constant's own comment.",
    )
    parser.add_argument(
        "--kick-aim-nominal-distance-m", type=float, default=5.0,
        help="Distance (m) from the ball to the commanded target point along the aimed bearing, "
        "matching BallConfig.kick_aim_nominal_distance_m's own default and the value "
        "mujoco_kick_survival_scan.py uses. The target is anchored to where the ball ACTUALLY "
        "lands this trial (nominal + this trial's jitter), in the robot's own frame at the flip.",
    )
    parser.add_argument(
        "--success-sigma-m", type=float, nargs="+", default=None,
        help="One SUMMARY_SUCCESS_<R> line per radius R (meters): the fraction of reached-handoff "
        "trials whose ball passed within R of the commanded target point. Omitted = no success "
        "lines (this scan's original output, unchanged). Same definition and same nearest-"
        "post-contact-distance contract as mujoco_kick_survival_scan.py's own --success-sigma-m, "
        "so the two scans' numbers are directly comparable.",
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
    from robojudo.utils.util_func import calc_heading_quat_np, my_quat_rotate_np, quat_rotate_inverse_np

    cfg = cfg_registry.get("g1_unified_loco_kick")()
    cfg.policy.onnx_path = args.onnx_path
    cfg.env.xml = SCENE_WITH_BALL
    pl = getattr(robojudo.pipeline, cfg.pipeline_type)(cfg=cfg)
    env = pl.env
    env.viewer.is_alive = False
    inner = pl.policy.policy
    # 2026-09-12, BUG FIX -- without this line the walk window does not walk. pl.step() calls the
    # policy's own _update_velocity_command every tick (unified_loco_kick_policy.py line ~1016),
    # which reads a joystick/keyboard and then UNCONDITIONALLY assigns self.lin_vel_command /
    # self.ang_vel_command from its smoothed reading (line ~610). With no controller attached --
    # which is every headless sweep -- that reading is zero, so it silently overwrote the command
    # this scan sets immediately before each step. Measured: commanding 1.0 m/s forward for 120
    # ticks moved the robot -0.0450 m, byte-identical to commanding 0.0, and the every-tick
    # command read back as 0.000. Every "random in-motion" number produced before this fix was
    # actually a STANDING entry with a 2-3s zero-command window, which is also why pre-handoff
    # falls were exactly 0.00 across all 112 (checkpoint, skill) cells -- a robot that never walks
    # never trips. mujoco_kick_survival_scan.py already neutralizes this for the same reason
    # (twice, see its own two call sites); this scan needed it just as much and did not have it.
    inner._update_velocity_command = lambda cd, ball_pos_b=None: None

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
    nominal_target_xy = inner.get_skill_target_xy(args.skill_id)
    if nominal_target_xy is None:
        raise ValueError(
            "--kick-aim-enabled requires this checkpoint's ONNX to carry skill_target_xy metadata "
            "-- get_skill_target_xy returned None for this skill_id."
        )
    # This skill's trained aim DIRECTION, derived from the checkpoint's own untouched
    # (ball_xy, target_xy) metadata -- same atan2 convention as SkillConfig.
    # resolved_nominal_bearing_deg() and as mujoco_kick_survival_scan.py's own derivation, so the
    # commanded target below is the same point that scan grades against.
    nominal_bearing_deg = float(
        np.degrees(
            np.arctan2(
                nominal_target_xy[1] - nominal_ball_xy[1], nominal_target_xy[0] - nominal_ball_xy[0]
            )
        )
    )
    print(f"[kick_aim] skill {args.skill_id} nominal_bearing_deg={nominal_bearing_deg:.2f}", flush=True)
    success_sigmas = sorted(set(args.success_sigma_m)) if args.success_sigma_m else []

    settle_steps = int(args.settle_s * args.fps)
    post_flip_hold_steps = int(args.post_flip_hold_s * args.fps)
    trigger_cmd = "[TRIGGER_KICK]" if args.skill_id == 0 else f"[TRIGGER_KICK:{args.skill_id}]"
    rng = np.random.default_rng(args.seed)

    try:
        num_pre_handoff_fail = 0
        num_reached_handoff = 0
        num_post_handoff_fall = 0
        num_hit = 0
        # Success is counted over the REACHED-HANDOFF population (the same denominator fall_rate
        # and hit_rate already use), not over all trials: a trial that fell during the random walk
        # never reached the kick and has no aim to grade. A trial that reached the handoff and
        # then whiffed IS counted, as a failure -- min_target_dist simply stays at its huge init
        # value -- matching mujoco_kick_survival_scan.py's own "a whiff is a failure" convention.
        num_success = {sigma: 0 for sigma in success_sigmas}
        hit_shot_errors: list[float] = []
        # Per-trial values from reached-handoff trials only -- same population fall_rate/hit_rate
        # average over. None entries (a trial too short for a given window) are skipped when
        # averaging below, not treated as zero.
        tracking_error_earlys: list[float] = []
        tracking_error_lates: list[float] = []
        jerk_earlys: list[float] = []
        jerk_lates: list[float] = []
        drifts: list[float] = []
        for trial in range(args.num_trials):
            mujoco.mj_resetDataKeyframe(env.model, env.data, 0)
            inner.reset()

            # Park the ball out of the walk's reach -- see this module's own docstring for why.
            env.data.qpos[ball_qpos_addr] = _BALL_PARK_XY[0]
            env.data.qpos[ball_qpos_addr + 1] = _BALL_PARK_XY[1]
            env.data.qpos[ball_qpos_addr + 2] = _BALL_REST_Z
            env.data.qvel[ball_qvel_addr : ball_qvel_addr + 6] = 0.0

            # Drawn ONLY when a range is actually requested, so --kick-aim-theta-max-deg 0.0
            # leaves the RNG stream (and therefore every walk command below) bit-identical to
            # every run recorded before this flag existed.
            kick_aim_theta = (
                float(rng.uniform(-args.kick_aim_theta_max_deg, args.kick_aim_theta_max_deg))
                if args.kick_aim_theta_max_deg > 0.0
                else 0.0
            )
            _install_ball_observation_patch(
                env, inner, mujoco, np, ball_qpos_addr, args.skill_id,
                kick_aim_theta_deg=kick_aim_theta,
                kick_aim_theta_ref_deg=args.kick_aim_theta_ref_deg,
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
            # Robot-relative placement jitter, drawn fresh each trial and applied BEFORE the
            # heading rotation -- see --ball-pos-randomization-x/-y's own help text for why this
            # is the correct local-frame equivalent of mujoco_kick_survival_scan.py's world-frame
            # jitter (that scan's robot is always at the origin with zero heading at placement
            # time, so world-frame and robot-frame jitter coincide there; this scan's robot is
            # not, so the jitter has to move with it to mean the same thing).
            ball_dx = float(rng.uniform(-args.ball_pos_randomization_x, args.ball_pos_randomization_x))
            ball_dy = float(rng.uniform(-args.ball_pos_randomization_y, args.ball_pos_randomization_y))
            local_xyz = np.array([nominal_ball_xy[0] + ball_dx, nominal_ball_xy[1] + ball_dy, 0.0])
            ball_world_xy = my_quat_rotate_np(heading_quat, local_xyz)[:2] + robot_pos_w[:2]
            env.data.qpos[ball_qpos_addr] = ball_world_xy[0]
            env.data.qpos[ball_qpos_addr + 1] = ball_world_xy[1]
            env.data.qpos[ball_qpos_addr + 2] = _BALL_REST_Z
            env.data.qvel[ball_qvel_addr : ball_qvel_addr + 6] = 0.0
            mujoco.mj_forward(env.model, env.data)

            # This trial's commanded target point, in world coordinates. Built in the robot's OWN
            # frame from the same local ball placement used just above (nominal + this trial's
            # jitter) and pushed out through the SAME heading transform, so the aimed bearing
            # rotates with the robot exactly as training's local_xy_to_world does. Rotating a
            # world-frame target instead would leave the commanded direction pointing wherever the
            # robot happened to start, which is the bug this scan already avoids for the ball.
            theta_rad = np.radians(nominal_bearing_deg + kick_aim_theta)
            local_target = np.array([
                local_xyz[0] + args.kick_aim_nominal_distance_m * np.cos(theta_rad),
                local_xyz[1] + args.kick_aim_nominal_distance_m * np.sin(theta_rad),
                0.0,
            ])
            target_xy = my_quat_rotate_np(heading_quat, local_target)[:2] + robot_pos_w[:2]

            inner.lin_vel_command = np.zeros(2)
            inner.ang_vel_command = 0.0
            env.update()
            inner.get_observation(env.get_data(), {})
            inner.post_step_callback([trigger_cmd])

            post_handoff_fall_step = -1
            hit_step = -1
            # A MINIMUM being tracked, so inf (not -1) is the right init -- converted to a -1
            # sentinel only at print time. Same latch as shooting.py's own error_ball_to_target.
            min_target_dist = float("inf")
            # Only accumulated when --track-transition-metrics -- see this module's own docstring
            # TRANSITION METRICS section. num_dofs from inner directly (not hardcoded 29) so this
            # keeps working if the robot config ever changes.
            tracking_errors: list[float] = []
            jerks: list[float] = []
            prev_action = inner.last_action.copy() if args.track_transition_metrics else None
            drift_this_trial: float | None = None
            for i in range(post_flip_hold_steps):
                step_zero_vel()
                z = float(env.base_pos[2])
                z_series.append(z)
                if post_handoff_fall_step == -1 and z < FALL_Z:
                    post_handoff_fall_step = i
                if hit_step == -1 and _ball_foot_contact_now():
                    hit_step = i
                # Nearest approach of the ball to the commanded target over the whole post-flip
                # hold, tracked every tick (not only after contact) -- identical contract to
                # mujoco_kick_survival_scan.py's own min_target_dist.
                ball_xy_now = np.array([
                    env.data.qpos[ball_qpos_addr], env.data.qpos[ball_qpos_addr + 1]
                ])
                dist_now = float(np.linalg.norm(ball_xy_now - target_xy))
                if dist_now < min_target_dist:
                    min_target_dist = dist_now
                if args.track_transition_metrics:
                    tracking_errors.append(
                        float(np.linalg.norm(env.dof_pos - inner.motion_command_t[: inner.num_dofs]))
                    )
                    jerks.append(float(np.linalg.norm(inner.last_action - prev_action)))
                    prev_action = inner.last_action.copy()
                    if i == args.transition_window_steps - 1:
                        # Robot-relative ball offset NOW vs at the flip instant (which, by
                        # construction of ball_world_xy above, is EXACTLY local_xyz[:2] --
                        # nominal_ball_xy PLUS this trial's own (ball_dx, ball_dy) placement
                        # jitter, not the bare nominal -- so a jittered trial's drift measures
                        # motion since the flip, not the placement jitter itself baked in as a
                        # spurious i==0 offset) -- the exact inverse of this file's own forward
                        # robot-relative-ball-placement transform a few lines up.
                        ball_pos_w_now = np.array([ball_world_xy[0], ball_world_xy[1], _BALL_REST_Z])
                        heading_quat_now = calc_heading_quat_np(env.base_quat.copy())
                        ball_pos_b_now = quat_rotate_inverse_np(
                            heading_quat_now, ball_pos_w_now - env.base_pos.copy()
                        )[:2]
                        drift_this_trial = float(np.linalg.norm(ball_pos_b_now - local_xyz[:2]))

            min_z = min(z_series) if z_series else float("nan")
            if pre_handoff_fall_step != -1:
                num_pre_handoff_fail += 1
            else:
                num_reached_handoff += 1
                if post_handoff_fall_step != -1:
                    num_post_handoff_fall += 1
                if hit_step != -1:
                    num_hit += 1
                    # Shot error is gated on contact: a whiff has no shot to grade. It still
                    # counts against success above, it just contributes no error sample here.
                    if np.isfinite(min_target_dist):
                        hit_shot_errors.append(min_target_dist)
                for sigma in success_sigmas:
                    if min_target_dist <= sigma:
                        num_success[sigma] += 1
                if args.track_transition_metrics:
                    w = args.transition_window_steps
                    early_end = min(w, len(tracking_errors))
                    late_start = max(early_end, len(tracking_errors) - w)  # non-overlapping with early
                    if early_end > 0:
                        tracking_error_earlys.append(float(np.mean(tracking_errors[:early_end])))
                        jerk_earlys.append(float(np.mean(jerks[:early_end])))
                    if late_start < len(tracking_errors):
                        tracking_error_lates.append(float(np.mean(tracking_errors[late_start:])))
                        jerk_lates.append(float(np.mean(jerks[late_start:])))
                    if drift_this_trial is not None:
                        drifts.append(drift_this_trial)

            print(
                f"RESULT {args.step_label} {trial} {lin_vel_x:.4f} {lin_vel_y:.4f} "
                f"{ang_vel_yaw:.4f} {loco_steps} {pre_handoff_fall_step} {post_handoff_fall_step} "
                f"{hit_step} {min_z:.4f} "
                f"{(min_target_dist if np.isfinite(min_target_dist) else -1.0):.4f} "
                f"{kick_aim_theta:.4f}"
                # 3 extra trailing fields ONLY under --track-transition-metrics (default output
                # stays byte-identical). Per-trial, not just the trial-averaged SUMMARY lines,
                # specifically so drift can be correlated against THIS trial's own entry velocity
                # (lin_vel_x/lin_vel_y above) -- the test of whether drift is momentum-limited
                # (~v^2/2a, nothing for a policy objective to reclaim) or policy-limited. "nan"
                # for a window this trial was too short to fill.
                + (
                    f" {_fmt(_trial_mean(tracking_errors, args.transition_window_steps))}"
                    f" {_fmt(_trial_mean(jerks, args.transition_window_steps))}"
                    f" {_fmt(drift_this_trial)}"
                    if args.track_transition_metrics
                    else ""
                ),
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
            for sigma in success_sigmas:
                rate = num_success[sigma] / num_reached_handoff
                print(
                    f"SUMMARY_SUCCESS_{sigma:g} {args.step_label} "
                    f"{num_success[sigma]}/{num_reached_handoff} {rate:.4f}",
                    flush=True,
                )
            if hit_shot_errors:
                print(
                    f"SUMMARY_SHOT_ERROR {args.step_label} {float(np.mean(hit_shot_errors)):.4f} "
                    f"{float(np.std(hit_shot_errors)):.4f} {len(hit_shot_errors)}",
                    flush=True,
                )
            else:
                print(f"SUMMARY_SHOT_ERROR {args.step_label} NA NA 0", flush=True)
        else:
            print(f"SUMMARY_LOCOTOKICKFALL {args.step_label} 0/0 NA", flush=True)
            print(f"SUMMARY_HIT {args.step_label} 0/0 NA", flush=True)
            for sigma in success_sigmas:
                print(f"SUMMARY_SUCCESS_{sigma:g} {args.step_label} 0/0 NA", flush=True)
            print(f"SUMMARY_SHOT_ERROR {args.step_label} NA NA 0", flush=True)

        if args.track_transition_metrics:
            def _print_mean(name: str, values: list[float]) -> None:
                value_str = f"{float(np.mean(values)):.4f}" if values else "NA"
                print(f"SUMMARY_{name} {args.step_label} {value_str}", flush=True)

            _print_mean("TRACKING_ERROR_EARLY", tracking_error_earlys)
            _print_mean("TRACKING_ERROR_LATE", tracking_error_lates)
            _print_mean("JERK_EARLY", jerk_earlys)
            _print_mean("JERK_LATE", jerk_lates)
            _print_mean("DRIFT", drifts)
        return 0
    finally:
        pl.env.shutdown()


def main() -> int:
    return run(_parse_args())


if __name__ == "__main__":
    sys.exit(main())
