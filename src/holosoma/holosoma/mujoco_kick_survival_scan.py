#!/usr/bin/env python3
"""Headless scan: for one onnx checkpoint, run the same RoboJuDo MuJoCo g1_unified_loco_kick
rollout as mujoco_kick_rollout_worker.py (real ball, task-mode-gated obs patch), no video, and
report the base height time series so many checkpoints can be swept quickly to characterize
survival-time noise across a training run.

2026-08-19: extended from a single deterministic trial to N trials with in-distribution ball
spawn/target jitter, to produce a genuine FALL RATE (not a single yes/no) for periodic training-
time evaluation -- see FastSACAgent._maybe_start_mujoco_survival_scan and
record_mujoco_survival_scan.py, which invoke this as a subprocess on the same cadence/thread/lock
architecture as the existing MuJoCo kick-rollout video. Motivation (2026-08-19 discussion):
Env/kick_topple_frac (the IsaacSim training-time metric) mixes three distinct sources of movement
that make it hard to trust as a standalone signal -- (1) small-sample EMA noise (each fold is only
~1-2 ended episodes, weighted equally regardless of batch size; measured sd falls ~1/sqrt(window),
the signature of pure sampling noise), (2) SAC exploration-noise-driven behavioral variance (the
IsaacSim rollout uses actor.explore(deterministic=False), not the deployed E[tanh] action), and
(3) termination-censoring (an unrelated termination, e.g. bad_tracking or the old 1.0N contact
term, can end an episode before a fall would have had the chance to develop, so kick_topple_frac
partly reflects HOW OFTEN OTHER TERMINATIONS FIRE FIRST, not just genuine falls -- see
MultiSkillConfig.contact_termination_force_threshold's own docstring for the measured 78.9%-of-
episodes example). This scan is immune to all three: it runs the DEPLOYED deterministic ONNX
action (matching what would actually ship), a fixed hold duration regardless of what an IsaacSim-
side termination would have done, and reports an EXACT count over N trials rather than a decaying
average -- a second, complementary signal, not a replacement (a real, independently-measured
PhysX<->MuJoCo contact-resolution gap means this will not numerically match the IsaacSim rate --
see memory stagec-kick-open-loop-physics-proof).

BUG FIX (2026-08-19, found while extending this file): as it stood before this change, this script
was BROKEN -- `_install_ball_observation_patch` gained two new required positional arguments
(`ball_qpos_addr`, `skill_id`) in mujoco_kick_rollout_worker.py at some point after this file was
last touched, and this file's single call site was never updated to match, so every invocation
raised `TypeError: _install_ball_observation_patch() missing 2 required positional arguments`
before ever reaching the RESULT line. Confirmed by running it as-is before this fix. Also: this
file's own sys.path.insert pointed at a DIFFERENT, sibling fork
(.../locomotion_and_ball_kicking/src/holosoma) instead of its own -- harmless today only because
Python's own script-directory auto-insert still resolved `mujoco_kick_rollout_worker` correctly in
practice for a script run by its own path, but a real cross-fork-contamination risk (the exact bug
class this session's own probe scripts hit earlier: an import silently resolving against the wrong
fork's copy of a module with the same name). Both are fixed here: ball_qpos_addr/skill_id are now
computed/passed exactly as mujoco_kick_rollout_worker.py's own run() does (including actually
POSITIONING the ball via get_skill_ball_xy, which this file never did before either -- the ball
was silently left at the scene XML's fixed keyframe spawn on every prior "run" of this script),
and the path insert now derives from this file's own location.

Per-trial output: "RESULT <step> <trial> <fall_step_or_-1> <min_z> <hit_step_or_-1>
<min_target_dist_or_-1> <max_ball_speed> <kick_aim_theta_or_NA>". fall_step is the first control tick (50Hz) where base z drops below
FALL_Z, or -1 if it never does, during that trial's hold window. hit_step (2026-08-21) is the
first control tick, counted from the same hold window (i.e. from the kick trigger onward, not the
settle phase before it), where MuJoCo's own contact solver reports a real ball<->foot geom contact
(env.data.contact, checked against every "{left,right}_footN_collision" geom -- see
_ball_foot_contact_now's own docstring), or -1 if the ball is never touched. min_target_dist
(2026-08-23) is this trial's closest approach (meters, over the same hold window) of the ball's
real position to the SAME target point training's own error_ball_to_target reward measures
against (ball_local_placed + kick_aim_nominal_distance_m * unit(nominal_bearing_deg +
kick_aim_theta)) -- -1 when --kick-aim-enabled isn't set (no per-trial commanded direction to
measure against). Summary lines: "SUMMARY <step> <num_fell>/<num_trials> <fall_rate>",
"SUMMARY_HIT <step> <num_hit>/<num_trials> <hit_rate>", and (only when --kick-aim-enabled)
"SUMMARY_DIRECTION <step> <num_direction_hit>/<num_hit> <direction_success_rate_or_NA>" --
direction success is gated on ACTUALLY hitting the ball (num_hit is the denominator, not
num_trials: a whiff has no departure direction to grade, and folding misses into this rate would
conflate "aimed badly" with "never kicked", which kick_ball_hit_rate already covers separately).
"NA" (not a float) when num_hit is 0 -- nothing to measure that checkpoint against yet.

2026-09-05: max_ball_speed (appended to RESULT) is the 2D (xy) peak ball speed observed from
hit_step onward -- matches shooting.py's ball_speed = norm(ball_vel_xy) exactly (vertical bounce
deliberately excluded), and is 0.0 for a whiff (never updated before contact). Two new summary
lines, both mean +/- std +/- n over HIT trials only (same "a whiff has nothing to grade" exclusion
as SUMMARY_DIRECTION, not counted in num_trials): "SUMMARY_BALL_SPEED <step> <mean> <std> <n>
<max>" (always printed, independent of --kick-aim-enabled -- a hit's peak speed doesn't need a
commanded target) and "SUMMARY_SHOT_ERROR <step> <mean> <std> <n>" (only under
--kick-aim-enabled, since it needs target_xy; RoboNaldo's own "nearest post-contact ball-target
distance", arXiv:2606.11092 -- NOTE their paper does not state whether misses are excluded or
included in their own reported average, so treat this as this project's own well-defined version
of the same statistic, not a verified bit-identical protocol match). Both print "NA NA 0" (plus a
4th "NA" on SUMMARY_BALL_SPEED, for <max>) when there are zero hit trials.

2026-09-06: <max> (5th field on SUMMARY_BALL_SPEED only, added at user request for a direct
comparison against RoboNaldo's own reported PEAK ball speed, not just its mean-of-hits) is simply
`max(hit_peak_speeds)` -- the single fastest hit observed across this scan's num_trials, backward-
compatible append (existing 4-field readers untouched; SUMMARY_SHOT_ERROR's own shape is
unchanged -- shot error has no analogous "max" requested).

2026-09-09: handoff_tick/post_handoff_fall_step (appended to RESULT, two trailing fields,
backward-compatible) isolate the Kick->Loco AUTO-HANDOFF this scan's hold window already runs
through but never previously measured -- UnifiedLocoKickPolicy._return_to_loco() fires on its own
once the authored clip concludes (either at pre_recovery_motion_end_idx for a
kick_recovery_locomotion_flip_enabled checkpoint, or via a 3s plateau-hold heuristic otherwise --
see that method's own call sites in unified_loco_kick_policy.py), no scripted [RETURN_TO_LOCO]
needed, so it was already happening inside this scan's existing hold_steps loop, just never
isolated from the in-kick fall_step above it. handoff_tick is the first control tick (relative to
the SAME kick-trigger-relative window fall_step uses) where `inner.task_mode` is observed back at
locomotion, or -1 if the hold window ends before that ever happens (a real failure mode: the clip
never resolved). post_handoff_fall_step is ticks SINCE handoff_tick (0 = the very first
post-handoff tick) where z first drops below FALL_Z, or -1 if it never does -- -1 whenever
handoff_tick is also -1 (nothing to measure a post-handoff fall against). Same
"first-tick-since-a-reference-point" convention as mujoco_kick_loco_flip_scan.py's own
post_flip_fall_step, deliberately NOT folded into that scan (a scripted forced flip at a
randomized early tick) since this measures the DIFFERENT, undisturbed "let the clip finish on its
own" condition -- see that module's own module docstring for why the two are reported as separate
Table rows, not variants of one number. Two new summary lines: "SUMMARY_HANDOFF_REACHED <step>
<num_reached>/<num_trials> <rate>" (always defined -- num_trials is never 0 by construction) and
"SUMMARY_POST_HANDOFF_ALIVE <step> <num_alive>/<num_reached> <rate_or_NA>" ("NA" when
num_reached is 0 -- no trial ever reached its own auto-handoff within the hold window, nothing to
measure post-handoff survival against).

2026-09-06: kick_aim_theta (appended to RESULT, trailing field, backward-compatible) is the exact
per-trial angle offset (degrees, from this skill's own nominal bearing) that trial's target_xy was
synthesized at and the policy's observation was patched with -- already computed at trial-spawn
time (see the `kick_aim_theta = rng.uniform(...)` draw above), previously used internally but never
surfaced. "NA" when `--kick-aim-enabled` isn't set (no angle was sampled at all -- distinct from a
real 0.0 draw, which DOES happen, e.g. --num-trials 1 with no jitter). Added so a downstream
consumer can bin trials by which angle they actually sampled and compute success-rate-vs-bearing
(needed for the polar coverage figure) instead of only the SUMMARY_DIRECTION/SUMMARY_SUCCESS_<R>
lines' one number pooled across the entire +/-theta_max sweep.

2026-09-05: also (only when --kick-aim-enabled) one "SUMMARY_SUCCESS_<R> <step>
<num_success>/<num_trials> <success_rate>" line PER radius R passed to --success-sigma-m (default
two lines, R in {0.5, 1.0} -- e.g. "SUMMARY_SUCCESS_0.5 500000 41/64 0.6406" and
"SUMMARY_SUCCESS_1 500000 55/64 0.8594" from the SAME 64-trial rollout, no extra MuJoCo cost for
reporting more than one radius since min_target_dist is already computed once per trial regardless).
R is formatted with :g (0.5 -> "0.5", 1.0 -> "1") -- a consumer should match the line prefix
"SUMMARY_SUCCESS_" and parse the trailing radius token, not hardcode specific tags, since
--success-sigma-m accepts any list. This is a DIFFERENT statistic from SUMMARY_DIRECTION, not a
threshold variant of it: denominator is num_trials (a whiff counts as a failure here), matching how
success rate is conventionally reported in prior humanoid-soccer work (e.g. RoboNaldo's
success@0.5m, arXiv:2606.11092) for direct cross-paper comparability, whereas SUMMARY_DIRECTION's
num_hit denominator is this project's own internal convention (a whiff is excluded as ungradeable).
Always printed when --kick-aim-enabled (no NA case -- num_trials is never 0 by construction, unlike
num_hit).

Usage (single trial, original deterministic behavior, byte-identical spawn to the nominal
skill_ball_xy/skill_target_xy metadata):
    /workspaces/isaaclab_arena/submodules/workspaces/conda_env/robojudo/bin/python \
        mujoco_kick_survival_scan.py --onnx-path /path/to/model_0005000.onnx --step-label 5000

Usage (N-trial in-distribution fall-rate scan, jittering ball spawn within the SAME uniform
half-range training itself draws from -- pass the checkpoint's own BallConfig.position_randomization,
read by the CALLER from the live training config, never invented here):
    ... mujoco_kick_survival_scan.py --onnx-path ... --step-label 500000 --num-trials 16 --seed 0 \
        --ball-pos-randomization-x 0.1 --ball-pos-randomization-y 0.1

Usage (kick_aim_enabled checkpoint, sampling kick_aim_theta instead of an independent target draw
-- see --kick-aim-enabled's own help below; 2026-08-22 azimuth-aim refactor, the ONLY way to vary
the observed target now that BallConfig.target_randomization has been removed):
    ... mujoco_kick_survival_scan.py --onnx-path ... --step-label 500000 --num-trials 16 --seed 0 \
        --ball-pos-randomization-x 0.1 --ball-pos-randomization-y 0.1 \
        --kick-aim-enabled --kick-aim-theta-max-deg 15.0 --kick-aim-theta-ref-deg 45.0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROBOJUDO_REPO = "/workspaces/isaaclab_arena/submodules/workspaces/humanoid_deployment/RoboJuDo"
FALL_Z = 0.4

sys.path.insert(0, ROBOJUDO_REPO)
# This file's OWN directory (not a hardcoded sibling fork -- see the bug-fix note in this module's
# own docstring), so `from mujoco_kick_rollout_worker import ...` below always resolves against
# whichever fork this copy of the script actually lives in.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mujoco_kick_rollout_worker import BALL_WORLD_POS, SCENE_WITH_BALL, _install_ball_observation_patch  # noqa: E402

# ------------------------------------------------------------------------------------------------
# POST-STRIKE RECOVERY POSTURE METRICS (--track-posture-metrics, 2026-09-09)
# ------------------------------------------------------------------------------------------------
# Table XI(a)'s ablation removes exactly one thing: kick_recovery_posture_reward_scale 1.0 -> 0.0,
# which disables the SIX-term penalty_kick_recovery_* family (see
# configs/task/task_config_stageB2-abl-a2-noposture.yaml's own header). Every one of those six is a
# CONTINUOUS geometric property of the recovered stance; none of them is "did it stay upright".
# Scoring that ablation on survival alone therefore measures a quantity the ablated reward does not
# directly optimise, which is why the arms did not separate. These six reproduce the training-side
# error terms so the ablation can be scored on what it actually changes.
#
# PHASE. This is the load-bearing detail, and it is why these live in THIS scan (clip plays to
# completion) rather than in mujoco_kick_loco_flip_scan.py (clip aborted mid-strike):
# _kick_recovery_gate (managers/reward/terms/locomotion.py) is ZERO during locomotion-approach and
# strike, and fades in LINEARLY over `grace_steps` ticks only once the clip is past its swinging
# content (`in_kicking_phase` False, i.e. at/after MotionCommand.stand_start_idx). A strike-window
# measurement scores the arms in the one phase where the ablated reward is switched off by its own
# definition. The windows below are keyed on the clip's own stand_start tick (sim2sim_eval.py's
# get_strike_window_ticks returns it as the 2nd element) so they line up with the real gate:
#   EARLY = [stand_start, stand_start + grace) -- the gate's own linear ramp-in.
#   LATE  = [stand_start + grace, end of hold) -- fully gated, settled.
# The early/late split is the same shape mujoco_loco_to_kick_handoff_scan.py's own
# --track-transition-metrics already uses, for the same reason: the difference between the two is
# the transition's signature, not a property of the checkpoint in general.
#
# Each error below is reproduced from its own training-side `_*_error` helper in
# managers/reward/terms/locomotion.py (cited per-term), with the nominal/deadzone parameters taken
# from config_values/unified/g1/reward.py's own _kick_recovery_standing_terms registration -- NOT
# re-chosen here. Reported as RAW ERRORS (never weighted/summed into a scalar): the reward weights
# are a training-time tradeoff, and collapsing six diagnostics into one number would reintroduce
# exactly the "one number that hides the mechanism" problem this rescope exists to fix.
POSTURE_GRACE_STEPS = 50  # _kick_recovery_standing_terms' own grace_steps=50.0, all six terms.
POSTURE_TARGET_HEIGHT = 0.76  # penalty_kick_recovery_stand_height's own target_height.
POSTURE_HEIGHT_DEADZONE = 0.015  # ...and its own deadzone.
POSTURE_ORIENTATION_DEADZONE = 0.025  # penalty_kick_recovery_stand_orientation's own deadzone.
POSTURE_FEET_WIDTH_NOMINAL = 0.24  # penalty_kick_recovery_stand_feet_width's own nominal_width.
POSTURE_FEET_WIDTH_DEADZONE = 0.03  # ...and its own deadzone.
POSTURE_KNEE_WIDTH_NOMINAL = 0.24  # penalty_kick_recovery_stand_knee_width's own nominal_width.
POSTURE_KNEE_WIDTH_DEADZONE = 0.03  # ...and its own deadzone.
# _LEFT_LEG_DOF_IDX / _RIGHT_LEG_DOF_IDX / _LEG_MIRROR_SIGN, locomotion.py:323-325, verbatim.
POSTURE_LEFT_LEG_DOF_IDX = [0, 1, 2, 3, 4, 5]
POSTURE_RIGHT_LEG_DOF_IDX = [6, 7, 8, 9, 10, 11]
POSTURE_LEG_MIRROR_SIGN = [1.0, -1.0, -1.0, 1.0, 1.0, -1.0]
# config_values/robot.py's own foot_body_name="ankle_roll_link" (so feet_indices point at the
# ankle-roll links, not the foot contact-point sites); knee links are the segment one up the leg
# _stand_knee_width_error's own docstring refers to. Both confirmed present in the compiled model.
POSTURE_FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
POSTURE_KNEE_BODIES = ("left_knee_link", "right_knee_link")
POSTURE_METRIC_NAMES = (
    "stance_asymmetry", "yaw_drift", "stand_height", "stand_orientation", "feet_width", "knee_width",
)


def _resolve_posture_body_ids(mujoco, model) -> tuple[tuple[int, int], tuple[int, int]]:
    """(left,right) body ids for the feet and knees. Raises rather than falling back to a guess --
    a silently-wrong body id would produce a plausible-looking width that measures the wrong pair."""
    ids = []
    for names in (POSTURE_FOOT_BODIES, POSTURE_KNEE_BODIES):
        pair = []
        for name in names:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid == -1:
                raise ValueError(f"posture metrics: body {name!r} not found in the compiled model.")
            pair.append(bid)
        ids.append((pair[0], pair[1]))
    return ids[0], ids[1]


def _heading_frame_lateral_separation(np, left_xy, right_xy, base_quat_xyzw) -> float:
    """|lateral component of (left - right) in the base's heading frame| -- the same construction
    _stand_feet_width_error/_stand_knee_width_error use (cos(yaw)*dy - sin(yaw)*dx), with yaw taken
    from the base's own forward vector exactly as those do."""
    x, y, z, w = base_quat_xyzw
    # Forward (+x) axis of the base, rotated into world -- the quat_apply(base_quat, forward) the
    # training terms do, written out for a single xyzw quaternion.
    fwd_x = 1.0 - 2.0 * (y * y + z * z)
    fwd_y = 2.0 * (x * y + z * w)
    base_yaw = np.arctan2(fwd_y, fwd_x)
    dx = left_xy[0] - right_xy[0]
    dy = left_xy[1] - right_xy[1]
    return float(np.abs(np.cos(base_yaw) * dy - np.sin(base_yaw) * dx))


def _posture_errors_now(np, env, inner, foot_ids, knee_ids) -> dict[str, float]:
    """The six penalty_kick_recovery_* error quantities at the current tick, each reproduced from
    its own training-side helper (cited inline). Raw errors, unweighted -- see this section's own
    header comment for why they are never collapsed into a single scalar."""
    # _stance_asymmetry_error (locomotion.py:428) -- sum of squared mirrored left/right deviation
    # from the default pose, over the 6 leg DOFs per side.
    rel = np.asarray(env.dof_pos) - np.asarray(inner.default_dof_pos)
    left = rel[POSTURE_LEFT_LEG_DOF_IDX]
    right = rel[POSTURE_RIGHT_LEG_DOF_IDX]
    stance_asymmetry = float(np.sum(np.square(left - np.array(POSTURE_LEG_MIRROR_SIGN) * right)))

    # _yaw_drift_error (locomotion.py:477) -- squared base yaw angular velocity.
    yaw_drift = float(np.square(env.base_ang_vel[2]))

    # _stand_height_error (locomotion.py:582) -- |base_z - target| beyond deadzone.
    stand_height = float(max(abs(float(env.base_pos[2]) - POSTURE_TARGET_HEIGHT) - POSTURE_HEIGHT_DEADZONE, 0.0))

    # _stand_orientation_error (locomotion.py:824) -- |projected_gravity_xy| beyond deadzone, i.e.
    # sin(tilt from vertical). Gravity rotated into the base frame, the inverse rotation
    # get_projected_gravity performs.
    x, y, z, w = (float(v) for v in env.base_quat)
    # Third row of R^T applied to (0,0,-1): the base-frame components of the world -z axis.
    g_bx = -2.0 * (x * z - w * y)
    g_by = -2.0 * (y * z + w * x)
    stand_orientation = float(max(np.hypot(g_bx, g_by) - POSTURE_ORIENTATION_DEADZONE, 0.0))

    # _stand_feet_width_error (locomotion.py:648) / _stand_knee_width_error (locomotion.py:742).
    xpos = env.data.xpos
    feet_w = _heading_frame_lateral_separation(np, xpos[foot_ids[0]][:2], xpos[foot_ids[1]][:2], env.base_quat)
    knee_w = _heading_frame_lateral_separation(np, xpos[knee_ids[0]][:2], xpos[knee_ids[1]][:2], env.base_quat)
    feet_width = float(max(abs(feet_w - POSTURE_FEET_WIDTH_NOMINAL) - POSTURE_FEET_WIDTH_DEADZONE, 0.0))
    knee_width = float(max(abs(knee_w - POSTURE_KNEE_WIDTH_NOMINAL) - POSTURE_KNEE_WIDTH_DEADZONE, 0.0))

    return {
        "stance_asymmetry": stance_asymmetry,
        "yaw_drift": yaw_drift,
        "stand_height": stand_height,
        "stand_orientation": stand_orientation,
        "feet_width": feet_width,
        "knee_width": knee_width,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-path", required=True)
    parser.add_argument("--settle-s", type=float, default=1.5)
    parser.add_argument("--hold-s", type=float, default=8.0)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--step-label", required=True)
    parser.add_argument(
        "--skill-id", type=int, default=0,
        help="Which of the ONNX's embedded motion skills to kick -- same convention as "
        "mujoco_kick_rollout_worker.py's own --skill-id.",
    )
    parser.add_argument(
        "--num-trials", type=int, default=1,
        help="1 (default) reproduces the original single-deterministic-trial behavior exactly "
        "(no RNG is even constructed when every randomization half-range is 0.0). > 1 requires a "
        "nonzero randomization half-range below to be meaningful -- with everything at 0.0, N "
        "trials just re-run the identical deterministic scenario N times.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed; trial i draws from seed+i.")
    parser.add_argument(
        "--ball-pos-randomization-x", type=float, default=0.0,
        help="Uniform +/- half-range (m) for the ball spawn x, drawn fresh per trial around that "
        "skill's own nominal get_skill_ball_xy. 0.0 (default) = no jitter, exact nominal spawn "
        "every trial. Pass the checkpoint's OWN BallConfig.position_randomization[0] -- this "
        "script never invents a range.",
    )
    parser.add_argument("--ball-pos-randomization-y", type=float, default=0.0)
    parser.add_argument(
        "--ball-pos-absolute-x", type=float, default=None,
        help="2026-09-10: place the ball at this ABSOLUTE world x (m) instead of jittering around "
        "the skill's own trained get_skill_ball_xy -- for sweeping a shared placement region across "
        "several skills (e.g. Table/Fig 'ball placement heatmap'), where the whole point is testing "
        "positions OUTSIDE any one skill's own trained box. Requires --ball-pos-absolute-y too. "
        "nominal_bearing_deg (the skill's own trained aim DIRECTION) is computed from the "
        "checkpoint's own untouched ball_xy/target_xy metadata BEFORE this override is applied, so "
        "the commanded target still anchors D meters from the NEW ball position along the skill's "
        "OWN trained bearing -- see the override's own inline comment for why that ordering matters. "
        "--ball-pos-randomization-x/-y still apply ON TOP of this (jitter around the absolute point, "
        "same as they'd jitter around the trained nominal point) -- pass 0.0 for a fixed placement.",
    )
    parser.add_argument("--ball-pos-absolute-y", type=float, default=None)
    parser.add_argument(
        "--ball-pos-range-x", type=float, nargs=2, default=None, metavar=("LO", "HI"),
        help="2026-09-10: draw a FRESH absolute ball x (m) per trial, uniform over [LO, HI] -- for "
        "a continuous placement sweep across many trials in ONE subprocess (e.g. a ball-placement "
        "hit-rate heatmap), as opposed to --ball-pos-absolute-x's single fixed point. Mutually "
        "exclusive in effect with --ball-pos-randomization-x (that jitters around ONE nominal/"
        "absolute point; this REPLACES the point itself every trial) -- if both are passed, this "
        "wins (ball_dx/dy below are computed as this draw's own delta from nominal_ball_xy, so the "
        "existing randomization-x/y draw, if also requested, would just be overwritten a moment "
        "later, not additive -- pass 0.0 for randomization-x/y when using this to avoid confusion). "
        "Requires --ball-pos-range-y too. Same 'bearing computed from UNTOUCHED metadata, target "
        "anchors to wherever this trial's ball actually landed' contract as --ball-pos-absolute-x.",
    )
    parser.add_argument("--ball-pos-range-y", type=float, nargs=2, default=None, metavar=("LO", "HI"))
    parser.add_argument(
        "--kick-aim-enabled", action="store_true",
        help="2026-08-22, azimuth-aim refactor: ONLY pass this for a checkpoint actually trained "
        "with kick_aim_enabled=True. The old independent target-point jitter (BallConfig."
        "target_randomization) was removed from the config layer entirely, so this is now the "
        "ONLY way to vary the observed target -- samples kick_aim_theta (uniform, +/- "
        "--kick-aim-theta-max-deg) around the skill's own calibrated nominal bearing -- derived "
        "from get_skill_ball_xy/get_skill_target_xy exactly like SkillConfig."
        "resolved_nominal_bearing_deg() derives it "
        "from x/y/target_x/target_y, so no separate bearing flag is needed.",
    )
    parser.add_argument(
        "--kick-aim-theta-max-deg", type=float, default=15.0,
        help="Uniform +/- half-range (degrees) for kick_aim_theta, only used when "
        "--kick-aim-enabled. Pass the checkpoint's own MultiSkillConfig/BallConfig."
        "kick_aim_theta_max_deg (or this skill's own override).",
    )
    parser.add_argument(
        "--kick-aim-theta-ref-deg", type=float, default=45.0,
        help="Normalization reference (degrees), only used when --kick-aim-enabled. MUST match "
        "the checkpoint's own MultiSkillConfig/BallConfig.kick_aim_theta_ref_deg -- a mismatch "
        "silently rescales every sampled theta before it reaches the policy.",
    )
    parser.add_argument(
        "--kick-aim-nominal-distance-m", type=float, default=5.0,
        help="2026-08-23: the fixed distance D each trial's commanded target point is synthesized "
        "at, only used when --kick-aim-enabled (for the direction-success-rate measurement -- see "
        "this module's own docstring). MUST match the checkpoint's own MultiSkillConfig/"
        "BallConfig.kick_aim_nominal_distance_m -- that field's own default, 5.0, is what this "
        "mirrors.",
    )
    parser.add_argument(
        "--direction-success-sigma-m", type=float, default=1.0,
        help="2026-08-23: a trial counts as a direction success iff it hit the ball AND the "
        "ball's closest approach to the commanded target point was within this many meters. "
        "Mirrors error_ball_to_target's own sigma default (shooting.py) exactly, rather than "
        "inventing a separate eval-only threshold -- at the kick_aim_nominal_distance_m default "
        "(5.0), sigma=1.0 implies roughly asin(1.0/5.0)=~11.5deg of angular tolerance (see "
        "MultiSkillConfig.kick_error_ball_to_target_sigma's own docstring for that derivation). "
        "No config in this project currently overrides kick_error_ball_to_target_sigma away from "
        "its default -- pass this explicitly if that ever changes.",
    )
    parser.add_argument(
        "--success-sigma-m", type=float, nargs="+", default=[0.5, 1.0],
        help="2026-09-05: one or more radii (meters); a trial counts as a (strict) success at "
        "radius R iff the ball's closest approach to the commanded target point was within R -- "
        "UNCONDITIONAL on hit_step (denominator is num_trials, not num_hit: a whiff naturally "
        "fails at every radius too, since min_target_dist stays at its huge init value when the "
        "ball is never touched). This is a DIFFERENT statistic from --direction-success-sigma-m "
        "above, not a threshold variant of it: direction success excludes whiffs as ungradeable "
        "(prior-work-internal convention), while this one counts a whiff as a failure, matching "
        "how success rate is conventionally reported in prior humanoid-soccer work (e.g. "
        "RoboNaldo's success@0.5m, arXiv:2606.11092) for direct cross-paper comparability. "
        "min_target_dist is computed once per trial regardless of how many radii are passed, so "
        "reporting several costs nothing extra -- no need to re-run the (expensive, real-MuJoCo) "
        "scan once per radius. Emits one SUMMARY_SUCCESS_<R> line per (deduplicated, sorted) "
        "radius -- see this module's own docstring. Default [0.5, 1.0]: 0.5m mirrors RoboNaldo's "
        "convention, 1.0m mirrors this project's own native error_ball_to_target sigma (also see "
        "--direction-success-sigma-m, whose num_hit-denominator version of 1.0m is a separate, "
        "already-existing statistic from this one).",
    )
    parser.add_argument(
        "--trajectory-output-path", type=str, default=None,
        help="2026-09-07: when set, writes ONE JSON to this path after the scan finishes, holding "
        "every trial's full ball (x, y) path over the hold window (post-trigger only, matching "
        "min_target_dist's own tracked window -- pre-trigger settle is excluded since the ball is "
        "just resting at its jittered spawn point then, not doing anything a trajectory plot would "
        "want to show). Purely additive: costs one extra ball-position array read per tick (already "
        "computed for min_target_dist when kick_aim is enabled; here read unconditionally instead), "
        "changes no existing RESULT/SUMMARY stdout line, and defaults to None (no file written, "
        "zero overhead) so every existing caller is unaffected. Shape: {'step_label', 'skill_id', "
        "'trials': [{'trial', 'kick_aim_theta' (float or null), 'fall_step', 'hit_step', "
        "'trajectory_xy': [[x, y], ...], 'trajectory_speed': [m/s, ...]}, ...]}. "
        "'trajectory_speed' (2026-09-07) is the same 2D ball speed max_ball_speed is built from "
        "(norm of ball qvel xy, vertical bounce excluded), sampled at EVERY recorded tick rather "
        "than only post-contact -- for speed-colored trajectory plots. max_ball_speed's own "
        "post-contact-only semantics are unchanged.",
    )
    parser.add_argument(
        "--track-posture-metrics", action="store_true",
        help="Additionally report the SIX penalty_kick_recovery_* posture error quantities over "
        "the clip's post-swing recovery tail, split into the gate's own EARLY (grace ramp) and "
        "LATE (settled) windows -- see this module's own POST-STRIKE RECOVERY POSTURE METRICS "
        "section for the per-term derivation and why this scan, not the strike-window flip scan, "
        "is where they belong. Off by default; requires --stand-start-tick. Adds no rollout cost "
        "(per-tick bookkeeping inside the existing hold loop only) and changes no existing "
        "RESULT/SUMMARY line.",
    )
    parser.add_argument(
        "--stand-start-tick", type=int, default=None,
        help="This skill's clip-relative stand_start tick -- the boundary where in_kicking_phase "
        "goes False and _kick_recovery_gate begins its linear ramp-in. sim2sim_eval.py's own "
        "get_strike_window_ticks returns it as the 2nd element of its tuple; pass that, do not "
        "guess. Required by (and only used with) --track-posture-metrics.",
    )
    parser.add_argument(
        "--hit-window-lo-tick", type=int, default=None,
        help="2026-09-10: gate ball<->foot contact detection (hit_step) to this hold-loop tick "
        "(inclusive, 0-indexed, counted from kick trigger -- the SAME 'i' this file's own hold "
        "loop already counts in, NOT inner.curr_motion_timestep -- see this flag's own --hit-"
        "window-hi-tick docstring for why that distinction matters) onward. Off by default (every "
        "tick from trigger to hold end is eligible, this file's original behavior, unchanged) -- "
        "without this, an incidental late-hold-window bump (the policy auto-returns to locomotion "
        "and keeps walking for the rest of the hold window once the clip ends) counts as a 'hit' "
        "indistinguishably from a real strike, which inflates any hit-rate measurement taken over "
        "a displaced/off-nominal ball placement where the strike itself is more likely to miss. "
        "sim2sim_eval.py's own get_strike_window_ticks(onnx_path, skill_id) returns (lo, hi) "
        "ALREADY in this 'ticks since trigger' unit -- pass its first element here directly, do "
        "not add skill_motion_start_idx (that offset is for inner.curr_motion_timestep, a "
        "DIFFERENT counter -- see --track-posture-metrics' own since_stand computation, which "
        "compares against curr_motion_timestep and is therefore only correct for skill_id 0 in a "
        "multi-skill checkpoint; do not copy that pattern for hit-window gating). Requires "
        "--hit-window-hi-tick too.",
    )
    parser.add_argument(
        "--hit-window-hi-tick", type=int, default=None,
        help="Inclusive upper bound, same 'ticks since trigger' unit as --hit-window-lo-tick. Pass "
        "get_strike_window_ticks's own 2nd element MINUS 1 (that function returns stand_start_tick, "
        "the boundary where the swing ends -- sim2sim_eval.py's own kick_to_loco_flip strike-window "
        "split already uses this exact 'stand_start_tick - 1' inclusive convention, replicated here "
        "for the same reason: a contact exactly AT stand_start_tick is post-swing, not a strike).",
    )
    parser.add_argument(
        "--posture-grace-steps", type=int, default=POSTURE_GRACE_STEPS,
        help="Ticks after --stand-start-tick that count as the EARLY window, matching "
        "_kick_recovery_standing_terms' own grace_steps. Override only for a checkpoint trained "
        "with a different grace.",
    )
    parser.add_argument(
        "--debug-trigger-frame", action="store_true",
        help="2026-09-12: print one DEBUG_TRIGGER_FRAME line per trial giving the ball's position "
        "in the ROBOT's heading frame at the kick trigger, alongside the nominal (spawn-time) "
        "offset it is supposed to equal and the robot's own world position. This is the direct "
        "measurement of the settle drift --reanchor-ball-at-trigger exists to remove, and the only "
        "honest way to verify that flag: outcome rates (hit/success) are far too noisy at "
        "practical trial counts to confirm a ~4cm placement change, whereas ball_b minus nominal "
        "is the artifact itself, measured to the millimetre. Expect |ball_b - nominal| ~= 0.043 m "
        "with the flag OFF and ~0 with it ON. Off by default; costs one quaternion rotation per "
        "trial when on.",
    )
    parser.add_argument(
        "--reanchor-ball-at-trigger", action="store_true",
        help="2026-09-12: re-place the ball (and the commanded target) relative to the robot's "
        "ACTUAL pose at the kick trigger, instead of leaving it at the world position it was "
        "spawned at before the settle. Off by default, so every previously-reported number "
        "reproduces bit-for-bit.\n\n"
        "WHY THIS EXISTS. This scan spawns the ball at a fixed WORLD point, then holds a zero "
        "velocity command for --settle-s before triggering. The robot does not hold still during "
        "that settle: measured across three checkpoints it walks BACKWARD 4.26-4.60 cm and yaws "
        "about -1.5 deg, deterministically. So by trigger time the ball sits ~4.3 cm further out "
        "in the robot's own frame than the skill was trained to expect -- a systematic offset "
        "against a +/-10 cm jitter, in the same direction every single trial.\n\n"
        "Training never has this offset. BOTH of training's ball-placement paths anchor the ball "
        "to the robot's live pose at placement time: managers/.../place_ball_at_entry reads the "
        "robot pose straight from the simulator, and reset() anchors to target_root_pos/"
        "target_root_rot (the actual post-noise pose), both via local_xy_to_world. "
        "mujoco_loco_to_kick_handoff_scan.py already does the same thing at its own flip instant "
        "(see that file's 'Flip instant' comment) -- which is precisely why its hit rate is not "
        "depressed the way this scan's is. This flag makes the two scans agree with each other "
        "AND with training. Measured effect (2 checkpoints with headroom, 2x2 against aim theta): "
        "+16.7/+13.3 pts on skill_1 @441k, +6.7/+5.0 pts on skill_2 @392k.",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    import mujoco
    import numpy as np

    import robojudo.config.g1  # noqa: F401
    import robojudo.pipeline  # noqa: F401
    from robojudo.config import cfg_registry
    # Same import-not-duplicate precedent mujoco_kick_rollout_worker.py already established for
    # this exact constant (line ~263 there: `from robojudo.policy.unified_loco_kick_policy import
    # _TASK_KICK`) -- read the deployed policy's own string values rather than re-declaring them
    # here, so a future rename can't silently desync this scan's handoff detection from the real
    # thing it is watching.
    from robojudo.policy.unified_loco_kick_policy import _TASK_LOCOMOTION
    # Same two helpers mujoco_loco_to_kick_handoff_scan.py uses for its own flip-instant ball
    # placement, and the numpy equivalents of training's own local_xy_to_world -- imported
    # unconditionally (cheap, already a dependency via the observation patch) rather than under
    # --reanchor-ball-at-trigger, so a typo here fails at import, not 40 minutes into a sweep.
    from robojudo.utils.util_func import calc_heading_quat_np, my_quat_rotate_np, quat_rotate_inverse_np

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
    # 2026-09-05: velocity address is DISTINCT from qpos address for a free joint (7 qpos: xyz +
    # wxyz quat; 6 dof: linear xyz + angular xyz) -- same jnt_dofadr pattern already used to ZERO
    # ball velocity in mujoco_loco_to_kick_handoff_scan.py, reused here to READ it instead. qvel's
    # first 3 components are linear velocity in the WORLD frame for a free joint (standard MuJoCo
    # convention, not project-specific).
    ball_qvel_addr = int(env.model.jnt_dofadr[ball_jid])

    # 2026-08-21, user-requested: alongside the fall-rate scan above, also report a ball CONTACT
    # HIT rate from the SAME N trials, no extra rollout needed. Unlike training's own has_kicked
    # (managers/reward/terms/shooting.py's _detect_kick), which on IsaacSim reads real PhysX
    # ContactSensors and on every OTHER backend (including MuJoCo, if it were ever used for
    # training) falls back to an offset-point-plus-margin approximation, this scan has direct
    # access to MuJoCo's own contact solver output (env.data.contact) -- so it checks the ACTUAL
    # geom-geom contact list for the ball against every foot collision geom, no approximation. Geom
    # ids resolved ONCE here (not per-tick) since XML/geom naming (scene_g1_29dof_with_ball.xml,
    # g1_29dof.xml's <default class="collision"> geoms named "{left,right}_footN_collision") is
    # fixed for the lifetime of this compiled model. ball_geom's contype=1/conaffinity=7 vs the
    # foot geoms' contype=1/conaffinity=1 (both files, read directly rather than assumed) already
    # satisfies MuJoCo's (contype1 & conaffinity2) pairwise test, so contacts between them are
    # generated automatically -- no explicit <pair> needed, confirmed by reading both XMLs.
    ball_geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
    assert ball_geom_id != -1, "ball_geom not found in compiled model -- is SCENE_WITH_BALL correct?"

    posture_foot_ids = posture_knee_ids = None
    if args.track_posture_metrics:
        if args.stand_start_tick is None:
            raise ValueError(
                "--track-posture-metrics requires --stand-start-tick (this skill's clip-relative "
                "stand_start boundary). sim2sim_eval.py's get_strike_window_ticks returns it as "
                "the 2nd tuple element -- pass that rather than guessing a value, since the whole "
                "point of these metrics is that they line up with _kick_recovery_gate's own phase."
            )
        posture_foot_ids, posture_knee_ids = _resolve_posture_body_ids(mujoco, env.model)
    foot_geom_ids = {
        gid
        for gid in range(env.model.ngeom)
        if (name := mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, gid)) is not None
        and name.endswith("_collision")
        and (name.startswith("left_foot") or name.startswith("right_foot"))
    }
    assert foot_geom_ids, "no left_foot*_collision/right_foot*_collision geoms found -- model changed?"

    def _ball_foot_contact_now() -> bool:
        """True iff THIS tick's already-computed contact list (env.data.contact, populated by the
        physics step(s) inside the just-finished pl.step()) contains a ball<->foot pair. Same
        once-per-control-tick sampling granularity as training's own foot-ball contact check (see
        shooting.py's _KICK_DETECT_FOOT_CONTACT_MARGIN_M docstring on why a discrete sample can
        miss a fleeting graze between ticks) -- a real contact that both starts and clears within
        one 50Hz tick's decimated substeps could be missed, same caveat, not fixed here."""
        contacts = env.data.contact
        for c in range(env.data.ncon):
            g1, g2 = int(contacts.geom1[c]), int(contacts.geom2[c])
            if (g1 == ball_geom_id and g2 in foot_geom_ids) or (g2 == ball_geom_id and g1 in foot_geom_ids):
                return True
        return False

    nominal_ball_xy = inner.get_skill_ball_xy(args.skill_id)
    if nominal_ball_xy is None:
        nominal_ball_xy = (BALL_WORLD_POS[0], BALL_WORLD_POS[1])
    nominal_target_xy = inner.get_skill_target_xy(args.skill_id)
    # Same TARGET_WORLD_POS-shaped fallback _install_ball_observation_patch itself uses when this
    # is None -- read it back out via the patch below rather than duplicate the constant here;
    # passing None through as target_xy_override in that case preserves that exact fallback.

    if args.kick_aim_enabled:
        if nominal_target_xy is None:
            raise ValueError(
                "--kick-aim-enabled requires this checkpoint's ONNX to carry skill_target_xy "
                "metadata (to derive the nominal bearing from) -- get_skill_target_xy returned "
                "None for this skill_id."
            )
        # Same atan2 convention as SkillConfig.resolved_nominal_bearing_deg() -- derived from the
        # same two points that training's own calibration wrote into the yaml, not re-declared here.
        nominal_bearing_deg = float(
            np.degrees(
                np.arctan2(
                    nominal_target_xy[1] - nominal_ball_xy[1], nominal_target_xy[0] - nominal_ball_xy[0]
                )
            )
        )
        # Diagnostic only -- not consumed by the trial loop below (the policy only ever needs
        # kick_aim_theta itself, not the absolute bearing; see _install_ball_observation_patch's
        # own docstring). Printed so a run of this script sanity-checks against the calibrated
        # value scripts/calibrate_nominal_bearing.py produced for this skill.
        print(f"[kick_aim] skill {args.skill_id} nominal_bearing_deg={nominal_bearing_deg:.2f}", flush=True)
    else:
        nominal_bearing_deg = None

    # 2026-09-10: absolute ball-placement override, for sweeping ball position OUTSIDE any one
    # skill's own trained box (e.g. a shared XY region spanning several skills' boxes) -- see
    # this module's own --ball-pos-absolute-x/-y docstring. Applied AFTER nominal_bearing_deg is
    # computed above, not before: that ordering is load-bearing. nominal_bearing_deg must come
    # from the skill's OWN trained (ball_xy, target_xy) metadata, unaffected by where we're about
    # to move the ball -- the whole point is "keep this skill's trained aim DIRECTION, move the
    # ball (and therefore the target, which anchors to wherever the ball actually is -- see
    # target_xy's own construction below) to a new spot". Computing the bearing from the NEW ball
    # position against the OLD (unmoved) nominal_target_xy would instead give the bearing FROM the
    # displaced ball TO the original target's fixed world position -- a different, undesired
    # quantity that also isn't at the standard kick_aim_nominal_distance_m from the new ball spot.
    if (args.ball_pos_absolute_x is None) != (args.ball_pos_absolute_y is None):
        raise ValueError("--ball-pos-absolute-x and --ball-pos-absolute-y must be given together.")
    if args.ball_pos_absolute_x is not None:
        nominal_ball_xy = (args.ball_pos_absolute_x, args.ball_pos_absolute_y)
    if (args.hit_window_lo_tick is None) != (args.hit_window_hi_tick is None):
        raise ValueError("--hit-window-lo-tick and --hit-window-hi-tick must be given together.")
    if (args.ball_pos_range_x is None) != (args.ball_pos_range_y is None):
        raise ValueError("--ball-pos-range-x and --ball-pos-range-y must be given together.")

    settle_steps = int(args.settle_s * args.fps)
    hold_steps = int(args.hold_s * args.fps)
    trigger_cmd = "[TRIGGER_KICK]" if args.skill_id == 0 else f"[TRIGGER_KICK:{args.skill_id}]"

    any_jitter = (
        args.ball_pos_randomization_x > 0.0 or args.ball_pos_randomization_y > 0.0
        or (args.kick_aim_enabled and args.kick_aim_theta_max_deg > 0.0)
        or args.ball_pos_range_x is not None
    )
    rng = np.random.default_rng(args.seed) if (args.num_trials > 1 or any_jitter) else None

    try:
        num_fell = 0
        num_hit = 0
        num_direction_hit = 0
        num_handoff_reached = 0
        num_post_handoff_alive = 0
        # Per-trial means, one entry per contributing trial, per window. Averaged across trials at
        # print time -- a per-trial mean first (not a flat pool of every tick) so a trial that
        # happened to spend more ticks in the window does not get more weight than one that spent
        # fewer, matching mujoco_loco_to_kick_handoff_scan.py's own transition-metric convention.
        posture_trial_means: dict[str, list[float]] = {
            f"{name}_{window}": [] for name in POSTURE_METRIC_NAMES for window in ("early", "late")
        }
        num_posture_trials = 0
        success_sigmas = sorted(set(args.success_sigma_m))  # dedupe + deterministic print order
        num_success = {sigma: 0 for sigma in success_sigmas}
        # 2026-09-05: shot error and peak ball speed, both gated on hit_step != -1 (a whiff has no
        # shot to grade and no post-contact speed to measure -- same "exclude the ungradeable case"
        # convention already established for num_direction_hit above and for training's own
        # mean_speed_of_hits in shooting.py, not a new convention invented here).
        hit_shot_errors = []
        hit_peak_speeds = []
        trajectory_trials = [] if args.trajectory_output_path else None
        for trial in range(args.num_trials):
            mujoco.mj_resetDataKeyframe(env.model, env.data, 0)
            inner.reset()
            inner._update_velocity_command = lambda cd, ball_pos_b=None: None

            if rng is not None:
                ball_dx = rng.uniform(-args.ball_pos_randomization_x, args.ball_pos_randomization_x)
                ball_dy = rng.uniform(-args.ball_pos_randomization_y, args.ball_pos_randomization_y)
                if args.ball_pos_range_x is not None:
                    # Overwrites the jitter draw above with an absolute per-trial placement,
                    # expressed as ITS OWN delta from nominal_ball_xy so every downstream use of
                    # ball_dx/dy (spawn placement, target anchor, trajectory recording) needs no
                    # separate code path -- see this flag's own docstring for why it wins over
                    # --ball-pos-randomization-x/y rather than composing with it.
                    ball_dx = rng.uniform(*args.ball_pos_range_x) - nominal_ball_xy[0]
                    ball_dy = rng.uniform(*args.ball_pos_range_y) - nominal_ball_xy[1]
                kick_aim_theta = (
                    rng.uniform(-args.kick_aim_theta_max_deg, args.kick_aim_theta_max_deg)
                    if args.kick_aim_enabled
                    else None
                )
            else:
                ball_dx = ball_dy = 0.0
                kick_aim_theta = 0.0 if args.kick_aim_enabled else None

            env.data.qpos[ball_qpos_addr] = nominal_ball_xy[0] + ball_dx
            env.data.qpos[ball_qpos_addr + 1] = nominal_ball_xy[1] + ball_dy

            # 2026-08-23: this trial's REAL commanded target point, for the direction-success
            # measurement below -- same construction as managers/command/terms/wbt.py's own
            # _synthesize_kick_aim_target_local: ball_local_placed + D * unit(bearing + theta).
            # Uses THIS trial's actual (jittered) ball spawn, not the nominal one, matching
            # training's own "target is anchored to where the ball actually landed" contract.
            target_xy = None
            if args.kick_aim_enabled:
                theta_rad = np.radians(nominal_bearing_deg + kick_aim_theta)
                ball_actual_xy = np.array([nominal_ball_xy[0] + ball_dx, nominal_ball_xy[1] + ball_dy])
                target_xy = ball_actual_xy + args.kick_aim_nominal_distance_m * np.array(
                    [np.cos(theta_rad), np.sin(theta_rad)]
                )

            # No independent target jitter (BallConfig.target_randomization was removed 2026-08-22)
            # -- target_xy_override=None lets _install_ball_observation_patch fall back to
            # inner.get_skill_target_xy(skill_id) (== nominal_target_xy) for a non-kick_aim
            # checkpoint, and that fallback is IGNORED entirely when kick_aim_theta is not None
            # (see that function's own docstring).
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

            z_series = []
            for _ in range(settle_steps):
                step_zero_vel()
                z_series.append(float(env.base_pos[2]))

            # Re-anchor the ball to the robot's ACTUAL pose right now, undoing the settle drift
            # (see --reanchor-ball-at-trigger's own help text for the measured magnitude and why
            # training never has this offset). Everything here is computed in the robot's heading
            # frame and pushed to world exactly once, so the SAME (ball_dx, ball_dy) draw and the
            # SAME nominal bearing now mean what they meant at training time: an offset from the
            # robot, not from the world origin the robot has since walked away from.
            if args.reanchor_ball_at_trigger:
                robot_pos_w = env.base_pos.copy()
                heading_quat = calc_heading_quat_np(env.base_quat.copy())

                def _to_world(local_xy):
                    local_xyz = np.array([local_xy[0], local_xy[1], 0.0])
                    return my_quat_rotate_np(heading_quat, local_xyz)[:2] + robot_pos_w[:2]

                ball_world_xy = _to_world((nominal_ball_xy[0] + ball_dx, nominal_ball_xy[1] + ball_dy))
                env.data.qpos[ball_qpos_addr] = ball_world_xy[0]
                env.data.qpos[ball_qpos_addr + 1] = ball_world_xy[1]
                # z left exactly as the settle left it (the ball has been resting on the floor for
                # --settle-s and is already at its own rest height -- unlike the handoff scan, which
                # teleports a ball that was never placed on this floor and so must set z itself).
                # Velocity zeroed for the same reason that scan zeroes it: a teleport must not carry
                # whatever residual drift the contact solver left in the free joint.
                env.data.qvel[ball_qvel_addr : ball_qvel_addr + 6] = 0.0

                # The commanded target is a direction in the ROBOT's frame, so it has to rotate with
                # the robot too -- re-derived here from the same local construction used above
                # rather than rotating the stale world point, which would double-apply the drift.
                if target_xy is not None:
                    theta_rad = np.radians(nominal_bearing_deg + kick_aim_theta)
                    target_xy = _to_world(
                        (
                            nominal_ball_xy[0] + ball_dx + args.kick_aim_nominal_distance_m * np.cos(theta_rad),
                            nominal_ball_xy[1] + ball_dy + args.kick_aim_nominal_distance_m * np.sin(theta_rad),
                        )
                    )

                # Re-install so the NON-kick_aim path observes a target that moved with the robot as
                # well. In kick_aim mode target_xy_override is ignored entirely (obs[157:159] is the
                # world-frame-independent [theta/ref, 0] command), so this is a no-op there -- but
                # leaving the old install in place would silently feed a stale world target to any
                # kick_aim_enabled=False checkpoint run under this flag.
                nominal_target_world = (
                    tuple(_to_world(nominal_target_xy)) if nominal_target_xy is not None else None
                )
                _install_ball_observation_patch(
                    env, inner, mujoco, np, ball_qpos_addr, args.skill_id,
                    target_xy_override=nominal_target_world,
                    kick_aim_theta_deg=kick_aim_theta,
                    kick_aim_theta_ref_deg=args.kick_aim_theta_ref_deg,
                )
                mujoco.mj_forward(env.model, env.data)

            # Measured at the trigger, AFTER any re-anchoring above -- the same heading-frame
            # transform _install_ball_observation_patch feeds the policy as obs[29:32], so this is
            # literally what the policy is about to see, not a reconstruction of it.
            if args.debug_trigger_frame:
                dbg_ball_b = quat_rotate_inverse_np(
                    calc_heading_quat_np(env.base_quat.copy()),
                    env.data.qpos[ball_qpos_addr : ball_qpos_addr + 3].copy() - env.base_pos.copy(),
                )
                dbg_nom_x = nominal_ball_xy[0] + ball_dx
                dbg_nom_y = nominal_ball_xy[1] + ball_dy
                print(
                    f"DEBUG_TRIGGER_FRAME {args.step_label} {trial} "
                    f"ball_b=({float(dbg_ball_b[0]):.4f},{float(dbg_ball_b[1]):.4f}) "
                    f"nominal=({dbg_nom_x:.4f},{dbg_nom_y:.4f}) "
                    f"err=({float(dbg_ball_b[0]) - dbg_nom_x:+.4f},{float(dbg_ball_b[1]) - dbg_nom_y:+.4f}) "
                    f"robot_xy=({float(env.base_pos[0]):+.4f},{float(env.base_pos[1]):+.4f})",
                    flush=True,
                )

            env.update()
            inner.get_observation(env.get_data(), {})
            inner.post_step_callback([trigger_cmd])

            fall_step = -1
            hit_step = -1
            handoff_tick = -1
            post_handoff_fall_step = -1
            # Per-tick posture errors for THIS trial, split by the recovery gate's own windows.
            # Accumulated only when --track-posture-metrics; empty lists otherwise (cheap no-op).
            posture_early: list[dict[str, float]] = []
            posture_late: list[dict[str, float]] = []
            # inf, not -1: this is a MINIMUM being tracked (mirrors shooting.py's own
            # min_target_dist latch), unlike fall_step/hit_step which are "first tick where X
            # happened" trackers -- -1 there means "never happened", but -1 here would look like a
            # (nonsensical, negative) distance. Converted to the -1 sentinel only at print time,
            # for a target_xy is None trial (kick_aim disabled -- no commanded direction at all).
            min_target_dist = float("inf")
            max_ball_speed = 0.0
            trial_trajectory_xy = [] if trajectory_trials is not None else None
            trial_trajectory_speed = [] if trajectory_trials is not None else None
            for i in range(hold_steps):
                step_zero_vel()
                z = float(env.base_pos[2])
                z_series.append(z)
                if fall_step == -1 and z < FALL_Z:
                    fall_step = i
                # One-way transition (KICK -> LOCOMOTION only, see _return_to_loco()'s own three
                # call sites in unified_loco_kick_policy.py) -- task_mode starts this trial at KICK
                # (set by the [TRIGGER_KICK] callback just above) and never flips back to KICK
                # within this same trial, so the first tick it reads LOCOMOTION again is
                # unambiguously the auto-handoff instant.
                if handoff_tick == -1 and inner.task_mode == _TASK_LOCOMOTION:
                    handoff_tick = i
                if handoff_tick != -1 and post_handoff_fall_step == -1 and z < FALL_Z:
                    post_handoff_fall_step = i - handoff_tick
                if args.track_posture_metrics:
                    # Keyed on the CLIP's own timestep, not the hold-loop tick: the gate is defined
                    # against MotionCommand.stand_start_idx (clip-relative), and the clip does not
                    # start at hold-loop tick 0 (the trigger fires the tick before this loop, then
                    # the approach/strike content plays first). curr_motion_timestep is the same
                    # counter post_step_callback clamps against the clip length.
                    clip_t = int(inner.curr_motion_timestep)
                    since_stand = clip_t - args.stand_start_tick
                    if since_stand >= 0:
                        errs = _posture_errors_now(np, env, inner, posture_foot_ids, posture_knee_ids)
                        (posture_early if since_stand < args.posture_grace_steps else posture_late).append(errs)
                # Only checked from the trigger onward (not during settle above) -- ball and feet
                # are far apart pre-trigger by construction (nominal_ball_xy ~1.3m out), so a
                # settle-phase contact would only ever mean the ball spawn itself is degenerate,
                # not a kick. Additionally gated to [--hit-window-lo-tick, --hit-window-hi-tick]
                # when both are given (None = unrestricted, this file's original behavior) -- see
                # those flags' own docstrings for why an ungated window over-counts incidental
                # post-clip contact as a "hit" (this matters for --ball-pos-absolute-x/-y sweeps,
                # where a genuine strike miss is exactly the case an ungated window would hide).
                in_hit_window = (
                    args.hit_window_lo_tick is None
                    or args.hit_window_lo_tick <= i <= args.hit_window_hi_tick
                )
                if hit_step == -1 and in_hit_window and _ball_foot_contact_now():
                    hit_step = i
                # 2D (xy) speed, matching shooting.py's ball_speed = norm(ball_vel_xy) exactly
                # -- vertical bounce is deliberately excluded from this project's own "how fast
                # does the ball travel toward the target" convention. max_ball_speed still only
                # ever updates AFTER contact (unchanged semantics); the unconditional read below
                # exists because trajectory recording wants a speed at EVERY tick, including the
                # pre-contact ones, to color a trajectory by speed along its length.
                ball_speed_now = (
                    float(np.linalg.norm(env.data.qvel[ball_qvel_addr : ball_qvel_addr + 2]))
                    if (hit_step != -1 or trial_trajectory_xy is not None)
                    else None
                )
                if hit_step != -1 and ball_speed_now > max_ball_speed:
                    max_ball_speed = ball_speed_now
                if target_xy is not None or trial_trajectory_xy is not None:
                    # Read unconditionally when trajectory recording is on (see --trajectory-
                    # output-path) even for a target_xy is None trial -- a trajectory plot wants
                    # to show where the ball went regardless of whether kick_aim was enabled to
                    # give it a target to grade against.
                    ball_xy_now = np.array(
                        [env.data.qpos[ball_qpos_addr], env.data.qpos[ball_qpos_addr + 1]]
                    )
                    if trial_trajectory_xy is not None:
                        trial_trajectory_xy.append([float(ball_xy_now[0]), float(ball_xy_now[1])])
                        trial_trajectory_speed.append(ball_speed_now)
                    if target_xy is not None:
                        # Same "closest approach so far, not gated to a narrow strike window"
                        # contract as shooting.py::error_ball_to_target's own min_target_dist --
                        # the ball keeps rolling well past the strike, so the true closest
                        # approach can happen many ticks after hit_step.
                        dist = float(np.linalg.norm(ball_xy_now - target_xy))
                        if dist < min_target_dist:
                            min_target_dist = dist

            min_z = min(z_series)
            # Posture is only meaningful for a trial that actually stayed up: a toppled robot's
            # stance geometry describes the topple, not the recovered stance the ablated reward
            # shapes. Same "keep did-it-fall separate from how-well-did-it-X" split this scan
            # already applies elsewhere -- fall_rate reports the excluded population.
            if args.track_posture_metrics and fall_step == -1 and (posture_early or posture_late):
                num_posture_trials += 1
                for window, samples in (("early", posture_early), ("late", posture_late)):
                    if not samples:
                        continue
                    for name in POSTURE_METRIC_NAMES:
                        posture_trial_means[f"{name}_{window}"].append(
                            float(np.mean([s[name] for s in samples]))
                        )
            if fall_step != -1:
                num_fell += 1
            if handoff_tick != -1:
                num_handoff_reached += 1
                if post_handoff_fall_step == -1:
                    num_post_handoff_alive += 1
            if hit_step != -1:
                num_hit += 1
                # Direction success REQUIRES a real hit -- a whiff has no departure direction to
                # grade (see this module's own docstring on why the rate below is num_direction_hit
                # / num_hit, not / num_trials).
                if target_xy is not None and min_target_dist <= args.direction_success_sigma_m:
                    num_direction_hit += 1
            # Strict success (2026-09-05): unconditional on hit_step, unlike direction success
            # above -- a whiff naturally fails here too, since min_target_dist stays at its huge
            # init value when the ball is never touched. Denominator is num_trials (see
            # SUMMARY_SUCCESS below), not num_hit -- see --success-sigma-m's own docstring for why
            # this is a different statistic from direction success, not a threshold variant of it.
            # Checked against EVERY configured radius from the same single min_target_dist value
            # -- a trial can count as a success at more than one radius simultaneously, and no
            # extra rollout cost is paid for reporting more than one.
            if target_xy is not None:
                for sigma in success_sigmas:
                    if min_target_dist <= sigma:
                        num_success[sigma] += 1
            min_target_dist_out = min_target_dist if target_xy is not None else -1.0
            # 2026-09-06: trailing kick_aim_theta field -- "NA" (not 0.0) when kick_aim wasn't
            # enabled at all, since 0.0 is a real, distinct value a trial can genuinely sample (see
            # this module's own docstring note on RESULT's shape).
            kick_aim_theta_out = f"{kick_aim_theta:.4f}" if kick_aim_theta is not None else "NA"
            print(
                f"RESULT {args.step_label} {trial} {fall_step} {min_z:.4f} {hit_step} "
                f"{min_target_dist_out:.4f} {max_ball_speed:.4f} {kick_aim_theta_out} "
                f"{handoff_tick} {post_handoff_fall_step}",
                flush=True,
            )
            if hit_step != -1:
                hit_peak_speeds.append(max_ball_speed)
                if target_xy is not None:
                    hit_shot_errors.append(min_target_dist)
            if trajectory_trials is not None:
                trajectory_trials.append({
                    "trial": trial,
                    "kick_aim_theta": kick_aim_theta,
                    "fall_step": fall_step,
                    "hit_step": hit_step,
                    "trajectory_xy": trial_trajectory_xy,
                    "trajectory_speed": trial_trajectory_speed,
                })

        fall_rate = num_fell / args.num_trials
        hit_rate = num_hit / args.num_trials
        if args.track_posture_metrics:
            # One line per term per window, each "NA" when no surviving trial contributed samples
            # to that window (e.g. every trial toppled, or the hold ended before the grace ramp
            # elapsed) -- never a fabricated 0.0, which would read as a perfect stance.
            for name in POSTURE_METRIC_NAMES:
                for window in ("early", "late"):
                    vals = posture_trial_means[f"{name}_{window}"]
                    value = f"{float(np.mean(vals)):.6f}" if vals else "NA"
                    print(
                        f"SUMMARY_POSTURE {args.step_label} {name} {window} "
                        f"{len(vals)}/{num_posture_trials} {value}",
                        flush=True,
                    )
        print(f"SUMMARY {args.step_label} {num_fell}/{args.num_trials} {fall_rate:.4f}", flush=True)
        print(f"SUMMARY_HIT {args.step_label} {num_hit}/{args.num_trials} {hit_rate:.4f}", flush=True)
        # 2026-09-09: the Kick->Loco AUTO-handoff this scan's hold window was already running
        # through, isolated -- see this module's own docstring section on handoff_tick/
        # post_handoff_fall_step for what "reached"/"post-handoff alive" mean here.
        handoff_reached_rate = num_handoff_reached / args.num_trials
        print(
            f"SUMMARY_HANDOFF_REACHED {args.step_label} {num_handoff_reached}/{args.num_trials} "
            f"{handoff_reached_rate:.4f}",
            flush=True,
        )
        if num_handoff_reached > 0:
            post_handoff_alive_rate = num_post_handoff_alive / num_handoff_reached
            print(
                f"SUMMARY_POST_HANDOFF_ALIVE {args.step_label} {num_post_handoff_alive}/"
                f"{num_handoff_reached} {post_handoff_alive_rate:.4f}",
                flush=True,
            )
        else:
            print(f"SUMMARY_POST_HANDOFF_ALIVE {args.step_label} 0/0 NA", flush=True)
        # 2026-09-05: mean +/- std peak post-contact ball speed over HIT trials only -- same
        # denominator convention as training's own mean_speed_of_hits (shooting.py), and the same
        # "NA, not 0.0, when there's nothing to average" convention as SUMMARY_DIRECTION below.
        # Unconditional on --kick-aim-enabled (ball speed needs a hit, not a commanded target).
        # 2026-09-06: trailing <max> field (single fastest hit across this scan, for a direct
        # RoboNaldo peak-speed comparison -- see this module's own docstring) -- backward-
        # compatible append; existing 4-field parsers are unaffected.
        if len(hit_peak_speeds) > 0:
            print(
                f"SUMMARY_BALL_SPEED {args.step_label} "
                f"{float(np.mean(hit_peak_speeds)):.4f} {float(np.std(hit_peak_speeds)):.4f} "
                f"{len(hit_peak_speeds)} {float(np.max(hit_peak_speeds)):.4f}",
                flush=True,
            )
        else:
            print(f"SUMMARY_BALL_SPEED {args.step_label} NA NA 0 NA", flush=True)
        if args.kick_aim_enabled:
            if num_hit > 0:
                direction_success_rate = num_direction_hit / num_hit
                print(
                    f"SUMMARY_DIRECTION {args.step_label} {num_direction_hit}/{num_hit} "
                    f"{direction_success_rate:.4f}",
                    flush=True,
                )
            else:
                print(f"SUMMARY_DIRECTION {args.step_label} 0/0 NA", flush=True)
            for sigma in success_sigmas:
                success_rate = num_success[sigma] / args.num_trials
                tag = f"SUMMARY_SUCCESS_{sigma:g}"
                print(
                    f"{tag} {args.step_label} {num_success[sigma]}/{args.num_trials} "
                    f"{success_rate:.4f}",
                    flush=True,
                )
            # 2026-09-05: mean +/- std shot error (RoboNaldo's "nearest post-contact ball-target
            # distance", arXiv:2606.11092) over HIT trials only -- a whiff has no shot to grade,
            # same convention as SUMMARY_DIRECTION/SUMMARY_BALL_SPEED above, not invented fresh
            # here. NOT necessarily the same whiff-handling RoboNaldo itself used for their own
            # reported number -- their paper does not state whether misses are excluded or
            # included in their average; verify before treating this as a bit-identical protocol
            # match rather than "our own best-defined version of the same statistic."
            if len(hit_shot_errors) > 0:
                print(
                    f"SUMMARY_SHOT_ERROR {args.step_label} "
                    f"{float(np.mean(hit_shot_errors)):.4f} {float(np.std(hit_shot_errors)):.4f} "
                    f"{len(hit_shot_errors)}",
                    flush=True,
                )
            else:
                print(f"SUMMARY_SHOT_ERROR {args.step_label} NA NA 0", flush=True)
        if trajectory_trials is not None:
            with open(args.trajectory_output_path, "w") as f:
                json.dump({
                    "step_label": args.step_label,
                    "skill_id": args.skill_id,
                    "trials": trajectory_trials,
                }, f)
            print(f"[trajectory] wrote {len(trajectory_trials)} trial(s) -> {args.trajectory_output_path}", flush=True)
        return 0
    finally:
        pl.env.shutdown()


def main() -> int:
    return run(_parse_args())


if __name__ == "__main__":
    sys.exit(main())
