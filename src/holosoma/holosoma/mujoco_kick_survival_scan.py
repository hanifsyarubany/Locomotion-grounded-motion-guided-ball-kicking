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
<min_target_dist_or_-1>". fall_step is the first control tick (50Hz) where base z drops below
FALL_Z, or -1 if it never does, during that trial's hold window. hit_step (2026-08-21) is the
first control tick, counted from the same hold window (i.e. from the kick trigger onward, not the
settle phase before it), where MuJoCo's own contact solver reports a real ball<->foot geom contact
(env.data.contact, checked against every "{left,right}_footN_collision" geom -- see
_ball_foot_contact_now's own docstring), or -1 if the ball is never touched. min_target_dist
(2026-08-23) is this trial's closest approach (meters, over the same hold window) of the ball's
real position to the SAME target point training's own error_ball_to_target reward measures
against (ball_local_placed + kick_aim_nominal_distance_m * unit(nominal_bearing_deg +
kick_aim_theta)) -- -1 when --kick-aim-enabled isn't set (no per-trial commanded direction to
measure against). Three final lines: "SUMMARY <step> <num_fell>/<num_trials> <fall_rate>",
"SUMMARY_HIT <step> <num_hit>/<num_trials> <hit_rate>", and (only when --kick-aim-enabled)
"SUMMARY_DIRECTION <step> <num_direction_hit>/<num_hit> <direction_success_rate_or_NA>" --
direction success is gated on ACTUALLY hitting the ball (num_hit is the denominator, not
num_trials: a whiff has no departure direction to grade, and folding misses into this rate would
conflate "aimed badly" with "never kicked", which kick_ball_hit_rate already covers separately).
"NA" (not a float) when num_hit is 0 -- nothing to measure that checkpoint against yet.

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
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    import mujoco
    import numpy as np

    import robojudo.config.g1  # noqa: F401
    import robojudo.pipeline  # noqa: F401
    from robojudo.config import cfg_registry

    cfg = cfg_registry.get("g1_unified_loco_kick")()
    cfg.policy.onnx_path = args.onnx_path
    cfg.env.xml = SCENE_WITH_BALL
    pl = getattr(robojudo.pipeline, cfg.pipeline_type)(cfg=cfg)
    env = pl.env
    env.viewer.is_alive = False
    inner = pl.policy.policy
    inner._update_velocity_command = lambda cd: None

    ball_jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_freejoint")
    assert ball_jid != -1, "ball_freejoint not found in compiled model -- is SCENE_WITH_BALL correct?"
    ball_qpos_addr = int(env.model.jnt_qposadr[ball_jid])

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

    settle_steps = int(args.settle_s * args.fps)
    hold_steps = int(args.hold_s * args.fps)
    trigger_cmd = "[TRIGGER_KICK]" if args.skill_id == 0 else f"[TRIGGER_KICK:{args.skill_id}]"

    any_jitter = (
        args.ball_pos_randomization_x > 0.0 or args.ball_pos_randomization_y > 0.0
        or (args.kick_aim_enabled and args.kick_aim_theta_max_deg > 0.0)
    )
    rng = np.random.default_rng(args.seed) if (args.num_trials > 1 or any_jitter) else None

    try:
        num_fell = 0
        num_hit = 0
        num_direction_hit = 0
        for trial in range(args.num_trials):
            mujoco.mj_resetDataKeyframe(env.model, env.data, 0)
            inner.reset()
            inner._update_velocity_command = lambda cd: None

            if rng is not None:
                ball_dx = rng.uniform(-args.ball_pos_randomization_x, args.ball_pos_randomization_x)
                ball_dy = rng.uniform(-args.ball_pos_randomization_y, args.ball_pos_randomization_y)
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

            env.update()
            inner.get_observation(env.get_data(), {})
            inner.post_step_callback([trigger_cmd])

            fall_step = -1
            hit_step = -1
            # inf, not -1: this is a MINIMUM being tracked (mirrors shooting.py's own
            # min_target_dist latch), unlike fall_step/hit_step which are "first tick where X
            # happened" trackers -- -1 there means "never happened", but -1 here would look like a
            # (nonsensical, negative) distance. Converted to the -1 sentinel only at print time,
            # for a target_xy is None trial (kick_aim disabled -- no commanded direction at all).
            min_target_dist = float("inf")
            for i in range(hold_steps):
                step_zero_vel()
                z = float(env.base_pos[2])
                z_series.append(z)
                if fall_step == -1 and z < FALL_Z:
                    fall_step = i
                # Only checked from the trigger onward (not during settle above) -- ball and feet
                # are far apart pre-trigger by construction (nominal_ball_xy ~1.3m out), so a
                # settle-phase contact would only ever mean the ball spawn itself is degenerate,
                # not a kick; restricting the window keeps this reading "did the kick land",
                # matching has_kicked's own in_strike_phase gate on IsaacSim (shooting.py).
                if hit_step == -1 and _ball_foot_contact_now():
                    hit_step = i
                if target_xy is not None:
                    # Same "closest approach so far, not gated to a narrow strike window" contract
                    # as shooting.py::error_ball_to_target's own min_target_dist -- the ball keeps
                    # rolling well past the strike, so the true closest approach can happen many
                    # ticks after hit_step.
                    ball_xy_now = np.array(
                        [env.data.qpos[ball_qpos_addr], env.data.qpos[ball_qpos_addr + 1]]
                    )
                    dist = float(np.linalg.norm(ball_xy_now - target_xy))
                    if dist < min_target_dist:
                        min_target_dist = dist

            min_z = min(z_series)
            if fall_step != -1:
                num_fell += 1
            if hit_step != -1:
                num_hit += 1
                # Direction success REQUIRES a real hit -- a whiff has no departure direction to
                # grade (see this module's own docstring on why the rate below is num_direction_hit
                # / num_hit, not / num_trials).
                if target_xy is not None and min_target_dist <= args.direction_success_sigma_m:
                    num_direction_hit += 1
            min_target_dist_out = min_target_dist if target_xy is not None else -1.0
            print(
                f"RESULT {args.step_label} {trial} {fall_step} {min_z:.4f} {hit_step} "
                f"{min_target_dist_out:.4f}",
                flush=True,
            )

        fall_rate = num_fell / args.num_trials
        hit_rate = num_hit / args.num_trials
        print(f"SUMMARY {args.step_label} {num_fell}/{args.num_trials} {fall_rate:.4f}", flush=True)
        print(f"SUMMARY_HIT {args.step_label} {num_hit}/{args.num_trials} {hit_rate:.4f}", flush=True)
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
        return 0
    finally:
        pl.env.shutdown()


def main() -> int:
    return run(_parse_args())


if __name__ == "__main__":
    sys.exit(main())
