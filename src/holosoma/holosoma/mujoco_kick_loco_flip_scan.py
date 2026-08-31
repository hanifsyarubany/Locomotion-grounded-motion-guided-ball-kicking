#!/usr/bin/env python3
"""Headless scan: for one onnx checkpoint, run the same RoboJuDo `g1_unified_loco_kick` MuJoCo
pipeline as mujoco_kick_survival_scan.py (settle -> trigger kick, real ball, task-mode-gated obs
patch), but instead of letting the clip run to its own natural stand/recovery boundary, FORCES a
kick->locomotion flip at a randomized mid-clip tick -- the sim2sim analogue of training's
kick_abort_prob (see MultiSkillConfig.kick_abort_prob's own docstring for the training-time
mechanism this mirrors). Reports the rate at which the robot survives that sudden flip without
falling, over N trials -- see FastSACConfig.mujoco_kick_to_loco_random_flip_every_n_saves's own
docstring for the full motivation and how this complements mujoco_kick_survival_scan.py's
kick_fall_rate (that one measures ordinary in-kick falls; this one isolates the flip itself).

MECHANISM. UnifiedLocoKickPolicy (robojudo/policy/unified_loco_kick_policy.py) already implements
the deployed counterpart of kick_recovery_locomotion_flip_enabled: `_return_to_loco()` sets
task_mode back to locomotion, and `_assemble_obs`/the ball-observation patch both branch on
`task_mode` every tick (see mujoco_kick_rollout_worker.py's `_install_ball_observation_patch`
docstring) -- so nothing here re-implements observation switching. This script only needs to call
the ALREADY-WIRED "[RETURN_TO_LOCO]" command (`inner.post_step_callback(["[RETURN_TO_LOCO]"])`,
the same scripted-command channel `mujoco_kick_survival_scan.py` already uses to inject
"[TRIGGER_KICK]") at a tick of its own choosing instead of the policy's own natural
clip-end/plateau auto-return.

WHY NO BALL-SPAWN JITTER OR BALL-AIM RANDOMIZATION (unlike mujoco_kick_survival_scan.py): this
scan measures the flip's effect, not the kick's. Ball spawn stays at that skill's own nominal
get_skill_ball_xy, un-jittered -- the ball is still present and positioned (a physically absent
ball would make the pre-flip kick portion itself out-of-distribution, since every project
checkpoint trained with a real ball), just not varied trial-to-trial. `--kick-aim-enabled` is
still supported (REQUIRED for a kick_aim_enabled checkpoint -- see this file's own note below) but
always centered at theta=0.0, never jittered.

WHY THE FLIP TICK WINDOW MIRRORS kick_abort_delay_min/max_steps (10/60 ticks, NOT exposed as CLI
flags with different names): direct comparability with the training-time mechanism this scan
evaluates. Both are "ticks since kick trigger" at the same 50Hz control rate (dt=0.02s, confirmed
against MultiSkillConfig.post_flip_termination_grace_steps's own docstring), so a checkpoint
trained with kick_abort_prob > 0 at the default window is being tested here on exactly the
distribution of flip timing it was trained against.

WHY "ALIVE RATE" EXCLUDES TRIALS THAT FELL BEFORE THE FLIP EVER FIRED: a trial that has already
collapsed on its own (an ordinary in-kick fall, already covered by kick_fall_rate) never actually
tested flip robustness -- counting it either way would conflate two different failure modes into
one number. Same "exclude the degenerate denominator case" pattern kick_direction_success_rate
already established (graded over num_hit, not num_trials, since a whiff has no direction to grade)
-- SUMMARY_LOCOFLIP's denominator is trials that were still upright AT the flip tick, and
SUMMARY_PREFLIPFAIL reports the excluded fraction separately so a checkpoint that cannot even
survive an ordinary kick doesn't silently produce a misleadingly high (or NA) flip-alive rate.

Per-trial output: "RESULT <step> <trial> <flip_tick> <pre_flip_fall_step_or_-1>
<post_flip_fall_step_or_-1> <min_z>". pre_flip_fall_step/post_flip_fall_step are ticks relative to
the start of their own window (kick trigger for the former, the flip itself for the latter), or -1
if no fall was observed in that window. Two summary lines: "SUMMARY_PREFLIPFAIL <step>
<num_pre_flip_fail>/<num_trials> <rate>" (always defined) and "SUMMARY_LOCOFLIP <step>
<num_alive>/<num_reached_flip> <rate_or_NA>" ("NA" when num_reached_flip is 0 -- every trial fell
before ever reaching its scheduled flip, nothing to measure the flip against).

Usage:
    /workspaces/isaaclab_arena/submodules/workspaces/conda_env/robojudo/bin/python \\
        mujoco_kick_loco_flip_scan.py --onnx-path /path/to/model_0005000.onnx --step-label 5000 \\
        --num-trials 32 --seed 0 --skill-id 0

Usage (kick_aim_enabled checkpoint -- REQUIRED for any checkpoint actually trained with
kick_aim_enabled=True, same rationale as mujoco_kick_survival_scan.py's own flag: omitting it feeds
the raw world-frame target_pos_b transform into obs[157:159] instead of the trained [0, 0]
azimuth-aim format, a large out-of-distribution magnitude):
    ... mujoco_kick_loco_flip_scan.py --onnx-path ... --step-label 500000 --num-trials 32 \\
        --seed 0 --kick-aim-enabled
"""
from __future__ import annotations

import argparse
import os
import sys

ROBOJUDO_REPO = "/workspaces/isaaclab_arena/submodules/workspaces/humanoid_deployment/RoboJuDo"
FALL_Z = 0.4  # same physical-fall threshold as mujoco_kick_survival_scan.py, for direct comparability

sys.path.insert(0, ROBOJUDO_REPO)
# This file's OWN directory -- same cross-fork-contamination guard as mujoco_kick_survival_scan.py
# (see that module's own bug-fix note for why this matters).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mujoco_kick_rollout_worker import SCENE_WITH_BALL, _install_ball_observation_patch  # noqa: E402

# Only used when --kick-aim-enabled and theta is always 0.0 (centered, never jittered) in this
# scan -- 0.0 / anything is 0.0, so this constant's actual value never affects the injected
# observation. Kept as an internal constant (not a CLI flag) rather than exposing a knob with no
# effect.
_KICK_AIM_THETA_REF_DEG_UNUSED = 45.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx-path", required=True)
    parser.add_argument("--settle-s", type=float, default=1.5)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--step-label", required=True)
    parser.add_argument(
        "--skill-id", type=int, default=0,
        help="Which of the ONNX's embedded motion skills to kick -- same convention as "
        "mujoco_kick_survival_scan.py's own --skill-id.",
    )
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0, help="Flip-tick RNG seed; trial i draws from seed+i.")
    parser.add_argument(
        "--flip-delay-min-steps", type=int, default=10,
        help="Inclusive lower bound (control ticks since kick trigger, 50Hz) for the randomized "
        "flip tick. Mirrors MultiSkillConfig.kick_abort_delay_min_steps's own default (10) -- see "
        "this module's own docstring for why the two are kept aligned.",
    )
    parser.add_argument(
        "--flip-delay-max-steps", type=int, default=60,
        help="Inclusive upper bound for the randomized flip tick. Mirrors "
        "MultiSkillConfig.kick_abort_delay_max_steps's own default (60).",
    )
    parser.add_argument(
        "--post-flip-hold-s", type=float, default=5.0,
        help="How long to keep stepping after the flip before judging alive/fallen. Shorter than "
        "mujoco_kick_survival_scan.py's own --hold-s (8.0) since this window starts AT the flip, "
        "not at kick trigger -- long enough to see whether the robot stabilizes or topples "
        "(training's own post_flip_termination_grace_steps=50 ticks=1.0s plus margin).",
    )
    parser.add_argument(
        "--kick-aim-enabled", action="store_true",
        help="Pass this for a checkpoint trained with kick_aim_enabled=True (every skill in this "
        "project, as of 2026-08-22) -- see this file's own module docstring for why omitting it "
        "on such a checkpoint is out-of-distribution.",
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

    nominal_ball_xy = inner.get_skill_ball_xy(args.skill_id)
    if nominal_ball_xy is None:
        from mujoco_kick_rollout_worker import BALL_WORLD_POS

        nominal_ball_xy = (BALL_WORLD_POS[0], BALL_WORLD_POS[1])

    if args.kick_aim_enabled and inner.get_skill_target_xy(args.skill_id) is None:
        raise ValueError(
            "--kick-aim-enabled requires this checkpoint's ONNX to carry skill_target_xy metadata "
            "-- get_skill_target_xy returned None for this skill_id."
        )

    settle_steps = int(args.settle_s * args.fps)
    post_flip_hold_steps = int(args.post_flip_hold_s * args.fps)
    trigger_cmd = "[TRIGGER_KICK]" if args.skill_id == 0 else f"[TRIGGER_KICK:{args.skill_id}]"
    rng = np.random.default_rng(args.seed)

    try:
        num_pre_flip_fail = 0
        num_reached_flip = 0
        num_post_flip_fall = 0
        for trial in range(args.num_trials):
            mujoco.mj_resetDataKeyframe(env.model, env.data, 0)
            inner.reset()
            inner._update_velocity_command = lambda cd: None

            env.data.qpos[ball_qpos_addr] = nominal_ball_xy[0]
            env.data.qpos[ball_qpos_addr + 1] = nominal_ball_xy[1]

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

            for _ in range(settle_steps):
                step_zero_vel()

            env.update()
            inner.get_observation(env.get_data(), {})
            inner.post_step_callback([trigger_cmd])

            flip_tick = int(rng.integers(args.flip_delay_min_steps, args.flip_delay_max_steps + 1))

            z_series = []
            pre_flip_fall_step = -1
            for i in range(flip_tick):
                step_zero_vel()
                z = float(env.base_pos[2])
                z_series.append(z)
                if pre_flip_fall_step == -1 and z < FALL_Z:
                    pre_flip_fall_step = i

            # Standalone injection, same pattern as the trigger call above -- not wrapped in
            # step_zero_vel(), fires the transition once between two control ticks. Fires
            # unconditionally regardless of pre_flip_fall_step: a robot already down when its
            # scheduled tick arrives just gets a harmless idempotent _return_to_loco() call (see
            # this module's own docstring on why the alive-rate denominator excludes it anyway).
            inner.post_step_callback(["[RETURN_TO_LOCO]"])

            post_flip_fall_step = -1
            for i in range(post_flip_hold_steps):
                step_zero_vel()
                z = float(env.base_pos[2])
                z_series.append(z)
                if post_flip_fall_step == -1 and z < FALL_Z:
                    post_flip_fall_step = i

            min_z = min(z_series) if z_series else float("nan")
            if pre_flip_fall_step != -1:
                num_pre_flip_fail += 1
            else:
                num_reached_flip += 1
                if post_flip_fall_step != -1:
                    num_post_flip_fall += 1

            print(
                f"RESULT {args.step_label} {trial} {flip_tick} {pre_flip_fall_step} "
                f"{post_flip_fall_step} {min_z:.4f}",
                flush=True,
            )

        pre_flip_fail_rate = num_pre_flip_fail / args.num_trials
        print(
            f"SUMMARY_PREFLIPFAIL {args.step_label} {num_pre_flip_fail}/{args.num_trials} "
            f"{pre_flip_fail_rate:.4f}",
            flush=True,
        )
        if num_reached_flip > 0:
            num_alive = num_reached_flip - num_post_flip_fall
            alive_rate = num_alive / num_reached_flip
            print(
                f"SUMMARY_LOCOFLIP {args.step_label} {num_alive}/{num_reached_flip} {alive_rate:.4f}",
                flush=True,
            )
        else:
            print(f"SUMMARY_LOCOFLIP {args.step_label} 0/0 NA", flush=True)
        return 0
    finally:
        pl.env.shutdown()


def main() -> int:
    return run(_parse_args())


if __name__ == "__main__":
    sys.exit(main())
