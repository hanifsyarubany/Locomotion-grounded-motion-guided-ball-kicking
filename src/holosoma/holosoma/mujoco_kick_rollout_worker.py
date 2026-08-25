#!/usr/bin/env python3
"""Standalone MuJoCo sim2sim kick-rollout worker, shared by two wandb rollouts: "Training rollout -
MuJoCo Kick" (settle-then-trigger, the original) and "Training rollout - MuJoCo Kick (Locomotion
Handoff)" (walk-then-trigger, added 2026-08-13 -- see `--walk-s` below). NOT part of the
`holosoma` package's own runtime -- this script is only ever invoked as a subprocess via the
`robojudo` conda env's interpreter (see `record_mujoco_kick_rollout.py`, which is what the
training process actually imports and which has zero dependency on this file's imports).

Builds the same single-env RoboJuDo `g1_unified_loco_kick` MuJoCo pipeline used throughout
holosoma development for manual sim2sim testing, scripts a stand/walk -> trigger-kick -> hold
sequence, captures offscreen frames via `mujoco.Renderer`, and pipes them straight to an `ffmpeg`
subprocess (robojudo's env has no cv2/imageio-ffmpeg, but does have ffmpeg on PATH) to produce an
mp4.

2026-08-13, `--walk-s` (default 0.0 = exact original behavior, byte-identical): when > 0, REPLACES
the zero-velocity settle phase with a forward-walk phase at `--forward-speed` m/s, and the kick is
triggered on the SAME tick the walk command cuts to zero -- i.e. the robot is still carrying real
forward velocity/momentum the instant the clip starts, not triggered from a settled stand. This is
a SIMPLE actor-robustness test (does the trained policy handle a kick triggered mid-stride), not a
port of holosoma's mid-episode entry-point search (UnifiedManager._maybe_enter_kick_from_locomotion
-- that mechanism picks WHICH clip frame to enter at based on live gait matching, and only exists
in the IsaacSim training env; this rollout still always triggers at the clip's own frame 0, exactly
like the settle-then-trigger rollout, just from a walking start instead of a standing one). User
directive: RoboJuDo's own kick trigger (`_trigger_kick`) has no such search either, so a faithful
port would require walking that mechanism into a second, independent codebase for a marginal gain
over this simpler, already-informative test.

A real, physically-simulated ball IS spawned (added 2026-07-17; the "no real ball" scope-reduction
noted below is now historical) -- see `_ball_observation_patch()`. Its absence was root-caused as a
real, independent bug: RoboJuDo's `UnifiedLocoKickPolicy.get_observation()` hardcodes
`kick_ball_pos_b`/`kick_target_pos_b` to zero, which is out-of-distribution for any Stage-C
checkpoint (trained on `shooting_reward_scale>0`, i.e. a real, observed ball) and measurably hurts
survival on its own (68.8% -> 0% in a controlled same-checkpoint test -- see memory
stagec-kick-definitive-root-cause-physx-mujoco). Without this fix, "Training rollout - MuJoCo Kick"
was silently confounding two separate problems into one video: the missing-ball observation bug
AND a genuine PhysX<->MuJoCo contact-resolution gap (see memory
stagec-kick-open-loop-physics-proof), with no way to tell from the video which one caused a given
fall. Feeding a real ball here isolates the physics-gap signal so this periodic rollout is a
meaningful read on whether contact-tuning fixes (see memory stagec-kick-physx-mujoco-contact-fix)
are actually working, without needing a separate manual verification pass.

Historical note (pre-2026-07-17): "No real ball is spawned (RoboJuDo's g1_unified_loco_kick
pipeline doesn't model one) -- this is a deliberate scope reduction from the 'real ball, full
DDS-bridged deployment stack' design in favor of a single self-contained process; see the approved
plan for the tradeoff." That tradeoff is unchanged (still no DDS bridge, still single-process) --
only the "no ball at all" part has been addressed, since it was found to materially bias the
signal this rollout exists to produce.

Usage (manual test):
    /workspaces/isaaclab_arena/submodules/workspaces/conda_env/robojudo/bin/python \
        mujoco_kick_rollout_worker.py --onnx-path /path/to/model_0005000.onnx \
        --output-video-path /tmp/test_kick.mp4
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import traceback

FFMPEG_FALLBACK = "/workspaces/isaaclab_arena/submodules/workspaces/conda_env/hssim_holosoma/bin/ffmpeg"
ROBOJUDO_REPO = "/workspaces/isaaclab_arena/submodules/workspaces/humanoid_deployment/RoboJuDo"
SCENE_WITH_BALL = ROBOJUDO_REPO + "/assets/robots/g1/holosoma_model/scene_g1_29dof_with_ball.xml"

# FALLBACK ball/target placement, relative to the robot's known, fixed keyframe spawn (world
# origin, identity quat, facing +x) -- only used for a checkpoint whose ONNX carries no
# skill_ball_xy/skill_target_xy metadata (older export, or trained without a ball). Checkpoints
# WITH that metadata get the SELECTED SKILL's own configured x/y/target_x/target_y instead (see
# run()'s ball-spawn override and _install_ball_observation_patch's target_world) -- these
# constants stop applying at that point. Chosen to approximate training's own documented
# ball_pos_b magnitude at kick-clip start (~1.38m; see
# managers/observation/terms/unified.py::ball_pos_b's docstring).
BALL_WORLD_POS = (1.3, 0.0, 0.11)
TARGET_WORLD_POS = (1.3 + 2.8, 0.0 - 0.46)  # matches training's ball->target relative offset


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-path", required=True)
    parser.add_argument("--output-video-path", required=True)
    parser.add_argument("--settle-s", type=float, default=1.5, help="Standing settle time before the kick trigger.")
    parser.add_argument(
        "--walk-s", type=float, default=0.0,
        help="Forward-walk duration before the kick trigger, REPLACING --settle-s when > 0 -- the "
        "trigger fires the instant the walk command cuts to zero, so the robot is still carrying "
        "real forward velocity when the clip starts (see this module's own docstring for what "
        "this does and does NOT test). 0.0 (default) preserves the original settle-then-trigger "
        "behavior exactly.",
    )
    parser.add_argument(
        "--forward-speed", type=float, default=0.5,
        help="Forward (+x, body-frame) velocity command in m/s during the walk phase. Only used "
        "when --walk-s > 0. Deliberately a moderate/stable speed, not "
        "mujoco_locomotion_rollout_worker.py's 0.8 m/s (that value targets a documented STOPPING "
        "failure mode -- a different stress test from this rollout's own purpose).",
    )
    parser.add_argument("--hold-s", type=float, default=8.0, help="Recording duration after the kick trigger.")
    parser.add_argument("--fps", type=int, default=50, help="Matches the 50Hz RL control rate.")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument(
        "--no-ball", action="store_true",
        help="Revert to the original no-ball, zero-observation rollout (pre-2026-07-17 behavior). "
        "Only meaningful for checkpoints trained BEFORE 2026-07-21 under the old scheme, where "
        "Stage B (shooting_reward_scale=0) held ball/target observations at a hard zero -- for "
        "those, feeding a real ball here is out-of-distribution. Checkpoints trained after "
        "2026-07-21 observe a real, live (if unrandomized) ball in every stage including Stage B "
        "(managers/observation/terms/unified.py::ball_pos_b), so this flag should NOT be used for "
        "them at any stage -- doing so is itself now the out-of-distribution condition.",
    )
    parser.add_argument(
        "--skill-id", type=int, default=0,
        help="Which of the ONNX's embedded motion skills to kick (see "
        "robojudo.policy.unified_loco_kick_policy's module docstring on skill_motion_start_idx/"
        "skill_motion_end_idx metadata). Default 0 -- the only skill that exists for a checkpoint "
        "trained without holosoma's N-skill mechanism, and the first-declared skill for one "
        "trained with it. Out-of-range values fall back to skill 0 with a logged warning rather "
        "than crashing the rollout.",
    )
    parser.add_argument(
        "--kick-aim-enabled", action="store_true",
        help="2026-08-23: this checkpoint was trained with SkillConfig.kick_aim_enabled=True -- "
        "feed obs[157:159] as the bounded [kick_aim_theta/kick_aim_theta_ref_deg, 0.0] command it "
        "actually trained on, instead of the pre-azimuth-refactor raw world-frame target_pos_b "
        "transform (a ~15-18x out-of-distribution magnitude for such a checkpoint -- see "
        "_install_ball_observation_patch's own docstring). Default off, so a checkpoint trained "
        "WITHOUT kick_aim_enabled (or before this refactor) is completely unaffected.",
    )
    parser.add_argument(
        "--kick-aim-theta-deg", type=float, default=0.0,
        help="The fixed kick_aim_theta value (degrees) to feed when --kick-aim-enabled is set -- "
        "0.0 (default) aims straight along the skill's own calibrated nominal bearing, the natural "
        "choice for a repeatable demo/diagnostic video (no reason to prefer a random angle here). "
        "Ignored entirely when --kick-aim-enabled is not set.",
    )
    parser.add_argument(
        "--kick-aim-theta-ref-deg", type=float, default=45.0,
        help="Normalization reference (degrees) matching the checkpoint's own MultiSkillConfig/"
        "BallConfig.kick_aim_theta_ref_deg -- 45.0 is that field's own default and, per its "
        "docstring, is meant to stay fixed across curriculum changes, so this default is correct "
        "for every run in this project unless a config explicitly overrode it (none currently "
        "do). Ignored when --kick-aim-enabled is not set.",
    )
    parser.add_argument(
        "--output-trajectory-path", default=None,
        help="2026-08-19, user-requested: CSV path to log the ball's live world (x, y, z) every "
        "control tick of the WHOLE rollout (settle/walk + hold), for accurately setting "
        "SkillConfig.target_x/target_y (configs/skill2.yaml) from where a REAL kick (this "
        "rollout's own deployed policy, real MuJoCo contact physics) actually sends the ball -- "
        "unlike replay.py's kinematic clip playback, which never generates real foot-ball "
        "contact (measured 2026-08-19: ball position identical bit-for-bit across an entire "
        "replay, including the strike window). Default None = derived from "
        "--output-video-path by replacing its extension with .csv, so video and trajectory land "
        "together without an extra flag in the common case. Ignored (nothing written) when "
        "--no-ball is set -- there is no ball to log.",
    )
    return parser.parse_args()


def _install_ball_observation_patch(
    env,
    inner,
    mujoco_module,
    np,
    ball_qpos_addr: int,
    skill_id: int,
    ball_x_shift: float = 0.0,
    target_xy_override: "tuple[float, float] | None" = None,
    kick_aim_theta_deg: "float | None" = None,
    kick_aim_theta_ref_deg: float = 45.0,
    kick_aim_theta_deg_getter: "Callable[[], float] | None" = None,
) -> None:
    """Feed real, task-mode-gated kick_ball_pos_b/kick_target_pos_b instead of the policy's
    hardcoded zero, computed every tick from the live ball body + robot root using training's
    exact formula (managers/observation/terms/unified.py::ball_pos_b/target_pos_b):
        rel = ball_pos_w - robot_root_pos_w                          (full 3D, real z)
        ball_pos_b = quat_rotate_inverse(yaw_quat(base_quat), rel)     (heading frame)
    target_pos_b is the same transform, xy-only, against the target world position.

    Target position: `inner.get_skill_target_xy(skill_id)` -- that skill's own configured
    target_x/target_y, read from the ONNX's skill_target_xy metadata (see
    get_skill_ball_target_metadata in holosoma/utils/inference_helpers.py) -- if the checkpoint
    has no such metadata (older export, or trained without a ball), falls back to the module-level
    TARGET_WORLD_POS constant, exactly this function's pre-multi-skill behavior.

    `ball_x_shift`: same forward-x shift applied to the ball's own spawn position in `run()` (see
    that function's own comment -- 0.0 for the original settle-then-trigger rollout, the walked
    distance for --walk-s > 0). Applied here too so the ball->target vector the policy conditions
    on stays exactly what training used -- both ball and target are configured in the SAME
    robot-spawn-anchored world frame (see get_skill_ball_target_metadata's docstring), so shifting
    only the ball would shrink the ball->target distance by the walked distance.

    Gated on inner.task_mode == "kick" to match training's own task_mode="kick"-tagged masking
    (managers/observation/manager.py's task_mode_mask zeroes ball_pos_b/target_pos_b -- and every
    other kick_*-tagged term -- for envs currently running the locomotion task, independent of
    which stage/config is training; this is a per-episode task selector, not the Stage B/C
    switch). Getting this wrong (patching unconditionally) feeds a nonzero
    ball reading during the pre-kick locomotion settle phase, which is itself out-of-distribution
    and was caught the hard way during the investigation this fix comes from -- see memory
    stagec-kick-definitive-root-cause-physx-mujoco for the full story of that bug.

    Verified bit-for-bit against training's formula and RoboJuDo's own dof/frame conventions in
    RoboJuDo's verify_ball_obs_root_cause.py before being folded in here.

    `target_xy_override` (2026-08-19, added for mujoco_kick_survival_scan.py's in-distribution
    trial-to-trial target jitter): when given, used verbatim as the observed target INSTEAD of
    `inner.get_skill_target_xy(skill_id)` -- still shifted by `ball_x_shift` exactly like the
    nominal path, so it stays consistent with a --walk-s > 0 rollout too. None (default) preserves
    the original nominal-target behavior exactly -- every existing caller (mujoco_kick_rollout_worker
    itself) is unaffected.

    `kick_aim_theta_deg`/`kick_aim_theta_ref_deg` (2026-08-22, azimuth-aim refactor): when
    `kick_aim_theta_deg` is not None, obs[157:159] is instead set to
    `[kick_aim_theta_deg / kick_aim_theta_ref_deg, 0.0]` -- the SAME constant, world-frame-
    independent value UnifiedPolicy.get_current_obs_buffer_dict feeds for a kick_aim_enabled
    checkpoint (see that method's own docstring) -- and `target_xy_override`/
    `inner.get_skill_target_xy` are IGNORED entirely for this term (there is no world-frame target
    point to compute a rel_target from in this mode). None (default) preserves the original
    target_pos_b world-frame-transform behavior exactly -- only pass this for a checkpoint actually
    trained with kick_aim_enabled=True; feeding it to any other checkpoint is out-of-distribution
    the same way any mismatched observation term is.

    `kick_aim_theta_deg_getter` (2026-08-23, mujoco_kick_interactive.py): when given, called FRESH
    EVERY TICK instead of using the fixed `kick_aim_theta_deg` value captured once at install time
    -- lets an interactive caller change the commanded angle between kicks without re-installing
    this patch (re-installing would capture the ALREADY-patched get_observation as this call's own
    "orig", nesting closures deeper on every change instead of replacing cleanly). Takes priority
    over `kick_aim_theta_deg` when both are given (only `kick_aim_theta_ref_deg` is still read from
    the fixed argument in that case). None (default) preserves the exact fixed-value behavior
    every other caller (mujoco_kick_rollout_worker.py's own run(), mujoco_kick_survival_scan.py)
    already uses -- both remain completely unaffected by this parameter's existence."""
    from robojudo.policy.unified_loco_kick_policy import _TASK_KICK
    from robojudo.utils.util_func import calc_heading_quat_np, quat_rotate_inverse_np

    aim_mode = kick_aim_theta_deg_getter is not None or kick_aim_theta_deg is not None

    if not aim_mode:
        if target_xy_override is not None:
            target_world = np.asarray(target_xy_override, dtype=np.float64)
        else:
            target_xy = inner.get_skill_target_xy(skill_id)
            target_world = np.asarray(target_xy if target_xy is not None else TARGET_WORLD_POS, dtype=np.float64)
        target_world = target_world + np.array([ball_x_shift, 0.0])
    elif kick_aim_theta_deg_getter is None:
        aim_command = np.array([kick_aim_theta_deg / kick_aim_theta_ref_deg, 0.0], dtype=np.float32)

    orig_get_observation = inner.get_observation

    def patched_get_observation(env_data, ctrl_data):
        obs, extras = orig_get_observation(env_data, ctrl_data)
        if inner.task_mode == _TASK_KICK:
            ball_pos_w = env.data.qpos[ball_qpos_addr : ball_qpos_addr + 3].copy()
            robot_pos_w = env.base_pos.copy()
            base_quat_xyzw = env.base_quat.copy()

            heading_quat = calc_heading_quat_np(base_quat_xyzw)
            ball_pos_b = quat_rotate_inverse_np(heading_quat, ball_pos_w - robot_pos_w)
            obs[29:32] = ball_pos_b.astype(np.float32)

            if not aim_mode:
                rel_target = np.array([target_world[0] - robot_pos_w[0], target_world[1] - robot_pos_w[1], 0.0])
                target_pos_b = quat_rotate_inverse_np(heading_quat, rel_target)[:2]
                obs[157:159] = target_pos_b.astype(np.float32)
            elif kick_aim_theta_deg_getter is not None:
                live_theta_deg = kick_aim_theta_deg_getter()
                obs[157:159] = np.array([live_theta_deg / kick_aim_theta_ref_deg, 0.0], dtype=np.float32)
            else:
                obs[157:159] = aim_command
        return obs, extras

    inner.get_observation = patched_get_observation


def run(args: argparse.Namespace) -> int:
    sys.path.insert(0, ROBOJUDO_REPO)
    # This script's own directory, so the sibling mujoco_robot_extent module imports cleanly when
    # run as a bare path under the robojudo interpreter (holosoma itself is never importable here).
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import csv

    import mujoco
    import numpy as np

    from mujoco_robot_extent import RobotExtent

    import robojudo.config.g1  # noqa: F401 -- registers config
    import robojudo.pipeline  # noqa: F401 -- registers pipeline
    from robojudo.config import cfg_registry

    ffmpeg = shutil.which("ffmpeg") or FFMPEG_FALLBACK

    cfg = cfg_registry.get("g1_unified_loco_kick")()
    cfg.policy.onnx_path = args.onnx_path
    if not args.no_ball:
        cfg.env.xml = SCENE_WITH_BALL
    pl = getattr(robojudo.pipeline, cfg.pipeline_type)(cfg=cfg)
    env = pl.env
    env.viewer.is_alive = False
    inner = pl.policy.policy
    inner._update_velocity_command = lambda cd: None

    # 2026-08-20, same latent bug found via mujoco_kick_interactive.py (see that script's own
    # comment at this same point for the full trace): CONTROLLER="both" (the default in
    # robojudo/config/g1/g1_unified_loco_kick_cfg.py) wires a native pynput-based KeyboardCtrl
    # bound to "k"->[TRIGGER_KICK] etc, polled on every pl.step() regardless of this script's own
    # explicit kick-trigger calls below. This script never waits on a live terminal the way the
    # interactive one does, so the exposure is lower, but it's still a global OS-level listener --
    # any stray 'k'/'l'/'j' keypress on the machine while a batch rollout is running (e.g. typed in
    # an unrelated terminal) would silently fire a real, untracked command mid-rollout. Neutralize
    # it the same way: clear its triggers so only this script's own deliberate calls can fire one.
    kb_ctrl = pl.ctrl_manager.controllers.get("KeyboardCtrl")
    if kb_ctrl is not None:
        kb_ctrl["inst"].triggers = {}

    try:
        renderer = mujoco.Renderer(env.model, height=args.height, width=args.width)
        camera = mujoco.MjvCamera()
        camera.distance = 4.0
        camera.azimuth = 90.0
        camera.elevation = -5.0

        mujoco.mj_resetDataKeyframe(env.model, env.data, 0)
        # How far the walk phase moves the robot (world +x, since it spawns at the origin facing
        # +x) before the trigger -- 0.0 for the original settle-then-trigger path. The ball (and,
        # in _install_ball_observation_patch, the target) get shifted forward by exactly this much
        # so the robot-relative geometry at the TRIGGER tick matches the original rollout's own
        # (the kick policy's trained distribution), regardless of --walk-s/--forward-speed: without
        # this, a fixed-world ball sits closer and closer as the robot walks toward it and gets
        # physically run into mid-walk instead of being kicked (reported 2026-08-13).
        ball_x_shift = args.forward_speed * args.walk_s if args.walk_s > 0 else 0.0
        if not args.no_ball:
            ball_jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_freejoint")
            assert ball_jid != -1, "ball_freejoint not found in compiled model -- is SCENE_WITH_BALL correct?"
            ball_qpos_addr = int(env.model.jnt_qposadr[ball_jid])
            # Move the ball to the SELECTED SKILL's own configured spawn (x, y) plus ball_x_shift,
            # overriding the scene XML's keyframe default -- inner.get_skill_ball_xy falls back to
            # None for checkpoints with no skill_ball_xy metadata, in which case BALL_WORLD_POS is
            # the base instead (still explicitly written, not left at the keyframe default, so the
            # shift applies uniformly regardless of whether per-skill metadata is present) --
            # UNLESS no shift is needed at all (ball_x_shift == 0.0), in which case the untouched
            # keyframe default is used exactly as before this shift mechanism existed. z is left as
            # the keyframe's own value (rest height on the ball's own radius, not a per-skill-
            # configured quantity).
            ball_xy = inner.get_skill_ball_xy(args.skill_id)
            if ball_xy is not None:
                env.data.qpos[ball_qpos_addr] = ball_xy[0] + ball_x_shift
                env.data.qpos[ball_qpos_addr + 1] = ball_xy[1]
            elif ball_x_shift != 0.0:
                env.data.qpos[ball_qpos_addr] = BALL_WORLD_POS[0] + ball_x_shift
                env.data.qpos[ball_qpos_addr + 1] = BALL_WORLD_POS[1]
            _install_ball_observation_patch(
                env,
                inner,
                mujoco,
                np,
                ball_qpos_addr,
                args.skill_id,
                ball_x_shift=ball_x_shift,
                kick_aim_theta_deg=(args.kick_aim_theta_deg if args.kick_aim_enabled else None),
                kick_aim_theta_ref_deg=args.kick_aim_theta_ref_deg,
            )
        mujoco.mj_forward(env.model, env.data)
        env.update()

        trajectory_rows: list[tuple[int, float, str, str, int, float, float, float, float, float, float, float, float, float]] = []
        trajectory_path = None
        if not args.no_ball:
            trajectory_path = args.output_trajectory_path or (os.path.splitext(args.output_video_path)[0] + ".csv")

        # 2026-08-21, user-requested: also record the robot's BASE (root) pose per tick, so a
        # rollout's stability can be read off the same CSV as its ball trajectory -- base_z is the
        # continuous quantity behind the thresholded topple metric, and it moves before a fall is
        # declared (UnifiedManager._KICK_FALL_HEIGHT_THRESHOLD = 0.40 m).
        #
        # Root resolved from the model's FIRST free joint rather than assuming qpos[0:3]: the ball
        # adds a SECOND freejoint, and scene_g1_29dof_with_ball.xml's own comment guarantees the
        # ordering ("Added AFTER the robot body (via <include>, which comes first) so the robot's
        # freejoint stays the first free joint in the compiled model"). Asserting it here turns
        # that documented invariant into a checked one rather than a silent assumption -- if a
        # future scene reorders them this fails loudly instead of logging the BALL's height as the
        # robot's. Matches what training logs as Env/kick_base_height, which reads the root body's
        # own world z (UnifiedManager: simulator.robot_root_states[:, 2]).
        _free_jids = [j for j in range(env.model.njnt) if env.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        assert _free_jids, "no free joint in the compiled model -- cannot locate the robot root"
        robot_root_qadr = int(env.model.jnt_qposadr[_free_jids[0]])
        if not args.no_ball:
            assert robot_root_qadr != ball_qpos_addr, (
                f"the first free joint IS the ball (qposadr {ball_qpos_addr}) -- the robot's freejoint "
                "must come first; see scene_g1_29dof_with_ball.xml's ordering comment"
            )

        # Full robot height (lowest surface -> highest surface) alongside the base/root height.
        # Base height says where the PELVIS is; total height says how extended/crouched the whole
        # robot is -- the pelvis can hold station while the legs bend. See mujoco_robot_extent.py
        # for why geom_rbound is unusable here (it puts the robot's bottom below the floor).
        robot_extent = RobotExtent(mujoco, env.model)

        def record_trajectory(phase: str) -> None:
            if trajectory_path is None:
                return
            pos = env.data.qpos[ball_qpos_addr : ball_qpos_addr + 3]
            base = env.data.qpos[robot_root_qadr : robot_root_qadr + 3]
            top_z, bottom_z, height = robot_extent.measure(env.data)
            trajectory_rows.append((
                len(trajectory_rows),
                len(trajectory_rows) / args.fps,
                phase,
                inner.task_mode,
                int(inner.curr_motion_timestep),
                float(pos[0]),
                float(pos[1]),
                float(pos[2]),
                float(base[0]),
                float(base[1]),
                float(base[2]),
                float(top_z),
                float(bottom_z),
                float(height),
            ))

        stderr_path = args.output_video_path + ".ffmpeg_stderr.log"
        with open(stderr_path, "wb") as stderr_f:
            proc = subprocess.Popen(
                [
                    ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", f"{args.width}x{args.height}", "-r", str(args.fps),
                    "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
                    args.output_video_path,
                ],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=stderr_f,
            )

            def capture_frame() -> None:
                camera.lookat[:] = [env.data.qpos[0], env.data.qpos[1], 0.6]
                renderer.update_scene(env.data, camera=camera)
                proc.stdin.write(renderer.render().tobytes())

            def step_zero_vel() -> None:
                inner.lin_vel_command = np.zeros(2)
                inner.ang_vel_command = 0.0
                pl.step()

            def step_vel(vx: float) -> None:
                inner.lin_vel_command = np.array([vx, 0.0])
                inner.ang_vel_command = 0.0
                pl.step()

            try:
                if args.walk_s > 0:
                    # Walk right up to the trigger tick -- the LAST step still carries the forward
                    # command, so the robot has real velocity/momentum when the clip starts (see
                    # this module's own docstring for what this tests). No zero-velocity settle
                    # phase at all in this branch, unlike the original --settle-s path below.
                    for _ in range(int(args.walk_s * args.fps)):
                        step_vel(args.forward_speed)
                        capture_frame()
                        record_trajectory("walk")
                else:
                    for _ in range(int(args.settle_s * args.fps)):
                        step_zero_vel()
                        capture_frame()
                        record_trajectory("settle")

                env.update()
                inner.get_observation(env.get_data(), {})
                trigger_cmd = "[TRIGGER_KICK]" if args.skill_id == 0 else f"[TRIGGER_KICK:{args.skill_id}]"
                inner.post_step_callback([trigger_cmd])

                for _ in range(int(args.hold_s * args.fps)):
                    step_zero_vel()
                    capture_frame()
                    record_trajectory("hold")
            finally:
                proc.stdin.close()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)

        if trajectory_path is not None and trajectory_rows:
            with open(trajectory_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "tick", "t_s", "phase", "task_mode", "curr_motion_timestep",
                    "ball_x", "ball_y", "ball_z", "base_x", "base_y", "base_z",
                    "robot_top_z", "robot_bottom_z", "robot_height",
                ])
                w.writerows(trajectory_rows)
            last = trajectory_rows[-1]
            base_z = [r[10] for r in trajectory_rows]
            min_base_z = min(base_z)
            min_tick = base_z.index(min_base_z)
            # 0.40 m is UnifiedManager._KICK_FALL_HEIGHT_THRESHOLD -- the same cut training's own
            # kick_topple_frac uses, so this line is directly comparable to that metric.
            fell = min_base_z < 0.40
            print(
                f"[trajectory] wrote {len(trajectory_rows)} ticks -> {trajectory_path} "
                f"(final ball pos: x={last[5]:.3f} y={last[6]:.3f} z={last[7]:.3f})",
                flush=True,
            )
            heights = [r[13] for r in trajectory_rows]
            print(
                f"[base-height] final {last[10]:.3f} m | min {min_base_z:.3f} m at tick {min_tick} "
                f"(t={min_tick / args.fps:.2f}s, clip frame {trajectory_rows[min_tick][4]}) | "
                f"{'FELL' if fell else 'stayed up'} vs the 0.40 m fall threshold",
                flush=True,
            )
            print(
                f"[robot-height] full height (feet->head): start {heights[0]:.3f} m | "
                f"min {min(heights):.3f} m | max {max(heights):.3f} m | final {heights[-1]:.3f} m "
                f"(top {last[11]:.3f}, bottom {last[12]:.3f})",
                flush=True,
            )

        if proc.returncode != 0:
            print(f"ffmpeg exited with code {proc.returncode}; see {stderr_path}", file=sys.stderr)
            return 1
        return 0
    finally:
        pl.env.shutdown()


def main() -> int:
    args = _parse_args()
    try:
        return run(args)
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
