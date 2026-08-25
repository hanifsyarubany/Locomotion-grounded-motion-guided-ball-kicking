#!/usr/bin/env python3
"""Interactive MuJoCo sim2sim session: drag the ball to wherever you want with the mouse, trigger
a real kick on command, and log where it actually goes -- for calibrating SkillConfig.x/y (ball
spawn) and target_x/target_y (configs/skill2.yaml) against a REAL policy's REAL kick physics.

2026-08-19, user-requested, following two earlier findings this session:
  - replay.py's clip-visualization tool cannot show this at all: it kinematically teleports the
    robot from the recorded reference motion every tick (envs/wbt/wbt_manager.py::
    step_visualize_motion), so it never generates real foot-ball contact -- measured: ball
    position identical bit-for-bit across an entire replay, including the strike window.
  - mujoco_kick_rollout_worker.py's rollout (real policy, real MuJoCo contact physics, now with
    --output-trajectory-path logging) is the right physics, but is a fixed batch script -- no way
    to choose the ball's spawn position by hand before the kick fires.

The mechanism this script needs -- freely dragging a body with the mouse -- is NOT new code: it's
MuJoCo's own standard perturbation system (mjv_applyPerturbPose/Force via the viewer's own
MjvPerturb, confirmed wired in RoboJuDo's third_party/mujoco_viewer package), the SAME interaction
every stock MuJoCo `simulate` session already has (hold Ctrl + right-click-drag on a body to
translate it; Ctrl + left-click-drag to rotate). This script's only job is to leave the viewer
ALIVE (mujoco_kick_rollout_worker.py deliberately sets `env.viewer.is_alive = False` because it
renders offscreen instead -- this script does the opposite) and drive an interactive
drag -> kick -> observe -> reset loop around it, reusing the exact same ball-spawn/observation-
patch/trajectory-recording pieces as that script.

Controls (typed in THIS terminal + Enter, same vocabulary as replay.py's ReplayTerminalController
-- while the mouse drag itself, per the paragraph above, is the viewer window's OWN native
Ctrl+drag, not a typed command):
    k / kick        trigger the kick from wherever the ball currently is
    r / restart     reset robot + ball to the nominal spawn, ready to drag again
    a <deg> / angle <deg>
                    (only with --kick-aim-enabled) set the commanded kick_aim_theta for the
                    NEXT kick, without resetting the ball -- so the same physical drag can be
                    kicked at several different commanded angles in a row. Persists (holding
                    at whatever it was last set to) across kicks and restarts alike, until
                    changed again.
    q / quit        exit (flushes a pending trial's trajectory CSV first)

2026-08-23, azimuth-aim refactor bugfix + feature: this script was silently feeding every
kick_aim_enabled checkpoint the PRE-refactor raw world-frame target_pos_b transform (obs[157:159]
computed from inner.get_skill_target_xy, ~15-18x out of that checkpoint's own trained
distribution) -- same bug class already found and fixed in mujoco_kick_rollout_worker.py's own
--kick-aim-enabled and confirmed already-correct in mujoco_kick_survival_scan.py. Fixed here via
the same --kick-aim-enabled/--kick-aim-theta-deg/--kick-aim-theta-ref-deg flags, PLUS (since this
script is interactive, unlike those two batch tools) the `a`/`angle` command above, so the
commanded angle can be varied per-kick without restarting the whole process. See
_install_ball_observation_patch's own `kick_aim_theta_deg_getter` docstring for the mechanism that
makes a live, changeable value possible without re-installing the observation patch on every
change (which would otherwise nest closures deeper each time).

Usage:
    /workspaces/isaaclab_arena/submodules/workspaces/conda_env/robojudo/bin/python \
        mujoco_kick_interactive.py --onnx-path /path/to/model_0005000.onnx \
        --output-trajectory-dir /tmp/kick_trials \
        --kick-aim-enabled --kick-aim-theta-deg 0.0 --kick-aim-theta-ref-deg 45.0
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import threading
import time
import traceback
from datetime import datetime

ROBOJUDO_REPO = "/workspaces/isaaclab_arena/submodules/workspaces/humanoid_deployment/RoboJuDo"
sys.path.insert(0, ROBOJUDO_REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mujoco_kick_rollout_worker import BALL_WORLD_POS, SCENE_WITH_BALL, _install_ball_observation_patch  # noqa: E402
from mujoco_robot_extent import RobotExtent  # noqa: E402


class _TerminalController:
    """Minimal, stdlib-only reimplementation of replay.py's ReplayTerminalController -- NOT
    imported directly because that module pulls in torch (a type-hint-only import at module
    level), which the `robojudo` conda env this script runs under does not have installed. Same
    command vocabulary (k/r/q) for consistency with replay.py's own UX, plus (2026-08-23) an
    `a`/`angle` command for live kick_aim_theta changes -- see this module's own docstring.

    `aim_enabled`: whether `a`/`angle` should be accepted at all -- set from the same
    `--kick-aim-enabled` flag that gates the observation patch, so typing `a 10` when the script
    wasn't launched with that flag gets a clear rejection instead of silently doing nothing (no
    observation patch would ever read the value in that case)."""

    def __init__(self, *, aim_enabled: bool = False) -> None:
        self.kick_requested = False
        self.restart_requested = False
        self.quit_requested = False
        self.aim_enabled = aim_enabled
        self.aim_theta_deg: float = 0.0
        self._thread = threading.Thread(target=self._read_loop, daemon=True)

    def start(self) -> None:
        aim_line = (
            "  a <deg> / angle <deg>  set the commanded kick_aim_theta for the NEXT kick "
            f"(currently {self.aim_theta_deg:g})\n"
            if self.aim_enabled
            else ""
        )
        print(
            "\n=== Interactive kick controls ===\n"
            "  Drag the ball: hold Ctrl + right-click-drag on it in the viewer window\n"
            "    (MuJoCo's own built-in body perturbation -- nothing to type for this part)\n"
            "  k / kick     trigger the kick from the ball's CURRENT position\n"
            "  r / restart  reset robot + ball to nominal spawn\n"
            f"{aim_line}"
            "  q / quit     exit\n",
            flush=True,
        )
        self._thread.start()

    def _read_loop(self) -> None:
        for line in sys.stdin:
            cmd = line.strip()
            lower = cmd.lower()
            if lower in ("k", "kick"):
                self.kick_requested = True
                print("[interactive] kick requested", flush=True)
            elif lower in ("r", "restart"):
                self.restart_requested = True
                print("[interactive] restart requested", flush=True)
            elif lower in ("q", "quit"):
                self.quit_requested = True
                print("[interactive] quitting...", flush=True)
                return
            elif lower.startswith("a ") or lower.startswith("angle "):
                if not self.aim_enabled:
                    print(
                        "[interactive] 'a'/'angle' has no effect -- this script wasn't launched "
                        "with --kick-aim-enabled, so the observation patch never reads a "
                        "kick_aim_theta value at all.", flush=True,
                    )
                    continue
                value_str = cmd.split(None, 1)[1].strip()
                try:
                    self.aim_theta_deg = float(value_str)
                except ValueError:
                    print(f"[interactive] couldn't parse {value_str!r} as a number -- try e.g. 'a 10' or 'a -5.5'", flush=True)
                    continue
                print(f"[interactive] kick_aim_theta set to {self.aim_theta_deg:g} deg (takes effect on the next kick)", flush=True)
            elif lower:
                extra = " (or 'a <deg>'/'angle <deg>')" if self.aim_enabled else ""
                print(f"[interactive] unknown command {cmd!r} (use k/r/q{extra})", flush=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx-path", required=True)
    p.add_argument("--skill-id", type=int, default=0)
    p.add_argument("--hold-s", type=float, default=8.0, help="How long to keep recording after a kick trigger.")
    p.add_argument("--fps", type=int, default=50, help="Matches the 50Hz RL control rate.")
    p.add_argument(
        "--output-trajectory-dir", default=None,
        help="Directory to write one timestamped ball-trajectory CSV per kick trial into (tick, "
        "t_s, phase, task_mode, curr_motion_timestep, ball_x/y/z, base_x/y/z, robot_top_z/"
        "bottom_z/height, kick_aim_theta_deg -- the last is the LIVE commanded angle at that "
        "exact tick, same attribute the 'a'/'angle' command sets, empty when --kick-aim-enabled "
        "wasn't passed at all). None (default) = don't record, just watch.",
    )
    p.add_argument(
        "--kick-aim-enabled", action="store_true",
        help="2026-08-23: this checkpoint was trained with SkillConfig.kick_aim_enabled=True -- "
        "feed obs[157:159] as the bounded [kick_aim_theta/kick_aim_theta_ref_deg, 0.0] command it "
        "actually trained on, instead of the pre-azimuth-refactor raw world-frame target_pos_b "
        "transform (a ~15-18x out-of-distribution magnitude for such a checkpoint -- see "
        "_install_ball_observation_patch's own docstring). Also enables the 'a'/'angle' terminal "
        "command to change the angle between kicks. Default off, so a checkpoint trained WITHOUT "
        "kick_aim_enabled is completely unaffected.",
    )
    p.add_argument(
        "--kick-aim-theta-deg", type=float, default=0.0,
        help="Starting kick_aim_theta (degrees), only used when --kick-aim-enabled -- 0.0 (default) "
        "aims straight along the skill's own calibrated nominal bearing. Change it live via the "
        "'a <deg>'/'angle <deg>' terminal command once the session is running.",
    )
    p.add_argument(
        "--kick-aim-theta-ref-deg", type=float, default=45.0,
        help="Normalization reference (degrees), only used when --kick-aim-enabled. MUST match "
        "the checkpoint's own MultiSkillConfig/BallConfig.kick_aim_theta_ref_deg -- a mismatch "
        "silently rescales every commanded theta before it reaches the policy.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    import mujoco
    import numpy as np

    import robojudo.config.g1  # noqa: F401 -- registers config
    import robojudo.pipeline  # noqa: F401 -- registers pipeline
    from robojudo.config import cfg_registry

    if args.output_trajectory_dir:
        os.makedirs(args.output_trajectory_dir, exist_ok=True)

    cfg = cfg_registry.get("g1_unified_loco_kick")()
    cfg.policy.onnx_path = args.onnx_path
    cfg.env.xml = SCENE_WITH_BALL
    pl = getattr(robojudo.pipeline, cfg.pipeline_type)(cfg=cfg)
    env = pl.env
    # Deliberately NOT setting env.viewer.is_alive = False here -- that's what
    # mujoco_kick_rollout_worker.py does to render offscreen instead. Leaving it at its
    # constructed default (True) is the entire mechanism: MujocoEnv.step() already renders to
    # this SAME window every tick (environment/mujoco_env.py) whenever the viewer is alive, with
    # native drag-to-move already wired (see this module's own docstring).
    inner = pl.policy.policy
    inner._update_velocity_command = lambda cd: None

    # 2026-08-20, root-caused after a recorded trajectory showed task_mode flipping to "kick"
    # while this script's own `phase` column stayed "idle" for all 267 rows: the g1_unified_
    # loco_kick pipeline config (CONTROLLER="both" default -- robojudo/config/g1/
    # g1_unified_loco_kick_cfg.py:75) ALREADY wires a native pynput-based KeyboardCtrl bound to
    # the SAME keys this script cares about ("k"->[TRIGGER_KICK], "l"->[RETURN_TO_LOCO],
    # "j"->[CYCLE_KICK_SKILL]). It polls a global OS-level key listener on every pl.step()
    # (ctrl_manager.py::get_ctrl_data -> KeyboardCtrl.process_triggers -> policy.
    # post_step_callback) -- completely independent of, and invisible to, this script's own
    # terminal-based _TerminalController. Any physical k/l/j keypress -- typing "k"<Enter> in
    # THIS terminal included, since pynput doesn't care which window has OS focus -- silently
    # fires a REAL command through that second path: task_mode genuinely flips and a real kick
    # happens, on top of whatever the mouse is doing to the ball at that same moment, but this
    # script's own `kicking`/`phase` bookkeeping never learns about it (measured: a captured
    # trial with task_mode=="kick" for 82/267 ticks, phase=="idle" throughout). Clearing the
    # native controller's triggers makes this script's own explicit, terminal-gated
    # post_step_callback() call below the ONLY way to fire one.
    kb_ctrl = pl.ctrl_manager.controllers.get("KeyboardCtrl")
    if kb_ctrl is not None:
        kb_ctrl["inst"].triggers = {}
        print("[interactive] disabled RoboJuDo's native keyboard controller (this script's own k/r/q still work)", flush=True)
    else:
        print(
            "[interactive] note: no native KeyboardCtrl found to disable (already off, or "
            "CONTROLLER env var isn't 'both'/'keyboard') -- if kicks still show up unrequested, "
            "that side channel isn't the cause here.", flush=True,
        )

    mujoco.mj_resetDataKeyframe(env.model, env.data, 0)
    ball_jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_freejoint")
    assert ball_jid != -1, "ball_freejoint not found in compiled model -- is SCENE_WITH_BALL correct?"
    ball_qpos_addr = int(env.model.jnt_qposadr[ball_jid])

    def place_ball_at_nominal() -> None:
        ball_xy = inner.get_skill_ball_xy(args.skill_id)
        if ball_xy is not None:
            env.data.qpos[ball_qpos_addr] = ball_xy[0]
            env.data.qpos[ball_qpos_addr + 1] = ball_xy[1]
        else:
            env.data.qpos[ball_qpos_addr] = BALL_WORLD_POS[0]
            env.data.qpos[ball_qpos_addr + 1] = BALL_WORLD_POS[1]

    # Instantiated before the observation patch below so the patch's kick_aim_theta_deg_getter
    # can close over `controller.aim_theta_deg` -- the SAME mutable attribute the 'a'/'angle'
    # terminal command updates, so a change takes effect on the very next tick, no re-patching.
    controller = _TerminalController(aim_enabled=args.kick_aim_enabled)
    controller.aim_theta_deg = args.kick_aim_theta_deg

    place_ball_at_nominal()
    _install_ball_observation_patch(
        env, inner, mujoco, np, ball_qpos_addr, args.skill_id,
        kick_aim_theta_ref_deg=args.kick_aim_theta_ref_deg,
        kick_aim_theta_deg_getter=(lambda: controller.aim_theta_deg) if args.kick_aim_enabled else None,
    )
    mujoco.mj_forward(env.model, env.data)
    env.update()

    controller.start()

    trajectory_rows: list[
        tuple[int, float, str, str, int, float, float, float, float, float, float, float, float, float, "float | None"]
    ] = []
    # Full robot height (feet -> head) alongside base height; see mujoco_robot_extent.py.
    robot_extent = RobotExtent(mujoco, env.model)

    # 2026-08-21: robot BASE (root) pose recorded alongside the ball -- same schema and same
    # first-free-joint resolution as mujoco_kick_rollout_worker.py's own recorder (see its comment
    # for why the ordering is asserted rather than assumed). base_z is the continuous quantity
    # behind the thresholded topple metric (fall threshold 0.40 m).
    _free_jids = [j for j in range(env.model.njnt) if env.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    assert _free_jids, "no free joint in the compiled model -- cannot locate the robot root"
    robot_root_qadr = int(env.model.jnt_qposadr[_free_jids[0]])
    assert robot_root_qadr != ball_qpos_addr, (
        f"the first free joint IS the ball (qposadr {ball_qpos_addr}) -- the robot's freejoint must "
        "come first; see scene_g1_29dof_with_ball.xml's ordering comment"
    )

    def record(phase: str) -> None:
        if not args.output_trajectory_dir:
            return
        pos = env.data.qpos[ball_qpos_addr : ball_qpos_addr + 3]
        base = env.data.qpos[robot_root_qadr : robot_root_qadr + 3]
        top_z, bottom_z, height = robot_extent.measure(env.data)
        # 2026-08-23: the LIVE value at this exact tick (same attribute the observation-patch
        # getter reads), not a fixed per-trial constant -- if the angle is changed mid-hold (the
        # 'a'/'angle' command doesn't block on kicking), that's a real event that changed what the
        # policy was actually fed, and the CSV should say so rather than paper over it with
        # whatever the value happened to be at kick-trigger time. None (written as an empty cell)
        # when this session wasn't launched with --kick-aim-enabled at all -- there's no commanded
        # angle to report, not a genuine 0.0.
        aim_theta_deg = controller.aim_theta_deg if args.kick_aim_enabled else None
        trajectory_rows.append((
            len(trajectory_rows), len(trajectory_rows) / args.fps, phase, inner.task_mode,
            int(inner.curr_motion_timestep), float(pos[0]), float(pos[1]), float(pos[2]),
            float(base[0]), float(base[1]), float(base[2]),
            float(top_z), float(bottom_z), float(height),
            aim_theta_deg,
        ))

    def flush_trajectory() -> None:
        nonlocal trajectory_rows
        if not args.output_trajectory_dir or not trajectory_rows:
            return
        out_path = os.path.join(
            args.output_trajectory_dir, f"ball_trajectory_{datetime.now():%Y%m%d_%H%M%S}.csv"
        )
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "tick", "t_s", "phase", "task_mode", "curr_motion_timestep",
                "ball_x", "ball_y", "ball_z", "base_x", "base_y", "base_z",
                "robot_top_z", "robot_bottom_z", "robot_height", "kick_aim_theta_deg",
            ])
            w.writerows(trajectory_rows)
        last = trajectory_rows[-1]
        base_z = [r[10] for r in trajectory_rows]
        min_base_z = min(base_z)
        min_tick = base_z.index(min_base_z)
        print(
            f"[interactive] wrote {len(trajectory_rows)} ticks -> {out_path} "
            f"(final ball pos: x={last[5]:.3f} y={last[6]:.3f} z={last[7]:.3f})",
            flush=True,
        )
        print(
            f"[base-height] final {last[10]:.3f} m | min {min_base_z:.3f} m at tick {min_tick} "
            f"| {'FELL' if min_base_z < 0.40 else 'stayed up'} vs the 0.40 m fall threshold",
            flush=True,
        )
        hh = [r[13] for r in trajectory_rows]
        print(
            f"[robot-height] full height (feet->head): start {hh[0]:.3f} m | min {min(hh):.3f} m | "
            f"max {max(hh):.3f} m | final {hh[-1]:.3f} m",
            flush=True,
        )
        trajectory_rows = []

    def step_zero_vel() -> None:
        inner.lin_vel_command = np.zeros(2)
        inner.ang_vel_command = 0.0
        pl.step()

    print("[interactive] ready -- ball at nominal spawn, drag freely, then type 'k' to kick.", flush=True)

    try:
        kicking = False
        hold_ticks_left = 0
        while not controller.quit_requested:
            if controller.restart_requested:
                flush_trajectory()
                mujoco.mj_resetDataKeyframe(env.model, env.data, 0)
                inner.reset()
                inner._update_velocity_command = lambda cd: None
                place_ball_at_nominal()
                mujoco.mj_forward(env.model, env.data)
                env.update()
                controller.restart_requested = False
                kicking = False
                print("[interactive] reset -- drag freely, then type 'k' to kick.", flush=True)

            if controller.kick_requested:
                controller.kick_requested = False
                # 2026-08-20, revised so no issued trigger is ever silently dropped or merged:
                # previously this whole block was gated on `not kicking`, so a 'k' pressed while a
                # PRIOR hold window was still counting down just sat in controller.kick_requested
                # (never cleared) until that window expired on its own -- fired eventually, but
                # attributed to whichever kick happened to be active when the flag was finally
                # read, not necessarily the one you meant right now. Every kick_requested is now
                # honored the instant it's seen: if one's already in progress, flush it first (even
                # if short -- still a real, if truncated, recording of that attempt) so this new
                # trigger starts its own clean file rather than silently overwriting the old one's
                # in-flight buffer.
                if kicking:
                    flush_trajectory()
                    print("[interactive] new kick requested mid-hold -- flushed the in-progress trial early", flush=True)
                env.update()
                inner.get_observation(env.get_data(), {})
                trigger_cmd = "[TRIGGER_KICK]" if args.skill_id == 0 else f"[TRIGGER_KICK:{args.skill_id}]"
                inner.post_step_callback([trigger_cmd])
                kicking = True
                hold_ticks_left = int(args.hold_s * args.fps)
                aim_note = f" (kick_aim_theta={controller.aim_theta_deg:g} deg)" if args.kick_aim_enabled else ""
                print(f"[interactive] kicked{aim_note} -- recording for {args.hold_s:.1f}s", flush=True)

            step_zero_vel()
            record("hold" if kicking else "idle")

            if kicking:
                hold_ticks_left -= 1
                if hold_ticks_left <= 0:
                    kicking = False
                    flush_trajectory()
                    print("[interactive] hold window finished. Type 'k' to kick again from here, 'r' to reset first, or 'q' to quit.", flush=True)

            if not env.viewer.is_alive:
                print("[interactive] viewer window closed -- exiting.", flush=True)
                break

            # render()'s own GLFW loop (mujoco_viewer.py) has no real-time pacing of its own --
            # purely event-driven (poll_events/swap_buffers), so without this an idle loop with a
            # trivial scene could spin far faster than 1/fps, making the window's redraw rate (and
            # therefore how responsive a live mouse-drag LOOKS) depend entirely on whatever
            # render/physics overhead happens to exist rather than something predictable.
            time.sleep(1.0 / args.fps)
    finally:
        flush_trajectory()

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
