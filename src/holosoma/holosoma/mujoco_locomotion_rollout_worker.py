#!/usr/bin/env python3
"""Standalone MuJoCo sim2sim locomotion-rollout worker for the 4th wandb rollout ("Training
rollout - MuJoCo Walk"). NOT part of the `holosoma` package's own runtime -- this script is only
ever invoked as a subprocess via the `robojudo` conda env's interpreter (see
`record_mujoco_locomotion_rollout.py`, which is what the training process actually imports and
which has zero dependency on this file's imports), mirroring `mujoco_kick_rollout_worker.py`'s
architecture exactly.

Scripts a forward-walk -> stop-and-stand sequence (no kick ever triggered; task_mode stays
"locomotion" for the whole rollout) with a real, physically-simulated ball in the scene, and
captures it the same way the kick rollout does (offscreen `mujoco.Renderer` frames piped to an
`ffmpeg` subprocess).

Why this exists (2026-07-21, user directive): the kick rollout (#3) only ever exercises kick-mode
episodes. It has no signal on the OTHER episode type this project trains -- plain locomotion --
under sim2sim conditions, or on how a fast walk-to-stop transition (the specific failure mode
documented in config_values/unified/g1/command.py's `_loco_command_stop_robust`: "reliably stops
from moderate speed but FALLS stopping from a fast walk (0.8 m/s)") holds up in MuJoCo/RoboJuDo
with a physical ball sitting in the scene as an added obstacle. Default `--forward-speed 0.8`
matches that documented failure threshold directly, rather than picking an arbitrary walking speed.

The ball is spawned but never observed here (task_mode never becomes "kick", so RoboJuDo's own
zeroed kick_ball_pos_b/kick_target_pos_b are never patched to real values) -- this matches
training's own task_mode_mask behavior for locomotion episodes (managers/observation/manager.py),
where kick_* observation terms are zeroed for envs not currently running the kick task regardless
of whether a ball is physically present in the scene.

Usage (manual test):
    /workspaces/isaaclab_arena/submodules/workspaces/conda_env/robojudo/bin/python \
        mujoco_locomotion_rollout_worker.py --onnx-path /path/to/model_0005000.onnx \
        --output-video-path /tmp/test_walk.mp4
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import traceback

FFMPEG_FALLBACK = "/workspaces/isaaclab_arena/submodules/workspaces/conda_env/hssim_holosoma/bin/ffmpeg"
ROBOJUDO_REPO = "/workspaces/isaaclab_arena/submodules/workspaces/humanoid_deployment/RoboJuDo"
SCENE_WITH_BALL = ROBOJUDO_REPO + "/assets/robots/g1/holosoma_model/scene_g1_29dof_with_ball.xml"

# Same placement as mujoco_kick_rollout_worker.py's BALL_WORLD_POS -- not imported from there since
# these are two independently-invoked standalone subprocess scripts by design (see module
# docstring). The robot starts at the world origin facing +x, so a ball at x=1.3 sits directly in
# the forward-walk path: at --forward-speed 0.8 m/s the robot reaches it ~1.6s into the 5s walk
# phase, making this a real physical-obstacle test, not just a decorative prop.
BALL_WORLD_POS = (1.3, 0.0, 0.11)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-path", required=True)
    parser.add_argument("--output-video-path", required=True)
    parser.add_argument("--walk-s", type=float, default=5.0, help="Forward-walk command duration.")
    parser.add_argument("--stand-s", type=float, default=3.0, help="Zero-velocity stand duration after the walk.")
    parser.add_argument(
        "--forward-speed", type=float, default=0.8,
        help="Forward (+x, body-frame) velocity command in m/s during the walk phase. Default 0.8 "
        "matches the documented 'falls stopping from a fast walk' threshold (config_values/"
        "unified/g1/command.py), so this rollout directly probes that failure mode.",
    )
    parser.add_argument("--fps", type=int, default=50, help="Matches the 50Hz RL control rate.")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    sys.path.insert(0, ROBOJUDO_REPO)
    import mujoco
    import numpy as np

    import robojudo.config.g1  # noqa: F401 -- registers config
    import robojudo.pipeline  # noqa: F401 -- registers pipeline
    from robojudo.config import cfg_registry

    ffmpeg = shutil.which("ffmpeg") or FFMPEG_FALLBACK

    cfg = cfg_registry.get("g1_unified_loco_kick")()
    cfg.policy.onnx_path = args.onnx_path
    cfg.env.xml = SCENE_WITH_BALL
    pl = getattr(robojudo.pipeline, cfg.pipeline_type)(cfg=cfg)
    env = pl.env
    env.viewer.is_alive = False
    inner = pl.policy.policy
    inner._update_velocity_command = lambda cd: None

    try:
        renderer = mujoco.Renderer(env.model, height=args.height, width=args.width)
        camera = mujoco.MjvCamera()
        camera.distance = 4.0
        camera.azimuth = 90.0
        camera.elevation = -5.0

        mujoco.mj_resetDataKeyframe(env.model, env.data, 0)
        mujoco.mj_forward(env.model, env.data)
        env.update()

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

            def step_vel(vx: float) -> None:
                inner.lin_vel_command = np.array([vx, 0.0])
                inner.ang_vel_command = 0.0
                pl.step()

            try:
                for _ in range(int(args.walk_s * args.fps)):
                    step_vel(args.forward_speed)
                    capture_frame()

                for _ in range(int(args.stand_s * args.fps)):
                    step_vel(0.0)
                    capture_frame()
            finally:
                proc.stdin.close()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)

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
