from __future__ import annotations

import tyro

from holosoma.config_types.env import get_tyro_env_config
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_values.experiment import AnnotatedExperimentConfig
from holosoma.utils.eval_utils import (
    init_sim_imports,
)
from holosoma.utils.helpers import get_class
from holosoma.utils.sim_utils import close_simulation_app
from holosoma.utils.tyro_utils import TYRO_CONIFG

# 2026-08-19, user-requested: opt-in ball-trajectory recorder for accurately setting
# SkillConfig.target_x/target_y (configs/skill2.yaml) -- ball spawn (x, y) is easy to set visually
# via BallPositionWindow's live sliders, but the target is where a KICKED ball actually ends up,
# which only exists as a trajectory over time, not a single visually-draggable point. Unset
# (default) = exact no-op, nothing recorded, byte-identical to before this existed. Set to a
# directory path to write one timestamped CSV per play-through (env 0 only -- see
# `--training.num_envs=1` in this project's own documented replay usage) of
# (tick, time_steps, t_s, ball_x, ball_y, ball_z), sourced from
# MotionCommand.live_ball_pos_w -- the same live, post-physics ball position property
# BallPositionWindow's own docstring distinguishes from the last-PLACED (pre-physics) position.
_RECORD_BALL_TRAJECTORY_ENV_VAR = "HOLOSOMA_REPLAY_RECORD_BALL_TRAJECTORY_DIR"


def replay(tyro_config: ExperimentConfig):
    simulation_app = init_sim_imports(tyro_config)

    import sys
    import time

    import torch

    # Kit's shutdown path bypasses normal interpreter exit (see close_simulation_app), which
    # skips flushing a block-buffered stdout — without this, prints from the control loop below
    # (and step_visualize_motion's own time_steps print) can silently vanish instead of showing
    # up in the terminal in real time.
    sys.stdout.reconfigure(line_buffering=True)

    import csv
    import os
    from datetime import datetime

    from holosoma.config_types.simulator import DEFAULT_BALL_CONFIG_YAML
    from holosoma.utils.common import seeding
    from holosoma.utils.replay_controls import BallPositionWindow, ReplayTerminalController

    seeding(42, torch_deterministic=False)

    record_dir = os.environ.get(_RECORD_BALL_TRAJECTORY_ENV_VAR, "").strip()
    if record_dir:
        os.makedirs(record_dir, exist_ok=True)
        print(f"[replay] ball trajectory recording ON -> {record_dir}/ (one CSV per play-through)", flush=True)

    env_target = tyro_config.env_class
    tyro_env_config = get_tyro_env_config(tyro_config)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = get_class(env_target)(tyro_env_config, device=device)

    motion_command = env.command_manager.get_state("motion_command")
    all_env_ids = torch.arange(env.num_envs, device=device)

    controller = ReplayTerminalController()
    controller.start()

    # only meaningful for the ball-kick experiment, and only when there's an actual window to
    # put it in
    has_ball = getattr(motion_command, "has_ball", False)
    ball_window = None
    if has_ball and not tyro_config.training.headless:
        ball_window = BallPositionWindow(env, motion_command, DEFAULT_BALL_CONFIG_YAML)
        print("[replay] Ball Position window created", flush=True)
    elif has_ball:
        print("[replay] ball is configured but running headless — no window to show it in", flush=True)
    else:
        print(
            "[replay] no ball configured for this experiment (scene.ball is None) — skipping "
            "Ball Position window",
            flush=True,
        )

    trajectory_rows: list[tuple[int, int, float, float, float, float]] = []
    # ReplayTerminalController starts with playing=True (replay_controls.py) -- the FIRST
    # play-through never goes through the restart branch below, so this must already reflect
    # "recording is on, nothing written yet for it" from the start, not just after a restart.
    trial_written = not bool(record_dir)

    def _write_trajectory_csv() -> None:
        nonlocal trial_written
        if not record_dir or not trajectory_rows or trial_written:
            return
        out_path = os.path.join(record_dir, f"ball_trajectory_{datetime.now():%Y%m%d_%H%M%S}.csv")
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["tick", "time_steps", "t_s", "ball_x", "ball_y", "ball_z"])
            w.writerows(trajectory_rows)
        last = trajectory_rows[-1]
        print(
            f"[replay] wrote {len(trajectory_rows)} ticks -> {out_path} "
            f"(final ball pos: x={last[3]:.3f} y={last[4]:.3f} z={last[5]:.3f})",
            flush=True,
        )
        trial_written = True

    while not controller.quit_requested:
        if controller.restart_requested:
            # A play-through interrupted by a restart (never reached `done`) still had real,
            # possibly-useful data recorded -- flush it before starting the fresh trial's buffer,
            # rather than silently discarding it.
            _write_trajectory_csv()
            trajectory_rows = []
            trial_written = not bool(record_dir)  # nothing to write yet unless recording is on

            # Deliberately not motion_command.reset(): that reuses training's randomized/adaptive
            # phase sampler, which can land anywhere in the clip (including right near the end) —
            # useless for "replay the kick from the top". Jump straight to frame 0 instead;
            # step_visualize_motion() (below) reads the robot's pose from whatever time_steps
            # currently is, so this alone is enough to visually snap the robot back too.
            motion_command.time_steps[all_env_ids] = 0
            if getattr(motion_command, "has_ball", False):
                # ball_reset_state's x,y are robot-local (forward/lateral) -- routed through the
                # same local_xy_to_world transform reset()'s real ball placement uses (anchored to
                # the robot's actual, clip-derived spawn position/heading at frame 0, now that
                # time_steps was just reset above), so restarting the replay shows the same
                # placement an actual training reset would produce.
                ball_states = motion_command.ball_reset_state.unsqueeze(0).expand(len(all_env_ids), -1).clone()
                ball_states[:, :2] = motion_command.local_xy_to_world(ball_states[:, :2], all_env_ids)
                env.simulator.set_actor_states([motion_command.ball_name], all_env_ids, ball_states)
            controller.restart_requested = False
            controller.playing = True
            print("[replay] restarted", flush=True)

        if controller.playing:
            env.simulator.sim.step()
            done = env.step_visualize_motion(None)  # type: ignore[attr-defined]
            if record_dir and getattr(motion_command, "has_ball", False):
                pos = motion_command.live_ball_pos_w[0]
                trajectory_rows.append((
                    len(trajectory_rows),
                    int(motion_command.time_steps[0].item()),
                    len(trajectory_rows) * env.dt,
                    float(pos[0].item()),
                    float(pos[1].item()),
                    float(pos[2].item()),
                ))
            if done:
                controller.playing = False
                _write_trajectory_csv()
                print(
                    "[replay] motion finished — paused. Type 'r' to restart or 'p' to replay in place.", flush=True
                )
        else:
            # keep the sim/UI responsive (window dragging, slider drags, viewport nav) without
            # advancing the motion
            simulation_app.update()
            time.sleep(0.03)

    _write_trajectory_csv()  # flush a trial interrupted by quit (never reached `done`)

    if ball_window is not None:
        ball_window.destroy()
    close_simulation_app(simulation_app)


def main() -> None:
    tyro_cfg = tyro.cli(AnnotatedExperimentConfig, config=TYRO_CONIFG)
    replay(tyro_cfg)


if __name__ == "__main__":
    main()
