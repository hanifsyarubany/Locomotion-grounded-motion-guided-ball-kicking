"""Interactive controls for holosoma/replay.py and holosoma/eval_interactive.py.

Independent pieces, all driven from the caller's main render/eval loop:
- ReplayTerminalController: play/stop/restart/quit/kick via the terminal (stdin).
- BallPositionWindow: a floating IsaacSim UI window with X/Y sliders to live-adjust the
  ball's spawn position, with a button to persist the chosen position to configs/ball.yaml.
- VelocityCommandWindow: a floating IsaacSim UI window with linear/angular velocity sliders
  that live-drive the robot's locomotion command (any experiment with locomotion_command
  registered — stock locomotion, or the locomotion side of UnifiedManager).

None of these are used during training.
"""

from __future__ import annotations

import re
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager
    from holosoma.managers.command.terms.wbt import MotionCommand


class ReplayTerminalController:
    """Reads play/stop/restart/quit commands from stdin on a background thread.

    Stdin reads block, so this runs on its own daemon thread; the main render loop just polls
    the plain bool flags below once per frame. No locks: only this thread ever writes them and
    only the main thread ever reads them, and only the main thread ever touches sim/tensor
    state, so there's no data race on the values that matter for correctness."""

    def __init__(self) -> None:
        self.playing = True
        self.restart_requested = False
        self.quit_requested = False
        self.kick_requested = False
        self._thread = threading.Thread(target=self._read_loop, daemon=True)

    def start(self) -> None:
        print(
            "\n=== Replay controls (type a command + Enter in this terminal) ===\n"
            "  p / play     resume playback\n"
            "  s / stop     pause playback\n"
            "  r / restart  restart the motion clip from the beginning\n"
            "  k / kick     (UnifiedManager only) force a reset into the kick task on command\n"
            "  q / quit     exit\n"
        )
        self._thread.start()

    def _read_loop(self) -> None:
        for line in sys.stdin:
            cmd = line.strip().lower()
            if cmd in ("p", "play"):
                self.playing = True
                print("[replay] playing", flush=True)
            elif cmd in ("s", "stop"):
                self.playing = False
                print("[replay] stopped", flush=True)
            elif cmd in ("r", "restart"):
                self.restart_requested = True
                print("[replay] restart requested", flush=True)
            elif cmd in ("k", "kick"):
                self.kick_requested = True
                print("[replay] kick requested", flush=True)
            elif cmd in ("q", "quit"):
                self.quit_requested = True
                print("[replay] quitting...", flush=True)
                return
            elif cmd:
                print(f"[replay] unknown command {cmd!r} (use p/s/r/k/q)", flush=True)


def save_ball_xy_to_yaml(yaml_path: str | Path, x: float, y: float) -> None:
    """Rewrite only the `x:`/`y:` value lines in the ball config YAML in place, preserving
    every comment and the rest of the file's formatting. A plain yaml.safe_dump round-trip
    would silently drop all the explanatory comments in the file, so this edits the raw text
    instead."""
    yaml_path = Path(yaml_path)
    text = yaml_path.read_text()
    text, n_x = re.subn(r"(?m)^x:\s*[-+]?[0-9]*\.?[0-9]+", f"x: {x:g}", text)
    text, n_y = re.subn(r"(?m)^y:\s*[-+]?[0-9]*\.?[0-9]+", f"y: {y:g}", text)
    if n_x != 1 or n_y != 1:
        raise ValueError(
            f"Expected exactly one 'x:' and one 'y:' line in {yaml_path}, found {n_x} and {n_y}. "
            "Refusing to write — the file may have been reformatted; edit it by hand instead."
        )
    yaml_path.write_text(text)


class BallPositionWindow:
    """Floating IsaacSim window with X/Y sliders that move the ball live in the running sim,
    plus a Save button that persists the current slider values to configs/ball.yaml.

    Moving the sliders also patches `motion_command.ball_reset_state` in place (not just the
    live sim pose) so that a subsequent restart — which re-applies `ball_reset_state` — keeps
    whatever position you dragged to, rather than snapping back to the position that was loaded
    from YAML at startup.
    """

    def __init__(
        self,
        env: WholeBodyTrackingManager,
        motion_command: MotionCommand,
        yaml_path: str | Path,
        x_range: tuple[float, float] = (0.0, 3.0),
        y_range: tuple[float, float] = (-1.5, 1.5),
    ) -> None:
        import omni.ui as ui
        from isaacsim.gui.components.ui_utils import btn_builder, combo_floatfield_slider_builder

        self._env = env
        self._motion_command = motion_command
        self._yaml_path = Path(yaml_path)
        self._env_ids = torch.arange(env.num_envs, device=env.device)

        init_x = float(motion_command.ball_reset_state[0].item())
        init_y = float(motion_command.ball_reset_state[1].item())

        self._window = ui.Window(
            "Ball Position", width=360, height=200, visible=True, dock_preference=ui.DockPreference.RIGHT_TOP
        )
        with self._window.frame:
            with ui.VStack(spacing=5, height=5):
                ui.Label(
                    "Drag to move the ball live. Save writes the current x,y to configs/ball.yaml.",
                    word_wrap=True,
                    height=30,
                )
                self._x_model, _ = combo_floatfield_slider_builder(
                    label="Ball X (m)",
                    default_val=init_x,
                    min=x_range[0],
                    max=x_range[1],
                    step=0.01,
                    tooltip=["Forward distance from the robot", ""],
                )
                self._y_model, _ = combo_floatfield_slider_builder(
                    label="Ball Y (m)",
                    default_val=init_y,
                    min=y_range[0],
                    max=y_range[1],
                    step=0.01,
                    tooltip=["Lateral offset from the robot", ""],
                )
                # both the field and the slider share this one model, so this catches drags,
                # scroll-wheel nudges, and typed values uniformly
                self._x_model.add_value_changed_fn(lambda _m: self._on_change())
                self._y_model.add_value_changed_fn(lambda _m: self._on_change())
                btn_builder(text="Save to ball.yaml", on_clicked_fn=self._on_save)

        # apply the loaded position immediately so the live ball matches the sliders as soon
        # as the window appears, even if something else had moved the ball beforehand
        self._on_change()

    def _current_xy(self) -> tuple[float, float]:
        return float(self._x_model.get_value_as_float()), float(self._y_model.get_value_as_float())

    def _on_change(self) -> None:
        x, y = self._current_xy()
        # x, y are ROBOT-LOCAL ("Forward distance from the robot" / "Lateral offset from the
        # robot", per this window's own tooltips above, and SkillConfig.x/y's meaning) -- NOT raw
        # world coordinates. Routed through MotionCommand.local_xy_to_world (the same transform
        # reset()'s real ball placement uses) so what you see while dragging matches what an
        # actual training reset will place, instead of the two silently disagreeing.
        # z, orientation, and velocity stay as already configured; only x,y move live.
        self._motion_command.ball_reset_state[0] = x
        self._motion_command.ball_reset_state[1] = y
        states = self._motion_command.ball_reset_state.unsqueeze(0).expand(len(self._env_ids), -1).clone()
        states[:, :2] = self._motion_command.local_xy_to_world(states[:, :2], self._env_ids)
        self._env.simulator.set_actor_states(["ball"], self._env_ids, states)

    def _on_save(self) -> None:
        x, y = self._current_xy()
        save_ball_xy_to_yaml(self._yaml_path, x, y)
        print(f"[replay] saved ball position to {self._yaml_path}: x={x:g}, y={y:g}", flush=True)

    def destroy(self) -> None:
        if self._window is not None:
            self._window.visible = False
            self._window.destroy()
            self._window = None


class VelocityCommandWindow:
    """Floating IsaacSim window with linear-x/linear-y/angular-yaw sliders that live-drive the
    robot's locomotion velocity command (`env.command_manager.commands`, a single [num_envs, 3]
    tensor shared by every env — same buffer LocomotionCommand normally resamples during
    training, held fixed here since `set_is_evaluating()` disables that resampling).

    Only has a visible effect on envs currently running the locomotion task: on a UnifiedManager
    checkpoint, the command_lin_vel/command_ang_vel observation terms are zeroed for envs in
    kick-mode (see managers/observation/manager.py's task_mode masking), so driving the command
    while the robot is mid-kick won't do anything until it's back in locomotion mode.
    """

    def __init__(
        self,
        env,
        lin_x_range: tuple[float, float] = (-1.0, 1.0),
        lin_y_range: tuple[float, float] = (-1.0, 1.0),
        ang_yaw_range: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        import omni.ui as ui
        from isaacsim.gui.components.ui_utils import btn_builder, combo_floatfield_slider_builder

        self._commands = env.command_manager.commands  # [num_envs, 3]: lin_x, lin_y, ang_yaw

        self._window = ui.Window(
            "Locomotion Velocity", width=360, height=240, visible=True, dock_preference=ui.DockPreference.RIGHT_BOTTOM
        )
        with self._window.frame:
            with ui.VStack(spacing=5, height=5):
                ui.Label(
                    "Drives the robot's velocity command live. Only affects envs currently in "
                    "locomotion mode.",
                    word_wrap=True,
                    height=30,
                )
                self._x_model, _ = combo_floatfield_slider_builder(
                    label="Lin X (m/s)",
                    default_val=0.0,
                    min=lin_x_range[0],
                    max=lin_x_range[1],
                    step=0.05,
                    tooltip=["Forward/backward velocity", ""],
                )
                self._y_model, _ = combo_floatfield_slider_builder(
                    label="Lin Y (m/s)",
                    default_val=0.0,
                    min=lin_y_range[0],
                    max=lin_y_range[1],
                    step=0.05,
                    tooltip=["Lateral (strafe) velocity", ""],
                )
                self._yaw_model, _ = combo_floatfield_slider_builder(
                    label="Ang Yaw (rad/s)",
                    default_val=0.0,
                    min=ang_yaw_range[0],
                    max=ang_yaw_range[1],
                    step=0.05,
                    tooltip=["Turning rate", ""],
                )
                self._x_model.add_value_changed_fn(lambda _m: self._on_change())
                self._y_model.add_value_changed_fn(lambda _m: self._on_change())
                self._yaw_model.add_value_changed_fn(lambda _m: self._on_change())
                btn_builder(text="Zero velocity", on_clicked_fn=self._on_zero)

        self._on_change()

    def _on_change(self) -> None:
        self._commands[:, 0] = self._x_model.get_value_as_float()
        self._commands[:, 1] = self._y_model.get_value_as_float()
        self._commands[:, 2] = self._yaw_model.get_value_as_float()

    def _on_zero(self) -> None:
        # each set_value triggers _on_change via add_value_changed_fn, so the command tensor
        # stays in sync automatically
        self._x_model.set_value(0.0)
        self._y_model.set_value(0.0)
        self._yaw_model.set_value(0.0)
        print("[replay] velocity command zeroed", flush=True)

    def destroy(self) -> None:
        if self._window is not None:
            self._window.visible = False
            self._window.destroy()
            self._window = None
