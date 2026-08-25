#!/usr/bin/env python3
"""Standalone entry point for NetworkControlledUnifiedPolicy -- the LL half of the genuinely
separate-process HL/LL deployment (see run_classical_hl_controller.py for the HL half). Same
run_policy.py-bypass rationale as run_approach_and_kick_policy.py.

Usage (three independent terminals -- see README's "Parallel HL/LL deployment" section):
    python3 -m holosoma_inference.run_network_controlled_policy \
        --task.model-path /path/to/model.onnx --task.interface lo \
        --command-port 5700

By default this also opens a small popup window (tkinter) showing the live vx/vy/wyaw/kick-trigger
values this process is actually receiving from Terminal 3 -- pure observability, read-only, on its
own thread; never touches the real 50Hz RL loop. Pass --no-show-command-gui to skip it (e.g. no
reachable display); if a display isn't reachable this degrades to a warning + continues without
the popup rather than failing the whole process, either way.
"""

from __future__ import annotations

import argparse
import sys
import traceback

import tyro
from loguru import logger

from holosoma_inference.config.config_types.inference import InferenceConfig
from holosoma_inference.config.config_values.inference import DEFAULTS
from holosoma_inference.config.utils import TYRO_CONFIG
from holosoma_inference.hl_command_gui import launch_hl_command_gui
from holosoma_inference.policies.network_controlled import NetworkControlledUnifiedPolicy
from holosoma_inference.utils.misc import restore_terminal_settings


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--command-port", type=int, default=5700)
    pre.add_argument(
        "--show-command-gui",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open a popup window showing live vx/vy/wyaw/kick-trigger as received from the HL "
        "process. --no-show-command-gui to skip (e.g. no reachable display).",
    )
    pre.add_argument(
        "--filter-out-loco", type=float, default=0.0, dest="filter_out_loco_s",
        help="Seconds, measured FROM the kick trigger, during which locomotion (linear velocity + "
        "yaw rate) is forced to strictly zero AND any new kick trigger is ignored. The window "
        "spans the kick clip itself and however much time is left once the robot returns to "
        "locomotion control, giving it dedicated settle time before it resumes whatever the HL's "
        "approach state machine commands next. A second trigger arriving before the window elapses "
        "is dropped; the controller is ready for a fresh trigger only after the count finishes. "
        "Default 0.0 -- disabled, identical to prior behavior.",
    )
    known, remaining = pre.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    preset = DEFAULTS["g1-29dof-unified-loco-kick"]
    config = tyro.cli(InferenceConfig, default=preset, config=TYRO_CONFIG)

    try:
        logger.info(f"🤖 Robot: {config.robot.robot_type}")
        logger.info(f"📁 Model path: {config.task.model_path}")
        logger.info(f"Listening for HL commands on UDP port {known.command_port}")
        policy = NetworkControlledUnifiedPolicy(
            config=config, command_port=known.command_port, filter_out_loco_s=known.filter_out_loco_s
        )
        logger.info("✅ NetworkControlledUnifiedPolicy initialized successfully!")
        logger.info(
            "💡 Controls: same as UnifiedPolicy (]=start, [=stop, 8/9=gantry) -- once started, "
            "velocity + kick trigger come from the HL command channel instead of keyboard input. "
            "W/A/S/D/Q/E and K/L are still dispatched if pressed (manual override)."
        )
        if known.filter_out_loco_s > 0:
            logger.info(f"Locomotion filter active: forcing zero velocity for {known.filter_out_loco_s}s after each trigger")
        if known.show_command_gui:
            launch_hl_command_gui(
                policy.receiver, known.command_port,
                get_loco_filter_remaining=policy.loco_filter_remaining_s,
                get_applied_command=policy.get_applied_command,
            )
        policy.run()
        logger.info("✅ Policy execution completed!")
    except Exception:
        logger.error("❌ Error running NetworkControlledUnifiedPolicy")
        traceback.print_exc()
        sys.exit(1)
    finally:
        restore_terminal_settings()


if __name__ == "__main__":
    main()
