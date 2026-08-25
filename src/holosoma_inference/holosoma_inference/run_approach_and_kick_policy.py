#!/usr/bin/env python3
"""Standalone entry point for ApproachAndKickPolicy -- deliberately bypasses run_policy.py's
_select_policy_class() auto-detection (which would route a UnifiedPolicy-shaped observation space
to UnifiedPolicy, not this subclass) rather than adding a new branch to that shared dispatch
logic. Zero changes to run_policy.py / dual_mode.py / unified.py; this file is purely additive.

Usage (same physics-side run_sim.py as UnifiedPolicy -- see this module's own README section for
current sim2real status, notably that no ball exists in the run_sim.py MuJoCo scene yet):
    python3 -m holosoma_inference.run_approach_and_kick_policy \
        --task.model-path /path/to/model.onnx --task.interface lo \
        --ball-source fixed --ball-fixed-dx 1.6 --ball-fixed-dy 0.0
"""

from __future__ import annotations

import argparse
import sys
import traceback

import tyro
from loguru import logger

from holosoma_inference.ball_pose_source import FixedBallPoseSource, UdpBallPoseSource
from holosoma_inference.config.config_types.inference import InferenceConfig
from holosoma_inference.config.config_values.inference import DEFAULTS
from holosoma_inference.config.utils import TYRO_CONFIG
from holosoma_inference.policies.approach_and_kick import ApproachAndKickPolicy
from holosoma_inference.utils.misc import restore_terminal_settings


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--ball-source", choices=["fixed", "udp"], default="fixed")
    pre.add_argument("--ball-fixed-dx", type=float, default=1.6)
    pre.add_argument("--ball-fixed-dy", type=float, default=0.0)
    pre.add_argument("--ball-udp-port", type=int, default=5599)
    pre.add_argument(
        "--target-dx", type=float, default=None,
        help="Fixed body-frame target to approach+trigger at. Default (None): 1.6 in classic mode, "
        "or auto-derived from --kick-region-map-path's best-supported cell if that's set.",
    )
    pre.add_argument("--target-dy", type=float, default=None)
    pre.add_argument(
        "--kick-region-map-path", type=str, default=None,
        help="Path (without extension) to a KickRegionMap saved by "
        "high_level_ball_approach_and_kicking/src/hl_kick/map_kick_region_prob.py (reads "
        "<path>.npz/<path>.json). If set, the trigger gates on live P(success) at the ball's "
        "current position crossing --success-probability-threshold, instead of distance to a "
        "fixed target -- see ClassicalApproachAndKickController's docstring.",
    )
    pre.add_argument("--success-probability-threshold", type=float, default=0.72)
    known, remaining = pre.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    preset = DEFAULTS["g1-29dof-unified-loco-kick"]
    config = tyro.cli(InferenceConfig, default=preset, config=TYRO_CONFIG)

    if known.ball_source == "fixed":
        logger.warning(
            "Using FixedBallPoseSource -- the robot will walk toward a FIXED virtual point "
            f"({known.ball_fixed_dx}, {known.ball_fixed_dy}), NOT a real or simulated ball. "
            "This mode only smoke-tests the control loop's integration with the real policy "
            "machinery. Use --ball-source udp once a ball-position publisher exists (see "
            "playground/high_level_ball_approach_and_kicking/README.md's sim2real section)."
        )
        ball_source = FixedBallPoseSource(known.ball_fixed_dx, known.ball_fixed_dy)
    else:
        logger.info(f"Listening for ball pose on UDP port {known.ball_udp_port}")
        ball_source = UdpBallPoseSource(known.ball_udp_port)

    try:
        logger.info(f"🤖 Robot: {config.robot.robot_type}")
        logger.info(f"📁 Model path: {config.task.model_path}")
        policy = ApproachAndKickPolicy(
            config=config, ball_pose_source=ball_source, target_dx=known.target_dx, target_dy=known.target_dy,
            kick_region_map_path=known.kick_region_map_path,
            success_probability_threshold=known.success_probability_threshold,
        )
        logger.info("✅ ApproachAndKickPolicy initialized successfully!")
        logger.info(
            "💡 Controls: same as UnifiedPolicy (]=start, [=stop, 8/9=gantry) -- once started, "
            "velocity + kick trigger are driven autonomously from the ball pose source instead of "
            "keyboard input. W/A/S/D/Q/E and K/L are still dispatched if pressed (manual override)."
        )
        policy.run()
        logger.info("✅ Policy execution completed!")
    except Exception:
        logger.error("❌ Error running ApproachAndKickPolicy")
        traceback.print_exc()
        sys.exit(1)
    finally:
        restore_terminal_settings()


if __name__ == "__main__":
    main()
