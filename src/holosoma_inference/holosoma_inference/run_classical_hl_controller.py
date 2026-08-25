#!/usr/bin/env python3
"""Standalone HL controller process -- the 3rd terminal in the genuinely-parallel HL/LL/sim
deployment (see run_network_controlled_policy.py for the LL half, run_sim.py for the sim side).

Connects to the real DDS bridge itself (read-only: only ever calls get_low_state(), never
send_low_command()) to get robot state, reads ball position from a BallPoseSource, runs the
classical approach+kick control law (ClassicalApproachAndKickController -- the exact same logic
ApproachAndKickPolicy runs in-process, not a reimplementation), and publishes the resulting
velocity + kick-trigger command over UDP to the LL process at a fixed tick rate.

This process has no ONNX model, no observation construction, no PD control, and never touches
low_cmd -- it is a pure decision-maker, structurally the closest thing in this codebase to what a
separate real HL compute stack talking to a separate real LL controller would look like.

Usage (three independent terminals):
    Terminal 1: python -m holosoma.run_sim simulator:mujoco robot:g1-29dof \
                    --broadcast-ball-udp-port 5599 --broadcast-ball-delay-s 15
    Terminal 2: python -m holosoma_inference.run_network_controlled_policy \
                    --task.model-path <path> --task.interface lo --command-port 5700
    Terminal 3: python -m holosoma_inference.run_classical_hl_controller \
                    --task.interface lo --ball-udp-port 5599 --command-port 5700
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback

from loguru import logger

from holosoma_inference.ball_pose_source import FixedBallPoseSource, UdpBallPoseSource
from holosoma_inference.config.config_values.inference import DEFAULTS
from holosoma_inference.hl_command_channel import HlCommand, send_hl_command
from holosoma_inference.policies.classical_controller import ClassicalApproachAndKickController
from holosoma_inference.sdk import create_interface


def main() -> None:
    import socket

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task.interface", dest="interface", type=str, default="lo")
    parser.add_argument("--task.domain-id", dest="domain_id", type=int, default=0)
    parser.add_argument("--rl-rate", type=float, default=50.0, help="HL decision/publish rate, Hz.")
    parser.add_argument("--command-port", type=int, default=5700, help="UDP port the LL process listens on.")
    parser.add_argument("--ball-source", choices=["fixed", "udp"], default="udp")
    parser.add_argument("--ball-fixed-dx", type=float, default=1.6)
    parser.add_argument("--ball-fixed-dy", type=float, default=0.0)
    parser.add_argument("--ball-udp-port", type=int, default=5599)
    parser.add_argument(
        "--target-dx", type=float, default=None,
        help="Fixed body-frame target. Default (None): 1.6 in classic mode, or auto-derived from "
        "--kick-region-map-path's best-supported cell if that's set.",
    )
    parser.add_argument("--target-dy", type=float, default=None)
    parser.add_argument(
        "--kick-region-map-path", type=str, default=None,
        help="Path (without extension) to a KickRegionMap saved by "
        "high_level_ball_approach_and_kicking/src/hl_kick/map_kick_region_prob.py. If set, the "
        "trigger gates on live P(success) at the ball's current position crossing "
        "--success-probability-threshold instead of distance to a fixed target.",
    )
    parser.add_argument("--success-probability-threshold", type=float, default=0.72)
    parser.add_argument(
        "--region-settle-fallback-radius-m", type=float, default=0.2,
        help="Map mode only. Also settle (and eventually kick) once the robot is within this "
        "body-frame distance of the auto-derived best-cell target, even if that exact cell's "
        "probability is just under --success-probability-threshold. Prevents the robot freezing "
        "parked on the target when the gate cell sits marginally below threshold. Set to 0 to "
        "disable and gate strictly on probability.",
    )
    args = parser.parse_args()

    robot_config = DEFAULTS["g1-29dof-unified-loco-kick"].robot

    if args.ball_source == "fixed":
        logger.warning(
            "Using FixedBallPoseSource -- see ball_pose_source.py's docstring for why this only "
            "smoke-tests the trivial always-triggered case, not real convergence."
        )
        ball_source = FixedBallPoseSource(args.ball_fixed_dx, args.ball_fixed_dy)
    else:
        logger.info(f"Listening for ball pose on UDP port {args.ball_udp_port}")
        ball_source = UdpBallPoseSource(args.ball_udp_port)

    controller = ClassicalApproachAndKickController(
        num_dofs=robot_config.num_joints, rl_rate=args.rl_rate, target_dx=args.target_dx, target_dy=args.target_dy,
        kick_region_map_path=args.kick_region_map_path,
        success_probability_threshold=args.success_probability_threshold,
        region_settle_fallback_radius_m=args.region_settle_fallback_radius_m,
    )
    if args.kick_region_map_path is not None:
        logger.info(
            f"Region-probabilistic trigger active: threshold={args.success_probability_threshold}, "
            f"auto-derived target={controller.target.tolist()}, "
            f"settle-fallback-radius={args.region_settle_fallback_radius_m}m"
        )

    command_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        logger.info(f"🤖 Robot: {robot_config.robot_type}")
        logger.info(f"Connecting to DDS bridge on interface: {args.interface} (read-only: get_low_state only)")
        interface = create_interface(robot_config, args.domain_id, args.interface, use_joystick=False)
        logger.info("✅ Classical HL controller initialized successfully!")
        logger.info(f"Publishing velocity + trigger commands to UDP port {args.command_port} at {args.rl_rate}Hz")

        dt = 1.0 / args.rl_rate
        tick = 0
        while True:
            tick_start = time.perf_counter()

            robot_state_data = interface.get_low_state()
            ball_body = ball_source.get_ball_pos_body_frame()
            out = controller.compute(robot_state_data, ball_body)

            if out.trigger_kick:
                p_str = f", p_success={out.region_probability:.3f}" if out.region_probability is not None else ""
                logger.info(f"Autonomous kick trigger ({out.trigger_reason}{p_str})")
                # compute()'s trigger branch already resets _approach_phase/_hold_step_count;
                # reset() additionally clears _prev_dof_pos, avoiding one bogus finite-differenced
                # dof_vel_norm reading (spanning the whole kick) on the first tick of the next
                # settling phase. This process has no visibility into when the LL side actually
                # returns to locomotion (that's LL-internal state it never reports back) -- but
                # the LL's own task_mode gate prevents any of this process's velocity commands
                # from doing anything until it's back in locomotion mode anyway, so resetting
                # immediately (rather than waiting for a round-trip acknowledgement) is sufficient.
                controller.reset()

            send_hl_command(command_sock, args.command_port, HlCommand(out.vx, out.vy, out.wyaw, out.trigger_kick))

            tick += 1
            if tick % int(args.rl_rate) == 0:
                logger.info(f"HL tick {tick} | vx={out.vx:.3f} vy={out.vy:.3f} wyaw={out.wyaw:.3f}")

            elapsed = time.perf_counter() - tick_start
            if elapsed < dt:
                time.sleep(dt - elapsed)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception:
        logger.error("❌ Error running classical HL controller")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
