#!/usr/bin/env python3
"""
Simulation Runner Script

This script provides a direct simulation runner for holosoma with bridge support without training
or evaluation environments.
"""

import argparse
import dataclasses
import sys
import traceback

import tyro
from loguru import logger

from holosoma.config_types.run_sim import RunSimConfig
from holosoma.config_types.simulator import BallConfig
from holosoma.utils.eval_utils import init_eval_logging
from holosoma.utils.sim_utils import DirectSimulation, setup_simulation_environment
from holosoma.utils.tyro_utils import TYRO_CONIFG


def run_simulation(
    config: RunSimConfig,
    broadcast_ball_udp_port: int | None = None,
    broadcast_ball_target_dx: float = 3.0,
    broadcast_ball_target_dy: float = 0.5,
    broadcast_ball_delay_s: float = 10.0,
    broadcast_ball_random_spawn: bool = False,
    broadcast_ball_dist_min: float = 0.3,
    broadcast_ball_dist_max: float = 4.0,
    broadcast_ball_bearing_max_deg: float = 90.0,
    broadcast_ball_seed: int | None = None,
    auto_release_gantry: bool = False,
    auto_release_gantry_delay_s: float = 2.0,
    auto_release_gantry_duration_s: float = 5.0,
    spawn_real_ball: bool = False,
    ball_radius: float = 0.11,
    ball_mass: float = 0.43,
):
    """Run simulation with direct simulator control.

    This function provides direct access to the simulator for continuous simulation
    with bridge support using the DirectSimulation class.

    Parameters
    ----------
    config : RunSimConfig
        Configuration containing all simulation settings.
    broadcast_ball_udp_port : int | None
        See DirectSimulation.__init__ -- if set, broadcasts a synthetic (not a real ball)
        body-frame target offset over UDP for smoke-testing holosoma_inference's
        ApproachAndKickPolicy / UdpBallPoseSource.
    broadcast_ball_target_dx, broadcast_ball_target_dy : float
        See DirectSimulation.__init__.
    """
    # Auto-set device for GPU-accelerated backends if still on default CPU
    if config.device == "cpu":
        # Check if using Warp backend (requires CUDA)
        if hasattr(config.simulator.config, "mujoco_backend"):
            from holosoma.config_types.simulator import MujocoBackend  # noqa: PLC0415 -- deferred

            if config.simulator.config.mujoco_backend == MujocoBackend.WARP:
                logger.info("Auto-detected MuJoCo Warp backend - setting device to cuda:0")
                config = dataclasses.replace(config, device="cuda:0")

    config = dataclasses.replace(config, device=config.device)

    if spawn_real_ball:
        ball_cfg = BallConfig(radius=ball_radius, mass=ball_mass)
        new_scene = dataclasses.replace(config.simulator.config.scene, ball=ball_cfg)
        new_sim_init_config = dataclasses.replace(config.simulator.config, scene=new_scene)
        config = dataclasses.replace(
            config, simulator=dataclasses.replace(config.simulator, config=new_sim_init_config)
        )
        logger.info(f"Real ball enabled in MuJoCo scene: radius={ball_radius}m, mass={ball_mass}kg")

    logger.info("Starting Holosoma Direct Simulation...")
    logger.info(f"Robot: {config.robot.asset.robot_type}")
    logger.info(f"Simulator: {config.simulator._target_}")
    logger.info(f"Terrain: {config.terrain.terrain_term.mesh_type} ({config.terrain.terrain_term.func})")

    try:
        # Use shared utils for setup
        env, device, simulation_app = setup_simulation_environment(config, device=config.device)

        # Create and run direct simulation using context manager for automatic clean-up
        with DirectSimulation(
            config,
            env,
            device,
            simulation_app,
            broadcast_ball_udp_port=broadcast_ball_udp_port,
            broadcast_ball_target_dx=broadcast_ball_target_dx,
            broadcast_ball_target_dy=broadcast_ball_target_dy,
            broadcast_ball_delay_s=broadcast_ball_delay_s,
            broadcast_ball_random_spawn=broadcast_ball_random_spawn,
            broadcast_ball_dist_min=broadcast_ball_dist_min,
            broadcast_ball_dist_max=broadcast_ball_dist_max,
            broadcast_ball_bearing_max_deg=broadcast_ball_bearing_max_deg,
            broadcast_ball_seed=broadcast_ball_seed,
            auto_release_gantry=auto_release_gantry,
            auto_release_gantry_delay_s=auto_release_gantry_delay_s,
            auto_release_gantry_duration_s=auto_release_gantry_duration_s,
        ) as sim:
            sim.run()

    except Exception as e:
        logger.error(f"Error during simulation: {e}")
        traceback.print_exc()
        sys.exit(1)


def main() -> None:
    """Main function using tyro configuration with compositional subcommands."""
    # Initialize logging
    init_eval_logging()

    logger.info("Holosoma Direct Simulation Runner")
    logger.info("Compositional configuration via subcommands (like eval_agent.py)")

    # Pre-parse the (optional) ground-truth ball-pose broadcaster flags before handing the rest
    # of argv to tyro -- same split pattern as holosoma_inference's
    # run_approach_and_kick_policy.py, kept out of RunSimConfig itself since it's a smoke-test
    # add-on, not a real simulation setting.
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--broadcast-ball-udp-port", type=int, default=None)
    pre.add_argument("--broadcast-ball-target-dx", type=float, default=3.0)
    pre.add_argument("--broadcast-ball-target-dy", type=float, default=0.5)
    pre.add_argument("--broadcast-ball-delay-s", type=float, default=10.0)
    pre.add_argument("--broadcast-ball-random-spawn", action="store_true")
    pre.add_argument("--broadcast-ball-dist-min", type=float, default=0.3)
    pre.add_argument("--broadcast-ball-dist-max", type=float, default=4.0)
    pre.add_argument("--broadcast-ball-bearing-max-deg", type=float, default=90.0)
    pre.add_argument("--broadcast-ball-seed", type=int, default=None)
    pre.add_argument(
        "--auto-release-gantry", action="store_true",
        help="Ramp then disable the virtual gantry programmatically instead of via keypresses "
        "(8/9), which only work through a GLFW window and never fire in headless mode -- without "
        "this, headless runs have no way to release the gantry at all. See "
        "DirectSimulation.__init__'s docstring.",
    )
    pre.add_argument("--auto-release-gantry-delay-s", type=float, default=2.0)
    pre.add_argument("--auto-release-gantry-duration-s", type=float, default=5.0)
    pre.add_argument(
        "--spawn-real-ball", action="store_true",
        help="Inject a real, visible, physically-simulated ball into the MuJoCo scene (a "
        "freejointed sphere + a supplementary foot collision volume for reliable kick contact -- "
        "see MujocoSceneManager.add_ball()/_add_foot_kick_volume()'s docstrings). Independent of "
        "--broadcast-ball-udp-port: with that flag also set, the ball is teleported to the "
        "broadcaster's randomly-sampled spawn point once armed, and the UDP broadcast switches "
        "from a synthetic canned target to the ball's real, live position.",
    )
    pre.add_argument("--ball-radius", type=float, default=0.11)
    pre.add_argument("--ball-mass", type=float, default=0.43)
    known, remaining = pre.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    # Parse configuration with tyro - same pattern as ExperimentConfig
    config = tyro.cli(
        RunSimConfig,
        description="Run simulation with direct simulator control and bridge support.\n\n"
        "Usage: python -m holosoma.run_sim simulator:<sim> robot:<robot> terrain:<terrain>\n"
        "Examples:\n"
        "  python -m holosoma.run_sim # defaults \n"
        "  python -m holosoma.run_sim simulator:mujoco robot:t1_29dof_waist_wrist terrain:terrain_locomotion_plane\n"
        "  python -m holosoma.run_sim simulator:isaacgym robot:g1_29dof terrain:terrain_locomotion_mix",
        config=TYRO_CONIFG,
    )

    # Run simulation directly with parsed config
    run_simulation(
        config,
        broadcast_ball_udp_port=known.broadcast_ball_udp_port,
        broadcast_ball_target_dx=known.broadcast_ball_target_dx,
        broadcast_ball_target_dy=known.broadcast_ball_target_dy,
        broadcast_ball_delay_s=known.broadcast_ball_delay_s,
        broadcast_ball_random_spawn=known.broadcast_ball_random_spawn,
        broadcast_ball_dist_min=known.broadcast_ball_dist_min,
        broadcast_ball_dist_max=known.broadcast_ball_dist_max,
        broadcast_ball_bearing_max_deg=known.broadcast_ball_bearing_max_deg,
        broadcast_ball_seed=known.broadcast_ball_seed,
        auto_release_gantry=known.auto_release_gantry,
        auto_release_gantry_delay_s=known.auto_release_gantry_delay_s,
        auto_release_gantry_duration_s=known.auto_release_gantry_duration_s,
        spawn_real_ball=known.spawn_real_ball,
        ball_radius=known.ball_radius,
        ball_mass=known.ball_mass,
    )


if __name__ == "__main__":
    main()
