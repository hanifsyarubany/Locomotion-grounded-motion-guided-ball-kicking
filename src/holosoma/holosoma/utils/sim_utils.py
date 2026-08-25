"""Shared simulation utilities for holosoma.

This module provides common functionality for setting up and running simulations,
shared between eval_agent.py and run_sim.py.
"""

from __future__ import annotations

import argparse
import math
import os
import socket
import struct
import sys
import threading
import time
import traceback
from typing import Any

import numpy as np

from loguru import logger
from typing_extensions import Self

from holosoma.config_types.env import get_tyro_env_config
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_types.full_sim import FullSimConfig
from holosoma.config_types.run_sim import RunSimConfig
from holosoma.managers.terrain.manager import TerrainManager
from holosoma.utils.common import seeding
from holosoma.utils.helpers import get_class
from holosoma.utils.rate import RateLimiter
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.simulator_config import SimulatorType, get_simulator_type, set_simulator_type
from holosoma.utils.torch_utils import to_torch


def setup_simulator_imports(config: ExperimentConfig | RunSimConfig) -> None:
    """Setup simulator-specific imports without side effects.

    Parameters
    ----------
    config : ExperimentConfig | RunSimConfig
        Configuration containing simulator settings.
    """
    set_simulator_type(config.simulator)
    simulator_type = get_simulator_type()

    if simulator_type == SimulatorType.MUJOCO:
        import mujoco

        assert mujoco is not None
    elif simulator_type == SimulatorType.ISAACGYM:
        import isaacgym

        assert isaacgym is not None

    # IsaacSim imports handled in setup_isaaclab_launcher


def setup_isaaclab_launcher(config: ExperimentConfig | RunSimConfig, device: str | None = None) -> Any | None:
    """Handle IsaacSim-specific launcher setup.

    Parameters
    ----------
    config : ExperimentConfig | RunSimConfig
        Configuration containing simulator and training settings.
    device : str
        Resolved device string (e.g., 'cuda:0', 'cpu').

    Returns
    -------
    Any | None
        IsaacSim simulation app instance, or None for other simulators.
    """
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Run simulation with IsaacSim.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
    parser.add_argument("--env_spacing", type=int, default=20, help="Distance between environments in simulator.")
    parser.add_argument("--output_dir", type=str, default="logs", help="Directory to store the output.")
    AppLauncher.add_app_launcher_args(parser)

    # Parse known arguments to get argparse params
    args_cli, unknown_args = parser.parse_known_args()

    # Set values from config — divide by world_size for multi-GPU so each rank's
    # AppLauncher only allocates resources for its share of environments.
    # (The full num_envs is divided again in train_agent.train(), but AppLauncher
    # needs the per-rank count at init time to avoid over-allocating GPU memory.)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    args_cli.num_envs = config.training.num_envs // world_size if world_size > 1 else config.training.num_envs
    args_cli.seed = config.training.seed
    args_cli.env_spacing = config.simulator.config.scene.env_spacing
    args_cli.output_dir = config.logger.base_dir
    args_cli.headless = config.training.headless
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        # Distribute simulator across GPUs when using multi-gpu training
        args_cli.device = f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}"
        args_cli.distributed = True
    elif device is not None:
        # Use the resolved device
        args_cli.device = device
    else:  # AppLauncher auto-detects
        pass

    # Check if video recording is enabled and add --enable_cameras flag
    video_enabled = config.logger.video.enabled or config.logger.headless_recording
    if video_enabled:
        args_cli.enable_cameras = True

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    logger.info(f"IsaacSim args_cli: {args_cli}")
    logger.info(f"IsaacSim unknown_args: {unknown_args}")
    sys.argv = [sys.argv[0]] + unknown_args

    return simulation_app


def setup_keyboard_listener(env) -> threading.Thread:
    """Setup keyboard listener thread for simulation control.

    Parameters
    ----------
    env
        Environment instance to control.

    Returns
    -------
    threading.Thread
        Keyboard listener thread (already started).
    """

    def on_press(key, env):
        """Handle keyboard input for simulation control."""
        try:
            if hasattr(key, "char") and key.char:
                if key.char == "n":
                    if hasattr(env, "next_task"):
                        env.next_task()
                        logger.info("Moved to the next task.")
                # Force Control
                elif key.char == "1":
                    if hasattr(env, "apply_force_scale"):
                        env.apply_force_scale /= 2.0
                        logger.info(f"apply_force_scale: {env.apply_force_scale}")
                elif key.char == "2":
                    if hasattr(env, "apply_force_scale"):
                        env.apply_force_scale *= 2.0
                        logger.info(f"apply_force_scale: {env.apply_force_scale}")
        except AttributeError:
            pass

    def listen_for_keypress(env):
        """Listen for keyboard input in a separate thread."""
        try:
            # Delay import so that one can run the rest of this script in headless mode.
            # Trying to import pynput in headless mode gives the following error:
            # ImportError: this platform is not supported:
            # ('failed to acquire X connection: Bad display name ""', DisplayNameError(''))
            from pynput import keyboard as pynput_keyboard

            logger.info("Keyboard controls:")
            logger.info("  n - Next task (if supported)")
            logger.info("  1/2 - Decrease/Increase force scale (if supported)")

            with pynput_keyboard.Listener(on_press=lambda key: on_press(key, env)) as listener:
                listener.join()
        except ImportError:
            logger.warning("pynput not available - keyboard controls disabled")
        except Exception as e:
            logger.warning(f"Keyboard listener failed: {e}")

    key_listener_thread = threading.Thread(target=listen_for_keypress, args=(env,))
    key_listener_thread.daemon = True
    key_listener_thread.start()
    return key_listener_thread


def setup_simulation_environment(
    config: ExperimentConfig | RunSimConfig, device: str | None = None
) -> tuple[Any, str, Any]:
    """Setup simulation environment with shared infrastructure.

    This function handles common setup for training, evaluation and direct simulation:
    - Simulator imports and initialization
    - Device selection and seeding
    - Environment creation
    - Keyboard listener setup (if not headless)

    Parameters
    ----------
    config : ExperimentConfig | RunSimConfig
        Configuration containing all simulation settings.
    device : str | None, optional
        Device to use for simulation. If None, auto-detects CUDA availability.

    Returns
    -------
    tuple[Any, str, Any]
        Tuple of (environment, device_string, simulation_app).
        simulation_app is None for simulators that don't need it (MuJoCo, IsaacGym).
    """
    logger.info("🚀 Setting up simulation environment...")

    # Setup simulator imports
    setup_simulator_imports(config)

    # Device selection - must happen before IsaacSim launcher setup
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Handle IsaacSim launcher if needed (for both ExperimentConfig and RunSimConfig)
    simulation_app = None
    if get_simulator_type() == SimulatorType.ISAACSIM:
        simulation_app = setup_isaaclab_launcher(config, device)

    # Set random seed if specified (only for ExperimentConfig)
    if isinstance(config, ExperimentConfig) and config.training.seed is not None:
        seeding(config.training.seed, torch_deterministic=config.training.torch_deterministic)
        logger.info(f"Seed: {config.training.seed}")

    # For RunSimConfig, we need a different approach since it doesn't have env_class or training configs
    if isinstance(config, RunSimConfig):
        # For run_sim.py, we'll create the simulator directly instead of using environment wrapper
        logger.info("Direct simulation mode - creating simulator directly, without experiment config")

        # Create FullSimConfig from RunSimConfig
        # Extract SimulatorInitConfig from SimulatorConfig
        full_config = FullSimConfig(
            simulator=config.simulator.config,  # Extract .config from SimulatorConfig
            robot=config.robot,
            training=config.training,
            logger=config.logger,
            experiment_dir=None,
        )

        # For compatibility, minimal proxy for TerrainManager since it depends on env
        class EnvProxy:
            def __init__(self, device):
                self.num_envs = 1
                self.device = device

        # For compatibility, wrap in a minimal object that has .sim attribute
        class DirectSimWrapper:
            def __init__(self, simulator):
                self.sim = simulator

            def reset(self):
                # Basic reset - just initialize the simulator if needed
                if hasattr(self.sim, "reset"):
                    self.sim.reset()

            def close(self):
                if hasattr(self.sim, "close"):
                    self.sim.close()

        # Use terrain configuration from RunSimConfig
        terrain_manager = TerrainManager(config.terrain, env=EnvProxy(device), device=device)

        # Create simulator using get_class() to avoid circular imports
        simulator_class = get_class(config.simulator._target_)
        simulator = simulator_class(full_config, terrain_manager, device)

        # Now we have an "env" to return which is actually the direct simulator
        env = DirectSimWrapper(simulator)
        logger.debug("Direct simulator created successfully!")

    else:
        # Original ExperimentConfig path
        env_target = config.env_class
        tyro_env_config = get_tyro_env_config(config)

        logger.info(f"Creating environment: {env_target}")
        env_class = get_class(env_target)
        env = env_class(tyro_env_config, device=device)

        logger.debug("Environment created successfully!")

        # Setup keyboard listener if not headless
        if not config.training.headless:
            setup_keyboard_listener(env)

    return env, device, simulation_app


def close_simulation_app(simulation_app):
    """Close simulation app with workarounds for known issues.

    Parameters
    ----------
    simulation_app : Any
        The simulation app instance returned by init_sim_imports().
        Can be None for simulators that don't have an app (e.g., IsaacGym).
    """
    if simulation_app is not None and get_simulator_type() == SimulatorType.ISAACSIM:
        logger.info("Shutting down simulation app...")
        try:
            # Work-around for IsaacLab hanging headless.
            # Patch the close_stage method to avoid hanging
            import omni.usd

            context = omni.usd.get_context()
            context_class = context.__class__

            # Replace with a no-op version
            def noop_close_stage(self, *args, **kwargs):
                logger.info("Skipping close_stage() to avoid hanging")
                return True

            # Apply the patch
            context_class.close_stage = noop_close_stage
            logger.info("Successfully patched close_stage method")
        except Exception as e:
            logger.warning(f"Could not patch close_stage method: {e}")

        try:
            # Work-around for IsaacLab SimulationContext._app_control_on_stop_handle_fn
            # hanging in an infinite render() loop on shutdown. When simulation_app.close()
            # triggers a timeline STOP event, the callback spins waiting for the timeline to
            # start playing again — which never happens. Disabling the callback prevents this.
            from isaaclab.sim import SimulationContext

            sim_context = SimulationContext.instance()
            if sim_context is not None:
                sim_context._disable_app_control_on_stop_handle = True
                logger.info("Disabled SimulationContext app_control_on_stop_handle to prevent shutdown hang")
        except Exception as e:
            logger.warning(f"Could not disable app_control_on_stop_handle: {e}")

        # Now close the app
        simulation_app.close(wait_for_replicator=False)
        logger.info("Simulation app closed.")
    else:
        logger.info("Simulation app closed.")


class DirectSimulation:
    """Encapsulates direct simulation logic for run_sim.py.

    This class provides a clean interface for running direct simulations without
    training or evaluation environments, handling all initialization,
    loop management, and cleanup logic.

    Can be used as a context manager for resource management.

    Examples
    --------
    >>> with DirectSimulation(config, env, device, simulation_app) as sim:
    ...     sim.run()
    """

    def __init__(
        self,
        config: RunSimConfig,
        env: Any,
        device: str,
        simulation_app: Any,
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
    ):
        """Initialize DirectSimulation instance.

        Parameters
        ----------
        config : RunSimConfig
            Configuration containing all simulation settings.
        env : Any
            Environment wrapper containing the simulator.
        device : str
            Device for tensor operations.
        simulation_app : Any
            Simulation app instance (if any).
        broadcast_ball_udp_port : int | None
            If set, broadcasts a body-frame (dx, dy) offset over UDP each control tick, in the
            wire format consumed by holosoma_inference's UdpBallPoseSource
            (struct.pack('<dd', dx, dy)). Two modes, depending on whether a real ball was also
            requested (`run_sim.py --spawn-real-ball`, which sets `SimulatorInitConfig.scene.ball`
            -- consumed here via `MuJoCo.has_ball`/`add_ball`/`get_ball_position_xy`, previously
            only read by the IsaacSim backend): with a real ball, this broadcasts that ball's
            actual, live physics position every tick (real contacts/kicks move it, and the
            broadcast reflects that); without one, it broadcasts a synthetic world-frame-fixed
            target instead, standing in for perception until a real ball is added -- useful for
            smoke-testing ApproachAndKickPolicy's closed-loop convergence without needing physics
            contact to work correctly yet.
        broadcast_ball_target_dx, broadcast_ball_target_dy : float
            Body-frame offset (meters) from the robot's pose *at broadcast_ball_delay_s seconds
            into the run loop* that defines the (real or synthetic) target's spawn point -- e.g.
            (3.0, 0.5) means "3m forward, 0.5m left of wherever the robot is facing at that
            moment." Deliberately not the robot's pose at simulator-init time: the operator still
            needs to manually drop the gantry (keys 8/9) and let the robot settle first, and that
            settling can reorient the robot significantly -- fixing the target before that
            happened was observed to sometimes require a near-180-degree correction, which the
            controller could handle but made for a needlessly punishing smoke test.
        broadcast_ball_delay_s : float
            Seconds after the run loop starts before the target is fixed (see above) and
            broadcasting begins. Give yourself enough time to drop the gantry and see the robot
            settle before this fires.
        broadcast_ball_random_spawn : bool
            If True, ignore broadcast_ball_target_dx/dy and instead draw a random (distance,
            bearing) in polar coordinates around the robot's pose at arm-time -- same
            distribution shape as high_level_ball_approach_and_kicking/configs/ball.yaml's
            `spawn` section (dist_min/dist_max/bearing_max_deg below default-match that file),
            for testing the classical HL controller against genuinely randomized approach
            conditions end-to-end over the real DDS bridge, not just a fixed point.
        broadcast_ball_dist_min, broadcast_ball_dist_max : float
            Spawn distance range (meters), only used if broadcast_ball_random_spawn.
        broadcast_ball_bearing_max_deg : float
            Spawn bearing drawn uniformly from [-this, +this] degrees (0 = directly ahead), only
            used if broadcast_ball_random_spawn.
        broadcast_ball_seed : int | None
            Seeds the random spawn draw for reproducibility. None = unseeded (genuinely random
            each run).
        auto_release_gantry : bool
            If True, programmatically ramp the virtual gantry's rest length up (same effect as
            repeatedly pressing key 8) then disable it (same as key 9), instead of requiring
            interactive keypresses. Necessary in headless mode: gantry keys are wired exclusively
            through GLFW window callbacks (see command_registry.py), which never fire without an
            actual GUI window -- there is no terminal-based fallback, so in headless mode the
            gantry is otherwise IMPOSSIBLE to release at all, silently applying a real, fairly
            stiff (default stiffness=200) spring-damper force back toward its fixed anchor point
            for the entire run. This produces a specific, easy-to-misdiagnose symptom: the robot
            can rotate roughly freely (the force is purely positional, not rotational) but can't
            translate away from the anchor point, while otherwise looking stable (upright, not
            falling) -- confirmed directly by reproducing it with plain keyboard-driven
            UnifiedPolicy, zero custom code involved.
        auto_release_gantry_delay_s : float
            Seconds after the run loop starts before the release ramp begins (give the simulator
            a moment to settle after spawn).
        auto_release_gantry_duration_s : float
            Seconds over which the rest length ramps from 0 to the gantry's configured height
            (so the spring force is already near-zero by the time it's disabled, avoiding a
            sudden jerk/drop) before being fully disabled.
        """
        self.config = config
        self.env = env
        self.device = device
        self.simulation_app = simulation_app
        self.simulator = env.sim
        self.broadcast_ball_udp_port = broadcast_ball_udp_port
        self.broadcast_ball_target_dx = broadcast_ball_target_dx
        self.broadcast_ball_target_dy = broadcast_ball_target_dy
        self.broadcast_ball_delay_s = broadcast_ball_delay_s
        self.broadcast_ball_random_spawn = broadcast_ball_random_spawn
        self.broadcast_ball_dist_min = broadcast_ball_dist_min
        self.broadcast_ball_dist_max = broadcast_ball_dist_max
        self.broadcast_ball_bearing_max_deg = broadcast_ball_bearing_max_deg
        self.broadcast_ball_seed = broadcast_ball_seed
        self.auto_release_gantry = auto_release_gantry
        self.auto_release_gantry_delay_s = auto_release_gantry_delay_s
        self.auto_release_gantry_duration_s = auto_release_gantry_duration_s
        self._gantry_release_done = False
        self._broadcast_sock: socket.socket | None = None
        self._broadcast_target_world_xy: tuple[float, float] | None = None
        self._broadcast_armed = False
        self._ball_rng: np.random.Generator | None = None
        self._rerandomize_ball_requested = False

    def __enter__(self) -> Self:
        """Context manager entry - initialize the simulation.

        Returns
        -------
        Self
            Self for use in the with statement.
        """
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - cleanup the simulation.

        Parameters
        ----------
        exc_type : type or None
            Exception type if an exception occurred.
        exc_val : Exception or None
            Exception instance if an exception occurred.
        exc_tb : traceback or None
            Traceback if an exception occurred.
        """
        self.cleanup()

    def initialize(self) -> None:
        """Handle the complete simulator initialization sequence.

        Performs the initialization process required for proper simulator
        lifecycle management. Ideally this is moved into the simulator interface and
        to simplify training, evaluation and direct usage.
        """
        logger.debug("Initializing simulator...")

        # Need to manually set headless since it's in training config currently
        self.simulator.set_headless(False)

        # Step 1: Basic setup
        self.simulator.setup()
        logger.debug("simulator.setup() completed")

        # Step 2: Setup terrain
        self.simulator.setup_terrain()
        logger.debug("simulator.setup_terrain() completed")

        # Step 3: Load assets (this initializes the bridge!)
        self.simulator.load_assets()
        logger.debug("simulator.load_assets() completed - bridge should now be initialized")

        # Step 4: Create environments (need to provide required parameters)
        # Create env_origins (single environment at origin)
        env_origins = torch.zeros(1, 3, device=self.device)

        # Create base_init_state from robot config
        base_init_state = self._create_base_init_state()

        self.simulator.create_envs(1, env_origins, base_init_state)
        logger.debug("simulator.create_envs() completed")

        # Step 5: Prepare simulation
        self.simulator.prepare_sim()
        logger.debug("simulator.prepare_sim() completed")

        # Step 5.5: Initialize episode (positions virtual gantry, etc.)
        self.simulator.on_episode_start(env_id=0)
        logger.debug("simulator.on_episode_start() completed")

        # Step 6: Setup viewer if not headless
        if not self.config.training.headless:
            self.simulator.setup_viewer()
            logger.debug("simulator.setup_viewer() completed")

        logger.info("Simulator initialized")

        if self.broadcast_ball_udp_port is not None:
            self._setup_ball_broadcast()

        # Step 7: Toggle start recording if enabled
        if self.simulator.video_recorder and self.simulator.video_recorder.enabled:
            # arbitrary episode ID given this is sim2sim, we may want to
            # actually support toggling recording and with better filenames too
            self.simulator.video_recorder.start_recording(episode_id=0)

    def _robot_world_pose_xy_yaw(self) -> tuple[float, float, float]:
        """Ground-truth robot (x, y, yaw) in world frame, read directly from MuJoCo qpos.

        Only valid for the MuJoCo backend (relies on robot_qpos_addr / root_data.qpos, which are
        MuJoCo-specific) -- _setup_ball_broadcast only wires this in for that backend.
        """
        addr = self.simulator.robot_qpos_addr
        qpos = self.simulator.root_data.qpos
        x, y = float(qpos[addr]), float(qpos[addr + 1])
        w, qx, qy, qz = (float(qpos[addr + 3 + i]) for i in range(4))
        yaw = math.atan2(2.0 * (w * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        return x, y, yaw

    def _setup_ball_broadcast(self) -> None:
        """Open the UDP socket used to broadcast the ball pose, seed the persistent RNG used for
        random spawns, and start the stdin listener that lets the operator re-randomize the ball
        by typing into this terminal. Does NOT place/fix anything yet -- that happens in
        _arm_ball_broadcast_target() once broadcast_ball_delay_s has elapsed, giving the operator
        time to drop the gantry and let the robot settle (or immediately, via the stdin trigger --
        see _start_ball_rerandomize_listener). Whether this ends up broadcasting a real,
        physically-simulated ball's live position or a synthetic canned target depends on whether
        --spawn-real-ball was also passed (see __init__'s docstring and _arm_ball_broadcast_target).
        """
        self._broadcast_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # One persistent generator, not a fresh np.random.default_rng(seed) per call -- with a
        # fixed seed, recreating it every call would replay the exact same first draw every time,
        # defeating the point of a re-randomize trigger (every subsequent draw would be identical).
        self._ball_rng = np.random.default_rng(self.broadcast_ball_seed)
        real_ball = getattr(self.simulator, "has_ball", False)
        logger.info(
            f"Ball-pose UDP broadcast ready on port {self.broadcast_ball_udp_port} -- "
            f"{'a real ball will be teleported into place' if real_ball else 'a synthetic target will be fixed'} "
            f"{self.broadcast_ball_delay_s:.1f}s into the run loop, "
            f"({self.broadcast_ball_target_dx:.3f}, {self.broadcast_ball_target_dy:.3f}) body-frame "
            f"from the robot's pose at that moment (or a random draw if --broadcast-ball-random-spawn)."
        )
        self._start_ball_rerandomize_listener()

    def _start_ball_rerandomize_listener(self) -> None:
        """Read stdin in a background thread so the operator can re-randomize/re-place the ball at
        any time by typing into this terminal (Enter is enough, content is ignored) -- works over
        plain tmux/ssh regardless of headless mode, unlike the gantry's GLFW-only keybindings
        (8/9 only ever fire from an actual GLFW window -- see auto_release_gantry's docstring)."""

        def _listen() -> None:
            logger.info("Press Enter in this terminal at any time to re-randomize the ball's position.")
            for _ in sys.stdin:
                self._rerandomize_ball_requested = True

        thread = threading.Thread(target=_listen, daemon=True)
        thread.start()

    def _arm_ball_broadcast_target(self) -> None:
        x0, y0, yaw0 = self._robot_world_pose_xy_yaw()
        cos_yaw, sin_yaw = math.cos(yaw0), math.sin(yaw0)

        if self.broadcast_ball_random_spawn:
            assert self._ball_rng is not None
            dist = self._ball_rng.uniform(self.broadcast_ball_dist_min, self.broadcast_ball_dist_max)
            bearing = math.radians(
                self._ball_rng.uniform(-self.broadcast_ball_bearing_max_deg, self.broadcast_ball_bearing_max_deg)
            )
            target_dx = dist * math.cos(bearing)
            target_dy = dist * math.sin(bearing)
            logger.info(
                f"Random ball spawn: dist={dist:.3f}m bearing={math.degrees(bearing):.1f}deg "
                f"(body-frame dx={target_dx:.3f}, dy={target_dy:.3f})"
            )
        else:
            target_dx, target_dy = self.broadcast_ball_target_dx, self.broadcast_ball_target_dy

        target_x = x0 + cos_yaw * target_dx - sin_yaw * target_dy
        target_y = y0 + sin_yaw * target_dx + cos_yaw * target_dy
        self._broadcast_target_world_xy = (target_x, target_y)
        self._broadcast_armed = True

        if getattr(self.simulator, "has_ball", False):
            # A real ball exists in the scene -- teleport it to the sampled spawn point once
            # (initial placement only). From here on it's a genuine physics body: kicks/contacts
            # move it for real, and _broadcast_ball_pose reads its live position, not this target.
            self.simulator.set_ball_position(target_x, target_y)
            logger.info(f"Real ball teleported to world ({target_x:.3f}, {target_y:.3f}).")
        else:
            logger.info(
                f"Ball-pose target fixed at world ({target_x:.3f}, {target_y:.3f}) -- broadcasting "
                "now. NOT a real ball (no --spawn-real-ball) -- synthetic body-frame offset only."
            )

    def _broadcast_ball_pose(self) -> None:
        assert self._broadcast_sock is not None
        x, y, yaw = self._robot_world_pose_xy_yaw()
        if getattr(self.simulator, "has_ball", False):
            # Real ball: read its live position every tick, so the broadcast reflects actual
            # physics (rolling, kicks) instead of a frozen target.
            target_x, target_y = self.simulator.get_ball_position_xy()
        else:
            assert self._broadcast_target_world_xy is not None
            target_x, target_y = self._broadcast_target_world_xy
        dx_world, dy_world = target_x - x, target_y - y
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        dx_body = cos_yaw * dx_world + sin_yaw * dy_world
        dy_body = -sin_yaw * dx_world + cos_yaw * dy_world
        self._broadcast_sock.sendto(struct.pack("<dd", dx_body, dy_body), ("127.0.0.1", self.broadcast_ball_udp_port))

    def _update_gantry_auto_release(self, elapsed_s: float) -> None:
        """See __init__'s auto_release_gantry docstring for why this exists. Ramps
        virtual_gantry.length up (driving the spring force toward zero) over
        auto_release_gantry_duration_s, then holds it there -- replicating repeated key-8
        presses, programmatically, but deliberately NOT a final key-9 full disable (see below).

        Adaptive, not a blind time-based schedule: `length` is capped at the robot's CURRENT
        actual distance from the anchor point (never allowed to exceed it), so the tether can
        only go as slack as the robot has already (safely) descended. A naive linear time-based
        ramp can outpace how fast the robot actually settles -- once `length` exceeds the current
        distance, the spring term `stiffness * (distance - length)` goes negative, which reverses
        the force direction from supportive (pulling toward the anchor) to actively driving the
        robot AWAY from the anchor, i.e. down into the ground -- confirmed directly: an earlier
        version of this ramp caused exactly that (height collapsing 0.79m->0.17m in 3 real
        seconds, far faster than a controlled descent, then falling over on landing).

        Deliberately never calls gantry.set_enable(False): a hard cutoff at the end of the ramp
        was ALSO confirmed to cause falls, even with the policy active for 90+ seconds beforehand
        -- the robot can be transiently stable while still adaptively capped (small residual
        force), then fall within ~4 seconds of the instant that residual is fully removed. Once
        `length` tracks within margin of the robot's actual resting distance, the residual force
        (stiffness * margin, ~10N at the default 0.05m margin) is small enough to be a light,
        permanent safety net rather than a meaningful assist, which is an acceptable tradeoff for
        testing the HL/LL control architecture itself -- not a claim that this matches how a real
        robot without any tether would behave."""
        gantry = getattr(self.simulator, "virtual_gantry", None)
        if gantry is None:
            return
        if elapsed_s < self.auto_release_gantry_delay_s:
            return

        # Deliberately keeps updating forever, even after the ramp completes (never early-returns
        # once "done") -- length must keep adaptively tracking the robot's CURRENT distance for
        # as long as the gantry stays enabled, or it'd freeze at whatever value it last had and
        # start resisting again the moment the robot walks further away than that.
        ramp_elapsed = elapsed_s - self.auto_release_gantry_delay_s
        if not self._gantry_release_done and ramp_elapsed >= self.auto_release_gantry_duration_s:
            self._gantry_release_done = True
            logger.info("Virtual gantry auto-release ramp complete (held as a light permanent safety net)")

        addr = self.simulator.robot_qpos_addr
        pos = np.asarray(self.simulator.root_data.qpos[addr : addr + 3])
        current_distance = float(np.linalg.norm(gantry.point - pos))
        time_based_target = gantry.height * min(1.0, ramp_elapsed / self.auto_release_gantry_duration_s)
        gantry.length = min(time_based_target, max(0.0, current_distance - 0.05))

    def run(self) -> None:
        """Run the direct simulation loop with viewer sync and FPS logging.

        Manages the complete simulation loop including rate limiting,
        viewer synchronization, FPS logging, and error handling.
        """
        # Setup rate limiting
        sim_frequency = self.config.simulator.config.sim.fps
        rate_limiter = RateLimiter(sim_frequency)

        # Calculate viewer sync frequency
        viewer_steps = self._calculate_viewer_steps()
        broadcast_steps = max(1, int(sim_frequency / 50)) if self.broadcast_ball_udp_port is not None else None
        broadcast_arm_step = (
            int(self.broadcast_ball_delay_s * sim_frequency) if self.broadcast_ball_udp_port is not None else None
        )

        logger.info(f"Simulation rate: {sim_frequency} Hz ({1.0 / sim_frequency * 1000:.2f} ms)")
        logger.info(f"Viewer rate: {1 / self.config.viewer_dt:.1f} Hz (sync every {viewer_steps} steps)")
        logger.info("Starting direct simulation loop...")
        logger.info("Press Ctrl+C to stop simulation")

        # Determine refresh strategy based on simulator type
        # IsaacGym/IsaacSim: need pre-step to refresh tensors to sync simulator state
        # MuJoCo: no pre-step refresh needed because we are NOT running an envs/tasks requiring
        #         those tensors e.g, _rigid_body_rot, _rigid_body_vel, etc.
        simulator_type = get_simulator_type()
        if simulator_type in [SimulatorType.ISAACGYM, SimulatorType.ISAACSIM]:
            pre_step_refresh = self.simulator.refresh_sim_tensors
        else:
            pre_step_refresh = lambda: None  # noqa: E731  (No-op for MuJoCo)

        # Direct simulation loop (like holosoma_inference's simulation_thread)
        step_count = 0
        start_time = time.time()
        fps_start_time = start_time

        while True:
            try:
                # Refresh tensors if needed (no-op for MuJoCo)
                pre_step_refresh()

                # Direct simulator step - this triggers bridge.step() inside simulate_at_each_physics_step()
                self.simulator.simulate_at_each_physics_step()

                if self.auto_release_gantry:
                    self._update_gantry_auto_release(step_count / sim_frequency)

                # Update viewer at display rate
                if step_count % viewer_steps == 0:
                    self.simulator.render()

                if broadcast_arm_step is not None and not self._broadcast_armed and step_count >= broadcast_arm_step:
                    self._arm_ball_broadcast_target()
                if self._rerandomize_ball_requested:
                    self._rerandomize_ball_requested = False
                    self._arm_ball_broadcast_target()
                if broadcast_steps is not None and self._broadcast_armed and step_count % broadcast_steps == 0:
                    self._broadcast_ball_pose()

                # Periodic FPS logging (every 1000 steps)
                if step_count > 0 and step_count % 1000 == 0:
                    fps_start_time = self._log_fps(step_count, fps_start_time)

                # TEMPORARY diagnostic (not a permanent feature): world pose + height/tilt, to
                # directly see whether the robot is actually moving/falling during the parallel
                # HL/LL architecture smoke test.
                if step_count > 0 and step_count % 1000 == 0:
                    import math as _math  # noqa: PLC0415

                    x, y, yaw = self._robot_world_pose_xy_yaw()
                    addr = self.simulator.robot_qpos_addr
                    qpos = self.simulator.root_data.qpos
                    z = float(qpos[addr + 2])
                    qw, qx, qy = float(qpos[addr + 3]), float(qpos[addr + 4]), float(qpos[addr + 5])
                    tilt = _math.degrees(_math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (qx * qx + qy * qy)))))
                    logger.info(
                        f"[diag] world=({x:.3f},{y:.3f}) yaw={_math.degrees(yaw):.1f}deg "
                        f"height={z:.3f}m tilt={tilt:.1f}deg"
                    )

                step_count += 1
                rate_limiter.sleep()

            except KeyboardInterrupt:  # noqa: PERF203
                logger.info("Simulation interrupted by user (Ctrl+C)")
                break
            except Exception as e:
                logger.error(f"Error during simulation step {step_count}: {e}")
                traceback.print_exc()
                break

        # Final statistics
        total_elapsed = time.time() - start_time
        avg_fps = step_count / total_elapsed if total_elapsed > 0 else 0
        logger.info(f"Simulation completed after {step_count} steps")
        logger.info(f"Average FPS: {avg_fps:.1f} (target: {sim_frequency})")

    def cleanup(self) -> None:
        """Handle simulation cleanup."""
        # Cleanup environment
        if hasattr(self.env, "close"):
            self.env.close()

        if self.simulator.video_recorder:
            self.simulator.video_recorder.cleanup()

        # Cleanup simulation app
        if self.simulation_app:
            close_simulation_app(self.simulation_app)

    def _create_base_init_state(self) -> torch.Tensor:
        """Create base initialization state tensor from robot configuration.

        Returns
        -------
        torch.Tensor
            Base initialization state tensor.
        """
        base_init_state_list = (
            self.config.robot.init_state.pos
            + self.config.robot.init_state.rot
            + self.config.robot.init_state.lin_vel
            + self.config.robot.init_state.ang_vel
        )
        return to_torch(base_init_state_list, device=self.device, requires_grad=False)

    def _calculate_viewer_steps(self) -> int:
        """Calculate viewer synchronization frequency.

        Returns
        -------
        int
            Number of simulation steps between viewer updates.
        """
        viewer_dt = self.config.viewer_dt
        sim_dt = 1.0 / self.config.simulator.config.sim.fps
        return max(1, int(viewer_dt / sim_dt))

    def _log_fps(self, step_count: int, fps_start_time: float) -> float:
        """Log FPS statistics for simulation performance monitoring.

        Parameters
        ----------
        step_count : int
            Current step count.
        fps_start_time : float
            Start time for FPS measurement.

        Returns
        -------
        float
            New start time for next FPS measurement.
        """
        elapsed = time.time() - fps_start_time
        fps = 1000 / elapsed
        logger.info(f"Simulation FPS: {fps:.1f}")
        return time.time()
