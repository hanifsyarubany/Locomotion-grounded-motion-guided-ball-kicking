"""Task configuration types for holosoma_inference."""

from __future__ import annotations

from typing import Literal

from pydantic.dataclasses import dataclass

InputSource = Literal["keyboard", "interface", "joystick", "ros2", "injected"]

DEFAULT_VELOCITY_INPUT: InputSource = "keyboard"
DEFAULT_STATE_INPUT: InputSource = "keyboard"


@dataclass(frozen=True)
class DebugConfig:
    """Debug overrides for quick testing."""

    force_upright_imu: bool = False
    """Override projected_gravity with [0, 0, -1] (perfectly upright)."""

    force_zero_angular_velocity: bool = False
    """Override base_ang_vel with [0, 0, 0]."""

    force_zero_action: bool = False
    """Zero out the scaled policy action (robot holds default pose)."""


@dataclass(frozen=True)
class Ros2DepthConsumerConfig:
    """Config for the ROS2 depth-image consumer (``Ros2DepthConsumer``).

    Defaults match the on-robot image_server preprocessing the policy was
    trained against. Leave ``topics`` empty to disable the depth sensor.
    """

    topics: tuple[str, ...] = ()
    """Raw depth topic(s) (``sensor_msgs/Image``, encoding ``32FC1``, metric
    meters). Empty disables the sensor. One per camera, in stack order (front
    first, back second). Multi-camera frames are time-synchronized; the policy
    reads a preprocessed ``(N, 1, resized_height, resized_width)`` stack via
    ``self._injected_sensors["depth"].get_latest()``."""

    resized_height: int = 27
    """Target height (bicubic resize) before clip+normalize."""

    resized_width: int = 48
    """Target width (bicubic resize) before clip+normalize."""

    near_clip: float = 0.1
    """Near clip (m); depth is normalized to [-0.5, 0.5] over [near, far]."""

    far_clip: float = 2.0
    """Far clip (m); depth is normalized to [-0.5, 0.5] over [near, far]."""

    frame_delay_ms: float = 0.0
    """Modeled depth latency to re-introduce, in absolute milliseconds.

    The ROS2 depth transport is effectively instantaneous (sub-1ms), but the
    policy was trained with the on-robot image_server's inherent capture/serve
    latency baked in. ``get_latest()`` holds back frames so it returns the
    freshest frame at least this old, reproducing that delay independent of
    publish rate (a fixed ms value is robust across fps, unlike a frame count).
    ``0.0`` (default) keeps freshest-frame behavior. Pin this per-policy from a
    preset to match the delay the model was trained with (e.g. ``200.0``)."""


@dataclass(frozen=True)
class TaskConfig:
    """Task execution configuration for policy inference."""

    model_path: str | list[str]
    """Path to ONNX model(s). Supports local paths and wandb:// URIs. Required field."""

    rl_rate: float = 50
    """Policy inference rate in Hz."""

    policy_action_scale: float = 0.25
    """Scaling factor applied to policy actions."""

    action_scales_by_effort_limit_over_p_gain: bool = False
    """Use per-joint scaling: ``policy_action_scale * effort_limit / p_gain``."""

    use_phase: bool = True
    """Whether to use gait phase observations."""

    gait_period: float = 1.0
    """Gait cycle period in seconds."""

    skip_stiff_prompt: bool = False
    """WBT: skip the blocking stdin prompt before entering stiff hold and enter
    immediately. Default False keeps the interactive 'Press Enter to continue'
    safety pause; non-interactive launches (e.g. the ROS2 service node) set True."""

    domain_id: int = 0
    """DDS domain ID for communication."""

    interface: str = "auto"
    """Network interface name. Use ``"auto"`` to auto-detect, or specify explicitly (e.g. ``"eth0"``)."""

    depth: Ros2DepthConsumerConfig = Ros2DepthConsumerConfig()
    """Depth-image consumer config (empty ``topics`` disables it)."""

    velocity_input: InputSource = DEFAULT_VELOCITY_INPUT
    """Source for velocity commands."""

    state_input: InputSource = DEFAULT_STATE_INPUT
    """Source for non-velocity inputs (start/stop, walk/stand, tuning)."""

    use_keyboard: bool = False
    """Shortcut: set both velocity_input and state_input to "keyboard".

    Cannot be combined with explicit input settings.
    """

    use_joystick: bool = False
    """Shortcut: set both velocity_input and state_input to "interface".

    Reads from the SDK's wireless controller (the dongle/controller shipped
    with Unitree G1, Booster T1, etc.). For host-side USB gamepads
    (Xbox/Logitech via /dev/input/event*), use ``use_usb_joystick`` instead.

    Cannot be combined with explicit input settings.
    """

    use_usb_joystick: bool = False
    """Shortcut: set both velocity_input and state_input to "joystick".

    Reads a USB gamepad on the host via evdev (``/dev/input/event*``).
    Linux-only. Cannot be combined with explicit input settings.
    """

    joystick_type: str = "xbox"
    """Joystick type."""

    joystick_device: int = 0
    """Joystick device index."""

    ros_cmd_vel_topic: str = "cmd_vel"
    """ROS2 topic name for velocity commands (used when velocity_input is "ros2")."""

    ros_state_input_topic: str = "holosoma/state_input"
    """ROS2 topic name for discrete commands (used when state_input is "ros2")."""

    ros_vel_timeout: float = 1.0
    """Seconds without a velocity message before zeroing commands. Set to 0 to disable."""

    auto_walk_on_vel_cmd: bool = False
    """Automatically enter walking mode when a non-zero velocity command is received."""

    use_sim_time: bool = False
    """Use synchronized simulation time for WBT policies."""

    # Deprecation candidates:
    desired_base_height: float = 0.75
    """Target base height in meters."""

    residual_upper_body_action: bool = False
    """Whether to use residual control for upper body."""

    print_observations: bool = False
    """Print observation vectors for debugging."""

    motion_start_timestep: int = 0
    """Starting timestep for motion clip playback."""

    motion_end_timestep: int | None = None
    """Ending timestep for motion clip playback. If None, plays until the end."""

    debug: DebugConfig = DebugConfig()
    """Debug overrides for quick testing."""

    use_previous_version_policy: bool = False
    """UnifiedPolicy (and its subclasses NetworkControlledUnifiedPolicy, ApproachAndKickPolicy)
    only: deploy a checkpoint trained before training v7 (RoboNaldo-style shooting rewards), whose
    actor_obs is 256-dim instead of the current 261-dim layout (missing kick_ball_pos_b/
    kick_target_pos_b). Swaps the observation config to
    config_values.observation:unified_g1_29dof_legacy inside UnifiedPolicy.__init__, before
    BasePolicy.__init__ builds the obs machinery from it -- everything else (robot, task, kick
    controls) is unchanged. Handled there rather than in any one script's main() so every entry
    point that constructs one of these policies (run_policy.py, run_network_controlled_policy.py,
    run_approach_and_kick_policy.py, ...) gets the same behavior automatically. Set this whenever
    --task.model-path points at a pre-v7 checkpoint (e.g.
    locomotion-policy-v7-kick-locomotion/model_0118000.onnx); leave it False for any checkpoint
    trained since (the ONNX raises a shape-mismatch error immediately if this is set wrong, so a
    misconfiguration can't silently produce garbled observations)."""

    kick_ball_source: Literal["none", "fixed", "udp"] = "none"
    """UnifiedPolicy only: source for the ball position fed into the kick_ball_pos_b observation
    while task_mode=="kick". "none" (default) leaves it zeroed -- correct behavior whenever
    task_mode isn't "kick" regardless of this setting (matches training's task-mode obs masking),
    and a safe default for locomotion-only checkpoints or any run that never triggers a kick.
    "fixed" uses a constant body-frame offset (kick_ball_fixed_dx/dy) -- smoke-testing only, see
    ball_pose_source.FixedBallPoseSource's docstring for why this isn't a genuine fixed-in-world
    point (the offset moves WITH the robot). "udp" reads live broadcasts on kick_ball_udp_port,
    same wire format as ball_pose_source.UdpBallPoseSource (e.g. from
    run_classical_hl_controller.py or a real perception pipeline)."""

    kick_ball_fixed_dx: float = 2.84
    kick_ball_fixed_dy: float = -0.46
    """Body-frame ball offset used when kick_ball_source == "fixed" (defaults match
    configs/ball.yaml's nominal spawn, in the training workspace)."""

    kick_ball_udp_port: int = 5599
    """UDP port for kick_ball_source == "udp"."""

    kick_ball_height_m: float = 0.11
    """Assumed ball resting height (meters, matches configs/ball.yaml's default radius) --
    combined with desired_base_height to approximate kick_ball_pos_b's z-component as
    kick_ball_height_m - desired_base_height. A real approximation, not a measurement:
    get_low_state() has no absolute world height on real hardware (see ball_pose_source.py's
    docstring for why), so the true robot-root-to-ball height can't be computed directly. Only
    matters once a checkpoint trained with real shooting-adaptation rewards is deployed; a
    locomotion-only (Stage A/B) checkpoint never reaches task_mode=="kick" so never reads this."""

    kick_target_dx: float = 7.84
    kick_target_dy: float = -0.46
    """UnifiedPolicy only: fixed body-frame shot target fed into kick_target_pos_b while
    task_mode=="kick" (defaults match configs/ball.yaml's nominal target, in the training
    workspace). Same body-frame-offset caveat as kick_ball_fixed_dx/dy -- moves with the robot
    rather than being genuinely world-fixed. Adequate for smoke-testing a shooting-adaptation
    (Stage C) checkpoint's aim; a real deployment wanting a world-anchored, live-commandable
    target needs a dedicated target source, not yet implemented here.

    IGNORED when kick_aim_enabled=True below -- that mode replaces this pair entirely with
    kick_aim_theta_deg/kick_aim_theta_ref_deg."""

    kick_aim_enabled: bool = False
    """2026-08-22, azimuth-aim refactor (deployment mirror of SkillConfig.kick_aim_enabled --
    see that field's own docstring for the full training-side mechanism). False (default): the
    kick_target_dx/dy pair above is used exactly as before this field existed. True: the
    kick_target_pos_b observation slot instead carries a normalized commanded angle
    (kick_aim_theta_deg / kick_aim_theta_ref_deg, dim 1 reserved at 0.0) -- a GENUINE constant
    matching what training holds constant for the whole attempt, unlike kick_target_dx/dy's own
    smoke-test-only approximation of a value training actually varies continuously. Only set this
    True when deploying a checkpoint actually TRAINED with kick_aim_enabled=True -- the observation
    means something different in the two modes, and deploying the wrong mode against a given
    checkpoint is out-of-distribution the same way an unset kick_ball_source is (see that field's
    own docstring)."""

    kick_aim_theta_deg: float = 0.0
    """The commanded strike direction (degrees), only read when kick_aim_enabled=True. 0.0 means
    "this skill's own calibrated nominal direction" -- see SkillConfig.resolved_nominal_bearing_deg
    ()'s own docstring for what that calibration measures and why an uncalibrated nominal silently
    biases every command. Must be within the range the deployed checkpoint was actually trained on
    (MultiSkillConfig/BallConfig.kick_aim_theta_max_deg) -- commanding well outside that range is
    out-of-distribution for the same reason any RL policy input is when pushed past its training
    envelope."""

    kick_aim_theta_ref_deg: float = 45.0
    """Fixed normalization reference (degrees) for kick_aim_theta_deg, only read when
    kick_aim_enabled=True. MUST match the value the deployed checkpoint was actually trained with
    (MultiSkillConfig/BallConfig.kick_aim_theta_ref_deg) -- this is a normalization constant baked
    into the checkpoint's learned input scale, not a free parameter to tune at deployment time.
    Mismatching it silently rescales every commanded angle without any error or warning."""

    def __post_init__(self):
        """Resolve use_keyboard/use_joystick/use_usb_joystick shortcuts into velocity_input/state_input."""
        active_shortcuts = [
            name
            for name, enabled in (
                ("keyboard", self.use_keyboard),
                ("joystick", self.use_joystick),
                ("usb-joystick", self.use_usb_joystick),
            )
            if enabled
        ]
        if len(active_shortcuts) > 1:
            joined = ", ".join(f"--task.use-{n}" for n in active_shortcuts)
            raise ValueError(
                f"Cannot combine multiple input shortcuts ({joined}). "
                "Use one shortcut or set --task.velocity-input and --task.state-input individually."
            )

        shortcut: InputSource | None = None
        flag_name: str | None = None
        if self.use_usb_joystick:
            shortcut = "joystick"
            flag_name = "usb-joystick"
        elif self.use_joystick:
            shortcut = "interface"
            flag_name = "joystick"
        elif self.use_keyboard:
            shortcut = "keyboard"
            flag_name = "keyboard"

        if shortcut is not None:
            has_custom_input = self.velocity_input != DEFAULT_VELOCITY_INPUT or self.state_input != DEFAULT_STATE_INPUT
            if has_custom_input:
                raise ValueError(
                    f"Cannot combine --task.use-{flag_name} with --task.velocity-input or "
                    "--task.state-input. Use either the shortcut flag or the individual "
                    "input settings, not both."
                )
            object.__setattr__(self, "velocity_input", shortcut)
            object.__setattr__(self, "state_input", shortcut)
