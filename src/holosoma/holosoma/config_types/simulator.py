from __future__ import annotations

import math
import os
import sys
from dataclasses import field
from enum import Enum
from pathlib import Path
from typing import Any

import tyro
import yaml
from pydantic import model_validator
from pydantic.dataclasses import dataclass
from typing_extensions import Annotated

from holosoma.config_types.viewer import ViewerConfig


class MujocoBackend(str, Enum):
    """MuJoCo physics backend selection.

    Determines which MuJoCo backend to use for physics simulation.
    """

    CLASSIC = "classic"
    """CPU-based single environment backend."""

    WARP = "warp"
    """GPU-accelerated multi-environment backend."""


@dataclass(frozen=True)
class MujocoWarpConfig:
    """Configuration for MuJoCo Warp backend memory allocation.

    Controls GPU memory allocation for batched parallel simulation.
    Increase these values if you encounter overflow warnings during training.
    """

    nconmax_per_env: int = 96
    """Maximum contacts per environment (default: 96).

    Increase for:
    - Complex terrains with many contact points
    - Robots with numerous collision geometries
    - Multi-contact scenarios (manipulation, climbing)

    Memory scales as: num_envs x nconmax_per_env
    """

    njmax_per_env: int | None = None
    """Maximum constraints per environment (default: auto-calculated).

    If None (default), automatically calculated as: max(nconmax * 6, nv * 4)
    where nv is the model's velocity dimension.

    Constraints include:
    - Contact constraints (friction cones: ~6 per contact)
    - Joint limits
    - Equality constraints

    Only override if you know you need more constraint capacity.
    """


@dataclass(frozen=True)
class ResetManagerConfig:
    """Configuration for the reset event manager."""

    events: list[Any] = field(default_factory=list)
    """List of reset event configurations to be managed."""


@dataclass(frozen=True)
class PhysxConfig:
    """Low-level PhysX solver settings."""

    solver_type: int
    """Solver type identifier passed to PhysX."""

    num_position_iterations: int
    """Number of position iterations per solver step."""

    num_velocity_iterations: int
    """Number of velocity iterations per solver step."""

    num_threads: int = 4
    """Worker thread count used by PhysX."""

    enable_dof_force_sensors: bool = False
    """Whether to enable force sensors on individual DOFs."""

    bounce_threshold_velocity: float = 0.5
    """Velocity threshold below which bounce responses are suppressed."""


@dataclass(frozen=True)
class MujocoXMLFilterCfg:
    """Configuration for filtering MuJoCo MJCF/XML robot files.

    This configuration controls how robot MJCF files are processed and filtered
    when loaded into the MuJoCo simulator. It allows removal of specific elements
    that may conflict with the simulation environment or cause issues.
    """

    enable: bool = False
    """Whether to enable XML filtering."""

    remove_lights: bool = True
    """Whether to remove <light> elements from the MJCF file."""

    remove_ground: bool = True
    """Whether to remove ground/floor/plane geometries from the MJCF file.
    Assumes these are top-level worldbody geoms."""

    ground_names: list[str] = field(default_factory=lambda: ["floor", "ground", "plane"])
    """List of geometry names to identify and remove as ground elements."""


@dataclass(frozen=True)
class SimEngineConfig:
    """Top-level simulation engine settings."""

    fps: int
    """Target simulation frames per second."""

    control_decimation: int
    """Number of physics steps between agent control updates."""

    substeps: int
    """Number of substeps per physics frame."""

    physx: PhysxConfig
    """PhysX solver configuration."""

    render_mode: str = "human"
    """Rendering mode requested from the simulator."""

    render_interval: int = 1
    """Number of physics frames between rendered frames."""

    max_episode_length_s: float = 20.0
    """Maximum episode length in seconds."""


@dataclass(frozen=True)
class IsaacGymPhysicsConfig:
    """Rigid-shape material properties for the IsaacGym simulator.

    Provides 1:1 mapping to IsaacGym RigidShapeProperties for physics simulation.
    See PhysicsConfig for common physics parameter descriptions.
    """

    friction: float = 1.0
    """Static friction coefficient. Defaults to 1.0."""

    rolling_friction: float = 0.0
    """Rolling resistance coefficient. Defaults to 0.0."""

    torsion_friction: float = 0.0
    """Torsion resistance coefficient. Defaults to 0.0."""

    restitution: float = 0.0
    """Bounce coefficient. Defaults to 0.0."""

    compliance: float = 0.0
    """Shape compliance. Defaults to 0.0."""


@dataclass(frozen=True)
class IsaacSimPhysicsConfig:
    """Rigid-body material properties for the IsaacSim simulator.

    Provides 1:1 mapping to IsaacLab RigidBodyMaterialCfg for physics simulation.
    See PhysicsConfig for common physics parameter descriptions.
    """

    static_friction: float = 1.0
    """Static friction coefficient. Defaults to 1.0."""

    dynamic_friction: float = 1.0
    """Dynamic friction coefficient. Defaults to 1.0."""

    restitution: float = 0.0
    """Bounce coefficient. Defaults to 0.0."""

    friction_combine_mode: str = "multiply"
    """Friction combination mode. Options: "multiply", "max", "min", "average". Defaults to "multiply"."""

    restitution_combine_mode: str = "multiply"
    """Restitution combination mode. Options: "multiply", "max", "min", "average". Defaults to "multiply"."""


@dataclass(frozen=True)
class PhysicsConfig:
    """Unified physics configuration shared across simulators.

    Provides a unified interface for physics properties that can be used across
    different simulators, with simulator-specific sections for detailed control.
    """

    # Essential properties (supported by both simulators)
    kinematic_enabled: bool = False
    """Whether to enable kinematic behavior (static vs dynamic). Defaults to False."""

    mass: float | None = None
    """Direct mass override (highest priority). Defaults to None."""

    density: float | None = None
    """Density for mass calculation (medium priority). Defaults to None."""

    # Damping (both simulators support this!)
    linear_damping: float = 0.1
    """Linear velocity damping coefficient. Defaults to 0.1."""

    angular_damping: float = 0.1
    """Angular velocity damping coefficient. Defaults to 0.1."""

    max_linear_velocity: float = 1000.0
    """Maximum linear velocity limit. Defaults to 1000.0."""

    max_angular_velocity: float = 1000.0
    """Maximum angular velocity limit. Defaults to 1000.0."""

    isaacgym: IsaacGymPhysicsConfig | None = None
    """IsaacGym-specific physics configuration. Defaults to None."""

    isaacsim: IsaacSimPhysicsConfig | None = None
    """IsaacSim-specific physics configuration. Defaults to None."""


@dataclass(frozen=True)
class URDFSettings:
    """URDF-specific loader settings."""

    transform_root_link: str | None = None
    """Root link name for applying transforms. Defaults to None."""


@dataclass(frozen=True)
class ObjectPatternConfig:
    """Configuration for object patterns in scene files."""

    physics: PhysicsConfig | None = None
    """Physics configuration to apply to matching objects. Defaults to None."""


@dataclass(frozen=True)
class SceneFileConfig:
    """Individual scene file configuration.

    Configuration for loading scene files (USD/URDF) with transforms, filtering,
    and object-specific settings.
    """

    usd_path: str | None = None
    """Path to USD scene file. Defaults to None."""

    urdf_path: str | None = None
    """Path to URDF scene file. Defaults to None."""

    position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """Position offset [x, y, z]. Defaults to [0.0, 0.0, 0.0]."""

    orientation: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])  # [w,x,y,z]
    """Orientation quaternion [w, x, y, z]. Defaults to [1.0, 0.0, 0.0, 0.0]."""

    scale: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    """Scale factors [x, y, z]. Defaults to [1.0, 1.0, 1.0]."""

    include_patterns: list[str] = field(default_factory=lambda: ["*"])
    """Patterns for including objects. Defaults to ["*"]."""

    exclude_patterns: list[str] = field(default_factory=list)
    """Patterns for excluding objects. Defaults to empty list."""

    object_configs: dict[str, ObjectPatternConfig] | None = None  # FIX: was Any
    """Object-specific configurations by pattern. Defaults to None."""

    asset_root: str | None = None
    """Root directory for resolving relative paths. Defaults to None."""

    urdf_settings: URDFSettings | None = None
    """URDF-specific settings. Defaults to None."""

    @model_validator(mode="after")
    def validate_urdf_transform_requirements(self) -> SceneFileConfig:
        """Require transform_root_link when URDF transform is applied"""
        if self.urdf_path:  # Only validate for URDF files
            has_transform = self.position != [0.0, 0.0, 0.0] or self.orientation != [1.0, 0.0, 0.0, 0.0]

            if has_transform:
                if not self.urdf_settings or not self.urdf_settings.transform_root_link:
                    raise ValueError(
                        f"URDF scene file '{self.urdf_path}' has position/orientation transform "
                        f"but missing required 'urdf_settings.transform_root_link'. "
                        f"Please specify which link to apply the transform to."
                    )
        return self

    def get_asset_path(self, format_type: str, fallback_root: str | None = None) -> str:
        """Get full path to asset file for specified format.

        Parameters
        ----------
        format_type : str
            Asset format type ('usd' or 'urdf').
        fallback_root : str, optional
            Fallback root directory if asset_root is not set.

        Returns
        -------
        str
            Full path to the asset file.

        Raises
        ------
        ValueError
            If the specified format is not configured for this source.
        """
        format_map = {
            "usd": self.usd_path,
            "urdf": self.urdf_path,
        }

        if format_type not in format_map or format_map[format_type] is None:
            raise ValueError(f"Asset format '{format_type}' not configured for this source")

        asset_path = format_map[format_type]
        assert asset_path is not None
        root_path = self.asset_root if self.asset_root is not None else fallback_root

        if not Path(asset_path).is_absolute():
            if not root_path:
                raise ValueError(f"Root path is required for relative path: {asset_path}")
            return str(Path(root_path) / asset_path)

        return asset_path

    def has_format(self, format_type: str, fallback_root: str | None = None) -> bool:
        """Check if format is configured and file exists.

        Parameters
        ----------
        format_type : str
            Asset format type ('usd' or 'urdf').
        fallback_root : str, optional
            Fallback root directory if asset_root is not set.

        Returns
        -------
        bool
            True if format is configured and file exists, False otherwise.
        """
        try:
            full_path = self.get_asset_path(format_type, fallback_root)
            return Path(full_path).exists()
        except ValueError:
            return False


@dataclass(frozen=True)
class RigidObjectConfig:
    """Configuration for individual rigid objects."""

    name: str
    """Name identifier for the rigid object."""

    urdf_path: str | None = None
    """Path to URDF file for the object. Defaults to None."""

    usd_path: str | None = None
    """Path to USD file for the object. Defaults to None."""

    position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """Position [x, y, z] of the object. Defaults to [0.0, 0.0, 0.0]."""

    orientation: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])  # [w,x,y,z]
    """Orientation quaternion [w, x, y, z] of the object. Defaults to [1.0, 0.0, 0.0, 0.0]."""

    physics: PhysicsConfig | None = None  # FIX: was Any
    """Physics configuration for the object. Defaults to None."""


@dataclass(frozen=True)
class BallConfig:
    """Configuration for a procedurally-spawned, freely-simulated ball (e.g. for kicking tasks).

    Unlike `RigidObjectConfig`/`rigid_objects` (which load USD asset files), this spawns a
    plain primitive sphere directly from these parameters — no mesh asset needed. The ball is
    never motion-tracked or teleported per-frame; it's a genuine physics body that only responds
    to gravity, contacts, and collisions.
    """

    radius: float = 0.11
    """Ball radius in meters. Default matches a regulation soccer ball (~22cm diameter)."""

    mass: float = 0.43
    """Ball mass in kg. Default matches a regulation soccer ball."""

    position: tuple[float, float, float] = (1.5, 0.0, 0.11)
    """Nominal spawn position [x, y, z], relative to each environment's own origin.
    Default places the ball 1.5m along local +X (in front of the robot) resting on the ground
    (z == radius). See ``position_randomization`` for per-reset randomization around this point."""

    position_randomization: tuple[float, float] = (0.0, 0.0)
    """Uniform ± randomization half-range [x, y] (meters) applied to ``position`` at every reset
    and clip replay. (0, 0) — the default — reproduces the previous fixed-spawn behavior exactly.
    Note the policy does not observe the ball, so large ranges mostly inject irreducible reward
    noise rather than learnable variation — see configs/ball.yaml's comment."""

    kick_foot: str = "right"
    """Which foot ("left" or "right") the shooting reward terms attribute the kick to — foot-ball
    proximity and contact-orientation rewards are computed for this foot's ankle body only."""

    target: tuple[float, float] = (5.0, 0.0)
    """Nominal goal coordinate [x, y] the shot should reach, in the same env-local ground-plane
    frame as ``position``. The ball-trajectory shooting rewards measure against this point, and the
    policy observes it (kick_target_pos_b). When ``kick_aim_enabled=True``, this point's role
    changes -- see that field's own docstring."""

    kick_aim_enabled: bool = False
    """2026-08-22, azimuth-aim refactor (legacy single-skill mirror of
    ``SkillConfig.kick_aim_enabled`` -- see that field's own docstring for the full mechanism).
    True: ``target`` is used ONLY to derive ``resolved_nominal_bearing_deg()`` -- a calibrated
    direction, NOT an absolute point (see that method's own docstring on why an uncalibrated
    target silently biases every commanded angle). Each attempt then samples a fresh
    ``kick_aim_theta`` (uniform, +/- ``kick_aim_theta_max_deg``) and synthesizes the real target at
    ``kick_aim_nominal_distance_m`` along (nominal_bearing_deg + kick_aim_theta) from the ball's
    own actual placed position.

    False (default): ``target`` is read as a FIXED per-attempt point -- no randomization at all.
    The old independent ``target_randomization`` mechanism this field used to gate against was
    removed 2026-08-22, once every skill in this project had moved to kick_aim_enabled=True and it
    had no live consumer left (see SkillConfig.kick_aim_enabled's own docstring for the same
    removal on the multi-skill side)."""

    kick_aim_theta_max_deg: float = 15.0
    """Per-reset uniform +/- half-range (degrees) sampled into ``kick_aim_theta``. Only meaningful
    when ``kick_aim_enabled=True``. Must be in (0.0, kick_aim_theta_ref_deg]."""

    kick_aim_theta_ref_deg: float = 45.0
    """Fixed normalization reference (degrees) for the ``kick_aim_command`` observation: the actor
    reads ``kick_aim_theta / kick_aim_theta_ref_deg``, not ``/ kick_aim_theta_max_deg`` -- see
    ``MultiSkillConfig.kick_aim_theta_ref_deg``'s own docstring (this field mirrors it exactly) for
    why keeping this fixed across curriculum changes to ``kick_aim_theta_max_deg`` matters."""

    kick_aim_nominal_distance_m: float = 5.0
    """Fixed distance ``D`` (meters) at which the per-attempt target point is synthesized along the
    commanded bearing, when ``kick_aim_enabled=True`` -- see ``MultiSkillConfig.
    kick_aim_nominal_distance_m``'s own docstring (this field mirrors it exactly)."""

    success_radius: float = 0.5
    """A shot succeeds (one-shot goal-burst reward) when the ball passes within this distance
    (meters, xy-plane) of ``target``."""

    shooting_reward_scale: float = 0.8
    """Global TARGET scale (RoboNaldo's stage-wise task weight w_g) on all shooting reward terms.
    0.0 disables them entirely (they're skipped, not just zeroed) — the pure-motion-tracking
    bootstrap stage. See configs/ball.yaml's staging comment. When ``shooting_reward_scale_ramp_
    iters > 0`` this is the value the ramp climbs TO, not an instant step -- see that field."""

    shooting_reward_scale_ramp_iters: int = 0
    """Ramp ``shooting_reward_scale`` from 0 up to its target value LINEARLY over this many control
    steps after the env is constructed, instead of applying it as a single discontinuous step.
    0 (default) = old behavior, an instant step -- every existing config is unaffected.

    Why this exists (2026-07-20, root-caused via [[stagec_obs_normalizer_shock]] and user
    directive): resuming a Stage-B checkpoint straight into Stage C used to be a DISCONTINUOUS
    change on two fronts at once -- the shooting reward terms jumping from weight 0 to their full
    target in one step, AND (until 2026-07-21) the ball/target observation flipping from a hard
    zero, held that way for the entire Stage-B run, to live values that normalized to +/-100-170 on
    the very first Stage-C step -- 30-100x any input the network had ever seen. Measured actor
    output change: 182% of its own norm. Measured critic Q-value swing: 5.18 units on a ~3.86
    baseline, with sign flip -- hidden from qf_loss because the target critic (an EMA copy of the
    online one) was shocked by the identical amount, so their difference barely moved while both
    were wrong. FastSAC has no PPO-style trust region to bound how far a sudden reward/input change
    can move the policy in one step, unlike RoboNaldo's own PPO recipe.

    As of 2026-07-21, `ball_pos_b`/`target_pos_b` (managers/observation/terms/unified.py) are live
    in every stage -- the observation-side half of that discontinuity no longer exists, so this
    field now ramps only the REWARD side. `holosoma.utils.shooting_curriculum.current_w_g(env)`
    reads it; `managers/reward/terms/shooting.py`'s six terms and `kick_safety.py`'s three penalty
    terms multiply by the LIVE ramped value each step instead of a config-time-baked constant.
    Progress is keyed off ``env.common_step_counter``, which starts at 0 for any fresh process
    regardless of the checkpoint's own absolute saved iteration count, so this Just Works on a
    resume without needing to know the checkpoint's own step number.

    ``warm_start_stage_c.py`` (checkpoint surgery to reset a stale obs-normalizer and zero the
    first-layer weight columns on these dims before resuming) addressed the now-removed
    observation-side discontinuity specifically -- it has no target left to fix for checkpoints
    trained under the live-from-Stage-B scheme, and only remains relevant for resuming an OLD
    checkpoint trained before 2026-07-21 under the zero-obs-in-Stage-B design."""

    shooting_reward_scale_hold_iters: int = 0
    """Hold ``shooting_reward_scale`` at EXACTLY 0 for this many control steps BEFORE
    ``shooting_reward_scale_ramp_iters`` starts climbing. 0 (default) = no hold, i.e. the ramp (or
    instant step, if ramping is also disabled) starts immediately on step 1 -- every existing
    config is unaffected.

    Why this exists (2026-07-20, user directive): the ramp alone still lets w_g move off zero on
    the very first Stage-C step (a tiny value, but nonzero), so the optimizer's Adam moment
    estimates and the critic's Q-estimates -- already disturbed by the resume itself (a fresh
    replay buffer, a changed num_envs/seed) independent of anything shooting-reward-related --
    never get a window to re-equilibrate on the UNCHANGED Stage-B task before a new objective
    starts arriving too. A hold reproduces exactly the w_g=0 "resume into an unchanged task"
    control condition measured earlier this project (see `stagec_obs_normalizer_shock.md`): let
    the resume transient settle first, then introduce the new signal.

    No-op with ``shooting_reward_scale_ramp_iters=0`` (an instant step ignores any hold -- there is
    no ramp phase for a hold to precede). See `holosoma.utils.shooting_curriculum.ramp_progress`
    for the exact schedule: 0 for the first `hold_iters` steps, then linear over the following
    `ramp_iters` steps, reaching `shooting_reward_scale` at `hold_iters + ramp_iters`."""

    orientation_tracking_alpha: float = 1.0
    """Sharpening factor on the two ORIENTATION motion-tracking kernels only (their sigma becomes
    sigma_0 / alpha). 1.0 = unchanged.

    This is RoboNaldo's per-stage `alpha` (Table B.1: "sigma = sigma_0/alpha with alpha=2 in Stages
    2-3"), applied SELECTIVELY. The paper tightens every tracking term; we tighten only orientation,
    for two reasons:

      * The paper's own per-term analysis says orientation is the balance-critical channel. Its
        Stage-3 relaxation schedule (Eq. 4) gives feet linear velocity near-total freedom
        (mu = 0.05, "near-total freedom in ankle placement during contact") while holding body
        orientation at mu = 0.6, explicitly "preserving balance during the lean".
      * Sharpening a kernel is not uniformly "more pressure": exp(-e^2/sigma^2) with smaller sigma
        discriminates harder near zero error but goes flat (and gradient-free) at moderate error.
        Applying that to position/velocity would clamp exactly the channels the policy must be free
        to deviate on in order to reach a randomized ball.

    Measured motivation (2026-07-20): Stage B (w_g = 0) survives a triggered kick in MuJoCo deploy
    with min_z 0.737-0.744, while every Stage C checkpoint collapses to min_z 0.06-0.11 — and does
    so BEFORE ball contact (fall at 1.30 s vs strike at 3.42 s), progressively earlier as Stage C
    training continues. Stage C is trading tracking fidelity for shooting reward, and the tracked
    motion is what supplies balance."""

    observation_bias: float = 0.0
    """Per-episode uniform ± half-range (meters) of a CONSTANT bias added to the kick_ball_pos_b
    observation in the robot's heading frame — drawn once per episode reset and held constant for
    the whole episode. 0.0 (default) = off; every existing config is bit-identical.

    Why this exists (2026-07-21): the existing per-tick observation noise (ObsTermCfg.noise = 0.05
    on kick_ball_pos_b) is i.i.d. each step, so the policy can low-pass/average it away over a
    temporal window and recover the clean underlying ball_pos_b trajectory — which is why 5 cm of
    per-tick jitter did NOT stop the live-but-unrewarded ball obs from becoming a clean self-motion
    proxy that overfits and destabilizes the kick at deploy (memory
    liveobs_stageb_depends_on_obs_deploy_fragility). A per-EPISODE constant bias CANNOT be averaged
    away within an episode, so it actually forces the policy to treat ball_pos_b as approximate. It
    also models the dominant real-world ball-perception error — a roughly-constant extrinsic/
    calibration offset per session — better than i.i.d. jitter does. Applied only to kick_ball_pos_b
    (a perceived object); kick_target_pos_b is a command and stays unbiased. Drawn in
    managers/randomization/terms/locomotion.py::randomize_ball_obs_bias, read in
    managers/observation/terms/unified.py::ball_pos_b."""

    observation_noise: float = 0.05
    """Flat per-step uniform ± half-range (meters) of i.i.d. noise on kick_ball_pos_b (a
    depth-camera/LiDAR-style measurement jitter, redrawn every control step) -- see
    ObsTermCfg.noise / ObservationManager._apply_noise. 0.05 is the pre-2026-07-24 hardcoded
    value; kept as the default so an unset config is bit-identical to before this field existed.
    Actor-only, like every field in this "sensor noise" group below -- the critic always reads
    the clean, instantaneous ground truth (ball_pos_b_ground_truth)."""

    observation_noise_range_coefficient: float = 0.03
    """Additional per-step noise magnitude PER METER of the ball's own distance, added on top of
    ``observation_noise`` (effective noise half-range = observation_noise +
    observation_noise_range_coefficient * ||ball_pos_b||) -- models a real depth sensor's error
    growing with range rather than staying flat. 0.03 (commonly-cited stereo-depth ballpark, ~3%
    of range) is the pre-2026-07-24 hardcoded default. 0.0 disables this term (pure flat noise)."""

    observation_delay_steps_min: int = 0
    """Lower bound (control steps, inclusive) of the per-episode-fixed randomized perception
    latency on kick_ball_pos_b -- models camera/LiDAR capture + Kalman-filter fusion lag. Redrawn
    per env at every reset, held fixed for that episode. See ObsTermCfg.delay_step_range /
    ObservationManager._apply_delay (mirrors the existing action-delay-queue mechanism in
    managers/action/terms/joint_control.py). 0 is the pre-2026-07-24 default (no minimum delay)."""

    observation_delay_steps_max: int = 3
    """Upper bound (control steps, inclusive) of the same randomized latency range. 3 steps (0-60ms
    at this project's dt=0.02) is the pre-2026-07-24 hardcoded default. Set equal to
    ``observation_delay_steps_min`` for a fixed (non-randomized) delay, or both to 0 to disable
    latency entirely."""

    observation_hold_steps_min: int = 0
    """Lower bound (control steps) of the randomized zero-order-hold period on kick_ball_pos_b --
    models the perception/fusion pipeline's own UPDATE RATE being slower than the control loop
    (distinct from observation_delay_steps_{min,max}'s transport LATENCY on a pipeline that still
    updates every tick). E.g. a fixed 25Hz fused ball-pose estimate under this project's 50Hz
    control loop -> both min and max = 2 (round(50/25)); a wider range also models the real rate
    jittering episode-to-episode. Mirrors MultiSkillConfig.ball_obs_hold_steps_min. 0 (default,
    paired with max=0) = off, update every control tick, bit-identical to before this field
    existed."""

    observation_hold_steps_max: int = 0
    """Upper bound (control steps) of the same hold-period range -- mirrors
    MultiSkillConfig.ball_obs_hold_steps_max."""

    observation_stale_probability: float = 0.0
    """Per-CONTROL-TICK probability (re-rolled every step, unlike observation_hold_steps_{min,max}'s
    per-episode-fixed period) that kick_ball_pos_b reuses the PREVIOUS tick's already-fully-
    processed reading instead of this tick's -- models a single dropped/repeated sensor frame,
    transient and self-correcting, distinct from observation_hold_steps_{min,max}'s deterministic
    slower-update-rate modeling and from ball_static_obs_probability's own freeze-for-the-rest-of-
    the-episode-to-an-independently-drawn-value mechanism. Mirrors
    MultiSkillConfig.ball_obs_stale_probability -- see that field's own docstring for the
    RoboNaldo cross-check (their lidar_stale_probability). See ObsTermCfg.stale_probability /
    ObservationManager._apply_stale. 0.0 (default) = off, bit-identical to before this field
    existed."""

    ood_spawn_probability: float = 0.0
    """Legacy single-skill counterpart of MultiSkillConfig.ood_spawn_probability / MotionConfig's
    field of the same name -- see that docstring for the full deployment-robustness rationale.
    0.0 (default) = off, bit-identical to before this field existed."""

    ood_region_multiplier: float = 3.0
    """Legacy single-skill counterpart of MultiSkillConfig.ood_region_multiplier."""

    ball_static_obs_probability: float = 0.0
    """Legacy single-skill counterpart of MultiSkillConfig.ball_static_obs_probability -- see that
    field's docstring. 0.0 (default) = off."""

    kick_contact_force_penalty_floor: float = 3.0
    """Legacy single-skill counterpart of MultiSkillConfig.kick_contact_force_penalty_floor -- see
    that field's docstring. 3.0 (default) matches the pre-2026-07-28 hardcoded
    ``_KICK_SAFETY_FLOOR`` in config_values/unified/g1/reward.py, so an unset config is
    bit-identical to before this field existed."""

    kick_contact_force_penalty_k: float = 15.0
    """Legacy single-skill counterpart of MultiSkillConfig.kick_contact_force_penalty_k -- see
    that field's docstring. 15.0 (default) matches the pre-2026-07-28 hardcoded value."""

    kick_contact_force_threshold_bodyweight_multiplier: float = 3.0
    """Legacy single-skill counterpart of MultiSkillConfig.kick_contact_force_threshold_bodyweight_multiplier
    -- see that field's docstring. 3.0 (default) matches the pre-2026-07-28 hardcoded value."""

    start_at_timestep_zero_prob: float = 1.0
    """Legacy single-skill counterpart of MultiSkillConfig.start_at_timestep_zero_prob -- see that
    field's docstring. 1.0 (default) matches the pre-2026-07-28 hardcoded value in
    config_values/unified/g1/command.py's ``_motion_config_unified_kick``."""

    rsi_scope_to_authored_clip: bool = False
    """Legacy single-skill counterpart of MultiSkillConfig.rsi_scope_to_authored_clip -- see that
    field's docstring. False (default) = current behavior, exact no-op."""

    critical_frame_oversampling_prob: float = 0.0
    """Legacy single-skill counterpart of MultiSkillConfig.critical_frame_oversampling_prob -- see
    that field's docstring. 0.0 (default) = exact no-op."""

    critical_frame_sampling_window: int = 10
    """Legacy single-skill counterpart of MultiSkillConfig.critical_frame_sampling_window -- see
    that field's docstring."""

    motion_head_velocity_smoothing_frames: int = 0
    """Legacy single-skill counterpart of
    MultiSkillConfig.motion_head_velocity_smoothing_frames -- see that field's docstring, and
    managers/command/terms/motion_head_velocity_smoothing.py's module docstring for the measured
    motivation. 0 (default) = OFF, exact no-op."""

    penalty_curriculum_enabled: bool = True
    """Legacy single-skill counterpart of MultiSkillConfig.penalty_curriculum_enabled -- see that
    field's docstring. True (default) = current behavior, unchanged."""

    post_flip_termination_grace_steps: float = 0.0
    """Legacy single-skill counterpart of MultiSkillConfig.post_flip_termination_grace_steps --
    see that field's docstring. 0.0 (default) = exact no-op."""

    post_flip_reward_decay_steps: float = 0.0
    """Legacy single-skill counterpart of MultiSkillConfig.post_flip_reward_decay_steps -- see
    that field's docstring. 0.0 (default) = exact no-op."""

    kick_target_entropy_ratio: float | None = None
    """Legacy single-skill counterpart of MultiSkillConfig.kick_target_entropy_ratio /
    FastSACConfig.kick_target_entropy_ratio -- see that field's docstring for the full design.
    None (default) = OFF, bit-identical to today's single shared SAC alpha."""

    kick_gamma: float | None = None
    """Legacy single-skill counterpart of MultiSkillConfig.kick_gamma / FastSACConfig.kick_gamma
    -- see that field's docstring for the full design. None (default) = OFF, bit-identical to
    today's single shared SAC discount factor."""

    critic_v_min: float | None = None
    """Legacy single-skill counterpart of MultiSkillConfig.critic_v_min / FastSACConfig.v_min --
    see that field's docstring for the measurement that motivated it and, importantly, for the two
    checkpoint-compatibility hazards. None (default) = leave the experiment preset's own value
    untouched, bit-identical to before this field existed."""

    critic_v_max: float | None = None
    """Legacy single-skill counterpart of MultiSkillConfig.critic_v_max / FastSACConfig.v_max."""

    critic_num_atoms: int | None = None
    """Legacy single-skill counterpart of MultiSkillConfig.critic_num_atoms /
    FastSACConfig.num_atoms -- changing this makes an existing checkpoint UNLOADABLE (shape
    mismatch); see MultiSkillConfig.critic_num_atoms' docstring."""

    replay_buffer_sanitize_enabled: bool = False
    """Legacy single-skill counterpart of MultiSkillConfig.replay_buffer_sanitize_enabled /
    FastSACConfig.replay_buffer_sanitize_enabled -- see that field's docstring for the full
    design. False (default) = OFF, bit-identical to today's unsanitized replay buffer writes."""

    l2sp_weight: float = 0.0
    """Legacy single-skill counterpart of MultiSkillConfig.l2sp_weight / FastSACConfig.l2sp_weight
    -- see that field's docstring for the full design and the value table. 0.0 (default) = OFF,
    bit-identical to today's training. Rarely useful here: L2-SP exists to protect PREVIOUSLY
    learned skills when adding new ones, which is inherently a multi-skill situation."""

    joint_pos_sanity_check_enabled: bool = False
    """Legacy single-skill counterpart of MultiSkillConfig.joint_pos_sanity_check_enabled -- see
    that field's docstring for the full design. False (default) = OFF, term not registered."""

    joint_pos_sanity_threshold: float = 20.0
    """Legacy single-skill counterpart of MultiSkillConfig.joint_pos_sanity_threshold -- see that
    field's docstring. Only read when joint_pos_sanity_check_enabled is True."""

    bad_tracking_swing_only: bool = False
    """Legacy single-skill counterpart of MultiSkillConfig.bad_tracking_swing_only -- see that
    field's docstring for the full rationale. False (default) = off, bit-identical to before this
    field existed."""

    kick_recovery_termination_handoff: bool = False
    """Legacy single-skill counterpart of MultiSkillConfig.kick_recovery_termination_handoff --
    see that field's docstring for the full rationale. False (default) = off, bit-identical to
    before this field existed."""

    bad_tracking_swing_threshold_multiplier: float = 1.0
    """Legacy single-skill counterpart of MultiSkillConfig.bad_tracking_swing_threshold_multiplier
    -- see that field's docstring for the full rationale, including the 2026-07-30 measurement
    showing values > 1.0 make the fall rate WORSE. 1.0 (default) = off, bit-identical to before
    this field existed."""

    swing_tracking_sigma_multiplier: float = 1.0
    """Legacy single-skill counterpart of MultiSkillConfig.swing_tracking_sigma_multiplier -- see
    that field's docstring for the full rationale, including the read-before-raising note about
    the sibling bad_tracking_swing_threshold_multiplier's own measured-worse outcome. 1.0
    (default) = off, bit-identical to before this field existed."""

    balance_potential_weight: float = 0.0
    """Legacy single-skill counterpart of MultiSkillConfig.balance_potential_weight -- see that
    field's docstring for the full rationale. 0.0 (default) = term absent, bit-identical to before
    this field existed."""

    use_foot_strike_pitch_reference_relative: bool = False
    """Legacy single-skill counterpart of MultiSkillConfig.use_foot_strike_pitch_reference_relative
    -- see that field's docstring for the full rationale. False (default) = current absolute-target
    behavior, exact no-op."""

    kick_terrain_light_rough_proportion: float = 0.0
    """Legacy single-skill counterpart of MultiSkillConfig.kick_terrain_light_rough_proportion --
    see that field's docstring for the full rationale (including the terrain-proportion
    normalization side effect). 0.0 (default) = tier not generated, bit-identical to before this
    field existed."""

    kick_terrain_light_rough_max_height: float = 0.008
    """Legacy single-skill counterpart of MultiSkillConfig.kick_terrain_light_rough_max_height."""

    kick_eligible_terrain_types: tuple[str, ...] = ("flat",)
    """Legacy single-skill counterpart of MultiSkillConfig.kick_eligible_terrain_types. Default
    ("flat",) reproduces the previous hardcoded flat-only kick gate exactly."""

    body_push_enabled: bool = False
    """Legacy single-skill counterpart of MultiSkillConfig.body_push_enabled -- see that field's
    docstring for the full rationale. False (default) = term absent, bit-identical to before this
    field existed."""

    body_push_interval_min_s: float = 4.0
    """Legacy single-skill counterpart of MultiSkillConfig.body_push_interval_min_s."""

    body_push_interval_max_s: float = 8.0
    """Legacy single-skill counterpart of MultiSkillConfig.body_push_interval_max_s."""

    body_push_force_min: float = 20.0
    """Legacy single-skill counterpart of MultiSkillConfig.body_push_force_min."""

    body_push_force_max: float = 80.0
    """Legacy single-skill counterpart of MultiSkillConfig.body_push_force_max."""

    body_push_duration_min_s: float = 0.05
    """Legacy single-skill counterpart of MultiSkillConfig.body_push_duration_min_s."""

    body_push_duration_max_s: float = 0.20
    """Legacy single-skill counterpart of MultiSkillConfig.body_push_duration_max_s."""

    body_push_vertical_fraction: float = 0.2
    """Legacy single-skill counterpart of MultiSkillConfig.body_push_vertical_fraction."""

    body_push_body_names: tuple[str, ...] | None = None
    """Legacy single-skill counterpart of MultiSkillConfig.body_push_body_names. None (default)
    uses DEFAULT_BODY_PUSH_BODIES."""

    kick_recovery_locomotion_flip_enabled: bool = False
    """Legacy single-skill counterpart of MultiSkillConfig.kick_recovery_locomotion_flip_enabled --
    see that field's docstring for the full rationale (Stage D's post-swing -> locomotion handoff).
    False (default) = current recovery/hold behavior, exact no-op."""

    bad_motion_body_pos_threshold: float = 0.25
    """Legacy single-skill counterpart of MultiSkillConfig.bad_motion_body_pos_threshold -- see
    that field's docstring for the full rationale (RoboNaldo's progressive ``ee_body_pos``
    threshold, 0.25/0.35/0.5 across their curriculum). 0.25 (default) matches this project's
    existing hardcoded value, bit-identical to before this field existed."""

    ee_body_pos_warmup_threshold: float = 0.25
    """Legacy single-skill counterpart of MultiSkillConfig.ee_body_pos_warmup_threshold -- see
    that field's docstring for the full rationale (RoboNaldo's non-monotonic 0.25/0.7/0.7
    ``warmup_threshold`` schedule across S1/S2a/S2b). 0.25 (default) matches its own hardcoded
    default, bit-identical to before this field existed."""

    kick_recovery_drift_deadzone: float = 0.15
    """Legacy single-skill counterpart of MultiSkillConfig.kick_recovery_drift_deadzone -- see
    that field's docstring for the full rationale. 0.15 (15cm, default) is the user-specified
    starting point; only takes effect when kick_recovery_termination_handoff is True."""

    kick_recovery_stand_height_deadzone: float = 0.015
    """Legacy single-skill counterpart of MultiSkillConfig.kick_recovery_stand_height_deadzone --
    see that field's docstring for the full rationale. 0.015 (default) matches its own hardcoded
    value, bit-identical to before this field existed."""

    kick_recovery_stand_orientation_deadzone: float = 0.025
    """Legacy single-skill counterpart of MultiSkillConfig.kick_recovery_stand_orientation_deadzone
    -- see that field's docstring for the full rationale. 0.025 (default) matches its own
    hardcoded value, bit-identical to before this field existed."""

    kick_recovery_stand_feet_width_deadzone: float = 0.03
    """Legacy single-skill counterpart of MultiSkillConfig.kick_recovery_stand_feet_width_deadzone
    -- see that field's docstring for the full rationale. 0.03 (default) matches its own hardcoded
    value, bit-identical to before this field existed."""

    kick_recovery_stand_knee_width_deadzone: float = 0.03
    """Legacy single-skill counterpart of MultiSkillConfig.kick_recovery_stand_knee_width_deadzone
    -- see that field's docstring for the full rationale. 0.03 (default) matches its own hardcoded
    value, bit-identical to before this field existed."""

    kick_swing_orientation_deadzone: float = 0.0
    """Legacy single-skill counterpart of MultiSkillConfig.kick_swing_orientation_deadzone -- see
    that field's docstring for the full rationale. 0.0 (default) matches its own hardcoded value,
    bit-identical to before this field existed."""

    kick_swing_torso_orientation_deadzone: float = 0.0
    """Legacy single-skill counterpart of MultiSkillConfig.kick_swing_torso_orientation_deadzone
    -- see that field's docstring for the full rationale. 0.0 (default) matches its own hardcoded
    value, bit-identical to before this field existed."""

    kick_ball_velocity_v_ref: float | None = None
    """Legacy single-skill counterpart of MultiSkillConfig.kick_ball_velocity_v_ref -- see that
    field's docstring. None (default) keeps ball_velocity's own registered 5.0."""

    kick_error_ball_to_target_sigma: float | None = None
    """Legacy single-skill counterpart of MultiSkillConfig.kick_error_ball_to_target_sigma -- see
    that field's docstring. None (default) keeps error_ball_to_target's own registered 1.0."""

    kick_ball_velocity_use_latched_peak_speed: bool = False
    """Legacy single-skill counterpart of MultiSkillConfig.kick_ball_velocity_use_latched_peak_speed."""

    kick_ball_velocity_use_post_locomotion_gate: bool = False
    """Legacy single-skill counterpart of MultiSkillConfig.kick_ball_velocity_use_post_locomotion_gate."""

    kick_ball_over_line_require_has_kicked: bool = False
    """Legacy single-skill counterpart of MultiSkillConfig.kick_ball_over_line_require_has_kicked
    -- see that field's docstring for the full rationale. False (default) = off, bit-identical to
    before this field existed."""

    static_friction: float = 0.7
    dynamic_friction: float = 0.6
    restitution: float = 0.4

    def resolved_nominal_bearing_deg(self) -> float:
        """Azimuth (degrees; atan2 convention, 0=+x/forward, positive=+y/the robot's own left) from
        ``position`` to ``target``. Legacy single-skill counterpart of
        ``SkillConfig.resolved_nominal_bearing_deg()`` -- see that method's own docstring for why
        this MUST be calibrated against a real rollout (scripts/calibrate_nominal_bearing.py)
        rather than asserted from an authored ``target``, when ``kick_aim_enabled=True``."""
        return math.degrees(math.atan2(self.target[1] - self.position[1], self.target[0] - self.position[0]))


# Fork root (.../ball_kicking_learning), not the holosoma package root, since this yaml is a
# user-facing project config, not a package-bundled resource under holosoma/data/.
#
# Overridable via the HOLOSOMA_BALL_CONFIG env var (not a CLI flag) because several call sites --
# notably config_values/unified/g1/reward.py's `_w_g = load_ball_config().shooting_reward_scale`
# -- read this at MODULE IMPORT time, to bake the shooting-reward weights into the tyro config
# tree before the CLI is even parsed. A flag would resolve too late: it would swap the ball
# position/randomization correctly but leave the reward weights built from the default file,
# silently. An env var is set before the process starts, so it's visible in time either way:
#   HOLOSOMA_BALL_CONFIG=configs/ball_stageB.yaml python src/holosoma/holosoma/train_agent.py ...
# Unset (the default) behaves exactly as before -- configs/ball.yaml.
DEFAULT_BALL_CONFIG_YAML = Path(
    os.environ.get("HOLOSOMA_BALL_CONFIG") or Path(__file__).resolve().parents[4] / "configs" / "ball.yaml"
)

# HOLOSOMA_BALL_X / HOLOSOMA_BALL_Y -- same env-var mechanism as HOLOSOMA_BALL_CONFIG above, and
# for the identical reason: a real `--ball.x`/`--ball.y` tyro flag on ExperimentConfig would
# resolve after config_values/unified/g1/reward.py's module-level `load_ball_config()` call, too
# late to matter. `apply_ball_cli_overrides()` below is the CLI-facing side of this: it lets an
# entrypoint accept `--ball.x`/`--ball.y` on argv and turns them into these two env vars before
# any holosoma.config_values module gets imported, so the effect is indistinguishable from having
# set the env vars by hand. x/y only (not the rest of BallConfig) because those are the two fields
# worth nudging per-run without editing/forking a whole ball*.yaml.
BALL_X_ENV_VAR = "HOLOSOMA_BALL_X"
BALL_Y_ENV_VAR = "HOLOSOMA_BALL_Y"


def apply_ball_cli_overrides(argv: list[str] | None = None) -> None:
    """Pull `--ball.x`/`--ball.y` out of argv (either `--ball.x 2.84` or `--ball.x=2.84`) and stash
    them as env vars that `load_ball_config()` applies on top of whatever configs/ball*.yaml sets.

    Must be called as the very first thing an entrypoint does -- before importing anything under
    holosoma.config_values -- for the same reason HOLOSOMA_BALL_CONFIG must be set before the
    process starts (see the comment above `DEFAULT_BALL_CONFIG_YAML`). Safe to import
    `config_types.simulator` itself for this: it does not import any `config_values` module.

    Mutates argv in place, removing the consumed flags so downstream tyro parsing (which knows
    nothing about `--ball.x`/`--ball.y` -- they're not part of ExperimentConfig) doesn't choke on
    an unrecognized argument.
    """
    if argv is None:
        argv = sys.argv
    flag_env_vars = {"--ball.x": BALL_X_ENV_VAR, "--ball.y": BALL_Y_ENV_VAR}
    i = 1
    while i < len(argv):
        arg = argv[i]
        consumed = False
        for flag, env_var in flag_env_vars.items():
            if arg == flag:
                if i + 1 >= len(argv):
                    raise ValueError(f"{flag} requires a value, e.g. {flag} 2.84")
                os.environ[env_var] = argv[i + 1]
                del argv[i : i + 2]
                consumed = True
                break
            if arg.startswith(flag + "="):
                os.environ[env_var] = arg[len(flag) + 1 :]
                del argv[i]
                consumed = True
                break
        if not consumed:
            i += 1


def load_ball_config(yaml_path: str | Path = DEFAULT_BALL_CONFIG_YAML) -> BallConfig:
    """Build a BallConfig from a YAML file (radius, mass, x, y — see configs/ball.yaml).

    z is not read from YAML; it's always set to `radius` so the ball rests on the ground.
    `static_friction`/`dynamic_friction`/`restitution` may optionally be included in the YAML too;
    otherwise they fall back to BallConfig's defaults.

    `x`/`y` are overridden by the HOLOSOMA_BALL_X/HOLOSOMA_BALL_Y env vars when set (see
    `apply_ball_cli_overrides` for the CLI-facing `--ball.x`/`--ball.y` side of this) -- applied
    before `target`'s default (x + 5.0, y) is computed, so an overridden spawn position still gets
    a sensibly-offset default target unless the YAML sets target_x/target_y explicitly.
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Ball config YAML not found: {yaml_path}\n"
            "Expected a file with at least 'radius', 'mass', 'x', 'y' keys."
        )
    with open(yaml_path) as f:
        raw = yaml.safe_load(f) or {}

    required = {"radius", "mass", "x", "y"}
    missing = required - raw.keys()
    if missing:
        raise ValueError(f"Ball config YAML {yaml_path} is missing required key(s): {sorted(missing)}")

    if BALL_X_ENV_VAR in os.environ:
        raw["x"] = float(os.environ[BALL_X_ENV_VAR])
    if BALL_Y_ENV_VAR in os.environ:
        raw["y"] = float(os.environ[BALL_Y_ENV_VAR])

    kwargs: dict[str, Any] = {
        "radius": float(raw["radius"]),
        "mass": float(raw["mass"]),
        "position": (float(raw["x"]), float(raw["y"]), float(raw["radius"])),
    }
    for optional_key in (
        "static_friction",
        "dynamic_friction",
        "restitution",
        "success_radius",
        "shooting_reward_scale",
        "shooting_reward_scale_ramp_iters",
        "shooting_reward_scale_hold_iters",
        "orientation_tracking_alpha",
        "observation_bias",
        "observation_noise",
        "observation_noise_range_coefficient",
        "observation_delay_steps_min",
        "observation_delay_steps_max",
        "observation_hold_steps_min",
        "observation_hold_steps_max",
        "observation_stale_probability",
        "ood_spawn_probability",
        "ood_region_multiplier",
        "ball_static_obs_probability",
        "kick_contact_force_penalty_floor",
        "kick_contact_force_penalty_k",
        "kick_contact_force_threshold_bodyweight_multiplier",
        "start_at_timestep_zero_prob",
        "rsi_scope_to_authored_clip",
        "critical_frame_oversampling_prob",
        "critical_frame_sampling_window",
        "motion_head_velocity_smoothing_frames",
        "penalty_curriculum_enabled",
        "post_flip_termination_grace_steps",
        "post_flip_reward_decay_steps",
        "kick_target_entropy_ratio",
        "kick_gamma",
        "critic_v_min",
        "critic_v_max",
        "critic_num_atoms",
        "replay_buffer_sanitize_enabled",
        "l2sp_weight",
        "joint_pos_sanity_check_enabled",
        "joint_pos_sanity_threshold",
        "bad_tracking_swing_only",
        "kick_recovery_termination_handoff",
        "bad_tracking_swing_threshold_multiplier",
        "swing_tracking_sigma_multiplier",
        "balance_potential_weight",
        "use_foot_strike_pitch_reference_relative",
        "kick_recovery_locomotion_flip_enabled",
        "bad_motion_body_pos_threshold",
        "ee_body_pos_warmup_threshold",
        "kick_recovery_drift_deadzone",
        "kick_recovery_stand_height_deadzone",
        "kick_recovery_stand_orientation_deadzone",
        "kick_recovery_stand_feet_width_deadzone",
        "kick_recovery_stand_knee_width_deadzone",
        "kick_swing_orientation_deadzone",
        "kick_swing_torso_orientation_deadzone",
        "kick_ball_over_line_require_has_kicked",
        "kick_ball_velocity_v_ref",
        "kick_error_ball_to_target_sigma",
        "kick_ball_velocity_use_latched_peak_speed",
        "kick_ball_velocity_use_post_locomotion_gate",
        "kick_aim_enabled",
        "kick_aim_theta_max_deg",
        "kick_aim_theta_ref_deg",
        "kick_aim_nominal_distance_m",
        "body_push_enabled",
        "body_push_interval_min_s",
        "body_push_interval_max_s",
        "body_push_force_min",
        "body_push_force_max",
        "body_push_duration_min_s",
        "body_push_duration_max_s",
        "body_push_vertical_fraction",
        "kick_terrain_light_rough_proportion",
        "kick_terrain_light_rough_max_height",
    ):
        if optional_key in raw:
            if optional_key in (
                "bad_tracking_swing_only",
                "kick_recovery_termination_handoff",
                "rsi_scope_to_authored_clip",
                "kick_ball_over_line_require_has_kicked",
                "kick_ball_velocity_use_latched_peak_speed",
                "kick_ball_velocity_use_post_locomotion_gate",
                "use_foot_strike_pitch_reference_relative",
                "kick_recovery_locomotion_flip_enabled",
                "replay_buffer_sanitize_enabled",
                "joint_pos_sanity_check_enabled",
                "penalty_curriculum_enabled",
                "kick_aim_enabled",
                "body_push_enabled",
            ):
                kwargs[optional_key] = bool(raw[optional_key])
                continue
            if optional_key in ("critical_frame_sampling_window", "motion_head_velocity_smoothing_frames"):
                kwargs[optional_key] = int(raw[optional_key])
                if kwargs[optional_key] < 0:
                    raise ValueError(
                        f"{yaml_path}: {optional_key} must be >= 0, got {kwargs[optional_key]}"
                    )
                continue
            caster = (
                int
                if (
                    optional_key.endswith("_iters")
                    or optional_key.endswith("_steps_min")
                    or optional_key.endswith("_steps_max")
                    or optional_key == "critic_num_atoms"
                )
                else float
            )
            kwargs[optional_key] = caster(raw[optional_key])
            if optional_key == "critic_num_atoms" and kwargs[optional_key] < 2:
                raise ValueError(
                    f"{yaml_path}: critic_num_atoms must be >= 2, got {kwargs[optional_key]}"
                )
            if optional_key in (
                "ood_spawn_probability",
                "ball_static_obs_probability",
                "start_at_timestep_zero_prob",
                "critical_frame_oversampling_prob",
                "observation_stale_probability",
            ) and not 0.0 <= kwargs[optional_key] <= 1.0:
                raise ValueError(f"{yaml_path}: {optional_key} must be in [0.0, 1.0], got {kwargs[optional_key]}")
            if optional_key == "kick_gamma" and not 0.0 < kwargs[optional_key] < 1.0:
                raise ValueError(f"{yaml_path}: kick_gamma must be in (0.0, 1.0), got {kwargs[optional_key]}")
            if optional_key in (
                "post_flip_termination_grace_steps",
                "post_flip_reward_decay_steps",
            ) and kwargs[optional_key] < 0.0:
                raise ValueError(f"{yaml_path}: {optional_key} must be >= 0.0, got {kwargs[optional_key]}")
            if optional_key == "swing_tracking_sigma_multiplier" and kwargs[optional_key] <= 0.0:
                raise ValueError(
                    f"{yaml_path}: swing_tracking_sigma_multiplier must be > 0.0 (it divides "
                    f"inside an exp() denominator), got {kwargs[optional_key]}"
                )
            if optional_key == "bad_motion_body_pos_threshold" and kwargs[optional_key] <= 0.0:
                raise ValueError(
                    f"{yaml_path}: bad_motion_body_pos_threshold must be > 0.0, got {kwargs[optional_key]}"
                )
            if optional_key == "ee_body_pos_warmup_threshold" and kwargs[optional_key] <= 0.0:
                raise ValueError(
                    f"{yaml_path}: ee_body_pos_warmup_threshold must be > 0.0, got {kwargs[optional_key]}"
                )
            if optional_key == "kick_recovery_drift_deadzone" and kwargs[optional_key] <= 0.0:
                raise ValueError(
                    f"{yaml_path}: kick_recovery_drift_deadzone must be > 0.0, got {kwargs[optional_key]}"
                )
            if optional_key == "joint_pos_sanity_threshold" and kwargs[optional_key] <= 0.0:
                raise ValueError(
                    f"{yaml_path}: joint_pos_sanity_threshold must be > 0.0, got {kwargs[optional_key]}"
                )
            # REWARD deadzones (tolerance before a penalty accrues) accept 0.0 -- unlike the
            # termination-threshold fields above, 0.0 is a legitimate value here, matching this
            # codebase's own existing deadzone=0.0 defaults elsewhere (e.g.
            # penalty_kick_swing_orientation) -- validated >= 0.0, not > 0.0.
            if optional_key in (
                "kick_recovery_stand_height_deadzone",
                "kick_recovery_stand_orientation_deadzone",
                "kick_recovery_stand_feet_width_deadzone",
                "kick_recovery_stand_knee_width_deadzone",
                "kick_swing_orientation_deadzone",
                "kick_swing_torso_orientation_deadzone",
            ) and kwargs[optional_key] < 0.0:
                raise ValueError(f"{yaml_path}: {optional_key} must be >= 0.0, got {kwargs[optional_key]}")
            if optional_key == "body_push_vertical_fraction" and not 0.0 <= kwargs[optional_key] <= 1.0:
                raise ValueError(
                    f"{yaml_path}: body_push_vertical_fraction must be in [0.0, 1.0], got "
                    f"{kwargs[optional_key]}"
                )

    _body_push_interval = (
        kwargs.get("body_push_interval_min_s", 4.0),
        kwargs.get("body_push_interval_max_s", 8.0),
    )
    if _body_push_interval[0] <= 0.0 or _body_push_interval[1] < _body_push_interval[0]:
        raise ValueError(
            f"{yaml_path}: body_push_interval_min_s/max_s must satisfy 0 < min <= max, got "
            f"{_body_push_interval}"
        )
    _body_push_force = (kwargs.get("body_push_force_min", 20.0), kwargs.get("body_push_force_max", 80.0))
    if _body_push_force[0] < 0.0 or _body_push_force[1] < _body_push_force[0]:
        raise ValueError(
            f"{yaml_path}: body_push_force_min/max must satisfy 0 <= min <= max, got {_body_push_force}"
        )
    _body_push_duration = (
        kwargs.get("body_push_duration_min_s", 0.05),
        kwargs.get("body_push_duration_max_s", 0.20),
    )
    if _body_push_duration[0] <= 0.0 or _body_push_duration[1] < _body_push_duration[0]:
        raise ValueError(
            f"{yaml_path}: body_push_duration_min_s/max_s must satisfy 0 < min <= max, got "
            f"{_body_push_duration}"
        )
    _light_rough_prop = kwargs.get("kick_terrain_light_rough_proportion", 0.0)
    if not 0.0 <= _light_rough_prop <= 1.0:
        raise ValueError(
            f"{yaml_path}: kick_terrain_light_rough_proportion must be in [0.0, 1.0], got "
            f"{_light_rough_prop}"
        )
    if kwargs.get("kick_terrain_light_rough_max_height", 0.008) <= 0.0:
        raise ValueError(
            f"{yaml_path}: kick_terrain_light_rough_max_height must be > 0.0, got "
            f"{kwargs['kick_terrain_light_rough_max_height']}"
        )
    if "kick_eligible_terrain_types" in raw:
        _types_raw = raw["kick_eligible_terrain_types"]
        if not (isinstance(_types_raw, list) and _types_raw):
            raise ValueError(
                f"{yaml_path}: kick_eligible_terrain_types must be a non-empty list of "
                f"terrain-type names if set, got {_types_raw!r}"
            )
        kwargs["kick_eligible_terrain_types"] = tuple(str(n) for n in _types_raw)
    # Same "eligible but never generated => zero kick envs" guard as the MultiSkillConfig loader.
    if "light_rough" in kwargs.get("kick_eligible_terrain_types", ("flat",)) and _light_rough_prop <= 0.0:
        raise ValueError(
            f"{yaml_path}: kick_eligible_terrain_types includes 'light_rough' but "
            f"kick_terrain_light_rough_proportion is {_light_rough_prop} -- that tier would never "
            "be generated, so no kick envs would exist."
        )

    if "body_push_body_names" in raw:
        _names_raw = raw["body_push_body_names"]
        if not (isinstance(_names_raw, list) and _names_raw):
            raise ValueError(
                f"{yaml_path}: body_push_body_names must be a non-empty list of link names if "
                f"set, got {_names_raw!r}"
            )
        kwargs["body_push_body_names"] = tuple(str(n) for n in _names_raw)

    # Shooting-task keys — all optional, defaults preserve pre-shooting-reward behavior
    # (no randomization; right foot; target 5m downrange of nominal spawn).
    if "randomize_x" in raw or "randomize_y" in raw:
        kwargs["position_randomization"] = (float(raw.get("randomize_x", 0.0)), float(raw.get("randomize_y", 0.0)))
    if "randomize_target_x" in raw or "randomize_target_y" in raw:
        raise ValueError(
            f"{yaml_path}: randomize_target_x/y was removed 2026-08-22 (azimuth-aim refactor) -- "
            "independent target randomization has no replacement; kick_aim_enabled=True derives "
            "all aim variation from kick_aim_theta instead (see BallConfig.kick_aim_enabled's own "
            "docstring), and kick_aim_enabled=False now reads a fixed, unrandomized target. "
            "Remove randomize_target_x/y from this file."
        )
    if "kick_foot" in raw:
        kick_foot = str(raw["kick_foot"]).lower()
        if kick_foot not in ("left", "right"):
            raise ValueError(f"Ball config YAML {yaml_path}: kick_foot must be 'left' or 'right', got {raw['kick_foot']!r}")
        kwargs["kick_foot"] = kick_foot
    if "target_x" in raw or "target_y" in raw:
        kwargs["target"] = (
            float(raw.get("target_x", float(raw["x"]) + 5.0)),
            float(raw.get("target_y", raw["y"])),
        )
    else:
        kwargs["target"] = (float(raw["x"]) + 5.0, float(raw["y"]))

    # kick_aim cross-field validation -- see BallConfig.kick_aim_enabled's own docstring for why
    # each check exists (double-randomized aim; an out-of-normalization-range theta_max_deg).
    _theta_ref = kwargs.get("kick_aim_theta_ref_deg", 45.0)
    _theta_max = kwargs.get("kick_aim_theta_max_deg", 15.0)
    if not 0.0 < _theta_max <= _theta_ref:
        raise ValueError(
            f"{yaml_path}: kick_aim_theta_max_deg must be in (0.0, kick_aim_theta_ref_deg="
            f"{_theta_ref}], got {_theta_max}"
        )
    if kwargs.get("kick_aim_nominal_distance_m", 5.0) <= 0.0:
        raise ValueError(
            f"{yaml_path}: kick_aim_nominal_distance_m must be > 0.0, got "
            f"{kwargs['kick_aim_nominal_distance_m']}"
        )
    if kwargs.get("kick_error_ball_to_target_sigma") is not None and kwargs["kick_error_ball_to_target_sigma"] <= 0.0:
        raise ValueError(
            f"{yaml_path}: kick_error_ball_to_target_sigma must be > 0.0 (it is squared into an "
            f"exp() denominator), got {kwargs['kick_error_ball_to_target_sigma']}"
        )
    _hold_min = kwargs.get("observation_hold_steps_min", 0)
    _hold_max = kwargs.get("observation_hold_steps_max", 0)
    if _hold_max < _hold_min:
        raise ValueError(
            f"{yaml_path}: observation_hold_steps_max ({_hold_max}) must be >= "
            f"observation_hold_steps_min ({_hold_min})"
        )

    # Same pair/ordering rule MultiSkillConfig's own parser enforces -- see critic_v_min's
    # docstring for why a half-set support is treated as a typo rather than honored.
    _vmin, _vmax = kwargs.get("critic_v_min"), kwargs.get("critic_v_max")
    if (_vmin is None) != (_vmax is None):
        raise ValueError(
            f"{yaml_path}: critic_v_min and critic_v_max must be set together "
            f"(got critic_v_min={_vmin}, critic_v_max={_vmax})"
        )
    if _vmin is not None and _vmax is not None and _vmax <= _vmin:
        raise ValueError(f"{yaml_path}: critic_v_max ({_vmax}) must be > critic_v_min ({_vmin})")

    return BallConfig(**kwargs)


@dataclass(frozen=True)
class SceneConfig:
    """Composition of scene assets for the simulator."""

    replicate_physics: bool = True
    """Whether to reuse physics properties across duplicated assets."""

    asset_root: str | None = None
    """Optional root directory for relative asset paths."""

    scene_files: Annotated[list[SceneFileConfig] | None, tyro.conf.Suppress] = None
    """List of scene files (USD/URDF) to load. Set programmatically, not via CLI."""

    rigid_objects: Annotated[list[RigidObjectConfig] | None, tyro.conf.Suppress] = None
    """Standalone rigid objects to instantiate. Set programmatically, not via CLI."""

    ball: BallConfig | None = None
    """Optional procedurally-spawned kickable ball. None means no ball is added to the scene."""

    env_spacing: float = 20.0
    """Distance between parallel environments in the grid layout."""


@dataclass(frozen=True)
class VirtualGantryCfg:
    """Configuration parameters for virtual gantry system."""

    enabled: bool = False
    """Whether to enable the virtual gantry system."""

    attachment_body_names: list[str] = field(
        default_factory=lambda: ["Trunk", "torso_link", "torso", "base_link", "pelvis", "base"]
    )
    """List of body names to try for attachment (in preference order)."""

    stiffness: float = 200.0
    """Spring stiffness coefficient for elastic band force calculation."""

    damping: float = 100.0
    """Damping coefficient for velocity-based force damping."""

    height: float = 3.0
    """Default height for gantry anchor point in world coordinates."""

    point: list[float] | None = None
    """3D position of gantry anchor point [x, y, z]. If None, defaults to [0, 0, height]."""

    length: float = 0.0
    """Rest length of the elastic band (zero force distance)."""

    apply_force: float = 0.0
    """Additional force magnitude to apply (for manual force adjustment)."""

    apply_force_sign: int = -1
    """Sign multiplier for apply_force direction (-1 or 1)."""


@dataclass(frozen=True)
class BridgeConfig:
    """Configuration for robot SDK bridge integration.

    This configuration matches the parameters used in holosoma_inference's BaseSimulator
    for robot SDK communication and control.
    """

    enabled: bool = False
    """Whether to enable the bridge."""

    # Core bridge settings (from holosoma_inference BaseSimulator)
    use_joystick: bool = False
    """Whether to enable joystick/wireless controller support."""

    joystick_device: int = 0
    """Joystick device ID (Linux only)."""

    joystick_type: str = "xbox"
    """Type of joystick controller."""

    # SDK connection settings
    domain_id: int = 0
    """Domain ID for robot communication."""

    interface: str | None = None
    """Network interface for robot communication. Auto-detected if None."""

    # Rate limiting
    rate_limit_dt: float | None = None
    """Rate limiting timestep. If None, uses simulation timestep."""

    # ROS settings
    use_ros: bool = False
    """Whether to use ROS for communication."""


@dataclass(frozen=True)
class SimulatorInitConfig:
    """Top-level simulator initialisation configuration."""

    name: str
    """Name of the simulator backend (e.g. ``isaacgym``)."""

    sim: SimEngineConfig
    """Simulation engine configuration settings."""

    debug_viz: bool = True
    """Enable debug visualization (gantry lines, etc.)."""

    viewer: ViewerConfig = field(default_factory=ViewerConfig)
    """Interactive viewer camera configuration.

    Configures camera tracking for the interactive viewer with advanced features:
    - Multiple camera modes (Fixed, Spherical, Cartesian)
    - Camera smoothing for stable viewing
    - Robot tracking with configurable body attachment

    Example:
        viewer=ViewerConfig(
            enabled=True,
            camera=SphericalCameraConfig(
                distance=3.0,
                azimuth=45.0,
                elevation=30.0,
            ),
        )
    """

    scene: SceneConfig = field(default_factory=SceneConfig)
    """Scene composition and asset configuration."""

    reset_manager: ResetManagerConfig = field(default_factory=ResetManagerConfig)
    """Reset event manager configuration."""

    contact_sensor_history_length: int = 3
    """Number of frames of contact data retained for sensors."""

    robot_mjcf_filter: MujocoXMLFilterCfg = field(default_factory=MujocoXMLFilterCfg)
    """MuJoCo-specific XML filtering configuration for robot MJCF files."""

    mujoco_backend: MujocoBackend = MujocoBackend.CLASSIC
    """MuJoCo physics backend selection.

    Determines which MuJoCo backend to use for physics simulation:
    - 'classic': CPU-based single environment (backward compatible, default)
    - 'warp': GPU-accelerated multi-environment with mujoco_warp

    This setting only applies when using the MuJoCo simulator (name='mujoco').
    For other simulators (isaacgym, isaacsim), this field is ignored.

    Command line usage:
        --simulator.config.mujoco-backend=warp
        --simulator.config.mujoco-backend=classic

    Or use the syntactic sugar configs:
        simulator:mujoco   (uses classic backend)
        simulator:mjwarp   (uses warp backend)
    """

    mujoco_warp: MujocoWarpConfig = field(default_factory=MujocoWarpConfig)
    """MuJoCo Warp backend memory allocation configuration.

    Controls GPU memory allocation for the Warp backend. Only used when
    mujoco_backend='warp'. Allows tuning contact and constraint capacity
    for different scenarios.

    Command line usage:
        --simulator.config.mujoco-warp.nconmax-per-env=128
        --simulator.config.mujoco-warp.njmax-per-env=1024
    """

    bridge: BridgeConfig = field(default_factory=BridgeConfig)
    """Robot SDK bridge configuration."""

    virtual_gantry: VirtualGantryCfg = field(default_factory=VirtualGantryCfg)
    """Virtual gantry system configuration."""


@dataclass(frozen=True)
class SimulatorConfig:
    """Wrapper for simulator instantiation."""

    _target_: str
    """Fully-qualified simulator factory target."""

    _recursive_: bool
    """Recursive instantiation flag."""

    config: SimulatorInitConfig
    """Structured simulator configuration passed to the factory."""
