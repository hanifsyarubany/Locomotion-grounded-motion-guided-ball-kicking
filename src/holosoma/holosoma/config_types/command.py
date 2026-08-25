"""Configuration types for the command & curriculum manager."""

from __future__ import annotations

from dataclasses import field
from pathlib import Path
from typing import Any

import yaml
from pydantic.dataclasses import dataclass

from holosoma.config_types.multi_skill import SkillConfig


@dataclass(frozen=True)
class CommandTermCfg:
    """Configuration for a single command or curriculum hook."""

    func: str
    """Import path for the command hook (function or callable class)."""

    params: dict[str, Any] = field(default_factory=dict)
    """Additional parameters forwarded to the hook."""

    task_mode: str | None = None
    """If set, only consulted for entries registered under ``reset_terms``: the term's ``reset()``
    is only called with the subset of ``env_ids`` currently in this task mode (per
    ``env.task_mode_mask(task_mode)``, only consulted when the env implements it — e.g.
    UnifiedManager). Has no runtime effect for ``setup_terms``/``step_terms`` entries — those
    hooks receive no ``env_ids`` at all, so there is nothing to filter (see CommandManager.step()).
    ``None`` (the default) means always active, matching every existing experiment's behavior
    exactly."""


@dataclass(frozen=True)
class CommandManagerCfg:
    """Configuration for the command manager."""

    params: dict[str, Any] = field(default_factory=dict)
    """Global parameters shared across command hooks."""

    setup_terms: dict[str, CommandTermCfg] = field(default_factory=dict)
    """Hooks invoked during environment setup."""

    reset_terms: dict[str, CommandTermCfg] = field(default_factory=dict)
    """Hooks invoked on environment reset."""

    step_terms: dict[str, CommandTermCfg] = field(default_factory=dict)


########################################################################################################################
# Motion command configuration
########################################################################################################################
@dataclass(frozen=True)
class NoiseToInitialPoseConfig:
    """Initial pose of the robot and object to those in the motion file."""

    overall_noise_scale: float = 0.0
    """Overall noise scale for the initial pose."""

    dof_pos: float = 0.0
    """Noise scale for the initial dof position."""

    root_pos: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """noise scale for root position x, y, z."""

    root_rot: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """noise scale for root rotation roll, pitch, yaw."""

    root_lin_vel: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """noise scale for root linear velocity vx, vy, vz."""

    root_ang_vel: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """noise scale for root angular velocity wx, wy, wz."""

    object_pos: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """noise scale for object position x, y, z."""


@dataclass(frozen=True)
class MotionConfig:
    """Motion related configuration for Whole Body Tracking.

    NOTE:
    - Motion file is assumed to be in the format of:
      - joint_pos: (T, J)
      - joint_vel: (T, J)

      - body_pos_w: (T, B, 3)
      - body_quat_w: (T, B, 4) # wxyz -> xyzw
      - body_lin_vel_w: (T, B, 3)
      - body_ang_vel_w: (T, B, 3)

      If object is present in the motion file, it is assumed to be in the format of:
      - object_pos_w: (T, 3)
      - object_quat_w: (T, 4)
      - object_lin_vel_w: (T, 3)
      - object_ang_vel_w: (T, 3)

      If the motion clip assumes a terrain, the terrain has to be specified in holosoma/config/terrain/terrain_wbt.yaml
    """

    motion_file: str
    """Motion file (.npz) that contains motion_clips to track. """

    body_name_ref: list[str]
    """Body name of the reference frame (in general, torso_link). """
    body_names_to_track: list[str]
    """Key body names to track, used for reward/termination computation."""

    motion_dir: str = ""
    """Directory (or comma-separated directories) of .npz motion files, loaded via
    MultiMotionLoader.from_dir (alphabetical glob order). When non-empty, takes precedence over
    motion_file. Superseded by motion_files when that's also set."""

    motion_files: list[str] = field(default_factory=list)
    """Explicit, ORDERED list of per-skill .npz paths -- index i becomes motion_id i, unlike
    motion_dir's alphabetical glob. Takes precedence over both motion_dir and motion_file. This is
    what a stacked N-motion-skill yaml (motion_skill_1, motion_skill_2, ...) wires in, so
    declaration order maps deterministically onto motion_ids."""

    motion_prepend_duration_s: list[float] = field(default_factory=list)
    """Per-motion windup duration (seconds), index-aligned with motion_files. Empty (default)
    broadcasts default_pose_prepend_duration_s to every motion -- bit-identical to the old single-
    scalar behavior, including for a single motion. Must be either empty or exactly length
    len(motion_files)."""

    motion_recovery_duration_s: list[float] = field(default_factory=list)
    """Per-motion post-clip recovery-transition duration (seconds), index-aligned with
    motion_files. Empty (default) broadcasts default_pose_append_duration_s to every motion."""

    motion_hold_duration_s: list[float] = field(default_factory=list)
    """Per-motion static-hold duration (seconds) after that motion's own recovery transition,
    index-aligned with motion_files. Empty (default) broadcasts post_transition_hold_duration_s to
    every motion."""

    motion_strike_start_frame: list[int] = field(default_factory=list)
    """Per-motion strike_start_frame (see SkillConfig), index-aligned with motion_files. Unlike
    motion_recovery_duration_s/motion_hold_duration_s, empty does NOT broadcast a scalar default --
    there is none. Empty means legacy/single-clip mode (in_kicking_phase/in_strike_phase fall back
    to their pre-2026-07-31 no-op behavior); non-empty must be exactly length len(motion_files)."""

    motion_stand_start_frame: list[int] = field(default_factory=list)
    """Per-motion stand_start_frame (see SkillConfig). Same no-broadcast semantics as
    motion_strike_start_frame above."""

    skill_ball_configs: list[SkillConfig] = field(default_factory=list)
    """Per-motion ball spawn/target/randomization/kick_foot/success_radius, index-aligned with
    motion_files -- one SkillConfig per motion skill (see config_types/multi_skill.py). Empty
    (default) means "not in N-skill mode": ball reset behavior falls back to the single global
    BallConfig on simulator_config.scene.ball, exactly as before. shooting_reward_scale on each
    SkillConfig here is also what makes that skill's reward Stage B (0.0) vs Stage C (>0.0) -- see
    managers/reward/terms/shooting.py and utils/shooting_curriculum.py."""

    ood_spawn_probability: float = 0.0
    """2026-07-24 deployment-robustness training: per-RESET probability (not per-skill, applies
    uniformly regardless of which skill an env is running) that the ball spawns in an
    OUT-OF-DISTRIBUTION region instead of the normal (x +/- randomize_x, y +/- randomize_y) box --
    e.g. behind the robot, too close, too far. Rationale: at deployment the ball can end up
    somewhere the training distribution never covered (bad detection, an unusual game situation),
    and the policy should stay stable (even if it misses the kick) rather than risk falling while
    reaching for it. Training WITH occasional OOD draws is what makes an "OOD" observation at
    deployment no longer novel to the network -- see ball_pos_b's own docstring and
    `stagec_obs_normalizer_shock.md` for why an unseen-scale input, not a bad reward incentive, is
    the actual destabilization mechanism this targets (confirmed precedent: the kick_alive
    eval-harness-artifact investigation, where an OOD heading BEARING alone -- no reward or
    contact difference -- was sufficient to flip topple rate from 32% to 100%). 0.0 (default) =
    off, bit-identical to before this field existed.

    2026-08-01 REVERSAL of an earlier decision: this field originally left the shooting reward
    untouched for OOD-spawn episodes (user directive, 2026-07-24), reasoning that
    ``ball_proximity``'s ``exp(-dist/sigma)`` and the ``has_kicked``-gated terms already decay to
    ~zero gradient for an unreachable ball. Revisited after realizing ``ood_region_multiplier``'s
    own draw is NOT rejection-sampled away from the normal box (see that field's own docstring) --
    a plain uniform draw over the wider OOD region can coincidentally land back inside the normal,
    reachable box, so a minority of "OOD" attempts were paying inconsistent partial shooting
    reward, noise the reward signal shouldn't have. All 6 shooting reward terms
    (``managers/reward/terms/shooting.py``) now explicitly multiply by the complement of
    ``MotionCommand.is_ood_spawn`` (see that buffer's own docstring and shooting.py's
    ``_ood_gate_multiplier``) -- zeroed for every attempt whose ball spawn was OOD-drawn,
    in-distribution attempts fully unaffected. 0.0 (default) keeps ``is_ood_spawn`` all-False
    forever, so this is a verified no-op wherever the field is unset."""

    ood_region_multiplier: float = 3.0
    """How far the OOD region (see ``ood_spawn_probability``) extends, as a multiple of each
    skill's own ``randomize_x``/``randomize_y`` half-range -- e.g. 3.0 means the OOD draw is
    uniform over [-3*randomize_x, 3*randomize_x] (independently for x and y, NOT rejection-sampled
    to guarantee landing outside the normal box -- a simple, GPU-friendly uniform draw that lands
    outside the normal box on at least one axis the large majority of the time is judged
    sufficient; occasional draws that coincidentally land back in the normal region are harmless
    since OOD episodes are already a small minority). Scales naturally per-skill since it's
    relative to each skill's own existing range, rather than a fixed absolute bound. Only
    meaningful when ``ood_spawn_probability > 0``."""

    kick_aim_theta_ref_deg: float = 45.0
    """Global mirror of MultiSkillConfig.kick_aim_theta_ref_deg / BallConfig.kick_aim_theta_ref_deg
    -- see either field's own docstring. Resolved once in config_values/unified/g1/command.py's
    dual-path pattern (same as ood_spawn_probability above) and threaded through here since it has
    no per-skill counterpart -- it's the FIXED observation-normalization constant, deliberately
    the same for every skill regardless of any skill's own kick_aim_theta_max_deg."""

    kick_aim_theta_max_deg: float = 15.0
    """Global DEFAULT mirror of MultiSkillConfig.kick_aim_theta_max_deg / BallConfig.
    kick_aim_theta_max_deg -- used by any kick_aim_enabled skill whose OWN SkillConfig.
    kick_aim_theta_max_deg is None (see that field's own docstring). Per-skill overrides don't
    need a _per_motion table here the way other fields in this class do -- they're already
    available directly off each skill's own SkillConfig in skill_ball_configs, which
    MotionCommand.setup() reads from alongside this global default."""

    kick_aim_nominal_distance_m: float = 5.0
    """Global mirror of MultiSkillConfig.kick_aim_nominal_distance_m / BallConfig.
    kick_aim_nominal_distance_m -- see either field's own docstring. Same no-per-skill-table
    rationale as kick_aim_theta_ref_deg above: every kick_aim_enabled skill synthesizes its target
    at the SAME fixed distance, by design (keeps metre-denominated reward terms comparable across
    skills)."""

    # motion sampling related
    use_adaptive_timesteps_sampler: bool = False
    """During training, whether to prioritize training on motion segments where the robot fails often."""

    start_at_timestep_zero_prob: float = 0.0
    """Probability of starting at timestep zero."""

    start_at_timestep_zero_prob_per_motion: list[float] = field(default_factory=list)
    """2026-08-15, "simultaneous per-skill task configs" -- per-motion override of
    start_at_timestep_zero_prob above, index-aligned with motion_files. Empty (default, the
    common case) = no override, MotionCommand.reset() uses the plain scalar unchanged, INCLUDING
    its prob>=1.0 fast path that skips the torch.rand draw entirely -- true byte-identical no-op,
    not just numerically equivalent (RNG-consumption order matters here, unlike a plain value
    field). Non-empty must be exactly length len(motion_files); resolved by
    config_values/unified/g1/command.py from each skill's own task_config file (a training-regime
    choice, NOT per-clip content -- unlike motion_head_velocity_smoothing_frames_per_motion above,
    this does NOT live on SkillConfig)."""

    rsi_scope_to_authored_clip: bool = False
    """When start_at_timestep_zero_prob < 1.0, whether the uniform-phase RSI draw's span is
    clamped to pre_recovery_motion_end_idx (authored clip content only) instead of the whole
    augmented buffer (including the synthetic recovery/hold tail). See
    MultiSkillConfig.rsi_scope_to_authored_clip's own docstring for the full rationale. False
    (default) = current behavior, exact no-op."""

    rsi_scope_to_authored_clip_per_motion: list[bool] = field(default_factory=list)
    """2026-08-15, "simultaneous per-skill task configs" -- per-motion override of
    rsi_scope_to_authored_clip above, index-aligned with motion_files. Empty (default) = no
    override, reset() uses the plain scalar. Non-empty must be exactly length len(motion_files).
    Same task_config-file-resolved, not-per-clip-content convention as
    start_at_timestep_zero_prob_per_motion above."""

    critical_frame_oversampling_prob: float = 0.0
    """Per-reset probability of drawing the RSI phase from a fixed window around strike_start_idx
    instead of the plain uniform draw. See MultiSkillConfig.critical_frame_oversampling_prob's own
    docstring. 0.0 (default) = exact no-op."""

    critical_frame_oversampling_prob_per_motion: list[float] = field(default_factory=list)
    """2026-08-15, "simultaneous per-skill task configs" -- per-motion override of
    critical_frame_oversampling_prob above, index-aligned with motion_files. Empty (default) = no
    override. Unlike start_at_timestep_zero_prob_per_motion, no RNG-consumption-order caveat:
    critical_frame_oversample_time_steps always draws both torch.rand calls regardless of the
    probability value (see that function's own docstring), so it's already elementwise-safe for a
    per-env tensor with no special-casing needed."""

    critical_frame_sampling_window: int = 10
    """Half-width (frames) of the critical_frame_oversampling_prob window. See
    MultiSkillConfig.critical_frame_sampling_window's own docstring."""

    critical_frame_sampling_window_per_motion: list[int] = field(default_factory=list)
    """2026-08-15, "simultaneous per-skill task configs" -- per-motion override of
    critical_frame_sampling_window above, index-aligned with motion_files. Empty (default) = no
    override."""

    mid_episode_kick_entry_ball_fixed: bool = False
    """Mirrors MultiSkillConfig.mid_episode_kick_entry_ball_fixed's own docstring -- resolved onto
    MotionConfig (unlike UnifiedManager's other pre_kick_*/mid_episode_kick_entry_* fields, which
    are read later, per-call, directly from command_cfg) because MotionCommand.setup() needs it
    AT TABLE-BUILD TIME, before UnifiedManager exists to hand it over: _build_entry_search_table
    is called once from setup(), and whether it builds a 5-column (gait-only) or 7-column
    (gait + ball-geometry) table is a setup-time structural decision, not a per-tick one. False
    (default) = exact no-op, same as every other field here.

    2026-08-15: deliberately has NO per-skill counterpart, unlike its siblings above. The table
    this field's value decides the WIDTH of (5 vs 7 columns) is a SINGLE SHARED tensor across
    every motion (`table = torch.zeros(num_motions, max_len, num_feats, ...)` in
    _build_entry_search_table) -- there is no per-motion column count to vary. Making this
    genuinely per-skill would mean either building separate variable-width tables per skill, or
    always allocating 7 columns and zero-padding the ball-geometry columns for skills that don't
    want them (a real semantics/architecture question, not a config-plumbing one) -- out of scope
    for the config-level per-skill mechanism the other fields in this file use."""

    freeze_at_timestep_zero_prob: float = 0.0
    """When starting at timestep 0, probability of freezing motion counter at 0 (not advancing).
    This makes the robot practice holding the initial pose. Only applies when episode starts at timestep 0.
    Sampled independently each policy step; expected wait is roughly 1 / (1 - p) steps before unfreezing."""

    enable_default_pose_prepend: bool = False
    """If True, pre-append interpolated frames from default pose to the motion's first pose.
    This provides a smooth transition trajectory that the policy can track."""

    default_pose_prepend_duration_s: float = 2.0
    """Duration in seconds of the pre-appended interpolation phase.
    Only used if enable_default_pose_prepend is True."""

    motion_head_velocity_smoothing_frames: int = 0
    """Number of leading frames of each motion's OWN authored clip whose velocity channels
    (joint_vel/body_lin_vel_w/body_ang_vel_w) are ramped up from zero, in memory, at setup time --
    see managers/command/terms/motion_head_velocity_smoothing.py's own module docstring for the
    measured motivation (video_012's frame 0 is the highest-velocity frame of all 250, 7.6x the
    clip's own median and higher than the kick strike itself, decaying to below-median by frame 3
    -- a pose-estimation boundary artifact the 1.0s prepend currently accelerates straight into).

    0 (default) = exact no-op, every existing config unaffected. 3 is the value the video_012
    measurement points at (frames 0/1/2 sit above the clip's own p95; frame 3 is already below its
    median). Applied per-motion BEFORE any prepend is spliced in, so the prepend interpolates
    toward the SMOOTHED frame 0 rather than the raw one -- that ordering is the whole point and is
    asserted by this field's own tests. Positions are deliberately not touched; see that module's
    docstring for why, and for the kinematic-inconsistency caveat that comes with it."""

    motion_head_velocity_smoothing_frames_per_motion: list[int] = field(default_factory=list)
    """2026-08-15, "simultaneous per-skill task configs" -- per-motion override of
    motion_head_velocity_smoothing_frames above, index-aligned with motion_files, same broadcast
    convention as motion_prepend_duration_s/motion_recovery_duration_s/motion_hold_duration_s
    (see each field's own docstring): empty (default) broadcasts the single scalar
    motion_head_velocity_smoothing_frames to every motion -- bit-identical to before this field
    existed, including for a single motion. Non-empty must be exactly length len(motion_files).

    Each skill's OWN video2robot clip has its own head-velocity-spike profile (this is a
    pose-estimation-boundary artifact of a SPECIFIC clip, not a shared training-regime property),
    so a skill on a clean clip can leave this unset while a noisier one sets 3 -- built by
    config_values/unified/g1/command.py from each skill's OWN SkillConfig.motion_head_velocity_
    smoothing_frames field (declared directly on that skill's motion_skill_N yaml block, alongside
    its other clip-authoring metadata like strike_start_frame/kick_foot), NOT via a per-skill
    task_config file the way reward/termination fields are -- see that SkillConfig field's own
    docstring for why. Unlike the reward/termination mechanisms, this is also a COMPILE-TIME
    per-clip preprocessing param (applied once at MotionCommand.setup(), before any env exists),
    not a runtime env.skill_id gather, so it has no weight_per_skill/params_per_skill
    counterpart."""

    enable_default_pose_append: bool = False
    """If True, post-append interpolated frames from the motion's last pose back to default pose.
    This provides a smooth return trajectory that the policy can track."""

    default_pose_append_duration_s: float = 2.0
    """Duration in seconds of the post-appended interpolation phase.
    Only used if enable_default_pose_append is True."""

    post_transition_hold_duration_s: float = 0.0
    """Duration in seconds of a genuinely static hold (zero velocity, fixed default pose) appended
    immediately after the default_pose_append transition. Unlike the transition itself (which
    continuously interpolates toward the default pose over its full duration), this holds the
    robot at a single fixed reference pose — a sustained balance/stability test rather than a
    moving target. Only used if enable_default_pose_append is True; ignored otherwise."""

    # noise related
    noise_to_initial_pose: NoiseToInitialPoseConfig = field(default_factory=NoiseToInitialPoseConfig)


@dataclass(frozen=True)
class StabilizationConfig:
    """Post-kick stabilization timing: recover from the motion clip's ending pose, then hold a
    static stance. Maps onto MotionConfig's enable_default_pose_append mechanism —
    recovery_duration_s becomes default_pose_append_duration_s (a moving transition back to the
    default pose) and hold_duration_s becomes post_transition_hold_duration_s (a genuinely static
    hold at that pose, zero velocity)."""

    recovery_duration_s: float = 1.0
    """Duration in seconds of the smooth transition from the motion clip's ending pose back to the
    robot's default standing pose."""

    hold_duration_s: float = 2.0
    """Duration in seconds of the static balance hold immediately after the recovery transition.
    Total stabilization time is recovery_duration_s + hold_duration_s."""


# Fork root (.../ball_kicking_learning), not the holosoma package root, since this yaml is a
# user-facing project config, not a package-bundled resource under holosoma/data/.
DEFAULT_STABILIZATION_CONFIG_YAML = Path(__file__).resolve().parents[4] / "configs" / "stabilization.yaml"


def load_stabilization_config(yaml_path: str | Path = DEFAULT_STABILIZATION_CONFIG_YAML) -> StabilizationConfig:
    """Build a StabilizationConfig from a YAML file (recovery_duration_s, hold_duration_s — see
    configs/stabilization.yaml)."""
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Stabilization config YAML not found: {yaml_path}\n"
            "Expected a file with 'recovery_duration_s' and 'hold_duration_s' keys."
        )
    with open(yaml_path) as f:
        raw = yaml.safe_load(f) or {}

    required = {"recovery_duration_s", "hold_duration_s"}
    missing = required - raw.keys()
    if missing:
        raise ValueError(f"Stabilization config YAML {yaml_path} is missing required key(s): {sorted(missing)}")

    return StabilizationConfig(
        recovery_duration_s=float(raw["recovery_duration_s"]),
        hold_duration_s=float(raw["hold_duration_s"]),
    )


@dataclass(frozen=True)
class SkillMixConfig:
    """Per-episode skill mix for the unified locomotion + ball-kicking policy: what fraction of
    episodes (on envs eligible for it — see UnifiedManager.env_terrain_is_flat) run the
    ball-kicking task instead of locomotion."""

    kick_probability: float = 0.5
    """Probability [0, 1] that an eligible env's episode is assigned kick-mode at reset, instead
    of locomotion-mode. Only eligible (flat-terrain) envs can ever be assigned kick-mode, so the
    realized global kick fraction is capped at the fraction of envs on flat terrain, regardless of
    how high this value goes — see configs/skill_mix.yaml for the full explanation."""


DEFAULT_SKILL_MIX_CONFIG_YAML = Path(__file__).resolve().parents[4] / "configs" / "skill_mix.yaml"


def load_skill_mix_config(yaml_path: str | Path = DEFAULT_SKILL_MIX_CONFIG_YAML) -> SkillMixConfig:
    """Build a SkillMixConfig from a YAML file (kick_probability — see configs/skill_mix.yaml)."""
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Skill-mix config YAML not found: {yaml_path}\nExpected a file with a 'kick_probability' key."
        )
    with open(yaml_path) as f:
        raw = yaml.safe_load(f) or {}

    required = {"kick_probability"}
    missing = required - raw.keys()
    if missing:
        raise ValueError(f"Skill-mix config YAML {yaml_path} is missing required key(s): {sorted(missing)}")

    kick_probability = float(raw["kick_probability"])
    if not 0.0 <= kick_probability <= 1.0:
        raise ValueError(f"kick_probability must be in [0.0, 1.0], got {kick_probability}")

    return SkillMixConfig(kick_probability=kick_probability)
