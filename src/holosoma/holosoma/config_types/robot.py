from __future__ import annotations

from dataclasses import field

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class RobotBridgeConfig:
    """Bridge-specific configuration for robot SDK communication.

    Currently supports sim2sim (holosoma/run_sim.py) only.
    """

    sdk_type: str = "unitree"
    """SDK type for robot communication ('unitree', 'booster', 'ros2')."""

    motor_type: str = "serial"
    """Motor communication type ('serial', etc.)."""


@dataclass(frozen=True)
class RobotInitState:
    pos: list[float]
    rot: list[float]
    lin_vel: list[float]
    ang_vel: list[float]
    default_joint_angles: dict[str, float]


@dataclass(frozen=True)
class RobotControlConfig:
    control_type: str
    stiffness: dict[str, float]
    damping: dict[str, float]
    action_scale: float
    action_clip_value: float
    clip_actions: bool
    clip_torques: bool
    action_scales_by_effort_limit_over_p_gain: bool = False
    per_joint_action_clip: dict[str, tuple[float, float]] | None = None
    """2026-08-05, ported from RoboNaldo (arXiv:2606.11092)'s per-joint ``clip={}`` dict on its
    ``JointPositionActionCfg`` (see ROBONALDO_PORT_SCOPE.md Sec 4). Maps joint TYPE (substring,
    same matching convention as ``stiffness``/``damping`` above -- e.g. ``"ankle_roll"`` matches
    both ``left_ankle_roll_joint`` and ``right_ankle_roll_joint``) -> (low, high) RAW policy-output
    clip bounds (applied BEFORE ``action_scale``, same point in the pipeline as the existing scalar
    ``action_clip_value``), overriding that scalar for the matched joints only -- any joint that
    matches no entry still uses ``(-action_clip_value, action_clip_value)`` unchanged. None
    (default, or an empty dict) = current behavior, exact no-op: every joint uses the scalar.

    IMPORTANT, verified by computation before staging any values (do not skip this when adding
    more joints): a raw clip bound is only physically meaningful relative to THAT joint's own
    ``action_scale`` (with ``action_scales_by_effort_limit_over_p_gain=True``, this project's own
    per-joint scale = ``action_scale * effort_limit / stiffness``, NOT a shared constant across
    joint types). RoboNaldo's own raw clip NUMBERS are calibrated to THEIR per-joint scales and do
    NOT transfer verbatim -- e.g. their uniform +/-3.0 arm/wrist clip, applied to THIS project's
    own computed shoulder_yaw/wrist_roll scale (~0.44 rad/unit, effort=25/stiffness=14.25), yields
    an effective +/-1.3 rad (~75 deg) range -- nowhere near tight enough to meaningfully constrain
    the measured 40-53 deg strike-phase drift on exactly those joints (see
    ``MotionStrikeDofPosErrorExp``'s own docstring for those numbers). That drift is
    ``ArmDefaultPose``'s job (reward-side), not this field's.

    What DOES verifiably transfer, because this project's G1 spec matches RoboNaldo's own hardware
    exactly (confirmed: identical URDF joint limits for every joint checked) and their clip values
    for these 3 specific joints are themselves derived as ~(URDF joint limit / their own per-joint
    scale) -- a physically-motivated bound, not an arbitrary one:
    - ``ankle_roll`` (both feet): URDF limit +/-0.2618 rad, this project's own computed scale
      ~0.4386 rad/unit (effort=50/stiffness=28.50) -> clip +/-0.6 reproduces RoboNaldo's own value
      exactly via this project's own numbers, not by copying theirs.
    - ``waist_roll``/``waist_pitch``: URDF limit +/-0.52 rad, same ~0.4386 rad/unit scale (effort=
      50/stiffness=28.50) -> clip +/-1.2, same exact-match derivation.

    None (default) is a true no-op. See ``config_values/unified/g1/experiment.py`` for the staged,
    reasoned-but-UNVALIDATED starting values for these 3 joints specifically -- same "ship inert,
    validate via a real training run" discipline as every other new mechanism in this project."""


@dataclass(frozen=True)
class RobotAssetConfig:
    asset_root: str
    collapse_fixed_joints: bool
    replace_cylinder_with_capsule: bool
    flip_visual_attachments: bool
    armature: float
    thickness: float
    max_angular_velocity: float
    max_linear_velocity: float
    angular_damping: float
    linear_damping: float
    urdf_file: str
    usd_file: str | None
    xml_file: str
    robot_type: str
    enable_self_collisions: bool
    default_dof_drive_mode: int
    fix_base_link: bool
    mesh_root: str | None = None
    density: float | None = None
    disable_gravity: bool | None = None
    contact_offset: float | None = None
    """PhysX collision contact_offset (m), applied via IsaacLab's CollisionPropertiesCfg at spawn
    time. None leaves it at the USD-imported/PhysX engine default (unconfigured anywhere in
    holosoma prior to 2026-07-17). NOTE (2026-07-18): despite being wired at the whole-robot level,
    this silently does NOT reach every body -- modify_collision_properties's apply_nested wrapper
    skips USD *instanced* prims without erroring, and on G1 that includes both feet
    (left/right_ankle_roll_link) plus ~10 other bodies (knees, hip-rolls, torso, some
    wrists/shoulders). A feet-only rescope was attempted and found to be blocked by the same USD
    instancing constraint even more directly (hard error, not a silent skip); reverted back to
    this whole-body form since it at least has a measured positive effect on SOME bodies. See
    memory stagec-kick-contact-offset-rest-offset-fix ("MAJOR CORRECTION" section) before trusting
    this actually affects the feet. Only read by the isaacsim simulator backend."""
    rest_offset: float | None = None
    """PhysX collision rest_offset (m) -- see contact_offset's docstring above, same caveats
    (whole-robot in principle, but silently skips instanced bodies including both feet). Only read
    by the isaacsim simulator backend."""


@dataclass(frozen=True)
class RobotForceControlConfig:
    apply_force_link: list[str] | None = None
    left_hand_link: str | None = None
    right_hand_link: str | None = None


@dataclass(frozen=True)
class ObjectConfig:
    object_urdf_path: str | None = None


@dataclass(frozen=True)
class RobotConfig:
    num_bodies: int
    dof_obs_size: int
    algo_obs_dim_dict: dict[str, int]
    actions_dim: int
    policy_obs_dim: int
    critic_obs_dim: int
    contact_pairs_multiplier: int
    key_bodies: list[str]
    num_feet: int
    foot_body_name: str
    """Name/pattern of the real foot link(s) used for contacts and kinematics."""
    foot_height_name: str
    """Name/pattern of auxiliary 'fake' foot link(s) used only to compute foot height/clearance"""
    knee_name: str
    torso_name: str
    dof_names: list[str]
    upper_dof_names: list[str]
    upper_left_arm_dof_names: list[str]
    upper_right_arm_dof_names: list[str]
    lower_dof_names: list[str]
    has_torso: bool
    has_upper_body_dof: bool
    left_ankle_dof_names: list[str]
    right_ankle_dof_names: list[str]
    knee_dof_names: list[str]
    hips_dof_names: list[str]
    dof_pos_lower_limit_list: list[float]
    dof_pos_upper_limit_list: list[float]
    dof_vel_limit_list: list[float]
    dof_effort_limit_list: list[float]
    dof_armature_list: list[float]
    dof_joint_friction_list: list[float]
    body_names: list[str]
    terminate_after_contacts_on: list[str]
    penalize_contacts_on: list[str]
    init_state: RobotInitState
    randomize_link_body_names: list[str]

    control: RobotControlConfig
    asset: RobotAssetConfig

    # TODO(jchen): talk to SAM, merge this into scene config
    object: ObjectConfig = field(default_factory=ObjectConfig)

    bridge: RobotBridgeConfig = field(default_factory=RobotBridgeConfig)
    """Bridge SDK configuration for this robot."""

    waist_dof_names: list[str] | None = None
    waist_yaw_dof_name: str | None = None
    waist_roll_dof_name: str | None = None
    waist_pitch_dof_name: str | None = None

    arm_dof_names: list[str] | None = None
    left_arm_dof_names: list[str] | None = None
    right_arm_dof_names: list[str] | None = None

    symmetry_joint_names: dict[str, str] | None = None
    flip_sign_joint_names: list[str] | None = None

    apply_dof_armature_in_isaacgym: bool = True
    knee_joint_min_threshold: float = 0.2
    lidar_height_offset: float = 0.5

    soft_dof_pos_limit: float = 0.95
    termination_close_to_dof_pos_limit: float = 0.98
