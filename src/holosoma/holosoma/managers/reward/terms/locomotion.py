"""Reward terms for locomotion tasks.

These terms are migrated from LeggedRobotBase._reward_* methods to be
compatible with the reward manager system.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from holosoma.managers.observation.terms.locomotion import (
    base_forward_vector,
    get_base_ang_vel,
    get_base_lin_vel,
    get_projected_gravity,
    gravity_vector,
)
from holosoma.utils.kick_reward_scales import kick_recovery_posture_reward_scale
from holosoma.utils.rotations import (
    quat_apply,
    quat_rotate_batched,
    quat_rotate_inverse,
)
from holosoma.utils.safe_torch_import import torch

if TYPE_CHECKING:
    from holosoma.envs.locomotion.locomotion_manager import LeggedRobotLocomotionManager


def _expected_foot_height(phi: torch.Tensor, swing_height: float) -> torch.Tensor:
    """Expected foot height from gait phase using a cubic Bézier profile."""

    def cubic_bezier_interpolation(y_start: torch.Tensor, y_end: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        y_diff = y_end - y_start
        bezier = x**3 + 3 * (x**2 * (1 - x))
        return y_start + y_diff * bezier

    x = (phi + torch.pi) / (2 * torch.pi)
    stance = cubic_bezier_interpolation(torch.zeros_like(x), torch.full_like(x, swing_height), 2 * x)
    swing = cubic_bezier_interpolation(torch.full_like(x, swing_height), torch.zeros_like(x), 2 * x - 1)
    return torch.where(x <= 0.5, stance, swing)


# ================================================================================================
# Termination Rewards
# ================================================================================================


def termination(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Terminal reward/penalty for early termination (excluding timeouts).

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    return (env.reset_buf * ~env.time_out_buf).float()


# ================================================================================================
# Penalty Rewards
# ================================================================================================


def penalty_action_rate(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Penalize changes in actions between steps.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    actions = env.action_manager.action
    prev_actions = env.action_manager.prev_action
    return torch.sum(torch.square(prev_actions - actions), dim=1)


def penalty_orientation(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Penalize non-flat base orientation.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    projected = get_projected_gravity(env)
    return torch.sum(torch.square(projected[:, :2]), dim=1)


def penalty_feet_ori(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Penalize feet orientation deviation from flat.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    left_quat = env.simulator._rigid_body_rot[:, env.feet_indices[0]]
    gravity = gravity_vector(env)
    left_gravity = quat_rotate_inverse(left_quat, gravity, w_last=True)
    right_quat = env.simulator._rigid_body_rot[:, env.feet_indices[1]]
    right_gravity = quat_rotate_inverse(right_quat, gravity, w_last=True)
    return (
        torch.sum(torch.square(left_gravity[:, :2]), dim=1) ** 0.5
        + torch.sum(torch.square(right_gravity[:, :2]), dim=1) ** 0.5
    )


# ================================================================================================
# Limit Rewards
# ================================================================================================


def limits_dof_pos(env: LeggedRobotLocomotionManager, soft_dof_pos_limit: float = 0.95) -> torch.Tensor:
    """Penalize joint positions too close to limits.

    Args:
        env: The environment instance
        soft_dof_pos_limit: Soft limit as fraction of hard limit

    Returns:
        Reward tensor [num_envs]
    """
    # Use soft limits as fraction of hard limits
    m = (env.simulator.hard_dof_pos_limits[:, 0] + env.simulator.hard_dof_pos_limits[:, 1]) / 2  # type: ignore[attr-defined]
    r = env.simulator.hard_dof_pos_limits[:, 1] - env.simulator.hard_dof_pos_limits[:, 0]  # type: ignore[attr-defined]
    lower_soft_limit = m - 0.5 * r * soft_dof_pos_limit
    upper_soft_limit = m + 0.5 * r * soft_dof_pos_limit

    out_of_limits = -(env.simulator.dof_pos - lower_soft_limit).clip(max=0.0)  # lower limit
    out_of_limits += (env.simulator.dof_pos - upper_soft_limit).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


# ================================================================================================
# Tracking and Task Rewards
# ================================================================================================


def tracking_lin_vel(env, tracking_sigma: float = 0.25) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes).

    Uses exponential reward: exp(-error / sigma)

    Args:
        env: The environment instance
        tracking_sigma: Sigma for exponential reward scaling

    Returns:
        Reward tensor [num_envs]
    """
    commands = env.command_manager.commands
    lin_vel_error = torch.sum(torch.square(commands[:, :2] - get_base_lin_vel(env)[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / tracking_sigma)


def tracking_ang_vel(env, tracking_sigma: float = 0.25) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw).

    Uses exponential reward: exp(-error / sigma)

    Args:
        env: The environment instance
        tracking_sigma: Sigma for exponential reward scaling

    Returns:
        Reward tensor [num_envs]
    """
    commands = env.command_manager.commands
    ang_vel = get_base_ang_vel(env)
    ang_vel_error = torch.square(commands[:, 2] - ang_vel[:, 2])
    return torch.exp(-ang_vel_error / tracking_sigma)


def penalty_ang_vel_xy(env) -> torch.Tensor:
    """Penalize xy axes base angular velocity.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    ang_vel = get_base_ang_vel(env)
    return torch.sum(torch.square(ang_vel[:, :2]), dim=1)


def penalty_close_feet_xy(env, close_feet_threshold: float = 0.05) -> torch.Tensor:
    """Penalize when feet are too close together in xy plane.

    Args:
        env: The environment instance
        close_feet_threshold: Minimum distance threshold between feet

    Returns:
        Reward tensor [num_envs]
    """
    left_foot_xy = env.simulator._rigid_body_pos[:, env.feet_indices[0], :2]
    right_foot_xy = env.simulator._rigid_body_pos[:, env.feet_indices[1], :2]

    # Get base orientation
    base_forward = quat_apply(env.base_quat, base_forward_vector(env), w_last=True)
    base_yaw = torch.atan2(base_forward[:, 1], base_forward[:, 0])

    # Calculate perpendicular distance in base-local coordinates
    feet_distance = torch.abs(
        torch.cos(base_yaw) * (left_foot_xy[:, 1] - right_foot_xy[:, 1])
        - torch.sin(base_yaw) * (left_foot_xy[:, 0] - right_foot_xy[:, 0])
    )

    # Return penalty when feet are too close
    return (feet_distance < close_feet_threshold).float()


def base_height(
    env, desired_base_height: float = 0.89, zero_vel_penalty_scale: float = 1.0, stance_penalty_scale: float = 1.0
) -> torch.Tensor:
    """Penalize base height away from target.

    Args:
        env: The environment instance
        desired_base_height: Target base height
        zero_vel_penalty_scale: Multiplier for base height penalty when robot has zero velocity commands
        stance_penalty_scale: Multiplier for base height penalty when robot is in stance mode

    Returns:
        Reward tensor [num_envs]
    """
    base_height_penalty = torch.square(
        env.terrain_manager.get_state("locomotion_terrain").base_heights - desired_base_height
    )

    # Apply stronger penalty for zero velocity commands if configured
    if zero_vel_penalty_scale != 1.0:
        commands = env.command_manager.commands
        zero_vel_mask = torch.norm(commands[:, :2], dim=1) < 0.1
        base_height_penalty = torch.where(
            zero_vel_mask, base_height_penalty * zero_vel_penalty_scale, base_height_penalty
        )

    # Apply stronger penalty for stance mode if configured (used in decoupled locomotion)
    if stance_penalty_scale != 1.0 and hasattr(env, "stance_mask"):
        base_height_penalty = torch.where(
            env.stance_mask, base_height_penalty * stance_penalty_scale, base_height_penalty
        )

    return base_height_penalty


def feet_phase(env, swing_height: float = 0.08, tracking_sigma: float = 0.25) -> torch.Tensor:
    """Reward for tracking desired foot height based on gait phase.

    Based on MuJoCo Playground's implementation.

    Args:
        env: The environment instance
        swing_height: Maximum height during swing phase
        tracking_sigma: Sigma for exponential reward scaling

    Returns:
        Reward tensor [num_envs]
    """
    # Get foot heights (relative to terrain)
    foot_z_left = env.terrain_manager.get_state("locomotion_terrain").feet_heights[:, 0]
    foot_z_right = env.terrain_manager.get_state("locomotion_terrain").feet_heights[:, 1]

    # Calculate expected foot heights based on phase
    gait_state = env.command_manager.get_state("locomotion_gait")
    rz_left = _expected_foot_height(gait_state.phase[:, 0], swing_height)
    rz_right = _expected_foot_height(gait_state.phase[:, 1], swing_height)

    # Calculate height tracking errors
    error_left = torch.square(foot_z_left - rz_left)
    error_right = torch.square(foot_z_right - rz_right)

    # Combine errors and apply exponential reward
    total_error = error_left + error_right

    return torch.exp(-total_error / tracking_sigma)


def pose(
    env,
    pose_weights: list[float],
) -> torch.Tensor:
    """Reward for maintaining default pose.

    Penalizes deviation from default joint positions with weighted importance.

    Args:
        env: The environment instance
        pose_weights: List of weights for each DOF (must match num_dof)

    Returns:
        Reward tensor [num_envs]
    """
    # Get current joint positions
    qpos = env.simulator.dof_pos

    # Convert pose_weights to tensor
    weights = torch.tensor(pose_weights, device=env.device, dtype=torch.float32)

    # Calculate squared deviation from default pose
    # Use env.default_dof_pos which is already set up from robot config
    pose_error = torch.square(qpos - env.default_dof_pos)

    # Weight and sum the errors
    weighted_error = pose_error * weights.unsqueeze(0)

    return torch.sum(weighted_error, dim=1)


# G1 dof_names order: left leg (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch,
# ankle_roll), then right leg in the same per-joint order — see
# config_values/robot.py's g1_29dof.dof_names. Mirror sign per joint: sagittal-plane
# joints (pitch-type: hip_pitch, knee, ankle_pitch) should match sign between legs in
# a symmetric double-stance; ab/adduction and rotation joints (hip_roll, hip_yaw,
# ankle_roll) should be opposite-signed (mirrored) between legs.
_LEFT_LEG_DOF_IDX = [0, 1, 2, 3, 4, 5]
_RIGHT_LEG_DOF_IDX = [6, 7, 8, 9, 10, 11]
_LEG_MIRROR_SIGN = [1.0, -1.0, -1.0, 1.0, 1.0, -1.0]


def _standing_gate(env, command_gate_sigma: float, grace_steps: float = 0.0) -> torch.Tensor:
    """Shared gate for the ``penalty_stand_*`` / standing-shaping family.

    The command gate (``exp(-|cmd|^2 / command_gate_sigma)``) makes these penalties full-strength
    at a zero velocity command and fades them out while walking, so they never fight the gait.

    ``grace_steps`` adds a second, multiplicative ramp that fades the penalties in linearly over
    the first ``grace_steps`` steps after the command goes to zero (via ``LocomotionGait``'s
    ``stand_steps`` counter). It exists because the command gate alone snaps to full strength the
    INSTANT the command hits zero, while the robot is still physically travelling -- and that was
    measurably causing the walk->stop falls:

    v9 Stage-A (15k steps), instant stop from 0.8 m/s: 7/10 gait phases FALL. The robot arrives at
    the stop still carrying ~0.65 m/s of forward momentum, but every standing penalty is already at
    full strength, demanding a square, symmetric, nominal-width stance. It complies -- fore-aft foot
    stagger collapses +0.18m -> -0.02m and stance asymmetry crushes 1.01 -> 0.04 within 0.4s, i.e.
    it pulls its trailing foot LEVEL instead of planting it ahead of the CoM. With both feet square
    under a CoM still moving forward there is no base of support ahead of the momentum, so it
    pitches over (lean 2.9 -> 9.9 -> 25.8 deg, vx re-accelerating 0.27 -> 0.85 m/s) and falls at
    ~1.8s. Arresting momentum REQUIRES a transiently staggered, asymmetric stance; a hard gate was
    paying the policy not to use one.

    The grace window keys on the COMMAND history (how long the command has been zero), never on the
    robot's own state. That distinction is load-bearing. A previous attempt gated on the robot's
    ACTUAL measured speed instead, reasoning that the penalties should relax "while still moving".
    Because the policy CONTROLS its own speed, it promptly learned to control the gate: v10 simply
    never stopped, drifting at a mean 0.64 m/s (12.6m in 20s) under a zero command, holding the gate
    at ~0.03 and suppressing all five standing penalties ~97% of the time. Any gate whose input the
    policy can manipulate is a reward hack waiting to happen. ``stand_steps`` cannot be manipulated:
    moving does not extend the window, so after ``grace_steps`` the penalties are at full strength
    regardless of what the robot is doing, and the only way to avoid them is to genuinely stand.

    ``grace_steps=0.0`` disables the ramp entirely (the original command-only behavior).
    """
    commands = env.command_manager.commands
    command_speed_sq = torch.sum(torch.square(commands[:, :2]), dim=1) + torch.square(commands[:, 2])
    gate = torch.exp(-command_speed_sq / command_gate_sigma)

    if grace_steps > 0.0:
        gait = env.command_manager.get_state("locomotion_gait")
        stand_steps = getattr(gait, "stand_steps", None) if gait is not None else None
        if stand_steps is not None:
            gate = gate * torch.clamp(stand_steps / grace_steps, max=1.0)

    return gate


def _kick_recovery_gate(env, grace_steps: float = 50.0) -> torch.Tensor:
    """Kick-mode counterpart to ``_standing_gate``: the shared gate for the
    ``penalty_kick_recovery_*`` family -- the SAME posture-shaping toolkit the locomotion-mode
    ``penalty_stand_*`` family already uses (see each pair's shared ``_stand_*_error``/
    ``_*_error`` helper above), ported into the post-kick recovery/hold tail. See
    ``config_values/unified/g1/reward.py``'s registration for the full rationale: the standing
    family is tagged ``task_mode="locomotion"`` and its own gate reads the locomotion velocity
    command, so a kick episode's recovery/hold phase -- which is also trying to settle into a
    stable standing pose -- gets none of it today, leaving posture during that phase governed
    only by the motion-tracking reward. This gate lets the identical error quantities apply
    there too, without touching the (already validated, already tuned) locomotion versions.

    Full strength once an env's current kick clip is past its "swinging mode" content
    (``motion_command.in_kicking_phase`` is False -- i.e. into post-kick-standing mode and the
    synthetic recovery-to-default-pose + static-hold tail appended after the clip, see
    ``MotionCommand.stand_start_idx``); zero during locomotion-approach + strike, so this never
    fights the kick's legitimate single-support lean, asymmetry, or height change.

    ``grace_steps`` fades the gate in LINEARLY over the first ``grace_steps`` steps after the
    swing ends, mirroring ``_standing_gate``'s own ``grace_steps`` rationale exactly: the robot
    is still carrying the kick's momentum in that literal instant recovery starts, and a hard
    snap to full strength would demand an already-square, already-symmetric stance before it's
    physically had a chance to settle. Keyed on ``time_steps`` -- each env's kick clip's own
    fixed per-env schedule, not the robot's own measured state -- for the identical
    non-manipulable-gate reason ``_standing_gate``'s docstring gives for keying its ramp on
    ``stand_steps`` rather than measured speed: a signal the policy can't influence can't be
    gamed into staying suppressed.

    Per-skill by construction, with no extra wiring: ``stand_start_idx`` and ``motion_ids`` are
    already per-motion tables (see ``MotionCommand.setup``), gathered per-env exactly as
    ``in_kicking_phase`` already does -- so this applies at the correct recovery/hold boundary for
    whichever skill each env is running, including any skill added later via
    ``stageB_and_C.yaml``.

    Returns an all-zero tensor (no shaping at all) for envs with no ball-kicking
    ``motion_command`` -- i.e. a locomotion-only env class, or ``has_ball=False`` -- so these
    terms are a safe, inert no-op if ever mis-registered outside a real kick task.
    """
    motion_command = env.command_manager.get_state("motion_command")
    if motion_command is None or not getattr(motion_command, "has_ball", False):
        return torch.zeros(env.num_envs, device=env.device)

    boundary_idx = motion_command.stand_start_idx[motion_command.motion_ids]
    in_recovery = ~motion_command.in_kicking_phase

    if grace_steps <= 0.0:
        return in_recovery.float()

    steps_into_recovery = torch.clamp((motion_command.time_steps - boundary_idx).float(), min=0.0)
    ramp = torch.clamp(steps_into_recovery / grace_steps, max=1.0)
    return torch.where(in_recovery, ramp, torch.zeros_like(ramp))


def _stance_asymmetry_error(env) -> torch.Tensor:
    """Left/right leg joint-angle asymmetry, squared and summed across the mirrored DOF set.
    Shared error computation for ``penalty_stance_asymmetry`` (locomotion, command-gated) and
    ``penalty_kick_recovery_stance_asymmetry`` (kick, phase-gated) — see each wrapper's own
    docstring for the gate that gives the quantity its meaning."""
    rel_dof_pos = env.simulator.dof_pos - env.default_dof_pos
    left = rel_dof_pos[:, _LEFT_LEG_DOF_IDX]
    right = rel_dof_pos[:, _RIGHT_LEG_DOF_IDX]
    mirror_sign = torch.tensor(_LEG_MIRROR_SIGN, device=env.device, dtype=torch.float32)
    return torch.sum(torch.square(left - mirror_sign * right), dim=1)


def penalty_stance_asymmetry(env, command_gate_sigma: float = 0.1, grace_steps: float = 0.0) -> torch.Tensor:
    """Penalize left/right leg asymmetry, strongest when the velocity command is near
    zero (fades out smoothly as commanded speed grows, so it doesn't fight normal
    stride asymmetry while actually walking).

    Root cause this addresses: with only ``pose``'s very low hip_pitch/knee weights
    (0.01, deliberately weak so walking strides aren't penalized) and ``feet_phase``
    only tracking foot *height* (not forward/backward position), nothing in training
    penalizes a policy that settles into a stable-but-asymmetric standing equilibrium
    (e.g. one leg bent forward, the other near default) — confirmed by direct
    left/right joint and foot-position comparison in eval: left hip_pitch/knee
    deviated ~0.27/-0.26 rad from default while right stayed near 0, producing a
    visibly asymmetric front-back foot placement at zero commanded velocity.

    Args:
        env: The environment instance
        command_gate_sigma: Sigma for the exponential command-magnitude gate — the
            penalty is near-full strength at zero command and decays as commanded
            speed grows.

    Returns:
        Reward tensor [num_envs] (positive error magnitude; pair with a negative
        weight, same convention as ``pose``).
    """
    return _stance_asymmetry_error(env) * _standing_gate(env, command_gate_sigma, grace_steps)


def penalty_kick_recovery_stance_asymmetry(env, grace_steps: float = 50.0) -> torch.Tensor:
    """Kick-mode counterpart to ``penalty_stance_asymmetry``: the identical left/right leg
    asymmetry error, gated on the post-kick recovery/hold phase (``_kick_recovery_gate``)
    instead of the locomotion velocity command. See ``_kick_recovery_gate`` and
    ``reward.py``'s registration for the full rationale -- this ports the already-validated
    standing-shaping toolkit into the one phase of a kick episode that currently lacks it.
    """
    return _stance_asymmetry_error(env) * _kick_recovery_gate(env, grace_steps) * kick_recovery_posture_reward_scale(env)


def _yaw_drift_error(env) -> torch.Tensor:
    """Squared base yaw angular velocity. Shared error computation for ``penalty_yaw_drift``
    (locomotion, command-gated) and ``penalty_kick_recovery_yaw_drift`` (kick, phase-gated)."""
    ang_vel = get_base_ang_vel(env)
    return torch.square(ang_vel[:, 2])


def penalty_yaw_drift(env, command_gate_sigma: float = 0.1, grace_steps: float = 0.0) -> torch.Tensor:
    """Penalize base yaw angular velocity, strongest when the velocity command is near
    zero (fades out smoothly as commanded speed grows, so it doesn't fight commanded
    turning).

    Root cause this addresses: ``tracking_ang_vel``'s soft ``exp(-error / sigma)`` reward
    saturates near 1.0 for small errors, so a slow-but-persistent yaw-rate bias barely
    dents it, yet accumulates into a large heading change over a multi-second stand.
    ``penalty_ang_vel_xy`` only covers roll/pitch (indices ``:2``) — nothing directly
    penalizes yaw rate itself. Confirmed via direct IsaacSim measurement on a unified
    checkpoint: walking, then holding a zero velocity command for 6s, produced a mean
    ~12.6 deg / max ~96 deg heading change during the stand alone, essentially
    uncorrelated with left/right leg-symmetry error (r=0.03) — i.e. this is a distinct
    failure mode from ``penalty_stance_asymmetry`` above, not a symptom of it.

    Args:
        env: The environment instance
        command_gate_sigma: Sigma for the exponential command-magnitude gate — the
            penalty is near-full strength at zero command and decays as commanded
            speed grows.

    Returns:
        Reward tensor [num_envs] (positive error magnitude; pair with a negative
        weight, same convention as ``penalty_ang_vel_xy``).
    """
    return _yaw_drift_error(env) * _standing_gate(env, command_gate_sigma, grace_steps)


def penalty_kick_recovery_yaw_drift(env, grace_steps: float = 50.0) -> torch.Tensor:
    """Kick-mode counterpart to ``penalty_yaw_drift``: the identical yaw-rate error, gated on
    the post-kick recovery/hold phase (``_kick_recovery_gate``) instead of the locomotion
    velocity command. See ``_kick_recovery_gate`` and ``reward.py``'s registration."""
    return _yaw_drift_error(env) * _kick_recovery_gate(env, grace_steps) * kick_recovery_posture_reward_scale(env)


def penalty_stand_height(
    env,
    command_gate_sigma: float = 0.1,
    target_height: float | None = None,
    deadzone: float = 0.0,
    grace_steps: float = 0.0,
) -> torch.Tensor:
    """Penalize base-height deviation from the standing-height target, strongest when the
    velocity command is near zero (fades out smoothly as commanded speed grows, so it
    doesn't fight the natural height bob of a walking gait).

    Root cause this addresses: nothing in the reward set directly constrains standing
    height. ``pose``'s hip_pitch/knee weights are deliberately near-zero (0.01) so walking
    strides aren't penalized, but that leaves exactly the two joints most responsible for
    squat depth almost free to deviate during standing too. ``penalty_stance_asymmetry``
    only fires on left/right *differences*, so a symmetric crouch (both legs bending the
    same way together) scores near-zero on it. Confirmed via direct IsaacSim measurement
    on a from-scratch unified checkpoint: base height settled at ~0.60m under a
    zero-velocity command, with hip_pitch at ~-0.78 rad and knee at ~+0.85 rad on *both*
    legs nearly identically — a large, coordinated, symmetric crouch invisible to every
    existing standing-related penalty.

    Error profile is ABSOLUTE (L1), not squared. The term originally used squared error
    and it decisively failed to prevent the crouch even at weight -80 with the penalty
    curriculum confirmed at full scale (v3 Stage-B run, 238k steps): the policy still
    settled 12.2cm below target, because 0.122^2 = 0.015 error * 80 = 1.2/step against
    ``alive``'s +10/step — and the squared profile's gradient also vanishes exactly as
    the crouch shrinks, so even a partial fix would stall (a residual 5cm crouch at -80
    costs just 0.2/step). |error| keeps both the magnitude and the gradient meaningful at
    every realistic crouch depth: 12cm * 40 = 4.8/step, 5cm * 40 = 2.0/step.

    ``deadzone`` (added after the v8 Stage-A run): with zero deadzone, the L1 profile
    penalizes literal zero-error standing, fighting the small continuous height bob
    that real balance requires. Confirmed via the run's own telemetry: whenever
    ``penalty_curriculum`` reached full scale (1.0), average episode length collapsed
    within a few hundred steps, then recovered once the curriculum eased back down —
    a repeating limit cycle (7 cycles over 72k steps, never converging), not a
    transient. A small free band around the target removes the demand for literally
    perfect height while leaving the original crouch failure mode (12cm, 5cm) fully
    penalized.

    Args:
        env: The environment instance
        command_gate_sigma: Sigma for the exponential command-magnitude gate — the
            penalty is near-full strength at zero command and decays as commanded
            speed grows.
        target_height: Standing base-height target in meters. ``None`` (default) falls
            back to ``init_state.pos[2]`` — but note for G1 that value (0.8) is with the
            legs nearly straight (~0.793 max) and the *default joint pose* (knee=0.669)
            grounds at 0.7565, so 0.8 is unattainable-by-7mm and pulls toward straight
            legs; the unified config passes 0.78 (matches the measured walking-height
            equilibrium, comfortably between default-pose and straight-leg heights).
        deadzone: Free half-band around ``target_height`` in meters (0.0 = no deadzone,
            the original behavior).

    Returns:
        Reward tensor [num_envs] (positive error magnitude; pair with a negative
        weight, same convention as ``penalty_stance_asymmetry``).
    """
    error = _stand_height_error(env, target_height, deadzone)
    return error * _standing_gate(env, command_gate_sigma, grace_steps)


def _stand_height_error(env, target_height: float | None, deadzone: float) -> torch.Tensor:
    """Base-height deviation from the standing-height target, beyond ``deadzone``. Shared error
    computation for ``penalty_stand_height`` (locomotion, command-gated) and
    ``penalty_kick_recovery_stand_height`` (kick, phase-gated)."""
    target = target_height if target_height is not None else env.robot_config.init_state.pos[2]
    base_height = env.simulator.robot_root_states[:, 2]
    return torch.clamp(torch.abs(base_height - target) - deadzone, min=0.0)


def penalty_kick_recovery_stand_height(
    env, target_height: float | None = None, deadzone: float = 0.0, grace_steps: float = 50.0
) -> torch.Tensor:
    """Kick-mode counterpart to ``penalty_stand_height``: the identical height error, gated on
    the post-kick recovery/hold phase (``_kick_recovery_gate``) instead of the locomotion
    velocity command. See ``_kick_recovery_gate`` and ``reward.py``'s registration."""
    error = _stand_height_error(env, target_height, deadzone)
    return error * _kick_recovery_gate(env, grace_steps) * kick_recovery_posture_reward_scale(env)


def penalty_stand_feet_width(
    env,
    command_gate_sigma: float = 0.1,
    nominal_width: float = 0.24,
    deadzone: float = 0.06,
    grace_steps: float = 0.0,
) -> torch.Tensor:
    """Penalize the lateral (heading-frame) distance between the feet deviating from the
    nominal standing stance width, strongest when the velocity command is near zero (fades
    out as commanded speed grows, so it never fights stride-width variation while walking).

    Root cause this addresses: measured on the v5 Stage-B checkpoint (which fixed height,
    symmetry, waist and pelvis lean), the robot initially stands correctly but its feet
    slide OUTWARD at a steady ~2.3 cm/s in a near-perfectly mirror-symmetric split
    (hip_roll L=+0.84 / R=-0.63 rad at the end, ankle_roll pinned at its +/-0.2618 joint
    limit from ~12s on), reaching a 1.04m stance and falling at ~24s. Every other standing
    term is blind to this axis: a symmetric spread scores ~0 on ``penalty_stance_asymmetry``
    (it measures the ANTI-symmetric component, left - mirror(right) = L+R for roll), keeps
    the pelvis level (``penalty_stand_orientation`` ~0), and costs almost no height at
    first (height ~ sqrt(l^2 - (w/2)^2) is quadratically insensitive to width near
    nominal, so ``penalty_stand_height`` only bites once the drift is already runaway) --
    and the fall lands at ~24s, just past the 20s episode horizon, so training never even
    observes the consequence. ``pose``'s hip_roll weight (1.0, term weight -0.5) charges
    a mere ~0.13/step at 0.5 rad. This term guards the axis directly: the actual
    foot-to-foot lateral separation.

    L1 beyond a small deadzone (same non-vanishing-profile principle as the other
    ``penalty_stand_*`` terms): |width - nominal_width| below ``deadzone`` is free (normal
    stance variation), beyond it the penalty grows linearly. At the measured drift onset
    (width 0.37, dev-beyond-deadzone 0.07) a -20 weight already costs 1.4/step; at the
    mid-drift 0.81m width it costs ~10/step -- decisive against ``alive``'s +10/step.

    Args:
        env: The environment instance
        command_gate_sigma: Sigma for the exponential command-magnitude gate.
        nominal_width: Target lateral feet separation in meters (the G1 default-pose
            stance measures 0.237m).
        deadzone: Free half-band around nominal_width in meters.

    Returns:
        Reward tensor [num_envs] (positive error magnitude; pair with a negative
        weight, same convention as the other ``penalty_stand_*`` terms).
    """
    error = _stand_feet_width_error(env, nominal_width, deadzone)
    return error * _standing_gate(env, command_gate_sigma, grace_steps)


def _stand_feet_width_error(env, nominal_width: float, deadzone: float) -> torch.Tensor:
    """Heading-frame lateral foot-to-foot separation, deviation from ``nominal_width`` beyond
    ``deadzone`` (same construction as ``penalty_close_feet_xy``). Shared error computation for
    ``penalty_stand_feet_width`` (locomotion, command-gated) and
    ``penalty_kick_recovery_stand_feet_width`` (kick, phase-gated)."""
    left_foot_xy = env.simulator._rigid_body_pos[:, env.feet_indices[0], :2]
    right_foot_xy = env.simulator._rigid_body_pos[:, env.feet_indices[1], :2]

    base_forward = quat_apply(env.base_quat, base_forward_vector(env), w_last=True)
    base_yaw = torch.atan2(base_forward[:, 1], base_forward[:, 0])
    feet_width = torch.abs(
        torch.cos(base_yaw) * (left_foot_xy[:, 1] - right_foot_xy[:, 1])
        - torch.sin(base_yaw) * (left_foot_xy[:, 0] - right_foot_xy[:, 0])
    )

    return torch.clamp(torch.abs(feet_width - nominal_width) - deadzone, min=0.0)


def penalty_kick_recovery_stand_feet_width(
    env, nominal_width: float = 0.24, deadzone: float = 0.06, grace_steps: float = 50.0
) -> torch.Tensor:
    """Kick-mode counterpart to ``penalty_stand_feet_width``: the identical feet-width error,
    gated on the post-kick recovery/hold phase (``_kick_recovery_gate``) instead of the
    locomotion velocity command. See ``_kick_recovery_gate`` and ``reward.py``'s registration."""
    error = _stand_feet_width_error(env, nominal_width, deadzone)
    return error * _kick_recovery_gate(env, grace_steps) * kick_recovery_posture_reward_scale(env)


def _resolve_knee_indices(env) -> torch.Tensor:
    """Lazily resolve and cache [left_knee_link, right_knee_link] body indices on the env
    (mirrors ``env.feet_indices``, which is built the same way in ``_setup_robot_body_indices``
    but only for the configured foot body — knees aren't part of that list)."""
    cached = getattr(env, "_knee_indices", None)
    if cached is not None:
        return cached
    idx = torch.tensor(
        [env.simulator.find_rigid_body_indice("left_knee_link"), env.simulator.find_rigid_body_indice("right_knee_link")],
        dtype=torch.long,
        device=env.device,
    )
    env._knee_indices = idx
    return idx


def penalty_stand_knee_width(
    env,
    command_gate_sigma: float = 0.1,
    nominal_width: float = 0.24,
    deadzone: float = 0.03,
    grace_steps: float = 0.0,
) -> torch.Tensor:
    """Penalize the lateral (heading-frame) distance between the KNEES deviating from the
    nominal standing stance width -- same construction as ``penalty_stand_feet_width``, one
    body segment up the leg. Command- and grace-gated identically to the rest of the standing
    family (see ``_standing_gate``).

    Root cause this addresses: after the unified DR broadening (mass/CoM/RFI randomization,
    added 2026-07-15 for the Stage-C PhysX->MuJoCo sim2sim gap), a visibly knock-kneed standing
    posture was observed on the resulting checkpoint. Measured directly in MuJoCo over an 8s
    zero-velocity hold (broadDR run, checkpoints 189k-205k): both ankle AND knee lateral
    separation drift inward together, from a default-pose baseline of ~0.237m down to
    0.157-0.20m -- i.e. this is NOT knee-specific decoupling from the feet (knee and ankle track
    each other closely at every checkpoint measured), it's the whole leg narrowing. The existing
    ``penalty_stand_feet_width`` already covers this axis in principle, but its deadzone (0.06)
    consumed nearly all of the observed drift: at width=0.157 (0.083 below nominal) the
    excess-beyond-deadzone was only 0.023, worth -0.46/step at weight -20 -- negligible against
    ``alive``'s +10/step, which is why it wasn't holding the line. This term adds knee-level
    coverage as a second, independent guard against the same failure (belt-and-suspenders: an
    ankle-only guard cannot distinguish "true parallel stance" from "ankles correctly spaced but
    knees adducted/legs bowed", a distinct real degree of freedom even though it wasn't the
    mechanism observed in this specific measurement) -- paired with tightening
    ``penalty_stand_feet_width``'s own deadzone (see its registration in reward.py) so both guards
    actually engage on the magnitude of drift now being observed.

    nominal_width=0.24 verified directly against the MuJoCo default-pose keyframe: knee lateral
    separation measures 0.2372m there, matching the ankle's 0.2370m almost exactly (the legs are
    parallel at rest, no natural taper) -- so the ankle-tuned nominal applies unchanged at knee
    level.

    Args:
        env: The environment instance
        command_gate_sigma: Sigma for the exponential command-magnitude gate.
        nominal_width: Target lateral knee separation in meters.
        deadzone: Free half-band around nominal_width in meters.
        grace_steps: See ``_standing_gate``.

    Returns:
        Reward tensor [num_envs] (positive error magnitude; pair with a negative weight, same
        convention as the other ``penalty_stand_*`` terms).
    """
    error = _stand_knee_width_error(env, nominal_width, deadzone)
    return error * _standing_gate(env, command_gate_sigma, grace_steps)


def _stand_knee_width_error(env, nominal_width: float, deadzone: float) -> torch.Tensor:
    """Heading-frame lateral knee-to-knee separation, deviation from ``nominal_width`` beyond
    ``deadzone`` (same construction as ``_stand_feet_width_error``, one body segment up the
    leg). Shared error computation for ``penalty_stand_knee_width`` (locomotion, command-gated)
    and ``penalty_kick_recovery_stand_knee_width`` (kick, phase-gated)."""
    knee_indices = _resolve_knee_indices(env)
    left_knee_xy = env.simulator._rigid_body_pos[:, knee_indices[0], :2]
    right_knee_xy = env.simulator._rigid_body_pos[:, knee_indices[1], :2]

    base_forward = quat_apply(env.base_quat, base_forward_vector(env), w_last=True)
    base_yaw = torch.atan2(base_forward[:, 1], base_forward[:, 0])
    knee_width = torch.abs(
        torch.cos(base_yaw) * (left_knee_xy[:, 1] - right_knee_xy[:, 1])
        - torch.sin(base_yaw) * (left_knee_xy[:, 0] - right_knee_xy[:, 0])
    )

    return torch.clamp(torch.abs(knee_width - nominal_width) - deadzone, min=0.0)


def penalty_kick_recovery_stand_knee_width(
    env, nominal_width: float = 0.24, deadzone: float = 0.03, grace_steps: float = 50.0
) -> torch.Tensor:
    """Kick-mode counterpart to ``penalty_stand_knee_width``: the identical knee-width error,
    gated on the post-kick recovery/hold phase (``_kick_recovery_gate``) instead of the
    locomotion velocity command. See ``_kick_recovery_gate`` and ``reward.py``'s registration."""
    error = _stand_knee_width_error(env, nominal_width, deadzone)
    return error * _kick_recovery_gate(env, grace_steps) * kick_recovery_posture_reward_scale(env)


def penalty_stand_orientation(
    env, command_gate_sigma: float = 0.1, deadzone: float = 0.0, grace_steps: float = 0.0
) -> torch.Tensor:
    """Penalize base (pelvis) tilt from vertical, strongest when the velocity command is
    near zero (fades out smoothly as commanded speed grows, so it doesn't fight the
    natural forward torso pitch of a walking gait -- that stays the job of the global,
    always-on ``penalty_orientation``).

    Root cause this addresses: measured on the v4 Stage-B checkpoint (which fixed the
    standing crouch -- correct 0.774m height, symmetric legs, straight waist, arms at
    default), the robot still stood with a constant ~13.6 deg forward PELVIS pitch, the
    whole upper body rigidly following it (torso 14.5 deg, waist joints near zero). The
    lean lives in a reward blind spot: ``pose``'s hip_pitch weight is deliberately ~0
    (0.01, so walking strides aren't penalized), so tilting the pelvis by extending the
    hips is nearly free at the joint level, and the waist -- which IS heavily
    pose-weighted -- dutifully stays aligned with the tilted pelvis. The only objection
    comes from ``penalty_orientation``, whose SQUARED gravity-xy profile at weight -10
    yields just 0.235^2*10 = 0.55/step at a 13.6 deg lean against ``alive``'s +10/step,
    with a gradient that vanishes as the lean shrinks -- the same profile failure,
    at almost the same magnitude, as the standing-crouch's squared height error.

    Same design as its standing-family siblings (``penalty_stand_height`` L1 reshape):
    ABSOLUTE (L1) error -- here the norm of the projected gravity's xy component, i.e.
    sin(tilt angle) -- so both magnitude and gradient stay meaningful at every realistic
    lean (13.6 deg -> |g_xy| = 0.235; 3 deg residual -> 0.052, still visible at the
    configured weight).

    ``deadzone`` (added after the v8 Stage-A run): with zero deadzone, the L1 profile
    demands literally zero tilt, fighting the small continuous postural sway real
    balance requires. Confirmed via the run's own telemetry: whenever
    ``penalty_curriculum`` reached full scale (1.0), average episode length collapsed
    within a few hundred steps, then recovered once the curriculum eased back down --
    a repeating limit cycle (7 cycles over 72k steps, never converging), not a
    transient. A small free band around vertical removes the demand for a literally
    perfect pelvis while leaving the original 13.6deg lean failure mode fully
    penalized.

    Args:
        env: The environment instance
        command_gate_sigma: Sigma for the exponential command-magnitude gate -- the
            penalty is near-full strength at zero command and decays as commanded
            speed grows.
        deadzone: Free half-band around vertical, in units of |g_xy| = sin(tilt)
            (0.0 = no deadzone, the original behavior).

    Returns:
        Reward tensor [num_envs] (positive error magnitude; pair with a negative
        weight, same convention as ``penalty_stance_asymmetry``).
    """
    error = _stand_orientation_error(env, deadzone)
    return error * _standing_gate(env, command_gate_sigma, grace_steps)


def _stand_orientation_error(env, deadzone: float) -> torch.Tensor:
    """Base (pelvis) tilt from vertical (``|g_xy| = sin(tilt)``), beyond ``deadzone``. Shared
    error computation for ``penalty_stand_orientation`` (locomotion, command-gated) and
    ``penalty_kick_recovery_stand_orientation`` (kick, phase-gated)."""
    projected = get_projected_gravity(env)
    return torch.clamp(torch.norm(projected[:, :2], dim=-1) - deadzone, min=0.0)


def penalty_kick_recovery_stand_orientation(env, deadzone: float = 0.0, grace_steps: float = 50.0) -> torch.Tensor:
    """Kick-mode counterpart to ``penalty_stand_orientation``: the identical pelvis-tilt error,
    gated on the post-kick recovery/hold phase (``_kick_recovery_gate``) instead of the
    locomotion velocity command. See ``_kick_recovery_gate`` and ``reward.py``'s registration."""
    error = _stand_orientation_error(env, deadzone)
    return error * _kick_recovery_gate(env, grace_steps) * kick_recovery_posture_reward_scale(env)


def penalty_kick_swing_orientation(env, deadzone: float = 0.0) -> torch.Tensor:
    """Pelvis tilt from vertical during the WHOLE kick episode (approach + strike + recovery/
    hold) -- UNGATED, unlike ``penalty_kick_recovery_stand_orientation`` above, which only fires
    post-swing. 2026-08-04, ported from RoboNaldo (arXiv:2606.11092)'s
    ``locomotion_phase_orientation_l2``: despite that function's name, reading their code shows
    it is NOT actually phase-gated either -- it applies continuously, including through their
    kick, at a small per-step weight (-0.2*reg_weight to -0.5*reg_weight in their shipped
    presets). The motivation ported directly: right now nothing gives the policy a dense,
    always-on "don't lean too far" signal during the swing itself -- the only orientation
    pressure there is clip tracking (measured, this session: error_body_rot_swing ~0.23-0.24,
    i.e. tracking is not doing this job well) and ``kick_balance_potential`` (a capture-point
    MARGIN shaping term, a different quantity from raw tilt). This fills that gap rather than
    competing with either.

    Formula deliberately reuses ``_stand_orientation_error`` (L1 tilt magnitude + deadzone), NOT
    RoboNaldo's own plain L2/squared ``sum(g_xy**2)`` -- this project already found, and fixed,
    the exact failure mode a squared profile has here: ``penalty_stand_orientation``'s own
    docstring records a real v8 Stage-A limit cycle (7 cycles over 72k steps) traced to the
    squared profile's vanishing gradient near small leans. Porting RoboNaldo's L2 form verbatim
    would reintroduce a bug this codebase has already diagnosed and moved past; L1 is the
    like-for-like upgrade, not a deviation from the port.

    Weight 0.0 by default -- explicit opt-in, unvalidated starting point, same discipline as
    every other addition this session. A reasoned (not yet trained) starting weight: -1.0 to
    -2.0, deliberately far below ``penalty_kick_recovery_stand_orientation``'s -20.0 -- the swing
    genuinely needs freedom to lean that the stationary recovery/hold phase does not."""
    return _stand_orientation_error(env, deadzone)


def _torso_orientation_error(env, deadzone: float, torso_body_name: str = "torso_link") -> torch.Tensor:
    """Torso-link tilt from vertical, same L1+deadzone profile as ``_stand_orientation_error``
    (pelvis) -- see ``penalty_kick_swing_orientation``'s own docstring for why L1 was chosen over
    RoboNaldo's plain L2. Resolves ``torso_body_name`` fresh via ``env.body_names.index(...)``
    every call, deliberately uncached -- matches RoboNaldo's own
    ``locomotion_phase_torso_orientation_l2`` (also recomputed every call, no caching), and the
    body list is short enough (~30 bodies) that this costs nothing measurable."""
    torso_index = env.body_names.index(torso_body_name)
    torso_quat = env.simulator._rigid_body_rot[:, torso_index]
    projected = quat_rotate_inverse(torso_quat, gravity_vector(env), w_last=True)
    return torch.clamp(torch.norm(projected[:, :2], dim=-1) - deadzone, min=0.0)


def penalty_kick_swing_torso_orientation(
    env, deadzone: float = 0.0, torso_body_name: str = "torso_link"
) -> torch.Tensor:
    """Torso tilt from vertical, ungated across the whole kick episode -- the torso-specific
    sibling of ``penalty_kick_swing_orientation`` immediately above (see that function's own
    docstring for the full RoboNaldo-port rationale; this ports their separate
    ``locomotion_phase_torso_orientation_l2``). Torso and pelvis can tilt independently (e.g. a
    counter-lean through the waist), which is exactly why RoboNaldo tracks them as two terms
    rather than one -- see ``reward.py``'s registration for how the two are wired.

    Weight 0.0 by default -- explicit opt-in, unvalidated starting point. Reasoned starting
    weight: -1.0, matching ``penalty_kick_swing_orientation``'s own reasoning at a slightly
    smaller magnitude (a secondary, correlated signal to the pelvis term, not the primary one)."""
    return _torso_orientation_error(env, deadzone, torso_body_name)


def penalty_kick_unstable(env, ang_vel_weight: float = 0.5, grace_steps: float = 50.0) -> torch.Tensor:
    """Penalize base linear + angular velocity during post-kick recovery/hold -- 2026-08-05,
    ported from RoboNaldo (arXiv:2606.11092)'s ``unstable_penalty`` (mdp/rewards.py). A settled
    stand should be nearly stationary; this prices ongoing motion (wobbling, still-decaying
    momentum, a slide toward a fall) directly, as a straightforward "don't keep moving" signal.

    IMPORTANT -- this is DISTINCT from RoboNaldo's ``stable_anchor_pos_tracking``, deliberately
    NOT ported here: that term latches the robot's own kick-entry anchor pose
    (``stabilize_anchor_pos_w``) and rewards pulling back to it, but is gated
    ``weight=10*reg_weight*float(adapt_motion_flag)`` in RoboNaldo's own source -- and
    ``adapt_motion_flag`` is False in every RoboNaldo stage this project follows (S1 "tracking",
    S2a/S2b "task_params_1/2"), only True in their Stage 3, which this project has already decided
    NOT to follow (see ROBONALDO_PORT_SCOPE.md). So in the stages actually being ported,
    ``unstable_penalty`` -- not the latched-anchor mechanism -- IS RoboNaldo's real S1/S2 post-kick
    stabilization pressure, alongside ``robot_alive``'s own 10x post-kick step-up (already ported,
    2026-08-05, as ``kick_alive_pre_kick_ratio`` -- see that field's own docstring). An earlier
    draft of this port's scope document described the latched-anchor mechanism as something to
    port here; that was written before reading ``stable_anchor_pos_tracking``'s exact
    ``adapt_motion_flag`` gate and has been corrected.

    Formula, ported faithfully: ``sum(base_lin_vel_xy^2) + ang_vel_weight * sum(base_ang_vel^2)``
    (RoboNaldo's own ``ang_vel_weight`` is hardcoded 0.5, exposed here as a parameter for
    consistency with this project's own tunable-term convention, default unchanged). Reads
    ``env.simulator.robot_root_states`` -- same tensor/slicing convention as
    ``kick_safety.py:penalty_excess_base_lin_vel`` (``[:, 7:10]`` lin_vel, ``[:, 10:13]`` ang_vel,
    world frame).

    ADAPTATION: RoboNaldo gates with a hard boolean (``time_steps > critic_frame_index +
    kick_hold_steps``) -- an instant 0-to-full-strength snap. This project already diagnosed and
    fixed exactly that failure mode once before (``_standing_gate``'s own v9 Stage-A history: 7/10
    falls traced to a hard instant snap to full standing demand while momentum was still live) and
    built the fix as a reusable, already-tested primitive -- ``_kick_recovery_gate``, a smooth
    linear ramp over ``grace_steps`` after ``stand_start_idx``. Reused directly here as a
    MULTIPLIER (not a hard mask) rather than duplicating a new hard gate for this one term.

    Weight 0.0 by default -- explicit opt-in, unvalidated starting point. Reasoned starting
    weight: -0.05 (RoboNaldo's own S2a/S2b effective weight is -0.04; see
    ROBONALDO_PORT_SCOPE.md Sec 3b's weight table)."""
    lin_vel_xy = env.simulator.robot_root_states[:, 7:9]
    ang_vel = env.simulator.robot_root_states[:, 10:13]
    lin_penalty = torch.sum(torch.square(lin_vel_xy), dim=-1)
    ang_penalty = torch.sum(torch.square(ang_vel), dim=-1)
    magnitude = lin_penalty + ang_vel_weight * ang_penalty
    return magnitude * _kick_recovery_gate(env, grace_steps)


def penalty_kick_lin_vel_z(env) -> torch.Tensor:
    """Penalize base vertical velocity -- 2026-08-05, ported from RoboNaldo (arXiv:2606.11092)'s
    ``locomotion_phase_lin_vel_z_l2`` (mdp/rewards.py). Formula matches exactly:
    ``base_lin_vel_z^2``. UNGATED across the whole kick episode -- RoboNaldo's own function body
    has no phase gate at all (registered flat, weight -0.025 S1 / -0.1 S2a-b, never relaxed), so
    none is added here either, despite the name's "locomotion_phase" prefix (a RoboNaldo naming
    convention for this whole reward family, not a live gating condition on this specific term)."""
    return torch.square(env.simulator.robot_root_states[:, 9])


def penalty_kick_dof_vel(env) -> torch.Tensor:
    """Penalize mechanical power (joint velocity times applied torque) -- 2026-08-05, ported from
    RoboNaldo (arXiv:2606.11092)'s ``dof_vel`` (mdp/rewards.py, confusingly named -- the real
    formula is a POWER penalty, not a velocity-magnitude one). Torque source:
    ``env.action_manager``'s own ``JointPositionActionTerm.torques`` -- the actually APPLIED
    torque for the just-completed control step (post-PD, post-clip, post-randomization), not a
    reward-side recomputation of it. No existing reward term in this project reads applied torque
    before this one; the access path (``env.action_manager._term_instances["joint_control"].
    torques``) was confirmed live against a real env during verification, matching the exact
    attribute this project's own action term already exposes for its own internal bookkeeping.

    2026-08-14 BUG FIX: formula changed from the signed ``sum_j(dof_vel_j * torque_j)`` (the
    literal, un-squared port) to ``sum_j((dof_vel_j * torque_j)^2)`` below. The signed version is
    NEGATIVE whenever a joint is doing net-negative mechanical work (braking/absorbing rather than
    driving) -- combined with this term's NEGATIVE weight (see registration in
    ``config_values/unified/g1/reward.py``), that made the WEIGHTED contribution POSITIVE (a
    reward, not a penalty) on exactly those ticks, breaking the "negative weight always penalizes
    larger magnitude" invariant every sibling ``penalty_kick_*``/``penalty_*`` term in this file
    maintains (e.g. ``penalty_kick_torque`` immediately below, already squared). Measured live on
    a real run (``20260813_005001-stageC1-1skill-locoflip-shooting05-new-fixes-3-locomotion``,
    diagnosing a striking-phase leg/arm motion becoming progressively more violent/unnatural over
    training): raw mean went from -18.4 (step 130-150k) to -66.2 (step 330-352k) -- i.e. MORE net
    braking/absorbing over training, the same antagonistic-motion signature as the jerk growth
    that motivated ``penalty_kick_strike_dof_acc`` below -- while the WEIGHTED contribution went
    from +0.000011 to +0.0000397: the policy was increasingly REWARDED for the exact failure mode
    under investigation. Squaring fixes the sign unconditionally and matches
    ``penalty_kick_torque``'s own form one function below -- the smallest change that restores the
    invariant, not a new formula. Raw magnitude/units change substantially (squared, not linear),
    so the paired ``weight`` override in ``configs/task_config_stageC1.yaml`` was recalibrated
    accordingly at the same time -- see that yaml's own comment for the measured numbers behind
    the new value. Same wandb key (``kick_penalty_dof_vel``) kept across the fix, deliberately, for
    dashboard continuity -- matches this project's own established precedent for renaming a
    term's internals without renaming its logged key (e.g. ``kick_*_swing``/``kick_*_hold``,
    ``_update_log_dict``'s own comment)."""
    torques = env.action_manager._term_instances["joint_control"].torques  # type: ignore[attr-defined]
    return torch.sum(torch.square(env.simulator.dof_vel * torques), dim=-1)


def penalty_kick_torque(env) -> torch.Tensor:
    """Penalize large actuator torques -- 2026-08-05, ported from RoboNaldo (arXiv:2606.11092)'s
    ``torque`` (mdp/rewards.py). Formula matches exactly: ``sum_j(torque_j^2)``. Same torque
    access path as ``penalty_kick_dof_vel`` immediately above -- see that function's own docstring
    for the rationale."""
    torques = env.action_manager._term_instances["joint_control"].torques  # type: ignore[attr-defined]
    return torch.sum(torch.square(torques), dim=-1)


def penalty_kick_strike_dof_acc(env, dof_names: list[str] | None = None) -> torch.Tensor:
    """Penalize joint-acceleration magnitude, gated to ``in_strike_phase`` only -- 2026-08-14,
    added while diagnosing the same run ``penalty_kick_dof_vel``'s bug-fix docstring above cites:
    striking-phase leg (and, to a lesser extent, arm) motion becoming progressively more violent
    and unnatural over training, with task outcome (ball velocity/success rate) flat throughout --
    i.e. the extra motion was buying nothing. Direct measurement (RoboJuDo/MuJoCo replay of
    checkpoints 150k and 352k of that run) found RMS joint jerk up 81-238% across the legs (worst:
    ``right_knee`` +360%, ``right_ankle_roll`` +387%) while joint-POSITION tracking error only
    grew ~8% -- i.e. the policy was oscillating harder AROUND a similar mean trajectory, not
    diverging further from it. Neither existing smoothness lever catches this: joint-position
    tracking (the 6 Cartesian ``motion_*_error_exp`` terms, and ``motion_strike_dof_pos_error_exp``
    below) is blind to in-tolerance oscillation once error is inside its exp kernel's flat region,
    and ``kick_action_smoothness`` (``KickActionSmoothness``) acts on the POLICY'S ACTIONS
    (commanded targets), not realized joint state -- it cannot see PD overshoot/oscillation the
    action sequence itself may look smooth on paper but still excites.

    Formula: ``sum_j(dof_acc_j^2)``, gated by ``motion_command.in_strike_phase`` (same window
    ``motion_strike_dof_pos_error_exp`` uses, and the window the user's own report specifically
    named -- "in striking phase"). Acceleration-squared is a standard, well-precedented
    smoothness-regularizer proxy for jerk (suppressing large acceleration directly suppresses the
    rapid velocity swings that produce large jerk too), and is smooth/differentiable everywhere
    unlike an abs-based alternative.

    2026-08-14 CORRECTION (live-verified, not just unit-tested): an earlier version of this
    function read ``env.simulator.dof_acc`` directly, believing every simulator backend computes
    it unconditionally. Live IsaacSim verification (this term crashing immediately on a real
    checkpoint rollout) found that's false for THIS project's actual training backend:
    ``simulator/isaacsim/isaacsim.py`` only allocates/updates ``dof_acc`` when
    ``simulator_config.bridge.enabled`` -- a real-hardware-deployment feature, off during normal
    training -- so the attribute doesn't exist at all in the configuration this term actually
    runs under. Fixed by computing the same finite-difference IsaacSim's own (bridge-only) code
    would, independently of that flag: ``env.action_manager``'s ``JointPositionActionTerm``
    already caches ``_prev_dof_vel`` every physics substep for its OWN internal use (the velocity-
    control D-term, ``managers/action/terms/joint_control.py``), exposed via the existing public
    ``get_prev_dof_vel()`` -- reused here rather than adding a second, redundant previous-velocity
    buffer. ``(dof_vel - prev_dof_vel) / env.sim_dt`` (substep dt, matching
    ``JointPositionActionTerm``'s own denominator exactly) is acceleration over the most recent
    physics SUBSTEP, not the full decimated control tick -- a real, valid (if higher-frequency)
    acceleration reading, not an approximation error. No new state on this function's side: reset
    behavior (zeroed on episode reset) is entirely inherited from ``JointPositionActionTerm``'s
    own ``reset()``. Same MuJoCo-backend behavior as before (``dof_acc`` there is unconditional,
    native ``qacc``) is NOT relied upon by this fix -- IsaacSim is this project's actual training
    backend and is now the one this function is verified against.

    ``dof_names`` (default None = all 29 DOF): unlike ``motion_strike_leg_null_dof_pos_error_exp``
    (which deliberately narrows to specific low-Cartesian-authority axes), this is left broad by
    default -- the measured jerk growth was NOT confined to null-space axes (e.g. ``right_knee``,
    a high-authority sagittal joint, was the single worst grower), so a narrow default would miss
    most of what was actually measured. Exposed as a parameter for future narrowing if this proves
    too broad in practice (e.g. fighting the kick leg's legitimate strike power).

    Weight 0.0 by default (explicit opt-in, unvalidated-by-training starting point, same
    discipline as every other term in this file) -- see ``configs/task_config_stageC1.yaml``'s own
    comment for the measured raw-magnitude calibration behind its active starting weight there
    (calibrated against a LIVE IsaacSim reading post-fix, not the original MuJoCo-proxy estimate --
    see that yaml comment for why the two differ). Returns an all-zero tensor for envs with no
    ball-kicking ``motion_command``, matching every other ``in_strike_phase``-gated term's safe-
    no-op convention in this codebase."""
    motion_command = env.command_manager.get_state("motion_command")
    if motion_command is None or not getattr(motion_command, "has_ball", False):
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    magnitude = _dof_acc_magnitude(env, dof_names)
    return magnitude * motion_command.in_strike_phase.float()


def _dof_acc_magnitude(env, dof_names: list[str] | None) -> torch.Tensor:
    """Shared ``sum_j(dof_acc_j^2)`` computation behind ``penalty_kick_strike_dof_acc`` and
    ``penalty_kick_approach_dof_acc`` below -- UNGATED (no phase mask, no has_ball check, no
    zero-tensor fallback); each caller applies its own phase gate and safety checks on top. See
    ``penalty_kick_strike_dof_acc``'s own docstring for the full derivation (finite-difference of
    ``dof_vel`` via the joint_control action term's ``prev_dof_vel``, substep ``env.sim_dt``, not
    ``env.simulator.dof_acc`` -- that attribute only exists when a hardware bridge is enabled)."""
    prev_dof_vel = env.action_manager._term_instances["joint_control"].get_prev_dof_vel()  # type: ignore[attr-defined]
    dof_acc = (env.simulator.dof_vel - prev_dof_vel) / env.sim_dt
    if dof_names is not None:
        dof_indexes = [env.simulator.dof_names.index(name) for name in dof_names]  # type: ignore[attr-defined]
        dof_acc = dof_acc[:, dof_indexes]
    return torch.sum(torch.square(dof_acc), dim=-1)


def penalty_kick_approach_dof_acc(env, dof_names: list[str] | None = None) -> torch.Tensor:
    """Locomotion-approach counterpart of ``penalty_kick_strike_dof_acc`` immediately above --
    2026-08-14, added as a SEPARATE term rather than widening that one's gate, because the two
    phases want OPPOSITE things from joint acceleration. The strike phase's whole purpose is high
    acceleration (that IS the kick's power); suppressing it there would fight the task itself --
    which is exactly why that term's weight is deliberately sized as a small guardrail, not a
    dominant lever (see its own docstring and configs/task_config_stageC1.yaml's calibration
    comment). The approach phase has no such excuse: any acceleration there is jitter, not power,
    with nothing to trade off against. A single shared weight across both phases is structurally a
    compromise between two opposing objectives -- too weak to matter during approach if it's kept
    safely gentle for the strike, or fights the strike's power if raised enough to matter for
    approach. Splitting lets each phase's weight answer only its own question.

    Also relevant to the locomotion->kick MID-EPISODE HANDOFF (Stage D,
    UnifiedManager._maybe_enter_kick_from_locomotion): entry always lands somewhere in
    [motion_start_idx, strike_start_idx) -- the approach window this term covers, never the
    strike itself (the entry search is constrained to that window,
    MotionCommand._build_entry_search_table) -- so this is also the term that would price any
    joint-level oscillation right after a handoff, complementing (not overlapping with)
    pre_kick_reward_ramp_steps/pre_kick_termination_grace_steps, which soften EXISTING terms'
    timing rather than pricing realized joint motion at all.

    Gate: ``in_kicking_phase & ~in_strike_phase`` -- equivalent to ``time_steps <
    strike_start_idx[motion_ids]`` (in_kicking_phase's own upper bound, stand_start_idx, is never
    reached without first passing through strike, so intersecting with NOT-in_strike_phase
    collapses to exactly the approach window, not "approach plus everything after strike").
    Formula and dof_names semantics identical to penalty_kick_strike_dof_acc -- see that
    function's own docstring.

    Weight 0.0 by default (explicit opt-in, unvalidated starting point, same discipline as every
    term in this file) -- deliberately NOT calibrated/activated in this change; the right starting
    weight needs its own live-measured approach-phase magnitude (this project's own established
    practice, see penalty_kick_strike_dof_acc's calibration history), not reused from the strike
    term's number, which was sized for a phase with different dynamics."""
    motion_command = env.command_manager.get_state("motion_command")
    if motion_command is None or not getattr(motion_command, "has_ball", False):
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    magnitude = _dof_acc_magnitude(env, dof_names)
    in_approach = motion_command.in_kicking_phase & ~motion_command.in_strike_phase
    return magnitude * in_approach.float()


def penalty_kick_ee_body_pos_divergence(
    env,
    threshold: float = 0.25,
    warmup_threshold: float = 0.25,
    warmup_steps: int = 20,
    body_names: list[str] | None = None,
) -> torch.Tensor:
    """Flag (1.0/0.0) end-effector bodies whose position has diverged from the reference clip
    beyond a threshold -- 2026-08-05, ported from RoboNaldo (arXiv:2606.11092)'s
    ``ee_body_pos_termination_penalty`` (mdp/rewards.py), which "returns 1.0 on the same step
    where the EE body-position termination fires" (their own docstring) -- an early-warning event
    marker for exactly the divergence their own ``ee_body_pos`` TERMINATION would also catch, at a
    per-event penalty (weight -100 in every stage) rather than ending the episode outright.

    ``threshold`` (default 0.25, RoboNaldo's own Stage-1 value) SHOULD be passed the same value as
    ``MultiSkillConfig.bad_motion_body_pos_threshold`` -- ``config_values/unified/g1/reward.py``'s
    registration does this automatically, keeping this reward term numerically synced with
    ``bad_tracking``'s ``bad_motion_body_pos`` termination sub-check it is meant to shadow (see
    that field's own docstring). Kept as an independent param, not read from the command/env
    directly, mirroring RoboNaldo's own ``ee_body_pos_termination_penalty`` and
    ``bad_motion_body_pos_z_only`` -- two independently-defined checks kept in sync only by their
    shared config-loading code (``task_overrides.py``), not by sharing a Python constant.

    TWO branches, both ported (2026-08-05 correction: an earlier version of this function dropped
    the second branch below, believing this project's ``MotionCommand`` had no equivalent to
    RoboNaldo's ``command.is_warmup`` gate -- re-reading their source
    (``mdp/commands.py::MotionCommand.is_warmup``) found ``is_warmup = real_time_steps <
    warmup_time_steps + warmup_steps``, and since ``real_time_steps``/``warmup_time_steps`` are
    both set to the RSI-sampled starting frame at reset and increment in lockstep thereafter, this
    reduces exactly to "fewer than ``warmup_steps`` steps since the last reset" -- precisely what
    this project's own ``env.episode_length_buf`` (already used the same way by ``bad_tracking``'s
    ``grace_period_steps``) already tracks. No new state needed):
    1. ``terminated``: Z-axis-only error per body exceeds ``threshold`` -- their own ``terminated``
       branch, active every step regardless of warmup.
    2. ``warmup_terminated``: full 3-axis error per body exceeds ``warmup_threshold`` (default
       0.25, RoboNaldo's own base value), but ONLY during the first ``warmup_steps`` (default 20,
       RoboNaldo's own default) steps after a reset -- a stricter, settling-window-only backstop
       for a just-RSI'd pose that hasn't snapped into XY+Z alignment yet, not merely a Z check.
    Either branch firing pays the reward.

    Body set default: the same 4 "end effector" bodies RoboNaldo's own registration uses (both
    ankles, both wrists) -- these are also this project's own ``motion_command.body_names_to_track``
    subset already exercised by ``motion_global_feet_lin_vel`` (see that function's own docstring
    for the index-space note: resolved against ``body_names_to_track``, NOT
    ``env.simulator.body_names``).

    Weight -100 in every RoboNaldo stage (S1/S2a/S2b) -- flat, no cross-stage staging needed."""
    motion_command = env.command_manager.get_state("motion_command")
    if motion_command is None or not getattr(motion_command, "has_ball", False):
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    if body_names is None:
        body_names = ["left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link"]
    tracked_names = motion_command.motion_cfg.body_names_to_track
    body_indexes = [tracked_names.index(name) for name in body_names]
    diff = motion_command.body_pos_relative_w[:, body_indexes] - motion_command.robot_body_pos_w[:, body_indexes]
    # 2026-08-15: threshold/warmup_threshold arrive as a per-env [num_envs] tensor whenever
    # params_per_skill diverges (see RewardManager.compute()'s generic gather) -- diff[:, :, 2]/
    # warmup_error are [num_envs, num_bodies], so a bare [num_envs] tensor broadcasts against the
    # WRONG trailing dim (num_bodies) instead of num_envs. unsqueeze to [num_envs, 1] so it lines
    # up per-env, same as a plain python float already broadcasts against every dim for free.
    per_env_threshold = threshold.unsqueeze(-1) if torch.is_tensor(threshold) else threshold
    terminated = torch.any(torch.abs(diff[:, :, 2]) > per_env_threshold, dim=-1)

    is_warmup = env.episode_length_buf < warmup_steps
    warmup_error = torch.norm(diff, dim=-1)
    per_env_warmup_threshold = warmup_threshold.unsqueeze(-1) if torch.is_tensor(warmup_threshold) else warmup_threshold
    warmup_terminated = torch.any(warmup_error > per_env_warmup_threshold, dim=-1) & is_warmup

    return (terminated | warmup_terminated).float()


def penalty_stumble(env) -> torch.Tensor:
    """Penalize feet hitting vertical surfaces.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    return torch.any(
        torch.norm(env.simulator.contact_forces[:, env.feet_indices, :2], dim=2)
        > 4 * torch.abs(env.simulator.contact_forces[:, env.feet_indices, 2]),
        dim=1,
    )


def penalty_foothold(env, foothold_epsilon: float = 0.01) -> torch.Tensor:
    """Sampling-based foothold penalty.

    For each foot in contact, sample a grid of points on the sole, transform to world,
    read terrain height at those XY, compute depth d_ij = z_sample - terrain_z, and count
    samples with d_ij < epsilon. Sum over both feet.

    Args:
        env: The environment instance
        foothold_epsilon: Threshold for foothold depth penalty

    Returns:
        Reward tensor [num_envs]
    """
    # Contact mask per foot
    contact = env.simulator.contact_forces[:, env.feet_indices, 2] > 1.0  # [E,2]
    if not (contact.any()):
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    # Accumulator
    penalty = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    for foot_idx_local in range(2):
        # Skip if no env has contact on this foot to save work
        if not contact[:, foot_idx_local].any():
            continue
        rb_idx = env.feet_indices[foot_idx_local]
        foot_pos_w = env.simulator._rigid_body_pos[:, rb_idx, :]  # [E,3]
        foot_quat_w = env.simulator._rigid_body_rot[:, rb_idx, :]  # [E,4]

        # Use precomputed sample points in the foot frame
        pts_local = env.foot_samples_local[foot_idx_local].unsqueeze(0).repeat(env.num_envs, 1, 1)

        # Rotate to world and translate
        pts_world = quat_rotate_batched(foot_quat_w, pts_local) + foot_pos_w.unsqueeze(1)

        # Query terrain height at those XY positions
        terrain_h = env._get_terrain_heights_at_points_world(pts_world)

        # Depth: world z minus terrain height
        depth = pts_world[:, :, 2] - terrain_h  # [E,S]

        # Indicator for d_ij > epsilon, only for envs with this foot in contact
        bad = (depth > foothold_epsilon).float()
        bad *= contact[:, foot_idx_local].unsqueeze(1).float()

        penalty += torch.sum(bad, dim=1)

    return penalty / env.num_foot_samples


def alive(env) -> torch.Tensor:
    """Reward for staying alive.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    return torch.ones(env.num_envs, dtype=torch.float, device=env.device)
