"""Locomotion-specific termination terms."""

from __future__ import annotations

from holosoma.managers.observation.terms.locomotion import get_projected_gravity
from holosoma.utils.safe_torch_import import torch


def _apply_probability(mask: torch.Tensor, probability: float, device: torch.device) -> torch.Tensor:
    """Optionally apply probabilistic gating to a mask."""
    if probability >= 1.0:
        return mask
    if probability <= 0.0:
        return torch.zeros_like(mask, dtype=torch.bool)
    sample = torch.rand(1, device=device)
    return mask & (sample < probability)


def contact_forces_exceeded(
    env, force_threshold: float = 1.0, contact_indices_attr: str = "termination_contact_indices"
) -> torch.Tensor:
    """Terminate if contact forces exceed threshold.

    Note: If you want to disable contact termination, simply don't add this term to your
    termination config instead of using a flag.
    """
    indices = getattr(env, contact_indices_attr)
    contact_forces = env.simulator.contact_forces[:, indices, :]
    return torch.any(torch.norm(contact_forces, dim=-1) > force_threshold, dim=1)


def gravity_tilt_exceeded(env, threshold_x: float, threshold_y: float) -> torch.Tensor:
    """Terminate if projected gravity exceeds roll/pitch thresholds."""
    if not getattr(env.config.termination, "terminate_by_gravity", False):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    grav = get_projected_gravity(env)
    tilt_x = torch.abs(grav[:, 0]) > threshold_x
    tilt_y = torch.abs(grav[:, 1]) > threshold_y
    return tilt_x | tilt_y


def base_height_below_threshold(env, min_height: float) -> torch.Tensor:
    """Terminate if base height drops below threshold."""
    if not getattr(env.config.termination, "terminate_by_low_height", False):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    base_height = env.simulator.robot_root_states[:, 2]
    return base_height < min_height


def base_height_below_threshold_sustained(
    env, min_height: float, consecutive_steps: int = 10, counter_attr: str = "_low_height_counter"
) -> torch.Tensor:
    """Terminate when base height stays below ``min_height`` for ``consecutive_steps``
    consecutive control steps.

    Differences from ``base_height_below_threshold``:
    - No dependency on a ``terminate_by_low_height`` config flag (``TerminationManagerCfg``
      doesn't define that field, so the stock term is effectively permanently disabled) --
      configuring this term IS enabling it.
    - Sustained-duration requirement: a brief height dip (e.g. the transient while
      decelerating from a fast walk to a stop) doesn't kill the episode; only *staying*
      low does. This makes it safe to set the threshold above a settled-crouch height
      without punishing legitimate transients.

    Why it exists: a persistent standing crouch (settling ~0.68m vs the 0.78m standing
    target) survived reward-side penalties because ``alive`` (+10/step) structurally
    rewards whatever posture maximizes survival, and a low CoM survives just fine --
    a per-step height tax competes with alive's +10/step and loses. Termination attacks
    the incentive directly: holding the crouch forfeits ALL future alive reward instead
    of paying a small tax. Pair it with ``task_mode="locomotion"`` in the unified config
    so kick-mode motions (whose single-support phases legitimately dip low) are unaffected.

    The per-env below-threshold counter zeroes itself on any step at/above the threshold;
    a reset spawns the robot well above the threshold, so the counter self-clears on the
    first post-reset step without needing an explicit reset hook.

    ``counter_attr`` namespaces the counter's storage attribute on ``env``. REQUIRED to differ
    across every distinct call site configured with different params (e.g. locomotion's
    ``low_height`` vs kick mode's ``kick_low_height``, config_values/unified/g1/termination.py) --
    with a single shared default attribute name, two term instances calling this same function
    with different min_height/consecutive_steps would silently overwrite and corrupt each other's
    counters every step (each TerminationManager tick runs every configured term for all envs,
    task_mode masking is applied AFTER the raw boolean is returned, so both terms' counter writes
    always execute regardless of which envs are in which task_mode). Caught 2026-07-18 when adding
    the second call site; not a bug for the original single-caller case, since a solitary term can
    only race with itself. If you add a third call site, give it a third counter_attr.
    """
    base_height = env.simulator.robot_root_states[:, 2]
    below = base_height < min_height
    counter = getattr(env, counter_attr, None)
    if counter is None or counter.shape[0] != env.num_envs:
        counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    counter = torch.where(below, counter + 1, torch.zeros_like(counter))
    setattr(env, counter_attr, counter)
    return counter >= consecutive_steps


def _post_flip_grace_active(env, grace_steps: float) -> torch.Tensor:
    """[num_envs] bool: True for envs currently within ``grace_steps`` ticks of having crossed
    UnifiedManager's kick->locomotion flip boundary (``kick_recovery_locomotion_flip_enabled``).
    ``grace_steps<=0`` (the field's own off default) returns an all-False mask unconditionally --
    an exact, not just numerical, no-op: ``steps_since_flip < grace_steps`` can never be satisfied
    for a non-negative step count once ``grace_steps<=0``, but this short-circuits before even
    computing ``steps_since_flip``, so a non-UnifiedManager env (no ``_post_flip_step``/
    ``post_flip_steps_since`` at all) never needs to be handled as a special case either.

    FIX 2 (2026-08-12), see MultiSkillConfig.post_flip_termination_grace_steps's own docstring for
    the full measured rationale: 600/643 (93.3%) of post-flip terminations in a live diagnostic
    were `contact` (managers/termination/terms/locomotion.py:contact_forces_exceeded), which has
    NO grace/sustained-duration concept at all -- a single incidental >1N contact on the instant a
    still-asymmetric, momentum-carrying post-kick pose is handed to locomotion mode is an instant
    kill. Mirrors the grace-ramp convention this project already uses at the OTHER kick-recovery
    boundary (kick_recovery_low_height_sustained/kick_recovery_drift_sustained's own grace_steps,
    itself mirroring _kick_recovery_gate) -- same idea, applied to the symmetric boundary these
    don't cover (task_mode=="locomotion" terms, not task_mode=="kick" ones).

    2026-08-15, "simultaneous per-skill task configs": ``grace_steps`` may also be a per-env
    ``[num_envs]`` tensor (TerminationManager's params_per_skill gather) instead of the plain
    float it always used to be -- the ``< grace_steps`` comparison below is already elementwise-
    safe against a tensor, so only the all-False fast path needs a tensor-aware form (envs whose
    OWN grace_steps<=0 are excluded via the extra AND rather than a global early return)."""
    if not torch.is_tensor(grace_steps) and grace_steps <= 0.0:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    post_flip_step = getattr(env, "_post_flip_step", None)
    if post_flip_step is None or not hasattr(env, "post_flip_steps_since"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    is_post_flip = post_flip_step >= 0
    active = is_post_flip & (env.post_flip_steps_since().float() < grace_steps)
    if torch.is_tensor(grace_steps):
        active = active & (grace_steps > 0.0)
    return active


def _pre_kick_grace_active(env, grace_steps: float) -> torch.Tensor:
    """[num_envs] bool: True for envs currently within ``grace_steps`` ticks of a mid-episode
    locomotion->kick entry (``UnifiedManager._maybe_enter_kick_from_locomotion``,
    ``mid_episode_kick_entry_prob``). Mirror of ``_post_flip_grace_active`` above, opposite
    boundary and opposite direction: that one guards LOCOMOTION-mode terms for envs just past a
    kick->locomotion flip; this one guards KICK-mode terms for envs just past a locomotion->kick
    mid-episode entry, since a still-locomotion-typical pose landing in kick-mode's stricter
    tracking tolerance is otherwise a plausible instant kill before the reward ramp
    (``pre_kick_reward_ramp_steps``) or the reference blend (increment 3, not yet implemented)
    have had any chance to take effect.

    ``grace_steps<=0`` (the field's own off default) returns an all-False mask unconditionally --
    same exact-no-op discipline as ``_post_flip_grace_active``: short-circuits before even
    touching ``env._pre_kick_step``, so a non-``UnifiedManager`` env (no ``_pre_kick_step``/
    ``pre_kick_steps_since`` at all) never needs to be handled as a special case either.

    2026-08-15, "simultaneous per-skill task configs": ``grace_steps`` may also be a per-env
    ``[num_envs]`` tensor -- same tensor-aware fast-path handling as ``_post_flip_grace_active``
    above, see that function's own 2026-08-15 comment for the full rationale."""
    if not torch.is_tensor(grace_steps) and grace_steps <= 0.0:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    pre_kick_step = getattr(env, "_pre_kick_step", None)
    if pre_kick_step is None or not hasattr(env, "pre_kick_steps_since"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    is_pre_kick = pre_kick_step >= 0
    active = is_pre_kick & (env.pre_kick_steps_since().float() < grace_steps)
    if torch.is_tensor(grace_steps):
        active = active & (grace_steps > 0.0)
    return active


def base_height_below_threshold_sustained_pre_kick_graced(
    env,
    min_height: float,
    consecutive_steps: int = 10,
    counter_attr: str = "_low_height_counter",
    pre_kick_grace_steps: float = 0.0,
) -> torch.Tensor:
    """Identical to ``base_height_below_threshold_sustained`` (calls it directly, unmodified, so
    its own below-threshold counter still advances normally every tick), except envs currently
    within ``pre_kick_grace_steps`` of a mid-episode locomotion->kick entry never contribute to
    the result. Mirror of ``base_height_below_threshold_sustained_post_flip_graced``, opposite
    boundary -- see that function's and ``_pre_kick_grace_active``'s own docstrings.
    ``pre_kick_grace_steps<=0`` (default) is an exact no-op, bit-identical to calling
    ``base_height_below_threshold_sustained`` directly. Registered IN PLACE OF ``kick_low_height``
    for the unified config only (config_values/unified/g1/termination.py)."""
    result = base_height_below_threshold_sustained(
        env, min_height=min_height, consecutive_steps=consecutive_steps, counter_attr=counter_attr
    )
    return result & ~_pre_kick_grace_active(env, pre_kick_grace_steps)


def contact_forces_exceeded_post_flip_graced(
    env,
    force_threshold: float = 1.0,
    contact_indices_attr: str = "termination_contact_indices",
    post_flip_grace_steps: float = 0.0,
) -> torch.Tensor:
    """Identical to ``contact_forces_exceeded`` (calls it directly, unmodified), except envs
    currently within ``post_flip_grace_steps`` of a kick->locomotion flip never contribute to the
    result -- see ``_post_flip_grace_active``'s own docstring. ``post_flip_grace_steps<=0``
    (default) is an exact no-op, bit-identical to calling ``contact_forces_exceeded`` directly.
    Registered IN PLACE OF ``contact`` for the unified config only (config_values/unified/g1/
    termination.py) -- the loco-only baseline's own registration is untouched, so a standalone
    locomotion experiment is unaffected regardless of this field's value."""
    result = contact_forces_exceeded(env, force_threshold=force_threshold, contact_indices_attr=contact_indices_attr)
    return result & ~_post_flip_grace_active(env, post_flip_grace_steps)


def base_height_below_threshold_sustained_post_flip_graced(
    env,
    min_height: float,
    consecutive_steps: int = 10,
    counter_attr: str = "_low_height_counter",
    post_flip_grace_steps: float = 0.0,
) -> torch.Tensor:
    """Identical to ``base_height_below_threshold_sustained`` (calls it directly, unmodified, so
    its own below-threshold counter still advances normally every tick -- grace only affects
    whether the ALREADY-SUSTAINED result is allowed to terminate, not the height check itself),
    except envs currently within ``post_flip_grace_steps`` of a kick->locomotion flip never
    contribute to the result. See ``contact_forces_exceeded_post_flip_graced``'s own docstring for
    the shared rationale and the exact-no-op guarantee at ``post_flip_grace_steps<=0``. Registered
    IN PLACE OF ``low_height`` for the unified config only -- untouched for the loco-only
    baseline."""
    result = base_height_below_threshold_sustained(
        env, min_height=min_height, consecutive_steps=consecutive_steps, counter_attr=counter_attr
    )
    return result & ~_post_flip_grace_active(env, post_flip_grace_steps)


def joint_pos_sanity_exceeded(env, joint_pos_sanity_threshold: float = 20.0) -> torch.Tensor:
    """Terminate an env whose ``env.simulator.dof_pos`` has gone non-finite (NaN/Inf) or exceeds
    ``joint_pos_sanity_threshold`` in absolute value -- a rare per-env physics-solver numerical
    explosion (e.g. a contact/collision-resolution edge case), not a reference-tracking-deviation
    check like ``BadTracking``.

    Deliberately checks ``torch.isfinite`` FIRST, before the magnitude comparison: a NaN
    comparison (``nan > threshold``) silently evaluates to False under IEEE 754 / PyTorch
    semantics, so a magnitude-only threshold would miss a true NaN blowup entirely -- exactly the
    failure mode this term exists to catch (see MultiSkillConfig.joint_pos_sanity_check_enabled's
    own docstring for the live incident this was built from: a single env's joint_pos spiking to
    2.36e8 for one tick, with the SAC critic loss going NaN at the same step).

    Deliberately task_mode-agnostic (register this term untagged/task_mode=None, matching
    ``timeout``) -- this is a robot-state integrity check, not a reference-fidelity check, so it
    should apply regardless of which task_mode is currently active.

    Only shortens how long an already-exploded env's corrupted state persists -- termination and
    reward are computed in the SAME tick, before any reset, so this alone cannot prevent that
    tick's already-corrupted reward/observation from reaching the replay buffer. See
    FastSACConfig.replay_buffer_sanitize_enabled for the complementary write-boundary guard.
    """
    dof_pos = env.simulator.dof_pos
    non_finite = ~torch.isfinite(dof_pos)
    exceeded = dof_pos.abs() > joint_pos_sanity_threshold
    return (non_finite | exceeded).any(dim=-1)


def dof_position_limit_exceeded(env, probability: float = 1.0) -> torch.Tensor:
    """Terminate when DOF position limits are exceeded."""
    if not getattr(env.config.termination, "terminate_when_close_to_dof_pos_limit", False):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    lower_violation = -(env.simulator.dof_pos - env.simulator.dof_pos_limits_termination[:, 0]).clip(max=0.0)
    upper_violation = (env.simulator.dof_pos - env.simulator.dof_pos_limits_termination[:, 1]).clip(min=0.0)
    violation = torch.sum(lower_violation + upper_violation, dim=1) > 0.0
    return _apply_probability(violation, probability, env.device)


def dof_velocity_limit_exceeded(env, probability: float = 1.0) -> torch.Tensor:
    """Terminate when DOF velocity limits are exceeded."""
    if not getattr(env.config.termination, "terminate_when_close_to_dof_vel_limit", False):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    delta = (
        torch.abs(env.simulator.dof_vel)
        - env.dof_vel_limits * env.config.termination_scales.termination_close_to_dof_vel_limit
    ).clip(min=0.0, max=1.0)
    violation = torch.sum(delta, dim=1) > 0.0
    return _apply_probability(violation, probability, env.device)


def torque_limit_exceeded(env, probability: float = 1.0) -> torch.Tensor:
    """Terminate when actuator torques exceed limits."""
    if not getattr(env.config.termination, "terminate_when_close_to_torque_limit", False):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    torques = env.action_manager.get_term("joint_control").torques
    delta = (
        torch.abs(torques) - env.torque_limits * env.config.termination_scales.termination_close_to_torque_limit
    ).clip(min=0.0, max=1.0)
    violation = torch.sum(delta, dim=1) > 0.0
    return _apply_probability(violation, probability, env.device)
