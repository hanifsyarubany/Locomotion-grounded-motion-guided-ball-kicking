"""Thin wrappers applying the new per-skill reward-category scales (utils/kick_reward_scales.py)
to reward terms that are SHARED with other task modes / experiments, so the scale can't be
applied by editing the shared function directly without also affecting locomotion mode or
standalone WBT training:

- ``kick_alive`` reuses locomotion's own ``alive`` (managers/reward/terms/locomotion.py) --
  editing ``alive`` itself would also rescale locomotion mode's identical-function reward.
- The 7 kick-mode motion-tracking terms reuse WBT's own reward functions
  (managers/reward/terms/wbt.py) -- editing those would also rescale standalone WBT training,
  which has nothing to do with this project's per-kick-skill scales. (6 terms until 2026-08-05,
  when motion_global_feet_lin_vel_scaled was added as the 7th -- RoboNaldo's own
  motion_feet_lin_vel, see that wrapper's own docstring below.)

Each wrapper below calls the ORIGINAL, UNMODIFIED function and multiplies by this term's live
per-skill scale; the underlying computation is untouched. (Contrast with the 6 kick-recovery-
posture terms and 3 kick-safety terms, which ARE kick-only-dedicated functions already -- those
get the scale multiplied in directly, in locomotion.py / kick_safety.py respectively, no wrapper
needed.)

The 7 motion-tracking wrappers ALSO apply _recovery_phase_tracking_multiplier, on top of
motion_tracking_reward_scale -- see that helper's own docstring and
SkillConfig.recovery_tracking_scale's docstring (config_types/multi_skill.py) for the full
rationale (the synthetic post-swing recovery/hold clip has no balance information, so tracking it
at full strength penalizes exactly the corrective motion that would prevent a fall). 1.0 during
"swinging mode" (locomotion-approach + strike) always; recovery_tracking_scale(env) once
in_kicking_phase is False. This is the phase-gating step -- kick_reward_scales.py's own
recovery_tracking_scale() has no phase awareness, it only resolves the flat per-skill value.

The 7 motion-tracking wrappers ALSO widen sigma during the strike specifically (2026-08-01, via
_swing_widened_sigma below), controlled by MultiSkillConfig.swing_tracking_sigma_multiplier
(global, not per-skill; 1.0 = off) -- see that field's own docstring for the full crossover-
arithmetic rationale and an explicit precedent warning worth reading before ever setting it above
1.0. Unlike motion_tracking_reward_scale (multiplies the OUTPUT), this changes sigma BEFORE it
reaches the underlying WBT function, since sigma sits inside the exp() denominator, not as an
output multiplier -- see _swing_widened_sigma's own docstring.

ONLY the 2 root/anchor wrappers (motion_global_ref_position_error_exp_scaled,
motion_global_ref_orientation_error_exp_scaled) ALSO apply root_tracking_reward_scale, 2026-08-05,
ported from RoboNaldo (arXiv:2606.11092) -- see SkillConfig.root_tracking_reward_scale's own
docstring (config_types/multi_skill.py) for the full rationale. The other 5 (2 relative-body pose
terms + 2 global body-velocity terms + the feet-linear-velocity term added the same day) never
read this scale; motion_tracking_reward_scale alone still governs them, unchanged from before this
field existed.

FIX 4 (2026-08-12): all 7 motion-tracking wrappers ALSO apply _post_flip_tracking_decay_multiplier
-- see that function's own docstring for the full measured motivation and exact-no-op guarantee at
MultiSkillConfig.post_flip_reward_decay_steps's 0.0 default. Unlike every other multiplier in this
module (which only ever attenuate KICK-mode reward), this one is the one place a motion-tracking
term's contribution can be nonzero for an env whose CURRENT task_mode is LOCOMOTION -- which is
also why enabling it requires switching these 7 terms' registration to task_mode=None
(config_values/unified/g1/reward.py's _apply_post_flip_reward_decay), not just adding a param.

2026-08-13: all 7 also apply _pre_kick_reward_ramp_multiplier -- the mirror of FIX 4 at the
opposite boundary (mid_episode_kick_entry_prob's locomotion->kick handoff). See that function's
own docstring for the full rationale and exact-no-op guarantee at
MultiSkillConfig.pre_kick_reward_ramp_steps's 0.0 default. Unlike FIX 4, no registration change is
needed -- a mid-episode entry's env is already task_mode="kick" by the time this runs, so the
existing masking already includes it; this multiplier only attenuates, never revives, a
contribution.
"""

from __future__ import annotations

from typing import Any

import torch

from holosoma.managers.reward.terms.locomotion import alive as _alive
from holosoma.managers.reward.terms.wbt import (
    motion_global_body_ang_vel as _motion_global_body_ang_vel,
    motion_global_body_lin_vel as _motion_global_body_lin_vel,
    motion_global_feet_lin_vel as _motion_global_feet_lin_vel,
    motion_global_ref_orientation_error_exp as _motion_global_ref_orientation_error_exp,
    motion_global_ref_position_error_exp as _motion_global_ref_position_error_exp,
    motion_relative_body_orientation_error_exp as _motion_relative_body_orientation_error_exp,
    motion_relative_body_position_error_exp as _motion_relative_body_position_error_exp,
)
from holosoma.utils.kick_reward_scales import (
    kick_alive_pre_kick_ratio,
    kick_alive_reward_scale,
    motion_tracking_reward_scale,
    recovery_tracking_scale,
    root_tracking_reward_scale,
)


def kick_alive_scaled(env: Any) -> torch.Tensor:
    """kick_alive, scaled by kick_alive_reward_scale (flat, every phase) and additionally
    phase-shaped by kick_alive_pre_kick_ratio (2026-08-05, ported from RoboNaldo -- see that
    field's own docstring in config_types/multi_skill.py for the full rationale): ratio==1.0 is an
    exact no-op (same discipline as _swing_widened_sigma below -- skip the phase read entirely at
    the default), otherwise base is multiplied by ratio while motion_command.in_kicking_phase is
    True (approach + strike), left unchanged during post-kick recovery/hold.

    No None-guard on motion_command itself: kick_alive_reward_scale (called just above, via
    _per_env_scale) already unconditionally dereferences motion_command.motion_ids, so a missing
    motion_command would already have raised there -- same "no-guard read" precedent
    _recovery_phase_tracking_multiplier's own docstring documents. has_ball IS guarded: a
    non-ball-kick MotionCommand variant (has_ball=False) has no in_kicking_phase semantics worth
    trusting here, so the ratio is skipped in that case."""
    base = _alive(env) * kick_alive_reward_scale(env)
    ratio = kick_alive_pre_kick_ratio(env)  # always a [num_envs] tensor, per _per_env_scale
    if bool(torch.all(ratio == 1.0)):  # exact no-op path when every env's skill leaves it at 1.0
        return base
    motion_command = env.command_manager.get_state("motion_command")
    if not getattr(motion_command, "has_ball", False):
        return base
    return torch.where(motion_command.in_kicking_phase, base * ratio, base)


def alive_post_flip_scaled(env: Any, post_flip_alive_scale: float = 1.0) -> torch.Tensor:
    """Locomotion's own ``alive`` (flat +1.0/step), scaled by ``post_flip_alive_scale`` for envs
    that have crossed the kick->locomotion flip -- see MultiSkillConfig.post_flip_alive_scale's
    own docstring in config_types/multi_skill.py for the full measured rationale (alive-farming:
    a policy that skips the kick and simply outlasts the clip collects this term at full value for
    the whole post-flip remainder of the episode, a cheaper payout than kicking and risking the
    fall/self-collision that swinging exposes it to).

    ``env._post_flip_step >= 0`` (UnifiedManager, same sentinel FIX 4's
    ``_post_flip_tracking_decay_multiplier`` and the post-flip obs-ramp fixes already key off) is
    the discriminator, NOT ``env._task_mode_partition == KICK`` alone -- a kick-partitioned env
    also spends live task_mode==LOCOMOTION time in its PRE-kick approach (before it has ever
    entered the clip), and that phase must NOT be scaled down; only genuine post-kick recovery
    should be. ``_post_flip_step`` stays -1 for the entire life of both that pre-kick-approach case
    and a genuinely locomotion-partitioned env (only ``_update_kick_recovery_locomotion_flip``
    ever writes to it, and only for envs currently in KICK mode), so both are correctly left
    unscaled here.

    ``post_flip_alive_scale == 1.0`` (default) returns ``base`` completely untouched -- an exact,
    not just numerically-equal, no-op (same discipline as ``kick_alive_scaled``'s own ratio==1.0
    short-circuit above): no ``_post_flip_step`` read, no ``torch.where``, so every existing config
    that doesn't set this field is bit-for-bit unaffected."""
    base = _alive(env)
    if not torch.is_tensor(post_flip_alive_scale) and post_flip_alive_scale == 1.0:
        return base
    if not hasattr(env, "_post_flip_step"):
        return base
    is_post_flip = env._post_flip_step >= 0
    scale = post_flip_alive_scale
    if not torch.is_tensor(scale):
        scale = torch.full_like(base, scale)
    return torch.where(is_post_flip, base * scale, base)


def _recovery_phase_tracking_multiplier(env: Any) -> torch.Tensor:
    """1.0 during "swinging mode" (locomotion-approach + strike; recovery_tracking_scale has zero
    effect there -- tracking strength in that window is governed by motion_tracking_reward_scale
    alone); recovery_tracking_scale(env) once the clip is past that (post-kick-standing mode,
    recovery, hold). No-guard read of motion_command/in_kicking_phase: Unified's MotionCommand is
    always the ball-kick variant (has_ball=True by construction, see
    config_values/unified/g1/command.py), and stand_start_idx (which in_kicking_phase is derived
    from) is populated unconditionally in MotionCommand.setup() regardless of N-skill vs
    single-clip mode -- same precedent as _kick_recovery_gate's own per-skill boundary read."""
    motion_command = env.command_manager.get_state("motion_command")
    scale = recovery_tracking_scale(env)
    return torch.where(motion_command.in_kicking_phase, torch.ones_like(scale), scale)


def _swing_widened_sigma(
    env: Any, sigma: float, swing_tracking_sigma_multiplier: float
) -> float | torch.Tensor:
    """The sigma value to actually pass into a motion-tracking term's underlying WBT function:
    ``sigma`` unchanged (same python float, same object) when swing_tracking_sigma_multiplier ==
    1.0 -- an exact type/value no-op, not just a numerically-equal one, so every existing caller
    and test of the *_scaled wrappers is untouched at the default. Otherwise, a per-env tensor:
    ``sigma * swing_tracking_sigma_multiplier`` while ``motion_command.in_strike_phase`` is True,
    plain ``sigma`` elsewhere.

    Widening sigma (not narrowing) is the only sane direction here -- the underlying WBT functions
    divide by sigma**2 inside exp(), so this must stay > 0.0 (enforced at config-load time in
    multi_skill.py / simulator.py, not re-checked here).

    Passing a tensor instead of a float works with zero changes to the underlying WBT functions:
    every one of the 6 (managers/reward/terms/wbt.py) computes exp(-error/sigma**2), which is
    already elementwise over sigma via normal torch broadcasting regardless of whether sigma is a
    python float or a [num_envs] tensor -- confirmed by reading all 6 function bodies, none do
    anything else with sigma."""
    if swing_tracking_sigma_multiplier == 1.0:
        return sigma
    motion_command = env.command_manager.get_state("motion_command")
    in_strike = motion_command.in_strike_phase
    widened = torch.full_like(in_strike, sigma * swing_tracking_sigma_multiplier, dtype=torch.float32)
    unwidened = torch.full_like(in_strike, sigma, dtype=torch.float32)
    return torch.where(in_strike, widened, unwidened)


def _post_flip_tracking_decay_multiplier(env: Any, post_flip_reward_decay_steps: float) -> float | torch.Tensor:
    """FIX 4 (2026-08-12): the multiplier that keeps motion-tracking reward alive for a decaying
    window after the kick->locomotion flip, instead of the instant cliff to exactly zero every
    tracking term hits today (task_mode_mask("kick") zeroing the whole reward the tick task_mode
    changes). Motivation: SAC's critic bootstraps Q(s,a) through the WHOLE trajectory, so the
    value of a mid-swing state already depends on what happens after the flip; a live diagnostic
    (2026-08-12, matched FLIP-vs-NOFLIP checkpoints, same clip, same [204,300) mechanism in both)
    found termination-condition firing rates 2.5x-6.6x higher for the FLIP-trained policy in that
    SHARED pre-boundary window -- i.e. training under the instant cliff measurably destabilizes
    behavior even before the boundary is reached, consistent with the value function seeing a
    cliff in its bootstrapped target the instant a state is close enough to the flip to reach past
    it. A ramped handoff gives the critic a smooth target to bootstrap through instead.

    ``post_flip_reward_decay_steps<=0.0`` (default) returns the plain python float ``1.0``
    unconditionally -- an exact, not just numerically-equal, no-op: the caller's own registration
    (config_values/unified/g1/reward.py) leaves these terms at task_mode="kick" in that case, so
    the outer RewardManager masking already zeroes every post-flip env's contribution regardless
    of what this returns; returning a bare float (not a tensor) also means multiplying by it can
    never introduce so much as a floating-point rounding difference for the still-task_mode="kick"
    case this project has run under until now.

    When enabled (registration also switches these terms to task_mode=None -- see
    _apply_post_flip_reward_decay below -- so the outer manager stops masking post-flip envs
    itself and this function becomes the ONLY thing deciding their contribution): 1.0 for envs
    currently in kick mode (unchanged); a linear ramp from 1.0 down to 0.0 over
    ``post_flip_reward_decay_steps`` ticks for envs within that many ticks of their OWN flip
    (env.post_flip_steps_since(), UnifiedManager); exactly 0.0 for every other env -- both
    envs that flipped longer ago than the decay window, AND genuinely locomotion-partitioned envs
    that were never kick-partitioned at all (env._post_flip_step stays -1 for those for their
    entire life, since only _maybe_flip_kick_recovery_to_locomotion ever writes to it, and that
    method only ever touches currently-KICK envs -- so a pure-locomotion env can never be
    misidentified as "post-flip" here)."""
    # 2026-08-15, "simultaneous per-skill task configs": post_flip_reward_decay_steps may also be
    # a per-env [num_envs] tensor (RewardManager's params_per_skill gather) instead of the plain
    # float it always used to be. The division/comparisons below are already elementwise-safe
    # against a tensor; only the all-1.0 fast path and the tensor's own <=0 envs need explicit
    # handling (those envs get decay=1.0 too, matching the scalar no-op exactly).
    if not torch.is_tensor(post_flip_reward_decay_steps) and post_flip_reward_decay_steps <= 0.0:
        return 1.0
    if not hasattr(env, "task_mode_mask") or not hasattr(env, "_post_flip_step"):
        return 1.0
    currently_kick = env.task_mode_mask("kick")
    is_post_flip = env._post_flip_step >= 0
    steps_since = env.post_flip_steps_since().float()
    denom = post_flip_reward_decay_steps
    if torch.is_tensor(denom):
        denom = denom.clamp(min=1e-6)
    decay = (1.0 - steps_since / denom).clamp(min=0.0, max=1.0)
    if torch.is_tensor(post_flip_reward_decay_steps):
        # Envs whose OWN decay_steps<=0 ("instant cliff" off-default) must land at exactly 0.0
        # once post-flip, matching legacy semantics precisely -- NOT the partial ramp value the
        # denom.clamp(min=1e-6) division above would otherwise produce, and NOT an unconditional
        # 1.0 either (that would wrongly revive their contribution post-flip). The outer
        # torch.where below already handles the pre-flip (currently_kick) case as an unconditional
        # 1.0 regardless of `decay`, so this only needs to correct the post-flip branch.
        decay = torch.where(post_flip_reward_decay_steps > 0.0, decay, torch.zeros_like(decay))
    return torch.where(currently_kick, torch.ones_like(decay), torch.where(is_post_flip, decay, torch.zeros_like(decay)))


def _pre_kick_reward_ramp_multiplier(env: Any, pre_kick_reward_ramp_steps: float) -> float | torch.Tensor:
    """2026-08-13: mirror of _post_flip_tracking_decay_multiplier, opposite direction and opposite
    boundary (mid_episode_kick_entry_prob's locomotion->kick handoff). Instead of an instant
    full-strength jump the moment a mid-episode entry flips task_mode to KICK, ramps these same 7
    terms' contribution LINEARLY UP from 0 to full strength over pre_kick_reward_ramp_steps ticks
    -- same "give the critic a smooth target to bootstrap through instead of a cliff" rationale as
    FIX 4, at the opposite boundary: increment 1's reference re-anchoring already makes the
    ACTOR's input near-continuous at entry, but the critic still bootstraps a value estimate
    through states that were, one tick earlier, worth a completely different reward (locomotion).

    Unlike FIX 4, no registration change is needed: a mid-episode entry's env is already
    task_mode="kick" by the time this runs (the flip happens within the same tick, in
    UnifiedManager._enter_kick, before RewardManager evaluates this tick's reward), so the
    existing task_mode="kick" masking already includes it -- this multiplier only needs to
    ATTENUATE an already-included contribution, never revive an excluded one.

    pre_kick_reward_ramp_steps<=0.0 (default) returns the plain python float 1.0 unconditionally
    -- exact no-op, multiplying by it can never introduce even a rounding difference.

    When enabled: 1.0 for every env whose kick episode did NOT begin with a mid-episode entry
    (_pre_kick_step == -1, the overwhelming majority -- ordinary teleport-at-reset kick envs are
    completely unaffected, matching today's behavior exactly); for envs that DID
    (_pre_kick_step >= 0), a linear ramp from 0.0 up to 1.0 over pre_kick_reward_ramp_steps ticks
    since entry (env.pre_kick_steps_since(), UnifiedManager), then 1.0 thereafter. D5: additionally
    gated on _post_flip_step < 0 -- an env currently inside FIX 4's own post-flip decay window is
    left at that mechanism's own multiplier untouched (1.0 here) rather than double-ramped;
    structurally this env can never also be pre-kick-recent (_maybe_enter_kick_from_locomotion
    never re-arms _kick_pending for an env post-flip), but the guard is kept explicit rather than
    relying on that invariant silently."""
    # 2026-08-15, "simultaneous per-skill task configs": pre_kick_reward_ramp_steps may also be a
    # per-env [num_envs] tensor -- same tensor-aware handling as _post_flip_tracking_decay_
    # multiplier above, see that function's own 2026-08-15 comment for the full rationale.
    if not torch.is_tensor(pre_kick_reward_ramp_steps) and pre_kick_reward_ramp_steps <= 0.0:
        return 1.0
    if not hasattr(env, "pre_kick_steps_since") or not hasattr(env, "_pre_kick_step"):
        return 1.0
    is_pre_kick = (env._pre_kick_step >= 0) & (env._post_flip_step < 0)
    steps_since = env.pre_kick_steps_since().float()
    denom = pre_kick_reward_ramp_steps
    if torch.is_tensor(denom):
        denom = denom.clamp(min=1e-6)
    ramp = (steps_since / denom).clamp(min=0.0, max=1.0)
    if torch.is_tensor(pre_kick_reward_ramp_steps):
        ramp = torch.where(pre_kick_reward_ramp_steps > 0.0, ramp, torch.ones_like(ramp))
    return torch.where(is_pre_kick, ramp, torch.ones_like(ramp))


def motion_global_ref_position_error_exp_scaled(
    env: Any,
    sigma: float,
    swing_tracking_sigma_multiplier: float = 1.0,
    post_flip_reward_decay_steps: float = 0.0,
    pre_kick_reward_ramp_steps: float = 0.0,
) -> torch.Tensor:
    return (
        _motion_global_ref_position_error_exp(env, _swing_widened_sigma(env, sigma, swing_tracking_sigma_multiplier))
        * motion_tracking_reward_scale(env)
        * root_tracking_reward_scale(env)
        * _recovery_phase_tracking_multiplier(env)
        * _post_flip_tracking_decay_multiplier(env, post_flip_reward_decay_steps)
        * _pre_kick_reward_ramp_multiplier(env, pre_kick_reward_ramp_steps)
    )


def motion_global_ref_orientation_error_exp_scaled(
    env: Any,
    sigma: float,
    swing_tracking_sigma_multiplier: float = 1.0,
    post_flip_reward_decay_steps: float = 0.0,
    pre_kick_reward_ramp_steps: float = 0.0,
) -> torch.Tensor:
    return (
        _motion_global_ref_orientation_error_exp(
            env, _swing_widened_sigma(env, sigma, swing_tracking_sigma_multiplier)
        )
        * motion_tracking_reward_scale(env)
        * root_tracking_reward_scale(env)
        * _recovery_phase_tracking_multiplier(env)
        * _post_flip_tracking_decay_multiplier(env, post_flip_reward_decay_steps)
        * _pre_kick_reward_ramp_multiplier(env, pre_kick_reward_ramp_steps)
    )


def motion_relative_body_position_error_exp_scaled(
    env: Any,
    sigma: float,
    swing_tracking_sigma_multiplier: float = 1.0,
    post_flip_reward_decay_steps: float = 0.0,
    pre_kick_reward_ramp_steps: float = 0.0,
) -> torch.Tensor:
    return (
        _motion_relative_body_position_error_exp(
            env, _swing_widened_sigma(env, sigma, swing_tracking_sigma_multiplier)
        )
        * motion_tracking_reward_scale(env)
        * _recovery_phase_tracking_multiplier(env)
        * _post_flip_tracking_decay_multiplier(env, post_flip_reward_decay_steps)
        * _pre_kick_reward_ramp_multiplier(env, pre_kick_reward_ramp_steps)
    )


def motion_relative_body_orientation_error_exp_scaled(
    env: Any,
    sigma: float,
    swing_tracking_sigma_multiplier: float = 1.0,
    post_flip_reward_decay_steps: float = 0.0,
    pre_kick_reward_ramp_steps: float = 0.0,
) -> torch.Tensor:
    return (
        _motion_relative_body_orientation_error_exp(
            env, _swing_widened_sigma(env, sigma, swing_tracking_sigma_multiplier)
        )
        * motion_tracking_reward_scale(env)
        * _recovery_phase_tracking_multiplier(env)
        * _post_flip_tracking_decay_multiplier(env, post_flip_reward_decay_steps)
        * _pre_kick_reward_ramp_multiplier(env, pre_kick_reward_ramp_steps)
    )


def motion_global_body_lin_vel_scaled(
    env: Any,
    sigma: float,
    swing_tracking_sigma_multiplier: float = 1.0,
    post_flip_reward_decay_steps: float = 0.0,
    pre_kick_reward_ramp_steps: float = 0.0,
) -> torch.Tensor:
    return (
        _motion_global_body_lin_vel(env, _swing_widened_sigma(env, sigma, swing_tracking_sigma_multiplier))
        * motion_tracking_reward_scale(env)
        * _recovery_phase_tracking_multiplier(env)
        * _post_flip_tracking_decay_multiplier(env, post_flip_reward_decay_steps)
        * _pre_kick_reward_ramp_multiplier(env, pre_kick_reward_ramp_steps)
    )


def motion_global_body_ang_vel_scaled(
    env: Any,
    sigma: float,
    swing_tracking_sigma_multiplier: float = 1.0,
    post_flip_reward_decay_steps: float = 0.0,
    pre_kick_reward_ramp_steps: float = 0.0,
) -> torch.Tensor:
    return (
        _motion_global_body_ang_vel(env, _swing_widened_sigma(env, sigma, swing_tracking_sigma_multiplier))
        * motion_tracking_reward_scale(env)
        * _recovery_phase_tracking_multiplier(env)
        * _post_flip_tracking_decay_multiplier(env, post_flip_reward_decay_steps)
        * _pre_kick_reward_ramp_multiplier(env, pre_kick_reward_ramp_steps)
    )


def motion_global_feet_lin_vel_scaled(
    env: Any,
    sigma: float,
    swing_tracking_sigma_multiplier: float = 1.0,
    body_names: list[str] | None = None,
    post_flip_reward_decay_steps: float = 0.0,
    pre_kick_reward_ramp_steps: float = 0.0,
) -> torch.Tensor:
    """The 7th motion-tracking wrapper -- 2026-08-05, ported from RoboNaldo's
    ``motion_feet_lin_vel`` (see ``wbt.py:motion_global_feet_lin_vel``'s own docstring for the
    formula-fidelity note). Same treatment as the 4 non-root siblings above (NOT
    root_tracking_reward_scale -- see this module's own docstring for why): motion_tracking_
    reward_scale, phase-gated recovery relaxation, swing-sigma widening, and (2026-08-12)
    post-flip decay, all unchanged from the same-shaped siblings.

    Unlike the other 6 wrappers, this one also forwards ``body_names`` -- the underlying
    ``motion_global_feet_lin_vel`` (unlike its 6 siblings) takes a body-subset parameter, and its
    registration (config_values/wbt/g1/reward.py) sets it explicitly (both ankle_roll links), so
    the wrapper must accept and pass it through rather than silently dropping it -- caught live by
    this project's own probe-based verification discipline (a bare ``TypeError: ... unexpected
    keyword argument 'body_names'`` the first time this wrapper actually ran in kick mode)."""
    return (
        _motion_global_feet_lin_vel(
            env, _swing_widened_sigma(env, sigma, swing_tracking_sigma_multiplier), body_names=body_names
        )
        * motion_tracking_reward_scale(env)
        * _recovery_phase_tracking_multiplier(env)
        * _post_flip_tracking_decay_multiplier(env, post_flip_reward_decay_steps)
        * _pre_kick_reward_ramp_multiplier(env, pre_kick_reward_ramp_steps)
    )
