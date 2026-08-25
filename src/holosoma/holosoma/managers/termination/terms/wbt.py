"""Whole Body Tracking-specific termination terms."""

from __future__ import annotations

from typing import Any, List

from holosoma.config_types.termination import TerminationTermCfg
from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager
from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.managers.observation.terms.wbt import gravity_vector
from holosoma.managers.termination.base import TerminationTermBase
from holosoma.managers.termination.terms.locomotion import _pre_kick_grace_active
from holosoma.utils.rotations import (
    quat_error_magnitude,
    quat_rotate_inverse,
)
from holosoma.utils.safe_torch_import import torch


#########################################################################################################
## Termination terms
#########################################################################################################
def motion_ends(env, **_) -> torch.Tensor:
    """Terminate if the motion ends."""
    motion_command = env.command_manager.get_state("motion_command")
    return motion_command.time_steps >= motion_command.motion.time_step_total - 2


def kick_recovery_low_height_sustained(
    env,
    min_height: float = 0.70,
    consecutive_steps: int = 10,
    grace_steps: float = 50.0,
    counter_attr: str = "_kick_recovery_low_height_counter",
    enabled: bool | torch.Tensor = True,
) -> torch.Tensor:
    """Kick-mode REPLACEMENT for ``bad_tracking`` during the post-kick recovery/hold window: a
    physically-grounded, clip-independent height floor -- the SAME check, with the SAME
    ``min_height=0.70``/``consecutive_steps=10`` values, that locomotion mode's own standing
    termination already uses and trusts (see ``base_height_below_threshold_sustained``'s
    ``_low_height_term`` registration in ``config_values/unified/g1/termination.py``) -- rather
    than a check that a specific scripted recovery-clip trajectory is being matched.

    Only ever wired into the term manager when ``kick_recovery_termination_handoff`` is True (see
    ``config_values/unified/g1/termination.py``), which ALSO forces ``bad_tracking_swing_only``
    True for the same envs at the same time -- structurally, this can never be active without
    ``bad_tracking`` simultaneously being suppressed for the same window, and vice versa. This
    matters because ``bad_tracking_swing_only`` alone (no replacement) was already measured to
    make things WORSE (per-cycle fall hazard 0.058-0.076 -> 0.100-0.130): with nothing installed
    in ``bad_tracking``'s place, ``kick_low_height``'s absolute 0.40m floor was the ONLY thing
    left watching recovery/hold, so a robot visibly losing control just kept running, uncaught, in
    a degraded state, for longer before an eventual real fall.

    ``grace_steps=50.0`` mirrors ``_kick_recovery_gate``'s own default
    (``managers/reward/terms/locomotion.py``) exactly, itself aligned to
    ``SkillConfig.recovery_duration_s``'s default (1.0s at dt=0.02) -- the counter cannot even
    START incrementing until ``grace_steps`` steps into recovery have elapsed, the same "don't
    demand a settled stance the instant recovery starts, the robot is still carrying real
    momentum" lesson ``_standing_gate``'s own grace_steps documents (v9 Stage-A: 7/10 falls from
    an instant snap to full standing demands mid-momentum).

    The phase+grace gate is folded INTO the increment condition itself (``active & below``), NOT
    applied as an external post-hoc mask on top of an otherwise phase-unaware counter:
    ``base_height_below_threshold_sustained``'s own counter starts incrementing the moment height
    drops below threshold regardless of phase, so an external AND-mask could inherit counter
    progress already accumulated during a legitimate low single-support dip in swing --
    prematurely tripping just as recovery begins. Duplicating (not reusing)
    ``base_height_below_threshold_sustained``'s counter mechanics here is deliberate for this
    reason.

    ``counter_attr`` MUST differ from every other call site's counter attribute name
    (``_low_height_counter``, ``_kick_low_height_counter``) -- every configured termination term's
    function body executes for every env every step regardless of ``task_mode`` (masking is
    post-hoc, see ``TerminationManager.check()``), so a shared attribute would let two terms
    silently corrupt each other's counter state.
    """
    motion_command = env.command_manager.get_state("motion_command")
    if motion_command is None or not getattr(motion_command, "has_ball", False):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    boundary_idx = motion_command.stand_start_idx[motion_command.motion_ids]
    in_recovery = ~motion_command.in_kicking_phase
    steps_into_recovery = torch.clamp((motion_command.time_steps - boundary_idx).float(), min=0.0)
    active = in_recovery & (steps_into_recovery >= grace_steps)

    base_height = env.simulator.robot_root_states[:, 2]
    below = base_height < min_height

    counter = getattr(env, counter_attr, None)
    if counter is None or counter.shape[0] != env.num_envs:
        counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    counter = torch.where(active & below, counter + 1, torch.zeros_like(counter))
    setattr(env, counter_attr, counter)
    result = counter >= consecutive_steps
    # 2026-08-15, "simultaneous per-skill task configs": this term is only ever REGISTERED at all
    # when kick_recovery_termination_handoff is True for at least one skill (config_values/
    # unified/g1/termination.py) -- `enabled` is what suppresses it PER-ENV for skills that don't
    # want it, via RewardManager's/TerminationManager's ordinary params_per_skill gather. True
    # (default) = exact no-op, matching every existing registration (which never set this param).
    if torch.is_tensor(enabled):
        result = result & (enabled > 0.5)
    elif not enabled:
        result = torch.zeros_like(result)
    return result


def kick_recovery_drift_sustained(
    env,
    deadzone: float = 0.15,
    consecutive_steps: int = 10,
    grace_steps: float = 50.0,
    counter_attr: str = "_kick_recovery_drift_counter",
    anchor_attr: str = "_kick_recovery_drift_anchor_xy",
    anchor_valid_attr: str = "_kick_recovery_drift_anchor_valid",
    enabled: bool | torch.Tensor = True,
) -> torch.Tensor:
    """2026-08-06, user-requested sibling to ``kick_recovery_low_height_sustained`` (same file,
    immediately above): terminate if the robot's base drifts more than ``deadzone`` meters
    (simple Euclidean XY radius, direction-agnostic) away from where it was the moment it entered
    the post-kick recovery/hold phase, sustained for ``consecutive_steps`` -- a self-referential
    check ("did you move away from where you were already stably standing"), distinct from both
    ``bad_tracking`` (divergence from the REFERENCE clip, which is itself not static during this
    window -- see the live drift-diagnostic investigation this field grew out of) and
    ``kick_recovery_low_height_sustained`` (height only, blind to horizontal sliding that
    precedes -- or substitutes for -- an outright fall).

    Structurally this is the SAME latch concept RoboNaldo (arXiv:2606.11092) uses for
    ``stabilize_anchor_pos_w`` (``mdp/commands.py``) -- capture the robot's own actual pose the
    instant recovery begins -- except RoboNaldo uses that latch as a REWARD target to pull toward
    (and only in their Stage 3, ``adapt_motion_flag``-gated, dead code in the stages this project
    follows), whereas this uses the identical latch as a TERMINATION boundary instead. Only ever
    wired into the term manager alongside ``kick_recovery_low_height_sustained`` -- both gated by
    ``kick_recovery_termination_handoff`` (``config_values/unified/g1/termination.py``) -- so it
    inherits that flag's own MEASURED CONCERNING history: read
    ``MultiSkillConfig.kick_recovery_termination_handoff``'s docstring before enabling either.

    ANCHOR LATCHING: the anchor is captured once per episode, at the first step where the robot is
    no longer ``in_kicking_phase`` (i.e. the swing→recovery transition), and held fixed from then
    on -- NOT re-latched every step (that would make drift unmeasurable by construction). Two
    conditions force re-validation of the anchor, so a stale value from a PRIOR episode can never
    leak into a new one:
      1. ``in_kicking_phase`` True (approach/strike) always invalidates -- guarantees a fresh
         latch is captured at the correct transition instant for the normal case.
      2. ``episode_length_buf <= 1`` (this env's very first control step after a reset) ALSO
         unconditionally invalidates, independent of phase -- covers the edge case where RSI (or
         critical-frame oversampling) samples a starting frame already inside recovery/hold for
         this episode, skipping ``in_kicking_phase=True`` entirely; without this, such an episode
         would silently reuse whatever anchor was last latched in a PREVIOUS episode, comparing
         drift against the wrong reference point. In that edge case the anchor becomes wherever
         the episode happens to start (a reasonable fallback: "did you move from your own
         starting point," same question the normal case asks, just answered from a different
         start).

    ``anchor_valid`` (a separate persisted per-env bool, not inferred from the anchor tensor's
    value) gates BOTH the latch-write and the drift check itself -- the anchor tensor's own
    default (zeros) is not a sentinel "unset" value (a real robot position could legitimately be
    near the origin), so validity must be tracked explicitly rather than inferred from the stored
    coordinates.

    ``grace_steps``/``consecutive_steps`` mirror ``kick_recovery_low_height_sustained``'s own
    defaults exactly (same 50-step/1.0s grace ramp, same phase-gate-folded-into-the-increment-
    condition discipline, same reasoning for why an external post-hoc mask would be wrong -- see
    that function's docstring). ``counter_attr``/``anchor_attr``/``anchor_valid_attr`` MUST each
    differ from every other term's own attribute names, for the identical reason documented there
    (every registered termination term's function body runs every step for every env regardless of
    task_mode; a shared attribute name would let two terms silently corrupt each other's state).

    Deadzone default 0.15m (15cm), user-specified starting point -- not yet measured against live
    telemetry. Cross-check once available: the drift-diagnostic probe this field grew out of found
    ~4-6cm of drift typical for envs that went on to survive their recovery window cleanly, so
    15cm is deliberately well above ordinary balancing variance, not a tight trigger."""
    motion_command = env.command_manager.get_state("motion_command")
    if motion_command is None or not getattr(motion_command, "has_ball", False):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    boundary_idx = motion_command.stand_start_idx[motion_command.motion_ids]
    in_kick = motion_command.in_kicking_phase
    in_recovery = ~in_kick
    steps_into_recovery = torch.clamp((motion_command.time_steps - boundary_idx).float(), min=0.0)
    active = in_recovery & (steps_into_recovery >= grace_steps)

    root_xy = env.simulator.robot_root_states[:, :2]

    anchor_xy = getattr(env, anchor_attr, None)
    if anchor_xy is None or anchor_xy.shape[0] != env.num_envs:
        anchor_xy = torch.zeros(env.num_envs, 2, dtype=torch.float32, device=env.device)
    anchor_valid = getattr(env, anchor_valid_attr, None)
    if anchor_valid is None or anchor_valid.shape[0] != env.num_envs:
        anchor_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    is_episode_start = env.episode_length_buf <= 1
    anchor_valid = anchor_valid & ~(is_episode_start | in_kick)

    need_latch = in_recovery & ~anchor_valid
    anchor_xy = torch.where(need_latch.unsqueeze(-1), root_xy, anchor_xy)
    anchor_valid = anchor_valid | need_latch

    setattr(env, anchor_attr, anchor_xy)
    setattr(env, anchor_valid_attr, anchor_valid)

    drift = torch.norm(root_xy - anchor_xy, dim=-1)
    over_deadzone = drift > deadzone

    counter = getattr(env, counter_attr, None)
    if counter is None or counter.shape[0] != env.num_envs:
        counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    counter = torch.where(active & anchor_valid & over_deadzone, counter + 1, torch.zeros_like(counter))
    setattr(env, counter_attr, counter)
    result = counter >= consecutive_steps
    # 2026-08-15, "simultaneous per-skill task configs": same per-skill enable/suppress mechanism
    # as kick_recovery_low_height_sustained's own `enabled` param -- see that function's own
    # 2026-08-15 comment for the full rationale. True (default) = exact no-op.
    if torch.is_tensor(enabled):
        result = result & (enabled > 0.5)
    elif not enabled:
        result = torch.zeros_like(result)
    return result


class BadTracking(TerminationTermBase):
    """Terminate if the tracking is bad.

    - bad ref pos
    - bad ref ori
    - bad motion body pos
    if has object:
        - bad object pos
        - bad object ori

    When bad tracking is detected, the motion_commmand.AdaptiveTimestepsSampler will be updated.
    """

    # 2026-08-15: this class reads bad_motion_body_pos_threshold/swing_threshold_multiplier ONCE
    # here in __init__ (see below) and can't use TerminationManager's generic per-call
    # params_per_skill override (that mechanism is stateless-only) -- opts in to building and
    # gathering its OWN per-skill tensors instead. See TerminationTermBase.handles_params_per_
    # skill's own docstring for why this flag exists.
    handles_params_per_skill = True

    def __init__(self, cfg: TerminationTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)

        self.bad_ref_pos_threshold = cfg.params["bad_ref_pos_threshold"]
        self.bad_ref_ori_threshold = cfg.params["bad_ref_ori_threshold"]

        self.bad_motion_body_pos_body_names = cfg.params["bad_motion_body_pos_body_names"]

        # NOTE: body_names_to_track is shared with command_manager
        self.body_names_to_track = cfg.params["body_names_to_track"]
        self.bad_motion_body_pos_threshold = cfg.params["bad_motion_body_pos_threshold"]
        self.bad_motion_body_pos_body_indexes = self._get_index_of_a_in_b(
            self.bad_motion_body_pos_body_names, self.body_names_to_track, self.env.device
        )

        self.bad_object_pos_threshold = cfg.params["bad_object_pos_threshold"]
        self.bad_object_ori_threshold = cfg.params["bad_object_ori_threshold"]

        # Opt-in only (default 0 = no behavior change): suppresses this termination for the first
        # N steps after any reset. A freshly-reset pose (teleported straight into an arbitrary
        # motion-clip frame) can momentarily interpenetrate the ground or itself by a couple of
        # centimeters — a normal consequence of the reference clip's body_pos_w and the actual
        # robot's forward kinematics not perfectly agreeing at extreme poses, not a real fall. The
        # physics engine's corrective impulse for that can itself cause a large single-step
        # tracking-error spike, which this term would otherwise immediately punish as a "bad
        # tracking" failure before the policy has taken a single action.
        self.grace_period_steps = int(cfg.params.get("grace_period_steps", 0))

        # Opt-in only (default False = no behavior change): once True, this termination can only
        # fire while a kick clip is still in "swinging mode" (motion_command.in_kicking_phase --
        # locomotion-approach + strike; renamed from in_swing_phase 2026-07-31, its boundary moved
        # from the whole clip's end to stand_start_idx, the mode-2/mode-3 split) -- fully
        # suppressed for post-kick-standing mode and the entire recovery/hold segment, not just an
        # early window. A
        # genuinely hard, committed kick leaves the robot with real residual momentum the
        # recovery/hold segment's own authored trajectory doesn't anticipate (that segment is a
        # synthetic interpolation back to default pose, not a recording of a real hard-kick
        # recovery) -- so arresting that momentum can read as clip divergence rather than
        # instability. kick_low_height (absolute height, clip-independent) remains the sole
        # termination-level fall backstop for the whole recovery/hold window when this is on --
        # deliberate: see config_values/unified/g1/termination.py's module docstring ("POST-SWING
        # RELAXATION") for the full tradeoff (in particular, orientation/lean is no longer a hard
        # termination post-swing under this setting, only a reward penalty). No-op for any
        # motion_command without has_ball (standalone WBT, or has_ball=False) -- see __call__.
        self.bad_tracking_swing_only = bool(cfg.params.get("bad_tracking_swing_only", False))

        # Opt-in only (default 1.0 = no behavior change): widens (NOT removes) bad_ref_pos/
        # bad_ref_ori/bad_motion_body_pos's thresholds by this factor for envs currently in
        # "swinging mode" (motion_command.in_kicking_phase True; see bad_tracking_swing_only's own
        # comment above for the 2026-07-31 rename/boundary-move) -- unlike bad_tracking_swing_only
        # above, this is a
        # REDUCTION in sensitivity, not full removal. Motivation: reaching an off-nominal ball
        # (ball position is randomized, e.g. +/-0.75m -- see stageC_2skills.yaml) requires the
        # robot's real trajectory to genuinely diverge from a reference clip that has no idea
        # where the ball actually is -- that divergence is legitimate, not instability, and is the
        # whole point of Stage C (a policy that refuses to deviate can't adapt the strike to where
        # the ball is). Live measurement (contact-speed-vs-termination-type correlation, this
        # project's own investigation): of 244 sampled bad_tracking terminations, 169 fired DURING
        # swing vs 75 post-swing -- swing is the LARGER source of tracking-deviation terminations,
        # not the smaller one bad_tracking_swing_only already addresses.
        #
        # Deliberately a WIDENING, not a swing-phase version of bad_tracking_swing_only's full
        # removal: unlike the post-swing recovery/hold segment (a synthetic interpolation with no
        # balance information), the authored swing content still carries real, non-ball-related
        # guidance -- general windup shape, single-support balance, self-collision avoidance --
        # that a full removal would discard along with the ball-chasing benefit. kick_low_height
        # (absolute height, clip-independent) remains fully active throughout swing regardless of
        # this field's value, so a genuine fall is still caught even with a widened threshold.
        # Applies to ALL THREE checks uniformly (including orientation) -- unlike the recovery/hold
        # case, this is a modest widening, not a full removal, so there's no "orientation has no
        # other backstop" argument for excluding it here; kick_low_height backstops genuine
        # instability regardless of which specific sub-check would otherwise have caught it.
        # No-op (returns the plain scalar, not even a tensor) whenever this is 1.0 or the
        # motion_command has no ball -- see _swing_widened_threshold.
        self.swing_threshold_multiplier = float(cfg.params.get("swing_threshold_multiplier", 1.0))

        # 2026-08-15, "simultaneous per-skill task configs": per-skill [n_skills] tensors for the
        # two fields above, built directly from cfg.params_per_skill (NOT TerminationManager's
        # generic mechanism -- see handles_params_per_skill's own docstring). None (the common
        # case) when a field has no genuine per-skill divergence -- _swing_widened_threshold below
        # falls back to the plain scalar attribute unchanged, byte-identical to before this
        # existed. Requires env.skill_id if either is set, checked once here rather than on every
        # __call__.
        per_skill = cfg.params_per_skill or {}
        self._bad_motion_body_pos_threshold_per_skill = (
            torch.tensor(per_skill["bad_motion_body_pos_threshold"], dtype=torch.float32, device=self.env.device)
            if "bad_motion_body_pos_threshold" in per_skill
            else None
        )
        self._swing_threshold_multiplier_per_skill = (
            torch.tensor(per_skill["swing_threshold_multiplier"], dtype=torch.float32, device=self.env.device)
            if "swing_threshold_multiplier" in per_skill
            else None
        )
        self._bad_tracking_swing_only_per_skill = (
            torch.tensor(per_skill["bad_tracking_swing_only"], dtype=torch.bool, device=self.env.device)
            if "bad_tracking_swing_only" in per_skill
            else None
        )
        # 2026-08-15, Tier 3 Group B: pre_kick_grace_steps (below) needs the same per-skill
        # tensor treatment -- see its own comment further down and the termination.py call site's
        # comment on why the scalar `pre_kick_grace_steps` param alone can't represent divergence.
        self._pre_kick_grace_steps_per_skill = (
            torch.tensor(per_skill["pre_kick_grace_steps"], dtype=torch.float32, device=self.env.device)
            if "pre_kick_grace_steps" in per_skill
            else None
        )
        if (
            self._bad_motion_body_pos_threshold_per_skill is not None
            or self._swing_threshold_multiplier_per_skill is not None
            or self._bad_tracking_swing_only_per_skill is not None
            or self._pre_kick_grace_steps_per_skill is not None
        ) and not hasattr(self.env, "skill_id"):
            raise AttributeError(
                "BadTracking has a per-skill params table (bad_motion_body_pos_threshold and/or "
                "swing_threshold_multiplier and/or bad_tracking_swing_only and/or "
                "pre_kick_grace_steps), but env has no "
                "`skill_id` attribute -- per-skill "
                "termination params require a UnifiedManager-family env."
            )

        # Opt-in only (default 0.0 = no behavior change): mirror of grace_period_steps above, but
        # keyed to a mid-episode locomotion->kick entry (env._pre_kick_step, UnifiedManager.
        # _maybe_enter_kick_from_locomotion) instead of episode start -- see
        # managers/termination/terms/locomotion.py's _pre_kick_grace_active for the full
        # rationale. grace_period_steps (episode-start) and this are independent and both apply
        # (ANDed together) when both are set; in practice a mid-episode entry's episode_length_buf
        # is already well past any reasonable grace_period_steps, so they never overlap in
        # practice, but nothing here assumes that.
        self.pre_kick_grace_steps = float(cfg.params.get("pre_kick_grace_steps", 0.0))

    def __call__(self, env: Any, **kwargs) -> torch.Tensor:
        motion_command = self.env.command_manager.get_state("motion_command")
        assert motion_command.motion_cfg.body_names_to_track == self.body_names_to_track, (
            "body_names_to_track in motion_command and termination.params are not the same"
            f"motion_command.motion_cfg.body_names_to_track: {motion_command.motion_cfg.body_names_to_track}"
            f"termination.params['body_names_to_track']: {self.body_names_to_track}"
        )

        # return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        bad_ref_pos = self.bad_ref_pos(motion_command)
        bad_ref_ori = self.bad_ref_ori(motion_command)
        bad_motion_body_pos = self.bad_motion_body_pos(motion_command)
        bad_tracking = bad_ref_pos | bad_ref_ori | bad_motion_body_pos

        if motion_command.motion.has_object:
            bad_object_pos = self.bad_object_pos(motion_command)
            bad_object_ori = self.bad_object_ori(motion_command)
            bad_tracking |= bad_object_pos | bad_object_ori

        if self.grace_period_steps > 0:
            bad_tracking &= self.env.episode_length_buf >= self.grace_period_steps

        if getattr(motion_command, "has_ball", False):
            if self._bad_tracking_swing_only_per_skill is not None:
                # 2026-08-15, "simultaneous per-skill task configs": per-env mask, only envs whose
                # OWN skill has bad_tracking_swing_only=True get the &= in_kicking_phase
                # constraint -- envs on a skill that left it False keep their bad_tracking result
                # untouched (torch.where's false-branch), rather than every env taking the same
                # scalar branch as before this field could diverge.
                swing_only_mask = self._bad_tracking_swing_only_per_skill[self.env.skill_id]
                # in_kicking_phase is the exact same per-skill signal locomotion.py's
                # _kick_recovery_gate already uses for the symmetric (posture-side) gate -- safe to
                # access unguarded once has_ball is True, mirroring that precedent.
                bad_tracking = torch.where(swing_only_mask, bad_tracking & motion_command.in_kicking_phase, bad_tracking)
            elif self.bad_tracking_swing_only:
                bad_tracking &= motion_command.in_kicking_phase

        if self._pre_kick_grace_steps_per_skill is not None:
            # 2026-08-15: per-env gather, NOT the scalar `if self.pre_kick_grace_steps > 0.0`
            # gate below -- that gate can be False (global default 0.0) even when a specific
            # skill's own per-skill value is >0.0, and _pre_kick_grace_active itself already
            # correctly no-ops per-env for whichever envs' gathered value is <=0.0 (see its own
            # tensor-support fix, managers/termination/terms/locomotion.py).
            per_env_grace_steps = self._pre_kick_grace_steps_per_skill[self.env.skill_id]
            bad_tracking &= ~_pre_kick_grace_active(self.env, per_env_grace_steps)
        elif self.pre_kick_grace_steps > 0.0:
            bad_tracking &= ~_pre_kick_grace_active(self.env, self.pre_kick_grace_steps)

        return bad_tracking

    def _per_env(self, scalar_value: float, per_skill_tensor: torch.Tensor | None) -> torch.Tensor | float:
        """Resolve a field to its per-env [num_envs] value if a per-skill table is set for it
        (gathered by env.skill_id), else the plain scalar unchanged -- true no-op in the common
        case, matching every other per-skill mechanism in this "simultaneous per-skill task
        configs" project (see e.g. RewardManager.compute()'s identical gather)."""
        if per_skill_tensor is None:
            return scalar_value
        return per_skill_tensor[self.env.skill_id]

    def _swing_widened_threshold(self, base_threshold: float, motion_command: MotionCommand) -> torch.Tensor | float:
        """Per-env effective threshold for bad_ref_pos/bad_ref_ori/bad_motion_body_pos:
        base_threshold * swing_threshold_multiplier for envs currently in_kicking_phase,
        base_threshold unchanged for envs past that. ``base_threshold`` may itself already be a
        per-env tensor (bad_motion_body_pos_threshold's own per-skill resolution, see
        bad_motion_body_pos below) or a plain scalar (bad_ref_pos_threshold/bad_ref_ori_threshold,
        never per-skill) -- both are handled uniformly here.

        Returns the plain scalar/tensor unchanged -- a genuine no-op, not a same-valued tensor
        rebuild -- whenever the (possibly per-skill) swing_threshold_multiplier has no per-skill
        table AND equals 1.0, or this motion_command has no ball at all (mirrors
        bad_tracking_swing_only's own has_ball guard)."""
        multiplier = self._per_env(self.swing_threshold_multiplier, self._swing_threshold_multiplier_per_skill)
        if not torch.is_tensor(multiplier) and multiplier == 1.0:
            return base_threshold
        if not getattr(motion_command, "has_ball", False):
            return base_threshold
        in_swing = motion_command.in_kicking_phase
        # Broadcast whichever of base_threshold/multiplier is still a plain scalar to [num_envs],
        # so the elementwise multiply+select below works uniformly regardless of which (if either)
        # side carries a per-skill table -- numerically identical to the old torch.full_like-based
        # scalar-only version when both truly are scalars (constant-fill then multiply ==
        # multiply-then-constant-fill).
        base_t = base_threshold if torch.is_tensor(base_threshold) else torch.full_like(in_swing, base_threshold, dtype=torch.float32)
        mult_t = multiplier if torch.is_tensor(multiplier) else torch.full_like(in_swing, multiplier, dtype=torch.float32)
        return torch.where(in_swing, base_t * mult_t, base_t)

    def bad_ref_pos(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if the reference position is too far from the robot's position."""
        threshold = self._swing_widened_threshold(self.bad_ref_pos_threshold, motion_command)
        return torch.norm(motion_command.ref_pos_w - motion_command.robot_ref_pos_w, dim=1) > threshold

    def bad_ref_ori(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if the reference orientation is too far from the robot's orientation."""
        motion_projected_gravity_b = quat_rotate_inverse(
            motion_command.ref_quat_w, gravity_vector(self.env), w_last=True
        )
        robot_projected_gravity_b = quat_rotate_inverse(
            motion_command.robot_ref_quat_w, gravity_vector(self.env), w_last=True
        )
        threshold = self._swing_widened_threshold(self.bad_ref_ori_threshold, motion_command)
        return torch.abs(motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]) > threshold

    def bad_motion_body_pos(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if the motion body position is too far from the robot's body position."""
        body_idx = self.bad_motion_body_pos_body_indexes
        error = torch.norm(
            motion_command.body_pos_relative_w[:, body_idx] - motion_command.robot_body_pos_w[:, body_idx], dim=-1
        )
        # Per-skill base threshold (see this class's own 2026-08-15 comment on
        # _bad_motion_body_pos_threshold_per_skill), resolved BEFORE swing-widening applies on
        # top -- a skill-specific tolerance that swing widening then further relaxes during swing,
        # same composition order the reward-side sibling (kick_penalty_ee_body_pos_divergence's
        # "threshold" param, config_values/unified/g1/reward.py) is kept in sync with.
        base = self._per_env(self.bad_motion_body_pos_threshold, self._bad_motion_body_pos_threshold_per_skill)
        threshold = self._swing_widened_threshold(base, motion_command)
        threshold = threshold.unsqueeze(-1) if torch.is_tensor(threshold) else threshold
        return torch.any(error > threshold, dim=-1)

    def bad_object_pos(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if the object position is too far from the simulator's object position."""
        return (
            torch.norm(motion_command.object_pos_w - motion_command.simulator_object_pos_w, dim=-1)
            > self.bad_object_pos_threshold
        )

    def bad_object_ori(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if the object orientation is too far from the simulator's object orientation."""
        return (
            quat_error_magnitude(motion_command.object_quat_w, motion_command.simulator_object_quat_w)
            > self.bad_object_ori_threshold
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset internal state for specified environments."""

    #########################################################################################################
    ## Internal Helper functions
    #########################################################################################################
    def _get_index_of_a_in_b(self, a_names: List[str], b_names: List[str], device: str = "cpu") -> torch.Tensor:
        indexes = []
        for name in a_names:
            assert name in b_names, f"The specified name ({name}) doesn't exist: {b_names}"
            indexes.append(b_names.index(name))
        return torch.tensor(indexes, dtype=torch.long, device=device)


class BadTrackingZOnly(BadTracking):
    """BadTracking variant using z-axis-only position checks for parity with BM Wo-State-Estimation."""

    def bad_ref_pos(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if the reference z position is too far from the robot's z position."""
        z_err = torch.abs(motion_command.ref_pos_w[:, -1] - motion_command.robot_ref_pos_w[:, -1])
        threshold = self._swing_widened_threshold(self.bad_ref_pos_threshold, motion_command)
        return z_err > threshold

    def bad_motion_body_pos(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if tracked bodies have too much z-axis position error."""
        body_idx = self.bad_motion_body_pos_body_indexes
        error = torch.abs(
            motion_command.body_pos_relative_w[:, body_idx, -1] - motion_command.robot_body_pos_w[:, body_idx, -1]
        )
        # 2026-08-15: must route through _per_env (per-skill base threshold), same as the parent
        # class's own bad_motion_body_pos above -- this override previously read the plain scalar
        # attribute directly, silently ignoring bad_motion_body_pos_threshold's per-skill table
        # whenever this z-only variant is the one actually registered (BadTrackingZOnly is what
        # config_values/unified/g1/termination.py wires in for this project's own exp presets).
        base = self._per_env(self.bad_motion_body_pos_threshold, self._bad_motion_body_pos_threshold_per_skill)
        threshold = self._swing_widened_threshold(base, motion_command)
        threshold = threshold.unsqueeze(-1) if torch.is_tensor(threshold) else threshold
        return torch.any(error > threshold, dim=-1)
