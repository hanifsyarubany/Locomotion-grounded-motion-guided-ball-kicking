"""Unified locomotion + ball-kicking termination preset for the G1 robot.

Merges locomotion's (g1_29dof_termination) `contact` term and WBT's (g1_29dof_wbt_termination)
`bad_tracking` term, each tagged with its task_mode so it can only trigger a reset for envs
currently running that task. `timeout` is identical in both sources (same function) and is left
untagged (task_mode=None) — it already applies unconditionally regardless of mode, matching
current semantics for both parents exactly.

`bad_tracking` additionally sets `grace_period_steps` (unset/0 for stock WBT/ball-kick — no
behavior change there). Originally 5 steps (0.1s at dt=0.02), just long enough to absorb a
teleport-interpenetration transient: teleporting straight into an arbitrary frame of a retargeted
motion clip can momentarily interpenetrate the ground/self by a couple of centimeters even when the
clip's own recorded body_pos_w never goes below ground (verified directly against the raw clip
data) — the mismatch comes from the reference's authoring skeleton vs. this URDF's actual forward
kinematics not perfectly agreeing at extreme poses, and the physics engine's corrective impulse for
that can spike single-step tracking error enough to immediately fail bad_tracking before the policy
has taken a single action.

RAISED 5 -> 100 (2.0s) on 2026-07-16 for a second, larger reason: `bad_tracking` (via
`grace_period_steps &= episode_length_buf >= grace_period_steps`, see
`managers/termination/terms/wbt.py`) is kick mode's ONLY fall-related termination -- there is no
separate height-based check the way locomotion has `low_height` (deliberately excluded above,
single-support/follow-through phases legitimately dip low). Direct instrumented testing on two
checkpoints (`unified-stageB-kick1/model_0145000`, `unified-stageC-randomization-safetyfix-kick3/
model_0341000` -- one pre-, one post-, this session's reward/DR robustness fixes) showed the SAME
result: a real single-support balance failure starts around t=0.4s into the kick (base height
already dropping: 0.765 -> 0.708 -> 0.670 -> 0.625m over 4 steps) and `bad_tracking` trips at
t=0.44s -- squarely inside that onset window, well before the fall (or a possible recovery) plays
out (full collapse, if allowed to continue, takes another ~1.0-1.5s). At 5 steps' grace, the episode
resets immediately once tracking crosses threshold, so the policy never experiences (and gets zero
reward signal from) the actual consequence of this specific failure mode -- it is structurally
invisible to training, which is consistent with more training and reward tuning alone not fixing it
(the safetyfix checkpoint, despite ~200k more Stage-C steps under corrected kick-safety weights,
fell just as fast). Kick mode has no `alive` reward to dilute the fix (`alive` is
task_mode="locomotion"-only, see reward.py), so the kick/tracking reward terms' exp(-error/sigma^2)
shape should crater to near-zero through an extended toppled window instead of the episode cleanly
ending -- this is what should give the policy real pressure against instability for the first time.

VERIFIED INSUFFICIENT ALONE (2026-07-18): `unified-stageC-randomization-contactoffsetfix-kick1`
trained 208k further Stage-C steps entirely under this grace_period=100 config (confirmed in that
run's own frozen holosoma_config.yaml). Its late checkpoint (model_0353000) was evaluated with
terminations disabled entirely and a matched IsaacSim/MuJoCo protocol (8s hold, real ball,
deterministic action) -- 100% of 128 envs still collapsed at a median of 58 ticks (~1.2s), the same
magnitude of failure as an untrained-under-this-fix checkpoint. So the theory this grace period was
based on (tracking-reward decay alone gives enough pressure once the reset is delayed) is now
falsified by an actual, substantial retrain, not just "unverified" -- see memory
stagec-kick-sim2sim-gap-was-a-measurement-artifact for the full matched-protocol evidence. Kept
anyway (still correct that 5 steps was pathologically too short for the original
teleport-interpenetration reason above), but no longer treated as sufficient on its own -- see
`_kick_low_height_term` below, the fallback this file's previous version already anticipated
("if a retrain still doesn't show improvement, the next lever is an explicit kick-mode fall
penalty").
See memory eval-interactive-fixes-and-termination-lesson for the full investigation (frame-by-frame
evidence of the topple, exact step/timing data).

POST-SWING RELAXATION, added 2026-07-29 (`bad_tracking_swing_only`, opt-in via configs/*.yaml,
False/off by default): a SEPARATE relaxation from the two grace periods above -- those guard
episode START; this guards the ENTIRE recovery/hold segment, not just an early window after
swing ends. Motivated by a MuJoCo sim2sim rollout (model_0380000_mujoco_kick_skill1.mp4) showing
a clean, committed strike followed by an apparently stable ~2s recovery, then a LATE collapse
well into the hold segment -- not an impact reaction. Live measurement traced this to
`bad_tracking` itself, not physical instability: harder-contact kicks do NOT predict more
`kick_low_height` falls (if anything fewer -- 6.7% vs 19.6% fall rate, harder vs softer than
median contact speed), but DO predict more post-swing `bad_tracking` terminations (mean
triggering contact speed 2.12 m/s vs 1.12 m/s survived) and a much lower rate of ever completing
the full clip through hold (26.5% vs 42.1%). A bounded grace-period version of this
(`bad_tracking_recovery_grace_steps`) was tried first and abandoned: the completion-fraction
shortfall was measured spread across the WHOLE recovery/hold window, not clustered right after
swing end, so a short grace window under-covers the actual problem -- a full swing-only gate is
both simpler to implement and better matched to the measured failure shape.

The recovery/hold segment is a synthetic interpolation back to default pose, not a recording of
a real hard-kick recovery, so a genuinely hard strike's real residual momentum reads as clip
divergence rather than instability -- and because those episodes get cut short before the robot
ever finishes a full stabilization, the policy is structurally under-practiced at exactly the
scenario the video shows. `kick_low_height` (absolute height, momentum/clip-independent) is
UNTOUCHED by this and remains the SOLE termination-level fall backstop throughout recovery/hold
when this is on. User-directed tradeoff, accepted explicitly rather than defaulted into: this
gate suppresses ALL of `bad_tracking`'s components post-swing, including orientation/lean
(`bad_ref_ori`) -- unlike height, nothing else backstops orientation at termination level, so a
badly-leaning-but-not-yet-below-0.40m robot is only reward-penalized
(`penalty_kick_recovery_stand_orientation`), not terminated, post-swing under this setting. See
MultiSkillConfig.bad_tracking_swing_only's docstring for the full measurement and rationale.

SWING-PHASE THRESHOLD WIDENING, added 2026-07-29 (`bad_tracking_swing_threshold_multiplier`,
opt-in via configs/*.yaml, 1.0/off by default): the companion fix for the OTHER phase --
POST-SWING RELAXATION above only touches recovery/hold; `bad_tracking` fires at full, un-widened
strength throughout swing regardless of that setting. Motivation: reaching an off-nominal ball
(randomized per skill, e.g. randomize_x/y=0.75 in stageC_2skills.yaml) requires the robot's real
trajectory to genuinely diverge from a reference clip that has no idea where the ball actually is
-- legitimate divergence, not instability, and the whole point of Stage C. Live measurement
(contact-speed-vs-termination-type correlation): of 244 sampled `bad_tracking` terminations, 169
fired DURING swing vs 75 post-swing -- swing is the LARGER source of tracking-deviation
terminations, the one POST-SWING RELAXATION does not address.

Deliberately a WIDENING (multiplies bad_ref_pos/bad_ref_ori/bad_motion_body_pos's thresholds),
not a swing-phase version of `bad_tracking_swing_only`'s full removal: the authored swing content
still carries real, non-ball-related guidance (windup shape, single-support balance,
self-collision avoidance) a full removal would discard. Applies to all three checks uniformly,
including orientation -- unlike the recovery/hold case, this is a modest widening with
`kick_low_height` still fully active throughout swing, so there's no "orientation has no other
backstop" argument for excluding it here. See MultiSkillConfig.bad_tracking_swing_threshold_
multiplier's docstring for the full measurement and rationale.

PROGRESSIVE ee_body_pos THRESHOLD, added 2026-08-05 (`bad_motion_body_pos_threshold`, opt-in via
configs/*.yaml, 0.25/off by default -- matching this project's existing hardcoded value exactly).
2026-08-05, ported from RoboNaldo (arXiv:2606.11092)'s `ee_body_pos` termination
(`tracking_env_cfg.py`): a re-read of their actual source (not just the paper) confirmed
`bad_motion_body_pos`'s existing configuration (`config_values/wbt/g1/termination.py`) is ALREADY
byte-similar to their mechanism -- same class of check (`BadTrackingZOnly`, Z-axis-only, matching
their `bad_motion_body_pos_z_only`), same 4 tracked bodies (`left_ankle_roll_link`,
`right_ankle_roll_link`, `left_wrist_yaw_link`, `right_wrist_yaw_link`), same Stage-1 threshold
value (0.25). What was missing was only the ABILITY to progressively widen it across a curriculum
resume the way RoboNaldo does (0.25 -> 0.35 -> 0.5 across their S1/S2a/S2b, uniformly across all 4
bodies) -- this field is that knob, reusing the same "per-skill-yaml-edit-then-resume" curriculum
mechanism already established for `root_tracking_reward_scale`/`motion_tracking_reward_scale`
(ROBONALDO_PORT_SCOPE.md Sec 1a), not an in-run ramp. Also feeds
`penalty_kick_ee_body_pos_divergence`'s own `threshold` param (`config_values/unified/g1/
reward.py`), so the termination and its paired reward-side early-warning penalty stay numerically
synced under one source of truth -- mirroring RoboNaldo's own `task_overrides.py`, where a single
`termination_overrides.ee_body_pos.threshold` yaml entry sets both terms together, by
construction. See MultiSkillConfig.bad_motion_body_pos_threshold's own docstring for the full
account.

POST-KICK TERMINATION HANDOFF, added 2026-08-02 (`kick_recovery_termination_handoff`, opt-in via
configs/*.yaml, False/off by default): a REPLACEMENT for `bad_tracking` during recovery/hold, not
merely a relaxation of it. Motivated by decomposing `bad_tracking` into its three sub-checks
against live checkpoint replay (checkpoint 325k skill1): 92.7% of post-kick recovery/hold resets
(38/41 sampled) are `bad_motion_body_pos` -- individual body positions diverging from the
synthetic, physically-uninformed recovery/hold clip -- not genuine falls (`kick_low_height` fired
only 2/41 times in the same sample). POST-SWING RELAXATION above (`bad_tracking_swing_only`) already
tried simply disabling `bad_tracking` for this window and was MEASURED to make things worse
(per-cycle fall hazard 0.058-0.076 -> 0.100-0.130): with nothing installed in `bad_tracking`'s
place, `kick_low_height`'s absolute 0.40m floor was the only thing left watching recovery/hold, so
a robot visibly losing control just kept running, uncaught, in a degraded state, for longer before
an eventual real fall, instead of being caught by a well-suited check.

This flag installs `kick_recovery_low_height_sustained` (managers/termination/terms/wbt.py) into
that same window -- the SAME height-only check, and the SAME min_height=0.70/consecutive_steps=10
values, locomotion mode's own standing termination (`low_height` above) already uses and trusts,
rather than a check tied to matching a scripted trajectory. Active only after a 50-step (1.0s)
grace ramp once recovery starts, mirroring `_kick_recovery_gate`'s own grace_steps=50.0
(managers/reward/terms/locomotion.py) -- itself aligned to SkillConfig.recovery_duration_s=1.0s,
the clip's own interpolation-back-to-standing duration -- so the check only starts watching once
"are you standing yet" becomes a meaningful question, the same "don't demand a settled stance the
instant recovery starts" lesson `_standing_gate`'s own grace period already encodes.

UPDATE 2026-08-06 -- no longer forces `bad_tracking_swing_only`. Originally this flag ALSO forced
`bad_tracking_swing_only` above True for the same envs (ORed into its resolution below), so the
two could not be enabled independently -- exactly the configuration the MEASURED note right below
describes. Per explicit user direction, that forcing has been REMOVED: this flag now installs
`kick_recovery_low_height_sustained` AND its drift sibling (see "DRIFT SIBLING" below) into
recovery/hold WITHOUT touching `bad_tracking_swing_only` at all, so `bad_tracking` stays fully
active there by default -- directly closing the coverage gap the MEASURED note identifies (nothing
was watching for ~50-60 steps because `bad_tracking` had been suppressed and the height check's
own grace period hadn't engaged yet). To reproduce the ORIGINAL measured configuration (the exact
setup the note below tested) for regression/reference, set BOTH flags explicitly:
`kick_recovery_termination_handoff: true` AND `bad_tracking_swing_only: true`. See
MultiSkillConfig.kick_recovery_termination_handoff's own docstring for the full rationale.

MEASURED 2026-08-02, CONCERNING (describes the ORIGINAL coupled behavior, before the 2026-08-06
decoupling above -- ships at False pending a clean re-test either way). See
MultiSkillConfig.kick_recovery_termination_handoff's own docstring for the full account: an
isolated frozen-checkpoint probe (325k skill1) found genuine `kick_low_height` falls in
post_kick_stabilization jumped 2->22 of 29 total resets, concentrated in the ~50-60 step window
where `kick_recovery_low_height_sustained` cannot have fired yet (grace_steps=50 +
consecutive_steps=10) -- nothing was watching there. A real training run resumed from the same
checkpoint corroborated the DIRECTION at far larger scale (kick_topple_frac roughly 3-4x higher
than its non-handoff parent run at a matched step count) but is CONFOUNDED by several simultaneous
unrelated config changes (kick_goal_success_burst weight 10->300, kick_balance_potential newly
active, new shooting reward terms, different clip phase boundaries) -- so the MAGNITUDE is not yet
cleanly attributable to this flag, only the isolated probe's smaller-scale signal is. NOTE: this
measurement predates BOTH the drift sibling and the 2026-08-06 decoupling -- it is evidence about
a configuration this flag no longer produces by itself, not a measurement of the current default
behavior. The current default (bad_tracking active + height + drift) is UNMEASURED.

DRIFT SIBLING, added 2026-08-06 (`kick_recovery_drift_sustained`, `managers/termination/terms/
wbt.py`), installed alongside `kick_recovery_low_height_sustained` under the same
`kick_recovery_termination_handoff` flag -- both are always on/off together. Adds a
self-referential check the height-only mechanism above cannot express: a simple Euclidean XY
radius (`kick_recovery_drift_deadzone`, 0.15m/15cm default) from the robot's own base position the
instant it entered recovery/hold, sustained for the same 10-step/50-step-grace discipline. Grew
directly out of a live drift-diagnostic investigation (checkpoint 240k, stageC1-1skill run) that
measured ~4-6cm of drift as typical/benign for envs that went on to survive their recovery window
cleanly -- 15cm is deliberately well above that, not a tight trigger. UNVALIDATED against live
telemetry -- inherits kick_recovery_termination_handoff's own MEASURED CONCERNING history above,
so the same "isolated probe before a real training run" discipline applies here too, not just to
the height check alone.
"""

from dataclasses import replace
from pathlib import Path

from holosoma.config_types.multi_skill import load_multi_skill_config, multi_skill_mode_enabled
from holosoma.config_types.reward_tuning import resolve_per_skill_param
from holosoma.config_types.simulator import load_ball_config
from holosoma.config_types.task_config_paths import resolve_task_config_path
from holosoma.config_types.termination import TerminationManagerCfg, TerminationTermCfg
from holosoma.config_values.loco.g1.termination import g1_29dof_termination
from holosoma.config_values.wbt.g1.termination import g1_29dof_wbt_termination

# Same before-tyro-CLI-parsing / independent-per-file-load discipline as command.py's own
# _multi_skill_cfg resolution (see that file's comment) -- each config_values module re-loads the
# yaml itself rather than sharing a cross-file singleton, since load_multi_skill_config/
# load_ball_config are cheap, pure, idempotent parses.
_multi_skill_cfg = load_multi_skill_config() if multi_skill_mode_enabled() else None

# "Simultaneous per-skill task configs" (2026-08-15) -- see config_values/unified/g1/reward.py's
# own _skill_task_config_paths comment for the full design. Independently re-resolved here (not
# imported from reward.py) for the same reason _multi_skill_cfg above is independently re-loaded
# rather than shared: each config_values module owns its own cheap, idempotent parse.
_skill_task_config_paths: list[Path | None] | None = (
    [
        resolve_task_config_path(sc.task_config) if sc.task_config is not None else None
        for sc in _multi_skill_cfg.skills
    ]
    if _multi_skill_cfg is not None
    else None
)

_kick_recovery_termination_handoff = (
    _multi_skill_cfg.kick_recovery_termination_handoff
    if _multi_skill_cfg is not None
    else load_ball_config().kick_recovery_termination_handoff
)
# "Simultaneous per-skill task configs" (2026-08-15, Tier 3 Group B) -- kick_recovery_low_height/
# kick_recovery_drift's REGISTRATION (below) is a per-TERM, not per-skill, structural decision
# (whether the term exists in the manager's term dict at all), so it must register the instant
# ANY skill wants it, not just when the global scalar does -- same "registration must react to
# ANY-skill-active, not just the global" pattern as post_flip_reward_decay_steps in reward.py.
# Once registered, each function's own new `enabled` param (managers/termination/terms/wbt.py)
# suppresses it PER-ENV for skills that don't want it, via the ordinary params_per_skill gather.
_kick_recovery_termination_handoff_per_skill = resolve_per_skill_param(
    _skill_task_config_paths, "kick_recovery_termination_handoff", _kick_recovery_termination_handoff
)
_kick_recovery_termination_handoff_active = _kick_recovery_termination_handoff or (
    _kick_recovery_termination_handoff_per_skill is not None
    and any(_kick_recovery_termination_handoff_per_skill)
)
_bad_tracking_swing_only = (
    _multi_skill_cfg.bad_tracking_swing_only
    if _multi_skill_cfg is not None
    else load_ball_config().bad_tracking_swing_only
)
# "Simultaneous per-skill task configs" (2026-08-15, Tier 3) -- "Mechanism B": BadTracking
# (STATEFUL) consumes this directly via cfg.params_per_skill (see its own 2026-08-15
# _bad_tracking_swing_only_per_skill comment in managers/termination/terms/wbt.py), same as
# bad_motion_body_pos_threshold/swing_threshold_multiplier above it. None (the common case)
# unless skills genuinely diverge.
_bad_tracking_swing_only_per_skill = resolve_per_skill_param(
    _skill_task_config_paths, "bad_tracking_swing_only", _bad_tracking_swing_only
)
# 2026-08-06, user-requested DECOUPLING (see this module's docstring, "DRIFT SIBLING" paragraph,
# and MultiSkillConfig.kick_recovery_termination_handoff's own docstring for the full account):
# kick_recovery_termination_handoff no longer auto-forces bad_tracking_swing_only True. Before
# this change, the two were ALWAYS coupled (enabling the handoff always suppressed bad_tracking
# for recovery/hold, no code path let them vary independently) -- that coupling is what the
# MEASURED 2026-08-02 CONCERNING result describes: a ~50-60 step window where bad_tracking was
# suppressed and kick_recovery_low_height_sustained's own grace period hadn't engaged yet, so
# NOTHING was watching. The user's own read of that finding: keep bad_tracking active alongside
# the handoff's new terms, closing that exact gap, rather than replacing it.
#
# Net effect of this decoupling: kick_recovery_termination_handoff=True alone now installs
# kick_recovery_low_height + kick_recovery_drift into recovery/hold WITHOUT touching
# bad_tracking_swing_only at all -- bad_tracking stays fully active there (at whatever threshold
# bad_motion_body_pos_threshold/bad_tracking_swing_threshold_multiplier already give it), same as
# if the handoff were off, MINUS this project's own documented 92.7%-bad_motion_body_pos-on-the-
# synthetic-clip concern which motivated suppressing it in the first place -- untested territory,
# not a validated fix.
#
# To reproduce the ORIGINAL measured configuration (bad_tracking suppressed + height check only,
# the exact setup the 2026-08-02 probe measured) for regression/reference, set BOTH flags
# explicitly: kick_recovery_termination_handoff: true AND bad_tracking_swing_only: true. The
# standalone bad_tracking_swing_only flag was already independently settable before this change
# (preserved for its own MEASURED OUTCOME note); this change only removes the automatic forcing,
# it does not remove the flag itself or its ability to combine with the handoff.
_bad_tracking_swing_threshold_multiplier = (
    _multi_skill_cfg.bad_tracking_swing_threshold_multiplier
    if _multi_skill_cfg is not None
    else load_ball_config().bad_tracking_swing_threshold_multiplier
)
# See MultiSkillConfig.bad_motion_body_pos_threshold's own docstring: 0.25 unless set in
# configs/*.yaml -- bit-identical by default to the value already hardcoded into
# g1_29dof_wbt_termination's "bad_tracking" registration below.
_bad_motion_body_pos_threshold = (
    _multi_skill_cfg.bad_motion_body_pos_threshold
    if _multi_skill_cfg is not None
    else load_ball_config().bad_motion_body_pos_threshold
)

# "Simultaneous per-skill task configs" (2026-08-15) -- "Mechanism B": BadTracking (below) is
# STATEFUL, so these two per-skill tables are consumed DIRECTLY by that class's own __init__ via
# cfg.params_per_skill (see BadTracking.handles_params_per_skill's own docstring), not via
# TerminationManager's generic per-call mechanism (Mechanism A -- stateless-only). None (the
# common case) unless skills genuinely diverge.
#
# bad_motion_body_pos_threshold is ALSO fed to the reward-side sibling
# kick_penalty_ee_body_pos_divergence's "threshold" param (config_values/unified/g1/reward.py,
# explicitly "Synced with bad_tracking's bad_motion_body_pos threshold") -- that term IS
# stateless, so it uses the ordinary Mechanism-A params_per_skill path there. The two stay in
# sync automatically: both resolve the SAME field name against each module's own independently-
# loaded (but identical) _skill_task_config_paths and the SAME _bad_motion_body_pos_threshold
# base value, not because either file imports the other's table.
_bad_motion_body_pos_threshold_per_skill = resolve_per_skill_param(
    _skill_task_config_paths, "bad_motion_body_pos_threshold", _bad_motion_body_pos_threshold
)
_bad_tracking_swing_threshold_multiplier_per_skill = resolve_per_skill_param(
    _skill_task_config_paths, "bad_tracking_swing_threshold_multiplier", _bad_tracking_swing_threshold_multiplier
)

# See MultiSkillConfig.kick_recovery_drift_deadzone's own docstring: 0.15 (15cm) unless set in
# configs/*.yaml. Only takes effect when kick_recovery_termination_handoff is True (see
# _kick_recovery_drift_term_dict below), same as kick_recovery_low_height_sustained.
_kick_recovery_drift_deadzone = (
    _multi_skill_cfg.kick_recovery_drift_deadzone
    if _multi_skill_cfg is not None
    else load_ball_config().kick_recovery_drift_deadzone
)
# "Simultaneous per-skill task configs" (2026-08-15) -- None (the common case) unless skills
# genuinely diverge on this specific field. kick_recovery_drift_sustained (the term this feeds) is
# a plain function -- state lives on env attributes (counter_attr/anchor_attr), not on a term
# instance -- so it's safe for the generic params_per_skill mechanism, unlike BadTracking below.
_kick_recovery_drift_deadzone_per_skill = resolve_per_skill_param(
    _skill_task_config_paths, "kick_recovery_drift_deadzone", _kick_recovery_drift_deadzone
)

# See MultiSkillConfig.joint_pos_sanity_check_enabled's own docstring (2026-08-10): opt-in,
# task_mode-agnostic termination catching a rare per-env physics-solver numerical explosion
# (NaN/Inf or an absurd-magnitude dof_pos), as opposed to BadTracking's reference-relative check.
_joint_pos_sanity_check_enabled = (
    _multi_skill_cfg.joint_pos_sanity_check_enabled
    if _multi_skill_cfg is not None
    else load_ball_config().joint_pos_sanity_check_enabled
)
_joint_pos_sanity_threshold = (
    _multi_skill_cfg.joint_pos_sanity_threshold
    if _multi_skill_cfg is not None
    else load_ball_config().joint_pos_sanity_threshold
)

# FIX 2 (2026-08-12): see MultiSkillConfig.post_flip_termination_grace_steps's own docstring for
# the full measured rationale. 0.0 (default) = exact no-op -- contact/low_height below stay
# bit-identical to the un-graced originals (see contact_forces_exceeded_post_flip_graced's own
# docstring for why that's guaranteed, not just typical).
_post_flip_termination_grace_steps = (
    _multi_skill_cfg.post_flip_termination_grace_steps
    if _multi_skill_cfg is not None
    else load_ball_config().post_flip_termination_grace_steps
)

# 2026-08-13: mirror of post_flip_termination_grace_steps above, opposite boundary -- see
# managers/termination/terms/locomotion.py's _pre_kick_grace_active and
# MultiSkillConfig.pre_kick_termination_grace_steps's own docstrings. 0.0 (default) = exact no-op.
# MultiSkillConfig-only, no legacy BallConfig counterpart -- same scope choice as command.py's own
# resolution of this field (config_values/unified/g1/command.py): the mechanism this backs is
# inherently scoped to a fixed per-env skill assignment that only exists in N-skill mode.
_pre_kick_termination_grace_steps = (
    _multi_skill_cfg.pre_kick_termination_grace_steps if _multi_skill_cfg is not None else 0.0
)

# "Simultaneous per-skill task configs" (2026-08-15, Tier 3 Group B) -- None (the common case)
# unless skills genuinely diverge. Both feed STATELESS functions (base_height_below_threshold_
# sustained_post_flip_graced/_pre_kick_graced, contact_forces_exceeded_post_flip_graced), so this
# is the ordinary Mechanism-A params_per_skill path -- see each function's own 2026-08-15 comment
# in managers/termination/terms/locomotion.py for the tensor-aware fast-path handling this relies
# on.
_post_flip_termination_grace_steps_per_skill = resolve_per_skill_param(
    _skill_task_config_paths, "post_flip_termination_grace_steps", _post_flip_termination_grace_steps
)
_pre_kick_termination_grace_steps_per_skill = resolve_per_skill_param(
    _skill_task_config_paths, "pre_kick_termination_grace_steps", _pre_kick_termination_grace_steps
)

_bad_tracking_with_grace_period = replace(
    g1_29dof_wbt_termination.terms["bad_tracking"],
    task_mode="kick",
    params={
        **g1_29dof_wbt_termination.terms["bad_tracking"].params,
        "grace_period_steps": 100,
        # See this module's docstring ("POST-SWING RELAXATION") and
        # MultiSkillConfig.bad_tracking_swing_only's own docstring for the full measured
        # rationale and tradeoffs. False unless set in configs/*.yaml -- bit-identical by default.
        "bad_tracking_swing_only": _bad_tracking_swing_only,
        # See this module's docstring ("SWING-PHASE THRESHOLD WIDENING") and
        # MultiSkillConfig.bad_tracking_swing_threshold_multiplier's own docstring for the full
        # measured rationale. 1.0 unless set in configs/*.yaml -- bit-identical by default.
        "swing_threshold_multiplier": _bad_tracking_swing_threshold_multiplier,
        # See MultiSkillConfig.bad_motion_body_pos_threshold's own docstring ("PROGRESSIVE
        # ee_body_pos THRESHOLD", ported from RoboNaldo). 0.25 unless set in configs/*.yaml --
        # bit-identical by default to the value this key already held below.
        "bad_motion_body_pos_threshold": _bad_motion_body_pos_threshold,
        # 2026-08-13: mirror of grace_period_steps (episode-start), keyed to a mid-episode
        # locomotion->kick entry instead -- see BadTracking.__init__'s own comment and
        # _pre_kick_grace_active's docstring. 0.0 unless set in configs/*.yaml -- bit-identical by
        # default.
        "pre_kick_grace_steps": _pre_kick_termination_grace_steps,
    },
    params_per_skill=(
        {
            **(
                {"bad_motion_body_pos_threshold": _bad_motion_body_pos_threshold_per_skill}
                if _bad_motion_body_pos_threshold_per_skill is not None
                else {}
            ),
            **(
                {"swing_threshold_multiplier": _bad_tracking_swing_threshold_multiplier_per_skill}
                if _bad_tracking_swing_threshold_multiplier_per_skill is not None
                else {}
            ),
            **(
                {"bad_tracking_swing_only": _bad_tracking_swing_only_per_skill}
                if _bad_tracking_swing_only_per_skill is not None
                else {}
            ),
            # 2026-08-15, Tier 3 Group B: BadTracking's OWN pre_kick_grace_steps gate (below,
            # separate from _kick_low_height_term's identically-named param) must also see
            # per-skill divergence -- otherwise a skill that sets pre_kick_termination_grace_steps
            # while the GLOBAL scalar stays 0.0 would silently get NO grace suppression at all
            # here (the scalar `if self.pre_kick_grace_steps > 0.0` gate in BadTracking.__call__
            # would never even fire), even though _kick_low_height_term correctly applies it.
            **(
                {"pre_kick_grace_steps": _pre_kick_termination_grace_steps_per_skill}
                if _pre_kick_termination_grace_steps_per_skill is not None
                else {}
            ),
        }
        or None
    ),
)

# Standing-crouch floor (locomotion mode only). The v3 Stage-B retrain proved reward-side
# penalties alone can't beat this: `alive` (+10/step) structurally rewards whatever posture
# survives, and a deep crouch (settled 0.678m vs the 0.78m standing target) survives fine, so any
# affordable per-step height tax just becomes a cost of doing business. Sustained low height now
# ENDS the episode -- forfeiting all future alive reward -- which no tax-paying crouch can be
# worth. Thresholds: 0.70 is decisively below anything legitimate (walking rides at ~0.78,
# default-pose standing grounds at 0.7565) but above the observed 0.678 crouch, so the current
# failure mode terminates while normal operation never does; 10 consecutive steps (0.2s at
# dt=0.02) forgives brief transient dips (e.g. hard deceleration to a stop). Kick mode is
# excluded via task_mode -- its single-support/follow-through phases legitimately dip low and
# already have their own `bad_tracking` termination.
_low_height_term = TerminationTermCfg(
    # FIX 2 (2026-08-12): _post_flip_graced variant, not the bare function -- see
    # post_flip_grace_steps's own docstring above. Bit-identical to the original at the field's
    # 0.0 default.
    func="holosoma.managers.termination.terms.locomotion:base_height_below_threshold_sustained_post_flip_graced",
    params={
        "min_height": 0.70,
        "consecutive_steps": 10,
        "post_flip_grace_steps": _post_flip_termination_grace_steps,
    },
    params_per_skill=(
        {"post_flip_grace_steps": _post_flip_termination_grace_steps_per_skill}
        if _post_flip_termination_grace_steps_per_skill is not None
        else None
    ),
    task_mode="locomotion",
)

# Kick-mode fall termination, added 2026-07-18 after grace_period_steps=100 alone was verified
# (via an actual 208k-step retrain, not just theory -- see the module docstring's "VERIFIED
# INSUFFICIENT ALONE" note) to NOT give the policy real pressure against single-support falls.
# Reuses the exact same function as locomotion's low_height, just with a much lower threshold:
# kick mode's single-support/follow-through phases legitimately dip low (that's WHY low_height
# above is task_mode="locomotion"-only), so this can't use 0.70m. 0.40m is not a guess -- it's
# the same FALL_Z threshold used throughout the whole matched-protocol IsaacSim/MuJoCo
# investigation (mujoco_kick_survival_scan.py, eval_kick_matched_protocol.py) to mean
# "unambiguously fallen, not a legitimate movement": it sits well below the entire collapse-onset
# trajectory documented above (0.765->0.625m over the FIRST 4 steps of a genuine fall) and well
# below any observed legitimate single-support dip. consecutive_steps=5 (0.1s) is deliberately
# much shorter than locomotion's 10 -- once actually below 0.40m the robot is not "about to
# recover" in any single-support kick observed so far (real collapses in the matched-protocol
# traces went from ~0.75m to <0.40m in ~20-30 ticks and never recovered), so there is little
# value in a long buffer and real cost (extra dead-state rollout time, degenerate contact-force
# transitions polluting the replay buffer) in making it longer. In practice this should almost
# always fire before bad_tracking's 100-step grace period would even engage, making THIS the
# operative fall-detection mechanism while bad_tracking's grace period continues to serve its
# original, narrower purpose (absorbing teleport-interpenetration transients at episode start).
_kick_low_height_term = TerminationTermCfg(
    # 2026-08-13: _pre_kick_graced variant, not the bare function -- see
    # pre_kick_termination_grace_steps's own docstring above. Bit-identical to the original at the
    # field's 0.0 default.
    func="holosoma.managers.termination.terms.locomotion:base_height_below_threshold_sustained_pre_kick_graced",
    # counter_attr MUST differ from _low_height_term's default ("_low_height_counter") -- both
    # terms run every step regardless of task_mode (masking happens after), so a shared counter
    # attribute would have the two terms silently corrupt each other's state. See
    # base_height_below_threshold_sustained's docstring.
    params={
        "min_height": 0.40,
        "consecutive_steps": 5,
        "counter_attr": "_kick_low_height_counter",
        "pre_kick_grace_steps": _pre_kick_termination_grace_steps,
    },
    params_per_skill=(
        {"pre_kick_grace_steps": _pre_kick_termination_grace_steps_per_skill}
        if _pre_kick_termination_grace_steps_per_skill is not None
        else None
    ),
    task_mode="kick",
)

# POST-KICK TERMINATION HANDOFF, added 2026-08-02 (`kick_recovery_termination_handoff`, opt-in via
# configs/*.yaml, False/off by default). See MultiSkillConfig.kick_recovery_termination_handoff's
# own docstring for the full rationale: a live investigation found 92.7% of post-kick recovery/hold
# resets (38/41 sampled, checkpoint 325k skill1) are `bad_tracking`'s `bad_motion_body_pos`
# sub-check firing on individual body positions diverging from the synthetic, physically-uninformed
# recovery/hold clip -- not genuine falls (`kick_low_height` fired only 2/41 times in the same
# sample). Simply disabling `bad_tracking` there (`bad_tracking_swing_only` alone) was already
# tried and MEASURED to make things worse (per-cycle fall hazard 0.058-0.076 -> 0.100-0.130) --
# with nothing installed in `bad_tracking`'s place, `kick_low_height`'s absolute 0.40m floor was
# the only thing left watching that window, so a robot visibly losing control just kept running,
# uncaught, in a degraded state, for longer before an eventual real fall. This flag instead
# REPLACES `bad_tracking` for that window with `kick_recovery_low_height_sustained` -- the SAME
# height-only check (min_height=0.70, consecutive_steps=10), and the SAME values, locomotion
# mode's own standing termination already uses and trusts -- active only after a 50-step (1.0s)
# grace ramp once recovery starts (mirroring `_kick_recovery_gate`'s own grace_steps=50.0,
# managers/reward/terms/locomotion.py, itself aligned to SkillConfig.recovery_duration_s=1.0s, the
# clip's own interpolation-back-to-standing duration).
_kick_recovery_low_height_term = TerminationTermCfg(
    func="holosoma.managers.termination.terms.wbt:kick_recovery_low_height_sustained",
    params={
        "min_height": 0.70,
        "consecutive_steps": 10,
        "grace_steps": 50.0,
        "counter_attr": "_kick_recovery_low_height_counter",
    },
    params_per_skill=(
        {"enabled": _kick_recovery_termination_handoff_per_skill}
        if _kick_recovery_termination_handoff_per_skill is not None
        else None
    ),
    task_mode="kick",
)
_kick_recovery_low_height_term_dict = (
    {"kick_recovery_low_height": _kick_recovery_low_height_term} if _kick_recovery_termination_handoff_active else {}
)

# 2026-08-06, user-requested sibling to _kick_recovery_low_height_term above: a self-referential
# "did you drift away from where you were already stably standing" check (simple Euclidean XY
# radius from the robot's own pose the instant it entered recovery/hold), installed into the SAME
# window under the SAME kick_recovery_termination_handoff flag -- see
# kick_recovery_drift_sustained's own docstring (managers/termination/terms/wbt.py) for the full
# mechanism and its relationship to RoboNaldo's stabilize_anchor_pos_w latch.
_kick_recovery_drift_term = TerminationTermCfg(
    func="holosoma.managers.termination.terms.wbt:kick_recovery_drift_sustained",
    params={
        "deadzone": _kick_recovery_drift_deadzone,
        "consecutive_steps": 10,
        "grace_steps": 50.0,
        "counter_attr": "_kick_recovery_drift_counter",
        "anchor_attr": "_kick_recovery_drift_anchor_xy",
        "anchor_valid_attr": "_kick_recovery_drift_anchor_valid",
    },
    params_per_skill=(
        {
            **(
                {"deadzone": _kick_recovery_drift_deadzone_per_skill}
                if _kick_recovery_drift_deadzone_per_skill is not None
                else {}
            ),
            **(
                {"enabled": _kick_recovery_termination_handoff_per_skill}
                if _kick_recovery_termination_handoff_per_skill is not None
                else {}
            ),
        }
        or None
    ),
    task_mode="kick",
)
_kick_recovery_drift_term_dict = (
    {"kick_recovery_drift": _kick_recovery_drift_term} if _kick_recovery_termination_handoff_active else {}
)

# NUMERICAL SANITY, added 2026-08-10 (`joint_pos_sanity_check_enabled`, opt-in via configs/*.yaml,
# False/off by default). See MultiSkillConfig.joint_pos_sanity_check_enabled's own docstring for
# the full rationale and the live incident (a single env's dof_pos exploding to 2.36e8 for one
# tick, SAC critic loss going NaN at the same step) this was built from. Deliberately left
# task_mode-unset (None = always active, same as `timeout` above) -- a robot-state integrity
# check, not a reference-fidelity check, so it applies regardless of task_mode.
_joint_pos_sanity_term = TerminationTermCfg(
    func="holosoma.managers.termination.terms.locomotion:joint_pos_sanity_exceeded",
    params={"joint_pos_sanity_threshold": _joint_pos_sanity_threshold},
)
_joint_pos_sanity_term_dict = (
    {"joint_pos_sanity": _joint_pos_sanity_term} if _joint_pos_sanity_check_enabled else {}
)

# FIX 2 (2026-08-12): _post_flip_graced variant of the stock `contact` term, force_threshold/
# contact_indices_attr preserved verbatim from g1_29dof_termination.terms["contact"] -- only func
# and the added post_flip_grace_steps param differ. Bit-identical to the original at the field's
# 0.0 default. See _low_height_term's own comment and post_flip_grace_steps's docstring above.
# 2026-08-18: unified-only force_threshold override. At the inherited baseline of 1.0 N this term
# was measured causing 78.9% of ALL kick-episode terminations while the robot stood UPRIGHT (base
# height mean 0.779 m, 100% above 0.5 m, 0% below 0.3 m -- not one was a fall), starving skill-2
# training of post-kick experience. See MultiSkillConfig.contact_termination_force_threshold's own
# docstring for the full measurement and why a termination is the wrong tool here. 0.0 (default)
# keeps the baseline value, so this dict is byte-identical to before unless a config opts in --
# and the loco-only baseline's own registration is untouched either way.
_contact_force_threshold_override = (
    _multi_skill_cfg.contact_termination_force_threshold if _multi_skill_cfg is not None else 0.0
)

_contact_term = replace(
    g1_29dof_termination.terms["contact"],
    func="holosoma.managers.termination.terms.locomotion:contact_forces_exceeded_post_flip_graced",
    params={
        **g1_29dof_termination.terms["contact"].params,
        "post_flip_grace_steps": _post_flip_termination_grace_steps,
        **(
            {"force_threshold": _contact_force_threshold_override}
            if _contact_force_threshold_override > 0.0
            else {}
        ),
    },
    params_per_skill=(
        {"post_flip_grace_steps": _post_flip_termination_grace_steps_per_skill}
        if _post_flip_termination_grace_steps_per_skill is not None
        else None
    ),
    task_mode="locomotion",
)

g1_29dof_unified_termination = TerminationManagerCfg(
    terms={
        "contact": _contact_term,
        "bad_tracking": _bad_tracking_with_grace_period,
        "timeout": g1_29dof_termination.terms["timeout"],
        "low_height": _low_height_term,
        "kick_low_height": _kick_low_height_term,
        **_kick_recovery_low_height_term_dict,
        **_kick_recovery_drift_term_dict,
        **_joint_pos_sanity_term_dict,
    }
)

__all__ = ["g1_29dof_unified_termination"]
