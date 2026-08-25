"""Unified locomotion + ball-kicking reward preset for the G1 robot.

Merges locomotion's (g1_29dof_loco_fast_sac) and WBT's (g1_29dof_wbt_fast_sac_reward) reward
terms, each tagged with its task_mode so RewardManager zeroes its contribution for envs not
currently running that task. Verified zero key collisions between the two source term sets, so
this is a plain dict union (no renaming needed, unlike observation.py's terms).

Adds five unified-only standing-shaping terms on top of the plain union:
``penalty_stance_asymmetry``, ``penalty_yaw_drift``, ``penalty_stand_height``,
``penalty_stand_orientation``, and ``penalty_stand_feet_width``. None are added to
``g1_29dof_loco_fast_sac`` itself (the standalone locomotion-only baseline experiment) — kept
scoped to unified so the validated baseline's reward function is left untouched; see each term's
own docstring for why it's needed here (each guards a standing degree of freedom the others are
provably blind to — height, left/right asymmetry, heading, pelvis tilt, stance width — found one
at a time as successive retrains escaped through the next unguarded axis).

All five carry ``grace_steps=50.0`` (1.0s at dt=0.02), which fades them in linearly over the first
second after the velocity command goes to zero, instead of snapping them to full strength the
instant it does. This is load-bearing, not a refinement: a hard gate demands a square, symmetric,
nominal-width stance from a robot that is still physically travelling, and measurably caused the
walk→stop falls (v9 Stage-A, 15k steps: 7/10 gait phases fall on an instant stop from 0.8 m/s —
the robot squares its feet up, stagger +0.18m → −0.02m and asymmetry 1.01 → 0.04 within 0.4s,
instead of planting the staggered catch step that arrests momentum, then topples forward).
Arresting momentum REQUIRES a transiently asymmetric, staggered stance; a hard gate was paying the
policy not to use one. 1.0s matches the measured deceleration window (vx 0.65 → ~0.2 m/s).

The window keys on the COMMAND history, never on the robot's own state — see ``_standing_gate`` in
managers/reward/terms/locomotion.py. That distinction is the whole design: an earlier attempt gated
on the robot's ACTUAL speed, and since the policy controls that input it learned to control the
gate, simply never stopping (v10: mean 0.64 m/s, 12.6m of drift in 20s under a zero command,
holding the standing penalties suppressed at ~3%). ``stand_steps`` cannot be manipulated — moving
does not extend the window — so the only way to escape the penalties is to genuinely stand.

Also adds three kick-only safety penalties, ``kick_penalty_excess_contact_force``,
``kick_penalty_hard_landing``, and ``kick_penalty_excess_base_lin_vel`` (see
managers/reward/terms/kick_safety.py) — direct, sim2real-oriented fixes for MuJoCo-only
instabilities found in sim2sim testing, each penalizing the physical quantity involved (contact
force, touchdown speed, base speed) rather than tuning either simulator's solver, so each fix
should generalize instead of being specific to MuJoCo. The first two address a launch instability
found in an early Stage-B checkpoint (model_0071000): the kick was generating foot/ground contact
forces that IsaacGym's solver tolerated but MuJoCo's didn't. The third addresses a later Stage-C
checkpoint learning to throw its own body momentum into the kick to satisfy the shooting reward —
see its registration below and its own docstring in kick_safety.py for the full measurement.

A ``kick_alive`` survival term was added and then REVERTED the same day (2026-07-19) once a
controlled measurement showed both that the "kick always falls" evidence behind it was an eval-harness
artifact and that the term moved the real topple rate by exactly zero. See the block above
``g1_29dof_unified_reward`` below for the full disproof before considering re-adding it.

Every individual kick-mode reward term's WEIGHT (not just each category's aggregate scale -- see
``utils/kick_reward_scales.py``'s 5 per-skill category multipliers), and every SIGMA-bearing
term's kernel width, is overridable via ``configs/kicking_motion_reward_tuning.yaml``, loaded
unconditionally at the very end of this module (``_apply_reward_weight_overrides``,
``_apply_reward_sigma_overrides``) -- see that yaml's own header comment and
``config_types/reward_tuning.py`` for the full design. A training run consumes BOTH that file and
whichever of ``configs/ball*.yaml``/``stageB_and_C.yaml``/``stageC_2skills.yaml`` is selected.
"""

from dataclasses import replace
from pathlib import Path

from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg
from holosoma.config_types.reward_tuning import (
    load_per_skill_reward_weight_overrides,
    load_reward_sigma_overrides,
    load_reward_weight_overrides,
    resolve_per_skill_param,
)
from holosoma.config_types.task_config_paths import resolve_task_config_path
from holosoma.config_values.loco.g1.reward import g1_29dof_loco_fast_sac
from holosoma.config_values.robot import g1_29dof
from holosoma.config_values.wbt.g1.reward import g1_29dof_wbt_fast_sac_reward


def _tagged(terms: dict[str, RewardTermCfg], task_mode: str) -> dict[str, RewardTermCfg]:
    return {name: replace(cfg, task_mode=task_mode) for name, cfg in terms.items()}


# Standing-shaping weights, strengthened (was -3 / -20 / -15). These three terms are the ONLY
# thing shaping a still, symmetric, full-height stance at zero velocity, and at the original
# weights they were far too quiet to win against `alive` (weight 10.0, ~40% of total reward
# magnitude, which indirectly *rewards* a crouch since a lower CoM survives longer): measured
# weighted contributions were stance_asymmetry ~0.19%, stand_height ~0.07% of the reward. All
# three are `error * gate` with the command gate (exp(-cmd_speed^2/0.1)), so they are near-zero
# during walking and vanish once the stance is actually correct -- i.e. raising the weights only
# punishes the *deviated* stance and can't hurt locomotion or a converged upright stance.
#
# Not raised further (e.g. 40x) on purpose: these carry the `penalty_curriculum` tag, so their
# effective weight is (nominal * reward_penalty_scale) with reward_penalty_scale in [0, 1], and the
# curriculum AUTO-REDUCES that scale if mean episode length drops below its threshold. Over-large
# standing penalties that destabilize the transient stance would shorten episodes and get scaled
# back down -- self-defeating. These values are strong enough to bite the crouch/asymmetry while
# staying in the range the robot can satisfy without falling.
_penalty_stance_asymmetry_term = {
    "penalty_stance_asymmetry": RewardTermCfg(
        func="holosoma.managers.reward.terms.locomotion:penalty_stance_asymmetry",
        weight=-20.0,  # was -3.0; joint-angle^2 error ~0.7 when asymmetric -> ~-14 weighted, ~0 when symmetric
        params={"command_gate_sigma": 0.1, "grace_steps": 50.0},
        tags=["penalty_curriculum"],
        task_mode="locomotion",
    ),
}

# yaw_rate_sq is in (rad/s)^2, an order of magnitude smaller than penalty_stance_asymmetry's
# joint-angle-error units -- already had the largest weight for comparable gradient strength (a
# 0.28 rad/s drift rate observed at the worst measured case squares to only ~0.08). It was also the
# least-broken of the three; a moderate bump keeps it decisively ahead of a slow heading creep.
_penalty_yaw_drift_term = {
    "penalty_yaw_drift": RewardTermCfg(
        func="holosoma.managers.reward.terms.locomotion:penalty_yaw_drift",
        weight=-50.0,  # was -20.0
        params={"command_gate_sigma": 0.1, "grace_steps": 50.0},
        tags=["penalty_curriculum"],
        task_mode="locomotion",
    ),
}

# The squared-error version of this term FAILED at weight -80: the v3 Stage-B retrain (238k steps,
# penalty curriculum confirmed at full scale=1.0 throughout) still settled into a 12.2cm standing
# crouch (base 0.678m, hip_pitch -0.95/knee 1.14 vs defaults -0.312/0.669, measured in MuJoCo
# sim2sim under an exact zero command). Numbers: 0.122^2 * 80 = 1.2/step vs alive's +10/step --
# and squared error's gradient vanishes as the crouch shrinks, so even partial progress stalls.
# The term is now |error| (L1, see its docstring): at this weight a 12cm crouch costs 4.8/step and
# a 5cm crouch still costs 2.0/step -- meaningful against alive at every realistic crouch depth,
# with a non-vanishing gradient all the way to the target. target_height=0.78 replaces the
# init_state fallback (0.8): 0.8 is physically unattainable (~0.793 with legs fully straight)
# and pulls toward straight-leg standing, while 0.78 matches the measured walking-height
# equilibrium and sits comfortably between the default-pose height (0.7565) and straight legs.
# Paired with the base_height termination floor in termination.py, which removes the
# alive-farming incentive that a per-step tax alone can't beat.
#
# deadzone=0.015 (1.5cm) added after the v8 Stage-A retrain (kick_probability=0, already
# running the recalibrated penalty_curriculum from curriculum.py): telemetry showed a
# repeating limit cycle -- average_episode_length climbing toward the 750 level_up_threshold,
# curriculum ramping penalty_scale to 1.0, episode length collapsing within a few hundred
# steps, curriculum easing back down, recovery, repeat (7 cycles over 72k steps, no net
# convergence -- see e.g. step 53600 scale=1.0 epl=849.7 -> step 54800 scale=0.611 epl=465.8).
# Root cause: at zero deadzone the L1 profile taxes literally zero height error, fighting the
# small continuous height bob real balance requires. 1.5cm leaves the original crouch failure
# (12cm, 5cm) essentially as decisive as before (12cm -> 4.2/step, 5cm -> 1.4/step) while
# giving the policy room to actually satisfy the term instead of chasing an unreachable exact
# target.
# target_height nudged 0.78 -> 0.76 (2cm total) after the deadzone fix above: 0.78 sat close to
# the straight-leg ceiling (0.793) and asked the policy to stand almost fully extended to avoid
# penalty. 0.76 (combined with deadzone=0.015 below) makes the free/unpenalized band
# [0.745, 0.775] -- and the natural default-pose grounding height (0.7565, knee at its default
# 0.669) now sits CENTERED in that band (1.15cm from the low edge, 1.85cm from the high edge),
# instead of pinned almost exactly to the band's edge as it was at target=0.77 (band
# [0.755, 0.785], natural height only 1.5mm above the low edge). Centering it gives the policy
# genuine slack in both directions around its natural relaxed stance rather than a one-sided
# tolerance. Still 6cm clear of the 0.70 termination floor.
#
# 2026-08-14: target_height/deadzone below are NOT the values actually used at runtime -- same
# forward-reference pattern _kick_recovery_standing_terms' own "Patch the 4 deadzone values"
# comment describes (search that string): patched further down once
# _multi_skill_cfg_for_contact_penalty exists to resolve MultiSkillConfig.base_robot_target_
# height/base_robot_deadzone from (HOLOSOMA_SKILLS_CONFIG's own `base_robot:` block). Edit that
# yaml, not these literals, which are dead weight kept only because RewardTermCfg's params dict
# needs SOME value at construction time.
_penalty_stand_height_term = {
    "penalty_stand_height": RewardTermCfg(
        func="holosoma.managers.reward.terms.locomotion:penalty_stand_height",
        weight=-40.0,  # L1 scale: 12cm -> -4.8/step, 5cm -> -2.0/step (was -80.0 on squared error)
        params={"command_gate_sigma": 0.1, "target_height": 0.76, "deadzone": 0.015, "grace_steps": 50.0},
        tags=["penalty_curriculum"],
        task_mode="locomotion",
    ),
}

# Standing pelvis-tilt penalty (L1 |g_xy| = sin(tilt), command-gated) -- added after the v4
# Stage-B checkpoint (crouch fixed: correct height, symmetric legs, straight waist) still stood
# with a constant ~13.6deg forward pelvis lean, upper body rigidly following. See the term's
# docstring for the full blind-spot analysis (hip_pitch ~free in `pose`, and the global squared
# penalty_orientation gives only 0.55/step at that lean vs alive's +10/step). Weight scale:
# |g_xy| at 13.6deg = 0.235 -> -20 gives 4.7/step (decisive vs alive); a 3deg residual still
# costs 1.0/step. Gated to standing only -- walking's natural forward pitch stays governed by
# the (untouched, always-on) penalty_orientation.
#
# deadzone=0.025 (|g_xy| ~= sin(1.4deg)) added after the v8 Stage-A retrain for the same
# reason and against the same telemetry as penalty_stand_height's deadzone above (see that
# term's comment for the measured limit-cycle) -- zero deadzone demands a literally vertical
# pelvis, fighting the small continuous postural sway real balance requires. 1.4deg of free
# tilt leaves the original 13.6deg lean failure fully penalized (4.2/step) while a 3deg
# residual still costs 0.5/step, comfortably visible.
_penalty_stand_orientation_term = {
    "penalty_stand_orientation": RewardTermCfg(
        func="holosoma.managers.reward.terms.locomotion:penalty_stand_orientation",
        weight=-20.0,
        params={"command_gate_sigma": 0.1, "deadzone": 0.025, "grace_steps": 50.0},
        tags=["penalty_curriculum"],
        task_mode="locomotion",
    ),
}

# Standing stance-width penalty (L1 beyond a small deadzone, command-gated) -- added after the
# v5 Stage-B checkpoint (height, symmetry, waist AND pelvis lean all fixed) failed a >20s
# standing hold in a new way: feet slide outward mirror-symmetrically at ~2.3 cm/s until a 1.04m
# splits and a fall at ~24s, just past the 20s episode horizon (so training never saw the fall).
# A symmetric spread is invisible to stance_asymmetry (anti-symmetric measure), keeps the pelvis
# level (stand_orientation ~0), and costs almost no height at first (quadratic insensitivity), so
# this guards the one axis the rest of the standing family can't see: the actual foot-to-foot
# lateral separation. See the term's docstring for measured numbers and weight arithmetic.
#
# deadzone tightened 0.06 -> 0.03 after the unified DR broadening (mass/CoM/RFI, 2026-07-15)
# produced the OPPOSITE drift -- a visibly knock-kneed, too-NARROW stance. Measured directly in
# MuJoCo (broadDR checkpoints 189k-205k, 8s zero-vel hold): width drifted to 0.157-0.20m, but at
# 0.06 the excess-beyond-deadzone was only 0.005-0.023m -> a max -0.46/step at weight -20,
# negligible against alive's +10/step, which is why the term wasn't holding the line. 0.03 lets
# real drift of this magnitude actually cost something (width=0.157 now costs 20*(0.083-0.03)=
# 1.06/step, ~2.3x more) while still tolerating normal double-stance sway. Paired with the new
# penalty_stand_knee_width below (same root cause, one body segment up the leg).
_penalty_stand_feet_width_term = {
    "penalty_stand_feet_width": RewardTermCfg(
        func="holosoma.managers.reward.terms.locomotion:penalty_stand_feet_width",
        weight=-20.0,  # drift onset (width 0.37) -> 1.4/step; mid-drift (0.81) -> ~10/step
        params={"command_gate_sigma": 0.1, "nominal_width": 0.24, "deadzone": 0.03, "grace_steps": 50.0},
        tags=["penalty_curriculum"],
        task_mode="locomotion",
    ),
}

# Standing KNEE-width penalty -- same construction as penalty_stand_feet_width, one body segment
# up the leg. See the term's own docstring in locomotion.py for the full measurement (broadDR
# checkpoints show ankle AND knee narrowing together, not knee-specific decoupling -- this is a
# second independent guard on the same failure, not a response to a different one).
_penalty_stand_knee_width_term = {
    "penalty_stand_knee_width": RewardTermCfg(
        func="holosoma.managers.reward.terms.locomotion:penalty_stand_knee_width",
        weight=-20.0,
        params={"command_gate_sigma": 0.1, "nominal_width": 0.24, "deadzone": 0.03, "grace_steps": 50.0},
        tags=["penalty_curriculum"],
        task_mode="locomotion",
    ),
}

# Kick-mode counterparts of the six standing-shaping terms above, ported into the post-kick
# recovery/hold tail (see managers/reward/terms/locomotion.py's penalty_kick_recovery_* and
# _kick_recovery_gate docstrings for the full rationale). The locomotion versions above are
# tagged task_mode="locomotion" and gated on the velocity command, so a kick episode's
# recovery/hold phase -- which is also trying to settle into a stable standing pose after the
# swing -- currently gets NONE of this already-validated toolkit; posture there is governed
# only by the motion-tracking reward. These six reuse the identical error quantities (same
# weights/deadzones/targets as their locomotion siblings -- the physical target, standing on
# this same robot, doesn't change with task_mode, so there's no principled reason to retune
# from scratch) under a kick-phase gate (_kick_recovery_gate) instead of the velocity-command
# one: zero during the authored swing (never fights legitimate single-support lean/asymmetry/
# height change), ramping to full strength over the first 50 steps (1.0s) once the clip enters
# its recovery-to-default-pose + hold tail -- mirroring the locomotion family's own
# grace_steps=50.0 rationale exactly.
#
# Per-skill by construction (see _kick_recovery_gate's docstring): the swing/recovery boundary
# is read per-env from MotionCommand's own per-motion tables, so this applies correctly for
# every skill in stageB_and_C.yaml, including any added later, with no extra wiring.
#
# Deliberately NOT tagged penalty_curriculum, matching kick_safety.py's three penalty terms
# (also kick-mode, also stability-not-shooting-skill) rather than the locomotion standing
# family above: that curriculum's auto-relax-on-short-episodes mechanism is calibrated against
# LOCOMOTION's own episode-length signal, and coupling a kick-mode penalty's strength to it
# would be an unverified, unrelated cross-task-mode dependency this change doesn't need to
# introduce.
#
# 2026-07-24: RETUNED using real kick-mode telemetry from a live run (wandb run iuqwyfx7,
# RawEpisode/raw_rew_penalty_kick_recovery_* + the tracking terms' own raw values), per the
# instruction above to retune once this became available. Measured: the locomotion-borrowed
# weights below were NOT a gentle nudge in practice -- weighted kick-recovery penalty grew from
# 0 early in this run (recovery/hold was barely ever reached) to ~39% of the ENTIRE tracking
# reward's weighted magnitude by step ~133k (tracking ~42, kick-recovery ~-16.4), and
# yaw_drift ALONE was ~73.5% of that total (~-12 of the ~-16.4) -- wildly disproportionate to
# the other four terms (~-0.6 to -2.9 each). Locomotion's own -50 was tuned against ITS OWN
# reward budget (alive=+10/step as the dominant competitor), not against WBT's much larger
# motion-tracking complex -- reused verbatim, it ended up structurally too strong here, not
# just "unmeasured" as the original comment above hedged.
#
# Fix: cut yaw_drift specifically (5x, -50 -> -10) -- it was the outlier, not the whole family.
# The other four are left at their original values; recheck the same telemetry after this lands
# and retune further if the total is still meaningfully off the ~10-15%-of-tracking target this
# aims for (~-6.4 estimated, vs tracking's ~42 -> ~15%).
#
# 2026-08-06: the 4 "deadzone" literals below (stand_height/orientation/feet_width/knee_width) are
# NOT the values actually used at runtime -- they're patched further down in this file (search
# "Patch the 4 deadzone values"), once _multi_skill_cfg_for_contact_penalty exists to resolve
# yaml-configurable overrides from (this dict is defined before that's available). Edit the yaml-
# configurable defaults in config_types/multi_skill.py/simulator.py, not these literals, which are
# now dead weight kept only because RewardTermCfg's params dict needs SOME value at construction.
_kick_recovery_standing_terms = {
    "penalty_kick_recovery_stance_asymmetry": RewardTermCfg(
        func="holosoma.managers.reward.terms.locomotion:penalty_kick_recovery_stance_asymmetry",
        weight=-20.0,
        params={"grace_steps": 50.0},
        task_mode="kick",
    ),
    "penalty_kick_recovery_yaw_drift": RewardTermCfg(
        func="holosoma.managers.reward.terms.locomotion:penalty_kick_recovery_yaw_drift",
        weight=-10.0,  # was -50.0 -- see the retune note above this dict
        params={"grace_steps": 50.0},
        task_mode="kick",
    ),
    "penalty_kick_recovery_stand_height": RewardTermCfg(
        func="holosoma.managers.reward.terms.locomotion:penalty_kick_recovery_stand_height",
        weight=-40.0,
        params={"target_height": 0.76, "deadzone": 0.015, "grace_steps": 50.0},
        task_mode="kick",
    ),
    "penalty_kick_recovery_stand_orientation": RewardTermCfg(
        func="holosoma.managers.reward.terms.locomotion:penalty_kick_recovery_stand_orientation",
        weight=-20.0,
        params={"deadzone": 0.025, "grace_steps": 50.0},
        task_mode="kick",
    ),
    "penalty_kick_recovery_stand_feet_width": RewardTermCfg(
        func="holosoma.managers.reward.terms.locomotion:penalty_kick_recovery_stand_feet_width",
        weight=-20.0,
        params={"nominal_width": 0.24, "deadzone": 0.03, "grace_steps": 50.0},
        task_mode="kick",
    ),
    "penalty_kick_recovery_stand_knee_width": RewardTermCfg(
        func="holosoma.managers.reward.terms.locomotion:penalty_kick_recovery_stand_knee_width",
        weight=-20.0,
        params={"nominal_width": 0.24, "deadzone": 0.03, "grace_steps": 50.0},
        task_mode="kick",
    ),
}

# RoboNaldo-Stage-2-style shooting rewards (arXiv:2606.11092 Table B.1, adapted — see
# managers/reward/terms/shooting.py's module docstring for the two deliberate adaptations).
# Weights follow the paper's Table B.1 ratios scaled by w_g, the stage-wise task weight, read
# from configs/ball.yaml's shooting_reward_scale (0.8 there = the paper's Stage-2 value: error
# 5->4, predicted 10->8, burst 300->240, contact 2->1.6, orientation 1->0.8, velocity 0.5->0.4).
# Set shooting_reward_scale to 0.0 for the pure-motion-tracking bootstrap stage — RewardManager
# skips weight-0 terms entirely, exactly the paper's Stage-1 w_g = 0. All terms are tagged
# task_mode="kick" so locomotion-mode envs are untouched, and all read the kick foot / target /
# success radius from configs/ball.yaml (yaml-tunable, no code changes). They sit ON TOP of the
# existing kick motion-tracking terms (w_motion stays 1.0, also matching RoboNaldo Stage 2) —
# the tracking prior keeps the swing stable while these shape it for contact and shot placement.
from holosoma.config_types.algo import FastSACConfig as _FastSACConfig
from holosoma.config_types.multi_skill import load_multi_skill_config, multi_skill_mode_enabled
from holosoma.config_types.simulator import load_ball_config as _load_ball_config

# w_g is NO LONGER baked into these weights at config-import time (previously `weight = k * _w_g`,
# with _w_g a single process-wide float from configs/ball.yaml). It can't be: in N-skill mode, which
# skill (and therefore which target shooting_reward_scale) a given env is running is only known at
# ENV-RUNTIME, via motion_ids -- never at config-import time, before any env exists. So `weight`
# below is now just each term's own relative scale k (RoboNaldo's Table B.1 ratios), and the actual
# w_g multiplication happens live, per env, per step, inside each term function via
# holosoma.utils.shooting_curriculum.current_w_g(env) -- same pattern kick_safety.py's three
# penalty terms already used (they were never baked at config-time to begin with). This is a
# uniform change for BOTH modes (not just N-skill): legacy single-skill runs get the identical net
# result (current_w_g returns a same-valued-for-every-env tensor there), but as a side effect these
# 6 terms are no longer skip-if-weight-0'd by RewardManager during a pure Stage-B run (weight is now
# always nonzero) -- they're computed and logged every step, just correctly evaluating to ~0 for
# any env whose current target w_g is 0. Deliberate tradeoff for correctness/uniformity; noted, not
# a bug.
_shooting_terms = {
    "kick_ball_proximity": RewardTermCfg(
        func="holosoma.managers.reward.terms.shooting:ball_proximity",
        params={"sigma": 0.35, "contact_offset_m": 0.09},
        weight=2.0,
        task_mode="kick",
    ),
    "kick_contact_orientation": RewardTermCfg(
        func="holosoma.managers.reward.terms.shooting:contact_orientation",
        params={"sigma": 0.35, "contact_offset_m": 0.09},
        weight=1.0,
        task_mode="kick",
    ),
    # 2026-08-07, user-requested -- not a RoboNaldo term. Rewards toe-down (plantarflexed) ankle
    # pitch near contact over toe-up (dorsiflexed), which the user observed producing weak,
    # sole-first ball contacts. See shooting.py:foot_strike_pitch's own docstring for the full
    # geometric rationale (roll-invariant by construction, no cached reference pose needed).
    # Ships at weight=0.0 (verified no-op registration first, per this project's own convention) --
    # override in configs/*.yaml once ready to enable.
    "kick_foot_strike_pitch": RewardTermCfg(
        func="holosoma.managers.reward.terms.shooting:foot_strike_pitch",
        params={"sigma": 0.35, "contact_offset_m": 0.09},
        weight=0.0,
        task_mode="kick",
    ),
    "kick_ball_velocity": RewardTermCfg(
        # v_ref/speed-source/phase-gate are all OPT-IN retunes now, resolved from the task-config
        # yaml and patched into these params further down this file (search
        # "_kick_ball_velocity_v_ref"). 5.0 here is the ORIGINAL pre-2026-08-21 value, so an
        # unedited config reproduces the old behavior exactly. See ball_velocity's own docstring.
        func="holosoma.managers.reward.terms.shooting:ball_velocity",
        params={"v_ref": 5.0},
        weight=0.5,
        task_mode="kick",
    ),
    "kick_error_ball_to_target": RewardTermCfg(
        func="holosoma.managers.reward.terms.shooting:error_ball_to_target",
        params={"sigma": 1.0},
        weight=5.0,
        task_mode="kick",
    ),
    "kick_predicted_error_ball_to_target": RewardTermCfg(
        func="holosoma.managers.reward.terms.shooting:predicted_error_ball_to_target",
        params={"sigma": 1.0, "v_min": 0.25},
        weight=10.0,
        task_mode="kick",
    ),
    "kick_goal_success_burst": RewardTermCfg(
        func="holosoma.managers.reward.terms.shooting:goal_success_burst",
        params={"burst_steps": 10},
        weight=300.0,
        task_mode="kick",
    ),
    # 2026-08-01 addition, NOT one of RoboNaldo's original 6 -- dense per-tick credit for a
    # CONFIRMED strike (has_kicked, which on IsaacSim is itself already the geometric-PhysX-
    # contact-sensor signal, not a proxy), gated to _post_locomotion_multiplier like the 3 outcome
    # terms above (accrues through recovery/hold, not just the strike window). See
    # shooting.py:ball_contact_hit's own docstring for the full rationale. weight is a fresh
    # starting point (no Table B.1 ratio to anchor to, since this term doesn't exist in the paper)
    # -- picked in the same order of magnitude as kick_contact_orientation (1.0) rather than the
    # much larger outcome-term weights (5.0-300.0): unlike those, this term pays out on EVERY tick
    # of a multi-hundred-tick post-kick window rather than once or as a decaying/bounded shaping
    # term, so a large weight here would silently dominate the whole shooting budget. Override in
    # configs/kicking_motion_reward_tuning.yaml.
    "kick_ball_contact_hit": RewardTermCfg(
        func="holosoma.managers.reward.terms.shooting:ball_contact_hit",
        weight=1.0,
        task_mode="kick",
    ),
    # 2026-08-01, also not a RoboNaldo term -- the ONLY shooting term active during the
    # locomotion-approach phase (all 7 above are zero there, leaving a ball-randomized approach
    # steered purely by ball-blind clip tracking). Rewards reaching the swing's authored stance
    # relative to the ACTUAL ball, not proximity to it. See shooting.py:ball_approach_stance.
    #
    # weight mirrors kick_ball_proximity's 2.0 deliberately: that term is this one's direct
    # analogue (RoboNaldo's own dense approach shaping, "the term that gives gradient on every
    # episode, contact or miss"), which the 2026-07-31 strike gating narrowed from the whole
    # episode down to the 34-tick swing -- this restores dense approach shaping to the phase that
    # lost it, so it starts at the same magnitude. NOTE this term fires on ~every locomotion tick
    # (65.6k env-ticks in the reference rollout) rather than on the ~4% of cycles the has_kicked-
    # gated terms reach, so its total contribution is large relative to the other shooting terms
    # even at an identical weight -- that asymmetry is exposure, not weight, and is the main thing
    # to watch when tuning. sigma (0.9m, see the term's docstring for why NOT 0.35) is the other
    # knob and is sigma-overridable from the same yaml.
    "kick_ball_approach_stance": RewardTermCfg(
        func="holosoma.managers.reward.terms.shooting:ball_approach_stance",
        params={"sigma": 0.9},
        weight=2.0,
        task_mode="kick",
    ),
}

# Contact-safety penalties (sim2sim/sim2real robustness, not shooting skill) -- see
# managers/reward/terms/kick_safety.py's module docstring for the MuJoCo launch this addresses.
# Deliberately NOT scaled by _w_g: this is basic kick stability ("post kick stabilization"), so it
# stays active through Stage B (shooting_reward_scale=0) as well as Stage C.
#
# kick_penalty_excess_base_lin_vel added after the prepend-fixed Stage C run (228k steps, resumed
# from a Stage-B checkpoint that itself kicks and recovers cleanly) still fell in MuJoCo -- not on
# trigger (the prepend fix confirmed correct via direct ONNX inspection), but ~1-1.5s into the
# swing, base horizontal speed climbing to ~2 m/s as it topples. Direct checkpoint comparison
# (121k pre-Stage-C: survives, peak speed 0.82 m/s vs 148k/175k/202k/228k post-Stage-C: all fall,
# progressively earlier and faster as training continues) shows Stage C teaching the policy to
# throw its own body momentum into the kick, undetected because BadTrackingZOnly's termination
# (config_values/unified/g1/termination.py) is deliberately Z-axis-only -- correct for real-robot
# state-estimation parity, but blind to exactly this horizontal-launch failure mode. See the
# term's own docstring in kick_safety.py for the full measurement and threshold rationale.
#
# 2026-07-15: all three raised from the original -3.0. Root cause of the recurrence (run
# unified-stageC-randomization, resumed from stageB-ballobs-gated-v10/model_0277000): these three
# are NOT scaled by shooting_reward_scale (by design, see above), but kick_goal_success_burst IS
# (shooting.py, weight 300*shooting_reward_scale = 240 at this run's shooting_reward_scale=0.8) --
# so as shooting_reward_scale is ramped up across the Stage C curriculum, the reward for hitting
# the ball harder/faster grows while the cost of the resulting excess momentum/impact stays fixed
# at whatever was calibrated for an earlier, lower-w_g Stage C run. Measured directly on
# model_0396000.pt: MuJoCo sim2sim rollouts collapse flat to the ground by t~=3.9s into the clip
# (during the swing itself), progressively earlier across this run's own checkpoint history
# (280k: airborne/inverted only late, t~=6-9s -> 395k: fully collapsed by t~=3.9s) -- the same
# momentum-throwing exploit this term was written for, just re-emerging at a higher w_g than it
# was last tuned against. A same-generation deterministic IsaacSim probe (eval_kick_robustness.py,
# 512 envs) survives 91.6% of the time, confirming this is PhysX tolerating the excess contact/
# momentum that MuJoCo does not -- not a broader policy failure -- so tightening these three
# physical-quantity penalties (rather than touching the shooting reward or the Z-only termination)
# is the targeted fix. kick_penalty_excess_base_lin_vel gets the largest bump: it's the term
# directly implicated by the measured mechanism above, and it's a sustained-over-the-ramp
# quantity (excess builds over ~1-1.5s / 50-75 steps) rather than a single-frame spike, so it
# needs a bigger multiplier for its accumulated total to matter against a one-time 240 burst.
# Retune from scratch using RawEpisode/raw_rew_kick_penalty_* on wandb once a run is going --
# these numbers are a reasoned starting point, not measured against live telemetry.
# 2026-07-19: made these _w_g-AWARE, which the 2026-07-15 note above had already identified as the
# structural bug ("the reward for hitting the ball harder grows while the cost of the resulting
# excess momentum stays fixed at whatever was calibrated for an earlier, lower-w_g run") but fixed
# only by hand-bumping the constants for that run's specific w_g=0.8. Raising shooting_reward_scale
# 0.8 -> 3.0 re-triggered the exploit immediately and measurably: step-matched at ~197k, raw
# kick_penalty_excess_base_lin_vel went 0.0185 -> 0.0865 (**4.7x**), undesired_contacts 1.68 -> 4.62,
# hard_landing 0.0172 -> 0.0329, and the train-conditions topple rate went 32.0% -> 54.7% with 0/128
# envs surviving the window (vs 22.7% before). Exactly the documented momentum-throw, re-emerging at
# a higher w_g than the constants were tuned against -- the third time this project has hit it.
#
# Form: -(stage_b_floor + k * _w_g), NOT plain proportionality. These must stay ACTIVE in Stage B
# (_w_g = 0), which is the whole reason the note above says "deliberately NOT scaled by _w_g" -- a
# bare `* _w_g` would silently zero all three during Stage B and reintroduce the original problem
# from the other side. The floor 3.0 is each term's pre-2026-07-15 Stage-B-adequate value; k is set
# so that at _w_g = 0.8 every weight evaluates to EXACTLY the hand-tuned 2026-07-15 number
# (-15/-15/-30), i.e. this is a strict generalization that changes nothing for past configs and
# auto-tracks w_g from here on:
#     contact_force / hard_landing:  -(3.0 + 15.00 * 0.8) = -15.0   -> at w_g 3.0: -48.0
#     excess_base_lin_vel:           -(3.0 + 33.75 * 0.8) = -30.0   -> at w_g 3.0: -104.25
# base_lin_vel keeps the largest coefficient for the reason given above (sustained accumulation over
# the ~50-75 step swing, competing against a one-shot burst). Still a reasoned starting point, not
# measured -- re-check RawEpisode/raw_rew_kick_penalty_* and the train-conditions topple rate once a
# run is going, and expect to retune k rather than the floor.
_KICK_SAFETY_FLOOR = 3.0

# 2026-07-20: weight moved from a config-time-STATIC -(FLOOR + k*_w_g) to a fixed -1.0, with the
# whole dynamic formula computed INSIDE each term function against the LIVE (possibly still-
# ramping) shooting_reward_scale value -- see holosoma.utils.shooting_curriculum.current_w_g and
# each function's own docstring in kick_safety.py. This was necessary once
# BallConfig.shooting_reward_scale_ramp_iters could make w_g smoothly ramp DURING a run: the old
# formula baked _w_g's value at config-import time (the RAMP TARGET), so these penalties would have
# instantly jumped to their full post-ramp strength on step 1 while the shooting rewards they exist
# to counterbalance were still near zero -- the opposite of the intended pairing ("the cost of
# excess momentum should grow in lockstep with the reward for hitting harder", not race ahead of
# it). floor/k values unchanged from the previous static formula.
#
# 2026-07-28: kick_penalty_excess_contact_force's floor/k/force_threshold are now YAML-configurable
# (configs/stageB_and_C.yaml's global fields in N-skill mode / configs/ball*.yaml in legacy mode) --
# same dual-path resolution every other global (not-per-skill) field in this project uses (see
# config_values/unified/g1/command.py's ood_spawn_probability for the reference pattern). Resolved
# at config-import time, same as _ori_alpha below, since RewardTermCfg.params is a config-time dict.
# hard_landing/excess_base_lin_vel are UNCHANGED (still the hardcoded _KICK_SAFETY_FLOOR/15.00/33.75
# below) -- only contact_force was asked to be made configurable; no reason yet to believe the other
# two need the same treatment.
_multi_skill_cfg_for_contact_penalty = load_multi_skill_config() if multi_skill_mode_enabled() else None
if _multi_skill_cfg_for_contact_penalty is not None:
    _contact_force_floor = _multi_skill_cfg_for_contact_penalty.kick_contact_force_penalty_floor
    _contact_force_k = _multi_skill_cfg_for_contact_penalty.kick_contact_force_penalty_k
    _contact_force_threshold_mult = (
        _multi_skill_cfg_for_contact_penalty.kick_contact_force_threshold_bodyweight_multiplier
    )
else:
    _legacy_ball_cfg_for_contact_penalty = _load_ball_config()
    _contact_force_floor = _legacy_ball_cfg_for_contact_penalty.kick_contact_force_penalty_floor
    _contact_force_k = _legacy_ball_cfg_for_contact_penalty.kick_contact_force_penalty_k
    _contact_force_threshold_mult = (
        _legacy_ball_cfg_for_contact_penalty.kick_contact_force_threshold_bodyweight_multiplier
    )

# "Simultaneous per-skill task configs" (2026-08-15, user-requested) -- each configured skill's OWN
# resolved task_config path (or None if that skill declares no task_config: at all), in skill
# order. Moved up here (rather than defined right before its first real use,
# _apply_per_skill_reward_weight_overrides much further below) because several PARAM-level
# per-skill mechanisms (deadzones, contact-force shape) need it far earlier in this module's
# execution than the reward-WEIGHT mechanism does. Depends on nothing but
# _multi_skill_cfg_for_contact_penalty immediately above, so moving it here changes nothing about
# what it resolves to -- see _apply_per_skill_reward_weight_overrides's own docstring (search
# "Trivial/no-op detection") for the full trivial-case/no-op contract, still accurate from here.
_skill_task_config_paths: list[Path | None] | None = (
    [
        resolve_task_config_path(sc.task_config) if sc.task_config is not None else None
        for sc in _multi_skill_cfg_for_contact_penalty.skills
    ]
    if _multi_skill_cfg_for_contact_penalty is not None
    else None
)


def _per_skill_param(field_name: str, base_value: float) -> list[float] | None:
    """This module's own ``_skill_task_config_paths`` closed over ``resolve_per_skill_param``
    (config_types/reward_tuning.py) -- see that function's own docstring for the full contract
    (no-op cases, fallback semantics). Thin wrapper so every call site in this file reads as a
    plain 2-arg call rather than threading ``_skill_task_config_paths`` through each one."""
    return resolve_per_skill_param(_skill_task_config_paths, field_name, base_value)


# See MultiSkillConfig.bad_motion_body_pos_threshold's own docstring: feeds
# penalty_kick_ee_body_pos_divergence's "threshold" param below, keeping it numerically synced
# with bad_tracking's bad_motion_body_pos termination sub-check (config_values/unified/g1/
# termination.py) under one source of truth -- 0.25 unless set in configs/*.yaml, matching both
# sides' pre-existing hardcoded default exactly. Reuses the SAME already-resolved
# _multi_skill_cfg_for_contact_penalty singleton rather than re-loading (same shortcut already
# used by swing_tracking_sigma_multiplier below).
_bad_motion_body_pos_threshold = (
    _multi_skill_cfg_for_contact_penalty.bad_motion_body_pos_threshold
    if _multi_skill_cfg_for_contact_penalty is not None
    else _load_ball_config().bad_motion_body_pos_threshold
)
# "Simultaneous per-skill task configs" (2026-08-15) -- "Mechanism B" sync: kept in sync with
# termination.py's BadTracking (Mechanism B, stateful -- consumes params_per_skill directly
# rather than through TerminationManager's generic path) not by sharing code, but by resolving
# the SAME field name from each module's own independently-loaded (identical)
# _skill_task_config_paths against the SAME base value -- see termination.py's own comment on
# _bad_motion_body_pos_threshold_per_skill for the full rationale. This term (below) IS
# stateless, so it uses the ordinary Mechanism-A params_per_skill path.
_bad_motion_body_pos_threshold_per_skill = _per_skill_param(
    "bad_motion_body_pos_threshold", _bad_motion_body_pos_threshold
)

# See MultiSkillConfig.ee_body_pos_warmup_threshold's own docstring: feeds
# penalty_kick_ee_body_pos_divergence's "warmup_threshold" param below -- a SEPARATE field from
# _bad_motion_body_pos_threshold above because RoboNaldo's own warmup_threshold follows a
# different (non-monotonic 0.25/0.7/0.7) schedule than their progressively-widening threshold.
_ee_body_pos_warmup_threshold = (
    _multi_skill_cfg_for_contact_penalty.ee_body_pos_warmup_threshold
    if _multi_skill_cfg_for_contact_penalty is not None
    else _load_ball_config().ee_body_pos_warmup_threshold
)
_ee_body_pos_warmup_threshold_per_skill = _per_skill_param(
    "ee_body_pos_warmup_threshold", _ee_body_pos_warmup_threshold
)

# 2026-08-14, user-requested: single GLOBAL standing-height target/deadzone, read from
# HOLOSOMA_SKILLS_CONFIG's own file's top-level `base_robot:` block (see MultiSkillConfig.
# base_robot_target_height/base_robot_deadzone's own docstrings in config_types/multi_skill.py for
# the full parsing/precedence rationale) -- governs BOTH penalty_stand_height (locomotion) and
# penalty_kick_recovery_stand_height (kick) uniformly. None (no multi-skill mode, or skills.yaml
# doesn't set base_robot) falls through to each term's own pre-existing hardcoded default (0.76 /
# 0.015) below -- a true no-op, same discipline as every other field in this section.
_base_robot_target_height = (
    _multi_skill_cfg_for_contact_penalty.base_robot_target_height
    if _multi_skill_cfg_for_contact_penalty is not None
    else None
)
_base_robot_deadzone = (
    _multi_skill_cfg_for_contact_penalty.base_robot_deadzone
    if _multi_skill_cfg_for_contact_penalty is not None
    else None
)
_stand_height_target_height = _base_robot_target_height if _base_robot_target_height is not None else 0.76
_stand_height_deadzone = _base_robot_deadzone if _base_robot_deadzone is not None else 0.015

# 2026-08-06, user-requested plumbing: 4 independent deadzone overrides for
# kick_recovery_posture_reward's own-design (no RoboNaldo mapping) terms -- see each field's own
# docstring in config_types/multi_skill.py for the full motivation. All 4 default to their
# existing hardcoded values below (true no-ops); landing the knobs is this step, an actual
# per-stage progression is a deliberate, separate decision.
_kick_recovery_stand_height_deadzone = (
    _multi_skill_cfg_for_contact_penalty.kick_recovery_stand_height_deadzone
    if _multi_skill_cfg_for_contact_penalty is not None
    else _load_ball_config().kick_recovery_stand_height_deadzone
)
# base_robot_deadzone (if set) supersedes the narrower field immediately above -- see
# MultiSkillConfig.base_robot_deadzone's own docstring for the precedence rationale.
_kick_recovery_stand_height_deadzone_final = (
    _base_robot_deadzone if _base_robot_deadzone is not None else _kick_recovery_stand_height_deadzone
)
_kick_recovery_stand_orientation_deadzone = (
    _multi_skill_cfg_for_contact_penalty.kick_recovery_stand_orientation_deadzone
    if _multi_skill_cfg_for_contact_penalty is not None
    else _load_ball_config().kick_recovery_stand_orientation_deadzone
)
_kick_recovery_stand_feet_width_deadzone = (
    _multi_skill_cfg_for_contact_penalty.kick_recovery_stand_feet_width_deadzone
    if _multi_skill_cfg_for_contact_penalty is not None
    else _load_ball_config().kick_recovery_stand_feet_width_deadzone
)
_kick_recovery_stand_knee_width_deadzone = (
    _multi_skill_cfg_for_contact_penalty.kick_recovery_stand_knee_width_deadzone
    if _multi_skill_cfg_for_contact_penalty is not None
    else _load_ball_config().kick_recovery_stand_knee_width_deadzone
)

# 2026-08-12, deadzone fix for the UNGATED swing-phase orientation penalties below
# (_kick_swing_stability_terms) -- see MultiSkillConfig.kick_swing_orientation_deadzone /
# kick_swing_torso_orientation_deadzone's own docstrings for the full npz-measured rationale.
# Both default to 0.0, matching the terms' own pre-existing hardcoded value -- a true no-op.
_kick_swing_orientation_deadzone = (
    _multi_skill_cfg_for_contact_penalty.kick_swing_orientation_deadzone
    if _multi_skill_cfg_for_contact_penalty is not None
    else _load_ball_config().kick_swing_orientation_deadzone
)
_kick_swing_torso_orientation_deadzone = (
    _multi_skill_cfg_for_contact_penalty.kick_swing_torso_orientation_deadzone
    if _multi_skill_cfg_for_contact_penalty is not None
    else _load_ball_config().kick_swing_torso_orientation_deadzone
)

# "Simultaneous per-skill task configs" (2026-08-15) -- per-skill tables for the 6 deadzones above
# plus the 3 contact-force shape params resolved earlier, via _per_skill_param (see that helper's
# own docstring). None (the overwhelmingly common case: 0/1 skills, or skills that agree) for every
# one of these unless skills GENUINELY diverge on that specific field -- patched into the relevant
# RewardTermCfg.params_per_skill below, alongside the existing GLOBAL "Patch the 4 deadzone
# values" block for the first 4. Deliberately does NOT cover bad_motion_body_pos_threshold /
# ee_body_pos_warmup_threshold (synced with the STATEFUL BadTracking termination class -- needs
# that class changed directly, not this generic mechanism) or motion_head_velocity_smoothing_frames
# (a compile-time per-clip preprocessing param, not a runtime reward/termination param at all) --
# both deliberately out of scope for this pass.
_kick_recovery_stand_height_deadzone_per_skill = _per_skill_param(
    "kick_recovery_stand_height_deadzone", _kick_recovery_stand_height_deadzone_final
)
_kick_recovery_stand_orientation_deadzone_per_skill = _per_skill_param(
    "kick_recovery_stand_orientation_deadzone", _kick_recovery_stand_orientation_deadzone
)
_kick_recovery_stand_feet_width_deadzone_per_skill = _per_skill_param(
    "kick_recovery_stand_feet_width_deadzone", _kick_recovery_stand_feet_width_deadzone
)
_kick_recovery_stand_knee_width_deadzone_per_skill = _per_skill_param(
    "kick_recovery_stand_knee_width_deadzone", _kick_recovery_stand_knee_width_deadzone
)
_kick_swing_orientation_deadzone_per_skill = _per_skill_param(
    "kick_swing_orientation_deadzone", _kick_swing_orientation_deadzone
)
_kick_swing_torso_orientation_deadzone_per_skill = _per_skill_param(
    "kick_swing_torso_orientation_deadzone", _kick_swing_torso_orientation_deadzone
)
_contact_force_floor_per_skill = _per_skill_param("kick_contact_force_penalty_floor", _contact_force_floor)
_contact_force_k_per_skill = _per_skill_param("kick_contact_force_penalty_k", _contact_force_k)
_contact_force_threshold_mult_per_skill = _per_skill_param(
    "kick_contact_force_threshold_bodyweight_multiplier", _contact_force_threshold_mult
)

# Patch the 4 deadzone values into _kick_recovery_standing_terms's already-built dict (defined much
# earlier in this file, BEFORE _multi_skill_cfg_for_contact_penalty exists to resolve from -- same
# forward-reference constraint _apply_reward_weight_overrides/_apply_reward_sigma_overrides solve
# by post-processing an already-assembled dict rather than requiring these values at definition
# time). Rebinding the same name is safe: its only other reference is the dict-merge assembling
# _g1_29dof_unified_reward_terms far below, which reads whatever this name is bound to AT THAT
# POINT in module execution, i.e. this patched version.
_kick_recovery_standing_terms = {
    **_kick_recovery_standing_terms,
    "penalty_kick_recovery_stand_height": replace(
        _kick_recovery_standing_terms["penalty_kick_recovery_stand_height"],
        params={
            **_kick_recovery_standing_terms["penalty_kick_recovery_stand_height"].params,
            # base_robot_target_height/base_robot_deadzone (2026-08-14) take priority when set --
            # see _kick_recovery_stand_height_deadzone_final's own comment above.
            "target_height": _stand_height_target_height,
            "deadzone": _kick_recovery_stand_height_deadzone_final,
        },
        params_per_skill=(
            {"deadzone": _kick_recovery_stand_height_deadzone_per_skill}
            if _kick_recovery_stand_height_deadzone_per_skill is not None
            else None
        ),
    ),
    "penalty_kick_recovery_stand_orientation": replace(
        _kick_recovery_standing_terms["penalty_kick_recovery_stand_orientation"],
        params={
            **_kick_recovery_standing_terms["penalty_kick_recovery_stand_orientation"].params,
            "deadzone": _kick_recovery_stand_orientation_deadzone,
        },
        params_per_skill=(
            {"deadzone": _kick_recovery_stand_orientation_deadzone_per_skill}
            if _kick_recovery_stand_orientation_deadzone_per_skill is not None
            else None
        ),
    ),
    "penalty_kick_recovery_stand_feet_width": replace(
        _kick_recovery_standing_terms["penalty_kick_recovery_stand_feet_width"],
        params={
            **_kick_recovery_standing_terms["penalty_kick_recovery_stand_feet_width"].params,
            "deadzone": _kick_recovery_stand_feet_width_deadzone,
        },
        params_per_skill=(
            {"deadzone": _kick_recovery_stand_feet_width_deadzone_per_skill}
            if _kick_recovery_stand_feet_width_deadzone_per_skill is not None
            else None
        ),
    ),
    "penalty_kick_recovery_stand_knee_width": replace(
        _kick_recovery_standing_terms["penalty_kick_recovery_stand_knee_width"],
        params={
            **_kick_recovery_standing_terms["penalty_kick_recovery_stand_knee_width"].params,
            "deadzone": _kick_recovery_stand_knee_width_deadzone,
        },
        params_per_skill=(
            {"deadzone": _kick_recovery_stand_knee_width_deadzone_per_skill}
            if _kick_recovery_stand_knee_width_deadzone_per_skill is not None
            else None
        ),
    ),
}

# 2026-08-14: patch _penalty_stand_height_term (locomotion-mode standing) the same way, for the
# same forward-reference reason -- see that dict's own comment (search "target_height/deadzone
# below are NOT the values actually used"). Unlike the kick-mode term above, this one had no
# pre-existing yaml-configurable deadzone at all before base_robot_deadzone -- base_robot_* is
# its ONLY override source.
_penalty_stand_height_term = {
    **_penalty_stand_height_term,
    "penalty_stand_height": replace(
        _penalty_stand_height_term["penalty_stand_height"],
        params={
            **_penalty_stand_height_term["penalty_stand_height"].params,
            "target_height": _stand_height_target_height,
            "deadzone": _stand_height_deadzone,
        },
    ),
}

# 2026-08-06, user-requested: ball_over_line's own require_has_kicked opt-in (see that field's
# docstring in shooting.py and MultiSkillConfig.kick_ball_over_line_require_has_kicked's own
# docstring for the full rationale). False (default) = off, bit-identical to before this existed.
_kick_ball_over_line_require_has_kicked = (
    _multi_skill_cfg_for_contact_penalty.kick_ball_over_line_require_has_kicked
    if _multi_skill_cfg_for_contact_penalty is not None
    else _load_ball_config().kick_ball_over_line_require_has_kicked
)
# "Simultaneous per-skill task configs" (2026-08-15, Tier 3) -- None (the common case) unless
# skills genuinely diverge. ball_over_line (stateless) now accepts a per-env [num_envs] tensor for
# require_has_kicked alongside the original plain bool -- see that function's own 2026-08-15
# comment in managers/reward/terms/shooting.py.
_kick_ball_over_line_require_has_kicked_per_skill = _per_skill_param(
    "kick_ball_over_line_require_has_kicked", _kick_ball_over_line_require_has_kicked
)

# 2026-08-21, user-requested: ball_velocity's three retunes (v_ref / latched-peak speed /
# post-locomotion gate) made OPT-IN and yaml-configurable rather than hardcoded. All three default
# to the ORIGINAL pre-retune behavior, so an unedited config is bit-identical. See
# shooting.py::ball_velocity's docstring for the measurement, and for its VALIDATION STATUS caveat.
# v_ref is None-able (None = keep the term's own registered 5.0); the two flags are plain bools.
_kick_ball_velocity_v_ref = (
    _multi_skill_cfg_for_contact_penalty.kick_ball_velocity_v_ref
    if _multi_skill_cfg_for_contact_penalty is not None
    else _load_ball_config().kick_ball_velocity_v_ref
)
_kick_ball_velocity_use_latched_peak_speed = (
    _multi_skill_cfg_for_contact_penalty.kick_ball_velocity_use_latched_peak_speed
    if _multi_skill_cfg_for_contact_penalty is not None
    else _load_ball_config().kick_ball_velocity_use_latched_peak_speed
)
_kick_ball_velocity_use_post_locomotion_gate = (
    _multi_skill_cfg_for_contact_penalty.kick_ball_velocity_use_post_locomotion_gate
    if _multi_skill_cfg_for_contact_penalty is not None
    else _load_ball_config().kick_ball_velocity_use_post_locomotion_gate
)
# 2026-08-22, azimuth-aim refactor: error_ball_to_target's sigma, made yaml-configurable rather
# than hardcoded -- see MultiSkillConfig.kick_error_ball_to_target_sigma's own docstring for the
# angular-tolerance reasoning. None (the common case) keeps the term's own registered 1.0.
_kick_error_ball_to_target_sigma = (
    _multi_skill_cfg_for_contact_penalty.kick_error_ball_to_target_sigma
    if _multi_skill_cfg_for_contact_penalty is not None
    else _load_ball_config().kick_error_ball_to_target_sigma
)
_kick_error_ball_to_target_sigma_per_skill = (
    _per_skill_param("kick_error_ball_to_target_sigma", _kick_error_ball_to_target_sigma)
    if _kick_error_ball_to_target_sigma is not None
    else None
)
# ball_velocity is a STATELESS plain function (like ball_over_line), so per-skill divergence is
# supported the same way -- a per-env [num_envs] tensor in place of the scalar. Only v_ref is
# offered per-skill: the two flags select a code PATH (which buffer / which gate), and a per-env
# tensor cannot index a Python `if`, so they stay global. None (the common case) unless skills
# genuinely diverge.
_kick_ball_velocity_v_ref_per_skill = (
    _per_skill_param("kick_ball_velocity_v_ref", _kick_ball_velocity_v_ref)
    if _kick_ball_velocity_v_ref is not None
    else None
)
# NOTE: patched into _shooting_terms further down (search "_kick_ball_velocity_v_ref" again) --
# that dict is defined ABOVE this resolution block, same forward-reference pattern as
# kick_ball_over_line's own patch.
# NOTE: the patch of this value into kick_ball_over_line's params (inside _p0_regularization_terms)
# happens further down this file, AFTER that dict is actually defined -- search
# "_kick_ball_over_line_require_has_kicked" again to find it. _p0_regularization_terms doesn't
# exist yet at this point in module execution (it's defined later than this resolution block).

_kick_safety_terms = {
    "kick_penalty_excess_contact_force": RewardTermCfg(
        func="holosoma.managers.reward.terms.kick_safety:penalty_excess_foot_contact_force",
        weight=-1.0,
        params={
            "floor": _contact_force_floor,
            "k": _contact_force_k,
            "force_threshold_bodyweight_multiplier": _contact_force_threshold_mult,
        },
        params_per_skill=(
            {
                **({"floor": _contact_force_floor_per_skill} if _contact_force_floor_per_skill is not None else {}),
                **({"k": _contact_force_k_per_skill} if _contact_force_k_per_skill is not None else {}),
                **(
                    {"force_threshold_bodyweight_multiplier": _contact_force_threshold_mult_per_skill}
                    if _contact_force_threshold_mult_per_skill is not None
                    else {}
                ),
            }
            or None
        ),
        task_mode="kick",
    ),
    "kick_penalty_hard_landing": RewardTermCfg(
        func="holosoma.managers.reward.terms.kick_safety:penalty_hard_foot_landing",
        weight=-1.0,
        params={"floor": _KICK_SAFETY_FLOOR, "k": 15.00},
        task_mode="kick",
    ),
    "kick_penalty_excess_base_lin_vel": RewardTermCfg(
        func="holosoma.managers.reward.terms.kick_safety:penalty_excess_base_lin_vel",
        weight=-1.0,
        params={"floor": _KICK_SAFETY_FLOOR, "k": 33.75},  # directly-implicated, sustained term
        task_mode="kick",
    ),
}

# kick_alive -- added 2026-07-19, REVERTED the same day, then RESTORED the same day. Read this
# before touching it again; the flip-flop is instructive and each step was driven by a measurement.
#
# ADDED on the belief that kick mode had no survival reward and the policy was "cashing out" a
# one-shot goal burst then falling. That evidence (100% of 128 envs collapsing) later proved to be a
# MEASUREMENT ARTIFACT of every set_is_evaluating() harness: set_is_evaluating() zeroes the
# locomotion velocity command, so the robot never walks/turns during the settle and its heading stays
# at spawn (~0.00 rad) instead of training's -1.22 +/- 0.81 rad. kick_ball_pos_b / kick_target_pos_b
# are expressed in the robot's HEADING frame, so that hands the policy a ball/target BEARING it never
# trained on -> it aims wrong and topples. Real training-condition topple rate is ~32%, not 100%.
# (memory: stagec-kick-eval-harness-artifact)
#
# REVERTED because, with that artifact removed, a step-matched IsaacSim comparison showed the term
# does nothing for hard toppling: pre-fix kicklowheight-kick1@239000 = 32.0%, post-fix
# kickalive-kick1@239000 = 32.0%.
#
# RESTORED because that reverting decision used an IsaacSim-ONLY metric and never checked MuJoCo --
# the engine that actually matters for sim2sim/hardware. Step-matched @239000, single variable, the
# MuJoCo survival scan says the opposite and says it loudly:
#     pre-kick_alive : fall_step 270, min_z 0.065  (ends up essentially flat on the ground)
#     with kick_alive: fall_step 396, min_z 0.260  (never gets near the ground)
# So the term does NOT reduce the hard-topple RATE, it increases post-kick stabilization MARGIN --
# which the binary "did min_z cross 0.40" test is too coarse to see (IsaacSim min_z p50 only moves
# 0.625 -> 0.674) but which dominates in MuJoCo, where margin is what survives the solver difference.
# It is the ONLY intervention in this project that ever measurably improved MuJoCo survival; six
# separate simulator-parameter matching attempts all moved it ~0 (memory:
# stagec-kick-sim2sim-model-audit-closes-parametric-hypothesis).
#
# Cost, kept in view: ~+13.3 reward/episode, which at shooting_reward_scale=0.8 is ~3x the entire
# shooting reward (~+4.6 vs motion tracking's ~+43.6) and dilutes shooting's share of the kick budget
# from ~9.3% to ~7.4%. That is a real tradeoff, accepted deliberately: sim2sim survival is the binding
# constraint, and the alternative (raising shooting_reward_scale to out-shout it) was tried at
# _w_g=3.0 and re-triggered the momentum-throw exploit, collapsing MuJoCo survival 396 -> 58.
# 2026-08-19: see MultiSkillConfig.post_flip_alive_scale's own docstring (multi_skill.py) for the
# full measured rationale (alive-farming: a policy that avoids the kick and simply outlasts the
# clip collects `alive` -- weight 10.0, task_mode="locomotion", the SHARED locomotion survival
# term, NOT `kick_alive` above -- at full value for the whole post-flip remainder, a cheaper payout
# than kicking and risking the fall/self-collision a real swing exposes it to). 1.0 (default) =
# exact no-op: `_alive_term_override` stays the empty dict, so `alive` is never touched here and
# the merge below resolves it purely from `_tagged(g1_29dof_loco_fast_sac.terms, "locomotion")`,
# byte-identical to every run before this field existed.
_post_flip_alive_scale = (
    _multi_skill_cfg_for_contact_penalty.post_flip_alive_scale
    if _multi_skill_cfg_for_contact_penalty is not None
    else 1.0
)
_alive_term_override = (
    {
        "alive": replace(
            g1_29dof_loco_fast_sac.terms["alive"],
            func="holosoma.managers.reward.terms.kick_scale_wrappers:alive_post_flip_scaled",
            params={
                **g1_29dof_loco_fast_sac.terms["alive"].params,
                "post_flip_alive_scale": _post_flip_alive_scale,
            },
            task_mode="locomotion",  # matches what _tagged(..., "locomotion") below would stamp
        )
    }
    if _post_flip_alive_scale != 1.0
    else {}
)

_kick_alive_term = {
    "kick_alive": RewardTermCfg(
        # kick_alive_scaled: a thin wrapper around locomotion:alive applying the new per-skill
        # kick_alive_reward_scale (utils/kick_reward_scales.py) -- must NOT point at `alive`
        # itself, which is SHARED with locomotion mode's own identical-function reward; wrapping
        # keeps that shared function completely untouched. weight stays 10.0 (the term's own base
        # magnitude); the per-skill scale is an ADDITIONAL multiplier applied at runtime inside
        # the wrapper, 1.0 (no-op) by default, same as every other new scale in this project.
        func="holosoma.managers.reward.terms.kick_scale_wrappers:kick_alive_scaled",
        weight=10.0,
        task_mode="kick",
    ),
}

# Potential-based shaping on a capture-point balance margin -- see
# managers/reward/terms/balance_potential.py's module docstring for the full measured rationale
# (kick reward has no gradient in the pre-fall regime) and for why potential-based specifically
# (Ng/Harada/Russell invariance: provably cannot change the optimal policy, so unlike every other
# knob tried on this task it cannot trade kick quality against stability).
#
# Resolved at config-import time, same discipline as _contact_force_floor / _ori_alpha above.
# Weight 0.0 -> the term is OMITTED ENTIRELY rather than registered with a zero weight, so a run
# that hasn't opted in is bit-identical to before this existed (no extra term in the manager's
# iteration order, no extra Episode/ logging key, no per-step compute).
#
# gamma MUST match the agent's own discount for the invariance theorem to hold.
#
# 2026-08-09 BUG FIX -- this was hardcoded `0.97` with a comment claiming algo config "is not in
# scope at this import point". Both halves went stale: FastSACConfig.kick_gamma (added 2026-07-30)
# installs a PER-TASK-MODE discount (agents/fast_sac/fast_sac_agent.py:378-387), and this term is
# registered task_mode="kick" -- so the discount its own envs are trained under is kick_gamma, not
# gamma. configs/task_config_stageC1.yaml sets kick_gamma: 0.99, so kick envs were being trained
# at 0.99 while this shaping kept telescoping at 0.97: still *a* shaping function, but no longer
# the INVARIANT one (balance_potential.py's module docstring, point 1). Near-harmless while the
# weight sat at 1.0 (contribution ~-0.004); NOT harmless at the 50.0 the term was designed for.
# The config IS reachable here -- `_multi_skill_cfg_for_contact_penalty` above is the very object
# that carries it -- so resolve it from the same field config_values/unified/g1/experiment.py's
# own `_kick_gamma` feeds to the agent, and take the fallback from FastSACConfig's own default
# rather than restating the literal, so the two cannot silently diverge a second time.
_balance_potential_weight = (
    _multi_skill_cfg_for_contact_penalty.balance_potential_weight
    if _multi_skill_cfg_for_contact_penalty is not None
    else _load_ball_config().balance_potential_weight
)

_balance_potential_kick_gamma = (
    _multi_skill_cfg_for_contact_penalty.kick_gamma
    if _multi_skill_cfg_for_contact_penalty is not None
    else _load_ball_config().kick_gamma
)
# kick_gamma unset => fast_sac runs ONE discount group and every env (kick included) uses
# FastSACConfig.gamma, so that is the correct fallback -- not an arbitrary default.
if _balance_potential_kick_gamma is None:
    _balance_potential_kick_gamma = _FastSACConfig.gamma

# "Simultaneous per-skill task configs" (2026-08-15): balance_potential_weight is a top-level
# scalar (not one of the 5 reward-tuning categories) feeding a TERM WEIGHT, not a params entry --
# reuses weight_per_skill (RewardManager already handles this uniformly for stateful AND
# stateless terms; CapturePointPotentialShaping below is stateful, which is exactly why this
# rides weight_per_skill rather than the params_per_skill mechanism, off-limits to stateful terms).
_balance_potential_weight_per_skill = _per_skill_param("balance_potential_weight", _balance_potential_weight)
# Representative/fallback scalar -- skill 0's own resolved value, same convention
# _apply_per_skill_reward_weight_overrides already uses for weight_per_skill's sibling scalar.
_balance_potential_weight_representative = (
    _balance_potential_weight_per_skill[0]
    if _balance_potential_weight_per_skill is not None
    else _balance_potential_weight
)

_kick_balance_potential_term = (
    {
        "kick_balance_potential": RewardTermCfg(
            func="holosoma.managers.reward.terms.balance_potential:CapturePointPotentialShaping",
            weight=_balance_potential_weight_representative,
            weight_per_skill=_balance_potential_weight_per_skill,
            task_mode="kick",
            params={
                "gamma": _balance_potential_kick_gamma,
                "foot_radius": 0.09,
                "contact_force_threshold": 1.0,
                # Deliberately the same 0.40 kick_low_height terminates on, so the potential
                # reaches 0 exactly where the fall termination fires.
                "fall_height": 0.40,
                "nominal_height": 0.78,
            },
        ),
    }
    # Registered if EITHER the global weight is positive (original condition, unchanged) OR any
    # skill genuinely diverges on it (_per_skill_param only returns non-None on genuine
    # divergence) -- otherwise a skill wanting balance_potential_weight>0 while the GLOBAL config
    # says 0.0 would have no RewardTermCfg to attach a per-skill table to at all.
    if _balance_potential_weight_representative > 0.0 or _balance_potential_weight_per_skill is not None
    else {}
)

# Per-joint DOF-position tracking, strike phase only -- see managers/reward/terms/wbt.py:
# MotionStrikeDofPosErrorExp for the full mechanism/rationale (arm/waist joints drifting up to
# 53 deg from the reference clip during the strike, on axes -- shoulder_yaw, wrist_roll -- the 6
# Cartesian motion-tracking terms above are structurally blind to; the kick leg itself already
# tracks well by the same measure and is deliberately excluded from the mask, along with the
# support leg for balance freedom).
#
# dof_names sourced from g1_29dof.upper_dof_names (config_values/robot.py) rather than a
# hand-typed duplicate literal -- this experiment's robot config passes upper_dof_names through
# unchanged from g1_29dof, and both IsaacSim and IsaacGym assert env.simulator.dof_names ==
# env.robot_config.dof_names at setup, so this is provably the same 17 names the term's own
# __init__ will resolve against env.simulator.dof_names at runtime.
#
# weight=0.0: explicit opt-in only, NOT a reasoned-nonzero default -- kept IN this dict (not
# omitted, contrast _kick_balance_potential_term above) specifically so
# configs/kicking_motion_reward_tuning.yaml's weight/sigma overrides can reach it later
# (_apply_reward_weight_overrides requires the term name already be registered). RewardManager
# skips weight==0.0 terms entirely (managers/reward/manager.py:60-62), so this is a true no-op:
# not instantiated, not computed, no Episode/ logging key, until the yaml deliberately turns it
# on. sigma=0.5 rad / weight=1.0 (once enabled) are a reasoned, UNVALIDATED starting point -- see
# the term's own docstring for the numbers behind them; retune against real
# raw_rew_motion_strike_dof_pos_error_exp telemetry before trusting this in production, same
# discipline as every other new mechanism this project ships.
_motion_strike_dof_pos_term = {
    "motion_strike_dof_pos_error_exp": RewardTermCfg(
        func="holosoma.managers.reward.terms.wbt:MotionStrikeDofPosErrorExp",
        params={"sigma": 0.5, "dof_names": list(g1_29dof.upper_dof_names)},
        weight=0.0,
        task_mode="kick",
    ),
    # 2026-08-14: leg counterpart of the term immediately above, same class
    # (MotionStrikeDofPosErrorExp is generic over dof_names, no leg-specific code needed) applied
    # to the leg's OWN low-Cartesian-authority axes -- rotation about a limb's long axis
    # (hip_yaw, ankle_roll) or ab/adduction (hip_roll), the exact same failure-mode CATEGORY the
    # term above was built to patch for the arms (see that term's docstring: those axes barely
    # move the tracked link's Cartesian position, so the 6 Cartesian motion_*_error_exp terms are
    # structurally blind to drift on them). Legs were originally excluded entirely (that term's own
    # docstring: "kick leg already tracks well... support leg needs freedom for single-support
    # balance", measured on ckpt 440k of 20260802_122622) -- a LATER run
    # (20260813_005001-stageC1-1skill-locoflip-shooting05-new-fixes-3-locomotion) contradicted the
    # first half of that: direct measurement found RIGHT leg peak joint velocity +85% and RMS jerk
    # +238% over training (worse than the arms, which the upper-body term above was already
    # successfully holding: arm peak velocity roughly FLAT/-7% over the same run). This deliberately
    # does NOT widen to the sagittal/high-authority axes (hip_pitch, knee, ankle_pitch) on EITHER
    # leg -- the swing leg's excursion on those is the actual, intentional kick, and the support
    # leg's own docstring reasoning (needs freedom for single-support balance) still applies; the
    # broader jerk/oscillation growth on those axes is instead addressed by
    # penalty_kick_strike_dof_acc (managers/reward/terms/locomotion.py), which regularizes
    # AMPLITUDE OF MOTION without constraining the mean trajectory the way dense position-tracking
    # would. sigma matches the upper-body term's own default exactly (0.5 rad) -- same formula,
    # same per-joint-mean-then-exp construction, directly comparable magnitude regardless of the
    # 17-vs-6 joint-count difference (mean, not sum, over joints). Weight 0.0 here (explicit
    # opt-in, same discipline as every other term in this file) -- see
    # configs/task_config_stageC1.yaml's own comment for the active starting weight and rationale.
    "motion_strike_leg_null_dof_pos_error_exp": RewardTermCfg(
        func="holosoma.managers.reward.terms.wbt:MotionStrikeDofPosErrorExp",
        params={
            "sigma": 0.5,
            "dof_names": [
                "left_hip_roll_joint",
                "left_hip_yaw_joint",
                "left_ankle_roll_joint",
                "right_hip_roll_joint",
                "right_hip_yaw_joint",
                "right_ankle_roll_joint",
            ],
        },
        weight=0.0,
        task_mode="kick",
    ),
    # 2026-08-24: the NON-SATURATING counterpart to the two exp terms above -- see
    # managers/reward/terms/wbt.py:StrikeDofDivergencePenalty's own docstring for the full
    # measurement (per-joint strike probes + a lag/ROM probe: the strike divergence is neither lag
    # nor shortfall but 1.7x-4.9x OVER-motion, and a Gaussian's gradient is 8-28x weaker on
    # exactly the joints that have drifted furthest). dof_names deliberately left at None = ALL 29
    # DOF: this is the only strike-phase term that prices the sagittal kick chain
    # (hip_pitch/knee/ankle_pitch, both legs), which the two exp terms above omit between them.
    # threshold 0.35 rad = 20 deg is a DEADBAND -- measured typical per-joint strike error is
    # ~20 deg and the pathological joints sit at 54-64 deg, so ordinary tracking is untouched.
    # weight=0.0 -> true no-op until a config opts in (RewardManager skips weight==0.0 terms).
    "kick_penalty_strike_dof_divergence": RewardTermCfg(
        func="holosoma.managers.reward.terms.wbt:StrikeDofDivergencePenalty",
        params={"threshold": 0.35, "dof_names": None},
        weight=0.0,
        task_mode="kick",
    ),
}

# Kick-mode stability regularizers ported from RoboNaldo (arXiv:2606.11092), 2026-08-04 -- see
# managers/reward/terms/locomotion.py:penalty_kick_swing_orientation/
# penalty_kick_swing_torso_orientation and managers/reward/terms/wbt.py:KickFeetSlip for the full
# per-term rationale (why each one, what it ports from, and why the L1+deadzone substitution on
# the two orientation terms is the like-for-like upgrade rather than a deviation from the port).
# All three UNGATED across the whole kick episode -- unlike the existing penalty_kick_recovery_*
# family, these are meant to give dense pressure during the SWING specifically, where nothing
# else currently discourages excess lean or a sliding stance foot.
#
# weight=0.0 for all three: explicit opt-in only, NOT reasoned-nonzero defaults -- kept IN this
# dict (not omitted) so configs/kicking_motion_reward_tuning.yaml's weight overrides can reach
# them once registered (_apply_reward_weight_overrides requires the term already exist).
# RewardManager skips weight==0.0 terms entirely (managers/reward/manager.py:60-62): true no-ops
# until deliberately enabled. Reasoned-but-UNVALIDATED starting weights are documented on each
# term's own docstring; retune against real telemetry before trusting them in production, same
# discipline as every other new mechanism this project ships.
_kick_swing_stability_terms = {
    "kick_penalty_swing_orientation": RewardTermCfg(
        func="holosoma.managers.reward.terms.locomotion:penalty_kick_swing_orientation",
        params={"deadzone": _kick_swing_orientation_deadzone},
        params_per_skill=(
            {"deadzone": _kick_swing_orientation_deadzone_per_skill}
            if _kick_swing_orientation_deadzone_per_skill is not None
            else None
        ),
        weight=0.0,
        task_mode="kick",
    ),
    "kick_penalty_swing_torso_orientation": RewardTermCfg(
        func="holosoma.managers.reward.terms.locomotion:penalty_kick_swing_torso_orientation",
        params={"deadzone": _kick_swing_torso_orientation_deadzone, "torso_body_name": "torso_link"},
        params_per_skill=(
            {"deadzone": _kick_swing_torso_orientation_deadzone_per_skill}
            if _kick_swing_torso_orientation_deadzone_per_skill is not None
            else None
        ),
        weight=0.0,
        task_mode="kick",
    ),
    "kick_feet_slip": RewardTermCfg(
        func="holosoma.managers.reward.terms.wbt:KickFeetSlip",
        params={"foot_body_names": ["left_ankle_roll_link", "right_ankle_roll_link"], "threshold": 1.0},
        weight=0.0,
        task_mode="kick",
    ),
}

# 2026-08-05, ported from RoboNaldo (arXiv:2606.11092) -- the "P0" regularization block per
# ROBONALDO_PORT_SCOPE.md Sec 3b/5: this project has ~4 real equivalents of RoboNaldo's 20
# regularization terms; these 4 are the highest-priority of the remaining gap. See each term's own
# docstring (managers/reward/terms/wbt.py) for the full formula and every deliberate adaptation
# from RoboNaldo's IsaacLab-specific mechanisms (ContactSensor, env.scene.env_origins,
# adapt_motion_flag-derived command magnitude -- none of which this project has or wants).
#
# weight=0.0 for all four: explicit opt-in only, same discipline as _kick_swing_stability_terms
# above -- kept IN this dict so configs/kicking_motion_reward_tuning.yaml's weight overrides can
# reach them once registered. Reasoned-but-UNVALIDATED starting weights, ratio-derived from
# RoboNaldo's own S1/S2a/S2b numbers (ROBONALDO_PORT_SCOPE.md Sec 3b), documented on each term's
# own docstring and in the tuning yaml's staged comment -- not yet checked against a real training
# run.
_p0_regularization_terms = {
    "kick_arm_default_pose": RewardTermCfg(
        func="holosoma.managers.reward.terms.wbt:ArmDefaultPose",
        params={
            "arm_dof_names": list(g1_29dof.upper_left_arm_dof_names) + list(g1_29dof.upper_right_arm_dof_names),
            "elbow_weight_multiplier": 5.0,
        },
        weight=0.0,
        task_mode="kick",
    ),
    "kick_feet_air_time": RewardTermCfg(
        func="holosoma.managers.reward.terms.wbt:KickFeetAirTime",
        params={
            "foot_body_names": ["left_ankle_roll_link", "right_ankle_roll_link"],
            "contact_force_threshold": 1.0,
            "air_time_threshold": 0.25,
        },
        weight=0.0,
        task_mode="kick",
    ),
    "kick_swing_feet_clearance": RewardTermCfg(
        func="holosoma.managers.reward.terms.wbt:KickSwingFeetClearance",
        params={
            "foot_body_names": ["left_ankle_roll_link", "right_ankle_roll_link"],
            "contact_force_threshold": 1.0,
            "target_height": 0.12,
            "max_penalty": 0.5,
        },
        weight=0.0,
        task_mode="kick",
    ),
    "kick_no_fly": RewardTermCfg(
        func="holosoma.managers.reward.terms.wbt:kick_no_fly",
        params={"height_threshold": 0.05},
        weight=0.0,
        task_mode="kick",
    ),
    # 2026-08-05 -- RoboNaldo's real S1/S2/S2b post-kick stabilization pressure (their
    # stable_anchor_pos_tracking is Stage-3-only, adapt_motion_flag-gated -- NOT ported, see
    # penalty_kick_unstable's own docstring for the full correction/rationale).
    "kick_penalty_unstable": RewardTermCfg(
        func="holosoma.managers.reward.terms.locomotion:penalty_kick_unstable",
        params={"ang_vel_weight": 0.5, "grace_steps": 50.0},
        weight=0.0,
        task_mode="kick",
    ),
    # 2026-08-05 -- the remaining §3b/3c RoboNaldo regularization terms (ROBONALDO_PORT_SCOPE.md),
    # all ships at weight=0.0 (explicit opt-in), same discipline as every term above.
    "kick_feet_contact_time": RewardTermCfg(
        func="holosoma.managers.reward.terms.wbt:KickFeetContactTime",
        params={
            "foot_body_names": ["left_ankle_roll_link", "right_ankle_roll_link"],
            "contact_force_threshold": 1.0,
            "contact_time_threshold": 0.5,
        },
        weight=0.0,
        task_mode="kick",
    ),
    "kick_penalty_lin_vel_z": RewardTermCfg(
        func="holosoma.managers.reward.terms.locomotion:penalty_kick_lin_vel_z",
        params={},
        weight=0.0,
        task_mode="kick",
    ),
    "kick_penalty_dof_vel": RewardTermCfg(
        func="holosoma.managers.reward.terms.locomotion:penalty_kick_dof_vel",
        params={},
        weight=0.0,
        task_mode="kick",
    ),
    "kick_penalty_torque": RewardTermCfg(
        func="holosoma.managers.reward.terms.locomotion:penalty_kick_torque",
        params={},
        weight=0.0,
        task_mode="kick",
    ),
    # 2026-08-14: strike-phase joint-acceleration smoothness penalty -- see
    # penalty_kick_strike_dof_acc's own docstring (managers/reward/terms/locomotion.py) for the
    # full measured rationale (the same run's jerk/oscillation growth that motivated
    # motion_strike_leg_null_dof_pos_error_exp above and the kick_penalty_dof_vel sign-bug fix
    # immediately above). dof_names=None (all 29 DOF) -- params={} takes the function's own
    # broad default. Weight 0.0 here; see configs/task_config_stageC1.yaml's own comment for the
    # active starting weight, calibrated against measured raw sum(dof_acc^2) magnitude.
    "kick_penalty_strike_dof_acc": RewardTermCfg(
        func="holosoma.managers.reward.terms.locomotion:penalty_kick_strike_dof_acc",
        params={},
        weight=0.0,
        task_mode="kick",
    ),
    # 2026-08-14: locomotion-approach counterpart -- SEPARATE term, not a widened gate on the one
    # above, because approach and strike want opposite things from joint acceleration (approach:
    # none, it's pure jitter; strike: that IS the kick's power). See
    # penalty_kick_approach_dof_acc's own docstring for the full rationale, including its
    # relevance to the Stage D locomotion->kick mid-episode handoff (entry always lands in this
    # exact approach window, never the strike). Weight 0.0 here -- deliberately NOT activated by
    # this change; needs its own live-measured approach-phase magnitude before a starting weight
    # is chosen, same discipline penalty_kick_strike_dof_acc's own calibration used.
    "kick_penalty_approach_dof_acc": RewardTermCfg(
        func="holosoma.managers.reward.terms.locomotion:penalty_kick_approach_dof_acc",
        params={},
        weight=0.0,
        task_mode="kick",
    ),
    # Reuses the SAME UndesiredContacts class already shipped for locomotion/WBT mode -- the
    # formula is identical, only the registration (task_mode, threshold) differs. threshold=100.0
    # (NOT this project's existing WBT-mode value of 1.0) matches RoboNaldo's OWN kick-specific
    # registration exactly -- a deliberately different, much higher contact-force bar for a task
    # that involves intentional, forceful ball contact, not a copy of the locomotion-mode value.
    "kick_undesired_contacts": RewardTermCfg(
        func="holosoma.managers.reward.terms.wbt:UndesiredContacts",
        params={
            "threshold": 100.0,
            "undesired_contacts_body_names": (
                "^(?!left_foot_contact_point$)(?!right_foot_contact_point$)"
                "(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$)"
                "(?!left_ankle_roll_link$)(?!right_ankle_roll_link$).+$"
            ),
        },
        weight=0.0,
        task_mode="kick",
    ),
    "kick_penalty_ee_body_pos_divergence": RewardTermCfg(
        func="holosoma.managers.reward.terms.locomotion:penalty_kick_ee_body_pos_divergence",
        params={
            # Synced with bad_tracking's bad_motion_body_pos threshold -- see
            # MultiSkillConfig.bad_motion_body_pos_threshold's own docstring.
            "threshold": _bad_motion_body_pos_threshold,
            # See MultiSkillConfig.ee_body_pos_warmup_threshold's own docstring.
            "warmup_threshold": _ee_body_pos_warmup_threshold,
            "warmup_steps": 20,
            "body_names": ["left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link"],
        },
        params_per_skill=(
            {
                **({"threshold": _bad_motion_body_pos_threshold_per_skill} if _bad_motion_body_pos_threshold_per_skill is not None else {}),
                **({"warmup_threshold": _ee_body_pos_warmup_threshold_per_skill} if _ee_body_pos_warmup_threshold_per_skill is not None else {}),
            }
            or None
        ),
        weight=0.0,
        task_mode="kick",
    ),
    # 2026-08-06: require_has_kicked is patched in further down this file (search
    # "_kick_ball_over_line_require_has_kicked"), once _multi_skill_cfg_for_contact_penalty exists
    # to resolve it from -- defaults False (unset here) unless overridden in configs/*.yaml.
    "kick_ball_over_line": RewardTermCfg(
        func="holosoma.managers.reward.terms.shooting:ball_over_line",
        params={"over_line_dist": 7.0, "back_line_dist": -1.0},
        weight=0.0,
        task_mode="kick",
    ),
    "kick_robot_com_ball_distance": RewardTermCfg(
        func="holosoma.managers.reward.terms.shooting:robot_com_ball_distance",
        params={"std": 0.5},
        weight=0.0,
        task_mode="kick",
    ),
    "kick_robot_torso_ball_distance": RewardTermCfg(
        func="holosoma.managers.reward.terms.shooting:robot_torso_ball_distance",
        params={"std": 0.5, "body_names": ["torso_link"]},
        weight=0.0,
        task_mode="kick",
    ),
    "kick_penalize_weak_foot_contact": RewardTermCfg(
        func="holosoma.managers.reward.terms.shooting:penalize_weak_foot_contact",
        params={"threshold": 0.12, "std": 0.1},
        weight=0.0,
        task_mode="kick",
    ),
    "kick_penalize_self_contact_feet": RewardTermCfg(
        func="holosoma.managers.reward.terms.shooting:penalize_self_contact_feet",
        params={
            "threshold": 0.2,
            "std": 0.05,
            "foot_body_names": ["left_ankle_roll_link", "right_ankle_roll_link"],
        },
        weight=0.0,
        task_mode="kick",
    ),
    "kick_action_smoothness": RewardTermCfg(
        func="holosoma.managers.reward.terms.wbt:KickActionSmoothness",
        params={},
        weight=0.0,
        task_mode="kick",
    ),
}

# 2026-08-06, user-requested: patch _kick_ball_over_line_require_has_kicked (resolved earlier in
# this file, search "_kick_ball_over_line_require_has_kicked = (") into kick_ball_over_line's
# params, now that _p0_regularization_terms (the dict it actually lives in) has just been defined
# immediately above. Same forward-reference pattern as _kick_recovery_standing_terms's own patch
# earlier in this file. Safe to rebind: this dict's only other reference is the dict-merge
# assembling _g1_29dof_unified_reward_terms far below.
_p0_regularization_terms = {
    **_p0_regularization_terms,
    "kick_ball_over_line": replace(
        _p0_regularization_terms["kick_ball_over_line"],
        params={
            **_p0_regularization_terms["kick_ball_over_line"].params,
            "require_has_kicked": _kick_ball_over_line_require_has_kicked,
        },
        params_per_skill=(
            {"require_has_kicked": _kick_ball_over_line_require_has_kicked_per_skill}
            if _kick_ball_over_line_require_has_kicked_per_skill is not None
            else None
        ),
    ),
}

# 2026-08-21: patch ball_velocity's three opt-in retunes into _shooting_terms (defined far above
# this file's resolution block), same forward-reference pattern as kick_ball_over_line just above.
# Every value defaults to the original behavior, so an unedited config rebinds params to exactly
# what was already there.
_shooting_terms = {
    **_shooting_terms,
    "kick_ball_velocity": replace(
        _shooting_terms["kick_ball_velocity"],
        params={
            **_shooting_terms["kick_ball_velocity"].params,
            **({"v_ref": _kick_ball_velocity_v_ref} if _kick_ball_velocity_v_ref is not None else {}),
            "use_latched_peak_speed": _kick_ball_velocity_use_latched_peak_speed,
            "use_post_locomotion_gate": _kick_ball_velocity_use_post_locomotion_gate,
        },
        params_per_skill=(
            {"v_ref": _kick_ball_velocity_v_ref_per_skill}
            if _kick_ball_velocity_v_ref_per_skill is not None
            else None
        ),
    ),
    "kick_error_ball_to_target": replace(
        _shooting_terms["kick_error_ball_to_target"],
        params={
            **_shooting_terms["kick_error_ball_to_target"].params,
            **(
                {"sigma": _kick_error_ball_to_target_sigma}
                if _kick_error_ball_to_target_sigma is not None
                else {}
            ),
        },
        params_per_skill=(
            {"sigma": _kick_error_ball_to_target_sigma_per_skill}
            if _kick_error_ball_to_target_sigma_per_skill is not None
            else None
        ),
    ),
}

# Selective orientation-tracking sharpening (RoboNaldo's per-stage alpha, applied to the balance
# channel only -- see BallConfig.orientation_tracking_alpha for the full rationale and the
# measurements behind it). alpha = 1.0 (the default) reproduces previous behavior exactly, so this
# is inert for every existing config; a Stage-C yaml opts in by setting orientation_tracking_alpha.
_ori_alpha = _load_ball_config().orientation_tracking_alpha


def _sharpen_orientation_tracking(terms: dict[str, RewardTermCfg]) -> dict[str, RewardTermCfg]:
    if _ori_alpha == 1.0:
        return terms
    targets = ("motion_global_ref_orientation_error_exp", "motion_relative_body_orientation_error_exp")
    out = dict(terms)
    for name in targets:
        cfg = out.get(name)
        if cfg is None or "sigma" not in cfg.params:
            continue
        out[name] = replace(cfg, params={**cfg.params, "sigma": cfg.params["sigma"] / _ori_alpha})
    return out


# Global (not per-skill) swing-phase tracking-sigma widening (2026-08-01) -- see
# MultiSkillConfig.swing_tracking_sigma_multiplier's own docstring for the full crossover-
# arithmetic rationale and its explicit precedent warning. Resolved once here, same
# MultiSkillConfig-or-legacy-BallConfig pattern as _balance_potential_weight below (reuses the
# SAME already-resolved _multi_skill_cfg_for_contact_penalty rather than re-loading). 1.0 unless
# set in configs/*.yaml -- bit-identical by default, injected into the 6 motion-tracking terms'
# params by _scale_kick_motion_tracking below.
_swing_tracking_sigma_multiplier = (
    _multi_skill_cfg_for_contact_penalty.swing_tracking_sigma_multiplier
    if _multi_skill_cfg_for_contact_penalty is not None
    else _load_ball_config().swing_tracking_sigma_multiplier
)

# FIX 4 (2026-08-12): see managers/reward/terms/kick_scale_wrappers.py's
# _post_flip_tracking_decay_multiplier for the full measured rationale. 0.0 (default) = exact
# no-op -- these 7 terms stay at task_mode="kick" (see _apply_post_flip_reward_decay below), so
# the outer manager's own masking already zeroes every post-flip env's contribution regardless of
# this value; only switched to task_mode=None (letting the decay multiplier take over) when > 0.
_post_flip_reward_decay_steps = (
    _multi_skill_cfg_for_contact_penalty.post_flip_reward_decay_steps
    if _multi_skill_cfg_for_contact_penalty is not None
    else _load_ball_config().post_flip_reward_decay_steps
)
# "Simultaneous per-skill task configs" (2026-08-15, Tier 3 Group B) -- None (the common case)
# unless skills genuinely diverge. Feeds a STATELESS function
# (_post_flip_tracking_decay_multiplier, kick_scale_wrappers.py) via the ordinary Mechanism-A
# params_per_skill path -- see that function's own 2026-08-15 comment for the tensor-aware
# handling (including a real bug caught and fixed there: envs whose OWN decay_steps<=0 must land
# at 0.0 post-flip, not an unconditional 1.0).
_post_flip_reward_decay_steps_per_skill = _per_skill_param(
    "post_flip_reward_decay_steps", _post_flip_reward_decay_steps
)
# _apply_post_flip_reward_decay below makes a STRUCTURAL task_mode registration decision (whether
# RewardManager's own masking or the decay multiplier is what zeroes post-flip envs) -- a
# per-TERM, not per-skill, property, so it must switch to the unmasked (task_mode=None) form the
# instant ANY skill wants a genuine decay window, not just when the global scalar does. This
# computes that "should ANY skill's decay be honored" signal as a plain float >0.0 iff so, keeping
# _apply_post_flip_reward_decay's own signature/behavior untouched (still just checks <= 0.0).
_post_flip_reward_decay_steps_for_registration = max(
    _post_flip_reward_decay_steps, *(_post_flip_reward_decay_steps_per_skill or [0.0])
)

# 2026-08-13: see managers/reward/terms/kick_scale_wrappers.py's _pre_kick_reward_ramp_multiplier
# for the full rationale -- mirror of FIX 4 above, opposite boundary. 0.0 (default) = exact no-op
# (that function returns a bare python float 1.0 unconditionally at that value); unlike FIX 4, no
# task_mode registration change is needed here (see that function's own docstring for why), so
# this is only ever injected into _scale_kick_motion_tracking's params below, nothing else.
#
# MultiSkillConfig-only, no legacy BallConfig counterpart -- same scope choice as command.py's own
# resolution of this field (config_values/unified/g1/command.py), for the same reason: the
# mechanism this backs is inherently scoped to a fixed per-env skill assignment that only exists
# in N-skill mode. Legacy single-skill configs always get the 0.0 off-default.
_pre_kick_reward_ramp_steps = (
    _multi_skill_cfg_for_contact_penalty.pre_kick_reward_ramp_steps
    if _multi_skill_cfg_for_contact_penalty is not None
    else 0.0
)
# "Simultaneous per-skill task configs" (2026-08-15, Tier 3 Group B) -- None (the common case)
# unless skills genuinely diverge. No registration-gate concern here (unlike post_flip_reward_
# decay_steps above): _pre_kick_reward_ramp_multiplier's own docstring confirms no task_mode
# change is needed regardless of value, so this is a plain Mechanism-A params_per_skill field.
_pre_kick_reward_ramp_steps_per_skill = _per_skill_param("pre_kick_reward_ramp_steps", _pre_kick_reward_ramp_steps)

# Global (not per-skill) switch for foot_strike_pitch's reference_relative mode -- see
# MultiSkillConfig.use_foot_strike_pitch_reference_relative's own docstring for the full
# rationale. Same resolve-late pattern as _swing_tracking_sigma_multiplier immediately above:
# _shooting_terms (defined earlier in this file, at kick_foot_strike_pitch's own literal params)
# is built before _multi_skill_cfg_for_contact_penalty exists to resolve this from, so the value
# is injected via replace() into a copy below rather than at the dict literal itself. False unless
# set in configs/*.yaml -- bit-identical by default (foot_strike_pitch's own reference_relative
# param also defaults False).
_use_foot_strike_pitch_reference_relative = (
    _multi_skill_cfg_for_contact_penalty.use_foot_strike_pitch_reference_relative
    if _multi_skill_cfg_for_contact_penalty is not None
    else _load_ball_config().use_foot_strike_pitch_reference_relative
)
# "Simultaneous per-skill task configs" (2026-08-15, Tier 3) -- None (the common case) unless
# skills genuinely diverge. foot_strike_pitch (stateless) now accepts a per-env [num_envs] tensor
# for reference_relative (torch.where select) alongside the original plain bool -- see that
# function's own 2026-08-15 comment in managers/reward/terms/shooting.py.
_use_foot_strike_pitch_reference_relative_per_skill = _per_skill_param(
    "use_foot_strike_pitch_reference_relative", _use_foot_strike_pitch_reference_relative
)
_shooting_terms = {
    **_shooting_terms,
    "kick_foot_strike_pitch": replace(
        _shooting_terms["kick_foot_strike_pitch"],
        params={
            **_shooting_terms["kick_foot_strike_pitch"].params,
            "reference_relative": _use_foot_strike_pitch_reference_relative,
        },
        params_per_skill=(
            {"reference_relative": _use_foot_strike_pitch_reference_relative_per_skill}
            if _use_foot_strike_pitch_reference_relative_per_skill is not None
            else None
        ),
    ),
}

# Route the 6 kick-mode motion-tracking terms through their kick_scale_wrappers equivalents, so
# the new per-skill motion_tracking_reward_scale (utils/kick_reward_scales.py) applies -- these 6
# funcs are otherwise SHARED with standalone WBT training (managers/reward/terms/wbt.py), so the
# scale can't be applied by editing them directly without also rescaling WBT-only experiments.
# Swaps `func` AND injects swing_tracking_sigma_multiplier (above) into `params` -- weight and the
# EXISTING params (including any sigma sharpening from _sharpen_orientation_tracking above, which
# MUST run first so its sigma edits survive into the swapped cfg) are otherwise untouched. Terms
# not in the map (action_rate_l2, limits_dof_pos, undesired_contacts -- also part of
# g1_29dof_wbt_fast_sac_reward.terms, but not "motion tracking") pass through unchanged.
_MOTION_TRACKING_SCALED_FUNC = {
    "motion_global_ref_position_error_exp": (
        "holosoma.managers.reward.terms.kick_scale_wrappers:motion_global_ref_position_error_exp_scaled"
    ),
    "motion_global_ref_orientation_error_exp": (
        "holosoma.managers.reward.terms.kick_scale_wrappers:motion_global_ref_orientation_error_exp_scaled"
    ),
    "motion_relative_body_position_error_exp": (
        "holosoma.managers.reward.terms.kick_scale_wrappers:motion_relative_body_position_error_exp_scaled"
    ),
    "motion_relative_body_orientation_error_exp": (
        "holosoma.managers.reward.terms.kick_scale_wrappers:motion_relative_body_orientation_error_exp_scaled"
    ),
    "motion_global_body_lin_vel": (
        "holosoma.managers.reward.terms.kick_scale_wrappers:motion_global_body_lin_vel_scaled"
    ),
    "motion_global_body_ang_vel": (
        "holosoma.managers.reward.terms.kick_scale_wrappers:motion_global_body_ang_vel_scaled"
    ),
    "motion_global_feet_lin_vel": (
        "holosoma.managers.reward.terms.kick_scale_wrappers:motion_global_feet_lin_vel_scaled"
    ),
}


def _scale_kick_motion_tracking(terms: dict[str, RewardTermCfg]) -> dict[str, RewardTermCfg]:
    out = dict(terms)
    for name, scaled_func in _MOTION_TRACKING_SCALED_FUNC.items():
        if name in out:
            out[name] = replace(
                out[name],
                func=scaled_func,
                params={
                    **out[name].params,
                    "swing_tracking_sigma_multiplier": _swing_tracking_sigma_multiplier,
                    "post_flip_reward_decay_steps": _post_flip_reward_decay_steps,
                    "pre_kick_reward_ramp_steps": _pre_kick_reward_ramp_steps,
                },
                params_per_skill=(
                    {
                        **(
                            {"post_flip_reward_decay_steps": _post_flip_reward_decay_steps_per_skill}
                            if _post_flip_reward_decay_steps_per_skill is not None
                            else {}
                        ),
                        **(
                            {"pre_kick_reward_ramp_steps": _pre_kick_reward_ramp_steps_per_skill}
                            if _pre_kick_reward_ramp_steps_per_skill is not None
                            else {}
                        ),
                    }
                    or None
                ),
            )
    return out


def _apply_post_flip_reward_decay(
    terms: dict[str, RewardTermCfg], decay_steps: float
) -> dict[str, RewardTermCfg]:
    """FIX 4 (2026-08-12): must run AFTER `_tagged(..., "kick")` has already stamped task_mode=
    "kick" onto every WBT term in the dict (including the 7 motion-tracking ones) -- this
    overrides JUST those 7 (by name, same key set as _MOTION_TRACKING_SCALED_FUNC) back to
    task_mode=None when decay_steps>0, so RewardManager's own masking stops zeroing post-flip
    envs and _post_flip_tracking_decay_multiplier (kick_scale_wrappers.py) becomes the sole thing
    deciding their contribution instead.

    decay_steps<=0.0 (default) is a true no-op: `name in _MOTION_TRACKING_SCALED_FUNC` entries are
    left exactly as `_tagged` set them (task_mode="kick", unchanged), and every non-tracking term
    in `terms` is returned untouched either way -- bit-identical to calling `_tagged(..., "kick")`
    alone."""
    if decay_steps <= 0.0:
        return terms
    out = dict(terms)
    for name in _MOTION_TRACKING_SCALED_FUNC:
        if name in out:
            out[name] = replace(out[name], task_mode=None)
    return out


# Per-term reward-WEIGHT overrides, loaded unconditionally from configs/kicking_motion_reward_
# tuning.yaml (config_types/reward_tuning.py -- see that file's own docstring and the yaml's own
# header comment for the full design). One level more granular than the per-skill, per-category
# scales above (motion_tracking_reward_scale etc.): this replaces each individual term's static
# RewardTermCfg.weight directly, rather than multiplying a whole category. {} (file missing) is a
# safe no-op; every value currently in the shipped yaml matches today's hardcoded defaults exactly,
# so an unedited file is bit-identical to before this mechanism existed.
_reward_weight_overrides = load_reward_weight_overrides()

# Sigma overrides, same yaml, same unconditional/no-op-by-default loading -- see this module's own
# docstring ("Every individual kick-mode reward term's WEIGHT...") and config_types/reward_tuning.py
# for the full design. Unlike weight (every term has one), only SOME terms have a sigma parameter
# at all -- _apply_reward_sigma_overrides below checks for that explicitly, so overriding sigma on
# a term that doesn't use a sigma-shaped kernel fails fast at config-import time instead of either
# silently doing nothing or crashing deep in the training loop with an unexpected-kwarg TypeError.
_reward_sigma_overrides = load_reward_sigma_overrides()


def _apply_reward_weight_overrides(
    terms: dict[str, RewardTermCfg], overrides: dict[str, float]
) -> dict[str, RewardTermCfg]:
    out = dict(terms)
    for name, weight in overrides.items():
        if name not in out:
            raise ValueError(
                f"configs/kicking_motion_reward_tuning.yaml overrides unknown reward term {name!r} "
                f"-- not one of this manager's {len(out)} registered terms. Check for a typo, or a "
                f"term that was renamed/removed."
            )
        out[name] = replace(out[name], weight=weight)
    return out


# "Simultaneous per-skill task configs" (2026-08-15, user-requested): N motion skills in one
# training run, each genuinely wanting a DIFFERENT task_config -- e.g. skill 1 mature under
# task_config_stageC1 (full shooting reward, kick_alive on) while skill 2 trains under
# task_config_stageB (pure motion imitation: shooting_reward_scale=0, kick_alive=0) at the same
# time, in the same process. holosoma/__init__.py's HOLOSOMA_TASK_CONFIG derivation resolves only
# ONE global config (every non-reward-weight field: deadzones, mechanism flags, l2sp_weight,
# kick_gamma, ...) -- see that module's own 2026-08-15 docstring update for why per-skill
# divergence THERE isn't supported (no per-env plumbing exists for those ~9 mechanism flags and 9
# deadzones without deeper surgery). Reward WEIGHTS are different: RewardManager.compute() already
# multiplies in a per-env task_mode_mask at exactly the point a per-env weight would go (see
# manager.py), and every one of this project's own SkillConfig category scales
# (motion_tracking_reward_scale etc., utils/kick_reward_scales.py) already proves the
# skill_id-gather pattern works at reward-compute time. This resolves the 5 reward-tuning
# categories' PER-TERM weights (motion_tracking_reward/shooting_reward/
# kick_recovery_posture_reward/kick_safety_reward/kick_alive_reward) the same way, per skill,
# reusing config_types/reward_tuning.py's existing per-file loader unchanged -- no new yaml
# parsing logic, just N calls to it instead of one.
#
# Each skill's OWN `task_config:` field (SkillConfig.task_config, already parsed by
# config_types/multi_skill.py) is the source -- independent of whatever HOLOSOMA_TASK_CONFIG
# resolved to for globals. A skill with no `task_config:` at all (None) contributes no overrides
# of its own, i.e. it simply inherits whatever the GLOBAL config says for every term, same meaning
# "no override" already has for the legacy single-file case.
#
# Trivial/no-op detection: if 0 or 1 skills are configured, or every configured skill resolves to
# the SAME task_config path (including today's real skills1.yaml: one skill, one task_config),
# this returns `terms` completely UNCHANGED -- no RewardTermCfg gets a weight_per_skill, no new
# tensor gather happens in RewardManager.compute(), byte-identical to before this mechanism
# existed. Only genuine divergence (2+ distinct resolved paths) does any extra work at all.
# (_skill_task_config_paths itself now lives much earlier in this module -- see its own comment
# there, right after _multi_skill_cfg_for_contact_penalty -- several PARAM-level per-skill
# mechanisms below need it far sooner than this WEIGHT-level one does.)
def _apply_per_skill_reward_weight_overrides(
    terms: dict[str, RewardTermCfg], skill_task_config_paths: list[Path | None] | None
) -> dict[str, RewardTermCfg]:
    if skill_task_config_paths is None or len(set(skill_task_config_paths)) <= 1:
        return terms  # 0/1 skills, or every skill agrees -- exact no-op, see this section's docstring

    per_skill_overrides = load_per_skill_reward_weight_overrides(skill_task_config_paths)
    touched = {name for overrides in per_skill_overrides for name in overrides}

    out = dict(terms)
    for name in touched:
        if name not in out:
            raise ValueError(
                f"a per-skill task_config overrides unknown reward term {name!r} -- not one of "
                f"this manager's {len(out)} registered terms. Check for a typo, or a term that was "
                f"renamed/removed. (Skill task_config files: "
                f"{[str(p) for p in skill_task_config_paths]})"
            )
        base_weight = out[name].weight  # already resolved through the GLOBAL override, if any
        per_skill_weights = [overrides.get(name, base_weight) for overrides in per_skill_overrides]
        if len(set(per_skill_weights)) > 1:
            # Genuine per-skill divergence for this specific term -- install the gather table.
            # `weight` stays populated with skill 0's value: a representative placeholder for any
            # code that reads `.weight` directly without per-skill awareness (see
            # RewardTermCfg.weight_per_skill's own docstring) -- RewardManager.compute() itself
            # ignores it in favor of the table whenever weight_per_skill is set.
            out[name] = replace(out[name], weight=per_skill_weights[0], weight_per_skill=per_skill_weights)
        elif per_skill_weights[0] != base_weight:
            # Every skill agrees on a value, but it differs from the (global-resolved) base -- a
            # plain scalar override suffices, no per-env gather needed for this term.
            out[name] = replace(out[name], weight=per_skill_weights[0])
    return out


def _apply_reward_sigma_overrides(
    terms: dict[str, RewardTermCfg], overrides: dict[str, float]
) -> dict[str, RewardTermCfg]:
    out = dict(terms)
    for name, sigma in overrides.items():
        if name not in out:
            raise ValueError(
                f"configs/kicking_motion_reward_tuning.yaml overrides sigma for unknown reward term "
                f"{name!r} -- not one of this manager's {len(out)} registered terms. Check for a typo, "
                f"or a term that was renamed/removed."
            )
        existing_params = out[name].params or {}
        if "sigma" not in existing_params:
            raise ValueError(
                f"configs/kicking_motion_reward_tuning.yaml overrides sigma for term {name!r}, but "
                f"that term has no 'sigma' parameter to override (its params are "
                f"{sorted(existing_params.keys())}) -- check for a typo, or a term that doesn't use "
                f"a sigma-shaped kernel. Only put a term under a category's '_sigma' key if it "
                f"genuinely accepts one."
            )
        out[name] = replace(out[name], params={**existing_params, "sigma": sigma})
    return out


_g1_29dof_unified_reward_terms = {
    **_tagged(g1_29dof_loco_fast_sac.terms, "locomotion"),
    **_apply_post_flip_reward_decay(
        _tagged(_scale_kick_motion_tracking(_sharpen_orientation_tracking(g1_29dof_wbt_fast_sac_reward.terms)), "kick"),
        # "Simultaneous per-skill task configs": use the ANY-skill-active signal, not the raw
        # global scalar -- see _post_flip_reward_decay_steps_for_registration's own comment.
        _post_flip_reward_decay_steps_for_registration,
    ),
    **_penalty_stance_asymmetry_term,
    **_penalty_yaw_drift_term,
    **_penalty_stand_height_term,
    **_penalty_stand_orientation_term,
    **_penalty_stand_feet_width_term,
    **_penalty_stand_knee_width_term,
    **_kick_recovery_standing_terms,
    **_shooting_terms,
    **_kick_safety_terms,
    **_kick_alive_term,
    **_kick_balance_potential_term,
    **_motion_strike_dof_pos_term,
    **_kick_swing_stability_terms,
    **_p0_regularization_terms,
    # Must come AFTER the locomotion spread above -- overrides its "alive" entry when
    # post_flip_alive_scale != 1.0 (empty dict, i.e. no-op, at the 1.0 default).
    **_alive_term_override,
}

_g1_29dof_unified_reward_terms_tuned = _apply_reward_sigma_overrides(
    _apply_per_skill_reward_weight_overrides(
        _apply_reward_weight_overrides(_g1_29dof_unified_reward_terms, _reward_weight_overrides),
        _skill_task_config_paths,
    ),
    _reward_sigma_overrides,
)

g1_29dof_unified_reward = RewardManagerCfg(
    only_positive_rewards=False,
    terms=_g1_29dof_unified_reward_terms_tuned,
)

__all__ = ["g1_29dof_unified_reward"]
