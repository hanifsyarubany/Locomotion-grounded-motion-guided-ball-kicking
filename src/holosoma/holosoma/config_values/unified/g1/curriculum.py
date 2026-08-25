"""Unified locomotion + ball-kicking curriculum preset for the G1 robot.

Locomotion's (g1_29dof_curriculum_fast_sac) and WBT's (g1_29dof_wbt_curriculum) curriculum
configs both use the identical AverageEpisodeLengthTracker term; locomotion's fast_sac variant
additionally has PenaltyCurriculum (which WBT doesn't use). Locomotion's is the superset, so its
structure is reused — UnifiedManager only ever marks locomotion-mode episode ends as "pending" for
the tracker (see _reset_buffers_callback), so kick episodes never dilute either curriculum term's
signal — but PenaltyCurriculum's ramp thresholds are recalibrated here for unified specifically
(see below), so it is no longer a bare re-export of g1_29dof_curriculum_fast_sac.

Recalibration: unified adds five standing-shaping terms on top of the stock locomotion reward set
(penalty_stance_asymmetry, penalty_yaw_drift, penalty_stand_height, penalty_stand_orientation,
penalty_stand_feet_width — all tagged "penalty_curriculum", see reward.py), taking the combined
penalty_curriculum-tagged weight budget from |sum|=28.5 (stock loco baseline) to |sum|=178.5 — a
6.3x increase. g1_29dof_curriculum_fast_sac's ramp thresholds (level_up=750, level_down=150,
min_scale=0.5) were tuned against the 28.5 budget and never adjusted for the larger one.

Measured failure mode this causes (v7 Stage-A-from-scratch run, kick_probability=0): episode
length climbed cleanly to 836 by ~6k steps (checkpoint 6000: stands perfectly for a full 10s
zero-velocity MuJoCo hold, max lean 1.2deg) — comfortably past level_up_threshold=750, so the
curriculum ramped penalty_scale to 1.0 by ~10k steps. The policy, having only learned basic
viability (not yet simultaneous height+symmetry+yaw+orientation+width control), could not
satisfy the full 178.5-magnitude budget: episode length crashed 836 -> ~485-525 and stayed there
through 23k+ steps (checkpoint 16000: falls at 3.36s, lean spikes to 11.2deg beforehand) —
"can't even stand still", a genuine regression, not just remaining immaturity. The safety net
never engaged: 525 stayed well above level_down_threshold=150 (only ~3s of survival), so the
curriculum had no signal that this WAS a regression, and even if it had triggered, min_scale=0.5
only floors the budget at 89 (178.5*0.5) — still ~3x the 28.5 this schedule was built for.

Fix: level_down_threshold raised 150->600 (comfortably below level_up=750 to avoid a tight
oscillating band, but above the ~485-525 plateau actually observed, so THIS regression would
trigger protective easing) and min_scale lowered 0.5->0.15 (178.5*0.15=~27, landing close to the
original 28.5 budget the ramp dynamics were tuned against, so the floor gives genuine relief
rather than a still-6x-larger one). level_up_threshold and degree are unchanged — the ramp-UP
side isn't implicated (836 was real, healthy performance), only the down-side protection was
miscalibrated for the larger budget.

FIX 1 (2026-08-12), opt-in via MultiSkillConfig.penalty_curriculum_enabled, default True (exact
no-op, current ramping behavior unchanged): a SEPARATE, later-discovered problem from the one
above, specific to kick_recovery_locomotion_flip_enabled. average_episode_length (this
curriculum's own input) is fed ONLY by the genuinely-locomotion-partitioned population
(_reset_buffers_callback tags kick episode-ends out of it unconditionally) -- but the SCALE it
computes is applied to every penalty_curriculum-tagged term, which since the flip now includes
_penalty_stand_* for any post-flip kick-partitioned env too (see reward.py). Those envs are
therefore scaled by a signal they cannot influence. Measured live on a matched FLIP-vs-NOFLIP
checkpoint pair (2026-08-12 diagnostic): Env/penalty_scale led Env/kick_alive_frac at lag+10 with
r=-0.753 (vs r=-0.122 at lag 0) -- the curriculum ramping up measurably preceded, not just
correlated with, kick survival collapsing. Setting penalty_curriculum_enabled=False makes
PenaltyCurriculum.setup()/.reset() both no-ops (see that class's own early-return on
`not self.enabled`), so every penalty_curriculum-tagged term's weight stays at its full,
un-scaled yaml value from the start -- appropriate for resuming an already-mature checkpoint
(these runs all warm-start from a converged Stage C1 checkpoint; the ramp-from-easy rationale this
curriculum exists for doesn't apply to a policy that's already stood up cleanly before).
"""

from dataclasses import replace

from holosoma.config_types.multi_skill import load_multi_skill_config, multi_skill_mode_enabled
from holosoma.config_types.simulator import load_ball_config
from holosoma.config_values.loco.g1.curriculum import g1_29dof_curriculum_fast_sac

# Re-resolved per import rather than sharing a cross-file singleton -- load_multi_skill_config/
# load_ball_config are cheap, pure, idempotent parses (same convention as command.py/
# termination.py/reward.py's own dual-path resolution blocks).
_multi_skill_cfg = load_multi_skill_config() if multi_skill_mode_enabled() else None
_penalty_curriculum_enabled = (
    _multi_skill_cfg.penalty_curriculum_enabled
    if _multi_skill_cfg is not None
    else load_ball_config().penalty_curriculum_enabled
)

_penalty_curriculum_recalibrated = replace(
    g1_29dof_curriculum_fast_sac.setup_terms["penalty_curriculum"],
    params={
        **g1_29dof_curriculum_fast_sac.setup_terms["penalty_curriculum"].params,
        "level_down_threshold": 600.0,  # was 150.0
        "min_scale": 0.15,  # was 0.5
        "enabled": _penalty_curriculum_enabled,  # True (default) = unchanged; see FIX 1 above
    },
)

g1_29dof_unified_curriculum = replace(
    g1_29dof_curriculum_fast_sac,
    setup_terms={
        **g1_29dof_curriculum_fast_sac.setup_terms,
        "penalty_curriculum": _penalty_curriculum_recalibrated,
    },
)

__all__ = ["g1_29dof_unified_curriculum"]
