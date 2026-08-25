"""Unified locomotion + ball-kicking observation preset for the G1 robot.

Merges locomotion's (g1_29dof_loco_single_wolinvel) and WBT's (g1_29dof_wbt_observation) terms
into one observation space: every term is tagged with the task_mode it belongs to, so
ObservationManager zeroes it (never omits it — the concatenated width stays constant regardless
of which mode a given env is in) for envs not currently running that task.

Several term names collide between the two sources (base_ang_vel, dof_pos, dof_vel, actions,
base_lin_vel) — same concept, but different function module (terms.locomotion vs terms.wbt) and
different noise/scale tuning per source. Each is kept as its own distinct term, prefixed by
source, rather than silently overwriting one with the other via a naive dict union.

A new task_mode_onehot term (no task_mode tag — always active) tells the policy which mode it's
in, so a single network can condition its behavior on it instead of just inferring from which
blocks are non-zero.
"""

from dataclasses import replace

from holosoma.config_types.multi_skill import load_multi_skill_config, multi_skill_mode_enabled
from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg
from holosoma.config_types.simulator import load_ball_config
from holosoma.config_values.loco.g1.observation import g1_29dof_loco_single_wolinvel
from holosoma.config_values.wbt.g1.observation import g1_29dof_wbt_observation

# Ball perception noise/latency source, 2026-07-24: read from whichever config is actually
# active for this process -- MultiSkillConfig's shared (not per-skill) ball_obs_* fields in
# N-skill mode, BallConfig's matching observation_* fields otherwise. Same
# multi_skill_mode_enabled()-gated, read-once-at-import-time pattern config_values/unified/g1/
# command.py already uses for _multi_skill_cfg, and the same "N-skill mode is opt-in, legacy
# path untouched" discipline HOLOSOMA_SKILLS_CONFIG's own docstring documents (config_types/
# multi_skill.py) -- an unset HOLOSOMA_SKILLS_CONFIG reproduces the pre-2026-07-24 hardcoded
# values (0.05 / 0.03 / 0 / 3) exactly, since BallConfig's own defaults match them.
if multi_skill_mode_enabled():
    _ball_obs_source = load_multi_skill_config()
    _ball_obs_noise = _ball_obs_source.ball_obs_noise
    _ball_obs_noise_range_coefficient = _ball_obs_source.ball_obs_noise_range_coefficient
    _ball_obs_delay_steps_min = _ball_obs_source.ball_obs_delay_steps_min
    _ball_obs_delay_steps_max = _ball_obs_source.ball_obs_delay_steps_max
    _ball_obs_hold_steps_min = _ball_obs_source.ball_obs_hold_steps_min
    _ball_obs_hold_steps_max = _ball_obs_source.ball_obs_hold_steps_max
    _ball_obs_stale_probability = _ball_obs_source.ball_obs_stale_probability
    # 2026-08-18 observation-side handoff-discontinuity fixes 1/2/4 (fix 3 lives on the env, not
    # here -- it ramps the manager's own mask). Same read-once-at-import-time pattern as the ball
    # perception fields above; all three are MultiSkillConfig-only with no legacy BallConfig
    # counterpart, so the else-branch below pins them to their exact-no-op defaults.
    _obs_target_pos_distance_scale = _ball_obs_source.obs_target_pos_distance_scale
    _obs_untag_shared_proprioception = _ball_obs_source.obs_untag_shared_proprioception
    _obs_ball_always_visible = _ball_obs_source.obs_ball_always_visible
else:
    _ball_cfg = load_ball_config()
    _ball_obs_noise = _ball_cfg.observation_noise
    _ball_obs_noise_range_coefficient = _ball_cfg.observation_noise_range_coefficient
    _ball_obs_delay_steps_min = _ball_cfg.observation_delay_steps_min
    _ball_obs_delay_steps_max = _ball_cfg.observation_delay_steps_max
    _ball_obs_hold_steps_min = _ball_cfg.observation_hold_steps_min
    _ball_obs_hold_steps_max = _ball_cfg.observation_hold_steps_max
    _ball_obs_stale_probability = _ball_cfg.observation_stale_probability
    _obs_target_pos_distance_scale = 0.0
    _obs_untag_shared_proprioception = False
    _obs_ball_always_visible = False


def _tagged(terms: dict[str, ObsTermCfg], task_mode: str, prefix: str) -> dict[str, ObsTermCfg]:
    return {f"{prefix}_{name}": replace(cfg, task_mode=task_mode) for name, cfg in terms.items()}


def _tagged_except(
    terms: dict[str, ObsTermCfg], task_mode: str, prefix: str, untagged: frozenset[str]
) -> dict[str, ObsTermCfg]:
    """``_tagged``, but leaves the named terms UNTAGGED (``task_mode=None``) so they stay live in
    every task mode -- FIX 2 of the 2026-08-18 observation work
    (``MultiSkillConfig.obs_untag_shared_proprioception``).

    Width-preserving by construction: the same keys are produced either way, each still carrying
    its own dims. Masking only ever zeroed these slots, never omitted them (see
    ObservationManager.compute_group step 4b's own comment), so removing the tag changes what a
    slot CONTAINS during the other mode and never where any slot sits in the concatenated vector.
    That is what keeps existing checkpoints loadable.
    """
    return {
        f"{prefix}_{name}": replace(cfg, task_mode=None if name in untagged else task_mode)
        for name, cfg in terms.items()
    }


# The terms that exist in BOTH source groups and are therefore fed to the policy twice -- once as
# loco_*, once as kick_*. Verified 2026-08-18 that every one of these pairs is byte-identical in
# implementation (e.g. terms.locomotion:dof_vel and terms.wbt:dof_vel are both literally
# `return env.simulator.dof_vel`; dof_pos is `env.simulator.dof_pos - env.default_dof_pos` on both
# sides; base_ang_vel is `get_base_ang_vel(env)`; actions is `env.action_manager.action`), so
# untagging them duplicates a VIEW, never a computation, and loses no information.
#
# base_lin_vel appears only in the critic pair (the actor group is the "wolinvel" preset and has no
# locomotion base_lin_vel term), which is why this is derived per-group by intersection below
# rather than hardcoded as one list -- hardcoding it would silently untag a nonexistent actor term
# today and silently MISS a newly-shared term if either source preset gains one later.
def _shared_term_names(loco_terms: dict[str, ObsTermCfg], kick_terms: dict[str, ObsTermCfg]) -> frozenset[str]:
    return frozenset(loco_terms) & frozenset(kick_terms) if _obs_untag_shared_proprioception else frozenset()


_task_mode_onehot_term = {
    "task_mode_onehot": ObsTermCfg(
        func="holosoma.managers.observation.terms.unified:task_mode_onehot",
        scale=1.0,
        noise=0.0,
    ),
}

# Shooting-task observations (RoboNaldo-style ego-centric ball + commanded target, both in the
# robot's heading frame — see each term's docstring in managers/observation/terms/unified.py).
# Tagged task_mode="kick" so they're zeroed (width preserved) during locomotion episodes, same
# masking as every other kick term. Ball gets modest positional noise in the actor group
# (perception won't be mocap-clean at deployment); the target is a command, so it's noiseless.
# NOTE: adding these changes the actor/critic obs widths (actor 256 -> 261) — checkpoints trained
# without them are NOT resumable; retrain from scratch via the staged bootstrap protocol
# (configs/ball.yaml's shooting_reward_scale comment / experiment.py's example block).
#
# Ball perception noise model, 2026-07-24: extended from flat noise to also cover
# range-dependent error and pipeline latency -- the two gaps flagged when comparing this
# simulated model against a real depth-camera + LiDAR + Kalman-filter fusion stack. Values come
# from _ball_obs_noise/_ball_obs_noise_range_coefficient/_ball_obs_delay_steps_{min,max} above
# (yaml-configurable via stageB_and_C.yaml's top-level ball_obs_* keys in N-skill mode, or
# configs/ball.yaml's observation_* keys in legacy mode) -- deliberately GLOBAL, not per-skill,
# unlike ball spawn/target/reward fields: this models the robot's perception HARDWARE, which
# doesn't change depending on which kick clip is running.
#   noise (default 0.05, flat 5cm) + noise_range_coefficient (default 0.03, additional +3% of
#   distance): a real depth sensor's error grows with range, not flat. Both are reasoned
#   starting points (3%/range is a commonly-cited stereo-depth error ballpark), not measured
#   against this project's own real sensor -- retune via the yaml once real perception-vs-mocap
#   error data exists.
#   delay_step_range (default (0, 3)): 0-3 control steps (0-60ms at this experiment's dt=0.02,
#   see config_values/unified/g1/reward.py's own "grace_steps=50.0 (1.0s at dt=0.02)" for the
#   same dt reference) of per-episode-fixed latency, modeling camera/LiDAR capture + KF fusion
#   lag. Mirrors managers/action/terms/joint_control.py's action-delay queue mechanism exactly
#   (same rolling-buffer-plus-per-env-index shape), applied to this observation instead of an
#   action -- see ObservationManager._apply_delay's docstring for the full mechanism.
#   hold_step_range (default (0, 0) = off, 2026-08-20): zero-order hold applied on TOP of the delay
#   above -- models the fusion pipeline's own UPDATE RATE being slower than the 50Hz control loop,
#   distinct from delay_step_range's transport latency on a pipeline that still updates every tick.
#   A per-episode period is drawn uniformly from this range (min==max for a fixed rate). E.g. the
#   real stack tops out at 25Hz -> (2, 2) (round(50/25)): the value only refreshes every other
#   control tick instead of every tick; a wider range (e.g. (2, 3)) also models the real rate
#   jittering between ~17-25Hz episode to episode. See ObservationManager._apply_hold's docstring.
#   stale_probability (default 0.0, 2026-08-23): per-CONTROL-TICK (re-rolled every step, unlike
#   hold's per-episode-fixed period) chance the reading reuses the PREVIOUS tick's already-
#   processed value instead of this tick's -- models a single dropped/repeated sensor frame,
#   transient and self-correcting (a stale streak keeps returning the SAME last-real frame, never
#   drifting further back). Cross-checked against RoboNaldo's own source
#   (mdp/commands.py::apply_lidar_stale_ball_pos_b / lidar_stale_probability): they stage 0.0
#   through Stage 1/2 and 0.01 once Stage 3 (moving-ball) begins -- see
#   MultiSkillConfig.ball_obs_stale_probability's own docstring for the full comparison. Distinct
#   from ball_static_obs_probability below (a full-episode freeze to an INDEPENDENTLY-DRAWN
#   value, modeling a broken sensor, not a dropped packet). See ObservationManager._apply_stale.
# All four gated on the SAME group.enable_noise flag flat noise already uses, so the critic group
# (enable_noise=False) is completely unaffected by any of them -- it keeps reading the
# instantaneous, noise-free ground-truth ball position, exactly as before this change.
# NOT modeled: SUSTAINED occlusion/dropout (the ball becoming unobservable for a stretch, e.g.
# behind the robot's own swinging leg) -- deliberately deferred, since a real fix needs the policy
# to have some representation of "no reading right now" (e.g. a validity flag), which is itself a
# new observation dimension and forces the same NOT-resumable retrain noted above a second time.
# stale_probability above covers only the much narrower single-dropped-frame case, which needs no
# validity flag since the policy is simply fed a slightly-old real reading, not a missing one.
#
# 2026-08-18, FIX 4 (``obs_ball_always_visible``): when True, `_shooting_task_mode` below is None,
# leaving both terms ungated so they read live during locomotion too. The ball is PHYSICALLY
# PRESENT the whole time -- zeroing it is a side effect of gating whole groups by task mode, not a
# modelling decision -- so this removes their share of the handoff discontinuity at the source
# rather than smoothing it, and (the actual point) lets the policy learn to walk TOWARD the ball,
# which is the Stage D goal it currently cannot even represent. Note this is the SAME continuity
# argument ball_pos_b's own docstring already makes, two paragraphs down, for keeping the ball live
# across the Stage B->C boundary; this applies it to the locomotion->kick boundary as well.
# Width-preserving (masking zeroed, never omitted), so checkpoints still warm-start.
#
# 2026-08-18, FIX 1 (``obs_target_pos_distance_scale``): kick_target_pos_b is passed the compression
# scale as a term param. At the 0.0 default target_pos_b returns the raw offset via its identical
# pre-existing path, so this is an exact no-op until set.
_shooting_task_mode = None if _obs_ball_always_visible else "kick"

_shooting_obs_terms = {
    "kick_ball_pos_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.unified:ball_pos_b",
        scale=1.0,
        noise=_ball_obs_noise,
        noise_range_coefficient=_ball_obs_noise_range_coefficient,
        delay_step_range=(_ball_obs_delay_steps_min, _ball_obs_delay_steps_max),
        hold_step_range=(_ball_obs_hold_steps_min, _ball_obs_hold_steps_max),
        stale_probability=_ball_obs_stale_probability,
        task_mode=_shooting_task_mode,
    ),
    # 2026-08-22, azimuth-aim refactor: kick_aim_command replaces target_pos_b as this slot's
    # function. Falls through to target_pos_b verbatim for any skill that isn't kick_aim_enabled
    # (bit-identical for those envs) -- see kick_aim_command's own docstring for the per-env
    # selection, the mixed-mode caveat, and why this reuses _shooting_task_mode rather than an
    # unconditional task_mode=None. Key stays "kick_target_pos_b" (not renamed) so a checkpoint's
    # saved ObsTermCfg still matches by name for _detect_shifted_obs_terms's warm-start comparison
    # -- only the underlying function and its behavior changed.
    "kick_target_pos_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.unified:kick_aim_command",
        scale=1.0,
        noise=0.0,
        task_mode=_shooting_task_mode,
        params={"distance_scale": _obs_target_pos_distance_scale},
    ),
}

# Critic-facing variant, 2026-07-24: kick_ball_pos_b swaps to ball_pos_b_ground_truth (identical
# transform, no perception bias -- see that function's own docstring) so the critic's privileged
# ground truth is genuinely clean, not just noise/delay-free. Before this existed, critic_obs
# reused _shooting_obs_terms verbatim, so BallConfig.observation_bias (a real, nonzero default in
# most yamls) leaked into the critic despite noise/delay already being correctly excluded via
# group.enable_noise=False -- bias couldn't use that same gate since it's ball-specific per-skill
# state set on env, not a generic ObsTermCfg field the manager applies uniformly. noise/
# noise_range_coefficient/delay_step_range are explicitly zeroed here too (redundant with
# enable_noise=False, which already no-ops them for this group -- kept anyway so this dict reads
# as unambiguously "no perception artifacts" on its own, without relying on a reader also
# checking the group's own flag).
_shooting_obs_terms_critic = {
    **_shooting_obs_terms,
    "kick_ball_pos_b": replace(
        _shooting_obs_terms["kick_ball_pos_b"],
        func="holosoma.managers.observation.terms.unified:ball_pos_b_ground_truth",
        noise=0.0,
        noise_range_coefficient=0.0,
        delay_step_range=None,
        stale_probability=0.0,
    ),
}

_loco_actor_terms = g1_29dof_loco_single_wolinvel.groups["actor_obs"].terms
_kick_actor_terms = g1_29dof_wbt_observation.groups["actor_obs"].terms
_loco_critic_terms = g1_29dof_loco_single_wolinvel.groups["critic_obs"].terms
_kick_critic_terms = g1_29dof_wbt_observation.groups["critic_obs"].terms

# Resolved per group: actor shares 4 terms (base_ang_vel/dof_pos/dof_vel/actions), critic shares
# those plus base_lin_vel. Empty frozenset when FIX 2 is off, which makes _tagged_except behave
# exactly like _tagged. See _shared_term_names above.
_shared_actor = _shared_term_names(_loco_actor_terms, _kick_actor_terms)
_shared_critic = _shared_term_names(_loco_critic_terms, _kick_critic_terms)

g1_29dof_unified_observation = ObservationManagerCfg(
    groups={
        "actor_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=True,
            history_length=1,
            terms={
                **_tagged_except(_loco_actor_terms, "locomotion", "loco", _shared_actor),
                **_tagged_except(_kick_actor_terms, "kick", "kick", _shared_actor),
                **_task_mode_onehot_term,
                **_shooting_obs_terms,
            },
        ),
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms={
                **_tagged_except(_loco_critic_terms, "locomotion", "loco", _shared_critic),
                **_tagged_except(_kick_critic_terms, "kick", "kick", _shared_critic),
                **_task_mode_onehot_term,
                **_shooting_obs_terms_critic,
            },
        ),
    },
)

__all__ = ["g1_29dof_unified_observation"]
