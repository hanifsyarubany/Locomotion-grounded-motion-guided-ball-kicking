"""Unified locomotion + ball-kicking command preset for the G1 robot.

Merges locomotion's (g1_29dof_command: LocomotionGait + LocomotionCommand) and the ball-kick
experiment's (g1_29dof_ball_kick_command: MotionCommand) command terms into one manager config.

reset_terms are tagged with task_mode so CommandManager.reset() only calls each term's reset()
with the subset of env_ids currently in that mode (see managers/command/manager.py) — e.g.
MotionCommand.reset() (which teleports the robot/ball to the kick clip's start) is never called
for envs currently running locomotion-mode. setup_terms/step_terms are left untagged: those
stages receive no env_ids at all (see CommandManager.step()), so a task_mode tag would have no
effect there — LocomotionCommand/LocomotionGait's step() only mutates their own buffers, which
are already correctly hidden from kick-mode envs by the observation/reward masking, so leaving
them running unconditionally is inert; MotionCommand.step() has its own internal task_mode gate
(see managers/command/terms/wbt.py) since it has real physical side effects (ball teleport on
clip-end) that step_terms' lack of env_ids can't filter generically.

kick_probability is read from configs/skill_mix.yaml (see load_skill_mix_config) and threaded
through as a command-manager-level param, the same way locomotion_command_resampling_time already
is — UnifiedManager reads it via self.command_manager.command_cfg.kick_probability.
"""

from dataclasses import replace

from holosoma.config_types.command import CommandManagerCfg, CommandTermCfg, load_skill_mix_config
from holosoma.config_types.multi_skill import load_multi_skill_config, multi_skill_mode_enabled
from holosoma.config_types.reward_tuning import resolve_per_skill_param
from holosoma.config_types.simulator import load_ball_config
from holosoma.config_types.task_config_paths import resolve_task_config_path
from holosoma.config_values.loco.g1.command import g1_29dof_command
from holosoma.config_values.wbt.g1.command import g1_29dof_ball_kick_command, motion_config_ball_kick

# Unified-only motion config: every kick episode starts at the clip's FIRST frame, not a random
# mid-clip phase. Stock WBT's random-phase + adaptive-failure-bin sampling is a curriculum for a
# policy that gets 100% kick experience; in the unified mix, kick-mode envs are a small minority,
# and teleporting an undertrained policy into an extreme mid-kick pose (one leg extended) means it
# falls immediately — measured: EVERY kick episode terminated within 4-7 steps (the moment the
# bad_tracking grace period expired), so kick collected ~1% of transitions, never bootstrapped,
# and the wandb kick-video was an endless loop of 0.1s-long episodes that looked frozen.
#
# enable_default_pose_prepend=True is REQUIRED for deployability, and its absence was the root
# cause of "the kick is robust in training but falls immediately on trigger in sim2sim/deploy":
# without a prepend, the clip's frame 0 is the RAW motion's first frame -- a mid-windup pose
# measured 0.556 rad (32deg, knees near-straight, arms cocked) away from a standing robot. Training
# hides this completely because start_at_timestep_zero_prob=1.0 TELEPORTS the robot exactly onto
# frame 0 at kick start, so tracking begins with zero error. Deployment cannot teleport: the robot
# stands (~default pose) and, on trigger, is told to instantly track that 32deg-away frame 0 -> it
# lurches to ~29deg lean within 0.24s and topples in <1s (reproduced in MuJoCo on model_0251000).
# The prepend makes frame 0 the DEFAULT (standing) pose and interpolates default -> motion over
# default_pose_prepend_duration_s, so: (1) training's frame-0 teleport lands on the real standing
# pose, and (2) deployment's standing robot already matches frame 0 and is then eased into the kick
# by the same ramp -- training and deployment finally cover the same stand->windup->kick trajectory.
# Mirrors the append (recovery) tail (motion -> default -> hold) that is already enabled for the
# post-kick side; this adds the symmetric pre-kick side. (Requires a Stage C retrain: the clip
# structure and time_step indexing change, so an existing prepend-less checkpoint won't transfer.)
_multi_skill_cfg = load_multi_skill_config() if multi_skill_mode_enabled() else None

# OOD ball-spawn deployment-robustness training (2026-07-24, user proposal): read UNCONDITIONALLY
# regardless of N-skill mode (mirrors config_values/unified/g1/reward.py's own _ori_alpha
# resolution) -- available to single-skill configs too, not gated behind HOLOSOMA_SKILLS_CONFIG
# the way skill_ball_configs/motion_files above are. See MotionConfig.ood_spawn_probability's own
# docstring for the full rationale and design.
if _multi_skill_cfg is not None:
    _ood_spawn_probability = _multi_skill_cfg.ood_spawn_probability
    _ood_region_multiplier = _multi_skill_cfg.ood_region_multiplier
    _start_at_timestep_zero_prob = _multi_skill_cfg.start_at_timestep_zero_prob
    _rsi_scope_to_authored_clip = _multi_skill_cfg.rsi_scope_to_authored_clip
    _critical_frame_oversampling_prob = _multi_skill_cfg.critical_frame_oversampling_prob
    _critical_frame_sampling_window = _multi_skill_cfg.critical_frame_sampling_window
    _motion_head_velocity_smoothing_frames = _multi_skill_cfg.motion_head_velocity_smoothing_frames
    _kick_recovery_locomotion_flip_enabled = _multi_skill_cfg.kick_recovery_locomotion_flip_enabled
    # Locomotion->kick direction (the mirror of the flip above) -- MultiSkillConfig-only, no
    # legacy BallConfig counterpart (unlike every field above): the mechanism's entry-point search
    # is inherently scoped to a fixed per-env skill assignment (UnifiedManager._build_task_mode_
    # partition), which only exists in N-skill mode. Legacy single-skill configs always see the
    # off-defaults below, same as never setting these fields at all -- a deliberate scope choice,
    # not an oversight.
    _mid_episode_kick_entry_prob = _multi_skill_cfg.mid_episode_kick_entry_prob
    _mid_episode_kick_entry_min_steps = _multi_skill_cfg.mid_episode_kick_entry_min_steps
    _mid_episode_kick_entry_max_residual = _multi_skill_cfg.mid_episode_kick_entry_max_residual
    _pre_kick_decel_steps = _multi_skill_cfg.pre_kick_decel_steps
    _pre_kick_decel_target = _multi_skill_cfg.pre_kick_decel_target
    _pre_kick_fallback_timeout_steps = _multi_skill_cfg.pre_kick_fallback_timeout_steps
    _pre_kick_reward_ramp_steps = _multi_skill_cfg.pre_kick_reward_ramp_steps
    _pre_kick_termination_grace_steps = _multi_skill_cfg.pre_kick_termination_grace_steps
    _pre_kick_reference_blend_steps = _multi_skill_cfg.pre_kick_reference_blend_steps
    _mid_episode_kick_entry_ball_fixed = _multi_skill_cfg.mid_episode_kick_entry_ball_fixed
    # FIX 3 of the 2026-08-18 observation work -- the 4th smoothing sibling of the three ramps
    # above. Same MultiSkillConfig-only scoping as the rest of this block.
    _pre_kick_obs_ramp_steps = _multi_skill_cfg.pre_kick_obs_ramp_steps
    _post_flip_obs_ramp_steps = _multi_skill_cfg.post_flip_obs_ramp_steps
    # FIX 6 of the 2026-08-18 observation work -- the load-time blend, superseding the
    # normalizer-reset guard for whichever terms it covers. Same MultiSkillConfig-only scoping.
    _warm_start_obs_ramp_steps = _multi_skill_cfg.warm_start_obs_ramp_steps
    # 2026-08-22, azimuth-aim refactor. No per-skill table needed here (unlike the fields above) --
    # see MotionConfig.kick_aim_theta_ref_deg's own docstring for why these three stay global.
    _kick_aim_theta_ref_deg = _multi_skill_cfg.kick_aim_theta_ref_deg
    _kick_aim_theta_max_deg = _multi_skill_cfg.kick_aim_theta_max_deg
    _kick_aim_nominal_distance_m = _multi_skill_cfg.kick_aim_nominal_distance_m
else:
    _legacy_ball_cfg = load_ball_config()
    _ood_spawn_probability = _legacy_ball_cfg.ood_spawn_probability
    _ood_region_multiplier = _legacy_ball_cfg.ood_region_multiplier
    _start_at_timestep_zero_prob = _legacy_ball_cfg.start_at_timestep_zero_prob
    _rsi_scope_to_authored_clip = _legacy_ball_cfg.rsi_scope_to_authored_clip
    _critical_frame_oversampling_prob = _legacy_ball_cfg.critical_frame_oversampling_prob
    _critical_frame_sampling_window = _legacy_ball_cfg.critical_frame_sampling_window
    _motion_head_velocity_smoothing_frames = _legacy_ball_cfg.motion_head_velocity_smoothing_frames
    _kick_recovery_locomotion_flip_enabled = _legacy_ball_cfg.kick_recovery_locomotion_flip_enabled
    # No legacy counterpart -- see the MultiSkillConfig branch's own comment above.
    _mid_episode_kick_entry_prob = 0.0
    _mid_episode_kick_entry_min_steps = 100
    _mid_episode_kick_entry_max_residual = 0.0
    _pre_kick_decel_steps = 0.0
    _pre_kick_decel_target = 0.1
    _pre_kick_fallback_timeout_steps = 0.0
    _pre_kick_reward_ramp_steps = 0.0
    _pre_kick_termination_grace_steps = 0.0
    _pre_kick_reference_blend_steps = 0.0
    _mid_episode_kick_entry_ball_fixed = False
    _pre_kick_obs_ramp_steps = 0.0
    _post_flip_obs_ramp_steps = 0.0
    _warm_start_obs_ramp_steps = 0.0
    _kick_aim_theta_ref_deg = _legacy_ball_cfg.kick_aim_theta_ref_deg
    _kick_aim_theta_max_deg = _legacy_ball_cfg.kick_aim_theta_max_deg
    _kick_aim_nominal_distance_m = _legacy_ball_cfg.kick_aim_nominal_distance_m

# "Simultaneous per-skill task configs" (2026-08-15) -- see config_values/unified/g1/reward.py's
# own _skill_task_config_paths comment for the full design. Independently re-resolved here (not
# imported from reward.py/termination.py) for the same reason _multi_skill_cfg above is
# independently re-loaded rather than shared: each config_values module owns its own cheap,
# idempotent parse.
_skill_task_config_paths = (
    [
        resolve_task_config_path(sc.task_config) if sc.task_config is not None else None
        for sc in _multi_skill_cfg.skills
    ]
    if _multi_skill_cfg is not None
    else None
)

# 4 of the 5 RSI/reset-time sampling fields resolved above get a per-skill table -- None (the
# common case) unless skills genuinely diverge, same "no override -> inherit the global" fallback
# as every other field this mechanism touches. mid_episode_kick_entry_ball_fixed (the 5th) is
# deliberately NOT here -- see MotionConfig.mid_episode_kick_entry_ball_fixed's own docstring for
# why it has no per-skill counterpart (it decides a SHARED table's column count, not a per-motion
# value). All 4 are training-regime choices (which task_config a skill trains under), NOT per-clip
# content, so they're resolved from each skill's OWN task_config file -- unlike
# motion_head_velocity_smoothing_frames_per_motion, they do NOT live on SkillConfig.
_start_at_timestep_zero_prob_per_skill = resolve_per_skill_param(
    _skill_task_config_paths, "start_at_timestep_zero_prob", _start_at_timestep_zero_prob
)
_rsi_scope_to_authored_clip_per_skill = resolve_per_skill_param(
    _skill_task_config_paths, "rsi_scope_to_authored_clip", _rsi_scope_to_authored_clip
)
_critical_frame_oversampling_prob_per_skill = resolve_per_skill_param(
    _skill_task_config_paths, "critical_frame_oversampling_prob", _critical_frame_oversampling_prob
)
_critical_frame_sampling_window_per_skill = resolve_per_skill_param(
    _skill_task_config_paths, "critical_frame_sampling_window", _critical_frame_sampling_window
)

_motion_config_unified_kick = replace(
    motion_config_ball_kick,
    # start_at_timestep_zero_prob: yaml-configurable (configs/stageB_and_C.yaml's
    # start_at_timestep_zero_prob / configs/ball*.yaml, see MultiSkillConfig's field docstring for
    # the full 2026-07-28 retune rationale -- real telemetry showing plateaued kick tracking error
    # alongside a shared-policy entropy that keeps shrinking, addressed here via training-time
    # STATE diversity instead of touching the shared SAC entropy machinery). 1.0 (every kick
    # episode starts at the assigned skill's own frame 0) is still the default if unset -- bit-
    # identical to pre-2026-07-28 behavior.
    start_at_timestep_zero_prob=_start_at_timestep_zero_prob,
    # 2026-08-05, ported from RoboNaldo (arXiv:2606.11092) -- see MultiSkillConfig.
    # rsi_scope_to_authored_clip / critical_frame_oversampling_prob's own docstrings for the full
    # rationale. Both default to their exact-no-op values (False / 0.0) if unset.
    rsi_scope_to_authored_clip=_rsi_scope_to_authored_clip,
    critical_frame_oversampling_prob=_critical_frame_oversampling_prob,
    critical_frame_sampling_window=_critical_frame_sampling_window,
    # 2026-08-11, opt-in (0 = exact no-op): ramps the first N velocity frames of each clip up from
    # zero, in memory, BEFORE the prepend below is spliced in -- so the prepend eases into a calm
    # frame 0 instead of accelerating for a full second toward what is measurably the noisiest
    # frame in the clip. See MultiSkillConfig.motion_head_velocity_smoothing_frames and
    # managers/command/terms/motion_head_velocity_smoothing.py for the measurement.
    motion_head_velocity_smoothing_frames=_motion_head_velocity_smoothing_frames,
    # MUST stay False in N-skill mode: AdaptiveTimestepsSampler samples a GLOBAL frame index
    # across ALL skills' concatenated clips and DERIVES motion_ids from wherever it lands
    # (wbt.py's reset(), self.motion_ids[env_ids] = motion_ids), silently overwriting each env's
    # PERMANENT per-skill assignment from UnifiedManager._build_task_mode_partition -- an env
    # could get reassigned to a different skill than the one its shooting_reward_scale/ball
    # spawn/target were configured for. This is the "AdaptiveTimestepsSampler + fixed-skill combo"
    # the original N-skill plan explicitly deferred as unimplemented (see
    # n_skill_stageB_and_C_implemented_verified memory); it still is. Lowering
    # start_at_timestep_zero_prob above is unaffected by this bug (it only decides whether to snap
    # an already-correctly-within-assigned-skill uniform phase back to frame 0, never touches
    # motion_ids) -- that's why it's the lever used for training-time state diversity instead of
    # this one. The adaptive sampler's whole job is choosing WHICH frame to start at anyway; with
    # frame-0-biased starts it would only accumulate failure statistics it barely uses.
    use_adaptive_timesteps_sampler=False,
    enable_default_pose_prepend=True,
    default_pose_prepend_duration_s=1.0,  # matches the append/recovery duration
    ood_spawn_probability=_ood_spawn_probability,
    ood_region_multiplier=_ood_region_multiplier,
    kick_aim_theta_ref_deg=_kick_aim_theta_ref_deg,
    kick_aim_theta_max_deg=_kick_aim_theta_max_deg,
    kick_aim_nominal_distance_m=_kick_aim_nominal_distance_m,
    # Increment 4 of the locomotion->kick handoff -- see MultiSkillConfig.mid_episode_kick_entry_
    # ball_fixed's own docstring. MotionCommand.setup() needs this at TABLE-BUILD time (whether
    # _build_entry_search_table produces 5 or 7 feature columns), unlike its per-tick siblings
    # threaded through CommandManagerCfg.params below. False (default) = exact no-op.
    mid_episode_kick_entry_ball_fixed=_mid_episode_kick_entry_ball_fixed,
    # N-skill mode (HOLOSOMA_SKILLS_CONFIG set, see multi_skill_mode_enabled): N explicit clips in
    # yaml declaration order (motion_files takes precedence over motion_file/motion_dir above, see
    # MotionCommand.setup()), each with its own recovery/hold duration and ball/target/reward
    # config (skill_ball_configs, consumed by wbt.py's ball-reset logic and shooting.py's
    # _ShotTracker). Prepend duration is NOT per-skill (not part of the yaml schema) -- every skill
    # gets the same 1.0s windup as the single-clip default above (motion_prepend_duration_s stays
    # empty -> broadcasts default_pose_prepend_duration_s, see
    # MotionCommand._resolve_per_motion_durations). Off (empty lists / motion_file unchanged) when
    # N-skill mode is disabled -- identical to the pre-existing single-clip config.
    **(
        {
            "motion_files": [sc.motion_npz for sc in _multi_skill_cfg.skills],
            "motion_recovery_duration_s": [sc.recovery_duration_s for sc in _multi_skill_cfg.skills],
            "motion_hold_duration_s": [sc.hold_duration_s for sc in _multi_skill_cfg.skills],
            "motion_strike_start_frame": [sc.strike_start_frame for sc in _multi_skill_cfg.skills],
            "motion_stand_start_frame": [sc.stand_start_frame for sc in _multi_skill_cfg.skills],
            # "Simultaneous per-skill task configs" (2026-08-15) -- see SkillConfig.
            # motion_head_velocity_smoothing_frames's own docstring for why this is a per-CLIP
            # field (declared per motion_skill_N block) rather than resolved via a task_config
            # file. None per-skill (no override) falls back to the shared/global scalar above --
            # same fallback _motion_head_velocity_smoothing_frames already resolved from
            # task_config-or-legacy-BallConfig.
            "motion_head_velocity_smoothing_frames_per_motion": [
                (
                    sc.motion_head_velocity_smoothing_frames
                    if sc.motion_head_velocity_smoothing_frames is not None
                    else _motion_head_velocity_smoothing_frames
                )
                for sc in _multi_skill_cfg.skills
            ],
            # "Simultaneous per-skill task configs" (2026-08-15) -- the 4 RSI/reset-time sampling
            # fields WITH a per-skill counterpart (see MotionConfig.mid_episode_kick_entry_ball_
            # fixed's own docstring for why that 5th field has none). `or []` converts
            # resolve_per_skill_param's None (no genuine divergence) to the empty list
            # MotionConfig's own per-motion fields treat as "no override" -- see each field's own
            # docstring in config_types/command.py.
            "start_at_timestep_zero_prob_per_motion": _start_at_timestep_zero_prob_per_skill or [],
            "rsi_scope_to_authored_clip_per_motion": _rsi_scope_to_authored_clip_per_skill or [],
            "critical_frame_oversampling_prob_per_motion": _critical_frame_oversampling_prob_per_skill or [],
            "critical_frame_sampling_window_per_motion": _critical_frame_sampling_window_per_skill or [],
            "skill_ball_configs": list(_multi_skill_cfg.skills),
        }
        if _multi_skill_cfg is not None
        else {}
    ),
)


def _tagged_reset(terms: dict[str, CommandTermCfg], task_mode: str) -> dict[str, CommandTermCfg]:
    return {name: replace(cfg, task_mode=task_mode) for name, cfg in terms.items()}


# Stop-from-motion robustness (unified-only; the standalone loco baseline preset is left exactly
# as validated). Measured on the v6 Stage-B checkpoint: the policy reliably stops from moderate
# speed but FALLS stopping from a fast walk (0.8 m/s) at most gait phases -- reproduced in both
# RoboJuDo and holosoma_inference deploy stacks, so it's the checkpoint, not the deploy code.
#
# Root cause is training FREQUENCY of the walk->stop transition. Stock resamples the command every
# 10s in a 20s episode (~1 mid-episode change) and only stand_prob=0.2 of those become a stop.
# Quantified by driving the real LocomotionCommand offline over many episodes: a mid-episode
# walk->stop happens in only ~15% of env-episodes stock. Shortening the interval (10->5s) and
# raising stand_prob (0.2->0.3) lifts that to ~84% -- a 5.4x increase in how often the policy
# practices arresting momentum. (62% of those stops are already from a fast walk, |v|>0.7, so once
# stops are frequent, high-speed stops are covered without any extra targeting.)
#
# NOTE: a per-env resample DESYNC was prototyped and REJECTED. The intuition was that stock
# resamples every env at the same steps (t % interval) so stops would be phase-locked -- but the
# same offline sim showed stop-phase coverage is ALREADY uniform (worst 10-bin: 9.3%), because the
# existing per-env gait_freq randomization (period 1.0 +/- 0.2) decouples "same step" from "same
# gait phase". Desync moved worst-bin coverage 9.3% -> 9.7% (noise), so it was not worth adding
# state to a shared term. The frequency levers below are the whole fix.
_loco_command_stop_robust = replace(
    g1_29dof_command.setup_terms["locomotion_command"],
    params={
        **g1_29dof_command.setup_terms["locomotion_command"].params,
        "stand_prob": 0.3,  # was 0.2
    },
)

# Clean gait initiation (unified-only). The DEPLOY policy force-resets the gait phase to the
# canonical [0, pi] gait-start on every stand->walk transition, but stock training resumes its
# free-running phase clock from an arbitrary phase there -- so the policy never specifically
# practices initiating a step from a standing body at [0, pi], and starts each walk with the same
# uncompensated first swing foot: a consistent ~-3.8deg yaw + lateral veer measured on v5.
# reanchor_phase_on_gait_start makes training re-anchor to [0, -pi] (== deploy's [0, pi] in
# sin/cos) at that transition, so the trained initiation matches the deployed one. See the term's
# __init__ in managers/command/terms/locomotion.py for the full rationale.
_loco_gait_clean_start = replace(
    g1_29dof_command.setup_terms["locomotion_gait"],
    params={
        **g1_29dof_command.setup_terms["locomotion_gait"].params,
        "reanchor_phase_on_gait_start": True,
    },
)


g1_29dof_unified_command = CommandManagerCfg(
    params={
        **g1_29dof_command.params,
        "locomotion_command_resampling_time": 5.0,  # was 10.0 -- see _loco_command_stop_robust
        # kick_probability: legacy 2-way (locomotion/kick) fallback, read by
        # UnifiedManager._build_task_mode_partition ONLY when skill_motion_training_ratios below is
        # empty (N-skill mode off) -- harmless to keep setting even when N-skill mode IS on, since
        # a non-empty skill_motion_training_ratios takes priority there and this is simply unused.
        "kick_probability": load_skill_mix_config().kick_probability,
        # Stage D's post-swing -> locomotion handoff switch -- read directly by UnifiedManager
        # control flow (self.command_manager.command_cfg.kick_recovery_locomotion_flip_enabled),
        # same access pattern as kick_probability above, NOT via config_values/unified/g1/reward.py
        # (that path is only for reward-term params, resolved at reward-manager-construction time;
        # this flag gates a per-tick task_mode mutation in _update_tasks_callback instead). See
        # MultiSkillConfig.kick_recovery_locomotion_flip_enabled's own docstring for the full
        # rationale. False (default) if unset -- exact no-op.
        "kick_recovery_locomotion_flip_enabled": _kick_recovery_locomotion_flip_enabled,
        # Locomotion->kick direction -- same direct-read access pattern (self.command_manager.
        # command_cfg.mid_episode_kick_entry_prob, in UnifiedManager._resample_task_mode), same
        # no-op guarantee at the 0.0 default. See MultiSkillConfig.mid_episode_kick_entry_prob's
        # own docstring for the full mechanism.
        "mid_episode_kick_entry_prob": _mid_episode_kick_entry_prob,
        "mid_episode_kick_entry_min_steps": _mid_episode_kick_entry_min_steps,
        "mid_episode_kick_entry_max_residual": _mid_episode_kick_entry_max_residual,
        "pre_kick_decel_steps": _pre_kick_decel_steps,
        "pre_kick_decel_target": _pre_kick_decel_target,
        "pre_kick_fallback_timeout_steps": _pre_kick_fallback_timeout_steps,
        "pre_kick_reward_ramp_steps": _pre_kick_reward_ramp_steps,
        "pre_kick_termination_grace_steps": _pre_kick_termination_grace_steps,
        "pre_kick_reference_blend_steps": _pre_kick_reference_blend_steps,
        "mid_episode_kick_entry_ball_fixed": _mid_episode_kick_entry_ball_fixed,
        # Read by UnifiedManager.__init__ (self._pre_kick_obs_ramp_steps) and consumed by its
        # task_mode_mask_soft/pre_kick_obs_ramp_alpha, which ObservationManager calls in place of
        # the hard task_mode_mask. 0.0 default = exact no-op.
        "pre_kick_obs_ramp_steps": _pre_kick_obs_ramp_steps,
        # FIX 5: the kick->locomotion flip direction. Same consumption path (UnifiedManager
        # reads it in __init__, task_mode_mask_soft applies it). 0.0 default = exact no-op.
        "post_flip_obs_ramp_steps": _post_flip_obs_ramp_steps,
        # FIX 6: load-time blend. Read by UnifiedManager.__init__ (self._warm_start_obs_ramp_
        # steps) and consumed by FastSACAgent.load(), which configures the observation
        # manager's blend state directly. 0.0 default = exact no-op.
        "warm_start_obs_ramp_steps": _warm_start_obs_ramp_steps,
        # N-skill mode: each skill's motion_training_ratio, in the same yaml declaration order as
        # motion_files above -- UnifiedManager._build_task_mode_partition draws a per-env
        # categorical over [locomotion, skill_1, ..., skill_N] weighted by
        # [1 - sum(ratios), *ratios]. Empty (N-skill mode off) -> legacy kick_probability path.
        **(
            {"skill_motion_training_ratios": [sc.motion_training_ratio for sc in _multi_skill_cfg.skills]}
            if _multi_skill_cfg is not None
            else {}
        ),
    },
    setup_terms={
        **g1_29dof_command.setup_terms,
        "locomotion_gait": _loco_gait_clean_start,
        "locomotion_command": _loco_command_stop_robust,
        **g1_29dof_ball_kick_command.setup_terms,
        "motion_command": replace(
            g1_29dof_ball_kick_command.setup_terms["motion_command"],
            params={
                **g1_29dof_ball_kick_command.setup_terms["motion_command"].params,
                "motion_config": _motion_config_unified_kick,
            },
        ),
    },
    reset_terms={
        **_tagged_reset(g1_29dof_command.reset_terms, "locomotion"),
        **_tagged_reset(g1_29dof_ball_kick_command.reset_terms, "kick"),
    },
    step_terms={
        **g1_29dof_command.step_terms,
        **g1_29dof_ball_kick_command.step_terms,
    },
)

__all__ = ["g1_29dof_unified_command"]
