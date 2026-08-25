from dataclasses import replace

from holosoma.config_types.experiment import ExperimentConfig, TrainingConfig
from holosoma.config_types.multi_skill import load_multi_skill_config, multi_skill_mode_enabled
from holosoma.config_types.simulator import load_ball_config
from holosoma.config_values import (
    action,
    algo,
    command,
    curriculum,
    observation,
    randomization,
    reward,
    robot,
    simulator,
    termination,
    terrain,
)

# Single FastSAC policy that learns BOTH locomotion (velocity tracking) and ball-kicking (motion
# tracking): each episode, per env, one task is picked at reset according to
# configs/skill_mix.yaml's kick_probability — restricted to envs on flat terrain (see
# UnifiedManager/env_terrain_is_flat), since a freely-simulated ball needs flat ground to rest at
# its configured position. Observation space is the union of both tasks' observations (each
# term zeroed, never omitted, when its task isn't active for a given env) plus a task_mode_onehot
# term so the policy can condition on which task it's currently running.
#
# terrain_unified_mix raises the "flat" proportion to 40% (vs. stock terrain_locomotion_mix's
# 20%) so there's enough flat-terrain envs to reasonably host the configured kick_probability —
# realized global kick fraction = (flat fraction) x kick_probability, logged each step as
# kick_eligible_frac / kick_active_frac (see UnifiedManager._update_log_dict).
#
# To bootstrap a locomotion-first policy before adding kicking: set configs/skill_mix.yaml's
# kick_probability low (or 0.0), train, then raise it and resume from that checkpoint — the
# observation/action space doesn't change with this value, only which reward/termination/command
# paths are active per episode, so resuming works cleanly.
#
# Ball radius/mass/x/y come from configs/ball.yaml (see load_ball_config's docstring); post-kick
# recovery/hold timing comes from configs/stabilization.yaml (see load_stabilization_config,
# wired in via command.g1_29dof_ball_kick_command's motion_config_ball_kick). In N-skill mode
# (HOLOSOMA_SKILLS_CONFIG set) radius/mass instead come from MultiSkillConfig (see
# _scene_ball_cfg below) -- shared physical geometry, still one physics ball per env regardless of
# which of the N skills that env is running; everything else on this scene-level BallConfig
# (position, kick_foot, etc.) goes unused in N-skill mode, superseded by each skill's own
# SkillConfig (config_values/unified/g1/command.py's skill_ball_configs, consumed directly by
# wbt.py/shooting.py) -- kept at load_ball_config()'s defaults here purely so the dataclass is
# still valid to construct.
#
_multi_skill_cfg_for_scene = load_multi_skill_config() if multi_skill_mode_enabled() else None
_scene_ball_cfg = (
    replace(load_ball_config(), radius=_multi_skill_cfg_for_scene.radius, mass=_multi_skill_cfg_for_scene.mass)
    if _multi_skill_cfg_for_scene is not None
    else load_ball_config()
)
# Opt-in per-task-mode SAC entropy target (2026-07-28) -- see FastSACConfig.kick_target_entropy_ratio's
# own docstring for the full design/rationale. None (unset in yaml, the default) reproduces
# today's single shared alpha exactly; same dual-path resolution every other global (not
# per-skill) field in this project uses.
_kick_target_entropy_ratio = (
    _multi_skill_cfg_for_scene.kick_target_entropy_ratio
    if _multi_skill_cfg_for_scene is not None
    else _scene_ball_cfg.kick_target_entropy_ratio
)
# Opt-in per-task-mode SAC discount factor (2026-07-30) -- see FastSACConfig.kick_gamma's own
# docstring for the full design/rationale. Same dual-path resolution as _kick_target_entropy_ratio
# immediately above.
_kick_gamma = (
    _multi_skill_cfg_for_scene.kick_gamma
    if _multi_skill_cfg_for_scene is not None
    else _scene_ball_cfg.kick_gamma
)
# Opt-in distributional-critic value support (2026-08-21) -- see MultiSkillConfig.critic_v_min's
# own docstring for the saturation measurement that motivated it, and critic_num_atoms' for the two
# checkpoint hazards. Same dual-path resolution as _kick_gamma above. None (unset in yaml, the
# default) falls through to the preset's own hardcoded values below, bit-identical to before.
_critic_v_min = (
    _multi_skill_cfg_for_scene.critic_v_min
    if _multi_skill_cfg_for_scene is not None
    else _scene_ball_cfg.critic_v_min
)
_critic_v_max = (
    _multi_skill_cfg_for_scene.critic_v_max
    if _multi_skill_cfg_for_scene is not None
    else _scene_ball_cfg.critic_v_max
)
_critic_num_atoms = (
    _multi_skill_cfg_for_scene.critic_num_atoms
    if _multi_skill_cfg_for_scene is not None
    else _scene_ball_cfg.critic_num_atoms
)
# Opt-in replay-buffer NaN/Inf guard (2026-08-10) -- see FastSACConfig.replay_buffer_sanitize_enabled's
# own docstring for the full design/rationale. Same dual-path resolution as _kick_gamma above.
_replay_buffer_sanitize_enabled = (
    _multi_skill_cfg_for_scene.replay_buffer_sanitize_enabled
    if _multi_skill_cfg_for_scene is not None
    else _scene_ball_cfg.replay_buffer_sanitize_enabled
)
# Opt-in per-skill replay weighting (2026-08-15) -- see FastSACConfig.skill_replay_weights' own
# docstring. DATA-side sibling of l2sp_weight below (parameter-side). Only ever available in
# N-skill mode: BallConfig has no per-skill concept at all, so the legacy path is always empty/OFF.
_skill_replay_weights = (
    list(_multi_skill_cfg_for_scene.skill_replay_weights) if _multi_skill_cfg_for_scene is not None else []
)
# Opt-in L2-SP continual-learning anchor (2026-08-15) -- see FastSACConfig.l2sp_weight's own
# docstring for the full design and the value table. Same dual-path resolution as
# _replay_buffer_sanitize_enabled above. 0.0 (unset in yaml, the default) = OFF, bit-identical.
_l2sp_weight = (
    _multi_skill_cfg_for_scene.l2sp_weight
    if _multi_skill_cfg_for_scene is not None
    else _scene_ball_cfg.l2sp_weight
)
#
# num_envs=2048 (not stock WBT/ball-kick's 4096): FastSAC's replay buffer
# (agents/fast_sac/fast_sac_utils.py::SimpleReplayBuffer) preallocates
# (num_envs, buffer_size, obs_dim) tensors up front regardless of how much real experience has
# been collected. Unified's merged observation space (critic_obs_dim=391 vs ball-kick's 286) makes
# each replay-buffer slot ~47% bigger, and terrain_unified_mix's mixed-terrain mesh needs more
# sim/render memory than ball-kick's flat plane on top of that — at num_envs=4096 this OOMs on a
# 32GB GPU (measured: ~21GB for the replay buffer alone, confirmed against the OOM traceback).
# num_envs=2048 roughly halves the replay buffer (~10GB) and fits comfortably. Raise it back if
# your GPU has more headroom, or lower --algo.config.buffer-size instead to keep num_envs higher.
g1_29dof_unified_fast_sac = ExperimentConfig(
    training=TrainingConfig(
        project="LocomotionAndBallKicking",
        name="g1_29dof_unified_fast_sac_manager",
        num_envs=2048,
    ),
    env_class="holosoma.envs.unified.unified_manager.UnifiedManager",
    algo=replace(
        algo.fast_sac,
        config=replace(
            algo.fast_sac.config,
            num_learning_iterations=400000,
            # 20.0/-20.0 stay the PRESET defaults; configs/*.yaml's critic_v_max/critic_v_min
            # override them when set (None = unset = keep these). See MultiSkillConfig.critic_v_min.
            v_max=_critic_v_max if _critic_v_max is not None else 20.0,
            v_min=_critic_v_min if _critic_v_min is not None else -20.0,
            # gamma/num_updates/num_atoms/policy_frequency/target_entropy_ratio/tau: locomotion's
            # proven values (g1_29dof_loco_fast_sac's implicit fast_sac defaults), NOT the
            # ball-kick/WBT experiment's tuned set this block originally copied wholesale. Measured
            # side-by-side at matched iteration counts (kick_probability=0, everything else
            # identical): the WBT values (gamma=0.99, num_updates=4, num_atoms=501,
            # policy_frequency=2, target_entropy_ratio=0.5, tau=0.05) give a policy that never
            # settles under a zero-velocity command — mean_lin_vel=0.56 m/s / mean_joint_vel_rms=
            # 0.76 at 12k iterations, WORSENING with more training (up to 0.80 m/s by 145k) — while
            # these locomotion values reach mean_lin_vel=0.031 / mean_joint_vel_rms=0.111 at the
            # same 12k iterations, already better than holosoma's own fully-converged 50k-iteration
            # baseline (0.079 / 0.147). target_entropy_ratio in particular (0.5 vs 0.0) pushes the
            # policy toward far noisier actions than locomotion needs; motion-tracking (kick) may
            # separately benefit from more exploration, but nothing so far shows it needs THIS
            # specific value, and a single shared policy only gets one setting.
            #
            # 2026-07-28: "a single shared policy only gets one setting" above is no longer strictly
            # true -- kick_target_entropy_ratio (opt-in, see FastSACConfig's own docstring) lets
            # kick-mode transitions use a SEPARATE entropy target from target_entropy_ratio below,
            # resolved from configs/stageB_and_C.yaml's kick_target_entropy_ratio (None/unset here
            # reproduces today's single-shared-alpha behavior exactly).
            gamma=0.97,
            num_steps=1,
            num_updates=8,
            num_atoms=_critic_num_atoms if _critic_num_atoms is not None else 101,
            policy_frequency=4,
            target_entropy_ratio=0.0,
            kick_target_entropy_ratio=_kick_target_entropy_ratio,
            kick_gamma=_kick_gamma,
            replay_buffer_sanitize_enabled=_replay_buffer_sanitize_enabled,
            l2sp_weight=_l2sp_weight,
            skill_replay_weights=_skill_replay_weights,
            tau=0.125,
            # use_symmetry stays False (unchanged from before): the standard symmetry-augmentation
            # code (agents/modules/augmentation_utils.py) dispatches mirror functions by exact
            # observation-term name (mirror_obs_<term>), but unified's observation.py prefixes every
            # term ("loco_base_ang_vel", "kick_dof_pos", ...) to dedupe locomotion/kick name
            # collisions, so none of those prefixed names match — turning this on would crash
            # immediately, not silently misbehave. Not needed for the fix above: the locomotion
            # values already converge better than baseline without it.
            use_symmetry=False,
        ),
    ),
    simulator=replace(
        simulator.isaacsim,
        config=replace(
            simulator.isaacsim.config,
            # 20s, NOT the ball-kick experiment's 10s. Two measured problems with 10s:
            # 1. The full kick sequence doesn't fit: kick clip (e.g. 7.64s for a 382-frame/50fps
            #    clip) + recovery (1s) + hold (2s) = ~10.6s, so the timeout cut every kick episode
            #    before the stabilization hold finished — the exact behavior being trained.
            # 2. The penalty curriculum (g1_29dof_curriculum_fast_sac) levels up at
            #    average_episode_length >= 750 steps; with a 500-step (10s) hard cap that's
            #    unreachable, silently freezing all penalty weights at initial_scale=0.5 forever.
            #    Stock locomotion trains with episodes long enough for this to engage.
            sim=replace(simulator.isaacsim.config.sim, max_episode_length_s=20.0),
            scene=replace(simulator.isaacsim.config.scene, ball=_scene_ball_cfg),
        ),
    ),
    robot=replace(
        robot.g1_29dof,
        control=replace(
            robot.g1_29dof.control,
            action_scale=0.25,
            action_scales_by_effort_limit_over_p_gain=True,
            # NOT YET ENABLED -- 2026-08-05, ported from RoboNaldo (arXiv:2606.11092), see
            # RobotControlConfig.per_joint_action_clip's own docstring for the full derivation.
            # None (default, current state) is a true no-op -- every joint uses the scalar
            # action_clip_value (100.0, effectively unconstraining) unchanged. The 3 values below
            # are independently re-derived from THIS project's own joint limits and computed
            # per-joint action scales (NOT copied from RoboNaldo's raw numbers, which are
            # calibrated to their own per-joint scales and would not transfer verbatim for most
            # joints -- see the field docstring for why arm/wrist joints specifically are excluded
            # here and left to ArmDefaultPose instead), and happen to reproduce RoboNaldo's own
            # values exactly because this project's G1 spec matches their hardware. Uncomment to
            # opt in as a deliberate, separate decision -- same "ship inert, validate via a real
            # training run" discipline as every other new mechanism in this port.
            # per_joint_action_clip={
            #     "ankle_roll": (-0.6, 0.6),    # URDF limit +/-0.2618 rad / scale ~0.4386 rad/unit
            #     "waist_roll": (-1.2, 1.2),    # URDF limit +/-0.52 rad / scale ~0.4386 rad/unit
            #     "waist_pitch": (-1.2, 1.2),   # URDF limit +/-0.52 rad / scale ~0.4386 rad/unit
            # },
        ),
        asset=replace(robot.g1_29dof.asset, enable_self_collisions=True),
        init_state=replace(robot.g1_29dof.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    terrain=terrain.terrain_unified_mix,
    observation=observation.g1_29dof_unified_observation,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_unified_termination,
    randomization=randomization.g1_29dof_unified_randomization,
    command=command.g1_29dof_unified_command,
    curriculum=curriculum.g1_29dof_unified_curriculum,
    reward=reward.g1_29dof_unified_reward,
)

# Stage D (2026-08-09): post-swing -> locomotion handoff, see MultiSkillConfig.
# kick_recovery_locomotion_flip_enabled's own docstring for the full mechanism.
#
# 2026-08-11, MEASURED NEGATIVE RESULT -- do not "fix" this by lowering the cap again without
# re-reading this note. Stage D's Env/kick_active_frac is roughly half C1's (a dead-flat 0.450 in
# the C1 parent run 20260810_083756, where task_mode never changes so it just reports the
# partition fraction, vs 0.226-0.255 in the Stage D run 20260810_144118), and the obvious
# suspected cause was this doubled cap: a flipped env freezes its motion clock (MotionCommand.
# step's advance_mask &= task_mode_mask("kick")) so it can never reach motion_end_idx and never
# triggers that function's mid-episode re-cycle (wbt.py: reset() + ball teleport WITHOUT ending
# the episode), leaving it to stand for the entire remainder of a 40s episode.
#
# That hypothesis was tested directly (scratchpad verify_episode_length_kick_data_rate.py: 512
# envs, ckpt 0333000, 4000 steps, 2433 recorded kick episodes, projecting every cap from one
# rollout; its 40s projection reproduces the training-logged number to within 0.002, and its
# direct measurement 0.2523 agrees with its own projection 0.2535) and is WRONG. The cap is
# almost never the binding constraint -- episodes terminate long before it:
#
#     episode length (ticks): p50=283  p75=364  p90=573  p99=1046  max=1616
#     surviving past 1000 ticks (20s): 32/2433 episodes (1.3%), contributing 0.8% of all ticks
#     projected kick_active_frac: 40s=0.2535  30s=0.2536  25s=0.2539  20s=0.2556  15s=0.2611
#
# 40s -> 20s moves kick_active_frac by +0.8% -- nothing. Kept at 40.0s: the cap is behaviorally
# near-inert either way, and the few long-surviving episodes it does admit are exactly the
# successful post-flip stands this stage exists to train, so truncating them is a small net loss.
#
# The real driver of the lower kick_active_frac is structural, not the cap: kick-mode ticks are
# hard-capped at the flip boundary (an env can accumulate at most pre_recovery_motion_end_idx
# ticks of kick mode per episode, ~199 on average given RSI) while post-flip standing ticks grow
# with however long the episode survives. NOTE also that most of the 0.45 -> 0.25 drop is the
# recovery/hold tail being RELABELLED from kick-mode to locomotion-mode, which is precisely what
# this stage intends -- so kick_active_frac OVERSTATES the loss of approach/strike training data.
# The narrower question (ticks spent specifically in the [prepend_end, stand_start_idx) approach+
# strike window, C1 vs Stage D) has NOT been measured, and should be before treating "Stage D
# starves approach/strike of data" as established.
#
# Nothing else differs from g1_29dof_unified_fast_sac -- Stage D reuses C1/C2's exact reward
# mechanism (task_mode-gated locomotion reward takes over post-flip via existing machinery, no new
# reward terms), so this preset is a pure derivation, not an independent experiment definition.
# (max_episode_length_s is a SimEngineConfig field, not MultiSkillConfig -- confirmed no
# task_config_stage*.yaml file can set it, which is why it lives here at all.)
g1_29dof_unified_fast_sac_stageD = replace(
    g1_29dof_unified_fast_sac,
    simulator=replace(
        g1_29dof_unified_fast_sac.simulator,
        config=replace(
            g1_29dof_unified_fast_sac.simulator.config,
            sim=replace(g1_29dof_unified_fast_sac.simulator.config.sim, max_episode_length_s=40.0),
        ),
    ),
)

__all__ = ["g1_29dof_unified_fast_sac", "g1_29dof_unified_fast_sac_stageD"]

"""
Example — fresh training run (edit configs/ball.yaml / stabilization.yaml / skill_mix.yaml to
tune the ball, post-kick hold, and kick probability first; motion_file defaults to a bundled
placeholder, override it with your own kick clip):
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-unified-fast-sac \
    logger:wandb \
    --command.setup_terms.motion_command.params.motion_config.motion_file=/path/to/your/kick_clip.npz

Example — the full three-stage bootstrap (RoboNaldo-style curriculum, arXiv:2606.11092; each
stage resumes from the previous checkpoint via --training.checkpoint, since none of these knobs
change the observation/action space):

Stage A — locomotion only:
  configs/skill_mix.yaml:  kick_probability: 0.0
  (ball.yaml's shooting settings are irrelevant this stage — kick-mode terms never run)

Stage B — + kick motion tracking (RoboNaldo Stage 1: pure imitation, w_g = 0):
  configs/skill_mix.yaml:  kick_probability: 0.7
  configs/ball.yaml:       shooting_reward_scale: 0.0, randomize_x/y: 0.0
  The policy learns a stable kick by tracking the clip; the shooting rewards are skipped
  entirely (weight 0), and the ball/target obs are present but constant.

Stage C — + shooting adaptation (RoboNaldo Stage 2, w_g = 0.8):
  configs/ball.yaml:       shooting_reward_scale: 0.8, randomize_x/y: 0.75,
                           kick_aim_enabled: true, kick_aim_theta_max_deg: e.g. 15.0
  Ball spawn now randomizes per attempt; the target is FIXED per skill (its own calibrated
  nominal bearing, see scripts/calibrate_nominal_bearing.py) and the observed direction varies
  via kick_aim_theta instead (the old independent target_randomization jitter was removed
  2026-08-22). Both ball and aim command are observed by the policy (kick_ball_pos_b /
  kick_aim_command), and the six shooting reward terms shape the tracked swing for contact and
  goal-directed placement. Optionally ramp randomize_x/y and kick_aim_theta_max_deg up gradually
  across resumes if contact rate collapses at the full range.
"""
