"""Configuration types for the stacked N-motion-skill training yaml (configs/stageB_and_C.yaml).

Replaces skill_mix.yaml + ball.yaml + stabilization.yaml for TRAINING a unified locomotion +
N-motion-skill policy. configs/ball.yaml (single skill) is unaffected and stays the mechanism for
single-clip eval/replay -- see load_ball_config in config_types/simulator.py.

Each motion_skill_N block is exactly one clip (motion_npz, a single file -- not a directory) plus
that skill's own ball spawn/target/reward-scale/recovery-hold configuration. motion_training_ratio
is each skill's share of envs assigned to it, PERMANENTLY, for the whole run (mirrors
SkillMixConfig.kick_probability's own per-env-fixed-for-life partition -- see
UnifiedManager._build_task_mode_partition's docstring for why per-episode resampling was rejected:
it measurably under-represents short-episode skills in the replay buffer). The remainder
(1 - sum(motion_training_ratio)) is locomotion's share.
"""

from __future__ import annotations

import math
import os
from dataclasses import field
from pathlib import Path

import yaml
from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class SkillConfig:
    """One motion skill: exactly one reference clip plus its own ball/target/reward/timing config.

    Stage is implied by shooting_reward_scale, not a separate field: 0.0 -> Stage B (pure motion
    tracking, no shooting adaptation); > 0.0 -> Stage C (ball shooting). A skill can be added, and
    trained, at Stage B only -- Stage C is optional per skill."""

    motion_npz: str
    """Path to this skill's single reference motion clip (.npz). One file per skill -- not a
    directory; skill:clip is 1:1."""

    x: float
    """Ball nominal spawn position, env-local frame, forward from the robot (meters)."""

    y: float
    """Ball nominal spawn position, env-local frame, lateral (meters)."""

    motion_training_ratio: float
    """This skill's permanent share of envs, in [0, 1]. See module docstring for why this is a
    per-env-fixed-for-life partition, not a per-episode probability."""

    strike_start_frame: int
    """Clip-local frame index (0-indexed, into the RAW, un-prepended npz -- exactly what
    motion_clip_scrubber.py displays) where this clip's locomotion-approach mode ends and the
    strike/swing mode begins. REQUIRED: unlike recovery_duration_s/hold_duration_s, there is no
    sensible single global default -- each clip's own kinematic content determines this, and
    silently defaulting to e.g. 0 would make in_strike_phase silently equal in_kicking_phase with
    no error, hiding a missing config value. Must satisfy
    0 <= strike_start_frame < stand_start_frame <= this clip's own raw frame count. Used to gate
    shooting.py's 6 reward terms to the strike itself (in_strike_phase) -- see
    MotionCommand.setup()."""

    stand_start_frame: int
    """Clip-local frame index (0-indexed, into the RAW npz) where the strike/swing mode ends and
    post-kick-standing mode begins. REQUIRED, same rationale as strike_start_frame. This is also
    the new END boundary for in_kicking_phase (renamed from in_swing_phase) -- moved from the
    whole clip's own end (pre_recovery_motion_end_idx) to here."""

    randomize_x: float = 0.0
    """Per-reset uniform ± randomization half-range on x (meters). 0.0 = fixed spawn (Stage B-style)."""

    randomize_y: float = 0.0
    """Per-reset uniform ± randomization half-range on y (meters)."""

    target_x: float | None = None
    """Nominal shot target x, env-local frame. Defaults to x + 5.0 (RoboNaldo's 5m goal-plane
    distance), matching load_ball_config's own default.

    When ``kick_aim_enabled=True``, this point's role changes: it is used ONLY (via
    ``resolved_nominal_bearing_deg()``) to derive this skill's calibrated nominal strike
    DIRECTION -- its absolute position is no longer read as a distance target anywhere (see
    scripts/calibrate_nominal_bearing.py, which measures this skill's real departure bearing from
    a genuine rollout and writes it back into target_x/target_y at whatever distance is
    convenient; the bearing is invariant to that choice). An uncalibrated target_x/target_y under
    kick_aim_enabled silently biases every commanded kick_aim_theta by the gap between the
    authored and actual strike direction -- measured as large as 39 degrees for one skill in this
    project before calibration. Under kick_aim_enabled=False, this point is instead read as a
    fixed, UNrandomized per-attempt target (see kick_aim_enabled's own docstring -- the old
    randomize_target_x/y independent-randomization mechanism was removed 2026-08-22, once every
    skill in this project had moved to kick_aim_enabled=True and it had no live consumer left)."""

    target_y: float | None = None
    """Nominal shot target y, env-local frame. Defaults to y. See target_x's own docstring for its
    changed role under kick_aim_enabled=True."""

    kick_aim_enabled: bool = False
    """2026-08-22, azimuth-aim refactor. True: this skill's target is no longer an independent
    point. At setup time, resolved_nominal_bearing_deg() derives this skill's calibrated nominal
    bearing from resolved_target(); at every reset (and mid-episode kick entry), a fresh
    kick_aim_theta is sampled (uniform, +/- MultiSkillConfig.kick_aim_theta_max_deg or this skill's
    own override), and the target point actually used is SYNTHESIZED from the ball's own actual
    placed position (post position_randomization/OOD noise) plus MultiSkillConfig.
    kick_aim_nominal_distance_m along (nominal_bearing_deg + kick_aim_theta) -- see the three
    ball/target placement paths in managers/command/terms/wbt.py. This makes target_w - ball_w
    exactly D * unit(bearing + theta) BY CONSTRUCTION, independent of the ball's own spawn noise,
    and lets kick_aim_theta be fed to the actor as a bounded, localization-free command (see
    managers/observation/terms/unified.py's kick_aim_command) instead of the world-frame target
    offset kick_target_pos_b required.

    False (default): target_x/target_y (resolved_target()) is read as a FIXED per-attempt target
    -- no randomization at all (the independent randomize_target_x/y mechanism this field used to
    gate was removed 2026-08-22: once every skill in this project had moved to kick_aim_enabled=
    True, it had no live consumer, and re-deriving a target's variation from the ball's own noise
    -- which kick_aim_enabled=True already does -- was judged the only mechanism worth keeping,
    rather than maintaining two ways to randomize a target side by side).

    Requires a CALIBRATED target_x/target_y for this skill (see target_x's own docstring) --
    this field does not calibrate anything itself, it only changes how target_x/target_y and the
    new theta command are consumed."""

    kick_foot: str = "right"
    """Which foot ("left" or "right") this skill's shooting rewards are computed for."""

    success_radius: float = 0.5
    """A shot succeeds when the ball passes within this distance (meters) of the target."""

    shooting_reward_scale: float = 0.0
    """Target w_g for this skill's 6 shooting reward terms. 0.0 (default) = Stage B, pure motion
    tracking -- this skill's shooting terms never fire."""

    recovery_duration_s: float = 1.0
    """Seconds of smooth transition from this clip's own ending pose back to the default standing
    pose, appended immediately after this clip's own frames."""

    hold_duration_s: float = 2.0
    """Seconds of static balance hold immediately after this skill's own recovery transition."""

    observation_bias: float = 0.0
    """Per-episode constant heading-frame bias (meters, uniform ± this half-range) added to this
    skill's kick_ball_pos_b observation -- mirrors BallConfig.observation_bias (configs/ball*.yaml)
    but per-skill instead of a single global value. 0.0 (default) = no bias, exact no-op. See
    managers/randomization/terms/locomotion.py::randomize_ball_obs_bias."""

    motion_tracking_reward_scale: float = 1.0
    """Multiplier on this skill's 6 kick-mode motion-tracking reward terms (the WBT-derived
    exp(-error/sigma^2) position/orientation/velocity tracking terms, tagged task_mode="kick").
    1.0 (default) = current behavior, exact no-op. Applied live, per env, per step, via
    utils/kick_reward_scales.py -- same runtime-resolution reason as shooting_reward_scale (which
    skill an env is on is only known at env-runtime via motion_ids, never at config-import time,
    so this can't be baked into RewardTermCfg.weight the way a single-skill config could).
    Applies uniformly across the WHOLE kick episode (swing + recovery/hold) -- for a
    phase-specific reduction during recovery/hold only, see recovery_tracking_scale below."""

    root_tracking_reward_scale: float = 1.0
    """ADDITIONAL multiplier on ONLY this skill's 2 root/anchor kick-mode motion-tracking reward
    terms (motion_global_ref_position_error_exp, motion_global_ref_orientation_error_exp) -- the
    other 5 (motion_relative_body_position_error_exp, motion_relative_body_orientation_error_exp,
    motion_global_body_lin_vel, motion_global_body_ang_vel, and motion_feet_lin_vel once ported)
    are UNAFFECTED by this field; motion_tracking_reward_scale alone still governs them. 1.0
    (default) = current behavior, exact no-op -- at the default, the root terms' effective
    multiplier is exactly motion_tracking_reward_scale(env), byte-identical to before this field
    existed.

    Motivation (2026-08-05, ported from RoboNaldo arXiv:2606.11092 -- see
    ROBONALDO_PORT_SCOPE.md Sec 1a "Key Finding 1"): RoboNaldo relaxes ONLY its root/anchor
    tracking terms across its curriculum (motion_global_anchor_pos/_ori: 1.0 -> 0.1 going their
    S2a -> S2b), while its 5 RELATIVE body-pose/velocity terms stay pinned at 1.0 in EVERY stage,
    forever -- that 10x drop on world-frame root tracking IS their entire "let the robot go get
    the ball" adaptation; the body-pose PRIOR that keeps the strike looking like the authored clip
    never relaxes. motion_tracking_reward_scale, by contrast, relaxes all 6-7 of our tracking
    terms UNIFORMLY -- a likely direct cause of the measured strike-phase divergence, since the
    body-pose prior that should stay pinned gets relaxed along with the root.

    This field is ADDITIVE, not a redefinition of motion_tracking_reward_scale's existing meaning
    -- existing skill yamls were tuned under "motion_tracking_reward_scale applies to all 6/7
    terms uniformly" semantics; silently redefining that would break them without warning. To
    reproduce RoboNaldo's recipe on a skill: set motion_tracking_reward_scale=1.0 (pins the 5
    relative terms at full strength) and vary ONLY root_tracking_reward_scale (1.0 -> 0.1) across
    resumed runs -- reusing the existing per-skill-yaml-edit-then-resume curriculum mechanism this
    project's shooting_reward_scale already established, not a new in-run ramp.

    Risk, not yet mitigated by this field alone: managers/termination/terms/wbt.py's
    bad_ref_pos/bad_ref_ori check the SAME root-position/orientation quantities at fixed absolute
    thresholds, entirely independent of reward weight. Lowering this field removes the reward pull
    toward the reference root exactly when the policy needs to diverge from it (off-nominal ball
    spawn) -- but does NOT relax the termination boundary that fires if it diverges too far. Tune
    this together with bad_tracking_swing_threshold_multiplier (already exists for widening those
    same thresholds during swing), not independently."""

    recovery_tracking_scale: float = 1.0
    """ADDITIONAL multiplier on this skill's 6 kick-mode motion-tracking reward terms, applied
    ONLY once the clip is past "swinging mode" (motion_command.in_kicking_phase False -- renamed
    from in_swing_phase 2026-07-31, boundary moved to stand_start_idx, the mode-2/mode-3 split) --
    during swing this has no effect (multiplier is 1.0 there regardless of this field's value), so
    motion_tracking_reward_scale alone still governs swing-phase tracking strength. 1.0 (default)
    = current behavior, exact no-op (recovery/hold gets the same full tracking strength as swing,
    same as before this field existed).

    Motivation (2026-07-29): the recovery/hold segment appended after a kick clip's authored
    content is a synthetic interpolation-then-static-hold at the default pose -- it contains no
    information about how to actively balance (no weight shift, no stabilizing step), only where
    to end up. Tracking it at full strength therefore penalizes exactly the corrective motion
    (e.g. planting a step to arrest forward momentum) that would prevent a fall, while rewarding
    passively holding a pose that looks fine until it suddenly isn't -- consistent with a MuJoCo
    sim2sim rollout (model_0380000_mujoco_kick_skill1.mp4) showing low hold-phase tracking error
    right up to a late, sudden collapse. This is the reward-side counterpart to
    bad_tracking_swing_only (which made the same "don't trust the synthetic recovery clip"
    judgment on the termination side) -- what should remain active during recovery/hold is
    kick_alive plus the 6 penalty_kick_recovery_* posture terms (which specify a target STANCE,
    not a target TRAJECTORY), the same alive+posture-only recipe locomotion's own standing
    behavior already relies on. Recommended starting point for an A/B: 0.2-0.3, not 0.0 -- some
    tracking pull during recovery/hold is still useful; the goal is to reduce its authority to
    dictate trajectory, not eliminate posture guidance entirely. NOT applied here by default --
    ships at 1.0, matching this project's convention of landing new mechanisms as a verified
    no-op and letting the value change be a deliberate, separate decision."""

    kick_recovery_posture_reward_scale: float = 1.0
    """Multiplier on this skill's 6 kick-recovery standing-posture penalty terms
    (penalty_kick_recovery_stand_height/orientation/feet_width/knee_width/stance_asymmetry/
    yaw_drift) -- these are already phase-gated to the post-swing recovery/hold tail on their own
    (see _kick_recovery_gate), this scale is an ADDITIONAL per-skill multiplier on top, not a
    replacement for that gating. 1.0 (default) = current behavior, exact no-op."""

    kick_safety_reward_scale: float = 1.0
    """Multiplier on this skill's 3 kick-safety penalty terms (excess foot contact force, hard
    foot landing, excess base linear velocity) -- these already have their own dynamic
    (floor + k * current_w_g(env)) magnitude tied to the shooting-reward ramp; this scale is an
    ADDITIONAL per-skill multiplier applied on top of that existing formula, not a replacement for
    it. 1.0 (default) = current behavior, exact no-op."""

    kick_alive_reward_scale: float = 1.0
    """Multiplier on this skill's kick_alive survival reward (see managers/reward/terms/
    locomotion.py::alive, reused for kick mode). 1.0 (default) = current behavior (weight 10.0
    applies unchanged), exact no-op."""

    kick_alive_pre_kick_ratio: float = 1.0
    """ADDITIONAL multiplier on kick_alive, applied ONLY while motion_command.in_kicking_phase is
    True (approach + strike) -- 1.0 elsewhere (post-kick recovery/hold keeps the full
    kick_alive_reward_scale-scaled value regardless of this field). 1.0 (default) = current
    behavior, exact no-op (flat payout across every phase, same as before this field existed).

    Motivation (2026-08-05, ported from RoboNaldo arXiv:2606.11092): RoboNaldo's own robot_alive
    term (mdp/rewards.py) is phase-shaped, not flat -- 1.0x base reward before
    critic_frame_index+50 (their approach+strike boundary), 10x base after. At their
    reg_weight=0.2 that's an effective ~0.2 pre-kick vs ~2.0 post-kick: pay almost nothing for
    surviving the approach, a lot for surviving the recovery. Our kick_alive is flat 10.0 through
    every phase -- the same payout for hesitating before the kick as for a successful
    stabilization, which is backwards, not merely mis-scaled: a large CONSTANT survival reward
    during approach is a direct incentive to minimize action and never commit to the swing.

    This field only touches the PRE-kick multiplier (kick_alive_reward_scale already controls the
    flat, always-on scale every phase gets); RoboNaldo's ~10x post/pre ratio, applied so as to
    preserve today's unchanged post-kick payout rather than porting their absolute magnitude
    (SAC's auto-tuned entropy temperature is reward-scale sensitive -- shrinking the total budget
    could let the entropy bonus dominate more than intended), works out to
    kick_alive_pre_kick_ratio = 0.1. That is a genuine value change, left for a deliberate,
    separate config edit -- this field ships at 1.0, the exact no-op, same convention as every
    other scale in this class.

    CORRECTION/precision note (verified against RoboNaldo's real source, not re-derived from
    memory): the boundary this field's own gate uses (``motion_command.in_kicking_phase``, i.e.
    ``stand_start_idx``) is this project's analog of RoboNaldo's ``critic_frame_index`` ALONE --
    it does NOT reproduce their additional ``+50`` frame offset (``post_mask = time_steps >
    critic_frame_index + 50``). So the step-up to full value fires ~50 frames EARLIER here than in
    RoboNaldo's own ``robot_alive``. The 1:10 RATIO across the boundary is faithfully ported; the
    exact FRAME the step happens on is not. Same class of simplification as
    ``KickFeetAirTime``'s own ``~stable_phase`` gating (see that class's docstring) -- this
    project's ``stand_start_idx``/``in_kicking_phase`` boundary is used uniformly as the analog for
    every RoboNaldo ``critic_frame_index``-relative boundary ported this session, without
    replicating each term's own individual grace-frame offset on top of it."""

    kick_ankle_pitch_correction_enabled: bool = True
    """Whether MotionCommand.setup()'s self-calibrating kick-foot ankle-pitch correction (see
    managers/command/terms/kick_ankle_pitch_correction.py's own module docstring) runs for THIS
    skill's clip. True (default) = current behavior (the correction is automatic, no yaml field
    was needed to turn it on originally) -- set False per-skill to opt a specific clip out, e.g.
    one whose own authored ankle trajectory doesn't need it, or while comparing a skill's
    trained behavior with/without the correction. Has no effect unless this skill also has
    strike_start_frame/stand_start_frame configured (skills without a real strike window have
    nothing for the correction to act on regardless of this flag)."""

    motion_head_velocity_smoothing_frames: int | None = None
    """2026-08-15, "simultaneous per-skill task configs": per-skill override of MultiSkillConfig.
    motion_head_velocity_smoothing_frames -- see that field's own docstring for the full measured
    rationale. None (default) = no per-skill opinion, inherit whatever the shared/global scalar
    resolves to (task_config-level or the legacy BallConfig default) -- same "None means inherit"
    convention as task_config above.

    Deliberately a SkillConfig field (declared directly on this skill's own motion_skill_N block),
    NOT resolved via a per-skill task_config file the way reward/termination fields are: this is a
    genuinely per-CLIP property (which video2robot-extracted clip has a noisy leading-frame
    velocity spike, not a training-regime choice), so it belongs alongside this skill's other
    clip-authoring metadata (strike_start_frame, kick_foot, ...) -- a skill on a clean clip can
    leave this unset while a noisier one sets 3, independent of which task_config either trains
    under. Threaded through MotionConfig.motion_head_velocity_smoothing_frames_per_motion (a
    COMPILE-TIME per-clip preprocessing param, applied once at MotionCommand.setup() before any
    env exists) -- not a runtime env.skill_id gather, so it has no weight_per_skill/
    params_per_skill counterpart."""

    task_config: str | None = None
    """2026-08-14: which task_config this skill was designed/tuned under -- e.g. 'task_config_
    stageC1' (this fork's configs/task_config_stageC1.yaml, no path, no .yaml extension). Purely
    informational to THIS dataclass (never read by anything downstream of SkillConfig itself) --
    its actual job happens BEFORE any of this file's own loading code ever runs: holosoma/
    __init__.py peeks at HOLOSOMA_SKILLS_CONFIG's raw yaml directly (not through SkillConfig) to
    derive HOLOSOMA_TASK_CONFIG when the caller hasn't set it explicitly, so a training launch
    only needs to export HOLOSOMA_SKILLS_CONFIG. See that module's own docstring for the full
    derivation (including the "every skill must agree" constraint) and why it can't live here as
    ordinary field-parsing logic. None (default) = no opinion; if HOLOSOMA_TASK_CONFIG is also
    unset, existing single-file/legacy behavior applies exactly as before this field existed."""

    kick_aim_theta_max_deg: float | None = None
    """Per-skill override of MultiSkillConfig.kick_aim_theta_max_deg. None (default) = inherit the
    global value. Only meaningful when kick_aim_enabled=True. Validated (> 0.0 and <= the global
    kick_aim_theta_ref_deg) at load_multi_skill_config() time, since that's the first point both
    this skill and the global config are available together."""

    def resolved_target(self) -> tuple[float, float]:
        tx = self.target_x if self.target_x is not None else self.x + 5.0
        ty = self.target_y if self.target_y is not None else self.y
        return (tx, ty)

    def resolved_nominal_bearing_deg(self) -> float:
        """Azimuth (degrees; atan2 convention, 0=+x/forward, positive=+y/the robot's own left,
        matching this dataclass's own x/y docstrings) from this skill's ball nominal spawn (x, y)
        to its nominal target sample point (resolved_target()). Under kick_aim_enabled=True,
        kick_aim_theta=0 means "this direction" -- see target_x's own docstring for why this MUST
        be calibrated against a real rollout (scripts/calibrate_nominal_bearing.py) rather than
        asserted from an authored target_x/target_y, and kick_aim_enabled's own docstring for how
        this value is used to synthesize each attempt's actual target point."""
        tx, ty = self.resolved_target()
        return math.degrees(math.atan2(ty - self.y, tx - self.x))


@dataclass(frozen=True)
class MultiSkillConfig:
    """N-motion-skill training configuration: shared ball physical properties + one SkillConfig
    per stacked motion_skill_N block, in yaml declaration order (index-aligned with motion_ids)."""

    radius: float = 0.11
    """Ball radius in meters, shared by every skill (one physical ball object per env)."""

    mass: float = 0.43
    """Ball mass in kg, shared by every skill."""

    skills: list[SkillConfig] = field(default_factory=list)
    """One entry per motion_skill_N block, in yaml declaration order."""

    shooting_reward_scale_ramp_iters: int = 0
    """Shared (not per-skill) ramp schedule length, mirroring BallConfig's field of the same name:
    each skill ramps from 0 toward ITS OWN shooting_reward_scale over this many control steps,
    under one common schedule. 0 (default) = instant step, matching BallConfig's default."""

    shooting_reward_scale_hold_iters: int = 0
    """Shared (not per-skill) hold-at-zero length before the ramp above starts, mirroring
    BallConfig's field of the same name. 0 (default) = no hold."""

    ball_obs_noise: float = 0.05
    """Shared (not per-skill) flat per-step noise half-range (meters) on kick_ball_pos_b --
    mirrors BallConfig.observation_noise, same default (the pre-2026-07-24 hardcoded value).
    Deliberately global rather than per-skill, unlike ball spawn/target/reward fields: this
    models the robot's PERCEPTION HARDWARE (depth camera + LiDAR + Kalman filter), which doesn't
    change depending on which kick clip is running -- same sensor, same noise, every skill."""

    ball_obs_noise_range_coefficient: float = 0.03
    """Shared (not per-skill) additional noise per meter of ball distance -- mirrors
    BallConfig.observation_noise_range_coefficient. See that field's docstring for the full
    rationale (real depth-sensor error grows with range)."""

    ball_obs_delay_steps_min: int = 0
    """Shared (not per-skill) lower bound (control steps) of the randomized perception-latency
    range on kick_ball_pos_b -- mirrors BallConfig.observation_delay_steps_min."""

    ball_obs_delay_steps_max: int = 3
    """Shared (not per-skill) upper bound (control steps) of the same latency range -- mirrors
    BallConfig.observation_delay_steps_max."""

    ball_obs_hold_steps_min: int = 0
    """Shared (not per-skill) lower bound (control steps) of the randomized zero-order-hold period
    on kick_ball_pos_b -- models the perception/fusion pipeline's own UPDATE RATE being slower
    than the control loop (distinct from ball_obs_delay_steps_{min,max}'s transport LATENCY on a
    pipeline that still updates every tick). E.g. a fixed 25Hz fused ball-pose estimate under this
    project's 50Hz control loop -> both min and max = 2 (round(50/25)); a wider range also models
    the real rate jittering episode-to-episode. Mirrors BallConfig.observation_hold_steps_min. 0
    (default, paired with max=0) = off, update every control tick, bit-identical to before this
    field existed. See ObsTermCfg.hold_step_range / ObservationManager._apply_hold."""

    ball_obs_hold_steps_max: int = 0
    """Shared (not per-skill) upper bound (control steps) of the same hold-period range -- mirrors
    BallConfig.observation_hold_steps_max."""

    ball_obs_stale_probability: float = 0.0
    """Shared (not per-skill) per-CONTROL-TICK probability (re-rolled every step, unlike
    ball_obs_hold_steps_{min,max}'s per-episode-fixed period) that kick_ball_pos_b reuses the
    PREVIOUS tick's already-fully-processed reading instead of this tick's -- models a single
    dropped/repeated sensor frame (e.g. one skipped LiDAR packet), a transient, self-correcting
    fault distinct from ball_obs_hold_steps_{min,max}'s deterministic slower-update-rate modeling
    AND from ball_static_obs_probability's own per-episode freeze-for-the-rest-of-the-episode-to-
    an-independently-drawn-value mechanism (that one simulates a fully broken sensor; this one
    simulates one dropped packet that self-corrects as soon as a tick isn't drawn stale again).
    Cross-checked against RoboNaldo's own arXiv:2606.11092 source
    (mdp/commands.py::apply_lidar_stale_ball_pos_b, lidar_stale_probability): they stage this at
    0.0 through their Stage 1/2 (task_params_1/2.yaml) and only 0.01 once Stage 3 (moving-ball)
    begins (task_params_3.yaml) -- 0.01 is a reasonable starting point to try here too, NOT yet
    validated against a live run in this project. Mirrors BallConfig.ball_obs_stale_probability.
    See ObsTermCfg.stale_probability / ObservationManager._apply_stale. 0.0 (default) = off, a
    true no-op bit-identical to before this field existed."""

    ood_spawn_probability: float = 0.0
    """Shared (not per-skill) per-reset probability of an out-of-distribution ball spawn --
    mirrors BallConfig.ood_spawn_probability / MotionConfig.ood_spawn_probability (see that
    field's own docstring for the full rationale). 0.0 (default) = off."""

    ood_region_multiplier: float = 3.0
    """Shared (not per-skill) OOD region size, as a multiple of each skill's own randomize_x/y --
    mirrors BallConfig.ood_region_multiplier / MotionConfig.ood_region_multiplier."""

    ball_static_obs_probability: float = 0.0
    """Shared (not per-skill) per-reset probability that kick_ball_pos_b freezes at its
    first-post-reset (already noise/bias-perturbed) reading for the entire episode, instead of
    updating live -- models a stuck/dead perception pipeline at deployment (sensor dropout, stale
    cached reading). Mirrors BallConfig.ball_static_obs_probability. Only kick_ball_pos_b freezes
    -- kick_target_pos_b (a command, not a sensor reading) stays live. See
    managers/observation/terms/unified.py::ball_pos_b and
    managers/randomization/terms/locomotion.py::randomize_ball_obs_freeze. 0.0 (default) = off."""

    kick_contact_force_penalty_floor: float = 3.0
    """Shared (not per-skill) base magnitude of ``kick_penalty_excess_contact_force``, active
    regardless of any skill's shooting_reward_scale -- see
    managers/reward/terms/kick_safety.py::penalty_excess_foot_contact_force's docstring for the
    full ``(floor + k * current_w_g(env))`` dynamic-magnitude design. Deliberately GLOBAL, not
    per-skill: this penalizes a physical property of foot/ground contact (a sim2real safety
    concern), not skill-specific content like ball spawn/target. 3.0 (default) matches the
    pre-2026-07-28 hardcoded ``_KICK_SAFETY_FLOOR`` in config_values/unified/g1/reward.py, so an
    unset config is bit-identical to before this field existed."""

    kick_contact_force_penalty_k: float = 15.0
    """Shared (not per-skill) coefficient on the LIVE current_w_g(env) value, added to the floor
    above -- see kick_contact_force_penalty_floor's docstring and
    penalty_excess_foot_contact_force's own docstring for the full rationale. 15.0 (default)
    matches the pre-2026-07-28 hardcoded value.

    2026-07-28 retune context (read before changing): raising shooting_reward_scale from 0.8 to
    3.0 was already tried once, WITH this floor+k coupling already active (k=15 then too) --
    raw_rew_kick_penalty_excess_base_lin_vel still grew 4.7x and MuJoCo sim2sim survival collapsed
    (train-conditions topple rate 32.0% -> 54.7%, 0/128 envs surviving vs 22.7% before) -- see
    config_values/unified/g1/reward.py's ``_kick_safety_terms`` comment block for the full
    measurement. Scaling the penalty in lockstep with w_g does NOT, by itself, prevent the
    momentum-throw exploit, because shooting harder and destabilizing contact are physically
    coupled, not just reward-coupled. Any future w_g increase paired with a k this large will
    reproduce that failure; lowering k (this field) is what actually gives shooting more relative
    room without also multiplying the safety cost by the same factor."""

    kick_contact_force_threshold_bodyweight_multiplier: float = 3.0
    """Shared (not per-skill) per-foot ground-reaction-force threshold, as a multiple of the
    robot's own bodyweight, above which ``kick_penalty_excess_contact_force`` starts penalizing --
    see penalty_excess_foot_contact_force's docstring for why 3x is "generous headroom" (athletic
    single-leg loading commonly reaches 2-4x bodyweight). 3.0 (default) matches the pre-2026-07-28
    hardcoded value."""

    start_at_timestep_zero_prob: float = 1.0
    """Shared (not per-skill) counterpart of MotionConfig.start_at_timestep_zero_prob, specifically
    for the unified locomotion+kick task (config_values/unified/g1/command.py's
    ``_motion_config_unified_kick``). 1.0 (default) matches the pre-2026-07-28 hardcoded value --
    every kick-mode episode starts at the assigned skill's own frame 0 (post-prepend: the eased
    standing->windup ramp), never a mid-clip phase.

    2026-07-28 retune context: this was set to 1.0 early in the project because an UNDERTRAINED
    policy teleported to a random mid-clip pose (one leg extended) fell almost immediately --
    EVERY kick episode terminated within 4-7 steps, so kick collected ~1% of transitions and never
    bootstrapped (see config_values/unified/g1/command.py's own comment for the full account). At
    a MATURE checkpoint (445k-800k range, run 5yq6yh5a) the concern this guarded against may no
    longer hold, and real telemetry from that same run shows a genuine "can't escape conservatism"
    signature worth addressing: kick tracking error plateaued for the full 355k-step window
    (error_body_pos_swing 0.070->0.071, no improvement; error_joint_pos_hold WORSENING 1.09->1.53)
    while the policy's own action_std stayed flat (~0.0596) and policy_entropy kept DECREASING
    (-0.166->-0.303, i.e. becoming MORE deterministic over time, not less) -- consistent with a
    policy that has very little exploration mechanism left (target_entropy_ratio=0.0, shared with
    locomotion, drives toward a fully deterministic policy) and, on top of that, zero state-space
    diversity during training (this field at 1.0). Lowering it gives the policy some fraction of
    resets starting mid-clip -- real training-time diversity in what states it has to recover
    from/track from -- WITHOUT touching the shared SAC entropy machinery (target_entropy_ratio=0.5
    was already tried project-wide and broke locomotion, see experiment.py's own comment; this is
    a deliberately different, lower-risk lever).

    IMPORTANT -- do NOT combine with use_adaptive_timesteps_sampler=True in N-skill mode:
    AdaptiveTimestepsSampler.sample_global_time_steps samples a GLOBAL frame index across ALL
    skills' concatenated clips and DERIVES motion_ids from wherever it lands (wbt.py reset(),
    ``self.motion_ids[env_ids] = motion_ids``), silently OVERWRITING each env's permanent
    per-skill assignment (fixed_motion_ids) from UnifiedManager._build_task_mode_partition -- an
    env could be reassigned to a completely different skill than the one its shooting_reward_scale
    /ball spawn/target were configured for. This is the "AdaptiveTimestepsSampler + fixed-skill
    combo" the original N-skill plan explicitly deferred as unimplemented; it still is. This
    field's own prob<1.0 path is NOT affected by that bug -- it only decides whether to snap an
    already-correctly-within-assigned-skill uniform phase back to frame 0, never touches
    motion_ids -- so it's safe to lower on its own, just don't ALSO flip on the adaptive sampler."""

    rsi_scope_to_authored_clip: bool = False
    """Shared (not per-skill), 2026-08-05, ported from RoboNaldo (arXiv:2606.11092)'s
    ``start_time_sampling_fraction`` (see ROBONALDO_PORT_SCOPE.md Sec 4). False (default) =
    current behavior, exact no-op: the ``start_at_timestep_zero_prob``-driven uniform-phase RSI
    draw above spans the WHOLE augmented per-motion buffer, including the synthetic recovery-lerp
    + static-hold tail spliced on after each authored clip (see MotionCommand.setup()'s
    ``_maybe_add_default_pose_transition``/``_maybe_add_post_transition_hold``) -- so a mid-clip
    RSI reset can land inside that synthetic tail, which carries no real kicking content. When
    True, the uniform draw's span is instead clamped to ``pre_recovery_motion_end_idx`` (already
    computed in ``setup()`` -- the exact frame where a motion's real authored content ends and the
    synthetic tail begins), scoping RSI to authored content only, matching RoboNaldo's own
    ``start_time_sampling_fraction`` (which is defined over their single authored clip, with no
    synthetic tail to accidentally land in). Does not change ``start_at_timestep_zero_prob``'s own
    semantics or value -- purely narrows WHERE the non-zero-prob branch's uniform draw can land."""

    critical_frame_oversampling_prob: float = 0.0
    """Shared (not per-skill), 2026-08-05, ported from RoboNaldo (arXiv:2606.11092)'s
    ``critical_frame_adaptive_sampling`` (see ROBONALDO_PORT_SCOPE.md Sec 4). 0.0 (default) = exact
    no-op: RSI resets always use the plain uniform-in-clip draw. When > 0.0, each RSI reset has
    this probability of instead being drawn from a fixed window around the assigned motion's own
    ``strike_start_idx`` (``critical_frame_sampling_window`` frames on either side, clamped to the
    motion's own valid range) -- oversampling practice time around the kick itself, not just the
    approach. Applied in MotionCommand.reset() BEFORE the start_at_timestep_zero_prob block, so
    prob=1.0 there can still deterministically override it -- mirrors RoboNaldo's own S1, which
    runs both ``start_time_sampling_fraction=1.0`` and ``critical_frame_adaptive_sampling=true``
    simultaneously. Must be in [0.0, 1.0]; validated at load time.

    NOT the same mechanism as ``use_adaptive_timesteps_sampler``/``AdaptiveTimestepsSampler`` --
    deliberately NOT reusing that class, which is hard-disabled in N-skill mode for a real reason
    (see start_at_timestep_zero_prob's own docstring: it derives motion_ids from a GLOBAL sampled
    frame index, silently overwriting each env's permanent per-skill assignment from
    UnifiedManager._build_task_mode_partition). This field's window draw stays strictly within the
    env's ALREADY-assigned motion_id -- it only chooses WHERE in that fixed clip to start, never
    which clip."""

    critical_frame_sampling_window: int = 10
    """Shared (not per-skill) half-width (in frames) of the oversampling window around
    ``strike_start_idx``, used only when ``critical_frame_oversampling_prob`` > 0.0 -- see that
    field's own docstring. 10 (default) matches RoboNaldo's own ``critical_frame_sampling_window``
    value. Must be >= 0; validated at load time."""

    motion_head_velocity_smoothing_frames: int = 0
    """Shared (not per-skill) counterpart of MotionConfig.motion_head_velocity_smoothing_frames --
    see that field's docstring (config_types/command.py) and
    managers/command/terms/motion_head_velocity_smoothing.py's own module docstring for the
    measured motivation. 0 (default) = OFF, exact no-op: every existing config is unaffected
    unless this is explicitly set.

    Deliberately GLOBAL rather than per-skill, matching how the prepend it exists to smooth into
    is already configured (see config_values/unified/g1/command.py's own note: "Prepend duration
    is NOT per-skill (not part of the yaml schema) -- every skill gets the same 1.0s windup").
    Must be >= 0; validated at load time."""

    penalty_curriculum_enabled: bool = True
    """FIX 1 (2026-08-12): whether PenaltyCurriculum (managers/curriculum/terms/locomotion.py)
    ramps penalty_curriculum-tagged reward weights based on average_episode_length. True (default)
    = current behavior, unchanged. Threaded into config_values/unified/g1/curriculum.py's
    ``enabled`` param -- see that file's own "FIX 1" comment block for the full measured rationale
    (a live diagnostic found Env/penalty_scale led Env/kick_alive_frac at lag+10 with r=-0.753,
    consistent with the curriculum ramping up while post-flip kick-partitioned envs -- which
    inherit penalty_curriculum-tagged _penalty_stand_* terms once flipped, see reward.py -- are
    scaled by a signal, average_episode_length, that only the genuinely-locomotion-partitioned
    population can move).

    Setting this False makes PenaltyCurriculum.setup()/.reset() both no-ops (that class's own
    early-return on `not self.enabled`), so every tagged term stays at its full, un-scaled yaml
    weight from the start -- intended for resuming an already-mature checkpoint, where the
    ramp-from-easy rationale this curriculum exists for no longer applies. Global, not per-skill:
    this is a property of the reward-weight schedule, not clip-specific content."""

    post_flip_termination_grace_steps: float = 0.0
    """FIX 2 (2026-08-12): grace window (ticks) after UnifiedManager's kick->locomotion flip
    (kick_recovery_locomotion_flip_enabled) during which the locomotion-mode `contact`/
    `low_height` termination terms are suppressed for that env -- see managers/termination/terms/
    locomotion.py's contact_forces_exceeded_post_flip_graced / _post_flip_grace_active for the
    mechanism and the full measured rationale (a live diagnostic found `contact` alone -- which has
    NO grace/sustained-duration concept -- responsible for 600/643, 93.3%, of post-flip
    terminations; a still-asymmetric, momentum-carrying post-kick pose handed straight to
    locomotion mode is an instant kill on the first incidental >1N contact).

    0.0 (default) = exact no-op: contact_forces_exceeded_post_flip_graced/
    base_height_below_threshold_sustained_post_flip_graced are registered in place of the plain
    `contact`/`low_height` for the unified config (config_values/unified/g1/termination.py), but
    at 0.0 they are BIT-IDENTICAL to calling the un-graced originals directly (see those wrappers'
    own docstrings for why this is guaranteed, not just typical). 50.0 (1.0s at dt=0.02) mirrors
    the grace this project already uses at the OTHER kick-recovery boundary
    (kick_recovery_low_height_sustained/kick_recovery_drift_sustained's own grace_steps, itself
    mirroring _kick_recovery_gate) -- same idea, applied to the symmetric boundary those don't
    cover. Global, not per-skill: a property of the flip mechanism, not clip-specific content.
    Must be >= 0.0; validated at load time."""

    post_flip_reward_decay_steps: float = 0.0
    """FIX 4 (2026-08-12): ticks over which motion-tracking reward (all 7 wrapped terms in
    managers/reward/terms/kick_scale_wrappers.py) ramps linearly to zero AFTER the kick->
    locomotion flip, instead of the instant cliff every existing config runs under today
    (task_mode_mask("kick") zeroing the whole reward the tick task_mode changes). See
    _post_flip_tracking_decay_multiplier's own docstring (kick_scale_wrappers.py) for the full
    measured rationale: SAC's critic bootstraps through the whole trajectory, so a live diagnostic
    (matched FLIP-vs-NOFLIP checkpoints, same clip, same pre-boundary mechanism in both) found
    termination-condition firing rates 2.5x-6.6x higher for the FLIP-trained policy in a window
    that exists identically in both conditions -- i.e. training under the instant cliff
    measurably destabilizes behavior even before the boundary is reached.

    0.0 (default) = exact no-op: these 7 terms stay at task_mode="kick" (config_values/unified/
    g1/reward.py's _apply_post_flip_reward_decay only switches them to task_mode=None when this
    is > 0), so RewardManager's own masking already zeroes every post-flip env's contribution
    regardless of this value -- the decay multiplier itself also short-circuits to a bare python
    float 1.0 at this default, so multiplying by it cannot introduce even a rounding difference.
    50.0 (1.0s) is a reasonable starting point matching post_flip_termination_grace_steps's own
    scale, not independently measured. Global, not per-skill: a property of the flip mechanism's
    credit-assignment horizon, not clip-specific content. Must be >= 0.0; validated at load time."""

    kick_target_entropy_ratio: float | None = None
    """Shared (not per-skill) counterpart of FastSACConfig.kick_target_entropy_ratio -- see that
    field's docstring for the full design (opt-in per-task-mode SAC entropy target: locomotion
    keeps using the algo-level target_entropy_ratio, kick-mode transitions use THIS value instead,
    via a second, independently-optimized log_alpha parameter). None (default) = OFF, bit-
    identical to today's single shared alpha -- every existing config unaffected unless this is
    explicitly set. Deliberately GLOBAL, not per-skill: this is a property of the SAC algorithm's
    exploration mechanism (locomotion vs kick, not clip-by-clip), not skill-specific content."""

    kick_gamma: float | None = None
    """Shared (not per-skill) counterpart of FastSACConfig.kick_gamma -- see that field's
    docstring for the full design (opt-in per-task-mode SAC discount factor: locomotion keeps
    using the algo-level gamma, kick-mode transitions bootstrap with THIS value instead). None
    (default) = OFF, bit-identical to today's single shared gamma. Deliberately GLOBAL, not
    per-skill, same rationale as kick_target_entropy_ratio above -- discounting is a property of
    the SAC algorithm's credit-assignment horizon, not skill-specific content.

    Motivation (2026-07-30): kick mode's effective horizon at the shared gamma=0.97 is
    1/(1-0.97)=33 control steps = 0.67s. Measured: a kick clip's authored swing spans ~4.6s and its
    recovery+hold tail another ~3s, and a phase-resolved fall probe found roughly a third of falls
    occur in that recovery/hold tail -- one to three seconds after the swing decisions that likely
    caused them, well beyond a 0.67s bootstrap horizon. The policy cannot receive credit/blame for
    a fall that far past its discount window; every reward-WEIGHT change tried against this same
    symptom this session (bad_tracking relaxation, kick_alive cut, capture-point shaping) either
    backfired or was neutral, because none of them touch the underlying credit-assignment horizon.
    Raising kick_gamma to e.g. 0.99 (100-step / 2.0s horizon) is a reasoned hypothesis, NOT yet
    validated by training."""

    critic_v_min: float | None = None
    """Shared (not per-skill) counterpart of FastSACConfig.v_min -- the LOWER bound of the
    distributional critic's value support. None (default) = leave the experiment preset's own
    value (-20.0 for g1_29dof_unified_fast_sac) untouched, bit-identical to before this field
    existed.

    2026-08-21, added after probing run 20260821_022846 found the support SATURATING at both ends:
    ``Loss/qf_max`` sat at 19.99989 (i.e. exactly on the +20.0 ceiling) and ``qf_min`` at -19.08,
    while the arithmetic says a merely GOOD episode overflows it -- mean per-step env reward 0.1033
    over a 685-step kick episode at kick_gamma=0.99 gives a discounted return of ~10.3, so an
    episode earning twice the mean return lands at ~20.6, OUTSIDE the support. Returns beyond the
    bound are clipped by the categorical projection, so the critic cannot rank "good" against
    "great" -- flattening exactly the gradient that would push kick performance past the plateau
    that run measured (hit rate rose to ~0.27 then stopped).

    Deliberately GLOBAL, not per-skill: the value support is a property of the SAC critic's own
    parameterization, not of skill-specific content -- same rationale as kick_gamma /
    kick_target_entropy_ratio above.

    RESUMING A CHECKPOINT: see critic_num_atoms' docstring for the two distinct compatibility
    hazards (a silent one for v_min/v_max, a loud one for num_atoms). Read it before setting any
    of these three on a run that resumes."""

    critic_v_max: float | None = None
    """Shared (not per-skill) counterpart of FastSACConfig.v_max -- the UPPER bound of the same
    support. None (default) = leave the experiment preset's own value (+20.0) untouched. See
    critic_v_min's docstring for the measurement that motivated making these configurable."""

    critic_num_atoms: int | None = None
    """Shared (not per-skill) counterpart of FastSACConfig.num_atoms -- how many atoms discretize
    [critic_v_min, critic_v_max]. None (default) = leave the experiment preset's own value (101)
    untouched.

    Bin width is ``(v_max - v_min) / (num_atoms - 1)``: the preset's 101 atoms over [-20, 20] give
    0.400. WIDENING THE SUPPORT WITHOUT RAISING num_atoms COARSENS RESOLUTION -- [-40, 40] at 101
    atoms is 0.800 per bin. Raising num_atoms to 201 restores 0.400 exactly.

    TWO DISTINCT CHECKPOINT HAZARDS, both verified against a real saved checkpoint:

    1. ``num_atoms`` changes TENSOR SHAPES -- the critic's final layer is
       ``nn.Linear(hidden_dim // 4, num_atoms)`` (agents/fast_sac/fast_sac.py), so a saved
       ``qnets.N.net.9.weight`` is (num_atoms, 192) and ``q_support`` is (num_atoms,). Changing it
       makes ``qnet.load_state_dict`` raise on shape mismatch (it is called WITHOUT strict=False),
       so a run that changes num_atoms CANNOT resume from an existing checkpoint at all. This one
       fails LOUDLY, which is the safe direction.

    2. ``v_min``/``v_max`` do NOT change shapes, so the checkpoint loads cleanly -- but
       ``q_support`` is a REGISTERED BUFFER (``self.register_buffer("q_support", torch.linspace(
       v_min, v_max, num_atoms))``), so it is IN the checkpoint, and ``load_state_dict``
       OVERWRITES the freshly-built support with the checkpoint's old one. Without the explicit
       re-derivation FastSACAgent.load() now performs (see its own ``_restore_critic_support``
       comment), setting these on a resumed run would be a SILENT no-op. That re-derivation logs
       loudly when the values differ, because rescaling the support re-interprets every learned
       atom probability -- the critic's value estimates are effectively rescaled and it must
       re-fit, even though the actor is preserved intact."""

    replay_buffer_sanitize_enabled: bool = False
    """Shared (not per-skill) counterpart of FastSACConfig.replay_buffer_sanitize_enabled -- see
    that field's docstring for the full design (opt-in NaN/Inf guard at the SAC replay buffer's
    write boundary: torch.nan_to_num applied to observations/rewards/next_observations/critic
    observations right before they're written into the circular buffer). False (default) = OFF,
    bit-identical to today's unsanitized writes. Deliberately GLOBAL, not per-skill: this guards
    against a rare per-env physics-solver numerical explosion, a simulator/training-infrastructure
    concern, not skill-specific content."""

    l2sp_weight: float = 0.0
    """Shared (not per-skill) counterpart of FastSACConfig.l2sp_weight -- see that field's
    docstring for the full design and for the value table (0.0 = OFF / 0.001 / 0.01 = start here /
    0.1 = practical upper end). 0.0 (default) = OFF, bit-identical to today's training.

    ALWAYS read from HOLOSOMA_SKILLS_CONFIG's OWN file (e.g. multi_skills.yaml), never from
    HOLOSOMA_TASK_CONFIG's file even when 2-file mode is active for the rest of the global fields
    -- same "always raw, never task_raw" carve-out base_robot gets, and for the same reason: this
    field exists to protect whichever skills THIS skill roster names, so it belongs with the
    roster, not with a task-config file that could be paired with a different roster entirely. See
    _parse_skill_replay_and_l2sp_fields' own docstring.

    Deliberately GLOBAL, not per-skill: L2-SP anchors the ONE shared Actor's parameters toward the
    resumed checkpoint's values. There is no per-skill subset of those parameters to anchor
    separately (no per-skill heads exist), so a per-skill weight would have nothing distinct to
    act on. Which SKILLS are being protected is expressed by the skill roster + each skill's
    motion_training_ratio; how hard they are protected is this single number.

    Only meaningful when RESUMING (the anchor is the checkpoint you resume from). Prefer joint
    training over all skills at once where you can afford it -- that avoids catastrophic
    forgetting outright, and this knob should stay 0.0."""

    skill_replay_weights: list[float] = field(default_factory=list)
    """Shared (not per-skill-file) counterpart of FastSACConfig.skill_replay_weights -- see that
    field's docstring for the full rationale and the weight formula. Empty (default) = OFF,
    bit-identical to today's training.

    ALWAYS read from HOLOSOMA_SKILLS_CONFIG's OWN file, same always-raw carve-out as l2sp_weight
    directly above (see that field's own note and _parse_skill_replay_and_l2sp_fields' docstring)
    -- doubly appropriate here since this field is literally indexed by skill_id against THIS
    file's own motion_skill_N roster.

    The DATA-side sibling of l2sp_weight above: l2sp_weight protects an old skill from the
    PARAMETER side (anchor the actor to the checkpoint it resumed from), this protects it from the
    DATA side (keep its share of every gradient batch high even when its motion_training_ratio is
    low). They address the same forgetting problem through independent mechanisms and can be used
    together.

    Indexed by skill_id, so its length must equal the number of motion_skill_N blocks. To equalize
    gradient across skills set each weight proportional to 1/motion_training_ratio_i (e.g. ratios
    0.1/0.8 -> [8.0, 1.0]); to merely soften the imbalance, use something smaller (e.g.
    [4.0, 1.0]). Validated at load time."""

    joint_pos_sanity_check_enabled: bool = False
    """Shared (not per-skill) switch for a new termination term (joint_pos_sanity, see
    managers/termination/terms/locomotion.py::joint_pos_sanity_exceeded and
    config_values/unified/g1/termination.py's registration) that resets an env whose
    ``env.simulator.dof_pos`` has gone non-finite (NaN/Inf) or exceeds
    ``joint_pos_sanity_threshold`` in absolute value. False (default) = OFF, term not registered
    at all -- bit-identical to before this field existed.

    Deliberately task_mode-agnostic (registered untagged, applies to every env regardless of
    current task_mode) and deliberately DIFFERENT from ``BadTracking``: BadTracking is
    reference-relative (how far the robot has deviated from the motion clip -- can and does fire
    for legitimate divergence, e.g. chasing an off-nominal ball) and checks a comparison-based
    threshold, which silently evaluates False for a true NaN input (`nan > threshold` is False in
    IEEE 754 / PyTorch semantics) -- so a reference-tracking check alone CANNOT catch this failure
    mode. This term instead checks the robot's OWN joint state against an absolute physical sanity
    bound, `torch.isfinite` first specifically to catch NaN/Inf that a plain magnitude check would
    miss.

    Motivation: directly observed corrupting a real Stage C run (2026-08-10) --
    `kick_motion/error_joint_pos_swing` spiked to 2.36e8 for exactly one logged tick (a rare
    per-env contact/collision-resolution edge case, most likely during the high-velocity swing --
    the phase already established as the dominant instability source, see
    bad_tracking_swing_only's own docstring), with `Loss/qf_loss` going to NaN at the same step.
    Because reward/termination are computed in the SAME tick, BEFORE any reset (this codebase's
    established step order), this termination alone cannot prevent that tick's already-corrupted
    transition from reaching the replay buffer -- see replay_buffer_sanitize_enabled above for the
    complementary fix that actually stops the NaN from reaching the learner. This term's own value
    is shortening how long a genuinely broken env's state persists, not preventing the one
    already-corrupted tick. NOT yet validated by a training run -- ships as an opt-in A/B."""

    joint_pos_sanity_threshold: float = 20.0
    """Absolute per-joint-angle bound (radians) joint_pos_sanity_check_enabled's termination
    compares ``env.simulator.dof_pos`` against -- see that field's own docstring. 20.0 is several
    multiples beyond ANY physically plausible G1 joint excursion (real joint ranges are within
    roughly +/-3.14 rad), chosen to never fire on legitimate motion, only a genuine numerical
    explosion. Only read when joint_pos_sanity_check_enabled is True."""

    bad_tracking_swing_only: bool = False
    """Shared (not per-skill) switch: when True, ``bad_tracking`` can only fire while a kick
    clip is still in "swinging mode" (``motion_command.in_kicking_phase``) -- fully suppressed
    for post-kick-standing mode and the ENTIRE recovery/hold segment, not just an early window
    after swing ends. False (default) = off, bit-identical to before this field existed.

    2026-07-31 boundary-move note: ``in_kicking_phase`` (renamed from ``in_swing_phase``) now ends
    at ``stand_start_idx`` (the mode-2 strike / mode-3 post-kick-standing split), NOT at the whole
    authored clip's own end as ``in_swing_phase`` did. If this field is ever turned on, its
    "swing" window therefore SHRINKS relative to before this rename: mode 3 (post-kick-standing,
    still real authored content, not synthetic) newly falls OUTSIDE the protected window, under
    the same full-suppression relaxation as the actually-synthetic recovery/hold tail. This field
    ships at its no-op default (False) specifically because the swing-window widening below
    (``bad_tracking_swing_threshold_multiplier``) was already measured to backfire when combined
    with a widened "swing" -- read that field's MEASURED OUTCOME note before enabling either.

    Motivation (2026-07-29 investigation, see managers/termination/terms/wbt.py::BadTracking and
    this file's own module docstring "POST-SWING RELAXATION" section for the full account): a
    genuinely hard, committed kick leaves the robot in a post-strike state the recovery/hold
    segment's own authored trajectory doesn't anticipate -- that segment is a synthetic
    interpolation back to the default pose, not a recording of a real hard-kick recovery -- so
    real residual momentum needing to be arrested reads as clip divergence, not instability. Live
    measurement (checkpoint model_0395000, run 20260729_014242-unified-stageC-2skills-locomotion):
    attempts with harder-than-median contact speed complete the full clip (through the end of
    hold) only 26.5% of the time vs 42.1% for softer-than-median contacts (correlation -0.18,
    n=1100 kicked attempts), and post-swing ``bad_tracking``'s mean triggering contact speed
    (2.12 m/s) is nearly double the speed of attempts that survive to timeout (1.12 m/s) --
    consistent with hard kicks being disproportionately truncated by this term before the policy
    ever completes (and learns from) a full post-kick stabilization. A bounded grace-period
    version of this (``bad_tracking_recovery_grace_steps``, superseded by this field) was tried
    first but abandoned: the completion-fraction shortfall was measured spread across the WHOLE
    recovery/hold window (554 of 723 cut-short attempts landed between 50%-98% clip completion,
    not clustered right after swing end), so a short grace window under-covers the actual problem.

    Separately, harder contact does NOT predict more genuine falls (``kick_low_height``, an
    absolute-height, clip-independent termination) -- if anything fewer (6.7% vs 19.6% fall rate)
    -- so this is specifically a clip-tracking artifact, not a stability problem.
    ``kick_low_height`` remains the sole termination-level fall backstop for the whole
    recovery/hold window when this is True -- a DELIBERATE tradeoff: orientation/lean divergence
    from the clip (``bad_ref_ori``, not just the height checks) is also suppressed post-swing
    under this field, so a badly-leaning-but-not-yet-below-0.40m robot is no longer terminated for
    it post-swing, only reward-penalized via ``penalty_kick_recovery_stand_orientation``. This
    project has previously seen a reward-tax-only deterrent prove insufficient on its own (the
    standing-crouch floor termination was added specifically because ``alive`` made a height tax
    "a cost of doing business" rather than a real constraint) -- the same risk applies here for
    orientation, accepted as a known, explicit tradeoff rather than an oversight. See a Stage-C
    sim2sim MuJoCo rollout (``model_0380000_mujoco_kick_skill1.mp4``) for the motivating
    observation: a clean, committed strike followed by an apparently stable ~2s recovery, then a
    late collapse well into hold."""

    bad_tracking_swing_threshold_multiplier: float = 1.0
    """Shared (not per-skill) factor widening (NOT removing) ``bad_tracking``'s
    bad_ref_pos/bad_ref_ori/bad_motion_body_pos thresholds for envs currently in "swinging mode"
    (``motion_command.in_kicking_phase`` True). 1.0 (default) = off, bit-identical to before this
    field existed -- values > 1.0 make the check more tolerant during swing specifically; the
    post-swing recovery/hold window is governed independently by ``bad_tracking_swing_only`` above
    (or the plain, un-widened thresholds if that's off).

    2026-07-31 boundary-move note: same as ``bad_tracking_swing_only`` above -- ``in_kicking_phase``
    now ends at ``stand_start_idx`` (mode-2/mode-3 split), not the whole clip's own end, so mode 3
    (post-kick-standing) newly falls OUTSIDE this widened-threshold window if this field is ever
    raised above 1.0, joining the un-widened regime governed by ``bad_tracking_swing_only``
    instead. Nothing numerically changes while this stays at 1.0 (see MEASURED OUTCOME below for
    why it should).

    Motivation (2026-07-29): reaching an off-nominal ball (randomized per skill, e.g.
    randomize_x/y=0.75 in stageC_2skills.yaml) requires the robot's real trajectory to genuinely
    diverge from a reference clip that has no idea where the ball actually is -- that divergence
    is legitimate, not instability, and is the whole point of Stage C (a policy that can't deviate
    can't adapt its strike to where the ball is). Live measurement (contact-speed-vs-termination-
    type correlation, this project's own investigation): of 244 sampled ``bad_tracking``
    terminations, 169 fired DURING swing vs 75 post-swing -- swing is the LARGER source of
    tracking-deviation terminations, not the smaller one ``bad_tracking_swing_only`` already
    addresses.

    Deliberately a WIDENING, not a swing-phase version of ``bad_tracking_swing_only``'s full
    removal: unlike the post-swing recovery/hold segment (a synthetic interpolation with no
    balance information), the authored swing content still carries real, non-ball-related guidance
    -- general windup shape, single-support balance, self-collision avoidance -- that a full
    removal would discard along with the ball-chasing benefit. Applies to ALL THREE checks
    uniformly, including orientation: unlike the recovery/hold case (a full removal, where
    orientation was deliberately left un-relaxed since nothing else backstops it), this is a
    modest widening with ``kick_low_height`` (absolute height, clip-independent) still fully
    active throughout swing regardless of this field's value -- a genuine fall is still caught.
    Recommended starting point for an A/B: 1.5-2.0 (50-100% more tolerance), not an extreme value
    -- a reasoned starting point, not yet measured against live telemetry. NOT applied here by
    default -- ships at 1.0, matching this project's convention of landing new mechanisms as a
    verified no-op and letting the value change be a deliberate, separate decision.

    MEASURED OUTCOME (2026-07-30) -- this field made things WORSE and should stay at 1.0. An
    accidental 8-run matched-checkpoint experiment gave a monotonic dose-response in the opposite
    direction to the reasoning above: strict (this=1.0, swing_only off) per-cycle fall hazard
    0.058-0.076, swing_only alone 0.100-0.130, swing_only + this=2.0 **0.240**, with hit rate
    statistically identical across all three (0.243-0.279). Ruled out measurement censoring three
    ways (episode length flat 600-671, early_term_frac flat, and the CONTINUOUS kick_min_base_height
    degrading 0.644 -> 0.522). The premise was not wrong that swing dominates tracking terminations
    -- it was wrong that those terminations were false positives worth suppressing."""

    kick_recovery_termination_handoff: bool = False
    """Shared (not per-skill) switch: when True, ADDS ``kick_recovery_low_height_sustained`` (a
    locomotion-style height-only check) and ``kick_recovery_drift_sustained`` (a base-drift check,
    see both in ``managers/termination/terms/wbt.py``) to the post-kick recovery/hold window.
    False (default) = off, bit-identical to before this field existed.

    2026-08-06, user-requested DECOUPLING from ``bad_tracking_swing_only`` -- READ BEFORE RELYING
    ON THE ORIGINAL 2026-08-02 BEHAVIOR DESCRIBED BELOW, it no longer applies by default. Until
    this date, setting this flag ALSO forced ``bad_tracking_swing_only`` (above) True for the same
    envs (ORed into that field's own resolution in
    ``config_values/unified/g1/termination.py``), making this a strict REPLACEMENT: ``bad_tracking``
    fully suppressed for recovery/hold, height check installed in its place. That coupling is
    exactly what the MEASURED 2026-08-02 CONCERNING note below found bad -- a ~50-60 step window
    where NOTHING was watching (bad_tracking already suppressed, the height check's own grace
    period not yet engaged). Per explicit user direction, the forcing is now REMOVED: this flag
    installs its two terms WITHOUT touching ``bad_tracking_swing_only`` at all, so ``bad_tracking``
    stays fully active during recovery/hold by default, directly closing that gap instead of
    reproducing it. To get the ORIGINAL measured configuration back (bad_tracking suppressed,
    height check only) for regression/reference, set BOTH flags explicitly:
    ``kick_recovery_termination_handoff: true`` AND ``bad_tracking_swing_only: true`` -- the latter
    remains independently settable on its own, unchanged (see its own docstring's MEASURED OUTCOME
    note: shipping it ALONE, no replacement, was separately measured to make things worse,
    per-cycle fall hazard 0.058-0.076 -> 0.100-0.130 -- see
    ``kick_recovery_low_height_sustained``'s own docstring for that mechanism). The new default
    (bad_tracking active + height + drift, all three simultaneously) is a genuinely new,
    never-before-tried configuration -- UNMEASURED, not assumed safe because each individual piece
    has been examined separately.

    Motivation (2026-08-02 investigation): decomposing ``bad_tracking`` into its three sub-checks
    against live checkpoint replay (checkpoint 325k, skill1) found 92.7% (38/41 sampled) of
    post-kick recovery/hold resets are ``bad_motion_body_pos`` -- individual body positions
    diverging from the synthetic, physically-uninformed recovery/hold clip -- not genuine falls
    (``kick_low_height`` fired only 2/41 times in the same sample). The clip is a scripted
    interpolation back to default pose with no momentum-arresting information, so the corrective
    motion a policy needs after a hard kick is by definition off-script and trips this check,
    plausibly cutting episodes short exactly when the policy attempts the correction it needs,
    before ever learning whether it would have worked.

    MEASURED 2026-08-02, CONCERNING -- LEAVE THIS AT False pending a clean re-test. An isolated
    frozen-checkpoint probe (325k skill1, this flag the ONLY change) found a real warning sign:
    genuine ``kick_low_height`` falls in post_kick_stabilization jumped 2->22 of 29 total resets in
    a single 1500-step/256-env rollout, concentrated in the first ~50-60 steps of recovery -- the
    exact window where ``kick_recovery_low_height_sustained``'s own ``grace_steps=50`` +
    ``consecutive_steps=10`` means it cannot have fired yet, so nothing was watching. A subsequent
    real training run resumed from the same 325k checkpoint (325k->408.6k) confirmed the direction
    at far larger scale, matched-step against its non-handoff parent run (also resumed from 325k,
    to 367k): ``kick_topple_frac`` 10.9%->33.7% (skill0) and 8.6%->38.8% (skill1), with
    ``kick_min_base_height`` correspondingly degrading (0.591->0.495, 0.597->0.472) and NOT
    recovering over the full run -- the same "continuous height metric degrading, not noise"
    signature ``bad_tracking_swing_threshold_multiplier``'s own MEASURED OUTCOME used to rule out
    censoring. HOWEVER that training run is CONFOUNDED, not a clean isolation of this flag: it also
    carries ``kick_goal_success_burst`` weight 10.0->300.0 (30x), ``kick_balance_potential`` newly
    active (0.0->weight=1.0), two new shooting reward terms absent in the parent run, and different
    per-skill strike/stand frame boundaries -- any of which could independently drive more
    aggressive, less stable behavior. Net: the isolated probe is a real, unconfounded negative
    signal on its own; the training run corroborates the DIRECTION at much higher confidence but
    not yet the MAGNITUDE specifically attributable to this flag. Do not re-enable without either
    (a) a training run changing ONLY this flag against a matched baseline, or (b) shrinking/
    removing ``kick_recovery_low_height_sustained``'s ``grace_steps`` first (the mechanism most
    directly implicated by the isolated probe) and re-measuring."""

    swing_tracking_sigma_multiplier: float = 1.0
    """Shared (not per-skill) factor widening (NOT narrowing -- must stay > 0.0) every kick-mode
    motion-tracking term's ``sigma`` (managers/reward/terms/wbt.py's six ``motion_*_error_exp``
    functions, all ``exp(-error^2/sigma^2)`` Gaussians) for envs currently ``in_strike_phase``.
    1.0 (default) = off, bit-identical to before this field existed. Applied in
    ``managers/reward/terms/kick_scale_wrappers.py`` (the wrapper layer that already routes these
    six terms through per-skill ``motion_tracking_reward_scale`` -- see that module's docstring),
    not in ``wbt.py`` itself, since those functions are SHARED with standalone WBT training and
    must stay untouched there.

    Motivation (2026-08-01, live gradient arithmetic): ``ball_proximity`` (the dense strike-phase
    approach-shaping term) used a Laplacian kernel (linear in error) against tracking's Gaussian
    (squared in error) -- at this project's real configured weights, a Gaussian's gradient
    collapses far faster than a Laplacian's past ~0.55-0.6m of error, and the live-measured stance
    error AT strike_start (checkpoint 325k skill 1) has median 0.58m, mean 0.78m, p90 1.48m -- the
    median attempt was already entering the strike past the point tracking's OWN gradient had
    effectively vanished (0.0013 at 1.0m). ``ball_proximity`` was switched to a matching Gaussian
    kernel the same day (see its own docstring) to remove the asymmetry at the source; this field
    is the second, independent lever on the same root cause -- widening tracking's own dead zone
    directly, rather than only narrowing shooting's, in case the kernel-matching fix alone proves
    insufficient once measured against a real training run.

    !! READ BEFORE RAISING THIS ABOVE 1.0 !! This is NOT the same mechanism as
    ``bad_tracking_swing_threshold_multiplier`` above (that one widens TERMINATION thresholds; this
    one widens a REWARD kernel's width -- it cannot itself let an episode survive longer or make a
    bad trajectory less likely to be cut short, since it never touches ``managers/termination/``
    at all), but it shares the same underlying "tolerate more deviation during swing" spirit that
    field's own MEASURED OUTCOME above found to make the fall rate WORSE, not better, despite a
    correct diagnosis that swing dominates tracking terminations. Whether a widened reward
    GRADIENT produces the same perverse outcome as a widened termination BOUNDARY is UNTESTED --
    plausible either way, not assumed safe by the mechanism being different. Ships at 1.0
    specifically because of this precedent: no "recommended starting point" is given here (unlike
    ``balance_potential_weight`` below), unlike every other new UNVALIDATED field in this file --
    pick and measure a value deliberately, with the sibling field's outcome in mind, rather than
    treating a widening as self-evidently helpful because the underlying diagnosis was correct."""

    balance_potential_weight: float = 0.0
    """Weight of ``kick_balance_potential`` -- potential-based reward shaping over a capture-point
    balance margin (``managers/reward/terms/balance_potential.py``). 0.0 (default) = term absent,
    exactly as before this field existed.

    Motivation (2026-07-30, measured): kick-mode reward has no usable gradient in the pre-fall
    regime. The six ``motion_*`` tracking terms are ``exp(-error/std)`` and realize only ~19% of
    their configured weight (1.21 of 6.5) -- at the large tracking errors that precede a fall, both
    the term and its derivative are ~0. ``kick_alive`` is 73-80% of positive reward and is a flat
    per-step constant with zero action-gradient. So exactly where 62-85% of measured falls
    originate (the authored swing, per a phase-resolved 256-env probe), the policy is flying blind.

    Every attempt to fix this by retuning existing scalars backfired in BOTH directions:
    loosening termination (``bad_tracking_swing_only`` / ``bad_tracking_swing_threshold_multiplier``)
    tripled the per-cycle hazard, and halving ``kick_alive`` raised it 37-76% because the saturated
    tracking terms could not take over the load. This term instead ADDS the missing gradient
    without removing or reweighting anything: Ng/Harada/Russell (1999) proved
    ``F = gamma*Phi(s') - Phi(s)`` provably cannot change the optimal policy, so unlike every knob
    above it cannot trade kick quality against stability.

    Recommended starting point for an A/B: **50.0**. At that weight the terminal spike on an actual
    fall is ``weight*dt*Phi ~= 1.0`` (about 5x a typical step's total reward, so clearly visible)
    while the steady-state drift a perfectly balanced robot sees is ``(gamma-1)*Phi*weight*dt
    ~= -0.03``/step, ~15% of ``kick_alive``'s +0.2/step -- present but not dominant. Reasoned from
    the measured reward magnitudes, NOT yet validated by a training run. Ships at 0.0 per this
    project's convention of landing new mechanisms as a verified no-op."""

    kick_terrain_light_rough_proportion: float = 0.0
    """Shared (not per-skill) proportion of the terrain bank generated as the ``light_rough`` tier
    (``TerrainBase._light_rough_terrain_func``). 0.0 (default) = the tier is not generated at all,
    a byte-identical no-op: ``Terrain._initialize_terrain_config`` filters zero-proportion types
    out before generation, so the terrain bank, its normalization, and every env's tile assignment
    are unchanged from before this field existed.

    Motivation (2026-08-27): kick-mode envs are gated to flat terrain because a freely-simulated
    ball needs a defined rest position. RoboNaldo hits the same wall -- their ball is spawned
    unconditionally in every stage, and every stage from 2a onward (i.e. the moment shooting
    reward turns on) sets ``use_rough_terrain: false``. Their ONE exception is an optional
    "Stage 1b" tracking-robustness fine-tune on very light terrain (8mm noise, 0.5% slopes) while
    ``goal_weight`` is still 0. This tier is the port of that idea: terrain gentle enough not to
    disturb the ball, aimed at hardening the single-support strike window (where 62-85% of this
    project's measured falls originate) against footing variation.

    **Side effect to weigh before setting this non-zero**: terrain proportions are NORMALIZED
    (``proportions / sum(proportions)``), so adding this tier dilutes every existing type's share
    -- it changes what LOCOMOTION-mode envs train on too, not just kick envs. Prefer taking the
    proportion explicitly out of ``flat``'s share rather than appending it, so locomotion's
    rough/obstacle exposure stays fixed. NOT VALIDATED BY A TRAINING RUN."""

    kick_terrain_light_rough_max_height: float = 0.008
    """Shared (not per-skill) peak height deviation (meters) of the ``light_rough`` tier -- see
    TerrainTermCfg.light_rough_max_height. Default 0.008 is RoboNaldo's own value. Inert unless
    kick_terrain_light_rough_proportion > 0."""

    kick_eligible_terrain_types: tuple[str, ...] = ("flat",)
    """Shared (not per-skill) list of terrain-type names a kick-mode env may be assigned to -- see
    TerrainTermCfg.kick_eligible_terrain_types and TerrainLocomotion.env_terrain_kick_eligible.
    Default ``("flat",)`` reproduces the previous hardcoded flat-only gate exactly.

    Deliberately a SEPARATE switch from kick_terrain_light_rough_proportion above: generating the
    tier and letting kick envs stand on it are independent, so an A/B can generate the tier in
    BOTH arms and vary only this -- keeping terrain-bank layout (hence locomotion's own training
    distribution) identical across arms, so any measured difference is attributable to kick
    eligibility alone rather than to a reshuffled terrain bank."""

    body_push_enabled: bool = False
    """Shared (not per-skill) switch for BodyPushRandomizerState -- sustained, body-targeted
    external-force disturbances (``managers/randomization/terms/locomotion.py``). False (default)
    = term absent, exactly as before this field existed; the existing root-velocity push
    (``push_randomizer_state`` / ``PushRandomizerState``) is untouched and keeps running regardless
    of this flag -- this is an ADDITIONAL disturbance channel, not a replacement.

    Motivation (2026-08-27): the existing push is a one-tick additive velocity impulse on the
    robot's ROOT (``robot_root_states[:, 7:13]``). A real collision -- a shin against a table leg,
    a shoulder against a doorframe -- differs in three ways: (1) it lands on a LIMB, inducing
    joint torques a root impulse never produces, (2) it is SUSTAINED (tens to hundreds of ms), not
    instantaneous, and (3) the obstacle stays there and blocks the recovery motion. This field
    addresses (1) and (2) only. It does NOT address (3) -- a force is not a collision constraint,
    and nothing here stops the robot moving through the notional obstacle; that would need
    collision geometry in the scene, a larger change left explicitly out of scope here.

    Not yet validated by a training run. Ships False per this project's convention of landing new
    mechanisms as a verified no-op."""

    body_push_interval_min_s: float = 4.0
    """Minimum seconds between body-push events per env, uniformly resampled per event (matches
    the unified root push's own loosened [4.0, 8.0]s interval -- see g1_29dof_unified_randomization
    for why that was widened from WBT's [1.0, 3.0]s). Only meaningful when body_push_enabled."""

    body_push_interval_max_s: float = 8.0
    """Maximum seconds between body-push events -- see body_push_interval_min_s."""

    body_push_force_min: float = 20.0
    """Minimum force magnitude (Newtons) applied at the chosen body for the push's duration.
    20-80N was chosen as plausible for a shin/elbow bump against furniture or a doorframe --
    roughly 3-13% of the robot's own body weight (~30kg x 9.81 ~= 294N) applied to a single limb,
    not the whole-body reaction a push of that fraction against the ROOT would represent. Not yet
    validated by a training run; the range is reasoned, not measured against real impact data."""

    body_push_force_max: float = 80.0
    """Maximum force magnitude (Newtons) -- see body_push_force_min."""

    body_push_duration_min_s: float = 0.05
    """Minimum duration (seconds) a body-push force is sustained once it fires -- see
    BodyPushRandomizerState's docstring for why a sustained force, not an instantaneous impulse
    like the existing root push, is the point of this mechanism. 50-200ms brackets a plausible
    real contact duration (a brief bump to a lingering shoulder-check)."""

    body_push_duration_max_s: float = 0.20
    """Maximum duration (seconds) a body-push force is sustained -- see body_push_duration_min_s."""

    body_push_vertical_fraction: float = 0.2
    """Fraction of the push direction's unit vector allowed on the vertical (z) axis, in [0, 1];
    the remainder is uniform in azimuth. Low by default (0.2) because an unmodelled-scenery
    collision is overwhelmingly horizontal -- a vertical component near 1.0 would mostly model
    being dropped on or lifted from below, not bumping into something while walking."""

    body_push_body_names: tuple[str, ...] | None = None
    """Which robot bodies are eligible push targets, one chosen at random per event. None
    (default) uses DEFAULT_BODY_PUSH_BODIES (knees, elbows, torso, pelvis -- see that constant's
    own comment for why feet are excluded). Names are validated against the robot's actual URDF
    link names at env setup; an unknown name raises rather than silently shrinking the target set."""

    use_foot_strike_pitch_reference_relative: bool = False
    """Shared (not per-skill) switch for ``shooting.py::foot_strike_pitch``'s ``reference_relative``
    param -- see that function's own docstring for the full reward-formula rationale. False
    (default) = current behavior, exact no-op: the term rewards ABSOLUTE toe-down pitch regardless
    of what the authored clip itself does at that instant.

    Motivation (2026-08-09): kick_ankle_pitch_correction (managers/command/terms/
    kick_ankle_pitch_correction.py) raises the reference clip's own ankle pitch toward toe-down
    during the strike window, but it is joint-limit-bounded and cannot always reach its +10deg
    target -- measured live, video_012's worst point only reached -52.8deg despite the target,
    because the ankle was already near its physical ceiling there. At those residual frames,
    foot_strike_pitch's absolute reward and the motion-tracking reward (which pulls toward the
    still-toe-up reference) directly oppose each other, at zero net benefit: perfectly reproducing
    the reference already scores this term negatively. Setting this True rescopes the target from a
    fixed absolute value to whatever THIS frame's reference clip actually supplies, so imitating it
    exactly scores 0 instead of negative, while still rewarding doing better and penalizing doing
    worse. NOT yet validated by a training run -- introduced as a configurable opt-in specifically
    so it can be A/B'd against the absolute version rather than assumed better."""

    kick_recovery_locomotion_flip_enabled: bool = False
    """Shared (not per-skill) switch for Stage D's post-swing -> locomotion handoff. False
    (default) = current behavior, exact no-op: a kick episode keeps tracking the authored clip
    through the synthetic recovery+hold splice for its whole duration, same as before this field
    existed. Deliberately named distinctly from the unrelated ``kick_recovery_termination_handoff``
    (a termination-check swap during the SAME window) -- this field instead swaps the whole
    task_mode, not just which termination fires.

    Motivation (2026-08-09): both live telemetry this session (topples split ~46% locomotion-
    approach / ~45% post-kick recovery+hold, with 52% of the recovery-phase failures showing both
    feet airborne at the failure tick -- a stumble signature, not a slow degrade) and the user's own
    RoboJuDo MuJoCo deployment observation agree the dominant failure is the swing-end handoff, not
    the swing itself. The current recovery/hold mechanism scores that window against a synthetic,
    momentum-blind scripted clip (``_maybe_add_default_pose_transition``/
    ``_maybe_add_post_transition_hold`` in managers/command/terms/wbt.py) that has no relationship
    to the robot's real post-swing state.

    When True: the instant ``MotionCommand.in_kicking_phase`` goes True->False for a kick-mode env
    (i.e. ``time_steps`` crosses that motion's ``stand_start_idx``), ``UnifiedManager`` flips that
    env's ``task_mode`` from KICK to LOCOMOTION and pins its locomotion velocity command to exact
    zero for the rest of the episode (see ``UnifiedManager._update_tasks_callback`` and
    ``LocomotionCommand.pin_zero``). Because ``task_mode`` already drives both the
    ``task_mode_onehot`` observation and every reward/termination term's ``task_mode_mask`` gating,
    this reuses the existing locomotion reward wholesale (``alive``, ``tracking_lin_vel``,
    ``tracking_ang_vel``, standard posture/safety) with NO new reward terms and NO observation-width
    change -- deliberately simpler than adding a dedicated post-kick stabilization mechanism (e.g.
    a RoboNaldo-style latched-anchor term), which was considered and set aside: this session already
    has a first-hand cautionary data point in this exact codebase (``balance_potential_weight``, a
    similarly-scoped "stability shaping across phases" change, produced a real, unresolved side
    effect and was reverted) favoring minimal machinery reuse as the first thing to try.

    Deliberately excludes (v1 scope, agreed with user): no randomized walking after the flip -- the
    command stays pinned at deterministic zero for the whole episode, not resampled. No
    locomotion->kick direction either; that needs a commandable skill-selector observation (an
    obs-width change breaking every existing checkpoint's warm-start) and is intentionally a
    separate, later increment. NOT yet validated by a training run."""

    kick_abort_prob: float = 0.0
    """Shared (not per-skill) per-reset probability that a kick-partitioned env becomes a KICK
    ABORT episode: it starts in kick mode as normal, tracks the clip for a randomized number of
    ticks (kick_abort_delay_min/max_steps), then flips KICK->LOCOMOTION mid-clip and must survive
    on locomotion rewards alone for the rest of the episode. 0.0 (default) = OFF, exact no-op.

    WHY (2026-08-28). ``kick_recovery_locomotion_flip_enabled`` already flips every kick env to
    locomotion, but only ever at ONE point: ``pre_recovery_motion_end_idx``, the end of the
    authored clip. So the policy learns "recover to locomotion from a single, always-identical
    pose". Measured consequence (512-env phase-resolved probe, see
    ``UnifiedManager.post_flip_obs_ramp_alpha``'s docstring): **76% of teacher-1's falls and 83%
    of the distilled student's skill-1 falls land in the post-stand phase** -- i.e. AFTER that
    flip, not during the kick -- while a locomotion control arm under identical terrain/push/DR
    toppled 0/1847. Locomotion itself is not fragile; ARRIVING in locomotion mode from an
    off-balance state is, and today the policy only ever practises one such arrival.

    This mechanism generalises that single arrival into a distribution over the whole clip.

    TWO ARRIVAL FLAVOURS, BOTH COVERED BY THIS ONE SWITCH -- because the existing RSI machinery
    (``start_at_timestep_zero_prob``, 0.5 today) already randomises where a kick episode STARTS:
      * env starts at frame 0 -> flips after N ticks: the pre-flip state was produced by the
        policy's own rollout, so contacts/momentum are physically self-consistent. Most REALISTIC,
        but only covers frames the policy actually survives to.
      * env starts at an RSI-sampled random frame -> flips after N ticks: reaches poses the policy
        might never roll into unaided (deep in the strike, say). Best COVERAGE.
    Nothing extra is needed to get both; the RSI draw supplies the mix. Lower
    ``start_at_timestep_zero_prob`` for more coverage, raise it for more realism.

    ON "these envs shouldn't train the kick policy": they DO contribute ordinary kick gradient
    during the pre-flip window, and that is correct, not a leak. FastSAC writes every env's
    transitions to the shared replay buffer unconditionally (``rb.extend(transition)``) with no
    per-env gradient gating, so excluding them would need new machinery -- and there is nothing to
    exclude: an abort env's pre-flip ticks are indistinguishable from any other RSI kick env's,
    tracking the same clip under the same rewards. Only the POST-flip ticks are the new thing, and
    those are locomotion-mode by construction (``task_mode_mask`` zeroes every kick term).

    SIZING. Start at 0.05-0.10, not 0.01. For calibration, the mirror-direction
    ``mid_episode_kick_entry_prob`` runs at 0.3 and still yields only ~1.7-2% of ALL envs in
    handoff state -- a share the Stage-D analysis found too small to move aggregate metrics. At
    0.01 the mechanism risks being both too sparse to learn from and invisible in telemetry. Watch
    the dedicated ``kick_abort_*`` metrics rather than pooled ``kick_topple_frac``.

    RISK TO WATCH: some poses (mid-strike, single support, kicking leg at peak angular velocity)
    may be genuinely unrecoverable, in which case those envs contribute gradient toward an
    impossible task. ``kick_abort_topple_frac`` sitting near 1.0 and never improving is the signal;
    the response is to restrict the delay range so flips land in the approach/follow-through rather
    than the strike window. NOT VALIDATED BY A TRAINING RUN."""

    kick_abort_delay_min_steps: int = 10
    """Lower bound (inclusive) of the uniformly-sampled tick offset from EPISODE START at which a
    kick-abort env flips to locomotion. Only read when kick_abort_prob > 0.0.

    Why a delay at all rather than flipping at reset: reset TELEPORTS the robot onto the clip pose,
    which sets joint positions/velocities but leaves contact forces to be resolved by physics over
    the next tick or two. Flipping instantly would train recovery from states with transient,
    physically-inconsistent contact configurations. A short tracking window lets the dynamics
    settle so the flip state is one the policy could actually be in."""

    kick_abort_delay_max_steps: int = 60
    """Upper bound (inclusive) of the kick-abort flip tick. Only read when kick_abort_prob > 0.0.

    The [min, max] window, combined with the RSI start frame, is what selects WHICH clip phase the
    flip lands in -- the single most consequential choice here. For a 250-frame clip with
    strike at 120-154 (skill012): a frame-0 env with this default [10, 60] window flips during the
    APPROACH, the safest phase. Widen toward 200+ to reach follow-through, and expect the strike
    window to be the hardest by far (this project's own swing-phase analysis puts 62-85% of kick
    falls there). If the flip tick would land past ``pre_recovery_motion_end_idx``, the ordinary
    boundary flip simply fires first and this episode is a normal one -- a safe, silent fallback,
    not an error."""

    mid_episode_kick_entry_prob: float = 0.0
    """Shared (not per-skill) switch for the LOCOMOTION->KICK direction of the handoff (the
    "separate, later increment" kick_recovery_locomotion_flip_enabled's own docstring names) --
    see the full design at https://claude.ai/code/artifact/53c1da51-d841-4979-8bf8-efd5ea652e06
    and memory locomotion_to_kick_handoff_design_settled.md (decisions D1-D8) for the complete
    rationale. Per-reset draw probability, kick-partitioned envs only: on a hit, that env resets
    into LOCOMOTION mode (walks under an ordinary command) carrying a kick-pending flag, then
    transitions into KICK mode mid-episode via UnifiedManager._maybe_enter_kick_from_locomotion
    once an entry-point search finds a covered state -- no teleport, matching enter_at_frame's own
    "not a reset() variant" design. On a miss (including every env at the 0.0 default): today's
    exact existing behavior, teleport-at-reset via MotionCommand.reset(), unchanged.

    0.0 (default) = exact no-op -- the draw in UnifiedManager._resample_task_mode is gated on this
    field being > 0.0 BEFORE any torch.rand() call, not drawn-and-discarded, so Stage A/B/C1/C2
    configs (which never set this) get zero RNG-stream perturbation from this mechanism's mere
    existence, not just zero behavioral effect. See config_values/unified/g1/reward.py's own
    dual-path resolution comment (kick_recovery_locomotion_flip_enabled) for why this reads from
    command_manager.command_cfg directly rather than through that path -- same reasoning, this also
    gates a per-tick task_mode mutation, not a reward-term param.

    The requested skill constrains the entry-point search (D2 of the design doc): a kick-pending
    env only searches its OWN fixed-for-life assigned skill's approach content
    (UnifiedManager._build_task_mode_partition already fixes this per env), never across the whole
    library -- so a caller asking for one skill can't accidentally get a different, easier-to-enter
    one substituted underneath it."""

    mid_episode_kick_entry_min_steps: int = 100
    """Minimum ticks a kick-pending env must walk under locomotion control before the entry-point
    search is even attempted -- avoids searching (and potentially firing) on the very first tick
    after a fresh reset, before the robot has settled into a genuine steady-state gait. Only
    meaningful when mid_episode_kick_entry_prob > 0.0; otherwise unread. Must be >= 0; validated at
    load time."""

    mid_episode_kick_entry_max_residual: float = 0.0
    """ABORT ceiling (D3 of the design doc), not a fire threshold -- the entry-point search fires
    the INSTANT its own best-match residual stops improving tick-over-tick (see
    UnifiedManager._maybe_enter_kick_from_locomotion's own docstring for why: modelling a real
    decel-and-search trajectory against all 4 available clips found every one has an INTERIOR
    minimum, and one clip's residual got 8x WORSE by the time a naive "wait for a low-enough
    threshold" rule would have fired -- see the design doc's D3 section for the full measured
    curve). This field only gates the DECLINE path: if even the best residual ever seen during the
    pre_kick_fallback_timeout_steps window exceeds this value, the kick is declined rather than
    forced from a state no configured skill covers. 0.0 (default) = no ceiling, never declines on
    residual grounds (still subject to the timeout). Must be >= 0.0; validated at load time."""

    pre_kick_decel_steps: float = 0.0
    """Ticks over which a kick-pending env's locomotion command decays toward
    pre_kick_decel_target once the entry-point search's best-so-far residual has stopped improving
    for one tick (the fallback path, D3). 0.0 (default) = exact no-op -- the fallback decay branch
    in _maybe_enter_kick_from_locomotion is only reachable when this is > 0.0. Reuses
    LocomotionCommand.pin_zero's own existing _pinned_zero exclusion mechanism (D8: the command
    would otherwise be overwritten by that term's own periodic resample mid-decay) rather than a
    new one. Must be >= 0.0; validated at load time."""

    pre_kick_decel_target: float = 0.1
    """m/s floor the fallback decays the locomotion command TOWARD, not to exact zero (D3: modelled
    decel curves showed every clip's slowest authored frames still carry some residual forward
    speed, floor ~0.05-0.07 m/s across the 4 available clips -- decaying all the way to zero would
    overshoot past what any clip's own content can match, the same "wait too long, get worse"
    failure the turning-point rule exists to avoid). Only read when pre_kick_decel_steps > 0.0.
    Must be >= 0.0; validated at load time."""

    pre_kick_fallback_timeout_steps: float = 0.0
    """Cap on how long the fallback decel-and-search loop runs before declining the kick outright,
    EXTENDED (not reset) by ticks where the best residual is still improving -- a genuinely
    converging search should not be cut off at a fixed count while it's still making progress; one
    that has plateaued or reversed should not run indefinitely either. 0.0 (default) = no timeout,
    exact no-op (only reachable at all when pre_kick_decel_steps > 0.0, i.e. the fallback path
    exists). Must be >= 0.0; validated at load time."""

    pre_kick_reward_ramp_steps: float = 0.0
    """Mirror of post_flip_reward_decay_steps (Stage D), for the opposite direction: ticks over
    which motion-tracking reward (the same 7 kick_scale_wrappers.py terms) ramps LINEARLY UP from
    0 to full strength after a mid-episode kick entry, instead of the instant full-strength jump
    every existing config gets today. Increment 1's reference re-anchoring already makes the ACTOR's
    input (motion_ref_pos_b/ori_b) start near-continuous at entry, but SAC's critic still bootstraps
    a value estimate through states that were, one tick earlier, worth a completely different
    reward (locomotion) -- same "give the critic a smooth target instead of a cliff" rationale as
    Stage D's own FIX 4, just at the opposite boundary.

    0.0 (default) = exact no-op, same guarantee as post_flip_reward_decay_steps's own docstring:
    the 7 terms stay at task_mode="kick" (RewardManager's own masking zeroes every locomotion-mode
    env's contribution regardless), and the ramp multiplier itself short-circuits to a bare python
    float 1.0 at this default -- multiplying by it cannot introduce even a rounding difference.
    D5 (Stage D interaction): gated on _post_flip_step < 0 for that env, not on elapsed-tick
    arithmetic, so it can never fire simultaneously with the opposite (decaying) ramp regardless of
    how short either ramp window is configured. Must be >= 0.0; validated at load time."""

    pre_kick_termination_grace_steps: float = 0.0
    """Mirror of post_flip_termination_grace_steps (Stage D FIX 2), for the opposite boundary:
    ticks after a mid-episode kick entry during which kick-mode's own bad_tracking/contact/
    low_height termination checks are suppressed for that env -- a still-locomotion-typical pose
    landing in kick-mode's stricter tracking tolerance is otherwise a plausible instant kill on the
    very first tick, before the reference blend below or the reward ramp above has had any chance
    to take effect. 0.0 (default) = exact no-op, same bit-identical-when-graced-window-is-zero
    guarantee as post_flip_termination_grace_steps's own docstring. Must be >= 0.0; validated at
    load time."""

    pre_kick_reference_blend_steps: float = 0.0
    """Increment 3 of the locomotion->kick handoff (2026-08-13):
    https://claude.ai/code/artifact/53c1da51-d841-4979-8bf8-efd5ea652e06. Increment 1's re-anchor
    makes the ROOT/ref-body target match the robot's actual pose EXACTLY at a mid-episode entry
    (motion_global_ref_position_error_exp/_orientation_error_exp see zero error immediately) --
    but the RELATIVE-body targets (motion_relative_body_position_error_exp/_orientation_error_exp,
    motion_global_body_lin_vel/_ang_vel) still snap discontinuously to the entered clip frame's own
    raw authored limb configuration on the very same tick, since `time_steps` jumps straight to the
    chosen entry frame -- a locomotion-typical arm/leg configuration is compared against an
    arbitrary point in the clip's swing/approach cycle with no transition. pre_kick_reward_ramp_
    steps above already ramps those 4 terms' REWARD WEIGHT from 0 up, but the TARGET itself stays a
    hard step function throughout -- as the ramp weight grows, the reward increasingly penalizes
    the robot for not already matching a limb configuration it was never given any transition into.

    When > 0.0: at the mid-episode entry tick, MotionCommand captures the robot's OWN actual live
    body pose/velocity for every tracked body (UnifiedManager._enter_kick ->
    MotionCommand.capture_ref_blend) and, for pre_kick_reference_blend_steps ticks afterward,
    linearly blends (slerp for orientation) FROM that captured actual pose TOWARD the clip's own
    raw per-tick target -- so the four terms' targets start at the robot's own current state
    (zero error, matching the actor's real proprioception) and glide smoothly onto the clip's
    authored trajectory instead of snapping to it. Independent of, and typically configured
    alongside, pre_kick_reward_ramp_steps -- one smooths WHAT is being asked for, the other smooths
    HOW MUCH it counts; using only the reward ramp still lets the (unramped) full-strength target
    dominate once the ramp finishes, while blending only the target still leaves an initially-small
    but nonzero gradient pointing at a physically-arbitrary pose.

    0.0 (default) = exact no-op: the blend multiplier's own sticky-active flag
    (MotionCommand._ref_blend_active) is never set to True (nothing calls capture_ref_blend when
    this is 0.0), so every accessor's blend step short-circuits to the exact pre-existing code path,
    not just a numerically-equal one -- same discipline as increment 1's own _ref_anchor_active.
    Must be >= 0.0; validated at load time."""

    mid_episode_kick_entry_ball_fixed: bool = False
    """Increment 4 of the locomotion->kick handoff (2026-08-13):
    https://claude.ai/code/artifact/53c1da51-d841-4979-8bf8-efd5ea652e06, closing the D2b training-
    scaffold gap ("this is a TRAINING SCAFFOLD, not deployable -- at deploy the ball doesn't move").
    D2's default (this field False) places the ball AT the robot's handoff position
    (place_ball_at_entry) so training only ever sees ball positions inside the kick policy's
    trained (ball_rel(f) +/- randomize_x/y) tube -- correct for getting increments 1-3 training
    signal right, but not what deployment can do (a real ball doesn't teleport to meet the robot).

    True: the ball is placed ONCE, at RESET time, at its configured spawn anchored to the CLIP's
    OWN frame-0 canonical pose (MotionCommand.place_ball_at_reset_pending) -- not the robot's
    pose, which for a kick-pending env isn't teleported anywhere in particular -- and then STAYS
    there for the whole episode; place_ball_at_entry is no longer called at all. Because the ball
    no longer moves to guarantee reachability, the entry-point search (search_entry_point) also
    starts weighing ball-relative geometry: _build_entry_search_table appends 2 more columns (the
    clip's own implied robot-to-ball offset at each approach-window frame, from ball_rel_at_frame)
    to the existing 5 gait features, pooled-z-scored together exactly like the existing 5 -- so a
    frame only scores well now if BOTH the gait AND the ball geometry roughly match the robot's
    real current state. This makes D8's decel-and-retry fallback load-bearing in practice (a
    narrow-coverage skill will decelerate and keep searching far more often than under the
    always-reachable D2 scaffold) rather than a rare edge case.

    False (default) = exact no-op for everything increment 4 touches: _build_entry_search_table/
    _live_entry_features stay at exactly 5 columns (not 7), place_ball_at_entry is still called at
    fire time exactly as increments 1-3 already validated, and place_ball_at_reset_pending is never
    called. Whether a narrow-coverage skill should tell its caller "cannot be entered from here"
    (as opposed to silently declining and retrying next episode, the existing behavior either way)
    is a still-open product decision, not resolved by this field -- see the design doc's "Still
    open" section."""

    # ==============================================================================================
    # 2026-08-18: the four OBSERVATION-side handoff-discontinuity fixes.
    #
    # Motivation (MEASURED, not speculative -- memory `stage-d-handoff-observation-discontinuity`,
    # ckpt 20260814_150032-stageD-1skill-handoff/model_0400000.pt, 256 envs, 86 real firing events):
    # the locomotion->kick handoff produces a ~11x action-rate spike (median ||da|| 0.20 -> 7.86) in
    # ONE control step. Root cause is structural: all three EXISTING smoothing fields
    # (pre_kick_reward_ramp_steps / _termination_grace_steps / _reference_blend_steps) smooth what
    # the policy is JUDGED ON; the OBSERVATION is the one channel with no smoothing at all
    # (ObservationManager.compute_group step 4b applies a hard binary `obs = obs * mask`). Since the
    # deployed action is a DETERMINISTIC function of the observation, a step input necessarily
    # produces a step output -- reward shaping provably cannot fix it. That run had all three ramps
    # ON at 50.0 and still jerked 11x.
    #
    # Measured decomposition of the ||dobs|| ~= 10.2 jump (vs ~0.1 steady): kick_target_pos_b 51%,
    # duplicated proprioception 35%, genuinely-new kick info ~10%, task_mode_onehot ~2%. So ~86% of
    # it carries ZERO new information -- fields 1 and 2 below target exactly that 86%.
    #
    # NONE of these change the observation WIDTH, so every existing checkpoint still warm-starts
    # (contrast the deliberately-deferred skill-selector observation, which does change width and is
    # sequenced last for that reason). They do change what some slots CONTAIN, so expect a critic
    # re-convergence transient after a resume -- judge them at 50k+ steps, not 10k.
    #
    # ALL FOUR DEFAULT TO EXACT NO-OP, same discipline as the pre_kick_* siblings above: unset
    # reproduces current behavior bit-for-bit, so they are opted into (and A/B'd) one at a time from
    # a task_config yaml. NOT ONE OF THEM IS VALIDATED BY A TRAINING RUN.
    #
    # IMPORTANT CAVEAT carried from that same measurement: the jerk is NOT proven harmful in sim --
    # handoff kicks are SAFER than pooled (kick_handoff_topple_frac 6.1% vs kick_topple_frac 11.8%).
    # The motivation for all four is sim2real (torque spikes / hardware wear on the real G1), not
    # the sim topple rate. Do not expect these to move kick_topple_frac.
    # ==============================================================================================

    obs_target_pos_distance_scale: float = 0.0
    """FIX 1 (~51% of the measured jump, the single largest contributor). ``kick_target_pos_b``
    ships at ``scale=1.0`` with no clipping, i.e. RAW METRES: the commanded target sits 5-7 m away,
    so those 2 dims read ~7.27 while essentially every other observation is O(1). Two dimensions out
    of ~450 account for over half the total input discontinuity purely because they are
    unnormalized.

    When > 0.0, ``target_pos_b`` returns ``unit_direction * tanh(distance / this)`` instead of the
    raw offset -- SAME 2 dims (width-preserving, checkpoints still load), bounded magnitude in
    [0, 1] (tanh never reaches 1 mathematically but saturates to exactly 1.0 in float32 at large
    ratios -- bounded either way), direction preserved exactly. Chosen over a plain smaller
    ``scale`` constant because a
    constant still grows without bound (at 20 m you are back to a large input); ``tanh`` saturates.

    Interpretation: this is the distance at which the compressed magnitude reaches tanh(1)=0.76, so
    set it near the working range you care most about resolving. 5.0 is the reasoned starting point
    for this project's 5-7 m targets (7 m -> 0.885, 5 m -> 0.762, 1 m -> 0.197 -- real gradient
    across the whole band rather than saturating at the top of it). NOT tuned by a run.

    0.0 (default) = exact no-op: the raw offset is returned by the identical pre-existing code path.
    Must be >= 0.0; validated at load time."""

    obs_untag_shared_proprioception: bool = False
    """FIX 2 (~35% of the measured jump, at ZERO information cost -- the cheapest of the four).

    ``config_values/unified/g1/observation.py``'s ``_tagged()`` duplicates the locomotion and WBT
    observation groups wholesale, so four terms exist TWICE, once as ``loco_*`` and once as
    ``kick_*``: ``base_ang_vel``, ``dof_pos``, ``dof_vel``, ``actions`` (actor_obs; critic_obs adds
    ``base_lin_vel``, 5 total). Verified 2026-08-18 that each pair is byte-identical in
    implementation -- e.g. both ``terms.locomotion:dof_vel`` and ``terms.wbt:dof_vel`` are literally
    ``return env.simulator.dof_vel``. They differ ONLY in the cfg's scale/noise.

    Because both copies are task-mode-gated, at the fire tick identical physical quantities teleport
    from one set of input neurons to the other -- while the robot's actual body state is unchanged.
    Worse, the two copies disagree on scale: ``dof_vel`` is 0.05 (loco) vs 1.0 (wbt), a **20x** jump
    on the same joint velocities; ``base_ang_vel`` is 0.25 vs 1.0 (4x); ``base_lin_vel`` (critic) is
    2.0 vs 1.0 (2x).

    True: drops the ``task_mode`` tag from exactly those overlapping terms, so BOTH copies stay live
    in BOTH modes and neither ever steps. Observation width is unchanged (the terms already occupied
    their slots at all times -- masking zeroes, never omits), so checkpoints still warm-start.

    Deliberately does NOT also unify the mismatched scales, though that was raised alongside this in
    the original write-up. Once both copies are permanently live the mismatch is a CONSTANT
    re-scaling of a redundant view, not a discontinuity -- it no longer contributes to the jump this
    field exists to remove. Unifying them would additionally mean rescaling a slot the policy has
    400k steps of learned weights for (20x on dof_vel), a far larger perturbation to warm-start than
    the problem it would solve. Left as a separate, independently-A/B-able decision.

    Mode information is NOT lost by untagging: ``task_mode_onehot`` (always live, never tagged)
    states the mode explicitly, so the policy never depended on "which block is zeroed" to infer it.

    False (default) = exact no-op. NOT validated by a training run."""

    obs_ball_always_visible: bool = False
    """FIX 4 (removes its share of the discontinuity at the SOURCE rather than smoothing it).

    ``kick_ball_pos_b``/``kick_target_pos_b`` are tagged ``task_mode="kick"``, so during locomotion
    they read exactly 0 -- even though **the ball is physically present on the field the whole
    time**. That zeroing is a side effect of gating whole groups by task mode, not a modelling
    decision anyone made: the robot is told the ball does not exist right up until the instant it
    does.

    True: leaves both terms ungated, so their contribution to the fire-tick jump becomes zero --
    not smoothed, removed. Width unchanged (same reasoning as ``obs_untag_shared_proprioception``).

    The real argument for this one is not smoothness but CAPABILITY: with the ball visible during
    locomotion the policy can finally learn to walk TOWARD it, which is precisely the Stage D goal.
    Today it structurally cannot, because it cannot see what it is approaching.

    Safe to compute in either mode (verified 2026-08-18): ``ball_pos_b`` reads the ball actor's
    simulator state directly and never consults ``task_mode``; ``target_pos_b`` already guards a
    missing ``target_xy_w`` by returning zeros. Note this is the same continuity argument
    ``ball_pos_b``'s own docstring already makes for keeping the ball live across the Stage B->C
    boundary (holding an input at hard zero collapses ``EmpiricalNormalization``'s running std for
    it, so the first real value arrives 30-100x outside anything the network has seen -- see
    `stagec_obs_normalizer_shock.md`). This field applies that identical, already-accepted reasoning
    to the locomotion->kick boundary.

    Biggest semantic change of the four -> expect the longest fine-tuning transient.
    False (default) = exact no-op. NOT validated by a training run."""

    pre_kick_obs_ramp_steps: float = 0.0
    """FIX 3 -- the missing fourth sibling of ``pre_kick_reward_ramp_steps`` (ramp the REWARD),
    ``pre_kick_reference_blend_steps`` (blend the TARGET) and ``pre_kick_termination_grace_steps``
    (grace the CONSEQUENCE). The absent one is: ramp the OBSERVATION.

    When > 0.0, ``ObservationManager``'s step-4b binary task-mode mask is replaced, for envs inside
    their post-entry window, by a linear ramp ``alpha = clamp(pre_kick_steps_since / this, 0, 1)``:
    kick-tagged terms fade IN (mask ``alpha``) while locomotion-tagged terms fade OUT (mask
    ``1 - alpha``), instead of both switching in a single tick. ``task_mode_onehot`` is ramped by
    the same alpha, so the semantics stay coherent -- the policy reads "belief in kick mode fading
    in", not the incoherent "you are 100% in kick mode but the ball is only 40% present".

    Unlike fixes 1/2/4 (each removes ONE specific term's contribution) this smooths EVERY still-
    gated term at once, including the ~10% that is genuinely-new kick information and therefore
    cannot be removed by any of the others. Reuses the ALREADY-EXISTING
    ``UnifiedManager.pre_kick_steps_since()`` -- the same per-env tick counter the reward ramp and
    termination grace already run on -- so there is no new state machine and the three windows
    cannot drift apart.

    Applies ONLY to the mid-episode locomotion->kick entry (it is keyed off ``_pre_kick_step``, the
    sentinel stamped by ``_enter_kick``). A kick-mode env that started that way at RESET has no
    entry tick and is unaffected -- correct, since there is no transition to smooth there.

    0.0 (default) = exact no-op: ``task_mode_mask_soft`` short-circuits and returns the identical
    BOOL tensor ``task_mode_mask`` already returned, so the multiply is bit-for-bit unchanged.
    Must be >= 0.0; validated at load time. NOT validated by a training run."""

    contact_termination_force_threshold: float = 0.0
    """Unified-config-only override for the ``contact`` termination's ``force_threshold`` (newtons)
    -- the L2 norm of contact force on ``terminate_after_contacts_on`` links (pelvis/shoulder/hip)
    above which an episode is terminated. 0.0 (default) = inherit the locomotion baseline's own
    value (1.0), an exact no-op.

    WHY THIS EXISTS (measured 2026-08-18, probe over 927 kick episodes of run
    20260818_021646-stageC-skill2-new-fixes at ckpt 297000):

      * ``contact`` caused **78.9%** of all kick-episode terminations -- far more than toppling
        (kick_low_height 13.4% + low_height 2.8%) or bad_tracking (9.0%). Zero episodes timed out.
      * At the firing tick the robot was UPRIGHT EVERY TIME: base height mean 0.779 m (the target
        standing height is 0.76), range 0.731-0.828, **100%** above 0.5 m and **0%** below 0.3 m.
        Not one of these was a fall.
      * The forces are real, not noise: median 25.8 N, p90 205 N, max 456 N (7.5% / 60% / 133% of
        this robot's ~343 N weight) -- consistent with the kicking leg's hip link striking the
        pelvis or stance leg during the swing, which the retargeted clip makes largely unavoidable.

    So at the inherited 1.0 N the policy loses ~4 of every 5 kick episodes to a self-collision
    while standing upright, and cannot accumulate the post-kick experience it needs -- matching
    that run's flat topple curve across 120k steps. Note this ALSO means ``kick_topple_frac``
    badly undercounts failure there: ``contact`` ends the episode long before the base can descend
    past the 0.40 m the topple metric requires.

    Terminating on this is the wrong tool regardless of the force magnitude: genuine falls are
    already covered by the two height terms, and a mechanical-wear concern about self-collision
    belongs in a PENALTY (the reward block already has ``kick_penalize_self_contact_feet``), not a
    hard episode kill that starves training.

    50.0 is read straight off the measured distribution -- 73.1% of observed firings are below it,
    so it removes roughly three quarters of the spurious kills while still catching genuinely hard
    impacts (p90 205 N, max 456 N still terminate). NOT validated by a training run. Scoped to the
    unified config's own ``contact`` registration (config_values/unified/g1/termination.py), so a
    standalone locomotion experiment keeps the baseline 1.0 regardless of this field.

    Must be >= 0.0; validated at load time."""

    post_flip_alive_scale: float = 1.0
    """Unified-config-only scale on the locomotion ``alive`` term's contribution (weight 10.0,
    ``managers/reward/terms/locomotion:alive``) for envs that have crossed the kick->locomotion
    flip (``kick_recovery_locomotion_flip_enabled``) -- i.e. genuinely POST-KICK recovery, not the
    pre-kick locomotion approach a kick-partitioned env may also spend time in before it ever
    enters the clip. 1.0 (default) = exact no-op: ``alive`` keeps returning its plain
    ``torch.ones(...)``, byte-identical to before this field existed.

    WHY THIS EXISTS (measured 2026-08-18/19, comparing 20260818_114352-stageC-skill2-contactfix50N
    against 20260818_140359-stageC-skill2-badmotion035-AB -- same start checkpoint, same reward
    config, only ``bad_motion_body_pos_threshold`` differs, 0.45 vs 0.35):

      * At steps 305k-340k the 0.35 run kicks LESS (hit_rate 0.338 vs 0.557, ball_velocity 1.29 vs
        2.68 -- ball_velocity_reward itself down 91%) and falls LESS (topple 0.093 vs 0.119,
        kick_episode_length +173 steps) -- i.e. the safer policy is also the worse kicker, on the
        SAME checkpoint pair.
      * That trade pays: ``rew_alive`` (locomotion, weight 10.0, ~57% of positive reward that
        window) is UP +1.20/step for the 0.35 run while every kick-reward term is down by
        ~0.001-0.006/step each -- roughly 200x smaller than the alive gain. ``kick_alive_frac``
        (fraction of episode time an originally-kick env spends alive post-flip) is +0.42.
        ``Train/mean_reward`` is +8.48/step (+25%) for the policy that kicks worse.
      * Mechanism: ``_update_kick_recovery_locomotion_flip`` (unified_manager.py) flips
        task_mode KICK->LOCOMOTION purely on ``motion_command.time_steps`` crossing
        ``pre_recovery_motion_end_idx`` -- there is no dependence on whether the env actually made
        ball contact. So avoiding the kick entirely (never triggering the high-force, fall-risking
        contact) and simply outlasting the clip is a strictly cheaper way to reach the same
        alive-farming payout than kicking and risking a topple.

    Implementation note: distinguished from the pre-kick locomotion approach via
    ``UnifiedManager._post_flip_step`` (>= 0 only after ``_update_kick_recovery_locomotion_flip``
    has fired for that env; stays -1 for its entire life otherwise, including for a kick-
    partitioned env still walking up to the clip and for a genuinely locomotion-partitioned env)
    -- the SAME per-env sentinel ``post_flip_reward_decay_steps``/FIX 4 and the obs-ramp fixes
    already use, so this cannot drift out of sync with the rest of the post-flip machinery.

    A value in (0.0, 1.0) shrinks the post-flip alive payout without removing the survival
    incentive entirely (an instant 0.0 cliff risks the same critic-bootstrap destabilization FIX 4
    was built to avoid for the tracking terms -- see ``post_flip_reward_decay_steps``'s own
    docstring). NOT validated by a training run -- no specific value measured yet, this field only
    makes one configurable.

    Scoped to the unified config's own ``alive`` registration (config_values/unified/g1/
    reward.py), so a standalone locomotion experiment is unaffected regardless of this field.

    Must be >= 0.0; validated at load time."""

    warm_start_obs_ramp_steps: float = 0.0
    """FIX 6 (2026-08-18) -- the LOAD-TIME sibling of FIX 3/5's mid-episode ramps, for the case
    those two do not cover: warm-starting a checkpoint into an observation config that changed
    since it was saved.

    WHY THE OTHER FIXES ARE NOT ENOUGH FOR THIS CASE (measured, not theoretical): resetting
    ``EmpiricalNormalization``'s running mean/var for shifted terms (the 2026-08-18 normalizer-
    shock guard, ``FastSACAgent._reset_normalizer_slots_for_shifted_obs_terms``) fires correctly
    but does NOT prevent the shock -- confirmed on a real run, action_std still spiked 4.7x
    (0.052->0.244) and kick_topple_frac still jumped 0->0.30 in one logging interval, despite the
    guard resetting exactly the right 10-12 terms. Root cause: fixes like
    ``obs_untag_shared_proprioception``/``obs_ball_always_visible`` are not a SCALE change on an
    already-live channel (which normalization can fix) -- they change which POPULATION a channel
    is live for, e.g. ``kick_dof_vel`` going from "hard-zero for every locomotion-mode env" to
    "genuinely live". The actor's early-layer weights were fit under the structural invariant
    "these dims are 0 in this context"; that invariant breaking is a JOINT/conditional
    distribution shift, not a marginal one, and a per-feature normalizer is structurally
    incapable of correcting joint structure -- only the weights themselves can adapt, via real
    gradient steps, which is exactly what a discontinuous step-function input makes hard to do
    safely.

    When > 0.0: at load time, for every observation term whose (params, scale, task_mode, clip)
    differs from what the checkpoint being loaded was saved with, the observation manager
    computes BOTH the OLD-style value (using the checkpoint's own saved params/scale/task_mode/
    clip, via the SAME underlying term function -- this is a reparameterization blend, not a
    different computation) and the NEW-style value (today's config, the existing pipeline
    unchanged), and linearly blends them: ``alpha = clamp(steps_since_load / this, 0, 1)``,
    ``blended = (1-alpha)*old + alpha*new``. At the instant of resume (``alpha=0``) the term's
    contribution to the observation is (as closely as a reparameterization blend can make it)
    EXACTLY what the checkpoint's own training saw -- no shift at step 0 -- and only gradually,
    continuously (not as a step function) introduces the new behavior over this many ticks.

    Supersedes the normalizer-reset guard for whichever terms it covers (the two would otherwise
    fight each other: resetting a term's normalizer to (0,1) while still feeding it OLD-scale
    blended values, at low alpha, would itself be wrong -- the checkpoint's ORIGINAL normalizer
    stats are still the correct ones to keep using while alpha is small). The guard remains the
    fallback for anyone who leaves this at its 0.0 default.

    0.0 (default) = exact no-op: no blend state is ever configured, `ObservationManager.
    compute_group` takes the identical code path it always has. Must be >= 0.0; validated at load
    time. NOT VALIDATED BY A TRAINING RUN -- mechanism argument plus the measurement that ruled
    out the alternative (normalizer-reset alone), not an outcome."""

    post_flip_obs_ramp_steps: float = 0.0
    """FIX 5 (2026-08-18) -- ``pre_kick_obs_ramp_steps``'s opposite-direction sibling, and the one
    aimed at where falls actually happen.

    ``kick_recovery_locomotion_flip_enabled`` flips an env KICK->LOCOMOTION at ``stand_start_idx``.
    That flip already ships two smoothers (``post_flip_termination_grace_steps``,
    ``post_flip_reward_decay_steps``) -- and, exactly as with the loco->kick entry before FIX 3,
    BOTH act on what the policy is JUDGED ON while the OBSERVATION snaps over in a single control
    step (every ``kick_*`` term live->0, every ``loco_*`` term 0->live).

    WHY THIS ONE IS THE HIGH-VALUE DIRECTION (measured 2026-08-18, phase-resolved probe, 512 envs,
    matched protocol across student and both teachers):

      | policy                | post-stand share of falls |
      |-----------------------|---------------------------|
      | teacher 1 (video_012) | 76.0%                     |
      | student, skill 1      | 82.6%                     |
      | student, skill 2      | 59.4%                     |
      | teacher 2 (video_011) | 44.3%                     |

    i.e. the MAJORITY of every measured policy's falls land after this flip, not during the kick.
    And standing itself is not the hard part: a locomotion control arm under identical terrain/
    push/DR toppled 0/1847 episodes (memory `kick-topple-localized-and-balance-potential-lever`).
    What is hard is ARRIVING in locomotion mode discontinuously -- off-balance, mid-momentum, with
    the observation vector swapping halves in one tick.

    When > 0.0, the observation crossfades over this many ticks after the flip instead of switching
    binary, driven by the ALREADY-EXISTING ``UnifiedManager.post_flip_steps_since()`` -- the same
    per-env counter the two existing post-flip smoothers run on, so all three windows share one
    definition of "since when" and cannot drift apart. 50.0 matches its two siblings' current
    values.

    0.0 (default) = exact no-op: ``task_mode_mask_soft`` short-circuits and returns the identical
    BOOL tensor ``task_mode_mask`` already returned. Must be >= 0.0; validated at load time.
    NOT VALIDATED BY A TRAINING RUN -- this is a mechanism-level argument plus a phase measurement,
    not an outcome. A/B it against the same config with this at 0.0."""

    bad_motion_body_pos_threshold: float = 0.25
    """Shared (not per-skill) override for ``bad_tracking``'s ``bad_motion_body_pos`` sub-check
    threshold (``managers/termination/terms/wbt.py::BadTrackingZOnly.bad_motion_body_pos``,
    Z-axis-only per-body error against ``left_ankle_roll_link``/``right_ankle_roll_link``/
    ``left_wrist_yaw_link``/``right_wrist_yaw_link``). 0.25 (default) is this project's existing
    hardcoded value (``config_values/wbt/g1/termination.py``) -- setting this field to anything
    else is the only behavior change; at 0.25 it is a true no-op, bit-identical to before this
    field existed.

    2026-08-05, ported from RoboNaldo (arXiv:2606.11092)'s ``ee_body_pos`` termination
    (``tracking_env_cfg.py``): they progressively WIDEN this exact same threshold across their
    curriculum -- 0.25 (their Stage 1, matching our existing hardcoded default exactly) -> 0.35
    (S2a) -> 0.5 (S2b), uniformly across the same 4 tracked bodies -- via a separate resumed
    training run per stage with an edited yaml, the same "per-skill-yaml-edit-then-resume"
    curriculum mechanism ``root_tracking_reward_scale``/``motion_tracking_reward_scale`` already
    use here (see ``ROBONALDO_PORT_SCOPE.md`` Sec 1a). This field is that same knob for this one
    threshold: set ``bad_motion_body_pos_threshold: 0.35`` (or ``0.5``) in configs/*.yaml and
    resume from checkpoint to reproduce their later-stage tolerance -- NOT auto-progressed
    in-run, matching RoboNaldo's own between-runs (not within-run) staging.

    Also feeds ``penalty_kick_ee_body_pos_divergence``'s own ``threshold`` param (see
    ``config_values/unified/g1/reward.py``), so the termination and its paired reward-side penalty
    stay numerically synced under one source of truth -- mirroring RoboNaldo's own
    ``task_overrides.py``, where a single ``termination_overrides.ee_body_pos.threshold`` yaml
    entry sets BOTH ``terminations.ee_body_pos`` and ``rewards.ee_body_pos_termination_penalty``
    together, by construction, so the two can never drift apart under their own config schema.

    NOT yet staged/enabled by this port -- landing the knob is this step; choosing to actually
    widen it during a real curriculum resume is a deliberate, separate decision, same discipline
    as every other new mechanism in this port."""

    kick_recovery_drift_deadzone: float = 0.15
    """Shared (not per-skill) tolerance radius (meters, simple Euclidean XY, direction-agnostic)
    for ``kick_recovery_drift_sustained`` (``managers/termination/terms/wbt.py``) -- terminates if
    the robot's base drifts more than this far from where it was the instant it entered post-kick
    recovery/hold, sustained for that term's own ``consecutive_steps``. Only takes effect when
    ``kick_recovery_termination_handoff`` (below/above) is True -- the two terms are always
    installed together, see that field's own docstring. 0.15 (15cm) is the user-specified starting
    point, not yet measured against live telemetry; see ``kick_recovery_drift_sustained``'s own
    docstring for the cross-check against this project's own drift-diagnostic measurements (~4-6cm
    typical for envs that survived their recovery window cleanly)."""

    ee_body_pos_warmup_threshold: float = 0.25
    """Shared (not per-skill) override for ``penalty_kick_ee_body_pos_divergence``'s
    ``warmup_threshold`` param (``managers/reward/terms/locomotion.py``) -- the full-3-axis check
    active only during the first ``warmup_steps`` after a reset (see that function's own
    docstring). 0.25 (default) matches its own hardcoded default -- a true no-op.

    2026-08-05, ported from RoboNaldo (arXiv:2606.11092)'s ``ee_body_pos_termination_penalty``
    (``tracking_env_cfg.py``): unlike ``threshold`` (synced with ``bad_motion_body_pos_threshold``
    above, which progressively WIDENS 0.25/0.35/0.5 across S1/S2a/S2b), their own
    ``warmup_threshold`` follows a DIFFERENT, non-monotonic schedule -- 0.25 (S1, their base
    default, unset in ``tracking_params.yaml``) -> 0.7 (S2a, ``task_params_1.yaml`` override) ->
    0.7 (S2b, ``task_params_2.yaml`` override, same value as S2a) -- confirmed directly from their
    yaml, not inferred. Needed a SEPARATE field from ``bad_motion_body_pos_threshold`` because the
    two params don't move together in their own schedule (this one jumps once and stays; that one
    keeps widening). Found missing (this field's own default of 0.25 was silently reached
    regardless of the yaml, since RewardManagerCfg's weight/sigma-only override mechanism has no
    generic per-param override path) during a live end-to-end verification of
    ``configs/task_config_stageC1.yaml`` -- the term's ``threshold`` param was correctly synced,
    but ``warmup_threshold`` stayed at 0.25 instead of the intended 0.7, caught by an explicit
    assertion, not by inspection. NOT yet staged/enabled by this port's own default -- landing the
    knob is this step, same discipline as every other new mechanism here."""

    base_robot_target_height: float | None = None
    """2026-08-14, user-requested: a single GLOBAL standing-height target (meters), read from
    HOLOSOMA_SKILLS_CONFIG's own file's top-level ``base_robot: {target_height: ..., deadzone:
    ...}`` block (see configs/skills.example.yaml) -- NOT parsed here via the normal per-field
    ``raw.get(...)`` machinery every other MultiSkillConfig field uses, because it must be read
    from the SKILLS file specifically, always, regardless of whether 1-file or 2-file
    (HOLOSOMA_TASK_CONFIG) mode is active -- every other global field here instead follows
    whichever file the mode currently points 'global fields' at. See ``load_multi_skill_config``'s
    own body for exactly where this gets parsed (a separate step, not
    ``_parse_multi_skill_global_fields``).

    Governs BOTH ``penalty_stand_height`` (locomotion-mode zero-velocity standing) and
    ``penalty_kick_recovery_stand_height`` (kick-mode post-strike standing) uniformly -- prior to
    this field, both terms' ``target_height`` was a Python-hardcoded literal (0.76) in
    ``config_values/unified/g1/reward.py``, independently, with no yaml override path at all for
    either. None (default) = no override, both terms keep that same hardcoded 0.76 -- a true
    no-op for every skills.yaml that doesn't set ``base_robot``."""

    base_robot_deadzone: float | None = None
    """2026-08-14, user-requested: a single GLOBAL free-tolerance band (meters) around
    ``base_robot_target_height``, read from the same ``base_robot:`` block -- see that field's own
    docstring immediately above for the full parsing/scope rationale (same file, same "always the
    skills file" rule).

    Governs BOTH ``penalty_stand_height`` and ``penalty_kick_recovery_stand_height``'s
    ``deadzone`` uniformly. For ``penalty_kick_recovery_stand_height`` specifically, this
    TAKES PRIORITY over the narrower, pre-existing ``kick_recovery_stand_height_deadzone`` field
    below when set (specific-vs-general precedence: this is the newer, broader knob layered on
    top, not a replacement -- any existing task-config yaml that already sets
    ``kick_recovery_stand_height_deadzone`` explicitly keeps working exactly as before as long as
    ``base_robot`` is left unset). ``penalty_stand_height`` (locomotion) had NO override path at
    all before this field -- it is this field's only source. None (default) = no override, both
    terms keep their existing behavior -- a true no-op for every skills.yaml that doesn't set
    ``base_robot``."""

    kick_recovery_stand_height_deadzone: float = 0.015
    """Shared (not per-skill) override for ``penalty_kick_recovery_stand_height``'s ``deadzone``
    param (``managers/reward/terms/locomotion.py``) -- how far base height may deviate from
    ``target_height`` before this term's penalty starts accruing. 0.015 (default) matches its own
    hardcoded value in ``config_values/unified/g1/reward.py`` -- a true no-op. Superseded by
    ``base_robot_deadzone`` above when that is set (see its own docstring for the precedence).

    2026-08-06, user-requested plumbing: this project's own ``kick_recovery_posture_reward``
    category (own design, no RoboNaldo mapping) has no yaml-configurable deadzone/tolerance at
    all today -- only ``weight`` (via the tuning-yaml mechanism, ``config_types/reward_tuning.py``)
    reaches these terms; every other param, including this one, is Python-hardcoded and identical
    across every stage's yaml file regardless of content. Motivation: as ball/target randomization
    widens across the curriculum (Stage B -> C1 -> C2), the natural landing-pose variance after a
    kick grows too, and a deadzone tuned tight for Stage B's narrow task may increasingly penalize
    legitimately-different-but-fine Stage C landings rather than genuine instability -- see the
    live drift-diagnostic investigation this field grew out of. Landing the knob is this step;
    choosing an actual per-stage progression is a deliberate, separate decision, same discipline as
    every other new mechanism in this port -- NOT yet staged/enabled by this port's own default."""

    kick_recovery_stand_orientation_deadzone: float = 0.025
    """Shared (not per-skill) override for ``penalty_kick_recovery_stand_orientation``'s
    ``deadzone`` param -- how far pelvis tilt may deviate from upright before this term's penalty
    starts accruing. 0.025 (default) matches its own hardcoded value -- a true no-op. See
    ``kick_recovery_stand_height_deadzone``'s own docstring immediately above for the full
    motivation (identical rationale, different term) -- landing the knob only, not yet staged."""

    kick_recovery_stand_feet_width_deadzone: float = 0.03
    """Shared (not per-skill) override for ``penalty_kick_recovery_stand_feet_width``'s
    ``deadzone`` param -- how far ankle-to-ankle width may deviate from ``nominal_width`` (0.24m,
    NOT itself made configurable here -- a robot-morphology constant, not something that should
    vary by curriculum stage) before this term's penalty starts accruing. 0.03 (default) matches
    its own hardcoded value -- a true no-op. Independent field from
    ``kick_recovery_stand_knee_width_deadzone`` below (2026-08-06, user-requested: kept separately
    settable even though both currently share the same 0.03 value, rather than one shared field).
    See ``kick_recovery_stand_height_deadzone``'s own docstring for the full motivation -- landing
    the knob only, not yet staged."""

    kick_recovery_stand_knee_width_deadzone: float = 0.03
    """Shared (not per-skill) override for ``penalty_kick_recovery_stand_knee_width``'s
    ``deadzone`` param -- same construction as ``kick_recovery_stand_feet_width_deadzone``
    immediately above, one body segment up the leg. 0.03 (default) matches its own hardcoded value
    -- a true no-op. Independent field, not shared with the feet-width deadzone (see that field's
    own docstring for why) -- landing the knob only, not yet staged."""

    kick_swing_orientation_deadzone: float = 0.0
    """Shared (not per-skill) override for ``penalty_kick_swing_orientation``'s ``deadzone`` param
    (``managers/reward/terms/locomotion.py``) -- how far PELVIS tilt from vertical may go, during
    the whole (ungated) kick episode, before this term's penalty starts accruing. 0.0 (default)
    matches its own hardcoded value in ``config_values/unified/g1/reward.py`` -- a true no-op
    (penalizes tilt from the very first radian).

    2026-08-12: this term's own docstring already flagged the risk this field closes -- "the swing
    genuinely needs freedom to lean" -- but shipped with deadzone=0.0 anyway (an explicit opt-in
    starting point, not a validated one). Measured directly against the reference clip's own npz
    (video_012, ``body_quat_w`` -> projected gravity): pelvis tilt during the STRIKE window
    (``[strike_start_idx, stand_start_idx)``) is 25.6 deg mean / 33.3 deg peak -- 3x the approach
    segment's 7.9 deg mean, and NOT a training artifact, an AUTHORED feature of the kick motion
    itself (leaning is how a biped counterbalances a swinging leg). At deadzone=0.0, this term
    penalizes the clip's own correct execution at full strength, in direct opposition to the
    tracking reward on the exact same DOFs at the exact same instant -- the same
    reward-fights-tracking pathology this project already investigated (and disproved) for
    ``kick_recovery_drift``, except here the data CONFIRMS it. 0.55 (sin(33.3 deg) = 0.549, this
    field's own measured peak with a small margin) is the reasoned value that flips this term's
    meaning to "penalize lean BEYOND what the kick itself demands" instead of penalizing the
    demand itself -- see ``configs/task_config_stageC1.yaml``'s own comment for the exact
    npz-measurement numbers this value is derived from. NOT validated by a training run."""

    kick_swing_torso_orientation_deadzone: float = 0.0
    """Shared (not per-skill) override for ``penalty_kick_swing_torso_orientation``'s ``deadzone``
    param -- same construction and motivation as ``kick_swing_orientation_deadzone`` immediately
    above, TORSO tilt instead of pelvis. 0.0 (default) matches its own hardcoded value -- a true
    no-op. Measured directly against the same npz: torso tilt during the STRIKE window is 14.7 deg
    mean / 19.6 deg peak (vs. 1.6 deg mean during approach). 0.35 (sin(19.6 deg) = 0.335, peak plus
    a small margin) is the reasoned value -- independent field from the pelvis deadzone above
    because the two bodies' authored strike-phase leans differ substantially (33.3 vs 19.6 deg
    peak), so a single shared deadzone would be miscalibrated for one of them. NOT validated by a
    training run."""

    kick_ball_velocity_v_ref: float | None = None
    """Shared (not per-skill) override for ``shooting.py::ball_velocity``'s ``v_ref`` -- the speed
    at which the Lorentzian ``s^2/(s^2+v_ref^2)`` reaches 0.5. None (default) keeps the term's own
    registered 5.0, bit-identical to before this field existed.

    2026-08-21: at the measured 1.6 m/s operating point, v_ref=5.0 sits deep in the flat tail
    (value 0.093, gradient 0.105 per m/s); 2.0 gives 0.390 and 0.297 -- 2.8x steeper exactly where
    the policy lives, while still asymptoting to 1 so a harder strike keeps paying more. See
    ball_velocity's own docstring for the full measurement and for the VALIDATION STATUS caveat
    (these shipped ON in three runs but their individual contribution is confounded)."""

    kick_error_ball_to_target_sigma: float | None = None
    """Shared (not per-skill) override for ``shooting.py::error_ball_to_target``'s ``sigma``. None
    (default) keeps the term's own registered 1.0, bit-identical to before this field existed.

    2026-08-22, azimuth-aim refactor: for any kick_aim_enabled skill this sigma is a distance at
    the FIXED kick_aim_nominal_distance_m, so it implicitly sets an angular tolerance of roughly
    asin(sigma/D). At the defaults (1.0, D=5.0) that's ~11.5 deg, in the same neighborhood as
    RoboNaldo's own reported ~10-14 deg hardware accuracy -- plausible, NOT independently
    re-validated against a live kick_aim_enabled training run. See error_ball_to_target's own
    docstring for the full reasoning; change this only against real measurement, not blind."""

    kick_ball_velocity_use_latched_peak_speed: bool = False
    """Shared (not per-skill) opt-in: make ``ball_velocity`` read ``_ShotTracker.max_ball_speed``
    (this attempt's LATCHED PEAK) instead of the instantaneous ``ball_speed``. False (default) =
    the original instantaneous reading, bit-identical.

    Mirrors ``error_ball_to_target``'s own use of the latched ``min_target_dist``. STRONGLY
    RECOMMENDED to enable together with kick_ball_velocity_use_post_locomotion_gate below --
    paying instantaneous speed over that wider window rewards a ball that merely keeps rolling and
    decays to ~0 as it decelerates."""

    kick_ball_velocity_use_post_locomotion_gate: bool = False
    """Shared (not per-skill) opt-in: pay ``ball_velocity`` over ``_post_locomotion_multiplier``
    (325 ticks) instead of ``_strike_phase_multiplier`` (67 ticks). False (default) = the original
    strike-only window, bit-identical.

    Motivation: ``ball_contact_hit`` (flat 1.0 for ANY confirmed touch) already pays over the wide
    window while ball_velocity -- the only term demanding POWER -- pays over the narrow one, a
    4.9x asymmetry that lets "just make contact" collect ~99.5% of the available shooting reward.
    See ball_velocity's own docstring."""

    kick_aim_theta_ref_deg: float = 45.0
    """Fixed normalization reference (degrees) for the ``kick_aim_command`` observation slot
    (2026-08-22, azimuth-aim refactor). The actor reads ``theta / kick_aim_theta_ref_deg`` -- NOT
    ``theta / kick_aim_theta_max_deg`` -- specifically so that widening the sampling range later
    (a natural curriculum: start narrow, widen once contact is reliable) does not change what an
    already-learned input value means to the network. Keep this fixed for the lifetime of a
    checkpoint lineage; only raise it if you deliberately need to command wider than +/-45 deg
    (verify no live skill's kick_aim_theta_max_deg would then need to exceed it first)."""

    kick_aim_theta_max_deg: float = 15.0
    """Global default per-reset uniform +/- half-range (degrees) sampled into this attempt's
    ``kick_aim_theta``, for any skill with ``SkillConfig.kick_aim_enabled=True``. Overridable per
    skill via ``SkillConfig.kick_aim_theta_max_deg``. Must be > 0.0 and <= kick_aim_theta_ref_deg
    (a range wider than the reference would saturate the observation outside [-1, 1] and silently
    clip, rather than erroring, so this is validated at parse time instead).

    2026-08-22 measurement (see scripts/calibrate_nominal_bearing.py, which fits a skill's real
    departure bearing from a genuine MuJoCo rollout and writes the result back into that skill's
    own target_x/target_y -- see ``SkillConfig.resolved_nominal_bearing_deg()``): RoboNaldo's own
    reported hardware accuracy is ~10-14 deg radial from a 3-5m free-kick, so a command range much
    narrower than that leaves too few distinguishable settings to demonstrate real steerability.
    15.0 is a starting point for an early curriculum stage, not a final value -- the union of
    several skills' own (nominal_bearing +/- this range) is the intended coverage story, not one
    skill's range alone."""

    kick_aim_nominal_distance_m: float = 5.0
    """Fixed distance ``D`` (meters), for any skill with ``kick_aim_enabled=True``: the per-attempt
    target point is synthesized at this distance from the ball's ACTUAL placed position (after its
    own position_randomization/OOD draw), along the commanded bearing
    ``skill.nominal_bearing_deg + kick_aim_theta``. This is what makes the aim geometry spawn-
    invariant -- ``target_w - ball_w`` is exactly ``D * unit(bearing)`` regardless of where the
    ball's noise draw landed, by construction (no longer a cancellation between two independent
    noise draws that could silently drift apart if one call site changed).

    5.0 matches this project's own existing ``target_x`` default-offset convention
    (``SkillConfig.resolved_target()``: ``x + 5.0``) and RoboNaldo's own fixed downfield goal-plane
    distance -- chosen so metre-denominated shot-error numbers stay directly comparable to both.
    Reward terms that read distance-to-target (``error_ball_to_target``'s sigma,
    ``goal_success_burst``'s success_radius) now implicitly define an ANGULAR tolerance
    (``atan(radius / D)``) once a skill is kick_aim_enabled -- re-tune those against this value,
    they are not automatically consistent with it."""

    kick_ball_over_line_require_has_kicked: bool = False
    """Shared (not per-skill) switch for ``ball_over_line``'s (``managers/reward/terms/
    shooting.py``) own ``require_has_kicked`` param -- when True, gates the term to
    ``has_kicked`` so an accidental non-kick-foot ball nudge (a torso lean, trailing-leg brush,
    wrong-foot graze) can't bias this term's reading. False (default) = off, bit-identical to
    before this field existed -- matches RoboNaldo's own ungated registration exactly.

    2026-08-06, user-requested: the same class of fix already landed (and live-measured) for
    ``predicted_error_ball_to_target``, applied here on reasoning alone -- ``ball_over_line``
    measures a DISPLACEMENT from spawn, not a per-tick reading, so an early accidental nudge can
    bias the whole rest of the attempt, not just one tick (see ``ball_over_line``'s own docstring
    for the full mechanism, including why the ``back_line_dist`` penalty side is more exposed than
    the ``over_line_dist`` reward side). Ships as an opt-in, not landed unconditionally, because
    unlike the ``predicted_error_ball_to_target`` case this hasn't been independently live-measured
    for THIS term yet -- same "reasoned starting point, not yet validated" discipline as every
    other new mechanism in this project."""


# N-skill training mode is opt-in, unlike HOLOSOMA_BALL_CONFIG (which always points SOMEWHERE, just
# defaulting to configs/ball.yaml): the single-clip path must keep working byte-for-byte for anyone
# not using this mechanism at all, so multi_skill_mode_enabled() below is a hard gate config_values/
# unified/g1/{command,reward}.py check BEFORE touching MultiSkillConfig -- unset (the default) means
# this module is not even consulted. Same before-tyro-CLI-parsing / before-config_values-import
# discipline as HOLOSOMA_BALL_CONFIG (see that var's docstring in config_types/simulator.py) --
# config_values/unified/g1/reward.py bakes per-skill values into RewardTermCfg at Python import
# time for the same reason ball's _w_g does, so this must be set (or not) before the process starts.
HOLOSOMA_SKILLS_CONFIG_ENV_VAR = "HOLOSOMA_SKILLS_CONFIG"

# 2026-08-05, 2-file config split (see configs/task_config*.yaml / configs/skills.example.yaml for
# the target shape). When set, load_multi_skill_config() reads MultiSkillConfig's ~24 GLOBAL fields
# (ball, ood_*, bad_tracking_*, kick_gamma, ...) AND the 15 fields that used to be per-skill
# (randomize_x, success_radius, the 8 per-category reward scales, misc timing/obs fields -- see
# _SHARED_SKILL_DEFAULT_FIELDS below) from THIS file instead of HOLOSOMA_SKILLS_CONFIG's file.
# HOLOSOMA_SKILLS_CONFIG's file, in this mode, is expected to carry ONLY motion_skill_N blocks with
# their 9 genuinely-per-clip fields (motion_npz/x/y/target_x/target_y/strike_start_frame/
# stand_start_frame/motion_training_ratio/kick_foot) -- the 15 shared fields, if a block doesn't
# set them itself, fall back to THIS file's values (uniform across every skill) rather than
# SkillConfig's own hardcoded dataclass defaults; a skill block MAY still override any of the 15
# individually (an escape hatch, not the expected path). config_types/reward_tuning.py's loader
# ALSO reads from this same file when set (its own reward-term-weight sections) -- see that
# module's HOLOSOMA_TASK_CONFIG_ENV_VAR usage. Unset (default): unchanged single-file behavior,
# HOLOSOMA_SKILLS_CONFIG's file carries everything, exactly as before this var existed.
HOLOSOMA_TASK_CONFIG_ENV_VAR = "HOLOSOMA_TASK_CONFIG"

# Overridable via HOLOSOMA_SKILLS_CONFIG when set; that same env var being SET AT ALL (to anything)
# is what multi_skill_mode_enabled() checks to decide whether N-skill mode is active.
#
# 2026-08-23: this used to fall back to a hardcoded configs/stageB_and_C.yaml when the env var was
# unset (same convention as DEFAULT_BALL_CONFIG_YAML). That file was deleted during the
# configs/skill//configs/task/ reorg with no designated replacement, and every real call site of
# load_multi_skill_config() already gates on multi_skill_mode_enabled() first (i.e. only calls this
# when the env var IS set) -- so the fallback was dead in practice, and re-pointing it at an
# arbitrary surviving file would silently pick a "default" training config nobody chose. None here
# means "no default": HOLOSOMA_SKILLS_CONFIG must be set explicitly, and load_multi_skill_config()
# raises immediately below if it's called without one.
DEFAULT_MULTI_SKILL_CONFIG_YAML = (
    Path(os.environ[HOLOSOMA_SKILLS_CONFIG_ENV_VAR]) if HOLOSOMA_SKILLS_CONFIG_ENV_VAR in os.environ else None
)


def multi_skill_mode_enabled() -> bool:
    """True iff HOLOSOMA_SKILLS_CONFIG is set (to any non-empty value) before this process
    started. False (the default) means N-skill training mode is off entirely -- every config_values
    module that would otherwise consult MultiSkillConfig instead uses its pre-existing single-clip
    behavior (load_ball_config/load_skill_mix_config), unchanged."""
    return bool(os.environ.get(HOLOSOMA_SKILLS_CONFIG_ENV_VAR))


# The 15 SkillConfig fields that, in 2-file (HOLOSOMA_TASK_CONFIG) mode, are read ONCE from the
# task-config file and applied uniformly to every skill -- see HOLOSOMA_TASK_CONFIG_ENV_VAR's own
# comment above. Each tuple entry is (field_name, default, caster); the default here is the SAME
# default SkillConfig's own dataclass field already carries (kept explicit, not introspected via
# __dataclass_fields__, so this list is grep-able and obviously in sync by inspection).
_SHARED_SKILL_DEFAULT_FIELDS: tuple[tuple[str, float, type], ...] = (
    ("randomize_x", 0.0, float),
    ("randomize_y", 0.0, float),
    ("success_radius", 0.5, float),
    ("shooting_reward_scale", 0.0, float),
    ("recovery_duration_s", 1.0, float),
    ("hold_duration_s", 2.0, float),
    ("observation_bias", 0.0, float),
    ("motion_tracking_reward_scale", 1.0, float),
    ("root_tracking_reward_scale", 1.0, float),
    ("recovery_tracking_scale", 1.0, float),
    ("kick_recovery_posture_reward_scale", 1.0, float),
    ("kick_safety_reward_scale", 1.0, float),
    ("kick_alive_reward_scale", 1.0, float),
    ("kick_alive_pre_kick_ratio", 1.0, float),
    ("kick_aim_enabled", False, bool),
)


def _parse_shared_skill_defaults(raw: dict) -> dict:
    """2-file mode only: read _SHARED_SKILL_DEFAULT_FIELDS from the task-config file's raw dict.
    Missing keys fall back to the SAME default SkillConfig's own field would use -- so a
    task-config file that doesn't set one of these 15 at all reproduces legacy single-file
    behavior for that one field, not a new default."""
    return {name: caster(raw[name]) if name in raw else default for name, default, caster in _SHARED_SKILL_DEFAULT_FIELDS}


def _parse_multi_skill_global_fields(raw: dict, source_path: Path) -> dict:
    """Extract MultiSkillConfig's ~24 GLOBAL fields (everything except `skills`) from a raw yaml
    dict -- the ball/ood/bad_tracking/RSI/SAC knobs, shared regardless of 1-file or 2-file mode.
    `source_path` is used only for error messages (points at whichever file is actually being read
    -- the combined file in legacy mode, the task-config file in 2-file mode)."""
    ball_raw = raw.get("ball", {})
    radius = float(ball_raw.get("radius", 0.11))
    mass = float(ball_raw.get("mass", 0.43))
    ramp_iters = int(raw.get("shooting_reward_scale_ramp_iters", 0))
    hold_iters = int(raw.get("shooting_reward_scale_hold_iters", 0))
    ball_obs_noise = float(raw.get("ball_obs_noise", 0.05))
    ball_obs_noise_range_coefficient = float(raw.get("ball_obs_noise_range_coefficient", 0.03))
    ball_obs_delay_steps_min = int(raw.get("ball_obs_delay_steps_min", 0))
    ball_obs_delay_steps_max = int(raw.get("ball_obs_delay_steps_max", 3))
    if ball_obs_delay_steps_max < ball_obs_delay_steps_min:
        raise ValueError(
            f"{source_path}: ball_obs_delay_steps_max ({ball_obs_delay_steps_max}) must be >= "
            f"ball_obs_delay_steps_min ({ball_obs_delay_steps_min})"
        )
    ball_obs_hold_steps_min = int(raw.get("ball_obs_hold_steps_min", 0))
    ball_obs_hold_steps_max = int(raw.get("ball_obs_hold_steps_max", 0))
    if ball_obs_hold_steps_min < 0:
        raise ValueError(f"{source_path}: ball_obs_hold_steps_min must be >= 0, got {ball_obs_hold_steps_min}")
    if ball_obs_hold_steps_max < ball_obs_hold_steps_min:
        raise ValueError(
            f"{source_path}: ball_obs_hold_steps_max ({ball_obs_hold_steps_max}) must be >= "
            f"ball_obs_hold_steps_min ({ball_obs_hold_steps_min})"
        )
    ball_obs_stale_probability = float(raw.get("ball_obs_stale_probability", 0.0))
    ood_spawn_probability = float(raw.get("ood_spawn_probability", 0.0))
    ood_region_multiplier = float(raw.get("ood_region_multiplier", 3.0))
    ball_static_obs_probability = float(raw.get("ball_static_obs_probability", 0.0))
    for _name, _prob in (
        ("ball_obs_stale_probability", ball_obs_stale_probability),
        ("ood_spawn_probability", ood_spawn_probability),
        ("ball_static_obs_probability", ball_static_obs_probability),
    ):
        if not 0.0 <= _prob <= 1.0:
            raise ValueError(f"{source_path}: {_name} must be in [0.0, 1.0], got {_prob}")
    kick_contact_force_penalty_floor = float(raw.get("kick_contact_force_penalty_floor", 3.0))
    kick_contact_force_penalty_k = float(raw.get("kick_contact_force_penalty_k", 15.0))
    kick_contact_force_threshold_bodyweight_multiplier = float(
        raw.get("kick_contact_force_threshold_bodyweight_multiplier", 3.0)
    )
    start_at_timestep_zero_prob = float(raw.get("start_at_timestep_zero_prob", 1.0))
    if not 0.0 <= start_at_timestep_zero_prob <= 1.0:
        raise ValueError(
            f"{source_path}: start_at_timestep_zero_prob must be in [0.0, 1.0], got {start_at_timestep_zero_prob}"
        )
    rsi_scope_to_authored_clip = bool(raw.get("rsi_scope_to_authored_clip", False))
    critical_frame_oversampling_prob = float(raw.get("critical_frame_oversampling_prob", 0.0))
    if not 0.0 <= critical_frame_oversampling_prob <= 1.0:
        raise ValueError(
            f"{source_path}: critical_frame_oversampling_prob must be in [0.0, 1.0], got "
            f"{critical_frame_oversampling_prob}"
        )
    critical_frame_sampling_window = int(raw.get("critical_frame_sampling_window", 10))
    if critical_frame_sampling_window < 0:
        raise ValueError(
            f"{source_path}: critical_frame_sampling_window must be >= 0, got {critical_frame_sampling_window}"
        )
    motion_head_velocity_smoothing_frames = int(raw.get("motion_head_velocity_smoothing_frames", 0))
    if motion_head_velocity_smoothing_frames < 0:
        raise ValueError(
            f"{source_path}: motion_head_velocity_smoothing_frames must be >= 0, got "
            f"{motion_head_velocity_smoothing_frames}"
        )
    penalty_curriculum_enabled = bool(raw.get("penalty_curriculum_enabled", True))
    post_flip_termination_grace_steps = float(raw.get("post_flip_termination_grace_steps", 0.0))
    if post_flip_termination_grace_steps < 0.0:
        raise ValueError(
            f"{source_path}: post_flip_termination_grace_steps must be >= 0.0, got "
            f"{post_flip_termination_grace_steps}"
        )
    post_flip_reward_decay_steps = float(raw.get("post_flip_reward_decay_steps", 0.0))
    if post_flip_reward_decay_steps < 0.0:
        raise ValueError(
            f"{source_path}: post_flip_reward_decay_steps must be >= 0.0, got "
            f"{post_flip_reward_decay_steps}"
        )
    kick_target_entropy_ratio = (
        float(raw["kick_target_entropy_ratio"]) if "kick_target_entropy_ratio" in raw else None
    )
    kick_gamma = float(raw["kick_gamma"]) if "kick_gamma" in raw else None
    if kick_gamma is not None and not 0.0 < kick_gamma < 1.0:
        raise ValueError(f"{source_path}: kick_gamma must be in (0.0, 1.0), got {kick_gamma}")
    critic_v_min = float(raw["critic_v_min"]) if "critic_v_min" in raw else None
    critic_v_max = float(raw["critic_v_max"]) if "critic_v_max" in raw else None
    critic_num_atoms = int(raw["critic_num_atoms"]) if "critic_num_atoms" in raw else None
    # Only the PAIR is meaningful -- a support with one end moved and the other left at the
    # preset's own value is almost always a typo, and would silently produce an asymmetric
    # support rather than failing.
    if (critic_v_min is None) != (critic_v_max is None):
        raise ValueError(
            f"{source_path}: critic_v_min and critic_v_max must be set together (got "
            f"critic_v_min={critic_v_min}, critic_v_max={critic_v_max})"
        )
    if critic_v_min is not None and critic_v_max is not None and critic_v_max <= critic_v_min:
        raise ValueError(
            f"{source_path}: critic_v_max ({critic_v_max}) must be > critic_v_min ({critic_v_min})"
        )
    if critic_num_atoms is not None and critic_num_atoms < 2:
        raise ValueError(
            f"{source_path}: critic_num_atoms must be >= 2 (the support needs at least two atoms "
            f"for a finite bin width), got {critic_num_atoms}"
        )
    replay_buffer_sanitize_enabled = bool(raw.get("replay_buffer_sanitize_enabled", False))
    joint_pos_sanity_check_enabled = bool(raw.get("joint_pos_sanity_check_enabled", False))
    joint_pos_sanity_threshold = float(raw.get("joint_pos_sanity_threshold", 20.0))
    if joint_pos_sanity_threshold <= 0.0:
        raise ValueError(
            f"{source_path}: joint_pos_sanity_threshold must be > 0.0, got {joint_pos_sanity_threshold}"
        )
    bad_tracking_swing_only = bool(raw.get("bad_tracking_swing_only", False))
    kick_recovery_termination_handoff = bool(raw.get("kick_recovery_termination_handoff", False))
    bad_tracking_swing_threshold_multiplier = float(raw.get("bad_tracking_swing_threshold_multiplier", 1.0))
    swing_tracking_sigma_multiplier = float(raw.get("swing_tracking_sigma_multiplier", 1.0))
    if swing_tracking_sigma_multiplier <= 0.0:
        raise ValueError(
            f"{source_path}: swing_tracking_sigma_multiplier must be > 0.0 (it divides inside an "
            f"exp() denominator), got {swing_tracking_sigma_multiplier}"
        )
    balance_potential_weight = float(raw.get("balance_potential_weight", 0.0))
    if balance_potential_weight < 0.0:
        raise ValueError(
            f"{source_path}: balance_potential_weight must be >= 0.0 (it is a magnitude; the shaping "
            f"itself is signed), got {balance_potential_weight}"
        )
    kick_terrain_light_rough_proportion = float(raw.get("kick_terrain_light_rough_proportion", 0.0))
    if not 0.0 <= kick_terrain_light_rough_proportion <= 1.0:
        raise ValueError(
            f"{source_path}: kick_terrain_light_rough_proportion must be in [0.0, 1.0], got "
            f"{kick_terrain_light_rough_proportion}"
        )
    kick_terrain_light_rough_max_height = float(raw.get("kick_terrain_light_rough_max_height", 0.008))
    if kick_terrain_light_rough_max_height <= 0.0:
        raise ValueError(
            f"{source_path}: kick_terrain_light_rough_max_height must be > 0.0, got "
            f"{kick_terrain_light_rough_max_height}"
        )
    _kick_eligible_raw = raw.get("kick_eligible_terrain_types")
    if _kick_eligible_raw is not None and not (isinstance(_kick_eligible_raw, list) and _kick_eligible_raw):
        raise ValueError(
            f"{source_path}: kick_eligible_terrain_types must be a non-empty list of terrain-type "
            f"names if set, got {_kick_eligible_raw!r}"
        )
    kick_eligible_terrain_types = (
        tuple(str(n) for n in _kick_eligible_raw) if _kick_eligible_raw is not None else ("flat",)
    )
    # A kick-eligible type that is never generated would silently yield ZERO kick envs -- the
    # partition would put every env in locomotion mode and the run would look like a locomotion-only
    # job for no stated reason. Catch the specific, likely version of that mistake here: opting kick
    # onto light_rough without also generating any light_rough tiles.
    if "light_rough" in kick_eligible_terrain_types and kick_terrain_light_rough_proportion <= 0.0:
        raise ValueError(
            f"{source_path}: kick_eligible_terrain_types includes 'light_rough' but "
            f"kick_terrain_light_rough_proportion is {kick_terrain_light_rough_proportion} -- that "
            "tier would never be generated, so those envs would not exist. Set a non-zero "
            "proportion (and see its docstring: take it out of 'flat''s share, since proportions "
            "are normalized and would otherwise dilute locomotion's terrain mix too)."
        )
    body_push_enabled = bool(raw.get("body_push_enabled", False))
    body_push_interval_min_s = float(raw.get("body_push_interval_min_s", 4.0))
    body_push_interval_max_s = float(raw.get("body_push_interval_max_s", 8.0))
    if body_push_interval_min_s <= 0.0 or body_push_interval_max_s < body_push_interval_min_s:
        raise ValueError(
            f"{source_path}: body_push_interval_min_s/max_s must satisfy 0 < min <= max, got "
            f"({body_push_interval_min_s}, {body_push_interval_max_s})"
        )
    body_push_force_min = float(raw.get("body_push_force_min", 20.0))
    body_push_force_max = float(raw.get("body_push_force_max", 80.0))
    if body_push_force_min < 0.0 or body_push_force_max < body_push_force_min:
        raise ValueError(
            f"{source_path}: body_push_force_min/max must satisfy 0 <= min <= max, got "
            f"({body_push_force_min}, {body_push_force_max})"
        )
    body_push_duration_min_s = float(raw.get("body_push_duration_min_s", 0.05))
    body_push_duration_max_s = float(raw.get("body_push_duration_max_s", 0.20))
    if body_push_duration_min_s <= 0.0 or body_push_duration_max_s < body_push_duration_min_s:
        raise ValueError(
            f"{source_path}: body_push_duration_min_s/max_s must satisfy 0 < min <= max, got "
            f"({body_push_duration_min_s}, {body_push_duration_max_s})"
        )
    body_push_vertical_fraction = float(raw.get("body_push_vertical_fraction", 0.2))
    if not 0.0 <= body_push_vertical_fraction <= 1.0:
        raise ValueError(
            f"{source_path}: body_push_vertical_fraction must be in [0.0, 1.0], got "
            f"{body_push_vertical_fraction}"
        )
    _body_push_body_names_raw = raw.get("body_push_body_names")
    if _body_push_body_names_raw is not None and not (
        isinstance(_body_push_body_names_raw, list) and _body_push_body_names_raw
    ):
        raise ValueError(
            f"{source_path}: body_push_body_names must be a non-empty list of link names if set, "
            f"got {_body_push_body_names_raw!r}"
        )
    body_push_body_names = (
        tuple(str(n) for n in _body_push_body_names_raw) if _body_push_body_names_raw is not None else None
    )
    use_foot_strike_pitch_reference_relative = bool(
        raw.get("use_foot_strike_pitch_reference_relative", False)
    )
    kick_recovery_locomotion_flip_enabled = bool(
        raw.get("kick_recovery_locomotion_flip_enabled", False)
    )
    kick_abort_prob = float(raw.get("kick_abort_prob", 0.0))
    if not (0.0 <= kick_abort_prob <= 1.0):
        raise ValueError(f"{source_path}: kick_abort_prob must be in [0.0, 1.0], got {kick_abort_prob}")
    kick_abort_delay_min_steps = int(raw.get("kick_abort_delay_min_steps", 10))
    kick_abort_delay_max_steps = int(raw.get("kick_abort_delay_max_steps", 60))
    if kick_abort_delay_min_steps < 1:
        raise ValueError(
            f"{source_path}: kick_abort_delay_min_steps must be >= 1 (flipping at tick 0 would "
            f"train recovery from an unsettled post-teleport contact state -- see that field's own "
            f"docstring), got {kick_abort_delay_min_steps}"
        )
    if kick_abort_delay_max_steps < kick_abort_delay_min_steps:
        raise ValueError(
            f"{source_path}: kick_abort_delay_max_steps ({kick_abort_delay_max_steps}) must be >= "
            f"kick_abort_delay_min_steps ({kick_abort_delay_min_steps})"
        )
    # An abort flip is a KICK->LOCOMOTION transition, so it can only happen where that transition
    # exists at all. Catching it here turns a silently-inert config into a startup error.
    if kick_abort_prob > 0.0 and not kick_recovery_locomotion_flip_enabled:
        raise ValueError(
            f"{source_path}: kick_abort_prob is {kick_abort_prob} but "
            "kick_recovery_locomotion_flip_enabled is False -- the abort mechanism reuses that "
            "flip's own path (task_mode switch, pin_zero, _post_flip_step stamping, and the "
            "post_flip_* smoothing keyed off it), so with the flip disabled it would silently do "
            "nothing at all. Enable the flip, or set kick_abort_prob to 0.0."
        )

    mid_episode_kick_entry_prob = float(raw.get("mid_episode_kick_entry_prob", 0.0))
    if not (0.0 <= mid_episode_kick_entry_prob <= 1.0):
        raise ValueError(
            f"{source_path}: mid_episode_kick_entry_prob must be in [0.0, 1.0], got "
            f"{mid_episode_kick_entry_prob}"
        )
    mid_episode_kick_entry_min_steps = int(raw.get("mid_episode_kick_entry_min_steps", 100))
    if mid_episode_kick_entry_min_steps < 0:
        raise ValueError(
            f"{source_path}: mid_episode_kick_entry_min_steps must be >= 0, got "
            f"{mid_episode_kick_entry_min_steps}"
        )
    mid_episode_kick_entry_max_residual = float(raw.get("mid_episode_kick_entry_max_residual", 0.0))
    if mid_episode_kick_entry_max_residual < 0.0:
        raise ValueError(
            f"{source_path}: mid_episode_kick_entry_max_residual must be >= 0.0, got "
            f"{mid_episode_kick_entry_max_residual}"
        )
    pre_kick_decel_steps = float(raw.get("pre_kick_decel_steps", 0.0))
    if pre_kick_decel_steps < 0.0:
        raise ValueError(f"{source_path}: pre_kick_decel_steps must be >= 0.0, got {pre_kick_decel_steps}")
    pre_kick_decel_target = float(raw.get("pre_kick_decel_target", 0.1))
    if pre_kick_decel_target < 0.0:
        raise ValueError(f"{source_path}: pre_kick_decel_target must be >= 0.0, got {pre_kick_decel_target}")
    pre_kick_fallback_timeout_steps = float(raw.get("pre_kick_fallback_timeout_steps", 0.0))
    if pre_kick_fallback_timeout_steps < 0.0:
        raise ValueError(
            f"{source_path}: pre_kick_fallback_timeout_steps must be >= 0.0, got "
            f"{pre_kick_fallback_timeout_steps}"
        )
    pre_kick_reward_ramp_steps = float(raw.get("pre_kick_reward_ramp_steps", 0.0))
    if pre_kick_reward_ramp_steps < 0.0:
        raise ValueError(
            f"{source_path}: pre_kick_reward_ramp_steps must be >= 0.0, got {pre_kick_reward_ramp_steps}"
        )
    pre_kick_termination_grace_steps = float(raw.get("pre_kick_termination_grace_steps", 0.0))
    if pre_kick_termination_grace_steps < 0.0:
        raise ValueError(
            f"{source_path}: pre_kick_termination_grace_steps must be >= 0.0, got "
            f"{pre_kick_termination_grace_steps}"
        )
    pre_kick_reference_blend_steps = float(raw.get("pre_kick_reference_blend_steps", 0.0))
    if pre_kick_reference_blend_steps < 0.0:
        raise ValueError(
            f"{source_path}: pre_kick_reference_blend_steps must be >= 0.0, got "
            f"{pre_kick_reference_blend_steps}"
        )
    mid_episode_kick_entry_ball_fixed = bool(raw.get("mid_episode_kick_entry_ball_fixed", False))
    # The 4 observation-side handoff-discontinuity fixes (2026-08-18) -- see each field's own
    # docstring on MultiSkillConfig above for the measurement that motivates it. All 4 default to
    # exact no-op, same discipline as the pre_kick_* ramp/grace/blend siblings parsed just above.
    pre_kick_obs_ramp_steps = float(raw.get("pre_kick_obs_ramp_steps", 0.0))
    if pre_kick_obs_ramp_steps < 0.0:
        raise ValueError(
            f"{source_path}: pre_kick_obs_ramp_steps must be >= 0.0, got {pre_kick_obs_ramp_steps}"
        )
    obs_target_pos_distance_scale = float(raw.get("obs_target_pos_distance_scale", 0.0))
    if obs_target_pos_distance_scale < 0.0:
        raise ValueError(
            f"{source_path}: obs_target_pos_distance_scale must be >= 0.0 (it divides inside a "
            f"tanh(); 0.0 means 'off', i.e. return the raw offset), got {obs_target_pos_distance_scale}"
        )
    obs_untag_shared_proprioception = bool(raw.get("obs_untag_shared_proprioception", False))
    obs_ball_always_visible = bool(raw.get("obs_ball_always_visible", False))
    post_flip_obs_ramp_steps = float(raw.get("post_flip_obs_ramp_steps", 0.0))
    if post_flip_obs_ramp_steps < 0.0:
        raise ValueError(
            f"{source_path}: post_flip_obs_ramp_steps must be >= 0.0, got {post_flip_obs_ramp_steps}"
        )
    contact_termination_force_threshold = float(raw.get("contact_termination_force_threshold", 0.0))
    if contact_termination_force_threshold < 0.0:
        raise ValueError(
            f"{source_path}: contact_termination_force_threshold must be >= 0.0 (0.0 means "
            f"'inherit the locomotion baseline'), got {contact_termination_force_threshold}"
        )
    post_flip_alive_scale = float(raw.get("post_flip_alive_scale", 1.0))
    if post_flip_alive_scale < 0.0:
        raise ValueError(
            f"{source_path}: post_flip_alive_scale must be >= 0.0 (1.0 means 'no scaling, exact "
            f"no-op'), got {post_flip_alive_scale}"
        )
    warm_start_obs_ramp_steps = float(raw.get("warm_start_obs_ramp_steps", 0.0))
    if warm_start_obs_ramp_steps < 0.0:
        raise ValueError(
            f"{source_path}: warm_start_obs_ramp_steps must be >= 0.0, got {warm_start_obs_ramp_steps}"
        )
    bad_motion_body_pos_threshold = float(raw.get("bad_motion_body_pos_threshold", 0.25))
    if bad_motion_body_pos_threshold <= 0.0:
        raise ValueError(
            f"{source_path}: bad_motion_body_pos_threshold must be > 0.0, got {bad_motion_body_pos_threshold}"
        )
    ee_body_pos_warmup_threshold = float(raw.get("ee_body_pos_warmup_threshold", 0.25))
    if ee_body_pos_warmup_threshold <= 0.0:
        raise ValueError(
            f"{source_path}: ee_body_pos_warmup_threshold must be > 0.0, got {ee_body_pos_warmup_threshold}"
        )
    kick_recovery_drift_deadzone = float(raw.get("kick_recovery_drift_deadzone", 0.15))
    if kick_recovery_drift_deadzone <= 0.0:
        raise ValueError(
            f"{source_path}: kick_recovery_drift_deadzone must be > 0.0, got {kick_recovery_drift_deadzone}"
        )
    # These four are REWARD deadzones (a tolerance band before a penalty starts accruing), not
    # termination thresholds -- 0.0 is a legitimate value here (matches this codebase's own
    # existing deadzone=0.0 defaults elsewhere, e.g. penalty_kick_swing_orientation), unlike the
    # two threshold fields above where 0.0 would be a degenerate always-fire value. Validated
    # >= 0.0, not > 0.0.
    kick_recovery_stand_height_deadzone = float(raw.get("kick_recovery_stand_height_deadzone", 0.015))
    if kick_recovery_stand_height_deadzone < 0.0:
        raise ValueError(
            f"{source_path}: kick_recovery_stand_height_deadzone must be >= 0.0, got "
            f"{kick_recovery_stand_height_deadzone}"
        )
    kick_recovery_stand_orientation_deadzone = float(raw.get("kick_recovery_stand_orientation_deadzone", 0.025))
    if kick_recovery_stand_orientation_deadzone < 0.0:
        raise ValueError(
            f"{source_path}: kick_recovery_stand_orientation_deadzone must be >= 0.0, got "
            f"{kick_recovery_stand_orientation_deadzone}"
        )
    kick_recovery_stand_feet_width_deadzone = float(raw.get("kick_recovery_stand_feet_width_deadzone", 0.03))
    if kick_recovery_stand_feet_width_deadzone < 0.0:
        raise ValueError(
            f"{source_path}: kick_recovery_stand_feet_width_deadzone must be >= 0.0, got "
            f"{kick_recovery_stand_feet_width_deadzone}"
        )
    kick_recovery_stand_knee_width_deadzone = float(raw.get("kick_recovery_stand_knee_width_deadzone", 0.03))
    if kick_recovery_stand_knee_width_deadzone < 0.0:
        raise ValueError(
            f"{source_path}: kick_recovery_stand_knee_width_deadzone must be >= 0.0, got "
            f"{kick_recovery_stand_knee_width_deadzone}"
        )
    kick_swing_orientation_deadzone = float(raw.get("kick_swing_orientation_deadzone", 0.0))
    if kick_swing_orientation_deadzone < 0.0:
        raise ValueError(
            f"{source_path}: kick_swing_orientation_deadzone must be >= 0.0, got "
            f"{kick_swing_orientation_deadzone}"
        )
    kick_swing_torso_orientation_deadzone = float(raw.get("kick_swing_torso_orientation_deadzone", 0.0))
    if kick_swing_torso_orientation_deadzone < 0.0:
        raise ValueError(
            f"{source_path}: kick_swing_torso_orientation_deadzone must be >= 0.0, got "
            f"{kick_swing_torso_orientation_deadzone}"
        )
    kick_ball_velocity_v_ref = (
        float(raw["kick_ball_velocity_v_ref"]) if "kick_ball_velocity_v_ref" in raw else None
    )
    if kick_ball_velocity_v_ref is not None and kick_ball_velocity_v_ref <= 0.0:
        raise ValueError(
            f"{source_path}: kick_ball_velocity_v_ref must be > 0.0 (it is squared into a "
            f"Lorentzian denominator), got {kick_ball_velocity_v_ref}"
        )
    kick_error_ball_to_target_sigma = (
        float(raw["kick_error_ball_to_target_sigma"]) if "kick_error_ball_to_target_sigma" in raw else None
    )
    if kick_error_ball_to_target_sigma is not None and kick_error_ball_to_target_sigma <= 0.0:
        raise ValueError(
            f"{source_path}: kick_error_ball_to_target_sigma must be > 0.0 (it is squared into an "
            f"exp() denominator), got {kick_error_ball_to_target_sigma}"
        )
    kick_ball_velocity_use_latched_peak_speed = bool(
        raw.get("kick_ball_velocity_use_latched_peak_speed", False)
    )
    kick_ball_velocity_use_post_locomotion_gate = bool(
        raw.get("kick_ball_velocity_use_post_locomotion_gate", False)
    )
    kick_ball_over_line_require_has_kicked = bool(raw.get("kick_ball_over_line_require_has_kicked", False))
    kick_aim_theta_ref_deg = float(raw.get("kick_aim_theta_ref_deg", 45.0))
    if kick_aim_theta_ref_deg <= 0.0:
        raise ValueError(f"{source_path}: kick_aim_theta_ref_deg must be > 0.0, got {kick_aim_theta_ref_deg}")
    kick_aim_theta_max_deg = float(raw.get("kick_aim_theta_max_deg", 15.0))
    if not 0.0 < kick_aim_theta_max_deg <= kick_aim_theta_ref_deg:
        raise ValueError(
            f"{source_path}: kick_aim_theta_max_deg must be in (0.0, kick_aim_theta_ref_deg="
            f"{kick_aim_theta_ref_deg}], got {kick_aim_theta_max_deg}"
        )
    kick_aim_nominal_distance_m = float(raw.get("kick_aim_nominal_distance_m", 5.0))
    if kick_aim_nominal_distance_m <= 0.0:
        raise ValueError(
            f"{source_path}: kick_aim_nominal_distance_m must be > 0.0, got {kick_aim_nominal_distance_m}"
        )
    return dict(
        radius=radius,
        mass=mass,
        shooting_reward_scale_ramp_iters=ramp_iters,
        shooting_reward_scale_hold_iters=hold_iters,
        ball_obs_noise=ball_obs_noise,
        ball_obs_noise_range_coefficient=ball_obs_noise_range_coefficient,
        ball_obs_delay_steps_min=ball_obs_delay_steps_min,
        ball_obs_delay_steps_max=ball_obs_delay_steps_max,
        ball_obs_hold_steps_min=ball_obs_hold_steps_min,
        ball_obs_hold_steps_max=ball_obs_hold_steps_max,
        ball_obs_stale_probability=ball_obs_stale_probability,
        ood_spawn_probability=ood_spawn_probability,
        ood_region_multiplier=ood_region_multiplier,
        ball_static_obs_probability=ball_static_obs_probability,
        kick_contact_force_penalty_floor=kick_contact_force_penalty_floor,
        kick_contact_force_penalty_k=kick_contact_force_penalty_k,
        kick_contact_force_threshold_bodyweight_multiplier=kick_contact_force_threshold_bodyweight_multiplier,
        kick_terrain_light_rough_proportion=kick_terrain_light_rough_proportion,
        kick_terrain_light_rough_max_height=kick_terrain_light_rough_max_height,
        kick_eligible_terrain_types=kick_eligible_terrain_types,
        body_push_enabled=body_push_enabled,
        body_push_interval_min_s=body_push_interval_min_s,
        body_push_interval_max_s=body_push_interval_max_s,
        body_push_force_min=body_push_force_min,
        body_push_force_max=body_push_force_max,
        body_push_duration_min_s=body_push_duration_min_s,
        body_push_duration_max_s=body_push_duration_max_s,
        body_push_vertical_fraction=body_push_vertical_fraction,
        body_push_body_names=body_push_body_names,
        start_at_timestep_zero_prob=start_at_timestep_zero_prob,
        rsi_scope_to_authored_clip=rsi_scope_to_authored_clip,
        critical_frame_oversampling_prob=critical_frame_oversampling_prob,
        critical_frame_sampling_window=critical_frame_sampling_window,
        motion_head_velocity_smoothing_frames=motion_head_velocity_smoothing_frames,
        penalty_curriculum_enabled=penalty_curriculum_enabled,
        post_flip_termination_grace_steps=post_flip_termination_grace_steps,
        post_flip_reward_decay_steps=post_flip_reward_decay_steps,
        kick_target_entropy_ratio=kick_target_entropy_ratio,
        kick_gamma=kick_gamma,
        critic_v_min=critic_v_min,
        critic_v_max=critic_v_max,
        critic_num_atoms=critic_num_atoms,
        replay_buffer_sanitize_enabled=replay_buffer_sanitize_enabled,
        joint_pos_sanity_check_enabled=joint_pos_sanity_check_enabled,
        joint_pos_sanity_threshold=joint_pos_sanity_threshold,
        bad_tracking_swing_only=bad_tracking_swing_only,
        kick_recovery_termination_handoff=kick_recovery_termination_handoff,
        bad_tracking_swing_threshold_multiplier=bad_tracking_swing_threshold_multiplier,
        swing_tracking_sigma_multiplier=swing_tracking_sigma_multiplier,
        balance_potential_weight=balance_potential_weight,
        use_foot_strike_pitch_reference_relative=use_foot_strike_pitch_reference_relative,
        kick_recovery_locomotion_flip_enabled=kick_recovery_locomotion_flip_enabled,
        kick_abort_prob=kick_abort_prob,
        kick_abort_delay_min_steps=kick_abort_delay_min_steps,
        kick_abort_delay_max_steps=kick_abort_delay_max_steps,
        mid_episode_kick_entry_prob=mid_episode_kick_entry_prob,
        mid_episode_kick_entry_min_steps=mid_episode_kick_entry_min_steps,
        mid_episode_kick_entry_max_residual=mid_episode_kick_entry_max_residual,
        pre_kick_decel_steps=pre_kick_decel_steps,
        pre_kick_decel_target=pre_kick_decel_target,
        pre_kick_fallback_timeout_steps=pre_kick_fallback_timeout_steps,
        pre_kick_reward_ramp_steps=pre_kick_reward_ramp_steps,
        pre_kick_termination_grace_steps=pre_kick_termination_grace_steps,
        pre_kick_reference_blend_steps=pre_kick_reference_blend_steps,
        mid_episode_kick_entry_ball_fixed=mid_episode_kick_entry_ball_fixed,
        pre_kick_obs_ramp_steps=pre_kick_obs_ramp_steps,
        obs_target_pos_distance_scale=obs_target_pos_distance_scale,
        obs_untag_shared_proprioception=obs_untag_shared_proprioception,
        obs_ball_always_visible=obs_ball_always_visible,
        post_flip_obs_ramp_steps=post_flip_obs_ramp_steps,
        warm_start_obs_ramp_steps=warm_start_obs_ramp_steps,
        contact_termination_force_threshold=contact_termination_force_threshold,
        post_flip_alive_scale=post_flip_alive_scale,
        bad_motion_body_pos_threshold=bad_motion_body_pos_threshold,
        ee_body_pos_warmup_threshold=ee_body_pos_warmup_threshold,
        kick_recovery_drift_deadzone=kick_recovery_drift_deadzone,
        kick_recovery_stand_height_deadzone=kick_recovery_stand_height_deadzone,
        kick_recovery_stand_orientation_deadzone=kick_recovery_stand_orientation_deadzone,
        kick_recovery_stand_feet_width_deadzone=kick_recovery_stand_feet_width_deadzone,
        kick_recovery_stand_knee_width_deadzone=kick_recovery_stand_knee_width_deadzone,
        kick_swing_orientation_deadzone=kick_swing_orientation_deadzone,
        kick_swing_torso_orientation_deadzone=kick_swing_torso_orientation_deadzone,
        kick_ball_velocity_v_ref=kick_ball_velocity_v_ref,
        kick_error_ball_to_target_sigma=kick_error_ball_to_target_sigma,
        kick_ball_velocity_use_latched_peak_speed=kick_ball_velocity_use_latched_peak_speed,
        kick_ball_velocity_use_post_locomotion_gate=kick_ball_velocity_use_post_locomotion_gate,
        kick_ball_over_line_require_has_kicked=kick_ball_over_line_require_has_kicked,
        kick_aim_theta_ref_deg=kick_aim_theta_ref_deg,
        kick_aim_theta_max_deg=kick_aim_theta_max_deg,
        kick_aim_nominal_distance_m=kick_aim_nominal_distance_m,
    )


def _parse_skill_blocks(raw: dict, yaml_path: Path, shared_defaults: dict | None = None) -> list[SkillConfig]:
    """Extract motion_skill_N blocks from a raw yaml dict into SkillConfig objects, in the order
    they appear in the file (Python dicts preserve insertion order from yaml.safe_load, and this
    order becomes each skill's motion_ids index -- keep motion_skill_N numbering contiguous from 1
    for clarity, but the KEY NAME is not parsed for its number; declaration order is what's used).

    `shared_defaults`: None (legacy single-file mode) -- each of the 15 _SHARED_SKILL_DEFAULT_FIELDS
    falls back to SkillConfig's own hardcoded dataclass default per block, exactly as before this
    parameter existed. A dict (2-file mode, from _parse_shared_skill_defaults) -- those 15 fields
    fall back to the SHARED value from the task-config file instead, uniform across every skill,
    UNLESS a block explicitly sets one itself (escape hatch, not the expected path in this mode)."""
    defaults = shared_defaults if shared_defaults is not None else {}
    skill_keys = [k for k in raw.keys() if k.startswith("motion_skill_")]
    if not skill_keys:
        raise ValueError(
            f"Multi-skill config YAML {yaml_path} has no 'motion_skill_N' blocks -- need at least one."
        )

    skills: list[SkillConfig] = []
    for key in skill_keys:
        block = raw[key]
        required = {"motion_npz", "x", "y", "motion_training_ratio", "strike_start_frame", "stand_start_frame"}
        missing = required - block.keys()
        if missing:
            raise ValueError(f"{yaml_path}:{key} is missing required key(s): {sorted(missing)}")

        kick_foot = str(block.get("kick_foot", "right")).lower()
        if kick_foot not in ("left", "right"):
            raise ValueError(f"{yaml_path}:{key}: kick_foot must be 'left' or 'right', got {block['kick_foot']!r}")

        if "randomize_target_x" in block or "randomize_target_y" in block:
            raise ValueError(
                f"{yaml_path}:{key}: randomize_target_x/y was removed 2026-08-22 (azimuth-aim "
                "refactor) -- independent target randomization has no replacement; "
                "kick_aim_enabled=True derives all aim variation from kick_aim_theta instead (see "
                "SkillConfig.kick_aim_enabled's own docstring), and kick_aim_enabled=False now "
                "reads a fixed, unrandomized target. Remove randomize_target_x/y from this block."
            )

        strike_start_frame = int(block["strike_start_frame"])
        stand_start_frame = int(block["stand_start_frame"])
        if not (0 <= strike_start_frame < stand_start_frame):
            raise ValueError(
                f"{yaml_path}:{key}: strike_start_frame={strike_start_frame}, "
                f"stand_start_frame={stand_start_frame} -- require 0 <= strike_start_frame < "
                "stand_start_frame (the <= raw clip length half of this check happens later, once "
                "the npz is actually loaded, in MotionCommand.setup())."
            )

        skills.append(
            SkillConfig(
                motion_npz=str(block["motion_npz"]),
                x=float(block["x"]),
                y=float(block["y"]),
                motion_training_ratio=float(block["motion_training_ratio"]),
                strike_start_frame=strike_start_frame,
                stand_start_frame=stand_start_frame,
                randomize_x=float(block.get("randomize_x", defaults.get("randomize_x", 0.0))),
                randomize_y=float(block.get("randomize_y", defaults.get("randomize_y", 0.0))),
                target_x=float(block["target_x"]) if "target_x" in block else None,
                target_y=float(block["target_y"]) if "target_y" in block else None,
                kick_aim_enabled=bool(block.get("kick_aim_enabled", defaults.get("kick_aim_enabled", False))),
                kick_aim_theta_max_deg=(
                    float(block["kick_aim_theta_max_deg"]) if "kick_aim_theta_max_deg" in block else None
                ),
                kick_foot=kick_foot,
                success_radius=float(block.get("success_radius", defaults.get("success_radius", 0.5))),
                shooting_reward_scale=float(block.get("shooting_reward_scale", defaults.get("shooting_reward_scale", 0.0))),
                recovery_duration_s=float(block.get("recovery_duration_s", defaults.get("recovery_duration_s", 1.0))),
                hold_duration_s=float(block.get("hold_duration_s", defaults.get("hold_duration_s", 2.0))),
                observation_bias=float(block.get("observation_bias", defaults.get("observation_bias", 0.0))),
                motion_tracking_reward_scale=float(
                    block.get("motion_tracking_reward_scale", defaults.get("motion_tracking_reward_scale", 1.0))
                ),
                root_tracking_reward_scale=float(
                    block.get("root_tracking_reward_scale", defaults.get("root_tracking_reward_scale", 1.0))
                ),
                recovery_tracking_scale=float(
                    block.get("recovery_tracking_scale", defaults.get("recovery_tracking_scale", 1.0))
                ),
                kick_recovery_posture_reward_scale=float(
                    block.get(
                        "kick_recovery_posture_reward_scale", defaults.get("kick_recovery_posture_reward_scale", 1.0)
                    )
                ),
                kick_safety_reward_scale=float(
                    block.get("kick_safety_reward_scale", defaults.get("kick_safety_reward_scale", 1.0))
                ),
                kick_alive_reward_scale=float(
                    block.get("kick_alive_reward_scale", defaults.get("kick_alive_reward_scale", 1.0))
                ),
                kick_alive_pre_kick_ratio=float(
                    block.get("kick_alive_pre_kick_ratio", defaults.get("kick_alive_pre_kick_ratio", 1.0))
                ),
                kick_ankle_pitch_correction_enabled=bool(
                    block.get(
                        "kick_ankle_pitch_correction_enabled",
                        defaults.get("kick_ankle_pitch_correction_enabled", True),
                    )
                ),
                task_config=str(block["task_config"]) if "task_config" in block else None,
                motion_head_velocity_smoothing_frames=(
                    int(block["motion_head_velocity_smoothing_frames"])
                    if "motion_head_velocity_smoothing_frames" in block
                    else None
                ),
            )
        )

    ratio_sum = sum(s.motion_training_ratio for s in skills)
    if ratio_sum > 1.0 + 1e-6:
        raise ValueError(
            f"{yaml_path}: sum of motion_training_ratio across all skills is {ratio_sum:.4f}, "
            "must be <= 1.0 (the remainder is locomotion's share)."
        )
    for s, key in zip(skills, skill_keys):
        if not 0.0 <= s.motion_training_ratio <= 1.0:
            raise ValueError(f"{yaml_path}:{key}: motion_training_ratio must be in [0.0, 1.0], got {s.motion_training_ratio}")

    return skills


def _parse_base_robot_fields(raw: dict, yaml_path: Path) -> dict:
    """2026-08-14: extract the top-level ``base_robot:`` block from the SKILLS file's own raw
    dict -- see MultiSkillConfig.base_robot_target_height's own docstring for why this is a
    separate step, always reading ``raw`` (HOLOSOMA_SKILLS_CONFIG's file), not
    ``_parse_multi_skill_global_fields``'s dict (which is ``task_raw`` in 2-file mode)."""
    block = raw.get("base_robot")
    if block is None:
        return {"base_robot_target_height": None, "base_robot_deadzone": None}
    if not isinstance(block, dict):
        raise ValueError(f"{yaml_path}: 'base_robot' must be a mapping, got {type(block).__name__}")
    unknown = set(block.keys()) - {"target_height", "deadzone"}
    if unknown:
        raise ValueError(f"{yaml_path}: 'base_robot' has unrecognized key(s) {sorted(unknown)} -- expected only "
                          "'target_height'/'deadzone'.")
    return {
        "base_robot_target_height": float(block["target_height"]) if "target_height" in block else None,
        "base_robot_deadzone": float(block["deadzone"]) if "deadzone" in block else None,
    }


def _parse_skill_replay_and_l2sp_fields(raw: dict, source_path: Path) -> dict:
    """2026-08-15: extract l2sp_weight/skill_replay_weights from the SKILLS file's own raw dict --
    same "ALWAYS raw (HOLOSOMA_SKILLS_CONFIG's file), never task_raw" contract as
    _parse_base_robot_fields immediately above, for the same reason: both fields are intrinsically
    tied to the skill ROSTER (skill_replay_weights is literally indexed by skill_id, one entry per
    motion_skill_N block; l2sp_weight protects whichever skills that roster names), not to a
    task-config file that could be paired with a different roster entirely. Previously read
    through _parse_multi_skill_global_fields (task_raw in 2-file mode) -- moved here so both are
    configurable directly in HOLOSOMA_SKILLS_CONFIG's file (e.g. multi_skills.yaml) regardless of
    whether HOLOSOMA_TASK_CONFIG is set for the rest of the global fields."""
    l2sp_weight = float(raw.get("l2sp_weight", 0.0))
    if l2sp_weight < 0.0:
        raise ValueError(f"{source_path}: l2sp_weight must be >= 0.0, got {l2sp_weight}")
    skill_replay_weights = [float(w) for w in (raw.get("skill_replay_weights") or [])]
    if any(w < 0.0 for w in skill_replay_weights):
        raise ValueError(f"{source_path}: skill_replay_weights must all be >= 0.0, got {skill_replay_weights}")
    if skill_replay_weights and all(w == 0.0 for w in skill_replay_weights):
        raise ValueError(
            f"{source_path}: skill_replay_weights are all zero -- that would zero out every kick "
            "transition's gradient. Use relative weights, e.g. [8.0, 1.0]."
        )
    return {"l2sp_weight": l2sp_weight, "skill_replay_weights": skill_replay_weights}


def load_multi_skill_config(yaml_path: str | Path | None = DEFAULT_MULTI_SKILL_CONFIG_YAML) -> MultiSkillConfig:
    """Build a MultiSkillConfig, either from ONE combined file (legacy: a top-level `ball:` block
    plus one or more `motion_skill_N:` blocks carrying every field) or from TWO files when
    HOLOSOMA_TASK_CONFIG is set (global fields + reward-term weights in that file, motion_skill_N
    content blocks in `yaml_path`'s file) -- see HOLOSOMA_TASK_CONFIG_ENV_VAR's own comment above
    for the full 2-file mode design.

    ``yaml_path=None`` (the default when HOLOSOMA_SKILLS_CONFIG isn't set -- see
    DEFAULT_MULTI_SKILL_CONFIG_YAML's own comment) raises immediately: there is no implicit
    default multi-skill config file, by design."""
    if yaml_path is None:
        raise RuntimeError(
            f"{HOLOSOMA_SKILLS_CONFIG_ENV_VAR} is not set and load_multi_skill_config() was called "
            "with no explicit yaml_path. There is no default multi-skill config file -- set "
            f"{HOLOSOMA_SKILLS_CONFIG_ENV_VAR} to an explicit configs/skill/*.yaml path, or guard "
            "this call with multi_skill_mode_enabled() first."
        )
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Multi-skill config YAML not found: {yaml_path}\n"
            "Expected a file with one or more 'motion_skill_N:' blocks (plus a 'ball:' block and "
            "the other global fields, unless HOLOSOMA_TASK_CONFIG carries those separately)."
        )
    with open(yaml_path) as f:
        raw = yaml.safe_load(f) or {}

    task_config_path_str = os.environ.get(HOLOSOMA_TASK_CONFIG_ENV_VAR)
    if task_config_path_str:
        task_config_path = Path(task_config_path_str)
        if not task_config_path.exists():
            raise FileNotFoundError(f"HOLOSOMA_TASK_CONFIG points to a missing file: {task_config_path}")
        with open(task_config_path) as f:
            task_raw = yaml.safe_load(f) or {}
        if any(k.startswith("motion_skill_") for k in task_raw.keys()):
            raise ValueError(
                f"{task_config_path}: HOLOSOMA_TASK_CONFIG's file contains 'motion_skill_N' block(s) -- "
                f"those belong in HOLOSOMA_SKILLS_CONFIG's file ({yaml_path}) instead. Likely the two "
                "env vars were swapped."
            )
        global_kwargs = _parse_multi_skill_global_fields(task_raw, task_config_path)
        skills = _parse_skill_blocks(raw, yaml_path, shared_defaults=_parse_shared_skill_defaults(task_raw))
    else:
        global_kwargs = _parse_multi_skill_global_fields(raw, yaml_path)
        skills = _parse_skill_blocks(raw, yaml_path)

    # base_robot is ALWAYS read from the skills file itself (raw), never task_raw -- see
    # _parse_base_robot_fields' own docstring.
    global_kwargs.update(_parse_base_robot_fields(raw, yaml_path))
    # l2sp_weight/skill_replay_weights: same always-raw contract -- see
    # _parse_skill_replay_and_l2sp_fields' own docstring.
    global_kwargs.update(_parse_skill_replay_and_l2sp_fields(raw, yaml_path))

    # kick_aim cross-field validation: needs both a skill's own fields AND the global
    # kick_aim_theta_ref_deg, so it can only happen here, after both are parsed -- see
    # SkillConfig.kick_aim_theta_max_deg's own docstring for why this check exists (an
    # out-of-normalization-range per-skill override).
    theta_ref = global_kwargs["kick_aim_theta_ref_deg"]
    for skill_key, s in zip((k for k in raw.keys() if k.startswith("motion_skill_")), skills):
        if s.kick_aim_theta_max_deg is not None and not 0.0 < s.kick_aim_theta_max_deg <= theta_ref:
            raise ValueError(
                f"{yaml_path}:{skill_key}: kick_aim_theta_max_deg must be in (0.0, "
                f"kick_aim_theta_ref_deg={theta_ref}], got {s.kick_aim_theta_max_deg}"
            )

    return MultiSkillConfig(skills=skills, **global_kwargs)
