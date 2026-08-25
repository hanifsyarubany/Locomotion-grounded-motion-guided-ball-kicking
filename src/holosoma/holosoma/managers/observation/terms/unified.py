"""Observation terms specific to UnifiedManager (locomotion + ball-kicking)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from holosoma.utils.rotations import quat_rotate_inverse, yaw_quat

if TYPE_CHECKING:
    from holosoma.envs.unified.unified_manager import UnifiedManager


def task_mode_onehot(env: UnifiedManager) -> torch.Tensor:
    """One-hot [num_envs, 2] encoding of each env's current task mode: [1, 0] for locomotion,
    [0, 1] for kick. Lets a single policy condition its behavior on which task it's running,
    rather than only inferring it indirectly from which observation blocks are zeroed.

    2026-08-18 (FIX 3's coherence pairing, ``MultiSkillConfig.pre_kick_obs_ramp_steps``): when the
    observation ramp is active this returns a SOFT ``[1 - alpha, alpha]`` during the crossfade
    window instead of a hard one-hot. This is deliberate and load-bearing rather than cosmetic --
    ramping the gated blocks while leaving this term binary would tell the policy "you are 100% in
    kick mode" on the very tick the ball is only 40% faded in, which is a state combination it is
    never trained on at steady state and reintroduces a (smaller) discontinuity of its own. Soft,
    the pair reads coherently as "belief in kick mode fading in".

    Still sums to exactly 1.0 across the two entries at every point of the ramp, so it remains a
    valid distribution over modes rather than becoming an arbitrary 2-vector. Exact no-op when the
    ramp is off (``pre_kick_obs_ramp_alpha()`` returns None) -- the hard one-hot path below is then
    the identical pre-existing computation. Deliberately NOT tagged with a task_mode itself (it
    never was): this term must stay live in both modes, which is exactly what makes it the channel
    carrying mode identity once ``obs_untag_shared_proprioception`` stops the proprioception blocks
    from implicitly signalling the mode by being zeroed.

    Returns:
        Tensor of shape [num_envs, 2]
    """
    is_kick = env.task_mode_mask("kick").float()
    ramp_alpha_fn = getattr(env, "pre_kick_obs_ramp_alpha", None)
    if ramp_alpha_fn is not None:
        alpha = ramp_alpha_fn()
        if alpha is not None:
            is_kick = torch.where(env.task_mode_mask("kick"), alpha, is_kick)
    return torch.stack([1.0 - is_kick, is_kick], dim=-1)


def _ball_actor_indices(env: UnifiedManager) -> torch.Tensor:
    indices = getattr(env, "_ball_obs_actor_indices", None)
    if indices is None:
        indices = env.simulator.get_actor_indices("ball", env_ids=None)
        env._ball_obs_actor_indices = indices
    return indices


def _ball_pos_b_raw(env: UnifiedManager) -> torch.Tensor:
    """Ground-truth ball position relative to the robot root, in the robot's HEADING frame
    (yaw-only rotation of the base frame — decouples the reading from body pitch/roll during the
    kick swing, which would otherwise make a stationary ball appear to move as the torso tilts).
    [num_envs, 3]. No perception noise, delay, or bias -- see ``ball_pos_b``/
    ``ball_pos_b_ground_truth`` for the two perception-fidelity-aware wrappers around this."""
    root_states = env.simulator.robot_root_states
    ball_pos_w = env.simulator.all_root_states[_ball_actor_indices(env)][:, :3]
    rel = ball_pos_w - root_states[:, :3]
    return quat_rotate_inverse(yaw_quat(env.base_quat, w_last=True), rel, w_last=True)


def ball_pos_b(env: UnifiedManager) -> torch.Tensor:
    """Ball position relative to the robot root, in the robot's HEADING frame -- see
    ``_ball_pos_b_raw`` for the core transform. [num_envs, 3]. This is the ACTOR-facing version:
    also applies the per-episode constant perception bias (below). Registered for the actor_obs
    group only -- the critic_obs group uses ``ball_pos_b_ground_truth`` (same transform, no
    bias), since the critic gets privileged, clean state, the same principle the noise/delay
    mechanisms in ObservationManager already follow via ``group.enable_noise`` (bias can't use
    that same external gate since it's ball-specific per-skill state, not a generic ObsTermCfg
    field, so it's gated here, at the two-function-registration level, instead).

    LIVE in every stage, including Stage B (2026-07-21 user directive — see
    `stagec_obs_normalizer_shock.md`). Matches RoboNaldo's own arrangement: their observation
    vector is a constant 547 dims with ball/target live from Stage 1 onward (arXiv:2606.11092
    Table A.1/2 — Stage 1 already gets 27.9% ball contact), and only the reward weight (``w_g``)
    changes between stages. Stage B keeps the ball UNRANDOMIZED (configs/ball.yaml's
    ``randomize_x``/``randomize_y`` == 0.0, so this reads a fixed in-scene ball each episode);
    Stage C turns on ``randomize_x``/``randomize_y`` (and ``shooting_reward_scale``) without this
    observation's computation changing at all.

    This replaces a previous design (zero this out while ``shooting_reward_scale == 0``, then fade
    it back in on a Stage-C resume) that avoided one failure mode -- an unrewarded-but-live input
    picking up arbitrary weights during Stage B -- at the cost of a worse one: holding these dims
    at a hard-constant zero for an entire Stage-B run collapses `EmpiricalNormalization`'s running
    std for them, so the first real value received on a Stage-C resume normalized to +/-100-170,
    30-100x anything the network had ever seen (full numbers in `stagec_obs_normalizer_shock.md`).
    Keeping the input distribution identical across the stage boundary removes that discontinuity
    outright rather than patching around it. This is what turns ball-spawn randomization from
    irreducible reward noise into learnable variation: with the ball observed, the policy can adapt
    its strike to where the ball actually is. At deployment, feed this from the ball pose source
    (perception / mocap) in the same heading-frame convention, and it must be genuinely dynamic,
    not a constant offset.
    """
    obs = _ball_pos_b_raw(env)
    # Per-episode constant heading-frame perception bias (BallConfig.observation_bias; drawn in
    # managers/randomization/terms/locomotion.py::randomize_ball_obs_bias). None/zero => no change.
    bias = getattr(env, "_ball_obs_bias", None)
    if bias is not None:
        obs = obs + bias

    # 2026-07-24 deployment-robustness training: a per-episode FROZEN/static mode (stuck/dead
    # perception pipeline) -- see randomize_ball_obs_freeze's own docstring for the full
    # rationale. When env._ball_obs_frozen_mask is set for an env, this returns that env's
    # captured-once STATIC reading for the REST of the episode, instead of a fresh live read each
    # step -- "static, not moving", per the user's own description.
    #
    # 2026-07-24 (revised same day, user directive): the captured value is drawn INDEPENDENTLY of
    # the ball's actual simulated position (nominal + draw_position_noise_with_ood, the SAME
    # mechanism/probability a real spawn uses), NOT a snapshot of the live reading as the first
    # version of this did. That first version only showed an OOD-looking frozen value when the
    # SAME episode's real ball placement also happened to roll OOD -- a rare (~ood_prob *
    # static_prob) coincidence, not a deliberate, independent chance. A genuinely stuck/broken
    # sensor can show an arbitrary reading unrelated to reality, so the frozen value should be
    # able to land in the OOD region on its own, decoupled from where the ball truly is.
    frozen_mask = getattr(env, "_ball_obs_frozen_mask", None)
    if frozen_mask is not None and bool(frozen_mask.any()):
        captured = env._ball_obs_frozen_captured
        value = env._ball_obs_frozen_value
        to_capture = frozen_mask & ~captured
        if bool(to_capture.any()):
            value[to_capture] = _draw_independent_frozen_ball_reading(env, to_capture)
            captured[to_capture] = True
        # captured now covers every currently-frozen env (either already true, or just set
        # above), so frozen_mask alone is the correct selector here -- applying it starting on
        # the VERY FIRST frozen step, not one step late (a real bug the first version of this had:
        # computing this selector BEFORE updating captured meant the just-drawn value was
        # silently skipped on its own capture step, only taking effect from the second call on).
        obs = torch.where(frozen_mask.unsqueeze(-1), value, obs)
    return obs


def _draw_independent_frozen_ball_reading(env: UnifiedManager, mask: torch.Tensor) -> torch.Tensor:
    """Draw a STATIC ball_pos_b-style reading independently of the ball's actual simulated
    position for the envs selected by ``mask`` -- nominal local (x, y) + a noise draw that may
    itself land in the OOD region (see ``holosoma.managers.command.terms.wbt.
    draw_position_noise_with_ood``, the SAME mechanism/probability a real ball spawn uses),
    decoupled from whether THIS episode's real ball placement also happened to roll OOD. Called
    from ``ball_pos_b`` at the moment a frozen env's reading is first captured this episode (by
    which point ``motion_command.motion_ids`` is guaranteed fresh -- ``command_manager.reset()``
    always runs before the first post-reset observation compute).
    """
    from holosoma.managers.command.terms.wbt import draw_position_noise_with_ood

    motion_command = env.command_manager.get_state("motion_command")
    env_motion_ids = motion_command.motion_ids[mask]
    nominal_local = motion_command.ball_reset_state_per_motion[env_motion_ids, :2]
    position_randomization = motion_command.ball_position_randomization_per_motion[env_motion_ids]
    # is_ood intentionally discarded: this mechanism (ball_static_obs_probability) models a
    # frozen/stuck reading decoupled from the real ball, so whether THIS synthetic reading landed
    # in the OOD region has no bearing on the real ball state shooting rewards are computed from
    # -- see draw_position_noise_with_ood's own docstring.
    noise, _ = draw_position_noise_with_ood(
        position_randomization,
        ood_prob=motion_command.motion_cfg.ood_spawn_probability,
        ood_multiplier=motion_command.motion_cfg.ood_region_multiplier,
        device=env.device,
    )
    frozen_xy = nominal_local + noise
    frozen_z = torch.zeros(int(mask.sum().item()), 1, device=env.device)
    return torch.cat([frozen_xy, frozen_z], dim=-1)


def ball_pos_b_ground_truth(env: UnifiedManager) -> torch.Tensor:
    """Ball position relative to the robot root, in the robot's HEADING frame -- the CRITIC-facing
    version. Identical to ``ball_pos_b`` (same ``_ball_pos_b_raw`` transform) but WITHOUT the
    per-episode perception bias: the critic estimates value from privileged, clean simulator
    state, so it should never see simulated perception artifacts, only the actor (the thing that
    actually gets deployed) should. Register this for critic_obs's ``kick_ball_pos_b`` term
    instead of ``ball_pos_b`` -- see config_values/unified/g1/observation.py.

    2026-07-24: fixes a real, pre-existing inconsistency -- until this function existed, BOTH
    actor_obs and critic_obs called the same ``ball_pos_b``, so the bias (unlike noise/delay,
    which are correctly gated by ObservationManager's per-group ``enable_noise`` check) was
    leaking into the critic's supposedly-privileged ground truth.
    """
    return _ball_pos_b_raw(env)


def target_pos_b(env: UnifiedManager, distance_scale: float = 0.0) -> torch.Tensor:
    """Commanded shot target relative to the robot root, in the robot's heading frame
    (ground-plane xy only — the target has no height). [num_envs, 2].

    LIVE in every stage, including Stage B — see ``ball_pos_b`` for why (same fix, same reasoning,
    2026-07-21 user directive). Under ``kick_aim_enabled=False`` this always reads target's fixed
    nominal point every episode (the old ``target_randomization`` independent-draw mechanism was
    removed 2026-08-22 -- see SkillConfig.kick_aim_enabled's own docstring); aim variation now
    comes exclusively from ``kick_aim_enabled=True`` skills' ``kick_aim_theta`` (see
    ``kick_aim_command`` below, which replaces this function's own obs slot for those skills).

    Reads MotionCommand's per-env target_xy_w — the SAME randomized draw the shooting reward
    terms score against (managers/reward/terms/shooting.py), so the policy is always rewarded
    against exactly the target it observes. Informative only when configs/ball.yaml randomizes
    the target (a constant offset is just a bias the network absorbs); with randomization on,
    this input is what makes the shooter steerable — command a different target at deployment
    and the policy aims there.

    ``distance_scale`` (2026-08-18, FIX 1 of the handoff observation-discontinuity work — see
    ``MultiSkillConfig.obs_target_pos_distance_scale``'s own docstring for the measurement):

    * ``0.0`` (default) — return the RAW heading-frame offset in metres, the exact pre-existing
      behavior, via the identical code path. Exact no-op.
    * ``> 0.0`` — return ``unit_direction * tanh(distance / distance_scale)``: same 2 dims (so the
      observation width, and therefore checkpoint warm-start, is untouched), direction preserved
      exactly, magnitude bounded in [0, 1] (tanh never reaches 1 mathematically, but saturates to
      exactly 1.0 in float32 once the ratio is large — bounded either way).

    Why this matters: at ``scale=1.0`` and a 5-7 m target these 2 dims read ~7.27 while nearly every
    other observation is O(1), and a live probe attributed **51% of the entire locomotion->kick
    observation discontinuity** to this one unnormalized term. Compression is ``tanh`` rather than a
    smaller constant ``scale`` because a constant still grows without bound (a 20 m target is a
    large input again) whereas ``tanh`` saturates gracefully while staying strictly monotone in
    distance — the policy can still tell "far" from "very far", just with diminishing resolution.

    The zero-distance case is well-defined: as ``distance -> 0`` the direction is undefined but
    ``tanh(0) == 0`` drives the whole vector to 0 anyway, so the epsilon-guarded division below
    cannot produce a NaN or an arbitrary unit vector with nonzero magnitude.
    """
    motion_command = env.command_manager.get_state("motion_command")
    target_xy_w = getattr(motion_command, "target_xy_w", None)
    if target_xy_w is None:  # no ball in the scene (e.g. a locomotion-only debugging config)
        return torch.zeros(env.num_envs, 2, device=env.device)
    root_states = env.simulator.robot_root_states
    rel = torch.zeros(env.num_envs, 3, device=env.device)
    rel[:, :2] = target_xy_w - root_states[:, :2]
    offset_b = quat_rotate_inverse(yaw_quat(env.base_quat, w_last=True), rel, w_last=True)[:, :2]
    if distance_scale <= 0.0:
        return offset_b
    distance = torch.linalg.vector_norm(offset_b, dim=-1, keepdim=True)
    # clamp_min guards ONLY the division; the numerator's own magnitude still goes to 0 with the
    # tanh, so a robot standing exactly on the target reads [0, 0] rather than a unit vector.
    unit_direction = offset_b / distance.clamp_min(1e-6)
    return unit_direction * torch.tanh(distance / distance_scale)


def kick_aim_command(env: UnifiedManager, distance_scale: float = 0.0) -> torch.Tensor:
    """Commanded strike direction, [num_envs, 2] -- the azimuth-command replacement for
    ``target_pos_b`` (2026-08-22 azimuth-aim refactor; see ``SkillConfig.kick_aim_enabled``'s own
    docstring for the full mechanism this observes).

    PER-ENV, by that env's assigned skill's ``kick_aim_enabled``:
      * False (legacy, untouched): falls through to ``target_pos_b`` verbatim (same function
        call, same ``distance_scale`` param) -- bit-identical to before this function existed, for
        any skill/config that doesn't opt in.
      * True: returns ``[kick_aim_theta / kick_aim_theta_ref_deg, 0.0]`` -- a CONSTANT for the
        whole attempt (``kick_aim_theta`` is sampled once at reset/clip-entry and held, unlike
        ``target_pos_b``'s live per-tick world-frame transform), bounded in [-1, 1] by
        construction (kick_aim_theta_max_deg <= kick_aim_theta_ref_deg is validated at config load
        time), needing no robot localization to compute or to reproduce at deployment. dim 1 is
        RESERVED (always 0.0 here) for a possible future per-skill elevation offset -- kept so
        that slot exists without another observation-width change if that's ever added.

    MIXED-MODE CAVEAT: if a run has some skills kick_aim_enabled and others not, this single 2-dim
    slot carries genuinely different units for different envs simultaneously (a normalized angle
    vs. a raw metre offset) -- there is no third bit here to disambiguate beyond
    ``task_mode_onehot``'s kick-vs-locomotion split. Harmless for a run where every kick skill
    shares one mode (the case this project uses today), but a real limitation to know about before
    mixing modes within one run.

    Registered under the SAME ``_shooting_task_mode`` gate as ``kick_ball_pos_b`` (see
    config_values/unified/g1/observation.py) rather than an unconditional ``task_mode=None`` --
    hardcoding always-on would un-gate this term for every skill regardless of kick_aim_enabled,
    which is not bit-identical for a run that doesn't use this mechanism at all. A constant command
    has no discontinuity to gate away in principle, so the intended "live from step 0" behavior is
    available -- via ``MultiSkillConfig.obs_ball_always_visible`` (already yaml-configurable,
    already shared with kick_ball_pos_b), not a new hardcoded default.
    """
    legacy = target_pos_b(env, distance_scale=distance_scale)

    motion_command = env.command_manager.get_state("motion_command")
    kick_aim_enabled_per_motion = getattr(motion_command, "kick_aim_enabled_per_motion", None)
    if kick_aim_enabled_per_motion is None:  # mechanism absent entirely (e.g. no ball in the scene)
        return legacy

    motion_ids = motion_command.motion_ids
    aim_enabled = kick_aim_enabled_per_motion[motion_ids]
    theta_ref = motion_command.kick_aim_theta_ref_deg
    aim_dim0 = (motion_command.kick_aim_theta / theta_ref).unsqueeze(-1)
    aim = torch.cat([aim_dim0, torch.zeros_like(aim_dim0)], dim=-1)

    return torch.where(aim_enabled.unsqueeze(-1), aim, legacy)
