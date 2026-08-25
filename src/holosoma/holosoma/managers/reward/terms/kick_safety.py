"""Contact-safety penalties for the kick task, aimed at sim2real transfer rather than shooting
skill (see managers/reward/terms/shooting.py for the RoboNaldo-style scoring rewards).

Motivation: a Stage-B unified checkpoint (model_0071000, unified-stageB-detq-fromscratch) that
looked robust in wandb rollouts and in the IsaacGym/IsaacSim training env collapsed almost every
time in MuJoCo sim2sim -- not by losing balance, but by launching itself several meters into the
air roughly 0.6s into the kick. The pelvis-height trace during that launch is a smooth,
continuously-*accelerating* climb over ~1-2s, not a discontinuous velocity jump followed by plain
ballistic decay -- i.e. not a single hard impact, but a restoring force that keeps growing the
longer the stance foot's contact is fought. That is consistent with the plant foot sinking into
the ground under load while the trained policy keeps commanding through the correction: PhysX
(IsaacGym/IsaacSim) tolerated this contact behaviour during training (hence "robust" rollouts),
MuJoCo's solver did not.

Both terms below directly target the physical quantity involved -- foot/ground contact force and
foot touchdown speed -- so the fix generalizes to any sufficiently strict contact model, including
real hardware, rather than being tuned to satisfy one simulator's solver. Both act on
``env.feet_indices`` (both feet, not just the configured kick foot): the stance leg is the leading
suspect, but the kicking foot also strikes the ground hard during wind-up/recovery, and sim2real
robustness benefits from constraining both regardless.

Registered with task_mode="kick" only (see config_values/unified/g1/reward.py) -- locomotion's
reward function is left untouched per the standing instruction that Stage B's reward stay scoped
to "reference tracking from the motion clip, and post kick stabilization". Neither term is gated
by configs/ball.yaml's shooting_reward_scale: contact safety is a basic stability property of the
kick, not a shooting-skill shaping term, so it stays active in Stage B (shooting_reward_scale=0)
as well as Stage C.
"""

from __future__ import annotations

from typing import Any

import torch

from holosoma.utils.kick_reward_scales import kick_safety_reward_scale
from holosoma.utils.shooting_curriculum import current_w_g

_G1_29DOF_MASS_KG = 33.34
"""Summed body mass from data/robots/g1/g1_29dof.xml -- used only to express force thresholds as
bodyweight multiples in docstrings/defaults, not read from the robot at runtime."""

_G1_29DOF_WEIGHT_N = _G1_29DOF_MASS_KG * 9.81


def penalty_excess_foot_contact_force(
    env: Any,
    force_threshold_bodyweight_multiplier: float = 3.0,
    floor: float = 3.0,
    k: float = 15.00,
) -> torch.Tensor:
    """Penalize per-foot ground-reaction force beyond ``force_threshold_bodyweight_multiplier``
    times G1's own bodyweight.

    Default threshold is 3x G1's bodyweight (~980N) -- generous headroom for legitimate dynamic
    single-support loading during a kick (athletic single-leg ground-reaction forces commonly
    reach 2-4x bodyweight), so normal stance/landing forces contribute exactly zero. Only force
    beyond that is penalized, normalized by bodyweight (so the returned magnitude is a
    dimensionless "how many bodyweights over the limit", comparable in scale to this file's other
    penalty terms) and squared (smooth near the threshold, steep for genuinely runaway force).

    ``floor``/``k`` implement this term's dynamic magnitude: ``(floor + k * current_w_g(env))``,
    where ``current_w_g`` is the LIVE (possibly still-ramping) shooting_reward_scale -- see
    ``holosoma.utils.shooting_curriculum`` and config_values/unified/g1/reward.py's module
    docstring for why this must stay active (at ``floor``) throughout Stage B and scale UP with
    shooting intensity rather than being scaled by ``_w_g`` the same way the shooting-skill terms
    are. Returned already multiplied by this dynamic magnitude, matched with
    ``RewardTermCfg.weight = -1.0`` in reward.py (moved OUT of the static config-time weight
    because it must track the LIVE ramp value, not a value baked in at config-import time).

    IMPORTANT (2026-07-28): this coupling alone does NOT prevent the momentum-throw exploit at a
    higher shooting_reward_scale -- see MultiSkillConfig.kick_contact_force_penalty_k's docstring
    for the measured counterexample (w_g 0.8->3.0, floor/k unchanged, MuJoCo survival still
    collapsed). ``floor``/``k``/``force_threshold_bodyweight_multiplier`` are yaml-configurable
    (configs/stageB_and_C.yaml's global fields / configs/ball*.yaml) precisely because raising w_g
    safely requires retuning these ALONGSIDE it, not relying on the formula to self-balance.

    Args:
        env: The environment instance.
        force_threshold_bodyweight_multiplier: Per-foot force threshold, as a multiple of the
            robot's own bodyweight, above which excess is penalized.
        floor: Stage-B-adequate base penalty magnitude, active regardless of w_g.
        k: Coefficient on the live shooting_reward_scale value.

    Returns:
        Reward tensor [num_envs] (positive excess magnitude; pair with a negative weight, same
        convention as ``penalty_stance_asymmetry``).
    """
    force_threshold_n = force_threshold_bodyweight_multiplier * _G1_29DOF_WEIGHT_N
    forces = env.simulator.contact_forces[:, env.feet_indices, :]  # [E, F, 3]
    force_mag = torch.norm(forces, dim=-1)  # [E, F]
    # Upper-clamped at 1000 (2026-07-18): excess was previously clamped only at the bottom
    # (min=0.0), leaving it unbounded above. A genuinely bad kick contributes excess in the
    # single-to-low-double digits; 1000 is already far beyond that and only ever engages for a
    # transient, otherwise-self-correcting physics anomaly (confirmed on simulator:mjwarp: even
    # after fixing mujoco_warp's own tolerance-loosening bug -- see warp_backend.py -- occasional
    # large-but-recoverable contact-force spikes, up to ~9500N, still occur under real stochastic
    # training actions). Squaring an unclamped excess of that scale can overflow float32 (excess
    # only needs to reach ~1.8e19 for its square to hit float32's ~3.4e38 max), producing inf/nan
    # that a single poisoned replay-buffer transition then propagates into the shared SAC network
    # on the next gradient step -- gradient clipping does NOT catch this, since clip_grad_norm_
    # can't repair a gradient that's already nan/inf, only rescale one that's large-but-finite.
    # 1000 keeps the normal training signal completely unchanged while removing the overflow path.
    excess = torch.clamp((force_mag - force_threshold_n) / _G1_29DOF_WEIGHT_N, min=0.0, max=1000.0)
    return torch.sum(torch.square(excess), dim=-1) * (floor + k * current_w_g(env)) * kick_safety_reward_scale(env)


def penalty_hard_foot_landing(
    env: Any,
    impact_speed_threshold_mps: float = 2.0,
    contact_force_threshold_n: float = 30.0,
    floor: float = 3.0,
    k: float = 15.00,
) -> torch.Tensor:
    """Penalize a foot's own speed at the exact instant it makes new ground contact.

    Complements ``penalty_excess_foot_contact_force``: that term catches sustained/quasi-static
    overload once in contact, this one catches a fast impact itself (impulsive force scales with
    impact velocity, and a hard touchdown is what most directly seeds the runaway penetration this
    file's module docstring describes). "New contact" is detected as a per-foot, per-env rising
    edge of (contact force > contact_force_threshold_n), tracked across steps on the env instance.

    2.0 m/s is generous headroom over normal walking/kick-recovery touchdown speeds (commonly
    well under 1 m/s); only touchdowns clearly faster than that are penalized.

    ``floor``/``k``: see ``penalty_excess_foot_contact_force``'s docstring -- same dynamic
    ``(floor + k * current_w_g(env))`` magnitude, same reason it lives here rather than in a
    config-time-static ``RewardTermCfg.weight``.

    Args:
        env: The environment instance.
        impact_speed_threshold_mps: Foot speed (m/s) at touchdown above which excess is penalized.
        contact_force_threshold_n: Force magnitude used only to detect "now in contact" (not a
            safety threshold itself -- kept low so real touchdowns are caught promptly).
        floor: Stage-B-adequate base penalty magnitude, active regardless of w_g.
        k: Coefficient on the live shooting_reward_scale value.

    Returns:
        Reward tensor [num_envs] (positive excess magnitude; pair with a negative weight, same
        convention as ``penalty_stance_asymmetry``).
    """
    forces = env.simulator.contact_forces[:, env.feet_indices, :]  # [E, F, 3]
    in_contact = torch.norm(forces, dim=-1) > contact_force_threshold_n  # [E, F]

    prev = getattr(env, "_kick_prev_foot_contact", None)
    if prev is None or prev.shape != in_contact.shape:
        prev = torch.zeros_like(in_contact)
    # Skip the first step after a reset: prev may hold a stale reading from whatever env/episode
    # last used this slot, which would otherwise register a spurious (or miss a real) rising edge.
    just_reset = env.episode_length_buf <= 1
    new_contact = in_contact & ~prev & ~just_reset.unsqueeze(-1)
    env._kick_prev_foot_contact = in_contact.clone()

    foot_speed = torch.norm(env.simulator._rigid_body_vel[:, env.feet_indices, :], dim=-1)  # [E, F]
    # Upper-clamped at 1000 -- see penalty_excess_foot_contact_force's comment for why (same
    # unbounded-square-can-overflow-float32-into-nan failure mode, same fix).
    excess = torch.clamp(foot_speed - impact_speed_threshold_mps, min=0.0, max=1000.0) * new_contact.float()
    return torch.sum(torch.square(excess), dim=-1) * (floor + k * current_w_g(env)) * kick_safety_reward_scale(env)


def penalty_excess_base_lin_vel(
    env: Any, speed_threshold_mps: float = 1.0, floor: float = 3.0, k: float = 33.75
) -> torch.Tensor:
    """Penalize the robot's own base (pelvis) horizontal speed during a kick beyond
    ``speed_threshold_mps``.

    Motivation: a Stage-C checkpoint (prepend-fixed motion config, resumed from a Stage-B
    checkpoint that itself kicks and recovers cleanly) falls in MuJoCo sim2sim -- not on trigger
    (the prepend fix confirmed correct: the exported ONNX's ``joint_pos`` reference starts exactly
    at ``default_dof_pos`` and ramps smoothly), but ~1-1.5s into the swing, with lean building
    monotonically to 40-50+deg. Direct comparison of the pre-Stage-C checkpoint (121k, survives,
    max lean 13.5deg, peak horizontal base speed ~0.5 m/s) against later Stage-C checkpoints
    (148k/175k/202k/228k, all fall, falling EARLIER and more violently as training progresses)
    shows the base's own horizontal speed climbing to ~2 m/s by the time it topples -- i.e. Stage C
    is teaching the policy to throw its own body momentum into the kick (bigger hip drive, more
    forward lunge), presumably because that also makes the ball go faster/straighter and
    ``kick_goal_success_burst`` (weight 300*shooting_reward_scale, an order of magnitude larger
    than every other shooting term) rewards exactly that outcome.

    Nothing in training was catching this. ``bad_tracking`` here is ``BadTrackingZOnly`` (see
    config_values/unified/g1/termination.py) -- deliberately Z-axis-only, since the real robot has
    no reliable global XY position estimate and training shouldn't terminate on a divergence
    deployment could never even measure. But that also makes it fully blind to exactly this
    failure: a growing horizontal (XY) launch is invisible to a Z-only position/height check, and
    ``bad_ref_ori``'s threshold (0.8, on a relative-to-the-clip's-own-lean orientation error) did
    not fire fast enough to stop the incentive from compounding across training. This term closes
    that gap the same way the two terms above do: penalize the physical quantity directly (here,
    the robot's own root speed) rather than touching the termination's deliberate Z-only design.

    1.0 m/s threshold: matches locomotion's own max commanded walking speed
    (``lin_vel_x`` range is [-1.0, 1.0] m/s) -- a kick is a stationary task with no velocity
    command, so any base speed is emergent, not deliberately walked, and exceeding the fastest
    speed the robot is ever asked to walk at is already a legitimate red flag. Comfortable
    headroom over the measured stable case (~0.5 m/s peak) and decisively below the measured
    runaway case (~2.0 m/s).

    ``floor``/``k``: see ``penalty_excess_foot_contact_force``'s docstring -- same dynamic
    ``(floor + k * current_w_g(env))`` magnitude. This term keeps the largest ``k`` (see the module
    docstring on ``_KICK_SAFETY_FLOOR`` usage in config_values/unified/g1/reward.py for why: it
    accumulates over the ~50-75 step swing rather than firing on a single frame, so it needs a
    bigger multiplier for its accumulated total to matter against the shooting terms' one-time burst).

    Args:
        env: The environment instance.
        speed_threshold_mps: Base horizontal speed (m/s) above which excess is penalized.
        floor: Stage-B-adequate base penalty magnitude, active regardless of w_g.
        k: Coefficient on the live shooting_reward_scale value.

    Returns:
        Reward tensor [num_envs] (positive excess magnitude; pair with a negative weight, same
        convention as the other two terms in this file).
    """
    lin_vel_world_xy = env.simulator.robot_root_states[:, 7:9]
    speed = torch.norm(lin_vel_world_xy, dim=-1)
    # Upper-clamped at 1000 -- see penalty_excess_foot_contact_force's comment for why (same
    # unbounded-square-can-overflow-float32-into-nan failure mode, same fix). This is the term
    # directly implicated by real mjwarp training logs showing raw_rew_kick_penalty_excess_
    # base_lin_vel reaching ~2.7e24 -- back-solving torch.square implies speed itself briefly hit
    # ~1.6e12 m/s, i.e. a genuine transient numerical corruption, not real robot motion.
    excess = torch.clamp(speed - speed_threshold_mps, min=0.0, max=1000.0)
    return torch.square(excess) * (floor + k * current_w_g(env)) * kick_safety_reward_scale(env)
