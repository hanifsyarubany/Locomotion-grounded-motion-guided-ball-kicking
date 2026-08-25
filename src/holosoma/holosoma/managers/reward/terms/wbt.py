"""Reward terms for Whole Body Tracking tasks."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, List

import torch

from holosoma.config_types.reward import RewardTermCfg
from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.managers.reward.base import RewardTermBase
from holosoma.utils.kick_reward_scales import motion_tracking_reward_scale
from holosoma.utils.rotations import quat_error_magnitude

if TYPE_CHECKING:
    from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager


def _get_motion_command_and_assert_type(env: WholeBodyTrackingManager) -> MotionCommand:
    motion_command = env.command_manager.get_state("motion_command")
    assert motion_command is not None, "motion_command not found in command manager"
    assert isinstance(motion_command, MotionCommand), f"Expected MotionCommand, got {type(motion_command)}"
    return motion_command


#########################################################################################################
## terms same to managers/reward/terms/locomotion.py
#########################################################################################################


def penalty_action_rate(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Penalize changes in actions between steps.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    actions = env.action_manager.action
    prev_actions = env.action_manager.prev_action
    return torch.sum(torch.square(prev_actions - actions), dim=1)


class KickActionSmoothness(RewardTermBase):
    """Penalize the second derivative of actions -- 2026-08-05, ported from RoboNaldo
    (arXiv:2606.11092)'s ``action_smoothness`` (mdp/rewards.py), "discourages rapid changes in
    action velocity, which produces smoother motion than a first-order action-rate penalty alone"
    (their own docstring). The remaining §3b regularization term this port had initially missed
    (found while replacing the port-scope table's "—" markers, not assumed complete without
    checking).

    Formula matches exactly: ``action_diff = action - prev_action``,
    ``action_diff2 = action_diff - (prev_action - prev_prev_action)``,
    ``reward = clamp(sum(action_diff2^2), 0, 10)``.

    ADAPTATION: RoboNaldo stores ``prev_prev_action`` on their own ``MotionCommand`` (a third
    action-history slot alongside the ``ActionManager``'s own ``action``/``prev_action``, which
    this project's ``ActionManager`` also already provides -- see ``penalty_action_rate`` above,
    same two properties). Rather than adding a third buffer to ``MotionCommand`` or
    ``ActionManager`` (shared infrastructure every other action/observation consumer would then
    also see), this term keeps its OWN ``_prev_prev_action`` buffer, self-contained -- the
    ``RewardTermBase`` state pattern this project already uses for exactly this kind of
    "one term needs history nothing else does" case (e.g. ``CapturePointPotentialShaping``'s own
    ``_phi_prev``). Lazily allocated on first ``__call__`` (not ``__init__``) -- same defensive
    reason ``CapturePointPotentialShaping`` defers its own ``env.reward_manager`` check: no
    assumption is made about whether ``ActionManager`` is guaranteed fully set up at the point
    reward terms are constructed.

    UNGATED across the whole kick episode -- RoboNaldo's own registration carries no phase
    multiplier (weight -0.0015 S1 / -0.1 S2a / -0.07 S2b)."""

    def __init__(self, cfg: RewardTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.env = env
        self._prev_prev_action: torch.Tensor | None = None

    def __call__(self, env: WholeBodyTrackingManager, **kwargs) -> torch.Tensor:
        action = env.action_manager.action  # type: ignore[attr-defined]
        prev_action = env.action_manager.prev_action  # type: ignore[attr-defined]
        if self._prev_prev_action is None or self._prev_prev_action.shape != prev_action.shape:
            self._prev_prev_action = prev_action.clone()
        action_diff = action - prev_action
        action_diff2 = action_diff - (prev_action - self._prev_prev_action)
        reward = torch.clamp(torch.sum(torch.square(action_diff2), dim=-1), min=0.0, max=10.0)
        self._prev_prev_action = prev_action.clone()
        return reward

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if self._prev_prev_action is None:
            return
        idx = env_ids if env_ids is not None else slice(None)
        self._prev_prev_action[idx] = 0.0


def limits_dof_pos(env: WholeBodyTrackingManager, soft_dof_pos_limit: float = 0.95) -> torch.Tensor:
    """Penalize joint positions too close to limits.

    Args:
        env: The environment instance
        soft_dof_pos_limit: Soft limit as fraction of hard limit

    Returns:
        Reward tensor [num_envs]
    """
    # Use soft limits as fraction of hard limits
    m = (env.simulator.hard_dof_pos_limits[:, 0] + env.simulator.hard_dof_pos_limits[:, 1]) / 2  # type: ignore[attr-defined]
    r = env.simulator.hard_dof_pos_limits[:, 1] - env.simulator.hard_dof_pos_limits[:, 0]  # type: ignore[attr-defined]
    lower_soft_limit = m - 0.5 * r * soft_dof_pos_limit
    upper_soft_limit = m + 0.5 * r * soft_dof_pos_limit

    out_of_limits = -(env.simulator.dof_pos - lower_soft_limit).clip(max=0.0)  # lower limit
    out_of_limits += (env.simulator.dof_pos - upper_soft_limit).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


#########################################################################################################
## terms specific to Whole Body Tracking
#########################################################################################################

# ================================================================================================
# Robot Tracking Rewards
# ================================================================================================


def motion_global_ref_position_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.ref_pos_w - motion_command.robot_ref_pos_w), dim=-1)
    return torch.exp(-error / sigma**2)


def motion_global_ref_orientation_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = quat_error_magnitude(motion_command.ref_quat_w, motion_command.robot_ref_quat_w) ** 2
    return torch.exp(-error / sigma**2)


def motion_relative_body_position_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.body_pos_relative_w - motion_command.robot_body_pos_w), dim=-1)
    return torch.exp(-error.mean(-1) / sigma**2)


def motion_relative_body_orientation_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = quat_error_magnitude(motion_command.body_quat_relative_w, motion_command.robot_body_quat_w) ** 2
    return torch.exp(-error.mean(-1) / sigma**2)


def motion_global_body_lin_vel(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.body_lin_vel_w - motion_command.robot_body_lin_vel_w), dim=-1)
    return torch.exp(-error.mean(-1) / sigma**2)


def motion_global_body_ang_vel(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.body_ang_vel_w - motion_command.robot_body_ang_vel_w), dim=-1)
    return torch.exp(-error.mean(-1) / sigma**2)


def motion_global_feet_lin_vel(
    env: WholeBodyTrackingManager, sigma: float, body_names: List[str] | None = None
) -> torch.Tensor:
    """The 7th motion-tracking term -- 2026-08-05, ported from RoboNaldo (arXiv:2606.11092)'s
    ``motion_global_feet_linear_velocity_error_exp`` (mdp/rewards.py). Verified against the real
    source: that function's body is LITERALLY IDENTICAL to ``motion_global_body_linear_velocity_
    error_exp`` (the same function ``motion_global_body_lin_vel`` above already ports) -- same
    formula, called with a different (feet-only, narrower) ``body_names`` and its own separate
    weight/sigma. Not a new formula; a second registration of the existing one, restricted to a
    subset of tracked bodies. RoboNaldo's own ``jump_flag``-gated warmup zeroing on this term is
    Stage-3-only (out of scope, see ROBONALDO_PORT_SCOPE.md) and not ported, same as every other
    ``jump_flag`` branch skipped elsewhere in this port.

    Index resolution: ``body_lin_vel_w``/``robot_body_lin_vel_w`` are indexed by
    ``motion_command.motion_cfg.body_names_to_track`` (a configured SUBSET of tracked bodies,
    confirmed to include both ankle_roll links -- see ``config_values/wbt/g1/command.py``'s
    ``motion_config``), NOT ``env.simulator.body_names`` (the full sim body list every OTHER
    body-indexed term in this file resolves against) -- a different index space entirely. Resolved
    fresh every call via a plain list ``.index()``, matching ``_torso_orientation_error``'s own
    established "uncached, resolved fresh every call" convention for a lookup this small (2 of 14
    entries) and this infrequent (once per reward computation, not per-env)."""
    motion_command = _get_motion_command_and_assert_type(env)
    if body_names is None:
        body_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
    tracked_names = motion_command.motion_cfg.body_names_to_track
    body_indexes = [tracked_names.index(name) for name in body_names]
    error = torch.sum(
        torch.square(
            motion_command.body_lin_vel_w[:, body_indexes] - motion_command.robot_body_lin_vel_w[:, body_indexes]
        ),
        dim=-1,
    )
    return torch.exp(-error.mean(-1) / sigma**2)


class MotionStrikeDofPosErrorExp(RewardTermBase):
    """Per-joint DOF-position tracking reward, active ONLY during the strike phase
    (``motion_command.in_strike_phase``), covering the 17 arm+waist joints (excludes all 12 leg
    DOF, both legs -- the kick leg is already the best-tracked group by the same live-error-vs-
    clip-ROM measure described below, and the support leg needs freedom for single-support
    balance).

    Exists because the 6 Cartesian ``motion_*_error_exp`` functions above are structurally blind
    to rotation about a limb's own long axis (``shoulder_yaw``, ``wrist_roll``, ...): those axes
    barely move the tracked link's Cartesian position, so a policy can drift arbitrarily far on
    them at almost no Cartesian-tracking cost. Measured live (checkpoint 440k,
    ``20260802_122622-unified-stageC-2skills-locomotion``): strike-phase arm/waist joint error
    16-53 deg (worst: ``left_shoulder_yaw`` 52.9, ``right_shoulder_yaw`` 44.9,
    ``right``/``left_wrist_roll`` 39.2/37.4 -- 7 of the 8 worst individual joints are arm joints),
    EXCEEDING the reference clip's own per-joint range of motion on exactly those axes (clip ROM
    11-22 deg for ``shoulder_yaw``/``wrist_roll``) -- the signature of null-space drift, not the
    policy tracking a different-but-comparable version of the clip's own motion. The kick leg
    itself is NOT included in the mask: its live error (~19.5 deg) is already SMALLER than the
    clip's own strike-phase range of motion for that leg (35-36 deg), i.e. it's already doing a
    genuinely large, well-tracked authored motion and doesn't show this failure mode.

    Formula is per-joint exp, THEN mean: ``mean_j(exp(-error_j^2/sigma^2))`` -- deliberately NOT
    ``exp(-mean_j(error_j^2)/sigma^2)``, the pattern the 6 Cartesian siblings above use (e.g.
    ``motion_relative_body_position_error_exp``'s ``error.mean(-1)`` INSIDE the exp). Averaging
    error before the exp lets one badly-diverged joint hide inside an average dominated by
    well-tracked ones -- by Jensen's inequality (exp is convex), mean-after-exp >=
    exp-of-mean-error always, strictly so whenever cross-joint error has any variance, which is
    exactly the "one joint blew up, the rest are fine" case this term exists to price correctly.
    See ``test_motion_strike_dof_pos_error_exp.py``'s regression guard, which computes both
    formulas explicitly on the same inputs and asserts the inequality.

    Ships at weight=0.0 in ``config_values/unified/g1/reward.py`` -- a REASONED, UNVALIDATED
    starting sigma/weight (see that registration's own comment for the numbers), not yet checked
    against a real training run. Turn on via ``configs/kicking_motion_reward_tuning.yaml``
    (uncomment the ``motion_strike_dof_pos_error_exp`` weight/``_sigma`` lines under
    ``motion_tracking_reward``) as a deliberate, separate decision -- this project has repeatedly
    found that code-level-correct, well-reasoned reward changes can still move real training in an
    unexpected direction (see the ``kick_recovery_termination_handoff`` episode this same
    session), so real validation is left to an actual run, not asserted here.
    """

    def __init__(self, cfg: RewardTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.env = env
        self.sigma = cfg.params["sigma"]
        self.dof_indexes = self._get_index_of_a_in_b(
            cfg.params["dof_names"],
            self.env.simulator.dof_names,  # type: ignore[attr-defined]
            self.env.device,
        )

    def __call__(self, env: WholeBodyTrackingManager, **kwargs) -> torch.Tensor:
        motion_command = _get_motion_command_and_assert_type(env)
        ref = motion_command.joint_pos[:, self.dof_indexes]
        actual = motion_command.robot_joint_pos[:, self.dof_indexes]
        per_joint_reward = torch.exp(-torch.square(ref - actual) / self.sigma**2)
        tracking_reward = per_joint_reward.mean(dim=-1)
        return tracking_reward * motion_command.in_strike_phase.float() * motion_tracking_reward_scale(env)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        pass

    #########################################################################################################
    ## Internal Helper functions
    #########################################################################################################
    def _get_index_of_a_in_b(self, a_names: List[str], b_names: List[str], device: str = "cpu") -> torch.Tensor:
        indexes = []
        for name in a_names:
            assert name in b_names, f"The specified name ({name}) doesn't exist: {b_names}"
            indexes.append(b_names.index(name))
        return torch.tensor(indexes, dtype=torch.long, device=device)


class StrikeDofDivergencePenalty(RewardTermBase):
    """LINEAR, NON-SATURATING per-joint price on strike-phase joint-tracking error beyond
    ``threshold`` -- ``mean_j(relu(|q_j - q_ref_j| - threshold))``, gated to ``in_strike_phase``.
    Pair with a NEGATIVE weight.

    WHY THIS EXISTS, and why the existing Gaussian sibling cannot do this job (measured
    2026-08-24, per-joint strike-phase probes on skill011 ckpt 262k / skill012 ckpt 291k, 512
    envs, plus a lag/ROM probe on the same checkpoints):

      * The strike-phase divergence is NOT a control lag and NOT an actuation shortfall. Shifting
        the reference back 1-6 frames improves per-joint error by 0-8% (so: not lag), and the
        robot's achieved range of motion is >= the clip's almost everywhere (so: not shortfall).
        It is OVER-motion: measured robot-vs-clip ROM ratios of 4.9x (skill011 left_shoulder_yaw,
        clip 19.5 deg -> robot 96.5 deg), 4.1x (skill012 left_elbow, 25.1 -> 102.6), 3.9x
        (right_elbow), 2.8x (left_knee), 1.7x (right_knee). Joints whose ERROR exceeds the clip's
        entire ROM are not tracking a comparable motion -- this is the null-space drift
        ``MotionStrikeDofPosErrorExp``'s own docstring first measured on 2026-08-02.

      * ``MotionStrikeDofPosErrorExp`` already COVERS those joints (all 17 arm/waist DOF) at
        weight 1.0 and has not corrected them, because a Gaussian gives up on exactly the joints
        that have drifted furthest: at sigma=0.5 the measured gradient (2e/sigma^2)exp(-e^2/
        sigma^2) is 0.21 on skill011's left_shoulder_yaw (54 deg) and 0.06 on skill012's
        left_elbow (64 deg), versus ~1.7 on a typical ~20 deg joint -- 8x to 28x LESS pull where
        the problem is worst. Raising that term's weight does not fix the shape: it scales every
        joint equally, leaving the worst/typical gradient ratio at 0.45-0.54.

    This term's gradient is CONSTANT (d/de of relu(e - threshold) is 1) for every joint past the
    threshold, however far past it is -- which is the whole point. It prices divergence the
    Gaussian's tail cannot reach, without touching the Gaussian's well-calibrated behavior near
    zero error (below ``threshold`` this term is exactly 0 and contributes no gradient at all, so
    the two compose rather than compete).

    ``mean`` over joints (not max, not sum): unlike the exp-then-mean subtlety its Gaussian
    sibling documents -- where averaging BEFORE the exp lets one blown-up joint hide inside an
    average of well-tracked ones -- a linear penalty has no such failure mode. Each joint's excess
    enters the mean proportionally and is never damped by its neighbours, so one joint at 0.6 rad
    of excess contributes exactly 0.6/N whatever the others do.

    ``threshold`` (radians, default 0.35 = 20 deg): the deadband below which normal tracking error
    is not priced at all. Chosen from the same probes: measured typical per-joint strike error is
    ~20 deg while the pathological joints sit at 54-64 deg, so 0.35 rad leaves ordinary tracking
    untouched and bites only the outliers. Deliberately a DEADBAND rather than a price on all
    error -- the Gaussian sibling already shapes the near-zero regime correctly, and double-
    pricing it would just re-weight what already works.

    ``dof_names`` (default None = all DOF): restrict to a joint subset, same convention as
    ``MotionStrikeDofPosErrorExp``. NOTE the coverage gap that motivated exposing this: the two
    existing strike-phase terms together cover 23 of 29 DOF, omitting hip_pitch/knee/ankle_pitch
    on BOTH legs -- exactly the sagittal kick chain. Passing None here (the default) is the only
    strike-phase term that prices those six joints at all.

    Ships at weight 0.0 in ``config_values/unified/g1/reward.py`` -- a true no-op until a config
    opts in, same discipline as every other new mechanism in this project. NOT VALIDATED BY A
    TRAINING RUN: the diagnosis above is measured, the remedy is not. Known risk worth stating --
    arm swing may be load-bearing for balance (a human swings arms to counter the kick leg's
    angular momentum, and retargeted clips often under-represent that), so suppressing it could
    cost survival or ball speed rather than helping. A/B it against the same config at 0.0.
    """

    def __init__(self, cfg: RewardTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.env = env
        self.threshold = float(cfg.params.get("threshold", 0.35))
        assert self.threshold >= 0.0, f"threshold must be >= 0.0, got {self.threshold}"
        dof_names = cfg.params.get("dof_names")
        if dof_names is None:
            self.dof_indexes = None
        else:
            self.dof_indexes = self._get_index_of_a_in_b(
                dof_names,
                self.env.simulator.dof_names,  # type: ignore[attr-defined]
                self.env.device,
            )

    def __call__(self, env: WholeBodyTrackingManager, **kwargs) -> torch.Tensor:
        motion_command = _get_motion_command_and_assert_type(env)
        ref = motion_command.joint_pos
        actual = motion_command.robot_joint_pos
        if self.dof_indexes is not None:
            ref = ref[:, self.dof_indexes]
            actual = actual[:, self.dof_indexes]
        excess = (torch.abs(ref - actual) - self.threshold).clamp(min=0.0)
        return excess.mean(dim=-1) * motion_command.in_strike_phase.float()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        pass

    def _get_index_of_a_in_b(self, a_names: List[str], b_names: List[str], device: str = "cpu") -> torch.Tensor:
        indexes = []
        for name in a_names:
            assert name in b_names, f"The specified name ({name}) doesn't exist: {b_names}"
            indexes.append(b_names.index(name))
        return torch.tensor(indexes, dtype=torch.long, device=device)


# ================================================================================================
# Object Tracking Rewards
# ================================================================================================


def object_global_ref_position_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.object_pos_w - motion_command.simulator_object_pos_w), dim=-1)
    return torch.exp(-error / sigma**2)


def object_global_ref_orientation_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = quat_error_magnitude(motion_command.object_quat_w, motion_command.simulator_object_quat_w) ** 2
    return torch.exp(-error / sigma**2)


# ================================================================================================
# Undesired Contacts Rewards
# ================================================================================================


class UndesiredContacts(RewardTermBase):
    def __init__(self, cfg: RewardTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.env = env
        undesired_contacts_body_names = [
            body_name
            for body_name in self.env.simulator.body_names  # type: ignore[attr-defined]
            if re.match(cfg.params.get("undesired_contacts_body_names", ""), body_name)
        ]
        self.undesired_contacts_body_indexes = self._get_index_of_a_in_b(
            undesired_contacts_body_names,
            self.env.simulator.body_names,  # type: ignore[attr-defined]
            self.env.device,
        )
        self.threshold = cfg.params.get("threshold", 1.0)

    def __call__(self, env: WholeBodyTrackingManager, **kwargs) -> torch.Tensor:
        # (num_envs, history_length, num_bodies, 3)
        net_contact_forces = self.env.simulator.contact_forces_history
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self.undesired_contacts_body_indexes], dim=-1), dim=1)[0]
            > self.threshold
        )
        return torch.sum(is_contact, dim=1)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        pass

    #########################################################################################################
    ## Internal Helper functions
    #########################################################################################################
    def _get_index_of_a_in_b(self, a_names: List[str], b_names: List[str], device: str = "cpu") -> torch.Tensor:
        indexes = []
        for name in a_names:
            assert name in b_names, f"The specified name ({name}) doesn't exist: {b_names}"
            indexes.append(b_names.index(name))
        return torch.tensor(indexes, dtype=torch.long, device=device)


class KickFeetSlip(RewardTermBase):
    """Penalize tangential (horizontal) foot velocity while that foot is carrying ground contact
    force -- 2026-08-04, ported directly from RoboNaldo (arXiv:2606.11092)'s ``feet_slip``
    (mdp/rewards.py), same formula and same default threshold (contact force magnitude > 1.0).

    Motivation: a kick is single-support -- the stance (non-kicking) foot sliding under load is a
    direct mechanical precursor to a fall, and this project has no existing analog.
    ``kick_penalty_excess_base_lin_vel`` catches the BASE moving too fast (the momentum-throw
    exploit this project already diagnosed and fixed -- see that term's own docstring); it says
    nothing about whether the planted FOOT is sliding out from under an otherwise-slow base. This
    is a mechanistically different signal, not a duplicate.

    Index resolution mirrors ``UndesiredContacts`` immediately above (same
    ``_get_index_of_a_in_b`` helper, same ``env.simulator.body_names`` ordering, resolved once at
    construction). ``contact_forces_history`` is ``(num_envs, history_length, num_bodies, 3)``,
    the same tensor ``UndesiredContacts`` reads; ``_rigid_body_vel`` is the world-frame per-body
    linear velocity sibling of ``_rigid_body_rot`` (already used by ``_ShotTracker``/
    ``MotionCommand`` elsewhere in this project under the SAME ``body_names``-derived index
    convention), so no new simulator plumbing is needed for either.

    Weight 0.0 by default -- explicit opt-in, unvalidated starting point, same discipline as
    every other addition this session. A reasoned (not yet trained) starting weight: -0.5."""

    def __init__(self, cfg: RewardTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.env = env
        foot_body_names = cfg.params.get("foot_body_names", ["left_ankle_roll_link", "right_ankle_roll_link"])
        self.foot_indexes = self._get_index_of_a_in_b(
            foot_body_names,
            self.env.simulator.body_names,  # type: ignore[attr-defined]
            self.env.device,
        )
        self.threshold = cfg.params.get("threshold", 1.0)

    def __call__(self, env: WholeBodyTrackingManager, **kwargs) -> torch.Tensor:
        # (num_envs, history_length, num_feet, 3) -> per-foot max-over-history contact force
        # magnitude, same "was this foot in contact at any point this control step" check
        # UndesiredContacts uses. Reads the `env` PARAMETER, not `self.env` -- harmless in
        # production (env is a persistent singleton, both point at the same object) but the
        # correct convention (matches MotionStrikeDofPosErrorExp's own __call__ elsewhere in this
        # file) and the one that actually matters for probe/eval scripts that construct env fresh.
        net_contact_forces = env.simulator.contact_forces_history[:, :, self.foot_indexes]
        in_contact = torch.max(torch.norm(net_contact_forces, dim=-1), dim=1)[0] > self.threshold  # (E, num_feet)

        foot_vel_xy = env.simulator._rigid_body_vel[:, self.foot_indexes, :2]  # (E, num_feet, 2)
        slip_speed_sq = torch.sum(torch.square(foot_vel_xy), dim=-1)  # (E, num_feet)

        return torch.sum(torch.where(in_contact, slip_speed_sq, torch.zeros_like(slip_speed_sq)), dim=-1)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        pass

    def _get_index_of_a_in_b(self, a_names: List[str], b_names: List[str], device: str = "cpu") -> torch.Tensor:
        indexes = []
        for name in a_names:
            assert name in b_names, f"The specified name ({name}) doesn't exist: {b_names}"
            indexes.append(b_names.index(name))
        return torch.tensor(indexes, dtype=torch.long, device=device)


class ArmDefaultPose(RewardTermBase):
    """Reward (positive magnitude) for arm joints staying near their default standing pose --
    2026-08-05, ported from RoboNaldo (arXiv:2606.11092)'s ``arm_default_pose_penalty``
    (mdp/rewards.py). Register with a NEGATIVE weight to reproduce RoboNaldo's own penalty
    (their function returns ``-error_sq`` directly; this project's convention is the opposite --
    every term returns a non-negative MAGNITUDE and the sign lives in ``RewardTermCfg.weight``,
    matching every other ``penalty_*`` term in this project).

    Motivation (ROBONALDO_PORT_SCOPE.md Sec 3b, "highest-ROI single term"): the strike-phase
    arm/waist joint divergence measured for ``MotionStrikeDofPosErrorExp`` above (16-53 deg, worse
    than the kick leg itself) is null-space drift -- the 6 Cartesian tracking terms are structurally
    blind to rotation about a limb's own long axis. RoboNaldo's answer is NOT a joint-space
    clip-tracking reward (unlike ``MotionStrikeDofPosErrorExp``): they simply stop the arms
    wandering FROM THE DEFAULT POSE and accept clip infidelity there, paired with tight per-joint
    action clipping on the same joints (ROBONALDO_PORT_SCOPE.md Sec 4, Phase 4 of this port).

    Formula, verified against the real RoboNaldo source (not the paper, not this project's own
    earlier unverified paraphrase of it): ``mean_j(error_j^2 * weight_j)`` over the 14 arm-only
    DOF (``upper_left_arm_dof_names + upper_right_arm_dof_names`` -- NOT ``upper_dof_names``,
    which also includes the 3 waist joints RoboNaldo's own arm-only regex excludes), where
    ``weight_j = 5.0`` for the 4 elbow joints and ``1.0`` elsewhere. This 5x elbow weighting is a
    literal port of what RoboNaldo's code actually does (``reward[:, elbow_joint_ids] *= 5.0``
    applied to the already-computed per-joint penalty) -- NOT what its docstring claims ("elbow
    joints use a task-specific 1.0 rad target instead of the URDF default"): the real
    implementation reads ``default_joint_pos`` unmodified for every joint including elbows, only
    the penalty MAGNITUDE gets the 5x multiplier. Ported the code, not the stale docstring.

    Unlike ``MotionStrikeDofPosErrorExp``, this term does NOT depend on ``motion_command`` at all
    -- it is a pure static-pose regularizer, unrelated to which clip/skill is active, so it needs
    no phase gating and applies across the whole kick episode (approach + strike + recovery/hold),
    same as RoboNaldo's own registration (present, at a nonzero weight, in every one of their
    stages -- S1/S2a/S2b all carry ``arm_default_pose`` weight_scale 0.2, only S3 differs at the
    same 0.2)."""

    def __init__(self, cfg: RewardTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.env = env
        arm_dof_names = cfg.params["arm_dof_names"]
        self.arm_dof_indexes = self._get_index_of_a_in_b(
            arm_dof_names,
            self.env.simulator.dof_names,  # type: ignore[attr-defined]
            self.env.device,
        )
        elbow_weight_multiplier = cfg.params.get("elbow_weight_multiplier", 5.0)
        is_elbow = [name.endswith("_elbow_joint") for name in arm_dof_names]
        assert any(is_elbow), f"arm_dof_names has no elbow joint to weight: {arm_dof_names}"
        elbow_mask = torch.tensor(is_elbow, dtype=torch.float32, device=self.env.device)
        self.per_joint_weight = 1.0 + elbow_mask * (elbow_weight_multiplier - 1.0)

    def __call__(self, env: WholeBodyTrackingManager, **kwargs) -> torch.Tensor:
        cur = env.simulator.dof_pos[:, self.arm_dof_indexes]  # type: ignore[attr-defined]
        default = env.default_dof_pos[:, self.arm_dof_indexes]  # type: ignore[attr-defined]
        weighted_error_sq = torch.square(cur - default) * self.per_joint_weight
        return weighted_error_sq.mean(dim=-1)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        pass

    def _get_index_of_a_in_b(self, a_names: List[str], b_names: List[str], device: str = "cpu") -> torch.Tensor:
        indexes = []
        for name in a_names:
            assert name in b_names, f"The specified name ({name}) doesn't exist: {b_names}"
            indexes.append(b_names.index(name))
        return torch.tensor(indexes, dtype=torch.long, device=device)


class KickFeetAirTime(RewardTermBase):
    """Reward feet that stayed airborne long enough before touchdown -- 2026-08-05, ported from
    RoboNaldo (arXiv:2606.11092)'s ``feet_air_time`` (mdp/rewards.py), adapted to this project's
    own contact-reading and phase-gating conventions (see ADAPTATIONS below).

    Mechanism, ported faithfully: a per-foot ``air_time`` buffer accumulates ``env.dt`` each step
    a foot is airborne; on the step contact is (re-)detected -- filtered as ``contact OR
    last_contact`` to reduce single-frame contact-sensor noise, matching RoboNaldo's own filter --
    the term pays ``air_time - threshold`` for that foot (so briefly grazing the ground doesn't
    pay, only a genuine, sufficiently long swing does), then the buffer resets. This is the exact
    mechanism verified against RoboNaldo's real source (not this project's own earlier, unverified
    worked-example pseudocode, which omitted the contact-filtering and touchdown-only-payout
    structure entirely).

    ADAPTATIONS from RoboNaldo's version, each because the RoboNaldo mechanism it replaces has no
    equivalent in this project (both deliberately, per ROBONALDO_PORT_SCOPE.md Sec 6's "out of
    scope" list):
    - Contact detection: ``env.simulator.contact_forces_history`` max-over-history-window NORM,
      same convention ``KickFeetSlip``/``UndesiredContacts`` already use -- NOT IsaacLab's
      ``ContactSensor.data.net_forces_w`` (no equivalent in this project's simulator abstraction).
      NOTE, verified against RoboNaldo's real source (not assumed): their OWN ``feet_air_time``
      uses a Z-COMPONENT-ONLY contact test (``fmat[..., 2] > threshold``), NOT the norm --
      different from their own ``locomotion_phase_feet_clearance``/``feet_slip``, which both DO
      use the norm (``fmat.norm(dim=-1) > threshold``, matching ``KickFeetSlip``'s already-ported
      convention exactly). RoboNaldo is internally inconsistent between these two of their own
      terms; this port picked the norm convention uniformly across all 3 of this project's
      contact-gated kick terms rather than replicating RoboNaldo's own inconsistency.
    - Per-foot state: a plain instance buffer (``self._air_time``/``self._last_contact``), reset
      via this class's own ``reset(env_ids)`` -- the correct, structural way to handle per-episode
      state in this project's ``RewardTermBase`` contract (``RewardManager`` calls ``reset`` at
      the right time). RoboNaldo instead hand-rolls a ``command._feet_air_time_state`` dict cache
      with a manual ``time_steps <= 1`` "just reset" check inside ``__call__`` -- a workaround for
      IsaacLab's reward functions having no per-term reset hook of their own. Not needed here.
    - Phase gating: RoboNaldo gates by ``command_mag > command_threshold`` (their own
      ``adapt_motion_flag``-derived locomotion-command magnitude, Stage-3-only, explicitly out of
      scope for this port) AND ``~stable_phase`` (``time_steps > critic_frame_index +
      kick_hold_steps``). This project has no ``adapt_motion_flag``/command-magnitude equivalent
      -- gated to ``motion_command.in_kicking_phase`` alone instead (True for approach+strike,
      False for recovery/hold). CORRECTION (previously overstated as "exactly the same boundary"):
      ``in_kicking_phase``'s boundary is ``stand_start_idx``, this project's analog of
      ``critic_frame_index`` alone -- it does NOT reproduce RoboNaldo's additional
      ``+kick_hold_steps`` grace offset past that boundary. The gate therefore switches ON
      slightly EARLIER (by ``kick_hold_steps`` frames) than RoboNaldo's own ``~stable_phase``
      would. This project's own established alternative for exactly this class of problem --
      ramping a post-kick gate in smoothly rather than reproducing a specific hard offset -- is
      ``_kick_recovery_gate`` (used by ``penalty_kick_unstable`` below instead of a hard step);
      it was not applied here because ``feet_air_time``'s own payout is inherently already
      touchdown-triggered/event-based, not a per-step magnitude a hard-vs-ramped boundary would
      meaningfully change the character of. During a genuinely stationary hold,
      ``in_kicking_phase`` is already False either way, so "don't reward air-time while standing
      still" is satisfied by the phase gate regardless of the exact offset.

    Weight 50 in every RoboNaldo stage (S1/S2a/S2b) -- flat, no cross-stage staging needed, unlike
    most other ported terms."""

    def __init__(self, cfg: RewardTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.env = env
        foot_body_names = cfg.params.get("foot_body_names", ["left_ankle_roll_link", "right_ankle_roll_link"])
        self.foot_indexes = self._get_index_of_a_in_b(
            foot_body_names,
            self.env.simulator.body_names,  # type: ignore[attr-defined]
            self.env.device,
        )
        self.contact_force_threshold = cfg.params.get("contact_force_threshold", 1.0)
        self.air_time_threshold = cfg.params.get("air_time_threshold", 0.25)
        num_feet = len(self.foot_indexes)
        self._air_time = torch.zeros(self.env.num_envs, num_feet, device=self.env.device)
        self._last_contact = torch.zeros(self.env.num_envs, num_feet, dtype=torch.bool, device=self.env.device)

    def __call__(self, env: WholeBodyTrackingManager, **kwargs) -> torch.Tensor:
        net_contact_forces = env.simulator.contact_forces_history[:, :, self.foot_indexes]  # type: ignore[attr-defined]
        contact = torch.max(torch.norm(net_contact_forces, dim=-1), dim=1)[0] > self.contact_force_threshold

        contact_filt = contact | self._last_contact
        first_contact = (self._air_time > 0.0) & contact_filt

        self._air_time = self._air_time + env.dt  # type: ignore[attr-defined]
        reward = torch.sum(torch.where(first_contact, self._air_time - self.air_time_threshold, torch.zeros_like(self._air_time)), dim=-1)

        motion_command = _get_motion_command_and_assert_type(env)
        reward = reward * motion_command.in_kicking_phase.float()

        self._air_time = torch.where(contact_filt, torch.zeros_like(self._air_time), self._air_time)
        self._last_contact = contact
        return reward

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        idx = env_ids if env_ids is not None else slice(None)
        self._air_time[idx] = 0.0
        self._last_contact[idx] = False

    def _get_index_of_a_in_b(self, a_names: List[str], b_names: List[str], device: str = "cpu") -> torch.Tensor:
        indexes = []
        for name in a_names:
            assert name in b_names, f"The specified name ({name}) doesn't exist: {b_names}"
            indexes.append(b_names.index(name))
        return torch.tensor(indexes, dtype=torch.long, device=device)


class KickFeetContactTime(RewardTermBase):
    """Reward feet that leave contact soon after landing -- 2026-08-05, ported from RoboNaldo
    (arXiv:2606.11092)'s ``feet_contact_time`` (mdp/rewards.py), the inverse-tempo sibling of
    ``KickFeetAirTime`` above: that term rewards a sufficiently LONG swing before touchdown; this
    term rewards a sufficiently SHORT stance before liftoff. Together they price both halves of a
    stepping gait rather than just one.

    Mechanism, ADAPTED (RoboNaldo's own version depends on IsaacLab's ``ContactSensor.
    compute_first_air``/``data.last_contact_time`` -- convenience accessors this project's
    simulator abstraction has no equivalent for; the underlying INTENT -- "pay when a contact
    ends, IF that contact was short" -- is reproduced faithfully with a plain per-foot instance
    buffer, same structural pattern as ``KickFeetAirTime``'s own ``_air_time``/``_last_contact``,
    just measuring contact duration instead of air duration and paying on LIFTOFF instead of
    touchdown): a per-foot ``_contact_time`` buffer accumulates ``env.dt`` each step a foot is in
    contact; on the step contact is lost (``first_air``), the term pays ``1.0`` for that foot IFF
    the just-ended contact duration was below ``threshold`` (a flat bonus, not a magnitude --
    matches RoboNaldo's own ``(last_contact_time < threshold) * first_air`` exactly: the boolean
    condition contributes its own 1.0, not a scaled duration), then the buffer resets.

    Contact detection: ``env.simulator.contact_forces_history`` norm, same convention as every
    other contact-gated term in this file (including this term's own sibling
    ``KickFeetAirTime`` -- see that class's docstring for the note that RoboNaldo itself uses the
    norm for ``locomotion_phase_feet_clearance``/``feet_slip`` but Z-only for ``feet_air_time``;
    this project picked the norm uniformly across all of them).

    UNGATED across the whole kick episode -- RoboNaldo's own registration carries no phase
    multiplier of its own (weight ``-0.5`` flat in every stage), so none is added here either.

    Weight -0.5 in every RoboNaldo stage (S1/S2a/S2b) -- flat, no cross-stage staging needed."""

    def __init__(self, cfg: RewardTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.env = env
        foot_body_names = cfg.params.get("foot_body_names", ["left_ankle_roll_link", "right_ankle_roll_link"])
        self.foot_indexes = self._get_index_of_a_in_b(
            foot_body_names,
            self.env.simulator.body_names,  # type: ignore[attr-defined]
            self.env.device,
        )
        self.contact_force_threshold = cfg.params.get("contact_force_threshold", 1.0)
        self.contact_time_threshold = cfg.params.get("contact_time_threshold", 0.5)
        num_feet = len(self.foot_indexes)
        self._contact_time = torch.zeros(self.env.num_envs, num_feet, device=self.env.device)
        self._last_contact = torch.zeros(self.env.num_envs, num_feet, dtype=torch.bool, device=self.env.device)

    def __call__(self, env: WholeBodyTrackingManager, **kwargs) -> torch.Tensor:
        net_contact_forces = env.simulator.contact_forces_history[:, :, self.foot_indexes]  # type: ignore[attr-defined]
        contact = torch.max(torch.norm(net_contact_forces, dim=-1), dim=1)[0] > self.contact_force_threshold

        first_air = self._last_contact & (~contact)
        pays = first_air & (self._contact_time < self.contact_time_threshold)
        reward = torch.sum(pays.float(), dim=-1)

        self._contact_time = torch.where(contact, self._contact_time + env.dt, torch.zeros_like(self._contact_time))  # type: ignore[attr-defined]
        self._last_contact = contact
        return reward

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        idx = env_ids if env_ids is not None else slice(None)
        self._contact_time[idx] = 0.0
        self._last_contact[idx] = False

    def _get_index_of_a_in_b(self, a_names: List[str], b_names: List[str], device: str = "cpu") -> torch.Tensor:
        indexes = []
        for name in a_names:
            assert name in b_names, f"The specified name ({name}) doesn't exist: {b_names}"
            indexes.append(b_names.index(name))
        return torch.tensor(indexes, dtype=torch.long, device=device)


class KickSwingFeetClearance(RewardTermBase):
    """Penalize swing (not-in-contact) feet that don't clear a target height above terrain --
    2026-08-05, ported from RoboNaldo (arXiv:2606.11092)'s ``locomotion_phase_feet_clearance``
    (mdp/rewards.py), adapted to this project's own terrain-relative foot-height and contact
    conventions.

    Formula, ported faithfully: for each foot NOT in contact (contact-force gated, same convention
    as ``KickFeetAirTime``/``KickFeetSlip`` above), ``clamp(target_height - foot_height, min=0)``
    squared, summed over feet, clamped to ``max_penalty`` -- a foot that clears the target pays
    zero; a foot that doesn't pays a bounded quadratic penalty, growing worse for the whole
    remainder of the swing rather than only at the single lowest instant.

    ADAPTATIONS: RoboNaldo computes foot height as world-Z minus ``env.scene.env_origins`` (their
    per-env grid-cell origin) -- this project instead uses
    ``terrain_manager.get_state("locomotion_terrain").feet_heights``, the SAME
    terrain-relative per-foot height buffer ``feet_phase`` (managers/reward/terms/locomotion.py)
    already relies on for locomotion mode's own swing-height shaping. This is a strictly more
    correct adaptation for this project specifically: kick-mode envs run on RANDOMIZED terrain
    tiles (gated to the FLAT subset via ``UnifiedManager._build_task_mode_partition``'s
    ``env_terrain_is_flat`` check, but "flat" tiles can still sit at different absolute heights),
    so a flat world-Z-minus-origin computation would misjudge clearance on a tile whose surface
    isn't exactly at the origin's Z. ``feet_heights`` is indexed ``[:, 0]``=left, ``[:, 1]``=right,
    the same fixed 2-column convention ``feet_phase`` uses -- this term always covers both feet,
    not just the active skill's kick foot, matching RoboNaldo's own ``body_names`` (both ankles).

    Contact detection: this project's own ``env.simulator.contact_forces_history`` convention
    (same as every other contact-gated term in this file), not IsaacLab's ``ContactSensor``.

    Phase gating: RoboNaldo gates to ``time_steps < critic_frame_index`` ("before the kick frame,
    so it does not fight the kicking motion" -- their own docstring). This project's closest
    equivalent is the pure walking-approach window, EXCLUDING the strike itself:
    ``motion_command.in_kicking_phase & ~motion_command.in_strike_phase`` -- during the strike,
    the kick foot's trajectory is authored, deliberate, and not meant to satisfy a generic
    ground-clearance target; RoboNaldo's own single ``critic_frame_index`` boundary doesn't
    distinguish approach from strike the way this project's 3-phase clip partition does, so gating
    to the narrower approach-only window is the more faithful match to their stated intent, not a
    looser one.

    Weight -20 (S1/S2a/S2b, no jump_flag) -- RoboNaldo's own weight formula is
    ``-80*jump_flag - 20``; ``jump_flag`` is a Stage-3-only mechanism, out of scope, so only the
    flat -20 base applies here."""

    def __init__(self, cfg: RewardTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.env = env
        foot_body_names = cfg.params.get("foot_body_names", ["left_ankle_roll_link", "right_ankle_roll_link"])
        self.foot_indexes = self._get_index_of_a_in_b(
            foot_body_names,
            self.env.simulator.body_names,  # type: ignore[attr-defined]
            self.env.device,
        )
        self.contact_force_threshold = cfg.params.get("contact_force_threshold", 1.0)
        self.target_height = cfg.params.get("target_height", 0.12)
        self.max_penalty = cfg.params.get("max_penalty", 0.5)

    def __call__(self, env: WholeBodyTrackingManager, **kwargs) -> torch.Tensor:
        motion_command = _get_motion_command_and_assert_type(env)
        active = motion_command.in_kicking_phase & (~motion_command.in_strike_phase)

        net_contact_forces = env.simulator.contact_forces_history[:, :, self.foot_indexes]  # type: ignore[attr-defined]
        in_contact = torch.max(torch.norm(net_contact_forces, dim=-1), dim=1)[0] > self.contact_force_threshold
        swing_mask = ~in_contact

        feet_heights = env.terrain_manager.get_state("locomotion_terrain").feet_heights  # type: ignore[attr-defined]
        clearance_error = torch.clamp(self.target_height - feet_heights, min=0.0)
        penalty = torch.sum(torch.square(clearance_error) * swing_mask.float(), dim=-1)
        penalty = torch.clamp(penalty, max=self.max_penalty)
        return penalty * active.float()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        pass

    def _get_index_of_a_in_b(self, a_names: List[str], b_names: List[str], device: str = "cpu") -> torch.Tensor:
        indexes = []
        for name in a_names:
            assert name in b_names, f"The specified name ({name}) doesn't exist: {b_names}"
            indexes.append(b_names.index(name))
        return torch.tensor(indexes, dtype=torch.long, device=device)


def kick_no_fly(env: WholeBodyTrackingManager, height_threshold: float = 0.05) -> torch.Tensor:
    """Penalize states where BOTH feet are simultaneously airborne -- 2026-08-05, ported from
    RoboNaldo (arXiv:2606.11092)'s ``no_fly`` (mdp/rewards.py). A plain function, not a
    ``RewardTermBase`` subclass: no per-episode state, and body indexes are resolved fresh each
    call the same way ``_torso_orientation_error`` (managers/reward/terms/locomotion.py) already
    does -- this project's own established convention for a term whose index resolution is cheap
    enough not to need construction-time caching.

    Formula, ported faithfully: a foot is "flying" when its height above terrain exceeds
    ``height_threshold``; the penalty fires (returns 1.0) only when EVERY tracked foot is flying
    at once. Height source: ``terrain_manager``'s ``feet_heights`` (same terrain-relative buffer
    ``KickSwingFeetClearance``/``feet_phase`` use), not RoboNaldo's flat world-Z -- same terrain-
    randomization rationale as ``KickSwingFeetClearance``'s own ADAPTATIONS note.

    UNGATED across the whole kick episode (approach + strike + recovery/hold) -- RoboNaldo
    registers this with no phase multiplier of its own (`weight=-1*reg_weight`, applied flat in
    every stage), so no gating is added here either. Weight -1 (S1) / -0.5 (S2a/S2b, reg_weight
    0.2 vs 0.05) in RoboNaldo's own numbers."""
    feet_heights = env.terrain_manager.get_state("locomotion_terrain").feet_heights  # type: ignore[attr-defined]
    is_flying = feet_heights > height_threshold
    both_flying = torch.sum(is_flying.float(), dim=-1) == is_flying.shape[-1]
    return both_flying.float()
