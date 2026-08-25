"""Potential-based reward shaping on a capture-point balance margin.

WHY THIS EXISTS (2026-07-30, measured, not speculative)
-------------------------------------------------------
Live telemetry on this project's own Stage-C runs found the kick-mode reward has **no usable
gradient in the pre-fall regime** -- precisely where the falls happen:

* the six ``motion_*`` tracking terms are ``exp(-error/std)``; they realize only ~19% of their
  configured weight sum (1.21 realized vs 6.5 configured). Where tracking error is large -- i.e.
  exactly when the robot is diverging toward a fall -- ``exp(-e/std) ~= 0`` AND so is its
  derivative. Saturated: no signal.
* ``kick_alive`` is ~73-80% of the positive reward and is a FLAT PER-STEP CONSTANT. Its gradient
  w.r.t. actions is zero by construction; it only influences behavior through the probability of
  termination.
* termination itself is sparse, delayed and binary.

A phase-resolved probe (256 envs, checkpoint 339000) localized 62-85% of falls to the authored
swing, clustered at the late-clip / early-recovery boundary. Every previous attempt to fix this by
retuning existing scalars made things worse, in both directions:
``bad_tracking_swing_only``/``swing_threshold_multiplier`` (loosening termination) tripled the
per-cycle fall hazard, and halving ``kick_alive`` raised it ~37-76% because the tracking terms were
in no state to take over the load.

This term adds the missing gradient WITHOUT removing or reweighting anything that currently works.

WHY POTENTIAL-BASED, SPECIFICALLY
---------------------------------
Ng, Harada & Russell (1999) proved that shaping of the exact form ``F(s, s') = gamma*Phi(s') -
Phi(s)`` leaves the optimal policy UNCHANGED -- and that potential-based shaping is the only class
with that property. That matters here more than usual: every other knob tried on this task traded
stability against kick quality in one direction or the other. This one provably cannot change what
the optimal policy is; it only changes how densely the agent is told about progress toward it. So
the coefficient can be set aggressively without the failure mode that has bitten this project
twice.

Two implementation details are load-bearing for that guarantee, and both are easy to get wrong:

1. **gamma must match the agent's own discount** (``algo.config.gamma``, 0.97 for fast_sac here).
   A mismatched gamma is still *a* shaping function but is no longer the invariant one.
2. **Only TRUE terminals get Phi := 0**, never time-outs. A time-out is a truncation: the value
   function bootstraps through it, so zeroing Phi there would inject a spurious penalty for
   surviving to the episode cap -- the exact opposite of the intent. This term reads
   ``reset_buf & ~time_out_buf`` to separate them, which is well-defined because
   ``BaseTask._post_physics_step`` runs ``_check_termination()`` BEFORE ``_compute_reward()``.

Also relied upon: ``RewardManagerCfg.only_positive_rewards`` is False for this project. If it were
ever turned on, the negative half of the shaping would be clipped away and the invariance would be
destroyed -- ``__init__`` asserts this rather than silently degrading.

WHY CAPTURE POINT RATHER THAN BASE HEIGHT
-----------------------------------------
``kick_low_height`` (z < 0.40) is a *lagging* detector -- by the time it fires the fall is long
committed and no ankle torque recovers it. The capture point (a.k.a. extrapolated centre of mass,
divergent component of motion) is the standard bipedal *leading* indicator:

    xi = com_xy + com_vel_xy * sqrt(z / g)

Once ``xi`` leaves the support polygon the fall is committed. Feeding the *margin* to that boundary
as a potential gives the policy a dense, physically meaningful, differentiable measure of how close
to committed it is.

A useful consequence, and the reason this needs no hand-written phase gating: during single support
(most of a kick swing) the support polygon collapses to one foot, so the margin tightens
AUTOMATICALLY exactly where the falls were measured to originate. Phase gating is what
``swing_threshold_multiplier`` tried and it backfired; this gets the phase sensitivity for free
from the contact state instead.

COM APPROXIMATION -- stated plainly rather than hidden
------------------------------------------------------
No mass-weighted centre-of-mass is exposed by the simulator abstraction, so the floating-base root
(pelvis) is used as the COM proxy, via the same ``robot_root_states`` /``_rigid_body_pos`` fields
every other reward term in this package already reads. For the G1 the pelvis sits close to the true
COM, and the capture point is dominated by the velocity term regardless. ``com_body_name`` allows
overriding it with a specific body if a better proxy is wanted later.
"""

from __future__ import annotations

from typing import Any

from holosoma.config_types.reward import RewardTermCfg
from holosoma.managers.reward.base import RewardTermBase
from holosoma.utils.safe_torch_import import torch

GRAVITY = 9.81


class CapturePointPotentialShaping(RewardTermBase):
    """``F(s, s') = gamma * Phi(s') - Phi(s)`` over a capture-point balance margin.

    ``Phi(s) = balance_margin(s) * height_margin(s)``, both normalized to ``[0, 1]``, so
    ``Phi in [0, 1]``: 1.0 is solidly balanced and well above the fall height, 0.0 is
    fall-committed (capture point outside the support polygon) or already collapsed. The product
    (rather than a sum or a min) means either failure route drives the potential to zero, and
    keeps the gradient smooth in both.

    The raw value returned here is multiplied by ``weight * dt`` by ``RewardManager.compute``.
    A constant factor on a potential is still a potential (``Phi' = weight*dt*Phi``), so the
    invariance property survives that scaling untouched.
    """

    def __init__(self, cfg: RewardTermCfg, env: Any):
        super().__init__(cfg, env)
        p = cfg.params

        # MUST equal the agent's own discount for the invariance theorem to apply -- see module
        # docstring. Exposed rather than hardcoded so it can't silently drift from algo.config.gamma.
        self.gamma = float(p.get("gamma", 0.97))
        # Effective support half-width contributed by a single foot, metres. The G1 foot is ~0.22m
        # long x ~0.09m wide; a single conservative radius is used rather than a true oriented
        # polygon because the margin only needs to be monotone in "how close to the edge", not
        # geometrically exact, for the shaping to point the right way.
        self.foot_radius = float(p.get("foot_radius", 0.09))
        self.contact_force_threshold = float(p.get("contact_force_threshold", 1.0))
        # Deliberately the SAME 0.40 that kick_low_height terminates on, so the height half of the
        # potential reaches exactly 0 at the point the fall termination fires -- the shaping and the
        # termination agree on what "fallen" means instead of disagreeing by an arbitrary offset.
        self.fall_height = float(p.get("fall_height", 0.40))
        self.nominal_height = float(p.get("nominal_height", 0.78))
        self.com_body_name = p.get("com_body_name", None)

        assert self.nominal_height > self.fall_height, (
            f"nominal_height ({self.nominal_height}) must exceed fall_height ({self.fall_height})"
        )
        # Checked lazily on the first __call__, NOT here: reward terms are constructed inside
        # RewardManager.__init__, which runs BEFORE `env.reward_manager` is assigned
        # (base_task.py:146) -- reading it here raises AttributeError.
        self._checked_positive_only = False

        device = env.device
        n = env.num_envs
        self._phi_prev = torch.zeros(n, dtype=torch.float, device=device)
        # False for the first step of an episode and for the first step after a mid-episode motion
        # restart -- in both cases _phi_prev refers to a state the policy did not act its way out
        # of, so differencing against it would credit/blame the policy for a teleport.
        self._valid = torch.zeros(n, dtype=torch.bool, device=device)
        self._prev_time_steps = torch.zeros(n, dtype=torch.long, device=device)

        self.com_body_index: int | None = None
        if self.com_body_name is not None:
            body_names = env.simulator.body_names  # type: ignore[attr-defined]
            assert self.com_body_name in body_names, (
                f"com_body_name '{self.com_body_name}' not found in {body_names}"
            )
            self.com_body_index = body_names.index(self.com_body_name)

    # ------------------------------------------------------------------
    def compute_phi(self, env: Any) -> torch.Tensor:
        """The potential itself, in ``[0, 1]``. Public so tests and probes can read it directly."""
        root = env.simulator.robot_root_states
        if self.com_body_index is None:
            com_xy = root[:, 0:2]
            com_z = root[:, 2]
            com_vel_xy = root[:, 7:9]
        else:
            com_xy = env.simulator._rigid_body_pos[:, self.com_body_index, 0:2]
            com_z = env.simulator._rigid_body_pos[:, self.com_body_index, 2]
            com_vel_xy = env.simulator._rigid_body_vel[:, self.com_body_index, 0:2]

        # Capture point / extrapolated CoM. clamp_min keeps sqrt finite if the robot is already
        # flat on the ground (z -> 0); the height margin is 0 there anyway so Phi is 0 regardless.
        omega = torch.sqrt(torch.clamp(com_z, min=1e-3) / GRAVITY)
        xi = com_xy + com_vel_xy * omega.unsqueeze(-1)

        feet = env.feet_indices
        foot_xy = env.simulator._rigid_body_pos[:, feet, 0:2]  # [E, F, 2]
        in_contact = env.simulator.contact_forces[:, feet, 2] > self.contact_force_threshold  # [E, F]
        n_contact = in_contact.sum(dim=-1)  # [E]

        # Support centre: mean of the feet that are actually bearing load. With nothing in contact
        # (flight) there is no support polygon at all, so fall back to the mean foot position and
        # let the shrunken radius below register it as the near-worst case it is.
        w = in_contact.to(foot_xy.dtype)
        w_sum = w.sum(dim=-1, keepdim=True)
        w = torch.where(w_sum > 0, w / w_sum.clamp(min=1e-6), torch.full_like(w, 1.0 / w.shape[-1]))
        center = (foot_xy * w.unsqueeze(-1)).sum(dim=1)  # [E, 2]

        # Support radius: one foot alone contributes foot_radius; two feet in contact additionally
        # span the distance between them. This is what makes the margin tighten by itself during
        # the single-support phase of a kick, with no phase gating anywhere.
        half_span = 0.5 * torch.norm(foot_xy[:, 0] - foot_xy[:, 1], dim=-1) if foot_xy.shape[1] >= 2 else torch.zeros_like(com_z)
        r = torch.where(n_contact >= 2, self.foot_radius + half_span, torch.full_like(half_span, self.foot_radius))
        r = torch.where(n_contact == 0, torch.full_like(r, 0.5 * self.foot_radius), r)

        # Smooth exponential falloff, NOT a clamped linear ``1 - d/r``. This is a correction from
        # live measurement (2026-07-30, 128 envs x 500 settled steps under a real trained policy):
        # the clamped-linear form read EXACTLY 0 in 46.9% of samples overall and 53.3% during the
        # swing, because a walking/kicking robot legitimately carries its capture point outside the
        # instantaneous support polygon much of the time -- that is what stepping IS. Clamping there
        # reproduced precisely the dead-gradient pathology this whole term exists to fix, and in the
        # worst possible place (the swing, where 62-85% of falls originate).
        #
        # exp(-d/r) instead: 1.0 when the capture point sits over the support centre, 0.368 at the
        # boundary, 0.135 at 2r, asymptotically but never exactly 0 -- so the gradient is alive at
        # every distance, including the far-outside states that precede a fall. Phi as a whole still
        # reaches exactly 0 on a real fall via the height factor below, which is what keeps it
        # consistent with the Phi(terminal)=0 convention and with kick_low_height's own 0.40 line.
        d = torch.norm(xi - center, dim=-1)
        balance = torch.exp(-d / r.clamp(min=1e-3))
        height = torch.clamp(
            (com_z - self.fall_height) / (self.nominal_height - self.fall_height), 0.0, 1.0
        )
        return balance * height

    # ------------------------------------------------------------------
    def __call__(self, env: Any, **kwargs) -> torch.Tensor:
        if not self._checked_positive_only:
            # See module docstring: positive-only clipping would discard the negative half of the
            # shaping and destroy the very invariance property this term is chosen for. Fail
            # loudly rather than silently degrading into an ordinary (non-invariant) bonus.
            mgr = getattr(env, "reward_manager", None)
            if mgr is not None and getattr(mgr.cfg, "only_positive_rewards", False):
                raise ValueError(
                    "CapturePointPotentialShaping requires only_positive_rewards=False; "
                    "positive-only clipping discards the negative half of the potential-based "
                    "shaping and destroys its policy-invariance guarantee."
                )
            self._checked_positive_only = True

        phi_now = self.compute_phi(env)

        # A mid-episode motion restart (wbt.py: MotionCommand advances past motion_end and calls
        # reset() WITHOUT ending the episode) teleports the robot. That restart runs in
        # _update_tasks_callback, which for IsaacSim executes AFTER _compute_reward -- so the jump
        # is first observable here on the FOLLOWING step, which is exactly when this detects it.
        restarted = torch.zeros_like(self._valid)
        mc = env.command_manager.get_state("motion_command") if hasattr(env, "command_manager") else None
        time_steps = getattr(mc, "time_steps", None) if mc is not None else None
        if time_steps is not None:
            restarted = time_steps < self._prev_time_steps
            self._prev_time_steps = time_steps.clone()

        # TRUE terminal -> Phi := 0 (the absorbing-state convention the theorem requires).
        # Time-outs are truncations, NOT terminals: the critic bootstraps through them, so their
        # Phi must be left alone. reset_buf already includes time_out_buf, hence the explicit
        # subtraction. Valid here only because _check_termination() runs before _compute_reward().
        failed = env.reset_buf.bool() & ~env.time_out_buf.bool()
        phi_next = torch.where(failed, torch.zeros_like(phi_now), phi_now)

        shaping = self.gamma * phi_next - self._phi_prev
        shaping = torch.where(self._valid & ~restarted, shaping, torch.zeros_like(shaping))

        # Store the UNZEROED potential: a failed env is about to be reset (which clears _valid
        # anyway), while a restarted env's phi_now is already measured on its post-teleport state
        # and is the correct baseline for its next step.
        self._phi_prev = phi_now
        self._valid = torch.ones_like(self._valid)
        return shaping

    # ------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._phi_prev.zero_()
            self._valid.zero_()
            self._prev_time_steps.zero_()
        else:
            self._phi_prev[env_ids] = 0.0
            self._valid[env_ids] = False
            self._prev_time_steps[env_ids] = 0
