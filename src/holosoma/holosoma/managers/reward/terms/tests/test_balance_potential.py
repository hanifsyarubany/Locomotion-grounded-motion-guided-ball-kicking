"""Unit tests for ``CapturePointPotentialShaping`` (managers/reward/terms/balance_potential.py).

Four things are load-bearing and each is covered below:

1. **The potential itself** -- Phi = capture-point margin x height margin, in [0,1]: 1.0 for a
   still, upright, double-support robot; 0.0 once the capture point leaves the support polygon or
   the base reaches the fall height.
2. **Single-support tightening** -- the SAME capture-point offset that is comfortably inside a
   double-support polygon must fall outside a single-foot one. This is what gives the term its
   swing-phase sensitivity without any hand-coded phase gating (the thing that backfired when
   ``swing_threshold_multiplier`` tried it).
3. **Potential-based shaping semantics** -- ``F = gamma*Phi(s') - Phi(s)``, zero on the first step
   of an episode (nothing to difference against), zero on the step after a mid-episode motion
   restart (the robot was teleported; the policy neither earned nor caused that jump), and the
   telescoping identity over a rollout.
4. **Terminal vs truncation** -- Phi(s') is forced to 0 on a TRUE failure termination (the
   absorbing-state convention the invariance theorem requires) but explicitly NOT on a time-out,
   which is a truncation the critic bootstraps through. Getting this backwards would penalize
   surviving to the episode cap.

Isolated with lightweight fakes rather than a live env (constructing a real one needs IsaacSim),
providing exactly the fields the term reads: ``simulator.robot_root_states``,
``simulator._rigid_body_pos``, ``simulator.contact_forces``, ``feet_indices``, ``reset_buf``,
``time_out_buf``, and ``command_manager.get_state("motion_command").time_steps``.
"""

from __future__ import annotations

import math

import pytest
import torch

from holosoma.managers.reward.terms.balance_potential import GRAVITY, CapturePointPotentialShaping

FOOT_R = 0.09
NOMINAL_Z = 0.78
FALL_Z = 0.40
GAMMA = 0.97


class _FakeCfg:
    def __init__(self, **params):
        self.params = params
        self.weight = 50.0


class _FakeSim:
    def __init__(self, n, device):
        # [x,y,z, qx,qy,qz,qw, vx,vy,vz, wx,wy,wz]
        self.robot_root_states = torch.zeros(n, 13, device=device)
        self.robot_root_states[:, 2] = NOMINAL_Z
        # two feet, symmetric about the origin, 0.20m apart in y
        self._rigid_body_pos = torch.zeros(n, 2, 3, device=device)
        self._rigid_body_pos[:, 0, 1] = 0.10
        self._rigid_body_pos[:, 1, 1] = -0.10
        self._rigid_body_vel = torch.zeros(n, 2, 3, device=device)
        self.contact_forces = torch.zeros(n, 2, 3, device=device)
        self.contact_forces[:, :, 2] = 100.0  # both feet loaded
        self.body_names = ["pelvis", "left_ankle_roll_link"]


class _FakeMotionCommand:
    def __init__(self, n, device):
        self.time_steps = torch.zeros(n, dtype=torch.long, device=device)


class _FakeCommandManager:
    def __init__(self, mc):
        self._mc = mc

    def get_state(self, name):
        return self._mc if name == "motion_command" else None


class _FakeRewardManagerCfg:
    only_positive_rewards = False


class _FakeRewardManager:
    cfg = _FakeRewardManagerCfg()


class _FakeEnv:
    def __init__(self, n=4, device="cpu", with_motion=True):
        self.num_envs = n
        self.device = device
        self.simulator = _FakeSim(n, device)
        self.feet_indices = torch.tensor([0, 1], dtype=torch.long, device=device)
        self.reset_buf = torch.zeros(n, dtype=torch.long, device=device)
        self.time_out_buf = torch.zeros(n, dtype=torch.bool, device=device)
        self.reward_manager = _FakeRewardManager()
        self.motion_command = _FakeMotionCommand(n, device) if with_motion else None
        self.command_manager = _FakeCommandManager(self.motion_command)


def _make(env=None, **overrides):
    env = env or _FakeEnv()
    params = dict(
        gamma=GAMMA, foot_radius=FOOT_R, contact_force_threshold=1.0,
        fall_height=FALL_Z, nominal_height=NOMINAL_Z,
    )
    params.update(overrides)
    return CapturePointPotentialShaping(_FakeCfg(**params), env), env


# ---------------------------------------------------------------- 1. the potential
def test_phi_is_one_for_still_upright_double_support():
    term, env = _make()
    assert torch.allclose(term.compute_phi(env), torch.ones(env.num_envs))


def test_phi_is_zero_at_the_fall_height():
    term, env = _make()
    env.simulator.robot_root_states[:, 2] = FALL_Z
    assert torch.allclose(term.compute_phi(env), torch.zeros(env.num_envs))


def test_height_margin_is_linear_between_fall_and_nominal():
    term, env = _make()
    env.simulator.robot_root_states[:, 2] = 0.5 * (FALL_Z + NOMINAL_Z)
    # still, so the balance margin is 1.0 and Phi is the height margin alone
    assert torch.allclose(term.compute_phi(env), torch.full((env.num_envs,), 0.5), atol=1e-6)


def test_phi_decays_but_never_saturates_when_capture_point_leaves_support():
    """Deliberately asymptotic rather than clamped: a clamped-linear margin measured EXACTLY 0 in
    46.9% of live samples (53.3% during swing), recreating the dead-gradient pathology this term
    exists to remove. exp(-d/r) keeps a live gradient arbitrarily far outside the polygon."""
    term, env = _make()
    env.simulator.robot_root_states[:, 7] = 5.0
    phi_far = term.compute_phi(env)
    assert torch.all(phi_far > 0.0), "must never saturate to exactly zero on the balance axis"
    assert torch.all(phi_far < 0.05), "but should be strongly decayed once well outside support"

    env.simulator.robot_root_states[:, 7] = 8.0
    assert torch.all(term.compute_phi(env) < phi_far), "must keep decreasing => gradient stays alive"


def test_phi_is_bounded_in_unit_interval_under_random_states():
    term, env = _make(env=_FakeEnv(n=64))
    g = torch.Generator().manual_seed(0)
    env.simulator.robot_root_states[:, 2] = torch.rand(64, generator=g) * 1.2
    env.simulator.robot_root_states[:, 7:9] = (torch.rand(64, 2, generator=g) - 0.5) * 8.0
    env.simulator.contact_forces[:, :, 2] = (torch.rand(64, 2, generator=g) * 200.0)
    phi = term.compute_phi(env)
    assert torch.all(phi >= 0.0) and torch.all(phi <= 1.0)
    assert torch.all(torch.isfinite(phi))


# ---------------------------------------------------------------- 2. single-support tightening
def test_single_support_tightens_the_margin_versus_double_support():
    """The property that gives this term swing-phase sensitivity for free.

    Uses a capture point BETWEEN the feet, which is the physically meaningful case: a robot whose
    CoM is centred between its feet is solidly stable on two feet and actively falling the instant
    one leaves the ground. (Note the support centre correctly follows the stance foot, so a CP
    already over that foot is NOT penalized by lifting the other one -- see the companion test.)
    """
    term, env = _make()
    phi_double = term.compute_phi(env).clone()  # still, CP at the midpoint -> fully stable
    env.simulator.contact_forces[:, 1, 2] = 0.0  # lift the right foot -> single support on left

    phi_single = term.compute_phi(env)
    assert torch.allclose(phi_double, torch.ones(env.num_envs)), "centred CP + two feet = margin 1"
    assert torch.all(phi_single < phi_double), "single support must tighten the margin"
    # CP is 0.10m from the stance foot against a single-foot radius of 0.09 -> ~exp(-1.11) of the
    # height margin: heavily penalized, but still differentiable rather than clamped flat.
    assert torch.all(phi_single < 0.4) and torch.all(phi_single > 0.0)


def test_capture_point_over_the_stance_foot_survives_lifting_the_other():
    """Complement to the above: single support is not penalized per se, only being off-balance
    relative to whatever is actually carrying load. A kick that shifts weight onto the stance foot
    BEFORE lifting should not be punished."""
    term, env = _make()
    # put the CP directly over the left foot (y = +0.10), then lift the right
    env.simulator.robot_root_states[:, 8] = 0.10 / math.sqrt(NOMINAL_Z / GRAVITY)
    env.simulator.contact_forces[:, 1, 2] = 0.0
    assert torch.allclose(term.compute_phi(env), torch.ones(env.num_envs), atol=1e-6)


def test_flight_phase_is_worse_than_single_support():
    term, env = _make()
    env.simulator.robot_root_states[:, 8] = 0.5 / math.sqrt(NOMINAL_Z / GRAVITY)
    env.simulator.contact_forces[:, 1, 2] = 0.0
    phi_single = term.compute_phi(env).clone()
    env.simulator.contact_forces[:, 0, 2] = 0.0  # both feet off the ground
    assert torch.all(term.compute_phi(env) <= phi_single)


# ---------------------------------------------------------------- 3. shaping semantics
def test_first_step_after_reset_emits_zero_shaping():
    term, env = _make()
    assert torch.allclose(term(env), torch.zeros(env.num_envs))


def test_steady_state_shaping_equals_gamma_minus_one_times_phi():
    term, env = _make()
    term(env)  # seed
    out = term(env)
    assert torch.allclose(out, torch.full((env.num_envs,), GAMMA - 1.0), atol=1e-6)


def test_shaping_matches_the_potential_difference_formula():
    term, env = _make()
    phi0 = term.compute_phi(env).clone()
    term(env)
    env.simulator.robot_root_states[:, 2] = 0.6  # degrade balance
    phi1 = term.compute_phi(env).clone()
    assert torch.allclose(term(env), GAMMA * phi1 - phi0, atol=1e-6)


def test_motion_restart_suppresses_shaping_for_exactly_one_step():
    term, env = _make()
    env.motion_command.time_steps[:] = 100
    term(env)
    term(env)
    env.motion_command.time_steps[:] = 0  # clip restarted: teleport, not policy-caused
    assert torch.allclose(term(env), torch.zeros(env.num_envs)), "restart step must be suppressed"
    env.motion_command.time_steps[:] = 1
    assert not torch.allclose(term(env), torch.zeros(env.num_envs)), "must resume the step after"


def test_reset_clears_state_for_selected_envs_only():
    term, env = _make()
    term(env)
    term(env)
    term.reset(torch.tensor([0, 2]))
    out = term(env)
    assert torch.allclose(out[[0, 2]], torch.zeros(2)), "reset envs are suppressed"
    assert torch.all(out[[1, 3]] != 0.0), "untouched envs keep shaping"


def test_telescoping_identity_over_a_rollout():
    """Sum of gamma*Phi(s') - Phi(s) reduces to the discounted endpoint minus the start."""
    term, env = _make()
    env.motion_command.time_steps[:] = 0
    heights = [0.78, 0.74, 0.70, 0.66, 0.62]
    env.simulator.robot_root_states[:, 2] = heights[0]
    term(env)  # seed; emits 0
    phi_start = term.compute_phi(env).clone()

    total = torch.zeros(env.num_envs)
    for k, h in enumerate(heights[1:], start=1):
        env.motion_command.time_steps[:] = k
        env.simulator.robot_root_states[:, 2] = h
        total = total + term(env)

    phis = []
    for h in heights:
        env.simulator.robot_root_states[:, 2] = h
        phis.append(term.compute_phi(env).clone())
    expected = sum(GAMMA * phis[i + 1] - phis[i] for i in range(len(phis) - 1))
    assert torch.allclose(total, expected, atol=1e-6)
    assert torch.allclose(phis[0], phi_start)


# ---------------------------------------------------------------- 4. terminal vs truncation
def test_true_failure_termination_zeroes_phi_next():
    term, env = _make()
    term(env)
    phi_prev = term.compute_phi(env).clone()
    env.reset_buf[:] = 1
    env.time_out_buf[:] = False  # genuine failure
    # Phi(s') := 0  =>  F = -Phi(s)
    assert torch.allclose(term(env), -phi_prev, atol=1e-6)


def test_timeout_does_not_zero_phi_next():
    """A time-out is a truncation, not an absorbing state -- zeroing here would penalize
    surviving to the episode cap, the exact opposite of the intent."""
    term, env = _make()
    term(env)
    env.reset_buf[:] = 1
    env.time_out_buf[:] = True  # reset_buf includes timeouts in production
    out = term(env)
    assert torch.allclose(out, torch.full((env.num_envs,), GAMMA - 1.0), atol=1e-6)
    assert torch.all(out > -0.5), "timeout must not produce the full terminal penalty"


# ---------------------------------------------------------------- misc / guards
def test_rejects_only_positive_rewards():
    term, env = _make()
    env.reward_manager.cfg.only_positive_rewards = True
    with pytest.raises(ValueError, match="only_positive_rewards"):
        term(env)
    env.reward_manager.cfg.only_positive_rewards = False


def test_rejects_nominal_height_below_fall_height():
    with pytest.raises(AssertionError):
        _make(nominal_height=0.3, fall_height=0.4)


def test_works_without_a_motion_command():
    """Standalone/locomotion-only envs have no motion_command; the term must still run."""
    term, env = _make(env=_FakeEnv(with_motion=False))
    term(env)
    assert torch.allclose(term(env), torch.full((env.num_envs,), GAMMA - 1.0), atol=1e-6)


def test_gravity_constant_is_sane():
    assert 9.0 < GRAVITY < 10.0
