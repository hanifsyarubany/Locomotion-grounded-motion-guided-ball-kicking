"""Unit tests for shooting.py's three DISJOINT phase-gate multipliers and their SPLIT application
across the 9 shooting reward terms:

- _locomotion_approach_multiplier (1.0 only while time_steps < strike_start_idx): gates the 1
  approach term (ball_approach_stance, added 2026-08-01 -- the only term that pays out in the
  phase the other terms are all excluded from; see its own docstring).
- _strike_phase_multiplier (1.0 only in_strike_phase, i.e. strike_start_idx..stand_start_idx):
  gates the 4 contact-mechanics terms (ball_proximity, contact_orientation, ball_velocity,
  foot_strike_pitch -- the last added 2026-08-07).
- _post_locomotion_multiplier (1.0 once time_steps >= strike_start_idx, i.e. in_strike_phase AND
  everything after -- post-kick-standing, recovery, hold): gates the 4 outcome terms
  (error_ball_to_target, predicted_error_ball_to_target, goal_success_burst, and, since
  2026-08-01, ball_contact_hit -- see that term's own docstring; it reuses this same
  already-established gate rather than repeating the history below).

The first and third are exact complements by construction, so between them every tick of a cycle
is covered by exactly one of the two -- test_approach_and_post_locomotion_gates_are_complements
below pins that invariant.

2026-08-04: within the 4 _strike_phase_multiplier terms, ball_proximity and contact_orientation
ALSO gate on ~has_kicked (zero once contact is confirmed), while ball_velocity gates on has_kicked
the OTHER way (zero until contact). One _FakeTracker.has_kicked value cannot make all 3 nonzero at
once, so this file's tests split into _PRE_CONTACT_STRIKE_GATED_TERMS (has_kicked must be False)
and _POST_CONTACT_STRIKE_GATED_TERMS (has_kicked must be True) -- see ball_proximity/
contact_orientation's own docstrings and the module docstring's 2026-08-04 paragraph in
shooting.py for the full rationale.

2026-08-21: ball_velocity gained three OPT-IN retunes (v_ref / use_latched_peak_speed /
use_post_locomotion_gate), all defaulting OFF -- so the gating described above is what an unedited
config gets, and every test here exercises those defaults. The opt-ins have their own dedicated
tests at the bottom of this file, including the post-locomotion gate that moves ball_velocity out
of the strike group when enabled.

2026-08-07: foot_strike_pitch joins the strike-phase-gated group but deliberately has NO
has_kicked gate at all (user-specified, weighing the tradeoff against ball_proximity/
contact_orientation's dribbling-exploit rationale -- see foot_strike_pitch's own docstring) --
nonzero for the WHOLE strike window regardless of contact state, tested separately via
_HAS_KICKED_INDEPENDENT_STRIKE_GATED_TERMS rather than joining either has_kicked-split list.

History (all 2026-07-31, same day, three iterations once live data was available at each step):
1. All 6 (of that day's 6) gated to in_strike_phase -- but this made goal_success_burst exactly 0
   in every live sample (a shot's roll-to-target time routinely exceeds the ~1s strike window).
2. The 3 outcome terms ungated entirely -- fixed goal_success_burst, but a live diagnostic then
   found each term's own internal temporal gate (has_kicked / moving / success_latched) isn't a
   precise enough proxy for "the kick attempt has begun": the robot's body/approaching leg can
   incidentally brush the ball during the locomotion-approach walk-up, tripping has_kicked well
   before strike_start_idx at a ball-to-target distance unchanged from spawn (a graze, not a
   strike) -- paying a small amount of undeserved credit.
3. The 3 outcome terms gated to _post_locomotion_multiplier instead -- excludes exactly
   locomotion-approach (where the graze problem lives) without reintroducing iteration 1's
   too-narrow-window problem (post-kick-standing/recovery/hold stay fully rewarded).

Isolated the same way test_kick_scale_wrappers.py isolates its wrappers: patch the term's own
dependencies (_tracker, current_w_g) to known values via unittest.mock, so these tests exercise
only the gating, not _ShotTracker's own (unrelated, real-simulator-dependent) bookkeeping.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import holosoma.managers.reward.terms.shooting as s

_STRIKE_START_IDX = 10
_STAND_START_IDX = 20


def _fake_env(time_steps: torch.Tensor, strike_start_idx: int = _STRIKE_START_IDX, stand_start_idx: int = _STAND_START_IDX):
    """Builds a fake MotionCommand exposing exactly what both multipliers read:
    in_strike_phase (for _strike_phase_multiplier) and time_steps/strike_start_idx/motion_ids
    (for _post_locomotion_multiplier) -- all derived consistently from the same time_steps."""
    motion_ids = torch.zeros_like(time_steps)
    strike_start = torch.tensor([strike_start_idx])
    stand_start = torch.tensor([stand_start_idx])
    in_strike_phase = (time_steps >= strike_start_idx) & (time_steps < stand_start_idx)
    motion_command = SimpleNamespace(
        in_strike_phase=in_strike_phase,
        time_steps=time_steps,
        strike_start_idx=strike_start,
        stand_start_idx=stand_start,
        motion_ids=motion_ids,
        # All-False (never OOD) -- this file isolates the two PHASE gates only; the OOD gate
        # (_ood_gate_multiplier, see test_shooting_ood_gate.py) is exercised separately. Every one
        # of the 6 term functions now also reads is_ood_spawn, so it must exist here or every call
        # below raises AttributeError.
        is_ood_spawn=torch.zeros_like(time_steps, dtype=torch.bool),
    )
    command_manager = SimpleNamespace(get_state=lambda name: motion_command if name == "motion_command" else None)
    return SimpleNamespace(command_manager=command_manager)


def test_strike_phase_multiplier_is_in_strike_phase_as_float():
    # locomotion (5), strike (10, 15), post-kick (20) -- only the strike-window ticks are 1.0
    env = _fake_env(torch.tensor([5, 10, 15, 20]))
    out = s._strike_phase_multiplier(env)
    assert torch.equal(out, torch.tensor([0.0, 1.0, 1.0, 0.0]))


def test_post_locomotion_multiplier_is_true_from_strike_start_onward():
    # locomotion (5, 9), strike (10, 15), post-kick (20, 100) -- everything from strike_start_idx
    # onward is 1.0, including well past stand_start_idx.
    env = _fake_env(torch.tensor([5, 9, 10, 15, 20, 100]))
    out = s._post_locomotion_multiplier(env)
    assert torch.equal(out, torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 1.0]))


def test_locomotion_approach_multiplier_is_true_only_before_strike_start():
    # locomotion (5, 9), strike (10, 15), post-kick (20, 100) -- the exact inverse of the above.
    env = _fake_env(torch.tensor([5, 9, 10, 15, 20, 100]))
    out = s._locomotion_approach_multiplier(env)
    assert torch.equal(out, torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0]))


def test_approach_and_post_locomotion_gates_are_complements():
    """The approach gate and the post-locomotion gate must partition every tick exactly -- no tick
    both-gated (which would double-pay) and none un-gated (a phase with no shooting reward at all,
    the very gap ball_approach_stance was added to close)."""
    env = _fake_env(torch.tensor([0, 5, 9, 10, 15, 19, 20, 100]))
    approach = s._locomotion_approach_multiplier(env)
    post = s._post_locomotion_multiplier(env)
    assert torch.equal(approach + post, torch.ones(8))


class _FakeTracker:
    """Provides every attribute/method the 6 term functions read from a real _ShotTracker, with
    fixed, nonzero values chosen so an UN-gated computation would be strictly positive for every
    env -- isolating the assertion to whether a phase multiplier actually zeroes it out.

    has_kicked defaults True: correct for _POST_LOCOMOTION_GATED_TERMS (they need has_kicked=True
    to be nonzero, matching real post-contact behavior), but _PRE_CONTACT_STRIKE_GATED_TERMS need
    the opposite -- tests exercising those flip it to False explicitly (2026-08-04, see this
    file's own module docstring)."""

    def __init__(self, num_envs: int):
        self.ball_radius = 0.11
        self._foot_pos = torch.zeros(num_envs, 3)
        self.ball_pos_w = torch.zeros(num_envs, 3)
        self._foot_vel = torch.tensor([[1.0, 0.0]] * num_envs)
        self.ball_to_target_xy = torch.tensor([[1.0, 0.0]] * num_envs)
        self.ball_speed = torch.full((num_envs,), 2.0)
        # 2026-08-21: ball_velocity now reads the LATCHED PEAK speed, not the instantaneous
        # one (see shooting.py::ball_velocity's docstring). Same nonzero value so a zero
        # result in the tests below can still only come from a gate.
        self.max_ball_speed = torch.full((num_envs,), 2.0)
        self.ball_vel_xy = torch.tensor([[1.0, 0.0]] * num_envs)
        self.has_kicked = torch.ones(num_envs, dtype=torch.bool)
        self.min_target_dist = torch.full((num_envs,), 0.2)
        self.burst_active = torch.ones(num_envs, dtype=torch.bool)
        # Zero stance error => exp(0) = 1.0, i.e. strictly positive like every other fake value
        # here, so a zero result in the tests below can only come from a gate.
        self.strike_stance_ball_b = torch.zeros(num_envs, 2)
        self._ball_b = torch.zeros(num_envs, 2)
        # +90 deg about local Y (xyzw): rotates local +x (toe-forward) to world (0, 0, -1), i.e.
        # toe pointing straight DOWN -- foot_strike_pitch's pitch_signal = -(-1) = +1, strictly
        # positive like every other fake value here (see class docstring's isolation rationale).
        self._foot_quat = torch.tensor([[0.0, math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4)]] * num_envs)

    def kick_foot_contact_pos_w(self, env, contact_offset_m):
        return self._foot_pos

    def kick_foot_vel_xy(self, env):
        return self._foot_vel

    def kick_foot_quat_w(self, env):
        return self._foot_quat

    def ball_pos_heading_frame_xy(self, env):
        return self._ball_b


_STRIKE_GATED_TERMS = [s.ball_proximity, s.contact_orientation, s.ball_velocity, s.foot_strike_pitch]
# 2026-08-04: the has_kicked split (see module docstring above) means _STRIKE_GATED_TERMS can no
# longer be exercised together against one _FakeTracker.has_kicked value -- these two subsets
# require opposite has_kicked values to be nonzero.
_PRE_CONTACT_STRIKE_GATED_TERMS = [s.ball_proximity, s.contact_orientation]  # nonzero iff ~has_kicked
_POST_CONTACT_STRIKE_GATED_TERMS = [s.ball_velocity]  # nonzero iff has_kicked (DEFAULT params)
# 2026-08-07: foot_strike_pitch is gated to strike phase like the two above, but deliberately NOT
# to has_kicked either way (see its own docstring for the tradeoff vs. ball_proximity/
# contact_orientation) -- nonzero throughout the whole strike window regardless of contact state,
# so it gets its own bucket rather than joining either has_kicked-split list above.
_HAS_KICKED_INDEPENDENT_STRIKE_GATED_TERMS = [s.foot_strike_pitch]
_LOCOMOTION_APPROACH_GATED_TERMS = [s.ball_approach_stance]
_POST_LOCOMOTION_GATED_TERMS = [
    s.error_ball_to_target,
    s.predicted_error_ball_to_target,
    s.goal_success_burst,
    s.ball_contact_hit,
]


def test_pre_contact_strike_gated_terms_zero_outside_strike_phase():
    """ball_proximity/contact_orientation, BEFORE contact (has_kicked=False): zero during BOTH
    locomotion and post-kick-stabilization, nonzero only inside the strike window itself."""
    num_envs = 4
    env = _fake_env(torch.tensor([5, 10, 15, 20]))  # loco, strike, strike, post-kick
    tracker = _FakeTracker(num_envs)
    tracker.has_kicked = torch.zeros(num_envs, dtype=torch.bool)
    with (
        patch.object(s, "_tracker", return_value=tracker),
        patch.object(s, "current_w_g", return_value=torch.ones(num_envs)),
    ):
        for term in _PRE_CONTACT_STRIKE_GATED_TERMS:
            out = term(env)
            assert torch.equal(out > 0.0, torch.tensor([False, True, True, False])), (
                f"{term.__name__} not correctly gated to the strike window: {out}"
            )



def test_has_kicked_independent_strike_gated_terms_zero_outside_strike_phase_regardless_of_contact():
    """foot_strike_pitch (2026-08-07): gated to the strike window like ball_proximity/
    contact_orientation, but NOT to has_kicked either way -- must be nonzero throughout the whole
    strike window for BOTH has_kicked=False (pre-contact) and has_kicked=True (post-contact,
    still-in-window follow-through), and zero outside it regardless."""
    num_envs = 4
    env = _fake_env(torch.tensor([5, 10, 15, 20]))  # loco, strike, strike, post-kick
    expected = torch.tensor([False, True, True, False])
    for has_kicked_value in (False, True):
        tracker = _FakeTracker(num_envs)
        tracker.has_kicked = torch.full((num_envs,), has_kicked_value, dtype=torch.bool)
        with (
            patch.object(s, "_tracker", return_value=tracker),
            patch.object(s, "current_w_g", return_value=torch.ones(num_envs)),
        ):
            for term in _HAS_KICKED_INDEPENDENT_STRIKE_GATED_TERMS:
                out = term(env)
                assert torch.equal(out > 0.0, expected), (
                    f"{term.__name__} not correctly gated to the strike window at has_kicked={has_kicked_value}: {out}"
                )


def test_post_locomotion_gated_terms_zero_only_during_locomotion():
    """The 4 outcome terms: zero during locomotion, nonzero during BOTH strike and post-kick-
    stabilization -- the whole point of this gate over the old (reverted) strike-only one."""
    num_envs = 4
    env = _fake_env(torch.tensor([5, 10, 15, 100]))  # loco, strike, strike, well past post-kick
    tracker = _FakeTracker(num_envs)
    with (
        patch.object(s, "_tracker", return_value=tracker),
        patch.object(s, "current_w_g", return_value=torch.ones(num_envs)),
    ):
        for term in _POST_LOCOMOTION_GATED_TERMS:
            out = term(env)
            assert torch.equal(out > 0.0, torch.tensor([False, True, True, True])), (
                f"{term.__name__} not correctly gated to exclude only locomotion: {out}"
            )


def test_approach_gated_term_zero_outside_locomotion_approach():
    """ball_approach_stance: nonzero ONLY during locomotion-approach -- zero during both the
    strike and post-kick-stabilization, the mirror image of the post-locomotion terms above."""
    num_envs = 4
    env = _fake_env(torch.tensor([5, 10, 15, 100]))  # loco, strike, strike, post-kick
    tracker = _FakeTracker(num_envs)
    with (
        patch.object(s, "_tracker", return_value=tracker),
        patch.object(s, "current_w_g", return_value=torch.ones(num_envs)),
    ):
        for term in _LOCOMOTION_APPROACH_GATED_TERMS:
            out = term(env)
            assert torch.equal(out > 0.0, torch.tensor([True, False, False, False])), (
                f"{term.__name__} not correctly gated to locomotion-approach: {out}"
            )


def test_approach_stance_reward_decays_with_stance_error():
    """The kernel itself, isolated from the gate: exp(-||ball_b - stance_target||/sigma) must be
    maximal (1.0) at zero stance error and strictly DECREASING in error -- including for a ball
    that is too CLOSE, which is what makes this a standoff optimum rather than a monotone
    'nearer is better' pull (see the term's own docstring for why that distinction is the whole
    point of choosing this kernel)."""
    env = _fake_env(torch.tensor([0, 0, 0, 0]))  # all in locomotion-approach, gate fully open
    tracker = _FakeTracker(4)
    tracker.strike_stance_ball_b = torch.tensor([[0.3, 0.0]] * 4)
    tracker._ball_b = torch.tensor(
        [
            [0.3, 0.0],  # exactly at the authored stance -> max
            [0.6, 0.0],  # 0.3m too far
            [0.0, 0.0],  # 0.3m too CLOSE -- must score the same as 0.3m too far, not better
            [1.4, 0.0],  # 1.1m out, i.e. top-of-approach distance
        ]
    )
    with (
        patch.object(s, "_tracker", return_value=tracker),
        patch.object(s, "current_w_g", return_value=torch.ones(4)),
    ):
        out = s.ball_approach_stance(env, sigma=0.9)
    assert torch.isclose(out[0], torch.tensor(1.0))
    assert torch.isclose(out[1], out[2]), "too-close and too-far by the same margin must score equally"
    assert out[0] > out[1] > out[3], "reward must decrease monotonically in stance error"
    # Top-of-approach still carries real signal at sigma=0.9 (the reason it is not 0.35).
    assert out[3] > 0.25


def test_mixed_phase_pre_contact_strike_gated_terms_per_env_independently():
    """Two envs, one in strike phase (no contact yet) and one in locomotion -- proves
    ball_proximity/contact_orientation's phase gate is applied per-env via elementwise
    multiplication, not as a single scalar short-circuit."""
    env = _fake_env(torch.tensor([10, 5]))
    tracker = _FakeTracker(2)
    tracker.has_kicked = torch.zeros(2, dtype=torch.bool)
    with (
        patch.object(s, "_tracker", return_value=tracker),
        patch.object(s, "current_w_g", return_value=torch.ones(2)),
    ):
        for term in _PRE_CONTACT_STRIKE_GATED_TERMS:
            out = term(env)
            assert out[0].item() > 0.0, f"{term.__name__} env 0 (in strike phase) should be nonzero"
            assert out[1].item() == 0.0, f"{term.__name__} env 1 (locomotion) should be zero"



def test_pre_contact_terms_zero_once_has_kicked_regardless_of_phase():
    """The has_kicked gate itself, isolated from phase (2026-08-04): both envs fully inside the
    strike window, only has_kicked differs -- ball_proximity/contact_orientation must be nonzero
    before contact and exactly zero from the tick contact is confirmed, proving this is a real,
    independent gate rather than incidentally implied by the phase gate."""
    env = _fake_env(torch.tensor([10, 10]))  # both in strike phase
    tracker = _FakeTracker(2)
    tracker.has_kicked = torch.tensor([False, True])
    with (
        patch.object(s, "_tracker", return_value=tracker),
        patch.object(s, "current_w_g", return_value=torch.ones(2)),
    ):
        for term in _PRE_CONTACT_STRIKE_GATED_TERMS:
            out = term(env)
            assert out[0].item() > 0.0, f"{term.__name__} env 0 (pre-contact) should be nonzero"
            assert out[1].item() == 0.0, f"{term.__name__} env 1 (post-contact) should be zero"


def test_mixed_phase_post_locomotion_gated_terms_per_env_independently():
    """Two envs, one in post-kick-stabilization (well past the strike) and one in locomotion --
    proves the 4 outcome terms' gate is applied per-env, and specifically that post-kick-
    stabilization pays out (the behavior the old strike-only gate got wrong)."""
    env = _fake_env(torch.tensor([100, 5]))
    tracker = _FakeTracker(2)
    with (
        patch.object(s, "_tracker", return_value=tracker),
        patch.object(s, "current_w_g", return_value=torch.ones(2)),
    ):
        for term in _POST_LOCOMOTION_GATED_TERMS:
            out = term(env)
            assert out[0].item() > 0.0, f"{term.__name__} env 0 (post-kick-stabilization) should be nonzero"
            assert out[1].item() == 0.0, f"{term.__name__} env 1 (locomotion) should be zero"


def test_predicted_error_requires_both_moving_and_has_kicked():
    """predicted_error_ball_to_target (2026-08-04): requires moving AND has_kicked, not moving
    alone -- the regression test for the measured contamination this closes (a live checkpoint
    probe found 12.8% of post-locomotion-phase ticks were moving-without-has_kicked, e.g. a
    torso/wrong-foot brush setting the ball rolling with no confirmed kick-foot contact; see the
    term's own docstring for the full numbers). All 4 envs are fully post-locomotion (phase gate
    open); only moving/has_kicked vary."""
    env = _fake_env(torch.tensor([100, 100, 100, 100]))
    tracker = _FakeTracker(4)
    tracker.ball_speed = torch.tensor([2.0, 2.0, 0.0, 0.0])  # moving, moving, still, still
    tracker.has_kicked = torch.tensor([True, False, True, False])
    with (
        patch.object(s, "_tracker", return_value=tracker),
        patch.object(s, "current_w_g", return_value=torch.ones(4)),
    ):
        out = s.predicted_error_ball_to_target(env)
    assert out[0].item() > 0.0, "moving AND has_kicked (the legitimate case) should be nonzero"
    assert out[1].item() == 0.0, "moving but NOT has_kicked (the contamination case) must be zero"
    assert out[2].item() == 0.0, "has_kicked but NOT moving (ball already stopped) should be zero"
    assert out[3].item() == 0.0, "neither moving nor has_kicked should be zero"


def test_predicted_error_mixed_batch_has_kicked_gate_per_env_independently():
    """Two envs, both moving, one with confirmed contact and one without -- proves the has_kicked
    gate is applied per-env via elementwise multiplication, not a batch-wide short-circuit."""
    env = _fake_env(torch.tensor([100, 100]))
    tracker = _FakeTracker(2)
    tracker.ball_speed = torch.tensor([2.0, 2.0])
    tracker.has_kicked = torch.tensor([True, False])
    with (
        patch.object(s, "_tracker", return_value=tracker),
        patch.object(s, "current_w_g", return_value=torch.ones(2)),
    ):
        out = s.predicted_error_ball_to_target(env)
    assert out[0].item() > 0.0, "env 0 (has_kicked) should be nonzero"
    assert out[1].item() == 0.0, "env 1 (moving without has_kicked) should be zero"


def test_ball_proximity_gaussian_kernel_matches_formula():
    """ball_proximity's kernel is exp(-[d-r]+^2/sigma^2) (Gaussian, changed 2026-08-01 from the
    original Laplacian exp(-[d-r]+/sigma) -- see the term's own docstring for why). Verified
    directly against the formula at d-r = 0, sigma, 2*sigma, isolated from the phase/OOD gates
    (env fully in strike phase, w_g=1)."""
    num_envs = 3
    env = _fake_env(torch.tensor([10, 10, 10]))  # all in strike phase (_STRIKE_START_IDX=10)
    tracker = _FakeTracker(num_envs)
    tracker.has_kicked = torch.zeros(num_envs, dtype=torch.bool)  # pre-contact: gate open (2026-08-04)
    sigma = 0.35
    # foot-to-ball-CENTER distances = ball_radius + {0, sigma, 2*sigma}, so (d - ball_radius),
    # the quantity the kernel actually uses, lands exactly on {0, sigma, 2*sigma}.
    tracker._foot_pos = torch.tensor(
        [
            [tracker.ball_radius + 0.0, 0.0, 0.0],
            [tracker.ball_radius + sigma, 0.0, 0.0],
            [tracker.ball_radius + 2 * sigma, 0.0, 0.0],
        ]
    )
    tracker.ball_pos_w = torch.zeros(num_envs, 3)
    with (
        patch.object(s, "_tracker", return_value=tracker),
        patch.object(s, "current_w_g", return_value=torch.ones(num_envs)),
    ):
        out = s.ball_proximity(env, sigma=sigma)
    expected = torch.tensor([1.0, math.exp(-1.0), math.exp(-4.0)])
    assert torch.allclose(out, expected, atol=1e-5), f"expected {expected}, got {out}"


def test_ball_proximity_gaussian_falloff_steeper_than_laplacian_at_large_distance():
    """Regression guard that the 2026-08-01 shape change is real, not just a docstring update: at
    d-r = 2*sigma, the Gaussian value must be strictly LESS than what the old Laplacian
    (exp(-d/sigma)) would give at the same distance -- if this ever silently reverts to linear,
    this test catches it even though both kernels agree at d-r=0 (where every other test in this
    file's _FakeTracker fixture happens to sit)."""
    env = _fake_env(torch.tensor([10]))
    tracker = _FakeTracker(1)
    tracker.has_kicked = torch.zeros(1, dtype=torch.bool)  # pre-contact: gate open (2026-08-04)
    sigma = 0.35
    d_minus_r = 2 * sigma
    tracker._foot_pos = torch.tensor([[tracker.ball_radius + d_minus_r, 0.0, 0.0]])
    tracker.ball_pos_w = torch.zeros(1, 3)
    with (
        patch.object(s, "_tracker", return_value=tracker),
        patch.object(s, "current_w_g", return_value=torch.ones(1)),
    ):
        out = s.ball_proximity(env, sigma=sigma)
    laplacian_value = math.exp(-d_minus_r / sigma)
    assert out.item() < laplacian_value, (
        f"Gaussian value {out.item()} should be strictly less than the old Laplacian's "
        f"{laplacian_value} at d-r=2*sigma -- did the kernel shape revert?"
    )


def _quat_pitch_about_y(theta: float) -> list[float]:
    """xyzw quaternion for a rotation of `theta` about local Y -- maps local +x (toe-forward) to
    world (cos(theta), 0, -sin(theta)). theta > 0 => toe points down (negative world Z)."""
    return [0.0, math.sin(theta / 2), 0.0, math.cos(theta / 2)]


def test_foot_strike_pitch_rewards_toe_down_penalizes_toe_up():
    """foot_strike_pitch's kernel is -toe_dir_world.z: a foot pitched toe-DOWN (plantarflexed, top
    of foot toward the ball) must score strictly positive, toe-UP (dorsiflexed, sole toward the
    ball) strictly negative, and dead-flat (world-horizontal toe, "straight") exactly zero -- the
    reward has no separate penalty term, so the sign flip itself is the whole mechanism (see the
    term's own docstring). Proximity gate pinned to 1.0 (foot exactly at ball_radius from ball
    center) so only the pitch kernel is under test."""
    env = _fake_env(torch.tensor([10, 10, 10]))  # all in strike phase
    tracker = _FakeTracker(3)
    tracker.has_kicked = torch.zeros(3, dtype=torch.bool)  # pre-contact: gate open
    tracker._foot_pos = torch.zeros(3, 3)
    tracker._foot_pos[:, 0] = tracker.ball_radius  # dist - ball_radius = 0 => proximity_gate = 1.0
    tracker.ball_pos_w = torch.zeros(3, 3)
    tracker._foot_quat = torch.tensor(
        [
            _quat_pitch_about_y(math.pi / 2),  # toe straight down
            _quat_pitch_about_y(0.0),  # toe horizontal ("straight")
            _quat_pitch_about_y(-math.pi / 2),  # toe straight up
        ]
    )
    with (
        patch.object(s, "_tracker", return_value=tracker),
        patch.object(s, "current_w_g", return_value=torch.ones(3)),
    ):
        out = s.foot_strike_pitch(env)
    assert torch.allclose(out, torch.tensor([1.0, 0.0, -1.0]), atol=1e-5), f"got {out}"


def test_foot_strike_pitch_invariant_to_ankle_roll():
    """The whole point of reading local +x (the same axis the ankle-roll joint itself rotates
    about, per the G1 URDF): rotating the foot about its own toe-forward axis must NOT change
    foot_strike_pitch's output at all -- roll/inversion never contaminates the pitch reading, by
    construction, not by any explicit filtering. Composes a fixed 30 deg pitch with a swept roll
    (about the resulting toe axis) and asserts the reward is bit-identical across the sweep."""
    from holosoma.utils.rotations import quat_mul

    env = _fake_env(torch.tensor([10, 10, 10]))
    tracker = _FakeTracker(3)
    tracker.has_kicked = torch.zeros(3, dtype=torch.bool)
    tracker._foot_pos = torch.zeros(3, 3)
    tracker._foot_pos[:, 0] = tracker.ball_radius
    tracker.ball_pos_w = torch.zeros(3, 3)

    pitch_q = torch.tensor(_quat_pitch_about_y(math.pi / 6))  # fixed 30 deg toe-down
    rolls = [0.0, math.pi / 4, math.pi]  # swept roll about the toe axis itself
    roll_quats = torch.stack([torch.tensor([math.sin(r / 2), 0.0, 0.0, math.cos(r / 2)]) for r in rolls])
    # Roll is applied about the FOOT's own local +x (its frame, post-pitch) -- intrinsic
    # composition, so roll is the second (right-hand) factor: q = pitch_q * roll_q.
    tracker._foot_quat = torch.stack([quat_mul(pitch_q, rq, w_last=True) for rq in roll_quats])

    with (
        patch.object(s, "_tracker", return_value=tracker),
        patch.object(s, "current_w_g", return_value=torch.ones(3)),
    ):
        out = s.foot_strike_pitch(env)
    assert torch.allclose(out, out[0].expand_as(out), atol=1e-5), (
        f"foot_strike_pitch must be invariant to ankle-roll, got {out}"
    )


# ============================================================================================
# reference_relative per-skill support (2026-08-15, "simultaneous per-skill task configs"):
# reference_relative may now arrive as a per-env [num_envs] tensor (RewardManager's
# params_per_skill gather) instead of the plain bool it always used to be.
# ============================================================================================


def test_foot_strike_pitch_reference_relative_scalar_true_unchanged():
    """Sanity: the original scalar True path (relative to the reference clip's own pitch) must
    still work exactly as before this mechanism existed."""
    env = _fake_env(torch.tensor([10, 10]))
    tracker = _FakeTracker(2)
    tracker.has_kicked = torch.zeros(2, dtype=torch.bool)
    tracker._foot_pos = torch.zeros(2, 3)
    tracker._foot_pos[:, 0] = tracker.ball_radius
    tracker.ball_pos_w = torch.zeros(2, 3)
    tracker._foot_quat = torch.tensor([_quat_pitch_about_y(math.pi / 2)] * 2)  # toe straight down, pitch_signal=1.0
    tracker.kick_foot_reference_quat_w = lambda env: torch.tensor([_quat_pitch_about_y(math.pi / 2)] * 2)  # ref matches robot -> relative=0

    with (
        patch.object(s, "_tracker", return_value=tracker),
        patch.object(s, "current_w_g", return_value=torch.ones(2)),
    ):
        out = s.foot_strike_pitch(env, reference_relative=True)
    assert torch.allclose(out, torch.zeros(2), atol=1e-5)


def test_foot_strike_pitch_reference_relative_per_env_tensor_selects_per_env():
    """env 0's skill has reference_relative=True (score = robot - reference = 0, since they
    match), env 1's skill has it False (score = absolute pitch_signal = 1.0) -- both envs share
    the SAME robot/reference pose, only the per-env flag differs, isolating the select logic."""
    env = _fake_env(torch.tensor([10, 10]))
    tracker = _FakeTracker(2)
    tracker.has_kicked = torch.zeros(2, dtype=torch.bool)
    tracker._foot_pos = torch.zeros(2, 3)
    tracker._foot_pos[:, 0] = tracker.ball_radius
    tracker.ball_pos_w = torch.zeros(2, 3)
    tracker._foot_quat = torch.tensor([_quat_pitch_about_y(math.pi / 2)] * 2)  # toe straight down both envs
    tracker.kick_foot_reference_quat_w = lambda env: torch.tensor([_quat_pitch_about_y(math.pi / 2)] * 2)  # matches

    with (
        patch.object(s, "_tracker", return_value=tracker),
        patch.object(s, "current_w_g", return_value=torch.ones(2)),
    ):
        out = s.foot_strike_pitch(env, reference_relative=torch.tensor([1.0, 0.0]))
    assert torch.allclose(out, torch.tensor([0.0, 1.0]), atol=1e-5), f"got {out}"


# ---------------------------------------------------------------------------------------------
# ball_velocity's three OPT-IN retunes (2026-08-21) -- v_ref / latched-peak speed / post-loco gate.
# All three default OFF; these pin both that the defaults reproduce the ORIGINAL behavior and that
# each toggle does what it claims, independently. See shooting.py::ball_velocity's docstring.
# ---------------------------------------------------------------------------------------------


class _SpeedTracker(_FakeTracker):
    """_FakeTracker with the two speed buffers set to DIFFERENT values, so a test can tell which
    one ball_velocity actually read."""

    def __init__(self, num_envs: int, peak: float, instantaneous: float):
        super().__init__(num_envs)
        self.max_ball_speed = torch.full((num_envs,), peak)
        self.ball_speed = torch.full((num_envs,), instantaneous)


def _ball_velocity_at(ts, tracker, **kwargs):
    env = _fake_env(ts)
    with (
        patch.object(s, "_tracker", return_value=tracker),
        patch.object(s, "current_w_g", return_value=torch.ones(len(ts))),
    ):
        return s.ball_velocity(env, **kwargs)


def test_ball_velocity_defaults_reproduce_original_behavior():
    """No kwargs => instantaneous ball_speed, strike-only gate, v_ref=5.0 -- bit-identical to
    before the 2026-08-21 retune existed."""
    ts = torch.tensor([5, 15, 25, 100])  # loco, strike, post-kick, hold
    out = _ball_velocity_at(ts, _SpeedTracker(4, peak=5.0, instantaneous=3.0))
    assert torch.equal(out > 0.0, torch.tensor([False, True, False, False])), f"strike-only gate: {out}"
    # instantaneous 3.0 with v_ref 5.0 -> 9/(9+25); had it read the peak (also 5.0 here it would
    # differ), or used v_ref=2.0, the value would not match this.
    assert out[1].item() == pytest.approx(9.0 / 34.0, rel=1e-5)


def test_ball_velocity_v_ref_changes_only_the_kernel_not_the_gate():
    ts = torch.tensor([5, 15, 25, 100])
    out = _ball_velocity_at(ts, _SpeedTracker(4, peak=5.0, instantaneous=3.0), v_ref=2.0)
    assert torch.equal(out > 0.0, torch.tensor([False, True, False, False])), "gate must be unchanged"
    assert out[1].item() == pytest.approx(9.0 / 13.0, rel=1e-5)  # 3^2/(3^2+2^2)


def test_ball_velocity_use_latched_peak_speed_reads_the_other_buffer():
    """Peak and instantaneous deliberately differ -- proves which buffer was read."""
    ts = torch.tensor([15])
    tr = _SpeedTracker(1, peak=6.0, instantaneous=1.0)
    off = _ball_velocity_at(ts, tr)[0].item()
    on = _ball_velocity_at(ts, tr, use_latched_peak_speed=True)[0].item()
    assert off == pytest.approx(1.0 / 26.0, rel=1e-5), "default must read INSTANTANEOUS"
    assert on == pytest.approx(36.0 / 61.0, rel=1e-5), "opt-in must read the LATCHED PEAK"
    assert on > off


def test_ball_velocity_use_post_locomotion_gate_widens_the_window():
    """The whole point of the opt-in: pay through post-kick-standing/recovery/hold, not just the
    strike window."""
    ts = torch.tensor([5, 15, 25, 100])
    tr = _SpeedTracker(4, peak=5.0, instantaneous=3.0)
    off = _ball_velocity_at(ts, tr)
    on = _ball_velocity_at(ts, tr, use_post_locomotion_gate=True)
    assert torch.equal(off > 0.0, torch.tensor([False, True, False, False]))
    assert torch.equal(on > 0.0, torch.tensor([False, True, True, True])), f"post-loco gate: {on}"


def test_ball_velocity_latched_peak_does_not_decay_as_the_ball_slows():
    """The two opt-ins are designed to be used together: with the wide gate but the INSTANTANEOUS
    speed, a decelerating ball's reward collapses -- which is exactly the failure the latched peak
    exists to prevent. Pins that pairing rather than leaving it to prose."""
    ts = torch.tensor([100])  # deep in the hold phase, ball has rolled to a near-stop
    tr = _SpeedTracker(1, peak=5.0, instantaneous=0.05)
    inst = _ball_velocity_at(ts, tr, use_post_locomotion_gate=True)[0].item()
    peak = _ball_velocity_at(ts, tr, use_post_locomotion_gate=True, use_latched_peak_speed=True)[0].item()
    assert inst < 0.001, "instantaneous speed collapses once the ball stops"
    assert peak == pytest.approx(25.0 / 50.0, rel=1e-5), "latched peak holds its value"


def test_ball_velocity_still_requires_has_kicked_under_every_toggle():
    """has_kicked gating is NOT one of the opt-ins -- it must hold in every combination."""
    ts = torch.tensor([15, 15])
    for kwargs in ({}, {"v_ref": 2.0}, {"use_latched_peak_speed": True},
                   {"use_post_locomotion_gate": True},
                   {"v_ref": 2.0, "use_latched_peak_speed": True, "use_post_locomotion_gate": True}):
        tr = _SpeedTracker(2, peak=5.0, instantaneous=3.0)
        tr.has_kicked = torch.tensor([False, True])
        out = _ball_velocity_at(ts, tr, **kwargs)
        assert out[0].item() == 0.0, f"{kwargs}: must be zero before contact"
        assert out[1].item() > 0.0, f"{kwargs}: must be nonzero after contact"
