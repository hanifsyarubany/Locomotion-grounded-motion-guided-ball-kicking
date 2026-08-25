"""Unit tests for shooting.py's kick-detection fix (2026-08-01): `_is_foot_contact`/`_detect_kick`,
the pure per-env functions factored out of `_ShotTracker.update()`'s previously ball-motion-only
has_kicked computation, plus their shared dependencies `_is_post_locomotion`/`_is_in_strike_phase`.

Four rounds of tightening, all still relevant to what's tested here:
1. First round: has_kicked gained a geometric, foot-TARGETED OR condition (the env's assigned
   kick foot's own contact point -- already skill-resolved left/right via kick_foot_index, the
   same body ball_proximity/contact_orientation use -- actually touching the ball), and the whole
   expression got gated by `_is_post_locomotion` (time_steps >= strike_start_idx) so has_kicked can
   no longer latch True during locomotion-approach at all.
2. Second round (same day, user-flagged): the first round still let the ORIGINAL ball-motion proxy
   (displacement from spawn / ball speed) trigger has_kicked completely independently of foot
   contact -- live measurement found this was driving 79% of all detections with ZERO geometric
   confirmation, an uncontrolled body-identity gap a wrong-foot/torso touch could exploit even
   inside the strike window (the post_locomotion gate only closes the TEMPORAL leak, not this
   one). Fix: ball_moved now REQUIRES `foot_contact_ever` (a sticky per-attempt latch on
   `_ShotTracker`, set the first tick `_is_foot_contact` fires in-phase) -- it supplies timing
   precision for a fast strike whose motion crosses threshold before/after the single geometric
   sample that confirmed contact, not an independent, unverified trigger.
3. Third round (same day): what COUNTS as foot_contact itself got more precise. `_ShotTracker`
   now prefers TRUE geometric contact on IsaacSim (two per-foot PhysX ContactSensors filtered
   against the ball, read via BaseSimulator.get_ball_foot_contact_pos_w -- real override only on
   IsaacSim's simulator subclass, None everywhere else), gathered per-env by `_geometric_foot_contact`
   (tested below), falling back to the original offset-point+margin `_is_foot_contact` only when
   the backend doesn't support it (MuJoCo, IsaacGym).
4. Fourth round (same day, user-requested): the phase gate on has_kicked's TRIGGER narrowed a
   second time, from `_is_post_locomotion` (True from strike_start_idx onward, unbounded) to
   `_is_in_strike_phase` (True only for strike_start_idx <= time_steps < stand_start_idx) --
   `_detect_kick`'s gating parameter is now named `in_strike_phase`, not `post_locomotion`. A kick
   attempt may now only START within the authored swing itself; it may NOT first trigger during
   post-kick-standing/recovery/hold. has_kicked is still a sticky latch (`|=`, only cleared on
   new_attempt), so a strike that DOES land within the strike window still correctly reads True
   for the rest of the attempt -- this only closes off LATE triggering, not the latch's own
   persistence.

These pure functions take plain tensors, no env/simulator dependency, so unlike the rest of
_ShotTracker (see test_shooting_strike_gate.py's own docstring on why THAT stays
simulator-verified rather than unit tested) this is directly, precisely unit-testable. The
foot_contact_ever LATCH's own update/reset lifecycle, and the actual sensor read
(get_ball_foot_contact_pos_w's IsaacSim override), live in _ShotTracker.update()/__init__ and
simulator/isaacsim/isaacsim.py respectively and are verified live (IsaacSim probe), consistent
with this project's convention.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

import holosoma.managers.reward.terms.shooting as s

_BALL_RADIUS = 0.11


def _fake_motion_command(time_steps, strike_start_idx=10, stand_start_idx=20):
    time_steps = torch.tensor(time_steps)
    return SimpleNamespace(
        time_steps=time_steps,
        strike_start_idx=torch.tensor([strike_start_idx]),
        stand_start_idx=torch.tensor([stand_start_idx]),
        motion_ids=torch.zeros_like(time_steps),
        in_strike_phase=(time_steps >= strike_start_idx) & (time_steps < stand_start_idx),
    )


def _fake_env(motion_command):
    return SimpleNamespace(
        command_manager=SimpleNamespace(get_state=lambda name: motion_command if name == "motion_command" else None)
    )


def test_is_post_locomotion_matches_time_steps_vs_strike_start_idx():
    # Still used by _post_locomotion_multiplier (the 3 outcome terms' own PAYOUT gate, unchanged
    # and deliberately wider than has_kicked's own trigger window) -- unaffected by round 4.
    mc = _fake_motion_command([5, 10, 15, 20])
    out = s._is_post_locomotion(_fake_env(mc))
    assert torch.equal(out, torch.tensor([False, True, True, True]))


def test_is_in_strike_phase_reads_motion_command_directly():
    # locomotion (5), strike (10, 15), post-kick (20) -- only the strike-window ticks are True.
    # Strictly narrower than _is_post_locomotion above: False again once time_steps >= stand_start_idx.
    mc = _fake_motion_command([5, 10, 15, 20], strike_start_idx=10, stand_start_idx=20)
    out = s._is_in_strike_phase(_fake_env(mc))
    assert torch.equal(out, torch.tensor([False, True, True, False]))


def test_is_foot_contact_margin_boundary():
    threshold = _BALL_RADIUS + s._KICK_DETECT_FOOT_CONTACT_MARGIN_M
    dist = torch.tensor([threshold - 1e-4, threshold + 1e-4])
    out = s._is_foot_contact(dist, _BALL_RADIUS)
    assert torch.equal(out, torch.tensor([True, False]))


def _detect(foot_contact, foot_contact_ever, displacement, speed, in_strike):
    return s._detect_kick(
        foot_contact=torch.tensor(foot_contact),
        foot_contact_ever=torch.tensor(foot_contact_ever),
        displacement=torch.tensor(displacement),
        ball_speed=torch.tensor(speed),
        in_strike_phase=torch.tensor(in_strike),
    )


def test_foot_contact_this_tick_triggers_regardless_of_foot_contact_ever():
    out = _detect(foot_contact=[True], foot_contact_ever=[False], displacement=[0.0], speed=[0.0], in_strike=[True])
    assert torch.equal(out, torch.tensor([True]))


def test_ball_motion_does_not_trigger_without_foot_contact_ever():
    """The core of the second-round fix: ball_moved alone, with no confirmed foot contact this
    attempt, must NOT set has_kicked -- this is exactly the body-blind gap that let a wrong-foot
    or torso touch count under the first-round fix."""
    out = _detect(foot_contact=[False], foot_contact_ever=[False], displacement=[0.5], speed=[2.0], in_strike=[True])
    assert torch.equal(out, torch.tensor([False]))


def test_ball_motion_triggers_once_foot_contact_ever_is_true():
    # Foot isn't touching THIS tick, but did earlier this attempt (foot_contact_ever=True) -- the
    # ball's own motion can now supply the exact triggering tick.
    out = _detect(foot_contact=[False], foot_contact_ever=[True], displacement=[0.5], speed=[0.0], in_strike=[True])
    assert torch.equal(out, torch.tensor([True]))
    out2 = _detect(foot_contact=[False], foot_contact_ever=[True], displacement=[0.0], speed=[1.0], in_strike=[True])
    assert torch.equal(out2, torch.tensor([True]))


def test_neither_signal_present_does_not_trigger():
    out = _detect(foot_contact=[False], foot_contact_ever=[False], displacement=[0.0], speed=[0.0], in_strike=[True])
    assert torch.equal(out, torch.tensor([False]))


def test_foot_contact_does_not_trigger_during_locomotion_even_though_touching():
    # in_strike_phase=False (locomotion-approach case) must suppress EVERYTHING, including a
    # same-tick geometric contact -- a torso/approaching-leg graze that happens to register within
    # threshold before the strike has even begun must not set has_kicked.
    out = _detect(foot_contact=[True], foot_contact_ever=[True], displacement=[0.5], speed=[2.0], in_strike=[False])
    assert torch.equal(out, torch.tensor([False]))


def test_foot_contact_does_not_trigger_post_kick_even_though_touching():
    """Round 4's own new behavior, distinct from the test above: in_strike_phase=False can also
    mean POST-kick-standing/recovery/hold (time_steps >= stand_start_idx), not just locomotion --
    a contact (or ball motion) that first occurs there must not trigger has_kicked either, even
    though the old (round-1-3) gate, _is_post_locomotion, would have allowed it. This is the
    behavioral difference round 4 actually introduces."""
    out = _detect(foot_contact=[True], foot_contact_ever=[True], displacement=[0.5], speed=[2.0], in_strike=[False])
    assert torch.equal(out, torch.tensor([False]))


def test_mixed_batch_per_env_independence():
    # env0: foot contact this tick -> True (foot_contact_ever irrelevant here).
    # env1: foot contact this tick, but outside strike phase -> False.
    # env2: no foot contact this tick, but foot_contact_ever True + ball moved -> True.
    # env3: ball moved but foot_contact_ever False -> False (the second-round fix).
    out = _detect(
        foot_contact=[True, True, False, False],
        foot_contact_ever=[False, False, True, False],
        displacement=[0.0, 0.0, 0.5, 0.5],
        speed=[0.0, 0.0, 0.0, 0.0],
        in_strike=[True, False, True, True],
    )
    assert torch.equal(out, torch.tensor([True, False, True, False]))


def test_geometric_foot_contact_none_when_either_side_missing():
    """Backends without real contact sensors (MuJoCo, IsaacGym) report None from
    get_ball_foot_contact_pos_w for BOTH sides -- but the function must also treat just one side
    being None as "unsupported", not silently ignore the missing side, since a real IsaacSim
    override always returns both or neither (both sensors are always constructed together)."""
    is_left = torch.tensor([True, False])
    valid = torch.zeros(2, 3)
    assert s._geometric_foot_contact(is_left, None, None) is None
    assert s._geometric_foot_contact(is_left, valid, None) is None
    assert s._geometric_foot_contact(is_left, None, valid) is None


def test_geometric_foot_contact_gathers_correct_side_per_env():
    # env0 kicks with left foot and IS touching on the left sensor (non-NaN) -- True.
    # env1 kicks with right foot and IS touching on the right sensor (non-NaN) -- True.
    is_left = torch.tensor([True, False])
    left_pos = torch.tensor([[1.0, 2.0, 3.0], [float("nan")] * 3])
    right_pos = torch.tensor([[float("nan")] * 3, [4.0, 5.0, 6.0]])
    out = s._geometric_foot_contact(is_left, left_pos, right_pos)
    assert torch.equal(out, torch.tensor([True, True]))


def test_geometric_foot_contact_nan_means_not_touching():
    # env0 kicks with left foot, left sensor reports NaN (not touching) -- False, regardless of
    # what the (irrelevant, not-this-env's-foot) right sensor says.
    is_left = torch.tensor([True])
    left_pos = torch.tensor([[float("nan")] * 3])
    right_pos = torch.tensor([[9.0, 9.0, 9.0]])
    out = s._geometric_foot_contact(is_left, left_pos, right_pos)
    assert torch.equal(out, torch.tensor([False]))


def test_geometric_foot_contact_mixed_batch_per_env_independence():
    # env0: left foot, touching -> True. env1: left foot, not touching -> False.
    # env2: right foot, touching -> True. env3: right foot, not touching -> False.
    is_left = torch.tensor([True, True, False, False])
    left_pos = torch.tensor([[1.0, 1.0, 1.0], [float("nan")] * 3, [8.0, 8.0, 8.0], [8.0, 8.0, 8.0]])
    right_pos = torch.tensor([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [float("nan")] * 3])
    out = s._geometric_foot_contact(is_left, left_pos, right_pos)
    assert torch.equal(out, torch.tensor([True, False, True, False]))
