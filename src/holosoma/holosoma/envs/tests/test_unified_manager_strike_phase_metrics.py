"""Unit tests for the strike-phase tracking-metric split (2026-08-24, user-requested).

WHY THIS EXISTS. ``kick_*_swing`` is masked by ``in_kicking_phase`` = modes 1+2 = the locomotion
APPROACH *plus* the strike. Measured against this project's own clip boundaries the strike is only
21%/22%/30% of that window for skill011/012/013, so ``error_body_pos_swing`` is 70-79% approach
WALK and cannot answer "did the leg swing diverge from the clip". ``kick_*_strike`` (masked by
``in_strike_phase``, modes 2 only -- the same signal shooting.py gates its 6 shooting terms on) is
the window that actually answers it, and ``kick_*_approach`` is the remainder.

These tests pin the invariant that makes the new keys trustworthy: _strike and _approach PARTITION
_swing exactly (no overlap, nothing dropped, correct membership), so a reader can always
reconstruct the old number and the two halves can never silently disagree with their parent.
"""

import torch


def _masks(kick_mask: torch.Tensor, in_kicking: torch.Tensor, in_strike: torch.Tensor):
    """The exact mask algebra from UnifiedManager's kick-stats block."""
    swing_mask = kick_mask & in_kicking
    hold_mask = kick_mask & ~in_kicking
    strike_mask = swing_mask & in_strike
    approach_mask = swing_mask & ~in_strike
    return swing_mask, hold_mask, strike_mask, approach_mask


def test_strike_and_approach_exactly_partition_swing():
    kick = torch.tensor([True, True, True, True, False, True])
    kicking = torch.tensor([True, True, True, False, True, False])
    strike = torch.tensor([True, False, True, True, True, True])
    swing, hold, strike_m, approach_m = _masks(kick, kicking, strike)
    # disjoint
    assert not bool((strike_m & approach_m).any())
    # exhaustive over swing
    assert torch.equal(strike_m | approach_m, swing)
    # swing/hold still partition the kick population
    assert torch.equal(swing | hold, kick)
    assert not bool((swing & hold).any())


def test_strike_never_includes_a_non_kick_or_hold_phase_env():
    """An env past stand_start_idx (hold) must never leak into _strike even if in_strike_phase is
    stale/True for it, and a locomotion-partitioned env never appears at all."""
    kick = torch.tensor([True, True, False])
    kicking = torch.tensor([False, True, True])   # env0 is in HOLD, env2 is not a kick env
    strike = torch.tensor([True, True, True])     # deliberately True everywhere
    _, hold, strike_m, _ = _masks(kick, kicking, strike)
    assert not bool((strike_m & hold).any()), "a hold-phase env leaked into _strike"
    assert not bool(strike_m[2]), "a non-kick env leaked into _strike"
    assert bool(strike_m[1])


def test_swing_mean_is_the_size_weighted_blend_of_strike_and_approach():
    """The dilution this split exists to expose: with the strike a minority of the swing window,
    the _swing mean sits far closer to the approach value than the strike value."""
    kick = torch.ones(10, dtype=torch.bool)
    kicking = torch.ones(10, dtype=torch.bool)
    # 2 of 10 envs mid-strike -- roughly this project's 21-30% strike share
    strike = torch.tensor([True, True] + [False] * 8)
    swing, _, strike_m, approach_m = _masks(kick, kicking, strike)
    val = torch.tensor([0.50, 0.50] + [0.05] * 8)  # strike tracks badly, approach tracks well
    s_mean = val[swing].mean()
    strike_mean = val[strike_m].mean()
    approach_mean = val[approach_m].mean()
    assert torch.isclose(strike_mean, torch.tensor(0.50))
    assert torch.isclose(approach_mean, torch.tensor(0.05))
    # the parent number is the size-weighted blend, and lands nowhere near the strike value
    expected = (strike_m.sum() * strike_mean + approach_m.sum() * approach_mean) / swing.sum()
    assert torch.isclose(s_mean, expected)
    assert torch.isclose(s_mean, torch.tensor(0.14))
    assert s_mean < 0.3 * strike_mean, "a 10x strike-phase error is hidden by the approach"


def test_no_strike_envs_yields_an_empty_mask_not_a_nan():
    """`.any()` guards the emit: with nobody mid-strike the key is skipped, never logged as NaN."""
    kick = torch.ones(4, dtype=torch.bool)
    kicking = torch.ones(4, dtype=torch.bool)
    strike = torch.zeros(4, dtype=torch.bool)
    swing, _, strike_m, approach_m = _masks(kick, kicking, strike)
    assert not bool(strike_m.any())
    assert torch.equal(approach_m, swing)
    assert torch.isnan(torch.tensor([1.0, 2.0, 3.0, 4.0])[strike_m].mean())


def test_strike_active_frac_is_over_all_envs_not_just_kick_envs():
    """Denominator convention matches kick_active_frac (share of ALL envs), so the two are
    directly comparable on one dashboard."""
    kick = torch.tensor([True, True, True, True, False, False, False, False])
    kicking = torch.ones(8, dtype=torch.bool)
    strike = torch.tensor([True, True, False, False, True, True, True, True])
    _, _, strike_m, _ = _masks(kick, kicking, strike)
    assert torch.isclose(strike_m.float().mean(), torch.tensor(0.25))
