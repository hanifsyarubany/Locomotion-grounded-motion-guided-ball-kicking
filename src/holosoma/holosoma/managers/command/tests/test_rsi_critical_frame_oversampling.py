"""Unit tests for ``rsi_span_end_idx`` and ``critical_frame_oversample_time_steps``
(managers/command/terms/wbt.py) -- 2026-08-05, ported from RoboNaldo (arXiv:2606.11092)'s
``start_time_sampling_fraction`` / ``critical_frame_adaptive_sampling``. Both are pure module-level
functions extracted from MotionCommand.reset()'s uniform-phase RSI branch specifically so they're
directly testable without faking the rest of that large, many-dependency method -- same rationale
and pattern as ``draw_position_noise_with_ood`` immediately above them in the same file (see
test_ood_position_noise.py for the established RNG-testing convention this file follows: seed
torch, then assert structural/statistical invariants rather than exact values).
"""

from __future__ import annotations

import torch

from holosoma.managers.command.terms.wbt import critical_frame_oversample_time_steps, rsi_span_end_idx

# ============================================================================================
# rsi_span_end_idx
# ============================================================================================


def test_rsi_span_end_idx_default_false_returns_motion_end_idx_unchanged():
    motion_end_idx = torch.tensor([100, 250, 400])
    pre_recovery = torch.tensor([80, 200, 350])
    out = rsi_span_end_idx(motion_end_idx, pre_recovery, rsi_scope_to_authored_clip=False)
    assert torch.equal(out, motion_end_idx)


def test_rsi_span_end_idx_true_returns_pre_recovery_motion_end_idx():
    motion_end_idx = torch.tensor([100, 250, 400])
    pre_recovery = torch.tensor([80, 200, 350])
    out = rsi_span_end_idx(motion_end_idx, pre_recovery, rsi_scope_to_authored_clip=True)
    assert torch.equal(out, pre_recovery)


def test_rsi_span_end_idx_is_the_exact_same_tensor_object_when_false():
    """Not just numerically equal -- the exact-no-op path must return the SAME tensor, matching
    this project's own established convention for verified no-ops (e.g. _swing_widened_sigma's
    own docstring: 'not just a numerically-equal one')."""
    motion_end_idx = torch.tensor([100, 250])
    pre_recovery = torch.tensor([80, 200])
    out = rsi_span_end_idx(motion_end_idx, pre_recovery, rsi_scope_to_authored_clip=False)
    assert out is motion_end_idx


# ============================================================================================
# critical_frame_oversample_time_steps
# ============================================================================================


def test_oversample_prob_zero_is_bit_identical_no_op():
    torch.manual_seed(0)
    time_steps = torch.tensor([10, 55, 200, 7])
    out = critical_frame_oversample_time_steps(
        time_steps,
        start_idx=torch.tensor([0, 0, 0, 0]),
        span_end_idx=torch.tensor([300, 300, 300, 300]),
        strike_start_idx=torch.tensor([150, 150, 150, 150]),
        oversample_prob=0.0,
        window=10,
        device="cpu",
    )
    assert torch.equal(out, time_steps)


def test_oversample_prob_one_always_replaces_and_stays_within_window():
    torch.manual_seed(0)
    n = 2000
    time_steps = torch.full((n,), 5)  # a value clearly OUTSIDE the window, to detect replacement
    strike_idx = torch.full((n,), 150)
    out = critical_frame_oversample_time_steps(
        time_steps,
        start_idx=torch.zeros(n, dtype=torch.long),
        span_end_idx=torch.full((n,), 300),
        strike_start_idx=strike_idx,
        oversample_prob=1.0,
        window=10,
        device="cpu",
    )
    assert torch.all(out != 5), "every env must have been replaced at oversample_prob=1.0"
    assert torch.all(out >= 140) and torch.all(out <= 160)  # strike_idx +/- window


def test_oversample_prob_fraction_selects_roughly_that_fraction():
    torch.manual_seed(0)
    n = 5000
    time_steps = torch.full((n,), 5)
    out = critical_frame_oversample_time_steps(
        time_steps,
        start_idx=torch.zeros(n, dtype=torch.long),
        span_end_idx=torch.full((n,), 300),
        strike_start_idx=torch.full((n,), 150),
        oversample_prob=0.2,
        window=10,
        device="cpu",
    )
    replaced_frac = (out != 5).float().mean().item()
    assert 0.15 < replaced_frac < 0.25


def test_oversample_window_clamped_to_start_idx_lower_bound():
    """strike_start_idx very close to start_idx -- the window's lower edge must clamp to
    start_idx, never draw a time_step below the motion's own start."""
    torch.manual_seed(0)
    n = 1000
    time_steps = torch.full((n,), -1)  # sentinel, replaced at prob=1.0
    out = critical_frame_oversample_time_steps(
        time_steps,
        start_idx=torch.full((n,), 100),
        span_end_idx=torch.full((n,), 300),
        strike_start_idx=torch.full((n,), 103),  # only 3 frames past start_idx, window=10
        oversample_prob=1.0,
        window=10,
        device="cpu",
    )
    assert torch.all(out >= 100), "must never draw below start_idx"
    assert torch.all(out <= 113)


def test_oversample_window_clamped_to_span_end_idx_upper_bound():
    """strike_start_idx very close to span_end_idx -- the window's upper edge must clamp to
    span_end_idx - 1, never draw a time_step at or past it (this is what makes the mechanism
    compose correctly with rsi_scope_to_authored_clip: pass pre_recovery_motion_end_idx as
    span_end_idx and the window can't reach the synthetic tail either)."""
    torch.manual_seed(0)
    n = 1000
    time_steps = torch.full((n,), -1)
    out = critical_frame_oversample_time_steps(
        time_steps,
        start_idx=torch.zeros(n, dtype=torch.long),
        span_end_idx=torch.full((n,), 200),
        strike_start_idx=torch.full((n,), 195),  # only 5 frames before span_end_idx, window=10
        oversample_prob=1.0,
        window=10,
        device="cpu",
    )
    assert torch.all(out <= 199), "must never draw at or past span_end_idx"
    assert torch.all(out >= 185)


def test_oversample_per_env_independent():
    """Two envs with very different strike_start_idx -- each drawn window must center on its OWN
    strike_start_idx, not a shared/averaged one."""
    torch.manual_seed(0)
    time_steps = torch.tensor([-1, -1])
    out = critical_frame_oversample_time_steps(
        time_steps,
        start_idx=torch.tensor([0, 0]),
        span_end_idx=torch.tensor([500, 500]),
        strike_start_idx=torch.tensor([50, 400]),
        oversample_prob=1.0,
        window=5,
        device="cpu",
    )
    assert 45 <= out[0].item() <= 55
    assert 395 <= out[1].item() <= 405


def test_oversample_zero_window_always_lands_exactly_on_strike_start_idx():
    torch.manual_seed(0)
    n = 50
    time_steps = torch.full((n,), -1)
    out = critical_frame_oversample_time_steps(
        time_steps,
        start_idx=torch.zeros(n, dtype=torch.long),
        span_end_idx=torch.full((n,), 300),
        strike_start_idx=torch.full((n,), 150),
        oversample_prob=1.0,
        window=0,
        device="cpu",
    )
    assert torch.all(out == 150)
