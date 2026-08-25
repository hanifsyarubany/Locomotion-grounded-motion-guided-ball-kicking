"""Unit tests for rsi_span_end_idx's per-skill tensor support (2026-08-15, "simultaneous
per-skill task configs", Tier 3 Group A). This function feeds the RSI (random-start-init) span
MotionCommand.reset() samples a uniform phase against -- see its own docstring for the full
rationale (RoboNaldo's start_time_sampling_fraction port).
"""

from __future__ import annotations

import torch

from holosoma.managers.command.terms.wbt import rsi_span_end_idx


def test_scalar_false_returns_motion_end_idx_unchanged():
    motion_end_idx = torch.tensor([100, 200])
    pre_recovery = torch.tensor([80, 150])
    out = rsi_span_end_idx(motion_end_idx, pre_recovery, rsi_scope_to_authored_clip=False)
    assert torch.equal(out, motion_end_idx)


def test_scalar_true_returns_pre_recovery_unchanged():
    motion_end_idx = torch.tensor([100, 200])
    pre_recovery = torch.tensor([80, 150])
    out = rsi_span_end_idx(motion_end_idx, pre_recovery, rsi_scope_to_authored_clip=True)
    assert torch.equal(out, pre_recovery)


def test_per_env_tensor_selects_per_env():
    """env 0's skill has rsi_scope_to_authored_clip=True (scoped to pre_recovery), env 1's has it
    False (full motion_end_idx) -- isolates the torch.where select."""
    motion_end_idx = torch.tensor([100, 200])
    pre_recovery = torch.tensor([80, 150])
    mask = torch.tensor([True, False])
    out = rsi_span_end_idx(motion_end_idx, pre_recovery, rsi_scope_to_authored_clip=mask)
    assert torch.equal(out, torch.tensor([80, 200]))


def test_per_env_tensor_all_true_matches_scalar_true():
    motion_end_idx = torch.tensor([100, 200, 300])
    pre_recovery = torch.tensor([80, 150, 250])
    mask = torch.ones(3, dtype=torch.bool)
    out = rsi_span_end_idx(motion_end_idx, pre_recovery, rsi_scope_to_authored_clip=mask)
    assert torch.equal(out, pre_recovery)


def test_per_env_tensor_all_false_matches_scalar_false():
    motion_end_idx = torch.tensor([100, 200, 300])
    pre_recovery = torch.tensor([80, 150, 250])
    mask = torch.zeros(3, dtype=torch.bool)
    out = rsi_span_end_idx(motion_end_idx, pre_recovery, rsi_scope_to_authored_clip=mask)
    assert torch.equal(out, motion_end_idx)


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
