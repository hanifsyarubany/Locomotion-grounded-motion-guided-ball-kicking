"""Unit tests for L2-SP (2026-08-15, see FastSACConfig.l2sp_weight's own docstring for the full
feature/rationale -- a continual-learning anchor that pulls the actor back toward the checkpoint
it resumed from, so previously learned motion skills erode more slowly while a new skill trains
on top of a shared Actor/Critic and a uniformly sampled shared replay buffer).

Exercises FastSACAgent._apply_l2sp_pull and the module-level _validate_l2sp_anchor guard directly,
WITHOUT constructing a full FastSACAgent (no env, no sim, no GPU needed) -- same isolation
approach as test_replay_buffer_sanitize.py uses for SimpleReplayBuffer. _apply_l2sp_pull is called
unbound, i.e. FastSACAgent._apply_l2sp_pull(fake_self), against a minimal object that only carries
the handful of attributes the method actually reads (_l2sp_anchor, _l2sp_params, config,
_last_l2sp_drift) -- a standard trick for testing one method of a heavy class in isolation.
"""

from __future__ import annotations

import sys
import types

import pytest
import torch

sys.path.insert(0, "/workspaces/isaaclab_arena/submodules/workspaces/playground/unified_ball_kick_robonaldo/src/holosoma")

from holosoma.agents.fast_sac.fast_sac_agent import FastSACAgent, _validate_l2sp_anchor


def _make_fake_self(params: list[torch.Tensor], anchor: list[torch.Tensor] | None, l2sp_weight: float, actor_lr: float):
    fake = types.SimpleNamespace()
    fake._l2sp_params = params
    fake._l2sp_anchor = anchor
    fake._last_l2sp_drift = torch.zeros(())
    fake.config = types.SimpleNamespace(l2sp_weight=l2sp_weight, actor_learning_rate=actor_lr)
    return fake


def test_apply_pull_is_exact_no_op_when_no_anchor():
    """Default (l2sp_weight=0.0) path: load() never populates _l2sp_anchor, so this must be a
    true no-op -- untouched params, untouched drift -- not just numerically negligible."""
    p = torch.tensor([1.0, 2.0, 3.0])
    p_before = p.clone()
    fake = _make_fake_self(params=[p], anchor=None, l2sp_weight=0.01, actor_lr=3e-4)

    FastSACAgent._apply_l2sp_pull(fake)

    assert torch.equal(p, p_before)
    assert fake._last_l2sp_drift.item() == 0.0


def test_apply_pull_moves_params_toward_anchor_by_the_expected_amount():
    """p += lr * l2sp_weight * (anchor - p), verified against a hand-computed expectation."""
    p = torch.tensor([1.0, 2.0, 3.0])
    anchor = torch.tensor([0.0, 0.0, 0.0])
    lr = 3e-4
    weight = 0.01
    expected = p + lr * weight * (anchor - p)

    fake = _make_fake_self(params=[p], anchor=[anchor], l2sp_weight=weight, actor_lr=lr)
    FastSACAgent._apply_l2sp_pull(fake)

    assert torch.allclose(p, expected)


def test_apply_pull_never_overshoots_past_the_anchor_at_realistic_step_sizes():
    """Sanity bound: at any lr*weight < 1.0 (true for every value in the yaml's suggested table --
    even the 0.1 upper bound times a 3e-4 actor_learning_rate is ~3e-5), a single pull must move
    p STRICTLY closer to the anchor, never past it and never away from it."""
    p = torch.tensor([5.0])
    anchor = torch.tensor([1.0])
    fake = _make_fake_self(params=[p], anchor=[anchor], l2sp_weight=0.1, actor_lr=3e-4)
    dist_before = (p - anchor).abs().item()

    FastSACAgent._apply_l2sp_pull(fake)

    dist_after = (p - anchor).abs().item()
    assert 0.0 < dist_after < dist_before


def test_apply_pull_is_a_no_op_when_params_already_equal_anchor():
    p = torch.tensor([2.0, -1.0])
    anchor = torch.tensor([2.0, -1.0])
    fake = _make_fake_self(params=[p], anchor=[anchor], l2sp_weight=0.1, actor_lr=3e-4)

    FastSACAgent._apply_l2sp_pull(fake)

    assert torch.equal(p, torch.tensor([2.0, -1.0]))
    assert fake._last_l2sp_drift.item() == 0.0


def test_apply_pull_drift_diagnostic_is_total_accumulated_distance_from_anchor_not_the_step_size():
    """_last_l2sp_drift is documented as ||theta - theta_anchor||_2, meant to be watched over the
    WHOLE run ("should plateau rather than grow without bound") -- i.e. total accumulated drift
    from the anchor, NOT the (tiny, lr*weight-scaled, near-constant and therefore uninformative)
    size of this single update's step. Deliberately computed from the pre-scaling `pull` (=
    anchor - p) rather than the post-scaling applied delta."""
    p1 = torch.tensor([1.0, 0.0])
    p2 = torch.tensor([0.0, 1.0])
    a1 = torch.tensor([0.0, 0.0])
    a2 = torch.tensor([0.0, 0.0])
    # pull1 = a1-p1 = [-1, 0], norm=1; pull2 = a2-p2 = [0, -1], norm=1.
    expected_drift = torch.linalg.vector_norm(torch.tensor([1.0, 1.0]))  # sqrt(2)

    fake = _make_fake_self(params=[p1, p2], anchor=[a1, a2], l2sp_weight=0.01, actor_lr=3e-4)
    FastSACAgent._apply_l2sp_pull(fake)

    assert torch.allclose(fake._last_l2sp_drift, expected_drift, atol=1e-6)


def test_validate_l2sp_anchor_noop_when_weight_zero_even_without_anchor():
    """0.0 (the default) never raises, regardless of whether a checkpoint was loaded -- this is
    the "L2-SP simply doesn't exist" path and must never be gated on load() having happened."""
    _validate_l2sp_anchor(0.0, None)  # must not raise


def test_validate_l2sp_anchor_noop_when_weight_positive_and_anchor_present():
    _validate_l2sp_anchor(0.01, [torch.zeros(3)])  # must not raise


def test_validate_l2sp_anchor_raises_when_weight_positive_but_no_checkpoint_loaded():
    with pytest.raises(ValueError, match="no checkpoint was loaded"):
        _validate_l2sp_anchor(0.01, None)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
