"""Unit tests for RewardManager's per-skill weight table (RewardTermCfg.weight_per_skill,
2026-08-15, "simultaneous per-skill task configs" -- see config_values/unified/g1/reward.py's
``_apply_per_skill_reward_weight_overrides`` for the full design).

Same SimpleNamespace-fake-env / RewardManagerCfg construction pattern as
managers/reward/terms/tests/test_kick_action_smoothness.py's own
``test_weight_zero_is_never_instantiated_by_reward_manager``, applied here to the manager itself
rather than to one term.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg
from holosoma.managers.reward.manager import RewardManager


def _stateless_term(weight: float = 1.0, weight_per_skill=None, task_mode=None) -> RewardTermCfg:
    return RewardTermCfg(
        func="holosoma.managers.reward.tests.test_reward_manager_per_skill_weight:_ones",
        weight=weight,
        weight_per_skill=weight_per_skill,
        task_mode=task_mode,
    )


def _ones(env) -> torch.Tensor:
    return torch.ones(env.num_envs)


def _fake_env(num_envs: int, skill_id: torch.Tensor | None = None):
    kwargs = dict(num_envs=num_envs, logger=None)
    if skill_id is not None:
        kwargs["skill_id"] = skill_id
    return SimpleNamespace(**kwargs)


def test_no_weight_per_skill_is_byte_identical_to_before(monkeypatch):
    """Default (weight_per_skill=None) path: plain scalar weight, no tensor built, no gather --
    must be indistinguishable from before this mechanism existed."""
    cfg = RewardManagerCfg(terms={"t": _stateless_term(weight=3.0)})
    env = _fake_env(num_envs=4)
    manager = RewardManager(cfg, env, device="cpu")

    assert manager._weight_per_skill_tensors == {}
    out = manager.compute(dt=0.02)
    assert torch.allclose(out, torch.full((4,), 3.0 * 0.02))


def test_per_skill_weight_gathers_by_env_skill_id():
    cfg = RewardManagerCfg(terms={"t": _stateless_term(weight_per_skill=[4.0, 0.0])})
    skill_id = torch.tensor([0, 1, 0, 1])
    env = _fake_env(num_envs=4, skill_id=skill_id)
    manager = RewardManager(cfg, env, device="cpu")

    out = manager.compute(dt=1.0)
    assert torch.allclose(out, torch.tensor([4.0, 0.0, 4.0, 0.0]))


def test_per_skill_weight_construction_raises_without_env_skill_id():
    """A per-skill table requires env.skill_id to gather against -- must fail at RewardManager
    CONSTRUCTION time (clear, immediate), not with a confusing AttributeError deep in the first
    compute() call during training."""
    cfg = RewardManagerCfg(terms={"t": _stateless_term(weight_per_skill=[4.0, 0.0])})
    env = _fake_env(num_envs=4)  # no skill_id
    with pytest.raises(AttributeError, match="skill_id"):
        RewardManager(cfg, env, device="cpu")


def test_term_nonzero_for_one_skill_zero_for_another_is_still_instantiated():
    """A term that's zero for skill 0 but nonzero for skill 1 must NOT be skipped by the
    zero-weight init check -- that check must consider every entry in weight_per_skill, not just
    the representative scalar `weight` (which could itself be 0 if skill 0 happens to be the zero
    one, an ordering this must not depend on)."""
    cfg = RewardManagerCfg(
        terms={"t": _stateless_term(weight=0.0, weight_per_skill=[0.0, 5.0])}
    )
    skill_id = torch.tensor([0, 1])
    env = _fake_env(num_envs=2, skill_id=skill_id)
    manager = RewardManager(cfg, env, device="cpu")

    assert "t" in manager._term_names
    out = manager.compute(dt=1.0)
    assert torch.allclose(out, torch.tensor([0.0, 5.0]))


def test_term_zero_for_every_skill_is_still_skipped():
    cfg = RewardManagerCfg(terms={"t": _stateless_term(weight=0.0, weight_per_skill=[0.0, 0.0])})
    env = _fake_env(num_envs=2, skill_id=torch.tensor([0, 1]))
    manager = RewardManager(cfg, env, device="cpu")

    assert "t" not in manager._term_names


def test_task_mode_mask_and_per_skill_weight_compose():
    """Both mechanisms multiply into the same reward -- a term gated to task_mode="kick" AND
    carrying a per-skill weight must be zero for non-kick envs regardless of their skill_id, and
    correctly weighted for kick envs by their own skill."""

    def _mask(task_mode):
        assert task_mode == "kick"
        return torch.tensor([1.0, 1.0, 0.0, 0.0])  # envs 0,1 are "kick", 2,3 are not

    cfg = RewardManagerCfg(terms={"t": _stateless_term(weight_per_skill=[4.0, 0.0], task_mode="kick")})
    skill_id = torch.tensor([0, 1, 0, 1])
    env = SimpleNamespace(num_envs=4, logger=None, skill_id=skill_id, task_mode_mask=_mask)
    manager = RewardManager(cfg, env, device="cpu")

    out = manager.compute(dt=1.0)
    assert torch.allclose(out, torch.tensor([4.0, 0.0, 0.0, 0.0]))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
