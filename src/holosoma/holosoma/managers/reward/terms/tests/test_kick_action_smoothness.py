"""Unit tests for ``KickActionSmoothness`` (managers/reward/terms/wbt.py) -- 2026-08-05, ported
from RoboNaldo (arXiv:2606.11092)'s ``action_smoothness``: penalizes the second derivative of
actions (action_diff2 = (action - prev_action) - (prev_action - prev_prev_action)).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import holosoma.managers.reward.terms.wbt as wbt
from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg
from holosoma.managers.reward.manager import RewardManager


class _FakeCfg:
    def __init__(self, **params):
        self.params = params
        self.weight = 1.0


def _fake_env(action, prev_action):
    action_manager = SimpleNamespace(action=action, prev_action=prev_action)
    return SimpleNamespace(action_manager=action_manager)


def _make_term():
    return wbt.KickActionSmoothness(_FakeCfg(), SimpleNamespace())


def test_reset_before_any_call_is_a_noop():
    term = _make_term()
    term.reset(torch.tensor([0]))  # must not raise -- _prev_prev_action is still None


def test_first_call_treats_prev_prev_as_prev_action_zero_jerk():
    """On the very first call, _prev_prev_action is lazily seeded from prev_action itself, so if
    action == prev_action (no change yet), the result must be exactly 0 -- not garbage from an
    uninitialized buffer."""
    term = _make_term()
    action = torch.zeros(2, 3)
    prev_action = torch.zeros(2, 3)
    env = _fake_env(action, prev_action)
    out = term(env)
    assert torch.allclose(out, torch.zeros(2))


def test_matches_formula_at_concrete_values():
    term = _make_term()
    # Seed _prev_prev_action = 0 via a first call with prev_action = 0.
    env1 = _fake_env(torch.zeros(1, 1), torch.zeros(1, 1))
    term(env1)  # _prev_prev_action now 0.0 (seeded), then updated to prev_action=0.0 after this call

    # Second call: action=3.0, prev_action=1.0 -- action_diff = 2.0, prev_prev_action=0.0 (from call 1)
    # action_diff2 = 2.0 - (1.0 - 0.0) = 1.0 -> reward = 1.0^2 = 1.0
    env2 = _fake_env(torch.tensor([[3.0]]), torch.tensor([[1.0]]))
    out = term(env2)
    assert torch.isclose(out, torch.tensor([1.0]))


def test_zero_at_constant_velocity_no_jerk():
    """A perfectly linear action ramp (constant first derivative, zero second derivative) must
    score exactly 0, regardless of the (nonzero) first-derivative magnitude."""
    term = _make_term()
    env1 = _fake_env(torch.tensor([[1.0]]), torch.tensor([[0.0]]))
    term(env1)  # _prev_prev_action seeded to 0.0, then set to 0.0 (prev_action of call1)
    env2 = _fake_env(torch.tensor([[2.0]]), torch.tensor([[1.0]]))
    out = term(env2)  # action_diff=1.0, prev_prev=0.0 -> action_diff2 = 1.0 - (1.0-0.0) = 0.0
    assert torch.isclose(out, torch.tensor([0.0]))


def test_clamped_at_10():
    term = _make_term()
    env1 = _fake_env(torch.zeros(1, 1), torch.zeros(1, 1))
    term(env1)
    env2 = _fake_env(torch.tensor([[100.0]]), torch.tensor([[0.0]]))
    out = term(env2)
    assert out.item() == 10.0


def test_non_negative():
    term = _make_term()
    env1 = _fake_env(torch.tensor([[5.0]]), torch.tensor([[0.0]]))
    term(env1)
    env2 = _fake_env(torch.tensor([[-3.0]]), torch.tensor([[-1.0]]))
    out = term(env2)
    assert out.item() >= 0.0


def test_per_env_independent():
    term = _make_term()
    env1 = _fake_env(torch.zeros(2, 1), torch.zeros(2, 1))
    term(env1)
    env2 = _fake_env(torch.tensor([[3.0], [0.0]]), torch.tensor([[1.0], [0.0]]))
    out = term(env2)
    assert out[0].item() > 0.0
    assert out[1].item() == 0.0


def test_reset_zeroes_only_the_targeted_envs():
    term = _make_term()
    env1 = _fake_env(torch.tensor([[5.0], [5.0]]), torch.tensor([[2.0], [2.0]]))
    term(env1)  # _prev_prev_action now [2.0, 2.0]
    term.reset(torch.tensor([0]))
    assert term._prev_prev_action[0].item() == 0.0
    assert term._prev_prev_action[1].item() == 2.0  # untouched


def test_weight_zero_is_never_instantiated_by_reward_manager():
    cfg = RewardManagerCfg(
        terms={
            "kick_action_smoothness": RewardTermCfg(
                func="holosoma.managers.reward.terms.wbt:KickActionSmoothness",
                params={},
                weight=0.0,
                task_mode="kick",
            )
        }
    )
    env = SimpleNamespace(num_envs=4, logger=None)
    manager = RewardManager(cfg, env, device="cpu")
    assert "kick_action_smoothness" not in manager._term_names
    assert "kick_action_smoothness" not in manager._term_instances
