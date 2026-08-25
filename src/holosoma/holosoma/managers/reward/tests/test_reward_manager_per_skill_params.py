"""Unit tests for RewardManager's per-skill PARAM table (RewardTermCfg.params_per_skill,
2026-08-15, "Tier 2 Mechanism A" of "simultaneous per-skill task configs" -- see
config_values/unified/g1/reward.py's ``_per_skill_param``/``_apply_per_skill_reward_weight_
overrides`` for the full design). Sibling to test_reward_manager_per_skill_weight.py, same
SimpleNamespace-fake-env pattern, testing the PARAMS mechanism instead of WEIGHT.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg
from holosoma.managers.reward.base import RewardTermBase
from holosoma.managers.reward.manager import RewardManager


def _echo_deadzone_term(weight: float = 1.0, params=None, params_per_skill=None) -> RewardTermCfg:
    return RewardTermCfg(
        func="holosoma.managers.reward.tests.test_reward_manager_per_skill_params:_echo_deadzone",
        weight=weight,
        params=params or {"deadzone": 1.0},
        params_per_skill=params_per_skill,
    )


def _echo_deadzone(env, deadzone: float = 0.0) -> torch.Tensor:
    """A minimal stand-in for the real deadzone term shape (torch.clamp(x - deadzone, min=0)) --
    returns `deadzone` itself broadcast to [num_envs] so the test can read off exactly what value
    each env's call received."""
    return torch.full((env.num_envs,), float(deadzone)) if not torch.is_tensor(deadzone) else deadzone.clone()


def _fake_env(num_envs: int, skill_id: torch.Tensor | None = None):
    kwargs = dict(num_envs=num_envs, logger=None)
    if skill_id is not None:
        kwargs["skill_id"] = skill_id
    return SimpleNamespace(**kwargs)


def test_no_params_per_skill_is_byte_identical_to_before():
    cfg = RewardManagerCfg(terms={"t": _echo_deadzone_term(params={"deadzone": 0.02})})
    env = _fake_env(num_envs=3)
    manager = RewardManager(cfg, env, device="cpu")

    assert manager._params_per_skill_tensors == {}
    out = manager.compute(dt=1.0)
    assert torch.allclose(out, torch.full((3,), 0.02))


def test_per_skill_param_gathers_by_env_skill_id():
    cfg = RewardManagerCfg(
        terms={"t": _echo_deadzone_term(params={"deadzone": 0.02}, params_per_skill={"deadzone": [0.02, 0.05]})}
    )
    skill_id = torch.tensor([0, 1, 0, 1])
    env = _fake_env(num_envs=4, skill_id=skill_id)
    manager = RewardManager(cfg, env, device="cpu")

    out = manager.compute(dt=1.0)
    assert torch.allclose(out, torch.tensor([0.02, 0.05, 0.02, 0.05]))


def test_per_skill_param_leaves_other_params_untouched():
    """A term with multiple params, only ONE of which has a per-skill override, must pass the
    OTHER params through exactly as configured -- proves the override is a targeted dict update,
    not a full params replacement."""
    cfg = RewardManagerCfg(
        terms={
            "t": RewardTermCfg(
                func="holosoma.managers.reward.tests.test_reward_manager_per_skill_params:_echo_two_module_level",
                params={"deadzone": 0.02, "nominal_width": 0.24},
                params_per_skill={"deadzone": [0.02, 0.05]},
            )
        }
    )
    skill_id = torch.tensor([0, 1])
    env = _fake_env(num_envs=2, skill_id=skill_id)
    manager = RewardManager(cfg, env, device="cpu")

    out = manager.compute(dt=1.0)
    assert torch.allclose(out, torch.tensor([0.02, 0.05]))


def _echo_two_module_level(env, deadzone: float = 0.0, nominal_width: float = 99.0) -> torch.Tensor:
    assert nominal_width == 0.24, "untouched param must reach the function unmodified"
    return torch.full((env.num_envs,), float(deadzone)) if not torch.is_tensor(deadzone) else deadzone.clone()


def test_params_per_skill_construction_raises_without_env_skill_id():
    cfg = RewardManagerCfg(terms={"t": _echo_deadzone_term(params_per_skill={"deadzone": [0.02, 0.05]})})
    env = _fake_env(num_envs=4)  # no skill_id
    with pytest.raises(AttributeError, match="skill_id"):
        RewardManager(cfg, env, device="cpu")


class _StatefulEcho(RewardTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.deadzone = cfg.params.get("deadzone", 0.0)

    def __call__(self, env, **kwargs):
        return torch.full((env.num_envs,), float(self.deadzone))

    def reset(self, env_ids=None):
        pass


def test_stateful_term_with_params_per_skill_raises_at_construction():
    """A stateful RewardTermBase caches params in __init__ -- params_per_skill would silently
    never be consulted, so this must raise loudly at RewardManager construction instead."""
    cfg = RewardManagerCfg(
        terms={
            "t": RewardTermCfg(
                func="holosoma.managers.reward.tests.test_reward_manager_per_skill_params:_StatefulEcho",
                params={"deadzone": 0.02},
                params_per_skill={"deadzone": [0.02, 0.05]},
                weight=1.0,
            )
        }
    )
    env = _fake_env(num_envs=2, skill_id=torch.tensor([0, 1]))
    with pytest.raises(ValueError, match="STATEFUL"):
        RewardManager(cfg, env, device="cpu")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
