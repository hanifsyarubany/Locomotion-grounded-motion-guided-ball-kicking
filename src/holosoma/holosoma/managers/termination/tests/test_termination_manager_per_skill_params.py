"""Unit tests for TerminationManager's per-skill PARAM table (TerminationTermCfg.params_per_skill,
2026-08-15, "Tier 2 Mechanism A'" of "simultaneous per-skill task configs" -- sibling to
managers/reward/tests/test_reward_manager_per_skill_params.py's RewardManager coverage, see that
file's own docstring for the full design pointer).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from holosoma.config_types.termination import TerminationManagerCfg, TerminationTermCfg
from holosoma.managers.termination.base import TerminationTermBase
from holosoma.managers.termination.manager import TerminationManager


def _echo_deadzone_term(params=None, params_per_skill=None, is_timeout: bool = False) -> TerminationTermCfg:
    return TerminationTermCfg(
        func="holosoma.managers.termination.tests.test_termination_manager_per_skill_params:_echo_deadzone_bool",
        params=params or {"deadzone": 0.15},
        params_per_skill=params_per_skill,
        is_timeout=is_timeout,
    )


def _echo_deadzone_bool(env, deadzone: float = 0.0) -> torch.Tensor:
    """Stand-in for a real per-env boolean termination check: fires (True) for envs whose
    per-env `deadzone` value is above 0.3, exercising a tensor-valued deadzone the same way
    kick_recovery_drift_sustained's real `deadzone` param is used."""
    dz = deadzone if torch.is_tensor(deadzone) else torch.full((env.num_envs,), float(deadzone))
    return dz > 0.3


def _fake_env(num_envs: int, skill_id: torch.Tensor | None = None):
    kwargs = dict(num_envs=num_envs, logger=None)
    if skill_id is not None:
        kwargs["skill_id"] = skill_id
    return SimpleNamespace(**kwargs)


def test_no_params_per_skill_is_byte_identical_to_before():
    cfg = TerminationManagerCfg(terms={"t": _echo_deadzone_term(params={"deadzone": 0.15})})
    env = _fake_env(num_envs=3)
    manager = TerminationManager(cfg, env, device="cpu")

    assert manager._params_per_skill_tensors == {}
    reset_flags, timeout_flags = manager.check()
    assert not reset_flags.any()  # 0.15 is below the 0.3 fire threshold
    assert not timeout_flags.any()


def test_per_skill_param_gathers_by_env_skill_id():
    """skill 0's deadzone (0.15) stays below the fire threshold; skill 1's (0.55) fires."""
    cfg = TerminationManagerCfg(
        terms={"t": _echo_deadzone_term(params={"deadzone": 0.15}, params_per_skill={"deadzone": [0.15, 0.55]})}
    )
    skill_id = torch.tensor([0, 1, 0, 1])
    env = _fake_env(num_envs=4, skill_id=skill_id)
    manager = TerminationManager(cfg, env, device="cpu")

    reset_flags, _ = manager.check()
    assert torch.equal(reset_flags, torch.tensor([False, True, False, True]))


def test_params_per_skill_construction_raises_without_env_skill_id():
    cfg = TerminationManagerCfg(terms={"t": _echo_deadzone_term(params_per_skill={"deadzone": [0.15, 0.55]})})
    env = _fake_env(num_envs=4)  # no skill_id
    with pytest.raises(AttributeError, match="skill_id"):
        TerminationManager(cfg, env, device="cpu")


class _StatefulEchoTermination(TerminationTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.deadzone = cfg.params.get("deadzone", 0.0)

    def __call__(self, env, **kwargs):
        return torch.full((env.num_envs,), self.deadzone > 0.3)

    def reset(self, env_ids=None):
        pass


class _OptedInStatefulEcho(TerminationTermBase):
    """Mirrors BadTracking's opt-out shape: reads cfg.params_per_skill directly in __init__."""

    handles_params_per_skill = True

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        per_skill = cfg.params_per_skill or {}
        self._deadzone_per_skill = (
            torch.tensor(per_skill["deadzone"], dtype=torch.float32, device=env.device) if "deadzone" in per_skill else None
        )
        self.deadzone = cfg.params.get("deadzone", 0.0)

    def __call__(self, env, **kwargs):
        dz = self._deadzone_per_skill[env.skill_id] if self._deadzone_per_skill is not None else self.deadzone
        return dz > 0.3 if torch.is_tensor(dz) else torch.full((env.num_envs,), dz > 0.3)

    def reset(self, env_ids=None):
        pass


def test_opted_in_stateful_term_is_excluded_from_the_generic_mechanism_and_does_not_raise():
    """A stateful term whose CLASS sets handles_params_per_skill=True (BadTracking's own opt-out
    shape) must construct without raising, must NOT appear in the manager's generic
    _params_per_skill_tensors (nothing for the manager to do -- the class handles it itself), and
    must still produce correct per-env behavior via its own internal gather."""
    cfg = TerminationManagerCfg(
        terms={
            "t": TerminationTermCfg(
                func="holosoma.managers.termination.tests.test_termination_manager_per_skill_params:_OptedInStatefulEcho",
                params={"deadzone": 0.15},
                params_per_skill={"deadzone": [0.15, 0.55]},
            )
        }
    )
    env = SimpleNamespace(num_envs=4, logger=None, device="cpu", skill_id=torch.tensor([0, 1, 0, 1]))
    manager = TerminationManager(cfg, env, device="cpu")

    assert "t" not in manager._params_per_skill_tensors
    reset_flags, _ = manager.check()
    assert torch.equal(reset_flags, torch.tensor([False, True, False, True]))


def test_stateful_term_with_params_per_skill_raises_at_construction():
    cfg = TerminationManagerCfg(
        terms={
            "t": TerminationTermCfg(
                func=(
                    "holosoma.managers.termination.tests."
                    "test_termination_manager_per_skill_params:_StatefulEchoTermination"
                ),
                params={"deadzone": 0.15},
                params_per_skill={"deadzone": [0.15, 0.55]},
            )
        }
    )
    env = _fake_env(num_envs=2, skill_id=torch.tensor([0, 1]))
    with pytest.raises(ValueError, match="STATEFUL"):
        TerminationManager(cfg, env, device="cpu")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
