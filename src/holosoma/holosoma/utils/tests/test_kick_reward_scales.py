"""Unit tests for the 4 per-skill reward-category scale resolvers (motion_tracking_reward_scale,
kick_recovery_posture_reward_scale, kick_safety_reward_scale, kick_alive_reward_scale), isolated
from the env-var-gated file-loading path by directly injecting the resolved per-category cache --
same pattern as test_shooting_curriculum.py (this module's own current_w_g)."""

from types import SimpleNamespace

import pytest
import torch

import holosoma.utils.kick_reward_scales as krs


@pytest.fixture(autouse=True)
def _reset_cache():
    krs._cached = None
    yield
    krs._cached = None


def _fake_env(motion_ids: torch.Tensor):
    motion_command = SimpleNamespace(motion_ids=motion_ids)
    command_manager = SimpleNamespace(get_state=lambda name: motion_command if name == "motion_command" else None)
    return SimpleNamespace(command_manager=command_manager)


def test_legacy_single_target_broadcasts_to_every_env():
    krs._cached = {cat: [1.0] for cat in krs._CATEGORIES}
    env = _fake_env(motion_ids=torch.tensor([0, 1, 2, 0]))  # varying motion_ids on purpose
    assert torch.allclose(krs.motion_tracking_reward_scale(env), torch.ones(4))
    assert torch.allclose(krs.kick_alive_reward_scale(env), torch.ones(4))


def test_n_skill_mode_gathers_correct_per_env_target_per_category():
    krs._cached = {
        "motion_tracking": [1.0, 1.5],
        "root_tracking": [1.0, 0.1],
        "recovery_tracking": [0.2, 0.3],
        "kick_recovery_posture": [0.5, 1.0],
        "kick_safety": [2.0, 1.0],
        "kick_alive": [1.0, 0.0],
    }
    env = _fake_env(motion_ids=torch.tensor([0, 1, 1, 0]))
    assert torch.allclose(krs.motion_tracking_reward_scale(env), torch.tensor([1.0, 1.5, 1.5, 1.0]))
    assert torch.allclose(krs.root_tracking_reward_scale(env), torch.tensor([1.0, 0.1, 0.1, 1.0]))
    assert torch.allclose(krs.recovery_tracking_scale(env), torch.tensor([0.2, 0.3, 0.3, 0.2]))
    assert torch.allclose(krs.kick_recovery_posture_reward_scale(env), torch.tensor([0.5, 1.0, 1.0, 0.5]))
    assert torch.allclose(krs.kick_safety_reward_scale(env), torch.tensor([2.0, 1.0, 1.0, 2.0]))
    assert torch.allclose(krs.kick_alive_reward_scale(env), torch.tensor([1.0, 0.0, 0.0, 1.0]))


def test_categories_are_independent_not_accidentally_sharing_a_target_list():
    """Regression guard: each category must read its OWN list, not silently fall back to
    whichever category happened to be resolved/cached first."""
    krs._cached = {
        "motion_tracking": [3.0],
        "root_tracking": [3.25],
        "recovery_tracking": [3.5],
        "kick_recovery_posture": [4.0],
        "kick_safety": [5.0],
        "kick_alive": [6.0],
    }
    env = _fake_env(motion_ids=torch.tensor([0]))
    assert krs.motion_tracking_reward_scale(env).item() == 3.0
    assert krs.root_tracking_reward_scale(env).item() == 3.25
    assert krs.recovery_tracking_scale(env).item() == 3.5
    assert krs.kick_recovery_posture_reward_scale(env).item() == 4.0
    assert krs.kick_safety_reward_scale(env).item() == 5.0
    assert krs.kick_alive_reward_scale(env).item() == 6.0


def test_no_ramp_or_hold_full_value_from_the_first_call():
    """Unlike shooting_reward_scale, these have no ramp/hold schedule -- the configured value
    applies immediately and unconditionally, with no dependence on any step counter."""
    krs._cached = {"motion_tracking": [0.7], "kick_recovery_posture": [1.0], "kick_safety": [1.0], "kick_alive": [1.0]}
    env = _fake_env(motion_ids=torch.tensor([0, 0, 0]))
    assert torch.allclose(krs.motion_tracking_reward_scale(env), torch.full((3,), 0.7))
