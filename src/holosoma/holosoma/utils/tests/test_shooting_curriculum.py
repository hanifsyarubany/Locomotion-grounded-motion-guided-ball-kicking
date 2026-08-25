"""Unit tests for current_w_g's per-env gather + ramp/hold math, isolated from the env-var-gated
file-loading path (config_types.multi_skill/simulator) by directly injecting the resolved
(targets, ramp_iters, hold_iters) cache -- that loading path is separately covered by the
config_values integration import checks (N-skill mode ON/OFF both import cleanly, see
config_values/unified/g1/reward.py's _shooting_terms weight checks)."""

from types import SimpleNamespace

import pytest
import torch

import holosoma.utils.shooting_curriculum as sc


@pytest.fixture(autouse=True)
def _reset_cache():
    """current_w_g/ramp_progress cache their resolved config in a module-level global, by design
    (documented as set-before-process-start, never mutated mid-run) -- reset it around each test
    so tests can inject different (targets, ramp_iters, hold_iters) tuples independently."""
    sc._cached = None
    yield
    sc._cached = None


def _fake_env(motion_ids: torch.Tensor, common_step_counter: int = 10_000_000):
    motion_command = SimpleNamespace(motion_ids=motion_ids)
    command_manager = SimpleNamespace(get_state=lambda name: motion_command if name == "motion_command" else None)
    return SimpleNamespace(command_manager=command_manager, common_step_counter=common_step_counter)


def test_legacy_single_target_broadcasts_to_every_env_regardless_of_motion_id():
    """Legacy mode: targets has exactly one entry. Every env must read that SAME value, even if
    motion_ids somehow varies (e.g. motion_dir with multiple untagged clips) -- reproduces the old
    single-float behavior exactly, just as a same-valued tensor."""
    sc._cached = ([0.8], 0, 0)  # instant step, single target 0.8
    env = _fake_env(motion_ids=torch.tensor([0, 1, 2, 0]))  # varying motion_ids on purpose
    w_g = sc.current_w_g(env)
    assert torch.allclose(w_g, torch.full((4,), 0.8))


def test_n_skill_mode_gathers_correct_per_env_target():
    """N-skill mode: 3 targets, one per skill. Different envs' motion_ids must read their OWN
    skill's target, not a shared/averaged/wrong one."""
    sc._cached = ([0.1, 0.0, 0.8], 0, 0)  # skill 0 -> 0.1, skill 1 -> 0.0 (Stage B), skill 2 -> 0.8
    env = _fake_env(motion_ids=torch.tensor([0, 1, 2, 1, 0]))
    w_g = sc.current_w_g(env)
    assert torch.allclose(w_g, torch.tensor([0.1, 0.0, 0.8, 0.0, 0.1]))


def test_ramp_schedule_applies_uniformly_across_different_per_skill_targets():
    """The ramp SCHEDULE (iteration-based progress) is shared/global, not per-skill -- at 50% of
    the way through the ramp, EVERY skill's contribution should be scaled by the same 0.5 factor,
    even though their targets differ."""
    sc._cached = ([0.2, 1.0], 1000, 0)  # ramp over 1000 steps, no hold
    env = _fake_env(motion_ids=torch.tensor([0, 1]), common_step_counter=500)  # halfway through ramp
    w_g = sc.current_w_g(env)
    assert torch.allclose(w_g, torch.tensor([0.1, 0.5]), atol=1e-5)  # each target * 0.5


def test_hold_then_ramp_reaches_zero_during_hold_regardless_of_target():
    sc._cached = ([0.3, 0.9], 500, 200)  # hold 200 steps, then ramp 500 steps
    env = _fake_env(motion_ids=torch.tensor([0, 1]), common_step_counter=100)  # still in the hold window
    w_g = sc.current_w_g(env)
    assert torch.allclose(w_g, torch.zeros(2))


def test_ramp_disabled_reaches_full_target_immediately():
    sc._cached = ([0.4, 0.6], 0, 0)  # ramp_iters <= 0 -> instant step
    env = _fake_env(motion_ids=torch.tensor([0, 1]), common_step_counter=0)
    w_g = sc.current_w_g(env)
    assert torch.allclose(w_g, torch.tensor([0.4, 0.6]))


def test_ramp_progress_stays_a_plain_float_matching_the_shared_schedule():
    """ramp_progress itself is unchanged -- a scalar float, the shared schedule fraction, used
    directly at kick_safety.py's three call sites (floor + k * current_w_g(env), which is a
    DIFFERENT function -- this confirms ramp_progress's own return type didn't regress)."""
    sc._cached = ([0.5], 1000, 0)
    env = _fake_env(motion_ids=torch.tensor([0]), common_step_counter=250)
    progress = sc.ramp_progress(env)
    assert isinstance(progress, float)
    assert abs(progress - 0.25) < 1e-6
