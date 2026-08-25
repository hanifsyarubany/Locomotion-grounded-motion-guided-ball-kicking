"""Unit tests for ``KickFeetContactTime`` (managers/reward/terms/wbt.py) -- 2026-08-05, ported
from RoboNaldo (arXiv:2606.11092)'s ``feet_contact_time``, the inverse-tempo sibling of
``KickFeetAirTime`` (see its own tests, test_p0_regularization_terms.py, for the equivalent
touchdown-side coverage): rewards a foot for leaving contact SOON after landing.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import holosoma.managers.reward.terms.wbt as wbt
from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg
from holosoma.managers.reward.manager import RewardManager

_BODY_NAMES = ["pelvis", "left_ankle_roll_link", "right_ankle_roll_link", "torso_link"]
_LEFT_FOOT_IDX = _BODY_NAMES.index("left_ankle_roll_link")
_RIGHT_FOOT_IDX = _BODY_NAMES.index("right_ankle_roll_link")


class _FakeCfg:
    def __init__(self, **params):
        self.params = params
        self.weight = 1.0


def _zero_state(num_envs: int, history: int = 3):
    contact = torch.zeros(num_envs, history, len(_BODY_NAMES), 3)
    return contact


def _fake_env(num_envs: int, contact_force_xyz: torch.Tensor, device: str = "cpu"):
    simulator = SimpleNamespace(body_names=list(_BODY_NAMES), contact_forces_history=contact_force_xyz)
    return SimpleNamespace(num_envs=num_envs, device=device, dt=0.02, simulator=simulator)


def _make_term(contact_time_threshold: float = 0.5, contact_force_threshold: float = 1.0, num_envs: int = 1):
    contact = _zero_state(num_envs)
    env = _fake_env(num_envs, contact)
    cfg = _FakeCfg(
        foot_body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
        contact_time_threshold=contact_time_threshold,
        contact_force_threshold=contact_force_threshold,
    )
    return wbt.KickFeetContactTime(cfg, env), env


def test_bad_foot_name_raises():
    contact = _zero_state(1)
    env = _fake_env(1, contact)
    cfg = _FakeCfg(foot_body_names=["not_a_real_foot"])
    with pytest.raises(AssertionError):
        wbt.KickFeetContactTime(cfg, env)


def test_reset_zeroes_buffers():
    term, env = _make_term(num_envs=2)
    term._contact_time[:] = 5.0
    term._last_contact[:] = True
    term.reset(torch.tensor([0]))
    assert term._contact_time[0].sum().item() == 0.0
    assert not term._last_contact[0].any()
    assert term._contact_time[1].sum().item() == 10.0  # env 1 untouched


def test_zero_while_foot_stays_in_contact_no_liftoff_yet():
    term, env = _make_term(num_envs=1)
    for _ in range(5):
        env.simulator.contact_forces_history[0, 0, _LEFT_FOOT_IDX, 2] = 50.0
        out = term(env)
    assert out.item() == 0.0


def test_zero_while_foot_stays_airborne_no_contact_ever():
    term, env = _make_term(num_envs=1)
    out = None
    for _ in range(5):
        out = term(env)  # never sets any contact force
    assert out.item() == 0.0


def test_pays_on_liftoff_after_short_contact():
    """Foot in contact for 3 steps (0.06s at dt=0.02), well under threshold=0.5 -- liftoff must
    pay 1.0 for that foot."""
    term, env = _make_term(contact_time_threshold=0.5, num_envs=1)
    for _ in range(3):
        env.simulator.contact_forces_history[0, 0, _LEFT_FOOT_IDX, 2] = 50.0
        term(env)
    env.simulator.contact_forces_history[0, 0, _LEFT_FOOT_IDX, 2] = 0.0  # liftoff
    out = term(env)
    assert out.item() == 1.0


def test_no_pay_on_liftoff_after_long_contact():
    """Foot in contact for 30 steps (0.6s), OVER threshold=0.5 -- liftoff must NOT pay."""
    term, env = _make_term(contact_time_threshold=0.5, num_envs=1)
    for _ in range(30):
        env.simulator.contact_forces_history[0, 0, _LEFT_FOOT_IDX, 2] = 50.0
        term(env)
    env.simulator.contact_forces_history[0, 0, _LEFT_FOOT_IDX, 2] = 0.0
    out = term(env)
    assert out.item() == 0.0


def test_both_feet_independent_short_contact_sums():
    term, env = _make_term(contact_time_threshold=0.5, num_envs=1)
    for _ in range(3):
        env.simulator.contact_forces_history[0, 0, _LEFT_FOOT_IDX, 2] = 50.0
        env.simulator.contact_forces_history[0, 0, _RIGHT_FOOT_IDX, 2] = 50.0
        term(env)
    env.simulator.contact_forces_history[0, 0, _LEFT_FOOT_IDX, 2] = 0.0
    env.simulator.contact_forces_history[0, 0, _RIGHT_FOOT_IDX, 2] = 0.0
    out = term(env)
    assert out.item() == 2.0, "both feet lifted off after a short contact -- must sum to 2.0"


def test_per_env_independent():
    term, env = _make_term(contact_time_threshold=0.5, num_envs=2)
    for _ in range(3):
        env.simulator.contact_forces_history[0, 0, _LEFT_FOOT_IDX, 2] = 50.0
        env.simulator.contact_forces_history[1, 0, _LEFT_FOOT_IDX, 2] = 50.0
        term(env)
    env.simulator.contact_forces_history[0, 0, _LEFT_FOOT_IDX, 2] = 0.0  # env 0 lifts off
    # env 1 stays in contact
    out = term(env)
    assert out[0].item() == 1.0
    assert out[1].item() == 0.0


def test_weight_zero_is_never_instantiated_by_reward_manager():
    cfg = RewardManagerCfg(
        terms={
            "kick_feet_contact_time": RewardTermCfg(
                func="holosoma.managers.reward.terms.wbt:KickFeetContactTime",
                params={"foot_body_names": ["left_ankle_roll_link", "right_ankle_roll_link"]},
                weight=0.0,
                task_mode="kick",
            )
        }
    )
    env = SimpleNamespace(num_envs=4, logger=None)
    manager = RewardManager(cfg, env, device="cpu")
    assert "kick_feet_contact_time" not in manager._term_names
    assert "kick_feet_contact_time" not in manager._term_instances
