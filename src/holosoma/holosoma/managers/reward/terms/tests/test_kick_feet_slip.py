"""Unit tests for ``KickFeetSlip`` (managers/reward/terms/wbt.py) -- ported from RoboNaldo
(arXiv:2606.11092)'s ``feet_slip``, 2026-08-04: penalize horizontal (XY) foot velocity ONLY while
that foot carries ground contact force above threshold, so swing feet are never penalized.

Isolated via a lightweight fake env exposing exactly what the term reads:
``simulator.body_names`` (for ``_get_index_of_a_in_b`` index resolution, same helper
``UndesiredContacts`` already uses), ``simulator.contact_forces_history``
(``[num_envs, history, num_bodies, 3]``, same tensor ``UndesiredContacts`` reads), and
``simulator._rigid_body_vel`` (world-frame per-body linear velocity, the sibling of
``_rigid_body_rot`` already used elsewhere in this project).
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

import holosoma.managers.reward.terms.wbt as wbt

_BODY_NAMES = ["pelvis", "left_ankle_roll_link", "right_ankle_roll_link", "torso_link"]
_LEFT_FOOT_IDX = _BODY_NAMES.index("left_ankle_roll_link")
_RIGHT_FOOT_IDX = _BODY_NAMES.index("right_ankle_roll_link")


class _FakeCfg:
    def __init__(self, **params):
        self.params = params
        self.weight = 1.0


def _make_env(
    num_envs: int,
    body_vel_xyz: torch.Tensor,  # [num_envs, num_bodies, 3]
    contact_force_xyz: torch.Tensor,  # [num_envs, history, num_bodies, 3]
    device: str = "cpu",
):
    simulator = SimpleNamespace(
        body_names=list(_BODY_NAMES),
        _rigid_body_vel=body_vel_xyz,
        contact_forces_history=contact_force_xyz,
    )
    return SimpleNamespace(num_envs=num_envs, device=device, simulator=simulator)


def _zero_state(num_envs: int, history: int = 3):
    body_vel = torch.zeros(num_envs, len(_BODY_NAMES), 3)
    contact = torch.zeros(num_envs, history, len(_BODY_NAMES), 3)
    return body_vel, contact


def _make_term(threshold: float = 1.0):
    cfg = _FakeCfg(foot_body_names=["left_ankle_roll_link", "right_ankle_roll_link"], threshold=threshold)
    env = _make_env(1, *_zero_state(1))  # placeholder env, only used at construction for index resolution
    return wbt.KickFeetSlip(cfg, env)


# ---------------------------------------------------------------- construction
def test_bad_foot_name_raises():
    cfg = _FakeCfg(foot_body_names=["not_a_real_foot"])
    env = _make_env(1, *_zero_state(1))
    try:
        wbt.KickFeetSlip(cfg, env)
        raised = False
    except AssertionError:
        raised = True
    assert raised


def test_reset_is_a_noop_and_callable():
    term = _make_term()
    assert term.reset() is None
    assert term.reset(env_ids=torch.tensor([0])) is None


def test_default_threshold_is_1_0():
    term = _make_term()
    assert term.threshold == 1.0


# ---------------------------------------------------------------- gating: contact required
def test_zero_when_no_contact_regardless_of_velocity():
    """A foot moving fast but NOT in contact (e.g. mid-swing) must contribute zero -- this is the
    entire point of the term: only penalize a PLANTED foot sliding, never the swing foot."""
    term = wbt.KickFeetSlip(
        _FakeCfg(foot_body_names=["left_ankle_roll_link", "right_ankle_roll_link"], threshold=1.0),
        _make_env(1, *_zero_state(1)),
    )
    body_vel, contact = _zero_state(1)
    body_vel[0, _LEFT_FOOT_IDX, 0] = 3.0  # fast horizontal motion
    body_vel[0, _RIGHT_FOOT_IDX, 1] = 2.0
    # contact stays all-zero -> below threshold -> not in contact
    env = _make_env(1, body_vel, contact)
    out = term(env)
    assert out.item() == 0.0


def test_zero_when_in_contact_but_not_moving():
    term = _make_term()
    body_vel, contact = _zero_state(1)
    contact[0, 0, _LEFT_FOOT_IDX, 2] = 50.0  # strong vertical contact force -> in contact
    env = _make_env(1, body_vel, contact)
    out = term(env)
    assert out.item() == 0.0


def test_nonzero_when_in_contact_and_sliding():
    term = _make_term(threshold=1.0)
    body_vel, contact = _zero_state(1)
    body_vel[0, _LEFT_FOOT_IDX, 0] = 0.6  # sliding horizontally
    contact[0, 0, _LEFT_FOOT_IDX, 2] = 50.0  # in contact
    env = _make_env(1, body_vel, contact)
    out = term(env)
    assert torch.isclose(out, torch.tensor(0.6**2), atol=1e-6)


def test_matches_formula_exactly_both_feet():
    """out = sum over feet of (vx^2+vy^2) if in_contact else 0 -- exact formula check with both
    feet contributing different amounts."""
    term = _make_term(threshold=1.0)
    body_vel, contact = _zero_state(1)
    body_vel[0, _LEFT_FOOT_IDX, 0] = 0.3
    body_vel[0, _LEFT_FOOT_IDX, 1] = 0.4  # left: speed^2 = 0.09+0.16 = 0.25
    body_vel[0, _RIGHT_FOOT_IDX, 0] = 1.0  # right: speed^2 = 1.0
    contact[0, 0, _LEFT_FOOT_IDX, 2] = 50.0
    contact[0, 0, _RIGHT_FOOT_IDX, 2] = 50.0
    env = _make_env(1, body_vel, contact)
    out = term(env)
    assert torch.isclose(out, torch.tensor(0.25 + 1.0), atol=1e-6)


def test_z_velocity_is_ignored():
    """Only horizontal (XY) slip counts -- vertical foot velocity (e.g. lifting off, landing)
    must not contribute, matching RoboNaldo's own XY-only formula."""
    term = _make_term()
    body_vel, contact = _zero_state(1)
    body_vel[0, _LEFT_FOOT_IDX, 2] = 5.0  # large Z velocity, XY untouched
    contact[0, 0, _LEFT_FOOT_IDX, 2] = 50.0
    env = _make_env(1, body_vel, contact)
    out = term(env)
    assert out.item() == 0.0


def test_contact_detected_from_any_history_frame():
    """Contact is 'was this foot in contact at ANY point in the history window', not just the
    latest frame -- mirrors UndesiredContacts's own max-over-history check."""
    term = _make_term(threshold=1.0)
    body_vel, contact = _zero_state(1, history=3)
    body_vel[0, _LEFT_FOOT_IDX, 0] = 0.5
    contact[0, 1, _LEFT_FOOT_IDX, 2] = 50.0  # contact in the MIDDLE history frame only
    env = _make_env(1, body_vel, contact)
    out = term(env)
    assert out.item() > 0.0


def test_threshold_is_exclusive_not_inclusive():
    """Contact force magnitude exactly AT threshold must NOT count as contact (production uses
    strict '>', matching UndesiredContacts's own convention)."""
    term = _make_term(threshold=1.0)
    body_vel, contact = _zero_state(1)
    body_vel[0, _LEFT_FOOT_IDX, 0] = 0.5
    contact[0, 0, _LEFT_FOOT_IDX, 2] = 1.0  # exactly at threshold
    env = _make_env(1, body_vel, contact)
    out = term(env)
    assert out.item() == 0.0


# ---------------------------------------------------------------- per-env independence
def test_per_env_independent():
    term = _make_term(threshold=1.0)
    body_vel, contact = _zero_state(2)
    body_vel[0, _LEFT_FOOT_IDX, 0] = 0.7
    contact[0, 0, _LEFT_FOOT_IDX, 2] = 50.0  # env 0: sliding + in contact
    # env 1: left blank (no contact, no velocity)
    env = _make_env(2, body_vel, contact)
    out = term(env)
    assert out[0].item() > 0.0
    assert out[1].item() == 0.0
