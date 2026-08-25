"""Unit tests for shooting.py's OOD-spawn gate (_ood_gate_multiplier) and its application to ALL 7
shooting reward terms -- 2026-08-01 reversal of the original 2026-07-24 "leave the shooting reward
alone for OOD-spawn episodes" decision (see MotionConfig.ood_spawn_probability's own docstring for
the full account: the OOD draw isn't rejection-sampled away from the normal box, so a minority of
"OOD" attempts were still landing reachable and paying inconsistent partial reward).

Unlike the three phase gates (test_shooting_strike_gate.py), which partition 1/3/4 across the 8
terms, _ood_gate_multiplier applies uniformly to all 8 (including ball_contact_hit and
ball_approach_stance, both added the same day as this gate) -- OOD-spawn is a per-ATTEMPT property
(MotionCommand.is_ood_spawn), not a phase. Isolated the same way test_shooting_strike_gate.py
isolates its own gates: patch _tracker and current_w_g to known values, AND patch all three phase
gates to an unconditional 1.0 so this file's assertions are decisive about ONLY the OOD gate, not
an interaction with the others.

2026-08-04: ball_proximity/contact_orientation gained a ~has_kicked gate (shooting.py, same day),
opposite in sign to the has_kicked-dependence the other 6 already have (ball_velocity,
ball_contact_hit, and the 3 outcome terms' own latched state all need has_kicked=True to be
nonzero; see test_shooting_strike_gate.py's own 2026-08-04 note). One _FakeTracker.has_kicked value
can no longer make all 8 terms nonzero at once, so _ALL_EIGHT_TERMS below is split into
_PRE_CONTACT_TERMS (needs has_kicked=False) and _POST_CONTACT_OR_UNGATED_TERMS (needs
has_kicked=True, or doesn't care) wherever they're exercised together.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import torch

import holosoma.managers.reward.terms.shooting as s


def _fake_env(is_ood_spawn: torch.Tensor):
    motion_command = SimpleNamespace(is_ood_spawn=is_ood_spawn)
    command_manager = SimpleNamespace(get_state=lambda name: motion_command if name == "motion_command" else None)
    return SimpleNamespace(command_manager=command_manager)


def test_ood_gate_multiplier_is_complement_of_is_ood_spawn_as_float():
    env = _fake_env(torch.tensor([False, True, False, True]))
    out = s._ood_gate_multiplier(env)
    assert torch.equal(out, torch.tensor([1.0, 0.0, 1.0, 0.0]))


class _FakeTracker:
    """Verbatim copy of test_shooting_strike_gate.py's _FakeTracker -- no shared conftest.py in
    this tests dir, every file in it owns its own fakes, matching this project's existing
    convention. Fixed, nonzero values chosen so an UN-gated computation would be strictly positive
    for every env, isolating the assertion to whether the OOD gate actually zeroes it out."""

    def __init__(self, num_envs: int):
        self.ball_radius = 0.11
        self._foot_pos = torch.zeros(num_envs, 3)
        self.ball_pos_w = torch.zeros(num_envs, 3)
        self._foot_vel = torch.tensor([[1.0, 0.0]] * num_envs)
        self.ball_to_target_xy = torch.tensor([[1.0, 0.0]] * num_envs)
        self.ball_speed = torch.full((num_envs,), 2.0)
        # 2026-08-21: ball_velocity now reads the LATCHED PEAK speed, not the instantaneous
        # one (see shooting.py::ball_velocity's docstring). Same nonzero value so a zero
        # result in the tests below can still only come from a gate.
        self.max_ball_speed = torch.full((num_envs,), 2.0)
        self.ball_vel_xy = torch.tensor([[1.0, 0.0]] * num_envs)
        self.has_kicked = torch.ones(num_envs, dtype=torch.bool)
        self.min_target_dist = torch.full((num_envs,), 0.2)
        self.burst_active = torch.ones(num_envs, dtype=torch.bool)
        self.strike_stance_ball_b = torch.zeros(num_envs, 2)
        self._ball_b = torch.zeros(num_envs, 2)

    def kick_foot_contact_pos_w(self, env, contact_offset_m):
        return self._foot_pos

    def kick_foot_vel_xy(self, env):
        return self._foot_vel

    def ball_pos_heading_frame_xy(self, env):
        return self._ball_b


_ALL_EIGHT_TERMS = [
    s.ball_proximity,
    s.contact_orientation,
    s.ball_velocity,
    s.error_ball_to_target,
    s.predicted_error_ball_to_target,
    s.goal_success_burst,
    s.ball_contact_hit,
    s.ball_approach_stance,
]
# 2026-08-04: split by has_kicked requirement -- see this file's own module docstring.
_PRE_CONTACT_TERMS = [s.ball_proximity, s.contact_orientation]
_POST_CONTACT_OR_UNGATED_TERMS = [
    s.ball_velocity,
    s.error_ball_to_target,
    s.predicted_error_ball_to_target,
    s.goal_success_burst,
    s.ball_contact_hit,
    s.ball_approach_stance,
]


def _patched_gates(s_mod, num_envs):
    """All three phase gates forced open, so only the OOD gate can zero anything below."""
    return (
        patch.object(s_mod, "_strike_phase_multiplier", return_value=torch.ones(num_envs)),
        patch.object(s_mod, "_post_locomotion_multiplier", return_value=torch.ones(num_envs)),
        patch.object(s_mod, "_locomotion_approach_multiplier", return_value=torch.ones(num_envs)),
    )


def test_all_eight_terms_zero_only_for_ood_spawn_envs():
    """4 envs, alternating OOD/not -- proves the gate zeroes exactly (and only) the OOD-flagged
    envs across every one of the 8 terms, matching test_shooting_strike_gate.py's own
    "zero outside / nonzero inside" pattern shape. Run twice (2026-08-04): once per has_kicked
    value, since ball_proximity/contact_orientation now need the opposite value from the other 6
    to be nonzero at all -- see this file's own module docstring."""
    num_envs = 4
    env = _fake_env(torch.tensor([True, False, True, False]))
    strike_gate, post_gate, approach_gate = _patched_gates(s, num_envs)

    pre_contact_tracker = _FakeTracker(num_envs)
    pre_contact_tracker.has_kicked = torch.zeros(num_envs, dtype=torch.bool)
    with (
        patch.object(s, "_tracker", return_value=pre_contact_tracker),
        patch.object(s, "current_w_g", return_value=torch.ones(num_envs)),
        strike_gate,
        post_gate,
        approach_gate,
    ):
        for term in _PRE_CONTACT_TERMS:
            out = term(env)
            assert torch.equal(out == 0.0, torch.tensor([True, False, True, False])), (
                f"{term.__name__} not correctly gated on is_ood_spawn: {out}"
            )

    post_contact_tracker = _FakeTracker(num_envs)
    with (
        patch.object(s, "_tracker", return_value=post_contact_tracker),
        patch.object(s, "current_w_g", return_value=torch.ones(num_envs)),
        strike_gate,
        post_gate,
        approach_gate,
    ):
        for term in _POST_CONTACT_OR_UNGATED_TERMS:
            out = term(env)
            assert torch.equal(out == 0.0, torch.tensor([True, False, True, False])), (
                f"{term.__name__} not correctly gated on is_ood_spawn: {out}"
            )


def test_mixed_ood_gate_applied_per_env_independently():
    """Two envs, one OOD-spawn and one not -- proves the gate is elementwise, not a scalar
    short-circuit (same rationale as test_shooting_strike_gate.py's own 2-env mixed tests). Run
    twice (2026-08-04), once per has_kicked value -- see test_all_eight_terms_zero_only_for_ood_
    spawn_envs above."""
    env = _fake_env(torch.tensor([True, False]))
    strike_gate, post_gate, approach_gate = _patched_gates(s, 2)

    pre_contact_tracker = _FakeTracker(2)
    pre_contact_tracker.has_kicked = torch.zeros(2, dtype=torch.bool)
    with (
        patch.object(s, "_tracker", return_value=pre_contact_tracker),
        patch.object(s, "current_w_g", return_value=torch.ones(2)),
        strike_gate,
        post_gate,
        approach_gate,
    ):
        for term in _PRE_CONTACT_TERMS:
            out = term(env)
            assert out[0].item() == 0.0, f"{term.__name__} env 0 (OOD spawn) should be zero"
            assert out[1].item() > 0.0, f"{term.__name__} env 1 (normal spawn) should be nonzero"

    post_contact_tracker = _FakeTracker(2)
    with (
        patch.object(s, "_tracker", return_value=post_contact_tracker),
        patch.object(s, "current_w_g", return_value=torch.ones(2)),
        strike_gate,
        post_gate,
        approach_gate,
    ):
        for term in _POST_CONTACT_OR_UNGATED_TERMS:
            out = term(env)
            assert out[0].item() == 0.0, f"{term.__name__} env 0 (OOD spawn) should be zero"
            assert out[1].item() > 0.0, f"{term.__name__} env 1 (normal spawn) should be nonzero"
