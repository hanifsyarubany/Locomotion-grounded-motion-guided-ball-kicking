"""Unit tests for BaseSimulator's no-op default methods -- the "rich on one backend, no-op
elsewhere" idiom this codebase already uses for draw_debug_viz/prepare_manager_fields (2026-08-01
adds get_ball_foot_contact_pos_w to the same family, real override only on IsaacSim's simulator
subclass). Bare-instantiated via object.__new__ (bypasses __init__, which needs real config
objects) since these methods don't touch self at all -- same convention as
test_strike_stand_boundaries.py's bare MultiMotionLoader construction.
"""

from __future__ import annotations

from holosoma.simulator.base_simulator.base_simulator import BaseSimulator


def test_get_ball_foot_contact_pos_w_default_is_none():
    """The base class's default must return None regardless of side -- callers (shooting.py's
    _geometric_foot_contact) rely on None meaning "fall back to the geometric approximation",
    not an empty tensor or a raised exception. A future refactor that accidentally makes this
    required (e.g. NotImplementedError) would silently break MuJoCo/IsaacGym; this pins the
    no-op contract down."""
    sim = object.__new__(BaseSimulator)
    assert sim.get_ball_foot_contact_pos_w("left") is None
    assert sim.get_ball_foot_contact_pos_w("right") is None
