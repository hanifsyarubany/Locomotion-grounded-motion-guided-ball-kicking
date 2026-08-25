"""Unit tests for mujoco_kick_interactive.py's _TerminalController, specifically the 2026-08-23
`a`/`angle` command that lets an interactive session change the commanded kick_aim_theta between
kicks without restarting the process or re-dragging the ball.

This module can be imported directly in a plain Python env (no `mujoco`/`robojudo` needed) --
those are only imported lazily inside `main()`, never at module level -- so this is a real,
importable unit test, not a robojudo-only manual check.

`_read_loop` is exercised directly (not via its background thread) by feeding it a fake stdin
(`io.StringIO`) -- deterministic, no real threading/timing involved.
"""

from __future__ import annotations

import io
from unittest.mock import patch

from holosoma.mujoco_kick_interactive import _TerminalController  # noqa: E402


def _feed(controller: _TerminalController, lines: str) -> None:
    """Runs `_read_loop` to completion against a fake stdin containing `lines` (one command per
    line) -- the loop returns on its own once stdin is exhausted (same as a real EOF'd pipe)."""
    with patch("sys.stdin", io.StringIO(lines)):
        controller._read_loop()


class TestAngleCommand:
    def test_a_sets_aim_theta_deg_when_enabled(self):
        c = _TerminalController(aim_enabled=True)
        _feed(c, "a 10\n")
        assert c.aim_theta_deg == 10.0

    def test_angle_spelled_out_also_works(self):
        c = _TerminalController(aim_enabled=True)
        _feed(c, "angle -5.5\n")
        assert c.aim_theta_deg == -5.5

    def test_case_insensitive_command_word(self):
        c = _TerminalController(aim_enabled=True)
        _feed(c, "A 7\n")
        assert c.aim_theta_deg == 7.0
        _feed(c, "ANGLE 8\n")
        assert c.aim_theta_deg == 8.0

    def test_value_persists_across_multiple_sets(self):
        c = _TerminalController(aim_enabled=True)
        _feed(c, "a 10\na -3\na 22.25\n")
        assert c.aim_theta_deg == 22.25

    def test_unparseable_value_is_rejected_and_does_not_change_state(self):
        c = _TerminalController(aim_enabled=True)
        c.aim_theta_deg = 5.0
        _feed(c, "a not-a-number\n")
        assert c.aim_theta_deg == 5.0  # unchanged

    def test_disabled_by_default_rejects_the_command_and_does_not_change_state(self):
        """The whole point of aim_enabled=False: 'a'/'angle' would silently do nothing useful
        anyway (no observation-patch getter is ever installed to read it), so it should be a
        clear, explicit rejection instead of a silent no-op that looks like it worked."""
        c = _TerminalController(aim_enabled=False)
        starting = c.aim_theta_deg
        _feed(c, "a 10\n")
        assert c.aim_theta_deg == starting

    def test_starting_value_can_be_set_before_start(self):
        """Mirrors how main() seeds controller.aim_theta_deg = args.kick_aim_theta_deg before
        calling .start() -- confirms this is a plain, freely-settable attribute, not something
        only _read_loop may touch."""
        c = _TerminalController(aim_enabled=True)
        c.aim_theta_deg = -12.0
        assert c.aim_theta_deg == -12.0


class TestExistingCommandsUnaffected:
    """Regression check: adding the a/angle branch must not change k/r/q/blank/unknown handling."""

    def test_kick_and_restart_and_quit_still_work(self):
        c = _TerminalController(aim_enabled=True)
        _feed(c, "k\nr\nq\n")
        assert c.kick_requested is True
        assert c.restart_requested is True
        assert c.quit_requested is True

    def test_long_form_kick_and_restart_still_work(self):
        c = _TerminalController(aim_enabled=False)
        _feed(c, "kick\nrestart\n")
        assert c.kick_requested is True
        assert c.restart_requested is True

    def test_blank_lines_are_silently_ignored(self):
        c = _TerminalController(aim_enabled=True)
        _feed(c, "\n   \n")
        assert c.kick_requested is False
        assert c.restart_requested is False
        assert c.aim_theta_deg == 0.0

    def test_quit_stops_the_loop_before_later_lines_are_processed(self):
        c = _TerminalController(aim_enabled=True)
        _feed(c, "q\na 99\n")
        assert c.quit_requested is True
        assert c.aim_theta_deg == 0.0  # never reached -- loop returned at 'q'
