"""Unit tests for ``kick_recovery_low_height_sustained`` (managers/termination/terms/wbt.py): the
post-kick recovery/hold REPLACEMENT for ``bad_tracking``, only ever wired in alongside
``bad_tracking_swing_only`` (see ``kick_recovery_termination_handoff`` in
``config_values/unified/g1/termination.py``).

Two things are load-bearing and each is covered below:
1. The phase+grace gate is folded INTO the sustained-counter's increment condition itself, NOT
   applied as a post-hoc mask on an otherwise phase-unaware counter -- a legitimate low
   single-support dip during swing must never contribute counter progress that carries over once
   recovery begins (this is exactly the bug an external mask would introduce; see the function's
   own docstring).
2. It behaves as a faithful port of locomotion's own ``base_height_below_threshold_sustained``
   (same min_height/consecutive_steps semantics) once active, including per-env independence and
   counter-attr isolation from sibling low-height terms.

Isolated via lightweight fakes (not a real MotionCommand/simulator -- constructing those needs a
live env), mirroring ``managers/reward/terms/tests/test_kick_recovery_gate.py``'s
``_FakeMotionCommand``/``_FakeCommandManager`` pattern (``in_kicking_phase`` computed with
production's own ``time_steps < stand_start_idx[motion_ids]`` expression, not a hardcoded bool)
and ``test_balance_potential.py``'s ``simulator.robot_root_states`` height mocking.
"""

from __future__ import annotations

import torch

from holosoma.managers.termination.terms import locomotion as loco_terms
from holosoma.managers.termination.terms import wbt as wbt_terms


class _FakeMotionCommand:
    def __init__(self, time_steps, motion_ids, stand_start_idx, has_ball=True):
        self.time_steps = time_steps
        self.motion_ids = motion_ids
        self.stand_start_idx = stand_start_idx
        self.has_ball = has_ball

    @property
    def in_kicking_phase(self) -> torch.Tensor:
        return self.time_steps < self.stand_start_idx[self.motion_ids]


class _FakeCommandManager:
    def __init__(self, motion_command):
        self._motion_command = motion_command

    def get_state(self, name):
        return self._motion_command if name == "motion_command" else None


class _FakeSim:
    def __init__(self, n: int, device: str):
        self.robot_root_states = torch.zeros(n, 13, device=device)


class _FakeEnv:
    def __init__(self, motion_command, n: int, device: str = "cpu"):
        self.command_manager = _FakeCommandManager(motion_command)
        self.simulator = _FakeSim(n, device)
        self.num_envs = n
        self.device = device


def _make_env(time_steps, motion_ids=None, stand_start_idx=(100,), heights=None, has_ball: bool = True) -> _FakeEnv:
    time_steps_t = torch.tensor(time_steps, dtype=torch.long)
    n = len(time_steps_t)
    motion_ids_t = torch.zeros(n, dtype=torch.long) if motion_ids is None else torch.tensor(motion_ids, dtype=torch.long)
    stand_start_idx_t = torch.tensor(stand_start_idx, dtype=torch.long)
    mc = _FakeMotionCommand(time_steps_t, motion_ids_t, stand_start_idx_t, has_ball=has_ball)
    env = _FakeEnv(mc, n=n)
    if heights is not None:
        env.simulator.robot_root_states[:, 2] = torch.tensor(heights, dtype=torch.float32)
    return env


def _step(env: _FakeEnv, time_step: int, height: float) -> None:
    """Advance the SAME fake env by one control step: overwrite time_steps and height in place
    (so the persisted counter_attr on env survives across calls, the same as production)."""
    motion_command = env.command_manager.get_state("motion_command")
    motion_command.time_steps.fill_(time_step)
    env.simulator.robot_root_states[:, 2] = height


def test_counter_does_not_start_before_grace_elapses():
    """Deep in recovery (in_kicking_phase already False) but still inside the grace window, with
    height held below min_height the WHOLE time -- must never terminate."""
    env = _make_env(time_steps=[101], stand_start_idx=(100,), heights=[0.30])
    for step in range(101, 110):  # 9 steps in, grace_steps=10 not yet elapsed
        _step(env, step, 0.30)
        result = wbt_terms.kick_recovery_low_height_sustained(
            env, min_height=0.70, consecutive_steps=3, grace_steps=10.0, counter_attr="_c"
        )
        assert not bool(result[0]), f"terminated prematurely at step {step}, before grace elapsed"


def test_counter_does_not_increment_during_swing_even_if_height_transiently_low():
    """Still swinging (in_kicking_phase True) with height below min_height for many consecutive
    steps -- must never terminate, and must not leave counter progress that would prematurely
    trip the check once recovery begins (the core correctness property: the gate is folded into
    the increment condition, not applied after)."""
    env = _make_env(time_steps=[0], stand_start_idx=(1000,), heights=[0.20])
    for step in range(0, 5):
        _step(env, step, 0.20)
        result = wbt_terms.kick_recovery_low_height_sustained(
            env, min_height=0.70, consecutive_steps=3, grace_steps=10.0, counter_attr="_c"
        )
        assert not bool(result[0])
    assert env._c.item() == 0, "counter must stay at 0 while in_kicking_phase is True"


def test_terminates_after_consecutive_steps_once_truly_low_post_grace():
    """Past grace, height held below threshold: False for the first consecutive_steps-1 calls,
    True on the consecutive_steps-th."""
    env = _make_env(time_steps=[110], stand_start_idx=(100,), heights=[0.30])
    results = []
    for step in range(110, 114):
        _step(env, step, 0.30)
        result = wbt_terms.kick_recovery_low_height_sustained(
            env, min_height=0.70, consecutive_steps=3, grace_steps=10.0, counter_attr="_c"
        )
        results.append(bool(result[0]))
    assert results == [False, False, True, True]


def test_counter_resets_when_height_recovers():
    """Below threshold for a couple of steps (not yet at consecutive_steps), then height goes
    back above threshold -- counter must drop back to 0, so a later dip must restart from
    scratch rather than resuming where it left off."""
    env = _make_env(time_steps=[110], stand_start_idx=(100,), heights=[0.30])
    _step(env, 110, 0.30)
    wbt_terms.kick_recovery_low_height_sustained(env, min_height=0.70, consecutive_steps=3, grace_steps=10.0, counter_attr="_c")
    _step(env, 111, 0.30)
    wbt_terms.kick_recovery_low_height_sustained(env, min_height=0.70, consecutive_steps=3, grace_steps=10.0, counter_attr="_c")
    assert env._c.item() == 2

    _step(env, 112, 0.90)  # recovers above threshold
    wbt_terms.kick_recovery_low_height_sustained(env, min_height=0.70, consecutive_steps=3, grace_steps=10.0, counter_attr="_c")
    assert env._c.item() == 0

    # a fresh dip must need the full consecutive_steps again, not resume from 2
    _step(env, 113, 0.30)
    result = wbt_terms.kick_recovery_low_height_sustained(env, min_height=0.70, consecutive_steps=3, grace_steps=10.0, counter_attr="_c")
    assert not bool(result[0])
    assert env._c.item() == 1


def test_per_env_counters_independent():
    """Two envs, one truly falling (post-grace, low height sustained) and one healthy (post-grace,
    height fine) -- only the falling env's result must be True."""
    env = _make_env(time_steps=[110, 110], stand_start_idx=(100,), heights=[0.30, 0.90])
    for step in range(110, 114):
        _step_multi(env, step, heights=[0.30, 0.90])
        result = wbt_terms.kick_recovery_low_height_sustained(
            env, min_height=0.70, consecutive_steps=3, grace_steps=10.0, counter_attr="_c"
        )
    assert bool(result[0]) is True
    assert bool(result[1]) is False


def _step_multi(env: _FakeEnv, time_step: int, heights: list[float]) -> None:
    motion_command = env.command_manager.get_state("motion_command")
    motion_command.time_steps.fill_(time_step)
    env.simulator.robot_root_states[:, 2] = torch.tensor(heights, dtype=torch.float32)


def test_no_motion_command_returns_all_false():
    """Defensive path: an env class with no motion_command at all must never crash -- returns an
    all-False BOOL tensor (dtype matters: TerminationManager.check() asserts torch.bool)."""

    class _NoMotionCommandEnv:
        def __init__(self):
            self.command_manager = _FakeCommandManager(None)
            self.simulator = _FakeSim(4, "cpu")
            self.num_envs = 4
            self.device = "cpu"

    result = wbt_terms.kick_recovery_low_height_sustained(_NoMotionCommandEnv())
    assert result.dtype == torch.bool
    assert not result.any()


def test_has_ball_false_returns_all_false():
    """Defensive path: has_ball=False must also return an inert all-False bool tensor, even at
    time_steps/height deep into what would otherwise be a genuine termination."""
    env = _make_env(time_steps=[500], stand_start_idx=(100,), heights=[0.10], has_ball=False)
    result = wbt_terms.kick_recovery_low_height_sustained(env, min_height=0.70, consecutive_steps=1, grace_steps=0.0)
    assert result.dtype == torch.bool
    assert not bool(result[0])


def test_grace_boundary_uses_stand_start_idx_not_swing_end():
    """Two envs on DIFFERENT skills with different recovery boundaries (skill 0 at 100, skill 1 at
    40), both read at the SAME absolute time_steps -- proves the grace window is measured from
    each env's OWN per-skill stand_start_idx, not a shared/hardcoded boundary."""
    # skill 0: boundary=100, at time_steps=105 -> only 5 steps into recovery, grace_steps=10 not
    # elapsed yet. skill 1: boundary=40, at time_steps=105 -> 65 steps into recovery, well past
    # grace. Both held at low height the whole time.
    env = _make_env(time_steps=[105, 105], motion_ids=[0, 1], stand_start_idx=(100, 40), heights=[0.30, 0.30])
    for step in range(100, 106):
        _step_multi(env, step, heights=[0.30, 0.30])
        result = wbt_terms.kick_recovery_low_height_sustained(
            env, min_height=0.70, consecutive_steps=3, grace_steps=10.0, counter_attr="_c"
        )
    assert bool(result[0]) is False, "skill 0 (boundary=100) should still be inside its own grace window"
    assert bool(result[1]) is True, "skill 1 (boundary=40) should already be past its own grace window"


# ============================================================================================
# 2026-08-15, Tier 3 Group B Wave 1: `enabled` param -- lets kick_recovery_termination_handoff's
# REGISTRATION (per-term: does this function even run at all) stay decoupled from its per-env
# EFFECT (does THIS env's own skill actually want it), via the ordinary params_per_skill gather.
# See config_values/unified/g1/termination.py's own comment above
# _kick_recovery_termination_handoff_active for why registration alone can't be per-skill.
# ============================================================================================


def test_enabled_true_default_is_bit_identical_to_before_the_param_existed():
    """enabled defaults to True -- every pre-existing test above already exercises this path
    implicitly (none of them pass `enabled`), this just makes the no-op explicit."""
    env = _make_env(time_steps=[110], stand_start_idx=(100,), heights=[0.30])
    for step in range(110, 114):
        _step(env, step, 0.30)
        result = wbt_terms.kick_recovery_low_height_sustained(
            env, min_height=0.70, consecutive_steps=3, grace_steps=10.0, counter_attr="_c", enabled=True
        )
    assert bool(result[0]) is True


def test_enabled_false_scalar_suppresses_an_otherwise_genuine_termination():
    """Same exact trajectory as test_terminates_after_consecutive_steps_once_truly_low_post_grace
    (which proves True on the 3rd call) -- enabled=False must keep it False throughout."""
    env = _make_env(time_steps=[110], stand_start_idx=(100,), heights=[0.30])
    results = []
    for step in range(110, 114):
        _step(env, step, 0.30)
        result = wbt_terms.kick_recovery_low_height_sustained(
            env, min_height=0.70, consecutive_steps=3, grace_steps=10.0, counter_attr="_c", enabled=False
        )
        results.append(bool(result[0]))
    assert results == [False, False, False, False]


def test_enabled_per_env_tensor_suppresses_only_the_disabled_env():
    """Two envs, identical sustained-low-height trajectory past grace -- env 0's skill has
    enabled=True (terminates), env 1's has enabled=False (never does), isolating the per-env
    params_per_skill gather TerminationManager performs before calling this function."""
    env = _make_env(time_steps=[110, 110], stand_start_idx=(100,), heights=[0.30, 0.30])
    enabled_mask = torch.tensor([1.0, 0.0])
    for step in range(110, 114):
        _step_multi(env, step, heights=[0.30, 0.30])
        result = wbt_terms.kick_recovery_low_height_sustained(
            env, min_height=0.70, consecutive_steps=3, grace_steps=10.0, counter_attr="_c", enabled=enabled_mask
        )
    assert bool(result[0]) is True
    assert bool(result[1]) is False


def test_enabled_all_true_tensor_matches_scalar_true():
    env = _make_env(time_steps=[110, 110], stand_start_idx=(100,), heights=[0.30, 0.30])
    enabled_mask = torch.tensor([1.0, 1.0])
    for step in range(110, 114):
        _step_multi(env, step, heights=[0.30, 0.30])
        result = wbt_terms.kick_recovery_low_height_sustained(
            env, min_height=0.70, consecutive_steps=3, grace_steps=10.0, counter_attr="_c", enabled=enabled_mask
        )
    assert bool(result[0]) is True
    assert bool(result[1]) is True


def test_counter_attr_isolated_from_other_low_height_terms():
    """Running this term and base_height_below_threshold_sustained against the SAME fake env with
    DIFFERENT counter_attr values must not let either corrupt the other's stored counter -- a
    regression guard for the documented 'shared counter_attr' hazard (every registered
    termination term's body runs every step for every env regardless of task_mode)."""
    env = _make_env(time_steps=[110], stand_start_idx=(100,), heights=[0.30])
    for step in range(110, 113):
        _step(env, step, 0.30)
        wbt_terms.kick_recovery_low_height_sustained(
            env, min_height=0.70, consecutive_steps=5, grace_steps=10.0, counter_attr="_kick_recovery_low_height_counter"
        )
        loco_terms.base_height_below_threshold_sustained(
            env, min_height=0.40, consecutive_steps=5, counter_attr="_kick_low_height_counter"
        )
    # both counters should read 3 (three low steps each), independently, not double-counted or
    # clobbered by the other call
    assert env._kick_recovery_low_height_counter.item() == 3
    assert env._kick_low_height_counter.item() == 3
