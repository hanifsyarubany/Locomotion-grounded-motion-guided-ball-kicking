"""Unit tests for ``kick_recovery_drift_sustained`` (managers/termination/terms/wbt.py): the
2026-08-06, user-requested drift sibling to ``kick_recovery_low_height_sustained``, only ever
wired in alongside it under ``kick_recovery_termination_handoff``.

Three things are load-bearing and each is covered below:
1. The anchor is latched ONCE, at the swing->recovery transition, and held fixed -- not re-latched
   every step (which would make drift unmeasurable by construction), and not corrupted by
   continued robot movement after the initial latch.
2. The phase+grace gate is folded into the sustained-counter's increment condition itself, same
   discipline as the height check's own test file documents.
3. ``episode_length_buf <= 1`` forces anchor re-validation independent of phase, covering the edge
   case where an episode's RSI-sampled start lands already inside recovery/hold (skipping
   ``in_kicking_phase=True`` entirely) -- without this a stale anchor from a PRIOR episode would
   leak into the new one.

Isolated via lightweight fakes, mirroring ``test_kick_recovery_low_height.py``'s pattern (same
``_FakeMotionCommand``/``_FakeCommandManager`` shape), extended with a 2D XY root position and
``episode_length_buf``.
"""

from __future__ import annotations

import torch

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
        self.episode_length_buf = torch.full((n,), 100, dtype=torch.long)  # "well into the episode" default


def _make_env(
    time_steps, motion_ids=None, stand_start_idx=(100,), xy=None, episode_length_buf=None, has_ball: bool = True
) -> _FakeEnv:
    time_steps_t = torch.tensor(time_steps, dtype=torch.long)
    n = len(time_steps_t)
    motion_ids_t = torch.zeros(n, dtype=torch.long) if motion_ids is None else torch.tensor(motion_ids, dtype=torch.long)
    stand_start_idx_t = torch.tensor(stand_start_idx, dtype=torch.long)
    mc = _FakeMotionCommand(time_steps_t, motion_ids_t, stand_start_idx_t, has_ball=has_ball)
    env = _FakeEnv(mc, n=n)
    if xy is not None:
        env.simulator.robot_root_states[:, :2] = torch.tensor(xy, dtype=torch.float32)
    if episode_length_buf is not None:
        env.episode_length_buf = torch.tensor(episode_length_buf, dtype=torch.long)
    return env


def _step(env: _FakeEnv, time_step: int, xy, episode_length_buf: int | None = None) -> None:
    motion_command = env.command_manager.get_state("motion_command")
    motion_command.time_steps.fill_(time_step)
    env.simulator.robot_root_states[:, :2] = torch.tensor(xy, dtype=torch.float32)
    if episode_length_buf is not None:
        env.episode_length_buf.fill_(episode_length_buf)
    else:
        env.episode_length_buf += 1


def test_anchor_latches_at_swing_to_recovery_transition_not_every_step():
    """The anchor must be captured ONCE, at the first in_kicking_phase=False step, and then held
    fixed even as the robot keeps moving -- if it were re-latched every step, drift would always
    read ~0 and the check would never fire."""
    env = _make_env(time_steps=[99], stand_start_idx=(100,), xy=[[1.0, 1.0]], episode_length_buf=[50])
    # still swinging at t=99
    wbt_terms.kick_recovery_drift_sustained(env, deadzone=0.15, consecutive_steps=3, grace_steps=10.0)
    assert not bool(env._kick_recovery_drift_anchor_valid[0])

    # transition: t=100, in_kicking_phase flips False here -- anchor should latch to (2.0, 2.0)
    _step(env, 100, xy=[2.0, 2.0])
    wbt_terms.kick_recovery_drift_sustained(env, deadzone=0.15, consecutive_steps=3, grace_steps=10.0)
    assert bool(env._kick_recovery_drift_anchor_valid[0])
    assert torch.allclose(env._kick_recovery_drift_anchor_xy[0], torch.tensor([2.0, 2.0]))

    # robot keeps moving -- anchor must NOT follow
    _step(env, 101, xy=[2.5, 2.0])
    wbt_terms.kick_recovery_drift_sustained(env, deadzone=0.15, consecutive_steps=3, grace_steps=10.0)
    assert torch.allclose(env._kick_recovery_drift_anchor_xy[0], torch.tensor([2.0, 2.0])), "anchor must stay fixed"


def test_counter_does_not_start_before_grace_elapses():
    """Deep in recovery but still inside the grace window, drifted well past the deadzone the
    whole time -- must never terminate."""
    env = _make_env(time_steps=[100], stand_start_idx=(100,), xy=[[0.0, 0.0]], episode_length_buf=[50])
    wbt_terms.kick_recovery_drift_sustained(env, deadzone=0.15, consecutive_steps=3, grace_steps=10.0)
    for step in range(101, 109):  # up to 8 steps in, grace_steps=10 not yet elapsed
        _step(env, step, xy=[1.0, 0.0])  # 1.0m drift, way over 0.15 deadzone
        result = wbt_terms.kick_recovery_drift_sustained(env, deadzone=0.15, consecutive_steps=3, grace_steps=10.0)
        assert not bool(result[0]), f"terminated prematurely at step {step}, before grace elapsed"


def test_counter_does_not_increment_during_swing_even_if_position_far_from_future_anchor():
    """Still swinging: no anchor exists yet, so drift is meaningless and must never count toward
    termination, regardless of how far the robot moves during swing."""
    env = _make_env(time_steps=[0], stand_start_idx=(1000,), xy=[[0.0, 0.0]], episode_length_buf=[0])
    for step in range(0, 5):
        _step(env, step, xy=[float(step), 0.0])  # moving fast during swing
        result = wbt_terms.kick_recovery_drift_sustained(env, deadzone=0.15, consecutive_steps=3, grace_steps=10.0)
        assert not bool(result[0])
    assert env._kick_recovery_drift_counter.item() == 0


def test_terminates_after_consecutive_steps_once_truly_drifted_post_grace():
    """Past grace, drift held over the deadzone: False for the first consecutive_steps-1 calls,
    True on the consecutive_steps-th."""
    env = _make_env(time_steps=[99], stand_start_idx=(100,), xy=[[0.0, 0.0]], episode_length_buf=[50])
    _step(env, 100, xy=[0.0, 0.0])  # latch anchor at origin
    wbt_terms.kick_recovery_drift_sustained(env, deadzone=0.15, consecutive_steps=3, grace_steps=10.0)
    results = []
    for step in range(110, 114):
        _step(env, step, xy=[0.5, 0.0])  # 0.5m drift, over the 0.15 deadzone
        result = wbt_terms.kick_recovery_drift_sustained(env, deadzone=0.15, consecutive_steps=3, grace_steps=10.0)
        results.append(bool(result[0]))
    assert results == [False, False, True, True]


def test_counter_resets_when_drift_comes_back_under_deadzone():
    env = _make_env(time_steps=[99], stand_start_idx=(100,), xy=[[0.0, 0.0]], episode_length_buf=[50])
    _step(env, 100, xy=[0.0, 0.0])
    wbt_terms.kick_recovery_drift_sustained(env, deadzone=0.15, consecutive_steps=3, grace_steps=10.0)

    _step(env, 110, xy=[0.5, 0.0])
    wbt_terms.kick_recovery_drift_sustained(env, deadzone=0.15, consecutive_steps=3, grace_steps=10.0)
    _step(env, 111, xy=[0.5, 0.0])
    wbt_terms.kick_recovery_drift_sustained(env, deadzone=0.15, consecutive_steps=3, grace_steps=10.0)
    assert env._kick_recovery_drift_counter.item() == 2

    _step(env, 112, xy=[0.05, 0.0])  # back under deadzone
    wbt_terms.kick_recovery_drift_sustained(env, deadzone=0.15, consecutive_steps=3, grace_steps=10.0)
    assert env._kick_recovery_drift_counter.item() == 0

    _step(env, 113, xy=[0.5, 0.0])
    result = wbt_terms.kick_recovery_drift_sustained(env, deadzone=0.15, consecutive_steps=3, grace_steps=10.0)
    assert not bool(result[0])
    assert env._kick_recovery_drift_counter.item() == 1


def test_per_env_counters_and_anchors_independent():
    """Two envs, one drifting past the deadzone post-grace, one holding still -- only the drifting
    env's result must be True, and each env's anchor must be its own."""
    env = _make_env(time_steps=[99, 99], stand_start_idx=(100,), xy=[[0.0, 0.0], [5.0, 5.0]], episode_length_buf=[50, 50])
    _step_multi(env, 100, xys=[[0.0, 0.0], [5.0, 5.0]])
    wbt_terms.kick_recovery_drift_sustained(env, deadzone=0.15, consecutive_steps=3, grace_steps=10.0)
    for step in range(110, 114):
        _step_multi(env, step, xys=[[0.5, 0.0], [5.0, 5.0]])  # env0 drifts, env1 holds still
        result = wbt_terms.kick_recovery_drift_sustained(env, deadzone=0.15, consecutive_steps=3, grace_steps=10.0)
    assert bool(result[0]) is True
    assert bool(result[1]) is False


def _step_multi(env: _FakeEnv, time_step: int, xys: list[list[float]]) -> None:
    motion_command = env.command_manager.get_state("motion_command")
    motion_command.time_steps.fill_(time_step)
    env.simulator.robot_root_states[:, :2] = torch.tensor(xys, dtype=torch.float32)
    env.episode_length_buf += 1


# ============================================================================================
# 2026-08-15, Tier 3 Group B Wave 1: `enabled` param -- same registration/per-env-effect split as
# kick_recovery_low_height_sustained's own `enabled` tests (test_kick_recovery_low_height.py),
# see that file's section header comment for the full rationale.
# ============================================================================================


def test_enabled_false_scalar_suppresses_an_otherwise_genuine_termination():
    """Same trajectory as test_terminates_after_consecutive_steps_once_truly_drifted_post_grace
    (True on the 3rd call) -- enabled=False must keep it False throughout."""
    env = _make_env(time_steps=[99], stand_start_idx=(100,), xy=[[0.0, 0.0]], episode_length_buf=[50])
    _step(env, 100, xy=[0.0, 0.0])
    wbt_terms.kick_recovery_drift_sustained(env, deadzone=0.15, consecutive_steps=3, grace_steps=10.0, enabled=False)
    results = []
    for step in range(110, 114):
        _step(env, step, xy=[0.5, 0.0])
        result = wbt_terms.kick_recovery_drift_sustained(
            env, deadzone=0.15, consecutive_steps=3, grace_steps=10.0, enabled=False
        )
        results.append(bool(result[0]))
    assert results == [False, False, False, False]


def test_enabled_per_env_tensor_suppresses_only_the_disabled_env():
    """Two envs both drifting identically past the deadzone post-grace -- env 0's skill has
    enabled=True (terminates), env 1's has enabled=False (never does)."""
    env = _make_env(time_steps=[99, 99], stand_start_idx=(100,), xy=[[0.0, 0.0], [0.0, 0.0]], episode_length_buf=[50, 50])
    enabled_mask = torch.tensor([1.0, 0.0])
    _step_multi(env, 100, xys=[[0.0, 0.0], [0.0, 0.0]])
    wbt_terms.kick_recovery_drift_sustained(
        env, deadzone=0.15, consecutive_steps=3, grace_steps=10.0, enabled=enabled_mask
    )
    for step in range(110, 114):
        _step_multi(env, step, xys=[[0.5, 0.0], [0.5, 0.0]])
        result = wbt_terms.kick_recovery_drift_sustained(
            env, deadzone=0.15, consecutive_steps=3, grace_steps=10.0, enabled=enabled_mask
        )
    assert bool(result[0]) is True
    assert bool(result[1]) is False


def test_episode_start_forces_fresh_anchor_even_mid_recovery_rsi():
    """Edge case: an episode whose RSI-sampled start lands ALREADY inside recovery/hold (skipping
    in_kicking_phase=True entirely) must not reuse a stale anchor from a PRIOR episode -- it must
    latch fresh, from wherever this episode actually starts."""
    env = _make_env(time_steps=[100], stand_start_idx=(100,), xy=[[0.0, 0.0]], episode_length_buf=[50])
    _step(env, 100, xy=[0.0, 0.0])
    wbt_terms.kick_recovery_drift_sustained(env, deadzone=0.15, consecutive_steps=3, grace_steps=10.0)
    assert torch.allclose(env._kick_recovery_drift_anchor_xy[0], torch.tensor([0.0, 0.0]))

    # New episode: RSI starts DIRECTLY in recovery/hold (in_kicking_phase False from the start),
    # at a totally different position -- episode_length_buf reset to 0 signals a fresh episode.
    _step(env, 150, xy=[10.0, 10.0], episode_length_buf=0)
    wbt_terms.kick_recovery_drift_sustained(env, deadzone=0.15, consecutive_steps=3, grace_steps=10.0)
    assert torch.allclose(
        env._kick_recovery_drift_anchor_xy[0], torch.tensor([10.0, 10.0])
    ), "must re-latch to the NEW episode's own starting position, not reuse the stale anchor"


def test_no_motion_command_returns_all_false():
    class _NoMotionCommandEnv:
        def __init__(self):
            self.command_manager = _FakeCommandManager(None)
            self.simulator = _FakeSim(4, "cpu")
            self.num_envs = 4
            self.device = "cpu"
            self.episode_length_buf = torch.zeros(4, dtype=torch.long)

    result = wbt_terms.kick_recovery_drift_sustained(_NoMotionCommandEnv())
    assert result.dtype == torch.bool
    assert not result.any()


def test_has_ball_false_returns_all_false():
    env = _make_env(time_steps=[500], stand_start_idx=(100,), xy=[[100.0, 100.0]], episode_length_buf=[500], has_ball=False)
    result = wbt_terms.kick_recovery_drift_sustained(env, deadzone=0.15, consecutive_steps=1, grace_steps=0.0)
    assert result.dtype == torch.bool
    assert not bool(result[0])


def test_counter_and_anchor_attrs_isolated_from_low_height_sibling():
    """Running this term and kick_recovery_low_height_sustained against the SAME fake env must not
    let either corrupt the other's stored state -- both are always installed together in
    production."""
    env = _make_env(time_steps=[99], stand_start_idx=(100,), xy=[[0.0, 0.0]], episode_length_buf=[50])
    env.simulator.robot_root_states[:, 2] = 0.30  # low height too, for the sibling check
    # step 100 is the latch instant -- xy=[0,0] there becomes the anchor; every step AFTER that
    # uses xy=[0.5,0] so there's real, measurable drift relative to the anchor (not zero).
    _step(env, 100, xy=[0.0, 0.0])
    env.simulator.robot_root_states[:, 2] = 0.30
    wbt_terms.kick_recovery_drift_sustained(env, deadzone=0.15, consecutive_steps=5, grace_steps=10.0)
    wbt_terms.kick_recovery_low_height_sustained(env, min_height=0.70, consecutive_steps=5, grace_steps=10.0)
    for step in range(101, 113):
        _step(env, step, xy=[0.5, 0.0])
        env.simulator.robot_root_states[:, 2] = 0.30
        wbt_terms.kick_recovery_drift_sustained(env, deadzone=0.15, consecutive_steps=5, grace_steps=10.0)
        wbt_terms.kick_recovery_low_height_sustained(env, min_height=0.70, consecutive_steps=5, grace_steps=10.0)
    assert env._kick_recovery_drift_counter.item() == 3
    assert env._kick_recovery_low_height_counter.item() == 3
