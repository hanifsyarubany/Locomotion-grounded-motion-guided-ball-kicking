"""Unit tests for the 2026-08-13 pre-kick-graced termination wrappers --
base_height_below_threshold_sustained_pre_kick_graced / _pre_kick_grace_active
(managers/termination/terms/locomotion.py). Mirror of test_post_flip_graced_termination.py's own
coverage, opposite boundary: mid_episode_kick_entry_prob's locomotion->kick handoff, not
kick_recovery_locomotion_flip_enabled's kick->locomotion one. See
MultiSkillConfig.pre_kick_termination_grace_steps's own docstring for the full rationale.
"""

from __future__ import annotations

import torch

from holosoma.managers.termination.terms.locomotion import (
    _pre_kick_grace_active,
    base_height_below_threshold_sustained_pre_kick_graced,
)


class _FakeEnv:
    def __init__(
        self,
        num_envs: int,
        *,
        root_states: torch.Tensor | None = None,
        pre_kick_step: list[int] | None = None,
        episode_length_buf: list[int] | None = None,
    ):
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self.simulator = _FakeSimulator(root_states)
        self._pre_kick_step = torch.tensor(pre_kick_step if pre_kick_step is not None else [-1] * num_envs)
        self.episode_length_buf = torch.tensor(
            episode_length_buf if episode_length_buf is not None else [0] * num_envs
        )

    def pre_kick_steps_since(self) -> torch.Tensor:
        is_pre_kick = self._pre_kick_step >= 0
        return torch.where(
            is_pre_kick,
            (self.episode_length_buf - self._pre_kick_step).clamp(min=0),
            torch.zeros_like(self.episode_length_buf),
        )


class _FakeSimulator:
    def __init__(self, root_states: torch.Tensor | None = None):
        self.robot_root_states = root_states


# ---------------------------------------------------------------------------
# _pre_kick_grace_active
# ---------------------------------------------------------------------------


def test_grace_steps_zero_is_all_false_even_for_a_freshly_entered_env():
    env = _FakeEnv(2, pre_kick_step=[0, -1], episode_length_buf=[0, 5])
    mask = _pre_kick_grace_active(env, grace_steps=0.0)
    assert mask.tolist() == [False, False]


def test_grace_active_for_a_recently_entered_env_within_window():
    env = _FakeEnv(1, pre_kick_step=[100], episode_length_buf=[110])  # 10 steps since entry
    mask = _pre_kick_grace_active(env, grace_steps=50.0)
    assert mask.tolist() == [True]


def test_grace_inactive_once_past_the_window():
    env = _FakeEnv(1, pre_kick_step=[100], episode_length_buf=[151])  # 51 steps since entry
    mask = _pre_kick_grace_active(env, grace_steps=50.0)
    assert mask.tolist() == [False]


def test_grace_inactive_exactly_at_the_boundary_tick():
    env = _FakeEnv(1, pre_kick_step=[100], episode_length_buf=[150])  # exactly 50 steps since
    mask = _pre_kick_grace_active(env, grace_steps=50.0)
    assert mask.tolist() == [False], "strictly less-than -- the boundary tick itself is not graced"


def test_grace_never_active_for_an_env_that_never_had_a_mid_episode_entry():
    env = _FakeEnv(1, pre_kick_step=[-1], episode_length_buf=[5])
    mask = _pre_kick_grace_active(env, grace_steps=1000.0)
    assert mask.tolist() == [False], "sentinel -1 must never be treated as pre-kick regardless of grace width"


def test_grace_missing_pre_kick_state_is_a_safe_all_false():
    """A non-UnifiedManager env (no _pre_kick_step/pre_kick_steps_since at all) must not crash --
    the field only ever means anything under UnifiedManager's mid-episode entry mechanism."""

    class _BareEnv:
        num_envs = 2
        device = torch.device("cpu")

    mask = _pre_kick_grace_active(_BareEnv(), grace_steps=50.0)
    assert mask.tolist() == [False, False]


# ---------------------------------------------------------------------------
# base_height_below_threshold_sustained_pre_kick_graced
# ---------------------------------------------------------------------------


def test_low_height_graced_matches_ungraced_at_zero_grace_steps():
    root_states = torch.zeros(1, 13)
    root_states[:, 2] = 0.3  # below any reasonable min_height
    env = _FakeEnv(1, root_states=root_states)
    counter_attr = "_test_counter_a"
    for _ in range(10):
        result = base_height_below_threshold_sustained_pre_kick_graced(
            env, min_height=0.7, consecutive_steps=10, counter_attr=counter_attr, pre_kick_grace_steps=0.0
        )
    assert result.tolist() == [True], "10 consecutive low-height ticks must trip the sustained check"


def test_low_height_graced_counter_still_advances_during_grace_but_result_is_suppressed():
    """Grace suppresses the RESULT, not the underlying sustained-duration bookkeeping -- so a
    pre-kick env's counter keeps accumulating exactly as it would without grace, it just can't
    terminate the episode until the grace window ends."""
    root_states = torch.zeros(1, 13)
    root_states[:, 2] = 0.3
    env = _FakeEnv(1, root_states=root_states, pre_kick_step=[0], episode_length_buf=[5])
    counter_attr = "_test_counter_b"

    for _ in range(10):
        result = base_height_below_threshold_sustained_pre_kick_graced(
            env, min_height=0.7, consecutive_steps=10, counter_attr=counter_attr, pre_kick_grace_steps=50.0
        )
    assert result.tolist() == [False], "within grace -- suppressed despite 10 consecutive low ticks"
    assert int(getattr(env, counter_attr)[0]) == 10, "the sustained counter itself must still have advanced"


def test_low_height_graced_fires_once_the_already_sustained_counter_exits_grace():
    root_states = torch.zeros(1, 13)
    root_states[:, 2] = 0.3
    env = _FakeEnv(1, root_states=root_states, pre_kick_step=[0], episode_length_buf=[5])
    counter_attr = "_test_counter_c"

    for _ in range(10):
        base_height_below_threshold_sustained_pre_kick_graced(
            env, min_height=0.7, consecutive_steps=10, counter_attr=counter_attr, pre_kick_grace_steps=50.0
        )
    env.episode_length_buf[:] = 60  # now past the 50-step grace window
    result = base_height_below_threshold_sustained_pre_kick_graced(
        env, min_height=0.7, consecutive_steps=10, counter_attr=counter_attr, pre_kick_grace_steps=50.0
    )
    assert result.tolist() == [True]


def test_low_height_graced_unaffected_for_a_non_pre_kick_env_regardless_of_grace():
    """An ordinary teleport-at-reset kick env (never mid-episode-entered, _pre_kick_step stays -1)
    must be judged exactly like the bare check, at ANY grace_steps value -- grace only ever
    suppresses envs that actually had a mid-episode entry."""
    root_states = torch.zeros(1, 13)
    root_states[:, 2] = 0.3
    env = _FakeEnv(1, root_states=root_states, pre_kick_step=[-1], episode_length_buf=[1000])
    counter_attr = "_test_counter_d"
    for _ in range(10):
        result = base_height_below_threshold_sustained_pre_kick_graced(
            env, min_height=0.7, consecutive_steps=10, counter_attr=counter_attr, pre_kick_grace_steps=50.0
        )
    assert result.tolist() == [True]
