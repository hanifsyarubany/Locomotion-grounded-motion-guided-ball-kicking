"""Unit tests for the FIX 2 (2026-08-12) post-flip-graced termination wrappers --
contact_forces_exceeded_post_flip_graced / base_height_below_threshold_sustained_post_flip_graced /
_post_flip_grace_active (managers/termination/terms/locomotion.py). See
MultiSkillConfig.post_flip_termination_grace_steps's own docstring for the full measured
rationale (contact alone was 600/643, 93.3%, of post-flip terminations in a live diagnostic).
"""

from __future__ import annotations

import torch

from holosoma.managers.termination.terms.locomotion import (
    _post_flip_grace_active,
    base_height_below_threshold_sustained_post_flip_graced,
    contact_forces_exceeded_post_flip_graced,
)


class _FakeSimulator:
    def __init__(self, contact_forces: torch.Tensor | None = None, root_states: torch.Tensor | None = None):
        self.contact_forces = contact_forces
        self.robot_root_states = root_states


class _FakeEnv:
    def __init__(
        self,
        num_envs: int,
        *,
        contact_forces: torch.Tensor | None = None,
        root_states: torch.Tensor | None = None,
        post_flip_step: list[int] | None = None,
        episode_length_buf: list[int] | None = None,
    ):
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self.simulator = _FakeSimulator(contact_forces, root_states)
        self.termination_contact_indices = torch.tensor([0])
        self._post_flip_step = torch.tensor(post_flip_step if post_flip_step is not None else [-1] * num_envs)
        self.episode_length_buf = torch.tensor(
            episode_length_buf if episode_length_buf is not None else [0] * num_envs
        )

    def post_flip_steps_since(self) -> torch.Tensor:
        is_post_flip = self._post_flip_step >= 0
        return torch.where(
            is_post_flip,
            (self.episode_length_buf - self._post_flip_step).clamp(min=0),
            torch.zeros_like(self.episode_length_buf),
        )


# ---------------------------------------------------------------------------
# _post_flip_grace_active
# ---------------------------------------------------------------------------


def test_grace_steps_zero_is_all_false_even_for_a_freshly_flipped_env():
    env = _FakeEnv(2, post_flip_step=[0, -1], episode_length_buf=[0, 5])
    mask = _post_flip_grace_active(env, grace_steps=0.0)
    assert mask.tolist() == [False, False]


def test_grace_active_for_a_recently_flipped_env_within_window():
    env = _FakeEnv(1, post_flip_step=[100], episode_length_buf=[110])  # 10 steps since flip
    mask = _post_flip_grace_active(env, grace_steps=50.0)
    assert mask.tolist() == [True]


def test_grace_inactive_once_past_the_window():
    env = _FakeEnv(1, post_flip_step=[100], episode_length_buf=[151])  # 51 steps since flip
    mask = _post_flip_grace_active(env, grace_steps=50.0)
    assert mask.tolist() == [False]


def test_grace_inactive_exactly_at_the_boundary_tick():
    env = _FakeEnv(1, post_flip_step=[100], episode_length_buf=[150])  # exactly 50 steps since
    mask = _post_flip_grace_active(env, grace_steps=50.0)
    assert mask.tolist() == [False], "strictly less-than -- the boundary tick itself is not graced"


def test_grace_never_active_for_an_env_that_never_flipped():
    env = _FakeEnv(1, post_flip_step=[-1], episode_length_buf=[5])
    mask = _post_flip_grace_active(env, grace_steps=1000.0)
    assert mask.tolist() == [False], "sentinel -1 must never be treated as post-flip regardless of grace width"


def test_grace_missing_post_flip_state_is_a_safe_all_false():
    """A non-UnifiedManager env (no _post_flip_step/post_flip_steps_since at all) must not crash --
    the field only ever means anything under UnifiedManager's flip mechanism."""

    class _BareEnv:
        num_envs = 2
        device = torch.device("cpu")

    mask = _post_flip_grace_active(_BareEnv(), grace_steps=50.0)
    assert mask.tolist() == [False, False]


# ---------------------------------------------------------------------------
# contact_forces_exceeded_post_flip_graced
# ---------------------------------------------------------------------------


def test_contact_graced_matches_ungraced_at_zero_grace_steps():
    # env0: force exceeds threshold; env1: does not.
    forces = torch.tensor([[[2.0, 0.0, 0.0]], [[0.5, 0.0, 0.0]]])
    env = _FakeEnv(2, contact_forces=forces)
    result = contact_forces_exceeded_post_flip_graced(env, force_threshold=1.0, post_flip_grace_steps=0.0)
    assert result.tolist() == [True, False]


def test_contact_graced_suppresses_a_freshly_flipped_envs_contact():
    forces = torch.tensor([[[5.0, 0.0, 0.0]]])  # would fire under the bare check
    env = _FakeEnv(1, contact_forces=forces, post_flip_step=[0], episode_length_buf=[5])
    result = contact_forces_exceeded_post_flip_graced(env, force_threshold=1.0, post_flip_grace_steps=50.0)
    assert result.tolist() == [False], "within grace window -- must be suppressed"


def test_contact_graced_still_fires_once_past_the_grace_window():
    forces = torch.tensor([[[5.0, 0.0, 0.0]]])
    env = _FakeEnv(1, contact_forces=forces, post_flip_step=[0], episode_length_buf=[60])
    result = contact_forces_exceeded_post_flip_graced(env, force_threshold=1.0, post_flip_grace_steps=50.0)
    assert result.tolist() == [True]


def test_contact_graced_unaffected_for_a_non_post_flip_env_regardless_of_grace():
    """A genuinely-locomotion env (never flipped, _post_flip_step stays -1) must be judged
    exactly like the bare check, at ANY grace_steps value -- grace only ever suppresses envs that
    actually flipped."""
    forces = torch.tensor([[[5.0, 0.0, 0.0]]])
    env = _FakeEnv(1, contact_forces=forces, post_flip_step=[-1], episode_length_buf=[1000])
    result = contact_forces_exceeded_post_flip_graced(env, force_threshold=1.0, post_flip_grace_steps=50.0)
    assert result.tolist() == [True]


# ---------------------------------------------------------------------------
# base_height_below_threshold_sustained_post_flip_graced
# ---------------------------------------------------------------------------


def test_low_height_graced_matches_ungraced_at_zero_grace_steps():
    root_states = torch.zeros(1, 13)
    root_states[:, 2] = 0.3  # below any reasonable min_height
    env = _FakeEnv(1, root_states=root_states)
    counter_attr = "_test_counter_a"
    for _ in range(10):
        result = base_height_below_threshold_sustained_post_flip_graced(
            env, min_height=0.7, consecutive_steps=10, counter_attr=counter_attr, post_flip_grace_steps=0.0
        )
    assert result.tolist() == [True], "10 consecutive low-height ticks must trip the sustained check"


def test_low_height_graced_counter_still_advances_during_grace_but_result_is_suppressed():
    """Grace suppresses the RESULT, not the underlying sustained-duration bookkeeping -- so a
    post-flip env's counter keeps accumulating exactly as it would without grace, it just can't
    terminate the episode until the grace window ends."""
    root_states = torch.zeros(1, 13)
    root_states[:, 2] = 0.3
    env = _FakeEnv(1, root_states=root_states, post_flip_step=[0], episode_length_buf=[5])
    counter_attr = "_test_counter_b"

    for _ in range(10):
        result = base_height_below_threshold_sustained_post_flip_graced(
            env, min_height=0.7, consecutive_steps=10, counter_attr=counter_attr, post_flip_grace_steps=50.0
        )
    assert result.tolist() == [False], "within grace -- suppressed despite 10 consecutive low ticks"
    assert int(getattr(env, counter_attr)[0]) == 10, "the sustained counter itself must still have advanced"


def test_low_height_graced_fires_once_the_already_sustained_counter_exits_grace():
    root_states = torch.zeros(1, 13)
    root_states[:, 2] = 0.3
    env = _FakeEnv(1, root_states=root_states, post_flip_step=[0], episode_length_buf=[5])
    counter_attr = "_test_counter_c"

    for _ in range(10):
        base_height_below_threshold_sustained_post_flip_graced(
            env, min_height=0.7, consecutive_steps=10, counter_attr=counter_attr, post_flip_grace_steps=50.0
        )
    env.episode_length_buf[:] = 60  # now past the 50-step grace window
    result = base_height_below_threshold_sustained_post_flip_graced(
        env, min_height=0.7, consecutive_steps=10, counter_attr=counter_attr, post_flip_grace_steps=50.0
    )
    assert result.tolist() == [True]
