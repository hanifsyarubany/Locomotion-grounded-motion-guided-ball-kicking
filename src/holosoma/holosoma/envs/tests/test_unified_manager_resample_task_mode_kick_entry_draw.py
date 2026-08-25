"""Unit tests for UnifiedManager._resample_task_mode's locomotion->kick pending-entry draw (the
mid_episode_kick_entry_prob gate at the env_ids that just resampled into TaskMode.KICK -- see that
method's own comment block for the full rationale). Covers both the pre-existing scalar behavior
(previously untested in isolation -- only indirectly exercised via
test_unified_manager_kick_recovery_locomotion_flip.py's sentinel-clearing test, which always used
prob=0.0, the no-op path) and the new per-skill divergence added 2026-08-15 (Tier 3 Group B Wave
2, "simultaneous per-skill task configs").

Deterministic draws are obtained by using prob values at the 0.0/1.0 boundary (torch.rand's
half-open [0, 1) range guarantees `rand() < 1.0` is always True and `rand() < 0.0` is always
False) rather than seeding+asserting on a specific draw outcome -- avoids coupling the test to
torch's RNG implementation.

Isolated via a bare instance, same pattern as test_unified_manager_kick_recovery_locomotion_flip.py
and test_unified_manager_kick_pending.py.
"""

from __future__ import annotations

import torch

from holosoma.envs.unified.unified_manager import TaskMode, UnifiedManager


class _FakeMotionCommand:
    def __init__(self, num_envs: int):
        self.motion_ids = torch.zeros(num_envs, dtype=torch.long)
        self.ball_fixed_calls: list[torch.Tensor] = []

    def place_ball_at_reset_pending(self, env_ids: torch.Tensor) -> None:
        self.ball_fixed_calls.append(env_ids.clone())


class _FakeCommandManager:
    def __init__(self, motion_command):
        self._motion_command = motion_command

    def get_state(self, name: str):
        return self._motion_command if name == "motion_command" else None


def _make_manager(
    num_envs: int,
    *,
    partition: list[int],
    prob: float = 0.0,
    prob_per_skill: list[float] | None = None,
    skill_id_partition: list[int] | None = None,
    ball_fixed: bool = False,
) -> UnifiedManager:
    m = object.__new__(UnifiedManager)
    m.num_envs = num_envs
    m.device = "cpu"
    m.task_mode = torch.full((num_envs,), int(TaskMode.LOCOMOTION), dtype=torch.long)
    m._task_mode_partition = torch.tensor(partition, dtype=torch.long)
    m._skill_id_partition = (
        torch.tensor(skill_id_partition, dtype=torch.long)
        if skill_id_partition is not None
        else torch.zeros(num_envs, dtype=torch.long)
    )
    m.skill_id = torch.zeros(num_envs, dtype=torch.long)
    m._forced_task_mode = torch.full((num_envs,), -1, dtype=torch.long)
    m.is_evaluating = False
    m._post_flip_step = torch.full((num_envs,), -1, dtype=torch.long)
    m._pre_kick_step = torch.full((num_envs,), -1, dtype=torch.long)
    m._kick_pending = torch.zeros(num_envs, dtype=torch.bool)
    m._kick_pending_best_residual = torch.full((num_envs,), float("inf"))
    m._kick_pending_best_frame = torch.full((num_envs,), -1, dtype=torch.long)
    m._pre_kick_fallback_active = torch.zeros(num_envs, dtype=torch.bool)
    m._pre_kick_fallback_start_step = torch.full((num_envs,), -1, dtype=torch.long)
    m._pre_kick_fallback_start_vx = torch.zeros(num_envs)
    m._pre_kick_fallback_stale_ticks = torch.zeros(num_envs, dtype=torch.long)
    m._mid_episode_kick_entry_prob = prob
    m._mid_episode_kick_entry_prob_per_skill = (
        torch.tensor(prob_per_skill, dtype=torch.float32) if prob_per_skill is not None else None
    )
    m._mid_episode_kick_entry_ball_fixed = ball_fixed
    m._pin_recording_env_modes = lambda env_ids: None
    motion_command = _FakeMotionCommand(num_envs)
    m.command_manager = _FakeCommandManager(motion_command)
    return m


def test_scalar_prob_zero_is_an_exact_no_op():
    m = _make_manager(num_envs=1, partition=[int(TaskMode.KICK)], prob=0.0)

    m._resample_task_mode(torch.tensor([0]))

    assert m.task_mode[0] == TaskMode.KICK, "no entry-pending draw at all -- immediate teleport, unchanged"
    assert not bool(m._kick_pending[0])


def test_scalar_prob_one_always_defers_a_kick_selected_env():
    m = _make_manager(num_envs=1, partition=[int(TaskMode.KICK)], prob=1.0)

    m._resample_task_mode(torch.tensor([0]))

    assert m.task_mode[0] == TaskMode.LOCOMOTION, "hit -- overridden back to LOCOMOTION pending entry"
    assert bool(m._kick_pending[0])


def test_scalar_prob_one_never_touches_a_locomotion_selected_env():
    m = _make_manager(num_envs=1, partition=[int(TaskMode.LOCOMOTION)], prob=1.0)

    m._resample_task_mode(torch.tensor([0]))

    assert m.task_mode[0] == TaskMode.LOCOMOTION
    assert not bool(m._kick_pending[0]), "never kick-selected in the first place -- draw must not apply"


def test_per_skill_only_the_prob_one_skills_env_defers():
    """Two envs both partitioned into KICK this reset -- env0's skill has prob=1.0 (always
    defers), env1's skill has prob=0.0 (never does). Isolates the per-env prob_threshold gather."""
    m = _make_manager(
        num_envs=2,
        partition=[int(TaskMode.KICK), int(TaskMode.KICK)],
        prob_per_skill=[1.0, 0.0],
        skill_id_partition=[0, 1],
    )

    m._resample_task_mode(torch.tensor([0, 1]))

    assert m.task_mode[0] == TaskMode.LOCOMOTION, "env0's skill (prob=1.0) must defer"
    assert bool(m._kick_pending[0])
    assert m.task_mode[1] == TaskMode.KICK, "env1's skill (prob=0.0) must teleport immediately as before"
    assert not bool(m._kick_pending[1])


def test_per_skill_still_defers_even_though_the_global_scalar_is_zero():
    """The scalar prob=0.0 (the base/global default -- what a legacy run or a skill with no
    override would resolve to) must not gate the whole block anymore once a per-skill table
    exists: proves the outer `if` became an "any skill active" check, not a read of the scalar."""
    m = _make_manager(
        num_envs=1, partition=[int(TaskMode.KICK)], prob=0.0, prob_per_skill=[1.0], skill_id_partition=[0]
    )

    m._resample_task_mode(torch.tensor([0]))

    assert m.task_mode[0] == TaskMode.LOCOMOTION
    assert bool(m._kick_pending[0])


def test_per_skill_motion_ids_set_to_each_envs_own_gathered_skill_id():
    m = _make_manager(
        num_envs=2,
        partition=[int(TaskMode.KICK), int(TaskMode.KICK)],
        prob_per_skill=[1.0, 1.0],
        skill_id_partition=[1, 0],  # deliberately reversed vs env index, to prove no hardcoding
    )

    m._resample_task_mode(torch.tensor([0, 1]))

    motion_command = m.command_manager.get_state("motion_command")
    assert motion_command.motion_ids.tolist() == [1, 0]


def test_per_skill_ball_fixed_places_ball_only_for_deferred_envs():
    m = _make_manager(
        num_envs=2,
        partition=[int(TaskMode.KICK), int(TaskMode.KICK)],
        prob_per_skill=[1.0, 0.0],
        skill_id_partition=[0, 1],
        ball_fixed=True,
    )

    m._resample_task_mode(torch.tensor([0, 1]))

    motion_command = m.command_manager.get_state("motion_command")
    assert len(motion_command.ball_fixed_calls) == 1
    assert motion_command.ball_fixed_calls[0].tolist() == [0]


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
