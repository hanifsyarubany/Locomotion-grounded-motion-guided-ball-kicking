"""Regression test for the 2026-08-09 Stage D bug fix: _reset_buffers_callback's kick_ids/loco_ids
split must be derived from the PERMANENT _task_mode_partition, not the live self.task_mode.

Before the fix, an env whose task_mode flipped mid-episode (kick_recovery_locomotion_flip_enabled)
would read LOCOMOTION by the time a timeout reset reached this method, even though the episode was
a real kick attempt -- silently dropping it from _record_kick_episode_ends (losing
kick_topple_frac/kick_episode_length/kick_ball_hit_rate data for exactly the episodes Stage D
exists to measure) and folding it into the locomotion curriculum tracker instead, contaminating
average_episode_length with a hybrid kick+locomotion episode.

Same bare-instance isolation pattern as test_unified_manager_kick_episode_stats.py /
test_unified_manager_task_mode_partition.py -- _record_kick_episode_ends itself is already
thoroughly tested elsewhere, so it's stubbed here to a simple call-recorder rather than exercised
for real, keeping this test focused on the kick_ids/loco_ids SELECTION logic only.
"""

import torch

from holosoma.envs.unified.unified_manager import TaskMode, UnifiedManager


def _make_manager(num_envs: int) -> UnifiedManager:
    m = object.__new__(UnifiedManager)
    m.num_envs = num_envs
    m.device = "cpu"
    m.need_to_refresh_envs = torch.zeros(num_envs, dtype=torch.bool)
    m.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
    m.reset_buf = torch.zeros(num_envs, dtype=torch.long)
    m.time_out_buf = torch.zeros(num_envs, dtype=torch.bool)
    m._kick_ep_min_height = torch.zeros(num_envs)
    m._KICK_MIN_HEIGHT_SENTINEL = 10.0
    m._kick_ep_is_handoff = torch.zeros(num_envs, dtype=torch.bool)
    m._pending_episode_update_mask = torch.zeros(num_envs, dtype=torch.bool)

    recorded = {"kick_ids": None}

    def _fake_record_kick_episode_ends(kick_ids):
        recorded["kick_ids"] = kick_ids.clone()

    m._record_kick_episode_ends = _fake_record_kick_episode_ends
    m._recorded = recorded
    return m


def test_mid_episode_flipped_env_still_counted_as_a_kick_episode():
    """The exact bug scenario: env 0 was kick-partitioned but flipped to LOCOMOTION mid-episode
    (Stage D). Live task_mode reads LOCOMOTION at reset time; the permanent partition still reads
    KICK. It must land in kick_ids (and record_kick_episode_ends must be called with it), and must
    NOT land in the locomotion curriculum tracker (_pending_episode_update_mask)."""
    m = _make_manager(num_envs=1)
    env_ids = torch.tensor([0])
    m.task_mode = torch.tensor([TaskMode.LOCOMOTION])  # live: flipped mid-episode
    m._task_mode_partition = torch.tensor([TaskMode.KICK])  # permanent: still a kick env

    m._reset_buffers_callback(env_ids, target_buf=None)

    assert m._recorded["kick_ids"] is not None, "flipped env must still be folded into kick episode stats"
    assert m._recorded["kick_ids"].tolist() == [0]
    assert not bool(m._pending_episode_update_mask[0]), (
        "a kick episode (even one that flipped mid-way) must NOT contaminate the locomotion "
        "curriculum tracker"
    )


def test_never_flipped_kick_env_unaffected_by_the_fix():
    """Baseline: an ordinary kick env (partition == live task_mode == KICK, never flipped) must
    still be correctly recorded as a kick episode -- the fix must not change this case."""
    m = _make_manager(num_envs=1)
    env_ids = torch.tensor([0])
    m.task_mode = torch.tensor([TaskMode.KICK])
    m._task_mode_partition = torch.tensor([TaskMode.KICK])

    m._reset_buffers_callback(env_ids, target_buf=None)

    assert m._recorded["kick_ids"].tolist() == [0]
    assert not bool(m._pending_episode_update_mask[0])


def test_ordinary_locomotion_env_unaffected_by_the_fix():
    """Baseline: an ordinary locomotion env (partition == live task_mode == LOCOMOTION, v1 never
    flips a locomotion env) must still correctly land in the curriculum tracker, not kick_ids."""
    m = _make_manager(num_envs=1)
    env_ids = torch.tensor([0])
    m.task_mode = torch.tensor([TaskMode.LOCOMOTION])
    m._task_mode_partition = torch.tensor([TaskMode.LOCOMOTION])

    m._reset_buffers_callback(env_ids, target_buf=None)

    assert m._recorded["kick_ids"] is None, "a genuine locomotion episode must not be recorded as a kick episode"
    assert bool(m._pending_episode_update_mask[0])


def test_mixed_batch_partition_split_is_per_env_correct():
    m = _make_manager(num_envs=4)
    env_ids = torch.arange(4)
    # env0: flipped kick (live=LOCO, partition=KICK) -- must count as kick
    # env1: ordinary kick (live=KICK, partition=KICK) -- must count as kick
    # env2: ordinary locomotion (live=LOCO, partition=LOCO) -- must count as locomotion
    # env3: ordinary locomotion (live=LOCO, partition=LOCO) -- must count as locomotion
    m.task_mode = torch.tensor([TaskMode.LOCOMOTION, TaskMode.KICK, TaskMode.LOCOMOTION, TaskMode.LOCOMOTION])
    m._task_mode_partition = torch.tensor([TaskMode.KICK, TaskMode.KICK, TaskMode.LOCOMOTION, TaskMode.LOCOMOTION])

    m._reset_buffers_callback(env_ids, target_buf=None)

    assert sorted(m._recorded["kick_ids"].tolist()) == [0, 1]
    assert m._pending_episode_update_mask.tolist() == [False, False, True, True]
