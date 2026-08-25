"""Unit tests for UnifiedManager._build_task_mode_partition's (N+1)-way categorical skill
assignment, isolated from the rest of UnifiedManager (which needs a full simulator/terrain/scene to
construct) by hand-building a bare instance with only the attributes this method touches.
"""

import torch

from holosoma.envs.unified.unified_manager import TaskMode, UnifiedManager


class _FakeTerrainState:
    def __init__(self, env_terrain_is_flat: torch.Tensor):
        self.env_terrain_is_flat = env_terrain_is_flat


class _FakeTerrainManager:
    def __init__(self, env_terrain_is_flat: torch.Tensor):
        self._state = _FakeTerrainState(env_terrain_is_flat)

    def get_state(self, name: str):
        assert name == "locomotion_terrain"
        return self._state


def _make_manager(num_envs: int, ratios: list[float], kick_probability: float = 0.5, all_flat: bool = True) -> UnifiedManager:
    m = object.__new__(UnifiedManager)
    m.num_envs = num_envs
    m.device = "cpu"
    m.terrain_manager = _FakeTerrainManager(torch.ones(num_envs, dtype=torch.bool) if all_flat else torch.zeros(num_envs, dtype=torch.bool))
    m._skill_motion_training_ratios = ratios
    m._kick_probability = kick_probability
    return m


def test_legacy_two_way_partition_unchanged_when_no_ratios():
    torch.manual_seed(0)
    m = _make_manager(num_envs=5000, ratios=[], kick_probability=0.6)
    m._build_task_mode_partition()

    assert m._task_mode_partition.dtype == torch.long
    kick_frac = (m._task_mode_partition == TaskMode.KICK).float().mean().item()
    assert abs(kick_frac - 0.6) < 0.03  # statistical, generous tolerance
    # legacy path: skill_id_partition is all zeros (unused)
    assert torch.all(m._skill_id_partition == 0)


def test_n_skill_categorical_partition_realized_fractions_match_ratios():
    torch.manual_seed(0)
    ratios = [0.3, 0.25]  # locomotion gets 0.45
    m = _make_manager(num_envs=20000, ratios=ratios)
    m._build_task_mode_partition()

    is_kick = m._task_mode_partition == TaskMode.KICK
    is_skill_0 = is_kick & (m._skill_id_partition == 0)
    is_skill_1 = is_kick & (m._skill_id_partition == 1)
    is_loco = m._task_mode_partition == TaskMode.LOCOMOTION

    assert abs(is_skill_0.float().mean().item() - 0.30) < 0.02
    assert abs(is_skill_1.float().mean().item() - 0.25) < 0.02
    assert abs(is_loco.float().mean().item() - 0.45) < 0.02
    # every env accounted for exactly once
    assert torch.all(is_skill_0.long() + is_skill_1.long() + is_loco.long() == 1)


def test_n_skill_partition_fixed_for_life_idempotent():
    """_build_task_mode_partition must be a no-op on a second call (matches the legacy path's own
    idempotency contract -- _init_buffers() runs more than once per env lifetime)."""
    torch.manual_seed(0)
    m = _make_manager(num_envs=1000, ratios=[0.4])
    m._build_task_mode_partition()
    first_partition = m._task_mode_partition.clone()
    first_skill_ids = m._skill_id_partition.clone()

    m._build_task_mode_partition()  # second call, should be a no-op
    assert torch.equal(m._task_mode_partition, first_partition)
    assert torch.equal(m._skill_id_partition, first_skill_ids)


def test_ineligible_envs_forced_to_locomotion_even_with_ratios_set():
    torch.manual_seed(0)
    m = _make_manager(num_envs=2000, ratios=[0.9], all_flat=False)  # no env is flat-terrain-eligible
    m._build_task_mode_partition()
    assert torch.all(m._task_mode_partition == TaskMode.LOCOMOTION)


def test_skill_id_within_range_for_multiple_skills():
    torch.manual_seed(1)
    ratios = [0.2, 0.2, 0.2]
    m = _make_manager(num_envs=5000, ratios=ratios)
    m._build_task_mode_partition()
    is_kick = m._task_mode_partition == TaskMode.KICK
    skill_ids_of_kick_envs = m._skill_id_partition[is_kick]
    assert torch.all(skill_ids_of_kick_envs >= 0)
    assert torch.all(skill_ids_of_kick_envs < len(ratios))
