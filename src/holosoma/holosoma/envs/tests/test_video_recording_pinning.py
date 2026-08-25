"""Unit tests for UnifiedManager's per-skill wandb kick-video recording:
_setup_task_video_recording (which env(s)/skill_id(s) get dedicated to recording) and
_pin_recording_env_modes (forcing those envs' task_mode/skill_id every reset).

Isolated from the rest of UnifiedManager by hand-building a bare instance with only the
attributes these two methods touch, and stubbing IsaacSimVideoRecorder's import so this doesn't
need a real IsaacSim runtime -- these tests only check WHICH env/skill each recorder is bound to
and how pinning is applied, not the recorder's own rendering behavior.
"""

import sys
import types
from dataclasses import dataclass, field
from unittest.mock import patch

import torch

from holosoma.config_types.video import VideoConfig
from holosoma.envs.unified.unified_manager import TaskMode, UnifiedManager
from holosoma.utils.simulator_config import SimulatorType


class _FakeTerrainState:
    def __init__(self, env_terrain_is_flat: torch.Tensor):
        self.env_terrain_is_flat = env_terrain_is_flat


class _FakeTerrainManager:
    def __init__(self, env_terrain_is_flat: torch.Tensor):
        self._state = _FakeTerrainState(env_terrain_is_flat)

    def get_state(self, name: str):
        assert name == "locomotion_terrain"
        return self._state


@dataclass
class _FakeRecorder:
    config: VideoConfig
    simulator: object = None

    @property
    def enabled(self) -> bool:
        return self.config.enabled


class _FakeSimulator:
    def __init__(self, video_recorder: VideoConfig | None):
        self.video_recorder = _FakeRecorder(video_recorder) if video_recorder is not None else None
        self.extra_video_recorders: list = []

    def get_simulator_type(self):
        return SimulatorType.ISAACSIM

    def add_video_recorder(self, recorder) -> None:
        self.extra_video_recorders.append(recorder)


def _install_fake_isaacsim_video_recorder_module():
    """IsaacSimVideoRecorder is imported LOCALLY inside _setup_task_video_recording, and its real
    module pulls in isaaclab.utils.math (a live-IsaacSim-ish dependency) at import time -- stub it
    out so this test can run anywhere, same trick as patching any other heavy optional import."""

    @dataclass
    class _FakeIsaacSimVideoRecorder:
        config: VideoConfig
        simulator: object

    fake_module = types.ModuleType("holosoma.simulator.isaacsim.video_recorder")
    fake_module.IsaacSimVideoRecorder = _FakeIsaacSimVideoRecorder
    return fake_module


def _make_manager(num_envs: int, ratios: list[float], video_enabled: bool = True, all_flat: bool = True) -> UnifiedManager:
    m = object.__new__(UnifiedManager)
    m.num_envs = num_envs
    m.device = "cpu"
    m.terrain_manager = _FakeTerrainManager(torch.ones(num_envs, dtype=torch.bool) if all_flat else torch.zeros(num_envs, dtype=torch.bool))
    m._skill_motion_training_ratios = ratios
    m.task_mode = torch.zeros(num_envs, dtype=torch.long)
    m.skill_id = torch.zeros(num_envs, dtype=torch.long)
    primary_cfg = VideoConfig(enabled=video_enabled, record_env_id=0) if video_enabled else None
    m.simulator = _FakeSimulator(primary_cfg)
    return m


def _setup(m: UnifiedManager) -> None:
    with patch.dict(sys.modules, {"holosoma.simulator.isaacsim.video_recorder": _install_fake_isaacsim_video_recorder_module()}):
        m._setup_task_video_recording()


def test_legacy_single_skill_gets_exactly_one_kick_recorder_pinned_to_skill_zero():
    m = _make_manager(num_envs=100, ratios=[])
    _setup(m)

    assert len(m._recording_kick_env_skill_ids) == 1
    (kick_env_id, skill_id), = m._recording_kick_env_skill_ids.items()
    assert skill_id == 0
    assert kick_env_id != 0  # distinct from the primary (locomotion) recorder's env 0
    assert len(m.simulator.extra_video_recorders) == 1
    assert m.simulator.extra_video_recorders[0].config.wandb_key == "isaacsim_media/Training rollout - Kick"


def test_n_skill_mode_gets_one_recorder_per_skill_with_distinct_envs_and_skill_ids():
    m = _make_manager(num_envs=100, ratios=[0.3, 0.2, 0.1])
    _setup(m)

    assert len(m._recording_kick_env_skill_ids) == 3
    assert sorted(m._recording_kick_env_skill_ids.values()) == [0, 1, 2]
    # every recorder bound to a distinct env
    assert len(set(m._recording_kick_env_skill_ids.keys())) == 3
    assert 0 not in m._recording_kick_env_skill_ids  # env 0 stays the primary/locomotion recorder

    wandb_keys = sorted(r.config.wandb_key for r in m.simulator.extra_video_recorders)
    assert wandb_keys == [
        "isaacsim_media/Training rollout - Kick - Skill 1",
        "isaacsim_media/Training rollout - Kick - Skill 2",
        "isaacsim_media/Training rollout - Kick - Skill 3",
    ]


def test_n_skill_mode_falls_back_gracefully_when_fewer_flat_envs_than_skills():
    # only 2 non-primary envs are flat-eligible (env 0 excluded as primary, so candidates = {1})
    m = _make_manager(num_envs=2, ratios=[0.3, 0.2, 0.1])
    _setup(m)

    assert len(m._recording_kick_env_skill_ids) == 1  # only 1 candidate env available


def test_pin_recording_env_modes_forces_task_mode_and_skill_id_for_kick_recorders():
    m = _make_manager(num_envs=100, ratios=[0.3, 0.2])
    _setup(m)

    # simulate a fresh random draw that disagrees with the pin
    all_ids = torch.arange(100)
    m.task_mode[:] = TaskMode.LOCOMOTION
    m.skill_id[:] = 99

    m._pin_recording_env_modes(all_ids)

    for kick_env_id, expected_skill_id in m._recording_kick_env_skill_ids.items():
        assert m.task_mode[kick_env_id] == TaskMode.KICK
        assert m.skill_id[kick_env_id] == expected_skill_id
    loco_env_id = m._recording_env_ids["locomotion"]
    assert m.task_mode[loco_env_id] == TaskMode.LOCOMOTION


def test_disabled_video_recording_is_a_full_noop():
    m = _make_manager(num_envs=100, ratios=[0.3, 0.2], video_enabled=False)
    _setup(m)

    assert m._recording_env_ids == {}
    assert m._recording_kick_env_skill_ids == {}
    # pinning must not crash or touch anything when there's nothing to pin
    all_ids = torch.arange(100)
    m.task_mode[:] = TaskMode.LOCOMOTION
    m._pin_recording_env_modes(all_ids)
    assert torch.all(m.task_mode == TaskMode.LOCOMOTION)


def test_setup_is_idempotent():
    m = _make_manager(num_envs=100, ratios=[0.3, 0.2])
    _setup(m)
    first = dict(m._recording_kick_env_skill_ids)
    n_recorders_first = len(m.simulator.extra_video_recorders)

    _setup(m)  # second call must be a no-op
    assert m._recording_kick_env_skill_ids == first
    assert len(m.simulator.extra_video_recorders) == n_recorders_first
