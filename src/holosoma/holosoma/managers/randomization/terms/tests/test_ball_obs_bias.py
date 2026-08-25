"""Unit tests for the per-skill ball-observation-bias mechanism:
_get_ball_obs_bias_scale_per_motion (per-motion magnitude table, N-skill-aware) and
randomize_ball_obs_bias (the reset hook that draws each env's per-episode bias from it).

Isolated via a bare fake env (only the specific attributes these two functions touch), matching
this project's existing bare-instance test convention -- no real env/simulator needed.
"""

from unittest.mock import patch

import torch

from holosoma.managers.randomization.terms.locomotion import (
    _get_ball_obs_bias_scale_per_motion,
    randomize_ball_obs_bias,
)


class _FakeSkillConfig:
    def __init__(self, observation_bias: float):
        self.observation_bias = observation_bias


class _FakeMotion:
    def __init__(self, num_motions: int):
        self.num_motions = num_motions


class _FakeMotionCommand:
    def __init__(self, num_motions: int, motion_ids: torch.Tensor, skill_ball_configs=None):
        self.motion = _FakeMotion(num_motions)
        self.motion_ids = motion_ids
        if skill_ball_configs is not None:
            self.skill_ball_configs = skill_ball_configs


class _FakeCommandManager:
    def __init__(self, motion_command):
        self._motion_command = motion_command

    def get_state(self, name: str):
        assert name == "motion_command"
        return self._motion_command


class _FakeEnv:
    def __init__(self, num_envs: int, motion_command: _FakeMotionCommand):
        self.num_envs = num_envs
        self.device = "cpu"
        self.command_manager = _FakeCommandManager(motion_command)


def test_legacy_mode_broadcasts_single_ball_config_value_to_the_only_motion():
    motion_ids = torch.zeros(10, dtype=torch.long)
    mc = _FakeMotionCommand(num_motions=1, motion_ids=motion_ids)  # no skill_ball_configs at all
    env = _FakeEnv(num_envs=10, motion_command=mc)

    fake_ball_cfg = type("FakeBallConfig", (), {"observation_bias": 0.25})()
    with patch("holosoma.config_types.simulator.load_ball_config", return_value=fake_ball_cfg):
        scale = _get_ball_obs_bias_scale_per_motion(env)

    assert scale.shape == (1,)
    assert float(scale[0]) == 0.25


def test_n_skill_mode_gathers_per_skill_values_in_declaration_order():
    motion_ids = torch.zeros(10, dtype=torch.long)
    mc = _FakeMotionCommand(
        num_motions=2,
        motion_ids=motion_ids,
        skill_ball_configs=[_FakeSkillConfig(0.0), _FakeSkillConfig(0.5)],
    )
    env = _FakeEnv(num_envs=10, motion_command=mc)

    scale = _get_ball_obs_bias_scale_per_motion(env)
    assert scale.tolist() == [0.0, 0.5]


def test_scale_is_cached_on_env_not_recomputed():
    motion_ids = torch.zeros(10, dtype=torch.long)
    mc = _FakeMotionCommand(
        num_motions=1, motion_ids=motion_ids, skill_ball_configs=[_FakeSkillConfig(0.3)]
    )
    env = _FakeEnv(num_envs=10, motion_command=mc)

    first = _get_ball_obs_bias_scale_per_motion(env)
    # Mutate the source SkillConfig's value -- if caching works, the cached tensor must NOT change.
    mc.skill_ball_configs[0].observation_bias = 999.0
    second = _get_ball_obs_bias_scale_per_motion(env)
    assert first is second
    assert abs(float(second[0]) - 0.3) < 1e-6


def test_randomize_ball_obs_bias_gives_zero_skill_zero_bias_regardless_of_other_skills():
    torch.manual_seed(0)
    num_envs = 20
    # envs 0-9 -> motion 0 (bias 0.0), envs 10-19 -> motion 1 (bias 2.0)
    motion_ids = torch.cat([torch.zeros(10, dtype=torch.long), torch.ones(10, dtype=torch.long)])
    mc = _FakeMotionCommand(
        num_motions=2,
        motion_ids=motion_ids,
        skill_ball_configs=[_FakeSkillConfig(0.0), _FakeSkillConfig(2.0)],
    )
    env = _FakeEnv(num_envs=num_envs, motion_command=mc)

    all_env_ids = torch.arange(num_envs)
    randomize_ball_obs_bias(env, all_env_ids)

    skill0_bias = env._ball_obs_bias[:10]
    skill1_bias = env._ball_obs_bias[10:]
    assert torch.all(skill0_bias == 0.0)  # skill 0's bias magnitude is 0 -> exact no-op
    assert torch.all(skill1_bias.abs() <= 2.0)  # within skill 1's +/- 2.0 half-range
    assert torch.any(skill1_bias != 0.0)  # actually drew something nonzero (not accidentally zeroed)


def test_randomize_ball_obs_bias_legacy_mode_matches_old_scalar_behavior():
    torch.manual_seed(1)
    num_envs = 15
    motion_ids = torch.zeros(num_envs, dtype=torch.long)
    mc = _FakeMotionCommand(num_motions=1, motion_ids=motion_ids)  # legacy, no skill_ball_configs
    env = _FakeEnv(num_envs=num_envs, motion_command=mc)

    fake_ball_cfg = type("FakeBallConfig", (), {"observation_bias": 1.5})()
    with patch("holosoma.config_types.simulator.load_ball_config", return_value=fake_ball_cfg):
        randomize_ball_obs_bias(env, torch.arange(num_envs))

    assert torch.all(env._ball_obs_bias.abs() <= 1.5)
    assert torch.any(env._ball_obs_bias != 0.0)


def test_randomize_ball_obs_bias_zero_everywhere_when_legacy_bias_is_zero():
    motion_ids = torch.zeros(10, dtype=torch.long)
    mc = _FakeMotionCommand(num_motions=1, motion_ids=motion_ids)
    env = _FakeEnv(num_envs=10, motion_command=mc)

    fake_ball_cfg = type("FakeBallConfig", (), {"observation_bias": 0.0})()
    with patch("holosoma.config_types.simulator.load_ball_config", return_value=fake_ball_cfg):
        randomize_ball_obs_bias(env, torch.arange(10))

    assert torch.all(env._ball_obs_bias == 0.0)
