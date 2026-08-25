"""Unit tests for the 4 P0 regularization terms (managers/reward/terms/wbt.py), 2026-08-05, ported
from RoboNaldo (arXiv:2606.11092): ``ArmDefaultPose``, ``KickFeetAirTime``,
``KickSwingFeetClearance``, ``kick_no_fly``. See ROBONALDO_PORT_SCOPE.md Sec 3b and each class's
own docstring for the full formula and every deliberate adaptation from RoboNaldo's IsaacLab-
specific mechanisms.

``_get_motion_command_and_assert_type`` is patched out for every test that needs a motion_command
(same isolation discipline as ``test_motion_strike_dof_pos_error_exp.py``, since a plain
``SimpleNamespace`` fake can never satisfy that helper's ``isinstance`` check).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import holosoma.managers.reward.terms.wbt as wbt
from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg
from holosoma.managers.reward.manager import RewardManager

_BODY_NAMES = ["pelvis", "left_ankle_roll_link", "right_ankle_roll_link", "torso_link"]
_LEFT_FOOT_IDX = _BODY_NAMES.index("left_ankle_roll_link")
_RIGHT_FOOT_IDX = _BODY_NAMES.index("right_ankle_roll_link")

_ARM_DOF_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
_ALL_DOF_NAMES = ["waist_yaw_joint", *_ARM_DOF_NAMES, "left_hip_pitch_joint"]


class _FakeCfg:
    def __init__(self, **params):
        self.params = params
        self.weight = 1.0


def _zero_state(num_envs: int, num_bodies: int = len(_BODY_NAMES), history: int = 3):
    body_vel = torch.zeros(num_envs, num_bodies, 3)
    contact = torch.zeros(num_envs, history, num_bodies, 3)
    return body_vel, contact


def _fake_env_with_contacts(num_envs: int, contact_force_xyz: torch.Tensor, num_dof: int = 0, device: str = "cpu"):
    simulator = SimpleNamespace(
        body_names=list(_BODY_NAMES),
        contact_forces_history=contact_force_xyz,
        dof_names=list(_ALL_DOF_NAMES) if num_dof == 0 else [f"dof_{i}" for i in range(num_dof)],
        dof_pos=torch.zeros(num_envs, len(_ALL_DOF_NAMES)),
    )
    return SimpleNamespace(num_envs=num_envs, device=device, dt=0.02, simulator=simulator)


def _fake_motion_command(in_kicking_phase: torch.Tensor, in_strike_phase: torch.Tensor | None = None):
    if in_strike_phase is None:
        in_strike_phase = torch.zeros_like(in_kicking_phase)
    return SimpleNamespace(in_kicking_phase=in_kicking_phase, in_strike_phase=in_strike_phase)


# ============================================================================================
# ArmDefaultPose
# ============================================================================================


class _FakeArmEnv:
    def __init__(self, num_envs: int, dof_pos: torch.Tensor, default_dof_pos: torch.Tensor, device: str = "cpu"):
        self.num_envs = num_envs
        self.device = device
        self.simulator = SimpleNamespace(dof_names=list(_ALL_DOF_NAMES), dof_pos=dof_pos)
        self.default_dof_pos = default_dof_pos


def _make_arm_term(arm_dof_names=_ARM_DOF_NAMES, elbow_weight_multiplier=5.0, num_envs=1):
    d = len(_ALL_DOF_NAMES)
    env = _FakeArmEnv(num_envs, torch.zeros(num_envs, d), torch.zeros(num_envs, d))
    cfg = _FakeCfg(arm_dof_names=arm_dof_names, elbow_weight_multiplier=elbow_weight_multiplier)
    return wbt.ArmDefaultPose(cfg, env), env


def test_arm_default_pose_bad_dof_name_raises():
    with pytest.raises(AssertionError):
        _make_arm_term(arm_dof_names=["not_a_real_joint", "left_elbow_joint"])


def test_arm_default_pose_no_elbow_joint_raises():
    with pytest.raises(AssertionError):
        _make_arm_term(arm_dof_names=["left_shoulder_pitch_joint", "waist_yaw_joint"])


def test_arm_default_pose_reset_is_a_noop_and_callable():
    term, _ = _make_arm_term()
    term.reset(torch.tensor([0]))


def test_arm_default_pose_exact_match_gives_zero():
    term, env = _make_arm_term(num_envs=2)
    out = term(env)
    assert torch.allclose(out, torch.zeros(2))


def test_arm_default_pose_formula_matches_weighted_mse_non_elbow():
    """Single non-elbow joint offset by 0.3 rad -> weighted_error_sq = 0.09*1.0, mean over the 14
    arm joints (13 exact + 1 off) = 0.09/14."""
    term, env = _make_arm_term(num_envs=1)
    idx = _ALL_DOF_NAMES.index("left_shoulder_pitch_joint")
    env.simulator.dof_pos[0, idx] = 0.3
    out = term(env)
    expected = (0.3**2) / len(_ARM_DOF_NAMES)
    assert torch.isclose(out[0], torch.tensor(expected), atol=1e-6)


def test_arm_default_pose_elbow_joint_weighted_5x():
    """Same 0.3 rad offset, but on an ELBOW joint -- weighted_error_sq = 0.09*5.0, mean over 14."""
    term, env = _make_arm_term(num_envs=1)
    idx = _ALL_DOF_NAMES.index("left_elbow_joint")
    env.simulator.dof_pos[0, idx] = 0.3
    out = term(env)
    expected = (0.3**2 * 5.0) / len(_ARM_DOF_NAMES)
    assert torch.isclose(out[0], torch.tensor(expected), atol=1e-6)


def test_arm_default_pose_elbow_vs_non_elbow_same_offset_elbow_scores_higher():
    """Regression guard: an equal-magnitude offset on an elbow joint must produce a strictly
    larger penalty than the same offset on a non-elbow joint -- the whole point of the 5x mask."""
    term, env = _make_arm_term(num_envs=2)
    non_elbow_idx = _ALL_DOF_NAMES.index("left_wrist_roll_joint")
    elbow_idx = _ALL_DOF_NAMES.index("right_elbow_joint")
    env.simulator.dof_pos[0, non_elbow_idx] = 0.4
    env.simulator.dof_pos[1, elbow_idx] = 0.4
    out = term(env)
    assert out[1] > out[0], f"elbow offset ({out[1]}) should score higher than non-elbow ({out[0]})"
    assert torch.isclose(out[1], out[0] * 5.0, atol=1e-6)


def test_arm_default_pose_unmasked_joints_ignored():
    """waist_yaw_joint and left_hip_pitch_joint are NOT in the arm-only mask -- huge deviation
    there must not affect the reward at all."""
    term, env = _make_arm_term(num_envs=1)
    env.simulator.dof_pos[0, _ALL_DOF_NAMES.index("waist_yaw_joint")] = 3.0
    env.simulator.dof_pos[0, _ALL_DOF_NAMES.index("left_hip_pitch_joint")] = -2.5
    out = term(env)
    assert torch.allclose(out, torch.zeros(1))


def test_arm_default_pose_weight_zero_is_never_instantiated_by_reward_manager():
    cfg = RewardManagerCfg(
        terms={
            "kick_arm_default_pose": RewardTermCfg(
                func="holosoma.managers.reward.terms.wbt:ArmDefaultPose",
                params={"arm_dof_names": ["left_elbow_joint"]},
                weight=0.0,
                task_mode="kick",
            )
        }
    )
    env = SimpleNamespace(num_envs=4, logger=None)  # no .simulator -- would AttributeError if touched
    manager = RewardManager(cfg, env, device="cpu")
    assert "kick_arm_default_pose" not in manager._term_names
    assert "kick_arm_default_pose" not in manager._term_instances


# ============================================================================================
# KickFeetAirTime
# ============================================================================================


def _make_air_time_term(threshold: float = 0.25, contact_force_threshold: float = 1.0, num_envs: int = 1):
    body_vel, contact = _zero_state(num_envs)
    env = _fake_env_with_contacts(num_envs, contact)
    cfg = _FakeCfg(
        foot_body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
        air_time_threshold=threshold,
        contact_force_threshold=contact_force_threshold,
    )
    return wbt.KickFeetAirTime(cfg, env), env


def test_feet_air_time_bad_foot_name_raises():
    _, contact = _zero_state(1)
    env = _fake_env_with_contacts(1, contact)
    cfg = _FakeCfg(foot_body_names=["not_a_real_foot"])
    with pytest.raises(AssertionError):
        wbt.KickFeetAirTime(cfg, env)


def test_feet_air_time_reset_zeroes_buffers():
    term, env = _make_air_time_term(num_envs=2)
    term._air_time[:] = 5.0
    term._last_contact[:] = True
    term.reset(torch.tensor([0]))
    assert term._air_time[0].sum().item() == 0.0
    assert not term._last_contact[0].any()
    assert term._air_time[1].sum().item() == 10.0  # env 1 untouched


def test_feet_air_time_zero_while_foot_stays_in_contact():
    """A foot that never leaves the ground never accumulates air time -> never pays."""
    term, env = _make_air_time_term(num_envs=1)
    mc = _fake_motion_command(in_kicking_phase=torch.tensor([True]))
    with patch.object(wbt, "_get_motion_command_and_assert_type", return_value=mc):
        for _ in range(5):
            env.simulator.contact_forces_history[0, 0, _LEFT_FOOT_IDX, 2] = 50.0
            env.simulator.contact_forces_history[0, 0, _RIGHT_FOOT_IDX, 2] = 50.0
            out = term(env)
    assert out.item() == 0.0


def test_feet_air_time_pays_on_touchdown_after_sufficient_swing():
    """LEFT foot airborne (no contact) for 13 steps, then touches down; RIGHT foot stays airborne
    throughout (never contributes). air_time reaches 14*0.02=0.28 by the touchdown step, exceeding
    threshold=0.25 -- a genuinely "sufficient" swing, paying the positive excess (0.03)."""
    term, env = _make_air_time_term(threshold=0.25, num_envs=1)
    mc = _fake_motion_command(in_kicking_phase=torch.tensor([True]))
    with patch.object(wbt, "_get_motion_command_and_assert_type", return_value=mc):
        out = None
        for _ in range(13):
            out = term(env)
        assert out.item() == 0.0, "still swinging -- no touchdown yet, no payout"

        # LEFT foot touches down on step 14; RIGHT foot never gets contact force.
        env.simulator.contact_forces_history[0, 0, _LEFT_FOOT_IDX, 2] = 50.0
        out = term(env)
    expected_air_time = 0.02 * 14  # accumulated through the touchdown step itself
    assert torch.isclose(out, torch.tensor(expected_air_time - 0.25), atol=1e-6)


def test_feet_air_time_gated_off_outside_kicking_phase():
    term, env = _make_air_time_term(num_envs=1)
    mc = _fake_motion_command(in_kicking_phase=torch.tensor([False]))
    with patch.object(wbt, "_get_motion_command_and_assert_type", return_value=mc):
        for _ in range(4):
            term(env)
        env.simulator.contact_forces_history[0, 0, _LEFT_FOOT_IDX, 2] = 50.0
        env.simulator.contact_forces_history[0, 0, _RIGHT_FOOT_IDX, 2] = 50.0
        out = term(env)
    assert out.item() == 0.0, "in_kicking_phase=False must zero the reward even at a genuine touchdown"


def test_feet_air_time_per_env_independent():
    """Env 0's left foot touches down (a short, sub-threshold hop -- pays a NEGATIVE excess, which
    is still the correct, nonzero, "this foot just landed" signal); env 1 keeps both feet airborne
    the whole time and must show exactly 0 (no touchdown event at all)."""
    term, env = _make_air_time_term(num_envs=2)
    mc = _fake_motion_command(in_kicking_phase=torch.tensor([True, True]))
    with patch.object(wbt, "_get_motion_command_and_assert_type", return_value=mc):
        for _ in range(4):
            term(env)
        env.simulator.contact_forces_history[0, 0, _LEFT_FOOT_IDX, 2] = 50.0
        out = term(env)
    assert out[0].item() != 0.0, "env 0's touchdown must register as a nonzero event"
    assert out[1].item() == 0.0, "env 1 never touched down -- must stay exactly 0"


def test_feet_air_time_weight_zero_is_never_instantiated_by_reward_manager():
    cfg = RewardManagerCfg(
        terms={
            "kick_feet_air_time": RewardTermCfg(
                func="holosoma.managers.reward.terms.wbt:KickFeetAirTime",
                params={"foot_body_names": ["left_ankle_roll_link", "right_ankle_roll_link"]},
                weight=0.0,
                task_mode="kick",
            )
        }
    )
    env = SimpleNamespace(num_envs=4, logger=None)
    manager = RewardManager(cfg, env, device="cpu")
    assert "kick_feet_air_time" not in manager._term_names
    assert "kick_feet_air_time" not in manager._term_instances


# ============================================================================================
# KickSwingFeetClearance
# ============================================================================================


def _fake_env_with_clearance(num_envs, contact_force_xyz, feet_heights, device="cpu"):
    simulator = SimpleNamespace(body_names=list(_BODY_NAMES), contact_forces_history=contact_force_xyz)
    terrain_state = SimpleNamespace(feet_heights=feet_heights)
    terrain_manager = SimpleNamespace(get_state=lambda name: terrain_state if name == "locomotion_terrain" else None)
    return SimpleNamespace(num_envs=num_envs, device=device, simulator=simulator, terrain_manager=terrain_manager)


def _make_clearance_term(target_height=0.12, max_penalty=0.5, threshold=1.0, num_envs=1):
    _, contact = _zero_state(num_envs)
    feet_heights = torch.zeros(num_envs, 2)
    env = _fake_env_with_clearance(num_envs, contact, feet_heights)
    cfg = _FakeCfg(
        foot_body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
        target_height=target_height,
        max_penalty=max_penalty,
        contact_force_threshold=threshold,
    )
    return wbt.KickSwingFeetClearance(cfg, env), env


def test_feet_clearance_bad_foot_name_raises():
    _, contact = _zero_state(1)
    env = _fake_env_with_clearance(1, contact, torch.zeros(1, 2))
    cfg = _FakeCfg(foot_body_names=["not_a_real_foot"])
    with pytest.raises(AssertionError):
        wbt.KickSwingFeetClearance(cfg, env)


def test_feet_clearance_reset_is_a_noop_and_callable():
    term, _ = _make_clearance_term()
    term.reset(torch.tensor([0]))


def test_feet_clearance_zero_when_foot_in_contact_even_if_low():
    """A planted (in-contact) foot at height 0 must NOT be penalized -- clearance only applies to
    swing feet."""
    term, env = _make_clearance_term(num_envs=1)
    env.simulator.contact_forces_history[0, 0, _LEFT_FOOT_IDX, 2] = 50.0
    env.simulator.contact_forces_history[0, 0, _RIGHT_FOOT_IDX, 2] = 50.0
    env.terrain_manager.get_state("locomotion_terrain").feet_heights[0] = torch.tensor([0.0, 0.0])
    mc = _fake_motion_command(in_kicking_phase=torch.tensor([True]), in_strike_phase=torch.tensor([False]))
    with patch.object(wbt, "_get_motion_command_and_assert_type", return_value=mc):
        out = term(env)
    assert out.item() == 0.0


def test_feet_clearance_nonzero_for_low_swing_foot():
    """A swing (no contact) foot below target_height pays clamp(target-height,0)^2."""
    term, env = _make_clearance_term(target_height=0.12, num_envs=1)
    env.terrain_manager.get_state("locomotion_terrain").feet_heights[0] = torch.tensor([0.02, 0.0])
    mc = _fake_motion_command(in_kicking_phase=torch.tensor([True]), in_strike_phase=torch.tensor([False]))
    with patch.object(wbt, "_get_motion_command_and_assert_type", return_value=mc):
        out = term(env)
    expected = (0.12 - 0.02) ** 2 + (0.12 - 0.0) ** 2
    assert torch.isclose(out, torch.tensor(expected), atol=1e-6)


def test_feet_clearance_capped_at_max_penalty():
    term, env = _make_clearance_term(target_height=0.5, max_penalty=0.1, num_envs=1)
    env.terrain_manager.get_state("locomotion_terrain").feet_heights[0] = torch.tensor([0.0, 0.0])
    mc = _fake_motion_command(in_kicking_phase=torch.tensor([True]), in_strike_phase=torch.tensor([False]))
    with patch.object(wbt, "_get_motion_command_and_assert_type", return_value=mc):
        out = term(env)
    assert torch.isclose(out, torch.tensor(0.1), atol=1e-6)


def test_feet_clearance_zero_during_strike_phase():
    """Gated to approach-only (in_kicking_phase & ~in_strike_phase) -- must not fight the
    authored strike motion."""
    term, env = _make_clearance_term(target_height=0.12, num_envs=1)
    env.terrain_manager.get_state("locomotion_terrain").feet_heights[0] = torch.tensor([0.0, 0.0])
    mc = _fake_motion_command(in_kicking_phase=torch.tensor([True]), in_strike_phase=torch.tensor([True]))
    with patch.object(wbt, "_get_motion_command_and_assert_type", return_value=mc):
        out = term(env)
    assert out.item() == 0.0


def test_feet_clearance_zero_during_recovery():
    term, env = _make_clearance_term(target_height=0.12, num_envs=1)
    env.terrain_manager.get_state("locomotion_terrain").feet_heights[0] = torch.tensor([0.0, 0.0])
    mc = _fake_motion_command(in_kicking_phase=torch.tensor([False]), in_strike_phase=torch.tensor([False]))
    with patch.object(wbt, "_get_motion_command_and_assert_type", return_value=mc):
        out = term(env)
    assert out.item() == 0.0


def test_feet_clearance_weight_zero_is_never_instantiated_by_reward_manager():
    cfg = RewardManagerCfg(
        terms={
            "kick_swing_feet_clearance": RewardTermCfg(
                func="holosoma.managers.reward.terms.wbt:KickSwingFeetClearance",
                params={"foot_body_names": ["left_ankle_roll_link", "right_ankle_roll_link"]},
                weight=0.0,
                task_mode="kick",
            )
        }
    )
    env = SimpleNamespace(num_envs=4, logger=None)
    manager = RewardManager(cfg, env, device="cpu")
    assert "kick_swing_feet_clearance" not in manager._term_names
    assert "kick_swing_feet_clearance" not in manager._term_instances


# ============================================================================================
# kick_no_fly
# ============================================================================================


def _fake_env_for_no_fly(feet_heights):
    terrain_state = SimpleNamespace(feet_heights=feet_heights)
    terrain_manager = SimpleNamespace(get_state=lambda name: terrain_state if name == "locomotion_terrain" else None)
    return SimpleNamespace(terrain_manager=terrain_manager)


def test_no_fly_zero_when_both_feet_grounded():
    env = _fake_env_for_no_fly(torch.tensor([[0.0, 0.0]]))
    out = wbt.kick_no_fly(env)
    assert out.item() == 0.0


def test_no_fly_zero_when_only_one_foot_flying():
    env = _fake_env_for_no_fly(torch.tensor([[0.2, 0.0]]))
    out = wbt.kick_no_fly(env)
    assert out.item() == 0.0


def test_no_fly_one_when_both_feet_flying():
    env = _fake_env_for_no_fly(torch.tensor([[0.2, 0.15]]))
    out = wbt.kick_no_fly(env)
    assert out.item() == 1.0


def test_no_fly_threshold_is_exclusive_not_inclusive():
    env = _fake_env_for_no_fly(torch.tensor([[0.05, 0.05]]))
    out = wbt.kick_no_fly(env, height_threshold=0.05)
    assert out.item() == 0.0, "exactly at threshold must not count as flying"


def test_no_fly_per_env_independent():
    env = _fake_env_for_no_fly(torch.tensor([[0.2, 0.2], [0.0, 0.0]]))
    out = wbt.kick_no_fly(env)
    assert out[0].item() == 1.0
    assert out[1].item() == 0.0
