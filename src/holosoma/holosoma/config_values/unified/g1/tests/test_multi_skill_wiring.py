"""Integration checks that config_values/unified/g1/{command,experiment,reward}.py wire
MultiSkillConfig correctly in N-skill mode, and are BIT-IDENTICAL to the pre-existing single-clip
path when N-skill mode is off.

HOLOSOMA_SKILLS_CONFIG gates N-skill mode at Python IMPORT time (same discipline as
HOLOSOMA_BALL_CONFIG, see config_types/multi_skill.py's docstring) and several config_values
modules cache resolved values in module-level state -- so switching modes mid-process via
importlib.reload is fragile (stale cached state in dependency modules). Each check here instead
runs in a FRESH subprocess with the env var set (or not), exactly how a real training launch would
set it, which is both more robust and more representative than an in-process reload would be.
"""

import json
import subprocess
import sys
from pathlib import Path

_HOLOSOMA_DIR = Path(__file__).resolve().parents[5]  # .../src/holosoma
_FORK_ROOT = _HOLOSOMA_DIR.parents[1]  # .../unified_ball_kicking_skills
# 2026-08-23: stageB_and_C.yaml (the old self-contained 1-file-mode fixture) was deleted as part
# of the configs/skill//configs/task/ reorg. multi_skills_highratio.yaml is its replacement here --
# a currently-shipped, real 2-skill file (2-file mode, its own `task_config:` declaration resolved
# via holosoma/__init__.py's bootstrap) rather than a synthetic tmp_path fixture, preserving these
# tests' original intent of exercising the real shipped config end-to-end.
_SKILLS_YAML = _FORK_ROOT / "configs" / "skill" / "multi_skills_highratio.yaml"


def _run_probe(extra_env: dict[str, str] | None, code: str) -> dict:
    """Run `code` in a fresh subprocess (cwd = the holosoma package dir, so `import holosoma...`
    resolves the same way normal training entrypoints do), with `code` printing a single JSON line
    to stdout as its result."""
    import os

    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(_HOLOSOMA_DIR),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"probe failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    return json.loads(result.stdout.strip().splitlines()[-1])


_PROBE_CODE = """
import json
import holosoma.config_values.unified.g1.command as command
import holosoma.config_values.unified.g1.experiment as experiment
import holosoma.config_values.unified.g1.reward as reward
import holosoma.config_values.unified.g1.termination as termination
from holosoma.config_types.multi_skill import load_multi_skill_config, multi_skill_mode_enabled

mc = command._motion_config_unified_kick
terms = reward.g1_29dof_unified_reward.terms
term_terms = termination.g1_29dof_unified_termination.terms
out = {
    "motion_file": mc.motion_file,
    "motion_files": mc.motion_files,
    "motion_recovery_duration_s": mc.motion_recovery_duration_s,
    "motion_head_velocity_smoothing_frames": mc.motion_head_velocity_smoothing_frames,
    "motion_head_velocity_smoothing_frames_per_motion": mc.motion_head_velocity_smoothing_frames_per_motion,
    "skill_ball_config_count": len(mc.skill_ball_configs),
    "rsi_scope_to_authored_clip": mc.rsi_scope_to_authored_clip,
    "critical_frame_oversampling_prob": mc.critical_frame_oversampling_prob,
    "critical_frame_sampling_window": mc.critical_frame_sampling_window,
    "start_at_timestep_zero_prob_per_motion": mc.start_at_timestep_zero_prob_per_motion,
    "rsi_scope_to_authored_clip_per_motion": mc.rsi_scope_to_authored_clip_per_motion,
    "critical_frame_oversampling_prob_per_motion": mc.critical_frame_oversampling_prob_per_motion,
    "critical_frame_sampling_window_per_motion": mc.critical_frame_sampling_window_per_motion,
    "command_params_keys": sorted(command.g1_29dof_unified_command.params.keys()),
    "scene_ball_radius": experiment._scene_ball_cfg.radius,
    "scene_ball_mass": experiment._scene_ball_cfg.mass,
    "kick_ball_proximity_weight": reward._shooting_terms["kick_ball_proximity"].weight,
    "kick_goal_success_burst_weight": reward._shooting_terms["kick_goal_success_burst"].weight,
    "kick_alive_func": terms["kick_alive"].func,
    "kick_alive_weight": terms["kick_alive"].weight,
    "motion_tracking_func": terms["motion_relative_body_position_error_exp"].func,
    "motion_tracking_weight": terms["motion_relative_body_position_error_exp"].weight,
    "kick_recovery_stand_height_weight": terms["penalty_kick_recovery_stand_height"].weight,
    "penalty_stand_height_target_height": terms["penalty_stand_height"].params["target_height"],
    "penalty_stand_height_deadzone": terms["penalty_stand_height"].params["deadzone"],
    "penalty_kick_recovery_stand_height_target_height": terms["penalty_kick_recovery_stand_height"].params["target_height"],
    "penalty_kick_recovery_stand_height_deadzone": terms["penalty_kick_recovery_stand_height"].params["deadzone"],
    "kick_safety_contact_force_weight": terms["kick_penalty_excess_contact_force"].weight,
    "motion_strike_dof_pos_func": terms["motion_strike_dof_pos_error_exp"].func,
    "motion_strike_dof_pos_weight": terms["motion_strike_dof_pos_error_exp"].weight,
    "motion_strike_dof_pos_task_mode": terms["motion_strike_dof_pos_error_exp"].task_mode,
    "motion_strike_dof_pos_dof_names": terms["motion_strike_dof_pos_error_exp"].params["dof_names"],
    "motion_strike_dof_pos_sigma": terms["motion_strike_dof_pos_error_exp"].params["sigma"],
    "kick_penalty_swing_orientation_deadzone": terms["kick_penalty_swing_orientation"].params["deadzone"],
    "kick_penalty_swing_torso_orientation_deadzone": terms["kick_penalty_swing_torso_orientation"].params["deadzone"],
    "action_rate_l2_weight": terms["action_rate_l2"].weight,
    "kick_contact_orientation_weight": terms["kick_contact_orientation"].weight,
    "kick_ball_velocity_weight": terms["kick_ball_velocity"].weight,
    "kick_error_ball_to_target_weight": terms["kick_error_ball_to_target"].weight,
    "kick_error_ball_to_target_sigma": terms["kick_error_ball_to_target"].params["sigma"],
    "kick_goal_success_burst_weight_tuned": terms["kick_goal_success_burst"].weight,
    "kick_predicted_error_ball_to_target_weight": terms["kick_predicted_error_ball_to_target"].weight,
    "skill_reward_scales": (
        [
            {
                "motion_tracking_reward_scale": sc.motion_tracking_reward_scale,
                "kick_recovery_posture_reward_scale": sc.kick_recovery_posture_reward_scale,
                "kick_safety_reward_scale": sc.kick_safety_reward_scale,
                "kick_alive_reward_scale": sc.kick_alive_reward_scale,
                "kick_alive_pre_kick_ratio": sc.kick_alive_pre_kick_ratio,
                "root_tracking_reward_scale": sc.root_tracking_reward_scale,
            }
            for sc in load_multi_skill_config().skills
        ]
        if multi_skill_mode_enabled()
        else []
    ),
    "kick_alive_weight_per_skill": terms["kick_alive"].weight_per_skill,
    "kick_ball_proximity_weight_per_skill": terms["kick_ball_proximity"].weight_per_skill,
    "kick_ball_velocity_weight_per_skill": terms["kick_ball_velocity"].weight_per_skill,
    "skill_task_config_paths": (
        [str(p) if p is not None else None for p in reward._skill_task_config_paths]
        if reward._skill_task_config_paths is not None
        else None
    ),
    "kick_recovery_stand_height_deadzone_params_per_skill": terms["penalty_kick_recovery_stand_height"].params_per_skill,
    "kick_penalty_swing_orientation_params_per_skill": terms["kick_penalty_swing_orientation"].params_per_skill,
    "kick_penalty_excess_contact_force_params_per_skill": terms["kick_penalty_excess_contact_force"].params_per_skill,
    "kick_balance_potential_registered": "kick_balance_potential" in terms,
    "kick_balance_potential_weight_per_skill": (
        terms["kick_balance_potential"].weight_per_skill if "kick_balance_potential" in terms else None
    ),
    "kick_recovery_drift_params_per_skill": (
        term_terms["kick_recovery_drift"].params_per_skill if "kick_recovery_drift" in term_terms else None
    ),
    "kick_recovery_low_height_registered": "kick_recovery_low_height" in term_terms,
    "kick_recovery_drift_registered": "kick_recovery_drift" in term_terms,
    "kick_recovery_low_height_params_per_skill": (
        term_terms["kick_recovery_low_height"].params_per_skill if "kick_recovery_low_height" in term_terms else None
    ),
    "low_height_params_per_skill": term_terms["low_height"].params_per_skill,
    "contact_params_per_skill": term_terms["contact"].params_per_skill,
    "kick_low_height_params_per_skill": term_terms["kick_low_height"].params_per_skill,
    "motion_tracking_params_per_skill": terms["motion_relative_body_position_error_exp"].params_per_skill,
    "kick_penalty_ee_body_pos_divergence_params_per_skill": terms["kick_penalty_ee_body_pos_divergence"].params_per_skill,
    "bad_tracking_params_per_skill": term_terms["bad_tracking"].params_per_skill,
    "kick_foot_strike_pitch_params_per_skill": terms["kick_foot_strike_pitch"].params_per_skill,
    "kick_ball_over_line_params_per_skill": terms["kick_ball_over_line"].params_per_skill,
}
print(json.dumps(out))
"""


def test_legacy_mode_unaffected_when_holosoma_skills_config_unset():
    result = _run_probe(None, _PROBE_CODE)
    assert result["motion_files"] == []  # N-skill mode off -> legacy single motion_file path
    assert result["skill_ball_config_count"] == 0
    assert "skill_motion_training_ratios" not in result["command_params_keys"]
    assert "kick_probability" in result["command_params_keys"]
    # Weights are the bare per-term k (RoboNaldo Table B.1 ratios), same in both modes now --
    # w_g multiplication moved to runtime in BOTH modes, see shooting.py/shooting_curriculum.py.
    assert result["kick_ball_proximity_weight"] == 2.0
    assert result["kick_goal_success_burst_weight"] == 300.0
    assert result["skill_reward_scales"] == []
    assert result["motion_head_velocity_smoothing_frames_per_motion"] == []
    assert result["start_at_timestep_zero_prob_per_motion"] == []
    assert result["rsi_scope_to_authored_clip_per_motion"] == []
    assert result["critical_frame_oversampling_prob_per_motion"] == []
    assert result["critical_frame_sampling_window_per_motion"] == []


def test_legacy_mode_new_reward_scales_are_bit_identical_no_ops():
    """2026-07-28: the 4 new per-skill reward-category scales (motion_tracking/kick_recovery_
    posture/kick_safety/kick_alive). In legacy mode there's no SkillConfig at all, so every
    category resolves to a flat 1.0 (see kick_reward_scales.py's legacy fallback) -- weights
    (the term's own base magnitude) must be completely unchanged from before this feature
    existed, and the wrapped funcs (kick_alive, the 6 motion-tracking terms) route through
    kick_scale_wrappers regardless of mode -- wrapping is unconditional, only the resolved scale
    VALUE differs between modes, not whether wrapping happens."""
    result = _run_probe(None, _PROBE_CODE)
    assert result["kick_alive_weight"] == 10.0
    assert result["kick_alive_func"] == "holosoma.managers.reward.terms.kick_scale_wrappers:kick_alive_scaled"
    assert result["motion_tracking_weight"] == 2.0
    assert (
        result["motion_tracking_func"]
        == "holosoma.managers.reward.terms.kick_scale_wrappers:motion_relative_body_position_error_exp_scaled"
    )
    assert result["kick_recovery_stand_height_weight"] == -40.0
    assert result["kick_safety_contact_force_weight"] == -1.0


def test_n_skill_mode_wires_motion_files_and_skill_configs():
    result = _run_probe({"HOLOSOMA_SKILLS_CONFIG": str(_SKILLS_YAML)}, _PROBE_CODE)
    assert len(result["motion_files"]) >= 1
    assert result["skill_ball_config_count"] == len(result["motion_files"])
    assert len(result["motion_recovery_duration_s"]) == len(result["motion_files"])
    assert "skill_motion_training_ratios" in result["command_params_keys"]
    # Weights are unaffected by N-skill mode -- same bare k in both modes (see above).
    assert result["kick_ball_proximity_weight"] == 2.0
    assert result["kick_goal_success_burst_weight"] == 300.0


def test_n_skill_mode_reward_scale_fields_default_to_1_0_per_skill(tmp_path):
    """The 4 new per-skill reward-category scales default to 1.0 (exact no-op) for every skill
    when a skills yaml doesn't set them, one entry per skill, index-aligned with motion_files --
    and wiring (which func each term points at) is identical to legacy mode either way (see
    kick_reward_scales.py)."""
    result = _run_probe({"HOLOSOMA_SKILLS_CONFIG": str(_default_two_skill_yaml(tmp_path))}, _PROBE_CODE)
    scales = result["skill_reward_scales"]
    assert len(scales) == len(result["motion_files"])
    for sc in scales:
        assert sc == {
            "motion_tracking_reward_scale": 1.0,
            "kick_recovery_posture_reward_scale": 1.0,
            "kick_safety_reward_scale": 1.0,
            "kick_alive_reward_scale": 1.0,
            "kick_alive_pre_kick_ratio": 1.0,
            "root_tracking_reward_scale": 1.0,
        }
    assert result["kick_alive_func"] == "holosoma.managers.reward.terms.kick_scale_wrappers:kick_alive_scaled"
    assert result["kick_alive_weight"] == 10.0


def test_n_skill_mode_scene_ball_radius_mass_match_multi_skill_yaml(tmp_path):
    skills_yaml = _default_two_skill_yaml(tmp_path)
    result = _run_probe({"HOLOSOMA_SKILLS_CONFIG": str(skills_yaml)}, _PROBE_CODE)
    import yaml

    raw = yaml.safe_load(skills_yaml.read_text())
    assert result["scene_ball_radius"] == raw["ball"]["radius"]
    assert result["scene_ball_mass"] == raw["ball"]["mass"]


def test_motion_strike_dof_pos_registered_as_true_no_op_by_default():
    """The new strike-phase joint-tracking term (2026-08-03) must ship as a byte-identical no-op:
    weight=0.0 (RewardManager skips weight==0.0 terms entirely, managers/reward/manager.py:60-62),
    correct func/task_mode, and dof_names sourced from g1_29dof.upper_dof_names (the exact 17
    arm+waist joints, config_values/robot.py) -- not a hand-typed duplicate that could silently
    drift from that list."""
    result = _run_probe(None, _PROBE_CODE)
    assert result["motion_strike_dof_pos_func"] == "holosoma.managers.reward.terms.wbt:MotionStrikeDofPosErrorExp"
    assert result["motion_strike_dof_pos_weight"] == 0.0
    assert result["motion_strike_dof_pos_task_mode"] == "kick"
    assert result["motion_strike_dof_pos_sigma"] == 0.5
    dof_names = result["motion_strike_dof_pos_dof_names"]
    assert len(dof_names) == 17
    assert dof_names == [
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
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
    # no leg DOF anywhere in the mask
    assert not any("hip" in n or "knee" in n or "ankle" in n for n in dof_names)


def test_motion_strike_dof_pos_weight_and_sigma_reach_the_term_via_tuning_yaml(tmp_path):
    """Proves the configs/kicking_motion_reward_tuning.yaml opt-in path actually reaches the
    registered term end-to-end: a minimal override yaml naming ONLY this term (no other category
    needs to be present -- _load_raw only rejects UNRECOGNIZED top-level sections, not missing
    ones) must flip both weight and sigma away from their shipped no-op defaults."""
    override = tmp_path / "tuning_override.yaml"
    override.write_text(
        "motion_tracking_reward:\n"
        "  motion_strike_dof_pos_error_exp: 1.0\n"
        "  _sigma:\n"
        "    motion_strike_dof_pos_error_exp: 0.35\n"
    )
    result = _run_probe({"HOLOSOMA_REWARD_TUNING_CONFIG": str(override)}, _PROBE_CODE)
    assert result["motion_strike_dof_pos_weight"] == 1.0
    assert result["motion_strike_dof_pos_sigma"] == 0.35
    # unrelated terms untouched by the override
    assert result["motion_tracking_weight"] == 2.0


def test_kick_swing_orientation_deadzone_defaults_zero_by_default():
    """penalty_kick_swing_orientation/_torso_orientation must ship with deadzone=0.0 (the
    pre-2026-08-12 hardcoded value) unless a task-config yaml explicitly opts in -- see
    MultiSkillConfig.kick_swing_orientation_deadzone's own docstring for the full rationale."""
    result = _run_probe(None, _PROBE_CODE)
    assert result["kick_penalty_swing_orientation_deadzone"] == 0.0
    assert result["kick_penalty_swing_torso_orientation_deadzone"] == 0.0


def test_swing_stability_fix_reaches_the_live_terms_via_the_real_stagec1_yaml():
    """End-to-end proof that configs/task_config_stageC1.yaml itself -- not a synthetic override
    fixture -- correctly resolves both the deadzone fix (part 1) and the strike-phase upper-body
    tracking term (part 2): kick_penalty_swing_orientation/_torso_orientation deadzones move away
    from 0.0 to the npz-measured values, AND motion_strike_dof_pos_error_exp's weight flips from
    its shipped no-op (0.0) to 1.0, simultaneously, from the one real file this project actually
    trains with."""
    task_config = _FORK_ROOT / "configs" / "task" / "task_config_stageC1.yaml"
    assert task_config.exists(), f"expected {task_config} to exist"
    result = _run_probe(
        {"HOLOSOMA_SKILLS_CONFIG": str(_FORK_ROOT / "configs" / "skills1.yaml"), "HOLOSOMA_TASK_CONFIG": str(task_config)},
        _PROBE_CODE,
    )
    assert result["kick_penalty_swing_orientation_deadzone"] == 0.55
    assert result["kick_penalty_swing_torso_orientation_deadzone"] == 0.35
    assert result["motion_strike_dof_pos_weight"] == 1.0
    assert result["motion_strike_dof_pos_sigma"] == 0.5  # unchanged from its own Python default


def test_phase1_budget_rebalance_reaches_the_live_terms_via_default_tuning_yaml():
    """2026-08-05, ROBONALDO_PORT_SCOPE.md Sec 1b/1c: proves the DEFAULT
    configs/kicking_motion_reward_tuning.yaml (no override -- this is what every real training
    launch reads unless HOLOSOMA_REWARD_TUNING_CONFIG is set) actually carries the RoboNaldo-ratio
    re-enable values through to the live, tuned RewardTermCfg objects, and that the two terms
    left deliberately at 0.0 (kick_goal_success_burst, kick_predicted_error_ball_to_target) stay
    there."""
    result = _run_probe(None, _PROBE_CODE)
    assert result["kick_contact_orientation_weight"] == 4.0
    assert result["kick_ball_velocity_weight"] == 1.0
    assert result["kick_error_ball_to_target_weight"] == 20.0
    assert result["kick_error_ball_to_target_sigma"] == 4.2
    assert result["action_rate_l2_weight"] == -0.04
    # deliberately untouched -- RoboNaldo's goal_reward_burst is 0.0 in S2a/S2b too, and
    # kick_predicted_error_ball_to_target has no RoboNaldo counterpart to ratio-derive from.
    assert result["kick_goal_success_burst_weight_tuned"] == 0.0
    assert result["kick_predicted_error_ball_to_target_weight"] == 0.0
    # unrelated, already-active terms untouched by this change
    assert result["kick_alive_weight"] == 10.0
    assert result["motion_tracking_weight"] == 2.0


def test_kick_alive_pre_kick_ratio_defaults_to_1_0_true_no_op(tmp_path):
    """SkillConfig.kick_alive_pre_kick_ratio (2026-08-05) must default to 1.0 -- the exact no-op
    value -- for every skill when a skills yaml doesn't set it, same convention as every other
    per-skill reward-category scale in this class."""
    result = _run_probe({"HOLOSOMA_SKILLS_CONFIG": str(_default_two_skill_yaml(tmp_path))}, _PROBE_CODE)
    for sc in result["skill_reward_scales"]:
        assert sc["kick_alive_pre_kick_ratio"] == 1.0


def test_kick_alive_pre_kick_ratio_loads_from_skill_yaml(tmp_path):
    """A skill yaml block that DOES set kick_alive_pre_kick_ratio must have it reach the resolved
    SkillConfig, index-aligned with its own skill, leaving other skills (and other fields on the
    same skill) untouched."""
    import yaml

    raw = yaml.safe_load(_default_two_skill_yaml(tmp_path).read_text())
    skill_keys = [k for k in raw if k.startswith("motion_skill_")]
    assert skill_keys, "fixture yaml must have at least one motion_skill_N block"
    raw[skill_keys[0]]["kick_alive_pre_kick_ratio"] = 0.1
    override = tmp_path / "skills_override.yaml"
    override.write_text(yaml.dump(raw))

    result = _run_probe({"HOLOSOMA_SKILLS_CONFIG": str(override)}, _PROBE_CODE)
    scales = result["skill_reward_scales"]
    assert scales[0]["kick_alive_pre_kick_ratio"] == 0.1
    assert scales[0]["kick_alive_reward_scale"] == 1.0  # untouched sibling field
    for sc in scales[1:]:
        assert sc["kick_alive_pre_kick_ratio"] == 1.0  # untouched sibling skill


def test_root_tracking_reward_scale_defaults_to_1_0_true_no_op():
    """SkillConfig.root_tracking_reward_scale (2026-08-05) must default to 1.0 -- the exact no-op
    value -- for every skill when a skills yaml doesn't set it, same convention as every other
    per-skill reward-category scale in this class."""
    result = _run_probe({"HOLOSOMA_SKILLS_CONFIG": str(_SKILLS_YAML)}, _PROBE_CODE)
    for sc in result["skill_reward_scales"]:
        assert sc["root_tracking_reward_scale"] == 1.0


def test_root_tracking_reward_scale_loads_from_skill_yaml(tmp_path):
    """A skill yaml block that DOES set root_tracking_reward_scale must have it reach the resolved
    SkillConfig, index-aligned with its own skill, leaving other skills (and other fields on the
    same skill, notably the still-independent motion_tracking_reward_scale) untouched."""
    import yaml

    raw = yaml.safe_load(_default_two_skill_yaml(tmp_path).read_text())
    skill_keys = [k for k in raw if k.startswith("motion_skill_")]
    assert skill_keys, "fixture yaml must have at least one motion_skill_N block"
    raw[skill_keys[0]]["root_tracking_reward_scale"] = 0.1
    override = tmp_path / "skills_override.yaml"
    override.write_text(yaml.dump(raw))

    result = _run_probe({"HOLOSOMA_SKILLS_CONFIG": str(override)}, _PROBE_CODE)
    scales = result["skill_reward_scales"]
    assert scales[0]["root_tracking_reward_scale"] == 0.1
    assert scales[0]["motion_tracking_reward_scale"] == 1.0  # untouched sibling field
    for sc in scales[1:]:
        assert sc["root_tracking_reward_scale"] == 1.0  # untouched sibling skill


def test_rsi_and_critical_frame_fields_default_to_exact_no_op(tmp_path):
    """MultiSkillConfig.rsi_scope_to_authored_clip / critical_frame_oversampling_prob (2026-08-05)
    must default to their exact-no-op values (False / 0.0) when a skills yaml doesn't set them --
    same convention as every other opt-in global field in this class."""
    result = _run_probe({"HOLOSOMA_SKILLS_CONFIG": str(_default_two_skill_yaml(tmp_path))}, _PROBE_CODE)
    assert result["rsi_scope_to_authored_clip"] is False
    assert result["critical_frame_oversampling_prob"] == 0.0
    assert result["critical_frame_sampling_window"] == 10


def test_rsi_and_critical_frame_fields_load_from_top_level_yaml_keys(tmp_path):
    """These are GLOBAL (not per-skill) fields, unlike kick_alive_pre_kick_ratio/
    root_tracking_reward_scale above -- set at the yaml's TOP LEVEL, mirroring
    start_at_timestep_zero_prob's own placement, and must reach MotionConfig via
    config_values/unified/g1/command.py's dual-path resolution."""
    import yaml

    raw = yaml.safe_load(_default_two_skill_yaml(tmp_path).read_text())
    raw["rsi_scope_to_authored_clip"] = True
    raw["critical_frame_oversampling_prob"] = 0.3
    raw["critical_frame_sampling_window"] = 7
    override = tmp_path / "skills_override.yaml"
    override.write_text(yaml.dump(raw))

    result = _run_probe({"HOLOSOMA_SKILLS_CONFIG": str(override)}, _PROBE_CODE)
    assert result["rsi_scope_to_authored_clip"] is True
    assert result["critical_frame_oversampling_prob"] == 0.3
    assert result["critical_frame_sampling_window"] == 7


# ============================================================================================
# 2026-08-14, user-requested: base_robot_target_height/base_robot_deadzone (single global
# standing-height target/deadzone, read from HOLOSOMA_SKILLS_CONFIG's own top-level `base_robot:`
# block) wired into BOTH penalty_stand_height (locomotion) and penalty_kick_recovery_stand_height
# (kick) -- and the HOLOSOMA_TASK_CONFIG derivation (holosoma/__init__.py) that lets a training
# launch omit that env var entirely when every skill agrees on one `task_config:` value.
# ============================================================================================


def test_base_robot_absent_leaves_both_stand_height_terms_at_their_hardcoded_defaults(tmp_path):
    result = _run_probe({"HOLOSOMA_SKILLS_CONFIG": str(_default_two_skill_yaml(tmp_path))}, _PROBE_CODE)
    assert result["penalty_stand_height_target_height"] == 0.76
    assert result["penalty_stand_height_deadzone"] == 0.015
    assert result["penalty_kick_recovery_stand_height_target_height"] == 0.76
    assert result["penalty_kick_recovery_stand_height_deadzone"] == 0.015


def test_base_robot_present_overrides_both_stand_height_terms_uniformly(tmp_path):
    import yaml

    raw = yaml.safe_load(_SKILLS_YAML.read_text())
    raw["base_robot"] = {"target_height": 0.70, "deadzone": 0.03}
    override = tmp_path / "skills_override.yaml"
    override.write_text(yaml.dump(raw))

    result = _run_probe({"HOLOSOMA_SKILLS_CONFIG": str(override)}, _PROBE_CODE)
    assert result["penalty_stand_height_target_height"] == 0.70
    assert result["penalty_stand_height_deadzone"] == 0.03
    assert result["penalty_kick_recovery_stand_height_target_height"] == 0.70
    assert result["penalty_kick_recovery_stand_height_deadzone"] == 0.03


def test_base_robot_deadzone_takes_priority_over_kick_recovery_stand_height_deadzone(tmp_path):
    """The narrower, pre-existing kick_recovery_stand_height_deadzone field must still work when
    base_robot is absent (test above's default-values check already covers that indirectly), but
    once base_robot.deadzone IS set, it must win -- see MultiSkillConfig.base_robot_deadzone's own
    docstring for the precedence rationale."""
    import yaml

    raw = yaml.safe_load(_SKILLS_YAML.read_text())
    raw["kick_recovery_stand_height_deadzone"] = 0.02  # the OLDER, narrower override
    raw["base_robot"] = {"deadzone": 0.05}  # the NEWER, broader override -- must win
    override = tmp_path / "skills_override.yaml"
    override.write_text(yaml.dump(raw))

    result = _run_probe({"HOLOSOMA_SKILLS_CONFIG": str(override)}, _PROBE_CODE)
    assert result["penalty_kick_recovery_stand_height_deadzone"] == 0.05
    # penalty_stand_height has no OTHER override path at all -- base_robot is its only source,
    # so it's unaffected by kick_recovery_stand_height_deadzone either way.
    assert result["penalty_stand_height_deadzone"] == 0.05


def test_real_skills1_yaml_derives_task_config_without_holosoma_task_config_set():
    """End-to-end proof (real shipped skills1.yaml, real shipped task_config_stageC1.yaml, no
    synthetic fixture) that a training launch can omit HOLOSOMA_TASK_CONFIG entirely and still
    resolve the exact same live reward config as the traditional explicit-env-var launch --
    skills1.yaml's motion_skill_1 declares `task_config: task_config_stageC1`."""
    real_skills1 = _FORK_ROOT / "configs" / "skills1.yaml"
    assert real_skills1.exists(), f"expected {real_skills1} to exist"

    derived = _run_probe({"HOLOSOMA_SKILLS_CONFIG": str(real_skills1)}, _PROBE_CODE)
    explicit = _run_probe(
        {
            "HOLOSOMA_SKILLS_CONFIG": str(real_skills1),
            "HOLOSOMA_TASK_CONFIG": str(_FORK_ROOT / "configs" / "task" / "task_config_stageC1.yaml"),
        },
        _PROBE_CODE,
    )
    assert derived == explicit
    # and specifically: base_robot's own (currently-default-matching) values from skills1.yaml
    # really did get read, not silently skipped.
    assert derived["penalty_stand_height_target_height"] == 0.76


# ============================================================================================
# "Simultaneous per-skill task configs" (2026-08-15) -- _apply_per_skill_reward_weight_overrides.
# Uses the REAL shipped task_config_stageC1.yaml / task_config_stageB.yaml (not synthetic
# fixtures): those two are the actual files this feature was built for.
# ============================================================================================


def _default_two_skill_yaml(tmp_path) -> Path:
    """A minimal, all-defaults 1-file-mode 2-skill yaml (own ``ball:`` block, no ``task_config:``
    anywhere, no per-skill reward-category overrides, no ``base_robot:``) -- the shape
    configs/stageB_and_C.yaml used to provide before it was deleted 2026-08-23 (see
    DEFAULT_MULTI_SKILL_CONFIG_YAML's own comment in multi_skill.py). Every currently-shipped real
    skill/task-config pair is a tuned production config, not a defaults fixture, so a test
    asserting "stays at its hardcoded default" needs this synthetic stand-in instead."""
    p = tmp_path / "skills_defaults.yaml"
    p.write_text(
        "ball: {radius: 0.11, mass: 0.43}\n"
        "motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.3, "
        "strike_start_frame: 10, stand_start_frame: 20}\n"
        "motion_skill_2: {motion_npz: b.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.3, "
        "strike_start_frame: 10, stand_start_frame: 20}\n"
    )
    return p


def _two_skill_yaml(tmp_path, *, top_level_task_config: str | None, skill1_tc: str, skill2_tc: str) -> Path:
    lines = []
    if top_level_task_config is not None:
        lines.append(f"task_config: {top_level_task_config}")
    lines += [
        "motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.3, "
        f"strike_start_frame: 10, stand_start_frame: 20, task_config: {skill1_tc}}}",
        "motion_skill_2: {motion_npz: b.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.3, "
        f"strike_start_frame: 10, stand_start_frame: 20, task_config: {skill2_tc}}}",
    ]
    p = tmp_path / "skills_two_divergent.yaml"
    p.write_text("\n".join(lines) + "\n")
    return p


def test_two_skills_agreeing_on_task_config_is_a_true_no_op(tmp_path):
    """Both skills on task_config_stageC1 -- genuinely no divergence, so NO term should get a
    weight_per_skill table at all (must match the single-skill/legacy behavior exactly)."""
    skills = _two_skill_yaml(
        tmp_path, top_level_task_config=None, skill1_tc="task_config_stageC1", skill2_tc="task_config_stageC1"
    )
    result = _run_probe({"HOLOSOMA_SKILLS_CONFIG": str(skills)}, _PROBE_CODE)
    assert result["skill_task_config_paths"] == [
        str(_FORK_ROOT / "configs" / "task" / "task_config_stageC1.yaml"),
        str(_FORK_ROOT / "configs" / "task" / "task_config_stageC1.yaml"),
    ]
    assert result["kick_alive_weight_per_skill"] is None
    assert result["kick_ball_proximity_weight_per_skill"] is None
    assert result["kick_ball_velocity_weight_per_skill"] is None


def test_two_skills_diverging_requires_top_level_task_config(tmp_path):
    skills = _two_skill_yaml(
        tmp_path, top_level_task_config=None, skill1_tc="task_config_stageC1", skill2_tc="task_config_stageB"
    )
    result = subprocess_probe_expect_failure(skills)
    assert "top-level `task_config:`" in result


def test_two_skills_diverging_with_top_level_produces_correct_per_skill_weights(tmp_path):
    """skill 1 = C1 (shooting/kick_alive live), skill 2 = B (shooting_reward_scale=0,
    kick_alive=0.0) -- the DEFINING properties of Stage B per its own header comment. Both trained
    in the same run; C1 governs every global (non-reward-weight) field via the top-level field."""
    skills = _two_skill_yaml(
        tmp_path,
        top_level_task_config="task_config_stageC1",
        skill1_tc="task_config_stageC1",
        skill2_tc="task_config_stageB",
    )
    result = _run_probe({"HOLOSOMA_SKILLS_CONFIG": str(skills)}, _PROBE_CODE)

    # kick_alive_reward.kick_alive: C1=4.0, B=0.0 (verified live against the real yaml files).
    assert result["kick_alive_weight_per_skill"] == [4.0, 0.0]
    # shooting_reward.kick_ball_proximity / kick_ball_velocity: C1=2.0/10.0, B=0.0/0.0.
    # (kick_ball_velocity was 1.0 until 2026-08-24, raised to 10.0 alongside the first-ever
    # kick_ball_velocity_v_ref: 2.0 -- see task_config_stageC1.yaml's own comment on both. These
    # are deliberately the REAL shipped values, so this assertion is expected to need updating
    # whenever they are retuned; that is the point of the test, not a brittleness bug.)
    assert result["kick_ball_proximity_weight_per_skill"] == [2.0, 0.0]
    assert result["kick_ball_velocity_weight_per_skill"] == [10.0, 0.0]

    # Globals came from the top-level field (C1), not some blend or skill-1-by-accident:
    # confirmed by kick_alive_weight (the plain scalar / skill-0-representative value) matching
    # C1's own value, and by the resolved skill_task_config_paths list itself.
    assert result["skill_task_config_paths"] == [
        str(_FORK_ROOT / "configs" / "task" / "task_config_stageC1.yaml"),
        str(_FORK_ROOT / "configs" / "task" / "task_config_stageB.yaml"),
    ]


def test_top_level_task_config_governs_globals_while_reward_weights_stay_per_skill(tmp_path):
    """penalty_kick_recovery_stand_height / swing-orientation deadzones etc. are GLOBAL fields --
    not per-skill-able today -- so both skills must see C1's values for those even though skill 2
    is nominally "Stage B", while shooting/kick_alive genuinely differ per skill in the same run."""
    skills = _two_skill_yaml(
        tmp_path,
        top_level_task_config="task_config_stageC1",
        skill1_tc="task_config_stageC1",
        skill2_tc="task_config_stageB",
    )
    result = _run_probe({"HOLOSOMA_SKILLS_CONFIG": str(skills)}, _PROBE_CODE)
    explicit_c1_only = _run_probe(
        {
            "HOLOSOMA_SKILLS_CONFIG": str(_FORK_ROOT / "configs" / "skills1.yaml"),
            "HOLOSOMA_TASK_CONFIG": str(_FORK_ROOT / "configs" / "task" / "task_config_stageC1.yaml"),
        },
        _PROBE_CODE,
    )
    assert result["kick_penalty_swing_orientation_deadzone"] == explicit_c1_only["kick_penalty_swing_orientation_deadzone"]
    assert result["motion_strike_dof_pos_weight"] == explicit_c1_only["motion_strike_dof_pos_weight"]
    # but the per-skill reward weight genuinely differs, unlike the pure-C1 single-skill baseline
    assert result["kick_alive_weight_per_skill"] == [4.0, 0.0]


def test_a_skill_with_no_task_config_at_all_inherits_the_global_value(tmp_path):
    """A skill that declares NO task_config: still gets a genuinely per-skill weight table
    wherever its sibling(s) diverge -- inheriting the GLOBAL (top-level) value for every term it
    doesn't itself override, exactly like a missing yaml file already means "no override" for the
    single-file case."""
    lines = [
        "task_config: task_config_stageC1",
        "motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.3, "
        "strike_start_frame: 10, stand_start_frame: 20, task_config: task_config_stageB}",
        # skill 2 declares NO task_config at all
        "motion_skill_2: {motion_npz: b.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.3, "
        "strike_start_frame: 10, stand_start_frame: 20}",
    ]
    skills = tmp_path / "skills_partial.yaml"
    skills.write_text("\n".join(lines) + "\n")

    result = _run_probe({"HOLOSOMA_SKILLS_CONFIG": str(skills)}, _PROBE_CODE)
    # skill 1 (B): kick_alive=0.0; skill 2 (no task_config -> inherits global C1): kick_alive=4.0.
    assert result["kick_alive_weight_per_skill"] == [0.0, 4.0]
    assert result["skill_task_config_paths"] == [str(_FORK_ROOT / "configs" / "task" / "task_config_stageB.yaml"), None]


# ============================================================================================
# Tier 2 "Mechanism A/A'" (2026-08-15): per-skill PARAM tables for stateless reward/termination
# functions -- deadzones, contact-force shape params, balance_potential_weight. Real values
# confirmed directly against the shipped files: kick_recovery_stand_height_deadzone B=0.015/
# C1=0.02 (diverges), kick_recovery_drift_deadzone B=0.15/C1=0.4 (diverges, termination-side),
# kick_contact_force_penalty_floor/_k and balance_potential_weight B=C1 (agree -- exercises the
# "no genuine divergence, stays None" path even though the two skills' files differ overall).
# ============================================================================================


def test_reward_side_deadzone_diverges_between_real_stagec1_and_stageb(tmp_path):
    skills = _two_skill_yaml(
        tmp_path,
        top_level_task_config="task_config_stageC1",
        skill1_tc="task_config_stageC1",
        skill2_tc="task_config_stageB",
    )
    result = _run_probe({"HOLOSOMA_SKILLS_CONFIG": str(skills)}, _PROBE_CODE)
    assert result["kick_recovery_stand_height_deadzone_params_per_skill"] == {"deadzone": [0.02, 0.015]}


def test_reward_side_param_with_no_real_divergence_stays_none(tmp_path):
    """kick_swing_orientation_deadzone: Stage B doesn't declare it at all, so it inherits the
    GLOBAL (C1) value -- both skills resolve to the SAME 0.44, so no per-skill table should be
    built despite the two skills using genuinely different task_config files overall."""
    skills = _two_skill_yaml(
        tmp_path,
        top_level_task_config="task_config_stageC1",
        skill1_tc="task_config_stageC1",
        skill2_tc="task_config_stageB",
    )
    result = _run_probe({"HOLOSOMA_SKILLS_CONFIG": str(skills)}, _PROBE_CODE)
    assert result["kick_penalty_swing_orientation_params_per_skill"] is None


def test_contact_force_and_balance_potential_agree_between_stagec1_and_stageb(tmp_path):
    """Both fields happen to share the SAME value in the two real files -- proves the mechanism
    doesn't force a table just because the skills' files differ elsewhere."""
    skills = _two_skill_yaml(
        tmp_path,
        top_level_task_config="task_config_stageC1",
        skill1_tc="task_config_stageC1",
        skill2_tc="task_config_stageB",
    )
    result = _run_probe({"HOLOSOMA_SKILLS_CONFIG": str(skills)}, _PROBE_CODE)
    assert result["kick_penalty_excess_contact_force_params_per_skill"] is None
    assert result["kick_balance_potential_weight_per_skill"] is None
    assert result["kick_balance_potential_registered"] is True  # global weight (1.0) alone registers it


def test_termination_side_deadzone_diverges_between_real_stagec1_and_stageb(tmp_path):
    skills = _two_skill_yaml(
        tmp_path,
        top_level_task_config="task_config_stageC1",
        skill1_tc="task_config_stageC1",
        skill2_tc="task_config_stageB",
    )
    result = _run_probe({"HOLOSOMA_SKILLS_CONFIG": str(skills)}, _PROBE_CODE)
    # "enabled" joins "deadzone" here because kick_recovery_termination_handoff ALSO genuinely
    # diverges between these same two real files (C1=true/B=false, see the Tier 3 Group B tests
    # below) -- both keys live on the same term's params_per_skill dict.
    assert result["kick_recovery_drift_params_per_skill"] == {"deadzone": [0.4, 0.15], "enabled": [1.0, 0.0]}


# ============================================================================================
# Tier 3 Group B Wave 1 (2026-08-15): kick_recovery_termination_handoff's REGISTRATION (per-term,
# reacts the instant ANY skill wants it -- see termination.py's own comment above
# _kick_recovery_termination_handoff_active) plus its per-skill `enabled` PARAM (suppresses the
# effect per-env for skills that don't want it, via the ordinary params_per_skill gather). Real
# values confirmed directly against the shipped files: kick_recovery_termination_handoff
# C1=true/B=false (diverges) -- the same pairing already proven to diverge on
# kick_recovery_drift_deadzone above, now ALSO exercising the registration-gate + enabled-param
# mechanism on the SAME two terms (kick_recovery_low_height, kick_recovery_drift).
# ============================================================================================


def test_kick_recovery_termination_handoff_registers_when_either_skill_wants_it(tmp_path):
    """C1 wants it (true), B doesn't (false) -- registration must fire because ANY skill is
    active, even though the GLOBAL/base value alone (whichever skill's file happens to be the
    top-level default) would be ambiguous about it."""
    skills = _two_skill_yaml(
        tmp_path,
        top_level_task_config="task_config_stageC1",
        skill1_tc="task_config_stageC1",
        skill2_tc="task_config_stageB",
    )
    result = _run_probe({"HOLOSOMA_SKILLS_CONFIG": str(skills)}, _PROBE_CODE)
    assert result["kick_recovery_low_height_registered"] is True
    assert result["kick_recovery_drift_registered"] is True
    assert result["kick_recovery_low_height_params_per_skill"] == {"enabled": [1.0, 0.0]}


def test_kick_recovery_termination_handoff_registers_off_when_top_level_is_stageb(tmp_path):
    """Sanity check on the OTHER direction: even with B as the top-level/global default (handoff
    off), the mechanism must still register because skill 1 (C1) individually wants it -- proves
    the "ANY skill active" gate isn't secretly just reading the top-level/global value."""
    skills = _two_skill_yaml(
        tmp_path,
        top_level_task_config="task_config_stageB",
        skill1_tc="task_config_stageC1",
        skill2_tc="task_config_stageB",
    )
    result = _run_probe({"HOLOSOMA_SKILLS_CONFIG": str(skills)}, _PROBE_CODE)
    assert result["kick_recovery_low_height_registered"] is True
    assert result["kick_recovery_drift_registered"] is True
    assert result["kick_recovery_low_height_params_per_skill"] == {"enabled": [1.0, 0.0]}


def test_kick_recovery_termination_handoff_both_off_never_registers(tmp_path):
    skills = _two_skill_yaml(
        tmp_path,
        top_level_task_config="task_config_stageB",
        skill1_tc="task_config_stageB",
        skill2_tc="task_config_stageB",
    )
    result = _run_probe({"HOLOSOMA_SKILLS_CONFIG": str(skills)}, _PROBE_CODE)
    assert result["kick_recovery_low_height_registered"] is False
    assert result["kick_recovery_drift_registered"] is False
    assert result["kick_recovery_low_height_params_per_skill"] is None


# ============================================================================================
# Tier 3 Group B Wave 1 (2026-08-15): post_flip_termination_grace_steps/pre_kick_termination_
# grace_steps/post_flip_reward_decay_steps/pre_kick_reward_ramp_steps -- Mechanism A per-skill
# param tables reusing the SAME tensor-support fix applied to _post_flip_grace_active/
# _pre_kick_grace_active/_post_flip_tracking_decay_multiplier/_pre_kick_reward_ramp_multiplier.
# Real values: C1 sets post_flip_termination_grace_steps=50.0/post_flip_reward_decay_steps=50.0
# explicitly; B doesn't set either (inherits the top-level/base value) -- so with C1 as the
# top-level default, both resolve to the SAME 50.0 for both skills -- a genuine "files differ
# overall but this field doesn't" case, exactly like Group A's start_at_timestep_zero_prob.
# ============================================================================================


def test_grace_and_decay_fields_have_no_real_divergence_between_stagec1_and_stageb(tmp_path):
    skills = _two_skill_yaml(
        tmp_path,
        top_level_task_config="task_config_stageC1",
        skill1_tc="task_config_stageC1",
        skill2_tc="task_config_stageB",
    )
    result = _run_probe({"HOLOSOMA_SKILLS_CONFIG": str(skills)}, _PROBE_CODE)
    assert result["low_height_params_per_skill"] is None
    assert result["contact_params_per_skill"] is None
    assert result["kick_low_height_params_per_skill"] is None
    assert result["motion_tracking_params_per_skill"] is None
    # bad_motion_body_pos_threshold (C1=0.35/B=0.25) already diverges between these two real
    # files for an unrelated (Tier 2, pre-existing) reason -- this test only proves the NEW
    # pre_kick_grace_steps key stays absent from bad_tracking's dict (no real divergence on it
    # between C1/B, same as the other grace/decay fields above), not that the dict is empty.
    assert result["bad_tracking_params_per_skill"] == {"bad_motion_body_pos_threshold": [0.35, 0.25]}


def test_post_flip_and_pre_kick_fields_diverge_with_synthetic_two_skill_override(tmp_path):
    """Synthetic divergence (neither real file disagrees on these 4 fields, proven above) --
    proves the mechanism actually builds a genuine per-skill table when skills DO disagree, using
    the same _two_skill_yaml two-skill harness but with a hand-built task_config pair."""
    import yaml

    stagec1 = yaml.safe_load((_FORK_ROOT / "configs" / "task" / "task_config_stageC1.yaml").read_text())
    stagec1["post_flip_termination_grace_steps"] = 80.0
    stagec1["pre_kick_termination_grace_steps"] = 30.0
    stagec1["post_flip_reward_decay_steps"] = 80.0
    stagec1["pre_kick_reward_ramp_steps"] = 30.0
    synthetic = _FORK_ROOT / "configs" / "task" / "task_config_stagec1_synthetic_grace_override.yaml"
    synthetic.write_text(yaml.dump(stagec1))
    try:
        # top-level task_config stays the REAL, unmodified C1 -- it governs the GLOBAL/base_value
        # every non-overriding skill inherits (post_flip=50.0/50.0, pre_kick=0.0/0.0 defaults).
        # Only skill 1 points at the synthetic override; skill 2 stays on plain C1, so it inherits
        # those exact base values -- genuine divergence against skill 1's explicit 80.0/30.0s.
        skills = _two_skill_yaml(
            tmp_path,
            top_level_task_config="task_config_stageC1",
            skill1_tc="task_config_stagec1_synthetic_grace_override",
            skill2_tc="task_config_stageC1",
        )
        result = _run_probe({"HOLOSOMA_SKILLS_CONFIG": str(skills)}, _PROBE_CODE)
        assert result["low_height_params_per_skill"] == {"post_flip_grace_steps": [80.0, 50.0]}
        assert result["contact_params_per_skill"] == {"post_flip_grace_steps": [80.0, 50.0]}
        assert result["kick_low_height_params_per_skill"] == {"pre_kick_grace_steps": [30.0, 0.0]}
        assert result["motion_tracking_params_per_skill"] == {
            "post_flip_reward_decay_steps": [80.0, 50.0],
            "pre_kick_reward_ramp_steps": [30.0, 0.0],
        }
        # bad_motion_body_pos_threshold and kick_recovery_termination_handoff both agree between
        # skill 1 (synthetic, copied verbatim from C1 except for the 4 grace/decay fields) and
        # skill 2 (plain C1) -- true no-op for those two keys, so only pre_kick_grace_steps
        # (the field this test actually varies) shows up in the dict.
        assert result["bad_tracking_params_per_skill"] == {"pre_kick_grace_steps": [30.0, 0.0]}
    finally:
        synthetic.unlink()


def test_single_skill_mode_has_no_params_per_skill_anywhere():
    real_skills1 = _FORK_ROOT / "configs" / "skills1.yaml"
    result = _run_probe({"HOLOSOMA_SKILLS_CONFIG": str(real_skills1)}, _PROBE_CODE)
    assert result["kick_recovery_stand_height_deadzone_params_per_skill"] is None
    assert result["kick_penalty_swing_orientation_params_per_skill"] is None
    assert result["kick_penalty_excess_contact_force_params_per_skill"] is None
    assert result["kick_balance_potential_weight_per_skill"] is None
    assert result["kick_recovery_drift_params_per_skill"] is None
    assert result["kick_penalty_ee_body_pos_divergence_params_per_skill"] is None
    assert result["bad_tracking_params_per_skill"] is None
    assert result["kick_foot_strike_pitch_params_per_skill"] is None
    assert result["kick_ball_over_line_params_per_skill"] is None
    # skills1.yaml has exactly 1 skill (still N-skill mode -- the splat block always builds this
    # list once _multi_skill_cfg is not None), so it broadcasts to a length-1 list, not [] --
    # [] is the TRUE legacy (HOLOSOMA_SKILLS_CONFIG unset) no-op, covered separately below.
    assert result["motion_head_velocity_smoothing_frames_per_motion"] == [
        result["motion_head_velocity_smoothing_frames"]
    ]


# ============================================================================================
# "Mechanism B" (2026-08-15): BadTracking (STATEFUL) reads params_per_skill directly, kept in
# sync with the reward-side stateless sibling kick_penalty_ee_body_pos_divergence's "threshold".
# Real values confirmed against the shipped files: bad_motion_body_pos_threshold B=0.25/C1=0.35
# (diverges), ee_body_pos_warmup_threshold B=0.25/C1=0.7 (diverges, reward-side only -- not
# synced with anything), bad_tracking_swing_threshold_multiplier B=C1=1.0 (agrees).
# ============================================================================================


def test_bad_motion_body_pos_threshold_stays_synced_between_termination_and_reward(tmp_path):
    skills = _two_skill_yaml(
        tmp_path,
        top_level_task_config="task_config_stageC1",
        skill1_tc="task_config_stageC1",
        skill2_tc="task_config_stageB",
    )
    result = _run_probe({"HOLOSOMA_SKILLS_CONFIG": str(skills)}, _PROBE_CODE)
    assert result["bad_tracking_params_per_skill"] == {"bad_motion_body_pos_threshold": [0.35, 0.25]}
    assert result["kick_penalty_ee_body_pos_divergence_params_per_skill"] == {
        "threshold": [0.35, 0.25],
        "warmup_threshold": [0.7, 0.25],
    }


def test_swing_threshold_multiplier_has_no_divergence_between_stagec1_and_stageb(tmp_path):
    """Both real files set this to 1.0 -- must stay out of bad_tracking's params_per_skill
    entirely, proving the mechanism doesn't force a table just because OTHER fields diverge."""
    skills = _two_skill_yaml(
        tmp_path,
        top_level_task_config="task_config_stageC1",
        skill1_tc="task_config_stageC1",
        skill2_tc="task_config_stageB",
    )
    result = _run_probe({"HOLOSOMA_SKILLS_CONFIG": str(skills)}, _PROBE_CODE)
    assert "swing_threshold_multiplier" not in result["bad_tracking_params_per_skill"]


# ============================================================================================
# "Mechanism C" (2026-08-15): motion_head_velocity_smoothing_frames per-skill override --
# SkillConfig field (declared directly on motion_skill_N), NOT resolved via task_config files,
# since this is per-CLIP content. task_config_stageC1.yaml sets the GLOBAL scalar to 3, confirmed
# directly against the shipped file.
# ============================================================================================


def test_motion_head_velocity_smoothing_frames_per_skill_override_reaches_motion_config(tmp_path):
    lines = [
        "task_config: task_config_stageC1",
        "motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.3, "
        "strike_start_frame: 10, stand_start_frame: 20, task_config: task_config_stageC1, "
        "motion_head_velocity_smoothing_frames: 7}",
        "motion_skill_2: {motion_npz: b.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.3, "
        "strike_start_frame: 10, stand_start_frame: 20, task_config: task_config_stageB}",
    ]
    skills = tmp_path / "skills_mechc.yaml"
    skills.write_text("\n".join(lines) + "\n")

    result = _run_probe({"HOLOSOMA_SKILLS_CONFIG": str(skills)}, _PROBE_CODE)
    assert result["motion_head_velocity_smoothing_frames"] == 3  # the real global, confirmed against the shipped file
    # skill 1's explicit override (7) vs skill 2's inherited global (3).
    assert result["motion_head_velocity_smoothing_frames_per_motion"] == [7, 3]


def test_motion_head_velocity_smoothing_frames_per_motion_empty_when_no_skill_overrides(tmp_path):
    skills = _two_skill_yaml(
        tmp_path,
        top_level_task_config="task_config_stageC1",
        skill1_tc="task_config_stageC1",
        skill2_tc="task_config_stageB",
    )
    result = _run_probe({"HOLOSOMA_SKILLS_CONFIG": str(skills)}, _PROBE_CODE)
    # Neither skill overrides -- both broadcast the same global (3), by the pre-existing
    # _resolve_per_motion_durations contract (non-empty, uniform, NOT an empty-list no-op --
    # broadcasting only ever collapses to an empty list in true single-skill/legacy mode).
    assert result["motion_head_velocity_smoothing_frames_per_motion"] == [3, 3]


# ============================================================================================
# Tier 3 "trio" (2026-08-15): the 3 reward/termination-param fields, reusing existing Mechanism
# A/B infrastructure directly. Real values confirmed against the shipped files:
# kick_ball_over_line_require_has_kicked B=false/C1=true (diverges); bad_tracking_swing_only
# B=C1=false (agrees); use_foot_strike_pitch_reference_relative C1=true/B unset->inherits C1's
# global True (agrees, since B has no override of its own) -- the last two need a synthetic
# override to exercise genuine divergence, same as Mechanism B's swing_threshold_multiplier did.
# ============================================================================================


def test_kick_ball_over_line_require_has_kicked_diverges_between_real_stagec1_and_stageb(tmp_path):
    skills = _two_skill_yaml(
        tmp_path,
        top_level_task_config="task_config_stageC1",
        skill1_tc="task_config_stageC1",
        skill2_tc="task_config_stageB",
    )
    result = _run_probe({"HOLOSOMA_SKILLS_CONFIG": str(skills)}, _PROBE_CODE)
    assert result["kick_ball_over_line_params_per_skill"] == {"require_has_kicked": [1.0, 0.0]}


def test_bad_tracking_swing_only_and_reference_relative_have_no_real_divergence(tmp_path):
    """bad_tracking_swing_only agrees (both False); use_foot_strike_pitch_reference_relative
    agrees too (B doesn't set it at all, so it inherits C1's own global True) -- neither should
    produce a per-skill table despite the two skills using genuinely different task_config files
    overall (kick_ball_over_line_require_has_kicked, tested above, DOES diverge in this same
    pairing)."""
    skills = _two_skill_yaml(
        tmp_path,
        top_level_task_config="task_config_stageC1",
        skill1_tc="task_config_stageC1",
        skill2_tc="task_config_stageB",
    )
    result = _run_probe({"HOLOSOMA_SKILLS_CONFIG": str(skills)}, _PROBE_CODE)
    assert "bad_tracking_swing_only" not in (result["bad_tracking_params_per_skill"] or {})
    assert result["kick_foot_strike_pitch_params_per_skill"] is None


# ============================================================================================
# Tier 3 "Group A" (2026-08-15): 4 RSI/reset-time sampling fields on MotionConfig -- resolved
# from each skill's own task_config file (a training-regime choice, not per-clip content, unlike
# motion_head_velocity_smoothing_frames). rsi_scope_to_authored_clip B=false/C1=true (diverges
# for real); the other 3 agree between B/C1 (need a synthetic override to exercise divergence).
# mid_episode_kick_entry_ball_fixed (the 5th RSI-group field) deliberately has no per-skill
# counterpart -- see MotionConfig's own docstring for why (a shared table's column count, not a
# per-motion value) -- so it isn't probed here.
# ============================================================================================


def test_rsi_scope_to_authored_clip_diverges_between_real_stagec1_and_stageb(tmp_path):
    skills = _two_skill_yaml(
        tmp_path,
        top_level_task_config="task_config_stageC1",
        skill1_tc="task_config_stageC1",
        skill2_tc="task_config_stageB",
    )
    result = _run_probe({"HOLOSOMA_SKILLS_CONFIG": str(skills)}, _PROBE_CODE)
    assert result["rsi_scope_to_authored_clip_per_motion"] == [True, False]


def test_start_at_timestep_zero_prob_and_critical_frame_fields_have_no_real_divergence(tmp_path):
    """All 3 agree between the real B/C1 files -- must stay empty despite
    rsi_scope_to_authored_clip (tested above, same pairing) genuinely diverging."""
    skills = _two_skill_yaml(
        tmp_path,
        top_level_task_config="task_config_stageC1",
        skill1_tc="task_config_stageC1",
        skill2_tc="task_config_stageB",
    )
    result = _run_probe({"HOLOSOMA_SKILLS_CONFIG": str(skills)}, _PROBE_CODE)
    assert result["start_at_timestep_zero_prob_per_motion"] == []
    assert result["critical_frame_oversampling_prob_per_motion"] == []
    assert result["critical_frame_sampling_window_per_motion"] == []


def subprocess_probe_expect_failure(skills_path: Path) -> str:
    """Like _run_probe, but for the case where the probe process is expected to raise (skills
    disagree, no top-level task_config) -- returns combined stderr instead of asserting success."""
    import os

    env = dict(os.environ)
    env["HOLOSOMA_SKILLS_CONFIG"] = str(skills_path)
    result = subprocess.run(
        [sys.executable, "-c", _PROBE_CODE],
        cwd=str(_HOLOSOMA_DIR),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0, f"expected the probe to fail, but it succeeded:\n{result.stdout}"
    return result.stderr
    assert derived["penalty_stand_height_deadzone"] == 0.015
