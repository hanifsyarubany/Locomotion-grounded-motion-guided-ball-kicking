"""Unit tests for the legacy (single-skill) path of the 2026-07-24 ball perception noise/latency
fields: BallConfig.observation_noise/observation_noise_range_coefficient/
observation_delay_steps_{min,max}, and load_ball_config's parsing of the matching optional yaml
keys -- the configs/ball.yaml-side counterpart to load_multi_skill_config's ball_obs_* fields
(see config_types/tests/test_multi_skill.py).
"""

from __future__ import annotations

import pytest

from holosoma.config_types.simulator import load_ball_config


def _write(tmp_path, contents: str):
    p = tmp_path / "ball.yaml"
    p.write_text(contents)
    return p


def test_observation_noise_fields_default_to_pre_2026_07_24_hardcoded_values_when_absent(tmp_path):
    """A ball.yaml with none of the new keys must reproduce the values that were hardcoded
    directly in config_values/unified/g1/observation.py before these fields existed."""
    p = _write(tmp_path, "radius: 0.11\nmass: 0.43\nx: 1.5\ny: 0.0\n")
    cfg = load_ball_config(p)
    assert cfg.observation_noise == 0.05
    assert cfg.observation_noise_range_coefficient == 0.03
    assert cfg.observation_delay_steps_min == 0
    assert cfg.observation_delay_steps_max == 3


def test_observation_noise_fields_parsed_when_present(tmp_path):
    p = _write(
        tmp_path,
        "radius: 0.11\nmass: 0.43\nx: 1.5\ny: 0.0\n"
        "observation_noise: 0.08\n"
        "observation_noise_range_coefficient: 0.02\n"
        "observation_delay_steps_min: 1\n"
        "observation_delay_steps_max: 4\n",
    )
    cfg = load_ball_config(p)
    assert cfg.observation_noise == 0.08
    assert cfg.observation_noise_range_coefficient == 0.02
    assert cfg.observation_delay_steps_min == 1
    assert cfg.observation_delay_steps_max == 4
    # int-cast keys must actually be ints, not floats leaking through
    assert isinstance(cfg.observation_delay_steps_min, int)
    assert isinstance(cfg.observation_delay_steps_max, int)


def test_observation_hold_steps_default_to_off_when_absent(tmp_path):
    p = _write(tmp_path, "radius: 0.11\nmass: 0.43\nx: 1.5\ny: 0.0\n")
    cfg = load_ball_config(p)
    assert cfg.observation_hold_steps_min == 0
    assert cfg.observation_hold_steps_max == 0


def test_observation_hold_steps_parsed_when_present(tmp_path):
    p = _write(
        tmp_path,
        "radius: 0.11\nmass: 0.43\nx: 1.5\ny: 0.0\n"
        "observation_hold_steps_min: 2\n"
        "observation_hold_steps_max: 4\n",
    )
    cfg = load_ball_config(p)
    assert cfg.observation_hold_steps_min == 2
    assert cfg.observation_hold_steps_max == 4
    assert isinstance(cfg.observation_hold_steps_min, int)
    assert isinstance(cfg.observation_hold_steps_max, int)


def test_observation_hold_steps_max_less_than_min_raises(tmp_path):
    p = _write(
        tmp_path,
        "radius: 0.11\nmass: 0.43\nx: 1.5\ny: 0.0\n"
        "observation_hold_steps_min: 5\n"
        "observation_hold_steps_max: 2\n",
    )
    with pytest.raises(ValueError, match="must be >="):
        load_ball_config(p)


def test_observation_stale_probability_defaults_to_off_when_absent(tmp_path):
    p = _write(tmp_path, "radius: 0.11\nmass: 0.43\nx: 1.5\ny: 0.0\n")
    cfg = load_ball_config(p)
    assert cfg.observation_stale_probability == 0.0


def test_observation_stale_probability_parsed_when_present(tmp_path):
    p = _write(tmp_path, "radius: 0.11\nmass: 0.43\nx: 1.5\ny: 0.0\nobservation_stale_probability: 0.01\n")
    cfg = load_ball_config(p)
    assert cfg.observation_stale_probability == 0.01


def test_observation_stale_probability_out_of_range_raises(tmp_path):
    p = _write(tmp_path, "radius: 0.11\nmass: 0.43\nx: 1.5\ny: 0.0\nobservation_stale_probability: 1.5\n")
    with pytest.raises(ValueError, match=r"must be in \[0.0, 1.0\]"):
        load_ball_config(p)


def test_kick_contact_force_penalty_fields_default_to_pre_2026_07_28_hardcoded_values_when_absent(tmp_path):
    """Legacy-path counterpart of test_multi_skill.py's equivalent test -- a ball.yaml with none
    of the new keys must reproduce the values that were hardcoded directly in
    config_values/unified/g1/reward.py / kick_safety.py before these fields existed."""
    p = _write(tmp_path, "radius: 0.11\nmass: 0.43\nx: 1.5\ny: 0.0\n")
    cfg = load_ball_config(p)
    assert cfg.kick_contact_force_penalty_floor == 3.0
    assert cfg.kick_contact_force_penalty_k == 15.0
    assert cfg.kick_contact_force_threshold_bodyweight_multiplier == 3.0


def test_kick_contact_force_penalty_fields_parsed_when_present(tmp_path):
    p = _write(
        tmp_path,
        "radius: 0.11\nmass: 0.43\nx: 1.5\ny: 0.0\n"
        "kick_contact_force_penalty_floor: 2.0\n"
        "kick_contact_force_penalty_k: 8.0\n"
        "kick_contact_force_threshold_bodyweight_multiplier: 4.0\n",
    )
    cfg = load_ball_config(p)
    assert cfg.kick_contact_force_penalty_floor == 2.0
    assert cfg.kick_contact_force_penalty_k == 8.0
    assert cfg.kick_contact_force_threshold_bodyweight_multiplier == 4.0


def test_start_at_timestep_zero_prob_defaults_to_pre_2026_07_28_hardcoded_value_when_absent(tmp_path):
    p = _write(tmp_path, "radius: 0.11\nmass: 0.43\nx: 1.5\ny: 0.0\n")
    cfg = load_ball_config(p)
    assert cfg.start_at_timestep_zero_prob == 1.0


def test_start_at_timestep_zero_prob_parsed_when_present(tmp_path):
    p = _write(tmp_path, "radius: 0.11\nmass: 0.43\nx: 1.5\ny: 0.0\nstart_at_timestep_zero_prob: 0.5\n")
    cfg = load_ball_config(p)
    assert cfg.start_at_timestep_zero_prob == 0.5


def test_start_at_timestep_zero_prob_out_of_range_raises(tmp_path):
    p = _write(tmp_path, "radius: 0.11\nmass: 0.43\nx: 1.5\ny: 0.0\nstart_at_timestep_zero_prob: -0.1\n")
    with pytest.raises(ValueError, match=r"must be in \[0\.0, 1\.0\]"):
        load_ball_config(p)


def test_kick_target_entropy_ratio_defaults_to_none_when_absent(tmp_path):
    p = _write(tmp_path, "radius: 0.11\nmass: 0.43\nx: 1.5\ny: 0.0\n")
    cfg = load_ball_config(p)
    assert cfg.kick_target_entropy_ratio is None


def test_kick_target_entropy_ratio_parsed_when_present(tmp_path):
    p = _write(tmp_path, "radius: 0.11\nmass: 0.43\nx: 1.5\ny: 0.0\nkick_target_entropy_ratio: 0.3\n")
    cfg = load_ball_config(p)
    assert cfg.kick_target_entropy_ratio == 0.3


def test_kick_target_entropy_ratio_of_zero_is_distinct_from_unset(tmp_path):
    p = _write(tmp_path, "radius: 0.11\nmass: 0.43\nx: 1.5\ny: 0.0\nkick_target_entropy_ratio: 0.0\n")
    cfg = load_ball_config(p)
    assert cfg.kick_target_entropy_ratio == 0.0
    assert cfg.kick_target_entropy_ratio is not None


def test_kick_gamma_defaults_to_none_when_absent(tmp_path):
    p = _write(tmp_path, "radius: 0.11\nmass: 0.43\nx: 1.5\ny: 0.0\n")
    cfg = load_ball_config(p)
    assert cfg.kick_gamma is None


def test_kick_gamma_parsed_when_present(tmp_path):
    p = _write(tmp_path, "radius: 0.11\nmass: 0.43\nx: 1.5\ny: 0.0\nkick_gamma: 0.99\n")
    cfg = load_ball_config(p)
    assert cfg.kick_gamma == 0.99


@pytest.mark.parametrize("bad_value", [0.0, 1.0, 1.5, -0.1])
def test_kick_gamma_out_of_range_raises(tmp_path, bad_value):
    p = _write(tmp_path, f"radius: 0.11\nmass: 0.43\nx: 1.5\ny: 0.0\nkick_gamma: {bad_value}\n")
    with pytest.raises(ValueError, match=r"kick_gamma must be in \(0\.0, 1\.0\)"):
        load_ball_config(p)


def test_kick_recovery_drift_deadzone_defaults_to_15cm_when_absent(tmp_path):
    p = _write(tmp_path, "radius: 0.11\nmass: 0.43\nx: 1.5\ny: 0.0\n")
    cfg = load_ball_config(p)
    assert cfg.kick_recovery_drift_deadzone == 0.15


def test_kick_recovery_drift_deadzone_parsed_when_present(tmp_path):
    p = _write(tmp_path, "radius: 0.11\nmass: 0.43\nx: 1.5\ny: 0.0\nkick_recovery_drift_deadzone: 0.25\n")
    cfg = load_ball_config(p)
    assert cfg.kick_recovery_drift_deadzone == 0.25


def test_kick_recovery_drift_deadzone_rejects_non_positive(tmp_path):
    p = _write(tmp_path, "radius: 0.11\nmass: 0.43\nx: 1.5\ny: 0.0\nkick_recovery_drift_deadzone: 0.0\n")
    with pytest.raises(ValueError, match="kick_recovery_drift_deadzone must be > 0.0"):
        load_ball_config(p)


@pytest.mark.parametrize(
    "field_name,default_value",
    [
        ("kick_recovery_stand_height_deadzone", 0.015),
        ("kick_recovery_stand_orientation_deadzone", 0.025),
        ("kick_recovery_stand_feet_width_deadzone", 0.03),
        ("kick_recovery_stand_knee_width_deadzone", 0.03),
    ],
)
def test_kick_recovery_posture_deadzone_defaults_when_absent(tmp_path, field_name, default_value):
    p = _write(tmp_path, "radius: 0.11\nmass: 0.43\nx: 1.5\ny: 0.0\n")
    cfg = load_ball_config(p)
    assert getattr(cfg, field_name) == default_value


@pytest.mark.parametrize(
    "field_name",
    [
        "kick_recovery_stand_height_deadzone",
        "kick_recovery_stand_orientation_deadzone",
        "kick_recovery_stand_feet_width_deadzone",
        "kick_recovery_stand_knee_width_deadzone",
    ],
)
def test_kick_recovery_posture_deadzone_parsed_when_present(tmp_path, field_name):
    p = _write(tmp_path, f"radius: 0.11\nmass: 0.43\nx: 1.5\ny: 0.0\n{field_name}: 0.5\n")
    cfg = load_ball_config(p)
    assert getattr(cfg, field_name) == 0.5


@pytest.mark.parametrize(
    "field_name",
    [
        "kick_recovery_stand_height_deadzone",
        "kick_recovery_stand_orientation_deadzone",
        "kick_recovery_stand_feet_width_deadzone",
        "kick_recovery_stand_knee_width_deadzone",
    ],
)
def test_kick_recovery_posture_deadzone_accepts_zero(tmp_path, field_name):
    p = _write(tmp_path, f"radius: 0.11\nmass: 0.43\nx: 1.5\ny: 0.0\n{field_name}: 0.0\n")
    cfg = load_ball_config(p)
    assert getattr(cfg, field_name) == 0.0


@pytest.mark.parametrize(
    "field_name",
    [
        "kick_recovery_stand_height_deadzone",
        "kick_recovery_stand_orientation_deadzone",
        "kick_recovery_stand_feet_width_deadzone",
        "kick_recovery_stand_knee_width_deadzone",
    ],
)
def test_kick_recovery_posture_deadzone_rejects_negative(tmp_path, field_name):
    p = _write(tmp_path, f"radius: 0.11\nmass: 0.43\nx: 1.5\ny: 0.0\n{field_name}: -0.01\n")
    with pytest.raises(ValueError, match=f"{field_name} must be >= 0.0"):
        load_ball_config(p)


def test_kick_ball_over_line_require_has_kicked_defaults_false_when_absent(tmp_path):
    p = _write(tmp_path, "radius: 0.11\nmass: 0.43\nx: 1.5\ny: 0.0\n")
    cfg = load_ball_config(p)
    assert cfg.kick_ball_over_line_require_has_kicked is False


def test_kick_ball_over_line_require_has_kicked_parsed_when_present(tmp_path):
    p = _write(tmp_path, "radius: 0.11\nmass: 0.43\nx: 1.5\ny: 0.0\nkick_ball_over_line_require_has_kicked: true\n")
    cfg = load_ball_config(p)
    assert cfg.kick_ball_over_line_require_has_kicked is True


def test_penalty_curriculum_enabled_defaults_true_when_absent(tmp_path):
    p = _write(tmp_path, "radius: 0.11\nmass: 0.43\nx: 1.5\ny: 0.0\n")
    cfg = load_ball_config(p)
    assert cfg.penalty_curriculum_enabled is True


def test_penalty_curriculum_enabled_parsed_when_present(tmp_path):
    p = _write(tmp_path, "radius: 0.11\nmass: 0.43\nx: 1.5\ny: 0.0\npenalty_curriculum_enabled: false\n")
    cfg = load_ball_config(p)
    assert cfg.penalty_curriculum_enabled is False


def test_post_flip_termination_grace_steps_defaults_zero_when_absent(tmp_path):
    p = _write(tmp_path, "radius: 0.11\nmass: 0.43\nx: 1.5\ny: 0.0\n")
    cfg = load_ball_config(p)
    assert cfg.post_flip_termination_grace_steps == 0.0


def test_post_flip_termination_grace_steps_parsed_when_present(tmp_path):
    p = _write(tmp_path, "radius: 0.11\nmass: 0.43\nx: 1.5\ny: 0.0\npost_flip_termination_grace_steps: 50.0\n")
    cfg = load_ball_config(p)
    assert cfg.post_flip_termination_grace_steps == 50.0


def test_post_flip_termination_grace_steps_negative_raises(tmp_path):
    p = _write(tmp_path, "radius: 0.11\nmass: 0.43\nx: 1.5\ny: 0.0\npost_flip_termination_grace_steps: -1.0\n")
    with pytest.raises(ValueError, match="post_flip_termination_grace_steps must be >= 0.0"):
        load_ball_config(p)


def test_post_flip_reward_decay_steps_defaults_zero_when_absent(tmp_path):
    p = _write(tmp_path, "radius: 0.11\nmass: 0.43\nx: 1.5\ny: 0.0\n")
    cfg = load_ball_config(p)
    assert cfg.post_flip_reward_decay_steps == 0.0


def test_post_flip_reward_decay_steps_parsed_when_present(tmp_path):
    p = _write(tmp_path, "radius: 0.11\nmass: 0.43\nx: 1.5\ny: 0.0\npost_flip_reward_decay_steps: 50.0\n")
    cfg = load_ball_config(p)
    assert cfg.post_flip_reward_decay_steps == 50.0


def test_post_flip_reward_decay_steps_negative_raises(tmp_path):
    p = _write(tmp_path, "radius: 0.11\nmass: 0.43\nx: 1.5\ny: 0.0\npost_flip_reward_decay_steps: -1.0\n")
    with pytest.raises(ValueError, match="post_flip_reward_decay_steps must be >= 0.0"):
        load_ball_config(p)


@pytest.mark.parametrize("field_name", ["kick_swing_orientation_deadzone", "kick_swing_torso_orientation_deadzone"])
def test_kick_swing_orientation_deadzone_defaults_zero_when_absent(tmp_path, field_name):
    p = _write(tmp_path, "radius: 0.11\nmass: 0.43\nx: 1.5\ny: 0.0\n")
    cfg = load_ball_config(p)
    assert getattr(cfg, field_name) == 0.0


@pytest.mark.parametrize("field_name", ["kick_swing_orientation_deadzone", "kick_swing_torso_orientation_deadzone"])
def test_kick_swing_orientation_deadzone_parsed_when_present(tmp_path, field_name):
    p = _write(tmp_path, f"radius: 0.11\nmass: 0.43\nx: 1.5\ny: 0.0\n{field_name}: 0.55\n")
    cfg = load_ball_config(p)
    assert getattr(cfg, field_name) == 0.55


@pytest.mark.parametrize("field_name", ["kick_swing_orientation_deadzone", "kick_swing_torso_orientation_deadzone"])
def test_kick_swing_orientation_deadzone_rejects_negative(tmp_path, field_name):
    p = _write(tmp_path, f"radius: 0.11\nmass: 0.43\nx: 1.5\ny: 0.0\n{field_name}: -0.01\n")
    with pytest.raises(ValueError, match=f"{field_name} must be >= 0.0"):
        load_ball_config(p)


def test_critic_support_defaults_to_none_when_absent(tmp_path):
    cfg = load_ball_config(_write(tmp_path, "radius: 0.11\nmass: 0.43\nx: 1.5\ny: 0.0\n"))
    assert cfg.critic_v_min is None and cfg.critic_v_max is None and cfg.critic_num_atoms is None


def test_critic_support_parsed_when_present(tmp_path):
    p = _write(tmp_path, "radius: 0.11\nmass: 0.43\nx: 1.5\ny: 0.0\n"
               "critic_v_min: -40.0\ncritic_v_max: 40.0\ncritic_num_atoms: 201\n")
    cfg = load_ball_config(p)
    assert cfg.critic_v_min == -40.0 and cfg.critic_v_max == 40.0
    assert cfg.critic_num_atoms == 201 and isinstance(cfg.critic_num_atoms, int)


def test_critic_support_half_set_raises(tmp_path):
    p = _write(tmp_path, "radius: 0.11\nmass: 0.43\nx: 1.5\ny: 0.0\ncritic_v_min: -40.0\n")
    with pytest.raises(ValueError, match="must be set together"):
        load_ball_config(p)


def test_critic_num_atoms_below_two_raises(tmp_path):
    p = _write(tmp_path, "radius: 0.11\nmass: 0.43\nx: 1.5\ny: 0.0\ncritic_num_atoms: 1\n")
    with pytest.raises(ValueError, match="must be >= 2"):
        load_ball_config(p)


# ---------------------------------------------------------------------------------------------
# Azimuth-aim refactor (2026-08-22) -- legacy single-skill (BallConfig) counterpart of
# config_types/tests/test_multi_skill.py's kick_aim_* tests.
# ---------------------------------------------------------------------------------------------


def test_kick_aim_fields_default_off(tmp_path):
    p = _write(tmp_path, "radius: 0.11\nmass: 0.43\nx: 1.0\ny: 0.0\ntarget_x: 6.0\ntarget_y: 1.0\n")
    cfg = load_ball_config(p)
    assert cfg.kick_aim_enabled is False
    assert cfg.kick_aim_theta_max_deg == 15.0
    assert cfg.kick_aim_theta_ref_deg == 45.0
    assert cfg.kick_aim_nominal_distance_m == 5.0


def test_kick_aim_enabled_bearing_derived_from_position_and_target(tmp_path):
    p = _write(
        tmp_path, "radius: 0.11\nmass: 0.43\nx: 1.0\ny: 0.0\ntarget_x: 6.0\ntarget_y: 1.0\n"
        "kick_aim_enabled: true\n",
    )
    cfg = load_ball_config(p)
    assert cfg.kick_aim_enabled is True
    # atan2(1.0 - 0.0, 6.0 - 1.0) = atan2(1, 5) ~= 11.31 deg
    assert abs(cfg.resolved_nominal_bearing_deg() - 11.31) < 0.01


@pytest.mark.parametrize("kick_aim_enabled", ["true", "false"])
def test_randomize_target_x_raises_regardless_of_kick_aim_enabled(tmp_path, kick_aim_enabled):
    """randomize_target_x/y was removed entirely 2026-08-22 (azimuth-aim refactor) -- a stray key
    must raise a clear "removed" error, not be silently ignored, whether or not kick_aim_enabled
    happens to be set."""
    p = _write(
        tmp_path, "radius: 0.11\nmass: 0.43\nx: 1.0\ny: 0.0\ntarget_x: 6.0\ntarget_y: 1.0\n"
        f"kick_aim_enabled: {kick_aim_enabled}\nrandomize_target_x: 1.0\n",
    )
    with pytest.raises(ValueError, match="was removed"):
        load_ball_config(p)


@pytest.mark.parametrize("bad", ["0.0", "-5.0", "50.0"])
def test_kick_aim_theta_max_deg_out_of_range_raises(tmp_path, bad):
    p = _write(
        tmp_path, f"radius: 0.11\nmass: 0.43\nx: 1.0\ny: 0.0\nkick_aim_theta_max_deg: {bad}\n",
    )
    with pytest.raises(ValueError, match="kick_aim_theta_max_deg"):
        load_ball_config(p)


def test_kick_aim_nominal_distance_m_must_be_positive(tmp_path):
    p = _write(tmp_path, "radius: 0.11\nmass: 0.43\nx: 1.0\ny: 0.0\nkick_aim_nominal_distance_m: 0.0\n")
    with pytest.raises(ValueError, match="kick_aim_nominal_distance_m must be > 0.0"):
        load_ball_config(p)


def test_kick_error_ball_to_target_sigma_default_off(tmp_path):
    p = _write(tmp_path, "radius: 0.11\nmass: 0.43\nx: 1.0\ny: 0.0\n")
    cfg = load_ball_config(p)
    assert cfg.kick_error_ball_to_target_sigma is None


@pytest.mark.parametrize("bad", ["0.0", "-1.0"])
def test_kick_error_ball_to_target_sigma_must_be_positive(tmp_path, bad):
    p = _write(tmp_path, f"radius: 0.11\nmass: 0.43\nx: 1.0\ny: 0.0\nkick_error_ball_to_target_sigma: {bad}\n")
    with pytest.raises(ValueError, match="must be > 0.0"):
        load_ball_config(p)
