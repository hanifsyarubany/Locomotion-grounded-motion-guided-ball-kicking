import pytest

from holosoma.config_types.multi_skill import (
    DEFAULT_MULTI_SKILL_CONFIG_YAML,
    SkillConfig,
    load_multi_skill_config,
)


def _write(tmp_path, contents: str):
    p = tmp_path / "multi_skill.yaml"
    p.write_text(contents)
    return p


def test_loads_two_skills_with_defaults_resolved(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
motion_skill_1:
  motion_npz: clip1.npz
  x: 2.84
  y: -0.46
  target_x: 7.84
  target_y: -0.46
  shooting_reward_scale: 0.1
  motion_training_ratio: 0.30
  strike_start_frame: 40
  stand_start_frame: 90
motion_skill_2:
  motion_npz: clip2.npz
  x: 2.84
  y: -0.46
  motion_training_ratio: 0.25
  strike_start_frame: 10
  stand_start_frame: 20
""",
    )
    cfg = load_multi_skill_config(p)
    assert cfg.radius == 0.11
    assert cfg.mass == 0.43
    assert len(cfg.skills) == 2
    assert cfg.skills[0].motion_npz == "clip1.npz"
    assert cfg.skills[0].shooting_reward_scale == 0.1
    assert cfg.skills[0].strike_start_frame == 40
    assert cfg.skills[0].stand_start_frame == 90
    # skill 2 omitted target_x/target_y and shooting_reward_scale -> defaults resolve correctly
    assert cfg.skills[1].shooting_reward_scale == 0.0  # Stage B by default
    assert cfg.skills[1].resolved_target() == (7.84, -0.46)  # x + 5.0, y
    assert cfg.skills[1].kick_foot == "right"
    assert cfg.skills[1].strike_start_frame == 10
    assert cfg.skills[1].stand_start_frame == 20


def test_declaration_order_preserved(tmp_path):
    # motion_skill_10 sorts before motion_skill_2 alphabetically -- must NOT reorder by name.
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
motion_skill_2:
  motion_npz: second.npz
  x: 1.0
  y: 0.0
  motion_training_ratio: 0.1
  strike_start_frame: 10
  stand_start_frame: 20
motion_skill_10:
  motion_npz: first_in_file.npz
  x: 1.0
  y: 0.0
  motion_training_ratio: 0.1
  strike_start_frame: 10
  stand_start_frame: 20
""",
    )
    cfg = load_multi_skill_config(p)
    assert [s.motion_npz for s in cfg.skills] == ["second.npz", "first_in_file.npz"]


def test_ratio_sum_over_one_raises(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.7, strike_start_frame: 10, stand_start_frame: 20}
motion_skill_2: {motion_npz: b.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    with pytest.raises(ValueError, match="must be <= 1.0"):
        load_multi_skill_config(p)


def test_missing_required_key_raises(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
motion_skill_1: {motion_npz: a.npz, x: 1.0, motion_training_ratio: 0.5}
""",
    )
    with pytest.raises(ValueError, match="missing required key"):
        load_multi_skill_config(p)


def test_invalid_kick_foot_raises(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, kick_foot: front, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    with pytest.raises(ValueError, match="kick_foot must be"):
        load_multi_skill_config(p)


def test_kick_ankle_pitch_correction_enabled_defaults_true(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    cfg = load_multi_skill_config(p)
    assert cfg.skills[0].kick_ankle_pitch_correction_enabled is True


def test_kick_ankle_pitch_correction_enabled_per_skill_override(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
motion_skill_1:
  motion_npz: a.npz
  x: 1.0
  y: 0.0
  motion_training_ratio: 0.3
  strike_start_frame: 10
  stand_start_frame: 20
  kick_ankle_pitch_correction_enabled: false
motion_skill_2:
  motion_npz: b.npz
  x: 1.0
  y: 0.0
  motion_training_ratio: 0.3
  strike_start_frame: 10
  stand_start_frame: 20
""",
    )
    cfg = load_multi_skill_config(p)
    assert cfg.skills[0].kick_ankle_pitch_correction_enabled is False
    assert cfg.skills[1].kick_ankle_pitch_correction_enabled is True  # untouched skill keeps the default


def test_strike_start_frame_not_less_than_stand_start_frame_raises(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 20, stand_start_frame: 20}
""",
    )
    with pytest.raises(ValueError, match="strike_start_frame"):
        load_multi_skill_config(p)


def test_no_skill_blocks_raises(tmp_path):
    p = _write(tmp_path, "ball: {radius: 0.11, mass: 0.43}\n")
    with pytest.raises(ValueError, match="no 'motion_skill_N' blocks"):
        load_multi_skill_config(p)


def test_load_multi_skill_config_raises_with_no_arg_when_env_var_unset():
    """2026-08-23: configs/stageB_and_C.yaml (the old hardcoded default file) was deleted with no
    designated replacement, and DEFAULT_MULTI_SKILL_CONFIG_YAML is now None whenever
    HOLOSOMA_SKILLS_CONFIG isn't set -- see that constant's own comment in multi_skill.py. Calling
    load_multi_skill_config() with no explicit path in that state must fail loud, not silently miss
    a file or resolve to something nobody chose."""
    assert DEFAULT_MULTI_SKILL_CONFIG_YAML is None
    with pytest.raises(RuntimeError, match="no default multi-skill config"):
        load_multi_skill_config()


def test_ball_obs_fields_default_to_pre_2026_07_24_hardcoded_values_when_absent(tmp_path):
    """2026-07-24: omitting ball_obs_* entirely must reproduce the values that were hardcoded
    directly in config_values/unified/g1/observation.py before these fields existed -- no
    behavior change for any config that doesn't opt in."""
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    cfg = load_multi_skill_config(p)
    assert cfg.ball_obs_noise == 0.05
    assert cfg.ball_obs_noise_range_coefficient == 0.03
    assert cfg.ball_obs_delay_steps_min == 0
    assert cfg.ball_obs_delay_steps_max == 3


def test_ball_obs_fields_parsed_when_present(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
ball_obs_noise: 0.1
ball_obs_noise_range_coefficient: 0.06
ball_obs_delay_steps_min: 1
ball_obs_delay_steps_max: 5
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    cfg = load_multi_skill_config(p)
    assert cfg.ball_obs_noise == 0.1
    assert cfg.ball_obs_noise_range_coefficient == 0.06
    assert cfg.ball_obs_delay_steps_min == 1
    assert cfg.ball_obs_delay_steps_max == 5


def test_ball_obs_fields_are_shared_not_per_skill(tmp_path):
    """These are GLOBAL (top-level) fields, deliberately not part of any motion_skill_N block --
    every skill sees the same value, unlike observation_bias and friends."""
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
ball_obs_noise: 0.2
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.3, strike_start_frame: 10, stand_start_frame: 20}
motion_skill_2: {motion_npz: b.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.3, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    cfg = load_multi_skill_config(p)
    assert cfg.ball_obs_noise == 0.2  # a single value on MultiSkillConfig, not per-SkillConfig
    assert not hasattr(cfg.skills[0], "ball_obs_noise")


def test_ball_obs_delay_max_less_than_min_raises(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
ball_obs_delay_steps_min: 5
ball_obs_delay_steps_max: 2
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    with pytest.raises(ValueError, match="must be >="):
        load_multi_skill_config(p)


def test_ball_obs_hold_steps_default_to_off_when_absent(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    cfg = load_multi_skill_config(p)
    assert cfg.ball_obs_hold_steps_min == 0
    assert cfg.ball_obs_hold_steps_max == 0


def test_ball_obs_hold_steps_parsed_when_present(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
ball_obs_hold_steps_min: 2
ball_obs_hold_steps_max: 4
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    cfg = load_multi_skill_config(p)
    assert cfg.ball_obs_hold_steps_min == 2
    assert cfg.ball_obs_hold_steps_max == 4


def test_ball_obs_hold_steps_min_rejects_negative(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
ball_obs_hold_steps_min: -1
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    with pytest.raises(ValueError, match="must be >= 0"):
        load_multi_skill_config(p)


def test_ball_obs_hold_steps_max_less_than_min_raises(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
ball_obs_hold_steps_min: 5
ball_obs_hold_steps_max: 2
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    with pytest.raises(ValueError, match="must be >="):
        load_multi_skill_config(p)


def test_ball_obs_stale_probability_defaults_to_off_when_absent(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    cfg = load_multi_skill_config(p)
    assert cfg.ball_obs_stale_probability == 0.0


def test_ball_obs_stale_probability_parsed_when_present(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
ball_obs_stale_probability: 0.01
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    cfg = load_multi_skill_config(p)
    assert cfg.ball_obs_stale_probability == 0.01


def test_ball_obs_stale_probability_rejects_out_of_range(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
ball_obs_stale_probability: 1.5
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    with pytest.raises(ValueError, match="must be in \\[0.0, 1.0\\]"):
        load_multi_skill_config(p)


def test_explicit_ball_obs_values_round_trip(tmp_path):
    """Explicit ball_obs_* values (matching the pre-2026-07-24 hardcoded defaults) must parse
    unchanged. Replaces the old test that loaded the real (now-deleted, 2026-08-23)
    configs/stageB_and_C.yaml default file -- see DEFAULT_MULTI_SKILL_CONFIG_YAML's own comment
    for why there's no more implicit default file to load them from."""
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
ball_obs_noise: 0.05
ball_obs_noise_range_coefficient: 0.03
ball_obs_delay_steps_min: 0
ball_obs_delay_steps_max: 3
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    cfg = load_multi_skill_config(p)
    assert cfg.ball_obs_noise == 0.05
    assert cfg.ball_obs_noise_range_coefficient == 0.03
    assert cfg.ball_obs_delay_steps_min == 0
    assert cfg.ball_obs_delay_steps_max == 3


def test_kick_contact_force_penalty_fields_default_to_pre_2026_07_28_hardcoded_values_when_absent(tmp_path):
    """Omitting these entirely must reproduce the values that were hardcoded directly in
    config_values/unified/g1/reward.py (_KICK_SAFETY_FLOOR=3.0, k=15.00, and
    kick_safety.py's force_threshold_bodyweight_multiplier=3.0 default) before these fields
    existed -- no behavior change for any config that doesn't opt in."""
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    cfg = load_multi_skill_config(p)
    assert cfg.kick_contact_force_penalty_floor == 3.0
    assert cfg.kick_contact_force_penalty_k == 15.0
    assert cfg.kick_contact_force_threshold_bodyweight_multiplier == 3.0


def test_kick_contact_force_penalty_fields_parsed_when_present(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
kick_contact_force_penalty_floor: 2.0
kick_contact_force_penalty_k: 8.0
kick_contact_force_threshold_bodyweight_multiplier: 4.0
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    cfg = load_multi_skill_config(p)
    assert cfg.kick_contact_force_penalty_floor == 2.0
    assert cfg.kick_contact_force_penalty_k == 8.0
    assert cfg.kick_contact_force_threshold_bodyweight_multiplier == 4.0


def test_kick_contact_force_penalty_fields_are_shared_not_per_skill(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
kick_contact_force_penalty_k: 8.0
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.3, strike_start_frame: 10, stand_start_frame: 20}
motion_skill_2: {motion_npz: b.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.3, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    cfg = load_multi_skill_config(p)
    assert cfg.kick_contact_force_penalty_k == 8.0
    assert not hasattr(cfg.skills[0], "kick_contact_force_penalty_k")


def test_start_at_timestep_zero_prob_defaults_to_pre_2026_07_28_hardcoded_value_when_absent(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    cfg = load_multi_skill_config(p)
    assert cfg.start_at_timestep_zero_prob == 1.0


def test_start_at_timestep_zero_prob_parsed_when_present(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
start_at_timestep_zero_prob: 0.5
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    cfg = load_multi_skill_config(p)
    assert cfg.start_at_timestep_zero_prob == 0.5


def test_start_at_timestep_zero_prob_out_of_range_raises(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
start_at_timestep_zero_prob: 1.5
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    with pytest.raises(ValueError, match="must be in \\[0.0, 1.0\\]"):
        load_multi_skill_config(p)


def test_kick_target_entropy_ratio_defaults_to_none_when_absent(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    cfg = load_multi_skill_config(p)
    assert cfg.kick_target_entropy_ratio is None


def test_kick_target_entropy_ratio_parsed_when_present(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
kick_target_entropy_ratio: 0.3
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    cfg = load_multi_skill_config(p)
    assert cfg.kick_target_entropy_ratio == 0.3


def test_kick_target_entropy_ratio_of_zero_is_distinct_from_unset(tmp_path):
    """0.0 is a legitimate, meaningful config value (same numeric target as locomotion's own
    default) -- must NOT be conflated with "unset" (None, meaning the whole 2-group mechanism is
    off). Guards against a `raw.get("kick_target_entropy_ratio", 0.0) or None`-style bug that
    would collapse 0.0 back into None."""
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
kick_target_entropy_ratio: 0.0
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    cfg = load_multi_skill_config(p)
    assert cfg.kick_target_entropy_ratio == 0.0
    assert cfg.kick_target_entropy_ratio is not None


def test_kick_gamma_defaults_to_none_when_absent(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    cfg = load_multi_skill_config(p)
    assert cfg.kick_gamma is None


def test_kick_gamma_parsed_when_present(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
kick_gamma: 0.99
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    cfg = load_multi_skill_config(p)
    assert cfg.kick_gamma == 0.99


@pytest.mark.parametrize("bad_value", [0.0, 1.0, 1.5, -0.1])
def test_kick_gamma_out_of_range_raises(tmp_path, bad_value):
    """gamma must be a genuine discount, strictly inside (0, 1) -- 0.0 makes bootstrapping
    meaningless and 1.0 makes the horizon infinite/undiscounted, neither is a sane per-mode value."""
    p = _write(
        tmp_path,
        f"""
ball: {{radius: 0.11, mass: 0.43}}
kick_gamma: {bad_value}
motion_skill_1: {{motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}}
""",
    )
    with pytest.raises(ValueError, match="kick_gamma must be in \\(0.0, 1.0\\)"):
        load_multi_skill_config(p)


def test_kick_recovery_termination_handoff_defaults_false_when_absent(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    cfg = load_multi_skill_config(p)
    assert cfg.kick_recovery_termination_handoff is False


def test_kick_recovery_termination_handoff_parsed_when_present(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
kick_recovery_termination_handoff: true
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    cfg = load_multi_skill_config(p)
    assert cfg.kick_recovery_termination_handoff is True


def test_kick_recovery_locomotion_flip_enabled_defaults_false_when_absent(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    cfg = load_multi_skill_config(p)
    assert cfg.kick_recovery_locomotion_flip_enabled is False


def test_kick_recovery_locomotion_flip_enabled_parsed_when_present(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
kick_recovery_locomotion_flip_enabled: true
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    cfg = load_multi_skill_config(p)
    assert cfg.kick_recovery_locomotion_flip_enabled is True


def test_replay_buffer_sanitize_enabled_defaults_false_when_absent(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    cfg = load_multi_skill_config(p)
    assert cfg.replay_buffer_sanitize_enabled is False


def test_replay_buffer_sanitize_enabled_parsed_when_present(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
replay_buffer_sanitize_enabled: true
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    cfg = load_multi_skill_config(p)
    assert cfg.replay_buffer_sanitize_enabled is True


def test_joint_pos_sanity_check_enabled_defaults_false_when_absent(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    cfg = load_multi_skill_config(p)
    assert cfg.joint_pos_sanity_check_enabled is False
    assert cfg.joint_pos_sanity_threshold == 20.0


def test_joint_pos_sanity_check_enabled_and_threshold_parsed_when_present(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
joint_pos_sanity_check_enabled: true
joint_pos_sanity_threshold: 15.0
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    cfg = load_multi_skill_config(p)
    assert cfg.joint_pos_sanity_check_enabled is True
    assert cfg.joint_pos_sanity_threshold == 15.0


def test_joint_pos_sanity_threshold_non_positive_raises(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
joint_pos_sanity_threshold: 0.0
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    with pytest.raises(ValueError, match="joint_pos_sanity_threshold"):
        load_multi_skill_config(p)


def test_motion_head_velocity_smoothing_frames_defaults_off_when_absent(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    cfg = load_multi_skill_config(p)
    assert cfg.motion_head_velocity_smoothing_frames == 0


def test_motion_head_velocity_smoothing_frames_parsed_when_present(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
motion_head_velocity_smoothing_frames: 3
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    cfg = load_multi_skill_config(p)
    assert cfg.motion_head_velocity_smoothing_frames == 3


def test_motion_head_velocity_smoothing_frames_negative_raises(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
motion_head_velocity_smoothing_frames: -1
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    with pytest.raises(ValueError, match="motion_head_velocity_smoothing_frames"):
        load_multi_skill_config(p)


def test_kick_recovery_drift_deadzone_defaults_to_15cm_when_absent(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    cfg = load_multi_skill_config(p)
    assert cfg.kick_recovery_drift_deadzone == 0.15


def test_penalty_curriculum_enabled_defaults_true_when_absent(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    cfg = load_multi_skill_config(p)
    assert cfg.penalty_curriculum_enabled is True


def test_penalty_curriculum_enabled_parsed_when_present(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
penalty_curriculum_enabled: false
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    cfg = load_multi_skill_config(p)
    assert cfg.penalty_curriculum_enabled is False


def test_post_flip_termination_grace_steps_defaults_zero_when_absent(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    cfg = load_multi_skill_config(p)
    assert cfg.post_flip_termination_grace_steps == 0.0


def test_post_flip_termination_grace_steps_parsed_when_present(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
post_flip_termination_grace_steps: 50.0
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    cfg = load_multi_skill_config(p)
    assert cfg.post_flip_termination_grace_steps == 50.0


def test_post_flip_termination_grace_steps_negative_raises(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
post_flip_termination_grace_steps: -1.0
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    with pytest.raises(ValueError, match="post_flip_termination_grace_steps"):
        load_multi_skill_config(p)


def test_post_flip_reward_decay_steps_defaults_zero_when_absent(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    cfg = load_multi_skill_config(p)
    assert cfg.post_flip_reward_decay_steps == 0.0


def test_post_flip_reward_decay_steps_parsed_when_present(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
post_flip_reward_decay_steps: 50.0
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    cfg = load_multi_skill_config(p)
    assert cfg.post_flip_reward_decay_steps == 50.0


def test_post_flip_reward_decay_steps_negative_raises(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
post_flip_reward_decay_steps: -1.0
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    with pytest.raises(ValueError, match="post_flip_reward_decay_steps"):
        load_multi_skill_config(p)


def test_kick_recovery_drift_deadzone_parsed_when_present(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
kick_recovery_drift_deadzone: 0.25
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    cfg = load_multi_skill_config(p)
    assert cfg.kick_recovery_drift_deadzone == 0.25


def test_kick_recovery_drift_deadzone_rejects_non_positive(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
kick_recovery_drift_deadzone: 0.0
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    with pytest.raises(ValueError, match="kick_recovery_drift_deadzone must be > 0.0"):
        load_multi_skill_config(p)


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
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    cfg = load_multi_skill_config(p)
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
def test_kick_recovery_posture_deadzone_parsed_when_present_and_independent(tmp_path, field_name):
    """Each of the 4 fields must be independently settable -- setting ONE must not affect the
    other 3, proving they're genuinely separate fields, not accidentally aliased (e.g. feet_width
    and knee_width sharing storage since they currently default to the same 0.03 value)."""
    p = _write(
        tmp_path,
        f"""
ball: {{radius: 0.11, mass: 0.43}}
{field_name}: 0.5
motion_skill_1: {{motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}}
""",
    )
    cfg = load_multi_skill_config(p)
    assert getattr(cfg, field_name) == 0.5
    other_fields = {
        "kick_recovery_stand_height_deadzone": 0.015,
        "kick_recovery_stand_orientation_deadzone": 0.025,
        "kick_recovery_stand_feet_width_deadzone": 0.03,
        "kick_recovery_stand_knee_width_deadzone": 0.03,
    }
    del other_fields[field_name]
    for other_name, other_default in other_fields.items():
        assert getattr(cfg, other_name) == other_default, f"{other_name} was affected by setting {field_name}"


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
    """Unlike the termination-threshold fields (bad_motion_body_pos_threshold etc.), these are
    REWARD deadzones -- 0.0 is a legitimate value (no tolerance), not a degenerate one."""
    p = _write(
        tmp_path,
        f"""
ball: {{radius: 0.11, mass: 0.43}}
{field_name}: 0.0
motion_skill_1: {{motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}}
""",
    )
    cfg = load_multi_skill_config(p)
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
    p = _write(
        tmp_path,
        f"""
ball: {{radius: 0.11, mass: 0.43}}
{field_name}: -0.01
motion_skill_1: {{motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}}
""",
    )
    with pytest.raises(ValueError, match=f"{field_name} must be >= 0.0"):
        load_multi_skill_config(p)


@pytest.mark.parametrize(
    "field_name",
    ["kick_swing_orientation_deadzone", "kick_swing_torso_orientation_deadzone"],
)
def test_kick_swing_orientation_deadzone_defaults_zero_when_absent(tmp_path, field_name):
    """Unlike the 4 kick_recovery_stand_* deadzones above (nonzero defaults), these 2 default to
    0.0 -- matching penalty_kick_swing_orientation/_torso_orientation's own pre-existing hardcoded
    value, a true no-op until this project's C1 yaml opts in."""
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    cfg = load_multi_skill_config(p)
    assert getattr(cfg, field_name) == 0.0


@pytest.mark.parametrize(
    "field_name",
    ["kick_swing_orientation_deadzone", "kick_swing_torso_orientation_deadzone"],
)
def test_kick_swing_orientation_deadzone_parsed_when_present_and_independent(tmp_path, field_name):
    """Each of the 2 fields must be independently settable -- setting ONE must not affect the
    other (pelvis vs torso deadzones are deliberately separate, per each field's own docstring)."""
    p = _write(
        tmp_path,
        f"""
ball: {{radius: 0.11, mass: 0.43}}
{field_name}: 0.55
motion_skill_1: {{motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}}
""",
    )
    cfg = load_multi_skill_config(p)
    assert getattr(cfg, field_name) == 0.55
    other = "kick_swing_torso_orientation_deadzone" if field_name == "kick_swing_orientation_deadzone" else "kick_swing_orientation_deadzone"
    assert getattr(cfg, other) == 0.0, f"{other} was affected by setting {field_name}"


@pytest.mark.parametrize(
    "field_name",
    ["kick_swing_orientation_deadzone", "kick_swing_torso_orientation_deadzone"],
)
def test_kick_swing_orientation_deadzone_rejects_negative(tmp_path, field_name):
    p = _write(
        tmp_path,
        f"""
ball: {{radius: 0.11, mass: 0.43}}
{field_name}: -0.01
motion_skill_1: {{motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}}
""",
    )
    with pytest.raises(ValueError, match=f"{field_name} must be >= 0.0"):
        load_multi_skill_config(p)


# ============================================================================================
# 2-file mode (HOLOSOMA_TASK_CONFIG), 2026-08-05 -- see HOLOSOMA_TASK_CONFIG_ENV_VAR's own
# docstring in config_types/multi_skill.py for the full design. A task-config file carries global
# fields (ball, ood_*, bad_tracking_*, ...) AND the 15 fields that used to be per-skill
# (randomize_x, success_radius, the 8 reward-category scales, misc); a skills file carries ONLY
# motion_skill_N blocks with their 9 genuinely-per-clip fields.
# ============================================================================================

_TASK_CONFIG_YAML = """
ball: {radius: 0.20, mass: 0.99}
kick_gamma: 0.99
randomize_x: 0.35
randomize_y: 0.40
success_radius: 0.6
motion_tracking_reward_scale: 2.0
kick_alive_pre_kick_ratio: 0.1
"""

_SKILLS_ONLY_YAML = """
motion_skill_1:
  motion_npz: clip1.npz
  x: 1.0
  y: 0.0
  target_x: 6.0
  target_y: 0.0
  strike_start_frame: 10
  stand_start_frame: 20
  motion_training_ratio: 0.4
  kick_foot: right
motion_skill_2:
  motion_npz: clip2.npz
  x: 2.0
  y: 0.5
  strike_start_frame: 15
  stand_start_frame: 25
  motion_training_ratio: 0.3
"""


def test_2file_mode_global_fields_come_from_task_config(tmp_path, monkeypatch):
    task_config_path = tmp_path / "task_config.yaml"
    task_config_path.write_text(_TASK_CONFIG_YAML)
    skills_path = tmp_path / "skills.yaml"
    skills_path.write_text(_SKILLS_ONLY_YAML)

    monkeypatch.setenv("HOLOSOMA_TASK_CONFIG", str(task_config_path))
    cfg = load_multi_skill_config(skills_path)

    assert cfg.radius == 0.20
    assert cfg.mass == 0.99
    assert cfg.kick_gamma == 0.99


def test_2file_mode_shared_fields_apply_uniformly_to_every_skill(tmp_path, monkeypatch):
    task_config_path = tmp_path / "task_config.yaml"
    task_config_path.write_text(_TASK_CONFIG_YAML)
    skills_path = tmp_path / "skills.yaml"
    skills_path.write_text(_SKILLS_ONLY_YAML)

    monkeypatch.setenv("HOLOSOMA_TASK_CONFIG", str(task_config_path))
    cfg = load_multi_skill_config(skills_path)

    assert len(cfg.skills) == 2
    for skill in cfg.skills:
        assert skill.randomize_x == 0.35
        assert skill.randomize_y == 0.40
        assert skill.success_radius == 0.6
        assert skill.motion_tracking_reward_scale == 2.0
        assert skill.kick_alive_pre_kick_ratio == 0.1
    # content fields still come from the skills file, per-skill, unaffected
    assert cfg.skills[0].motion_npz == "clip1.npz"
    assert cfg.skills[1].motion_npz == "clip2.npz"
    assert cfg.skills[0].x == 1.0
    assert cfg.skills[1].x == 2.0


def test_2file_mode_shared_field_omitted_from_task_config_falls_back_to_skillconfig_default(tmp_path, monkeypatch):
    """A shared field the task-config file doesn't set at all must reproduce SkillConfig's own
    dataclass default -- same default, just resolved from a different file than legacy mode."""
    task_config_path = tmp_path / "task_config.yaml"
    task_config_path.write_text("ball: {radius: 0.11, mass: 0.43}\n")  # no randomize_x etc. at all
    skills_path = tmp_path / "skills.yaml"
    skills_path.write_text(_SKILLS_ONLY_YAML)

    monkeypatch.setenv("HOLOSOMA_TASK_CONFIG", str(task_config_path))
    cfg = load_multi_skill_config(skills_path)

    assert cfg.skills[0].randomize_x == 0.0  # SkillConfig's own default
    assert cfg.skills[0].success_radius == 0.5  # SkillConfig's own default
    assert cfg.skills[0].kick_alive_pre_kick_ratio == 1.0  # SkillConfig's own default


def test_2file_mode_skill_block_can_override_a_shared_field_individually(tmp_path, monkeypatch):
    """Escape hatch: a skill block explicitly setting one of the 15 shared fields wins over the
    task-config file's shared value, for that skill only."""
    task_config_path = tmp_path / "task_config.yaml"
    task_config_path.write_text(_TASK_CONFIG_YAML)
    skills_path = tmp_path / "skills.yaml"
    skills_path.write_text(
        _SKILLS_ONLY_YAML.replace(
            "motion_skill_1:\n  motion_npz: clip1.npz",
            "motion_skill_1:\n  randomize_x: 0.99\n  motion_npz: clip1.npz",
        )
    )

    monkeypatch.setenv("HOLOSOMA_TASK_CONFIG", str(task_config_path))
    cfg = load_multi_skill_config(skills_path)

    assert cfg.skills[0].randomize_x == 0.99  # overridden on this skill only
    assert cfg.skills[1].randomize_x == 0.35  # everyone else still gets the task-config's shared value


def test_2file_mode_missing_task_config_file_raises(tmp_path, monkeypatch):
    skills_path = tmp_path / "skills.yaml"
    skills_path.write_text(_SKILLS_ONLY_YAML)

    monkeypatch.setenv("HOLOSOMA_TASK_CONFIG", str(tmp_path / "does_not_exist.yaml"))
    with pytest.raises(FileNotFoundError, match="HOLOSOMA_TASK_CONFIG"):
        load_multi_skill_config(skills_path)


def test_2file_mode_motion_skill_block_in_task_config_file_raises_swap_hint(tmp_path, monkeypatch):
    """A motion_skill_N block accidentally left in the task-config file (e.g. the two env vars
    were swapped) must raise a clear, actionable error, not silently ignore it or crash obscurely
    later."""
    task_config_path = tmp_path / "task_config.yaml"
    task_config_path.write_text(_TASK_CONFIG_YAML + "\n" + _SKILLS_ONLY_YAML)
    skills_path = tmp_path / "skills.yaml"
    skills_path.write_text(_SKILLS_ONLY_YAML)

    monkeypatch.setenv("HOLOSOMA_TASK_CONFIG", str(task_config_path))
    with pytest.raises(ValueError, match="swapped"):
        load_multi_skill_config(skills_path)


def test_legacy_single_file_mode_unaffected_when_task_config_env_var_unset(tmp_path, monkeypatch):
    """Explicitly confirms HOLOSOMA_TASK_CONFIG being unset reproduces the exact legacy single-file
    behavior -- the combined file carries everything, no 2-file logic engaged at all."""
    monkeypatch.delenv("HOLOSOMA_TASK_CONFIG", raising=False)
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20, randomize_x: 0.25}
""",
    )
    cfg = load_multi_skill_config(p)
    assert cfg.radius == 0.11
    assert cfg.skills[0].randomize_x == 0.25


def test_kick_ball_over_line_require_has_kicked_defaults_false_when_absent(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    cfg = load_multi_skill_config(p)
    assert cfg.kick_ball_over_line_require_has_kicked is False


def test_kick_ball_over_line_require_has_kicked_parsed_when_present(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
kick_ball_over_line_require_has_kicked: true
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    cfg = load_multi_skill_config(p)
    assert cfg.kick_ball_over_line_require_has_kicked is True


# ============================================================================================
# 2026-08-14, user-requested: SkillConfig.task_config (per-skill "which task_config was this
# designed under" field, consumed by holosoma/__init__.py's bootstrap, not by this loader) and
# MultiSkillConfig.base_robot_target_height/base_robot_deadzone (a single global standing-height
# target/deadzone, read from the SKILLS file's own top-level `base_robot:` block regardless of
# 1-file/2-file mode -- see configs/skills.example.yaml for the shape this mirrors).
# ============================================================================================


def test_task_config_field_defaults_to_none_when_absent(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    cfg = load_multi_skill_config(p)
    assert cfg.skills[0].task_config is None


def test_task_config_field_parsed_when_present(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20, task_config: task_config_stageC1}
""",
    )
    cfg = load_multi_skill_config(p)
    assert cfg.skills[0].task_config == "task_config_stageC1"


def test_task_config_field_is_per_skill_not_shared(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.3, strike_start_frame: 10, stand_start_frame: 20, task_config: task_config_stageB}
motion_skill_2: {motion_npz: b.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.3, strike_start_frame: 10, stand_start_frame: 20, task_config: task_config_stageC1}
""",
    )
    cfg = load_multi_skill_config(p)
    assert cfg.skills[0].task_config == "task_config_stageB"
    assert cfg.skills[1].task_config == "task_config_stageC1"


def test_motion_head_velocity_smoothing_frames_field_defaults_to_none_when_absent(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    cfg = load_multi_skill_config(p)
    assert cfg.skills[0].motion_head_velocity_smoothing_frames is None


def test_motion_head_velocity_smoothing_frames_field_parsed_when_present(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20, motion_head_velocity_smoothing_frames: 3}
""",
    )
    cfg = load_multi_skill_config(p)
    assert cfg.skills[0].motion_head_velocity_smoothing_frames == 3


def test_motion_head_velocity_smoothing_frames_field_is_per_skill_not_shared(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.3, strike_start_frame: 10, stand_start_frame: 20, motion_head_velocity_smoothing_frames: 3}
motion_skill_2: {motion_npz: b.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.3, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    cfg = load_multi_skill_config(p)
    assert cfg.skills[0].motion_head_velocity_smoothing_frames == 3
    assert cfg.skills[1].motion_head_velocity_smoothing_frames is None


def test_base_robot_fields_default_to_none_when_absent(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    cfg = load_multi_skill_config(p)
    assert cfg.base_robot_target_height is None
    assert cfg.base_robot_deadzone is None


def test_base_robot_fields_parsed_when_present(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
base_robot: {target_height: 0.70, deadzone: 0.02}
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    cfg = load_multi_skill_config(p)
    assert cfg.base_robot_target_height == 0.70
    assert cfg.base_robot_deadzone == 0.02


def test_base_robot_only_target_height_set_leaves_deadzone_none(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
base_robot: {target_height: 0.72}
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    cfg = load_multi_skill_config(p)
    assert cfg.base_robot_target_height == 0.72
    assert cfg.base_robot_deadzone is None


def test_base_robot_not_a_mapping_raises(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
base_robot: 0.76
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    with pytest.raises(ValueError, match="base_robot"):
        load_multi_skill_config(p)


def test_base_robot_unrecognized_key_raises(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
base_robot: {target_height: 0.76, typo_field: 1.0}
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    with pytest.raises(ValueError, match="typo_field"):
        load_multi_skill_config(p)


def test_l2sp_weight_defaults_to_zero_when_absent(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    cfg = load_multi_skill_config(p)
    assert cfg.l2sp_weight == 0.0


def test_l2sp_weight_parsed_when_present(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
l2sp_weight: 0.01
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    cfg = load_multi_skill_config(p)
    assert cfg.l2sp_weight == 0.01


def test_l2sp_weight_negative_raises(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
l2sp_weight: -0.01
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    with pytest.raises(ValueError, match="l2sp_weight must be >= 0.0"):
        load_multi_skill_config(p)


# ============================================================================================
# MultiSkillConfig.skill_replay_weights (2026-08-15) -- single-file parsing/validation, mirroring
# l2sp_weight's own tests above exactly (same field family, same "empty/0.0 = OFF" discipline).
# ============================================================================================


def test_skill_replay_weights_defaults_to_empty_when_absent(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    cfg = load_multi_skill_config(p)
    assert cfg.skill_replay_weights == []


def test_skill_replay_weights_parsed_when_present(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
skill_replay_weights: [4.0, 1.0]
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.1, strike_start_frame: 10, stand_start_frame: 20}
motion_skill_2: {motion_npz: b.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.8, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    cfg = load_multi_skill_config(p)
    assert cfg.skill_replay_weights == [4.0, 1.0]


def test_skill_replay_weights_negative_entry_raises(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
skill_replay_weights: [4.0, -1.0]
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    with pytest.raises(ValueError, match="skill_replay_weights must all be >= 0.0"):
        load_multi_skill_config(p)


def test_skill_replay_weights_all_zero_raises(tmp_path):
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
skill_replay_weights: [0.0, 0.0]
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    with pytest.raises(ValueError, match="skill_replay_weights are all zero"):
        load_multi_skill_config(p)


def test_skill_replay_weights_single_zero_entry_is_legal(tmp_path):
    """Only ALL-zero is rejected -- a single 0.0 alongside a nonzero entry is a legitimate
    'fully suppress this one skill's gradient' configuration."""
    p = _write(
        tmp_path,
        """
ball: {radius: 0.11, mass: 0.43}
skill_replay_weights: [0.0, 1.0]
motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
motion_skill_2: {motion_npz: b.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, strike_start_frame: 10, stand_start_frame: 20}
""",
    )
    cfg = load_multi_skill_config(p)
    assert cfg.skill_replay_weights == [0.0, 1.0]


# ============================================================================================
# 2026-08-15: l2sp_weight/skill_replay_weights' 2-FILE MODE carve-out -- ALWAYS read from the
# SKILLS file (like base_robot), never from HOLOSOMA_TASK_CONFIG's file, even though every other
# global field in this same 2-file-mode section (test_2file_mode_global_fields_come_from_task_
# config above) comes from task_config. Both fields are intrinsically tied to the skill roster
# (skill_replay_weights is literally indexed by skill_id against THIS file's own motion_skill_N
# blocks), so a task_config file that could be paired with a different roster must never supply
# them. See _parse_skill_replay_and_l2sp_fields' own docstring in config_types/multi_skill.py.
# ============================================================================================


def test_l2sp_and_skill_replay_weights_come_from_skills_file_not_task_config(tmp_path, monkeypatch):
    task_config_path = tmp_path / "task_config.yaml"
    task_config_path.write_text(_TASK_CONFIG_YAML)  # sets neither field
    skills_path = tmp_path / "skills.yaml"
    skills_path.write_text(
        _SKILLS_ONLY_YAML + "\nl2sp_weight: 0.01\nskill_replay_weights: [4.0, 1.0]\n"
    )

    monkeypatch.setenv("HOLOSOMA_TASK_CONFIG", str(task_config_path))
    cfg = load_multi_skill_config(skills_path)

    assert cfg.l2sp_weight == 0.01
    assert cfg.skill_replay_weights == [4.0, 1.0]
    # sanity check this really is 2-file mode -- an ordinary global field DID come from task_config
    assert cfg.kick_gamma == 0.99


def test_l2sp_and_skill_replay_weights_in_task_config_only_are_ignored(tmp_path, monkeypatch):
    """The value placed in the WRONG file (task_config) must not leak through -- proves the
    carve-out isn't just 'skills file wins on conflict' but 'task_config is never even read for
    these two fields'."""
    task_config_path = tmp_path / "task_config.yaml"
    task_config_path.write_text(_TASK_CONFIG_YAML + "\nl2sp_weight: 0.5\nskill_replay_weights: [99.0, 99.0]\n")
    skills_path = tmp_path / "skills.yaml"
    skills_path.write_text(_SKILLS_ONLY_YAML)  # sets neither field

    monkeypatch.setenv("HOLOSOMA_TASK_CONFIG", str(task_config_path))
    cfg = load_multi_skill_config(skills_path)

    assert cfg.l2sp_weight == 0.0
    assert cfg.skill_replay_weights == []


# ==================================================================================================
# The 4 observation-side handoff-discontinuity fixes (2026-08-18) -- see each field's own docstring
# on MultiSkillConfig for the measurement motivating it. Every one ships at an exact-no-op default,
# so the first two tests below are the ones that actually protect existing runs.
# ==================================================================================================

_MINIMAL_SKILL = """
motion_skill_1:
  motion_npz: clip1.npz
  x: 1.0
  y: 0.0
  motion_training_ratio: 0.9
  strike_start_frame: 10
  stand_start_frame: 20
"""


def test_obs_fix_fields_default_to_exact_no_op(tmp_path):
    """A yaml that never mentions any of them must reproduce pre-2026-08-18 behavior exactly."""
    cfg = load_multi_skill_config(_write(tmp_path, _MINIMAL_SKILL))
    assert cfg.pre_kick_obs_ramp_steps == 0.0
    assert cfg.obs_target_pos_distance_scale == 0.0
    assert cfg.obs_untag_shared_proprioception is False
    assert cfg.obs_ball_always_visible is False


def test_obs_fix_fields_round_trip_from_yaml(tmp_path):
    cfg = load_multi_skill_config(
        _write(
            tmp_path,
            _MINIMAL_SKILL
            + """
pre_kick_obs_ramp_steps: 50.0
obs_target_pos_distance_scale: 5.0
obs_untag_shared_proprioception: true
obs_ball_always_visible: true
""",
        )
    )
    assert cfg.pre_kick_obs_ramp_steps == 50.0
    assert cfg.obs_target_pos_distance_scale == 5.0
    assert cfg.obs_untag_shared_proprioception is True
    assert cfg.obs_ball_always_visible is True


def test_negative_pre_kick_obs_ramp_steps_raises(tmp_path):
    """Mirrors the validation its three pre_kick_* siblings already have -- a negative window is
    meaningless and would produce a negative alpha that inverts the crossfade."""
    with pytest.raises(ValueError, match="pre_kick_obs_ramp_steps must be >= 0.0"):
        load_multi_skill_config(_write(tmp_path, _MINIMAL_SKILL + "\npre_kick_obs_ramp_steps: -1.0\n"))


def test_negative_obs_target_pos_distance_scale_raises(tmp_path):
    """It divides inside a tanh(); a negative would silently flip the sign of every target reading
    rather than erroring at use time."""
    with pytest.raises(ValueError, match="obs_target_pos_distance_scale must be >= 0.0"):
        load_multi_skill_config(_write(tmp_path, _MINIMAL_SKILL + "\nobs_target_pos_distance_scale: -2.0\n"))


def test_zero_obs_target_pos_distance_scale_is_legal_and_means_off(tmp_path):
    """0.0 is the OFF sentinel, not an invalid value -- must not be rejected alongside negatives."""
    cfg = load_multi_skill_config(_write(tmp_path, _MINIMAL_SKILL + "\nobs_target_pos_distance_scale: 0.0\n"))
    assert cfg.obs_target_pos_distance_scale == 0.0


def test_obs_fix_fields_are_read_from_task_config_in_two_file_mode(tmp_path, monkeypatch):
    """These are GLOBAL fields, so in the 2-file mode this project always runs (HOLOSOMA_TASK_CONFIG
    set) they must come from the TASK CONFIG file, not the skills file -- the same contract every
    other global field follows, and the one that determines which yaml a user should edit."""
    task_cfg = tmp_path / "task_config.yaml"
    task_cfg.write_text("pre_kick_obs_ramp_steps: 30.0\nobs_ball_always_visible: true\n")
    monkeypatch.setenv("HOLOSOMA_TASK_CONFIG", str(task_cfg))
    cfg = load_multi_skill_config(_write(tmp_path, _MINIMAL_SKILL))
    assert cfg.pre_kick_obs_ramp_steps == 30.0
    assert cfg.obs_ball_always_visible is True


def test_contact_termination_force_threshold_defaults_to_inherit(tmp_path):
    """0.0 default means 'inherit the locomotion baseline' -- the unified termination registration
    then leaves force_threshold untouched, an exact no-op."""
    cfg = load_multi_skill_config(_write(tmp_path, _MINIMAL_SKILL))
    assert cfg.contact_termination_force_threshold == 0.0


def test_contact_termination_force_threshold_round_trips(tmp_path):
    cfg = load_multi_skill_config(
        _write(tmp_path, _MINIMAL_SKILL + "\ncontact_termination_force_threshold: 50.0\n")
    )
    assert cfg.contact_termination_force_threshold == 50.0


def test_negative_contact_termination_force_threshold_raises(tmp_path):
    with pytest.raises(ValueError, match="contact_termination_force_threshold must be >= 0.0"):
        load_multi_skill_config(
            _write(tmp_path, _MINIMAL_SKILL + "\ncontact_termination_force_threshold: -5.0\n")
        )


# ---------------------------------------------------------------------------------------------
# Distributional-critic value support (2026-08-21) -- critic_v_min/critic_v_max/critic_num_atoms
# ---------------------------------------------------------------------------------------------

_CRITIC_SKILL = (
    "motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, "
    "strike_start_frame: 10, stand_start_frame: 20}"
)


def _critic_yaml(tmp_path, extra: str):
    return _write(tmp_path, f"\nball: {{radius: 0.11, mass: 0.43}}\n{extra}\n{_CRITIC_SKILL}\n")


def test_critic_support_defaults_to_none_when_absent(tmp_path):
    """Unset => None => experiment.py keeps the preset's own 20.0/-20.0/101, bit-identical to
    before these fields existed."""
    cfg = load_multi_skill_config(_critic_yaml(tmp_path, ""))
    assert cfg.critic_v_min is None
    assert cfg.critic_v_max is None
    assert cfg.critic_num_atoms is None


def test_critic_support_parsed_when_present(tmp_path):
    cfg = load_multi_skill_config(
        _critic_yaml(tmp_path, "critic_v_min: -40.0\ncritic_v_max: 40.0\ncritic_num_atoms: 201")
    )
    assert cfg.critic_v_min == -40.0
    assert cfg.critic_v_max == 40.0
    assert cfg.critic_num_atoms == 201
    assert isinstance(cfg.critic_num_atoms, int)


@pytest.mark.parametrize("half", ["critic_v_min: -40.0", "critic_v_max: 40.0"])
def test_critic_support_half_set_raises(tmp_path, half):
    """A support with one end moved and the other left at the preset's value is almost always a
    typo -- must fail rather than silently produce an asymmetric support."""
    with pytest.raises(ValueError, match="must be set together"):
        load_multi_skill_config(_critic_yaml(tmp_path, half))


def test_critic_v_max_not_greater_than_min_raises(tmp_path):
    with pytest.raises(ValueError, match="must be >"):
        load_multi_skill_config(_critic_yaml(tmp_path, "critic_v_min: 40.0\ncritic_v_max: -40.0"))


def test_critic_num_atoms_below_two_raises(tmp_path):
    with pytest.raises(ValueError, match="must be >= 2"):
        load_multi_skill_config(_critic_yaml(tmp_path, "critic_num_atoms: 1"))


def test_shipped_stagec1_skill013_yaml_leaves_critic_support_at_the_preset():
    """Regression guard on a FALSIFIED experiment, not on a feature.

    The +-40 widening was tried (run 20260821_045037) and reverted: the critic did not settle
    inside the wider support, it pinned at the new ceiling for 87% of the run (over-estimating
    3.5x vs 2.1x at +-20), and the sim2sim survival scan went from 0.123 mean fall rate to 0.417
    with three scans at 1.00. See that yaml's own comment block for the full account.

    This asserts all three keys stay UNSET so the preset (+-20 / 101 atoms) governs -- if someone
    re-adds them, this test fires and points them at the comment explaining why it didn't work."""
    from pathlib import Path

    p = Path(__file__).resolve().parents[4].parent / "configs" / "task_config_stageC1-skill013.yaml"
    if not p.exists():
        pytest.skip(f"shipped config not found at {p}")
    raw = __import__("yaml").safe_load(p.read_text())
    for key in ("critic_v_min", "critic_v_max", "critic_num_atoms"):
        assert key not in raw, (
            f"{key} is set again in task_config_stageC1-skill013.yaml -- the +-40 widening was "
            "measured and reverted (sim2sim fall rate 0.123 -> 0.417). Read that file's own "
            "'Distributional-critic value support' comment before re-enabling."
        )


# ---------------------------------------------------------------------------------------------
# ball_velocity's three opt-in retunes, yaml side (2026-08-21). All default to the ORIGINAL
# pre-retune behavior; see shooting.py::ball_velocity's docstring.
# ---------------------------------------------------------------------------------------------


def test_kick_ball_velocity_opt_ins_default_off(tmp_path):
    cfg = load_multi_skill_config(_critic_yaml(tmp_path, ""))
    assert cfg.kick_ball_velocity_v_ref is None, "None => keep the term's own registered 5.0"
    assert cfg.kick_ball_velocity_use_latched_peak_speed is False
    assert cfg.kick_ball_velocity_use_post_locomotion_gate is False


def test_kick_ball_velocity_opt_ins_parsed_when_present(tmp_path):
    cfg = load_multi_skill_config(
        _critic_yaml(
            tmp_path,
            "kick_ball_velocity_v_ref: 2.0\n"
            "kick_ball_velocity_use_latched_peak_speed: true\n"
            "kick_ball_velocity_use_post_locomotion_gate: true",
        )
    )
    assert cfg.kick_ball_velocity_v_ref == 2.0
    assert cfg.kick_ball_velocity_use_latched_peak_speed is True
    assert cfg.kick_ball_velocity_use_post_locomotion_gate is True


@pytest.mark.parametrize("bad", ["0.0", "-1.0"])
def test_kick_ball_velocity_v_ref_must_be_positive(tmp_path, bad):
    """v_ref is squared into a Lorentzian denominator -- 0 would divide by zero for a stationary
    ball, and a negative value is meaningless."""
    with pytest.raises(ValueError, match="must be > 0.0"):
        load_multi_skill_config(_critic_yaml(tmp_path, f"kick_ball_velocity_v_ref: {bad}"))


# ---------------------------------------------------------------------------------------------
# Azimuth-aim refactor (2026-08-22) -- kick_aim_enabled/kick_aim_theta_max_deg/
# kick_aim_theta_ref_deg/kick_aim_nominal_distance_m/kick_error_ball_to_target_sigma
# ---------------------------------------------------------------------------------------------


def _aim_skill_yaml(tmp_path, skill_extra: str = "", global_extra: str = ""):
    return _write(
        tmp_path,
        f"""
ball: {{radius: 0.11, mass: 0.43}}
{global_extra}
motion_skill_1:
  motion_npz: a.npz
  x: 1.0
  y: 0.0
  target_x: 6.0
  target_y: 1.0
  motion_training_ratio: 0.5
  strike_start_frame: 10
  stand_start_frame: 20
  {skill_extra}
""",
    )


def test_kick_aim_globals_default_off(tmp_path):
    cfg = load_multi_skill_config(_aim_skill_yaml(tmp_path))
    assert cfg.kick_aim_theta_ref_deg == 45.0
    assert cfg.kick_aim_theta_max_deg == 15.0
    assert cfg.kick_aim_nominal_distance_m == 5.0
    assert cfg.skills[0].kick_aim_enabled is False
    assert cfg.skills[0].kick_aim_theta_max_deg is None


def test_kick_aim_enabled_parsed_and_bearing_derived(tmp_path):
    cfg = load_multi_skill_config(_aim_skill_yaml(tmp_path, skill_extra="kick_aim_enabled: true"))
    assert cfg.skills[0].kick_aim_enabled is True
    # atan2(1.0 - 0.0, 6.0 - 1.0) = atan2(1, 5) ~= 11.31 deg
    assert abs(cfg.skills[0].resolved_nominal_bearing_deg() - 11.31) < 0.01


@pytest.mark.parametrize("kick_aim_enabled", ["true", "false"])
def test_randomize_target_x_raises_regardless_of_kick_aim_enabled(tmp_path, kick_aim_enabled):
    """randomize_target_x/y was removed entirely 2026-08-22 (azimuth-aim refactor) -- a stray key
    must raise a clear "removed" error, not be silently ignored, whether or not kick_aim_enabled
    happens to be set on the same skill."""
    with pytest.raises(ValueError, match="was removed"):
        load_multi_skill_config(
            _aim_skill_yaml(
                tmp_path, skill_extra=f"kick_aim_enabled: {kick_aim_enabled}\n  randomize_target_x: 1.0"
            )
        )


@pytest.mark.parametrize("bad", ["0.0", "-5.0", "50.0"])
def test_kick_aim_theta_max_deg_out_of_range_raises(tmp_path, bad):
    """Must be in (0.0, kick_aim_theta_ref_deg] -- 0.0/negative is meaningless, and anything above
    the fixed reference would saturate the normalized observation outside [-1, 1]."""
    with pytest.raises(ValueError, match="kick_aim_theta_max_deg"):
        load_multi_skill_config(_aim_skill_yaml(tmp_path, global_extra=f"kick_aim_theta_max_deg: {bad}"))


def test_kick_aim_theta_max_deg_per_skill_override(tmp_path):
    cfg = load_multi_skill_config(
        _aim_skill_yaml(tmp_path, skill_extra="kick_aim_enabled: true\n  kick_aim_theta_max_deg: 25.0")
    )
    assert cfg.skills[0].kick_aim_theta_max_deg == 25.0
    assert cfg.kick_aim_theta_max_deg == 15.0  # global default untouched


def test_kick_aim_theta_max_deg_per_skill_override_out_of_range_raises(tmp_path):
    with pytest.raises(ValueError, match="kick_aim_theta_max_deg"):
        load_multi_skill_config(
            _aim_skill_yaml(tmp_path, skill_extra="kick_aim_enabled: true\n  kick_aim_theta_max_deg: 90.0")
        )


def test_kick_aim_nominal_distance_m_must_be_positive(tmp_path):
    with pytest.raises(ValueError, match="kick_aim_nominal_distance_m must be > 0.0"):
        load_multi_skill_config(_aim_skill_yaml(tmp_path, global_extra="kick_aim_nominal_distance_m: 0.0"))


def test_kick_error_ball_to_target_sigma_default_off(tmp_path):
    cfg = load_multi_skill_config(_critic_yaml(tmp_path, ""))
    assert cfg.kick_error_ball_to_target_sigma is None


@pytest.mark.parametrize("bad", ["0.0", "-1.0"])
def test_kick_error_ball_to_target_sigma_must_be_positive(tmp_path, bad):
    with pytest.raises(ValueError, match="must be > 0.0"):
        load_multi_skill_config(_critic_yaml(tmp_path, f"kick_error_ball_to_target_sigma: {bad}"))
