"""Unit tests for config_types/reward_tuning.py, including the 2026-08-05 HOLOSOMA_TASK_CONFIG
lenient-top-level-parsing addition (see DEFAULT_STRICT_TOP_LEVEL's own docstring)."""

from __future__ import annotations

import pytest

from holosoma.config_types.reward_tuning import (
    load_per_skill_reward_weight_overrides,
    load_per_skill_top_level_override,
    load_reward_sigma_overrides,
    load_reward_weight_overrides,
    resolve_per_skill_param,
)


def _write(tmp_path, name: str, contents: str):
    p = tmp_path / name
    p.write_text(contents)
    return p


# ============================================================================================
# Pre-existing (strict) behavior -- explicit yaml_path, default strict_top_level=True.
# ============================================================================================


def test_flattens_five_categories_into_one_dict(tmp_path):
    p = _write(
        tmp_path,
        "tuning.yaml",
        """
motion_tracking_reward:
  motion_global_ref_position_error_exp: 1.0
shooting_reward:
  kick_ball_proximity: 2.0
""",
    )
    overrides = load_reward_weight_overrides(p)
    assert overrides == {"motion_global_ref_position_error_exp": 1.0, "kick_ball_proximity": 2.0}


def test_missing_file_returns_empty_dict(tmp_path):
    assert load_reward_weight_overrides(tmp_path / "does_not_exist.yaml") == {}


def test_unrecognized_top_level_section_raises_when_strict(tmp_path):
    p = _write(tmp_path, "tuning.yaml", "not_a_real_category:\n  foo: 1.0\n")
    with pytest.raises(ValueError, match="unrecognized top-level"):
        load_reward_weight_overrides(p)


def test_term_in_two_categories_raises(tmp_path):
    p = _write(
        tmp_path,
        "tuning.yaml",
        """
motion_tracking_reward:
  dup: 1.0
shooting_reward:
  dup: 2.0
""",
    )
    with pytest.raises(ValueError, match="appears in both"):
        load_reward_weight_overrides(p)


def test_sigma_overrides_flatten_the_reserved_subsection(tmp_path):
    p = _write(
        tmp_path,
        "tuning.yaml",
        """
shooting_reward:
  kick_error_ball_to_target: 20.0
  _sigma:
    kick_error_ball_to_target: 4.2
""",
    )
    assert load_reward_sigma_overrides(p) == {"kick_error_ball_to_target": 4.2}
    # _sigma itself must never be treated as a term name in the weight loader
    assert "_sigma" not in load_reward_weight_overrides(p)


# ============================================================================================
# 2-file mode: strict_top_level=False tolerates a merged task-config file's many other keys.
# ============================================================================================


def test_lenient_mode_ignores_non_reward_top_level_keys(tmp_path):
    p = _write(
        tmp_path,
        "task_config.yaml",
        """
ball: {radius: 0.11, mass: 0.43}
kick_gamma: 0.99
randomize_x: 0.35
motion_tracking_reward:
  motion_global_ref_position_error_exp: 0.5
""",
    )
    overrides = load_reward_weight_overrides(p, strict_top_level=False)
    assert overrides == {"motion_global_ref_position_error_exp": 0.5}


def test_strict_mode_still_raises_on_the_same_merged_file(tmp_path):
    """The exact same file that's fine in lenient mode must still raise in strict mode -- proves
    the two modes are genuinely different, not that validation was silently dropped altogether."""
    p = _write(
        tmp_path,
        "task_config.yaml",
        """
ball: {radius: 0.11, mass: 0.43}
motion_tracking_reward:
  motion_global_ref_position_error_exp: 0.5
""",
    )
    with pytest.raises(ValueError, match="unrecognized top-level"):
        load_reward_weight_overrides(p, strict_top_level=True)


def test_lenient_mode_still_catches_a_term_in_two_categories(tmp_path):
    """Lenient only relaxes the TOP-LEVEL unknown-key check -- within-category validation (a term
    appearing twice) still applies."""
    p = _write(
        tmp_path,
        "task_config.yaml",
        """
ball: {radius: 0.11, mass: 0.43}
motion_tracking_reward:
  dup: 1.0
shooting_reward:
  dup: 2.0
""",
    )
    with pytest.raises(ValueError, match="appears in both"):
        load_reward_weight_overrides(p, strict_top_level=False)


def test_default_strict_top_level_follows_holosoma_task_config_env_var(tmp_path, monkeypatch):
    """DEFAULT_STRICT_TOP_LEVEL is resolved once at import time, but the per-call default
    parameter still reflects it -- reload the module fresh under each env state to check both
    directions rather than trusting whatever state happened to be resolved at collection time."""
    import importlib

    p = _write(
        tmp_path,
        "task_config.yaml",
        """
ball: {radius: 0.11, mass: 0.43}
motion_tracking_reward:
  motion_global_ref_position_error_exp: 0.5
""",
    )

    monkeypatch.setenv("HOLOSOMA_TASK_CONFIG", "irrelevant-for-this-check.yaml")
    import holosoma.config_types.reward_tuning as reward_tuning_module

    importlib.reload(reward_tuning_module)
    try:
        assert reward_tuning_module.DEFAULT_STRICT_TOP_LEVEL is False
        # explicit yaml_path + the reloaded module's own default strict_top_level=False
        overrides = reward_tuning_module.load_reward_weight_overrides(p)
        assert overrides == {"motion_global_ref_position_error_exp": 0.5}
    finally:
        monkeypatch.delenv("HOLOSOMA_TASK_CONFIG", raising=False)
        importlib.reload(reward_tuning_module)
        assert reward_tuning_module.DEFAULT_STRICT_TOP_LEVEL is True


# ============================================================================================
# load_per_skill_reward_weight_overrides (2026-08-15, "simultaneous per-skill task configs")
# ============================================================================================


def test_per_skill_loads_one_dict_per_path_in_order(tmp_path):
    p1 = _write(tmp_path, "c1.yaml", "shooting_reward:\n  kick_ball_proximity: 2.0\n")
    p2 = _write(tmp_path, "b.yaml", "shooting_reward:\n  kick_ball_proximity: 0.0\n")

    result = load_per_skill_reward_weight_overrides([p1, p2])

    assert result == [{"kick_ball_proximity": 2.0}, {"kick_ball_proximity": 0.0}]


def test_per_skill_none_path_yields_empty_dict_for_that_skill(tmp_path):
    """A skill that declares no task_config: at all contributes NO overrides of its own -- same
    meaning "no override" already has for the single-file case (it simply inherits whatever the
    global config says for every term)."""
    p1 = _write(tmp_path, "c1.yaml", "shooting_reward:\n  kick_ball_proximity: 2.0\n")

    result = load_per_skill_reward_weight_overrides([p1, None])

    assert result == [{"kick_ball_proximity": 2.0}, {}]


def test_per_skill_uses_lenient_top_level_parsing(tmp_path):
    """Real task_config files carry many non-reward-tuning keys (ball, ood_*, kick_gamma, ...) --
    per-skill loading must tolerate them the same way the global loader does when reading from a
    merged HOLOSOMA_TASK_CONFIG file (strict_top_level=False), not raise on every extra key."""
    p = _write(
        tmp_path,
        "full_task_config.yaml",
        """
ball: {radius: 0.11, mass: 0.43}
kick_gamma: 0.99
shooting_reward:
  kick_ball_proximity: 2.0
""",
    )

    result = load_per_skill_reward_weight_overrides([p])

    assert result == [{"kick_ball_proximity": 2.0}]


def test_per_skill_missing_file_is_a_safe_empty_dict(tmp_path):
    result = load_per_skill_reward_weight_overrides([tmp_path / "does_not_exist.yaml"])
    assert result == [{}]


def test_per_skill_empty_list_returns_empty_list():
    assert load_per_skill_reward_weight_overrides([]) == []


# ============================================================================================
# load_per_skill_top_level_override (2026-08-15, "Tier 2 Mechanism A" -- deadzones/thresholds/
# contact-force shape params, distinct from the 5 nested reward-tuning categories above)
# ============================================================================================


def test_top_level_reads_field_at_root(tmp_path):
    p = _write(tmp_path, "task_config.yaml", "kick_recovery_stand_height_deadzone: 0.05\n")
    result = load_per_skill_top_level_override([p], "kick_recovery_stand_height_deadzone")
    assert result == [0.05]


def test_top_level_missing_field_is_none(tmp_path):
    p = _write(tmp_path, "task_config.yaml", "kick_gamma: 0.99\n")
    result = load_per_skill_top_level_override([p], "kick_recovery_stand_height_deadzone")
    assert result == [None]


def test_top_level_none_path_is_none():
    result = load_per_skill_top_level_override([None], "kick_recovery_stand_height_deadzone")
    assert result == [None]


def test_top_level_tolerates_a_full_task_config_files_other_keys(tmp_path):
    p = _write(
        tmp_path,
        "task_config.yaml",
        """
ball: {radius: 0.11, mass: 0.43}
kick_gamma: 0.99
kick_recovery_stand_height_deadzone: 0.05
shooting_reward:
  kick_ball_proximity: 2.0
""",
    )
    result = load_per_skill_top_level_override([p], "kick_recovery_stand_height_deadzone")
    assert result == [0.05]


def test_top_level_one_entry_per_path_in_order(tmp_path):
    p1 = _write(tmp_path, "a.yaml", "kick_recovery_stand_height_deadzone: 0.05\n")
    p2 = _write(tmp_path, "b.yaml", "kick_recovery_stand_height_deadzone: 0.02\n")
    result = load_per_skill_top_level_override([p1, None, p2], "kick_recovery_stand_height_deadzone")
    assert result == [0.05, None, 0.02]


# ============================================================================================
# resolve_per_skill_param (shared no-op/fallback algorithm behind reward.py's _per_skill_param
# and termination.py's own kick_recovery_drift_deadzone use)
# ============================================================================================


def test_resolve_none_paths_returns_none():
    assert resolve_per_skill_param(None, "kick_recovery_stand_height_deadzone", 0.02) is None


def test_resolve_single_path_returns_none():
    assert resolve_per_skill_param(["only_one.yaml"], "kick_recovery_stand_height_deadzone", 0.02) is None


def test_resolve_agreeing_paths_returns_none(tmp_path):
    """Two DIFFERENT files that happen to produce the SAME resolved value for this field must
    still return None -- no per-skill table needed when there's no genuine divergence."""
    p1 = _write(tmp_path, "a.yaml", "kick_recovery_stand_height_deadzone: 0.05\n")
    p2 = _write(tmp_path, "b.yaml", "kick_recovery_stand_height_deadzone: 0.05\n")
    assert resolve_per_skill_param([p1, p2], "kick_recovery_stand_height_deadzone", 0.02) is None


def test_resolve_genuine_divergence_returns_the_per_skill_list(tmp_path):
    p1 = _write(tmp_path, "a.yaml", "kick_recovery_stand_height_deadzone: 0.05\n")
    p2 = _write(tmp_path, "b.yaml", "kick_recovery_stand_height_deadzone: 0.02\n")
    assert resolve_per_skill_param([p1, p2], "kick_recovery_stand_height_deadzone", 0.99) == [0.05, 0.02]


def test_resolve_a_skill_with_no_override_falls_back_to_base_value(tmp_path):
    p1 = _write(tmp_path, "a.yaml", "kick_recovery_stand_height_deadzone: 0.05\n")
    p2 = _write(tmp_path, "b.yaml", "kick_gamma: 0.99\n")  # no override for this field
    assert resolve_per_skill_param([p1, p2], "kick_recovery_stand_height_deadzone", 0.02) == [0.05, 0.02]
