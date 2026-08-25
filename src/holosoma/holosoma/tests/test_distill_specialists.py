"""Unit tests for distill_specialists.py's pure-Python pieces.

`parse_teacher_ckpts`/`parse_teacher_ckpts_from_skills_yaml` are testable without a live IsaacSim
env + real checkpoints -- everything else in that module (Teacher construction, the DAgger loop)
needs a real env/GPU. See distill_specialists.py's own module docstring for the design this
implements.
"""

from __future__ import annotations

import pytest

from holosoma.distill_specialists import parse_teacher_ckpts, parse_teacher_ckpts_from_skills_yaml


def test_parses_two_entries():
    result = parse_teacher_ckpts("0=logs/a/model_0400000.pt,1=logs/b/model_0119000.pt")
    assert result == {0: "logs/a/model_0400000.pt", 1: "logs/b/model_0119000.pt"}


def test_parses_single_entry():
    assert parse_teacher_ckpts("0=logs/a/model_0400000.pt") == {0: "logs/a/model_0400000.pt"}


def test_strips_whitespace_around_entries_and_values():
    result = parse_teacher_ckpts(" 0 = logs/a/model.pt , 1 = logs/b/model.pt ")
    assert result == {0: "logs/a/model.pt", 1: "logs/b/model.pt"}


def test_tolerates_trailing_comma():
    result = parse_teacher_ckpts("0=logs/a/model.pt,1=logs/b/model.pt,")
    assert result == {0: "logs/a/model.pt", 1: "logs/b/model.pt"}


def test_empty_string_raises():
    # No longer phrased as "required" -- the env var is now optional (falls back to
    # parse_teacher_ckpts_from_skills_yaml), but calling this directly with "" must still raise:
    # an empty explicit override is never valid on its own.
    with pytest.raises(ValueError, match="must declare at least one teacher"):
        parse_teacher_ckpts("")


def test_missing_equals_raises():
    with pytest.raises(ValueError, match="missing '='"):
        parse_teacher_ckpts("0-logs/a/model.pt")


def test_non_integer_skill_id_raises():
    with pytest.raises(ValueError, match="not a non-negative integer"):
        parse_teacher_ckpts("skill1=logs/a/model.pt")


def test_duplicate_skill_id_raises():
    with pytest.raises(ValueError, match="declared twice"):
        parse_teacher_ckpts("0=logs/a/model.pt,0=logs/b/model.pt")


def test_empty_path_raises():
    with pytest.raises(ValueError, match="empty path"):
        parse_teacher_ckpts("0=")


def test_non_dense_skill_ids_raise():
    """0,2 (skipping 1) would silently index-error at runtime (teachers[env.skill_id]) rather
    than fail at parse time -- this must be caught here instead."""
    with pytest.raises(ValueError, match="not a dense 0..N-1 range"):
        parse_teacher_ckpts("0=logs/a/model.pt,2=logs/b/model.pt")


def test_ids_not_starting_at_zero_raise():
    with pytest.raises(ValueError, match="not a dense 0..N-1 range"):
        parse_teacher_ckpts("1=logs/a/model.pt,2=logs/b/model.pt")


def test_windows_style_or_unusual_but_valid_paths_pass_through_unmodified():
    """Paths can legitimately contain characters other than '=' and ',' -- only the FIRST '=' in
    each entry is meaningful (split(...,1)), everything after it is the path verbatim."""
    result = parse_teacher_ckpts("0=/abs/path with spaces/model=checkpoint.pt")
    assert result == {0: "/abs/path with spaces/model=checkpoint.pt"}


# ----------------------------------------------------------------------------------------------
# parse_teacher_ckpts_from_skills_yaml -- the yaml-field alternative to the long env-var export.
# ----------------------------------------------------------------------------------------------

_TWO_SKILL_YAML_WITH_TEACHERS = """
task_config: task_config_stageC1
base_robot:
  target_height: 0.76
  deadzone: 0.015
motion_skill_1:
  motion_npz: a.npz
  x: 1.0
  y: 0.0
  strike_start_frame: 10
  stand_start_frame: 20
  motion_training_ratio: 0.45
  teacher_checkpoint: logs/skill1_ckpt.pt
motion_skill_2:
  motion_npz: b.npz
  x: 1.0
  y: 0.0
  strike_start_frame: 10
  stand_start_frame: 20
  motion_training_ratio: 0.45
  teacher_checkpoint: logs/skill2_ckpt.pt
"""


def test_reads_teacher_checkpoint_per_skill_in_declaration_order(tmp_path):
    p = tmp_path / "skills.yaml"
    p.write_text(_TWO_SKILL_YAML_WITH_TEACHERS)
    result = parse_teacher_ckpts_from_skills_yaml(str(p))
    assert result == {0: "logs/skill1_ckpt.pt", 1: "logs/skill2_ckpt.pt"}


def test_order_follows_yaml_declaration_not_alphabetical_block_names(tmp_path):
    """motion_skill_2 declared BEFORE motion_skill_1 in the file -- skill_id must follow file
    order (dict/yaml.safe_load insertion order), matching _parse_skill_blocks' own contract, NOT
    numeric/alphabetical sort of the block names."""
    p = tmp_path / "skills.yaml"
    p.write_text(
        "motion_skill_2:\n"
        "  motion_npz: b.npz\n  x: 1.0\n  y: 0.0\n  strike_start_frame: 10\n  stand_start_frame: 20\n"
        "  motion_training_ratio: 0.45\n  teacher_checkpoint: logs/second_declared.pt\n"
        "motion_skill_1:\n"
        "  motion_npz: a.npz\n  x: 1.0\n  y: 0.0\n  strike_start_frame: 10\n  stand_start_frame: 20\n"
        "  motion_training_ratio: 0.45\n  teacher_checkpoint: logs/first_declared.pt\n"
    )
    result = parse_teacher_ckpts_from_skills_yaml(str(p))
    assert result == {0: "logs/second_declared.pt", 1: "logs/first_declared.pt"}


def test_single_skill(tmp_path):
    p = tmp_path / "skills.yaml"
    p.write_text(
        "motion_skill_1:\n"
        "  motion_npz: a.npz\n  x: 1.0\n  y: 0.0\n  strike_start_frame: 10\n  stand_start_frame: 20\n"
        "  motion_training_ratio: 0.9\n  teacher_checkpoint: logs/only.pt\n"
    )
    assert parse_teacher_ckpts_from_skills_yaml(str(p)) == {0: "logs/only.pt"}


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        parse_teacher_ckpts_from_skills_yaml(str(tmp_path / "nope.yaml"))


def test_no_skill_blocks_raises(tmp_path):
    p = tmp_path / "skills.yaml"
    p.write_text("task_config: task_config_stageC1\nbase_robot: {target_height: 0.76, deadzone: 0.015}\n")
    with pytest.raises(ValueError, match="no 'motion_skill_N' blocks"):
        parse_teacher_ckpts_from_skills_yaml(str(p))


def test_missing_teacher_checkpoint_field_raises_naming_the_block(tmp_path):
    p = tmp_path / "skills.yaml"
    p.write_text(
        "motion_skill_1:\n"
        "  motion_npz: a.npz\n  x: 1.0\n  y: 0.0\n  strike_start_frame: 10\n  stand_start_frame: 20\n"
        "  motion_training_ratio: 0.9\n"  # no teacher_checkpoint
    )
    with pytest.raises(ValueError, match=r"motion_skill_1.*no 'teacher_checkpoint:'"):
        parse_teacher_ckpts_from_skills_yaml(str(p))


def test_empty_teacher_checkpoint_field_raises(tmp_path):
    p = tmp_path / "skills.yaml"
    p.write_text(
        "motion_skill_1:\n"
        "  motion_npz: a.npz\n  x: 1.0\n  y: 0.0\n  strike_start_frame: 10\n  stand_start_frame: 20\n"
        "  motion_training_ratio: 0.9\n  teacher_checkpoint: ''\n"
    )
    with pytest.raises(ValueError, match="is empty"):
        parse_teacher_ckpts_from_skills_yaml(str(p))


def test_the_real_distill_config_file_parses_end_to_end():
    """Not a synthetic fixture -- the actual configs/skill/distill_skill1_skill2.yaml this project
    uses, proving the real file (and its currently-declared checkpoint paths) parses cleanly."""
    result = parse_teacher_ckpts_from_skills_yaml("configs/skill/distill_skill1_skill2.yaml")
    assert set(result.keys()) == {0, 1}
    assert all(v.endswith(".pt") for v in result.values())
