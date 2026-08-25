"""Unit tests for holosoma/__init__.py's HOLOSOMA_TASK_CONFIG derivation (2026-08-14) -- see that
module's own docstring for why this has to live in the package's own __init__.py rather than
inside train_agent.py's main() (module-level imports elsewhere read the env var before main()'s
body would ever run).

Calls _derive_task_config_from_skills_yaml() directly rather than relying on package-import
side effects -- the function is idempotent/re-entrant by construction (its very first check is
"is HOLOSOMA_TASK_CONFIG already set, if so no-op"), so calling it again after the package's own
one-time import-time run is safe and exercises the exact same logic a fresh process would run.
"""

from __future__ import annotations

import os

import pytest

import holosoma


@pytest.fixture(autouse=True)
def _clean_env():
    """Every test starts from a clean slate for both env vars, restored via a manual
    save/restore -- NOT monkeypatch.setenv/delenv, which only auto-undoes changes made THROUGH
    monkeypatch itself. The function under test writes os.environ["HOLOSOMA_TASK_CONFIG"]
    directly (that's its actual job in production), so monkeypatch has no visibility into that
    mutation and won't undo it -- caught the hard way: running this file's tests alongside
    config_types/tests/test_multi_skill.py's OWN tests broke dozens of them, because a successful
    "derives" test here left HOLOSOMA_TASK_CONFIG set in the real process environment for every
    test that ran afterward, silently flipping load_multi_skill_config into 2-file mode for tests
    written assuming legacy single-file mode."""
    saved = {k: os.environ.get(k) for k in ("HOLOSOMA_TASK_CONFIG", "HOLOSOMA_SKILLS_CONFIG")}
    for k in saved:
        os.environ.pop(k, None)
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def test_noop_when_task_config_already_set_explicitly(monkeypatch):
    monkeypatch.setenv("HOLOSOMA_TASK_CONFIG", "/some/explicit/path.yaml")
    monkeypatch.setenv("HOLOSOMA_SKILLS_CONFIG", "/does/not/matter.yaml")
    holosoma._derive_task_config_from_skills_yaml()
    assert os.environ["HOLOSOMA_TASK_CONFIG"] == "/some/explicit/path.yaml"


def test_noop_when_neither_env_var_set():
    holosoma._derive_task_config_from_skills_yaml()
    assert "HOLOSOMA_TASK_CONFIG" not in os.environ


def test_noop_when_skills_config_points_to_missing_file(monkeypatch):
    monkeypatch.setenv("HOLOSOMA_SKILLS_CONFIG", "/definitely/does/not/exist.yaml")
    holosoma._derive_task_config_from_skills_yaml()  # must not raise -- real loader raises later
    assert "HOLOSOMA_TASK_CONFIG" not in os.environ


def test_noop_when_no_skill_declares_task_config(tmp_path, monkeypatch):
    p = tmp_path / "skills.yaml"
    p.write_text(
        "motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, "
        "strike_start_frame: 10, stand_start_frame: 20}\n"
    )
    monkeypatch.setenv("HOLOSOMA_SKILLS_CONFIG", str(p))
    holosoma._derive_task_config_from_skills_yaml()
    assert "HOLOSOMA_TASK_CONFIG" not in os.environ


def test_derives_task_config_when_single_skill_declares_it(tmp_path, monkeypatch):
    p = tmp_path / "skills.yaml"
    p.write_text(
        "motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, "
        "strike_start_frame: 10, stand_start_frame: 20, task_config: task_config_stageC1}\n"
    )
    monkeypatch.setenv("HOLOSOMA_SKILLS_CONFIG", str(p))
    holosoma._derive_task_config_from_skills_yaml()
    resolved = os.environ["HOLOSOMA_TASK_CONFIG"]
    assert resolved == str(holosoma.resolve_task_config_path("task_config_stageC1"))
    assert os.path.exists(resolved)  # the real, shipped file -- proves this resolves correctly


def test_derives_task_config_when_multiple_skills_agree(tmp_path, monkeypatch):
    p = tmp_path / "skills.yaml"
    p.write_text(
        "motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.3, "
        "strike_start_frame: 10, stand_start_frame: 20, task_config: task_config_stageD}\n"
        "motion_skill_2: {motion_npz: b.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.3, "
        "strike_start_frame: 10, stand_start_frame: 20, task_config: task_config_stageD}\n"
    )
    monkeypatch.setenv("HOLOSOMA_SKILLS_CONFIG", str(p))
    holosoma._derive_task_config_from_skills_yaml()
    assert os.environ["HOLOSOMA_TASK_CONFIG"] == str(holosoma.resolve_task_config_path("task_config_stageD"))


def test_raises_when_skills_disagree_on_task_config(tmp_path, monkeypatch):
    p = tmp_path / "skills.yaml"
    p.write_text(
        "motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.3, "
        "strike_start_frame: 10, stand_start_frame: 20, task_config: task_config_stageB}\n"
        "motion_skill_2: {motion_npz: b.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.3, "
        "strike_start_frame: 10, stand_start_frame: 20, task_config: task_config_stageC1}\n"
    )
    monkeypatch.setenv("HOLOSOMA_SKILLS_CONFIG", str(p))
    with pytest.raises(ValueError, match="different task_config values"):
        holosoma._derive_task_config_from_skills_yaml()
    assert "HOLOSOMA_TASK_CONFIG" not in os.environ


def test_raises_when_derived_file_does_not_exist(tmp_path, monkeypatch):
    p = tmp_path / "skills.yaml"
    p.write_text(
        "motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, "
        "strike_start_frame: 10, stand_start_frame: 20, task_config: task_config_totally_made_up_xyz}\n"
    )
    monkeypatch.setenv("HOLOSOMA_SKILLS_CONFIG", str(p))
    with pytest.raises(FileNotFoundError, match="task_config_totally_made_up_xyz"):
        holosoma._derive_task_config_from_skills_yaml()
    assert "HOLOSOMA_TASK_CONFIG" not in os.environ


def test_top_level_task_config_wins_even_when_skills_disagree(tmp_path, monkeypatch):
    """The 2026-08-15 escape hatch for simultaneous per-skill task configs: a top-level
    `task_config:` field resolves GLOBAL settings even when individual skills genuinely disagree
    (per-skill reward-weight divergence is handled separately, in reward.py)."""
    p = tmp_path / "skills.yaml"
    p.write_text(
        "task_config: task_config_stageC1\n"
        "motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.3, "
        "strike_start_frame: 10, stand_start_frame: 20, task_config: task_config_stageB}\n"
        "motion_skill_2: {motion_npz: b.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.3, "
        "strike_start_frame: 10, stand_start_frame: 20, task_config: task_config_stageC1}\n"
    )
    monkeypatch.setenv("HOLOSOMA_SKILLS_CONFIG", str(p))
    holosoma._derive_task_config_from_skills_yaml()
    assert os.environ["HOLOSOMA_TASK_CONFIG"] == str(holosoma.resolve_task_config_path("task_config_stageC1"))


def test_top_level_task_config_wins_over_a_single_agreeing_skill_too(tmp_path, monkeypatch):
    """Top-level, when present, is authoritative regardless of skill agreement -- not just a
    disagreement tiebreaker."""
    p = tmp_path / "skills.yaml"
    p.write_text(
        "task_config: task_config_stageD\n"
        "motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, "
        "strike_start_frame: 10, stand_start_frame: 20, task_config: task_config_stageC1}\n"
    )
    monkeypatch.setenv("HOLOSOMA_SKILLS_CONFIG", str(p))
    holosoma._derive_task_config_from_skills_yaml()
    assert os.environ["HOLOSOMA_TASK_CONFIG"] == str(holosoma.resolve_task_config_path("task_config_stageD"))


def test_top_level_task_config_raises_when_derived_file_does_not_exist(tmp_path, monkeypatch):
    p = tmp_path / "skills.yaml"
    p.write_text(
        "task_config: task_config_totally_made_up_xyz\n"
        "motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, "
        "strike_start_frame: 10, stand_start_frame: 20}\n"
    )
    monkeypatch.setenv("HOLOSOMA_SKILLS_CONFIG", str(p))
    with pytest.raises(FileNotFoundError, match="task_config_totally_made_up_xyz"):
        holosoma._derive_task_config_from_skills_yaml()
    assert "HOLOSOMA_TASK_CONFIG" not in os.environ


def test_disagreement_error_mentions_the_top_level_field_as_the_fix(tmp_path, monkeypatch):
    p = tmp_path / "skills.yaml"
    p.write_text(
        "motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.3, "
        "strike_start_frame: 10, stand_start_frame: 20, task_config: task_config_stageB}\n"
        "motion_skill_2: {motion_npz: b.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.3, "
        "strike_start_frame: 10, stand_start_frame: 20, task_config: task_config_stageC1}\n"
    )
    monkeypatch.setenv("HOLOSOMA_SKILLS_CONFIG", str(p))
    with pytest.raises(ValueError, match="top-level `task_config:`"):
        holosoma._derive_task_config_from_skills_yaml()


# ----------------------------------------------------------------------------------------------
# 2026-08-16: HOLOSOMA_TASK_CONFIG mismatch guard. Regression tests for the stale-shell-export
# failure that silently ran two `stageB-skill2` runs under Stage D -- see
# _derive_task_config_from_skills_yaml's docstring for the full incident.
# ----------------------------------------------------------------------------------------------

_SKILL2_YAML = (
    "motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.9, "
    "strike_start_frame: 170, stand_start_frame: 215, task_config: task_config_stageB}\n"
)


def test_raises_when_explicit_export_disagrees_with_skills_yaml(tmp_path, monkeypatch):
    """THE regression test: skills yaml says stageB, a stale export says stageD."""
    p = tmp_path / "skill2.yaml"
    p.write_text(_SKILL2_YAML)
    monkeypatch.setenv("HOLOSOMA_SKILLS_CONFIG", str(p))
    monkeypatch.setenv("HOLOSOMA_TASK_CONFIG", "configs/task_config_stageD.yaml")
    with pytest.raises(ValueError, match="HOLOSOMA_TASK_CONFIG mismatch"):
        holosoma._derive_task_config_from_skills_yaml()


def test_mismatch_error_names_both_configs_and_the_fix(tmp_path, monkeypatch):
    p = tmp_path / "skill2.yaml"
    p.write_text(_SKILL2_YAML)
    monkeypatch.setenv("HOLOSOMA_SKILLS_CONFIG", str(p))
    monkeypatch.setenv("HOLOSOMA_TASK_CONFIG", "configs/task_config_stageD.yaml")
    with pytest.raises(ValueError) as ei:
        holosoma._derive_task_config_from_skills_yaml()
    msg = str(ei.value)
    assert "task_config_stageB" in msg and "task_config_stageD" in msg
    assert "unset HOLOSOMA_TASK_CONFIG" in msg
    assert "HOLOSOMA_TASK_CONFIG_OVERRIDE=1" in msg
    assert "STALE" in msg


def test_no_raise_when_explicit_export_agrees(tmp_path, monkeypatch):
    """Same file, expressed as a relative path -- must canonicalize, not string-compare."""
    p = tmp_path / "skills.yaml"
    p.write_text(
        "motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.9, "
        "strike_start_frame: 10, stand_start_frame: 20, task_config: task_config_stageB}\n"
    )
    monkeypatch.setenv("HOLOSOMA_SKILLS_CONFIG", str(p))
    real = holosoma.resolve_task_config_path("task_config_stageB")
    monkeypatch.setenv("HOLOSOMA_TASK_CONFIG", os.path.relpath(real))
    holosoma._derive_task_config_from_skills_yaml()  # must not raise
    assert os.environ["HOLOSOMA_TASK_CONFIG"] == os.path.relpath(real)  # left untouched


def test_override_env_var_permits_deliberate_mismatch(tmp_path, monkeypatch):
    """The real workflow this must not break: multi_skills.yaml declares stageC1, a Stage D
    experiment deliberately exports stageD."""
    p = tmp_path / "multi_skills.yaml"
    p.write_text("task_config: task_config_stageC1\n" + _SKILL2_YAML)
    monkeypatch.setenv("HOLOSOMA_SKILLS_CONFIG", str(p))
    explicit = str(holosoma.resolve_task_config_path("task_config_stageD"))
    monkeypatch.setenv("HOLOSOMA_TASK_CONFIG", explicit)
    monkeypatch.setenv("HOLOSOMA_TASK_CONFIG_OVERRIDE", "1")
    holosoma._derive_task_config_from_skills_yaml()  # must not raise
    assert os.environ["HOLOSOMA_TASK_CONFIG"] == explicit


@pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "on"])
def test_override_accepts_truthy_spellings(tmp_path, monkeypatch, val):
    p = tmp_path / "skills.yaml"
    p.write_text(_SKILL2_YAML)
    monkeypatch.setenv("HOLOSOMA_SKILLS_CONFIG", str(p))
    monkeypatch.setenv("HOLOSOMA_TASK_CONFIG", str(holosoma.resolve_task_config_path("task_config_stageD")))
    monkeypatch.setenv("HOLOSOMA_TASK_CONFIG_OVERRIDE", val)
    holosoma._derive_task_config_from_skills_yaml()  # must not raise


@pytest.mark.parametrize("val", ["0", "false", "no", "", "  "])
def test_override_rejects_falsy_spellings(tmp_path, monkeypatch, val):
    p = tmp_path / "skills.yaml"
    p.write_text(_SKILL2_YAML)
    monkeypatch.setenv("HOLOSOMA_SKILLS_CONFIG", str(p))
    monkeypatch.setenv("HOLOSOMA_TASK_CONFIG", str(holosoma.resolve_task_config_path("task_config_stageD")))
    monkeypatch.setenv("HOLOSOMA_TASK_CONFIG_OVERRIDE", val)
    with pytest.raises(ValueError, match="mismatch"):
        holosoma._derive_task_config_from_skills_yaml()


def test_explicit_export_still_resolves_skill_disagreement_without_override(tmp_path, monkeypatch):
    """Skills disagree AND no top-level field: without an export that's a hard error. WITH one,
    the export legitimately resolves the ambiguity -- there's no single declared value to
    contradict, so it must not require the override flag."""
    p = tmp_path / "skills.yaml"
    p.write_text(
        "motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.3, "
        "strike_start_frame: 10, stand_start_frame: 20, task_config: task_config_stageB}\n"
        "motion_skill_2: {motion_npz: b.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.3, "
        "strike_start_frame: 10, stand_start_frame: 20, task_config: task_config_stageC1}\n"
    )
    monkeypatch.setenv("HOLOSOMA_SKILLS_CONFIG", str(p))
    explicit = str(holosoma.resolve_task_config_path("task_config_stageD"))
    monkeypatch.setenv("HOLOSOMA_TASK_CONFIG", explicit)
    holosoma._derive_task_config_from_skills_yaml()  # must not raise
    assert os.environ["HOLOSOMA_TASK_CONFIG"] == explicit


def test_legacy_single_file_mode_with_explicit_export_unaffected(monkeypatch):
    """No skills yaml at all -- the guard must never fire."""
    monkeypatch.setenv("HOLOSOMA_TASK_CONFIG", "/some/explicit/path.yaml")
    holosoma._derive_task_config_from_skills_yaml()
    assert os.environ["HOLOSOMA_TASK_CONFIG"] == "/some/explicit/path.yaml"
