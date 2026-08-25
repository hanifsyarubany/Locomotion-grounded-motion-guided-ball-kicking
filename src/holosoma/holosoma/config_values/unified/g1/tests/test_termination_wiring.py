"""Integration checks that config_values/unified/g1/termination.py wires
``kick_recovery_termination_handoff`` correctly -- specifically, that the flag ALWAYS brings both
halves of the recovery/hold termination replacement online together (registering
``kick_recovery_low_height`` AND forcing ``bad_tracking_swing_only`` True for the same envs), and
that the pre-existing standalone ``bad_tracking_swing_only`` flag is left independently settable,
unaffected by this new mechanism.

Same subprocess discipline as ``test_multi_skill_wiring.py`` (see that file's own module
docstring): module-level flag resolution in ``config_values/unified/g1/termination.py`` happens
once at Python IMPORT time, so switching yaml/env-var state mid-process via importlib.reload is
fragile -- each check here runs in a FRESH subprocess instead.
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
import holosoma.config_values.unified.g1.termination as termination

terms = termination.g1_29dof_unified_termination.terms
out = {
    "has_kick_recovery_low_height": "kick_recovery_low_height" in terms,
    "has_kick_recovery_drift": "kick_recovery_drift" in terms,
    "bad_tracking_swing_only": terms["bad_tracking"].params["bad_tracking_swing_only"],
}
if "kick_recovery_low_height" in terms:
    t = terms["kick_recovery_low_height"]
    out["kick_recovery_low_height_func"] = t.func
    out["kick_recovery_low_height_params"] = t.params
    out["kick_recovery_low_height_task_mode"] = t.task_mode
if "kick_recovery_drift" in terms:
    t = terms["kick_recovery_drift"]
    out["kick_recovery_drift_func"] = t.func
    out["kick_recovery_drift_params"] = t.params
    out["kick_recovery_drift_task_mode"] = t.task_mode
print(json.dumps(out))
"""


def _write_ball_yaml(tmp_path, extra: str = "") -> Path:
    p = tmp_path / "ball.yaml"
    p.write_text(f"radius: 0.11\nmass: 0.43\nx: 2.84\ny: -0.46\n{extra}")
    return p


def _write_skills_yaml(tmp_path, extra: str = "") -> Path:
    p = tmp_path / "skills.yaml"
    p.write_text(
        "ball: {radius: 0.11, mass: 0.43}\n"
        f"{extra}"
        "motion_skill_1: {motion_npz: a.npz, x: 1.0, y: 0.0, motion_training_ratio: 0.5, "
        "strike_start_frame: 10, stand_start_frame: 20}\n"
    )
    return p


def test_flag_off_by_default_is_true_no_op():
    """No env vars set at all (legacy single-clip path, no HOLOSOMA_BALL_CONFIG/
    HOLOSOMA_SKILLS_CONFIG): kick_recovery_low_height AND its drift sibling must both be absent,
    and bad_tracking_swing_only must resolve to its own bare default (False) -- byte-identical to
    before this feature existed."""
    result = _run_probe(None, _PROBE_CODE)
    assert result["has_kick_recovery_low_height"] is False
    assert result["has_kick_recovery_drift"] is False
    assert result["bad_tracking_swing_only"] is False


def test_flag_on_alone_via_ball_config_adds_terms_without_suppressing_bad_tracking(tmp_path):
    """2026-08-06, user-requested decoupling: setting ONLY the handoff flag (single-skill,
    HOLOSOMA_BALL_CONFIG path) must register BOTH the height term and its drift sibling, WITHOUT
    forcing bad_tracking_swing_only True -- bad_tracking stays fully active during recovery/hold
    by default now. This is the opposite assertion from before the decoupling (see git history /
    the module docstring's "UPDATE 2026-08-06" note for the prior, coupled behavior)."""
    p = _write_ball_yaml(tmp_path, "kick_recovery_termination_handoff: true\n")
    result = _run_probe({"HOLOSOMA_BALL_CONFIG": str(p)}, _PROBE_CODE)
    assert result["has_kick_recovery_low_height"] is True
    assert result["has_kick_recovery_drift"] is True
    assert result["bad_tracking_swing_only"] is False


def test_flag_on_alone_via_multi_skill_config_adds_terms_without_suppressing_bad_tracking(tmp_path):
    """N-skill (HOLOSOMA_SKILLS_CONFIG) path: same guarantee as the single-skill path above."""
    p = _write_skills_yaml(tmp_path, "kick_recovery_termination_handoff: true\n")
    result = _run_probe({"HOLOSOMA_SKILLS_CONFIG": str(p)}, _PROBE_CODE)
    assert result["has_kick_recovery_low_height"] is True
    assert result["has_kick_recovery_drift"] is True
    assert result["bad_tracking_swing_only"] is False


def test_both_flags_together_reproduces_the_original_measured_configuration(tmp_path):
    """Setting BOTH kick_recovery_termination_handoff AND bad_tracking_swing_only explicitly must
    reproduce the ORIGINAL (pre-2026-08-06) coupled behavior exactly -- bad_tracking suppressed for
    recovery/hold, height+drift terms installed -- so the configuration the 2026-08-02 MEASURED
    CONCERNING probe tested remains reachable for regression/reference, just no longer the
    default reached by the handoff flag alone."""
    p = _write_ball_yaml(tmp_path, "kick_recovery_termination_handoff: true\nbad_tracking_swing_only: true\n")
    result = _run_probe({"HOLOSOMA_BALL_CONFIG": str(p)}, _PROBE_CODE)
    assert result["has_kick_recovery_low_height"] is True
    assert result["has_kick_recovery_drift"] is True
    assert result["bad_tracking_swing_only"] is True


def test_standalone_bad_tracking_swing_only_still_independently_settable(tmp_path):
    """Setting ONLY the pre-existing bad_tracking_swing_only flag (handoff flag absent/false) must
    still suppress bad_tracking for recovery/hold on its own, WITHOUT registering either
    replacement term -- proves the old, already-measured-bad-alone knob is preserved unchanged for
    reference/regression use, not silently merged into the new mechanism."""
    p = _write_ball_yaml(tmp_path, "bad_tracking_swing_only: true\n")
    result = _run_probe({"HOLOSOMA_BALL_CONFIG": str(p)}, _PROBE_CODE)
    assert result["bad_tracking_swing_only"] is True
    assert result["has_kick_recovery_low_height"] is False
    assert result["has_kick_recovery_drift"] is False
    assert result["has_kick_recovery_drift"] is False


def test_kick_recovery_low_height_term_params(tmp_path):
    """When on, the registered term's func/params/task_mode must match the plan exactly: the SAME
    min_height/consecutive_steps locomotion's own standing termination uses, grace_steps aligned
    to recovery_duration_s, task_mode='kick' so it never fires for locomotion-mode envs."""
    p = _write_ball_yaml(tmp_path, "kick_recovery_termination_handoff: true\n")
    result = _run_probe({"HOLOSOMA_BALL_CONFIG": str(p)}, _PROBE_CODE)
    assert result["kick_recovery_low_height_func"] == "holosoma.managers.termination.terms.wbt:kick_recovery_low_height_sustained"
    assert result["kick_recovery_low_height_params"] == {
        "min_height": 0.70,
        "consecutive_steps": 10,
        "grace_steps": 50.0,
        "counter_attr": "_kick_recovery_low_height_counter",
    }
    assert result["kick_recovery_low_height_task_mode"] == "kick"


def test_kick_recovery_drift_term_params_default_deadzone(tmp_path):
    """When on, the drift sibling's func/params/task_mode must match: 0.15m default deadzone, same
    consecutive_steps/grace_steps as its height sibling, task_mode='kick'."""
    p = _write_ball_yaml(tmp_path, "kick_recovery_termination_handoff: true\n")
    result = _run_probe({"HOLOSOMA_BALL_CONFIG": str(p)}, _PROBE_CODE)
    assert result["kick_recovery_drift_func"] == "holosoma.managers.termination.terms.wbt:kick_recovery_drift_sustained"
    assert result["kick_recovery_drift_params"] == {
        "deadzone": 0.15,
        "consecutive_steps": 10,
        "grace_steps": 50.0,
        "counter_attr": "_kick_recovery_drift_counter",
        "anchor_attr": "_kick_recovery_drift_anchor_xy",
        "anchor_valid_attr": "_kick_recovery_drift_anchor_valid",
    }
    assert result["kick_recovery_drift_task_mode"] == "kick"


def test_kick_recovery_drift_deadzone_override_via_multi_skill_config(tmp_path):
    """kick_recovery_drift_deadzone must reach the live term's params, independently settable from
    the height check's own min_height."""
    p = _write_skills_yaml(tmp_path, "kick_recovery_termination_handoff: true\nkick_recovery_drift_deadzone: 0.30\n")
    result = _run_probe({"HOLOSOMA_SKILLS_CONFIG": str(p)}, _PROBE_CODE)
    assert result["kick_recovery_drift_params"]["deadzone"] == 0.30
    assert result["kick_recovery_low_height_params"]["min_height"] == 0.70  # unaffected


# ============================================================================================
# bad_motion_body_pos_threshold (2026-08-05, ported from RoboNaldo's progressive ee_body_pos
# termination threshold) -- see MultiSkillConfig.bad_motion_body_pos_threshold's own docstring.
# ============================================================================================

_THRESHOLD_PROBE_CODE = """
import json
import holosoma.config_values.unified.g1.termination as termination

terms = termination.g1_29dof_unified_termination.terms
print(json.dumps({"bad_motion_body_pos_threshold": terms["bad_tracking"].params["bad_motion_body_pos_threshold"]}))
"""


def test_bad_motion_body_pos_threshold_default_matches_preexisting_hardcoded_value():
    """No override set: must resolve to 0.25, the value this key already held in
    g1_29dof_wbt_termination before this field existed -- a true no-op."""
    result = _run_probe(None, _THRESHOLD_PROBE_CODE)
    assert result["bad_motion_body_pos_threshold"] == 0.25


def test_bad_motion_body_pos_threshold_override_via_ball_config(tmp_path):
    p = _write_ball_yaml(tmp_path, "bad_motion_body_pos_threshold: 0.35\n")
    result = _run_probe({"HOLOSOMA_BALL_CONFIG": str(p)}, _THRESHOLD_PROBE_CODE)
    assert result["bad_motion_body_pos_threshold"] == 0.35


def test_bad_motion_body_pos_threshold_override_via_multi_skill_config(tmp_path):
    p = _write_skills_yaml(tmp_path, "bad_motion_body_pos_threshold: 0.5\n")
    result = _run_probe({"HOLOSOMA_SKILLS_CONFIG": str(p)}, _THRESHOLD_PROBE_CODE)
    assert result["bad_motion_body_pos_threshold"] == 0.5


_THRESHOLD_SYNC_PROBE_CODE = """
import json
import holosoma.config_values.unified.g1.termination as termination
import holosoma.config_values.unified.g1.reward as reward

term_threshold = termination.g1_29dof_unified_termination.terms["bad_tracking"].params["bad_motion_body_pos_threshold"]
reward_threshold = reward.g1_29dof_unified_reward.terms["kick_penalty_ee_body_pos_divergence"].params["threshold"]
print(json.dumps({"term_threshold": term_threshold, "reward_threshold": reward_threshold}))
"""


def test_bad_motion_body_pos_threshold_stays_synced_with_reward_side_penalty(tmp_path):
    """The termination's bad_motion_body_pos_threshold and penalty_kick_ee_body_pos_divergence's
    own threshold param must move together under ONE override -- the regression test proving the
    two cannot silently drift apart, mirroring RoboNaldo's own task_overrides.py guarantee."""
    p = _write_skills_yaml(tmp_path, "bad_motion_body_pos_threshold: 0.35\n")
    result = _run_probe({"HOLOSOMA_SKILLS_CONFIG": str(p)}, _THRESHOLD_SYNC_PROBE_CODE)
    assert result["term_threshold"] == 0.35
    assert result["reward_threshold"] == 0.35


_WARMUP_THRESHOLD_PROBE_CODE = """
import json
import holosoma.config_values.unified.g1.reward as reward

warmup_threshold = reward.g1_29dof_unified_reward.terms["kick_penalty_ee_body_pos_divergence"].params["warmup_threshold"]
print(json.dumps({"warmup_threshold": warmup_threshold}))
"""


def test_ee_body_pos_warmup_threshold_default_matches_function_own_default():
    """No override set: must resolve to 0.25 -- the same default
    penalty_kick_ee_body_pos_divergence's own signature already carries -- a true no-op. Caught
    missing entirely (previously silently reached this default REGARDLESS of the yaml, since
    weight/sigma-only override plumbing has no generic per-param path) during a live end-to-end
    verification of configs/task_config_stageC1.yaml -- this field is the fix."""
    result = _run_probe(None, _WARMUP_THRESHOLD_PROBE_CODE)
    assert result["warmup_threshold"] == 0.25


def test_ee_body_pos_warmup_threshold_override_via_multi_skill_config(tmp_path):
    """RoboNaldo's own S2a/S2b value (0.7) must reach the live term -- independently of
    bad_motion_body_pos_threshold, proving the two fields are genuinely decoupled (RoboNaldo's own
    warmup_threshold schedule is non-monotonic, unlike the progressively-widening threshold)."""
    p = _write_skills_yaml(tmp_path, "ee_body_pos_warmup_threshold: 0.7\nbad_motion_body_pos_threshold: 0.35\n")
    result = _run_probe({"HOLOSOMA_SKILLS_CONFIG": str(p)}, _WARMUP_THRESHOLD_PROBE_CODE)
    assert result["warmup_threshold"] == 0.7


# ============================================================================================
# kick_recovery_posture_reward deadzone plumbing (2026-08-06, user-requested) -- see each field's
# own docstring in MultiSkillConfig for the full motivation. _kick_recovery_standing_terms is
# defined BEFORE _multi_skill_cfg_for_contact_penalty exists in reward.py, so these values are
# patched into the already-built dict later in that module -- this wiring test is the regression
# guard that the patch actually reaches the live term's params, not just that the config field
# itself parses (already covered in test_multi_skill.py/test_ball_config_observation_noise.py).
# ============================================================================================

_POSTURE_DEADZONE_PROBE_CODE = """
import json
import holosoma.config_values.unified.g1.reward as reward

terms = reward.g1_29dof_unified_reward.terms
print(json.dumps({
    "stand_height_deadzone": terms["penalty_kick_recovery_stand_height"].params["deadzone"],
    "stand_orientation_deadzone": terms["penalty_kick_recovery_stand_orientation"].params["deadzone"],
    "stand_feet_width_deadzone": terms["penalty_kick_recovery_stand_feet_width"].params["deadzone"],
    "stand_knee_width_deadzone": terms["penalty_kick_recovery_stand_knee_width"].params["deadzone"],
    # target_height/nominal_width were NOT made configurable (user decision) -- confirm they stay
    # at their original hardcoded values regardless, i.e. this change didn't accidentally touch them.
    "stand_height_target_height": terms["penalty_kick_recovery_stand_height"].params["target_height"],
    "stand_feet_width_nominal_width": terms["penalty_kick_recovery_stand_feet_width"].params["nominal_width"],
}))
"""


def test_posture_deadzones_default_match_preexisting_hardcoded_values():
    """No overrides set: all 4 must resolve to their original hardcoded values -- a true no-op --
    and target_height/nominal_width must be untouched (not made configurable)."""
    result = _run_probe(None, _POSTURE_DEADZONE_PROBE_CODE)
    assert result["stand_height_deadzone"] == 0.015
    assert result["stand_orientation_deadzone"] == 0.025
    assert result["stand_feet_width_deadzone"] == 0.03
    assert result["stand_knee_width_deadzone"] == 0.03
    assert result["stand_height_target_height"] == 0.76
    assert result["stand_feet_width_nominal_width"] == 0.24


def test_posture_deadzones_independently_overridable_via_multi_skill_config(tmp_path):
    """Setting all 4 to distinct values must reach each of the 4 live terms independently --
    proves feet_width/knee_width in particular are genuinely separate (not aliased), the specific
    design decision requested for this change."""
    p = _write_skills_yaml(
        tmp_path,
        "kick_recovery_stand_height_deadzone: 0.10\n"
        "kick_recovery_stand_orientation_deadzone: 0.20\n"
        "kick_recovery_stand_feet_width_deadzone: 0.30\n"
        "kick_recovery_stand_knee_width_deadzone: 0.40\n",
    )
    result = _run_probe({"HOLOSOMA_SKILLS_CONFIG": str(p)}, _POSTURE_DEADZONE_PROBE_CODE)
    assert result["stand_height_deadzone"] == 0.10
    assert result["stand_orientation_deadzone"] == 0.20
    assert result["stand_feet_width_deadzone"] == 0.30
    assert result["stand_knee_width_deadzone"] == 0.40


def test_posture_deadzones_override_via_ball_config(tmp_path):
    """Single-skill (HOLOSOMA_BALL_CONFIG) path: same guarantee as the N-skill path above."""
    p = _write_ball_yaml(
        tmp_path,
        "kick_recovery_stand_feet_width_deadzone: 0.30\nkick_recovery_stand_knee_width_deadzone: 0.40\n",
    )
    result = _run_probe({"HOLOSOMA_BALL_CONFIG": str(p)}, _POSTURE_DEADZONE_PROBE_CODE)
    assert result["stand_feet_width_deadzone"] == 0.30
    assert result["stand_knee_width_deadzone"] == 0.40
    # untouched fields stay at their own defaults
    assert result["stand_height_deadzone"] == 0.015
    assert result["stand_orientation_deadzone"] == 0.025


# ============================================================================================
# kick_ball_over_line's require_has_kicked opt-in (2026-08-06, user-requested). Lives in
# _p0_regularization_terms, defined even EARLIER in reward.py than _kick_recovery_standing_terms
# -- this wiring test is the regression guard that the patch reaches the live term's params, not
# just that the config field itself parses (already covered in test_multi_skill.py/
# test_ball_config_observation_noise.py).
# ============================================================================================

_BALL_OVER_LINE_PROBE_CODE = """
import json
import holosoma.config_values.unified.g1.reward as reward

term = reward.g1_29dof_unified_reward.terms["kick_ball_over_line"]
print(json.dumps({"require_has_kicked": term.params["require_has_kicked"]}))
"""


def test_ball_over_line_require_has_kicked_default_is_false():
    """No override set: must resolve to False -- bit-identical to before this param existed,
    matching RoboNaldo's own ungated registration."""
    result = _run_probe(None, _BALL_OVER_LINE_PROBE_CODE)
    assert result["require_has_kicked"] is False


def test_ball_over_line_require_has_kicked_override_via_multi_skill_config(tmp_path):
    p = _write_skills_yaml(tmp_path, "kick_ball_over_line_require_has_kicked: true\n")
    result = _run_probe({"HOLOSOMA_SKILLS_CONFIG": str(p)}, _BALL_OVER_LINE_PROBE_CODE)
    assert result["require_has_kicked"] is True


def test_ball_over_line_require_has_kicked_override_via_ball_config(tmp_path):
    p = _write_ball_yaml(tmp_path, "kick_ball_over_line_require_has_kicked: true\n")
    result = _run_probe({"HOLOSOMA_BALL_CONFIG": str(p)}, _BALL_OVER_LINE_PROBE_CODE)
    assert result["require_has_kicked"] is True
