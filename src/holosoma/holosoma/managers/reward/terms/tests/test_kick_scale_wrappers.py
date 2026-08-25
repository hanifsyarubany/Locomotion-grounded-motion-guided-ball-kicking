"""Unit tests for kick_scale_wrappers.py's 7 wrapper functions (kick_alive_scaled + the 6
motion-tracking *_scaled functions). Each wrapper must call through to the ORIGINAL, UNMODIFIED
base function (locomotion.alive / wbt.motion_*) and multiply its output by the live per-skill
scale from kick_reward_scales.py -- verified here by mocking the base function to a KNOWN tensor
(isolating the wrapping/multiply logic from the base functions' own, unrelated env-data
requirements) and injecting a controlled scale cache, same pattern as
utils/tests/test_kick_reward_scales.py.

Also covers _swing_widened_sigma (2026-08-01) and its threading through the 6 motion-tracking
wrappers via the new swing_tracking_sigma_multiplier parameter: verified by inspecting what SIGMA
value each wrapper actually calls its base function with (mock_base.call_args), not the wrapper's
output -- the widening happens to an INPUT, not an output multiply, so assertions here check the
call arguments rather than the returned tensor, unlike every test above this point in the file.

Also covers _post_flip_tracking_decay_multiplier (FIX 4, 2026-08-12) in isolation (its own
env-attribute-driven branches) and its composition into one of the 7 wrappers end-to-end."""

from __future__ import annotations
from types import SimpleNamespace
from unittest.mock import patch

import torch

import holosoma.managers.reward.terms.kick_scale_wrappers as w
import holosoma.utils.kick_reward_scales as krs


def _fake_env(
    motion_ids: torch.Tensor,
    in_kicking_phase: torch.Tensor | None = None,
    in_strike_phase: torch.Tensor | None = None,
    has_ball: bool = True,
):
    # Default in_kicking_phase=True for every env: the safe "no effect from the new recovery-phase
    # gate" default, so tests that only care about motion_tracking_reward_scale (unrelated to the
    # swing/recovery split) don't need to think about it.
    if in_kicking_phase is None:
        in_kicking_phase = torch.ones_like(motion_ids, dtype=torch.bool)
    # Default in_strike_phase=False for every env: the safe "no effect from swing-sigma-widening"
    # default (matches _swing_widened_sigma's own no-op-unless-multiplier!=1.0 short circuit, so
    # this is only ever READ by tests that explicitly set a non-1.0 multiplier).
    if in_strike_phase is None:
        in_strike_phase = torch.zeros_like(motion_ids, dtype=torch.bool)
    # has_ball defaults to True: matches real production (Unified's MotionCommand in kick mode is
    # always the ball-kick variant, has_ball=True by construction -- same precedent
    # _recovery_phase_tracking_multiplier's own docstring already documents). Only
    # kick_alive_scaled's kick_alive_pre_kick_ratio path reads this attribute today; every other
    # wrapper in this file ignores it, so this default doesn't change any other test's behavior.
    motion_command = SimpleNamespace(
        motion_ids=motion_ids,
        in_kicking_phase=in_kicking_phase,
        in_strike_phase=in_strike_phase,
        has_ball=has_ball,
    )
    command_manager = SimpleNamespace(get_state=lambda name: motion_command if name == "motion_command" else None)
    return SimpleNamespace(command_manager=command_manager)


def setup_function(_fn):
    krs._cached = None


def teardown_function(_fn):
    krs._cached = None


def test_kick_alive_scaled_multiplies_base_alive_by_kick_alive_scale():
    krs._cached = {
        "motion_tracking": [1.0],
        "recovery_tracking": [1.0],
        "kick_recovery_posture": [1.0],
        "kick_safety": [1.0],
        "kick_alive": [2.5],
        "kick_alive_pre_kick_ratio": [1.0],
    }
    env = _fake_env(motion_ids=torch.tensor([0, 0]))
    with patch.object(w, "_alive", return_value=torch.tensor([10.0, 10.0])) as mock_alive:
        out = w.kick_alive_scaled(env)
    mock_alive.assert_called_once_with(env)
    assert torch.allclose(out, torch.tensor([25.0, 25.0]))


def test_kick_alive_scaled_default_1_0_is_a_no_op():
    krs._cached = {cat: [1.0] for cat in krs._CATEGORIES}
    env = _fake_env(motion_ids=torch.tensor([0]))
    with patch.object(w, "_alive", return_value=torch.tensor([10.0])):
        out = w.kick_alive_scaled(env)
    assert torch.allclose(out, torch.tensor([10.0]))


def test_kick_alive_pre_kick_ratio_default_1_0_never_reads_motion_command():
    """At the default ratio==1.0, kick_alive_scaled must take the exact no-op path and never touch
    command_manager at all -- proven here by NOT providing has_ball on the fake motion_command
    (would raise if the phase-read branch were reached) and asserting the output is bit-identical
    to base*kick_alive_reward_scale alone."""
    krs._cached = {
        "motion_tracking": [1.0],
        "recovery_tracking": [1.0],
        "kick_recovery_posture": [1.0],
        "kick_safety": [1.0],
        "kick_alive": [2.0],
        "kick_alive_pre_kick_ratio": [1.0],
    }
    env = _fake_env(motion_ids=torch.tensor([0, 0]), has_ball=False)
    with patch.object(w, "_alive", return_value=torch.tensor([10.0, 10.0])):
        out = w.kick_alive_scaled(env)
    assert torch.allclose(out, torch.tensor([20.0, 20.0]))


def test_kick_alive_pre_kick_ratio_applies_during_kicking_phase():
    krs._cached = {
        "motion_tracking": [1.0],
        "recovery_tracking": [1.0],
        "kick_recovery_posture": [1.0],
        "kick_safety": [1.0],
        "kick_alive": [1.0],
        "kick_alive_pre_kick_ratio": [0.1],
    }
    env = _fake_env(motion_ids=torch.tensor([0]), in_kicking_phase=torch.tensor([True]))
    with patch.object(w, "_alive", return_value=torch.tensor([10.0])):
        out = w.kick_alive_scaled(env)
    assert torch.allclose(out, torch.tensor([1.0])), f"expected 10.0 * 1.0(kick_alive_scale) * 0.1(pre_kick_ratio) = 1.0, got {out}"


def test_kick_alive_pre_kick_ratio_noop_after_kicking_phase():
    """Same nonzero ratio as above, but in_kicking_phase=False (post-kick recovery/hold) -- the
    ratio must have NO effect there, only kick_alive_reward_scale applies."""
    krs._cached = {
        "motion_tracking": [1.0],
        "recovery_tracking": [1.0],
        "kick_recovery_posture": [1.0],
        "kick_safety": [1.0],
        "kick_alive": [1.0],
        "kick_alive_pre_kick_ratio": [0.1],
    }
    env = _fake_env(motion_ids=torch.tensor([0]), in_kicking_phase=torch.tensor([False]))
    with patch.object(w, "_alive", return_value=torch.tensor([10.0])):
        out = w.kick_alive_scaled(env)
    assert torch.allclose(out, torch.tensor([10.0])), f"expected pre_kick_ratio to have no effect post-kick, got {out}"


def test_kick_alive_pre_kick_ratio_per_env_mixed_phase():
    """Two envs, same skill (same ratio), different phases -- confirms the gate is applied per-env
    via torch.where, not as a single scalar branch (same convention as
    test_recovery_tracking_scale_is_per_env_mixed_phase above)."""
    krs._cached = {
        "motion_tracking": [1.0],
        "recovery_tracking": [1.0],
        "kick_recovery_posture": [1.0],
        "kick_safety": [1.0],
        "kick_alive": [1.0],
        "kick_alive_pre_kick_ratio": [0.1],
    }
    env = _fake_env(motion_ids=torch.tensor([0, 0]), in_kicking_phase=torch.tensor([True, False]))
    with patch.object(w, "_alive", return_value=torch.tensor([10.0, 10.0])):
        out = w.kick_alive_scaled(env)
    assert torch.allclose(out, torch.tensor([1.0, 10.0])), f"expected [1.0 (pre-kick, x0.1), 10.0 (post-kick, untouched)], got {out}"


def test_kick_alive_pre_kick_ratio_has_ball_false_falls_back_to_base():
    """A non-1.0 ratio on a motion_command whose has_ball is False (the legacy/non-ball-kick
    MotionCommand variant) must also fall back to base -- mirrors the guard
    _recovery_phase_tracking_multiplier's own docstring documents NOT needing (because Unified
    kick envs always have has_ball=True), but kick_alive_scaled is reached for the shared `alive`
    function regardless of task mode, so this guard is load-bearing here, unlike there. (No
    corresponding "motion_command is None" test: kick_alive_reward_scale, called earlier in the
    same function via _per_env_scale, already unconditionally dereferences motion_command.motion_ids,
    so a missing motion_command is not a reachable scenario for this function at all.)"""
    krs._cached = {
        "motion_tracking": [1.0],
        "recovery_tracking": [1.0],
        "kick_recovery_posture": [1.0],
        "kick_safety": [1.0],
        "kick_alive": [1.0],
        "kick_alive_pre_kick_ratio": [0.1],
    }
    env = _fake_env(motion_ids=torch.tensor([0]), in_kicking_phase=torch.tensor([True]), has_ball=False)
    with patch.object(w, "_alive", return_value=torch.tensor([10.0])):
        out = w.kick_alive_scaled(env)
    assert torch.allclose(out, torch.tensor([10.0]))


def test_motion_tracking_wrappers_multiply_base_output_by_motion_tracking_scale():
    # in_kicking_phase defaults to True (see _fake_env) -- recovery_tracking_scale therefore has NO
    # effect here regardless of its value, isolating motion_tracking_reward_scale's own behavior.
    # root_tracking left at 1.0 for both skills (a no-op) -- isolates motion_tracking_reward_scale
    # even for the 2 root-term wrappers in the loop below.
    krs._cached = {
        "motion_tracking": [1.0, 0.5],
        "root_tracking": [1.0, 1.0],
        "recovery_tracking": [1.0, 1.0],
        "kick_recovery_posture": [1.0, 1.0],
        "kick_safety": [1.0, 1.0],
        "kick_alive": [1.0, 1.0],
    }
    env = _fake_env(motion_ids=torch.tensor([0, 1]))
    cases = [
        ("_motion_global_ref_position_error_exp", w.motion_global_ref_position_error_exp_scaled),
        ("_motion_global_ref_orientation_error_exp", w.motion_global_ref_orientation_error_exp_scaled),
        ("_motion_relative_body_position_error_exp", w.motion_relative_body_position_error_exp_scaled),
        ("_motion_relative_body_orientation_error_exp", w.motion_relative_body_orientation_error_exp_scaled),
        ("_motion_global_body_lin_vel", w.motion_global_body_lin_vel_scaled),
        ("_motion_global_body_ang_vel", w.motion_global_body_ang_vel_scaled),
        # motion_global_feet_lin_vel_scaled also forwards body_names (defaults to None) -- the
        # only one of the 7 with this extra parameter, so its call-shape assertion differs below.
        ("_motion_global_feet_lin_vel", w.motion_global_feet_lin_vel_scaled),
    ]
    for base_name, wrapped_fn in cases:
        with patch.object(w, base_name, return_value=torch.tensor([1.0, 1.0])) as mock_base:
            out = wrapped_fn(env, sigma=0.35)
        if base_name == "_motion_global_feet_lin_vel":
            mock_base.assert_called_once_with(env, 0.35, body_names=None)
        else:
            mock_base.assert_called_once_with(env, 0.35)
        assert torch.allclose(out, torch.tensor([1.0, 0.5])), f"{base_name} failed: {out}"


def test_motion_global_feet_lin_vel_scaled_forwards_custom_body_names():
    """Regression guard for a real bug caught by live IsaacSim verification (2026-08-05): the
    wrapper originally had no body_names parameter at all, so registering the term with an
    explicit body_names in its params (config_values/wbt/g1/reward.py) crashed with
    'unexpected keyword argument body_names' the first time it ran in kick mode -- unit tests
    alone (mocking the base function) never exercised the real call signature closely enough to
    catch it. This pins the forwarding explicitly, with a non-default value."""
    krs._cached = {cat: [1.0] for cat in krs._CATEGORIES}
    env = _fake_env(motion_ids=torch.tensor([0]))
    custom_names = ["left_knee_link", "right_knee_link"]
    with patch.object(w, "_motion_global_feet_lin_vel", return_value=torch.tensor([1.0])) as mock_base:
        w.motion_global_feet_lin_vel_scaled(env, sigma=0.4, body_names=custom_names)
    mock_base.assert_called_once_with(env, 0.4, body_names=custom_names)


def test_root_tracking_scale_applies_only_to_the_2_root_terms():
    """root_tracking_reward_scale (2026-08-05) must multiply ONLY
    motion_global_ref_position_error_exp_scaled/_orientation_error_exp_scaled -- the other 4
    wrappers (2 relative-body pose + 2 global body-velocity) must be UNAFFECTED by it, even though
    they all share the same motion_tracking_reward_scale."""
    krs._cached = {
        "motion_tracking": [1.0],
        "root_tracking": [0.1],
        "recovery_tracking": [1.0],
        "kick_recovery_posture": [1.0],
        "kick_safety": [1.0],
        "kick_alive": [1.0],
    }
    env = _fake_env(motion_ids=torch.tensor([0]))
    root_cases = [
        ("_motion_global_ref_position_error_exp", w.motion_global_ref_position_error_exp_scaled),
        ("_motion_global_ref_orientation_error_exp", w.motion_global_ref_orientation_error_exp_scaled),
    ]
    for base_name, wrapped_fn in root_cases:
        with patch.object(w, base_name, return_value=torch.tensor([1.0])):
            out = wrapped_fn(env, sigma=0.3)
        assert torch.allclose(out, torch.tensor([0.1])), f"{base_name}: expected 1.0*1.0(motion_tracking)*0.1(root_tracking)=0.1, got {out}"

    relative_cases = [
        ("_motion_relative_body_position_error_exp", w.motion_relative_body_position_error_exp_scaled),
        ("_motion_relative_body_orientation_error_exp", w.motion_relative_body_orientation_error_exp_scaled),
        ("_motion_global_body_lin_vel", w.motion_global_body_lin_vel_scaled),
        ("_motion_global_body_ang_vel", w.motion_global_body_ang_vel_scaled),
    ]
    for base_name, wrapped_fn in relative_cases:
        with patch.object(w, base_name, return_value=torch.tensor([1.0])):
            out = wrapped_fn(env, sigma=0.3)
        assert torch.allclose(out, torch.tensor([1.0])), f"{base_name}: root_tracking_reward_scale must NOT apply, got {out}"


def test_root_tracking_scale_default_1_0_is_a_byte_identical_no_op():
    """At the default (1.0), the root terms' effective multiplier must be exactly
    motion_tracking_reward_scale(env) -- byte-identical to before this field existed."""
    krs._cached = {
        "motion_tracking": [0.5],
        "root_tracking": [1.0],
        "recovery_tracking": [1.0],
        "kick_recovery_posture": [1.0],
        "kick_safety": [1.0],
        "kick_alive": [1.0],
    }
    env = _fake_env(motion_ids=torch.tensor([0]))
    with patch.object(w, "_motion_global_ref_position_error_exp", return_value=torch.tensor([10.0])):
        out = w.motion_global_ref_position_error_exp_scaled(env, sigma=0.3)
    assert torch.allclose(out, torch.tensor([5.0])), f"expected 10.0*0.5(motion_tracking)*1.0(root, no-op)=5.0, got {out}"


def test_root_tracking_scale_per_env_independent():
    """Two envs on different skills, different root_tracking_reward_scale values -- confirms the
    gather is per-env via motion_ids, not a shared scalar."""
    krs._cached = {
        "motion_tracking": [1.0, 1.0],
        "root_tracking": [1.0, 0.1],
        "recovery_tracking": [1.0, 1.0],
        "kick_recovery_posture": [1.0, 1.0],
        "kick_safety": [1.0, 1.0],
        "kick_alive": [1.0, 1.0],
    }
    env = _fake_env(motion_ids=torch.tensor([0, 1]))
    with patch.object(w, "_motion_global_ref_orientation_error_exp", return_value=torch.tensor([1.0, 1.0])):
        out = w.motion_global_ref_orientation_error_exp_scaled(env, sigma=0.4)
    assert torch.allclose(out, torch.tensor([1.0, 0.1])), f"expected [1.0 (skill0, no-op), 0.1 (skill1, scaled)], got {out}"


def test_recovery_tracking_scale_is_noop_during_swing():
    krs._cached = {
        "motion_tracking": [1.0],
        "recovery_tracking": [0.25],  # would matter a lot if NOT phase-gated
        "kick_recovery_posture": [1.0],
        "kick_safety": [1.0],
        "kick_alive": [1.0],
    }
    env = _fake_env(motion_ids=torch.tensor([0]), in_kicking_phase=torch.tensor([True]))
    with patch.object(w, "_motion_relative_body_position_error_exp", return_value=torch.tensor([2.0])):
        out = w.motion_relative_body_position_error_exp_scaled(env, sigma=0.3)
    assert torch.allclose(out, torch.tensor([2.0])), "recovery_tracking_scale must not apply during swing"


def test_recovery_tracking_scale_applies_after_swing():
    krs._cached = {
        "motion_tracking": [1.0],
        "recovery_tracking": [0.25],
        "kick_recovery_posture": [1.0],
        "kick_safety": [1.0],
        "kick_alive": [1.0],
    }
    env = _fake_env(motion_ids=torch.tensor([0]), in_kicking_phase=torch.tensor([False]))
    with patch.object(w, "_motion_relative_body_position_error_exp", return_value=torch.tensor([2.0])):
        out = w.motion_relative_body_position_error_exp_scaled(env, sigma=0.3)
    assert torch.allclose(out, torch.tensor([0.5])), f"expected 2.0 * 1.0(motion_tracking) * 0.25(recovery) = 0.5, got {out}"


def test_recovery_tracking_scale_is_per_env_mixed_phase():
    # Two envs, same skill (same recovery_tracking_scale=0.25), different phases -- confirms the
    # gate is applied per-env via torch.where, not as a single scalar branch.
    krs._cached = {
        "motion_tracking": [1.0],
        "recovery_tracking": [0.25],
        "kick_recovery_posture": [1.0],
        "kick_safety": [1.0],
        "kick_alive": [1.0],
    }
    env = _fake_env(motion_ids=torch.tensor([0, 0]), in_kicking_phase=torch.tensor([True, False]))
    with patch.object(w, "_motion_global_body_lin_vel", return_value=torch.tensor([4.0, 4.0])):
        out = w.motion_global_body_lin_vel_scaled(env, sigma=1.0)
    assert torch.allclose(out, torch.tensor([4.0, 1.0])), f"expected [4.0 (swing, untouched), 1.0 (recovery, x0.25)], got {out}"


def test_swing_widened_sigma_default_1_0_is_a_byte_identical_no_op():
    """swing_tracking_sigma_multiplier's default (1.0) must pass sigma through UNCHANGED -- same
    exact call as before this mechanism existed, proving the no-op path isn't just numerically
    close, it's the identical float. Deliberately reuses this exact assertion style from
    test_motion_tracking_wrappers_multiply_base_output_by_motion_tracking_scale above, now also
    covering every wrapper's new (unset) parameter."""
    krs._cached = {cat: [1.0] for cat in krs._CATEGORIES}
    env = _fake_env(motion_ids=torch.tensor([0]))
    cases = [
        ("_motion_global_ref_position_error_exp", w.motion_global_ref_position_error_exp_scaled),
        ("_motion_global_ref_orientation_error_exp", w.motion_global_ref_orientation_error_exp_scaled),
        ("_motion_relative_body_position_error_exp", w.motion_relative_body_position_error_exp_scaled),
        ("_motion_relative_body_orientation_error_exp", w.motion_relative_body_orientation_error_exp_scaled),
        ("_motion_global_body_lin_vel", w.motion_global_body_lin_vel_scaled),
        ("_motion_global_body_ang_vel", w.motion_global_body_ang_vel_scaled),
        ("_motion_global_feet_lin_vel", w.motion_global_feet_lin_vel_scaled),
    ]
    for base_name, wrapped_fn in cases:
        with patch.object(w, base_name, return_value=torch.tensor([1.0])) as mock_base:
            wrapped_fn(env, sigma=0.3)
        if base_name == "_motion_global_feet_lin_vel":
            mock_base.assert_called_once_with(env, 0.3, body_names=None)
        else:
            mock_base.assert_called_once_with(env, 0.3), f"{base_name}: sigma changed at the default multiplier"


def test_swing_widened_sigma_widens_during_strike_phase():
    env = _fake_env(motion_ids=torch.tensor([0]), in_strike_phase=torch.tensor([True]))
    with patch.object(w, "_motion_relative_body_position_error_exp", return_value=torch.tensor([1.0])) as mock_base:
        w.motion_relative_body_position_error_exp_scaled(env, sigma=0.3, swing_tracking_sigma_multiplier=2.0)
    called_sigma = mock_base.call_args.args[1]
    assert torch.allclose(called_sigma, torch.tensor([0.6])), f"expected sigma widened to 0.3*2.0=0.6, got {called_sigma}"


def test_swing_widened_sigma_unchanged_outside_strike_phase():
    """Same multiplier as the widening test above, but in_strike_phase=False -- proves the
    widening is genuinely phase-gated, not just "multiplier != 1.0 -> always widen"."""
    env = _fake_env(motion_ids=torch.tensor([0]), in_strike_phase=torch.tensor([False]))
    with patch.object(w, "_motion_relative_body_position_error_exp", return_value=torch.tensor([1.0])) as mock_base:
        w.motion_relative_body_position_error_exp_scaled(env, sigma=0.3, swing_tracking_sigma_multiplier=2.0)
    called_sigma = mock_base.call_args.args[1]
    assert torch.allclose(called_sigma, torch.tensor([0.3])), f"expected sigma unchanged at 0.3, got {called_sigma}"


def test_swing_widened_sigma_per_env_mixed_phase():
    """Two envs, one in strike phase and one not, same multiplier -- proves the widening is
    applied elementwise via torch.where, not as a single scalar short-circuit (same rationale as
    every other per-env mixed-phase test in this project)."""
    env = _fake_env(motion_ids=torch.tensor([0, 0]), in_strike_phase=torch.tensor([True, False]))
    with patch.object(
        w, "_motion_global_body_lin_vel", return_value=torch.tensor([1.0, 1.0])
    ) as mock_base:
        w.motion_global_body_lin_vel_scaled(env, sigma=1.0, swing_tracking_sigma_multiplier=3.0)
    called_sigma = mock_base.call_args.args[1]
    assert torch.allclose(called_sigma, torch.tensor([3.0, 1.0])), (
        f"expected [3.0 (strike, widened x3), 1.0 (not strike, unchanged)], got {called_sigma}"
    )


def test_swing_widened_sigma_composes_with_motion_tracking_scale_and_recovery_gate():
    """The widening changes what sigma the base function SEES; motion_tracking_reward_scale and
    the recovery-phase gate still apply to the OUTPUT exactly as before -- the two mechanisms are
    independent layers, verified together here so a future refactor can't accidentally couple
    them."""
    krs._cached = {
        "motion_tracking": [0.5],
        "recovery_tracking": [1.0],
        "kick_recovery_posture": [1.0],
        "kick_safety": [1.0],
        "kick_alive": [1.0],
    }
    env = _fake_env(
        motion_ids=torch.tensor([0]), in_kicking_phase=torch.tensor([True]), in_strike_phase=torch.tensor([True])
    )
    with patch.object(w, "_motion_global_body_ang_vel", return_value=torch.tensor([10.0])) as mock_base:
        out = w.motion_global_body_ang_vel_scaled(env, sigma=3.14, swing_tracking_sigma_multiplier=1.5)
    called_sigma = mock_base.call_args.args[1]
    assert torch.allclose(called_sigma, torch.tensor([4.71])), f"expected sigma 3.14*1.5=4.71, got {called_sigma}"
    assert torch.allclose(out, torch.tensor([5.0])), f"expected output 10.0*0.5(motion_tracking)=5.0, got {out}"


def _fake_post_flip_env(currently_kick: torch.Tensor, post_flip_step: torch.Tensor, episode_length_buf: torch.Tensor):
    is_post_flip = post_flip_step >= 0
    steps_since = torch.where(
        is_post_flip, (episode_length_buf - post_flip_step).clamp(min=0), torch.zeros_like(episode_length_buf)
    ).float()
    return SimpleNamespace(
        task_mode_mask=lambda name: currently_kick if name == "kick" else ~currently_kick,
        _post_flip_step=post_flip_step,
        post_flip_steps_since=lambda: steps_since,
    )


def test_post_flip_decay_default_zero_steps_is_a_bare_float_one_no_op():
    """decay_steps<=0.0 (the default) must return the exact python float 1.0 -- not just a
    numerically-equal tensor -- and must never touch env at all, proven by passing a bare
    object() (no task_mode_mask/_post_flip_step) that would AttributeError if read."""
    out = w._post_flip_tracking_decay_multiplier(object(), post_flip_reward_decay_steps=0.0)
    assert out == 1.0 and isinstance(out, float)


def test_post_flip_decay_missing_env_attrs_falls_back_to_1_0():
    """A non-UnifiedManager env (no task_mode_mask / _post_flip_step) must fall back to the exact
    no-op float even with a nonzero decay window -- this mechanism only ever means something under
    UnifiedManager's flip machinery."""
    out = w._post_flip_tracking_decay_multiplier(object(), post_flip_reward_decay_steps=50.0)
    assert out == 1.0 and isinstance(out, float)


def test_post_flip_decay_is_1_0_for_envs_currently_in_kick_mode():
    env = _fake_post_flip_env(
        currently_kick=torch.tensor([True]), post_flip_step=torch.tensor([-1]), episode_length_buf=torch.tensor([5])
    )
    out = w._post_flip_tracking_decay_multiplier(env, post_flip_reward_decay_steps=50.0)
    assert torch.allclose(out, torch.tensor([1.0]))


def test_post_flip_decay_ramps_linearly_within_the_window():
    env = _fake_post_flip_env(
        currently_kick=torch.tensor([False]),
        post_flip_step=torch.tensor([0]),
        episode_length_buf=torch.tensor([10]),  # 10 steps since flip
    )
    out = w._post_flip_tracking_decay_multiplier(env, post_flip_reward_decay_steps=50.0)
    assert torch.allclose(out, torch.tensor([0.8])), f"expected 1.0 - 10/50 = 0.8, got {out}"


def test_post_flip_decay_clamped_to_zero_past_the_window():
    env = _fake_post_flip_env(
        currently_kick=torch.tensor([False]),
        post_flip_step=torch.tensor([0]),
        episode_length_buf=torch.tensor([100]),  # far past a 50-step window
    )
    out = w._post_flip_tracking_decay_multiplier(env, post_flip_reward_decay_steps=50.0)
    assert torch.allclose(out, torch.tensor([0.0]))


def test_post_flip_decay_is_zero_for_an_env_that_never_flipped():
    """A genuinely-locomotion env (sentinel -1, never touched by the flip) currently NOT in kick
    mode must get exactly 0.0, not the ramp -- it never had a swing worth bootstrapping through."""
    env = _fake_post_flip_env(
        currently_kick=torch.tensor([False]),
        post_flip_step=torch.tensor([-1]),
        episode_length_buf=torch.tensor([1000]),
    )
    out = w._post_flip_tracking_decay_multiplier(env, post_flip_reward_decay_steps=50.0)
    assert torch.allclose(out, torch.tensor([0.0]))


def test_post_flip_decay_per_env_independent_mixed_states():
    """Three envs in one batch -- currently-kick, freshly-flipped, and never-flipped -- confirms
    the three branches are applied elementwise via nested torch.where, not a shared scalar."""
    env = _fake_post_flip_env(
        currently_kick=torch.tensor([True, False, False]),
        post_flip_step=torch.tensor([-1, 0, -1]),
        episode_length_buf=torch.tensor([5, 25, 5]),
    )
    out = w._post_flip_tracking_decay_multiplier(env, post_flip_reward_decay_steps=50.0)
    assert torch.allclose(out, torch.tensor([1.0, 0.5, 0.0])), f"got {out}"


def test_post_flip_decay_composes_into_a_wrapper_end_to_end():
    """Wires post_flip_reward_decay_steps through motion_relative_body_position_error_exp_scaled
    (one of the 5 non-root wrappers) end-to-end: base output * motion_tracking_reward_scale *
    recovery-phase multiplier (both 1.0, isolated away) * the post-flip decay ramp."""
    krs._cached = {cat: [1.0] for cat in krs._CATEGORIES}
    motion_command = SimpleNamespace(
        motion_ids=torch.tensor([0]), in_kicking_phase=torch.tensor([False]), in_strike_phase=torch.tensor([False])
    )
    command_manager = SimpleNamespace(get_state=lambda name: motion_command if name == "motion_command" else None)
    env = SimpleNamespace(
        command_manager=command_manager,
        task_mode_mask=lambda name: torch.tensor([False]) if name == "kick" else torch.tensor([True]),
        _post_flip_step=torch.tensor([0]),
        post_flip_steps_since=lambda: torch.tensor([10.0]),
    )
    with patch.object(w, "_motion_relative_body_position_error_exp", return_value=torch.tensor([5.0])):
        out = w.motion_relative_body_position_error_exp_scaled(env, sigma=0.3, post_flip_reward_decay_steps=50.0)
    assert torch.allclose(out, torch.tensor([4.0])), f"expected 5.0 * 1.0(motion_tracking) * 1.0(recovery, post-kick but scale=1.0) * 0.8(decay: 1-10/50) = 4.0, got {out}"


def _fake_pre_kick_env(pre_kick_step: torch.Tensor, episode_length_buf: torch.Tensor, post_flip_step: torch.Tensor | None = None):
    is_pre_kick = pre_kick_step >= 0
    steps_since = torch.where(
        is_pre_kick, (episode_length_buf - pre_kick_step).clamp(min=0), torch.zeros_like(episode_length_buf)
    ).float()
    if post_flip_step is None:
        post_flip_step = torch.full_like(pre_kick_step, -1)
    return SimpleNamespace(
        _pre_kick_step=pre_kick_step,
        pre_kick_steps_since=lambda: steps_since,
        _post_flip_step=post_flip_step,
    )


def test_pre_kick_ramp_default_zero_steps_is_a_bare_float_one_no_op():
    """ramp_steps<=0.0 (the default) must return the exact python float 1.0 -- not just a
    numerically-equal tensor -- and must never touch env at all, proven by passing a bare
    object() (no pre_kick_steps_since/_pre_kick_step) that would AttributeError if read."""
    out = w._pre_kick_reward_ramp_multiplier(object(), pre_kick_reward_ramp_steps=0.0)
    assert out == 1.0 and isinstance(out, float)


def test_pre_kick_ramp_missing_env_attrs_falls_back_to_1_0():
    """A non-UnifiedManager env (no pre_kick_steps_since / _pre_kick_step) must fall back to the
    exact no-op float even with a nonzero ramp window -- this mechanism only ever means something
    under UnifiedManager's mid-episode entry machinery."""
    out = w._pre_kick_reward_ramp_multiplier(object(), pre_kick_reward_ramp_steps=50.0)
    assert out == 1.0 and isinstance(out, float)


def test_pre_kick_ramp_is_1_0_for_an_env_that_never_had_a_mid_episode_entry():
    """An ordinary teleport-at-reset kick env (sentinel -1) must get exactly 1.0, unaffected --
    matching today's behavior exactly regardless of how the window is configured."""
    env = _fake_pre_kick_env(pre_kick_step=torch.tensor([-1]), episode_length_buf=torch.tensor([1000]))
    out = w._pre_kick_reward_ramp_multiplier(env, pre_kick_reward_ramp_steps=50.0)
    assert torch.allclose(out, torch.tensor([1.0]))


def test_pre_kick_ramp_ramps_linearly_within_the_window():
    env = _fake_pre_kick_env(
        pre_kick_step=torch.tensor([0]),
        episode_length_buf=torch.tensor([10]),  # 10 steps since the mid-episode entry
    )
    out = w._pre_kick_reward_ramp_multiplier(env, pre_kick_reward_ramp_steps=50.0)
    assert torch.allclose(out, torch.tensor([0.2])), f"expected 10/50 = 0.2, got {out}"


def test_pre_kick_ramp_clamped_to_1_0_past_the_window():
    env = _fake_pre_kick_env(
        pre_kick_step=torch.tensor([0]),
        episode_length_buf=torch.tensor([100]),  # far past a 50-step window
    )
    out = w._pre_kick_reward_ramp_multiplier(env, pre_kick_reward_ramp_steps=50.0)
    assert torch.allclose(out, torch.tensor([1.0]))


def test_pre_kick_ramp_d5_guard_leaves_a_post_flip_env_untouched():
    """D5: an env simultaneously inside FIX 4's own post-flip decay window must be left at 1.0
    here regardless of its (structurally-impossible-in-practice, but not assumed) pre-kick state
    -- never double-ramped."""
    env = _fake_pre_kick_env(
        pre_kick_step=torch.tensor([0]),
        episode_length_buf=torch.tensor([10]),
        post_flip_step=torch.tensor([0]),
    )
    out = w._pre_kick_reward_ramp_multiplier(env, pre_kick_reward_ramp_steps=50.0)
    assert torch.allclose(out, torch.tensor([1.0]))


def test_pre_kick_ramp_per_env_independent_mixed_states():
    """Three envs in one batch -- never-mid-episode, freshly-entered, and past-the-window --
    confirms the branches are applied elementwise via torch.where, not a shared scalar."""
    env = _fake_pre_kick_env(
        pre_kick_step=torch.tensor([-1, 0, 0]),
        episode_length_buf=torch.tensor([5, 25, 100]),
    )
    out = w._pre_kick_reward_ramp_multiplier(env, pre_kick_reward_ramp_steps=50.0)
    assert torch.allclose(out, torch.tensor([1.0, 0.5, 1.0])), f"got {out}"


def test_pre_kick_ramp_composes_into_a_wrapper_end_to_end():
    """Wires pre_kick_reward_ramp_steps through motion_relative_body_position_error_exp_scaled
    (one of the 5 non-root wrappers) end-to-end: base output * motion_tracking_reward_scale *
    recovery-phase multiplier (both 1.0, isolated away) * the pre-kick ramp."""
    krs._cached = {cat: [1.0] for cat in krs._CATEGORIES}
    motion_command = SimpleNamespace(
        motion_ids=torch.tensor([0]), in_kicking_phase=torch.tensor([True]), in_strike_phase=torch.tensor([False])
    )
    command_manager = SimpleNamespace(get_state=lambda name: motion_command if name == "motion_command" else None)
    env = SimpleNamespace(
        command_manager=command_manager,
        _pre_kick_step=torch.tensor([0]),
        pre_kick_steps_since=lambda: torch.tensor([10.0]),
        _post_flip_step=torch.tensor([-1]),
    )
    with patch.object(w, "_motion_relative_body_position_error_exp", return_value=torch.tensor([5.0])):
        out = w.motion_relative_body_position_error_exp_scaled(env, sigma=0.3, pre_kick_reward_ramp_steps=50.0)
    assert torch.allclose(out, torch.tensor([1.0])), f"expected 5.0 * 1.0(motion_tracking) * 1.0(recovery, in_kicking_phase) * 0.2(ramp: 10/50) = 1.0, got {out}"
