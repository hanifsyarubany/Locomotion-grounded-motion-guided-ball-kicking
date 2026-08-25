"""Unit tests for BadTracking's per-skill support (2026-08-15, "Mechanism B" of "simultaneous
per-skill task configs"): bad_motion_body_pos_threshold and swing_threshold_multiplier are read
ONCE into self.x at __init__ (a STATEFUL term), so they can't use TerminationManager's generic
per-call params_per_skill override -- BadTracking reads cfg.params_per_skill directly itself
instead (handles_params_per_skill=True) and gathers by env.skill_id inside
_swing_widened_threshold/bad_motion_body_pos at call time.

Fixture style follows test_kick_recovery_drift.py's own precedent (lightweight fakes, not a real
env), extended with the pieces BadTracking.__init__/bad_motion_body_pos actually touch: device,
skill_id, body_pos_relative_w/robot_body_pos_w.
"""

from __future__ import annotations

import pytest
import torch

from holosoma.config_types.termination import TerminationTermCfg
from holosoma.managers.termination.terms import wbt as wbt_terms

_BODY_NAMES = ["a", "b"]


def _cfg(params_per_skill=None, swing_threshold_multiplier: float = 1.0, bad_motion_body_pos_threshold: float = 0.25):
    return TerminationTermCfg(
        func="holosoma.managers.termination.terms.wbt:BadTracking",
        params={
            "bad_ref_pos_threshold": 1.0,
            "bad_ref_ori_threshold": 1.0,
            "bad_motion_body_pos_body_names": _BODY_NAMES,
            "body_names_to_track": _BODY_NAMES,
            "bad_motion_body_pos_threshold": bad_motion_body_pos_threshold,
            "bad_object_pos_threshold": 1.0,
            "bad_object_ori_threshold": 1.0,
            "swing_threshold_multiplier": swing_threshold_multiplier,
        },
        params_per_skill=params_per_skill,
    )


def _fake_env(num_envs: int, skill_id: torch.Tensor | None = None, device: str = "cpu"):
    from types import SimpleNamespace

    kwargs = dict(device=device)
    if skill_id is not None:
        kwargs["skill_id"] = skill_id
    return SimpleNamespace(**kwargs)


def _fake_motion_command(num_envs: int, has_ball: bool, in_kicking_phase: torch.Tensor):
    from types import SimpleNamespace

    return SimpleNamespace(has_ball=has_ball, in_kicking_phase=in_kicking_phase)


def test_no_params_per_skill_is_byte_identical_to_before():
    """Legacy path: no per-skill tensors built, _swing_widened_threshold returns the plain float
    unchanged when multiplier==1.0, same as before this mechanism existed."""
    env = _fake_env(num_envs=3)
    term = wbt_terms.BadTracking(_cfg(), env)

    assert term._bad_motion_body_pos_threshold_per_skill is None
    assert term._swing_threshold_multiplier_per_skill is None

    mc = _fake_motion_command(3, has_ball=True, in_kicking_phase=torch.tensor([True, False, True]))
    out = term._swing_widened_threshold(0.25, mc)
    assert out == 0.25  # plain float, not a tensor


def test_legacy_scalar_widening_matches_old_formula():
    """multiplier != 1.0, no per-skill tables -- must match the exact old torch.full_like-based
    formula numerically."""
    env = _fake_env(num_envs=3)
    term = wbt_terms.BadTracking(_cfg(swing_threshold_multiplier=2.0), env)
    in_swing = torch.tensor([True, False, True])
    mc = _fake_motion_command(3, has_ball=True, in_kicking_phase=in_swing)

    out = term._swing_widened_threshold(0.25, mc)
    expected = torch.where(in_swing, torch.full_like(in_swing, 0.5, dtype=torch.float32), torch.full_like(in_swing, 0.25, dtype=torch.float32))
    assert torch.allclose(out, expected)


def test_no_ball_returns_base_threshold_unchanged_even_with_per_skill_multiplier():
    env = _fake_env(num_envs=2, skill_id=torch.tensor([0, 1]))
    term = wbt_terms.BadTracking(
        _cfg(params_per_skill={"swing_threshold_multiplier": [2.0, 3.0]}), env
    )
    mc = _fake_motion_command(2, has_ball=False, in_kicking_phase=torch.tensor([True, True]))
    out = term._swing_widened_threshold(0.25, mc)
    assert out == 0.25


def test_per_skill_multiplier_gathers_by_skill_id():
    """skill 0 -> multiplier 2.0, skill 1 -> multiplier 3.0; both envs in swing."""
    env = _fake_env(num_envs=2, skill_id=torch.tensor([0, 1]))
    term = wbt_terms.BadTracking(
        _cfg(params_per_skill={"swing_threshold_multiplier": [2.0, 3.0]}), env
    )
    in_swing = torch.tensor([True, True])
    mc = _fake_motion_command(2, has_ball=True, in_kicking_phase=in_swing)

    out = term._swing_widened_threshold(0.25, mc)
    assert torch.allclose(out, torch.tensor([0.5, 0.75]))


def test_per_skill_multiplier_only_widens_envs_currently_in_swing():
    env = _fake_env(num_envs=2, skill_id=torch.tensor([0, 1]))
    term = wbt_terms.BadTracking(
        _cfg(params_per_skill={"swing_threshold_multiplier": [2.0, 3.0]}), env
    )
    in_swing = torch.tensor([True, False])  # env 1 is past swing
    mc = _fake_motion_command(2, has_ball=True, in_kicking_phase=in_swing)

    out = term._swing_widened_threshold(0.25, mc)
    assert torch.allclose(out, torch.tensor([0.5, 0.25]))  # env 1 unwidened


def test_bad_motion_body_pos_threshold_gathers_by_skill_id():
    """skill 0's threshold (0.10) vs skill 1's (0.90) -- error of 0.5 fires for skill 0, not
    skill 1, with swing widening off (multiplier=1.0) to isolate the base-threshold gather."""
    env = _fake_env(num_envs=2, skill_id=torch.tensor([0, 1]))
    term = wbt_terms.BadTracking(
        _cfg(params_per_skill={"bad_motion_body_pos_threshold": [0.10, 0.90]}), env
    )
    mc_state = _fake_motion_command(2, has_ball=False, in_kicking_phase=torch.tensor([False, False]))
    mc_state.body_pos_relative_w = torch.zeros(2, 2, 3)
    mc_state.robot_body_pos_w = torch.zeros(2, 2, 3)
    mc_state.body_pos_relative_w[:, 1, 0] = 0.5  # 0.5m error on body index 1, both envs

    out = term.bad_motion_body_pos(mc_state)
    assert torch.equal(out, torch.tensor([True, False]))


# ============================================================================================
# 2026-08-15: BadTrackingZOnly (managers/termination/terms/wbt.py) is a BadTracking subclass that
# OVERRIDES bad_motion_body_pos for a z-axis-only comparison -- and this project's own exp presets
# (config_values/unified/g1/termination.py) register THIS class, not the plain BadTracking parent
# tested above. The override's first version read self.bad_motion_body_pos_threshold directly
# (the plain scalar attribute) instead of routing through self._per_env like the parent class's
# own bad_motion_body_pos does -- silently ignoring bad_motion_body_pos_threshold's per-skill
# table entirely. Caught via a real 2-skill training launch (multi_skills.yaml, genuinely
# divergent bad_motion_body_pos_threshold) rather than by this file's own pre-existing coverage,
# which only ever exercised the parent class.
# ============================================================================================


def test_z_only_variant_bad_motion_body_pos_threshold_gathers_by_skill_id():
    """Mirror of test_bad_motion_body_pos_threshold_gathers_by_skill_id above, but against
    BadTrackingZOnly specifically -- the class actually registered in this project's own exp
    presets, not just its parent."""
    env = _fake_env(num_envs=2, skill_id=torch.tensor([0, 1]))
    term = wbt_terms.BadTrackingZOnly(
        _cfg(params_per_skill={"bad_motion_body_pos_threshold": [0.10, 0.90]}), env
    )
    mc_state = _fake_motion_command(2, has_ball=False, in_kicking_phase=torch.tensor([False, False]))
    mc_state.body_pos_relative_w = torch.zeros(2, 2, 3)
    mc_state.robot_body_pos_w = torch.zeros(2, 2, 3)
    mc_state.body_pos_relative_w[:, 1, -1] = 0.5  # 0.5m Z error on body index 1, both envs

    out = term.bad_motion_body_pos(mc_state)
    assert torch.equal(out, torch.tensor([True, False]))


def test_z_only_variant_no_per_skill_table_uses_plain_scalar():
    """Legacy path: no per-skill table -- the plain scalar threshold still applies uniformly to
    both envs, byte-identical to before this mechanism existed (and to before this bug fix)."""
    env = _fake_env(num_envs=2, skill_id=torch.tensor([0, 1]))
    term = wbt_terms.BadTrackingZOnly(_cfg(bad_motion_body_pos_threshold=0.25), env)
    mc_state = _fake_motion_command(2, has_ball=False, in_kicking_phase=torch.tensor([False, False]))
    mc_state.body_pos_relative_w = torch.zeros(2, 2, 3)
    mc_state.robot_body_pos_w = torch.zeros(2, 2, 3)
    mc_state.body_pos_relative_w[:, 1, -1] = 0.5  # exceeds 0.25 for both envs

    out = term.bad_motion_body_pos(mc_state)
    assert torch.equal(out, torch.tensor([True, True]))


def test_per_skill_construction_raises_without_env_skill_id():
    env = _fake_env(num_envs=2)  # no skill_id
    with pytest.raises(AttributeError, match="skill_id"):
        wbt_terms.BadTracking(_cfg(params_per_skill={"swing_threshold_multiplier": [2.0, 3.0]}), env)


def test_handles_params_per_skill_class_attribute_is_true():
    """The opt-out flag TerminationManager's own guard checks -- proves this class is correctly
    exempted from the generic per-call mechanism's stateful-term rejection."""
    assert wbt_terms.BadTracking.handles_params_per_skill is True


def test_swing_only_per_skill_mask_gates_independently():
    """2026-08-15, Tier 3: env 0's skill has bad_tracking_swing_only=True (only fires while
    in_kicking_phase), env 1's has it False (fires unconditionally) -- both envs share the SAME
    underlying bad_tracking=True and are NOT currently in_kicking_phase, only the per-skill flag
    differs, isolating __call__'s own masking branch. Sub-checks (bad_ref_pos/_ori/
    bad_motion_body_pos) are monkey-patched on the instance directly, same technique as this
    file's other tests isolate _swing_widened_threshold/bad_motion_body_pos -- avoids needing a
    full physics env just to exercise __call__'s own control flow."""
    from types import SimpleNamespace

    class _FakeMotion:
        has_object = False

    class _FakeMotionCfg:
        body_names_to_track = _BODY_NAMES

    mc = SimpleNamespace(
        motion_cfg=_FakeMotionCfg(),
        motion=_FakeMotion(),
        has_ball=True,
        in_kicking_phase=torch.tensor([False, False]),  # neither env currently in swing
    )
    env = SimpleNamespace(
        device="cpu",
        skill_id=torch.tensor([0, 1]),
        episode_length_buf=torch.tensor([1000, 1000]),
        command_manager=SimpleNamespace(get_state=lambda name: mc),
    )
    term = wbt_terms.BadTracking(_cfg(params_per_skill={"bad_tracking_swing_only": [True, False]}), env)
    term.bad_ref_pos = lambda motion_command: torch.tensor([True, True])
    term.bad_ref_ori = lambda motion_command: torch.tensor([False, False])
    term.bad_motion_body_pos = lambda motion_command: torch.tensor([False, False])

    out = term(env)
    # env 0 (swing_only=True, NOT in swing) -> gated off -> False.
    # env 1 (swing_only=False) -> ungated, underlying bad_tracking=True passes through -> True.
    assert torch.equal(out, torch.tensor([False, True])), f"got {out}"


def test_swing_only_no_per_skill_table_uses_plain_scalar():
    """Legacy path: no per-skill table -- the plain self.bad_tracking_swing_only scalar still
    gates ALL envs uniformly, byte-identical to before this mechanism existed."""
    from types import SimpleNamespace

    class _FakeMotion:
        has_object = False

    class _FakeMotionCfg:
        body_names_to_track = _BODY_NAMES

    mc = SimpleNamespace(
        motion_cfg=_FakeMotionCfg(),
        motion=_FakeMotion(),
        has_ball=True,
        in_kicking_phase=torch.tensor([False, False]),
    )
    env = SimpleNamespace(
        device="cpu",
        episode_length_buf=torch.tensor([1000, 1000]),
        command_manager=SimpleNamespace(get_state=lambda name: mc),
    )
    cfg = _cfg()
    cfg = TerminationTermCfg(func=cfg.func, params={**cfg.params, "bad_tracking_swing_only": True})
    term = wbt_terms.BadTracking(cfg, env)
    term.bad_ref_pos = lambda motion_command: torch.tensor([True, True])
    term.bad_ref_ori = lambda motion_command: torch.tensor([False, False])
    term.bad_motion_body_pos = lambda motion_command: torch.tensor([False, False])

    out = term(env)
    assert torch.equal(out, torch.tensor([False, False]))  # gated off for both -- neither is in swing


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
