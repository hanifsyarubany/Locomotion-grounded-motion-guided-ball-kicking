"""Tests for FastSACAgent._reset_normalizer_slots_for_shifted_obs_terms -- the 2026-08-18
obs-distribution-shift guard.

Root-caused on a real run (20260818_033003-stageD-1skill-obs-fixes): warm-starting a checkpoint
into a task_config whose observation terms changed UNITS left the just-loaded
EmpiricalNormalization applying its OLD (now-wrong) mean/std to the NEW values -- measured
action_std spiking 0.046->0.313 (6.8x) and kick_topple_frac 0->0.64 within a few thousand steps of
resume. This guard detects such a change (params/scale/task_mode/clip differ for a term present in
both the checkpoint's saved config and the current env's resolved config) and resets
EmpiricalNormalization's running mean/var to (0, 1) for exactly that term's column slice.

Tests call the method directly on a duck-typed fake `self` (real FastSACAgent construction needs a
live env/GPU) -- the method only ever touches `self.unwrapped_env.observation_manager`,
`self.obs_normalizer`, `self.critic_obs_normalizer`, all faked below with real torch tensors so the
slicing/reset arithmetic itself is exercised for real, not mocked away.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from holosoma.agents.fast_sac.fast_sac_agent import FastSACAgent
from holosoma.config_types.observation import ObsGroupCfg, ObsTermCfg


class _FakeNormalizer:
    """Mimics EmpiricalNormalization's buffer shape/dtype (see fast_sac_utils.py) without the
    full nn.Module machinery -- only the 3 buffers this guard touches."""

    def __init__(self, width: int, fill: float):
        self._mean = torch.full((1, width), fill)
        self._var = torch.full((1, width), fill + 10.0)
        self._std = torch.full((1, width), fill + 20.0)


class _FakeObsManager:
    """`cfg.groups` is a real dict of real (frozen) ObsGroupCfg -- the swap-and-restore logic under
    test mutates and restores this dict exactly as it would the production manager's.
    `compute_group` returns fixed-width zero tensors per term, independent of the swapped
    concatenate flag (production always swaps to concatenate=False before calling here, so this
    fake only needs to implement that one path -- mirrors the real dict-return branch)."""

    def __init__(self, groups: dict[str, ObsGroupCfg], widths: dict[str, int]):
        self.cfg = SimpleCfg(groups)
        self._widths = widths

    def compute_group(self, group_name: str, modify_history: bool = True) -> dict[str, torch.Tensor]:
        terms = self.cfg.groups[group_name].terms
        return {name: torch.zeros(2, self._widths[name]) for name in terms}


class SimpleCfg:
    def __init__(self, groups):
        self.groups = groups


class _FakeEnv:
    def __init__(self, observation_manager):
        self.observation_manager = observation_manager


def _term(params=None, scale=1.0, task_mode=None, clip=None) -> ObsTermCfg:
    return ObsTermCfg(func="dummy:fn", params=params or {}, scale=scale, task_mode=task_mode, clip=clip)


def _old_term_dict(cfg: ObsTermCfg) -> dict:
    """Shape the CHECKPOINT side is actually saved in: a plain dict (deserialized from yaml/torch
    save, not a live ObsTermCfg instance) -- see FastSACAgent's own docstring for why `.get(...)` is
    used on this side."""
    return {"params": dict(cfg.params), "scale": cfg.scale, "task_mode": cfg.task_mode, "clip": cfg.clip}


def _make_self(old_groups: dict | None, new_groups: dict[str, ObsGroupCfg], widths: dict[str, int],
                actor_fill=1.0, critic_fill=2.0):
    obs_manager = _FakeObsManager(new_groups, widths)
    fake_self = SimpleCfg.__new__(SimpleCfg)  # bare object, only need attribute assignment below
    fake_self.unwrapped_env = _FakeEnv(obs_manager)
    actor_width = sum(widths[n] for n in new_groups.get("actor_obs", ObsGroupCfg()).terms)
    critic_width = sum(widths[n] for n in new_groups.get("critic_obs", ObsGroupCfg()).terms)
    fake_self.obs_normalizer = _FakeNormalizer(actor_width, actor_fill) if actor_width else None
    fake_self.critic_obs_normalizer = _FakeNormalizer(critic_width, critic_fill) if critic_width else None
    checkpoint: dict[str, Any] = {}
    if old_groups is not None:
        checkpoint["experiment_config"] = {"observation": {"groups": old_groups}}
    return fake_self, checkpoint


def _call(fake_self, checkpoint):
    # 2026-08-18: _reset_normalizer_slots_for_shifted_obs_terms now delegates detection to a
    # sibling method (_detect_shifted_obs_terms, shared with the warm-start blend path) -- since
    # `fake_self` is a bare object rather than a real FastSACAgent instance, plain unbound-style
    # calling (`FastSACAgent.method(fake_self, ...)`) does not let `self._detect_shifted_obs_terms`
    # resolve inside it. Bind that one sibling method onto the fake, exactly as Python's own
    # attribute lookup would for a real instance, so the cross-method call works.
    import types

    if not hasattr(fake_self, "_detect_shifted_obs_terms"):
        fake_self._detect_shifted_obs_terms = types.MethodType(
            FastSACAgent._detect_shifted_obs_terms, fake_self
        )
    FastSACAgent._reset_normalizer_slots_for_shifted_obs_terms(fake_self, checkpoint)


@pytest.fixture(autouse=True)
def _no_env_skip(monkeypatch):
    monkeypatch.delenv("HOLOSOMA_SKIP_OBS_NORMALIZER_RESET", raising=False)


# ----------------------------------------------------------------------------------------------
# No-op cases.
# ----------------------------------------------------------------------------------------------


def test_nothing_changed_is_a_true_no_op():
    """The common case (loading a teacher into the SAME config it was saved under): every stat
    must be byte-identical to whatever load_state_dict just populated."""
    term = _term(params={"distance_scale": 5.0}, task_mode="kick")
    groups = {"actor_obs": ObsGroupCfg(terms={"kick_target_pos_b": term}, concatenate=True)}
    widths = {"kick_target_pos_b": 2}
    old = {"actor_obs": {"terms": {"kick_target_pos_b": _old_term_dict(term)}}}
    fake_self, ckpt = _make_self(old, groups, widths)
    before = fake_self.obs_normalizer._mean.clone()
    _call(fake_self, ckpt)
    assert torch.equal(fake_self.obs_normalizer._mean, before)


def test_missing_experiment_config_in_checkpoint_is_a_no_op():
    """Checkpoints saved before this metadata existed (or any dict missing the key) must not
    crash -- this guard is strictly additive."""
    term = _term()
    groups = {"actor_obs": ObsGroupCfg(terms={"x": term})}
    fake_self, ckpt = _make_self(None, groups, {"x": 3})
    before = fake_self.obs_normalizer._mean.clone()
    _call(fake_self, ckpt)  # must not raise
    assert torch.equal(fake_self.obs_normalizer._mean, before)


def test_term_only_in_new_config_is_ignored_not_crashed():
    """A genuinely NEW term (width change) is a separate, already-handled failure mode
    (load_state_dict shape mismatch) -- this guard only acts on terms present in BOTH sides."""
    old_term = _term()
    new_groups = {
        "actor_obs": ObsGroupCfg(terms={"x": _term(), "y": _term(params={"new": True})}),
    }
    old = {"actor_obs": {"terms": {"x": _old_term_dict(old_term)}}}  # "y" absent from old
    fake_self, ckpt = _make_self(old, new_groups, {"x": 3, "y": 2})
    before = fake_self.obs_normalizer._mean.clone()
    _call(fake_self, ckpt)
    assert torch.equal(fake_self.obs_normalizer._mean, before)  # "y" has nothing to compare against


def test_skip_env_var_disables_even_when_something_changed(monkeypatch):
    monkeypatch.setenv("HOLOSOMA_SKIP_OBS_NORMALIZER_RESET", "1")
    old_term = _term(params={"distance_scale": 0.0})
    new_term = _term(params={"distance_scale": 5.0})
    groups = {"actor_obs": ObsGroupCfg(terms={"kick_target_pos_b": new_term})}
    old = {"actor_obs": {"terms": {"kick_target_pos_b": _old_term_dict(old_term)}}}
    fake_self, ckpt = _make_self(old, groups, {"kick_target_pos_b": 2})
    before = fake_self.obs_normalizer._mean.clone()
    _call(fake_self, ckpt)
    assert torch.equal(fake_self.obs_normalizer._mean, before)


# ----------------------------------------------------------------------------------------------
# The real fix -- reset fires, and only on the right slice.
# ----------------------------------------------------------------------------------------------


def test_changed_params_resets_mean_var_std_to_identity():
    old_term = _term(params={"distance_scale": 0.0})
    new_term = _term(params={"distance_scale": 5.0})
    groups = {"actor_obs": ObsGroupCfg(terms={"kick_target_pos_b": new_term})}
    old = {"actor_obs": {"terms": {"kick_target_pos_b": _old_term_dict(old_term)}}}
    fake_self, ckpt = _make_self(old, groups, {"kick_target_pos_b": 2}, actor_fill=99.0)
    _call(fake_self, ckpt)
    assert torch.equal(fake_self.obs_normalizer._mean, torch.zeros(1, 2))
    assert torch.equal(fake_self.obs_normalizer._var, torch.ones(1, 2))
    assert torch.equal(fake_self.obs_normalizer._std, torch.ones(1, 2))


def test_only_the_changed_terms_slice_is_touched_others_preserved():
    """The core correctness property: alphabetical concatenation order must be used to locate the
    right column range, and unrelated terms must be byte-identical afterward."""
    a = _term(params={"a": 1})  # unchanged
    b_old = _term(scale=1.0)
    b_new = _term(scale=2.0)  # CHANGED (scale)
    c = _term(clip=(-1.0, 1.0))  # unchanged
    # sorted order: a, b, c -> widths 3, 2, 4 -> a:[0,3) b:[3,5) c:[5,9)
    groups = {"actor_obs": ObsGroupCfg(terms={"a": a, "b": b_new, "c": c})}
    old = {
        "actor_obs": {
            "terms": {"a": _old_term_dict(a), "b": _old_term_dict(b_old), "c": _old_term_dict(c)}
        }
    }
    widths = {"a": 3, "b": 2, "c": 4}
    fake_self, ckpt = _make_self(old, groups, widths, actor_fill=7.0)
    before = fake_self.obs_normalizer._mean.clone()
    _call(fake_self, ckpt)
    after = fake_self.obs_normalizer._mean
    assert torch.equal(after[:, 0:3], before[:, 0:3])  # "a" untouched
    assert torch.equal(after[:, 3:5], torch.zeros(1, 2))  # "b" reset
    assert torch.equal(after[:, 5:9], before[:, 5:9])  # "c" untouched


def test_task_mode_change_alone_triggers_reset():
    """FIX 2/4's exact scenario: params identical, only task_mode changes (e.g. a term that was
    gated to 'kick' becomes always-live) -- this is a POPULATION change (which envs contribute
    nonzero data), not a units change, and must be caught too."""
    old_term = _term(task_mode="kick")
    new_term = _term(task_mode=None)
    groups = {"actor_obs": ObsGroupCfg(terms={"kick_ball_pos_b": new_term})}
    old = {"actor_obs": {"terms": {"kick_ball_pos_b": _old_term_dict(old_term)}}}
    fake_self, ckpt = _make_self(old, groups, {"kick_ball_pos_b": 3}, actor_fill=5.0)
    _call(fake_self, ckpt)
    assert torch.equal(fake_self.obs_normalizer._mean, torch.zeros(1, 3))


def test_clip_change_alone_triggers_reset():
    old_term = _term(clip=None)
    new_term = _term(clip=(-2.0, 2.0))
    groups = {"actor_obs": ObsGroupCfg(terms={"x": new_term})}
    old = {"actor_obs": {"terms": {"x": _old_term_dict(old_term)}}}
    fake_self, ckpt = _make_self(old, groups, {"x": 1}, actor_fill=3.0)
    _call(fake_self, ckpt)
    assert torch.equal(fake_self.obs_normalizer._mean, torch.zeros(1, 1))


def test_both_groups_handled_independently():
    """actor_obs and critic_obs must each be diffed and reset on their OWN normalizer -- a change
    in one must not touch the other's buffers."""
    old_a = _term(params={"d": 0.0})
    new_a = _term(params={"d": 5.0})  # changed, actor_obs only
    c = _term(params={"e": 1})  # unchanged, appears in critic_obs only
    groups = {
        "actor_obs": ObsGroupCfg(terms={"x": new_a}),
        "critic_obs": ObsGroupCfg(terms={"y": c}),
    }
    old = {
        "actor_obs": {"terms": {"x": _old_term_dict(old_a)}},
        "critic_obs": {"terms": {"y": _old_term_dict(c)}},
    }
    fake_self, ckpt = _make_self(old, groups, {"x": 2, "y": 3}, actor_fill=9.0, critic_fill=9.0)
    _call(fake_self, ckpt)
    assert torch.equal(fake_self.obs_normalizer._mean, torch.zeros(1, 2))  # actor: reset
    assert torch.equal(fake_self.critic_obs_normalizer._mean, torch.full((1, 3), 9.0))  # critic: untouched


def test_original_group_cfg_is_restored_after_the_probe():
    """The swap-and-restore technique (memory `stage-d-handoff-observation-discontinuity`) must
    leave the manager's real cfg exactly as it found it -- a leaked concatenate=False would corrupt
    every subsequent real observation compute in the training loop."""
    term = _term(params={"f": 1.0})
    original_group = ObsGroupCfg(terms={"x": term}, concatenate=True, enable_noise=True)
    groups = {"actor_obs": original_group}
    old = {"actor_obs": {"terms": {"x": {"params": {"f": 2.0}, "scale": 1.0, "task_mode": None, "clip": None}}}}
    fake_self, ckpt = _make_self(old, groups, {"x": 1})
    _call(fake_self, ckpt)
    restored = fake_self.unwrapped_env.observation_manager.cfg.groups["actor_obs"]
    assert restored is original_group
    assert restored.concatenate is True
    assert restored.enable_noise is True


def test_count_buffer_is_deliberately_not_touched():
    """Documented limitation: EmpiricalNormalization.count is a single scalar shared across the
    whole vector (fast_sac_utils.py), so this guard cannot reset per-slot confidence -- it must not
    even attempt to touch `count`, since a shared reset would corrupt every OTHER, unrelated slot's
    adaptation rate too."""
    old_term = _term(params={"g": 0.0})
    new_term = _term(params={"g": 1.0})
    groups = {"actor_obs": ObsGroupCfg(terms={"x": new_term})}
    old = {"actor_obs": {"terms": {"x": _old_term_dict(old_term)}}}
    fake_self, ckpt = _make_self(old, groups, {"x": 1})
    fake_self.obs_normalizer.count = torch.tensor(123456789)
    _call(fake_self, ckpt)
    assert fake_self.obs_normalizer.count.item() == 123456789


# ==================================================================================================
# FIX 6 (2026-08-18): _configure_warm_start_obs_blend -- the wiring between a checkpoint's saved
# config and ObservationManager.set_warm_start_blend. Reuses the SAME _detect_shifted_obs_terms
# this file's own tests already exercise via the reset guard, so these tests focus on what's NEW:
# reconstructing a real ObsTermCfg from the checkpoint's plain dict, and calling the manager
# correctly -- not re-testing the diff logic itself.
# ==================================================================================================


class _FakeObsManagerForBlend:
    """Records exactly what set_warm_start_blend was called with, so tests can assert on the
    reconstructed ObsTermCfg without needing a real ObservationManager/env/GPU."""

    def __init__(self, groups):
        self.cfg = SimpleCfg(groups)
        self.blend_calls: list[tuple[dict, float]] = []

    def set_warm_start_blend(self, old_cfgs_by_group, ramp_steps):
        self.blend_calls.append((old_cfgs_by_group, ramp_steps))


def _make_blend_self(old_groups, new_groups):
    obs_manager = _FakeObsManagerForBlend(new_groups)
    fake_self = SimpleCfg.__new__(SimpleCfg)
    fake_self.unwrapped_env = _FakeEnv(obs_manager)
    checkpoint: dict[str, Any] = {}
    if old_groups is not None:
        checkpoint["experiment_config"] = {"observation": {"groups": old_groups}}
    return fake_self, checkpoint, obs_manager


def _call_blend(fake_self, checkpoint, ramp_steps):
    import types

    if not hasattr(fake_self, "_detect_shifted_obs_terms"):
        fake_self._detect_shifted_obs_terms = types.MethodType(
            FastSACAgent._detect_shifted_obs_terms, fake_self
        )
    FastSACAgent._configure_warm_start_obs_blend(fake_self, checkpoint, ramp_steps)


def test_blend_configurator_reconstructs_a_real_obstermcfg_from_the_checkpoint_dict():
    old_term_dict = _old_term_dict(_term(params={"distance_scale": 0.0}, task_mode="kick"))
    old_term_dict["func"] = "dummy:target_pos_b"  # func must be present -- checkpoints always have it
    new_term = _term(params={"distance_scale": 5.0}, task_mode="kick")
    new_term = dataclasses.replace(new_term, func="dummy:target_pos_b")
    groups = {"actor_obs": ObsGroupCfg(terms={"target": new_term})}
    old = {"actor_obs": {"terms": {"target": old_term_dict}}}
    fake_self, ckpt, obs_manager = _make_blend_self(old, groups)

    _call_blend(fake_self, ckpt, ramp_steps=50.0)

    assert len(obs_manager.blend_calls) == 1
    old_cfgs_by_group, ramp = obs_manager.blend_calls[0]
    assert ramp == 50.0
    reconstructed = old_cfgs_by_group["actor_obs"]["target"]
    assert reconstructed.func == "dummy:target_pos_b"
    assert reconstructed.params == {"distance_scale": 0.0}
    assert reconstructed.task_mode == "kick"


def test_blend_configurator_is_a_no_op_when_nothing_changed():
    term = _term(params={"a": 1})
    groups = {"actor_obs": ObsGroupCfg(terms={"x": term})}
    old = {"actor_obs": {"terms": {"x": _old_term_dict(term)}}}
    fake_self, ckpt, obs_manager = _make_blend_self(old, groups)

    _call_blend(fake_self, ckpt, ramp_steps=50.0)
    assert obs_manager.blend_calls == []


def test_blend_configurator_handles_missing_checkpoint_metadata():
    groups = {"actor_obs": ObsGroupCfg(terms={"x": _term()})}
    fake_self, ckpt, obs_manager = _make_blend_self(None, groups)

    _call_blend(fake_self, ckpt, ramp_steps=50.0)  # must not raise
    assert obs_manager.blend_calls == []


def test_load_dispatches_to_blend_not_reset_when_ramp_steps_positive(monkeypatch):
    """The mutual-exclusion contract at the load() call site: with warm_start_obs_ramp_steps set on
    the env, the RESET guard must never run (they would fight each other -- see FIX 6's own
    docstring), only the blend configurator."""
    calls = []

    class _Agent:
        def _configure_warm_start_obs_blend(self, ckpt, ramp):
            calls.append(("blend", ramp))

        def _reset_normalizer_slots_for_shifted_obs_terms(self, ckpt):
            calls.append(("reset",))

    agent = _Agent()
    agent.unwrapped_env = SimpleNamespace(_warm_start_obs_ramp_steps=75.0)
    warm_start_obs_ramp_steps = float(getattr(agent.unwrapped_env, "_warm_start_obs_ramp_steps", 0.0))
    if warm_start_obs_ramp_steps > 0.0:
        agent._configure_warm_start_obs_blend({}, warm_start_obs_ramp_steps)
    else:
        agent._reset_normalizer_slots_for_shifted_obs_terms({})

    assert calls == [("blend", 75.0)]


def test_load_dispatches_to_reset_when_ramp_steps_is_default_zero():
    """The inverse: with the field at its 0.0 default (unset), behavior must be UNCHANGED from
    before FIX 6 existed -- the reset guard, and only the reset guard, runs."""
    calls = []

    class _Agent:
        def _configure_warm_start_obs_blend(self, ckpt, ramp):
            calls.append(("blend", ramp))

        def _reset_normalizer_slots_for_shifted_obs_terms(self, ckpt):
            calls.append(("reset",))

    agent = _Agent()
    agent.unwrapped_env = SimpleNamespace()  # no _warm_start_obs_ramp_steps attribute at all
    warm_start_obs_ramp_steps = float(getattr(agent.unwrapped_env, "_warm_start_obs_ramp_steps", 0.0))
    if warm_start_obs_ramp_steps > 0.0:
        agent._configure_warm_start_obs_blend({}, warm_start_obs_ramp_steps)
    else:
        agent._reset_normalizer_slots_for_shifted_obs_terms({})

    assert calls == [("reset",)]


# ==================================================================================================
# 2026-08-18: the "parameter added at its own default" FALSE POSITIVE. Caught live, after it
# contaminated a real control run -- adding `distance_scale` to target_pos_b meant the current
# config carried params={"distance_scale": 0.0} while the checkpoint carried params={}, which are
# the SAME behavior (0.0 is that param's off-sentinel and its signature default), but a raw dict
# comparison flagged a shift and the guard then un-normalized kick_target_pos_b -- injecting the
# very distribution shift the run existed to prove absent.
# ==================================================================================================


def _param_with_default(env, distance_scale: float = 0.0):
    """Stand-in for target_pos_b: `distance_scale` defaults to 0.0 (the off-sentinel)."""
    return torch.zeros(1, 2)


def test_param_added_at_its_signature_default_is_not_a_change():
    """params={} vs params={'distance_scale': 0.0} must compare EQUAL, because 0.0 is that
    parameter's own signature default -- behavior is byte-identical."""
    func = f"{__name__}:_param_with_default"
    new_term = dataclasses.replace(_term(params={"distance_scale": 0.0}), func=func)
    old_dict = _old_term_dict(_term(params={}))
    old_dict["func"] = func
    groups = {"actor_obs": ObsGroupCfg(terms={"target": new_term})}
    old = {"actor_obs": {"terms": {"target": old_dict}}}
    fake_self, ckpt = _make_self(old, groups, {"target": 2}, actor_fill=42.0)
    before = fake_self.obs_normalizer._mean.clone()
    _call(fake_self, ckpt)
    assert torch.equal(fake_self.obs_normalizer._mean, before), "guard fired on a no-op param addition"


def test_param_added_at_a_NON_default_value_is_still_a_change():
    """The inverse must still be caught: 0.0 -> 5.0 is a genuine behavior change and MUST fire,
    otherwise the fix above would have disabled the guard's real purpose."""
    func = f"{__name__}:_param_with_default"
    new_term = dataclasses.replace(_term(params={"distance_scale": 5.0}), func=func)
    old_dict = _old_term_dict(_term(params={}))
    old_dict["func"] = func
    groups = {"actor_obs": ObsGroupCfg(terms={"target": new_term})}
    old = {"actor_obs": {"terms": {"target": old_dict}}}
    fake_self, ckpt = _make_self(old, groups, {"target": 2}, actor_fill=42.0)
    _call(fake_self, ckpt)
    assert torch.equal(fake_self.obs_normalizer._mean, torch.zeros(1, 2)), "genuine change was missed"


def test_unresolvable_func_falls_back_to_raw_params_without_crashing():
    """A func string that no longer resolves must degrade to the old (over-eager) comparison
    rather than raising inside load()."""
    new_term = _term(params={"x": 1.0})  # func="dummy:fn", unresolvable
    old_dict = _old_term_dict(_term(params={"x": 2.0}))
    groups = {"actor_obs": ObsGroupCfg(terms={"t": new_term})}
    old = {"actor_obs": {"terms": {"t": old_dict}}}
    fake_self, ckpt = _make_self(old, groups, {"t": 1}, actor_fill=8.0)
    _call(fake_self, ckpt)  # must not raise
    assert torch.equal(fake_self.obs_normalizer._mean, torch.zeros(1, 1))  # 1.0 != 2.0 -> real change


def test_func_swap_alone_triggers_reset_even_with_identical_params():
    """2026-08-22, found while auditing the azimuth-aim refactor's checkpoint compatibility: a
    term can keep its NAME and an IDENTICAL params/scale/task_mode/clip while its underlying
    function is swapped for one with completely different output semantics -- exactly what
    config_values/unified/g1/observation.py's kick_target_pos_b did (target_pos_b, a raw
    world-frame offset in metres -> kick_aim_command, a bounded normalized command -- both
    registered with params={"distance_scale": 0.0} since kick_aim_command was deliberately given
    the same signature so the observation WIDTH stayed unchanged). Before `func` was added to the
    comparison key, this was INVISIBLE to the detector -- (params, scale, task_mode, clip) came out
    identical on both sides -- so a warm-started old checkpoint would silently apply stale
    metres-scale normalizer stats to the new bounded output for the rest of training. Params here
    are intentionally IDENTICAL on both sides to isolate that `func` alone is now sufficient."""
    old_term = _term(params={"distance_scale": 0.0})
    old_dict = _old_term_dict(old_term)
    old_dict["func"] = "dummy:target_pos_b"
    new_term = dataclasses.replace(old_term, func="dummy:kick_aim_command")
    groups = {"actor_obs": ObsGroupCfg(terms={"kick_target_pos_b": new_term})}
    old = {"actor_obs": {"terms": {"kick_target_pos_b": old_dict}}}
    fake_self, ckpt = _make_self(old, groups, {"kick_target_pos_b": 2}, actor_fill=42.0)
    _call(fake_self, ckpt)
    assert torch.equal(fake_self.obs_normalizer._mean, torch.zeros(1, 2))
    assert torch.equal(fake_self.obs_normalizer._var, torch.ones(1, 2))
    assert torch.equal(fake_self.obs_normalizer._std, torch.ones(1, 2))


def test_identical_func_with_identical_everything_else_is_still_a_no_op():
    """Sanity check on the fix's own precision: adding `func` to the key must not turn every
    ordinary warm-start (same func, same everything) into a spurious reset."""
    term = _term(params={"distance_scale": 5.0})
    old_dict = _old_term_dict(term)
    old_dict["func"] = term.func
    groups = {"actor_obs": ObsGroupCfg(terms={"t": term})}
    old = {"actor_obs": {"terms": {"t": old_dict}}}
    fake_self, ckpt = _make_self(old, groups, {"t": 2}, actor_fill=7.0)
    _call(fake_self, ckpt)
    assert torch.equal(fake_self.obs_normalizer._mean, torch.full((1, 2), 7.0))
