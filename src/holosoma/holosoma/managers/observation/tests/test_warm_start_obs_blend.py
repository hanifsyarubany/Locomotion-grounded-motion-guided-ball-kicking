"""Tests for FIX 6 (2026-08-18): ObservationManager's load-time warm-start blend
(``set_warm_start_blend`` / ``_warm_start_alpha`` / the blend step inside ``compute_group``).

WHY THIS EXISTS: the normalizer-reset guard (fast_sac_agent.py) fires correctly but does NOT
prevent a real shock when a changed observation term is a POPULATION change (e.g. a term that was
hard-zero for most envs becomes genuinely live) rather than a pure units change -- measured on a
real run, action_std still spiked 4.7x despite the guard resetting exactly the right terms. A
per-feature normalizer cannot correct the JOINT/conditional structure the actor's weights were fit
to; only a change to the actual INPUT the network sees can. This blend is that fix: at the instant
of resume (alpha=0) a changed term's contribution to the observation is the checkpoint's own
OLD-style value -- no shift at all -- fading continuously (not as a step function) to the NEW value
over the configured ramp.

The three properties that matter most, mirrored from the sibling ramp tests
(test_pre_kick_obs_ramp.py) for consistency:
  1. OFF (never configured, or ramp_steps<=0) is an EXACT no-op.
  2. At alpha=0 (the instant of resume) the blended value equals the OLD-style computation
     exactly -- this is the literal requirement ("I don't want the obs to be different from the
     previous checkpoint").
  3. The blend is linear and completes at alpha=1, converging exactly onto today's plain pipeline
     output -- so nothing is lost once the ramp finishes, and unrelated terms are never touched.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from holosoma.config_types.observation import ObsGroupCfg, ObsTermCfg, ObservationManagerCfg
from holosoma.managers.observation.manager import ObservationManager

_LOCOMOTION, _KICK = 0, 1


class _FakeUnifiedEnv:
    """Provides exactly what ObservationManager/its blend logic reads: task_mode_mask (hard) and
    common_step_counter (the blend's own ramp clock, per its own docstring)."""

    def __init__(self, num_envs: int, task_mode=None, common_step_counter: int = 0):
        self.num_envs = num_envs
        self.device = "cpu"
        self.task_mode = torch.tensor(task_mode if task_mode is not None else [_KICK] * num_envs)
        self.common_step_counter = common_step_counter
        self.logger = None

    def task_mode_mask(self, name: str) -> torch.Tensor:
        return self.task_mode == (_KICK if name == "kick" else _LOCOMOTION)


def _raw_metres(env, distance_scale: float = 0.0) -> torch.Tensor:
    """Stand-in for target_pos_b: distance_scale=0.0 (old) -> raw value; >0.0 (new) -> compressed.
    Deliberately simple (no tanh) so tests can assert exact expected numbers."""
    raw = torch.full((env.num_envs, 2), 6.0)
    if distance_scale <= 0.0:
        return raw
    return raw / distance_scale  # a stand-in "compression": bigger scale -> smaller value


def _always_one(env) -> torch.Tensor:
    return torch.ones(env.num_envs, 3)


def _mgr(groups: dict[str, ObsGroupCfg], env) -> ObservationManager:
    return ObservationManager(ObservationManagerCfg(groups=groups), env, device="cpu")


# ----------------------------------------------------------------------------------------------
# Property 1 -- never configured, or ramp_steps<=0, is an exact no-op.
# ----------------------------------------------------------------------------------------------


def test_alpha_is_none_when_never_configured():
    env = _FakeUnifiedEnv(2)
    groups = {"actor_obs": ObsGroupCfg(terms={"x": ObsTermCfg(func=f"{__name__}:_always_one")}, concatenate=False)}
    mgr = _mgr(groups, env)
    assert mgr._warm_start_alpha() is None


def test_alpha_is_none_when_ramp_steps_is_zero():
    env = _FakeUnifiedEnv(2)
    groups = {"actor_obs": ObsGroupCfg(terms={"x": ObsTermCfg(func=f"{__name__}:_always_one")}, concatenate=False)}
    mgr = _mgr(groups, env)
    mgr.set_warm_start_blend({"actor_obs": {"x": ObsTermCfg(func=f"{__name__}:_always_one")}}, ramp_steps=0.0)
    assert mgr._warm_start_alpha() is None


def test_compute_group_unaffected_when_blend_never_configured():
    env = _FakeUnifiedEnv(2)
    groups = {"actor_obs": ObsGroupCfg(terms={"x": ObsTermCfg(func=f"{__name__}:_always_one")}, concatenate=False)}
    mgr = _mgr(groups, env)
    out = mgr.compute_group("actor_obs")
    assert torch.equal(out["x"], torch.ones(2, 3))


# ----------------------------------------------------------------------------------------------
# Property 2 -- at alpha=0, the blended value equals the OLD-style computation exactly.
# ----------------------------------------------------------------------------------------------


def test_at_resume_instant_output_equals_old_style_value_exactly():
    """The literal requirement: at step 0 of the blend window, the term's contribution to the
    observation must be what the checkpoint's own config would have produced -- here, raw metres
    (distance_scale=0.0), NOT the new compressed value."""
    env = _FakeUnifiedEnv(2, common_step_counter=1000)
    new_term = ObsTermCfg(func=f"{__name__}:_raw_metres", params={"distance_scale": 3.0})
    groups = {"actor_obs": ObsGroupCfg(terms={"target": new_term}, concatenate=False)}
    mgr = _mgr(groups, env)

    old_term = ObsTermCfg(func=f"{__name__}:_raw_metres", params={"distance_scale": 0.0})
    mgr.set_warm_start_blend({"actor_obs": {"target": old_term}}, ramp_steps=100.0)
    # No steps have elapsed since set_warm_start_blend was called -- alpha must be exactly 0.
    out = mgr.compute_group("actor_obs")
    assert torch.allclose(out["target"], torch.full((2, 2), 6.0))  # the OLD (raw) value


def test_new_style_value_would_have_been_different_confirming_the_test_is_meaningful():
    """Sanity check that the scenario above is actually testing something: without any blend, the
    SAME new_term config produces a DIFFERENT value than what test 2 asserted."""
    env = _FakeUnifiedEnv(2)
    new_term = ObsTermCfg(func=f"{__name__}:_raw_metres", params={"distance_scale": 3.0})
    groups = {"actor_obs": ObsGroupCfg(terms={"target": new_term}, concatenate=False)}
    mgr = _mgr(groups, env)
    out = mgr.compute_group("actor_obs")
    assert torch.allclose(out["target"], torch.full((2, 2), 2.0))  # 6.0 / 3.0, NOT 6.0


# ----------------------------------------------------------------------------------------------
# Property 3 -- linear, completes at alpha=1, unrelated terms untouched.
# ----------------------------------------------------------------------------------------------


def test_blend_is_linear_partway_through_the_ramp():
    env = _FakeUnifiedEnv(2, common_step_counter=0)
    new_term = ObsTermCfg(func=f"{__name__}:_raw_metres", params={"distance_scale": 3.0})  # -> 2.0
    old_term = ObsTermCfg(func=f"{__name__}:_raw_metres", params={"distance_scale": 0.0})  # -> 6.0
    groups = {"actor_obs": ObsGroupCfg(terms={"target": new_term}, concatenate=False)}
    mgr = _mgr(groups, env)
    mgr.set_warm_start_blend({"actor_obs": {"target": old_term}}, ramp_steps=100.0)

    env.common_step_counter = 25  # alpha == 0.25
    out = mgr.compute_group("actor_obs")
    expected = 0.75 * 6.0 + 0.25 * 2.0  # (1-alpha)*old + alpha*new
    assert torch.allclose(out["target"], torch.full((2, 2), expected), atol=1e-5)


def test_blend_converges_exactly_onto_the_new_value_once_ramp_completes():
    env = _FakeUnifiedEnv(2, common_step_counter=0)
    new_term = ObsTermCfg(func=f"{__name__}:_raw_metres", params={"distance_scale": 3.0})
    old_term = ObsTermCfg(func=f"{__name__}:_raw_metres", params={"distance_scale": 0.0})
    groups = {"actor_obs": ObsGroupCfg(terms={"target": new_term}, concatenate=False)}
    mgr = _mgr(groups, env)
    mgr.set_warm_start_blend({"actor_obs": {"target": old_term}}, ramp_steps=50.0)

    env.common_step_counter = 500  # far past the ramp
    out = mgr.compute_group("actor_obs")
    assert torch.allclose(out["target"], torch.full((2, 2), 2.0))  # exactly the new value


def test_unrelated_term_is_never_touched_by_the_blend():
    env = _FakeUnifiedEnv(2, common_step_counter=0)
    target_new = ObsTermCfg(func=f"{__name__}:_raw_metres", params={"distance_scale": 3.0})
    target_old = ObsTermCfg(func=f"{__name__}:_raw_metres", params={"distance_scale": 0.0})
    other = ObsTermCfg(func=f"{__name__}:_always_one")
    groups = {"actor_obs": ObsGroupCfg(terms={"target": target_new, "other": other}, concatenate=False)}
    mgr = _mgr(groups, env)
    mgr.set_warm_start_blend({"actor_obs": {"target": target_old}}, ramp_steps=100.0)  # "other" not listed

    out = mgr.compute_group("actor_obs")
    assert torch.equal(out["other"], torch.ones(2, 3))  # untouched, even mid-ramp


# ----------------------------------------------------------------------------------------------
# task_mode blending -- the case the normalizer-reset guard could NOT fix (population change).
# ----------------------------------------------------------------------------------------------


def test_old_style_hard_gating_is_reproduced_at_resume_for_a_population_change():
    """The exact category FIX 6 was built for: a term that WAS gated to task_mode='kick' (hard
    zero for locomotion envs) becomes ungated (always live). At alpha=0, locomotion-mode envs must
    read the OLD (zero, gated) value -- not the new live one -- even though the NEW term config has
    no gating at all."""
    env = _FakeUnifiedEnv(4, task_mode=[_KICK, _LOCOMOTION, _KICK, _LOCOMOTION], common_step_counter=0)
    new_term = ObsTermCfg(func=f"{__name__}:_always_one", task_mode=None)  # always live now
    old_term = ObsTermCfg(func=f"{__name__}:_always_one", task_mode="kick")  # WAS gated
    groups = {"actor_obs": ObsGroupCfg(terms={"ball": new_term}, concatenate=False)}
    mgr = _mgr(groups, env)
    mgr.set_warm_start_blend({"actor_obs": {"ball": old_term}}, ramp_steps=100.0)

    out = mgr.compute_group("actor_obs")
    # locomotion-mode envs (index 1, 3): old value was hard-gated to 0 -> at alpha=0, still 0.
    assert torch.equal(out["ball"][1], torch.zeros(3))
    assert torch.equal(out["ball"][3], torch.zeros(3))
    # kick-mode envs (index 0, 2): old value was live (task_mode="kick" matches) -> still 1.
    assert torch.equal(out["ball"][0], torch.ones(3))
    assert torch.equal(out["ball"][2], torch.ones(3))


def test_locomotion_envs_gain_the_new_live_value_gradually_not_as_a_step():
    """Same scenario, mid-ramp: a locomotion-mode env's reading must be a PARTIAL (blended) value,
    not the full new value -- proving the population-change case ramps continuously too, not just
    the pure-units-change case."""
    env = _FakeUnifiedEnv(2, task_mode=[_LOCOMOTION, _LOCOMOTION], common_step_counter=0)
    new_term = ObsTermCfg(func=f"{__name__}:_always_one", task_mode=None)
    old_term = ObsTermCfg(func=f"{__name__}:_always_one", task_mode="kick")
    groups = {"actor_obs": ObsGroupCfg(terms={"ball": new_term}, concatenate=False)}
    mgr = _mgr(groups, env)
    mgr.set_warm_start_blend({"actor_obs": {"ball": old_term}}, ramp_steps=100.0)

    env.common_step_counter = 40  # alpha == 0.4
    out = mgr.compute_group("actor_obs")
    # old (gated, locomotion) = 0; new (ungated) = 1 -> blended = 0.4
    assert torch.allclose(out["ball"], torch.full((2, 3), 0.4), atol=1e-5)


# ----------------------------------------------------------------------------------------------
# set_warm_start_blend's own bookkeeping.
# ----------------------------------------------------------------------------------------------


def test_offset_is_captured_at_call_time_not_at_construction():
    env = _FakeUnifiedEnv(2, common_step_counter=777)
    groups = {"actor_obs": ObsGroupCfg(terms={"x": ObsTermCfg(func=f"{__name__}:_always_one")})}
    mgr = _mgr(groups, env)
    mgr.set_warm_start_blend({"actor_obs": {"x": ObsTermCfg(func=f"{__name__}:_always_one")}}, ramp_steps=10.0)
    assert mgr._warm_start_step_offset == 777


def test_empty_group_dict_in_blend_config_is_treated_as_nothing_to_blend():
    """set_warm_start_blend filters out empty per-group dicts (`if terms`) -- a caller passing
    `{"actor_obs": {}}` (e.g. from a detector that found changes only in critic_obs) must not leave
    a dangling, always-false-but-present entry."""
    env = _FakeUnifiedEnv(2)
    groups = {"actor_obs": ObsGroupCfg(terms={"x": ObsTermCfg(func=f"{__name__}:_always_one")})}
    mgr = _mgr(groups, env)
    mgr.set_warm_start_blend({"actor_obs": {}}, ramp_steps=10.0)
    assert mgr._warm_start_blend == {}
    assert mgr._warm_start_alpha() is None  # nothing to blend -> exact no-op, same as never configured
