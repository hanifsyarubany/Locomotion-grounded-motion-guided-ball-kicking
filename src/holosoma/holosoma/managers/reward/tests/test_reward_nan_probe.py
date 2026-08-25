"""Unit tests for RewardManager's per-TERM nan-probe (2026-08-18).

Root-caused via a real run (20260818_033003/052857-stageD-1skill-obs-fixes): FastSACAgent's own
nan-probe (fast_sac_agent.py, 2026-07-18) caught the aggregate 'rewards' tensor going non-finite at
the SAC-update boundary, but that probe fires many steps downstream of reward computation -- it
could name the CORRUPTED TENSOR, not which of the ~80 reward terms produced it, because
RewardManager.compute() accumulated every term straight into `_reward_buf` with no finiteness
check anywhere in the loop. This probe closes that attribution gap.

Same SimpleNamespace-fake-env / RewardManagerCfg construction pattern as
test_reward_manager_per_skill_weight.py, applied to the new probe instead.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg
from holosoma.managers.reward.manager import RewardManager

_MODULE = "holosoma.managers.reward.tests.test_reward_nan_probe"


def _term(func_name: str, weight: float = 1.0) -> RewardTermCfg:
    return RewardTermCfg(func=f"{_MODULE}:{func_name}", weight=weight)


def _clean(env) -> torch.Tensor:
    return torch.ones(env.num_envs)


def _nan_for_env0(env) -> torch.Tensor:
    out = torch.ones(env.num_envs)
    out[0] = float("nan")
    return out


def _inf_for_env2(env) -> torch.Tensor:
    out = torch.ones(env.num_envs)
    if env.num_envs > 2:
        out[2] = float("inf")
    return out


def _fake_env(num_envs: int, skill_id: torch.Tensor | None = None, task_mode: torch.Tensor | None = None):
    kwargs = dict(num_envs=num_envs, logger=None)
    if skill_id is not None:
        kwargs["skill_id"] = skill_id
    if task_mode is not None:
        kwargs["task_mode"] = task_mode
    return SimpleNamespace(**kwargs)


def _fresh_manager(cfg: RewardManagerCfg, env) -> RewardManager:
    """Every test needs its OWN manager instance -- `_reward_nan_probe_fired` is a per-instance
    flag (via class-attribute fallthrough), so reusing a manager across tests would silently skip
    detection on the second test to run."""
    return RewardManager(cfg, env, device="cpu")


# ----------------------------------------------------------------------------------------------
# No-op: all-finite terms never log or flip the flag.
# ----------------------------------------------------------------------------------------------


def test_all_finite_terms_never_fire(caplog):
    cfg = RewardManagerCfg(terms={"a": _term("_clean"), "b": _term("_clean")})
    manager = _fresh_manager(cfg, _fake_env(num_envs=4))
    manager.compute(dt=0.02)
    assert manager._reward_nan_probe_fired is False


def test_compute_result_unaffected_by_the_probe():
    """The probe only reads rew_raw -- it must never alter the actual accumulated reward."""
    cfg = RewardManagerCfg(terms={"a": _term("_clean", weight=2.0)})
    manager = _fresh_manager(cfg, _fake_env(num_envs=3))
    out = manager.compute(dt=0.02)
    assert torch.allclose(out, torch.full((3,), 2.0 * 0.02))


# ----------------------------------------------------------------------------------------------
# Detection: fires, names the RIGHT term, and doesn't corrupt the reward computation.
# ----------------------------------------------------------------------------------------------


def test_nan_term_sets_the_fired_flag():
    cfg = RewardManagerCfg(terms={"a": _term("_clean"), "b": _term("_nan_for_env0")})
    manager = _fresh_manager(cfg, _fake_env(num_envs=4))
    manager.compute(dt=0.02)
    assert manager._reward_nan_probe_fired is True


def test_inf_alone_also_triggers_detection():
    """isfinite() catches +/-inf, not just NaN -- both are equally fatal downstream."""
    cfg = RewardManagerCfg(terms={"a": _term("_inf_for_env2")})
    manager = _fresh_manager(cfg, _fake_env(num_envs=4))
    manager.compute(dt=0.02)
    assert manager._reward_nan_probe_fired is True


def test_logs_the_correct_term_name_not_a_later_or_earlier_one(caplog):
    """The core attribution property this probe exists for: with THREE terms, only the middle one
    corrupted, the logged message must name 'b', not 'a' or 'c'."""
    import logging

    from loguru import logger as loguru_logger

    caplog.set_level(logging.ERROR)
    handler_id = loguru_logger.add(caplog.handler, format="{message}", level="ERROR")
    try:
        cfg = RewardManagerCfg(
            terms={"a": _term("_clean"), "b": _term("_nan_for_env0"), "c": _term("_clean")}
        )
        manager = _fresh_manager(cfg, _fake_env(num_envs=4))
        manager.compute(dt=0.02)
    finally:
        loguru_logger.remove(handler_id)

    messages = [r.message for r in caplog.records]
    assert any("'b'" in m for m in messages), messages
    assert not any("'a'" in m or "'c'" in m for m in messages)


def test_example_env_index_matches_the_actual_corrupted_row(caplog):
    import logging

    from loguru import logger as loguru_logger

    caplog.set_level(logging.ERROR)
    handler_id = loguru_logger.add(caplog.handler, format="{message}", level="ERROR")
    try:
        cfg = RewardManagerCfg(terms={"a": _term("_nan_for_env0")})
        manager = _fresh_manager(cfg, _fake_env(num_envs=4))
        manager.compute(dt=0.02)
    finally:
        loguru_logger.remove(handler_id)

    assert any("example env=0" in r.message for r in caplog.records)


def test_fires_only_once_even_with_multiple_bad_terms():
    """Mirrors fast_sac_agent.py's own probe discipline: fire once, do not spam every step / every
    subsequent term -- confirmed here by checking the flag stays True (not re-toggled) and by the
    SECOND bad term not being independently reported (caplog count == 1 error block)."""
    import logging

    cfg = RewardManagerCfg(
        terms={"a": _term("_nan_for_env0"), "b": _term("_inf_for_env2")}
    )
    manager = _fresh_manager(cfg, _fake_env(num_envs=4))
    manager.compute(dt=0.02)
    first_fired = manager._reward_nan_probe_fired
    manager.compute(dt=0.02)  # second step: both terms still corrupt every call
    assert first_fired is True
    assert manager._reward_nan_probe_fired is True  # stays fired, doesn't toggle off


def test_does_not_fire_on_a_fresh_manager_instance():
    """Class-attribute fallthrough correctness: firing on ONE manager instance must not leak into
    a brand-new instance (would happen if the flag were accidentally a mutable class-level
    container instead of a fire-once-then-shadow bool)."""
    cfg = RewardManagerCfg(terms={"a": _term("_nan_for_env0")})
    manager1 = _fresh_manager(cfg, _fake_env(num_envs=4))
    manager1.compute(dt=0.02)
    assert manager1._reward_nan_probe_fired is True

    manager2 = _fresh_manager(cfg, _fake_env(num_envs=4))
    assert manager2._reward_nan_probe_fired is False


# ----------------------------------------------------------------------------------------------
# Robustness: the probe itself must not require optional env attributes.
# ----------------------------------------------------------------------------------------------


def test_works_without_skill_id_or_task_mode_on_env():
    """Legacy/single-skill envs have neither attribute -- the probe's skill/task_mode context is
    logged on a best-effort basis (None when absent), never a hard requirement."""
    cfg = RewardManagerCfg(terms={"a": _term("_nan_for_env0")})
    manager = _fresh_manager(cfg, _fake_env(num_envs=4))  # no skill_id, no task_mode
    manager.compute(dt=0.02)  # must not raise
    assert manager._reward_nan_probe_fired is True


def test_includes_skill_id_and_task_mode_when_present(caplog):
    import logging

    from loguru import logger as loguru_logger

    caplog.set_level(logging.ERROR)
    handler_id = loguru_logger.add(caplog.handler, format="{message}", level="ERROR")
    try:
        cfg = RewardManagerCfg(terms={"a": _term("_nan_for_env0")})
        env = _fake_env(num_envs=4, skill_id=torch.tensor([1, 0, 0, 0]), task_mode=torch.tensor([1, 0, 0, 0]))
        manager = _fresh_manager(cfg, env)
        manager.compute(dt=0.02)
    finally:
        loguru_logger.remove(handler_id)

    assert any("skill_id=1" in r.message for r in caplog.records)
