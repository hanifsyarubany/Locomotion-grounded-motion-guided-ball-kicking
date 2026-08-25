"""Configuration types for reward manager."""

from __future__ import annotations

from dataclasses import field
from typing import Any

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class RewardTermCfg:
    """Configuration for a single reward term."""

    func: str
    """Import path to the reward function or class"""
    """(e.g. ``holosoma.managers.reward.terms.locomotion:tracking_lin_vel``)."""

    params: dict[str, Any] = field(default_factory=dict)
    """Additional parameters forwarded to the reward term."""

    weight: float = 1.0
    """Weight applied to the reward term (manager multiplies by ``dt``). Ignored in favor of
    ``weight_per_skill`` when that's set (kept populated even then, as a representative value for
    any code -- logging, the zero-weight skip check -- that reads ``.weight`` directly without
    per-skill awareness)."""

    weight_per_skill: list[float] | None = None
    """2026-08-15: opt-in per-skill weight override, ``[n_skills]``, indexed by ``env.skill_id``.
    ``None`` (default) means every env uses the plain scalar ``weight`` above -- byte-identical to
    before this field existed. Set only by config_values/unified/g1/reward.py's
    ``_apply_per_skill_reward_weight_overrides``, when 2+ skills in one training run genuinely
    declare DIFFERENT ``task_config:`` files (see that function's own docstring for the full
    design -- "simultaneous per-skill task configs"). Requires the env to expose a ``skill_id``
    tensor (e.g. UnifiedManager); RewardManager raises at construction time if this is set on an
    env that doesn't."""

    params_per_skill: dict[str, list[float]] | None = None
    """2026-08-15: opt-in per-skill overrides for specific entries in ``params``, ``{param_name:
    [n_skills]}``, indexed by ``env.skill_id`` -- e.g. ``{"deadzone": [0.02, 0.06]}``. ``None``
    (default) means every env uses the plain ``params`` dict as-is, byte-identical to before this
    field existed. Only meaningful for STATELESS term functions (params re-read on every
    ``compute()`` call) -- a stateful ``RewardTermBase`` subclass that caches a param as ``self.x``
    at ``__init__`` time will NOT see per-skill values here; that needs the class itself changed to
    gather at call time. Same sibling mechanism as ``weight_per_skill`` above -- see that field's
    own docstring for the full design ("simultaneous per-skill task configs"). Requires the env to
    expose a ``skill_id`` tensor; RewardManager raises at construction time if this is set on an
    env that doesn't."""

    tags: list[str] = field(default_factory=list)
    """Tags for categorizing reward terms (e.g., ["penalty", "tracking"])."""

    task_mode: str | None = None
    """If set, this term's contribution is zeroed for envs not currently in this task mode
    (per ``env.task_mode_mask(task_mode)``, only consulted when the env implements it — e.g.
    UnifiedManager). ``None`` (the default) means always active, matching every existing
    experiment's behavior exactly."""


@dataclass(frozen=True)
class RewardManagerCfg:
    """Configuration for the reward manager."""

    terms: dict[str, RewardTermCfg] = field(default_factory=dict)
    """Mapping of reward term name to configuration."""

    only_positive_rewards: bool = False
    """If ``True``, clip the total reward to be non-negative."""
