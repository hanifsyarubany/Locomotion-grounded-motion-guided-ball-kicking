"""Configuration types for termination manager."""

from __future__ import annotations

from dataclasses import field
from typing import Any

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class TerminationTermCfg:
    """Configuration for a single termination term."""

    func: str
    """Import path of the termination hook."""

    params: dict[str, Any] = field(default_factory=dict)
    """Additional parameters forwarded to the hook."""

    params_per_skill: dict[str, list[float]] | None = None
    """2026-08-15: opt-in per-skill overrides for specific entries in ``params``, ``{param_name:
    [n_skills]}``, indexed by ``env.skill_id`` -- e.g. ``{"deadzone": [0.15, 0.40]}``. ``None``
    (default) means every env uses the plain ``params`` dict as-is, byte-identical to before this
    field existed. Only meaningful for STATELESS hook functions (params re-read on every
    ``check()`` call) -- a stateful ``TerminationTermBase`` subclass that caches a param as
    ``self.x`` at ``__init__`` time will NOT see per-skill values here; that needs the class
    itself changed to gather at call time. See RewardTermCfg.weight_per_skill's own docstring for
    the sibling mechanism/full design (same underlying motivation: simultaneous per-skill task
    configs). Requires the env to expose a ``skill_id`` tensor; TerminationManager raises at
    construction time if this is set on an env that doesn't."""

    is_timeout: bool = False
    """Whether this term should be treated as a timeout condition."""

    task_mode: str | None = None
    """If set, this term can only trigger a reset for envs currently in this task mode (per
    ``env.task_mode_mask(task_mode)``, only consulted when the env implements it — e.g.
    UnifiedManager). ``None`` (the default) means always active, matching every existing
    experiment's behavior exactly."""


@dataclass(frozen=True)
class TerminationManagerCfg:
    """Configuration for the termination manager."""

    terms: dict[str, TerminationTermCfg] = field(default_factory=dict)
    """Mapping of termination term name to configuration."""
