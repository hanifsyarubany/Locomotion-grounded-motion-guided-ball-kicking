"""Base classes and protocols for termination terms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from holosoma.utils.safe_torch_import import torch

if TYPE_CHECKING:
    from holosoma.config_types.termination import TerminationTermCfg


class TerminationTermBase(ABC):
    """Base class for stateful termination terms."""

    handles_params_per_skill: bool = False
    """2026-08-15: set True on a subclass that reads ``cfg.params_per_skill`` itself in its own
    ``__init__`` (building its own per-skill tensor(s) and gathering by ``env.skill_id`` at
    ``__call__`` time) -- e.g. ``BadTracking`` (managers/termination/terms/wbt.py), whose
    ``bad_motion_body_pos_threshold``/``swing_threshold_multiplier`` are read once into
    ``self.x`` and can't use TerminationManager's generic per-call override (that mechanism is
    stateless-only, see TerminationTermCfg.params_per_skill's own docstring). False (default)
    means TerminationManager's own construction-time guard rejects params_per_skill on this term
    as a likely mistake (silently-never-consulted), matching every OTHER stateful term that
    hasn't opted into handling it itself."""

    def __init__(self, cfg: TerminationTermCfg, env: Any):
        self.cfg = cfg
        self.env = env

    @abstractmethod
    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset internal state for specified environments."""

    @abstractmethod
    def __call__(self, env: Any, **kwargs) -> torch.Tensor:
        """Evaluate termination condition."""
