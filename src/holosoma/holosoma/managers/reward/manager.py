"""Reward manager for computing reward signals."""

from __future__ import annotations

import importlib
from typing import Any

import torch
from loguru import logger

from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg

from .base import RewardTermBase


class RewardManager:
    """Manages reward computation as a weighted sum of individual terms.

    The reward manager computes the total reward by evaluating each configured
    reward term, multiplying by its weight and the environment's time step (dt),
    and summing the results. It tracks episodic sums for logging and supports
    both stateless (function) and stateful (class) reward terms.

    Parameters
    ----------
    cfg : RewardManagerCfg
        Configuration specifying reward terms and settings.
    env : Any
        Environment instance (typically a ``BaseTask`` subclass).
    device : str
        Device where tensors should be allocated.
    """

    def __init__(self, cfg: RewardManagerCfg, env: Any, device: str):
        self.cfg = cfg
        self.env = env
        self.device = device
        self.logger = getattr(env, "logger", None)

        # Storage for resolved functions and stateful terms
        self._term_funcs: dict[str, Any] = {}
        self._term_instances: dict[str, RewardTermBase] = {}
        self._term_names: list[str] = []
        self._term_cfgs: list[RewardTermCfg] = []

        # Initialize terms
        self._initialize_terms()

        # 2026-08-15: per-skill weight tensors, one per term whose RewardTermCfg.weight_per_skill
        # is set (see that field's own docstring -- "simultaneous per-skill task configs").
        # Precomputed once here, not per compute() call: building an [n_skills] tensor from a
        # plain float list is trivial, but there's no reason to redo it every control step.
        self._weight_per_skill_tensors: dict[str, torch.Tensor] = {
            name: torch.tensor(cfg.weight_per_skill, dtype=torch.float, device=self.device)
            for name, cfg in zip(self._term_names, self._term_cfgs)
            if cfg.weight_per_skill is not None
        }
        if self._weight_per_skill_tensors and not hasattr(self.env, "skill_id"):
            raise AttributeError(
                f"{len(self._weight_per_skill_tensors)} reward term(s) "
                f"({sorted(self._weight_per_skill_tensors)}) have a per-skill weight table, but "
                f"env {type(self.env).__name__} has no `skill_id` attribute -- per-skill reward "
                "weights require a UnifiedManager-family env. Failing at construction time rather "
                "than on the first compute() call."
            )

        # 2026-08-15: per-skill PARAM tensors, sibling to _weight_per_skill_tensors above -- one
        # nested dict per term with a RewardTermCfg.params_per_skill set, {param_name: [n_skills]
        # tensor}. Only meaningful for STATELESS terms (a stateful RewardTermBase caches its
        # params as self.x once in __init__ -- params_per_skill would silently never be consulted,
        # so that's rejected below rather than silently doing nothing).
        self._params_per_skill_tensors: dict[str, dict[str, torch.Tensor]] = {
            name: {
                pname: torch.tensor(pvals, dtype=torch.float, device=self.device)
                for pname, pvals in cfg.params_per_skill.items()
            }
            for name, cfg in zip(self._term_names, self._term_cfgs)
            if cfg.params_per_skill is not None
        }
        stateful_with_params_per_skill = sorted(set(self._params_per_skill_tensors) & set(self._term_instances))
        if stateful_with_params_per_skill:
            raise ValueError(
                f"reward term(s) {stateful_with_params_per_skill} are STATEFUL (a RewardTermBase "
                "subclass) but have params_per_skill set -- a stateful term reads its params ONCE "
                "in __init__ and caches them, so a per-call override here would silently never be "
                "consulted. Per-skill support for a stateful term requires changing that term's own "
                "class to gather from a tensor at call time, not this generic mechanism."
            )
        if self._params_per_skill_tensors and not hasattr(self.env, "skill_id"):
            raise AttributeError(
                f"{len(self._params_per_skill_tensors)} reward term(s) "
                f"({sorted(self._params_per_skill_tensors)}) have a per-skill params table, but "
                f"env {type(self.env).__name__} has no `skill_id` attribute -- per-skill reward "
                "params require a UnifiedManager-family env. Failing at construction time rather "
                "than on the first compute() call."
            )

        # Buffers for reward tracking
        self._reward_buf = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Episode sums for each term (for logging)
        self._episode_sums: dict[str, torch.Tensor] = {}
        self._episode_sums_raw: dict[str, torch.Tensor] = {}
        for term_name in self._term_names:
            self._episode_sums[term_name] = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            self._episode_sums_raw[term_name] = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

    def _initialize_terms(self) -> None:
        """Initialize reward terms and resolve their functions/classes."""
        for term_name, term_cfg in self.cfg.terms.items():
            # Skip terms that are inert for EVERY skill. A term with a per-skill table can be zero
            # for one skill and nonzero for another (e.g. shooting terms: nonzero under Stage C1,
            # zero under Stage B) -- .weight alone (just skill 0's representative value, see
            # RewardTermCfg.weight_per_skill's own docstring) isn't a reliable zero-check on its
            # own once that's set, order-dependent in a way this must not be.
            weights_all_zero = (
                term_cfg.weight == 0.0
                if term_cfg.weight_per_skill is None
                else all(w == 0.0 for w in term_cfg.weight_per_skill)
            )
            if weights_all_zero:
                continue

            # Resolve function or class
            func = self._resolve_function(term_cfg.func)

            # Check if it's a class (stateful) or function (stateless)
            if isinstance(func, type) and issubclass(func, RewardTermBase):
                # Stateful term - instantiate
                instance = func(term_cfg, self.env)
                self._term_instances[term_name] = instance
            else:
                # Stateless function
                self._term_funcs[term_name] = func

            self._term_names.append(term_name)
            self._term_cfgs.append(term_cfg)

    def _resolve_function(self, func: Any | str) -> Any:
        """Resolve a reward callable or class from a string specification.

        Parameters
        ----------
        func : Any or str
            Function or class reference, or a string like ``"module:object_name"``.

        Returns
        -------
        Any
            Resolved callable or class.

        Raises
        ------
        ValueError
            If the string path is malformed or the target cannot be imported.
        """
        if isinstance(func, str):
            # Parse string like "module.path:function_name"
            if ":" not in func:
                raise ValueError(f"Function string must be in format 'module:function', got: {func}")

            module_path, func_name = func.split(":", 1)
            try:
                module = importlib.import_module(module_path)
                return getattr(module, func_name)
            except (ImportError, AttributeError) as e:
                raise ValueError(f"Failed to import function '{func}': {e}") from e
        return func

    @property
    def active_terms(self) -> list[str]:
        """Names of active reward terms."""
        return self._term_names

    @property
    def episode_sums(self) -> dict[str, torch.Tensor]:
        """Episodic sums for each reward term (scaled)."""
        return self._episode_sums

    @property
    def episode_sums_raw(self) -> dict[str, torch.Tensor]:
        """Episodic sums for each reward term (raw, unscaled)."""
        return self._episode_sums_raw

    # 2026-08-18: per-TERM nan-probe. Root-caused via the run this fires on
    # (20260818_033003/052857-stageD-1skill-obs-fixes): FastSACAgent's own nan-probe
    # (fast_sac_agent.py, 2026-07-18) caught 'rewards' going non-finite at the SAC-update boundary
    # (1/6600 elements, one env), but that probe fires many steps downstream of reward computation
    # -- it can name the CORRUPTED TENSOR, not which of the ~80 reward terms produced it, because
    # RewardManager.compute() (below) accumulates every term straight into _reward_buf with NO
    # finiteness check anywhere in the loop. This closes that attribution gap by checking `rew_raw`
    # per term, at the exact point it's computed -- mirrors fast_sac_agent.py's own _nan_probe
    # construction (fire-once, log the first offending tensor + example value + finite-range
    # context) so the two probes read consistently if both ever appear in the same log.
    _reward_nan_probe_fired = False

    def _reward_nan_probe(self, term_name: str, rew_raw: torch.Tensor) -> None:
        if self._reward_nan_probe_fired:
            return
        finite = torch.isfinite(rew_raw)
        if finite.all():
            return
        self._reward_nan_probe_fired = True
        bad_envs = (~finite).nonzero(as_tuple=False).flatten()
        example_env = bad_envs[0].item()
        skill = int(self.env.skill_id[example_env].item()) if hasattr(self.env, "skill_id") else None
        task_mode = self.env.task_mode[example_env].item() if hasattr(self.env, "task_mode") else None
        logger.error(
            f"[reward-nan-probe] FIRST non-finite reward TERM: '{term_name}' -- "
            f"{bad_envs.numel()}/{rew_raw.shape[0]} envs non-finite, example env={example_env} "
            f"value={rew_raw[example_env].item()} skill_id={skill} task_mode={task_mode} "
            f"global_step={getattr(self.env, 'common_step_counter', 'unknown')}. "
            f"env indices (first 20): {bad_envs[:20].tolist()}{'...' if bad_envs.numel() > 20 else ''}"
        )
        finite_vals = rew_raw[finite]
        if finite_vals.numel() > 0:
            logger.error(
                f"[reward-nan-probe] '{term_name}' finite-value range: "
                f"min={finite_vals.min().item()} max={finite_vals.max().item()}"
            )

    def compute(self, dt: float) -> torch.Tensor:
        """Compute the total reward as a weighted sum of individual terms.

        Each reward term is evaluated, scaled by its configured weight and the
        environment time step, and accumulated into the total reward. Episodic
        sums are updated for logging purposes.

        Notes
        -----
        Curriculum scaling is handled by directly modifying term weights via
        :meth:`set_term_cfg`, rather than through extra scaling parameters.

        Parameters
        ----------
        dt : float
            Environment time-step interval.

        Returns
        -------
        torch.Tensor
            Net reward tensor with shape ``[num_envs]``.
        """
        # Reset computation
        self._reward_buf[:] = 0.0

        # Iterate over all reward terms
        for term_name, term_cfg in zip(self._term_names, self._term_cfgs):
            # Per-skill PARAM overrides (see RewardTermCfg.params_per_skill): gather each env's
            # own skill's value for the specific param(s) that diverge, leaving every other param
            # untouched. Only ever set on stateless terms (enforced at __init__ time above), so
            # this never needs to consider _term_instances.
            if term_name in self._params_per_skill_tensors:
                call_params = dict(term_cfg.params)
                for pname, tensor in self._params_per_skill_tensors[term_name].items():
                    call_params[pname] = tensor[self.env.skill_id]
            else:
                call_params = term_cfg.params

            # Compute raw reward value
            if term_name in self._term_instances:
                # Stateful term
                instance = self._term_instances[term_name]
                rew_raw = instance(self.env, **call_params)
            else:
                # Stateless function
                func = self._term_funcs[term_name]
                rew_raw = func(self.env, **call_params)

            # Validate shape
            if rew_raw.shape[0] != self.env.num_envs:
                raise ValueError(
                    f"Reward term '{term_name}' returned wrong shape. "
                    f"Expected [{self.env.num_envs}], got {rew_raw.shape}"
                )

            # Checked on the RAW value, before weight/dt scaling or task-mode masking: a term that
            # produces NaN/Inf is a bug in that term regardless of its configured weight, and
            # checking pre-mask also catches a term going non-finite for an env whose CURRENT mode
            # happens to zero it out here but won't always (task_mode changes across an episode).
            self._reward_nan_probe(term_name, rew_raw)

            # Zero out envs not currently in this term's task mode (no-op unless the env
            # implements task_mode_mask, e.g. UnifiedManager, and the term opts in via task_mode)
            if term_cfg.task_mode is not None and hasattr(self.env, "task_mode_mask"):
                rew_raw = rew_raw * self.env.task_mode_mask(term_cfg.task_mode).to(rew_raw.dtype)

            # Scale by weight and dt -- a per-skill table (see RewardTermCfg.weight_per_skill),
            # when present, wins over the plain scalar: gather each env's OWN skill's weight,
            # broadcasting to an [num_envs] tensor exactly like task_mode_mask's own mask above.
            if term_name in self._weight_per_skill_tensors:
                weight = self._weight_per_skill_tensors[term_name][self.env.skill_id]
            else:
                weight = term_cfg.weight
            rew_scaled = rew_raw * weight * dt

            # Accumulate
            self._reward_buf += rew_scaled

            # Track episodic sums
            self._episode_sums[term_name] += rew_scaled
            self._episode_sums_raw[term_name] += rew_raw

        # Optionally clip to positive
        if self.cfg.only_positive_rewards:
            self._reward_buf[:] = torch.clip(self._reward_buf, min=0.0)

        return self._reward_buf

    def reset(self, env_ids: torch.Tensor | None = None) -> dict[str, dict[str, torch.Tensor]]:
        """Reset reward tracking and return episodic sums for logging.

        Parameters
        ----------
        env_ids : torch.Tensor or None, optional
            Environment IDs to reset. If ``None``, reset all environments.

        Returns
        -------
        dict[str, dict[str, torch.Tensor]]
            Dictionary mirroring the direct reward path structure::

                {
                    "episode": {term_name: tensor_per_reset_env},
                    "episode_all": {term_name: tensor_per_all_envs},
                    "raw_episode": {...},
                    "raw_episode_all": {...},
                }
        """
        extras: dict[str, dict[str, torch.Tensor]] = {
            "episode": {},
            "episode_all": {},
            "raw_episode": {},
            "raw_episode_all": {},
        }

        # Resolve environment ids to operate on
        if env_ids is None:
            env_ids_tensor: torch.Tensor | None = None
            env_ids_slice: slice | torch.Tensor = slice(None)
        else:
            if isinstance(env_ids, torch.Tensor):
                env_ids_tensor = env_ids.to(device=self.device, dtype=torch.long)
            else:
                env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

            env_ids_slice = env_ids_tensor

        # Helper to detach values before zeroing internal buffers
        def _clone(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.detach().clone()

        # Populate scaled reward statistics
        for term_name in self._term_names:
            rew_all = self._episode_sums[term_name] / self.env.max_episode_length_s
            extras["episode_all"][f"rew_{term_name}"] = _clone(rew_all)
            if env_ids_tensor is None:
                extras["episode"][f"rew_{term_name}"] = _clone(rew_all)
            else:
                extras["episode"][f"rew_{term_name}"] = _clone(rew_all[env_ids_slice])

            # Reset episodic sums for the completed environments
            self._episode_sums[term_name][env_ids_slice] = 0.0

        # Populate raw (unscaled) reward statistics
        for term_name in self._term_names:
            rew_raw_all = self._episode_sums_raw[term_name] / self.env.max_episode_length_s
            extras["raw_episode_all"][f"raw_rew_{term_name}"] = _clone(rew_raw_all)
            if env_ids_tensor is None:
                extras["raw_episode"][f"raw_rew_{term_name}"] = _clone(rew_raw_all)
            else:
                extras["raw_episode"][f"raw_rew_{term_name}"] = _clone(rew_raw_all[env_ids_slice])

            self._episode_sums_raw[term_name][env_ids_slice] = 0.0

        # Reset stateful reward terms
        for instance in self._term_instances.values():
            instance.reset(env_ids=env_ids_tensor)

        return extras

    def get_term(self, name: str) -> Any:
        """Get reward term function or instance by name.

        Parameters
        ----------
        name : str
            Name of the reward term.

        Returns
        -------
        Any
            Reward term function or instance.

        Raises
        ------
        KeyError
            If the term name is not found.
        """
        if name in self._term_instances:
            return self._term_instances[name]
        if name in self._term_funcs:
            return self._term_funcs[name]
        raise KeyError(f"Reward term '{name}' not found")

    def get_term_cfg(self, name: str) -> RewardTermCfg:
        """Get reward term configuration by name.

        Parameters
        ----------
        name : str
            Name of the reward term.

        Returns
        -------
        RewardTermCfg
            Configuration for the specified reward term.

        Raises
        ------
        KeyError
            If the term name is not found.
        """
        try:
            idx = self._term_names.index(name)
            return self._term_cfgs[idx]
        except ValueError:
            raise KeyError(f"Reward term '{name}' not found")

    def set_term_cfg(self, name: str, cfg: RewardTermCfg) -> None:
        """Set reward term configuration by name.

        Parameters
        ----------
        name : str
            Name of the reward term.
        cfg : RewardTermCfg
            New configuration for the term.

        Raises
        ------
        KeyError
            If the term name is not found.
        """
        try:
            idx = self._term_names.index(name)
            self._term_cfgs[idx] = cfg
        except ValueError:
            raise KeyError(f"Reward term '{name}' not found")

    def __str__(self) -> str:
        """String representation of reward manager."""
        msg = f"<RewardManager> contains {len(self._term_names)} active terms.\n"
        msg += "Terms:\n"
        for name, cfg in zip(self._term_names, self._term_cfgs):
            msg += f"  - {name}: weight={cfg.weight}\n"
        return msg
