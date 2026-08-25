"""Observation manager for computing and processing observations."""

from __future__ import annotations

from collections import deque
from typing import Any, Callable

import torch

from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg
from holosoma.managers.utils import resolve_callable

from .base import ObservationTermBase


class ObservationManager:
    """Manages observation computation with groups, terms, and processing.

    The manager orchestrates the computation of observations from multiple terms,
    applying scaling, noise, clipping, and history buffering to produce the final
    observation tensors. It maintains exact equivalence with the direct observation
    system while providing a more modular structure.

    Parameters
    ----------
    cfg : ObservationManagerCfg
        Configuration specifying observation groups and terms.
    env : Any
        Environment instance (typically a ``BaseTask`` subclass).
    device : str
        Device to place tensors on.
    """

    def __init__(self, cfg: ObservationManagerCfg, env: Any, device: str):
        self.cfg = cfg
        self.env = env
        self.device = device
        self.logger = getattr(env, "logger", None)

        # Storage for resolved functions and stateful terms
        self._term_funcs: dict[str, dict[str, Callable]] = {}
        self._term_instances: dict[str, dict[str, ObservationTermBase]] = {}

        # History buffers: group_name -> term_name -> deque
        self._history_buffers: dict[str, dict[str, deque]] = {}

        # Latency (delay_step_range) buffers: group_name -> term_name -> rolling buffer
        # [num_envs, max_delay + 1, obs_dim], and the matching per-env delay index
        # [num_envs] each term reads from. Lazily allocated on first compute (obs_dim isn't
        # known until a term actually runs) -- mirrors the action-delay queue pattern in
        # managers/action/terms/joint_control.py (same "rolling buffer + per-env index" shape),
        # just for observations instead of actions.
        self._delay_buffers: dict[str, dict[str, torch.Tensor]] = {}
        self._delay_idx: dict[str, dict[str, torch.Tensor]] = {}

        # Zero-order hold (hold_step_range) state: group_name -> term_name -> the last-refreshed
        # value [num_envs, obs_dim], the per-env refresh PERIOD drawn from hold_step_range at each
        # reset and held fixed for that episode [num_envs] (same contract as _delay_idx), and the
        # per-env countdown-to-next-refresh within that period [num_envs]. Same lazy-allocation
        # rationale as the delay buffers above. See ObsTermCfg.hold_step_range / this class's
        # _apply_hold.
        self._hold_cache: dict[str, dict[str, torch.Tensor]] = {}
        self._hold_period: dict[str, dict[str, torch.Tensor]] = {}
        self._hold_countdown: dict[str, dict[str, torch.Tensor]] = {}

        # Stale-frame reuse (stale_probability) cache: group_name -> term_name -> the last
        # NON-stale (i.e. actually freshly computed, post noise/delay/hold) value [num_envs,
        # obs_dim]. Simpler than the hold cache above -- no period/countdown, since this is a
        # fresh per-tick Bernoulli draw, not a per-episode-fixed refresh schedule. Same lazy-
        # allocation rationale. See ObsTermCfg.stale_probability / this class's _apply_stale.
        self._stale_cache: dict[str, dict[str, torch.Tensor]] = {}

        # 2026-08-18, FIX 6 (warm_start_obs_ramp_steps): load-time blend state, configured via
        # set_warm_start_blend() -- see that method's own docstring. Empty/0.0 by construction
        # until a caller (FastSACAgent.load()) opts in, so a manager that's never told about a
        # warm-start config change behaves exactly as before this existed.
        self._warm_start_blend: dict[str, dict[str, ObsTermCfg]] = {}
        self._warm_start_ramp_steps: float = 0.0
        self._warm_start_step_offset: int = 0

        # Initialize groups
        self._initialize_groups()

    def _initialize_groups(self) -> None:
        """Initialize observation groups and resolve term functions."""
        for group_name, group_cfg in self.cfg.groups.items():
            self._term_funcs[group_name] = {}
            self._term_instances[group_name] = {}
            self._history_buffers[group_name] = {}
            self._delay_buffers[group_name] = {}
            self._delay_idx[group_name] = {}
            self._hold_cache[group_name] = {}
            self._hold_period[group_name] = {}
            self._hold_countdown[group_name] = {}
            self._stale_cache[group_name] = {}

            for term_name, term_cfg in group_cfg.terms.items():
                # Resolve function
                func = resolve_callable(term_cfg.func, context="observation term")

                # Check if it's a class (stateful) or function (stateless)
                if isinstance(func, type) and issubclass(func, ObservationTermBase):
                    # Stateful term - instantiate
                    instance = func(term_cfg, self.env)
                    self._term_instances[group_name][term_name] = instance
                else:
                    # Stateless function
                    self._term_funcs[group_name][term_name] = func

                # Initialize history buffer if needed (using group-level history_length)
                if group_cfg.history_length > 1:
                    self._history_buffers[group_name][term_name] = deque(maxlen=group_cfg.history_length)

    def compute(self, *, modify_history: bool = True) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Compute all observation groups.

        Parameters
        ----------
        modify_history : bool, optional
            If ``True``, update history buffers; if ``False``, preserve them
            for bootstrapping. Defaults to ``True``.

        Returns
        -------
        dict[str, torch.Tensor | dict[str, torch.Tensor]]
            Mapping from group names to observation tensors or dictionaries of tensors.
        """
        obs_dict = {}
        for group_name in self.cfg.groups:
            obs_dict[group_name] = self.compute_group(group_name, modify_history=modify_history)
        return obs_dict

    def set_warm_start_blend(self, old_term_cfgs_by_group: dict[str, dict[str, ObsTermCfg]], ramp_steps: float) -> None:
        """FIX 6 (2026-08-18): configure a load-time blend from a checkpoint's OWN observation
        config toward the current one, for exactly the terms that changed between them.

        Called by ``FastSACAgent.load()`` right after it detects a config mismatch (the same
        detection ``_reset_normalizer_slots_for_shifted_obs_terms`` uses) -- this method itself
        does no detection, it only stores what it's given.

        WHY THIS EXISTS (measured, not theoretical -- see ``MultiSkillConfig.
        warm_start_obs_ramp_steps``'s own docstring for the full writeup): the normalizer-reset
        guard fires correctly but does not prevent a real shock (action_std 0.052->0.244, 4.7x, in
        one logging interval on a real run) when the changed terms are POPULATION changes (a term
        that was hard-zero for most envs becomes genuinely live), because a per-feature normalizer
        cannot correct the JOINT/conditional structure the actor's weights were fit to. Blending
        toward the checkpoint's own old-style value at load time, instead of cutting over
        instantly, is a distribution-shift fix at the right level -- the input the network SEES,
        not just its scale.

        Parameters
        ----------
        old_term_cfgs_by_group : dict[str, dict[str, ObsTermCfg]]
            Per group, per term: the checkpoint's OWN saved ``ObsTermCfg`` for every term whose
            (params, scale, task_mode, clip) differs from what this manager's CURRENT config has
            for that same term name. Terms not listed here are never blended (todays' value is
            used immediately, alpha implicitly 1.0) -- this includes terms present in only one
            side (a width change, out of scope for blending same as for the normalizer guard) and
            terms whose config didn't change at all.
        ramp_steps : float
            Ticks over which alpha goes 0 -> 1. <= 0.0 disables blending entirely (same effect as
            never calling this method) -- checked by ``_warm_start_alpha``, not here, so a caller
            can pass the raw config value unconditionally without its own branch.

        The offset step is captured HERE, from ``self.env.common_step_counter`` at the moment this
        is called (i.e. at the checkpoint load), not from the agent's own ``global_step`` -- this
        class only ever sees ``self.env``, never the agent, so it needs its own clock, and the
        env's per-tick counter is the one already used throughout this codebase for exactly this
        kind of "ticks since X happened" bookkeeping (see ``UnifiedManager.pre_kick_steps_since``/
        ``post_flip_steps_since``, the same pattern applied at the manager level instead of per-env
        because a load-time event affects every env simultaneously, unlike a per-env mid-episode
        transition).
        """
        self._warm_start_blend = {group: dict(terms) for group, terms in old_term_cfgs_by_group.items() if terms}
        self._warm_start_ramp_steps = float(ramp_steps)
        self._warm_start_step_offset = int(getattr(self.env, "common_step_counter", 0))

    def _warm_start_alpha(self) -> float | None:
        """``None`` when no blend is active (ramp off, or nothing was ever configured to blend) --
        checked BEFORE touching any per-term state, so ``compute_group`` pays nothing once the
        default (never-configured) state applies. Otherwise a float in [0, 1]; 1.0 once
        ``ramp_steps`` ticks have elapsed since ``set_warm_start_blend`` was called, at which point
        callers should treat blending as complete (this class does: see ``compute_group``'s
        ``alpha < 1.0`` guard, which skips the extra old-value computation once fully faded rather
        than doing it forever and multiplying by a no-op 0.0 weight)."""
        if self._warm_start_ramp_steps <= 0.0 or not self._warm_start_blend:
            return None
        elapsed = int(getattr(self.env, "common_step_counter", self._warm_start_step_offset)) - self._warm_start_step_offset
        return max(0.0, min(1.0, elapsed / self._warm_start_ramp_steps))

    def compute_group(self, group_name: str, *, modify_history: bool = True) -> torch.Tensor | dict[str, torch.Tensor]:
        """Compute observations for a specific group.

        This method replicates the exact behavior of the direct
        ``parse_observation()`` function to ensure equivalence.

        Parameters
        ----------
        group_name : str
            Name of the observation group to compute.
        modify_history : bool, optional
            If ``True``, update history buffers; if ``False``, preserve them
            for bootstrapping. Defaults to ``True``.

        Returns
        -------
        torch.Tensor | dict[str, torch.Tensor]
            Concatenated tensor when ``group.concatenate`` is ``True``; otherwise
            a dictionary of tensors keyed by term name.
        """
        group_cfg = self.cfg.groups[group_name]
        obs_tensors = {}

        for term_name, term_cfg in group_cfg.terms.items():
            # 1. Compute base observation
            obs = self._compute_term(group_name, term_name, term_cfg)

            # 2. Apply noise (matches direct: noise before scaling)
            if group_cfg.enable_noise and (term_cfg.noise > 0 or term_cfg.noise_range_coefficient > 0):
                obs = self._apply_noise(obs, term_cfg.noise, term_cfg.noise_range_coefficient)

            # 2b. Apply latency (delayed reading of an ALREADY-noised value -- a real delayed
            # sensor reading was itself noisy at the instant it was captured, not perturbed with
            # today's noise applied to a stale position). Same enable_noise gate as noise itself:
            # only "realistic perception" groups experience delay; a privileged/critic group
            # always reads the instantaneous value regardless of this setting.
            if group_cfg.enable_noise and term_cfg.delay_step_range is not None:
                obs = self._apply_delay(group_name, term_name, obs, term_cfg, modify_buffer=modify_history)

            # 2c. Apply zero-order hold (a real held/frozen reading was captured -- with whatever
            # delay it already has above -- at the moment of the pipeline's LAST update, and
            # should stay exactly that until the next one; recomputing today's noise/delay on top
            # of a "stale" read would contradict what "stale" means). Same enable_noise gate.
            if group_cfg.enable_noise and term_cfg.hold_step_range is not None:
                obs = self._apply_hold(group_name, term_name, obs, term_cfg, modify_buffer=modify_history)

            # 2d. Apply stale-frame reuse (a single dropped/repeated sensor frame, transient and
            # self-correcting -- distinct from _apply_hold's deterministic slower-update-rate
            # modeling). Same enable_noise gate. Called on _apply_hold's OUTPUT -- see
            # ObsTermCfg.stale_probability / _apply_stale's own docstrings for why.
            if group_cfg.enable_noise and term_cfg.stale_probability > 0.0:
                obs = self._apply_stale(group_name, term_name, obs, term_cfg, modify_buffer=modify_history)

            # 3. Apply scaling (matches direct: scale after noise)
            obs = self._apply_scale(obs, term_cfg.scale)

            # 4. Apply clipping (if specified)
            if term_cfg.clip is not None:
                obs = obs.clip(term_cfg.clip[0], term_cfg.clip[1])

            # 4b. Zero (never omit — the concatenated group width must stay constant) this term
            # for envs not currently in its task mode (no-op unless the env implements
            # task_mode_mask, e.g. UnifiedManager, AND the term opts in via task_mode).
            # Deliberately applied AFTER noise: masking before noise let _apply_noise re-inject
            # nonzero values into slots that were just zeroed, leaking small random values from
            # the inactive task's observation block into the policy input.
            #
            # 2026-08-18 (FIX 3, pre_kick_obs_ramp_steps): prefer the env's SOFT mask when it
            # offers one. UnifiedManager.task_mode_mask_soft returns the identical bool tensor
            # while the ramp is off (its 0.0 default), so this is bit-for-bit the previous line in
            # every existing config; when the ramp is on it returns a float crossfade over the
            # ticks following a mid-episode locomotion->kick entry, which is the only way to stop
            # this gate from stepping the policy's INPUT (a deterministic policy given a step input
            # must give a step output -- reward-side smoothing provably cannot fix it, which is why
            # the three existing pre_kick_* ramps did not). hasattr(task_mode_mask) stays the gate
            # that decides whether masking applies AT ALL, so a non-UnifiedManager env is untouched.
            if term_cfg.task_mode is not None and hasattr(self.env, "task_mode_mask"):
                soft_mask_fn = getattr(self.env, "task_mode_mask_soft", None)
                raw_mask = (
                    soft_mask_fn(term_cfg.task_mode)
                    if soft_mask_fn is not None
                    else self.env.task_mode_mask(term_cfg.task_mode)
                )
                mask = raw_mask.to(obs.dtype).unsqueeze(-1)
                obs = obs * mask

            # 4c. 2026-08-18 (FIX 6, warm_start_obs_ramp_steps): if this term is under an active
            # load-time blend, `obs` above is the fully-processed NEW-style value (today's config,
            # unchanged) -- blend it toward the checkpoint's own OLD-style value, computed via the
            # SAME pipeline shape (compute -> scale -> clip -> HARD task_mode mask) but with the
            # checkpoint's saved params/scale/clip/task_mode. Deliberately skips noise/delay/soft-
            # masking for the old path (a reference point for CONTINUITY of the blend, not a full
            # statistical replica -- the new-path component still carries real noise, and the old
            # checkpoint's own training never had soft masking to replicate in the first place).
            # alpha is None (skip, zero overhead) unless a blend is actively configured; alpha>=1.0
            # is treated as "done" so this extra compute doesn't run forever after the ramp ends.
            alpha = self._warm_start_alpha()
            if alpha is not None and alpha < 1.0:
                old_term_cfg = self._warm_start_blend.get(group_name, {}).get(term_name)
                if old_term_cfg is not None:
                    old_obs = self._compute_term(group_name, term_name, old_term_cfg)
                    old_obs = self._apply_scale(old_obs, old_term_cfg.scale)
                    if old_term_cfg.clip is not None:
                        old_obs = old_obs.clip(old_term_cfg.clip[0], old_term_cfg.clip[1])
                    if old_term_cfg.task_mode is not None and hasattr(self.env, "task_mode_mask"):
                        old_mask = self.env.task_mode_mask(old_term_cfg.task_mode).to(old_obs.dtype).unsqueeze(-1)
                        old_obs = old_obs * old_mask
                    obs = (1.0 - alpha) * old_obs + alpha * obs

            # 5. Handle history buffering
            if group_cfg.history_length > 1:
                obs = self._apply_history(group_name, term_name, obs, group_cfg, modify_buffer=modify_history)

            obs_tensors[term_name] = obs

        # Concatenate or return dict
        if group_cfg.concatenate:
            # Concatenate in alphabetically sorted order (to match direct system behavior)
            # Direct system does: sorted(obs_config) before concatenation
            sorted_keys = sorted(obs_tensors.keys())
            return torch.cat([obs_tensors[key] for key in sorted_keys], dim=-1)
        return obs_tensors

    def _compute_term(self, group_name: str, term_name: str, term_cfg: ObsTermCfg) -> torch.Tensor:
        """Compute a single observation term.

        Parameters
        ----------
        group_name : str
            Name of the observation group.
        term_name : str
            Name of the observation term.
        term_cfg : ObsTermCfg
            Configuration for the observation term.

        Returns
        -------
        torch.Tensor
            Observation tensor with shape ``[num_envs, obs_dim]``.
        """
        # Check if stateful or stateless
        if term_name in self._term_instances[group_name]:
            # Stateful term
            instance = self._term_instances[group_name][term_name]
            obs = instance(self.env, **term_cfg.params)
        else:
            # Stateless function
            func = self._term_funcs[group_name][term_name]
            obs = func(self.env, **term_cfg.params)

        return obs.clone()

    def _apply_noise(self, obs: torch.Tensor, noise_scale: float, range_coefficient: float = 0.0) -> torch.Tensor:
        """Apply uniform observation noise.

        The implementation matches the direct path, sampling uniform noise in
        ``[-effective_scale, effective_scale]``, where ``effective_scale = noise_scale +
        range_coefficient * ||obs||`` -- ``range_coefficient=0.0`` (the default for every term
        registered before this parameter existed) makes ``effective_scale`` exactly
        ``noise_scale``, bit-identical to the original flat-noise behavior.

        Parameters
        ----------
        obs : torch.Tensor
            Observation tensor.
        noise_scale : float
            Flat magnitude of the uniform noise.
        range_coefficient : float, optional
            Additional noise magnitude per unit of ``obs``'s own L2 norm (e.g. distance-dependent
            depth-sensor error). 0.0 (default) disables this term entirely.

        Returns
        -------
        torch.Tensor
            Noisy observation tensor.
        """
        effective_scale = noise_scale
        if range_coefficient > 0.0:
            obs_range = torch.linalg.norm(obs, dim=-1, keepdim=True)
            effective_scale = noise_scale + range_coefficient * obs_range
        # Direct uses: (torch.rand_like(obs) * 2.0 - 1.0) * noise_scale
        noise = (torch.rand_like(obs) * 2.0 - 1.0) * effective_scale
        return obs + noise

    def _apply_delay(
        self,
        group_name: str,
        term_name: str,
        obs: torch.Tensor,
        term_cfg: ObsTermCfg,
        *,
        modify_buffer: bool = True,
    ) -> torch.Tensor:
        """Apply per-env randomized latency to an observation term.

        Maintains a rolling buffer ``[num_envs, max_delay + 1, obs_dim]`` of this term's
        (already-noised) recent values -- newest at index 0, oldest at ``max_delay`` -- and
        returns each env's own fixed-for-the-episode delayed slot, selected by
        ``self._delay_idx[group_name][term_name]`` (redrawn at every reset, see ``reset()``).
        Mirrors ``managers/action/terms/joint_control.py``'s action-delay queue exactly (same
        shift-and-insert buffer shape, same per-env index-select read), for observations instead
        of actions.

        Parameters
        ----------
        group_name, term_name : str
            Identify which term's buffer/index to use (lazily allocated here on first call, since
            ``obs_dim`` and ``num_envs`` aren't known until a term has actually run once).
        obs : torch.Tensor
            This step's (already noise-perturbed) observation, to push into the buffer.
        term_cfg : ObsTermCfg
            Must have ``delay_step_range`` set (checked by the caller before this is invoked).
        modify_buffer : bool, optional
            If ``True``, shift the buffer and insert ``obs`` at slot 0 (the normal per-step
            update). If ``False`` (the ``_compute_final_observations`` bootstrap call, same
            ``modify_history`` flag the history-stacking mechanism already threads through),
            read the delayed slot WITHOUT mutating the buffer -- otherwise a terminal-step
            "final observations" read would double-shift the buffer before the real post-reset
            compute() call later in the same physical step corrupts the delay sequence.

        Returns
        -------
        torch.Tensor
            This env's delayed reading, shape matching ``obs``.
        """
        # Lazily allocated here (not eagerly at __init__/_initialize_groups) because obs_dim
        # isn't known without actually calling the term function, and the env isn't guaranteed
        # to be fully constructed (simulator/scene ready) that early -- get_obs_dims() has this
        # same constraint, which is why IT also defers calling _compute_term(). One consequence:
        # reset(), called during the env's very first reset_all() (before compute() has ever run
        # once), can't yet redraw a delay index for a buffer that doesn't exist -- so every env's
        # FIRST episode gets delay_idx=0 (no delay) until its first real mid-training reset draws
        # a proper value. Self-correcting after one episode; not worth the risk of calling a real
        # observation term before the env is known to be ready just to avoid it.
        buffer = self._delay_buffers[group_name].get(term_name)
        if buffer is None:
            max_delay = term_cfg.delay_step_range[1]
            obs_dim = obs.shape[1]
            buffer = torch.zeros(self.env.num_envs, max_delay + 1, obs_dim, device=self.device)
            self._delay_buffers[group_name][term_name] = buffer
            self._delay_idx[group_name][term_name] = torch.zeros(
                self.env.num_envs, dtype=torch.long, device=self.device
            )

        if modify_buffer:
            buffer[:, 1:] = buffer[:, :-1].clone()
            buffer[:, 0] = obs.clone()

        delay_idx = self._delay_idx[group_name][term_name]
        return buffer[torch.arange(self.env.num_envs, device=self.device), delay_idx].clone()

    def _apply_hold(
        self,
        group_name: str,
        term_name: str,
        obs: torch.Tensor,
        term_cfg: ObsTermCfg,
        *,
        modify_buffer: bool = True,
    ) -> torch.Tensor:
        """Apply per-env zero-order hold to an observation term: refresh only every N control
        ticks (N drawn per-env from ``term_cfg.hold_step_range`` at each reset, held fixed for
        that episode), returning the same cached value on every tick in between -- models a
        perception/fusion pipeline whose own UPDATE RATE is slower than the control loop (e.g. a
        25Hz ball-pose estimator under 50Hz control -> ``hold_step_range=(2, 2)``), distinct from
        ``_apply_delay``'s transport LATENCY on a pipeline that still updates every tick.
        Deliberately called on ``_apply_delay``'s OUTPUT (see ``compute_group`` step 2c), so the
        value being held already carries whatever delay it would have had at the moment it was
        captured -- a real held reading doesn't ALSO keep re-sampling a fresh delay every tick
        while frozen.

        Each env keeps its own countdown-to-next-refresh, seeded (at ``reset()``, or on this
        term's very first call for an env whose episode predates any reset -- same first-episode
        quirk ``_apply_delay`` documents) to a random PHASE within its drawn period, so envs aren't
        all refreshing on the same global tick, mirroring real independent, unsynchronized sensors
        rather than one shared clock.

        Parameters
        ----------
        group_name, term_name : str
            Identify which term's cache/period/countdown to use (lazily allocated here, same
            rationale as ``_apply_delay``).
        obs : torch.Tensor
            This step's already noise/delay-processed observation -- the candidate NEXT cached
            value, used only for envs whose countdown says a refresh is due this tick.
        term_cfg : ObsTermCfg
            Must have ``hold_step_range`` set (checked by the caller before this is invoked).
        modify_buffer : bool, optional
            Same ``modify_history``-threaded flag ``_apply_delay`` takes: ``False`` (the
            ``_compute_final_observations`` bootstrap call) reads the current cache WITHOUT
            advancing the countdown or refreshing it, so a terminal-step read can't steal a tick
            from the real post-reset sequence.

        Returns
        -------
        torch.Tensor
            This env's currently-held reading, shape matching ``obs``.
        """
        cache = self._hold_cache[group_name].get(term_name)
        if cache is None:
            # Seed directly from this first-ever observed value (not zeros, unlike the delay
            # buffer -- there's no "N steps of real history" to wait for here, just one cached
            # slot). Period defaults to the range's own LOW end (the mildest/least-holding value,
            # same "safe until the first real reset" bias delay_idx's own 0-default has) and
            # countdown starts at 0 so this same call immediately treats itself as "due",
            # refreshing the (already-correct) cache and drawing a real `period - 1` countdown for
            # next time -- self-consistent, no wrong-stale-read window on the very first call.
            cache = obs.clone()
            self._hold_cache[group_name][term_name] = cache
            self._hold_period[group_name][term_name] = torch.full(
                (self.env.num_envs,), term_cfg.hold_step_range[0], dtype=torch.long, device=self.device
            )
            self._hold_countdown[group_name][term_name] = torch.zeros(
                self.env.num_envs, dtype=torch.long, device=self.device
            )

        period = self._hold_period[group_name][term_name]
        countdown = self._hold_countdown[group_name][term_name]
        if modify_buffer:
            due = countdown <= 0
            if bool(due.any()):
                cache[due] = obs[due]
            countdown -= 1
            countdown[due] = period[due] - 1

        return cache.clone()

    def _apply_stale(
        self,
        group_name: str,
        term_name: str,
        obs: torch.Tensor,
        term_cfg: ObsTermCfg,
        *,
        modify_buffer: bool = True,
    ) -> torch.Tensor:
        """Apply per-env stale-frame reuse to an observation term: with probability
        ``term_cfg.stale_probability`` (re-rolled EVERY control tick, unlike ``_apply_hold``'s
        per-episode-fixed period), return the previous tick's already-fully-processed reading
        instead of this tick's -- models a single dropped/repeated sensor frame (e.g. one skipped
        LiDAR packet), distinct from ``_apply_hold``'s deterministic "pipeline update rate is
        slower than control rate" modeling. Deliberately called on ``_apply_hold``'s OUTPUT (see
        ``compute_group`` step 2d), so the value reused is whatever the pipeline would have
        legitimately returned last tick (already noised/delayed/held) -- a dropped frame doesn't
        ALSO re-derive a fresh delay/hold while it's being skipped.

        During a run of consecutive stale ticks for one env, the cache does NOT advance: it only
        updates on a tick that was NOT stale, so a 3-tick stale streak returns the SAME frame three
        times, not three increasingly-old ones -- mirrors a real dropped-packet sensor, which keeps
        re-publishing its last successfully-received reading, not silently drifting further back
        in time with every additional drop.

        Parameters
        ----------
        group_name, term_name : str
            Identify which term's cache to use (lazily allocated here, same rationale as
            ``_apply_hold``).
        obs : torch.Tensor
            This step's already noise/delay/hold-processed observation -- the candidate return
            value for envs whose per-tick draw says this reading is NOT stale.
        term_cfg : ObsTermCfg
            Must have ``stale_probability > 0.0`` (checked by the caller before this is invoked).
        modify_buffer : bool, optional
            Same ``modify_history``-threaded flag ``_apply_delay``/``_apply_hold`` take: ``False``
            (the ``_compute_final_observations`` bootstrap call) reads the current cache WITHOUT
            drawing a fresh stale mask or advancing it, so a terminal-step read can't consume a
            tick from the real post-reset sequence.

        Returns
        -------
        torch.Tensor
            This env's returned reading (fresh or reused), shape matching ``obs``.
        """
        cache = self._stale_cache[group_name].get(term_name)
        if cache is None:
            # Seed from this first-ever value, same "no history to wait for" rationale as
            # _apply_hold's own cache seeding.
            cache = obs.clone()
            self._stale_cache[group_name][term_name] = cache

        if not modify_buffer:
            return cache.clone()

        stale_mask = torch.rand(self.env.num_envs, device=self.device) < term_cfg.stale_probability
        result = torch.where(stale_mask.unsqueeze(-1), cache, obs)
        # Advances the cache to whatever was actually returned this tick: a fresh env's cache
        # becomes today's real value; a stale env's cache is set to itself (result == cache for
        # that env), a no-op that keeps the SAME frame available for the next tick's possible
        # reuse -- this is what makes a stale streak return one unchanging frame, not a chain.
        cache.copy_(result)
        return result.clone()

    def _apply_scale(self, obs: torch.Tensor, scale: float | tuple) -> torch.Tensor:
        """Apply scaling to an observation tensor.

        Parameters
        ----------
        obs : torch.Tensor
            Observation tensor.
        scale : float or tuple
            Scalar or per-dimension scale factors.

        Returns
        -------
        torch.Tensor
            Scaled observation tensor.
        """
        return obs * scale

    def _apply_history(
        self, group_name: str, term_name: str, obs: torch.Tensor, group_cfg: ObsGroupCfg, *, modify_buffer: bool = True
    ) -> torch.Tensor:
        """Apply history buffering to an observation term.

        Maintains a circular buffer of past observations and returns them
        concatenated along the feature dimension.

        Parameters
        ----------
        group_name : str
            Name of the observation group.
        term_name : str
            Name of the observation term.
        obs : torch.Tensor
            Current observation tensor with shape ``[num_envs, obs_dim]``.
        group_cfg : ObsGroupCfg
            Configuration for the observation group (contains ``history_length``).
        modify_buffer : bool, optional
            If ``True``, append to the history buffer; if ``False``, preserve the
            buffer contents. Defaults to ``True``.

        Returns
        -------
        torch.Tensor
            Historical observations with shape ``[num_envs, obs_dim * history_length]``.
        """
        buffer = self._history_buffers[group_name][term_name]

        if modify_buffer:
            # Add current observation to buffer
            buffer.append(obs)
            history = list(buffer)
        else:
            # Don't modify buffer - create temporary history with current obs
            history = list(buffer) + [obs]
            # Trim to history_length if needed
            if len(history) > group_cfg.history_length:
                history = history[-group_cfg.history_length :]

        # If buffer not full yet, pad with zeros (same as direct behavior)
        if len(history) < group_cfg.history_length:
            num_missing = group_cfg.history_length - len(history)
            obs_dim = obs.shape[1]
            padding = [torch.zeros(self.env.num_envs, obs_dim, device=self.device) for _ in range(num_missing)]
            history = padding + history

        # Stack along time dimension: [num_envs, history_length, obs_dim]
        stacked = torch.stack(history, dim=1)

        # Flatten to [num_envs, history_length * obs_dim]
        return stacked.reshape(self.env.num_envs, -1)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset observation history and stateful terms.

        Parameters
        ----------
        env_ids : torch.Tensor or None, optional
            Environment IDs to reset. If ``None``, reset all environments.
        """
        # Normalize environment ids
        env_ids_tensor: torch.Tensor | None
        if env_ids is None:
            env_ids_tensor = None
        elif isinstance(env_ids, torch.Tensor):
            env_ids_tensor = env_ids.to(device=self.device, dtype=torch.long)
        else:
            env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        # Reset or clear history buffers
        for group_buffers in self._history_buffers.values():
            for buffer in group_buffers.values():
                if env_ids_tensor is None:
                    buffer.clear()
                else:
                    if env_ids_tensor.numel() == 0:
                        continue
                    for history in buffer:
                        history[env_ids_tensor] = 0.0

        # Reset latency buffers + redraw each reset env's own delay index. Zeroing the buffer
        # (not just redrawing the index) matters: without it, a delayed read right after reset
        # could return a stale value from the PREVIOUS episode's ball position, not this one's --
        # mirrors action_queue's own reset-time zeroing in joint_control.py.
        for group_name, term_buffers in self._delay_buffers.items():
            group_cfg = self.cfg.groups[group_name]
            for term_name, buffer in term_buffers.items():
                if env_ids_tensor is None:
                    buffer.zero_()
                elif env_ids_tensor.numel() > 0:
                    buffer[env_ids_tensor] = 0.0

                delay_step_range = group_cfg.terms[term_name].delay_step_range
                idx_buf = self._delay_idx[group_name][term_name]
                low, high = delay_step_range
                if env_ids_tensor is None:
                    idx_buf[:] = torch.randint(low, high + 1, (self.env.num_envs,), device=self.device)
                elif env_ids_tensor.numel() > 0:
                    idx_buf[env_ids_tensor] = torch.randint(
                        low, high + 1, (env_ids_tensor.numel(),), device=self.device
                    )

        # Reset hold caches + redraw each reset env's own countdown phase. Zeroing the cache (not
        # just the countdown) matters for the same reason as the delay buffer above: without it, a
        # held read right after reset could return a stale value from the PREVIOUS episode until
        # its countdown happens to reach 0. Both the PERIOD (this episode's own refresh interval,
        # same "held fixed for the episode" contract as _delay_idx) and, within it, a random PHASE
        # are redrawn -- not "phase=0, forcing an immediate refresh" -- so envs stay desynchronized
        # from each other across resets, same as mid-episode; forcing phase=0 here would gradually
        # resync every env's refresh tick to whatever global step resets happen to cluster around.
        # Phase is drawn as (rand() * period) rather than randint(0, period) so a period that
        # happens to draw 0 (only possible if hold_step_range's low end is 0) degrades gracefully
        # to phase=0 instead of randint raising on an empty [0, 0) range -- period=0 already means
        # "refresh every tick" (see _apply_hold), so phase is moot for that env anyway.
        for group_name, term_caches in self._hold_cache.items():
            group_cfg = self.cfg.groups[group_name]
            for term_name, cache in term_caches.items():
                if env_ids_tensor is None:
                    cache.zero_()
                elif env_ids_tensor.numel() > 0:
                    cache[env_ids_tensor] = 0.0

                low, high = group_cfg.terms[term_name].hold_step_range
                period_buf = self._hold_period[group_name][term_name]
                countdown_buf = self._hold_countdown[group_name][term_name]
                if env_ids_tensor is None:
                    new_period = torch.randint(low, high + 1, (self.env.num_envs,), device=self.device)
                    period_buf[:] = new_period
                    countdown_buf[:] = (torch.rand(self.env.num_envs, device=self.device) * new_period.float()).long()
                elif env_ids_tensor.numel() > 0:
                    n = env_ids_tensor.numel()
                    new_period = torch.randint(low, high + 1, (n,), device=self.device)
                    period_buf[env_ids_tensor] = new_period
                    countdown_buf[env_ids_tensor] = (torch.rand(n, device=self.device) * new_period.float()).long()

        # Reset stale-frame-reuse caches. Zeroing (not leaving it) matters for the same reason as
        # the delay/hold buffers above: without it, a stale-lucky read right after reset could
        # return the PREVIOUS episode's last real frame instead of this one's. No period/countdown
        # to redraw here (see _apply_stale's own docstring for why this cache is simpler than the
        # hold cache) -- clearing it is the whole reset contract.
        for group_name, term_caches in self._stale_cache.items():
            for term_name, cache in term_caches.items():
                if env_ids_tensor is None:
                    cache.zero_()
                elif env_ids_tensor.numel() > 0:
                    cache[env_ids_tensor] = 0.0

        # Reset stateful term instances
        for group_instances in self._term_instances.values():
            for instance in group_instances.values():
                instance.reset(env_ids_tensor)

    def get_obs_dims(self) -> dict[str, int | dict[str, int]]:
        """Get observation dimensions for each group.

        Returns
        -------
        dict[str, int or dict[str, int]]
            Mapping from group names to observation dimensions. Values are integers
            when the group concatenates terms, otherwise dictionaries of per-term
            dimensions.
        """
        dims: dict[str, int | dict[str, int]] = {}
        for group_name, group_cfg in self.cfg.groups.items():
            if group_cfg.concatenate:
                # Sum up all term dimensions
                total_dim = 0
                for term_name, term_cfg in group_cfg.terms.items():
                    # Compute term once to get its dimension
                    obs = self._compute_term(group_name, term_name, term_cfg)
                    term_dim = obs.shape[1]

                    # Account for history at group level
                    if group_cfg.history_length > 1:
                        term_dim *= group_cfg.history_length

                    total_dim += term_dim
                dims[group_name] = total_dim
            else:
                # Return dict of individual dimensions
                term_dims: dict[str, int] = {
                    term_name: self._compute_term(group_name, term_name, term_cfg).shape[1]
                    for term_name, term_cfg in group_cfg.terms.items()
                }
                dims[group_name] = term_dims
        return dims
