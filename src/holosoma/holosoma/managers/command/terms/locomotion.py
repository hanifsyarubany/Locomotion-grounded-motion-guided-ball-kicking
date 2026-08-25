"""Locomotion command sampling and gait state terms."""

from __future__ import annotations

from typing import Any, Sequence, cast

from holosoma.managers.command.base import CommandTermBase
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.torch_utils import torch_rand_float


class LocomotionCommand(CommandTermBase):
    """Stateful command term that owns command buffers and resampling logic."""

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)
        params = cfg.params or {}
        ranges = params.get("command_ranges")
        if ranges is None:
            raise ValueError("LocomotionCommand requires 'command_ranges' in params.")
        self.command_ranges: dict[str, Sequence[float]] = {key: tuple(value) for key, value in ranges.items()}
        self.stand_prob: float = float(params.get("stand_prob", 0.0))
        self.command_dim: int = params.get("command_dim", 3)
        self.commands: torch.Tensor | None = None
        # Stage D's post-swing -> locomotion handoff (pin_zero, below): per-env "don't touch this
        # env's command" flag. Nothing in the command stack skips step()'s periodic resample by
        # task_mode -- it runs unconditionally for every env, kick-mode included (see step()'s own
        # docstring note) -- so once an env is handed live control via a mid-episode task_mode
        # flip, its command buffer needs an explicit way to stay at the deterministic value the
        # flip set, rather than drifting to whatever this term's own periodic resample next draws.
        self._pinned_zero: torch.Tensor | None = None

    # ------------------------------------------------------------------ #
    # Lifecycle hooks
    # ------------------------------------------------------------------ #

    def setup(self) -> None:
        env = self.env
        commands = torch.zeros(env.num_envs, self.command_dim, dtype=torch.float32, device=env.device)
        self.commands = commands
        self._pinned_zero = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        if hasattr(env, "simulator"):
            env.simulator.commands = commands

    def reset(self, env_ids: torch.Tensor | None) -> None:
        commands = self.commands
        if commands is None:
            return
        idx = self._ensure_index_tensor(env_ids)
        if idx.numel() == 0:
            return

        # A genuine new episode un-pins -- it legitimately gets a fresh draw, same as before
        # pin_zero existed. Only a live mid-episode flip (pin_zero, below) should hold an env at
        # zero; a reset always supersedes it.
        if self._pinned_zero is not None:
            self._pinned_zero[idx] = False
        self._resample(idx)

    def step(self) -> None:
        commands = self.commands
        if commands is None or self.env.is_evaluating:
            return

        command_cfg = getattr(self.manager, "command_cfg", None) if hasattr(self, "manager") else None
        resample_time = getattr(command_cfg, "locomotion_command_resampling_time", None) if command_cfg else None
        if resample_time is None or resample_time <= 0:
            return

        interval = int(resample_time / self.env.dt)
        if interval <= 0 or not hasattr(self.env, "episode_length_buf"):
            return

        env_ids = (self.env.episode_length_buf % interval == 0).nonzero(as_tuple=False).flatten()
        env_ids = env_ids.to(device=self.env.device, dtype=torch.long)
        if self._pinned_zero is not None and env_ids.numel() > 0:
            env_ids = env_ids[~self._pinned_zero[env_ids]]
        if env_ids.numel() == 0:
            return

        self._resample(env_ids)

    def pin_zero(self, env_ids: torch.Tensor) -> None:
        """Force the given envs' command to exact deterministic [0, 0, 0] and hold it there --
        excluded from step()'s periodic resample -- until their next real reset() re-enables
        drawing. Used by UnifiedManager's Stage D post-swing -> locomotion handoff
        (kick_recovery_locomotion_flip_enabled): the instant a kick-mode env's task_mode flips to
        LOCOMOTION mid-episode, its command buffer becomes live-relevant for the first time (kick-
        mode envs' commands are otherwise inert garbage -- step()'s resample runs unconditionally
        regardless of task_mode, see step()'s own comment -- because nothing reads it while
        task_mode is KICK). Writing commands directly here, not just setting the pin flag, makes
        this robust to call-ordering against CommandManager.step() within the same tick."""
        commands = self.commands
        if commands is None:
            return
        idx = self._ensure_index_tensor(env_ids)
        if idx.numel() == 0:
            return
        commands[idx, :] = 0.0
        if self._pinned_zero is not None:
            self._pinned_zero[idx] = True

    def pin(self, env_ids: torch.Tensor, values: torch.Tensor) -> None:
        """Like pin_zero, but pins to an arbitrary per-env command vector instead of exact
        [0, 0, 0] -- excluded from step()'s periodic resample the same way, until unpin() or the
        next real reset(). Used by UnifiedManager's locomotion->kick D8 decel-and-retry fallback
        (mid_episode_kick_entry_prob's mirror-direction handoff), which needs to command a
        decaying-toward-a-floor forward speed rather than an outright stop -- same _pinned_zero
        exclusion mechanism pin_zero already established, not a new one."""
        commands = self.commands
        if commands is None:
            return
        idx = self._ensure_index_tensor(env_ids)
        if idx.numel() == 0:
            return
        commands[idx, :] = values
        if self._pinned_zero is not None:
            self._pinned_zero[idx] = True

    def unpin(self, env_ids: torch.Tensor) -> None:
        """Release env_ids from any pin (pin_zero or pin), restoring step()'s normal periodic
        resample without waiting for a full reset(). Used by UnifiedManager's D8 fallback decline
        path: giving up on a mid-episode kick entry means the env should resume ordinary
        randomized locomotion for the rest of the episode, not stay frozen at whatever decel value
        the fallback last held."""
        idx = self._ensure_index_tensor(env_ids)
        if idx.numel() == 0 or self._pinned_zero is None:
            return
        self._pinned_zero[idx] = False

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _resample(self, env_ids: torch.Tensor) -> None:
        commands = self.commands
        if commands is None or env_ids.numel() == 0:
            return

        device = self.env.device
        ranges = self.command_ranges

        commands[env_ids, 0] = torch_rand_float(
            ranges["lin_vel_x"][0],
            ranges["lin_vel_x"][1],
            (env_ids.shape[0], 1),
            device=device,
        ).squeeze(1)
        commands[env_ids, 1] = torch_rand_float(
            ranges["lin_vel_y"][0],
            ranges["lin_vel_y"][1],
            (env_ids.shape[0], 1),
            device=device,
        ).squeeze(1)
        commands[env_ids, 2] = torch_rand_float(
            ranges["ang_vel_yaw"][0],
            ranges["ang_vel_yaw"][1],
            (env_ids.shape[0], 1),
            device=device,
        ).squeeze(1)

        manager = getattr(self, "manager", None)
        if manager is not None:
            gait_state = manager.get_state("locomotion_gait")
        else:
            gait_state = None

        if gait_state is not None:
            cast("LocomotionGait", gait_state).resample_frequency(env_ids)

        if self.stand_prob > 0.0:
            stand_mask = torch.rand(env_ids.shape[0], device=device) <= self.stand_prob
            if stand_mask.any():
                commands[env_ids[stand_mask], :3] = 0.0

    def _ensure_index_tensor(self, env_ids: torch.Tensor | Sequence[int] | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.env.num_envs, device=self.env.device, dtype=torch.long)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.env.device, dtype=torch.long)
        return torch.as_tensor(list(env_ids), device=self.env.device, dtype=torch.long)


class LocomotionGait(CommandTermBase):
    """Stateful term that owns gait phase buffers and updates them each step."""

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)
        params = cfg.params or {}
        self.gait_period: float = float(params.get("gait_period", 1.0))
        self.gait_period_randomization_width: float = float(params.get("gait_period_randomization_width", 0.0))
        self.randomize_phase: bool = bool(params.get("randomize_phase", True))
        self.stand_phase_value: float = float(params.get("stand_phase_value", torch.pi))

        # Re-anchor the gait phase to the canonical gait-start phase [0, -pi] on every stand->walk
        # transition, matching what the DEPLOY policy does (UnifiedLocoKickPolicy._update_phase /
        # holosoma_inference's update_phase_time both force the phase to the [0, pi] gait-start on
        # the first walking step out of a stand). Opt-in; default off leaves the baseline exactly
        # as validated. Why it's needed: without it, training's phase is a free-running clock that,
        # coming out of a stand, resumes at whatever phase the clock happens to be at -- so the
        # policy only ever practices gait INITIATION (standing body, first step) at arbitrary
        # phases, never specifically at the [0, pi] the deployed policy ALWAYS starts from. That
        # train/deploy gap produced a consistent ~-3.8deg first-step yaw/lateral veer measured in
        # MuJoCo on the v5 checkpoint (same first swing foot every start, uncompensated). Note
        # [0, -pi] is what eval-mode already uses (see _initialize_indices) and is observationally
        # identical to deploy's [0, pi] (equal sin/cos), just kept in the wrapped [-pi, pi) range.
        self.reanchor_phase_on_gait_start: bool = bool(params.get("reanchor_phase_on_gait_start", False))

        self.phase_offset: torch.Tensor | None = None
        self.phase: torch.Tensor | None = None
        self.gait_freq: torch.Tensor | None = None
        self.phase_dt: torch.Tensor | None = None
        self._was_standing: torch.Tensor | None = None
        # Consecutive steps for which this env has been under a (near-)zero velocity command.
        # Resets to 0 the moment a nonzero command arrives. Read by the standing-shaping reward
        # family (see _standing_gate in managers/reward/terms/locomotion.py) to fade those
        # penalties in over a grace window after a walk->stop, giving the policy room to take the
        # staggered catch step that arrests momentum before it is asked to hold a square stance.
        # Deliberately a function of the COMMAND history only -- never of the robot's own state --
        # so the policy cannot manipulate it. (An earlier attempt gated on the robot's ACTUAL
        # speed; because the policy controls that input, it learned to simply never stop, drifting
        # 12.6m in 20s under a zero command to hold the standing penalties suppressed at ~3%.)
        self.stand_steps: torch.Tensor | None = None
        self.mean_gait_freq: float = 1.0 / self.gait_period

    def setup(self) -> None:
        env = self.env
        device = env.device
        num_envs = env.num_envs

        self.phase_offset = torch.zeros((num_envs, 2), dtype=torch.float32, device=device)
        self.phase = torch.zeros((num_envs, 2), dtype=torch.float32, device=device)
        self.gait_freq = torch.zeros((num_envs, 1), dtype=torch.float32, device=device)
        self.phase_dt = torch.zeros((num_envs, 1), dtype=torch.float32, device=device)
        self.stand_steps = torch.zeros(num_envs, dtype=torch.float32, device=device)
        if self.reanchor_phase_on_gait_start:
            self._was_standing = torch.zeros(num_envs, dtype=torch.bool, device=device)

        self._initialize_indices(None, evaluating=env.is_evaluating)

    def reset(self, env_ids: torch.Tensor | None) -> None:
        self._initialize_indices(env_ids, evaluating=self.env.is_evaluating)
        idx = self._ensure_index_tensor(env_ids)
        # Clear the stand-tracking latch for reset envs: a fresh episode's step 1 has no prior
        # command to have transitioned FROM, so it must never register as a stand->walk (step 1
        # then repopulates the latch from the real command; see step()).
        if self._was_standing is not None and idx.numel() > 0:
            self._was_standing[idx] = False
        # A fresh episode that starts standing is NOT a walk->stop transition -- there is no
        # momentum to arrest -- so it gets no grace window. step() repopulates this from the real
        # command; leaving it at 0 here would hand every episode start a free grace period.
        if self.stand_steps is not None and idx.numel() > 0:
            self.stand_steps[idx] = 0.0

    def step(self) -> None:
        if self.phase is None or self.phase_offset is None or self.phase_dt is None:
            return

        env = self.env
        command_tensor = getattr(self.manager, "commands", None) if hasattr(self, "manager") else None

        stand_mask = None
        if command_tensor is not None:
            stand_mask = torch.logical_and(
                torch.linalg.norm(command_tensor[:, :2], dim=1) < 0.01,
                torch.abs(command_tensor[:, 2]) < 0.01,
            )
            # Re-anchor phase to [0, -pi] at the exact stand->walk transition, BEFORE recomputing
            # phase this step, so the transitioning envs' first walking step lands on the canonical
            # gait-start (matching deploy). Re-anchoring only shifts the phase origin (phase_offset)
            # -- gait_freq / phase_dt are untouched, so gait speed is preserved.
            if self.reanchor_phase_on_gait_start and self._was_standing is not None:
                transition = torch.logical_and(self._was_standing, ~stand_mask)
                if transition.any():
                    t = env.episode_length_buf[transition].unsqueeze(1).float()
                    pdt = self.phase_dt[transition]
                    self.phase_offset[transition, 0] = (-t * pdt).squeeze(1)
                    self.phase_offset[transition, 1] = (-torch.pi - t * pdt).squeeze(1)
                self._was_standing = stand_mask.clone()

            # Consecutive-steps-under-a-zero-command counter, for the standing rewards' grace
            # window. Depends only on the command, never on the robot's state (see __init__).
            if self.stand_steps is not None:
                self.stand_steps = torch.where(
                    stand_mask, self.stand_steps + 1.0, torch.zeros_like(self.stand_steps)
                )

        phase_tp1 = env.episode_length_buf.unsqueeze(1) * self.phase_dt + self.phase_offset
        self.phase.copy_(torch.fmod(phase_tp1 + torch.pi, 2 * torch.pi) - torch.pi)

        if stand_mask is not None and stand_mask.any():
            self.phase[stand_mask] = torch.full(
                (int(stand_mask.sum().item()), 2), self.stand_phase_value, device=env.device
            )

    def set_eval_mode(self, evaluating: bool) -> None:
        self._initialize_indices(None, evaluating=evaluating)

    def resample_frequency(self, env_ids: torch.Tensor) -> None:
        if self.gait_freq is None or self.phase_dt is None:
            return

        idx = self._ensure_index_tensor(env_ids)
        if idx.numel() == 0:
            return

        if self.env.is_evaluating or self.gait_period_randomization_width <= 0.0:
            self.gait_freq[idx] = self.mean_gait_freq
        else:
            low = self.mean_gait_freq - self.gait_period_randomization_width
            high = self.mean_gait_freq + self.gait_period_randomization_width
            self.gait_freq[idx] = torch_rand_float(low, high, (idx.shape[0], 1), device=self.env.device)

        self.phase_dt[idx] = 2 * torch.pi * self.env.dt * self.gait_freq[idx]

    # ------------------------------------------------------------------ #
    # Internal utilities
    # ------------------------------------------------------------------ #

    def _initialize_indices(self, env_ids: torch.Tensor | None, *, evaluating: bool) -> None:
        if self.phase_offset is None or self.phase is None or self.gait_freq is None or self.phase_dt is None:
            return

        idx = self._ensure_index_tensor(env_ids)
        if idx.numel() == 0:
            return

        if evaluating:
            self.phase_offset[idx, 0] = 0.0
            self.phase_offset[idx, 1] = -torch.pi
        elif self.randomize_phase:
            self.phase_offset[idx, 0] = torch_rand_float(
                -torch.pi, torch.pi, (idx.shape[0], 1), device=self.env.device
            ).squeeze(1)
            self.phase_offset[idx, 1] = torch.fmod(self.phase_offset[idx, 0] + 2 * torch.pi, 2 * torch.pi) - torch.pi
        else:
            self.phase_offset[idx] = 0.0

        self.phase[idx] = self.phase_offset[idx]
        self.resample_frequency(idx)

    def _ensure_index_tensor(self, env_ids: torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.env.num_envs, device=self.env.device, dtype=torch.long)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.env.device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=self.env.device, dtype=torch.long)
