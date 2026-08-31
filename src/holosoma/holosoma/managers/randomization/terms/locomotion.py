"""Randomization terms for locomotion environments."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import torch
from loguru import logger

from holosoma.config_types.simulator import MujocoBackend
from holosoma.managers.action.terms.joint_control import JointPositionActionTerm
from holosoma.managers.randomization.base import RandomizationTermBase
from holosoma.managers.randomization.exceptions import RandomizerNotSupportedError
from holosoma.simulator import mujoco_required_field
from holosoma.simulator.shared.field_decorators import MUJOCO_FIELD_ATTR
from holosoma.utils.torch_utils import torch_rand_float

if TYPE_CHECKING:
    from isaaclab.managers import SceneEntityCfg

    from holosoma.simulator.isaacsim.isaacsim import IsaacSim


def _ensure_env_ids_tensor(env: Any, env_ids: torch.Tensor | Sequence[int] | None) -> torch.Tensor:
    """Convert environment indices to a tensor on the correct device."""
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=env.device, dtype=torch.long)
    return torch.as_tensor(list(env_ids), device=env.device, dtype=torch.long)


def _get_joint_action_term(env: Any) -> JointPositionActionTerm | None:
    """Return the joint-position action term registered with the action manager."""
    action_manager = getattr(env, "action_manager", None)
    if action_manager is None:
        return None

    get_term = getattr(action_manager, "get_term", None)
    if callable(get_term):
        term = get_term("joint_control")
        if isinstance(term, JointPositionActionTerm):
            return term

    iter_terms = getattr(action_manager, "iter_terms", None)
    if callable(iter_terms):
        for _, term in iter_terms():
            if isinstance(term, JointPositionActionTerm):
                return term

    return None


def _isaacsim_randomize_rigid_body_mass(
    simulator: IsaacSim,
    env_ids_cpu: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    mass_distribution_params: tuple[float, float],
    operation: str,
):
    try:
        from isaaclab.envs import mdp
        from isaaclab.managers import EventTermCfg
    except ImportError as exc:  # pragma: no cover - defensive
        raise RuntimeError("IsaacSim mass randomization requires isaaclab.") from exc
    func = mdp.randomize_rigid_body_mass(
        EventTermCfg(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "env_ids": env_ids_cpu,
                "asset_cfg": asset_cfg,
                "mass_distribution_params": mass_distribution_params,
                "operation": operation,
            },
        ),
        env=simulator,
    )
    func(
        simulator,
        env_ids_cpu,
        asset_cfg=asset_cfg,
        mass_distribution_params=mass_distribution_params,
        operation=operation,
    )


def _isaacsim_randomize_rigid_body_material(
    simulator: IsaacSim,
    env_ids_cpu: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    static_friction_range: tuple[float, float],
    dynamic_friction_range: tuple[float, float],
    restitution_range: tuple[float, float],
    num_buckets: int,
):
    try:
        from isaaclab.envs import mdp
        from isaaclab.managers import EventTermCfg
    except ImportError as exc:  # pragma: no cover - defensive
        raise RuntimeError("IsaacSim material randomization requires isaaclab.") from exc
    func = mdp.randomize_rigid_body_material(
        EventTermCfg(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "env_ids": env_ids_cpu,
                "asset_cfg": asset_cfg,
                "static_friction_range": static_friction_range,
                "dynamic_friction_range": dynamic_friction_range,
                "restitution_range": restitution_range,
                "num_buckets": num_buckets,
            },
        ),
        simulator,
    )
    func(
        simulator,
        env_ids_cpu,
        asset_cfg=asset_cfg,
        static_friction_range=static_friction_range,
        dynamic_friction_range=dynamic_friction_range,
        restitution_range=restitution_range,
        num_buckets=num_buckets,
    )


def _isaacsim_randomize_joint_parameters(
    simulator: IsaacSim,
    env_ids_cpu: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    armature_distribution_params: tuple[float, float] | None,
    friction_distribution_params: tuple[float, float] | None,
    operation: str,
):
    try:
        from isaaclab.envs import mdp
        from isaaclab.managers import EventTermCfg
    except ImportError as exc:  # pragma: no cover - defensive
        raise RuntimeError("IsaacSim joint-parameter randomization requires isaaclab.") from exc
    params = {
        "env_ids": env_ids_cpu,
        "asset_cfg": asset_cfg,
        "armature_distribution_params": armature_distribution_params,
        "friction_distribution_params": friction_distribution_params,
        "operation": operation,
    }
    func = mdp.randomize_joint_parameters(
        EventTermCfg(func=mdp.randomize_joint_parameters, mode="startup", params=params),
        env=simulator,
    )
    func(
        simulator,
        env_ids_cpu,
        asset_cfg=asset_cfg,
        armature_distribution_params=armature_distribution_params,
        friction_distribution_params=friction_distribution_params,
        operation=operation,
    )


class PushRandomizerState(RandomizationTermBase):
    """Stateful randomizer that owns push scheduling buffers and counters."""

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)
        params = cfg.params or {}
        interval = params.get("push_interval_s", [5, 16])
        self.push_interval_range: Sequence[float] = [float(interval[0]), float(interval[1])]
        vector_max = params.get("max_push_vel")
        if vector_max is None:
            raise ValueError("PushRandomizerState requires `max_push_vel` to be specified.")
        self._max_push_vel_tensor = torch.empty(0, dtype=torch.float32, device=env.device)
        self._set_max_push_tensor(vector_max)
        self.enabled: bool = bool(params.get("enabled", True))
        logger.info(
            f"[Randomization] PushRandomizerState initialized (enabled={self.enabled}, \
                max_push_vel={self._max_push_vel_tensor.tolist()}, \
                interval_s={self.push_interval_range})",
        )

        self.push_interval_s: torch.Tensor | None = None
        self.push_robot_counter: torch.Tensor | None = None
        self.push_robot_plot_counter: torch.Tensor | None = None

    def setup(self) -> None:
        env = self.env
        device = env.device
        num_envs = env.num_envs

        self.push_interval_s = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.push_robot_counter = torch.zeros(num_envs, dtype=torch.int, device=device)
        self.push_robot_plot_counter = torch.zeros(num_envs, dtype=torch.int, device=device)

        all_ids = torch.arange(num_envs, device=device, dtype=torch.long)
        self._resample_intervals(all_ids)

    def reset(self, env_ids: torch.Tensor | None) -> None:
        if self.push_robot_counter is None or self.push_robot_plot_counter is None:
            return
        idx = self._ensure_indices(env_ids)
        if idx.numel() == 0:
            return
        self.push_robot_counter[idx] = 0
        self.push_robot_plot_counter[idx] = 0

    def step(self) -> None:
        if not self.enabled:
            return
        if self.push_robot_counter is None or self.push_robot_plot_counter is None:
            return
        self.push_robot_counter += 1
        self.push_robot_plot_counter += 1

    # ------------------------------------------------------------------ #
    # Public helpers for other randomization hooks
    # ------------------------------------------------------------------ #

    def configure(
        self,
        *,
        enabled: bool | None = None,
        push_interval_s: Sequence[float] | None = None,
        max_push_vel: Sequence[float] | None = None,
    ) -> None:
        if enabled is not None:
            self.enabled = bool(enabled)
        if push_interval_s is not None:
            self.push_interval_range = [float(push_interval_s[0]), float(push_interval_s[1])]
        if max_push_vel is not None:
            self._set_max_push_tensor(max_push_vel)

    def resample(self, env_ids: torch.Tensor | None = None) -> None:
        idx = self._ensure_indices(env_ids)
        if idx.numel() == 0:
            return
        self._resample_intervals(idx)

    def due_envs(self, dt: float) -> torch.Tensor:
        if not self.enabled:
            return torch.empty(0, device=self.env.device, dtype=torch.long)
        if self.push_interval_s is None or self.push_robot_counter is None:
            return torch.empty(0, device=self.env.device, dtype=torch.long)
        interval_steps = (self.push_interval_s / dt).to(torch.int)
        return (self.push_robot_counter == interval_steps).nonzero(as_tuple=False).flatten()

    def zero_counters(self, env_ids: torch.Tensor) -> None:
        if self.push_robot_counter is None or self.push_robot_plot_counter is None:
            return
        self.push_robot_counter[env_ids] = 0
        self.push_robot_plot_counter[env_ids] = 0

    @property
    def max_push_vel(self) -> torch.Tensor:
        return self._max_push_vel_tensor

    def _ensure_indices(self, env_ids: torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.env.num_envs, device=self.env.device, dtype=torch.long)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.env.device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=self.env.device, dtype=torch.long)

    def _resample_intervals(self, env_ids: torch.Tensor) -> None:
        if self.push_interval_s is None:
            return
        low, high = self.push_interval_range
        low_i = max(1, int(low))
        high_i = max(low_i + 1, int(high))
        samples = torch_rand_float(low_i, high_i, (env_ids.shape[0], 1), device=self.env.device).squeeze(1)
        self.push_interval_s[env_ids] = samples

    def _set_max_push_tensor(self, values: Sequence[float]) -> None:
        tensor = torch.as_tensor(values, dtype=torch.float32, device=self.env.device).flatten()
        if tensor.numel() == 0:
            raise ValueError("max_push_vel must contain at least one value.")
        self._max_push_vel_tensor = tensor.clone()


# Default target bodies for BodyPushRandomizerState. Chosen as the contacts a walking humanoid
# actually makes with unmodelled scenery: knees/shins into low obstacles, elbows and hands
# brushing walls and doorframes, torso/pelvis into people and furniture. Feet are deliberately
# EXCLUDED -- they are already in near-continuous commanded contact with the ground, so an
# external force there is indistinguishable from a ground-reaction modelling error rather than a
# collision. Names verified against data/robots/g1/g1_29dof.urdf, not assumed.
DEFAULT_BODY_PUSH_BODIES: tuple[str, ...] = (
    "left_knee_link",
    "right_knee_link",
    "left_elbow_link",
    "right_elbow_link",
    "torso_link",
    "pelvis",
)


class BodyPushRandomizerState(RandomizationTermBase):
    """Sustained, body-targeted external-force disturbances -- a collision model, not a push model.

    Why this exists (2026-08-27). ``PushRandomizerState``/``_push_robots`` is the only disturbance
    this project has ever trained against, and it is a one-tick additive velocity impulse written
    straight into ``robot_root_states[:, 7:13]``. A real collision differs from that in three ways:

    1. **Location.** The impulse always lands on the ROOT. A real contact lands on a shin, an
       elbow, a hand -- which induces joint torques (ankle, knee, shoulder) that a root-frame
       velocity change never produces at all.
    2. **Duration.** The impulse is instantaneous. A real contact sustains force for roughly
       50-200 ms, and the recovery for a sustained push is a different behavior from catching a
       step change in velocity.
    3. **Constraint.** After a velocity impulse the robot is free to move anywhere. A real obstacle
       is still physically there and BLOCKS the recovery motion.

    This term addresses (1) and (2). **It does NOT address (3)** -- a force is not a constraint,
    and nothing here stops the robot moving straight through the notional obstacle. Genuine
    blocking needs collision geometry in the scene, which is a scene-level change of a different
    size and is deliberately left out of scope. Do not describe this mechanism as covering (3).

    Ships DISABLED (``enabled=False``), a verified no-op: with no config change every run is
    bit-identical to before this class existed. It is additive to the existing root push rather
    than a replacement -- both can be active, and disabling the old one is a separate deliberate
    choice (``push_enabled`` in the task-config yaml), so that enabling this does not silently
    remove the disturbance coverage all 21 prior runs trained under.
    """

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)
        params = cfg.params or {}

        self.enabled: bool = bool(params.get("enabled", False))
        interval = params.get("interval_s", [4.0, 8.0])
        self.interval_range: list[float] = [float(interval[0]), float(interval[1])]
        force_range = params.get("force_range", [20.0, 80.0])
        self.force_range: list[float] = [float(force_range[0]), float(force_range[1])]
        duration = params.get("duration_s", [0.05, 0.20])
        self.duration_range: list[float] = [float(duration[0]), float(duration[1])]
        self.vertical_fraction: float = float(params.get("vertical_fraction", 0.2))
        body_names = params.get("body_names") or DEFAULT_BODY_PUSH_BODIES
        self.body_names: list[str] = [str(n) for n in body_names]

        self._validate()

        # Allocated in setup()
        self.counter: torch.Tensor | None = None
        self.interval_steps: torch.Tensor | None = None
        self.remaining_steps: torch.Tensor | None = None
        self.force_buf: torch.Tensor | None = None
        self.body_indices: torch.Tensor | None = None
        # True while a non-zero force is standing in the simulator's persistent buffer, so the
        # active->idle transition writes exactly one clearing frame and idle steps then cost
        # nothing. Without this the term would either leak a force forever or pay a full
        # num_envs x num_bodies write on every step of a run that never fires.
        self._forces_written: bool = False

    def _validate(self) -> None:
        if self.interval_range[0] <= 0.0 or self.interval_range[1] < self.interval_range[0]:
            raise ValueError(f"body push interval_s must be 0 < min <= max, got {self.interval_range}")
        if self.force_range[0] < 0.0 or self.force_range[1] < self.force_range[0]:
            raise ValueError(f"body push force_range must be 0 <= min <= max, got {self.force_range}")
        if self.duration_range[0] <= 0.0 or self.duration_range[1] < self.duration_range[0]:
            raise ValueError(f"body push duration_s must be 0 < min <= max, got {self.duration_range}")
        if not 0.0 <= self.vertical_fraction <= 1.0:
            raise ValueError(f"body push vertical_fraction must be in [0, 1], got {self.vertical_fraction}")
        if not self.body_names:
            raise ValueError("body push body_names must be non-empty")

    def setup(self) -> None:
        env = self.env
        device = env.device
        num_envs = env.num_envs

        simulator = env.simulator
        # NOT simulator.num_bodies: that attribute is backend-inconsistent (on MuJoCo it is
        # root_model.nbody, which additionally counts the `world` body and the ball -- see
        # mujoco.py's own comment on this exact trap on `_holosoma_body_to_mujoco_id`). body_names
        # is the one handle both set_external_body_forces implementations actually index against
        # (holosoma order, robot-only), so it is the only correct source for this shape here.
        num_bodies = len(simulator.body_names)

        self.counter = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self.interval_steps = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self.remaining_steps = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self.force_buf = torch.zeros(num_envs, int(num_bodies), 3, dtype=torch.float32, device=device)

        resolved: list[int] = []
        missing: list[str] = []
        for name in self.body_names:
            idx = simulator.find_rigid_body_indice(name)
            if idx is None:
                missing.append(name)
            elif isinstance(idx, list):
                resolved.extend(int(i) for i in idx)
            else:
                resolved.append(int(idx))
        if missing:
            # Loud, not silent: a typo'd link name would otherwise quietly shrink the target set,
            # and a disturbance randomizer that pushes fewer bodies than configured is exactly the
            # kind of thing that surfaces as an unexplained robustness result months later.
            raise ValueError(
                f"body push body_names not found on this robot: {missing}. "
                f"Available example names come from robot_config.body_names."
            )
        self.body_indices = torch.as_tensor(resolved, dtype=torch.long, device=device)

        self._resample_intervals(torch.arange(num_envs, device=device, dtype=torch.long))

    def reset(self, env_ids: torch.Tensor | None) -> None:
        if self.counter is None or self.force_buf is None or self.remaining_steps is None:
            return
        idx = self._ensure_indices(env_ids)
        if idx.numel() == 0:
            return
        self.counter[idx] = 0
        self.remaining_steps[idx] = 0
        self.force_buf[idx] = 0.0
        self._resample_intervals(idx)

    def step(self) -> None:
        if not self.enabled or self.counter is None:
            return
        self.counter += 1

    # ------------------------------------------------------------------ #

    def _ensure_indices(self, env_ids: torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.env.num_envs, device=self.env.device, dtype=torch.long)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.env.device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=self.env.device, dtype=torch.long)

    def _resample_intervals(self, env_ids: torch.Tensor) -> None:
        if self.interval_steps is None or env_ids.numel() == 0:
            return
        dt = float(self.env.dt)
        low = max(1, int(round(self.interval_range[0] / dt)))
        high = max(low + 1, int(round(self.interval_range[1] / dt)))
        samples = torch.randint(low, high, (env_ids.shape[0],), device=self.env.device, dtype=torch.int32)
        self.interval_steps[env_ids] = samples

    def _sample_forces(self, env_ids: torch.Tensor) -> None:
        """Assign each due env a fresh (body, direction, magnitude, duration)."""
        assert self.force_buf is not None and self.remaining_steps is not None
        assert self.body_indices is not None
        n = env_ids.shape[0]
        device = self.env.device

        # Direction: uniform in azimuth, with a bounded vertical component. A collision with
        # scenery is overwhelmingly horizontal, hence vertical_fraction defaulting well below 1.
        azimuth = torch.rand(n, device=device) * (2.0 * torch.pi)
        z = (torch.rand(n, device=device) * 2.0 - 1.0) * self.vertical_fraction
        horizontal = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
        direction = torch.stack([horizontal * torch.cos(azimuth), horizontal * torch.sin(azimuth), z], dim=1)

        magnitude = torch_rand_float(
            self.force_range[0], self.force_range[1], (n, 1), device=str(device)
        )
        force = direction * magnitude

        body_slot = torch.randint(0, self.body_indices.numel(), (n,), device=device)
        body_idx = self.body_indices[body_slot]

        # Clear whatever this env had before, then write the new single-body force.
        self.force_buf[env_ids] = 0.0
        self.force_buf[env_ids, body_idx] = force

        dt = float(self.env.dt)
        low = max(1, int(round(self.duration_range[0] / dt)))
        high = max(low + 1, int(round(self.duration_range[1] / dt)))
        self.remaining_steps[env_ids] = torch.randint(
            low, high, (n,), device=device, dtype=torch.int32
        )


class ActuatorRandomizerState(RandomizationTermBase):
    """Stateful actuator randomizer managing PD gain and RFI scales."""

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)
        params = cfg.params or {}

        kp_range = params.get("kp_range", [1.0, 1.0])
        kd_range = params.get("kd_range", [1.0, 1.0])
        rfi_lim_range = params.get("rfi_lim_range", [1.0, 1.0])

        self.enable_pd_gain = bool(params.get("enable_pd_gain", True))
        self.enable_rfi_lim = bool(params.get("enable_rfi_lim", False))

        self.kp_range: Sequence[float] = [float(kp_range[0]), float(kp_range[1])]
        self.kd_range: Sequence[float] = [float(kd_range[0]), float(kd_range[1])]
        self.rfi_lim_range: Sequence[float] = [float(rfi_lim_range[0]), float(rfi_lim_range[1])]

        self.rfi_lim = float(params.get("rfi_lim", 0.1))

        self.kp_scale: torch.Tensor | None = None
        self.kd_scale: torch.Tensor | None = None
        self.rfi_lim_scale: torch.Tensor | None = None

    def setup(self) -> None:
        env = self.env
        device = env.device
        num_envs = env.num_envs
        num_dof = env.num_dof

        self.kp_scale = torch.ones(num_envs, num_dof, dtype=torch.float32, device=device)
        self.kd_scale = torch.ones(num_envs, num_dof, dtype=torch.float32, device=device)
        self.rfi_lim_scale = torch.ones(num_envs, num_dof, dtype=torch.float32, device=device)

        term = _get_joint_action_term(env)
        if term is not None:
            term.attach_actuator_scales(self.kp_scale, self.kd_scale, self.rfi_lim_scale)
        else:
            logger.debug(
                "JointPositionActionTerm not ready during ActuatorRandomizerState.setup(); "
                "the term will attach shared actuator scales once its setup() runs."
            )

    def reset(self, env_ids: torch.Tensor | None) -> None:
        if self.kp_scale is None or self.kd_scale is None or self.rfi_lim_scale is None:
            raise RuntimeError("ActuatorRandomizerState.setup() must be called before reset().")

        idx = _ensure_env_ids_tensor(self.env, env_ids)
        if idx.numel() == 0:
            return

        device = self.env.device

        if self.enable_pd_gain:
            self.kp_scale[idx] = torch_rand_float(
                self.kp_range[0], self.kp_range[1], (idx.shape[0], self.env.num_dof), device=device
            )
            self.kd_scale[idx] = torch_rand_float(
                self.kd_range[0], self.kd_range[1], (idx.shape[0], self.env.num_dof), device=device
            )
        else:
            self.kp_scale[idx] = 1.0
            self.kd_scale[idx] = 1.0

        if self.enable_rfi_lim:
            self.rfi_lim_scale[idx] = torch_rand_float(
                self.rfi_lim_range[0], self.rfi_lim_range[1], (idx.shape[0], self.env.num_dof), device=device
            )
        else:
            self.rfi_lim_scale[idx] = 1.0

    def step(self) -> None:
        """No per-step behaviour required."""

    @property
    def kp_scale_tensor(self) -> torch.Tensor:
        if self.kp_scale is None:
            raise RuntimeError("ActuatorRandomizerState.setup() has not been called yet.")
        return self.kp_scale

    @property
    def kd_scale_tensor(self) -> torch.Tensor:
        if self.kd_scale is None:
            raise RuntimeError("ActuatorRandomizerState.setup() has not been called yet.")
        return self.kd_scale

    @property
    def rfi_lim_scale_tensor(self) -> torch.Tensor:
        if self.rfi_lim_scale is None:
            raise RuntimeError("ActuatorRandomizerState.setup() has not been called yet.")
        return self.rfi_lim_scale


def setup_action_delay_buffers(env, *, ctrl_delay_step_range: Sequence[int], enabled: bool = True, **_) -> None:
    """Initialize action delay index buffer during setup.

    Note: The action_queue itself is managed by the action manager.
    This only sets up the delay index that determines which queued action to use.
    """
    env._randomize_ctrl_delay = bool(enabled)
    env._ctrl_delay_step_range = list(ctrl_delay_step_range)

    if not enabled:
        return

    # Initialize action delay indices (determines which action from the queue to use)
    env.action_delay_idx = torch.randint(
        ctrl_delay_step_range[0],
        ctrl_delay_step_range[1] + 1,
        (env.num_envs,),
        device=env.device,
        requires_grad=False,
    )


def setup_torque_rfi(env, *, enabled: bool = False, rfi_lim: float = 0.1, **_) -> None:
    """Configure torque RFI at startup."""
    term = _get_joint_action_term(env)
    env._pending_torque_rfi = (bool(enabled), float(rfi_lim))
    if term is None:
        return
    term.configure_torque_rfi(enabled=env._pending_torque_rfi[0], rfi_lim=env._pending_torque_rfi[1])


def setup_dof_pos_bias(env, *, dof_pos_bias_range: Sequence[float], enabled: bool = False, **_) -> None:
    """Apply startup DOF position bias randomization."""
    env._randomize_dof_pos_bias = bool(enabled)
    env._dof_pos_bias_range = list(dof_pos_bias_range)

    if not enabled:
        return

    default_dof_pos_bias = torch_rand_float(
        dof_pos_bias_range[0],
        dof_pos_bias_range[1],
        (env.num_envs, env.num_dof),
        device=env.device,
    )
    env.default_dof_pos = env.default_dof_pos_base + default_dof_pos_bias


def randomize_push_schedule(
    env,
    env_ids,
    *,
    push_interval_s: Sequence[float] | None = None,
    enabled: bool | None = None,
    max_push_vel: Sequence[float] | None = None,
    **_,
) -> None:
    """Resample push intervals for selected environments."""
    state = env.randomization_manager.get_state("push_randomizer_state")
    if state is None:
        raise AttributeError("PushRandomizerState is not registered with the randomization manager.")

    state.configure(enabled=enabled, push_interval_s=push_interval_s, max_push_vel=max_push_vel)
    env._randomize_push_robots = state.enabled
    env._max_push_vel = state.max_push_vel.clone()

    if not state.enabled:
        return

    idx = _ensure_env_ids_tensor(env, env_ids)
    if idx.numel() == 0:
        return

    state.zero_counters(idx)
    state.resample(idx)


def randomize_pd_gains(
    env, env_ids, *, kp_range: Sequence[float], kd_range: Sequence[float], enabled: bool = True, **_
):
    """Randomize proportional and derivative gain scales."""
    state = env.randomization_manager.get_state("actuator_randomizer_state")
    term = _get_joint_action_term(env)
    if state is None:
        if term is None:
            logger.warning("JointPositionActionTerm not found; PD gain randomization skipped.")
            return

        idx = _ensure_env_ids_tensor(env, env_ids)
        if idx.numel() == 0:
            return

        if not enabled:
            kp_scale, kd_scale = term.get_pd_scale_tensors()
            term.update_pd_scales(idx, torch.ones_like(kp_scale[idx]), torch.ones_like(kd_scale[idx]))
            return

        kp_samples = torch_rand_float(kp_range[0], kp_range[1], (idx.shape[0], env.num_dof), device=env.device)
        kd_samples = torch_rand_float(kd_range[0], kd_range[1], (idx.shape[0], env.num_dof), device=env.device)
        term.update_pd_scales(idx, kp_samples, kd_samples)
        return

    state.enable_pd_gain = bool(enabled)
    state.kp_range = [float(kp_range[0]), float(kp_range[1])]
    state.kd_range = [float(kd_range[0]), float(kd_range[1])]
    state.reset(env_ids)


def randomize_rfi_limits(
    env,
    env_ids,
    *,
    rfi_lim_range: Sequence[float],
    enabled: bool = True,
    **_,
) -> None:
    """Randomize residual force injection limits."""
    state = env.randomization_manager.get_state("actuator_randomizer_state")
    term = _get_joint_action_term(env)
    if state is None:
        if term is None:
            logger.warning("JointPositionActionTerm not found; RFI randomization skipped.")
            return

        idx = _ensure_env_ids_tensor(env, env_ids)
        if idx.numel() == 0:
            return

        if not enabled:
            term.update_rfi_scales(idx, torch.ones_like(term.get_rfi_scale_tensor()[idx]))
            return

        rfi_samples = torch_rand_float(
            rfi_lim_range[0], rfi_lim_range[1], (idx.shape[0], env.num_dof), device=env.device
        )
        term.update_rfi_scales(idx, rfi_samples)
        return

    state.enable_rfi_lim = bool(enabled)
    state.rfi_lim_range = [float(rfi_lim_range[0]), float(rfi_lim_range[1])]
    state.reset(env_ids)


def randomize_action_delay(
    env,
    env_ids,
    *,
    ctrl_delay_step_range: Sequence[int] | None = None,
    enabled: bool | None = None,
    **_,
) -> None:
    """Randomize control delay indices.

    If ``ctrl_delay_step_range``/``enabled`` are omitted the values captured during
    ``setup_action_delay_buffers`` are reused.
    """
    if enabled is not None:
        env._randomize_ctrl_delay = bool(enabled)
    elif not hasattr(env, "_randomize_ctrl_delay"):
        raise AttributeError(
            "randomize_action_delay() requires setup_action_delay_buffers to run before it can infer 'enabled'."
        )

    if ctrl_delay_step_range is not None:
        env._ctrl_delay_step_range = list(ctrl_delay_step_range)
    elif not hasattr(env, "_ctrl_delay_step_range"):
        raise AttributeError(
            "randomize_action_delay() requires setup_action_delay_buffers \
                to run before it can infer ctrl_delay_step_range."
        )

    if not env._randomize_ctrl_delay:
        return

    idx = _ensure_env_ids_tensor(env, env_ids)
    if idx.numel() == 0:
        return

    # Reset action queue in the action manager
    if hasattr(env.action_manager, "action_queue"):
        env.action_manager.action_queue[idx] *= 0.0

    delay_low = int(env._ctrl_delay_step_range[0])
    delay_high = int(env._ctrl_delay_step_range[1])
    if delay_high < delay_low:
        raise ValueError("ctrl_delay_step_range upper bound must be >= lower bound.")

    # Randomize delay indices
    env.action_delay_idx[idx] = torch.randint(
        delay_low,
        delay_high + 1,
        (idx.shape[0],),
        device=env.device,
        requires_grad=False,
    )


def randomize_dof_state(
    env,
    env_ids,
    *,
    joint_pos_scale_range: Sequence[float],
    joint_pos_bias_range: Sequence[float],
    joint_vel_range: Sequence[float],
    randomize_dof_pos_bias: bool = False,
    **_,
) -> None:
    """Randomize DOF positions and velocities."""
    env._joint_pos_scale_range = list(joint_pos_scale_range)
    env._joint_pos_bias_range = list(joint_pos_bias_range)
    env._joint_vel_range = list(joint_vel_range)
    env._randomize_dof_pos_bias = bool(randomize_dof_pos_bias)

    idx = _ensure_env_ids_tensor(env, env_ids)
    if idx.numel() == 0:
        return

    scale_factor = torch_rand_float(
        joint_pos_scale_range[0],
        joint_pos_scale_range[1],
        (idx.shape[0], env.num_dof),
        device=env.device,
    )
    if randomize_dof_pos_bias:
        bias_offset = torch_rand_float(
            joint_pos_bias_range[0],
            joint_pos_bias_range[1],
            (idx.shape[0], env.num_dof),
            device=env.device,
        )
    else:
        bias_offset = torch.zeros((idx.shape[0], env.num_dof), device=env.device)

    env.simulator.dof_pos[idx] = env.default_dof_pos[idx] * scale_factor + bias_offset
    env.simulator.dof_vel[idx] = torch_rand_float(
        joint_vel_range[0],
        joint_vel_range[1],
        (idx.shape[0], env.num_dof),
        device=env.device,
    )


@mujoco_required_field("body_ipos")
def randomize_base_com_startup(
    env,
    env_ids: Sequence[int] | torch.Tensor | None = None,
    *,
    base_com_range: dict[str, Sequence[float]],
    enabled: bool = True,
    **_,
) -> None:
    """Randomize base (torso) center of mass.

    Note: Uses ADDITION operation to offset CoM position (e.g., x: [-0.01, 0.01] m).
    """
    env._randomize_base_com = bool(enabled)
    env._base_com_range = base_com_range
    if not enabled:
        return

    logger.info(
        f"[Randomization] Base CoM: "
        f"x={base_com_range.get('x', [0, 0])}, "
        f"y={base_com_range.get('y', [0, 0])}, "
        f"z={base_com_range.get('z', [0, 0])} (operation=add)"
    )

    simulator = env.simulator

    idx = _ensure_env_ids_tensor(env, env_ids)
    if idx.numel() == 0:
        return

    if hasattr(simulator, "gym"):
        gym = simulator.gym
        torso_name = env.robot_config.torso_name
        if not hasattr(simulator, "_base_com_bias"):
            simulator._base_com_bias = torch.zeros(
                env.num_envs, 3, dtype=torch.float, device=env.device, requires_grad=False
            )

        for env_id in idx.tolist():
            env_ptr = simulator.envs[env_id]
            actor = simulator.robot_handles[env_id]
            body_props = gym.get_actor_rigid_body_properties(env_ptr, actor)
            body_index = gym.find_actor_rigid_body_handle(env_ptr, actor, torso_name)
            if body_index < 0:
                raise RuntimeError(f"Body '{torso_name}' not found when randomizing base COM.")

            xrange = base_com_range["x"]
            yrange = base_com_range["y"]
            zrange = base_com_range["z"]

            bias = torch.tensor(
                [
                    torch_rand_float(xrange[0], xrange[1], (1, 1), device=env.device).item(),
                    torch_rand_float(yrange[0], yrange[1], (1, 1), device=env.device).item(),
                    torch_rand_float(zrange[0], zrange[1], (1, 1), device=env.device).item(),
                ],
                dtype=torch.float,
                device=env.device,
            )
            simulator._base_com_bias[env_id] = bias
            body_props[body_index].com.x += bias[0].item()
            body_props[body_index].com.y += bias[1].item()
            body_props[body_index].com.z += bias[2].item()
            gym.set_actor_rigid_body_properties(env_ptr, actor, body_props, recomputeInertia=True)
    elif simulator.__class__.__name__ == "IsaacSim":
        try:
            from isaaclab.managers import SceneEntityCfg
        except ImportError as exc:  # pragma: no cover - dependency optional
            raise RuntimeError("IsaacSim base COM randomization requires isaaclab.") from exc
        from holosoma.simulator.isaacsim.events import randomize_body_com

        torso_name = env.robot_config.torso_name
        env_ids_cpu = idx.to(device="cpu", dtype=torch.long)
        if env_ids_cpu.numel() == 0:
            return

        low = torch.tensor(
            [base_com_range["x"][0], base_com_range["y"][0], base_com_range["z"][0]],
            dtype=torch.float,
            device="cpu",
        )
        high = torch.tensor(
            [base_com_range["x"][1], base_com_range["y"][1], base_com_range["z"][1]],
            dtype=torch.float,
            device="cpu",
        )
        asset_cfg = SceneEntityCfg("robot", body_names=[torso_name])
        asset_cfg.resolve(simulator.scene)  # Required to avoid applying randomization to all bodies
        randomize_body_com(
            simulator,
            env_ids_cpu,
            asset_cfg,
            (low, high),
            operation="add",
            distribution="uniform",
            num_envs=simulator.training_config.num_envs,
        )
    elif simulator.simulator_config.mujoco_backend == MujocoBackend.WARP:
        from holosoma.simulator.mujoco.backends.warp_randomization import randomize_field

        # convert xyz to 012
        base_com_range_remapped = {}
        for key, value in base_com_range.items():
            assert len(value) == 2, f"Range for '{key}' must have exactly 2 elements, got {len(value)}"
            base_com_range_remapped["xyz".index(key)] = (value[0], value[1])
        randomize_field(
            simulator,
            field=getattr(randomize_base_com_startup, MUJOCO_FIELD_ATTR),
            ranges=base_com_range_remapped,
            env_ids=idx,
            entity_names=[env.robot_config.torso_name],
            entity_type="body",
            operation="add",
            distribution="uniform",
        )

    else:  # pragma: no cover - defensive
        raise RandomizerNotSupportedError(
            f"Unsupported simulator type '{type(simulator).__name__}' for base COM randomization."
        )


@mujoco_required_field("body_mass")
def randomize_mass_startup(
    env,
    env_ids: Sequence[int] | torch.Tensor | None = None,
    *,
    enable_link_mass: bool = True,
    link_mass_range: Sequence[float] = (1.0, 1.0),
    enable_base_mass: bool = True,
    added_mass_range: Sequence[float] = (0.0, 0.0),
    enabled: bool = True,
    **_,
) -> None:
    """Randomize link and base masses at startup.

    Note: link_mass_range uses SCALING (e.g., 0.9-1.2 = 90-120% of original),
          added_mass_range uses ADDITION (e.g., -1.0 to 3.0 kg offset).
    """
    if not enabled:
        return

    logger.info(
        f"[Randomization] Mass: "
        f"link_mass={link_mass_range} (operation=scale, enabled={enable_link_mass}), "
        f"base_mass={added_mass_range} (operation=add, enabled={enable_base_mass})"
    )

    simulator = env.simulator
    idx = _ensure_env_ids_tensor(env, env_ids)
    if idx.numel() == 0:
        return

    env._randomize_link_mass = bool(enable_link_mass)
    env._randomize_base_mass = bool(enable_base_mass)

    if hasattr(simulator, "gym"):
        gym = simulator.gym
        body_names = list(env.robot_config.randomize_link_body_names or [])
        torso_name = env.robot_config.torso_name
        if idx.numel() > 0:
            sample_env = idx[0].item()
            sample_env_ptr = simulator.envs[sample_env]
            sample_actor = simulator.robot_handles[sample_env]
            sample_props = gym.get_actor_rigid_body_properties(sample_env_ptr, sample_actor)
            if enable_link_mass and body_names:
                link_masses = [
                    float(sample_props[simulator._body_list.index(name)].mass)
                    for name in body_names
                    if name in simulator._body_list
                ]
                if link_masses:
                    logger.debug(
                        "[randomize_mass_startup][IsaacGym] default link mass range: "
                        f"min={min(link_masses):.6f}, max={max(link_masses):.6f}"
                    )
            if enable_base_mass and torso_name in simulator._body_list:
                base_mass = float(sample_props[simulator._body_list.index(torso_name)].mass)
                logger.debug(f"[randomize_mass_startup][IsaacGym] default torso mass: {base_mass:.6f}")
        for env_id in idx.tolist():
            env_ptr = simulator.envs[env_id]
            actor = simulator.robot_handles[env_id]
            body_props = gym.get_actor_rigid_body_properties(env_ptr, actor)
            if enable_link_mass and body_names:
                for body_name in body_names:
                    if body_name not in simulator._body_list:
                        continue
                    body_index = simulator._body_list.index(body_name)
                    scale = np.random.uniform(link_mass_range[0], link_mass_range[1])
                    body_props[body_index].mass *= scale  # Scale operation: multiply by factor
            if enable_base_mass and torso_name in simulator._body_list:
                base_index = simulator._body_list.index(torso_name)
                delta = np.random.uniform(added_mass_range[0], added_mass_range[1])
                body_props[base_index].mass += delta  # Add operation: offset by delta
            gym.set_actor_rigid_body_properties(env_ptr, actor, body_props, recomputeInertia=True)
    elif simulator.__class__.__name__ == "IsaacSim":
        try:
            from isaaclab.managers import SceneEntityCfg
        except ImportError as exc:  # pragma: no cover - defensive
            raise RuntimeError("IsaacSim mass randomization requires isaaclab.") from exc

        env_ids_cpu = idx.to(device="cpu", dtype=torch.long)
        if env_ids_cpu.numel() == 0:
            return

        if enable_link_mass:
            asset_cfg = SceneEntityCfg("robot", body_names=env.robot_config.randomize_link_body_names)
            asset_cfg.resolve(simulator.scene)  # Required to avoid applying randomization to all bodies
            _isaacsim_randomize_rigid_body_mass(
                simulator,
                env_ids_cpu,
                asset_cfg,
                (link_mass_range[0], link_mass_range[1]),
                operation="scale",
            )

        if enable_base_mass:
            asset_cfg = SceneEntityCfg("robot", body_names=[env.robot_config.torso_name])
            asset_cfg.resolve(simulator.scene)  # Required to avoid applying randomization to all bodies
            _isaacsim_randomize_rigid_body_mass(
                simulator,
                env_ids_cpu,
                asset_cfg,
                (added_mass_range[0], added_mass_range[1]),
                operation="add",
            )
    elif simulator.simulator_config.mujoco_backend == MujocoBackend.WARP:
        from holosoma.simulator.mujoco.backends.warp_randomization import randomize_field

        # randomize over the range (scale and/or shift)
        if idx.numel() == 0:
            return

        if enable_link_mass:
            assert len(link_mass_range) == 2, (
                f"link_mass_range must have exactly 2 elements, got {len(link_mass_range)}"
            )
            randomize_field(
                simulator,
                field=getattr(randomize_mass_startup, MUJOCO_FIELD_ATTR),
                ranges=(link_mass_range[0], link_mass_range[1]),
                env_ids=idx,
                entity_names=env.robot_config.randomize_link_body_names,
                entity_type="body",
                operation="scale",
            )

        if enable_base_mass:
            assert len(added_mass_range) == 2, (
                f"added_mass_range must have exactly 2 elements, got {len(added_mass_range)}"
            )
            randomize_field(
                simulator,
                field=getattr(randomize_mass_startup, MUJOCO_FIELD_ATTR),
                ranges=(added_mass_range[0], added_mass_range[1]),
                env_ids=idx,
                entity_names=[env.robot_config.torso_name],
                entity_type="body",
                operation="add",
            )

    else:  # pragma: no cover - defensive
        raise RandomizerNotSupportedError(
            f"Mass randomization not supported for simulator type '{type(simulator).__name__}'."
        )


@mujoco_required_field("geom_friction")
def randomize_friction_startup(
    env,
    env_ids: Sequence[int] | torch.Tensor | None = None,
    *,
    friction_range: Sequence[float],
    enabled: bool = True,
    **_,
) -> None:
    """Randomize contact friction coefficients for robot rigid shapes.

    Note: Uses ABSOLUTE operation to set friction values (e.g., [0.5, 1.5]).
    """
    env._randomize_friction = bool(enabled)
    env._friction_range = list(friction_range)
    if not enabled:
        return

    logger.info(f"[Randomization] Friction: range={friction_range} (operation=abs)")

    idx = _ensure_env_ids_tensor(env, env_ids)
    if idx.numel() == 0:
        return

    simulator = env.simulator

    num_buckets = 64
    buckets = torch_rand_float(
        friction_range[0],
        friction_range[1],
        (num_buckets, 1),
        device="cpu",
    )

    idx_cpu = idx.to(device="cpu", dtype=torch.long)
    bucket_ids = torch.randint(0, num_buckets, (idx_cpu.shape[0],), device="cpu")
    friction_samples_cpu = buckets[bucket_ids]

    if hasattr(simulator, "gym"):
        gym = simulator.gym
        for offset, env_id in enumerate(idx_cpu.tolist()):
            env_ptr = simulator.envs[env_id]
            actor = simulator.robot_handles[env_id]
            shape_props = gym.get_actor_rigid_shape_properties(env_ptr, actor)
            friction_value = friction_samples_cpu[offset].item()
            for prop in shape_props:
                prop.friction = friction_value
            gym.set_actor_rigid_shape_properties(env_ptr, actor, shape_props)
    elif simulator.__class__.__name__ == "IsaacSim":
        try:
            from isaaclab.managers import SceneEntityCfg
        except ImportError as exc:  # pragma: no cover - defensive
            raise RuntimeError("IsaacSim friction randomization requires isaaclab.") from exc
        env_ids_cpu = idx.to(device="cpu", dtype=torch.long)
        if env_ids_cpu.numel() == 0:
            return

        asset_cfg = SceneEntityCfg("robot", body_names=".*")
        asset_cfg.resolve(simulator.scene)  # Not stricly required, but a good practice

        _isaacsim_randomize_rigid_body_material(
            simulator,
            env_ids_cpu,
            asset_cfg,
            static_friction_range=(friction_range[0], friction_range[1]),
            dynamic_friction_range=(friction_range[0], friction_range[1]),
            restitution_range=(0.0, 0.0),
            num_buckets=num_buckets,
        )

    elif simulator.simulator_config.mujoco_backend == MujocoBackend.WARP:
        from holosoma.simulator.mujoco.backends.warp_randomization import randomize_field

        assert len(friction_range) == 2, f"friction_range must have exactly 2 elements, got {len(friction_range)}"
        randomize_field(
            simulator,
            field=getattr(randomize_friction_startup, MUJOCO_FIELD_ATTR),
            ranges={0: (friction_range[0], friction_range[1])},
            env_ids=idx,
            operation="abs",
        )

    else:  # pragma: no cover - defensive
        raise RandomizerNotSupportedError(
            f"Unsupported simulator type '{type(simulator).__name__}' for friction randomization."
        )


@mujoco_required_field("geom_friction")
def randomize_robot_rigid_body_material_startup(
    env,
    env_ids: Sequence[int] | torch.Tensor | None = None,
    *,
    static_friction_range: Sequence[float],
    dynamic_friction_range: Sequence[float],
    restitution_range: Sequence[float],
    enabled: bool = True,
    **_,
) -> None:
    """Randomize robot rigid body material properties (friction, restitution)."""
    if not enabled:
        return

    idx = _ensure_env_ids_tensor(env, env_ids)
    if idx.numel() == 0:
        return

    simulator = env.simulator
    # MuJoCo/Warp support added 2026-07-19. Previously this raised, which forced Stage-C mjwarp runs
    # to pass --randomization.ignore_unsupported=True and therefore train/fine-tune with foot
    # friction pinned at the MJCF default instead of the randomized range the policy was trained
    # against -- a silent train-vs-finetune mismatch. Mirrors randomize_friction_startup's existing
    # WARP branch (same randomize_field helper, same "abs" operation, same geom_friction field).
    #
    # Two deliberate fidelity notes:
    #  - MuJoCo's geom_friction is [sliding, torsional, rolling]; only index 0 (sliding) is
    #    randomized here. It has no separate static/dynamic split, so the two IsaacSim ranges are
    #    merged into one sliding range spanning both (min of the mins, max of the maxes).
    #  - restitution has no direct MuJoCo model field (it is encoded in solref/solimp), so
    #    restitution_range is NOT applied on this backend. Flagged rather than silently dropped.
    if simulator.__class__.__name__ != "IsaacSim":
        if getattr(simulator.simulator_config, "mujoco_backend", None) == MujocoBackend.WARP:
            from holosoma.simulator.mujoco.backends.warp_randomization import randomize_field

            lo = float(min(static_friction_range[0], dynamic_friction_range[0]))
            hi = float(max(static_friction_range[1], dynamic_friction_range[1]))
            randomize_field(
                simulator,
                field=getattr(randomize_robot_rigid_body_material_startup, MUJOCO_FIELD_ATTR),
                ranges={0: (lo, hi)},
                env_ids=idx,
                operation="abs",
            )
            return
        raise RandomizerNotSupportedError(
            f"randomize_robot_rigid_body_material_startup supports IsaacSim and MuJoCo-Warp, "
            f"got {type(simulator).__name__}"
        )

    try:
        from isaaclab.managers import SceneEntityCfg
    except ImportError as exc:  # pragma: no cover - defensive
        raise RuntimeError("IsaacSim material randomization requires isaaclab.") from exc

    env_ids_cpu = idx.to(device="cpu", dtype=torch.long)
    if env_ids_cpu.numel() == 0:
        return

    asset_cfg = SceneEntityCfg("robot", body_names=".*")
    asset_cfg.resolve(simulator.scene)

    num_buckets = 64
    _isaacsim_randomize_rigid_body_material(
        simulator,
        env_ids_cpu,
        asset_cfg,
        static_friction_range=(static_friction_range[0], static_friction_range[1]),
        dynamic_friction_range=(dynamic_friction_range[0], dynamic_friction_range[1]),
        restitution_range=(restitution_range[0], restitution_range[1]),
        num_buckets=num_buckets,
    )


@mujoco_required_field("geom_solref")
def randomize_contact_solver_startup(
    env,
    env_ids: Sequence[int] | torch.Tensor | None = None,
    *,
    timeconst_range: Sequence[float] = (0.010, 0.040),
    dampratio_range: Sequence[float] = (0.70, 1.40),
    enabled: bool = True,
    **_,
) -> None:
    """Randomize MuJoCo's CONTACT SOLVER response (geom_solref) per environment.

    Why this exists (2026-07-20). Every other randomizer in this file perturbs the ROBOT -- mass,
    CoM, inertia, armature, joint friction, PD gains, action delay, RFI torque, pushes -- or the
    surface friction coefficient. NONE of them perturbs how contact itself is RESOLVED. Under
    MuJoCo that response is governed by geom_solref = [timeconst, dampratio] (and geom_solimp),
    which were constant across every env and every episode. A policy trained in MuJoCo therefore
    faced one exact contact response for its whole life and could overfit it for free.

    That is the measured failure mode, not a hypothesis: with 6 matched checkpoints per arm and
    n=128, a mjwarp-trained policy topples 26.6% in MuJoCo but 100.0% in IsaacSim, while the
    IsaacSim-trained policy is the mirror image (68.4% / 99.2%). Each policy is exploiting its own
    engine's contact model. Randomizing solref attacks exactly that: the policy can no longer
    assume a single contact stiffness/damping and must find a solution with real margin.

    MuJoCo's default solref is [0.02, 1.0]. The default timeconst range here spans 0.010-0.040,
    i.e. from markedly stiffer to markedly softer contact than nominal, and dampratio 0.7-1.4
    covers under- to over-damped. Widen with care -- very small timeconst approaches the stiffness
    limit the integrator can resolve at this timestep and will start costing solver stability
    before it buys robustness.

    IsaacSim has no equivalent field (PhysX parameterizes contact through compliance/offsets, not
    solref), so this is a deliberate no-op there rather than an error -- the unified randomization
    preset is shared by both backends and must stay loadable on each.
    """
    if not enabled:
        return

    idx = _ensure_env_ids_tensor(env, env_ids)
    if idx.numel() == 0:
        return

    simulator = env.simulator
    if simulator.__class__.__name__ == "IsaacSim":
        # No PhysX analogue -- see docstring. Intentionally silent: this term is expected to be
        # present-but-inert in every IsaacSim run using the shared unified preset.
        return

    if getattr(simulator.simulator_config, "mujoco_backend", None) == MujocoBackend.WARP:
        from holosoma.simulator.mujoco.backends.warp_randomization import randomize_field

        randomize_field(
            simulator,
            field=getattr(randomize_contact_solver_startup, MUJOCO_FIELD_ATTR),
            ranges={
                0: (float(timeconst_range[0]), float(timeconst_range[1])),
                1: (float(dampratio_range[0]), float(dampratio_range[1])),
            },
            env_ids=idx,
            operation="abs",
        )
        return

    raise RandomizerNotSupportedError(
        f"randomize_contact_solver_startup supports MuJoCo-Warp (and is inert on IsaacSim), "
        f"got {type(simulator).__name__}"
    )


@mujoco_required_field("dof_armature")
def randomize_joint_armature_startup(
    env,
    env_ids: Sequence[int] | torch.Tensor | None = None,
    *,
    armature_range: Sequence[float] | None = None,
    joint_friction_range: Sequence[float] | None = None,
    enabled: bool = True,
    **_,
) -> None:
    """Randomize per-joint armature (rotor inertia) and/or joint friction coefficient at startup.

    IsaacSim-only. Both ranges use SCALE operation (e.g. armature_range=[0.8, 1.2] means 80-120% of
    the URDF default per joint). Pass None for either range to skip that property.

    Targets a PhysX-side axis the rest of this config's DR does not touch: armature changes each
    joint's effective rotor inertia, and joint friction changes its internal (non-contact) drag —
    both shape how a joint responds to a fast torque command, which is exactly the failure mode
    diagnosed for the Stage-C kick (stance knee under-extends at the strike because the policy's
    feedback controller overfit PhysX's specific joint response). Mass/CoM/RFI (added earlier this
    file) did not close the MuJoCo sim2sim gap; see memory stagec-kick-physx-mujoco-gap for the
    2026-07-15 negative result that motivated this term.
    """
    if not enabled:
        return
    if armature_range is None and joint_friction_range is None:
        return

    idx = _ensure_env_ids_tensor(env, env_ids)
    if idx.numel() == 0:
        return

    simulator = env.simulator
    # MuJoCo/Warp support added 2026-07-19, same motivation as the material randomizer above:
    # without it, Stage-C mjwarp runs had to pass --randomization.ignore_unsupported=True and so
    # trained/fine-tuned against a NARROWER DR distribution than the policy originally saw.
    # Only ARMATURE is randomized here: mujoco_required_field attaches exactly one field per
    # function, and armature is the meaningful half. joint_friction_range is skipped, which loses
    # very little -- on the IsaacSim side it SCALES the URDF default, and this robot's URDF joint
    # friction is 0 for all 29 joints (dof_joint_friction_list is all zeros), so scaling it there is
    # already a no-op. (MuJoCo's own MJCF does carry frictionloss=0.1, but matching that is a
    # separate model-fidelity question, not DR.)
    if simulator.__class__.__name__ != "IsaacSim":
        if getattr(simulator.simulator_config, "mujoco_backend", None) == MujocoBackend.WARP:
            if armature_range is not None:
                from holosoma.simulator.mujoco.backends.warp_randomization import randomize_field

                randomize_field(
                    simulator,
                    field=getattr(randomize_joint_armature_startup, MUJOCO_FIELD_ATTR),
                    ranges=(float(armature_range[0]), float(armature_range[1])),
                    env_ids=idx,
                    operation="scale",
                )
            return
        raise RandomizerNotSupportedError(
            f"randomize_joint_armature_startup supports IsaacSim and MuJoCo-Warp, "
            f"got {type(simulator).__name__}"
        )

    try:
        from isaaclab.managers import SceneEntityCfg
    except ImportError as exc:  # pragma: no cover - defensive
        raise RuntimeError("IsaacSim joint-parameter randomization requires isaaclab.") from exc

    env_ids_cpu = idx.to(device="cpu", dtype=torch.long)
    if env_ids_cpu.numel() == 0:
        return

    logger.info(
        f"[Randomization] Joint params: armature={armature_range}, joint_friction={joint_friction_range} "
        "(operation=scale)"
    )

    asset_cfg = SceneEntityCfg("robot", joint_names=".*")
    asset_cfg.resolve(simulator.scene)

    _isaacsim_randomize_joint_parameters(
        simulator,
        env_ids_cpu,
        asset_cfg,
        armature_distribution_params=tuple(armature_range) if armature_range is not None else None,
        friction_distribution_params=tuple(joint_friction_range) if joint_friction_range is not None else None,
        operation="scale",
    )


def randomize_object_rigid_body_material_startup(
    env,
    env_ids: Sequence[int] | torch.Tensor | None = None,
    *,
    static_friction_range: Sequence[float],
    dynamic_friction_range: Sequence[float],
    restitution_range: Sequence[float],
    enabled: bool = True,
    **_,
) -> None:
    """Randomize object rigid body material properties (friction, restitution)."""
    if not enabled:
        return

    idx = _ensure_env_ids_tensor(env, env_ids)
    if idx.numel() == 0:
        return

    simulator = env.simulator
    if simulator.__class__.__name__ != "IsaacSim":
        raise RandomizerNotSupportedError(
            f"randomize_object_rigid_body_material_startup only supports IsaacSim, got {type(simulator).__name__}"
        )

    try:
        from isaaclab.managers import SceneEntityCfg
    except ImportError as exc:  # pragma: no cover - defensive
        raise RuntimeError("IsaacSim material randomization requires isaaclab.") from exc

    env_ids_cpu = idx.to(device="cpu", dtype=torch.long)
    if env_ids_cpu.numel() == 0:
        return

    asset_cfg = SceneEntityCfg("object", body_names=".*")
    asset_cfg.resolve(simulator.scene)

    num_buckets = 64
    _isaacsim_randomize_rigid_body_material(
        simulator,
        env_ids_cpu,
        asset_cfg,
        static_friction_range=(static_friction_range[0], static_friction_range[1]),
        dynamic_friction_range=(dynamic_friction_range[0], dynamic_friction_range[1]),
        restitution_range=(restitution_range[0], restitution_range[1]),
        num_buckets=num_buckets,
    )


def randomize_object_rigid_body_mass_startup(
    env,
    env_ids: Sequence[int] | torch.Tensor | None = None,
    *,
    mass_distribution_params: Sequence[float],
    enabled: bool = True,
    **_,
) -> None:
    """Randomize object rigid body mass."""
    if not enabled:
        return

    idx = _ensure_env_ids_tensor(env, env_ids)
    if idx.numel() == 0:
        return

    simulator = env.simulator
    if simulator.__class__.__name__ != "IsaacSim":
        raise RandomizerNotSupportedError(
            f"randomize_object_rigid_body_mass_startup only supports IsaacSim, got {type(simulator).__name__}"
        )

    try:
        from isaaclab.managers import SceneEntityCfg

    except ImportError as exc:  # pragma: no cover - defensive
        raise RuntimeError("IsaacSim mass randomization requires isaaclab.") from exc

    env_ids_cpu = idx.to(device="cpu", dtype=torch.long)
    if env_ids_cpu.numel() == 0:
        return

    asset_cfg = SceneEntityCfg("object", body_names=".*")
    asset_cfg.resolve(simulator.scene)

    _isaacsim_randomize_rigid_body_mass(
        simulator,
        env_ids_cpu,
        asset_cfg,
        (mass_distribution_params[0], mass_distribution_params[1]),
        operation="add",
    )


def randomize_object_rigid_body_inertia_startup(
    env,
    env_ids: Sequence[int] | torch.Tensor | None = None,
    *,
    inertia_distribution_params_dict: dict[str, tuple[float, float]],
    enabled: bool = True,
    **_,
) -> None:
    """Randomize object rigid body inertia."""
    if not enabled:
        return

    idx = _ensure_env_ids_tensor(env, env_ids)
    if idx.numel() == 0:
        return

    simulator = env.simulator
    if simulator.__class__.__name__ != "IsaacSim":
        raise RandomizerNotSupportedError(
            f"randomize_object_rigid_body_inertia_startup only supports IsaacSim, got {type(simulator).__name__}"
        )

    try:
        from isaaclab.managers import SceneEntityCfg
    except ImportError as exc:  # pragma: no cover - defensive
        raise RuntimeError("IsaacSim inertia randomization requires isaaclab.") from exc

    from holosoma.simulator.isaacsim.events import randomize_rigid_body_inertia

    env_ids_cpu = idx.to(device="cpu", dtype=torch.long)
    if env_ids_cpu.numel() == 0:
        return

    asset_cfg = SceneEntityCfg("object", body_names=".*")
    asset_cfg.resolve(simulator.scene)

    ordering = ["Ixx", "Iyy", "Izz", "Ixy", "Iyz", "Ixz"]
    lower_bounds = [inertia_distribution_params_dict[key][0] for key in ordering]
    upper_bounds = [inertia_distribution_params_dict[key][1] for key in ordering]
    inertia_distribution_params = (torch.tensor(lower_bounds, device="cpu"), torch.tensor(upper_bounds, device="cpu"))

    randomize_rigid_body_inertia(
        simulator,
        env_ids_cpu,
        asset_cfg,
        inertia_distribution_params,
        operation="scale",
        distribution="uniform",
    )


def configure_torque_rfi(
    env,
    env_ids,
    *,
    enabled: bool | None = None,
    rfi_lim: float | None = None,
    **_,
) -> None:
    """Toggle torque RFI injection flag."""
    prev_enabled, prev_lim = env._pending_torque_rfi
    enabled_flag = prev_enabled if enabled is None else bool(enabled)
    rfi_limit = prev_lim if rfi_lim is None else float(rfi_lim)
    env._pending_torque_rfi = (enabled_flag, rfi_limit)

    state = env.randomization_manager.get_state("actuator_randomizer_state")
    if state is not None:
        state.enable_rfi_lim = enabled_flag
    term = _get_joint_action_term(env)
    if term is not None:
        term.configure_torque_rfi(enabled=enabled_flag, rfi_lim=rfi_limit)


def apply_pushes(
    env,
    *,
    enabled: bool | None = None,
    push_interval_s: Sequence[float] | None = None,
    max_push_vel: Sequence[float] | None = None,
    **_,
) -> None:
    """Apply random pushes based on the current schedule."""
    state = env.randomization_manager.get_state("push_randomizer_state")
    if state is None:
        raise AttributeError("PushRandomizerState is not registered with the randomization manager.")

    state.configure(enabled=enabled, push_interval_s=push_interval_s, max_push_vel=max_push_vel)
    env._push_robots_enabled = state.enabled

    if env.is_evaluating or not state.enabled:
        return

    push_robot_env_ids = state.due_envs(env.dt)
    if push_robot_env_ids.numel() == 0:
        return

    state.zero_counters(push_robot_env_ids)
    state.resample(push_robot_env_ids)
    env._max_push_vel = state.max_push_vel.clone()
    env._push_robots(push_robot_env_ids)


def apply_body_pushes(env, **_) -> None:
    """Apply sustained, body-targeted disturbance forces based on the current schedule.

    See BodyPushRandomizerState's docstring for the full rationale. Structurally mirrors
    apply_pushes (schedule check -> resample -> apply) but forces are STATEFUL across steps
    (a duration, not an instant), so this also has to tick down and clear expired ones -- apply_
    pushes never needs that, a velocity impulse has nothing left to hold after the tick it fires.

    Gated on env.is_evaluating exactly like apply_pushes, so eval rollouts see zero disturbance
    from this mechanism, matching the existing push's own behavior.
    """
    state = env.randomization_manager.get_state("body_push_randomizer_state")
    if state is None:
        raise AttributeError("BodyPushRandomizerState is not registered with the randomization manager.")

    if env.is_evaluating or not state.enabled:
        # One clearing write if a disturbance was mid-flight when eval started or the term was
        # disabled -- otherwise a force already resident in the simulator's persistent external-
        # force buffer would keep being reapplied every physics step forever (see
        # set_external_body_forces's docstring: both backends hold the last value written until
        # explicitly overwritten, neither auto-clears per step).
        if state._forces_written and state.force_buf is not None:
            state.force_buf.zero_()
            env.simulator.set_external_body_forces(state.force_buf)
            state._forces_written = False
        return

    assert state.counter is not None and state.interval_steps is not None
    assert state.remaining_steps is not None and state.force_buf is not None

    due = (state.counter >= state.interval_steps).nonzero(as_tuple=False).flatten()

    # Snapshot BEFORE sampling: a freshly-sampled disturbance must get its full first tick applied
    # untouched. Without this exclusion, an env due this exact call would be sampled (remaining_
    # steps set to its full duration) and then immediately decremented by the block below in the
    # SAME call -- at a 1-step duration that zeros it before a single write ever reaches the
    # simulator, silently dropping the disturbance entirely. Caught by
    # test_apply_body_pushes_fires_when_due_and_clears_on_expiry.
    already_active = state.remaining_steps > 0

    if due.numel() > 0:
        state.counter[due] = 0
        state._resample_intervals(due)
        state._sample_forces(due)

    if already_active.any():
        state.remaining_steps[already_active] -= 1
        just_expired = already_active & (state.remaining_steps == 0)
        if just_expired.any():
            state.force_buf[just_expired] = 0.0

    still_active = bool((state.remaining_steps > 0).any())
    # Write whenever something is live OR the last call left a nonzero buffer (covers the exact
    # step every active env expires simultaneously: still_active is False but the simulator still
    # holds last step's nonzero forces until this write clears them).
    if still_active or state._forces_written:
        env.simulator.set_external_body_forces(state.force_buf)
    state._forces_written = still_active


# --- Per-episode ball-observation bias (2026-07-21; made per-SKILL 2026-07-23) ---------------------
# See BallConfig.observation_bias / SkillConfig.observation_bias for the full rationale. Cached on
# `env` (not a module-level global) since building the per-motion tensor needs env.device/motion
# count, and the value is a set-before-start config constant, never mutated mid-run -- same
# fixed-for-the-whole-run caching rationale as the old module-level float, just keyed per-env
# instead of per-process since multiple envs (e.g. tests) could otherwise cross-contaminate a
# module-level cache.
def _get_ball_obs_bias_scale_per_motion(env) -> torch.Tensor:
    """(num_motions,) tensor of each skill's observation_bias magnitude (meters).

    N-skill mode: from motion_command.skill_ball_configs, one entry per skill (declaration order,
    index-aligned with motion_ids -- same convention as every other per-motion table in wbt.py).
    Legacy mode (no N-skill config, or a ball-less env where skill_ball_configs was never
    populated): BallConfig.observation_bias broadcast to every motion -- for num_motions == 1 this
    is bit-identical to the old cached-global-scalar behavior."""
    cached = getattr(env, "_ball_obs_bias_scale_per_motion", None)
    if cached is not None:
        return cached

    motion_command = env.command_manager.get_state("motion_command")
    num_motions = motion_command.motion.num_motions
    skill_ball_configs = getattr(motion_command, "skill_ball_configs", None)
    if skill_ball_configs:
        scale = torch.tensor(
            [float(getattr(sc, "observation_bias", 0.0)) for sc in skill_ball_configs],
            dtype=torch.float32,
            device=env.device,
        )
    else:
        from holosoma.config_types.simulator import load_ball_config

        scale = torch.full(
            (num_motions,),
            float(getattr(load_ball_config(), "observation_bias", 0.0)),
            dtype=torch.float32,
            device=env.device,
        )
    env._ball_obs_bias_scale_per_motion = scale
    return scale


def randomize_ball_obs_bias(env, env_ids, **_) -> None:
    """Reset hook: draw a per-episode CONSTANT heading-frame bias (meters) for the just-reset envs'
    kick_ball_pos_b observation, stored on ``env._ball_obs_bias`` and read by
    managers/observation/terms/unified.py::ball_pos_b. Magnitude is PER-SKILL
    (SkillConfig.observation_bias) in N-skill mode, gathered per-env via motion_ids; legacy mode
    broadcasts BallConfig.observation_bias (configs/ball*.yaml) to every env, bit-identical to
    before this was made per-skill. 0.0 (either mode) => that env's buffer entry ends up exactly
    zero, the same no-op as before. Applied to all reset env_ids regardless of task_mode --
    harmless for locomotion-mode envs since their kick_ball_pos_b is zeroed by the observation
    task-mode mask anyway."""
    buf = getattr(env, "_ball_obs_bias", None)
    if buf is None or buf.shape[0] != env.num_envs:
        buf = torch.zeros(env.num_envs, 3, device=env.device)
        env._ball_obs_bias = buf

    scale_per_motion = _get_ball_obs_bias_scale_per_motion(env)
    motion_command = env.command_manager.get_state("motion_command")
    env_bias_scale = scale_per_motion[motion_command.motion_ids[env_ids]]
    buf[env_ids] = (torch.rand(len(env_ids), 3, device=env.device) * 2.0 - 1.0) * env_bias_scale.unsqueeze(-1)


def _get_ball_static_obs_probability(env) -> float:
    """Cached (per-process-lifetime) GLOBAL probability that a reset freezes kick_ball_pos_b for
    the whole episode -- unlike observation_bias, this is not per-skill (see
    MultiSkillConfig.ball_static_obs_probability's own docstring for why: it models the
    perception PIPELINE being stuck, not a property of any one skill), so a single cached float
    is enough -- no per-motion table needed."""
    cached = getattr(env, "_ball_static_obs_probability", None)
    if cached is not None:
        return cached

    from holosoma.config_types.multi_skill import load_multi_skill_config, multi_skill_mode_enabled

    if multi_skill_mode_enabled():
        prob = float(load_multi_skill_config().ball_static_obs_probability)
    else:
        from holosoma.config_types.simulator import load_ball_config

        prob = float(load_ball_config().ball_static_obs_probability)
    env._ball_static_obs_probability = prob
    return prob


def randomize_ball_obs_freeze(env, env_ids, **_) -> None:
    """Reset hook: with probability ``ball_static_obs_probability`` (global, see
    ``_get_ball_static_obs_probability``), mark the just-reset envs' ``kick_ball_pos_b``
    observation to FREEZE for the whole upcoming episode instead of updating live -- 2026-07-24
    deployment-robustness training for a stuck/dead perception pipeline (sensor dropout, a stale
    cached reading held by whatever's upstream of the policy). User directive: only
    ``kick_ball_pos_b`` freezes -- ``kick_target_pos_b`` is a command, not a sensor reading, and
    stays live.

    The actual frozen VALUE is drawn lazily, on this episode's first real post-reset observation
    compute (see ``managers/observation/terms/unified.py::ball_pos_b`` /
    ``_draw_independent_frozen_ball_reading``) -- NOT here, since at reset-hook time (inside
    ``reset_envs_idx()``, before ``command_manager.reset()`` has assigned this episode's fresh
    ``motion_ids``) the per-skill tables that draw needs aren't valid yet. 2026-07-24 (revised
    same day, user directive): that value is drawn INDEPENDENTLY of the ball's actual simulated
    position -- nominal + the same OOD-aware noise mechanism a real spawn uses -- so it can land
    in the OOD region on its own, not only when the same episode's real ball placement also
    happens to roll OOD. Not further perturbed once drawn (that's what makes it "static, not
    moving", per the user's own description).

    ``env._ball_obs_frozen_mask``: bool, per-env, whether THIS episode is frozen.
    ``env._ball_obs_frozen_captured``: bool, per-env, whether the frozen value has been captured
    yet this episode -- always reset to False here (whether or not freeze fires this time), so
    ``ball_pos_b`` knows to (re)capture fresh on this episode's first real call.
    ``env._ball_obs_frozen_value``: the captured value itself, once captured.
    """
    prob = _get_ball_static_obs_probability(env)

    mask = getattr(env, "_ball_obs_frozen_mask", None)
    if mask is None or mask.shape[0] != env.num_envs:
        mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._ball_obs_frozen_mask = mask
    captured = getattr(env, "_ball_obs_frozen_captured", None)
    if captured is None or captured.shape[0] != env.num_envs:
        captured = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._ball_obs_frozen_captured = captured
    value = getattr(env, "_ball_obs_frozen_value", None)
    if value is None or value.shape[0] != env.num_envs:
        value = torch.zeros(env.num_envs, 3, device=env.device)
        env._ball_obs_frozen_value = value

    if prob > 0.0:
        mask[env_ids] = torch.rand(len(env_ids), device=env.device) < prob
    else:
        mask[env_ids] = False
    captured[env_ids] = False
