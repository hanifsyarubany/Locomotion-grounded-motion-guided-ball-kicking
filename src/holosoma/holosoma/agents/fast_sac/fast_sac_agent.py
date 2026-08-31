from __future__ import annotations

import copy
import dataclasses
import inspect
import itertools
import math
import os
import queue
import threading
from contextlib import contextmanager
from typing import Any, Callable, Dict, Sequence

import numpy as np
import tqdm
import wandb
from loguru import logger

from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.agents.callbacks.base_callback import RLEvalCallback
from holosoma.agents.fast_sac.fast_sac import Actor, CNNActor, CNNCritic, Critic
from holosoma.agents.fast_sac.fast_sac_utils import (
    EmpiricalNormalization,
    SimpleReplayBuffer,
    save_params,
)
from holosoma.agents.modules.augmentation_utils import SymmetryUtils
from holosoma.agents.modules.logging_utils import LoggingHelper
from holosoma.config_types.algo import FastSACConfig
from holosoma.config_types.observation import ObsTermCfg
from holosoma.managers.utils import resolve_callable
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.utils.average_meters import TensorAverageMeterDict
from holosoma.utils.helpers import instantiate
from holosoma.utils.inference_helpers import (
    attach_onnx_metadata,
    export_motion_and_policy_as_onnx,
    export_policy_as_onnx,
    get_command_ranges_from_env,
    get_control_gains_from_config,
    get_kick_recovery_locomotion_flip_metadata,
    get_skill_ball_target_metadata,
    get_skill_motion_boundaries_metadata,
    get_skill_pre_recovery_metadata,
    get_urdf_text_from_robot_config,
)
from holosoma.utils.safe_torch_import import (
    F,
    GradScaler,
    TensorboardSummaryWriter,
    TensorDict,
    autocast,
    nn,
    optim,
    torch,
)

torch.set_float32_matmul_precision("high")


class FastSACEnv:
    def __init__(
        self,
        env: BaseTask,
        actor_obs_keys: Sequence[str],
        critic_obs_keys: Sequence[str],
    ):
        self._env = env
        self._actor_obs_keys = actor_obs_keys
        self._critic_obs_keys = critic_obs_keys

        # Initialize per-joint action boundaries for proper tanh scaling
        self._action_boundaries = self._compute_action_boundaries()

    def __getattr__(self, name: str):
        """Delegate attribute access to the wrapped environment."""
        return getattr(self._env, name)

    def reset(self) -> torch.Tensor:
        obs_dict = self._env.reset_all()
        return torch.cat([obs_dict[k] for k in self._actor_obs_keys], dim=1)

    def reset_with_critic_obs(self) -> tuple[torch.Tensor, torch.Tensor]:
        obs_dict = self._env.reset_all()
        actor_obs = torch.cat([obs_dict[k] for k in self._actor_obs_keys], dim=1)
        critic_obs = torch.cat([obs_dict[k] for k in self._critic_obs_keys], dim=1)
        return actor_obs, critic_obs

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        # Actions are now already scaled by the actor, so pass them directly to the environment
        obs_dict, rew_buf, reset_buf, info_dict = self._env.step({"actions": actions})  # type: ignore[attr-defined]
        actor_obs = torch.cat([obs_dict[k] for k in self._actor_obs_keys], dim=1)
        critic_obs = torch.cat([obs_dict[k] for k in self._critic_obs_keys], dim=1)
        if "final_observations" in info_dict:
            # Use true final observations when available
            final_actor_obs = torch.cat([info_dict["final_observations"][k] for k in self._actor_obs_keys], dim=1)
            final_critic_obs = torch.cat([info_dict["final_observations"][k] for k in self._critic_obs_keys], dim=1)
        else:
            final_actor_obs = actor_obs
            final_critic_obs = critic_obs
        extras = {
            "time_outs": info_dict["time_outs"],
            "observations": {
                "critic": critic_obs,
                "final": {
                    "actor_obs": final_actor_obs,
                    "critic_obs": final_critic_obs,
                },
            },
            "episode": info_dict["episode"],
            "episode_all": info_dict["episode_all"],
            "raw_episode": info_dict.get("raw_episode", {}),
            "raw_episode_all": info_dict.get("raw_episode_all", {}),
            "to_log": info_dict["to_log"],
        }
        return actor_obs, rew_buf, reset_buf, extras

    def _compute_action_boundaries(self) -> torch.Tensor:
        """
        Compute per-joint action scaling factors based on robot configuration.
        Returns tensor of shape (num_dof,) containing the scaling factor for each joint.

        The scaling factor is the maximum difference between default and joint limits,
        ensuring that action=0 corresponds to default position and action=±1 reaches
        the furthest limit from default.
        """
        robot_config = self._env.robot_config

        # Get joint limits and default positions
        dof_pos_lower_limits = torch.tensor(robot_config.dof_pos_lower_limit_list, device=self._env.device)
        dof_pos_upper_limits = torch.tensor(robot_config.dof_pos_upper_limit_list, device=self._env.device)

        # Get default joint angles
        default_joint_angles = torch.zeros(len(robot_config.dof_names), device=self._env.device)
        for i, joint_name in enumerate(robot_config.dof_names):
            if joint_name in robot_config.init_state.default_joint_angles:
                default_joint_angles[i] = robot_config.init_state.default_joint_angles[joint_name]

        # Get action scale from robot config
        action_scale = robot_config.control.action_scale

        # Compute maximum range from default to either limit for each joint
        # This ensures symmetric scaling where action=0 -> default position
        range_to_lower = torch.abs(dof_pos_lower_limits - default_joint_angles)
        range_to_upper = torch.abs(dof_pos_upper_limits - default_joint_angles)
        max_range = torch.maximum(range_to_lower, range_to_upper)

        # Account for action_scale: the environment applies actions_scaled = actions * action_scale
        # So our scaling factor should be: max_range / action_scale
        action_scaling_factors = max_range / action_scale

        logger.info(f"Computed action scaling factors for {len(robot_config.dof_names)} DOFs")
        logger.info(f"Action scale: {action_scale}")
        logger.info(f"Scaling: {action_scaling_factors}")

        return action_scaling_factors


def _validate_l2sp_anchor(l2sp_weight: float, anchor: list[torch.Tensor] | None) -> None:
    """Raises if L2-SP (see FastSACConfig.l2sp_weight) is enabled but no checkpoint was ever
    loaded to anchor it to -- regularizing toward a randomly initialized actor would quietly fight
    training instead of protecting anything, so this fails loudly instead. Extracted as a free
    function (rather than inlined in FastSACAgent.learn()) so it's testable without constructing a
    full agent."""
    if l2sp_weight > 0.0 and anchor is None:
        raise ValueError(
            f"l2sp_weight={l2sp_weight} > 0 but no checkpoint was loaded, so there is no anchor to "
            "regularize toward. L2-SP only makes sense when RESUMING from a checkpoint whose "
            "skills you want to protect -- pass --training.checkpoint, or set l2sp_weight back to "
            "0.0 for a from-scratch run."
        )


# 2026-08-18: obs-distribution-shift guard's own env var -- module-level (not a class attribute)
# so it's reachable without a real FastSACAgent instance, matching the same pattern
# distill_specialists.py already uses for its own env-var knobs (_TRUTHY there is a separate,
# independent set for that module -- not shared with this one on purpose, each file owns its own).
_SKIP_OBS_NORMALIZER_RESET_ENV_VAR = "HOLOSOMA_SKIP_OBS_NORMALIZER_RESET"
_OBS_NORMALIZER_RESET_TRUTHY = {"1", "true", "yes", "on"}


class FastSACAgent(BaseAlgo):
    """
    FastSAC is an efficient variant of Soft Actor-Critic (SAC) tuned for
    large-scale training with massively parallel simulation.
    See https://arxiv.org/abs/2505.22642 for more details about FastTD3.
    Detailed technical report for FastSAC will be available soon.
    """

    config: FastSACConfig
    env: FastSACEnv  # type: ignore[assignment]
    actor: Actor
    qnet: Critic

    def __init__(
        self, env: BaseTask, config: FastSACConfig, device: str, log_dir: str, multi_gpu_cfg: dict | None = None
    ):
        wrapped_env = FastSACEnv(env, config.actor_obs_keys, config.critic_obs_keys)

        super().__init__(wrapped_env, config, device, multi_gpu_cfg)  # type: ignore[arg-type]
        self.unwrapped_env = env
        self.log_dir = log_dir
        self.global_step = 0
        self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self.logging_helper = LoggingHelper(
            self.writer,
            self.log_dir,
            device=self.device,
            num_envs=self.env.num_envs,
            num_steps_per_env=config.logging_interval,
            num_learning_iterations=config.num_learning_iterations,
            is_main_process=self.is_main_process,
            num_gpus=self.gpu_world_size,
        )

        self.training_metrics = TensorAverageMeterDict()
        self.eval_callbacks: list[RLEvalCallback] = []

        self._kick_rollout_thread: threading.Thread | None = None
        self._kick_rollout_video_queue: queue.Queue[tuple[int, str, str]] = queue.Queue()
        self._walk_rollout_thread: threading.Thread | None = None
        self._walk_rollout_video_queue: queue.Queue[tuple[int, str]] = queue.Queue()
        # 2026-08-13: walk-then-trigger variant (see record_mujoco_kick_rollout.py's own
        # walk_s/forward_speed docstring) -- own thread/queue, same "run concurrently, not
        # sequentially" rationale as the kick vs walk rollouts above.
        self._kick_handoff_rollout_thread: threading.Thread | None = None
        self._kick_handoff_rollout_video_queue: queue.Queue[tuple[int, str, str]] = queue.Queue()
        # 2026-08-19: N-trial in-distribution fall-rate scan -- own thread/queue, same "run
        # concurrently, not sequentially" rationale as the other three rollouts above. Queue item
        # is (triggered_at_step, wandb_key, fall_rate) rather than a video path -- see
        # _drain_mujoco_survival_scan_queue.
        self._survival_scan_thread: threading.Thread | None = None
        self._survival_scan_result_queue: queue.Queue[tuple[int, str, float]] = queue.Queue()
        # 2026-08-30: forced kick->locomotion flip alive-rate scan -- own thread/queue, same
        # "run concurrently, not sequentially" rationale as the other rollouts above.
        self._kick_to_loco_flip_scan_thread: threading.Thread | None = None
        self._kick_to_loco_flip_scan_result_queue: queue.Queue[tuple[int, str, float]] = queue.Queue()
        # 2026-08-30: reverse direction -- random locomotion then forced flip into kick, fall-rate
        # scan. Own thread/queue, same "run concurrently, not sequentially" rationale.
        self._loco_to_kick_handoff_scan_thread: threading.Thread | None = None
        self._loco_to_kick_handoff_scan_result_queue: queue.Queue[tuple[int, str, float]] = queue.Queue()

    def setup(self) -> None:
        logger.info("Setting up FastSAC")

        # Log curriculum synchronization status for multi-GPU training
        if self.is_multi_gpu:
            if self.has_curricula_enabled():
                logger.info(f"Multi-GPU curriculum synchronization enabled across {self.gpu_world_size} GPUs")

        args = self.config
        device = self.device
        env = self.env

        algo_obs_dim_dict = self.env.observation_manager.get_obs_dims()

        algo_history_length_dict: Dict[str, int] = {}

        for group_cfg in self.env.observation_manager.cfg.groups.values():
            history_len = getattr(group_cfg, "history_length", 1)
            for term_name in group_cfg.terms:
                algo_history_length_dict[term_name] = history_len

        actor_obs_keys = self.config.actor_obs_keys
        critic_obs_keys = self.config.critic_obs_keys

        n_act = self.env.robot_config.actions_dim

        # Compute actor observation dimensions and store indices
        actor_obs_dim = 0
        self.actor_obs_indices = {}
        for obs_key in actor_obs_keys:
            history_len = algo_history_length_dict.get(obs_key, 1)
            obs_size = algo_obs_dim_dict[obs_key] * history_len

            # Store start and end indices for this observation key
            self.actor_obs_indices[obs_key] = {
                "start": actor_obs_dim,
                "end": actor_obs_dim + obs_size,
                "size": obs_size,
            }
            actor_obs_dim += obs_size

        self.actor_obs_dim = actor_obs_dim

        # Compute critic observation dimensions and store indices
        critic_obs_dim = 0
        self.critic_obs_indices = {}
        for obs_key in critic_obs_keys:
            history_len = algo_history_length_dict.get(obs_key, 1)
            obs_size = algo_obs_dim_dict[obs_key] * history_len

            # Store start and end indices for this observation key
            self.critic_obs_indices[obs_key] = {
                "start": critic_obs_dim,
                "end": critic_obs_dim + obs_size,
                "size": obs_size,
            }
            critic_obs_dim += obs_size

        self.scaler = GradScaler(enabled=args.amp)

        self.obs_normalization = args.obs_normalization
        if args.obs_normalization:
            self.obs_normalizer: nn.Module = EmpiricalNormalization(shape=actor_obs_dim, device=device)
            self.critic_obs_normalizer: nn.Module = EmpiricalNormalization(shape=critic_obs_dim, device=device)
        else:
            self.obs_normalizer = nn.Identity()
            self.critic_obs_normalizer = nn.Identity()

        # Get action scaling parameters from the environment
        action_scale = env._action_boundaries if args.use_tanh else torch.ones(n_act, device=device)
        action_bias = torch.zeros(n_act, device=device)  # Assuming zero bias for now

        # Handle CNN actor/critic
        if args.use_cnn_encoder:
            # We assume that MLP doesn't take raw encoder observations
            actor_mlp_obs_keys = [k for k in actor_obs_keys if k != args.encoder_obs_key]
            critic_mlp_obs_keys = [k for k in critic_obs_keys if k != args.encoder_obs_key]
        else:
            actor_mlp_obs_keys = list(actor_obs_keys)
            critic_mlp_obs_keys = list(critic_obs_keys)
        actor_cls, critic_cls = (CNNActor, CNNCritic) if args.use_cnn_encoder else (Actor, Critic)

        self.actor = actor_cls(
            obs_indices=self.actor_obs_indices,
            obs_keys=actor_mlp_obs_keys,
            n_act=n_act,
            num_envs=env.num_envs,
            device=device,
            hidden_dim=args.actor_hidden_dim,
            log_std_max=args.log_std_max,
            log_std_min=args.log_std_min,
            use_tanh=args.use_tanh,
            use_layer_norm=args.use_layer_norm,
            action_scale=action_scale,
            action_bias=action_bias,
            encoder_obs_key=args.encoder_obs_key,
            encoder_obs_shape=args.encoder_obs_shape,
        )
        self.qnet = critic_cls(
            obs_indices=self.critic_obs_indices,
            obs_keys=critic_mlp_obs_keys,
            n_act=n_act,
            num_atoms=args.num_atoms,
            v_min=args.v_min,
            v_max=args.v_max,
            hidden_dim=args.critic_hidden_dim,
            device=device,
            use_layer_norm=args.use_layer_norm,
            num_q_networks=args.num_q_networks,
            encoder_obs_key=args.encoder_obs_key,
            encoder_obs_shape=args.encoder_obs_shape,
        )

        print(self.actor)
        print(self.qnet)

        # 2026-07-28: opt-in per-task-mode entropy target -- see
        # FastSACConfig.kick_target_entropy_ratio's own docstring for the full design/rationale.
        # None (default) -> num_alpha_groups=1, log_alpha/target_entropy stay EXACTLY as before
        # (plain scalar tensor / plain float) -- zero behavior change for every experiment that
        # doesn't opt in, including every existing Unified config until its yaml sets this.
        self.num_alpha_groups = 2 if args.kick_target_entropy_ratio is not None else 1
        if self.num_alpha_groups == 2 and not hasattr(env, "task_mode"):
            raise ValueError(
                "FastSACConfig.kick_target_entropy_ratio is set, but this env has no `task_mode` "
                "attribute -- per-task-mode entropy targets only make sense for a UnifiedManager-"
                "family env (locomotion vs kick). Unset kick_target_entropy_ratio for this "
                "experiment, or run it against an env that exposes task_mode."
            )
        self.log_alpha = torch.tensor(
            [math.log(args.alpha_init)] * self.num_alpha_groups, requires_grad=True, device=device
        )
        self.policy = self.actor.explore

        self.qnet_target = critic_cls(
            obs_indices=self.critic_obs_indices,
            obs_keys=critic_mlp_obs_keys,
            n_act=n_act,
            num_atoms=args.num_atoms,
            v_min=args.v_min,
            v_max=args.v_max,
            hidden_dim=args.critic_hidden_dim,
            device=device,
            use_layer_norm=args.use_layer_norm,
            num_q_networks=args.num_q_networks,
            encoder_obs_key=args.encoder_obs_key,
            encoder_obs_shape=args.encoder_obs_shape,
        )
        self.qnet_target.load_state_dict(self.qnet.state_dict())

        self.q_optimizer = optim.AdamW(
            list(self.qnet.parameters()),
            lr=args.critic_learning_rate,
            weight_decay=args.weight_decay,
            fused=True,
            betas=(0.9, 0.95),
        )
        self.actor_optimizer = optim.AdamW(
            list(self.actor.parameters()),
            lr=args.actor_learning_rate,
            weight_decay=args.weight_decay,
            fused=True,
            betas=(0.9, 0.95),
        )

        # group 0 = locomotion (target_entropy_ratio, unchanged), group 1 = kick
        # (kick_target_entropy_ratio). Plain float in the num_alpha_groups==1 case -- see the
        # log_alpha comment above for why this stays exactly the pre-existing type/shape.
        self.target_entropy = (
            torch.tensor(
                [-n_act * args.target_entropy_ratio, -n_act * args.kick_target_entropy_ratio], device=device
            )
            if self.num_alpha_groups == 2
            else -n_act * args.target_entropy_ratio
        )

        # 2026-07-30: opt-in per-task-mode discount factor -- see FastSACConfig.kick_gamma's own
        # docstring for the full design/rationale. Deliberately mirrors num_alpha_groups/log_alpha/
        # target_entropy above exactly: None (default) -> num_gamma_groups=1, `discount` in
        # `learn()` below is computed with the plain `args.gamma` float, byte-identical to before
        # this existed. gamma_by_group is a plain buffer (not an nn.Parameter -- unlike log_alpha,
        # gamma is never learned), so it needs no optimizer entry and no special checkpoint
        # save/load handling; `self.gamma_by_group[0]` is always locomotion, `[1]` always kick.
        self.num_gamma_groups = 2 if args.kick_gamma is not None else 1
        if self.num_gamma_groups == 2 and not hasattr(env, "task_mode"):
            raise ValueError(
                "FastSACConfig.kick_gamma is set, but this env has no `task_mode` attribute -- "
                "per-task-mode discount factors only make sense for a UnifiedManager-family env "
                "(locomotion vs kick). Unset kick_gamma for this experiment, or run it against an "
                "env that exposes task_mode."
            )
        self.gamma_by_group = (
            torch.tensor([args.gamma, args.kick_gamma], device=device) if self.num_gamma_groups == 2 else None
        )
        # Per-skill replay weighting (2026-08-15) -- see FastSACConfig.skill_replay_weights'
        # docstring for the full rationale (motion_training_ratio silently sets GRADIENT share,
        # not just data share, because the replay buffer is per-env). Same construction-time
        # validation discipline as kick_gamma directly above; empty list (default) leaves
        # skill_weight_by_group None and every loss reduction takes its exact original path.
        self.skill_replay_weights_enabled = len(args.skill_replay_weights) > 0
        if self.skill_replay_weights_enabled:
            if not hasattr(env, "skill_id"):
                raise ValueError(
                    "FastSACConfig.skill_replay_weights is set, but this env has no `skill_id` "
                    "attribute -- per-skill replay weighting only makes sense for a "
                    "UnifiedManager-family env. Unset skill_replay_weights for this experiment."
                )
            num_skills = max(1, len(getattr(env, "_skill_motion_training_ratios", []) or []))
            if len(args.skill_replay_weights) != num_skills:
                raise ValueError(
                    f"FastSACConfig.skill_replay_weights has {len(args.skill_replay_weights)} "
                    f"entries but this run has {num_skills} motion skill(s) -- one weight per "
                    "skill is required, indexed by skill_id."
                )
            if any(w < 0.0 for w in args.skill_replay_weights):
                raise ValueError(
                    f"FastSACConfig.skill_replay_weights must all be >= 0, got "
                    f"{args.skill_replay_weights}"
                )
            if all(w == 0.0 for w in args.skill_replay_weights):
                raise ValueError(
                    "FastSACConfig.skill_replay_weights are all zero -- that would zero out every "
                    "kick transition's gradient. Use relative weights, e.g. [8.0, 1.0]."
                )
        self.skill_weight_by_group = (
            torch.tensor(args.skill_replay_weights, device=device, dtype=torch.float)
            if self.skill_replay_weights_enabled
            else None
        )

        self.alpha_optimizer = optim.AdamW([self.log_alpha], lr=args.alpha_learning_rate, fused=True, betas=(0.9, 0.95))

        # L2-SP continual-learning anchor (2026-08-15) -- see FastSACConfig.l2sp_weight's docstring.
        # Stays None here even when enabled: the anchor is "the weights this run STARTED from", so
        # it can only be captured once load() has actually populated the actor (see load() below).
        # learn() refuses to start if the weight is set but nothing was ever loaded.
        self._l2sp_anchor: list[torch.Tensor] | None = None
        self._l2sp_params: list[torch.Tensor] | None = None
        self._last_l2sp_drift = torch.zeros((), device=device)

        logger.info(f"actor_obs_dim: {actor_obs_dim}, critic_obs_dim: {critic_obs_dim}")

        self.rb = SimpleReplayBuffer(
            n_env=env.num_envs,
            buffer_size=args.buffer_size,
            n_obs=actor_obs_dim,
            n_act=n_act,
            n_critic_obs=critic_obs_dim,
            n_steps=args.num_steps,
            gamma=args.gamma,
            device=device,
            sanitize_enabled=args.replay_buffer_sanitize_enabled,
        )

        if args.use_symmetry:
            # using env._env is not really ideal..
            self.symmetry_utils = SymmetryUtils(env._env)

        # Synchronize model parameters across GPUs for consistent initialization
        if self.is_multi_gpu:
            self._synchronize_model_parameters()

    @contextmanager
    def _maybe_amp(self):
        amp_dtype = torch.bfloat16 if self.config.amp_dtype == "bf16" else torch.float16
        with autocast(device_type="cuda", dtype=amp_dtype, enabled=self.config.amp):
            yield

    def _synchronize_model_parameters(self):
        """Synchronize actor, qnet, and log_alpha parameters across all GPUs."""
        # Broadcast actor weights from rank 0 to all other ranks
        for param in self.actor.parameters():
            torch.distributed.broadcast(param.data, src=0)

        # Broadcast qnet weights from rank 0 to all other ranks
        for param in self.qnet.parameters():
            torch.distributed.broadcast(param.data, src=0)

        # Broadcast log_alpha parameter from rank 0 to all other ranks
        torch.distributed.broadcast(self.log_alpha.data, src=0)

        # Load qnet_target weights from synced qnet
        self.qnet_target.load_state_dict(self.qnet.state_dict())

        logger.info(f"Synchronized model parameters across {self.gpu_world_size} GPUs")

    def _all_reduce_model_grads(self, model: nn.Module) -> None:
        """Batches and all-reduces gradients across GPUs to reduce NCCL call count.

        This flattens all existing parameter gradients into a single contiguous
        tensor, performs one all_reduce, averages by world size, and then
        scatters the reduced values back into the original gradient tensors.
        """
        if not self.is_multi_gpu:
            return
        grads = [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
        if not grads:
            return
        flat = torch.cat(grads)
        torch.distributed.all_reduce(flat, op=torch.distributed.ReduceOp.SUM)
        flat /= self.gpu_world_size
        offset = 0
        for p in model.parameters():
            if p.grad is not None:
                n = p.numel()
                p.grad.copy_(flat[offset : offset + n].view_as(p.grad))
                offset += n

    def _skill_replay_weight(self, data: TensorDict) -> torch.Tensor | None:
        """Per-sample loss weight from FastSACConfig.skill_replay_weights, or None when the
        feature is off. Called ONLY from _sample_and_prepare_batches (plain eager Python, never
        torch.compile'd) -- see that method's own comment for why this must NOT be called from
        inside _update_main/_update_pol despite both of those being the natural-looking call
        sites: a real 3300-env production run hit
        "RuntimeError: size of tensor a (2) must match size of tensor b (6600)" specifically
        inside the compiled _update_main when this fancy-indexing lookup ran there, root cause
        not fully pinned down (num_q_networks also happens to be 2, a suspicious but unconfirmed
        coincidence) -- moving the materialization out of the compiled graph entirely sidesteps
        the whole failure class rather than chasing the exact dynamo mechanism. The compiled
        functions now only ever see a plain already-materialized tensor (data["skill_weight"]),
        the same shape/dtype discipline every other per-sample field they consume already follows.

        Kick transitions get their own skill's configured weight; LOCOMOTION transitions always
        get 1.0 -- gated on `is_kick` rather than `skill_id` alone because `skill_id` is copied
        from the fixed partition for every env including locomotion-partitioned ones, where it is
        meaningless. `is_kick` reads the LIVE task_mode, so a Stage-D handoff env's pre-handoff
        locomotion ticks correctly stay at 1.0 and only its actual kick ticks get reweighted.

        Renormalized to mean 1.0 over the batch so total gradient magnitude (and hence the
        effective learning rate) is unchanged by enabling this -- only the RELATIVE contribution
        of each skill moves. Normalizing against the realized batch also self-corrects for the
        fact that env->skill assignment is a stochastic categorical draw, so the realized env
        counts never exactly equal the configured motion_training_ratio."""
        if self.skill_weight_by_group is None:
            return None
        w = self.skill_weight_by_group[data["skill_id"]]
        w = torch.where(data["is_kick"].bool(), w, torch.ones_like(w))
        return w / w.mean().clamp(min=1e-8)

    def _update_main(
        self, data: TensorDict, actor_frozen: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """`actor_frozen` (2026-08-28): suppress the alpha auto-tune step for this update.

        Set during `critic_warmup_iters` (see FastSACConfig.critic_warmup_iters). SAC's alpha
        controller is an INTEGRATOR: it raises alpha whenever policy entropy sits below
        `target_entropy`, expecting the actor to respond by becoming more stochastic. While the
        actor is frozen the actor CANNOT respond, so the error signal never clears and alpha
        integrates in one direction without bound -- textbook integral windup, a controller whose
        actuator has been disconnected.

        Measured on run 20260827_221312 before this gate existed: alpha/kick ran 0.0010 -> 0.0106
        -> 1.379 -> 14.38 -> 45.59 -> 142.54 across 5000 warmup steps (142,000x). At unfreeze the
        actor's loss `alpha*log_probs - qf_value` was then utterly dominated by the entropy term
        (actor_loss 11.8 -> -4251 in one interval), so the fastest descent direction was "become
        maximally random": policy_entropy -23.4 -> +74.9, kick_topple_frac 0.028 -> 1.0000, and it
        never recovered.

        Only `_update_pol` was gated on `warmup_done` before this; the alpha step lives here in
        `_update_main` and ran unconditionally, which is why freezing the actor silently broke
        alpha. The workaround at the time was `use_autotune=False` for the whole run, which stops
        the windup but also removes entropy regulation entirely for the run's whole remaining life
        -- this gate is the targeted fix, so autotune can stay ON and resume normal regulation the
        moment the actor unfreezes.
        """
        args = self.config

        scaler = self.scaler
        actor = self.actor
        qnet = self.qnet
        qnet_target = self.qnet_target
        q_optimizer = self.q_optimizer
        alpha_optimizer = self.alpha_optimizer

        with self._maybe_amp():
            next_observations = data["next"]["observations"]
            critic_observations = data["critic_observations"]
            next_critic_observations = data["next"]["critic_observations"]
            actions = data["actions"]
            rewards = data["next"]["rewards"]
            dones = data["next"]["dones"].bool()
            truncations = data["next"]["truncations"].bool()
            bootstrap = (truncations | ~dones).float()

            # TEMP DIAGNOSTIC (2026-07-18, see memory mjwarp-training-instability-two-real-bugs-
            # fixed): pinpoints the first tensor to go non-finite in the critic update, since
            # env-level rewards were already confirmed clean right up to the nan transition --
            # the corruption is somewhere in this method. Fires once, does not affect training.
            self._nan_probe_fired = getattr(self, "_nan_probe_fired", False)

            def _nan_probe(name: str, tensor: torch.Tensor) -> None:
                if self._nan_probe_fired:
                    return
                finite = torch.isfinite(tensor)
                if not finite.all():
                    self._nan_probe_fired = True
                    flat = tensor.reshape(tensor.shape[0], -1)
                    finite_flat = torch.isfinite(flat)
                    bad_ij = (~finite_flat).nonzero(as_tuple=False)
                    n_bad = bad_ij.shape[0]
                    n_total = flat.numel()
                    cols_hit = bad_ij[:, 1].unique()
                    logger.error(
                        f"[nan-probe] FIRST non-finite tensor: '{name}' shape={tuple(tensor.shape)} "
                        f"{n_bad}/{n_total} elements non-finite, columns hit: "
                        f"{cols_hit[:20].tolist()}{'...' if cols_hit.numel() > 20 else ''} "
                        f"({cols_hit.numel()} distinct columns) global_step={self.global_step}"
                    )
                    if n_bad > 0:
                        r, c = bad_ij[0, 0].item(), bad_ij[0, 1].item()
                        logger.error(f"[nan-probe] example: row={r} col={c} value={flat[r, c].item()}")
                        logger.error(f"[nan-probe] value range (finite elements only): "
                                     f"min={flat[finite_flat].min().item() if finite_flat.any() else 'N/A'} "
                                     f"max={flat[finite_flat].max().item() if finite_flat.any() else 'N/A'}")

            for _name, _t in [
                ("next_observations", next_observations), ("critic_observations", critic_observations),
                ("next_critic_observations", next_critic_observations), ("actions", actions),
                ("rewards", rewards),
            ]:
                _nan_probe(_name, _t)

            with torch.no_grad():
                next_state_actions, next_state_log_probs = actor.get_actions_and_log_probs(next_observations)
                _nan_probe("next_state_actions", next_state_actions)
                _nan_probe("next_state_log_probs", next_state_log_probs)

                # 2026-07-30: per-sample discount when per-task-mode gamma is active (see
                # FastSACConfig.kick_gamma's docstring) -- is_kick is env-permanent, so this
                # transition's own group is also correct for its "next" state, same reasoning as
                # alpha_per_sample immediately below. num_gamma_groups==1 (the default) takes the
                # EXACT original path unchanged. Only affects the OUTER bootstrap discount here;
                # the replay buffer's inner n-step reward combination stays at the single global
                # gamma (see FastSACConfig.kick_gamma's "Implementation note" -- confirmed inert
                # for any num_steps=1 config, which is every experiment in this project today).
                if self.num_gamma_groups == 2:
                    gamma_per_sample = self.gamma_by_group[data["is_kick"]]
                    discount = gamma_per_sample ** data["next"]["effective_n_steps"]
                else:
                    discount = args.gamma ** data["next"]["effective_n_steps"]

                # 2026-07-28: per-sample alpha when per-task-mode entropy is active (see
                # FastSACConfig.kick_target_entropy_ratio's docstring) -- is_kick is env-permanent,
                # so this transition's own group is also correct for its "next" state (same env,
                # same task, never changes mid-episode). num_alpha_groups==1 (the default, every
                # experiment that hasn't opted in) takes the EXACT original path unchanged.
                if self.num_alpha_groups == 2:
                    alpha_per_sample = self.log_alpha.exp()[data["is_kick"]]
                else:
                    alpha_per_sample = self.log_alpha.exp()

                target_distributions = qnet_target.projection(
                    next_critic_observations,
                    next_state_actions,
                    rewards - discount * bootstrap * alpha_per_sample * next_state_log_probs,
                    bootstrap,
                    discount,
                )
                _nan_probe("target_distributions", target_distributions)
                target_values = qnet_target.get_value(target_distributions)
                target_value_max = target_values.max()
                target_value_min = target_values.min()

            q_outputs = qnet(critic_observations, actions)
            _nan_probe("q_outputs", q_outputs)
            critic_log_probs = F.log_softmax(q_outputs, dim=-1)
            _nan_probe("critic_log_probs", critic_log_probs)
            critic_losses = -torch.sum(target_distributions * critic_log_probs, dim=-1)
            _nan_probe("critic_losses", critic_losses)
            # 2026-08-15 FIX (found while debugging an unrelated skill_replay_weights crash): this
            # was `.mean(dim=1)`. critic_losses' shape is [num_q_networks, batch] -- qnet(...) ==
            # Critic.forward's own `torch.stack(outputs, dim=0)` puts the Q-ENSEMBLE dimension
            # FIRST, not the batch. `.mean(dim=1)` therefore averaged over the BATCH, leaving a
            # [num_q_networks]-shaped "per_sample_critic_loss" (a lie -- it was per-Q-network,
            # averaged over 6600 samples down to 2 values) that `.sum(dim=0)` two lines below
            # silently collapsed into a valid-looking scalar, so nothing before ever surfaced this:
            # `qf_loss` was actually sum_q(mean_n(x)) = (num_q_networks/batch_size) *
            # sum_n(mean_q(x)) -- for a 6600-sample batch and num_q_networks=2, a ~0.0003x
            # (~3300x too SMALL) scale versus the genuinely-per-sample loss the variable name and
            # the actor-loss path's own IDENTICAL-SHAPE reduction (`qf_value = q_values.mean(
            # dim=0)`, ~120 lines below -- CORRECTLY dim=0 on the same [num_q, batch]-shaped
            # tensor family) both imply was intended. `.mean(dim=0)` here makes this internally
            # consistent with that actor-side reduction and makes per_sample_critic_loss ACTUALLY
            # per-sample (shape [batch]), which is also what exposed the bug in the first place:
            # multiplying it by a genuinely per-sample skill_weight tensor is the first operation
            # in this codebase to ever require this tensor's shape to be correct instead of merely
            # summable to a scalar.
            #
            # CONSEQUENCE, not just a crash fix: this changes qf_loss's effective SCALE by
            # ~batch_size/num_q_networks (~3300x for a 6600-sample batch, 2 Q-networks) versus
            # every prior run's critic loss magnitude. A checkpoint resumed from BEFORE this fix
            # was trained under the OLD (much smaller) scale -- expect a real, one-time regime
            # shift in qf_loss/qf_max/critic grad norm on resume, not a bug in the fix itself.
            per_sample_critic_loss = critic_losses.mean(dim=0)
            if self.skill_replay_weights_enabled:
                per_sample_critic_loss = per_sample_critic_loss * data["skill_weight"]
            qf_loss = per_sample_critic_loss.sum(dim=0)

        q_optimizer.zero_grad(set_to_none=True)
        scaler.scale(qf_loss).backward()

        if self.is_multi_gpu:
            self._all_reduce_model_grads(qnet)

        scaler.unscale_(q_optimizer)
        if args.max_grad_norm > 0:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                qnet.parameters(),
                max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf"),
            )
        else:
            critic_grad_norm = torch.tensor(0.0, device=self.device)
        scaler.step(q_optimizer)
        scaler.update()
        alpha_loss = torch.tensor(0.0, device=self.device)
        # `and not actor_frozen`: see this method's own docstring for the measured windup this
        # prevents. Deliberately skips the ENTIRE block (no backward, no step) rather than
        # zeroing the gradient afterward -- Adam carries momentum state (exp_avg/exp_avg_sq), so
        # accumulating gradients through warmup and only suppressing the step would still leave a
        # primed optimizer that lurches on the first post-warmup update.
        if self.config.use_autotune and not actor_frozen:
            alpha_optimizer.zero_grad(set_to_none=True)
            with self._maybe_amp():
                if self.num_alpha_groups == 2:
                    # PER-GROUP masked mean, not a flat mean over the mixed batch: kick is a small
                    # minority of transitions (see UnifiedManager._build_task_mode_partition), so a
                    # naive flat .mean() would make kick's own log_alpha[1] converge toward its
                    # target far slower than locomotion's log_alpha[0] converges toward ITS target,
                    # purely because of how rarely kick samples appear per batch -- not a
                    # correctness bug (gradients still route to the right group's element via the
                    # gather, see torch.autograd's indexing backward), but a real, avoidable
                    # convergence-rate bias. Each group's own mean is over ONLY that group's own
                    # samples, then the two groups are averaged with EQUAL weight regardless of
                    # how many samples of each were in this batch.
                    is_kick_mask = data["is_kick"].bool()
                    per_sample_alpha = self.log_alpha.exp()[data["is_kick"]]
                    per_sample_target_entropy = self.target_entropy[data["is_kick"]]
                    per_sample_term = -per_sample_alpha * (next_state_log_probs.detach() + per_sample_target_entropy)
                    loco_mask = ~is_kick_mask
                    group_losses = []
                    if loco_mask.any():
                        group_losses.append(per_sample_term[loco_mask].mean())
                    if is_kick_mask.any():
                        group_losses.append(per_sample_term[is_kick_mask].mean())
                    alpha_loss = torch.stack(group_losses).mean()
                else:
                    alpha_loss = (-self.log_alpha.exp() * (next_state_log_probs.detach() + self.target_entropy)).mean()

            scaler.scale(alpha_loss).backward()

            if self.is_multi_gpu:
                if self.log_alpha.grad is not None:
                    torch.distributed.all_reduce(self.log_alpha.grad.data, op=torch.distributed.ReduceOp.SUM)
                    self.log_alpha.grad.data.copy_(self.log_alpha.grad.data / self.gpu_world_size)

            scaler.unscale_(alpha_optimizer)

            scaler.step(alpha_optimizer)
            scaler.update()

        return (
            rewards.mean(),
            critic_grad_norm.detach(),
            qf_loss.detach(),
            target_value_max.detach(),
            target_value_min.detach(),
            alpha_loss.detach(),
        )

    def _update_pol(
        self, data: TensorDict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        actor = self.actor
        qnet = self.qnet
        actor_optimizer = self.actor_optimizer
        scaler = self.scaler
        args = self.config

        with self._maybe_amp():
            critic_observations = data["critic_observations"]

            actions, log_probs = actor.get_actions_and_log_probs(data["observations"])
            # For logging, this is a bit wasteful though, but could be useful
            with torch.no_grad():
                _, _, log_std = actor(data["observations"])
                action_std = log_std.exp().mean()
                # Compute policy entropy (negative log probability)
                policy_entropy = -log_probs.mean()
                # 2026-07-28: per-group breakdown when per-task-mode entropy is active, so the
                # fix's actual effect (kick's own entropy/action_std moving independently of
                # locomotion's) is directly observable in wandb rather than only visible in the
                # pooled average these two scalars already were.
                self._last_alpha_group_metrics: dict[str, torch.Tensor] = {}
                if self.num_alpha_groups == 2:
                    is_kick_mask = data["is_kick"].bool()
                    per_sample_std = log_std.exp().mean(dim=-1)
                    per_sample_entropy = -log_probs
                    for name, mask in (("locomotion", ~is_kick_mask), ("kick", is_kick_mask)):
                        if mask.any():
                            self._last_alpha_group_metrics[f"action_std/{name}"] = per_sample_std[mask].mean()
                            self._last_alpha_group_metrics[f"policy_entropy/{name}"] = per_sample_entropy[mask].mean()

            q_outputs = qnet(critic_observations, actions)
            q_probs = F.softmax(q_outputs, dim=-1)
            q_values = qnet.get_value(q_probs)
            qf_value = q_values.mean(dim=0)
            # Per-sample alpha gather when per-task-mode entropy is active (see _update_main's
            # identical comment) -- detached, matching the original: the actor loss's entropy term
            # uses alpha as a fixed coefficient, alpha itself is only ever updated via alpha_loss.
            # A flat .mean() over the gathered per-sample values is CORRECT here (unlike
            # alpha_loss): each sample's own gradient contribution to the actor already uses its
            # own group's alpha, exactly mirroring how current_w_g(env) already weights the
            # shooting reward terms per-env before a shared mean/sum.
            alpha_detached = self.log_alpha.exp().detach()[data["is_kick"]] if self.num_alpha_groups == 2 else self.log_alpha.exp().detach()
            per_sample_actor_loss = alpha_detached * log_probs - qf_value
            # See _update_main's own comment on data["skill_weight"] -- same pre-materialized,
            # compile-safe pattern, same static-bool gate.
            if self.skill_replay_weights_enabled:
                per_sample_actor_loss = per_sample_actor_loss * data["skill_weight"]
            actor_loss = per_sample_actor_loss.mean()

            # Optional additional term: directly evaluate -Q(s, deterministic_action), where
            # deterministic_action is EXACTLY what deployment runs (Actor.forward's own `action`
            # output, tanh(mean)*scale+bias -- see config_types/algo.py's deterministic_loss_weight
            # docstring for the full motivation). Standard SAC's actor_loss above only ever scores
            # SAMPLED actions; nothing in it evaluates the mean directly, so this closes that gap
            # without relying on sigma collapsing first. Gated on the weight to skip the extra
            # forward pass entirely when disabled (default), zero behavior change.
            deterministic_loss = torch.zeros((), device=self.device)
            if args.deterministic_loss_weight > 0.0:
                deterministic_actions, _, _ = actor(data["observations"])
                q_det_outputs = qnet(critic_observations, deterministic_actions)
                q_det_probs = F.softmax(q_det_outputs, dim=-1)
                q_det_values = qnet.get_value(q_det_probs)
                qf_det_value = q_det_values.mean(dim=0)
                deterministic_loss = -qf_det_value.mean()
                actor_loss = actor_loss + args.deterministic_loss_weight * deterministic_loss

        actor_optimizer.zero_grad(set_to_none=True)
        scaler.scale(actor_loss).backward()

        if self.is_multi_gpu:
            self._all_reduce_model_grads(actor)

        scaler.unscale_(actor_optimizer)

        if args.max_grad_norm > 0:
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                actor.parameters(),
                max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf"),
            )
        else:
            actor_grad_norm = torch.tensor(0.0, device=self.device)
        scaler.step(actor_optimizer)
        scaler.update()
        return (
            actor_grad_norm.detach(),
            actor_loss.detach(),
            policy_entropy.detach(),
            action_std.detach(),
            deterministic_loss.detach(),
        )

    def _apply_l2sp_pull(self) -> None:
        """L2-SP (see FastSACConfig.l2sp_weight): pull the actor back toward the checkpoint it
        started from, once per actor update, immediately after that update's optimizer step.

        p += lr * l2sp_weight * (anchor - p)

        This is the SAME decoupled mechanism as AdamW's own weight_decay -- applied straight to the
        parameters rather than through the loss -- except the target is theta_anchor instead of 0,
        which is what makes `weight_decay` (0.001) a meaningful reference scale for choosing
        l2sp_weight. Deliberately NOT an added loss term: a loss-side L2 penalty gets rescaled
        per-parameter by Adam's second-moment normalization (exactly the coupling AdamW exists to
        remove), so its effective strength would drift with gradient statistics and would not be
        comparable to weight_decay at all.

        Lives OUT here, called from learn(), rather than at the end of _update_pol: _update_pol is
        wrapped in torch.compile (config.compile defaults to True), and in-place mutation of module
        parameters plus the self._last_l2sp_drift side-effect assignment below are exactly the kind
        of thing that forces a graph break or gets functionalized away. Same reason -- and the same
        _foreach_ style -- as the qnet_target soft update, which is likewise done in learn().

        No-op unless load() captured an anchor, which only happens when l2sp_weight > 0."""
        if self._l2sp_anchor is None:
            return
        with torch.no_grad():
            pull = torch._foreach_sub(self._l2sp_anchor, self._l2sp_params)
            torch._foreach_add_(
                self._l2sp_params, pull, alpha=self.config.actor_learning_rate * self.config.l2sp_weight
            )
            # Diagnostic only, never part of any objective: ||theta - theta_anchor||_2 over the
            # whole actor. Watch it next to the per-skill Kick_skills_{i} EMAs -- it should plateau
            # once the pull balances the gradients, rather than growing without bound.
            self._last_l2sp_drift = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(pull)))

    def _sample_and_prepare_batches(
        self, batch_size: int, num_updates: int, normalize_obs, normalize_critic_obs
    ) -> list[TensorDict]:
        """
        Sample a large batch once and split it into smaller batches for each update.
        This reduces sampling overhead by `num_updates` and normalization overhead by `num_updates`.
        """
        # Sample a large batch (batch_size * num_updates)
        large_batch_size = batch_size * num_updates
        large_data = self.rb.sample(large_batch_size)
        samples_per_update = batch_size * self.env.num_envs

        if self.config.use_symmetry:
            samples_per_update *= 2

            augmented_large_data: Dict[str, torch.Tensor | Dict[str, torch.Tensor]] = {"next": {}}

            augmented_large_data["observations"] = self.symmetry_utils.augment_observations(
                obs=large_data["observations"],
                env=self.env,
                obs_list=self.config.actor_obs_keys,
            )
            augmented_large_data["actions"] = self.symmetry_utils.augment_actions(actions=large_data["actions"])
            assert isinstance(augmented_large_data["next"], dict)
            augmented_large_data["next"]["observations"] = self.symmetry_utils.augment_observations(
                obs=large_data["next"]["observations"],
                env=self.env,
                obs_list=self.config.actor_obs_keys,
            )
            augmented_large_data["critic_observations"] = self.symmetry_utils.augment_observations(
                obs=large_data["critic_observations"],
                env=self.env,
                obs_list=self.config.critic_obs_keys,
            )
            augmented_large_data["next"]["critic_observations"] = self.symmetry_utils.augment_observations(
                obs=large_data["next"]["critic_observations"],
                env=self.env,
                obs_list=self.config.critic_obs_keys,
            )

            # Calculate augmentation factor and repeat non-augmented data
            observations_tensor = augmented_large_data["observations"]
            assert isinstance(observations_tensor, torch.Tensor), (
                "observations should be a Tensor after data augmentation"
            )
            num_aug = int(observations_tensor.shape[0] / large_data["next"]["rewards"].shape[0])
            augmented_large_data["next"]["rewards"] = large_data["next"]["rewards"].repeat(num_aug)  # type: ignore[index]
            augmented_large_data["next"]["dones"] = large_data["next"]["dones"].repeat(num_aug)  # type: ignore[index]
            augmented_large_data["next"]["truncations"] = large_data["next"]["truncations"].repeat(num_aug)  # type: ignore[index]
            augmented_large_data["next"]["effective_n_steps"] = large_data["next"]["effective_n_steps"].repeat(num_aug)  # type: ignore[index]
            # is_kick is a per-transition TASK label, unaffected by the left/right mirror
            # augmentation itself -- repeat like dones/truncations/effective_n_steps above, not
            # augmented like observations/actions.
            augmented_large_data["is_kick"] = large_data["is_kick"].repeat(num_aug)  # type: ignore[index]
            # Same reasoning as is_kick directly above: a per-transition TASK/SKILL label is
            # unaffected by the left/right mirror, so it repeats rather than being transformed.
            augmented_large_data["skill_id"] = large_data["skill_id"].repeat(num_aug)  # type: ignore[index]

            # Override large_data
            large_data = augmented_large_data

        # Normalize all data once
        large_data["observations"] = normalize_obs(large_data["observations"])
        large_data["next"]["observations"] = normalize_obs(large_data["next"]["observations"])
        large_data["critic_observations"] = normalize_critic_obs(large_data["critic_observations"])
        large_data["next"]["critic_observations"] = normalize_critic_obs(large_data["next"]["critic_observations"])

        # Split into smaller batches
        prepared_batches = []

        for i in range(num_updates):
            start_idx = i * samples_per_update
            end_idx = (i + 1) * samples_per_update

            is_kick_slice = large_data["is_kick"][start_idx:end_idx]
            skill_id_slice = large_data["skill_id"][start_idx:end_idx]

            # Per-skill replay weighting (see _skill_replay_weight's own docstring): computed HERE
            # from the raw slices, BEFORE batch_data exists, and folded into the SAME initial
            # TensorDict(...) constructor call below -- not added via a later batch_data["..."] =
            # assignment. 2026-08-15: a first version computed this correctly (verified by 14
            # passing unit tests covering this exact production shape) but assigned it into
            # batch_data via __setitem__ AFTER construction, and STILL hit
            # "size of tensor a (2) must match size of tensor b (6600)" inside the compiled
            # _update_main at real 3300-env scale, even though _update_main only ever read it as a
            # plain already-materialized tensor. skill_weight was the only TOP-LEVEL key in this
            # dict added post-construction (is_kick/skill_id are both in the initial dict; the
            # other post-construction assignment, next.critic_observations, is NESTED, not
            # top-level) -- suspected (not confirmed) TensorDict/torch.compile pytree-registration
            # difference between keys present at construction vs added via later __setitem__.
            # Folding it into the initial constructor call sidesteps that difference regardless of
            # whether this specific theory is the exact mechanism.
            skill_weight = (
                self._skill_replay_weight(
                    TensorDict({"skill_id": skill_id_slice, "is_kick": is_kick_slice}, batch_size=samples_per_update)
                )
                if self.skill_replay_weights_enabled
                else None
            )

            # Create a slice of the large batch
            batch_data = TensorDict(
                {
                    "observations": large_data["observations"][start_idx:end_idx],
                    "actions": large_data["actions"][start_idx:end_idx],
                    "is_kick": is_kick_slice,
                    "skill_id": skill_id_slice,
                    "next": {
                        "rewards": large_data["next"]["rewards"][start_idx:end_idx],
                        "dones": large_data["next"]["dones"][start_idx:end_idx],
                        "truncations": large_data["next"]["truncations"][start_idx:end_idx],
                        "observations": large_data["next"]["observations"][start_idx:end_idx],
                        "effective_n_steps": large_data["next"]["effective_n_steps"][start_idx:end_idx],
                    },
                    "critic_observations": large_data["critic_observations"][start_idx:end_idx],
                    **({"skill_weight": skill_weight} if skill_weight is not None else {}),
                },
                batch_size=samples_per_update,
            )
            batch_data["next"]["critic_observations"] = large_data["next"]["critic_observations"][start_idx:end_idx]

            prepared_batches.append(batch_data)

        return prepared_batches

    @staticmethod
    def _effective_obs_params(func_path: str, params: dict) -> dict:
        """Params dict with the term function's OWN defaults filled in, so a term that simply
        GAINED a parameter (with its default explicitly present in the new config) does not
        register as a change when the behavior is byte-identical.

        2026-08-18, caught live and it contaminated a real control run: adding ``distance_scale``
        to ``target_pos_b`` meant the current config carried ``params={"distance_scale": 0.0}``
        while an older checkpoint carried ``params={}``. Those are the SAME behavior (0.0 is that
        parameter's own "off" sentinel AND its signature default -- the function returns the raw
        offset either way), but a naive dict comparison flagged a shift, and the reset guard then
        un-normalized ``kick_target_pos_b`` -- 2 dims previously measured as ~51% of the entire
        fire-tick observation jump -- injecting the exact distribution shift the run existed to
        prove absent. Comparing EFFECTIVE params (defaults resolved on both sides) makes
        "parameter added at its default" correctly a no-op, while a genuine value change
        (0.0 -> 5.0) still registers.

        Falls back to the raw params on any resolution/introspection failure (a term whose func
        string no longer resolves, a C-extension callable with no signature, etc.) -- degrading to
        the previous over-eager behavior is acceptable; crashing inside ``load()`` is not.
        """
        try:
            func = resolve_callable(func_path, context="observation term")
            target = func.__init__ if isinstance(func, type) else func
            defaults = {
                name: p.default
                for name, p in inspect.signature(target).parameters.items()
                if p.default is not inspect.Parameter.empty
            }
        except Exception:
            return dict(params)
        return {**defaults, **params}

    def _detect_shifted_obs_terms(self, torch_checkpoint: dict, obs_manager) -> dict[str, dict[str, dict]]:
        """Shared detector behind BOTH the normalizer-reset guard and the warm-start blend (FIX 6)
        -- one diff, two possible responses to it, never two independent (and possibly
        disagreeing) implementations of "did this term's config change".

        Returns ``{group_name: {term_name: old_raw_dict}}`` for every term present in BOTH the
        checkpoint's saved config and the current env's resolved config whose
        ``(func, params, scale, task_mode, clip)`` differ -- ``func`` compared by identity
        (2026-08-22: a term can keep its name and an identical effective params dict while its
        underlying function is swapped for one with entirely different output semantics; see this
        method's own inline comment on the real, previously-undetected case that motivated adding
        it). ``old_raw_dict`` is the checkpoint's own serialized term dict (has ``func``/
        ``params``/``scale``/``task_mode``/``clip`` keys, among others -- see a real checkpoint's
        ``experiment_config.observation.groups`` for the exact shape). Empty dict (never ``None``)
        when the checkpoint predates this metadata, or nothing changed -- callers do not need a
        separate "was there even anything to compare" branch.

        Terms present in only one side (genuinely added/removed, a WIDTH change) are left alone --
        that case already fails loudly and separately at ``load_state_dict`` (shape mismatch), so
        it needs no handling here or in either caller.
        """
        old_groups = (torch_checkpoint.get("experiment_config") or {}).get("observation", {}).get("groups")
        if not old_groups:
            return {}
        out: dict[str, dict[str, dict]] = {}
        for group_name, new_group_cfg in obs_manager.cfg.groups.items():
            if group_name not in old_groups:
                continue
            old_terms: dict = old_groups[group_name]["terms"]
            new_terms = new_group_cfg.terms
            changed: dict[str, dict] = {}
            for name in sorted(set(old_terms) & set(new_terms)):  # sorted == concatenation order
                old_t, new_t = old_terms[name], new_terms[name]
                # Effective params on BOTH sides -- see _effective_obs_params for why a raw dict
                # comparison produces false positives (and what one already cost). `func` itself
                # is now PART of the key (2026-08-22, found while auditing the azimuth-aim
                # refactor's checkpoint compatibility): a term can keep its NAME and an IDENTICAL
                # params dict while its underlying function is swapped for one with completely
                # different output semantics -- exactly what config_values/unified/g1/
                # observation.py's kick_target_pos_b did (target_pos_b, raw world-frame metres ->
                # kick_aim_command, a bounded normalized command -- both registered with
                # params={"distance_scale": 0.0} since kick_aim_command was deliberately given the
                # same signature). Without `func` in the key, that swap was INVISIBLE to this
                # detector: (effective_params, scale, task_mode, clip) came out byte-identical on
                # both sides, so neither the reset guard nor the FIX-6 blend would ever fire, and a
                # warm-started old checkpoint would silently apply stale metres-scale normalizer
                # stats to the new bounded output for the rest of training -- the exact class of
                # shock this whole mechanism exists to prevent, undetected by it.
                new_key = (
                    new_t.func,
                    FastSACAgent._effective_obs_params(new_t.func, dict(new_t.params)),
                    new_t.scale,
                    new_t.task_mode,
                    new_t.clip,
                )
                old_key = (
                    old_t.get("func", new_t.func),
                    FastSACAgent._effective_obs_params(old_t.get("func", new_t.func), dict(old_t.get("params") or {})),
                    old_t.get("scale", 1.0),
                    old_t.get("task_mode"),
                    old_t.get("clip"),
                )
                if new_key != old_key:
                    changed[name] = old_t
            if changed:
                out[group_name] = changed
        return out

    def _reset_normalizer_slots_for_shifted_obs_terms(self, torch_checkpoint: dict) -> None:
        """Reset ``EmpiricalNormalization``'s running mean/var to (0, 1) for exactly the column
        slices of terms ``_detect_shifted_obs_terms`` flags -- exact no-op when nothing relevant
        changed. This is the FALLBACK path: ``load()`` only calls this when
        ``warm_start_obs_ramp_steps`` (FIX 6, ``_configure_warm_start_obs_blend`` below) is at its
        0.0 off-default, since the two would otherwise fight each other -- see FIX 6's own
        docstring (``MultiSkillConfig.warm_start_obs_ramp_steps``) for why.

        WHY THIS EXISTS: EmpiricalNormalization's mean/std are fit to whatever a term's saved
        checkpoint actually emitted. If that term's UNITS change (e.g. ``distance_scale`` turns raw
        metres into a bounded tanh), the stale statistics compute a wildly wrong z-score for every
        new reading. Resetting the affected slots to (0, 1) makes the first post-resume reading
        pass through close to unchanged instead of through a stale multi-sigma-wrong scale.

        MEASURED LIMITATION (2026-08-18, not merely theoretical): this guard fires correctly but
        does NOT by itself prevent a real shock when the changed terms are POPULATION changes
        rather than pure units changes (e.g. ``obs_ball_always_visible``, a term going from
        hard-zero-for-most-envs to genuinely live) -- confirmed on a real run, action_std still
        spiked 0.052->0.244 (4.7x) in one logging interval despite this guard resetting exactly the
        right 10 terms. A per-feature normalizer cannot correct JOINT/conditional structure, only
        marginal scale. FIX 6 is the response built for that case.

        SEPARATE LIMITATION, stated plainly regardless of which case applies:
        ``EmpiricalNormalization.count`` is a single scalar shared across the WHOLE feature vector
        (see fast_sac_utils.py), not per-dimension -- there is no way to also reset "how confident"
        the normalizer is for just these slots without changing that class's structure. So reset
        slots start at the right SCALE immediately but adapt to genuinely new data more SLOWLY than
        a truly fresh normalizer would, since each subsequent update is weighted by the large
        pre-existing global count.

        Set HOLOSOMA_SKIP_OBS_NORMALIZER_RESET=1 to disable (e.g. to deliberately reproduce the
        pre-2026-08-18 behavior for a controlled comparison).
        """
        if os.environ.get(_SKIP_OBS_NORMALIZER_RESET_ENV_VAR, "").strip().lower() in _OBS_NORMALIZER_RESET_TRUTHY:
            return
        obs_manager = getattr(self.unwrapped_env, "observation_manager", None)
        if obs_manager is None:
            return
        changed_by_group = self._detect_shifted_obs_terms(torch_checkpoint, obs_manager)
        if not changed_by_group:
            return
        group_normalizers = {"actor_obs": self.obs_normalizer, "critic_obs": self.critic_obs_normalizer}

        for group_name, changed in changed_by_group.items():
            normalizer = group_normalizers.get(group_name)
            if normalizer is None:
                continue
            new_terms = obs_manager.cfg.groups[group_name].terms

            # Width probe, read-only: reuses the exact technique this project's own handoff-
            # discontinuity investigation validated (memory `stage-d-handoff-observation-
            # discontinuity`) for computing per-term values with NO side effects on real env
            # state -- temporarily swap ONLY this group's cfg entry to concatenate=False (so
            # compute_group returns a dict, not a fused tensor) and enable_noise=False (isolates
            # the structural shape from injected noise), call with modify_history=False (so
            # _apply_delay's ring buffer is never mutated), then restore the original entry.
            # cfg.groups is a plain dict holding FROZEN dataclasses -- swap the entry, not the
            # instance.
            original_group_cfg = obs_manager.cfg.groups[group_name]
            probe_cfg = dataclasses.replace(original_group_cfg, concatenate=False, enable_noise=False)
            obs_manager.cfg.groups[group_name] = probe_cfg
            try:
                per_term = obs_manager.compute_group(group_name, modify_history=False)
            finally:
                obs_manager.cfg.groups[group_name] = original_group_cfg

            offset = 0
            reset_ranges: list[tuple[str, int, int]] = []
            for name in sorted(new_terms):  # exact concatenation order compute_group itself uses
                width = per_term[name].shape[-1]
                if name in changed:
                    reset_ranges.append((name, offset, offset + width))
                offset += width

            for _name, lo, hi in reset_ranges:
                normalizer._mean[:, lo:hi] = 0.0
                normalizer._var[:, lo:hi] = 1.0
                normalizer._std[:, lo:hi] = 1.0

            logger.warning(
                f"[FastSACAgent.load] obs distribution shift detected in group '{group_name}': "
                f"{sorted(changed)} -- (params, scale, task_mode, clip) differ from what this "
                f"checkpoint was saved with. Reset EmpiricalNormalization mean/var to (0, 1) for "
                f"these slots so training doesn't start from wrong-unit z-scores. "
                f"Set {_SKIP_OBS_NORMALIZER_RESET_ENV_VAR}=1 to disable this."
            )

    def _configure_warm_start_obs_blend(self, torch_checkpoint: dict, ramp_steps: float) -> None:
        """FIX 6 (2026-08-18): the response to the normalizer-reset guard's measured limitation --
        see ``MultiSkillConfig.warm_start_obs_ramp_steps`` and ``ObservationManager.
        set_warm_start_blend`` for the full mechanism and rationale. This method's own job is
        narrow: run the shared detector, reconstruct each changed term's CHECKPOINT-TIME config as
        a real ``ObsTermCfg`` (the checkpoint stores it as a plain dict; the manager needs the
        dataclass to call the same term function/scale/clip machinery with it), and hand the result
        to the observation manager -- no detection logic of its own, no normalizer access at all
        (this and the reset guard are mutually exclusive per ``load()``'s own dispatch, not merged
        here).
        """
        obs_manager = getattr(self.unwrapped_env, "observation_manager", None)
        if obs_manager is None:
            return
        changed_by_group = self._detect_shifted_obs_terms(torch_checkpoint, obs_manager)
        if not changed_by_group:
            return
        old_cfgs_by_group: dict[str, dict[str, ObsTermCfg]] = {
            group_name: {
                name: ObsTermCfg(
                    func=old_t["func"],
                    params=dict(old_t.get("params") or {}),
                    scale=old_t.get("scale", 1.0),
                    noise=old_t.get("noise", 0.0),
                    noise_range_coefficient=old_t.get("noise_range_coefficient", 0.0),
                    delay_step_range=old_t.get("delay_step_range"),
                    clip=old_t.get("clip"),
                    task_mode=old_t.get("task_mode"),
                )
                for name, old_t in changed.items()
            }
            for group_name, changed in changed_by_group.items()
        }
        obs_manager.set_warm_start_blend(old_cfgs_by_group, ramp_steps)
        all_changed = sorted({n for terms in changed_by_group.values() for n in terms})
        logger.warning(
            f"[FastSACAgent.load] obs distribution shift detected: {all_changed} -- (params, "
            f"scale, task_mode, clip) differ from what this checkpoint was saved with. Blending "
            f"from the checkpoint's own old-style values toward the new config over "
            f"{ramp_steps:.0f} steps (warm_start_obs_ramp_steps) instead of cutting over instantly."
        )

    def _restore_critic_support(self) -> None:
        """Re-derive the distributional critic's ``q_support`` from THIS run's live config after a
        checkpoint load, instead of silently inheriting the checkpoint's own.

        Why this is needed (2026-08-21, verified against a real saved checkpoint): ``q_support`` is
        a REGISTERED BUFFER (``register_buffer("q_support", torch.linspace(v_min, v_max,
        num_atoms))`` in agents/fast_sac/fast_sac.py), so it is serialized into
        ``qnet_state_dict``/``qnet_target_state_dict`` like any parameter. ``load_state_dict``
        therefore OVERWRITES the freshly-constructed support with whatever the checkpoint was
        trained under -- meaning a run that sets configs/*.yaml's ``critic_v_min``/``critic_v_max``
        and then RESUMES would silently train under the OLD support, with no error and no log line.
        The change would look applied (it IS applied in the config, and in the saved
        holosoma_config.yaml) while having zero effect.

        Note the asymmetry with ``num_atoms``: that one changes tensor SHAPES (the critic head is
        ``nn.Linear(hidden_dim // 4, num_atoms)``), so a mismatch already fails loudly inside
        ``load_state_dict`` above -- there is nothing to rescue here and nothing to silently
        mis-apply. Only ``v_min``/``v_max`` are shape-compatible-but-semantically-different, which
        is exactly what makes them the silent case worth guarding.

        Rescaling the support is NOT free and the warning below says so: each atom's learned
        probability keeps its index but that index now denotes a DIFFERENT return value, so the
        critic's value estimates are effectively rescaled and it must re-fit. The actor is
        untouched, which is the part worth preserving across the change."""
        expected = torch.linspace(
            self.config.v_min, self.config.v_max, self.config.num_atoms, device=self.device
        )
        for name, net in (("qnet", self.qnet), ("qnet_target", self.qnet_target)):
            loaded = getattr(net, "q_support", None)
            if loaded is None:
                continue
            if loaded.shape == expected.shape and torch.allclose(loaded, expected):
                continue
            logger.warning(
                f"[critic-support] {name}: checkpoint was trained with value support "
                f"[{loaded[0].item():.3f}, {loaded[-1].item():.3f}] ({loaded.numel()} atoms) but "
                f"this run's config specifies [{self.config.v_min}, {self.config.v_max}] "
                f"({self.config.num_atoms} atoms). Overriding with the CONFIG's support -- without "
                "this the config change would be a silent no-op. Expect the critic's value "
                "estimates to be rescaled and to need re-fitting for a while; the actor is "
                "unaffected. See MultiSkillConfig.critic_v_min's docstring."
            )
            net.q_support.copy_(expected)

    def load(self, ckpt_path: str | None) -> None:
        if not ckpt_path:
            return
        # Load checkpoint if specified
        torch_checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        # Handle DDP-wrapped models
        actor_state_dict = torch_checkpoint["actor_state_dict"]
        qnet_state_dict = torch_checkpoint["qnet_state_dict"]

        self.actor.load_state_dict(actor_state_dict)
        self.qnet.load_state_dict(qnet_state_dict)

        # L2-SP: snapshot the actor's just-loaded weights as the anchor this run is regularized
        # toward (see FastSACConfig.l2sp_weight). Captured HERE, after load_state_dict, rather than
        # by re-reading the file: this way it is by construction exactly the tensor set training
        # starts from, including any DDP unwrapping/dtype coercion load_state_dict applied.
        # Keyed by parameter NAME, not positional order, so it stays correct if the actor's
        # submodules are ever reordered. Cloned + detached: the anchor must NOT alias live
        # parameters (that would make the pull below a no-op against a moving target).
        if self.config.l2sp_weight > 0.0:
            named = list(self.actor.named_parameters())
            self._l2sp_params = [p for _, p in named]
            self._l2sp_anchor = [p.detach().clone() for _, p in named]
            logger.info(
                f"L2-SP active: weight={self.config.l2sp_weight}, anchored to {ckpt_path} "
                f"({len(self._l2sp_anchor)} actor parameter tensors)."
            )

        self.obs_normalizer.load_state_dict(torch_checkpoint["obs_normalizer_state"])
        self.critic_obs_normalizer.load_state_dict(torch_checkpoint["critic_obs_normalizer_state"])
        # 2026-08-18: obs-distribution-shift response. Root-caused on a real run
        # (20260818_033003-stageD-1skill-obs-fixes): warm-starting into a task_config whose
        # observation terms changed left the just-loaded normalizer applying its OLD (now wrong)
        # mean/std to the NEW values -- measured action_std spiking 0.046->0.313 (6.8x) and
        # kick_topple_frac 0->0.64 within the first few thousand steps after resume. Same
        # underlying category as the Stage B->C ball-observation shock this project hit once
        # before (`stagec_obs_normalizer_shock.md`), just triggered by a different field.
        #
        # TWO responses exist, MUTUALLY EXCLUSIVE by design (see FIX 6's own docstring,
        # MultiSkillConfig.warm_start_obs_ramp_steps, for why running both would fight each
        # other): warm_start_obs_ramp_steps > 0.0 selects FIX 6 (blend toward the checkpoint's own
        # old-style observation values, fading to the new config over N steps -- the response
        # measured to actually work for POPULATION changes like obs_ball_always_visible, not just
        # units changes); its 0.0 default falls back to the normalizer-reset guard (measured
        # sufficient for a pure units change like obs_target_pos_distance_scale, but NOT for a
        # population change -- see that method's own docstring for the measurement). Both are
        # exact no-ops when nothing in the observation config actually changed since this
        # checkpoint was saved. Must run AFTER both load_state_dict calls above (both can
        # override state those just loaded) and BEFORE anything reads the normalizer/observation
        # manager for real.
        warm_start_obs_ramp_steps = float(getattr(self.unwrapped_env, "_warm_start_obs_ramp_steps", 0.0))
        if warm_start_obs_ramp_steps > 0.0:
            self._configure_warm_start_obs_blend(torch_checkpoint, warm_start_obs_ramp_steps)
        else:
            self._reset_normalizer_slots_for_shifted_obs_terms(torch_checkpoint)
        self.qnet_target.load_state_dict(torch_checkpoint["qnet_target_state_dict"])
        self._restore_critic_support()
        # 2026-07-28: log_alpha's SHAPE can change across a resume (num_alpha_groups 1<->2, see
        # FastSACConfig.kick_target_entropy_ratio's docstring) -- the raw VALUE copy below
        # broadcasts safely (an old [1]-shaped checkpoint's single value seeds every group
        # identically; verified: torch.Tensor.copy_ broadcasts [1]->[N] but correctly raises on
        # [N]->[1] rather than silently truncating). Adam's OWN internal moment buffers
        # (exp_avg/exp_avg_sq inside alpha_optimizer's state) do NOT broadcast the same way --
        # load_state_dict() itself doesn't validate shapes, but the optimizer's first .step() after
        # loading a mismatched-shape buffer raises. Detect the mismatch here and skip loading
        # alpha_optimizer's state in that case (falls back to Adam's normal fresh-parameter
        # initialization -- exactly what already happens for any newly-added parameter, safe by
        # construction) rather than letting it crash on the next training step.
        checkpoint_log_alpha = torch_checkpoint["log_alpha"]
        self.log_alpha.data.copy_(checkpoint_log_alpha.to(self.device))
        self.actor_optimizer.load_state_dict(torch_checkpoint["actor_optimizer_state_dict"])
        self.q_optimizer.load_state_dict(torch_checkpoint["q_optimizer_state_dict"])
        if checkpoint_log_alpha.shape == self.log_alpha.shape:
            self.alpha_optimizer.load_state_dict(torch_checkpoint["alpha_optimizer_state_dict"])
        else:
            logger.warning(
                f"[resume] checkpoint's log_alpha shape {tuple(checkpoint_log_alpha.shape)} != "
                f"current {tuple(self.log_alpha.shape)} (num_alpha_groups changed across this "
                "resume) -- log_alpha's VALUE was still broadcast-copied above, but "
                "alpha_optimizer's Adam moment state does NOT carry over (would crash on the next "
                "step otherwise); it starts fresh instead, same as any newly-added parameter."
            )
        self.scaler.load_state_dict(torch_checkpoint["grad_scaler_state_dict"])
        self.global_step = torch_checkpoint["global_step"]
        self._restore_env_state(torch_checkpoint.get("env_state"))

    def learn(self) -> None:
        args = self.config
        device = self.device
        # L2-SP is anchored to the checkpoint this run resumed from (captured in load()); see
        # _validate_l2sp_anchor's own docstring for why this fails loudly here rather than
        # silently regularizing toward a randomly initialized actor.
        _validate_l2sp_anchor(args.l2sp_weight, self._l2sp_anchor)
        if args.compile:
            update_main = torch.compile(self._update_main)
            update_pol = torch.compile(self._update_pol)
            policy = torch.compile(self.policy)
            normalize_obs = torch.compile(self.obs_normalizer.forward)
            normalize_critic_obs = torch.compile(self.critic_obs_normalizer.forward)
        else:
            update_main = self._update_main
            update_pol = self._update_pol
            policy = self.policy
            normalize_obs = self.obs_normalizer.forward
            normalize_critic_obs = self.critic_obs_normalizer.forward
        qnet = self.qnet
        qnet_target = self.qnet_target
        env = self.env
        rb = self.rb

        obs, critic_obs = env.reset_with_critic_obs()
        critic_obs = torch.as_tensor(critic_obs, device=device, dtype=torch.float)

        dones = None
        # Initialize metrics that might not be updated every step
        policy_entropy = torch.tensor(0.0, device=device)
        action_std = torch.tensor(0.0, device=device)
        actor_loss = torch.tensor(0.0, device=device)
        actor_grad_norm = torch.tensor(0.0, device=device)
        deterministic_loss = torch.tensor(0.0, device=device)
        pbar = tqdm.tqdm(total=args.num_learning_iterations, initial=self.global_step)

        # Critic warmup (see FastSACConfig.critic_warmup_iters for the full rationale). Counted
        # from THIS process's own starting global_step, not from 0 -- self.global_step is loaded
        # from the checkpoint on a resume and does NOT reset (unlike env.common_step_counter, which
        # the reward ramp uses for exactly this reason). 0 (default) => warmup_end_step ==
        # warmup_start_step, so `self.global_step >= warmup_end_step` is true from the first
        # iteration and every existing run is bit-identical.
        warmup_start_step = self.global_step
        warmup_end_step = warmup_start_step + args.critic_warmup_iters
        if args.critic_warmup_iters > 0 and self.is_main_process:
            logger.info(
                f"[critic-warmup] Actor frozen for {args.critic_warmup_iters} steps (global_step "
                f"{warmup_start_step} -> {warmup_end_step}): critic-only updates, replay buffer "
                "filling from the current (frozen) actor's rollout."
            )
        _warmup_done_logged = False

        while self.global_step <= args.num_learning_iterations:
            # Synchronize curriculum metrics across GPUs before rollout
            if self.is_multi_gpu:
                self._synchronize_curriculum_metrics()

            with self.logging_helper.record_collection_time():
                with torch.no_grad(), self._maybe_amp():
                    norm_obs = normalize_obs(obs, update=False)
                    actions = policy(obs=norm_obs, dones=dones)

                next_obs, rewards, dones, infos = env.step(actions.float())
                truncations = infos["time_outs"]

                # Update episode stats using logging helper
                self.logging_helper.update_episode_stats(rewards, dones, infos)

                next_critic_obs = infos["observations"]["critic"]

                # Compute 'true' next_obs and next_critic_obs for saving
                true_next_obs = torch.where(
                    truncations[:, None] > 0, infos["observations"]["final"]["actor_obs"], next_obs
                )
                true_next_critic_obs = torch.where(
                    truncations[:, None] > 0,
                    infos["observations"]["final"]["critic_obs"],
                    next_critic_obs,
                )
                # 2026-07-28: which SAC entropy group this transition belongs to (see
                # FastSACConfig.kick_target_entropy_ratio's docstring). task_mode is env-permanent
                # (never changes within/across episodes -- UnifiedManager._build_task_mode_partition),
                # so capturing it here (rather than at sample time) is unambiguous; duck-typed via
                # getattr since only UnifiedManager-family envs have a task_mode at all -- every
                # other env (locomotion-only/WBT-only) gets an all-zero (all-locomotion-group)
                # tensor, which is simply never consulted downstream when kick_target_entropy_ratio
                # is unset (the default). LOCOMOTION is 0 in TaskMode (unified_manager.py); "not
                # locomotion" rather than importing the enum, since a 2-value task_mode only needs
                # that one fixed point to stay stable, not KICK's own numeric value.
                task_mode = getattr(env, "task_mode", None)
                is_kick = (
                    (task_mode != 0).long()
                    if task_mode is not None
                    else torch.zeros(env.num_envs, dtype=torch.long, device=device)
                )

                # 2026-08-15 sibling of is_kick above, for FastSACConfig.skill_replay_weights.
                # env.skill_id is fixed-for-life per env (UnifiedManager._build_task_mode_partition
                # assigns it once and _resample_task_mode only ever re-copies the same fixed
                # partition value), so unlike is_kick -- which reads the LIVE task_mode and does
                # legitimately flip mid-episode under the Stage D handoff -- this label is stable
                # for the whole life of the env. Zeros fallback for any env class with no skill
                # concept (locomotion-only/WBT-only), mirroring is_kick's own fallback.
                skill_id_t = getattr(env, "skill_id", None)
                skill_id_t = (
                    skill_id_t.long()
                    if skill_id_t is not None
                    else torch.zeros(env.num_envs, dtype=torch.long, device=device)
                )

                transition = TensorDict(
                    {
                        "observations": obs,
                        "actions": torch.as_tensor(actions, device=device, dtype=torch.float),
                        "is_kick": is_kick,
                        "skill_id": skill_id_t,
                        "next": {
                            "observations": true_next_obs,
                            "rewards": torch.as_tensor(rewards, device=device, dtype=torch.float),
                            "truncations": truncations.long(),
                            "dones": dones.long(),
                        },
                    },
                    batch_size=(env.num_envs,),
                    device=device,
                )
                transition["critic_observations"] = critic_obs
                transition["next"]["critic_observations"] = true_next_critic_obs

                obs = next_obs
                critic_obs = next_critic_obs

                rb.extend(transition)

            # NOTE: args.batch_size is the global batch size
            batch_size = max(args.batch_size // env.num_envs // self.gpu_world_size, 1)
            warmup_done = self.global_step >= warmup_end_step
            if warmup_done and not _warmup_done_logged and args.critic_warmup_iters > 0:
                if self.is_main_process:
                    logger.info(
                        f"[critic-warmup] Complete at global_step={self.global_step} -- actor "
                        "updates resuming."
                    )
                _warmup_done_logged = True
            if self.global_step > args.learning_starts:
                with self.logging_helper.record_learn_time():
                    # Use batched sampling: sample once, normalize once, split into updates
                    prepared_batches = self._sample_and_prepare_batches(
                        batch_size, args.num_updates, normalize_obs, normalize_critic_obs
                    )
                    for i, data in enumerate(prepared_batches):
                        # Data is already normalized, just run the updates.
                        # actor_frozen=not warmup_done (2026-08-28): suppresses the alpha
                        # auto-tune step while the actor is frozen -- see _update_main's own
                        # docstring for the measured 142,000x windup this prevents. Under
                        # args.compile this bool is a torch.compile guard, so it triggers exactly
                        # ONE recompile when warmup ends (it never flips back), which is a
                        # negligible one-off against a 5000-step warmup.
                        (
                            buffer_rewards,
                            critic_grad_norm,
                            qf_loss,
                            qf_max,
                            qf_min,
                            alpha_loss,
                        ) = update_main(data, actor_frozen=not warmup_done)
                        if args.num_updates > 1:
                            if warmup_done and i % args.policy_frequency == 1:
                                actor_grad_norm, actor_loss, policy_entropy, action_std, deterministic_loss = (
                                    update_pol(data)
                                )
                                self._apply_l2sp_pull()
                        elif warmup_done and self.global_step % args.policy_frequency == 0:
                            actor_grad_norm, actor_loss, policy_entropy, action_std, deterministic_loss = update_pol(
                                data
                            )
                            self._apply_l2sp_pull()

                        # Accumulate training metrics for smoother logging
                        current_metrics = {
                            "actor_loss": actor_loss,
                            "qf_loss": qf_loss,
                            "qf_max": qf_max,
                            "qf_min": qf_min,
                            "actor_grad_norm": actor_grad_norm,
                            "critic_grad_norm": critic_grad_norm,
                            "buffer_rewards": buffer_rewards,
                            "alpha_loss": alpha_loss,
                            # combined/pooled value, unchanged meaning -- kept for anyone's existing
                            # dashboards/queries even when the 2-group split below is also active.
                            "alpha_value": self.log_alpha.exp().detach().mean(),
                            "deterministic_loss": deterministic_loss,
                            "policy_entropy": policy_entropy,
                            "action_std": action_std,
                        }
                        if self.num_alpha_groups == 2:
                            alpha_per_group = self.log_alpha.exp().detach()
                            current_metrics["alpha_value/locomotion"] = alpha_per_group[0]
                            current_metrics["alpha_value/kick"] = alpha_per_group[1]
                            current_metrics.update(getattr(self, "_last_alpha_group_metrics", {}))
                        # Gated so a disabled L2-SP adds no new wandb series at all (rather than a
                        # flat-zero one), keeping existing dashboards byte-identical when off.
                        if self._l2sp_anchor is not None:
                            current_metrics["l2sp_drift"] = self._last_l2sp_drift
                        self.training_metrics.add(current_metrics)

                        with torch.no_grad():
                            src_ps = [p.data for p in qnet.parameters()]
                            tgt_ps = [p.data for p in qnet_target.parameters()]
                            torch._foreach_mul_(tgt_ps, 1.0 - args.tau)
                            torch._foreach_add_(tgt_ps, src_ps, alpha=args.tau)

                if self.global_step % args.logging_interval == 0:
                    with torch.no_grad():
                        # Use accumulated training metrics for smoother logging (reduces noise)
                        accumulated_metrics = self.training_metrics.mean_and_clear()

                        # Convert tensor values to float for logging
                        loss_dict = {}
                        for key, value in accumulated_metrics.items():
                            if isinstance(value, torch.Tensor):
                                loss_dict[key] = value.item()
                            else:
                                loss_dict[key] = float(value)

                        # Add current env rewards (not part of training loop accumulation)
                        loss_dict["env_rewards"] = rewards.mean().item()

                    # Use logging helper
                    self.logging_helper.post_epoch_logging(it=self.global_step, loss_dict=loss_dict, extra_log_dicts={})
                if args.save_interval > 0 and self.global_step > 0 and self.global_step % args.save_interval == 0:
                    if self.is_main_process:
                        logger.info(f"Saving model at global step {self.global_step}")
                        onnx_path = os.path.join(self.log_dir, f"model_{self.global_step:07d}.onnx")
                        self.save(os.path.join(self.log_dir, f"model_{self.global_step:07d}.pt"))
                        self.export(onnx_file_path=onnx_path)
                        self._maybe_start_mujoco_kick_rollout(onnx_path)
                        self._maybe_start_mujoco_walk_rollout(onnx_path)
                        self._maybe_start_mujoco_kick_handoff_rollout(onnx_path)
                        self._maybe_start_mujoco_survival_scan(onnx_path)
                        self._maybe_start_mujoco_kick_to_loco_flip_scan(onnx_path)
                        self._maybe_start_mujoco_loco_to_kick_handoff_scan(onnx_path)
                if self.is_main_process:
                    self._drain_mujoco_kick_rollout_queue()
                    self._drain_mujoco_walk_rollout_queue()
                    self._drain_mujoco_kick_handoff_rollout_queue()
                    self._drain_mujoco_survival_scan_queue()
                    self._drain_mujoco_kick_to_loco_flip_scan_queue()
                    self._drain_mujoco_loco_to_kick_handoff_scan_queue()

            # Avoid global_step being incremented beyond args.num_learning_iterations, so that the final checkpoint is
            # saved at exactly args.num_learning_iterations. In the `while` condition, we check for self.global_step <=
            # args.num_learning_iterations, so that we have complete logging data at the final step too (assuming
            # `args.num_learning_iterations` is a multiple of `args.logging_interval`).
            if self.global_step >= args.num_learning_iterations:
                break
            self.global_step += 1
            pbar.update(1)

        if self.is_main_process:
            self.save(os.path.join(self.log_dir, f"model_{self.global_step:07d}.pt"))
            self.export(onnx_file_path=os.path.join(self.log_dir, f"model_{self.global_step:07d}.onnx"))

    # ------------------------------------------------------------------
    # MuJoCo sim2sim kick rollout (holosoma.record_mujoco_kick_rollout) -- an early-warning wandb
    # video signal for the PhysX<->MuJoCo sim2sim gap. Runs entirely out-of-process (a subprocess
    # in a separate conda env) so it never touches this process's CUDA/torch state.
    # ------------------------------------------------------------------

    def _mujoco_rollout_gate_open(self, every_n_saves: int) -> bool:
        """Shared cadence/eligibility gate for ALL FOUR periodic MuJoCo sim2sim checks (kick
        rollout, kick-handoff/walk-then-trigger rollout, walk rollout, survival scan): feature
        enabled, this run actually trains kicking, and this checkpoint lands on the configured
        multiple of save_interval. The walk rollout deliberately reuses the KICK cadence knob
        (`mujoco_kick_rollout_every_n_saves`) rather than getting its own -- both are "is this
        checkpoint worth an expensive sim2sim check" signals on the same schedule, and the user
        asked for a fixed 4-rollout bundle, not an independently configurable 4th.

        2026-08-23 bugfix: "this run actually trains kicking" has TWO independent sources, and
        this gate used to check only one of them. N-skill mode (any skill setting
        `motion_training_ratio`) computes its own locomotion/kick partition entirely independently
        of `configs/skill_mix.yaml`'s `kick_probability`
        (`UnifiedManager._build_task_mode_partition`'s `if ratios: ... return` branch never
        reaches the `kick_probability`-based draw below it) -- but `self._kick_probability` is set
        UNCONDITIONALLY from `kick_probability` regardless of which path actually ran
        (`unified_manager.py`, `_kick_probability = float(getattr(..., "kick_probability", 0.5))`
        at `__init__`-time), so it stays stale at whatever `skill_mix.yaml` happened to hold --
        0.0 for any N-skill run launched after a Stage A run left it there, even while the N-skill
        partition itself was genuinely training kicking on a real fraction of envs (confirmed live:
        a run with `skill_mix.yaml` at 0.0 and `motion_training_ratio: 0.9` logged "0.450 of all
        envs permanently dedicated to a motion skill" at startup, yet every sim2sim mechanism
        sharing this gate silently never fired for its entire run). Checking BOTH sources fixes
        this for N-skill runs without changing legacy behavior at all -- an empty
        `_skill_motion_training_ratios` list sums to 0.0, falling through to the exact same
        `_kick_probability` check this gate always used."""
        if every_n_saves <= 0:
            return False
        skill_ratios_sum = sum(getattr(self.unwrapped_env, "_skill_motion_training_ratios", []))
        kick_probability = getattr(self.unwrapped_env, "_kick_probability", 0.0)
        if kick_probability <= 0.0 and skill_ratios_sum <= 0.0:
            return False  # e.g. Stage A -- no kick behavior to validate, under either mechanism
        return self.global_step % (self.config.save_interval * every_n_saves) == 0

    def _kick_aim_info_per_motion(self) -> "tuple[list[bool], float] | None":
        """``(kick_aim_enabled_per_motion, kick_aim_theta_ref_deg)`` read from the live
        MotionCommand state, or ``None`` if this run has no ball -- same resolution
        ``_maybe_start_mujoco_survival_scan`` already does inline (see that method's own comment
        for why ``kick_aim_theta_ref_deg`` has no per-motion table), factored out here since BOTH
        kick-rollout call sites (settle-then-trigger and walk-then-trigger) need it identically.

        2026-08-23 bugfix: without this, ``record_kick_rollout`` fed the pre-azimuth-refactor raw
        world-frame ``target_pos_b`` transform into obs[157:159] for EVERY checkpoint, including
        every currently-trained one (``kick_aim_enabled=True`` on every skill in this project) --
        a ~15-18x out-of-distribution magnitude versus the bounded command those checkpoints
        actually trained on. Called from the MAIN thread (before a rollout worker thread is
        spawned), same discipline as the survival-scan's own per-motion tensor reads -- a
        background thread must never touch live env/torch state directly."""
        motion_command = self.unwrapped_env.command_manager.get_state("motion_command")
        if motion_command is None or not getattr(motion_command, "has_ball", False):
            return None
        kick_aim_enabled_per_motion = motion_command.kick_aim_enabled_per_motion.detach().cpu().tolist()
        kick_aim_theta_ref_deg = float(motion_command.kick_aim_theta_ref_deg)
        return kick_aim_enabled_per_motion, kick_aim_theta_ref_deg

    def _maybe_start_mujoco_kick_rollout(self, onnx_path: str) -> None:
        """Never raises; only ever logs and returns. See `_mujoco_rollout_gate_open` for the
        cadence/eligibility gate; also skips if a previous rollout of this type is still running
        (pileup guard)."""
        args = self.config
        n = args.mujoco_kick_rollout_every_n_saves
        if not self._mujoco_rollout_gate_open(n):
            return

        if self._kick_rollout_thread is not None and self._kick_rollout_thread.is_alive():
            logger.warning(
                f"[sim2sim] Previous MuJoCo kick rollout still running at global_step="
                f"{self.global_step} -- skipping this checkpoint, will retry at the next "
                f"scheduled interval (every {n} saves)."
            )
            return

        kick_aim_info = self._kick_aim_info_per_motion()

        self._kick_rollout_thread = threading.Thread(
            target=self._mujoco_kick_rollout_worker,
            args=(onnx_path, self.global_step, args.mujoco_kick_rollout_with_ball, kick_aim_info),
            daemon=True,
            name="MuJoCoKickRollout",
        )
        self._kick_rollout_thread.start()

    def _mujoco_kick_rollout_worker(
        self,
        onnx_path: str,
        global_step: int,
        with_ball: bool,
        kick_aim_info: "tuple[list[bool], float] | None" = None,
    ) -> None:
        """Thread target. Loops SEQUENTIALLY over every configured motion skill (legacy/no N-skill
        config: exactly 1 "skill", producing the original single video with the original wandb
        key/filename, bit-identical to before this method looped at all). Deliberately sequential,
        not N parallel threads/subprocesses: record_kick_rollout()'s own cross-process lock is a
        single global file (see its docstring) shared cluster-wide across every training process on
        this machine -- firing N concurrent calls from ONE process would just make N-1 of them lose
        the lock race and silently skip that checkpoint's rollout, not actually parallelize
        anything. Only spawns + waits on OS subprocesses -- never touches torch/CUDA, so this
        thread never contends with the GPU training loop regardless of how many skills it loops
        over. Never raises."""
        from holosoma.record_mujoco_kick_rollout import MUJOCO_KICK_WANDB_KEY, record_kick_rollout  # noqa: PLC0415

        num_skills = max(len(getattr(self.unwrapped_env, "_skill_motion_training_ratios", [])), 1)
        for skill_idx in range(num_skills):
            if num_skills == 1:
                # legacy naming, untouched -- no "skill" concept exists for this run
                video_path = os.path.join(self.log_dir, "sim2sim_rollouts", f"model_{global_step:07d}_mujoco_kick.mp4")
                wandb_key = MUJOCO_KICK_WANDB_KEY
            else:
                # 1-indexed to match the yaml's own motion_skill_1/motion_skill_2/... naming and
                # the IsaacSim "Training rollout - Kick - Skill N" recorders' own convention.
                video_path = os.path.join(
                    self.log_dir, "sim2sim_rollouts", f"model_{global_step:07d}_mujoco_kick_skill{skill_idx + 1}.mp4"
                )
                wandb_key = f"{MUJOCO_KICK_WANDB_KEY} - Skill {skill_idx + 1}"

            kick_aim_enabled = False
            kick_aim_theta_ref_deg = 45.0
            if kick_aim_info is not None:
                kick_aim_enabled_per_motion, kick_aim_theta_ref_deg = kick_aim_info
                row = skill_idx if skill_idx < len(kick_aim_enabled_per_motion) else 0
                kick_aim_enabled = bool(kick_aim_enabled_per_motion[row])

            try:
                ok = record_kick_rollout(
                    onnx_path=onnx_path,
                    output_video_path=video_path,
                    with_ball=with_ball,
                    skill_id=skill_idx,
                    kick_aim_enabled=kick_aim_enabled,
                    kick_aim_theta_ref_deg=kick_aim_theta_ref_deg,
                )
            except Exception:
                logger.exception(
                    f"[sim2sim] MuJoCo kick rollout crashed for skill_id={skill_idx} at global_step={global_step}"
                )
                continue

            if not ok:
                logger.warning(
                    f"[sim2sim] MuJoCo kick rollout produced no video for skill_id={skill_idx} "
                    f"at global_step={global_step}"
                )
                continue

            self._kick_rollout_video_queue.put((global_step, video_path, wandb_key))

    def _drain_mujoco_kick_rollout_queue(self) -> None:
        """Logs any finished rollout video(s) to wandb. Runs on the MAIN thread only (called from
        learn()'s loop, never from the worker thread) so every wandb.log call in this class stays
        on one thread, and `step=` is always the CURRENT self.global_step at call time rather than
        a value captured back when the rollout was launched (which could be stale/behind by the
        time a several-second rollout finishes, risking wandb's step-must-not-decrease behavior).
        Cheap no-op when nothing is pending -- safe to call every iteration."""
        while True:
            try:
                triggered_at_step, video_path, wandb_key = self._kick_rollout_video_queue.get_nowait()
            except queue.Empty:
                return
            try:
                wandb.log({wandb_key: wandb.Video(video_path, format="mp4")}, step=self.global_step)
                logger.info(
                    f"[sim2sim] Logged MuJoCo kick rollout ({wandb_key}, triggered at step {triggered_at_step}) "
                    f"to wandb at step {self.global_step}: {video_path}"
                )
            except Exception:
                logger.exception(f"[sim2sim] Failed to wandb.log MuJoCo kick rollout video at {video_path}")

    # ------------------------------------------------------------------
    # MuJoCo sim2sim N-trial FALL RATE scan (holosoma.record_mujoco_survival_scan) -- see
    # FastSACConfig.mujoco_survival_scan_every_n_saves's own docstring for the full measured
    # rationale (a deployed-action, termination-censoring-immune, non-EMA complement to
    # Env/kick_topple_frac). Same thread/queue/subprocess/lock architecture as the kick rollout
    # above; own independent cadence knob and lock file, not folded into
    # mujoco_kick_rollout_every_n_saves.
    # ------------------------------------------------------------------

    def _maybe_start_mujoco_survival_scan(self, onnx_path: str) -> None:
        """Never raises; only ever logs and returns. See `_mujoco_rollout_gate_open` for the
        cadence/eligibility gate (reused as-is, keyed on this field's own
        mujoco_survival_scan_every_n_saves value); also skips if a previous scan is still running
        (pileup guard), and if the live env has no ball configured at all (nothing to jitter).

        Randomization ranges are read from `motion_command.ball_position_randomization_per_motion`
        (managers/command/terms/wbt.py) -- the SAME already-resolved, per-motion tensor training
        itself draws from at every reset -- rather than reconstructed from a config path. This is
        deliberate: that tensor is correct regardless
        of whether this run is in "legacy" mode (one shared BallConfig broadcast to every motion,
        `self._env.simulator.simulator_config.scene.ball`) or N-skill mode (`skill_ball_configs`,
        one entry per motion/skill) -- wbt.py's own setup() already resolved that branch once, so
        reading its output sidesteps replicating that branch logic (and getting it wrong) here.
        `self._experiment_config.simulator.ball` was tried first and found to always be None for
        an N-skill run (confirmed live, 2026-08-19) -- that field is the legacy-mode-only path."""
        args = self.config
        n = args.mujoco_survival_scan_every_n_saves
        if not self._mujoco_rollout_gate_open(n):
            return

        motion_command = self.unwrapped_env.command_manager.get_state("motion_command")
        if motion_command is None or not getattr(motion_command, "has_ball", False):
            logger.warning(
                f"[sim2sim] mujoco_survival_scan_every_n_saves={n} > 0 but this run has no ball "
                "(motion_command.has_ball is False) -- skipping, nothing to jitter or trigger a "
                "kick against."
            )
            return

        pos_rand_per_motion = motion_command.ball_position_randomization_per_motion.detach().cpu().tolist()
        # 2026-08-22, azimuth-aim refactor: same "read the already-resolved per-motion tensor,
        # don't reconstruct from a config path" rationale as the line above -- see this
        # method's own docstring. kick_aim_theta_ref_deg has no per-motion table (it's a single
        # global scalar, MotionCommand.kick_aim_theta_ref_deg -- see MotionConfig.
        # kick_aim_theta_ref_deg's own docstring for why).
        kick_aim_enabled_per_motion = motion_command.kick_aim_enabled_per_motion.detach().cpu().tolist()
        kick_aim_theta_max_deg_per_motion = motion_command.kick_aim_theta_max_deg_per_motion.detach().cpu().tolist()
        kick_aim_theta_ref_deg = float(motion_command.kick_aim_theta_ref_deg)
        # 2026-08-23, direction-success-rate metric: same "single global scalar, no per-motion
        # table" contract as kick_aim_theta_ref_deg above -- see MotionConfig.
        # kick_aim_nominal_distance_m's own docstring for why.
        kick_aim_nominal_distance_m = float(motion_command.kick_aim_nominal_distance_m)

        if self._survival_scan_thread is not None and self._survival_scan_thread.is_alive():
            logger.warning(
                f"[sim2sim] Previous MuJoCo survival scan still running at global_step="
                f"{self.global_step} -- skipping this checkpoint, will retry at the next "
                f"scheduled interval (every {n} saves)."
            )
            return

        self._survival_scan_thread = threading.Thread(
            target=self._mujoco_survival_scan_worker,
            args=(
                onnx_path,
                self.global_step,
                args.mujoco_survival_scan_num_trials,
                pos_rand_per_motion,
                kick_aim_enabled_per_motion,
                kick_aim_theta_max_deg_per_motion,
                kick_aim_theta_ref_deg,
                kick_aim_nominal_distance_m,
            ),
            daemon=True,
            name="MuJoCoSurvivalScan",
        )
        self._survival_scan_thread.start()

    def _mujoco_survival_scan_worker(
        self,
        onnx_path: str,
        global_step: int,
        num_trials: int,
        pos_rand_per_motion: list[list[float]],
        kick_aim_enabled_per_motion: list[bool],
        kick_aim_theta_max_deg_per_motion: list[float],
        kick_aim_theta_ref_deg: float,
        kick_aim_nominal_distance_m: float,
    ) -> None:
        """Thread target. Loops SEQUENTIALLY over every configured motion skill, same rationale as
        `_mujoco_kick_rollout_worker` (one cross-process lock, so N concurrent subprocess launches
        from this one process would just lose the lock race against each other, not parallelize
        anything). Only spawns + waits on an OS subprocess -- never touches torch/CUDA, so this
        thread never contends with the GPU training loop. Never raises.

        `pos_rand_per_motion`: one [x, y] half-range pair per motion, same row indexing
        (motion_id == skill_idx, 0-based) as the ONNX's own skill_ball_xy/skill_target_xy metadata
        (get_skill_ball_target_metadata reads the SAME per-motion tensor family) -- so
        `[skill_idx]` here always lines up with the checkpoint's own skill_id selection on the
        RoboJuDo/scan side.

        2026-08-21, user-requested: `record_survival_scan` now returns `(fall_rate, hit_rate,
        direction_success_rate)` from the SAME N trials (mujoco_kick_survival_scan.py checks
        MuJoCo's own real ball<->foot geom contact per trial alongside the existing fall check --
        no second scan, no extra subprocess; 2026-08-23 added the direction-success read the same
        way). All three are queued independently below under their own wandb keys, so any one can
        be missing (e.g. an unparseable line, or -- for direction_success_rate specifically -- a
        non-kick_aim-enabled skill or zero trials that hit the ball) without losing the others."""
        from holosoma.record_mujoco_survival_scan import (
            MUJOCO_SURVIVAL_SCAN_DIRECTION_WANDB_KEY,
            MUJOCO_SURVIVAL_SCAN_HIT_WANDB_KEY,
            MUJOCO_SURVIVAL_SCAN_WANDB_KEY,
            record_survival_scan,
        )

        num_skills = max(len(getattr(self.unwrapped_env, "_skill_motion_training_ratios", [])), 1)
        for skill_idx in range(num_skills):
            # ALWAYS "Kick_skills_{i}/..." -- even in single-skill mode, matching UnifiedManager's
            # own established convention ("Kick_skills_0" is populated even then, 2026-08-06) so
            # this lands in the SAME per-skill wandb section as its training-time siblings
            # (Kick_skills_0/kick_alive_frac etc.), not a separate top-level "sim2sim/" section.
            fall_wandb_key = f"Kick_skills_{skill_idx}/{MUJOCO_SURVIVAL_SCAN_WANDB_KEY}"
            hit_wandb_key = f"Kick_skills_{skill_idx}/{MUJOCO_SURVIVAL_SCAN_HIT_WANDB_KEY}"
            direction_wandb_key = f"Kick_skills_{skill_idx}/{MUJOCO_SURVIVAL_SCAN_DIRECTION_WANDB_KEY}"
            row = skill_idx if skill_idx < len(pos_rand_per_motion) else 0
            try:
                fall_rate, hit_rate, direction_success_rate = record_survival_scan(
                    onnx_path=onnx_path,
                    step_label=str(global_step),
                    num_trials=num_trials,
                    skill_id=skill_idx,
                    seed=global_step,  # varies per checkpoint; identical trials would be a weaker read
                    ball_pos_randomization=tuple(pos_rand_per_motion[row]),
                    kick_aim_enabled=bool(kick_aim_enabled_per_motion[row]),
                    kick_aim_theta_max_deg=float(kick_aim_theta_max_deg_per_motion[row]),
                    kick_aim_theta_ref_deg=kick_aim_theta_ref_deg,
                    kick_aim_nominal_distance_m=kick_aim_nominal_distance_m,
                )
            except Exception:
                logger.exception(
                    f"[sim2sim] MuJoCo survival scan crashed for skill_id={skill_idx} at global_step={global_step}"
                )
                continue

            if fall_rate is None and hit_rate is None:
                logger.warning(
                    f"[sim2sim] MuJoCo survival scan produced no result for skill_id={skill_idx} "
                    f"at global_step={global_step}"
                )
                continue

            if fall_rate is not None:
                self._survival_scan_result_queue.put((global_step, fall_wandb_key, fall_rate))
            if hit_rate is not None:
                self._survival_scan_result_queue.put((global_step, hit_wandb_key, hit_rate))
            if direction_success_rate is not None:
                self._survival_scan_result_queue.put((global_step, direction_wandb_key, direction_success_rate))

    def _drain_mujoco_survival_scan_queue(self) -> None:
        """Logs any finished scan's rate(s) to wandb as scalars. Runs on the MAIN thread only --
        see `_drain_mujoco_kick_rollout_queue` for why. Cheap no-op when nothing is pending -- safe
        to call every iteration. Generic over WHICH rate this is (fall or, 2026-08-21, ball-hit) --
        `_mujoco_survival_scan_worker` queues one entry per metric, each already carrying its own
        fully-formed wandb_key, so this drain loop never needs to know how many metrics a scan
        produces."""
        while True:
            try:
                triggered_at_step, wandb_key, rate = self._survival_scan_result_queue.get_nowait()
            except queue.Empty:
                return
            try:
                wandb.log({wandb_key: rate}, step=self.global_step)
                logger.info(
                    f"[sim2sim] Logged MuJoCo survival scan ({wandb_key}={rate:.4f}, triggered at "
                    f"step {triggered_at_step}) to wandb at step {self.global_step}"
                )
            except Exception:
                logger.exception(f"[sim2sim] Failed to wandb.log MuJoCo survival scan result ({wandb_key})")

    # ------------------------------------------------------------------
    # MuJoCo sim2sim FORCED KICK->LOCOMOTION FLIP alive-rate scan (holosoma.
    # record_mujoco_kick_to_loco_flip_scan) -- the sim2sim deployment-side counterpart of
    # training's kick_abort_prob. See FastSACConfig.mujoco_kick_to_loco_random_flip_every_n_saves's
    # own docstring for the full motivation and how this complements the survival scan above.
    # Own thread + own result queue + own cross-process lock, same "run concurrently, not
    # sequentially" rationale as every other sim2sim mechanism in this class.
    # ------------------------------------------------------------------

    def _maybe_start_mujoco_kick_to_loco_flip_scan(self, onnx_path: str) -> None:
        """Never raises; only ever logs and returns. See `_mujoco_rollout_gate_open` for the
        cadence/eligibility gate (reused as-is, keyed on this field's own
        mujoco_kick_to_loco_random_flip_every_n_saves value); also skips if a previous scan is
        still running (pileup guard), and if the live env has no ball configured at all (nothing
        to kick before the forced flip)."""
        args = self.config
        n = args.mujoco_kick_to_loco_random_flip_every_n_saves
        if not self._mujoco_rollout_gate_open(n):
            return

        motion_command = self.unwrapped_env.command_manager.get_state("motion_command")
        if motion_command is None or not getattr(motion_command, "has_ball", False):
            logger.warning(
                f"[sim2sim] mujoco_kick_to_loco_random_flip_every_n_saves={n} > 0 but this run has "
                "no ball (motion_command.has_ball is False) -- skipping, nothing to kick before "
                "the forced flip."
            )
            return

        if self._kick_to_loco_flip_scan_thread is not None and self._kick_to_loco_flip_scan_thread.is_alive():
            logger.warning(
                f"[sim2sim] Previous MuJoCo kick-to-loco-flip scan still running at global_step="
                f"{self.global_step} -- skipping this checkpoint, will retry at the next "
                f"scheduled interval (every {n} saves)."
            )
            return

        kick_aim_info = self._kick_aim_info_per_motion()

        self._kick_to_loco_flip_scan_thread = threading.Thread(
            target=self._kick_to_loco_flip_scan_worker,
            args=(onnx_path, self.global_step, args.mujoco_survival_scan_num_trials, kick_aim_info),
            daemon=True,
            name="MuJoCoKickToLocoFlipScan",
        )
        self._kick_to_loco_flip_scan_thread.start()

    def _kick_to_loco_flip_scan_worker(
        self,
        onnx_path: str,
        global_step: int,
        num_trials: int,
        kick_aim_info: "tuple[list[bool], float] | None",
    ) -> None:
        """Thread target. Loops SEQUENTIALLY over every configured motion skill, same rationale as
        `_mujoco_survival_scan_worker` (one cross-process lock, so N concurrent subprocess launches
        from this one process would just lose the lock race against each other). Only spawns +
        waits on an OS subprocess -- never touches torch/CUDA, so this thread never contends with
        the GPU training loop. Never raises."""
        from holosoma.record_mujoco_kick_to_loco_flip_scan import (
            MUJOCO_KICK_TO_LOCO_FLIP_ALIVE_WANDB_KEY,
            MUJOCO_KICK_TO_LOCO_FLIP_PRE_FLIP_FAIL_WANDB_KEY,
            record_kick_to_loco_flip_scan,
        )

        num_skills = max(len(getattr(self.unwrapped_env, "_skill_motion_training_ratios", [])), 1)
        for skill_idx in range(num_skills):
            # Same "always Kick_skills_{i}/..." convention as the survival scan -- lands in the
            # SAME per-skill wandb section as its training-time and survival-scan siblings.
            alive_wandb_key = f"Kick_skills_{skill_idx}/{MUJOCO_KICK_TO_LOCO_FLIP_ALIVE_WANDB_KEY}"
            pre_flip_fail_wandb_key = f"Kick_skills_{skill_idx}/{MUJOCO_KICK_TO_LOCO_FLIP_PRE_FLIP_FAIL_WANDB_KEY}"

            kick_aim_enabled = False
            if kick_aim_info is not None:
                kick_aim_enabled_per_motion, _kick_aim_theta_ref_deg = kick_aim_info
                row = skill_idx if skill_idx < len(kick_aim_enabled_per_motion) else 0
                kick_aim_enabled = bool(kick_aim_enabled_per_motion[row])

            try:
                alive_rate, pre_flip_fail_rate = record_kick_to_loco_flip_scan(
                    onnx_path=onnx_path,
                    step_label=str(global_step),
                    num_trials=num_trials,
                    skill_id=skill_idx,
                    seed=global_step,  # varies per checkpoint; identical trials would be a weaker read
                    kick_aim_enabled=kick_aim_enabled,
                )
            except Exception:
                logger.exception(
                    f"[sim2sim] MuJoCo kick-to-loco-flip scan crashed for skill_id={skill_idx} at "
                    f"global_step={global_step}"
                )
                continue

            if alive_rate is None and pre_flip_fail_rate is None:
                logger.warning(
                    f"[sim2sim] MuJoCo kick-to-loco-flip scan produced no result for skill_id={skill_idx} "
                    f"at global_step={global_step}"
                )
                continue

            if alive_rate is not None:
                self._kick_to_loco_flip_scan_result_queue.put((global_step, alive_wandb_key, alive_rate))
            if pre_flip_fail_rate is not None:
                self._kick_to_loco_flip_scan_result_queue.put(
                    (global_step, pre_flip_fail_wandb_key, pre_flip_fail_rate)
                )

    def _drain_mujoco_kick_to_loco_flip_scan_queue(self) -> None:
        """Logs any finished scan's rate(s) to wandb as scalars. Runs on the MAIN thread only --
        see `_drain_mujoco_kick_rollout_queue` for why. Cheap no-op when nothing is pending -- safe
        to call every iteration."""
        while True:
            try:
                triggered_at_step, wandb_key, rate = self._kick_to_loco_flip_scan_result_queue.get_nowait()
            except queue.Empty:
                return
            try:
                wandb.log({wandb_key: rate}, step=self.global_step)
                logger.info(
                    f"[sim2sim] Logged MuJoCo kick-to-loco-flip scan ({wandb_key}={rate:.4f}, "
                    f"triggered at step {triggered_at_step}) to wandb at step {self.global_step}"
                )
            except Exception:
                logger.exception(f"[sim2sim] Failed to wandb.log MuJoCo kick-to-loco-flip scan result ({wandb_key})")

    # ------------------------------------------------------------------
    # MuJoCo sim2sim LOCOMOTION->KICK HANDOFF fall-rate scan (holosoma.
    # record_mujoco_loco_to_kick_handoff_scan) -- the reverse direction of the kick->loco-flip scan
    # above: random locomotion for a randomized 2-3s window, then a forced flip into kick mode. See
    # FastSACConfig.mujoco_loco_to_kick_handoff_every_n_saves's own docstring for the full
    # motivation. Own thread + own result queue + own cross-process lock, same "run concurrently,
    # not sequentially" rationale as every other sim2sim mechanism in this class.
    # ------------------------------------------------------------------

    def _maybe_start_mujoco_loco_to_kick_handoff_scan(self, onnx_path: str) -> None:
        """Never raises; only ever logs and returns. See `_mujoco_rollout_gate_open` for the
        cadence/eligibility gate (reused as-is, keyed on this field's own
        mujoco_loco_to_kick_handoff_every_n_saves value); also skips if a previous scan is still
        running (pileup guard), and if the live env has no ball configured at all (nothing to
        place at the handoff)."""
        args = self.config
        n = args.mujoco_loco_to_kick_handoff_every_n_saves
        if not self._mujoco_rollout_gate_open(n):
            return

        motion_command = self.unwrapped_env.command_manager.get_state("motion_command")
        if motion_command is None or not getattr(motion_command, "has_ball", False):
            logger.warning(
                f"[sim2sim] mujoco_loco_to_kick_handoff_every_n_saves={n} > 0 but this run has no "
                "ball (motion_command.has_ball is False) -- skipping, nothing to place at the "
                "handoff."
            )
            return

        if (
            self._loco_to_kick_handoff_scan_thread is not None
            and self._loco_to_kick_handoff_scan_thread.is_alive()
        ):
            logger.warning(
                f"[sim2sim] Previous MuJoCo loco-to-kick-handoff scan still running at "
                f"global_step={self.global_step} -- skipping this checkpoint, will retry at the "
                f"next scheduled interval (every {n} saves)."
            )
            return

        self._loco_to_kick_handoff_scan_thread = threading.Thread(
            target=self._loco_to_kick_handoff_scan_worker,
            args=(onnx_path, self.global_step, args.mujoco_survival_scan_num_trials),
            daemon=True,
            name="MuJoCoLocoToKickHandoffScan",
        )
        self._loco_to_kick_handoff_scan_thread.start()

    def _loco_to_kick_handoff_scan_worker(self, onnx_path: str, global_step: int, num_trials: int) -> None:
        """Thread target. Loops SEQUENTIALLY over every configured motion skill, same rationale as
        every sibling scan worker (one cross-process lock, so N concurrent subprocess launches
        from this one process would just lose the lock race against each other). Only spawns +
        waits on an OS subprocess -- never touches torch/CUDA, so this thread never contends with
        the GPU training loop. Never raises.

        Unlike the sibling workers, no per-skill kick_aim_enabled gather is needed here --
        record_loco_to_kick_handoff_scan always passes --kick-aim-enabled (this scan REQUIRES it;
        see mujoco_loco_to_kick_handoff_scan.py's own module docstring for why). A skill trained
        WITHOUT kick_aim_enabled is simply not a valid target for this particular scan -- same
        "only ever fires for runs where the mechanism actually applies" contract every sibling
        gate already enforces at a coarser (whole-run) grain."""
        from holosoma.record_mujoco_loco_to_kick_handoff_scan import (
            MUJOCO_LOCO_TO_KICK_HANDOFF_FALL_WANDB_KEY,
            MUJOCO_LOCO_TO_KICK_HANDOFF_HIT_WANDB_KEY,
            MUJOCO_LOCO_TO_KICK_HANDOFF_PRE_HANDOFF_FAIL_WANDB_KEY,
            record_loco_to_kick_handoff_scan,
        )

        num_skills = max(len(getattr(self.unwrapped_env, "_skill_motion_training_ratios", [])), 1)
        for skill_idx in range(num_skills):
            fall_wandb_key = f"Kick_skills_{skill_idx}/{MUJOCO_LOCO_TO_KICK_HANDOFF_FALL_WANDB_KEY}"
            hit_wandb_key = f"Kick_skills_{skill_idx}/{MUJOCO_LOCO_TO_KICK_HANDOFF_HIT_WANDB_KEY}"
            pre_handoff_fail_wandb_key = (
                f"Kick_skills_{skill_idx}/{MUJOCO_LOCO_TO_KICK_HANDOFF_PRE_HANDOFF_FAIL_WANDB_KEY}"
            )

            try:
                fall_rate, hit_rate, pre_handoff_fail_rate = record_loco_to_kick_handoff_scan(
                    onnx_path=onnx_path,
                    step_label=str(global_step),
                    num_trials=num_trials,
                    skill_id=skill_idx,
                    seed=global_step,  # varies per checkpoint; identical trials would be a weaker read
                )
            except Exception:
                logger.exception(
                    f"[sim2sim] MuJoCo loco-to-kick-handoff scan crashed for skill_id={skill_idx} "
                    f"at global_step={global_step}"
                )
                continue

            if fall_rate is None and hit_rate is None and pre_handoff_fail_rate is None:
                logger.warning(
                    f"[sim2sim] MuJoCo loco-to-kick-handoff scan produced no result for "
                    f"skill_id={skill_idx} at global_step={global_step}"
                )
                continue

            if fall_rate is not None:
                self._loco_to_kick_handoff_scan_result_queue.put((global_step, fall_wandb_key, fall_rate))
            if hit_rate is not None:
                self._loco_to_kick_handoff_scan_result_queue.put((global_step, hit_wandb_key, hit_rate))
            if pre_handoff_fail_rate is not None:
                self._loco_to_kick_handoff_scan_result_queue.put(
                    (global_step, pre_handoff_fail_wandb_key, pre_handoff_fail_rate)
                )

    def _drain_mujoco_loco_to_kick_handoff_scan_queue(self) -> None:
        """Logs any finished scan's rate(s) to wandb as scalars. Runs on the MAIN thread only --
        see `_drain_mujoco_kick_rollout_queue` for why. Cheap no-op when nothing is pending -- safe
        to call every iteration."""
        while True:
            try:
                triggered_at_step, wandb_key, rate = self._loco_to_kick_handoff_scan_result_queue.get_nowait()
            except queue.Empty:
                return
            try:
                wandb.log({wandb_key: rate}, step=self.global_step)
                logger.info(
                    f"[sim2sim] Logged MuJoCo loco-to-kick-handoff scan ({wandb_key}={rate:.4f}, "
                    f"triggered at step {triggered_at_step}) to wandb at step {self.global_step}"
                )
            except Exception:
                logger.exception(
                    f"[sim2sim] Failed to wandb.log MuJoCo loco-to-kick-handoff scan result ({wandb_key})"
                )

    # ------------------------------------------------------------------
    # MuJoCo sim2sim WALK rollout (holosoma.record_mujoco_locomotion_rollout) -- forward-walk ->
    # stop-and-stand with a physical ball present, sim2sim signal on the locomotion side that the
    # kick rollout above has no coverage for (2026-07-21, user directive). Same
    # thread/queue/subprocess architecture as the kick rollout, own thread + own video queue + own
    # cross-process lock so the two rollout types run concurrently instead of contending.
    # ------------------------------------------------------------------

    def _maybe_start_mujoco_walk_rollout(self, onnx_path: str) -> None:
        """Never raises; only ever logs and returns. See `_mujoco_rollout_gate_open` for the
        cadence/eligibility gate (shared with the kick rollout); also skips if a previous rollout
        of this type is still running (pileup guard)."""
        args = self.config
        n = args.mujoco_kick_rollout_every_n_saves
        if not self._mujoco_rollout_gate_open(n):
            return

        if self._walk_rollout_thread is not None and self._walk_rollout_thread.is_alive():
            logger.warning(
                f"[sim2sim] Previous MuJoCo walk rollout still running at global_step="
                f"{self.global_step} -- skipping this checkpoint, will retry at the next "
                f"scheduled interval (every {n} saves)."
            )
            return

        self._walk_rollout_thread = threading.Thread(
            target=self._mujoco_walk_rollout_worker,
            args=(onnx_path, self.global_step),
            daemon=True,
            name="MuJoCoWalkRollout",
        )
        self._walk_rollout_thread.start()

    def _mujoco_walk_rollout_worker(self, onnx_path: str, global_step: int) -> None:
        """Thread target. Only spawns + waits on an OS subprocess -- never touches torch/CUDA, so
        it never contends with the GPU training loop. Never raises."""
        from holosoma.record_mujoco_locomotion_rollout import record_locomotion_rollout  # noqa: PLC0415

        video_path = os.path.join(self.log_dir, "sim2sim_rollouts", f"model_{global_step:07d}_mujoco_walk.mp4")
        try:
            ok = record_locomotion_rollout(onnx_path=onnx_path, output_video_path=video_path)
        except Exception:
            logger.exception(f"[sim2sim] MuJoCo walk rollout crashed for checkpoint at global_step={global_step}")
            return

        if not ok:
            logger.warning(f"[sim2sim] MuJoCo walk rollout produced no video for global_step={global_step}")
            return

        self._walk_rollout_video_queue.put((global_step, video_path))

    def _drain_mujoco_walk_rollout_queue(self) -> None:
        """Logs any finished rollout video(s) to wandb. Same main-thread-only contract as
        `_drain_mujoco_kick_rollout_queue` -- see there for why."""
        from holosoma.record_mujoco_locomotion_rollout import MUJOCO_WALK_WANDB_KEY  # noqa: PLC0415

        while True:
            try:
                triggered_at_step, video_path = self._walk_rollout_video_queue.get_nowait()
            except queue.Empty:
                return
            try:
                wandb.log({MUJOCO_WALK_WANDB_KEY: wandb.Video(video_path, format="mp4")}, step=self.global_step)
                logger.info(
                    f"[sim2sim] Logged MuJoCo walk rollout (triggered at step {triggered_at_step}) "
                    f"to wandb at step {self.global_step}: {video_path}"
                )
            except Exception:
                logger.exception(f"[sim2sim] Failed to wandb.log MuJoCo walk rollout video at {video_path}")

    # ------------------------------------------------------------------
    # MuJoCo sim2sim KICK-HANDOFF rollout (holosoma.record_mujoco_kick_rollout, walk_s>0 variant)
    # -- walk-then-trigger, a simple actor-robustness check on top of the settle-then-trigger kick
    # rollout above (2026-08-13, user directive). NOT a port of UnifiedManager._maybe_enter_kick_
    # from_locomotion's entry-point search -- see mujoco_kick_rollout_worker.py's own docstring.
    # Same thread/queue/subprocess architecture, own thread + own video queue + own cross-process
    # lock (DEFAULT_HANDOFF_LOCK_PATH) so all three rollout types run concurrently.
    # ------------------------------------------------------------------

    def _maybe_start_mujoco_kick_handoff_rollout(self, onnx_path: str) -> None:
        """Never raises; only ever logs and returns. Same cadence/eligibility gate as the other two
        rollouts (_mujoco_rollout_gate_open); also skips if a previous rollout of this type is
        still running (pileup guard)."""
        args = self.config
        n = args.mujoco_kick_rollout_every_n_saves
        if not self._mujoco_rollout_gate_open(n):
            return

        if self._kick_handoff_rollout_thread is not None and self._kick_handoff_rollout_thread.is_alive():
            logger.warning(
                f"[sim2sim] Previous MuJoCo kick-handoff rollout still running at global_step="
                f"{self.global_step} -- skipping this checkpoint, will retry at the next "
                f"scheduled interval (every {n} saves)."
            )
            return

        kick_aim_info = self._kick_aim_info_per_motion()

        self._kick_handoff_rollout_thread = threading.Thread(
            target=self._mujoco_kick_handoff_rollout_worker,
            args=(onnx_path, self.global_step, args.mujoco_kick_rollout_with_ball, kick_aim_info),
            daemon=True,
            name="MuJoCoKickHandoffRollout",
        )
        self._kick_handoff_rollout_thread.start()

    def _mujoco_kick_handoff_rollout_worker(
        self,
        onnx_path: str,
        global_step: int,
        with_ball: bool,
        kick_aim_info: "tuple[list[bool], float] | None" = None,
    ) -> None:
        """Thread target. Loops sequentially over every configured motion skill, same rationale as
        `_mujoco_kick_rollout_worker` (record_kick_rollout()'s cross-process lock is per-rollout-
        type, not per-skill, so N parallel calls here would just lose the lock race against each
        other). Never raises."""
        from holosoma.record_mujoco_kick_rollout import (  # noqa: PLC0415
            DEFAULT_HANDOFF_LOCK_PATH,
            MUJOCO_KICK_HANDOFF_WANDB_KEY,
            record_kick_rollout,
        )

        num_skills = max(len(getattr(self.unwrapped_env, "_skill_motion_training_ratios", [])), 1)
        for skill_idx in range(num_skills):
            if num_skills == 1:
                video_path = os.path.join(
                    self.log_dir, "sim2sim_rollouts", f"model_{global_step:07d}_mujoco_kick_handoff.mp4"
                )
                wandb_key = MUJOCO_KICK_HANDOFF_WANDB_KEY
            else:
                video_path = os.path.join(
                    self.log_dir,
                    "sim2sim_rollouts",
                    f"model_{global_step:07d}_mujoco_kick_handoff_skill{skill_idx + 1}.mp4",
                )
                wandb_key = f"{MUJOCO_KICK_HANDOFF_WANDB_KEY} - Skill {skill_idx + 1}"

            kick_aim_enabled = False
            kick_aim_theta_ref_deg = 45.0
            if kick_aim_info is not None:
                kick_aim_enabled_per_motion, kick_aim_theta_ref_deg = kick_aim_info
                row = skill_idx if skill_idx < len(kick_aim_enabled_per_motion) else 0
                kick_aim_enabled = bool(kick_aim_enabled_per_motion[row])

            try:
                ok = record_kick_rollout(
                    onnx_path=onnx_path,
                    output_video_path=video_path,
                    with_ball=with_ball,
                    skill_id=skill_idx,
                    walk_s=3.0,
                    forward_speed=0.5,
                    lock_path=DEFAULT_HANDOFF_LOCK_PATH,
                    kick_aim_enabled=kick_aim_enabled,
                    kick_aim_theta_ref_deg=kick_aim_theta_ref_deg,
                )
            except Exception:
                logger.exception(
                    f"[sim2sim] MuJoCo kick-handoff rollout crashed for skill_id={skill_idx} at "
                    f"global_step={global_step}"
                )
                continue

            if not ok:
                logger.warning(
                    f"[sim2sim] MuJoCo kick-handoff rollout produced no video for skill_id={skill_idx} "
                    f"at global_step={global_step}"
                )
                continue

            self._kick_handoff_rollout_video_queue.put((global_step, video_path, wandb_key))

    def _drain_mujoco_kick_handoff_rollout_queue(self) -> None:
        """Logs any finished rollout video(s) to wandb. Same main-thread-only contract as
        `_drain_mujoco_kick_rollout_queue` -- see there for why."""
        while True:
            try:
                triggered_at_step, video_path, wandb_key = self._kick_handoff_rollout_video_queue.get_nowait()
            except queue.Empty:
                return
            try:
                wandb.log({wandb_key: wandb.Video(video_path, format="mp4")}, step=self.global_step)
                logger.info(
                    f"[sim2sim] Logged MuJoCo kick-handoff rollout ({wandb_key}, triggered at step "
                    f"{triggered_at_step}) to wandb at step {self.global_step}: {video_path}"
                )
            except Exception:
                logger.exception(f"[sim2sim] Failed to wandb.log MuJoCo kick-handoff rollout video at {video_path}")

    def save(self, path: str) -> None:  # type: ignore[override]
        env_state = self._collect_env_state()
        save_params(
            self.global_step,
            self.actor,
            self.qnet,
            self.qnet_target,
            self.log_alpha,
            self.obs_normalizer,
            self.critic_obs_normalizer,
            self.actor_optimizer,
            self.q_optimizer,
            self.alpha_optimizer,
            self.scaler,
            self.config,
            path,
            save_fn=self.logging_helper.save_checkpoint_artifact,
            env_state=env_state or None,
            metadata=self._checkpoint_metadata(iteration=self.global_step),
        )

    @torch.no_grad()
    def get_example_obs(self):
        """Used for exporting policy as onnx."""
        obs_dict = self.unwrapped_env.reset_all()
        for k in obs_dict:
            obs_dict[k] = obs_dict[k].cpu()
        return {
            "actor_obs": torch.cat([obs_dict[k] for k in self.config.actor_obs_keys], dim=1),
            "critic_obs": torch.cat([obs_dict[k] for k in self.config.critic_obs_keys], dim=1),
        }

    def get_inference_policy(self, device: str | None = None) -> Callable[[dict[str, torch.Tensor]], torch.Tensor]:
        device = device or self.device
        # Use the underlying module for inference
        policy = self.actor.to(device)
        obs_normalizer = self.obs_normalizer.to(device)
        policy.eval()
        obs_normalizer.eval()

        def policy_fn(obs: dict[str, torch.Tensor]) -> torch.Tensor:
            if self.obs_normalization:
                normalized_obs = obs_normalizer(obs["actor_obs"], update=False)
            else:
                normalized_obs = obs["actor_obs"]
            # Actions are already scaled by the actor
            return policy(normalized_obs)[0]

        return policy_fn

    @property
    def actor_onnx_wrapper(self):
        # Use the underlying module for ONNX export
        actor = copy.deepcopy(self.actor).to("cpu")
        obs_normalizer = copy.deepcopy(self.obs_normalizer).to("cpu")
        # Expected-action export only applies to tanh-squashed actors -- without the squash,
        # E[mu + sigma*eps] == mu and the correction is an exact no-op, so skip the extra compute.
        use_expected = bool(getattr(self.config, "export_expected_action", True)) and bool(
            getattr(actor, "use_tanh", True)
        )

        class ActorWrapper(nn.Module):
            def __init__(self, actor, obs_normalizer, use_expected: bool):
                super().__init__()
                self.actor = actor
                self.obs_normalizer = obs_normalizer
                self.use_expected = use_expected
                if use_expected:
                    # Fixed 8-node Gauss-Hermite quadrature for E[tanh(mu + sigma*Z)], Z~N(0,1):
                    # E[f(mu + sigma*Z)] = (1/sqrt(pi)) * sum_j w_j * f(mu + sqrt(2)*sigma*t_j).
                    # sqrt(2) is folded into the nodes and 1/sqrt(pi) into the weights, so the
                    # traced graph is just 8 tanh evaluations + a weighted sum -- deterministic,
                    # ONNX-representable, no runtime sampling. See config_types/algo.py's
                    # export_expected_action docstring for why this (and not tanh(mu)) is the
                    # right deterministic deployment action.
                    t_nodes, w_nodes = np.polynomial.hermite.hermgauss(8)
                    self.register_buffer(
                        "gh_nodes", torch.tensor(t_nodes * math.sqrt(2.0), dtype=torch.float32)
                    )
                    self.register_buffer(
                        "gh_weights", torch.tensor(w_nodes / math.sqrt(math.pi), dtype=torch.float32)
                    )

            def forward(self, actor_obs):
                if self.obs_normalizer is not None:
                    normalized_obs = self.obs_normalizer(actor_obs, update=False)
                else:
                    normalized_obs = actor_obs
                if not self.use_expected:
                    # Actions are already scaled by the actor
                    return self.actor(normalized_obs)[0]
                _, mean, log_std = self.actor(normalized_obs)
                std = log_std.exp()
                shifted = mean.unsqueeze(0) + std.unsqueeze(0) * self.gh_nodes.view(-1, 1, 1)
                tanh_avg = (torch.tanh(shifted) * self.gh_weights.view(-1, 1, 1)).sum(dim=0)
                return tanh_avg * self.actor.action_scale + self.actor.action_bias

        return ActorWrapper(actor, obs_normalizer if self.obs_normalization else None, use_expected)

    def extract_actor_obs(self, obs: torch.Tensor, obs_key: str) -> torch.Tensor:
        """
        Extract a specific observation component from the flattened actor observation tensor.

        Args:
            obs: Flattened actor observation tensor of shape [batch_size, actor_obs_dim]
            obs_key: The observation key to extract (e.g., 'perception_obs', 'actor_state_obs')

        Returns:
            Extracted observation tensor of shape [batch_size, obs_size]
        """
        if obs_key not in self.actor_obs_indices:
            raise ValueError(
                f"Observation key '{obs_key}' not found in actor observations. "
                f"Available keys: {list(self.actor_obs_indices.keys())}"
            )

        indices = self.actor_obs_indices[obs_key]
        return obs[..., indices["start"] : indices["end"]]

    def extract_critic_obs(self, obs: torch.Tensor, obs_key: str) -> torch.Tensor:
        """
        Extract a specific observation component from the flattened critic observation tensor.

        Args:
            obs: Flattened critic observation tensor of shape [batch_size, critic_obs_dim]
            obs_key: The observation key to extract (e.g., 'perception_obs', 'critic_state_obs')

        Returns:
            Extracted observation tensor of shape [batch_size, obs_size]
        """
        if obs_key not in self.critic_obs_indices:
            raise ValueError(
                f"Observation key '{obs_key}' not found in critic observations. "
                f"Available keys: {list(self.critic_obs_indices.keys())}"
            )

        indices = self.critic_obs_indices[obs_key]
        return obs[..., indices["start"] : indices["end"]]

    def get_actor_obs_info(self) -> dict[str, dict[str, int]]:
        """
        Get information about actor observation indices.

        Returns:
            Dictionary with obs_key -> {'start': int, 'end': int, 'size': int}
        """
        return self.actor_obs_indices.copy()

    def get_critic_obs_info(self) -> dict[str, dict[str, int]]:
        """
        Get information about critic observation indices.

        Returns:
            Dictionary with obs_key -> {'start': int, 'end': int, 'size': int}
        """
        return self.critic_obs_indices.copy()

    def export(self, onnx_file_path: str) -> None:
        """Export the `.onnx` of the policy to & save it to `path`.

        This is intended to enable deployment, but not resuming training.
        For storing checkpoints to resume training, see `FastSACAgent.save()`
        """
        # Save current training state
        was_training = self.actor.training

        # Set model to evaluation mode for export so we don't affect gradients mid-rollout
        self.actor.eval()
        if self.obs_normalization:
            self.obs_normalizer.eval()

        # Create dummy all-zero input for ONNX tracing.
        example_input_list = torch.zeros(1, self.actor_obs_dim, device="cpu")

        motion_command = self.unwrapped_env.command_manager.get_state("motion_command")
        if motion_command is not None:
            export_motion_and_policy_as_onnx(
                self.actor_onnx_wrapper,
                motion_command,
                onnx_file_path,
                self.device,
            )
        else:
            export_policy_as_onnx(
                wrapper=self.actor_onnx_wrapper,
                onnx_file_path=onnx_file_path,
                example_obs_dict={"actor_obs": example_input_list},
            )

        # Extract control gains and velocity limits & attach to onnx as metadata
        kp_list, kd_list = get_control_gains_from_config(self.env.robot_config)
        cmd_ranges = get_command_ranges_from_env(self.unwrapped_env)
        action_scales = getattr(self.unwrapped_env, "action_scales", None)
        if action_scales is None:
            action_scale_metadata: float | list[float] = float(self.env.robot_config.control.action_scale)
        else:
            action_scale_metadata = action_scales.detach().cpu().tolist()
        # Extract URDF text from the robot config
        urdf_file_path, urdf_str = get_urdf_text_from_robot_config(self.env.robot_config)

        metadata = {
            "dof_names": self.env.robot_config.dof_names,
            "kp": kp_list,
            "kd": kd_list,
            "action_scale": action_scale_metadata,
            "command_ranges": cmd_ranges,
            "robot_urdf": urdf_str,
            "robot_urdf_path": urdf_file_path,
        }
        metadata.update(get_skill_motion_boundaries_metadata(motion_command))
        metadata.update(get_kick_recovery_locomotion_flip_metadata(self.unwrapped_env))
        metadata.update(get_skill_pre_recovery_metadata(motion_command))
        metadata.update(get_skill_ball_target_metadata(motion_command))
        metadata.update(self._checkpoint_metadata(iteration=self.global_step))

        attach_onnx_metadata(
            onnx_path=onnx_file_path,
            metadata=metadata,
        )

        # Restore original training state
        if was_training:
            self.actor.train()
            if self.obs_normalization:
                self.obs_normalizer.train()

    @torch.no_grad()
    def evaluate_policy(self, max_eval_steps: int | None = None):
        self._create_eval_callbacks()
        self._pre_evaluate_policy()

        obs = self.env.reset()

        for step in itertools.islice(itertools.count(), max_eval_steps):
            if self.obs_normalization:
                normalized_obs = self.obs_normalizer(obs, update=False)
            else:
                normalized_obs = obs
            # Actions are already scaled by the actor
            actions = self.actor(normalized_obs)[0]

            actor_state = {"step": step, "actions": actions, "obs": obs}
            actor_state = self._pre_eval_env_step(actor_state)

            obs, _, _, _ = self.env.step(actor_state["actions"])
            actor_state["obs"] = obs
            actor_state = self._post_eval_env_step(actor_state)

        self._post_evaluate_policy()

    def _create_eval_callbacks(self):
        if self.config.eval_callbacks is not None:
            for cb_name in self.config.eval_callbacks:
                self.eval_callbacks.append(instantiate(self.config.eval_callbacks[cb_name], training_loop=self))

    def _pre_evaluate_policy(self):
        self.env.set_is_evaluating()
        for c in self.eval_callbacks:
            c.on_pre_evaluate_policy()

    def _post_evaluate_policy(self):
        for c in self.eval_callbacks:
            c.on_post_evaluate_policy()

    def _pre_eval_env_step(self, actor_state: dict) -> dict:
        for c in self.eval_callbacks:
            actor_state = c.on_pre_eval_env_step(actor_state)
        return actor_state

    def _post_eval_env_step(self, actor_state: dict) -> dict:
        for c in self.eval_callbacks:
            actor_state = c.on_post_eval_env_step(actor_state)
        return actor_state
