from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Tuple

import onnx
import torch

from holosoma.config_types.robot import RobotConfig
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.utils.module_utils import get_holosoma_root


def _find_input_dim_from_module(module: torch.nn.Module) -> int:
    """Finds the input dimension by examining a torch module's structure.

    Tries multiple strategies to find the first Linear layer's input features.
    """
    # Strategy 1: PPO-style - actor_module.module (torch.nn.Sequential)
    if hasattr(module, "actor_module") and hasattr(module.actor_module, "module"):
        core_model = module.actor_module.module
        if hasattr(core_model[0], "in_features"):
            return core_model[0].in_features

    # Strategy 2: FastSAC/FastTD3-style - .net attribute
    if hasattr(module, "net") and len(module.net) > 0:
        if hasattr(module.net[0], "in_features"):
            return module.net[0].in_features

    # Strategy 3: Find first Linear layer in module tree
    for submodule in module.modules():
        if isinstance(submodule, torch.nn.Linear):
            return submodule.in_features

    raise ValueError(f"Cannot determine input dimension from module: {type(module)}")


def _extract_actor_model_and_input_dim(actor_wrapper) -> Tuple[torch.nn.Module, int]:
    """Extracts the underlying actor model and input dimension from various actor wrapper types.

    This function handles the complete actor pipeline including observation normalization.

    Parameters
    ----------
    actor_wrapper : object
        Actor wrapper containing the actor model. Can be various types including:
        - PPO actors: ActorWrapper -> PPOActor -> actor_module.module (torch.nn.Sequential)
        - FastSAC actors: ActorWrapper(with obs_normalizer) -> FastSAC Actor (custom module)
        - FastTD3 actors: ActorWrapper(with obs_normalizer) -> FastTD3 Actor (custom module)

    Returns
    -------
    complete_actor_model : torch.nn.Module
        The complete actor model including observation normalization if present.
    input_dimension : int
        The input dimension of the actor model.

    Notes
    -----
    The returned actor_model includes observation normalization if present,
    so it can be called directly with raw observations.
    """
    # If it's already a complete wrapper (from actor_onnx_wrapper), use it directly
    if hasattr(actor_wrapper, "forward") and hasattr(actor_wrapper, "actor"):
        # Use the complete wrapper that includes obs normalization
        complete_model = actor_wrapper
        # Find input dim from the inner actor
        input_dim = _find_input_dim_from_module(actor_wrapper.actor)
        return complete_model, input_dim

    # Otherwise, extract the inner actor and find its input dimension
    inner_actor = getattr(actor_wrapper, "actor", actor_wrapper)

    if not isinstance(inner_actor, torch.nn.Module):
        raise ValueError(
            f"Unsupported actor type: {type(inner_actor)}. Expected torch.nn.Module or wrapper with .actor attribute"
        )

    input_dim = _find_input_dim_from_module(inner_actor)

    # For unwrapped actors, we might need to return the inner model for certain types
    # PPO: Return the core Sequential model, others: return the actor itself
    if hasattr(inner_actor, "actor_module") and hasattr(inner_actor.actor_module, "module"):
        return inner_actor.actor_module.module, input_dim
    return inner_actor, input_dim


def export_policy_as_onnx(wrapper, onnx_file_path: str, example_obs_dict):
    # Ensure parent directory exists
    os.makedirs(Path(onnx_file_path).parent, exist_ok=True)
    example_input_list = example_obs_dict["actor_obs"]

    # --- SUPPRESS LOGS START ---
    # Silence onnxscript and onnx_ir debug/info noise
    import logging

    for logger_name in ["onnxscript", "onnx_ir", "torch.onnx"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    # --- SUPPRESS LOGS END ---

    torch.onnx.export(
        wrapper,
        example_input_list,  # Pass x1 and x2 as separate inputs
        onnx_file_path,
        verbose=False,
        input_names=["actor_obs"],  # Specify the input names
        output_names=["action"],  # Name the output
        opset_version=13,
        dynamo=False,
    )


def export_multi_agent_decouple_policy_as_onnx(wrapper, path, exported_policy_name, example_obs_dict, config):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, exported_policy_name)
    body_keys = config.robot.get("body_keys", ["lower_body", "upper_body"])
    actor_obs_keys = {}
    for body_key in body_keys:
        actor_obs_keys[body_key] = config.algo.config.module_dict[f"actor_{body_key}"].input_dim

    # Prepare example inputs
    example_input_list = []
    for body_key in body_keys:
        actor_obs = torch.cat([example_obs_dict[value] for value in actor_obs_keys[body_key]], dim=-1)
        example_input_list.append(actor_obs)

    # Export to ONNX
    torch.onnx.export(
        wrapper,
        example_input_list,
        path,
        verbose=False,
        input_names=[f"actor_obs_{body_key}" for body_key in body_keys],
        output_names=["action"],
        opset_version=13,
        dynamo=False,
    )


class _OnnxMotionPolicyExporter(torch.nn.Module):
    def __init__(self, motion_command, actor, device):
        super().__init__()
        self.device = device
        # Extract the underlying actor model and input dimension generically
        actor_model, self.input_dim = _extract_actor_model_and_input_dim(actor)
        # Wrap the actor to handle different return signatures
        self._wrapped_actor = self._create_actor_wrapper(actor_model)

        motion = motion_command.motion

        joint_pos = motion.joint_pos
        joint_vel = motion.joint_vel

        body_pos_w = motion.body_pos_w
        body_quat_w = motion.body_quat_w
        ref_body_index = motion_command.ref_body_index
        ref_body_pos_w = body_pos_w[:, ref_body_index, :]
        ref_body_quat_w = body_quat_w[:, ref_body_index, :]  # in xyzw

        self.joint_pos = joint_pos.to("cpu")
        self.joint_vel = joint_vel.to("cpu")
        self.ref_body_pos_w = ref_body_pos_w.to("cpu")
        self.ref_body_quat_w = ref_body_quat_w.to("cpu")

        self.time_step_total = self.joint_pos.shape[0]

    def _create_actor_wrapper(self, actor_model):
        """Creates a wrapper that normalizes actor output to just return actions."""

        class ActorWrapper(torch.nn.Module):
            def __init__(self, actor):
                super().__init__()
                self.actor = actor

            def forward(self, x):
                output = self.actor(x)
                # Handle different return signatures:
                # - PPO Sequential: returns tensor directly
                # - PPO ActorWrapper: returns tensor directly
                # - FastSAC/FastTD3: returns tuple (action, mean, log_std) or (action, ...)
                # - FastSAC/FastTD3 ActorWrapper: already returns action tensor
                if isinstance(output, tuple):
                    return output[0]  # Return first element (action)
                return output  # Return as-is for tensors

        return ActorWrapper(actor_model)

    def forward(self, x, time_step):
        time_step_clamped = torch.clamp(time_step.long().squeeze(-1), max=self.time_step_total - 1)
        return (
            self._wrapped_actor(x),
            self.joint_pos[time_step_clamped],
            self.joint_vel[time_step_clamped],
            self.ref_body_pos_w[time_step_clamped],
            self.ref_body_quat_w[time_step_clamped],
        )

    def export(self, onnx_file_path: str):
        onnx_file_dir = os.path.dirname(onnx_file_path)
        os.makedirs(onnx_file_dir, exist_ok=True)
        self.to("cpu")
        obs = torch.zeros(1, self.input_dim)
        time_step = torch.zeros(1, 1)
        torch.onnx.export(
            self,
            (obs, time_step),
            onnx_file_path,
            export_params=True,
            opset_version=13,
            verbose=False,
            input_names=["obs", "time_step"],
            output_names=["actions", "joint_pos", "joint_vel", "ref_pos_xyz", "ref_quat_xyzw"],
            dynamo=False,
        )
        self.to(self.device)


def export_motion_and_policy_as_onnx(
    actor: object,
    motion_command: object,
    onnx_file_path: str,
    device: str,
):
    policy_exporter = _OnnxMotionPolicyExporter(motion_command, actor, device)
    policy_exporter.export(onnx_file_path)


def attach_onnx_metadata(onnx_path: str, metadata: dict[str, Any]) -> None:
    """Attach custom metadata to an ONNX model file.

    Loads the ONNX model, appends metadata key-value pairs (values are serialized as JSON),
    and saves the modified model back to the same file.

    Parameters
    ----------
    onnx_path : str
        Path to the ONNX model file to modify.
    metadata : dict[str, Any]
        Dictionary of metadata to attach. Values are JSON-serialized before storage.
    """
    model = onnx.load(onnx_path)

    for k, v in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = k
        entry.value = json.dumps(v)
        model.metadata_props.append(entry)

    onnx.save(model, onnx_path)


def get_skill_motion_boundaries_metadata(motion_command: object | None) -> dict[str, list[int]]:
    """Per-skill [start, end) absolute frame indices into the motion buffer that
    _OnnxMotionPolicyExporter embeds into the exported `.onnx` (its `joint_pos`/`joint_vel`/
    `ref_pos_xyz`/`ref_quat_xyzw` outputs, indexed by the `time_step` input).

    Deployment code (e.g. RoboJuDo's UnifiedLocoKickPolicy) has no other way to know where a given
    skill's segment starts in that embedded buffer -- without this, `time_step` starting at 0
    always implicitly addresses skill 0 only, and nothing on the deployment side can select, or
    even be aware of, any other skill.

    Uniform across legacy single-clip mode and N-skill mode: MotionLoader and MultiMotionLoader
    (managers/command/terms/wbt.py) both expose the same `motion_start_idx`/`motion_end_idx`
    interface -- MotionLoader's degenerate case is exactly `[0]`/`[time_step_total]`, i.e. "one
    skill covering the whole buffer" -- so this needs no legacy/N-skill branching.

    Returns {} if there's no motion_command at all (non-WBT/kick env classes), so callers can
    `metadata.update(...)` unconditionally without an extra guard.
    """
    if motion_command is None:
        return {}
    motion = motion_command.motion
    return {
        "skill_motion_start_idx": motion.motion_start_idx.cpu().tolist(),
        "skill_motion_end_idx": motion.motion_end_idx.cpu().tolist(),
    }


def get_kick_recovery_locomotion_flip_metadata(env: object | None) -> dict[str, bool]:
    """Whether this checkpoint was trained with kick_recovery_locomotion_flip_enabled (Stage D's
    post-swing -> locomotion handoff, see MultiSkillConfig's own docstring).

    Lets deployment code (RoboJuDo's UnifiedLocoKickPolicy) know whether to auto-return to
    locomotion the instant a kick episode's whole authored/raw clip ends (see
    get_skill_pre_recovery_metadata below for the actual frame-index boundary this gates) instead
    of the older "embedded clip has fully plateaued" heuristic. Gating on this flag (rather than
    just "skill_pre_recovery_motion_end_idx metadata is present") matters because that boundary
    metadata is exported for EVERY kick-capable checkpoint, Stage D or not -- auto-returning for a
    checkpoint that was never trained to expect an early flip would be a real behavior regression,
    not a harmless no-op.

    Returns {} if there's no env at all, so callers can `metadata.update(...)` unconditionally
    without an extra guard, same as get_skill_motion_boundaries_metadata. Absent metadata (older
    export) reads as False on the deployment side -- the old plateau heuristic, unchanged.
    """
    if env is None:
        return {}
    return {"kick_recovery_locomotion_flip_enabled": bool(getattr(env, "_kick_recovery_locomotion_flip_enabled", False))}


def get_skill_pre_recovery_metadata(motion_command: object | None) -> dict[str, list[int]]:
    """Per-skill absolute frame index (same buffer-index space as skill_motion_start_idx/
    skill_motion_end_idx above) marking where that skill's WHOLE authored/raw clip ends --
    approach + strike + the actor's own captured post-kick follow-through -- and the synthetic
    recovery-transition + static-hold tail begins (``MotionCommand.pre_recovery_motion_end_idx``,
    captured once per motion in wbt.py's setup() right before that motion's own append call).

    The actual frame-index boundary get_kick_recovery_locomotion_flip_metadata's flag gates:
    together, deployment code (RoboJuDo's UnifiedLocoKickPolicy) reads this to know WHERE to
    auto-return to locomotion, and that flag to know WHETHER to (this checkpoint's own
    kick_recovery_locomotion_flip_enabled setting) -- see that function's docstring for why the
    two are kept separate rather than gating on this metadata's mere presence. Deliberately NOT
    stand_start_idx (narrower -- swing end, excludes real captured follow-through footage) and NOT
    skill_motion_end_idx (wider -- includes the synthetic append+hold tail).

    Returns {} if there's no motion_command at all, so callers can `metadata.update(...)`
    unconditionally without an extra guard, same as get_skill_motion_boundaries_metadata.
    """
    if motion_command is None:
        return {}
    return {"skill_pre_recovery_motion_end_idx": motion_command.pre_recovery_motion_end_idx.cpu().tolist()}


def get_skill_ball_target_metadata(motion_command: object | None) -> dict[str, list[list[float]]]:
    """Per-skill nominal ball spawn (x, y) and shot-target (x, y), both in the env-LOCAL frame --
    i.e. relative to that env's own spawn-tile origin, robot facing +x. For a single fixed-spawn
    deployment rollout (world origin, identity quat, facing +x -- e.g.
    mujoco_kick_rollout_worker.py's MuJoCo scene) that frame IS world coordinates directly, no
    transform needed: these are exactly the `x`/`y`/`target_x`/`target_y` values from the training
    yaml, in the same convention that worker script's own BALL_WORLD_POS/TARGET_WORLD_POS
    constants were manually (and only approximately) trying to match.

    Returns {} if there's no motion_command, or it has no ball (has_ball=False -- a pure
    locomotion/tracking env never populates these tables at all).
    """
    if motion_command is None or not getattr(motion_command, "has_ball", False):
        return {}
    return {
        "skill_ball_xy": motion_command.ball_reset_state_per_motion[:, :2].cpu().tolist(),
        "skill_target_xy": motion_command.target_nominal_local_per_motion.cpu().tolist(),
    }


def get_control_gains_from_config(robot_config: RobotConfig) -> tuple[list[float], list[float]]:
    """Extract Kp & Kd gains from env.

    The order of returned lists is determined by `env.robot_config.dof_names`.
    """

    kp_list = []
    kd_list = []
    stiffness_dict = robot_config.control.stiffness
    damping_dict = robot_config.control.damping

    for dof_name in robot_config.dof_names:
        # Map each DOF to its corresponding kp/kd value using substring matching
        # e.g. `left_hip_pitch_joint` from `dof_names` to  `hip_pitch` in `robot_config.control.stiffness`
        matches = [p for p in stiffness_dict if p in dof_name]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly 1 pattern match for '{dof_name}', got {len(matches)}: {matches}")

        pattern = matches[0]
        kp_list.append(float(stiffness_dict[pattern]))
        kd_list.append(float(damping_dict[pattern]))

    return kp_list, kd_list


def get_command_ranges_from_env(env: BaseTask) -> dict | None:
    """Extract command limits from env command manager."""

    if env.command_manager is not None:
        locomotion_cmd = env.command_manager.get_state("locomotion_command")
        if locomotion_cmd is not None and hasattr(locomotion_cmd, "command_ranges"):
            return locomotion_cmd.command_ranges
    return None


def get_urdf_text_from_robot_config(robot_config: RobotConfig) -> tuple[str, str]:
    """Extract URDF text from the robot config.

    Returns
    -------
    tuple[str, str]
        (urdf_file_path, urdf_str) - Path to URDF file and its contents
    """
    asset_root = robot_config.asset.asset_root
    if asset_root.startswith("@holosoma/"):
        asset_root = asset_root.replace("@holosoma", get_holosoma_root())

    asset_file = robot_config.asset.urdf_file
    urdf_file_path = os.path.join(asset_root, asset_file)
    urdf_str = Path(urdf_file_path).read_text(encoding="utf-8")
    return urdf_file_path, urdf_str
