from __future__ import annotations

import dataclasses
import json

import numpy as np
import onnx
import onnxruntime
from termcolor import colored

from holosoma_inference.ball_pose_source import BallPoseSource, FixedBallPoseSource, UdpBallPoseSource
from holosoma_inference.inputs.api.commands import StateCommand
from holosoma_inference.policies.base import BasePolicy
from holosoma_inference.policies.wbt_utils import PinocchioRobot
from holosoma_inference.utils.math.quat import (
    matrix_from_quat,
    quat_mul,
    quat_to_rpy,
    rpy_to_quat,
    subtract_frame_transforms,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)


class UnifiedPolicy(BasePolicy):
    """Single policy that does BOTH locomotion (continuous velocity-command walking)
    and ball-kicking (one-shot motion-clip tracking), mirroring UnifiedManager's
    training-side task_mode mechanism.

    Modeled as a standalone sibling of LocomotionPolicy/WholeBodyTrackingPolicy (not a
    subclass of either) since it needs pieces of both: continuous velocity commands +
    gait phase from locomotion, and the ONNX-embedded time_step-indexed motion clip
    from WBT. See docs/workflows/ for the training-side observation-masking mechanism
    this class mirrors term-for-term.
    """

    def __init__(self, config):
        # Deploy a pre-v7 (256-dim) checkpoint: swap in the legacy observation config before
        # BasePolicy.__init__ (called at the end of this method) builds obs_dict/obs_terms_sorted
        # from it. Done HERE, in the base class every unified entry point constructs (UnifiedPolicy
        # itself, NetworkControlledUnifiedPolicy, ApproachAndKickPolicy), rather than in any one
        # script's main() -- run_policy.py originally did this swap itself, which silently didn't
        # apply to run_network_controlled_policy.py/run_approach_and_kick_policy.py (each has its
        # own main(), none share run_policy.py's). Centralizing here means any current or future
        # entry point that constructs one of these policies gets the flag's actual behavior for
        # free, instead of needing to remember to re-implement it.
        if config.task.use_previous_version_policy:
            from holosoma_inference.config.config_values.observation import unified_g1_29dof_legacy

            config = dataclasses.replace(config, observation=unified_g1_29dof_legacy)

        # Kick-clip state, same shape as WholeBodyTrackingPolicy.__init__.
        self.task_mode: str = "locomotion"  # "locomotion" | "kick"
        self.motion_clip_progressing = False
        self.curr_motion_timestep = 0
        self.motion_command_t: np.ndarray | None = None
        self.ref_quat_xyzw_t: np.ndarray | None = None
        self.motion_command_0: np.ndarray | None = None
        self.ref_quat_xyzw_0: np.ndarray | None = None
        self.robot_yaw_offset = 0.0
        self.motion_yaw_offset = 0.0
        self.is_standing = False

        # Auto-return-to-locomotion bookkeeping: the ONNX's embedded clip clamps at
        # its final (post-kick hold) frame once time_step runs past the clip length,
        # so consecutive identical motion_command_t outputs are how we detect "clip
        # has finished" without needing to separately know the clip's frame count.
        self._kick_hold_ticks = 0

        # kick_ball_pos_b / kick_target_pos_b support (training v7 -- see
        # locomotion_and_ball_kicking/src/holosoma/README.md's "Shooting rewards &
        # observations" section). Built from config.task so this works through the fully
        # generic run_policy.py dispatch (policy_class(config=config), no extra kwargs) --
        # see TaskConfig's kick_ball_source/kick_ball_fixed_*/kick_ball_udp_port/
        # kick_target_dx/dy field docstrings for what each source means and its caveats.
        self.ball_pose_source: BallPoseSource | None = {
            "none": lambda: None,
            "fixed": lambda: FixedBallPoseSource(config.task.kick_ball_fixed_dx, config.task.kick_ball_fixed_dy),
            "udp": lambda: UdpBallPoseSource(config.task.kick_ball_udp_port),
        }[config.task.kick_ball_source]()
        # 2026-08-22, azimuth-aim refactor: kick_aim_enabled swaps this slot's meaning from an
        # absolute body-frame target OFFSET (kick_target_dx/dy, metres, moves with the robot --
        # see that field's own docstring for why this was flagged smoke-test-only) to a NORMALIZED
        # COMMANDED ANGLE, matching training's kick_aim_command observation term exactly:
        # dim 0 = kick_aim_theta_deg / kick_aim_theta_ref_deg (bounded in [-1, 1] the same way
        # training's config-load-time validation enforces theta_max_deg <= theta_ref_deg), dim 1
        # reserved (0.0). This constant, unlike kick_target_dx/dy under the OLD scheme, is now
        # actually CORRECT relative to what training used -- training also holds theta constant
        # for the whole attempt (see MotionCommand.kick_aim_theta's own docstring) -- rather than
        # an approximation of a value that was supposed to shrink/rotate as the robot moved.
        # kick_aim_theta_ref_deg here MUST match the value the deployed checkpoint was trained
        # with (MultiSkillConfig/BallConfig.kick_aim_theta_ref_deg) -- it is a normalization
        # constant baked into the checkpoint's own learned input scale, not a free parameter.
        if config.task.kick_aim_enabled:
            self._target_pos_body_frame = np.array(
                [[config.task.kick_aim_theta_deg / config.task.kick_aim_theta_ref_deg, 0.0]]
            )
        else:
            self._target_pos_body_frame = np.array([[config.task.kick_target_dx, config.task.kick_target_dy]])
        # Approximation, not a measurement -- see kick_ball_height_m's docstring for why a real
        # value can't be computed from get_low_state() alone.
        self._ball_z_rel_approx = config.task.kick_ball_height_m - config.task.desired_base_height
        self._warned_no_ball_reading = False

        super().__init__(config)

    # ------------------------------------------------------------------
    # Policy / ONNX setup — same shape as WholeBodyTrackingPolicy.setup_policy,
    # since our ONNX export also embeds the kick clip as a time_step-indexed lookup
    # (UnifiedManager registers a real MotionCommand term, so FastSACAgent.export()
    # takes the same export_motion_and_policy_as_onnx path WBT uses).

    def setup_policy(self, model_path):
        self.onnx_policy_session = onnxruntime.InferenceSession(model_path)
        self.onnx_input_names = [inp.name for inp in self.onnx_policy_session.get_inputs()]
        self.onnx_output_names = [out.name for out in self.onnx_policy_session.get_outputs()]

        onnx_model = onnx.load(model_path)
        metadata = {}
        for prop in onnx_model.metadata_props:
            metadata[prop.key] = json.loads(prop.value)

        assert "robot_urdf" in metadata, "Robot urdf text not found in ONNX metadata"
        self.pinocchio_robot = PinocchioRobot(self.config.robot, metadata["robot_urdf"])

        self.onnx_kp = np.array(metadata["kp"]) if "kp" in metadata else None
        self.onnx_kd = np.array(metadata["kd"]) if "kd" in metadata else None

        # Get the clip's frame-0 target once, up front (used before any kick trigger,
        # and as the "reset to start" snapshot on each new trigger).
        time_step = np.array([[0.0]], dtype=np.float32)
        actor_obs_template = self.obs_buf_dict.get("actor_obs")
        if actor_obs_template is None:
            raise ValueError("Observation group 'actor_obs' must be configured for UnifiedPolicy.")
        obs = actor_obs_template.copy()
        outputs = self.onnx_policy_session.run(["joint_pos", "joint_vel", "ref_quat_xyzw"], {"obs": obs, "time_step": time_step})
        self.motion_command_t = np.concatenate(outputs[0:2], axis=1)  # (1, 58)
        self.ref_quat_xyzw_t = outputs[2]
        self.motion_command_0 = self.motion_command_t.copy()
        self.ref_quat_xyzw_0 = self.ref_quat_xyzw_t.copy()

        def policy_act(input_feed):
            output = self.onnx_policy_session.run(["actions", "joint_pos", "joint_vel", "ref_quat_xyzw"], input_feed)
            action = output[0]
            motion_command = np.concatenate(output[1:3], axis=1)
            ref_quat_xyzw = output[3]
            return action, motion_command, ref_quat_xyzw

        self.policy = policy_act

    def _configure_action_scales(self) -> None:
        """Per-joint action scale from ONNX metadata, same resolution order as WBT."""
        raw_metadata = dict(self.onnx_policy_session.get_modelmeta().custom_metadata_map)
        raw_scale = raw_metadata.get("action_scale")
        self.per_joint_policy_action_scale = None
        if raw_scale is None:
            return
        try:
            parsed = json.loads(raw_scale)
        except json.JSONDecodeError:
            parsed = raw_scale
        values = np.asarray(parsed, dtype=np.float32).reshape(-1)
        if values.size == 1:
            values = np.full(self.num_dofs, values.item(), dtype=np.float32)
        elif values.size != self.num_dofs:
            raise ValueError(f"Action scale must contain 1 or {self.num_dofs} values, got {values.size}.")
        self.per_joint_policy_action_scale = values.reshape(1, -1)

    def _on_policy_switched(self, model_path: str):
        super()._on_policy_switched(model_path)
        self._configure_action_scales()

    # ------------------------------------------------------------------
    # Ref-body orientation (for kick_motion_ref_ori_b) — identical to
    # WholeBodyTrackingPolicy._get_ref_body_orientation_in_world.

    def _get_ref_body_orientation_in_world(self, robot_state_data):
        root_pos = robot_state_data[0, :3]
        root_ori_xyzw = wxyz_to_xyzw(robot_state_data[:, 3:7])[0]
        dof_pos_in_real = robot_state_data[0, 7 : 7 + self.num_dofs]
        dof_pos_in_pinocchio = dof_pos_in_real[self.pinocchio_robot.real2pinocchio_index]
        configuration = np.concatenate([root_pos, root_ori_xyzw, dof_pos_in_pinocchio], axis=0)
        ref_ori_xyzw = self.pinocchio_robot.fk_and_get_ref_body_orientation_in_world(configuration)
        return xyzw_to_wxyz(ref_ori_xyzw)

    def _remove_yaw_offset(self, quat_wxyz: np.ndarray, yaw_offset: float) -> np.ndarray:
        if abs(yaw_offset) < 1e-6:
            return quat_wxyz
        yaw_quat = rpy_to_quat((0.0, 0.0, -yaw_offset)).reshape(1, 4)
        yaw_quat = np.broadcast_to(yaw_quat, quat_wxyz.shape)
        return quat_mul(yaw_quat, quat_wxyz)

    @staticmethod
    def _quat_yaw(quat_wxyz: np.ndarray) -> float:
        quat_flat = quat_wxyz.reshape(-1, 4)[0]
        _, _, yaw = quat_to_rpy(quat_flat)
        return float(yaw)

    def _capture_yaw_offsets(self, robot_state_data):
        """Capture robot/motion yaw at the moment a kick is triggered, so
        kick_motion_ref_ori_b compares orientations relative to the robot's actual
        heading at trigger time (matching WBT's own _capture_robot_yaw_offset/
        _capture_motion_yaw_offset, just triggered by TRIGGER_KICK instead of START)."""
        robot_ref_ori = self._get_ref_body_orientation_in_world(robot_state_data)
        self.robot_yaw_offset = self._quat_yaw(robot_ref_ori)
        self.motion_yaw_offset = self._quat_yaw(xyzw_to_wxyz(self.ref_quat_xyzw_0))

    # ------------------------------------------------------------------
    # Observations — build BOTH full term sets each tick (identical physical
    # readings feed both loco_* and kick_* proprioceptive terms; only the kick-clip
    # terms and command terms actually differ), then zero out whichever task isn't
    # active — mirroring UnifiedManager.task_mode_mask()'s all-or-nothing masking of
    # every term tagged with the inactive task, not just the command terms.

    def get_current_obs_buffer_dict(self, robot_state_data):
        base = super().get_current_obs_buffer_dict(robot_state_data)  # base_ang_vel, dof_pos, dof_vel, ...
        is_kick = self.task_mode == "kick"

        obs = {}

        # --- locomotion terms ---
        obs["loco_base_ang_vel"] = np.zeros((1, 3)) if is_kick else base["base_ang_vel"]
        obs["loco_projected_gravity"] = np.zeros((1, 3)) if is_kick else base["projected_gravity"]
        obs["loco_command_lin_vel"] = np.zeros((1, 2)) if is_kick else self.lin_vel_command
        obs["loco_command_ang_vel"] = np.zeros((1, 1)) if is_kick else self.ang_vel_command
        obs["loco_dof_pos"] = np.zeros((1, self.num_dofs)) if is_kick else base["dof_pos"]
        obs["loco_dof_vel"] = np.zeros((1, self.num_dofs)) if is_kick else base["dof_vel"]
        obs["loco_actions"] = np.zeros((1, self.num_dofs)) if is_kick else self.last_policy_action
        obs["loco_sin_phase"] = np.zeros((1, 2)) if is_kick else np.array([np.sin(self.phase[0, :])])
        obs["loco_cos_phase"] = np.zeros((1, 2)) if is_kick else np.array([np.cos(self.phase[0, :])])

        # --- kick terms ---
        if is_kick:
            obs["kick_motion_command"] = self.motion_command_t

            motion_ref_ori = xyzw_to_wxyz(self.ref_quat_xyzw_t)
            motion_ref_ori = self._remove_yaw_offset(motion_ref_ori, self.motion_yaw_offset)
            robot_ref_ori = self._get_ref_body_orientation_in_world(robot_state_data)
            robot_ref_ori = self._remove_yaw_offset(robot_ref_ori, self.robot_yaw_offset)
            motion_ref_ori_b = matrix_from_quat(subtract_frame_transforms(robot_ref_ori, motion_ref_ori))
            obs["kick_motion_ref_ori_b"] = motion_ref_ori_b[..., :2].reshape(1, -1)

            obs["kick_base_ang_vel"] = base["base_ang_vel"]
            obs["kick_dof_pos"] = base["dof_pos"]
            obs["kick_dof_vel"] = base["dof_vel"]
            obs["kick_actions"] = self.last_policy_action

            # kick_ball_pos_b / kick_target_pos_b -- see training's ball_pos_b/target_pos_b
            # (managers/observation/terms/unified.py) for the ground-truth heading-frame
            # definition this approximates from real/sim sensor data. Both zeroed together if no
            # ball reading is available -- an is_kick tick with a real ball position but a
            # zeroed/undefined target (or vice versa) doesn't correspond to anything the network
            # saw in training, so partial data is treated the same as no data.
            ball_dxdy = self.ball_pose_source.get_ball_pos_body_frame() if self.ball_pose_source is not None else None
            if ball_dxdy is not None:
                obs["kick_ball_pos_b"] = np.array([[ball_dxdy[0], ball_dxdy[1], self._ball_z_rel_approx]])
                obs["kick_target_pos_b"] = self._target_pos_body_frame
                self._warned_no_ball_reading = False
            else:
                obs["kick_ball_pos_b"] = np.zeros((1, 3))
                obs["kick_target_pos_b"] = np.zeros((1, 2))
                if not self._warned_no_ball_reading:
                    self.logger.warning(
                        colored(
                            "task_mode=='kick' but no ball reading available (kick_ball_source="
                            f"{self.config.task.kick_ball_source!r}) -- zeroing kick_ball_pos_b/"
                            "kick_target_pos_b, which is out-of-distribution relative to training "
                            "(these are only zero when task_mode=='locomotion' during training). "
                            "Set --task.kick-ball-source fixed|udp to provide a real reading.",
                            "yellow",
                        )
                    )
                    self._warned_no_ball_reading = True
        else:
            obs["kick_motion_command"] = np.zeros((1, 58))
            obs["kick_motion_ref_ori_b"] = np.zeros((1, 6))
            obs["kick_base_ang_vel"] = np.zeros((1, 3))
            obs["kick_dof_pos"] = np.zeros((1, self.num_dofs))
            obs["kick_dof_vel"] = np.zeros((1, self.num_dofs))
            obs["kick_actions"] = np.zeros((1, self.num_dofs))
            obs["kick_ball_pos_b"] = np.zeros((1, 3))
            obs["kick_target_pos_b"] = np.zeros((1, 2))

        obs["task_mode_onehot"] = np.array([[0.0, 1.0]]) if is_kick else np.array([[1.0, 0.0]])

        return obs

    # ------------------------------------------------------------------
    # Phase (locomotion gait signal) — identical to LocomotionPolicy.update_phase_time.

    def update_phase_time(self):
        phase_tp1 = self.phase + self.phase_dt
        self.phase = np.fmod(phase_tp1 + np.pi, 2 * np.pi) - np.pi
        if np.linalg.norm(self.lin_vel_command[0]) < 0.01 and np.linalg.norm(self.ang_vel_command[0]) < 0.01:
            # Robot should stand still - set both feet to same phase
            self.phase[0, :] = np.pi * np.ones(2)
            self.is_standing = True
        elif self.is_standing:
            # When the robot starts to move, reset the phase to initial state (one-shot:
            # clearing is_standing here means this branch only fires on the
            # standing->walking transition, not every tick while already walking --
            # otherwise the gait phase would never progress past its reset value).
            self.phase = np.array([[0.0, np.pi]])
            self.is_standing = False

    # ------------------------------------------------------------------
    # RL inference — same time_step-driven ONNX call as
    # WholeBodyTrackingPolicy.rl_inference: the clip is always available from the
    # ONNX (time_step input), we just only advance it (and only feed its output into
    # the actor's kick_* obs slots) while task_mode == "kick".

    def rl_inference(self, robot_state_data):
        obs = self.prepare_obs_for_rl(robot_state_data)
        if self.config.task.print_observations:
            self._print_observations(obs)

        prev_motion_command_t = self.motion_command_t

        input_feed = {"time_step": np.array([[self.curr_motion_timestep]], dtype=np.float32), "obs": obs["actor_obs"]}
        policy_action, self.motion_command_t, self.ref_quat_xyzw_t = self.policy(input_feed)

        policy_action = np.clip(policy_action, -100, 100)
        self.last_policy_action = policy_action.copy()
        if self.per_joint_policy_action_scale is None:
            self.scaled_policy_action = policy_action * self.policy_action_scale
        else:
            self.scaled_policy_action = policy_action * self.per_joint_policy_action_scale

        # Auto-return to locomotion once the kick clip has clearly finished and held
        # for a few seconds — the ONNX clamps time_step at the clip's last (hold)
        # frame, so the clip output stops changing tick-to-tick; that's the only
        # signal available on this side (see class docstring). Avoids relying on the
        # user remembering to press the return-to-locomotion key.
        #
        # Guarded by a minimum-elapsed-time floor: some clips have an early
        # low-motion/held segment (e.g. a wind-up pose before the actual kick), which
        # would otherwise look identical to "the clip has finished" to this
        # tick-to-tick-unchanged check and cut the kick off prematurely — confirmed
        # empirically (fired ~3.4s into a clip whose kick+recovery portion runs
        # ~7.6s). The floor doesn't need to be exact, just safely shorter than any
        # realistic kick clip's true length.
        _MIN_TICKS_BEFORE_AUTO_RETURN = int(5 * self.rl_rate)
        if (
            self.task_mode == "kick"
            and self.motion_clip_progressing
            and self.curr_motion_timestep >= _MIN_TICKS_BEFORE_AUTO_RETURN
        ):
            if prev_motion_command_t is not None and np.allclose(self.motion_command_t, prev_motion_command_t):
                self._kick_hold_ticks += 1
            else:
                self._kick_hold_ticks = 0
            if self._kick_hold_ticks >= int(3 * self.rl_rate):
                self.logger.info(colored("Kick clip finished, auto-returning to locomotion control", "blue"))
                self._handle_return_to_locomotion()

        if self.motion_clip_progressing:
            self.curr_motion_timestep += 1

        return self.scaled_policy_action

    # ------------------------------------------------------------------
    # Commands

    def _handle_trigger_kick(self, robot_state_data):
        self.task_mode = "kick"
        self.curr_motion_timestep = 0
        self.motion_clip_progressing = True
        self._kick_hold_ticks = 0
        self._capture_yaw_offsets(robot_state_data)
        self.logger.info(colored("Kick triggered", "blue"))

    def _handle_return_to_locomotion(self):
        self.task_mode = "locomotion"
        self.motion_clip_progressing = False
        self.curr_motion_timestep = 0
        self._kick_hold_ticks = 0
        self.motion_command_t = self.motion_command_0.copy()
        self.ref_quat_xyzw_t = self.ref_quat_xyzw_0.copy()
        self.robot_yaw_offset = 0.0
        self.motion_yaw_offset = 0.0
        self.logger.info(colored("Returned to locomotion control", "blue"))

    def _handle_zero_velocity(self):
        self._velocity_input.zero()
        self.ang_vel_command[0, 0] = 0.0
        self.lin_vel_command[0, 0] = 0.0
        self.lin_vel_command[0, 1] = 0.0
        self.logger.info(colored("Velocities set to zero", "blue"))

    def _dispatch_command(self, cmd):
        if cmd == StateCommand.TRIGGER_KICK:
            robot_state_data = self.interface.get_low_state()
            self._handle_trigger_kick(robot_state_data)
        elif cmd == StateCommand.RETURN_TO_LOCOMOTION:
            self._handle_return_to_locomotion()
        elif cmd == StateCommand.ZERO_VELOCITY:
            self._handle_zero_velocity()
        else:
            super()._dispatch_command(cmd)

    def _print_control_status(self):
        super()._print_control_status()
        lin_vel_x, lin_vel_y = self.lin_vel_command[0]
        ang_vel_z = self.ang_vel_command[0, 0]
        print(f"Task mode: {self.task_mode}")
        print(f"Linear velocity: x={lin_vel_x:+.2f} m/s, y={lin_vel_y:+.2f} m/s")
        print(f"Angular velocity: {ang_vel_z:+.2f} rad/s")
        print("💡 Terminal keys: W/A/S/D (lin) | Q/E (ang) | K (trigger kick) | L (return to locomotion)")
