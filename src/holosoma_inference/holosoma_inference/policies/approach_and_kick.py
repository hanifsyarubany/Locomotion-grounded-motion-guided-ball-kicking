from __future__ import annotations

from termcolor import colored

from holosoma_inference.ball_pose_source import BallPoseSource
from holosoma_inference.policies.classical_controller import ClassicalApproachAndKickController
from holosoma_inference.policies.unified import UnifiedPolicy


class ApproachAndKickPolicy(UnifiedPolicy):
    """Autonomous approach+kick controller for real (or bridge-simulated) deployment: drives
    velocity commands and the kick trigger from a ball-position estimate instead of a human at
    the keyboard.

    Deliberately a thin wrapper, not a reimplementation: reuses UnifiedPolicy's LL execution
    (observation construction, ONNX inference, the kick-clip state machine) completely unchanged
    -- overrides only `policy_action()` to inject the classical control law before calling
    `super().policy_action()` for the actual LL step. The control law itself lives in
    `ClassicalApproachAndKickController` (classical_controller.py), shared with the genuinely
    separate-process version (run_classical_hl_controller.py) -- this class is the single-process
    consumer of that shared logic, not a second implementation of it.

    Ball position comes from a `BallPoseSource` (see ball_pose_source.py) -- this class has no
    opinion on whether that's simulated ground truth, a UDP test publisher, or eventually real
    perception; it only consumes `get_ball_pos_body_frame() -> (dx, dy) | None`.

    Safety: autonomous control only runs once the operator has sent the normal START command
    (`self.use_policy_action`, same gate BasePolicy already uses for keyboard/joystick control) --
    this does not remove the existing human-in-the-loop start gate, only replaces manual
    velocity/kick-trigger input with the autonomous controller once started.
    """

    def __init__(
        self,
        config,
        ball_pose_source: BallPoseSource,
        target_dx: float | None = None,
        target_dy: float | None = None,
        kick_region_map_path: str | None = None,
        success_probability_threshold: float = 0.72,
    ):
        super().__init__(config)
        self.ball_pose_source = ball_pose_source
        # Tuned defaults -- see README's "Alternative to HL-RL" section for the validation
        # history behind these exact values. Kept in sync manually; not re-derived here.
        # With kick_region_map_path set, target_dx/target_dy (if left None) auto-derive from the
        # map's best-supported cell instead, and the trigger gates on live P(success) crossing
        # success_probability_threshold rather than distance to a fixed point -- see
        # ClassicalApproachAndKickController's docstring for the full mechanism.
        self.controller = ClassicalApproachAndKickController(
            num_dofs=self.num_dofs, rl_rate=self.rl_rate, target_dx=target_dx, target_dy=target_dy,
            kick_region_map_path=kick_region_map_path, success_probability_threshold=success_probability_threshold,
        )

    # ------------------------------------------------------------------

    def policy_action(self) -> None:
        if self.use_policy_action and not self.get_ready_state and self.task_mode == "locomotion":
            self._update_autonomous_control()
        super().policy_action()

    def _update_autonomous_control(self) -> None:
        robot_state_data = self.interface.get_low_state()
        ball_body = self.ball_pose_source.get_ball_pos_body_frame()
        out = self.controller.compute(robot_state_data, ball_body)

        self.lin_vel_command[0] = [out.vx, out.vy]
        self.ang_vel_command[0, 0] = out.wyaw

        if out.trigger_kick:
            p_str = f", p_success={out.region_probability:.3f}" if out.region_probability is not None else ""
            self.logger.info(colored(f"Autonomous kick trigger ({out.trigger_reason}{p_str})", "blue"))
            self._handle_trigger_kick(robot_state_data)

    def _dispatch_command(self, cmd) -> None:
        super()._dispatch_command(cmd)
        # Reset the approach state machine whenever a kick concludes and control returns to
        # locomotion, so the next autonomous cycle starts clean (matches
        # HighLevelKickEnv._handle_episode_end's per-trial reset in the IsaacSim/MuJoCo sweeps).
        if self.task_mode == "locomotion":
            self.controller.reset()
