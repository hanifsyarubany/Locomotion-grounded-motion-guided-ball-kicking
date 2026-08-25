from __future__ import annotations

import dataclasses
import math

import numpy as np


@dataclasses.dataclass
class ControllerOutput:
    vx: float
    vy: float
    wyaw: float
    trigger_kick: bool
    trigger_reason: str | None = None  # "natural" | "forced", only set when trigger_kick is True
    region_probability: float | None = None  # p_success at the current ball position, if a
    # KickRegionMap is active -- None in classic fixed-target mode. Useful for logging/telemetry
    # even on non-trigger ticks (e.g. to watch the approach climb toward the threshold).


class ClassicalApproachAndKickController:
    """The approach+kick control law itself -- continuous proportional feedback on the
    body-frame ball-offset error + a settle/gate/trigger state machine -- extracted into a
    standalone, framework-agnostic class so it can run either in-process (see
    ApproachAndKickPolicy, which calls this directly and applies the output to its own
    lin_vel_command/ang_vel_command/_handle_trigger_kick) or as a genuinely separate process
    (see run_classical_hl_controller.py, which calls this and publishes the output over UDP to a
    separate LL process instead). Deliberately extracted rather than duplicated: this is the same
    validated logic either way, not a reimplementation -- see
    playground/high_level_ball_approach_and_kicking/README.md's "Alternative to HL-RL" section for
    the full derivation and tuning history behind these defaults.

    `compute()` is pure aside from the internal state machine (`_approach_phase`,
    `_hold_step_count`, `_prev_dof_pos`) -- takes the current robot low-state array (same fixed
    schema as UnitreeInterface.get_low_state()) and a body-frame ball offset (or None if
    unavailable), returns what velocity to command and whether to trigger a kick this tick. Does
    not touch any interface/network/policy object itself.

    Two trigger-gating modes, selected by whether `kick_region_map_path` is given:
    - **Classic (default, unchanged from before)**: seeking->settling transition gated on
      Euclidean distance to a single fixed `(target_dx, target_dy)` (`align_gate_radius_m`/
      `abort_radius_m`).
    - **Region-probabilistic** (opt-in): transition gated on the live body-frame ball position's
      P(success) (`KickRegionMap.probability_at`, a 95% Wilson-lower-bound estimate from
      `map_kick_region_prob.py`'s empirical sweep -- see
      playground/high_level_ball_approach_and_kicking/README.md's "Probabilistic kick-success
      region map" section) crossing `success_probability_threshold` -- e.g. "only start settling,
      and eventually trigger, once the ball is in a cell measured at >=72% success" -- instead of
      a hand-picked distance radius around a hand-picked point. The P-law still needs a single
      point to steer toward (proportional control has no notion of "anywhere in this region");
      that point defaults to the map's own best-supported cell (`KickRegionMap.best_point`) unless
      `target_dx`/`target_dy` are given explicitly, in which case those win regardless of the map.
      The stillness gate (`angular_velocity_gate_radps`/`joint_velocity_gate`/`settle_steps`) and
      the `max_settle_wait_steps` forced-timeout fallback are unchanged either way -- being inside
      a high-probability region doesn't mean the robot is currently still.
    """

    def __init__(
        self,
        num_dofs: int,
        rl_rate: float,
        target_dx: float | None = None,
        target_dy: float | None = None,
        kp_lin: float = 0.8,
        kp_yaw: float = 2.0,
        max_speed: float = 0.7,
        max_yaw_rate: float = 0.9,
        align_gate_radius_m: float = 0.5,
        abort_radius_m: float = 1.0,
        settle_steps: int = 100,
        max_settle_wait_steps: int = 150,
        angular_velocity_gate_radps: float = 0.3,
        joint_velocity_gate: float = 4.0,
        kick_region_map_path: str | None = None,
        success_probability_threshold: float = 0.72,
        region_abort_threshold: float | None = None,
        region_min_samples: int = 100,
        region_settle_fallback_radius_m: float = 0.2,
    ):
        self.num_dofs = num_dofs
        self.rl_rate = rl_rate
        self.kp_lin = kp_lin
        self.kp_yaw = kp_yaw
        self.max_speed = max_speed
        self.max_yaw_rate = max_yaw_rate
        self.align_gate_radius_m = align_gate_radius_m
        self.abort_radius_m = abort_radius_m
        self.settle_steps = settle_steps
        self.max_settle_wait_steps = max_settle_wait_steps
        self.angular_velocity_gate_radps = angular_velocity_gate_radps
        self.joint_velocity_gate = joint_velocity_gate

        self.region_map = None
        if kick_region_map_path is not None:
            from holosoma_inference.kick_region_map import KickRegionMap  # noqa: PLC0415

            self.region_map = KickRegionMap.load(kick_region_map_path)
            if target_dx is None or target_dy is None:
                best_dx, best_dy, best_p = self.region_map.best_point(min_samples=region_min_samples)
                target_dx = best_dx if target_dx is None else target_dx
                target_dy = best_dy if target_dy is None else target_dy

        # Classic fixed-target default, only applied if still unset (no map, or map given but
        # somehow empty -- best_point() itself raises in that case, so this is just the plain
        # no-map fallback in practice).
        self.target = np.array([target_dx if target_dx is not None else 1.6, target_dy if target_dy is not None else 0.0])
        self.success_probability_threshold = success_probability_threshold
        self.region_abort_threshold = (
            region_abort_threshold if region_abort_threshold is not None else success_probability_threshold * 0.5
        )
        self.region_min_samples = region_min_samples
        self.region_settle_fallback_radius_m = region_settle_fallback_radius_m

        self._approach_phase = "seeking"  # "seeking" | "settling"
        self._hold_step_count = 0
        self._prev_dof_pos: np.ndarray | None = None

    def reset(self) -> None:
        """Call whenever control returns to locomotion after a kick concludes, so the next
        autonomous cycle starts clean (matches ApproachAndKickPolicy._dispatch_command)."""
        self._approach_phase = "seeking"
        self._hold_step_count = 0
        self._prev_dof_pos = None

    def compute(self, robot_state_data: np.ndarray, ball_body: np.ndarray | None) -> ControllerOutput:
        if ball_body is None:
            # No (or stale) ball reading -- hold position rather than continue on stale data.
            return ControllerOutput(vx=0.0, vy=0.0, wyaw=0.0, trigger_kick=False)

        error = ball_body - self.target
        error_dist = float(np.linalg.norm(error))
        bearing = math.atan2(ball_body[1], ball_body[0])  # bearing to the ball itself, not the error vector

        vx = float(np.clip(self.kp_lin * error[0], -self.max_speed, self.max_speed))
        vy = float(np.clip(self.kp_lin * error[1], -self.max_speed, self.max_speed))
        wyaw = float(np.clip(self.kp_yaw * bearing, -self.max_yaw_rate, self.max_yaw_rate))

        region_probability = None
        if self.region_map is not None:
            region_probability = self.region_map.probability_at(float(ball_body[0]), float(ball_body[1]))
            # Proximity fallback: self.target IS the map's best-supported cell (best_point), so
            # once the robot has physically arrived within region_settle_fallback_radius_m of it,
            # settle and kick even if this exact discretized cell's probability is marginally under
            # threshold. Without this the robot deadlocks: P-control parks it right on the target
            # with ~zero error (nothing left to drive further motion) while the gate stays closed
            # in an adjacent below-threshold cell -- it freezes next to the ball without ever
            # kicking. Whether it froze depended on which side of a ~0.1m cell boundary the ball
            # happened to settle, which is exactly why it was intermittent. Reaching the target is
            # itself the "best spot available" signal, so kicking there is the right call.
            reached_target = error_dist < self.region_settle_fallback_radius_m
            if self._approach_phase == "seeking" and (
                region_probability >= self.success_probability_threshold or reached_target
            ):
                self._approach_phase = "settling"
                self._hold_step_count = 0
            # Don't abort out of settling while parked on the target -- otherwise a target cell
            # whose probability sits between region_abort_threshold and success_probability_threshold
            # would ping-pong seeking<->settling forever (the fallback re-settles it, the abort
            # kicks it back out) instead of proceeding to a kick.
            elif (
                self._approach_phase == "settling"
                and region_probability < self.region_abort_threshold
                and not reached_target
            ):
                self._approach_phase = "seeking"
                self._hold_step_count = 0
        else:
            if self._approach_phase == "seeking" and error_dist < self.align_gate_radius_m:
                self._approach_phase = "settling"
                self._hold_step_count = 0
            elif self._approach_phase == "settling" and error_dist > self.abort_radius_m:
                self._approach_phase = "seeking"
                self._hold_step_count = 0

        if self._approach_phase != "settling":
            return ControllerOutput(vx=vx, vy=vy, wyaw=wyaw, trigger_kick=False, region_probability=region_probability)

        dof_pos = np.asarray(robot_state_data[0, 7 : 7 + self.num_dofs])
        dof_vel_norm = (
            0.0
            if self._prev_dof_pos is None
            else float(np.linalg.norm((dof_pos - self._prev_dof_pos) * self.rl_rate))
        )
        self._prev_dof_pos = dof_pos

        base_ang_vel = np.asarray(robot_state_data[0, 7 + self.num_dofs + 3 : 7 + self.num_dofs + 6])

        # NOTE real-hardware limitation, not a placeholder to "fix" later: get_low_state()'s wire
        # schema has no base LINEAR velocity field at all (always zeros). See
        # ApproachAndKickPolicy's original docstring for the full reasoning -- angular velocity
        # (real IMU) and joint velocity (real, encoder-derived) are what's actually available.
        gate_ok = (
            float(np.linalg.norm(base_ang_vel)) < self.angular_velocity_gate_radps
            and dof_vel_norm < self.joint_velocity_gate
        )
        settle_ok = self._hold_step_count >= self.settle_steps
        forced = self._hold_step_count >= self.max_settle_wait_steps

        if (settle_ok and gate_ok) or forced:
            self._approach_phase = "seeking"
            self._hold_step_count = 0
            return ControllerOutput(
                vx=0.0, vy=0.0, wyaw=0.0, trigger_kick=True,
                trigger_reason="forced" if not (settle_ok and gate_ok) else "natural",
                region_probability=region_probability,
            )

        self._hold_step_count += 1
        return ControllerOutput(vx=vx, vy=vy, wyaw=wyaw, trigger_kick=False, region_probability=region_probability)
