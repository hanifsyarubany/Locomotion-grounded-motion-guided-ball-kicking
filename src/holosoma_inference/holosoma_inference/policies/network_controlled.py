from __future__ import annotations

import time

from termcolor import colored

from holosoma_inference.hl_command_channel import HlCommandReceiver
from holosoma_inference.policies.unified import UnifiedPolicy


class NetworkControlledUnifiedPolicy(UnifiedPolicy):
    """The LL-process half of the genuinely-separate-process HL/LL split: identical to
    UnifiedPolicy except velocity commands and the kick trigger come from a UDP channel (see
    hl_command_channel.py) written by a separate HL controller process
    (run_classical_hl_controller.py), instead of the keyboard.

    This is the parallel-processes counterpart to ApproachAndKickPolicy, which runs the same
    classical control law in-process instead. Both consume the exact same
    ClassicalApproachAndKickController logic (see classical_controller.py) -- the only difference
    is whether the HL decision-making and LL inference share a process (ApproachAndKickPolicy) or
    run as two independent OS processes communicating over UDP (this class + the HL controller
    script), to directly test whether process separation itself changes closed-loop behavior.

    No ball-position handling here at all -- that lives entirely on the HL controller side, which
    already consumes a BallPoseSource itself. This class only ever sees velocity + trigger
    commands, matching what a real hardware LL controller receiving commands from a separate
    motion-planning process would see.
    """

    def __init__(self, config, command_port: int, filter_out_loco_s: float = 0.0):
        super().__init__(config)
        self.receiver = HlCommandReceiver(command_port)
        self._last_trigger_flag = False
        self.filter_out_loco_s = filter_out_loco_s
        # Monotonic timestamp of the most recent kick trigger. The --filter-out-loco window is
        # measured from HERE -- the instant the trigger is received -- exactly per the spec:
        # "the locomotion will be zeroed for N seconds right after the kicking trigger; if a new
        # trigger arrives before the N seconds finish it is ignored; the controller is ready for
        # a new trigger only after the count finishes."
        #
        # Note the window spans the kick clip itself: for the first ~7.6s of a typical window the
        # robot is executing the kick (task_mode=="kick"), during which _apply_network_command()
        # isn't even called (policy_action() only calls it while task_mode=="locomotion") and the
        # locomotion command is already zeroed by task-mode obs masking anyway. The window's real
        # work happens on the tail end -- once the kick returns to locomotion control with time
        # still left on the clock, it holds the command at zero for the remainder -- plus the
        # trigger-suppression, which spans the whole window (see _apply_network_command).
        self._filter_start_monotonic: float | None = None

    def policy_action(self) -> None:
        if self.use_policy_action and not self.get_ready_state and self.task_mode == "locomotion":
            self._apply_network_command()
        super().policy_action()

    def _handle_trigger_kick(self, robot_state_data) -> None:
        # Stamp the filter-window start on EVERY kick trigger, whatever its source (the network
        # command path below, or a manual keyboard 'k' via UnifiedPolicy._dispatch_command) --
        # this override is the single choke point all triggers pass through.
        super()._handle_trigger_kick(robot_state_data)
        self._filter_start_monotonic = time.monotonic()

    def get_applied_command(self) -> tuple[float, float, float]:
        """The ACTUAL (post-filter) command currently applied to the robot -- vx, vy, wyaw. Not
        the same thing as the raw HlCommand received over UDP (see hl_command_channel.py) --
        --filter-out-loco can override this to zero while the raw received command is still
        whatever the HL process is sending. Pure observability (e.g. hl_command_gui.py); the
        control path uses self.lin_vel_command/self.ang_vel_command directly, this is a read-only
        accessor onto that same state."""
        return float(self.lin_vel_command[0][0]), float(self.lin_vel_command[0][1]), float(self.ang_vel_command[0][0])

    def loco_filter_remaining_s(self) -> float:
        """Seconds left in the current --filter-out-loco window (measured from the last trigger),
        or 0.0 if no window is active. Used both as the control-path gate in
        _apply_network_command AND as the GUI's LOCO: FILTERED countdown, so the display and the
        actual behavior can never disagree."""
        if self.filter_out_loco_s <= 0.0 or self._filter_start_monotonic is None:
            return 0.0
        remaining = self.filter_out_loco_s - (time.monotonic() - self._filter_start_monotonic)
        return max(0.0, remaining)

    def _apply_network_command(self) -> None:
        cmd = self.receiver.get_latest()
        if cmd is None:
            # No (or stale) command from the HL process -- hold position rather than continue on
            # stale data, same discipline as ApproachAndKickPolicy's ball_pose_source handling.
            self.lin_vel_command[0] = 0.0
            self.ang_vel_command[0, 0] = 0.0
            self._last_trigger_flag = False
            return

        self.lin_vel_command[0] = [cmd.vx, cmd.vy]
        self.ang_vel_command[0, 0] = cmd.wyaw

        # Kick trigger: edge-triggered (False->True only, since trigger_kick is a held level in
        # the wire format, not a one-shot event) AND suppressed while a --filter-out-loco window
        # is still counting down. The suppression is the spec's "if a new trigger arrives before
        # the N seconds finish it is ignored" -- during the window the controller stays in
        # locomotion, force-zeroed, and simply drops any incoming trigger; only once the window
        # elapses (loco_filter_remaining_s() back to 0) does a fresh trigger edge take effect.
        # _handle_trigger_kick (overridden above) restamps the window start, so an accepted
        # trigger begins its own N-second window from that instant.
        if cmd.trigger_kick and not self._last_trigger_flag and self.loco_filter_remaining_s() <= 0.0:
            robot_state_data = self.interface.get_low_state()
            self.logger.info(colored("Received remote kick trigger", "blue"))
            self._handle_trigger_kick(robot_state_data)
        self._last_trigger_flag = cmd.trigger_kick

        # Force-zero locomotion for the remainder of the window. Re-checked AFTER the trigger
        # handling above so the very tick a trigger is accepted is already zeroed (the trigger
        # just started the window). In practice task_mode flips to "kick" on that same tick and
        # obs-masking zeroes the loco command regardless, but keeping the two consistent here
        # avoids any single-tick leak of a non-zero command.
        if self.loco_filter_remaining_s() > 0.0:
            self.lin_vel_command[0] = 0.0
            self.ang_vel_command[0, 0] = 0.0
