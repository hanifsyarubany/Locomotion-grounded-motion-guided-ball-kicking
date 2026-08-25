"""UDP channel carrying velocity + kick-trigger commands from a standalone HL controller process
(run_classical_hl_controller.py) to a standalone LL policy process (run_network_controlled_policy.py
/ NetworkControlledUnifiedPolicy) -- the genuinely-separate-process counterpart to
ApproachAndKickPolicy's in-process call into ClassicalApproachAndKickController.

Same wire-format spirit as ball_pose_source.py: a fixed-size struct-packed UDP datagram, sent at
the HL controller's own tick rate (not request/response), consumed non-blockingly by the LL
process. `trigger_kick` is a level (not edge) in the wire format -- the sender is responsible for
only setting it True for the tick(s) it actually wants to signal a trigger (it already only fires
once per approach cycle, see ClassicalApproachAndKickController), and the receiver edge-detects
(only acts on False->True transitions) so a repeated or duplicated packet can't double-trigger.
"""

from __future__ import annotations

import socket
import struct
import threading
import time
from typing import NamedTuple, Optional

_STRUCT_FMT = "<ddd?"
_STRUCT_SIZE = struct.calcsize(_STRUCT_FMT)


class HlCommand(NamedTuple):
    vx: float
    vy: float
    wyaw: float
    trigger_kick: bool


def send_hl_command(sock: socket.socket, port: int, cmd: HlCommand) -> None:
    sock.sendto(struct.pack(_STRUCT_FMT, cmd.vx, cmd.vy, cmd.wyaw, cmd.trigger_kick), ("127.0.0.1", port))


class HlCommandReceiver:
    """Non-blocking receiver: get_latest() always returns immediately with the most recent
    command, or None if nothing has arrived within `staleness_timeout_s` (guards against acting
    on a reading from before the HL process died or a network hiccup -- same staleness discipline
    as UdpBallPoseSource)."""

    def __init__(self, port: int, staleness_timeout_s: float = 0.5):
        self._staleness_timeout_s = staleness_timeout_s
        self._lock = threading.Lock()
        self._latest: Optional[HlCommand] = None
        self._latest_time: float = 0.0
        # Tracks the last time a trigger_kick=True packet was actually RECEIVED (in _recv_loop,
        # which sees every packet as it arrives), separate from get_latest()'s instantaneous
        # snapshot. trigger_kick is only True for a single ~1-tick pulse (one UDP packet) when the
        # HL controller fires it -- any consumer that only samples the instantaneous value on its
        # own slower timer (e.g. a GUI polling at 10Hz against a 50Hz command stream) has a high
        # chance of landing between packets and missing the pulse entirely. was_recently_triggered
        # exists for exactly that kind of consumer; the real RL control path
        # (NetworkControlledUnifiedPolicy._apply_network_command) still edge-detects off
        # get_latest() directly and is unaffected by this.
        self._last_trigger_seen_time: float = float("-inf")

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("127.0.0.1", port))
        self._sock.settimeout(0.1)

        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def _recv_loop(self) -> None:
        while True:
            try:
                data, _ = self._sock.recvfrom(_STRUCT_SIZE)
            except socket.timeout:
                continue
            except OSError:
                return
            if len(data) != _STRUCT_SIZE:
                continue
            vx, vy, wyaw, trigger_kick = struct.unpack(_STRUCT_FMT, data)
            now = time.monotonic()
            with self._lock:
                self._latest = HlCommand(vx, vy, wyaw, trigger_kick)
                self._latest_time = now
                if trigger_kick:
                    self._last_trigger_seen_time = now

    def get_latest(self) -> Optional[HlCommand]:
        with self._lock:
            if self._latest is None:
                return None
            if time.monotonic() - self._latest_time > self._staleness_timeout_s:
                return None
            return self._latest

    def was_recently_triggered(self, window_s: float = 1.5) -> bool:
        """True if a trigger_kick=True packet was received within the last `window_s` seconds --
        for slow-polling display/observability consumers (see class docstring), not for control
        logic (which should edge-detect off get_latest() directly, as the real RL loop does)."""
        with self._lock:
            return (time.monotonic() - self._last_trigger_seen_time) < window_s
