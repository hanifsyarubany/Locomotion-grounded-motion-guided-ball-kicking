"""Pluggable source of "ball position in the robot's body frame" for ApproachAndKickPolicy.

Deliberately body-frame, not world-frame: real deployment's `get_low_state()` has NO absolute
world position (no GPS on a real humanoid -- see BasicSdk2Bridge/UnitreeSdk2Bridge, `low_state`
carries only relative IMU/joint state), so there is no world-frame robot pose to convert against
even if a source produced world-frame ball coordinates. This also matches this project's own
original design decision (see high_level_ball_approach_and_kicking/README.md): the HL
approach+kick logic was always specified to consume body-frame ball position only, never world
coordinates -- which turns out to be exactly what a real camera-relative detection pipeline would
naturally produce anyway, not an extra abstraction layered on top of it.

Implementations here are intentionally swappable: a future real vision pipeline (camera + object
detection + a known camera-to-body extrinsic) implements the same `get_ball_pos_body_frame()`
interface and drops in without touching ApproachAndKickPolicy at all.
"""

from __future__ import annotations

import socket
import struct
import threading
import time
from typing import Optional, Protocol

import numpy as np


class BallPoseSource(Protocol):
    def get_ball_pos_body_frame(self) -> Optional[np.ndarray]:
        """Returns (dx, dy) in the robot's current body frame, or None if no recent reading is
        available (e.g. ball out of view, sensor dropout) -- callers must treat None as "don't
        act on stale/missing data", not "ball at origin"."""
        ...


class FixedBallPoseSource:
    """Constant BODY-FRAME offset -- for smoke-testing ApproachAndKickPolicy's control loop and
    integration with the real UnifiedPolicy/BasePolicy machinery without any real or simulated
    perception. NOT a stand-in for real ball tracking, and NOT a fixed point in the world: since
    the offset is constant in the robot's own frame, it moves WITH the robot as it walks and
    turns, like a target painted on the robot's visor -- the error therefore never decreases and
    the robot walks/turns at a constant commanded rate forever (confirmed directly: this is only
    useful for testing the trivial always-triggered case where dx/dy already match the
    controller's own target, e.g. the default (1.6, 0.0), giving error_dist=0 from tick one).
    Testing real closed-loop convergence to a fixed point requires either UdpBallPoseSource fed
    by a world-frame-aware publisher (see run_sim.py's --broadcast-ball-udp-port, which computes
    a genuinely world-fixed synthetic target from the robot's own ground-truth pose), or real
    perception."""

    def __init__(self, dx: float, dy: float):
        self._pos = np.array([dx, dy])

    def get_ball_pos_body_frame(self) -> Optional[np.ndarray]:
        return self._pos.copy()


class UdpBallPoseSource:
    """Receives (dx, dy) body-frame offsets broadcast over UDP from a sim-side (or future
    perception-side) publisher -- see README for the wire format (two float64s, little-endian,
    matching struct.pack('<dd', dx, dy)). Non-blocking: get_ball_pos_body_frame() always returns
    immediately with the latest received value, or None if nothing has arrived within
    `staleness_timeout_s` (guards against acting on a reading from before a sensor dropout).

    NOTE: as of this writing, no sim-side publisher exists yet that computes this from a REAL
    ball physically present in the run_sim.py MuJoCo bridge scene -- see README's "sim2real"
    section for why that's a real, not-yet-done follow-up (the ball needs to exist in the SAME
    physics simulation the robot's real DDS bridge drives, which requires modifying run_sim.py's
    scene construction, not a separate disconnected MuJoCo instance). This class is ready for that
    publisher once it exists, and is directly usable today with any other UDP publisher speaking
    the same wire format (e.g. a manual test script, or eventually real perception).
    """

    def __init__(self, port: int, staleness_timeout_s: float = 0.5):
        self._staleness_timeout_s = staleness_timeout_s
        self._lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None
        self._latest_time: float = 0.0

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("127.0.0.1", port))
        self._sock.settimeout(0.1)

        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def _recv_loop(self) -> None:
        while True:
            try:
                data, _ = self._sock.recvfrom(16)
            except socket.timeout:
                continue
            except OSError:
                return
            if len(data) != 16:
                continue
            dx, dy = struct.unpack("<dd", data)
            with self._lock:
                self._latest = np.array([dx, dy])
                self._latest_time = time.monotonic()

    def get_ball_pos_body_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest is None:
                return None
            if time.monotonic() - self._latest_time > self._staleness_timeout_s:
                return None
            return self._latest.copy()


def send_ball_pose_udp(sock: socket.socket, port: int, dx: float, dy: float) -> None:
    """Sender-side helper matching UdpBallPoseSource's wire format -- for a future sim-side or
    perception-side publisher."""
    sock.sendto(struct.pack("<dd", dx, dy), ("127.0.0.1", port))
