"""Generic cross-process file lock for serializing CPU-bound MuJoCo sim2sim rollout subprocesses.

Extracted from `record_mujoco_kick_rollout.py` (2026-07-21) so a second, independent rollout type
(`record_mujoco_locomotion_rollout.py`) can reuse the exact same non-blocking, stale-steal-capable
locking behavior against its own lock file, without duplicating the atomicity/staleness subtlety.

A busy lock just means "skip this rollout" -- callers never block or wait.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import time

from loguru import logger

DEFAULT_STALE_LOCK_TIMEOUT_S = 300.0


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, just owned by someone else
    return True


def read_lock(path: str) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, ValueError, OSError):
        return None


def acquire_global_lock(path: str, stale_timeout_s: float) -> str | None:
    """Single, non-blocking attempt. Returns a unique token to pass to `release_global_lock` on
    success, or None if another live, non-stale holder already has it.

    Fast path: `os.O_CREAT|O_EXCL` is atomic on a local filesystem.

    Stale-steal path: if the existing lock is older than `stale_timeout_s`, or its PID is confirmed
    dead (only trusted for locks written on this host), overwrite it. Two processes racing to steal
    at once is possible but harmless: after writing, we re-read and only proceed if our own token
    won -- losing that race just means we also skip this rollout, never a corrupted lock file.
    """
    token = f"{os.getpid()}-{time.monotonic_ns()}"
    payload = json.dumps(
        {"pid": os.getpid(), "token": token, "acquired_at": time.time(), "hostname": socket.gethostname()}
    ).encode()

    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        try:
            os.write(fd, payload)
        finally:
            os.close(fd)
        return token
    except FileExistsError:
        pass

    existing = read_lock(path)
    if existing is None:
        stale = True  # corrupt/unreadable -- safer to proceed than wedge forever
    else:
        age_s = time.time() - existing.get("acquired_at", 0.0)
        same_host = existing.get("hostname") == socket.gethostname()
        pid_dead = same_host and not pid_alive(existing.get("pid", -1))
        stale = age_s > stale_timeout_s or pid_dead

    if not stale:
        return None

    logger.warning(f"[sim2sim] Stealing stale rollout lock at {path}: {existing}")
    with open(path, "wb") as f:
        f.write(payload)
    time.sleep(0.05)
    winner = read_lock(path)
    return token if (winner is not None and winner.get("token") == token) else None


def release_global_lock(path: str, token: str) -> None:
    """Only deletes the lock if it still contains OUR token -- if not, our lock was stolen (we
    overran stale_timeout_s ourselves), and deleting whatever's there now would release a lock we
    no longer own."""
    existing = read_lock(path)
    if existing is not None and existing.get("token") == token:
        with contextlib.suppress(OSError):
            os.remove(path)
