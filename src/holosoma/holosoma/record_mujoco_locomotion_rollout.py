"""Thin, stdlib-only wrapper around `mujoco_locomotion_rollout_worker.py`, importable from the
training process (which does not have RoboJuDo installed -- see that file's module docstring).
Mirrors `record_mujoco_kick_rollout.py` exactly, including the never-block/never-raise contract.

Uses its OWN lock file (`DEFAULT_LOCK_PATH` below), separate from the kick rollout's -- both fire
from the same checkpoint-save event at the same cadence (see fast_sac_agent.py), and sharing one
lock would mean whichever rollout starts its subprocess first wins the lock and the other always
gets skipped, defeating the point of running both. Two independent lock files mean the two
rollout TYPES run concurrently; each still serializes against copies of ITSELF across any other
concurrently-training runs on this machine, same rationale as the kick lock.
"""

from __future__ import annotations

import os
import subprocess

from loguru import logger

from holosoma.utils.rollout_lock import (
    DEFAULT_STALE_LOCK_TIMEOUT_S,
    acquire_global_lock,
    release_global_lock,
)

# "mujoco_media/" prefix -- see record_mujoco_kick_rollout.py's MUJOCO_KICK_WANDB_KEY for the full
# rationale (wandb UI panel-section grouping, mirrors "isaacsim_media/" on the IsaacSim side).
MUJOCO_WALK_WANDB_KEY = "mujoco_media/Training rollout - MuJoCo Walk"

ROBOJUDO_PYTHON = os.environ.get(
    "HOLOSOMA_ROBOJUDO_PYTHON",
    "/workspaces/isaaclab_arena/submodules/workspaces/conda_env/robojudo/bin/python",
)
WORKER_SCRIPT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "mujoco_locomotion_rollout_worker.py"
)

DEFAULT_LOCK_PATH = os.environ.get("HOLOSOMA_SIM2SIM_WALK_LOCK_PATH", "/tmp/holosoma_sim2sim_walk_rollout.lock")


def record_locomotion_rollout(
    onnx_path: str,
    output_video_path: str,
    *,
    # 2026-08-16: raised from 60.0 -- with several training processes sharing this host (plus
    # whatever else launches on it, e.g. an unrelated concurrent IsaacSim job), the worker
    # subprocess can genuinely take longer than 60s under contention without being stuck/hung (it
    # keeps burning real CPU, see rollout_lock.py's non-blocking lock -- this isn't lock
    # contention). A hard 60s budget was killing otherwise-fine rollouts mid-recording, producing
    # tiny partial clips instead of no clip. This doesn't fix the underlying contention, just gives
    # a slow-but-progressing rollout enough rope to actually finish -- rollouts fire every N
    # checkpoint saves (minutes apart), so 180s of slack here is negligible against that cadence.
    timeout_s: float = 180.0,
    walk_s: float = 5.0,
    stand_s: float = 3.0,
    forward_speed: float = 0.8,
    lock_path: str = DEFAULT_LOCK_PATH,
    stale_lock_timeout_s: float = DEFAULT_STALE_LOCK_TIMEOUT_S,
) -> bool:
    """Record a single-env MuJoCo sim2sim forward-walk -> stand rollout of `onnx_path` (RoboJuDo
    pipeline), with a real physical ball in the scene (never observed -- task_mode stays
    "locomotion" throughout), and save it to `output_video_path`.

    Serialized cluster-wide via a lock file at `lock_path` -- if already held, returns False
    immediately without launching anything (does not block/wait).

    Never raises. Returns True iff `output_video_path` exists on return.
    """
    token = acquire_global_lock(lock_path, stale_lock_timeout_s)
    if token is None:
        logger.warning(f"[sim2sim] Global walk-rollout lock busy -- skipping rollout for {onnx_path}.")
        return False

    try:
        os.makedirs(os.path.dirname(output_video_path) or ".", exist_ok=True)
        argv = [
            ROBOJUDO_PYTHON, WORKER_SCRIPT_PATH,
            "--onnx-path", onnx_path,
            "--output-video-path", output_video_path,
            "--walk-s", str(walk_s),
            "--stand-s", str(stand_s),
            "--forward-speed", str(forward_speed),
        ]
        try:
            result = subprocess.run(argv, timeout=timeout_s, capture_output=True, text=True)
        except subprocess.TimeoutExpired:
            logger.warning(f"[sim2sim] MuJoCo walk rollout timed out after {timeout_s:.0f}s for {onnx_path}")
            return False

        if result.returncode != 0:
            logger.warning(
                f"[sim2sim] MuJoCo walk rollout worker exited {result.returncode} for {onnx_path}\n"
                f"stdout(tail): {result.stdout[-2000:]}\nstderr(tail): {result.stderr[-2000:]}"
            )
            return False

        if not os.path.exists(output_video_path):
            logger.warning(f"[sim2sim] MuJoCo walk rollout worker exited 0 but no video at {output_video_path}")
            return False

        return True

    except Exception:
        logger.exception(f"[sim2sim] Unhandled error recording MuJoCo walk rollout for {onnx_path}")
        return False
    finally:
        release_global_lock(lock_path, token)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Manual test of record_locomotion_rollout().")
    parser.add_argument("--onnx-path", required=True)
    parser.add_argument("--output-video-path", default="/tmp/holosoma_mujoco_walk_rollout_test.mp4")
    parser.add_argument("--timeout-s", type=float, default=180.0)
    ns = parser.parse_args()

    ok = record_locomotion_rollout(
        onnx_path=ns.onnx_path,
        output_video_path=ns.output_video_path,
        timeout_s=ns.timeout_s,
    )
    print(f"record_locomotion_rollout: {'SUCCESS' if ok else 'FAILED'} -> {ns.output_video_path}")
    raise SystemExit(0 if ok else 1)
