"""Thin, stdlib-only wrapper around `mujoco_ref_clip_recovery_scan.py`, importable from the eval
process (which does not have RoboJuDo installed -- that's why the actual work happens in a
subprocess under the separate `robojudo` conda env). Same lock/subprocess architecture as
`record_mujoco_kick_to_loco_flip_scan.py` -- see that module's own docstring for the full
rationale (busy lock just means "skip this checkpoint's scan", never blocks/waits).

Never raises -- every failure mode is caught, logged via loguru, and turned into a `None` return.
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

ROBOJUDO_PYTHON = os.environ.get(
    "HOLOSOMA_ROBOJUDO_PYTHON",
    "/workspaces/isaaclab_arena/submodules/workspaces/conda_env/robojudo/bin/python",
)
WORKER_SCRIPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mujoco_ref_clip_recovery_scan.py")

# Own lock, separate from every other sim2sim mechanism's lock file -- same rationale as
# record_mujoco_kick_to_loco_flip_scan.py's own DEFAULT_LOCK_PATH comment.
DEFAULT_LOCK_PATH = os.environ.get(
    "HOLOSOMA_SIM2SIM_REF_CLIP_RECOVERY_LOCK_PATH", "/tmp/holosoma_sim2sim_ref_clip_recovery.lock"
)


def record_ref_clip_recovery_scan(
    onnx_path: str,
    step_label: str,
    *,
    motion_npz: str,
    frame_max: int,
    num_trials: int,
    skill_id: int = 0,
    seed: int = 0,
    kick_aim_enabled: bool = False,
    timeout_s: float = 300.0,
    frame_min: int = 0,
    hold_s: float = 5.0,
    recentre_xy: bool = True,
    recovery_metrics_out: dict | None = None,
    lock_path: str = DEFAULT_LOCK_PATH,
    stale_lock_timeout_s: float = DEFAULT_STALE_LOCK_TIMEOUT_S,
) -> float | None:
    """Run an N-trial MuJoCo sim2sim scan of `onnx_path` (RoboJuDo pipeline) that teleports the
    robot into a pose sampled from `motion_npz`'s authored clip content (`frame_min`..`frame_max`,
    excluding the synthetic recovery/hold tail -- see mujoco_ref_clip_recovery_scan.py's own FRAME
    WINDOW note for what to pass as `frame_max`), starts it in ordinary locomotion mode, and
    returns the fraction of trials that never dip below FALL_Z during the post-injection hold
    window, or None on failure (busy lock, timeout, nonzero exit, or an unparseable/missing
    SUMMARY line).

    `kick_aim_enabled`: pass the checkpoint's OWN kick_aim_enabled for this skill, same rationale
    as every sibling record_* wrapper's own flag.

    `recovery_metrics_out` (default None -- opt-in, same output-param pattern
    record_kick_to_loco_flip_scan's own `recovery_metrics_out` establishes, for a consistent
    calling convention across both wrappers): when a dict is passed, it is populated with
    `"recovered_rate"` and `"mean_recovery_steps"` from the worker's SUMMARY_RECOVERY_STEPS line --
    see mujoco_ref_clip_recovery_scan.py's own RECOVERY_MIN_HEIGHT/RECOVERY_CONSECUTIVE_STEPS for
    what "recovered" means.

    Serialized cluster-wide via a lock file at `lock_path` -- if already held, returns None
    immediately without launching anything (does not block/wait).

    Never raises."""
    token = acquire_global_lock(lock_path, stale_lock_timeout_s)
    if token is None:
        logger.warning(f"[sim2sim] Ref-clip-recovery scan lock busy -- skipping scan for {onnx_path}.")
        return None

    try:
        argv = [
            ROBOJUDO_PYTHON, WORKER_SCRIPT_PATH,
            "--onnx-path", onnx_path,
            "--motion-npz", motion_npz,
            "--step-label", str(step_label),
            "--skill-id", str(skill_id),
            "--num-trials", str(num_trials),
            "--seed", str(seed),
            "--frame-min", str(frame_min),
            "--frame-max", str(frame_max),
            "--hold-s", str(hold_s),
        ]
        argv.append("--recentre-xy" if recentre_xy else "--no-recentre-xy")
        if kick_aim_enabled:
            argv.append("--kick-aim-enabled")
        try:
            # ROBOJUDO_ORT_INTRA_OP_NUM_THREADS=1: caps this worker's own ONNX Runtime session to
            # a single thread -- see unified_loco_kick_policy.py's own docstring on that env var,
            # and record_mujoco_loco_to_kick_handoff_scan.py's identical guard for the full
            # oversubscription story (observed: ~67 threads/session on a 128-core box).
            worker_env = {**os.environ, "ROBOJUDO_ORT_INTRA_OP_NUM_THREADS": "1"}
            result = subprocess.run(argv, timeout=timeout_s, capture_output=True, text=True, env=worker_env)
        except subprocess.TimeoutExpired:
            logger.warning(f"[sim2sim] MuJoCo ref-clip-recovery scan timed out after {timeout_s:.0f}s for {onnx_path}")
            return None

        if result.returncode != 0:
            logger.warning(
                f"[sim2sim] MuJoCo ref-clip-recovery scan worker exited {result.returncode} for {onnx_path}\n"
                f"stdout(tail): {result.stdout[-2000:]}\nstderr(tail): {result.stderr[-2000:]}"
            )
            return None

        alive_rate: float | None = None
        recovered_rate: float | None = None
        mean_recovery_steps: float | None = None
        recovery_line_seen = False
        # Forward print order is SUMMARY_REFCLIP then SUMMARY_RECOVERY_STEPS (worker's own
        # docstring) -- a REVERSED pass meets RECOVERY_STEPS first, one pass finds both.
        for line in reversed(result.stdout.splitlines()):
            if not recovery_line_seen and line.startswith("SUMMARY_RECOVERY_STEPS "):
                recovery_line_seen = True
                parts = line.split()
                if len(parts) >= 5:
                    if parts[3] != "NA":
                        try:
                            recovered_rate = float(parts[3])
                        except ValueError:
                            logger.warning(f"[sim2sim] Unparseable SUMMARY_RECOVERY_STEPS rate from ref-clip-recovery scan: {line!r}")
                    if parts[4] != "NA":
                        try:
                            mean_recovery_steps = float(parts[4])
                        except ValueError:
                            logger.warning(f"[sim2sim] Unparseable SUMMARY_RECOVERY_STEPS mean from ref-clip-recovery scan: {line!r}")
                continue
            if alive_rate is None and line.startswith("SUMMARY_REFCLIP "):
                parts = line.split()
                try:
                    alive_rate = float(parts[3])
                except (IndexError, ValueError):
                    logger.warning(f"[sim2sim] Unparseable SUMMARY_REFCLIP line from ref-clip-recovery scan: {line!r}")
            if alive_rate is not None and recovery_line_seen:
                break

        if alive_rate is None:
            logger.warning(
                f"[sim2sim] MuJoCo ref-clip-recovery scan worker exited 0 but printed no usable "
                f"SUMMARY_REFCLIP line for {onnx_path}\nstdout(tail): {result.stdout[-2000:]}"
            )
        if recovery_metrics_out is not None:
            recovery_metrics_out["recovered_rate"] = recovered_rate
            recovery_metrics_out["mean_recovery_steps"] = mean_recovery_steps
        return alive_rate

    except Exception:
        logger.exception(f"[sim2sim] Unhandled error running MuJoCo ref-clip-recovery scan for {onnx_path}")
        return None
    finally:
        release_global_lock(lock_path, token)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Manual test of record_ref_clip_recovery_scan().")
    parser.add_argument("--onnx-path", required=True)
    parser.add_argument("--motion-npz", required=True)
    parser.add_argument("--step-label", default="manual")
    parser.add_argument("--num-trials", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skill-id", type=int, default=0)
    parser.add_argument("--frame-min", type=int, default=0)
    parser.add_argument("--frame-max", type=int, required=True)
    parser.add_argument("--hold-s", type=float, default=5.0)
    parser.add_argument("--kick-aim-enabled", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=300.0)
    ns = parser.parse_args()

    alive_rate = record_ref_clip_recovery_scan(
        onnx_path=ns.onnx_path,
        step_label=ns.step_label,
        motion_npz=ns.motion_npz,
        frame_max=ns.frame_max,
        num_trials=ns.num_trials,
        skill_id=ns.skill_id,
        seed=ns.seed,
        frame_min=ns.frame_min,
        hold_s=ns.hold_s,
        kick_aim_enabled=ns.kick_aim_enabled,
        timeout_s=ns.timeout_s,
        recovery_metrics_out=(recovery_metrics := {}),
    )
    ok = alive_rate is not None
    print(
        "record_ref_clip_recovery_scan: "
        + (
            f"SUCCESS alive_rate={alive_rate} "
            f"recovered_rate={recovery_metrics.get('recovered_rate')} "
            f"mean_recovery_steps={recovery_metrics.get('mean_recovery_steps')}"
            if ok else "FAILED"
        )
    )
    raise SystemExit(0 if ok else 1)
