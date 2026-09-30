"""Thin, stdlib-only wrapper around `mujoco_locomotion_metrics_worker.py`, importable from the
`hssim` training/eval process (which does not have RoboJuDo installed). Mirrors
`record_mujoco_survival_scan.py`'s subprocess-to-the-separate-`robojudo`-conda-env boundary, but
returns the worker's parsed metrics dict rather than a bool/tuple -- there is no legacy
RESULT-line text protocol to match here (this worker is net-new, see its own module docstring),
so it writes one JSON summary to `--output-json` and this wrapper just reads that file back.

Never raises. Returns `None` on any failure (busy lock, timeout, crash, missing output file) --
same never-block/never-raise contract as every other record_*_scan module in this project, so a
single bad checkpoint or a contended host degrades one sim2sim_eval.py data point, not the run.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import uuid

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
WORKER_SCRIPT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "mujoco_locomotion_metrics_worker.py"
)

DEFAULT_LOCK_PATH = os.environ.get(
    "HOLOSOMA_SIM2SIM_LOCOMOTION_METRICS_LOCK_PATH", "/tmp/holosoma_sim2sim_locomotion_metrics.lock"
)


def record_locomotion_metrics_scan(
    onnx_path: str,
    *,
    num_episodes: int = 5,
    control_hz: float = 50.0,
    settle_s: float = 1.0,
    command_s: float = 3.0,
    warmup_s: float = 1.0,
    fall_height_m: float = 0.70,
    forward_speed: float = 0.8,
    backward_speed: float = -0.8,
    lateral_speed: float = 0.5,
    yaw_rate: float = 0.5,
    push_force_n: float = 80.0,
    push_duration_s: float = 0.20,
    push_recovery_window_s: float = 2.0,
    seed: int = 0,
    timeout_s: float = 300.0,
    lock_path: str = DEFAULT_LOCK_PATH,
    stale_lock_timeout_s: float = DEFAULT_STALE_LOCK_TIMEOUT_S,
) -> dict | None:
    """Runs `mujoco_locomotion_metrics_worker.py` against `onnx_path` (RoboJuDo pipeline, flat
    scene) and returns its parsed JSON result -- see that worker's own module docstring for the
    full protocol (command magnitudes, push magnitude, fall/recovery threshold, all grounded in
    this project's own trained command ranges and push-randomization config, not arbitrary).

    Serialized cluster-wide via a lock file at `lock_path` -- if already held, returns None
    immediately without launching anything (does not block/wait). Callers that want independent
    concurrent runs (e.g. sim2sim_eval.py's own thread pool) should pass a fresh, never-reused
    `lock_path` per call -- see that script's own `_fresh_lock_path`.

    Returns None on lock contention, timeout, non-zero exit, or a missing/unparseable output file.
    Never raises.
    """
    token = acquire_global_lock(lock_path, stale_lock_timeout_s)
    if token is None:
        logger.warning(f"[sim2sim] Global locomotion-metrics lock busy -- skipping scan for {onnx_path}.")
        return None

    output_json_path = os.path.join(tempfile.gettempdir(), f"holosoma_loco_metrics_{uuid.uuid4().hex}.json")
    try:
        argv = [
            ROBOJUDO_PYTHON, WORKER_SCRIPT_PATH,
            "--onnx-path", onnx_path,
            "--output-json", output_json_path,
            "--num-episodes", str(num_episodes),
            "--control-hz", str(control_hz),
            "--settle-s", str(settle_s),
            "--command-s", str(command_s),
            "--warmup-s", str(warmup_s),
            "--fall-height-m", str(fall_height_m),
            "--forward-speed", str(forward_speed),
            "--backward-speed", str(backward_speed),
            "--lateral-speed", str(lateral_speed),
            "--yaw-rate", str(yaw_rate),
            "--push-force-n", str(push_force_n),
            "--push-duration-s", str(push_duration_s),
            "--push-recovery-window-s", str(push_recovery_window_s),
            "--seed", str(seed),
        ]
        try:
            result = subprocess.run(argv, timeout=timeout_s, capture_output=True, text=True)
        except subprocess.TimeoutExpired:
            logger.warning(f"[sim2sim] MuJoCo locomotion-metrics scan timed out after {timeout_s:.0f}s for {onnx_path}")
            return None

        if result.returncode != 0 or not os.path.exists(output_json_path):
            logger.warning(
                f"[sim2sim] MuJoCo locomotion-metrics scan failed (exit {result.returncode}) for {onnx_path}\n"
                f"stdout(tail): {result.stdout[-2000:]}\nstderr(tail): {result.stderr[-2000:]}"
            )
            return None

        with open(output_json_path) as f:
            return json.load(f)

    except Exception:
        logger.exception(f"[sim2sim] Unhandled error recording MuJoCo locomotion-metrics scan for {onnx_path}")
        return None
    finally:
        release_global_lock(lock_path, token)
        try:
            os.remove(output_json_path)
        except OSError:
            pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Manual test of record_locomotion_metrics_scan().")
    parser.add_argument("--onnx-path", required=True)
    parser.add_argument("--num-episodes", type=int, default=3)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    ns = parser.parse_args()

    out = record_locomotion_metrics_scan(onnx_path=ns.onnx_path, num_episodes=ns.num_episodes, timeout_s=ns.timeout_s)
    print(json.dumps(out, indent=2) if out is not None else "FAILED")
    raise SystemExit(0 if out is not None else 1)
