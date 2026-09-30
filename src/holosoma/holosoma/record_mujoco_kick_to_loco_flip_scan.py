"""Thin, stdlib-only wrapper around `mujoco_kick_loco_flip_scan.py`, importable from the training
process (which does not have RoboJuDo installed -- that's why the actual work happens in a
subprocess under the separate `robojudo` conda env). Same lock/subprocess architecture as
`record_mujoco_survival_scan.py` -- see that module's own docstring for the full rationale (busy
lock just means "skip this checkpoint's scan", never blocks/waits).

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

# METRIC-NAME suffix only -- the caller (FastSACAgent._mujoco_kick_to_loco_flip_scan_worker)
# prefixes this with "Kick_skills_{skill_idx}/", same established convention as
# MUJOCO_SURVIVAL_SCAN_WANDB_KEY (record_mujoco_survival_scan.py).
MUJOCO_KICK_TO_LOCO_FLIP_ALIVE_WANDB_KEY = "sim2sim/kick_to_loco_random_flip_alive_rate"

# Diagnostic companion metric, same N trials, no extra rollout cost: the fraction of trials that
# fell BEFORE ever reaching their scheduled flip tick (an ordinary in-kick fall, already covered by
# kick_fall_rate) -- excluded from the alive-rate denominator above, reported separately so a
# checkpoint that cannot even survive an ordinary kick doesn't silently produce a misleading (or
# NA) flip-alive rate. See mujoco_kick_loco_flip_scan.py's own module docstring.
MUJOCO_KICK_TO_LOCO_FLIP_PRE_FLIP_FAIL_WANDB_KEY = "sim2sim/kick_to_loco_random_flip_pre_flip_fail_rate"

ROBOJUDO_PYTHON = os.environ.get(
    "HOLOSOMA_ROBOJUDO_PYTHON",
    "/workspaces/isaaclab_arena/submodules/workspaces/conda_env/robojudo/bin/python",
)
WORKER_SCRIPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mujoco_kick_loco_flip_scan.py")

# Own lock, separate from every other sim2sim mechanism's lock file -- same rationale as
# record_mujoco_survival_scan.py's own DEFAULT_LOCK_PATH comment: this fires from the same
# checkpoint-save event, on its own independent cadence knob, so sharing a lock would make
# whichever starts first always win.
DEFAULT_LOCK_PATH = os.environ.get(
    "HOLOSOMA_SIM2SIM_KICK_TO_LOCO_FLIP_LOCK_PATH", "/tmp/holosoma_sim2sim_kick_to_loco_flip.lock"
)


def record_kick_to_loco_flip_scan(
    onnx_path: str,
    step_label: str,
    *,
    num_trials: int,
    skill_id: int = 0,
    seed: int = 0,
    kick_aim_enabled: bool = False,
    timeout_s: float = 300.0,
    settle_s: float = 1.5,
    flip_delay_min_steps: int = 10,
    flip_delay_max_steps: int = 60,
    post_flip_hold_s: float = 5.0,
    recovery_metrics_out: dict | None = None,
    lock_path: str = DEFAULT_LOCK_PATH,
    stale_lock_timeout_s: float = DEFAULT_STALE_LOCK_TIMEOUT_S,
) -> tuple[float | None, float | None]:
    """Run an N-trial MuJoCo sim2sim scan of `onnx_path` (RoboJuDo pipeline) that forces a
    kick->locomotion flip at a randomized mid-clip tick, and return `(alive_rate,
    pre_flip_fail_rate)`, each in [0, 1] or None on failure (busy lock, timeout, nonzero exit, or
    that particular SUMMARY line missing/unparseable). `alive_rate` is additionally None when every
    trial fell before ever reaching its scheduled flip tick (nothing to measure the flip against --
    see mujoco_kick_loco_flip_scan.py's own docstring).

    `kick_aim_enabled`: pass the checkpoint's OWN kick_aim_enabled for this skill (read by the
    caller from the live training config) -- REQUIRED True for a checkpoint actually trained with
    it, same rationale as record_survival_scan's own flag.

    `flip_delay_min_steps`/`flip_delay_max_steps` (default 10/60): mirrors
    MultiSkillConfig.kick_abort_delay_min_steps/max_steps's own defaults for direct comparability
    with the training-time mechanism this scan evaluates -- override only to test a different
    window than the checkpoint was (or would be) trained against.

    `recovery_metrics_out` (default None -- opt-in, same output-param pattern
    record_survival_scan's own `extra_metrics_out` already establishes): when a dict is passed, it
    is populated with `"recovered_rate"` and `"mean_recovery_steps"` (the latter None when
    `recovered_rate` is 0 -- nothing to average) from the worker's SUMMARY_RECOVERY_STEPS line --
    see mujoco_kick_loco_flip_scan.py's own RECOVERY_MIN_HEIGHT/RECOVERY_CONSECUTIVE_STEPS for what
    "recovered" means. The RETURN SIGNATURE stays `(alive_rate, pre_flip_fail_rate)` unchanged --
    this function is shared with FastSACAgent's own training-time periodic sim2sim checks (see
    `_pool_flip_rates`'s own docstring in sim2sim_eval.py), so that isn't something to change for
    just this; every existing caller that omits this parameter is completely unaffected.

    Serialized cluster-wide via a lock file at `lock_path` -- if already held, returns (None, None)
    immediately without launching anything (does not block/wait).

    Never raises."""
    token = acquire_global_lock(lock_path, stale_lock_timeout_s)
    if token is None:
        logger.warning(f"[sim2sim] Kick-to-loco-flip scan lock busy -- skipping scan for {onnx_path}.")
        return None, None

    try:
        argv = [
            ROBOJUDO_PYTHON, WORKER_SCRIPT_PATH,
            "--onnx-path", onnx_path,
            "--step-label", str(step_label),
            "--settle-s", str(settle_s),
            "--skill-id", str(skill_id),
            "--num-trials", str(num_trials),
            "--seed", str(seed),
            "--flip-delay-min-steps", str(flip_delay_min_steps),
            "--flip-delay-max-steps", str(flip_delay_max_steps),
            "--post-flip-hold-s", str(post_flip_hold_s),
        ]
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
            logger.warning(f"[sim2sim] MuJoCo kick-to-loco-flip scan timed out after {timeout_s:.0f}s for {onnx_path}")
            return None, None

        if result.returncode != 0:
            logger.warning(
                f"[sim2sim] MuJoCo kick-to-loco-flip scan worker exited {result.returncode} for {onnx_path}\n"
                f"stdout(tail): {result.stdout[-2000:]}\nstderr(tail): {result.stderr[-2000:]}"
            )
            return None, None

        alive_rate: float | None = None
        pre_flip_fail_rate: float | None = None
        recovered_rate: float | None = None
        mean_recovery_steps: float | None = None
        recovery_line_seen = False
        # Forward print order is PREFLIPFAIL, LOCOFLIP, RECOVERY_STEPS (worker's own docstring) --
        # a REVERSED pass over stdout therefore meets RECOVERY_STEPS first, one pass finds all
        # three regardless of caller-requested subset.
        for line in reversed(result.stdout.splitlines()):
            if not recovery_line_seen and line.startswith("SUMMARY_RECOVERY_STEPS "):
                recovery_line_seen = True
                parts = line.split()
                if len(parts) >= 5:
                    if parts[3] != "NA":
                        try:
                            recovered_rate = float(parts[3])
                        except ValueError:
                            logger.warning(f"[sim2sim] Unparseable SUMMARY_RECOVERY_STEPS rate from kick-to-loco-flip scan: {line!r}")
                    # "NA" (num_recovered == 0, an expected state -- nothing recovered to average)
                    # stays None with no warning, same convention as SUMMARY_LOCOFLIP's own "NA".
                    if parts[4] != "NA":
                        try:
                            mean_recovery_steps = float(parts[4])
                        except ValueError:
                            logger.warning(f"[sim2sim] Unparseable SUMMARY_RECOVERY_STEPS mean from kick-to-loco-flip scan: {line!r}")
            if pre_flip_fail_rate is None and line.startswith("SUMMARY_PREFLIPFAIL "):
                parts = line.split()
                try:
                    pre_flip_fail_rate = float(parts[3])
                except (IndexError, ValueError):
                    logger.warning(f"[sim2sim] Unparseable SUMMARY_PREFLIPFAIL line from kick-to-loco-flip scan: {line!r}")
            if alive_rate is None and line.startswith("SUMMARY_LOCOFLIP "):
                parts = line.split()
                # "NA" (num_reached_flip == 0) is an EXPECTED, non-error state, not a parse
                # failure -- stays None with no warning.
                if len(parts) >= 4 and parts[3] != "NA":
                    try:
                        alive_rate = float(parts[3])
                    except ValueError:
                        logger.warning(f"[sim2sim] Unparseable SUMMARY_LOCOFLIP line from kick-to-loco-flip scan: {line!r}")
            if alive_rate is not None and pre_flip_fail_rate is not None and recovery_line_seen:
                break

        if alive_rate is None and pre_flip_fail_rate is None:
            logger.warning(
                f"[sim2sim] MuJoCo kick-to-loco-flip scan worker exited 0 but printed no SUMMARY line for {onnx_path}\n"
                f"stdout(tail): {result.stdout[-2000:]}"
            )
        if recovery_metrics_out is not None:
            recovery_metrics_out["recovered_rate"] = recovered_rate
            recovery_metrics_out["mean_recovery_steps"] = mean_recovery_steps
        return alive_rate, pre_flip_fail_rate

    except Exception:
        logger.exception(f"[sim2sim] Unhandled error running MuJoCo kick-to-loco-flip scan for {onnx_path}")
        return None, None
    finally:
        release_global_lock(lock_path, token)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Manual test of record_kick_to_loco_flip_scan().")
    parser.add_argument("--onnx-path", required=True)
    parser.add_argument("--step-label", default="manual")
    parser.add_argument("--num-trials", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skill-id", type=int, default=0)
    parser.add_argument("--kick-aim-enabled", action="store_true")
    parser.add_argument("--flip-delay-min-steps", type=int, default=10)
    parser.add_argument("--flip-delay-max-steps", type=int, default=60)
    parser.add_argument("--post-flip-hold-s", type=float, default=5.0)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    ns = parser.parse_args()

    alive_rate, pre_flip_fail_rate = record_kick_to_loco_flip_scan(
        onnx_path=ns.onnx_path,
        step_label=ns.step_label,
        num_trials=ns.num_trials,
        skill_id=ns.skill_id,
        seed=ns.seed,
        kick_aim_enabled=ns.kick_aim_enabled,
        flip_delay_min_steps=ns.flip_delay_min_steps,
        flip_delay_max_steps=ns.flip_delay_max_steps,
        post_flip_hold_s=ns.post_flip_hold_s,
        timeout_s=ns.timeout_s,
        recovery_metrics_out=(recovery_metrics := {}),
    )
    ok = alive_rate is not None or pre_flip_fail_rate is not None
    print(
        "record_kick_to_loco_flip_scan: "
        + (
            f"SUCCESS alive_rate={alive_rate} pre_flip_fail_rate={pre_flip_fail_rate} "
            f"recovered_rate={recovery_metrics.get('recovered_rate')} "
            f"mean_recovery_steps={recovery_metrics.get('mean_recovery_steps')}"
            if ok else "FAILED"
        )
    )
    raise SystemExit(0 if ok else 1)
