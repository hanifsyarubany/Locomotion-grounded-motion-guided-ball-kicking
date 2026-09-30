"""Thin, stdlib-only wrapper around `mujoco_loco_to_kick_handoff_scan.py`, importable from the
training process (which does not have RoboJuDo installed -- that's why the actual work happens in
a subprocess under the separate `robojudo` conda env). Same lock/subprocess architecture as
`record_mujoco_survival_scan.py`/`record_mujoco_kick_to_loco_flip_scan.py` -- see those modules'
own docstrings for the full rationale (busy lock just means "skip this checkpoint's scan", never
blocks/waits).

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

# METRIC-NAME suffix only -- the caller (FastSACAgent._loco_to_kick_handoff_scan_worker) prefixes
# this with "Kick_skills_{skill_idx}/", same established convention as every sibling sim2sim scan.
MUJOCO_LOCO_TO_KICK_HANDOFF_FALL_WANDB_KEY = "sim2sim/loco_to_kick_handoff_fall_rate"

# 2026-08-30, user-requested companion metric: same N trials, same subprocess invocation, no extra
# rollout cost -- mirrors mujoco_kick_survival_scan.py's own ball CONTACT HIT rate (a real MuJoCo
# geom-geom ball<->foot contact, not an approximation). Shares fall_rate's OWN denominator
# (num_reached_handoff, not num_trials) -- see mujoco_loco_to_kick_handoff_scan.py's own module
# docstring for why both metrics are defined over the same "trials that got a fair test of the
# handoff" population.
MUJOCO_LOCO_TO_KICK_HANDOFF_HIT_WANDB_KEY = "sim2sim/loco_to_kick_handoff_ball_hit_rate"

# Diagnostic companion metric, same N trials, no extra rollout cost: the fraction of trials that
# fell DURING the random-velocity walk itself, before ever reaching the handoff -- a locomotion
# robustness failure, not a handoff failure. Excluded from the fall-rate/hit-rate denominator
# above, reported separately. See mujoco_loco_to_kick_handoff_scan.py's own module docstring.
MUJOCO_LOCO_TO_KICK_HANDOFF_PRE_HANDOFF_FAIL_WANDB_KEY = "sim2sim/loco_to_kick_handoff_pre_handoff_fail_rate"

ROBOJUDO_PYTHON = os.environ.get(
    "HOLOSOMA_ROBOJUDO_PYTHON",
    "/workspaces/isaaclab_arena/submodules/workspaces/conda_env/robojudo/bin/python",
)
WORKER_SCRIPT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "mujoco_loco_to_kick_handoff_scan.py"
)

# Own lock, separate from every other sim2sim mechanism's lock file -- same rationale as every
# sibling scan's own DEFAULT_LOCK_PATH comment.
DEFAULT_LOCK_PATH = os.environ.get(
    "HOLOSOMA_SIM2SIM_LOCO_TO_KICK_HANDOFF_LOCK_PATH", "/tmp/holosoma_sim2sim_loco_to_kick_handoff.lock"
)


# The 5 keys transition_metrics_out is populated with when return_transition_metrics=True --
# see record_loco_to_kick_handoff_scan's own docstring TRANSITION METRICS paragraph. Named exactly
# as the worker's own SUMMARY_<NAME> suffixes, lowercased.
_TRANSITION_METRIC_SUMMARY_NAMES = {
    "tracking_error_early": "SUMMARY_TRACKING_ERROR_EARLY ",
    "tracking_error_late": "SUMMARY_TRACKING_ERROR_LATE ",
    "jerk_early": "SUMMARY_JERK_EARLY ",
    "jerk_late": "SUMMARY_JERK_LATE ",
    "drift": "SUMMARY_DRIFT ",
}


def record_loco_to_kick_handoff_scan(
    onnx_path: str,
    step_label: str,
    *,
    num_trials: int,
    skill_id: int = 0,
    seed: int = 0,
    timeout_s: float = 300.0,
    settle_s: float = 1.5,
    loco_duration_min_s: float = 2.0,
    loco_duration_max_s: float = 3.0,
    post_flip_hold_s: float = 8.0,
    # 2026-09-12: walk-command envelope, for the entry-speed sweep that asks whether the
    # in-motion row's small success advantage scales with how fast the robot is actually moving.
    # None (default) = omit the flag entirely, so the worker keeps its own +/-1.0 training-range
    # default and every previously-recorded invocation is unchanged.
    lin_vel_x_range: tuple[float, float] | None = None,
    lin_vel_y_range: tuple[float, float] | None = None,
    ang_vel_yaw_range: tuple[float, float] | None = None,
    ball_pos_randomization: tuple[float, float] = (0.0, 0.0),
    # 2026-09-12: aim sweep + success scoring. Defaults reproduce this function's original
    # behavior exactly (no theta sweep, no success lines, argv unchanged).
    kick_aim_theta_max_deg: float = 0.0,
    kick_aim_theta_ref_deg: float = 45.0,
    kick_aim_nominal_distance_m: float = 5.0,
    success_sigma_m: list[float] | None = None,
    extra_metrics_out: dict[str, float | None] | None = None,
    return_transition_metrics: bool = False,
    transition_window_steps: int = 50,
    transition_metrics_out: dict[str, float | None] | None = None,
    lock_path: str = DEFAULT_LOCK_PATH,
    stale_lock_timeout_s: float = DEFAULT_STALE_LOCK_TIMEOUT_S,
) -> tuple[float | None, float | None, float | None]:
    """Run an N-trial MuJoCo sim2sim scan of `onnx_path` (RoboJuDo pipeline) that drives random
    locomotion for a randomized 2-3s window, then forces a locomotion->kick handoff, and return
    `(fall_rate, hit_rate, pre_handoff_fail_rate)`, each in [0, 1] or None on failure (busy lock,
    timeout, nonzero exit, or that particular SUMMARY line missing/unparseable). `fall_rate` and
    `hit_rate` are additionally None when every trial fell during the walk itself, before ever
    reaching the handoff (nothing to measure the handoff against -- see
    mujoco_loco_to_kick_handoff_scan.py's own docstring). One subprocess invocation produces all
    three -- the worker script's own N trials already run the deployed policy through
    walk -> handoff -> hold with real MuJoCo contact physics, so the ball-hit read piggybacks on
    the same rollouts as the fall-rate read, no separate scan needed.

    `ball_pos_randomization`: uniform +/- half-range (meters, x/y) jittering the ball's
    ROBOT-RELATIVE spawn offset at the flip instant -- same half-range convention and semantics as
    record_survival_scan's own `ball_pos_randomization`, just applied in the robot's own frame
    (see mujoco_loco_to_kick_handoff_scan.py's own --ball-pos-randomization-x/-y help text for
    why). Default (0.0, 0.0) -- exact nominal placement, unchanged from before this parameter
    existed. Pass the checkpoint's own BallConfig.position_randomization[0] to match
    record_survival_scan's own placement noise for an apples-to-apples Table VIII comparison
    between the settled-state and in-motion handoff rows.

    Unlike record_survival_scan/record_kick_to_loco_flip_scan, `kick_aim_enabled` is not a
    parameter here -- this scan REQUIRES it (the worker script raises if it isn't passed; see that
    script's own module docstring for why the non-aim-mode ball/target placement is incorrect once
    the robot has moved from the origin). Only call this for a checkpoint actually trained with
    kick_aim_enabled=True on this skill.

    TRANSITION METRICS (2026-09-02, opt-in): fall_rate/hit_rate can't distinguish a smooth handoff
    from one that survives while drifting/jittering through it -- see
    mujoco_loco_to_kick_handoff_scan.py's own TRANSITION METRICS docstring section for the full
    rationale and exact definitions (tracking error and jerk, each split into an early/late
    window, plus a single drift scalar). This function's own RETURN TYPE stays the fixed 3-tuple
    above UNCONDITIONALLY -- deliberately NOT a 4th return value, since fast_sac_agent.py's own
    training-time caller already unpacks exactly 3 -- `return_transition_metrics=True` instead
    populates the CALLER-OWNED `transition_metrics_out` dict (mutated in place) with keys
    "tracking_error_early"/"tracking_error_late"/"jerk_early"/"jerk_late"/"drift", each a float or
    None (busy lock/timeout/nonzero-exit/no-reached-handoff-trials/unparseable). Passing
    `transition_metrics_out` without `return_transition_metrics=True` is a no-op (the dict is left
    untouched) -- the flag alone controls whether the worker is even asked to compute these; the
    dict is just where the answer goes if it was.

    Serialized cluster-wide via a lock file at `lock_path` -- if already held, returns
    (None, None, None) immediately without launching anything (does not block/wait).

    Never raises."""
    if return_transition_metrics and transition_metrics_out is not None:
        # Populated with this exact shape up front so every return path below (lock busy, timeout,
        # nonzero exit, or genuine success) leaves the dict in a consistent, fully-keyed state.
        transition_metrics_out.update(dict.fromkeys(_TRANSITION_METRIC_SUMMARY_NAMES, None))

    token = acquire_global_lock(lock_path, stale_lock_timeout_s)
    if token is None:
        logger.warning(f"[sim2sim] Loco-to-kick-handoff scan lock busy -- skipping scan for {onnx_path}.")
        return None, None, None

    try:
        argv = [
            ROBOJUDO_PYTHON, WORKER_SCRIPT_PATH,
            "--onnx-path", onnx_path,
            "--step-label", str(step_label),
            "--settle-s", str(settle_s),
            "--skill-id", str(skill_id),
            "--num-trials", str(num_trials),
            "--seed", str(seed),
            "--loco-duration-min-s", str(loco_duration_min_s),
            "--loco-duration-max-s", str(loco_duration_max_s),
            "--post-flip-hold-s", str(post_flip_hold_s),
            "--ball-pos-randomization-x", str(ball_pos_randomization[0]),
            "--ball-pos-randomization-y", str(ball_pos_randomization[1]),
            "--kick-aim-enabled",
        ]
        for _flag, _rng in (
            ("--lin-vel-x-range", lin_vel_x_range),
            ("--lin-vel-y-range", lin_vel_y_range),
            ("--ang-vel-yaw-range", ang_vel_yaw_range),
        ):
            if _rng is not None:
                argv += [_flag, str(_rng[0]), str(_rng[1])]
        # Appended ONLY when actually requested, so a caller that asks for neither leaves argv
        # byte-identical to every invocation recorded before these flags existed.
        resolved_sigmas = sorted(set(success_sigma_m)) if success_sigma_m else []
        if kick_aim_theta_max_deg > 0.0 or resolved_sigmas:
            argv += [
                "--kick-aim-theta-ref-deg", str(kick_aim_theta_ref_deg),
                "--kick-aim-nominal-distance-m", str(kick_aim_nominal_distance_m),
            ]
        if kick_aim_theta_max_deg > 0.0:
            argv += ["--kick-aim-theta-max-deg", str(kick_aim_theta_max_deg)]
        if resolved_sigmas:
            argv += ["--success-sigma-m", *(str(s) for s in resolved_sigmas)]
        if return_transition_metrics:
            argv += ["--track-transition-metrics", "--transition-window-steps", str(transition_window_steps)]
        try:
            # ROBOJUDO_ORT_INTRA_OP_NUM_THREADS=1: caps this worker's own ONNX Runtime session to
            # a single thread -- see unified_loco_kick_policy.py's own docstring on that env var.
            # This harness's parallelism is PROCESS count (sim2sim_eval.py's max_concurrent_scans),
            # not per-session threading; an unset session default (observed: ~67 threads/session on
            # a 128-core box) oversubscribes every other concurrent worker instead of adding real
            # throughput. Merged onto a copy of the current environment, not replacing it, so
            # ROBOJUDO_PYTHON and everything else this subprocess needs still passes through.
            worker_env = {**os.environ, "ROBOJUDO_ORT_INTRA_OP_NUM_THREADS": "1"}
            result = subprocess.run(argv, timeout=timeout_s, capture_output=True, text=True, env=worker_env)
        except subprocess.TimeoutExpired:
            logger.warning(f"[sim2sim] MuJoCo loco-to-kick-handoff scan timed out after {timeout_s:.0f}s for {onnx_path}")
            return None, None, None

        if result.returncode != 0:
            logger.warning(
                f"[sim2sim] MuJoCo loco-to-kick-handoff scan worker exited {result.returncode} for {onnx_path}\n"
                f"stdout(tail): {result.stdout[-2000:]}\nstderr(tail): {result.stderr[-2000:]}"
            )
            return None, None, None

        fall_rate: float | None = None
        hit_rate: float | None = None
        pre_handoff_fail_rate: float | None = None
        for line in reversed(result.stdout.splitlines()):
            # "SUMMARY_HIT " never matches "SUMMARY_LOCOTOKICKFALL "/"SUMMARY_PREHANDOFFFAIL " as a
            # prefix -- one reversed pass over stdout finds all three, in whichever order printed.
            if pre_handoff_fail_rate is None and line.startswith("SUMMARY_PREHANDOFFFAIL "):
                parts = line.split()
                try:
                    pre_handoff_fail_rate = float(parts[3])
                except (IndexError, ValueError):
                    logger.warning(
                        f"[sim2sim] Unparseable SUMMARY_PREHANDOFFFAIL line from loco-to-kick-handoff scan: {line!r}"
                    )
            if fall_rate is None and line.startswith("SUMMARY_LOCOTOKICKFALL "):
                parts = line.split()
                # "NA" (num_reached_handoff == 0) is an EXPECTED, non-error state, not a parse
                # failure -- stays None with no warning.
                if len(parts) >= 4 and parts[3] != "NA":
                    try:
                        fall_rate = float(parts[3])
                    except ValueError:
                        logger.warning(
                            f"[sim2sim] Unparseable SUMMARY_LOCOTOKICKFALL line from loco-to-kick-handoff scan: {line!r}"
                        )
            if hit_rate is None and line.startswith("SUMMARY_HIT "):
                parts = line.split()
                if len(parts) >= 4 and parts[3] != "NA":
                    try:
                        hit_rate = float(parts[3])
                    except ValueError:
                        logger.warning(
                            f"[sim2sim] Unparseable SUMMARY_HIT line from loco-to-kick-handoff scan: {line!r}"
                        )
            if fall_rate is not None and hit_rate is not None and pre_handoff_fail_rate is not None:
                break

        if fall_rate is None and hit_rate is None and pre_handoff_fail_rate is None:
            logger.warning(
                f"[sim2sim] MuJoCo loco-to-kick-handoff scan worker exited 0 but printed no SUMMARY line for {onnx_path}\n"
                f"stdout(tail): {result.stdout[-2000:]}"
            )

        if return_transition_metrics and transition_metrics_out is not None:
            for key, prefix in _TRANSITION_METRIC_SUMMARY_NAMES.items():
                for line in reversed(result.stdout.splitlines()):
                    if not line.startswith(prefix):
                        continue
                    parts = line.split()
                    if len(parts) >= 3 and parts[2] != "NA":
                        try:
                            transition_metrics_out[key] = float(parts[2])
                        except ValueError:
                            logger.warning(f"[sim2sim] Unparseable {prefix.strip()} line from loco-to-kick-handoff scan: {line!r}")
                    break

        # Success / shot-error, mirroring record_mujoco_survival_scan.py's own extra_metrics_out
        # contract: a key that could not be read stays absent rather than being invented as 0.0,
        # and sim2sim_eval's own "never log a None" rule drops it.
        if extra_metrics_out is not None:
            lines = result.stdout.splitlines()
            for sigma in resolved_sigmas:
                prefix = f"SUMMARY_SUCCESS_{sigma:g} "
                for line in reversed(lines):
                    if not line.startswith(prefix):
                        continue
                    parts = line.split()
                    # "NA" (no trial reached the handoff) is an EXPECTED state, not a parse error.
                    if len(parts) >= 4 and parts[3] != "NA":
                        try:
                            extra_metrics_out[f"success_rate_{sigma:g}"] = float(parts[3])
                        except ValueError:
                            logger.warning(f"[sim2sim] Unparseable {prefix.strip()} line: {line!r}")
                    break
            for line in reversed(lines):
                if not line.startswith("SUMMARY_SHOT_ERROR "):
                    continue
                parts = line.split()
                if len(parts) >= 5 and parts[2] != "NA":
                    try:
                        extra_metrics_out["shot_error_mean"] = float(parts[2])
                        extra_metrics_out["shot_error_std"] = float(parts[3])
                        extra_metrics_out["shot_error_n"] = float(parts[4])
                    except ValueError:
                        logger.warning(f"[sim2sim] Unparseable SUMMARY_SHOT_ERROR line: {line!r}")
                break

        return fall_rate, hit_rate, pre_handoff_fail_rate

    except Exception:
        logger.exception(f"[sim2sim] Unhandled error running MuJoCo loco-to-kick-handoff scan for {onnx_path}")
        return None, None, None
    finally:
        release_global_lock(lock_path, token)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Manual test of record_loco_to_kick_handoff_scan().")
    parser.add_argument("--onnx-path", required=True)
    parser.add_argument("--step-label", default="manual")
    parser.add_argument("--num-trials", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skill-id", type=int, default=0)
    parser.add_argument("--loco-duration-min-s", type=float, default=2.0)
    parser.add_argument("--loco-duration-max-s", type=float, default=3.0)
    parser.add_argument("--post-flip-hold-s", type=float, default=8.0)
    parser.add_argument("--ball-pos-randomization-x", type=float, default=0.0)
    parser.add_argument("--ball-pos-randomization-y", type=float, default=0.0)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    ns = parser.parse_args()

    fall_rate, hit_rate, pre_handoff_fail_rate = record_loco_to_kick_handoff_scan(
        onnx_path=ns.onnx_path,
        step_label=ns.step_label,
        num_trials=ns.num_trials,
        skill_id=ns.skill_id,
        seed=ns.seed,
        loco_duration_min_s=ns.loco_duration_min_s,
        loco_duration_max_s=ns.loco_duration_max_s,
        post_flip_hold_s=ns.post_flip_hold_s,
        ball_pos_randomization=(ns.ball_pos_randomization_x, ns.ball_pos_randomization_y),
        timeout_s=ns.timeout_s,
    )
    ok = fall_rate is not None or hit_rate is not None or pre_handoff_fail_rate is not None
    print(
        "record_loco_to_kick_handoff_scan: "
        + (
            f"SUCCESS fall_rate={fall_rate} hit_rate={hit_rate} pre_handoff_fail_rate={pre_handoff_fail_rate}"
            if ok
            else "FAILED"
        )
    )
    raise SystemExit(0 if ok else 1)
