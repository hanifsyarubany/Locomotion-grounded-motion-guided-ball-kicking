"""Thin, stdlib-only wrapper around `mujoco_kick_survival_scan.py`, importable from the training
process (which does not have RoboJuDo installed -- that's why the actual work happens in a
subprocess under the separate `robojudo` conda env; see that file's module docstring). Same
lock/subprocess architecture as `record_mujoco_kick_rollout.py` -- see that module's own docstring
for the full rationale (busy lock just means "skip this checkpoint's scan", never blocks/waits).

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

# 2026-08-19: the METRIC-NAME suffix only, not a full wandb key -- the caller (FastSACAgent.
# _mujoco_survival_scan_worker) prefixes this with "Kick_skills_{skill_idx}/" so the result lands
# in the SAME per-skill wandb section as UnifiedManager's own training-time per-skill kick metrics
# (e.g. "Kick_skills_0/kick_alive_frac"), not a separate top-level "sim2sim/" section -- wandb
# groups by "/"-delimited prefix regardless of which code path constructed the key (confirmed
# against logging_utils.py's own "Section::metric" -> "Section/metric" translation for the
# log_dict-routed per-skill metrics). Always per-skill, even in single-skill mode -- matching the
# established "Kick_skills_0 is populated even in single-skill mode" convention (unified_manager.py,
# 2026-08-06), so this metric always lives in the same section as its training-time siblings rather
# than switching key shape based on skill count.
MUJOCO_SURVIVAL_SCAN_WANDB_KEY = "sim2sim/kick_fall_rate"

# 2026-08-21, user-requested companion metric: same N trials, same subprocess invocation, no extra
# rollout cost -- mujoco_kick_survival_scan.py now also reports a ball CONTACT HIT rate (a real
# MuJoCo geom-geom ball<->foot contact, not an approximation -- see that file's own
# _ball_foot_contact_now docstring) via a second "SUMMARY_HIT " line. Same per-skill wandb
# section-prefixing convention as MUJOCO_SURVIVAL_SCAN_WANDB_KEY above (this is a METRIC-NAME
# suffix only; the caller prefixes "Kick_skills_{skill_idx}/").
MUJOCO_SURVIVAL_SCAN_HIT_WANDB_KEY = "sim2sim/kick_ball_hit_rate"

# 2026-08-23, user-requested companion metric: same N trials, no extra rollout cost -- among
# trials that ACTUALLY hit the ball (num_hit, not num_trials -- a whiff has no departure direction
# to grade), what fraction landed within direction_success_sigma_m of the commanded kick_aim_theta
# target. Only meaningful for kick_aim_enabled checkpoints -- None for any other. Same per-skill
# wandb section-prefixing convention as the two keys above.
MUJOCO_SURVIVAL_SCAN_DIRECTION_WANDB_KEY = "sim2sim/kick_direction_success_rate"

ROBOJUDO_PYTHON = os.environ.get(
    "HOLOSOMA_ROBOJUDO_PYTHON",
    "/workspaces/isaaclab_arena/submodules/workspaces/conda_env/robojudo/bin/python",
)
WORKER_SCRIPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mujoco_kick_survival_scan.py")

# Own lock, separate from record_mujoco_kick_rollout.py's DEFAULT_LOCK_PATH/DEFAULT_HANDOFF_LOCK_PATH
# -- same rationale as that module's own comment on why the kick/walk/handoff rollouts each get
# their own lock file: this fires from the same checkpoint-save event, on the same cadence knob's
# family, as the video rollouts, so sharing a lock would make whichever starts first always win.
DEFAULT_LOCK_PATH = os.environ.get(
    "HOLOSOMA_SIM2SIM_SURVIVAL_SCAN_LOCK_PATH", "/tmp/holosoma_sim2sim_survival_scan.lock"
)


def record_survival_scan(
    onnx_path: str,
    step_label: str,
    *,
    num_trials: int,
    skill_id: int = 0,
    seed: int = 0,
    ball_pos_randomization: tuple[float, float] = (0.0, 0.0),
    kick_aim_enabled: bool = False,
    kick_aim_theta_max_deg: float = 15.0,
    kick_aim_theta_ref_deg: float = 45.0,
    kick_aim_nominal_distance_m: float = 5.0,
    direction_success_sigma_m: float = 1.0,
    timeout_s: float = 300.0,
    settle_s: float = 1.5,
    hold_s: float = 8.0,
    lock_path: str = DEFAULT_LOCK_PATH,
    stale_lock_timeout_s: float = DEFAULT_STALE_LOCK_TIMEOUT_S,
) -> tuple[float | None, float | None, float | None]:
    """Run an N-trial in-distribution MuJoCo sim2sim scan of `onnx_path` (RoboJuDo pipeline) and
    return `(fall_rate, hit_rate, direction_success_rate)`, each in [0, 1], or None for any/all on
    failure (busy lock, timeout, nonzero exit, or that particular SUMMARY line missing/
    unparseable/not applicable). One subprocess invocation produces all three -- mujoco_kick_
    survival_scan.py's own N trials already run the deployed policy through settle -> trigger ->
    hold with real MuJoCo contact physics, so the ball-hit read (2026-08-21) and direction-success
    read (2026-08-23) both piggyback on the same rollouts as the fall-rate read, no separate scan
    needed. `direction_success_rate` is additionally None whenever `kick_aim_enabled=False` (no
    commanded direction exists to grade) or when zero trials hit the ball (nothing to measure yet
    -- distinct from a genuine 0.0 rate, which means trials hit but missed the direction).

    `ball_pos_randomization`: the uniform +/- half-range (meters) each trial jitters the ball spawn
    within, AROUND that skill's own nominal get_skill_ball_xy -- pass the checkpoint's OWN
    BallConfig.position_randomization (read by the caller from the live training config; this
    function never invents a range). (0.0, 0.0) reproduces a fixed, non-jittered spawn.

    `kick_aim_enabled`/`kick_aim_theta_max_deg`/`kick_aim_theta_ref_deg` (2026-08-22, azimuth-aim
    refactor): the ONLY way to vary the observed target now that BallConfig.target_randomization
    has been removed from the config layer entirely (it had no live consumer once every skill in
    this project moved to kick_aim_enabled=True). When `kick_aim_enabled` is True, each trial
    samples kick_aim_theta (uniform, +/- `kick_aim_theta_max_deg`) around this skill's own
    calibrated nominal bearing -- see mujoco_kick_survival_scan.py's own --kick-aim-enabled
    docstring. Pass the checkpoint's OWN kick_aim_enabled/kick_aim_theta_max_deg/
    kick_aim_theta_ref_deg (read by the caller from the live training config) -- ONLY True for a
    checkpoint actually trained with kick_aim_enabled=True on that skill.

    `kick_aim_nominal_distance_m`/`direction_success_sigma_m` (2026-08-23): only used when
    `kick_aim_enabled` is True. The first MUST match the checkpoint's own MultiSkillConfig/
    BallConfig.kick_aim_nominal_distance_m (the fixed distance each trial's commanded target point
    is synthesized at); the second is the success threshold, meters, mirroring shooting.py's own
    error_ball_to_target `sigma` default (1.0) rather than inventing a separate eval-only number --
    see mujoco_kick_survival_scan.py's own --direction-success-sigma-m docstring for the implied
    angular-tolerance derivation.

    Serialized cluster-wide via a lock file at `lock_path` -- if already held, returns
    (None, None, None) immediately without launching anything (does not block/wait).

    Never raises."""
    token = acquire_global_lock(lock_path, stale_lock_timeout_s)
    if token is None:
        logger.warning(f"[sim2sim] Survival-scan rollout lock busy -- skipping scan for {onnx_path}.")
        return None, None, None

    try:
        argv = [
            ROBOJUDO_PYTHON, WORKER_SCRIPT_PATH,
            "--onnx-path", onnx_path,
            "--step-label", str(step_label),
            "--settle-s", str(settle_s),
            "--hold-s", str(hold_s),
            "--skill-id", str(skill_id),
            "--num-trials", str(num_trials),
            "--seed", str(seed),
            "--ball-pos-randomization-x", str(ball_pos_randomization[0]),
            "--ball-pos-randomization-y", str(ball_pos_randomization[1]),
        ]
        if kick_aim_enabled:
            argv += [
                "--kick-aim-enabled",
                "--kick-aim-theta-max-deg", str(kick_aim_theta_max_deg),
                "--kick-aim-theta-ref-deg", str(kick_aim_theta_ref_deg),
                "--kick-aim-nominal-distance-m", str(kick_aim_nominal_distance_m),
                "--direction-success-sigma-m", str(direction_success_sigma_m),
            ]
        try:
            result = subprocess.run(argv, timeout=timeout_s, capture_output=True, text=True)
        except subprocess.TimeoutExpired:
            logger.warning(f"[sim2sim] MuJoCo survival scan timed out after {timeout_s:.0f}s for {onnx_path}")
            return None, None, None

        if result.returncode != 0:
            logger.warning(
                f"[sim2sim] MuJoCo survival scan worker exited {result.returncode} for {onnx_path}\n"
                f"stdout(tail): {result.stdout[-2000:]}\nstderr(tail): {result.stderr[-2000:]}"
            )
            return None, None, None

        fall_rate: float | None = None
        hit_rate: float | None = None
        direction_success_rate: float | None = None
        for line in reversed(result.stdout.splitlines()):
            # "SUMMARY_HIT "/"SUMMARY_DIRECTION " never match "SUMMARY " as a prefix (the char
            # right after "SUMMARY" differs), so all three are unambiguous -- one reversed pass
            # over stdout finds them, in whichever order they were printed.
            if hit_rate is None and line.startswith("SUMMARY_HIT "):
                parts = line.split()
                try:
                    hit_rate = float(parts[3])
                except (IndexError, ValueError):
                    logger.warning(f"[sim2sim] Unparseable SUMMARY_HIT line from survival scan: {line!r}")
            if direction_success_rate is None and line.startswith("SUMMARY_DIRECTION "):
                parts = line.split()
                # "NA" (num_hit == 0 -- nothing to measure yet) is an EXPECTED, non-error state,
                # not a parse failure -- stays None with no warning, same as kick_aim_enabled=False
                # never printing this line at all.
                if len(parts) >= 4 and parts[3] != "NA":
                    try:
                        direction_success_rate = float(parts[3])
                    except ValueError:
                        logger.warning(
                            f"[sim2sim] Unparseable SUMMARY_DIRECTION line from survival scan: {line!r}"
                        )
            if fall_rate is None and line.startswith("SUMMARY "):
                parts = line.split()
                try:
                    fall_rate = float(parts[3])
                except (IndexError, ValueError):
                    logger.warning(f"[sim2sim] Unparseable SUMMARY line from survival scan: {line!r}")
            if fall_rate is not None and hit_rate is not None and direction_success_rate is not None:
                break

        if fall_rate is None and hit_rate is None:
            logger.warning(
                f"[sim2sim] MuJoCo survival scan worker exited 0 but printed no SUMMARY line for {onnx_path}\n"
                f"stdout(tail): {result.stdout[-2000:]}"
            )
        return fall_rate, hit_rate, direction_success_rate

    except Exception:
        logger.exception(f"[sim2sim] Unhandled error running MuJoCo survival scan for {onnx_path}")
        return None, None, None
    finally:
        release_global_lock(lock_path, token)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Manual test of record_survival_scan().")
    parser.add_argument("--onnx-path", required=True)
    parser.add_argument("--step-label", default="manual")
    parser.add_argument(
        "--num-trials", type=int, default=32,
        help="32 (2026-08-19, raised from 8) resolves a rate to ~3.1%% steps -- see "
        "FastSACConfig.mujoco_survival_scan_num_trials's own docstring for why 8 was too coarse.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skill-id", type=int, default=0)
    parser.add_argument("--ball-pos-randomization-x", type=float, default=0.0)
    parser.add_argument("--ball-pos-randomization-y", type=float, default=0.0)
    parser.add_argument("--kick-aim-enabled", action="store_true")
    parser.add_argument("--kick-aim-theta-max-deg", type=float, default=15.0)
    parser.add_argument("--kick-aim-theta-ref-deg", type=float, default=45.0)
    parser.add_argument("--kick-aim-nominal-distance-m", type=float, default=5.0)
    parser.add_argument("--direction-success-sigma-m", type=float, default=1.0)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    ns = parser.parse_args()

    fall_rate, hit_rate, direction_success_rate = record_survival_scan(
        onnx_path=ns.onnx_path,
        step_label=ns.step_label,
        kick_aim_enabled=ns.kick_aim_enabled,
        kick_aim_theta_max_deg=ns.kick_aim_theta_max_deg,
        kick_aim_theta_ref_deg=ns.kick_aim_theta_ref_deg,
        kick_aim_nominal_distance_m=ns.kick_aim_nominal_distance_m,
        direction_success_sigma_m=ns.direction_success_sigma_m,
        num_trials=ns.num_trials,
        skill_id=ns.skill_id,
        seed=ns.seed,
        ball_pos_randomization=(ns.ball_pos_randomization_x, ns.ball_pos_randomization_y),
        timeout_s=ns.timeout_s,
    )
    ok = fall_rate is not None or hit_rate is not None
    print(
        "record_survival_scan: "
        + (
            f"SUCCESS fall_rate={fall_rate} hit_rate={hit_rate} "
            f"direction_success_rate={direction_success_rate}"
            if ok
            else "FAILED"
        )
    )
    raise SystemExit(0 if ok else 1)
