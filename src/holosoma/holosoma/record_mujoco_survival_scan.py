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

# 2026-09-05: keys record_survival_scan populates into a caller-supplied `extra_metrics_out` dict
# (see that function's own docstring) -- bare metric-name suffixes, same "caller adds its own
# section prefix" convention as the three WANDB_KEY constants above, just not wired into
# FastSACAgent's own wandb logging yet (this patch only extends the wrapper + worker; no training
# call site passes extra_metrics_out, so training's own behavior is untouched by this addition).
# 2026-09-06: ball_speed_max added -- the single fastest hit across the scan (not a mean), for a
# direct comparison against RoboNaldo's own reported peak ball speed (arXiv:2606.11092) rather
# than only its mean-of-hits via ball_speed_mean.
# 2026-09-09: handoff_reached_rate/post_handoff_alive_rate added -- isolates the Kick->Loco AUTO-
# handoff this scan's hold window already runs through (UnifiedLocoKickPolicy._return_to_loco()
# fires on its own at the clip's natural end, no scripted command needed) from the in-kick
# fall_rate above, which conflates the two. See mujoco_kick_survival_scan.py's own docstring
# section on handoff_tick/post_handoff_fall_step for exactly what "reached"/"post-handoff alive"
# mean. Table VIII's "Kick->Loco, end of authored clip" row.
# 2026-09-09: posture_{term}_{early,late} added (12 keys) -- the six penalty_kick_recovery_* error
# quantities over the clip's post-swing recovery tail, opt-in via `stand_start_tick`. Table XI(a)'s
# rescoped outcome metric: that ablation turns off exactly this reward family, so scoring it on
# survival alone measured a phase where the ablated reward is gated OFF entirely. See
# mujoco_kick_survival_scan.py's own POST-STRIKE RECOVERY POSTURE METRICS section for the per-term
# derivation, the early/late window definition, and why these belong in this scan rather than the
# strike-window flip scan.
POSTURE_METRIC_KEYS = tuple(
    f"posture_{name}_{window}"
    for name in ("stance_asymmetry", "yaw_drift", "stand_height", "stand_orientation", "feet_width", "knee_width")
    for window in ("early", "late")
)

EXTRA_METRIC_KEYS = (
    "ball_speed_mean", "ball_speed_std", "ball_speed_n", "ball_speed_max",
    "shot_error_mean", "shot_error_std", "shot_error_n",
    "handoff_reached_rate", "post_handoff_alive_rate",
) + POSTURE_METRIC_KEYS

# 2026-09-07: wires the above into FastSACAgent's own wandb logging (the gap the 2026-09-05 comment
# flagged -- "not wired into training's own wandb logging yet"). Same "METRIC-NAME suffix only, caller
# prefixes Kick_skills_{skill_idx}/" convention as the three WANDB_KEY constants above. Only the
# scalars worth watching as a live training curve are mapped: `_n` fields are sample-size bookkeeping,
# not a rate/error to plot, so deliberately excluded here -- present in extra_metrics_out for a
# caller that wants them (e.g. sim2sim_eval.py), just not queued to wandb by the training call site.
# success_rate_* has no fixed key here (see EXTRA_METRIC_KEYS docstring above -- one per
# success_sigma_m radius, dynamic) -- the training call site builds those directly from
# extra_metrics_out's own `success_rate_{sigma:g}` keys instead of a static mapping.
EXTRA_METRIC_WANDB_KEYS = {
    "ball_speed_mean": "sim2sim/kick_ball_speed_mean",
    "ball_speed_max": "sim2sim/kick_ball_speed_max",
    "shot_error_mean": "sim2sim/kick_shot_error_mean",
    "shot_error_std": "sim2sim/kick_shot_error_std",
}

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
    success_sigma_m: list[float] | None = None,
    extra_metrics_out: dict[str, float | None] | None = None,
    trial_records_out: list[dict] | None = None,
    trajectory_output_path: str | None = None,
    stand_start_tick: int | None = None,
    ball_pos_absolute: tuple[float, float] | None = None,
    ball_pos_range: tuple[tuple[float, float], tuple[float, float]] | None = None,
    hit_window_ticks: tuple[int, int] | None = None,
    reanchor_ball_at_trigger: bool = False,
    stream_output: bool = False,
    stream_prefix: str | None = None,
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

    EXTRA METRICS (2026-09-05, opt-in): mujoco_kick_survival_scan.py's own RESULT/SUMMARY lines
    now additionally carry peak post-contact ball speed, shot error (RoboNaldo's "nearest
    post-contact ball-target distance", arXiv:2606.11092), and a success rate PER radius in
    `success_sigma_m` (strict, unconditional-on-hit -- see that script's own --success-sigma-m
    docstring for why this is a DIFFERENT statistic from `direction_success_sigma_m` above, not a
    threshold variant of it). This function's own RETURN TYPE stays the fixed 3-tuple above
    UNCONDITIONALLY -- same "don't change the shape FastSACAgent's training-time call site already
    unpacks" rationale as record_loco_to_kick_handoff_scan's own `transition_metrics_out` -- these
    go through the caller-owned `extra_metrics_out` dict (mutated in place) instead, ONLY when
    that dict is provided (a non-None dict is what actually enables `--success-sigma-m` on the
    subprocess call at all; passing `success_sigma_m` without `extra_metrics_out` is a no-op).
    Populated keys: `ball_speed_mean`/`_std`/`_n` (mean+/-std peak ball speed over HIT trials,
    always meaningful regardless of kick_aim_enabled), `ball_speed_max` (the single fastest hit
    across the same HIT trials, None under the same "zero hits" condition as the other three --
    2026-09-06, added for a direct comparison against RoboNaldo's own reported PEAK ball speed,
    arXiv:2606.11092, rather than only ball_speed_mean), `shot_error_mean`/`_std`/`_n` (same
    HIT-trials-only convention, meaningful only when kick_aim_enabled -- stays None otherwise, not
    printed by the worker at all in that case), and one `success_rate_{sigma:g}` key per
    (deduplicated, sorted) radius in `success_sigma_m` (defaults to [0.5, 1.0], matching the
    worker's own default exactly, when `success_sigma_m` is omitted but `extra_metrics_out` is
    given) -- also None (absent from the worker's own output) when kick_aim_enabled is False. A
    `_n` of 0 (mean/std None) means zero hit trials -- nothing to average, not a genuine 0.0.

    TRIAL RECORDS (2026-09-06, opt-in, separate from EXTRA METRICS above by design): `extra_
    metrics_out` is a flat dict of SCALAR summary statistics (one number per key, aggregated over
    all trials) -- `kick_aim_theta` doesn't fit that shape at all, since it's a DIFFERENT value on
    every trial (the whole point is comparing outcome vs. angle PER trial, e.g. for a polar
    success-rate-vs-bearing plot -- collapsing it to a mean/std the way ball_speed is would throw
    away exactly the information that made adding it worthwhile). So this is its own output
    parameter, not a new `extra_metrics_out` key: when a list is passed, one dict per trial is
    APPENDED to it (mutated in place, same "caller-owned, opt-in" contract as `extra_metrics_out`),
    parsed straight from that trial's own `RESULT` line, in trial order:
    `{"trial": int, "fall_step": int (-1 = never fell), "min_z": float, "hit_step": int (-1 = never
    hit), "min_target_dist": float | None (None when kick_aim_enabled=False -- no commanded target
    to measure against, translated from the worker's own -1.0 "not applicable" sentinel so this
    field can't be confused with a real, small distance), "max_ball_speed": float (0.0 for a
    whiff), "kick_aim_theta": float | None (degrees, None when kick_aim_enabled=False -- distinct
    from a real 0.0 draw, which does happen)}`. Independent of `extra_metrics_out`/
    `success_sigma_m` -- passing `trial_records_out` alone does NOT enable `--success-sigma-m` on
    the subprocess call (that still requires `extra_metrics_out`), and vice versa.

    TRAJECTORY OUTPUT (`trajectory_output_path`, 2026-09-07, opt-in). Distinct from `trial_records_
    out` above (per-trial SCALAR summaries) and from `extra_metrics_out` (aggregate scalars) --
    this asks the worker to also record each trial's full ball (x, y) PATH over the hold window
    (post-trigger only) and write it to ONE JSON file, since a per-trial array doesn't fit either
    of those two shapes. When set, appends `--trajectory-output-path <path>` to the subprocess
    call; the worker writes `{"step_label", "skill_id", "trials": [{"trial", "kick_aim_theta",
    "fall_step", "hit_step", "trajectory_xy": [[x, y], ...]}, ...]}` there on exit -- this function
    does NOT read that file back (unlike `trial_records_out`, which parses stdout in-process); the
    caller reads it directly once this call returns. None (default) = no file written, exact
    no-op, matching every existing caller.

    ABSOLUTE BALL PLACEMENT (`ball_pos_absolute`, 2026-09-10, opt-in). Places the ball at this
    exact (x, y) instead of jittering around the skill's own trained get_skill_ball_xy -- for
    sweeping a shared placement region across several skills (e.g. a ball-placement hit-rate
    heatmap), where the point is testing positions OUTSIDE any one skill's own trained box.
    `ball_pos_randomization` still applies ON TOP of this (jitter around the absolute point) --
    pass (0.0, 0.0) for an exact fixed placement. The commanded target still anchors D meters from
    the new ball position along the skill's OWN trained bearing (unaffected by this override) --
    see the worker's own --ball-pos-absolute-x/-y docstring for the full ordering rationale. None
    (default) = no override, exact prior behavior.

    RANGE-SAMPLED BALL PLACEMENT (`ball_pos_range`, 2026-09-10, opt-in). `((x_lo, x_hi), (y_lo,
    y_hi))` -- unlike `ball_pos_absolute` (one fixed point for the whole call), this draws a FRESH
    absolute placement EVERY TRIAL, uniform over the given box. Built for a continuous placement
    sweep across many trials in ONE subprocess call (e.g. a ball-placement hit-rate heatmap) --
    the same `seed` across several skill_id calls reproduces the IDENTICAL per-trial placement
    sequence for every skill (confirmed live), which is what lets a caller test several skills
    against the SAME shared set of placements without passing the placements explicitly. Wins
    over `ball_pos_randomization` if both are given (pass (0.0, 0.0) for the latter to avoid
    confusion) -- see the worker's own --ball-pos-range-x/-y docstring.

    HIT-WINDOW GATING (`hit_window_ticks`, 2026-09-10, opt-in). `(lo, hi)`, inclusive, in "ticks
    since kick trigger" -- the SAME unit `get_strike_window_ticks(onnx_path, skill_id)` already
    returns (pass `(window[0], window[1] - 1)` directly, matching kick_to_loco_flip's own
    "stand_start_tick - 1" inclusive convention). When set, a ball<->foot contact only counts as
    `hit_step` inside this window -- without it, an incidental late-hold-window bump (the policy
    auto-returns to locomotion and keeps walking for the rest of the hold window once the clip
    ends) counts as a "hit" indistinguishably from a real strike, which matters most exactly when
    it's least wanted: a displaced/off-nominal ball placement where the strike itself is more
    likely to miss. None (default) = unrestricted, exact prior behavior -- every existing caller
    (training's own periodic scans, sim2sim_eval.py, 60_eval_shooting.py) is unaffected.

    STREAMING (`stream_output`, 2026-09-06, default False -- every existing caller unaffected).
    The worker subprocess already prints one `RESULT` line per trial as it runs, but the default
    `subprocess.run(capture_output=True)` path only makes that text visible AFTER the whole scan
    exits -- silent for the full duration of a `num_trials=150`-sized call, which reads as a hung
    terminal to an interactive caller (60_eval_shooting.py's own use case). `stream_output=True`
    echoes each line to this process's own stdout AS IT ARRIVES, via `subprocess.Popen` instead of
    `subprocess.run` -- everything downstream of that (SUMMARY-line reversed-scan, extra_metrics_
    out, trial_records_out) is completely unchanged, since the streamed text is ALSO accumulated
    into the same `result.stdout` shape those parsers already read. stderr is merged into the same
    stream in this mode (interleaved chronologically with stdout, matching what a directly-invoked
    terminal session would show) rather than captured separately -- `result.stderr` is a fixed
    placeholder string in this mode, since the real content already went to the terminal live.
    `stream_prefix` (only meaningful with `stream_output=True`): prefixes every echoed line with
    `[stream_prefix] ` -- pass the caller's own target label when several scans might stream
    concurrently (e.g. `60_eval_shooting.py`'s own `max_concurrent_targets` > 1), so interleaved
    output from different subprocesses stays attributable instead of reading as one garbled feed.
    Default off, and off for every OTHER existing caller (FastSACAgent's own periodic background
    scans, sim2sim_eval.py) -- streaming raw subprocess output into a training loop's own logs, or
    from several concurrent sim2sim_eval.py scans at once, would be pure noise there, not signal.

    Serialized cluster-wide via a lock file at `lock_path` -- if already held, returns
    (None, None, None) immediately without launching anything (does not block/wait).

    Never raises."""
    resolved_success_sigmas: list[float] = sorted(set(success_sigma_m)) if success_sigma_m else [0.5, 1.0]
    if extra_metrics_out is not None:
        extra_metrics_out.update(dict.fromkeys(EXTRA_METRIC_KEYS, None))
        extra_metrics_out.update({f"success_rate_{sigma:g}": None for sigma in resolved_success_sigmas})

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
        if extra_metrics_out is not None:
            argv += ["--success-sigma-m", *(str(s) for s in resolved_success_sigmas)]
        if trajectory_output_path is not None:
            argv += ["--trajectory-output-path", trajectory_output_path]
        if stand_start_tick is not None:
            argv += ["--track-posture-metrics", "--stand-start-tick", str(stand_start_tick)]
        if ball_pos_absolute is not None:
            argv += ["--ball-pos-absolute-x", str(ball_pos_absolute[0]), "--ball-pos-absolute-y", str(ball_pos_absolute[1])]
        if ball_pos_range is not None:
            (x_lo, x_hi), (y_lo, y_hi) = ball_pos_range
            argv += ["--ball-pos-range-x", str(x_lo), str(x_hi), "--ball-pos-range-y", str(y_lo), str(y_hi)]
        if hit_window_ticks is not None:
            argv += ["--hit-window-lo-tick", str(hit_window_ticks[0]), "--hit-window-hi-tick", str(hit_window_ticks[1])]
        # 2026-09-12: place the ball relative to the robot's actual pose at the trigger rather than
        # at a fixed world point chosen before the settle -- see the worker flag's own help text for
        # the measured settle drift (~4.3cm back, ~1.5deg yaw) and why training never has it.
        if reanchor_ball_at_trigger:
            argv += ["--reanchor-ball-at-trigger"]
        # ROBOJUDO_ORT_INTRA_OP_NUM_THREADS=1: caps this worker's own ONNX Runtime session to a
        # single thread -- see unified_loco_kick_policy.py's own docstring on that env var, and
        # record_mujoco_loco_to_kick_handoff_scan.py's identical guard for the full
        # oversubscription story (observed: ~67 threads/session on a 128-core box). Shared by both
        # launch paths below.
        worker_env = {**os.environ, "ROBOJUDO_ORT_INTRA_OP_NUM_THREADS": "1"}
        try:
            if stream_output:
                # See this function's own STREAMING docstring section. `result` ends up a real
                # subprocess.CompletedProcess so every line below this block (the reversed SUMMARY
                # scan, extra_metrics_out/trial_records_out parsing) reads it identically to the
                # subprocess.run() path -- only how it gets populated differs.
                prefix = f"[{stream_prefix}] " if stream_prefix else ""
                proc = subprocess.Popen(
                    argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=worker_env
                )
                lines: list[str] = []
                try:
                    for line in proc.stdout:
                        print(f"{prefix}{line}", end="", flush=True)
                        lines.append(line)
                    proc.wait(timeout=timeout_s)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                    raise
                result = subprocess.CompletedProcess(
                    argv, proc.returncode, stdout="".join(lines),
                    stderr="(merged into streamed stdout above)",
                )
            else:
                result = subprocess.run(argv, timeout=timeout_s, capture_output=True, text=True, env=worker_env)
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

        if extra_metrics_out is not None:
            # Independent second pass, deliberately NOT folded into the reversed loop above -- that
            # loop's own early-break is tuned to exactly 3 targets; adding a variable-length set of
            # extra tags to its exit condition would complicate an already-verified path for a
            # strictly opt-in feature. mean/std/n share one "TAG step mean_or_NA std_or_NA n" shape
            # (SUMMARY_BALL_SPEED, SUMMARY_SHOT_ERROR); SUMMARY_SUCCESS_<R> shares SUMMARY_HIT's own
            # "TAG step num/denom rate" shape (never NA -- see mujoco_kick_survival_scan.py's own
            # --success-sigma-m docstring for why num_trials is never 0 by construction there).
            def _parse_mean_std_n(tag: str) -> tuple[float | None, float | None, int]:
                for line in reversed(result.stdout.splitlines()):
                    if not line.startswith(tag + " "):
                        continue
                    parts = line.split()
                    if len(parts) >= 5 and parts[2] != "NA":
                        try:
                            return float(parts[2]), float(parts[3]), int(parts[4])
                        except ValueError:
                            logger.warning(f"[sim2sim] Unparseable {tag} line from survival scan: {line!r}")
                    return None, None, 0
                return None, None, 0

            # 2026-09-06: SUMMARY_BALL_SPEED only -- a trailing 5th field (single fastest hit, for
            # a direct RoboNaldo peak-speed comparison; see this file's own EXTRA_METRIC_KEYS
            # comment) appended AFTER the 4-field shape _parse_mean_std_n above already handles.
            # Deliberately a SEPARATE helper rather than widening _parse_mean_std_n's own return
            # shape -- SUMMARY_SHOT_ERROR shares that function and has no analogous max field.
            def _parse_ball_speed_max() -> float | None:
                for line in reversed(result.stdout.splitlines()):
                    if not line.startswith("SUMMARY_BALL_SPEED "):
                        continue
                    parts = line.split()
                    if len(parts) >= 6 and parts[5] != "NA":
                        try:
                            return float(parts[5])
                        except ValueError:
                            logger.warning(f"[sim2sim] Unparseable SUMMARY_BALL_SPEED max field: {line!r}")
                    return None
                return None

            mean, std, n = _parse_mean_std_n("SUMMARY_BALL_SPEED")
            extra_metrics_out["ball_speed_mean"] = mean
            extra_metrics_out["ball_speed_std"] = std
            extra_metrics_out["ball_speed_n"] = n
            extra_metrics_out["ball_speed_max"] = _parse_ball_speed_max()

            # "TAG step num/denom rate_or_NA" shape, same as SUMMARY_HIT/SUMMARY_DIRECTION above --
            # unconditional on kick_aim_enabled (the auto-handoff this measures doesn't depend on
            # aim mode). SUMMARY_HANDOFF_REACHED is never NA (num_trials is never 0 by
            # construction); SUMMARY_POST_HANDOFF_ALIVE is NA when no trial ever reached its own
            # auto-handoff within the hold window.
            def _parse_num_denom_rate(tag: str) -> float | None:
                for line in reversed(result.stdout.splitlines()):
                    if not line.startswith(tag + " "):
                        continue
                    parts = line.split()
                    if len(parts) >= 4 and parts[3] != "NA":
                        try:
                            return float(parts[3])
                        except ValueError:
                            logger.warning(f"[sim2sim] Unparseable {tag} line from survival scan: {line!r}")
                    return None
                return None

            extra_metrics_out["handoff_reached_rate"] = _parse_num_denom_rate("SUMMARY_HANDOFF_REACHED")
            extra_metrics_out["post_handoff_alive_rate"] = _parse_num_denom_rate("SUMMARY_POST_HANDOFF_ALIVE")

            if stand_start_tick is not None:
                # "SUMMARY_POSTURE <step> <term> <window> <n>/<trials> <value_or_NA>" -- one line
                # per (term, window), so this is keyed on parts[2]/parts[3] rather than the tag
                # alone. Left as None (never 0.0) on NA/absent: a fabricated zero would read as a
                # flawless stance, the exact opposite of what "no samples" means.
                for line in result.stdout.splitlines():
                    if not line.startswith("SUMMARY_POSTURE "):
                        continue
                    parts = line.split()
                    if len(parts) < 6:
                        logger.warning(f"[sim2sim] Malformed SUMMARY_POSTURE line from survival scan: {line!r}")
                        continue
                    key = f"posture_{parts[2]}_{parts[3]}"
                    if key not in POSTURE_METRIC_KEYS:
                        logger.warning(f"[sim2sim] Unknown SUMMARY_POSTURE term/window from survival scan: {line!r}")
                        continue
                    if parts[5] == "NA":
                        continue
                    try:
                        extra_metrics_out[key] = float(parts[5])
                    except ValueError:
                        logger.warning(f"[sim2sim] Unparseable SUMMARY_POSTURE value from survival scan: {line!r}")

            if kick_aim_enabled:
                mean, std, n = _parse_mean_std_n("SUMMARY_SHOT_ERROR")
                extra_metrics_out["shot_error_mean"] = mean
                extra_metrics_out["shot_error_std"] = std
                extra_metrics_out["shot_error_n"] = n

                for sigma in resolved_success_sigmas:
                    tag = f"SUMMARY_SUCCESS_{sigma:g}"
                    for line in reversed(result.stdout.splitlines()):
                        if not line.startswith(tag + " "):
                            continue
                        parts = line.split()
                        if len(parts) >= 4:
                            try:
                                extra_metrics_out[f"success_rate_{sigma:g}"] = float(parts[3])
                            except ValueError:
                                logger.warning(f"[sim2sim] Unparseable {tag} line from survival scan: {line!r}")
                        break

        if trial_records_out is not None:
            # Chronological order (NOT reversed, unlike the SUMMARY-line passes above -- those
            # want the LAST occurrence of a one-off tag; this wants every RESULT line, trial 0
            # first) -- see this function's own TRIAL RECORDS docstring section for the field
            # shapes and why this is a separate parameter from extra_metrics_out.
            for line in result.stdout.splitlines():
                if not line.startswith("RESULT "):
                    continue
                parts = line.split()
                if len(parts) < 8:
                    logger.warning(f"[sim2sim] Unparseable RESULT line from survival scan (too few fields): {line!r}")
                    continue
                try:
                    min_target_dist = float(parts[6])
                    trial_records_out.append({
                        "trial": int(parts[2]),
                        "fall_step": int(parts[3]),
                        "min_z": float(parts[4]),
                        "hit_step": int(parts[5]),
                        # -1.0 is the worker's own "no commanded target" sentinel (kick_aim_enabled
                        # =False) -- translated to None here so this field can never be confused
                        # with a real, small (and plausible) distance.
                        "min_target_dist": None if min_target_dist == -1.0 else min_target_dist,
                        "max_ball_speed": float(parts[7]),
                        "kick_aim_theta": float(parts[8]) if len(parts) >= 9 and parts[8] != "NA" else None,
                    })
                except ValueError:
                    logger.warning(f"[sim2sim] Unparseable RESULT line from survival scan: {line!r}")

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
        extra_metrics_out=(extra_metrics := {}),
    )
    ok = fall_rate is not None or hit_rate is not None
    print(
        "record_survival_scan: "
        + (
            f"SUCCESS fall_rate={fall_rate} hit_rate={hit_rate} "
            f"direction_success_rate={direction_success_rate} "
            f"handoff_reached_rate={extra_metrics.get('handoff_reached_rate')} "
            f"post_handoff_alive_rate={extra_metrics.get('post_handoff_alive_rate')}"
            if ok
            else "FAILED"
        )
    )
    raise SystemExit(0 if ok else 1)
