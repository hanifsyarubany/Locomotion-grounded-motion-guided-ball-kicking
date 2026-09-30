#!/usr/bin/env python3
"""Angle-resolved shooting-coverage orchestrator, for the ICRA benchmark plan's F2 polar coverage
figure and Table 3's per-skill "Success (%)" column (documents/proposal/benchmark_plan.md, §0/§6
P0-1). Thin wrapper over `record_survival_scan` (record_mujoco_survival_scan.py) -- all the actual
subprocess/lock/RoboJuDo-boundary machinery already lives (and is tested) there; this script's own
job is: read each skill's geometry straight from its own ONNX metadata, run the survival scan with
`trial_records_out` to get every trial's own sampled `kick_aim_theta`, bucket those trials into
angle bins, and write one JSON covering every target given.

WHY GEOMETRY COMES FROM THE ONNX, NOT configs/skill/*.yaml (2026-09-06 finding): a skill's nominal
bearing depends on its own `x`/`y`/`target_x`/`target_y`, which the ONNX already embeds verbatim
as `skill_ball_xy`/`skill_target_xy` custom metadata (the SAME fields UnifiedLocoKickPolicy's own
metadata reader and sim2sim_eval.py's discover_num_skills/get_strike_window_ticks already read --
see get_skill_geometry's own docstring). Reading configs/skill/*.yaml directly instead would be
reading a file that can silently drift out of sync with what a specific checkpoint was actually
trained on -- caught live this session: skill_016.yaml's target_x/target_y had an uncommitted edit
that changed its real nominal bearing by 6 degrees, and the benchmark plan's own §1 table had been
computed against the OLDER value. A checkpoint's own embedded metadata can't drift this way -- it
was frozen at export time.

REUSABLE FOR SINGLE-SKILL OR UNIFIED MULTI-SKILL COVERAGE, BY DESIGN: `targets` is just a list of
(onnx_path, skill_id) pairs -- a single-entry list evaluates one skill in isolation (e.g. a
specialist checkpoint), a six-entry list (same unified checkpoint, skill_id 0..5, OR six different
specialist checkpoints, one per skill) produces the full library sweep -- same code path either
way, no special-casing. This is exactly what makes the SAME script back both Table 3's per-skill
row and A2's specialist-vs-unified negative-transfer comparison (§4 A2).

ANGLE BINNING: each skill's OWN [-kick_aim_theta_max_deg, +kick_aim_theta_max_deg] sweep (read from
the checkpoint, not assumed) is split into `num_bins` equal-width bins. Trials are NOT stratified
per bin -- the underlying scan draws kick_aim_theta uniformly at random per trial (mujoco_kick_
survival_scan.py's own `rng.uniform(-theta_max, theta_max)`) and this script bins the RESULTing
per-trial records after the fact. Expected trials/bin is num_trials/num_bins (uniform sampling over
equal-width bins has no systematic bias toward the center or the edges), but actual per-bin counts
have real sampling variance -- a bin CAN legitimately land on 0 trials for a small num_trials/large
num_bins combination, reported as `success_rate: null`, not a fabricated 0.0 (see bin_trials_by_
angle's own docstring). Size num_trials accordingly before trusting a lobe's shape.

OUTPUT: one JSON, `{generated_at, config_path, num_trials_per_target, num_bins, success_radius_m,
seed, skills: [...]}`. Each `skills[i]` carries this skill's own geometry (nominal_bearing_deg,
reachable_band_deg, angular_tolerance_deg -- all read/derived from the ONNX, not retyped from any
markdown table) alongside its measured `bins` (one per angle bin: theta_offset_lo/hi/center_deg,
n, success_rate, hit_rate, mean_min_target_dist_m). A plotting script can overlay the geometric
`reachable_band_deg` boundary against the measured `bins` curve as a direct consistency check
(measured success should fall to ~0 at/before the band edge, not inside it).

OUTPUT LOCATION (2026-09-06): `--output` names a BASE directory, not a file -- the actual JSON
lands at `<output>/<run_timestamp>-<run_name>/eval_shooting.json` (bare `<run_timestamp>`, no
dash/name segment, when the config omits `run_name`) -- same "<timestamp>-<name>" folder-naming
convention sim2sim_eval.py already uses for its own output/ runs, so the two tools' outputs read
the same way at a glance. Defaults to out/eval_shooting (2026-09-07) -- pass --output explicitly
only to redirect elsewhere.

Usage (config file, the reproducible path -- see configs/eval_shooting/example.yaml):
    python scripts/60_eval_shooting.py --config configs/eval_shooting/example.yaml
    # -> out/eval_shooting/<run_timestamp>-<run_name>/eval_shooting.json

Usage (ad-hoc, one or more onnx/skill_id pairs, paired positionally):
    python scripts/60_eval_shooting.py \\
        --onnx-path logs/.../model_0400000.onnx --skill-id 0 \\
        --num-trials 20 --num-bins 4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime

import yaml
from loguru import logger

# scripts/60_eval_shooting.py lives at <repo>/scripts/, a SIBLING of <repo>/src/, not inside the
# holosoma package itself -- so what needs to be on sys.path is <repo>/src/holosoma (the package's
# PARENT dir), same "import the dotted package name, prepend its parent" reasoning sim2sim_eval.py
# already documents for its own, differently-located, sys.path.insert (that file lives INSIDE the
# package and prepends its own parent-of-parent; this one lives OUTSIDE the package entirely and
# must instead point INTO src/holosoma explicitly).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "holosoma"))

from holosoma.record_mujoco_survival_scan import record_survival_scan  # noqa: E402

DEFAULT_NUM_TRIALS = 150
DEFAULT_NUM_BINS = 10
DEFAULT_SUCCESS_RADIUS_M = 0.5  # matches RoboNaldo's success@0.5m convention (Table 2)


# --------------------------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------------------------- #


@dataclass
class ShootingTarget:
    path: str
    skill_id: int = 0
    label: str | None = None
    # 2026-09-06, diagnostic use only: widens the sweep's HALF-WIDTH for THIS target beyond the
    # checkpoint's own ONNX-embedded kick_aim_theta_max_deg -- e.g. to hunt for a policy's real
    # aim direction when it's suspected to sit outside the checkpoint's own nominal band (confirmed
    # live: a distilled skill_016's measured mean_min_target_dist was still monotonically improving
    # at the most-negative bin of its normal +/-15 deg sweep, meaning its true optimum sits further
    # out than that window could ever see). None (default) -- the normal, ONNX-derived half-width,
    # unaffected.
    #
    # SYMMETRIC ONLY, NOT RECENTERABLE -- this is a real constraint of the underlying worker, not
    # a design choice: mujoco_kick_survival_scan.py derives its own nominal_bearing_deg INTERNALLY
    # from the checkpoint's own embedded ball/target xy and samples kick_aim_theta uniformly in
    # +/-kick_aim_theta_max_deg around THAT fixed internal center -- there is no parameter, at any
    # layer (record_survival_scan's own signature has none either), to recenter the sweep on a
    # different bearing. Widening this half-width is the only lever that exists; it necessarily
    # also re-samples the already-known side of the checkpoint's normal band, not just the new
    # territory being explored.
    theta_max_deg_override: float | None = None
    # 2026-09-06: per-target override of the config's own ball_pos_randomization_x/y (below) --
    # None (default, both) means "use the config-level value for this target." Exists because
    # this quantity is a real per-CHECKPOINT training property (BallConfig.position_randomization,
    # e.g. randomize_x/y in that run's own resolved holosoma_config.yaml), not a universal
    # constant -- a config sweeping several differently-trained checkpoints may legitimately need
    # a different value per target rather than one shared guess.
    ball_pos_randomization_x_override: float | None = None
    ball_pos_randomization_y_override: float | None = None


@dataclass
class ShootingEvalConfig:
    targets: list[ShootingTarget] = field(default_factory=list)
    num_trials: int = DEFAULT_NUM_TRIALS
    num_bins: int = DEFAULT_NUM_BINS
    success_radius_m: float = DEFAULT_SUCCESS_RADIUS_M
    seed: int = 0
    max_concurrent_targets: int = 1
    timeout_s: float = 300.0
    # 2026-09-07: when set, each target ALSO gets its own full per-trial ball (x, y) trajectory
    # recorded (see record_mujoco_survival_scan.py's own TRAJECTORY OUTPUT docstring section) and
    # written to "<trajectory_output_dir>/<label>_trajectories.json" -- built for
    # 72_plot_trajectory_heatmap.py, not consumed by anything in this file itself. None (default)
    # = no trajectory recording, matching every prior run's exact behavior/cost.
    trajectory_output_dir: str | None = None
    # 2026-09-06: 0.0 (default) reproduces every prior run's exact behavior -- the ball spawns at
    # the checkpoint's own nominal (x, y) every single trial, no jitter. Set these to match the
    # checkpoint(s) being swept own REAL training-time BallConfig.position_randomization (verified,
    # not guessed, against that run's own resolved holosoma_config.yaml -- e.g. the six-skill
    # distilled checkpoint this file's own example targets were trained with randomize_x/y=0.1
    # uniformly, confirmed directly from its own config, NOT the 0.15 an individual specialist's
    # own config uses -- these are genuinely different per training run, not a shared constant).
    # Without this, every trial's initial condition is identical except for kick_aim_theta, testing
    # a narrower slice of the distribution than the checkpoint actually trained on.
    ball_pos_randomization_x: float = 0.0
    ball_pos_randomization_y: float = 0.0
    # 2026-09-06: names this run's own OUTPUT SUBFOLDER (see main()'s own --output help) --
    # "<run_timestamp>-<run_name>" under the --output base dir, same "<timestamp>-<name>" naming
    # sim2sim_eval.py already uses for its own output/ folders, so a reader who knows one
    # convention already knows both. None (default) -> the bare timestamp, no name segment.
    run_name: str | None = None


def load_config(path: str) -> ShootingEvalConfig:
    """Same hand-parsed, explicit-error-on-bad-value style as sim2sim_eval.py's own
    load_eval_config -- kept consistent rather than introducing a second config-loading
    convention for just this tool."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    raw_targets = raw.get("targets")
    if not raw_targets:
        raise ValueError(f"{path}: 'targets' is required and must be a non-empty list.")
    targets = []
    for i, t in enumerate(raw_targets):
        onnx_path = t.get("path")
        if not onnx_path:
            raise ValueError(f"{path}: targets[{i}] is missing required key 'path'.")
        targets.append(ShootingTarget(
            path=onnx_path, skill_id=int(t.get("skill_id", 0)), label=t.get("label"),
            theta_max_deg_override=(
                float(t["theta_max_deg_override"]) if "theta_max_deg_override" in t else None
            ),
            ball_pos_randomization_x_override=(
                float(t["ball_pos_randomization_x_override"]) if "ball_pos_randomization_x_override" in t else None
            ),
            ball_pos_randomization_y_override=(
                float(t["ball_pos_randomization_y_override"]) if "ball_pos_randomization_y_override" in t else None
            ),
        ))

    def _positive_int(key: str, default: int) -> int:
        v = raw.get(key, default)
        if not isinstance(v, int) or v < 1:
            raise ValueError(f"{path}: '{key}' must be a positive int, got {v!r}")
        return v

    return ShootingEvalConfig(
        targets=targets,
        num_trials=_positive_int("num_trials", DEFAULT_NUM_TRIALS),
        num_bins=_positive_int("num_bins", DEFAULT_NUM_BINS),
        success_radius_m=float(raw.get("success_radius_m", DEFAULT_SUCCESS_RADIUS_M)),
        seed=int(raw.get("seed", 0)),
        max_concurrent_targets=_positive_int("max_concurrent_targets", 1),
        timeout_s=float(raw.get("timeout_s", 300.0)),
        ball_pos_randomization_x=float(raw.get("ball_pos_randomization_x", 0.0)),
        ball_pos_randomization_y=float(raw.get("ball_pos_randomization_y", 0.0)),
        run_name=raw.get("run_name"),
        trajectory_output_dir=raw.get("trajectory_output_dir"),
    )


# --------------------------------------------------------------------------------------------- #
# ONNX-embedded geometry (self-contained -- see this module's own docstring for why)
# --------------------------------------------------------------------------------------------- #


def get_skill_geometry(onnx_path: str, skill_id: int) -> dict:
    """Reads this skill's ball/target spawn point and global kick_aim config straight from the
    ONNX's own embedded custom metadata -- `skill_ball_xy`/`skill_target_xy` (confirmed present,
    2026-09-06: `[[x, y], ...]` / `[[target_x, target_y], ...]`, one entry per embedded skill,
    index-aligned with skill_motion_start_idx -- the SAME fields sim2sim_eval.py's own
    discover_num_skills/get_strike_window_ticks already read from this checkpoint type) and
    `experiment_config`'s embedded `motion_config.kick_aim_theta_max_deg`/
    `kick_aim_nominal_distance_m` (the GLOBAL per-run values -- this checkpoint format does not
    embed a PER-SKILL override list for either, only the training run's own global scalar;
    confirmed empirically uniform, 15.0/5.0, across all six real skill_011-016 checkpoints this
    session, so this is not a live limitation today, only a documented one).

    `nominal_bearing_deg` uses the SAME atan2 convention as SkillConfig.resolved_nominal_bearing_
    deg() (multi_skill.py): 0=+x/forward, positive=+y/the robot's own left -- reimplemented here
    (not imported) since holosoma's training-time dataclasses aren't meant to be constructed from
    this ONNX-only, RoboJuDo-adjacent tool.

    Raises KeyError if this checkpoint predates skill_ball_xy/skill_target_xy metadata, or if
    skill_id is out of range for this checkpoint's own embedded lists -- both indicate the wrong
    checkpoint/skill_id was given, not a normal runtime condition to degrade quietly from.
    """
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    meta = sess.get_modelmeta().custom_metadata_map
    ball_xy = json.loads(meta["skill_ball_xy"])[skill_id]
    target_xy = json.loads(meta["skill_target_xy"])[skill_id]
    nominal_bearing_deg = math.degrees(math.atan2(target_xy[1] - ball_xy[1], target_xy[0] - ball_xy[0]))

    cfg = json.loads(meta["experiment_config"])
    motion_config = cfg["command"]["setup_terms"]["motion_command"]["params"]["motion_config"]
    return {
        "nominal_bearing_deg": nominal_bearing_deg,
        "kick_aim_theta_max_deg": float(motion_config.get("kick_aim_theta_max_deg", 15.0)),
        "kick_aim_nominal_distance_m": float(motion_config.get("kick_aim_nominal_distance_m", 5.0)),
        "ball_xy": ball_xy,
        "target_xy": target_xy,
    }


# --------------------------------------------------------------------------------------------- #
# Angle binning
# --------------------------------------------------------------------------------------------- #


def bin_trials_by_angle(trials: list[dict], theta_max_deg: float, num_bins: int, success_radius_m: float) -> list[dict]:
    """Buckets per-trial `record_survival_scan(trial_records_out=...)` records into `num_bins`
    equal-width bins spanning [-theta_max_deg, +theta_max_deg] (this skill's own sampled range --
    see get_skill_geometry), by each trial's own `kick_aim_theta`. Per bin: `n` (trial count),
    `success_rate` (fraction with min_target_dist <= success_radius_m -- the SAME strict,
    unconditional-on-hit statistic mujoco_kick_survival_scan.py's own SUMMARY_SUCCESS_<R> uses,
    just localized to one angle bin instead of pooled across the whole sweep), `hit_rate`
    (fraction that made contact at all, hit_step != -1), `mean_min_target_dist_m`.

    `success_rate`/`hit_rate`/`mean_min_target_dist_m` are None (not 0.0) for an EMPTY bin (n=0)
    -- "no trials landed here" is a real, distinct state from "trials landed here and none
    succeeded," same "don't fabricate a number where there's nothing to average" convention this
    project's own record_mujoco_survival_scan.py already establishes for ball_speed_mean/etc.
    Bins are NOT guaranteed non-empty -- see this module's own docstring on trial/bin sizing.

    A trial with `kick_aim_theta is None` (kick_aim wasn't enabled for it) is skipped outright --
    should never happen in practice, since this script always calls record_survival_scan with
    kick_aim_enabled=True, but skipped defensively rather than crashing on a malformed record.
    """
    bin_width = 2.0 * theta_max_deg / num_bins
    buckets: list[list[dict]] = [[] for _ in range(num_bins)]
    for t in trials:
        theta = t.get("kick_aim_theta")
        if theta is None:
            continue
        idx = int((theta - (-theta_max_deg)) / bin_width)
        idx = max(0, min(idx, num_bins - 1))  # theta == +theta_max_deg lands in the LAST bin, not a phantom num_bins-th one
        buckets[idx].append(t)

    bins = []
    for i, trials_in_bin in enumerate(buckets):
        lo = -theta_max_deg + i * bin_width
        hi = lo + bin_width
        n = len(trials_in_bin)
        entry = {
            "theta_offset_lo_deg": lo,
            "theta_offset_hi_deg": hi,
            "theta_offset_center_deg": (lo + hi) / 2.0,
            "n": n,
            "success_rate": None,
            "hit_rate": None,
            "mean_min_target_dist_m": None,
        }
        if n > 0:
            n_success = sum(
                1 for t in trials_in_bin if t["min_target_dist"] is not None and t["min_target_dist"] <= success_radius_m
            )
            n_hit = sum(1 for t in trials_in_bin if t["hit_step"] != -1)
            dists = [t["min_target_dist"] for t in trials_in_bin if t["min_target_dist"] is not None]
            entry["success_rate"] = n_success / n
            entry["hit_rate"] = n_hit / n
            entry["mean_min_target_dist_m"] = (sum(dists) / len(dists)) if dists else None
        bins.append(entry)
    return bins


# --------------------------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------------------------- #


def _fresh_lock_path(tag: str) -> str:
    """Same unconditional-unique-lock-path rationale as sim2sim_eval.py's own _fresh_lock_path
    (not imported from there since it's that module's own private helper) -- this script is,
    exactly like that one, its own sole coordinator over how many record_survival_scan calls run
    at once (max_concurrent_targets), so reusing record_survival_scan's shared, fixed-path default
    lock would either serialize this script's own concurrent targets against each other or collide
    with an unrelated live training process's own periodic scans."""
    return os.path.join(tempfile.gettempdir(), f"holosoma_60_eval_shooting_{tag}_{uuid.uuid4().hex}.lock")


def _default_label(onnx_path: str, skill_id: int) -> str:
    run_dir = os.path.basename(os.path.dirname(os.path.abspath(onnx_path)))
    return f"{run_dir}/skill{skill_id}" if run_dir else f"skill{skill_id}"


def _sanitize_run_subdir(name: str) -> str:
    """A `run_name` is a free-form config string but also becomes a filesystem directory NAME
    (one path segment) here -- same rationale and fix as sim2sim_eval.py's own
    _sanitize_run_subdir: collapse any path separator to "_" rather than silently nesting extra
    directories or escaping the --output base dir."""
    return name.replace(os.sep, "_").replace("/", "_")


def eval_one_target(target: ShootingTarget, cfg: ShootingEvalConfig) -> dict:
    """Runs ONE (onnx_path, skill_id)'s full angle sweep and returns its geometry + measured bins
    -- see this module's own docstring for the overall shape. Raises on failure (unlike sim2sim_
    eval.py's own dispatch functions, which absorb a busy-lock/timeout into a None return) --
    get_skill_geometry raising (missing metadata, bad skill_id) or record_survival_scan returning
    an empty trial_records_out (busy lock, timeout, crash -- logged by record_survival_scan itself
    already) both leave nothing meaningful to bin, so the caller (run()) catches and logs per-
    target rather than this function pretending a target with zero data succeeded."""
    geometry = get_skill_geometry(target.path, target.skill_id)
    # theta_max_deg_override widens (never recenters -- see ShootingTarget's own docstring for
    # why recentering isn't possible) the symmetric sweep beyond this checkpoint's own ONNX-
    # embedded kick_aim_theta_max_deg. auto_theta_max is kept around purely for the output's own
    # "was this overridden, and from what" transparency below.
    auto_theta_max = geometry["kick_aim_theta_max_deg"]
    theta_max = target.theta_max_deg_override if target.theta_max_deg_override is not None else auto_theta_max
    ball_pos_randomization_x = (
        target.ball_pos_randomization_x_override
        if target.ball_pos_randomization_x_override is not None
        else cfg.ball_pos_randomization_x
    )
    ball_pos_randomization_y = (
        target.ball_pos_randomization_y_override
        if target.ball_pos_randomization_y_override is not None
        else cfg.ball_pos_randomization_y
    )
    label = target.label or _default_label(target.path, target.skill_id)

    trials: list[dict] = []
    logger.info(
        f"[60-eval-shooting] {label}: starting {cfg.num_trials} trials (theta_max={theta_max:.1f} deg, "
        f"ball_pos_randomization=({ball_pos_randomization_x:.3f}, {ball_pos_randomization_y:.3f}))..."
    )
    trajectory_output_path = None
    if cfg.trajectory_output_dir is not None:
        os.makedirs(cfg.trajectory_output_dir, exist_ok=True)
        trajectory_output_path = os.path.join(cfg.trajectory_output_dir, f"{label}_trajectories.json")
    fall_rate, hit_rate, _direction_success_rate = record_survival_scan(
        onnx_path=target.path,
        step_label="60_eval_shooting",
        num_trials=cfg.num_trials,
        skill_id=target.skill_id,
        seed=cfg.seed,
        ball_pos_randomization=(ball_pos_randomization_x, ball_pos_randomization_y),
        kick_aim_enabled=True,
        kick_aim_theta_max_deg=theta_max,
        kick_aim_nominal_distance_m=geometry["kick_aim_nominal_distance_m"],
        trial_records_out=trials,
        # Live per-trial RESULT-line echo instead of staying silent until the whole scan exits
        # (record_survival_scan's own default) -- this tool is interactive, single-scan-focused
        # diagnostic use, exactly the case that default protects OTHER callers (training's own
        # periodic scans, sim2sim_eval.py) from having to opt out of. stream_prefix keeps
        # concurrent targets (max_concurrent_targets > 1) attributable instead of interleaved into
        # one unlabeled feed.
        stream_output=True,
        stream_prefix=label,
        timeout_s=cfg.timeout_s,
        trajectory_output_path=trajectory_output_path,
        lock_path=_fresh_lock_path("shooting"),
    )
    if not trials:
        raise RuntimeError(
            f"record_survival_scan produced zero trial records for {target.path!r} skill_id="
            f"{target.skill_id} -- busy lock, timeout, or crash (see warnings logged above)."
        )

    nominal = geometry["nominal_bearing_deg"]
    D = geometry["kick_aim_nominal_distance_m"]
    # asin's domain is [-1, 1] -- clamped defensively so a success_radius_m larger than D (an
    # unusual config, not one any real skill uses today) can't raise instead of just saturating at
    # a 90-degree tolerance.
    angular_tolerance_deg = math.degrees(math.asin(min(1.0, cfg.success_radius_m / D)))
    logger.info(f"[60-eval-shooting] {label}: done ({len(trials)} trials, fall_rate={fall_rate}, hit_rate={hit_rate})")

    return {
        "label": label,
        "onnx_path": target.path,
        "skill_id": target.skill_id,
        "nominal_bearing_deg": nominal,
        "kick_aim_theta_max_deg": theta_max,
        "kick_aim_theta_max_deg_auto": auto_theta_max,
        "theta_max_overridden": target.theta_max_deg_override is not None,
        "kick_aim_nominal_distance_m": D,
        "ball_pos_randomization_m": [ball_pos_randomization_x, ball_pos_randomization_y],
        "reachable_band_deg": [nominal - theta_max, nominal + theta_max],
        "angular_tolerance_deg": angular_tolerance_deg,
        "success_radius_m": cfg.success_radius_m,
        "num_trials": len(trials),
        "fall_rate": fall_rate,
        "hit_rate": hit_rate,
        "bins": bin_trials_by_angle(trials, theta_max, cfg.num_bins, cfg.success_radius_m),
    }


def run(cfg: ShootingEvalConfig) -> list[dict]:
    """Evaluates every target in `cfg.targets` -- concurrently, up to `cfg.max_concurrent_targets`
    at once (same ThreadPoolExecutor-over-independent-subprocess-calls rationale as sim2sim_eval.
    py's own run_eval; each target's real work is a blocking subprocess.run() call inside record_
    survival_scan, which releases the GIL, so threads give real parallelism here too). Submitted
    and consumed in `cfg.targets`' own order regardless of which finishes first, so the output
    JSON's `skills` list is always in a stable, predictable order matching the config file. A
    target that raises (see eval_one_target's own docstring) is logged and excluded from the
    result list entirely, rather than the whole run aborting for one bad target."""
    results = []
    with ThreadPoolExecutor(max_workers=cfg.max_concurrent_targets) as pool:
        submissions = [(t, pool.submit(eval_one_target, t, cfg)) for t in cfg.targets]
        for target, future in submissions:
            try:
                results.append(future.result())
            except Exception:
                logger.exception(f"[60-eval-shooting] target failed: path={target.path!r} skill_id={target.skill_id}")
    return results


def write_output(results: list[dict], cfg: ShootingEvalConfig, output_path: str, config_path: str | None) -> None:
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": config_path,
        "num_trials_per_target": cfg.num_trials,
        "num_bins": cfg.num_bins,
        "success_radius_m": cfg.success_radius_m,
        "seed": cfg.seed,
        "skills": results,
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="YAML config listing targets -- see configs/eval_shooting/example.yaml")
    parser.add_argument(
        "--onnx-path", action="append", default=[],
        help="Ad-hoc target (repeatable), paired POSITIONALLY with --skill-id -- the i-th "
        "--onnx-path goes with the i-th --skill-id. Added on top of --config's own targets, if "
        "both are given. For anything you intend to re-run or compare later, use a config instead.",
    )
    parser.add_argument("--skill-id", type=int, action="append", default=[])
    parser.add_argument("--num-trials", type=int, default=None, help=f"Overrides config/default ({DEFAULT_NUM_TRIALS}).")
    parser.add_argument("--num-bins", type=int, default=None, help=f"Overrides config/default ({DEFAULT_NUM_BINS}).")
    parser.add_argument("--success-radius-m", type=float, default=None, help=f"Overrides config/default ({DEFAULT_SUCCESS_RADIUS_M}).")
    parser.add_argument("--seed", type=int, default=None, help="Overrides config/default (0).")
    parser.add_argument("--max-concurrent-targets", type=int, default=None, help="Overrides config/default (1).")
    parser.add_argument(
        "--output", default="out/eval_shooting",
        help="BASE output directory, not a file path -- results land in "
        "<output>/<run_timestamp>-<run_name>/eval_shooting.json (bare <run_timestamp>, no dash/"
        "name segment, if 'run_name' isn't set in the config). Same '<timestamp>-<name>' output-"
        "folder convention sim2sim_eval.py already uses, so a wandb-style run and this one pair "
        "up the same way. All parent dirs created as needed. Default: out/eval_shooting (matches "
        "sim2sim_eval.py's own --output-dir default of out/sim2sim_eval).",
    )
    parser.add_argument(
        "--with-trajectories", action="store_true",
        help="Also record each target's full per-trial ball (x, y) trajectory, written to "
        "<output>/<run_timestamp>-<run_name>/trajectories/<label>_trajectories.json (co-located "
        "with eval_shooting.json, one JSON per target) -- built for "
        "72_plot_trajectory_heatmap.py. Overridden by the config's own 'trajectory_output_dir' "
        "key when that's set explicitly. Off by default: costs one extra array read per tick "
        "(negligible) but produces real files on disk (150 trials x ~400 ticks x 2 floats is a "
        "few MB per target, not free at scale).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else ShootingEvalConfig()

    if args.onnx_path or args.skill_id:
        if len(args.onnx_path) != len(args.skill_id):
            parser.error("--onnx-path and --skill-id must each be passed the same number of times.")
        for p, sid in zip(args.onnx_path, args.skill_id):
            cfg.targets.append(ShootingTarget(path=p, skill_id=sid))
    if not cfg.targets:
        parser.error("No targets given -- pass --config and/or --onnx-path/--skill-id pairs.")

    if args.num_trials is not None:
        cfg.num_trials = args.num_trials
    if args.num_bins is not None:
        cfg.num_bins = args.num_bins
    if args.success_radius_m is not None:
        cfg.success_radius_m = args.success_radius_m
    if args.seed is not None:
        cfg.seed = args.seed
    if args.max_concurrent_targets is not None:
        cfg.max_concurrent_targets = args.max_concurrent_targets

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{run_timestamp}-{_sanitize_run_subdir(cfg.run_name)}" if cfg.run_name else run_timestamp
    output_path = os.path.join(args.output, folder_name, "eval_shooting.json")
    if args.with_trajectories and cfg.trajectory_output_dir is None:
        cfg.trajectory_output_dir = os.path.join(args.output, folder_name, "trajectories")

    results = run(cfg)
    write_output(results, cfg, output_path, args.config)
    logger.info(f"[60-eval-shooting] wrote {len(results)}/{len(cfg.targets)} target result(s) -> {output_path}")
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
