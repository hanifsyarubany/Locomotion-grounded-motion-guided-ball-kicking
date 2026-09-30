#!/usr/bin/env python3
"""Renders a per-skill ball-trajectory density heatmap from 60_eval_shooting.py's own
`--with-trajectories` output: one small-multiple panel per skill, each a 2D histogram of every
recorded ball (x, y) sample pooled across every trial's hold window (post-trigger only -- see
mujoco_kick_survival_scan.py's own --trajectory-output-path docstring), not a raw overlay of
individual paths.

WHY A DENSITY HEATMAP, NOT RAW OVERLAID LINES: with num_trials=150 (this project's usual sweep
size), 150 overlaid raw trajectory lines per panel is visual noise, not signal -- individual paths
become indistinguishable and the plot can't show where the ball ACTUALLY spends its time. A pooled
2D histogram answers a different, more useful question ("where does this skill's ball go, and how
consistently") at a glance. A handful of individual sample paths (--num-sample-paths, default 6)
are drawn thinly on top for texture/context, not as the primary signal.

WHY THIS IS DIAGNOSTIC, NOT JUST DECORATIVE: a trajectory shows HOW a kick failed, which a scalar
success/hit rate cannot. Confirmed live before writing this script (2026-09-07): skill_012 on the
current distilled checkpoint (model_0600000) shows EVERY sampled trajectory a single, unmoving
point (first sample == last sample, hit_step always -1) -- a total whiff, the ball is never
touched at all -- a materially different failure mode from "hits the ball but aims it wrong" (the
one skill_012 showed on an EARLIER checkpoint, model_0400000, per 71_plot_polar_coverage.py's own
docstring: hit_rate=1.0 but success_rate=0.0 everywhere). A bare success-rate number can't tell
these apart; this plot does, instantly.

COORDINATE FRAME: raw MuJoCo world (x, y) -- NOT transformed. This is only valid because every
trial resets the robot to the origin, facing +x, via the same fixed keyframe (mujoco_kick_
rollout_worker.py's own reset convention) -- so "world frame" and "robot-relative, robot facing
+x" are the SAME frame here, and plotting raw ball_qpos directly needs no rotation/translation.
Robot origin (0, 0) is marked on every panel as a fixed reference point.

AXIS LIMITS: shared across every panel in the figure (computed from the union of all skills' own
data, not per-panel) -- so a skill whose ball barely moves and one whose ball travels several
meters are visually comparable by density/extent, not silently rescaled to fill their own panel.

Usage (paste 60_eval_shooting.py's own logged run-folder path straight in):
    python scripts/72_plot_trajectory_heatmap.py --input out/eval_shooting/<run_timestamp>-<run_name>
    (requires that run to have been produced with --with-trajectories -- see 60_eval_shooting.py's
    own module docstring; without it, that run's folder has no trajectories/ subdirectory and this
    script raises a clear FileNotFoundError rather than silently plotting nothing.)
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os

import matplotlib

matplotlib.use("Agg")  # headless -- this environment has no display
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

TRAJECTORIES_SUBDIR = "trajectories"  # matches 60_eval_shooting.py's own --with-trajectories default
EVAL_SHOOTING_FILENAME = "eval_shooting.json"


def resolve_run_dir(path: str) -> str:
    """Accepts 60_eval_shooting.py's own run folder directly (the ONLY supported shape here,
    unlike 71_plot_polar_coverage.py's file-or-folder flexibility -- a trajectory heatmap needs
    the whole `trajectories/` subfolder, not one file, so there is no single-file equivalent to
    accept)."""
    if not os.path.isdir(path):
        raise NotADirectoryError(f"{path!r} is not a directory -- pass 60_eval_shooting.py's own run folder.")
    traj_dir = os.path.join(path, TRAJECTORIES_SUBDIR)
    if not os.path.isdir(traj_dir):
        raise FileNotFoundError(
            f"{path!r} has no {TRAJECTORIES_SUBDIR}/ subdirectory -- this run wasn't produced with "
            "--with-trajectories (see 60_eval_shooting.py's own module docstring)."
        )
    return path


def load_trajectories(run_dir: str) -> dict[str, dict]:
    """Returns {label: worker's own trajectory JSON dict} for every "<label>_trajectories.json"
    file found in run_dir's trajectories/ subfolder, in filename-sorted order."""
    traj_dir = os.path.join(run_dir, TRAJECTORIES_SUBDIR)
    out: dict[str, dict] = {}
    for path in sorted(glob.glob(os.path.join(traj_dir, "*_trajectories.json"))):
        label = os.path.basename(path)[: -len("_trajectories.json")]
        with open(path) as f:
            out[label] = json.load(f)
    return out


def load_eval_shooting_geometry(run_dir: str) -> dict[str, dict]:
    """{label: skill entry} from eval_shooting.json in the SAME run folder, for the reachable-band
    reference rays drawn on each panel -- geometry the trajectory JSON itself doesn't carry. Empty
    dict (not an error) if eval_shooting.json is missing, so a trajectories/-only folder still
    plots, just without the reference rays."""
    path = os.path.join(run_dir, EVAL_SHOOTING_FILENAME)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    return {s["label"]: s for s in data.get("skills", [])}


def _pooled_points(traj: dict) -> np.ndarray:
    """(N, 2) array of every (x, y) sample across every trial in one skill's trajectory JSON."""
    pts = [p for trial in traj["trials"] for p in trial["trajectory_xy"]]
    return np.array(pts) if pts else np.zeros((0, 2))


def plot_trajectory_heatmap(
    trajectories: dict[str, dict],
    geometry: dict[str, dict],
    output_path: str,
    *,
    bins: int = 60,
    num_sample_paths: int = 6,
    seed: int = 0,
    dpi: int = 200,
) -> None:
    labels = list(trajectories.keys())
    n = len(labels)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)

    all_pts = np.concatenate([_pooled_points(trajectories[label]) for label in labels], axis=0)
    if all_pts.shape[0] == 0:
        raise ValueError("Every trajectory file is empty (zero trials or zero samples) -- nothing to plot.")
    # 5% padding around the data's own bounding box, not a hardcoded window -- this project's
    # skills span wildly different distances (1.3m-5.3m, see benchmark_plan.md's own skill
    # inventory), so no single fixed window is right for all of them.
    pad_x = 0.05 * max(1.0, all_pts[:, 0].ptp())
    pad_y = 0.05 * max(1.0, all_pts[:, 1].ptp())
    xlim = (all_pts[:, 0].min() - pad_x, all_pts[:, 0].max() + pad_x)
    ylim = (all_pts[:, 1].min() - pad_y, all_pts[:, 1].max() + pad_y)

    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5.0 * nrows), squeeze=False)

    for idx, label in enumerate(labels):
        ax = axes[idx // ncols][idx % ncols]
        traj = trajectories[label]
        pts = _pooled_points(traj)
        trials = traj["trials"]

        if pts.shape[0] > 0:
            ax.hist2d(pts[:, 0], pts[:, 1], bins=bins, range=[xlim, ylim], cmap="magma", cmin=1)

        # A handful of individual sample paths on top, thin and semi-transparent -- context/texture
        # for the density field above, not the primary signal (see module docstring). Preferentially
        # samples HIT trials (a whiff's "path" is a single static point, not informative to trace).
        hit_trials = [t for t in trials if t["hit_step"] != -1 and len(t["trajectory_xy"]) > 1]
        pool = hit_trials if hit_trials else [t for t in trials if len(t["trajectory_xy"]) > 1]
        sample = rng.choice(len(pool), size=min(num_sample_paths, len(pool)), replace=False) if pool else []
        for i in sample:
            path_xy = np.array(pool[i]["trajectory_xy"])
            ax.plot(path_xy[:, 0], path_xy[:, 1], color="cyan", linewidth=0.8, alpha=0.6)

        # Reachable-band reference rays from the mean sample-0 point (this JSON carries no exact
        # ball-spawn field of its own -- the mean starting sample across trials is a good proxy,
        # jitter is at most a few cm per SkillConfig.randomize_x/y).
        geo = geometry.get(label)
        if geo is not None and pts.shape[0] > 0:
            origin = np.array([t["trajectory_xy"][0] for t in trials if t["trajectory_xy"]]).mean(axis=0)
            band_lo, band_hi = geo["reachable_band_deg"]
            D = geo["kick_aim_nominal_distance_m"]
            for deg in (band_lo, geo["nominal_bearing_deg"], band_hi):
                rad = math.radians(deg)
                ax.plot(
                    [origin[0], origin[0] + D * math.cos(rad)],
                    [origin[1], origin[1] + D * math.sin(rad)],
                    color="white", linewidth=0.8, linestyle="--", alpha=0.5,
                )

        ax.scatter([0], [0], marker="^", color="white", s=60, zorder=5, label="robot origin")
        n_trials = len(trials)
        n_hit = sum(1 for t in trials if t["hit_step"] != -1)
        ax.set_title(f"{label}  (hit {n_hit}/{n_trials})", fontsize=10)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)", fontsize=8)
        ax.set_ylabel("y (m)", fontsize=8)
        ax.tick_params(labelsize=7)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle("Ball trajectory density per skill (pooled across all trials, hold window only)", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    pdf_path = output_path.rsplit(".", 1)[0] + ".pdf" if "." in output_path else output_path + ".pdf"
    fig.savefig(pdf_path, dpi=dpi)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input", required=True,
        help="60_eval_shooting.py's own run folder (the one --with-trajectories was passed for) "
        "-- e.g. 'out/eval_shooting/<run_timestamp>-<run_name>'.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output image path (.png -- a .pdf twin is written alongside it). Default: "
        "'trajectory_heatmap.png' in the resolved run folder.",
    )
    parser.add_argument("--bins", type=int, default=60, help="2D histogram bin count per axis.")
    parser.add_argument("--num-sample-paths", type=int, default=6, help="Individual paths drawn per panel, 0 to disable.")
    parser.add_argument("--seed", type=int, default=0, help="Sample-path selection RNG seed.")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.input)
    trajectories = load_trajectories(run_dir)
    if not trajectories:
        logger.error(f"[72-plot-trajectory-heatmap] {run_dir!r}'s trajectories/ folder is empty -- nothing to plot.")
        return 1
    geometry = load_eval_shooting_geometry(run_dir)
    output_path = args.output or os.path.join(run_dir, "trajectory_heatmap.png")

    plot_trajectory_heatmap(
        trajectories, geometry, output_path,
        bins=args.bins, num_sample_paths=args.num_sample_paths, seed=args.seed, dpi=args.dpi,
    )
    logger.info(f"[72-plot-trajectory-heatmap] wrote {len(trajectories)} skill panel(s) -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
