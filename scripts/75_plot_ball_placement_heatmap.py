#!/usr/bin/env python3
"""Renders FOUR figures from 74_eval_ball_placement_sweep.py's output, as two pairs (a heatmap +
a matching trajectory plot) gated on two different criteria:

  CONTACT pair (`<out>_contact_heatmap.png`, `<out>_contact_trajectories.png`) -- did a foot touch
  the ball at all, strike-window-gated but otherwise undiscriminating. The two halves deliberately
  use different skill selectors:
    - heatmap: NEAREST-TRAINED-SKILL -- per placement, only the skill whose trained ball_xy is
      closest counts. "Any of N skills" saturated (98.3% on the 2026-09-10 1500-placement/7-skill
      run -- only 26 placements had zero contacts) because it picks the skill AFTER seeing the
      outcome; one skill per attempt, chosen by a stated rule, is how the deployed system runs.
    - trajectories: every contact from every skill (per_skill_hit), to show the full shot spread.

  SUCCESS pair (`<out>_success_heatmap.png`, `<out>_success_trajectories.png`) -- gated on
  any_direction_hit/per_skill_direction_hit (present since 2026-09-10; older summary JSONs lack
  these fields and this script skips this pair with a warning rather than crashing): a hit ALSO
  requires that trial's min_target_dist to fall within --direction-success-sigma-m of the commanded
  target. Answers "does the library ever put a SHOT ON TARGET from this XY region".

  WHY BOTH, NOT JUST SUCCESS: confirmed on a real 1500-placement/7-skill run (2026-09-10) that these
  tell very different stories on the exact same placement grid -- any_hit_rate was 98.3% (nearly the
  whole swept region lights up green) while any_direction_hit_rate was far lower, because most
  contacts across a wide sweep are glancing touches, not directed kicks (per-skill direction-
  success-of-hits ranged 6-36% that run). A reader shown ONLY the contact figure would reasonably
  read "hit" as "successfully kicked", which the contact criterion alone does not support -- hence
  both pairs, clearly labeled by what each one actually measures, rather than picking one.

  Each heatmap cell is the fraction of placements landing in it meeting that heatmap's criterion
  under its selector (nearest-trained skill for contact, any skill for success). Each skill's own trained box (its ball_xy +/- trained_jitter_m) is
  outlined and labeled on top, directly answering "does this extend beyond training, or stop at the
  trained boxes". Each trajectory plot draws every qualifying trial's shot, speed-colored, from
  whichever skill(s) actually met that pair's criterion there (RoboNaldo-style: reuses
  73_plot_shot_trajectories.py's own pitch/speed-color/closest-approach-truncation machinery).
  Unlike 73's own combined figure (fixed tight placement per skill, wide bearing spread), this one
  holds bearing roughly fixed per skill and spreads the PLACEMENT instead -- the RoboNaldo-style
  complement 73's own docstring named as a follow-up, now with the honest denominator (the heatmap)
  that a bare trajectory scatter can't provide on its own.

BINNING, NOT PER-PLACEMENT SCATTER: 74's own placements are continuous, uniformly-drawn (x, y)
draws, not a regular grid -- but a raw scatter of hit/miss points reads as noisy dots, not a
heatmap, and doesn't answer "what's the rate roughly HERE" at a glance. Binning into --bins x
--bins cells and averaging the chosen criterion within each (nan, rendered gray, for an empty cell)
is the standard way to turn scattered Bernoulli samples into a readable rate map -- this is display
binning only; 74's own placements stay continuous, nothing about the SWEEP is gridded.

Usage:
    python scripts/75_plot_ball_placement_heatmap.py \\
        --input out/ball_placement_sweep/<timestamp>/ball_placement_sweep.json
"""

from __future__ import annotations

import argparse
import json
import math
import os

import matplotlib

matplotlib.use("Agg")  # headless -- this environment has no display
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
from matplotlib.ticker import MultipleLocator
from loguru import logger

_PITCH_GREEN = "#4e7a3e"
_PITCH_STRIPE = "#568742"
_SPEED_CMAP = "plasma"


def load_summary(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_trajectories(run_dir: str, skill_ids: list[int]) -> dict[int, dict]:
    out = {}
    for sid in skill_ids:
        path = os.path.join(run_dir, "trajectories", f"skill_{sid}_trajectories.json")
        with open(path) as f:
            out[sid] = json.load(f)
    return out


def tag_nearest_skill_hit(summary: dict) -> None:
    """Adds nearest_skill / nearest_skill_hit to every placement: the contact result of ONLY the
    skill whose trained ball_xy is closest to that placement (see the module docstring's CONTACT pair)."""
    spots = {sid: np.asarray(info["ball_xy"], dtype=float) for sid, info in summary["per_skill"].items()}
    for p in summary["placements"]:
        xy = np.array([p["x"], p["y"]])
        sid = min(spots, key=lambda s: float(np.linalg.norm(spots[s] - xy)))
        p["nearest_skill"] = sid
        p["nearest_skill_hit"] = p["per_skill_hit"][sid]


def _draw_pitch(ax, xlim, ylim, stripe_width: float = 0.2) -> None:
    ax.set_facecolor(_PITCH_GREEN)
    x = math.floor(xlim[0] / stripe_width) * stripe_width
    i = 0
    while x < xlim[1]:
        if i % 2 == 0:
            ax.axvspan(max(x, xlim[0]), min(x + stripe_width, xlim[1]), color=_PITCH_STRIPE, linewidth=0, zorder=0)
        x += stripe_width
        i += 1


def plot_heatmap(
    summary: dict, output_path: str, bins: int, dpi: int,
    hit_field: str = "any_hit", metric_label: str = "any-skill contact rate", tick_step: float = 0.1,
) -> None:
    """hit_field selects which per-placement boolean this heatmap grades: "nearest_skill_hit" (see
    tag_nearest_skill_hit), "any_hit" (any skill made contact), or "any_direction_hit" (any skill's
    contact also ended up on target).
    tick_step (2026-09-10): axis ticks every tick_step meters, not matplotlib's auto-picked spacing
    -- default auto-spacing landed on 0.2 m steps here, too coarse to read a placement's position
    off the axis against the bbox's own ~1 m span; --bins picks the DOT grid, this picks the RULER.
    The bins/placements/selector/overall-rate caption that used to run below the x-axis (using the
    now-removed kind_word/selector_desc params) was dropped 2026-09-10 on request -- that summary
    now belongs in the paper's own caption, not baked into the figure."""
    placements = summary["placements"]
    xs = np.array([p["x"] for p in placements])
    ys = np.array([p["y"] for p in placements])
    hits = np.array([1.0 if p[hit_field] else 0.0 for p in placements])
    overall_rate = float(np.mean(hits)) if len(hits) else float("nan")

    bbox_x, bbox_y = summary["bbox_x"], summary["bbox_y"]
    x_edges = np.linspace(bbox_x[0], bbox_x[1], bins + 1)
    y_edges = np.linspace(bbox_y[0], bbox_y[1], bins + 1)
    ix = np.clip(np.digitize(xs, x_edges) - 1, 0, bins - 1)
    iy = np.clip(np.digitize(ys, y_edges) - 1, 0, bins - 1)

    grid_sum = np.zeros((bins, bins))
    grid_n = np.zeros((bins, bins))
    for i, j, h in zip(iy, ix, hits):
        grid_sum[i, j] += h
        grid_n[i, j] += 1
    with np.errstate(invalid="ignore"):
        grid_rate = np.where(grid_n > 0, grid_sum / np.maximum(grid_n, 1), np.nan)

    cell_w = (bbox_x[1] - bbox_x[0]) / bins
    cell_h = (bbox_y[1] - bbox_y[0]) / bins
    radius = 0.36 * min(cell_w, cell_h)  # data units -- dots scale with the grid, not the dpi
    cx = bbox_x[0] + (np.arange(bins) + 0.5) * cell_w
    cy = bbox_y[0] + (np.arange(bins) + 0.5) * cell_h

    cmap = plt.get_cmap("RdYlGn")
    fig, ax = plt.subplots(figsize=(6.6, 6.6))

    empty = [Circle((cx[j], cy[i]), radius) for i in range(bins) for j in range(bins) if grid_n[i, j] == 0]
    if empty:
        ax.add_collection(PatchCollection(empty, facecolor="#e6e6e6", edgecolor="none", zorder=2))
    filled = [(Circle((cx[j], cy[i]), radius), grid_rate[i, j]) for i in range(bins) for j in range(bins) if grid_n[i, j] > 0]
    pc = PatchCollection([c for c, _ in filled], cmap=cmap, norm=Normalize(0.0, 1.0), edgecolor="none", zorder=3)
    pc.set_array(np.array([v for _, v in filled]))
    ax.add_collection(pc)

    ax.set_xlim(bbox_x)
    ax.set_ylim(bbox_y)
    ax.set_aspect("equal")
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(MultipleLocator(tick_step))
    ax.yaxis.set_major_locator(MultipleLocator(tick_step))
    ax.grid(True, color="#cfcfcf", linewidth=0.6)
    # fontsize=16 (2026-09-11, bumped again from an initial 14; originally matplotlib's plain
    # 10pt default) on request -- the axis titles and the colorbar label just below/above were
    # the two text elements the shared-dimension match with 73_plot_shot_trajectories.py's own
    # compact figure was tuned to (both bumped identically there too); tick numbers untouched.
    ax.set_xlabel("ball x (m)", fontsize=16)
    ax.set_ylabel("ball y (m)", fontsize=16)
    # Width-matched to the axes (2026-09-10), not fig.colorbar(..., location="top")'s own
    # auto-sizing -- that sizes off the axes' bbox BEFORE ax.set_aspect("equal")'s own
    # letterboxing (this bbox's x/y spans aren't quite equal), so it overhangs both ends of the
    # actually-visible plot box (same issue and same fix as 73_plot_shot_trajectories.py's own
    # compact colorbar). Drawing once first makes ax.get_position() reflect the real, post-squeeze
    # box to build the colorbar's own axes from.
    fig.canvas.draw()
    pos = ax.get_position()
    cax = fig.add_axes([pos.x0, pos.y1 + 0.02, pos.width, 0.035])
    cbar = fig.colorbar(pc, cax=cax, orientation="horizontal")
    cax.xaxis.set_ticks_position("top")
    cax.xaxis.set_label_position("top")
    cbar.set_label(metric_label, fontsize=16)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_path.rsplit(".", 1)[0] + ".pdf", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def truncate_hit_trajectory(xy: np.ndarray, speeds: list) -> tuple[np.ndarray, list]:
    """Cut a hit trajectory once ball speed has decayed to under 15% of its own post-contact peak
    -- the ball has settled/mostly stopped rolling by then. Different criterion from 73_plot_shot_
    trajectories.py's own truncate_at_closest_approach (which cuts at closest approach to an
    explicit commanded target point): this script's summary JSON (74_eval_ball_placement_sweep.py)
    doesn't carry each skill's own nominal_bearing_deg, so there is no target point to reconstruct
    here -- kick_aim_theta was forced to 0.0 for this whole sweep, but the DIRECTION itself still
    needs the checkpoint's own target_xy metadata, which this script deliberately avoids re-reading
    (74's own summary already carries everything needed without a second onnxruntime dependency
    here). Speed decay is a target-free proxy for "the graded event is over" that works from the
    trajectory data alone."""
    sp = np.array([s if s is not None else 0.0 for s in speeds])
    if len(sp) < 3:
        return xy, speeds
    peak_i = int(np.argmax(sp))
    threshold = 0.15 * sp[peak_i]
    end = len(sp)
    for k in range(peak_i, len(sp)):
        if sp[k] < threshold:
            end = k + 1
            break
    return xy[:end], speeds[:end]


def plot_trajectories(
    summary: dict, trajectories: dict[int, dict], output_path: str, dpi: int,
    hit_field: str = "per_skill_hit", title_word: str = "contacts",
) -> None:
    """hit_field selects which per-placement/per-skill dict gates which trials get drawn:
    "per_skill_hit" (every contact, including glancing non-directed touches) or
    "per_skill_direction_hit" (only contacts that also landed on target -- see this module's own
    docstring)."""
    placements = summary["placements"]
    hit_trials_by_skill: dict[int, list[int]] = {sid: [] for sid in trajectories}
    for i, p in enumerate(placements):
        for sid_str, hit in p[hit_field].items():
            if hit:
                hit_trials_by_skill[int(sid_str)].append(i)

    all_speeds = [
        s
        for sid, idxs in hit_trials_by_skill.items()
        for i in idxs
        for s in trajectories[sid]["trials"][i]["trajectory_speed"]
        if s is not None
    ]
    vmax = max(all_speeds) if all_speeds else 1.0
    norm = plt.Normalize(vmin=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(10, 9))
    all_pts_arr = []
    drawn = {sid: [] for sid in hit_trials_by_skill}
    for sid, idxs in hit_trials_by_skill.items():
        for i in idxs:
            t = trajectories[sid]["trials"][i]
            xy = np.asarray(t["trajectory_xy"], dtype=float)
            sp = t["trajectory_speed"]
            if xy.shape[0] < 2:
                continue
            xy_cut, sp_cut = truncate_hit_trajectory(xy, sp)
            drawn[sid].append((xy_cut, sp_cut))
            all_pts_arr.extend(xy_cut.tolist())

    pts = np.asarray(all_pts_arr) if all_pts_arr else np.zeros((1, 2))
    pad = 0.3
    xlim = (pts[:, 0].min() - pad, pts[:, 0].max() + pad)
    ylim = (pts[:, 1].min() - pad, pts[:, 1].max() + pad)
    _draw_pitch(ax, xlim, ylim)

    n_drawn = 0
    for sid, entries in drawn.items():
        for xy_cut, sp_cut in entries:
            sp_arr = np.array([s if s is not None else 0.0 for s in sp_cut])
            segments = np.stack([xy_cut[:-1], xy_cut[1:]], axis=1)
            lc = LineCollection(segments, cmap=_SPEED_CMAP, norm=norm, linewidth=1.0, alpha=0.75, zorder=3)
            lc.set_array(sp_arr[:-1])
            ax.add_collection(lc)
            ax.scatter(xy_cut[0, 0], xy_cut[0, 1], c=[sp_arr[0]], cmap=_SPEED_CMAP, norm=norm, s=14, edgecolors="white", linewidths=0.3, zorder=4)
            n_drawn += 1

    for sid_str, info in summary["per_skill"].items():
        bx, by = info["ball_xy"]
        ax.add_patch(Circle((bx, by), summary["trained_jitter_m"], facecolor="none", edgecolor="black", linewidth=1.0, linestyle="--", zorder=5))
        ax.annotate(f"skill_{sid_str}", (bx, by), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7, weight="bold", zorder=6)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("forward x (m)")
    ax.set_ylabel("lateral y (m)")
    ax.set_title(f"Shot trajectories across swept ball placements ({title_word}, {n_drawn} drawn)")
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=_SPEED_CMAP, norm=norm), ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("ball speed (m/s)")
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_path.rsplit(".", 1)[0] + ".pdf", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="74_eval_ball_placement_sweep.py's own ball_placement_sweep.json.")
    parser.add_argument("--output-prefix", default=None, help="Default: alongside --input, same basename.")
    parser.add_argument("--bins", type=int, default=12, help="Grid resolution for the heatmap (display binning only -- see module docstring).")
    parser.add_argument("--tick-step", type=float, default=0.1, help="Heatmap axis tick spacing in meters (see plot_heatmap's own docstring).")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    run_dir = os.path.dirname(os.path.abspath(args.input))
    summary = load_summary(args.input)
    skill_ids = [int(s) for s in summary["per_skill"].keys()]
    trajectories = load_trajectories(run_dir, skill_ids)

    prefix = args.output_prefix or os.path.join(run_dir, "ball_placement")

    tag_nearest_skill_hit(summary)
    plot_heatmap(
        summary, f"{prefix}_contact_heatmap.png", bins=args.bins, dpi=args.dpi,
        hit_field="nearest_skill_hit", metric_label="contact rate", tick_step=args.tick_step,
    )
    plot_trajectories(
        summary, trajectories, f"{prefix}_contact_trajectories.png", dpi=args.dpi,
        hit_field="per_skill_hit", title_word="ALL contacts, incl. non-directed touches",
    )
    written = [f"{prefix}_contact_heatmap.png", f"{prefix}_contact_trajectories.png"]

    has_direction_data = all(
        "any_direction_hit" in p and "per_skill_direction_hit" in p for p in summary["placements"]
    )
    if has_direction_data:
        plot_heatmap(
            summary, f"{prefix}_success_heatmap.png", bins=args.bins, dpi=args.dpi,
            hit_field="any_direction_hit", metric_label="any-skill directed-shot rate",
            tick_step=args.tick_step,
        )
        plot_trajectories(
            summary, trajectories, f"{prefix}_success_trajectories.png", dpi=args.dpi,
            hit_field="per_skill_direction_hit", title_word="direction-successful shots only",
        )
        written += [f"{prefix}_success_heatmap.png", f"{prefix}_success_trajectories.png"]
    else:
        logger.warning(
            "[75-plot-ball-placement-heatmap] --input has no any_direction_hit/per_skill_direction_hit "
            "fields (produced by a 74_eval_ball_placement_sweep.py from before 2026-09-10) -- skipping "
            "the direction-success-gated figures. Re-run 74 to get them."
        )
    logger.info(f"[75-plot-ball-placement-heatmap] wrote: {', '.join(written)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
