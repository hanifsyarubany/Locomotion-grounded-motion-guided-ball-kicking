#!/usr/bin/env python3
"""Renders RoboNaldo-style shot-trajectory figures (arXiv:2606.11092, their Fig. B/D) from
60_eval_shooting.py's own `--with-trajectories` output: ball paths on a pitch, colored by ball
speed along their length, with the ball's start position marked and the robot at the origin.

TWO FIGURES, not one -- and the reason is a real structural difference from RoboNaldo's own
setup, not a style preference. Theirs is a SINGLE forward-facing goal at 3 m, so every trajectory
in their paper fits one panel spanning about +/-1 m laterally. This project's library spans 234
degrees of nominal bearings (benchmark_plan.md's own skill inventory -- skill_016 at +138 deg and
skill_017 at -122 deg both point behind the robot's shoulder line), which that layout physically
cannot hold: one panel wide enough to contain it renders each skill's own detail down to a smear.

  1. `<out>_per_skill.png` -- one panel per skill, each framed to ITS OWN data, closest to a
     like-for-like reproduction of their Fig. B. Best for reading a single skill's dispersion.
  2. `<out>_combined.png` -- every skill's trajectories on ONE axes from the shared robot origin,
     which is the figure their single-forward-cone layout structurally cannot produce. This is the
     one that makes the angular-coverage claim visible in a single image instead of asking a
     reader to assemble it from six panels.

Both share ONE speed color scale (vmin/vmax pooled over every trajectory in the run), so a fast
skill and a slow one are directly comparable across panels rather than each being normalized to
its own range -- the same reason RoboNaldo's own figure carries a single shared colorbar.

THE "GOAL" ANALOG. RoboNaldo draw a literal goal because they have one fixed target. Ours is a
per-skill target POINT (that skill's nominal bearing at kick_aim_nominal_distance_m from the ball)
plus this project's own success radius (`success_radius_m`, 0.5 m by default -- the same threshold
Table 2/3's success rate is computed at). A circle at the success radius marks it; the per-skill
figure additionally marks the exact point with a plain X (its subplot title already names the
skill, so the circle needs no label of its own). The combined figure instead marks the point with
a small shaded badge carrying just the skill's NUMBER (2026-09-10) -- an offset text label reading
the full "skill_N" name, placed outside the circle, was the original approach, but it collides
into an unreadable run-together when two skills' bearings (and therefore targets) sit close
together (skill_5 at 48.4deg and skill_7 at 43.9deg landed 0.4m apart, `skill_5skill_7`). A tight
badge exactly at the target point scales down with the two targets' own separation instead of
needing room for the full label text, and stays legible over the busy trajectory fan behind it.
The commanded target actually MOVES per trial (each trial samples its own kick_aim_theta within
+/-theta_max), so the dashed wedge marks that full commanded range -- a trajectory ending inside
the circle is a success at the NOMINAL aim only; the honest per-trial success number is the one
60_eval_shooting.py already computes, not something to eyeball off this figure.

TRUNCATED AT CLOSEST APPROACH, BY DEFAULT. MuJoCo's floor is low-friction enough that a struck
ball keeps rolling for the remainder of the 8 s hold window -- measured on this project's own
data, paths run ~3x past their own 5 m target (skill_017's ball reached -14 m against a 4.7 m
target). Drawing that pushes every panel's framing out until the actual shot is a smudge near the
origin. Each path is therefore cut at its closest approach to that trial's OWN commanded target --
which is precisely the sample `min_target_dist` grades, so the drawn path is exactly the graded
shot, not an arbitrary visual crop. `--no-truncate` restores the full roll.

HITS ONLY, BY DEFAULT. A whiff has no shot trajectory -- its "path" is the ball sitting still at
its spawn point for the whole hold window, which renders as a meaningless dot pile that also drags
the speed color scale to zero. `--all-trials` includes them (they plot as that dot at the spawn
point); the panel title always reports how many trials were drawn out of how many ran, so an
excluded whiff is never silently invisible.

Usage (paste 60_eval_shooting.py's own logged run-folder path straight in):
    python scripts/73_plot_shot_trajectories.py --input out/eval_shooting/<run_timestamp>-<run_name>
    (requires that run to have been produced with --with-trajectories; a run recorded before
    2026-09-07 has trajectory_xy but no trajectory_speed, and is rejected with a clear error rather
    than silently plotted in a single flat color.)
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re

import matplotlib

matplotlib.use("Agg")  # headless -- this environment has no display
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Ellipse, Wedge
from matplotlib.ticker import MultipleLocator
from loguru import logger
from PIL import Image

TRAJECTORIES_SUBDIR = "trajectories"
EVAL_SHOOTING_FILENAME = "eval_shooting.json"

# RoboNaldo's own figure reads as a mown pitch under a plasma speed ramp; these approximate that
# without copying anything -- a mid-green ground with slightly lighter mowing stripes.
_PITCH_GREEN = "#4e7a3e"
_PITCH_STRIPE = "#568742"
_SPEED_CMAP = "plasma"


def resolve_run_dir(path: str) -> str:
    if not os.path.isdir(path):
        raise NotADirectoryError(f"{path!r} is not a directory -- pass 60_eval_shooting.py's own run folder.")
    if not os.path.isdir(os.path.join(path, TRAJECTORIES_SUBDIR)):
        raise FileNotFoundError(
            f"{path!r} has no {TRAJECTORIES_SUBDIR}/ subdirectory -- this run wasn't produced with "
            "--with-trajectories (see 60_eval_shooting.py's own module docstring)."
        )
    return path


def load_trajectories(run_dir: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in sorted(glob.glob(os.path.join(run_dir, TRAJECTORIES_SUBDIR, "*_trajectories.json"))):
        label = os.path.basename(path)[: -len("_trajectories.json")]
        with open(path) as f:
            out[label] = json.load(f)
    return out


def load_geometry(run_dir: str) -> dict[str, dict]:
    path = os.path.join(run_dir, EVAL_SHOOTING_FILENAME)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    return {s["label"]: s for s in data.get("skills", [])}


def select_trials(traj: dict, hits_only: bool) -> list[dict]:
    trials = traj["trials"]
    if hits_only:
        return [t for t in trials if t["hit_step"] != -1]
    return trials


def _require_speed(trajectories: dict[str, dict]) -> None:
    """A run recorded before trajectory_speed existed would otherwise plot every path in one flat
    color with a meaningless colorbar -- fail loudly instead, naming the fix."""
    for label, traj in trajectories.items():
        for t in traj["trials"]:
            if "trajectory_speed" not in t:
                raise KeyError(
                    f"{label}'s trajectory records have no 'trajectory_speed' field -- this run "
                    "predates per-tick speed logging (2026-09-07). Re-run 60_eval_shooting.py "
                    "--with-trajectories to record it."
                )
            return  # one trial is enough to establish the schema


def speed_limits(trajectories: dict[str, dict], hits_only: bool) -> tuple[float, float]:
    speeds = [
        s
        for traj in trajectories.values()
        for t in select_trials(traj, hits_only)
        for s in t["trajectory_speed"]
        if s is not None
    ]
    if not speeds:
        return 0.0, 1.0
    return 0.0, float(max(speeds))


def _draw_pitch(ax, xlim: tuple[float, float], ylim: tuple[float, float], stripe_width: float = 1.0) -> None:
    ax.set_facecolor(_PITCH_GREEN)
    x = math.floor(xlim[0] / stripe_width) * stripe_width
    i = 0
    while x < xlim[1]:
        if i % 2 == 0:
            ax.axvspan(max(x, xlim[0]), min(x + stripe_width, xlim[1]), color=_PITCH_STRIPE, linewidth=0, zorder=0)
        x += stripe_width
        i += 1


def trial_target_xy(geo: dict | None, start_xy: np.ndarray, kick_aim_theta: float | None) -> np.ndarray | None:
    """This trial's OWN commanded target point, reconstructed exactly the way
    mujoco_kick_survival_scan.py's own run() builds it: ball spawn + D * unit(nominal_bearing +
    kick_aim_theta). Not stored in the trajectory JSON (it's derivable, and duplicating it would
    let the two drift), so it's rebuilt here from the same two inputs the worker used."""
    if geo is None or kick_aim_theta is None:
        return None
    rad = math.radians(geo["nominal_bearing_deg"] + kick_aim_theta)
    return start_xy + geo["kick_aim_nominal_distance_m"] * np.array([math.cos(rad), math.sin(rad)])


def truncate_at_closest_approach(xy: np.ndarray, target: np.ndarray | None) -> np.ndarray:
    """Cut the path at its closest approach to `target` -- the exact sample min_target_dist grades
    (see mujoco_kick_survival_scan.py's own "closest approach so far" contract). Everything after
    that is post-scoring roll: MuJoCo's floor is low-friction enough that a struck ball keeps
    going for the rest of the 8 s hold window, which on these axes reads as a 15 m ray shooting
    past a 5 m target and pushes every panel's framing out to where the actual shot is a smudge
    near the origin. Truncating shows the graded shot at a readable scale instead."""
    if target is None or xy.shape[0] < 2:
        return xy
    dists = np.linalg.norm(xy - target, axis=1)
    return xy[: int(np.argmin(dists)) + 1]


def _draw_trials(
    ax, trials: list[dict], norm, stride: int, linewidth: float,
    geo: dict | None = None, truncate: bool = True,
    color_by: str = "speed", cmap: str = _SPEED_CMAP,
) -> None:
    """One LineCollection per trial. color_by="speed" (default) colors PER SEGMENT by that
    segment's own ball speed -- the color varies ALONG each path (a ball decelerating as it rolls
    fades down the ramp), which is what carries the speed information a single per-trial color
    would flatten away. color_by="kick_aim_theta" (2026-09-10, for the kick-aim variant) instead
    paints the WHOLE trial one solid color from that trial's own single commanded theta value --
    unlike speed, theta doesn't vary along a trial's path, so a per-segment gradient would just be
    a constant-color line drawn the hard way; this is that same fact made explicit rather than
    feeding a single repeated value through the per-segment machinery anyway."""
    for t in trials:
        xy_full = np.asarray(t["trajectory_xy"], dtype=float)
        sp_full = np.asarray([s if s is not None else 0.0 for s in t["trajectory_speed"]], dtype=float)
        if truncate and xy_full.shape[0] >= 2:
            target = trial_target_xy(geo, xy_full[0], t.get("kick_aim_theta"))
            n_keep = truncate_at_closest_approach(xy_full, target).shape[0]
            xy_full, sp_full = xy_full[:n_keep], sp_full[:n_keep]
        xy = xy_full[::stride]
        sp = sp_full[::stride]
        if xy.shape[0] < 2:
            continue
        if color_by == "kick_aim_theta":
            theta = t.get("kick_aim_theta")
            values = np.full(xy.shape[0] - 1, theta if theta is not None else 0.0)
        else:
            values = sp[:-1]
        segments = np.stack([xy[:-1], xy[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=linewidth, alpha=0.85, zorder=3)
        lc.set_array(values)
        ax.add_collection(lc)
        # Ball start position, colored the same way -- RoboNaldo mark these too (their scattered
        # dots), and here they also make the per-trial spawn jitter visible.
        ax.scatter(
            xy[0, 0], xy[0, 1], c=[values[0]], cmap=cmap, norm=norm,
            s=22, edgecolors="white", linewidths=0.4, zorder=4,
        )


def _skill_number(label: str) -> str:
    """"skill_1" -> "1", "skill_012" -> "12" (leading zeros stripped -- a compact badge has no room
    for them and this run's own labels are never ambiguous without them). Falls back to the label
    unchanged if it doesn't end in digits, rather than raising over a badge's own cosmetics."""
    m = re.search(r"(\d+)$", label)
    return str(int(m.group(1))) if m else label


def _draw_target(
    ax, geo: dict, origin: np.ndarray, success_radius_m: float,
    label: str | None = None, badge_radius_m: float | None = None,
    badge_fontsize: float = 8.0, linewidth_scale: float = 1.0,
) -> None:
    """The per-skill analog of RoboNaldo's goal -- see this module's own docstring's THE "GOAL"
    ANALOG section for the X-vs-badge choice `label` switches between. badge_radius_m is in DATA
    units but should be computed from the target figure's own points-per-data-unit scale (see
    plot_combined's own call site) so the badge reads as a fixed ~9pt circle regardless of this
    run's D/success_radius_m -- a badge sized as a fraction of success_radius_m would shrink below
    legible with a tight --success-radius-m override, or bloat past the circle with a loose one.
    badge_fontsize/linewidth_scale (2026-09-10): plot_combined's own `compact` mode uses these to
    make the badge digit and the dashed aim lines survive a large print-time shrink -- see that
    function's own compact docstring section for why line widths and font sizes tuned to look
    right in an on-screen preview don't survive being placed at quarter-page width in a paper."""
    D = geo["kick_aim_nominal_distance_m"]
    nominal = geo["nominal_bearing_deg"]
    band_lo, band_hi = geo["reachable_band_deg"]
    # Commanded-aim wedge: where a trial's own target point can land across the theta sweep.
    # zorder=8 (2026-09-10) puts these above EVERYTHING else this function and _draw_trials draw
    # (trajectory lines at 3, start-point dots at 4, badges at 6/7) -- at zorder=2 the dense
    # trajectory fan drawn afterward buried these dashed lines exactly where they're most needed
    # (crossing the busy interior near the origin).
    ax.add_patch(Wedge(
        tuple(origin), D, band_lo, band_hi, width=0.0,
        facecolor="none", edgecolor="white", linestyle="--", linewidth=1.2 * linewidth_scale, alpha=0.55, zorder=8,
    ))
    for deg in (band_lo, band_hi):
        rad = math.radians(deg)
        ax.plot(
            [origin[0], origin[0] + D * math.cos(rad)], [origin[1], origin[1] + D * math.sin(rad)],
            color="white", linestyle="--", linewidth=1.2 * linewidth_scale, alpha=0.55, zorder=8,
        )
    rad = math.radians(nominal)
    target = origin + D * np.array([math.cos(rad), math.sin(rad)])
    # Nominal-bearing line (2026-09-10): the band edges above mark where a trial's target CAN
    # land, but nothing previously drew the skill's own trained/reference direction -- the single
    # bearing the band is centered on. Thicker and more opaque than the band edges so it reads as
    # the primary aim line, not a third indistinguishable dashed edge.
    ax.plot(
        [origin[0], target[0]], [origin[1], target[1]],
        color="white", linestyle="--", linewidth=1.8 * linewidth_scale, alpha=0.8, zorder=8,
    )
    if label is None:
        # Success-radius circle -- skipped for the badge case below: with a badge already marking
        # the point, the translucent circle mostly just added visual clutter over the trajectory
        # fan without carrying information the badge doesn't already convey more directly.
        ax.add_patch(Circle(
            tuple(target), success_radius_m, facecolor="white", alpha=0.18, edgecolor="white",
            linewidth=1.0, zorder=2,
        ))
        ax.scatter(*target, marker="x", color="black", s=70, linewidths=2.0, zorder=5)
        return
    badge_r = badge_radius_m if badge_radius_m is not None else 0.3 * success_radius_m
    # zorder 9/10 (2026-09-10): the skill tag is the single most important label on this figure,
    # so it sits above even the white aim lines above (zorder=8) -- those lines terminate exactly
    # at this badge's own center and were otherwise drawn AFTER it, cutting visibly across the
    # number.
    ax.add_patch(Circle(
        tuple(target), badge_r, facecolor="#003366", edgecolor="white", linewidth=1.0, zorder=9,
    ))
    ax.annotate(
        _skill_number(label), tuple(target), ha="center", va="center",
        color="white", fontsize=badge_fontsize, weight="bold", zorder=10,
    )


def _draw_robot(ax, fontsize: float = 7.0, size_scale: float = 1.0, bold: bool = False) -> None:
    ax.add_patch(Ellipse(
        (0.0, 0.0), 0.34 * size_scale, 0.22 * size_scale,
        facecolor="#141b2e", edgecolor="white", linewidth=0.8, zorder=6,
    ))
    ax.annotate(
        "robot", (0.0, 0.0), textcoords="offset points", xytext=(0, 2 * size_scale),
        ha="center", va="bottom", color="white", fontsize=fontsize,
        weight="bold" if bold else "normal", zorder=6,
    )


def _bounds(points: np.ndarray, extra: list[np.ndarray], pad_frac: float = 0.08) -> tuple[tuple, tuple]:
    pts = np.concatenate([points] + [e.reshape(-1, 2) for e in extra if e.size], axis=0)
    pad_x = max(0.3, pad_frac * np.ptp(pts[:, 0]))
    pad_y = max(0.3, pad_frac * np.ptp(pts[:, 1]))
    return (
        (pts[:, 0].min() - pad_x, pts[:, 0].max() + pad_x),
        (pts[:, 1].min() - pad_y, pts[:, 1].max() + pad_y),
    )


def truncated_paths(trials: list[dict], geo: dict | None, truncate: bool) -> list[np.ndarray]:
    """The (possibly truncated) xy path per trial -- shared by the drawing code and the axis-bounds
    computation so the framing matches exactly what actually gets drawn."""
    out = []
    for t in trials:
        xy = np.asarray(t["trajectory_xy"], dtype=float)
        if truncate and xy.shape[0] >= 2:
            xy = truncate_at_closest_approach(xy, trial_target_xy(geo, xy[0], t.get("kick_aim_theta")))
        out.append(xy)
    return out


def _origin_of(trials: list[dict]) -> np.ndarray:
    """Mean ball start across trials -- the trajectory JSON has no explicit spawn field, and the
    per-trial jitter this averages over is at most a few cm (SkillConfig.randomize_x/y)."""
    starts = [t["trajectory_xy"][0] for t in trials if t["trajectory_xy"]]
    return np.asarray(starts, dtype=float).mean(axis=0) if starts else np.zeros(2)


def plot_per_skill(
    trajectories: dict[str, dict], geometry: dict[str, dict], output_path: str,
    *, hits_only: bool, stride: int, dpi: int, success_radius_m: float, truncate: bool,
) -> None:
    labels = list(trajectories.keys())
    ncols = min(3, len(labels))
    nrows = math.ceil(len(labels) / ncols)
    vmin, vmax = speed_limits(trajectories, hits_only)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.9 * nrows), squeeze=False)
    for idx, label in enumerate(labels):
        ax = axes[idx // ncols][idx % ncols]
        traj = trajectories[label]
        trials = select_trials(traj, hits_only)
        geo = geometry.get(label)
        origin = _origin_of(traj["trials"])

        paths = truncated_paths(trials, geo, truncate)
        pts = np.concatenate(paths, axis=0) if paths and any(p.size for p in paths) else np.zeros((1, 2))
        extra = [np.zeros((1, 2)), origin.reshape(1, 2)]
        if geo is not None:
            rad = math.radians(geo["nominal_bearing_deg"])
            extra.append((origin + geo["kick_aim_nominal_distance_m"] * np.array([math.cos(rad), math.sin(rad)])).reshape(1, 2))
        xlim, ylim = _bounds(pts, extra)

        _draw_pitch(ax, xlim, ylim)
        if geo is not None:
            _draw_target(ax, geo, origin, success_radius_m)
        _draw_trials(ax, trials, norm, stride, linewidth=1.1, geo=geo, truncate=truncate)
        _draw_robot(ax)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_title(f"{label}   ({len(trials)}/{len(traj['trials'])} shots drawn)", fontsize=10)
        ax.set_xlabel("forward x (m)", fontsize=8)
        ax.set_ylabel("lateral y (m)", fontsize=8)
        ax.tick_params(labelsize=7)

    for idx in range(len(labels), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    sm = plt.cm.ScalarMappable(cmap=_SPEED_CMAP, norm=norm)
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    cbar.set_label("ball speed (m/s)", fontsize=9)
    fig.suptitle("Shot trajectories per skill" + ("  (hits only)" if hits_only else "  (all trials)"), fontsize=12)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_path.rsplit(".", 1)[0] + ".pdf", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_combined(
    trajectories: dict[str, dict], geometry: dict[str, dict], output_path: str,
    *, hits_only: bool, stride: int, dpi: int, success_radius_m: float, truncate: bool,
    compact: bool = False, color_by: str = "speed",
) -> None:
    """compact (2026-09-10): a variant tuned to survive being placed at ~quarter-page width in a
    paper, not just to look right in an on-screen preview -- confirmed by literally rendering the
    non-compact figure at 1.75in/300dpi and viewing the result: the trajectory fan's SHAPE
    survived, but the title, axis ticks, colorbar, and (most importantly) every skill badge's own
    digit were all illegible smudges. Uses a 6.6x6.6in figure (2026-09-10, was 9x9) and matplotlib's
    plain default (10pt) for tick/axis/colorbar labels -- matching 75_plot_ball_placement_
    heatmap.py's own plot_heatmap figsize and fonts EXACTLY, on request, so the two figure types
    sit at the same physical size/scale in the paper. The elements that figure has no equivalent
    of (skill badges, aim lines, robot marker) stay oversized/thickened on their own terms -- badge
    digits, aim-line width, and robot label were independently confirmed legible at 1.75in/300dpi
    (a bigger relative margin now than when this was checked at 9in, since 6.6in needs LESS
    shrinking to reach the same print width).

    color_by (2026-09-10): "speed" (default, data-driven vmin/vmax -- see speed_limits) or
    "kick_aim_theta" (FIXED [-15, 15] deg range regardless of what this run's own
    --kick-aim-theta-max-deg was, since the request this exists for was an explicit, comparable-
    across-runs +/-15 scale, not this run's own sampled extent -- confirmed this run's own trials
    land within [-14.99, 14.92], safely inside that fixed range). Both share the SAME _SPEED_CMAP
    (plasma) palette (2026-09-10 -- an earlier version used a diverging coolwarm for theta, on the
    reasoning that a signed, zero-centered quantity reads better diverging; changed on request to
    match the speed figure's own palette so the two compact figures read as one consistent family
    rather than two different color languages). A trial's theta doesn't vary along its own path
    (unlike speed), so each whole trial gets ONE solid color -- see _draw_trials' own color_by
    docstring."""
    if color_by == "kick_aim_theta":
        norm = plt.Normalize(vmin=-15.0, vmax=15.0)
        cmap = _SPEED_CMAP
    else:
        vmin, vmax = speed_limits(trajectories, hits_only)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cmap = _SPEED_CMAP
    # compact uses 6.6x6.6in (2026-09-10, was 9x9) -- matches 75_plot_ball_placement_heatmap.py's
    # own plot_heatmap figsize exactly, on request, so the two figures sit at the same physical
    # size/scale when placed together in the paper.
    fig, ax = plt.subplots(figsize=(6.6, 6.6) if compact else (9, 9))

    all_pts, extra, total_drawn, total_ran = [], [np.zeros((1, 2))], 0, 0
    for label, traj in trajectories.items():
        trials = select_trials(traj, hits_only)
        total_drawn += len(trials)
        total_ran += len(traj["trials"])
        geo = geometry.get(label)
        for path in truncated_paths(trials, geo, truncate):
            if path.size:
                all_pts.extend(path.tolist())
        if geo is not None:
            origin = _origin_of(traj["trials"])
            rad = math.radians(geo["nominal_bearing_deg"])
            extra.append((origin + geo["kick_aim_nominal_distance_m"] * np.array([math.cos(rad), math.sin(rad)])).reshape(1, 2))
    pts = np.asarray(all_pts, dtype=float) if all_pts else np.zeros((1, 2))
    # pad_frac=0.03 (tighter than _bounds' own 0.08 default, 2026-09-10): a compact, zoomed-in
    # frame hugging the trajectory fan + badges, not the wide margin of pitch the default leaves.
    xlim, ylim = _bounds(pts, extra, pad_frac=0.03)
    span = max(xlim[1] - xlim[0], ylim[1] - ylim[0])
    # A fixed FRACTION of the plotted span, not a fixed meter value or a fraction of
    # success_radius_m: scales down automatically when the framing is tight (many skills, small D)
    # and up when it's wide, so the badge stays a legible, non-overlapping size across runs with
    # very different D/success_radius_m/skill-count without hand-tuning per run. compact roughly
    # doubles it -- the single biggest legibility win for a quarter-page render (see docstring).
    badge_radius_m = (0.030 if compact else 0.014) * span
    badge_fontsize = 15.0 if compact else 8.0
    linewidth_scale = 1.8 if compact else 1.0
    traj_linewidth = 1.6 if compact else 0.9

    _draw_pitch(ax, xlim, ylim)
    for label, traj in trajectories.items():
        trials = select_trials(traj, hits_only)
        geo = geometry.get(label)
        origin = _origin_of(traj["trials"])
        if geo is not None:
            _draw_target(
                ax, geo, origin, success_radius_m, label=label, badge_radius_m=badge_radius_m,
                badge_fontsize=badge_fontsize, linewidth_scale=linewidth_scale,
            )
        _draw_trials(ax, trials, norm, stride, linewidth=traj_linewidth, geo=geo, truncate=truncate, color_by=color_by, cmap=cmap)
    _draw_robot(ax, fontsize=17.0 if compact else 7.0, size_scale=1.5 if compact else 1.0, bold=compact)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    if compact:
        # Tick label size and colorbar thickness match 75_plot_ball_placement_heatmap.py's own
        # plot_heatmap EXACTLY (2026-09-10 -- was an oversized 13pt tuned to survive a 1.75in/
        # 300dpi print-time shrink; that figure used matplotlib's plain default, 10pt, so 10pt
        # here too). Axis-title and colorbar-title fontsize bumped to 14 (2026-09-11, on request,
        # matched identically in plot_heatmap) -- tick NUMBERS stay at 10, only the two titles.
        ax.tick_params(axis="both", labelsize=10, length=4)
        ax.set_xlabel("forward x (m)", fontsize=16)
        ax.set_ylabel("lateral y (m)", fontsize=16)
        # Horizontal, on top, WIDTH-MATCHED to the axes (2026-09-10) -- same placement AND same
        # pad/thickness (0.02/0.035) this project's own ball-placement heatmap
        # (75_plot_ball_placement_heatmap.py) uses, on request, so the two colorbars read as the
        # same visual weight when the figures sit side by side. fig.colorbar(..., ax=ax,
        # location="top")'s own auto-sizing was measured at (2026-09-10 sanity check) 1392px wide
        # against a 1240px-wide axes box -- ~6% wider AND centered the same, i.e. overhanging both
        # ends -- because it sizes off the axes' bbox BEFORE ax.set_aspect("equal")'s own
        # letterboxing (this run's xlim/ylim spans aren't quite equal), and location="top" doesn't
        # re-measure after that squeeze. A draw() first, then building the colorbar's own axes
        # directly from ax.get_position() (which DOES reflect the post-squeeze box), makes the two
        # ends land exactly together instead of guessing a fraction/shrink that happens to cancel out.
        fig.canvas.draw()
        pos = ax.get_position()
        cax = fig.add_axes([pos.x0, pos.y1 + 0.02, pos.width, 0.035])
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, orientation="horizontal")
        cax.xaxis.set_ticks_position("top")
        cax.xaxis.set_label_position("top")
        cbar.set_label("kick aim theta (deg)" if color_by == "kick_aim_theta" else "ball speed (m/s)", fontsize=16)
        cbar.ax.tick_params(labelsize=10)
    else:
        ax.set_xlabel("forward x (m)")
        ax.set_ylabel("lateral y (m)")
        ax.set_title(
            f"Shot trajectories, all skills from one robot origin  ({total_drawn}/{total_ran} shots drawn"
            + (", hits only)" if hits_only else ", all trials)"),
            fontsize=12,
        )
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("kick aim theta (deg)" if color_by == "kick_aim_theta" else "ball speed (m/s)")
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_path.rsplit(".", 1)[0] + ".pdf", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def fit_canvas_to(png_path: str, reference_png_path: str) -> tuple[int, int]:
    """Pad/crop png_path (in place) to exactly reference_png_path's own pixel dimensions.

    WHY THIS EXISTS: two figures rendered with matching figsize+dpi (e.g. this script's own
    compact=True output and 75_plot_ball_placement_heatmap.py's plot_heatmap, both 6.6x6.6in) do
    NOT automatically end up pixel-identical once both are saved with bbox_inches="tight" -- that
    option crops each one down to ITS OWN content bounding box, and how far that trims depends on
    how far THAT figure's own tick/axis labels happen to extend past the (otherwise identical)
    plot box -- e.g. "lateral y (m)" + "-3".."6" vs "ball y (m)" + "0.5".."-0.4" occupy different
    pixel widths. Measured concretely (2026-09-10): both figures' own axes boxes (spine-to-spine)
    already land within ~5px of each other, so the mismatch is confined entirely to the outer
    margin, not the plotted content -- padding/cropping just that margin is exact and lossless.

    Width: centered pad (if narrower) or centered crop (if wider) -- the plot content is never off
    -center either way. Height: pad/crop from the BOTTOM only, never the top -- the compact
    layout's colorbar sits flush against the top edge (see plot_combined's own cax placement), so
    trimming height from the top risks cutting into it; the bottom margin (x-axis label) has the
    same one-or-two-pixel slack the top does not."""
    im = Image.open(png_path).convert("RGB")
    ref_w, ref_h = Image.open(reference_png_path).size
    w, h = im.size
    bg = im.getpixel((0, 0))  # matplotlib's own white figure background, sampled rather than assumed

    if w < ref_w:
        canvas = Image.new("RGB", (ref_w, h), bg)
        canvas.paste(im, ((ref_w - w) // 2, 0))
        im = canvas
    elif w > ref_w:
        left = (w - ref_w) // 2
        im = im.crop((left, 0, left + ref_w, h))
    w, h = im.size

    if h < ref_h:
        canvas = Image.new("RGB", (w, ref_h), bg)
        canvas.paste(im, (0, 0))
        im = canvas
    elif h > ref_h:
        im = im.crop((0, 0, w, ref_h))

    im.save(png_path)
    return im.size


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input", required=True,
        help="60_eval_shooting.py's own run folder (the one --with-trajectories was passed for).",
    )
    parser.add_argument(
        "--output-prefix", default=None,
        help="Output path prefix; '<prefix>_per_skill.png' and '<prefix>_combined.png' are written "
        "(each with a .pdf twin). Default: 'shot_trajectories' inside the run folder.",
    )
    parser.add_argument("--all-trials", dest="hits_only", action="store_false", default=True,
                        help="Include whiffs (hit_step == -1). Default: hits only -- see module docstring.")
    parser.add_argument("--stride", type=int, default=1,
                        help="Plot every Nth recorded tick (default 1 = all). Raise it if rendering a "
                        "large sweep gets slow; it only thins the drawn polyline, never the data.")
    parser.add_argument("--no-truncate", dest="truncate", action="store_false", default=True,
                        help="Draw the FULL recorded path instead of cutting it at the closest "
                        "approach to that trial's own commanded target. Default is to truncate -- "
                        "see truncate_at_closest_approach's own docstring for why (the ball keeps "
                        "rolling for the rest of the 8s hold window, so untruncated paths run "
                        "~3x past the target and force the framing out to where the actual shot "
                        "is unreadable).")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--compact-combined", action="store_true",
        help="ALSO write '<prefix>_combined_compact.png/.pdf' -- plot_combined's own compact=True "
        "variant, tuned to survive being placed at ~quarter-page width in a paper (no title/"
        "colorbar/tick numbers, oversized badges and lines) instead of the full '_combined' figure, "
        "which was confirmed by literal 1.75in/300dpi rendering to lose every badge digit and all "
        "text at that size. See plot_combined's own compact docstring section.",
    )
    parser.add_argument(
        "--kick-aim-combined", action="store_true",
        help="ALSO write '<prefix>_combined_compact_kick_aim.png/.pdf' -- the same compact=True "
        "layout as --compact-combined, but each trial's line is one solid color from its own "
        "kick_aim_theta (fixed [-15, 15] deg colorbar, diverging coolwarm) instead of a per-tick "
        "speed gradient. A SEPARATE file -- does not touch '_combined_compact' even if both flags "
        "are given. See plot_combined's own color_by docstring section.",
    )
    parser.add_argument(
        "--match-size-to", default=None, metavar="REFERENCE_PNG",
        help="Pad/crop '<prefix>_combined_compact.png' (only -- not the kick-aim variant) in place "
        "to REFERENCE_PNG's own exact pixel dimensions after writing it, e.g. "
        "--match-size-to out/ball_placement_sweep/<run>/ball_placement_contact_heatmap.png so the "
        "two figures land at IDENTICAL width/height for a paper. Requires --compact-combined. See "
        "fit_canvas_to's own docstring for why same figsize+dpi alone isn't already enough.",
    )
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.input)
    trajectories = load_trajectories(run_dir)
    if not trajectories:
        logger.error(f"[73-plot-shot-trajectories] {run_dir!r}'s trajectories/ folder is empty -- nothing to plot.")
        return 1
    _require_speed(trajectories)
    geometry = load_geometry(run_dir)
    # success_radius_m is a property of the RUN (60_eval_shooting.py's own config), identical
    # across skills, so read it from any one skill entry rather than re-declaring a default here.
    success_radius_m = next(iter(geometry.values()))["success_radius_m"] if geometry else 0.5

    prefix = args.output_prefix or os.path.join(run_dir, "shot_trajectories")
    per_skill_path = f"{prefix}_per_skill.png"
    combined_path = f"{prefix}_combined.png"
    kwargs = dict(
        hits_only=args.hits_only, stride=args.stride, dpi=args.dpi,
        success_radius_m=success_radius_m, truncate=args.truncate,
    )
    plot_per_skill(trajectories, geometry, per_skill_path, **kwargs)
    plot_combined(trajectories, geometry, combined_path, **kwargs)
    written = [per_skill_path, combined_path]
    if args.compact_combined:
        compact_path = f"{prefix}_combined_compact.png"
        plot_combined(trajectories, geometry, compact_path, compact=True, **kwargs)
        written.append(compact_path)
        if args.match_size_to:
            new_size = fit_canvas_to(compact_path, args.match_size_to)
            logger.info(f"[73-plot-shot-trajectories] fit {compact_path} to {new_size[0]}x{new_size[1]} (matching {args.match_size_to})")
    if args.kick_aim_combined:
        kick_aim_path = f"{prefix}_combined_compact_kick_aim.png"
        plot_combined(trajectories, geometry, kick_aim_path, compact=True, color_by="kick_aim_theta", **kwargs)
        written.append(kick_aim_path)
    logger.info(f"[73-plot-shot-trajectories] wrote {', '.join(written)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
