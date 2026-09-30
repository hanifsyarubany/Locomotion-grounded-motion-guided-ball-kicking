#!/usr/bin/env python3
"""Renders F2 (documents/proposal/benchmark_plan.md, §3 -- "the money figure") from
`60_eval_shooting.py`'s own output JSON: a full 0-360-degree polar plot, one lobe per skill,
radial = measured success rate per angle bin. Reusable for a single skill (one-entry `skills`
list -- a specialist checkpoint in isolation) or the full library (a six-entry list) with NO
special-casing -- the plot just draws whatever's in the input JSON.

MAGNITUDE CHOICE (2026-09-06 design discussion): radial = `success_rate` (the strict,
unconditional-on-hit, num_trials-denominator statistic 60_eval_shooting.py already computes per
bin -- RoboNaldo-comparable, matching Table 2's own success@0.5m), NOT `hit_rate`. A real example
from this project's own data motivates why the distinction matters: one smoke-tested checkpoint
(skill_012/model_0400000) measured hit_rate=1.0 in every bin but success_rate=0.0 in every bin --
plotting hit_rate would draw a big, healthy-looking, fully-covered lobe for a checkpoint that
reliably kicks the ball without aiming it anywhere near the target. hit_rate is still drawn, but
only as a thin, desaturated secondary reference (see --show-hit-rate), never sized/colored the
same as the primary success curve, so it can't be mistaken for it at a glance.

SMALL-n HANDLING: a bin's `success_rate` is a bare fraction with no built-in indication of how
many trials it's actually averaged over -- a bin with n=1 reports a coin-flip 0% or 100%,
visually indistinguishable from an n=40 bin unless something encodes the difference. This script
encodes trial count as BOTH marker size and per-point alpha (`--min-n-for-full-confidence`,
default 15 -- a bin at or above this count draws at full opacity/size; below it, scaled down
linearly; n=0 bins are skipped from the line entirely, not drawn as a fabricated 0). This is a
visual backstop, not a substitute for sizing num_trials/num_bins sensibly in the eval config
itself (60_eval_shooting.py's own module docstring has the sizing guidance) -- see this project's
own measurement-protocol rule (benchmark_plan.md §5) against single-run/small-n results standing
in for a real measurement.

GAPS ARE NEVER INTERPOLATED ACROSS: matplotlib will happily draw a straight line across a `nan`
gap in a data array unless told not to -- this script explicitly breaks the line at every empty
bin (n=0) AND leaves the angular space between skills' own reachable bands completely blank (no
line, no point) rather than connecting one skill's edge to the next skill's edge, which would
visually manufacture continuity across a real, meaningful gap (see benchmark_plan.md §1: "Do not
describe coverage as continuous"). The light background wedge per skill (its own geometric
`reachable_band_deg`, read from the checkpoint's own metadata by 60_eval_shooting.py, not
retyped) marks where a lobe COULD exist independent of whether it was actually measured to be
large there -- a lobe shrinking well inside its own shaded wedge is a real, visible finding
(the policy can't hit what it's nominally allowed to aim at), not a plotting artifact.

NOT drawn: a RoboNaldo reference arc on these same axes. benchmark_plan.md's own Table 2 note is
explicit that RoboNaldo's forward-cone shot-placement accuracy and this project's per-skill
strike-direction coverage are "not the same kind of quantity," and overlaying an arc of
comparable visual size on the same polar axes would invite exactly the magnitude-comparison
reviewers are warned against inviting. Cite RoboNaldo's number in the caption/prose instead.

Usage (paste 60_eval_shooting.py's own logged run-folder path straight in -- no filename needed,
no --output needed either; the plot lands as f2_polar_coverage.png/.pdf in that SAME folder):
    python scripts/71_plot_polar_coverage.py --input out/eval_shooting/<run_timestamp>-<run_name>

Usage (explicit file + output path, still supported):
    python scripts/71_plot_polar_coverage.py --input out/eval_shooting/library.json \\
        --output out/figures/f2_polar_coverage.png
    (a .pdf twin is always written alongside whatever --output ends in .png, for direct LaTeX
    inclusion)
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os

import matplotlib

matplotlib.use("Agg")  # headless -- this environment has no display
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

# tab10 alone (10 colors) SILENTLY WRAPS AND REPEATS once a sweep passes 10 skills -- confirmed
# live, 2026-09-06: a 12-target sweep (two 6-skill checkpoints compared side by side) put
# run0905_600k_skill_019/_020 in the exact same colors as run0904_400k_skill_012/_016, making
# them visually indistinguishable in both the lobes and the legend. tab10 + tab20b + tab20c (all
# matplotlib built-ins, no extra dependency) gives 50 distinct colors -- the first 10 are BYTE-
# IDENTICAL to plain tab10, so every plot with <=10 skills already reviewed is visually unchanged.
_PALETTE = list(plt.get_cmap("tab10").colors) + list(plt.get_cmap("tab20b").colors) + list(plt.get_cmap("tab20c").colors)


# --------------------------------------------------------------------------------------------- #
# Data loading + small pure-geometry helpers (unit-testable without matplotlib)
# --------------------------------------------------------------------------------------------- #


EVAL_SHOOTING_FILENAME = "eval_shooting.json"  # the exact name 60_eval_shooting.py's write_output always uses


def resolve_input_json(path: str) -> str:
    """`path` may be either a direct JSON file (backward compatible) or a RUN FOLDER -- e.g. the
    `<output>/<run_timestamp>-<run_name>/` directory 60_eval_shooting.py's own --output produces
    -- in which case this resolves to `<path>/eval_shooting.json`, that script's own fixed output
    filename. Lets a caller paste that one folder path straight from 60_eval_shooting.py's own
    "wrote ... -> ..." log line without editing it into a file path first."""
    if os.path.isdir(path):
        candidate = os.path.join(path, EVAL_SHOOTING_FILENAME)
        if not os.path.exists(candidate):
            raise FileNotFoundError(
                f"{path!r} is a directory but has no {EVAL_SHOOTING_FILENAME} inside it -- pass "
                "60_eval_shooting.py's own run folder (or a direct path to its JSON output)."
            )
        return candidate
    return path


def load_eval_shooting_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def merge_intervals_deg(bands: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Standard LINEAR interval merge (sort by lo, merge overlapping/touching) -- NOT circular-
    aware. Correct for this project's actual six-skill library today (no band straddles the
    +/-180 degree wrap point -- verified 2026-09-06 against every skill's own real geometry), but
    would under-merge two bands that genuinely wrap around 0/360 if a future skill's reachable
    band ever crosses that boundary. Documented here rather than silently handled, since a wrong
    "looks merged when it should wrap" answer is worse than a documented gap in generality for a
    summary-annotation feature that is secondary to the plot itself.
    """
    if not bands:
        return []
    ordered = sorted(bands)
    merged = [ordered[0]]
    for lo, hi in ordered[1:]:
        prev_lo, prev_hi = merged[-1]
        if lo <= prev_hi:
            merged[-1] = (prev_lo, max(prev_hi, hi))
        else:
            merged.append((lo, hi))
    return merged


def coverage_summary(skills: list[dict]) -> dict:
    """Union width, gap list, and edge-to-edge extent across every skill's own reachable_band_deg
    -- the SAME computation this session did by hand for benchmark_plan.md's own §1, now derived
    from the checkpoints' own metadata (via 60_eval_shooting.py) instead of retyped into a table
    that can drift out of sync with the checkpoints it describes."""
    bands = [tuple(s["reachable_band_deg"]) for s in skills]
    merged = merge_intervals_deg(bands)
    union_deg = sum(hi - lo for lo, hi in merged)
    gaps = [(merged[i][1], merged[i + 1][0]) for i in range(len(merged) - 1)]
    extent_deg = (merged[-1][1] - merged[0][0]) if merged else 0.0
    return {"union_deg": union_deg, "gaps_deg": [hi - lo for lo, hi in gaps], "extent_deg": extent_deg, "islands": merged}


# --------------------------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------------------------- #


def _label_radii(skills: list[dict], ylim_top: float, min_angle_sep_deg: float = 12.0) -> list[float]:
    """One label placement radius per entry in `skills`, in ITS OWN original order (matching the
    order the caller will draw skills/colors in). Two skills whose nominal bearings fall within
    `min_angle_sep_deg` of each other draw at the SAME angle-ish position and their text would
    otherwise overlap into an unreadable smear -- confirmed live, 2026-09-06: skill_012 (-1.32 deg)
    and skill_020 (0.00 deg) are only 1.32 deg apart in a real unified checkpoint's own library.
    Fix: walk skills in ANGULAR order and push each label further out (radius_step) than the
    previous one whenever it's within min_angle_sep_deg of it, resetting back to base_radius once
    a big enough angular jump is seen -- so a cluster of N close skills staggers into N concentric
    labels instead of unreadably overlapping at one radius.

    `base_radius`/`radius_step` are fractions of `ylim_top`, NOT fixed absolute values -- a real
    bug this session's own auto-scaling addition exposed: this axis's top is no longer always the
    old fixed ~108, so a hardcoded absolute label radius (the original 118/13) could land WAY
    outside a much-smaller auto-scaled plot, breaking the whole figure's layout (confirmed live:
    labels scattered into the figure's outer margins, matplotlib's own tight_layout warning fired).
    Scaling both by ylim_top preserves the exact same "~9% beyond the axis's own edge" positioning
    the original fixed values gave when ylim_top happened to be ~108.
    """
    base_radius = ylim_top * (118.0 / 108.0)
    radius_step = ylim_top * (13.0 / 108.0)
    order = sorted(range(len(skills)), key=lambda i: skills[i]["nominal_bearing_deg"])
    radii = [base_radius] * len(skills)
    last_bearing = None
    radius = base_radius
    for idx in order:
        bearing = skills[idx]["nominal_bearing_deg"]
        if last_bearing is not None and abs(bearing - last_bearing) < min_angle_sep_deg:
            radius += radius_step
        else:
            radius = base_radius
        radii[idx] = radius
        last_bearing = bearing
    return radii


def _alpha_and_size_for_n(n: int, min_n_for_full_confidence: int, base_size: float) -> tuple[float, float]:
    """Linear ramp from a floor (never fully invisible -- a low-n bin is still real data, just
    less trustworthy) up to full confidence at `min_n_for_full_confidence` trials. Both alpha and
    marker size scale together so the encoding reads the same way in grayscale print as in color."""
    frac = min(1.0, n / max(1, min_n_for_full_confidence))
    alpha = 0.25 + 0.75 * frac
    size = base_size * (0.35 + 0.65 * frac)
    return alpha, size


def _resolve_radial_scale(skills: list[dict], show_hit_rate: bool, radial_max: float | None) -> tuple[float, list[float]]:
    """Returns (ylim_top, yticks) for the radial axis. A FIXED 0-100% scale (this plot's own
    original design) makes every lobe unreadably tiny whenever every skill's real success_rate
    sits well under 100% -- confirmed live, 2026-09-06: a real 6-skill run where every success
    curve stayed under ~20% rendered as barely-visible slivers hugging the center, even though the
    underlying data has real, meaningful shape.

    Auto-scales to the ACTUAL data instead (`radial_max=None`, the default): rounds UP to the next
    5%-multiple above the largest value actually plotted, with a 20% floor (so a near-all-zero run
    doesn't zoom in on pure noise) and headroom above that for labels/markers. Deliberately
    computed over BOTH success_rate AND hit_rate (when `show_hit_rate`) -- hit_rate is often much
    higher than success_rate (a policy can hit reliably while aiming badly), and scaling to
    success_rate alone would clip the hit-rate reference curve off the top of the plot instead of
    just making it small.

    Pass an explicit `radial_max` (percent) to force a fixed scale instead -- e.g. for several
    figures meant to sit side by side where consistent scaling matters more than per-figure
    legibility (auto-scaling means two runs' figures are NOT directly comparable by lobe size
    alone unless both happen to auto-scale to the same range -- read the numbers, not just shape,
    when comparing across separately-generated figures).

    Gridline step is chosen from a "nice number" table (5/10/20/25/50), not a naive rounded-max/4
    -- that naive version produces ugly ticks like 11%/22%/34%/45% whenever the rounded max isn't
    a multiple of 4 (confirmed live, 2026-09-06). The chosen step is the smallest table entry that
    still keeps the axis to <=4 gridlines, matching how the original fixed 25/50/75/100 scheme
    read.
    """
    if radial_max is not None:
        data_max = radial_max
    else:
        values = []
        for skill in skills:
            for b in skill["bins"]:
                if b["success_rate"] is not None:
                    values.append(100.0 * b["success_rate"])
                if show_hit_rate and b["hit_rate"] is not None:
                    values.append(100.0 * b["hit_rate"])
        data_max = max(values) if values else 100.0
        data_max = max(data_max, 20.0)  # floor -- don't zoom in on a near-all-zero run's own noise
    step = next((s for s in (5.0, 10.0, 20.0, 25.0, 50.0) if data_max <= s * 4), 100.0)
    axis_max = math.ceil(data_max / step) * step
    ylim_top = axis_max * 1.08  # ~8% headroom so the outermost point/label isn't jammed against the edge
    yticks = list(np.arange(step, axis_max + step * 0.5, step))
    return ylim_top, yticks


def plot_polar_coverage(
    skills: list[dict],
    output_path: str,
    min_n_for_full_confidence: int = 15,
    show_hit_rate: bool = True,
    show_band_shading: bool = True,
    radial_max: float | None = None,
    dpi: int = 200,
) -> None:
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="polar")
    # 0 degrees (forward) at the top, positive (the robot's own left -- SkillConfig.resolved_
    # nominal_bearing_deg's own convention) sweeping counter-clockwise, i.e. toward the left side
    # of the page when looking down at the robot from above facing up the page -- matches how a
    # reader would intuitively lay out "robot facing away from them."
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(1)

    ylim_top, yticks = _resolve_radial_scale(skills, show_hit_rate, radial_max)
    ax.set_ylim(0, ylim_top)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{v:.0f}%" for v in yticks], fontsize=8, color="0.4")
    ax.set_rlabel_position(200)
    ax.grid(alpha=0.3)

    colors = itertools.cycle(_PALETTE)
    label_radii = _label_radii(skills, ylim_top)

    for skill, color, label_r in zip(skills, colors, label_radii):
        band_lo, band_hi = skill["reachable_band_deg"]
        nominal = skill["nominal_bearing_deg"]

        if show_band_shading:
            width_rad = math.radians(band_hi - band_lo)
            center_rad = math.radians((band_lo + band_hi) / 2.0)
            ax.bar(
                center_rad, height=ylim_top, width=width_rad, bottom=0,
                color=color, alpha=0.07, edgecolor="none", zorder=0,
            )

        bins = skill["bins"]
        bearings_deg = [nominal + b["theta_offset_center_deg"] for b in bins]
        theta_rad = np.radians(bearings_deg)

        # NaN, not 0.0, for an empty bin -- matplotlib breaks a line at a NaN vertex instead of
        # interpolating across it (see this module's own "GAPS ARE NEVER INTERPOLATED" docstring
        # section), so an unmeasured bin leaves a real hole in the curve, not a fabricated floor.
        success_r = np.array([100.0 * b["success_rate"] if b["success_rate"] is not None else np.nan for b in bins])
        ax.plot(theta_rad, success_r, "-", color=color, linewidth=1.6, alpha=0.9, zorder=3)

        base_size = 46.0
        point_colors, sizes = [], []
        for b in bins:
            n = b["n"]
            a, s = _alpha_and_size_for_n(n, min_n_for_full_confidence, base_size)
            if n == 0:
                a = 0.0  # invisible marker for a truly empty bin -- the line break already shows it
            point_colors.append(mcolors.to_rgba(color, alpha=a))
            sizes.append(s)
        ax.scatter(theta_rad, success_r, s=sizes, c=point_colors, zorder=4, edgecolors="none")

        if show_hit_rate:
            # 2026-09-06: bolder than the original 0.9/0.35 (confirmed live: unreadably faint) --
            # still visibly SECONDARY to the primary success curve above (thinner, dashed, and a
            # touch more transparent: 1.3 vs 1.6 linewidth, 0.65 vs 0.9 alpha), just no longer
            # invisible at a glance.
            hit_r = np.array([100.0 * b["hit_rate"] if b["hit_rate"] is not None else np.nan for b in bins])
            ax.plot(theta_rad, hit_r, "--", color=color, linewidth=1.3, alpha=0.65, zorder=2)

        # Skill label just outside the plotted radius, at its own nominal bearing.
        label_rad = math.radians(nominal)
        ax.text(
            label_rad, label_r, skill["label"], color=color, fontsize=9, fontweight="bold",
            ha="center", va="center",
        )

    # Legend: one proxy line per skill (color) + one generic dashed gray line explaining the
    # secondary hit-rate curve, rather than doubling every skill's own legend entry.
    handles = [plt.Line2D([0], [0], color=c, lw=1.6, label=s["label"]) for s, c in zip(skills, itertools.cycle(_PALETTE))]
    if show_hit_rate:
        handles.append(plt.Line2D([0], [0], color="0.4", lw=1.3, ls="--", alpha=0.65, label="hit rate (reference only)"))
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.32, 1.08), fontsize=8, frameon=False)

    summary = coverage_summary(skills)
    gaps_str = ", ".join(f"{g:.1f}°" for g in sorted(summary["gaps_deg"], reverse=True)) or "none"
    fig.text(
        0.5, 0.02,
        f"Reachable union: {summary['union_deg']:.1f}°  |  gaps: {gaps_str}  |  "
        f"marker size/opacity ∝ trials per bin (full confidence at n≥{min_n_for_full_confidence})",
        ha="center", fontsize=8, color="0.35",
    )

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if output_path.lower().endswith(".png"):
        pdf_path = output_path[: -len(".png")] + ".pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
        logger.info(f"[71-plot-polar-coverage] also wrote vector twin -> {pdf_path}")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input", required=True,
        help="60_eval_shooting.py's own output JSON, OR the run folder containing it (its "
        "--output run folder, e.g. 'out/eval_shooting/<run_timestamp>-<run_name>') -- paste "
        "that script's own logged folder path directly, no need to append the filename.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output image path (.png recommended -- a .pdf twin is written alongside it). "
        "Default: 'f2_polar_coverage.png' next to the resolved input JSON, so the plot lands in "
        "the same run folder as the data it came from.",
    )
    parser.add_argument(
        "--min-n-for-full-confidence", type=int, default=15,
        help="Trials/bin at or above which a bin draws at full opacity/size (default 15). Below "
        "this, both scale down linearly toward (but never reaching) fully invisible.",
    )
    parser.add_argument("--show-hit-rate", dest="show_hit_rate", action="store_true", default=True)
    parser.add_argument("--no-show-hit-rate", dest="show_hit_rate", action="store_false")
    parser.add_argument("--show-band-shading", dest="show_band_shading", action="store_true", default=True)
    parser.add_argument("--no-show-band-shading", dest="show_band_shading", action="store_false")
    parser.add_argument(
        "--radial-max", type=float, default=None,
        help="Force the radial axis to this fixed percent (e.g. 100) instead of auto-scaling to "
        "the actual data. Default: auto -- rounds up to the next 5%% above the largest "
        "success_rate/hit_rate value actually plotted (20%% floor), so a low-success run doesn't "
        "render as unreadable slivers hugging the center. Auto-scaled figures from different runs "
        "are NOT directly comparable by lobe size alone -- use a fixed value for a side-by-side set.",
    )
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    input_json = resolve_input_json(args.input)
    output_path = args.output or os.path.join(os.path.dirname(os.path.abspath(input_json)), "f2_polar_coverage.png")

    data = load_eval_shooting_json(input_json)
    skills = data["skills"]
    if not skills:
        logger.error(f"[71-plot-polar-coverage] {input_json!r} has an empty 'skills' list -- nothing to plot.")
        return 1

    plot_polar_coverage(
        skills, output_path,
        min_n_for_full_confidence=args.min_n_for_full_confidence,
        show_hit_rate=args.show_hit_rate,
        show_band_shading=args.show_band_shading,
        radial_max=args.radial_max,
        dpi=args.dpi,
    )
    logger.info(f"[71-plot-polar-coverage] wrote {len(skills)} skill lobe(s) -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
