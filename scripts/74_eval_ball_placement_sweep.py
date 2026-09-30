#!/usr/bin/env python3
"""Ball-placement sweep across the FULL skill library: for a unified checkpoint, roll out EVERY
skill against the SAME shared sequence of continuous, randomly-drawn absolute ball placements
(not jitter around each skill's own nominal spot), and record, per placement: whether ANY skill
made a real strike-window contact, plus each contributing skill's full shot trajectory. Feeds
both a ball-placement hit-rate heatmap and a "shots from wherever the ball actually was" combined
trajectory plot (75_plot_ball_placement_heatmap.py) -- see that script's own module docstring for
the rendering side.

WHY "ANY skill", NOT one specific skill: this measures the LIBRARY's aggregate reach, matching how
the real deployed system actually works -- an operator (or autonav) selects a skill and the ball is
wherever it is; the question this answers is "does the library, as a whole, cover this XY region",
not "does skill_012 alone cover it". A per-skill breakdown is still fully recoverable from this
same run's own trial_records_out-derived per-skill hit list (see write_output's own "per_skill"
field in the summary JSON) without a re-run, if that's ever wanted for a follow-up.

WHY GATED TO THE STRIKE WINDOW, NOT ANY CONTACT DURING THE HOLD: after a skill's clip ends the
policy auto-returns to locomotion and keeps walking for the rest of the hold window -- an
incidental late bump would count as a "hit" indistinguishably from a real strike, which inflates
exactly the displaced placements this sweep exists to measure honestly (a genuine miss at an
off-nominal placement is the interesting/expected outcome at the edges of this sweep, not
something to paper over). See mujoco_kick_survival_scan.py's own --hit-window-lo/hi-tick docstring.
The window opens --strike-window-lead-ticks (default 15) BEFORE the annotated strike start, because
the distilled policy can strike ahead of the annotation: skill_020 hit at ticks 105-112 against an
annotated start of 115, so an unpadded window rejected 56/57 of its in-box kicks (2026-09-10).

CONTACT vs DIRECTED SHOT (2026-09-10): any_hit/per_skill_hit answer "did a foot touch the ball",
nothing more -- confirmed on a real 1500-placement/7-skill run that most contacts across a wide
sweep are glancing touches, not directed kicks (per-skill direction-success-of-hits ranged 6-36%).
per_skill_direction_hit/any_direction_hit are the stricter statistic: hit_step != -1 AND the same
trial's min_target_dist (already computed live by the worker from the SAME commanded target
training's error_ball_to_target reward uses -- no extra rollout cost) falls within
--direction-success-sigma-m. Use any_hit for "does the library's foot ever reach this XY region"
and any_direction_hit for "does the library ever put a shot on target from this XY region" -- they
answer different questions and read very differently on the same placement grid.

BBOX DERIVATION: by default, the union of every swept skill's own trained ball spawn point
(get_skill_ball_xy, from the checkpoint's own embedded metadata) padded by --trained-jitter-m
(default 0.1 -- THIS CHECKPOINT LINEAGE's own verified BallConfig.position_randomization; a
DIFFERENT checkpoint may use a different value -- check its own resolved holosoma_config.yaml
rather than trusting this default, same caveat configs/eval_shooting/example.yaml's own comment
already states for the identical number), and then --pad-m (default 0.0) beyond that union -- so
the default sweep exactly covers "everywhere any one skill was ever trained to expect the ball",
and --pad-m is the explicit lever for deliberately probing outside that. Override with --bbox-x/
--bbox-y directly to skip auto-derivation entirely.

SHARED PLACEMENT SEQUENCE ACROSS SKILLS: --ball-pos-range-x/-y draws a fresh uniform placement
EVERY TRIAL inside the worker; passing the SAME --seed to every skill_id's own subprocess call
reproduces the IDENTICAL per-trial placement sequence for all of them (confirmed live,
2026-09-10 -- three skill_ids at seed 42 produced byte-identical first-trajectory-points), which
is what lets this script test every skill against the same shared set of placements without
passing the placement list explicitly.

Usage:
    python scripts/74_eval_ball_placement_sweep.py \\
        --onnx-path logs/UnifiedBallKickingEnhanced/<run>/model_NNNNNN.onnx \\
        --num-placements 300 --output out/ball_placement_sweep
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "holosoma"))

from holosoma.record_mujoco_survival_scan import record_survival_scan  # noqa: E402
from holosoma.sim2sim_eval import discover_num_skills, get_strike_window_ticks  # noqa: E402

DEFAULT_TRAINED_JITTER_M = 0.1  # see module docstring's BBOX DERIVATION section


def get_skill_ball_xy(onnx_path: str, skill_id: int) -> tuple[float, float]:
    """That skill's trained ball spawn point, straight from the checkpoint's own embedded
    metadata -- same field/index convention as 60_eval_shooting.py's own get_skill_geometry,
    reimplemented minimally here (this script needs only the point, not the full geometry dict)
    rather than importing across scripts/ files, matching this project's own convention of each
    scripts/ file being self-contained (see e.g. 71/72's own duplicated resolve_input_json-style
    helpers instead of cross-importing 60_eval_shooting.py)."""
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    meta = sess.get_modelmeta().custom_metadata_map
    x, y = json.loads(meta["skill_ball_xy"])[skill_id]
    return float(x), float(y)


def _fresh_lock_path(tag: str) -> str:
    return os.path.join(tempfile.gettempdir(), f"holosoma_ball_placement_sweep_{tag}_{uuid.uuid4().hex}.lock")


def derive_bbox(
    onnx_path: str, skill_ids: list[int], trained_jitter_m: float, pad_m: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    xs, ys = [], []
    for sid in skill_ids:
        x, y = get_skill_ball_xy(onnx_path, sid)
        xs.append(x)
        ys.append(y)
    margin = trained_jitter_m + pad_m
    bbox_x = (min(xs) - margin, max(xs) + margin)
    bbox_y = (min(ys) - margin, max(ys) + margin)
    logger.info(
        f"[74-ball-placement-sweep] auto-derived bbox from {len(skill_ids)} skill(s)' own ball_xy "
        f"(+/-{trained_jitter_m} trained jitter, +/-{pad_m} extra pad): "
        f"x=[{bbox_x[0]:.3f}, {bbox_x[1]:.3f}] y=[{bbox_y[0]:.3f}, {bbox_y[1]:.3f}]"
    )
    return bbox_x, bbox_y


def run_one_skill(
    onnx_path: str, skill_id: int, num_placements: int, seed: int, bbox_x: tuple, bbox_y: tuple,
    settle_s: float, hold_s: float, timeout_s: float, trajectory_dir: str, direction_success_sigma_m: float,
    strike_window_lead_ticks: int,
) -> dict:
    window = get_strike_window_ticks(onnx_path, skill_id)
    hit_window_ticks = (
        (max(0, window[0] - strike_window_lead_ticks), window[1] - 1) if window is not None else None
    )
    if hit_window_ticks is None:
        logger.warning(
            f"[74-ball-placement-sweep] skill_id={skill_id}: no strike/stand boundary metadata -- "
            "hit detection will be UNGATED (any contact during the whole hold window counts)."
        )

    trial_records: list[dict] = []
    trajectory_path = os.path.join(trajectory_dir, f"skill_{skill_id}_trajectories.json")
    fall_rate, hit_rate, _ = record_survival_scan(
        onnx_path=onnx_path,
        step_label="ball_placement_sweep",
        num_trials=num_placements,
        skill_id=skill_id,
        seed=seed,
        ball_pos_randomization=(0.0, 0.0),
        kick_aim_enabled=True,
        kick_aim_theta_max_deg=0.0,
        ball_pos_range=(bbox_x, bbox_y),
        hit_window_ticks=hit_window_ticks,
        direction_success_sigma_m=direction_success_sigma_m,
        trial_records_out=trial_records,
        trajectory_output_path=trajectory_path,
        settle_s=settle_s,
        hold_s=hold_s,
        timeout_s=timeout_s,
        stream_output=True,
        stream_prefix=f"skill_{skill_id}",
        lock_path=_fresh_lock_path(f"skill{skill_id}"),
    )
    if not trial_records:
        raise RuntimeError(f"skill_id={skill_id}: record_survival_scan produced zero trial records (busy lock, timeout, or crash -- see warnings above).")
    return {
        "skill_id": skill_id,
        "fall_rate": fall_rate,
        "hit_rate": hit_rate,
        "hit_window_ticks": hit_window_ticks,
        "trial_records": trial_records,
        "trajectory_path": trajectory_path,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx-path", required=True)
    parser.add_argument(
        "--skill-ids", type=int, nargs="+", default=None,
        help="Which skills to sweep (omit to auto-discover every skill this checkpoint embeds).",
    )
    parser.add_argument("--num-placements", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0, help="SAME seed across every skill -- this is what makes them share one placement sequence.")
    parser.add_argument("--bbox-x", type=float, nargs=2, default=None, metavar=("LO", "HI"), help="Explicit override -- skips auto-derivation entirely.")
    parser.add_argument("--bbox-y", type=float, nargs=2, default=None, metavar=("LO", "HI"))
    parser.add_argument("--trained-jitter-m", type=float, default=DEFAULT_TRAINED_JITTER_M, help="See module docstring's BBOX DERIVATION section -- verify against this checkpoint's OWN resolved holosoma_config.yaml, do not assume.")
    parser.add_argument("--pad-m", type=float, default=0.0, help="Extra margin beyond the union-of-trained-boxes bbox, for deliberately probing outside training.")
    parser.add_argument("--settle-s", type=float, default=1.0)
    parser.add_argument("--hold-s", type=float, default=4.0, help="Shorter than the 8.0s default elsewhere -- hits are strike-window-gated, so nothing past the swing is being measured; the plotting script's own closest-approach truncation needs only enough post-strike roll to find that point, not the full settle tail.")
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument(
        "--direction-success-sigma-m", type=float, default=1.0,
        help="Radius (meters) around the commanded target that a contact's min_target_dist must fall "
        "within to count as a DIRECTED shot, not just a touch -- matches mujoco_kick_survival_scan.py's "
        "own --direction-success-sigma-m default (record_survival_scan already forwards 1.0 by default "
        "either way; this just exposes it here so the placements list's own per_skill_direction_hit/ "
        "any_direction_hit fields use the same threshold, instead of silently trusting the callee's "
        "default). See 75_plot_ball_placement_heatmap.py's own module docstring for why any_hit alone "
        "(mere foot-ball contact) overstates what a reader would call a 'successful kick' -- most "
        "contacts across a wide sweep are glancing touches, not directed shots (confirmed 2026-09-10: "
        "per-skill direction-success-of-hits ranged 6-36% on a real 1500-placement/7-skill run).",
    )
    parser.add_argument(
        "--strike-window-lead-ticks", type=int, default=15,
        help="Open each skill's contact window this many ticks (50 Hz) before its annotated strike start; the end "
        "stays at stand start, which is what excludes post-kick walking bumps. 15 (0.3 s) covers the policy striking "
        "ahead of the annotation (skill_020: ticks 105-112 vs annotated 115). Much larger values start counting "
        "run-up steps as contacts (skill_011's earliest touches came up to 107 ticks early).",
    )
    parser.add_argument("--max-concurrent-skills", type=int, default=4)
    parser.add_argument("--output", default="out/ball_placement_sweep", help="Base output directory -- results land in <output>/<timestamp>/.")
    args = parser.parse_args()

    skill_ids = args.skill_ids if args.skill_ids is not None else list(range(discover_num_skills(args.onnx_path)))
    logger.info(f"[74-ball-placement-sweep] sweeping {len(skill_ids)} skill(s): {skill_ids}")

    if args.bbox_x is not None or args.bbox_y is not None:
        if args.bbox_x is None or args.bbox_y is None:
            parser.error("--bbox-x and --bbox-y must be given together.")
        bbox_x, bbox_y = tuple(args.bbox_x), tuple(args.bbox_y)
    else:
        bbox_x, bbox_y = derive_bbox(args.onnx_path, skill_ids, args.trained_jitter_m, args.pad_m)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output, run_timestamp)
    trajectory_dir = os.path.join(run_dir, "trajectories")
    os.makedirs(trajectory_dir, exist_ok=True)

    results: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=args.max_concurrent_skills) as pool:
        futures = {
            pool.submit(
                run_one_skill, args.onnx_path, sid, args.num_placements, args.seed, bbox_x, bbox_y,
                args.settle_s, args.hold_s, args.timeout_s, trajectory_dir, args.direction_success_sigma_m,
                args.strike_window_lead_ticks,
            ): sid
            for sid in skill_ids
        }
        for future in futures:
            sid = futures[future]
            try:
                results[sid] = future.result()
            except Exception:
                logger.exception(f"[74-ball-placement-sweep] skill_id={sid} failed")

    if not results:
        logger.error("[74-ball-placement-sweep] every skill failed -- nothing to write.")
        return 1

    # Cross-skill aggregation: trial i is the SAME placement for every skill (shared seed) -- zip
    # by index, not by any explicit placement id.
    n = args.num_placements
    placements = []
    for i in range(n):
        per_skill_hit = {}
        # A contact ("hit") only means the foot touched the ball -- most contacts across a wide
        # sweep are glancing touches, not directed shots (see --direction-success-sigma-m's own
        # help). per_skill_direction_hit additionally requires the SAME trial's min_target_dist
        # (already computed live by the worker, no extra rollout cost) to land within
        # args.direction_success_sigma_m of the commanded target -- the stat 75's own success-
        # gated figures are built from.
        per_skill_direction_hit = {}
        xy = None
        for sid, r in results.items():
            rec = r["trial_records"][i]
            per_skill_hit[str(sid)] = rec["hit_step"] != -1
            per_skill_direction_hit[str(sid)] = (
                rec["hit_step"] != -1
                and rec["min_target_dist"] is not None
                and rec["min_target_dist"] <= args.direction_success_sigma_m
            )
            if xy is None:
                with open(r["trajectory_path"]) as f:
                    xy = json.load(f)["trials"][i]["trajectory_xy"][0]
        placements.append({
            "x": xy[0], "y": xy[1],
            "any_hit": any(per_skill_hit.values()), "per_skill_hit": per_skill_hit,
            "any_direction_hit": any(per_skill_direction_hit.values()), "per_skill_direction_hit": per_skill_direction_hit,
        })

    summary = {
        "onnx_path": args.onnx_path,
        "skill_ids": skill_ids,
        "num_placements": args.num_placements,
        "seed": args.seed,
        "bbox_x": list(bbox_x),
        "bbox_y": list(bbox_y),
        "trained_jitter_m": args.trained_jitter_m,
        "pad_m": args.pad_m,
        "direction_success_sigma_m": args.direction_success_sigma_m,
        "strike_window_lead_ticks": args.strike_window_lead_ticks,
        "per_skill": {
            str(sid): {
                "fall_rate": r["fall_rate"], "hit_rate": r["hit_rate"], "hit_window_ticks": r["hit_window_ticks"],
                "ball_xy": list(get_skill_ball_xy(args.onnx_path, sid)),
            }
            for sid, r in results.items()
        },
        "any_hit_rate": sum(p["any_hit"] for p in placements) / n if n else None,
        "any_direction_hit_rate": sum(p["any_direction_hit"] for p in placements) / n if n else None,
        "placements": placements,
    }
    summary_path = os.path.join(run_dir, "ball_placement_sweep.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f)
    logger.info(
        f"[74-ball-placement-sweep] wrote {len(results)}/{len(skill_ids)} skill(s), "
        f"any_hit_rate={summary['any_hit_rate']:.3f} -> {summary_path}"
    )
    return 0 if len(results) == len(skill_ids) else 1


if __name__ == "__main__":
    raise SystemExit(main())
