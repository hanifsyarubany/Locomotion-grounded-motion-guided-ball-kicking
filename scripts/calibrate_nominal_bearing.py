"""Fit a skill's nominal strike bearing from real MuJoCo kick-rollout trajectory CSVs.

Reads one or more CSVs produced by ``mujoco_kick_rollout_worker.py --output-trajectory-path``
(columns: tick, t_s, phase, task_mode, curr_motion_timestep, ball_x, ball_y, ball_z, ...) and
fits the ball's departure bearing in the ROBOT-LOCAL xy plane -- azimuth 0 is the actor's
+x (forward) axis, positive is the actor's +y (its own left), matching SkillConfig.x/y's own
convention. This rollout worker spawns the robot at the world origin with identity heading, so
world xy IS robot-local xy here; nothing else in this script assumes that, but the CALLER's
choice of tool does.

WHY NOT a single near point (rejected during design): bearing error from a position error e at
distance d is ~e/d. A point 0.3 m from the ball turns a 2 cm tracking/contact-noise error into
~3.8 deg of bearing error; a point 4 m out turns the same error into ~0.29 deg -- a 13x
difference. The early post-contact ticks are also the noisiest (impact transient), so a near
point samples exactly the wrongest part of the trajectory. This script instead fits over a
WINDOW placed well after contact, where the lever arm is long and the motion is a straight line.

CONTACT DETECTION IS XY-DISPLACEMENT-BASED, NOT Z-HEIGHT-BASED (2026-08-22, corrected after a
real measurement contradicted the first version of this script -- see CLAUDE.md's own rule to
trust measurement over assumption). An earlier version keyed "contact" on the ball rising above
its rest height, mirroring ball_trajectory_scrubber.py's kick/landing estimator. Run against a
real Stage B checkpoint (skill_012), it reported "no contact" for a trial whose ball plainly
travelled 2.9 m (1.39,0.02) -> (4.26,-0.85): a ground-rolling strike (inside-foot pass) barely
lifts the ball off the floor (max z only ~3 mm above rest, well under the airborne threshold that
correctly flags skill_013's visibly lofted strike). z-based detection silently conflates "this
kick looked airborne" with "this kick happened" -- wrong for calibration, where a low driven pass
is exactly as valid a nominal direction as a chip. Displacement from the pre-contact rest XY
position detects both.

USAGE (one trial):
    python scripts/calibrate_nominal_bearing.py trial_00.csv

USAGE (many trials -> median + spread, the intended calibration workflow):
    python scripts/calibrate_nominal_bearing.py trial_*.csv --skill-name skill_013
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

_REST_SAMPLE_ROWS = 10  # rows to establish the pre-contact rest xy
_CONTACT_DISPLACEMENT_M = 0.03  # xy displacement from rest past which "contact happened"
_WINDOW_START_DISPLACEMENT_M = 0.15  # xy displacement past which the impact transient is over
_MIN_WINDOW_LEN_M = 0.20  # fit window must span at least this much to trust the bearing


@dataclass
class TrialBearing:
    path: str
    bearing_deg: float
    window_start_tick: int
    window_end_tick: int
    window_len_m: float


def _read_rows(csv_path: Path) -> list[dict[str, float]]:
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            rows.append(
                {
                    "tick": int(r["tick"]),
                    "ball_x": float(r["ball_x"]),
                    "ball_y": float(r["ball_y"]),
                    "ball_z": float(r["ball_z"]),
                }
            )
    if len(rows) < _REST_SAMPLE_ROWS + 2:
        raise ValueError(f"{csv_path}: only {len(rows)} rows, too short to calibrate from")
    return rows


def fit_bearing_deg(rows: list[dict[str, float]]) -> TrialBearing:
    """Fit the post-contact departure bearing, robot-local xy, degrees.

    Method: (1) establish rest xy from the first _REST_SAMPLE_ROWS rows (ball not yet struck);
    (2) find the first tick the ball's xy displacement from rest exceeds _CONTACT_DISPLACEMENT_M
    (contact happened -- works for a ground-rolling strike exactly as well as a lofted one, since
    it never looks at z); (3) advance further to the first tick displacement exceeds
    _WINDOW_START_DISPLACEMENT_M, to fit from past the initial impact transient rather than at
    it; (4) fit the straight line from there to the final recorded tick. If the ball never moves
    (no contact this trial), raises ValueError rather than silently returning a meaningless
    bearing from position noise.
    """
    rest_x = statistics.median(r["ball_x"] for r in rows[:_REST_SAMPLE_ROWS])
    rest_y = statistics.median(r["ball_y"] for r in rows[:_REST_SAMPLE_ROWS])

    def _disp(r: dict[str, float]) -> float:
        return math.hypot(r["ball_x"] - rest_x, r["ball_y"] - rest_y)

    contact_idx = next((i for i, r in enumerate(rows) if _disp(r) > _CONTACT_DISPLACEMENT_M), None)
    if contact_idx is None:
        raise ValueError("ball never moved from its rest position -- no contact detected in this trial")

    window_start_idx = next(
        (i for i in range(contact_idx, len(rows)) if _disp(rows[i]) > _WINDOW_START_DISPLACEMENT_M),
        None,
    )
    if window_start_idx is None:
        # Contact happened but the ball never got far enough from rest for a transient-free fit
        # (e.g. --hold-s too short, or a very weak tap) -- fit from contact_idx anyway; the
        # window_len_m check below will flag it as untrustworthy rather than silently accepting.
        window_start_idx = contact_idx

    start = rows[window_start_idx]
    end = rows[-1]
    dx = end["ball_x"] - start["ball_x"]
    dy = end["ball_y"] - start["ball_y"]
    window_len = math.hypot(dx, dy)
    if window_len < _MIN_WINDOW_LEN_M:
        raise ValueError(
            f"post-contact window too short ({window_len:.3f} m, ticks {start['tick']}->{end['tick']}) "
            "to fit a reliable bearing -- increase --hold-s and re-run"
        )
    bearing = math.degrees(math.atan2(dy, dx))
    return TrialBearing(
        path="", bearing_deg=bearing,
        window_start_tick=start["tick"], window_end_tick=end["tick"], window_len_m=window_len,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_paths", nargs="+", help="one or more trajectory CSVs from mujoco_kick_rollout_worker.py")
    ap.add_argument("--skill-name", default=None, help="label only, for the printed summary")
    ns = ap.parse_args()

    results: list[TrialBearing] = []
    failures: list[str] = []
    for p in ns.csv_paths:
        path = Path(p)
        try:
            rows = _read_rows(path)
            tb = fit_bearing_deg(rows)
            tb.path = str(path)
            results.append(tb)
            print(
                f"{path.name}: bearing={tb.bearing_deg:+7.2f} deg  "
                f"window=ticks[{tb.window_start_tick},{tb.window_end_tick}] len={tb.window_len_m:.2f}m"
            )
        except ValueError as e:
            failures.append(f"{path.name}: SKIPPED -- {e}")

    for f in failures:
        print(f)

    if not results:
        raise SystemExit("no trial produced a usable bearing -- check --hold-s and that contact actually occurred")

    bearings = [r.bearing_deg for r in results]
    label = ns.skill_name or "skill"
    print()
    print(f"=== {label}: {len(results)}/{len(ns.csv_paths)} trials usable ===")
    print(f"median bearing: {statistics.median(bearings):+.2f} deg")
    if len(bearings) > 1:
        print(f"mean:           {statistics.mean(bearings):+.2f} deg")
        print(f"stdev:          {statistics.stdev(bearings):.2f} deg")
        print(f"range:          [{min(bearings):+.2f}, {max(bearings):+.2f}] deg")
    if len(bearings) > 1 and statistics.stdev(bearings) > 5.0:
        print(
            "WARNING: spread > 5 deg -- 'nominal direction' may not be well-defined for this "
            "skill; consider more trials or treat theta=0 as approximate."
        )


if __name__ == "__main__":
    main()
