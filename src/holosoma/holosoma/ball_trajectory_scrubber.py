"""Standalone tool: generate an interactive, browser-viewable scrubber for one or more recorded
ball-trajectory CSVs (as written by mujoco_kick_interactive.py, mujoco_kick_rollout_worker.py's
--output-trajectory-path, or replay.py's HOLOSOMA_REPLAY_RECORD_BALL_TRAJECTORY_DIR recorder), for
reading off the ball's exact (x, y) at the moment of the kick and where it actually lands -- to
calibrate SkillConfig's ball spawn (x, y) and configs/skill2.yaml's (target_x, target_y) against a
real recorded trial instead of eyeballing a drag in the viewer.

No simulator, no GPU, no training context needed -- this only reads the CSV's own recorded
(tick, t_s, ball_x, ball_y, ball_z[, phase]) columns and renders the FULL, un-thinned path client-
side: every recorded row, drawn continuously in the order it was logged (drag-around included, not
just the kick), same "scrub through every frame" idea as this project's other tool,
motion_clip_scrubber.py, applied to a logged trajectory instead of a motion clip. Companion asset
``ball_trajectory_scrubber_template.html`` (same directory) holds the actual viewer UI; this
script's job is purely loading + shaping the per-CSV data and splicing it in.

Usage
-----
    python ball_trajectory_scrubber.py TRAJ1.csv [TRAJ2.csv ...] [--labels NAME1 NAME2 ...] \\
        [--out FILE] [--no-open] [--serve] [--serve-port PORT]

Example (every trial recorded by mujoco_kick_interactive.py so far)::

    python ball_trajectory_scrubber.py trajectories/*.csv --out trajectory_scrubber.html

Then open the output HTML in any browser (or, in a remote/headless dev environment, use --serve and
forward the port, same as motion_clip_scrubber.py) -- no server process, no dependency on this repo
being reachable afterward.

Output: a single self-contained HTML file with every trial's trajectory embedded inline.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import webbrowser
from pathlib import Path

import numpy as np

# Columns every known trajectory-CSV writer in this project emits (mujoco_kick_interactive.py,
# mujoco_kick_rollout_worker.py, replay.py's recorder) -- "phase" is writer-specific (replay.py's
# recorder doesn't have a phase concept) so it's read if present but never required.
_REQUIRED_COLUMNS = ("tick", "t_s", "ball_x", "ball_y", "ball_z")

# How far above the resting height counts as "actually left the ground" -- distinguishes a real
# kick from a trial that's pure drag-around (ball resting/rolling the whole time, e.g. 5 of the 7
# trials recorded in this project's first interactive session). Also how close back to resting
# height counts as "landed" once past the flight peak.
_AIRBORNE_EPS_M = 0.02


def _estimate_kick_and_landing(z: np.ndarray, hold_mask: np.ndarray | None) -> tuple[int, int, bool, float]:
    """Auto-estimate (kick_tick, landing_tick, kicked, rest_z) from the z-height trace alone, as a
    starting point to correct in the viewer -- NOT authoritative, same spirit as motion_clip_
    scrubber.py's _estimate_boundaries. Prefers the recorded "hold" phase (written by
    mujoco_kick_interactive.py / mujoco_kick_rollout_worker.py right when [TRIGGER_KICK] fires) when
    available, since that's a ground-truth event marker, not a heuristic; falls back to a pure
    height-based guess when no phase column exists (replay.py's recorder) or the phase never goes
    "hold" in this file."""
    T = len(z)
    rest_z = float(np.median(z[: max(1, min(10, T))]))

    if hold_mask is not None and bool(hold_mask.any()):
        kick_tick = int(np.argmax(hold_mask))
    else:
        airborne = z > rest_z + _AIRBORNE_EPS_M
        kick_tick = int(np.argmax(airborne)) if bool(airborne.any()) else 0

    peak_idx = kick_tick + int(np.argmax(z[kick_tick:])) if kick_tick < T else T - 1
    kicked = bool(z[peak_idx] - rest_z > _AIRBORNE_EPS_M)

    landing_tick = T - 1
    if kicked:
        for i in range(peak_idx, T):
            if z[i] <= rest_z + _AIRBORNE_EPS_M:
                landing_tick = i
                break

    return kick_tick, landing_tick, kicked, rest_z


def _load_trajectory(csv_path: Path, label: str) -> dict:
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        missing = [c for c in _REQUIRED_COLUMNS if c not in fieldnames]
        if missing:
            raise ValueError(
                f"{csv_path}: missing required column(s) {missing} (found {fieldnames}) -- not a "
                "recognized ball-trajectory CSV. Expected the schema written by "
                "mujoco_kick_interactive.py / mujoco_kick_rollout_worker.py / replay.py's recorder."
            )
        has_phase = "phase" in fieldnames
        rows = list(reader)

    if not rows:
        raise ValueError(f"{csv_path}: no data rows")

    tick = np.array([int(r["tick"]) for r in rows])
    t_s = np.array([float(r["t_s"]) for r in rows])
    x = np.array([float(r["ball_x"]) for r in rows])
    y = np.array([float(r["ball_y"]) for r in rows])
    z = np.array([float(r["ball_z"]) for r in rows])
    phase = [r["phase"] for r in rows] if has_phase else None
    hold_mask = np.array([p == "hold" for p in phase]) if phase is not None else None

    T = len(rows)
    dt = np.gradient(t_s)
    dt_safe = np.where(dt > 0, dt, np.nan)
    speed = np.hypot(np.gradient(x), np.gradient(y)) / dt_safe
    speed = np.nan_to_num(speed, nan=0.0, posinf=0.0, neginf=0.0)

    kick_tick, landing_tick, kicked, rest_z = _estimate_kick_and_landing(z, hold_mask)

    return dict(
        name=label,
        T=T,
        hasPhase=has_phase,
        tick=tick.tolist(),
        t=np.round(t_s, 4).tolist(),
        x=np.round(x, 4).tolist(),
        y=np.round(y, 4).tolist(),
        z=np.round(z, 4).tolist(),
        speed=np.round(speed, 4).tolist(),
        phase=phase if phase is not None else [],
        restZ=round(rest_z, 4),
        kicked=kicked,
        kickTick=kick_tick,
        landingTick=landing_tick,
    )


def _build_html(csv_paths: list[Path], labels: list[str]) -> str:
    trials = {}
    for i, (path, label) in enumerate(zip(csv_paths, labels)):
        print(f"[{i}] loading {path} as '{label}'...", file=sys.stderr)
        trials[str(i)] = _load_trajectory(path, label)
        c = trials[str(i)]
        status = (
            f"kicked -- kick@tick {c['kickTick']} landing@tick {c['landingTick']}"
            if c["kicked"] else "drag-only (never left rest height, no kick detected)"
        )
        print(f"    {c['T']} rows, rest_z={c['restZ']:.3f}m -- {status} (verify visually)", file=sys.stderr)

    data = dict(trials=trials)

    template_path = Path(__file__).with_name("ball_trajectory_scrubber_template.html")
    template = template_path.read_text()
    marker = "/*__TRAJECTORY_DATA__*/"
    if marker not in template:
        raise RuntimeError(f"template at {template_path} is missing the {marker} injection point")
    return template.replace(marker, json.dumps(data, separators=(",", ":")))


def serve(csv_paths: list[Path], labels: list[str], host: str = "127.0.0.1", port: int = 8788) -> None:
    """Build the page and serve it from memory over plain HTTP, for an SSH-only session with no
    $BROWSER forwarding. Binds to 127.0.0.1 only -- same rationale as motion_clip_scrubber.py's
    serve(), see there for the full explanation. Default port differs (8788 vs 8787) so both tools
    can run side by side without colliding."""
    import http.server

    html_bytes = _build_html(csv_paths, labels).encode("utf-8")

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 -- http.server's own naming convention
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html_bytes)))
            self.end_headers()
            self.wfile.write(html_bytes)

        def log_message(self, fmt, *fmt_args):  # quieter default access log
            print(f"  [{self.address_string()}] {fmt % fmt_args}", file=sys.stderr)

    httpd = http.server.HTTPServer((host, port), _Handler)
    print(f"\nserving on http://{host}:{port}  (Ctrl+C to stop)", file=sys.stderr)
    print(
        "if this doesn't just work: forward the port over your SSH connection --\n"
        f"  ssh -L {port}:localhost:{port} <your-usual-ssh-target>\n"
        f"then open http://localhost:{port} in your LOCAL browser. VS Code Remote-SSH users: check "
        "the Ports tab first, it often auto-forwards a freshly-opened listening port.",
        file=sys.stderr,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.", file=sys.stderr)


def build(csv_paths: list[Path], labels: list[str], out_path: Path, open_browser: bool = True) -> None:
    html = _build_html(csv_paths, labels)
    out_path.write_text(html)
    print(f"wrote {out_path} ({out_path.stat().st_size / 1024:.0f} KB)", file=sys.stderr)

    if open_browser:
        url = out_path.resolve().as_uri()
        try:
            opened = webbrowser.open(url)
        except Exception as exc:  # noqa: BLE001 -- never let a browser-launch failure fail the run
            opened = False
            print(f"note: could not open a browser ({exc}); open manually: {url}", file=sys.stderr)
        if opened:
            print(f"opened {url}", file=sys.stderr)
        else:
            print(f"note: no browser opened (headless / no $BROWSER?); open manually: {url}", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs="+", type=Path, help="one or more ball-trajectory CSV files")
    p.add_argument(
        "--labels", nargs="*", default=None,
        help="display label per trial, same order as csv args (default: filename stem of each)",
    )
    p.add_argument(
        "--out", type=Path, default=Path("ball_trajectory_scrubber.html"),
        help="output HTML path (default: ./ball_trajectory_scrubber.html)",
    )
    p.add_argument(
        "--no-open", action="store_true",
        help="don't open the result in a browser after writing it (default: open automatically). "
        "Relies on $BROWSER / webbrowser.open, which VS Code Remote/devcontainers forward to your "
        "LOCAL machine's default browser -- a plain headless SSH session with no such forwarding "
        "may instead print an error or silently do nothing, so this never blocks or fails the run.",
    )
    p.add_argument(
        "--serve", action="store_true",
        help="instead of writing a file and trying $BROWSER, start a local HTTP server on "
        "127.0.0.1 and block -- for a plain SSH session where there's no browser to forward to. "
        "Forward the port over your existing SSH connection (or via VS Code Remote-SSH's Ports "
        "tab) and open it in your OWN local browser. Ctrl+C to stop. Ignores --out/--no-open.",
    )
    p.add_argument(
        "--serve-port", type=int, default=8788,
        help="port for --serve (default: 8788)",
    )
    args = p.parse_args()

    for path in args.csv:
        if not path.exists():
            p.error(f"csv file not found: {path}")

    if args.labels:
        labels = args.labels
        if len(labels) != len(args.csv):
            p.error(f"--labels has {len(labels)} entries but {len(args.csv)} csv files were given")
    else:
        # Trajectory CSVs are timestamp-named (ball_trajectory_YYYYMMDD_HHMMSS.csv) so, unlike
        # motion_clip_scrubber.py's npz inputs, the stem is already unique in every real case here
        # -- no ancestor-directory disambiguation needed.
        stems = [path.stem for path in args.csv]
        labels = stems if len(set(stems)) == len(stems) else [str(path) for path in args.csv]

    if args.serve:
        serve(args.csv, labels, port=args.serve_port)
    else:
        build(args.csv, labels, args.out, open_browser=not args.no_open)


if __name__ == "__main__":
    main()
