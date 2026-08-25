"""Standalone tool: generate an interactive, browser-viewable skeleton scrubber for one or more
holosoma-format motion npz clips, for visually finding phase-boundary frame indices (e.g. where a
kick clip's locomotion-approach ends and the strike begins, or where the strike ends and standing
begins).

No simulator, no GPU, no training context needed -- this only reads the motion npz's own kinematic
data (body_pos_w, joint_pos/joint_vel) and renders it as a stick-figure skeleton client-side in the
browser. Companion asset ``motion_clip_scrubber_template.html`` (same directory) holds the actual
viewer UI; this script's job is purely extracting + shaping the per-clip data and splicing it in.

Usage
-----
    python motion_clip_scrubber.py CLIP1.npz [CLIP2.npz ...] [--labels NAME1 NAME2 ...] [--out FILE]

Example (this project's two current kick clips)::

    python motion_clip_scrubber.py \\
        ../../locomotion_and_motion_tracking/video2robot/data/video_011/npz_holosoma/robot_motion_track_1.npz \\
        ../../locomotion_and_motion_tracking/video2robot/data/video_012/npz_holosoma/robot_motion_track_1.npz \\
        --labels video_011 video_012 \\
        --out clip_scrubber.html

Then open the output HTML in any browser (or, in a remote/headless dev environment, publish it as
an Artifact / serve it) -- no server process, no dependency on this repo being reachable afterward.

Output: a single self-contained HTML file with every clip's motion data embedded inline. Typical
size is a few hundred KB per clip; fine to open directly, no build step needed downstream.
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path

import numpy as np

# Reduced skeleton for a readable stick-figure render: (name, parent-name-or-None). Matches this
# project's G1 body naming. Deliberately skips the ankle_roll_sphere_* contact-geometry links and
# finger sub-links (thumb/pinky) -- real bodies in the npz, just visual clutter for a phase-timing
# tool where only the gross limb structure matters.
SKELETON_CHAIN: list[tuple[str, str | None]] = [
    ("pelvis", None),
    ("left_hip_pitch_link", "pelvis"),
    ("left_hip_roll_link", "left_hip_pitch_link"),
    ("left_hip_yaw_link", "left_hip_roll_link"),
    ("left_knee_link", "left_hip_yaw_link"),
    ("left_ankle_pitch_link", "left_knee_link"),
    ("left_ankle_roll_link", "left_ankle_pitch_link"),
    ("right_hip_pitch_link", "pelvis"),
    ("right_hip_roll_link", "right_hip_pitch_link"),
    ("right_hip_yaw_link", "right_hip_roll_link"),
    ("right_knee_link", "right_hip_yaw_link"),
    ("right_ankle_pitch_link", "right_knee_link"),
    ("right_ankle_roll_link", "right_ankle_pitch_link"),
    ("waist_yaw_link", "pelvis"),
    ("torso_link", "waist_yaw_link"),
    ("left_shoulder_pitch_link", "torso_link"),
    ("left_shoulder_roll_link", "left_shoulder_pitch_link"),
    ("left_elbow_link", "left_shoulder_roll_link"),
    ("left_rubber_hand_link", "left_elbow_link"),
    ("right_shoulder_pitch_link", "torso_link"),
    ("right_shoulder_roll_link", "right_shoulder_pitch_link"),
    ("right_elbow_link", "right_shoulder_roll_link"),
    ("right_rubber_hand_link", "right_elbow_link"),
]

_LEG_KEYWORDS = ("hip", "knee", "ankle")


def _load_clip(npz_path: Path, label: str) -> dict:
    with np.load(npz_path) as d:
        body_names = [str(x) for x in d["body_names"]]
        joint_names = [str(x) for x in d["joint_names"]]
        body_pos_w = d["body_pos_w"]  # (T, num_bodies, 3)
        joint_pos = d["joint_pos"]  # (T, 7 + num_joints): root [xyz, wxyz] then joints
        joint_vel = d["joint_vel"]  # (T, 6 + num_joints): root [vel_xyz, vel_wxyz] then joints
        fps = float(np.atleast_1d(d["fps"])[0])

    n_joints = len(joint_names)
    # NB the off-by-7/off-by-6 gotcha: joint_pos/joint_vel carry the root DOFs as leading columns,
    # NOT present in joint_names. Skipping this shifts everything by one body/joint silently (it
    # did, once, earlier in this same project's own analysis -- see project memory) and produces a
    # plausible-looking but wrong skeleton, not a crash. Assert rather than risk repeating that.
    if joint_pos.shape[1] != n_joints + 7:
        raise ValueError(
            f"{npz_path}: joint_pos has {joint_pos.shape[1]} columns, expected "
            f"{n_joints + 7} (= {n_joints} joints + 7 root DOFs [xyz, wxyz]). "
            "This npz may not be in holosoma format."
        )
    if joint_vel.shape[1] != n_joints + 6:
        raise ValueError(
            f"{npz_path}: joint_vel has {joint_vel.shape[1]} columns, expected "
            f"{n_joints + 6} (= {n_joints} joints + 6 root DOFs [vel_xyz, vel_wxyz])."
        )

    # Body lookup by NAME, not a hardcoded index -- robust to a differently-ordered npz export
    # (same robot, different retargeting run) silently producing a wrong-but-plausible skeleton.
    name_to_idx = {n: i for i, n in enumerate(body_names)}
    missing = [n for n, _ in SKELETON_CHAIN if n not in name_to_idx]
    if missing:
        raise ValueError(
            f"{npz_path}: missing expected body name(s) {missing} -- this npz's skeleton doesn't "
            "match the G1 body naming SKELETON_CHAIN assumes. If this is a different robot, "
            "SKELETON_CHAIN needs updating for its body names before this tool will work."
        )

    chain_names = [n for n, _ in SKELETON_CHAIN]
    chain_local_idx = {n: i for i, n in enumerate(chain_names)}
    bones = [
        [chain_local_idx[parent], chain_local_idx[name]]
        for name, parent in SKELETON_CHAIN
        if parent is not None
    ]
    body_idx = [name_to_idx[n] for n in chain_names]

    T = body_pos_w.shape[0]
    pts = body_pos_w[:, body_idx, :].astype(np.float64)  # (T, len(chain), 3)
    # Root-center the XY so the figure doesn't drift off-canvas as the clip walks forward -- Z
    # (height) stays absolute so crouch/jump reads correctly.
    pelvis_xy = pts[:, 0, :2].copy()
    pts[:, :, 0] -= pelvis_xy[:, 0:1]
    pts[:, :, 1] -= pelvis_xy[:, 1:2]

    root_xyz = joint_pos[:, :3]
    root_speed = np.linalg.norm(np.gradient(root_xyz[:, :2], 1.0 / fps, axis=0), axis=-1)

    leg_cols = [i for i, n in enumerate(joint_names) if any(k in n.lower() for k in _LEG_KEYWORDS)]
    if not leg_cols:
        raise ValueError(
            f"{npz_path}: no joint names matched {_LEG_KEYWORDS} -- can't compute a leg-speed "
            "curve. Check joint_names, or extend _LEG_KEYWORDS for this robot's naming."
        )
    leg_speed = np.abs(joint_vel[:, 6:][:, leg_cols]).max(axis=1)

    strike_start, stand_start = _estimate_boundaries(leg_speed)

    return dict(
        name=label,
        fps=fps,
        T=int(T),
        points=np.round(pts, 4).tolist(),
        rootSpeed=np.round(root_speed, 4).tolist(),
        legSpeed=np.round(leg_speed, 4).tolist(),
        rootSpeedMax=float(root_speed.max()),
        legSpeedMax=float(leg_speed.max()),
        bones=bones,
        strikeStart=strike_start,
        standStart=stand_start,
    )


def _estimate_boundaries(leg_speed: np.ndarray) -> tuple[int, int]:
    """Rough auto-estimate of (strike_start, stand_start) from the leg-speed curve alone, as a
    starting point to drag/correct in the viewer -- NOT authoritative. Same heuristic used by hand
    earlier in this project: strike_start = last frame before peak where speed first climbs above
    25% of peak (searching backward from the peak); stand_start = first frame after peak where
    speed drops below 10% of peak and does not exceed it again for the rest of the clip (avoids
    latching onto a transient dip mid-swing)."""
    peak_idx = int(np.argmax(leg_speed))
    peak = float(leg_speed[peak_idx])
    if peak <= 0:
        return 0, len(leg_speed) - 1

    strike_start = 0
    for i in range(peak_idx, -1, -1):
        if leg_speed[i] < 0.25 * peak:
            strike_start = i
            break

    stand_start = len(leg_speed) - 1
    below = leg_speed < 0.10 * peak
    for i in range(peak_idx, len(leg_speed)):
        if below[i] and bool(below[i:].all()):
            stand_start = i
            break

    return strike_start, stand_start


def _build_html(npz_paths: list[Path], labels: list[str]) -> str:
    clips = {}
    for i, (path, label) in enumerate(zip(npz_paths, labels)):
        print(f"[{i}] loading {path} as '{label}'...", file=sys.stderr)
        clips[str(i)] = _load_clip(path, label)
        c = clips[str(i)]
        print(
            f"    {c['T']} frames @ {c['fps']:g}fps -- auto boundaries: "
            f"strike_start={c['strikeStart']}, stand_start={c['standStart']} (verify visually)",
            file=sys.stderr,
        )

    # bones/boneNames are identical across clips (same SKELETON_CHAIN) -- hoist once, keep each
    # clip's own dict focused on its actual per-frame data.
    bones = clips[next(iter(clips))]["bones"]
    bone_names = [n for n, _ in SKELETON_CHAIN]
    for c in clips.values():
        del c["bones"]

    data = dict(
        bones=bones,
        boneNames=bone_names,
        clips=clips,
        defaults={
            sk: {"strikeStart": c["strikeStart"], "standStart": c["standStart"]}
            for sk, c in clips.items()
        },
    )

    template_path = Path(__file__).with_name("motion_clip_scrubber_template.html")
    template = template_path.read_text()
    marker = "/*__MOTION_DATA__*/"
    if marker not in template:
        raise RuntimeError(f"template at {template_path} is missing the {marker} injection point")
    return template.replace(marker, json.dumps(data, separators=(",", ":")))


def serve(npz_paths: list[Path], labels: list[str], host: str = "127.0.0.1", port: int = 8787) -> None:
    """Build the page and serve it from memory over plain HTTP, for an SSH-only session with no
    $BROWSER forwarding (webbrowser.open has nothing to forward to there). Binds to 127.0.0.1, NOT
    0.0.0.0 -- reachable only through your own SSH tunnel (``ssh -L PORT:localhost:PORT host``, or
    VS Code Remote-SSH's automatic port-forwarding if that's what you're using -- it auto-detects a
    newly-listening local port and offers to forward + open it, no manual -L needed there), not
    exposed to the rest of the machine or network. Serves the SAME in-memory HTML on every path --
    deliberately not http.server's directory-listing SimpleHTTPRequestHandler, which would expose
    the whole current directory over HTTP, not just this one page."""
    import http.server

    html_bytes = _build_html(npz_paths, labels).encode("utf-8")

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


def build(npz_paths: list[Path], labels: list[str], out_path: Path, open_browser: bool = True) -> None:
    html = _build_html(npz_paths, labels)
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
    p.add_argument("npz", nargs="+", type=Path, help="one or more holosoma-format motion npz files")
    p.add_argument(
        "--labels", nargs="*", default=None,
        help="display label per clip, same order as npz args (default: filename stem of each)",
    )
    p.add_argument(
        "--out", type=Path, default=Path("motion_clip_scrubber.html"),
        help="output HTML path (default: ./motion_clip_scrubber.html)",
    )
    p.add_argument(
        "--no-open", action="store_true",
        help="don't open the result in a browser after writing it (default: open automatically). "
        "Relies on $BROWSER / webbrowser.open, which VS Code Remote/devcontainers forward to your "
        "LOCAL machine's default browser -- confirmed working in that setup; a plain headless SSH "
        "session with no such forwarding may instead print an error or silently do nothing, so "
        "this never blocks or fails the run either way.",
    )
    p.add_argument(
        "--serve", action="store_true",
        help="instead of writing a file and trying $BROWSER, start a local HTTP server on "
        "127.0.0.1 and block -- for a plain SSH session where there's no browser to forward to. "
        "Forward the port over your existing SSH connection (or via VS Code Remote-SSH's Ports "
        "tab) and open it in your OWN local browser. Ctrl+C to stop. Ignores --out/--no-open.",
    )
    p.add_argument(
        "--serve-port", type=int, default=8787,
        help="port for --serve (default: 8787)",
    )
    args = p.parse_args()

    for path in args.npz:
        if not path.exists():
            p.error(f"npz file not found: {path}")

    if args.labels:
        labels = args.labels
        if len(labels) != len(args.npz):
            p.error(f"--labels has {len(labels)} entries but {len(args.npz)} npz files were given")
    else:
        # holosoma npz exports are all literally named "robot_motion_track_1.npz" regardless of
        # clip identity (see this project's own directory layout), so bare path.stem collides for
        # any real multi-clip call. Auto-disambiguate using just enough trailing path components
        # to make every default label unique, rather than silently emitting duplicate tab labels.
        stems = [path.stem for path in args.npz]
        if len(set(stems)) == len(stems):
            labels = stems
        else:
            # Filenames collide (true for every clip in this project -- all literally named
            # robot_motion_track_1.npz). Find the single ancestor directory ONE COMPONENT deep
            # that actually differs across the given paths, rather than concatenating a full
            # (redundant, filename-repeating) path suffix.
            max_depth = max(len(path.parts) for path in args.npz)
            labels = stems  # fallback if nothing ever disambiguates
            for depth in range(2, max_depth + 1):
                candidate = [
                    path.parts[-depth] if depth <= len(path.parts) else path.stem
                    for path in args.npz
                ]
                if len(set(candidate)) == len(candidate):
                    labels = candidate
                    break
            print(f"note: default labels collided on filename; using ancestor dirs: {labels}", file=sys.stderr)

    if args.serve:
        serve(args.npz, labels, port=args.serve_port)
    else:
        build(args.npz, labels, args.out, open_browser=not args.no_open)


if __name__ == "__main__":
    main()
