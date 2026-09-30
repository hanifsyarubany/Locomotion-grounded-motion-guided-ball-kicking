#!/usr/bin/env python3
"""Zip a small bundle of checkpoint files around a target iteration.

Given a training run directory that holds files like ``model_0074000.pt`` /
``model_0074000.onnx``, this grabs the target checkpoint plus its N nearest
neighbours on each side and writes them into a single zip, ready to ``scp`` down.

``holosoma_config.yaml`` and ``train.log`` are always added too (see ALWAYS_INCLUDE);
pass ``--no-defaults`` to leave them out.

The run dir can be a full path, just the run name (looked up under
``logs/UnifiedBallKickingEnhanced/``), or a unique substring of it.

Examples
--------
    # target 74000, +/- 2 neighbours (72k,73k,74k,75k,76k), both .pt and .onnx
    python scripts/zip_checkpoint_bundle.py 20260831_124621-stageB-skill017-h074-locomotion -t 74000

    # a unique substring is enough
    python scripts/zip_checkpoint_bundle.py skill017-h074 -t 74000

    # just the target, only onnx, custom output path
    python scripts/zip_checkpoint_bundle.py RUN_DIR -t 74k -n 0 --exts .onnx -o /tmp/one.zip

    # see what would be included without writing the zip
    python scripts/zip_checkpoint_bundle.py RUN_DIR -t 0074000 --dry-run

    # also tuck in extra files on top of the always-included ones
    python scripts/zip_checkpoint_bundle.py RUN_DIR -t 74000 --extra "*.yaml"
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import re
import sys
import zipfile
from pathlib import Path

CKPT_RE = re.compile(r"model_0*(\d+)\.(?:pt|onnx)$")

# Always bundled (if present), regardless of the target checkpoint.
ALWAYS_INCLUDE = ["holosoma_config.yaml", "train.log"]

# A bare run name (no path separators, not an existing dir) is looked up here,
# relative to this repo root. Override with --log-root.
DEFAULT_LOG_ROOT = "logs/UnifiedBallKickingEnhanced"
REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_target(raw: str) -> int:
    """Accept 74000, 0074000, '74k', 'model_0074000', 'model_0074000.pt'."""
    s = raw.strip().lower()
    m = re.search(r"model_0*(\d+)", s)
    if m:
        return int(m.group(1))
    m = re.fullmatch(r"(\d+)\s*k", s)
    if m:
        return int(m.group(1)) * 1000
    m = re.fullmatch(r"0*(\d+)", s)
    if m:
        return int(m.group(1))
    raise argparse.ArgumentTypeError(f"cannot parse checkpoint target: {raw!r}")


def resolve_run_dir(raw: Path, log_root: Path) -> Path:
    """Turn whatever the user typed into an actual run directory.

    Tries, in order: the path as given (abs or cwd-relative), the path under
    <log_root>, and finally a unique fuzzy match of <log_root>/*<name>*.
    """
    candidates = [raw, log_root / raw]
    for c in candidates:
        if c.is_dir():
            return c.resolve()

    # fuzzy: bare name that is a substring of exactly one run dir
    if log_root.is_dir() and len(raw.parts) == 1:
        matches = sorted(
            p for p in log_root.iterdir()
            if p.is_dir() and raw.name in p.name
        )
        if len(matches) == 1:
            print(f"note: resolved {raw.name!r} -> {matches[0]}", file=sys.stderr)
            return matches[0].resolve()
        if len(matches) > 1:
            names = "\n  ".join(m.name for m in matches)
            raise SystemExit(f"{raw.name!r} matches several runs under {log_root}:\n  {names}")

    tried = "\n  ".join(str(c) for c in candidates)
    raise SystemExit(f"could not find run directory. tried:\n  {tried}")


def discover_iters(run_dir: Path) -> list[int]:
    iters = set()
    for name in os.listdir(run_dir):
        m = CKPT_RE.match(name)
        if m:
            iters.add(int(m.group(1)))
    return sorted(iters)


def pick_iters(target: int, available: list[int], neighbors: int, step: int | None) -> list[int]:
    if target not in available:
        near = min(available, key=lambda x: abs(x - target)) if available else None
        raise SystemExit(
            f"target {target} has no checkpoint in this run.\n"
            f"  nearest available: {near}\n"
            f"  available range:   {available[0]}..{available[-1]}" if available
            else f"target {target}: no model_*.pt/.onnx files found here at all."
        )
    if neighbors <= 0:
        return [target]

    if step and step > 0:
        wanted = [target + k * step for k in range(-neighbors, neighbors + 1)]
        chosen = [i for i in wanted if i in set(available)]
    else:
        # index-based: N entries on each side in the sorted checkpoint list
        idx = available.index(target)
        lo = max(0, idx - neighbors)
        hi = min(len(available), idx + neighbors + 1)
        chosen = available[lo:hi]

    missing = [i for i in
               (target + k * step for k in range(-neighbors, neighbors + 1))
               if step and i not in set(available) and i >= 0] if step else []
    if missing:
        print(f"note: requested neighbours not present, skipped: {missing}", file=sys.stderr)
    return sorted(set(chosen) | {target})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path, 
                    help="run directory, OR just the run name (looked up under --log-root), "
                         "OR a unique substring of it")
    ap.add_argument("-t", "--target", required=True, type=parse_target,
                    help="target checkpoint (e.g. 74000, 0074000, 74k, model_0074000)")
    ap.add_argument("-n", "--neighbors", type=int, default=2,
                    help="how many checkpoints to include on EACH side of the target (default: 2)")
    ap.add_argument("--step", type=int, default=1000,
                    help="iteration spacing between checkpoints; 0 = use position in the sorted list "
                         "instead of arithmetic (default: 1000)")
    ap.add_argument("--exts", default=".pt,.onnx",
                    help="comma-separated extensions to include per checkpoint (default: .pt,.onnx)")
    ap.add_argument("--extra", action="append", default=[],
                    help="extra file or glob (relative to run_dir) to add; repeatable")
    ap.add_argument("--no-defaults", action="store_true",
                    help=f"do not auto-include {ALWAYS_INCLUDE}")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="output zip path (default: <runname>_ckpt<target>_bundle.zip in cwd)")
    ap.add_argument("--dry-run", action="store_true", help="list what would be zipped, write nothing")
    ap.add_argument("--log-root", type=Path, default=None,
                    help=f"base dir for bare run names (default: <repo>/{DEFAULT_LOG_ROOT})")
    args = ap.parse_args()

    log_root = (args.log_root or REPO_ROOT / DEFAULT_LOG_ROOT).expanduser()
    run_dir = resolve_run_dir(args.run_dir.expanduser(), log_root)

    exts = [e if e.startswith(".") else "." + e for e in args.exts.split(",") if e.strip()]
    available = discover_iters(run_dir)
    if not available:
        raise SystemExit(f"no model_*.pt / model_*.onnx files under {run_dir}")

    iters = pick_iters(args.target, available, args.neighbors, args.step)

    # Build the file list.
    files: list[Path] = []
    for it in iters:
        for ext in exts:
            p = run_dir / f"model_{it:07d}{ext}"
            if p.exists():
                files.append(p)
            else:
                print(f"note: missing {p.name}", file=sys.stderr)

    extra_patterns = list(args.extra)
    if not args.no_defaults:
        extra_patterns = ALWAYS_INCLUDE + extra_patterns

    for pat in extra_patterns:
        hits = sorted(run_dir.glob(pat))
        if not hits:
            print(f"note: extra pattern {pat!r} matched nothing", file=sys.stderr)
        files.extend(h for h in hits if h.is_file())

    # de-dup, keep order
    seen, ordered = set(), []
    for f in files:
        if f not in seen:
            seen.add(f)
            ordered.append(f)
    files = ordered

    if not files:
        raise SystemExit("nothing to zip")

    out = args.out or Path.cwd() / f"output/{run_dir.name}.zip"
    out = out.expanduser().resolve()

    total = 0
    print(f"run dir : {run_dir}")
    print(f"target  : {args.target}  ->  checkpoints {iters}")
    print(f"zip     : {out}")
    print("-" * 72)
    for f in files:
        sz = f.stat().st_size
        total += sz
        print(f"  {sz/1e6:8.2f} MB  {f.relative_to(run_dir)}")
    print("-" * 72)
    print(f"  {total/1e6:8.2f} MB  ({len(files)} files, uncompressed)")

    if args.dry_run:
        print("\n[dry-run] no zip written")
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in files:
            zf.write(f, arcname=str(Path(run_dir.name) / f.relative_to(run_dir)))
    tmp.replace(out)
    print(f"\nwrote {out}  ({out.stat().st_size/1e6:.2f} MB)")
    print("\nNext, from your LOCAL PC:")
    print(f"  scp <ssh-host>:{out} ~/Downloads/")


if __name__ == "__main__":
    main()
