#!/usr/bin/env python3
"""Copy a trained checkpoint's .onnx into RoboJuDo's skill_library, preserving its run-directory
name so multiple checkpoints/runs never collide on a flattened filename.

Usage:
    python scripts/copy_checkpoint_to_skill_library.py \
        20260901_115721-stageC-skill012-h074-locomotion/model_0390000.onnx

Copies:
    logs/UnifiedBallKickingEnhanced/<that path>
 -> humanoid_deployment/RoboJuDo/assets/motions/g1/football_play/skill_library/<that path>
creating the destination's parent directory (the run-name subfolder) if it doesn't exist yet.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

SOURCE_BASE = Path(
    "/workspaces/isaaclab_arena/submodules/workspaces/playground/unified_ball_kick_enhanced/"
    "logs/UnifiedBallKickingEnhanced"
)
DEST_BASE = Path(
    "/workspaces/isaaclab_arena/submodules/workspaces/humanoid_deployment/RoboJuDo/assets/motions/"
    "g1/football_play/skill_library"
)


def copy_checkpoint(relative_path: str) -> Path:
    """Copies SOURCE_BASE/relative_path to DEST_BASE/relative_path, creating the destination's
    parent directory as needed. Returns the destination path. Raises FileNotFoundError if the
    source doesn't exist."""
    src = SOURCE_BASE / relative_path
    if not src.is_file():
        raise FileNotFoundError(f"source checkpoint not found: {src}")

    dest = DEST_BASE / relative_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "relative_path",
        help="Path relative to logs/UnifiedBallKickingEnhanced, "
        "e.g. '20260901_115721-stageC-skill012-h074-locomotion/model_0390000.onnx'",
    )
    args = parser.parse_args()

    try:
        dest = copy_checkpoint(args.relative_path)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Copied to: {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
