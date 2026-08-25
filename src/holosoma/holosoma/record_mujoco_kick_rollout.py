"""Thin, stdlib-only wrapper around `mujoco_kick_rollout_worker.py`, importable from the training
process (which does not have RoboJuDo installed -- that's why the actual work happens in a
subprocess under the separate `robojudo` conda env; see that file's module docstring).

Serializes concurrent rollouts cluster-wide via a simple file lock: this workspace routinely runs
several Stage B/C training processes at once, and without a lock a checkpoint-save coincidence
could launch that many simultaneous CPU-bound MuJoCo render subprocesses. A busy lock just means
"skip this checkpoint's rollout" -- never blocks or waits.

Never raises -- every failure mode is caught, logged via loguru, and turned into a `False` return.
"""

from __future__ import annotations

import os
import subprocess

from loguru import logger

from holosoma.utils.rollout_lock import (
    DEFAULT_STALE_LOCK_TIMEOUT_S,
    acquire_global_lock,
    release_global_lock,
)

# "mujoco_media/" prefix: wandb groups logged media keys into UI panel sections by "/"-delimited
# prefix (same convention as "train/loss" vs "eval/loss") -- this puts every RoboJuDo/MuJoCo
# sim2sim rollout video under its own "mujoco_media" section, separate from the "isaacsim_media"
# section the live-training IsaacSim recorders log under (see unified_manager.py's
# _setup_task_video_recording). fast_sac_agent.py's per-skill f-string
# (f"{MUJOCO_KICK_WANDB_KEY} - Skill {i}") inherits this prefix automatically.
MUJOCO_KICK_WANDB_KEY = "mujoco_media/Training rollout - MuJoCo Kick"

# 2026-08-13: walk-then-trigger variant of the same rollout (see mujoco_kick_rollout_worker.py's
# own docstring for --walk-s) -- own wandb key so it shows as a separate video, same "mujoco_media/"
# prefix/section as every other sim2sim rollout.
MUJOCO_KICK_HANDOFF_WANDB_KEY = "mujoco_media/Training rollout - MuJoCo Kick (Locomotion Handoff)"

ROBOJUDO_PYTHON = os.environ.get(
    "HOLOSOMA_ROBOJUDO_PYTHON",
    "/workspaces/isaaclab_arena/submodules/workspaces/conda_env/robojudo/bin/python",
)
WORKER_SCRIPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mujoco_kick_rollout_worker.py")

DEFAULT_LOCK_PATH = os.environ.get("HOLOSOMA_SIM2SIM_LOCK_PATH", "/tmp/holosoma_sim2sim_rollout.lock")
# Own lock file, separate from DEFAULT_LOCK_PATH above -- same rationale
# record_mujoco_locomotion_rollout.py's own DEFAULT_LOCK_PATH docstring gives for the walk rollout:
# both this and the settle-then-trigger rollout fire from the same checkpoint-save event at the
# same cadence, so sharing one lock would make whichever starts first win and the other always get
# skipped, defeating the point of running both.
DEFAULT_HANDOFF_LOCK_PATH = os.environ.get(
    "HOLOSOMA_SIM2SIM_KICK_HANDOFF_LOCK_PATH", "/tmp/holosoma_sim2sim_kick_handoff_rollout.lock"
)


def record_kick_rollout(
    onnx_path: str,
    output_video_path: str,
    *,
    with_ball: bool = True,
    skill_id: int = 0,
    kick_aim_enabled: bool = False,
    kick_aim_theta_deg: float = 0.0,
    kick_aim_theta_ref_deg: float = 45.0,
    # 2026-08-16: raised from 60.0 -- see record_mujoco_locomotion_rollout.py's own timeout_s
    # comment for the full rationale (host contention, not a hang; this just gives a genuinely
    # slow-but-progressing rollout enough time to finish instead of getting killed mid-recording).
    timeout_s: float = 180.0,
    settle_s: float = 1.5,
    walk_s: float = 0.0,
    forward_speed: float = 0.5,
    hold_s: float = 8.0,
    lock_path: str = DEFAULT_LOCK_PATH,
    stale_lock_timeout_s: float = DEFAULT_STALE_LOCK_TIMEOUT_S,
) -> bool:
    """Record a single-env MuJoCo sim2sim kick rollout of `onnx_path` (RoboJuDo pipeline) and save
    it to `output_video_path`. `with_ball` (default True) spawns a real ball and feeds
    kick_ball_pos_b/kick_target_pos_b -- set False to revert to the original no-ball, zero-
    observation rollout, appropriate for Stage B checkpoints that never trained on a real ball
    (see `mujoco_kick_rollout_worker.py` and FastSACConfig.mujoco_kick_rollout_with_ball).

    `skill_id` (default 0) selects which of the ONNX's embedded motion skills to kick, for
    checkpoints trained under holosoma's N-skill mechanism -- see
    `robojudo.policy.unified_loco_kick_policy`'s module docstring. Meaningless (harmlessly ignored,
    always skill 0) for checkpoints trained without it.

    `kick_aim_enabled`/`kick_aim_theta_deg`/`kick_aim_theta_ref_deg` (2026-08-23, azimuth-aim
    refactor bugfix): pass `kick_aim_enabled=True` for a checkpoint whose selected skill has
    `SkillConfig.kick_aim_enabled=True` -- without this, obs[157:159] silently falls back to the
    PRE-refactor raw world-frame target_pos_b transform, a ~15-18x out-of-distribution magnitude
    for a checkpoint actually trained on the bounded [-kick_aim_theta_max_deg/theta_ref_deg,
    +.../...] command (see `mujoco_kick_rollout_worker.py::_install_ball_observation_patch`'s own
    docstring for the full explanation). `kick_aim_theta_deg` (default 0.0) is the fixed value fed
    when enabled -- 0.0 aims straight along the skill's own calibrated nominal bearing, the natural
    choice for a repeatable video. False (default) preserves the exact prior behavior for any
    checkpoint NOT trained with kick_aim_enabled.

    `walk_s` (default 0.0 = original settle-then-trigger behavior, byte-identical) / `forward_speed`:
    see mujoco_kick_rollout_worker.py's own `--walk-s`/`--forward-speed` docstrings -- when
    `walk_s > 0`, replaces the zero-velocity settle phase with a forward walk, triggering the kick
    the instant the walk command cuts to zero (still carrying momentum), instead of from a settled
    stand. Used by the "MuJoCo Kick (Locomotion Handoff)" rollout, a simple actor-robustness check
    -- NOT a port of the IsaacSim training env's mid-episode entry-point search.

    Serialized cluster-wide via a lock file at `lock_path` -- if already held, returns False
    immediately without launching anything (does not block/wait). Pass `DEFAULT_HANDOFF_LOCK_PATH`
    (own lock, separate from `DEFAULT_LOCK_PATH`) when calling this for the walk-then-trigger
    variant alongside the original, so the two run concurrently instead of contending.

    Never raises. Returns True iff `output_video_path` exists on return.
    """
    token = acquire_global_lock(lock_path, stale_lock_timeout_s)
    if token is None:
        logger.warning(f"[sim2sim] Global rollout lock busy -- skipping rollout for {onnx_path}.")
        return False

    try:
        os.makedirs(os.path.dirname(output_video_path) or ".", exist_ok=True)
        argv = [
            ROBOJUDO_PYTHON, WORKER_SCRIPT_PATH,
            "--onnx-path", onnx_path,
            "--output-video-path", output_video_path,
            "--settle-s", str(settle_s),
            "--walk-s", str(walk_s),
            "--forward-speed", str(forward_speed),
            "--hold-s", str(hold_s),
            "--skill-id", str(skill_id),
        ]
        if not with_ball:
            argv.append("--no-ball")
        if kick_aim_enabled:
            argv += [
                "--kick-aim-enabled",
                "--kick-aim-theta-deg", str(kick_aim_theta_deg),
                "--kick-aim-theta-ref-deg", str(kick_aim_theta_ref_deg),
            ]
        try:
            result = subprocess.run(argv, timeout=timeout_s, capture_output=True, text=True)
        except subprocess.TimeoutExpired:
            logger.warning(f"[sim2sim] MuJoCo kick rollout timed out after {timeout_s:.0f}s for {onnx_path}")
            return False

        if result.returncode != 0:
            logger.warning(
                f"[sim2sim] MuJoCo kick rollout worker exited {result.returncode} for {onnx_path}\n"
                f"stdout(tail): {result.stdout[-2000:]}\nstderr(tail): {result.stderr[-2000:]}"
            )
            return False

        if not os.path.exists(output_video_path):
            logger.warning(f"[sim2sim] MuJoCo kick rollout worker exited 0 but no video at {output_video_path}")
            return False

        return True

    except Exception:
        logger.exception(f"[sim2sim] Unhandled error recording MuJoCo kick rollout for {onnx_path}")
        return False
    finally:
        release_global_lock(lock_path, token)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Manual test of record_kick_rollout().")
    parser.add_argument("--onnx-path", required=True)
    parser.add_argument("--output-video-path", default="/tmp/holosoma_mujoco_kick_rollout_test.mp4")
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--no-ball", action="store_true", help="Revert to the original no-ball rollout.")
    parser.add_argument("--skill-id", type=int, default=0, help="Which embedded motion skill to kick.")
    parser.add_argument(
        "--walk-s", type=float, default=0.0,
        help="Forward-walk duration before the trigger (0.0 = original settle-then-trigger).",
    )
    parser.add_argument("--forward-speed", type=float, default=0.5, help="Walk-phase forward speed, m/s.")
    ns = parser.parse_args()

    ok = record_kick_rollout(
        onnx_path=ns.onnx_path,
        output_video_path=ns.output_video_path,
        timeout_s=ns.timeout_s,
        with_ball=not ns.no_ball,
        skill_id=ns.skill_id,
        walk_s=ns.walk_s,
        forward_speed=ns.forward_speed,
    )
    print(f"record_kick_rollout: {'SUCCESS' if ok else 'FAILED'} -> {ns.output_video_path}")
    raise SystemExit(0 if ok else 1)
