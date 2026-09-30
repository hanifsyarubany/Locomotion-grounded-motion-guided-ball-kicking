#!/usr/bin/env python3
"""Standalone MuJoCo sim2sim locomotion COMMAND-TRACKING + push-recovery metrics worker, built for
Table 5 (documents/proposal/benchmark_plan.md) -- "Locomotion generality with the skill library
attached." NOT part of the `holosoma` package's own runtime -- only ever invoked as a subprocess
via the `robojudo` conda env's interpreter (see `record_mujoco_locomotion_metrics_scan.py`),
mirroring `mujoco_kick_survival_scan.py` / `mujoco_locomotion_rollout_worker.py`'s architecture.

WHY THIS EXISTS, DISTINCT FROM mujoco_locomotion_rollout_worker.py: that script is a FIXED,
forward-only VIDEO recorder for wandb media panels (one episode, no lateral/yaw command at all, no
metrics -- it returns a video success/failure bool). This script measures actual command-tracking
accuracy across all three command axes (v_x forward AND backward, v_y lateral, omega_z yaw) plus
push-recovery, over several independent episodes per axis, and emits one JSON summary -- no video.

COMMAND MAGNITUDES: training samples lin_vel_x / lin_vel_y / ang_vel_yaw uniformly over [-1.0, 1.0]
each (config_values/loco/g1/command.py's own command_ranges) -- this worker's defaults (forward
+0.8, backward -0.8, lateral +0.5 m/s, yaw +0.5 rad/s) are all comfortably in-distribution, not
arbitrary. +0.8 also matches mujoco_locomotion_rollout_worker.py's own documented "falls stopping
from a fast walk" probe speed, so the forward-axis number here is directly comparable to that
existing rollout's own qualitative finding.

PUSH MAGNITUDE: reuses this project's own body_push_force_max (80.0 N) and body_push_duration_max_s
(0.20 s) -- task_config's PUSH RANDOMIZATION CONFIG section, MultiSkillConfig's own fields -- as
defaults: the upper/hardest end of what the policy was actually trained against, applied to
"pelvis" (one of DEFAULT_BODY_PUSH_BODIES in managers/randomization/terms/locomotion.py), not an
arbitrary new magnitude. NOTE: this is a NEW, discrete push-then-measure eval, not an invocation of
that IsaacLab randomization term -- that term is unconditionally disabled during evaluation
(env.is_evaluating) and has no MuJoCo/RoboJuDo equivalent; this script independently reimplements
"apply a force via mj xfrc_applied, then check recovery" using ITS OWN magnitude reference point.
Verified live (2026-09-07) before writing this file: an 80 N/0.2 s pelvis push visibly accelerates
the base (vx 0 -> 0.39 m/s) and xfrc_applied persists across mj steps until explicitly zeroed (not
auto-cleared per step) -- both load-bearing assumptions below are empirically confirmed, not
guessed from the MuJoCo docs.

VELOCITY GROUND TRUTH: env.base_lin_vel / env.base_ang_vel (robojudo/environment/base_env.py,
backed by mujoco_env.py's own body-frame-rotated qvel) -- the SAME quantities
UnifiedLocoKickPolicy itself reads as its loco_command_lin_vel / loco_command_ang_vel observation
inputs (robojudo/policy/unified_loco_kick_policy.py), not independently re-derived from raw qvel --
so there is no frame-convention mismatch between what the policy was told to track and what this
worker scores it against. Confirmed live: nonzero and responding correctly to a step command
before this script was written (env.base_lin_vel goes from [0,0,0] to a real forward speed under a
sustained +vx command).

FALL / RECOVERY THRESHOLD: 0.70 m base height -- the same "low_height" floor already referenced
elsewhere in this project (see MultiSkillConfig.kick_state_init_prob's own docstring, which cites
locomotion's 0.70 m low_height floor), not a new arbitrary number.

Each of the 4 velocity axes and the push test runs `--num-episodes` (default 5) independent
episodes, resetting to the standing keyframe between each. Falls are EXCLUDED from that axis's MAE
(reported separately as fall_rate) so a single topple doesn't dominate a tracking-quality number --
mirrors this project's existing convention of keeping "did it fall" and "how well did it
track/aim" as separate metrics (kick_topple_frac vs shot-error, per benchmark_plan.md).

CONTROL RATE: 50 Hz (--control-hz), matching mujoco_locomotion_rollout_worker.py's own documented
RL control rate and empirically confirmed above (10 steps at this rate == the 0.20 s push duration
producing the expected physical effect).

+-- Terrain traversal (Table 5's 5th row) is NOT implemented here -- no terrain-capable MuJoCo
scene exists yet for this robot (only flat scene_g1_29dof*.xml assets). `terrain_traversal_success`
is always emitted as `null` with a `"not_implemented"` note, so Table 5's row stays honestly blank
rather than silently fabricated. Building it needs a new MJCF terrain asset -- tracked separately,
not attempted here.

Usage (manual test):
    /workspaces/isaaclab_arena/submodules/workspaces/conda_env/robojudo/bin/python \
        mujoco_locomotion_metrics_worker.py --onnx-path /path/to/model.onnx \
        --output-json /tmp/loco_metrics.json --num-episodes 3
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback

ROBOJUDO_REPO = "/workspaces/isaaclab_arena/submodules/workspaces/humanoid_deployment/RoboJuDo"
FLAT_SCENE = ROBOJUDO_REPO + "/assets/robots/g1/holosoma_model/scene_g1_29dof.xml"
# 2026-09-09: terrain-traversal test scene (Table 5's 6th row, see _run_terrain_traversal below).
# Same robot/keyframe as FLAT_SCENE; only the floor geom differs (hfield instead of plane) -- see
# that scene's own file-header comment for the elevation-mapping math, empirically confirmed
# against a live MuJoCo instance before it was written. Used for EVERY test in this worker, not
# just terrain -- hfield_data is set to all-1.0 (the flat baseline, see TERRAIN_FLAT_DATA below)
# before the velocity-axis/push-recovery tests, which is byte-equivalent ground behavior to
# FLAT_SCENE's own plane (confirmed live: both settle a real policy at the same base height).
# Kept as ONE scene rather than two so hfield_data (already-loaded, no recompile) can just be
# rewritten between tests instead of tearing down and rebuilding the whole MuJoCo model.
TERRAIN_SCENE = ROBOJUDO_REPO + "/assets/robots/g1/holosoma_model/scene_g1_29dof_terrain.xml"

# See DEFAULT_BODY_PUSH_BODIES in config_values/../managers/randomization/terms/locomotion.py --
# "pelvis" is the root/base body there too, the natural single-body choice for a whole-body
# push-recovery test (the limb entries in that tuple model collision, not a shove).
PUSH_BODY_NAME = "pelvis"

# See module docstring's FALL / RECOVERY THRESHOLD note.
DEFAULT_FALL_HEIGHT_M = 0.70

# 2026-09-09: SUSTAINED, not instantaneous -- matches training's OWN locomotion-mode fall
# definition exactly (config_values/unified/g1/termination.py's _low_height_term:
# base_height_below_threshold_sustained_post_flip_graced, min_height=0.70,
# consecutive_steps=10 -- the plain, non-kick-graced 10-step value, since every test in this
# worker runs task_mode=locomotion throughout, never touching the post-flip grace window that
# param also guards). Root-caused 2026-09-09 on the ORIGINAL single-tick `min(z_series) >=
# fall_height_m` check: instrumented 6 rough-terrain "failures" directly (tilt magnitude +
# per-tick z) and every one was an ordinary single-support gait-cycle height dip -- tilt stayed
# 1-3 degrees, z recovered above 0.70 within 2-10 ticks, gait continued normally for the rest of
# the trial. None was a topple. The single-tick check was measuring "did this tile's terrain
# height plus normal gait oscillation ever cross an absolute threshold", a materially stricter and
# different question than "did the robot fall over", which is what base_height_below_threshold_
# _sustained's own docstring is explicit about: "a brief height dip... doesn't kill the episode;
# only *staying* low does. This makes it safe to set the threshold above a settled-crouch height
# without punishing legitimate transients." Same counter semantics reproduced here: a per-tick
# below-threshold streak counter, reset to 0 on ANY tick at/above threshold, fall confirmed only
# once the streak reaches FALL_SUSTAINED_STEPS.
FALL_SUSTAINED_STEPS = 10


class _SustainedFallTracker:
    """Per-trial state for the sustained-below-threshold fall check, mirroring
    base_height_below_threshold_sustained's own counter exactly (see FALL_SUSTAINED_STEPS'
    citation above): a below-threshold streak that resets to 0 on any at/above-threshold tick,
    with `fell` latching True (and staying True) the instant the streak reaches
    `consecutive_steps`. One instance per trial -- construct fresh in each episode's loop, same
    "no cross-trial state" contract every per-trial counter in this worker already follows."""

    def __init__(self, fall_height_m: float, consecutive_steps: int = FALL_SUSTAINED_STEPS) -> None:
        self._fall_height_m = fall_height_m
        self._consecutive_steps = consecutive_steps
        self._streak = 0
        self.fell = False

    def update(self, z: float) -> None:
        if z < self._fall_height_m:
            self._streak += 1
            if self._streak >= self._consecutive_steps:
                self.fell = True
        else:
            self._streak = 0


# ------------------------------------------------------------------------------------------------
# TERRAIN TRAVERSAL (Table 5's 6th row, added 2026-09-09)
# ------------------------------------------------------------------------------------------------
# Reproduces this project's own real terrain generation -- simulator/shared/terrain.py's
# _flat_terrain_func/_rough_terrain_func/_low_obstacles_terrain_func -- INLINE below rather than
# importing that module, because Terrain (the class those methods live on) unconditionally
# `import trimesh` at module load, and trimesh is not installed in the `robojudo` conda env this
# worker runs under (confirmed live, 2026-09-09) -- the mesh-conversion step trimesh is for is not
# even needed here (this worker wants the raw height_field_raw array for a MuJoCo hfield, not a
# trimesh). holosoma.utils.terrain_utils.SubTerrain itself has no such dependency and DOES import
# cleanly here (confirmed live) -- used below for its own height_field_raw buffer, exactly the way
# Terrain.make_terrain() uses it.
#
# PROPORTIONS/TIERS: {flat: 0.4, rough: 0.45, low_obstacles: 0.15} are this project's OWN real
# terrain_config values for the unified/kick training runs (read directly from a real run's saved
# holosoma_config.yaml, 2026-09-09 -- NOT the docstring defaults elsewhere in this codebase, which
# describe a different, kick-eligible-only carve-out at different proportions). light_rough,
# rough_slope, and smooth_slope are all 0.0 proportion in that same real config, so they are
# omitted here too -- generating a tier the actual training run never included would misrepresent
# what "the terrain the substrate trains on" means for this comparison.
TERRAIN_TYPE_PROPORTIONS = {"flat": 0.4, "rough": 0.45, "low_obstacles": 0.15}
# randomized_terrain()'s own per-tile difficulty draw (simulator/shared/terrain.py) -- reproduced
# exactly, not re-chosen, so a difficulty-weighted comparison stays meaningful against training.
TERRAIN_DIFFICULTY_CHOICES = (0.5, 0.75, 0.9)
TERRAIN_HORIZONTAL_SCALE = 0.1  # meters/pixel -- config_values/terrain.py's own default, and this
# project's own real resolved training config (both agree).
TERRAIN_VERTICAL_SCALE = 0.005  # meters/height-unit -- same source, same agreement.
TERRAIN_TILE_M = 8.0  # terrain_length == terrain_width in that same real config.
TERRAIN_TILE_PIXELS = int(round(TERRAIN_TILE_M / TERRAIN_HORIZONTAL_SCALE))  # 80 -- must match
# scene_g1_29dof_terrain.xml's own <hfield nrow="80" ncol="80">.
# Worst-case deviation across all three tiers at the hardest difficulty (0.9): _rough_terrain_func
# reaches -0.075m (see below), _low_obstacles_terrain_func reaches -0.03m, flat is always 0 -- so
# 0.08 covers all three with margin. Must match scene_g1_29dof_terrain.xml's own <hfield ...
# size="4 4 0.08 0.1">'s third value (elevation_z) exactly -- see that scene's own file-header
# comment for the world_z = hfield_data * elevation_z mapping this constant is sized against.
TERRAIN_Z_RANGE_M = 0.08


def _generate_terrain_tile_m(rng: "np.random.Generator", terrain_type: str, difficulty: float) -> "np.ndarray":
    """Returns one TERRAIN_TILE_PIXELS x TERRAIN_TILE_PIXELS tile of REAL height, in meters,
    relative to nominal (undisturbed) ground level -- i.e. what simulator/shared/terrain.py's own
    height_field_raw * vertical_scale would give, for the SAME formula that module's
    _{terrain_type}_terrain_func methods use (cited per-tier below). Always <= 0 (every tier this
    project actually uses is a depression, never a rise above nominal ground -- see each cited
    function's own docstring)."""
    import numpy as np

    shape = (TERRAIN_TILE_PIXELS, TERRAIN_TILE_PIXELS)
    if terrain_type == "flat":
        # simulator/shared/terrain.py:329, _flat_terrain_func -- height_field_raw[:] = 0.0.
        return np.zeros(shape, dtype=np.float64)
    if terrain_type == "rough":
        # simulator/shared/terrain.py:341-357, _rough_terrain_func. max_height = 0.025*d/0.9;
        # uniform(-max_height*2 - 0.025, -0.025) -- e.g. at d=0.9: uniform(-0.075, -0.025).
        max_height = 0.025 * difficulty / 0.9
        return rng.uniform(-max_height * 2 - 0.025, -0.025, shape)
    if terrain_type == "low_obstacles":
        # simulator/shared/terrain.py:518-544, _low_obstacles_terrain_func. 30 square depressions
        # of side terrain.width//10 (== TERRAIN_TILE_PIXELS//10 here), each at a FIXED depth
        # -max_height (not a random range per-obstacle -- only WHICH cells are depressed is
        # randomized), max_height = 0.03*d/0.9.
        max_height = 0.03 * difficulty / 0.9
        obst_size = TERRAIN_TILE_PIXELS // 10
        tile = np.zeros(shape, dtype=np.float64)
        xs = rng.integers(0, TERRAIN_TILE_PIXELS - obst_size, (30,))
        ys = rng.integers(0, TERRAIN_TILE_PIXELS - obst_size, (30,))
        for x, y in zip(xs, ys):
            tile[x : x + obst_size, y : y + obst_size] = -max_height
        return tile
    raise ValueError(f"Unknown terrain_type {terrain_type!r} -- not in TERRAIN_TYPE_PROPORTIONS.")


def _terrain_tile_to_hfield_data(tile_m: "np.ndarray") -> "np.ndarray":
    """Converts a real-meters height tile (from _generate_terrain_tile_m, always in
    [-TERRAIN_Z_RANGE_M, 0]) to MuJoCo's own normalized-[0,1] hfield_data convention. See
    scene_g1_29dof_terrain.xml's own file-header comment for the full world_z = hfield_data *
    elevation_z derivation this inverts: data=1.0 (a flat tile) lands at world z=0, the SAME
    ground reference FLAT_SCENE's plane and this project's own default_stand keyframe both use.
    Clipped to [0, 1] as a safety margin against float rounding at the exact tier boundaries, not
    because any tier is expected to exceed TERRAIN_Z_RANGE_M in practice."""
    import numpy as np

    return np.clip((tile_m + TERRAIN_Z_RANGE_M) / TERRAIN_Z_RANGE_M, 0.0, 1.0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx-path", required=True)
    parser.add_argument("--output-json", required=True, help="Where to write the final metrics summary.")
    parser.add_argument("--num-episodes", type=int, default=5, help="Independent episodes per axis/push test.")
    parser.add_argument("--control-hz", type=float, default=50.0)
    parser.add_argument("--settle-s", type=float, default=1.0, help="Zero-command settle time after reset.")
    parser.add_argument("--command-s", type=float, default=3.0, help="Total duration the test command is held.")
    parser.add_argument(
        "--warmup-s", type=float, default=1.0,
        help="Leading portion of --command-s excluded from the MAE window (lets the tracking "
        "transient settle before scoring) -- measurement window is (command_s - warmup_s) long.",
    )
    parser.add_argument("--fall-height-m", type=float, default=DEFAULT_FALL_HEIGHT_M)
    parser.add_argument("--forward-speed", type=float, default=0.8)
    parser.add_argument("--backward-speed", type=float, default=-0.8)
    parser.add_argument("--lateral-speed", type=float, default=0.5)
    parser.add_argument("--yaw-rate", type=float, default=0.5)
    parser.add_argument("--push-force-n", type=float, default=80.0, help="Matches body_push_force_max.")
    parser.add_argument("--push-duration-s", type=float, default=0.20, help="Matches body_push_duration_max_s.")
    parser.add_argument("--push-recovery-window-s", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _build_pipeline(onnx_path: str):
    sys.path.insert(0, ROBOJUDO_REPO)
    import robojudo.config.g1  # noqa: F401 -- registers config
    import robojudo.pipeline  # noqa: F401 -- registers pipeline
    from robojudo.config import cfg_registry

    cfg = cfg_registry.get("g1_unified_loco_kick")()
    cfg.policy.onnx_path = onnx_path
    # TERRAIN_SCENE, not FLAT_SCENE -- used for every test in this worker (see that constant's own
    # comment for why one scene, not two, and why hfield_data=all-1.0 is byte-equivalent ground
    # behavior to FLAT_SCENE's own plane for the velocity-axis/push-recovery tests).
    cfg.env.xml = TERRAIN_SCENE
    pl = getattr(robojudo.pipeline, cfg.pipeline_type)(cfg=cfg)
    pl.env.viewer.is_alive = False
    # Same override mujoco_locomotion_rollout_worker.py uses -- no joystick/game controller is
    # attached to this headless subprocess, and without this the policy's own controller-polling
    # path throws on every step trying to read one.
    pl.policy.policy._update_velocity_command = lambda cd, ball_pos_b=None: None
    # Flat baseline (hfield_data == 1.0 everywhere -> world_z == 0 everywhere, see TERRAIN_SCENE's
    # own comment) for every test EXCEPT _run_terrain_traversal, which overwrites this per episode
    # and restores it afterward. Set once here, not per-_reset() call: hfield_data is untouched by
    # mj_resetDataKeyframe (confirmed live) so it persists across every episode until deliberately
    # rewritten.
    pl.env.model.hfield_data[:] = 1.0
    return pl


def _reset(pl) -> None:
    import mujoco

    env = pl.env
    mujoco.mj_resetDataKeyframe(env.model, env.data, 0)
    mujoco.mj_forward(env.model, env.data)
    env.update()


def _step_n(pl, n: int, vx: float, vy: float, wz: float) -> None:
    inner = pl.policy.policy
    inner.lin_vel_command[:] = (vx, vy)
    inner.ang_vel_command = wz
    for _ in range(n):
        pl.step()


def _run_velocity_axis(pl, args: argparse.Namespace, vx: float, vy: float, wz: float) -> dict:
    import numpy as np

    settle_steps = int(round(args.settle_s * args.control_hz))
    warmup_steps = int(round(args.warmup_s * args.control_hz))
    total_steps = int(round(args.command_s * args.control_hz))
    measure_steps = total_steps - warmup_steps
    if measure_steps <= 0:
        raise ValueError(f"--command-s ({args.command_s}) must exceed --warmup-s ({args.warmup_s}).")

    env = pl.env
    per_episode_mae: list[float] = []
    fall_count = 0
    for _ in range(args.num_episodes):
        _reset(pl)
        _step_n(pl, settle_steps, 0.0, 0.0, 0.0)

        fall_tracker = _SustainedFallTracker(args.fall_height_m)
        errs: list[float] = []
        inner = pl.policy.policy
        inner.lin_vel_command[:] = (vx, vy)
        inner.ang_vel_command = wz
        for step_i in range(total_steps):
            pl.step()
            fall_tracker.update(float(env.data.qpos[2]))
            if step_i >= warmup_steps:
                achieved_vx, achieved_vy, _ = env.base_lin_vel
                achieved_wz = float(env.base_ang_vel[2])
                err = abs(achieved_vx - vx) + abs(achieved_vy - vy) if (vx != 0.0 or vy != 0.0) else abs(achieved_wz - wz)
                errs.append(err)

        if fall_tracker.fell:
            fall_count += 1
        else:
            per_episode_mae.append(float(np.mean(errs)))

    return {
        "mae": float(np.mean(per_episode_mae)) if per_episode_mae else None,
        "fall_rate": fall_count / args.num_episodes,
        "num_episodes": args.num_episodes,
        "num_falls": fall_count,
    }


def _run_terrain_traversal(pl, args: argparse.Namespace) -> dict:
    """Table 5's 6th row. Each episode samples a fresh terrain TYPE (weighted by
    TERRAIN_TYPE_PROPORTIONS, this project's own real training mix, not an invented one) and
    DIFFICULTY (uniform over TERRAIN_DIFFICULTY_CHOICES, exactly matching randomized_terrain()'s
    own per-tile draw), writes it into the shared hfield via MjModel.hfield_data (no recompile),
    resets to the standing keyframe, and commands the SAME forward speed/duration as the
    v_x_forward axis test above -- success = FALL_SUSTAINED_STEPS consecutive ticks below
    args.fall_height_m never occurs (see that constant's own citation: training's own locomotion
    fall definition, not an instantaneous dip).

    SUSTAINED, NOT INSTANTANEOUS -- root-caused 2026-09-09. An earlier version used a single-tick
    `min(z_series) >= fall_height_m` check and reported "rough" terrain traversal at ~6% (vs ~88%
    for the sparser "low_obstacles" tier, backwards from any reasonable difficulty ordering).
    Instrumented the failures directly (per-tick z AND tilt magnitude): every one was an ordinary
    single-support gait-cycle height dip on a tile already ~3-5cm lower than flat ground -- tilt
    stayed 1-3 degrees (essentially upright), z recovered above threshold within 2-10 ticks, and
    the SAME trial then walked normally for the rest of its window. None was a topple. The
    single-tick check was conflating "terrain is locally lower" with "robot fell over" -- exactly
    the failure mode base_height_below_threshold_sustained's own docstring names ("a brief height
    dip... doesn't kill the episode; only staying low does"), and exactly why training itself
    never uses an instantaneous check for this. Root cause confirmed, not assumed: this was never
    a MuJoCo-vs-IsaacGym terrain-mesh fidelity gap (a hypothesis considered and left open in an
    earlier pass) -- terrain generation was independently verified byte-faithful to
    simulator/shared/terrain.py before this fix, and the mesh-fidelity question was always a red
    herring once the success criterion itself didn't match training's own fall definition.

    SETTLE-PHASE SAFETY: every tier this project trains on is a DEPRESSION or flat (never a rise
    above nominal ground -- see _generate_terrain_tile_m's own per-tier citations), and the
    standing keyframe's own feet height is tuned for flat (nominal-0) ground, so the robot can
    only ever spawn AT or ABOVE the local terrain surface under it, never embedded in it. A brief,
    physically-valid settle onto whatever local height is actually under its feet (same
    zero-command settle every other test here uses) is expected, not a symptom of a bad
    conversion.

    Returns a pooled overall_success_rate (the number Table 5 wants) plus a per-tier breakdown
    (num_episodes/num_success per tier actually drawn) for diagnostics -- the breakdown is NOT
    surfaced through sim2sim_eval.py's own dispatcher, only the pooled rate is (see that script's
    own _run_locomotion_metrics)."""
    import numpy as np

    env = pl.env
    settle_steps = int(round(args.settle_s * args.control_hz))
    traverse_steps = int(round(args.command_s * args.control_hz))
    rng = np.random.default_rng(args.seed + 1)  # +1: distinct stream from the velocity-axis tests'
    # own np.random.seed(args.seed) global seed above, so this test's draws don't silently
    # consume/perturb those tests' own (global-RNG-based) randomness or vice versa.

    tier_names = list(TERRAIN_TYPE_PROPORTIONS.keys())
    tier_weights = np.array([TERRAIN_TYPE_PROPORTIONS[t] for t in tier_names])
    tier_weights = tier_weights / tier_weights.sum()

    per_tier_episodes: dict[str, int] = dict.fromkeys(tier_names, 0)
    per_tier_success: dict[str, int] = dict.fromkeys(tier_names, 0)
    success_count = 0
    try:
        for _ in range(args.num_episodes):
            terrain_type = str(rng.choice(tier_names, p=tier_weights))
            difficulty = float(rng.choice(TERRAIN_DIFFICULTY_CHOICES))
            tile_m = _generate_terrain_tile_m(rng, terrain_type, difficulty)
            env.model.hfield_data[:] = _terrain_tile_to_hfield_data(tile_m).flatten()

            _reset(pl)
            _step_n(pl, settle_steps, 0.0, 0.0, 0.0)

            fall_tracker = _SustainedFallTracker(args.fall_height_m)
            inner = pl.policy.policy
            inner.lin_vel_command[:] = (args.forward_speed, 0.0)
            inner.ang_vel_command = 0.0
            for _ in range(traverse_steps):
                pl.step()
                fall_tracker.update(float(env.data.qpos[2]))

            per_tier_episodes[terrain_type] += 1
            if not fall_tracker.fell:
                success_count += 1
                per_tier_success[terrain_type] += 1
    finally:
        # Restore the flat baseline other tests in this worker rely on -- see _build_pipeline's
        # own comment on why hfield_data persists across resets and must be explicitly restored.
        env.model.hfield_data[:] = 1.0

    return {
        "overall_success_rate": success_count / args.num_episodes,
        "num_episodes": args.num_episodes,
        "num_success": success_count,
        "per_tier": {
            t: {"num_episodes": per_tier_episodes[t], "num_success": per_tier_success[t]} for t in tier_names
        },
        "tier_proportions": dict(TERRAIN_TYPE_PROPORTIONS),
    }


def _run_push_recovery(pl, args: argparse.Namespace) -> dict:
    import mujoco

    env = pl.env
    pelvis_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, PUSH_BODY_NAME)
    if pelvis_id < 0:
        raise ValueError(f"Body {PUSH_BODY_NAME!r} not found in model -- cannot run push-recovery test.")

    settle_steps = int(round(args.settle_s * args.control_hz))
    push_steps = int(round(args.push_duration_s * args.control_hz))
    recovery_steps = int(round(args.push_recovery_window_s * args.control_hz))

    recovered_count = 0
    for _ in range(args.num_episodes):
        _reset(pl)
        _step_n(pl, settle_steps, 0.0, 0.0, 0.0)

        fall_tracker = _SustainedFallTracker(args.fall_height_m)
        env.data.xfrc_applied[pelvis_id, 0] = args.push_force_n
        try:
            for _ in range(push_steps):
                pl.step()
                fall_tracker.update(float(env.data.qpos[2]))
        finally:
            env.data.xfrc_applied[pelvis_id, :] = 0.0

        for _ in range(recovery_steps):
            pl.step()
            fall_tracker.update(float(env.data.qpos[2]))

        if not fall_tracker.fell:
            recovered_count += 1

    return {
        "recovery_rate": recovered_count / args.num_episodes,
        "num_episodes": args.num_episodes,
        "num_recovered": recovered_count,
        "push_force_n": args.push_force_n,
        "push_duration_s": args.push_duration_s,
    }


def run(args: argparse.Namespace) -> dict:
    import numpy as np

    np.random.seed(args.seed)
    pl = _build_pipeline(args.onnx_path)
    try:
        terrain_result = _run_terrain_traversal(pl, args)
        result = {
            "onnx_path": args.onnx_path,
            "control_hz": args.control_hz,
            "fall_height_m": args.fall_height_m,
            "v_x_forward": _run_velocity_axis(pl, args, args.forward_speed, 0.0, 0.0),
            "v_x_backward": _run_velocity_axis(pl, args, args.backward_speed, 0.0, 0.0),
            "v_y_lateral": _run_velocity_axis(pl, args, 0.0, args.lateral_speed, 0.0),
            "omega_z_yaw": _run_velocity_axis(pl, args, 0.0, 0.0, args.yaw_rate),
            "push_recovery": _run_push_recovery(pl, args),
            # Flat scalar, matching the shape sim2sim_eval.py's own _run_locomotion_metrics
            # dispatcher already expects (result["terrain_traversal_success"], previously always
            # None) -- zero changes needed there now that this is a real value. The rich per-tier
            # breakdown lives separately in terrain_traversal_detail, not surfaced through that
            # dispatcher, for whoever reads this worker's own JSON output directly.
            "terrain_traversal_success": terrain_result["overall_success_rate"],
            "terrain_traversal_detail": terrain_result,
        }
    finally:
        pl.env.shutdown()
    return result


def main() -> int:
    args = _parse_args()
    try:
        result = run(args)
    except Exception:
        traceback.print_exc()
        return 1
    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[locomotion-metrics-worker] wrote {args.output_json}")
    for key in ("v_x_forward", "v_x_backward", "v_y_lateral", "omega_z_yaw"):
        r = result[key]
        print(f"  {key}: mae={r['mae']} fall_rate={r['fall_rate']}")
    print(f"  push_recovery: rate={result['push_recovery']['recovery_rate']}")
    print(
        f"  terrain_traversal: rate={result['terrain_traversal_success']} "
        f"per_tier={result['terrain_traversal_detail']['per_tier']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
