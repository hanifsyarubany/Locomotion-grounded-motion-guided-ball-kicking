#!/usr/bin/env python3
"""Dedicated, standalone sim2sim evaluation driver: feed it one or more ONNX checkpoints and a
YAML config naming which MuJoCo/RoboJuDo scans to run against them, and it publishes every result
to wandb -- no training run required. Runs under the TRAINING-side `hssim` conda env (needs
wandb + onnxruntime, both confirmed present there), and reaches into RoboJuDo the same way
FastSACAgent's own periodic sim2sim checks already do: by calling the existing
`record_mujoco_survival_scan` / `record_mujoco_kick_to_loco_flip_scan` /
`record_mujoco_loco_to_kick_handoff_scan` wrapper functions directly. Those already own the
subprocess-to-the-separate-`robojudo`-conda-env boundary (see each module's own docstring) --
nothing here re-implements it.

WHY A SEPARATE TOOL, NOT A TRAINING-LOOP FEATURE: FastSACAgent's own `_maybe_start_mujoco_*`
methods exist to sample a LIVE training run's own current checkpoint periodically, on that run's
OWN wandb run. This script is for a DIFFERENT job -- evaluating one or more ALREADY-SAVED
checkpoints (which may be from different runs, different lineages, or a run that finished days
ago), on demand, without needing a live training process at all.

CONFIG. See configs/sim2sim_eval/example.yaml for a fully-commented template. Two top-level
blocks:
  - `onnx_targets`: the checkpoint(s) to evaluate. `kick_aim_enabled` must be stated explicitly
    per target -- unlike almost everything else a checkpoint needs (skill count, ball geometry,
    the kick->locomotion flip boundary), it is NOT embedded in the ONNX export at all (confirmed
    by reading utils/inference_helpers.py's own metadata-attachment call site: only
    skill_motion_start_idx/skill_motion_end_idx/kick_recovery_locomotion_flip_enabled/
    skill_pre_recovery_motion_end_idx/skill_ball_xy/skill_target_xy are attached --
    kick_aim_enabled_per_motion is read LIVE from a training env by FastSACAgent's own
    _kick_aim_info_per_motion, which this script has no equivalent of, having no live env at
    all). Every project checkpoint as of 2026-08-22 trains with kick_aim_enabled=True.
  - `evaluations`: which scan(s) to run, and with what parameters -- one block per scan type
    (`kick_survival`, `kick_to_loco_flip`, `loco_to_kick_handoff`, `locomotion_metrics`), applied to
    EVERY onnx target and EVERY one of that target's skills -- EXCEPT `locomotion_metrics`, which
    is checkpoint-level, not skill-level (Table 5's locomotion generality: v_x/v_y/yaw
    command-tracking MAE, push-recovery rate -- none of that varies by which skill_id you ask for),
    so it runs ONCE per target regardless of skill count (see `_PER_TARGET_EVAL_TYPES`).

SKILL AUTO-DISCOVERY. `skill_ids: null` (the default) reads `skill_motion_start_idx` straight out
of the ONNX's own embedded metadata via onnxruntime -- the SAME metadata UnifiedLocoKickPolicy
itself parses at deployment (robojudo/policy/unified_loco_kick_policy.py's own
`meta["skill_motion_start_idx"]`/json.loads -- mirrored here, not re-derived independently, so a
change to the export format breaks both readers identically rather than drifting apart). No
robojudo pipeline is constructed just to count skills.

WANDB. One run per script invocation (not one per onnx target) -- comparing checkpoints side by
side in a single place is the whole point of feeding several at once. Metrics land under
`{target_name}/{skill_id}/{eval_group}/<metric>` -- e.g. `skill011/0/loco_to_kick/fall_rate`.
`eval_group` is a short name per evaluation type (`kick_survival` -> `kick`, `kick_to_loco_flip`
-> `kick_to_loco`, `loco_to_kick_handoff` -> `loco_to_kick` -- see `_EVAL_TYPE_GROUP`), and
`<metric>` is deliberately bare (`fall_rate`, not `loco_to_kick_handoff_fall_rate`) since the
group segment already carries that -- repeating it in both would just be noise in every wandb
panel title. 2026-09-01: simplified from an earlier `Kick_skills_{i}/sim2sim/<full_metric_name>`
scheme (which mirrored FastSACAgent's own training-time sim2sim wandb keys) at user request --
this tool's keys are intentionally its own convention now, not a mirror of training's. Defaults
to the SAME wandb project training itself uses (`UnifiedBallKickingEnhanced`), tagged
`group="sim2sim-eval"` and `job_type="sim2sim-eval"` so these runs are visually distinct from
actual training runs in that project rather than needing an entirely separate project to find them.
The run's wandb `name` is always stamped with a `<run_timestamp>` (`%Y%m%d_%H%M%S`, generated once
per invocation -- same format this project's own training run directories already use, e.g.
`20260827_044728-...`) -- `<run_timestamp>-<wandb.run_name>` when `run_name` is configured,
otherwise the bare timestamp -- so two invocations of the SAME config (e.g. re-running after
changing a checkpoint) never collide on one wandb run name. This same `<run_name>` string also
names the local output subfolder below, so a wandb run and its on-disk record are trivially
paired up by matching folder name to wandb run name directly.

LOCAL OUTPUT FILES (`out/sim2sim_eval/` by default, `--output-dir` to override). On top of wandb,
every invocation through `main()` also writes two local files under
`out/sim2sim_eval/<run_name>/` (the SAME `<run_name>` -- `<run_timestamp>-<wandb.run_name>` or the
bare timestamp -- stamped onto wandb's own run name above; sanitized via `_sanitize_run_subdir` if
it contains a path separator):
  - `.../sim2sim_eval.csv` -- every single (iteration, target, skill, eval, metric) data point
    actually logged, one row each -- the literal long-format dump of every `wandb.log` call this
    run made, loadable straight into `pandas.read_csv` for anyone who wants to slice/replot
    without the wandb UI (or without wandb access at all).
  - `.../sim2sim_eval.json` -- one summary block per (target, skill_id, eval_group, metric):
    n/mean/std/min/max/last collapsed across iterations, plus run metadata (config path, target
    list, wandb run id/url) -- the "read this one file to know how the run went" artifact, not a
    replacement for the CSV's row-level detail.
Both are written from `run_eval`'s own `finally` block (see `_write_output_files`), so a mid-run
crash or Ctrl-C still leaves whatever was actually collected on disk rather than losing a long
run's partial results outright. `run_eval()` itself defaults `output_dir=None` (no local files --
only `main()`'s CLI path opts into the "out/sim2sim_eval" default), so calling it directly (as
this file's own test suite does) never has an on-disk side effect unless a caller explicitly asks
for one.

ITERATIONS (`iterations:`, top-level config key, default 1). Repeats the ENTIRE `evaluations`
list against the SAME fixed checkpoint(s) this many times, each time drawing a fresh, independently
seeded batch of trials, and logs every repeat under the SAME metric keys at `step=iteration` --
so with `iterations: 100` you watch each metric progress across 100 independent trial batches as
an ordinary wandb line chart (wandb's default x-axis is the logged step), rather than getting a
single point per checkpoint. With `iterations: 1` (the default), `step` is instead the
checkpoint's own training step when inferable from its filename (`model_0180000.onnx` -> 180000)
so a one-shot comparison across DIFFERENT checkpoints from the same lineage lands on a shared,
meaningful x-axis; otherwise an internal running counter.

Usage:
    python sim2sim_eval.py --config configs/sim2sim_eval/example.yaml

    # Ad-hoc: evaluate one checkpoint not (yet) listed in any config file. Uses kick_aim_enabled
    # from the CLI (defaults True, matching every current project checkpoint) rather than a config
    # target block -- for a quick one-off check, not a substitute for a real config for anything
    # you intend to re-run or compare against later.
    python sim2sim_eval.py --config configs/sim2sim_eval/example.yaml \\
        --onnx-path /path/to/model_0500000.onnx

    # Verify the whole pipeline (config parsing, skill discovery, the actual MuJoCo scans, wandb
    # logging) without touching the real team wandb project:
    python sim2sim_eval.py --config configs/sim2sim_eval/example.yaml --wandb-mode offline
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
import tempfile
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime

import yaml
from loguru import logger

# This file lives at <repo>/src/holosoma/holosoma/sim2sim_eval.py and imports its sibling scan
# wrappers via the DOTTED package name ("holosoma.record_..."), not bare names -- so what needs to
# be on sys.path is the package's PARENT directory (.../src/holosoma), not this file's own
# directory (.../src/holosoma/holosoma, which is the package's own insides and contains no
# importable top-level "holosoma" module at all). This matters because this conda env's `holosoma`
# editable install (`pip install -e`) actually points at a DIFFERENT, sibling project
# (playground/locomotion_and_ball_kicking's copy of holosoma/) -- confirmed via
# `python -c "import holosoma; print(holosoma.__file__)"`. Without prepending the real parent dir
# here, `import holosoma.record_mujoco_kick_to_loco_flip_scan` below silently resolves against
# that OTHER project's holosoma package (missing this module entirely) whenever this script is
# invoked directly (`python .../sim2sim_eval.py`, which puts only this file's own directory, not
# its parent, at sys.path[0]) -- reproduced as `ModuleNotFoundError:
# No module named 'holosoma.record_mujoco_kick_to_loco_flip_scan'` on 2026-09-01.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from holosoma.record_mujoco_kick_to_loco_flip_scan import record_kick_to_loco_flip_scan  # noqa: E402
from holosoma.record_mujoco_loco_to_kick_handoff_scan import record_loco_to_kick_handoff_scan  # noqa: E402
from holosoma.record_mujoco_locomotion_metrics_scan import record_locomotion_metrics_scan  # noqa: E402
from holosoma.record_mujoco_ref_clip_recovery_scan import record_ref_clip_recovery_scan  # noqa: E402
from holosoma.record_mujoco_survival_scan import record_survival_scan  # noqa: E402

DEFAULT_WANDB_PROJECT = "UnifiedBallKickingEnhanced"

# Same "step since kick trigger" naming convention every mujoco_*_scan.py script already uses --
# read once here rather than re-declaring per eval-type dispatcher below.
_ONNX_STEP_RE = re.compile(r"model_(\d+)\.onnx$")


# --------------------------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------------------------- #


@dataclass
class OnnxTarget:
    path: str
    name: str | None = None
    kick_aim_enabled: bool = True
    skill_ids: list[int] | None = None  # None = auto-discover every skill from ONNX metadata


@dataclass
class EvalSpec:
    type: str
    params: dict = field(default_factory=dict)


@dataclass
class Sim2SimEvalConfig:
    onnx_targets: list[OnnxTarget]
    evaluations: list[EvalSpec]
    iterations: int = 1
    max_concurrent_scans: int = 1
    wandb_project: str = DEFAULT_WANDB_PROJECT
    wandb_entity: str | None = None
    wandb_group: str = "sim2sim-eval"
    wandb_run_name: str | None = None
    wandb_tags: list[str] = field(default_factory=list)


_KNOWN_EVAL_TYPES = {
    "kick_survival", "kick_to_loco_flip", "loco_to_kick_handoff", "locomotion_metrics",
    "ref_clip_recovery",
}

# Eval types that are a property of the CHECKPOINT, not of any one skill (Table 5's locomotion
# generality: v_x/v_y/yaw tracking, push recovery -- these don't vary by skill_id the way kick
# metrics do). Dispatched ONCE per target instead of once per skill_id, and logged under a
# skill_id-free wandb key (`{target}/{group}/{metric}`, not `{target}/{skill_id}/{group}/{metric}`)
# -- see run_eval's own submission-loop branch below for where this set is actually consulted.
_PER_TARGET_EVAL_TYPES = {"locomotion_metrics"}


def load_eval_config(path: str) -> Sim2SimEvalConfig:
    """Hand-parsed + validated, same `raw.get(key, default)` / explicit-error-on-bad-value style
    as this project's own config_types/multi_skill.py -- not a pydantic schema, so this stays
    consistent with the rest of the codebase's config-loading conventions rather than introducing
    a second one for just this tool."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    raw_targets = raw.get("onnx_targets")
    if not raw_targets:
        raise ValueError(f"{path}: 'onnx_targets' is required and must be a non-empty list.")
    onnx_targets = []
    for i, t in enumerate(raw_targets):
        onnx_path = t.get("path")
        if not onnx_path:
            raise ValueError(f"{path}: onnx_targets[{i}] is missing required key 'path'.")
        skill_ids = t.get("skill_ids")
        if skill_ids is not None and (not isinstance(skill_ids, list) or not all(isinstance(s, int) for s in skill_ids)):
            raise ValueError(f"{path}: onnx_targets[{i}].skill_ids must be a list of ints or omitted, got {skill_ids!r}")
        onnx_targets.append(
            OnnxTarget(
                path=onnx_path,
                name=t.get("name"),
                kick_aim_enabled=bool(t.get("kick_aim_enabled", True)),
                skill_ids=skill_ids,
            )
        )

    raw_evals = raw.get("evaluations")
    if not raw_evals:
        raise ValueError(f"{path}: 'evaluations' is required and must be a non-empty list.")
    evaluations = []
    for i, e in enumerate(raw_evals):
        ev_type = e.get("type")
        if ev_type not in _KNOWN_EVAL_TYPES:
            raise ValueError(
                f"{path}: evaluations[{i}].type must be one of {sorted(_KNOWN_EVAL_TYPES)}, got {ev_type!r}"
            )
        params = {k: v for k, v in e.items() if k != "type"}
        # locomotion_metrics uses num_episodes, not num_trials (see mujoco_locomotion_metrics_
        # worker.py's own CLI) -- skip this check for it rather than validate an unrelated,
        # never-consumed field.
        if ev_type not in _PER_TARGET_EVAL_TYPES:
            num_trials = params.get("num_trials", 32)
            if not isinstance(num_trials, int) or num_trials < 1:
                raise ValueError(f"{path}: evaluations[{i}].num_trials must be a positive int, got {num_trials!r}")
        evaluations.append(EvalSpec(type=ev_type, params=params))

    iterations = raw.get("iterations", 1)
    if not isinstance(iterations, int) or iterations < 1:
        raise ValueError(f"{path}: 'iterations' must be a positive int, got {iterations!r}")

    max_concurrent_scans = raw.get("max_concurrent_scans", 1)
    if not isinstance(max_concurrent_scans, int) or max_concurrent_scans < 1:
        raise ValueError(f"{path}: 'max_concurrent_scans' must be a positive int, got {max_concurrent_scans!r}")

    wandb_cfg = raw.get("wandb", {}) or {}
    return Sim2SimEvalConfig(
        onnx_targets=onnx_targets,
        evaluations=evaluations,
        iterations=iterations,
        max_concurrent_scans=max_concurrent_scans,
        wandb_project=wandb_cfg.get("project", DEFAULT_WANDB_PROJECT),
        wandb_entity=wandb_cfg.get("entity"),
        wandb_group=wandb_cfg.get("group", "sim2sim-eval"),
        wandb_run_name=wandb_cfg.get("run_name"),
        wandb_tags=list(wandb_cfg.get("tags", [])),
    )


# --------------------------------------------------------------------------------------------- #
# ONNX metadata introspection (no robojudo pipeline construction needed for this)
# --------------------------------------------------------------------------------------------- #


def discover_num_skills(onnx_path: str) -> int:
    """Reads `skill_motion_start_idx` straight out of the ONNX's own embedded custom metadata --
    the SAME field UnifiedLocoKickPolicy itself parses at deployment
    (robojudo/policy/unified_loco_kick_policy.py, `meta["skill_motion_start_idx"]`). Falls back to
    1 (legacy single-skill export, no such metadata at all) rather than raising -- an older
    checkpoint exported before multi-skill support existed is still a valid, single-skill target.

    Uses onnxruntime directly (confirmed available under the training-side `hssim` env this
    script runs in) rather than shelling out to the separate `robojudo` conda env just to read a
    metadata string -- no MuJoCo/pipeline construction is needed for this."""
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    meta = sess.get_modelmeta().custom_metadata_map
    if "skill_motion_start_idx" not in meta:
        return 1
    return len(json.loads(meta["skill_motion_start_idx"]))


def infer_step_from_onnx_path(onnx_path: str) -> int | None:
    """`model_0180000.onnx` -> 180000, matching every checkpoint this project's own training loop
    saves (FastSACAgent's own `f"model_{self.global_step:07d}.onnx"`). None for any other naming
    (e.g. a manually renamed export) -- the caller falls back to a running counter in that case."""
    m = _ONNX_STEP_RE.search(os.path.basename(onnx_path))
    return int(m.group(1)) if m else None


def default_target_name(onnx_path: str) -> str:
    """`.../logs/UnifiedBallKickingEnhanced/<run_name>/model_0180000.onnx` ->
    `<run_name>/model_0180000` -- the run directory name plus the checkpoint stem, enough to
    distinguish two checkpoints from different lineages sharing the same step number, without
    requiring the caller to name every target explicitly."""
    run_dir = os.path.basename(os.path.dirname(os.path.abspath(onnx_path)))
    stem = os.path.splitext(os.path.basename(onnx_path))[0]
    return f"{run_dir}/{stem}" if run_dir else stem


def _compute_strike_window_ticks(motion_config: dict, sim_config: dict, skill_id: int) -> tuple[int, int] | None:
    """Pure arithmetic half of get_strike_window_ticks below -- split out so the frame-index math
    is unit-testable against a hand-built fake config, with no onnxruntime/real-checkpoint
    dependency (mirrors this file's own discover_num_skills / get_strike_window_ticks split:
    real-ONNX-only pieces get a "verified by hand" note instead of an automated test).

    `motion_config`/`sim_config` are `experiment_config["command"]["setup_terms"]["motion_command"]
    ["params"]["motion_config"]` / `experiment_config["simulator"]["config"]["sim"]` respectively
    (both dicts, already `json.loads`-ed out of the ONNX's own embedded `experiment_config`
    metadata string -- see get_strike_window_ticks below for where those actually come from).

    Returns (strike_start_tick, stand_start_tick), both in "ticks since kick trigger" -- the SAME
    unit kick_to_loco_flip's own --flip-delay-min/max-steps use, i.e. directly comparable to a
    flip_tick draw -- or None if this checkpoint has no scrubbed strike/stand boundaries at all
    (legacy/single-clip mode; motion_strike_start_frame/motion_stand_start_frame absent or empty).

    THE OFFSET THIS EXISTS FOR: strike_start_frame/stand_start_frame (SkillConfig, configs/skill/
    *.yaml) are frame indices relative to that skill's OWN raw clip content -- "0 <=
    strike_start_frame < stand_start_frame <= this clip's own raw frame count" (SkillConfig's own
    docstring) -- NOT directly the same coordinate as flip_tick, which counts ticks since
    [TRIGGER_KICK] (UnifiedLocoKickPolicy._trigger_kick sets curr_motion_timestep =
    skill_motion_start_idx[skill_id] at trigger, +1 per control tick thereafter -- confirmed by
    reading that method and post_step_callback directly). The two coordinate spaces differ by
    exactly the length of a SYNTHETIC PREPEND TRANSITION (an interpolated windup from the default
    pose, spliced in BEFORE the raw clip's own frame 0 when enable_default_pose_prepend is set) --
    skill_motion_start_idx itself does NOT shift to make room for it (confirmed via wbt.py's own
    _maybe_smooth_motion_head_velocities docstring: "the synthetic prepend occupies exactly that
    slot ... The prepend length is therefore added back"). Formula mirrors that same function's
    own prepend_len computation exactly (wbt.py, motion_head_velocity_smoothing call site) --
    reimplemented here (not imported) because holosoma's own training-time classes aren't
    importable from this ROS/robojudo-free, onnxruntime-only tool without constructing a full env.

    Verified end-to-end against two real checkpoints (2026-09-02): skill011
    (strike_start_frame=170, stand_start_frame=215, default_pose_prepend_duration_s=1.0,
    dt=0.02 -> prepend_len=50) -> (220, 265); skill015 (70, 116, same prepend) -> (120, 166).
    """
    strike_frames = motion_config.get("motion_strike_start_frame") or []
    stand_frames = motion_config.get("motion_stand_start_frame") or []
    if skill_id >= len(strike_frames) or skill_id >= len(stand_frames):
        return None
    strike_frame = int(strike_frames[skill_id])
    stand_frame = int(stand_frames[skill_id])

    dt = float(sim_config["control_decimation"]) / float(sim_config["fps"])
    prepend_len = 0
    if motion_config.get("enable_default_pose_prepend", False):
        per_motion = motion_config.get("motion_prepend_duration_s") or []
        duration = (
            float(per_motion[skill_id]) if per_motion else float(motion_config.get("default_pose_prepend_duration_s", 2.0))
        )
        if duration > 0.0:
            steps = round(duration / dt)
            # Mirrors _maybe_add_default_pose_transition's/_maybe_smooth_motion_head_velocities's
            # own identical skip condition exactly -- a duration too short for dt inserts NOTHING,
            # so no offset must be applied either.
            if steps > 1:
                prepend_len = steps

    return prepend_len + strike_frame, prepend_len + stand_frame


def get_strike_window_ticks(onnx_path: str, skill_id: int) -> tuple[int, int] | None:
    """(strike_start_tick, stand_start_tick) for this skill, or None if this checkpoint's ONNX has
    no `experiment_config` metadata at all, or no scrubbed strike/stand boundaries for this
    skill_id -- see _compute_strike_window_ticks above for the full derivation and unit coordinate
    system. Like discover_num_skills, uses onnxruntime directly rather than constructing a
    robojudo pipeline just to read a metadata string.

    NOT unit-tested directly (real onnxruntime + a real checkpoint's full experiment_config are
    both required) -- exercised by hand against real checkpoints instead, same convention this
    file's own discover_num_skills docstring already established. _compute_strike_window_ticks
    above carries the actual arithmetic under an automated test.
    """
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    meta = sess.get_modelmeta().custom_metadata_map
    if "experiment_config" not in meta:
        return None
    cfg = json.loads(meta["experiment_config"])
    try:
        motion_config = cfg["command"]["setup_terms"]["motion_command"]["params"]["motion_config"]
        sim_config = cfg["simulator"]["config"]["sim"]
    except KeyError:
        return None
    return _compute_strike_window_ticks(motion_config, sim_config, skill_id)


def get_skill_clip_length_ticks(onnx_path: str, skill_id: int) -> int | None:
    """This skill's total buffer segment length, in ticks -- `skill_motion_end_idx[skill_id] -
    skill_motion_start_idx[skill_id]`, i.e. the LAST valid flip_tick for this skill is this value
    minus 1 (curr_motion_timestep clamps there -- UnifiedLocoKickPolicy.post_step_callback's own
    `skill_last_frame = self._skill_end_idx[self.kick_skill_id] - 1` clamp). Unlike
    get_strike_window_ticks, this needs only the CLEAN top-level `skill_motion_start_idx`/
    `skill_motion_end_idx` metadata (the same fields discover_num_skills already reads) -- no
    `experiment_config` parsing, no prepend-offset reconstruction, since both ends of this span
    are already in the SAME "ticks since kick trigger" coordinate space flip_tick uses (confirmed:
    skill_motion_start_idx[skill_id] IS curr_motion_timestep's value at trigger).

    Returns None if this checkpoint has no such metadata at all (legacy single-clip export with
    no per-skill boundaries -- see discover_num_skills's own docstring for that fallback case) or
    skill_id is out of range.

    NOT unit-tested directly (real onnxruntime + a real checkpoint required) -- same "exercised by
    hand" convention as get_strike_window_ticks/discover_num_skills; this one has no separate pure
    arithmetic to split out (the subtraction is a one-liner, not worth its own tested helper).
    """
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    meta = sess.get_modelmeta().custom_metadata_map
    if "skill_motion_start_idx" not in meta or "skill_motion_end_idx" not in meta:
        return None
    starts = json.loads(meta["skill_motion_start_idx"])
    ends = json.loads(meta["skill_motion_end_idx"])
    if skill_id >= len(starts) or skill_id >= len(ends):
        return None
    return int(ends[skill_id]) - int(starts[skill_id])


def _pool_flip_rates(sub_results: list[tuple[int, float | None, float | None]]) -> tuple[float | None, float | None]:
    """Combines multiple independent record_kick_to_loco_flip_scan results -- each
    (num_trials_for_that_subcall, alive_rate, pre_flip_fail_rate) -- into ONE statistically
    correct pooled (alive_rate, pre_flip_fail_rate).

    WHY RECONSTRUCTED COUNTS, NOT A NAIVE AVERAGE OF THE TWO RATES: record_kick_to_loco_flip_scan
    only ever returns already-divided rates (parses the wrapper's own `.4f`-formatted SUMMARY
    line, never the raw "num/denom" text also on that line -- see
    record_mujoco_kick_to_loco_flip_scan.py's own parsing code) -- there is no raw-count API to
    call instead, and that function is SHARED with FastSACAgent's own training-time periodic sim2sim
    checks, so its return signature isn't something to change just for this. A naive
    `(rate_a + rate_b) / 2` would silently misweight two sub-calls with different trial counts
    (e.g. a 2-trial and an 8-trial sub-call must NOT count equally). Reconstructing
    `num_pre_flip_fail = round(pre_flip_fail_rate * n)` (denominator is always `n`, always
    defined) and `num_alive = round(alive_rate * num_reached)` (denominator is `n -
    num_pre_flip_fail`, None when that's 0) recovers the exact integer counts whenever
    rate*n rounds unambiguously -- true for every num_trials this project actually uses (the
    4-decimal-place print precision distinguishes far more distinct counts than any of this
    project's trial counts need) -- and is an approximation only in the abstract general case.

    Skips any (n, ...) entry with n<=0 outright (that sub-window was never run -- see
    _run_kick_to_loco_flip_split_by_strike_phase's own num_trials-splitting logic). Returns
    (None, None) if every entry was skipped or empty."""
    total_trials = 0
    total_pre_flip_fail = 0
    total_reached = 0
    total_alive = 0
    for n, alive_rate, pre_flip_fail_rate in sub_results:
        if n <= 0 or pre_flip_fail_rate is None:
            continue
        num_pre_flip_fail = round(pre_flip_fail_rate * n)
        num_reached = n - num_pre_flip_fail
        total_trials += n
        total_pre_flip_fail += num_pre_flip_fail
        total_reached += num_reached
        if alive_rate is not None:
            total_alive += round(alive_rate * num_reached)

    if total_trials == 0:
        return None, None
    pooled_pre_flip_fail_rate = total_pre_flip_fail / total_trials
    pooled_alive_rate = (total_alive / total_reached) if total_reached > 0 else None
    return pooled_alive_rate, pooled_pre_flip_fail_rate


def _pool_recovery_metrics(
    sub_results: list[tuple[int, float | None, float | None, float | None]],
) -> tuple[float | None, float | None]:
    """Sibling of _pool_flip_rates for the recovery-steps metrics -- kept SEPARATE rather than
    folded into that function, since it pools a different STATISTIC SHAPE (a mean over a subset,
    not a plain rate) and _pool_flip_rates's own docstring already explains why its signature
    stays fixed (shared training-time caller downstream of record_kick_to_loco_flip_scan).

    Each sub_results entry is (num_trials_for_that_subcall, pre_flip_fail_rate, recovered_rate,
    mean_recovery_steps) -- pre_flip_fail_rate is required (not alive_rate) because
    RECOVERED_RATE'S OWN DENOMINATOR is num_reached_flip (same population alive_rate uses, see
    mujoco_kick_loco_flip_scan.py's own SUMMARY_RECOVERY_STEPS docstring), which is reconstructed
    from pre_flip_fail_rate the SAME way _pool_flip_rates reconstructs it for alive_rate -- see
    that function's own docstring for why reconstructed counts, not a naive average, are used.

    mean_recovery_steps is pooled by weighting each sub-call's own mean by its OWN num_recovered
    (reconstructed as round(recovered_rate * num_reached)) -- a plain average-of-means would
    misweight a 1-recovery sub-call equally against a 20-recovery one.

    Skips any (n, ...) entry with n<=0 or pre_flip_fail_rate is None. Returns (None, None) if every
    entry was skipped, empty, or nothing ever reached the flip."""
    total_reached = 0
    total_recovered = 0
    total_recovery_step_sum = 0.0
    for n, pre_flip_fail_rate, recovered_rate, mean_recovery_steps in sub_results:
        if n <= 0 or pre_flip_fail_rate is None:
            continue
        num_reached = n - round(pre_flip_fail_rate * n)
        total_reached += num_reached
        if recovered_rate is None or num_reached <= 0:
            continue
        num_recovered = round(recovered_rate * num_reached)
        total_recovered += num_recovered
        if mean_recovery_steps is not None and num_recovered > 0:
            total_recovery_step_sum += mean_recovery_steps * num_recovered

    if total_reached == 0:
        return None, None
    pooled_recovered_rate = total_recovered / total_reached
    pooled_mean_recovery_steps = (total_recovery_step_sum / total_recovered) if total_recovered > 0 else None
    return pooled_recovered_rate, pooled_mean_recovery_steps


# --------------------------------------------------------------------------------------------- #
# Per-eval-type dispatch -- each returns {bare_metric_name: value_or_None}. Bare, not prefixed
# with the eval type or "sim2sim/" -- run_eval() prepends `{skill_id}/{_EVAL_TYPE_GROUP[ev.type]}/`
# to build the full wandb key, so repeating the eval type inside the metric name here too would
# just duplicate what the group segment already says (see run_eval's own WANDB docstring note).
# --------------------------------------------------------------------------------------------- #

# Short group name per evaluation type, used as the third wandb key segment (see run_eval's own
# WANDB docstring note) -- deliberately its own short vocabulary, not each scan's full type string
# (e.g. "kick" not "kick_survival"), since the key is already scoped by target and skill_id.
_EVAL_TYPE_GROUP = {
    "kick_survival": "kick",
    "kick_to_loco_flip": "kick_to_loco",
    "loco_to_kick_handoff": "loco_to_kick",
    "locomotion_metrics": "locomotion",
    "ref_clip_recovery": "ref_clip_recovery",
}


def _fresh_lock_path(tag: str) -> str:
    """A brand-new, never-reused lock file path for one dispatch call. Every record_*_scan
    function accepts `lock_path` as a plain override of its own module-level DEFAULT_LOCK_PATH
    (see e.g. record_survival_scan's own docstring: "Serialized cluster-wide via a lock file at
    `lock_path`") -- that default exists to protect a LIVE TRAINING PROCESS's own periodic
    background scans from piling up on themselves (see record_mujoco_survival_scan.py's own
    module docstring), a concurrency model this tool doesn't share: run_eval() is ALREADY the
    sole coordinator deciding how much runs at once (via `max_concurrent_scans`), so reusing that
    shared, fixed-path lock here would only ever cost this tool something -- either colliding with
    an unrelated LIVE training process's own scans (busy-lock skip, silent data loss -- reproduced
    live, 2026-09-06: two of this tool's own scans were skipped this way against a real run in
    progress), or serializing this tool's own concurrent workers against EACH OTHER, defeating
    `max_concurrent_scans` entirely. A fresh path per call sidesteps both -- there is nothing left
    to protect against once every call has its own lock, so this is applied unconditionally, not
    just when `max_concurrent_scans > 1`."""
    return os.path.join(tempfile.gettempdir(), f"holosoma_sim2sim_eval_{tag}_{uuid.uuid4().hex}.lock")


def _run_kick_survival(onnx_path: str, step_label: str, skill_id: int, kick_aim_enabled: bool, params: dict) -> dict:
    # 2026-09-05: extra_metrics_out is ALWAYS requested here (not gated behind a config flag) --
    # mujoco_kick_survival_scan.py's own docstring is explicit that success_sigma_m/shot_error/
    # ball_speed cost nothing extra (same rollout, same trials, already-computed values just not
    # previously surfaced), so there's no tradeoff to opt into. Any key that isn't applicable for
    # this target (e.g. shot_error_* when kick_aim_enabled=False) comes back None and is simply
    # dropped by run_eval's own "never log a None" rule below -- same as direction_success_rate
    # already was.
    # 2026-09-09: track_posture_metrics (default False) resolves THIS skill's own stand_start tick
    # from the checkpoint's own ONNX metadata and hands it to the worker, which then reports the
    # six penalty_kick_recovery_* posture errors over the clip's post-swing recovery tail. Table
    # XI(a)'s rescoped outcome metric -- see mujoco_kick_survival_scan.py's own POST-STRIKE
    # RECOVERY POSTURE METRICS section. Resolved here rather than configured per-arm because it is
    # a property of the skill's clip, identical across the arms being compared; a hand-entered
    # value would silently mis-window one arm and invent a difference between them.
    stand_start_tick = None
    if params.get("track_posture_metrics", False):
        window = get_strike_window_ticks(onnx_path, skill_id)
        if window is None:
            logger.warning(
                f"[sim2sim] track_posture_metrics requested but {onnx_path} carries no strike/stand "
                f"boundaries for skill {skill_id} -- posture metrics skipped for this target."
            )
        else:
            stand_start_tick = window[1]

    extra_metrics: dict = {}
    fall_rate, hit_rate, direction_rate = record_survival_scan(
        onnx_path=onnx_path,
        step_label=step_label,
        stand_start_tick=stand_start_tick,
        num_trials=params.get("num_trials", 32),
        skill_id=skill_id,
        seed=params.get("seed", 0),
        ball_pos_randomization=tuple(params.get("ball_pos_randomization", (0.0, 0.0))),
        kick_aim_enabled=kick_aim_enabled,
        kick_aim_theta_max_deg=params.get("kick_aim_theta_max_deg", 15.0),
        kick_aim_theta_ref_deg=params.get("kick_aim_theta_ref_deg", 45.0),
        kick_aim_nominal_distance_m=params.get("kick_aim_nominal_distance_m", 5.0),
        direction_success_sigma_m=params.get("direction_success_sigma_m", 1.0),
        # success_sigma_m: [0.5, 1.0] (this eval type's own default when omitted, matching the
        # worker's own default exactly) -- one strict success_rate_<R> per radius, R in meters.
        success_sigma_m=params.get("success_sigma_m"),
        # 2026-09-12, default False so every previously-reported sweep reproduces unchanged: when
        # set, the ball is anchored to the robot's real pose at the trigger instead of to the world
        # point it was spawned at before the settle. See mujoco_kick_survival_scan.py's own
        # --reanchor-ball-at-trigger help for the drift measurement and the training-side contract
        # it restores.
        reanchor_ball_at_trigger=params.get("reanchor_ball_at_trigger", False),
        extra_metrics_out=extra_metrics,
        timeout_s=params.get("timeout_s", 300.0),
        lock_path=_fresh_lock_path("survival"),
    )
    out = {
        "fall_rate": fall_rate,
        "hit_rate": hit_rate,
        "direction_success_rate": direction_rate,
    }
    out.update(extra_metrics)  # ball_speed_mean/_std/_n/_max, shot_error_mean/_std/_n, success_rate_<R>
    return out


def _run_kick_to_loco_flip(onnx_path: str, step_label: str, skill_id: int, kick_aim_enabled: bool, params: dict) -> dict:
    if params.get("split_by_strike_phase", False):
        return _run_kick_to_loco_flip_split_by_strike_phase(onnx_path, step_label, skill_id, kick_aim_enabled, params)

    recovery_metrics: dict = {}
    alive_rate, pre_flip_fail_rate = record_kick_to_loco_flip_scan(
        onnx_path=onnx_path,
        step_label=step_label,
        num_trials=params.get("num_trials", 32),
        skill_id=skill_id,
        seed=params.get("seed", 0),
        kick_aim_enabled=kick_aim_enabled,
        flip_delay_min_steps=params.get("flip_delay_min_steps", 10),
        flip_delay_max_steps=params.get("flip_delay_max_steps", 60),
        timeout_s=params.get("timeout_s", 300.0),
        recovery_metrics_out=recovery_metrics,
        lock_path=_fresh_lock_path("kick_to_loco_flip"),
    )
    return {
        "alive_rate": alive_rate,
        "pre_flip_fail_rate": pre_flip_fail_rate,
        "recovered_rate": recovery_metrics.get("recovered_rate"),
        "mean_recovery_steps": recovery_metrics.get("mean_recovery_steps"),
    }


def _run_ref_clip_recovery(onnx_path: str, step_label: str, skill_id: int, kick_aim_enabled: bool, params: dict) -> dict:
    # motion_npz/frame_max are REQUIRED, not defaulted here -- see
    # mujoco_ref_clip_recovery_scan.py's own FRAME WINDOW note for why guessing frame_max would be
    # worse than a loud config error (pass this skill's own stand_start_frame from
    # configs/skill/skill_XXX.yaml).
    if "motion_npz" not in params:
        raise ValueError("ref_clip_recovery eval requires 'motion_npz' in its config params.")
    if "frame_max" not in params:
        raise ValueError("ref_clip_recovery eval requires 'frame_max' in its config params.")

    recovery_metrics: dict = {}
    alive_rate = record_ref_clip_recovery_scan(
        onnx_path=onnx_path,
        step_label=step_label,
        motion_npz=params["motion_npz"],
        frame_max=params["frame_max"],
        num_trials=params.get("num_trials", 32),
        skill_id=skill_id,
        seed=params.get("seed", 0),
        kick_aim_enabled=kick_aim_enabled,
        frame_min=params.get("frame_min", 0),
        hold_s=params.get("hold_s", 5.0),
        recentre_xy=params.get("recentre_xy", True),
        timeout_s=params.get("timeout_s", 300.0),
        recovery_metrics_out=recovery_metrics,
        lock_path=_fresh_lock_path("ref_clip_recovery"),
    )
    return {
        "alive_rate": alive_rate,
        "recovered_rate": recovery_metrics.get("recovered_rate"),
        "mean_recovery_steps": recovery_metrics.get("mean_recovery_steps"),
    }


# Lower bound of the auto-derived NONSTRIKE window in split_by_strike_phase mode (ticks since
# kick trigger) -- matches mujoco_kick_loco_flip_scan.py's own --flip-delay-min-steps CLI default,
# which itself mirrors MultiSkillConfig.kick_abort_delay_min_steps's default, for the same
# "flipping in the first few ticks tests nothing new" rationale that default already encodes. Not
# a config knob: split_by_strike_phase's whole point is removing manual window bookkeeping (the
# ONNX supplies the one number -- strike_start_tick -- that actually varies per checkpoint/skill).
_NONSTRIKE_MIN_STEPS = 10


def _run_kick_to_loco_flip_split_by_strike_phase(
    onnx_path: str, step_label: str, skill_id: int, kick_aim_enabled: bool, params: dict
) -> dict:
    """`split_by_strike_phase: true` variant of _run_kick_to_loco_flip -- runs the SAME underlying
    scan (no changes to mujoco_kick_loco_flip_scan.py itself; it already accepts an arbitrary
    flip-delay window via its own --flip-delay-min/max-steps) up to THREE times: once with the
    flip tick forced into the ONNX-derived STRIKE window, and once each for the two ONNX-derived
    NONSTRIKE sub-windows (before strike starts, and after it ends) -- pooled into one combined
    nonstrike rate (see _pool_flip_rates). ALL THREE windows are fully auto-derived, nothing to
    hand-tune or keep non-overlapping (2026-09-02: flip_delay_min/max_steps removed from this path
    entirely at user request). flip_delay_min_steps/flip_delay_max_steps in `params` are simply
    not read here -- they still apply to the OTHER (unsplit) path in _run_kick_to_loco_flip above.
    Returns keys prefixed "strike/"/"nonstrike/" -- run_eval's own
    `{name}/{skill_id}/{group}/{metric}` key-building then produces e.g.
    "skill011/0/kick_to_loco/strike/alive_rate" with no changes needed there. Each prefix also
    carries "recovered_rate"/"mean_recovery_steps" (2026-09-09, see
    mujoco_kick_loco_flip_scan.py's own SUMMARY_RECOVERY_STEPS docstring), pooled across nonstrike's
    own sub-calls via _pool_recovery_metrics -- a SEPARATE pooling function from _pool_flip_rates,
    since it pools a different statistic shape (a mean over a subset, not a plain rate).

    WINDOWS:
      - strike:       [strike_start_tick, stand_start_tick - 1]     (the whole scrubbed strike span)
      - nonstrike_pre:  [_NONSTRIKE_MIN_STEPS, strike_start_tick - 1)      (before strike starts)
      - nonstrike_post: (stand_start_tick, clip_last_tick]                (after strike ends, through
        the clip's own synthetic recovery+hold tail -- clip_last_tick from
        get_skill_clip_length_ticks, the LAST valid flip_tick for this skill)
    nonstrike_pre and nonstrike_post are run as SEPARATE scan invocations (a single
    --flip-delay-min/max-steps window can only sample ONE contiguous range) and their results
    POOLED into a single "nonstrike/*" rate via _pool_flip_rates -- see that function's own
    docstring for why pooling needs reconstructed counts, not a naive average of the two rates.

    Cost & trial split: nonstrike's num_trials is SPLIT roughly in half between pre and post
    (whichever sub-window(s) actually exist for this skill get the full share; a sub-window with 0
    trials allocated, or that doesn't exist for this skill -- e.g. stand_start_tick is already at
    the clip's last tick -- is skipped, not invoked with num_trials=0), so nonstrike's TOTAL trial
    count matches strike's -- this eval type still costs 2x its unsplit trial count overall, not
    3x, despite nonstrike now running up to 2 sub-scans.

    FAILS LOUD (raises) if the checkpoint has no scrubbed strike/stand boundaries for this skill at
    all, or no skill_motion_start_idx/skill_motion_end_idx metadata (needed for clip_last_tick) --
    both indicate this checkpoint predates the metadata this mode depends on, not a normal runtime
    condition to degrade quietly from. If NEITHER nonstrike sub-window has room to run (degenerate:
    strike spans nearly the checkpoint's whole clip), nonstrike/* comes back None (dropped by
    run_eval's own "never log a None" rule) with a warning logged -- NOT an error, since strike
    itself is still perfectly measurable.
    """
    strike_window = get_strike_window_ticks(onnx_path, skill_id)
    if strike_window is None:
        raise ValueError(
            f"split_by_strike_phase=True for {onnx_path!r} skill_id={skill_id}, but this checkpoint's "
            "embedded experiment_config has no motion_strike_start_frame/motion_stand_start_frame for "
            "this skill (legacy/single-clip export, or scrubbed boundaries were never configured for "
            "this skill) -- there is no strike window to split by."
        )
    strike_lo, stand_start_tick = strike_window
    strike_hi = stand_start_tick - 1  # inclusive, matching flip_delay_max_steps's own inclusive convention

    clip_length = get_skill_clip_length_ticks(onnx_path, skill_id)
    if clip_length is None:
        raise ValueError(
            f"split_by_strike_phase=True for {onnx_path!r} skill_id={skill_id}, but this checkpoint's "
            "ONNX has no skill_motion_start_idx/skill_motion_end_idx metadata for this skill -- needed "
            "to bound the post-stand nonstrike window at this skill's own clip end."
        )
    clip_last_tick = clip_length - 1

    nonstrike_pre = (_NONSTRIKE_MIN_STEPS, strike_lo - 1) if strike_lo > _NONSTRIKE_MIN_STEPS else None
    nonstrike_post = (stand_start_tick, clip_last_tick) if clip_last_tick >= stand_start_tick else None
    if nonstrike_pre is None and nonstrike_post is None:
        logger.warning(
            f"[sim2sim-eval] {onnx_path!r} skill_id={skill_id}: strike window [{strike_lo}, "
            f"{strike_hi}] leaves no room for either nonstrike sub-window (clip spans ticks "
            f"[0, {clip_last_tick}]) -- skipping nonstrike entirely for this (target, skill); "
            "strike/* is unaffected."
        )

    num_trials = params.get("num_trials", 32)
    n_sub_windows = sum(w is not None for w in (nonstrike_pre, nonstrike_post))
    seed = params.get("seed", 0)
    kwargs_common = dict(
        onnx_path=onnx_path, skill_id=skill_id, seed=seed, kick_aim_enabled=kick_aim_enabled,
        timeout_s=params.get("timeout_s", 300.0),
    )

    def _run(step_suffix: str, lo: int, hi: int, n: int) -> tuple[int, float | None, float | None, float | None, float | None]:
        recovery_metrics: dict = {}
        alive_rate, pre_flip_fail_rate = record_kick_to_loco_flip_scan(
            step_label=f"{step_label}/{step_suffix}", num_trials=n,
            flip_delay_min_steps=lo, flip_delay_max_steps=hi,
            recovery_metrics_out=recovery_metrics,
            lock_path=_fresh_lock_path(f"kick_to_loco_flip_{step_suffix}"), **kwargs_common,
        )
        return (
            n, alive_rate, pre_flip_fail_rate,
            recovery_metrics.get("recovered_rate"), recovery_metrics.get("mean_recovery_steps"),
        )

    strike_result = _run("strike", strike_lo, strike_hi, num_trials)
    _, strike_alive_rate, strike_pre_flip_fail_rate, strike_recovered_rate, strike_mean_recovery_steps = strike_result

    nonstrike_results = []
    if n_sub_windows > 0:
        n_pre = num_trials // n_sub_windows if nonstrike_pre is not None else 0
        n_post = num_trials - n_pre if nonstrike_post is not None else 0
        if nonstrike_pre is not None and n_pre > 0:
            nonstrike_results.append(_run("nonstrike_pre", nonstrike_pre[0], nonstrike_pre[1], n_pre))
        if nonstrike_post is not None and n_post > 0:
            nonstrike_results.append(_run("nonstrike_post", nonstrike_post[0], nonstrike_post[1], n_post))
    nonstrike_alive_rate, nonstrike_pre_flip_fail_rate = _pool_flip_rates(
        [(n, a, p) for n, a, p, r, m in nonstrike_results]
    )
    nonstrike_recovered_rate, nonstrike_mean_recovery_steps = _pool_recovery_metrics(
        [(n, p, r, m) for n, a, p, r, m in nonstrike_results]
    )

    return {
        "strike/alive_rate": strike_alive_rate,
        "strike/pre_flip_fail_rate": strike_pre_flip_fail_rate,
        "strike/recovered_rate": strike_recovered_rate,
        "strike/mean_recovery_steps": strike_mean_recovery_steps,
        "nonstrike/alive_rate": nonstrike_alive_rate,
        "nonstrike/pre_flip_fail_rate": nonstrike_pre_flip_fail_rate,
        "nonstrike/recovered_rate": nonstrike_recovered_rate,
        "nonstrike/mean_recovery_steps": nonstrike_mean_recovery_steps,
    }


def _run_loco_to_kick_handoff(onnx_path: str, step_label: str, skill_id: int, kick_aim_enabled: bool, params: dict) -> dict:
    # kick_aim_enabled is NOT forwarded -- this scan REQUIRES it unconditionally (see
    # mujoco_loco_to_kick_handoff_scan.py's own module docstring for why the non-aim-mode
    # ball/target placement is incorrect once the robot has moved from the origin). A target
    # configured with kick_aim_enabled=False is simply not a valid target for this one scan type;
    # the underlying worker script raises a clear error rather than silently misbehaving.
    track_transition_metrics = params.get("track_transition_metrics", False)
    transition_metrics: dict = {}
    # 2026-09-12: success/shot-error for the in-motion Table VIII row. Defaults are off, so a
    # config that does not ask for them gets this eval type's original three metrics unchanged.
    extra_metrics: dict = {}
    fall_rate, hit_rate, pre_handoff_fail_rate = record_loco_to_kick_handoff_scan(
        onnx_path=onnx_path,
        step_label=step_label,
        num_trials=params.get("num_trials", 32),
        skill_id=skill_id,
        seed=params.get("seed", 0),
        loco_duration_min_s=params.get("loco_duration_min_s", 2.0),
        loco_duration_max_s=params.get("loco_duration_max_s", 3.0),
        post_flip_hold_s=params.get("post_flip_hold_s", 8.0),
        # Entry-speed sweep knobs. Omitted from a config = None = the worker's own +/-1.0
        # training-range default, i.e. the standard in-motion row.
        lin_vel_x_range=(tuple(params["lin_vel_x_range"]) if "lin_vel_x_range" in params else None),
        lin_vel_y_range=(tuple(params["lin_vel_y_range"]) if "lin_vel_y_range" in params else None),
        ang_vel_yaw_range=(tuple(params["ang_vel_yaw_range"]) if "ang_vel_yaw_range" in params else None),
        # Same yaml key/convention as kick_survival's own ball_pos_randomization (_run_kick_
        # survival above) -- pass the SAME value in both eval blocks of a sweep to make the
        # settled-state and in-motion Table VIII rows placement-comparable (see
        # record_loco_to_kick_handoff_scan's own docstring for why this scan applies it in the
        # robot's local frame rather than world frame).
        ball_pos_randomization=tuple(params.get("ball_pos_randomization", (0.0, 0.0))),
        # Pass the SAME kick_aim_theta_max_deg the companion kick_survival block uses. At the
        # worker's 0.0 default this scan aims dead-centre every trial, which is an easier aim
        # task than the settled row's +/-15 sweep -- scoring success across that mismatch would
        # flatter the in-motion row for a reason unrelated to the handoff.
        kick_aim_theta_max_deg=params.get("kick_aim_theta_max_deg", 0.0),
        kick_aim_theta_ref_deg=params.get("kick_aim_theta_ref_deg", 45.0),
        kick_aim_nominal_distance_m=params.get("kick_aim_nominal_distance_m", 5.0),
        success_sigma_m=params.get("success_sigma_m"),
        extra_metrics_out=extra_metrics,
        timeout_s=params.get("timeout_s", 300.0),
        return_transition_metrics=track_transition_metrics,
        transition_window_steps=params.get("transition_window_steps", 50),
        transition_metrics_out=transition_metrics,
        lock_path=_fresh_lock_path("loco_to_kick_handoff"),
    )
    out = {
        "fall_rate": fall_rate,
        "hit_rate": hit_rate,
        "pre_handoff_fail_rate": pre_handoff_fail_rate,
    }
    out.update(extra_metrics)  # success_rate_<R>, shot_error_mean/_std/_n (absent if not requested)
    if track_transition_metrics:
        out.update(transition_metrics)  # tracking_error_early/_late, jerk_early/_late, drift
    return out


def _run_locomotion_metrics(onnx_path: str, step_label: str, skill_id: int | None, kick_aim_enabled: bool, params: dict) -> dict:
    """Table 5 (documents/proposal/benchmark_plan.md) -- locomotion generality: v_x forward/
    backward, v_y lateral, and omega_z yaw command-tracking MAE (each paired with its own
    fall_rate), plus push-recovery rate. `skill_id` is accepted only to match every other
    dispatcher's call signature (see run_eval's own uniform `pool.submit(fn, ...)` call) -- this
    eval type is checkpoint-level, not skill-level (see _PER_TARGET_EVAL_TYPES), so it's always
    None here and never consulted. See mujoco_locomotion_metrics_worker.py's own module docstring
    for why each default magnitude is what it is (all grounded in this project's own trained
    command ranges / push-randomization config, not arbitrary) and for why
    terrain_traversal_success always comes back None (Table 5's 5th row -- no terrain-capable
    MuJoCo scene exists yet, tracked separately, not silently fabricated here).
    """
    del skill_id, kick_aim_enabled  # not applicable to this eval type -- see docstring above
    result = record_locomotion_metrics_scan(
        onnx_path=onnx_path,
        num_episodes=params.get("num_episodes", 5),
        control_hz=params.get("control_hz", 50.0),
        settle_s=params.get("settle_s", 1.0),
        command_s=params.get("command_s", 3.0),
        warmup_s=params.get("warmup_s", 1.0),
        fall_height_m=params.get("fall_height_m", 0.70),
        forward_speed=params.get("forward_speed", 0.8),
        backward_speed=params.get("backward_speed", -0.8),
        lateral_speed=params.get("lateral_speed", 0.5),
        yaw_rate=params.get("yaw_rate", 0.5),
        push_force_n=params.get("push_force_n", 80.0),
        push_duration_s=params.get("push_duration_s", 0.20),
        push_recovery_window_s=params.get("push_recovery_window_s", 2.0),
        seed=params.get("seed", 0),
        timeout_s=params.get("timeout_s", 300.0),
        lock_path=_fresh_lock_path("locomotion_metrics"),
    )
    if result is None:
        return {}
    return {
        "v_x_forward_mae": result["v_x_forward"]["mae"],
        "v_x_forward_fall_rate": result["v_x_forward"]["fall_rate"],
        "v_x_backward_mae": result["v_x_backward"]["mae"],
        "v_x_backward_fall_rate": result["v_x_backward"]["fall_rate"],
        "v_y_lateral_mae": result["v_y_lateral"]["mae"],
        "v_y_lateral_fall_rate": result["v_y_lateral"]["fall_rate"],
        "omega_z_yaw_mae": result["omega_z_yaw"]["mae"],
        "omega_z_yaw_fall_rate": result["omega_z_yaw"]["fall_rate"],
        "push_recovery_rate": result["push_recovery"]["recovery_rate"],
        "terrain_traversal_success": result["terrain_traversal_success"],
    }


_DISPATCH = {
    "kick_survival": _run_kick_survival,
    "kick_to_loco_flip": _run_kick_to_loco_flip,
    "loco_to_kick_handoff": _run_loco_to_kick_handoff,
    "locomotion_metrics": _run_locomotion_metrics,
    "ref_clip_recovery": _run_ref_clip_recovery,
}


_CSV_FIELDNAMES = ["iteration", "step", "target", "skill_id", "eval_type", "group", "metric", "wandb_key", "value"]


def _sanitize_run_subdir(name: str) -> str:
    """A wandb run `name` is a free-form string (e.g. a user-supplied `run_name` could in
    principle contain "/") but this same string also becomes a filesystem directory NAME (one
    path segment), not a path, under `_write_output_files` -- so any path separator in it is
    replaced with "_" rather than silently nesting extra directories or (with "..") escaping
    `output_dir` entirely."""
    return name.replace(os.sep, "_").replace("/", "_")


def _write_output_files(
    output_dir: str,
    run_timestamp: str,
    run_name: str,
    cfg: Sim2SimEvalConfig,
    config_path: str | None,
    target_infos: list[tuple[OnnxTarget, str, int, list[int]]],
    all_rows: list[dict],
    logged_count: int,
    wandb_info: dict,
) -> tuple[str, str]:
    """Writes the two local, wandb-independent records of one run_eval() invocation, under
    `output_dir/<run_name>/` -- `run_name` is the SAME string used for wandb's own run `name`
    (`<run_timestamp>-<wandb_run_name>`, or the bare timestamp -- see run_eval's own WANDB/LOCAL
    OUTPUT FILES docstring notes), so a wandb run and its on-disk record share one identifier,
    findable by matching folder name to wandb run name directly rather than a filename convention
    to remember. Returns (csv_path, json_path).

    CSV = every single (iteration, target, skill, eval, metric) data point actually logged --
    literally the same rows `wandb.log` received, one row per data point, in long format, so it
    loads straight into `pandas.read_csv` for anyone who wants to slice/replot without wandb.

    JSON = one summary block per (target, skill_id, eval_group, metric) -- n/mean/std(population,
    matching the ddof=0 convention this project's own numpy-computed `_std`-suffixed metrics
    already use elsewhere, e.g. mujoco_kick_survival_scan.py's ball_speed_std)/min/max/last,
    collapsed across iterations -- plus run metadata (config path, resolved target list, wandb run
    id/url). The "read this one file to know how the run went" artifact; the CSV carries the
    row-level detail this necessarily throws away.

    Called from run_eval's own `finally` block, so a mid-run crash or Ctrl-C still leaves whatever
    was actually collected (`all_rows` up to that point) on disk rather than losing a long run's
    partial results outright -- an empty or partial `all_rows` still produces valid (if sparse)
    output files, not an error.
    """
    run_output_dir = os.path.join(output_dir, _sanitize_run_subdir(run_name))
    os.makedirs(run_output_dir, exist_ok=True)
    csv_path = os.path.join(run_output_dir, "sim2sim_eval.csv")
    json_path = os.path.join(run_output_dir, "sim2sim_eval.json")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    grouped: dict[tuple[str, int, str, str], list[float]] = defaultdict(list)
    for r in all_rows:
        grouped[(r["target"], r["skill_id"], r["group"], r["metric"])].append(r["value"])

    results_summary: dict = {}
    for (name, skill_id, group, metric), values in grouped.items():
        n = len(values)
        entry = {
            "n": n,
            "mean": statistics.fmean(values),
            "std": statistics.pstdev(values) if n > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "last": values[-1],
        }
        results_summary.setdefault(name, {}).setdefault(str(skill_id), {}).setdefault(group, {})[metric] = entry

    summary = {
        "run_timestamp": run_timestamp,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": config_path,
        "iterations": cfg.iterations,
        "max_concurrent_scans": cfg.max_concurrent_scans,
        "logged_count": logged_count,
        "wandb": wandb_info,
        "targets": [
            {
                "name": name,
                "path": target.path,
                "base_step": base_step,
                "skill_ids": skill_ids,
                "kick_aim_enabled": target.kick_aim_enabled,
            }
            for target, name, base_step, skill_ids in target_infos
        ],
        "results": results_summary,
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    return csv_path, json_path


# --------------------------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------------------------- #


def run_eval(
    cfg: Sim2SimEvalConfig,
    wandb_mode: str | None = None,
    run_timestamp: str | None = None,
    output_dir: str | None = None,
    config_path: str | None = None,
) -> int:
    """Runs every (onnx target x evaluation x skill) combination -- `cfg.iterations` times each --
    and logs every result to ONE wandb run covering the whole invocation. Returns the number of
    (metric, value) pairs actually logged -- 0 means every single scan failed to produce a usable
    result (busy lock, timeout, crash), which the caller should treat as a real failure even
    though no exception was ever raised (every record_* wrapper this script calls is itself
    designed to never raise -- see each module's own docstring -- so a systemic problem here would
    otherwise be silent).

    ITERATIONS (`cfg.iterations`, default 1 -- a single pass, unchanged from before this existed).
    Re-runs the SAME (fixed) checkpoint(s) through the SAME evaluations `iterations` times, each
    time drawing a fresh, independent batch of `num_trials` trials -- each dispatch call gets
    `seed = params.get("seed", 0) + iteration`, so iteration 0 reproduces the old single-shot
    behavior exactly (offset 0) and every later iteration samples differently rather than repeating
    an identical, flat result. Every iteration logs under the SAME metric keys at `step=iteration`,
    which is exactly what makes wandb draw them as a progressing line chart (wandb's default chart
    x-axis is the logged step) -- one line per (target, eval type, skill), so several checkpoints
    given at once still compare side by side, now across iterations instead of a single point.
    When `iterations > 1` this REPLACES the single-shot step (which is the checkpoint's own
    training step, read from its filename, or a fallback counter) with the iteration index --
    those two x-axis meanings don't compose: the training step doesn't move across iterations
    since it's the same fixed checkpoint being repeated, so keeping it fixed as `step` would just
    overwrite one wandb point instead of drawing a line. Target existence-check and skill
    auto-discovery still happen ONCE per target regardless of `iterations` (not once per
    iteration) -- both are properties of the checkpoint file itself, not of any one trial batch.

    TARGET ORDER WITHIN AN ITERATION: turn-by-turn, not target-by-target -- iteration is the OUTER
    loop and target is the INNER one (turn 1 = onnx_targets[0], turn 2 = onnx_targets[1], ..., then
    the next iteration), not "finish every iteration of target 0, then start target 1". With target
    as the outer loop, a slow multi-target, high-`iterations` run would leave target 1's wandb line
    completely flat/absent until target 0's entire `iterations` count finished -- defeating the
    point of watching several targets progress side by side while the run is still going. 2026-09-01:
    changed from target-outer after exactly this was observed on a real 2-target run.

    CONCURRENCY (`cfg.max_concurrent_scans`, default 1 -- exact prior sequential behavior, unchanged
    unless a caller opts in). 2026-09-06: this whole pipeline is CPU-only by design (no GPU path
    exists anywhere in it -- RoboJuDo's own policy class hardcodes `providers=
    ["CPUExecutionProvider"]`, `mujoco`, not the GPU-batched `mujoco.mjx`, is the simulator, and
    neither conda env this project uses even has CUDAExecutionProvider available), and every
    (target, eval, skill) combination within ONE iteration is an independent subprocess launch --
    so on a machine with real spare cores (confirmed live: 128 cores, ~3 in active use during a
    real 100-iteration run), running them one at a time leaves nearly all of that capacity idle.
    `max_concurrent_scans > 1` submits one iteration's combinations to a thread pool instead of a
    plain loop (threads, not processes: the actual work is `subprocess.run()`, which releases the
    GIL while blocked) -- concurrency stays WITHIN one iteration (the next iteration's own
    submissions wait for this one's to finish), so the turn-by-turn step semantics above are
    unaffected; only the order work COMPLETES in changes, not the order it's SUBMITTED or LOGGED
    in (results are still consumed, and hit wandb.log, in the exact same submission order the old
    sequential loop used, from this one main thread only -- wandb.log is never called
    concurrently). Every dispatch call also gets its OWN never-reused lock file
    (`_fresh_lock_path`) regardless of this setting -- reusing each scan's shared, fixed-path
    default lock would otherwise either serialize this pool's own workers against each other
    (defeating the point) or collide with an unrelated LIVE TRAINING PROCESS's own periodic scans
    (reproduced live, 2026-09-06: two of this tool's own scans were silently skipped this way
    against a real training run in progress, well before concurrency was ever a factor). Pick a
    value with real headroom below full core count -- this machine's own training runs and other
    tenants still need cores too.

    `run_timestamp` (default None -> generated here via `datetime.now()`) -- exposed as a parameter
    purely for testability (a caller, e.g. this file's own test suite, can pin it instead of
    depending on wall-clock time); real callers should just omit it. `output_dir` (default None --
    no local files written at all) and `config_path` (default None -- omitted from the JSON
    summary's metadata) feed `_write_output_files` in the `finally` block below; see this
    function's own LOCAL OUTPUT FILES docstring section above for what gets written and why only
    `main()`'s CLI path opts into a non-None `output_dir` by default.
    """
    run_timestamp = run_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    import wandb

    wandb_kwargs: dict = {
        "project": cfg.wandb_project,
        "group": cfg.wandb_group,
        "job_type": "sim2sim-eval",
        "config": {
            "onnx_targets": [t.path for t in cfg.onnx_targets],
            "evaluations": [{"type": e.type, **e.params} for e in cfg.evaluations],
            "iterations": cfg.iterations,
        },
    }
    if cfg.wandb_entity:
        wandb_kwargs["entity"] = cfg.wandb_entity
    # Always stamped with run_timestamp (see this function's own WANDB/LOCAL OUTPUT FILES
    # docstring notes) -- "<run_timestamp>-<wandb_run_name>" when configured, else the bare
    # timestamp, so repeated invocations of the same config never collide on one wandb run name.
    wandb_kwargs["name"] = f"{run_timestamp}-{cfg.wandb_run_name}" if cfg.wandb_run_name else run_timestamp
    if cfg.wandb_tags:
        wandb_kwargs["tags"] = cfg.wandb_tags
    if wandb_mode:
        wandb_kwargs["mode"] = wandb_mode

    wandb.init(**wandb_kwargs)
    logged_count = 0
    # Every single (metric, value) pair actually logged, across every iteration/target/eval/skill
    # -- the raw material for the CSV `_write_output_files` writes below (one row per entry here,
    # verbatim) and for the JSON summary it derives by grouping these. Populated regardless of
    # whether `output_dir` ends up None (skipping the write) -- the bookkeeping cost is negligible
    # next to a MuJoCo subprocess call, so there's no reason to gate it behind that flag.
    all_rows: list[dict] = []

    # Resolved ONCE per target, before any iteration runs -- existence, name, base_step, and
    # skill_ids are all properties of the checkpoint file itself, not of any one trial batch, and
    # doing this up front (rather than inside the iteration loop below) is what makes the
    # turn-by-turn interleaving across targets possible in the first place.
    fallback_step = 0
    target_infos = []
    for target in cfg.onnx_targets:
        if not os.path.exists(target.path):
            logger.error(f"[sim2sim-eval] onnx path does not exist, skipping target: {target.path}")
            continue
        name = target.name or default_target_name(target.path)
        base_step = infer_step_from_onnx_path(target.path)
        if base_step is None:
            base_step = fallback_step
            fallback_step += 1

        if target.skill_ids is not None:
            skill_ids = target.skill_ids
        else:
            try:
                num_skills = discover_num_skills(target.path)
            except Exception:
                logger.exception(
                    f"[sim2sim-eval] failed to read skill metadata from {target.path} -- "
                    "assuming 1 skill. Pass skill_ids explicitly to override."
                )
                num_skills = 1
            skill_ids = list(range(num_skills))
        logger.info(f"[sim2sim-eval] target={name!r} base_step={base_step} skills={skill_ids}")
        target_infos.append((target, name, base_step, skill_ids))

    try:
        # cfg.max_concurrent_scans (default 1 = exact prior sequential behavior) -- one pool for
        # the WHOLE run, reused across iterations, rather than spun up fresh per iteration. Every
        # dispatch call already gets its own never-reused lock file (_fresh_lock_path), so
        # concurrent workers can never collide with each other OR with an unrelated LIVE training
        # process's own periodic scans -- see run_eval's own CONCURRENCY docstring note below for
        # why that matters (this was, before that fix, a real source of silently skipped scans).
        with ThreadPoolExecutor(max_workers=cfg.max_concurrent_scans) as pool:
            # iteration is the OUTER loop, target the INNER one -- turn-by-turn across targets
            # within each iteration (see the TARGET ORDER note in this function's own docstring).
            # Each iteration's own work is fully collected before the next iteration starts --
            # concurrency is WITHIN one iteration's (target x eval x skill) combinations, not
            # across iterations, keeping the turn-by-turn wandb step semantics unchanged.
            for iteration in range(cfg.iterations):
                if cfg.iterations > 1:
                    logger.info(f"[sim2sim-eval] iteration={iteration}/{cfg.iterations}")

                # Flattened and submitted in SUBMISSION order (== the exact order the old
                # sequential loop would have run them in), so results below are consumed/logged
                # in a stable, readable order regardless of which worker actually finishes first.
                futures = []
                for target, name, base_step, skill_ids in target_infos:
                    # See the ITERATIONS note in this function's own docstring for why these two
                    # branches use different step values.
                    step = iteration if cfg.iterations > 1 else base_step
                    step_label = f"{base_step}/iter{iteration}" if cfg.iterations > 1 else str(base_step)
                    for ev in cfg.evaluations:
                        fn = _DISPATCH[ev.type]
                        if ev.type in _PER_TARGET_EVAL_TYPES:
                            # One dispatch for the whole checkpoint, not one per skill_id -- see
                            # _PER_TARGET_EVAL_TYPES' own comment. skill_id=None here (not 0 or any
                            # real skill index) is what tells the logging loop below to build a
                            # skill_id-free wandb key instead of embedding a misleading skill index
                            # into a metric that has nothing to do with any one skill.
                            iter_params = dict(ev.params)
                            iter_params["seed"] = ev.params.get("seed", 0) + iteration
                            logger.info(f"[sim2sim-eval]   submitting {ev.type} (target={name!r}, per-target) ...")
                            future = pool.submit(fn, target.path, step_label, None, target.kick_aim_enabled, iter_params)
                            futures.append((name, step, ev, None, future))
                        else:
                            for skill_id in skill_ids:
                                iter_params = dict(ev.params)
                                iter_params["seed"] = ev.params.get("seed", 0) + iteration
                                logger.info(f"[sim2sim-eval]   submitting {ev.type} skill_id={skill_id} (target={name!r}) ...")
                                future = pool.submit(fn, target.path, step_label, skill_id, target.kick_aim_enabled, iter_params)
                                futures.append((name, step, ev, skill_id, future))

                for name, step, ev, skill_id, future in futures:
                    try:
                        results = future.result()
                    except Exception:
                        logger.exception(f"[sim2sim-eval] {ev.type} crashed for target={name!r} skill_id={skill_id}")
                        continue
                    group = _EVAL_TYPE_GROUP[ev.type]
                    log_dict = {}
                    for metric, value in results.items():
                        if value is None:
                            continue
                        # skill_id is None for _PER_TARGET_EVAL_TYPES (see run_eval's own
                        # submission-loop branch above) -- omit that segment entirely rather than
                        # embed a misleading "None" into every wandb key/panel title.
                        key = f"{name}/{group}/{metric}" if skill_id is None else f"{name}/{skill_id}/{group}/{metric}"
                        log_dict[key] = value
                        all_rows.append({
                            "iteration": iteration,
                            "step": step,
                            "target": name,
                            "skill_id": skill_id,
                            "eval_type": ev.type,
                            "group": group,
                            "metric": metric,
                            "wandb_key": key,
                            "value": value,
                        })
                    if log_dict:
                        wandb.log(log_dict, step=step)
                        logged_count += len(log_dict)
                        logger.info(f"[sim2sim-eval]   -> {log_dict}")
                    else:
                        logger.warning(
                            f"[sim2sim-eval]   {ev.type} skill_id={skill_id} produced no usable result "
                            "(busy lock, timeout, or crash inside the scan -- see warnings above)"
                        )
    finally:
        # Captured BEFORE wandb.finish() -- wandb.run typically goes away (or becomes unreliable
        # to introspect) once the run is finished. getattr-guarded throughout: this file's own
        # test suite's _FakeWandb stand-ins never define a `run` attribute at all, and this must
        # not raise inside a finally block regardless.
        wandb_info = {
            "project": cfg.wandb_project,
            "entity": cfg.wandb_entity,
            "group": cfg.wandb_group,
            "run_name": wandb_kwargs.get("name"),
            "mode": wandb_mode or "online",
        }
        run_obj = getattr(wandb, "run", None)
        if run_obj is not None:
            wandb_info["run_id"] = getattr(run_obj, "id", None)
            try:
                wandb_info["url"] = run_obj.get_url()
            except Exception:
                wandb_info["url"] = None

        wandb.finish()

        if output_dir is not None:
            try:
                csv_path, json_path = _write_output_files(
                    output_dir=output_dir,
                    run_timestamp=run_timestamp,
                    run_name=wandb_kwargs["name"],
                    cfg=cfg,
                    config_path=config_path,
                    target_infos=target_infos,
                    all_rows=all_rows,
                    logged_count=logged_count,
                    wandb_info=wandb_info,
                )
                logger.info(f"[sim2sim-eval] wrote {len(all_rows)} data rows -> {csv_path}")
                logger.info(f"[sim2sim-eval] wrote run summary -> {json_path}")
            except Exception:
                logger.exception(
                    "[sim2sim-eval] failed to write local output files (csv/json) -- wandb logging "
                    "above is unaffected either way"
                )

    if logged_count == 0:
        logger.error("[sim2sim-eval] zero metrics logged across every target/evaluation/skill -- treat as a failure.")
    return logged_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to a sim2sim_eval YAML config.")
    parser.add_argument(
        "--onnx-path", action="append", default=[],
        help="Evaluate an additional onnx checkpoint not listed in --config, using "
        "--kick-aim-enabled/--no-kick-aim-enabled and auto-discovered skill_ids. Repeatable. "
        "For anything you intend to re-run or compare later, put it in the config file instead.",
    )
    parser.add_argument(
        "--kick-aim-enabled", dest="kick_aim_enabled", action="store_true", default=True,
        help="Only applies to --onnx-path targets (config targets set this per-target). Default "
        "True, matching every project checkpoint as of 2026-08-22.",
    )
    parser.add_argument("--no-kick-aim-enabled", dest="kick_aim_enabled", action="store_false")
    parser.add_argument(
        "--wandb-mode", default=None, choices=["online", "offline", "disabled"],
        help="Override wandb's run mode -- 'offline' verifies the full pipeline (config parsing, "
        "skill discovery, the real MuJoCo scans, wandb logging) without publishing anywhere; "
        "'disabled' skips wandb entirely. Omit to use wandb's own default (online).",
    )
    parser.add_argument(
        "--output-dir", default="out/sim2sim_eval",
        help="Local base folder to also write this run's results to, alongside wandb: "
        "<output-dir>/<run_timestamp>-<run_name>/sim2sim_eval.csv (every logged data point, long "
        "format) and .../sim2sim_eval.json (per-metric mean/std/min/max/last summary + run "
        "metadata) -- the subfolder is named identically to wandb's own run name, so the two are "
        "trivially paired up. See sim2sim_eval.py's own LOCAL OUTPUT FILES docstring section. "
        "Default 'out/sim2sim_eval' (created if missing). Pass an empty string to skip writing these "
        "entirely.",
    )
    args = parser.parse_args()

    cfg = load_eval_config(args.config)
    for p in args.onnx_path:
        cfg.onnx_targets.append(OnnxTarget(path=p, kick_aim_enabled=args.kick_aim_enabled))

    logged_count = run_eval(
        cfg,
        wandb_mode=args.wandb_mode,
        output_dir=args.output_dir or None,
        config_path=args.config,
    )
    return 0 if logged_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
