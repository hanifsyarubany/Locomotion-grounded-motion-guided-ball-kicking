# Locomotion + Ball-Kicking (G1 + FastSAC + IsaacSim/MuJoCo)

**Current version: v9** (see [Version history](#version-history) for the full changelog).

This workspace trains a Unitree G1 (29-DOF) humanoid with FastSAC/IsaacSim on three related
experiments:

- **`exp:g1-29dof-unified-fast-sac`** — **the main experiment in this workspace**: a single
  policy that learns BOTH locomotion (velocity-command walking) and ball-kicking in one training
  run. Each environment is permanently assigned either `locomotion` or `kick` mode for the whole
  run (not re-rolled per episode — see [v3](#v3--training-quality-fixes) for why). See
  [Unified locomotion + ball-kicking — how it works](#unified-locomotion--ball-kicking--how-it-works) below.
- **`exp:g1-29dof-fast-sac`** — dedicated locomotion only (velocity tracking, no ball).
- **`exp:g1-29dof-ball-kick-fast-sac`** — dedicated ball-kicking only (kick a stationary ball,
  then recover and hold a stable stance for ~3s). Observation/action space byte-identical to
  stock whole-body-tracking (WBT).

| Category | This workspace |
|---|---|
| Simulator (training) | IsaacSim |
| Simulator (sim-to-sim deployment) | MuJoCo |
| Algorithm | FastSAC |
| Robot | Unitree G1 (29-DOF) |

For the general-purpose, multi-simulator/multi-robot/multi-algorithm holosoma pipeline this was
forked from, see the upstream holosoma repo — that broader surface (MJWarp, T1, PPO, etc.) is
intentionally not covered here since it's unused in this fork.

---

## Setup

**Conda env: `hssim`** — this is the env with this fork's `src/holosoma` editable-installed (not
the generic upstream `holosoma` package, and not `hssim_holosoma`, which points at the *other*
`locomotion_and_motion_tracking/holosoma` workspace — training against the wrong env silently runs
the wrong code). `source scripts/source_isaacsim_setup.sh` below does the activation for you
(equivalent to `conda activate hssim`, just via this repo's own conda root), so an explicit
`conda activate` isn't required — but if you ever run a command in a fresh shell without sourcing
that script first, `conda activate hssim` is the one-line fix.

```bash
cd /workspaces/isaaclab_arena/submodules/workspaces/playground/locomotion_and_ball_kicking

# activates the `hssim` conda env + sets OMNI_KIT_ACCEPT_EULA
source scripts/source_isaacsim_setup.sh
```

All training/eval commands below — **including every stage of the [three-stage bootstrap
protocol](#bootstrapping-in-three-stages)** — assume this has been sourced (`hssim` active) and
that you're running from this workspace's root. Pass `--help` to any script to see the full flag
list, e.g. `python src/holosoma/holosoma/train_agent.py --help`.

**Running on a non-primary GPU** (`cuda:1`/`cuda:2`/`cuda:3`): don't pass a device index directly —
IsaacLab's `AppLauncher` (which owns PhysX/rendering) is initialized before this repo's own device
resolution runs, and in single-process mode always defaults to GPU 0 regardless of what device the
env's tensors end up on, so the two can end up on different physical GPUs and the process dies
with no Python traceback. Use `CUDA_VISIBLE_DEVICES=<N>` instead, e.g.:
```bash
CUDA_VISIBLE_DEVICES=1 python src/holosoma/holosoma/train_agent.py ...
```
so both sides agree on `cuda:0` within the process's own restricted view.

---

## Configuration (all read fresh on every launch — edit and re-run, no code changes needed)

- **`configs/skill_mix.yaml`** — `kick_probability` (default `0.5`). For the **unified**
  experiment only: at startup, each flat-terrain-eligible env is **permanently** (for the whole
  training run) assigned kick-mode with this probability, otherwise locomotion-mode (see
  [v3](#v3--training-quality-fixes) for why this is a one-time-per-run partition, not a
  per-episode roll). Realized global kick fraction = `(fraction of envs on flat terrain) x
  kick_probability`, capped at the flat-terrain fraction. Training logs `kick_eligible_frac` /
  `kick_active_frac` each step so the realized rates are visible.
- **`configs/ball.yaml`** — the ball's physical properties, spawn position, and (as of
  [v7](#v7--robonaldo-style-shooting-rewards-and-ball-and-target-observations)) the shooting task's
  target/randomization/staging knobs:
  ```yaml
  radius: 0.11   # meters
  mass: 0.43     # kg
  x: 2.84        # meters, nominal forward spawn (env-local)
  y: -0.46       # meters, nominal lateral spawn (env-local)

  randomize_x: 0.75   # ± meters, per-attempt uniform spawn noise
  randomize_y: 0.75
  kick_foot: right     # "left" or "right" — which foot the shooting rewards attribute the kick to

  target_x: 7.84   # meters, nominal shot target (env-local)
  target_y: -0.46
  randomize_target_x: 1.0   # ± meters, per-attempt uniform target noise
  randomize_target_y: 2.0
  success_radius: 0.5   # meters — shot counted as a hit within this distance of the target

  shooting_reward_scale: 0.8   # RoboNaldo's w_g — global scale on all 6 shooting reward terms; 0.0 disables them
  ```
  `z` isn't configurable — always set to `radius` so the ball rests on the ground. The ball and
  target ARE now observed by the policy (`kick_ball_pos_b`, `kick_target_pos_b` — robot-heading
  frame), which is what makes the randomization ranges learnable variation instead of pure reward
  noise, and what makes the target commandable at deployment. See
  [Shooting rewards & observations](#shooting-rewards--observations-robonaldo-style) below for the
  mechanism, and [Bootstrapping in three stages](#bootstrapping-in-three-stages) for how to use
  `shooting_reward_scale`/the randomization knobs during training.

  **Pointing at a different file**: set the `HOLOSOMA_BALL_CONFIG` env var (not a CLI flag) before
  launching — e.g. `HOLOSOMA_BALL_CONFIG=configs/ball_stageB.yaml python
  src/holosoma/holosoma/train_agent.py ...`. Has to be an env var rather than a `--flag`:
  `config_values/unified/g1/reward.py` reads `shooting_reward_scale` at **module import time** to
  bake the six shooting-reward weights into the tyro config tree, which happens before tyro parses
  the command line — a flag would swap the ball position/randomization correctly but silently
  leave the reward weights built from the default file. An env var is visible before the process
  even starts, so both sides agree. Unset (the default) behaves exactly as before —
  `configs/ball.yaml`. `print_active_kick_rewards.py` prints whichever file actually resolved, so
  you can always confirm before spending GPU time.

  **Ball perception noise model** — `kick_ball_pos_b` (the policy's observed ball position, as
  opposed to the simulator's real ground-truth position the critic and every reward term use)
  goes through a small stack of simulated sensor artifacts, actor-only (`enable_noise=False` on
  the critic group means it always sees the clean instantaneous value regardless of any of this).
  Set these as top-level keys in `configs/ball.yaml` (legacy single-skill mode) or in whichever
  file supplies `MultiSkillConfig`'s global fields (`configs/task/task_config_stageX.yaml` in
  2-file N-skill mode — see the Configuration section above for 1-file vs. 2-file mode):
  ```yaml
  observation_bias: 0.05                 # ball.yaml key; MultiSkillConfig: observation_bias
  ball_obs_noise: 0.05                   # ball.yaml key: observation_noise
  ball_obs_noise_range_coefficient: 0.03 # ball.yaml key: observation_noise_range_coefficient
  ball_obs_delay_steps_min: 0            # ball.yaml key: observation_delay_steps_min
  ball_obs_delay_steps_max: 3            # ball.yaml key: observation_delay_steps_max
  ball_obs_hold_steps_min: 0             # ball.yaml key: observation_hold_steps_min
  ball_obs_hold_steps_max: 0             # ball.yaml key: observation_hold_steps_max
  ball_obs_stale_probability: 0.01       # ball.yaml key: observation_stale_probability
  ball_static_obs_probability: 0.0       # ball.yaml key: unchanged, same name both modes
  ```
  (Shown with `MultiSkillConfig`'s field names; the legacy single-skill `BallConfig` reads the
  same knobs under the `observation_*`-prefixed names noted in each comment, everything else
  identical.)
  - **`observation_bias`** — a per-episode constant heading-frame offset, drawn once and held for
    the whole episode. Models a fixed calibration error, not per-step jitter.
  - **`ball_obs_noise`** / **`ball_obs_noise_range_coefficient`** — flat per-step uniform noise
    magnitude, plus an additional magnitude proportional to the ball's own observed distance
    (`effective_scale = noise + range_coefficient * distance`) — real depth/stereo sensors get
    noisier the farther away the target is.
  - **`ball_obs_delay_steps_{min,max}`** — a per-episode-fixed random transport latency (control
    steps), redrawn at every reset. Models camera/LiDAR capture + fusion lag on a pipeline that
    still updates every tick.
  - **`ball_obs_hold_steps_{min,max}`** — zero-order hold: the reading only *refreshes* every N
    control steps (N drawn per-episode from this range), staying frozen in between. Models a
    fusion pipeline whose own update rate is slower than the 50 Hz control loop (e.g. a 25 Hz
    estimator → `(2, 2)`).
  - **`ball_obs_stale_probability`** (2026-08-23) — per-**control-tick** probability (re-rolled
    every step, unlike the hold period above) of reusing the *previous* tick's already-processed
    reading instead of computing a fresh one — models a single dropped/repeated sensor frame
    (e.g. one skipped LiDAR packet). A run of consecutive stale ticks keeps returning the same
    frame, never drifting to progressively older ones. Cross-checked against RoboNaldo's own
    source (`lidar_stale_probability`) — see `MultiSkillConfig.ball_obs_stale_probability`'s
    docstring for the full comparison and staging discussion. `0.01` is enabled by default across
    every stage in this project's own shipped `task_config_*.yaml` files as of 2026-08-23.
  - **`ball_static_obs_probability`** — per-episode probability the reading freezes at an
    **independently-drawn synthetic** value (decoupled from the real ball, may land anywhere
    including out-of-distribution) for the *rest* of the episode — models a fully broken/stuck
    sensor, a stronger fault than the single-dropped-frame case above.

  See `ObservationManager._apply_noise`/`_apply_delay`/`_apply_hold`/`_apply_stale`
  (`managers/observation/manager.py`) for the exact mechanics, applied in that order (noise →
  delay → hold → stale), and `ball_pos_b`/`randomize_ball_obs_freeze`
  (`managers/observation/terms/unified.py`, `managers/randomization/terms/locomotion.py`) for the
  bias and static-freeze mechanisms specifically.
- **`configs/stabilization.yaml`** — post-kick recovery/hold timing:
  ```yaml
  recovery_duration_s: 1.0  # smooth transition from the kick's ending pose to default pose
  hold_duration_s: 2.0      # then a genuinely static balance hold (1.0 + 2.0 = 3s total)
  ```

---

## Where the kick motion comes from

The kick experiments' motion command points at either a single `.npz` (`motion_file=`) or a
directory of them (`motion_dir=`). If you produced a clip via the `video2robot` pipeline (see
`../../locomotion_and_motion_tracking/video2robot/`), point at the converted output, e.g. the
directory this session has been using:
```
--command.setup_terms.motion_command.params.motion_config.motion_dir=../../locomotion_and_motion_tracking/video2robot/data/video_011/npz_holosoma
```

**Before spending GPU time on a full run**, visualize the clip to confirm the kicking foot's
trajectory actually reaches the ball's configured position — a kick clip that never touches the
ball will train for hours without anyone noticing:

```bash
python src/holosoma/holosoma/replay.py \
    exp:g1-29dof-ball-kick-fast-sac \
    --command.setup_terms.motion_command.params.motion_config.motion_dir=/path/to/your/npz_dir \
    --training.headless=False \
    --training.num_envs=1
```

### Interactive replay controls

**Terminal playback control** — type a command + Enter in the terminal you launched `replay.py`
from:

| Command | Effect |
|---|---|
| `p` / `play` | resume playback |
| `s` / `stop` | pause playback |
| `r` / `restart` | jump back to frame 0 of the clip and reset the ball, then resume playing |
| `q` / `quit` | exit |

Playback also auto-pauses when the clip reaches its end (recovery + hold included) — type `r` to
watch it again from the top.

**Ball Position window** — a floating panel next to the IsaacSim viewport (only appears when the
experiment has a ball configured and `--training.headless=False`) with X/Y sliders:

- Dragging a slider moves the ball live in the running sim so you can see immediately whether the
  kicking foot lines up with it.
- **Save to ball.yaml** writes the current slider values into `configs/ball.yaml` in place,
  preserving the file's comments — the next run (training or replay) picks it up automatically.

Typical workflow: launch replay, let the clip play once to see where the kicking foot lands, drag
the ball sliders to that spot, hit Save, then `r` to restart and confirm visually, repeating until
the foot connects. (A **fully headless/scripted** version of this same geometric check — no
viewport needed — is also available; see [Headless kick-geometry check](#headless-kick-geometry-check-no-viewport).)

---

## Training

### Unified (locomotion + kicking, one policy)

```bash
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-unified-fast-sac \
    logger:wandb \
    --command.setup_terms.motion_command.params.motion_config.motion_dir=/path/to/your/npz_dir \
    --training.headless=True
```

**Memory note**: `num_envs` defaults to 2048 here (not 4096 like the kick-only/stock experiments)
— FastSAC's replay buffer preallocates `(num_envs, buffer_size, obs_dim)` tensors up front, and
the unified merged observation space (256-dim actor obs) is meaningfully bigger than either
source task alone, so 4096 envs OOMs on a 32GB GPU. Raise `--training.num-envs` if your GPU has
more headroom, or lower `--algo.config.buffer-size` (default 1024) to claw back room.

**Video recording — two videos, not one**: with `logger:wandb`, the wandb media panel gets *two*
"Training rollout" videos each interval — `Training rollout - Locomotion` and
`Training rollout - Kick` — both permanently pinned to their task (never subject to the
`kick_probability` partition), so every recorded episode reliably shows the right behavior.

**Stage B/C only — two more, MuJoCo/RoboJuDo sim2sim rollouts**: whenever `kick_probability > 0`
(never on Stage A), every `mujoco_kick_rollout_every_n_saves`-th checkpoint save (default 5) also
logs `Training rollout - MuJoCo Kick` (stand → kick-trigger → hold, `record_mujoco_kick_rollout.py`)
and `Training rollout - MuJoCo Walk` (5s forward walk → 3s stand, `record_mujoco_locomotion_rollout.py`,
2026-07-21) — both run out-of-process under the `robojudo` conda env with a real physical ball
present in the scene in every stage, an early-warning signal for the PhysX↔MuJoCo sim2sim gap that
the two IsaacSim-only videos above can't show. See `FastSACConfig.mujoco_kick_rollout_every_n_saves`.

**Stage B/C only — MuJoCo survival scan (numeric, not video)**: on the same cadence family
(`mujoco_survival_scan_every_n_saves`, its own independent knob), `mujoco_kick_survival_scan.py`
runs N deployed-policy trials (settle → trigger → hold, real MuJoCo contact physics, no video) and
logs three scalars per skill under `Kick_skills_{i}/sim2sim/...`:
  - `kick_fall_rate` — fraction of trials where base height drops below the fall threshold. A
    genuine complement to the IsaacSim-side `Env/kick_topple_frac`, not a replacement — this one
    runs the *deployed* deterministic action (not SAC's exploration noise), isn't subject to
    other-termination censoring, and reports an exact N-trial count instead of a decaying average
    (so the two will not numerically match; a real PhysX↔MuJoCo gap is expected).
  - `kick_ball_hit_rate` — fraction of trials with a real MuJoCo ball↔foot geom contact (direct
    contact-solver read, not an approximation).
  - `kick_direction_success_rate` (2026-08-23, `kick_aim_enabled` skills only) — of the trials that
    actually hit the ball, what fraction landed within 1.0m (mirroring `error_ball_to_target`'s own
    default `sigma` — roughly 11.5° at the default `kick_aim_nominal_distance_m=5.0`) of the
    commanded `kick_aim_theta` target. Deliberately gated on real contact (not `/num_trials`) so a
    checkpoint that rarely connects doesn't read as "bad aim" when the real problem is contact rate
    — that's what `kick_ball_hit_rate` already measures. Absent from wandb (not logged as 0.0)
    whenever a scan's N trials produced zero hits — there's nothing to grade direction on yet, and
    a missing point reads correctly on a chart in a way a fabricated 0.0 would not.

See `FastSACConfig.mujoco_survival_scan_every_n_saves`/`mujoco_survival_scan_num_trials`, and
`mujoco_kick_survival_scan.py`'s own module docstring for the full per-trial mechanics.

### Dedicated locomotion only

```bash
python src/holosoma/holosoma/train_agent.py exp:g1-29dof-fast-sac logger:wandb --training.headless=True
```

### Dedicated ball-kicking only

```bash
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-ball-kick-fast-sac \
    logger:wandb \
    --command.setup_terms.motion_command.params.motion_config.motion_dir=/path/to/your/npz_dir \
    --training.headless=True
```

Checkpoints are written under `logs/<Project>/<timestamp>-<training.name>-<task>/model_<iter>.pt`
(`.onnx` is exported alongside automatically every save).

### Resuming / continuing training from a checkpoint

`--training.checkpoint <path>` restores network weights and optimizer state only — the CLI config
(motion dir, ball config, reward weights, etc.) is **not** restored from the checkpoint, so pass
the same flags you'd use for a fresh run (plus whatever you're deliberately changing).

```bash
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-unified-fast-sac \
    logger:wandb \
    --command.setup_terms.motion_command.params.motion_config.motion_dir=/path/to/your/npz_dir \
    --training.checkpoint logs/LocomotionAndBallKicking/<timestamp>-.../model_<iter>.pt
```

`--training.checkpoint` also accepts a W&B checkpoint URI:
`wandb://<ENTITY>/<PROJECT>/<RUN_ID>/<CHECKPOINT_NAME>`.

**Resuming is not always the cheaper option — watch for distribution-shift changes.**
`FastSACAgent.load()` restores `log_alpha` (SAC's auto-tuned entropy/exploration coefficient) and
both optimizer states along with the network weights, but **not** the replay buffer (each resume
starts with an empty one, so all near-term experience already reflects whatever you just
changed). A reward-term addition is a small, same-dynamics change — resuming works well for that
(see [v4](#v4--mujoco-sim-to-sim-deployment--stance-symmetry-reward)'s
`penalty_stance_asymmetry`, [v6](#v6--zero-velocity-heading-drift-reward)'s
`penalty_yaw_drift`). But a change to the environment *dynamics themselves* (e.g. newly enabling
PD-gain/action-delay randomization, [v5](#v5--pd-gaindelay-domain-randomization-for-mujoco-robustness))
is a bigger deal: a checkpoint converged under the old (narrower) dynamics also has an already
low, annealed exploration coefficient, so it has little exploration noise left to work with while
un-learning old assumptions and building in the new robustness — resuming visibly regressed
before we caught this and restarted from scratch instead. Rule of thumb: reward tweaks → resume;
randomization/dynamics changes → strongly consider training from scratch instead.

**Bootstrapping locomotion first, then adding kicking (and, from there, shooting adaptation)**:
because the unified observation/action space is identical regardless of `kick_probability` or
`configs/ball.yaml`'s shooting settings (only which reward/termination/command paths are active
per episode changes, never tensor shapes), you can train in stages on the *same* checkpoint
lineage. See [Bootstrapping in three stages](#bootstrapping-in-three-stages) below for the full
locomotion → kick-tracking → shooting-adaptation protocol with exact commands.

You can also continue a **stock dedicated-locomotion** checkpoint (`exp:g1-29dof-fast-sac`)
*into* the unified experiment this same way — the network architecture and locomotion observation
terms are compatible (unified's `loco_*` terms are the identical stock terms, just prefixed) —
though as of this writing that cross-experiment resume path hasn't been separately validated in
this workspace; test on a short run first if you try it.

### Naming your run (and getting local-folder / wandb names to match)

The local checkpoint folder name and the wandb display name are generated **independently** and
use different naming schemes by default:
- Local folder: `get_experiment_dir(...)` → `<timestamp>-<training.name>-<task_name>` (dashes;
  `task_name` comes from the env class, e.g. `locomotion` or `unified_locomotion_kick` — not
  directly configurable).
- WandB name: `<timestamp>_<training.name>_<group>_<robot_type>` (underscores; includes a
  `group`/`robot_type` suffix you didn't ask for).

To force a specific, predictable wandb display name (fully overriding the auto-generated one),
pass `--logger.name`:

```bash
python src/holosoma/holosoma/train_agent.py exp:g1-29dof-unified-fast-sac logger:wandb \
    --training.name my_run --logger.name my_run \
    ...
```

This makes the wandb run display exactly as `my_run`; the local folder will still be
`<timestamp>-my_run-unified_locomotion_kick` (the timestamp prefix and task-name suffix on the
local folder always apply — they exist to keep concurrent/repeated runs from colliding on disk).

---

## Evaluation

### Scripted headless probe (quantitative, no viewport)

`eval_probe.py` is a non-interactive way to get quantitative numbers on a checkpoint — e.g.
checking progress mid-training over SSH:

```bash
python src/holosoma/holosoma/eval_probe.py \
    --checkpoint logs/LocomotionAndBallKicking/<timestamp>-.../model_<iter>.pt \
    --num-envs 32 --zero-vel-seconds 8 --kick-seconds 9
```

Runs two headless rollouts and prints JSON:
- **`zero_velocity`** — forces a `(0, 0, 0)` locomotion command and reports mean/max base
  linear/angular velocity, joint-velocity RMS, standing-height stability, and fall fraction. A
  policy that's actually standing still should show near-zero velocities and near-constant height.
- **`kick`** (UnifiedManager checkpoints only) — calls `trigger_kick()` and reports minimum
  foot-to-ball distance, how much the ball moved, and whether the robot fell during the post-kick
  stabilization window. Automatically restricted to flat-terrain-eligible envs.

### Headless kick-geometry check (no viewport)

A scripted, non-interactive alternative to the `replay.py` + Ball-Position-window workflow above,
for confirming the kick clip's foot trajectory reaches the ball's configured position without
needing a GUI at all — kinematically replays the clip (no policy, no RL) and reports the minimum
foot-to-ball distance achieved. Useful over SSH / in CI:

```bash
python src/holosoma/holosoma/check_kick_geometry.py --motion-dir /path/to/your/npz_dir
# or: --motion-file /path/to/clip.npz
```

Prints a JSON block including `min_foot_ball_dist_mean/min/max` and
`frac_envs_within_10cm/20cm` — if these stay stuck near your ball's spawn radius across the whole
clip, the foot never reaches it and no amount of training will fix that on its own; move the ball
(`configs/ball.yaml`) or re-check the clip's retargeting.

### Interactive policy deployment (IsaacSim)

For open-ended, hands-on inspection of a trained checkpoint — watching whether the policy stays
upright after the kick, adjusting the ball position live:

```bash
python src/holosoma/holosoma/eval_interactive.py \
    --checkpoint=logs/LocomotionAndBallKicking/<timestamp>-.../model_<iter>.pt
```

(Also accepts a W&B checkpoint URI.) Headless=False and num_envs=1 by default.

- **Terminal controls** (type + Enter): `p`/`play`, `s`/`stop`, `r`/`restart` (resets the whole
  episode — robot to initial pose, ball to configured position, motion reference to frame 0 — and
  lets the **policy** drive from there), `q`/`quit`. Playback auto-pauses whenever the episode
  ends (fell, or a termination tripped) so a bad rollout doesn't silently reset and march on
  unnoticed.
- **`k` / `kick`** (UnifiedManager checkpoints only) — forces a full reset into the kick task's
  starting pose and resumes playback. This is a genuine reset (matching how kick-mode episodes are
  trained), not a live mid-stride switch. By default the robot runs locomotion; typing `k` at any
  point switches it into kicking. For non-unified checkpoints, `k` prints a message and does
  nothing.
- **Ball Position window** — identical to `replay.py`'s: drag to move the ball live, **Save to
  ball.yaml** to persist it.
- **Locomotion Velocity window** — linear/angular velocity sliders to drive the robot live (any
  checkpoint with a locomotion command registered — stock locomotion, or the locomotion side of
  the unified experiment).

The key difference from `replay.py`: there, the robot is teleported directly from the motion clip
every frame (pure kinematic visualization, no gravity/contacts). Here, the loaded policy's own
actions drive the robot through real physics, so whether it stays standing is a genuine test of
that checkpoint's robustness.

---

## Sim-to-sim deployment (MuJoCo, two terminals)

Deploys a trained unified (or stock locomotion) ONNX checkpoint against a **different** physics
simulator (MuJoCo) than it was trained in (IsaacSim) — the standard "does this transfer" check
before considering real-hardware deployment. Two separate terminals/processes talking over a
loopback SDK interface, matching how the eventual real-robot deployment would also work.

### One-time setup

The MuJoCo-side tooling (`run_sim.py`) lives in the reference `holosoma` checkout, and the
inference-side tooling (`holosoma_inference`, `run_policy.py`) has been forked into **this**
workspace at `src/holosoma_inference/` with the new unified-policy support added (see
[v4](#v4--mujoco-sim-to-sim-deployment--stance-symmetry-reward)). Conda env: `hsinference_unified`
(cloned from the reference `hsinference` env, `holosoma_inference` editable-installed from this
workspace's fork instead of the reference copy — keeps the reference env untouched).

### Terminal 1 — MuJoCo physics

```bash
cd /workspaces/isaaclab_arena/submodules/workspaces/locomotion_and_motion_tracking/holosoma
source scripts/source_mujoco_setup.sh   # activates hsmujoco
python src/holosoma/holosoma/run_sim.py robot:g1-29dof
```

The robot spawns hanging from a gantry (elastic-band safety tether). In the MuJoCo window:
- `8` — lower the gantry (repeat until the robot's feet touch the ground)
- `9` — toggle/remove the gantry once stable
- `BACKSPACE` — reset the simulation

### Terminal 2 — policy inference

```bash
cd /workspaces/isaaclab_arena/submodules/workspaces/playground/locomotion_and_ball_kicking/src/holosoma_inference
conda activate hsinference_unified
python3 holosoma_inference/run_policy.py inference:g1-29dof-unified-loco-kick \
    --task.model-path /path/to/your/model_<iter>.onnx \
    --task.interface lo
```

Terminal-only control (no joystick needed) — everything below is typed in **this** terminal
(make sure it has keyboard focus, not the MuJoCo window):

| Key | Action |
|---|---|
| `]` | start the policy |
| `o` | stop the policy |
| `i` | interpolate to default pose |
| `w` / `s` | increase/decrease forward velocity |
| `a` / `d` | increase/decrease lateral velocity |
| `q` / `e` | increase/decrease angular (turn) velocity |
| `z` | zero all velocity commands |
| `k` | **trigger kick** (one-shot; plays the full kick+recovery+hold clip) |
| `l` | **return to locomotion control** (also happens automatically ~a few seconds after the kick clip settles into its hold pose — no need to remember this key, it's a safety net) |

For a stock locomotion-only checkpoint instead, use `inference:g1-29dof-loco` in place of
`inference:g1-29dof-unified-loco-kick` (same two-terminal workflow, same `hsmujoco`/reference env
on the simulator side — no fork needed for that path).

### Known limitations as of v6 — sim-to-sim standing robustness

Three related but distinct symptoms have shown up in MuJoCo deployment that IsaacSim itself does
not show (or shows far less severely) on the same checkpoint — all under active investigation
via [v5](#v5--pd-gaindelay-domain-randomization-for-mujoco-robustness)/
[v6](#v6--zero-velocity-heading-drift-reward):

1. **Hip-roll (stance-width) divergence** — a checkpoint standing with near-neutral,
   symmetric hip-roll in IsaacSim (~±2°) showed a much wider, slowly-diverging stance in MuJoCo
   (one leg drifting outward from -10° to -34° over ~10s), a visible "V-shape". Ruled out:
   default-angle mismatches, `dof_names` ordering, `motor2joint`/`joint2motor` mapping, left/right
   PD-gain asymmetry — all matched exactly between training and inference configs.
2. **Single-leg standing** — repeated walk→stop cycling in MuJoCo (single continuous session,
   randomized commands) deterministically ends every cycle with one foot permanently lifted
   (double-support never achieved), while the same test in IsaacSim settles cleanly to a static
   stand in effectively every trial (0 failures across 704+ trials pre-v5). Backward-walking
   commands specifically triggered several seconds of repeated foot-replanting before settling —
   [v5](#v5--pd-gaindelay-domain-randomization-for-mujoco-robustness)'s randomization fix shortened
   this (isolated bursts instead of spanning the whole window) but did not eliminate the
   underlying single-leg bias (it just flipped which foot).
3. **Heading (yaw) drift while standing** — unlike 1-2, this one *does* reproduce in IsaacSim
   directly (mean ~12.6°, max ~96° heading change over a 6s zero-command stand), so it's a genuine
   reward-shaping gap, not a MuJoCo-only artifact — see
   [v6](#v6--zero-velocity-heading-drift-reward) for the root cause and fix (results pending at
   time of writing).

Diagnostic method for all three: construct the MuJoCo simulator (`holosoma`'s `DirectSimulation`)
and the deployment-side policy (`holosoma_inference`'s `UnifiedPolicy`) in **one Python process**
(eliminates a multi-second inter-process startup gap that otherwise confounds the very first
few seconds of any test), inject scripted velocity commands instead of a real keyboard, and read
`simulator.contact_forces` / `simulator.root_data.qpos` directly — after calling
`simulator.refresh_sim_tensors()` explicitly, since it's a non-proxy tensor the bare deployment
physics loop never refreshes on its own — to measure foot-contact switching and base
position/orientation over time. Run the *same* scripted scenario against the *same* checkpoint in
IsaacSim (via `eval_probe.py`-style headless rollouts) as the decisive control: present in both
sims → policy/reward problem; present only in MuJoCo → deployment/sim-to-sim gap.

---

## Unified locomotion + ball-kicking — how it works

Each environment is (as of [v3](#v3--training-quality-fixes)) assigned either `locomotion` or
`kick` task mode **once, permanently, for the entire training run** — not re-rolled every
episode. Kick-mode is only ever assigned to envs sitting on **flat terrain** (a freely-simulated
ball needs flat ground to rest at its configured position) — see `terrain_unified_mix` in
`config_values/terrain.py`, which raises the flat-terrain proportion to 40% specifically so
there's enough eligible envs. Locomotion-mode envs get the remaining rough/obstacle terrain for
gait robustness.

The observation space is the union of both tasks' observations — every term is zeroed (never
omitted; the vector width never changes, 261-dim total as of
[v7](#v7--robonaldo-style-shooting-rewards-and-ball-and-target-observations)) for envs not currently
running that task — plus a `task_mode_onehot` term so the policy can explicitly condition its
behavior on which task it's in. Locomotion terms are prefixed `loco_*`, kick terms `kick_*`
(several source term names collide, e.g. `dof_pos`/`actions`, hence the prefixing). Action space
is unchanged (both tasks already used the identical joint-position action space).

**Post-kick stabilization** (the ~3s "stand stably without falling" requirement) is implemented
via reward/reference engineering, not a hardcoded freeze:
1. **During the kick clip**: the policy tracks the reference motion as in stock WBT.
2. **Recovery** (`recovery_duration_s`, default 1.0s): reference pose smoothly transitions from
   the clip's last frame back to the robot's default standing pose — still a moving target the
   policy tracks with the existing tracking-reward terms.
3. **Hold** (`hold_duration_s`, default 2.0s): reference pose then freezes completely (zero
   reference velocity). Every existing tracking-reward/termination term reads the reference via
   the same motion buffer, so this segment automatically rewards "return to and hold a stable
   stance" with zero new reward terms.

Implemented in `MotionCommand` (`managers/command/terms/wbt.py`) by appending the recovery+hold
segments directly onto the motion buffer at setup time, wired via `configs/stabilization.yaml`.

---

## Shooting rewards & observations (RoboNaldo-style)

As of [v7](#v7--robonaldo-style-shooting-rewards-and-ball-and-target-observations), the kick task can be
shaped for **contact + goal-directed shot placement**, not just motion tracking — adapted from
"RoboNaldo: Accurate, Stable and Powerful Humanoid Soccer Shooting via Motion-Guided Curriculum
Reinforcement Learning" (arXiv:2606.11092)'s Stage-2 reward set.

**Observations** (`managers/observation/terms/unified.py`, tagged `task_mode="kick"`, zeroed
during locomotion episodes like every other `kick_*` term):
- `kick_ball_pos_b` (3-dim) — ball position relative to the robot, in the robot's **heading
  frame** (yaw-only rotation — decoupled from torso pitch/roll during the swing).
- `kick_target_pos_b` (2-dim) — the commanded shot target, same frame.

These are what make `configs/ball.yaml`'s spawn/target randomization *learnable* rather than pure
reward noise: a policy that can't see the ball can only learn the best-average swing over the
spawn distribution, and a constant (unobserved or unrandomized) target is just a bias the network
ignores. With both observed and randomized, the policy adapts its strike per-attempt and — the
practically important part — becomes **commandable at deployment**: feed a different
`kick_target_pos_b` and the trained policy aims there instead of one direction baked in at
training time.

**Rewards** (`managers/reward/terms/shooting.py`, all `task_mode="kick"`, weight scaled by
`configs/ball.yaml`'s `shooting_reward_scale`):

| Term | What it shapes |
|---|---|
| `kick_ball_proximity` | Dense foot→ball approach — nonzero even on a total miss, so there's gradient before contact ever happens |
| `kick_contact_orientation` | Kick-foot velocity aligned with ball→target direction, gated by proximity — the direct aiming term |
| `kick_ball_velocity` | Saturating shot-power term — stops the policy settling for a weak tap |
| `kick_error_ball_to_target` | Outcome accuracy: how close the ball's closest approach got to the target |
| `kick_predicted_error_ball_to_target` | Per-step reward on where the shot is *currently heading* (velocity-ray extrapolation) — the credit-assignment term that gives feedback at the instant of contact instead of waiting for the ball to finish rolling |
| `kick_goal_success_burst` | One-shot bonus (10 steps) the first time an attempt crosses `success_radius` |

Contact is detected from the ball's own displacement/speed off its spawn point (no per-body force
sensor distinguishes ball contact from ground contact), and each of the up-to-3 kick attempts
within a 20s episode (the clip auto-replays after the post-kick hold) is scored independently —
latches reset on every clip restart, including the target itself if
`randomize_target_x`/`randomize_target_y` are nonzero.

`kick_foot` in `configs/ball.yaml` (`"left"`/`"right"`) selects which ankle body the proximity and
orientation rewards are computed against — matches whichever foot the loaded kick clip actually
swings.

---

## Bootstrapping in three stages

Recommended training order (mirrors RoboNaldo's own curriculum): **locomotion → kick motion
tracking → shooting adaptation**, each stage resuming from the previous stage's checkpoint via
`--training.checkpoint`. This works because the observation/action tensor shapes are **fixed
across all three stages** — `kick_ball_pos_b`/`kick_target_pos_b` are always part of the unified
observation config (see above), so `kick_probability` and `shooting_reward_scale` only change
*which reward/command paths are active*, never the network's input/output width. No checkpoint
surgery needed at any transition.

**Conda env for every command below: `hssim`** (`conda activate hssim`, or just
`source scripts/source_isaacsim_setup.sh` from the workspace root — see [Setup](#setup)). Each
stage is typically a separate session run on a different day, so it's easy to forget — if
`train_agent.py` fails immediately with an import error, this is the first thing to check.

### Stage A — locomotion only

`configs/skill_mix.yaml`: `kick_probability: 0.0` (`configs/ball.yaml`'s shooting settings don't
matter yet — no env is ever assigned kick-mode).

```bash
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-unified-fast-sac \
    logger:wandb \
    --training.name unified-stageA-locomotion \
    --training.headless=True
```

### Stage B — add kick motion tracking (RoboNaldo Stage 1: pure imitation, `w_g = 0`)

`configs/skill_mix.yaml`: raise `kick_probability` (e.g. `0.7`).
`configs/ball.yaml`: `shooting_reward_scale: 0.0`, `randomize_x: 0.0`, `randomize_y: 0.0`,
`randomize_target_x: 0.0`, `randomize_target_y: 0.0` — the six shooting reward terms are skipped
entirely (`RewardManager` never evaluates a weight-0 term), and the ball/target observations are
present but constant, so the policy learns a stable kick purely by tracking the clip, same as
before shooting rewards existed.

```bash
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-unified-fast-sac \
    logger:wandb \
    --command.setup_terms.motion_command.params.motion_config.motion_dir=/path/to/your/npz_dir \
    --training.name unified-stageB-kick-tracking \
    --training.checkpoint logs/LocomotionAndBallKicking/<stageA-dir>/model_<iter>.pt \
    --training.headless=True \
    --algo.config.target-entropy-ratio 0.0 \
    --training.num-envs 3300
```

### Stage C — shooting adaptation (RoboNaldo Stage 2, `w_g = 0.8`)

`configs/ball.yaml`: `shooting_reward_scale: 0.8`, `randomize_x: 0.75`, `randomize_y: 0.75`,
`randomize_target_x: 1.0`, `randomize_target_y: 2.0` (or your own ranges). Ball spawn and target
now vary per attempt, both observed by the policy, and the shooting reward terms shape the
already-stable tracked swing for contact and goal-directed placement.

```bash
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-unified-fast-sac \
    logger:wandb \
    --command.setup_terms.motion_command.params.motion_config.motion_dir=/path/to/your/npz_dir \
    --training.name unified-stageC-shooting \
    --training.checkpoint logs/LocomotionAndBallKicking/<stageB-dir>/model_<iter>.pt \
    --training.headless=True
```

Watch `kick_goal_success_burst`'s episode-sum in wandb — dividing by `240 * 10` (its weight ×
`burst_steps`) gives an approximate per-episode success count. If it stays flat for a few thousand
iterations, the spawn/target ranges are likely too wide for the policy to bootstrap contact from —
shrink `randomize_x`/`randomize_y` (e.g. to `0.15`) and ramp up across further resumes, same
distribution-shift caution as any other domain-randomization change (see [the resuming
caveat](#resuming--continuing-training-from-a-checkpoint) above).

**Skipping straight to Stage C from a Stage-A (locomotion-only) checkpoint is not recommended** —
Stage B's pure-tracking phase is what gives the policy a stable, human-like kick to *adapt*; without
it, the shooting rewards (especially `kick_ball_proximity`, which only starts mattering once the
foot is anywhere near the ball) have to discover both "how to kick" and "how to aim" simultaneously,
which is a substantially harder exploration problem — the entire point of a curriculum.

---

## Version history

### v1 — Initial fork

Faithful copy of the `ball_kicking_learning` (kick-only WBT fork) and stock `locomotion_and_motion_tracking/holosoma`
(locomotion) pipelines into this workspace, merged into a new `UnifiedManager` env class: a single
policy driven by a per-env `task_mode` that both reward/observation/termination/command managers
consult (via `task_mode_mask()`) to zero out inactive-task contributions while keeping tensor
widths constant. `terrain_unified_mix` added (40% flat terrain, up from stock's 20%, to host
enough kick-eligible envs). `BallConfig`/ball-spawning ported onto IsaacSim's scene setup.

### v2 — Early correctness fixes

- **Frozen kick training video**: render-gate only checked the primary video recorder; fixed to
  check all active recorders.
- **Camera index / recorder crash bugs**: `_update_camera_position` used an absolute env ID as a
  view-local index (crashed for non-zero recorder IDs); filename collisions between concurrent
  recorders.
- **Robot/ball origin mismatch**: IsaacLab's `scene.env_origins` silently falls back to a uniform
  grid whenever holosoma's own terrain system isn't registered with it (which it never is) —
  kick-mode robots/balls were teleporting tens of meters off their assigned terrain tile. Fixed via
  `UnifiedManager._sync_scene_env_origins_with_terrain()`.
- **Ball reset missing origin offset**: `MotionCommand.reset()`'s ball placement was missing the
  per-env origin offset `root_pos_w`/`object_pos_w` both already applied — every env's ball
  collapsed to one literal world coordinate. Fixed.
- **Kick episodes terminating in 0-5 steps**: a real FK/retargeting-mismatch contact transient
  right after teleporting into an arbitrary clip frame tripped `bad_tracking` before the policy
  took a single action. Added opt-in `grace_period_steps` to `BadTracking`/`BadTrackingZOnly`
  (0 = off by default, no effect on stock experiments; unified sets it to 5).
- **`motion_dir` split the kick sequence into 3 independently-sampled clips**:
  `MultiMotionLoader.extend_with_segments` used to register appended recovery/hold segments as
  separate motions with their own sampling boundaries — most episodes never practiced the actual
  kick, and episodes that did fire never flowed into recovery+hold. Fixed to extend the adjacent
  motion's boundary instead (matching single-file semantics).
- **Random mid-clip kick starts** caused instant failure for an undertrained policy; unified sets
  `start_at_timestep_zero_prob=1.0` so every kick episode starts from frame 0 (matching
  `trigger_kick()`'s deployment behavior).
- **Episode length too short** (10s) for kick+recovery+hold to fit, and the penalty curriculum's
  750-step level-up threshold was unreachable; raised `max_episode_length_s` to 20.
- **Observation-mask noise leak**: task-mode masking was applied *before* additive observation
  noise, letting noise leak nonzero values into supposedly-zeroed inactive-task slots; moved
  masking to after noise/scale/clip.

### v3 — Training-quality fixes

Two significant bugs were found causing the unified policy to train much worse than either
stock reference pipeline (`exp:g1-29dof-fast-sac`, `exp:g1-29dof-ball-kick-fast-sac`):

- **SAC hyperparameters copy-pasted from the kick/WBT template, not locomotion's proven values**:
  `config_values/unified/g1/experiment.py`'s FastSAC block (`target_entropy_ratio`, `tau`,
  `policy_frequency`, `num_updates`, `num_atoms`) matched the WBT/kick experiment's tuning, not
  stock locomotion's. Measured side-by-side at matched iteration counts (`kick_probability=0`,
  everything else identical): the WBT-tuned values never let the policy settle under a
  zero-velocity command (mean base speed 0.56 m/s at 12k iterations, *worsening* with more
  training to ~0.80 m/s by 145k), while locomotion-tuned values reached 0.03 m/s at the same 12k
  iterations — already better than the fully-converged stock baseline (0.079 m/s). Reverted to
  locomotion's values.
- **Kick-training starvation from per-episode task-mode rolling**: kick episodes under an
  undertrained policy survive only ~4-10 steps vs. locomotion's ~1000-step episodes, so a
  per-episode `kick_probability` coin-flip gave kick a fair chance at being *assigned* but not at
  *accumulating training signal* — measured `kick_active_frac` stuck at 0.3-1% even at
  `kick_probability=0.5-0.8`, vs. the ~20-24% expected from `flat_fraction x kick_probability`.
  Fixed by replacing the per-episode roll with a **permanent per-env partition**, decided once at
  env-construction time — kick's aggregate training-time share now matches
  `kick_probability x kick_eligible_frac` regardless of episode-length imbalance.

After both fixes (measured on the same checkpoint lineage after ~22k further iterations): kick
foot-to-ball distance improved from ~0.95-1.2m (never closing, in 145k iterations pre-fix) to
0.44m and closing; kick episode survival went from 4-8 steps to 124-258 steps; locomotion
zero-velocity drift stayed at the good ~0.03 m/s level even with kick now sharing the policy.

### v4 — MuJoCo sim-to-sim deployment + stance-symmetry reward

- **New**: sim-to-sim MuJoCo deployment support (`src/holosoma_inference/`, forked from the
  reference `holosoma_inference` package) — a new `UnifiedPolicy` inference class, a matching
  256-dim observation config, keyboard commands for kick trigger/return-to-locomotion, and a new
  `inference:g1-29dof-unified-loco-kick` preset. See
  [Sim-to-sim deployment](#sim-to-sim-deployment-mujoco-two-terminals) above.
- **Fixed bug**: `UnifiedPolicy`'s gait-phase signal (`sin_phase`/`cos_phase`) was being reset to a
  constant value on *every* tick while walking, instead of only once on the standing→walking
  transition (a one-shot `is_standing`-flag pattern the stock `LocomotionPolicy` already used
  correctly) — this froze the leg-alternation timing cue the whole time the robot walked. Root
  cause of "weird gait / forward command produces sideways drift" reports. Fixed to match
  `LocomotionPolicy`'s exact one-shot-reset semantics.
- **Added**: auto-return-to-locomotion after a kick — previously required remembering to press a
  key; now detects when the kick clip's embedded reference has settled into its final hold pose
  (tick-to-tick unchanged, past a 5s minimum-duration floor to avoid misfiring on any early
  held/wind-up segment in the raw clip) and automatically hands control back to locomotion after a
  further 3s.
- **New reward term**: `penalty_stance_asymmetry` (`managers/reward/terms/locomotion.py`, wired
  into `config_values/unified/g1/reward.py` only — **not** into the shared
  `g1_29dof_loco_fast_sac` preset, so the standalone locomotion baseline experiment is untouched).
  Root cause it addresses: confirmed via direct IsaacSim measurement that the trained policy was
  settling into a stable-but-asymmetric standing equilibrium (e.g. one leg's hip-pitch/knee
  deviating far from default while the other stayed near-neutral, producing a visible front-back
  foot-placement mismatch) — nothing in the existing reward function penalized this specifically
  (the `pose` term's hip-pitch/knee weights are deliberately weak to not fight walking strides;
  `feet_phase` only tracks foot *height*, not forward/backward position). The new term penalizes
  left/right leg mirror-asymmetry, gated by an exponential falloff on commanded velocity magnitude
  (near-full strength at zero command, fading out while actually walking so it doesn't fight
  normal stride asymmetry). Measured effect after 14k iterations resuming from a 163k-iteration
  checkpoint: left-right foot-position offset improved from -0.060m (std 0.105m) to -0.012m
  (std 0.068m); hip-pitch left/right difference from 0.268 rad to 0.056 rad; knee difference from
  0.216 rad to 0.032 rad.
- **Open issue** (not yet resolved): a MuJoCo-specific hip-roll (stance-width) sim-to-sim gap —
  see [Known limitations](#known-limitations-as-of-v6--sim-to-sim-standing-robustness) above.

### v5 — PD-gain/delay domain randomization for MuJoCo robustness

- **Root cause identified**: neither PD-gain randomization nor action/control-delay randomization
  were enabled for the unified experiment — `config_values/unified/g1/randomization.py` reused
  WBT's randomization preset as-is, which has both `enable_pd_gain` and
  `setup_action_delay_buffers.enabled` off (tuned for kick-task stability, not sim-to-sim
  robustness). Stock dedicated locomotion's own preset — which never showed the MuJoCo single-leg
  standing symptom — has both on. The policy had never been trained to tolerate any actuator
  gain/timing mismatch, unlike stock locomotion.
- **Fix**: flipped both on in `config_values/unified/g1/randomization.py`, matching stock
  locomotion's settings (`kp_range`/`kd_range` already `[0.9, 1.1]` in both presets, just
  previously inert; `ctrl_delay_step_range` already `[0, 1]`, likewise previously inert).
- **Resuming from the v4 checkpoint regressed, not improved** — see the exploration-inheritance
  caveat under [Resuming / continuing training](#resuming--continuing-training-from-a-checkpoint)
  above. Killed the resumed run at ~15k iterations in and restarted from scratch
  (`unified-policy-v5-scratch`) once the mechanism was understood.
- **Measured effect** (from-scratch run, checkpoint ~85k, same 8-cycle randomized MuJoCo
  walk→stop test as the known-limitations section): 6/8 cycles now settle with zero post-settle
  contact switches (vs. roughly half before); the 2 residual cycles show switches tightly
  clustered within ~0.6s of the settle mark and fully resolved after, instead of spanning the
  whole post-stop window. `jvel_rms` in the tail dropped to ~0.002 across all 8 cycles (previously
  more mixed). The underlying single-leg-stance bias was not fixed by this change alone (see
  [Known limitations](#known-limitations-as-of-v6--sim-to-sim-standing-robustness)).

### v6 — Zero-velocity heading-drift reward

- **Root cause identified**: direct IsaacSim measurement (walk → hold a zero-velocity command for
  6s → resume walking, same command as before) showed mean ~12.6° / max ~96° heading (yaw) change
  accumulating *during the stand itself* — essentially uncorrelated with left/right leg-symmetry
  error (r=0.03), i.e. a distinct failure mode from v4's `penalty_stance_asymmetry`, not a
  symptom of it. `tracking_ang_vel`'s soft `exp(-error / sigma)` reward saturates near 1.0 for
  small errors, so a slow-but-persistent yaw-rate bias barely registers as a penalty yet
  compounds into a large heading change over several seconds; `penalty_ang_vel_xy` only covers
  roll/pitch (indices `:2`), never yaw.
- **New reward term**: `penalty_yaw_drift` (`managers/reward/terms/locomotion.py`), a hard squared
  base-yaw-angular-velocity penalty, gated the same way as `penalty_stance_asymmetry` (near-full
  strength at zero command, fading out during real turning so it doesn't fight commanded
  rotation). Wired into `config_values/unified/g1/reward.py` only. Weight `-20.0` — yaw-rate² is a
  much smaller-magnitude quantity than the joint-angle-error terms `penalty_stance_asymmetry`
  uses, so it needs a proportionally larger weight for comparable gradient strength; a starting
  point, may need tuning once training results come in.
- **Training**: resumed from `unified-policy-v5-scratch`'s checkpoint (reward-term addition is a
  small, same-dynamics change — the opposite case from v5's randomization change — so resuming is
  expected to work well here, per the caveat above).
- **Status at time of writing**: training in progress (`unified-policy-v6`); results not yet
  measured.

### v7 — RoboNaldo-style shooting rewards and ball-and-target observations

- **Motivation**: a deployed unified checkpoint could approach and kick, but with no notion of
  *where* the shot should go — the kick clip is tracked open-loop, so contact quality and shot
  direction are whatever the motion-tracking reward happened to converge to, not something the
  policy can adapt per-attempt. Adapted the task-reward design from "RoboNaldo: Accurate, Stable
  and Powerful Humanoid Soccer Shooting via Motion-Guided Curriculum Reinforcement Learning"
  (arXiv:2606.11092)'s Stage 2 (ball/target-conditioned shooting adaptation on top of a Stage-1
  motion-tracking prior).
- **New observations**: `kick_ball_pos_b`, `kick_target_pos_b` (`managers/observation/terms/unified.py`,
  robot-heading frame) — actor obs grows 256 → 261 dims. **Not resumable from any pre-v7
  checkpoint** — the input layer width changed, so this requires training from scratch (see
  [Bootstrapping in three stages](#bootstrapping-in-three-stages)).
- **New rewards**: six terms in the new `managers/reward/terms/shooting.py` (foot-ball proximity,
  contact orientation, shot velocity, outcome accuracy, a densified predicted-landing term, and a
  one-shot success burst) — see [Shooting rewards & observations](#shooting-rewards--observations-robonaldo-style)
  above for the full table. Two adaptations from the paper: contact is detected from the ball's
  own displacement/speed (no per-body force sensor distinguishes ball vs. ground contact) rather
  than a force-sensor term, and the goal-line ballistic extrapolation becomes a velocity-ray
  closest-approach prediction (the target is a ground point here, and the ball mostly rolls rather
  than flies). Per-attempt state (each 20s episode replays the kick clip up to ~3x) latches and
  resets independently per attempt, including on target randomization.
- **`configs/ball.yaml` extended**: `randomize_x`/`randomize_y` (per-attempt spawn noise, 0 by
  default — no behavior change for existing configs), `kick_foot` (`"left"`/`"right"`),
  `target_x`/`target_y` + `randomize_target_x`/`randomize_target_y`, `success_radius`, and
  `shooting_reward_scale` (RoboNaldo's stage-wise `w_g`, doubling as the Stage-B/Stage-C on/off
  switch in the new bootstrap protocol). All new keys are optional with backward-compatible
  defaults — a pre-v7 `ball.yaml` (just `radius`/`mass`/`x`/`y`) still loads and behaves exactly
  as before.
- **Verified**: scripted mock-env tests (no simulator) covering reward-term math (proximity/
  orientation peak at contact, predicted-landing reward tracks shot heading, burst pays exactly
  once per attempt, all latches reset on clip replay), heading-frame observation math (yawed robot
  sees a world-east ball correctly rotated into its own frame; torso pitch does *not* leak into the
  reading), per-env randomized-target correctness (each env's reward is scored against its own
  drawn target, not a shared one), and config parsing (new yaml keys, legacy yaml back-compat,
  invalid `kick_foot` rejected). Not yet validated against a live training run.
- **Not yet done**: `holosoma_inference`'s deployment-side `UnifiedPolicy` doesn't build these two
  new obs terms yet — needed before a Stage-C checkpoint can actually be deployed (ball position
  from the existing UDP ball pose source, target from a new commandable input).

### v8 — Kick contact-safety reward (sim2sim/sim2real launch instability fix)

- **Motivation**: a Stage-B checkpoint (`model_0071000`, `unified-stageB-detq-fromscratch`) that
  looked robust in wandb rollouts and in the IsaacSim training env launched itself several meters
  into the air roughly 0.6s into the kick, essentially every time, in MuJoCo sim-to-sim deployment.
  Live-reproduced with real ground truth (MuJoCo's own `qpos`, not the DDS-bridged state, plus
  offscreen-rendered frames): the pelvis-height trace is a smooth, *continuously-accelerating*
  climb over ~1-2s (not a discontinuous velocity jump followed by plain ballistic decay), escalating
  across repeated bounces from ~4m to 14.5m, and recurring once, unprompted, during ordinary
  locomotion. Consistent with the stance/kick foot's ground contact developing a runaway restoring
  force (deepening penetration under a controller that keeps commanding through the correction) —
  something PhysX (IsaacGym/IsaacSim) tolerated during training but MuJoCo's solver didn't. This is
  a sim2sim/sim2real gap in the trained motion itself, not a MuJoCo config issue — deliberately
  **not** fixed by loosening MuJoCo's contact solver, since that would hide the same excess-force
  behavior a real robot's actuators/structure would also have to absorb.
- **New rewards**: two terms in the new `managers/reward/terms/kick_safety.py` —
  `penalty_excess_foot_contact_force` (per-foot ground-reaction force beyond 3x G1's bodyweight,
  generous headroom for legitimate dynamic single-support loading) and `penalty_hard_foot_landing`
  (a foot's own speed at the instant of new ground contact, beyond 2 m/s). Both act on
  `env.feet_indices` (both feet — the stance leg is the leading suspect, but sim2real robustness
  benefits from constraining both) and are wired with `task_mode="kick"` in
  `config_values/unified/g1/reward.py`. Unlike the v7 shooting terms, **not** scaled by
  `configs/ball.yaml`'s `shooting_reward_scale` — contact safety is basic kick stability ("post
  kick stabilization"), not shooting skill, so it's active in Stage B as well as Stage C (confirmed
  via `print_active_kick_rewards.py`).
- **Resumable**: no observation-space or action-space change — unlike v7, this needs no
  from-scratch retrain; continue training any existing unified checkpoint and the new terms take
  effect immediately.
- **Verified**: scripted mock-env tests (no simulator) covering both terms — zero penalty for
  normal double-support and generous (2x bodyweight) single-support loading, strictly positive and
  quadratically-scaling penalty for force beyond the threshold, per-env independence, correct
  new-contact edge detection (including the reset-boundary guard) and zero penalty for sustained
  (non-new) contact or gentle touchdowns. Not yet validated against a live training run or a
  re-tested MuJoCo sim2sim deployment — that requires retraining first.

### v9 — Stage-B ball-observation gating (**the actual root cause of the bad v7+ kick**)

- **Symptom**: every v7+ Stage-B checkpoint deployed a broken, "loose" kick in MuJoCo sim2sim,
  while the pre-v7 checkpoint (`20260708_040125-.../model_0118000.onnx`, 256-dim) stayed robust —
  despite Stage B's kick reward being *byte-identical* between them.
- **Root cause**: v7 added `kick_ball_pos_b`/`kick_target_pos_b` (+5 dims, 256 → 261) and fed them
  **real, dynamic values during Stage B** — even though Stage B sets `shooting_reward_scale = 0`,
  so **no reward term reads the ball**. An unrewarded-but-live input is the worst of both worlds:
  the policy has no reason to *use* it, but also no pressure to *ignore* it, so gradient descent
  leaves arbitrary nonzero weights on those dims that co-adapt with everything downstream. In
  training these dims sweep smoothly (|ball| ≈ **1.38 m** at clip start → **0.21 m** at strike);
  at deployment the only available sources supply a **constant** — `FixedBallPoseSource` is a
  body-frame offset that rides *with* the robot (its own docstring: *"NOT a fixed point in the
  world"*), and the `none` default is a hard zero. Neither matches anything the policy saw
  mid-kick. **Measured on `model_0155000`: perturbing only those 5 dims from their training values
  to what deployment actually feeds changed the output action vector by 16–51 % of its own norm
  (up to 0.68 rad on a single joint) — on every tick of the kick.** That is what collapsed the
  dynamic single-support motion. The pre-v7 checkpoint has no ball observation at all and was
  therefore *structurally immune* — the entire explanation for the old-vs-new gap.
- **Fix**: `ball_pos_b`/`target_pos_b` (`managers/observation/terms/unified.py`) now return **zeros
  whenever `configs/ball.yaml`'s `shooting_reward_scale == 0`** — i.e. the observation follows the
  same Stage-B/Stage-C switch that already gates the shooting *reward*. Stage B therefore becomes
  exactly what it is supposed to be — robust kick motion-tracking, immune to any ball-obs
  mismatch, behaviourally equivalent to the old robust 256-dim policy — while the terms stay
  **registered**, so `actor_obs` remains **261-dim** and a Stage-C run can resume from a Stage-B
  checkpoint with no input-layer change. Stage C flips `shooting_reward_scale > 0`, turning the
  real values and the shooting reward on *together*: the policy first sees a meaningful ball
  exactly when it first has a reason to care about one.
- **Deployment consequence**: a **Stage-B checkpoint needs NO ball flags at all** — the default
  `--task.kick-ball-source none` (zeros) now matches training exactly. A **Stage-C** checkpoint
  must be fed a genuinely *dynamic*, heading-frame ball reading (`UdpBallPoseSource` +
  `run_sim.py --broadcast-ball-udp-port`), **never** `--task.kick-ball-source fixed`, which is a
  smoke-test stub, not a world-anchored ball.
- **Also**: `--algo.config.deterministic-loss-weight 1.0` should be dropped (back to its `0.0`
  default) for Stage B. Config-diffing the robust old run against the current one showed the SAC
  hyperparameters are otherwise *identical*, and this term (added while chasing an earlier,
  unrelated `tanh(mu)` export bug) was **absent** from the robust run. With weight 1.0 it roughly
  doubles the Q-maximisation pressure against an unchanged entropy term, over-sharpening the
  policy (observed `policy_entropy` 0.28, `action_std` 0.058) — which is also why checkpoint
  155k deployed *worse* than 131k despite 24k more gradient steps.
- **Resumable**: no shape change (still 261-dim), so Stage B can resume from the Stage-A
  checkpoint as before.
- **Verified**: mock-env tests — zeros under `shooting_reward_scale = 0` with shapes unchanged;
  real values under `> 0`; shape-invariance across the B→C switch; and the Stage-C value
  cross-checked against an independent hand-computation of `ball_pos_b` from the raw motion-clip
  NPZ (both give |ball| = 1.38 m at clip start).

---

## Troubleshooting

**Headless rendering errors** (`GLXBadFBConfig`, `eglInitialize failed`): disable video with
`--logger.video.enabled False`, or force EGL with `DISPLAY= python ...`, or use a virtual display
with `xvfb-run -a python ...`.

**Video recording**: enabled by default with `logger:wandb`. Adjust with
`--logger.video.enabled False`, `--logger.video.interval <episodes>`, or
`--logger.video.width/height`.

**Conda env cloning gotcha**: if you ever raw-copy (`cp -a`) a conda env to isolate a risky change
(rather than modifying a shared env in place), the clone's own `bin/pip` script has a hardcoded
shebang pointing at the *original* env's Python — invoking it will silently modify the wrong
environment. Always use the clone's own interpreter directly instead:
`<clone_path>/bin/python3.X -m pip install ...`, never `<clone_path>/bin/pip ...`.
