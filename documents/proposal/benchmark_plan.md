# Benchmark Plan — ICRA Submission

Everything that needs measuring, in one place: tables, figures, ablations, protocol.

Written 2026-09-05. Companion to `main.tex`. Structural values (bearings, distances, tolerances)
in this document are **real**, computed from `configs/skill/*.yaml`. Every result cell is
**unmeasured** unless explicitly marked otherwise.

---

## 0. Status — what exists today

### Already instrumented (reusable)

| Metric | Where | Notes |
|---|---|---|
| `kick_ball_success_rate` | `unified_manager.py` | `min_target_dist ≤ success_radius`, latched per attempt — matches RoboNaldo's shot-error definition |
| `kick_ball_hit_rate` | same | contact rate |
| `kick_ball_velocity` | same | post-contact ball speed |
| `kick_alive_frac` | same | ⚠️ conflates `bad_tracking` (can fire while standing) with real falls |
| `kick_topple_frac` | same | **strict fall metric — use this one for the paper** |
| `kick_min_base_height` | same | continuous stability proxy |
| `kick_early_term_frac`, `kick_episode_length` | same | |
| Per-skill split | `Kick_skills_{i}::` sections | all of the above, per skill |
| `kick_handoff_*` | same | alive/topple/episode-length for end-of-clip handoff |
| `sim2sim/kick_to_loco_random_flip_alive_rate` | `mujoco_kick_loco_flip_scan.py` | random mid-clip flip, MuJoCo, strike/nonstrike split, proper pre-flip-failure exclusion |
| `SUMMARY` (fall_rate), `SUMMARY_HIT` (hit_rate) | `mujoco_kick_survival_scan.py` | RoboJuDo MuJoCo, **deployed ONNX action**, per-trial `min_target_dist` logged |
| `SUMMARY_DIRECTION` (direction_success_rate) | same, `--kick-aim-enabled` | per-trial `kick_aim_theta` sampled uniformly within ±θmax — **this is the real aim-angle sweep**, run through the actual deployed policy on real MuJoCo, not just IsaacSim training telemetry |

⚠️ **`direction_success_rate` is NOT the same statistic as Table 2's success@0.5m** — it uses
`direction_success_sigma_m` (default **1.0 m**, looser) and is conditioned on `num_hit`, not
`num_trials` (whiffs are deliberately excluded — "a whiff has no departure direction to grade").
**No new experiment needed to fix this**: the script already logs `min_target_dist` per trial
regardless of `--kick-aim-enabled`, so both the loose direction-success number *and* a strict
`success@0.5m / num_trials` (matching RoboNaldo's actual definition) are computable from one run's
raw output by re-thresholding — this is a post-processing difference, not a missing measurement.

⚠️ **This does not yet give the F2 polar coverage figure.** `SUMMARY_DIRECTION` aggregates the
*entire* ±θmax sweep into one number — it doesn't bin by which θ each trial actually sampled. The
script already draws and uses `kick_aim_theta` internally; it just isn't in the printed `RESULT`
line yet. Adding it there (one field) is what turns this into angle-resolved success — small
addition, not a new pipeline.

### Existing scripts

- `mujoco_kick_survival_scan.py` + `record_mujoco_survival_scan.py` — **the real aim-angle sweep**,
  fall/hit/direction-success rates, deployed ONNX, per-trial `min_target_dist`
- `mujoco_kick_loco_flip_scan.py` + `record_mujoco_kick_to_loco_flip_scan.py` — interruptibility
- `calibrate_nominal_bearing.py`, `validate_checkpoint_real_rollout.py`
- `distill_specialists.py`, `seed_critic_from_teacher.py`

### The gap (narrower than first assessed)

The per-skill MuJoCo aim-angle sweep, fall rate, hit rate, and direction success **already exist**
and already run through the deployed policy on real MuJoCo — Table 2's "Ours" rows and much of
Table 3 are a post-processing/orchestration job over `mujoco_kick_survival_scan.py`'s output, not a
new eval harness. What's actually missing:

1. **Per-trial `kick_aim_theta` in the `RESULT` line** (small script change) — needed for F2's
   angle-resolved polar plot; currently only the aggregate-over-the-whole-sweep number exists.
2. **An orchestrator across all 6 skills** running this scan per skill (each with its own θmax,
   distance, nominal bearing from §1) and collecting output into one JSON.
3. **A strict-threshold re-derivation** (0.5 m / num_trials) alongside the existing loose
   direction-success number, from the same collected `min_target_dist` values.
4. **Nothing yet for Table 5** (locomotion generality with the library attached) or **Table 4a**
   (negative transfer, specialist vs. unified) — these still need building from scratch.

Proposed: `scripts/60_eval_shooting.py` (thin orchestrator over `mujoco_kick_survival_scan.py`,
items 1–3 above), `61_eval_locomotion.py` (Table 5, net-new), `62_eval_negative_transfer.py`
(Table 4a, net-new), each emitting JSON consumed by a single `70_make_tables.py`.

---

## 1. Skill inventory (real values — the basis for every coverage claim)

| Skill | Kick foot | Distance | Nominal bearing | Reachable band (θmax ±15°) | Ang. tol @0.5m |
|---|---|---|---|---|---|
| skill_011 | right | 5.00 m | +14.08° | [−0.9°, +29.1°] | ±5.72° |
| skill_012 | right | 5.17 m | −1.32° | [−16.3°, +13.7°] | ±5.53° |
| skill_013 | right | 5.30 m | −29.93° | [−44.9°, −14.9°] | ±5.39° |
| skill_014 | right | 5.00 m | +75.39° | [+60.4°, +90.4°] | ±5.71° |
| skill_015 | right | 3.00 m | −90.00° | [−105.0°, −75.0°] | ±9.46° |
| skill_016 | right | 1.34 m | +143.82° | [+128.8°, +158.8°] | ±20.5° |

**Nominal-bearing span: 234°** (−90.00° to +143.82°, skill_015 to skill_016) — the spread between
the two extreme skills' *center* aim directions. This is NOT a coverage claim; see below.

**Actually-reachable union (θmax = ±15° per skill): 164.0°**, across 4 disjoint islands, out of a
263.8° edge-to-edge extent — 99.8° of that extent is gap, not coverage.

⚠️ **Three uncovered gaps at θmax = ±15°** (not two — the earlier count predates skill_016):
skill_015↔013 (30.1°), skill_011↔014 (31.3°), and **skill_014↔016 (38.4°, the largest)**.
Continuous tiling would need θmax ≈ ±34° (set by the largest gap, not ±31° as previously stated).
Either widen θmax (blocked on the aim-conflict question, §4 A8), add clips nearer the gap centers
(≈−60°, ≈+45°, ≈+109°), or report the gaps honestly. **Do not describe coverage as "continuous,"
and do not quote the 234° nominal span as if it were the covered figure — use 164.0° for that.**

---

## 2. Tables

### Table 1 — Capability comparison *(exists in main.tex)*

Qualitative ✓/○/✗ across prior systems. Two corrections pending:

- **Haarnoja `Free cmd.` ✓ → ✗ or ○.** Verified: their policy receives game state (ball/opponent/goal
  positions), **no velocity commands at all**. By your own caption's definition it fails this column.
- **Haarnoja `Multi skill` ✓ → ✗.** Your caption defines it as "more than one distinct strike/contact
  type." They have walk/turn/kick/get-up — multiple *behaviors*, exactly **one** strike type.
  Add a footnote acknowledging the behavior repertoire so the downgrade reads as precision.

### Table 2 — Quantitative context (cited prior work + ours)

| Method | Bearing range | Dist. (m) | Ang. tol. | suc.@0.5m (%) | Shot err. (m) | v_ball max (m/s) | Alive (%) |
|---|---|---|---|---|---|---|---|
| PPO | forward | n/r | n/r | 0.0 | 4.721 | 0.780 | 0.0 |
| AMP | forward | n/r | n/r | 0.9 | 3.733 | 1.771 | 41.6 |
| PAiD | forward | n/r | n/r | 8.2 | 1.850 | 4.986 | 67.3 |
| RoboNaldo (free-kick) | forward, ±38.7° azimuth | 5.0 | n/r | 28.8 | 0.899 | 14.792 | 100.0 |
| RoboNaldo (moving) | forward, ±38.7° azimuth | 5.0 | n/r | 32.4 | 1.131 | 13.875 | 98.8 |
| **Ours (single skill)** | ≈forward, ±15° (skill_012) | 5.17 | ±5.5° | — | — | — | — |
| **Ours (library, N=6)** | **164.0° reachable** (234° nominal span, 3 gaps) | 1.3–5.3 | ±5.4°–±20.5° | — | — | — | — |

**Verified**: RoboNaldo's simulation targets are sampled on an *8 m × 2 m goal plane 5 m ahead*
— **but only the 8 m width is a bearing (azimuth) spread**, giving ±38.7° = atan(4/5), entirely
within the forward hemisphere (never sideways or behind). **The 2 m is vertical placement within
the goal (high/low shots), a different axis than bearing entirely** — do not read "8×2 m" as 2D
directional coverage; RoboNaldo is forward-only, exactly as Table 1's `Lateral tgt.` column already
scores it (○, "degradation at extreme lateral and high targets," never tested outside one
forward-facing zone). Their **hardware** numbers (0.73 m free-kick / 0.86 m moving) are at **3 m**
— a different protocol. Keep hardware and simulation in separate rows; never mix the two distances.

⚠️ Even at its most generous reading, RoboNaldo's angular coverage is one 77.3° arc dead ahead.
Ours spans 234° across six separately-aimed directions, including −90° and +144° — outside
anything a single forward-facing goal plane could ever sample. These are not the same *kind* of
quantity (shot placement within one goal vs. which direction the robot can strike at all); don't
let the table imply a magnitude comparison between them.

**Two rows for yourself, not one.** The single-skill row proves the substrate is competitive
(defuses "maybe their kick is just bad"); the library row shows what the architecture buys.
Report `v_ball` only for shooting skills — for skill_015/016 (3.0 m and 1.34 m passes) report
**speed-control error against a target speed** instead; peak speed is the wrong objective there.

Caption convention: *"Numbers as reported by each paper; evaluation geometry differs — see Distance
and Angular tolerance."* Use **n/r** for unreported metrics, never 0.

### Table 3 — Per-skill coverage

Columns: Skill | Kick foot | Distance | Nominal bearing | Reachable band | Ang. tol | **Success (%)** |
**Hit rate (%)** | **Topple (%)**

One row per skill (values from §1), plus a union row. No prior-work rows — nobody reports coverage.

### Table 4 — Ablations

Four sub-tables. See §4 for full specification.

### Table 5 — Locomotion generality with the skill library attached

| Command axis / task | Locomotion-only policy | With skill library (N=6) | Δ |
|---|---|---|---|
| v_x forward MAE (m/s) | — | — | — |
| v_x backward MAE (m/s) | — | — | — |
| v_y lateral MAE (m/s) | — | — | — |
| ω_z yaw MAE (rad/s) | — | — | — |
| Terrain traversal success (%) | — | — | — |
| Push recovery rate (%) | — | — | — |

**This is the paper's headline table.** No prior humanoid soccer system reports this axis at all.
Paired comparison — the claim is *no degradation* from attaching the library.

### Table 6 — Hardware results

| Condition | Trials | Success (%) | Shot err. (m) | Contact (%) | Alive (%) |
|---|---|---|---|---|---|

Report sim→real gap explicitly on the same metric (Haarnoja report 70% sim vs. 58% real; that
honesty reads well). ≥20–50 trials per condition for usable confidence intervals.

---

## 3. Figures

| # | Figure | What it shows | Defends |
|---|---|---|---|
| **F1** | System/architecture diagram | Task-mode gating, locomotion hub, N skills, handoff | Novelty 1, 2 |
| **F2** | **Polar coverage plot** | Angular axis = target bearing, radial = success rate. Prior work = one forward lobe. Ours = six lobes spanning 234° | **The money figure.** Novelty 4 |
| **F3** | Locomotion command-tracking heatmap | Tracking error over (v_x, v_y) grid + ω_z sweep, with and without skills attached | Novelty 1, 5 |
| **F4** | Gait-space envelope | Reachable sustained-velocity polytope, ours vs. tracking-substrate ablation | Novelty 1 |
| **F5** | Interruptibility sweep | Alive rate vs. flip tick k across the swing (pre-strike / strike / post-strike) | Novelty 3 — **strongest single result** |
| **F6** | Phase-resolved failure decomposition | Fraction of failures in approach / strike / post-skill | Novelty 3 |
| **F7** | Skill-count scaling | Union coverage and mean per-skill success vs. N = 1,2,3,4,6 | Novelty 2 |
| **F8** | Hardware filmstrip | Real-robot frames: commanded navigation interleaved with multi-skill strikes | ICRA expects this |

F2 and F5 are the two figures that carry the paper. Build them first.

---

## 4. Ablations

### A1 — Substrate inversion *(needs building)*

| Arm | Description |
|---|---|
| Task-gated locomotion hub (ours) | Kick reward hard-zeroed in LOCO mode |
| Tracking-reward-always-active | Tracking ungated through approach, no separate locomotion mode |

Metrics: shooting **and** locomotion (both matter). Prediction: comparable shooting, collapsed
lateral/yaw command envelope under always-active tracking.

⚠️ **This arm does not exist yet.** The `-robonaldo` configs in `configs/artifacts/` are a
**reward-term** variant (they zero `kick_foot_strike_pitch`, `kick_ball_contact_hit`,
`kick_ball_approach_stance`, the strike-divergence penalty, and enable body-push DR) — useful for
"which of our reward terms earn their keep," but **not** an architectural ablation. Building the
real one means ungating the tracking reward and removing the locomotion mode, holding clips,
retargeting, simulator and reward implementation fixed.

### A2 — Negative transfer *(the central multi-skill result)*

Per-skill success in the unified N=6 policy vs. that skill's own specialist checkpoint
(`stageC1-skill011…016` already exist as references). Small Δ across all skills = interference-free
coexistence. **Promote this from side-ablation to a main table** — it is the evidence for the
multi-skill claim.

### A3 — Skill-count scaling

N = 1, 2, 3, 4, 6. Report union bearing span and mean/min per-skill success.
Configs `3skills.yaml` / `4skills.yaml` already set this up.

### A4 — Handoff (kick → loco)

| Arm | Post-skill alive (%) | Topple (%) | Time-to-command-recovery (steps) |
|---|---|---|---|
| Learned locomotion handoff (ours) | | | |
| Scripted stabilization tail | | | |
| No stabilization | | | |

Aggregate across all six skills — generality across the library *is* the claim.
Reference point: RoboNaldo's own ablation drops 98.8% → 24.4% alive without their scripted tail.

### A5 — Interruptibility (mid-swing abort) — **partially measured**

Sweep flip tick k across the swing; report alive rate per phase bin.
`mujoco_kick_loco_flip_scan.py` already does this, with a strike/nonstrike split.

**Current status**: skill015 at 100% (both windows) and skill011-nonstrike at 100% are confirmed in
logged data. skill011 strike-window at 100% is **user-reported on the correct locoflip-trained
checkpoint** — the Sep 2 wandb runs showing 0.2–0.8 came from checkpoints *without* locoflip
training and must not be cited. **Action: tag the correct run/checkpoint ID so this number is
traceable.**

### A6 — Skill-conditioning interface

One-hot vs. learned embedding vs. no conditioning. Promised explicitly in the proposal's own
timeline (Aug 26 – Sep 8 window).

### A7 — Motion guidance

With vs. without the motion-tracking reward (pure task reward). Separates the method from Haarnoja,
who use no motion reference at all.

### A8 — Aim conflict

`kick_aim_theta_max_deg` pinned vs. free. Tests whether θ ≠ 0 creates a shooting-vs-tracking
conflict (the reference target moves but the clip does not rotate). **Blocks any θmax widening**,
and therefore blocks closing the coverage gaps in §1. Publishable either way.

### A9 — Joint training vs. distillation

Task-gated joint training vs. `distill_specialists`. **The paper must state which produced the
headline result.** Both paths exist in the tree right now.

### A10 — Body-push domain randomization

On/off → push recovery and sim2real. Newest mechanism, currently unvalidated.

### A11 — Negative result worth reporting

The perception-noise A/B came back null, within the ±27% replicate noise floor. Pre-registered
arithmetic that predicted a null, confirmed by experiment, is a credibility asset — include it.

---

## 5. Measurement protocol

- **≥3 seeds, ideally 5**, per configuration. Report mean ± std or 95% CI. Single-run deltas are
  not results.
- **Publish the ±27% replicate noise floor** and refuse to claim any effect below it. This is the
  single most credibility-building thing available, and the number is already known.
- State evaluation episode count per data point (≥1000 sim episodes).
- Use `kick_topple_frac`, **not** `kick_alive_frac`, wherever "did it fall" is the question.
- Pair seeds across ablation arms where possible.
- MuJoCo sim2sim numbers: medians over 6–8 checkpoints, never a single reading — this project has
  documented history of single-checkpoint MuJoCo numbers being pure lottery.

---

## 6. Priority order

**P0 — nothing else matters until these exist**

1. **Add per-trial `kick_aim_theta` to `mujoco_kick_survival_scan.py`'s `RESULT` line**, then write
   the thin orchestrator (`60_eval_shooting.py`) running it per skill. This unlocks Table 2, Table 3,
   and F2 nearly for free — the sweep, the deployed policy, and the distance measurement already
   exist; only the angle-binning and cross-skill aggregation are missing.
2. **A2 negative transfer** — if the six skills don't specialize, the multi-skill claim collapses.
   Check this before further training investment. Needs a net-new script (`62_eval_negative_transfer.py`).
3. **Tag the locoflip checkpoint** for the A5 100% number.
4. **Table 5 (locomotion generality) harness** — genuinely net-new, nothing reusable exists for this
   one. This is the headline table; don't leave it for last just because it's the hardest to start.

**P1 — carries the contribution**

5. Table 5 / F3 locomotion generality with library attached
6. A4 handoff + F5 interruptibility sweep + F6 phase-resolved failures
7. A9 — decide and document which training path is the paper's method

**P2 — completes the story**

8. A1 substrate ablation (needs building)
9. A3 skill scaling, A6 conditioning interface, A8 aim conflict
10. Hardware: MuJoCo sim2sim gate → indoor calibrated trials → outdoor demo

---

## 7. Explicitly out of scope

- **Do not run a PPO ablation.** Your reward stack, curriculum, entropy targets and discounts are
  tuned for FastSAC; PPO failing on a SAC-tuned config demonstrates nothing, and a reviewer will
  say so. Cite Seo et al. 2025 (arXiv:2512.01996) Figure 5 instead — FastSAC vs. FastTD3 vs. PPO on
  Unitree G1 whole-body tracking, in holosoma, already run by the framework's own authors.
- **Do not reproduce RoboNaldo or PAiD.** Cite their published numbers with evaluation geometry
  stated. Controlled comparison comes from A1, which holds clips/retargeting/simulator fixed.
- **Do not make impossibility claims.** Not "substrate inversion is necessary," not "PPO cannot
  build this," not "unified shared network is novel" (RoboNaldo and Haarnoja are both
  single-network). The defensible form is always *we did this, they didn't* — that needs only your
  own results. Claiming another approach *cannot* requires running their experiment properly and
  having it fail, which you have not done and cannot afford to do.

---

## 8. Claim scoping — known limits to state honestly

- **Handoff is not symmetric.** Kick→Loco is robust to a sudden flip at any tick (100%).
  Loco→Kick currently requires decelerating to v≈0 before entry; a sudden in-motion flip drifts and
  misses the ball. Scope the "initiates from the same commandable locomotion state" claim
  accordingly, and state in-motion kick entry as future work.
- **Coverage has gaps**, 30.1° and 31.3° wide, at the current θmax. Don't say "continuous."
- **Skill selection is not learned** — it's a permanent per-environment assignment. The proposal
  criticizes RoboNaldo for handcrafted triggering; scope this explicitly as out of scope, or add a
  simple learned selector.
- **The architecture is algorithm-agnostic.** State it: *"we use FastSAC for its wall-clock and
  exploration properties; nothing in the task-gating, locomotion-hub, or handoff design depends on
  off-policy learning."* This makes the contribution more portable, which reviewers reward.
- **Stage B→C reshaping is shared methodology with RoboNaldo**, not novelty. The novelty is Stage A
  being *first* and Stage D's learned handoff.

---

## 9. Open question that affects everything

`main.tex` states ICRA 2027 submission ≈ **September 2026**, but its own work plan runs through
**Dec 01, 2026**. These are inconsistent. If the real deadline is this month, this plan needs
aggressive triage down to P0 plus Table 5. Resolve the target date before committing to the full
scope above.
