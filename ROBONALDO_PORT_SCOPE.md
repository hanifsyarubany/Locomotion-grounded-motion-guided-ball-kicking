# RoboNaldo port scope — fork plan

## IMPLEMENTATION STATUS (2026-08-05)

This fork (`playground/unified_ball_kick_robonaldo/`, copied from `unified_ball_kicking_skills/`)
implements Phases 1–5 below. Every mechanism ships **registered at weight=0.0 / False / 0.0 (a
verified true no-op)**, staged commented-out in `configs/kicking_motion_reward_tuning.yaml` or
`config_values/unified/g1/experiment.py` with reasoned-but-**unvalidated** starting values —
turning any of them on for a real training run is a deliberate, separate decision, not made here.
All code-level correctness was checked with unit tests (SimpleNamespace fakes) AND a live probe
against a real IsaacSim env + a real trained checkpoint (256 envs, cross-project checkpoint load —
this fork excludes `logs/`, obs/action dims are unchanged by every mechanism below so warm-start
compatibility holds).

- **Phase 1 (reward budget rebalance)** — `kick_alive` phase-shaped via new
  `kick_alive_pre_kick_ratio` (0.2 during approach+strike / 1.0 post-kick, matching RoboNaldo's own
  `robot_alive`'s 10x post-kick step — not flat as originally assumed); 3 previously-zeroed
  shooting terms re-enabled (`kick_contact_orientation`, `kick_ball_velocity`,
  `kick_error_ball_to_target`); `action_rate_l2` -1.0 → -0.04 (RoboNaldo's own ratio). **Landed on
  the ORIGINAL `unified_ball_kicking_skills` codebase too** (config-only, no fork needed), then
  synced into this fork.
- **Phase 2 (per-term tracking relaxation)** — new `root_tracking_reward_scale` (per-skill),
  applied ONLY to the 2 root/anchor tracking terms, separate from `motion_tracking_reward_scale`
  (which still governs the other 4). Also landed on the original codebase first, then synced.
- **Phase 3 (P0 regularization block, fork-only)** — 4 new terms in `wbt.py`: `ArmDefaultPose`,
  `KickFeetAirTime`, `KickSwingFeetClearance`, `kick_no_fly`. **Correction found while
  implementing**: `arm_default_pose_penalty`'s real RoboNaldo formula is `-mean(error²)` (elbow
  joints 5× weighted) — NOT an exp-kernel reward as this doc originally (incorrectly) described in
  §3b below; the docstring in `wbt.py:ArmDefaultPose` documents the correction.
- **Phase 4 (fork-only)** — `rsi_scope_to_authored_clip` + `critical_frame_oversampling_prob`
  (RSI, `MotionCommand.reset()`, extracted as 2 pure testable functions:
  `rsi_span_end_idx`/`critical_frame_oversample_time_steps`); `per_joint_action_clip`
  (`joint_control.py`). **Important correction from naive porting**: RoboNaldo's raw per-joint clip
  *values* don't transfer verbatim — they're calibrated to RoboNaldo's own per-joint action scale.
  Verified by computation that this project's G1 spec matches RoboNaldo's hardware exactly (same
  URDF joint limits), so only `ankle_roll`/`waist_roll`/`waist_pitch` — where the clip is itself
  derived as `URDF limit / action_scale`, a physically-motivated bound — transfer correctly.
  Copying RoboNaldo's loose arm/wrist clip values (±3.0) would NOT meaningfully constrain the
  measured shoulder_yaw/wrist_roll divergence (see `RobotControlConfig.per_joint_action_clip`'s
  docstring for the full arithmetic) — that remains `ArmDefaultPose`'s job.
- **Phase 5 (fork-only)** — `penalty_kick_unstable` (`locomotion.py`), ported from RoboNaldo's
  `unstable_penalty`. **Major scope correction**: RoboNaldo's `stable_anchor_pos_tracking`
  (latched-anchor pull-back — what §5's "GUT the synthetic recovery clip" section below assumed
  was the port target) is gated `weight = 10*reg_weight*float(adapt_motion_flag)` in RoboNaldo's
  own source, and `adapt_motion_flag` is **False** in every stage this project follows (S1/S2a/S2b)
  — it's Stage-3-only, which this project has already decided not to follow. So
  `stable_anchor_pos_tracking` is **not ported**; `penalty_kick_unstable` (base velocity damping,
  ramped via the ALREADY-EXISTING `_kick_recovery_gate` rather than RoboNaldo's own hard-boolean
  snap) is RoboNaldo's real S1/S2 post-kick stabilization pressure, alongside `kick_alive`'s own
  phase-shaping from Phase 1. The "GUT the synthetic recovery clip, replace with latched-anchor"
  idea in §5 below does not apply to the stages actually being ported and was not implemented.

**Update (2026-08-05)**: `reference_relative_targets` was found live-enabled (`True`) in
`experiment.py`, inherited from the "refrel" Stage-B experiment the user had already abandoned
("not worth it, very low performance"). Per explicit follow-up request, the whole mechanism has
since been **removed from this fork entirely** — not just reverted to `False`. Deleted: the
`RobotControlConfig.reference_relative_targets` field, `JointPositionActionTerm._compute_pivot`
(reverted `_compute_torques`'s `P` branch to pivot on `env.default_dof_pos` directly, the
pre-existing behavior), the `FastSACAgent.load()` checkpoint-consistency guard and
`HOLOSOMA_ALLOW_REFERENCE_RELATIVE_MISMATCH` env var, the `experiment.py` opt-in, and the 2 test
files dedicated to this feature (`test_joint_control_reference_relative.py`,
`test_checkpoint_reference_relative_guard.py`). Full test suite reconfirmed green after removal.

- **Phase 6 (`motion_global_feet_lin_vel`, 2026-08-05, fork-only)** — the 7th motion-tracking
  term, previously the one genuinely MISSING tracking term (§3a below). Verified against the real
  RoboNaldo source: `motion_global_feet_linear_velocity_error_exp`'s body is LITERALLY IDENTICAL
  to `motion_global_body_linear_velocity_error_exp` (this project's existing `motion_global_body_
  lin_vel`) — same formula, just called with a feet-only `body_names` and its own weight/sigma.
  Registered in `config_values/wbt/g1/reward.py` (the SHARED WBT base, so it also applies to
  standalone WBT training, same as its 6 siblings) and wired into `_MOTION_TRACKING_SCALED_FUNC`
  for the kick-mode `*_scaled` wrapper treatment, identically to the 4 non-root siblings (NOT
  `root_tracking_reward_scale` — that's root/anchor-only). **Unlike every other addition this
  session, this ships LIVE (weight=1.0, not a staged no-op)**: it completes an already-active
  category rather than introducing new behavior, doesn't touch obs/action dims, and reuses an
  already-battle-tested formula — a materially lower-risk change than the regularization/RSI/
  action-clip additions above. Live-verified against a real checkpoint (loads fine, finite
  [0,1]-bounded signal throughout a real rollout, appears correctly in the RewardManager's episode
  summary alongside its 6 siblings).

- **Phase 7 (§3b/3c completion, 2026-08-05, fork-only)** — the remaining 12 RoboNaldo
  regularization/shooting terms (`kick_feet_contact_time`, `kick_penalty_lin_vel_z`,
  `kick_penalty_dof_vel`, `kick_penalty_torque`, `kick_undesired_contacts` (reuses the existing
  `UndesiredContacts` class, new kick-mode registration only), `kick_penalty_ee_body_pos_
  divergence`, `kick_action_smoothness`, `kick_ball_over_line`, `kick_robot_com_ball_distance`,
  `kick_robot_torso_ball_distance`, `kick_penalize_weak_foot_contact`,
  `kick_penalize_self_contact_feet`). §3b and §3c are now **fully accounted for**: every RoboNaldo
  term in both tables is either implemented, an existing pre-port equivalent, or explicitly
  reasoned as correctly skipped (Stage-3-only / `stand_still`-only mechanisms with weight 0 in
  every stage this project follows) — no remaining "—" gaps. All 12 ship at weight=0.0 (staged,
  reasoned-but-unvalidated), same discipline as every other addition; live-verified together in
  one combined 400-step rollout (all 16 of this session's new terms firing simultaneously) against
  a real checkpoint — finite, correctly signed, non-degenerate throughout. `kick_action_smoothness`
  was found missing only while replacing this doc's own "—" markers (not caught by the original
  §3b/3c audit) — a reminder that this port doc is not itself a substitute for re-checking the
  actual RoboNaldo source term-by-term.
- **Phase 8 (§4 P1 completion, 2026-08-05, fork-only)** — cloned RoboNaldo's actual source
  (`github.com/OpenDriveLab/RoboNaldo`, not just the paper) to resolve §4's last 2 P1 rows.
  Finding: `bad_tracking`'s existing `bad_motion_body_pos` sub-check was ALREADY byte-similar to
  RoboNaldo's `ee_body_pos` termination (same Z-only class, same 4 bodies, same Stage-1 value
  0.25) — only the progressive-widening knob was missing. Added
  `MultiSkillConfig.bad_motion_body_pos_threshold: float = 0.25` (opt-in via configs/\*.yaml,
  no-op at default) so it can be staged to RoboNaldo's 0.35/0.5 across a curriculum resume, same
  as `root_tracking_reward_scale`. Also corrected Phase 7: `kick_penalty_ee_body_pos_divergence`
  had dropped RoboNaldo's `is_warmup`-gated second branch, believing no equivalent existed —
  re-reading `mdp/commands.py` found `is_warmup` reduces exactly to "fewer than `warmup_steps`
  steps since reset," i.e. `env.episode_length_buf < warmup_steps`, already used elsewhere in this
  codebase (`bad_tracking`'s `grace_period_steps`). Ported the branch; synced the term's
  `threshold` param with `bad_motion_body_pos_threshold` under one source of truth, mirroring
  RoboNaldo's `task_overrides.py` (a single yaml override sets both their termination and this
  reward term together). Also confirmed via their actual yaml (`adapt_motion_flag: false` in
  `tracking_params.yaml`/`task_params_1.yaml`/`task_params_2.yaml`, `true` only in
  `task_params_3.yaml`) that §5's "swap post-kick stabilization" item has nothing left to do — the
  latched-anchor mechanism really is Stage-3-only, and `penalty_kick_unstable` (Phase 5) really is
  RoboNaldo's REAL S1/S2 mechanism, formula-verified line-for-line against `mdp/rewards.py`. 10 new
  tests across 2 files (full suite 464 -> 474 passed, same 1 pre-existing unrelated `isaacgym`-
  import failure); live-verified against a real checkpoint (256 envs, 400 steps), including
  exercising the warmup branch on real post-reset `episode_length_buf` state, not just a
  SimpleNamespace fake.

---

Source of truth: `github.com/OpenDriveLab/RoboNaldo` @ shallow clone, paper arXiv:2606.11092.
All weights below are **computed**, not read off the page: RoboNaldo's yaml `weight_scale` does
**not** multiply the Python base weight — it *replaces* it, as `weight = weight_scale *
category_weight` (`tasks/tracking/task_overrides.py:91-100`). Effective weights were evaluated by
applying that rule to `tracking_env_cfg.py`'s base expressions under each stage yaml's scalars.

---

## 0. The curriculum is not what we assumed

We modelled RoboNaldo as "Stage 1 = motion tracking, Stage 2 = add shooting." The `right_kick`
yaml set is actually **four** configs, and the stage-2 sub-curriculum is where the work happens.

| | `tracking_params` | `task_params_1` | `task_params_2` | `task_params_3` |
|---|---|---|---|---|
| paper stage | 1 | 2a | 2b | 3 |
| `stage` | `tracking` | `task` | `task` | `task` |
| `motion_weight` | 1.0 | 1.0 | 1.0 | 1.0 |
| `goal_weight` | **0.0** | 0.8 | 0.8 | 1.0 |
| `reg_weight` | **0.05** | **0.20** | 0.20 | 0.25 |
| `adapt_motion_flag` | false | false | false | **true** |
| `jump_flag` | false | false | false | true |
| ball spawn ±XY | 0.1 | 0.1 | **0.5** | **1.0** |
| `start_time_sampling_fraction` | **1.0** | 0.05 | 0.0 | 0.0 |
| `critical_frame_adaptive_sampling` | **true** (win 10) | false | false | true (win 10) |
| `ee_body_pos` term. threshold | 0.25 | 0.35 | 0.5 | 0.6 |
| `std_difficulty_multiplier` | 1.0 | 1.0 | 1.0 | 1.5 |

Three mechanisms carry the curriculum, and none of them is "add more reward terms":

1. **`goal_weight: 0.0 → 0.8`.** In stage 1 *every* ball reward is exactly zero. Pure imitation.
2. **`reg_weight: 0.05 → 0.20` — regularization is quadrupled at the same moment the ball task
   turns on.** They scale stabilization *up* in lockstep with the destabilizing task reward.
3. **`start_time_sampling_fraction: 1.0 → 0.05 → 0.0`.** Stage 1 is reference-state
   initialization — start anywhere in the clip, with `critical_frame_adaptive_sampling`
   oversampling a ±10-frame window around the kick frame. Later stages always start at frame 0.

---

## 1. The two findings that should drive the fork

### 1a. Only the *root anchor* is ever relaxed. The other six tracking terms never move.

| tracking term | S1 | S2a | S2b | S3 |
|---|---|---|---|---|
| `motion_global_anchor_pos` | 0.5 | 1.0 | **0.1** | **0.1** |
| `motion_global_anchor_ori` | 0.5 | 0.5 | **0.1** | **0.1** |
| `motion_body_pos` | 1.0 | 1.0 | 1.0 | 1.0 |
| `motion_body_ori` | 1.0 | 1.0 | 1.0 | 1.0 |
| `motion_body_lin_vel` | 1.0 | 1.0 | 1.0 | 1.0 |
| `motion_body_ang_vel` | 1.0 | 1.0 | 1.0 | 1.0 |
| `motion_feet_lin_vel` | 1.0 | 1.0 | 1.0 | 1.0 |

That 10× drop on the world-frame root position **is** their entire "ball shooting adaptation."
Free the root to go where the ball is; hold every *relative* body pose pinned at full strength,
in every stage, forever.

Ours does the opposite. `motion_tracking_reward_scale` is a per-skill multiplier on the **whole
category**, and `recovery_tracking_scale: 0.2` relaxes **all six terms** together. When we relax,
we relax the body-pose prior along with the root — which is precisely the prior that stops the
strike from diverging. This is a direct, mechanical explanation for the measured strike-phase
divergence, and it costs one refactor, not 40 new terms.

### 1b. Their alive reward is phase-shaped. Ours is flat, and ~50× too large during the swing.

`mdp/rewards.py:531-537` — `robot_alive` is **not** constant:

```python
post_mask = command.time_steps > (command.motion.critic_frame_index + 50)
rew = torch.ones(env.num_envs, device=env.device)
rew[post_mask] = 10.0
```

So at `reg_weight = 0.2` the effective per-step survival reward is **0.2 during approach + strike**
and **2.0 during post-kick stabilization** — a deliberate 10× step at the same boundary that gates
their stabilization mechanism. They pay almost nothing for surviving the approach and a lot for
surviving the aftermath: don't reward hesitation before the kick, reward the recovery after it.

Ours is flat `kick_alive: 10.0` through every phase — the same payout for not committing to the
swing as for a successful stabilization. That is backwards, not merely mis-scaled.

Both systems total **6.5** across motion tracking, so that baseline is directly comparable:

| | RoboNaldo S2a | ours (live `kicking_motion_reward_tuning.yaml`) |
|---|---|---|
| motion tracking (sum) | 6.5 | 6.5 |
| ball / shooting (sum) | **14.4** | 6.0 |
| alive, approach + strike | **0.2** | **10.0** |
| alive, post-kick recovery | **2.0** | **10.0** |

Our own telemetry already flags `kick_alive` as "by far the largest single contributor to the kick
reward budget." A large constant survival reward during the approach is a direct incentive to
minimize action and never commit to the swing — which matches every symptom on the list. Caveat:
PPO and SAC normalize value differently, so absolute numbers aren't portable. The **ratio within
one budget** is, and 50× during the swing against an identical 6.5 tracking baseline is not a
tuning difference.

**SAC-specific hazard when acting on this.** Our `log_alpha` is auto-tuned against
`target_entropy = -n_act * ratio` (`fast_sac_agent.py:324,363,389`). SAC's entropy temperature is
reward-scale sensitive — the entropy term competes with reward in the Q-target. Porting RoboNaldo's
*absolute* magnitudes (most |w| < 1) would shrink the total budget enough that the entropy bonus
could dominate. Port the **ratios**, then rescale the whole budget to preserve our current total
magnitude. `kick_target_entropy_ratio` (already implemented, per-task-mode alpha group) is the
right knob if kick mode then needs different exploration from locomotion.

Note also our live shooting config is heavily ablated — `kick_contact_orientation`,
`kick_ball_velocity`, `kick_error_ball_to_target`, `kick_predicted_error_ball_to_target`,
`kick_goal_success_burst` are all at 0.0. RoboNaldo's equivalents sum to 14.4.

---

## 2. Fork scope: copy / gut / keep

### KEEP (our differentiators, all verified as genuine)
- Base locomotion policy + lego-like N-skill addition. No RoboNaldo equivalent — this is the thesis.
- Geometric `has_kicked` sensor from the feet. RoboNaldo uses a contact-force sensor
  (`ball_contact_forces`, `force_threshold: 2.0`) — ours is arguably cleaner.
- `LocomotionGait` / `feet_phase` gait-phase shaping for locomotion mode.
- Clip partitioning. **Not a differentiator** — RoboNaldo partitions on the same axis via
  `critic_frame_index` (`commands.py:673,951`). Convergent design; keep ours.
- Post-kick stabilization. **Not a differentiator, and do not delete it** — RoboNaldo has it
  (`stabilize_anchor_pos_w`, `stable_anchor_pos_tracking`, `unstable_penalty`), and its removal is
  their single largest ablation (alive rate 98.8% → 24.4%).

### GUT
- ~~**Our synthetic spliced recovery/hold clip.** Replace with RoboNaldo's reward-side mechanism:
  latch the robot's *own* anchor pose at kick entry, then pull back to it.~~ **CORRECTED
  (2026-08-05, see IMPLEMENTATION STATUS at top):** the latched-anchor mechanism
  (`stable_anchor_pos_tracking`) is gated `weight = 10*reg_weight*float(adapt_motion_flag)` in
  RoboNaldo's own source, and `adapt_motion_flag` is False in S1/S2a/S2b — Stage-3-only. Not
  ported. `penalty_kick_unstable` (ported, see below) is the real S1/S2 mechanism instead.
- **Category-wide tracking relaxation** (`motion_tracking_reward_scale`,
  `recovery_tracking_scale`). Replace with per-term relaxation on the root anchor only (§1a).
- **`kick_alive: 10.0`.** Re-derive from the RoboNaldo ratio (§1b).
- ~~**The 5 per-skill category scales.** Replace with RoboNaldo's two global scalars
  (`reg_weight`, `goal_weight`) plus per-term `weight_scale`. Far smaller tuning surface.~~
  **SUPERSEDED (2026-08-05, see §4):** not ported — collapsing to 2 scalars would silently
  redefine every existing tuned skill yaml. Kept the 5 scales; added `root_tracking_reward_scale`
  additively instead (§1a/§2), byte-identical at defaults.
- **Reference-relative targets.** RoboNaldo uses `use_default_offset=True`
  (`tracking_env_cfg.py:380`) — the same default-pose pivot we already have. They never needed it.
  Confirms the decision to abandon this track.

### COPY
Everything in §3, plus the non-reward mechanisms in §4.

---

## 3. Term-by-term port list

Effective weights computed per stage. **S1** = `tracking_params.yaml`, **S2a/S2b** =
`task_params_1/2.yaml`. Stage 3 omitted — we are not following it.

### 3a. Motion tracking

| RoboNaldo | S1 | S2a | S2b | ours today | action |
|---|---|---|---|---|---|
| `motion_global_anchor_pos` | 0.5 | 1.0 | 0.1 | `motion_global_ref_position_error_exp` 1.0 | **re-stage per §1a** |
| `motion_global_anchor_ori` | 0.5 | 0.5 | 0.1 | `motion_global_ref_orientation_error_exp` 0.5 | **re-stage per §1a** |
| `motion_body_pos` | 1.0 | 1.0 | 1.0 | `motion_relative_body_position_error_exp` **2.0** | lower to 1.0 |
| `motion_body_ori` | 1.0 | 1.0 | 1.0 | `motion_relative_body_orientation_error_exp` 1.0 | ✓ match |
| `motion_body_lin_vel` | 1.0 | 1.0 | 1.0 | `motion_global_body_lin_vel` 1.0 | ✓ match |
| `motion_body_ang_vel` | 1.0 | 1.0 | 1.0 | `motion_global_body_ang_vel` 1.0 | ✓ match |
| `motion_feet_lin_vel` | 1.0 | 1.0 | 1.0 | `motion_global_feet_lin_vel` **1.0, LIVE** | ✅ **implemented, active** |

`motion_feet_lin_vel` is a foot-only linear-velocity tracking term (`std: 1.0`, both ankle_roll
links) — the 7th tracking term, previously missing entirely, now ported and shipped live (not
staged at 0.0) since it completes an already-proven-active category rather than introducing new
behavior. Weight/sigma match RoboNaldo's own value exactly in every stage (S1/S2a/S2b never
relax it, unlike the root/anchor terms). Routed through the relative-body family (unscaled by
`root_tracking_reward_scale`, per §1a/§2), matching RoboNaldo's own never-relaxed treatment.

### 3b. Regularization — **all 20 of 20 checked; 16 implemented, 4 deliberately skipped**

| RoboNaldo | S1 | S2a/2b | ours | priority |
|---|---|---|---|---|
| `arm_default_pose` (std 0.4) | 0.025 | 0.04 | `kick_arm_default_pose` @ **0.0**, staged -0.04 | ✅ **implemented** |
| `feet_air_time` (thr 0.15) | 50 | 50 | `kick_feet_air_time` @ **0.0**, staged 50.0 | ✅ **implemented** |
| `feet_clearance` (target 0.12) | -20 | -20 | `kick_swing_feet_clearance` @ **0.0**, staged -20.0 | ✅ **implemented** |
| `feet_contact_time` (thr 0.25) | -0.5 | -0.5 | `kick_feet_contact_time` @ **0.0**, staged -0.5 | ✅ **implemented** |
| `feet_slip` | -0.025 | -0.02 | `kick_feet_slip` **-0.5, LIVE** | ✅ **implemented, active** |
| `no_fly` | -0.05 | -0.02 | `kick_no_fly` @ **0.0**, staged -0.02 | ✅ **implemented** |
| `action_smoothness` (2nd-order) | -0.0015 | -0.1 / -0.07 | `kick_action_smoothness` @ **0.0**, staged -0.1 | ✅ **implemented** |
| `action_rate_l2` | -0.1 | -0.04 | `action_rate_l2` **-0.04, LIVE** | ✅ **fixed** (was -1.0, 25× too strong) |
| `locomotion_phase_orientation_l2` | -0.01 | -0.04 | `kick_penalty_swing_orientation` **-1.0, LIVE** | ✅ **implemented, active** |
| `locomotion_phase_torso_orientation_l2` | -0.01 | -0.04 | `kick_penalty_swing_torso_orientation` **-1.0, LIVE** | ✅ **implemented, active** |
| `locomotion_phase_lin_vel_z_l2` | -0.025 | -0.1 | `kick_penalty_lin_vel_z` @ **0.0**, staged -0.1 | ✅ **implemented** |
| `unstable_penalty` | -0.01 | -0.04 | `kick_penalty_unstable` @ **0.0**, staged -0.05 | ✅ **implemented** (§2 GUT correction: replaces the assumed `stable_anchor_pos_tracking` port — see row below) |
| `stable_anchor_pos_tracking` | 0 | 0 | — | ⏭️ **skip, correctly** — Stage-3-only (`weight = 10*reg_weight*adapt_motion_flag`, False in every stage we follow); `unstable_penalty` above is RoboNaldo's REAL S1/S2 post-kick mechanism |
| `undesired_contacts` (thr 100) | -0.1 | -0.1 | `kick_undesired_contacts` @ **0.0**, staged -0.1 | ✅ **implemented** (reuses the existing `UndesiredContacts` class, new kick-mode registration + threshold) |
| `feet_contact_force` (thr 400) | -1.0 | -1.0 | `kick_penalty_excess_contact_force` (dyn) | ✓ match (pre-existing, different dynamic-formula design) |
| `joint_limit` | -10 | -10 | `limits_dof_pos` -10 | ✓ match |
| `loco_dof_vel` | -3e-5 | -3e-5 | `kick_penalty_dof_vel` @ **0.0**, staged -3e-5 | ✅ **implemented** |
| `loco_torque` | -2e-7 | -2e-7 | `kick_penalty_torque` @ **0.0**, staged -2e-7 | ✅ **implemented** |
| `robot_alive` | 0 | **0.2** pre / **2.0** post-kick | `kick_alive` phase-shaped via `kick_alive_pre_kick_ratio` **0.1, LIVE** | ✅ **implemented, active** (§1b) |
| `ee_body_pos_termination_penalty` | -100 | -100 | `kick_penalty_ee_body_pos_divergence` @ **0.0**, staged -100.0 | ✅ **implemented** (Z-axis-only branch ported; RSI-`is_warmup` branch has no equivalent, documented in the term's own docstring) |
| `stand_still_*` (3 terms) | 0 | 0 | — | ⏭️ skip (needs `stand_still_env_ratio > 0`, unused in every stage we follow) |
| `hand_height_penalty` | 0 | 0 | — | ⏭️ skip (S3-only, weight 0 in every stage we follow) |
| `arm_pitch_same_sign_penalty` | 0 | 0 | — | ⏭️ skip (S3-only, weight 0 in every stage we follow) |

All 16 "✅ implemented" rows: live IsaacSim-verified (real checkpoint, 256 envs, 400-step
combined rollout, all firing together) — finite, correctly signed, non-degenerate. Full formulas,
adaptations, and citations to the exact RoboNaldo source line in each term's own docstring
(`managers/reward/terms/{wbt,locomotion}.py`).

**`arm_default_pose` is the highest-ROI single term.** It penalizes arm deviation from the
**default standing pose**, not from the clip. Against our measured strike-phase arm drift
(27.8°/27.6°, worse than the kick leg; 7 of 8 worst joints are arm joints), RoboNaldo's answer is
neither a joint-space tracking reward nor a moved action pivot — they simply stop the arms
wandering and accept clip infidelity there. One term, active weight, fully resumable, no retrain.
**Formula correction (verified against real source when implementing, 2026-08-05):** the real
`arm_default_pose_penalty` is `-mean_j(error_j^2 * weight_j)` (elbow joints weighted 5×) — a plain
negative-squared-error penalty, NOT an exp-kernel reward as originally described here. Ported as
`wbt.py:ArmDefaultPose`, returning the positive magnitude (this project's sign convention),
registered with a negative weight.
It also pairs with their per-joint action clipping (`tracking_env_cfg.py:384-408`), which scales
wrists to ~0.07 rad/unit — the wrist_roll joints were two of our four worst offenders.

### 3c. Shooting — **all 12 of 12 checked; 12 of 12 implemented** (5 newly ported, 6 re-enabled/live, 1 correctly deferred)

| RoboNaldo | S2a | S2b | ours (live yaml) | action |
|---|---|---|---|---|
| `error_ball_to_target` | 8.0 | 8.0 | `kick_error_ball_to_target` **20.0, LIVE** | ✅ **re-enabled** (§1b, k=2.5 ratio-scaled) |
| `ball_contact_orientation` | 1.6 | 2.4 | `kick_contact_orientation` **4.0, LIVE** | ✅ **re-enabled** |
| `robot_ball_contact` | 0.8 | 0.8 | ≈ `kick_goal_success_burst` **0.0** | correctly deferred — RoboNaldo's own `goal_reward_burst` (the closer analog) is ALSO 0.0 in S2a/S2b, only 500.0 in S3; see note below |
| `robot_ball_contact_count` | 0.8 | 0.8 | `kick_ball_contact_hit` **2.0, LIVE** | ✓ have (pre-existing, own design — dense per-tick credit while `has_kicked`, not RoboNaldo's own contact-count formula) |
| `robot_feet_ball_distance` | 0.8 | 0.4 | `kick_ball_proximity` **2.0, LIVE** | ✓ have (pre-existing, own design — dense foot-to-ball approach shaping) |
| `ball_velocity` | 0.4 | 0.4 | `kick_ball_velocity` **1.0, LIVE** | ✅ **re-enabled** (k=2.5 ratio-scaled) |
| `ball_over_line` | 0.4 | 0.4 | `kick_ball_over_line` @ **0.0**, staged 1.0 (k=2.5) | ✅ **implemented** (geometric adaptation: RoboNaldo's fixed-world-Y goal line → this project's own per-attempt spawn→target axis projection, same reward magnitudes) |
| `robot_com_ball_distance` | 0.8 | 0.8 | `kick_robot_com_ball_distance` @ **0.0**, staged 2.0 (k=2.5) | ✅ **implemented** |
| `robot_head_torso_ball_distance` | 0.8 | 0.8 | `kick_robot_torso_ball_distance` @ **0.0**, staged 2.0 (k=2.5) | ✅ **implemented** |
| `penalize_weak_foot_contact` | -0.4 | -0.4 | `kick_penalize_weak_foot_contact` @ **0.0**, staged -1.0 (k=2.5) | ✅ **implemented** (ported the exact Gaussian-bump-at-threshold shape, not re-derived to be monotonic) |
| `penalize_self_contact_feet` | -0.16 | -0.16 | `kick_penalize_self_contact_feet` @ **0.0**, staged -0.4 (k=2.5) | ✅ **implemented** |
| `goal_reward_burst` | 0 | 0 | `kick_goal_success_burst` **0.0** | ✓ both off until S3 (RoboNaldo's own value is 0 in S2a/S2b too) |
| — | — | — | `kick_predicted_error_ball_to_target` **10.0, LIVE** | ours only (densified shot-direction feedback, no RoboNaldo analog) |
| — | — | — | `kick_ball_approach_stance` **2.0, LIVE** | ours only (locomotion-approach stance shaping, no RoboNaldo analog) |

All 5 "✅ implemented" (newly-ported) rows: live IsaacSim-verified (real checkpoint, 256 envs,
400-step combined rollout, all 16 new terms firing together) — finite, correctly signed,
non-degenerate. `robot_ball_contact`'s own closer RoboNaldo analog (`goal_reward_burst`) is a
Stage-3 device (sparse success bursts only meaningful once `adapt_motion_flag`'s ball-steering is
active) — porting it active in S2-equivalent stages would not reproduce RoboNaldo's own design,
not a gap in this port.

---

## 4. Non-reward mechanisms to port — **4 of 4 P0 + 2 of 3 P1 checked; 5 implemented, 1 deliberately superseded, 1 out of scope**

| mechanism | RoboNaldo | ours | note |
|---|---|---|---|
| Reference-state init | `start_time_sampling_fraction: 1.0` in S1 | `rsi_scope_to_authored_clip: bool = False` (opt-in, no-op by default) | ✅ **implemented** (Phase 4a) — scopes RSI to `pre_recovery_motion_end_idx`, excluding the synthetic recovery/hold tail |
| Critical-frame oversampling | `critical_frame_adaptive_sampling`, window 10 | `critical_frame_oversampling_prob: float = 0.0`, `critical_frame_sampling_window: int = 10` (opt-in, no-op by default) | ✅ **implemented** (Phase 4b) — oversamples a ±window frames around `strike_start_idx` |
| Two global scalars | `reg_weight`, `goal_weight` | 5 per-skill category scales, **kept** | ⏭️ **deliberately superseded, not ported** — collapsing to 2 global scalars would silently redefine every existing tuned skill yaml under new semantics; Phase 2 instead added `root_tracking_reward_scale` *additively* alongside the existing 5, preserving byte-identical behavior at defaults. See §2 GUT correction above. |
| Per-joint action clipping | `clip={}` per joint group, wrists ~0.07 rad/unit | `RobotControlConfig.per_joint_action_clip: dict[str, tuple[float,float]] \| None = None` (opt-in, no-op by default) | ✅ **implemented** (Phase 4c) — RoboNaldo's raw clip *values* don't transfer verbatim (calibrated to their own action scale); only `ankle_roll`/`waist_roll`/`waist_pitch` re-derive correctly from this project's own URDF limits ÷ action_scale, staged commented-out in `experiment.py`. Wrist/shoulder-yaw drift is `ArmDefaultPose`'s job instead (§3b). |
| Proprioceptive history | `history_length=5` on base lin/ang vel, joint pos/vel | — | ⏭️ **out of scope** — changes obs width, breaks every checkpoint's warm-start (`FastSACAgent.load()` is `strict=True` throughout); see "Explicitly out of scope" below |
| Progressive termination loosening | `ee_body_pos` 0.25→0.35→0.5 (`tracking_env_cfg.py`, re-verified against source) | `MultiSkillConfig.bad_motion_body_pos_threshold: float = 0.25` (opt-in, no-op by default) | ✅ **implemented** (2026-08-05, Phase 8) — re-reading RoboNaldo's actual source found `bad_tracking`'s existing `bad_motion_body_pos` sub-check (`config_values/wbt/g1/termination.py`) was ALREADY byte-similar (same `BadTrackingZOnly` Z-only class, same 4 tracked bodies, same Stage-1 value 0.25) — only the progressive-widening KNOB was missing. Set `bad_motion_body_pos_threshold: 0.35` (or `0.5`) in configs/\*.yaml + resume to reproduce S2a/S2b tolerance, same per-skill-yaml-edit-then-resume curriculum mechanism as `root_tracking_reward_scale`. |
| Terminal penalty | `ee_body_pos_termination_penalty` -100, two branches (Z-only `terminated` + a full-3D `warmup_terminated` gated on `command.is_warmup`) | `kick_penalty_ee_body_pos_divergence` (Phase 7, §3c) — **extended** 2026-08-05 with the previously-dropped warmup branch | ✅ **implemented** (2026-08-05, Phase 8) — correction to Phase 7: re-reading `mdp/commands.py::MotionCommand.is_warmup` found `is_warmup = real_time_steps < warmup_time_steps + warmup_steps` reduces exactly to "fewer than `warmup_steps` (20) steps since the last reset", which this project's own `env.episode_length_buf` (already used by `bad_tracking`'s `grace_period_steps`) already tracks — no missing infra after all. Both branches now ported; `threshold` param synced with `bad_motion_body_pos_threshold` above under one source of truth (`config_values/unified/g1/reward.py`), mirroring RoboNaldo's own `task_overrides.py` (a single `termination_overrides.ee_body_pos.threshold` yaml entry sets both their termination AND this reward term together, by construction). Still a reward-side penalty only — does not itself terminate the episode; pairs with the termination row above. |

---

## 5. Suggested order

The terms are tuned **as a set** under two global scalars — one-at-a-time testing across 20+ of
them is combinatorially hopeless. That is the actual argument for forking: the current workspace
keeps the one-variable-at-a-time discipline; the fork gets to do a wholesale transplant. Run both;
don't let either inherit the other's methodology.

1. **Budget rebalance first, before porting anything.** `kick_alive` 10.0 → RoboNaldo ratio;
   re-enable the zeroed shooting terms; `action_rate_l2` -1.0 → -0.04. This is a config-only
   change on the *existing* codebase and it tests §1b, the largest single discrepancy, cheaply.
   **Done** (Phase 1).
2. **Per-term tracking relaxation** (§1a) — replaces category-wide scaling. Also existing-codebase.
   **Done** (Phase 2).
3. **P0 regularization block**, ported and enabled as one atomic set at the S2a weights.
   **Done** (Phase 3; §3b/§3c now fully ported, Phase 7).
4. **RSI + critical-frame oversampling + per-joint action clipping.** **Done** (Phase 4).
5. ~~**Swap post-kick stabilization** to the latched-anchor form.~~ **RESOLVED, not a swap
   (2026-08-05, re-verified against RoboNaldo's actual source, not just the paper):** the
   latched-anchor mechanism (`stabilize_anchor_pos_w`, `stable_anchor_pos_tracking`) is gated
   `weight = 10*reg_weight*float(adapt_motion_flag)` in their `tracking_env_cfg.py`, and
   `adapt_motion_flag` reads `false` in `tracking_params.yaml`/`task_params_1.yaml`/
   `task_params_2.yaml` (S1/S2a/S2b) and only `true` in `task_params_3.yaml` — confirmed directly
   from their yaml, not inferred. There is nothing to swap for the stages this fork follows: their
   REAL S1/S2 mechanism, `unstable_penalty` (weight `-0.2*reg_weight`, always active, gated only
   on `time_steps > critic_frame_index + kick_hold_steps`), is already ported as
   `penalty_kick_unstable` (§3b) with a formula-identical match
   (`base_lin_penalty + 0.5*base_ang_penalty`, confirmed against their `mdp/rewards.py` line for
   line). Porting the latch itself would be porting a Stage-3-only mechanism this project doesn't
   follow — deliberately not done.
6. P1 block. **2 of 3 done** (2026-08-05, Phase 8): `bad_motion_body_pos_threshold` (progressive
   termination loosening) and the warmup branch of `penalty_kick_ee_body_pos_divergence` (terminal
   penalty). Proprioceptive history remains explicitly out of scope. See §4 below.

## 6. The risk to name up front

If SAC is the binding constraint, all of the above lands and still fails. RoboNaldo's recipe is
tuned for on-policy PPO, and our permanent `_build_task_mode_partition` exists *specifically*
because SAC's replay buffer starved kick mode — a problem PPO does not structurally have. RoboNaldo
ships `config/g1/agents/rsl_rl_ppo_cfg.py`. Decide early whether the fork also runs a PPO arm on
the kick task alone, so that a null result is interpretable rather than ambiguous.
