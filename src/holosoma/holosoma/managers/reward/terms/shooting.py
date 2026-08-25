"""RoboNaldo-Stage-2-style shooting reward terms for the unified locomotion + ball-kicking task.

Implements the task-reward set of "RoboNaldo: Accurate, Stable and Powerful Humanoid Soccer
Shooting via Motion-Guided Curriculum Reinforcement Learning" (arXiv:2606.11092, Table B.1),
adapted to this project's setup. Two deliberate adaptations from the paper, both because of what
this environment actually measures:

1. **No foot force sensor is used.** The simulator's per-body contact forces can't distinguish
   ball contact from ground contact on the stance/swing foot, so "solid strike" is detected from
   the ball's own response instead: the ball is considered kicked the moment it moves away from
   its (per-env, possibly randomized) spawn point or picks up speed. Only the robot can move it,
   so this is unambiguous — and it measures the physical event the force sensor would proxy.
2. **The target is a ground point, not a goal plane**, so the paper's ballistic goal-line
   extrapolation (their Eq. 3) becomes a closest-approach-of-the-velocity-ray prediction: at every
   step post-contact, extend the ball's current planar velocity forward and reward small predicted
   miss distance to the target. Same densification purpose (per-step feedback on where the shot is
   *heading*, closing the credit-assignment gap between contact and outcome), robust to the ball
   rolling rather than flying.

All terms read the ball/target/kick-foot configuration PER ENV, from that env's currently assigned
motion skill (``MotionCommand.skill_ball_configs[motion_ids]`` in N-skill mode; broadcast from the
single ``configs/ball.yaml`` / ``env.simulator.simulator_config.scene.ball`` — see ``BallConfig`` —
in legacy single-clip mode), so the kick foot (left/right), target coordinate, spawn randomization,
and success radius are yaml-tunable with no code changes, independently per skill.

Per-attempt bookkeeping (the kick clip replays BeyondMimic-style 2-3x within one 20s episode,
teleporting the ball back to spawn each time — see MotionCommand.step's ended_env_ids block) is
handled by a single ``_ShotTracker`` shared by all terms, cached on the env and updated
idempotently once per control step. Latches reset at every clip restart and env reset, so each
kick attempt is scored independently.

All terms are meant to be registered with ``task_mode="kick"`` (RewardManager zeroes them for
locomotion-mode envs) — see config_values/unified/g1/reward.py.

Every term below also multiplies its raw output by ``holosoma.utils.shooting_curriculum.
current_w_g(env)`` (2026-07-20, extended to per-env 2026-07-23): a PER-ENV tensor (not a process-
wide scalar) — each env's assigned skill has its own TARGET shooting_reward_scale, which it ramps
toward under one shared ramp/hold SCHEDULE (iteration counts only, not per-skill). At
``BallConfig.shooting_reward_scale_ramp_iters > 0`` (or, N-skill mode, ``MultiSkillConfig``'s field
of the same name) this fades linearly 0 -> 1 over that many control steps instead of snapping
straight to full weight on the very first Stage-C step, closing the reward-side half of the Stage
B->C shock (see ``managers/observation/terms/unified.py``'s ``ball_pos_b``/``target_pos_b`` for the
paired observation-magnitude fade, and `stagec_obs_normalizer_shock.md` for the measurement). The
per-term weight (RewardTermCfg.weight in config_values/unified/g1/reward.py) is now just each
term's own relative scale k (RoboNaldo's Table B.1 ratios) — current_w_g folds in the (per-env)
target-w_g multiplication that weight used to bake in at config-import time, which stopped being
possible once which skill (and therefore which target) an env is running became a runtime-only
fact. ramp_iters <= 0 makes current_w_g's ramp factor 1.0 unconditionally.

3 of the 6 terms below (``ball_proximity``, ``contact_orientation``, ``ball_velocity``) multiply by
``_strike_phase_multiplier(env)`` (2026-07-31): 1.0 while ``MotionCommand.in_strike_phase`` (the
clip's authored strike/swing mode only, per-skill ``strike_start_idx``..``stand_start_idx``), 0.0
outside it. These three measure foot-ball CONTACT MECHANICS -- meaningless once the foot has moved
away from the ball -- so gating them to the strike specifically avoids rewarding e.g. a
locomotion-mode foot swing that happens to pass near the ball.

2026-08-04: ``ball_proximity`` and ``contact_orientation`` (but NOT ``ball_velocity``, see below)
additionally multiply by ``(~has_kicked).float()`` -- zero from the tick contact is first confirmed
onward, not just while ``in_strike_phase``. Root cause this closes: neither term has any OTHER
reason to decay after contact, and the authored clip's own follow-through moves the kick foot AWAY
from the ball for most of the remaining strike window (measured on this project's two production
clips: foot-ball distance goes from contact to 0.55m within 6 frames post-contact, then stays
near/above that for the ~35 remaining strike-window frames). Un-gated, that made
``exp(-d^2/sigma^2)`` collapse toward 0 for any policy that completes the authored swing, while a
policy that instead arrests the kick and parks the foot near the ball kept collecting ~1.0 for the
rest of the window -- a large, sustained incentive to NOT follow through, paid every tick, that had
nothing to do with shot power or accuracy. ``ball_velocity`` is the mirror image and stays gated the
OTHER way (``has_kicked.float()``, unchanged) -- it measures the ball's speed, which only exists
post-contact, so it should start (not stop) exactly where these two now stop. ``ball_contact_hit``
(weight 10.0, 5x ball_proximity's 2.0) already pays for confirmed contact from the same tick these
two shut off, so this does not reduce the incentive to actually make contact -- it removes a
SEPARATE, competing incentive to loiter near the ball once contact has already happened.

Also 2026-08-04: ``predicted_error_ball_to_target`` (below) gained a ``has_kicked`` gate on top of
its own pre-existing ``moving`` gate, for the mirror-image reason -- see that function's own
docstring for the measured contamination this closes.

The other 3 (``error_ball_to_target``, ``predicted_error_ball_to_target``, ``goal_success_burst``)
measure the OUTCOME of the kick, which by construction can't be known until after the ball has
traveled -- gating them to ``in_strike_phase`` (tried 2026-07-31, reverted same day) made
``goal_success_burst`` exactly 0 in every live sample, since a shot's roll-to-target time routinely
exceeds the ~1s strike window. Un-gating them entirely (also tried the same day) fixed that, but
live diagnosis then found a second, smaller problem: each of the three's OWN internal temporal
gate (``has_kicked``, ``moving``, ``success_latched`` respectively) isn't a precise enough proxy
for "the kick attempt has begun" on its own -- ``has_kicked`` in particular can flip True from the
robot's body/approaching leg incidentally brushing the ball during the locomotion-approach walk-up
(measured: 37 such flips over 1500 ticks x 256 envs, 92% NOT near a clip restart, ball-to-target
distance 4-6m at the flip -- i.e. a graze, not a strike), paying a small amount of undeserved
credit for that graze. Fix: these three also multiply by ``_post_locomotion_multiplier(env)``
(added 2026-07-31) -- 1.0 once ``time_steps >= strike_start_idx`` (covers ``in_strike_phase`` AND
everything after, i.e. every mode except locomotion-approach), 0.0 during locomotion. Narrower than
"ungated" but deliberately wider than ``in_strike_phase`` alone -- excludes exactly the one mode
(locomotion-approach) these terms were never meant to reward, without reintroducing the
too-narrow-window problem that made ``goal_success_burst`` unable to pay out in the first place.

Both ``_strike_phase_multiplier`` and ``_post_locomotion_multiplier`` collapse to a no-op in
legacy/single-clip mode (no scrubbed strike/stand frames configured) -- see MotionCommand.setup()
-- matching every other phase-gated mechanism in this project.

A 6th multiplier, ``_ood_gate_multiplier(env)`` (2026-08-01), applies to ALL 6 terms uniformly
(unlike the 3/3 split above) -- 0.0 for any attempt whose ball spawn was drawn from the OOD region
(``MotionCommand.is_ood_spawn``, set once per reset/clip-replay by ``draw_position_noise_with_ood``
-- see ``managers/command/terms/wbt.py``), 1.0 otherwise. This REVERSES an earlier decision
(2026-07-24) to leave the shooting reward untouched for OOD-spawn episodes: that decision assumed
the existing proximity/has_kicked-gated terms already decay to ~zero for an unreachable ball, but
``ood_region_multiplier``'s draw is NOT rejection-sampled away from the normal box, so a minority
of "OOD" attempts were still landing reachable and paying inconsistent partial reward -- explicit
zeroing removes that noise regardless of where the draw landed. Orthogonal to the strike/
post-locomotion split above (OOD-spawn is a per-ATTEMPT property, not a phase), so it multiplies
into every term alongside whichever of the other two gates that term already carries. Collapses
to a no-op whenever ``ood_spawn_probability<=0.0`` (the default), same convention as the other two
gates.

``has_kicked`` itself (``_ShotTracker.update()``, consumed by all 3 outcome terms above and by
``ball_velocity``) was also tightened at the root (2026-08-01), via new pure helpers
``_is_foot_contact``/``_detect_kick``: previously it fired purely from the ball's own motion
(displacement from spawn or ball speed), blind to which body caused it and unbounded by phase --
the actual mechanism behind the locomotion-approach graze problem ``_post_locomotion_multiplier``
above was added to patch downstream. It now requires ``_is_in_strike_phase`` (originally
``_is_post_locomotion``, tightened same day -- see below) to gate the latch's TRIGGER, AND
requires the env's assigned kick foot's own contact point (``kick_foot_index``, already
skill-resolved to left/right per ``SkillConfig.kick_foot``, the same body
``ball_proximity``/``contact_orientation`` use) to have confirmed geometric contact with the ball
-- either THIS tick directly, or at some earlier point in the SAME attempt via a sticky
per-attempt latch (``_ShotTracker.foot_contact_ever``), before the original ball-motion proxy is
allowed to trigger at all. The ball-motion proxy is kept, not removed -- a fast strike can clear
the ball's contact zone between two sampled 50Hz control-step ticks without the geometric check
ever firing at that exact tick -- but it's no longer an independent, body-blind trigger; it only
supplies timing precision once foot contact is already confirmed for the attempt. (An earlier
version of this fix OR'd the ball-motion proxy in unconditionally, gated only by phase -- live
measurement then found it was still driving 79% of all detections with zero geometric
confirmation, an uncontrolled gap a wrong-foot/torso touch could exploit even inside the strike
window; this is the tightened version.)

The phase gate on has_kicked's trigger was narrowed a second time the same day, from
``_is_post_locomotion`` (True from ``strike_start_idx`` onward, unbounded) to
``_is_in_strike_phase`` (True only for ``strike_start_idx <= time_steps < stand_start_idx`` --
user-requested: a kick attempt should only be able to START during the authored swing itself, not
during post-kick-standing/recovery/hold). ``has_kicked`` remains a STICKY latch
(``self.has_kicked |= _detect_kick(...)``, OR-accumulated, only cleared on ``new_attempt``), so
this only closes off LATE triggering -- a strike that genuinely lands within the strike window
still correctly reads True for the rest of the attempt, including all of post-kick-standing/
recovery/hold; only a foot-ball contact (or the ball-motion proxy) that first occurs AFTER
``stand_start_idx`` no longer counts as a strike at all. Net effect: ``has_kicked`` can now only
ever be TRIGGERED while ``in_strike_phase`` is True, and can no longer latch True at all without
the ASSIGNED kick foot having confirmed contact with the ball at some point during that window --
see ``_detect_kick``'s own docstring for the full breakdown. The 3 outcome terms' own
``_post_locomotion_multiplier`` PAYOUT gate deliberately stays on the wider
``_is_post_locomotion`` boundary, unchanged -- a shot's roll-to-target time routinely exceeds the
strike window, so those terms still need to keep paying out well after ``stand_start_idx`` once
``has_kicked`` has (necessarily earlier) latched True.

What counts as "confirmed contact" itself got more precise the same day: ``foot_contact``
(``_ShotTracker.update()``) now prefers TRUE geometric contact on IsaacSim -- two per-foot PhysX
``ContactSensor``s filtered against the ball (``BaseSimulator.get_ball_foot_contact_pos_w``, real
override only on IsaacSim's simulator subclass) report the actual world-frame contact position,
NaN exactly when that foot isn't touching the ball, no offset-point or margin approximation
involved. ``_geometric_foot_contact`` gathers the correct per-env side (left/right, per-skill) and
returns ``None`` on backends without this (MuJoCo, IsaacGym) -- ``_ShotTracker.update()`` falls
back to the original single-offset-point-plus-margin approximation (``_is_foot_contact``) only
then. Deliberately IsaacSim-only, not backported everywhere: reward only ever trains on IsaacSim
(MuJoCo runs are diagnostic/sim2sim eval, no gradients flow through them), so the geometric
precision only matters where it's implemented, and the fallback keeps every other backend's
existing (if less precise) behavior working unchanged.

A 7th shooting term, ``ball_contact_hit`` (2026-08-01), was added on top of RoboNaldo's original 6
once has_kicked's precision above made it worth rewarding directly: it pays a flat, dense per-tick
amount for every tick has_kicked is True, gated to ``_post_locomotion_multiplier`` (the same wide
gate the 3 outcome terms use) so credit accrues through recovery/hold, not just the strike itself.
See its own docstring for the full rationale.

An 8th, ``ball_approach_stance`` (2026-08-01), closes the remaining phase gap: terms 1-7 are ALL
zero during locomotion-approach, so that phase -- 170 of ~400 ticks per cycle for this project's
production clips, and 4.8x the strike window's env-tick exposure -- had no ball-aware reward at
all, leaving the approach driven purely by ball-blind clip tracking even though the ball spawn is
randomized +/-0.35m and the policy observes it live. It is the ONLY term here gated to
``_locomotion_approach_multiplier``, and it rewards reaching the swing's authored STANCE relative
to the actual ball rather than proximity to the ball (which would re-create the very
locomotion-approach graze the has_kicked work above exists to prevent) -- see its own docstring.

So the 8 terms partition across three DISJOINT phase gates: ``_locomotion_approach_multiplier``
(1 term: ball_approach_stance), ``_strike_phase_multiplier`` (3 contact-mechanics terms), and
``_post_locomotion_multiplier`` (4 outcome terms). All 8 additionally multiply by
``_ood_gate_multiplier`` and ``current_w_g``. Within the 3 ``_strike_phase_multiplier`` terms,
2026-08-04's ``has_kicked`` split (above) further divides them by CONTACT state, not just phase: 2
pay only BEFORE contact (``ball_proximity``, ``contact_orientation``), 1 pays only AFTER
(``ball_velocity``) -- so no env is ever "in strike phase with no contact-mechanics reward at all",
but no tick pays more than one of the two at once either.

2026-08-21, OPT-IN: ``ball_velocity`` can be moved to ``_post_locomotion_multiplier`` (and switched
to the latched ``max_ball_speed``) via its ``use_post_locomotion_gate`` / ``use_latched_peak_speed``
params -- OFF by default, so the partition above is what an unedited config gets. When enabled,
``ball_velocity`` leaves the strike group for the outcome group, both remaining strike-gated terms
are pre-contact, and an env past contact draws its shooting reward entirely from the
post-locomotion outcome group. See that function's own docstring.

``ball_proximity`` was changed from a Laplacian to a Gaussian kernel (2026-08-01, same day as
``ball_approach_stance`` above): live gradient arithmetic found its old exp(-d/sigma) shape
outlived every ``motion_*`` tracking term's own Gaussian exp(-e^2/sigma^2) at the errors an
off-nominal ball actually produces at strike_start (median 0.58m, measured) -- tracking's own
gradient, not this term's strength, is what had vanished by then. See ``ball_proximity``'s own
docstring for the full crossover arithmetic and why matching the kernel shape was chosen over
raising the weight.
"""

from __future__ import annotations

from typing import Any

import torch

from holosoma.utils.rotations import quat_apply, quat_rotate, quat_rotate_inverse, yaw_quat
from holosoma.utils.shooting_curriculum import current_w_g

_KICK_DETECT_DISPLACEMENT_M = 0.05
"""Ball xy-displacement from its own spawn point beyond which the attempt counts as kicked."""

_KICK_DETECT_SPEED_MPS = 0.25
"""Ball planar speed beyond which the attempt counts as kicked (catches a moving ball before it
has displaced far)."""

_KICK_DETECT_FOOT_CONTACT_OFFSET_M = 0.09
"""Same forward-along-toe offset as ball_proximity/contact_orientation's own default
contact_offset_m -- kick detection reads the kick foot's real striking surface (see
kick_foot_contact_pos_w's own docstring), not the raw ankle-roll-link origin ~9cm behind/above
it.

2026-08-01: since the addition of TRUE geometric ball-foot contact sensing on IsaacSim (see
_geometric_foot_contact / BaseSimulator.get_ball_foot_contact_pos_w), this constant (and
_KICK_DETECT_FOOT_CONTACT_MARGIN_M below) only governs the FALLBACK offset-point approximation
used on backends without real contact sensors (MuJoCo, IsaacGym) -- it no longer describes
IsaacSim's own has_kicked detection, which now reads real PhysX contact instead."""

_KICK_DETECT_FOOT_CONTACT_MARGIN_M = 0.09
"""Extra tolerance beyond the ball's own radius for "the kick foot's contact point is touching
the ball" (2026-08-01, retuned same day from an initial 0.03). Ticks are sampled once per control
step (50Hz) -- true continuous-time contact can be a fleeting graze that a single discrete sample
misses by a few cm even when a real touch occurred; this margin absorbs that without meaningfully
loosening what counts as contact.

Retuning rationale (data-driven, not guessed): once `foot_contact_ever` made ball_moved require a
confirmed foot touch (see _detect_kick), a live IsaacSim probe (checkpoint 335k, skill 0, 606
cycles) found 115 cycles where the ball moved post-locomotion with the ORIGINAL 0.03 margin never
confirming contact (measured with a per-tick-accurate capture of production's own internal
foot-distance/cycle-boundary signals -- an earlier, less careful version of this same probe that
recomputed foot position externally after env.step() returned had a one-tick cycle-boundary
misalignment and reported a differently-shaped, less reliable 189; the 115 figure is the
validated one). The achieved minimum foot-to-ball distance in those 115 cycles was bimodal, not
uniformly "a bit too tight": 60.0% (69) clustered tightly in [0.14, 0.20)m just past the old
boundary (min itself was 0.1414m, essentially grazing it) -- the shape expected from a genuine
discrete-sampling gap -- while 16.5% (19) were >=1.0m away (max 6.0m), clearly NOT the same foot
narrowly missing, almost certainly a different body part or the other foot. 0.09m was chosen to
fully absorb the tight cluster (raises the threshold to ball_radius + 0.09 = 0.20m) while leaving
the far tail excluded -- confirmed exactly on a rerun: 46 cycles remained unconfirmed at the new
threshold, precisely 115 - 69 = 46, and the same 19 far-outlier (>=1.0m) cycles reappeared
unchanged (margin-invariant, as expected). Net effect on this checkpoint/skill: per-cycle hit rate
5.8% -> 19.5% (35/606 -> 118/606) from the retune alone, no policy change. A wider margin (e.g.
0.19m, which would also absorb the 0.20-0.30m "moderate near-miss" bucket) was considered and
rejected: at that point the threshold approaches 3x the ball radius, eroding the foot-targeting
precision this whole mechanism exists for."""


def _is_post_locomotion(env: Any) -> torch.Tensor:
    """Bool [num_envs]: True once time_steps >= strike_start_idx (in_strike_phase itself, PLUS
    everything after -- post-kick-standing, recovery, hold), False during locomotion-approach.
    Single source of truth for this boundary, consumed by _post_locomotion_multiplier below (float
    reward gate for the 3 outcome terms). Was also used to gate has_kicked's own LATCH in
    _ShotTracker.update() until 2026-08-01, when that was narrowed to _is_in_strike_phase below
    (has_kicked may now only be TRIGGERED within the strike window itself, not any time
    post_locomotion is true) -- see _detect_kick's docstring for the full rationale."""
    motion_command = env.command_manager.get_state("motion_command")
    return motion_command.time_steps >= motion_command.strike_start_idx[motion_command.motion_ids]


def _is_in_strike_phase(env: Any) -> torch.Tensor:
    """Bool [num_envs]: True only during in_strike_phase itself (mode 2, the authored swing) --
    strike_start_idx <= time_steps < stand_start_idx. False during BOTH locomotion-approach (mode
    1) and post-kick-standing/recovery/hold (mode 3 onward) -- strictly narrower than
    _is_post_locomotion above, which stays True through mode 3 onward too. Single source of truth
    for this boundary, consumed two ways: _strike_phase_multiplier below converts it to a float
    reward gate for the 3 contact-mechanics terms; _ShotTracker.update() (2026-08-01) uses the raw
    bool to gate has_kicked's own LATCH -- see _detect_kick's docstring for why has_kicked's
    TRIGGER window was narrowed here (from the wider _is_post_locomotion it originally used) while
    has_kicked itself still correctly stays True through post-kick once triggered, via the latch's
    own sticky (OR-accumulate, never cleared mid-attempt) semantics."""
    motion_command = env.command_manager.get_state("motion_command")
    return motion_command.in_strike_phase


def _is_locomotion_approach(env: Any) -> torch.Tensor:
    """Bool [num_envs]: True ONLY during locomotion-approach (time_steps < strike_start_idx) --
    the exact complement of _is_post_locomotion, and therefore disjoint from BOTH other phase
    gates. Added 2026-08-01 for ball_approach_stance, the first shooting term that pays out in
    this phase rather than being excluded from it; every other term in this module is gated to
    _is_in_strike_phase or _is_post_locomotion, both of which are False here by construction.

    IMPORTANT -- collapses to an all-False no-op in legacy/single-clip mode, unlike the other two
    gates which collapse to all-TRUE: with no scrubbed strike/stand frames configured,
    strike_start_idx collapses onto motion_start_idx (see MotionCommand.setup()), so
    time_steps >= strike_start_idx is true from the very first tick and there is no
    locomotion-approach phase at all. ball_approach_stance is therefore inert in legacy mode --
    the safe direction (a term that never fires, not one that fires everywhere), and consistent
    with the fact that legacy single-clip configs also don't randomize the ball spawn, which is
    the entire problem this term exists to address."""
    return ~_is_post_locomotion(env)


def _is_foot_contact(foot_to_ball_dist: torch.Tensor, ball_radius: float) -> torch.Tensor:
    """Bool [num_envs]: True iff the (already skill-resolved, left/right per SkillConfig.kick_foot
    -- see _ShotTracker.__init__'s kick_foot_index) kick foot's own contact point is within
    ball_radius + _KICK_DETECT_FOOT_CONTACT_MARGIN_M of the ball THIS tick. Pure geometric
    primitive, factored out so both _ShotTracker.update() (to advance the foot_contact_ever
    latch) and _detect_kick (to trigger has_kicked directly) share one definition of "touching"."""
    return foot_to_ball_dist <= (ball_radius + _KICK_DETECT_FOOT_CONTACT_MARGIN_M)


def _detect_kick(
    *,
    foot_contact: torch.Tensor,
    foot_contact_ever: torch.Tensor,
    displacement: torch.Tensor,
    ball_speed: torch.Tensor,
    in_strike_phase: torch.Tensor,
) -> torch.Tensor:
    """Pure per-env kick-detection logic, factored out of _ShotTracker.update() so it's
    unit-testable without a real/fake simulator (everything else update() touches needs one;
    this doesn't). True iff in_strike_phase AND (foot_contact this tick, OR the ball
    moved/sped up AND the kick foot has confirmed contact at some point this attempt):

    - foot_contact (this tick, from _is_foot_contact): the ASSIGNED kick foot actually reaching
      the ball right now -- precise and foot-TARGETED, so a wrong-foot or torso brush never
      triggers this branch, unlike the ball-motion checks below which are blind to which body
      caused it.
    - ball_moved (displacement > _KICK_DETECT_DISPLACEMENT_M or ball_speed >
      _KICK_DETECT_SPEED_MPS) AND foot_contact_ever (2026-08-01, tightened from a bare OR):
      ball_moved alone is body-blind and, measured live, was driving 79% of all detections with
      no geometric confirmation at that tick -- an uncontrolled gap a wrong-foot/torso touch
      could exploit even inside the strike window, which phase-gating alone does NOT close.
      Requiring foot_contact_ever (the caller's sticky per-attempt latch, already True the
      instant foot_contact first fires) makes ball_moved's role purely about TIMING PRECISION --
      catching the tick a fast strike's resulting motion crosses threshold, which foot_contact's
      own single-sample check might miss between two 50Hz ticks -- not an independent, unverified
      trigger. A real ~1s swing should register at least one geometric-contact sample somewhere
      in its approach/contact/follow-through, making foot_contact_ever a realistic bar to clear,
      not a stricter version of the original sampling-gap problem.
    - in_strike_phase gates the WHOLE expression (2026-08-01, tightened from the wider
      _is_post_locomotion -- strike_start_idx onward unbounded -- this originally used):
      has_kicked can now ONLY be TRIGGERED (transition False->True) while
      strike_start_idx <= time_steps < stand_start_idx, not any time after strike_start_idx.
      Once triggered, it stays True regardless -- has_kicked is a sticky per-attempt latch
      (`self.has_kicked |= _detect_kick(...)` in _ShotTracker.update(), OR-accumulated, only
      cleared on new_attempt), so a kick that lands during the strike window still correctly
      reads True throughout post-kick-standing/recovery/hold; this change only closes off
      LATE triggering -- a foot-ball contact (or the ball-motion proxy) that first occurs AFTER
      stand_start_idx no longer counts as a strike at all, on the theory that a genuine strike's
      contact should occur within the authored swing itself, not during standing/recovery/hold.
      (Narrower than the locomotion-approach-only exclusion _is_post_locomotion still provides for
      the 3 outcome terms' own _post_locomotion_multiplier gate, which stays unchanged and wide --
      those terms need to keep PAYING OUT well past stand_start_idx, since a shot's roll-to-target
      time routinely exceeds the ~1s strike window; this is has_kicked's TRIGGER condition only,
      not those terms' own payout window.)"""
    ball_moved = (displacement > _KICK_DETECT_DISPLACEMENT_M) | (ball_speed > _KICK_DETECT_SPEED_MPS)
    return in_strike_phase & (foot_contact | (ball_moved & foot_contact_ever))


def _geometric_foot_contact(
    kick_foot_is_left: torch.Tensor,
    left_contact_pos: torch.Tensor | None,
    right_contact_pos: torch.Tensor | None,
) -> torch.Tensor | None:
    """Bool [num_envs], or None if the backend doesn't support geometric contact sensors (every
    backend except IsaacSim, 2026-08-01) -- callers must fall back to _is_foot_contact's
    offset-point+margin approximation in that case, NOT treat None as "no contact". Gathers each
    env's own assigned kick foot's contact position (per-skill left/right, mirroring the same
    per-env gather kick_foot_index already does for kick_foot_contact_pos_w/kick_foot_vel_xy) from
    two IsaacSim ContactSensors (one per foot, filtered against the ball -- see
    BaseSimulator.get_ball_foot_contact_pos_w), then checks for NaN, which
    ContactSensorData.contact_pos_w reports exactly when that env's filtered pair isn't touching.
    torch.where is a pure elementwise select, so a NaN on the UNSELECTED side never contaminates
    the result -- no special handling needed beyond the isnan check on the gathered side."""
    if left_contact_pos is None or right_contact_pos is None:
        return None
    contact_pos = torch.where(kick_foot_is_left.unsqueeze(-1), left_contact_pos, right_contact_pos)
    return ~torch.isnan(contact_pos).any(dim=-1)


def _strike_stance_ball_b_per_motion(motion_command: Any, device: Any) -> torch.Tensor:
    """(num_motions, 2): where the ball sits in the robot's PELVIS HEADING FRAME at the instant
    the authored strike begins (strike_start_idx), for a NOMINAL (un-randomized) ball spawn --
    i.e. "the stance the swing was authored for". Per-skill CONSTANT, computed once at
    _ShotTracker construction; consumed by ball_approach_stance as its target (2026-08-01).

    Derived, not configured: re-deriving it from the clip + the same nominal ball offset the
    spawner itself uses means it can never silently drift out of sync with either, the way a
    hand-entered pair of yaml numbers would if a clip were re-scrubbed or SkillConfig.x/y retuned.
    Two facts make this well-defined:

    1. ``MotionCommand.ball_reset_state_per_motion[:, :2]`` is the ROBOT-LOCAL nominal (forward,
       lateral) ball offset -- the single source of truth the real spawner reads (see that
       method's own docstring), already uniform across N-skill mode (SkillConfig.x/y) and legacy
       mode (BallConfig.position broadcast), so this needs no mode branching of its own.
    2. The ball is anchored to the robot's pose at CLIP START and then the clip walks the robot
       forward; both poses come from the same reference trajectory (``motion.body_pos_w`` /
       ``body_quat_w`` at ``motion_start_idx`` and ``strike_start_idx``), so the offset between
       them is exactly the approach the clip intends.

    env_origins deliberately does NOT appear anywhere below (unlike MotionCommand.root_pos_w,
    which adds it): every quantity here is a DIFFERENCE between two points in the same reference
    clip's own raw frame, so the per-env origin cancels identically -- adding it would be a no-op
    at best and a per-env-varying constant leaking into a per-MOTION table at worst.

    Yaw-only rotation (``yaw_quat``), matching both the spawner's own placement convention and
    ``observation/terms/unified.py``'s ``_ball_pos_b_raw`` -- so the target this term steers
    toward is expressed in exactly the same frame as the ``kick_ball_pos_b`` the policy observes,
    rather than a subtly different one the policy would have to learn to reconcile."""
    motion = motion_command.motion
    num_motions = motion.num_motions
    out = torch.zeros(num_motions, 2, dtype=torch.float32, device=device)
    for m in range(num_motions):
        start_idx = int(motion.motion_start_idx[m].item())
        strike_idx = int(motion_command.strike_start_idx[m].item())
        pos_at_start = motion.body_pos_w[start_idx, 0].to(device).unsqueeze(0)  # (1, 3)
        quat_at_start = motion.body_quat_w[start_idx, 0].to(device).unsqueeze(0)  # (1, 4) xyzw
        pos_at_strike = motion.body_pos_w[strike_idx, 0].to(device).unsqueeze(0)
        quat_at_strike = motion.body_quat_w[strike_idx, 0].to(device).unsqueeze(0)

        local_xy = motion_command.ball_reset_state_per_motion[m, :2].to(device)
        local_xyz = torch.cat([local_xy, torch.zeros(1, device=device)]).unsqueeze(0)  # (1, 3)
        # Nominal ball, in the clip's own frame: same transform as MotionCommand.local_xy_to_world
        # (yaw-rotate the robot-local offset, add the anchor position), minus the env_origins term.
        ball_w = quat_rotate(yaw_quat(quat_at_start, w_last=True), local_xyz, w_last=True) + pos_at_start
        rel = ball_w - pos_at_strike
        out[m] = quat_rotate_inverse(yaw_quat(quat_at_strike, w_last=True), rel, w_last=True)[0, :2]
    return out


class _ShotTracker:
    """Shared per-step shot state for all shooting reward terms.

    Lazily constructed and cached on the env (``env._shot_tracker``); ``update()`` is called by
    every term but only executes once per control step (guarded by ``env.common_step_counter``).
    Reward computation runs BEFORE MotionCommand.step()'s clip-end mid-episode reset (see
    BaseTask._compute_reward's call order), so a step's rewards always see the pre-teleport ball
    state, and a clip restart is observed here on the FOLLOWING step as time_steps decreasing.
    """

    def __init__(self, env: Any):
        num_envs, device = env.num_envs, env.device
        ball_cfg = env.simulator.simulator_config.scene.ball
        assert ball_cfg is not None, "shooting reward terms require a ball in the scene (scene.ball)"
        self.ball_radius = float(ball_cfg.radius)

        self.ball_indices = env.simulator.get_actor_indices("ball", env_ids=None)
        self._env_arange = torch.arange(num_envs, device=device)

        # Per-motion kick_foot / success_radius, gathered per-env (via each env's fixed motion_id)
        # fresh every update() -- not resolved once here and cached, so this stays correct
        # regardless of exactly when _ShotTracker happens to get lazily constructed relative to
        # env resets. Empty skill_ball_configs (legacy single-clip mode) broadcasts ball_cfg's
        # single kick_foot/success_radius to every motion -- bit-identical to the old scalar
        # behavior whenever there's only one motion.
        motion_command = env.command_manager.get_state("motion_command")
        skill_configs = getattr(motion_command, "skill_ball_configs", [])
        num_motions = motion_command.motion.num_motions
        foot_body_name = env.robot_config.foot_body_name

        def _resolve_foot_index(kick_foot: str) -> int:
            # Kick foot rigid-body index, resolved from a kick_foot ("left"/"right") + the robot's
            # configured foot body name (e.g. "ankle_roll_link" -> "right_ankle_roll_link").
            candidates = [s for s in env.body_names if s.startswith(kick_foot) and foot_body_name in s]
            assert len(candidates) == 1, (
                f"kick_foot={kick_foot!r} + foot_body_name={foot_body_name!r} matched "
                f"{candidates!r} in body_names — expected exactly one body"
            )
            return int(env.simulator.find_rigid_body_indice(candidates[0]))

        # Tracked-body-list column for the kick foot, per motion -- mirrors _resolve_foot_index
        # above but indexes into MotionCommand.motion_cfg.body_names_to_track (the reduced ~14-body
        # tracked list) rather than the simulator's own full body_names, since it feeds
        # tracked_body_indexes (see kick_foot_reference_quat_w below), not _rigid_body_rot. 2026-08-09,
        # added for foot_strike_pitch's reference_relative mode.
        body_names_to_track = list(motion_command.motion_cfg.body_names_to_track)

        def _resolve_tracked_foot_col(kick_foot: str) -> int:
            candidates = [i for i, s in enumerate(body_names_to_track) if s.startswith(kick_foot) and foot_body_name in s]
            assert len(candidates) == 1, (
                f"kick_foot={kick_foot!r} + foot_body_name={foot_body_name!r} matched "
                f"{[body_names_to_track[i] for i in candidates]!r} in body_names_to_track — expected exactly one"
            )
            return candidates[0]

        if skill_configs:
            assert len(skill_configs) == num_motions, (
                f"skill_ball_configs has {len(skill_configs)} entries but {num_motions} motions loaded"
            )
            self.kick_foot_index_per_motion = torch.tensor(
                [_resolve_foot_index(sc.kick_foot) for sc in skill_configs], dtype=torch.long, device=device
            )
            # Which side (left/right) each motion's kick_foot is, per motion -- used to gather
            # between the two IsaacSim ball-foot ContactSensors (see _geometric_foot_contact).
            # Both "left"/"right" are the only values kick_foot can ever validate to
            # (config_types/multi_skill.py), so this is a direct, exhaustive boolean, not an
            # approximation derived by comparing kick_foot_index against known body indices.
            self.kick_foot_is_left_per_motion = torch.tensor(
                [sc.kick_foot == "left" for sc in skill_configs], dtype=torch.bool, device=device
            )
            self.success_radius_per_motion = torch.tensor(
                [sc.success_radius for sc in skill_configs], dtype=torch.float32, device=device
            )
            self.tracked_foot_col_per_motion = torch.tensor(
                [_resolve_tracked_foot_col(sc.kick_foot) for sc in skill_configs], dtype=torch.long, device=device
            )
        else:
            idx = _resolve_foot_index(ball_cfg.kick_foot)
            self.kick_foot_index_per_motion = torch.full((num_motions,), idx, dtype=torch.long, device=device)
            self.kick_foot_is_left_per_motion = torch.full(
                (num_motions,), ball_cfg.kick_foot == "left", dtype=torch.bool, device=device
            )
            self.success_radius_per_motion = torch.full(
                (num_motions,), float(ball_cfg.success_radius), dtype=torch.float32, device=device
            )
            tracked_col = _resolve_tracked_foot_col(ball_cfg.kick_foot)
            self.tracked_foot_col_per_motion = torch.full((num_motions,), tracked_col, dtype=torch.long, device=device)
        # Placeholder per-env values, refreshed every update() from the per-motion tables above via
        # each env's current motion_id (env 0's motion for the very first update() before that).
        self.kick_foot_index = self.kick_foot_index_per_motion[torch.zeros(num_envs, dtype=torch.long, device=device)]
        self.tracked_foot_col = self.tracked_foot_col_per_motion[torch.zeros(num_envs, dtype=torch.long, device=device)]
        self.kick_foot_is_left = self.kick_foot_is_left_per_motion[
            torch.zeros(num_envs, dtype=torch.long, device=device)
        ]
        self.success_radius = self.success_radius_per_motion[torch.zeros(num_envs, dtype=torch.long, device=device)]

        # Per-motion ideal pre-strike stance (2026-08-01), consumed by ball_approach_stance -- a
        # derived CONSTANT, unlike the three tables above which come straight from config. Built
        # here rather than lazily on first use so a clip/config mismatch surfaces at construction
        # time, alongside _resolve_foot_index's own assert, not mid-training.
        self.strike_stance_ball_b_per_motion = _strike_stance_ball_b_per_motion(motion_command, device)
        self.strike_stance_ball_b = self.strike_stance_ball_b_per_motion[
            torch.zeros(num_envs, dtype=torch.long, device=device)
        ]

        # NOTE: the target itself is NOT cached here — it's per-env, re-randomized at every
        # reset/clip replay, and owned by MotionCommand (target_xy_w), which is also what the
        # kick_target_pos_b observation reads. Reading the same buffer in update() guarantees the
        # rewards score against exactly the target the policy observes.

        # Per-attempt latches.
        self.has_kicked = torch.zeros(num_envs, dtype=torch.bool, device=device)
        # True once the ASSIGNED kick foot (left/right, per this env's skill) has confirmed
        # geometric contact with the ball at least once this attempt (2026-08-01) -- makes
        # _detect_kick's ball_moved branch require foot identity, not just ball motion. See
        # _detect_kick's own docstring for the full rationale.
        self.foot_contact_ever = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.min_target_dist = torch.full((num_envs,), float("inf"), device=device)
        self.max_ball_speed = torch.zeros(num_envs, device=device)
        self.success_latched = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.burst_remaining = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.burst_active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._prev_time_steps = torch.zeros(num_envs, dtype=torch.long, device=device)

        # Per-step snapshots consumed by the terms.
        self.ball_pos_w = torch.zeros(num_envs, 3, device=device)
        self.ball_vel_xy = torch.zeros(num_envs, 2, device=device)
        self.ball_speed = torch.zeros(num_envs, device=device)
        self.ball_to_target_xy = torch.zeros(num_envs, 2, device=device)

        self._last_update_step = -1

    def update(self, env: Any, burst_steps: int) -> None:
        step = int(env.common_step_counter)
        if step == self._last_update_step:
            return
        self._last_update_step = step

        motion_command = env.command_manager.get_state("motion_command")

        # Refresh this env's kick_foot/success_radius from its (fixed-for-life, see
        # UnifiedManager._build_task_mode_partition) assigned motion_id every step -- cheap gather,
        # and avoids any dependency on exactly when this tracker was first constructed relative to
        # env resets.
        self.kick_foot_index = self.kick_foot_index_per_motion[motion_command.motion_ids]
        self.kick_foot_is_left = self.kick_foot_is_left_per_motion[motion_command.motion_ids]
        self.success_radius = self.success_radius_per_motion[motion_command.motion_ids]
        self.strike_stance_ball_b = self.strike_stance_ball_b_per_motion[motion_command.motion_ids]
        self.tracked_foot_col = self.tracked_foot_col_per_motion[motion_command.motion_ids]

        # New attempt: the clip restarted (mid-episode replay resets time_steps backwards) or the
        # env itself was reset (first post-reset step). Both must clear the per-attempt latches.
        time_steps = motion_command.time_steps
        new_attempt = (time_steps < self._prev_time_steps) | (env.episode_length_buf <= 1)
        self._prev_time_steps = time_steps.clone()
        if new_attempt.any():
            self.has_kicked[new_attempt] = False
            self.foot_contact_ever[new_attempt] = False
            self.min_target_dist[new_attempt] = float("inf")
            self.max_ball_speed[new_attempt] = 0.0
            self.success_latched[new_attempt] = False
            self.burst_remaining[new_attempt] = 0

        # Ball state snapshot (world frame; planar quantities — the target is a ground point).
        # Target read fresh from MotionCommand every step: it's re-randomized per attempt.
        ball_states = env.simulator.all_root_states[self.ball_indices]
        self.ball_pos_w = ball_states[:, :3]
        self.ball_vel_xy = ball_states[:, 7:9]
        self.ball_speed = torch.norm(self.ball_vel_xy, dim=-1)
        self.ball_to_target_xy = motion_command.target_xy_w - self.ball_pos_w[:, :2]

        # Kick detection (2026-08-01): foot-targeted geometric contact, OR the ball-motion proxy
        # gated by foot_contact_ever (this attempt's own kick foot has confirmed contact at least
        # once) -- both further gated by _is_in_strike_phase (has_kicked may only be TRIGGERED
        # within the strike window itself; the latch's own sticky semantics below keep it True
        # through post-kick once triggered). See _detect_kick's own docstring for the full
        # rationale. foot_contact itself prefers TRUE geometric contact (IsaacSim's per-foot
        # ContactSensors, via _geometric_foot_contact) when the backend supports it, falling back
        # to the offset-point+margin approximation elsewhere (MuJoCo, IsaacGym) -- see
        # _geometric_foot_contact's own docstring. kick_foot_index/kick_foot_is_left were already
        # refreshed above, so both paths read this env's correct (left/right, per-skill) foot.
        geometric_contact = _geometric_foot_contact(
            self.kick_foot_is_left,
            env.simulator.get_ball_foot_contact_pos_w("left"),
            env.simulator.get_ball_foot_contact_pos_w("right"),
        )
        if geometric_contact is not None:
            foot_contact = geometric_contact
        else:
            foot_to_ball_dist = torch.norm(
                self.kick_foot_contact_pos_w(env, _KICK_DETECT_FOOT_CONTACT_OFFSET_M) - self.ball_pos_w, dim=-1
            )
            foot_contact = _is_foot_contact(foot_to_ball_dist, self.ball_radius)
        in_strike_phase = _is_in_strike_phase(env)
        self.foot_contact_ever |= foot_contact & in_strike_phase
        displacement = torch.norm(self.ball_pos_w[:, :2] - motion_command.ball_spawn_pos_w[:, :2], dim=-1)
        self.has_kicked |= _detect_kick(
            foot_contact=foot_contact,
            foot_contact_ever=self.foot_contact_ever,
            displacement=displacement,
            ball_speed=self.ball_speed,
            in_strike_phase=in_strike_phase,
        )

        # Per-attempt outcome latches (only meaningful once kicked — before that, the standing
        # ball's distance to target is a property of the spawn draw, not of the policy).
        target_dist = torch.norm(self.ball_to_target_xy, dim=-1)
        self.min_target_dist = torch.where(
            self.has_kicked, torch.minimum(self.min_target_dist, target_dist), self.min_target_dist
        )
        self.max_ball_speed = torch.where(
            self.has_kicked, torch.maximum(self.max_ball_speed, self.ball_speed), self.max_ball_speed
        )

        # One-shot success burst per attempt: fires when the (latched) closest approach crosses
        # success_radius, pays out for burst_steps consecutive steps, never re-arms this attempt.
        newly_succeeded = self.has_kicked & ~self.success_latched & (self.min_target_dist <= self.success_radius)
        self.success_latched |= newly_succeeded
        self.burst_remaining[newly_succeeded] = burst_steps
        self.burst_active = self.burst_remaining > 0
        self.burst_remaining = torch.clamp(self.burst_remaining - 1, min=0)

    def kick_foot_contact_pos_w(self, env: Any, contact_offset_m: float) -> torch.Tensor:
        """The kick foot's actual contact point: ankle body origin + contact_offset_m along the
        foot's own local +x (toe direction). The ankle-roll link's origin sits at the joint, well
        behind/above the surface that strikes the ball — using it raw undercounts proximity by
        ~9cm (measured in this project's kick-region sweep; see that script's
        --foot-contact-offset-m).

        self.kick_foot_index is now PER-ENV (different envs' assigned skills can use different
        feet) -- `[self._env_arange, self.kick_foot_index]` gathers exactly one body per env
        (advanced indexing on both dims), NOT `[:, self.kick_foot_index]`, which would instead
        select ALL of kick_foot_index's entries for EVERY row (wrong shape, wrong values) once
        kick_foot_index stopped being a single scalar shared by every env."""
        foot_pos = env.simulator._rigid_body_pos[self._env_arange, self.kick_foot_index]
        foot_quat = env.simulator._rigid_body_rot[self._env_arange, self.kick_foot_index]  # xyzw
        offset_local = torch.zeros_like(foot_pos)
        offset_local[:, 0] = contact_offset_m
        return foot_pos + quat_apply(foot_quat, offset_local, w_last=True)

    def kick_foot_vel_xy(self, env: Any) -> torch.Tensor:
        return env.simulator._rigid_body_vel[self._env_arange, self.kick_foot_index, :2]

    def kick_foot_quat_w(self, env: Any) -> torch.Tensor:
        """The kick foot body's own world-frame orientation (xyzw), per-env gathered exactly like
        kick_foot_contact_pos_w's foot_quat -- factored out so foot_strike_pitch (below) doesn't
        duplicate this indexing."""
        return env.simulator._rigid_body_rot[self._env_arange, self.kick_foot_index]

    def kick_foot_reference_quat_w(self, env: Any) -> torch.Tensor:
        """The kick foot's REFERENCE (authored-clip) world-frame orientation (xyzw) at the env's
        current motion timestep -- 2026-08-09, feeds foot_strike_pitch's reference_relative mode.

        Reuses MotionCommand.body_quat_w (managers/command/terms/wbt.py:1413), which is already
        ``motion.body_quat_w[time_steps][:, tracked_body_indexes]`` -- the correctly-time-indexed,
        correctly-tracked-body-reduced reference orientation, the SAME source
        MotionRelativeBodyOrientationErrorExp tracks against. ``tracked_foot_col`` (resolved in
        __init__/update() above, mirroring kick_foot_index but indexing into
        motion_cfg.body_names_to_track instead of the simulator's full body list) selects this
        env's kick-foot column within that already-reduced tensor -- per-env, NOT
        ``self.kick_foot_index``, which is a DIFFERENT index space (env.body_names / raw rigid-body
        order) that would silently misindex here.

        Do NOT index ``motion_command.motion.body_quat_w`` (the ~32-entry raw robot-body-ordered
        tensor) directly with ``tracked_foot_col`` -- that column is only valid against the
        ~14-entry ``body_names_to_track`` ordering. A live-verification script this session
        (2026-08-08) made exactly that mistake and it silently produced a WRONG body's orientation
        with a plausible-looking, non-crashing shape (fixed there via
        ``tracked_body_indexes[tracked_foot_col]`` to remap into the 32-entry space first). Going
        through ``MotionCommand.body_quat_w`` here sidesteps the whole class of bug: that property
        already applies the ``[:, tracked_body_indexes]`` reduction, so ``tracked_foot_col`` is the
        correct index space for it directly, with no manual remap to get wrong."""
        motion_command = env.command_manager.get_state("motion_command")
        return motion_command.body_quat_w[self._env_arange, self.tracked_foot_col]

    def ball_pos_heading_frame_xy(self, env: Any) -> torch.Tensor:
        """[num_envs, 2] planar ball position relative to the robot root, in the robot's HEADING
        (yaw-only) frame. Deliberately the SAME transform as observation/terms/unified.py's
        ``_ball_pos_b_raw`` -- reward and observation must agree on the frame, or
        ball_approach_stance would be steering toward a target expressed differently from the
        ``kick_ball_pos_b`` the policy actually reads.

        Reads the tracker's own already-snapshotted ``ball_pos_w`` (refreshed once per control
        step in update()) rather than re-fetching root states, so it can't drift a tick out of
        sync with the rest of this tracker's per-step state. Yaw-only (not the full base quat) so
        a stationary ball doesn't appear to move as the torso pitches/rolls through the swing --
        same rationale as that observation's own docstring."""
        rel = self.ball_pos_w - env.simulator.robot_root_states[:, :3]
        return quat_rotate_inverse(yaw_quat(env.base_quat, w_last=True), rel, w_last=True)[:, :2]


def _tracker(env: Any, burst_steps: int = 10) -> _ShotTracker:
    tracker = getattr(env, "_shot_tracker", None)
    if tracker is None:
        tracker = _ShotTracker(env)
        env._shot_tracker = tracker
    tracker.update(env, burst_steps)
    return tracker


def _strike_phase_multiplier(env: Any) -> torch.Tensor:
    """1.0 while in_strike_phase (swinging mode only, mode 2), 0.0 outside it -- gates the 3
    contact-mechanics shooting terms (ball_proximity, contact_orientation, ball_velocity -- the
    last of which can OPT OUT to _post_locomotion_multiplier via its own use_post_locomotion_gate
    param, off by default) to the strike itself (excludes both the locomotion-approach mode 1 and
    post-kick-standing mode 3 / recovery / hold). NOT applied to the 3 outcome terms
    (error_ball_to_target, predicted_error_ball_to_target, goal_success_burst) -- those use the wider
    _post_locomotion_multiplier below instead, see this module's own docstring for why. Legacy/
    single-clip mode (no scrubbed strike/stand frames configured): in_strike_phase collapses onto
    in_kicking_phase's full window (see MotionCommand.setup()), so this is a no-op there, matching
    every other new gate in this project's convention. Float wrapper around _is_in_strike_phase --
    same underlying boolean _ShotTracker.update() (2026-08-01) now also uses to gate has_kicked's
    own latch TRIGGER (see _detect_kick's docstring)."""
    return _is_in_strike_phase(env).float()


def _post_locomotion_multiplier(env: Any) -> torch.Tensor:
    """1.0 once time_steps >= strike_start_idx (in_strike_phase itself, PLUS everything after --
    post-kick-standing, recovery, hold), 0.0 during locomotion-approach. Gates the 3 outcome terms
    (error_ball_to_target, predicted_error_ball_to_target, goal_success_burst) -- not to narrow
    their window back down to the strike alone (that already failed once, see this module's own
    docstring), but specifically to exclude locomotion-approach. Float wrapper around
    _is_post_locomotion. Until 2026-08-01 this same boolean also gated has_kicked's own latch
    TRIGGER in _ShotTracker.update() -- that was narrowed to the strictly-narrower
    _is_in_strike_phase instead (see _detect_kick's docstring for why), so this function is no
    longer shared with has_kicked's gating; it stays here, independent, because these 3 outcome
    terms' own PAYOUT window is deliberately wider than has_kicked's TRIGGER window (a shot's
    roll-to-target time routinely exceeds the ~1s strike window, so the reward needs to keep
    paying out well past stand_start_idx even though has_kicked itself can no longer newly
    trigger there). Also independently load-bearing for predicted_error_ball_to_target
    specifically, whose own gate (`moving`, i.e. ball_speed > v_min) is independent of has_kicked
    entirely -- a residual-momentum ball from the previous cycle's kick could still be `moving`
    during THIS attempt's locomotion-approach even though has_kicked itself can never flip true
    that early. Legacy/single-clip mode (no scrubbed strike/stand frames configured):
    strike_start_idx collapses onto motion_start_idx (see MotionCommand.setup()), so
    time_steps >= strike_start_idx is true from the first tick onward -- a verified no-op,
    matching every other new gate in this project's convention."""
    return _is_post_locomotion(env).float()


def _locomotion_approach_multiplier(env: Any) -> torch.Tensor:
    """1.0 during locomotion-approach only (time_steps < strike_start_idx), 0.0 from the strike
    onward -- gates ball_approach_stance, the only term in this module that pays out in the phase
    every other term is excluded from. Float wrapper around _is_locomotion_approach; see that
    function for the legacy-mode caveat (all-False, i.e. the term is inert there, unlike the other
    two gates which go all-True)."""
    return _is_locomotion_approach(env).float()


def _ood_gate_multiplier(env: Any) -> torch.Tensor:
    """1.0 for a normal (in-distribution) ball spawn, 0.0 for an attempt whose ball spawn was
    OOD-drawn (``MotionCommand.is_ood_spawn``, set once per reset/clip-replay by
    ``draw_position_noise_with_ood`` -- see ``MotionConfig.ood_spawn_probability``'s own
    docstring). Multiplied into ALL 6 shooting reward terms below (unlike the two phase gates
    above, which split 3/3) -- OOD-spawn is a per-ATTEMPT property, not a phase, orthogonal to
    which of the two phase gates a given term also carries.

    2026-08-01 reversal of the original design (2026-07-24): OOD-spawn episodes were deliberately
    left un-gated on the theory that ball_proximity's exp(-dist/sigma) and the has_kicked-gated
    terms already decay to ~zero for an unreachable ball. Revisited after realizing
    ood_region_multiplier's own draw is NOT rejection-sampled away from the normal box (see that
    field's own docstring) -- a plain uniform draw over the wider OOD region can coincidentally
    land back inside it, so a minority of "OOD" attempts were still landing at a reachable
    distance and paying inconsistent partial reward, exactly the kind of noise a reward signal
    shouldn't carry. Explicitly zeroing all 6 terms removes that noise regardless of where the
    draw happened to land, at the cost of also zeroing the (correctly reachable) minority --
    judged the simpler, more consistent tradeoff. is_ood_spawn is all-False forever whenever
    ood_spawn_probability<=0.0 (the default), making this a verified no-op for every config that
    doesn't opt in."""
    motion_command = env.command_manager.get_state("motion_command")
    return (~motion_command.is_ood_spawn).float()


def ball_approach_stance(env: Any, sigma: float = 0.9) -> torch.Tensor:
    """Locomotion-approach shaping (2026-08-01): exp(-||ball_b - ideal_strike_stance_b|| / sigma),
    active ONLY while time_steps < strike_start_idx. Rewards the robot for walking itself into the
    stance the authored swing expects RELATIVE TO WHERE THE BALL ACTUALLY IS -- not for getting
    close to the ball.

    Why this exists. Every other term in this module is zero during locomotion-approach, so until
    now the approach was driven purely by the WBT motion-tracking terms, which follow the clip's
    authored path and are completely blind to the ball. But the ball spawn IS randomized
    (SkillConfig.randomize_x/randomize_y, +/-0.35m each in this project's production config, a
    0.7x0.7m box) while the reference trajectory is deliberately NOT re-derived to match it (see
    MotionCommand.reset()'s own comment: ball placement "does NOT touch root_quat_w / the
    reference trajectory"). So the clip walks the robot to a stance authored for a NOMINAL ball
    while the real one can be ~0.5m away, and -- since the policy does observe the ball the whole
    time via kick_ball_pos_b -- the entire correction was being deferred into the ~0.7s strike
    window. configs/stageC_2skills.yaml's own bad_tracking_swing_threshold_multiplier note records
    the measured consequence: swing-phase tracking-deviation terminations ("reaching an off-nominal
    ball legitimately requires diverging from a clip that doesn't know where the ball is") are the
    LARGER termination source, 169 of 244 sampled vs 75 post-swing. This term moves that correction
    into the ~3.4s approach, where there is time to make it by walking.

    Why a STANCE-ERROR kernel rather than plain foot/base-to-ball distance shaping (considered and
    rejected):
    - It has a genuine OPTIMUM at the correct standoff instead of being monotone in "closer" --
      approaching nearer than the authored stance INCREASES the error and reduces reward. A
      monotone distance reward would pull the robot into the ball during the walk-up, which is
      precisely the locomotion-approach graze that _detect_kick's foot_contact_ever gate and the
      strike-only trigger window were added to eliminate; re-introducing it on the reward side
      would have fought both.
    - It is INERT when the ball is nominal. The clip's own path already ends at ~zero stance error,
      so the term sits near its maximum with a vanishing gradient and exerts no pull away from
      motion tracking; it only develops a meaningful gradient when the ball is actually off-nominal,
      which is exactly when overriding the clip is the correct behavior.

    sigma defaults to 0.9m, NOT ball_proximity's 0.35m: stance error starts around 1.1m at the top
    of the approach (nominal ball ~1.39m ahead of the pelvis at clip start vs ~0.2-0.3m at the
    strike stance), and at 0.35m both exp(-e/sigma) AND its derivative are ~0 across most of that
    range -- the same saturation failure configs/stageC_2skills.yaml's balance_potential_weight
    note documents for the motion-tracking terms ("realize only ~19% of their configured weight...
    both the term and its derivative are ~0"). 0.9 keeps real gradient over the whole approach.

    Carries current_w_g and the OOD gate like every other term here. Uses the PELVIS heading frame,
    not the kick foot: during the walk-up the swing foot is cycling through the gait, so a
    foot-referenced target would reward gait-phase artifacts rather than body positioning."""
    t = _tracker(env)
    stance_error = torch.norm(t.ball_pos_heading_frame_xy(env) - t.strike_stance_ball_b, dim=-1)
    return (
        torch.exp(-stance_error / sigma)
        * current_w_g(env)
        * _locomotion_approach_multiplier(env)
        * _ood_gate_multiplier(env)
    )


def ball_proximity(env: Any, sigma: float = 0.35, contact_offset_m: float = 0.09) -> torch.Tensor:
    """Dense kick-foot-to-ball approach shaping: exp(-[d - r_ball]+^2 / sigma^2), where d is the
    distance from the foot's contact point to the ball center and r_ball the ball radius (so the
    reward saturates at genuine touch, not at impossible interpenetration). RoboNaldo's
    robot_ball_contact analog — this is the term that gives gradient on every episode, contact or
    miss, and prevents the sparse-contact bootstrap problem.

    GAUSSIAN, not Laplacian (2026-08-01, changed from the original exp(-d/sigma)): every
    motion-tracking term this competes with during the strike (motion_relative_body_position_error
    _exp et al., managers/reward/terms/wbt.py) is Gaussian, exp(-e^2/sigma^2) -- and a Gaussian's
    gradient collapses far faster than a Laplacian's at large error. Live gradient arithmetic at
    this project's real configured weights (tracking 2.0 x motion_tracking_reward_scale 2.0 = 4.0;
    this term 2.0 x shooting_reward_scale w_g 1.2 = 2.4) found the crossover where this term's
    gradient overtook tracking's sat at only ~0.55-0.6m of error, and by 1.0m tracking's gradient
    had fallen to 0.0013 against this term's 0.54 (~400x) -- i.e. tracking's OWN gradient is what
    vanishes at the errors a randomized ball produces, not this term being disproportionately
    strong. Live-measured stance error at strike_start (checkpoint 325k skill 1, via
    ball_approach_stance's verification run) has median 0.58m, mean 0.78m, p90 1.48m -- the median
    attempt was already entering the strike past the crossover. Matching this term's kernel shape
    to tracking's (rather than raising its weight, which live measurement showed would need a
    ~400x increase at 1.0m to compete -- clearly disproportionate) removes the asymmetry at the
    source: both kernels now saturate together, so neither dominates the other's dead zone.

    sigma stays 0.35 (the pre-existing value, UNCHANGED number) -- isolates the shape change from
    a magnitude retune, one variable at a time, matching this project's own convention for every
    prior reward change this session.

    Additionally gated to ``~has_kicked`` (2026-08-04): zero from the tick contact is first
    confirmed, not just outside ``in_strike_phase``. The authored clip's own follow-through moves
    the kick foot away from the ball for most of the remaining strike window, so leaving this
    ungated past contact rewards NOT completing the swing (parking the foot near the ball) over
    following through -- see the module docstring's 2026-08-04 paragraph for the measured
    per-frame decay and the full rationale. ``ball_contact_hit`` (weight 10.0) already covers
    confirmed-contact credit from this same tick onward."""
    t = _tracker(env)
    foot_pos = t.kick_foot_contact_pos_w(env, contact_offset_m)
    dist = torch.clamp(torch.norm(foot_pos - t.ball_pos_w, dim=-1) - t.ball_radius, min=0.0)
    return (
        torch.exp(-torch.square(dist) / sigma**2)
        * current_w_g(env)
        * _strike_phase_multiplier(env)
        * (~t.has_kicked).float()
        * _ood_gate_multiplier(env)
    )


def contact_orientation(env: Any, sigma: float = 0.35, contact_offset_m: float = 0.09) -> torch.Tensor:
    """Aiming-at-contact: alignment (clamped cosine) of the kick foot's planar velocity with the
    ball->target direction, gated by foot-ball proximity so it only matters in the strike window.
    RoboNaldo's contact_orientation ("ankle velocity vs. ball->target") — this is the term that
    acts at the exact instant the shot direction is physically determined.

    Additionally gated to ``~has_kicked`` (2026-08-04, same change and rationale as
    ``ball_proximity`` immediately above -- see that function's docstring and the module
    docstring's 2026-08-04 paragraph): past contact, "foot velocity aimed at the target" no longer
    measures aiming the shot, it measures continuing to push the ball toward the target after
    it's already moving -- i.e. rewards dribbling, not a clean strike. ``ball_velocity`` and
    ``error_ball_to_target``/``predicted_error_ball_to_target`` already shape post-contact outcome;
    this term's job ends at the moment of contact."""
    t = _tracker(env)
    foot_vel = t.kick_foot_vel_xy(env)
    foot_speed = torch.norm(foot_vel, dim=-1)
    to_target = t.ball_to_target_xy / torch.norm(t.ball_to_target_xy, dim=-1, keepdim=True).clamp(min=1e-6)
    alignment = torch.clamp(
        torch.sum(foot_vel * to_target, dim=-1) / foot_speed.clamp(min=1e-6), min=0.0
    )
    foot_pos = t.kick_foot_contact_pos_w(env, contact_offset_m)
    dist = torch.norm(foot_pos - t.ball_pos_w, dim=-1)
    proximity_gate = torch.exp(-torch.clamp(dist - t.ball_radius, min=0.0) / sigma)
    return (
        alignment
        * proximity_gate
        * current_w_g(env)
        * _strike_phase_multiplier(env)
        * (~t.has_kicked).float()
        * _ood_gate_multiplier(env)
    )


def foot_strike_pitch(
    env: Any, sigma: float = 0.35, contact_offset_m: float = 0.09, reference_relative: bool = False
) -> torch.Tensor:
    """Shape the kick foot's ANKLE PITCH near contact: reward toe-down (plantarflexion, the
    top/instep of the foot presents to the ball) over toe-up (dorsiflexion, the flat sole presents
    to the ball instead) -- 2026-08-07, user-requested after observing weak, sole-first contacts in
    rollout.

    Reads the kick foot's local +x (toe-forward, same axis kick_foot_contact_pos_w's contact_offset
    already uses) rotated into world frame; the NEGATIVE of its vertical (Z) component is the
    reward: negative Z (toe pointing down) -> positive reward, positive Z (toe pointing up) ->
    negative (this term self-penalizes on the bad side, no separate penalty term needed):

        toe_dir_world = R(kick_foot_quat) @ [1, 0, 0]
        reward        = -toe_dir_world.z

    Deliberately NOT decomposed via a full roll/pitch/yaw Euler extraction, and deliberately NOT
    measured against a cached "neutral standing" reference pose (no forward-kinematics utility
    exists in this codebase to produce one, and the ankle-roll DOF's own joint axis makes one
    unnecessary here regardless):

    - The G1 URDF's ankle_roll_joint axis is exactly the same local axis as toe-forward ([1, 0, 0]
      in ankle_roll_link's own frame, confirmed against the sole-collision geometry, both matching
      kick_foot_contact_pos_w's "local +x = toe direction" docstring). A rotation about an axis
      never moves points that lie ON that axis, so toe_dir_world is mathematically INVARIANT to the
      ankle-roll DOF -- roll/inversion never contaminates this reading, with no explicit filtering
      needed. What's left is exactly the pitch contribution of the rest of the kinematic chain
      (hip, knee, ankle-pitch) baked into the foot body's actual world orientation -- the physically
      correct quantity, since it's the link's real pose at contact that determines which surface
      strikes the ball, not any single joint's own DOF reading in isolation.
    - "Straight" (the zero-reward point) is simply world-horizontal (toe_dir_world.z == 0), not a
      per-robot/per-episode latched reference: G1 stands and walks with the foot flat, so world
      level already coincides with the neutral pose for free.

    Gated to strike phase + foot-ball proximity (reused from contact_orientation's own
    contact_offset_m/sigma), but deliberately NOT to ``~has_kicked`` (2026-08-07, differs from
    ball_proximity/contact_orientation immediately above on this one point -- user-specified after
    weighing the tradeoff): those two are gated off at contact because their OWN reward would
    otherwise pay for lingering near the ball / continuing to push it post-contact ("dribbling",
    see their own docstrings) -- proximity there is the payoff, not just a gate. Here proximity is
    only ever the GATE; the payoff is ankle ORIENTATION, which a clean kick has no reason to snap
    away from the instant contact registers -- a real follow-through keeps a similar plantarflexed
    pose for a beat afterward, so shaping it through the rest of the strike window (still bounded
    by _strike_phase_multiplier, ending at stand_start_idx same as before) rewards that rather than
    fighting it. The proximity_gate itself still decays as the foot naturally moves away during
    follow-through, so this doesn't newly incentivize parking near the ball -- and even if it did,
    ball_proximity/contact_orientation (the larger-weighted terms) are still cut off at contact
    regardless, so the dominant "stay close" incentive is unaffected.

    ``reference_relative`` (2026-08-09, opt-in via ``MultiSkillConfig.use_foot_strike_pitch_
    reference_relative`` / ``BallConfig`` legacy counterpart -- see that field's docstring for the
    full motivation): False (default) keeps the ABSOLUTE reward above unchanged -- toe-down is
    rewarded regardless of what the authored clip itself does at that instant. That is a genuine
    conflict at frames where the clip's own reference pitch is still toe-up despite
    kick_ankle_pitch_correction's best effort (the correction is joint-limit-bounded and cannot
    always reach its target -- see kick_ankle_pitch_correction.py): motion-tracking pulls the
    policy toward the reference's own (still toe-up) pose at exactly the same frames this term
    pushes it away from, at zero net benefit to the policy since perfectly reproducing the
    reference already scores this term negatively regardless.

    When True, the reward is instead the robot's pitch_signal MINUS the reference clip's own
    pitch_signal at the same tracked timestep (kick_foot_reference_quat_w, same roll-invariant
    toe-direction kernel applied to the reference's stored orientation, not a re-derived
    approximation):

        reward = pitch_signal(robot) - pitch_signal(reference)

    Telescoping consequence: perfectly reproducing the reference (whatever its own pitch is, good
    or joint-limited-bad) now scores exactly 0 -- no more double-penalty against tracking at the
    frames tracking can't win. Doing WORSE than the reference (more toe-up than the clip itself)
    is still penalized; doing BETTER (toe-down beyond what the reference achieves, e.g. at frames
    kick_ankle_pitch_correction only partially corrected) is still rewarded -- the genuine signal
    survives, only the reference-achievable baseline shifts from a fixed absolute target to
    whatever this specific frame's clip actually supplies."""
    t = _tracker(env)
    foot_quat = t.kick_foot_quat_w(env)
    toe_dir_local = torch.zeros(foot_quat.shape[0], 3, device=foot_quat.device, dtype=foot_quat.dtype)
    toe_dir_local[:, 0] = 1.0
    toe_dir_world = quat_apply(foot_quat, toe_dir_local, w_last=True)
    pitch_signal = -toe_dir_world[:, 2]

    # 2026-08-15, "simultaneous per-skill task configs": reference_relative may now be a per-env
    # [num_envs] tensor (RewardManager's params_per_skill gather, see this term's own registration
    # in config_values/unified/g1/reward.py) instead of the plain bool it always used to be --
    # torch.where selects PER-ENV rather than branching once for the whole batch. A plain
    # True/False still takes the original scalar `if` path, byte-identical to before this existed.
    if torch.is_tensor(reference_relative) or reference_relative:
        ref_quat = t.kick_foot_reference_quat_w(env)
        ref_toe_dir_world = quat_apply(ref_quat, toe_dir_local, w_last=True)
        pitch_signal_relative = pitch_signal - (-ref_toe_dir_world[:, 2])
        if torch.is_tensor(reference_relative):
            pitch_signal = torch.where(reference_relative > 0.5, pitch_signal_relative, pitch_signal)
        else:
            pitch_signal = pitch_signal_relative

    foot_pos = t.kick_foot_contact_pos_w(env, contact_offset_m)
    dist = torch.norm(foot_pos - t.ball_pos_w, dim=-1)
    proximity_gate = torch.exp(-torch.clamp(dist - t.ball_radius, min=0.0) / sigma)

    return (
        pitch_signal
        * proximity_gate
        * current_w_g(env)
        * _strike_phase_multiplier(env)
        * _ood_gate_multiplier(env)
    )


def ball_velocity(
    env: Any,
    v_ref: float = 5.0,
    use_latched_peak_speed: bool = False,
    use_post_locomotion_gate: bool = False,
) -> torch.Tensor:
    """Shot power: saturating Lorentzian on the ball's planar speed, active only post-contact --
    s^2 / (s^2 + v_ref^2), reaching 0.5 at v_ref and asymptoting to 1. Prevents the policy from
    satisfying the target rewards with a weak tap from close range (RoboNaldo's ball_velocity).

    DEFAULTS ARE THE ORIGINAL, PRE-2026-08-21 BEHAVIOR, bit-identical: instantaneous
    ``t.ball_speed``, ``_strike_phase_multiplier``, ``v_ref=5.0``. The three parameters below are
    opt-in retunes, each independently togglable from the task-config yaml
    (``kick_ball_velocity_v_ref`` / ``_use_latched_peak_speed`` / ``_use_post_locomotion_gate`` --
    see MultiSkillConfig's own docstrings), following this file's standard "new behavior ships
    OFF, an unedited config reproduces today's training exactly" discipline.

    THE MEASUREMENT behind all three (run 20260820_130329, 350k steps): hit rate was flat
    0.160->0.182 while ball speed REGRESSED 2.06->1.69 m/s, with corr(ball_velocity, alive_frac) =
    -0.279 -- the policy measurably traded shot power for survival. Root cause: a proper 5 m/s
    strike earned only 1.29x what a weak 1.6 m/s graze earned, so declining to strike was the
    rationally correct policy (break-even was a mere +17.7% added fall probability, against a
    measured 11% topple / 36% early-termination rate).

    ``use_latched_peak_speed`` -- read ``t.max_ball_speed`` (this attempt's latched peak) instead
    of ``t.ball_speed`` (instantaneous). Mirrors ``error_ball_to_target``'s own use of the latched
    ``min_target_dist``; the peak already existed in ``_ShotTracker`` but was read only by the
    logging path (envs/unified/unified_manager.py's Kick_skills_N/kick_ball_velocity metric),
    never by a reward. STRONGLY RECOMMENDED whenever ``use_post_locomotion_gate`` is also on:
    paying INSTANTANEOUS speed over the long window would reward a ball that merely keeps rolling
    and would decay to ~0 as it decelerates.

    ``use_post_locomotion_gate`` -- pay over ``_post_locomotion_multiplier`` (325 ticks) instead
    of ``_strike_phase_multiplier`` (67 ticks). The default pairing is structurally lopsided:
    ``ball_contact_hit`` (flat 1.0 for ANY confirmed touch) pays over the wider window while this
    term -- the only one demanding POWER -- pays over the narrow one, a 4.9x asymmetry that lets
    "just make contact" collect ~99.5% of the available shooting reward.

    ``v_ref`` -- 5.0 (default) vs 2.0. At the measured 1.6 m/s operating point, 5.0 sits deep in
    the Lorentzian's flat tail: value 0.093, gradient 0.105 per m/s. 2.0 gives 0.390 and 0.297 --
    2.8x steeper exactly where the policy lives, while still asymptoting to 1 so a genuinely hard
    strike keeps being rewarded more.

    Together (with ball_contact_hit's weight cut 15->5 and this term's raised 20->40) these move
    the strike-vs-graze differential from +1.16 to +6.76 episode-reward, i.e. a break-even fall
    probability of ~103% instead of ~17.7%.

    VALIDATION STATUS -- read before enabling: these shipped ON in runs 20260820_235327 /
    20260821_014825 / 20260821_022846, but their individual contribution is CONFOUNDED. The first
    run carrying them (235327) was WORSE than baseline on hit rate; what actually improved things
    later was ``start_at_timestep_zero_prob`` 0.5 -> 1.0, a separate change. The one clean
    measurement is that per-hit payout rose 1.284 -> 1.502 (+17%) with behavior held identical --
    an incentive measurement, not an outcome one. Treat as a reasoned, still-unvalidated A/B, and
    guard it with the MuJoCo sim2sim survival scan (train.log, "kick_fall_rate"), NOT with
    kick_ball_hit_rate alone -- that metric latches has_kicked once per attempt and is blind to
    the robot falling over afterwards (see task_config_stageC1-skill013.yaml's critic-support
    comment for the run where exactly that misled us)."""
    t = _tracker(env)
    speed = t.max_ball_speed if use_latched_peak_speed else t.ball_speed
    s2 = torch.square(speed)
    phase_gate = _post_locomotion_multiplier(env) if use_post_locomotion_gate else _strike_phase_multiplier(env)
    return (
        t.has_kicked.float()
        * s2
        / (s2 + v_ref**2)
        * current_w_g(env)
        * phase_gate
        * _ood_gate_multiplier(env)
    )


def ball_contact_hit(env: Any) -> torch.Tensor:
    """Persistent per-tick credit for having landed a genuine kick (2026-08-01): flat 1.0 every
    tick from the moment has_kicked first latches True through the rest of the attempt (recovery/
    hold included), not just an instantaneous bonus. NOT one of RoboNaldo's Table B.1 terms —
    added on top of them because has_kicked is, as of this session's earlier work, already a
    HIGH-PRECISION signal (geometric PhysX per-foot contact sensor on IsaacSim, see
    _geometric_foot_contact/BaseSimulator.get_ball_foot_contact_pos_w -- confirmed via a live
    shadow comparison that the pre-sensor point+margin approximation over-counted contact by
    roughly 40% on cycles the sensor correctly rejected), so directly rewarding "has_kicked is
    True" is now rewarding a confirmed strike, not a proxy that could be gamed by a wrong-foot
    graze. Sticky/dense (every tick, not one-shot) so the credit accumulates in proportion to how
    much of the attempt is spent past a confirmed strike -- deliberately simple (no distance/speed
    shaping of its own) since ball_velocity/error_ball_to_target/predicted_error_ball_to_target
    already shape those; this term exists purely to reward the binary "did the kick land" event
    densely rather than only via goal_success_burst's much narrower, target-proximity-gated
    one-shot.

    Gated to _post_locomotion_multiplier, NOT the narrower _strike_phase_multiplier the 3 contact-
    mechanics terms use -- user-specified ("this can only happen in post locomotion"): once landed,
    credit should keep accruing through recovery/hold just like the other outcome terms
    (error_ball_to_target, predicted_error_ball_to_target, goal_success_burst), not cut off the
    instant stand_start_idx is reached."""
    t = _tracker(env)
    return t.has_kicked.float() * current_w_g(env) * _post_locomotion_multiplier(env) * _ood_gate_multiplier(env)


def error_ball_to_target(env: Any, sigma: float = 1.0) -> torch.Tensor:
    """Outcome accuracy: exp(-d_hat^2 / sigma^2) where d_hat is this attempt's latched minimum
    ball-to-target distance (closest approach so far). Active only post-contact — before the kick,
    the standing ball's distance is a property of the spawn draw, not the policy. RoboNaldo's
    error_ball_to_target (their d_hat = min_t ||p_b - p_t||).

    NOT gated to in_strike_phase (2026-07-31): d_hat is a LATCHED closest-approach-so-far, and the
    ball routinely keeps rolling toward the target well past stand_start_idx (a ~5m shot's travel
    time regularly exceeds the ~1s strike window) -- gating this to the strike would reward the
    wrong thing: whatever the ball's distance happened to be the instant the strike window closed,
    instead of how close the shot actually got. IS gated to _post_locomotion_multiplier (added same
    day) -- has_kicked alone isn't a precise enough proxy for "the kick attempt has begun" during
    locomotion-approach specifically; see this module's own docstring for the measurement.

    ANGULAR-TOLERANCE READING (2026-08-22, azimuth-aim refactor): for any kick_aim_enabled skill,
    the target is synthesized at a FIXED distance D (MultiSkillConfig/BallConfig.
    kick_aim_nominal_distance_m, default 5.0m) along the commanded ray -- so d_hat's DISTANCE
    reading is now approximately D*sin(angular_error) for small errors, and this sigma implicitly
    sets an ANGULAR tolerance of roughly asin(sigma/D) at half-value. sigma=1.0 at D=5.0 is
    asin(0.2)=~11.5 deg, in the same neighborhood as RoboNaldo's own reported ~10-14 deg hardware
    accuracy -- plausible, NOT independently re-validated against a live kick_aim_enabled training
    run. Configurable per this term's own `kick_error_ball_to_target_sigma` override (see
    MultiSkillConfig/BallConfig's own docstring) rather than changed here blind."""
    t = _tracker(env)
    return (
        t.has_kicked.float()
        * torch.exp(-torch.square(t.min_target_dist) / sigma**2)
        * current_w_g(env)
        * _post_locomotion_multiplier(env)
        * _ood_gate_multiplier(env)
    )


def predicted_error_ball_to_target(env: Any, sigma: float = 1.0, v_min: float = 0.25) -> torch.Tensor:
    """Densified shot-direction feedback (RoboNaldo Eq. 3, adapted to a ground-point target): at
    every step the ball is moving, extend its current planar velocity ray forward and reward
    exp(-d_pred^2 / sigma^2) on the predicted closest approach to the target. This is the
    credit-assignment workhorse — the policy gets immediate per-step feedback on where the shot is
    HEADING from the instant of contact, instead of waiting for the ball to finish its travel
    (by which point the strike that determined the outcome is many steps in the past).

    NOT gated to in_strike_phase (2026-07-31): the ball keeps rolling (and this term keeps giving
    useful directional feedback on it) well past stand_start_idx -- see error_ball_to_target's
    docstring for the same reasoning. `moving` (ball_speed > v_min) is this term's own natural
    temporal gate for "has the ball stopped" -- it already goes to zero once the ball rolls to a
    stop, regardless of phase. IS gated to _post_locomotion_multiplier (added same day): `moving`
    alone doesn't rule out the ball being incidentally nudged (not struck) during the locomotion
    approach, which would otherwise still register as "moving" and pay a small amount of
    undeserved directional-feedback reward; see this module's own docstring for the measurement.

    Additionally gated to has_kicked (2026-08-04): _post_locomotion_multiplier alone turned out to
    narrow, not close, the "incidentally nudged" gap above -- it only rules out locomotion-approach,
    but `moving` still has no requirement that the ASSIGNED KICK FOOT caused the motion, so a torso
    lean, swing-arm brush, or wrong-foot graze DURING the strike itself can set the ball rolling and
    pay this term with zero confirmed contact. Live-measured on a real checkpoint (491k, w_g=1.2,
    kick-eligible envs): of 119,611 post-locomotion-phase ticks, 15,298 (12.8%) were
    moving-without-has_kicked -- 62% as many ticks as the legitimate moving-with-has_kicked bucket
    (24,618), contributing an estimated ~38% as much total reward. Same category of problem this
    project already found and fixed once for has_kicked's OWN internal ball-motion fallback (see
    this module's docstring: an earlier, ball-motion-only version of that fallback was driving 79%
    of detections with zero geometric confirmation). has_kicked specifically requires the assigned
    kick foot's confirmed contact (direct geometric, or a proxy gated behind that same foot's own
    earlier contact this attempt), so it closes exactly the gap `moving` alone cannot. Costs nothing
    on the legitimate path: for any real strike, has_kicked latches at-or-before the ball's speed
    first crosses v_min (contact triggers the latch; the ball then needs a physics step to
    accelerate), so `has_kicked & moving` is identical to `moving` whenever the kick foot actually
    caused the motion."""
    t = _tracker(env)
    moving = t.ball_speed > v_min
    v = t.ball_vel_xy
    # Closest approach of the ray p + v*t (t >= 0) to the target: t* = <to_target, v> / |v|^2,
    # clamped to t* >= 0 — a ball moving AWAY from the target gets its current (worse) distance.
    t_star = torch.clamp(
        torch.sum(t.ball_to_target_xy * v, dim=-1) / torch.sum(v * v, dim=-1).clamp(min=1e-6), min=0.0
    )
    predicted_miss = torch.norm(v * t_star.unsqueeze(-1) - t.ball_to_target_xy, dim=-1)
    return (
        moving.float()
        * t.has_kicked.float()
        * torch.exp(-torch.square(predicted_miss) / sigma**2)
        * current_w_g(env)
        * _post_locomotion_multiplier(env)
        * _ood_gate_multiplier(env)
    )


def goal_success_burst(env: Any, burst_steps: int = 10) -> torch.Tensor:
    """One-shot success bonus: pays 1.0 for burst_steps consecutive steps the first time this
    attempt's closest approach crosses configs/ball.yaml's success_radius, then never re-arms
    within the attempt (RoboNaldo's goal_reward_burst — "constant for 10 steps on success").
    Registered with a large weight so a genuine hit unambiguously dominates the dense shaping.

    NOT gated to in_strike_phase (2026-07-31): this is the term that gate hurt most -- tied (with
    predicted_error_ball_to_target) for the largest of the 6 shooting weights (weight 10.0,
    configs/kicking_motion_reward_tuning.yaml; reward.py's own hardcoded 300.0 default is
    overridden by that yaml and was never the active value), yet a shot's closest approach
    routinely isn't reached until well after stand_start_idx, so the strike-gated version could
    never pay out in practice (confirmed live: exactly 0 across both a MuJoCo n=4 and an IsaacSim
    606-cycle sample). IS gated to _post_locomotion_multiplier (added same day) -- success_latched
    provides this term's own correct temporal scope once a real attempt is underway (False until
    has_kicked AND min_target_dist <= success_radius, never re-arms within the attempt, reset on
    the next new_attempt), but has_kicked itself can flip early from an incidental locomotion-phase
    ball graze; excluding locomotion specifically closes that without reintroducing the
    too-narrow-window problem the strike-only gate had.

    ANGULAR-TOLERANCE READING (2026-08-22, azimuth-aim refactor): for any kick_aim_enabled skill,
    success_radius (already yaml-configurable per-skill, SkillConfig/BallConfig.success_radius) is
    a lateral distance at the FIXED kick_aim_nominal_distance_m -- e.g. 0.5m at D=5.0m is an
    angular tolerance of asin(0.5/5.0)=~5.7 deg, noticeably TIGHTER than RoboNaldo's own reported
    ~10-14 deg hardware accuracy. Not changed here blind -- widen a specific skill's own
    success_radius in its yaml if that tightness turns out to suppress its success rate once
    trained; this is a per-skill knob already, no new config surface needed."""
    t = _tracker(env, burst_steps=burst_steps)
    return (
        t.burst_active.float() * current_w_g(env) * _post_locomotion_multiplier(env) * _ood_gate_multiplier(env)
    )


def _kicking_phase_multiplier(env: Any) -> torch.Tensor:
    """1.0 during in_kicking_phase (locomotion-approach + strike), 0.0 during post-kick recovery/
    hold -- 2026-08-05, new gate added specifically for robot_com_ball_distance/robot_torso_ball_
    distance (see those functions' own docstrings): unlike every OTHER gate in this module
    (_post_locomotion_multiplier excludes approach; _is_locomotion_approach excludes everything
    but approach), these two terms need to stay active THROUGH BOTH approach and strike -- exactly
    ``motion_command.in_kicking_phase``'s own boundary, RoboNaldo's own ``~stable_phase`` analog
    (see KickFeetAirTime's docstring for the general note on this project's stand_start_idx-as-
    critic_frame_index-analog simplification, which applies here identically)."""
    motion_command = env.command_manager.get_state("motion_command")
    return motion_command.in_kicking_phase.float()


def _weak_foot_index(env: Any, t: "_ShotTracker") -> torch.Tensor:
    """Per-env raw rigid-body index of the NON-kicking ankle -- the opposite of ``t.kick_foot_index``.
    Resolved fresh each call (both ankle_roll_link indices are a 2-entry lookup, matching this
    file's own established convention for cheap, non-cached per-call lookups -- see
    ``kick_foot_contact_pos_w``'s sibling functions for the pattern), then selected per-env via
    ``t.kick_foot_is_left`` (already tracked per-env for the ball-foot-contact-sensor gather --
    see ``_ShotTracker.__init__``'s own comment)."""
    left_idx = env.simulator.body_names.index("left_ankle_roll_link")
    right_idx = env.simulator.body_names.index("right_ankle_roll_link")
    return torch.where(
        t.kick_foot_is_left,
        torch.full_like(t.kick_foot_index, right_idx),
        torch.full_like(t.kick_foot_index, left_idx),
    )


def penalize_weak_foot_contact(env: Any, threshold: float = 0.12, std: float = 0.1) -> torch.Tensor:
    """Penalize the NON-kicking foot for getting close enough to contact the ball -- 2026-08-05,
    ported from RoboNaldo (arXiv:2606.11092)'s ``penalize_weak_foot_contact`` (mdp/rewards.py),
    "keeps the shot focused on the configured main foot" (their own docstring).

    Formula matches exactly: ``exp(-(dist - threshold)^2 / std^2)`` where ``dist`` is the weak
    foot's distance to the ball. NOTE (ported faithfully, not re-derived): this is a Gaussian BUMP
    centered AT ``threshold``, not a monotonic "closer = worse" penalty -- it peaks when the weak
    foot sits exactly at the critical near-contact distance and decays in EITHER direction (a weak
    foot far away OR already past/through the ball both score lower). Registered with a NEGATIVE
    weight (RoboNaldo's own convention, matching this project's sign convention throughout this
    port), so this is a genuine penalty peaking at the danger-zone distance, not a reward.

    UNGATED across the whole kick episode -- RoboNaldo's own registration carries no phase
    multiplier (weight -0.4/-0.4, S2a/S2b), matching ``kick_penalize_self_contact_feet`` below."""
    t = _tracker(env)
    weak_foot_index = _weak_foot_index(env, t)
    weak_foot_pos = env.simulator._rigid_body_pos[t._env_arange, weak_foot_index]
    dist = torch.norm(weak_foot_pos - t.ball_pos_w, dim=-1)
    return torch.exp(-torch.square(dist - threshold) / std**2) * current_w_g(env) * _ood_gate_multiplier(env)


def penalize_self_contact_feet(
    env: Any, threshold: float = 0.2, std: float = 0.05, foot_body_names: list[str] | None = None
) -> torch.Tensor:
    """Penalize the two feet for coming too close to each other -- 2026-08-05, ported from
    RoboNaldo (arXiv:2606.11092)'s ``penalize_self_contact_feet`` (mdp/rewards.py).

    Formula matches exactly: zero while ``left_right_dist >= threshold``; once closer,
    ``10.0 * (1 - exp(-(dist - threshold)^2 / std^2))`` -- grows toward 10.0 as the feet approach
    each other (dist -> 0), a genuine "closer = worse" ramp once inside the threshold zone (unlike
    ``penalize_weak_foot_contact`` above, whose bump shape is NOT monotonic -- these two RoboNaldo
    terms have deliberately different shapes, both ported as given, not reconciled to match).

    UNGATED across the whole kick episode -- RoboNaldo's own registration carries no phase
    multiplier (weight -0.16/-0.16, S2a/S2b)."""
    if foot_body_names is None:
        foot_body_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
    left_idx = env.simulator.body_names.index(foot_body_names[0])
    right_idx = env.simulator.body_names.index(foot_body_names[1])
    left_pos = env.simulator._rigid_body_pos[:, left_idx]
    right_pos = env.simulator._rigid_body_pos[:, right_idx]
    dist = torch.norm(left_pos - right_pos, dim=-1)
    close = dist < threshold
    reward = torch.zeros_like(dist)
    reward = torch.where(close, 10.0 * (1.0 - torch.exp(-torch.square(dist - threshold) / std**2)), reward)
    return reward * current_w_g(env) * _ood_gate_multiplier(env)


def robot_com_ball_distance(env: Any, std: float = 0.5) -> torch.Tensor:
    """Reward the robot's root (RoboNaldo's own "CoM" -- their ``robot.data.root_state_w``, a
    plain root/pelvis read, not a true mass-weighted center-of-mass) for staying close to the ball
    in the horizontal plane -- 2026-08-05, ported from RoboNaldo (arXiv:2606.11092)'s
    ``robot_com_ball_distance`` (mdp/rewards.py), "encourages the whole body to move into a useful
    striking position, not only the foot" (their own docstring).

    Formula matches exactly: ``clamped_dist = clamp(dist, min=0.25) - 0.25``, ``reward = 1/(1 +
    (clamped_dist/std)^2)`` -- saturates at 1.0 once within 0.25m, decays smoothly beyond it. Once
    contact has occurred this attempt (``has_kicked``, this project's own geometric analog of
    RoboNaldo's ``contacted_flag``), ``dist`` is PINNED to the 0.25m clamp floor (their own
    ``dist[contacted_flag] = 0.25``) -- ported exactly: an attempt that already struck the ball
    keeps this term saturated at its max regardless of where the robot moves afterward, rather
    than penalizing the natural follow-through/recovery motion moving the CoM away again.

    Gated to _kicking_phase_multiplier (approach + strike), NOT _post_locomotion_multiplier or
    _is_in_strike_phase -- see that gate's own docstring for why this term specifically needs the
    wider approach-inclusive window (RoboNaldo's own gate here is ``~stable_phase``, active
    through BOTH their approach and strike, zeroed only once "stable" -- the walk-up positioning
    this term rewards matters most BEFORE contact, not just during/after it)."""
    t = _tracker(env)
    com_pos = env.simulator.robot_root_states[:, :3]
    dist = torch.norm(com_pos[:, :2] - t.ball_pos_w[:, :2], dim=-1)
    dist = torch.where(t.has_kicked, torch.full_like(dist, 0.25), dist)
    clamped_dist = torch.clamp(dist, min=0.25) - 0.25
    reward = 1.0 / (1.0 + torch.square(clamped_dist / std))
    return reward * current_w_g(env) * _kicking_phase_multiplier(env) * _ood_gate_multiplier(env)


def robot_torso_ball_distance(env: Any, std: float = 0.5, body_names: list[str] | None = None) -> torch.Tensor:
    """Reward the torso for staying close to the ball in the horizontal plane -- 2026-08-05,
    ported from RoboNaldo (arXiv:2606.11092)'s ``robot_torso_ball_distance`` (mdp/rewards.py),
    "regularizes body placement around the ball and can reduce awkward reaches from the kicking
    leg" (their own docstring). Same formula/contact-latch/phase-gate as ``robot_com_ball_distance``
    immediately above, just torso position instead of root -- see that function's own docstring
    for the full rationale on each piece; 0.3m clamp floor here (vs 0.25m there), matching
    RoboNaldo's own distinct values for the two terms."""
    t = _tracker(env)
    if body_names is None:
        body_names = ["torso_link"]
    body_indexes = [env.simulator.body_names.index(name) for name in body_names]
    torso_pos = env.simulator._rigid_body_pos[:, body_indexes].mean(dim=1)
    dist = torch.norm(torso_pos[:, :2] - t.ball_pos_w[:, :2], dim=-1)
    dist = torch.where(t.has_kicked, torch.full_like(dist, 0.3), dist)
    clamped_dist = torch.clamp(dist, min=0.3) - 0.3
    reward = 1.0 / (1.0 + torch.square(clamped_dist / std))
    return reward * current_w_g(env) * _kicking_phase_multiplier(env) * _ood_gate_multiplier(env)


def ball_over_line(
    env: Any, over_line_dist: float = 7.0, back_line_dist: float = -1.0, require_has_kicked: bool = False
) -> torch.Tensor:
    """Reward the ball crossing well past the target, penalize it moving backward past the robot
    -- 2026-08-05, ported from RoboNaldo (arXiv:2606.11092)'s ``ball_over_line`` (mdp/rewards.py).

    ADAPTATION (geometric, not formula): RoboNaldo's own version compares the ball's WORLD-FRAME Y
    coordinate against two fixed absolute distances (a shared soccer-pitch layout every env sits
    on identically). This project has no shared field -- each env's ball spawn/target is its own
    per-skill, per-attempt randomized point (``MotionCommand.ball_spawn_pos_w``/``target_xy_w``),
    so there is no single fixed axis to test a raw world coordinate against. Adapted by projecting
    the ball's CURRENT displacement from its own spawn point onto that attempt's own spawn-to-
    target UNIT vector: ``over_line_dist``/``back_line_dist`` become distances measured ALONG that
    per-attempt axis instead of a shared world Y-axis, preserving RoboNaldo's own reward
    MAGNITUDES (``2*over_line - over_back_line``) exactly, and preserving the geometric INTENT
    (reward significant forward progress past the target, penalize the ball sliding backward past
    where the robot started) rather than the literal (inapplicable) coordinate test.

    UNGATED across the whole kick episode by default (``require_has_kicked=False``) -- RoboNaldo's
    own registration carries no phase multiplier (weight 0.4/0.4, S2a/S2b), and this project's own
    default preserves that exactly.

    ``require_has_kicked`` (2026-08-06, user-requested, opt-in): when True, multiplies by
    ``t.has_kicked.float()``, the same class of fix already landed (and live-measured) for
    ``predicted_error_ball_to_target`` above -- unlike that term's own gate, THIS one is reasoned
    from the mechanism, not yet independently live-measured for ``ball_over_line`` specifically, so
    it ships as an opt-in rather than landing unconditionally the way that one did. The risk this
    closes: ``projected`` is a DISPLACEMENT from spawn, not a per-tick reading, so unlike a
    memoryless term, a single accidental non-kick-foot contact (a torso lean, trailing-leg brush,
    or wrong-foot graze during locomotion-approach or the strike -- PhysX collision physics don't
    restrict ball contact to the assigned kick foot) can move the ball far enough to bias this
    term's reading for the REST of the attempt, not just one tick. The asymmetry is sharper on the
    ``back_line_dist`` (penalty) side than ``over_line_dist`` (reward): reaching 7.0m forward
    plausibly already implies a real strike happened (large ball motion this project's own
    ``_detect_kick`` ball-motion fallback would likely have already caught, given it also requires
    ``foot_contact_ever``), but ``back_line_dist``'s much smaller 1.0m threshold is easier to cross
    from an incidental early nudge with no foot contact at all yet. False (default) = off,
    bit-identical to before this parameter existed."""
    t = _tracker(env)
    motion_command = env.command_manager.get_state("motion_command")
    spawn_to_target = motion_command.target_xy_w - motion_command.ball_spawn_pos_w[:, :2]
    axis_len = torch.norm(spawn_to_target, dim=-1).clamp(min=1e-6)
    axis_unit = spawn_to_target / axis_len.unsqueeze(-1)
    displacement = t.ball_pos_w[:, :2] - motion_command.ball_spawn_pos_w[:, :2]
    projected = torch.sum(displacement * axis_unit, dim=-1)
    over_line = projected > over_line_dist
    over_back_line = projected < back_line_dist
    reward = 2.0 * over_line.float() - over_back_line.float()
    # 2026-08-15, "simultaneous per-skill task configs": require_has_kicked may now be a per-env
    # [num_envs] tensor (RewardManager's params_per_skill gather) instead of the plain bool it
    # always used to be -- torch.where selects PER-ENV. A plain True/False still takes the
    # original scalar path, byte-identical to before this existed.
    if torch.is_tensor(require_has_kicked):
        gate = torch.where(require_has_kicked > 0.5, t.has_kicked.float(), torch.ones_like(t.has_kicked, dtype=torch.float))
    else:
        gate = t.has_kicked.float() if require_has_kicked else 1.0
    return reward * gate * current_w_g(env) * _ood_gate_multiplier(env)
