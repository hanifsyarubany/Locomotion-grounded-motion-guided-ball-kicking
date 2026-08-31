"""Specialist -> generalist distillation via DAgger.

Trains ONE student actor to imitate N frozen "specialist" teacher actors simultaneously, routing
each rollout env to its teacher via that env's own `skill_id` -- so every training batch carries
supervision from ALL skills at once, by construction. This is the answer to "how do I add skill 2
without forgetting skill 1": checkpoint weight-space merging doesn't exist as an operation (there
is no op that takes skill1-weights + skill2-weights -> both-skills-weights), but distillation is a
real merge, and it's what Any2Track/UniTracker/PHC all do to combine per-motion(-cluster)
specialists into one generalist. See memory (search "distillation"/"specialist") for the full
design discussion this implements.

Why DAgger and not plain behavior cloning: the STUDENT drives the rollout (env.step gets the
student's own action), and teachers only ever LABEL the states the student actually visits. Plain
BC would instead roll out under the teachers' own trajectories, which the student never has to
recover from once it starts drifting off-distribution at deployment -- DAgger's on-policy labeling
is what prevents that compounding-error failure mode.

Two things that will silently produce a garbage-but-plausible-looking student if you get them
wrong (both handled below, see the Teacher class):
  1. Each teacher's `EmpiricalNormalization` has its OWN running mean/var, fit to whatever that
     specialist's own training distribution looked like -- pushing raw obs through the STUDENT's
     normalizer into a TEACHER actor produces confident, wrong actions with no error raised
     anywhere. Every teacher is paired with the normalizer state saved alongside it.
  2. `kick_ankle_pitch_correction_enabled` (and each skill's own strike/stand frames, spawn coords,
     task_config) must match what each teacher actually trained with, because the correction
     rewrites `motion_command` -- part of the observation -- so a mismatched setting is an
     out-of-distribution reference for that teacher, again with no error raised. This is set PER
     SKILL in the skills yaml this script consumes (see configs/skill/distill_skill1_skill2.yaml), not
     globally, so each teacher gets exactly what it was trained under.

Locomotion-mode envs have no meaningful skill_id -- `UnifiedManager._build_task_mode_partition`
already defaults `skill_id` to 0 for them (the draw==0 / locomotion case maps to
`(draw-1).clamp_min(0) == 0`), so they route to skill_id 0's teacher automatically. Override via
HOLOSOMA_DISTILL_LOCOMOTION_TEACHER_SKILL_ID if a different skill should own locomotion instead.

CLI grammar is identical to train_agent.py (same exp:X/logger:Y/--training.X/--algo.config.X
parsing). Teacher checkpoints go in the skills yaml itself, one `teacher_checkpoint:` field per
motion_skill_N block (see configs/skill/distill_skill1_skill2.yaml) -- this is the recommended
path, no long export line needed. `logger:disabled` (fast, no visualization -- used for smoke
tests) and `logger:wandb-ball-kick` (full wandb + tensorboard) are BOTH fully supported; only the
logger preset changes:

    export PYTHONPATH=.../src/holosoma:$PYTHONPATH
    export HOLOSOMA_SKILLS_CONFIG=configs/skill/distill_skill1_skill2.yaml
    python -m holosoma.distill_specialists exp:g1-29dof-unified-fast-sac logger:wandb-ball-kick \\
      --training.project UnifiedBallKickingEnhanced --training.name distill-skill1-skill2 \\
      --training.headless=True --training.num-envs 3300

What actually shows up in wandb, and why it's meaningful beyond the raw regression loss: this
reuses the SAME `LoggingHelper` (student_agent.logging_helper) every other training run in this
project uses -- "Loss/distill_mse" (the DAgger regression loss), "Perf/*" (throughput), AND,
because the student's OWN action genuinely drives env.step() every iteration (this is what makes
it DAgger rather than plain BC), the environment's normal per-skill diagnostics come along for
free as a side effect of stepping with real actions -- "Kick_skills_0/kick_topple_frac",
"..._episode_length", "..._ball_hit_rate", etc., per skill, the exact same keys/sections used
throughout this project's RL runs. This is a MORE meaningful signal than the loss alone: MSE
dropping tells you the student is matching teacher actions on states it visits, but not whether
ITS OWN resulting rollout is actually stable -- the topple/episode-length numbers answer that
directly, live, during distillation, before you ever evaluate the checkpoint separately.

Every checkpoint save (periodic in-loop AND the final one) also exports a `.onnx` alongside the
`.pt`, via FastSACAgent.export() (unmodified, reused as-is) -- read-only w.r.t. env/actor state
(dummy zero input for ONNX tracing, metadata pulled from command_manager/robot_config), verified to
never call env.reset_all()/env.step(), so it can't disrupt this loop's own rollout. Periodic saves
additionally trigger the SAME 4-rollout MuJoCo/RoboJuDo sim2sim bundle (kick + walk + kick-handoff,
N-skill-aware) every other training run in this project gets, reusing FastSACAgent's own
_maybe_start_mujoco_*_rollout/_drain_mujoco_*_rollout_queue methods unmodified -- same cadence gate
(--algo.config.mujoco-kick-rollout-every-n-saves), same pileup guard, same "mujoco_media/" wandb
key convention. Each spawns a background thread -> subprocess and never touches this process's own
CUDA/torch state, so it doesn't block training.

Everything else specific to THIS script (not teacher checkpoints) goes through env vars, mirroring
this project's own existing HOLOSOMA_* knob convention (HOLOSOMA_KICK_RECORDER_INTERVAL etc.)
rather than adding new fields to the shared ExperimentConfig schema -- all optional, shown here at
their defaults:

    export HOLOSOMA_DISTILL_STEPS=200000
    export HOLOSOMA_DISTILL_LR=3e-4
    export HOLOSOMA_DISTILL_SAVE_INTERVAL=5000

WHAT IS BEING REGRESSED (2026-08-18 -- this changed, read before comparing runs across that date)
-------------------------------------------------------------------------------------------------
The loss matches the quantity that is ACTUALLY DEPLOYED, which is not tanh(mu).

`FastSACConfig.export_expected_action` defaults True, and under it the exported ONNX
(`FastSACAgent.actor_onnx_wrapper`) emits ``E[tanh(mu + sigma*Z)]`` via 8-node Gauss-Hermite
quadrature -- a function of BOTH mu and sigma. But `Actor.explore(deterministic=True)` returns a
bare ``tanh(mu)*scale + bias``, ignoring sigma entirely (fast_sac.py). Until 2026-08-18 this script
regressed the latter, so:
  * the student's ``fc_logstd`` received EXACTLY ZERO gradient and stayed frozen at its zero-init
    for an entire run (verified directly on 20260818_004912's step-85000 checkpoint: fc_logstd
    weight sum 0.000000, all biases 0.0), which under this actor's parameterization means a uniform
    sigma = exp(-2.5) = 0.082 for every joint in every state, against teachers whose own sigma spans
    0.009-0.26 and is state-dependent;
  * so the deployed student action differed from the deployed teacher action EVEN WHERE THE MEANS
    MATCHED PERFECTLY. Measured size of that discrepancy at identical mu, in tanh units: ~0.0021
    averaged over mu in [-2, 2], up to ~0.0223 on the joints where the teacher's sigma is largest,
    against a then-current mean-matching RMS error of ~0.0113. Secondary on average, dominant on a
    few joints -- a real correctness bug, but NOT on its own an explanation for the residual
    skill-2 behavioral gap.
Now `match_deployed_action` (default: follow `export_expected_action`) makes BOTH the teacher target
and the student prediction the quadrature form, which additionally routes real gradient into
fc_logstd through the existing MSE -- no second loss term and no weighting hyperparameter.

    export HOLOSOMA_DISTILL_MATCH_DEPLOYED_ACTION=0   # force the pre-2026-08-18 tanh(mu) target
    export HOLOSOMA_DISTILL_LOGSTD_LOSS_WEIGHT=0.0    # optional aux term, see below

``E[tanh(mu + sigma*Z)]`` is not injective in (mu, sigma), so a student can match the expectation
with individually-wrong mu and sigma. THIS IS NOT COSMETIC (corrected 2026-08-18 -- an earlier
version of this docstring called it "harmless here", which a real run disproved): the DAgger rollout
consumes the same (mu, sigma), so a degenerate sigma corrupts the state distribution the loss is
measured on. Left unpenalized, sigma inflates because that is a cheaper way to shrink the output
than fixing mu -- measured on run 20260818_052857 at step 200000, the student's log_std mean had
drifted to -0.802 vs the teacher's -3.256 (sigma ~12x too large), realized sigma p99 pinned against
its exp(LOG_STD_MAX)=1.0 ceiling, and distill_mse/hit_rate/topple all degraded monotonically.
HOLOSOMA_DISTILL_LOGSTD_LOSS_WEIGHT adds an auxiliary MSE on raw log_std to break the degeneracy and
now DEFAULTS TO 0.001 whenever match_deployed_action is on (see its own comment for how that value
was scaled from the measurement); set it to 0.0 to restore the old, known-degenerate behavior.

OPTIONAL CRITIC DISTILLATION (2026-08-28)
-------------------------------------------------------------------------------------------------
Off by default (HOLOSOMA_DISTILL_CRITIC_LOSS_WEIGHT=0.0, exact no-op -- no extra teacher forward
passes, no q_optimizer built, byte-identical to every distillation run before this existed).

Why this exists. This script only ever optimized the actor -- `optimizer = Adam(student_actor.
parameters())`. The student's qnet/qnet_target are whatever `setup()` randomly initialized and are
NEVER touched here, so a distilled checkpoint's critic is pure noise (verified: qnet and
qnet_target are byte-identical, only possible pre-training). Resuming real RL training straight
from that checkpoint reproduces the stale-critic failure `critic_warmup_iters` exists for, except
worse: a frozen actor during warmup only ever visits its OWN narrow rollout, so the critic becomes
accurate in a tube around the distilled policy and unconstrained everywhere else -- the instant the
actor unfreezes it is rewarded for finding exactly the states where that critic is wrong. Measured
on a real run seeded from a SINGLE teacher's critic instead (`scripts/seed_critic_from_teacher.py`):
the disturbance shrank enormously (peak topple 0.40 vs. 1.0 with a random critic) and, unlike the
random-critic run, genuinely recovered -- action_std/entropy converged back to the teacher's own
healthy band within ~6-7k steps. But the seeded critic only ever saw ONE teacher's skill during
ITS OWN training: per-skill telemetry on that run showed the seed skill outperforming the other
three by 5-9x on topple_frac. This section is the fix that removes that asymmetry -- a critic that
has genuine signal from ALL N teachers, not one.

WHY CROSS-ENTROPY AGAINST THE TEACHER'S SOFTMAX, NOT MSE ON A COLLAPSED SCALAR. This project's
critic is a C51-style DISTRIBUTIONAL critic (`Critic`/`DistributionalQNetwork` in fast_sac.py,
`num_atoms=101`): `forward()` returns raw logits per q-network, and the REAL training loss
(`_update_main`) is `-sum(target_distribution * log_softmax(q_outputs))` -- cross-entropy against a
projected 101-atom target distribution, never MSE on `E[Q]`. Distilling only the collapsed scalar
value would throw away exactly the information (variance/skew across atoms) the real critic update
is built around, and would be a second train/deploy-shaped mismatch of the same kind
`match_deployed_action` already had to fix twice on the actor side. So the target here is the
teacher's own softmax output, matched via cross-entropy -- the identical loss FORM the critic is
normally trained with, just against a frozen teacher instead of a projected TD target.

TWIN CRITICS PAIR INDEPENDENTLY (qnets[0]<->qnets[0], qnets[1]<->qnets[1]), NOT toward a shared
min. `Critic.forward` stacks both q-networks' outputs along dim 0 in declared order, for both
teacher and student (same class, same construction), so pairing is correct by construction with no
extra bookkeeping. Regressing both student heads toward `min(teacher_Q1, teacher_Q2)` would
collapse the two student heads toward an identical function, defeating the reason SAC keeps two
(the min-over-both in the real TD target is exactly what controls overestimation -- it needs two
genuinely different estimators, not one duplicated).

WHAT (obs, action) PAIR IS THE Q EVALUATED AT: the state the student actually visited, and actions
that TEACHER's own policy would take there -- never the student's own predicted action. A teacher's
Q is only calibrated where its own policy goes; asking it to value the student's (possibly quite
different, especially early) action would query it off-distribution and teach the student a
fiction. This mirrors how the actor loss is framed: regress toward what the teacher would do here.

ACTION COVERAGE -- why ONE action per state is not enough (2026-08-28, measured defect + fix).
The first version of this regressed Q at exactly one action per state (`target_action`, the
teacher's recommended action). That is sufficient to make the student's Q *values* match, and
completely insufficient for what the critic is actually FOR. SAC's actor update is `-Q(s, pi(s))`:
it consumes **dQ/da**, the way Q varies with the action. Constraining Q at a single point per state
leaves that gradient entirely unconstrained -- free to be arbitrary. A real TD-trained critic never
has this problem because SAC feeds it many *sampled* actions per state from the replay buffer.

Measured consequence, comparing two RL resumes that differ in the critic they warm-started from:
  * critic SEEDED from one real teacher critic (scripts/seed_critic_from_teacher.py, run
    20260828_003018): action_std peaked at 1.4x its start, entropy drifted +0.19, recovered.
  * critic DISTILLED with single-action coverage (run 20260828_085838): action_std peaked at 4.7x,
    entropy drifted +16.34 (-23.71 -> -7.37, i.e. AWAY from target), topple peaked 0.982.
The actor walked wherever the unconstrained dQ/da pointed, which included inflating sigma.
(Caveat, stated honestly: those two runs also differ in skill set (4 vs 3) and warmup length, so
this is a strong mechanistic hypothesis plus two data points, not a controlled experiment.)

ATTEMPTED FIX THAT DID NOT WORK -- `HOLOSOMA_DISTILL_CRITIC_ACTION_SAMPLES`, default 0 (OFF).
The obvious repair is to regress Q at extra actions per state (drawn from the teacher's own
stochastic policy, where its Q is genuinely calibrated) so the student sees Q's action-dependence.
It is implemented and available, but **measured not to help**, so it ships off.

Direct measurement (student critic fit against a teacher critic, dQ/da cosine similarity at the
reference action, 800 Adam steps, 256 states):

    n_extra_samples   perturbation_scale   dQ/da cosine
          0                  --                0.328     <- BEST
          1                 0.03               0.281
          1                 0.10               0.284
          1                 0.30               0.298
          2                 0.30               0.269
          4                 0.30               0.252
          8                 0.50               0.264

More coverage is slightly WORSE, monotonically-ish, at every scale tried. The reason is a general
property, not a tuning failure: **matching a function's VALUES at sampled points does not constrain
its GRADIENT.** Pinning Q at k points per state leaves dQ/da free between them, and in a 29-dim
action space no feasible k gives dense coverage. Spending the same optimizer budget across more
points also fits each one less well, which is the small degradation visible above.

The principled fix is Sobolev/gradient matching -- add an explicit loss on
||dQ_student/da - dQ_teacher/da||^2, which constrains the derivative directly instead of hoping it
falls out of value matching. That needs a backward through the teacher for the target gradient and
a double-backward through the student, and is NOT implemented here.

PRACTICAL CONSEQUENCE, stated plainly: critic DISTILLATION as implemented in this module gives the
student a Q whose values match the teachers' but whose action-gradient does not, and the actor
update consumes precisely that gradient. Single-teacher critic SEEDING
(`scripts/seed_critic_from_teacher.py`) copies a real TD-trained critic wholesale and so preserves
a meaningful action-gradient by construction -- it is the empirically better option today (run
20260828_003018, sigma peak 1.4x, recovered) and should be preferred until gradient matching
exists.

CRITIC_OBS IS THREADED THROUGH THE LOOP IN PARALLEL WITH obs, sourced from `env.reset_with_critic_
obs()` (initial) and `infos["observations"]["critic"]` (every step after) -- this script never
needed critic_obs before since the actor-only loss doesn't touch it. Zero cost when
critic_loss_weight is 0.0 beyond one extra tensor carried around; the extra teacher/student
qnet forward passes only run when the feature is actually on.

QNET_TARGET IS DELIBERATELY LEFT ALONE HERE, NOT POLYAK-AVERAGED DURING DISTILLATION. `setup()`
hard-copies `qnet_target = qnet` at init (fast_sac_agent.py), so without this note qnet would move
toward the teachers' signal while qnet_target stayed at that stale random-init copy -- silently
recreating close to the ORIGINAL problem this section exists to fix. The chosen fix is NOT to
half-reimplement Polyak averaging inside what is fundamentally a supervised-regression loop:
`save_params` hard-copies the freshly-distilled `qnet_state_dict` into `qnet_target_state_dict` too
at save time (both start in sync, byte-identical, analogous to `setup()`'s own initial hard-copy --
just now copying a TRAINED qnet instead of a random one). A short REAL `critic_warmup_iters` after
resuming (the existing mechanism, likely shorter than needed for a random-init critic since this
starting point is already close) is what should build the genuine Polyak divergence between them,
through actual TD learning -- not this script.

Per-skill diagnostic breakdown (`distill_q_loss_skill{N}`, logging only) mirrors the existing
per-skill actor-MSE split below, same rationale: a persistently high q-loss on one skill after
distillation says that skill's critic signal needs more steps or a higher HOLOSOMA_DISTILL_
CRITIC_LOSS_WEIGHT, not a capacity problem in the actor.

    export HOLOSOMA_DISTILL_CRITIC_LOSS_WEIGHT=0.01   # 0.0 (default) = off, exact no-op
    export HOLOSOMA_DISTILL_Q_LR=3e-4                 # matches FastSACConfig.critic_learning_rate's own default
    export HOLOSOMA_DISTILL_CRITIC_ACTION_SAMPLES=0   # default 0 -- measured NOT to help, see ACTION COVERAGE

CRITIC-ONLY MODE: HOLOSOMA_DISTILL_FREEZE_ACTOR (2026-08-28)
-------------------------------------------------------------------------------------------------
Freezes the actor entirely -- no actor loss computed, no `optimizer.step()`, actor weights come out
of the run BYTE-IDENTICAL to what went in -- while the critic distills normally. Off by default.

WHY THIS EXISTS, measured. Turning critic distillation on for a +20k continuation run
(200k -> 220k) left the actor materially WORSE under real RL rollout: resumed training showed
kick_topple_frac 0.67-0.74 *while the actor was still frozen by critic_warmup_iters*, versus
0.07-0.16 for the same lineage's 200k actor. Critic distillation itself was NOT the cause and
provably cannot be -- `optimizer` holds only `student_actor.parameters()`, `q_optimizer` only
`student_qnet.parameters()`, `target_action`/`critic_targets` are both detached (computed under
`torch.no_grad()`), the two `.backward()`/`.step()` pairs are fully separate, AND the DAgger
rollout action is actor-only (`predict_action`), so the critic has no path -- direct or indirect,
gradient or state-distribution -- to influence the actor's trajectory. Running with
CRITIC_LOSS_WEIGHT=0.0 would have produced the same actor.

The real cause was simply that the actor kept training for 20k more DAgger steps. Measured actor
RMS drift vs the 200k checkpoint: 0.0039 (201k) -> 0.0073 (205k) -> 0.0098 (210k) -> 0.0119 (215k)
-> 0.0140 (220k), steady and monotonic, against a mean parameter magnitude of ~0.115 -- i.e. a
~12% relative weight change, not incidental noise. Whether continued DAgger training degrades real-
sample behavior generally (prime suspect: the sigma-inflation pathology documented in the
2026-08-18 section above, whose `logstd_loss_weight=0.001` counter may be too weak over a 200k+
horizon) is UNRESOLVED -- this flag sidesteps the question rather than answering it.

WHEN TO USE IT: you already have an actor you have VALIDATED under real rollout (see
`scripts/validate_checkpoint_real_rollout.py`) and want only to upgrade its critic -- e.g. going
from a single-teacher `scripts/seed_critic_from_teacher.py` splice to a genuine multi-teacher
distilled critic. Freezing removes the actor as a variable entirely: the output checkpoint's actor
is bit-identical to the input's, so any behavior change on a later RL resume is attributable to the
critic alone.

WHAT STILL RUNS while frozen: the DAgger rollout itself (the student's own action still drives
`env.step()`, so the critic is still trained on the state distribution the student actually
visits -- this is NOT the same as offline regression on a fixed dataset), the teachers' target
computation, all per-skill/per-phase diagnostics, checkpointing, ONNX export, and the MuJoCo
rollout bundle. Only the actor's loss and optimizer step are skipped. `distill_mse` is still
LOGGED (it remains a meaningful read on how far the frozen actor sits from the teachers) but no
longer has any gradient effect.

REQUIRES critic distillation to be on. Freezing the actor with CRITIC_LOSS_WEIGHT=0.0 would train
nothing at all -- an expensive no-op that would silently burn a GPU for hours and emit checkpoints
identical to its input, so it is rejected at startup rather than allowed to run.

    export HOLOSOMA_DISTILL_FREEZE_ACTOR=1            # 0/unset (default) = actor trains normally

DIAGNOSTIC BREAKDOWN (logging only, no gradient effect)
--------------------------------------------------------
Alongside ``Loss/distill_mse`` the loop logs ``distill_mse_skill{N}`` per skill and
``distill_mse_strike``/``distill_mse_nonstrike`` (split on the same ``in_strike_phase`` accessor the
6 shooting reward terms gate on). These exist to separate two explanations for "MSE is tiny but the
behavioral gap persists" that a single pooled number cannot distinguish and that call for opposite
fixes: a persistent per-skill split means one network is struggling to hold two specialists at once
(capacity / per-skill head), whereas error concentrated in the strike means the batch mean is
hiding it and a phase-weighted loss would target it directly.

HOLOSOMA_DISTILL_TEACHER_CKPTS (the same "0=path,1=path" form as before) still works and takes
priority if set -- useful for a one-off override without editing the yaml -- but is no longer
required.

RESUMING (2026-08-18): pass `--training.checkpoint <student.pt>` to continue an interrupted
distillation. The student's actor, obs_normalizer AND global_step are restored, so the run picks up
exactly where it stopped and still targets the same HOLOSOMA_DISTILL_STEPS total. Before this,
--training.checkpoint was silently ignored by this script (only the TEACHERS ever loaded) and an
interrupted run could only be restarted from zero.

The output checkpoint is written via the SAME `save_params`/`FastSACAgent.save` format every other
checkpoint in this project uses (actor_state_dict, obs_normalizer_state, plus a qnet/optimizer
state that's just whatever `setup()` randomly initialized -- the student has no critic yet, see
the module docstring's DAgger-loop discussion). That makes it a valid `--training.checkpoint` for
`train_agent.py` if you later want to RL-finetune the distilled actor -- expect the critic to need
re-learning from scratch against the now-frozen-then-resumed actor.
"""

from __future__ import annotations

import dataclasses
import math
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import tyro
from loguru import logger

from holosoma.config_types.simulator import BALL_X_ENV_VAR, BALL_Y_ENV_VAR, apply_ball_cli_overrides

apply_ball_cli_overrides()

from holosoma.config_types.env import get_tyro_env_config  # noqa: E402
from holosoma.config_values.experiment import AnnotatedExperimentConfig  # noqa: E402
from holosoma.train_agent import configure_logging, get_device  # noqa: E402
from holosoma.utils.config_utils import CONFIG_NAME  # noqa: E402
from holosoma.utils.eval_utils import init_sim_imports  # noqa: E402
from holosoma.utils.experiment_paths import get_experiment_dir, get_timestamp  # noqa: E402
from holosoma.utils.helpers import get_class  # noqa: E402
from holosoma.utils.sim_utils import close_simulation_app  # noqa: E402
from holosoma.utils.tyro_utils import TYRO_CONIFG  # noqa: E402

_TEACHER_CKPTS_ENV_VAR = "HOLOSOMA_DISTILL_TEACHER_CKPTS"
_STEPS_ENV_VAR = "HOLOSOMA_DISTILL_STEPS"
_LR_ENV_VAR = "HOLOSOMA_DISTILL_LR"
_SAVE_INTERVAL_ENV_VAR = "HOLOSOMA_DISTILL_SAVE_INTERVAL"
_LOG_INTERVAL_ENV_VAR = "HOLOSOMA_DISTILL_LOG_INTERVAL"
_LOCOMOTION_TEACHER_ENV_VAR = "HOLOSOMA_DISTILL_LOCOMOTION_TEACHER_SKILL_ID"
# 2026-08-18, the train/deploy-mismatch fix. See the module docstring's own section and
# expected_action()'s docstring inside distill().
_MATCH_DEPLOYED_ENV_VAR = "HOLOSOMA_DISTILL_MATCH_DEPLOYED_ACTION"
_LOGSTD_LOSS_WEIGHT_ENV_VAR = "HOLOSOMA_DISTILL_LOGSTD_LOSS_WEIGHT"
# 2026-08-18 capacity-hypothesis probe: widen ONLY the student's actor. Teachers are NOT affected
# (they stay on algo_config's own actor_hidden_dim, i.e. whatever --algo.config.actor-hidden-dim
# resolved to) -- every teacher checkpoint in this project was trained at hidden_dim=512, so
# passing --algo.config.actor-hidden-dim on this script's own CLI to widen the student also
# resizes teacher_agent's config and load_state_dict raises a size-mismatch on the teacher
# checkpoint. This var is the ONLY supported way to give the student a different width.
_STUDENT_ACTOR_HIDDEN_DIM_ENV_VAR = "HOLOSOMA_DISTILL_STUDENT_ACTOR_HIDDEN_DIM"
# 2026-08-28, optional critic distillation. See the module docstring's own section. 0.0 (default)
# = off: no q_optimizer built, no extra teacher/student qnet forward passes, byte-identical to
# every distillation run before this existed.
_CRITIC_LOSS_WEIGHT_ENV_VAR = "HOLOSOMA_DISTILL_CRITIC_LOSS_WEIGHT"
_Q_LR_ENV_VAR = "HOLOSOMA_DISTILL_Q_LR"
# 2026-08-28: freeze the actor, train ONLY the critic. See the module docstring's
# "CRITIC-ONLY MODE" section for the measurement that motivated this.
_FREEZE_ACTOR_ENV_VAR = "HOLOSOMA_DISTILL_FREEZE_ACTOR"
# 2026-08-28: how many EXTRA sampled actions per state the critic loss covers, beyond the
# teacher's own recommended action. DEFAULT 0 (off) -- this was implemented as a fix for the
# unconstrained-dQ/da defect and then MEASURED NOT TO HELP (it is slightly worse at every
# scale/count tried). Kept available for further investigation only. See the module
# docstring's ACTION COVERAGE section for the measurement table and why value-matching
# fundamentally cannot constrain a gradient.
_CRITIC_ACTION_SAMPLES_ENV_VAR = "HOLOSOMA_DISTILL_CRITIC_ACTION_SAMPLES"
_DEFAULT_CRITIC_ACTION_SAMPLES = 0
# Noise scales for critic action coverage, cycled across the requested sample slots. Spans
# near-optimal (0.03, the teachers' own policy sigma) to near-saturating (1.0) so the student
# critic learns the full Q-vs-action falloff -- see Teacher.perturbed_action's docstring for the
# measurement showing narrow-only sampling is ~flat and therefore useless here.
_CRITIC_ACTION_SIGMAS = (0.3, 1.0, 0.03)

_DEFAULT_STEPS = 200_000
_DEFAULT_LR = 3e-4
_DEFAULT_SAVE_INTERVAL = 5_000
_DEFAULT_LOG_INTERVAL = 100
_TRUTHY = {"1", "true", "yes", "on"}


def _validate_dense_skill_ids(ckpts: dict[int, str], source: str) -> None:
    """Shared by both parse_teacher_ckpts (env var) and parse_teacher_ckpts_from_skills_yaml
    (yaml field): skill_id is used as a direct tensor index (teachers[env.skill_id]), so a gap or
    an out-of-range id must be caught HERE, not left to index-error deep inside the training loop.
    """
    if not ckpts:
        raise ValueError(f"{source}: no teacher checkpoints declared.")
    expected = set(range(len(ckpts)))
    if set(ckpts) != expected:
        raise ValueError(
            f"{source}: skill ids {sorted(ckpts)} are not a dense 0..N-1 range -- expected exactly "
            f"{sorted(expected)}."
        )


def parse_teacher_ckpts(raw: str) -> dict[int, str]:
    """Parse "0=path/to/a.pt,1=path/to/b.pt" -> {0: "path/to/a.pt", 1: "path/to/b.pt"}.

    Split out of the entry point so it's unit-testable without a live env (see
    tests/test_distill_specialists.py) -- this is exactly the kind of stringly-typed parsing that
    silently does the wrong thing on a typo (stray space, "=" inside a path, duplicate key) if
    it isn't independently verified.
    """
    ckpts: dict[int, str] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(
                f"{_TEACHER_CKPTS_ENV_VAR}: entry {entry!r} is missing '=' -- expected "
                "'<skill_id>=<path>[,<skill_id>=<path>...]', e.g. '0=ckpt_a.pt,1=ckpt_b.pt'."
            )
        skill_str, path = entry.split("=", 1)
        skill_str = skill_str.strip()
        path = path.strip()
        if not skill_str.isdigit():
            raise ValueError(
                f"{_TEACHER_CKPTS_ENV_VAR}: skill id {skill_str!r} in entry {entry!r} is not a "
                "non-negative integer."
            )
        skill_id = int(skill_str)
        if skill_id in ckpts:
            raise ValueError(f"{_TEACHER_CKPTS_ENV_VAR}: skill id {skill_id} declared twice.")
        if not path:
            raise ValueError(f"{_TEACHER_CKPTS_ENV_VAR}: entry {entry!r} has an empty path.")
        ckpts[skill_id] = path
    if not ckpts:
        raise ValueError(
            f"{_TEACHER_CKPTS_ENV_VAR} must declare at least one teacher if set at all, e.g. "
            f"export {_TEACHER_CKPTS_ENV_VAR}='0=logs/.../skill1_ckpt.pt,1=logs/.../skill2_ckpt.pt' "
            "-- or leave it unset entirely and declare 'teacher_checkpoint:' per skill in the "
            "skills yaml instead (see parse_teacher_ckpts_from_skills_yaml)."
        )
    _validate_dense_skill_ids(ckpts, _TEACHER_CKPTS_ENV_VAR)
    return ckpts


def parse_teacher_ckpts_from_skills_yaml(skills_yaml_path: str) -> dict[int, str]:
    """Read each motion_skill_N block's own `teacher_checkpoint:` field -- the no-long-export-line
    alternative to HOLOSOMA_DISTILL_TEACHER_CKPTS, e.g.:

        motion_skill_1:
          ...
          teacher_checkpoint: logs/.../skill1_ckpt.pt
        motion_skill_2:
          ...
          teacher_checkpoint: logs/.../skill2_ckpt.pt

    Blocks are read in the SAME declaration order `_parse_skill_blocks` (config_types/multi_skill.py)
    uses to assign skill_id (dict insertion order from yaml.safe_load, matching that function's own
    documented ordering contract) -- so skill_id N here is guaranteed to be the SAME skill N
    everywhere else (env.skill_id, per-skill reward/termination tables, etc.), never independently
    re-derived or guessable-wrong.

    `teacher_checkpoint` is intentionally NOT threaded through MultiSkillConfig/SkillConfig
    (config_types/multi_skill.py) -- it's meaningless to every other consumer of that yaml (RL
    training via train_agent.py has no use for it), so it stays a distillation-only concern parsed
    directly here rather than growing the shared schema. Confirmed safe to add as an extra key:
    `_parse_skill_blocks` generally rejects only MISSING required keys, never unrecognized extra
    ones (that stricter check otherwise exists only for the separate base_robot: block) -- the ONE
    exception, since 2026-08-22, is randomize_target_x/y specifically (a removed field, rejected
    on sight rather than silently ignored, see SkillConfig.kick_aim_enabled's own docstring), which
    `teacher_checkpoint` is unrelated to. So this file loads identically through the normal
    training path too -- an RL run pointed at this same yaml would just never look at the field.
    """
    import yaml

    path = Path(skills_yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"skills yaml {path} does not exist (from HOLOSOMA_SKILLS_CONFIG).")
    raw = yaml.safe_load(path.read_text()) or {}
    skill_keys = [k for k in raw if isinstance(k, str) and k.startswith("motion_skill_")]
    if not skill_keys:
        raise ValueError(f"{path} has no 'motion_skill_N' blocks.")

    ckpts: dict[int, str] = {}
    for i, key in enumerate(skill_keys):
        block = raw[key]
        if not isinstance(block, dict) or "teacher_checkpoint" not in block:
            raise ValueError(
                f"{path}:{key} has no 'teacher_checkpoint:' field. Add one (that skill's specialist "
                f".pt path), or set {_TEACHER_CKPTS_ENV_VAR} explicitly instead to bypass this file "
                "entirely."
            )
        checkpoint = str(block["teacher_checkpoint"]).strip()
        if not checkpoint:
            raise ValueError(f"{path}:{key}: 'teacher_checkpoint:' is empty.")
        ckpts[i] = checkpoint
    _validate_dense_skill_ids(ckpts, str(path))
    return ckpts


def _sync_qnet_target(student_agent) -> None:
    """Hard-copy qnet_target <- qnet, called just before saving whenever critic distillation ran
    this step. See module docstring's own section for why: qnet_target starts as setup()'s
    hard-copy of a RANDOM qnet and is never Polyak-updated here (no TD learning happens in this
    script), so without this it would silently stay stale while qnet moved toward the teachers'
    signal. Mirrors setup()'s own initial hard-copy, just copying a trained qnet instead of a
    random one -- deliberately not a partial/EMA blend, since after this only a real
    critic_warmup_iters phase on resume is meant to build genuine Polyak divergence."""
    student_agent.qnet_target.load_state_dict(student_agent.qnet.state_dict())


def _log_env_int(var: str, default: int) -> int:
    raw = os.environ.get(var)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as e:
        raise ValueError(f"{var}={raw!r} is not an integer.") from e


def _log_env_float(var: str, default: float) -> float:
    raw = os.environ.get(var)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as e:
        raise ValueError(f"{var}={raw!r} is not a float.") from e


def distill(tyro_config) -> None:
    simulation_app = init_sim_imports(tyro_config)
    try:
        # torch (and anything that transitively imports it, e.g. FastSACAgent) must not be
        # imported at this module's top level -- only after init_sim_imports() above -- same
        # constraint train_agent.py's own train() documents at its own `import torch` line.
        import torch
        import wandb
        from torch.nn import functional as F

        from holosoma.agents.fast_sac.fast_sac_agent import FastSACAgent
        from holosoma.utils.common import seeding

        class Teacher:
            """A frozen specialist: one Actor + the EmpiricalNormalization it was trained with,
            paired together and never updated. See this module's docstring for why the pairing
            matters. Defined here (not at module level) because it needs `torch` in scope, which
            per the constraint above cannot be imported before this point."""

            def __init__(self, skill_id: int, ckpt_path: str, agent: FastSACAgent) -> None:
                self.skill_id = skill_id
                self.ckpt_path = ckpt_path
                agent.setup()
                agent.load(ckpt_path)
                self.actor = agent.actor
                self.actor.eval()
                for p in self.actor.parameters():
                    p.requires_grad_(False)
                self.obs_normalizer = agent.obs_normalizer
                self.obs_normalizer.eval()
                # Critic side (2026-08-28, only meaningfully used when critic_loss_weight > 0.0,
                # but constructing the references is free -- agent.setup()/.load() already build
                # these as part of the normal teacher agent). Frozen for the identical reason the
                # actor is: this teacher's qnet is a fixed regression TARGET, never trained here.
                self.qnet = agent.qnet
                self.qnet.eval()
                for p in self.qnet.parameters():
                    p.requires_grad_(False)
                self.critic_obs_normalizer = agent.critic_obs_normalizer
                self.critic_obs_normalizer.eval()
                # Keep the agent referenced only so its actor/normalizer stay alive with correct
                # device placement -- never call .learn()/.env.step() on it; only .actor/.obs_normalizer/
                # .qnet/.critic_obs_normalizer are used, all accessed exclusively through the
                # wrapper methods below.
                self._agent = agent

            def act(self, raw_obs: torch.Tensor) -> torch.Tensor:
                """The teacher's REGRESSION TARGET. Must stay the same functional form as the
                student's own prediction (predict_action below) or the loss is comparing two
                different quantities -- see match_deployed_action's docstring."""
                normed = self.obs_normalizer(raw_obs, update=False)
                if match_deployed_action:
                    return expected_action(self.actor, normed)
                # explore() is already @torch.no_grad()-decorated internally (fast_sac.py) -- no
                # need to wrap here, and teachers are frozen (requires_grad_(False)) regardless.
                return self.actor.explore(normed, deterministic=True)

            def perturbed_action(self, raw_obs: torch.Tensor, sigma: float) -> torch.Tensor:
                """tanh(mu + sigma*Z) at an EXPLICIT noise scale -- NOT the teacher's own policy
                sigma. Used by critic distillation's action-coverage term.

                CORRECTED 2026-08-28. The first version sampled from the teacher's own policy
                (`explore(deterministic=False)`, sigma ~0.03) on the reasoning that a teacher's Q
                is only calibrated where its own policy goes. That reasoning was wrong in a way
                that mattered: the point of action coverage is to teach the student that WIDE
                (bad) actions have LOW Q, and actions within 0.03 of the mean are all
                near-optimal, so they convey none of that. Measured on the real skill012 teacher,
                distilling a fresh critic and then reading dE[Q] as sigma goes 0.01 -> 1.0:

                    teacher (reference)              -0.8168   <- strong restoring force
                    student, narrow actions (0.03)   -0.0050   <- essentially flat
                    student, wide actions (.03/.3/1) -0.0538   <- 10x better, still 15x short

                And the teacher IS calibrated at wide actions -- TD bootstrapping through the
                target network covers the whole action space, which is exactly why its own Q
                falls off so cleanly with sigma. The extrapolation worry was unfounded.

                WHY THIS MATTERS: SAC's actor loss is `alpha*log_probs - Q`. The entropy term
                always pushes sigma UP; Q pushing it DOWN is what balances it (equilibrium
                sigma ~0.033). A critic that is flat in sigma supplies no downward force, so
                sigma inflates unopposed to the ceiling -- the measured failure on run
                20260828_115812 (action_std 0.0335 -> 0.1305 in one logging interval, immediately
                at actor unfreeze, WITH the alpha windup already fixed and alpha at just 0.001).
                """
                normed = self.obs_normalizer(raw_obs, update=False)
                with torch.no_grad():
                    _, mean, _log_std = self.actor(normed)
                    noisy = mean + sigma * torch.randn_like(mean)
                    if self.actor.use_tanh:
                        return torch.tanh(noisy) * self.actor.action_scale + self.actor.action_bias
                    return noisy

            def q_distribution(self, raw_critic_obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
                """This teacher's Q-value distribution at (state, ACTION) -- the regression
                TARGET for critic distillation, see module docstring's own section for why the
                action must be this teacher's own recommended action, not the student's.

                Returns softmax probabilities, shape [num_q_networks, batch, num_atoms] -- same
                stacking order `Critic.forward` always produces, so pairing against the student's
                own qnets[i] needs no extra bookkeeping (see module docstring: twin critics pair
                independently, index i <-> index i).
                """
                normed = self.critic_obs_normalizer(raw_critic_obs, update=False)
                return F.softmax(self.qnet(normed, action), dim=-1)

            def raw_dist(self, raw_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                """(mean, log_std) BEFORE any squashing -- only used by the optional auxiliary
                log_std term, which needs the teacher's raw scale parameter rather than any
                action-space quantity derived from it."""
                normed = self.obs_normalizer(raw_obs, update=False)
                _, mean, log_std = self.actor(normed)
                return mean, log_std

        def expected_action(actor, normalized_obs: torch.Tensor) -> torch.Tensor:
            """E[tanh(mu + sigma*Z)] * action_scale + action_bias -- the EXACT quantity the exported
            ONNX computes when `export_expected_action` is True (the default, and what every
            checkpoint in this project ships with).

            Gradient-safe: goes through actor.forward(), never explore() (which is hard-
            @torch.no_grad()-decorated). Crucially this is a function of BOTH mu and sigma, so
            unlike the tanh(mu) path it delivers real gradient to fc_logstd.
            """
            _, mean, log_std = actor(normalized_obs)
            if not actor.use_tanh:
                # Without the squash E[mu + sigma*Z] == mu exactly, so the quadrature is a no-op --
                # same short-circuit the exporter itself applies via its `use_expected` flag.
                return mean
            std = log_std.exp()
            shifted = mean.unsqueeze(0) + std.unsqueeze(0) * gh_nodes.view(-1, 1, 1)
            tanh_avg = (torch.tanh(shifted) * gh_weights.view(-1, 1, 1)).sum(dim=0)
            return tanh_avg * actor.action_scale + actor.action_bias

        def deterministic_action_with_grad(actor, normalized_obs: torch.Tensor) -> torch.Tensor:
            """The same quantity Actor.explore(deterministic=True) returns (tanh(mean)*scale+bias,
            or bare mean if use_tanh=False), but via forward() instead of explore(): explore() is
            hard-@torch.no_grad()-decorated UNCONDITIONALLY in fast_sac.py (even under
            deterministic=True), so calling it here for the STUDENT's prediction would silently
            detach the graph and make loss.backward() a no-op (or raise, if the whole graph --
            forward autograd is not gradient-tracked at all, not just this one action detached from
            an otherwise-live graph).

            NOTE this is NOT the deployed action when export_expected_action is True -- see
            expected_action above and HOLOSOMA_DISTILL_MATCH_DEPLOYED_ACTION.
            """
            _, mean, _log_std = actor(normalized_obs)
            if actor.use_tanh:
                return torch.tanh(mean) * actor.action_scale + actor.action_bias
            return mean

        def predict_action(actor, normalized_obs: torch.Tensor) -> torch.Tensor:
            """The STUDENT's prediction, in whichever form the loss is currently matching."""
            if match_deployed_action:
                return expected_action(actor, normalized_obs)
            return deterministic_action_with_grad(actor, normalized_obs)

        device: str = get_device(tyro_config, None)
        seeding(tyro_config.training.seed, torch_deterministic=tyro_config.training.torch_deterministic)

        # 8-node Gauss-Hermite quadrature for E[tanh(mu + sigma*Z)], Z~N(0,1) -- byte-for-byte the
        # same construction FastSACAgent.actor_onnx_wrapper uses (fast_sac_agent.py), including
        # folding sqrt(2) into the nodes and 1/sqrt(pi) into the weights. Deliberately duplicated
        # from there (rather than imported) because it lives inside a locally-defined nn.Module; if
        # that construction ever changes, THIS MUST CHANGE WITH IT or the distillation target
        # silently stops matching what actually gets deployed. Built here, after `device` exists,
        # and closed over by expected_action above (closures resolve at call time, so the
        # definition order between the two does not matter -- only that this runs before the loop).
        _gh_t, _gh_w = np.polynomial.hermite.hermgauss(8)
        gh_nodes = torch.tensor(_gh_t * math.sqrt(2.0), dtype=torch.float32, device=device)
        gh_weights = torch.tensor(_gh_w / math.sqrt(math.pi), dtype=torch.float32, device=device)

        timestamp = get_timestamp()
        experiment_dir = get_experiment_dir(tyro_config.logger, tyro_config.training, timestamp, task_name="distill")
        configure_logging(log_dir=experiment_dir)
        experiment_dir.mkdir(exist_ok=True, parents=True)
        tyro_config.save_config(str(experiment_dir / CONFIG_NAME))

        # wandb setup -- condensed from train_agent.py's own train() (same block, no multi-GPU
        # rank-0 gating since this script never runs distributed). Fully optional: with
        # logger:disabled, wandb_enabled is False, wandb.run stays None throughout, and
        # LoggingHelper's own "if wandb.run is not None" guard (logging_utils.py) makes every
        # wandb.log call below a silent no-op -- tensorboard (always on) and the console panel
        # still work either way.
        wandb_run_path: str | None = None
        wandb_enabled = tyro_config.logger.type == "wandb"
        if wandb_enabled:
            from holosoma.config_types.logger import WandbLoggerConfig

            assert isinstance(tyro_config.logger, WandbLoggerConfig), "logger.type == 'wandb' but config isn't"
            wandb_cfg = tyro_config.logger
            default_project = tyro_config.training.project or wandb_cfg.project or "default_project"
            default_run_name = f"{timestamp}_{tyro_config.training.name or 'distill'}_{wandb_cfg.group or 'default'}"
            wandb_dir = Path(wandb_cfg.dir or (experiment_dir / ".wandb"))
            wandb_dir.mkdir(exist_ok=True, parents=True)
            logger.info(f"[distill] saving wandb logs to {wandb_dir}")

            wandb_config = dataclasses.asdict(tyro_config)
            ball_x_override = os.environ.get(BALL_X_ENV_VAR)
            ball_y_override = os.environ.get(BALL_Y_ENV_VAR)
            if ball_x_override is not None or ball_y_override is not None:
                wandb_config["ball_cli_override"] = {"x": ball_x_override, "y": ball_y_override}
            wandb_kwargs: dict = {
                "project": wandb_cfg.project or default_project,
                "name": wandb_cfg.name or default_run_name,
                "config": wandb_config,
                "dir": str(wandb_dir),
                "mode": wandb_cfg.mode,
            }
            if wandb_cfg.entity:
                wandb_kwargs["entity"] = wandb_cfg.entity
            if wandb_cfg.group:
                wandb_kwargs["group"] = wandb_cfg.group
            if wandb_cfg.id:
                wandb_kwargs["id"] = wandb_cfg.id
            if wandb_cfg.tags:
                wandb_kwargs["tags"] = list(wandb_cfg.tags)
            if wandb_cfg.resume is not None:
                wandb_kwargs["resume"] = wandb_cfg.resume
            wandb.init(**wandb_kwargs)
            if wandb.run is not None:
                wandb_run_path = f"{wandb.run.entity}/{wandb.run.project}/{wandb.run.id}"

        teacher_ckpts_env = os.environ.get(_TEACHER_CKPTS_ENV_VAR, "").strip()
        if teacher_ckpts_env:
            teacher_ckpts = parse_teacher_ckpts(teacher_ckpts_env)
            logger.info(f"[distill] teacher checkpoints from {_TEACHER_CKPTS_ENV_VAR} (explicit override)")
        else:
            skills_yaml_path = os.environ.get("HOLOSOMA_SKILLS_CONFIG")
            if not skills_yaml_path:
                raise ValueError(
                    f"Need teacher checkpoints from somewhere: either export {_TEACHER_CKPTS_ENV_VAR}, "
                    "or set HOLOSOMA_SKILLS_CONFIG to a skills yaml where every motion_skill_N block "
                    "declares its own 'teacher_checkpoint:' field."
                )
            teacher_ckpts = parse_teacher_ckpts_from_skills_yaml(skills_yaml_path)
            logger.info(f"[distill] teacher checkpoints from {skills_yaml_path}'s own motion_skill_N blocks")
        num_steps = _log_env_int(_STEPS_ENV_VAR, _DEFAULT_STEPS)
        lr = _log_env_float(_LR_ENV_VAR, _DEFAULT_LR)
        save_interval = _log_env_int(_SAVE_INTERVAL_ENV_VAR, _DEFAULT_SAVE_INTERVAL)
        log_interval = _log_env_int(_LOG_INTERVAL_ENV_VAR, _DEFAULT_LOG_INTERVAL)
        locomotion_teacher_id = _log_env_int(_LOCOMOTION_TEACHER_ENV_VAR, 0)
        if locomotion_teacher_id not in teacher_ckpts:
            raise ValueError(
                f"{_LOCOMOTION_TEACHER_ENV_VAR}={locomotion_teacher_id} has no matching teacher in "
                f"{_TEACHER_CKPTS_ENV_VAR} (declared skill ids: {sorted(teacher_ckpts)})."
            )
        # 2026-08-18 train/deploy-mismatch fix. Default TRUE when the agent config says the export
        # uses the expected action (which is FastSACConfig's own default and what every checkpoint
        # in this project ships with): matching tanh(mu) while deploying E[tanh(mu + sigma*Z)] is
        # simply distilling the wrong quantity, so the correct behavior is the default rather than
        # an opt-in. Set HOLOSOMA_DISTILL_MATCH_DEPLOYED_ACTION=0 to force the old tanh(mu) target
        # (e.g. to reproduce a pre-2026-08-18 run exactly, or if a future export drops the
        # expected-action path).
        _export_expected = bool(getattr(tyro_config.algo.config, "export_expected_action", True))
        _match_env = os.environ.get(_MATCH_DEPLOYED_ENV_VAR)
        match_deployed_action = (
            (_match_env.strip().lower() in _TRUTHY) if _match_env is not None else _export_expected
        )
        if match_deployed_action and not _export_expected:
            logger.warning(
                f"[distill] {_MATCH_DEPLOYED_ENV_VAR} is on but algo.config.export_expected_action is "
                "False -- the exported ONNX will use tanh(mu), so this makes the training target "
                "DISAGREE with deployment rather than agree with it. Almost certainly not what you want."
            )
        # Auxiliary term: MSE on raw log_std against the teacher's. ON BY DEFAULT whenever
        # match_deployed_action is on -- 2026-08-18, after its previous 0.0 default was measured to
        # be actively harmful rather than merely optional.
        #
        # I previously documented this term as guarding only a cosmetic concern ("harmless here;
        # deployment consumes only the expectation") -- that was WRONG, and the 20260818_052857 run
        # is the counterexample. E[tanh(mu + sigma*Z)] is not injective in (mu, sigma): the student
        # can hit the target either by fixing mu or by inflating sigma, and inflating sigma is the
        # cheaper path when nothing penalizes it. Measured on that run at step 200000: the student's
        # log_std mean had drifted to -0.802 against the teacher's -3.256 (sigma ~0.45 vs ~0.038,
        # about 12x too large), with realized sigma p99 pinned at 0.992 against the hard
        # exp(LOG_STD_MAX)=1.0 ceiling. Since the DAgger rollout consumes the SAME (mu, sigma) to
        # decide which states get visited, a degenerate sigma is not cosmetic at all -- it corrupts
        # the state distribution the loss itself is measured on.
        #
        # 0.001 is scaled from that measurement rather than guessed: at the diverged state the raw
        # logstd MSE was 9.11 against a distill_mse of 0.0349, so 0.001 makes the aux term ~21% of
        # the total loss THERE (a strong corrective pull exactly when sigma is wrong) while decaying
        # toward 0 as sigma converges onto the teacher's -- roughly 10% of the loss at a healthy
        # fresh start. 0.01 would have made it 72% of the loss and effectively turned this into a
        # log_std-matching run with action matching as an afterthought. NOT yet validated by a
        # completed training run; override via the env var to A/B it (0.0 restores the old,
        # known-degenerate behavior).
        _default_logstd_weight = 0.001 if match_deployed_action else 0.0
        logstd_loss_weight = _log_env_float(_LOGSTD_LOSS_WEIGHT_ENV_VAR, _default_logstd_weight)
        if logstd_loss_weight < 0.0:
            raise ValueError(f"{_LOGSTD_LOSS_WEIGHT_ENV_VAR} must be >= 0.0, got {logstd_loss_weight}")
        logger.info(
            f"[distill] {len(teacher_ckpts)} teacher(s): {teacher_ckpts} | steps={num_steps} lr={lr} "
            f"save_interval={save_interval} locomotion->skill_id {locomotion_teacher_id}"
        )
        logger.info(
            f"[distill] regression target = "
            f"{'E[tanh(mu+sigma*Z)] (matches the exported ONNX)' if match_deployed_action else 'tanh(mu)'}"
            f" | logstd_loss_weight={logstd_loss_weight}"
        )

        # 2026-08-28, optional critic distillation -- see module docstring's own section. 0.0
        # (default) = off, exact no-op: no q_optimizer built below, no extra teacher/student qnet
        # forward passes in the loop.
        critic_loss_weight = _log_env_float(_CRITIC_LOSS_WEIGHT_ENV_VAR, 0.0)
        if critic_loss_weight < 0.0:
            raise ValueError(f"{_CRITIC_LOSS_WEIGHT_ENV_VAR} must be >= 0.0, got {critic_loss_weight}")
        q_lr = _log_env_float(_Q_LR_ENV_VAR, _DEFAULT_LR)
        # See the module docstring's "ACTION COVERAGE" section for why the default is 1, not 0.
        critic_action_samples = _log_env_int(_CRITIC_ACTION_SAMPLES_ENV_VAR, _DEFAULT_CRITIC_ACTION_SAMPLES)
        if critic_action_samples < 0:
            raise ValueError(f"{_CRITIC_ACTION_SAMPLES_ENV_VAR} must be >= 0, got {critic_action_samples}")
        logger.info(
            f"[distill] critic distillation {'ENABLED' if critic_loss_weight > 0.0 else 'off'} "
            f"| critic_loss_weight={critic_loss_weight}"
            + (f" q_lr={q_lr} action_samples={critic_action_samples}" if critic_loss_weight > 0.0 else "")
        )
        if critic_loss_weight > 0.0:
            logger.warning(
                "[distill] NOTE: critic distillation constrains Q's VALUES but not its "
                "action-gradient dQ/da -- which is exactly what SAC's actor update consumes. "
                f"({_CRITIC_ACTION_SAMPLES_ENV_VAR}={critic_action_samples}; extra action coverage "
                "was measured NOT to fix this -- see the module docstring's ACTION COVERAGE "
                "section.) Single-teacher critic SEEDING "
                "(scripts/seed_critic_from_teacher.py) preserves a real action-gradient by "
                "construction and is the empirically better option today."
            )

        # 2026-08-28, critic-only mode -- see module docstring's "CRITIC-ONLY MODE" section for the
        # measured actor drift (0.0140 RMS over a 20k continuation, ~12% relative) that motivated it.
        freeze_actor = os.environ.get(_FREEZE_ACTOR_ENV_VAR, "").strip().lower() in _TRUTHY
        if freeze_actor and critic_loss_weight <= 0.0:
            raise ValueError(
                f"{_FREEZE_ACTOR_ENV_VAR} is set but {_CRITIC_LOSS_WEIGHT_ENV_VAR} is "
                f"{critic_loss_weight} -- with the actor frozen AND no critic loss, this run would "
                "train nothing at all: it would step the env for the full step budget and emit "
                "checkpoints byte-identical to its input. Set a non-zero critic loss weight, or "
                "unset the freeze flag."
            )
        if freeze_actor:
            logger.info(
                "[distill] ACTOR FROZEN -- critic-only mode. No actor loss, no actor optimizer "
                "step; the output checkpoint's actor will be byte-identical to the input's. "
                "distill_mse is still logged (a valid read on frozen-actor-vs-teacher agreement) "
                "but has no gradient effect."
            )

        env_target = tyro_config.env_class
        tyro_env_config = get_tyro_env_config(tyro_config)
        env = get_class(env_target)(tyro_env_config, device=device)

        algo_config = tyro_config.algo.config
        # setup() unconditionally allocates a SimpleReplayBuffer sized (num_envs, buffer_size,
        # obs_dim) per agent -- 2026-08-17 finding: at num_envs=3300, default buffer_size=1024,
        # critic_obs_dim=396, that's a single 3300*1024*396*4-byte allocation of ~4.99 GiB (matches
        # a real OutOfMemoryError seen on the 2nd teacher, once the 1st teacher's + its own qnet
        # etc. had already eaten most of the GPU). Nothing in this script ever touches self.rb --
        # no .sample()/.extend(), those only happen inside FastSACAgent.learn(), which this script
        # never calls -- so the buffer is pure dead weight for all three agents (both teachers AND
        # the student) here. Shrink it to the smallest usable value (2, since num_steps=1 in this
        # project per fast_sac_utils.py -- SimpleReplayBuffer needs no more) rather than the
        # training-scale default. Frees ~15-20 GiB combined at num_envs=3300 across 3 agents.
        agent_config = dataclasses.replace(algo_config, buffer_size=2)

        teachers: dict[int, Teacher] = {}
        for skill_id in sorted(teacher_ckpts):
            ckpt_path = teacher_ckpts[skill_id]
            logger.info(f"[distill] loading teacher skill_id={skill_id} from {ckpt_path}")
            teacher_agent = FastSACAgent(
                env=env, config=agent_config, device=device, log_dir=str(experiment_dir / f"_teacher{skill_id}_scratch")
            )
            teachers[skill_id] = Teacher(skill_id, ckpt_path, teacher_agent)

        _student_hidden_dim = _log_env_int(_STUDENT_ACTOR_HIDDEN_DIM_ENV_VAR, agent_config.actor_hidden_dim)
        student_agent_config = (
            dataclasses.replace(agent_config, actor_hidden_dim=_student_hidden_dim)
            if _student_hidden_dim != agent_config.actor_hidden_dim
            else agent_config
        )
        logger.info(
            f"[distill] building student (fresh weights), actor_hidden_dim={_student_hidden_dim}"
            + (f" (teachers stay at {agent_config.actor_hidden_dim})" if _student_hidden_dim != agent_config.actor_hidden_dim else "")
        )
        student_agent = FastSACAgent(env=env, config=student_agent_config, device=device, log_dir=str(experiment_dir))
        student_agent.setup()
        # setup() has already consumed agent_config.buffer_size to size the (now-irrelevant)
        # replay buffer -- restore the REAL config here so save()'s "args" field in the checkpoint
        # records the actual production config, not the buffer_size=2 shrink. Otherwise the saved
        # checkpoint would disagree with itself: metadata.experiment_config (below, from
        # attach_checkpoint_metadata) already correctly uses the untrimmed tyro_config, but "args"
        # is written straight from self.config. save_interval is ALSO overridden here (not just
        # restored) to this script's own save_interval: _mujoco_rollout_gate_open (below, reused
        # unchanged from FastSACAgent) fires on `global_step % (self.config.save_interval *
        # every_n_saves) == 0` -- if algo_config's own save_interval (an RL-training-loop value
        # this script never uses to actually save anything) disagreed with save_interval (the
        # value this loop's own periodic-save block actually checks against), mujoco rollouts
        # would silently fire on steps where no checkpoint was ever written, or never fire at all.
        # Base is algo_config (the UNTRIMMED production config), per the paragraph above -- NOT
        # agent_config/student_agent_config, whose buffer_size=2 shrink must never reach the saved
        # "args". actor_hidden_dim is then re-applied on top so the recorded width matches what the
        # student was ACTUALLY built with when _STUDENT_ACTOR_HIDDEN_DIM_ENV_VAR widened it;
        # algo_config's own value would be the teachers' width, and a later .load() of this
        # checkpoint would rebuild the actor at the WRONG size and hit the exact size-mismatch this
        # env var exists to avoid on the teacher side.
        student_agent.config = dataclasses.replace(
            algo_config, save_interval=save_interval, actor_hidden_dim=_student_hidden_dim
        )
        # Required before ANY .save() call (base_algo.py's _checkpoint_metadata raises otherwise) --
        # train_agent.py's own train() calls this between .setup() and .load()/.learn() for the
        # exact same reason. wandb_run_path is None when logger:disabled (set above only if the
        # wandb branch ran), matching train_agent.py's own non-wandb behavior exactly.
        # tyro_config's algo.config is swapped for student_agent_config here for the SAME reason as
        # the .config assignment just above: metadata.experiment_config (read back by
        # load_saved_experiment_config -> get_eval_config, the standard path every eval/probe
        # script uses to reconstruct the network before .load()) must record the width the student
        # was ACTUALLY built with, not algo_config's un-widened value -- otherwise every future load
        # of this checkpoint reproduces the exact teacher size-mismatch crash this env var exists to
        # avoid, just one step removed (at eval time instead of at training-launch time).
        # Same algo_config base + actor_hidden_dim override as the .config assignment above, and for
        # the same reason: student_agent_config here would leak buffer_size=2 into the recorded
        # experiment_config. Untouched (identity) when the env var is unset, so default runs record
        # byte-identical metadata to before this env var existed.
        student_tyro_config = (
            dataclasses.replace(
                tyro_config,
                algo=dataclasses.replace(
                    tyro_config.algo, config=dataclasses.replace(algo_config, actor_hidden_dim=_student_hidden_dim)
                ),
            )
            if _student_hidden_dim != agent_config.actor_hidden_dim
            else tyro_config
        )
        student_agent.attach_checkpoint_metadata(student_tyro_config, wandb_run_path=wandb_run_path)
        # 2026-08-18: resume support. Until now the student was ALWAYS built from scratch and
        # --training.checkpoint was silently ignored here (only the TEACHERS ever called .load()),
        # so an interrupted distillation could only be restarted from step 0. FastSACAgent.load()
        # restores actor + obs_normalizer + global_step, which is the complete state this loop
        # carries between steps (the qnet/optimizers it also restores are unused here -- this
        # script never calls learn()), so resuming is exact rather than approximate. The while-loop
        # below is already written against student_agent.global_step, so a resumed run continues
        # counting from the restored value and stops at the same num_steps target.
        resume_ckpt = getattr(tyro_config.training, "checkpoint", None)
        if resume_ckpt:
            logger.info(f"[distill] resuming student from {resume_ckpt}")
            student_agent.load(str(resume_ckpt))
            logger.info(f"[distill] resumed at global_step={student_agent.global_step}")
            if student_agent.global_step >= num_steps:
                logger.warning(
                    f"[distill] resumed global_step ({student_agent.global_step}) is already >= "
                    f"{_STEPS_ENV_VAR}={num_steps}; the training loop will not execute. Raise "
                    f"{_STEPS_ENV_VAR} to continue training from here."
                )
        # LoggingHelper.__init__ (inside FastSACAgent.__init__, before agent_config's buffer_size
        # shrink was even decided) set num_learning_iterations from algo_config's own field, which
        # this script doesn't use for anything -- overwrite with the REAL step budget so the
        # console panel's "Learning iteration X/Y" and wandb's implicit progress reflect this run.
        student_agent.logging_helper.num_learning_iterations = num_steps
        # Same category of fix, more consequential: num_steps_per_env was set from
        # algo_config.logging_interval (a DIFFERENT, RL-training-loop config value), but
        # post_epoch_logging()'s tot_timesteps AND fps calculations both multiply by
        # self.num_steps_per_env assuming it equals "how many steps actually happened between
        # calls" -- which here is log_interval, not algo_config.logging_interval. Both default to
        # 100, so this is silently RIGHT by coincidence at defaults and silently WRONG (fps/
        # tot_timesteps scaled by the mismatch ratio, no error raised) the moment either value is
        # overridden independently of the other. Drive it from the value this loop actually uses.
        student_agent.logging_helper.num_steps_per_env = log_interval
        student_actor = student_agent.actor
        student_norm = student_agent.obs_normalizer
        student_qnet = student_agent.qnet
        student_critic_norm = student_agent.critic_obs_normalizer

        # None in critic-only mode -- kept as a sentinel rather than an unused-but-built optimizer,
        # so "the actor is frozen" is true of the optimizer state too, not just the skipped step.
        optimizer = None if freeze_actor else torch.optim.Adam(student_actor.parameters(), lr=lr)
        # None when critic distillation is off (default) -- kept as a sentinel rather than an
        # unused-but-built optimizer, so "critic_loss_weight=0.0 is an exact no-op" is true of the
        # optimizer state too, not just the loss value.
        q_optimizer = torch.optim.Adam(student_qnet.parameters(), lr=q_lr) if critic_loss_weight > 0.0 else None

        num_skills = len(teachers)
        # skill_id -> teacher index into a stacked action tensor, built once, indexed every step.
        teacher_actors_ordered = [teachers[i] for i in range(num_skills)]

        # reset_with_critic_obs (not plain reset()) only because critic distillation needs
        # critic_obs threaded alongside obs -- see module docstring. critic_obs is unused
        # (carried for nothing) when critic_loss_weight is 0.0, which costs one extra tensor, not
        # any extra computation.
        obs, critic_obs = student_agent.env.reset_with_critic_obs()

        while student_agent.global_step < num_steps:
            # The rollout action MUST be the SAME quantity the loss regresses (predict_action) --
            # not merely "a deterministic action". This is load-bearing, and getting it wrong is
            # what broke the 20260818_052857 run:
            #
            # 2026-08-17 (first fix): switched explore(deterministic=False) -> True, because the
            # tanh(mean) loss gave fc_logstd literally zero gradient (verified on a finished 200k
            # run: fc_logstd was EXACTLY at its zero init throughout), so rolling out with
            # mean + noise was injecting variance from a permanently-untrained, state-blind
            # log_std that nothing was correcting.
            #
            # 2026-08-18 (this fix): once the target became E[tanh(mu + sigma*Z)] (matching what
            # export() actually deploys), `explore(deterministic=True)` -- which returns
            # tanh(mu)*scale+bias -- STOPPED being the trained quantity. fc_logstd finally trains
            # under that target, and because E[tanh(mu,sigma)] is NOT injective in (mu, sigma) and
            # nothing penalized sigma, sigma ran away to its ceiling (measured on that run:
            # realized sigma p99 0.992 against the hard exp(LOG_STD_MAX)=1.0 max, mean 0.365 ->
            # 0.536). The rollout/target gap grew with it -- |tanh(mu) - E[tanh]| mean 0.021 ->
            # 0.040, p99 0.184, max 0.211, i.e. up to ~21% of the action range -- so the student
            # EXECUTED one action while being TRAINED toward a different one. In DAgger that is
            # self-amplifying: the executed action decides which states get visited, so a wrong
            # executed action poisons the very distribution the loss is measured on. Symptoms were
            # monotonic: distill_mse 0.006 -> 0.035, hit_rate 0.43 -> 0.10, topple 0.06 -> 0.38.
            # Tell-tale that isolates it: that run's MuJoCo rollouts (which run the EXPORTED onnx,
            # i.e. E[tanh]) still walked and kicked correctly while the IsaacSim training metrics
            # (running tanh(mu)) collapsed -- deployed policy fine, executed-during-training policy
            # broken. Routing the rollout through predict_action makes the two identical BY
            # CONSTRUCTION, in either target mode, so this class of divergence cannot recur.
            with student_agent.logging_helper.record_collection_time(), torch.no_grad():
                normed_obs = student_norm(obs, update=True)
                rollout_action = predict_action(student_actor, normed_obs)

            raw_skill_id = env.skill_id
            is_kick = env.task_mode_mask("kick").bool() if hasattr(env, "task_mode_mask") else torch.ones_like(raw_skill_id, dtype=torch.bool)
            effective_skill_id = torch.where(is_kick, raw_skill_id, torch.full_like(raw_skill_id, locomotion_teacher_id))

            with torch.no_grad():
                # zeros, not empty: if any_matched ever has a gap (should not happen -- see the
                # warning below), those rows are excluded from the loss entirely rather than
                # letting uninitialized memory silently pollute a batch-averaged MSE.
                target_action = torch.zeros_like(rollout_action)
                any_matched = torch.zeros(obs.shape[0], dtype=torch.bool, device=device)
                # `and not freeze_actor`: this target feeds ONLY the actor's auxiliary log_std loss,
                # so computing it in critic-only mode would be pure waste (an extra teacher forward
                # pass per teacher per step, for a gradient that is never taken).
                target_log_std = (
                    torch.zeros_like(rollout_action) if (logstd_loss_weight > 0.0 and not freeze_actor) else None
                )
                # Critic-distillation targets, as a list of (action, target_dist) PAIRS -- one for
                # the teacher's recommended action plus `critic_action_samples` extra actions
                # sampled from each teacher's own policy. Multiple points per state are what give
                # the student critic a real dQ/da; see the module docstring's ACTION COVERAGE
                # section for the measured defect that single-point coverage caused. None (not
                # merely empty) when critic distillation is off, so the loss block gates on
                # identity rather than a second boolean.
                critic_targets = None
                if critic_loss_weight > 0.0:
                    probe_dist = teacher_actors_ordered[0].q_distribution(critic_obs[:1], target_action[:1])
                    n_q, n_atoms = probe_dist.shape[0], probe_dist.shape[2]
                    # Slot 0 is always target_action itself (shared with the actor loss, not
                    # recomputed); slots 1..N are per-state stochastic samples, filled per-teacher.
                    critic_actions = [target_action] + [
                        torch.zeros_like(rollout_action) for _ in range(critic_action_samples)
                    ]
                    critic_dists = [
                        torch.zeros(n_q, obs.shape[0], n_atoms, device=device)
                        for _ in range(critic_action_samples + 1)
                    ]
                    critic_targets = (critic_actions, critic_dists)
                skill_masks: list[tuple[int, torch.Tensor]] = []
                for teacher in teacher_actors_ordered:
                    mask = effective_skill_id == teacher.skill_id
                    if mask.any():
                        target_action[mask] = teacher.act(obs[mask])
                        if target_log_std is not None:
                            target_log_std[mask] = teacher.raw_dist(obs[mask])[1]
                        if critic_targets is not None:
                            critic_actions, critic_dists = critic_targets
                            # Slot 0: the teacher's recommended action (already computed above).
                            critic_dists[0][:, mask, :] = teacher.q_distribution(
                                critic_obs[mask], target_action[mask]
                            )
                            # Slots 1..N: extra actions from THIS teacher's own stochastic policy,
                            # giving the student critic multiple points per state to fit dQ/da
                            # against -- see module docstring's ACTION COVERAGE section. Each
                            # sample is an independent draw, so N>1 spans the policy's spread
                            # rather than repeating one perturbation.
                            for slot in range(1, critic_action_samples + 1):
                                # WIDE noise scales, not the teacher's own ~0.03 policy sigma --
                                # see perturbed_action's docstring for the measurement showing
                                # narrow sampling conveys nothing about bad actions. The ladder
                                # spans near-optimal to near-saturating so the student learns the
                                # whole Q-vs-action falloff, which is the restoring force that
                                # keeps sigma from inflating.
                                sigma = _CRITIC_ACTION_SIGMAS[(slot - 1) % len(_CRITIC_ACTION_SIGMAS)]
                                sampled = teacher.perturbed_action(obs[mask], sigma)
                                critic_actions[slot][mask] = sampled
                                critic_dists[slot][:, mask, :] = teacher.q_distribution(
                                    critic_obs[mask], sampled
                                )
                        any_matched |= mask
                        skill_masks.append((teacher.skill_id, mask))
                if not bool(any_matched.all()):
                    missing = (~any_matched).sum().item()
                    logger.warning(
                        f"[distill] {missing} env(s) matched no teacher this step (skill_id outside "
                        f"declared range {sorted(teachers)}) -- excluded from this step's loss. This "
                        "should not happen if HOLOSOMA_DISTILL_TEACHER_CKPTS covers every skill_id "
                        "the skills yaml can produce."
                    )

            with student_agent.logging_helper.record_learn_time():
                # In critic-only mode the actor's MSE is still COMPUTED (under no_grad -- it stays
                # a meaningful diagnostic of how far the frozen actor sits from the teachers, and
                # is still logged below) but carries no gradient and drives no optimizer step.
                if freeze_actor:
                    with torch.no_grad():
                        normed_obs_grad = student_norm(obs, update=False)
                        student_pred = predict_action(student_actor, normed_obs_grad)
                        loss = F.mse_loss(student_pred[any_matched], target_action[any_matched])
                else:
                    normed_obs_grad = student_norm(obs, update=False)
                    student_pred = predict_action(student_actor, normed_obs_grad)
                    loss = F.mse_loss(student_pred[any_matched], target_action[any_matched])
                    total_loss = loss
                    if target_log_std is not None:
                        _, _student_mean, student_log_std = student_actor(normed_obs_grad)
                        logstd_loss = F.mse_loss(student_log_std[any_matched], target_log_std[any_matched])
                        total_loss = loss + logstd_loss_weight * logstd_loss
                        student_agent.training_metrics.add({"distill_logstd_mse": logstd_loss.detach()})

                    optimizer.zero_grad(set_to_none=True)
                    total_loss.backward()
                    optimizer.step()

                # Critic distillation -- a SEPARATE backward/step from the actor's above, on
                # purpose: different parameter set (student_qnet, not student_actor), and nothing
                # is gained by fusing them into one graph (target_action/critic_targets are both
                # already detached, computed under the no_grad block above, so there is no shared
                # computation to preserve by keeping them in the same .backward() call).
                if critic_targets is not None:
                    critic_actions, critic_dists = critic_targets
                    normed_critic_obs_grad = student_critic_norm(critic_obs, update=False)
                    # One cross-entropy term per (action, target_dist) pair -- the student's Q is
                    # evaluated at exactly the same actions the teacher's was, so each pair is a
                    # like-for-like comparison. Averaging ACROSS pairs (rather than summing) keeps
                    # the loss magnitude, and hence the effective learning rate, independent of
                    # critic_action_samples -- so changing coverage does not silently also change
                    # how hard the critic is being pushed.
                    per_env_ce_terms = []
                    for action_i, target_dist_i in zip(critic_actions, critic_dists):
                        student_q_logits = student_qnet(normed_critic_obs_grad, action_i)
                        student_q_log_probs = F.log_softmax(student_q_logits, dim=-1)
                        # Cross-entropy per q-network (dim 0), per env; twin critics pair
                        # independently (index i <-> index i) by construction -- see docstring.
                        per_qnet_ce = -(target_dist_i * student_q_log_probs).sum(dim=-1)  # [n_q, batch]
                        per_env_ce_terms.append(per_qnet_ce.mean(dim=0))  # [batch]
                    per_env_q_ce_all = torch.stack(per_env_ce_terms).mean(dim=0)  # [batch]
                    q_loss = per_env_q_ce_all[any_matched].mean()

                    q_optimizer.zero_grad(set_to_none=True)
                    (critic_loss_weight * q_loss).backward()
                    q_optimizer.step()
                    student_agent.training_metrics.add({"distill_q_loss": q_loss.detach()})
            # detach before handing to training_metrics: TensorAverageMeter (average_meters.py)
            # just appends whatever tensor it's given to a plain list -- a grad-tracked tensor
            # there would keep this WHOLE step's autograd graph alive until the next log_interval
            # clears it, growing unboundedly across up to log_interval steps' worth of graphs.
            student_agent.training_metrics.add({"distill_mse": loss.detach()})

            # ------------------------------------------------------------------------------
            # 2026-08-18 DIAGNOSTIC BREAKDOWN (logging only -- never touches the gradient).
            #
            # Why: the pooled distill_mse converged to ~0.008 (about 1.1% of the action range) on
            # the 20260818_004912 run while skill 2's topple_frac stayed ~2.9x its teacher's. A
            # single batch-averaged number cannot distinguish the two candidate explanations, and
            # they call for opposite fixes:
            #   * per-skill split -- if one skill's MSE is persistently higher, one network is
            #     struggling to represent two specialists at once (fix: capacity / per-skill head),
            #     NOT a loss-shape problem.
            #   * per-phase split -- if the error concentrates in the strike (single-support, the
            #     phase the topple root-cause analysis already localized falls to), the mean is
            #     hiding it and a phase-weighted loss targets it directly.
            # Computed under no_grad from the already-materialized tensors, so the cost is a few
            # masked means per step and nothing is added to the optimizer's objective.
            #
            # BOTH a raw and a NORMALIZED version of each split are logged, and the normalized one
            # (`nmse`, = MSE / E[target^2]) is the one to read. Raw MSE alone is confounded by the
            # target's own magnitude, which differs enormously between these groups -- a trap that
            # already produced one wrong reading of these very metrics (2026-08-18): `fc_mu` is
            # ZERO-initialized (fast_sac.py setup_network) and `action_bias` is all zeros, so an
            # untrained student predicts exactly 0 and its raw MSE is identically E[target^2]. The
            # resulting "strike error is 10x non-strike" was therefore purely a statement that
            # strike actions are ~3x larger, not that the strike is harder to fit. Dividing by
            # E[target^2] removes that entirely: nmse == 1.0 means "no better than predicting
            # zero", and it is comparable across phases and skills at any point in training.
            # ------------------------------------------------------------------------------
            with torch.no_grad():
                sq_err = (student_pred.detach() - target_action).pow(2).mean(dim=-1)
                tgt_sq = target_action.pow(2).mean(dim=-1)
                diag: dict[str, torch.Tensor] = {}

                def _add_split(name: str, m: torch.Tensor) -> None:
                    """Raw + normalized for one mask. eps guards a group whose targets are all
                    ~0 (e.g. a phase where the teacher commands almost nothing); there the ratio
                    is meaningless rather than infinite, so it reports 0."""
                    denom = tgt_sq[m].mean()
                    diag[f"distill_mse_{name}"] = sq_err[m].mean()
                    diag[f"distill_nmse_{name}"] = sq_err[m].mean() / denom.clamp_min(1e-8)

                for skill_id, mask in skill_masks:
                    _add_split(f"skill{skill_id}", mask)

                # Per-skill critic-distillation loss, same rationale as the actor split above: a
                # persistently high q-loss on one skill says that skill needs more steps or a
                # higher HOLOSOMA_DISTILL_CRITIC_LOSS_WEIGHT, not a capacity problem in the actor.
                # per_env_q_ce_all is still in scope from the learn-time block above (a `with`
                # block is not a new Python scope). Already averaged over BOTH q-networks and all
                # action-coverage samples, so this matches the pooled distill_q_loss exactly --
                # just split per skill.
                if critic_targets is not None:
                    per_env_q_ce = per_env_q_ce_all.detach()  # [batch]
                    for skill_id, mask in skill_masks:
                        diag[f"distill_q_loss_skill{skill_id}"] = per_env_q_ce[mask].mean()

                # Strike-phase split, kick-mode envs only. in_strike_phase is the SAME accessor the
                # 6 shooting reward terms gate on (managers/command/terms/wbt.py), so "strike" here
                # means exactly what it means everywhere else in this project rather than a new
                # definition invented for this diagnostic.
                motion_command = getattr(env, "command_manager", None)
                if motion_command is not None:
                    mc = motion_command.get_state("motion_command")
                    in_strike = getattr(mc, "in_strike_phase", None)
                    if in_strike is not None:
                        strike = in_strike() if callable(in_strike) else in_strike
                        strike = strike.bool() & any_matched
                        non_strike = (~strike) & any_matched
                        if strike.any():
                            _add_split("strike", strike)
                        if non_strike.any():
                            _add_split("nonstrike", non_strike)
                if diag:
                    student_agent.training_metrics.add(diag)

            with student_agent.logging_helper.record_collection_time():
                # dones now genuinely used (update_episode_stats below), unlike the earlier
                # discarded version of this line -- Actor.explore/.forward still never read it
                # (fast_sac.py -- dead parameter for this non-recurrent MLP actor), but episode
                # bookkeeping needs it.
                next_obs, rewards, dones, infos = student_agent.env.step(rollout_action.float())
                obs = next_obs
                # Carried in parallel with obs (2026-08-28) -- see module docstring's critic
                # distillation section. Unused, but harmlessly still fetched, when
                # critic_loss_weight is 0.0; this dict entry is always present regardless
                # (FastSACEnv.step always computes it, see fast_sac_agent.py:EnvWrapper.step).
                critic_obs = infos["observations"]["critic"]
                # Free diagnostic signal: the student's OWN action just drove this real env step,
                # so infos["to_log"]/["episode"] already contain the same per-skill kick_topple_frac
                # /kick_episode_length/etc. every RL run in this project logs -- feeding them here
                # is what makes "Kick_skills_N/..." show up in wandb/tensorboard for a distillation
                # run too, telling you whether the student's own rollout is stable, not just whether
                # its predicted actions numerically match the teachers'.
                student_agent.logging_helper.update_episode_stats(rewards, dones, infos)

            student_agent.global_step += 1

            if student_agent.global_step % log_interval == 0:
                loss_dict = {k: (v.item() if torch.is_tensor(v) else float(v)) for k, v in student_agent.training_metrics.mean_and_clear().items()}
                student_agent.logging_helper.post_epoch_logging(
                    it=student_agent.global_step, loss_dict=loss_dict, extra_log_dicts={}
                )

            if student_agent.global_step % save_interval == 0:
                save_path = str(experiment_dir / f"model_{student_agent.global_step:07d}.pt")
                onnx_path = str(experiment_dir / f"model_{student_agent.global_step:07d}.onnx")
                logger.info(f"[distill] saving {save_path}")
                if critic_loss_weight > 0.0:
                    _sync_qnet_target(student_agent)
                student_agent.save(save_path)
                # export() only reads env/actor state (dummy zero input for ONNX tracing, plus
                # metadata pulled from command_manager/robot_config) -- verified it never calls
                # env.reset_all()/env.step(), so this can't disrupt this loop's own `obs` tracking.
                student_agent.export(onnx_file_path=onnx_path)
                # Same 4-rollout bundle (kick + walk + kick-handoff, all N-skill-aware) every other
                # training run in this project gets at each checkpoint -- reusing FastSACAgent's
                # OWN methods (not reimplemented here) so it's the exact same tested cadence gate
                # (_mujoco_rollout_gate_open, keyed off algo.config.mujoco-kick-rollout-every-n-
                # saves), pileup guard, and mujoco_media/ wandb key convention as every RL run.
                # Each spawns a background thread -> subprocess (never touches this process's own
                # CUDA/torch state), so this does not block the training loop.
                student_agent._maybe_start_mujoco_kick_rollout(onnx_path)  # noqa: SLF001
                student_agent._maybe_start_mujoco_walk_rollout(onnx_path)  # noqa: SLF001
                student_agent._maybe_start_mujoco_kick_handoff_rollout(onnx_path)  # noqa: SLF001

            # Drain queues of any rollout(s) that finished since the last iteration, logging their
            # video(s) to wandb -- cheap no-op when nothing's pending, called every iteration
            # (not just at save_interval) to match FastSACAgent.learn()'s own unconditional
            # per-iteration drain, since a rollout started several saves ago may only finish now.
            student_agent._drain_mujoco_kick_rollout_queue()  # noqa: SLF001
            student_agent._drain_mujoco_walk_rollout_queue()  # noqa: SLF001
            student_agent._drain_mujoco_kick_handoff_rollout_queue()  # noqa: SLF001

        final_path = str(experiment_dir / f"model_{student_agent.global_step:07d}.pt")
        final_onnx_path = str(experiment_dir / f"model_{student_agent.global_step:07d}.onnx")
        logger.info(f"[distill] done -- saving final checkpoint {final_path}")
        if critic_loss_weight > 0.0:
            _sync_qnet_target(student_agent)
        student_agent.save(final_path)
        student_agent.export(onnx_file_path=final_onnx_path)
        # No mujoco rollout re-trigger here, matching FastSACAgent.learn()'s own final-save block
        # (fast_sac_agent.py) exactly -- only the periodic in-loop save fires one; training is over
        # by this point so there's nothing further to log progress against, and if this step also
        # happened to land on save_interval, the in-loop block above already triggered it.
        #
        # 2026-08-17 finding, NOT present in learn()'s own equivalent block: rollout threads are
        # daemon=True (fast_sac_agent.py), so process exit kills them mid-flight with no chance to
        # queue.put() their result -- confirmed live on a short verification run: skill 1's kick/
        # kick-handoff videos reached wandb, skill 2's did NOT, even though both existed correctly
        # on disk, because _mujoco_kick_rollout_worker loops skills SEQUENTIALLY in one thread and
        # this script exited (wandb.teardown()) before skill 2's turn finished. learn() itself
        # doesn't hit this in practice (its loop keeps running -- and draining -- for potentially
        # hours after any given save), but this script's final save has no such runway. Bounded
        # join (matches record_kick_rollout's own 180s subprocess timeout -- see that module's
        # own docstring -- so this can't hang past what the rollout itself would've taken) before
        # the last drain, so the run's FINAL rollout gets a real chance to actually reach wandb
        # instead of racing process exit.
        _ROLLOUT_JOIN_TIMEOUT_S = 200.0
        for _thread_attr in ("_kick_rollout_thread", "_walk_rollout_thread", "_kick_handoff_rollout_thread"):
            _thread = getattr(student_agent, _thread_attr, None)
            if _thread is not None and _thread.is_alive():
                logger.info(f"[distill] waiting up to {_ROLLOUT_JOIN_TIMEOUT_S:.0f}s for {_thread_attr} to finish...")
                _thread.join(timeout=_ROLLOUT_JOIN_TIMEOUT_S)
                if _thread.is_alive():
                    logger.warning(f"[distill] {_thread_attr} still running after the wait -- its video may be lost.")
        student_agent._drain_mujoco_kick_rollout_queue()  # noqa: SLF001
        student_agent._drain_mujoco_walk_rollout_queue()  # noqa: SLF001
        student_agent._drain_mujoco_kick_handoff_rollout_queue()  # noqa: SLF001

        if wandb_enabled:
            logger.info("[distill] shutting down wandb...")
            wandb.teardown()

    except Exception:
        logger.error(f"Exception during distillation:\n{traceback.format_exc()}")
        sys.exit(1)
    finally:
        close_simulation_app(simulation_app)


def main() -> None:
    tyro_cfg = tyro.cli(AnnotatedExperimentConfig, config=TYRO_CONIFG)
    distill(tyro_cfg)


if __name__ == "__main__":
    main()
