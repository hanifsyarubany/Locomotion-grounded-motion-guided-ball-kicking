"""Smooth Stage B -> Stage C ramp for the shooting REWARD, with an optional zero-hold before the
ramp starts, and (N-skill mode) a genuinely PER-ENV target shooting_reward_scale.

Root cause and full rationale: `BallConfig.shooting_reward_scale_ramp_iters`'s docstring
(config_types/simulator.py) and memory `stagec_obs_normalizer_shock.md`. This module originally
also smoothed a second, observation-side discontinuity -- ball/target obs held at zero in Stage B,
flipping to live in Stage C, which collapsed `EmpiricalNormalization`'s running std for those dims
and then shocked the policy with normalized values of +/-100-170 on the first Stage-C step. That
discontinuity no longer exists (2026-07-21, user directive): `ball_pos_b`/`target_pos_b`
(managers/observation/terms/unified.py) are live in every stage now, matching RoboNaldo's own
arrangement, so there is nothing left on the observation side to ramp or hold -- this module ramps
only the REWARD side.

`shooting_reward_scale_hold_iters` (2026-07-20, user directive) adds an optional HOLD AT EXACTLY 0
for this many steps BEFORE the ramp begins climbing. Motivation: the ramp alone still lets w_g move
off zero on step 1 (a tiny value, but nonzero), so the optimizer's Adam moment estimates and the
critic's Q-estimates -- which were fit against a resume-induced replay-buffer reset independent of
the shooting reward at all -- never get a chance to re-equilibrate purely on the unchanged Stage-B
task before any new signal arrives. A hold period is exactly the w_g=0 CONTROL condition measured
earlier this project (see `stageB_resume_control` in `stagec_obs_normalizer_shock.md`'s appendix):
resuming into an UNCHANGED task, letting the resume transient settle, THEN introducing the new
objective. 0 (default) = no hold, i.e. exactly the previous ramp-only behavior.

Progress is keyed off `env.common_step_counter`, which starts at 0 for any fresh process regardless
of the checkpoint's own absolute saved step count, so a resume "just works" without needing to know
the checkpoint's iteration number. `ramp_iters <= 0` reproduces the old instant-step behavior exactly
(`ramp_progress` returns 1.0 unconditionally, regardless of any hold setting).

N-SKILL MODE (2026-07-23): `current_w_g` now returns a PER-ENV `torch.Tensor`, not a process-wide
`float` -- each env's assigned skill (see UnifiedManager.skill_id / MotionCommand.motion_ids) has
its OWN target shooting_reward_scale (SkillConfig.shooting_reward_scale in the stacked yaml), so
skill_1 can already be at its full Stage-C target while skill_2 is still at 0 (Stage B), in the SAME
step, on the SAME env batch -- impossible with a single process-wide scalar. The ramp/hold SCHEDULE
(iteration counts) stays shared/global across all skills (not per-skill -- simpler, and not
requested); only the value each skill ramps TOWARD differs. Legacy (non-N-skill) mode is unaffected
in substance: current_w_g still returns a tensor now (a genuine per-env one, not a bare float), but
every entry is identical (there's only one target, from BallConfig), so every existing call site
(`raw * current_w_g(env)`) broadcasts to the exact same numeric result as before.
"""

from __future__ import annotations

from typing import Any

import torch

# Cached once per process, not re-read every call: this module's functions run from reward
# TERMS, i.e. up to ~10 times per control step at 50Hz. Loading/parsing the yaml is fine for a
# couple of pre-existing call sites but would multiply into real overhead if repeated here too.
# Safe to cache -- HOLOSOMA_BALL_CONFIG/_X/_Y/HOLOSOMA_SKILLS_CONFIG are documented as
# set-before-process-start, never mutated mid-run, and both BallConfig/MultiSkillConfig are frozen.
_cached: tuple[list[float], int, int] | None = None


def _target_per_motion_ramp_hold() -> tuple[list[float], int, int]:
    """(target shooting_reward_scale per motion, ramp_iters, hold_iters). Length of the first
    element is 1 in legacy (non-N-skill) mode -- a single shared target, broadcasting via modulo
    indexing below regardless of how many motions are actually loaded."""
    global _cached
    if _cached is None:
        from holosoma.config_types.multi_skill import load_multi_skill_config, multi_skill_mode_enabled

        if multi_skill_mode_enabled():
            msc = load_multi_skill_config()
            _cached = (
                [sc.shooting_reward_scale for sc in msc.skills],
                msc.shooting_reward_scale_ramp_iters,
                msc.shooting_reward_scale_hold_iters,
            )
        else:
            from holosoma.config_types.simulator import load_ball_config

            bc = load_ball_config()
            _cached = (
                [bc.shooting_reward_scale],
                bc.shooting_reward_scale_ramp_iters,
                getattr(bc, "shooting_reward_scale_hold_iters", 0),
            )
    return _cached


def ramp_progress(env: Any) -> float:
    """Fraction in [0, 1] of the way through the shared Stage-B -> C ramp SCHEDULE (iteration
    counts only -- not per-skill, see module docstring). 1.0 if ramping is disabled
    (``shooting_reward_scale_ramp_iters <= 0``) -- the pre-ramp instant-step behavior. Stays
    exactly 0.0 for the first ``shooting_reward_scale_hold_iters`` steps if a hold is configured,
    then ramps linearly over the following ``shooting_reward_scale_ramp_iters`` steps."""
    _, ramp_iters, hold_iters = _target_per_motion_ramp_hold()
    if ramp_iters <= 0:
        return 1.0
    step = float(getattr(env, "common_step_counter", ramp_iters + hold_iters)) - hold_iters
    return min(1.0, max(0.0, step / ramp_iters))


def current_w_g(env: Any) -> torch.Tensor:
    """The LIVE (possibly still-holding-at-zero or still-ramping) shooting_reward_scale value for
    this step, PER ENV -- shape [num_envs]. Multiply raw shooting-reward-term outputs by this
    instead of relying on a config-time-baked constant or a process-wide scalar; see
    config_values/unified/g1/reward.py's shooting-term weights (now just the term's own relative
    scale k, with w_g folded in here instead of at config-import time) for why this had to move
    off config-time baking in the first place -- which skill an env is running is only known at
    ENV-RUNTIME (via motion_ids), never at config-import time.

    Each env's target is gathered from its assigned skill via motion_ids, modulo the number of
    configured targets -- in legacy (non-N-skill) mode there's exactly one target, so every env
    reads the same value (index 0 for every motion_id, via the modulo), reproducing the old
    single-float behavior exactly, just as a same-valued tensor instead of a bare float."""
    targets, _, _ = _target_per_motion_ramp_hold()
    motion_command = env.command_manager.get_state("motion_command")
    motion_ids = motion_command.motion_ids
    targets_t = torch.tensor(targets, dtype=torch.float32, device=motion_ids.device)
    per_env_target = targets_t[motion_ids % targets_t.shape[0]]
    return per_env_target * ramp_progress(env)
