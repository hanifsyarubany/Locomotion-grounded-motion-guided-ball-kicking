"""Per-skill runtime reward-CATEGORY scale multipliers, configurable via stageB_and_C.yaml:
motion_tracking_reward_scale, recovery_tracking_scale, kick_recovery_posture_reward_scale,
kick_safety_reward_scale, kick_alive_reward_scale (see SkillConfig in config_types/multi_skill.py
for each field's own docstring).

recovery_tracking_scale is the odd one out in this module: the other four apply uniformly for the
whole kick episode, but recovery_tracking_scale is ADDITIONALLY phase-gated (full no-op during
swing, only takes effect during recovery/hold) -- see managers/reward/terms/kick_scale_wrappers.py,
which is where that gating actually happens (this module only resolves the flat per-skill VALUE,
same as every other category here; it has no phase awareness of its own).

Same architecture as utils/shooting_curriculum.py's current_w_g (per-env resolution via
motion_ids, since which skill an env is running is only known at env-runtime, never at
config-import time -- see that module's docstring for the full rationale), but deliberately
WITHOUT a ramp/hold schedule: these are simple per-skill constants, live from step 1, not a
Stage-B->C curriculum like shooting_reward_scale is. If a ramp/hold ever becomes needed for one of
these, add it there, not here -- don't speculatively carry complexity nothing asked for yet.

Legacy (non-N-skill) mode: every category resolves to a flat 1.0 for every env, i.e. a pure no-op,
reproducing current behavior exactly. This intentionally does NOT read configs/ball.yaml (unlike
shooting_curriculum.py's legacy fallback) -- these 4 scales were requested as an N-skill-only,
stageB_and_C.yaml-only feature; BallConfig was not asked to grow matching fields.
"""

from __future__ import annotations

from typing import Any

import torch

_CATEGORIES = (
    "motion_tracking",
    "root_tracking",
    "recovery_tracking",
    "kick_recovery_posture",
    "kick_safety",
    "kick_alive",
    "kick_alive_pre_kick_ratio",
)

# Cached once per process -- read from a stageB_and_C.yaml gated by HOLOSOMA_SKILLS_CONFIG, set
# before process start and never mutated mid-run (same discipline as shooting_curriculum.py's own
# cache; see that module's docstring for why re-parsing per-call would be real overhead here too,
# since these run from reward TERMS, up to ~10x per control step at 50Hz).
_cached: dict[str, list[float]] | None = None


def _per_skill_scales() -> dict[str, list[float]]:
    global _cached
    if _cached is None:
        from holosoma.config_types.multi_skill import load_multi_skill_config, multi_skill_mode_enabled

        if multi_skill_mode_enabled():
            msc = load_multi_skill_config()
            _cached = {
                "motion_tracking": [sc.motion_tracking_reward_scale for sc in msc.skills],
                "root_tracking": [sc.root_tracking_reward_scale for sc in msc.skills],
                "recovery_tracking": [sc.recovery_tracking_scale for sc in msc.skills],
                "kick_recovery_posture": [sc.kick_recovery_posture_reward_scale for sc in msc.skills],
                "kick_safety": [sc.kick_safety_reward_scale for sc in msc.skills],
                "kick_alive": [sc.kick_alive_reward_scale for sc in msc.skills],
                "kick_alive_pre_kick_ratio": [sc.kick_alive_pre_kick_ratio for sc in msc.skills],
            }
        else:
            _cached = {cat: [1.0] for cat in _CATEGORIES}
    return _cached


def _per_env_scale(env: Any, category: str) -> torch.Tensor:
    """Gather each env's configured category scale via its assigned skill (motion_ids) -- shape
    [num_envs]. In legacy mode there's exactly one target (1.0), so every env reads the same
    value via the modulo, same broadcast-to-scalar-behavior pattern as current_w_g."""
    targets = _per_skill_scales()[category]
    motion_command = env.command_manager.get_state("motion_command")
    motion_ids = motion_command.motion_ids
    targets_t = torch.tensor(targets, dtype=torch.float32, device=motion_ids.device)
    return targets_t[motion_ids % targets_t.shape[0]]


def motion_tracking_reward_scale(env: Any) -> torch.Tensor:
    """Live per-env motion_tracking_reward_scale -- shape [num_envs]. Multiply a kick-mode
    motion-tracking term's raw output by this."""
    return _per_env_scale(env, "motion_tracking")


def root_tracking_reward_scale(env: Any) -> torch.Tensor:
    """Live per-env root_tracking_reward_scale -- shape [num_envs]. Multiply ONLY the 2 root/
    anchor motion-tracking terms' raw output by this, ON TOP OF motion_tracking_reward_scale (not
    instead of it) -- see SkillConfig.root_tracking_reward_scale's own docstring for the full
    rationale. The 5 relative-body tracking terms never read this category."""
    return _per_env_scale(env, "root_tracking")


def recovery_tracking_scale(env: Any) -> torch.Tensor:
    """Live per-env recovery_tracking_scale -- shape [num_envs]. This is the flat per-skill VALUE
    only; the swing-vs-recovery PHASE GATING that decides when it actually applies happens in
    managers/reward/terms/kick_scale_wrappers.py, not here -- this module has no phase awareness
    of its own, matching every other category it resolves."""
    return _per_env_scale(env, "recovery_tracking")


def kick_recovery_posture_reward_scale(env: Any) -> torch.Tensor:
    """Live per-env kick_recovery_posture_reward_scale -- shape [num_envs]. Multiply a
    kick-recovery standing-posture penalty term's raw output by this (additional to, not instead
    of, that term's own _kick_recovery_gate phase-gating)."""
    return _per_env_scale(env, "kick_recovery_posture")


def kick_safety_reward_scale(env: Any) -> torch.Tensor:
    """Live per-env kick_safety_reward_scale -- shape [num_envs]. Multiply a kick-safety penalty
    term's raw output by this (additional to, not instead of, that term's own
    (floor + k * current_w_g(env)) dynamic magnitude)."""
    return _per_env_scale(env, "kick_safety")


def kick_alive_reward_scale(env: Any) -> torch.Tensor:
    """Live per-env kick_alive_reward_scale -- shape [num_envs]. Multiply kick_alive's raw
    (shared with locomotion's own `alive`) output by this."""
    return _per_env_scale(env, "kick_alive")


def kick_alive_pre_kick_ratio(env: Any) -> torch.Tensor:
    """Live per-env kick_alive_pre_kick_ratio -- shape [num_envs]. This is the flat per-skill
    VALUE only; the in_kicking_phase-vs-not PHASE GATING that decides when it actually applies
    happens in managers/reward/terms/kick_scale_wrappers.py::kick_alive_scaled, not here -- same
    "this module has no phase awareness of its own" convention as recovery_tracking_scale."""
    return _per_env_scale(env, "kick_alive_pre_kick_ratio")
