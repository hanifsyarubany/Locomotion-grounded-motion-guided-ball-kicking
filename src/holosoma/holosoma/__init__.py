"""holosoma package bootstrap.

2026-08-14: if HOLOSOMA_TASK_CONFIG isn't already set explicitly, derive it from
HOLOSOMA_SKILLS_CONFIG's own file instead -- each `motion_skill_N` block there may declare its own
`task_config: task_config_stageX` (no .yaml extension, resolved against this fork's own configs/
directory), letting a single skills.yaml record which task config it was designed for, rather than
requiring a second, separately-tracked env var on every launch (see configs/skills.example.yaml's
own `base_robot`/`task_config` fields for the target shape this reads).

2026-08-15: a NEW top-level `task_config:` field (sibling to `base_robot:`, NOT inside any
motion_skill_N block) may also be set directly in the skills yaml. If present, it always wins as
the GLOBAL config (every field this derivation resolves) -- regardless of what individual skills
declare, and regardless of whether skills agree with each other. This is what makes "simultaneous
per-skill task configs" (2+ skills genuinely training under different task configs at once)
possible: skills disagreeing on task_config is no longer a hard error as long as the top-level
field says which one governs global settings (deadzones, mechanism flags, l2sp_weight, kick_gamma,
etc. -- everything that ISN'T a per-term reward weight). Per-skill divergence in the 5 reward-
tuning categories (motion_tracking_reward/shooting_reward/kick_recovery_posture_reward/
kick_safety_reward/kick_alive_reward) is handled SEPARATELY and per-env, in
config_values/unified/g1/reward.py's `_apply_per_skill_reward_weight_overrides` -- see that
function's own docstring. This module only ever resolves ONE HOLOSOMA_TASK_CONFIG value; it has no
awareness of per-skill reward weights at all.

An EXPLICIT HOLOSOMA_TASK_CONFIG always wins over this derivation -- this is a pure convenience
fallback for when it's unset, never an override of something the caller deliberately set. Every
existing launch command / script in this project that already exports HOLOSOMA_TASK_CONFIG
continues to work completely unchanged.

Runs unconditionally on first `import holosoma` (or any `from holosoma... import`), before any
OTHER holosoma module is imported -- load-bearing: config_types/reward_tuning.py and
config_types/multi_skill.py both read HOLOSOMA_TASK_CONFIG as MODULE-LEVEL code (executed the
instant either module is first imported, not deferred inside a function called later), so this
derivation must complete before either of those modules gets imported for the first time. That
non-negotiably means living here, in the package's own __init__.py -- NOT inside
train_agent.py's main(): by the time main()'s body starts running, the `from holosoma.config_
values...` imports earlier in that same file (near its own top) have already triggered both
modules' own env-var reads, too late for a derived value to matter.
"""

from __future__ import annotations

import os
from pathlib import Path

from holosoma.config_types.task_config_paths import resolve_task_config_path

_TASK_CONFIG_ENV_VAR = "HOLOSOMA_TASK_CONFIG"
_SKILLS_CONFIG_ENV_VAR = "HOLOSOMA_SKILLS_CONFIG"
# 2026-08-16: opt-in acknowledgement that an explicit HOLOSOMA_TASK_CONFIG export is MEANT to
# override what the skills yaml itself declares. See _derive_task_config_from_skills_yaml's
# mismatch guard for why silent override was removed.
_TASK_CONFIG_OVERRIDE_ENV_VAR = "HOLOSOMA_TASK_CONFIG_OVERRIDE"
_TRUTHY = {"1", "true", "yes", "on"}


def _derive_task_config_from_skills_yaml() -> None:
    """Resolve HOLOSOMA_TASK_CONFIG from the skills yaml's own `task_config:` declaration.

    2026-08-16 -- MISMATCH GUARD (behavior change): this used to early-return the instant
    HOLOSOMA_TASK_CONFIG was set, so an explicit export silently won and the skills yaml's own
    declaration was never even read. That cost a full day of GPU time: two `stageB-skill2` runs
    (20260816_144423, 20260816_193616) were launched with `configs/skill2.yaml` declaring
    `task_config: task_config_stageB`, but a STALE `HOLOSOMA_TASK_CONFIG=configs/task_config_stageD.yaml`
    export -- left in the shell by an earlier, unrelated launch in the same terminal -- overrode it.
    Both runs silently trained under Stage D (ball rewards ON at shooting_reward_scale 0.5,
    mid_episode_kick_entry_prob 0.3, threshold 0.35) instead of the intended Stage B tracking-only
    bootstrap (ball rewards 0.0, no handoff). Nothing in the logs, the run name (`--training.name
    stageB-skill2`), or the console said otherwise -- the mistake was only visible by grepping the
    saved holosoma_config.yaml. An intended A/B on bad_motion_body_pos_threshold never happened:
    both runs used the same value.

    Now: the declaration is ALWAYS resolved, and a disagreement with an explicit export is a hard
    error unless HOLOSOMA_TASK_CONFIG_OVERRIDE is truthy. Deliberate overrides stay possible (they
    are a real workflow -- e.g. multi_skills.yaml declares task_config_stageC1 while a Stage D
    experiment exports task_config_stageD), they just have to be stated rather than inherited.
    """
    skills_path_str = os.environ.get(_SKILLS_CONFIG_ENV_VAR)
    explicit = os.environ.get(_TASK_CONFIG_ENV_VAR)

    if not skills_path_str:
        return  # legacy single-file mode (with or without an explicit export) -- unaffected

    skills_path = Path(skills_path_str)
    if not skills_path.exists():
        return  # let the real loader raise its own, better-contextualized FileNotFoundError later

    import yaml

    with open(skills_path) as f:
        raw = yaml.safe_load(f) or {}

    # Top-level `task_config:` (sibling to `base_robot:`) always wins as the GLOBAL config when
    # present -- explicit and unambiguous, regardless of what any individual skill declares.
    top_level_name = raw.get("task_config")
    if top_level_name is not None:
        name = str(top_level_name)
    else:
        refs = {
            str(block["task_config"])
            for key, block in raw.items()
            if key.startswith("motion_skill_") and isinstance(block, dict) and "task_config" in block
        }
        if not refs:
            return  # no skill declares one, no top-level field either -- legacy behavior, unset
        if len(refs) > 1 and explicit:
            # Explicit export resolves the ambiguity that would otherwise be a hard error below --
            # there is no single declared value to disagree with, so nothing to guard against.
            return
        if len(refs) > 1:
            raise ValueError(
                f"{skills_path}: motion_skill_N blocks declare different task_config values "
                f"({sorted(refs)}), and no top-level `task_config:` field resolves which one governs "
                "GLOBAL settings (deadzones, mechanism flags, l2sp_weight, kick_gamma, etc.). Add a "
                "top-level `task_config: task_config_stageX` to this skills yaml to designate one "
                "(per-skill divergence in the 5 reward-tuning categories is still fully honored "
                "separately, per env -- see config_values/unified/g1/reward.py's "
                "_apply_per_skill_reward_weight_overrides). Or set HOLOSOMA_TASK_CONFIG explicitly "
                "to bypass this derivation entirely."
            )
        (name,) = refs

    resolved = resolve_task_config_path(name)
    if not resolved.exists():
        source = "top-level `task_config:`" if top_level_name is not None else "motion_skill block(s)"
        raise FileNotFoundError(
            f"{skills_path}: {source} declare task_config: {name!r}, which resolved to {resolved} -- "
            "that file doesn't exist. task_config names are the target yaml's own basename with no "
            ".yaml extension and no path (e.g. 'task_config_stageC1'), always resolved against this "
            "fork's own configs/ directory."
        )

    if explicit:
        # Compare canonically: the export is typically a relative path ("configs/task_config_X.yaml")
        # while `resolved` is absolute, so a raw string compare would report a false mismatch.
        explicit_canon = Path(explicit).resolve()
        if explicit_canon == resolved.resolve():
            return  # agree -- nothing to do, and nothing to warn about
        if os.environ.get(_TASK_CONFIG_OVERRIDE_ENV_VAR, "").strip().lower() in _TRUTHY:
            print(
                f"[holosoma] {_TASK_CONFIG_ENV_VAR} override acknowledged: using {explicit_canon} "
                f"instead of {resolved} (declared by {skills_path}). "
                f"{_TASK_CONFIG_OVERRIDE_ENV_VAR} is set."
            )
            return
        source = "top-level `task_config:`" if top_level_name is not None else "motion_skill block(s)"
        raise ValueError(
            f"HOLOSOMA_TASK_CONFIG mismatch -- refusing to start.\n"
            f"  {_SKILLS_CONFIG_ENV_VAR} = {skills_path}\n"
            f"    its {source} declares task_config: {name!r}\n"
            f"    -> {resolved}\n"
            f"  {_TASK_CONFIG_ENV_VAR} (explicitly exported) = {explicit}\n"
            f"    -> {explicit_canon}\n"
            "\n"
            "These select DIFFERENT global settings (reward weights, termination thresholds, "
            "mechanism flags like mid_episode_kick_entry_prob). The export wins, so the skills "
            "yaml's own declaration would be silently ignored.\n"
            "\n"
            "Most likely cause: a STALE export left in this shell by an earlier, unrelated launch "
            f"(`export {_TASK_CONFIG_ENV_VAR}=...`). Note the run name does NOT affect which config "
            "is used.\n"
            "\n"
            "Fix one of:\n"
            f"  unset {_TASK_CONFIG_ENV_VAR}            # use what the skills yaml declares (usual fix)\n"
            f"  export {_TASK_CONFIG_OVERRIDE_ENV_VAR}=1   # keep the export, acknowledging it overrides\n"
        )

    os.environ[_TASK_CONFIG_ENV_VAR] = str(resolved)


_derive_task_config_from_skills_yaml()
