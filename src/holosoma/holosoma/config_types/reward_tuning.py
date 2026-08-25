"""Per-term reward-WEIGHT and reward-SIGMA overrides for the kick task, loaded from a standalone
yaml (``configs/kicking_motion_reward_tuning.yaml`` by default).

This is one level more granular than the per-skill, per-CATEGORY reward scales already built for
this project (``motion_tracking_reward_scale``, ``shooting_reward_scale``,
``kick_recovery_posture_reward_scale``, ``kick_safety_reward_scale``, ``kick_alive_reward_scale``
-- see ``utils/kick_reward_scales.py``): those multiply an entire category uniformly, per skill,
at runtime. This mechanism instead overrides each individual reward TERM's own static
``RewardTermCfg.weight`` -- e.g. ``kick_ball_velocity`` vs ``kick_goal_success_burst`` within the
shooting category -- resolved once at config-import time (see
``config_values/unified/g1/reward.py``'s ``_apply_reward_weight_overrides``), the same simple
mechanism ``kick_contact_force_penalty_floor``/``_k`` already use for their own single global
knobs. No per-skill/per-env runtime resolution is needed here (unlike the category scales): a
term's WEIGHT (and SIGMA, see below) is the same static value for every skill, so a config-time
float is sufficient -- any per-skill/live behavior (e.g. shooting's ``current_w_g(env)`` ramp)
happens separately, inside each term's own function, multiplying this weight further at runtime.

SIGMA overrides (added 2026-07-29, alongside weight): several terms use an
``exp(-error^2 / sigma^2)`` (or similar) kernel -- the WIDTH of that kernel, not just the
multiplier in front of it, determines whether the term produces any usable gradient at the error
magnitudes actually being produced. A too-narrow sigma relative to current typical error goes
gradient-flat everywhere except very close to zero error, no matter how large weight is (see
``kick_error_ball_to_target``'s own analysis: at sigma=1.0 and a typical 4.2m miss,
exp(-4.2^2/1^2) ~ 6e-8 -- multiplying that by any realistic weight is still ~0). Each category's
``_sigma`` sub-block (a reserved key, not a term name) lists ONLY the terms in that category that
actually accept a ``sigma`` parameter -- overriding sigma for a term that doesn't have one is a
config-time error (see ``_apply_reward_sigma_overrides`` in reward.py), not a silent no-op or a
confusing runtime TypeError.

Deliberately loaded UNCONDITIONALLY (no ``HOLOSOMA_SKILLS_CONFIG``-style opt-in gate) -- unlike
N-skill mode, this is meant to apply automatically to every training run of this project, single-
skill or multi-skill alike, mirroring ``DEFAULT_BALL_CONFIG_YAML``'s always-read-unless-overridden
convention (see ``config_types/simulator.py``).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from holosoma.config_types.multi_skill import HOLOSOMA_TASK_CONFIG_ENV_VAR

# Mirrors this project's own 5-category reward-scale split (utils/kick_reward_scales.py) --
# each category's yaml section lists that category's member reward-term names -> weight, plus an
# optional reserved "_sigma" sub-key (see SIGMA_SUBSECTION_KEY) for sigma overrides.
REWARD_TUNING_CATEGORIES = (
    "motion_tracking_reward",
    "shooting_reward",
    "kick_recovery_posture_reward",
    "kick_safety_reward",
    "kick_alive_reward",
)

# Reserved key within each category's yaml block -- NOT a reward-term name, so both loaders below
# must skip/handle it specially rather than treating it as one more term: term entries -> {"weight"
# loader}, "_sigma" entries -> {"sigma" loader}. Leading underscore deliberately mirrors this
# project's own "not a real field" convention and can never collide with a real term name (reward
# term names are plain identifiers from config_values/unified/g1/reward.py, never underscore-led).
SIGMA_SUBSECTION_KEY = "_sigma"

HOLOSOMA_REWARD_TUNING_CONFIG_ENV_VAR = "HOLOSOMA_REWARD_TUNING_CONFIG"

# 2026-08-05, 2-file config split: HOLOSOMA_TASK_CONFIG (see config_types/multi_skill.py's own
# docstring for the full design) takes priority over HOLOSOMA_REWARD_TUNING_CONFIG when both are
# set -- the merged task-config file carries the 5 reward-term-weight sections directly, so a
# separate dedicated reward-tuning file is redundant once that mode is on. Fork root, same
# convention as DEFAULT_BALL_CONFIG_YAML / DEFAULT_MULTI_SKILL_CONFIG_YAML -- always read from
# the hardcoded default unless overridden via one of the two env vars above.
DEFAULT_REWARD_TUNING_CONFIG_YAML = Path(
    os.environ.get(HOLOSOMA_TASK_CONFIG_ENV_VAR)
    or os.environ.get(HOLOSOMA_REWARD_TUNING_CONFIG_ENV_VAR)
    or Path(__file__).resolve().parents[4] / "configs" / "kicking_motion_reward_tuning.yaml"
)

# False (lenient) iff the resolved DEFAULT_REWARD_TUNING_CONFIG_YAML above came from
# HOLOSOMA_TASK_CONFIG -- a MERGED file expected to carry many other non-reward-tuning top-level
# keys (ball, ood_*, bad_tracking_*, kick_gamma, ...) that _load_raw must NOT treat as
# "unrecognized section" errors. True (strict, today's exact behavior, catches a typo'd section
# name immediately) for a dedicated reward-tuning-only file, whether that's the hardcoded default
# or an explicit HOLOSOMA_REWARD_TUNING_CONFIG override. Resolved once at import time, same
# before-tyro-CLI-parsing discipline as every other env-var-gated default in this codebase.
DEFAULT_STRICT_TOP_LEVEL = not bool(os.environ.get(HOLOSOMA_TASK_CONFIG_ENV_VAR))


def _load_raw(yaml_path: str | Path, strict_top_level: bool = True) -> dict[str, Any]:
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        return {}

    with open(yaml_path) as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    if strict_top_level:
        unknown_sections = set(raw.keys()) - set(REWARD_TUNING_CATEGORIES)
        if unknown_sections:
            raise ValueError(
                f"{yaml_path}: unrecognized top-level section(s) {sorted(unknown_sections)} -- "
                f"expected only {list(REWARD_TUNING_CATEGORIES)}"
            )
    return raw


def load_reward_weight_overrides(
    yaml_path: str | Path = DEFAULT_REWARD_TUNING_CONFIG_YAML,
    strict_top_level: bool = DEFAULT_STRICT_TOP_LEVEL,
) -> dict[str, float]:
    """Flatten the yaml's 5 category sections into a single ``{term_name: weight}`` dict.

    Missing file -> empty dict (no overrides at all) -- deliberately non-fatal, so this mechanism
    stays a safe no-op if the file is ever deleted or the path is misconfigured, rather than
    breaking every training run. A term name appearing in more than one category section is a real
    yaml authoring mistake and raises immediately (cheap to catch here, confusing to debug later as
    a silently-wrong reward magnitude). An unrecognized top-level section name ALSO raises, but
    only when ``strict_top_level`` is True (the default for a dedicated reward-tuning file) --
    False (the default when reading from a merged HOLOSOMA_TASK_CONFIG file, see
    DEFAULT_STRICT_TOP_LEVEL above) tolerates the file's many other legitimate non-reward keys.
    The reserved ``_sigma`` sub-key within a category (see ``load_reward_sigma_overrides``) is
    skipped here, not treated as a term named ``_sigma``.

    Does NOT validate term names against the actual registered reward terms -- this module can't
    see config_values/unified/g1/reward.py's final term dict without a circular import (that
    module imports FROM here). That cross-check happens where it belongs: in reward.py itself,
    against the real, fully-assembled terms dict, at config-import time -- see
    ``_apply_reward_weight_overrides``.
    """
    raw = _load_raw(yaml_path, strict_top_level=strict_top_level)

    overrides: dict[str, float] = {}
    seen_in: dict[str, str] = {}
    for category in REWARD_TUNING_CATEGORIES:
        section = raw.get(category) or {}
        if not isinstance(section, dict):
            raise ValueError(
                f"{yaml_path}: '{category}' must be a mapping of term_name: weight, got {type(section).__name__}"
            )
        for term_name, weight in section.items():
            if term_name == SIGMA_SUBSECTION_KEY:
                continue
            if term_name in seen_in:
                raise ValueError(
                    f"{yaml_path}: term {term_name!r} appears in both {seen_in[term_name]!r} and "
                    f"{category!r} -- each term belongs to exactly one category."
                )
            seen_in[term_name] = category
            overrides[term_name] = float(weight)

    return overrides


def load_per_skill_reward_weight_overrides(
    task_config_paths: list[str | Path | None],
) -> list[dict[str, float]]:
    """``load_reward_weight_overrides``, once per skill, for "simultaneous per-skill task configs"
    (2026-08-15, see config_values/unified/g1/reward.py's
    ``_apply_per_skill_reward_weight_overrides`` for the full design/motivation).

    ``task_config_paths[i]`` is skill ``i``'s own resolved task_config yaml path (or ``None`` if
    that skill declared no ``task_config:`` at all, meaning it simply inherits whatever the
    GLOBAL/top-level config says for every term -- same "no override" meaning a missing file
    already has for the single-file case). Always loaded with ``strict_top_level=False``: these
    are full task_config files carrying many non-reward-tuning keys, same as the global loader
    does whenever it's reading from a merged HOLOSOMA_TASK_CONFIG file rather than a
    dedicated reward-tuning-only yaml.

    Returns one ``{term_name: weight}`` dict per skill, same order as ``task_config_paths``. Pure
    reuse of ``load_reward_weight_overrides`` -- no new parsing logic, so a term-appears-twice or
    malformed-section error in any one skill's file raises with that loader's own, already
    file-path-qualified message.
    """
    return [
        load_reward_weight_overrides(p, strict_top_level=False) if p is not None else {}
        for p in task_config_paths
    ]


def load_per_skill_top_level_override(
    task_config_paths: list[str | Path | None], field_name: str
) -> list[Any]:
    """For each skill's own task_config file, read a single TOP-LEVEL scalar field (e.g.
    ``kick_recovery_stand_height_deadzone``, ``kick_contact_force_penalty_floor``) if present,
    else ``None`` for that skill -- meaning "no override, inherit whatever the GLOBAL
    (HOLOSOMA_TASK_CONFIG-resolved) config says for this field", same fallback meaning ``None``
    already has throughout this "simultaneous per-skill task configs" mechanism (see
    config_values/unified/g1/reward.py's ``_apply_per_skill_reward_weight_overrides`` for the
    sibling reward-WEIGHT mechanism this generalizes).

    Unlike ``load_per_skill_reward_weight_overrides``, this is NOT limited to the 5 nested
    reward-tuning categories -- it reads a field living at the yaml's own top level, which is
    where deadzones/thresholds/mechanism flags/contact-force shape params all live in a real
    task_config file. Reuses ``_load_raw`` with ``strict_top_level=False`` (a full task_config
    file carries many keys this function doesn't care about) -- no new parsing logic.
    """
    return [_load_raw(p, strict_top_level=False).get(field_name) if p is not None else None for p in task_config_paths]


def resolve_per_skill_param(
    skill_task_config_paths: list[str | Path | None] | None, field_name: str, base_value: Any
) -> list[Any] | None:
    """Resolve a single top-level scalar field to a per-skill ``[n_skills]`` list, IF skills
    genuinely diverge on it -- shared algorithm behind config_values/unified/g1/reward.py's
    ``_per_skill_param`` and termination.py's own use for ``kick_recovery_drift_deadzone`` (kept
    here, not duplicated in each config_values module, since the two need IDENTICAL no-op/
    fallback semantics -- see ``load_per_skill_top_level_override``'s own docstring for the
    field-reading half this wraps).

    Returns ``None`` (no table needed -- caller should leave the plain scalar/params dict alone)
    in every case that must stay an exact no-op: ``skill_task_config_paths`` is ``None`` (no
    multi-skill mode, or a module that hasn't resolved one), 0/1 skills, skills agreeing on the
    same task_config file, or skills whose files happen to produce the SAME resolved value for
    this specific field despite being different files.

    Each skill's own value is: its task_config file's own setting for ``field_name`` if present,
    else ``base_value`` (the caller's already-resolved GLOBAL/HOLOSOMA_TASK_CONFIG value) -- same
    "no override -> inherit the global" fallback used throughout this "simultaneous per-skill task
    configs" mechanism.
    """
    if skill_task_config_paths is None or len(set(skill_task_config_paths)) <= 1:
        return None
    raw_per_skill = load_per_skill_top_level_override(skill_task_config_paths, field_name)
    resolved = [v if v is not None else base_value for v in raw_per_skill]
    return resolved if len(set(resolved)) > 1 else None


def load_reward_sigma_overrides(
    yaml_path: str | Path = DEFAULT_REWARD_TUNING_CONFIG_YAML,
    strict_top_level: bool = DEFAULT_STRICT_TOP_LEVEL,
) -> dict[str, float]:
    """Flatten each category's reserved ``_sigma`` sub-block into a single ``{term_name: sigma}``
    dict -- e.g. ``shooting_reward._sigma.kick_error_ball_to_target: 4.0``.

    Same missing-file/error-handling/strict_top_level contract as ``load_reward_weight_overrides``.
    Only terms that actually accept a ``sigma`` parameter belong under a category's ``_sigma`` key;
    this loader only flattens the yaml structure, it does NOT know which real reward terms have a
    sigma parameter (same circular-import reason as the weight loader) -- that check happens in
    ``_apply_reward_sigma_overrides`` in reward.py, against the real term param dicts.
    """
    raw = _load_raw(yaml_path, strict_top_level=strict_top_level)

    overrides: dict[str, float] = {}
    seen_in: dict[str, str] = {}
    for category in REWARD_TUNING_CATEGORIES:
        section = raw.get(category) or {}
        if not isinstance(section, dict):
            continue  # already raised above in load_reward_weight_overrides's own pass, if malformed
        sigma_section = section.get(SIGMA_SUBSECTION_KEY) or {}
        if not isinstance(sigma_section, dict):
            raise ValueError(
                f"{yaml_path}: '{category}.{SIGMA_SUBSECTION_KEY}' must be a mapping of "
                f"term_name: sigma, got {type(sigma_section).__name__}"
            )
        for term_name, sigma in sigma_section.items():
            if term_name in seen_in:
                raise ValueError(
                    f"{yaml_path}: sigma override for term {term_name!r} appears under both "
                    f"{seen_in[term_name]!r} and {category!r} -- each term belongs to exactly one category."
                )
            seen_in[term_name] = category
            overrides[term_name] = float(sigma)

    return overrides


__all__ = [
    "REWARD_TUNING_CATEGORIES",
    "SIGMA_SUBSECTION_KEY",
    "HOLOSOMA_REWARD_TUNING_CONFIG_ENV_VAR",
    "DEFAULT_REWARD_TUNING_CONFIG_YAML",
    "DEFAULT_STRICT_TOP_LEVEL",
    "load_reward_weight_overrides",
    "load_reward_sigma_overrides",
]
