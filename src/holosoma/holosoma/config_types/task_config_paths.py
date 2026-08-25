"""Pure ``task_config: task_config_stageX`` name -> absolute file path resolution.

Deliberately its own tiny, side-effect-free module (no env-var reads at import time, no other
holosoma imports): both ``holosoma/__init__.py``'s bootstrap (which MUST run and complete before
``config_types/multi_skill.py``/``config_types/reward_tuning.py`` are first imported, since those
read ``HOLOSOMA_TASK_CONFIG``/``HOLOSOMA_SKILLS_CONFIG`` as module-level code -- see that file's
own docstring) and ``config_values/unified/g1/reward.py``'s per-skill reward-weight override
loading (2026-08-15, see that module's own ``_apply_per_skill_reward_weight_overrides``) need the
IDENTICAL name -> path resolution rule, and importing anything heavier from the bootstrap path
would risk exactly the premature-env-var-read problem this module exists to avoid.
"""

from __future__ import annotations

from pathlib import Path

# This file lives at <fork_root>/src/holosoma/holosoma/config_types/task_config_paths.py --
# parents[4] is <fork_root>. Same depth/convention as config_types/multi_skill.py's own
# DEFAULT_MULTI_SKILL_CONFIG_YAML (also parents[4], also config_types/*.py).
_FORK_ROOT = Path(__file__).resolve().parents[4]


def resolve_task_config_path(name: str) -> Path:
    """``'task_config_stageC1'`` -> ``<fork_root>/configs/task/task_config_stageC1.yaml``.

    ``name`` is always a bare basename with no path and no ``.yaml`` extension (the shape every
    ``task_config:`` field in this project's skills yamls uses), always resolved against this
    fork's own ``configs/task/`` directory (2026-08-23: all ``task_config_*.yaml`` files were
    moved into that subdirectory -- see ``configs/skill/`` for the matching per-skill move) --
    never cwd-relative, never skills-yaml-path-relative.
    """
    return _FORK_ROOT / "configs" / "task" / f"{name}.yaml"
