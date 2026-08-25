#!/usr/bin/env python3
"""Prints exactly which reward terms will be ACTIVE for kick-mode envs under the current
configs/ball.yaml (and any --command.setup_terms... overrides), before spending GPU time on a
training run.

Why this exists: RewardManager skips any term whose weight is exactly 0.0
(_initialize_terms in managers/reward/manager.py), so "shooting_reward_scale: 0.0" in
configs/ball.yaml really does fully disable the six shooting-task terms (managers/reward/terms/
shooting.py) -- not just numerically, they're never even evaluated. But that guarantee only holds
if the yaml file actually has the values you think it does at the moment training starts (it's
read fresh via load_ball_config(), no caching). A prior training run intended as "Stage B: pure
kick-clip tracking, no shooting adaptation" was accidentally launched with configs/ball.yaml still
holding Stage C's values (shooting_reward_scale=0.8, full ball/target randomization) -- silently
training motion-tracking and full shooting-adaptation pressure simultaneously, which is a
substantially harder combined problem and is the leading suspect for that run's degraded post-kick
stabilization (kick_ball_proximity in particular remains active for the ENTIRE kick episode,
including the recovery-to-neutral-stance + hold tail, directly rewarding the foot for staying near
the ball instead of returning to a stable stance).

Run this BEFORE every training launch, not just once -- it reads the same live config path
train_agent.py does, so it never goes stale.

Usage:
    python src/holosoma/holosoma/print_active_kick_rewards.py

No GPU/simulator/IsaacSim needed -- reward preset construction is plain Python/dataclass
composition (verified: config_values/unified/g1/reward.py's only non-trivial import is
managers/reward/terms/{locomotion,wbt,shooting}.py, none of which touch a simulator at import
time, only when actually called with a live `env`).
"""

from __future__ import annotations

import sys


def main() -> None:
    from holosoma.config_types.simulator import DEFAULT_BALL_CONFIG_YAML, load_ball_config
    from holosoma.config_values.unified.g1.reward import g1_29dof_unified_reward

    ball_cfg = load_ball_config()
    active = {name: cfg for name, cfg in g1_29dof_unified_reward.terms.items() if cfg.weight != 0.0}
    kick_terms = {name: cfg for name, cfg in active.items() if cfg.task_mode == "kick"}
    shooting_terms = {name: cfg for name, cfg in kick_terms.items() if ".shooting:" in cfg.func}
    tracking_terms = {name: cfg for name, cfg in kick_terms.items() if name not in shooting_terms}

    # DEFAULT_BALL_CONFIG_YAML reflects HOLOSOMA_BALL_CONFIG if set (see its own docstring) --
    # print the actual resolved path, not a hardcoded "configs/ball.yaml", so this stays a
    # trustworthy check of whichever file training will really read.
    print(f"{DEFAULT_BALL_CONFIG_YAML}: shooting_reward_scale={ball_cfg.shooting_reward_scale}, "
          f"position_randomization={ball_cfg.position_randomization}, "
          f"kick_aim_enabled={ball_cfg.kick_aim_enabled}, "
          f"nominal_bearing_deg={ball_cfg.resolved_nominal_bearing_deg():.2f}")
    print()
    print(f"Active kick-mode (task_mode='kick') reward terms: {len(kick_terms)} total")
    print(f"  -- motion-tracking / regularization / stabilization ({len(tracking_terms)}):")
    for name in sorted(tracking_terms):
        print(f"       {name}  (weight={tracking_terms[name].weight})")
    print(f"  -- shooting-adaptation ({len(shooting_terms)}):")
    for name in sorted(shooting_terms):
        print(f"       {name}  (weight={shooting_terms[name].weight})")

    print()
    if shooting_terms:
        print(
            f"RESULT: shooting-adaptation rewards ARE active ({len(shooting_terms)} terms) -- "
            "this is a Stage C (or later) configuration, NOT a pure kick-tracking Stage B run. "
            f"If you intended Stage B, set shooting_reward_scale to 0.0 in {DEFAULT_BALL_CONFIG_YAML} "
            "and re-run this script to confirm before training."
        )
        sys.exit(1)
    else:
        print(
            "RESULT: zero shooting-adaptation terms active -- kick-mode reward is restricted to "
            "motion-clip tracking and post-kick stabilization only (Stage B semantics confirmed)."
        )
        sys.exit(0)


if __name__ == "__main__":
    main()
