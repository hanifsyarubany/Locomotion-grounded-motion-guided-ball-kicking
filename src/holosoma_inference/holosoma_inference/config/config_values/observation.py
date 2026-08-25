"""Default observation configurations for holosoma_inference.

This module provides pre-configured observation spaces for different
robot types and tasks, converted from the original YAML configurations.
"""

from __future__ import annotations

from holosoma_inference.compat import entry_points
from holosoma_inference.config.config_types.observation import ObservationConfig

# =============================================================================
# Locomotion Observation Configurations
# =============================================================================

loco_g1_29dof = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "base_ang_vel",
            "projected_gravity",
            "command_lin_vel",
            "command_ang_vel",
            "dof_pos",
            "dof_vel",
            "actions",
            "sin_phase",
            "cos_phase",
        ]
    },
    obs_dims={
        "base_lin_vel": 3,
        "base_ang_vel": 3,
        "projected_gravity": 3,
        "command_lin_vel": 2,
        "command_ang_vel": 1,
        "dof_pos": 29,
        "dof_vel": 29,
        "actions": 29,
        "sin_phase": 2,
        "cos_phase": 2,
    },
    obs_scales={
        "base_lin_vel": 2.0,
        "base_ang_vel": 0.25,
        "projected_gravity": 1.0,
        "command_lin_vel": 1.0,
        "command_ang_vel": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 0.05,
        "actions": 1.0,
        "sin_phase": 1.0,
        "cos_phase": 1.0,
    },
    history_length_dict={
        "actor_obs": 1,
    },
)

loco_t1_29dof = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "base_ang_vel",
            "projected_gravity",
            "command_lin_vel",
            "command_ang_vel",
            "dof_pos",
            "dof_vel",
            "actions",
            "sin_phase",
            "cos_phase",
        ]
    },
    obs_dims={
        "base_lin_vel": 3,
        "base_ang_vel": 3,
        "projected_gravity": 3,
        "command_lin_vel": 2,
        "command_ang_vel": 1,
        "dof_pos": 29,
        "dof_vel": 29,
        "actions": 29,
        "sin_phase": 2,
        "cos_phase": 2,
    },
    obs_scales={
        "base_lin_vel": 1.0,  # T1 uses 1.0 (vs G1's 2.0)
        "base_ang_vel": 1.0,  # T1 uses 1.0 (vs G1's 0.25)
        "projected_gravity": 1.0,
        "command_lin_vel": 1.0,
        "command_ang_vel": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 0.1,  # T1 uses 0.1 (vs G1's 0.05)
        "actions": 1.0,
        "sin_phase": 1.0,
        "cos_phase": 1.0,
    },
    history_length_dict={
        "actor_obs": 1,
    },
)


# =============================================================================
# WBT (Whole Body Tracking) Observation Configurations
# =============================================================================

wbt = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "motion_command",
            "motion_ref_ori_b",
            "base_ang_vel",
            "dof_pos",
            "dof_vel",
            "actions",
        ]
    },
    obs_dims={
        "motion_command": 58,
        "motion_ref_pos_b": 3,
        "motion_ref_ori_b": 6,
        "base_lin_vel": 3,
        "base_ang_vel": 3,
        "dof_pos": 29,
        "dof_vel": 29,
        "actions": 29,
    },
    obs_scales={
        "actions": 1.0,
        "motion_command": 1.0,
        "motion_ref_pos_b": 1.0,
        "motion_ref_ori_b": 1.0,
        "base_lin_vel": 1.0,
        "base_ang_vel": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 1.0,
        "robot_body_pos_b": 1.0,
        "robot_body_ori_b": 1.0,
    },
    history_length_dict={
        "actor_obs": 1,
    },
)

# =============================================================================
# Unified Locomotion + Ball-Kicking Observation Configuration (G1 only)
# =============================================================================
#
# Single-policy observation space merging locomotion (velocity-tracking) and kick
# (motion-tracking) terms, exactly matching UnifiedManager's training-side
# config_values/unified/g1/observation.py: every locomotion term prefixed "loco_",
# every kick term prefixed "kick_" (both source term sets reused several names, e.g.
# "base_ang_vel"/"dof_pos"/"dof_vel"/"actions", so prefixing avoids collisions), plus
# a bare "task_mode_onehot" term (no prefix, always present) that tells the policy
# which mode it's in. Term order below doesn't matter for correctness — both the
# training-side ObservationManager and this package's BasePolicy independently sort
# term names alphabetically before concatenating (verified in both codebases) — but
# every name/dim/scale must match training exactly, since dims/scales are looked up
# by name, not position.
unified_g1_29dof = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "loco_base_ang_vel",
            "loco_projected_gravity",
            "loco_command_lin_vel",
            "loco_command_ang_vel",
            "loco_dof_pos",
            "loco_dof_vel",
            "loco_actions",
            "loco_sin_phase",
            "loco_cos_phase",
            "kick_motion_command",
            "kick_motion_ref_ori_b",
            "kick_base_ang_vel",
            "kick_dof_pos",
            "kick_dof_vel",
            "kick_actions",
            "kick_ball_pos_b",
            "kick_target_pos_b",
            "task_mode_onehot",
        ]
    },
    obs_dims={
        "loco_base_ang_vel": 3,
        "loco_projected_gravity": 3,
        "loco_command_lin_vel": 2,
        "loco_command_ang_vel": 1,
        "loco_dof_pos": 29,
        "loco_dof_vel": 29,
        "loco_actions": 29,
        "loco_sin_phase": 2,
        "loco_cos_phase": 2,
        "kick_motion_command": 58,
        "kick_motion_ref_ori_b": 6,
        "kick_base_ang_vel": 3,
        "kick_dof_pos": 29,
        "kick_dof_vel": 29,
        "kick_actions": 29,
        # Added in training v7 (RoboNaldo-style shooting rewards, see
        # locomotion_and_ball_kicking/src/holosoma/README.md) -- ANY checkpoint trained after
        # that change has a 261-dim actor_obs input (was 256), so these two terms are REQUIRED
        # here even for locomotion-only (Stage A/B) checkpoints, since the training-side
        # observation config is identical across all three bootstrap stages -- only whether
        # kick-mode envs ever get sampled differs, not the tensor shape. Omitting these terms
        # produces an ONNX shape-mismatch error ("Expected: 261, Got: 256") the moment
        # setup_policy's warmup call runs, before the policy ever ticks.
        "kick_ball_pos_b": 3,
        "kick_target_pos_b": 2,
        "task_mode_onehot": 2,
    },
    obs_scales={
        "loco_base_ang_vel": 0.25,
        "loco_projected_gravity": 1.0,
        "loco_command_lin_vel": 1.0,
        "loco_command_ang_vel": 1.0,
        "loco_dof_pos": 1.0,
        "loco_dof_vel": 0.05,
        "loco_actions": 1.0,
        "loco_sin_phase": 1.0,
        "loco_cos_phase": 1.0,
        "kick_motion_command": 1.0,
        "kick_motion_ref_ori_b": 1.0,
        "kick_base_ang_vel": 1.0,
        "kick_dof_pos": 1.0,
        "kick_dof_vel": 1.0,
        "kick_actions": 1.0,
        "kick_ball_pos_b": 1.0,
        "kick_target_pos_b": 1.0,
        "task_mode_onehot": 1.0,
    },
    history_length_dict={
        "actor_obs": 1,
    },
)

# Pre-v7 (256-dim) unified observation space, for checkpoints trained before RoboNaldo-style
# shooting rewards added kick_ball_pos_b/kick_target_pos_b (e.g.
# locomotion-policy-v7-kick-locomotion/model_0118000.onnx). Derived from unified_g1_29dof by
# dropping exactly those two terms -- both sides sort term names alphabetically before
# concatenating (see unified_g1_29dof's docstring), and removing entries from an alphabetically
# sorted list can't reorder the ones that remain, so this reconstructs the old checkpoint's exact
# 256-dim layout term-for-term. Selected via --task.use-previous-version-policy (see
# run_policy.py's main()), not through the normal observation:<preset> CLI path.
_LEGACY_UNIFIED_DROPPED_TERMS = frozenset({"kick_ball_pos_b", "kick_target_pos_b"})

unified_g1_29dof_legacy = ObservationConfig(
    obs_dict={
        group: [t for t in terms if t not in _LEGACY_UNIFIED_DROPPED_TERMS]
        for group, terms in unified_g1_29dof.obs_dict.items()
    },
    obs_dims=unified_g1_29dof.obs_dims,
    obs_scales=unified_g1_29dof.obs_scales,
    history_length_dict=unified_g1_29dof.history_length_dict,
)

# =============================================================================
# Default Configurations Dictionary
# =============================================================================

DEFAULTS = {
    "loco-g1-29dof": loco_g1_29dof,
    "loco-t1-29dof": loco_t1_29dof,
    "wbt": wbt,
    "unified-g1-29dof": unified_g1_29dof,
}
"""Dictionary of all available observation configurations.

Keys use hyphen-case naming convention for CLI compatibility.
"""

# Auto-discover observation configs from installed extensions
for ep in entry_points(group="holosoma.config.observation"):
    DEFAULTS[ep.name] = ep.load()
