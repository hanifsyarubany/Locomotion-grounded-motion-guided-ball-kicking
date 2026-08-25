"""Headless, non-interactive geometric sanity check: does the kick clip's kicking foot
actually reach the ball's configured spawn position?

Kinematically replays the reference motion clip (same underlying mechanics as
`step_visualize_motion`, minus the viewport-only debug markers, which segfault
headless) and reports the minimum foot-to-ball distance achieved — no policy, no RL,
no viewport needed. Run this before spending GPU time on a kick clip: a clip whose
foot never reaches the ball's configured position will train for hours without
anyone noticing, since the reward signal stays uniformly poor the whole time.

Usage:
    python check_kick_geometry.py --motion-dir /path/to/npz_dir
    python check_kick_geometry.py --motion-file /path/to/clip.npz
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys

RESULT_MARKER_START = "===GEOM_CHECK_RESULT_START==="
RESULT_MARKER_END = "===GEOM_CHECK_RESULT_END==="


def main() -> None:
    parser = argparse.ArgumentParser()
    motion_group = parser.add_mutually_exclusive_group(required=True)
    motion_group.add_argument("--motion-dir", type=str, default=None)
    motion_group.add_argument("--motion-file", type=str, default=None)
    parser.add_argument("--num-envs", type=int, default=32)
    args = parser.parse_args()

    from holosoma.config_values import experiment as experiment_values
    from holosoma.train_agent import get_tyro_env_config, training_context
    from holosoma.utils.common import seeding
    from holosoma.utils.helpers import get_class
    from holosoma.utils.safe_torch_import import torch

    seeding(0)
    device = "cuda:0"
    num_envs = args.num_envs

    base = experiment_values.g1_29dof_unified_fast_sac
    motion_command_cfg = base.command.setup_terms["motion_command"]
    new_params = dict(motion_command_cfg.params)
    motion_config_overrides = {"motion_dir": args.motion_dir} if args.motion_dir else {"motion_file": args.motion_file}
    new_params["motion_config"] = dataclasses.replace(new_params["motion_config"], **motion_config_overrides)
    new_motion_command_cfg = dataclasses.replace(motion_command_cfg, params=new_params)
    new_command = dataclasses.replace(
        base.command, setup_terms={**base.command.setup_terms, "motion_command": new_motion_command_cfg}
    )

    tyro_config = dataclasses.replace(
        base,
        command=new_command,
        training=dataclasses.replace(base.training, num_envs=num_envs, headless=True),
    )

    with training_context(tyro_config):
        tyro_env_config = get_tyro_env_config(tyro_config)
        env = get_class(tyro_config.env_class)(tyro_env_config, device=device)
        env.reset_all()

        motion_command = env.command_manager.get_state("motion_command")
        assert motion_command.has_ball, "This experiment has no ball configured (scene.ball is None)."

        all_env_ids = torch.arange(num_envs, device=device)
        flat = env.terrain_manager.get_state("locomotion_terrain").env_terrain_is_flat
        kick_env_ids = all_env_ids[flat[all_env_ids]]
        print(f"kick-eligible (flat-terrain) envs: {kick_env_ids.numel()}/{num_envs}", flush=True)
        if kick_env_ids.numel() == 0:
            print("No flat-terrain envs — cannot check. Try a larger --num-envs.", flush=True)
            sys.exit(1)

        # Force kick mode + reset to frame 0 (start_at_timestep_zero_prob=1.0 for unified's kick
        # motion config already lands here, this just makes it explicit/guaranteed).
        env.trigger_kick(kick_env_ids)
        motion_command.time_steps[kick_env_ids] = 0

        ball_local_pos = motion_command.ball_reset_state[:3]
        env_origins = env.simulator.env_origins[kick_env_ids]
        ball_world_xyz = env_origins + ball_local_pos.unsqueeze(0)  # (E, 3)

        n = kick_env_ids.numel()
        min_dist = torch.full((n,), float("inf"), device=device)
        min_dist_step = torch.zeros((n,), dtype=torch.long, device=device)
        time_step_total = int(motion_command.motion.time_step_total)
        max_steps = min(time_step_total, 1000)

        env_ids_all = torch.arange(env.num_envs, device=device)
        for step in range(max_steps):
            motion_command.step()
            joint_pos = motion_command.joint_pos.clone()
            joint_vel = motion_command.joint_vel.clone()

            env.simulator.dof_pos[env_ids_all] = joint_pos
            env.simulator.dof_vel[env_ids_all] = joint_vel
            env.simulator.robot_root_states[env_ids_all, :3] = motion_command.root_pos_w.clone()
            env.simulator.robot_root_states[env_ids_all, 3:7] = motion_command.root_quat_w.clone()
            env.simulator.robot_root_states[env_ids_all, 7:10] = motion_command.body_lin_vel_w[:, 0].clone()
            env.simulator.robot_root_states[env_ids_all, 10:13] = motion_command.body_ang_vel_w[:, 0].clone()
            env.simulator.set_actor_root_state_tensor(env_ids_all, env.simulator.all_root_states)
            env.simulator.set_dof_state_tensor(env_ids_all, env.simulator.dof_state)
            env.simulator.scene.write_data_to_sim()
            env.simulator.sim.forward()
            env.simulator.refresh_sim_tensors()

            done = bool(motion_command.time_steps[0].item() >= motion_command.motion.time_step_total - 2)
            feet_xyz = env.simulator._rigid_body_pos[kick_env_ids][:, env.feet_indices, :]  # (E, num_feet, 3)
            dist = (feet_xyz - ball_world_xyz.unsqueeze(1)).norm(dim=-1).min(dim=-1).values  # (E,)
            improved = dist < min_dist
            min_dist_step[improved] = step
            min_dist = torch.minimum(min_dist, dist)
            if done:
                break

        result = {
            "motion_dir_or_file": args.motion_dir or args.motion_file,
            "kick_eligible_envs": int(n),
            "clip_time_step_total": time_step_total,
            "steps_replayed": step + 1,
            "ball_local_pos_xyz": [float(x) for x in ball_local_pos.tolist()],
            "min_foot_ball_dist_mean": float(min_dist.mean().item()),
            "min_foot_ball_dist_min": float(min_dist.min().item()),
            "min_foot_ball_dist_max": float(min_dist.max().item()),
            "min_dist_step_mean": float(min_dist_step.float().mean().item()),
            "frac_envs_within_10cm": float((min_dist < 0.10).float().mean().item()),
            "frac_envs_within_20cm": float((min_dist < 0.20).float().mean().item()),
        }

        print(RESULT_MARKER_START, flush=True)
        print(json.dumps(result, indent=2), flush=True)
        print(RESULT_MARKER_END, flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        print("===EXCEPTION_START===", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        print("===EXCEPTION_END===", flush=True)
        raise
