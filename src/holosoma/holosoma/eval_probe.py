"""Scripted, headless, non-interactive checkpoint probe for the unified locomotion+kick policy.

Built for iterating on training without a human watching a viewport: runs two quantitative
rollouts against a trained checkpoint and prints structured metrics.

1. Zero-velocity standing stability: force locomotion-mode with a (0, 0, 0) velocity command and
   measure how still the robot actually holds (base linear/angular velocity, joint velocity RMS,
   height drift, fall rate).
2. Kick trigger: force kick-mode via trigger_kick() and measure whether the kicking foot actually
   reaches the ball, whether the ball moves, and whether the robot stays upright through the
   post-kick stabilization window.

Usage:
    python src/holosoma/holosoma/eval_probe.py --checkpoint <path/to/model_XXXXXXX.pt> [--num-envs 32]
"""

from __future__ import annotations

import argparse
import json
import sys

import torch
from loguru import logger

from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.utils.eval_utils import CheckpointConfig, load_checkpoint, load_saved_experiment_config
from holosoma.utils.helpers import get_class
from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment

from eval_interactive import _compute_expected_action


def _rollout_metrics(env, num_steps: int) -> dict:
    sim = env.simulator
    lin_vel_norms, ang_vel_norms, joint_vel_rms, heights = [], [], [], []
    for _ in range(num_steps):
        lin_vel_norms.append(sim.robot_root_states[:, 7:10].norm(dim=-1))
        ang_vel_norms.append(sim.robot_root_states[:, 10:13].norm(dim=-1))
        joint_vel_rms.append(sim.dof_vel.pow(2).mean(dim=-1).sqrt())
        heights.append(sim.robot_root_states[:, 2])
    return {
        "lin_vel": torch.stack(lin_vel_norms),
        "ang_vel": torch.stack(ang_vel_norms),
        "joint_vel_rms": torch.stack(joint_vel_rms),
        "height": torch.stack(heights),
    }


def run_zero_velocity_probe(algo: BaseAlgo, env, num_steps: int, env_ids: torch.Tensor) -> dict:
    env.command_manager.commands[env_ids] = 0.0

    obs = algo.env.reset()
    device = env.device
    dones_any = torch.zeros(env.num_envs, dtype=torch.bool, device=device)

    lin_vels, ang_vels, jvel_rmss, heights = [], [], [], []
    with torch.no_grad():
        for _ in range(num_steps):
            env.command_manager.commands[env_ids] = 0.0
            normalized_obs = algo.obs_normalizer(obs, update=False) if algo.obs_normalization else obs
            actions = _compute_expected_action(algo.actor, normalized_obs)
            obs, _, reset_buf, _ = algo.env.step(actions)
            dones_any |= reset_buf.bool()

            sim = env.simulator
            lin_vels.append(sim.robot_root_states[env_ids, 7:10].norm(dim=-1))
            ang_vels.append(sim.robot_root_states[env_ids, 10:13].norm(dim=-1))
            jvel_rmss.append(sim.dof_vel[env_ids].pow(2).mean(dim=-1).sqrt())
            heights.append(sim.robot_root_states[env_ids, 2])

    lin_vel = torch.stack(lin_vels)
    ang_vel = torch.stack(ang_vels)
    joint_vel_rms = torch.stack(jvel_rmss)
    height = torch.stack(heights)

    return {
        "mean_lin_vel": float(lin_vel.mean().item()),
        "max_lin_vel": float(lin_vel.max().item()),
        "mean_ang_vel": float(ang_vel.mean().item()),
        "max_ang_vel": float(ang_vel.max().item()),
        "mean_joint_vel_rms": float(joint_vel_rms.mean().item()),
        "max_joint_vel_rms": float(joint_vel_rms.max().item()),
        "height_std": float(height.std().item()),
        "height_mean": float(height.mean().item()),
        "any_reset_during_window_frac": float(dones_any[env_ids].float().mean().item()),
    }


def run_kick_probe(algo: BaseAlgo, env, num_steps: int, env_ids: torch.Tensor) -> dict:
    if not hasattr(env, "trigger_kick"):
        return {"skipped": "env has no trigger_kick (not UnifiedManager)"}

    device = env.device
    env.trigger_kick(env_ids)
    obs_dict = env.observation_manager.compute()
    obs = torch.cat([obs_dict[k] for k in algo.config.actor_obs_keys], dim=1)

    sim = env.simulator
    ball_indices = sim.get_actor_indices(["ball"], env_ids=env_ids)
    initial_ball_xyz = sim.all_root_states[ball_indices, :3].clone()

    n = env_ids.numel()
    min_foot_ball_dist = torch.full((n,), float("inf"), device=device)
    # Steps survived before the *first* reset after trigger_kick — distinct from "any reset
    # during the window": once an env resets, later steps in the window may belong to a
    # completely different (possibly locomotion) episode, so "did a reset ever fire" would
    # conflate normal episode churn with the kick attempt actually failing.
    still_on_first_attempt = torch.ones(n, dtype=torch.bool, device=device)
    steps_survived = torch.full((n,), num_steps, dtype=torch.long, device=device)
    heights = []

    with torch.no_grad():
        for step in range(num_steps):
            normalized_obs = algo.obs_normalizer(obs, update=False) if algo.obs_normalization else obs
            actions = _compute_expected_action(algo.actor, normalized_obs)
            obs, _, reset_buf, _ = algo.env.step(actions)

            newly_done = reset_buf.bool()[env_ids] & still_on_first_attempt
            steps_survived[newly_done] = step
            still_on_first_attempt &= ~newly_done

            ball_xyz = sim.all_root_states[ball_indices, :3]
            feet_xyz = sim._rigid_body_pos[env_ids][:, env.feet_indices, :]  # (E, num_feet, 3)
            dist = (feet_xyz - ball_xyz.unsqueeze(1)).norm(dim=-1).min(dim=-1).values  # (E,)
            min_foot_ball_dist = torch.minimum(min_foot_ball_dist, dist)
            heights.append(sim.robot_root_states[env_ids, 2])

    final_ball_xyz = sim.all_root_states[ball_indices, :3]
    ball_displacement = (final_ball_xyz - initial_ball_xyz).norm(dim=-1)
    height = torch.stack(heights)

    return {
        "min_foot_ball_dist_mean": float(min_foot_ball_dist.mean().item()),
        "min_foot_ball_dist_max": float(min_foot_ball_dist.max().item()),
        "ball_displacement_mean": float(ball_displacement.mean().item()),
        "ball_displacement_frac_moved_gt_5cm": float((ball_displacement > 0.05).float().mean().item()),
        "first_attempt_steps_survived_mean": float(steps_survived.float().mean().item()),
        "first_attempt_steps_survived_min": int(steps_survived.min().item()),
        "first_attempt_fell_frac": float((~still_on_first_attempt).float().mean().item()),
        "final_height_mean": float(height[-1].mean().item()),
        "final_height_std": float(height[-1].std().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--zero-vel-seconds", type=float, default=8.0)
    parser.add_argument("--kick-seconds", type=float, default=9.0)
    args = parser.parse_args()

    checkpoint_cfg = CheckpointConfig(checkpoint=args.checkpoint)
    saved_config, saved_wandb_path = load_saved_experiment_config(checkpoint_cfg)
    tyro_config = saved_config.get_eval_config()

    import dataclasses

    tyro_config = dataclasses.replace(
        tyro_config,
        training=dataclasses.replace(tyro_config.training, num_envs=args.num_envs, headless=True),
    )

    env, device, simulation_app = setup_simulation_environment(tyro_config)

    eval_log_dir_parent = "/tmp/eval_probe_logs"
    checkpoint_path = load_checkpoint(checkpoint_cfg.checkpoint, eval_log_dir_parent)

    algo_class = get_class(tyro_config.algo._target_)
    algo: BaseAlgo = algo_class(
        device=device, env=env, config=tyro_config.algo.config, log_dir=eval_log_dir_parent, multi_gpu_cfg=None
    )
    algo.setup()
    algo.attach_checkpoint_metadata(saved_config, saved_wandb_path)
    algo.load(str(checkpoint_path))

    unwrapped_env = algo.unwrapped_env
    unwrapped_env.set_is_evaluating()

    env_ids = torch.arange(args.num_envs, device=device)

    results = {"checkpoint": args.checkpoint}

    zero_vel_steps = int(args.zero_vel_seconds / unwrapped_env.dt)
    logger.info(f"Running zero-velocity probe for {zero_vel_steps} steps ({args.zero_vel_seconds}s)...")
    results["zero_velocity"] = run_zero_velocity_probe(algo, unwrapped_env, zero_vel_steps, env_ids)

    kick_steps = int(args.kick_seconds / unwrapped_env.dt)
    # trigger_kick() itself doesn't check terrain eligibility (it's a raw force-override, used by
    # eval_interactive.py where the human deciding to press 'k' is the eligibility check). Real
    # training only ever assigns kick-mode to flat-terrain envs (a freely-simulated ball needs flat
    # ground); forcing it on a rough/obstacle-terrain env here would place the robot/ball somewhere
    # the kick clip's reference was never meant to be run, dragging every metric here toward
    # meaningless garbage instead of measuring the trained policy's actual kick quality.
    kick_env_ids = env_ids[unwrapped_env.terrain_manager.get_state("locomotion_terrain").env_terrain_is_flat[env_ids]]
    if kick_env_ids.numel() == 0:
        logger.warning("No flat-terrain envs in this eval batch — skipping kick probe. Try a larger --num-envs.")
        results["kick"] = {"skipped": "no flat-terrain envs in this batch"}
    else:
        logger.info(
            f"Running kick probe for {kick_steps} steps ({args.kick_seconds}s) on "
            f"{kick_env_ids.numel()}/{args.num_envs} flat-terrain-eligible envs..."
        )
        results["kick"] = run_kick_probe(algo, unwrapped_env, kick_steps, kick_env_ids)

    print("===EVAL_PROBE_RESULT_START===", flush=True)
    print(json.dumps(results, indent=2), flush=True)
    print("===EVAL_PROBE_RESULT_END===", flush=True)

    close_simulation_app(simulation_app)


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
