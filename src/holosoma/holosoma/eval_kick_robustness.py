"""Headless IsaacSim kick-robustness eval: force ALL envs into the kick task at once and measure
what fraction stay upright through the kick + recovery. Answers directly: is the Stage-C kick
actually robust in the TRAINING simulator (IsaacSim/PhysX, with the ball + full domain
randomization), as the wandb rollout video suggests, or does it fall there too?

Modeled on eval_interactive.run_interactive_eval but non-interactive: one scripted trigger_kick()
across many parallel envs, logging the base-height distribution and cumulative fall fraction.
"""

from __future__ import annotations

import tyro
from loguru import logger

from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.utils.config_utils import CONFIG_NAME
from holosoma.utils.eval_utils import (
    CheckpointConfig,
    init_eval_logging,
    load_checkpoint,
    load_saved_experiment_config,
)
from holosoma.utils.experiment_paths import get_experiment_dir, get_timestamp
from holosoma.utils.helpers import get_class
from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment
from holosoma.utils.tyro_utils import TYRO_CONIFG

from eval_interactive import _compute_expected_action


def run(tyro_config, checkpoint_cfg, saved_config, saved_wandb_path):
    env, device, simulation_app = setup_simulation_environment(tyro_config)
    import torch

    eval_log_dir = get_experiment_dir(tyro_config.logger, tyro_config.training, get_timestamp(), task_name="eval")
    eval_log_dir.mkdir(parents=True, exist_ok=True)
    tyro_config.save_config(str(eval_log_dir / CONFIG_NAME))

    assert checkpoint_cfg.checkpoint is not None
    checkpoint_path = str(load_checkpoint(checkpoint_cfg.checkpoint, str(eval_log_dir)))

    algo_class = get_class(tyro_config.algo._target_)
    algo: BaseAlgo = algo_class(device=device, env=env, config=tyro_config.algo.config,
                                log_dir=str(eval_log_dir), multi_gpu_cfg=None)
    algo.setup()
    algo.attach_checkpoint_metadata(saved_config, saved_wandb_path)
    algo.load(checkpoint_path)

    uenv = algo.unwrapped_env
    uenv.set_is_evaluating()
    assert hasattr(uenv, "trigger_kick"), "checkpoint env is not a UnifiedManager (no trigger_kick)"

    def base_z():
        return uenv.simulator.robot_root_states[:, 2].detach().clone()

    n = uenv.num_envs
    print(f"\n[kick-eval] num_envs={n}  simulator={type(uenv.simulator).__name__}", flush=True)

    with torch.no_grad():
        obs = algo.env.reset()
        # settle standing briefly
        for _ in range(60):
            no = algo.obs_normalizer(obs, update=False) if algo.obs_normalization else obs
            obs, _, _, _ = algo.env.step(_compute_expected_action(algo.actor, no))

        # force ALL envs into the kick task simultaneously
        uenv.trigger_kick()
        obs_dict = uenv.observation_manager.compute()
        obs = torch.cat([obs_dict[k] for k in algo.config.actor_obs_keys], dim=1)

        z0 = base_z()
        ever_fell = torch.zeros(n, dtype=torch.bool, device=device)
        print(f"[kick-eval] at trigger: base_z mean={z0.mean():.3f} min={z0.min():.3f}", flush=True)
        print(f"\n{'step':>5} {'t(s)':>6} {'z_mean':>7} {'z_min':>7} {'z_p10':>7} {'fell%':>6}", flush=True)

        FALL = 0.35
        for step in range(400):  # 8s
            no = algo.obs_normalizer(obs, update=False) if algo.obs_normalization else obs
            obs, _, reset_buf, _ = algo.env.step(_compute_expected_action(algo.actor, no))
            z = base_z()
            ever_fell |= z < FALL
            if step % 20 == 0 or step == 399:
                zc = z.clamp(min=0.0)
                p10 = torch.quantile(zc, 0.10)
                print(f"{step:5d} {step/50:6.2f} {z.mean():7.3f} {z.min():7.3f} {p10:7.3f} "
                      f"{100*ever_fell.float().mean():6.1f}", flush=True)

        surv = 100 * (1 - ever_fell.float().mean())
        print(f"\n[kick-eval] RESULT: {surv:.1f}% of {n} envs SURVIVED the kick "
              f"(stayed above {FALL}m the whole time); {100-surv:.1f}% fell.", flush=True)

    close_simulation_app(simulation_app)


def main() -> None:
    init_eval_logging()
    checkpoint_cfg, remaining = tyro.cli(CheckpointConfig, return_unknown_args=True, add_help=False)
    saved_cfg, saved_wandb_path = load_saved_experiment_config(checkpoint_cfg)
    eval_cfg = saved_cfg.get_eval_config()
    cfg = tyro.cli(ExperimentConfig, default=eval_cfg, args=remaining,
                   description="Overriding config on top of what's loaded.", config=TYRO_CONIFG)
    run(cfg, checkpoint_cfg, saved_cfg, saved_wandb_path)


if __name__ == "__main__":
    main()
