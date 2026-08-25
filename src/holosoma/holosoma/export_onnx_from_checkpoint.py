#!/usr/bin/env python3
"""Re-export a deployment ONNX from an existing training checkpoint (.pt) -- no retraining.

Exists because of the tanh(mu)-vs-E[tanh] deployment bug (see config_types/algo.py's
export_expected_action docstring): every ONNX auto-exported before that fix bakes in tanh(mu) as
the deployed action, which systematically differs from the policy's trained effective behavior
whenever sigma isn't tiny -- enough to make a dynamic kick collapse deterministically while
training/wandb rollouts (stochastic) look perfect. This script rebuilds the export from any
existing .pt with the current export code (Gauss-Hermite expected action by default), so already-
trained checkpoints get the fix without a single training step.

Also runs a numerical parity check: the exported ONNX's "actions" output is compared against the
in-process torch wrapper on random observations, so a silently-wrong export can't slip through.

Usage:
    python src/holosoma/holosoma/export_onnx_from_checkpoint.py \
        --checkpoint logs/.../model_XXXXXXX.pt \
        [--output /path/out.onnx]          # default: alongside the .pt as <stem>_expected.onnx
        [--no-expected-action]             # reproduce the OLD tanh(mu) export instead

Needs the full training env construction (the exporter embeds the motion clip from the env's
motion_command), so run it in the same env/GPU setup as eval_probe.py.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.utils.eval_utils import CheckpointConfig, load_checkpoint, load_saved_experiment_config
from holosoma.utils.helpers import get_class
from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--no-expected-action", action="store_true",
        help="Export the old tanh(mu) action instead of the Gauss-Hermite expected action.",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    out_path = Path(args.output) if args.output else ckpt_path.with_name(ckpt_path.stem + "_expected.onnx")

    checkpoint_cfg = CheckpointConfig(checkpoint=str(ckpt_path))
    saved_config, saved_wandb_path = load_saved_experiment_config(checkpoint_cfg)
    tyro_config = saved_config.get_eval_config()
    # 1 env is enough -- we only need the env for the motion clip + robot config, not rollouts.
    tyro_config = dataclasses.replace(
        tyro_config,
        training=dataclasses.replace(tyro_config.training, num_envs=1, headless=True),
        algo=dataclasses.replace(
            tyro_config.algo,
            config=dataclasses.replace(tyro_config.algo.config, export_expected_action=not args.no_expected_action),
        ),
    )

    env, device, simulation_app = setup_simulation_environment(tyro_config)
    resolved_ckpt = load_checkpoint(checkpoint_cfg.checkpoint, "/tmp/export_onnx_logs")
    algo_class = get_class(tyro_config.algo._target_)
    algo: BaseAlgo = algo_class(
        device=device, env=env, config=tyro_config.algo.config, log_dir="/tmp/export_onnx_logs", multi_gpu_cfg=None
    )
    algo.setup()
    algo.attach_checkpoint_metadata(saved_config, saved_wandb_path)
    algo.load(str(resolved_ckpt))

    logger.info(f"Exporting ONNX (expected_action={not args.no_expected_action}) -> {out_path}")
    algo.export(str(out_path))

    # ---- Numerical parity check: ONNX "actions" vs the in-process torch wrapper ----
    import onnxruntime

    wrapper = algo.actor_onnx_wrapper  # CPU copy, same rule as what was just exported
    session = onnxruntime.InferenceSession(str(out_path))
    input_names = [inp.name for inp in session.get_inputs()]
    rng = np.random.default_rng(0)
    obs = rng.standard_normal((1, algo.actor_obs_dim), dtype=np.float32) * 0.5
    feed = {"obs" if "obs" in input_names else "actor_obs": obs}
    if "time_step" in input_names:
        feed["time_step"] = np.zeros((1, 1), dtype=np.float32)
    onnx_actions = session.run(["actions"], feed)[0]
    with torch.no_grad():
        torch_actions = wrapper(torch.from_numpy(obs)).numpy()
    max_err = float(np.abs(onnx_actions - torch_actions).max())
    logger.info(f"Parity check: max |onnx - torch| action difference = {max_err:.2e}")
    if max_err > 1e-4:
        raise RuntimeError(f"Exported ONNX does not match the torch action rule (max err {max_err}) -- do not deploy.")
    logger.info(f"OK -- exported and verified: {out_path}")

    close_simulation_app(simulation_app)


if __name__ == "__main__":
    main()
