"""Smoke-test a checkpoint's actor under the REAL RL rollout convention, before spending a full
resumed-training run to find out it was broken.

WHY THIS EXISTS (2026-08-28). `distill_specialists.py`'s own rollout and loss target both go
through `predict_action`/`expected_action` -- E[tanh(mu + sigma*Z)] via Gauss-Hermite quadrature,
deliberately chosen to match what the exported ONNX computes at deployment. But
`FastSACAgent.learn()`'s real rollout (`self.policy = self.actor.explore`, called WITHOUT
`deterministic=True`) uses a DIFFERENT convention: a genuine stochastic sample,
`tanh(Normal(mu, sigma).rsample())`. These are not the same execution path, and nothing before
this script ever checked whether a checkpoint that looks perfect under one looks fine under the
other.

Measured consequence on a real run (2026-08-28): a multi-teacher-distilled checkpoint showed
excellent telemetry throughout its ENTIRE 20k-step training window under the smoothed
convention -- kick_topple_frac 0.02-0.08, kick_episode_length 800-899, the best numbers seen all
session -- then collapsed to topple 0.44-0.99 within the first 300 steps of a real RL resume,
BEFORE the actor ever received a single gradient update (critic_warmup_iters was still active).
Damage that shows up while the actor is provably frozen cannot be a critic-unfreeze artifact; it
can only mean the actor itself behaves differently under real sampling than under the smoothed
rollout distillation was validated against. This script exists to catch that gap directly,
cheaply, before committing to an expensive resumed-training run.

WHAT IT DOES: builds the real env + a FastSACAgent exactly as train_agent.py/distill_specialists.py
do, loads ONLY the actor + its normalizer from the checkpoint (the critic is irrelevant to this
question -- this is purely about whether the ACTOR is safe to roll out), then steps the env for
`--num-steps` using `agent.actor.explore(deterministic=False)` -- the EXACT function
`FastSACAgent.learn()` itself uses for real rollout, not a re-implementation of it. No gradient
updates happen at all (this is pure rollout, not training) -- there is nothing to warm up and
nothing to freeze; the check is entirely about the checkpoint's own actor weights.

WHAT IT REPORTS, and against what reference: `kick_topple_frac`/`kick_episode_length` are read
directly off `env.log_dict` (the same EMA-over-ended-episodes values every training run logs to
wandb -- see UnifiedManager._update_log_dict), plus the RAW per-step sigma (`log_std.exp()`) from
the actor's own forward pass, which the smoothed convention can hide entirely (a large sigma
barely moves the SMOOTHED expected action, but dominates every RAW sample). Reference bands are
this project's own measured healthy values, not invented thresholds:
    kick_topple_frac   healthy ~0.02-0.09   (teacher references, multiple runs this session)
    action_std (sigma) healthy ~0.025-0.03  (same teacher references)
Both were measured catastrophic (topple >0.4, action_std >0.09) on the checkpoint that motivated
this script. The verdict is WARN, not a hard FAIL, deliberately: this is a cheap smoke test with a
short window and no claim to statistical rigor, meant to catch an obviously-broken checkpoint
before an expensive resume -- a borderline reading should prompt a longer look, not an automatic
rejection.

USAGE. This script's OWN knobs (checkpoint/num-envs/num-steps/seed) go through env vars, NOT CLI
flags -- same reasoning distill_specialists.py's HOLOSOMA_DISTILL_* knobs already established:
interleaving a second dataclass's flags into tyro's own big Union/subcommand CLI grammar
(exp:X algo:Y --training.foo --algo.config.bar) is fragile to parse correctly, and env vars
sidestep that entirely. The remaining argv is passed through to AnnotatedExperimentConfig
UNCHANGED, so env/task selection is byte-identical to a real training launch.

    export PYTHONPATH=.../src/holosoma:$PYTHONPATH
    export HOLOSOMA_SKILLS_CONFIG=configs/skill/4skills.yaml
    export HOLOSOMA_VALIDATE_CHECKPOINT=logs/.../model_0220000.pt   # required
    export HOLOSOMA_VALIDATE_NUM_STEPS=1000                          # optional, default 1000
    python3 scripts/validate_checkpoint_real_rollout.py exp:g1-29dof-unified-fast-sac \\
        --training.num-envs 512

NOT what this script is: not a replacement for a real resumed-training run, not a claim that a
PASS verdict guarantees the checkpoint is safe for the FULL RL resume (only that its actor
doesn't immediately fall over under real sampling) -- and not a critic check at all (the critic is
never loaded from the checkpoint here; see scripts/seed_critic_from_teacher.py for that side).
"""

from __future__ import annotations

import dataclasses
import os
import sys

from loguru import logger

_CHECKPOINT_ENV_VAR = "HOLOSOMA_VALIDATE_CHECKPOINT"
_NUM_STEPS_ENV_VAR = "HOLOSOMA_VALIDATE_NUM_STEPS"
_SEED_ENV_VAR = "HOLOSOMA_VALIDATE_SEED"
_DEFAULT_NUM_STEPS = 1000
_DEFAULT_SEED = 42
_DEFAULT_SETTLE_STEPS = 200  # discard this many initial steps before aggregating verdict stats,
# same rationale as every other diagnostic in this project that trims a warm-up transient: a
# freshly-reset env hasn't had time to reach a representative kick/locomotion phase mix yet.

# Reference bands, this project's own measured healthy values (see module docstring) -- not
# invented thresholds. WARN, not FAIL, is deliberate: see module docstring's last paragraph.
_TOPPLE_WARN_THRESHOLD = 0.15
_ACTION_STD_WARN_THRESHOLD = 0.05


def _log_env_int(var: str, default: int) -> int:
    raw = os.environ.get(var)
    return int(raw) if raw else default


def validate(tyro_config, checkpoint_path: str, num_steps: int, seed: int) -> bool:
    """Returns True on PASS, False on WARN (see module docstring: never hard-raises on a bad
    reading -- a borderline checkpoint should prompt a longer look, not silently exit nonzero in
    a way that could be scripted past without a human ever reading the printed numbers)."""
    from holosoma.utils.eval_utils import init_sim_imports
    from holosoma.utils.sim_utils import close_simulation_app

    simulation_app = init_sim_imports(tyro_config)
    try:
        # torch (and anything importing it, e.g. FastSACAgent) must not be imported before
        # init_sim_imports() -- same constraint distill_specialists.py's own module docstring
        # documents at its own `import torch` line.
        import torch

        from holosoma.agents.fast_sac.fast_sac_agent import FastSACAgent
        from holosoma.config_types.env import get_tyro_env_config
        from holosoma.train_agent import get_device
        from holosoma.utils.common import seeding
        from holosoma.utils.helpers import get_class

        device = get_device(tyro_config, None)
        seeding(seed, torch_deterministic=False)

        env_target = tyro_config.env_class
        tyro_env_config = get_tyro_env_config(tyro_config)
        env = get_class(env_target)(tyro_env_config, device=device)

        # Same buffer_size=2 shrink distill_specialists.py uses, same reason: this script never
        # calls .learn(), so a training-scale replay buffer allocation here is pure dead weight.
        # See that script's own comment (search "SimpleReplayBuffer") for the measured GiB figure.
        agent_config = dataclasses.replace(tyro_config.algo.config, buffer_size=2)
        agent = FastSACAgent(env=env, config=agent_config, device=device, log_dir="/tmp/validate_checkpoint_scratch")
        agent.setup()

        logger.info(f"[validate] loading actor + obs_normalizer from {checkpoint_path}")
        torch_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        agent.actor.load_state_dict(torch_checkpoint["actor_state_dict"])
        agent.obs_normalizer.load_state_dict(torch_checkpoint["obs_normalizer_state"])
        agent.actor.eval()

        # agent.env is the FastSACEnv wrapper (built inside FastSACAgent.__init__), NOT the raw
        # `env` object above -- its .reset()/.step() already handle obs-dict-to-tensor
        # concatenation internally (same object distill_specialists.py's own loop uses via
        # `student_agent.env.reset()`/`.step()`). Reusing it here rather than reimplementing the
        # concatenation is deliberate: a second, hand-rolled version is exactly the kind of thing
        # that could silently drift from the real training path.
        obs = agent.env.reset()

        topple_readings: list[float] = []
        ep_len_readings: list[float] = []
        std_readings: list[float] = []

        logger.info(
            f"[validate] rolling out {num_steps} steps "
            f"(discarding first {_DEFAULT_SETTLE_STEPS} as settle-in) via REAL stochastic "
            "sampling -- actor.explore(deterministic=False), the exact function "
            "FastSACAgent.learn() itself uses for rollout."
        )
        for step in range(num_steps):
            with torch.no_grad():
                normed_obs = agent.obs_normalizer(obs, update=False)
                # The real rollout convention -- see module docstring. Deliberately NOT
                # predict_action/expected_action (distill_specialists.py's smoothed convention);
                # the entire point of this script is to exercise the OTHER one.
                action = agent.actor.explore(normed_obs, deterministic=False)
                _, _mean, log_std = agent.actor(normed_obs)

            next_obs, _rewards, _dones, _infos = agent.env.step(action.float())
            obs = next_obs

            if step >= _DEFAULT_SETTLE_STEPS:
                topple = env.log_dict.get("kick_topple_frac")
                ep_len = env.log_dict.get("kick_episode_length")
                if topple is not None:
                    topple_readings.append(float(topple))
                if ep_len is not None:
                    ep_len_readings.append(float(ep_len))
                std_readings.append(float(log_std.exp().mean()))

        if not topple_readings:
            logger.warning(
                "[validate] kick_topple_frac never appeared in env.log_dict -- this env/task_config "
                "combination may not be kick-capable (e.g. kick_probability=0 / no flat-eligible "
                "envs), so this script cannot assess kick health. Nothing else about this run is "
                "wrong; there is just nothing kick-related to check."
            )
            return True

        mean_topple = sum(topple_readings) / len(topple_readings)
        mean_ep_len = sum(ep_len_readings) / len(ep_len_readings) if ep_len_readings else float("nan")
        mean_std = sum(std_readings) / len(std_readings)

        logger.info(
            f"[validate] RESULT (mean over last {len(topple_readings)} steps): "
            f"kick_topple_frac={mean_topple:.4f} kick_episode_length={mean_ep_len:.1f} "
            f"action_std(raw sigma)={mean_std:.4f}"
        )
        logger.info(
            f"[validate] reference (this project's own measured healthy values): "
            f"kick_topple_frac ~0.02-0.09, action_std ~0.025-0.03"
        )

        passed = True
        if mean_topple > _TOPPLE_WARN_THRESHOLD:
            logger.warning(
                f"[validate] WARN: kick_topple_frac={mean_topple:.4f} exceeds "
                f"{_TOPPLE_WARN_THRESHOLD} -- this checkpoint's actor is falling far more than "
                "this project's healthy reference under REAL stochastic rollout, even though "
                "distillation's own smoothed-rollout telemetry may have looked fine. Do not "
                "resume real RL training from this checkpoint without investigating further."
            )
            passed = False
        if mean_std > _ACTION_STD_WARN_THRESHOLD:
            logger.warning(
                f"[validate] WARN: action_std={mean_std:.4f} exceeds {_ACTION_STD_WARN_THRESHOLD} "
                "-- raw policy sigma is elevated well above this project's healthy reference. A "
                "large sigma is exactly what the smoothed E[tanh(mu+sigma*Z)] rollout convention "
                "can hide (see module docstring) while dominating real stochastic samples."
            )
            passed = False
        if passed:
            logger.info("[validate] PASS -- within this project's healthy reference bands.")
        return passed
    finally:
        close_simulation_app(simulation_app)


def main() -> None:
    import tyro

    from holosoma.config_values.experiment import AnnotatedExperimentConfig
    from holosoma.utils.tyro_utils import TYRO_CONIFG

    checkpoint_path = os.environ.get(_CHECKPOINT_ENV_VAR)
    if not checkpoint_path:
        print(f"error: {_CHECKPOINT_ENV_VAR} is required (path to the .pt checkpoint to validate)", file=sys.stderr)
        sys.exit(1)
    num_steps = _log_env_int(_NUM_STEPS_ENV_VAR, _DEFAULT_NUM_STEPS)
    seed = _log_env_int(_SEED_ENV_VAR, _DEFAULT_SEED)

    # argv passed through UNCHANGED -- this script's own knobs are all env vars (see module
    # docstring for why), so there is nothing of this script's own to strip out first. Same
    # parser every other entry point in this project uses, so env/task selection (exp:X, algo:Y,
    # --training.num-envs, etc.) is byte-identical to a real training launch.
    tyro_config = tyro.cli(AnnotatedExperimentConfig, config=TYRO_CONIFG)

    passed = validate(tyro_config, checkpoint_path, num_steps, seed)
    sys.exit(0 if passed else 2)


if __name__ == "__main__":
    main()
