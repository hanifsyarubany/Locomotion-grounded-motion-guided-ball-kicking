"""Interactively deploy a trained checkpoint in IsaacSim.

Same play/stop/restart/quit terminal controls and Ball Position window as replay.py, but here
the actual trained policy drives the robot under real physics (gravity, contacts) instead of
kinematically replaying the reference motion clip — so whether the robot stays upright after the
kick depends entirely on the policy's own robustness, not on scripted/teleported poses.

Also adds, when applicable to the loaded checkpoint's experiment:
- A "kick" terminal command (UnifiedManager checkpoints only) that forces a reset into the kick
  task on demand.
- A Locomotion Velocity window (any checkpoint with locomotion_command registered — stock
  locomotion, or the locomotion side of UnifiedManager) with linear/angular velocity sliders to
  drive the robot live.

Unlike eval_agent.py, this does not run algo.evaluate_policy()'s fixed-length rollout or export
ONNX; it's a separate, small interactive loop built from the same lower-level pieces
(algo.actor / algo.obs_normalizer / algo.env) for manual, open-ended inspection.
"""

from __future__ import annotations

import dataclasses
import sys
import time

import tyro
from loguru import logger
from pydantic.dataclasses import dataclass as pydantic_dataclass

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


@pydantic_dataclass(frozen=True)
class WatchOptions:
    ignore_bad_tracking: bool = False
    """If set, the strict motion-tracking curriculum check (bad_tracking -- reference-clip
    position/orientation/per-body tracking, see managers/termination/terms/wbt.py) is prevented
    from ending kick episodes early, so playback continues past the point it would otherwise pause.

    Use this to see what the policy actually does after a bad_tracking trip, not to assume the
    result is harmless -- it can go either way. Base height/contact termination checks are
    unaffected and stay active either way, so a fall severe enough to trip those will still end
    the episode normally."""

    use_raw_action: bool = False
    """By default this tool now runs the SAME deterministic action ONNX export/deployment uses:
    the expected squashed action E[tanh(mu + sigma*eps)] via Gauss-Hermite quadrature (see
    _compute_expected_action, config_types/algo.py's export_expected_action). Set this to fall
    back to the old behavior -- tanh(mu) straight from algo.actor(obs)[0] -- for comparison.

    This is not a cosmetic difference. Confirmed directly: a Stage-B checkpoint
    (unified-stageB-kick1/model_0145000.pt) with sigma large enough to matter visibly topples
    starting ~t=1.0s and lands flat by t=1.5s under tanh(mu) (with --ignore-bad-tracking, since
    bad_tracking correctly catches this as an early symptom at t=0.56s and pauses first) -- while
    the SAME checkpoint was independently reported to fully complete the same kick under MuJoCo
    sim2sim deployment, which runs the ONNX-exported expected action. This matches
    export_expected_action's own docstring exactly: "tanh(mu) deployment collapsed a dynamic
    single-support kick within ~0.4s, 100% of episodes, while the SAME checkpoint deployed with
    the GH expected action survived 295/300 steps" on a different checkpoint. As sigma -> 0 (a
    well-converged policy) the two converge to the same action, so this mostly matters for
    earlier/less-converged checkpoints -- which is exactly when you're most likely to be using
    this tool to sanity-check one."""


def _compute_expected_action(actor, normalized_obs):
    """E[tanh(mu + sigma*eps)] via fixed 8-node Gauss-Hermite quadrature -- byte-for-byte the same
    formula agents/fast_sac/fast_sac_agent.py's ActorWrapper uses for ONNX export. Deliberately
    NOT reusing ActorWrapper itself: it's a nn.Module built for tracing (obs-normalization baked
    in, single-tensor-in/out forward), whereas here obs are already normalized by the caller and
    actor(normalized_obs) is already available as (action, mean, log_std) -- reimplementing the
    ~6 lines of quadrature math directly avoids constructing and immediately discarding a wrapper
    module every call.
    """
    import math

    import numpy as np
    import torch

    _, mean, log_std = actor(normalized_obs)
    std = log_std.exp()
    t_nodes, w_nodes = np.polynomial.hermite.hermgauss(8)
    nodes = torch.tensor(t_nodes * math.sqrt(2.0), dtype=mean.dtype, device=mean.device)
    weights = torch.tensor(w_nodes / math.sqrt(math.pi), dtype=mean.dtype, device=mean.device)
    shifted = mean.unsqueeze(0) + std.unsqueeze(0) * nodes.view(-1, 1, 1)
    tanh_avg = (torch.tanh(shifted) * weights.view(-1, 1, 1)).sum(dim=0)
    return tanh_avg * actor.action_scale + actor.action_bias


def _ignore_termination_term(env, term_name: str) -> bool:
    """Neuter a single termination term so it never contributes to reset/timeout flags, without
    touching any other term. Returns True if the term existed and was neutered.

    Moves the term from _term_instances to _term_funcs rather than just replacing the callable in
    place: TerminationManager.reset() calls .reset() on every entry in _term_instances, and a bare
    replacement function has no such method -- confirmed the hard way (AttributeError: 'function'
    object has no attribute 'reset', thrown on the very next env reset after a naive in-place
    replacement)."""
    tm = env.termination_manager
    if term_name in tm._term_instances:
        del tm._term_instances[term_name]
    elif term_name not in tm._term_funcs:
        return False

    def _always_false(env, **_kwargs):
        import torch as _torch

        return _torch.zeros(env.num_envs, dtype=_torch.bool, device=env.device)

    tm._term_funcs[term_name] = _always_false
    return True


def _wrap_termination_check_for_diagnostics(env) -> dict:
    """Wrap env.termination_manager.check so the caller can report which specific term(s) caused
    a reset, instead of a bare "fell, or timed out".

    Has to capture this at check() time, not after env.step() returns: reset_envs_idx() (called
    later in the same _post_physics_step()) clears termination_manager.terminated/time_outs for
    the just-reset envs, and re-evaluating the individual term functions after the fact would read
    the ALREADY-RESET (fresh, healthy) state rather than whatever actually triggered the
    termination a moment earlier -- confirmed directly: a checkpoint that looked like it was
    "falling" in the interactive viewer was actually tripping bad_motion_body_pos (an arm-tracking
    check), with base height completely normal throughout; checking termination state after
    env.step() returned showed every sub-check as False, which is simply wrong for what fired.

    Returns a dict with a "names" key the caller can read right after env.step() returns True in
    its reset_buf -- populated by this wrapper during the termination check that just ran inside
    that same env.step() call, so it still reflects the pre-reset state.
    """
    tm = env.termination_manager
    original_check = tm.check
    cause: dict = {"names": []}

    def wrapped_check():
        reset_flags, timeout_flags = original_check()
        fired = []
        if bool(reset_flags.any()) or bool(timeout_flags.any()):
            for name, term_cfg in zip(tm._term_names, tm._term_cfgs):
                if name in tm._term_instances:
                    result = tm._term_instances[name](env, **term_cfg.params)
                else:
                    result = tm._term_funcs[name](env, **term_cfg.params)
                if term_cfg.task_mode is not None and hasattr(env, "task_mode_mask"):
                    result = result & env.task_mode_mask(term_cfg.task_mode)
                if bool(result.any()):
                    fired.append(name)
        cause["names"] = fired
        return reset_flags, timeout_flags

    tm.check = wrapped_check
    return cause


def run_interactive_eval(
    tyro_config: ExperimentConfig,
    checkpoint_cfg: CheckpointConfig,
    saved_config: ExperimentConfig,
    saved_wandb_path: str | None,
    watch_options: WatchOptions | None = None,
):
    env, device, simulation_app = setup_simulation_environment(tyro_config)

    import torch

    from holosoma.config_types.simulator import DEFAULT_BALL_CONFIG_YAML
    from holosoma.utils.replay_controls import BallPositionWindow, ReplayTerminalController, VelocityCommandWindow

    # Kit's shutdown path bypasses normal interpreter exit, which skips flushing a
    # block-buffered stdout — without this, prints from the control loop below can silently
    # vanish instead of showing up in the terminal in real time.
    sys.stdout.reconfigure(line_buffering=True)
    print(
        f"[replay] headless={tyro_config.training.headless} num_envs={tyro_config.training.num_envs} "
        f"scene.ball={tyro_config.simulator.config.scene.ball}",
        flush=True,
    )

    eval_log_dir = get_experiment_dir(tyro_config.logger, tyro_config.training, get_timestamp(), task_name="eval")
    eval_log_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving eval logs to {eval_log_dir}")
    tyro_config.save_config(str(eval_log_dir / CONFIG_NAME))

    assert checkpoint_cfg.checkpoint is not None
    checkpoint = load_checkpoint(checkpoint_cfg.checkpoint, str(eval_log_dir))
    checkpoint_path = str(checkpoint)

    algo_class = get_class(tyro_config.algo._target_)
    algo: BaseAlgo = algo_class(
        device=device,
        env=env,
        config=tyro_config.algo.config,
        log_dir=str(eval_log_dir),
        multi_gpu_cfg=None,
    )
    algo.setup()
    algo.attach_checkpoint_metadata(saved_config, saved_wandb_path)
    algo.load(checkpoint_path)

    unwrapped_env = algo.unwrapped_env
    # Deterministic resets: MotionCommand.reset() always starts at the motion's frame 0 when
    # is_evaluating is set (see managers/command/terms/wbt.py), instead of training's
    # randomized/adaptive phase sampler — exactly what "restart from the top" should do here.
    unwrapped_env.set_is_evaluating()

    use_raw_action = watch_options is not None and watch_options.use_raw_action
    print(
        f"[replay] action = {'tanh(mu) raw actor output' if use_raw_action else 'E[tanh(mu+sigma*eps)] expected action (matches ONNX export)'}",
        flush=True,
    )

    if watch_options is not None and watch_options.ignore_bad_tracking:
        neutered = _ignore_termination_term(unwrapped_env, "bad_tracking")
        print(
            f"[replay] --ignore-bad-tracking set: bad_tracking termination {'disabled' if neutered else '(term not found — nothing to disable)'}. "
            "Playback will continue past what would normally pause it — that may reveal a real fall "
            "playing out, not just a benign tracking mismatch. See WatchOptions' docstring.",
            flush=True,
        )

    motion_command = unwrapped_env.command_manager.get_state("motion_command")
    has_ball = getattr(motion_command, "has_ball", False)

    controller = ReplayTerminalController()
    controller.start()

    ball_window = None
    if has_ball and not tyro_config.training.headless:
        ball_window = BallPositionWindow(unwrapped_env, motion_command, DEFAULT_BALL_CONFIG_YAML)
        print("[replay] Ball Position window created", flush=True)
    elif has_ball:
        print("[replay] ball is configured but running headless — no window to show it in", flush=True)
    else:
        print(
            "[replay] no ball configured for this checkpoint (scene.ball is None in its saved "
            "config) — skipping Ball Position window",
            flush=True,
        )

    has_locomotion_command = unwrapped_env.command_manager.get_state("locomotion_command") is not None
    velocity_window = None
    if has_locomotion_command and not tyro_config.training.headless:
        velocity_window = VelocityCommandWindow(unwrapped_env)
        print("[replay] Locomotion Velocity window created", flush=True)
    elif has_locomotion_command:
        print("[replay] locomotion command available but running headless — no window to show it in", flush=True)
    else:
        print(
            "[replay] no locomotion command registered for this checkpoint — skipping Locomotion "
            "Velocity window",
            flush=True,
        )

    obs = algo.env.reset()

    can_trigger_kick = hasattr(unwrapped_env, "trigger_kick")
    reset_cause = _wrap_termination_check_for_diagnostics(unwrapped_env)

    with torch.no_grad():
        while not controller.quit_requested:
            if controller.restart_requested:
                obs = algo.env.reset()
                controller.restart_requested = False
                controller.playing = True
                print("[replay] restarted", flush=True)

            if controller.kick_requested:
                controller.kick_requested = False
                if can_trigger_kick:
                    # Full reset into the kick clip's starting pose — matches how kick-mode
                    # episodes are trained (one task per episode), not a live mid-stride switch.
                    # Deliberately NOT algo.env.reset(): that calls unwrapped_env.reset_all(),
                    # which re-runs _init_buffers() and would wipe task_mode back to all-locomotion,
                    # undoing the kick assignment trigger_kick() just set. Recompute observations
                    # from the current (just-reset) state directly instead.
                    unwrapped_env.trigger_kick()
                    obs_dict = unwrapped_env.observation_manager.compute()
                    obs = torch.cat([obs_dict[k] for k in algo.config.actor_obs_keys], dim=1)
                    controller.playing = True
                    print("[replay] kick triggered — reset into kick task", flush=True)
                else:
                    print(
                        "[replay] 'kick' has no effect here — this checkpoint's env doesn't support "
                        "trigger_kick() (only UnifiedManager does)",
                        flush=True,
                    )

            if controller.playing:
                if algo.obs_normalization:
                    normalized_obs = algo.obs_normalizer(obs, update=False)
                else:
                    normalized_obs = obs
                if use_raw_action:
                    # Actions are already scaled by the actor (matches BaseAlgo.evaluate_policy)
                    actions = algo.actor(normalized_obs)[0]
                else:
                    # Matches ONNX export/deployment's default action (export_expected_action) —
                    # see WatchOptions.use_raw_action's docstring for why this isn't cosmetic.
                    actions = _compute_expected_action(algo.actor, normalized_obs)
                obs, _, reset_buf, _ = algo.env.step(actions)
                if bool(reset_buf.any()):
                    controller.playing = False
                    # reset_cause is populated by the wrapper below at termination-check time,
                    # before reset_envs_idx() clears the state that caused it -- reading
                    # termination_manager.terminated/time_outs here (after env.step() returns)
                    # would show False for the very env that just reset, since reset_envs_idx()
                    # already zeroed them by this point. Found the hard way: a checkpoint that
                    # looked like it was "falling" was actually failing bad_motion_body_pos (an
                    # arm-tracking check, not a height/fall check at all) -- height stayed normal
                    # the whole time.
                    cause = ", ".join(reset_cause["names"]) or "unknown"
                    print(
                        f"[replay] episode ended (terminated by: {cause}) — paused. "
                        "Type 'r' to restart from the top, or 'p' to keep going from here.",
                        flush=True,
                    )
            else:
                # keep the sim/UI responsive (window dragging, slider drags, viewport nav)
                # without stepping the policy
                simulation_app.update()
                time.sleep(0.03)

    if ball_window is not None:
        ball_window.destroy()
    if velocity_window is not None:
        velocity_window.destroy()
    close_simulation_app(simulation_app)


def main() -> None:
    init_eval_logging()
    checkpoint_cfg, remaining_args = tyro.cli(CheckpointConfig, return_unknown_args=True, add_help=False)
    watch_options, remaining_args = tyro.cli(
        WatchOptions, args=remaining_args, return_unknown_args=True, add_help=False
    )
    saved_cfg, saved_wandb_path = load_saved_experiment_config(checkpoint_cfg)
    eval_cfg = saved_cfg.get_eval_config()

    # get_eval_config() sets randomize_tiles=False, which (see
    # managers/terrain/terms/locomotion.py's _get_env_origins) collapses every env onto grid cell
    # (level=0, type=0) of the checkpoint's OWN training terrain -- not a flat-guaranteed cell.
    # For terrain_unified_mix (what every unified/kick checkpoint trains on), each grid cell's
    # type is an independent random draw (flat 0.4 / rough 0.45 / low_obstacles 0.15, see
    # simulator/shared/terrain.py's randomized_terrain -- np.random.choice per cell, no
    # flat-biasing toward cell (0,0)), so cell (0,0) lands on non-flat terrain more often than not,
    # and -- since eval seeding is deterministic -- it's the *same* cell every single launch of a
    # given checkpoint, not something that improves on retry. A kick triggered on non-flat ground
    # is running on a starting condition kick-mode episodes never trained on (real training only
    # ever assigns kick-mode to flat-terrain-eligible envs, see UnifiedManager's task_mode
    # partition), so it can look badly broken for reasons that have nothing to do with the
    # checkpoint's actual quality.
    #
    # Force every cell flat WITHOUT swapping to a different terrain preset (e.g.
    # terrain_locomotion_plane): that preset's horizontal_scale (1.0) differs from
    # terrain_unified_mix's (0.1), and grid-cell (0,0)'s world-space origin is computed from
    # horizontal_scale/num_rows/num_cols/border_size together (simulator/shared/terrain.py) -- a
    # wholesale preset swap changes that origin by tens of meters (verified directly: (4, 4, 0) ->
    # (76, 4, 0) on this checkpoint), which is why an earlier version of this fix spawned the robot
    # meters outside any camera's default framing ("robot not spawned anywhere"). Only replacing
    # terrain_config leaves every other grid/scale parameter -- and therefore the origin -- exactly
    # as the checkpoint's own training config computed it; only which terrain *type* fills each
    # cell changes. Matches the flat-ground requirement the HL project's frozen_ll.py handles the
    # same way for the identical reason -- there's no legitimate reason an interactive kick-testing
    # session wants rough terrain under the ball.
    if "unified" in eval_cfg.env_class.lower():
        flat_terrain_term = dataclasses.replace(eval_cfg.terrain.terrain_term, terrain_config={"flat": 1.0})
        eval_cfg = dataclasses.replace(eval_cfg, terrain=dataclasses.replace(eval_cfg.terrain, terrain_term=flat_terrain_term))

    overwritten_tyro_config = tyro.cli(
        ExperimentConfig,
        default=eval_cfg,
        args=remaining_args,
        description="Overriding config on top of what's loaded.",
        config=TYRO_CONIFG,
    )
    run_interactive_eval(overwritten_tyro_config, checkpoint_cfg, saved_cfg, saved_wandb_path, watch_options)


if __name__ == "__main__":
    main()
