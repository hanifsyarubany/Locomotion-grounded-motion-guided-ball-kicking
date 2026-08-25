"""Phase-resolved topple probe: WHERE in the kick cycle does a checkpoint actually fall?

Answers the question aggregate ``kick_topple_frac`` cannot: a single scalar says how often the
robot falls, not whether those falls are in the locomotion approach, the leg swing, after ball
contact, or during post-kick recovery -- and those call for completely different fixes. Built
(2026-08-18) to compare a DISTILLED student against the specialist teachers it was distilled from,
where the student's topple rate sits ~3x the teacher's despite a converged regression loss.

Reports, per skill: total topple fraction, the phase breakdown of the falls, cycle index, and the
episode length at the moment of the fall.

USAGE (run from the FORK ROOT, not the package dir -- the skills yaml's motion npz paths are
relative and silently resolve to nothing otherwise):

    HOLOSOMA_SKILLS_CONFIG=configs/skill/distill_skill1_skill2.yaml \\
    CUDA_VISIBLE_DEVICES=2 python -m holosoma.eval_topple_phase_probe \\
        --checkpoint logs/.../model_0093000.pt --num-envs 1024 --seconds 60

THREE GOTCHAS THIS PROBE IS BUILT AROUND -- each already cost this project a wrong conclusion:

1. **NEVER call ``set_is_evaluating()``.** Same checkpoint, terminations/DR held constant, measured
   32% topple with it False and 100% with it True (memory `stagec-kick-eval-harness-artifact`,
   which invalidated an entire round of kick evaluations). ``eval_probe.py`` DOES call it -- that
   is fine for its own relative comparisons but makes it unusable for absolute topple numbers, so
   this is a separate script rather than a flag on that one.

2. **Kick envs only ever exist on FLAT terrain.** ``UnifiedManager._build_task_mode_partition``
   forces every non-flat env to locomotion, so forcing kick mode on a rough-terrain env measures
   the clip being run somewhere it was never trained. Two earlier probe designs were invalidated by
   exactly this (0.1026 measured vs 0.031 real). This probe assigns kick ONLY where
   ``env_terrain_is_flat``.

3. **Validate before trusting the breakdown.** The headline ``topple_frac`` printed here must land
   near the env's OWN ``log_dict["kick_topple_frac"]`` EMA (also printed, as
   ``env_ema_topple_frac``). If the two disagree materially, the phase split underneath them is
   describing something other than what training measures -- fix the probe before reading it.

TOPPLE DEFINITION is taken from training verbatim, not re-invented: an episode topples if its base
height EVER drops below 0.40 m (``UnifiedManager._KICK_FALL_HEIGHT_THRESHOLD``, itself deliberately
equal to the ``kick_low_height`` termination's own threshold).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys

import torch
from loguru import logger

from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.utils.eval_utils import CheckpointConfig, load_checkpoint, load_saved_experiment_config
from holosoma.utils.helpers import get_class
from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment

# Phase labels, ordered by position in the clip. "strike_pre_contact"/"strike_post_contact" split
# the swing at the first moment the ball is seen to move, which is the boundary the earlier
# teacher-side probe found decisive (it measured 0% of falls after contact).
_PHASES = ("approach", "strike_pre_contact", "strike_post_contact", "post_stand", "not_in_clip")

_FALL_HEIGHT = 0.40  # UnifiedManager._KICK_FALL_HEIGHT_THRESHOLD
_BALL_MOVED_EPS = 0.02  # metres of ball displacement that counts as "contact has happened"
# UnifiedManager.TaskMode.KICK. Hardcoded rather than imported so this module stays importable
# without pulling in the env/simulator stack; asserted against the real enum at startup in main().
_TASK_MODE_KICK = 1


def _phase_of(motion_command, ball_moved: torch.Tensor) -> torch.Tensor:
    """[num_envs] long tensor indexing into _PHASES.

    Uses the SAME accessors the reward terms gate on (``in_strike_phase`` bounds the 6 shooting
    terms; ``stand_start_idx`` bounds ``in_kicking_phase``) so "strike" here means exactly what it
    means everywhere else in this codebase rather than a definition invented for this probe.
    """
    in_strike = motion_command.in_strike_phase
    in_kicking = motion_command.in_kicking_phase
    # approach = in the clip, before the strike window opens.
    approach = in_kicking & ~in_strike
    out = torch.full_like(motion_command.time_steps, _PHASES.index("not_in_clip"))
    out[~in_kicking] = _PHASES.index("post_stand")
    out[approach] = _PHASES.index("approach")
    out[in_strike & ~ball_moved] = _PHASES.index("strike_pre_contact")
    out[in_strike & ball_moved] = _PHASES.index("strike_post_contact")
    return out


def run_probe(algo: BaseAlgo, env, num_steps: int) -> dict:
    device = env.device
    n = env.num_envs

    flat = env.terrain_manager.get_state("locomotion_terrain").env_terrain_is_flat
    logger.info(f"[probe] {int(flat.sum().item())}/{n} envs are flat-terrain (kick-eligible)")
    if int(flat.sum().item()) == 0:
        return {"skipped": "no flat-terrain envs; increase --num-envs"}

    obs = algo.env.reset()
    mc = env.command_manager.get_state("motion_command")
    sim = env.simulator
    ball_idx = sim.get_actor_indices("ball", env_ids=None)

    # Per-episode accumulators, reset when an episode ends.
    ep_min_h = torch.full((n,), 10.0, device=device)
    ep_fall_phase = torch.full((n,), -1, dtype=torch.long, device=device)
    ep_fall_step = torch.full((n,), -1, dtype=torch.long, device=device)
    ep_ball_moved = torch.zeros(n, dtype=torch.bool, device=device)
    ep_ball_start = sim.all_root_states[ball_idx, :3].clone()
    ep_cycle = torch.zeros(n, dtype=torch.long, device=device)
    prev_t = mc.time_steps.clone()

    # Completed-episode records (kick-mode, flat-terrain episodes only).
    rec_skill, rec_fell, rec_phase, rec_step, rec_cycle = [], [], [], [], []

    with torch.no_grad():
        for _ in range(num_steps):
            normalized_obs = algo.obs_normalizer(obs, update=False) if algo.obs_normalization else obs
            actions = algo.actor.explore(normalized_obs, deterministic=True)
            obs, _, reset_buf, _ = algo.env.step(actions.float())

            h = sim.robot_root_states[:, 2]
            ep_min_h = torch.minimum(ep_min_h, h)
            moved = (sim.all_root_states[ball_idx, :3] - ep_ball_start).norm(dim=-1) > _BALL_MOVED_EPS
            ep_ball_moved |= moved
            # A clip index that jumped backwards means the reference wrapped -> next kick cycle.
            ep_cycle += (mc.time_steps < prev_t).long()
            prev_t = mc.time_steps.clone()

            # Stamp the phase at the FIRST tick the robot crosses the fall line -- the moment the
            # fall is committed, not where it happens to be lying several ticks later.
            newly_fallen = (h < _FALL_HEIGHT) & (ep_fall_phase < 0)
            if newly_fallen.any():
                ph = _phase_of(mc, ep_ball_moved)
                ep_fall_phase[newly_fallen] = ph[newly_fallen]
                ep_fall_step[newly_fallen] = env.episode_length_buf[newly_fallen]

            done = reset_buf.bool()
            if done.any():
                # Record exactly the population training's own kick_topple_frac is computed over:
                # `_task_mode_partition == KICK` (the PERMANENT per-env assignment), NOT the live
                # `task_mode`. This distinction is load-bearing, not pedantic -- with
                # kick_recovery_locomotion_flip_enabled (on in every current config) an env that
                # SUCCESSFULLY completes its kick flips its live task_mode to locomotion for the
                # rest of the episode, so filtering on the live mode silently drops precisely the
                # successful episodes and inflates topple_frac. Measured on teacher 1 before this
                # was fixed: 0.0445 via live mode vs 0.0197 via the partition (2.3x). The bias is
                # WORSE for better policies (they flip more often), so it shrinks exactly the
                # teacher-vs-student gap this probe exists to measure. Mirrors
                # UnifiedManager._post_physics_step's own `kick_ids` line verbatim.
                partition = getattr(env, "_task_mode_partition", None)
                if partition is None:  # non-UnifiedManager env: fall back to live mode
                    is_kick_pop = env.task_mode_mask("kick")
                else:
                    is_kick_pop = partition == _TASK_MODE_KICK
                keep = done & flat & is_kick_pop
                if keep.any():
                    rec_skill.append(env.skill_id[keep].clone())
                    rec_fell.append((ep_min_h[keep] < _FALL_HEIGHT).clone())
                    rec_phase.append(ep_fall_phase[keep].clone())
                    rec_step.append(ep_fall_step[keep].clone())
                    rec_cycle.append(ep_cycle[keep].clone())
                ep_min_h[done] = 10.0
                ep_fall_phase[done] = -1
                ep_fall_step[done] = -1
                ep_ball_moved[done] = False
                ep_cycle[done] = 0
                ep_ball_start[done] = sim.all_root_states[ball_idx][done, :3]

    if not rec_skill:
        return {"skipped": "no kick episodes completed; increase --seconds"}

    skill = torch.cat(rec_skill)
    fell = torch.cat(rec_fell)
    phase = torch.cat(rec_phase)
    step = torch.cat(rec_step)
    cycle = torch.cat(rec_cycle)

    out: dict = {"n_kick_episodes": int(skill.numel())}
    for sk in sorted(set(skill.tolist())):
        m = skill == sk
        f = fell & m
        d: dict = {
            "n_episodes": int(m.sum().item()),
            "topple_frac": float(fell[m].float().mean().item()),
            "n_topples": int(f.sum().item()),
        }
        if int(f.sum().item()) > 0:
            d["phase_breakdown"] = {
                _PHASES[i]: float((phase[f] == i).float().mean().item()) for i in range(len(_PHASES))
            }
            d["median_step_at_fall"] = float(step[f].float().median().item())
            d["cycle0_frac"] = float((cycle[f] == 0).float().mean().item())
        out[f"skill_{sk}"] = d

    # Gotcha 3: the env's own EMA, for validating the numbers above before trusting them.
    ema = env.log_dict.get("kick_topple_frac")
    out["env_ema_topple_frac"] = float(ema) if ema is not None else None
    out["pooled_topple_frac"] = float(fell.float().mean().item())
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--seconds", type=float, default=60.0)
    args = p.parse_args()

    checkpoint_cfg = CheckpointConfig(checkpoint=args.checkpoint)
    saved_config, saved_wandb_path = load_saved_experiment_config(checkpoint_cfg)
    tyro_config = saved_config.get_eval_config()
    tyro_config = dataclasses.replace(
        tyro_config,
        training=dataclasses.replace(tyro_config.training, num_envs=args.num_envs, headless=True),
    )

    env, device, simulation_app = setup_simulation_environment(tyro_config)
    ckpt = load_checkpoint(checkpoint_cfg.checkpoint, "/tmp/eval_topple_phase_probe")

    algo_class = get_class(tyro_config.algo._target_)
    algo: BaseAlgo = algo_class(
        device=device, env=env, config=tyro_config.algo.config, log_dir="/tmp/eval_topple_phase_probe",
        multi_gpu_cfg=None,
    )
    algo.setup()
    algo.attach_checkpoint_metadata(saved_config, saved_wandb_path)
    algo.load(str(ckpt))

    unwrapped_env = algo.unwrapped_env
    # DELIBERATELY NOT calling set_is_evaluating() -- see gotcha 1 in the module docstring. This is
    # the single most important line in this file, and it is a line that is NOT here.

    # Fail loudly rather than silently mis-filtering if TaskMode's values are ever reordered.
    from holosoma.envs.unified.unified_manager import TaskMode

    assert int(TaskMode.KICK) == _TASK_MODE_KICK, (
        f"TaskMode.KICK is {int(TaskMode.KICK)}, this probe assumes {_TASK_MODE_KICK} -- update _TASK_MODE_KICK."
    )

    steps = int(args.seconds / unwrapped_env.dt)
    logger.info(f"[probe] {steps} steps ({args.seconds}s) x {args.num_envs} envs")
    results = {"checkpoint": args.checkpoint, "num_envs": args.num_envs, "seconds": args.seconds}
    results.update(run_probe(algo, unwrapped_env, steps))

    print("===TOPPLE_PHASE_RESULT_START===", flush=True)
    print(json.dumps(results, indent=2), flush=True)
    print("===TOPPLE_PHASE_RESULT_END===", flush=True)
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
