"""Splice a trained teacher's CRITIC apparatus into a distilled student checkpoint.

WHY THIS EXISTS (2026-08-28). distill_specialists.py trains only the actor
(`optimizer = Adam(student_actor.parameters())`) -- the student's qnet/qnet_target are whatever
`FastSACAgent.setup()` randomly initialized, and are never touched. Verified directly on a real
distilled checkpoint: `qnet_state_dict` and `qnet_target_state_dict` are BYTE-IDENTICAL (a trained
target network is an EMA of the online one and always drifts from it -- identical is only possible
pre-training), and `log_alpha` sits at exactly ln(alpha_init) for both groups.

Resuming RL training straight from that checkpoint reproduces the Stage-B->C "stale critic"
failure this project already has a name for (critic_warmup_iters' own docstring), except worse:
with `critic_warmup_iters` set, the critic trains for N steps but ONLY on the frozen actor's own
narrow rollout distribution, so it becomes locally accurate in a tube around the distilled policy
and unconstrained everywhere else. The instant the actor unfreezes it is REWARDED for finding
exactly the states where that young critic is wrong -- measured on a real run: `action_std/kick`
0.0291 (in-band with the teachers' own healthy 0.0245-0.0302) -> 0.0904 one logging interval after
unfreeze (3.1x), `kick_topple_frac` 0.028 -> 0.61, with `qf_min` pinned at the critic's own -20
support floor (a healthy teacher never saturates, sitting around -16). Longer warmup does not fix
this -- a frozen actor keeps generating the same narrow data, so the critic only grows MORE
confidently wrong outside it. L2-SP on the actor (0.01) measurably dampens the collapse (~5-10%
smaller topple spike) but does not prevent it, because the actor was never the miscalibrated part.

THE FIX: a teacher's critic is not random -- it is a fully-trained Q-function for a policy the
distilled student closely resembles (that resemblance is the entire premise of DAgger distillation
succeeding). Seeding the student's critic apparatus from one teacher converts "critic learns from
scratch while the actor is forced to wait" into "critic starts approximately right, refines from
there" -- the SAME role play as any other warm start in this project.

HONEST LIMIT, not swept under the rug: a single teacher's critic is well-calibrated for ITS OWN
skill's states and only approximately transferable to the other ~3/4 of the student's state space
(the other skills' clips + locomotion). This is a warm start, not a correct multi-skill critic --
still schedule SOME `critic_warmup_iters` after using this (a few hundred to ~1000, short, since
the seed is now approximately right rather than starting from nothing) so the seeded critic gets a
chance to adapt to the OTHER skills' states before the actor is allowed to exploit it. The clean
long-term fix is critic distillation (regress student Q toward the routed teacher's Q, the same
skill-routing the actor loss already uses) -- not implemented; this script is the fast path.

WHAT GETS REPLACED, AND WHY EACH BOUNDARY IS WHERE IT IS
----------------------------------------------------------
Everything on the CRITIC side is taken from the teacher, as one atomic bundle, because a qnet and
the normalizer that fed its inputs during training are not independently meaningful -- this is the
exact "teacher normalizer mismatch" trap distill_specialists.py's own docstring warns about for the
actor side, applied here to the critic side:
    qnet_state_dict, qnet_target_state_dict, critic_obs_normalizer_state,
    log_alpha, q_optimizer_state_dict, alpha_optimizer_state_dict

Everything on the ACTOR side stays from the student, unchanged, because the distilled actor is the
entire point of the exercise and already behaves like the teachers (verified: action_std in-band):
    actor_state_dict, obs_normalizer_state, actor_optimizer_state_dict

Everything else (grad_scaler_state_dict, global_step, env_state, args, experiment_config,
wandb_run_path, iteration) is carried over from the STUDENT checkpoint verbatim -- these are
process/run bookkeeping, not model state, and `FastSACAgent.load()` never reads them in a way that
depends on the critic.

VALIDATED BEFORE WRITING, NOT ASSUMED:
  - both checkpoints' qnet_state_dict have the SAME key set and per-key SHAPE (a size mismatch
    means the teacher was trained under a different critic_obs_dim/num_atoms/architecture and
    splicing it in would crash on the FIRST forward pass of the next training run, not silently
    misbehave -- caught here, before that run is ever launched, with a clear message naming the
    offending key).
  - both checkpoints carry every key FastSACAgent.load() unconditionally reads (listed above) --
    an older/differently-produced checkpoint missing one would otherwise fail this script with a
    confusing KeyError; caught up front with a clear message instead.
  - log_alpha's shape is compatible between teacher and student (same num_alpha_groups) -- a
    mismatch here is a real config discrepancy between the two runs (e.g. one used
    kick_target_entropy_ratio and the other didn't) worth surfacing, not silently broadcasting.

USAGE
    python scripts/seed_critic_from_teacher.py \\
        --student logs/UnifiedBallKickingEnhanced/20260827_122002-distill-4skills-distill/model_0200000.pt \\
        --teacher logs/UnifiedBallKickingEnhanced/20260827_044801-stageB-skill012-h074-locomotion/model_0250000.pt \\
        --output  logs/UnifiedBallKickingEnhanced/20260827_122002-distill-4skills-distill/model_0200000_critic-seeded.pt

Then launch train_agent.py with `--training.checkpoint` pointed at --output, same as any other
resume. Recommended: keep a short critic_warmup_iters (few hundred-1000) and use_autotune=False for
that short warmup so alpha (also seeded from the teacher, already well-calibrated) cannot wind up
against the still-frozen actor -- see the module docstring above for why longer warmup does not
help here.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch

# Critic-side keys travel together as one atomic bundle from the teacher -- see module docstring.
_CRITIC_BUNDLE_KEYS = (
    "qnet_state_dict",
    "qnet_target_state_dict",
    "critic_obs_normalizer_state",
    "log_alpha",
    "q_optimizer_state_dict",
    "alpha_optimizer_state_dict",
)

# Every key FastSACAgent.load() reads unconditionally (i.e. not gated on a config flag) --
# see fast_sac_agent.py:load(). alpha_optimizer_state_dict is technically conditional there
# (only loaded if log_alpha's shape matches), but we require it present on BOTH checkpoints
# anyway so the bundle-copy above is always well-defined regardless of which run has which shape.
_REQUIRED_KEYS = (
    "actor_state_dict",
    "qnet_state_dict",
    "qnet_target_state_dict",
    "obs_normalizer_state",
    "critic_obs_normalizer_state",
    "log_alpha",
    "actor_optimizer_state_dict",
    "q_optimizer_state_dict",
    "alpha_optimizer_state_dict",
    "grad_scaler_state_dict",
    "global_step",
)


def _state_dict_shape_map(sd) -> dict[str, tuple[int, ...]]:
    return {k: tuple(v.shape) for k, v in sd.items() if hasattr(v, "shape")}


def _validate_required_keys(ckpt: dict, label: str, path: Path) -> None:
    missing = [k for k in _REQUIRED_KEYS if k not in ckpt]
    if missing:
        raise ValueError(
            f"{label} checkpoint ({path}) is missing key(s) FastSACAgent.load() requires "
            f"unconditionally: {missing}. This does not look like a checkpoint produced by "
            "FastSACAgent.save() / distill_specialists.py's save_params -- refusing to splice."
        )


def _validate_qnet_shape_compatibility(student_ckpt: dict, teacher_ckpt: dict, teacher_path: Path) -> None:
    """Same key set AND same per-key shape, or refuse. A mismatch here means the teacher was
    trained under a different critic architecture (critic_obs_dim / num_atoms / hidden width) and
    load_state_dict would raise mid-training-launch rather than at this much cheaper, much clearer
    checkpoint-prep step."""
    student_shapes = _state_dict_shape_map(student_ckpt["qnet_state_dict"])
    teacher_shapes = _state_dict_shape_map(teacher_ckpt["qnet_state_dict"])

    student_keys, teacher_keys = set(student_shapes), set(teacher_shapes)
    if student_keys != teacher_keys:
        only_student = sorted(student_keys - teacher_keys)
        only_teacher = sorted(teacher_keys - student_keys)
        raise ValueError(
            f"qnet_state_dict key sets differ between student and teacher ({teacher_path}) -- "
            f"different critic architecture, cannot splice.\n"
            f"  keys only in student: {only_student}\n"
            f"  keys only in teacher: {only_teacher}"
        )

    mismatched = {
        k: (student_shapes[k], teacher_shapes[k])
        for k in student_shapes
        if student_shapes[k] != teacher_shapes[k]
    }
    if mismatched:
        lines = "\n".join(f"  {k}: student={s} vs teacher={t}" for k, (s, t) in sorted(mismatched.items())[:10])
        more = f"\n  ... and {len(mismatched) - 10} more" if len(mismatched) > 10 else ""
        raise ValueError(
            f"qnet_state_dict shape mismatch between student and teacher ({teacher_path}) -- "
            f"the teacher was trained under a different critic_obs_dim/num_atoms/hidden width. "
            f"Splicing would crash on the first training step, not silently misbehave, so this is "
            f"refused here instead:\n{lines}{more}"
        )


def _validate_log_alpha_shape(student_ckpt: dict, teacher_ckpt: dict, teacher_path: Path) -> None:
    s_shape = tuple(student_ckpt["log_alpha"].shape)
    t_shape = tuple(teacher_ckpt["log_alpha"].shape)
    if s_shape != t_shape:
        raise ValueError(
            f"log_alpha shape differs between student ({s_shape}) and teacher ({teacher_path}, "
            f"{t_shape}) -- num_alpha_groups differs (kick_target_entropy_ratio set on one run "
            "and not the other?). Reconcile the two runs' configs before seeding; broadcasting "
            "here would silently paper over a real training-setup discrepancy, unlike "
            "FastSACAgent.load()'s own resume-time broadcast (which handles [1]->[N] on a "
            "genuine mid-project config change, not a mismatch between two checkpoints picked "
            "for this splice)."
        )


def seed_critic(student_ckpt: dict, teacher_ckpt: dict, teacher_path: Path) -> dict:
    """Returns a NEW dict (does not mutate either input) -- the student checkpoint with its
    critic-side bundle replaced by the teacher's."""
    _validate_qnet_shape_compatibility(student_ckpt, teacher_ckpt, teacher_path)
    _validate_log_alpha_shape(student_ckpt, teacher_ckpt, teacher_path)

    out = dict(student_ckpt)  # shallow copy: values below are replaced wholesale, not mutated in place
    for key in _CRITIC_BUNDLE_KEYS:
        out[key] = teacher_ckpt[key]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed a distilled student checkpoint's critic apparatus from a trained teacher.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--student", required=True, type=Path, help="Distilled student .pt (actor kept).")
    parser.add_argument("--teacher", required=True, type=Path, help="Trained teacher .pt (critic taken).")
    parser.add_argument("--output", required=True, type=Path, help="Path to write the spliced .pt to.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite --output if it already exists (default: refuse, since a checkpoint file "
        "is exactly the kind of thing a silent overwrite makes hard to notice).",
    )
    args = parser.parse_args()

    for label, path in (("--student", args.student), ("--teacher", args.teacher)):
        if not path.is_file():
            print(f"error: {label} path does not exist or is not a file: {path}", file=sys.stderr)
            sys.exit(1)
    if args.output.exists() and not args.force:
        print(f"error: --output already exists: {args.output} (pass --force to overwrite)", file=sys.stderr)
        sys.exit(1)

    print(f"[seed_critic] loading student:  {args.student}")
    student_ckpt = torch.load(args.student, map_location="cpu", weights_only=False)
    print(f"[seed_critic] loading teacher:  {args.teacher}")
    teacher_ckpt = torch.load(args.teacher, map_location="cpu", weights_only=False)

    _validate_required_keys(student_ckpt, "student", args.student)
    _validate_required_keys(teacher_ckpt, "teacher", args.teacher)

    # Byte-identical qnet/qnet_target is this script's whole reason to exist -- confirm the
    # student actually has the symptom before "fixing" it, so a mistaken --student path (e.g. an
    # already-RL-trained checkpoint) is caught here instead of silently overwriting a perfectly
    # good critic with the teacher's.
    q, qt = student_ckpt["qnet_state_dict"], student_ckpt["qnet_target_state_dict"]
    qnet_target_diff = sum(
        float((q[k].float() - qt[k].float()).abs().sum()) for k in q if hasattr(q[k], "shape")
    )
    if qnet_target_diff > 1e-6:
        print(
            f"warning: student's qnet/qnet_target already differ (sum|diff|={qnet_target_diff:.4f}) "
            "-- this checkpoint may already have a trained critic. Proceeding anyway since you "
            "asked to seed it, but double-check --student is the checkpoint you meant.",
            file=sys.stderr,
        )
    else:
        print("[seed_critic] confirmed: student's qnet/qnet_target are byte-identical (untrained critic).")

    out_ckpt = seed_critic(student_ckpt, teacher_ckpt, args.teacher)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(out_ckpt, tmp_path)
    shutil.move(str(tmp_path), str(args.output))  # atomic-ish: no half-written file at the final path

    print(f"[seed_critic] wrote: {args.output}")
    print(
        "[seed_critic] critic bundle taken from teacher "
        f"({', '.join(_CRITIC_BUNDLE_KEYS)}); actor bundle kept from student "
        "(actor_state_dict, obs_normalizer_state, actor_optimizer_state_dict)."
    )
    print(
        "[seed_critic] reminder: still use a SHORT critic_warmup_iters (few hundred-1000) with "
        "use_autotune=False on the resumed run -- see this script's module docstring for why."
    )


if __name__ == "__main__":
    main()
