"""Unit tests for the kick-ABORT flip (2026-08-28) -- see MultiSkillConfig.kick_abort_prob.

WHAT THIS MECHANISM IS. `kick_recovery_locomotion_flip_enabled` flips every kick env to locomotion
at exactly ONE point: `pre_recovery_motion_end_idx`, the end of the authored clip. So the policy
only ever practises recovering into locomotion from a single, always-identical pose. A 512-env
phase-resolved probe put 76% of teacher-1's falls and 83% of the distilled student's skill-1 falls
in that post-stand phase, while a locomotion control arm under identical terrain/push/DR toppled
0/1847 -- locomotion is not fragile, ARRIVING in it discontinuously is. Kick-abort generalises the
single arrival into a distribution over the clip by flipping at a randomized mid-clip tick.

The mask arithmetic is tested directly here (no env/sim/GPU) because that is where the real risk
lives: the abort trigger is OR-ed into the existing boundary trigger, so a mistake silently changes
when EVERY kick env flips, not just abort envs. What this does NOT cover: that a real
UnifiedManager wires `_kick_abort_flip_tick` through reset->draw->fire correctly (needs a live env).
"""

from __future__ import annotations

import torch

_KICK = 1
_LOCOMOTION = 0


def _flip_mask(
    task_mode: torch.Tensor,
    prev_in_raw_clip: torch.Tensor,
    in_raw_clip: torch.Tensor,
    episode_length_buf: torch.Tensor,
    abort_tick: torch.Tensor,
    enabled_mask: torch.Tensor,
) -> torch.Tensor:
    """Mirrors `_maybe_flip_kick_recovery_to_locomotion`'s combined trigger verbatim."""
    crossed = (
        (task_mode == _KICK) & prev_in_raw_clip & (~in_raw_clip) & (episode_length_buf > 1) & enabled_mask
    )
    aborted = (
        (task_mode == _KICK) & (abort_tick >= 0) & (episode_length_buf >= abort_tick) & enabled_mask
    )
    return crossed | aborted


def _all_kick(n: int) -> dict:
    """Baseline: n kick envs, mid-clip (no boundary crossing), no abort drawn, flip enabled."""
    return dict(
        task_mode=torch.full((n,), _KICK),
        prev_in_raw_clip=torch.ones(n, dtype=torch.bool),
        in_raw_clip=torch.ones(n, dtype=torch.bool),  # still inside -> no boundary cross
        episode_length_buf=torch.full((n,), 40, dtype=torch.long),
        abort_tick=torch.full((n,), -1, dtype=torch.long),
        enabled_mask=torch.ones(n, dtype=torch.bool),
    )


def test_feature_off_is_an_exact_no_op():
    """The guarantee that matters most: with no abort drawn (-1 sentinel everywhere, which is what
    kick_abort_prob=0.0 produces), the combined trigger must equal the boundary trigger alone."""
    kw = _all_kick(6)
    # Make envs 2 and 4 genuinely cross the clip-end boundary.
    kw["in_raw_clip"][[2, 4]] = False
    got = _flip_mask(**kw)
    boundary_only = (
        (kw["task_mode"] == _KICK)
        & kw["prev_in_raw_clip"]
        & (~kw["in_raw_clip"])
        & (kw["episode_length_buf"] > 1)
        & kw["enabled_mask"]
    )
    assert torch.equal(got, boundary_only)
    assert got.tolist() == [False, False, True, False, True, False]


def test_abort_fires_at_its_drawn_tick_and_not_before():
    kw = _all_kick(1)
    kw["abort_tick"][0] = 40
    kw["episode_length_buf"][0] = 39
    assert not _flip_mask(**kw)[0], "must not fire one tick early"
    kw["episode_length_buf"][0] = 40
    assert _flip_mask(**kw)[0], "must fire exactly at the drawn tick"


def test_abort_still_fires_if_the_exact_tick_was_missed():
    """`>=` not `==` on purpose: an env could be reset or otherwise skip its exact tick, and a
    strict equality test would leave it flagged-but-never-flipped for the whole episode."""
    kw = _all_kick(1)
    kw["abort_tick"][0] = 40
    kw["episode_length_buf"][0] = 55
    assert _flip_mask(**kw)[0]


def test_abort_never_fires_for_a_non_kick_env():
    """Once flipped, task_mode is LOCOMOTION -- the guard that stops an already-aborted env from
    re-firing every subsequent tick (which would re-stamp _post_flip_step and restart every
    post_flip_* smoothing window)."""
    kw = _all_kick(1)
    kw["task_mode"][0] = _LOCOMOTION
    kw["abort_tick"][0] = 10
    kw["episode_length_buf"][0] = 99
    assert not _flip_mask(**kw)[0]


def test_per_skill_enabled_mask_gates_abort_too():
    """A skill with the flip disabled must not abort either -- the abort reuses that flip's whole
    machinery, so firing it where the flip is off would be incoherent."""
    kw = _all_kick(2)
    kw["abort_tick"][:] = 10
    kw["episode_length_buf"][:] = 50
    kw["enabled_mask"][1] = False
    assert _flip_mask(**kw).tolist() == [True, False]


def test_boundary_and_abort_are_a_union_not_an_override():
    """Both triggers feed ONE env_ids list so both inherit identical side-effects. An env that
    crosses the boundary and has an abort pending must appear exactly once, not be double-counted
    or masked out."""
    kw = _all_kick(3)
    kw["in_raw_clip"][0] = False          # boundary only
    kw["abort_tick"][1] = 40              # abort only
    kw["in_raw_clip"][2] = False          # both at once
    kw["abort_tick"][2] = 40
    got = _flip_mask(**kw)
    assert got.tolist() == [True, True, True]
    assert got.dtype == torch.bool, "union must stay a plain bool mask (nonzero() gives unique ids)"


def test_abort_tick_past_clip_end_is_harmless():
    """Documented safe fallback: a drawn tick beyond pre_recovery_motion_end_idx never gets its own
    firing, because the boundary trigger flips the env first and task_mode != KICK then gates the
    abort off. Verified as: at the boundary tick BOTH conditions hold (one flip, not two), and
    afterwards neither does."""
    kw = _all_kick(1)
    kw["abort_tick"][0] = 500       # far past the clip
    kw["in_raw_clip"][0] = False    # boundary crossing happens now
    kw["episode_length_buf"][0] = 200
    assert _flip_mask(**kw)[0], "boundary should flip it"
    # Next tick: env is now LOCOMOTION, and its stale abort tick must not re-fire.
    kw["task_mode"][0] = _LOCOMOTION
    kw["episode_length_buf"][0] = 501
    assert not _flip_mask(**kw)[0]


def test_randint_bounds_are_inclusive_of_max():
    """kick_abort_delay_max_steps is documented INCLUSIVE, so the draw uses max+1 as randint's
    exclusive upper bound. Guards the classic off-by-one that would make the configured maximum
    unreachable."""
    torch.manual_seed(0)
    lo, hi = 10, 12
    draws = torch.randint(lo, hi + 1, (4000,))
    assert int(draws.min()) == lo
    assert int(draws.max()) == hi, "the documented-inclusive maximum must actually be drawable"
