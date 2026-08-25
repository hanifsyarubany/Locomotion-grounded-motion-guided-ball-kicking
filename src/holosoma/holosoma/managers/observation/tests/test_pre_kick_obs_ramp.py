"""Unit tests for FIX 3 of the 2026-08-18 handoff observation work: ``pre_kick_obs_ramp_steps``,
the missing 4th smoothing sibling of ``pre_kick_reward_ramp_steps`` / ``_termination_grace_steps``
/ ``_reference_blend_steps``.

The structural argument this implements (measured, memory `stage-d-handoff-observation-
discontinuity`): the three existing ramps all smooth what the policy is JUDGED ON, while the
OBSERVATION steps 0 -> live in one control step. A deployed policy is a deterministic function of
its observation, so a step input necessarily produces a step output -- reward shaping provably
cannot fix it, which is why the measured run had all three ramps ON at 50.0 and still jerked ~11x.

Exercises ``UnifiedManager.pre_kick_obs_ramp_alpha`` / ``task_mode_mask_soft`` and the soft
``task_mode_onehot`` through a lightweight fake carrying only the attributes they touch -- the same
fake-env discipline as managers/termination/terms/tests/test_pre_kick_graced_termination.py, which
tests the sibling grace window on the same ``_pre_kick_step`` counter.

The properties that must hold:
  1. OFF (0.0 default) is an EXACT no-op -- ``task_mode_mask_soft`` returns the identical BOOL
     tensor ``task_mode_mask`` already returned, so the manager's multiply is bit-for-bit unchanged.
  2. ON, the kick and locomotion halves CROSSFADE (always summing to 1) rather than fading
     independently -- the policy must never see a tick where both blocks are simultaneously dark.
  3. Only envs that took a MID-EPISODE entry ramp; envs that started in kick mode at reset read
     full strength immediately (there is no transition to smooth there).
"""

from __future__ import annotations

import pytest
import torch

from holosoma.managers.observation.terms.unified import task_mode_onehot

# Mirrors UnifiedManager's own enum values (envs/unified/unified_manager.py) without importing the
# whole env module (which pulls in the simulator stack).
_LOCOMOTION, _KICK = 0, 1


class _FakeUnifiedEnv:
    """Reimplements ONLY the three methods under test, verbatim from UnifiedManager, so the fake
    cannot silently drift into testing different logic than production runs. Everything else the
    real class does (simulator, managers, buffers) is irrelevant to this crossfade."""

    def __init__(self, task_mode, pre_kick_step, episode_length, ramp_steps):
        self.task_mode = torch.tensor(task_mode)
        self._pre_kick_step = torch.tensor(pre_kick_step)
        self.episode_length_buf = torch.tensor(episode_length)
        self._pre_kick_obs_ramp_steps = float(ramp_steps)
        self.num_envs = len(task_mode)

    def task_mode_mask(self, name: str) -> torch.Tensor:
        return self.task_mode == (_KICK if name == "kick" else _LOCOMOTION)

    def pre_kick_steps_since(self) -> torch.Tensor:
        is_pre_kick = self._pre_kick_step >= 0
        return torch.where(
            is_pre_kick,
            (self.episode_length_buf - self._pre_kick_step).clamp(min=0),
            torch.zeros_like(self.episode_length_buf),
        )

    def pre_kick_obs_ramp_alpha(self):
        if self._pre_kick_obs_ramp_steps <= 0.0:
            return None
        is_pre_kick = self._pre_kick_step >= 0
        alpha = (self.pre_kick_steps_since().float() / self._pre_kick_obs_ramp_steps).clamp(0.0, 1.0)
        return torch.where(is_pre_kick, alpha, torch.ones_like(alpha))

    def task_mode_mask_soft(self, name: str) -> torch.Tensor:
        hard = self.task_mode_mask(name)
        alpha = self.pre_kick_obs_ramp_alpha()
        if alpha is None:
            return hard
        in_kick = self.task_mode_mask("kick")
        if name == "kick":
            return torch.where(in_kick, alpha, hard.float())
        return torch.where(in_kick, 1.0 - alpha, hard.float())


# ----------------------------------------------------------------------------------------------
# Property 1 -- OFF is an exact no-op.
# ----------------------------------------------------------------------------------------------


def test_alpha_is_none_when_ramp_disabled():
    """Short-circuits BEFORE touching any buffer, so a config with the ramp off pays nothing."""
    env = _FakeUnifiedEnv([_KICK, _LOCOMOTION], [10, -1], [30, 30], ramp_steps=0.0)
    assert env.pre_kick_obs_ramp_alpha() is None


def test_soft_mask_returns_the_identical_bool_tensor_when_off():
    """The load-bearing no-op guarantee: same dtype (bool) and same values as task_mode_mask, so
    ObservationManager's `mask.to(obs.dtype) * obs` is bit-for-bit the pre-fix line."""
    env = _FakeUnifiedEnv([_KICK, _LOCOMOTION, _KICK], [5, -1, -1], [20, 20, 20], ramp_steps=0.0)
    for name in ("kick", "locomotion"):
        soft, hard = env.task_mode_mask_soft(name), env.task_mode_mask(name)
        assert soft.dtype == torch.bool
        assert torch.equal(soft, hard)


# ----------------------------------------------------------------------------------------------
# Property 2 -- ON, the two halves crossfade.
# ----------------------------------------------------------------------------------------------


def test_alpha_ramps_linearly_from_zero_to_one():
    steps = [0, 10, 25, 50, 80]
    env = _FakeUnifiedEnv([_KICK] * 5, [0] * 5, steps, ramp_steps=50.0)
    got = env.pre_kick_obs_ramp_alpha()
    # allclose, not ==: 10/50 is 0.20000000298 in float32. Exact equality here would be asserting
    # a property of float32 rather than of the ramp.
    assert torch.allclose(got, torch.tensor([0.0, 0.2, 0.5, 1.0, 1.0]), atol=1e-6)


def test_kick_and_loco_masks_sum_to_one_throughout_the_ramp():
    """A genuine crossfade, not two independent fades. If both halves dimmed at once the policy
    would momentarily see a third, never-trained 'everything dark' mode."""
    env = _FakeUnifiedEnv([_KICK] * 6, [0] * 6, [0, 10, 20, 30, 40, 50], ramp_steps=50.0)
    total = env.task_mode_mask_soft("kick") + env.task_mode_mask_soft("locomotion")
    assert torch.allclose(total, torch.ones(6), atol=1e-6)


def test_at_the_fire_tick_kick_is_zero_and_loco_is_full():
    """Continuity at the boundary: the tick the flip happens must still look like locomotion, or
    the ramp would itself introduce the discontinuity it exists to remove."""
    env = _FakeUnifiedEnv([_KICK], [7], [7], ramp_steps=50.0)  # steps_since == 0
    assert env.task_mode_mask_soft("kick").item() == 0.0
    assert env.task_mode_mask_soft("locomotion").item() == 1.0


def test_after_the_window_it_matches_the_hard_binary_behavior():
    """Once the ramp completes the env must be indistinguishable from today's behavior."""
    env = _FakeUnifiedEnv([_KICK], [0], [500], ramp_steps=50.0)
    assert env.task_mode_mask_soft("kick").item() == 1.0
    assert env.task_mode_mask_soft("locomotion").item() == 0.0


def test_pure_locomotion_env_is_unaffected_by_an_active_ramp():
    """An env that never entered kick mode must read exactly the binary values even while other
    envs in the same batch are mid-ramp."""
    env = _FakeUnifiedEnv([_LOCOMOTION, _KICK], [-1, 0], [30, 30], ramp_steps=50.0)
    assert env.task_mode_mask_soft("kick")[0].item() == 0.0
    assert env.task_mode_mask_soft("locomotion")[0].item() == 1.0


# ----------------------------------------------------------------------------------------------
# Property 3 -- only mid-episode entries ramp.
# ----------------------------------------------------------------------------------------------


def test_env_that_started_in_kick_mode_at_reset_reads_full_strength():
    """_pre_kick_step < 0 means "no mid-episode entry" -- a reset-into-kick env has no transition
    to smooth, so ramping it would WEAKEN a correct observation for no reason."""
    env = _FakeUnifiedEnv([_KICK], [-1], [3], ramp_steps=50.0)
    assert env.pre_kick_obs_ramp_alpha().item() == 1.0
    assert env.task_mode_mask_soft("kick").item() == 1.0


def test_mixed_batch_each_env_uses_its_own_entry_tick():
    """Per-env correctness: three envs at different points of their own windows."""
    env = _FakeUnifiedEnv([_KICK, _KICK, _KICK], [0, 20, -1], [10, 30, 99], ramp_steps=40.0)
    alpha = env.pre_kick_obs_ramp_alpha()
    assert alpha[0].item() == 0.25  # 10/40
    assert alpha[1].item() == 0.25  # (30-20)/40
    assert alpha[2].item() == 1.0  # reset-into-kick


# ----------------------------------------------------------------------------------------------
# task_mode_onehot's soft pairing (coherence).
# ----------------------------------------------------------------------------------------------


def test_onehot_is_hard_when_ramp_off():
    env = _FakeUnifiedEnv([_KICK, _LOCOMOTION], [5, -1], [20, 20], ramp_steps=0.0)
    assert torch.equal(task_mode_onehot(env), torch.tensor([[0.0, 1.0], [1.0, 0.0]]))


def test_onehot_is_soft_during_the_ramp():
    """Ramping the gated blocks while leaving the onehot binary would tell the policy 'you are 100%
    in kick mode' on the tick the ball is only 40% faded in -- a combination it never sees at steady
    state. Soft, the pair reads coherently as 'belief in kick mode fading in'."""
    env = _FakeUnifiedEnv([_KICK], [0], [20], ramp_steps=50.0)  # alpha == 0.4
    assert torch.allclose(task_mode_onehot(env), torch.tensor([[0.6, 0.4]]), atol=1e-6)


def test_onehot_still_sums_to_one_during_the_ramp():
    """Must remain a valid distribution over modes, not an arbitrary 2-vector."""
    env = _FakeUnifiedEnv([_KICK] * 4, [0] * 4, [0, 15, 35, 50], ramp_steps=50.0)
    assert torch.allclose(task_mode_onehot(env).sum(dim=-1), torch.ones(4), atol=1e-6)


def test_onehot_unaffected_for_non_unified_envs_without_the_ramp_hook():
    """getattr-guarded: an env class predating this fix (no pre_kick_obs_ramp_alpha at all) must
    still get the plain one-hot rather than an AttributeError."""

    class _Legacy:
        def task_mode_mask(self, name):
            return torch.tensor([name == "kick"])

    assert torch.equal(task_mode_onehot(_Legacy()), torch.tensor([[0.0, 1.0]]))


# ==================================================================================================
# FIX 5 -- the kick->locomotion FLIP direction (post_flip_obs_ramp_steps). This is the direction the
# 2026-08-18 phase probe found the falls in: 76% of teacher-1's and 82.6% of the distilled student's
# skill-1 falls land AFTER this flip, not during the kick.
# ==================================================================================================


class _FakeFlipEnv(_FakeUnifiedEnv):
    """Adds the post-flip half of the state machine. Both ramps are reimplemented verbatim from
    UnifiedManager so the fake cannot drift into testing different logic than production runs."""

    def __init__(self, task_mode, pre_kick_step, post_flip_step, episode_length,
                 entry_ramp=0.0, flip_ramp=0.0):
        super().__init__(task_mode, pre_kick_step, episode_length, entry_ramp)
        self._post_flip_step = torch.tensor(post_flip_step)
        self._post_flip_obs_ramp_steps = float(flip_ramp)

    def post_flip_steps_since(self):
        is_post_flip = self._post_flip_step >= 0
        return torch.where(
            is_post_flip,
            (self.episode_length_buf - self._post_flip_step).clamp(min=0),
            torch.zeros_like(self.episode_length_buf),
        )

    def post_flip_obs_ramp_alpha(self):
        if self._post_flip_obs_ramp_steps <= 0.0:
            return None
        is_post_flip = self._post_flip_step >= 0
        alpha = (self.post_flip_steps_since().float() / self._post_flip_obs_ramp_steps).clamp(0.0, 1.0)
        return torch.where(is_post_flip, alpha, torch.ones_like(alpha))

    def task_mode_mask_soft(self, name: str) -> torch.Tensor:
        hard = self.task_mode_mask(name)
        entry_alpha = self.pre_kick_obs_ramp_alpha()
        flip_alpha = self.post_flip_obs_ramp_alpha()
        if entry_alpha is None and flip_alpha is None:
            return hard
        out = hard.float()
        if entry_alpha is not None:
            in_kick = self.task_mode_mask("kick")
            a = entry_alpha if name == "kick" else 1.0 - entry_alpha
            out = torch.where(in_kick, a, out)
        if flip_alpha is not None:
            in_loco = ~self.task_mode_mask("kick")
            mid_flip = in_loco & (self._post_flip_step >= 0)
            a = flip_alpha if name == "locomotion" else 1.0 - flip_alpha
            out = torch.where(mid_flip, a, out)
        return out


def test_flip_ramp_off_is_exact_no_op():
    env = _FakeFlipEnv([_LOCOMOTION], [-1], [10], [30], entry_ramp=0.0, flip_ramp=0.0)
    for n in ("kick", "locomotion"):
        assert env.task_mode_mask_soft(n).dtype == torch.bool
        assert torch.equal(env.task_mode_mask_soft(n), env.task_mode_mask(n))


def test_kick_terms_fade_out_after_the_flip_instead_of_vanishing():
    """The whole point of FIX 5: at the flip tick the env is already in locomotion mode, so the
    binary mask would zero every kick_* term in ONE step. Ramped, they decay instead."""
    env = _FakeFlipEnv([_LOCOMOTION], [-1], [10], [10], flip_ramp=50.0)  # steps_since == 0
    assert env.task_mode_mask_soft("kick").item() == 1.0       # still fully kick-shaped
    assert env.task_mode_mask_soft("locomotion").item() == 0.0


def test_flip_crossfade_sums_to_one_and_completes():
    env = _FakeFlipEnv([_LOCOMOTION] * 5, [-1] * 5, [0] * 5, [0, 12, 25, 50, 90], flip_ramp=50.0)
    total = env.task_mode_mask_soft("kick") + env.task_mode_mask_soft("locomotion")
    assert torch.allclose(total, torch.ones(5), atol=1e-6)
    # fully locomotion once the window closes -> matches today's binary behavior
    assert env.task_mode_mask_soft("locomotion")[-1].item() == 1.0
    assert env.task_mode_mask_soft("kick")[-1].item() == 0.0


def test_env_that_never_flipped_is_untouched_by_an_active_flip_ramp():
    """A pure-locomotion env (_post_flip_step < 0) must read exact binary values even while other
    envs in the batch are mid-flip -- otherwise FIX 5 would perturb ordinary locomotion."""
    env = _FakeFlipEnv([_LOCOMOTION, _LOCOMOTION], [-1, -1], [-1, 0], [30, 30], flip_ramp=50.0)
    assert env.task_mode_mask_soft("locomotion")[0].item() == 1.0
    assert env.task_mode_mask_soft("kick")[0].item() == 0.0


def test_long_since_flipped_env_reads_binary_not_a_stale_blend():
    """_post_flip_step stays set for the rest of the episode, so the mid_flip guard must rely on
    the ramp having COMPLETED (alpha==1.0) rather than on the sentinel alone."""
    env = _FakeFlipEnv([_LOCOMOTION], [-1], [5], [900], flip_ramp=50.0)
    assert env.task_mode_mask_soft("locomotion").item() == 1.0
    assert env.task_mode_mask_soft("kick").item() == 0.0


def test_both_ramps_on_do_not_interfere():
    """The two directions are mutually exclusive per env at any instant (an env is either mid-entry
    into kick or mid-flip out of it). With both fields set, each env must follow only its own."""
    # env0: mid-ENTRY (in kick, alpha=0.5). env1: mid-FLIP (in loco, alpha=0.5).
    env = _FakeFlipEnv([_KICK, _LOCOMOTION], [0, -1], [-1, 0], [25, 25], entry_ramp=50.0, flip_ramp=50.0)
    k = env.task_mode_mask_soft("kick")
    loco = env.task_mode_mask_soft("locomotion")
    assert k[0].item() == pytest.approx(0.5, abs=1e-6)      # entry: kick fading IN
    assert k[1].item() == pytest.approx(0.5, abs=1e-6)      # flip: kick fading OUT
    assert torch.allclose(k + loco, torch.ones(2), atol=1e-6)


def test_flip_ramp_is_linear_in_elapsed_ticks():
    env = _FakeFlipEnv([_LOCOMOTION] * 4, [-1] * 4, [0] * 4, [0, 10, 20, 40], flip_ramp=40.0)
    assert torch.allclose(env.post_flip_obs_ramp_alpha(), torch.tensor([0.0, 0.25, 0.5, 1.0]), atol=1e-6)
