"""Unit tests for ``target_pos_b``'s optional tanh distance compression -- FIX 1 of the 2026-08-18
locomotion->kick observation-discontinuity work (see
``MultiSkillConfig.obs_target_pos_distance_scale``).

Why this term specifically: a live probe on ckpt 400k attributed **51% of the entire fire-tick
observation jump** to these 2 dims alone, purely because they ship at ``scale=1.0`` in raw metres
against a 5-7 m target while nearly every other observation is O(1).

The two properties that must hold, and that these tests pin:
  1. ``distance_scale=0.0`` (the default) is an EXACT no-op -- byte-identical to the pre-fix
     behavior, so every existing checkpoint and config is untouched.
  2. When on, the output stays **2-dimensional** (width-preserving => checkpoints still warm-start,
     which is the whole reason this was chosen over a direction+distance 3-vector) and bounded,
     while preserving direction exactly.

Uses a lightweight fake env exposing exactly what the real ``target_pos_b`` reads, so these
exercise the REAL production function rather than a reimplementation of its math -- same discipline
as test_ball_pos_b_critic_bias_isolation.py in this directory.
"""

from __future__ import annotations

import math

import pytest
import torch

from holosoma.managers.observation.terms.unified import target_pos_b


class _FakeMotionCommand:
    def __init__(self, target_xy_w: torch.Tensor | None):
        if target_xy_w is not None:
            self.target_xy_w = target_xy_w


class _FakeCommandManager:
    def __init__(self, motion_command):
        self._motion_command = motion_command

    def get_state(self, name):
        assert name == "motion_command"
        return self._motion_command


class _FakeSimulator:
    def __init__(self, robot_root_states):
        self.robot_root_states = robot_root_states


class _FakeEnv:
    def __init__(self, robot_xy: torch.Tensor, target_xy: torch.Tensor | None, yaw: float = 0.0):
        n = robot_xy.shape[0]
        root = torch.zeros(n, 13)
        root[:, :2] = robot_xy
        self.simulator = _FakeSimulator(root)
        self.base_quat = torch.tensor([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]).repeat(n, 1)
        self.command_manager = _FakeCommandManager(_FakeMotionCommand(target_xy))
        self.num_envs = n
        self.device = "cpu"


def _env(dx: float, dy: float, yaw: float = 0.0) -> _FakeEnv:
    """Robot at the origin, target at (dx, dy) -- so with yaw=0 the heading-frame offset is
    exactly (dx, dy) and the expected values are readable by inspection."""
    return _FakeEnv(torch.zeros(1, 2), torch.tensor([[dx, dy]]), yaw=yaw)


# ----------------------------------------------------------------------------------------------
# Property 1: the default is an exact no-op.
# ----------------------------------------------------------------------------------------------


def test_default_returns_raw_offset_in_metres():
    """distance_scale defaults to 0.0 => the raw heading-frame offset, unchanged. This is the
    pre-fix behavior and must survive untouched so existing checkpoints/configs are unaffected."""
    out = target_pos_b(_env(6.0, 0.0))
    assert torch.allclose(out, torch.tensor([[6.0, 0.0]]), atol=1e-6)


def test_explicit_zero_matches_default():
    assert torch.allclose(target_pos_b(_env(3.0, 4.0)), target_pos_b(_env(3.0, 4.0), distance_scale=0.0))


def test_negative_scale_is_treated_as_off_not_as_an_error_at_term_level():
    """The load-time validator rejects negatives (config_types/multi_skill.py); the term itself
    must still degrade to the safe raw path rather than producing a sign-flipped magnitude if one
    ever reaches it by another route."""
    out = target_pos_b(_env(6.0, 0.0), distance_scale=-1.0)
    assert torch.allclose(out, torch.tensor([[6.0, 0.0]]), atol=1e-6)


# ----------------------------------------------------------------------------------------------
# Property 2: when on -- 2 dims, bounded, direction preserved.
# ----------------------------------------------------------------------------------------------


def test_output_is_still_two_dimensional_when_compressed():
    """The load-bearing property for warm-start: compression must NOT widen the observation.
    A direction(2) + distance(1) encoding would have been 3 and broken every checkpoint."""
    assert target_pos_b(_env(6.0, 2.0), distance_scale=5.0).shape == (1, 2)


def test_magnitude_is_bounded_by_one_even_at_absurd_range():
    """tanh saturates, so no target distance can reproduce the raw-metres blowup this fixes.

    Bound is <= 1.0, not < 1.0: mathematically tanh never reaches 1, but in float32 tanh(2000.0)
    rounds to exactly 1.0. That is still bounded (the only property that matters here), but the
    strict-inequality phrasing would be wrong -- this test pins the achievable bound, not the
    idealized one."""
    for d in (7.0, 20.0, 100.0, 10_000.0):
        out = target_pos_b(_env(d, 0.0), distance_scale=5.0)
        assert out.norm().item() <= 1.0


def test_direction_is_preserved_exactly():
    """Only the magnitude is compressed -- the unit direction must be untouched, or the policy
    would be aiming somewhere other than the commanded target."""
    raw = target_pos_b(_env(3.0, 4.0))
    compressed = target_pos_b(_env(3.0, 4.0), distance_scale=5.0)
    assert torch.allclose(raw / raw.norm(), compressed / compressed.norm(), atol=1e-6)


def test_magnitude_equals_tanh_of_distance_over_scale():
    """Pins the exact formula, not just its qualitative shape: ||out|| == tanh(d / scale)."""
    out = target_pos_b(_env(3.0, 4.0), distance_scale=5.0)  # d == 5.0 exactly
    assert out.norm().item() == pytest.approx(math.tanh(1.0), abs=1e-6)


def test_strictly_monotone_in_distance():
    """Saturating is fine; losing the ordering is not -- 'far' must still read as larger than
    'near' or the compression would destroy the information it is meant to preserve."""
    norms = [target_pos_b(_env(d, 0.0), distance_scale=5.0).norm().item() for d in (1.0, 3.0, 5.0, 9.0)]
    assert all(a < b for a, b in zip(norms, norms[1:]))


def test_zero_distance_returns_zero_not_an_arbitrary_unit_vector():
    """The epsilon guard protects the DIVISION only; the tanh numerator must still collapse the
    whole vector to 0. A naive `offset / max(d, eps)` would emit a full-magnitude unit vector for a
    robot standing exactly on its target."""
    out = target_pos_b(_env(0.0, 0.0), distance_scale=5.0)
    assert torch.allclose(out, torch.zeros(1, 2), atol=1e-6)
    assert torch.isfinite(out).all()


def test_scale_controls_the_compression_rate():
    """Smaller scale => saturates sooner. Pins that the parameter actually does what its docstring
    says, so a future retune has a defined direction."""
    near = target_pos_b(_env(5.0, 0.0), distance_scale=1.0).norm().item()
    far = target_pos_b(_env(5.0, 0.0), distance_scale=20.0).norm().item()
    assert near > far


# ----------------------------------------------------------------------------------------------
# Pre-existing behavior that must not regress.
# ----------------------------------------------------------------------------------------------


def test_missing_target_still_returns_zeros_with_compression_on():
    """The `target_xy_w is None` guard (locomotion-only debugging configs) predates this change and
    must short-circuit BEFORE the compression math -- otherwise it would divide a zeros tensor."""
    env = _FakeEnv(torch.zeros(2, 2), None)
    out = target_pos_b(env, distance_scale=5.0)
    assert out.shape == (2, 2)
    assert torch.allclose(out, torch.zeros(2, 2))


def test_heading_frame_rotation_still_applied_under_compression():
    """Compression happens AFTER the heading-frame transform, so a rotated robot must still see the
    target rotated into its own frame -- a target dead ahead reads +x regardless of world yaw."""
    out = target_pos_b(_env(0.0, 6.0, yaw=math.pi / 2), distance_scale=5.0)
    assert out[0, 0].item() == pytest.approx(math.tanh(6.0 / 5.0), abs=1e-5)
    assert out[0, 1].item() == pytest.approx(0.0, abs=1e-5)


def test_batch_of_mixed_distances_is_handled_per_env():
    """keepdim/broadcasting correctness: each env must be compressed by its OWN distance, not by a
    batch-pooled one."""
    env = _FakeEnv(torch.zeros(3, 2), torch.tensor([[1.0, 0.0], [5.0, 0.0], [50.0, 0.0]]))
    out = target_pos_b(env, distance_scale=5.0)
    expected = [math.tanh(1.0 / 5.0), math.tanh(5.0 / 5.0), math.tanh(50.0 / 5.0)]
    for i, e in enumerate(expected):
        assert out[i, 0].item() == pytest.approx(e, abs=1e-6)
