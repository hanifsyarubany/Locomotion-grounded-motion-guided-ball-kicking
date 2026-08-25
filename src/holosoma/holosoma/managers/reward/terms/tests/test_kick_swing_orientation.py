"""Unit tests for the two swing-phase orientation penalties ported from RoboNaldo
(arXiv:2606.11092), 2026-08-04 -- ``penalty_kick_swing_orientation`` (pelvis) and
``penalty_kick_swing_torso_orientation`` (torso), both in managers/reward/terms/locomotion.py.

Both are deliberately UNGATED (no phase check, no recovery-gate multiply) unlike their
``penalty_kick_recovery_stand_*`` siblings -- these fire across the whole kick episode, including
the swing. Load-bearing properties covered below:

1. ``penalty_kick_swing_orientation`` is a pure passthrough of ``_stand_orientation_error`` (the
   SAME L1+deadzone pelvis-tilt error the locomotion/kick-recovery families already use) -- proven
   by direct equality against that helper, not a reimplementation of the math.
2. ``penalty_kick_swing_torso_orientation`` reads the TORSO body specifically, independent of
   pelvis tilt -- proven by a fake env where pelvis is level but torso is tilted (and vice versa).

Fakes provide exactly what ``_stand_orientation_error``/``_torso_orientation_error`` read:
``env.base_quat`` (pelvis, xyzw) for the former; ``env.body_names`` + ``env.simulator._rigid_body_rot``
for the latter. Real ``quat_rotate_inverse``/``gravity_vector`` are used (not reimplemented), so
these tests exercise the real rotation math, just with hand-picked quaternions.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from holosoma.managers.reward.terms import locomotion as loco_terms

_IDENTITY_XYZW = [0.0, 0.0, 0.0, 1.0]


def _quat_pitch_xyzw(angle_rad: float) -> list[float]:
    """xyzw quaternion for a pure pitch (rotation about Y) of ``angle_rad``."""
    half = angle_rad / 2.0
    return [0.0, float(torch.sin(torch.tensor(half))), 0.0, float(torch.cos(torch.tensor(half)))]


def _make_env(base_quats: list[list[float]], body_names=None, body_quats=None, device: str = "cpu"):
    n = len(base_quats)
    env = SimpleNamespace(
        num_envs=n,
        device=device,
        base_quat=torch.tensor(base_quats, dtype=torch.float32),
    )
    if body_names is not None:
        env.body_names = list(body_names)
        env.simulator = SimpleNamespace(_rigid_body_rot=torch.tensor(body_quats, dtype=torch.float32))
    return env


# ---------------------------------------------------------------- pelvis (penalty_kick_swing_orientation)
def test_swing_orientation_zero_at_identity():
    env = _make_env([_IDENTITY_XYZW, _IDENTITY_XYZW])
    out = loco_terms.penalty_kick_swing_orientation(env)
    assert torch.allclose(out, torch.zeros(2), atol=1e-6)


def test_swing_orientation_is_pure_passthrough_of_stand_orientation_error():
    """No gate, no scale -- must be byte-identical to _stand_orientation_error's own output for
    the same input, distinguishing this from the gated penalty_kick_recovery_stand_orientation
    sibling."""
    env = _make_env([_IDENTITY_XYZW, _quat_pitch_xyzw(0.3)])
    for deadzone in (0.0, 0.05, 0.5):
        expected = loco_terms._stand_orientation_error(env, deadzone)
        actual = loco_terms.penalty_kick_swing_orientation(env, deadzone=deadzone)
        assert torch.equal(actual, expected)


def test_swing_orientation_nonzero_for_a_real_tilt():
    env = _make_env([_quat_pitch_xyzw(0.3)])
    out = loco_terms.penalty_kick_swing_orientation(env)
    assert out.item() > 0.0


def test_swing_orientation_deadzone_absorbs_small_tilt():
    """A tilt whose |g_xy| sits below the deadzone must read exactly 0 -- the deadzone this
    project already found necessary for penalty_stand_orientation (see that history in
    _stand_orientation_error's own docstring) applies identically here since the math is shared."""
    small_tilt = 0.01  # rad, |g_xy| ~= sin(0.01) ~= 0.01
    env = _make_env([_quat_pitch_xyzw(small_tilt)])
    out = loco_terms.penalty_kick_swing_orientation(env, deadzone=0.5)
    assert out.item() == 0.0


def test_swing_orientation_per_env_independent():
    env = _make_env([_IDENTITY_XYZW, _quat_pitch_xyzw(0.3)])
    out = loco_terms.penalty_kick_swing_orientation(env)
    assert out[0].item() == 0.0
    assert out[1].item() > 0.0


# ---------------------------------------------------------------- torso (penalty_kick_swing_torso_orientation)
def test_swing_torso_orientation_zero_at_identity():
    env = _make_env(
        base_quats=[_IDENTITY_XYZW],
        body_names=["pelvis", "torso_link"],
        body_quats=[[_IDENTITY_XYZW, _IDENTITY_XYZW]],
    )
    out = loco_terms.penalty_kick_swing_torso_orientation(env)
    assert torch.allclose(out, torch.zeros(1), atol=1e-6)


def test_swing_torso_orientation_reads_torso_not_pelvis():
    """Pelvis (base_quat) tilted, torso (in body_names/_rigid_body_rot) level -- the torso term
    must read ~0 despite the pelvis being way off, proving it resolves the TORSO body
    specifically and does not silently fall back to base_quat."""
    env = _make_env(
        base_quats=[_quat_pitch_xyzw(0.5)],  # pelvis: way off vertical
        body_names=["pelvis", "torso_link"],
        body_quats=[[_quat_pitch_xyzw(0.5), _IDENTITY_XYZW]],  # torso: perfectly level
    )
    out = loco_terms.penalty_kick_swing_torso_orientation(env)
    assert torch.allclose(out, torch.zeros(1), atol=1e-6)


def test_swing_torso_orientation_nonzero_when_torso_itself_tilted():
    env = _make_env(
        base_quats=[_IDENTITY_XYZW],  # pelvis: level
        body_names=["pelvis", "torso_link"],
        body_quats=[[_IDENTITY_XYZW, _quat_pitch_xyzw(0.4)]],  # torso: tilted
    )
    out = loco_terms.penalty_kick_swing_torso_orientation(env)
    assert out.item() > 0.0


def test_swing_torso_orientation_matches_real_rotation_math():
    """Cross-check against the real quat_rotate_inverse/gravity_vector pipeline directly, not a
    hand re-derived analytic value -- proves the WIRING (right body index, right tensors) is
    correct, given the underlying rotation math is already exercised elsewhere in this project."""
    from holosoma.managers.observation.terms.locomotion import gravity_vector
    from holosoma.utils.rotations import quat_rotate_inverse

    torso_quat = torch.tensor([_quat_pitch_xyzw(0.35)])
    env = _make_env(
        base_quats=[_IDENTITY_XYZW],
        body_names=["pelvis", "torso_link"],
        body_quats=[[_IDENTITY_XYZW, _quat_pitch_xyzw(0.35)]],
    )
    expected_projected = quat_rotate_inverse(torso_quat, gravity_vector(env), w_last=True)
    expected = torch.norm(expected_projected[:, :2], dim=-1)
    actual = loco_terms.penalty_kick_swing_torso_orientation(env)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_swing_torso_orientation_respects_custom_body_name():
    """torso_body_name is a real, honored parameter -- not hardcoded to 'torso_link'."""
    env = _make_env(
        base_quats=[_IDENTITY_XYZW],
        body_names=["pelvis", "not_torso_link"],
        body_quats=[[_IDENTITY_XYZW, _quat_pitch_xyzw(0.4)]],
    )
    out = loco_terms.penalty_kick_swing_torso_orientation(env, torso_body_name="not_torso_link")
    assert out.item() > 0.0


def test_swing_torso_orientation_per_env_independent():
    env = _make_env(
        base_quats=[_IDENTITY_XYZW, _IDENTITY_XYZW],
        body_names=["pelvis", "torso_link"],
        body_quats=[
            [_IDENTITY_XYZW, _IDENTITY_XYZW],
            [_IDENTITY_XYZW, _quat_pitch_xyzw(0.4)],
        ],
    )
    out = loco_terms.penalty_kick_swing_torso_orientation(env)
    assert out[0].item() == 0.0
    assert out[1].item() > 0.0
