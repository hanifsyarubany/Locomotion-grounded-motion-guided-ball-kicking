"""Unit tests for ``motion_head_velocity_smoothing`` (2026-08-11) -- the opt-in, default-off
in-memory ramp of a clip's leading velocity frames. See that module's own docstring for the
measured motivation (video_012 frame 0 is the highest-velocity frame of all 250, 7.6x the clip's
own median, decaying to below-median by frame 3).
"""

from __future__ import annotations

import torch

from holosoma.managers.command.terms.motion_head_velocity_smoothing import (
    head_velocity_ramp,
    smooth_motion_head_velocities,
)


def _make_channels(n_frames: int = 10, n_joints: int = 29, n_bodies: int = 51):
    """All-ones velocity channels, so any scaling is directly readable as the ramp itself."""
    return (
        torch.ones(n_frames, n_joints),
        torch.ones(n_frames, n_bodies, 3),
        torch.ones(n_frames, n_bodies, 3),
    )


def test_zero_frames_is_exact_no_op():
    jv, blv, bav = _make_channels()
    jv0, blv0, bav0 = jv.clone(), blv.clone(), bav.clone()

    smooth_motion_head_velocities(joint_vel=jv, body_lin_vel_w=blv, body_ang_vel_w=bav, n_frames=0)

    assert torch.equal(jv, jv0)
    assert torch.equal(blv, blv0)
    assert torch.equal(bav, bav0)


def test_negative_frames_is_also_a_no_op():
    jv, blv, bav = _make_channels()
    jv0 = jv.clone()
    smooth_motion_head_velocities(joint_vel=jv, body_lin_vel_w=blv, body_ang_vel_w=bav, n_frames=-5)
    assert torch.equal(jv, jv0)


def test_frame_zero_is_driven_to_exactly_rest():
    """The whole point: the clip must start at rest, because the prepend that precedes it ends at
    a static zero-velocity default pose."""
    jv, blv, bav = _make_channels()

    smooth_motion_head_velocities(joint_vel=jv, body_lin_vel_w=blv, body_ang_vel_w=bav, n_frames=3)

    assert torch.all(jv[0] == 0.0)
    assert torch.all(blv[0] == 0.0)
    assert torch.all(bav[0] == 0.0)


def test_frames_past_the_window_are_untouched():
    jv, blv, bav = _make_channels(n_frames=10)

    smooth_motion_head_velocities(joint_vel=jv, body_lin_vel_w=blv, body_ang_vel_w=bav, n_frames=3)

    assert torch.all(jv[3:] == 1.0)
    assert torch.all(blv[3:] == 1.0)
    assert torch.all(bav[3:] == 1.0)


def test_ramp_is_monotonically_increasing_across_the_window():
    jv, blv, bav = _make_channels(n_frames=10)

    smooth_motion_head_velocities(joint_vel=jv, body_lin_vel_w=blv, body_ang_vel_w=bav, n_frames=5)

    head = jv[:5, 0]
    assert torch.all(head[1:] > head[:-1]), f"ramp not monotonic: {head}"
    assert head[0] == 0.0
    assert head[-1] < 1.0  # reaches full value AT n_frames, not before it


def test_all_three_channels_get_the_same_ramp():
    jv, blv, bav = _make_channels(n_frames=8)

    smooth_motion_head_velocities(joint_vel=jv, body_lin_vel_w=blv, body_ang_vel_w=bav, n_frames=4)

    for i in range(4):
        expected = jv[i, 0]
        assert torch.allclose(blv[i], torch.full_like(blv[i], expected))
        assert torch.allclose(bav[i], torch.full_like(bav[i], expected))


def test_window_longer_than_clip_is_clamped_not_an_index_error():
    jv, blv, bav = _make_channels(n_frames=4)

    smooth_motion_head_velocities(joint_vel=jv, body_lin_vel_w=blv, body_ang_vel_w=bav, n_frames=100)

    assert jv.shape[0] == 4
    assert torch.all(jv[0] == 0.0)
    assert torch.all(torch.isfinite(jv))


def test_scales_real_magnitudes_rather_than_replacing_them():
    """Ramp is multiplicative -- the authored direction/shape of each frame survives, only its
    magnitude is attenuated."""
    jv = torch.tensor([[3.0, -4.0], [3.0, -4.0], [3.0, -4.0], [3.0, -4.0]])
    blv = torch.ones(4, 2, 3)
    bav = torch.ones(4, 2, 3)

    smooth_motion_head_velocities(joint_vel=jv, body_lin_vel_w=blv, body_ang_vel_w=bav, n_frames=2)

    # frame 1 keeps the 3:-4 ratio, just smaller
    assert jv[1, 0] / jv[1, 1] == 3.0 / -4.0
    assert abs(jv[1, 0]) < 3.0


def test_in_place_mutation_not_a_returned_copy():
    """MotionLoader's buffers are mutated through slice views -- the function must write through,
    exactly like correct_kick_foot_ankle_pitch does."""
    buf = torch.ones(10, 29)
    view = buf[2:8]  # a genuine slice view into a larger buffer

    smooth_motion_head_velocities(
        joint_vel=view,
        body_lin_vel_w=torch.ones(6, 51, 3),
        body_ang_vel_w=torch.ones(6, 51, 3),
        n_frames=3,
    )

    assert torch.all(buf[2] == 0.0), "did not write through the slice into the parent buffer"
    assert torch.all(buf[:2] == 1.0), "wrote outside the given slice"
    assert torch.all(buf[5:] == 1.0), "wrote past the smoothing window"


def test_ramp_shape_endpoints_and_smoothness():
    ramp = head_velocity_ramp(5, device=torch.device("cpu"), dtype=torch.float32)

    assert ramp.shape == (5,)
    assert ramp[0] == 0.0
    assert torch.all(ramp >= 0.0) and torch.all(ramp < 1.0)
    # raised cosine: first step is smaller than a linear ramp's would be (eases in)
    assert ramp[1] < 1.0 / 5.0


def test_ramp_zero_length_is_empty():
    ramp = head_velocity_ramp(0, device=torch.device("cpu"), dtype=torch.float32)
    assert ramp.numel() == 0
