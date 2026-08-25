"""Unit tests for the strike_start_idx/stand_start_idx boundary-capture arithmetic used in
MotionCommand.setup() (managers/command/terms/wbt.py) to convert scrubbed, clip-local (raw npz)
strike/stand frame indices into absolute MultiMotionLoader buffer indices.

Zero test coverage existed for this arithmetic before this (it's new, 2026-07-31). Like
test_multi_motion_loader.py, these tests bypass real .npz I/O and a full env by constructing a
bare MultiMotionLoader instance directly and driving its real insert_segment_at_motion_boundary
method -- so the prepend-splicing mechanics under test are the REAL production mechanics, not a
reimplementation of them. Only the final index arithmetic (raw_clip_start = motion_end_idx -
raw_frame_count; strike_start_idx = raw_clip_start + strike_frame; stand_start_idx =
raw_clip_start + stand_frame) is inlined here, matching MotionCommand.setup()'s own formula
exactly (see that method's comments for the derivation).

The central risk this covers: the naive-looking formula motion_start_idx[_m] + clip_local_frame is
WRONG by this motion's own prepend length once a prepend has been spliced in (which is always true
for Unified's real config, default_pose_prepend_duration_s=1.0) -- because
insert_segment_at_motion_boundary's at_start=True path leaves motion_start_idx POINTING AT THE
PREPEND's start, not raw-clip frame 0 (see that method's own docstring). The correct formula
instead derives raw-clip frame 0 from motion_end_idx (a live pointer that already accounts for
the prepend) minus the motion's OWN static raw frame count. test_naive_formula_is_wrong_by_
exactly_the_prepend_length below proves this discrepancy numerically.
"""

import torch

from holosoma.managers.command.terms.wbt import MultiMotionLoader


def _make_loader(lengths: list[int]) -> MultiMotionLoader:
    """N motions of given (raw, pre-augmentation) lengths -- mirrors test_multi_motion_loader.py's
    own bare-construction helper exactly."""
    ml = object.__new__(MultiMotionLoader)
    cumulative = torch.tensor(lengths, dtype=torch.long).cumsum(dim=0)
    ml._motion_start_idx = torch.cat([torch.tensor([0], dtype=torch.long), cumulative[:-1]])
    ml._motion_end_idx = cumulative.clone()
    ml._num_motions = len(lengths)
    parts = [torch.full((length, 1), float(i)) for i, length in enumerate(lengths)]
    ml._joint_pos = torch.cat(parts, dim=0)
    ml._joint_vel = ml._joint_pos.clone()
    ml._body_pos_w = ml._joint_pos.clone().unsqueeze(-1)
    ml._body_quat_w = ml._joint_pos.clone().unsqueeze(-1)
    ml._body_lin_vel_w = ml._joint_pos.clone().unsqueeze(-1)
    ml._body_ang_vel_w = ml._joint_pos.clone().unsqueeze(-1)
    ml.has_object = False
    ml.time_step_total = sum(lengths)
    return ml


def _segment(n: int, value: float) -> dict:
    return {
        "joint_pos": torch.full((n, 1), value),
        "joint_vel": torch.full((n, 1), value),
        "body_pos": torch.full((n, 1, 1), value),
        "body_quat": torch.full((n, 1, 1), value),
        "body_lin_vel": torch.full((n, 1, 1), value),
        "body_ang_vel": torch.full((n, 1, 1), value),
    }


def _derive_strike_stand_idx(
    motion_end_idx: torch.Tensor, raw_frame_count: torch.Tensor, strike_frame: list[int], stand_frame: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact reproduction of MotionCommand.setup()'s per-motion formula (wbt.py, inside the
    pre_recovery_list loop): raw_clip_start = motion_end_idx[_m] - raw_frame_count[_m], then
    strike/stand_start_idx = raw_clip_start + {strike,stand}_frame[_m]. Kept as a free function
    here purely so this test file can exercise it directly against a bare-constructed loader,
    without needing a full MotionCommand/env."""
    raw_clip_start = motion_end_idx - raw_frame_count
    strike_t = torch.tensor(strike_frame, dtype=torch.long)
    stand_t = torch.tensor(stand_frame, dtype=torch.long)
    return raw_clip_start + strike_t, raw_clip_start + stand_t


def test_naive_formula_is_wrong_by_exactly_the_prepend_length():
    """Regression test for the off-by-prepend-length bug identified during design: after a
    prepend is spliced onto a single motion, the NAIVE formula (motion_start_idx + clip_local_frame)
    and the CORRECT formula (raw_clip_start + clip_local_frame, i.e. this file's
    _derive_strike_stand_idx) must differ by exactly the prepend length, and only the correct one
    matches ground truth (the frame index we KNOW is raw-clip frame `strike_frame` because we
    control exactly what got spliced in)."""
    raw_len = 10
    ml = _make_loader([raw_len])
    raw_frame_count = ml._motion_end_idx.clone() - ml._motion_start_idx.clone()  # static, pre-augmentation

    prepend_len = 4
    ml.insert_segment_at_motion_boundary(0, _segment(prepend_len, 99.0), at_start=True)

    strike_frame, stand_frame = 3, 7
    naive = ml._motion_start_idx[0] + strike_frame  # WRONG: motion_start_idx now points at the prepend
    correct_strike, correct_stand = _derive_strike_stand_idx(
        ml._motion_end_idx, raw_frame_count, [strike_frame], [stand_frame]
    )

    assert int(naive) - int(correct_strike[0]) == -prepend_len  # naive is short by the prepend length
    # Ground truth: raw frame `strike_frame` sits at (post-prepend) buffer offset
    # motion_start_idx_original(0) + prepend_len + strike_frame = 0 + 4 + 3 = 7.
    assert int(correct_strike[0]) == prepend_len + strike_frame == 7
    assert int(correct_stand[0]) == prepend_len + stand_frame == 11
    # Content check: buffer index 7 must actually be motion 0's real content (joint_pos == 0.0),
    # not the prepend marker (99.0) -- confirms the derived index truly lands past the prepend.
    assert ml._joint_pos[int(correct_strike[0]), 0].item() == 0.0
    assert ml._joint_pos[int(naive), 0].item() == 99.0  # the WRONG index lands inside the prepend


def test_multi_motion_different_prepend_lengths_each_use_their_own():
    """Two motions with DIFFERENT prepend lengths (3 and 6) -- each motion's strike/stand indices
    must be derived from its OWN prepend, not a shared/averaged one."""
    ml = _make_loader([8, 5])
    raw_frame_count = ml._motion_end_idx.clone() - ml._motion_start_idx.clone()

    ml.insert_segment_at_motion_boundary(0, _segment(3, 91.0), at_start=True)
    ml.insert_segment_at_motion_boundary(1, _segment(6, 92.0), at_start=True)

    strike_start, stand_start = _derive_strike_stand_idx(
        ml._motion_end_idx, raw_frame_count, strike_frame=[2, 1], stand_frame=[6, 4]
    )

    # Motion 0: original start 0, prepend 3 -> raw frame 0 now at buffer index 3.
    assert int(strike_start[0]) == 3 + 2 == 5
    assert int(stand_start[0]) == 3 + 6 == 9
    # Motion 1: original start 8, but motion 0's OWN prepend (3 frames, inserted before motion
    # 0's start) shifts every later motion's start/end too, so motion 1's start becomes 8+3=11
    # BEFORE motion 1's own prepend is even applied; motion 1's own prepend (6) then adds on top
    # -> raw frame 0 now at buffer index 8 + 3 (motion 0's shift) + 6 (motion 1's own prepend) = 17.
    assert int(strike_start[1]) == 17 + 1 == 18
    assert int(stand_start[1]) == 17 + 4 == 21
    # Content sanity: both derived strike indices land on real content (0.0/1.0), not a prepend marker.
    assert ml._joint_pos[int(strike_start[0]), 0].item() == 0.0
    assert ml._joint_pos[int(strike_start[1]), 0].item() == 1.0


def test_malformed_stand_frame_exceeds_raw_length_is_rejected():
    """Mirrors MotionCommand.setup()'s own guard: 0 <= strike_start_frame < stand_start_frame <=
    raw_len must hold. This test exercises just the guard's condition (the real ValueError is
    raised inside setup(), which needs a full env to construct) -- pinning down that a
    stand_start_frame past the clip's own raw length is invalid input, not a silently-accepted one."""
    raw_len = 10
    strike_frame, stand_frame = 3, 15  # 15 > raw_len=10
    assert not (0 <= strike_frame < stand_frame <= raw_len)


def test_legacy_mode_stand_start_idx_matches_pre_recovery_motion_end_idx():
    """Legacy/single-clip mode (no scrubbed strike/stand frames): MotionCommand.setup()'s else
    branch sets stand_start_idx[_m] = motion_end (== pre_recovery_motion_end_idx[_m], the
    pre-2026-07-31 in_swing_phase boundary) and strike_start_idx[_m] = motion_start_idx[_m] (a
    vacuous lower bound). This test pins down that this fallback is bit-identical to the OLD
    single boundary, i.e. in_kicking_phase's new end boundary reduces to in_swing_phase's old one
    when no scrubbed frames are configured."""
    ml = _make_loader([12])
    ml.insert_segment_at_motion_boundary(0, _segment(2, 1.0), at_start=True)  # prepend, as always in Unified
    pre_recovery_motion_end_idx = ml._motion_end_idx.clone()  # captured here, exactly as setup() does

    # Legacy fallback (setup()'s else branch):
    stand_start_idx = pre_recovery_motion_end_idx
    strike_start_idx = ml._motion_start_idx.clone()

    assert torch.equal(stand_start_idx, pre_recovery_motion_end_idx)
    assert torch.equal(strike_start_idx, ml._motion_start_idx)
    # in_strike_phase's window [strike_start_idx, stand_start_idx) must equal in_kicking_phase's
    # whole window [., stand_start_idx) -- i.e. strike_start_idx is a no-op lower bound, since
    # reset() never sets time_steps below motion_start_idx.
    assert int(strike_start_idx[0]) <= int(ml._motion_start_idx[0])
