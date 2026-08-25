"""Unit tests for MultiMotionLoader.insert_segment_at_motion_boundary's index-shift math.

Zero test coverage existed for MultiMotionLoader/extend_with_segments before this (motion_dir was
never exercised by any shipped config), and the boundary-shift arithmetic here is exactly the kind
of code that's easy to get off-by-one wrong -- these tests bypass real .npz file I/O by constructing
a bare instance and hand-setting its internal tensors, so they run fast and need no motion assets.
"""

import torch

from holosoma.managers.command.terms.wbt import MultiMotionLoader


def _make_loader(lengths: list[int]) -> MultiMotionLoader:
    """N motions of given lengths; each frame's joint_pos value == its own motion index, so tests
    can verify CONTENT ends up in the right place after splicing, not just boundary indices."""
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


def test_append_into_first_of_three_motions_only_shifts_later_motions():
    ml = _make_loader([5, 4, 3])  # motion0=[0,5) motion1=[5,9) motion2=[9,12)
    ml.insert_segment_at_motion_boundary(0, _segment(2, 99.0), at_start=False)

    assert torch.equal(ml._motion_start_idx, torch.tensor([0, 7, 11]))
    assert torch.equal(ml._motion_end_idx, torch.tensor([7, 11, 14]))
    assert ml.time_step_total == 14
    assert torch.all(ml._joint_pos[5:7, 0] == 99.0)  # inserted content
    assert torch.all(ml._joint_pos[7:11, 0] == 1.0)  # old motion 1, shifted
    assert torch.all(ml._joint_pos[11:14, 0] == 2.0)  # old motion 2, shifted


def test_prepend_into_middle_motion_grows_backward_without_disturbing_earlier_motions():
    ml = _make_loader([5, 4, 3])  # motion0=[0,5) motion1=[5,9) motion2=[9,12)
    ml.insert_segment_at_motion_boundary(1, _segment(3, 77.0), at_start=True)

    assert torch.equal(ml._motion_start_idx, torch.tensor([0, 5, 12]))
    assert torch.equal(ml._motion_end_idx, torch.tensor([5, 12, 15]))
    assert ml.time_step_total == 15
    assert torch.all(ml._joint_pos[0:5, 0] == 0.0)  # motion 0 untouched
    assert torch.all(ml._joint_pos[5:8, 0] == 77.0)  # inserted content
    assert torch.all(ml._joint_pos[8:12, 0] == 1.0)  # old motion 1, shifted
    assert torch.all(ml._joint_pos[12:15, 0] == 2.0)  # old motion 2, shifted


def test_n1_reduction_matches_old_extend_with_segments_exactly():
    """motion_idx=0/at_start=True and motion_idx=num_motions-1/at_start=False must reproduce the
    pre-existing extend_with_segments behavior bit-for-bit -- this is what makes an N=1 config
    through the new per-skill mechanism a valid regression baseline against today's single-clip
    path."""
    ml_a = _make_loader([6])
    ml_a.insert_segment_at_motion_boundary(0, _segment(2, 5.0), at_start=True)
    ml_a.insert_segment_at_motion_boundary(0, _segment(3, 8.0), at_start=False)

    ml_b = _make_loader([6])
    ml_b.extend_with_segments(_segment(2, 5.0), prepend=True)
    ml_b.extend_with_segments(_segment(3, 8.0), prepend=False)

    assert torch.equal(ml_a._joint_pos, ml_b._joint_pos)
    assert torch.equal(ml_a._motion_start_idx, ml_b._motion_start_idx)
    assert torch.equal(ml_a._motion_end_idx, ml_b._motion_end_idx)


def test_ascending_multi_insert_prepend_and_append_across_all_motions():
    """The realistic per-skill usage pattern: loop prepend across all N motions (ascending), then
    loop append across all N motions (ascending) -- each motion should end up with its OWN
    prepend/append markers bracketing its OWN original content, not another motion's."""
    ml = _make_loader([5, 4, 3])
    for m in range(3):
        ml.insert_segment_at_motion_boundary(m, _segment(1, 100.0 + m), at_start=True)
    for m in range(3):
        ml.insert_segment_at_motion_boundary(m, _segment(1, 200.0 + m), at_start=False)

    actual_lengths = ml._motion_end_idx - ml._motion_start_idx
    assert torch.equal(actual_lengths, torch.tensor([7, 6, 5]))  # each +1 prepend +1 append
    assert ml.time_step_total == 18

    m1_start, m1_end = int(ml._motion_start_idx[1]), int(ml._motion_end_idx[1])
    assert ml._joint_pos[m1_start, 0].item() == 101.0  # motion 1's own prepend marker
    assert ml._joint_pos[m1_end - 1, 0].item() == 201.0  # motion 1's own append marker
    assert torch.any(ml._joint_pos[m1_start:m1_end, 0] == 1.0)  # motion 1's real content, in between


def test_per_motion_snapshot_must_be_captured_inside_the_append_loop_not_before_it():
    """Regression test for a real bug caught via a live 2-skill IsaacSim run: capturing
    "each motion's real-content-end index" (MotionCommand.pre_recovery_motion_end_idx, used to gate
    shooting rewards to the actual swing rather than the recovery/hold tail) as ONE batch snapshot
    BEFORE looping appends across all motions goes stale for every motion after the first, because
    an EARLIER motion's append/hold insertion (later in the same loop) shifts every LATER motion's
    absolute position -- but a pre-loop snapshot never sees that shift. The correct approach
    (MotionCommand.setup() as fixed) captures each motion's end index INSIDE the loop, immediately
    before that motion's own append runs -- by which point all earlier motions' shifts have already
    landed. This test proves the difference numerically, independent of any full env."""
    # 2 motions, lengths 5 and 4 (matching this file's other tests' scale); motion 0 gets an
    # append of 3 frames, motion 1 gets an append of 2 frames -- mirrors real recovery+hold sizes.
    ml_batch = _make_loader([5, 4])
    batch_snapshot = ml_batch._motion_end_idx.clone()  # WRONG pattern: captured before the loop
    for m in range(2):
        ml_batch.insert_segment_at_motion_boundary(m, _segment(3 if m == 0 else 2, 9.0), at_start=False)

    ml_correct = _make_loader([5, 4])
    correct_snapshot = []
    for m in range(2):
        correct_snapshot.append(int(ml_correct._motion_end_idx[m].item()))  # RIGHT: captured in-loop
        ml_correct.insert_segment_at_motion_boundary(m, _segment(3 if m == 0 else 2, 9.0), at_start=False)
    correct_snapshot = torch.tensor(correct_snapshot)

    # Motion 0's own snapshot value is identical either way -- nothing shifts it before its own append.
    assert batch_snapshot[0] == correct_snapshot[0] == 5

    # Motion 1's snapshot: the batch approach is stale by exactly motion 0's inserted frame count (3).
    assert int(batch_snapshot[1]) == 9  # STALE: motion 1's pre-append end in the ORIGINAL numbering
    assert int(correct_snapshot[1]) == 12  # CORRECT: 9 (original) + 3 (motion 0's append shifted it)
    assert int(ml_correct._motion_end_idx[1]) - int(ml_correct._motion_start_idx[1]) == 4 + 2  # sanity: final motion 1 length
    assert correct_snapshot[1] == int(ml_correct._motion_start_idx[1]) + 4  # snapshot lands exactly at real-content-end
