"""Unit tests for get_skill_motion_boundaries_metadata, get_skill_pre_recovery_metadata,
get_skill_ball_target_metadata and get_kick_recovery_locomotion_flip_metadata: the per-skill
(start, end) frame indices, whole-authored-clip-end (pre_recovery_motion_end_idx) frame index,
(ball_x, ball_y)/(target_x, target_y), and the Stage D flip flag attached as ONNX metadata so
deployment code (RoboJuDo's UnifiedLocoKickPolicy / mujoco_kick_rollout_worker.py) can address a
specific skill's motion, detect where its whole authored/raw clip ends, know its own configured
ball/target geometry, AND know whether to actually auto-return to locomotion at that instant (the
flag) instead of always implicitly getting skill 0's motion with a hardcoded, skill-unaware
ball/target position and the older "embedded clip has fully plateaued" auto-return heuristic.

Isolated with a bare fake motion_command/env (only the specific attributes each function touches),
matching this test suite's existing bare-instance convention -- no real env/onnx needed.
"""

import pytest
import torch

from holosoma.utils.inference_helpers import (
    get_kick_recovery_locomotion_flip_metadata,
    get_skill_ball_target_metadata,
    get_skill_motion_boundaries_metadata,
    get_skill_pre_recovery_metadata,
)


class _FakeMotion:
    def __init__(self, start_idx: list[int], end_idx: list[int]):
        self.motion_start_idx = torch.tensor(start_idx, dtype=torch.long)
        self.motion_end_idx = torch.tensor(end_idx, dtype=torch.long)


class _FakeMotionCommand:
    def __init__(
        self,
        motion: _FakeMotion,
        has_ball: bool = False,
        ball_xy=None,
        target_xy=None,
        pre_recovery_motion_end_idx: list[int] | None = None,
    ):
        self.motion = motion
        self.has_ball = has_ball
        if has_ball:
            self.ball_reset_state_per_motion = torch.tensor(ball_xy, dtype=torch.float32)
            self.target_nominal_local_per_motion = torch.tensor(target_xy, dtype=torch.float32)
        if pre_recovery_motion_end_idx is not None:
            self.pre_recovery_motion_end_idx = torch.tensor(pre_recovery_motion_end_idx, dtype=torch.long)


def test_legacy_single_motion_gives_one_skill_covering_the_whole_buffer():
    mc = _FakeMotionCommand(_FakeMotion([0], [900]))
    meta = get_skill_motion_boundaries_metadata(mc)
    assert meta == {"skill_motion_start_idx": [0], "skill_motion_end_idx": [900]}


def test_n_skill_mode_gives_one_entry_per_skill_in_declaration_order():
    mc = _FakeMotionCommand(_FakeMotion([0, 432], [432, 847]))
    meta = get_skill_motion_boundaries_metadata(mc)
    assert meta == {"skill_motion_start_idx": [0, 432], "skill_motion_end_idx": [432, 847]}


def test_no_motion_command_returns_empty_dict_not_an_error():
    assert get_skill_motion_boundaries_metadata(None) == {}


def test_returned_values_are_plain_python_ints_not_tensors():
    mc = _FakeMotionCommand(_FakeMotion([0, 100], [100, 250]))
    meta = get_skill_motion_boundaries_metadata(mc)
    for v in meta["skill_motion_start_idx"] + meta["skill_motion_end_idx"]:
        assert isinstance(v, int)


def test_ball_target_metadata_absent_when_env_has_no_ball():
    mc = _FakeMotionCommand(_FakeMotion([0], [900]), has_ball=False)
    assert get_skill_ball_target_metadata(mc) == {}


def test_ball_target_metadata_absent_when_no_motion_command():
    assert get_skill_ball_target_metadata(None) == {}


def test_ball_target_metadata_one_entry_per_skill_in_declaration_order():
    mc = _FakeMotionCommand(
        _FakeMotion([0, 432], [432, 847]),
        has_ball=True,
        ball_xy=[[7.0, -0.46], [3.5, 0.6]],
        target_xy=[[7.84, -0.46], [8.5, 0.6]],
    )
    meta = get_skill_ball_target_metadata(mc)

    def _flat(pairs):
        return [v for pair in pairs for v in pair]

    # float32 round-trip -- compare approximately, not exactly
    assert _flat(meta["skill_ball_xy"]) == pytest.approx(_flat([[7.0, -0.46], [3.5, 0.6]]), abs=1e-6)
    assert _flat(meta["skill_target_xy"]) == pytest.approx(_flat([[7.84, -0.46], [8.5, 0.6]]), abs=1e-6)


class _FakeEnv:
    def __init__(self, kick_recovery_locomotion_flip_enabled: bool | None = None):
        if kick_recovery_locomotion_flip_enabled is not None:
            self._kick_recovery_locomotion_flip_enabled = kick_recovery_locomotion_flip_enabled


def test_flip_metadata_true_when_flag_enabled():
    env = _FakeEnv(kick_recovery_locomotion_flip_enabled=True)
    assert get_kick_recovery_locomotion_flip_metadata(env) == {"kick_recovery_locomotion_flip_enabled": True}


def test_flip_metadata_false_when_flag_disabled():
    env = _FakeEnv(kick_recovery_locomotion_flip_enabled=False)
    assert get_kick_recovery_locomotion_flip_metadata(env) == {"kick_recovery_locomotion_flip_enabled": False}


def test_flip_metadata_defaults_false_when_attribute_absent():
    env = _FakeEnv()  # no _kick_recovery_locomotion_flip_enabled at all -- non-UnifiedManager env class
    assert get_kick_recovery_locomotion_flip_metadata(env) == {"kick_recovery_locomotion_flip_enabled": False}


def test_flip_metadata_absent_when_no_env():
    assert get_kick_recovery_locomotion_flip_metadata(None) == {}


def test_pre_recovery_metadata_one_entry_per_skill_in_declaration_order():
    mc = _FakeMotionCommand(_FakeMotion([0, 432], [432, 847]), pre_recovery_motion_end_idx=[400, 820])
    meta = get_skill_pre_recovery_metadata(mc)
    assert meta == {"skill_pre_recovery_motion_end_idx": [400, 820]}


def test_pre_recovery_metadata_legacy_single_motion():
    mc = _FakeMotionCommand(_FakeMotion([0], [900]), pre_recovery_motion_end_idx=[870])
    meta = get_skill_pre_recovery_metadata(mc)
    assert meta == {"skill_pre_recovery_motion_end_idx": [870]}


def test_pre_recovery_metadata_absent_when_no_motion_command():
    assert get_skill_pre_recovery_metadata(None) == {}


def test_pre_recovery_metadata_returned_values_are_plain_python_ints_not_tensors():
    mc = _FakeMotionCommand(_FakeMotion([0, 432], [432, 847]), pre_recovery_motion_end_idx=[400, 820])
    meta = get_skill_pre_recovery_metadata(mc)
    for v in meta["skill_pre_recovery_motion_end_idx"]:
        assert isinstance(v, int)
