"""Unit tests for FastSACAgent's MuJoCo locomotion->kick handoff fall-rate/hit-rate scan worker
(`_loco_to_kick_handoff_scan_worker`, 2026-08-30) -- the reverse-direction sim2sim counterpart of
`_kick_to_loco_flip_scan_worker`, and the sim2sim analogue of training's
mid_episode_kick_entry_prob. Mirrors test_mujoco_kick_to_loco_flip_scan.py's own bare-instance
convention exactly -- no real onnx, MuJoCo, or wandb run needed.

`record_loco_to_kick_handoff_scan` returns `(fall_rate, hit_rate, pre_handoff_fail_rate)` -- all
three can be None independently (fall_rate/hit_rate are None when every trial fell during the walk
itself, before ever reaching the handoff; pre_handoff_fail_rate is only None on an outright scan
failure). fall_rate and hit_rate share the SAME "reached the handoff" denominator on the worker
script's own side -- nothing about that sharing is visible here, since this test layer only checks
that whatever the wrapper returns gets queued under the right keys.

Unlike record_survival_scan/record_kick_to_loco_flip_scan, this worker takes no kick_aim_info
parameter -- the scan REQUIRES kick_aim_enabled unconditionally (see
mujoco_loco_to_kick_handoff_scan.py's own module docstring for why), so there is no per-skill
gather to test here.
"""

import queue
from unittest.mock import patch

from holosoma.agents.fast_sac.fast_sac_agent import FastSACAgent


class _FakeUnwrappedEnv:
    def __init__(self, skill_motion_training_ratios):
        self._skill_motion_training_ratios = skill_motion_training_ratios


def _make_agent(skill_motion_training_ratios, log_dir="/tmp/holosoma_test_log_dir") -> FastSACAgent:
    a = object.__new__(FastSACAgent)
    a.unwrapped_env = _FakeUnwrappedEnv(skill_motion_training_ratios)
    a.log_dir = log_dir
    a._loco_to_kick_handoff_scan_result_queue = queue.Queue()
    a.global_step = 12345
    return a


def _drain(q: queue.Queue) -> list[tuple]:
    items = []
    while True:
        try:
            items.append(q.get_nowait())
        except queue.Empty:
            return items


class TestMujocoLocoToKickHandoffScanWorker:
    def test_all_three_rates_queued_under_correct_per_skill_keys(self):
        a = _make_agent([0.5])
        with patch(
            "holosoma.record_mujoco_loco_to_kick_handoff_scan.record_loco_to_kick_handoff_scan",
            return_value=(0.1, 0.6, 0.05),
        ):
            a._loco_to_kick_handoff_scan_worker(onnx_path="/fake/model.onnx", global_step=500, num_trials=32)

        items = _drain(a._loco_to_kick_handoff_scan_result_queue)
        keys_and_values = {key: val for (_step, key, val) in items}
        assert keys_and_values["Kick_skills_0/sim2sim/loco_to_kick_handoff_fall_rate"] == 0.1
        assert keys_and_values["Kick_skills_0/sim2sim/loco_to_kick_handoff_ball_hit_rate"] == 0.6
        assert keys_and_values["Kick_skills_0/sim2sim/loco_to_kick_handoff_pre_handoff_fail_rate"] == 0.05

    def test_fall_and_hit_rate_none_are_simply_not_queued(self):
        """None means every trial fell during the walk itself -- a valid, expected state (nothing
        to measure the handoff against), not an error. pre_handoff_fail_rate still queues
        normally. fall_rate and hit_rate are independent Nones here (both happen to be None
        together in this scenario, but the worker doesn't assume they're coupled)."""
        a = _make_agent([0.5])
        with patch(
            "holosoma.record_mujoco_loco_to_kick_handoff_scan.record_loco_to_kick_handoff_scan",
            return_value=(None, None, 1.0),
        ):
            a._loco_to_kick_handoff_scan_worker(onnx_path="/fake/model.onnx", global_step=600, num_trials=32)

        items = _drain(a._loco_to_kick_handoff_scan_result_queue)
        keys_and_values = {key: val for (_step, key, val) in items}
        assert "Kick_skills_0/sim2sim/loco_to_kick_handoff_fall_rate" not in keys_and_values
        assert "Kick_skills_0/sim2sim/loco_to_kick_handoff_ball_hit_rate" not in keys_and_values
        assert keys_and_values["Kick_skills_0/sim2sim/loco_to_kick_handoff_pre_handoff_fail_rate"] == 1.0

    def test_all_none_is_a_no_op_not_a_crash(self):
        """A scan failure (busy lock, timeout, crash) returns (None, None, None) -- nothing
        queued, no exception propagates."""
        a = _make_agent([0.5])
        with patch(
            "holosoma.record_mujoco_loco_to_kick_handoff_scan.record_loco_to_kick_handoff_scan",
            return_value=(None, None, None),
        ):
            a._loco_to_kick_handoff_scan_worker(onnx_path="/fake/model.onnx", global_step=700, num_trials=32)

        assert _drain(a._loco_to_kick_handoff_scan_result_queue) == []

    def test_multi_skill_rates_land_under_correct_per_skill_keys(self):
        a = _make_agent([0.5, 0.3])  # 2 skills

        def fake_record(**kwargs):
            if kwargs["skill_id"] == 0:
                return (0.2, 0.7, 0.1)
            return (0.4, 0.3, 0.0)

        with patch(
            "holosoma.record_mujoco_loco_to_kick_handoff_scan.record_loco_to_kick_handoff_scan",
            side_effect=fake_record,
        ) as mock_record:
            a._loco_to_kick_handoff_scan_worker(onnx_path="/fake/model.onnx", global_step=900, num_trials=32)

        assert len(mock_record.call_args_list) == 2
        items = _drain(a._loco_to_kick_handoff_scan_result_queue)
        keys_and_values = {key: val for (_step, key, val) in items}
        assert keys_and_values["Kick_skills_0/sim2sim/loco_to_kick_handoff_fall_rate"] == 0.2
        assert keys_and_values["Kick_skills_0/sim2sim/loco_to_kick_handoff_ball_hit_rate"] == 0.7
        assert keys_and_values["Kick_skills_1/sim2sim/loco_to_kick_handoff_fall_rate"] == 0.4
        assert keys_and_values["Kick_skills_1/sim2sim/loco_to_kick_handoff_ball_hit_rate"] == 0.3

    def test_single_skill_mode_still_uses_kick_skills_0_prefix(self):
        """Empty _skill_motion_training_ratios (legacy single-skill mode) must still land under
        'Kick_skills_0/...', matching the established convention every sibling sim2sim mechanism
        follows (num_skills = max(len(...), 1))."""
        a = _make_agent([])  # legacy single-skill mode
        with patch(
            "holosoma.record_mujoco_loco_to_kick_handoff_scan.record_loco_to_kick_handoff_scan",
            return_value=(0.0, 0.0, 0.0),
        ) as mock_record:
            a._loco_to_kick_handoff_scan_worker(onnx_path="/fake/model.onnx", global_step=1000, num_trials=32)

        assert len(mock_record.call_args_list) == 1
        assert mock_record.call_args_list[0].kwargs["skill_id"] == 0
        items = _drain(a._loco_to_kick_handoff_scan_result_queue)
        keys = {key for (_step, key, _val) in items}
        assert "Kick_skills_0/sim2sim/loco_to_kick_handoff_fall_rate" in keys
        assert "Kick_skills_0/sim2sim/loco_to_kick_handoff_ball_hit_rate" in keys

    def test_seed_varies_with_global_step(self):
        """Same rationale as every sibling scan's own seed=global_step -- identical trials across
        checkpoints would be a weaker read on whether the checkpoint actually improved."""
        a = _make_agent([0.5])
        with patch(
            "holosoma.record_mujoco_loco_to_kick_handoff_scan.record_loco_to_kick_handoff_scan",
            return_value=(0.0, 0.0, 0.0),
        ) as mock_record:
            a._loco_to_kick_handoff_scan_worker(onnx_path="/fake/model.onnx", global_step=54321, num_trials=32)

        assert mock_record.call_args_list[0].kwargs["seed"] == 54321
