"""Unit tests for FastSACAgent's MuJoCo forced kick->locomotion flip alive-rate scan worker
(`_kick_to_loco_flip_scan_worker`, 2026-08-30) -- the sim2sim counterpart of training's
kick_abort_prob. Mirrors test_mujoco_survival_scan_direction.py's own bare-instance convention
exactly (same rationale: no real onnx, MuJoCo, or wandb run needed to verify the per-skill
kick_aim threading and queue/wandb-key wiring).

`record_kick_to_loco_flip_scan` returns `(alive_rate, pre_flip_fail_rate)` -- both can be None
independently (alive_rate is None when every trial fell before ever reaching its scheduled flip;
pre_flip_fail_rate is only ever None on an outright scan failure, never as a normal "nothing to
report" state, since it doesn't depend on kick_aim_enabled the way direction_success_rate does).
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
    a._kick_to_loco_flip_scan_result_queue = queue.Queue()
    a.global_step = 12345
    return a


def _drain(q: queue.Queue) -> list[tuple]:
    items = []
    while True:
        try:
            items.append(q.get_nowait())
        except queue.Empty:
            return items


class TestMujocoKickToLocoFlipScanWorker:
    def test_kick_aim_enabled_per_skill_reaches_record_kick_to_loco_flip_scan(self):
        a = _make_agent([0.5, 0.3])  # 2 skills

        def fake_record(**kwargs):
            return (1.0, 0.0)

        with patch(
            "holosoma.record_mujoco_kick_to_loco_flip_scan.record_kick_to_loco_flip_scan",
            side_effect=fake_record,
        ) as mock_record:
            a._kick_to_loco_flip_scan_worker(
                onnx_path="/fake/model.onnx",
                global_step=500,
                num_trials=32,
                kick_aim_info=([True, False], 45.0),
            )

        calls = {c.kwargs["skill_id"]: c.kwargs["kick_aim_enabled"] for c in mock_record.call_args_list}
        assert calls == {0: True, 1: False}

    def test_both_rates_queued_under_correct_per_skill_keys(self):
        a = _make_agent([0.5])
        with patch(
            "holosoma.record_mujoco_kick_to_loco_flip_scan.record_kick_to_loco_flip_scan",
            return_value=(0.9, 0.05),
        ):
            a._kick_to_loco_flip_scan_worker(
                onnx_path="/fake/model.onnx",
                global_step=500,
                num_trials=32,
                kick_aim_info=([True], 45.0),
            )

        items = _drain(a._kick_to_loco_flip_scan_result_queue)
        keys_and_values = {key: val for (_step, key, val) in items}
        assert keys_and_values["Kick_skills_0/sim2sim/kick_to_loco_random_flip_alive_rate"] == 0.9
        assert keys_and_values["Kick_skills_0/sim2sim/kick_to_loco_random_flip_pre_flip_fail_rate"] == 0.05

    def test_alive_rate_none_is_simply_not_queued(self):
        """None means every trial fell before ever reaching its scheduled flip -- a valid,
        expected state (nothing to measure the flip against), not an error.
        pre_flip_fail_rate still queues normally."""
        a = _make_agent([0.5])
        with patch(
            "holosoma.record_mujoco_kick_to_loco_flip_scan.record_kick_to_loco_flip_scan",
            return_value=(None, 1.0),
        ):
            a._kick_to_loco_flip_scan_worker(
                onnx_path="/fake/model.onnx",
                global_step=600,
                num_trials=32,
                kick_aim_info=([True], 45.0),
            )

        items = _drain(a._kick_to_loco_flip_scan_result_queue)
        keys_and_values = {key: val for (_step, key, val) in items}
        assert "Kick_skills_0/sim2sim/kick_to_loco_random_flip_alive_rate" not in keys_and_values
        assert keys_and_values["Kick_skills_0/sim2sim/kick_to_loco_random_flip_pre_flip_fail_rate"] == 1.0

    def test_both_none_is_a_no_op_not_a_crash(self):
        """A scan failure (busy lock, timeout, crash) returns (None, None) -- nothing queued,
        no exception propagates."""
        a = _make_agent([0.5])
        with patch(
            "holosoma.record_mujoco_kick_to_loco_flip_scan.record_kick_to_loco_flip_scan",
            return_value=(None, None),
        ):
            a._kick_to_loco_flip_scan_worker(
                onnx_path="/fake/model.onnx",
                global_step=700,
                num_trials=32,
                kick_aim_info=None,
            )

        assert _drain(a._kick_to_loco_flip_scan_result_queue) == []

    def test_kick_aim_info_none_disables_aim_for_every_skill(self):
        """kick_aim_info is None when the live env has no ball at all -- every skill must still
        get scanned (this scan needs no ball beyond what get_skill_ball_xy's own fallback
        provides), just with kick_aim_enabled=False throughout."""
        a = _make_agent([0.5, 0.3])

        with patch(
            "holosoma.record_mujoco_kick_to_loco_flip_scan.record_kick_to_loco_flip_scan",
            return_value=(1.0, 0.0),
        ) as mock_record:
            a._kick_to_loco_flip_scan_worker(
                onnx_path="/fake/model.onnx",
                global_step=800,
                num_trials=32,
                kick_aim_info=None,
            )

        assert all(c.kwargs["kick_aim_enabled"] is False for c in mock_record.call_args_list)
        assert len(mock_record.call_args_list) == 2

    def test_multi_skill_rates_land_under_correct_per_skill_keys(self):
        a = _make_agent([0.5, 0.3])  # 2 skills

        def fake_record(**kwargs):
            if kwargs["skill_id"] == 0:
                return (0.8, 0.1)
            return (0.6, 0.2)

        with patch(
            "holosoma.record_mujoco_kick_to_loco_flip_scan.record_kick_to_loco_flip_scan",
            side_effect=fake_record,
        ):
            a._kick_to_loco_flip_scan_worker(
                onnx_path="/fake/model.onnx",
                global_step=900,
                num_trials=32,
                kick_aim_info=([True, True], 45.0),
            )

        items = _drain(a._kick_to_loco_flip_scan_result_queue)
        keys_and_values = {key: val for (_step, key, val) in items}
        assert keys_and_values["Kick_skills_0/sim2sim/kick_to_loco_random_flip_alive_rate"] == 0.8
        assert keys_and_values["Kick_skills_1/sim2sim/kick_to_loco_random_flip_alive_rate"] == 0.6

    def test_single_skill_mode_still_uses_kick_skills_0_prefix(self):
        """Empty _skill_motion_training_ratios (legacy single-skill mode) must still land under
        'Kick_skills_0/...', matching the established convention every sibling sim2sim mechanism
        follows (num_skills = max(len(...), 1))."""
        a = _make_agent([])  # legacy single-skill mode
        with patch(
            "holosoma.record_mujoco_kick_to_loco_flip_scan.record_kick_to_loco_flip_scan",
            return_value=(1.0, 0.0),
        ) as mock_record:
            a._kick_to_loco_flip_scan_worker(
                onnx_path="/fake/model.onnx",
                global_step=1000,
                num_trials=32,
                kick_aim_info=None,
            )

        assert len(mock_record.call_args_list) == 1
        items = _drain(a._kick_to_loco_flip_scan_result_queue)
        keys = {key for (_step, key, _val) in items}
        assert "Kick_skills_0/sim2sim/kick_to_loco_random_flip_alive_rate" in keys
