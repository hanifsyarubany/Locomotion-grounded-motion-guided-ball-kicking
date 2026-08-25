"""Unit tests for the 2026-08-23 direction-success-rate addition to FastSACAgent's MuJoCo
survival-scan worker: `record_survival_scan` now returns `(fall_rate, hit_rate,
direction_success_rate)`, and `_mujoco_survival_scan_worker` must thread `kick_aim_nominal_distance_m`
through to it and queue `direction_success_rate` under its own wandb key whenever it isn't None
(None means either kick_aim_enabled=False for that skill, or zero trials hit the ball -- both
valid "nothing to report" states, not errors).

Isolated via a bare instance (object.__new__), matching this project's existing bare-instance
convention for FastSACAgent unit tests (see test_mujoco_kick_rollout_per_skill.py) -- no real
onnx, MuJoCo, or wandb run needed.
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
    a._survival_scan_result_queue = queue.Queue()
    a.global_step = 12345
    return a


def _drain(q: queue.Queue) -> list[tuple]:
    items = []
    while True:
        try:
            items.append(q.get_nowait())
        except queue.Empty:
            return items


class TestMujocoSurvivalScanDirectionSuccessRate:
    def test_kick_aim_nominal_distance_m_reaches_record_survival_scan(self):
        a = _make_agent([0.5])
        with patch(
            "holosoma.record_mujoco_survival_scan.record_survival_scan", return_value=(0.1, 0.6, 0.75)
        ) as mock_record:
            a._mujoco_survival_scan_worker(
                onnx_path="/fake/model.onnx",
                global_step=500,
                num_trials=10,
                pos_rand_per_motion=[[0.1, 0.1]],
                kick_aim_enabled_per_motion=[True],
                kick_aim_theta_max_deg_per_motion=[15.0],
                kick_aim_theta_ref_deg=45.0,
                kick_aim_nominal_distance_m=5.0,
            )

        _, kwargs = mock_record.call_args
        assert kwargs["kick_aim_nominal_distance_m"] == 5.0

    def test_direction_success_rate_queued_under_its_own_key(self):
        a = _make_agent([0.5])
        with patch(
            "holosoma.record_mujoco_survival_scan.record_survival_scan", return_value=(0.1, 0.6, 0.75)
        ):
            a._mujoco_survival_scan_worker(
                onnx_path="/fake/model.onnx",
                global_step=500,
                num_trials=10,
                pos_rand_per_motion=[[0.0, 0.0]],
                kick_aim_enabled_per_motion=[True],
                kick_aim_theta_max_deg_per_motion=[15.0],
                kick_aim_theta_ref_deg=45.0,
                kick_aim_nominal_distance_m=5.0,
            )

        items = _drain(a._survival_scan_result_queue)
        keys_and_values = {key: val for (_step, key, val) in items}
        assert keys_and_values["Kick_skills_0/sim2sim/kick_fall_rate"] == 0.1
        assert keys_and_values["Kick_skills_0/sim2sim/kick_ball_hit_rate"] == 0.6
        assert keys_and_values["Kick_skills_0/sim2sim/kick_direction_success_rate"] == 0.75

    def test_direction_success_rate_none_is_simply_not_queued(self):
        """None means "no data" (kick_aim disabled for this skill, or zero hits) -- a valid,
        expected state, not an error. fall_rate/hit_rate still queue normally."""
        a = _make_agent([0.5])
        with patch(
            "holosoma.record_mujoco_survival_scan.record_survival_scan", return_value=(0.2, 0.0, None)
        ):
            a._mujoco_survival_scan_worker(
                onnx_path="/fake/model.onnx",
                global_step=600,
                num_trials=10,
                pos_rand_per_motion=[[0.0, 0.0]],
                kick_aim_enabled_per_motion=[True],
                kick_aim_theta_max_deg_per_motion=[15.0],
                kick_aim_theta_ref_deg=45.0,
                kick_aim_nominal_distance_m=5.0,
            )

        items = _drain(a._survival_scan_result_queue)
        keys = {key for (_step, key, _val) in items}
        assert "Kick_skills_0/sim2sim/kick_fall_rate" in keys
        assert "Kick_skills_0/sim2sim/kick_ball_hit_rate" in keys
        assert "Kick_skills_0/sim2sim/kick_direction_success_rate" not in keys

    def test_multi_skill_direction_rates_land_under_correct_per_skill_keys(self):
        a = _make_agent([0.5, 0.3])  # 2 skills

        def fake_record(**kwargs):
            # skill 0 aim-enabled with real data, skill 1 not aim-enabled at all
            if kwargs["skill_id"] == 0:
                return (0.1, 0.5, 0.4)
            return (0.3, 0.2, None)

        with patch(
            "holosoma.record_mujoco_survival_scan.record_survival_scan", side_effect=fake_record
        ):
            a._mujoco_survival_scan_worker(
                onnx_path="/fake/model.onnx",
                global_step=700,
                num_trials=10,
                pos_rand_per_motion=[[0.0, 0.0], [0.0, 0.0]],
                kick_aim_enabled_per_motion=[True, False],
                kick_aim_theta_max_deg_per_motion=[15.0, 15.0],
                kick_aim_theta_ref_deg=45.0,
                kick_aim_nominal_distance_m=5.0,
            )

        items = _drain(a._survival_scan_result_queue)
        keys_and_values = {key: val for (_step, key, val) in items}
        assert keys_and_values["Kick_skills_0/sim2sim/kick_direction_success_rate"] == 0.4
        assert "Kick_skills_1/sim2sim/kick_direction_success_rate" not in keys_and_values
