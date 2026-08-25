"""Unit tests for FastSACAgent's per-skill MuJoCo kick rollout looping:
_mujoco_kick_rollout_worker now loops sequentially over every configured motion skill (instead of
always just skill 0), and _drain_mujoco_kick_rollout_queue logs each one under its own wandb key.

Also covers the 2026-08-23 kick_aim_theta bugfix: record_kick_rollout used to always take the
kick_aim_info=None path, silently feeding every checkpoint (including every currently-trained
kick_aim_enabled=True one) the pre-azimuth-refactor raw world-frame target_pos_b transform instead
of the bounded command it actually trained on -- see _kick_aim_info_per_motion's own docstring.

Isolated via a bare instance (object.__new__, bypassing __init__'s real network/env construction)
with only the specific attributes these two methods touch, plus record_kick_rollout and wandb.log
mocked out -- no real onnx, MuJoCo, or wandb run needed, matching this project's existing
bare-instance test convention for this class of problem.
"""

import queue
from unittest.mock import MagicMock, patch

from holosoma.agents.fast_sac.fast_sac_agent import FastSACAgent


class _FakeUnwrappedEnv:
    def __init__(self, skill_motion_training_ratios):
        self._skill_motion_training_ratios = skill_motion_training_ratios


def _make_agent(skill_motion_training_ratios, log_dir="/tmp/holosoma_test_log_dir") -> FastSACAgent:
    a = object.__new__(FastSACAgent)
    a.unwrapped_env = _FakeUnwrappedEnv(skill_motion_training_ratios)
    a.log_dir = log_dir
    a._kick_rollout_video_queue = queue.Queue()
    a.global_step = 12345
    return a


class TestMujocoKickRolloutPerSkill:
    def test_legacy_no_skills_produces_exactly_one_video_with_original_naming(self):
        a = _make_agent([])
        with patch("holosoma.record_mujoco_kick_rollout.record_kick_rollout", return_value=True) as mock_record:
            a._mujoco_kick_rollout_worker(onnx_path="/fake/model.onnx", global_step=100, with_ball=True)

        assert mock_record.call_count == 1
        _, kwargs = mock_record.call_args
        assert kwargs["skill_id"] == 0
        assert kwargs["output_video_path"].endswith("model_0000100_mujoco_kick.mp4")

        queued = a._kick_rollout_video_queue.get_nowait()
        _, video_path, wandb_key = queued
        assert wandb_key == "mujoco_media/Training rollout - MuJoCo Kick"
        assert video_path.endswith("model_0000100_mujoco_kick.mp4")

    def test_n_skill_mode_produces_one_video_per_skill_with_skill_specific_naming(self):
        a = _make_agent([0.3, 0.2, 0.1])  # 3 skills
        with patch("holosoma.record_mujoco_kick_rollout.record_kick_rollout", return_value=True) as mock_record:
            a._mujoco_kick_rollout_worker(onnx_path="/fake/model.onnx", global_step=200, with_ball=True)

        assert mock_record.call_count == 3
        called_skill_ids = [call.kwargs["skill_id"] for call in mock_record.call_args_list]
        assert called_skill_ids == [0, 1, 2]

        queued_items = []
        while True:
            try:
                queued_items.append(a._kick_rollout_video_queue.get_nowait())
            except queue.Empty:
                break
        assert len(queued_items) == 3
        wandb_keys = sorted(item[2] for item in queued_items)
        assert wandb_keys == [
            "mujoco_media/Training rollout - MuJoCo Kick - Skill 1",
            "mujoco_media/Training rollout - MuJoCo Kick - Skill 2",
            "mujoco_media/Training rollout - MuJoCo Kick - Skill 3",
        ]
        video_paths = sorted(item[1] for item in queued_items)
        assert video_paths[0].endswith("model_0000200_mujoco_kick_skill1.mp4")
        assert video_paths[1].endswith("model_0000200_mujoco_kick_skill2.mp4")
        assert video_paths[2].endswith("model_0000200_mujoco_kick_skill3.mp4")

    def test_one_skills_rollout_failing_does_not_stop_the_others(self):
        a = _make_agent([0.3, 0.2])  # 2 skills

        def fake_record(**kwargs):
            return kwargs["skill_id"] != 0  # skill 0 "fails" (returns False), skill 1 succeeds

        with patch("holosoma.record_mujoco_kick_rollout.record_kick_rollout", side_effect=fake_record) as mock_record:
            a._mujoco_kick_rollout_worker(onnx_path="/fake/model.onnx", global_step=300, with_ball=True)

        assert mock_record.call_count == 2  # both attempted, despite skill 0 "failing"
        queued_items = []
        while True:
            try:
                queued_items.append(a._kick_rollout_video_queue.get_nowait())
            except queue.Empty:
                break
        assert len(queued_items) == 1  # only skill 1's succeeded and got queued
        assert queued_items[0][2] == "mujoco_media/Training rollout - MuJoCo Kick - Skill 2"

    def test_one_skills_rollout_raising_does_not_stop_the_others(self):
        a = _make_agent([0.3, 0.2])  # 2 skills

        def fake_record(**kwargs):
            if kwargs["skill_id"] == 0:
                raise RuntimeError("boom")
            return True

        with patch("holosoma.record_mujoco_kick_rollout.record_kick_rollout", side_effect=fake_record) as mock_record:
            a._mujoco_kick_rollout_worker(onnx_path="/fake/model.onnx", global_step=400, with_ball=True)

        assert mock_record.call_count == 2  # skill 1 still attempted after skill 0 raised
        queued_items = []
        while True:
            try:
                queued_items.append(a._kick_rollout_video_queue.get_nowait())
            except queue.Empty:
                break
        assert len(queued_items) == 1
        assert queued_items[0][2] == "mujoco_media/Training rollout - MuJoCo Kick - Skill 2"

    def test_drain_logs_each_queued_video_under_its_own_wandb_key(self):
        a = _make_agent([])
        a._kick_rollout_video_queue.put((100, "/tmp/a.mp4", "mujoco_media/Training rollout - MuJoCo Kick - Skill 1"))
        a._kick_rollout_video_queue.put((100, "/tmp/b.mp4", "mujoco_media/Training rollout - MuJoCo Kick - Skill 2"))

        fake_video = MagicMock()
        with patch("wandb.log") as mock_log, patch("wandb.Video", return_value=fake_video):
            a._drain_mujoco_kick_rollout_queue()

        assert mock_log.call_count == 2
        logged_keys = sorted(next(iter(call.args[0])) for call in mock_log.call_args_list)
        assert logged_keys == ["mujoco_media/Training rollout - MuJoCo Kick - Skill 1", "mujoco_media/Training rollout - MuJoCo Kick - Skill 2"]
        assert a._kick_rollout_video_queue.empty()


class _FakeMotionCommand:
    def __init__(self, has_ball, kick_aim_enabled_per_motion, kick_aim_theta_ref_deg=45.0):
        self.has_ball = has_ball
        self._kick_aim_enabled_per_motion = kick_aim_enabled_per_motion
        self.kick_aim_theta_ref_deg = kick_aim_theta_ref_deg

    @property
    def kick_aim_enabled_per_motion(self):
        # .detach().cpu().tolist() chain, same as the real torch tensor _kick_aim_info_per_motion
        # actually reads -- a plain object exposing the same three calls, no torch dependency.
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self._kick_aim_enabled_per_motion


class _FakeCommandManager:
    def __init__(self, motion_command):
        self._motion_command = motion_command

    def get_state(self, name):
        assert name == "motion_command"
        return self._motion_command


def _make_agent_with_command_manager(
    skill_motion_training_ratios, motion_command, log_dir="/tmp/holosoma_test_log_dir"
) -> FastSACAgent:
    a = _make_agent(skill_motion_training_ratios, log_dir=log_dir)
    a.unwrapped_env.command_manager = _FakeCommandManager(motion_command)
    return a


class TestKickAimInfoPerMotion:
    """_kick_aim_info_per_motion: the 2026-08-23 fix that threads kick_aim_enabled/
    kick_aim_theta_ref_deg through to record_kick_rollout so the MuJoCo rollout feeds the SAME
    bounded command a kick_aim_enabled checkpoint actually trained on, instead of silently falling
    back to the pre-refactor raw world-frame transform."""

    def test_no_ball_returns_none(self):
        a = _make_agent_with_command_manager([], _FakeMotionCommand(has_ball=False, kick_aim_enabled_per_motion=[]))
        assert a._kick_aim_info_per_motion() is None

    def test_no_motion_command_returns_none(self):
        a = _make_agent_with_command_manager([], None)
        assert a._kick_aim_info_per_motion() is None

    def test_returns_per_motion_flags_and_theta_ref(self):
        mc = _FakeMotionCommand(has_ball=True, kick_aim_enabled_per_motion=[True, False], kick_aim_theta_ref_deg=45.0)
        a = _make_agent_with_command_manager([0.5, 0.5], mc)
        result = a._kick_aim_info_per_motion()
        assert result == ([True, False], 45.0)


class TestKickAimEnabledThreadedIntoRecordKickRollout:
    """The actual bugfix: record_kick_rollout must receive THIS skill's own kick_aim_enabled flag
    (and the shared kick_aim_theta_ref_deg), not silently default to False/45.0 for a checkpoint
    that was really trained with kick_aim_enabled=True."""

    def test_kick_aim_enabled_per_skill_reaches_record_kick_rollout(self):
        a = _make_agent([0.5, 0.5])  # 2 skills
        kick_aim_info = ([True, False], 45.0)  # skill 0 aim-enabled, skill 1 not
        with patch("holosoma.record_mujoco_kick_rollout.record_kick_rollout", return_value=True) as mock_record:
            a._mujoco_kick_rollout_worker(
                onnx_path="/fake/model.onnx", global_step=500, with_ball=True, kick_aim_info=kick_aim_info
            )

        calls_by_skill = {call.kwargs["skill_id"]: call.kwargs for call in mock_record.call_args_list}
        assert calls_by_skill[0]["kick_aim_enabled"] is True
        assert calls_by_skill[0]["kick_aim_theta_ref_deg"] == 45.0
        assert calls_by_skill[1]["kick_aim_enabled"] is False

    def test_kick_aim_info_none_preserves_prior_behavior(self):
        """The bug this fixes: before this feature existed (or for a run with no ball),
        kick_aim_info=None must produce the exact old behavior -- kick_aim_enabled=False for
        every skill, not a crash or a silently-wrong True."""
        a = _make_agent([0.5])
        with patch("holosoma.record_mujoco_kick_rollout.record_kick_rollout", return_value=True) as mock_record:
            a._mujoco_kick_rollout_worker(
                onnx_path="/fake/model.onnx", global_step=600, with_ball=True, kick_aim_info=None
            )

        _, kwargs = mock_record.call_args
        assert kwargs["kick_aim_enabled"] is False
        assert kwargs["kick_aim_theta_ref_deg"] == 45.0

    def test_out_of_range_skill_idx_falls_back_to_row_zero(self):
        """Mirrors the existing pos_rand_per_motion row-clamping convention elsewhere in this
        class: an ONNX with fewer per-motion rows than skills looped here must not IndexError."""
        a = _make_agent([0.5, 0.5, 0.5])  # 3 skills
        kick_aim_info = ([True], 45.0)  # only 1 row
        with patch("holosoma.record_mujoco_kick_rollout.record_kick_rollout", return_value=True) as mock_record:
            a._mujoco_kick_rollout_worker(
                onnx_path="/fake/model.onnx", global_step=700, with_ball=True, kick_aim_info=kick_aim_info
            )

        for call in mock_record.call_args_list:
            assert call.kwargs["kick_aim_enabled"] is True  # every skill falls back to row 0 == True
