"""Unit tests for record_mujoco_loco_to_kick_handoff_scan.py's own subprocess-invocation and
SUMMARY-line parsing -- specifically the `return_transition_metrics`/`transition_metrics_out`
addition (2026-09-02). The pre-existing (fall_rate, hit_rate, pre_handoff_fail_rate) parsing has
no dedicated test file of its own (only mocked-out coverage one layer up, in
agents/fast_sac/tests/test_mujoco_loco_to_kick_handoff_scan.py and
sim2sim_eval's own tests) -- covered incidentally here since every test below exercises it too.

Mocks only `subprocess.run` -- acquire_global_lock/release_global_lock run for real against a
throwaway lock path (fast, local file lock, no reason to mock a correct implementation).
"""

from __future__ import annotations

import subprocess
from unittest.mock import patch

from holosoma.record_mujoco_loco_to_kick_handoff_scan import record_loco_to_kick_handoff_scan


def _fake_result(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")


_BASE_STDOUT = (
    "SUMMARY_PREHANDOFFFAIL 100 1/10 0.1000\n"
    "SUMMARY_LOCOTOKICKFALL 100 0/9 0.0000\n"
    "SUMMARY_HIT 100 7/9 0.7778\n"
)


class TestReturnTransitionMetricsOff:
    def test_default_behavior_unchanged_and_out_dict_untouched(self, tmp_path):
        out: dict = {}
        with patch("subprocess.run", return_value=_fake_result(_BASE_STDOUT)):
            result = record_loco_to_kick_handoff_scan(
                onnx_path="/fake.onnx", step_label="100", num_trials=10,
                lock_path=str(tmp_path / "lock"), transition_metrics_out=out,
            )
        assert result == (0.0, 0.7778, 0.1)
        assert out == {}  # never touched -- return_transition_metrics defaults False

    def test_argv_has_no_transition_flags_by_default(self, tmp_path):
        captured_argv = []

        def fake_run(argv, **kwargs):
            captured_argv.extend(argv)
            return _fake_result(_BASE_STDOUT)

        with patch("subprocess.run", side_effect=fake_run):
            record_loco_to_kick_handoff_scan(
                onnx_path="/fake.onnx", step_label="100", num_trials=10, lock_path=str(tmp_path / "lock"),
            )
        assert "--track-transition-metrics" not in captured_argv
        assert "--transition-window-steps" not in captured_argv


class TestReturnTransitionMetricsOn:
    _STDOUT_WITH_TRANSITION = _BASE_STDOUT + (
        "SUMMARY_TRACKING_ERROR_EARLY 100 0.1234\n"
        "SUMMARY_TRACKING_ERROR_LATE 100 0.0456\n"
        "SUMMARY_JERK_EARLY 100 0.9876\n"
        "SUMMARY_JERK_LATE 100 0.0123\n"
        "SUMMARY_DRIFT 100 0.0789\n"
    )

    def test_argv_carries_the_flag_and_window(self, tmp_path):
        captured_argv = []

        def fake_run(argv, **kwargs):
            captured_argv.extend(argv)
            return _fake_result(self._STDOUT_WITH_TRANSITION)

        with patch("subprocess.run", side_effect=fake_run):
            record_loco_to_kick_handoff_scan(
                onnx_path="/fake.onnx", step_label="100", num_trials=10, lock_path=str(tmp_path / "lock"),
                return_transition_metrics=True, transition_window_steps=42, transition_metrics_out={},
            )
        assert "--track-transition-metrics" in captured_argv
        i = captured_argv.index("--transition-window-steps")
        assert captured_argv[i + 1] == "42"

    def test_all_five_values_parsed_into_out_dict(self, tmp_path):
        out: dict = {}
        with patch("subprocess.run", return_value=_fake_result(self._STDOUT_WITH_TRANSITION)):
            fall_rate, hit_rate, pre_handoff_fail_rate = record_loco_to_kick_handoff_scan(
                onnx_path="/fake.onnx", step_label="100", num_trials=10, lock_path=str(tmp_path / "lock"),
                return_transition_metrics=True, transition_metrics_out=out,
            )
        # the original 3-tuple is completely unaffected by requesting transition metrics too
        assert (fall_rate, hit_rate, pre_handoff_fail_rate) == (0.0, 0.7778, 0.1)
        assert out == {
            "tracking_error_early": 0.1234,
            "tracking_error_late": 0.0456,
            "jerk_early": 0.9876,
            "jerk_late": 0.0123,
            "drift": 0.0789,
        }

    def test_na_values_parse_to_none(self, tmp_path):
        stdout = _BASE_STDOUT + (
            "SUMMARY_TRACKING_ERROR_EARLY 100 NA\n"
            "SUMMARY_TRACKING_ERROR_LATE 100 NA\n"
            "SUMMARY_JERK_EARLY 100 NA\n"
            "SUMMARY_JERK_LATE 100 NA\n"
            "SUMMARY_DRIFT 100 NA\n"
        )
        out: dict = {}
        with patch("subprocess.run", return_value=_fake_result(stdout)):
            record_loco_to_kick_handoff_scan(
                onnx_path="/fake.onnx", step_label="100", num_trials=10, lock_path=str(tmp_path / "lock"),
                return_transition_metrics=True, transition_metrics_out=out,
            )
        assert out == {k: None for k in ("tracking_error_early", "tracking_error_late", "jerk_early", "jerk_late", "drift")}

    def test_unparseable_line_warns_and_stays_none_not_a_crash(self, tmp_path):
        stdout = _BASE_STDOUT + "SUMMARY_DRIFT 100 garbage\n"
        out: dict = {}
        with patch("subprocess.run", return_value=_fake_result(stdout)):
            record_loco_to_kick_handoff_scan(
                onnx_path="/fake.onnx", step_label="100", num_trials=10, lock_path=str(tmp_path / "lock"),
                return_transition_metrics=True, transition_metrics_out=out,
            )
        assert out["drift"] is None

    def test_missing_lines_stay_none(self, tmp_path):
        """A checkpoint's worker predating this feature, or a subprocess that otherwise never
        printed these 5 lines -- absence must not raise, everything stays None."""
        out: dict = {}
        with patch("subprocess.run", return_value=_fake_result(_BASE_STDOUT)):
            record_loco_to_kick_handoff_scan(
                onnx_path="/fake.onnx", step_label="100", num_trials=10, lock_path=str(tmp_path / "lock"),
                return_transition_metrics=True, transition_metrics_out=out,
            )
        assert out == {k: None for k in ("tracking_error_early", "tracking_error_late", "jerk_early", "jerk_late", "drift")}

    def test_nonzero_exit_leaves_out_dict_fully_keyed_but_all_none(self, tmp_path):
        out: dict = {}
        with patch("subprocess.run", return_value=_fake_result("", returncode=1)):
            result = record_loco_to_kick_handoff_scan(
                onnx_path="/fake.onnx", step_label="100", num_trials=10, lock_path=str(tmp_path / "lock"),
                return_transition_metrics=True, transition_metrics_out=out,
            )
        assert result == (None, None, None)
        assert out == {k: None for k in ("tracking_error_early", "tracking_error_late", "jerk_early", "jerk_late", "drift")}

    def test_timeout_leaves_out_dict_fully_keyed_but_all_none(self, tmp_path):
        out: dict = {}
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=[], timeout=1.0)):
            result = record_loco_to_kick_handoff_scan(
                onnx_path="/fake.onnx", step_label="100", num_trials=10, lock_path=str(tmp_path / "lock"),
                return_transition_metrics=True, transition_metrics_out=out,
            )
        assert result == (None, None, None)
        assert out == {k: None for k in ("tracking_error_early", "tracking_error_late", "jerk_early", "jerk_late", "drift")}
