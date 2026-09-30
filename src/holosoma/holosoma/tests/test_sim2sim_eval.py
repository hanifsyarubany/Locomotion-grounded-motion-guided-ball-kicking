"""Unit tests for sim2sim_eval.py -- config parsing/validation, ONNX-path-derived helpers, and the
per-eval-type dispatch functions' own kwarg threading. No real onnx/MuJoCo/wandb/robojudo needed:
the dispatch tests patch the underlying `record_*` wrapper functions (same pattern
test_mujoco_kick_to_loco_flip_scan.py / test_mujoco_loco_to_kick_handoff_scan.py already
established for FastSACAgent's own equivalent workers), and discover_num_skills is exercised
against a real checkpoint separately, by hand, not as part of this automated suite (it needs a
real .onnx file and onnxruntime).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from holosoma.sim2sim_eval import (
    Sim2SimEvalConfig,
    _compute_strike_window_ticks,
    _pool_flip_rates,
    default_target_name,
    infer_step_from_onnx_path,
    load_eval_config,
    run_eval,
)


def _write_config(tmp_path: Path, content: dict) -> str:
    path = tmp_path / "eval.yaml"
    path.write_text(yaml.dump(content))
    return str(path)


class TestInferStepFromOnnxPath:
    def test_standard_checkpoint_name(self):
        assert infer_step_from_onnx_path("/a/b/model_0180000.onnx") == 180000

    def test_no_leading_zeros_still_matches(self):
        assert infer_step_from_onnx_path("model_5.onnx") == 5

    def test_non_matching_name_returns_none(self):
        assert infer_step_from_onnx_path("/a/b/my_custom_export.onnx") is None

    def test_only_matches_at_the_end(self):
        """A path containing 'model_0180000' somewhere in a DIRECTORY name, but not as the actual
        filename, must not match -- this reads the checkpoint's OWN training step, not any
        coincidental digit run earlier in the path."""
        assert infer_step_from_onnx_path("/model_0180000-stageC/checkpoint_final.onnx") is None


class TestDefaultTargetName:
    def test_combines_run_dir_and_stem(self):
        p = "/logs/UnifiedBallKickingEnhanced/20260831_012859-stageC/model_0440000.onnx"
        assert default_target_name(p) == "20260831_012859-stageC/model_0440000"

    def test_relative_path_still_works(self):
        p = "logs/UnifiedBallKickingEnhanced/some_run/model_0180000.onnx"
        assert default_target_name(p) == "some_run/model_0180000"


class TestLoadEvalConfig:
    def test_minimal_valid_config_parses(self, tmp_path):
        path = _write_config(
            tmp_path,
            {
                "onnx_targets": [{"path": "/a/model_0001000.onnx"}],
                "evaluations": [{"type": "kick_survival"}],
            },
        )
        cfg = load_eval_config(path)
        assert cfg.onnx_targets[0].path == "/a/model_0001000.onnx"
        assert cfg.onnx_targets[0].kick_aim_enabled is True  # default
        assert cfg.onnx_targets[0].skill_ids is None  # default -> auto-discover
        assert cfg.evaluations[0].type == "kick_survival"
        assert cfg.wandb_project == "UnifiedBallKickingEnhanced"  # default, same as training

    def test_missing_onnx_targets_raises(self, tmp_path):
        path = _write_config(tmp_path, {"evaluations": [{"type": "kick_survival"}]})
        with pytest.raises(ValueError, match="onnx_targets"):
            load_eval_config(path)

    def test_missing_evaluations_raises(self, tmp_path):
        path = _write_config(tmp_path, {"onnx_targets": [{"path": "/a.onnx"}]})
        with pytest.raises(ValueError, match="evaluations"):
            load_eval_config(path)

    def test_target_missing_path_raises(self, tmp_path):
        path = _write_config(
            tmp_path,
            {"onnx_targets": [{"name": "no path here"}], "evaluations": [{"type": "kick_survival"}]},
        )
        with pytest.raises(ValueError, match="path"):
            load_eval_config(path)

    def test_unknown_eval_type_raises(self, tmp_path):
        path = _write_config(
            tmp_path,
            {"onnx_targets": [{"path": "/a.onnx"}], "evaluations": [{"type": "not_a_real_scan"}]},
        )
        with pytest.raises(ValueError, match="not_a_real_scan"):
            load_eval_config(path)

    def test_negative_num_trials_raises(self, tmp_path):
        path = _write_config(
            tmp_path,
            {
                "onnx_targets": [{"path": "/a.onnx"}],
                "evaluations": [{"type": "kick_survival", "num_trials": -1}],
            },
        )
        with pytest.raises(ValueError, match="num_trials"):
            load_eval_config(path)

    def test_skill_ids_wrong_type_raises(self, tmp_path):
        path = _write_config(
            tmp_path,
            {
                "onnx_targets": [{"path": "/a.onnx", "skill_ids": "all"}],
                "evaluations": [{"type": "kick_survival"}],
            },
        )
        with pytest.raises(ValueError, match="skill_ids"):
            load_eval_config(path)

    def test_extra_eval_params_pass_through_untouched(self, tmp_path):
        """Non-'type' keys under an evaluation block flow through verbatim into `.params` -- the
        dispatch functions own their own defaults/validation, so this loader deliberately doesn't
        duplicate a second copy of every scan-specific knob's schema."""
        path = _write_config(
            tmp_path,
            {
                "onnx_targets": [{"path": "/a.onnx"}],
                "evaluations": [
                    {"type": "kick_to_loco_flip", "num_trials": 16, "flip_delay_min_steps": 100}
                ],
            },
        )
        cfg = load_eval_config(path)
        assert cfg.evaluations[0].params == {"num_trials": 16, "flip_delay_min_steps": 100}

    def test_iterations_defaults_to_one(self, tmp_path):
        path = _write_config(
            tmp_path,
            {"onnx_targets": [{"path": "/a.onnx"}], "evaluations": [{"type": "kick_survival"}]},
        )
        assert load_eval_config(path).iterations == 1

    def test_iterations_explicit_value_parses(self, tmp_path):
        path = _write_config(
            tmp_path,
            {
                "onnx_targets": [{"path": "/a.onnx"}],
                "evaluations": [{"type": "kick_survival"}],
                "iterations": 100,
            },
        )
        assert load_eval_config(path).iterations == 100

    def test_non_positive_iterations_raises(self, tmp_path):
        path = _write_config(
            tmp_path,
            {
                "onnx_targets": [{"path": "/a.onnx"}],
                "evaluations": [{"type": "kick_survival"}],
                "iterations": 0,
            },
        )
        with pytest.raises(ValueError, match="iterations"):
            load_eval_config(path)

    def test_max_concurrent_scans_defaults_to_one(self, tmp_path):
        path = _write_config(
            tmp_path,
            {"onnx_targets": [{"path": "/a.onnx"}], "evaluations": [{"type": "kick_survival"}]},
        )
        assert load_eval_config(path).max_concurrent_scans == 1

    def test_max_concurrent_scans_explicit_value_parses(self, tmp_path):
        path = _write_config(
            tmp_path,
            {
                "onnx_targets": [{"path": "/a.onnx"}],
                "evaluations": [{"type": "kick_survival"}],
                "max_concurrent_scans": 8,
            },
        )
        assert load_eval_config(path).max_concurrent_scans == 8

    def test_non_positive_max_concurrent_scans_raises(self, tmp_path):
        path = _write_config(
            tmp_path,
            {
                "onnx_targets": [{"path": "/a.onnx"}],
                "evaluations": [{"type": "kick_survival"}],
                "max_concurrent_scans": 0,
            },
        )
        with pytest.raises(ValueError, match="max_concurrent_scans"):
            load_eval_config(path)

    def test_explicit_wandb_block_overrides_defaults(self, tmp_path):
        path = _write_config(
            tmp_path,
            {
                "wandb": {"project": "my-other-project", "entity": "me", "group": "custom-group", "tags": ["x"]},
                "onnx_targets": [{"path": "/a.onnx"}],
                "evaluations": [{"type": "kick_survival"}],
            },
        )
        cfg = load_eval_config(path)
        assert cfg.wandb_project == "my-other-project"
        assert cfg.wandb_entity == "me"
        assert cfg.wandb_group == "custom-group"
        assert cfg.wandb_tags == ["x"]


class TestRunEvalDispatch:
    """Exercises run_eval's own orchestration (target/eval/skill loop, key naming, step
    resolution, wandb calls) with every `record_*` wrapper mocked -- no real onnx/MuJoCo/robojudo,
    matching test_mujoco_kick_to_loco_flip_scan.py's own patch-the-wrapper-function convention."""

    def _cfg(self, **overrides) -> Sim2SimEvalConfig:
        from holosoma.sim2sim_eval import EvalSpec, OnnxTarget

        defaults = dict(
            onnx_targets=[OnnxTarget(path="/fake/model_0100000.onnx", name="fake-target", skill_ids=[0])],
            evaluations=[EvalSpec(type="kick_survival", params={"num_trials": 4})],
        )
        defaults.update(overrides)
        return Sim2SimEvalConfig(**defaults)

    def test_logs_under_the_expected_key_shape_with_inferred_step(self):
        cfg = self._cfg()
        logged = {}

        class _FakeWandb:
            def init(self, **kwargs):
                pass

            def log(self, d, step=None):
                logged["dict"] = d
                logged["step"] = step

            def finish(self):
                pass

        with patch("holosoma.sim2sim_eval.record_survival_scan", return_value=(0.1, 0.6, None)), \
             patch("os.path.exists", return_value=True), \
             patch.dict("sys.modules", {"wandb": _FakeWandb()}):
            count = run_eval(cfg)

        assert count == 2  # fall_rate + hit_rate; direction_rate is None (kick_aim disabled), excluded
        assert logged["step"] == 100000  # inferred from "model_0100000.onnx"
        assert logged["dict"] == {
            "fake-target/0/kick/fall_rate": 0.1,
            "fake-target/0/kick/hit_rate": 0.6,
        }

    def test_kick_survival_extra_metrics_are_always_requested_and_merged(self):
        """2026-09-05 patch: success_sigma_m/ball_speed/shot_error are always requested via
        extra_metrics_out (no config flag needed -- the underlying scan says these cost nothing
        extra), and whatever the wrapper writes into that dict (mutated in place, mirroring its
        real output-parameter contract) is merged into the logged dict under kick/."""
        cfg = self._cfg()
        captured_kwargs = {}

        def fake_survival(**kwargs):
            captured_kwargs.update(kwargs)
            kwargs["extra_metrics_out"].update({
                "ball_speed_mean": 3.5, "ball_speed_std": 0.3, "ball_speed_n": 16, "ball_speed_max": 5.1,
                "shot_error_mean": None, "shot_error_std": None, "shot_error_n": 0,
                "success_rate_0.5": 0.25, "success_rate_1": 0.5,
            })
            return (0.0, 1.0, None)

        logged_dicts = []

        class _FakeWandb:
            def init(self, **kwargs):
                pass

            def log(self, d, step=None):
                logged_dicts.append(d)

            def finish(self):
                pass

        with patch("holosoma.sim2sim_eval.record_survival_scan", side_effect=fake_survival), \
             patch("os.path.exists", return_value=True), \
             patch.dict("sys.modules", {"wandb": _FakeWandb()}):
            run_eval(cfg)

        assert "extra_metrics_out" in captured_kwargs
        assert captured_kwargs["success_sigma_m"] is None  # not configured -> wrapper's own default
        # shot_error_mean/_std are None (zero hit trials, per this fixture) and dropped -- same
        # "never log a None" rule as direction_success_rate already gets. shot_error_n=0 is NOT
        # None -- it's a real, meaningful value ("zero hit trials", not "undefined") -- so it IS
        # logged, same as ball_speed_n.
        assert logged_dicts == [{
            "fake-target/0/kick/fall_rate": 0.0,
            "fake-target/0/kick/hit_rate": 1.0,
            "fake-target/0/kick/ball_speed_mean": 3.5,
            "fake-target/0/kick/ball_speed_std": 0.3,
            "fake-target/0/kick/ball_speed_n": 16,
            "fake-target/0/kick/ball_speed_max": 5.1,
            "fake-target/0/kick/shot_error_n": 0,
            "fake-target/0/kick/success_rate_0.5": 0.25,
            "fake-target/0/kick/success_rate_1": 0.5,
        }]

    def test_kick_survival_success_sigma_m_forwarded_when_configured(self):
        from holosoma.sim2sim_eval import EvalSpec

        cfg = self._cfg(
            evaluations=[EvalSpec(type="kick_survival", params={"num_trials": 4, "success_sigma_m": [0.3, 2.0]})]
        )
        captured_kwargs = {}

        def fake_survival(**kwargs):
            captured_kwargs.update(kwargs)
            return (0.0, 1.0, None)

        class _FakeWandb:
            def init(self, **kwargs):
                pass

            def log(self, d, step=None):
                pass

            def finish(self):
                pass

        with patch("holosoma.sim2sim_eval.record_survival_scan", side_effect=fake_survival), \
             patch("os.path.exists", return_value=True), \
             patch.dict("sys.modules", {"wandb": _FakeWandb()}):
            run_eval(cfg)

        assert captured_kwargs["success_sigma_m"] == [0.3, 2.0]

    def test_none_values_are_never_logged(self):
        cfg = self._cfg()
        logged_dicts = []

        class _FakeWandb:
            def init(self, **kwargs):
                pass

            def log(self, d, step=None):
                logged_dicts.append(d)

            def finish(self):
                pass

        with patch("holosoma.sim2sim_eval.record_survival_scan", return_value=(None, None, None)), \
             patch("os.path.exists", return_value=True), \
             patch.dict("sys.modules", {"wandb": _FakeWandb()}):
            count = run_eval(cfg)

        assert count == 0
        assert logged_dicts == []  # a fully-None result must not produce an empty wandb.log call

    def test_nonexistent_onnx_path_is_skipped_not_a_crash(self):
        cfg = self._cfg()

        class _FakeWandb:
            def init(self, **kwargs):
                pass

            def log(self, d, step=None):
                raise AssertionError("must not be called -- the target was skipped")

            def finish(self):
                pass

        with patch("os.path.exists", return_value=False), patch.dict("sys.modules", {"wandb": _FakeWandb()}):
            count = run_eval(cfg)  # must not raise
        assert count == 0

    def test_dispatch_crash_is_caught_and_other_work_continues(self):
        from holosoma.sim2sim_eval import EvalSpec, OnnxTarget

        cfg = self._cfg(
            onnx_targets=[OnnxTarget(path="/fake/model_0100000.onnx", name="t", skill_ids=[0, 1])],
        )
        calls = []

        def fake_survival(**kwargs):
            calls.append(kwargs["skill_id"])
            if kwargs["skill_id"] == 0:
                raise RuntimeError("boom")
            return (0.2, 0.5, None)

        logged_dicts = []

        class _FakeWandb:
            def init(self, **kwargs):
                pass

            def log(self, d, step=None):
                logged_dicts.append(d)

            def finish(self):
                pass

        with patch("holosoma.sim2sim_eval.record_survival_scan", side_effect=fake_survival), \
             patch("os.path.exists", return_value=True), \
             patch.dict("sys.modules", {"wandb": _FakeWandb()}):
            count = run_eval(cfg)

        assert calls == [0, 1]  # skill 1 still ran despite skill 0 crashing
        assert count == 2
        assert logged_dicts == [{"t/1/kick/fall_rate": 0.2, "t/1/kick/hit_rate": 0.5}]

    def test_loco_to_kick_handoff_does_not_forward_kick_aim_enabled(self):
        """That scan requires kick_aim_enabled unconditionally (see its own module docstring) --
        the dispatch adapter must not pass a kick_aim_enabled kwarg the wrapper doesn't accept."""
        from holosoma.sim2sim_eval import EvalSpec, OnnxTarget

        cfg = self._cfg(
            onnx_targets=[OnnxTarget(path="/fake/model_0100000.onnx", name="t", kick_aim_enabled=True, skill_ids=[0])],
            evaluations=[EvalSpec(type="loco_to_kick_handoff", params={"num_trials": 4})],
        )
        captured_kwargs = {}

        def fake_handoff(**kwargs):
            captured_kwargs.update(kwargs)
            return (0.0, 0.0, 0.0)

        class _FakeWandb:
            def init(self, **kwargs):
                pass

            def log(self, d, step=None):
                pass

            def finish(self):
                pass

        with patch("holosoma.sim2sim_eval.record_loco_to_kick_handoff_scan", side_effect=fake_handoff), \
             patch("os.path.exists", return_value=True), \
             patch.dict("sys.modules", {"wandb": _FakeWandb()}):
            run_eval(cfg)

        assert "kick_aim_enabled" not in captured_kwargs

    def test_track_transition_metrics_forwards_flags_and_merges_output_dict(self):
        """track_transition_metrics: true -- return_transition_metrics/transition_window_steps get
        forwarded, and whatever the wrapper writes into its transition_metrics_out dict (mutated
        in place, mirroring the real wrapper's own output-parameter contract) lands in the final
        metric dict alongside fall_rate/hit_rate/pre_handoff_fail_rate."""
        from holosoma.sim2sim_eval import EvalSpec, OnnxTarget

        cfg = self._cfg(
            onnx_targets=[OnnxTarget(path="/fake/model_0100000.onnx", name="t", kick_aim_enabled=True, skill_ids=[0])],
            evaluations=[
                EvalSpec(
                    type="loco_to_kick_handoff",
                    params={"num_trials": 4, "track_transition_metrics": True, "transition_window_steps": 42},
                )
            ],
        )
        captured_kwargs = {}

        def fake_handoff(**kwargs):
            captured_kwargs.update(kwargs)
            kwargs["transition_metrics_out"].update(
                {"tracking_error_early": 0.1, "tracking_error_late": 0.05, "jerk_early": 0.9, "jerk_late": 0.02, "drift": 0.03}
            )
            return (0.0, 0.8, 0.0)

        logged_dicts = []

        class _FakeWandb:
            def init(self, **kwargs):
                pass

            def log(self, d, step=None):
                logged_dicts.append(d)

            def finish(self):
                pass

        with patch("holosoma.sim2sim_eval.record_loco_to_kick_handoff_scan", side_effect=fake_handoff), \
             patch("os.path.exists", return_value=True), \
             patch.dict("sys.modules", {"wandb": _FakeWandb()}):
            run_eval(cfg)

        assert captured_kwargs["return_transition_metrics"] is True
        assert captured_kwargs["transition_window_steps"] == 42
        assert logged_dicts == [{
            "t/0/loco_to_kick/fall_rate": 0.0,
            "t/0/loco_to_kick/hit_rate": 0.8,
            "t/0/loco_to_kick/pre_handoff_fail_rate": 0.0,
            "t/0/loco_to_kick/tracking_error_early": 0.1,
            "t/0/loco_to_kick/tracking_error_late": 0.05,
            "t/0/loco_to_kick/jerk_early": 0.9,
            "t/0/loco_to_kick/jerk_late": 0.02,
            "t/0/loco_to_kick/drift": 0.03,
        }]

    def test_track_transition_metrics_off_by_default_no_extra_keys(self):
        from holosoma.sim2sim_eval import EvalSpec, OnnxTarget

        cfg = self._cfg(
            onnx_targets=[OnnxTarget(path="/fake/model_0100000.onnx", name="t", kick_aim_enabled=True, skill_ids=[0])],
            evaluations=[EvalSpec(type="loco_to_kick_handoff", params={"num_trials": 4})],
        )
        captured_kwargs = {}

        def fake_handoff(**kwargs):
            captured_kwargs.update(kwargs)
            return (0.0, 0.8, 0.0)

        logged_dicts = []

        class _FakeWandb:
            def init(self, **kwargs):
                pass

            def log(self, d, step=None):
                logged_dicts.append(d)

            def finish(self):
                pass

        with patch("holosoma.sim2sim_eval.record_loco_to_kick_handoff_scan", side_effect=fake_handoff), \
             patch("os.path.exists", return_value=True), \
             patch.dict("sys.modules", {"wandb": _FakeWandb()}):
            run_eval(cfg)

        assert captured_kwargs["return_transition_metrics"] is False
        assert logged_dicts == [{
            "t/0/loco_to_kick/fall_rate": 0.0,
            "t/0/loco_to_kick/hit_rate": 0.8,
            "t/0/loco_to_kick/pre_handoff_fail_rate": 0.0,
        }]


class TestRunEvalIterations:
    """`iterations` (default 1, unchanged path) repeats the whole evaluations list against the
    SAME fixed checkpoint(s), each repeat re-seeded and logged at step=<iteration index> so wandb
    draws a line -- see run_eval's own ITERATIONS docstring section."""

    def _cfg(self, **overrides):
        from holosoma.sim2sim_eval import EvalSpec, OnnxTarget

        defaults = dict(
            onnx_targets=[OnnxTarget(path="/fake/model_0100000.onnx", name="fake-target", skill_ids=[0])],
            evaluations=[EvalSpec(type="kick_survival", params={"num_trials": 4})],
            iterations=1,
        )
        defaults.update(overrides)
        return Sim2SimEvalConfig(**defaults)

    @staticmethod
    def _fake_wandb(logged_dicts, logged_steps):
        class _FakeWandb:
            def init(self, **kwargs):
                pass

            def log(self, d, step=None):
                logged_dicts.append(d)
                logged_steps.append(step)

            def finish(self):
                pass

        return _FakeWandb()

    def test_default_iterations_one_keeps_old_step_semantics(self):
        """No behavior change for the pre-existing single-shot path: step is still the inferred
        onnx training step, not the iteration index."""
        cfg = self._cfg()
        logged_dicts, logged_steps = [], []

        with patch("holosoma.sim2sim_eval.record_survival_scan", return_value=(0.1, 0.6, None)), \
             patch("os.path.exists", return_value=True), \
             patch.dict("sys.modules", {"wandb": self._fake_wandb(logged_dicts, logged_steps)}):
            run_eval(cfg)

        assert logged_steps == [100000]

    def test_iterations_runs_the_dispatch_once_per_iteration_with_increasing_step(self):
        cfg = self._cfg(iterations=3)
        logged_dicts, logged_steps = [], []
        seeds_seen = []

        def fake_survival(**kwargs):
            seeds_seen.append(kwargs["seed"])
            return (0.1, 0.6, None)

        with patch("holosoma.sim2sim_eval.record_survival_scan", side_effect=fake_survival), \
             patch("os.path.exists", return_value=True), \
             patch.dict("sys.modules", {"wandb": self._fake_wandb(logged_dicts, logged_steps)}):
            count = run_eval(cfg)

        assert logged_steps == [0, 1, 2]  # iteration index, NOT the checkpoint's training step
        assert seeds_seen == [0, 1, 2]  # base seed (default 0) + iteration -- a fresh batch each time
        assert count == 3 * 2  # 3 iterations x (fall_rate, hit_rate)
        # Same metric key at every step -- this is what makes wandb draw a single progressing line.
        assert {tuple(d.keys()) for d in logged_dicts} == {
            ("fake-target/0/kick/fall_rate", "fake-target/0/kick/hit_rate")
        }

    def test_explicit_base_seed_still_offsets_per_iteration(self):
        from holosoma.sim2sim_eval import EvalSpec, OnnxTarget

        cfg = self._cfg(
            onnx_targets=[OnnxTarget(path="/fake/model_0100000.onnx", name="t", skill_ids=[0])],
            evaluations=[EvalSpec(type="kick_survival", params={"num_trials": 4, "seed": 7})],
            iterations=2,
        )
        seeds_seen = []

        def fake_survival(**kwargs):
            seeds_seen.append(kwargs["seed"])
            return (0.1, 0.6, None)

        with patch("holosoma.sim2sim_eval.record_survival_scan", side_effect=fake_survival), \
             patch("os.path.exists", return_value=True), \
             patch.dict("sys.modules", {"wandb": self._fake_wandb([], [])}):
            run_eval(cfg)

        assert seeds_seen == [7, 8]

    def test_skill_discovery_and_existence_check_happen_once_per_target_not_per_iteration(self):
        """Both are properties of the checkpoint file itself, not of any one trial batch -- an
        expensive onnxruntime session load (or a filesystem stat) shouldn't repeat 100 times just
        because iterations=100."""
        from holosoma.sim2sim_eval import EvalSpec, OnnxTarget

        cfg = self._cfg(
            onnx_targets=[OnnxTarget(path="/fake/model_0100000.onnx", name="t", skill_ids=None)],
            evaluations=[EvalSpec(type="kick_survival", params={"num_trials": 4})],
            iterations=5,
        )

        with patch("holosoma.sim2sim_eval.record_survival_scan", return_value=(0.1, 0.6, None)), \
             patch("holosoma.sim2sim_eval.discover_num_skills", return_value=1) as mock_discover, \
             patch("os.path.exists", return_value=True) as mock_exists, \
             patch.dict("sys.modules", {"wandb": self._fake_wandb([], [])}):
            run_eval(cfg)

        assert mock_discover.call_count == 1
        assert mock_exists.call_count == 1

    def test_multiple_targets_interleave_turn_by_turn_within_each_iteration(self):
        """iteration is the OUTER loop, target the INNER one: turn 1 = target A, turn 2 = target B,
        THEN iteration 1 (not: every iteration of A, then every iteration of B) -- otherwise
        target B's wandb line stays flat/absent until target A's entire `iterations` count
        finishes, which is exactly what motivated this ordering (see run_eval's own TARGET ORDER
        docstring note)."""
        from holosoma.sim2sim_eval import EvalSpec, OnnxTarget

        cfg = self._cfg(
            onnx_targets=[
                OnnxTarget(path="/fake/skillA/model_0100000.onnx", name="skillA", skill_ids=[0]),
                OnnxTarget(path="/fake/skillB/model_0200000.onnx", name="skillB", skill_ids=[0]),
            ],
            evaluations=[EvalSpec(type="kick_survival", params={"num_trials": 4})],
            iterations=3,
        )
        turns = []  # (target_name, iteration_step) in call order

        def fake_survival(**kwargs):
            turns.append(kwargs["step_label"])
            return (0.1, 0.6, None)

        with patch("holosoma.sim2sim_eval.record_survival_scan", side_effect=fake_survival), \
             patch("os.path.exists", return_value=True), \
             patch.dict("sys.modules", {"wandb": self._fake_wandb([], [])}):
            run_eval(cfg)

        # step_label encodes base_step, which differs per target (100000 vs 200000) -- so the call
        # sequence below doubles as a target-identity trace: A, B, A, B, A, B, never A, A, ..., B, B.
        assert turns == [
            "100000/iter0", "200000/iter0",
            "100000/iter1", "200000/iter1",
            "100000/iter2", "200000/iter2",
        ]


class TestComputeStrikeWindowTicks:
    """Pure arithmetic: strike_start_frame/stand_start_frame (clip-local) -> strike_start_tick/
    stand_start_tick (ticks since [TRIGGER_KICK], the SAME unit kick_to_loco_flip's own
    flip_delay_min/max_steps use). See _compute_strike_window_ticks's own docstring for the full
    derivation. The skill011/skill015 fixtures below are the REAL values read from those two
    checkpoints' own experiment_config on 2026-09-02 -- regression-pinned, not invented."""

    def _sim(self, decimation=4, fps=200):
        return {"control_decimation": decimation, "fps": fps}

    def test_skill011_real_checkpoint_values(self):
        # experiment_config.command.setup_terms.motion_command.params.motion_config, verbatim.
        mc = {
            "motion_strike_start_frame": [170],
            "motion_stand_start_frame": [215],
            "enable_default_pose_prepend": True,
            "motion_prepend_duration_s": [],
            "default_pose_prepend_duration_s": 1.0,
        }
        assert _compute_strike_window_ticks(mc, self._sim(), skill_id=0) == (220, 265)

    def test_skill015_real_checkpoint_values(self):
        mc = {
            "motion_strike_start_frame": [70],
            "motion_stand_start_frame": [116],
            "enable_default_pose_prepend": True,
            "motion_prepend_duration_s": [],
            "default_pose_prepend_duration_s": 1.0,
        }
        assert _compute_strike_window_ticks(mc, self._sim(), skill_id=0) == (120, 166)

    def test_prepend_disabled_no_offset(self):
        mc = {
            "motion_strike_start_frame": [170],
            "motion_stand_start_frame": [215],
            "enable_default_pose_prepend": False,
            "default_pose_prepend_duration_s": 1.0,  # present but irrelevant -- prepend is off
        }
        assert _compute_strike_window_ticks(mc, self._sim(), skill_id=0) == (170, 215)

    def test_per_motion_prepend_duration_overrides_scalar_default(self):
        mc = {
            "motion_strike_start_frame": [0, 170],
            "motion_stand_start_frame": [0, 215],
            "enable_default_pose_prepend": True,
            "motion_prepend_duration_s": [0.0, 0.5],  # non-empty -- overrides the scalar for BOTH
            "default_pose_prepend_duration_s": 1.0,  # would give 50, must be ignored here
        }
        # skill 1: duration 0.5s / dt 0.02s = 25 steps (> 1, so it counts)
        assert _compute_strike_window_ticks(mc, self._sim(), skill_id=1) == (195, 240)

    def test_too_short_prepend_duration_is_skipped_same_as_disabled(self):
        # round(0.02 / 0.02) == 1 -- not > 1, mirrors _maybe_add_default_pose_transition's own
        # "too short for dt" skip exactly.
        mc = {
            "motion_strike_start_frame": [170],
            "motion_stand_start_frame": [215],
            "enable_default_pose_prepend": True,
            "motion_prepend_duration_s": [0.02],
            "default_pose_prepend_duration_s": 1.0,
        }
        assert _compute_strike_window_ticks(mc, self._sim(), skill_id=0) == (170, 215)

    def test_legacy_mode_missing_boundaries_returns_none(self):
        assert _compute_strike_window_ticks({}, self._sim(), skill_id=0) is None

    def test_skill_id_out_of_range_returns_none(self):
        mc = {"motion_strike_start_frame": [170], "motion_stand_start_frame": [215]}
        assert _compute_strike_window_ticks(mc, self._sim(), skill_id=1) is None


class TestPoolFlipRates:
    """Pure arithmetic: combining N independent (num_trials, alive_rate, pre_flip_fail_rate)
    sub-results into one statistically correct pooled rate via reconstructed integer counts. See
    _pool_flip_rates's own docstring for why this isn't a naive average of the two rates."""

    def test_pools_two_equal_sized_subcalls(self):
        # pre: n=2, pre_flip_fail_rate=0.0 (0/2), alive_rate=1.0 (2/2 reached, both alive)
        # post: n=2, pre_flip_fail_rate=0.0 (0/2), alive_rate=0.0 (2/2 reached, both dead)
        # pooled: pre_flip_fail = 0/4 = 0.0; alive = (2+0)/(2+2) = 0.5
        alive_rate, pre_flip_fail_rate = _pool_flip_rates([(2, 1.0, 0.0), (2, 0.0, 0.0)])
        assert pre_flip_fail_rate == 0.0
        assert alive_rate == 0.5

    def test_pools_unequal_sized_subcalls_without_misweighting(self):
        # A naive (rate_a + rate_b) / 2 would give (1.0 + 0.0) / 2 = 0.5 here too by coincidence --
        # use trial counts where misweighting would visibly differ from the correct pooled value.
        # pre: n=1, alive_rate=1.0 (1/1). post: n=3, alive_rate=0.0 (0/3).
        # correct pooled: (1 + 0) / (1 + 3) = 0.25, NOT (1.0 + 0.0)/2 = 0.5.
        alive_rate, pre_flip_fail_rate = _pool_flip_rates([(1, 1.0, 0.0), (3, 0.0, 0.0)])
        assert pre_flip_fail_rate == 0.0
        assert alive_rate == 0.25

    def test_accounts_for_pre_flip_fail_in_the_alive_denominator(self):
        # n=4, pre_flip_fail_rate=0.5 -> 2 pre-flip-fails -> num_reached=2, alive_rate=0.5 -> 1 alive.
        alive_rate, pre_flip_fail_rate = _pool_flip_rates([(4, 0.5, 0.5)])
        assert pre_flip_fail_rate == 0.5
        assert alive_rate == 0.5

    def test_none_alive_rate_subcall_contributes_zero_alive_but_still_counts_trials(self):
        # alive_rate=None means num_reached==0 for that subcall (every trial pre-flip-failed) --
        # it must still count toward total_trials/total_pre_flip_fail, just contribute 0 alive.
        alive_rate, pre_flip_fail_rate = _pool_flip_rates([(2, None, 1.0), (2, 1.0, 0.0)])
        assert pre_flip_fail_rate == 0.5  # (2 + 0) / 4
        assert alive_rate == 1.0  # (0 + 2) / (0 + 2)

    def test_skips_non_positive_n_entries(self):
        assert _pool_flip_rates([(0, 1.0, 0.0), (2, 0.5, 0.5)]) == _pool_flip_rates([(2, 0.5, 0.5)])

    def test_empty_input_returns_none_none(self):
        assert _pool_flip_rates([]) == (None, None)


class TestKickToLocoFlipStrikePhaseSplit:
    """split_by_strike_phase: true -- run_eval-level behavior. get_strike_window_ticks/
    get_skill_clip_length_ticks are mocked (both need a real onnxruntime + checkpoint, exercised
    by hand instead -- see this file's own top-of-file docstring); record_kick_to_loco_flip_scan
    is mocked the same way every other dispatch test in this file already mocks it."""

    def _cfg(self, **overrides):
        from holosoma.sim2sim_eval import EvalSpec, OnnxTarget

        defaults = dict(
            onnx_targets=[OnnxTarget(path="/fake/model_0100000.onnx", name="t", skill_ids=[0])],
            evaluations=[
                EvalSpec(type="kick_to_loco_flip", params={"num_trials": 4, "split_by_strike_phase": True})
            ],
        )
        defaults.update(overrides)
        return Sim2SimEvalConfig(**defaults)

    @staticmethod
    def _fake_wandb(logged_dicts):
        class _FakeWandb:
            def init(self, **kwargs):
                pass

            def log(self, d, step=None):
                logged_dicts.append(d)

            def finish(self):
                pass

        return _FakeWandb()

    def test_all_three_windows_fully_auto_derived_and_nonstrike_pooled(self):
        """No flip_delay_min/max_steps anywhere in this config -- strike, nonstrike_pre, and
        nonstrike_post all come entirely from get_strike_window_ticks/get_skill_clip_length_ticks,
        num_trials=4 split 2/2 between the two nonstrike sub-windows, pooled into one nonstrike
        rate."""
        cfg = self._cfg()
        calls = []

        def fake_flip(**kwargs):
            calls.append((kwargs["flip_delay_min_steps"], kwargs["flip_delay_max_steps"], kwargs["step_label"], kwargs["num_trials"]))
            lo = kwargs["flip_delay_min_steps"]
            if lo == 200:
                return (0.9, 0.1)  # strike
            if lo == 10:
                return (1.0, 0.0)  # nonstrike_pre: n=2, 0 pre-flip-fail, both reached alive
            return (0.0, 0.0)  # nonstrike_post (lo == 250): n=2, 0 pre-flip-fail, both reached dead

        logged_dicts = []
        with patch("holosoma.sim2sim_eval.get_strike_window_ticks", return_value=(200, 250)), \
             patch("holosoma.sim2sim_eval.get_skill_clip_length_ticks", return_value=300), \
             patch("holosoma.sim2sim_eval.record_kick_to_loco_flip_scan", side_effect=fake_flip), \
             patch("os.path.exists", return_value=True), \
             patch.dict("sys.modules", {"wandb": self._fake_wandb(logged_dicts)}):
            count = run_eval(cfg)

        assert count == 4  # strike x2 + pooled nonstrike x2
        assert calls == [
            (200, 249, "100000/strike", 4),          # stand_start_tick=250 -> inclusive hi=249
            (10, 199, "100000/nonstrike_pre", 2),     # [_NONSTRIKE_MIN_STEPS, strike_lo-1], n=4//2
            (250, 299, "100000/nonstrike_post", 2),   # (stand_start_tick, clip_last_tick=299], n=4-2
        ]
        # pooled nonstrike: pre_flip_fail = (0+0)/4 = 0.0; alive = (2+0)/(2+2) = 0.5
        assert logged_dicts == [{
            "t/0/kick_to_loco/strike/alive_rate": 0.9,
            "t/0/kick_to_loco/strike/pre_flip_fail_rate": 0.1,
            "t/0/kick_to_loco/nonstrike/alive_rate": 0.5,
            "t/0/kick_to_loco/nonstrike/pre_flip_fail_rate": 0.0,
        }]

    def test_no_strike_boundaries_raises_and_is_caught_by_run_eval(self):
        """run_eval's own per-(eval,skill) try/except catches this -- logged and skipped, not a
        crash of the whole run (same "one bad target doesn't stop the rest" convention as every
        other dispatch crash)."""
        cfg = self._cfg()
        with patch("holosoma.sim2sim_eval.get_strike_window_ticks", return_value=None), \
             patch("holosoma.sim2sim_eval.record_kick_to_loco_flip_scan") as mock_flip, \
             patch("os.path.exists", return_value=True), \
             patch.dict("sys.modules", {"wandb": self._fake_wandb([])}):
            count = run_eval(cfg)

        assert count == 0
        mock_flip.assert_not_called()

    def test_no_clip_length_metadata_raises_and_is_caught_by_run_eval(self):
        cfg = self._cfg()
        with patch("holosoma.sim2sim_eval.get_strike_window_ticks", return_value=(200, 250)), \
             patch("holosoma.sim2sim_eval.get_skill_clip_length_ticks", return_value=None), \
             patch("holosoma.sim2sim_eval.record_kick_to_loco_flip_scan") as mock_flip, \
             patch("os.path.exists", return_value=True), \
             patch.dict("sys.modules", {"wandb": self._fake_wandb([])}):
            count = run_eval(cfg)

        assert count == 0
        mock_flip.assert_not_called()

    def test_strike_starting_too_early_skips_nonstrike_pre_but_post_still_runs(self):
        """strike_lo <= _NONSTRIKE_MIN_STEPS (10) leaves no room for nonstrike_pre, but
        nonstrike_post can still run if the clip extends past stand_start_tick -- nonstrike ends
        up entirely from the post sub-window (gets the FULL num_trials, not half, since only one
        nonstrike sub-window exists)."""
        cfg = self._cfg()

        def fake_flip(**kwargs):
            lo = kwargs["flip_delay_min_steps"]
            if lo == 5:
                return (0.7, 0.2)  # strike
            return (0.5, 0.5)  # nonstrike_post (lo == 40): n=4 -> 2 pre-flip-fail, 1 of 2 alive

        logged_dicts = []
        with patch("holosoma.sim2sim_eval.get_strike_window_ticks", return_value=(5, 40)), \
             patch("holosoma.sim2sim_eval.get_skill_clip_length_ticks", return_value=100), \
             patch("holosoma.sim2sim_eval.record_kick_to_loco_flip_scan", side_effect=fake_flip) as mock_flip, \
             patch("os.path.exists", return_value=True), \
             patch.dict("sys.modules", {"wandb": self._fake_wandb(logged_dicts)}):
            count = run_eval(cfg)

        assert mock_flip.call_count == 2  # strike + nonstrike_post (nonstrike_pre skipped)
        assert count == 4
        assert logged_dicts == [{
            "t/0/kick_to_loco/strike/alive_rate": 0.7,
            "t/0/kick_to_loco/strike/pre_flip_fail_rate": 0.2,
            "t/0/kick_to_loco/nonstrike/alive_rate": 0.5,
            "t/0/kick_to_loco/nonstrike/pre_flip_fail_rate": 0.5,
        }]

    def test_no_room_for_either_nonstrike_subwindow_skips_nonstrike_entirely(self):
        """Degenerate: strike starts too early for nonstrike_pre AND the clip ends before
        nonstrike_post could start -- nonstrike/* is None (dropped), strike/* unaffected."""
        cfg = self._cfg()

        def fake_flip(**kwargs):
            return (1.0, 0.0)

        logged_dicts = []
        with patch("holosoma.sim2sim_eval.get_strike_window_ticks", return_value=(5, 6)), \
             patch("holosoma.sim2sim_eval.get_skill_clip_length_ticks", return_value=6), \
             patch("holosoma.sim2sim_eval.record_kick_to_loco_flip_scan", side_effect=fake_flip) as mock_flip, \
             patch("os.path.exists", return_value=True), \
             patch.dict("sys.modules", {"wandb": self._fake_wandb(logged_dicts)}):
            count = run_eval(cfg)

        assert mock_flip.call_count == 1  # only strike ran
        assert count == 2  # strike/alive_rate + strike/pre_flip_fail_rate only
        assert logged_dicts == [{
            "t/0/kick_to_loco/strike/alive_rate": 1.0,
            "t/0/kick_to_loco/strike/pre_flip_fail_rate": 0.0,
        }]

    def test_split_off_by_default_keeps_old_unprefixed_keys(self):
        """split_by_strike_phase omitted (default False) -- byte-identical to the pre-split
        behavior, including the plain (non-prefixed) key names."""
        from holosoma.sim2sim_eval import EvalSpec

        cfg = self._cfg(evaluations=[EvalSpec(type="kick_to_loco_flip", params={"num_trials": 4})])
        logged_dicts = []
        with patch("holosoma.sim2sim_eval.get_strike_window_ticks") as mock_window, \
             patch("holosoma.sim2sim_eval.get_skill_clip_length_ticks") as mock_clip_len, \
             patch("holosoma.sim2sim_eval.record_kick_to_loco_flip_scan", return_value=(0.8, 0.05)), \
             patch("os.path.exists", return_value=True), \
             patch.dict("sys.modules", {"wandb": self._fake_wandb(logged_dicts)}):
            run_eval(cfg)

        mock_window.assert_not_called()  # never even consulted when the split isn't requested
        mock_clip_len.assert_not_called()
        assert logged_dicts == [{"t/0/kick_to_loco/alive_rate": 0.8, "t/0/kick_to_loco/pre_flip_fail_rate": 0.05}]


class TestRunEvalConcurrency:
    """max_concurrent_scans (default 1 -- exact prior sequential behavior) and the unconditional
    unique-per-call lock path (_fresh_lock_path), added 2026-09-06 after CPU-idle capacity (128
    cores, ~3 in use) and cross-process lock collisions were both found live against a real
    100-iteration run."""

    def _cfg(self, **overrides):
        from holosoma.sim2sim_eval import EvalSpec, OnnxTarget

        defaults = dict(
            onnx_targets=[
                OnnxTarget(path="/fake/skillA/model_0100000.onnx", name="skillA", skill_ids=[0]),
                OnnxTarget(path="/fake/skillB/model_0200000.onnx", name="skillB", skill_ids=[0]),
            ],
            evaluations=[EvalSpec(type="kick_survival", params={"num_trials": 4})],
        )
        defaults.update(overrides)
        return Sim2SimEvalConfig(**defaults)

    @staticmethod
    def _fake_wandb(logged_dicts):
        class _FakeWandb:
            def init(self, **kwargs):
                pass

            def log(self, d, step=None):
                logged_dicts.append(d)

            def finish(self):
                pass

        return _FakeWandb()

    def test_default_max_concurrent_scans_one_runs_and_logs_identically_to_before(self):
        """max_concurrent_scans omitted (default 1) -- byte-identical result shape/order to the
        pre-concurrency sequential loop, just now routed through a 1-worker pool."""
        cfg = self._cfg()
        logged_dicts = []
        with patch("holosoma.sim2sim_eval.record_survival_scan", return_value=(0.1, 0.6, None)), \
             patch("os.path.exists", return_value=True), \
             patch.dict("sys.modules", {"wandb": self._fake_wandb(logged_dicts)}):
            count = run_eval(cfg)

        assert count == 4  # 2 targets x (fall_rate, hit_rate)
        assert logged_dicts == [
            {"skillA/0/kick/fall_rate": 0.1, "skillA/0/kick/hit_rate": 0.6},
            {"skillB/0/kick/fall_rate": 0.1, "skillB/0/kick/hit_rate": 0.6},
        ]

    def test_two_dispatch_calls_genuinely_overlap_when_max_concurrent_scans_is_2(self):
        """Proof of actual concurrency, not just that the config value is accepted: two
        synchronization barriers force each fake scan call to block until BOTH have started,
        which can only succeed if they're running on separate threads at the same time -- with
        max_concurrent_scans=1 this would deadlock (the second call could never start until the
        first, still waiting on the barrier, returns)."""
        import threading

        cfg = self._cfg(max_concurrent_scans=2)
        barrier = threading.Barrier(2, timeout=5.0)

        def fake_survival(**kwargs):
            barrier.wait()  # raises BrokenBarrierError on timeout if never joined by a 2nd caller
            return (0.1, 0.6, None)

        with patch("holosoma.sim2sim_eval.record_survival_scan", side_effect=fake_survival), \
             patch("os.path.exists", return_value=True), \
             patch.dict("sys.modules", {"wandb": self._fake_wandb([])}):
            count = run_eval(cfg)  # would raise BrokenBarrierError inside a worker thread if serialized

        assert count == 4

    def test_max_concurrent_scans_one_still_processes_all_targets_sequentially_safe(self):
        """With the default (no barrier trickery needed), a slow/blocking fake call must not hang
        the whole run -- confirms the 1-worker pool still drains every submitted job in order."""
        cfg = self._cfg()  # max_concurrent_scans defaults to 1
        call_order = []

        def fake_survival(**kwargs):
            call_order.append(kwargs["onnx_path"])
            return (0.0, 1.0, None)

        with patch("holosoma.sim2sim_eval.record_survival_scan", side_effect=fake_survival), \
             patch("os.path.exists", return_value=True), \
             patch.dict("sys.modules", {"wandb": self._fake_wandb([])}):
            run_eval(cfg)

        assert call_order == ["/fake/skillA/model_0100000.onnx", "/fake/skillB/model_0200000.onnx"]

    def test_each_dispatch_call_gets_a_unique_never_reused_lock_path(self):
        """_fresh_lock_path is applied unconditionally (not just when max_concurrent_scans > 1) --
        see that function's own docstring for why reusing the shared default lock is a pure
        liability for this tool even when running fully sequentially (collision with an unrelated
        live training process's own periodic scans, reproduced live 2026-09-06)."""
        cfg = self._cfg()
        seen_lock_paths = []

        def fake_survival(**kwargs):
            seen_lock_paths.append(kwargs["lock_path"])
            return (0.0, 1.0, None)

        with patch("holosoma.sim2sim_eval.record_survival_scan", side_effect=fake_survival), \
             patch("os.path.exists", return_value=True), \
             patch.dict("sys.modules", {"wandb": self._fake_wandb([])}):
            run_eval(cfg)

        assert len(seen_lock_paths) == 2
        assert len(set(seen_lock_paths)) == 2  # both unique
        for p in seen_lock_paths:
            assert "survival" in p
            assert p.endswith(".lock")

    def test_kick_to_loco_flip_split_gives_each_of_its_three_sub_scans_a_unique_lock(self):
        from holosoma.sim2sim_eval import EvalSpec, OnnxTarget

        cfg = self._cfg(
            onnx_targets=[OnnxTarget(path="/fake/model_0100000.onnx", name="t", skill_ids=[0])],
            evaluations=[EvalSpec(type="kick_to_loco_flip", params={"num_trials": 4, "split_by_strike_phase": True})],
        )
        seen_lock_paths = []

        def fake_flip(**kwargs):
            seen_lock_paths.append(kwargs["lock_path"])
            return (0.9, 0.1)

        with patch("holosoma.sim2sim_eval.get_strike_window_ticks", return_value=(200, 250)), \
             patch("holosoma.sim2sim_eval.get_skill_clip_length_ticks", return_value=300), \
             patch("holosoma.sim2sim_eval.record_kick_to_loco_flip_scan", side_effect=fake_flip), \
             patch("os.path.exists", return_value=True), \
             patch.dict("sys.modules", {"wandb": self._fake_wandb([])}):
            run_eval(cfg)

        assert len(seen_lock_paths) == 3  # strike, nonstrike_pre, nonstrike_post
        assert len(set(seen_lock_paths)) == 3


class TestRunEvalTimestampedRunName:
    """2026-09-06: wandb's own run `name` is always stamped with run_timestamp, so re-running the
    same config never collides on one wandb run name -- see run_eval's own WANDB docstring note."""

    def _cfg(self, **overrides):
        from holosoma.sim2sim_eval import EvalSpec, OnnxTarget

        defaults = dict(
            onnx_targets=[OnnxTarget(path="/fake/model_0100000.onnx", name="t", skill_ids=[0])],
            evaluations=[EvalSpec(type="kick_survival", params={"num_trials": 4})],
        )
        defaults.update(overrides)
        return Sim2SimEvalConfig(**defaults)

    def test_bare_timestamp_used_when_no_run_name_configured(self):
        captured_init_kwargs = {}

        class _FakeWandb:
            def init(self, **kwargs):
                captured_init_kwargs.update(kwargs)

            def log(self, d, step=None):
                pass

            def finish(self):
                pass

        with patch("holosoma.sim2sim_eval.record_survival_scan", return_value=(0.1, 0.6, None)), \
             patch("os.path.exists", return_value=True), \
             patch.dict("sys.modules", {"wandb": _FakeWandb()}):
            run_eval(self._cfg(), run_timestamp="20260906_064355")

        assert captured_init_kwargs["name"] == "20260906_064355"

    def test_run_name_gets_timestamp_prefixed_when_configured(self):
        captured_init_kwargs = {}

        class _FakeWandb:
            def init(self, **kwargs):
                captured_init_kwargs.update(kwargs)

            def log(self, d, step=None):
                pass

            def finish(self):
                pass

        with patch("holosoma.sim2sim_eval.record_survival_scan", return_value=(0.1, 0.6, None)), \
             patch("os.path.exists", return_value=True), \
             patch.dict("sys.modules", {"wandb": _FakeWandb()}):
            run_eval(self._cfg(wandb_run_name="skill011-vs-skill015"), run_timestamp="20260906_064355")

        assert captured_init_kwargs["name"] == "20260906_064355-skill011-vs-skill015"

    def test_timestamp_auto_generated_when_not_passed(self):
        """No run_timestamp given -- run_eval generates its own via datetime.now(), still in the
        %Y%m%d_%H%M%S shape (this project's own training-run-directory timestamp convention)."""
        import re

        captured_init_kwargs = {}

        class _FakeWandb:
            def init(self, **kwargs):
                captured_init_kwargs.update(kwargs)

            def log(self, d, step=None):
                pass

            def finish(self):
                pass

        with patch("holosoma.sim2sim_eval.record_survival_scan", return_value=(0.1, 0.6, None)), \
             patch("os.path.exists", return_value=True), \
             patch.dict("sys.modules", {"wandb": _FakeWandb()}):
            run_eval(self._cfg())

        assert re.fullmatch(r"\d{8}_\d{6}", captured_init_kwargs["name"])


class TestRunEvalOutputFiles:
    """output_dir (default None on run_eval() itself -- only main()'s CLI path opts into
    "out/sim2sim_eval") -- writes a long-format CSV of every logged data point plus a per-metric
    summary JSON, both under output_dir/<run_name>/ (same string as wandb's own run name) as
    sim2sim_eval.csv / sim2sim_eval.json, so the folder pairs up with the wandb run of the same
    name.

    `output_dir` in every test below is `tmp_path` itself (already exists on disk), NOT a fresh
    `tmp_path / "output"` -- this whole file's own `patch("os.path.exists", return_value=True)`
    convention (needed for run_eval's own onnx-target existence check) makes `os.makedirs`
    believe every path's parent already exists too (its recursion check IS `os.path.exists`),
    so it skips creating any but the LAST path segment -- fine when that segment's real parent
    already exists (`tmp_path`), a `FileNotFoundError` when it doesn't (a fresh two-level
    `tmp_path/output/<run_name>`). Not a production bug -- confirmed via a real, unpatched
    end-to-end run creating a brand new nested --output-dir just fine; purely an artifact of this
    test file's own broad exists()-patching."""

    def _cfg(self, **overrides):
        from holosoma.sim2sim_eval import EvalSpec, OnnxTarget

        defaults = dict(
            onnx_targets=[
                OnnxTarget(path="/fake/skillA/model_0100000.onnx", name="skillA", skill_ids=[0]),
            ],
            evaluations=[EvalSpec(type="kick_survival", params={"num_trials": 4})],
        )
        defaults.update(overrides)
        return Sim2SimEvalConfig(**defaults)

    @staticmethod
    def _fake_wandb():
        class _FakeWandb:
            def init(self, **kwargs):
                pass

            def log(self, d, step=None):
                pass

            def finish(self):
                pass

        return _FakeWandb()

    def test_output_dir_none_writes_nothing(self, tmp_path):
        """The default (run_eval called directly, as every other test in this file does) must have
        zero on-disk side effects -- this is what keeps the rest of this suite from littering the
        real repo with output/ files every time it runs."""
        with patch("holosoma.sim2sim_eval.record_survival_scan", return_value=(0.1, 0.6, None)), \
             patch("os.path.exists", return_value=True), \
             patch.dict("sys.modules", {"wandb": self._fake_wandb()}):
            run_eval(self._cfg())  # output_dir defaults to None

        assert list(tmp_path.iterdir()) == []  # nothing written anywhere this test could see

    def test_files_land_under_a_subfolder_named_after_the_wandb_run_name(self, tmp_path):
        """No wandb.run_name configured -> the wandb run name (and thus the output subfolder) is
        the bare run_timestamp."""
        import csv
        import json

        out_dir = tmp_path  # already exists -- see this class's own note on os.path.exists patching below
        with patch("holosoma.sim2sim_eval.record_survival_scan", return_value=(0.1, 0.6, None)), \
             patch("os.path.exists", return_value=True), \
             patch.dict("sys.modules", {"wandb": self._fake_wandb()}):
            count = run_eval(
                self._cfg(), run_timestamp="20260906_064355", output_dir=str(out_dir), config_path="cfg.yaml"
            )

        assert count == 2  # fall_rate + hit_rate
        run_dir = out_dir / "20260906_064355"
        csv_path = run_dir / "sim2sim_eval.csv"
        json_path = run_dir / "sim2sim_eval.json"
        assert csv_path.exists()
        assert json_path.exists()

        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        by_metric = {r["metric"]: r for r in rows}
        assert by_metric["fall_rate"]["value"] == "0.1"
        assert by_metric["fall_rate"]["target"] == "skillA"
        assert by_metric["fall_rate"]["skill_id"] == "0"
        assert by_metric["fall_rate"]["group"] == "kick"
        assert by_metric["fall_rate"]["wandb_key"] == "skillA/0/kick/fall_rate"
        assert by_metric["hit_rate"]["value"] == "0.6"

        with open(json_path) as f:
            summary = json.load(f)
        assert summary["run_timestamp"] == "20260906_064355"
        assert summary["config_path"] == "cfg.yaml"
        assert summary["logged_count"] == 2
        assert summary["wandb"]["run_name"] == "20260906_064355"
        fall_rate_summary = summary["results"]["skillA"]["0"]["kick"]["fall_rate"]
        assert fall_rate_summary == {"n": 1, "mean": 0.1, "std": 0.0, "min": 0.1, "max": 0.1, "last": 0.1}
        assert summary["targets"] == [
            {"name": "skillA", "path": "/fake/skillA/model_0100000.onnx", "base_step": 100000,
             "skill_ids": [0], "kick_aim_enabled": True}
        ]

    def test_subfolder_name_includes_configured_run_name(self, tmp_path):
        """wandb.run_name configured -> the output subfolder is "<run_timestamp>-<run_name>",
        exactly matching wandb's own run name -- see run_eval's own WANDB docstring note."""
        out_dir = tmp_path  # already exists -- see this class's own note on os.path.exists patching below
        with patch("holosoma.sim2sim_eval.record_survival_scan", return_value=(0.1, 0.6, None)), \
             patch("os.path.exists", return_value=True), \
             patch.dict("sys.modules", {"wandb": self._fake_wandb()}):
            run_eval(
                self._cfg(wandb_run_name="skill011-vs-skill015"),
                run_timestamp="20260906_064355", output_dir=str(out_dir),
            )

        run_dir = out_dir / "20260906_064355-skill011-vs-skill015"
        assert (run_dir / "sim2sim_eval.csv").exists()
        assert (run_dir / "sim2sim_eval.json").exists()

    def test_multiple_iterations_pool_into_one_summary_entry_per_metric(self, tmp_path):
        """3 iterations of the same metric -> one CSV row each, but ONE summary entry aggregating
        all 3 (n=3), not 3 separate summary entries -- the whole point of the JSON being a
        collapsed-across-iterations view rather than a second copy of the CSV."""
        import json

        out_dir = tmp_path  # already exists -- see this class's own note on os.path.exists patching below
        values = iter([(0.0, 1.0, None), (0.2, 0.8, None), (0.4, 0.6, None)])

        with patch("holosoma.sim2sim_eval.record_survival_scan", side_effect=lambda **kw: next(values)), \
             patch("os.path.exists", return_value=True), \
             patch.dict("sys.modules", {"wandb": self._fake_wandb()}):
            run_eval(self._cfg(iterations=3), run_timestamp="20260906_064355", output_dir=str(out_dir))

        run_dir = out_dir / "20260906_064355"
        with open(run_dir / "sim2sim_eval.json") as f:
            summary = json.load(f)
        fall_rate_summary = summary["results"]["skillA"]["0"]["kick"]["fall_rate"]
        assert fall_rate_summary["n"] == 3
        assert fall_rate_summary["mean"] == pytest.approx(0.2)
        assert fall_rate_summary["min"] == 0.0
        assert fall_rate_summary["max"] == 0.4
        assert fall_rate_summary["last"] == pytest.approx(0.4)

        with open(run_dir / "sim2sim_eval.csv", newline="") as f:
            import csv as _csv

            rows = list(_csv.DictReader(f))
        assert len([r for r in rows if r["metric"] == "fall_rate"]) == 3  # one CSV row per iteration

    def test_a_scan_crash_still_leaves_partial_results_on_disk(self, tmp_path):
        """A crash in one dispatch call must not prevent already-collected rows from earlier
        targets/iterations from being written -- see _write_output_files' own "called from finally"
        docstring note."""
        import json

        from holosoma.sim2sim_eval import EvalSpec, OnnxTarget

        cfg = self._cfg(
            onnx_targets=[
                OnnxTarget(path="/fake/skillA/model_0100000.onnx", name="skillA", skill_ids=[0]),
                OnnxTarget(path="/fake/skillB/model_0200000.onnx", name="skillB", skill_ids=[0]),
            ],
        )
        out_dir = tmp_path  # already exists -- see this class's own note on os.path.exists patching below

        def fake_survival(**kwargs):
            if kwargs["onnx_path"].endswith("skillB/model_0200000.onnx"):
                raise RuntimeError("boom")
            return (0.1, 0.6, None)

        with patch("holosoma.sim2sim_eval.record_survival_scan", side_effect=fake_survival), \
             patch("os.path.exists", return_value=True), \
             patch.dict("sys.modules", {"wandb": self._fake_wandb()}):
            count = run_eval(cfg, run_timestamp="20260906_064355", output_dir=str(out_dir))

        assert count == 2  # only skillA's fall_rate + hit_rate made it through
        with open(out_dir / "20260906_064355" / "sim2sim_eval.json") as f:
            summary = json.load(f)
        assert "skillA" in summary["results"]
        assert "skillB" not in summary["results"]  # crashed before producing any usable result

    def test_run_name_containing_a_slash_is_sanitized_to_one_directory_segment(self, tmp_path):
        """A wandb run name is a free-form string and could contain "/" (e.g. a hand-typed
        run_name) -- _sanitize_run_subdir must collapse it to one path segment rather than nesting
        unexpected subdirectories or escaping output_dir."""
        out_dir = tmp_path  # already exists -- see this class's own note on os.path.exists patching below
        with patch("holosoma.sim2sim_eval.record_survival_scan", return_value=(0.1, 0.6, None)), \
             patch("os.path.exists", return_value=True), \
             patch.dict("sys.modules", {"wandb": self._fake_wandb()}):
            run_eval(
                self._cfg(wandb_run_name="a/b"),
                run_timestamp="20260906_064355", output_dir=str(out_dir),
            )

        run_dir = out_dir / "20260906_064355-a_b"
        assert (run_dir / "sim2sim_eval.csv").exists()
        assert not (out_dir / "20260906_064355-a").exists()  # no unintended extra nesting
