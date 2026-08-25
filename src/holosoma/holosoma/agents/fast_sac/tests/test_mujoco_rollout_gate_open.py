"""Unit tests for FastSACAgent._mujoco_rollout_gate_open, the shared cadence/eligibility gate for
all four periodic MuJoCo sim2sim checks (kick rollout, kick-handoff rollout, walk rollout,
survival scan).

Covers the 2026-08-23 bugfix: this gate used to check ONLY `self._kick_probability` (set
unconditionally from configs/skill_mix.yaml's kick_probability, regardless of whether N-skill
mode's own `motion_training_ratio`-based partition -- which never consults kick_probability at
all -- is what actually decided the run's real locomotion/kick split). A real run with
skill_mix.yaml left at kick_probability=0.0 (a Stage-A leftover) but motion_training_ratio=0.9 on
its one skill logged "0.450 of all envs permanently dedicated to a motion skill" at startup, yet
every sim2sim mechanism sharing this gate silently never fired for the whole run. The fix checks
BOTH `_kick_probability` and `sum(_skill_motion_training_ratios)`, either one being > 0.0 opens the
gate (subject to the cadence check).

Isolated via a bare instance (object.__new__), matching this project's existing bare-instance
convention for FastSACAgent unit tests (see test_mujoco_kick_rollout_per_skill.py) -- no real
onnx, MuJoCo, or training loop needed.
"""

from types import SimpleNamespace

from holosoma.agents.fast_sac.fast_sac_agent import FastSACAgent


class _FakeUnwrappedEnv:
    def __init__(self, kick_probability=0.0, skill_motion_training_ratios=None):
        self._kick_probability = kick_probability
        if skill_motion_training_ratios is not None:
            self._skill_motion_training_ratios = skill_motion_training_ratios


def _make_agent(
    *,
    kick_probability=0.0,
    skill_motion_training_ratios=None,
    global_step=0,
    save_interval=1000,
) -> FastSACAgent:
    a = object.__new__(FastSACAgent)
    a.unwrapped_env = _FakeUnwrappedEnv(kick_probability, skill_motion_training_ratios)
    a.global_step = global_step
    a.config = SimpleNamespace(save_interval=save_interval)
    return a


class TestMujocoRolloutGateOpen:
    def test_every_n_saves_zero_or_negative_always_closed(self):
        a = _make_agent(kick_probability=0.7, global_step=0)
        assert a._mujoco_rollout_gate_open(0) is False
        assert a._mujoco_rollout_gate_open(-1) is False

    def test_legacy_mode_kick_probability_zero_closes_the_gate(self):
        """Byte-identical to pre-fix behavior: no N-skill ratios at all (attribute absent, e.g. a
        non-UnifiedManager env or an older env class), kick_probability=0.0 -> Stage A, closed."""
        a = _make_agent(kick_probability=0.0, global_step=0)
        assert a._mujoco_rollout_gate_open(5) is False

    def test_legacy_mode_kick_probability_nonzero_opens_on_schedule(self):
        a = _make_agent(kick_probability=0.7, global_step=5000, save_interval=1000)
        assert a._mujoco_rollout_gate_open(5) is True  # 5000 % (1000*5) == 0

    def test_legacy_mode_kick_probability_nonzero_but_off_schedule_stays_closed(self):
        a = _make_agent(kick_probability=0.7, global_step=4000, save_interval=1000)
        assert a._mujoco_rollout_gate_open(5) is False  # 4000 % (1000*5) != 0

    def test_n_skill_mode_with_stale_zero_kick_probability_now_opens(self):
        """The actual bug: kick_probability=0.0 (Stage-A leftover in skill_mix.yaml) but N-skill
        mode's own motion_training_ratio=0.9 means this run genuinely trains kicking. Gate must
        now open."""
        a = _make_agent(
            kick_probability=0.0,
            skill_motion_training_ratios=[0.9],
            global_step=5000,
            save_interval=1000,
        )
        assert a._mujoco_rollout_gate_open(5) is True

    def test_n_skill_mode_empty_ratios_list_falls_through_to_kick_probability(self):
        """An empty list (N-skill mechanism present on the env class but not actually configured
        with any skill) must behave exactly like the legacy-mode case -- sums to 0.0, no
        different from the attribute being absent entirely."""
        a = _make_agent(kick_probability=0.0, skill_motion_training_ratios=[], global_step=0)
        assert a._mujoco_rollout_gate_open(5) is False

    def test_n_skill_mode_multi_skill_ratios_sum_correctly(self):
        a = _make_agent(
            kick_probability=0.0,
            skill_motion_training_ratios=[0.1, 0.05, 0.0],  # sums to 0.15 > 0.0
            global_step=5000,
            save_interval=1000,
        )
        assert a._mujoco_rollout_gate_open(5) is True

    def test_both_sources_zero_stays_closed(self):
        a = _make_agent(
            kick_probability=0.0,
            skill_motion_training_ratios=[0.0, 0.0],
            global_step=5000,
            save_interval=1000,
        )
        assert a._mujoco_rollout_gate_open(5) is False

    def test_both_sources_nonzero_still_respects_cadence(self):
        """Not a real N-skill/legacy simultaneous case in practice (only one path ever actually
        runs), but the gate's OR logic must not accidentally bypass the cadence check either way."""
        a = _make_agent(
            kick_probability=0.7,
            skill_motion_training_ratios=[0.9],
            global_step=4000,
            save_interval=1000,
        )
        assert a._mujoco_rollout_gate_open(5) is False  # 4000 % 5000 != 0
