"""Unit tests for UnifiedManager._maybe_flip_kick_recovery_to_locomotion (2026-08-09, Stage D's
post-swing -> locomotion handoff, kick_recovery_locomotion_flip_enabled; boundary settled
2026-08-10 at pre_recovery_motion_end_idx -- the end of the WHOLE authored/raw clip (approach +
strike + the actor's own captured post-kick follow-through), after two same-day experiments with
narrower (stand_start_idx) and wider (motion_end_idx) boundaries were tried, live-tested, and
superseded by this one). Isolated via a bare instance, same pattern as
test_unified_manager_task_mode_partition.py -- the method only touches
task_mode/episode_length_buf/_prev_in_raw_clip_phase/command_manager, not the full sim/terrain/scene.
"""

import torch

from holosoma.envs.unified.unified_manager import TaskMode, UnifiedManager


class _FakeMotionCommand:
    def __init__(self, time_steps: torch.Tensor, pre_recovery_motion_end_idx: list[int], motion_ids: torch.Tensor | None = None):
        self.time_steps = time_steps
        self.pre_recovery_motion_end_idx = torch.tensor(pre_recovery_motion_end_idx, dtype=torch.long)
        self.motion_ids = motion_ids if motion_ids is not None else torch.zeros_like(time_steps)


class _FakeLocomotionCommand:
    def __init__(self):
        self.pinned_calls: list[torch.Tensor] = []

    def pin_zero(self, env_ids: torch.Tensor) -> None:
        self.pinned_calls.append(env_ids.clone())


class _FakeCommandManager:
    def __init__(self, motion_command: _FakeMotionCommand, locomotion_command: _FakeLocomotionCommand):
        self._states = {"motion_command": motion_command, "locomotion_command": locomotion_command}

    def get_state(self, name: str):
        return self._states.get(name)


def _make_manager(
    num_envs: int, boundary: int = 300, enabled: bool = True, enabled_per_skill: list[bool] | None = None
) -> tuple[UnifiedManager, _FakeLocomotionCommand]:
    m = object.__new__(UnifiedManager)
    m.num_envs = num_envs
    m.device = "cpu"
    m.task_mode = torch.full((num_envs,), int(TaskMode.KICK), dtype=torch.long)
    m.episode_length_buf = torch.full((num_envs,), 10, dtype=torch.long)  # well past the ep_len<=1 guard
    m.skill_id = torch.zeros(num_envs, dtype=torch.long)
    m._kick_recovery_locomotion_flip_enabled = enabled
    # 2026-08-15, Tier 3 Group B Wave 2: None (the legacy default) unless a test explicitly passes
    # enabled_per_skill -- see test_enabled_per_skill_diverges_between_envs below.
    m._kick_recovery_locomotion_flip_enabled_per_skill = (
        torch.tensor(enabled_per_skill, dtype=torch.bool) if enabled_per_skill is not None else None
    )
    m._prev_in_raw_clip_phase = torch.ones(num_envs, dtype=torch.bool)  # "was in raw clip" last tick
    m._post_flip_step = torch.full((num_envs,), -1, dtype=torch.long)  # sentinel: not post-flip
    # Kick ABORT (2026-08-28): sentinel -1 = "no abort drawn", i.e. feature off -- which is what
    # every config this file targets has. _maybe_flip_kick_recovery_to_locomotion reads this
    # unconditionally (the abort trigger is OR-ed into the boundary one), so it must exist even
    # though these tests exercise only the boundary path. See test_kick_abort_flip.py.
    m._kick_abort_flip_tick = torch.full((num_envs,), -1, dtype=torch.long)
    m._kick_abort_prob = 0.0  # feature off -- _resample_task_mode's draw block short-circuits
    # Locomotion -> kick direction's own state (2026-08-13): off by default here (0.0), matching
    # every config this test's own module docstring targets (Stage A/B/C1/C2 never set it) --
    # _resample_task_mode touches these buffers unconditionally (_clear_kick_pending), so they
    # must exist even though this file's tests only exercise the opposite (kick->locomotion)
    # direction. See test_unified_manager_kick_pending.py for the direction this state backs.
    m._mid_episode_kick_entry_prob = 0.0
    m._mid_episode_kick_entry_prob_per_skill = None
    m._pre_kick_step = torch.full((num_envs,), -1, dtype=torch.long)
    m._kick_pending = torch.zeros(num_envs, dtype=torch.bool)
    m._kick_pending_best_residual = torch.full((num_envs,), float("inf"))
    m._kick_pending_best_frame = torch.full((num_envs,), -1, dtype=torch.long)
    m._pre_kick_fallback_active = torch.zeros(num_envs, dtype=torch.bool)
    m._pre_kick_fallback_start_step = torch.full((num_envs,), -1, dtype=torch.long)
    m._pre_kick_fallback_start_vx = torch.zeros(num_envs)
    m._pre_kick_fallback_stale_ticks = torch.zeros(num_envs, dtype=torch.long)
    loco_cmd = _FakeLocomotionCommand()
    time_steps = torch.zeros(num_envs, dtype=torch.long)  # well below any reasonable boundary
    m.command_manager = _FakeCommandManager(_FakeMotionCommand(time_steps, [boundary]), loco_cmd)
    return m, loco_cmd


def test_disabled_flag_is_an_exact_no_op():
    m, loco_cmd = _make_manager(num_envs=3, boundary=300, enabled=False)
    m.command_manager.get_state("motion_command").time_steps[:] = 300  # crossing

    m._maybe_flip_kick_recovery_to_locomotion()

    assert torch.all(m.task_mode == TaskMode.KICK), "flag off must never touch task_mode"
    assert loco_cmd.pinned_calls == []


def test_fires_exactly_on_the_true_to_false_crossing():
    m, loco_cmd = _make_manager(num_envs=1, boundary=300)
    mc = m.command_manager.get_state("motion_command")

    # tick 1: still inside the raw clip (True->True) -- must NOT fire
    mc.time_steps[:] = 299
    m._maybe_flip_kick_recovery_to_locomotion()
    assert m.task_mode[0] == TaskMode.KICK
    assert loco_cmd.pinned_calls == []

    # tick 2: crosses into the synthetic tail (True->False) -- must fire exactly here
    mc.time_steps[:] = 300
    m._maybe_flip_kick_recovery_to_locomotion()
    assert m.task_mode[0] == TaskMode.LOCOMOTION
    assert len(loco_cmd.pinned_calls) == 1
    assert loco_cmd.pinned_calls[0].tolist() == [0]

    # tick 3: already flipped to LOCOMOTION -- task_mode == KICK guard prevents re-firing even
    # though still past the boundary (stays False->False)
    mc.time_steps[:] = 301
    m._maybe_flip_kick_recovery_to_locomotion()
    assert m.task_mode[0] == TaskMode.LOCOMOTION
    assert len(loco_cmd.pinned_calls) == 1, "must not re-fire/re-pin an already-flipped env"


def test_does_not_false_fire_on_rsi_landing_straight_past_the_boundary():
    """A fresh reset whose RSI draw happens to land AFTER pre_recovery_motion_end_idx (already past
    the boundary at episode_length_buf<=1) must not be treated as a crossing -- there was no raw
    clip content this episode to hand off FROM."""
    m, loco_cmd = _make_manager(num_envs=1, boundary=300)
    m.episode_length_buf[:] = 1  # fresh post-reset tick
    m._prev_in_raw_clip_phase[:] = False  # matches _init_buffers' own False initialization
    mc = m.command_manager.get_state("motion_command")
    mc.time_steps[:] = 350  # RSI landed past the boundary already

    m._maybe_flip_kick_recovery_to_locomotion()

    assert m.task_mode[0] == TaskMode.KICK, "no real crossing happened this episode -- must not flip"
    assert loco_cmd.pinned_calls == []


def test_only_kick_mode_envs_are_eligible_to_flip():
    m, loco_cmd = _make_manager(num_envs=2, boundary=300)
    m.task_mode[1] = TaskMode.LOCOMOTION  # env 1 is a genuine locomotion env
    mc = m.command_manager.get_state("motion_command")
    mc.time_steps[:] = 300  # both cross this tick

    m._maybe_flip_kick_recovery_to_locomotion()

    assert m.task_mode[0] == TaskMode.LOCOMOTION, "kick env 0 crosses and flips"
    assert m.task_mode[1] == TaskMode.LOCOMOTION, "loco env 1 was already locomotion, unaffected"
    assert loco_cmd.pinned_calls == [torch.tensor([0])], "pin_zero must only be called for the env that actually flipped"


def test_mixed_batch_only_crossing_envs_flip_and_pin():
    m, loco_cmd = _make_manager(num_envs=4, boundary=300)
    mc = m.command_manager.get_state("motion_command")
    # env0: crosses (True->False); env1: stays in raw clip (True->True); env2: crosses (True->False);
    # env3: already past (was False last tick too -- shouldn't be possible if _prev tracked
    # correctly, but exercise it defensively) -- set _prev accordingly per env.
    m._prev_in_raw_clip_phase = torch.tensor([True, True, True, False])
    mc.time_steps = torch.tensor([300, 250, 305, 310])

    m._maybe_flip_kick_recovery_to_locomotion()

    assert m.task_mode.tolist() == [TaskMode.LOCOMOTION, TaskMode.KICK, TaskMode.LOCOMOTION, TaskMode.KICK]
    assert sorted(loco_cmd.pinned_calls[0].tolist()) == [0, 2]


def test_uses_the_envs_own_motion_boundary_not_motion_zeros():
    """Per-motion (not global) boundary: env0 assigned to motion 0 (boundary 300), env1 to motion 1
    (boundary 500) -- crossing 300 must flip env0 but NOT env1."""
    m, loco_cmd = _make_manager(num_envs=2, boundary=300)
    mc = m.command_manager.get_state("motion_command")
    mc.pre_recovery_motion_end_idx = torch.tensor([300, 500])
    mc.motion_ids = torch.tensor([0, 1])
    mc.time_steps = torch.tensor([300, 300])

    m._maybe_flip_kick_recovery_to_locomotion()

    assert m.task_mode[0] == TaskMode.LOCOMOTION, "env0's own boundary (300) reached"
    assert m.task_mode[1] == TaskMode.KICK, "env1's own boundary (500) not yet reached"


def test_prev_in_raw_clip_phase_updates_every_call_regardless_of_enabled_flag():
    """Even when disabled, the buffer should stay consistent (harmless either way since it's
    unused while disabled) -- but specifically verify enabled-mode keeps it in sync across ticks,
    since a stale buffer would corrupt crossing detection on a later flag-flip."""
    m, _ = _make_manager(num_envs=1, boundary=300)
    mc = m.command_manager.get_state("motion_command")
    mc.time_steps[:] = 250
    m._maybe_flip_kick_recovery_to_locomotion()
    assert bool(m._prev_in_raw_clip_phase[0]) is True

    mc.time_steps[:] = 300
    m._maybe_flip_kick_recovery_to_locomotion()
    assert bool(m._prev_in_raw_clip_phase[0]) is False


def test_missing_motion_command_does_not_crash():
    m, loco_cmd = _make_manager(num_envs=2, boundary=300)
    m.command_manager._states["motion_command"] = None

    m._maybe_flip_kick_recovery_to_locomotion()  # must not raise

    assert torch.all(m.task_mode == TaskMode.KICK)
    assert loco_cmd.pinned_calls == []


# ---------------------------------------------------------------------------
# _post_flip_step / post_flip_steps_since (2026-08-12) -- shared state backing FIX 2
# (post_flip_termination_grace_steps) and FIX 4 (post_flip_reward_decay_steps).
# ---------------------------------------------------------------------------


def test_post_flip_step_stays_sentinel_until_the_flip_fires():
    m, _ = _make_manager(num_envs=1, boundary=300)
    mc = m.command_manager.get_state("motion_command")
    mc.time_steps[:] = 299
    m._maybe_flip_kick_recovery_to_locomotion()
    assert int(m._post_flip_step[0]) == -1


def test_post_flip_step_captures_episode_length_buf_at_the_flip_tick():
    m, _ = _make_manager(num_envs=1, boundary=300)
    m.episode_length_buf[:] = 205  # this env is 205 ticks into its episode when it flips
    mc = m.command_manager.get_state("motion_command")
    mc.time_steps[:] = 300

    m._maybe_flip_kick_recovery_to_locomotion()

    assert int(m._post_flip_step[0]) == 205


def test_post_flip_step_only_set_for_envs_that_actually_crossed_this_tick():
    m, _ = _make_manager(num_envs=2, boundary=300)
    m.episode_length_buf = torch.tensor([100, 250])
    m._prev_in_raw_clip_phase = torch.tensor([True, True])
    mc = m.command_manager.get_state("motion_command")
    mc.time_steps = torch.tensor([250, 300])  # env0 doesn't cross, env1 crosses

    m._maybe_flip_kick_recovery_to_locomotion()

    assert int(m._post_flip_step[0]) == -1, "env0 never flipped -- must stay the sentinel"
    assert int(m._post_flip_step[1]) == 250


def test_post_flip_step_disabled_flag_never_touches_the_buffer():
    m, _ = _make_manager(num_envs=1, boundary=300, enabled=False)
    m.command_manager.get_state("motion_command").time_steps[:] = 300

    m._maybe_flip_kick_recovery_to_locomotion()

    assert int(m._post_flip_step[0]) == -1


def test_post_flip_steps_since_is_zero_before_any_flip():
    m, _ = _make_manager(num_envs=1, boundary=300)
    m.episode_length_buf[:] = 500  # far into the episode, but never flipped
    assert int(m.post_flip_steps_since()[0]) == 0


def test_post_flip_steps_since_counts_up_after_the_flip():
    m, _ = _make_manager(num_envs=1, boundary=300)
    m.episode_length_buf[:] = 204
    mc = m.command_manager.get_state("motion_command")
    mc.time_steps[:] = 300
    m._maybe_flip_kick_recovery_to_locomotion()
    assert int(m.post_flip_steps_since()[0]) == 0, "the flip tick itself is 0 steps since"

    m.episode_length_buf[:] = 204 + 7  # 7 more ticks have elapsed
    assert int(m.post_flip_steps_since()[0]) == 7


def test_post_flip_steps_since_per_env_independent():
    m, _ = _make_manager(num_envs=3, boundary=300)
    m.episode_length_buf = torch.tensor([100, 200, 300])
    m._post_flip_step = torch.tensor([-1, 190, 250])  # env0 never flipped, env1/env2 did

    since = m.post_flip_steps_since()

    assert int(since[0]) == 0, "sentinel envs must read 0, not a garbage negative-derived value"
    assert int(since[1]) == 10
    assert int(since[2]) == 50


def test_resample_task_mode_clears_the_sentinel_on_genuine_reset():
    m, _ = _make_manager(num_envs=2, boundary=300)
    m._post_flip_step = torch.tensor([250, 250])
    m._forced_task_mode = torch.full((2,), -1, dtype=torch.long)
    m._task_mode_partition = torch.full((2,), int(TaskMode.KICK), dtype=torch.long)
    m._skill_id_partition = torch.zeros(2, dtype=torch.long)
    m.skill_id = torch.zeros(2, dtype=torch.long)
    m.is_evaluating = False
    m._pin_recording_env_modes = lambda env_ids: None  # not under test here

    m._resample_task_mode(torch.tensor([0]))  # only env 0 is genuinely resetting

    assert int(m._post_flip_step[0]) == -1, "resetting env must have its stale flip-tick cleared"
    assert int(m._post_flip_step[1]) == 250, "non-resetting env must be untouched"


# ============================================================================================
# 2026-08-15, Tier 3 Group B Wave 2 ("simultaneous per-skill task configs"):
# kick_recovery_locomotion_flip_enabled_per_skill -- lets different skills in the SAME run
# genuinely disagree on whether Stage D's post-swing -> locomotion handoff is active, via each
# skill's own task_config file (see UnifiedManager.__init__'s own comment on
# _skill_task_config_paths for how this is resolved -- mirrors reward.py/termination.py's
# per-skill mechanism but reads at __init__ time into a plain tensor, not through
# RewardTermCfg/TerminationTermCfg.params_per_skill, since this gates a state transition this
# class runs directly rather than a term-level param).
# ============================================================================================


def test_enabled_per_skill_only_the_enabled_skills_envs_flip():
    """Two envs both crossing the boundary the same tick -- env0's skill wants the flip, env1's
    doesn't. Isolates the new per-env enabled_mask folded into `crossed`."""
    m, loco_cmd = _make_manager(num_envs=2, boundary=300, enabled_per_skill=[True, False])
    m.skill_id = torch.tensor([0, 1])
    mc = m.command_manager.get_state("motion_command")
    mc.pre_recovery_motion_end_idx = torch.tensor([300])
    mc.motion_ids = torch.zeros(2, dtype=torch.long)
    mc.time_steps = torch.tensor([300, 300])

    m._maybe_flip_kick_recovery_to_locomotion()

    assert m.task_mode[0] == TaskMode.LOCOMOTION, "env0's skill wants the flip"
    assert m.task_mode[1] == TaskMode.KICK, "env1's skill does not"
    assert loco_cmd.pinned_calls == [torch.tensor([0])]


def test_enabled_per_skill_all_true_matches_scalar_true():
    m, loco_cmd = _make_manager(num_envs=2, boundary=300, enabled_per_skill=[True, True])
    m.skill_id = torch.tensor([0, 1])
    mc = m.command_manager.get_state("motion_command")
    mc.pre_recovery_motion_end_idx = torch.tensor([300])
    mc.motion_ids = torch.zeros(2, dtype=torch.long)
    mc.time_steps = torch.tensor([300, 300])

    m._maybe_flip_kick_recovery_to_locomotion()

    assert torch.all(m.task_mode == TaskMode.LOCOMOTION)
    assert sorted(loco_cmd.pinned_calls[0].tolist()) == [0, 1]


def test_enabled_per_skill_still_flips_even_though_the_global_scalar_is_off():
    """The scalar `enabled=False` (what a legacy single-skill run, or a skill with no override,
    would resolve to as the base/global value) must NOT gate the whole method anymore once a
    per-skill table exists -- only the per-env mask should. Proves the early-return gate became
    an "any skill active" check, not a read of the (possibly False) scalar."""
    m, loco_cmd = _make_manager(num_envs=1, boundary=300, enabled=False, enabled_per_skill=[True])
    m.skill_id = torch.tensor([0])
    mc = m.command_manager.get_state("motion_command")
    mc.time_steps[:] = 300

    m._maybe_flip_kick_recovery_to_locomotion()

    assert m.task_mode[0] == TaskMode.LOCOMOTION
    assert len(loco_cmd.pinned_calls) == 1


def test_enabled_per_skill_all_false_never_flips_even_with_crossing():
    m, loco_cmd = _make_manager(num_envs=2, boundary=300, enabled_per_skill=[False, False])
    m.skill_id = torch.tensor([0, 1])
    mc = m.command_manager.get_state("motion_command")
    mc.pre_recovery_motion_end_idx = torch.tensor([300])
    mc.motion_ids = torch.zeros(2, dtype=torch.long)
    mc.time_steps = torch.tensor([300, 300])

    m._maybe_flip_kick_recovery_to_locomotion()

    assert torch.all(m.task_mode == TaskMode.KICK)
    assert loco_cmd.pinned_calls == []
