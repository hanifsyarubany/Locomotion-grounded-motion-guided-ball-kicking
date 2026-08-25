"""Unit tests for UnifiedManager's locomotion->kick mid-episode entry state machine
(_maybe_enter_kick_from_locomotion / _enter_kick / _decline_kick / _clear_kick_pending,
2026-08-13 -- mid_episode_kick_entry_prob's own direction, the mirror of
kick_recovery_locomotion_flip_enabled covered by test_unified_manager_kick_recovery_locomotion_
flip.py). Full design: https://claude.ai/code/artifact/53c1da51-d841-4979-8bf8-efd5ea652e06 and
memory locomotion_to_kick_handoff_design_settled.md (decisions D2/D3/D8).

Isolated via a bare instance, same pattern as the flip test file -- the method under test only
touches task_mode/episode_length_buf/the kick-pending+fallback buffers/command_manager, not the
full sim/terrain/scene. MotionCommand.search_entry_point is faked with a scripted residual
sequence (one entry per tick) so the turning-point/ceiling/timeout logic can be exercised
deterministically without a real motion clip.
"""

import torch

from holosoma.envs.unified.unified_manager import TaskMode, UnifiedManager


class _FakeMotionCommand:
    def __init__(self, residuals: list[torch.Tensor], frames: list[torch.Tensor] | None = None):
        self._residuals = residuals
        self._frames = frames
        self._call = 0
        self.enter_calls: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.ball_calls: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.blend_calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def search_entry_point(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = self._residuals[self._call]
        frame = self._frames[self._call] if self._frames is not None else torch.full_like(env_ids, 100 + self._call)
        self._call += 1
        return frame, residual

    def enter_at_frame(self, env_ids: torch.Tensor, frames: torch.Tensor) -> None:
        self.enter_calls.append((env_ids.clone(), frames.clone()))

    def place_ball_at_entry(self, env_ids: torch.Tensor, frames: torch.Tensor) -> None:
        self.ball_calls.append((env_ids.clone(), frames.clone()))

    def capture_ref_blend(self, env_ids: torch.Tensor, blend_steps: torch.Tensor) -> None:
        self.blend_calls.append((env_ids.clone(), blend_steps.clone()))


class _FakeLocomotionCommand:
    def __init__(self, num_envs: int, command_dim: int = 3):
        self.commands = torch.zeros(num_envs, command_dim)
        self.pin_calls: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.unpin_calls: list[torch.Tensor] = []

    def pin(self, env_ids: torch.Tensor, values: torch.Tensor) -> None:
        self.commands[env_ids] = values
        self.pin_calls.append((env_ids.clone(), values.clone()))

    def unpin(self, env_ids: torch.Tensor) -> None:
        self.unpin_calls.append(env_ids.clone())

    def pin_zero(self, env_ids: torch.Tensor) -> None:
        self.commands[env_ids] = 0.0


class _FakeCommandManager:
    def __init__(self, motion_command, locomotion_command):
        self._states = {"motion_command": motion_command, "locomotion_command": locomotion_command}

    def get_state(self, name: str):
        return self._states.get(name)


def _make_manager(
    num_envs: int,
    motion_command,
    locomotion_command,
    *,
    prob: float = 1.0,
    min_steps: int = 0,
    max_residual: float = 0.0,
    decel_steps: float = 0.0,
    decel_target: float = 0.1,
    timeout_steps: float = 0.0,
    reference_blend_steps: float = 0.0,
    ball_fixed: bool = False,
    skill_id: list[int] | None = None,
    prob_per_skill: list[float] | None = None,
    min_steps_per_skill: list[int] | None = None,
    max_residual_per_skill: list[float] | None = None,
    decel_steps_per_skill: list[float] | None = None,
    decel_target_per_skill: list[float] | None = None,
    timeout_steps_per_skill: list[float] | None = None,
    reference_blend_steps_per_skill: list[float] | None = None,
) -> UnifiedManager:
    m = object.__new__(UnifiedManager)
    m.num_envs = num_envs
    m.device = "cpu"
    m.task_mode = torch.full((num_envs,), int(TaskMode.LOCOMOTION), dtype=torch.long)
    m.episode_length_buf = torch.full((num_envs,), 1000, dtype=torch.long)
    m.skill_id = torch.tensor(skill_id, dtype=torch.long) if skill_id is not None else torch.zeros(num_envs, dtype=torch.long)
    m._mid_episode_kick_entry_prob = prob
    m._mid_episode_kick_entry_min_steps = min_steps
    m._mid_episode_kick_entry_max_residual = max_residual
    m._pre_kick_decel_steps = decel_steps
    m._pre_kick_decel_target = decel_target
    m._pre_kick_fallback_timeout_steps = timeout_steps
    m._pre_kick_reference_blend_steps = reference_blend_steps
    m._mid_episode_kick_entry_ball_fixed = ball_fixed
    # 2026-08-15, Tier 3 Group B Wave 2: None (the legacy default) unless a test explicitly passes
    # a per-skill override -- see the "per-skill divergence" tests at the bottom of this file.
    m._mid_episode_kick_entry_prob_per_skill = (
        torch.tensor(prob_per_skill, dtype=torch.float32) if prob_per_skill is not None else None
    )
    m._mid_episode_kick_entry_min_steps_per_skill = (
        torch.tensor(min_steps_per_skill, dtype=torch.long) if min_steps_per_skill is not None else None
    )
    m._mid_episode_kick_entry_max_residual_per_skill = (
        torch.tensor(max_residual_per_skill, dtype=torch.float32) if max_residual_per_skill is not None else None
    )
    m._pre_kick_decel_steps_per_skill = (
        torch.tensor(decel_steps_per_skill, dtype=torch.float32) if decel_steps_per_skill is not None else None
    )
    m._pre_kick_decel_target_per_skill = (
        torch.tensor(decel_target_per_skill, dtype=torch.float32) if decel_target_per_skill is not None else None
    )
    m._pre_kick_fallback_timeout_steps_per_skill = (
        torch.tensor(timeout_steps_per_skill, dtype=torch.float32) if timeout_steps_per_skill is not None else None
    )
    m._pre_kick_reference_blend_steps_per_skill = (
        torch.tensor(reference_blend_steps_per_skill, dtype=torch.float32)
        if reference_blend_steps_per_skill is not None
        else None
    )
    m._kick_pending = torch.zeros(num_envs, dtype=torch.bool)
    m._kick_pending_best_residual = torch.full((num_envs,), float("inf"))
    m._kick_pending_best_frame = torch.full((num_envs,), -1, dtype=torch.long)
    m._pre_kick_fallback_active = torch.zeros(num_envs, dtype=torch.bool)
    m._pre_kick_fallback_start_step = torch.full((num_envs,), -1, dtype=torch.long)
    m._pre_kick_fallback_start_vx = torch.zeros(num_envs)
    m._pre_kick_fallback_stale_ticks = torch.zeros(num_envs, dtype=torch.long)
    m._pre_kick_step = torch.full((num_envs,), -1, dtype=torch.long)
    m._kick_ep_is_handoff = torch.zeros(num_envs, dtype=torch.bool)
    m.command_manager = _FakeCommandManager(motion_command, locomotion_command)
    return m


def test_disabled_prob_is_an_exact_no_op():
    motion_cmd = _FakeMotionCommand([])
    loco_cmd = _FakeLocomotionCommand(2)
    m = _make_manager(2, motion_cmd, loco_cmd, prob=0.0)
    m._kick_pending[:] = True
    before = m._kick_pending.clone()

    m._maybe_enter_kick_from_locomotion()

    assert torch.equal(m._kick_pending, before)
    assert motion_cmd._call == 0, "must not even call search_entry_point when the feature is off"


def test_no_pending_envs_is_a_no_op():
    motion_cmd = _FakeMotionCommand([])
    loco_cmd = _FakeLocomotionCommand(1)
    m = _make_manager(1, motion_cmd, loco_cmd, prob=1.0)

    m._maybe_enter_kick_from_locomotion()

    assert motion_cmd._call == 0


def test_not_yet_min_steps_skips_the_search():
    motion_cmd = _FakeMotionCommand([])
    loco_cmd = _FakeLocomotionCommand(1)
    m = _make_manager(1, motion_cmd, loco_cmd, prob=1.0, min_steps=100)
    m._kick_pending[:] = True
    m.episode_length_buf[:] = 5

    m._maybe_enter_kick_from_locomotion()

    assert motion_cmd._call == 0


def test_first_search_tick_records_best_without_firing():
    motion_cmd = _FakeMotionCommand([torch.tensor([1.0])])
    loco_cmd = _FakeLocomotionCommand(1)
    m = _make_manager(1, motion_cmd, loco_cmd, prob=1.0)
    m._kick_pending[:] = True

    m._maybe_enter_kick_from_locomotion()

    assert int(m.task_mode[0]) == int(TaskMode.LOCOMOTION)
    assert float(m._kick_pending_best_residual[0]) == 1.0
    assert len(motion_cmd.enter_calls) == 0


def test_fires_at_the_turning_point_using_the_best_seen_frame():
    residuals = [torch.tensor([1.0]), torch.tensor([0.5]), torch.tensor([0.8])]
    frames = [torch.tensor([10]), torch.tensor([20]), torch.tensor([30])]
    motion_cmd = _FakeMotionCommand(residuals, frames)
    loco_cmd = _FakeLocomotionCommand(1)
    m = _make_manager(1, motion_cmd, loco_cmd, prob=1.0, max_residual=0.0)
    m._kick_pending[:] = True

    m._maybe_enter_kick_from_locomotion()  # tick1: 1.0, best=1.0, no fire (first ever search)
    m._maybe_enter_kick_from_locomotion()  # tick2: 0.5 improves, best=0.5/frame 20, no fire
    assert int(m.task_mode[0]) == int(TaskMode.LOCOMOTION)

    m._maybe_enter_kick_from_locomotion()  # tick3: 0.8 stops improving -> turning point -> fire

    assert int(m.task_mode[0]) == int(TaskMode.KICK)
    assert len(motion_cmd.enter_calls) == 1
    env_ids, entered_frames = motion_cmd.enter_calls[0]
    assert entered_frames.item() == 20, "must fire at the BEST frame seen, not the current tick's worse one"
    assert len(motion_cmd.ball_calls) == 1
    assert not bool(m._kick_pending[0])
    assert int(m._pre_kick_step[0]) == int(m.episode_length_buf[0])


def test_declines_immediately_when_ceiling_exceeded_and_fallback_disabled():
    residuals = [torch.tensor([5.0]), torch.tensor([6.0])]
    motion_cmd = _FakeMotionCommand(residuals)
    loco_cmd = _FakeLocomotionCommand(1)
    m = _make_manager(1, motion_cmd, loco_cmd, prob=1.0, max_residual=1.0, decel_steps=0.0)
    m._kick_pending[:] = True

    m._maybe_enter_kick_from_locomotion()  # tick1: records 5.0
    m._maybe_enter_kick_from_locomotion()  # tick2: 6.0 -> turning point, exceeds ceiling, no fallback -> decline

    assert int(m.task_mode[0]) == int(TaskMode.LOCOMOTION)
    assert not bool(m._kick_pending[0])
    assert len(motion_cmd.enter_calls) == 0
    assert len(loco_cmd.unpin_calls) == 1


def test_enters_fallback_and_decays_command_toward_target():
    residuals = [torch.tensor([5.0]), torch.tensor([6.0]), torch.tensor([6.0])]
    motion_cmd = _FakeMotionCommand(residuals)
    loco_cmd = _FakeLocomotionCommand(1)
    m = _make_manager(1, motion_cmd, loco_cmd, prob=1.0, max_residual=1.0, decel_steps=10.0, decel_target=0.1)
    m._kick_pending[:] = True
    loco_cmd.commands[0, 0] = 1.0
    m.episode_length_buf[:] = 100

    m._maybe_enter_kick_from_locomotion()  # tick1 (ep=100): records 5.0
    m.episode_length_buf[:] = 101
    m._maybe_enter_kick_from_locomotion()  # tick2 (ep=101): turning point -> fallback starts

    assert bool(m._pre_kick_fallback_active[0])
    assert int(m._pre_kick_fallback_start_step[0]) == 101
    assert float(m._pre_kick_fallback_start_vx[0]) == 1.0
    assert len(loco_cmd.pin_calls) == 0, "the turning-point tick only captures the baseline, doesn't decay yet"

    m.episode_length_buf[:] = 106  # 5/10 decel ticks elapsed -> ratio 0.5
    m._maybe_enter_kick_from_locomotion()  # tick3: still 6.0, continues decaying

    assert len(loco_cmd.pin_calls) == 1
    expected_vx = 1.0 + 0.5 * (0.1 - 1.0)
    assert abs(float(loco_cmd.commands[0, 0]) - expected_vx) < 1e-5
    assert bool(m._kick_pending[0]), "still pending -- hasn't fired or declined yet"


def test_fallback_fires_once_best_residual_clears_the_ceiling():
    residuals = [torch.tensor([5.0]), torch.tensor([6.0]), torch.tensor([0.5])]
    frames = [torch.tensor([1]), torch.tensor([2]), torch.tensor([3])]
    motion_cmd = _FakeMotionCommand(residuals, frames)
    loco_cmd = _FakeLocomotionCommand(1)
    m = _make_manager(1, motion_cmd, loco_cmd, prob=1.0, max_residual=1.0, decel_steps=10.0)
    m._kick_pending[:] = True

    m._maybe_enter_kick_from_locomotion()  # tick1: 5.0
    m._maybe_enter_kick_from_locomotion()  # tick2: turning point -> fallback starts
    assert bool(m._pre_kick_fallback_active[0])

    m._maybe_enter_kick_from_locomotion()  # tick3: 0.5 clears the ceiling -> fires

    assert int(m.task_mode[0]) == int(TaskMode.KICK)
    assert len(motion_cmd.enter_calls) == 1
    assert motion_cmd.enter_calls[0][1].item() == 3
    assert not bool(m._pre_kick_fallback_active[0])
    assert not bool(m._kick_pending[0])


def test_fallback_declines_after_the_non_improving_timeout():
    residuals = [torch.tensor([5.0]), torch.tensor([6.0]), torch.tensor([6.0]), torch.tensor([6.0])]
    motion_cmd = _FakeMotionCommand(residuals)
    loco_cmd = _FakeLocomotionCommand(1)
    m = _make_manager(1, motion_cmd, loco_cmd, prob=1.0, max_residual=1.0, decel_steps=10.0, timeout_steps=1.0)
    m._kick_pending[:] = True

    m._maybe_enter_kick_from_locomotion()  # tick1: 5.0
    m._maybe_enter_kick_from_locomotion()  # tick2: turning point -> fallback starts, stale=0
    m._maybe_enter_kick_from_locomotion()  # tick3: not improved -> stale=1 (not > 1.0 yet)
    assert bool(m._kick_pending[0])

    m._maybe_enter_kick_from_locomotion()  # tick4: not improved -> stale=2 (> 1.0) -> decline

    assert not bool(m._kick_pending[0])
    assert int(m.task_mode[0]) == int(TaskMode.LOCOMOTION)
    assert len(loco_cmd.unpin_calls) == 1
    assert len(motion_cmd.enter_calls) == 0


def test_improving_tick_during_fallback_does_not_consume_the_timeout_budget():
    residuals = [
        torch.tensor([5.0]),
        torch.tensor([6.0]),
        torch.tensor([6.0]),
        torch.tensor([4.0]),  # improves -- must be a free tick, not counted against the timeout
        torch.tensor([6.0]),
        torch.tensor([6.0]),
    ]
    motion_cmd = _FakeMotionCommand(residuals)
    loco_cmd = _FakeLocomotionCommand(1)
    m = _make_manager(1, motion_cmd, loco_cmd, prob=1.0, max_residual=1.0, decel_steps=10.0, timeout_steps=2.0)
    m._kick_pending[:] = True

    for _ in range(4):
        m._maybe_enter_kick_from_locomotion()
    # tick1: 5.0 record; tick2: 6.0 turning point -> fallback (stale=0); tick3: 6.0 not improved
    # (stale=1); tick4: 4.0 improves the running best -> stale stays 1 (free tick).
    assert int(m._pre_kick_fallback_stale_ticks[0]) == 1
    assert bool(m._kick_pending[0])

    m._maybe_enter_kick_from_locomotion()  # tick5: 6.0 not improved (vs best 4.0) -> stale=2 (not > 2.0)
    assert int(m._pre_kick_fallback_stale_ticks[0]) == 2
    assert bool(m._kick_pending[0])

    m._maybe_enter_kick_from_locomotion()  # tick6: 6.0 not improved -> stale=3 (> 2.0) -> decline
    assert not bool(m._kick_pending[0])


def test_env0_fires_env1_declines_independently_in_the_same_tick():
    residuals = [
        torch.tensor([1.0, 5.0]),
        torch.tensor([2.0, 6.0]),
    ]
    frames = [torch.tensor([10, 10]), torch.tensor([20, 20])]
    motion_cmd = _FakeMotionCommand(residuals, frames)
    loco_cmd = _FakeLocomotionCommand(2)
    m = _make_manager(2, motion_cmd, loco_cmd, prob=1.0, max_residual=1.0, decel_steps=0.0)
    m._kick_pending[:] = True

    m._maybe_enter_kick_from_locomotion()  # tick1: records 1.0 (env0), 5.0 (env1)
    m._maybe_enter_kick_from_locomotion()  # tick2: env0 2.0 worsens but within ceiling -> fires;
    # env1 6.0 worsens and exceeds ceiling, no fallback -> declines

    assert int(m.task_mode[0]) == int(TaskMode.KICK)
    assert int(m.task_mode[1]) == int(TaskMode.LOCOMOTION)
    assert not bool(m._kick_pending[0])
    assert not bool(m._kick_pending[1])
    assert len(motion_cmd.enter_calls) == 1
    assert motion_cmd.enter_calls[0][0].item() == 0
    assert len(loco_cmd.unpin_calls) == 1
    assert loco_cmd.unpin_calls[0].item() == 1


def test_fire_does_not_capture_ref_blend_when_disabled():
    """pre_kick_reference_blend_steps defaults to 0.0 (off) -- _enter_kick must not call
    capture_ref_blend at all in that case, matching the exact-no-op discipline every other field
    in this mechanism follows."""
    residuals = [torch.tensor([1.0]), torch.tensor([2.0])]
    motion_cmd = _FakeMotionCommand(residuals)
    loco_cmd = _FakeLocomotionCommand(1)
    m = _make_manager(1, motion_cmd, loco_cmd, prob=1.0, max_residual=0.0, reference_blend_steps=0.0)
    m._kick_pending[:] = True

    m._maybe_enter_kick_from_locomotion()
    m._maybe_enter_kick_from_locomotion()  # fires

    assert int(m.task_mode[0]) == int(TaskMode.KICK)
    assert len(motion_cmd.blend_calls) == 0


def test_fire_captures_ref_blend_with_the_configured_window_when_enabled():
    residuals = [torch.tensor([1.0]), torch.tensor([2.0])]
    motion_cmd = _FakeMotionCommand(residuals)
    loco_cmd = _FakeLocomotionCommand(1)
    m = _make_manager(1, motion_cmd, loco_cmd, prob=1.0, max_residual=0.0, reference_blend_steps=25.0)
    m._kick_pending[:] = True

    m._maybe_enter_kick_from_locomotion()
    m._maybe_enter_kick_from_locomotion()  # fires

    assert int(m.task_mode[0]) == int(TaskMode.KICK)
    assert len(motion_cmd.blend_calls) == 1
    env_ids, blend_steps = motion_cmd.blend_calls[0]
    assert env_ids.tolist() == [0]
    assert torch.allclose(blend_steps, torch.tensor([25.0]))


def test_enter_kick_marks_the_episode_as_handoff_for_the_wandb_stats_split():
    """2026-08-14 (user-requested): is the handoff mechanism learnable, and can we see it in
    wandb? _enter_kick is the ONLY call site that commits a mid-episode handoff -- it must mark
    _kick_ep_is_handoff so _record_kick_episode_ends/_update_log_dict can report handoff-specific
    stats separately from the pooled (mostly reset-triggered) kick_* aggregate."""
    residuals = [torch.tensor([1.0]), torch.tensor([2.0])]
    motion_cmd = _FakeMotionCommand(residuals)
    loco_cmd = _FakeLocomotionCommand(1)
    m = _make_manager(1, motion_cmd, loco_cmd, prob=1.0, max_residual=0.0)
    m._kick_pending[:] = True
    assert not bool(m._kick_ep_is_handoff[0])

    m._maybe_enter_kick_from_locomotion()
    m._maybe_enter_kick_from_locomotion()  # fires

    assert int(m.task_mode[0]) == int(TaskMode.KICK)
    assert bool(m._kick_ep_is_handoff[0])


def test_declined_kick_does_not_mark_the_episode_as_handoff():
    """The mirror baseline: a mid-episode attempt that DECLINES (ceiling exceeded, no fallback)
    never actually entered kick mode -- _kick_ep_is_handoff must stay False, or a later scripted
    kick this same episode would be wrongly folded into the handoff stats."""
    residuals = [torch.tensor([5.0]), torch.tensor([6.0])]
    motion_cmd = _FakeMotionCommand(residuals)
    loco_cmd = _FakeLocomotionCommand(1)
    m = _make_manager(1, motion_cmd, loco_cmd, prob=1.0, max_residual=1.0, decel_steps=0.0)
    m._kick_pending[:] = True

    m._maybe_enter_kick_from_locomotion()
    m._maybe_enter_kick_from_locomotion()  # declines

    assert int(m.task_mode[0]) == int(TaskMode.LOCOMOTION)
    assert not bool(m._kick_ep_is_handoff[0])


# ============================================================================================
# 2026-08-15, Tier 3 Group B Wave 2 ("simultaneous per-skill task configs"): per-skill
# divergence for mid_episode_kick_entry_min_steps/max_residual, pre_kick_decel_steps/target,
# pre_kick_fallback_timeout_steps, pre_kick_reference_blend_steps -- each read from
# UnifiedManager's own _skill_task_config_paths (see __init__'s comment), gathered by env_ids'
# own skill_id at each call site. Every test below pairs two envs on DIFFERENT skills to isolate
# the specific per-env gather being tested, while every test ABOVE this section (unchanged)
# continues to cover the legacy scalar path these per-skill branches fall back to.
# ============================================================================================


def test_per_skill_min_steps_diverges_gates_readiness_independently():
    """env0's skill is immediately ready (min_steps=0), env1's skill needs 100 ticks -- only
    env0 should be included in this tick's `ready` mask / search."""
    residuals = [torch.tensor([5.0])]
    motion_cmd = _FakeMotionCommand(residuals)
    loco_cmd = _FakeLocomotionCommand(2)
    m = _make_manager(
        2, motion_cmd, loco_cmd, prob=1.0, max_residual=0.0, min_steps_per_skill=[0, 100], skill_id=[0, 1]
    )
    m._kick_pending[:] = True
    m.episode_length_buf[:] = 0

    m._maybe_enter_kick_from_locomotion()

    assert float(m._kick_pending_best_residual[0]) == 5.0, "env0 (min_steps=0) was searched"
    assert m._kick_pending_best_residual[1] == float("inf"), "env1 (min_steps=100, not ready) was never searched"


def test_per_skill_ceiling_diverges_one_fires_one_declines():
    """Both envs hit the same worsening residual sequence (turning point at tick2, best=5.0) --
    env0's skill has a wide ceiling (10.0, within) and fires; env1's skill has a narrow one (1.0,
    exceeded) and, with no per-skill fallback configured, declines immediately."""
    residuals = [torch.tensor([5.0, 5.0]), torch.tensor([6.0, 6.0])]
    frames = [torch.tensor([10, 20]), torch.tensor([11, 21])]
    motion_cmd = _FakeMotionCommand(residuals, frames)
    loco_cmd = _FakeLocomotionCommand(2)
    m = _make_manager(2, motion_cmd, loco_cmd, prob=1.0, max_residual_per_skill=[10.0, 1.0], skill_id=[0, 1])
    m._kick_pending[:] = True

    m._maybe_enter_kick_from_locomotion()  # tick1: records 5.0 both
    m._maybe_enter_kick_from_locomotion()  # tick2: turning point both -- env0 within ceiling, env1 not

    assert int(m.task_mode[0]) == int(TaskMode.KICK)
    assert int(m.task_mode[1]) == int(TaskMode.LOCOMOTION)
    assert not bool(m._kick_pending[0])
    assert not bool(m._kick_pending[1])


def test_per_skill_decel_steps_diverges_one_enters_fallback_one_declines_immediately():
    """Both envs exceed the SAME global ceiling (max_residual=1.0) at the same turning point --
    env0's skill wants the decel-and-retry fallback (decel_steps=10.0), env1's doesn't (0.0) and
    declines outright, same as the legacy scalar-off path would."""
    residuals = [torch.tensor([5.0, 5.0]), torch.tensor([6.0, 6.0])]
    motion_cmd = _FakeMotionCommand(residuals)
    loco_cmd = _FakeLocomotionCommand(2)
    m = _make_manager(
        2, motion_cmd, loco_cmd, prob=1.0, max_residual=1.0, decel_steps_per_skill=[10.0, 0.0], skill_id=[0, 1]
    )
    m._kick_pending[:] = True

    m._maybe_enter_kick_from_locomotion()  # tick1: records 5.0 both
    m._maybe_enter_kick_from_locomotion()  # tick2: turning point both, ceiling exceeded both

    assert bool(m._pre_kick_fallback_active[0]), "skill 0 wants the fallback"
    assert bool(m._kick_pending[0]), "still pending -- hasn't fired or declined"
    assert not bool(m._kick_pending[1]), "skill 1 has no fallback -- declines immediately"
    assert int(m.task_mode[1]) == int(TaskMode.LOCOMOTION)
    assert len(loco_cmd.unpin_calls) == 1
    assert loco_cmd.unpin_calls[0].item() == 1


def test_per_skill_decel_target_diverges_for_envs_already_in_fallback():
    """Both envs already in fallback (shared decel_steps=10.0), decaying toward DIFFERENT
    per-skill targets -- isolates the continue_ids-side target_vx gather."""
    residuals = [torch.tensor([5.0, 5.0]), torch.tensor([6.0, 6.0]), torch.tensor([6.0, 6.0])]
    motion_cmd = _FakeMotionCommand(residuals)
    loco_cmd = _FakeLocomotionCommand(2)
    m = _make_manager(
        2,
        motion_cmd,
        loco_cmd,
        prob=1.0,
        max_residual=1.0,
        decel_steps=10.0,
        decel_target_per_skill=[0.1, 0.5],
        skill_id=[0, 1],
    )
    m._kick_pending[:] = True
    loco_cmd.commands[:, 0] = 1.0
    m.episode_length_buf[:] = 100

    m._maybe_enter_kick_from_locomotion()  # tick1 (ep=100): records 5.0 both
    m.episode_length_buf[:] = 101
    m._maybe_enter_kick_from_locomotion()  # tick2 (ep=101): turning point -> fallback starts both

    m.episode_length_buf[:] = 106  # 5/10 decel ticks elapsed -> ratio 0.5
    m._maybe_enter_kick_from_locomotion()  # tick3: continues decaying, per-skill target

    expected_vx0 = 1.0 + 0.5 * (0.1 - 1.0)
    expected_vx1 = 1.0 + 0.5 * (0.5 - 1.0)
    assert abs(float(loco_cmd.commands[0, 0]) - expected_vx0) < 1e-5
    assert abs(float(loco_cmd.commands[1, 0]) - expected_vx1) < 1e-5


def test_per_skill_timeout_diverges_one_times_out_one_keeps_running():
    """Both envs in fallback with an identical non-improving residual sequence -- env0's skill has
    a tight timeout (1.0, times out this tick), env1's has none (0.0, runs indefinitely)."""
    residuals = [
        torch.tensor([5.0, 5.0]),
        torch.tensor([6.0, 6.0]),
        torch.tensor([6.0, 6.0]),
        torch.tensor([6.0, 6.0]),
    ]
    motion_cmd = _FakeMotionCommand(residuals)
    loco_cmd = _FakeLocomotionCommand(2)
    m = _make_manager(
        2,
        motion_cmd,
        loco_cmd,
        prob=1.0,
        max_residual=1.0,
        decel_steps=10.0,
        timeout_steps_per_skill=[1.0, 0.0],
        skill_id=[0, 1],
    )
    m._kick_pending[:] = True

    m._maybe_enter_kick_from_locomotion()  # tick1: 5.0 both
    m._maybe_enter_kick_from_locomotion()  # tick2: turning point -> fallback, stale=0 both
    m._maybe_enter_kick_from_locomotion()  # tick3: stale=1 both (skill0's 1.0 threshold not exceeded yet)
    assert bool(m._kick_pending[0])
    assert bool(m._kick_pending[1])

    m._maybe_enter_kick_from_locomotion()  # tick4: stale=2 -> skill0 (timeout=1.0) times out and declines

    assert not bool(m._kick_pending[0]), "skill 0 timed out"
    assert bool(m._kick_pending[1]), "skill 1 has no timeout -- keeps running"


def test_per_skill_reference_blend_steps_diverges_only_one_env_gets_blend_captured():
    """Both envs fire at the same turning point (no ceiling) -- env0's skill wants the reference
    blend, env1's doesn't. capture_ref_blend must only be called for env0's subset."""
    residuals = [torch.tensor([5.0, 5.0]), torch.tensor([6.0, 6.0])]
    motion_cmd = _FakeMotionCommand(residuals)
    loco_cmd = _FakeLocomotionCommand(2)
    m = _make_manager(
        2, motion_cmd, loco_cmd, prob=1.0, max_residual=0.0, reference_blend_steps_per_skill=[25.0, 0.0], skill_id=[0, 1]
    )
    m._kick_pending[:] = True

    m._maybe_enter_kick_from_locomotion()
    m._maybe_enter_kick_from_locomotion()  # both fire

    assert int(m.task_mode[0]) == int(TaskMode.KICK)
    assert int(m.task_mode[1]) == int(TaskMode.KICK)
    assert len(motion_cmd.blend_calls) == 1
    env_ids, blend_steps = motion_cmd.blend_calls[0]
    assert env_ids.tolist() == [0]
    assert torch.allclose(blend_steps, torch.tensor([25.0]))
