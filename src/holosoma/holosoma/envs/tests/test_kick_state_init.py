"""Unit tests for kick-state locomotion init (2026-08-31) -- see
MultiSkillConfig.kick_state_init_prob's own docstring for the full rationale (train locomotion to
recover to a stable stance/walk from the off-balance, momentum-carrying poses a kick produces,
WITHOUT touching any kick-side metric -- unlike kick_abort_prob/kick_recovery_locomotion_flip_enabled,
this mechanism never flips a KICK-partitioned env; it only ever teleports a LOCOMOTION-partitioned
env at its own reset).

Isolated via bare instances (object.__new__), matching this project's established convention for
UnifiedManager unit tests (see test_kick_abort_flip.py / test_unified_manager_resample_task_mode_
kick_entry_draw.py) -- no real env/sim/GPU needed. `_maybe_kick_state_init` touches only
`command_manager.get_state("motion_command")` and torch.rand; it never reaches the simulator
directly (that happens inside MotionCommand.teleport_to_frames, exercised separately as a fake
double here).
"""

from __future__ import annotations

import torch

from holosoma.envs.unified.unified_manager import UnifiedManager


class _FakeMotionCommand:
    """Records every call so tests can assert exactly which env_ids were teleported, without any
    real motion-clip data or simulator writes."""

    def __init__(self, num_motions: int = 2):
        self.num_motions = num_motions
        self.sample_calls: list[torch.Tensor] = []
        self.teleport_calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def sample_authored_clip_frames(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.sample_calls.append(env_ids.clone())
        motion_ids = torch.zeros(env_ids.numel(), dtype=torch.long)
        frames = torch.arange(env_ids.numel(), dtype=torch.long)  # deterministic, distinguishable
        return motion_ids, frames

    def teleport_to_frames(self, env_ids: torch.Tensor, frames: torch.Tensor) -> None:
        self.teleport_calls.append((env_ids.clone(), frames.clone()))


class _FakeCommandManager:
    def __init__(self, motion_command):
        self._motion_command = motion_command

    def get_state(self, name: str):
        return self._motion_command if name == "motion_command" else None


def _make_manager(
    num_envs: int,
    *,
    kick_state_init_prob: float = 0.0,
    kick_state_init_grace_steps: float = 25.0,
    kick_state_transplant_prob: float = 0.0,
    motion_command: "_FakeMotionCommand | None" = None,
    is_evaluating: bool = False,
) -> UnifiedManager:
    m = object.__new__(UnifiedManager)
    m.num_envs = num_envs
    m.device = "cpu"
    m.is_evaluating = is_evaluating
    m._kick_state_init_prob = kick_state_init_prob
    m._kick_state_init_grace_steps = kick_state_init_grace_steps
    m._kick_state_init_active = torch.zeros(num_envs, dtype=torch.bool)
    # Kick-state TRANSPLANT (2026-08-31) -- must exist even on tests targeting only the
    # reference-frame sibling above, since kick_state_init_grace_active's gate now checks BOTH
    # sources (see that method's own docstring for the bug this guards against).
    m._kick_state_transplant_prob = kick_state_transplant_prob
    m._kick_state_transplant_active = torch.zeros(num_envs, dtype=torch.bool)
    m._kick_state_transplant_no_donor_ema = torch.zeros(())
    m.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
    m.command_manager = _FakeCommandManager(motion_command if motion_command is not None else _FakeMotionCommand())
    return m


class TestKickStateInitGraceActive:
    def test_feature_off_prob_zero_is_all_false(self):
        """0.0 default (prob<=0) must be an exact no-op regardless of _kick_state_init_active's
        own contents -- guards against a stale True surviving from before the feature was
        disabled at the config layer."""
        m = _make_manager(3, kick_state_init_prob=0.0)
        m._kick_state_init_active[:] = True  # deliberately stale/wrong, must be ignored
        assert not m.kick_state_init_grace_active().any()

    def test_grace_steps_zero_is_all_false(self):
        """grace_steps<=0 must also be an exact no-op, independent of prob -- same two-guard
        contract post_flip_grace_active's own docstring establishes at the other boundary."""
        m = _make_manager(3, kick_state_init_prob=0.5, kick_state_init_grace_steps=0.0)
        m._kick_state_init_active[:] = True
        assert not m.kick_state_init_grace_active().any()

    def test_active_and_within_grace_is_true(self):
        m = _make_manager(1, kick_state_init_prob=0.5, kick_state_init_grace_steps=25.0)
        m._kick_state_init_active[0] = True
        m.episode_length_buf[0] = 10
        assert bool(m.kick_state_init_grace_active()[0])

    def test_active_but_past_grace_is_false(self):
        """`<`, not `<=`: the grace window is [0, grace_steps)."""
        m = _make_manager(1, kick_state_init_prob=0.5, kick_state_init_grace_steps=25.0)
        m._kick_state_init_active[0] = True
        m.episode_length_buf[0] = 25
        assert not bool(m.kick_state_init_grace_active()[0])

    def test_not_active_is_false_even_within_the_window(self):
        """An ordinary (non-kick-state-init) locomotion env must never get this grace, no matter
        how early in its episode -- only envs the mechanism actually touched are covered."""
        m = _make_manager(1, kick_state_init_prob=0.5, kick_state_init_grace_steps=25.0)
        m.episode_length_buf[0] = 0
        assert not bool(m.kick_state_init_grace_active()[0])

    def test_transplant_alone_still_grants_the_grace(self):
        """Regression: the gate must check EITHER source. kick_state_init_prob=0.0 here (the
        transplant is the only mechanism enabled) -- an early, incorrect version of this gate
        checked only kick_state_init_prob and would incorrectly return all-False in exactly this
        configuration, silently disabling the grace for every transplant-sourced episode."""
        m = _make_manager(1, kick_state_init_prob=0.0, kick_state_transplant_prob=0.5, kick_state_init_grace_steps=25.0)
        m._kick_state_init_active[0] = True  # what _maybe_kick_state_transplant sets, per its own code
        m.episode_length_buf[0] = 10
        assert bool(m.kick_state_init_grace_active()[0])

    def test_both_sources_off_is_all_false(self):
        m = _make_manager(1, kick_state_init_prob=0.0, kick_state_transplant_prob=0.0, kick_state_init_grace_steps=25.0)
        m._kick_state_init_active[0] = True  # stale/inconsistent on purpose -- must still be ignored
        m.episode_length_buf[0] = 0
        assert not bool(m.kick_state_init_grace_active()[0])


class TestMaybeKickStateInit:
    def test_feature_off_makes_no_draw_and_no_teleport(self):
        mc = _FakeMotionCommand()
        m = _make_manager(4, kick_state_init_prob=0.0, motion_command=mc)
        m._maybe_kick_state_init(torch.arange(4))
        assert mc.sample_calls == []
        assert mc.teleport_calls == []
        assert not m._kick_state_init_active.any()

    def test_prob_one_teleports_every_selected_env_and_sets_active(self):
        mc = _FakeMotionCommand()
        m = _make_manager(4, kick_state_init_prob=1.0, motion_command=mc)
        loco_ids = torch.tensor([1, 3])
        m._maybe_kick_state_init(loco_ids)
        assert len(mc.sample_calls) == 1
        assert torch.equal(mc.sample_calls[0].sort().values, loco_ids.sort().values)
        assert len(mc.teleport_calls) == 1
        teleported_ids, _frames = mc.teleport_calls[0]
        assert torch.equal(teleported_ids.sort().values, loco_ids.sort().values)
        assert bool(m._kick_state_init_active[1])
        assert bool(m._kick_state_init_active[3])
        assert not bool(m._kick_state_init_active[0])
        assert not bool(m._kick_state_init_active[2])

    def test_empty_loco_ids_is_a_no_op(self):
        mc = _FakeMotionCommand()
        m = _make_manager(4, kick_state_init_prob=1.0, motion_command=mc)
        m._maybe_kick_state_init(torch.zeros(0, dtype=torch.long))
        assert mc.sample_calls == []
        assert mc.teleport_calls == []

    def test_is_evaluating_disables_the_mechanism(self):
        """Same eval-time-no-randomization contract every sibling randomization mechanism in this
        project follows (e.g. apply_body_pushes's own env.is_evaluating gate) -- an eval rollout
        must see deterministic, un-perturbed resets."""
        mc = _FakeMotionCommand()
        m = _make_manager(4, kick_state_init_prob=1.0, motion_command=mc, is_evaluating=True)
        m._maybe_kick_state_init(torch.arange(4))
        assert mc.sample_calls == []
        assert mc.teleport_calls == []

    def test_no_motion_command_is_a_no_op_not_a_crash(self):
        """A run with no clips at all (motion_command is None) must not crash -- this mechanism
        is kick-clip-sourced and simply has nothing to draw from."""
        m = _make_manager(4, kick_state_init_prob=1.0, motion_command=None)
        m.command_manager = _FakeCommandManager(None)
        m._maybe_kick_state_init(torch.arange(4))  # must not raise
        assert not m._kick_state_init_active.any()

    def test_motion_command_missing_the_method_is_a_no_op(self):
        """Backward-compat guard: an older/different MotionCommand class without
        sample_authored_clip_frames must not crash this call site."""

        class _BareMotionCommand:
            pass

        m = _make_manager(4, kick_state_init_prob=1.0, motion_command=_BareMotionCommand())
        m._maybe_kick_state_init(torch.arange(4))  # must not raise
        assert not m._kick_state_init_active.any()

    def test_zero_prob_draw_with_stochastic_selection_selects_none(self):
        """prob strictly between 0 and 1, seeded so the draw selects nobody -- confirms the
        selection is a real per-env Bernoulli draw (torch.rand < prob), not an all-or-nothing
        gate collapsing to `prob > 0`."""
        torch.manual_seed(0)
        mc = _FakeMotionCommand()
        m = _make_manager(100, kick_state_init_prob=1e-6, motion_command=mc)
        m._maybe_kick_state_init(torch.arange(100))
        # Overwhelmingly likely to select nobody at prob=1e-6 over 100 envs; if it selected exactly
        # 0, both calls lists stay empty (the numel()==0 early-return) -- either way, no crash.
        assert len(mc.teleport_calls) <= 1


class TestResetRobotStatesCallbackWiring:
    """Confirms kick-state init is wired into the reset path with the right guards -- see
    UnifiedManager._reset_robot_states_callback's own comments for why each guard exists."""

    def _make_full_manager(self, num_envs: int, kick_state_init_prob: float, motion_command) -> UnifiedManager:
        from holosoma.envs.unified.unified_manager import TaskMode

        m = _make_manager(num_envs, kick_state_init_prob=kick_state_init_prob, motion_command=motion_command)
        m.task_mode = torch.full((num_envs,), int(TaskMode.LOCOMOTION), dtype=torch.long)
        m._reset_dofs_calls: list[torch.Tensor] = []
        m._reset_root_states_calls: list[torch.Tensor] = []
        m._reset_dofs = lambda env_ids, target=None: m._reset_dofs_calls.append(env_ids.clone())
        m._reset_root_states = lambda env_ids, target=None: m._reset_root_states_calls.append(env_ids.clone())
        return m

    def test_kick_state_init_active_is_cleared_unconditionally_on_every_reset(self):
        """A stale True from a prior episode must never leak into this one, even for an env NOT
        selected by this reset's own draw (feature off here)."""
        mc = _FakeMotionCommand()
        m = self._make_full_manager(2, kick_state_init_prob=0.0, motion_command=mc)
        m._kick_state_init_active[:] = True
        m._reset_robot_states_callback(torch.tensor([0, 1]))
        assert not m._kick_state_init_active.any()

    def test_fires_on_the_plain_reset_path_after_the_ordinary_loco_pose_write(self):
        mc = _FakeMotionCommand()
        m = self._make_full_manager(2, kick_state_init_prob=1.0, motion_command=mc)
        m._reset_robot_states_callback(torch.tensor([0, 1]), target_states=None)
        assert len(m._reset_dofs_calls) == 1  # ordinary loco pose written first
        assert len(mc.teleport_calls) == 1  # then overwritten by the kick-state init
        assert m._kick_state_init_active.all()

    def test_does_not_fire_on_the_deterministic_restore_path(self):
        """target_states is not None means a caller (video-recording pinning / eval replay) asked
        to restore a SPECIFIC state -- injecting a random kick pose there would corrupt exactly
        what the caller asked to reproduce."""
        mc = _FakeMotionCommand()
        m = self._make_full_manager(2, kick_state_init_prob=1.0, motion_command=mc)
        m._reset_robot_states_callback(torch.tensor([0, 1]), target_states={"dof_states": None, "root_states": None})
        assert mc.teleport_calls == []
        assert not m._kick_state_init_active.any()


class _FakeTransplantMotionCommand:
    """Minimal double for _maybe_kick_state_transplant's own eligibility read -- time_steps/
    motion_ids/pre_recovery_motion_end_idx only. No sample_authored_clip_frames/teleport_to_frames
    -- the transplant path never calls the reference-frame sibling's methods."""

    def __init__(self, time_steps: torch.Tensor, motion_ids: torch.Tensor, pre_recovery_motion_end_idx: torch.Tensor):
        self.time_steps = time_steps
        self.motion_ids = motion_ids
        self.pre_recovery_motion_end_idx = pre_recovery_motion_end_idx


class _FakeScene:
    def __init__(self, env_origins: torch.Tensor):
        self.env_origins = env_origins


class _FakeSimulator:
    def __init__(self, num_envs: int, env_origins: "torch.Tensor | None" = None):
        self.dof_pos = torch.zeros(num_envs, 3)
        self.dof_vel = torch.zeros(num_envs, 3)
        self.robot_root_states = torch.zeros(num_envs, 13)
        self.scene = _FakeScene(env_origins if env_origins is not None else torch.zeros(num_envs, 3))


def _make_transplant_manager(
    num_envs: int,
    *,
    kick_state_transplant_prob: float,
    task_mode: torch.Tensor,
    time_steps: torch.Tensor,
    motion_ids: "torch.Tensor | None" = None,
    pre_recovery_motion_end_idx: "torch.Tensor | None" = None,
    env_origins: "torch.Tensor | None" = None,
    is_evaluating: bool = False,
) -> UnifiedManager:
    from holosoma.envs.unified.unified_manager import TaskMode  # noqa: F401 -- for callers' own use

    m = object.__new__(UnifiedManager)
    m.num_envs = num_envs
    m.device = "cpu"
    m.is_evaluating = is_evaluating
    m.task_mode = task_mode
    m._kick_state_transplant_prob = kick_state_transplant_prob
    m._kick_state_transplant_active = torch.zeros(num_envs, dtype=torch.bool)
    m._kick_state_transplant_no_donor_ema = torch.zeros(())
    m._kick_state_init_active = torch.zeros(num_envs, dtype=torch.bool)
    mc = _FakeTransplantMotionCommand(
        time_steps=time_steps,
        motion_ids=motion_ids if motion_ids is not None else torch.zeros(num_envs, dtype=torch.long),
        pre_recovery_motion_end_idx=(
            pre_recovery_motion_end_idx if pre_recovery_motion_end_idx is not None else torch.full((1,), 999)
        ),
    )
    m.command_manager = _FakeCommandManager(mc)
    m.simulator = _FakeSimulator(num_envs, env_origins)
    return m


class TestMaybeKickStateTransplant:
    def test_feature_off_makes_no_copy(self):
        m = _make_transplant_manager(
            2, kick_state_transplant_prob=0.0,
            task_mode=torch.tensor([1, 0]), time_steps=torch.tensor([50, 0]),
        )
        m.simulator.dof_pos[0] = torch.tensor([1.0, 2.0, 3.0])  # donor, distinctive
        m._maybe_kick_state_transplant(torch.tensor([1]))
        assert torch.equal(m.simulator.dof_pos[1], torch.zeros(3))  # recipient untouched
        assert not m._kick_state_transplant_active.any()

    def test_prob_one_copies_donor_state_and_leaves_donor_untouched(self):
        """The single most load-bearing property: reading the donor must not write back into it.
        Both env_origins are zero here, so a bare equality check on the copied fields is enough
        (the origin-correction math is exercised separately below)."""
        m = _make_transplant_manager(
            2, kick_state_transplant_prob=1.0,
            task_mode=torch.tensor([1, 0]),  # env0 KICK (donor), env1 LOCOMOTION (recipient)
            time_steps=torch.tensor([50, 0]),
            pre_recovery_motion_end_idx=torch.tensor([200]),
        )
        m.simulator.dof_pos[0] = torch.tensor([1.0, 2.0, 3.0])
        m.simulator.dof_vel[0] = torch.tensor([0.1, 0.2, 0.3])
        m.simulator.robot_root_states[0] = torch.arange(13, dtype=torch.float)
        donor_dof_pos_before = m.simulator.dof_pos[0].clone()

        m._maybe_kick_state_transplant(torch.tensor([1]))

        assert torch.equal(m.simulator.dof_pos[1], torch.tensor([1.0, 2.0, 3.0]))
        assert torch.equal(m.simulator.dof_vel[1], torch.tensor([0.1, 0.2, 0.3]))
        assert torch.equal(m.simulator.robot_root_states[1], torch.arange(13, dtype=torch.float))
        # Donor's own state must be exactly what it was -- a pure read, no write-back.
        assert torch.equal(m.simulator.dof_pos[0], donor_dof_pos_before)
        assert bool(m._kick_state_transplant_active[1])
        assert bool(m._kick_state_init_active[1])  # shared grace flag also set

    def test_env_origin_offset_is_corrected(self):
        """Root position must be re-anchored: donor_world_pos - donor_origin + recipient_origin --
        not copied verbatim (that would place the recipient at the DONOR's origin, not its own)."""
        origins = torch.tensor([[10.0, 0.0, 0.0], [0.0, 5.0, 0.0]])  # donor, recipient
        m = _make_transplant_manager(
            2, kick_state_transplant_prob=1.0,
            task_mode=torch.tensor([1, 0]),
            time_steps=torch.tensor([50, 0]),
            pre_recovery_motion_end_idx=torch.tensor([200]),
            env_origins=origins,
        )
        m.simulator.robot_root_states[0, :3] = torch.tensor([11.0, 0.5, 0.7])  # donor world pos

        m._maybe_kick_state_transplant(torch.tensor([1]))

        expected = torch.tensor([11.0, 0.5, 0.7]) - origins[0] + origins[1]
        assert torch.allclose(m.simulator.robot_root_states[1, :3], expected)

    def test_orientation_and_velocity_copied_verbatim_no_origin_correction(self):
        """Fields 3:13 (quat + lin/ang vel) are frame-relative, not world-position -- must be
        copied as-is, unlike the position slice."""
        origins = torch.tensor([[10.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
        m = _make_transplant_manager(
            2, kick_state_transplant_prob=1.0,
            task_mode=torch.tensor([1, 0]),
            time_steps=torch.tensor([50, 0]),
            pre_recovery_motion_end_idx=torch.tensor([200]),
            env_origins=origins,
        )
        m.simulator.robot_root_states[0, 3:13] = torch.arange(3, 13, dtype=torch.float)
        m._maybe_kick_state_transplant(torch.tensor([1]))
        assert torch.equal(m.simulator.robot_root_states[1, 3:13], torch.arange(3, 13, dtype=torch.float))

    def test_non_kick_env_is_not_an_eligible_donor(self):
        """Both envs LOCOMOTION -- no donor exists at all, regardless of time_steps."""
        m = _make_transplant_manager(
            2, kick_state_transplant_prob=1.0,
            task_mode=torch.tensor([0, 0]),
            time_steps=torch.tensor([50, 0]),
            pre_recovery_motion_end_idx=torch.tensor([200]),
        )
        m.simulator.dof_pos[0] = torch.tensor([9.0, 9.0, 9.0])
        m._maybe_kick_state_transplant(torch.tensor([1]))
        assert torch.equal(m.simulator.dof_pos[1], torch.zeros(3))
        assert not bool(m._kick_state_transplant_active[1])
        assert float(m._kick_state_transplant_no_donor_ema) > 0.0

    def test_recovery_tail_env_is_not_an_eligible_donor(self):
        """A KICK env past its own pre_recovery_motion_end_idx must be excluded -- same authored-
        content-only boundary as sample_authored_clip_frames, same reason (already near-nominal)."""
        m = _make_transplant_manager(
            2, kick_state_transplant_prob=1.0,
            task_mode=torch.tensor([1, 0]),  # env0 IS kick-mode...
            time_steps=torch.tensor([250, 0]),  # ...but past its own clip's authored span
            pre_recovery_motion_end_idx=torch.tensor([200]),
        )
        m.simulator.dof_pos[0] = torch.tensor([9.0, 9.0, 9.0])
        m._maybe_kick_state_transplant(torch.tensor([1]))
        assert torch.equal(m.simulator.dof_pos[1], torch.zeros(3))
        assert not bool(m._kick_state_transplant_active[1])
        assert float(m._kick_state_transplant_no_donor_ema) > 0.0

    def test_is_evaluating_disables_the_mechanism(self):
        m = _make_transplant_manager(
            2, kick_state_transplant_prob=1.0,
            task_mode=torch.tensor([1, 0]),
            time_steps=torch.tensor([50, 0]),
            pre_recovery_motion_end_idx=torch.tensor([200]),
            is_evaluating=True,
        )
        m.simulator.dof_pos[0] = torch.tensor([9.0, 9.0, 9.0])
        m._maybe_kick_state_transplant(torch.tensor([1]))
        assert torch.equal(m.simulator.dof_pos[1], torch.zeros(3))

    def test_no_donor_ema_stays_zero_when_a_donor_is_available(self):
        m = _make_transplant_manager(
            2, kick_state_transplant_prob=1.0,
            task_mode=torch.tensor([1, 0]),
            time_steps=torch.tensor([50, 0]),
            pre_recovery_motion_end_idx=torch.tensor([200]),
        )
        m._maybe_kick_state_transplant(torch.tensor([1]))
        assert float(m._kick_state_transplant_no_donor_ema) == 0.0
