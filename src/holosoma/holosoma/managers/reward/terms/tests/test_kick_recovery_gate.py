"""Unit tests for ``_kick_recovery_gate`` and the ``penalty_kick_recovery_*`` reward term family
(managers/reward/terms/locomotion.py): the kick-mode counterpart to the locomotion
``_standing_gate``/``penalty_stand_*`` family, porting the SAME already-validated standing
posture toolkit into the post-kick recovery/hold tail.

Two things are load-bearing here, both covered below:
1. The gate is ZERO during "swinging mode" (locomotion-approach + strike; never fights legitimate
   single-support lean/asymmetry/height change) and ramps LINEARLY to full strength over
   ``grace_steps`` once the clip enters post-kick-standing mode / the synthetic
   recovery-to-default-pose + hold tail -- mirroring ``_standing_gate``'s own grace-window
   rationale for the identical reason (the robot is still carrying momentum right as recovery
   starts).
2. It is keyed on each env's OWN per-motion ``stand_start_idx``/``motion_ids`` -- i.e. per-skill
   by construction, with no extra wiring needed for any additional skill added to
   ``stageB_and_C.yaml``.

Isolated via lightweight fakes (not a real MotionCommand instance -- constructing one needs a
live env) that provide exactly what the gate reads: ``time_steps``, ``motion_ids``,
``stand_start_idx``, ``has_ball``, and ``in_kicking_phase`` -- the last computed with the EXACT
same expression production uses (``time_steps < stand_start_idx[motion_ids]``, see
``MotionCommand.in_kicking_phase``), so these tests exercise the real phase-boundary semantics,
not a reimplementation of them.

``in_kicking_phase``/``stand_start_idx`` were renamed from ``in_swing_phase``/
``pre_recovery_motion_end_idx`` on 2026-07-31, when the boundary moved from the whole authored
clip's end to just the mode-2 (strike)/mode-3 (post-kick-standing) split -- see
``managers/command/terms/wbt.py``'s ``MotionCommand.setup()``. Most fakes below set
``stand_start_idx`` equal to what ``pre_recovery_motion_end_idx`` would have been (the
legacy/no-scrubbed-boundary case), so existing assertions are unchanged; one new test below
(``test_gate_boundary_is_stand_start_idx_not_pre_recovery_motion_end_idx``) pins down the case
where they differ.
"""

from __future__ import annotations

import torch

from holosoma.managers.reward.terms import locomotion as loco_terms


class _FakeMotionCommand:
    """Provides time_steps/motion_ids/stand_start_idx/has_ball as plain tensors, plus
    in_kicking_phase computed with production's own expression (see module docstring)."""

    def __init__(self, time_steps, motion_ids, stand_start_idx, has_ball=True):
        self.time_steps = time_steps
        self.motion_ids = motion_ids
        self.stand_start_idx = stand_start_idx
        self.has_ball = has_ball

    @property
    def in_kicking_phase(self) -> torch.Tensor:
        return self.time_steps < self.stand_start_idx[self.motion_ids]


class _FakeCommandManager:
    def __init__(self, motion_command):
        self._motion_command = motion_command

    def get_state(self, name):
        return self._motion_command if name == "motion_command" else None


class _FakeEnv:
    def __init__(self, motion_command, num_envs: int, device: str = "cpu"):
        self.command_manager = _FakeCommandManager(motion_command)
        self.num_envs = num_envs
        self.device = device


def _make_env(time_steps, motion_ids=None, stand_start_idx=(100,), has_ball: bool = True) -> _FakeEnv:
    time_steps_t = torch.tensor(time_steps, dtype=torch.long)
    n = len(time_steps_t)
    motion_ids_t = torch.zeros(n, dtype=torch.long) if motion_ids is None else torch.tensor(motion_ids, dtype=torch.long)
    stand_start_idx_t = torch.tensor(stand_start_idx, dtype=torch.long)
    mc = _FakeMotionCommand(time_steps_t, motion_ids_t, stand_start_idx_t, has_ball=has_ball)
    return _FakeEnv(mc, num_envs=n)


def test_zero_during_swing():
    """Well before the recovery boundary -- gate must be exactly 0, never fighting the swing."""
    env = _make_env(time_steps=[0, 50, 99], stand_start_idx=(100,))
    gate = loco_terms._kick_recovery_gate(env, grace_steps=50.0)
    assert torch.allclose(gate, torch.zeros(3))


def test_full_strength_well_past_grace_steps():
    """Deep into the hold, well past the grace window -- gate must be exactly 1."""
    env = _make_env(time_steps=[200, 500], stand_start_idx=(100,))
    gate = loco_terms._kick_recovery_gate(env, grace_steps=50.0)
    assert torch.allclose(gate, torch.ones(2))


def test_ramps_linearly_over_grace_steps():
    """Boundary at 100, grace_steps=50: right at the boundary -> 0, halfway through the grace
    window (step 125) -> 0.5, at the end of the grace window (step 150) -> 1.0."""
    env = _make_env(time_steps=[100, 125, 150], stand_start_idx=(100,))
    gate = loco_terms._kick_recovery_gate(env, grace_steps=50.0)
    assert torch.allclose(gate, torch.tensor([0.0, 0.5, 1.0]), atol=1e-5)


def test_grace_steps_zero_is_hard_step():
    """grace_steps<=0 disables the ramp entirely -- a hard 0/1 step exactly at the boundary,
    matching _standing_gate's own grace_steps=0.0 behavior."""
    env = _make_env(time_steps=[99, 100, 150], stand_start_idx=(100,))
    gate = loco_terms._kick_recovery_gate(env, grace_steps=0.0)
    assert torch.allclose(gate, torch.tensor([0.0, 1.0, 1.0]))


def test_boundary_step_itself_is_already_recovery_not_swing():
    """time_steps == stand_start_idx is already recovery (in_kicking_phase's own '<' is strict)
    -- the ramp value there is 0 (grace window just starting), which is a DIFFERENT thing from
    still being in "swinging mode"; this pins down that distinction explicitly."""
    env = _make_env(time_steps=[100], stand_start_idx=(100,))
    motion_command = env.command_manager.get_state("motion_command")
    assert not bool(motion_command.in_kicking_phase[0])  # already recovery...
    gate = loco_terms._kick_recovery_gate(env, grace_steps=50.0)
    assert gate.item() == 0.0  # ...but the grace ramp hasn't risen yet


def test_per_skill_boundary_independent():
    """Two envs on DIFFERENT skills with different recovery boundaries (skill 0 at 100, skill 1
    at 40), both read at the SAME absolute time_steps=60: env 0 (skill 0) must still be in
    "swinging mode" (gate 0), env 1 (skill 1) must be 20 steps into ITS OWN recovery (gate
    20/50=0.4) -- proving the gate is per-skill by construction, with no per-skill code needed."""
    env = _make_env(time_steps=[60, 60], motion_ids=[0, 1], stand_start_idx=(100, 40))
    gate = loco_terms._kick_recovery_gate(env, grace_steps=50.0)
    assert gate[0].item() == 0.0
    assert torch.isclose(gate[1], torch.tensor(20.0 / 50.0), atol=1e-5)


def test_gate_boundary_is_stand_start_idx_not_pre_recovery_motion_end_idx():
    """N-skill mode with scrubbed strike/stand boundaries: stand_start_idx (mode-2/mode-3 split,
    e.g. 60) is EARLIER than what the whole authored clip's own end would be (e.g. 100, the old
    in_swing_phase/pre_recovery_motion_end_idx boundary) -- this pins down that the gate now
    ramps from the EARLIER boundary (stand_start_idx), not the later whole-clip-end one, proving
    the 2026-07-31 boundary move actually took effect and isn't silently still reading the old
    quantity under a new name."""
    env = _make_env(time_steps=[60, 85, 110], stand_start_idx=(60,))
    gate = loco_terms._kick_recovery_gate(env, grace_steps=50.0)
    # at stand_start_idx=60: step 60 -> ramp start (0.0), step 85 -> 25/50=0.5, step 110 -> 1.0.
    # Were the gate still keyed on a hypothetical pre_recovery_motion_end_idx=100, step 85 would
    # still read as swing (gate 0) instead of 0.5 -- this is the discriminating assertion.
    assert torch.allclose(gate, torch.tensor([0.0, 0.5, 1.0]), atol=1e-5)


def test_no_motion_command_returns_zero():
    """Defensive path: an env class with no motion_command at all (e.g. locomotion-only) must
    never crash -- returns an all-zero tensor, an inert no-op if ever mis-registered there."""

    class _NoMotionCommandEnv:
        def __init__(self):
            self.command_manager = _FakeCommandManager(None)
            self.num_envs = 4
            self.device = "cpu"

    gate = loco_terms._kick_recovery_gate(_NoMotionCommandEnv(), grace_steps=50.0)
    assert torch.allclose(gate, torch.zeros(4))


def test_has_ball_false_returns_zero():
    """Defensive path: has_ball=False (a WBT/UnifiedManager config with no ball at all) must
    also return an inert all-zero tensor, even at time_steps deep into what would otherwise be
    the recovery tail."""
    env = _make_env(time_steps=[200, 200], stand_start_idx=(100,), has_ball=False)
    gate = loco_terms._kick_recovery_gate(env, grace_steps=50.0)
    assert torch.allclose(gate, torch.zeros(2))


# ---------------------------------------------------------------------------------------------
# Refactor-equivalence check: penalty_yaw_drift (locomotion) vs. penalty_kick_recovery_yaw_drift
# (kick) must compute the IDENTICAL underlying error from the shared _yaw_drift_error helper --
# proving the extraction that both now call through didn't change the locomotion term's math.
# ---------------------------------------------------------------------------------------------


class _FakeYawDriftEnv:
    """Minimal env for get_base_ang_vel: base_quat (identity) + simulator.robot_root_states
    with a nonzero angular velocity in columns 10:13."""

    def __init__(self, ang_vel_z: float, num_envs: int = 1):
        self.num_envs = num_envs
        self.device = "cpu"
        self.base_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]] * num_envs)  # identity, xyzw
        root_states = torch.zeros(num_envs, 13)
        root_states[:, 12] = ang_vel_z  # world-frame ang_vel z; identity quat -> same in base frame
        self.simulator = type("_Sim", (), {"robot_root_states": root_states})()
        # command_manager only needed by the *locomotion* gate path (command-gated); zero
        # command so _standing_gate's command term is at full strength (gate ~= 1) for a clean
        # side-by-side comparison against the kick-mode gate forced to 1 via time_steps.
        self.command_manager = _FakeCommandManagerWithCommandsAndMotion(
            commands=torch.zeros(num_envs, 3),
            motion_command=_FakeMotionCommand(
                time_steps=torch.tensor([500] * num_envs),
                motion_ids=torch.zeros(num_envs, dtype=torch.long),
                stand_start_idx=torch.tensor([100]),
            ),
        )


class _FakeCommandManagerWithCommandsAndMotion:
    def __init__(self, commands, motion_command):
        self.commands = commands
        self._motion_command = motion_command

    def get_state(self, name):
        if name == "motion_command":
            return self._motion_command
        if name == "locomotion_gait":
            return None  # grace_steps ramp no-ops without a gait state, matching _standing_gate
        return None


def test_kick_recovery_yaw_drift_matches_locomotion_yaw_drift_error():
    """Same robot state -> both terms' underlying error must match exactly; only the gate
    (command-based vs kick-phase-based) may differ. Both envs here are constructed so their
    respective gate is at full strength (locomotion: zero command, grace_steps=0; kick: deep
    into recovery, grace_steps=0), isolating the comparison to the error math alone."""
    env = _FakeYawDriftEnv(ang_vel_z=0.7)

    loco_reward = loco_terms.penalty_yaw_drift(env, command_gate_sigma=0.1, grace_steps=0.0)
    kick_reward = loco_terms.penalty_kick_recovery_yaw_drift(env, grace_steps=0.0)

    assert torch.allclose(loco_reward, kick_reward, atol=1e-6)
    assert loco_reward.item() > 0.0  # sanity: not a degenerate zero-error comparison
