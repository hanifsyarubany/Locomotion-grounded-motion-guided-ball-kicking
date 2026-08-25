"""Unit tests for the 2026-07-24 frozen/static-sensor mode on ball_pos_b
(managers/observation/terms/unified.py + managers/randomization/terms/locomotion.py::
randomize_ball_obs_freeze): with a per-episode probability, kick_ball_pos_b freezes at a
STATIC reading for the whole episode instead of updating live, modeling a stuck/dead perception
pipeline.

Revised same day as first implemented (user directive): the frozen value is drawn
INDEPENDENTLY of the ball's actual simulated position (nominal + the same OOD-aware noise a
real spawn uses), not a snapshot of the live reading -- so it can land in the OOD region on its
own, not only when the same episode's real ball placement also happens to roll OOD.

Exercises the REAL production ball_pos_b/_draw_independent_frozen_ball_reading functions (not a
reimplementation) against a lightweight fake env with a fake motion_command providing exactly
the per-motion tables the draw needs.
"""

from __future__ import annotations

import torch

from holosoma.managers.observation.terms.unified import ball_pos_b


class _FakeSimulator:
    def __init__(self, robot_root_states: torch.Tensor, all_root_states: torch.Tensor):
        self.robot_root_states = robot_root_states
        self.all_root_states = all_root_states


class _FakeMotionCfg:
    def __init__(self, ood_spawn_probability: float = 0.0, ood_region_multiplier: float = 3.0):
        self.ood_spawn_probability = ood_spawn_probability
        self.ood_region_multiplier = ood_region_multiplier


class _FakeMotionCommand:
    """Provides exactly what _draw_independent_frozen_ball_reading reads: motion_ids,
    ball_reset_state_per_motion, ball_position_randomization_per_motion, motion_cfg."""

    def __init__(
        self,
        motion_ids: torch.Tensor,
        ball_reset_state_per_motion: torch.Tensor,
        ball_position_randomization_per_motion: torch.Tensor,
        ood_spawn_probability: float = 0.0,
        ood_region_multiplier: float = 3.0,
    ):
        self.motion_ids = motion_ids
        self.ball_reset_state_per_motion = ball_reset_state_per_motion
        self.ball_position_randomization_per_motion = ball_position_randomization_per_motion
        self.motion_cfg = _FakeMotionCfg(ood_spawn_probability, ood_region_multiplier)
        self.has_ball = True


class _FakeCommandManager:
    def __init__(self, motion_command):
        self._motion_command = motion_command

    def get_state(self, name):
        return self._motion_command if name == "motion_command" else None


class _FakeEnv:
    def __init__(self, robot_pos: torch.Tensor, ball_pos: torch.Tensor, motion_command=None):
        n = robot_pos.shape[0]
        robot_root_states = torch.zeros(n, 13)
        robot_root_states[:, :3] = robot_pos
        self.simulator = _FakeSimulator(robot_root_states, ball_pos)
        self.base_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]] * n)  # identity, xyzw
        self._ball_obs_actor_indices = torch.arange(n)
        self.num_envs = n
        self.device = "cpu"
        self.command_manager = _FakeCommandManager(motion_command)


def _make_motion_command(n: int, nominal_xy=(2.0, 0.5), randomize=(0.75, 0.75), ood_prob=0.0, ood_mult=3.0):
    return _FakeMotionCommand(
        motion_ids=torch.zeros(n, dtype=torch.long),
        ball_reset_state_per_motion=torch.tensor([[*nominal_xy]]),  # (num_motions=1, 2)
        ball_position_randomization_per_motion=torch.tensor([[*randomize]]),
        ood_spawn_probability=ood_prob,
        ood_region_multiplier=ood_mult,
    )


def test_no_frozen_mask_set_behaves_exactly_as_before():
    """env._ball_obs_frozen_mask absent entirely (every existing config) must be a complete
    no-op -- ball_pos_b keeps reading the live value every call, and never touches
    command_manager at all (no motion_command needed)."""
    env = _FakeEnv(torch.zeros(1, 3), torch.tensor([[1.0, 0.0, 0.11]]))
    first = ball_pos_b(env)
    env.simulator.all_root_states = torch.tensor([[2.0, 0.0, 0.11]])
    second = ball_pos_b(env)
    assert not torch.allclose(first, second), "should track the live (changed) ball position"


def test_unfrozen_env_in_a_batch_stays_live():
    """A batch with SOME envs frozen and others not: the not-frozen envs must keep tracking live
    values, unaffected by their neighbors' frozen state."""
    mc = _make_motion_command(2)
    env = _FakeEnv(torch.zeros(2, 3), torch.tensor([[1.0, 0.0, 0.11], [1.0, 0.0, 0.11]]), motion_command=mc)
    env._ball_obs_frozen_mask = torch.tensor([True, False])
    env._ball_obs_frozen_captured = torch.tensor([False, False])
    env._ball_obs_frozen_value = torch.zeros(2, 3)

    ball_pos_b(env)  # capture step for env 0
    env.simulator.all_root_states = torch.tensor([[1.0, 0.0, 0.11], [5.0, 0.0, 0.11]])
    out = ball_pos_b(env)

    assert not torch.allclose(out[0], torch.tensor([5.0, 0.0, 0.11]))  # frozen: not tracking live
    assert torch.allclose(out[1], torch.tensor([5.0, 0.0, 0.11]))  # live: tracks the new value


def test_frozen_env_draws_once_then_holds_fixed_across_many_steps():
    """The decisive check: a frozen env's value is drawn ONCE, and every subsequent call -- even
    as the true ball position keeps changing -- returns that same first drawn value."""
    mc = _make_motion_command(1)
    env = _FakeEnv(torch.zeros(1, 3), torch.tensor([[1.0, 0.0, 0.11]]), motion_command=mc)
    env._ball_obs_frozen_mask = torch.tensor([True])
    env._ball_obs_frozen_captured = torch.tensor([False])
    env._ball_obs_frozen_value = torch.zeros(1, 3)

    first = ball_pos_b(env).clone()

    for fake_pos in ([2.0, 0.5, 0.11], [-3.0, 1.0, 0.11], [0.0, 0.0, 0.11]):
        env.simulator.all_root_states = torch.tensor([fake_pos])
        out = ball_pos_b(env)
        assert torch.allclose(out, first), "frozen value must never change once drawn"

    assert bool(env._ball_obs_frozen_captured[0])


def test_frozen_value_is_independent_of_the_true_ball_position():
    """THE decisive check for this session's fix: the frozen value must be drawn from the
    nominal+noise distribution, NOT the ball's actual (here, wildly different) simulated
    position -- i.e. genuinely decoupled from ground truth, not a snapshot of it."""
    torch.manual_seed(0)
    mc = _make_motion_command(1, nominal_xy=(2.0, 0.5), randomize=(0.1, 0.1), ood_prob=0.0)
    # True ball position is FAR from the nominal (2.0, 0.5) -- if the frozen value were a live
    # snapshot (the old, superseded behavior), it would reflect this instead.
    env = _FakeEnv(torch.zeros(1, 3), torch.tensor([[50.0, 50.0, 0.11]]), motion_command=mc)
    env._ball_obs_frozen_mask = torch.tensor([True])
    env._ball_obs_frozen_captured = torch.tensor([False])
    env._ball_obs_frozen_value = torch.zeros(1, 3)

    out = ball_pos_b(env)
    # Must be close to the NOMINAL (2.0, 0.5) +/- 0.1, not anywhere near the true (50, 50).
    assert (out[0, 0] - 2.0).abs() < 0.1 + 1e-4
    assert (out[0, 1] - 0.5).abs() < 0.1 + 1e-4
    assert torch.norm(out[0, :2] - torch.tensor([50.0, 50.0])) > 40.0


def test_frozen_value_can_independently_land_in_the_ood_region():
    """With ood_prob=1.0 on the motion_cfg, the frozen draw must land in the WIDER OOD region --
    exceeding the normal randomize_x/y half-range -- decoupled from the true ball position
    entirely (set to exactly the nominal position here, so a normal-region draw would be
    indistinguishable from ground truth; only an OOD draw proves the mechanism is really firing)."""
    torch.manual_seed(0)
    nominal_xy = (2.0, 0.5)
    randomize = (0.1, 0.1)
    mc = _make_motion_command(1, nominal_xy=nominal_xy, randomize=randomize, ood_prob=1.0, ood_mult=5.0)
    env = _FakeEnv(torch.zeros(1, 3), torch.tensor([[*nominal_xy, 0.11]]), motion_command=mc)
    env._ball_obs_frozen_mask = torch.tensor([True])
    env._ball_obs_frozen_captured = torch.tensor([False])
    env._ball_obs_frozen_value = torch.zeros(1, 3)

    out = ball_pos_b(env)
    dx = (out[0, 0] - nominal_xy[0]).abs().item()
    dy = (out[0, 1] - nominal_xy[1]).abs().item()
    # Must exceed the NORMAL half-range on at least one axis (proving OOD fired), while staying
    # within the OOD bound (randomize * ood_multiplier) on both.
    assert dx > randomize[0] + 1e-4 or dy > randomize[1] + 1e-4
    assert dx <= randomize[0] * 5.0 + 1e-4
    assert dy <= randomize[1] * 5.0 + 1e-4


def test_reset_hook_marks_freeze_and_resets_captured_flag():
    from holosoma.managers.randomization.terms.locomotion import randomize_ball_obs_freeze

    env = _FakeEnv(torch.zeros(3, 3), torch.zeros(3, 3))
    env._ball_static_obs_probability = 1.0  # force-cache: always freeze
    env_ids = torch.tensor([0, 1, 2])

    randomize_ball_obs_freeze(env, env_ids)
    assert torch.all(env._ball_obs_frozen_mask)
    assert torch.all(~env._ball_obs_frozen_captured)


def test_reset_hook_probability_zero_never_freezes():
    from holosoma.managers.randomization.terms.locomotion import randomize_ball_obs_freeze

    env = _FakeEnv(torch.zeros(50, 3), torch.zeros(50, 3))
    env._ball_static_obs_probability = 0.0
    env_ids = torch.arange(50)

    randomize_ball_obs_freeze(env, env_ids)
    assert not bool(env._ball_obs_frozen_mask.any())


def test_reset_hook_reselects_freeze_state_on_each_call():
    """A subsequent reset must be free to CHANGE an env's freeze state (on -> off or vice versa)
    -- freeze is a per-EPISODE draw, not a permanent per-env property."""
    from holosoma.managers.randomization.terms.locomotion import randomize_ball_obs_freeze

    env = _FakeEnv(torch.zeros(1, 3), torch.zeros(1, 3))
    env_ids = torch.tensor([0])

    env._ball_static_obs_probability = 1.0
    randomize_ball_obs_freeze(env, env_ids)
    assert bool(env._ball_obs_frozen_mask[0])

    env._ball_static_obs_probability = 0.0
    randomize_ball_obs_freeze(env, env_ids)
    assert not bool(env._ball_obs_frozen_mask[0])
