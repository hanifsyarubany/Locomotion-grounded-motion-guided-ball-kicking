"""Unit tests for LocomotionCommand.pin_zero (2026-08-09, Stage D's post-swing -> locomotion
handoff, kick_recovery_locomotion_flip_enabled). step()'s periodic resample is unconditional --
it runs for every env regardless of task_mode (see step()'s own comment) -- so once a kick-mode
env's task_mode flips to LOCOMOTION mid-episode, its command buffer becomes live-relevant for the
first time, and needs an explicit way to be forced to a deterministic value and held there,
instead of drifting to whatever this term's own periodic resample next draws.
"""

from types import SimpleNamespace

import torch

from holosoma.managers.command.terms.locomotion import LocomotionCommand


def _make_term(num_envs: int, resampling_time: float = 1.0, dt: float = 0.02) -> LocomotionCommand:
    env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        dt=dt,
        is_evaluating=False,
        episode_length_buf=torch.zeros(num_envs, dtype=torch.long),
    )
    cfg = SimpleNamespace(
        params={"command_ranges": {"lin_vel_x": (-1.0, 1.0), "lin_vel_y": (-1.0, 1.0), "ang_vel_yaw": (-1.0, 1.0)}}
    )
    term = LocomotionCommand(cfg, env)
    term.manager = SimpleNamespace(
        command_cfg=SimpleNamespace(locomotion_command_resampling_time=resampling_time),
        get_state=lambda name: None,
    )
    term.setup()
    return term


def test_pin_zero_forces_exact_zero():
    term = _make_term(num_envs=4)
    torch.manual_seed(0)
    term.reset(None)  # give every env a real (likely nonzero) draw first
    assert not torch.all(term.commands == 0.0), "test setup should start from a nonzero draw"

    term.pin_zero(torch.tensor([1, 3]))

    assert torch.all(term.commands[1] == 0.0)
    assert torch.all(term.commands[3] == 0.0)
    assert bool(term._pinned_zero[1])
    assert bool(term._pinned_zero[3])
    assert not bool(term._pinned_zero[0])
    assert not bool(term._pinned_zero[2])


def test_step_never_resamples_a_pinned_env():
    term = _make_term(num_envs=6, resampling_time=1.0, dt=0.02)
    term.reset(None)
    term.pin_zero(torch.arange(6))  # pin every env
    assert torch.all(term.commands == 0.0)

    interval_steps = int(1.0 / 0.02)  # 50
    torch.manual_seed(0)
    for _ in range(5):
        term.env.episode_length_buf += interval_steps  # land on a resample-eligible tick every call
        term.step()
        assert torch.all(term.commands == 0.0), "a pinned env's command must never drift from zero"


def test_step_still_resamples_unpinned_envs_normally():
    """Regression guard: the pin mechanism must not accidentally freeze envs that were never
    pinned -- only the specific env_ids passed to pin_zero are affected."""
    term = _make_term(num_envs=4, resampling_time=1.0, dt=0.02)
    term.reset(None)
    term.pin_zero(torch.tensor([0]))  # pin only env 0

    interval_steps = int(1.0 / 0.02)
    term.env.episode_length_buf[:] = interval_steps
    torch.manual_seed(1)
    before = term.commands.clone()
    term.step()

    assert torch.all(term.commands[0] == 0.0), "env 0 stays pinned"
    # envs 1-3 were eligible for resample this tick and were NOT pinned -- at least one of them
    # should differ from its pre-step value (a fresh random draw landing on the exact same values
    # by chance is astronomically unlikely across 3 envs x 3 dims).
    assert not torch.equal(term.commands[1:], before[1:]), "unpinned envs must still resample normally"


def test_reset_unpins_and_allows_a_fresh_draw():
    term = _make_term(num_envs=2)
    term.reset(None)
    term.pin_zero(torch.tensor([0]))
    assert bool(term._pinned_zero[0])

    torch.manual_seed(2)
    term.reset(torch.tensor([0]))  # a genuine new episode for env 0

    assert not bool(term._pinned_zero[0]), "a real reset must clear the pin -- a fresh episode gets a fresh draw"


def test_pin_zero_on_empty_env_ids_is_a_no_op():
    term = _make_term(num_envs=3)
    term.reset(None)
    before = term.commands.clone()
    before_pins = term._pinned_zero.clone()

    term.pin_zero(torch.tensor([], dtype=torch.long))

    assert torch.equal(term.commands, before)
    assert torch.equal(term._pinned_zero, before_pins)


def test_pin_zero_before_setup_does_not_crash():
    """setup() allocates self.commands/_pinned_zero -- calling pin_zero before that (shouldn't
    happen in practice, but mirrors the existing None-guard convention in reset()/step()) must not
    raise."""
    env = SimpleNamespace(num_envs=2, device="cpu", dt=0.02, is_evaluating=False)
    cfg = SimpleNamespace(params={"command_ranges": {"lin_vel_x": (-1, 1), "lin_vel_y": (-1, 1), "ang_vel_yaw": (-1, 1)}})
    term = LocomotionCommand(cfg, env)

    term.pin_zero(torch.tensor([0]))  # must not raise


# ----------------------------------------------------------------------------------------------
# pin/unpin (2026-08-13, D8's decel-and-retry fallback for the locomotion->kick direction) --
# same _pinned_zero exclusion mechanism as pin_zero above, generalized to an arbitrary command
# vector, plus the ability to release it before a full reset().


def test_pin_forces_the_given_values_and_excludes_from_resample():
    term = _make_term(num_envs=3)
    torch.manual_seed(0)
    term.reset(None)

    values = torch.tensor([[0.4, 0.0, 0.0], [0.2, 0.0, 0.0]])
    term.pin(torch.tensor([0, 2]), values)

    assert torch.equal(term.commands[0], values[0])
    assert torch.equal(term.commands[2], values[1])
    assert bool(term._pinned_zero[0])
    assert bool(term._pinned_zero[2])
    assert not bool(term._pinned_zero[1])

    interval_steps = int(1.0 / 0.02)
    term.env.episode_length_buf[:] = interval_steps
    torch.manual_seed(1)
    term.step()
    assert torch.equal(term.commands[0], values[0]), "a pinned env's command must never drift"
    assert torch.equal(term.commands[2], values[1])


def test_unpin_restores_normal_periodic_resample():
    term = _make_term(num_envs=2, resampling_time=1.0, dt=0.02)
    term.reset(None)
    term.pin(torch.tensor([0]), torch.tensor([[0.4, 0.0, 0.0]]))
    assert bool(term._pinned_zero[0])

    term.unpin(torch.tensor([0]))
    assert not bool(term._pinned_zero[0])

    interval_steps = int(1.0 / 0.02)
    term.env.episode_length_buf[:] = interval_steps
    torch.manual_seed(3)
    before = term.commands[0].clone()
    term.step()
    assert not torch.equal(term.commands[0], before), "unpinned env must resample normally again"


def test_unpin_on_empty_env_ids_is_a_no_op():
    term = _make_term(num_envs=2)
    term.reset(None)
    term.pin_zero(torch.tensor([0]))
    before_pins = term._pinned_zero.clone()

    term.unpin(torch.tensor([], dtype=torch.long))

    assert torch.equal(term._pinned_zero, before_pins)


def test_pin_and_unpin_before_setup_do_not_crash():
    env = SimpleNamespace(num_envs=2, device="cpu", dt=0.02, is_evaluating=False)
    cfg = SimpleNamespace(params={"command_ranges": {"lin_vel_x": (-1, 1), "lin_vel_y": (-1, 1), "ang_vel_yaw": (-1, 1)}})
    term = LocomotionCommand(cfg, env)

    term.pin(torch.tensor([0]), torch.tensor([[0.1, 0.0, 0.0]]))  # must not raise
    term.unpin(torch.tensor([0]))  # must not raise
