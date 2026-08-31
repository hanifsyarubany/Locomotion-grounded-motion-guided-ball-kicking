"""Unit tests for BodyPushRandomizerState / apply_body_pushes (2026-08-27): sustained,
body-targeted collision-style disturbance forces, additive to the existing root-velocity push.

Isolated via bare fake env/simulator objects exposing only the specific attributes these hooks
touch, matching this project's existing bare-instance test convention (see test_ball_obs_bias.py)
-- no real IsaacSim/MuJoCo env needed. What these tests do NOT cover: whether a force written via
set_external_body_forces actually reaches PhysX/MuJoCo correctly (frame conversion, body-index
mapping) -- that needs a live probe against a real simulator, same as every other randomizer in
this project's history.
"""

import torch

from holosoma.managers.randomization.terms.locomotion import (
    DEFAULT_BODY_PUSH_BODIES,
    BodyPushRandomizerState,
    apply_body_pushes,
)


class _FakeCfg:
    def __init__(self, params: dict):
        self.params = params


class _FakeSimulator:
    """body_names is holosoma order; find_rigid_body_indice mimics both real backends' contract
    (returns an int index, or None if the name doesn't exist -- see isaacsim.py's own docstring)."""

    def __init__(self, body_names: list[str]):
        self.body_names = body_names
        self.num_bodies = len(body_names)  # deliberately holosoma-order here, unlike MuJoCo's
        # raw nbody -- see randomize_body_push_startup's own comment on why num_bodies is NOT
        # trusted for this shape.
        self.forces_written: list[torch.Tensor] = []

    def find_rigid_body_indice(self, name: str):
        if name not in self.body_names:
            return None
        return self.body_names.index(name)

    def set_external_body_forces(self, forces_w: torch.Tensor) -> None:
        self.forces_written.append(forces_w.clone())


class _FakeEnv:
    def __init__(self, num_envs: int, body_names: list[str], dt: float = 0.02):
        self.num_envs = num_envs
        self.device = "cpu"
        self.dt = dt
        self.is_evaluating = False
        self.simulator = _FakeSimulator(body_names)
        self.randomization_manager = _FakeRandomizationManager()


class _FakeRandomizationManager:
    def __init__(self):
        self._states: dict[str, object] = {}

    def get_state(self, name: str):
        return self._states.get(name)


_BODY_NAMES = [
    "pelvis",
    "left_knee_link",
    "right_knee_link",
    "left_elbow_link",
    "right_elbow_link",
    "torso_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
]


def _make_state(env, **param_overrides) -> BodyPushRandomizerState:
    params = {
        "enabled": True,
        "interval_s": [4.0, 8.0],
        "force_range": [20.0, 80.0],
        "duration_s": [0.05, 0.20],
        "vertical_fraction": 0.2,
    }
    params.update(param_overrides)
    state = BodyPushRandomizerState(_FakeCfg(params), env)
    state.setup()
    env.randomization_manager._states["body_push_randomizer_state"] = state
    return state


def test_setup_resolves_default_bodies_against_real_link_names():
    env = _FakeEnv(num_envs=4, body_names=_BODY_NAMES)
    state = _make_state(env)
    assert state.body_indices is not None
    resolved_names = {_BODY_NAMES[i] for i in state.body_indices.tolist()}
    assert resolved_names == set(DEFAULT_BODY_PUSH_BODIES)


def test_setup_raises_on_unknown_body_name():
    env = _FakeEnv(num_envs=4, body_names=_BODY_NAMES)
    try:
        _make_state(env, body_names=["not_a_real_link"])
    except ValueError as e:
        assert "not_a_real_link" in str(e)
    else:
        raise AssertionError("expected ValueError for an unresolvable body name")


def test_force_buf_shape_is_holosoma_body_count_not_num_bodies_attr():
    # Regression guard for the exact bug class this codebase has hit twice before: MuJoCo's
    # simulator.num_bodies is raw model nbody (world + ball included), NOT the holosoma-ordered
    # count set_external_body_forces expects. Give the fake simulator a num_bodies that
    # deliberately DISAGREES with len(body_names) and confirm force_buf still follows body_names.
    env = _FakeEnv(num_envs=3, body_names=_BODY_NAMES)
    env.simulator.num_bodies = len(_BODY_NAMES) + 2  # simulate MuJoCo's world+ball inflation
    state = _make_state(env)
    assert state.force_buf.shape == (3, len(_BODY_NAMES), 3)


def test_sample_forces_respects_magnitude_and_vertical_bounds():
    torch.manual_seed(0)
    env = _FakeEnv(num_envs=200, body_names=_BODY_NAMES)
    state = _make_state(env, force_range=[20.0, 80.0], vertical_fraction=0.2)
    env_ids = torch.arange(200)
    state._sample_forces(env_ids)

    forces = state.force_buf[env_ids]  # (200, num_bodies, 3)
    magnitudes = forces.norm(dim=-1).sum(dim=-1)  # exactly one nonzero body per env
    assert torch.all(magnitudes >= 20.0 - 1e-4)
    assert torch.all(magnitudes <= 80.0 + 1e-4)

    nonzero_per_env = (forces.abs().sum(dim=-1) > 0).sum(dim=-1)
    assert torch.all(nonzero_per_env == 1), "exactly one body should carry force per env"

    z_component = forces.sum(dim=1)[:, 2]  # the single nonzero body's z, summed harmlessly
    assert torch.all(z_component.abs() <= 80.0 * 0.2 + 1e-4)


def test_apply_body_pushes_fires_when_due_and_clears_on_expiry():
    torch.manual_seed(0)
    env = _FakeEnv(num_envs=2, body_names=_BODY_NAMES, dt=0.02)
    state = _make_state(
        env, interval_s=[0.02, 0.02], duration_s=[0.02, 0.02], force_range=[50.0, 50.0]
    )
    # interval_steps resolves to exactly 1 step at dt=0.02 -> due on the very first tick.
    state.step()  # counter: 0 -> 1
    apply_body_pushes(env)

    assert len(env.simulator.forces_written) == 1
    written = env.simulator.forces_written[0]
    assert written.abs().sum() > 0, "expected a nonzero force on the first due tick"
    assert state.remaining_steps is not None
    # duration_steps resolves to >=1; after sampling this step it has not yet been decremented
    # inside apply_body_pushes's own tick-down (sampling happens first, then the tick), so at
    # least one env should show remaining_steps > 0 immediately after firing.
    assert bool((state.remaining_steps > 0).any())

    # Drive it forward until every disturbance expires, confirming a final zero-clearing write.
    for _ in range(20):
        state.step()
        apply_body_pushes(env)
        if not bool((state.remaining_steps > 0).any()):
            break

    assert bool((state.remaining_steps == 0).all())
    last_written = env.simulator.forces_written[-1]
    assert torch.all(last_written == 0.0), "expired disturbances must clear the simulator's buffer"


def test_apply_body_pushes_is_noop_when_disabled():
    env = _FakeEnv(num_envs=2, body_names=_BODY_NAMES)
    _make_state(env, enabled=False)
    apply_body_pushes(env)
    assert env.simulator.forces_written == [], "a disabled term must never touch the simulator"


def test_apply_body_pushes_clears_residual_force_when_evaluation_starts():
    torch.manual_seed(0)
    env = _FakeEnv(num_envs=2, body_names=_BODY_NAMES, dt=0.02)
    state = _make_state(
        env, interval_s=[0.02, 0.02], duration_s=[10.0, 10.0], force_range=[50.0, 50.0]
    )
    state.step()
    apply_body_pushes(env)
    assert len(env.simulator.forces_written) == 1
    assert env.simulator.forces_written[-1].abs().sum() > 0

    env.is_evaluating = True
    apply_body_pushes(env)
    assert len(env.simulator.forces_written) == 2
    assert torch.all(env.simulator.forces_written[-1] == 0.0), (
        "entering eval mid-disturbance must clear the simulator's persistent force buffer, "
        "since neither backend auto-clears it per step"
    )

    # A further call while still evaluating must not re-touch the simulator (no residual left).
    apply_body_pushes(env)
    assert len(env.simulator.forces_written) == 2


def test_reset_zeroes_state_for_given_envs_only():
    torch.manual_seed(0)
    env = _FakeEnv(num_envs=3, body_names=_BODY_NAMES, dt=0.02)
    state = _make_state(
        env, interval_s=[0.02, 0.02], duration_s=[10.0, 10.0], force_range=[50.0, 50.0]
    )
    state.step()
    apply_body_pushes(env)
    assert bool((state.remaining_steps > 0).all())

    state.reset(torch.tensor([1]))
    assert state.remaining_steps[1] == 0
    assert torch.all(state.force_buf[1] == 0.0)
    assert state.remaining_steps[0] > 0, "reset must not touch envs outside env_ids"
    assert state.remaining_steps[2] > 0
