"""Unit tests for MotionCommand._synthesize_kick_aim_target_local (2026-08-22 azimuth-aim
refactor) -- the shared per-env target-placement logic called from all three ball/target
placement paths (reset, place_ball_at_entry, place_ball_at_reset_pending). See
SkillConfig.kick_aim_enabled's own docstring for the full mechanism.

Isolated via a lightweight fake object (not a real MotionCommand instance -- constructing one
needs a live env), same pattern as test_local_xy_to_world.py: plain instance attributes plus the
REAL, unbound method, so these tests exercise production code, not a reimplementation of it.
"""

from __future__ import annotations

import math

import torch

from holosoma.managers.command.terms.wbt import MotionCommand


class _FakeMotionCommand:
    """Provides exactly the attributes _synthesize_kick_aim_target_local reads/writes, as plain
    tensors, and borrows the REAL unbound method."""

    def __init__(
        self,
        kick_aim_enabled_per_motion: torch.Tensor,
        nominal_bearing_deg_per_motion: torch.Tensor,
        kick_aim_theta_max_deg_per_motion: torch.Tensor,
        kick_aim_nominal_distance_m: float,
        num_envs: int,
    ):
        self.kick_aim_enabled_per_motion = kick_aim_enabled_per_motion
        self.nominal_bearing_deg_per_motion = nominal_bearing_deg_per_motion
        self.kick_aim_theta_max_deg_per_motion = kick_aim_theta_max_deg_per_motion
        self.kick_aim_nominal_distance_m = kick_aim_nominal_distance_m
        self.kick_aim_theta = torch.zeros(num_envs)
        self.device = "cpu"

    _synthesize_kick_aim_target_local = MotionCommand._synthesize_kick_aim_target_local


def _all_disabled_fake(num_motions: int = 1, num_envs: int = 1) -> _FakeMotionCommand:
    return _FakeMotionCommand(
        kick_aim_enabled_per_motion=torch.zeros(num_motions, dtype=torch.bool),
        nominal_bearing_deg_per_motion=torch.zeros(num_motions),
        kick_aim_theta_max_deg_per_motion=torch.full((num_motions,), 15.0),
        kick_aim_nominal_distance_m=5.0,
        num_envs=num_envs,
    )


def test_legacy_target_is_the_fixed_nominal_point_no_randomization():
    """kick_aim_enabled=False: the old independent randomize_target_x/y draw was removed
    2026-08-22 -- the legacy branch is now a pure constant (target_nominal_local), unconditionally,
    with NO RNG draw at all. Two calls under different seeds must return the identical value."""
    fake = _all_disabled_fake()
    env_ids = torch.tensor([0])
    env_motion_ids = torch.tensor([0])
    ball_local_placed = torch.tensor([[1.0, 0.0]])
    target_nominal_local = torch.tensor([[6.0, 0.0]])

    torch.manual_seed(42)
    result_a = fake._synthesize_kick_aim_target_local(
        env_ids, env_motion_ids, ball_local_placed, target_nominal_local
    )
    torch.manual_seed(999)  # different seed -- must not matter, no RNG is consumed
    result_b = fake._synthesize_kick_aim_target_local(
        env_ids, env_motion_ids, ball_local_placed, target_nominal_local
    )

    assert torch.allclose(result_a, target_nominal_local)
    assert torch.allclose(result_a, result_b)
    # kick_aim_theta must stay untouched (still its zero-init) -- no sampling happened at all.
    assert torch.all(fake.kick_aim_theta == 0.0)


def test_legacy_skill_target_is_independent_of_ball_position():
    """A non-aim-enabled skill's target must be the fixed nominal point -- changing
    ball_local_placed must NOT change the result at all."""
    fake = _all_disabled_fake()
    env_ids = torch.tensor([0])
    env_motion_ids = torch.tensor([0])
    target_nominal_local = torch.tensor([[6.0, 0.0]])

    result_a = fake._synthesize_kick_aim_target_local(
        env_ids, env_motion_ids, torch.tensor([[1.0, 0.0]]), target_nominal_local
    )
    result_b = fake._synthesize_kick_aim_target_local(
        env_ids, env_motion_ids, torch.tensor([[99.0, -50.0]]), target_nominal_local
    )
    assert torch.allclose(result_a, result_b)
    assert torch.allclose(result_a, target_nominal_local)


def test_kick_aim_target_minus_ball_is_exactly_d_along_bearing_regardless_of_ball_spawn_noise():
    """The core spawn-invariance property: for a kick_aim_enabled skill, target - ball must equal
    D * unit(nominal_bearing + theta) EXACTLY, no matter where the ball's own noise draw landed --
    checked across several different ball_local_placed values with theta forced to 0 (via
    theta_max_deg=0, so the sampled theta is bounded to exactly 0)."""
    nominal_bearing_deg = 30.0
    D = 5.0
    fake = _FakeMotionCommand(
        kick_aim_enabled_per_motion=torch.tensor([True]),
        nominal_bearing_deg_per_motion=torch.tensor([nominal_bearing_deg]),
        kick_aim_theta_max_deg_per_motion=torch.tensor([0.0]),  # forces theta == 0 always
        kick_aim_nominal_distance_m=D,
        num_envs=1,
    )
    env_ids = torch.tensor([0])
    env_motion_ids = torch.tensor([0])
    target_nominal_local = torch.tensor([[999.0, 999.0]])  # must be IGNORED entirely in aim mode

    expected_dx = D * math.cos(math.radians(nominal_bearing_deg))
    expected_dy = D * math.sin(math.radians(nominal_bearing_deg))

    for ball_xy in ([1.0, 0.0], [-3.5, 2.2], [0.0, 0.0], [50.0, -50.0]):
        ball_local_placed = torch.tensor([ball_xy])
        result = fake._synthesize_kick_aim_target_local(
            env_ids, env_motion_ids, ball_local_placed, target_nominal_local
        )
        delta = result - ball_local_placed
        assert torch.allclose(delta, torch.tensor([[expected_dx, expected_dy]]), atol=1e-5), (
            f"ball_xy={ball_xy}: target-ball={delta.tolist()}, expected=[{expected_dx}, {expected_dy}]"
        )


def test_theta_sampled_within_configured_range_and_stored():
    """kick_aim_theta must land in [-theta_max, theta_max] and be persisted to self.kick_aim_theta
    for the calling env_ids -- the buffer the observation term reads as a held constant."""
    fake = _FakeMotionCommand(
        kick_aim_enabled_per_motion=torch.ones(1, dtype=torch.bool),
        nominal_bearing_deg_per_motion=torch.zeros(1),
        kick_aim_theta_max_deg_per_motion=torch.tensor([15.0]),
        kick_aim_nominal_distance_m=5.0,
        num_envs=200,
    )
    env_ids = torch.arange(200)
    env_motion_ids = torch.zeros(200, dtype=torch.long)
    ball_local_placed = torch.zeros(200, 2)
    target_nominal_local = torch.zeros(200, 2)

    torch.manual_seed(7)
    fake._synthesize_kick_aim_target_local(
        env_ids, env_motion_ids, ball_local_placed, target_nominal_local
    )
    assert torch.all(fake.kick_aim_theta.abs() <= 15.0 + 1e-5)
    # Not a degenerate all-zero sample -- confirms randomness is actually happening.
    assert fake.kick_aim_theta.std().item() > 1.0


def test_mixed_batch_selects_per_env_independently():
    """Two motions in the same call, one kick_aim_enabled, one not -- each env's OWN assigned
    skill must decide its own behavior, not whichever the batch mostly is."""
    fake = _FakeMotionCommand(
        kick_aim_enabled_per_motion=torch.tensor([True, False]),
        nominal_bearing_deg_per_motion=torch.tensor([90.0, 0.0]),
        kick_aim_theta_max_deg_per_motion=torch.tensor([0.0, 0.0]),
        kick_aim_nominal_distance_m=5.0,
        num_envs=2,
    )
    env_ids = torch.tensor([0, 1])
    env_motion_ids = torch.tensor([0, 1])  # env 0 -> aim-enabled skill, env 1 -> legacy skill
    ball_local_placed = torch.tensor([[2.0, 0.0], [2.0, 0.0]])
    target_nominal_local = torch.tensor([[0.0, 0.0], [7.0, 1.0]])

    result = fake._synthesize_kick_aim_target_local(
        env_ids, env_motion_ids, ball_local_placed, target_nominal_local
    )
    # env 0: aim-enabled, bearing=90deg (straight +y), D=5 from ball (2,0) -> (2, 5).
    assert torch.allclose(result[0], torch.tensor([2.0, 5.0]), atol=1e-5)
    # env 1: legacy, independent of ball position entirely -> exactly its own nominal target.
    assert torch.allclose(result[1], torch.tensor([7.0, 1.0]), atol=1e-5)
    # theta buffer: sampled (but forced to 0 via theta_max=0) for env 0, exact zero for env 1.
    assert fake.kick_aim_theta[0].item() == 0.0
    assert fake.kick_aim_theta[1].item() == 0.0
