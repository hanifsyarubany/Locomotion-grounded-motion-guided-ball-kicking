"""Unit tests for 4 P1-tier regularization terms (managers/reward/terms/locomotion.py), 2026-08-05,
ported from RoboNaldo (arXiv:2606.11092): ``penalty_kick_lin_vel_z``, ``penalty_kick_dof_vel``,
``penalty_kick_torque``, ``penalty_kick_ee_body_pos_divergence``.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from holosoma.managers.reward.terms import locomotion as loco_terms

# ============================================================================================
# penalty_kick_lin_vel_z
# ============================================================================================


def _root_states(lin_vel_z: float, num_envs: int = 1):
    rs = torch.zeros(num_envs, 13)
    rs[:, 9] = lin_vel_z
    return rs


def test_lin_vel_z_zero_at_zero_velocity():
    env = SimpleNamespace(simulator=SimpleNamespace(robot_root_states=_root_states(0.0)))
    out = loco_terms.penalty_kick_lin_vel_z(env)
    assert out.item() == 0.0


def test_lin_vel_z_matches_squared_formula():
    env = SimpleNamespace(simulator=SimpleNamespace(robot_root_states=_root_states(2.0)))
    out = loco_terms.penalty_kick_lin_vel_z(env)
    assert torch.isclose(out, torch.tensor([4.0]))


def test_lin_vel_z_sign_agnostic():
    env_pos = SimpleNamespace(simulator=SimpleNamespace(robot_root_states=_root_states(1.5)))
    env_neg = SimpleNamespace(simulator=SimpleNamespace(robot_root_states=_root_states(-1.5)))
    assert torch.isclose(
        loco_terms.penalty_kick_lin_vel_z(env_pos), loco_terms.penalty_kick_lin_vel_z(env_neg)
    )


def test_lin_vel_z_per_env_independent():
    rs = torch.zeros(2, 13)
    rs[0, 9] = 1.0
    rs[1, 9] = 3.0
    env = SimpleNamespace(simulator=SimpleNamespace(robot_root_states=rs))
    out = loco_terms.penalty_kick_lin_vel_z(env)
    assert torch.allclose(out, torch.tensor([1.0, 9.0]))


# ============================================================================================
# penalty_kick_dof_vel / penalty_kick_torque
# ============================================================================================


def _env_with_torques(dof_vel: torch.Tensor, torques: torch.Tensor):
    action_term = SimpleNamespace(torques=torques)
    action_manager = SimpleNamespace(_term_instances={"joint_control": action_term})
    simulator = SimpleNamespace(dof_vel=dof_vel)
    return SimpleNamespace(action_manager=action_manager, simulator=simulator)


def test_dof_vel_zero_when_either_factor_zero():
    dof_vel = torch.zeros(1, 3)
    torques = torch.tensor([[5.0, -3.0, 2.0]])
    env = _env_with_torques(dof_vel, torques)
    out = loco_terms.penalty_kick_dof_vel(env)
    assert out.item() == 0.0


def test_dof_vel_matches_squared_power_formula():
    """2026-08-14: formula is sum_j((dof_vel_j * torque_j)^2), NOT the raw signed power sum --
    see penalty_kick_dof_vel's own docstring for the bug the squaring fixes (a signed raw value
    under a negative weight was REWARDING net-negative-power ticks)."""
    dof_vel = torch.tensor([[1.0, 2.0, -1.0]])
    torques = torch.tensor([[3.0, -1.0, 4.0]])
    env = _env_with_torques(dof_vel, torques)
    out = loco_terms.penalty_kick_dof_vel(env)
    per_joint_power = torch.tensor([1.0 * 3.0, 2.0 * -1.0, -1.0 * 4.0])  # = [3, -2, -4]
    expected = torch.sum(per_joint_power**2)  # 9 + 4 + 16 = 29
    assert torch.isclose(out, torch.tensor([expected]))


def test_dof_vel_always_non_negative():
    """2026-08-14 (was test_dof_vel_can_be_negative_unlike_a_pure_magnitude_penalty, inverted):
    the PRE-fix formula (raw signed power, no square) could go negative, which -- combined with
    this term's negative weight -- silently REWARDED net-negative-power ticks (braking/absorbing)
    instead of penalizing them; measured live on a real run growing more antagonistic/oscillatory
    over training (see the function's own docstring for the exact numbers). The fix makes this
    term behave like its sibling penalty_kick_torque (test_torque_always_non_negative below):
    always >= 0, so a negative weight always penalizes larger magnitude, regardless of sign."""
    dof_vel = torch.tensor([[1.0]])
    torques = torch.tensor([[-1.0]])
    env = _env_with_torques(dof_vel, torques)
    out = loco_terms.penalty_kick_dof_vel(env)
    assert out.item() >= 0.0
    assert out.item() == 1.0  # (1.0 * -1.0)^2 = 1.0, not the old formula's -1.0


def test_torque_matches_squared_formula():
    dof_vel = torch.zeros(1, 3)  # unused by penalty_kick_torque
    torques = torch.tensor([[3.0, -4.0, 0.0]])
    env = _env_with_torques(dof_vel, torques)
    out = loco_terms.penalty_kick_torque(env)
    assert torch.isclose(out, torch.tensor([9.0 + 16.0]))


def test_torque_always_non_negative():
    dof_vel = torch.zeros(1, 2)
    torques = torch.tensor([[-5.0, -5.0]])
    env = _env_with_torques(dof_vel, torques)
    out = loco_terms.penalty_kick_torque(env)
    assert out.item() >= 0.0
    assert torch.isclose(out, torch.tensor([50.0]))


def test_dof_vel_and_torque_per_env_independent():
    dof_vel = torch.tensor([[1.0], [2.0]])
    torques = torch.tensor([[1.0], [1.0]])
    env = _env_with_torques(dof_vel, torques)
    out = loco_terms.penalty_kick_dof_vel(env)
    assert torch.allclose(out, torch.tensor([1.0, 4.0]))  # (1*1)^2=1, (2*1)^2=4


# ============================================================================================
# penalty_kick_strike_dof_acc (2026-08-14, added alongside the dof_vel bug fix above -- see that
# function's own docstring for the full measured rationale: striking-phase leg motion becoming
# progressively more violent/jerky over training, which neither the dof_vel/torque penalties
# above nor kick_action_smoothness catch, since none of them read realized joint acceleration.)
# ============================================================================================


def _env_with_dof_acc(
    dof_vel, in_strike_phase, in_kicking_phase=None, has_ball=True, dof_names=None, prev_dof_vel=None, sim_dt=1.0
):
    """2026-08-14 CORRECTION: penalty_kick_strike_dof_acc does NOT read env.simulator.dof_acc
    (live IsaacSim verification found that attribute doesn't exist unless a hardware bridge
    feature is enabled -- see the function's own docstring) -- it computes
    (dof_vel - prev_dof_vel) / sim_dt from env.action_manager's joint_control term, mirroring
    _env_with_torques' access-path fake immediately above. Defaults (prev_dof_vel=0, sim_dt=1.0)
    make the resulting dof_acc numerically equal to the passed-in dof_vel, so most tests below
    can keep naming their input after the dof_acc value they want to test against; only
    test_strike_dof_acc_uses_finite_difference_of_dof_vel below exercises non-trivial
    prev_dof_vel/sim_dt directly.

    in_kicking_phase (2026-08-14, added for penalty_kick_approach_dof_acc below): defaults to
    all-True (irrelevant no-op for the strike term, which never reads it) -- tests of the
    approach term pass this explicitly alongside in_strike_phase, since that term's gate is
    in_kicking_phase & ~in_strike_phase."""
    if prev_dof_vel is None:
        prev_dof_vel = torch.zeros_like(dof_vel)
    if in_kicking_phase is None:
        in_kicking_phase = torch.ones_like(in_strike_phase)
    action_term = SimpleNamespace(get_prev_dof_vel=lambda: prev_dof_vel)
    action_manager = SimpleNamespace(_term_instances={"joint_control": action_term})
    simulator = SimpleNamespace(dof_vel=dof_vel, dof_names=list(dof_names) if dof_names else [])
    motion_command = SimpleNamespace(
        has_ball=has_ball, in_strike_phase=in_strike_phase, in_kicking_phase=in_kicking_phase
    )
    command_manager = SimpleNamespace(get_state=lambda name: motion_command if name == "motion_command" else None)
    return SimpleNamespace(
        simulator=simulator,
        action_manager=action_manager,
        command_manager=command_manager,
        num_envs=dof_vel.shape[0],
        device=dof_vel.device,
        sim_dt=sim_dt,
    )


def test_strike_dof_acc_zero_outside_strike_phase():
    dof_vel = torch.tensor([[2.0, 3.0]])
    env = _env_with_dof_acc(dof_vel, in_strike_phase=torch.tensor([False]))
    out = loco_terms.penalty_kick_strike_dof_acc(env)
    assert out.item() == 0.0


def test_strike_dof_acc_matches_sum_of_squares_during_strike():
    dof_vel = torch.tensor([[2.0, 3.0, -1.0]])
    env = _env_with_dof_acc(dof_vel, in_strike_phase=torch.tensor([True]))
    out = loco_terms.penalty_kick_strike_dof_acc(env)
    assert torch.isclose(out, torch.tensor([4.0 + 9.0 + 1.0]))


def test_strike_dof_acc_uses_finite_difference_of_dof_vel():
    """The actual formula under test: (dof_vel - prev_dof_vel) / sim_dt, read from
    action_manager's joint_control term -- not simulator.dof_acc (see _env_with_dof_acc's own
    docstring for why)."""
    dof_vel = torch.tensor([[5.0, 1.0]])
    prev_dof_vel = torch.tensor([[1.0, 3.0]])
    env = _env_with_dof_acc(
        dof_vel, in_strike_phase=torch.tensor([True]), prev_dof_vel=prev_dof_vel, sim_dt=0.5
    )
    out = loco_terms.penalty_kick_strike_dof_acc(env)
    # dof_acc = ([5,1]-[1,3])/0.5 = [8, -4]  ->  sum(sq) = 64 + 16 = 80
    assert torch.isclose(out, torch.tensor([80.0]))


def test_strike_dof_acc_no_motion_command_returns_zero():
    action_term = SimpleNamespace(get_prev_dof_vel=lambda: torch.zeros(1, 1))
    action_manager = SimpleNamespace(_term_instances={"joint_control": action_term})
    simulator = SimpleNamespace(dof_vel=torch.tensor([[1.0]]), dof_names=[])
    command_manager = SimpleNamespace(get_state=lambda name: None)
    env = SimpleNamespace(
        simulator=simulator, action_manager=action_manager, command_manager=command_manager,
        num_envs=1, device="cpu", sim_dt=1.0,
    )
    out = loco_terms.penalty_kick_strike_dof_acc(env)
    assert out.item() == 0.0


def test_strike_dof_acc_has_ball_false_returns_zero():
    """Locomotion-only env classes / has_ball=False configs must be a safe no-op, matching every
    other in_strike_phase-gated term's convention in this file (e.g. penalty_kick_ee_body_pos_
    divergence's own test of the same name)."""
    dof_vel = torch.tensor([[5.0]])
    env = _env_with_dof_acc(dof_vel, in_strike_phase=torch.tensor([True]), has_ball=False)
    out = loco_terms.penalty_kick_strike_dof_acc(env)
    assert out.item() == 0.0


def test_strike_dof_acc_per_env_independent():
    dof_vel = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    env = _env_with_dof_acc(dof_vel, in_strike_phase=torch.tensor([True, True]))
    out = loco_terms.penalty_kick_strike_dof_acc(env)
    assert torch.allclose(out, torch.tensor([1.0, 4.0]))


def test_strike_dof_acc_mixed_phase_mask_is_per_env_not_global():
    """One env mid-strike, one not (e.g. different clip progress) -- the gate must be a per-env
    mask, not an all-or-nothing switch for the whole batch."""
    dof_vel = torch.tensor([[2.0], [2.0]])
    env = _env_with_dof_acc(dof_vel, in_strike_phase=torch.tensor([True, False]))
    out = loco_terms.penalty_kick_strike_dof_acc(env)
    assert torch.allclose(out, torch.tensor([4.0, 0.0]))


def test_strike_dof_acc_dof_names_subset_restricts_coverage():
    """dof_names=None (the registered default) covers all DOF; passing an explicit subset must
    restrict to exactly those joints, resolved by name against env.simulator.dof_names -- same
    resolution convention MotionStrikeDofPosErrorExp uses."""
    dof_vel = torch.tensor([[3.0, 4.0, 100.0]])  # 3rd joint's huge accel must be excluded
    env = _env_with_dof_acc(
        dof_vel, in_strike_phase=torch.tensor([True]), dof_names=["a", "b", "c"]
    )
    out = loco_terms.penalty_kick_strike_dof_acc(env, dof_names=["a", "b"])
    assert torch.isclose(out, torch.tensor([9.0 + 16.0]))


# ============================================================================================
# penalty_kick_approach_dof_acc (2026-08-14, user-requested split from penalty_kick_strike_
# dof_acc above -- approach and strike want OPPOSITE things from joint acceleration, so a shared
# gate/weight would be a compromise between two opposing objectives. Same formula/access-path as
# the strike term (both go through _dof_acc_magnitude); only the phase gate differs:
# in_kicking_phase & ~in_strike_phase instead of in_strike_phase.)
# ============================================================================================


def test_approach_dof_acc_zero_during_strike():
    dof_vel = torch.tensor([[2.0, 3.0]])
    env = _env_with_dof_acc(
        dof_vel, in_strike_phase=torch.tensor([True]), in_kicking_phase=torch.tensor([True])
    )
    out = loco_terms.penalty_kick_approach_dof_acc(env)
    assert out.item() == 0.0


def test_approach_dof_acc_zero_during_recovery_hold():
    """Past stand_start_idx: in_kicking_phase is False (recovery/hold), so the approach term must
    stay zero there too -- it is NOT simply "not strike", it is strictly the approach window."""
    dof_vel = torch.tensor([[2.0, 3.0]])
    env = _env_with_dof_acc(
        dof_vel, in_strike_phase=torch.tensor([False]), in_kicking_phase=torch.tensor([False])
    )
    out = loco_terms.penalty_kick_approach_dof_acc(env)
    assert out.item() == 0.0


def test_approach_dof_acc_matches_sum_of_squares_during_approach():
    dof_vel = torch.tensor([[2.0, 3.0, -1.0]])
    env = _env_with_dof_acc(
        dof_vel, in_strike_phase=torch.tensor([False]), in_kicking_phase=torch.tensor([True])
    )
    out = loco_terms.penalty_kick_approach_dof_acc(env)
    assert torch.isclose(out, torch.tensor([4.0 + 9.0 + 1.0]))


def test_approach_and_strike_terms_are_mutually_exclusive_across_a_full_cycle():
    """The decisive property motivating the split: for any given tick, at most one of the two
    terms is ever nonzero -- approach, strike, and recovery/hold partition the cycle, they never
    overlap. Exercises all three phase combinations against the SAME dof_vel."""
    dof_vel = torch.tensor([[3.0, 4.0]])  # sum of squares = 25.0 if the gate is open
    phases = [
        ("approach", torch.tensor([False]), torch.tensor([True])),   # in_kicking & ~in_strike
        ("strike", torch.tensor([True]), torch.tensor([True])),      # in_strike
        ("recovery_hold", torch.tensor([False]), torch.tensor([False])),  # ~in_kicking
    ]
    for name, in_strike, in_kicking in phases:
        env = _env_with_dof_acc(dof_vel, in_strike_phase=in_strike, in_kicking_phase=in_kicking)
        approach_out = loco_terms.penalty_kick_approach_dof_acc(env).item()
        strike_out = loco_terms.penalty_kick_strike_dof_acc(env).item()
        if name == "approach":
            assert approach_out == 25.0 and strike_out == 0.0
        elif name == "strike":
            assert approach_out == 0.0 and strike_out == 25.0
        else:
            assert approach_out == 0.0 and strike_out == 0.0


def test_approach_dof_acc_no_motion_command_returns_zero():
    action_term = SimpleNamespace(get_prev_dof_vel=lambda: torch.zeros(1, 1))
    action_manager = SimpleNamespace(_term_instances={"joint_control": action_term})
    simulator = SimpleNamespace(dof_vel=torch.tensor([[1.0]]), dof_names=[])
    command_manager = SimpleNamespace(get_state=lambda name: None)
    env = SimpleNamespace(
        simulator=simulator, action_manager=action_manager, command_manager=command_manager,
        num_envs=1, device="cpu", sim_dt=1.0,
    )
    out = loco_terms.penalty_kick_approach_dof_acc(env)
    assert out.item() == 0.0


def test_approach_dof_acc_has_ball_false_returns_zero():
    dof_vel = torch.tensor([[5.0]])
    env = _env_with_dof_acc(
        dof_vel, in_strike_phase=torch.tensor([False]), in_kicking_phase=torch.tensor([True]), has_ball=False
    )
    out = loco_terms.penalty_kick_approach_dof_acc(env)
    assert out.item() == 0.0


def test_approach_dof_acc_per_env_independent():
    dof_vel = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    env = _env_with_dof_acc(
        dof_vel, in_strike_phase=torch.tensor([False, False]), in_kicking_phase=torch.tensor([True, True])
    )
    out = loco_terms.penalty_kick_approach_dof_acc(env)
    assert torch.allclose(out, torch.tensor([1.0, 4.0]))


def test_approach_dof_acc_mixed_phase_mask_is_per_env_not_global():
    """One env in approach, one in strike -- the gate must be per-env."""
    dof_vel = torch.tensor([[2.0], [2.0]])
    env = _env_with_dof_acc(
        dof_vel, in_strike_phase=torch.tensor([False, True]), in_kicking_phase=torch.tensor([True, True])
    )
    out = loco_terms.penalty_kick_approach_dof_acc(env)
    assert torch.allclose(out, torch.tensor([4.0, 0.0]))


def test_approach_dof_acc_dof_names_subset_restricts_coverage():
    dof_vel = torch.tensor([[3.0, 4.0, 100.0]])
    env = _env_with_dof_acc(
        dof_vel, in_strike_phase=torch.tensor([False]), in_kicking_phase=torch.tensor([True]),
        dof_names=["a", "b", "c"],
    )
    out = loco_terms.penalty_kick_approach_dof_acc(env, dof_names=["a", "b"])
    assert torch.isclose(out, torch.tensor([9.0 + 16.0]))


# ============================================================================================
# penalty_kick_ee_body_pos_divergence
# ============================================================================================

_TRACKED = ["pelvis", "left_ankle_roll_link", "right_ankle_roll_link", "torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link"]


def _fake_motion_command(ref, actual, has_ball=True, tracked=_TRACKED):
    motion_cfg = SimpleNamespace(body_names_to_track=list(tracked))
    return SimpleNamespace(
        motion_cfg=motion_cfg, body_pos_relative_w=ref, robot_body_pos_w=actual, has_ball=has_ball
    )


def _env_with_motion_command(mc, num_envs, device="cpu", episode_length_buf=None):
    command_manager = SimpleNamespace(get_state=lambda name: mc if name == "motion_command" else None)
    # Default well past warmup_steps=20 so tests not specifically about the warmup branch exercise
    # only the always-active Z-only `terminated` branch, unchanged from before that branch existed.
    if episode_length_buf is None:
        episode_length_buf = torch.full((num_envs,), 1000, dtype=torch.long)
    return SimpleNamespace(
        command_manager=command_manager, num_envs=num_envs, device=device, episode_length_buf=episode_length_buf
    )


def test_ee_divergence_zero_when_no_error():
    n, b = 2, len(_TRACKED)
    pos = torch.zeros(n, b, 3)
    mc = _fake_motion_command(pos.clone(), pos.clone())
    env = _env_with_motion_command(mc, n)
    out = loco_terms.penalty_kick_ee_body_pos_divergence(env)
    assert torch.allclose(out, torch.zeros(n))


def test_ee_divergence_fires_when_z_error_exceeds_threshold():
    n, b = 1, len(_TRACKED)
    ref = torch.zeros(n, b, 3)
    actual = torch.zeros(n, b, 3)
    actual[0, _TRACKED.index("left_ankle_roll_link"), 2] = 0.5  # exceeds default threshold=0.25
    mc = _fake_motion_command(ref, actual)
    env = _env_with_motion_command(mc, n)
    out = loco_terms.penalty_kick_ee_body_pos_divergence(env)
    assert out.item() == 1.0


def test_ee_divergence_does_not_fire_below_threshold():
    n, b = 1, len(_TRACKED)
    ref = torch.zeros(n, b, 3)
    actual = torch.zeros(n, b, 3)
    actual[0, _TRACKED.index("left_ankle_roll_link"), 2] = 0.2  # below default threshold=0.25
    mc = _fake_motion_command(ref, actual)
    env = _env_with_motion_command(mc, n)
    out = loco_terms.penalty_kick_ee_body_pos_divergence(env)
    assert out.item() == 0.0


def test_ee_divergence_ignores_xy_error_z_only():
    n, b = 1, len(_TRACKED)
    ref = torch.zeros(n, b, 3)
    actual = torch.zeros(n, b, 3)
    actual[0, _TRACKED.index("left_ankle_roll_link"), 0] = 5.0  # huge X error
    actual[0, _TRACKED.index("left_ankle_roll_link"), 1] = 5.0  # huge Y error
    mc = _fake_motion_command(ref, actual)
    env = _env_with_motion_command(mc, n)
    out = loco_terms.penalty_kick_ee_body_pos_divergence(env)
    assert out.item() == 0.0, "only Z-axis error should be read, matching RoboNaldo's own formula"


def test_ee_divergence_ignores_non_ee_bodies():
    n, b = 1, len(_TRACKED)
    ref = torch.zeros(n, b, 3)
    actual = torch.zeros(n, b, 3)
    actual[0, _TRACKED.index("pelvis"), 2] = 5.0
    actual[0, _TRACKED.index("torso_link"), 2] = 5.0
    mc = _fake_motion_command(ref, actual)
    env = _env_with_motion_command(mc, n)
    out = loco_terms.penalty_kick_ee_body_pos_divergence(env)
    assert out.item() == 0.0


def test_ee_divergence_no_motion_command_returns_zero():
    command_manager = SimpleNamespace(get_state=lambda name: None)
    env = SimpleNamespace(command_manager=command_manager, num_envs=3, device="cpu")
    out = loco_terms.penalty_kick_ee_body_pos_divergence(env)
    assert torch.equal(out, torch.zeros(3))


def test_ee_divergence_has_ball_false_returns_zero():
    n, b = 1, len(_TRACKED)
    actual = torch.zeros(n, b, 3)
    actual[0, _TRACKED.index("left_ankle_roll_link"), 2] = 5.0
    mc = _fake_motion_command(torch.zeros(n, b, 3), actual, has_ball=False)
    env = _env_with_motion_command(mc, n)
    out = loco_terms.penalty_kick_ee_body_pos_divergence(env)
    assert out.item() == 0.0


def test_ee_divergence_per_env_independent():
    n, b = 2, len(_TRACKED)
    ref = torch.zeros(n, b, 3)
    actual = torch.zeros(n, b, 3)
    actual[1, _TRACKED.index("right_wrist_yaw_link"), 2] = 1.0  # only env 1 diverges
    mc = _fake_motion_command(ref, actual)
    env = _env_with_motion_command(mc, n)
    out = loco_terms.penalty_kick_ee_body_pos_divergence(env)
    assert out[0].item() == 0.0
    assert out[1].item() == 1.0


def test_ee_divergence_custom_body_names_and_threshold():
    n, b = 1, len(_TRACKED)
    ref = torch.zeros(n, b, 3)
    actual = torch.zeros(n, b, 3)
    actual[0, _TRACKED.index("torso_link"), 2] = 0.15
    mc = _fake_motion_command(ref, actual)
    env = _env_with_motion_command(mc, n)
    out = loco_terms.penalty_kick_ee_body_pos_divergence(env, threshold=0.1, body_names=["torso_link"])
    assert out.item() == 1.0


# ============================================================================================
# penalty_kick_ee_body_pos_divergence -- warmup branch (2026-08-05, correction: RoboNaldo's
# ee_body_pos_termination_penalty has a second branch gated on command.is_warmup, previously
# dropped as "no equivalent in this project" -- env.episode_length_buf < warmup_steps is that
# equivalent, see the function's own docstring for the derivation).
# ============================================================================================


def test_ee_divergence_warmup_branch_fires_on_xy_only_error_within_window():
    """XY-only divergence (Z=0) does NOT trip the always-active Z-only `terminated` branch, but
    DOES trip the warmup branch's full 3-axis check while still within warmup_steps of a reset."""
    n, b = 1, len(_TRACKED)
    ref = torch.zeros(n, b, 3)
    actual = torch.zeros(n, b, 3)
    actual[0, _TRACKED.index("left_ankle_roll_link"), 0] = 0.5  # X-only, exceeds warmup_threshold=0.25
    mc = _fake_motion_command(ref, actual)
    env = _env_with_motion_command(mc, n, episode_length_buf=torch.tensor([5]))  # well within warmup_steps=20
    out = loco_terms.penalty_kick_ee_body_pos_divergence(env)
    assert out.item() == 1.0


def test_ee_divergence_warmup_branch_does_not_fire_outside_window():
    """The SAME XY-only divergence that fires inside warmup must NOT fire once past warmup_steps
    -- only the always-active Z-only branch applies there (matches the pre-existing
    test_ee_divergence_ignores_xy_error_z_only, restated explicitly at the boundary)."""
    n, b = 1, len(_TRACKED)
    ref = torch.zeros(n, b, 3)
    actual = torch.zeros(n, b, 3)
    actual[0, _TRACKED.index("left_ankle_roll_link"), 0] = 0.5
    mc = _fake_motion_command(ref, actual)
    env = _env_with_motion_command(mc, n, episode_length_buf=torch.tensor([20]))  # exactly at warmup_steps
    out = loco_terms.penalty_kick_ee_body_pos_divergence(env)
    assert out.item() == 0.0, "is_warmup uses strict '<', so episode_length_buf==warmup_steps is NOT warmup"


def test_ee_divergence_warmup_branch_below_warmup_threshold_does_not_fire():
    n, b = 1, len(_TRACKED)
    ref = torch.zeros(n, b, 3)
    actual = torch.zeros(n, b, 3)
    actual[0, _TRACKED.index("left_ankle_roll_link"), 0] = 0.1  # below default warmup_threshold=0.25
    mc = _fake_motion_command(ref, actual)
    env = _env_with_motion_command(mc, n, episode_length_buf=torch.tensor([0]))
    out = loco_terms.penalty_kick_ee_body_pos_divergence(env)
    assert out.item() == 0.0


def test_ee_divergence_warmup_custom_params():
    n, b = 1, len(_TRACKED)
    ref = torch.zeros(n, b, 3)
    actual = torch.zeros(n, b, 3)
    actual[0, _TRACKED.index("left_ankle_roll_link"), 1] = 0.6  # Y-only
    mc = _fake_motion_command(ref, actual)
    env = _env_with_motion_command(mc, n, episode_length_buf=torch.tensor([3]))
    # default warmup_threshold=0.25 would fire; a wider custom warmup_threshold should not.
    out_default = loco_terms.penalty_kick_ee_body_pos_divergence(env)
    out_widened = loco_terms.penalty_kick_ee_body_pos_divergence(env, warmup_threshold=1.0)
    assert out_default.item() == 1.0
    assert out_widened.item() == 0.0
    # shrinking warmup_steps below episode_length_buf=3 takes the env out of warmup entirely.
    out_short_warmup = loco_terms.penalty_kick_ee_body_pos_divergence(env, warmup_steps=2)
    assert out_short_warmup.item() == 0.0


def test_ee_divergence_warmup_per_env_independent():
    n, b = 2, len(_TRACKED)
    ref = torch.zeros(n, b, 3)
    actual = torch.zeros(n, b, 3)
    actual[:, _TRACKED.index("left_ankle_roll_link"), 0] = 0.5  # both envs have the same XY-only error
    mc = _fake_motion_command(ref, actual)
    env = _env_with_motion_command(mc, n, episode_length_buf=torch.tensor([2, 50]))  # env0 in warmup, env1 not
    out = loco_terms.penalty_kick_ee_body_pos_divergence(env)
    assert out[0].item() == 1.0
    assert out[1].item() == 0.0


def test_ee_divergence_default_threshold_is_0_25_matching_termination():
    """The function's own default (not just the registered params) should match
    bad_tracking's pre-existing hardcoded bad_motion_body_pos_threshold=0.25, so a caller that
    doesn't explicitly pass threshold still gets a value consistent with the termination side."""
    n, b = 1, len(_TRACKED)
    ref = torch.zeros(n, b, 3)
    actual = torch.zeros(n, b, 3)
    actual[0, _TRACKED.index("left_ankle_roll_link"), 2] = 0.26
    mc = _fake_motion_command(ref, actual)
    env = _env_with_motion_command(mc, n)
    out = loco_terms.penalty_kick_ee_body_pos_divergence(env)
    assert out.item() == 1.0


# ============================================================================================
# 2026-08-15, "simultaneous per-skill task configs": threshold/warmup_threshold arrive as a
# per-env [num_envs] tensor whenever kick_penalty_ee_body_pos_divergence's params_per_skill
# genuinely diverges (RewardManager.compute()'s generic gather) -- diff[:, :, 2]/warmup_error are
# [num_envs, num_bodies], so a bare [num_envs] tensor must be unsqueezed to broadcast against the
# right dim. Regression coverage for a real crash: a live multi-skill launch (2 skills, genuinely
# divergent threshold) hit "RuntimeError: The size of tensor a (4) must match the size of tensor
# b (32)" here before this fix.
# ============================================================================================


def test_ee_divergence_per_env_tensor_threshold_only_fires_for_the_lower_threshold_env():
    """Both envs have the SAME 0.3 z-error on one tracked body -- env0's per-env threshold (0.5)
    is above it (no fire), env1's (0.1) is below it (fires). A scalar threshold could never
    produce this split; isolates the tensor-threshold broadcast path."""
    n, b = 2, len(_TRACKED)
    ref = torch.zeros(n, b, 3)
    actual = torch.zeros(n, b, 3)
    actual[:, _TRACKED.index("left_ankle_roll_link"), 2] = 0.3
    mc = _fake_motion_command(ref, actual)
    env = _env_with_motion_command(mc, n)
    out = loco_terms.penalty_kick_ee_body_pos_divergence(env, threshold=torch.tensor([0.5, 0.1]))
    assert out[0].item() == 0.0
    assert out[1].item() == 1.0


def test_ee_divergence_per_env_tensor_warmup_threshold_only_fires_for_the_lower_threshold_env():
    """Same split as above, but exercised during the warmup window (episode_length_buf < 20) via
    a full 3-axis (not just Z) error, isolating warmup_threshold's own broadcast path."""
    n, b = 2, len(_TRACKED)
    ref = torch.zeros(n, b, 3)
    actual = torch.zeros(n, b, 3)
    actual[:, _TRACKED.index("left_ankle_roll_link"), 0] = 0.3  # X-only error -- only reaches
    # the warmup branch (Z-only `terminated` branch ignores X), so this isolates warmup_threshold.
    mc = _fake_motion_command(ref, actual)
    env = _env_with_motion_command(mc, n, episode_length_buf=torch.zeros(n, dtype=torch.long))
    out = loco_terms.penalty_kick_ee_body_pos_divergence(env, warmup_threshold=torch.tensor([0.5, 0.1]))
    assert out[0].item() == 0.0
    assert out[1].item() == 1.0


def test_ee_divergence_per_env_tensor_threshold_matches_scalar_when_uniform():
    """A [num_envs] tensor with the SAME value in every slot must behave identically to passing
    that value as a plain scalar -- proves the tensor path is a true generalization, not a
    different formula."""
    n, b = 2, len(_TRACKED)
    ref = torch.zeros(n, b, 3)
    actual = torch.zeros(n, b, 3)
    actual[:, _TRACKED.index("left_ankle_roll_link"), 2] = 0.3
    mc = _fake_motion_command(ref, actual)
    env = _env_with_motion_command(mc, n)
    out_scalar = loco_terms.penalty_kick_ee_body_pos_divergence(env, threshold=0.1)
    out_tensor = loco_terms.penalty_kick_ee_body_pos_divergence(env, threshold=torch.tensor([0.1, 0.1]))
    assert torch.equal(out_scalar, out_tensor)
