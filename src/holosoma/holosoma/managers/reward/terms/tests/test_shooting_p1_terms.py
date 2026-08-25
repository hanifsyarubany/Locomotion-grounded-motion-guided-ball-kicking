"""Unit tests for 5 new shooting reward terms (managers/reward/terms/shooting.py), 2026-08-05,
ported from RoboNaldo (arXiv:2606.11092): ``penalize_weak_foot_contact``,
``penalize_self_contact_feet``, ``robot_com_ball_distance``, ``robot_torso_ball_distance``,
``ball_over_line``. Reuses this file's own established isolation discipline (patch ``_tracker``/
``current_w_g`` to known values, same as test_shooting_strike_gate.py's ``_FakeTracker``).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import torch

import holosoma.managers.reward.terms.shooting as s

_BODY_NAMES = ["pelvis", "left_ankle_roll_link", "right_ankle_roll_link", "torso_link"]
_LEFT_FOOT_IDX = _BODY_NAMES.index("left_ankle_roll_link")
_RIGHT_FOOT_IDX = _BODY_NAMES.index("right_ankle_roll_link")
_TORSO_IDX = _BODY_NAMES.index("torso_link")


def _fake_motion_command(
    in_kicking_phase=None, is_ood_spawn=None, target_xy_w=None, ball_spawn_pos_w=None, num_envs=1
):
    if in_kicking_phase is None:
        in_kicking_phase = torch.ones(num_envs, dtype=torch.bool)
    if is_ood_spawn is None:
        is_ood_spawn = torch.zeros(num_envs, dtype=torch.bool)
    if target_xy_w is None:
        target_xy_w = torch.zeros(num_envs, 2)
    if ball_spawn_pos_w is None:
        ball_spawn_pos_w = torch.zeros(num_envs, 3)
    return SimpleNamespace(
        in_kicking_phase=in_kicking_phase,
        is_ood_spawn=is_ood_spawn,
        target_xy_w=target_xy_w,
        ball_spawn_pos_w=ball_spawn_pos_w,
    )


def _fake_env(motion_command, num_envs, rigid_body_pos=None, robot_root_states=None, device="cpu"):
    command_manager = SimpleNamespace(get_state=lambda name: motion_command if name == "motion_command" else None)
    simulator = SimpleNamespace(
        body_names=list(_BODY_NAMES),
        _rigid_body_pos=rigid_body_pos if rigid_body_pos is not None else torch.zeros(num_envs, len(_BODY_NAMES), 3),
        robot_root_states=robot_root_states if robot_root_states is not None else torch.zeros(num_envs, 13),
    )
    return SimpleNamespace(command_manager=command_manager, simulator=simulator, num_envs=num_envs, device=device)


class _FakeTracker:
    def __init__(self, num_envs, ball_pos_w=None, has_kicked=None, kick_foot_is_left=None, kick_foot_index=None):
        self.ball_pos_w = ball_pos_w if ball_pos_w is not None else torch.zeros(num_envs, 3)
        self.has_kicked = has_kicked if has_kicked is not None else torch.zeros(num_envs, dtype=torch.bool)
        self.kick_foot_is_left = (
            kick_foot_is_left if kick_foot_is_left is not None else torch.zeros(num_envs, dtype=torch.bool)
        )
        self.kick_foot_index = (
            kick_foot_index if kick_foot_index is not None else torch.full((num_envs,), _RIGHT_FOOT_IDX)
        )
        self._env_arange = torch.arange(num_envs)


# ============================================================================================
# penalize_weak_foot_contact
# ============================================================================================


def test_weak_foot_peaks_at_threshold_distance():
    """The weak (non-kicking) foot sitting EXACTLY at threshold distance from the ball -> reward
    exactly 1.0 (the Gaussian bump's peak)."""
    rigid_body_pos = torch.zeros(1, len(_BODY_NAMES), 3)
    rigid_body_pos[0, _LEFT_FOOT_IDX, 0] = 0.12  # weak foot = left (kick foot is right)
    mc = _fake_motion_command(num_envs=1)
    env = _fake_env(mc, 1, rigid_body_pos=rigid_body_pos)
    tracker = _FakeTracker(1, kick_foot_is_left=torch.zeros(1, dtype=torch.bool))  # kick foot = right -> weak = left
    with patch.object(s, "_tracker", return_value=tracker), patch.object(s, "current_w_g", return_value=torch.ones(1)):
        out = s.penalize_weak_foot_contact(env, threshold=0.12, std=0.1)
    assert torch.isclose(out, torch.tensor([1.0]), atol=1e-5)


def test_weak_foot_decays_away_from_threshold_in_either_direction():
    rigid_body_pos_near = torch.zeros(1, len(_BODY_NAMES), 3)
    rigid_body_pos_near[0, _LEFT_FOOT_IDX, 0] = 0.0  # far below threshold
    rigid_body_pos_far = torch.zeros(1, len(_BODY_NAMES), 3)
    rigid_body_pos_far[0, _LEFT_FOOT_IDX, 0] = 1.0  # far above threshold
    mc = _fake_motion_command(num_envs=1)
    tracker = _FakeTracker(1, kick_foot_is_left=torch.zeros(1, dtype=torch.bool))
    for rbp in (rigid_body_pos_near, rigid_body_pos_far):
        env = _fake_env(mc, 1, rigid_body_pos=rbp)
        with patch.object(s, "_tracker", return_value=tracker), patch.object(s, "current_w_g", return_value=torch.ones(1)):
            out = s.penalize_weak_foot_contact(env, threshold=0.12, std=0.1)
        assert out.item() < 1.0, "must decay away from the peak in both directions"


def test_weak_foot_selects_correct_foot_when_kick_foot_is_left():
    """When the KICK foot is left, the weak foot must be RIGHT -- verified by moving only the
    right foot and confirming the term responds."""
    rigid_body_pos = torch.zeros(1, len(_BODY_NAMES), 3)
    rigid_body_pos[0, _RIGHT_FOOT_IDX, 0] = 0.12
    mc = _fake_motion_command(num_envs=1)
    env = _fake_env(mc, 1, rigid_body_pos=rigid_body_pos)
    tracker = _FakeTracker(1, kick_foot_is_left=torch.ones(1, dtype=torch.bool), kick_foot_index=torch.full((1,), _LEFT_FOOT_IDX))
    with patch.object(s, "_tracker", return_value=tracker), patch.object(s, "current_w_g", return_value=torch.ones(1)):
        out = s.penalize_weak_foot_contact(env, threshold=0.12, std=0.1)
    assert torch.isclose(out, torch.tensor([1.0]), atol=1e-5)


def test_weak_foot_ood_gate_zeroes_output():
    rigid_body_pos = torch.zeros(1, len(_BODY_NAMES), 3)
    rigid_body_pos[0, _LEFT_FOOT_IDX, 0] = 0.12
    mc = _fake_motion_command(is_ood_spawn=torch.ones(1, dtype=torch.bool), num_envs=1)
    env = _fake_env(mc, 1, rigid_body_pos=rigid_body_pos)
    tracker = _FakeTracker(1, kick_foot_is_left=torch.zeros(1, dtype=torch.bool))
    with patch.object(s, "_tracker", return_value=tracker), patch.object(s, "current_w_g", return_value=torch.ones(1)):
        out = s.penalize_weak_foot_contact(env, threshold=0.12, std=0.1)
    assert out.item() == 0.0


# ============================================================================================
# penalize_self_contact_feet
# ============================================================================================


def test_self_contact_zero_when_feet_far_apart():
    rigid_body_pos = torch.zeros(1, len(_BODY_NAMES), 3)
    rigid_body_pos[0, _LEFT_FOOT_IDX, 1] = 1.0
    rigid_body_pos[0, _RIGHT_FOOT_IDX, 1] = -1.0  # 2.0m apart, >> threshold=0.2
    mc = _fake_motion_command(num_envs=1)
    env = _fake_env(mc, 1, rigid_body_pos=rigid_body_pos)
    with patch.object(s, "current_w_g", return_value=torch.ones(1)):
        out = s.penalize_self_contact_feet(env, threshold=0.2, std=0.05)
    assert out.item() == 0.0


def test_self_contact_grows_toward_10_as_feet_approach():
    mc = _fake_motion_command(num_envs=1)
    dists = [0.15, 0.05, 0.0]
    prev = -1.0
    for d in dists:
        rigid_body_pos = torch.zeros(1, len(_BODY_NAMES), 3)
        rigid_body_pos[0, _LEFT_FOOT_IDX, 1] = d / 2
        rigid_body_pos[0, _RIGHT_FOOT_IDX, 1] = -d / 2
        env = _fake_env(mc, 1, rigid_body_pos=rigid_body_pos)
        with patch.object(s, "current_w_g", return_value=torch.ones(1)):
            out = s.penalize_self_contact_feet(env, threshold=0.2, std=0.05)
        assert out.item() > prev, f"expected monotonic increase as feet approach, dist={d}"
        assert out.item() <= 10.0
        prev = out.item()


def test_self_contact_exact_formula_at_zero_distance():
    rigid_body_pos = torch.zeros(1, len(_BODY_NAMES), 3)  # both feet at the same point -> dist=0
    mc = _fake_motion_command(num_envs=1)
    env = _fake_env(mc, 1, rigid_body_pos=rigid_body_pos)
    with patch.object(s, "current_w_g", return_value=torch.ones(1)):
        out = s.penalize_self_contact_feet(env, threshold=0.2, std=0.05)
    expected = 10.0 * (1.0 - torch.exp(-torch.square(torch.tensor(0.0 - 0.2)) / 0.05**2))
    assert torch.isclose(out, expected, atol=1e-4)


# ============================================================================================
# robot_com_ball_distance / robot_torso_ball_distance
# ============================================================================================


def test_com_ball_distance_saturates_within_clamp_floor():
    root_states = torch.zeros(1, 13)  # CoM at origin
    mc = _fake_motion_command(num_envs=1)  # in_kicking_phase=True by default
    tracker = _FakeTracker(1, ball_pos_w=torch.zeros(1, 3))  # ball also at origin -> dist=0 < 0.25
    env = _fake_env(mc, 1, robot_root_states=root_states)
    with patch.object(s, "_tracker", return_value=tracker), patch.object(s, "current_w_g", return_value=torch.ones(1)):
        out = s.robot_com_ball_distance(env, std=0.5)
    assert torch.isclose(out, torch.tensor([1.0]), atol=1e-5), "within the 0.25m clamp floor -> saturated reward"


def test_com_ball_distance_decays_beyond_clamp_floor():
    root_states = torch.zeros(1, 13)
    mc = _fake_motion_command(num_envs=1)
    tracker = _FakeTracker(1, ball_pos_w=torch.tensor([[5.0, 0.0, 0.0]]))  # far away
    env = _fake_env(mc, 1, robot_root_states=root_states)
    with patch.object(s, "_tracker", return_value=tracker), patch.object(s, "current_w_g", return_value=torch.ones(1)):
        out = s.robot_com_ball_distance(env, std=0.5)
    assert out.item() < 1.0


def test_com_ball_distance_pinned_to_clamp_floor_once_has_kicked():
    root_states = torch.zeros(1, 13)
    mc = _fake_motion_command(num_envs=1)
    tracker = _FakeTracker(1, ball_pos_w=torch.tensor([[5.0, 0.0, 0.0]]), has_kicked=torch.ones(1, dtype=torch.bool))
    env = _fake_env(mc, 1, robot_root_states=root_states)
    with patch.object(s, "_tracker", return_value=tracker), patch.object(s, "current_w_g", return_value=torch.ones(1)):
        out = s.robot_com_ball_distance(env, std=0.5)
    assert torch.isclose(out, torch.tensor([1.0]), atol=1e-5), "has_kicked must pin dist to the clamp floor regardless of actual distance"


def test_com_ball_distance_zero_outside_kicking_phase():
    root_states = torch.zeros(1, 13)
    mc = _fake_motion_command(in_kicking_phase=torch.zeros(1, dtype=torch.bool), num_envs=1)
    tracker = _FakeTracker(1, ball_pos_w=torch.zeros(1, 3))
    env = _fake_env(mc, 1, robot_root_states=root_states)
    with patch.object(s, "_tracker", return_value=tracker), patch.object(s, "current_w_g", return_value=torch.ones(1)):
        out = s.robot_com_ball_distance(env, std=0.5)
    assert out.item() == 0.0


def test_torso_ball_distance_saturates_within_own_clamp_floor():
    rigid_body_pos = torch.zeros(1, len(_BODY_NAMES), 3)  # torso at origin
    mc = _fake_motion_command(num_envs=1)
    tracker = _FakeTracker(1, ball_pos_w=torch.zeros(1, 3))
    env = _fake_env(mc, 1, rigid_body_pos=rigid_body_pos)
    with patch.object(s, "_tracker", return_value=tracker), patch.object(s, "current_w_g", return_value=torch.ones(1)):
        out = s.robot_torso_ball_distance(env, std=0.5)
    assert torch.isclose(out, torch.tensor([1.0]), atol=1e-5)


# ============================================================================================
# ball_over_line
# ============================================================================================


def test_ball_over_line_zero_at_spawn():
    mc = _fake_motion_command(
        target_xy_w=torch.tensor([[5.0, 0.0]]), ball_spawn_pos_w=torch.tensor([[0.0, 0.0, 0.0]]), num_envs=1
    )
    tracker = _FakeTracker(1, ball_pos_w=torch.tensor([[0.0, 0.0, 0.0]]))  # at spawn -> projected=0
    env = _fake_env(mc, 1)
    with patch.object(s, "_tracker", return_value=tracker), patch.object(s, "current_w_g", return_value=torch.ones(1)):
        out = s.ball_over_line(env, over_line_dist=7.0, back_line_dist=-1.0)
    assert out.item() == 0.0


def test_ball_over_line_rewards_crossing_past_target():
    mc = _fake_motion_command(
        target_xy_w=torch.tensor([[5.0, 0.0]]), ball_spawn_pos_w=torch.tensor([[0.0, 0.0, 0.0]]), num_envs=1
    )
    tracker = _FakeTracker(1, ball_pos_w=torch.tensor([[8.0, 0.0, 0.0]]))  # projected = 8.0 > 7.0
    env = _fake_env(mc, 1)
    with patch.object(s, "_tracker", return_value=tracker), patch.object(s, "current_w_g", return_value=torch.ones(1)):
        out = s.ball_over_line(env, over_line_dist=7.0, back_line_dist=-1.0)
    assert out.item() == 2.0


def test_ball_over_line_penalizes_going_backward():
    mc = _fake_motion_command(
        target_xy_w=torch.tensor([[5.0, 0.0]]), ball_spawn_pos_w=torch.tensor([[0.0, 0.0, 0.0]]), num_envs=1
    )
    tracker = _FakeTracker(1, ball_pos_w=torch.tensor([[-2.0, 0.0, 0.0]]))  # projected = -2.0 < -1.0
    env = _fake_env(mc, 1)
    with patch.object(s, "_tracker", return_value=tracker), patch.object(s, "current_w_g", return_value=torch.ones(1)):
        out = s.ball_over_line(env, over_line_dist=7.0, back_line_dist=-1.0)
    assert out.item() == -1.0


def test_ball_over_line_neutral_in_between():
    mc = _fake_motion_command(
        target_xy_w=torch.tensor([[5.0, 0.0]]), ball_spawn_pos_w=torch.tensor([[0.0, 0.0, 0.0]]), num_envs=1
    )
    tracker = _FakeTracker(1, ball_pos_w=torch.tensor([[3.0, 0.0, 0.0]]))  # projected=3.0, between -1 and 7
    env = _fake_env(mc, 1)
    with patch.object(s, "_tracker", return_value=tracker), patch.object(s, "current_w_g", return_value=torch.ones(1)):
        out = s.ball_over_line(env, over_line_dist=7.0, back_line_dist=-1.0)
    assert out.item() == 0.0


def test_ball_over_line_per_env_independent():
    mc = _fake_motion_command(
        target_xy_w=torch.tensor([[5.0, 0.0], [5.0, 0.0]]),
        ball_spawn_pos_w=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        num_envs=2,
    )
    tracker = _FakeTracker(2, ball_pos_w=torch.tensor([[8.0, 0.0, 0.0], [-2.0, 0.0, 0.0]]))
    env = _fake_env(mc, 2)
    with patch.object(s, "_tracker", return_value=tracker), patch.object(s, "current_w_g", return_value=torch.ones(2)):
        out = s.ball_over_line(env, over_line_dist=7.0, back_line_dist=-1.0)
    assert torch.allclose(out, torch.tensor([2.0, -1.0]))


def test_ball_over_line_require_has_kicked_false_is_unchanged_default():
    """require_has_kicked defaults to False -- byte-identical to before this param existed, even
    when has_kicked itself is False (the ball moved without a confirmed kick)."""
    mc = _fake_motion_command(
        target_xy_w=torch.tensor([[5.0, 0.0]]), ball_spawn_pos_w=torch.tensor([[0.0, 0.0, 0.0]]), num_envs=1
    )
    tracker = _FakeTracker(1, ball_pos_w=torch.tensor([[8.0, 0.0, 0.0]]), has_kicked=torch.zeros(1, dtype=torch.bool))
    env = _fake_env(mc, 1)
    with patch.object(s, "_tracker", return_value=tracker), patch.object(s, "current_w_g", return_value=torch.ones(1)):
        out = s.ball_over_line(env, over_line_dist=7.0, back_line_dist=-1.0, require_has_kicked=False)
    assert out.item() == 2.0  # pays out regardless of has_kicked, same as before this param existed


def test_ball_over_line_require_has_kicked_true_zeroes_without_confirmed_kick():
    """The exact case this param exists to fix: significant ball displacement (e.g. from an
    accidental non-kick-foot nudge) with has_kicked still False -- must pay ZERO when gated."""
    mc = _fake_motion_command(
        target_xy_w=torch.tensor([[5.0, 0.0]]), ball_spawn_pos_w=torch.tensor([[0.0, 0.0, 0.0]]), num_envs=1
    )
    tracker = _FakeTracker(1, ball_pos_w=torch.tensor([[8.0, 0.0, 0.0]]), has_kicked=torch.zeros(1, dtype=torch.bool))
    env = _fake_env(mc, 1)
    with patch.object(s, "_tracker", return_value=tracker), patch.object(s, "current_w_g", return_value=torch.ones(1)):
        out = s.ball_over_line(env, over_line_dist=7.0, back_line_dist=-1.0, require_has_kicked=True)
    assert out.item() == 0.0


def test_ball_over_line_require_has_kicked_true_pays_out_with_confirmed_kick():
    """Same displacement as above, but has_kicked=True this time -- must pay out normally, proving
    the gate doesn't suppress the legitimate path."""
    mc = _fake_motion_command(
        target_xy_w=torch.tensor([[5.0, 0.0]]), ball_spawn_pos_w=torch.tensor([[0.0, 0.0, 0.0]]), num_envs=1
    )
    tracker = _FakeTracker(1, ball_pos_w=torch.tensor([[8.0, 0.0, 0.0]]), has_kicked=torch.ones(1, dtype=torch.bool))
    env = _fake_env(mc, 1)
    with patch.object(s, "_tracker", return_value=tracker), patch.object(s, "current_w_g", return_value=torch.ones(1)):
        out = s.ball_over_line(env, over_line_dist=7.0, back_line_dist=-1.0, require_has_kicked=True)
    assert out.item() == 2.0


def test_ball_over_line_require_has_kicked_true_zeroes_the_penalty_side_too():
    """The back_line (penalty) side must ALSO be gated -- an accidental backward nudge with no
    confirmed kick must not register as a penalty either, not just the reward side."""
    mc = _fake_motion_command(
        target_xy_w=torch.tensor([[5.0, 0.0]]), ball_spawn_pos_w=torch.tensor([[0.0, 0.0, 0.0]]), num_envs=1
    )
    tracker = _FakeTracker(1, ball_pos_w=torch.tensor([[-2.0, 0.0, 0.0]]), has_kicked=torch.zeros(1, dtype=torch.bool))
    env = _fake_env(mc, 1)
    with patch.object(s, "_tracker", return_value=tracker), patch.object(s, "current_w_g", return_value=torch.ones(1)):
        out = s.ball_over_line(env, over_line_dist=7.0, back_line_dist=-1.0, require_has_kicked=True)
    assert out.item() == 0.0


def test_ball_over_line_require_has_kicked_per_env_independent():
    mc = _fake_motion_command(
        target_xy_w=torch.tensor([[5.0, 0.0], [5.0, 0.0]]),
        ball_spawn_pos_w=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        num_envs=2,
    )
    tracker = _FakeTracker(
        2,
        ball_pos_w=torch.tensor([[8.0, 0.0, 0.0], [8.0, 0.0, 0.0]]),
        has_kicked=torch.tensor([True, False]),
    )
    env = _fake_env(mc, 2)
    with patch.object(s, "_tracker", return_value=tracker), patch.object(s, "current_w_g", return_value=torch.ones(2)):
        out = s.ball_over_line(env, over_line_dist=7.0, back_line_dist=-1.0, require_has_kicked=True)
    assert torch.allclose(out, torch.tensor([2.0, 0.0]))


def test_ball_over_line_require_has_kicked_per_env_tensor_selects_per_skill():
    """2026-08-15, 'simultaneous per-skill task configs': env 0's skill has
    require_has_kicked=True (gated -- has_kicked=False here -> pays 0), env 1's skill has it
    False (ungated -- pays out regardless of has_kicked) -- both envs share the SAME ball
    displacement and has_kicked=False, only the per-env flag differs, isolating the select."""
    mc = _fake_motion_command(
        target_xy_w=torch.tensor([[5.0, 0.0], [5.0, 0.0]]),
        ball_spawn_pos_w=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        num_envs=2,
    )
    tracker = _FakeTracker(
        2,
        ball_pos_w=torch.tensor([[8.0, 0.0, 0.0], [8.0, 0.0, 0.0]]),
        has_kicked=torch.zeros(2, dtype=torch.bool),
    )
    env = _fake_env(mc, 2)
    with patch.object(s, "_tracker", return_value=tracker), patch.object(s, "current_w_g", return_value=torch.ones(2)):
        out = s.ball_over_line(
            env, over_line_dist=7.0, back_line_dist=-1.0, require_has_kicked=torch.tensor([1.0, 0.0])
        )
    assert torch.allclose(out, torch.tensor([0.0, 2.0])), f"got {out}"
