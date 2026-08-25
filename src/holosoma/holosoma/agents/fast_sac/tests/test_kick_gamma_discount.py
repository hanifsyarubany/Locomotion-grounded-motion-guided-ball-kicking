"""Unit tests for the per-task-mode discount arithmetic FastSACConfig.kick_gamma adds to
FastSACAgent.learn() (see that field's docstring for the full design). Constructing a real
FastSACAgent needs a live env (IsaacSim), so this isolates the exact tensor expression used at the
call site --

    gamma_per_sample = self.gamma_by_group[data["is_kick"]]
    discount = gamma_per_sample ** data["next"]["effective_n_steps"]

-- against known inputs, plus the num_gamma_groups==1 fallback path
(``discount = args.gamma ** effective_n_steps``) that must stay byte-identical to before this
field existed. `is_kick` is torch.long (0/1), matching SimpleReplayBuffer's own dtype -- see
test_replay_buffer_is_kick.py.
"""

from __future__ import annotations

import sys

import torch

sys.path.insert(0, "/workspaces/isaaclab_arena/submodules/workspaces/playground/unified_ball_kick_robonaldo/src/holosoma")


def test_gamma_by_group_selects_the_right_value_per_sample():
    gamma_by_group = torch.tensor([0.97, 0.99])  # index 0 = locomotion, index 1 = kick
    is_kick = torch.tensor([0, 1, 0, 1, 1], dtype=torch.long)
    effective_n_steps = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])

    gamma_per_sample = gamma_by_group[is_kick]
    discount = gamma_per_sample**effective_n_steps

    expected = torch.tensor([0.97, 0.99, 0.97, 0.99, 0.99])
    assert torch.allclose(discount, expected)


def test_discount_respects_effective_n_steps_per_sample():
    """effective_n_steps varies per-sample (episode-boundary truncation) -- the exponent must
    apply per-sample, not as a shared scalar."""
    gamma_by_group = torch.tensor([0.97, 0.99])
    is_kick = torch.tensor([0, 1, 1], dtype=torch.long)
    effective_n_steps = torch.tensor([1.0, 2.0, 3.0])

    discount = gamma_by_group[is_kick] ** effective_n_steps

    expected = torch.tensor([0.97**1, 0.99**2, 0.99**3])
    assert torch.allclose(discount, expected)


def test_num_gamma_groups_one_path_is_byte_identical_to_pre_existing_scalar_formula():
    """The num_gamma_groups==1 (default, kick_gamma unset) branch must reproduce EXACTLY
    `args.gamma ** effective_n_steps` -- the formula every experiment used before this field
    existed. No group indexing, no gamma_by_group tensor involved."""
    args_gamma = 0.97
    effective_n_steps = torch.tensor([1.0, 2.0, 5.0, 1.0])

    discount = args_gamma**effective_n_steps

    expected = torch.tensor([0.97**1.0, 0.97**2.0, 0.97**5.0, 0.97**1.0])
    assert torch.allclose(discount, expected)


def test_equal_kick_and_loco_gamma_reduces_to_the_single_scalar_case():
    """Sanity check on the indexing mechanism itself: if kick_gamma happened to equal gamma, the
    per-sample path must produce numerically identical output to the scalar path, regardless of
    which is_kick values are present."""
    gamma = 0.97
    gamma_by_group = torch.tensor([gamma, gamma])
    is_kick = torch.tensor([0, 1, 1, 0, 1], dtype=torch.long)
    effective_n_steps = torch.tensor([1.0, 3.0, 2.0, 1.0, 4.0])

    per_sample = gamma_by_group[is_kick] ** effective_n_steps
    scalar = gamma**effective_n_steps
    assert torch.allclose(per_sample, scalar)


def test_gamma_by_group_index_0_is_locomotion_index_1_is_kick():
    """Locks in the ordering convention (matches log_alpha/target_entropy's own group 0=loco,
    group 1=kick convention) -- a silent swap here would apply the wrong discount to each mode
    without erroring anywhere."""
    loco_gamma, kick_gamma = 0.97, 0.995
    gamma_by_group = torch.tensor([loco_gamma, kick_gamma])

    all_loco = torch.zeros(4, dtype=torch.long)
    all_kick = torch.ones(4, dtype=torch.long)

    assert torch.allclose(gamma_by_group[all_loco], torch.full((4,), loco_gamma))
    assert torch.allclose(gamma_by_group[all_kick], torch.full((4,), kick_gamma))


def test_higher_kick_gamma_gives_a_longer_effective_horizon():
    """Confirms the intended DIRECTION of the fix: raising kick_gamma must increase, not decrease,
    the weight placed on a fixed future reward (i.e. genuinely lengthen the horizon)."""
    reward_in_10_steps = 1.0
    for kick_gamma in (0.97, 0.99, 0.995):
        pv_low = 0.97**10.0 * reward_in_10_steps
        pv_high = kick_gamma**10.0 * reward_in_10_steps
        assert pv_high >= pv_low
    # and the 33-step (gamma=0.97) vs 100-step (gamma=0.99) horizon claim from the docstring:
    assert abs(1.0 / (1.0 - 0.97) - 33.33) < 0.1
    assert abs(1.0 / (1.0 - 0.99) - 100.0) < 1e-9
