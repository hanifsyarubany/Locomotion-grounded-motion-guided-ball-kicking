"""Unit tests for SimpleReplayBuffer's is_kick field (2026-07-28, see
FastSACConfig.kick_target_entropy_ratio's docstring for the full feature). Verifies is_kick is
stored and gathered CONSISTENTLY with the transition it belongs to -- i.e. sample() never
mismatches an is_kick value against a different transition's observations/actions/etc -- for both
the n_steps==1 and n_steps>1 code paths, which have separate gather logic.
"""

from __future__ import annotations

import sys

import pytest
import torch
from tensordict import TensorDict

sys.path.insert(0, "/workspaces/isaaclab_arena/submodules/workspaces/playground/unified_ball_kick_robonaldo/src/holosoma")

from holosoma.agents.fast_sac.fast_sac_utils import SimpleReplayBuffer


def _make_buffer(n_env=4, buffer_size=32, n_steps=1):
    return SimpleReplayBuffer(
        n_env=n_env, buffer_size=buffer_size, n_obs=3, n_act=2, n_critic_obs=3, n_steps=n_steps, device="cpu"
    )


def _fill_with_marker_pattern(rb: SimpleReplayBuffer, n_env: int, n_transitions: int):
    """Each transition's observations[:, 0] == its own buffer slot index (a unique marker), and
    is_kick alternates 0/1/0/1... by slot -- so after sampling, we can verify is_kick always
    matches observations[:, 0] % 2 exactly, proving the gather never mismatches transitions."""
    for slot in range(n_transitions):
        obs = torch.full((n_env, 3), float(slot))
        is_kick = torch.full((n_env,), slot % 2, dtype=torch.long)
        transition = TensorDict(
            {
                "observations": obs,
                "actions": torch.zeros(n_env, 2),
                "is_kick": is_kick,
                "next": {
                    "observations": obs + 1,
                    "rewards": torch.zeros(n_env),
                    "dones": torch.zeros(n_env, dtype=torch.long),
                    "truncations": torch.zeros(n_env, dtype=torch.long),
                },
            },
            batch_size=(n_env,),
        )
        transition["critic_observations"] = obs
        transition["next"]["critic_observations"] = obs + 1
        rb.extend(transition)


def test_is_kick_defaults_to_zero_before_any_extend():
    rb = _make_buffer()
    assert torch.all(rb.is_kick == 0)


def test_n_steps_1_is_kick_matches_its_own_transitions_marker():
    n_env = 4
    rb = _make_buffer(n_env=n_env, buffer_size=32, n_steps=1)
    _fill_with_marker_pattern(rb, n_env, n_transitions=20)

    for _ in range(20):  # many samples to exercise the random indices broadly
        batch = rb.sample(batch_size=8)
        marker = batch["observations"][:, 0]
        expected_is_kick = (marker.long() % 2)
        assert torch.equal(batch["is_kick"], expected_is_kick), (
            f"is_kick mismatched its own transition's marker: is_kick={batch['is_kick']}, "
            f"marker%2={expected_is_kick}"
        )


def test_n_steps_greater_than_1_is_kick_matches_its_own_transitions_marker():
    n_env = 4
    rb = _make_buffer(n_env=n_env, buffer_size=32, n_steps=3)
    _fill_with_marker_pattern(rb, n_env, n_transitions=20)

    for _ in range(20):
        batch = rb.sample(batch_size=8)
        marker = batch["observations"][:, 0]
        expected_is_kick = (marker.long() % 2)
        assert torch.equal(batch["is_kick"], expected_is_kick), (
            f"is_kick mismatched its own transition's marker (n_steps=3): is_kick={batch['is_kick']}, "
            f"marker%2={expected_is_kick}"
        )


def test_is_kick_all_kick_when_all_transitions_are_kick():
    n_env = 4
    rb = _make_buffer(n_env=n_env, buffer_size=16, n_steps=1)
    for slot in range(10):
        obs = torch.full((n_env, 3), float(slot))
        transition = TensorDict(
            {
                "observations": obs,
                "actions": torch.zeros(n_env, 2),
                "is_kick": torch.ones(n_env, dtype=torch.long),
                "next": {
                    "observations": obs + 1,
                    "rewards": torch.zeros(n_env),
                    "dones": torch.zeros(n_env, dtype=torch.long),
                    "truncations": torch.zeros(n_env, dtype=torch.long),
                },
            },
            batch_size=(n_env,),
        )
        transition["critic_observations"] = obs
        transition["next"]["critic_observations"] = obs + 1
        rb.extend(transition)

    batch = rb.sample(batch_size=16)
    assert torch.all(batch["is_kick"] == 1)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
