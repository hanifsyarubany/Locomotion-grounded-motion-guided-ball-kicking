"""Unit tests for SimpleReplayBuffer's sanitize_enabled option (2026-08-10, see
FastSACConfig.replay_buffer_sanitize_enabled's own docstring for the full feature/rationale --
a last-line-of-defense NaN/Inf guard at the replay buffer's write boundary, against a rare
per-env physics-solver numerical explosion observed corrupting a real Stage C run).

Same TensorDict/_make_buffer pattern as test_replay_buffer_is_kick.py.
"""

from __future__ import annotations

import sys

import pytest
import torch
from tensordict import TensorDict

sys.path.insert(0, "/workspaces/isaaclab_arena/submodules/workspaces/playground/unified_ball_kick_robonaldo/src/holosoma")

from holosoma.agents.fast_sac.fast_sac_utils import SimpleReplayBuffer


def _make_buffer(n_env=4, buffer_size=32, sanitize_enabled=False):
    return SimpleReplayBuffer(
        n_env=n_env,
        buffer_size=buffer_size,
        n_obs=3,
        n_act=2,
        n_critic_obs=3,
        n_steps=1,
        device="cpu",
        sanitize_enabled=sanitize_enabled,
    )


def _corrupted_transition(n_env: int, n_obs: int = 3, n_act: int = 2) -> TensorDict:
    """One env's observation/reward/next_observation/critic_observations are NaN or Inf; the
    other envs are clean -- mirrors the real incident (one exploded env among thousands)."""
    obs = torch.zeros(n_env, n_obs)
    obs[0, 0] = float("nan")
    obs[1, 1] = float("inf")
    rewards = torch.zeros(n_env)
    rewards[0] = float("nan")
    next_obs = obs.clone() + 1.0
    next_obs[2, 2] = float("-inf")
    critic_obs = obs.clone()
    next_critic_obs = next_obs.clone()

    transition = TensorDict(
        {
            "observations": obs,
            "actions": torch.zeros(n_env, n_act),
            "is_kick": torch.zeros(n_env, dtype=torch.long),
            "next": {
                "observations": next_obs,
                "rewards": rewards,
                "dones": torch.zeros(n_env, dtype=torch.long),
                "truncations": torch.zeros(n_env, dtype=torch.long),
            },
        },
        batch_size=(n_env,),
    )
    transition["critic_observations"] = critic_obs
    transition["next"]["critic_observations"] = next_critic_obs
    return transition


def test_disabled_is_an_exact_no_op_nan_inf_pass_through():
    """Regression guard: sanitize_enabled=False (the default) must be byte-identical to before
    this feature existed -- NaN/Inf written verbatim, no silent behavior change for anyone who
    hasn't opted in."""
    rb = _make_buffer(sanitize_enabled=False)
    rb.extend(_corrupted_transition(n_env=4))

    assert torch.isnan(rb.observations[0, 0, 0])
    assert torch.isinf(rb.observations[1, 0, 1])
    assert torch.isnan(rb.rewards[0, 0])
    assert torch.isinf(rb.next_observations[2, 0, 2])
    assert torch.isnan(rb.critic_observations[0, 0, 0])


def test_enabled_replaces_nan_and_inf_with_finite_values():
    rb = _make_buffer(sanitize_enabled=True)
    rb.extend(_corrupted_transition(n_env=4))

    assert torch.isfinite(rb.observations[:, 0]).all(), "observations must be all-finite after sanitize"
    assert torch.isfinite(rb.rewards[:, 0]).all(), "rewards must be all-finite after sanitize"
    assert torch.isfinite(rb.next_observations[:, 0]).all(), "next_observations must be all-finite after sanitize"
    assert torch.isfinite(rb.critic_observations[:, 0]).all(), "critic_observations must be all-finite after sanitize"
    assert torch.isfinite(rb.next_critic_observations[:, 0]).all(), (
        "next_critic_observations must be all-finite after sanitize"
    )


def test_enabled_leaves_clean_envs_numerically_unchanged():
    """Sanitizing must not perturb envs that were never corrupted -- only the exact NaN/Inf
    entries change, everything else passes through exactly."""
    rb = _make_buffer(sanitize_enabled=True)
    transition = _corrupted_transition(n_env=4)
    clean_obs_env3 = transition["observations"][3].clone()
    rb.extend(transition)

    assert torch.equal(rb.observations[3, 0], clean_obs_env3), "a clean env's observation must be untouched"


def test_enabled_does_not_touch_actions_dones_truncations_is_kick():
    """Deliberate scope decision (see FastSACConfig.replay_buffer_sanitize_enabled's docstring):
    only the environment-sourced, continuous-valued tensors are sanitized -- actions come from the
    already-clipped policy and dones/truncations/is_kick are integer/bool flags with no blowup
    mode, so they're passed through raw even when sanitize_enabled=True."""
    rb = _make_buffer(n_env=2, sanitize_enabled=True)
    obs = torch.zeros(2, 3)
    actions = torch.full((2, 2), float("nan"))  # would never happen for real, but proves scope
    transition = TensorDict(
        {
            "observations": obs,
            "actions": actions,
            "is_kick": torch.zeros(2, dtype=torch.long),
            "next": {
                "observations": obs + 1,
                "rewards": torch.zeros(2),
                "dones": torch.zeros(2, dtype=torch.long),
                "truncations": torch.zeros(2, dtype=torch.long),
            },
        },
        batch_size=(2,),
    )
    transition["critic_observations"] = obs
    transition["next"]["critic_observations"] = obs + 1
    rb.extend(transition)

    assert torch.isnan(rb.actions[:, 0]).all(), "actions must be passed through untouched, even NaN"


def test_sample_after_sanitize_never_returns_non_finite_values():
    """End-to-end: sample() output must be finite too, not just the raw stored tensors -- proves
    the fix actually reaches what the learner consumes."""
    rb = _make_buffer(n_env=4, buffer_size=16, sanitize_enabled=True)
    for _ in range(10):
        rb.extend(_corrupted_transition(n_env=4))

    for _ in range(10):
        batch = rb.sample(batch_size=8)
        assert torch.isfinite(batch["observations"]).all()
        assert torch.isfinite(batch["next"]["rewards"]).all()
        assert torch.isfinite(batch["next"]["observations"]).all()
        assert torch.isfinite(batch["critic_observations"]).all()
        assert torch.isfinite(batch["next"]["critic_observations"]).all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
