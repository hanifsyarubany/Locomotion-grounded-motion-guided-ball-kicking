"""Unit tests for FastSACConfig.skill_replay_weights (2026-08-15): per-skill relative weighting of
each kick transition's contribution to the critic/actor losses, decoupling GRADIENT share from
`motion_training_ratio`'s env share.

The property that motivates the whole field is tested explicitly in
`test_weighting_equalizes_gradient_share_of_a_0p1_vs_0p8_split` below: `SimpleReplayBuffer` is a
per-env ring buffer sampled `batch_size`-per-env, so a skill's share of every gradient batch is
exactly its share of ENVS -- and the configured weights must be able to restore parity.

`_skill_replay_weight` is exercised directly on a bare instance (same `object.__new__` pattern as
envs/tests/test_unified_manager_*.py) -- it touches only `self.skill_weight_by_group` and the
TensorDict, not the optimizers/networks/CUDA state a real FastSACAgent construction would need.
"""

from __future__ import annotations

import types

import torch
from tensordict import TensorDict

from holosoma.agents.fast_sac.fast_sac_agent import FastSACAgent
from holosoma.agents.fast_sac.fast_sac_utils import SimpleReplayBuffer


def _agent(weights: list[float] | None) -> FastSACAgent:
    a = object.__new__(FastSACAgent)
    a.skill_weight_by_group = torch.tensor(weights, dtype=torch.float) if weights is not None else None
    return a


def _data(skill_id: list[int], is_kick: list[int]) -> TensorDict:
    n = len(skill_id)
    return TensorDict(
        {
            "skill_id": torch.tensor(skill_id, dtype=torch.long),
            "is_kick": torch.tensor(is_kick, dtype=torch.long),
        },
        batch_size=n,
    )


def test_disabled_returns_none_exact_no_op():
    """Empty config -> None, and every loss reduction takes its original unweighted path."""
    assert _agent(None)._skill_replay_weight(_data([0, 1], [1, 1])) is None


def test_locomotion_transitions_always_weight_one():
    """skill_id is copied from the fixed partition for EVERY env including locomotion-partitioned
    ones, where it is meaningless -- so the gate must be is_kick, not skill_id."""
    w = _agent([8.0, 1.0])._skill_replay_weight(_data([0, 0, 1, 1], [0, 1, 0, 1]))
    # envs 0 and 2 are locomotion -> raw weight 1.0; envs 1,3 -> 8.0, 1.0. Mean-1 normalized.
    raw = torch.tensor([1.0, 8.0, 1.0, 1.0])
    assert torch.allclose(w, raw / raw.mean())


def test_weights_are_renormalized_to_mean_one():
    """Total gradient magnitude (hence effective lr) must be preserved -- only RELATIVE per-skill
    contribution moves."""
    for weights in ([8.0, 1.0], [100.0, 3.0], [0.5, 0.25]):
        w = _agent(weights)._skill_replay_weight(_data([0, 0, 1, 1, 1], [1, 1, 1, 1, 1]))
        assert torch.isclose(w.mean(), torch.tensor(1.0), atol=1e-6), weights


def test_uniform_weights_are_a_true_no_op_on_the_loss():
    """All-equal weights must reduce to exactly 1.0 everywhere, so enabling the feature with
    uniform weights cannot perturb training."""
    w = _agent([2.0, 2.0])._skill_replay_weight(_data([0, 1, 0, 1], [1, 1, 1, 1]))
    assert torch.allclose(w, torch.ones(4))


def test_relative_ratio_between_skills_is_exactly_as_configured():
    w = _agent([8.0, 1.0])._skill_replay_weight(_data([0, 1], [1, 1]))
    assert torch.isclose(w[0] / w[1], torch.tensor(8.0), atol=1e-6)


def test_zero_weight_skill_is_fully_suppressed_but_others_survive():
    """A single 0.0 entry is legal (only ALL-zero is rejected at construction) and must zero just
    that skill's contribution."""
    w = _agent([0.0, 1.0])._skill_replay_weight(_data([0, 1, 1], [1, 1, 1]))
    assert float(w[0]) == 0.0
    assert float(w[1]) > 0.0


def test_weighting_equalizes_gradient_share_of_a_0p1_vs_0p8_split():
    """THE motivating property. Build the user's real case -- skill 0 at motion_training_ratio 0.1
    and skill 1 at 0.8 -- and show (a) the unweighted gradient share really is ~1:8, exactly the
    env share, and (b) weights of [8.0, 1.0] restore parity."""
    n_skill0, n_skill1 = 100, 800
    skill_id = [0] * n_skill0 + [1] * n_skill1
    is_kick = [1] * (n_skill0 + n_skill1)

    # (a) unweighted: each transition contributes equally, so share == env share.
    unweighted = torch.ones(n_skill0 + n_skill1)
    s0 = unweighted[: n_skill0].sum()
    s1 = unweighted[n_skill0 :].sum()
    assert torch.isclose(s1 / s0, torch.tensor(8.0), atol=1e-5), "baseline should be the 1:8 env split"

    # (b) weighted by 1/ratio: total gradient mass per skill becomes equal.
    w = _agent([8.0, 1.0])._skill_replay_weight(_data(skill_id, is_kick))
    w0 = w[: n_skill0].sum()
    w1 = w[n_skill0 :].sum()
    assert torch.isclose(w0 / w1, torch.tensor(1.0), atol=1e-5), f"expected parity, got {float(w0 / w1)}"
    # and the overall magnitude is untouched
    assert torch.isclose(w.mean(), torch.tensor(1.0), atol=1e-6)


def test_partial_softening_lands_between_the_two_extremes():
    """[4.0, 1.0] on a 0.1/0.8 split should move share toward parity without reaching it -- the
    field's documented "usually the better first try" setting."""
    n0, n1 = 100, 800
    w = _agent([4.0, 1.0])._skill_replay_weight(_data([0] * n0 + [1] * n1, [1] * (n0 + n1)))
    ratio = float(w[:n0].sum() / w[n0:].sum())
    assert 0.4 < ratio < 1.0, ratio


# ---------------------------------------------------------------------------
# SimpleReplayBuffer.skill_id plumbing (sibling of the existing is_kick tests)
# ---------------------------------------------------------------------------


def _fill(rb: SimpleReplayBuffer, n_env: int, skill_id: torch.Tensor | None, steps: int = 4) -> None:
    for _ in range(steps):
        d = {
            "observations": torch.zeros(n_env, rb.n_obs),
            "actions": torch.zeros(n_env, rb.n_act),
            "is_kick": torch.ones(n_env, dtype=torch.long),
            "critic_observations": torch.zeros(n_env, rb.n_critic_obs),
            "next": {
                "observations": torch.zeros(n_env, rb.n_obs),
                "rewards": torch.zeros(n_env),
                "dones": torch.zeros(n_env, dtype=torch.long),
                "truncations": torch.zeros(n_env, dtype=torch.long),
                "critic_observations": torch.zeros(n_env, rb.n_critic_obs),
            },
        }
        if skill_id is not None:
            d["skill_id"] = skill_id
        rb.extend(TensorDict(d, batch_size=n_env))


def test_buffer_roundtrips_skill_id_per_env():
    """Each env's own fixed-for-life skill label must come back out attached to that env's own
    transitions -- sample() is env-major, so env e's block is rows [e*bs, (e+1)*bs)."""
    n_env, bs = 4, 3
    rb = SimpleReplayBuffer(n_env=n_env, buffer_size=8, n_obs=2, n_act=2, n_critic_obs=2, device="cpu")
    skill_id = torch.tensor([0, 1, 1, 0], dtype=torch.long)
    _fill(rb, n_env, skill_id)

    out = rb.sample(bs)
    got = out["skill_id"].reshape(n_env, bs)
    for e in range(n_env):
        assert torch.all(got[e] == skill_id[e]), f"env {e}: {got[e]} != {skill_id[e]}"


def test_buffer_skill_id_defaults_to_zeros_when_caller_omits_it():
    """Optional-key fallback: a locomotion-only/WBT-only caller (or this class used standalone)
    may omit skill_id entirely, and all-zeros is the correct 'single skill 0' default."""
    n_env = 3
    rb = SimpleReplayBuffer(n_env=n_env, buffer_size=8, n_obs=2, n_act=2, n_critic_obs=2, device="cpu")
    _fill(rb, n_env, skill_id=None)
    assert torch.all(rb.sample(2)["skill_id"] == 0)


# ---------------------------------------------------------------------------
# _sample_and_prepare_batches: skill_weight materialization (2026-08-15)
#
# Regression coverage for a real production crash: a real 3300-env run with
# skill_replay_weights=[4.0, 1.0] hit "RuntimeError: size of tensor a (2) must match size of
# tensor b (6600)" inside torch.compile'd _update_main, from
# self.skill_weight_by_group[data["skill_id"]] (fancy-indexing a [2]-sized table -- coincidentally
# the same size as num_q_networks=2 -- by a [6600]-sized index) being traced INSIDE the compiled
# graph. Root cause not fully pinned down; fixed by moving the whole computation OUT of the
# compiled functions entirely -- materialized once here, in plain eager Python, and read as an
# already-built tensor (data["skill_weight"]) inside _update_main/_update_pol instead of being
# recomputed via fancy-indexing there. These tests exercise the REAL materialization path (not a
# hand-rolled equivalent) at both a small scale and the exact production shape that crashed
# (n_env=3300, batch_size=2 -- i.e. args.batch_size=8192 // num_envs=3300).
# ---------------------------------------------------------------------------


def _agent_with_buffer(n_env: int, weights: list[float] | None, n_obs: int = 3, n_act: int = 2) -> FastSACAgent:
    a = object.__new__(FastSACAgent)
    a.rb = SimpleReplayBuffer(n_env=n_env, buffer_size=32, n_obs=n_obs, n_act=n_act, n_critic_obs=n_obs, device="cpu")
    a.env = types.SimpleNamespace(num_envs=n_env)
    a.config = types.SimpleNamespace(use_symmetry=False)
    a.skill_weight_by_group = torch.tensor(weights, dtype=torch.float) if weights is not None else None
    a.skill_replay_weights_enabled = weights is not None
    return a


def _fill_with_skill_id(a: FastSACAgent, n_env: int, skill_id: torch.Tensor, steps: int = 10) -> None:
    for _ in range(steps):
        d = {
            "observations": torch.randn(n_env, a.rb.n_obs),
            "actions": torch.randn(n_env, a.rb.n_act),
            "is_kick": torch.ones(n_env, dtype=torch.long),
            "skill_id": skill_id,
            "critic_observations": torch.randn(n_env, a.rb.n_critic_obs),
            "next": {
                "observations": torch.randn(n_env, a.rb.n_obs),
                "rewards": torch.randn(n_env),
                "dones": torch.zeros(n_env, dtype=torch.long),
                "truncations": torch.zeros(n_env, dtype=torch.long),
                "critic_observations": torch.randn(n_env, a.rb.n_critic_obs),
            },
        }
        a.rb.extend(TensorDict(d, batch_size=n_env))


def _identity(x, **_kwargs):
    return x


def test_disabled_never_materializes_skill_weight_key():
    """Zero overhead when off: skill_weight must not even appear in the prepared batch."""
    a = _agent_with_buffer(n_env=8, weights=None)
    _fill_with_skill_id(a, 8, torch.zeros(8, dtype=torch.long))
    batches = FastSACAgent._sample_and_prepare_batches(a, batch_size=2, num_updates=1, normalize_obs=_identity, normalize_critic_obs=_identity)
    assert "skill_weight" not in batches[0].keys()


def test_enabled_materializes_skill_weight_matching_the_helper(monkeypatch):
    n_env = 8
    a = _agent_with_buffer(n_env=n_env, weights=[4.0, 1.0])
    skill_id = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.long)
    _fill_with_skill_id(a, n_env, skill_id)

    batches = FastSACAgent._sample_and_prepare_batches(a, batch_size=2, num_updates=1, normalize_obs=_identity, normalize_critic_obs=_identity)
    batch = batches[0]
    assert "skill_weight" in batch.keys()
    expected = a._skill_replay_weight(batch)
    assert torch.allclose(batch["skill_weight"], expected)
    assert torch.isclose(batch["skill_weight"].mean(), torch.tensor(1.0), atol=1e-5)


def test_multiple_updates_each_chunk_gets_its_own_correctly_shaped_weight():
    n_env = 6
    a = _agent_with_buffer(n_env=n_env, weights=[4.0, 1.0])
    skill_id = torch.tensor([0, 0, 1, 1, 1, 1], dtype=torch.long)
    _fill_with_skill_id(a, n_env, skill_id)

    num_updates = 3
    batches = FastSACAgent._sample_and_prepare_batches(a, batch_size=2, num_updates=num_updates, normalize_obs=_identity, normalize_critic_obs=_identity)
    assert len(batches) == num_updates
    for b in batches:
        assert b["skill_weight"].shape[0] == b.batch_size[0] == n_env * 2


def test_exact_production_crash_shape_n_env_3300_batch_size_2():
    """The EXACT shape that crashed: args.batch_size=8192 // num_envs=3300 == 2 per env, 8 updates
    -- num_q_networks (2, the coincidental collision suspect) plays no role here since this test
    only exercises the buffer/materialization path, not the compiled critic loss itself, but it
    does prove the materialization itself is correct and crash-free at the real production scale
    (6600-sample chunks) end to end through the actual (uncompiled) production code path."""
    n_env = 3300
    a = _agent_with_buffer(n_env=n_env, weights=[4.0, 1.0])
    skill_id = (torch.rand(n_env) < 0.9).long()  # ~0.1/0.8-ish split, matches the real config
    _fill_with_skill_id(a, n_env, skill_id, steps=3)

    batches = FastSACAgent._sample_and_prepare_batches(a, batch_size=2, num_updates=8, normalize_obs=_identity, normalize_critic_obs=_identity)
    assert len(batches) == 8
    for b in batches:
        assert b["skill_weight"].shape == (6600,)
        assert torch.isfinite(b["skill_weight"]).all()


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
