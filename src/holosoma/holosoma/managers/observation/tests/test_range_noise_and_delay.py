"""Unit tests for ObservationManager's perception-fidelity mechanisms
(config_types/observation.py's ``noise_range_coefficient``/``delay_step_range``/
``hold_step_range``, managers/observation/manager.py's ``_apply_noise`` extension and
``_apply_delay``/``_apply_hold``):

1. Range-dependent noise: additional per-step noise magnitude that scales with the observation's
   own L2 norm (e.g. distance-dependent depth-sensor error), on top of the existing flat ``noise``.
2. Latency: a per-env, per-episode-fixed random delay (in control steps) on a term's value,
   mirroring managers/action/terms/joint_control.py's action-delay queue mechanism exactly (same
   rolling-buffer-plus-per-env-index shape) but for an observation instead of an action.
3. Zero-order hold: a term's value only refreshes every N control steps instead of every tick,
   models a fusion pipeline whose own update rate is slower than the control loop (2026-08-20).
4. Stale-frame reuse: a per-CONTROL-TICK (re-rolled every step, unlike hold's per-episode-fixed
   period) chance of reusing the previous tick's already-fully-processed value -- models a single
   dropped/repeated sensor frame, transient and self-correcting (2026-08-23, ``_apply_stale``).

All four are gated on the SAME ``group.enable_noise`` flag the pre-existing flat ``noise``
already uses, so a privileged/critic group (``enable_noise=False``) must be completely unaffected
by any of them -- exercised directly below, not just asserted.

Uses a REAL ObservationManager against a minimal fake env (only what the manager itself needs:
``num_envs``, ``device``) and trivial, fully-controllable term functions (reading a mutable
attribute the test sets before each compute) -- this exercises the actual generic manager
mechanism directly, not a reimplementation of it, without needing to fake an entire simulator to
test math that has nothing to do with ball_pos_b's own physics.
"""

from __future__ import annotations

import torch

from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg
from holosoma.managers.observation.manager import ObservationManager


class _FakeEnv:
    def __init__(self, num_envs: int, device: str = "cpu"):
        self.num_envs = num_envs
        self.device = device
        self.current_value = torch.zeros(num_envs, 3)


def _controllable_term(env: _FakeEnv) -> torch.Tensor:
    """Returns whatever env.current_value currently holds -- fully test-controlled, so the raw
    (pre-noise, pre-delay) value at each simulated step is exactly known."""
    return env.current_value.clone()


def _make_manager(
    *,
    noise: float = 0.0,
    noise_range_coefficient: float = 0.0,
    delay_step_range: tuple[int, int] | None = None,
    hold_step_range: tuple[int, int] | None = None,
    stale_probability: float = 0.0,
    enable_noise: bool = True,
    num_envs: int = 8,
) -> tuple[ObservationManager, _FakeEnv]:
    env = _FakeEnv(num_envs)
    term_cfg = ObsTermCfg(
        func="holosoma.managers.observation.tests.test_range_noise_and_delay:_controllable_term",
        noise=noise,
        noise_range_coefficient=noise_range_coefficient,
        delay_step_range=delay_step_range,
        hold_step_range=hold_step_range,
        stale_probability=stale_probability,
    )
    cfg = ObservationManagerCfg(
        groups={
            "actor_obs": ObsGroupCfg(terms={"val": term_cfg}, concatenate=False, enable_noise=enable_noise),
        }
    )
    manager = ObservationManager(cfg, env, device="cpu")
    return manager, env


# ---------------------------------------------------------------------------------------------
# Range-dependent noise
# ---------------------------------------------------------------------------------------------


def test_range_coefficient_zero_is_bit_identical_to_flat_noise_only():
    """noise_range_coefficient=0.0 (the default for every term registered before this field
    existed) must reduce the effective noise scale to exactly `noise`, no behavior change."""
    manager, env = _make_manager(noise=0.2, noise_range_coefficient=0.0)
    env.current_value[:] = torch.tensor([5.0, 0.0, 0.0])  # large norm -- would matter if coefficient leaked in
    torch.manual_seed(0)
    out = manager.compute_group("actor_obs")["val"]
    deviation = (out - env.current_value).abs()
    assert torch.all(deviation <= 0.2 + 1e-6), "deviation exceeded the flat noise bound"


def test_range_dependent_noise_grows_with_observation_magnitude():
    """A far observation (large L2 norm) must show a larger MAXIMUM possible deviation than a
    near one, given the same noise/noise_range_coefficient config -- sampled many times to make
    the statistical claim decisive rather than a single lucky/unlucky draw."""
    manager, env = _make_manager(noise=0.05, noise_range_coefficient=0.1, num_envs=1)

    torch.manual_seed(0)
    near_devs = []
    env.current_value[:] = torch.tensor([1.0, 0.0, 0.0])  # norm = 1 -> effective scale = 0.05 + 0.1*1 = 0.15
    for _ in range(200):
        out = manager.compute_group("actor_obs")["val"]
        near_devs.append((out - env.current_value).abs().max().item())

    far_devs = []
    env.current_value[:] = torch.tensor([10.0, 0.0, 0.0])  # norm = 10 -> effective scale = 0.05 + 0.1*10 = 1.05
    for _ in range(200):
        out = manager.compute_group("actor_obs")["val"]
        far_devs.append((out - env.current_value).abs().max().item())

    # Bounds check (deterministic, not statistical): near deviation never exceeds its scale,
    # far deviation clearly does exceed the near scale's bound at least once (extremely likely
    # over 200 uniform draws spanning up to 1.05 vs a 0.15 ceiling).
    assert max(near_devs) <= 0.15 + 1e-6
    assert max(far_devs) > 0.15
    # And the average magnitude should be meaningfully larger for the far case.
    assert sum(far_devs) / len(far_devs) > 3 * (sum(near_devs) / len(near_devs))


def test_range_noise_disabled_when_group_enable_noise_false():
    """Critic-style group (enable_noise=False) must return the exact clean value every step,
    regardless of noise/noise_range_coefficient on the term."""
    env = _FakeEnv(4)
    term_cfg = ObsTermCfg(
        func="holosoma.managers.observation.tests.test_range_noise_and_delay:_controllable_term",
        noise=0.5,
        noise_range_coefficient=0.5,
    )
    cfg = ObservationManagerCfg(
        groups={"critic_obs": ObsGroupCfg(terms={"val": term_cfg}, concatenate=False, enable_noise=False)}
    )
    manager = ObservationManager(cfg, env, device="cpu")
    env.current_value[:] = torch.tensor([3.0, -2.0, 1.0])
    for _ in range(20):
        out = manager.compute_group("critic_obs")["val"]
        assert torch.equal(out, env.current_value)


# ---------------------------------------------------------------------------------------------
# Latency (delay_step_range)
# ---------------------------------------------------------------------------------------------


def test_zero_delay_range_reproduces_current_value_every_step():
    """delay_step_range=(0, 0) must behave exactly like no delay: the just-inserted current
    value is immediately read back."""
    manager, env = _make_manager(delay_step_range=(0, 0))
    manager.reset(None)
    for i in range(5):
        env.current_value[:] = float(i)
        out = manager.compute_group("actor_obs")["val"]
        assert torch.all(out == float(i))


def test_delay_returns_value_from_n_steps_ago():
    """With a KNOWN, forced delay index, feeding a distinct value every step must return the
    value from exactly `delay` steps ago once the buffer has filled."""
    manager, env = _make_manager(delay_step_range=(0, 5), num_envs=3)
    manager.reset(None)  # allocates nothing yet (lazy) -- fine, first compute() allocates it

    # First compute() call lazily allocates the buffer/idx (delay_idx starts at 0, the
    # documented first-episode quirk) -- force it to a known value for a decisive test instead
    # of depending on the random draw.
    env.current_value[:] = 0.0
    manager.compute_group("actor_obs")
    manager._delay_idx["actor_obs"]["val"][:] = 2  # force every env's delay to exactly 2 steps

    fed_values = []
    observed_values = []
    for i in range(1, 10):
        env.current_value[:] = float(i)
        fed_values.append(float(i))
        out = manager.compute_group("actor_obs")["val"]
        observed_values.append(out[0, 0].item())

    # Once the buffer has filled past the delay (index >= 2 into this loop), observed[i] should
    # equal fed[i - 2].
    for i in range(2, len(fed_values)):
        assert observed_values[i] == fed_values[i - 2], (i, observed_values[i], fed_values[i - 2])


def test_modify_buffer_false_does_not_mutate_or_advance_the_buffer():
    """Mirrors _compute_final_observations()'s modify_history=False bootstrap call: repeated
    non-mutating reads must return the SAME value each time (not drift), and a later real
    (modify_buffer=True) step must continue exactly as if the non-mutating calls never
    happened."""
    manager, env = _make_manager(delay_step_range=(0, 3), num_envs=2)
    env.current_value[:] = 0.0
    manager.compute_group("actor_obs")  # lazy-allocate
    manager._delay_idx["actor_obs"]["val"][:] = 1

    env.current_value[:] = 1.0
    manager.compute_group("actor_obs")  # real step: buffer slots become [1.0, 0.0, 0.0, 0.0]
    # (slot 0 = this step's 1.0; slot 1 = the PRIOR step's 0.0, shifted back -- delay_idx=1
    # therefore currently reads slot 1 = 0.0, the older of the two values fed so far.)

    env.current_value[:] = 999.0  # a value that must NEVER leak into the buffer below
    term_cfg = manager.cfg.groups["actor_obs"].terms["val"]
    read1 = manager._apply_delay("actor_obs", "val", env.current_value, term_cfg, modify_buffer=False)
    read2 = manager._apply_delay("actor_obs", "val", env.current_value, term_cfg, modify_buffer=False)
    assert torch.equal(read1, read2), "non-mutating reads must be stable, not drifting"
    assert torch.all(read1 == 0.0), "non-mutating read must reflect the buffer as it was (slot 1 = 0.0), not the 999.0 probe"

    # Now do a REAL step with a fresh value -- must continue as if the two probes above never
    # happened (buffer should now hold [2.0, 1.0, 0.0, ...], delay=1 -> reads 1.0 still, one
    # step further advanced than before, not two).
    env.current_value[:] = 2.0
    out = manager.compute_group("actor_obs")["val"]
    assert torch.all(out == 1.0)


def test_delay_disabled_when_group_enable_noise_false():
    """Critic-style group (enable_noise=False) must never apply delay, even if delay_step_range
    is set on the term -- always the instantaneous value."""
    env = _FakeEnv(2)
    term_cfg = ObsTermCfg(
        func="holosoma.managers.observation.tests.test_range_noise_and_delay:_controllable_term",
        delay_step_range=(2, 5),
    )
    cfg = ObservationManagerCfg(
        groups={"critic_obs": ObsGroupCfg(terms={"val": term_cfg}, concatenate=False, enable_noise=False)}
    )
    manager = ObservationManager(cfg, env, device="cpu")
    for i in range(5):
        env.current_value[:] = float(i)
        out = manager.compute_group("critic_obs")["val"]
        assert torch.all(out == float(i)), "critic group must never see a delayed value"


def test_reset_zeroes_buffer_and_redraws_delay_idx_within_range():
    manager, env = _make_manager(delay_step_range=(2, 4), num_envs=100)
    env.current_value[:] = 7.0
    manager.compute_group("actor_obs")  # lazy-allocate + fill buffer with 7.0 at all slots eventually

    manager.reset(None)
    buf = manager._delay_buffers["actor_obs"]["val"]
    idx = manager._delay_idx["actor_obs"]["val"]
    assert torch.all(buf == 0.0), "reset must zero the delay buffer"
    assert int(idx.min()) >= 2 and int(idx.max()) <= 4, "redrawn delay index must respect delay_step_range"
    # With 100 envs redrawn uniformly over {2,3,4}, all three values should appear -- a decisive
    # (not just "in range") check that this is actually a per-env random draw, not a constant.
    assert set(idx.tolist()) == {2, 3, 4}


def test_reset_partial_env_ids_only_affects_those_envs():
    manager, env = _make_manager(delay_step_range=(0, 3), num_envs=6)
    env.current_value[:] = 5.0
    for _ in range(4):  # fill every slot of the depth-4 buffer with 5.0, not just slot 0
        manager.compute_group("actor_obs")
    manager._delay_idx["actor_obs"]["val"][:] = 1  # force known baseline

    reset_ids = torch.tensor([0, 1])
    torch.manual_seed(1)
    manager.reset(reset_ids)

    idx = manager._delay_idx["actor_obs"]["val"]
    buf = manager._delay_buffers["actor_obs"]["val"]
    assert torch.all(buf[reset_ids] == 0.0)
    assert torch.all(buf[2:] == 5.0), "untouched envs' buffer must be unaffected"
    # idx[0], idx[1] were redrawn (may or may not equal 1 by chance -- check the buffer is the
    # decisive signal); idx[2:] must still be exactly 1 (untouched).
    assert torch.all(idx[2:] == 1)


# ---------------------------------------------------------------------------------------------
# Zero-order hold (hold_step_range)
# ---------------------------------------------------------------------------------------------


def test_hold_step_range_none_is_bit_identical_to_no_hold():
    """hold_step_range=None (the default for every term registered before this field existed)
    must return the fresh value every single tick, never holding."""
    manager, env = _make_manager(hold_step_range=None)
    manager.reset(None)
    for i in range(5):
        env.current_value[:] = float(i)
        out = manager.compute_group("actor_obs")["val"]
        assert torch.all(out == float(i))


def test_hold_step_range_of_zero_zero_is_also_a_no_op():
    """(0, 0) ("refresh every 0 ticks" -> every drawn period is 0 -> always due) is a
    degenerate-but-valid spelling of "no hold" -- must behave identically to None through the
    arithmetic itself, no special-casing, not a crash."""
    manager, env = _make_manager(hold_step_range=(0, 0))
    manager.reset(None)
    for i in range(5):
        env.current_value[:] = float(i)
        out = manager.compute_group("actor_obs")["val"]
        assert torch.all(out == float(i))


def test_hold_step_range_of_one_one_is_also_a_no_op():
    """(1, 1) ("refresh every 1 tick") is the other degenerate-but-valid "no hold" spelling --
    must also behave identically to None."""
    manager, env = _make_manager(hold_step_range=(1, 1))
    manager.reset(None)
    for i in range(5):
        env.current_value[:] = float(i)
        out = manager.compute_group("actor_obs")["val"]
        assert torch.all(out == float(i))


def test_hold_refreshes_every_period_ticks_and_holds_in_between():
    """With every env's period/countdown forced to a known fixed value, feeding a distinct value
    every tick must reproduce the classic zero-order-hold staircase: the value refreshes once
    every `period` ticks and stays constant for the ticks in between -- the actual real-sensor
    shape this mechanism exists to model (e.g. a 25Hz estimate under 50Hz control ->
    hold_step_range=(2, 2))."""
    period = 3
    manager, env = _make_manager(hold_step_range=(period, period), num_envs=3)
    manager.reset(None)

    env.current_value[:] = 0.0
    manager.compute_group("actor_obs")  # lazy-allocate
    manager._hold_countdown["actor_obs"]["val"][:] = 0  # force every env "due" right now

    fed_values = []
    observed_values = []
    for i in range(1, 10):
        env.current_value[:] = float(i)
        fed_values.append(float(i))
        out = manager.compute_group("actor_obs")["val"]
        observed_values.append(out[0, 0].item())

    for i in range(len(fed_values)):
        block_start = (i // period) * period
        assert observed_values[i] == fed_values[block_start], (i, observed_values[i], fed_values[block_start])
    # Decisive check that it's actually a staircase, not just "always the first value" or
    # "always fresh": at least two distinct held values must appear across the run.
    assert len(set(observed_values)) > 1


def test_hold_period_itself_is_randomized_per_env_within_range():
    """A genuine range (not min==max) must draw a DIFFERENT period per env, not just a different
    phase within one shared period -- exercised by checking manager._hold_period, the per-episode
    value _apply_hold reads to decide when the next refresh is due."""
    manager, env = _make_manager(hold_step_range=(2, 5), num_envs=200)
    env.current_value[:] = 0.0
    manager.compute_group("actor_obs")  # lazy-allocate (period defaults to the range's low end)
    manager.reset(None)

    period = manager._hold_period["actor_obs"]["val"]
    countdown = manager._hold_countdown["actor_obs"]["val"]
    assert int(period.min()) >= 2 and int(period.max()) <= 5, "redrawn period must respect [low, high]"
    # With 200 envs redrawn uniformly over {2,3,4,5}, all four values should appear -- decisive
    # evidence this is a per-env random draw, not a constant.
    assert set(period.tolist()) == {2, 3, 4, 5}
    # The phase drawn alongside it must respect THAT env's own period, not the range's max.
    assert bool((countdown < period).all()), "countdown phase must be strictly less than that env's own period"
    assert bool((countdown >= 0).all())


def test_modify_buffer_false_does_not_mutate_or_advance_the_hold():
    """Mirrors the analogous delay test: repeated non-mutating reads must return the SAME held
    value each time, and a later real (modify_buffer=True) step must continue exactly as if the
    non-mutating calls never happened."""
    manager, env = _make_manager(hold_step_range=(3, 3), num_envs=2)
    env.current_value[:] = 0.0
    manager.compute_group("actor_obs")  # lazy-allocate, refreshes to 0.0, countdown -> 2
    manager._hold_countdown["actor_obs"]["val"][:] = 2  # force "not due yet" for two more ticks

    env.current_value[:] = 999.0  # must NEVER leak into the cache below
    term_cfg = manager.cfg.groups["actor_obs"].terms["val"]
    read1 = manager._apply_hold("actor_obs", "val", env.current_value, term_cfg, modify_buffer=False)
    read2 = manager._apply_hold("actor_obs", "val", env.current_value, term_cfg, modify_buffer=False)
    assert torch.equal(read1, read2), "non-mutating reads must be stable, not drifting"
    assert torch.all(read1 == 0.0), "non-mutating read must reflect the held 0.0, not the 999.0 probe"

    # A real step now, with a fresh value, must still be "not due" (countdown was 2, one real
    # step should bring it to 1, still holding 0.0) -- not corrupted by the two probes above.
    env.current_value[:] = 5.0
    out = manager.compute_group("actor_obs")["val"]
    assert torch.all(out == 0.0)


def test_hold_disabled_when_group_enable_noise_false():
    """Critic-style group (enable_noise=False) must never apply hold, even if hold_step_range is
    set on the term -- always the instantaneous value."""
    env = _FakeEnv(2)
    term_cfg = ObsTermCfg(
        func="holosoma.managers.observation.tests.test_range_noise_and_delay:_controllable_term",
        hold_step_range=(4, 4),
    )
    cfg = ObservationManagerCfg(
        groups={"critic_obs": ObsGroupCfg(terms={"val": term_cfg}, concatenate=False, enable_noise=False)}
    )
    manager = ObservationManager(cfg, env, device="cpu")
    for i in range(6):
        env.current_value[:] = float(i)
        out = manager.compute_group("critic_obs")["val"]
        assert torch.all(out == float(i)), "critic group must never see a held/stale value"


def test_reset_zeroes_cache_and_redraws_period_and_phase_within_range():
    manager, env = _make_manager(hold_step_range=(2, 4), num_envs=100)
    env.current_value[:] = 7.0
    manager.compute_group("actor_obs")  # lazy-allocate + refresh cache to 7.0

    manager.reset(None)
    cache = manager._hold_cache["actor_obs"]["val"]
    period = manager._hold_period["actor_obs"]["val"]
    countdown = manager._hold_countdown["actor_obs"]["val"]
    assert torch.all(cache == 0.0), "reset must zero the hold cache"
    assert int(period.min()) >= 2 and int(period.max()) <= 4, "redrawn period must respect [low, high]"
    assert set(period.tolist()) == {2, 3, 4}, "100 envs over {2,3,4} should hit every value"
    assert bool((countdown >= 0).all()) and bool((countdown < period).all()), "phase must respect its own env's period"


def test_reset_partial_env_ids_only_affects_those_envs_hold():
    manager, env = _make_manager(hold_step_range=(3, 3), num_envs=6)
    env.current_value[:] = 5.0
    manager.compute_group("actor_obs")  # lazy-allocate, refreshes cache to 5.0
    manager._hold_countdown["actor_obs"]["val"][:] = 1  # force known baseline

    reset_ids = torch.tensor([0, 1])
    torch.manual_seed(1)
    manager.reset(reset_ids)

    countdown = manager._hold_countdown["actor_obs"]["val"]
    cache = manager._hold_cache["actor_obs"]["val"]
    assert torch.all(cache[reset_ids] == 0.0)
    assert torch.all(cache[2:] == 5.0), "untouched envs' cache must be unaffected"
    assert torch.all(countdown[2:] == 1), "untouched envs' countdown must be unaffected"


# ---------------------------------------------------------------------------------------------
# Stale-frame reuse (stale_probability)
# ---------------------------------------------------------------------------------------------


def test_stale_probability_zero_is_bit_identical_to_no_stale():
    """stale_probability=0.0 (the default for every term registered before this field existed)
    must return the fresh value every single tick, never reusing a cached one."""
    manager, env = _make_manager(stale_probability=0.0)
    manager.reset(None)
    for i in range(10):
        env.current_value[:] = float(i)
        out = manager.compute_group("actor_obs")["val"]
        assert torch.all(out == float(i))


def test_stale_probability_one_always_reuses_the_cached_frame():
    """probability=1.0 means every tick after the first is (deterministically) stale: the value
    must freeze at whatever it was seeded with and never advance, even though fresh distinct
    values are fed every tick."""
    manager, env = _make_manager(stale_probability=1.0, num_envs=5)
    env.current_value[:] = 0.0
    manager.compute_group("actor_obs")  # lazy-allocate, seeds the cache at 0.0

    for i in range(1, 6):
        env.current_value[:] = float(i)
        out = manager.compute_group("actor_obs")["val"]
        assert torch.all(out == 0.0), f"tick {i}: probability=1.0 must always reuse the seeded frame"


def test_stale_reuse_does_not_chain_forward():
    """A run of consecutive stale ticks must return the SAME last-real frame every time, not
    progressively older ones -- the cache only advances on a tick that was NOT stale. Forced via
    a monkeypatched stale mask (probability alone can't guarantee a specific streak length
    deterministically) so this is a decisive, not statistical, check."""
    manager, env = _make_manager(stale_probability=0.5, num_envs=1)
    env.current_value[:] = 10.0
    manager.compute_group("actor_obs")  # lazy-allocate, cache seeded at 10.0

    real_rand = torch.rand

    def _always_stale(*args, **kwargs):
        return torch.zeros(*args, **kwargs)  # < any positive probability -> always "stale"

    torch.rand = _always_stale
    try:
        seen = []
        for i in range(1, 5):
            env.current_value[:] = float(20 + i)  # distinct fresh value every tick, must be ignored
            out = manager.compute_group("actor_obs")["val"]
            seen.append(out[0, 0].item())
    finally:
        torch.rand = real_rand

    assert seen == [10.0, 10.0, 10.0, 10.0], "every stale tick must reuse the SAME original frame"


def test_modify_buffer_false_does_not_mutate_or_advance_the_stale_cache():
    """Mirrors the analogous delay/hold tests: repeated non-mutating reads must return the SAME
    cached value each time, and a later real (modify_buffer=True) step must continue exactly as
    if the non-mutating calls never happened."""
    manager, env = _make_manager(stale_probability=1.0, num_envs=2)
    env.current_value[:] = 0.0
    manager.compute_group("actor_obs")  # lazy-allocate, seeds cache at 0.0

    env.current_value[:] = 999.0  # must NEVER leak into the cache below
    term_cfg = manager.cfg.groups["actor_obs"].terms["val"]
    read1 = manager._apply_stale("actor_obs", "val", env.current_value, term_cfg, modify_buffer=False)
    read2 = manager._apply_stale("actor_obs", "val", env.current_value, term_cfg, modify_buffer=False)
    assert torch.equal(read1, read2), "non-mutating reads must be stable, not drifting"
    assert torch.all(read1 == 0.0), "non-mutating read must reflect the seeded 0.0, not the 999.0 probe"

    # A real step now, with probability=1.0, must still reuse the original 0.0 -- not corrupted
    # by the two probes above (which must not have advanced anything).
    env.current_value[:] = 5.0
    out = manager.compute_group("actor_obs")["val"]
    assert torch.all(out == 0.0)


def test_stale_disabled_when_group_enable_noise_false():
    """Critic-style group (enable_noise=False) must never apply stale-reuse, even if
    stale_probability is set on the term -- always the instantaneous value."""
    env = _FakeEnv(3)
    term_cfg = ObsTermCfg(
        func="holosoma.managers.observation.tests.test_range_noise_and_delay:_controllable_term",
        stale_probability=1.0,
    )
    cfg = ObservationManagerCfg(
        groups={"critic_obs": ObsGroupCfg(terms={"val": term_cfg}, concatenate=False, enable_noise=False)}
    )
    manager = ObservationManager(cfg, env, device="cpu")
    for i in range(6):
        env.current_value[:] = float(i)
        out = manager.compute_group("critic_obs")["val"]
        assert torch.all(out == float(i)), "critic group must never see a stale-reused value"


def test_reset_zeroes_stale_cache():
    manager, env = _make_manager(stale_probability=0.5, num_envs=10)
    env.current_value[:] = 7.0
    manager.compute_group("actor_obs")  # lazy-allocate + seed cache at 7.0

    manager.reset(None)
    cache = manager._stale_cache["actor_obs"]["val"]
    assert torch.all(cache == 0.0), "reset must zero the stale cache"


def test_reset_partial_env_ids_only_affects_those_envs_stale():
    manager, env = _make_manager(stale_probability=1.0, num_envs=6)
    env.current_value[:] = 5.0
    manager.compute_group("actor_obs")  # lazy-allocate, seeds cache at 5.0

    reset_ids = torch.tensor([0, 1])
    manager.reset(reset_ids)

    cache = manager._stale_cache["actor_obs"]["val"]
    assert torch.all(cache[reset_ids] == 0.0)
    assert torch.all(cache[2:] == 5.0), "untouched envs' cache must be unaffected"


# ---------------------------------------------------------------------------------------------
# Integration: all four mechanisms together, exactly as configured for kick_ball_pos_b
# ---------------------------------------------------------------------------------------------


def test_range_noise_and_delay_combined_no_crash_finite_output():
    manager, env = _make_manager(
        noise=0.05,
        noise_range_coefficient=0.03,
        delay_step_range=(0, 3),
        hold_step_range=(2, 3),
        stale_probability=0.05,
        num_envs=16,
    )
    manager.reset(None)
    for i in range(20):
        env.current_value[:] = torch.rand(16, 3) * 5.0
        out = manager.compute_group("actor_obs")["val"]
        assert out.shape == (16, 3)
        assert torch.isfinite(out).all()
