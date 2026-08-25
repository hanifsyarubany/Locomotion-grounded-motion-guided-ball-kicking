"""Unit tests for UnifiedManager's kick-episode-end rolling stats (2026-07-28, user-requested
aliveness/ball-shooting evaluation logging): kick_alive_frac, kick_ball_hit_rate,
kick_ball_success_rate, kick_ball_velocity. Isolated via a bare instance (same pattern as
test_unified_manager_task_mode_partition.py), since the full method only touches
episode_length_buf/time_out_buf/_shot_tracker, not the whole simulator/terrain/scene.
"""

import torch

from holosoma.envs.unified.unified_manager import UnifiedManager


def _make_manager(num_envs: int, num_skills: int = 0) -> UnifiedManager:
    m = object.__new__(UnifiedManager)
    m.num_envs = num_envs
    m.device = "cpu"
    m.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
    m.time_out_buf = torch.zeros(num_envs, dtype=torch.bool)
    # skill_id always exists on the real UnifiedManager (_init_buffers sets it unconditionally,
    # staying 0 for every env in legacy/single-skill mode) -- mirrored here regardless of
    # num_skills so the `> 0` gate (2026-08-06) has something to index in every mode.
    m.skill_id = torch.zeros(num_envs, dtype=torch.long)
    if num_skills > 0:
        m._skill_motion_training_ratios = [1.0 / num_skills] * num_skills
    return m


class _FakeShotTracker:
    def __init__(self, has_kicked, success_latched, max_ball_speed):
        self.has_kicked = has_kicked
        self.success_latched = success_latched
        self.max_ball_speed = max_ball_speed


def test_hit_and_success_rate_match_batch_mean_exactly():
    m = _make_manager(num_envs=8)
    kick_ids = torch.arange(8)
    m.episode_length_buf[kick_ids] = 500
    m.time_out_buf[kick_ids] = False  # all terminated, irrelevant to this test
    m._shot_tracker = _FakeShotTracker(
        has_kicked=torch.tensor([True] * 4 + [False] * 4),
        success_latched=torch.tensor([True] * 2 + [False] * 6),
        max_ball_speed=torch.tensor([3.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0]),
    )

    m._record_kick_episode_ends(kick_ids)

    assert abs(m._kick_stat(m._kick_ep_hit_ema).item() - 0.5) < 1e-5
    assert abs(m._kick_stat(m._kick_ep_success_ema).item() - 0.25) < 1e-5


def test_ball_velocity_averages_only_over_hits_not_diluted_by_misses():
    """A 0.0 max_ball_speed from a MISS must not drag down the reported velocity -- that's a
    property of the attempt never making contact, not evidence the ball moved slowly."""
    m = _make_manager(num_envs=8)
    kick_ids = torch.arange(8)
    m.episode_length_buf[kick_ids] = 500
    m.time_out_buf[kick_ids] = False
    m._shot_tracker = _FakeShotTracker(
        has_kicked=torch.tensor([True] * 4 + [False] * 4),
        success_latched=torch.zeros(8, dtype=torch.bool),
        max_ball_speed=torch.tensor([3.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0]),
    )

    m._record_kick_episode_ends(kick_ids)

    reported = m._kick_stat(m._kick_ep_ball_speed_ema, m._kick_ep_ball_speed_decay_pow).item()
    assert abs(reported - 4.5) < 1e-4  # mean(3,4,5,6), NOT mean(3,4,5,6,0,0,0,0)=2.25


def test_ball_velocity_decay_pow_does_not_advance_on_all_miss_batch():
    """The exact bug this design avoids: if ball_velocity shared the hit/success/len/term EMAs'
    decay_pow, an all-miss batch would still advance that shared denominator (since hit/success
    DO have a well-defined batch mean of 0), silently under-reporting ball_velocity relative to
    how many batches actually contributed real speed data."""
    m = _make_manager(num_envs=8)
    kick_ids = torch.arange(8)
    m.episode_length_buf[kick_ids] = 500
    m.time_out_buf[kick_ids] = False

    m._shot_tracker = _FakeShotTracker(
        has_kicked=torch.tensor([True] * 4 + [False] * 4),
        success_latched=torch.zeros(8, dtype=torch.bool),
        max_ball_speed=torch.tensor([3.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0]),
    )
    m._record_kick_episode_ends(kick_ids)
    velocity_before = m._kick_stat(m._kick_ep_ball_speed_ema, m._kick_ep_ball_speed_decay_pow).item()
    hit_rate_before = m._kick_stat(m._kick_ep_hit_ema).item()

    # All-miss batch: hit_rate's shared-decay EMA must still fold in (and drop toward 0), but
    # ball_velocity's OWN decay_pow must not advance at all.
    m._shot_tracker = _FakeShotTracker(
        has_kicked=torch.zeros(8, dtype=torch.bool),
        success_latched=torch.zeros(8, dtype=torch.bool),
        max_ball_speed=torch.zeros(8),
    )
    m._record_kick_episode_ends(kick_ids)

    velocity_after = m._kick_stat(m._kick_ep_ball_speed_ema, m._kick_ep_ball_speed_decay_pow).item()
    hit_rate_after = m._kick_stat(m._kick_ep_hit_ema).item()

    assert abs(velocity_after - velocity_before) < 1e-6, (
        f"ball_velocity must be UNCHANGED by an all-miss batch, got {velocity_before} -> {velocity_after}"
    )
    assert hit_rate_after < hit_rate_before, "hit_rate must still drop after an all-miss batch"


def test_missing_shot_tracker_does_not_crash_and_leaves_ball_metrics_at_zero():
    """The very first-ever kick episode end can fire before any kick-mode reward computation has
    lazily constructed _shot_tracker -- must not crash."""
    m = _make_manager(num_envs=4)
    kick_ids = torch.arange(4)
    m.episode_length_buf[kick_ids] = 100
    m.time_out_buf[kick_ids] = True
    assert not hasattr(m, "_shot_tracker")

    m._record_kick_episode_ends(kick_ids)  # must not raise

    assert m._kick_stat(m._kick_ep_hit_ema).item() == 0.0
    assert m._kick_stat(m._kick_ep_ball_speed_ema, m._kick_ep_ball_speed_decay_pow).item() == 0.0


def test_alive_frac_is_exact_complement_of_early_term_frac():
    m = _make_manager(num_envs=6)
    kick_ids = torch.arange(6)
    m.episode_length_buf[kick_ids] = 300
    m.time_out_buf[kick_ids] = torch.tensor([True, True, False, False, False, False])  # 4/6 terminated
    m._shot_tracker = _FakeShotTracker(
        has_kicked=torch.zeros(6, dtype=torch.bool),
        success_latched=torch.zeros(6, dtype=torch.bool),
        max_ball_speed=torch.zeros(6),
    )

    m._record_kick_episode_ends(kick_ids)

    early_term = m._kick_stat(m._kick_ep_term_ema).item()
    alive = 1.0 - early_term
    assert abs(early_term - (4.0 / 6.0)) < 1e-5
    assert abs((alive + early_term) - 1.0) < 1e-9


def test_per_skill_breakdown_isolates_a_new_immature_skill_from_a_mature_one():
    """The scenario this was built to diagnose: a blended kick_alive_frac plateau that's really
    one mature skill (near-perfect survival) averaged with one brand-new skill (frequent early
    termination) -- the per-skill split must recover both true rates, not just echo the blend."""
    m = _make_manager(num_envs=8, num_skills=2)
    kick_ids = torch.arange(8)
    m.skill_id[:4] = 0  # mature skill: all 4 survive (timeout)
    m.skill_id[4:] = 1  # new skill: all 4 terminate early
    m.episode_length_buf[kick_ids] = 500
    m.time_out_buf[:] = torch.tensor([True] * 4 + [False] * 4)
    m._shot_tracker = _FakeShotTracker(
        has_kicked=torch.zeros(8, dtype=torch.bool),
        success_latched=torch.zeros(8, dtype=torch.bool),
        max_ball_speed=torch.zeros(8),
    )

    m._record_kick_episode_ends(kick_ids)

    term_sk = m._kick_stat(m._kick_ep_term_ema_sk, m._kick_ep_decay_pow_sk)
    assert abs(term_sk[0].item() - 0.0) < 1e-5  # skill 0 (mature): 0% early-term
    assert abs(term_sk[1].item() - 1.0) < 1e-5  # skill 1 (new): 100% early-term
    # The pre-existing GLOBAL/blended stat is untouched by this addition -- still the mixed 50%.
    assert abs(m._kick_stat(m._kick_ep_term_ema).item() - 0.5) < 1e-5


def test_per_skill_decay_pow_does_not_advance_for_a_skill_with_no_episodes_this_call():
    """Mirrors the existing ball_speed decay_pow test: a skill that contributes zero ended
    episodes in a given call must not have its own bias-correction denominator advance, or its
    reported rate would silently be dragged toward the update it never actually received."""
    m = _make_manager(num_envs=4, num_skills=2)
    m.skill_id[:] = 0  # every env this call belongs to skill 0 -- skill 1 gets nothing
    kick_ids = torch.arange(4)
    m.episode_length_buf[kick_ids] = 200
    m.time_out_buf[kick_ids] = torch.tensor([True, True, False, False])
    m._shot_tracker = _FakeShotTracker(
        has_kicked=torch.zeros(4, dtype=torch.bool),
        success_latched=torch.zeros(4, dtype=torch.bool),
        max_ball_speed=torch.zeros(4),
    )

    m._record_kick_episode_ends(kick_ids)

    assert m._kick_ep_decay_pow_sk[0].item() > 0.0
    assert m._kick_ep_decay_pow_sk[1].item() == 0.0
    assert m._kick_stat(m._kick_ep_term_ema_sk, m._kick_ep_decay_pow_sk)[1].item() == 0.0


def test_topple_frac_is_a_strict_min_height_check_distinct_from_early_term():
    """The user-requested distinction: kick_alive_frac conflates bad_tracking (can fire while
    still standing) with kick_low_height (an actual fall). kick_topple_frac must reflect ONLY
    whether base height ever crossed the fall threshold, regardless of why (or whether) the
    episode terminated -- a bad_tracking-only episode that never got low must read topple=False
    even though it's counted as an early termination."""
    m = _make_manager(num_envs=4)
    kick_ids = torch.arange(4)
    m.episode_length_buf[kick_ids] = 300
    m.time_out_buf[kick_ids] = torch.tensor([False, False, True, True])  # 0,1 terminated early
    # Manually set the per-env running min-height tracker (normally maintained by
    # _update_counters_each_step over the course of the episode) -- env 0 terminated early via
    # bad_tracking WITHOUT ever getting low (min height stayed well above threshold); env 1
    # terminated early via an actual fall; envs 2/3 timed out normally, never low.
    m._init_kick_episode_stats()  # must run first -- it's what allocates _kick_ep_min_height
    m._kick_ep_min_height = torch.tensor([0.70, 0.15, 0.75, 0.72])
    m._shot_tracker = _FakeShotTracker(
        has_kicked=torch.zeros(4, dtype=torch.bool),
        success_latched=torch.zeros(4, dtype=torch.bool),
        max_ball_speed=torch.zeros(4),
    )

    m._record_kick_episode_ends(kick_ids)

    early_term = m._kick_stat(m._kick_ep_term_ema).item()
    topple = m._kick_stat(m._kick_ep_topple_ema).item()
    assert abs(early_term - 0.5) < 1e-5  # 2/4 terminated early
    assert abs(topple - 0.25) < 1e-5  # only 1/4 (env 1) actually fell
    assert topple < early_term  # the whole point: strictly fewer topples than early terminations

    min_height = m._kick_stat(m._kick_ep_min_height_ema).item()
    assert abs(min_height - torch.tensor([0.70, 0.15, 0.75, 0.72]).mean().item()) < 1e-5


def test_topple_frac_per_skill_breakdown():
    m = _make_manager(num_envs=4, num_skills=2)
    kick_ids = torch.arange(4)
    m.skill_id[:2] = 0  # skill 0: neither falls
    m.skill_id[2:] = 1  # skill 1: both fall
    m.episode_length_buf[kick_ids] = 300
    m.time_out_buf[kick_ids] = torch.tensor([True, True, False, False])
    m._init_kick_episode_stats()  # must run first -- it's what allocates _kick_ep_min_height
    m._kick_ep_min_height = torch.tensor([0.70, 0.72, 0.10, 0.05])
    m._shot_tracker = _FakeShotTracker(
        has_kicked=torch.zeros(4, dtype=torch.bool),
        success_latched=torch.zeros(4, dtype=torch.bool),
        max_ball_speed=torch.zeros(4),
    )

    m._record_kick_episode_ends(kick_ids)

    topple_sk = m._kick_stat(m._kick_ep_topple_ema_sk, m._kick_ep_decay_pow_sk)
    assert abs(topple_sk[0].item() - 0.0) < 1e-5
    assert abs(topple_sk[1].item() - 1.0) < 1e-5


def test_min_height_tracker_reset_to_sentinel_produces_no_topple():
    """A freshly-initialized (never-stepped) env's min-height tracker sits at the sentinel
    (taller than any real robot) -- must not be misread as a topple."""
    m = _make_manager(num_envs=2)
    kick_ids = torch.arange(2)
    m.episode_length_buf[kick_ids] = 50
    m.time_out_buf[kick_ids] = True
    m._shot_tracker = _FakeShotTracker(
        has_kicked=torch.zeros(2, dtype=torch.bool),
        success_latched=torch.zeros(2, dtype=torch.bool),
        max_ball_speed=torch.zeros(2),
    )

    m._record_kick_episode_ends(kick_ids)  # _kick_ep_min_height auto-inits to the sentinel here

    assert bool((m._kick_ep_min_height == m._KICK_MIN_HEIGHT_SENTINEL).all())
    assert m._kick_stat(m._kick_ep_topple_ema).item() == 0.0


def test_single_skill_mode_still_populates_per_skill_bookkeeping():
    """num_skills defaults to 1 (no _skill_motion_training_ratios set) -- the legacy/pre-N-skill
    path this codebase runs today. 2026-08-06 (user-requested): the per-skill breakdown is now
    populated even here (gate is `> 0`, not `> 1`), so "Kick_skills_0" appears in the dashboard
    for single-skill runs too, not just once a run happens to add a 2nd skill. skill_id stays 0
    for every env in this mode (matching the real UnifiedManager's _init_buffers), so the lone
    per-skill entry is numerically identical to the global EMA it's folded into alongside it."""
    m = _make_manager(num_envs=4)
    kick_ids = torch.arange(4)
    m.episode_length_buf[kick_ids] = 100
    m.time_out_buf[kick_ids] = True

    m._record_kick_episode_ends(kick_ids)

    assert m._kick_num_skills == 1
    assert m._kick_ep_decay_pow_sk[0].item() > 0.0  # skill 0's per-skill EMA WAS folded into
    global_term = m._kick_stat(m._kick_ep_term_ema).item()
    per_skill_term = m._kick_stat(m._kick_ep_term_ema_sk, m._kick_ep_decay_pow_sk)[0].item()
    assert per_skill_term == global_term == 0.0


def test_handoff_split_ignores_ordinary_episodes_with_no_handoff_flag_set():
    """Baseline (2026-08-14, user-requested handoff learning-progress signal): when no ended
    episode is handoff-triggered (_kick_ep_is_handoff all False, the default -- ordinary
    reset-triggered kicks never set it, see _enter_kick), the handoff-only EMA's decay_pow must
    stay at exactly 0 -- no sample was ever folded in -- while the pooled/global EMA is unaffected."""
    m = _make_manager(num_envs=4)
    kick_ids = torch.arange(4)
    m.episode_length_buf[kick_ids] = 200
    m.time_out_buf[kick_ids] = torch.tensor([True, True, False, False])

    m._record_kick_episode_ends(kick_ids)

    assert m._kick_ep_is_handoff.tolist() == [False, False, False, False]
    assert m._kick_ep_decay_pow_handoff.item() == 0.0
    assert m._kick_stat(m._kick_ep_term_ema_handoff, m._kick_ep_decay_pow_handoff).item() == 0.0
    assert abs(m._kick_stat(m._kick_ep_term_ema).item() - 0.5) < 1e-5  # global still folds in normally


def test_handoff_split_isolates_handoff_episodes_from_ordinary_ones():
    """The scenario this split exists to surface: 2 ordinary kick episodes (both survive) mixed
    with 2 handoff-triggered ones (both terminate early, e.g. the policy hasn't learned to handle
    being spliced in mid-stride yet) -- the handoff-only stat must recover the true 100% handoff
    early-term rate, not the blended 50%."""
    m = _make_manager(num_envs=4)
    kick_ids = torch.arange(4)
    m.episode_length_buf[kick_ids] = 300
    m.time_out_buf[kick_ids] = torch.tensor([True, True, False, False])  # 2,3 terminated early
    m._init_kick_episode_stats()  # must run first -- allocates _kick_ep_is_handoff
    m._kick_ep_is_handoff[2:] = True  # envs 2,3 are the handoff-triggered ones

    m._record_kick_episode_ends(kick_ids)

    handoff_term = m._kick_stat(m._kick_ep_term_ema_handoff, m._kick_ep_decay_pow_handoff).item()
    global_term = m._kick_stat(m._kick_ep_term_ema).item()
    assert abs(handoff_term - 1.0) < 1e-5, "both handoff episodes terminated early"
    assert abs(global_term - 0.5) < 1e-5, "the pooled/blended stat is untouched by the split"


def test_handoff_decay_pow_does_not_advance_for_a_batch_with_no_handoff_episodes():
    """Mirrors the existing ball-speed/per-skill decay_pow tests: a batch of ended episodes that
    happens to contain zero handoff-triggered ones must not advance the handoff EMA's own
    denominator, or its reported rate would be silently dragged toward an update it never got."""
    m = _make_manager(num_envs=2)
    kick_ids = torch.arange(2)
    m.episode_length_buf[kick_ids] = 100
    m.time_out_buf[kick_ids] = True

    m._record_kick_episode_ends(kick_ids)  # no handoff episodes in this batch

    assert m._kick_ep_decay_pow_handoff.item() == 0.0

    m._init_kick_episode_stats()  # idempotent -- already inited by the call above
    m._kick_ep_is_handoff[:] = True
    m._record_kick_episode_ends(kick_ids)  # now both are handoff episodes

    assert m._kick_ep_decay_pow_handoff.item() > 0.0
