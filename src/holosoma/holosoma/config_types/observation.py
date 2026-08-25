"""Configuration types for observation manager."""

from __future__ import annotations

from dataclasses import field
from typing import Any

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class ObsTermCfg:
    """Configuration for a single observation term."""

    func: str
    """Import path to the observation function (e.g. ``holosoma.managers.observation.terms.locomotion:base_lin_vel``)."""

    params: dict[str, Any] = field(default_factory=dict)
    """Additional keyword arguments forwarded to ``func``."""

    scale: float | tuple[float, ...] = 1.0
    """Scaling factor(s) applied after noise and clipping."""

    noise: float = 0.0
    """Flat noise magnitude applied before scaling when the group enables noise."""

    noise_range_coefficient: float = 0.0
    """Additional per-step noise magnitude that scales linearly with the observation's own L2
    norm (e.g. distance-dependent depth-sensor error): effective noise half-range =
    ``noise + noise_range_coefficient * ||obs||``. Applied at the same point and under the same
    ``group.enable_noise`` gate as ``noise`` -- 0.0 (default) is a pure no-op, bit-identical to
    every term registered before this field existed."""

    delay_step_range: tuple[int, int] | None = None
    """If set, this term's value is delayed by a per-env uniform-random number of steps in
    ``[low, high]`` (inclusive), redrawn at every reset -- models sensor/perception pipeline
    latency (e.g. camera/LiDAR + Kalman-filter fusion lag), distinct from ``noise`` (which
    perturbs magnitude, not time). Only applied when the owning group has ``enable_noise=True``
    (the same "this group is realistic/deployable perception, not privileged ground truth" gate
    ``noise`` uses) -- a group with ``enable_noise=False`` always reads the instantaneous,
    undelayed value regardless of this setting. ``None`` (default) is a pure no-op, bit-identical
    to every term registered before this field existed."""

    hold_step_range: tuple[int, int] | None = None
    """If set, this term's value only REFRESHES every N control ticks -- zero-order hold, N drawn
    per-env uniformly in ``[low, high]`` (inclusive) at every reset, held fixed for that episode
    (same contract as ``delay_step_range``'s index -- models a real pipeline's update rate
    fluctuating somewhat episode-to-episode, e.g. with scene load). Applied AFTER
    ``delay_step_range`` (a real perception pipeline's own update rate is what's slow, not the
    transport delay of any single sample it does publish; the value being held is therefore
    whatever the delay stage already produced, not a second, independent lag). Models a
    fusion/estimation pipeline that runs slower than the control loop (e.g. a 25Hz ball-pose
    estimator under a 50Hz control loop -> ``hold_step_range=(2, 2)`` for a fixed rate, or a wider
    range to also model rate jitter) -- distinct from ``delay_step_range``, which models transport
    LATENCY on a pipeline that still updates every tick. Within the drawn N-tick period, each env
    also gets a random PHASE offset (not just the period itself) so envs aren't all refreshing on
    the same global tick -- see ``ObservationManager._apply_hold``. Only applied when the owning
    group has ``enable_noise=True``, same gate ``noise``/``delay_step_range`` use. ``None``
    (default) is a pure no-op, bit-identical to every term registered before this field existed;
    ``(0, 0)`` or ``(1, 1)`` are also no-ops through the arithmetic itself (every tick refreshes),
    no special-casing needed."""

    stale_probability: float = 0.0
    """Per-CONTROL-TICK probability (checked fresh every step, unlike ``hold_step_range``'s
    per-episode-fixed period) that this term returns the PREVIOUS tick's already-fully-processed
    reading instead of this tick's -- models a single dropped/repeated sensor frame (e.g. one
    skipped LiDAR packet), a transient, self-correcting fault distinct from ``hold_step_range``'s
    deterministic slower-update-rate modeling. Applied AFTER ``hold_step_range`` (see
    ``ObservationManager._apply_stale``'s own docstring) -- the "previous reading" reused is
    whatever the pipeline would have legitimately returned last tick (already noised/delayed/
    held), not a second, independent corruption on top. During a run of consecutive stale ticks
    for one env, the cache does NOT advance -- every stale tick reuses the SAME last-real frame,
    not a chain of increasingly-old ones. Only applied when the owning group has
    ``enable_noise=True``, same gate every other perception-artifact field here uses. ``0.0``
    (default) is a pure no-op, bit-identical to every term registered before this field existed."""

    clip: tuple[float, float] | None = None
    """Optional min/max clip bounds applied after scaling."""

    task_mode: str | None = None
    """If set, this term's output is zeroed (never omitted — the concatenated group width stays
    constant) for envs not currently in this task mode (per ``env.task_mode_mask(task_mode)``,
    only consulted when the env implements it — e.g. UnifiedManager). ``None`` (the default)
    means always active, matching every existing experiment's behavior exactly."""


@dataclass(frozen=True)
class ObsGroupCfg:
    """Configuration for an observation group (e.g. actor or critic input)."""

    terms: dict[str, ObsTermCfg] = field(default_factory=dict)
    """Mapping of term name to its configuration."""

    concatenate: bool = True
    """Whether to concatenate term outputs into a single tensor."""

    enable_noise: bool = False
    """If ``True``, apply each term's noise setting before scaling."""

    history_length: int = 1
    """Number of timesteps to retain for history stacking (``1`` disables history)."""


@dataclass(frozen=True)
class ObservationManagerCfg:
    """Configuration for the observation manager."""

    groups: dict[str, ObsGroupCfg] = field(default_factory=dict)
    """Mapping of group name to its configuration."""

    clip_observations: float = 100.0
    """Global observation clipping threshold (applied to all observations)."""
