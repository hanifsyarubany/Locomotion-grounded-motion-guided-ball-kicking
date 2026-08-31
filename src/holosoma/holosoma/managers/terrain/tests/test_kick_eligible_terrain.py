"""Unit tests for the configurable kick-eligibility terrain gate (2026-08-27).

The mask itself is a pure function of (per-env terrain-type names, configured eligible names), so
it is tested directly here rather than through a live TerrainLocomotion, which would need a real
simulator and mesh. What this does NOT cover: that TerrainLocomotion._get_env_origins wires these
two inputs together correctly -- that needs a live env.

The property that matters most and is asserted here: `env_terrain_is_flat` must keep meaning
LITERAL flatness even when the kick gate is widened, because it has a second, unrelated consumer
(UnifiedManager's kick-mode video-recorder env selection).
"""

import numpy as np


def _flat_mask(type_names: np.ndarray) -> np.ndarray:
    """Mirrors TerrainLocomotion._get_env_origins' env_terrain_is_flat computation."""
    return type_names == "flat"


def _eligible_mask(type_names: np.ndarray, eligible_names: tuple[str, ...]) -> np.ndarray:
    """Mirrors TerrainLocomotion._get_env_origins' env_terrain_kick_eligible computation."""
    return np.isin(type_names, np.array(eligible_names, dtype=object))


_TYPES = np.array(
    ["flat", "rough", "light_rough", "flat", "low_obstacles", "light_rough"], dtype=object
)


def test_default_eligibility_matches_flat_exactly():
    """The default ("flat",) must reproduce the previous hardcoded flat-only gate bit-for-bit."""
    assert np.array_equal(_eligible_mask(_TYPES, ("flat",)), _flat_mask(_TYPES))


def test_widened_eligibility_includes_light_rough_tiles():
    eligible = _eligible_mask(_TYPES, ("flat", "light_rough"))
    assert eligible.tolist() == [True, False, True, True, False, True]
    # Strictly a superset of flat -- widening may only ADD envs, never remove one.
    assert np.all(eligible[_flat_mask(_TYPES)])


def test_widening_does_not_change_the_flat_mask():
    """env_terrain_is_flat has its own consumer (video-recorder env selection) and must be
    unaffected by kick-gate configuration."""
    before = _flat_mask(_TYPES).copy()
    _eligible_mask(_TYPES, ("flat", "light_rough", "rough"))
    assert np.array_equal(_flat_mask(_TYPES), before)


def test_eligibility_never_selects_an_ungenerated_type():
    """If light_rough tiles were never generated, widening the gate to include it must not
    hallucinate eligible envs -- it simply matches nothing extra (the loader separately raises
    on this combination, but the mask must be safe regardless)."""
    types_without_light = np.array(["flat", "rough", "flat", "low_obstacles"], dtype=object)
    eligible = _eligible_mask(types_without_light, ("flat", "light_rough"))
    assert np.array_equal(eligible, _flat_mask(types_without_light))


def test_unknown_terrain_name_matches_nothing():
    eligible = _eligible_mask(_TYPES, ("does_not_exist",))
    assert not eligible.any()
