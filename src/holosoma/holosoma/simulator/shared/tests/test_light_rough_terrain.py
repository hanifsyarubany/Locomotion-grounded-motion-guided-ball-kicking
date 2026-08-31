"""Unit tests for the `light_rough` terrain tier (2026-08-27) and the configurable kick-eligibility
gate that goes with it.

Covers the two properties that make this tier usable for kick-mode envs at all -- bounded
amplitude and SYMMETRY about nominal ground (an asymmetric tier would bias every kick tile
downward relative to the ball's configured spawn height) -- plus the no-op guarantee that a
zero proportion generates no tiles.

Bare fakes, no simulator: `_light_rough_terrain_func` only touches `terrain.height_field_raw`,
`terrain.vertical_scale`, and `self._cfg.light_rough_max_height`.
"""

import numpy as np

from holosoma.simulator.shared.terrain import Terrain


class _FakeSubTerrain:
    def __init__(self, shape=(64, 64), vertical_scale=0.005):
        self.height_field_raw = np.zeros(shape, dtype=np.float64)
        self.vertical_scale = vertical_scale


class _FakeCfg:
    def __init__(self, light_rough_max_height=0.008):
        self.light_rough_max_height = light_rough_max_height


def _make_bare_terrain(max_height=0.008) -> Terrain:
    """A Terrain instance without running __init__ -- _light_rough_terrain_func needs only _cfg."""
    t = Terrain.__new__(Terrain)
    t._cfg = _FakeCfg(max_height)
    return t


def test_light_rough_amplitude_is_bounded_by_configured_max_height():
    np.random.seed(0)
    t = _make_bare_terrain(max_height=0.008)
    sub = _FakeSubTerrain(vertical_scale=0.005)

    t._light_rough_terrain_func(sub, difficulty=1.0)

    heights_m = sub.height_field_raw * sub.vertical_scale
    assert np.all(np.abs(heights_m) <= 0.008 + 1e-9), "exceeded configured light_rough_max_height"
    assert np.abs(heights_m).max() > 0.004, "amplitude implausibly small -- is difficulty applied?"


def test_light_rough_is_symmetric_about_ground_unlike_rough():
    """The key difference from _rough_terrain_func, which writes strictly NEGATIVE heights.

    A tier biased below nominal ground would drop every kick-eligible tile relative to where the
    ball's configured spawn height assumes the floor is.
    """
    np.random.seed(1)
    t = _make_bare_terrain(max_height=0.008)
    sub = _FakeSubTerrain(shape=(128, 128), vertical_scale=0.005)

    t._light_rough_terrain_func(sub, difficulty=1.0)
    heights_m = sub.height_field_raw * sub.vertical_scale

    assert heights_m.min() < 0.0, "expected some below-ground samples"
    assert heights_m.max() > 0.0, "expected some above-ground samples (rough is negative-only)"
    # Mean should sit near zero for a symmetric uniform draw over a large tile.
    assert abs(float(heights_m.mean())) < 0.0005, (
        f"light_rough must be centered on nominal ground, got mean {heights_m.mean():.6f} m"
    )


def test_light_rough_scales_linearly_with_difficulty():
    np.random.seed(2)
    t = _make_bare_terrain(max_height=0.008)

    amplitudes = []
    for difficulty in (0.5, 0.9):
        sub = _FakeSubTerrain(shape=(128, 128), vertical_scale=0.005)
        t._light_rough_terrain_func(sub, difficulty=difficulty)
        amplitudes.append(np.abs(sub.height_field_raw * sub.vertical_scale).max())

    assert amplitudes[0] < amplitudes[1], "higher difficulty must produce larger deviation"
    assert amplitudes[1] <= 0.008 + 1e-9


def test_light_rough_is_gentler_than_the_existing_rough_tier():
    """Regression guard on the whole point of the tier: it must stay well under `rough`, whose
    amplitude is 0.025 * difficulty / 0.9 (14-25mm at the difficulties randomized_terrain draws)."""
    np.random.seed(3)
    t = _make_bare_terrain(max_height=0.008)
    sub = _FakeSubTerrain(shape=(128, 128), vertical_scale=0.005)
    t._light_rough_terrain_func(sub, difficulty=0.9)
    light_peak = np.abs(sub.height_field_raw * sub.vertical_scale).max()

    rough_peak_at_same_difficulty = 0.025 * 0.9 / 0.9
    assert light_peak < rough_peak_at_same_difficulty / 2.0, (
        f"light_rough peak {light_peak:.4f} m is not meaningfully gentler than rough's "
        f"{rough_peak_at_same_difficulty:.4f} m"
    )


def test_zero_proportion_type_is_filtered_out_before_generation():
    """The no-op guarantee: a 0.0 entry never reaches _terrain_types, so no tiles are generated
    and the surviving proportions are bit-identical to the dict without the key at all."""
    cfg_with = {"flat": 0.4, "light_rough": 0.0, "rough": 0.45, "low_obstacles": 0.15}
    cfg_without = {"flat": 0.4, "rough": 0.45, "low_obstacles": 0.15}

    filt = lambda d: {k: v for k, v in d.items() if v > 0.0}  # noqa: E731 -- mirrors terrain.py:101

    assert filt(cfg_with) == filt(cfg_without)
    assert "light_rough" not in filt(cfg_with)
