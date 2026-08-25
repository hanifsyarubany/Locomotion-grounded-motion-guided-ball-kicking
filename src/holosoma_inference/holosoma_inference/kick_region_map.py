"""Query interface for a probabilistic kick-success region map produced by
map_kick_region_prob.py. Shared between offline visualization/analysis and the classical HL
controller's threshold-gated trigger (ClassicalApproachAndKickController's
kick_region_map_path/success_probability_threshold params) -- one map format, one lookup
implementation, used identically whether it's being plotted or gating a real kick commit.

This is a synced copy: the source of truth is
playground/high_level_ball_approach_and_kicking/src/hl_kick/kick_region_map.py (where the sweep
that produces map files also lives). Copied rather than imported cross-project, matching this
session's established fork-per-workspace convention -- this module is deliberately dependency-free
(numpy/json/dataclasses/pathlib/math only, no simulator imports), so `holosoma_inference` (the real
deployment framework) never needs to depend on the IsaacSim-only sweep tooling at runtime, only on
the map *files* it produces. If you change the binning/statistics here, update both copies.

Statistical design note: `p_success` (the field everything here thresholds against) is NOT the
raw per-cell success fraction (k/n) -- it's the 95% Wilson score interval LOWER bound. A raw
fraction from a sparsely-sampled cell (e.g. 1/1 = 100%) would look deceptively safe; the Wilson
lower bound shrinks toward 0 as sample count drops, so a cell only reads as reliably safe once
enough trials have actually landed there. Cells with zero samples are 0.0, not 0.5 -- "never
tested" must never look like "safe" to a caller gating a physical kick commit on this value.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np


def wilson_lower_bound(k: int, n: int, z: float = 1.96) -> float:
    """95% Wilson score interval lower bound for a binomial proportion k/n. Returns 0.0 for n=0
    (no data -> no confidence, not "50/50")."""
    if n <= 0:
        return 0.0
    p_hat = k / n
    denom = 1.0 + z * z / n
    center = p_hat + z * z / (2 * n)
    adj = z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))
    return max(0.0, (center - adj) / denom)


@dataclasses.dataclass
class KickRegionMap:
    dx_edges: np.ndarray  # (nx+1,) bin edges, body-frame meters
    dy_edges: np.ndarray  # (ny+1,) bin edges, body-frame meters
    n_trials: np.ndarray  # (nx, ny) int, raw sample count per cell
    n_success: np.ndarray  # (nx, ny) int, raw success count per cell
    p_hat: np.ndarray  # (nx, ny) float, raw success fraction (k/n, 0.0 where n=0) -- reference only
    p_success: np.ndarray  # (nx, ny) float, Wilson lower bound -- THE field to threshold against
    kick_foot: str  # "left" | "right"
    kick_foot_body_name: str
    meta: dict  # everything else worth keeping (checkpoint path, sweep args, timestamp, ...)

    @property
    def dx_centers(self) -> np.ndarray:
        return 0.5 * (self.dx_edges[:-1] + self.dx_edges[1:])

    @property
    def dy_centers(self) -> np.ndarray:
        return 0.5 * (self.dy_edges[:-1] + self.dy_edges[1:])

    def probability_at(self, dx: float, dy: float) -> float:
        """Bilinear-interpolated p_success at a live body-frame (dx, dy). 0.0 for any query
        outside the mapped grid bounds -- extrapolating a safety claim past collected data isn't
        justified, and the same "unknown = not safe" convention as untested cells applies here."""
        xc, yc = self.dx_centers, self.dy_centers
        if dx < xc[0] or dx > xc[-1] or dy < yc[0] or dy > yc[-1]:
            return 0.0

        ix = int(np.searchsorted(xc, dx) - 1)
        ix = min(max(ix, 0), len(xc) - 2) if len(xc) > 1 else 0
        iy = int(np.searchsorted(yc, dy) - 1)
        iy = min(max(iy, 0), len(yc) - 2) if len(yc) > 1 else 0

        if len(xc) == 1 and len(yc) == 1:
            return float(self.p_success[0, 0])

        x0, x1 = xc[ix], xc[min(ix + 1, len(xc) - 1)]
        y0, y1 = yc[iy], yc[min(iy + 1, len(yc) - 1)]
        tx = 0.0 if x1 == x0 else (dx - x0) / (x1 - x0)
        ty = 0.0 if y1 == y0 else (dy - y0) / (y1 - y0)

        ix1 = min(ix + 1, len(xc) - 1)
        iy1 = min(iy + 1, len(yc) - 1)
        p00, p10 = self.p_success[ix, iy], self.p_success[ix1, iy]
        p01, p11 = self.p_success[ix, iy1], self.p_success[ix1, iy1]
        p0 = p00 * (1 - tx) + p10 * tx
        p1 = p01 * (1 - tx) + p11 * tx
        return float(p0 * (1 - ty) + p1 * ty)

    def best_point(self, min_samples: int = 20) -> tuple[float, float, float]:
        """(dx, dy, p_success) of the highest-p_success cell with at least min_samples support --
        an approach target that aims for the region's well-supported interior, not a noisy edge
        cell that happened to get lucky on a handful of trials."""
        eligible = self.n_trials >= min_samples
        if not eligible.any():
            raise ValueError(f"No cell has >= {min_samples} samples -- run more episodes or lower min_samples.")
        scored = np.where(eligible, self.p_success, -1.0)
        ix, iy = np.unravel_index(np.argmax(scored), scored.shape)
        return float(self.dx_centers[ix]), float(self.dy_centers[iy]), float(self.p_success[ix, iy])

    def region_above(self, threshold: float, min_samples: int = 20) -> list[tuple[float, float, float]]:
        """(dx, dy, p_success) for every cell at/above `threshold` with enough support -- the
        "safe region" the HL controller should treat as trigger-eligible at that confidence
        level."""
        eligible = (self.n_trials >= min_samples) & (self.p_success >= threshold)
        out = []
        for ix, iy in zip(*np.where(eligible)):
            out.append((float(self.dx_centers[ix]), float(self.dy_centers[iy]), float(self.p_success[ix, iy])))
        return out

    def save(self, path: str | Path) -> None:
        path = Path(path)
        np.savez(
            path.with_suffix(".npz"),
            dx_edges=self.dx_edges,
            dy_edges=self.dy_edges,
            n_trials=self.n_trials,
            n_success=self.n_success,
            p_hat=self.p_hat,
            p_success=self.p_success,
        )
        with open(path.with_suffix(".json"), "w") as f:
            json.dump(
                {"kick_foot": self.kick_foot, "kick_foot_body_name": self.kick_foot_body_name, "meta": self.meta},
                f,
                indent=2,
            )

    @classmethod
    def load(cls, path: str | Path) -> "KickRegionMap":
        path = Path(path)
        arrs = np.load(path.with_suffix(".npz"))
        with open(path.with_suffix(".json")) as f:
            side = json.load(f)
        return cls(
            dx_edges=arrs["dx_edges"],
            dy_edges=arrs["dy_edges"],
            n_trials=arrs["n_trials"],
            n_success=arrs["n_success"],
            p_hat=arrs["p_hat"],
            p_success=arrs["p_success"],
            kick_foot=side["kick_foot"],
            kick_foot_body_name=side["kick_foot_body_name"],
            meta=side["meta"],
        )

    @classmethod
    def from_samples(
        cls,
        dx: np.ndarray,
        dy: np.ndarray,
        success: np.ndarray,
        dx_min: float,
        dx_max: float,
        dy_min: float,
        dy_max: float,
        bin_size: float,
        kick_foot: str,
        kick_foot_body_name: str,
        meta: dict,
    ) -> "KickRegionMap":
        """Bin raw per-trial (dx, dy, success) samples (as produced by many random-spawn episodes
        in map_kick_region_prob.py) into a probability grid."""
        nx = max(1, round((dx_max - dx_min) / bin_size))
        ny = max(1, round((dy_max - dy_min) / bin_size))
        dx_edges = np.linspace(dx_min, dx_max, nx + 1)
        dy_edges = np.linspace(dy_min, dy_max, ny + 1)

        ix = np.clip(np.digitize(dx, dx_edges) - 1, 0, nx - 1)
        iy = np.clip(np.digitize(dy, dy_edges) - 1, 0, ny - 1)

        n_trials = np.zeros((nx, ny), dtype=np.int64)
        n_success = np.zeros((nx, ny), dtype=np.int64)
        np.add.at(n_trials, (ix, iy), 1)
        np.add.at(n_success, (ix, iy), success.astype(np.int64))

        p_hat = np.divide(n_success, n_trials, out=np.zeros_like(n_success, dtype=np.float64), where=n_trials > 0)
        p_success = np.zeros((nx, ny), dtype=np.float64)
        for i in range(nx):
            for j in range(ny):
                p_success[i, j] = wilson_lower_bound(int(n_success[i, j]), int(n_trials[i, j]))

        return cls(
            dx_edges=dx_edges,
            dy_edges=dy_edges,
            n_trials=n_trials,
            n_success=n_success,
            p_hat=p_hat,
            p_success=p_success,
            kick_foot=kick_foot,
            kick_foot_body_name=kick_foot_body_name,
            meta=meta,
        )
