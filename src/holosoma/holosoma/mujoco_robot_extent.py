"""Exact world-z extent (top / bottom / total height) of the robot in a MuJoCo scene.

2026-08-21, user-requested: record the robot's FULL height -- bottom of the feet to top of the
head -- per tick during a kick rollout, alongside the base (root) height the recorders already
log. Base height alone says where the pelvis is; total height says how EXTENDED or CROUCHED the
whole robot is, which is a different signal (the pelvis can hold station while the legs bend).

Why not ``model.geom_rbound``: that is the bounding-SPHERE radius, which is far too loose for
this robot. The G1's foot collision capsules lie horizontally, so their rbound (~0.10 m) is
dominated by the capsule's half-LENGTH; ``geom_xpos_z - rbound`` then puts the robot's bottom at
-0.108 m -- BELOW THE FLOOR -- and its standing height at 1.444 m. Measured on this project's own
scene at the standing keyframe; that is why this module computes true per-geom z-extents instead:

    rbound method : top 1.336  bottom -0.108  height 1.444   <- wrong, bottom is under the floor
    this module   : top 1.288  bottom -0.001  height 1.289   <- feet on the floor, as expected

Per geom type the z half-extent is exact analytic geometry (capsule/cylinder/box/sphere/
ellipsoid); mesh geoms use their ACTUAL vertices transformed into world orientation, not a
bounding box, so a rotated link mesh is measured rather than over-estimated. Cost measured on
this scene (71 robot geoms, 38 of them meshes, 218k vertices total): 0.48 ms per call, i.e. 2.4%
of one 20 ms control tick -- affordable to call every tick, so no approximation is used.

The floor plane and the ball are excluded by body name (``exclude_bodies``), so "bottom" is the
robot's own lowest surface point, never the ground plane itself.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Bodies that are in the scene but are NOT the robot. 'world' carries the floor plane; 'ball' is
# the kickable ball added by scene_g1_29dof_with_ball.xml. Excluded so `bottom` means the robot's
# own lowest surface, and `top` can never be the ball mid-flight.
DEFAULT_EXCLUDE_BODIES = ("world", "ball")


def _geom_z_halfextent(mujoco: Any, model: Any, gi: int, rz: np.ndarray) -> float | None:
    """Half-extent along WORLD z for a primitive geom, given ``rz`` = the world-z row of the
    geom's rotation matrix (i.e. how the geom's own local axes project onto world z).

    Returns None for types this function does not handle analytically (mesh, plane) -- the caller
    deals with those. Sizes follow MuJoCo's own ``geom_size`` conventions: sphere (r), capsule
    (r, half_length), cylinder (r, half_length), box (half_x, half_y, half_z), ellipsoid (rx, ry,
    rz), with capsule/cylinder axes along the geom's LOCAL +z."""
    t = model.geom_type[gi]
    s = model.geom_size[gi]
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        return float(s[0])
    if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        # Hemispherical caps: the radius adds fully regardless of orientation.
        return float(abs(rz[2]) * s[1] + s[0])
    if t == mujoco.mjtGeom.mjGEOM_CYLINDER:
        # Flat end caps: the radius only contributes through the component perpendicular to the
        # axis, so a cylinder standing upright contributes half_length, not half_length + radius.
        return float(abs(rz[2]) * s[1] + s[0] * np.sqrt(max(0.0, 1.0 - float(rz[2]) ** 2)))
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        return float(np.abs(rz) @ s[:3])
    if t == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        return float(np.sqrt(np.sum((rz * s[:3]) ** 2)))
    return None


class RobotExtent:
    """Precomputes the static per-geom data needed to measure the robot's z extent, then reports
    (top, bottom, height) for whatever state ``data`` currently holds.

    Build once per model; call :meth:`measure` per tick.
    """

    def __init__(self, mujoco: Any, model: Any, exclude_bodies: tuple[str, ...] = DEFAULT_EXCLUDE_BODIES):
        self._mujoco = mujoco
        self._model = model
        body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
        self.geom_ids: list[int] = []
        self._mesh_verts: dict[int, np.ndarray] = {}
        for gi in range(model.ngeom):
            if body_names[model.geom_bodyid[gi]] in exclude_bodies:
                continue
            if model.geom_type[gi] == mujoco.mjtGeom.mjGEOM_PLANE:
                continue
            self.geom_ids.append(gi)
            if model.geom_type[gi] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = model.geom_dataid[gi]
                adr, num = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
                self._mesh_verts[gi] = np.asarray(
                    model.mesh_vert[adr : adr + num], dtype=np.float64
                ).reshape(-1, 3)
        if not self.geom_ids:
            raise ValueError(
                f"no robot geoms found (every geom belonged to {exclude_bodies} or was a plane) -- "
                "check exclude_bodies against this scene's actual body names"
            )

    def measure(self, data: Any) -> tuple[float, float, float]:
        """Returns ``(top_z, bottom_z, height)`` in world metres for the current ``data`` state."""
        top, bottom = -np.inf, np.inf
        for gi in self.geom_ids:
            rot = data.geom_xmat[gi].reshape(3, 3)
            z = float(data.geom_xpos[gi][2])
            verts = self._mesh_verts.get(gi)
            if verts is not None:
                # Only the world-z component of each vertex is needed, so project onto R's z row
                # rather than doing the full 3xN rotation.
                zs = verts @ rot[2]
                lo, hi = z + float(zs.min()), z + float(zs.max())
            else:
                half = _geom_z_halfextent(self._mujoco, self._model, gi, rot[2])
                if half is None:
                    continue
                lo, hi = z - half, z + half
            top = max(top, hi)
            bottom = min(bottom, lo)
        return top, bottom, top - bottom
