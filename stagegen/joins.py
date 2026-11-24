# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-FileCopyrightText: 2026 iceone88
# SPDX-License-Identifier: GPL-3.0-or-later AND MIT

"""Constrained, tile-boundary join shears for recombined road sections.

The shear is applied in the boundary-normal coordinate. The public join API
accepts a requested reach measured along each road; it converts this to the
normal-coordinate reach so shallow boundary crossings do not accidentally
stretch a nominal transition over several hundred metres of road.

The current compact-support profile is C1 (position and tangent continuous)
but not C2 at the outer edge. Replacing it with a higher-order profile is
intentionally deferred until source-stage curvature and in-game measurements
can be used to calibrate the trade-off.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _g(d: np.ndarray, D: float) -> np.ndarray:
    """Compact-support displacement profile; g(0)=0, g'(0)=1, g(D)=g'(D)=0."""
    if not np.isfinite(D) or D <= 0:
        raise ValueError("shear reach must be finite and positive")
    u = np.clip(d / D, 0.0, 1.0)
    return d * (1.0 - u) ** 2 * (d < D)


def _dg(d: np.ndarray, D: float) -> np.ndarray:
    if not np.isfinite(D) or D <= 0:
        raise ValueError("shear reach must be finite and positive")
    u = np.clip(d / D, 0.0, 1.0)
    return ((1.0 - u) ** 2 - 2.0 * u * (1.0 - u)) * (d < D)


@dataclass(frozen=True)
class JoinAssessment:
    """Geometric quantities required to construct both sides of a join."""

    a_in: float
    a_out: float
    kink_deg: float
    normal_in: float
    normal_out: float
    A_prev: float
    A_next: float
    Az_prev: float
    Az_next: float
    D_prev_normal: float
    D_next_normal: float
    max_boundary_parallel_shift_m: float
    max_vertical_shift_m: float
    reach_prev_m: float
    reach_next_m: float


def assess_join(line_angle: float, h_prev: float, h_next: float,
                g_prev: float, g_next: float,
                reach_prev_m: float, reach_next_m: float,
                max_kink_deg: float = 60.0,
                min_normal_component: float = 0.2) -> JoinAssessment | None:
    """Assess a join before modifying geometry; return None if ill-conditioned.

    ``reach_prev_m`` and ``reach_next_m`` are nominal source-road arclengths.
    The normal-component threshold caps the inverse boundary-projection factor
    at ``1 / min_normal_component``. It is a rejection criterion, not a
    denominator clamp: near-parallel crossings are not silently treated as
    ordinary crossings.
    """
    vals = (line_angle, h_prev, h_next, g_prev, g_next, reach_prev_m,
            reach_next_m, max_kink_deg, min_normal_component)
    if not all(np.isfinite(v) for v in vals):
        return None
    if reach_prev_m <= 0 or reach_next_m <= 0 or max_kink_deg < 0:
        return None
    if not (0 < min_normal_component < 1):
        return None

    e = np.array([np.cos(line_angle), np.sin(line_angle)], dtype=np.float64)
    n = np.array([-e[1], e[0]], dtype=np.float64)
    u_out = np.array([np.cos(h_next), np.sin(h_next)], dtype=np.float64)
    if u_out @ n < 0:
        n = -n
    u_in = np.array([np.cos(h_prev), np.sin(h_prev)], dtype=np.float64)
    normal_in = float(u_in @ n)
    normal_out = float(u_out @ n)
    if normal_in < min_normal_component or normal_out < min_normal_component:
        return None

    a_in = float(np.arctan2(u_in @ e, normal_in))
    a_out = float(np.arctan2(u_out @ e, normal_out))
    kink_deg = float(np.degrees(a_out - a_in))
    if abs(kink_deg) > max_kink_deg:
        return None

    # Solve the exact tangent-matching shear coefficients on either side.
    a_mid = (a_in + a_out) * 0.5
    tan_in, tan_out, tan_mid = np.tan(a_in), np.tan(a_out), np.tan(a_mid)
    A_next = tan_mid - tan_out
    A_prev = tan_in - tan_mid
    g_mid = (g_prev + g_next) * 0.5
    # Horizontal shear changes the horizontal speed along the transformed
    # curve. Its normal component remains cos(a), while its target direction
    # is a_mid, so |dXY_new/ds_source| = cos(a) / cos(a_mid). Match grade
    # dz / ds_horizontal after that speed change, not dz per source parameter.
    cos_mid = float(np.cos(a_mid))
    Az_next = g_mid / cos_mid - g_next / normal_out
    Az_prev = g_prev / normal_in - g_mid / cos_mid

    # The requested reach is measured along the un-deformed source road;
    # convert it to the boundary-normal coordinate used by Shear.apply().
    D_prev = reach_prev_m * normal_in
    D_next = reach_next_m * normal_out
    if min(D_prev, D_next) <= 1e-6:
        return None

    # max of d*(1-d/D)^2 is 4D/27 at d=D/3.
    max_boundary_parallel = max(abs(A_prev) * 4 * D_prev / 27,
                                 abs(A_next) * 4 * D_next / 27)
    max_vertical = max(abs(Az_prev) * 4 * D_prev / 27,
                       abs(Az_next) * 4 * D_next / 27)
    derived = (A_prev, A_next, Az_prev, Az_next, D_prev, D_next,
               max_boundary_parallel, max_vertical)
    if not all(np.isfinite(v) for v in derived):
        return None

    return JoinAssessment(a_in, a_out, abs(kink_deg), normal_in, normal_out,
                          A_prev, A_next, Az_prev, Az_next, D_prev, D_next,
                          max_boundary_parallel, max_vertical, reach_prev_m, reach_next_m)


def source_aligned_edge_interval(coordinate: float, grid_origin: float, cell: float) -> tuple[float, float]:
    """Finite tile-edge interval measured along the transformed source +axis.

    The source grid's positive axis is mapped to ``e`` by the piece's rigid
    rotation.  Thus the scalar interval remains ``[edge_start-coordinate,
    edge_end-coordinate]`` even when that transformed ``e`` points toward a
    negative world X or Y direction.
    """
    if not all(np.isfinite(v) for v in (coordinate, grid_origin, cell)) or cell <= 0:
        raise ValueError("tile edge coordinate/origin must be finite and cell positive")
    edge_start = grid_origin + np.floor((coordinate - grid_origin) / cell) * cell
    q0 = float(edge_start - coordinate)
    return q0, q0 + float(cell)


def seam_edge_weights(points: np.ndarray, p0: np.ndarray, e: np.ndarray,
                      along_min: float, along_max: float, width: float) -> tuple[np.ndarray, np.ndarray]:
    """Smooth blending weights only inside a finite, actual tile seam band.

    ``p0`` is a point on the shared tile edge, ``e`` its unit direction, and
    ``along_min/max`` bound the edge segment in metres from ``p0``.  Points
    outside that finite edge or farther than ``width`` from it get zero weight.
    This prevents unrelated roads (including stacked or crossing routes) from
    being treated as terrain neighbours merely because they are close in XY.
    """
    P = np.atleast_2d(np.asarray(points, dtype=np.float64))
    p0 = np.asarray(p0, dtype=np.float64)[:2]
    e = np.asarray(e, dtype=np.float64)[:2]
    en = float(np.linalg.norm(e))
    if not np.isfinite(width) or width <= 0 or en <= 1e-12:
        return np.zeros(len(P), dtype=bool), np.zeros(len(P), dtype=np.float64)
    e = e / en
    n = np.array([-e[1], e[0]])
    delta = P[:, :2] - p0
    q = delta @ e
    d = delta @ n
    mask = ((q >= min(along_min, along_max)) & (q <= max(along_min, along_max)) &
            (np.abs(d) < width))
    w = np.clip(1.0 - np.abs(d) / width, 0.0, 1.0)
    w = w * w * (3.0 - 2.0 * w)
    w = np.where(mask, w, 0.0)
    return mask, w


@dataclass(frozen=True)
class Shear:
    p0: np.ndarray        # (2,) a point on the join line (layout frame, m)
    e: np.ndarray         # (2,) unit vector along the line
    n: np.ndarray         # (2,) unit normal pointing into this side
    A: float              # along-line shear at the line (dx_e / dd)
    Az: float             # vertical shear at the line (dz / dd)
    D: float              # reach in the boundary-normal coordinate (m)

    def shifted(self, off: np.ndarray) -> "Shear":
        return Shear(self.p0 + off[:2], self.e, self.n, self.A, self.Az, self.D)

    @property
    def max_boundary_parallel_shift_m(self) -> float:
        """Maximum displacement parallel to the tile boundary, not road-lateral."""
        return abs(self.A) * 4.0 * self.D / 27.0

    @property
    def max_vertical_shift_m(self) -> float:
        return abs(self.Az) * 4.0 * self.D / 27.0

    def _d(self, xy):
        return (np.atleast_2d(xy)[:, :2] - self.p0) @ self.n

    def apply(self, P: np.ndarray) -> np.ndarray:
        """(N,2|3) -> same shape: points on this side within D moved."""
        P = np.array(P, np.float64, copy=True)
        d = self._d(P)
        side = d > 0
        if not side.any():
            return P
        g = np.where(side, _g(np.maximum(d, 0), self.D), 0.0)
        P[:, :2] += g[:, None] * self.A * self.e
        if P.shape[1] > 2:
            P[:, 2] += g * self.Az
        return P

    def yaw(self, xy: np.ndarray, heading: np.ndarray | float) -> np.ndarray:
        """Yaw change for the supplied heading under the shear Jacobian.

        For F(P)=P+g(d) A e and d=n·(P-p0), the horizontal Jacobian is
        J=I+A g'(d) e n^T.  A direction v must therefore map to Jv; the
        normal-only shortcut is correct only when v is parallel to n.
        """
        P = np.atleast_2d(np.asarray(xy, dtype=np.float64))
        h = np.broadcast_to(np.asarray(heading, dtype=np.float64), (len(P),))
        d = self._d(P)
        v = np.column_stack([np.cos(h), np.sin(h)])
        # The map is only piecewise differentiable at the seam. Use the
        # inward derivative g'(0+) for directions entering this piece, and
        # the identity derivative for directions leaving it.
        inward = (d > 0) | ((d == 0) & ((v @ self.n) > 0))
        dg = np.where(inward, _dg(np.maximum(d, 0), self.D), 0.0)
        mapped = v + (self.A * dg * (v @ self.n))[:, None] * self.e[None, :]
        return np.arctan2(mapped[:, 1], mapped[:, 0]) - h


def join_shears(p0, line_angle: float, h_prev: float, h_next: float,
                g_prev: float, g_next: float,
                reach_prev_m: float, reach_next_m: float,
                max_kink_deg: float = 60.0,
                min_normal_component: float = 0.2):
    """Return both shear transforms, or ``(None, None)`` for an unsafe join.

    Reach arguments are nominal source-road arclengths, not normal to the
    grid boundary. Use :func:`assess_join` to inspect predicted offsets.
    """
    assessment = assess_join(line_angle, h_prev, h_next, g_prev, g_next,
                             reach_prev_m, reach_next_m, max_kink_deg,
                             min_normal_component)
    if assessment is None:
        return None, None

    e = np.array([np.cos(line_angle), np.sin(line_angle)], dtype=np.float64)
    n = np.array([-e[1], e[0]], dtype=np.float64)
    if np.array([np.cos(h_next), np.sin(h_next)]) @ n < 0:
        n = -n
    p0 = np.asarray(p0, np.float64)[:2]
    if p0.shape != (2,) or not np.all(np.isfinite(p0)):
        return None, None
    return (
        Shear(p0, e, -n, assessment.A_prev, assessment.Az_prev, assessment.D_prev_normal),
        Shear(p0, e, n, assessment.A_next, assessment.Az_next, assessment.D_next_normal),
    )
