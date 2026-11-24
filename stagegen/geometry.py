# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-FileCopyrightText: 2026 iceone88
# SPDX-License-Identifier: GPL-3.0-or-later AND MIT

"""Polyline helpers. Coordinates are metres, x/y horizontal, z up."""

from __future__ import annotations

import numpy as np


def resample(points: np.ndarray, step: float) -> np.ndarray:
    """Resample a 3D polyline to approximately uniform spacing along XY arclength.

    The returned intervals are uniform, but their exact spacing is
    ``length / round(length / step)`` rather than necessarily ``step``.  The
    represented polyline can also be slightly shorter when a source bend falls
    between samples.  Consumers that need physical stationing must therefore
    use :func:`arclength` on the returned points, not ``index * step``.
    """
    d = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    n = max(2, int(round(s[-1] / step)) + 1)
    t = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(t, s, points[:, i]) for i in range(3)], axis=1)


def arclength(points: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])


def _stations(points: np.ndarray, station_or_step=None) -> np.ndarray:
    """Resolve explicit stations, a legacy scalar step, or XY arclength."""
    n = len(points)
    if station_or_step is None:
        s = arclength(points)
    elif np.ndim(station_or_step) == 0:
        step = float(station_or_step)
        if not np.isfinite(step) or step <= 0:
            raise ValueError("step must be finite and positive")
        s = np.arange(n, dtype=np.float64) * step
    else:
        s = np.asarray(station_or_step, dtype=np.float64)
        if s.shape != (n,):
            raise ValueError(f"station must have shape ({n},), got {s.shape}")
    if n > 1 and (not np.all(np.isfinite(s)) or np.any(np.diff(s) <= 0)):
        raise ValueError("station values must be finite and strictly increasing")
    return s


def headings(points: np.ndarray, station=None) -> np.ndarray:
    """Yaw (radians) at each point, unwrapped; derivatives use actual station."""
    P = np.asarray(points, dtype=np.float64)
    if len(P) < 2:
        return np.zeros(len(P), dtype=np.float64)
    s = _stations(P, station)
    v = np.gradient(P[:, :2], s, axis=0)
    return np.unwrap(np.arctan2(v[:, 1], v[:, 0]))


def smooth(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    k = np.ones(window) / window
    pad = window // 2
    xp = np.pad(x, pad, mode="edge")
    return np.convolve(xp, k, mode="valid")[: len(x)]


def curvature(points: np.ndarray, station=None, window: int = 5) -> np.ndarray:
    """Signed horizontal curvature (1/m); positive = turning left.

    ``station`` may be an explicit cumulative-distance array.  A scalar is
    retained as a backwards-compatible nominal step; new pipeline code should
    pass actual stations or omit it to use the polyline's XY arclength.
    """
    P = np.asarray(points, dtype=np.float64)
    if len(P) < 2:
        return np.zeros(len(P), dtype=np.float64)
    s = _stations(P, station)
    h = smooth(headings(P, s), window)
    return np.gradient(h, s)


def grade(points: np.ndarray, station=None, window: int = 5) -> np.ndarray:
    """dz/ds along the road, differentiated against actual station."""
    P = np.asarray(points, dtype=np.float64)
    if len(P) < 2:
        return np.zeros(len(P), dtype=np.float64)
    s = _stations(P, station)
    return np.gradient(smooth(P[:, 2], window), s)


def rot_z(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def wrap_angle(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi
