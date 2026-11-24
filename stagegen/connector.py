# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Generated road pieces that bridge two segment ends.

Horizontal: cubic Hermite curve between the two headings (a gentle bend; yaw
offsets are capped so it stays a fast, unremarkable kink).
Vertical: cubic Hermite in z matching both grades, so there's no slope step.
Width: linear blend.

In-game, this centreline is what a mesh loft (road cross-section + verges +
collision) would be generated along. Scenery around it is fill, not scan.
"""

from __future__ import annotations

import numpy as np

from .model import EndState


def hermite(p0, p1, m0, m1, t):
    t = t[:, None] if np.ndim(p0) else t
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1


def connector(
    a: EndState, yaw_b: float, length: float, grade_b: float, width_b: float, step: float
) -> tuple[np.ndarray, np.ndarray, EndState]:
    """Build a connector starting at `a` and ending with heading `yaw_b` /
    grade `grade_b`. The end position is derived (not chosen), so the caller
    places the next segment where the connector ends.

    Returns (points, widths, end_state)."""
    dyaw = yaw_b - a.yaw
    # Chord of a circular arc of this length & total turn approximates the end pos.
    chord = length if abs(dyaw) < 1e-6 else 2 * length / dyaw * np.sin(dyaw / 2)
    mid = a.yaw + dyaw / 2
    end_xy = a.pos[:2] + chord * np.array([np.cos(mid), np.sin(mid)])
    # z end: average of the two grades over the length = smallest vertical curve.
    end_z = a.pos[2] + length * (a.grade + grade_b) / 2

    n = max(3, int(round(length / step)) + 1)
    t = np.linspace(0, 1, n)
    tan0 = length * np.array([np.cos(a.yaw), np.sin(a.yaw)])
    tan1 = length * np.array([np.cos(yaw_b), np.sin(yaw_b)])
    xy = hermite(a.pos[:2], end_xy, tan0, tan1, t)
    z = hermite(np.float64(a.pos[2]), np.float64(end_z), a.grade * length, grade_b * length, t)
    pts = np.column_stack([xy, z])
    widths = np.linspace(a.width, width_b, n)
    end = EndState(pts[-1].copy(), float(yaw_b), float(grade_b), float(width_b))
    return pts, widths, end


def max_abs_grade(points: np.ndarray) -> float:
    d = np.diff(points, axis=0)
    h = np.linalg.norm(d[:, :2], axis=1)
    return float(np.max(np.abs(d[:, 2] / np.maximum(h, 1e-6))))
