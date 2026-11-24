# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-FileCopyrightText: 2026 iceone88
# SPDX-License-Identifier: MIT

"""Pacenote generation from centreline geometry.

Used for connectors, reversed segments, and to fake 'original' notes for test
stages. The scale is a common 1-6 (6 = fastest) system keyed on minimum radius.
Real ACR notes should be carried over wherever a segment is used forwards;
see Segment.pacenotes().
"""
from __future__ import annotations

import numpy as np

from .geometry import arclength, curvature, grade, smooth
from .model import Pacenote

# (max radius in m, call)
SCALE = [(15, "hairpin"), (25, "1"), (40, "2"), (60, "3"),
         (90, "4"), (140, "5"), (250, "6")]
CORNER_K = 1 / 250   # anything tighter than R250 is a corner
CUE_LEAD = 40.0      # trigger the call this far before the corner starts


def _call(radius: float) -> str:
    for r, name in SCALE:
        if radius <= r:
            return name
    return "6"


def generate(points: np.ndarray, step: float | None = None) -> list[Pacenote]:
    """Generate calls against the polyline's measured XY arclength.

    ``step`` remains accepted for compatibility, but is deliberately not used
    to reconstruct stations: resampled endpoints and exact tile cuts can have
    fractional intervals.
    """
    station = arclength(points)
    k = curvature(points, station, window=9)
    notes: list[tuple[float, float, str]] = []   # (start_s, end_s, text)
    corner = np.abs(k) > CORNER_K
    i, n = 0, len(k)
    while i < n:
        if not corner[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and corner[j + 1] and np.sign(k[j + 1]) == np.sign(k[i]):
            j += 1
        corner_length = float(station[j] - station[i])
        if corner_length >= 8.0:
            kmax = np.abs(k[i : j + 1]).max()
            side = "Right" if k[i] > 0 else "Left"
            call = _call(1 / kmax)
            text = f"{side}HP" if call == "hairpin" else f"{side}{call}"
            if corner_length > 160:
                text += " VeryLong"
            elif corner_length > 80:
                text += " Long"
            notes.append((float(station[i]), float(station[j]), text))
        i = j + 1

    out: list[Pacenote] = []
    for idx, (s0, s1, text) in enumerate(notes):
        if idx and s0 - notes[idx - 1][1] < 30:
            text = "Into " + text
        out.append(Pacenote(max(0.0, s0 - CUE_LEAD), text, "generated"))

    # Crests: strongly negative vertical curvature (grade dropping fast).
    g = smooth(grade(points, station, 9), 9)
    vc = np.gradient(g, station) if len(g) > 1 else np.zeros_like(g)
    crest = vc < -0.0025
    i = 0
    while i < n:
        if crest[i]:
            j = i
            while j + 1 < n and crest[j + 1]:
                j += 1
            mid = (i + j) // 2
            out.append(Pacenote(max(0.0, float(station[mid]) - CUE_LEAD), "Crest", "generated"))
            i = j + 1
        else:
            i += 1
    return sorted(out, key=lambda p: p.s)
