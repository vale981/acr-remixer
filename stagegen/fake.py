# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Synthetic 'source stages' for developing the generator before real data
is extracted. Roads are built from a curvature program (straights, corners
with linear-curvature transitions) plus smooth random elevation."""
from __future__ import annotations

import numpy as np

from .geometry import resample, smooth
from .model import Pacenote, SourceStage
from .pacenotes import generate


def fake_stage(name: str, length: float, seed: int, location="wales",
               surface="gravel", step=2.0) -> SourceStage:
    rng = np.random.default_rng(seed)
    n = int(length / step)
    k = np.zeros(n)
    i = 0
    while i < n:
        straight = int(rng.uniform(20, 300) / step)
        i += straight
        if i >= n:
            break
        radius = rng.choice([18, 30, 45, 70, 110, 180, 300], p=[.05, .1, .2, .25, .2, .12, .08])
        # cap the turn at ~160° so tight radii don't spiral into loops
        arc = int(min(rng.uniform(15, 90), radius * np.radians(160)) / step)
        trans = int(rng.uniform(10, 30) / step)
        sign = rng.choice([-1, 1])
        prof = np.concatenate([np.linspace(0, 1, trans), np.ones(arc), np.linspace(1, 0, trans)])
        j = min(n, i + len(prof))
        k[i:j] += sign / radius * prof[: j - i]
        i = j
    yaw = np.cumsum(k) * step
    xy = np.cumsum(np.column_stack([np.cos(yaw), np.sin(yaw)]) * step, axis=0)
    g = smooth(rng.normal(0, 1, n), 150) * 0.9
    g = np.clip(g / (np.abs(g).max() + 1e-9) * 0.12, -0.12, 0.12)
    z = 200 + np.cumsum(g) * step
    pts = resample(np.column_stack([xy, z]), step)
    width = 6.0 + smooth(rng.normal(0, 1, len(pts)), 300) * 3
    width = np.clip(width, 4.5, 8.0)
    notes = [Pacenote(p.s, p.text, "original") for p in generate(pts, step)]
    return SourceStage(name, location, surface, pts, width, notes)


def self_clear(points: np.ndarray, clearance: float, step: float, min_sep: float = 400.0) -> bool:
    """True if the road never comes within `clearance` of itself, ignoring
    points less than `min_sep` apart along the road. Real stages satisfy this
    by construction; random curvature programs often don't."""
    p = points[:: max(1, int(10 / step)), :2]
    gap = int(min_sep / 10)
    d2 = ((p[:, None] - p[None]) ** 2).sum(-1)
    idx = np.arange(len(p))
    far = np.abs(idx[:, None] - idx[None]) > gap
    return not (d2[far] < clearance ** 2).any()


def fake_location(seed: int = 1, clearance: float = 70.0) -> list[SourceStage]:
    names = ["Afon Biga", "Cwmbiga", "Hafren", "Dyfnant", "Sweet Lamb"]
    out = []
    for i, n in enumerate(names):
        length = float(np.random.default_rng(seed + i).uniform(7000, 12000))
        s = seed + i * 17
        while True:
            st = fake_stage(n, length, s)
            if self_clear(st.points, clearance, 2.0):
                break
            s += 1000
        out.append(st)
    return out
