# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-FileCopyrightText: 2026 iceone88
# SPDX-License-Identifier: GPL-3.0-or-later AND MIT

"""Cut source stages into reusable segments at 'quiet' points of the road.

A good cut point is somewhere a seam will be least noticeable and easiest to
blend: low horizontal curvature, roughly constant grade, and steady width.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import curvature, grade
from .model import EndState, Segment, SourceStage


@dataclass
class CutConfig:
    step: float = 2.0
    max_curvature: float = 1 / 250  # straight-ish: radius > 250 m around the cut
    max_grade_change: float = 0.015  # grade must be steady across the window
    window_m: float = 30.0  # half-window that has to be quiet
    min_len: float = 250.0
    max_len: float = 1200.0
    anchor_ends: bool = False  # first segment starts at the stage start, last ends at the finish


def cut_candidates(stage: SourceStage, cfg: CutConfig) -> np.ndarray:
    s = stage.station
    k = np.abs(curvature(stage.points, s, window=7))
    g = grade(stage.points, s, window=7)
    n = len(stage.points)
    ok = np.zeros(n, dtype=bool)
    for i in range(n):
        if s[i] - s[0] < cfg.window_m or s[-1] - s[i] < cfg.window_m:
            continue
        lo = int(np.searchsorted(s, s[i] - cfg.window_m, side="left"))
        hi = int(np.searchsorted(s, s[i] + cfg.window_m, side="right"))
        win = slice(lo, hi)
        ok[i] = (
            k[win].max() < cfg.max_curvature
            and np.ptp(g[win]) < cfg.max_grade_change
            and np.ptp(stage.width[win]) < 0.6
        )
    # Always allow the original stage start/end as cut points.
    ok[0] = ok[-1] = True
    return np.flatnonzero(ok)


def cut_stage(stage: SourceStage, cfg: CutConfig) -> list[Segment]:
    """Greedy cut selection using measured source arclength, not index * step."""
    cand = cut_candidates(stage, cfg)
    station = stage.station
    segs: list[Segment] = []
    i = int(cand[0])
    if cfg.anchor_ends:
        first = cand[(station[cand] >= station[0] + cfg.min_len) & (station[cand] <= station[0] + cfg.max_len)]
        if len(first) == 0:
            first = cand[station[cand] >= station[0] + cfg.min_len][:1]
        if len(first):
            j = int(first[-1])
            segs.append(Segment(f"{stage.name}#0", stage, 0, j, cfg.step))
            i = j
    while True:
        lo_s = station[i] + cfg.min_len
        hi_s = station[i] + cfg.max_len
        nxt = cand[(station[cand] >= lo_s) & (station[cand] <= hi_s)]
        if len(nxt) == 0:
            later = cand[station[cand] > hi_s]
            if len(later) == 0:
                break
            i = int(later[0])
            continue
        j = int(nxt[-1])
        segs.append(Segment(f"{stage.name}#{len(segs)}", stage, int(i), j, cfg.step))
        i = j
    n = len(stage.points) - 1
    if cfg.anchor_ends and segs and segs[-1].i1 != n:
        last = segs[-1]
        if station[n] - station[last.i1] < cfg.min_len or last.i1 != i:
            segs[-1] = Segment(last.id, stage, last.i0, n, cfg.step)
        else:
            segs.append(Segment(f"{stage.name}#{len(segs)}", stage, last.i1, n, cfg.step))
    return segs


def end_states(seg: Segment, reversed_: bool, step: float) -> tuple[EndState, EndState]:
    """(start, end) state of a segment in its *source* coordinates."""
    p = seg.points(reversed_)
    w = seg.widths(reversed_)
    ps = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1))]
    span = min(20.0, max(1.0, ps[-1] * 0.25))
    end_span = max(1, min(len(p) - 1, int(np.searchsorted(ps, span, side="left"))))
    start_of_end = max(0, int(np.searchsorted(ps, ps[-1] - span, side="right")) - 1)

    def state(a: int, b: int, at: int) -> EndState:
        d = p[b] - p[a]
        horiz = np.linalg.norm(d[:2])
        return EndState(p[at].copy(), float(np.arctan2(d[1], d[0])), float(d[2] / horiz), float(w[at]))

    return state(0, end_span, 0), state(start_of_end, -1, -1)
