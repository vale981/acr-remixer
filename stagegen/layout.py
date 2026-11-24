# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Turn a search result into a flat layout: one centreline, one pacenote list,
and a manifest of exactly which source slices/transforms were used. The
manifest is what a future in-game build step consumes."""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from .geometry import arclength
from .model import Pacenote, Piece
from .pacenotes import generate
from .search import State


@dataclass
class Layout:
    pieces: list[Piece]
    centreline: np.ndarray
    widths: np.ndarray
    pacenotes: list[Pacenote]
    piece_ranges: list[tuple[float, float]]   # start/end distance of each piece

    @property
    def length(self) -> float:
        return float(arclength(self.centreline)[-1])

    @property
    def synthetic_share(self) -> float:
        syn = sum(b - a for (a, b), p in zip(self.piece_ranges, self.pieces) if p.kind == "connector")
        return syn / self.length


def build_layout(state: State, step: float, regen_joins: dict | None = None) -> Layout:
    """regen_joins: {join index k (between piece k-1 and k): reach m} - joins
    whose geometry was changed (smoothed kinks): there the original calls no
    longer fit, so within +-reach they are replaced by calls generated from
    the new route."""
    lay = _build_layout(state, step)
    if not regen_joins:
        return lay
    gen = generate(lay.centreline, step)
    wins = [(lay.piece_ranges[k][0] - r, lay.piece_ranges[k][0] + r) for k, r in regen_joins.items()]

    def inside(s):
        return any(a <= s <= b for a, b in wins)
    notes = [n for n in lay.pacenotes if not inside(n.s)]
    notes += [Pacenote(n.s, n.text, "generated") for n in gen if inside(n.s)]
    lay.pacenotes = sorted(notes, key=lambda n: n.s)
    return lay


def _build_layout(state: State, step: float) -> Layout:
    pieces = state.all_pieces()
    pts, ws, notes, ranges = [], [], [], []
    s = 0.0
    for i, p in enumerate(pieces):
        a = p.points if i == 0 else p.points[1:]       # drop duplicated joint point
        w = p.widths if i == 0 else p.widths[1:]
        plen = float(arclength(p.points)[-1])
        if p.kind == "segment" and p.pacenotes:
            notes += [Pacenote(n.s + s, n.text, "original", n.length) for n in p.pacenotes]
        else:
            # connectors and reversed segments: generate from geometry.
            notes += [Pacenote(n.s + s, n.text, "generated") for n in generate(p.points, step)]
        ranges.append((s, s + plen))
        pts.append(a)
        ws.append(w)
        s += plen
    return Layout(pieces, np.vstack(pts), np.concatenate(ws),
                  sorted(notes, key=lambda n: n.s), ranges)


def to_json(layout: Layout, meta: dict) -> str:
    return json.dumps({
        "format": "acr-stagegen/layout@0",
        **meta,
        "length_m": round(layout.length, 1),
        "synthetic_share": round(layout.synthetic_share, 4),
        "pieces": [{
            "kind": p.kind,
            "segment": p.segment_id,
            "reversed": p.reversed,
            "yaw_rad": round(float(p.yaw), 6),
            "translation": None if p.translation is None else [round(float(v), 3) for v in p.translation],
            "s_start": round(a, 1), "s_end": round(b, 1),
            # Connectors carry their own centreline; segments are reproduced from
            # source geometry + transform, so the build step never needs this copy.
            "centreline": [[round(float(v), 2) for v in q] for q in p.points[::5]] if p.kind == "connector" else None,
        } for p, (a, b) in zip(layout.pieces, layout.piece_ranges)],
        "pacenotes": [{"s": round(n.s, 1), "call": n.text, "origin": n.origin} for n in layout.pacenotes],
        # whole route, every ~10 m (x, y, z in m; layout frame): for previews
        "preview": [[round(float(v), 1) for v in q] for q in layout.centreline[::max(1, int(round(
            10.0 / max(1e-6, layout.length / max(1, len(layout.centreline) - 1)))))]],
    }, indent=1)
