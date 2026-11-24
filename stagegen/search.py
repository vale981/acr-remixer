# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-FileCopyrightText: 2026 iceone88
# SPDX-License-Identifier: GPL-3.0-or-later AND MIT

"""Beam search that chains segments (via connectors) into a new stage layout."""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from .connector import connector, max_abs_grade
from .geometry import arclength, rot_z
from .model import EndState, Piece, Segment
from .segments import end_states


@dataclass
class SearchConfig:
    target_length: float = 10_000.0
    length_tolerance: float = 0.08      # finish within Â±8 %
    beam_width: int = 48
    branch_per_state: int = 24          # best expansions kept per parent
    yaw_offsets_deg: tuple = (-25, -12, 0, 12, 25)
    allow_reversed: bool = True
    corridor_half_width: float = 35.0   # scanned corridor each side of the road
    max_grade_mismatch: float = 0.06
    max_width_mismatch: float = 1.5
    max_connector_grade: float = 0.14
    max_elevation_range: float = 450.0
    same_surface_only: bool = True
    seed: int = 0
    variety: float = 12.0               # random per-segment preference; bigger = more seed-to-seed variation
    step: float = 2.0
    coarse_step: float = 10.0            # spacing used for overlap tests
    start_segment: str | None = None     # force this segment (forwards) as the opening piece
    end_segment: str | None = None       # force this segment (forwards) as the closing piece
    novelty: float = 0.0                 # penalty for re-using the source order (seg k+1 straight after seg k, both forwards)


@dataclass
class State:
    parent: "State | None"
    pieces: list[Piece]                 # pieces added by this step (connector, segment)
    end: EndState
    length: float
    used: frozenset
    coarse: np.ndarray                  # (M, 2) xy points added by this step
    zmin: float
    zmax: float
    penalty: float
    synthetic: float                    # total connector length so far
    last_stage: str
    surface: str
    children_tail_skip: int = 0         # coarse points at our tail to skip in overlap test
    max_geometry_mismatch: float = 0.0  # worst normalized connector/join constraint ratio
    geometry_penalty: float = 0.0        # sum of squared normalized geometry ratios
    preference_penalty: float = 0.0      # source variety/novelty preferences only

    def chain(self):
        s = self
        while s is not None:
            yield s
            s = s.parent

    def all_pieces(self) -> list[Piece]:
        out: list[Piece] = []
        for s in reversed(list(self.chain())):
            out.extend(s.pieces)
        return out


def _coarse(points: np.ndarray, step: float, coarse_step: float) -> np.ndarray:
    station = arclength(points)
    targets = np.arange(0.0, station[-1], max(1.0, coarse_step))
    idx = np.searchsorted(station, targets, side="left")
    idx = np.unique(np.r_[idx, len(points) - 1].clip(0, len(points) - 1))
    return points[idx, :2]


def _overlaps(parent: State, new_xy: np.ndarray, clearance: float, joint_skip: int) -> bool:
    """True if new_xy (minus its first joint_skip points, which sit next to the
    joint) comes within `clearance` of anything already laid down. The parent's
    own tail near the joint is skipped too."""
    test = new_xy[joint_skip:]
    if len(test) == 0:
        return False
    lo, hi = test.min(0) - clearance, test.max(0) + clearance
    c2 = clearance * clearance
    for i, s in enumerate(parent.chain()):
        pts = s.coarse[:-joint_skip] if i == 0 and joint_skip else s.coarse
        if len(pts) == 0:
            continue
        m = np.all((pts >= lo) & (pts <= hi), axis=1)
        if not m.any():
            continue
        p = pts[m]
        d2 = ((test[:, None, :] - p[None, :, :]) ** 2).sum(-1)
        if (d2 < c2).any():
            return True
    return False


def _place(seg: Segment, rev: bool, start_src: EndState, at: EndState, cfg: SearchConfig):
    """Rigidly place a segment so its start state lands on `at` (yaw + translation only)."""
    yaw = at.yaw - start_src.yaw
    R = rot_z(yaw)
    pts = (seg.points(rev) - start_src.pos) @ R.T + at.pos
    return pts, yaw, at.pos - R @ start_src.pos


def _initial_states(segments: list[Segment], cfg: SearchConfig) -> list[State]:
    out = []
    origin = EndState(np.zeros(3), 0.0, 0.0, 0.0)
    for seg in segments:
        if cfg.start_segment and seg.id != cfg.start_segment:
            continue
        for rev in ((False, True) if cfg.allow_reversed and not cfg.start_segment else (False,)):
            s0, e0 = end_states(seg, rev, cfg.step)
            pts, yaw, trans = _place(seg, rev, s0, EndState(np.zeros(3), 0.0, s0.grade, s0.width), cfg)
            _, e = end_states(seg, rev, cfg.step)
            R = rot_z(yaw)
            end = EndState(R @ e.pos + trans, e.yaw + yaw, e.grade, e.width)
            piece = Piece("segment", pts, seg.widths(rev), seg.id, rev, yaw, trans,
                          seg.pacenotes(rev) or [])
            out.append(State(None, [piece], end, seg.length, frozenset([seg.id]),
                             _coarse(pts, cfg.step, cfg.coarse_step),
                             float(pts[:, 2].min()), float(pts[:, 2].max()),
                             0.0, 0.0, seg.stage.name, seg.stage.surface))
    return out


def _expand(st: State, segments: list[Segment], cfg: SearchConfig, rng: random.Random,
            bias: dict[str, float]) -> list[State]:
    clearance = 2 * cfg.corridor_half_width
    joint_skip = int(np.ceil(clearance * 1.2 / cfg.coarse_step))
    kids: list[State] = []
    for seg in segments:
        if seg.id in st.used:
            continue
        if cfg.same_surface_only and seg.stage.surface != st.surface:
            continue
        is_end = seg.id == cfg.end_segment
        for rev in ((False, True) if cfg.allow_reversed and not is_end else (False,)):
            s0, e0 = end_states(seg, rev, cfg.step)
            dg = abs(s0.grade - st.end.grade)
            dw = abs(s0.width - st.end.width)
            if dg > cfg.max_grade_mismatch or dw > cfg.max_width_mismatch:
                continue
            for off in cfg.yaw_offsets_deg:
                dyaw = np.radians(off)
                L = float(np.clip(40 + 900 * dg + 160 * abs(dyaw) + 10 * dw, 40, 160))
                cpts, cw, cend = connector(st.end, st.end.yaw + dyaw, L, s0.grade, s0.width, cfg.step)
                if max_abs_grade(cpts) > cfg.max_connector_grade:
                    continue
                spts, yaw, trans = _place(seg, rev, s0, cend, cfg)
                zmin = min(st.zmin, cpts[:, 2].min(), spts[:, 2].min())
                zmax = max(st.zmax, cpts[:, 2].max(), spts[:, 2].max())
                if zmax - zmin > cfg.max_elevation_range:
                    continue
                new_xy = np.vstack([_coarse(cpts, cfg.step, cfg.coarse_step),
                                    _coarse(spts, cfg.step, cfg.coarse_step)])
                if _overlaps(st, new_xy, clearance, joint_skip):
                    continue
                R = rot_z(yaw)
                end = EndState(R @ e0.pos + trans, e0.yaw + yaw, e0.grade, e0.width)
                yaw_limit = max((abs(float(v)) for v in cfg.yaw_offsets_deg), default=0.0)
                ratios = (
                    _normalised_mismatch(dg, cfg.max_grade_mismatch),
                    _normalised_mismatch(abs(off), yaw_limit),
                    _normalised_mismatch(dw, cfg.max_width_mismatch),
                    _normalised_mismatch(max_abs_grade(cpts), cfg.max_connector_grade),
                )
                step_geometry = float(sum(q * q for q in ratios))
                worst_geometry = float(max(ratios))
                step_preference = ((3.0 if seg.stage.name == st.last_stage else 0.0)
                                   + (cfg.novelty if _follows_source(st, seg, rev) else 0.0)
                                   + bias[seg.id])
                # Keep the historic scalar for backwards-compatible inspection.
                penalty = (st.penalty
                           + 40 * dg / max(cfg.max_grade_mismatch, 1e-12)
                           + 6 * abs(off) / max(yaw_limit, 1e-12)
                           + 4 * dw
                           + step_preference)
                kid = State(
                    st,
                    [Piece("connector", cpts, cw),
                     Piece("segment", spts, seg.widths(rev), seg.id, rev, yaw, trans,
                           seg.pacenotes(rev) or [])],
                    end, st.length + L + seg.length, st.used | {seg.id}, new_xy,
                    float(zmin), float(zmax), penalty, st.synthetic + L,
                    seg.stage.name, st.surface)
                kid.max_geometry_mismatch = max(st.max_geometry_mismatch, worst_geometry)
                kid.geometry_penalty = st.geometry_penalty + step_geometry
                kid.preference_penalty = st.preference_penalty + step_preference
                kids.append(kid)
    rng.shuffle(kids)  # tie-break randomly so different seeds give different stages
    kids.sort(key=_score)
    return kids[: cfg.branch_per_state]


def _follows_source(st: State, seg: Segment, rev: bool) -> bool:
    last = st.pieces[-1]
    if rev or last.reversed or last.segment_id is None:
        return False
    a, _, ka = last.segment_id.rpartition("#")
    b, _, kb = seg.id.rpartition("#")
    return a == b and ka.isdigit() and kb.isdigit() and int(kb) == int(ka) + 1


def _normalised_mismatch(value: float, limit: float) -> float:
    """Absolute mismatch as a fraction of its configured hard limit."""
    value = abs(float(value))
    if limit <= 0:
        return 0.0 if value <= 1e-12 else float("inf")
    return value / float(limit)


def _rank_key(s: State) -> tuple[float]:
    """The scalar score (joins, connector share and preferences together), so
    the seed keeps deciding between similar candidates (see tiles._tile_rank_key)."""
    return (_score(s),)


def _score(s: State) -> float:
    """Legacy scalar score retained for callers outside the search loop."""
    km = max(s.length / 1000, 0.3)
    return s.penalty / km + 120 * s.synthetic / max(s.length, 1)


def search(segments: list[Segment], cfg: SearchConfig) -> State | None:
    rng = random.Random(cfg.seed)
    lo = cfg.target_length * (1 - cfg.length_tolerance)
    hi = cfg.target_length * (1 + cfg.length_tolerance)
    bias = {seg.id: rng.uniform(0, cfg.variety) for seg in segments}
    beam = _initial_states(segments, cfg)
    for st in beam:
        st.penalty = bias[st.pieces[0].segment_id]
        st.preference_penalty = float(bias[st.pieces[0].segment_id])
    rng.shuffle(beam)
    beam = sorted(beam, key=_rank_key)[: cfg.beam_width]
    finished: list[State] = []
    for _ in range(200):
        if not beam:
            break
        nxt: list[State] = []
        end_len = next((s.length for s in segments if s.id == cfg.end_segment), 0.0)
        for st in beam:
            for kid in _expand(st, segments, cfg, rng, bias):
                if cfg.end_segment:
                    if kid.pieces[-1].segment_id == cfg.end_segment:
                        if lo <= kid.length <= hi:
                            finished.append(kid)
                    elif kid.length + end_len + 40 < hi:
                        nxt.append(kid)
                    continue
                if lo <= kid.length <= hi:
                    finished.append(kid)
                elif kid.length < lo:
                    nxt.append(kid)
        # Diversity: dedupe states that just reached the same place by the same last segment.
        seen, uniq = set(), []
        for s in sorted(nxt, key=_rank_key):
            key = (s.pieces[-1].segment_id, round(s.end.pos[0] / 50), round(s.end.pos[1] / 50))
            if key not in seen:
                seen.add(key)
                uniq.append(s)
        beam = uniq[: cfg.beam_width]
        if len(finished) >= cfg.beam_width:
            break
    return min(finished, key=_rank_key) if finished else None
