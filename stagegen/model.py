# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-FileCopyrightText: 2026 iceone88
# SPDX-License-Identifier: GPL-3.0-or-later AND MIT

"""Core data types: source stages, cut segments, end states, placed pieces."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import arclength


@dataclass
class Pacenote:
    s: float  # distance along the piece/stage where the call is triggered (m)
    text: str  # e.g. "4 left long", "crest", "into 2 right"
    origin: str = "original"  # "original" (carried over) or "generated"
    length: float = 30.0  # extent of the call (m), as stored in DT_Pacenote tables


@dataclass
class SourceStage:
    """A stage as extracted from the game: centreline + metadata."""

    name: str
    location: str  # e.g. "wales"; only segments from one location are mixed
    surface: str  # "gravel" / "tarmac" / "snow"
    points: np.ndarray  # (N, 3) centreline, resampled to `step`
    width: np.ndarray  # (N,) road width (m)
    pacenotes: list[Pacenote] = field(default_factory=list)
    # hash of the raw game data the route comes from (its spline control
    # points), independent of how they are sampled; "" for synthetic stages
    source_id: str = field(default="", compare=False)
    station: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.points = np.asarray(self.points, dtype=np.float64)
        self.width = np.asarray(self.width, dtype=np.float64)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError("SourceStage.points must have shape (N, 3)")
        if self.width.shape != (len(self.points),):
            raise ValueError("SourceStage.width must have shape (N,)")
        if len(self.points) < 2:
            raise ValueError("SourceStage requires at least two centreline points")
        if not np.all(np.isfinite(self.points)) or not np.all(np.isfinite(self.width)):
            raise ValueError("SourceStage geometry and widths must be finite")
        self.station = arclength(self.points)
        if np.any(np.diff(self.station) <= 0):
            raise ValueError("SourceStage centreline must have strictly increasing XY arclength")


@dataclass(frozen=True)
class EndState:
    pos: np.ndarray  # (3,)
    yaw: float  # radians
    grade: float  # dz/ds
    width: float  # m


@dataclass
class Segment:
    """A source interval, optionally bounded by exact interpolated cut points.

    ``i0``/``i1`` remain source-index bounds for compatibility with existing
    reports.  ``start_s``/``end_s`` and their endpoint values are authoritative
    for geometry and stationing, especially for tile-grid cuts between samples.
    """

    id: str
    stage: SourceStage
    i0: int
    i1: int
    step: float
    start_s: float | None = None
    end_s: float | None = None
    start_point: np.ndarray | None = None
    end_point: np.ndarray | None = None
    start_width: float | None = None
    end_width: float | None = None

    def __post_init__(self):
        if not (0 <= self.i0 < len(self.stage.points) and 0 <= self.i1 < len(self.stage.points)):
            raise ValueError("Segment source indices are out of range")
        if self.i1 < self.i0:
            raise ValueError("Segment i1 must be >= i0")
        s = self.stage.station
        self.start_s = float(s[self.i0] if self.start_s is None else self.start_s)
        self.end_s = float(s[self.i1] if self.end_s is None else self.end_s)
        if not (np.isfinite(self.start_s) and np.isfinite(self.end_s) and self.end_s > self.start_s):
            raise ValueError("Segment stations must be finite and increasing")
        self.start_point = (
            self.stage.points[self.i0].copy()
            if self.start_point is None
            else np.asarray(self.start_point, dtype=np.float64).copy()
        )
        self.end_point = (
            self.stage.points[self.i1].copy()
            if self.end_point is None
            else np.asarray(self.end_point, dtype=np.float64).copy()
        )
        if self.start_point.shape != (3,) or self.end_point.shape != (3,):
            raise ValueError("Segment endpoint points must have shape (3,)")
        self.start_width = float(self.stage.width[self.i0] if self.start_width is None else self.start_width)
        self.end_width = float(self.stage.width[self.i1] if self.end_width is None else self.end_width)

    def _interior_bounds(self):
        s = self.stage.station
        lo = int(np.searchsorted(s, self.start_s, side="right"))
        hi = int(np.searchsorted(s, self.end_s, side="left"))
        return lo, hi

    def points(self, reversed_: bool = False) -> np.ndarray:
        lo, hi = self._interior_bounds()
        p = np.vstack([self.start_point, self.stage.points[lo:hi], self.end_point])
        return p[::-1].copy() if reversed_ else p

    def widths(self, reversed_: bool = False) -> np.ndarray:
        lo, hi = self._interior_bounds()
        w = np.r_[self.start_width, self.stage.width[lo:hi], self.end_width]
        return w[::-1].copy() if reversed_ else w

    @property
    def length(self) -> float:
        return float(self.end_s - self.start_s)

    @property
    def s0(self) -> float:
        return float(self.start_s)

    @property
    def s1(self) -> float:
        return float(self.end_s)

    def pacenotes(self, reversed_: bool = False) -> list[Pacenote] | None:
        """Original notes re-keyed to exact segment-local station. None when reversed."""
        if reversed_:
            return None
        s0, s1 = self.s0, self.s1
        return [Pacenote(n.s - s0, n.text, "original", n.length) for n in self.stage.pacenotes if s0 <= n.s < s1]


@dataclass
class Piece:
    """One element of a generated layout: either a placed segment or a connector."""

    kind: str  # "segment" | "connector"
    points: np.ndarray  # world-space centreline (N, 3)
    widths: np.ndarray
    segment_id: str | None = None
    reversed: bool = False
    yaw: float = 0.0  # rigid transform applied to the source geometry
    translation: np.ndarray | None = None
    pacenotes: list[Pacenote] = field(default_factory=list)
