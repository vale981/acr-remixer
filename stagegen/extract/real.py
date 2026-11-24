# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Build SourceStage objects from the game's own data.

Per level:
  * centreline + road edges  <- SplinesActor (CenterSpline, Left/RightBorderSpline)
  * pacenotes                <- the DT_Pacenote* table the level references
  * surface                  <- dominant surface physical material in BakedData names

Levels are read straight from the IoStore containers; nothing is written.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from ..geometry import arclength, resample
from ..model import Pacenote, SourceStage
from .iostore import GameFS
from .pacenote_table import read_pacenote_table
from .splines import level_splines
from .zen import Package

DEFAULT_PAKS = os.path.join(os.path.dirname(__file__), "..", "..", "..", "acr", "Content", "Paks")


REGION_SURFACE = {"wales": "gravel", "greece": "gravel", "sweden": "snow", "livigno": "snow",
                  "alsace": "tarmac", "montecarlo": "tarmac", "wrcges": "tarmac"}


def _surface(pk: Package, level: str = "") -> str:
    for region, surf in REGION_SURFACE.items():
        if level.lower().startswith(region):
            return surf
    names = " ".join(pk.names).lower()
    for key, surf in (("snow", "snow"), ("ice", "snow"), ("gravel", "gravel"), ("tarmac", "tarmac")):
        if f"pm_surface{key}" in names:
            return surf
    return "tarmac"


def _pacenote_tables(pk: Package) -> list[str]:
    return [n for n in pk.names if n.startswith("/Game/Data/Pacenote/Tables/") and "DT_Pacenote" in n]


def _game_path(game_ref: str) -> str:
    return "acr/Content/" + game_ref[len("/Game/"):] + ".uasset"


def load_level(fs: GameFS, level_path: str, step: float = 2.0, name: str | None = None,
               location: str | None = None) -> SourceStage:
    pk = Package(fs.read(level_path))
    sp = level_splines(pk)
    if "CenterSpline" not in sp:
        # World Partition levels keep actors in _Generated_ cell packages.
        cell_dir = level_path[:-5] + "/_Generated_/"
        for cell in sorted(p for p in fs.index if p.startswith(cell_dir)):
            d = fs.read(cell)
            if b"CenterSpline" not in d:
                continue
            try:
                sp = level_splines(Package(d))
            except Exception:      # noqa: BLE001
                continue
            if "CenterSpline" in sp:
                break
    if "CenterSpline" not in sp:
        raise ValueError(f"{level_path}: no CenterSpline")
    centre = resample(sp["CenterSpline"], step)
    width = np.full(len(centre), 7.0)
    if "LeftBorderSpline" in sp and "RightBorderSpline" in sp:
        dl, _ = cKDTree(sp["LeftBorderSpline"][:, :2]).query(centre[:, :2])
        dr, _ = cKDTree(sp["RightBorderSpline"][:, :2]).query(centre[:, :2])
        width = np.clip(dl + dr, 3.0, 20.0)

    lvl = name or os.path.splitext(os.path.basename(level_path))[0]
    notes: list[Pacenote] = []
    # Prefer the table named after this level; mods sometimes leave a stray
    # reference to another stage's table in the name map.
    tables = sorted(_pacenote_tables(pk), key=lambda t: (lvl.lower() not in t.lower(), t))
    if not tables:
        # Base game: one table per variant, named after the stage without the
        # region/stage-number prefix (WalesS3HafrenNorth -> HafrenNorth).
        short = re.sub(r"^[A-Za-z]+?S\d+", "", lvl) or lvl
        tables = ["/Game/" + p[len("acr/Content/"):-len(".uasset")]
                  for p in fs.find(f"DT_Pacenote{short}FullForward") if p.endswith(".uasset")]
    for t in tables:
        path = _game_path(t)
        if path in fs.index:
            for row in read_pacenote_table(Package(fs.read(path))):
                notes.append(Pacenote(float(row.distance), " ".join(row.tokens), "original", float(row.length)))
            break
    # Table distances are measured along the original spline; rescale to
    # our resampled arclength (they agree to <0.5 % but keep it exact).
    L_src = float(arclength(sp["CenterSpline"])[-1])
    L_new = float(arclength(centre)[-1])
    for n in notes:
        n.s *= L_new / L_src
    loc = location or re.sub(r"(SS\d+|S\d+.*)$", "", lvl)
    return SourceStage(lvl, loc, _surface(pk, lvl), centre, width, sorted(notes, key=lambda n: n.s))


def _tables_for(fs: GameFS, lvl: str) -> dict[str, str]:
    """Base-game pacenote tables of a level: {'forward': path, 'reverse': path}."""
    short = re.sub(r"^[A-Za-z]+?S\d+", "", lvl) or lvl
    out = {}
    for p in fs.find(f"/DT_Pacenote{short}"):
        if not p.endswith(".uasset") or "Cut" in p:
            continue
        stem = p.rsplit("/", 1)[-1][:-7]
        for key in ("forward", "reverse"):
            if stem.lower().endswith(key) and (key not in out or "Full" in stem):
                out[key] = p
    return out


def _turn_agreement(points: np.ndarray, notes: list[Pacenote], step: float) -> float:
    from ..geometry import curvature
    s = arclength(points)
    k = curvature(points, s, window=9)
    good = bad = 0
    for n in notes:
        side = "Left" if "Left" in n.text else "Right" if "Right" in n.text else None
        if side is None:
            continue
        m = (s > n.s) & (s < n.s + 40)          # table distance ~ start of the corner
        if not m.any():
            continue
        left = k[m].mean() < 0                  # UE is left-handed: k > 0 turns right
        if left == (side == "Left"):
            good += 1
        else:
            bad += 1
    return good / max(good + bad, 1)


@dataclass
class WPSource:
    stage: SourceStage
    main: str
    cells: list
    route_cells: dict          # path -> "fwd" | "rev" | "cut"
    tables: dict               # "forward"/"reverse" -> path
    baked: str


def load_wp_source(fs: GameFS, lvl: str, step: float = 2.0) -> WPSource:
    """A World Partition base-game level, oriented like its Forward variant."""
    main = f"acr/Content/Levels/{lvl}.umap"
    cells = sorted(p for p in fs.index if p.startswith(main[:-5] + "/_Generated_/"))
    routes = {}
    for c in cells:
        d = fs.read(c)
        if b"CenterSpline" not in d:
            continue
        sp = level_splines(Package(d))
        if "CenterSpline" in sp:
            routes[c] = sp
    if not routes:
        raise ValueError(f"{lvl}: no route cells")
    L = {c: float(arclength(sp["CenterSpline"])[-1]) for c, sp in routes.items()}
    Lmax = max(L.values())
    full = [c for c in routes if L[c] > 0.97 * Lmax]
    tables = _tables_for(fs, lvl)
    notes: list[Pacenote] = []
    if "forward" in tables:
        notes = [Pacenote(float(r.distance), " ".join(r.tokens), "original", float(r.length))
                 for r in read_pacenote_table(Package(fs.read(tables["forward"])))]
    best = None
    for c in full:
        pts = resample(routes[c]["CenterSpline"], step)
        sc = _turn_agreement(pts, notes, step) if notes else 0.5
        if best is None or sc > best[1]:
            best = (c, sc, pts)
    fwd_cell, score, centre = best
    ref0 = centre[0, :2]
    kinds = {}
    for c, sp in routes.items():
        P = sp["CenterSpline"]
        if c not in full:
            kinds[c] = "cut"
        elif c == fwd_cell:
            kinds[c] = "fwd"
        else:
            kinds[c] = "fwd" if np.linalg.norm(P[0, :2] - ref0) < np.linalg.norm(P[-1, :2] - ref0) else "rev"
    sp = routes[fwd_cell]
    width = np.full(len(centre), 7.0)
    if "LeftBorderSpline" in sp and "RightBorderSpline" in sp:
        dl, _ = cKDTree(sp["LeftBorderSpline"][:, :2]).query(centre[:, :2])
        dr, _ = cKDTree(sp["RightBorderSpline"][:, :2]).query(centre[:, :2])
        width = np.clip(dl + dr, 3.0, 20.0)
    scale = float(arclength(centre)[-1]) / L[fwd_cell]
    for n in notes:
        n.s *= scale
    loc = re.sub(r"(SS\d+|S\d+.*)$", "", lvl)
    stage = SourceStage(lvl, loc, _surface(Package(fs.read(fwd_cell)), lvl), centre, width,
                        sorted(notes, key=lambda n: n.s))
    baked = next(p for p in fs.index if p.endswith(f"DA_{lvl}BakedData.uasset"))
    print(f"  {lvl}: forward route {fwd_cell.rsplit('/', 1)[-1]} (turn agreement {score:.0%}), "
          f"{sum(v == 'rev' for v in kinds.values())} reverse, {sum(v == 'cut' for v in kinds.values())} cut variants")
    return WPSource(stage, main, cells, kinds, tables, baked)


def find_levels(fs: GameFS) -> list[str]:
    return [p for p in fs.find("/Content/Levels/") if p.endswith(".umap") and "_Generated_" not in p]


def load_location(location: str, paks_dir: str = DEFAULT_PAKS, step: float = 2.0,
                  include_mods: bool = True) -> list[SourceStage]:
    fs = GameFS(paks_dir, include_mods=include_mods)
    stages = []
    for lv in find_levels(fs):
        base = os.path.splitext(os.path.basename(lv))[0]
        if not base.lower().startswith(location.lower()):
            continue
        try:
            stages.append(load_level(fs, lv, step, location=location))
        except Exception as ex:   # noqa: BLE001 - report and keep going
            print(f"  skip {base}: {ex}")
    return stages
