# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Offline drive check of a built stage: probe the baked collision (what the
car drives on) along the route, the way the wheels would, without the game.

    python -m stagegen.cli check out/<build dir>

Every `step` m along the route and at each wheel line (lateral offsets) it
finds the highest collision surface at most `above` m over the route line and
reports:

- holes: no surface (the car drops in, e.g. a gap where two roads meet), or a
  pit: the surface more than `hole_depth` below the route line and below its
  own level `pit_window` m around;
- steps: the surface jumps more than `max_step` between two neighbouring
  samples, beyond what the route's own slope explains (the car hits an edge);
- surfaces: which physical material the wheels are on, in runs along the
  route; wheels off the road material (e.g. on grass or background terrain)
  are flagged;
- upside-down collision: ground under the wheels wound facing down (the
  underside of a solid to the physics: the car stops dead on it).

Lines in lower case are fine; UPPER CASE marks a problem (like the build
report's checks).
"""

from __future__ import annotations

import json
import os
import re
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from ..geometry import arclength, resample
from .writers import baked_up

WHEELS = (-0.8, 0.8)  # wheel lines, m left/right of the route line (track ~1.6 m)


@dataclass
class Collision:
    """Baked collision triangles in metres, world frame (x, y, z up)."""

    tris: np.ndarray  # (T, 3, 3)
    mesh: np.ndarray  # (T,) index into materials/actors
    up: np.ndarray  # (T,) up component of the unit normal in the game's winding (> 0: faces up)
    materials: list[str]  # per mesh: physical material short name
    actors: list[str]  # per mesh: short source actor name


def material_names(pk, refs) -> dict[int, str]:
    """Physical material name per raw package index (imports of the baked
    data package: FPackageObjectIndex type 2 = package import, the package
    is named in the imported package names)."""
    pkg_names = pk.imported_package_names()
    out = {}
    for ref in set(refs):
        name = f"import{-ref - 1}" if ref < 0 else f"export{ref}"
        if ref < 0 and -ref - 1 < len(pk.imports):
            v = pk.imports[-ref - 1]
            k = (v >> 32) & 0x3FFFFFFF
            if v >> 62 == 2 and k < len(pkg_names):
                name = pkg_names[k].rsplit("/", 1)[-1]
        out[ref] = name
    return out


def short_actor(soft_sub: bytes) -> str:
    s = soft_sub[4:].decode("latin-1", "replace").rstrip("\0")
    m = re.search(r"(Track\d{4}|Terrain\d{4}|BG\d*)", s)
    return m.group(1) if m else s.rsplit(".", 1)[-1][:40]


def collision_from_meshes(meshes, names: dict[int, str]) -> Collision:
    """MeshOut list (baked frame x, up, y) -> Collision."""
    tris, mesh, up = [], [], []
    for k, mo in enumerate(meshes):
        if len(mo.indices) < 3:
            continue
        V = np.asarray(mo.vertices, np.float64)[:, [0, 2, 1]]
        T = V[np.asarray(mo.indices, np.int64).reshape(-1, 3)]
        tris.append(T)
        mesh.append(np.full(len(T), k))
        up.append(baked_up(mo.vertices, mo.indices))
    return Collision(
        np.vstack(tris) if tris else np.zeros((0, 3, 3)),
        np.concatenate(mesh) if mesh else np.zeros(0, int),
        np.concatenate(up) if up else np.zeros(0),
        [names.get(mo.material_ref, str(mo.material_ref)) for mo in meshes],
        [short_actor(mo.soft_sub) for mo in meshes],
    )


def probe(col: Collision, Q: np.ndarray, zref: np.ndarray, above: float = 1.5):
    """Highest surface at each query point (x, y), at most `above` m over
    zref. Returns (z, triangle index) with NaN / -1 where there is none."""
    z_out = np.full(len(Q), np.nan)
    m_out = np.full(len(Q), -1)
    if not len(col.tris) or not len(Q):
        return z_out, m_out
    T = col.tris
    # only triangles near the query points
    lo, hi = Q.min(0) - 1.0, Q.max(0) + 1.0
    tlo, thi = T[:, :, :2].min(1), T[:, :, :2].max(1)
    near = np.all(thi >= lo, 1) & np.all(tlo <= hi, 1)
    sel0 = np.flatnonzero(near)
    T = T[sel0]
    C = T[:, :, :2].mean(1)
    rad = np.linalg.norm(T[:, :, :2] - C[:, None, :], axis=2).max(1)
    a, b, c3 = T[:, 0], T[:, 1], T[:, 2]
    v0, v1 = c3[:, :2] - a[:, :2], b[:, :2] - a[:, :2]
    den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    zbest = np.full(len(Q), -np.inf)
    mbest = np.full(len(Q), -1)
    # size classes, each searched with its own radius (big background
    # triangles would otherwise make every search huge)
    for lo_r, hi_r in ((0.0, 2.0), (2.0, 8.0), (8.0, 32.0), (32.0, np.inf)):
        sel = np.flatnonzero((rad > lo_r) & (rad <= hi_r)) if lo_r else np.flatnonzero(rad <= hi_r)
        if not len(sel):
            continue
        tree, R = cKDTree(C[sel]), float(rad[sel].max())
        for q0 in range(0, len(Q), 4000):
            cands = tree.query_ball_point(Q[q0 : q0 + 4000], R, workers=-1)
            lens = np.fromiter((len(cc) for cc in cands), int, len(cands))
            if not lens.sum():
                continue
            qi = np.repeat(np.arange(q0, q0 + len(cands)), lens)
            ci = sel[np.fromiter((x for cc in cands for x in cc), int, int(lens.sum()))]
            ok = den[ci] != 0
            qi, ci = qi[ok], ci[ok]
            v2 = Q[qi] - a[ci, :2]
            u = (v2[:, 0] * v1[ci, 1] - v1[ci, 0] * v2[:, 1]) / den[ci]
            v = (v0[ci, 0] * v2[:, 1] - v2[:, 0] * v0[ci, 1]) / den[ci]
            inside = (u >= -1e-6) & (v >= -1e-6) & (u + v <= 1 + 1e-6)
            qi, ci, u, v = qi[inside], ci[inside], u[inside], v[inside]
            zz = a[ci, 2] + u * (c3[ci, 2] - a[ci, 2]) + v * (b[ci, 2] - a[ci, 2])
            low = zz <= zref[qi] + above
            qi, ci, zz = qi[low], ci[low], zz[low]
            if not len(qi):
                continue
            # keep the highest hit per query point (sort, last per qi wins)
            o = np.lexsort((zz, qi))
            qi, ci, zz = qi[o], ci[o], zz[o]
            last = np.r_[qi[1:] != qi[:-1], True]
            qi, ci, zz = qi[last], ci[last], zz[last]
            better = zz > zbest[qi]
            zbest[qi[better]] = zz[better]
            mbest[qi[better]] = sel0[ci[better]]
    has = np.isfinite(zbest)
    z_out[has] = zbest[has]
    m_out[has] = mbest[has]
    return z_out, m_out


def _runs(s: np.ndarray, flag: np.ndarray, gap: float) -> list[tuple[float, float]]:
    """Runs of flagged stations, merged across gaps shorter than `gap` m."""
    out = []
    for x in s[flag]:
        if out and x - out[-1][1] <= gap:
            out[-1][1] = float(x)
        else:
            out.append([float(x), float(x)])
    return [(a, b) for a, b in out]


def _fmt_runs(runs, n=8) -> str:
    txt = ", ".join(f"{a:.0f} m" if b - a < 1 else f"{a:.0f}-{b:.0f} m" for a, b in runs[:n])
    return txt + (f" ... ({len(runs)} in all)" if len(runs) > n else "")


def _pct(x: float) -> str:
    return "<1 %" if 0 < x < 0.005 else f"{100 * x:.0f} %"


def _rolling_nanmedian(v: np.ndarray, k: int) -> np.ndarray:
    """Median of v over i-k..i+k, ignoring NaN."""
    pad = np.r_[np.full(k, np.nan), v, np.full(k, np.nan)]
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * k + 1)
    with warnings.catch_warnings():  # all-NaN windows: NaN, quietly
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(win, axis=1)


@dataclass
class DriveReport:
    length: float
    lines: list[str] = field(default_factory=list)
    holes: list = field(default_factory=list)  # [(s0, s1)]
    steps: list = field(default_factory=list)  # [(s, lateral, jump)]
    surfaces: list = field(default_factory=list)  # [(s0, s1, material, actor)] at the route line
    off_road: list = field(default_factory=list)  # [(s0, s1)] wheels off the road material
    upside_down: list = field(default_factory=list)  # [(s0, s1)] ground wound facing down
    route_above: float = 0.0  # largest distance between the route line and the surface (m), for information

    @property
    def ok(self) -> bool:
        return not (self.holes or self.steps or self.off_road or self.upside_down)


def drive_check(
    col: Collision,
    route: np.ndarray,
    step: float = 0.25,
    wheels=WHEELS,
    hole_depth: float = 0.3,
    max_step: float = 0.08,
    road_ref_m: float = 100.0,
    min_off_road: float = 2.0,
    pit_window: float = 10.0,
) -> DriveReport:
    """Probe the collision along `route` ((N, 3) world metres, the route
    line). The road material is the one most often under the route line in
    its first `road_ref_m` m (the official start area)."""
    P = resample(route, step)
    s = arclength(P)
    t = np.gradient(P[:, :2], axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    n = np.c_[-t[:, 1], t[:, 0]]
    rep = DriveReport(float(s[-1]))
    lats = (0.0, *wheels)
    Z, M, F = {}, {}, {}
    for lat in lats:
        Z[lat], tri = probe(col, P[:, :2] + n * lat, P[:, 2])
        M[lat] = np.where(tri >= 0, col.mesh[np.maximum(tri, 0)], -1)
        F[lat] = (tri >= 0) & (col.up[np.maximum(tri, 0)] < 0)

    # holes: nothing under a wheel, or a pit: the surface drops more than
    # hole_depth below the route line *and* below its own level around it.
    # (The route line itself may run a few decimetres above the road over
    # tens of metres, e.g. at smoothed joins: the car does not notice that.)
    hole = np.zeros(len(P), bool)
    k = max(1, int(round(pit_window / step)))
    below = {}
    for lat in lats:
        below[lat] = P[:, 2] - Z[lat]  # how far the surface is under the route line
        level = _rolling_nanmedian(below[lat], k)
        hole |= ~np.isfinite(Z[lat]) | ((below[lat] > hole_depth) & (below[lat] - level > hole_depth))
    rep.holes = _runs(s, hole, gap=2.0)
    with np.errstate(all="ignore"):
        rep.route_above = float(np.nanmax([np.nanmax(np.abs(b)) for b in below.values()]))

    # steps: jumps between neighbours beyond the route's own slope
    dz_route = np.diff(P[:, 2])
    found = {}
    for lat in lats:
        jump = np.diff(Z[lat]) - dz_route
        for i in np.flatnonzero(np.abs(np.nan_to_num(jump)) > max_step):
            key = round(s[i + 1])
            if key not in found or abs(jump[i]) > abs(found[key][2]):
                found[key] = (float(s[i + 1]), lat, float(jump[i]))
    # one entry per metre and spot (neighbouring metres merged)
    steps = []
    for v in sorted(found.values()):
        if steps and v[0] - steps[-1][0] < 1.5:
            if abs(v[2]) > abs(steps[-1][2]):
                steps[-1] = v
        else:
            steps.append(v)
    rep.steps = steps

    # surfaces at the route line, in runs of the same mesh material/actor
    m0 = M[0.0]
    key = [(col.materials[m], col.actors[m]) if m >= 0 else ("none", "") for m in m0]
    a = 0
    for i in range(1, len(key) + 1):
        if i == len(key) or key[i] != key[a]:
            rep.surfaces.append((float(s[a]), float(s[i - 1]), *key[a]))
            a = i
    ref = s <= road_ref_m
    mats = [col.materials[m] for m in m0[ref] if m >= 0]
    road_mat = max(set(mats), key=mats.count) if mats else None
    off = np.zeros(len(P), bool)
    for lat in lats:
        has = M[lat] >= 0
        off[has] |= np.array([col.materials[m] != road_mat for m in M[lat][has]])
    rep.off_road = [(a_, b_) for a_, b_ in _runs(s, off, gap=1.0) if b_ - a_ >= min_off_road]
    # ground wound the other way round: the underside of a solid to the car,
    # which stops dead driving onto it from correctly wound ground
    rep.upside_down = _runs(s, np.logical_or.reduce([F[lat] for lat in lats]), gap=1.0)

    # report lines
    L = rep.lines
    where = f"every {step:g} m at the route line and the wheels ({', '.join(f'{w:+g} m' for w in wheels)})"
    if rep.holes:
        L.append(f"HOLES under the wheels (no ground, or a pit over {hole_depth:g} m deep) at " + _fmt_runs(rep.holes))
    else:
        L.append(f"holes: none ({where}; the route line is within {rep.route_above * 100:.0f} cm of the surface)")
    if rep.steps:
        L.append(
            f"STEPS in the road (> {max_step * 100:.0f} cm between samples {step:g} m apart) at "
            + ", ".join(f"{s_:.0f} m ({lat:+g} m: {dz * 100:+.0f} cm)" for s_, lat, dz in rep.steps[:8])
            + (f" ... ({len(rep.steps)} in all)" if len(rep.steps) > 8 else "")
        )
    else:
        worst = max((np.nanmax(np.abs(np.diff(Z[lat]) - dz_route), initial=0) for lat in lats), default=0)
        L.append(f"steps: none over {max_step * 100:.0f} cm (largest {worst * 100:.1f} cm)")
    share = {}
    for a_, b_, mat, _ in rep.surfaces:
        share[mat] = share.get(mat, 0.0) + (b_ - a_ + step)
    L.append(
        "surface at the route line: "
        + ", ".join(f"{m} {_pct(v / (rep.length + step))}" for m, v in sorted(share.items(), key=lambda x: -x[1]))
        + f" (road material: {road_mat}, from the first {road_ref_m:g} m)"
    )
    if rep.off_road:
        L.append(f"WHEELS OFF THE ROAD MATERIAL ({road_mat}) at " + _fmt_runs(rep.off_road))
    else:
        L.append(f"wheels: on {road_mat} all the way")
    if rep.upside_down:
        L.append(
            "UPSIDE-DOWN COLLISION under the wheels (wound facing down: the car hits it) at "
            + _fmt_runs(rep.upside_down)
        )
    else:
        L.append("collision: faces up all the way")
    return rep


# ------------------------------------------------------------ built stages
def load_build(build_dir: str):
    """(Collision, {name: route}) from a build directory's container.

    Routes are the CenterSplines of the level's route cells; when the
    build's layout.json is there, the forward one is named 'forward' (its
    height profile matches the layout preview), the other 'reverse'."""
    from ..extract.iostore import Container
    from ..extract.splines import level_splines
    from ..extract.zen import Package
    from .writers import read_baked_raw

    utocs = [f for f in os.listdir(build_dir) if f.endswith(".utoc")]
    if len(utocs) != 1:
        raise SystemExit(f"expected one .utoc in {build_dir}, found {len(utocs)}")
    c = Container(os.path.join(build_dir, utocs[0]))
    baked = [p for p in c.files if p.endswith("BakedData.uasset")]
    if not baked:
        raise SystemExit("no baked collision (DA_*BakedData) in the container")
    pk = Package(c.read(baked[0]))
    meshes = read_baked_raw(pk)[0]
    col = collision_from_meshes(meshes, material_names(pk, [m.material_ref for m in meshes]))
    routes = {}
    for p in c.files:
        if "/_Generated_/" in p and p.endswith(".umap"):
            d = c.read(p)
            if b"CenterSpline" in d:
                sp = level_splines(Package(d))
                if "CenterSpline" in sp and len(sp["CenterSpline"]) > 1:
                    routes[p.rsplit("/", 1)[-1][:-5]] = np.asarray(sp["CenterSpline"], float)
    lay = os.path.join(build_dir, "layout.json")
    if len(routes) == 2 and os.path.exists(lay):
        prev = np.asarray(json.load(open(lay))["preview"], float)
        if len(prev) > 2:
            zp = resample(prev, 5.0)[:, 2]

            def mismatch(r):
                z = resample(r, 5.0)[:, 2]
                k = min(len(z), len(zp))
                return float(np.abs(z[:k] - zp[:k]).mean())

            (fa, ra), (fb, rb) = routes.items()
            fwd, rev = ((fa, ra), (fb, rb)) if mismatch(ra) <= mismatch(rb) else ((fb, rb), (fa, ra))
            routes = {"forward": fwd[1], "reverse": rev[1]}
    return col, routes
