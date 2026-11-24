# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-FileCopyrightText: 2026 iceone88
# SPDX-License-Identifier: GPL-3.0-or-later AND MIT

"""Turn a generated layout into an installable mod container.

The generated stage replaces one existing stage level (by default the WRC
Generations mod's WrcgEsSS1) through a higher-priority container that holds
only the three rewritten packages:

  * the level (.umap): scenery/gameplay actors of the used segments are moved
    with their segment's rigid transform to a fresh area next to the original
    map; the route splines (centre, ideal, borders, contour) are rebuilt; the
    runtime-virtual-texture volumes are moved over the new area.
  * DA_<Level>BakedData: the dmphysics collision triangles of the used
    segments, transformed the same way, plus generated connector road.
  * DT_Pacenote<Level>: the layout's calls (carried over + generated).

Removing the container restores the original stage.
"""

from __future__ import annotations

import collections
import os
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from ..extract.iostore import Container, GameFS
from ..extract.splines import read_spline_points
from ..extract.zen import Package
from ..geometry import arclength, headings, resample, rot_z
from ..layout import Layout
from ..model import Segment, SourceStage
from . import writers as W
from .cityhash import name_hash
from .iostore_write import ChunkIn, container_header_chunk_id, retarget_container_header, write_container
from .level import decode_components, patched_payload
from .zen_write import rewrite_package

SPLINES = ("CenterSpline", "IdealSpline", "LeftBorderSpline", "RightBorderSpline", "ContourSpline")


@dataclass
class BuildConfig:
    lateral_props: float = 60.0  # m from the source centreline for actors to come along
    lateral_tiles: float = 100.0  # terrain tiles / RVT writers (128 m tiles, pivot at centre)
    lateral_baked: float = 45.0  # collision triangles
    margin: float = 1500.0  # gap between the original map and the new stage (m)
    shoulder: float = 4.0  # connector verge width (m)
    ctrl_spacing: float = 8.0  # spline control point spacing (m)
    container_tag: str = "Stagegen"


@dataclass
class BuildReport:
    out_files: list = field(default_factory=list)
    moved: collections.Counter = field(default_factory=collections.Counter)
    relocated: list = field(default_factory=list)
    baked_meshes: int = 0
    baked_tris: int = 0
    connector_tris: int = 0
    notes: int = 0
    offset: tuple = (0, 0, 0)
    checks: list = field(default_factory=list)


def _actor_kind(name: str) -> str:
    base = name.split("_UAID")[0]
    for k in ("WrcgTerrain", "WrcgRvtWriter"):
        if base.startswith(k):
            return "tile"
    return base.rstrip("0123456789_") or base


class _Pieces:
    """Segment pieces of the layout with their source ranges and transforms."""

    def __init__(self, layout: Layout, seg_by_id: dict[str, Segment], offset: np.ndarray):
        self.items = []
        for p in layout.pieces:
            if p.kind != "segment":
                continue
            seg = seg_by_id[p.segment_id]
            self.items.append((seg.s0, seg.s1, rot_z(p.yaw), np.asarray(p.translation) + offset, p.yaw, p))
        self.items.sort(key=lambda t: t[0])
        self.lo = np.array([t[0] for t in self.items])
        self.hi = np.array([t[1] for t in self.items])

    def index(self, s: np.ndarray, first_open: bool, last_open: bool) -> np.ndarray:
        s = np.atleast_1d(s)
        k = np.searchsorted(self.lo, s, side="right") - 1
        ok = (k >= 0) & (s <= self.hi[np.clip(k, 0, None)])
        if first_open:
            ok |= s < self.lo[0]
            k = np.where(s < self.lo[0], 0, k)
        if last_open:
            ok |= s > self.hi[-1]
            k = np.where(s > self.hi[-1], len(self.items) - 1, k)
        return np.where(ok, k, -1)

    def apply(self, k: int, pts_m: np.ndarray) -> np.ndarray:
        _, _, R, t, _, _ = self.items[k]
        return pts_m @ R.T + t


def _level_paths(fs: GameFS, level: str):
    umap = next(p for p in fs.index if p.endswith(f"/Levels/{level}.umap"))
    baked = next(p for p in fs.index if p.endswith(f"DA_{level}BakedData.uasset"))
    pk = Package(fs.read(umap))
    tables = [
        n
        for n in pk.names
        if n.startswith("/Game/Data/Pacenote/Tables/") and level.lower() in n.lower() and "/DT_" in n
    ]
    table = "acr/Content/" + tables[0][len("/Game/") :] + ".uasset"
    return umap, baked, table, pk


def build_stage(
    layout: Layout,
    src: SourceStage,
    seg_by_id: dict[str, Segment],
    level: str,
    paks_dir: str,
    out_dir: str,
    cfg: BuildConfig | None = None,
) -> BuildReport:
    cfg = cfg or BuildConfig()
    rep = BuildReport()
    fs = GameFS(paks_dir, include_mods=True)
    umap_path, baked_path, table_path, lvl = _level_paths(fs, level)
    src_container, _ = fs.index[umap_path]

    comps = decode_components(lvl)
    roots = [c for c in comps if c.root and c.loc is not None]

    # ---- where to put the new stage: clear of everything in the original map
    old_xy = np.array([c.loc[:2] / 100 for c in roots])
    old_min, old_max = old_xy.min(0), old_xy.max(0)
    cl = layout.centreline
    offset = np.array([old_max[0] + cfg.margin - cl[:, 0].min(), old_min[1] - cl[:, 1].min(), 0.0])
    rep.offset = tuple(np.round(offset, 1))
    pieces = _Pieces(layout, seg_by_id, offset)
    new_cl = cl + offset
    new_s = arclength(new_cl)
    new_head = headings(new_cl)

    src_xy = cKDTree(src.points[:, :2])
    src_head = headings(src.points)
    src_station = src.station
    L_src = float(src_station[-1])

    # ---- level actors
    payloads: dict[int, bytes] = {}
    ex = lvl.exports
    for c in roots:
        owner = ex[c.owner].name if c.owner < len(ex) else ""
        kind = _actor_kind(owner)
        if c.rule.L == 26:  # runtime virtual texture volumes: cover the new area
            lo = np.r_[new_cl[:, :2].min(0) - 300, new_cl[:, 2].min() - 150]
            hi = np.r_[new_cl[:, :2].max(0) + 300, new_cl[:, 2].max() + 300]
            payloads[c.export] = patched_payload(
                lvl, c, lo * 100, c.rot if c.rot is not None else np.zeros(3), (hi - lo) * 100
            )
            rep.moved["RVT volume"] += 1
            continue
        p_m = c.loc / 100
        dist, idx = src_xy.query(p_m[:2])
        s = src_station[idx]
        limit = cfg.lateral_tiles if kind == "tile" else cfg.lateral_props
        k = int(pieces.index(np.array([s]), True, True)[0]) if dist <= limit else -1
        rot = c.rot if c.rot is not None else np.zeros(3)
        if kind.startswith("RaceSector") and k not in (0, len(pieces.items) - 1):
            k = -1  # intermediate splits: re-place along the route, facing forwards, in order
        if k >= 0:
            new_loc = pieces.apply(k, p_m[None])[0] * 100
            yaw = np.degrees(pieces.items[k][4])
            payloads[c.export] = patched_payload(lvl, c, new_loc, rot + np.array([0, yaw, 0]))
            rep.moved[kind] += 1
        elif kind.startswith("RaceSector"):
            # split/finish triggers: keep their relative position along the route
            s_new = s / L_src * new_s[-1]
            j = int(np.searchsorted(new_s, s_new))
            j = min(j, len(new_cl) - 1)
            dh = new_head[j] - src_head[idx]
            off = (p_m[:2] - src.points[idx, :2]) @ rot_z(dh)[:2, :2].T
            new_loc = np.r_[new_cl[j, :2] + off, new_cl[j, 2] + (p_m[2] - src.points[idx, 2])] * 100
            payloads[c.export] = patched_payload(lvl, c, new_loc, rot + np.array([0, np.degrees(dh), 0]))
            rep.relocated.append((owner, round(s), round(s_new)))

    # ---- visuals for connectors: transplant unused terrain tiles (+ their RVT
    # writers, which paint the road texture) whose source road matches.
    _fill_connectors(
        lvl, roots, payloads, layout, src, pieces, offset, new_cl, new_s, src_station, Package(fs.read(baked_path)), rep
    )

    # ---- splines
    ctrl = resample(new_cl, cfg.ctrl_spacing)
    s_ctrl = arclength(ctrl)
    w_ctrl = np.interp(s_ctrl, new_s, layout.widths)
    t = np.gradient(ctrl[:, :2], axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    n_left = np.c_[t[:, 1], -t[:, 0]]  # matches the source data's left side
    _, sc_P, _, _ = read_spline_points(lvl, "CenterSpline")
    _, co_P, _, _ = read_spline_points(lvl, "ContourSpline")
    dc, _ = cKDTree(sc_P[:, :2]).query(co_P[:, :2])
    contour_margin = max(1.0, float(np.median(dc) / 100 - np.median(src.width) / 2))
    left = np.c_[ctrl[:, :2] + n_left * (w_ctrl / 2)[:, None], ctrl[:, 2]]
    right = np.c_[ctrl[:, :2] - n_left * (w_ctrl / 2)[:, None], ctrl[:, 2]]
    cleft = np.c_[ctrl[:, :2] + n_left * (w_ctrl / 2 + contour_margin)[:, None], ctrl[:, 2]]
    cright = np.c_[ctrl[:, :2] - n_left * (w_ctrl / 2 + contour_margin)[:, None], ctrl[:, 2]]
    contour = np.vstack([cleft, cright[::-1], cleft[:1]])
    spline_pts = {
        "CenterSpline": ctrl,
        "IdealSpline": ctrl,
        "LeftBorderSpline": left,
        "RightBorderSpline": right,
        "ContourSpline": contour,
    }
    for name in SPLINES:
        i = next((i for i, e in enumerate(ex) if e.name == name), None)
        if i is not None:
            payloads[i] = W.spline_payload(lvl, name, spline_pts[name] * 100)
    new_umap = rewrite_package(lvl, payloads)

    # ---- baked collision
    bpk = Package(fs.read(baked_path))
    meshes, head, tb, tp, ts = W.read_baked_raw(bpk)
    out_meshes: list[W.MeshOut] = []
    mat_road = collections.Counter()
    mat_terr = collections.Counter()
    path_road = path_terr = None
    winding = []
    for m in meshes:
        if b"OriginQuad" in m.soft_sub:
            out_meshes.append(m)
            continue
        if len(m.indices) < 3:
            continue
        if b"Road" in m.soft_sub:
            mat_road[m.material_ref] += len(m.indices)
            path_road = path_road or m
            v = m.vertices
            tri = m.indices.reshape(-1, 3)[:200]
            nrm = np.cross(v[tri[:, 1]] - v[tri[:, 0]], v[tri[:, 2]] - v[tri[:, 0]])
            winding.append(np.sign(nrm[:, 1]).mean())
        elif b"Terrain" in m.soft_sub:
            mat_terr[m.material_ref] += len(m.indices)
            path_terr = path_terr or m
        U = m.vertices[:, [0, 2, 1]]  # baked (x, up, y) -> UE metres
        tri = m.indices.reshape(-1, 3)
        cen = U[tri].mean(1)
        dist, idx = src_xy.query(cen[:, :2])
        k = pieces.index(src_station[idx], True, True)
        k[dist > cfg.lateral_baked] = -1
        for kk in np.unique(k[k >= 0]):
            sel = tri[k == kk]
            used, inv = np.unique(sel, return_inverse=True)
            V = pieces.apply(int(kk), U[used])[:, [0, 2, 1]]
            out_meshes.append(
                W.MeshOut(
                    V, inv.ravel().astype(np.uint32), m.material_ref, m.soft_pkg, m.soft_asset, m.soft_sub, m.extra
                )
            )
            rep.baked_tris += len(sel)
    up_sign = 1.0 if (np.mean(winding) if winding else 1.0) >= 0 else -1.0

    def strip(pts_ue: np.ndarray, offs: list[tuple[float, float]], mesh_like: W.MeshOut, mat: int):
        """Loft a strip through stations; offs = [(lateral m, dz m), ...] left->right."""
        tt = np.gradient(pts_ue[:, :2], axis=0)
        tt /= np.linalg.norm(tt, axis=1, keepdims=True)
        nl = np.c_[tt[:, 1], -tt[:, 0]]
        rows = [np.c_[pts_ue[:, :2] + nl * o[:, None], pts_ue[:, 2] + dz] for o, dz in offs]
        V = np.stack(rows, 1).reshape(-1, 3)
        ncol = len(offs)
        tris = []
        for a in range(len(pts_ue) - 1):
            for b in range(ncol - 1):
                i0 = a * ncol + b
                q = [i0, i0 + 1, i0 + ncol + 1, i0 + ncol]
                tris += [(q[0], q[1], q[2]), (q[0], q[2], q[3])]
        tri = np.array(tris)
        Vb = V[:, [0, 2, 1]]
        nrm = np.cross(Vb[tri[:, 1]] - Vb[tri[:, 0]], Vb[tri[:, 2]] - Vb[tri[:, 0]])
        flip = np.sign(nrm[:, 1]) != up_sign
        tri[flip] = tri[flip][:, [0, 2, 1]]
        rep.connector_tris += len(tri)
        return W.MeshOut(
            Vb,
            tri.ravel().astype(np.uint32),
            mat,
            mesh_like.soft_pkg,
            mesh_like.soft_asset,
            mesh_like.soft_sub,
            mesh_like.extra,
        )

    rm = mat_road.most_common(1)[0][0] if mat_road else meshes[0].material_ref
    tm = mat_terr.most_common(1)[0][0] if mat_terr else rm
    for p in layout.pieces:
        if p.kind != "connector":
            continue
        P = p.points + offset
        w = np.asarray(p.widths)
        # extend a little into the neighbouring segments so there is no seam
        half = np.median(w) / 2
        out_meshes.append(
            strip(P, [(np.full(len(P), half), 0.0), (np.full(len(P), -half), 0.0)], path_road or meshes[0], rm)
        )
        out_meshes.append(
            strip(
                P,
                [(np.full(len(P), half + cfg.shoulder), -0.15), (np.full(len(P), half), 0.0)],
                path_terr or path_road or meshes[0],
                tm,
            )
        )
        out_meshes.append(
            strip(
                P,
                [(np.full(len(P), -half), 0.0), (np.full(len(P), -half - cfg.shoulder), -0.15)],
                path_terr or path_road or meshes[0],
                tm,
            )
        )
    rep.baked_meshes = len(out_meshes)
    new_baked = rewrite_package(bpk, {0: W.baked_payload(head, tb, out_meshes, tp, ts)})

    # ---- pacenotes
    tpk = Package(fs.read(table_path))
    notes = [
        W.NoteOut(float(n.s), float(getattr(n, "length", 30.0)), n.text.split())
        for n in layout.pacenotes
        if n.text.strip()
    ]
    pay, names = W.pacenote_payload(tpk, notes)
    new_table = rewrite_package(tpk, {0: pay}, names)
    rep.notes = len(notes)

    # ---- container
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.basename(src_container.utoc)[:-5]  # pakchunk99-WrcgEsSS1_P
    stem = base_name[:-2] if base_name.endswith("_P") else base_name
    pak_name = f"{stem}_{cfg.container_tag}_9_P"
    cid = name_hash(pak_name)
    hdr_idx = next(i for i, x in enumerate(src_container.chunk_ids) if x[11] == 6)
    chunks = []
    for path, data in ((umap_path, new_umap), (baked_path, new_baked), (table_path, new_table)):
        c, i = fs.index[path]
        chunks.append(ChunkIn(c.chunk_ids[i], data, path))
    chunks.append(
        ChunkIn(container_header_chunk_id(cid), retarget_container_header(src_container.read_entry(hdr_idx), cid))
    )
    with open(src_container.utoc[:-5] + ".pak", "rb") as f:
        pak_template = f.read()
    base = os.path.join(out_dir, pak_name)
    write_container(base, cid, chunks, pak_template)
    rep.out_files = [base + ext for ext in (".utoc", ".ucas", ".pak")]

    # ---- verify by reading our own output back
    rep.checks += verify(base + ".utoc", umap_path, baked_path, table_path, payloads, len(notes))
    return rep


TILE = 64.0  # half size of a terrain tile (m)


def _fill_connectors(lvl, roots, payloads, layout, src, pieces, offset, new_cl, new_s, src_station, bpk, rep):
    ex = lvl.exports
    tiles: dict[str, dict] = {}
    for c in roots:
        n = ex[c.owner].name
        for pre, key in (("WrcgTerrain_", "terrain"), ("WrcgRvtWriter_", "rvt")):
            if n.startswith(pre):
                tiles.setdefault(n[len(pre) :], {})[key] = c
    free = {k: v for k, v in tiles.items() if "terrain" in v and all(cc.export not in payloads for cc in v.values())}
    if not free:
        return
    keys = list(free)
    centres = np.array([free[k]["terrain"].loc[:2] / 100 for k in keys])

    # source terrain surface (collision) to check the transplanted tile never
    # rises over the drivable road
    meshes, *_ = W.read_baked_raw(bpk)
    terr = np.vstack([m.vertices[:, [0, 2, 1]] for m in meshes if b"Terrain" in m.soft_sub and len(m.vertices)])
    terr_tree = cKDTree(terr[:, :2])
    new_tree = cKDTree(new_cl[:, :2])

    P = src.points
    n = len(P)
    hd = headings(P)
    used_s = np.zeros(n, bool)
    for lo, hi, *_ in pieces.items:
        used_s |= (src_station >= lo) & (src_station <= hi)
    half_window = 60.0
    cand = []
    straightness = {}
    for i, si in enumerate(src_station):
        if si - src_station[0] < half_window or src_station[-1] - si < half_window:
            continue
        i0 = int(np.searchsorted(src_station, si - half_window, side="left"))
        i1 = int(np.searchsorted(src_station, si + half_window, side="right")) - 1
        if not used_s[i0 : i1 + 1].any():
            turn = abs(hd[i1] - hd[i0])
            if turn < np.radians(15):
                cand.append(i)
                straightness[i] = turn
    cand.sort(key=lambda i: straightness[i])
    taken: set[str] = set()

    for p in layout.pieces:
        if p.kind != "connector":
            continue
        C = p.points + offset
        A, B = C[0], C[-1]
        M = (A + B) / 2
        h = np.arctan2(B[1] - A[1], B[0] - A[0])
        half_len = np.linalg.norm(B[:2] - A[:2]) / 2
        _, ja = new_tree.query(A[:2])
        _, jb = new_tree.query(B[:2])
        s_a, s_b = new_s[min(ja, jb)], new_s[max(ja, jb)]
        placed = False
        for i in cand[:400]:
            Ps = P[i]
            # tiles holding the source road around i
            i0 = int(np.searchsorted(src_station, src_station[i] - half_len, side="left"))
            i1 = int(np.searchsorted(src_station, src_station[i] + half_len, side="right"))
            road = P[i0:i1, :2]
            sel = [k for k, cxy in zip(keys, centres) if k not in taken and (np.abs(road - cxy).max(1) <= TILE).any()]
            if not sel or len(sel) > 3:
                continue
            R = rot_z(h - hd[i])
            shift = M - Ps @ R.T
            # obstruction check: transplanted terrain inside the footprint vs new road elsewhere
            pts = []
            for k in sel:
                cxy = free[k]["terrain"].loc[:2] / 100
                idx = terr_tree.query_ball_point(cxy, TILE * 1.42)
                q = terr[idx]
                q = q[(np.abs(q[:, :2] - cxy) <= TILE).all(1)]
                pts.append(q)
            q = np.vstack(pts) @ R.T + shift if pts and sum(len(x) for x in pts) else np.zeros((0, 3))
            if len(q) == 0:
                continue
            d, j = new_tree.query(q[:, :2])
            outside = (new_s[j] < s_a - 5) | (new_s[j] > s_b + 5)
            on_road = (d < 5.0) & outside
            above = q[:, 2] - new_cl[j, 2]
            if (on_road & (above > 0.3)).any():
                continue
            for k in sel:
                for cc in free[k].values():
                    loc = cc.loc / 100 @ R.T + shift
                    rot = (cc.rot if cc.rot is not None else np.zeros(3)) + np.array([0, np.degrees(h - hd[i]), 0])
                    payloads[cc.export] = patched_payload(lvl, cc, loc * 100, rot)
                taken.add(k)
            rep.moved["connector tile"] += len(sel)
            placed = True
            break
        rep.checks.append(
            f"connector at {s_a:.0f}-{s_b:.0f} m: "
            + ("visual tile transplanted" if placed else "no safe tile found (collision only)")
        )


def verify(utoc: str, umap_path: str, baked_path: str, table_path: str, payloads, n_notes) -> list[str]:
    from ..extract.bakeddata import read_baked
    from ..extract.pacenote_table import read_pacenote_table
    from ..extract.splines import level_splines

    out = []
    c = Container(utoc)
    lvl = Package(c.read(umap_path))
    sp = level_splines(lvl)
    out.append(
        f"level: {len(lvl.exports)} exports, splines "
        + ", ".join(f"{k}={float(arclength(v)[-1]):.0f}m" for k, v in sp.items())
    )
    comps = {cc.export: cc for cc in decode_components(lvl)}
    patched = [i for i in payloads if i in comps]
    out.append(f"level: {len(patched)} patched components re-decode")
    meshes = read_baked(Package(c.read(baked_path)))
    nv = sum(len(m.vertices) for m in meshes)
    out.append(
        f"baked: {len(meshes)} meshes, {nv} vertices, all indices in range: "
        f"{all((m.indices.max() < len(m.vertices)) if len(m.indices) else True for m in meshes)}"
    )
    rows = read_pacenote_table(Package(c.read(table_path)))
    out.append(f"pacenotes: {len(rows)} rows (expected {n_notes})")
    return out
