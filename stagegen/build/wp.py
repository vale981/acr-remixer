# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-FileCopyrightText: 2026 iceone88
# SPDX-License-Identifier: GPL-3.0-or-later AND MIT

"""Build a recombined base-game (World Partition) stage as a mod container.

What gets rewritten (everything else is still served by the game):
  * main level: streaming LoadingRange raised so every cell stays loaded
    (cells' actors no longer sit where the grid expects them); RVT volume
    moved over the new area.
  * every World Partition cell that holds a moved actor: root scene
    components get their piece's rigid transform. Track tiles move whole.
  * route cells (Forward/Reverse data layers): splines rebuilt along the new
    route, start/finish/split triggers re-placed.
  * the track tile meshes that are used: triangles outside the piece's road
    corridor are made degenerate, so no terrain sticks into other pieces.
  * DA_<Level>BakedData collision: tile triangles clipped with the same rule
    and transformed; props follow their piece.
  * DT_Pacenote<Name>Forward/Reverse.
"""

from __future__ import annotations

import collections
import os
import re
import struct
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from ..extract.iostore import Container, GameFS, base_game_fs
from ..extract.real import WPSource
from ..extract.splines import level_splines, read_spline_points
from ..extract.zen import Package
from ..geometry import arclength, headings, resample, rot_z
from ..layout import Layout
from ..pacenotes import generate
from ..tiles import Cut, TileGrid, TState

from . import writers as W
from .cityhash import name_hash
from . import container_header as CH
from .iostore_write import ChunkIn, container_header_chunk_id, retarget_container_header, write_container
from .rename import level_renamer, rename_package
from .level import decode_components, learn_rules, patched_payload
from .mesh_clip import clip_mesh_payload, displace_payload, find_lods, mesh_export_index, warp_payload
from .wp_hash import all_cells_in_top_level, drop_cells, widen_cell_data
from ..progress import progress, span
from .foliage import hide_instances, move_hism
from .hlod import mesh_world_bounds, transform_mesh
from .zen_write import rewrite_package

SPLINES = ("CenterSpline", "IdealSpline", "LeftBorderSpline", "RightBorderSpline", "ContourSpline")
SKIP_OWNERS = (
    "InstancedFoliageActor",
    "PCG",
    "HLOD",
    "WorldDataLayers",
    "WorldSettings",
    "RespawnPlanes",
    "BC_ReplayCamerasActor",
    "SplinesActor",
)
SKIP_WORDS = ("Sky", "Weather", "Fog", "Cloud", "Atmosphere", "PostProcess", "Moon", "Sun")


@dataclass
class WPConfig:
    prop_lateral: float = 60.0  # m: props/decals this close to the route move with their piece
    terrain_clip: float = 2000.0  # m: tile terrain kept this far either side of its road (ownership
    # still decides; a narrow corridor looks like floating ribbons from afar)
    road_clip: float = 30.0
    range_ext: float = 80.0  # m: along-route slack at piece ends (tiles end at the grid line anyway)
    end_radius: float = 400.0  # m: around the route ends (start/finish areas) keep everything this far
    margin: float = 1500.0  # m: gap between the original map and the new stage
    ctrl_spacing: float = 8.0
    loading_range_cm: float = 3.0e6
    tag: str = "Stagegen"
    debug_spawn_s: float | None = None  # test builds: spawn the car this far along the new route
    all_cells_loaded: bool = False  # experimental, hides HLODs (see build_wp_stage)
    hlod_visuals: bool = False  # experiment: tile cells never load (also hides the forest)
    hlod_hide: bool = True  # empty all HLOD proxy meshes (cells are always loaded)
    filler: bool = True  # backdrop terrain becomes ground under the whole new stage
    filler_below: float = 1.5  # m: filler sits this far below the stage terrain
    own_level: str | None = None  # write the level under this name (/Game/Levels/<name>), not over the original
    display_name: str | None = None  # menu name of an own-level stage
    build_params: dict | None = None  # how it was built (stored with it: regenerate / load settings)
    join_shears: list | None = None  # per piece: join smoothing shears in the layout frame (from the search)
    # barrier tapes hang between two poles; poles of different pieces (or a
    # parked one) stretch them across the map. "hide": park every tape pole
    # (the tapes go with them); "move": move poles like props (stretches)
    tapes: str = "hide"
    debug_dup_cells: tuple = ()  # test builds (own level): duplicate these cells (source IDs), unmoved
    debug_dup_grid: bool = True  # test builds: register the duplicates in the streaming grid
    debug_dup_same_pkg: bool = False  # test builds: duplicates stream the source cell package (no copy)
    debug_dup_new_guid: bool = True  # test builds: duplicates get their own cell GUID
    giant_mesh_m: float = 400.0  # m: meshes wider than this are scenery, never moved as props
    bg_clear: float = 2000.0  # m: background (satellite map) mesh hidden this close to the new route
    # foliage, each "move" (follows the piece owning its spot) or "hide" (left out, faster)
    forest: str = "move"  # PCG forest trees (PCGPartitionActor HISMs)
    grass: str = "move"  # PCG grass, branches, rocks (PCGGrassContainer)
    foliage_tool: str = "move"  # foliage-tool grass (InstancedFoliageActor)
    hide_unused_tiles: bool = True  # tiles the layout does not use would float at the old location
    seam_width: float = 60.0  # m: terrain height blending either side of piece boundaries (0 = off)
    debug_skip_kinds: tuple = ()  # test builds: actor kinds left unmoved
    debug_no_backdrop: bool = False  # test builds: hide the coarse backdrop terrain entirely
    debug_no_clip: bool = False  # test builds: keep tile/backdrop meshes unclipped
    debug_offset_zero: bool = False  # test builds: build on top of the original map (piece 0 stays put)
    debug_offset: tuple | None = None  # test builds: explicit placement offset (m)


@dataclass
class WPReport:
    out_files: list = field(default_factory=list)
    moved: collections.Counter = field(default_factory=collections.Counter)
    relocated: list = field(default_factory=list)
    clipped: list = field(default_factory=list)
    baked_meshes: int = 0
    baked_tris: int = 0
    notes: dict = field(default_factory=dict)
    offset: tuple = (0, 0, 0)
    packages: int = 0
    checks: list = field(default_factory=list)


@dataclass
class PieceX:
    sa: float
    sb: float
    R: np.ndarray
    t: np.ndarray
    yaw: float
    tiles: frozenset
    layer: int = 0  # 0: the original cells; k: the k-th copy of its cells (repeated tiles)
    shears: list = field(default_factory=list)  # join smoothing (stagegen/joins.py), world frame


class _Assign:
    """Source position -> piece (of one layer: within a layer every source
    stretch belongs to at most one piece)."""

    def __init__(self, src_pts: np.ndarray, source_station: np.ndarray, pieces: list[PieceX], ext: float):
        self.tree = cKDTree(src_pts[:, :2])  # XY is intentional for lateral ownership/clipping.
        self.station = np.asarray(source_station, dtype=np.float64)
        if self.station.shape != (len(src_pts),):
            raise ValueError("source station array does not match source points")
        self.p = pieces
        self.ext = ext

    def __call__(self, xy: np.ndarray, ext: float | None = None, layer: int = 0):
        d, idx = self.tree.query(np.atleast_2d(xy)[:, :2])
        s = self.station[idx]
        k = np.full(len(s), -1)
        e = self.ext if ext is None else ext
        for j, pc in enumerate(self.p):  # exact ranges first
            if pc.layer == layer:
                k[(k < 0) & (s >= pc.sa) & (s <= pc.sb)] = j
        for j, pc in enumerate(self.p):
            if pc.layer == layer:
                k[(k < 0) & (s >= pc.sa - e) & (s <= pc.sb + e)] = j
        return k, d, s, idx


FOLIAGE_OWNERS = ("PCGPartitionActor", "PCGGrassContainer", "InstancedFoliageActor", "PCGStamp")
# instanced trees in the main level whose root is the world origin (stored as
# default, so it isn't decoded): their instance positions are world positions
ROOTLESS_TREES = ("HLODTrees", "BP_TreelinesContainer")
# report label per kind of tree container moved like the forest
TREE_LABELS = {
    "HLODTrees": "distant (HLOD) trees",
    "BP_TreelinesContainer": "treelines (distant ridges)",
    "PCGStamp": "stamped trees",
}


def _foliage_mode(cfg, owner: str, name: str):
    """'move' / 'hide' for an instanced foliage component, None for anything else."""
    if owner.startswith("PCGPartitionActor"):
        return cfg.forest if name.startswith(("HISM_", "ISM_")) else None
    if owner.startswith(ROOTLESS_TREES + ("PCGStamp",)):
        # more trees: the forest's distant HLOD stand-ins (one per real tree),
        # treelines on distant ridges, stamped groups: they follow the forest,
        # else they float where the terrain used to be
        return cfg.forest
    if owner.startswith("PCGGrassContainer"):
        return cfg.grass
    if owner.startswith("InstancedFoliageActor"):
        return cfg.foliage_tool
    return None


def _filler_height_field(
    surf,
    pieces,
    tree_new,
    tree_new_all,
    seam_dz,
    new_cl,
    mesh_list,
    bd_re_bytes,
    bd_piece,
    bd_keep,
    bd_xform,
    below,
    cell=20.0,
    pad=2500.0,
    erode_m=45.0,
):
    """Height (m) of the filler ground at new-layout xy: the stage's own terrain
    (owned samples, seam-corrected) lowered by `below`; the start/finish areas
    without track tiles at the exact (moved) backdrop height; elsewhere the
    nearest heights, smoothed, so the gaps between stretches get plausible hills."""
    from scipy.interpolate import RegularGridInterpolator
    from scipy.ndimage import distance_transform_edt, gaussian_filter

    pts, zs = [], []
    for j, (tr, z) in surf.items():
        P = np.c_[tr.data, z]
        d_j, _ = tree_new[j].query(P[:, :2])
        d_a, _ = tree_new_all.query(P[:, :2])
        P = P[d_j <= d_a + 2.0]
        P[:, 2] += seam_dz(j, P) - below
        pts.append(P[:, :2])
        zs.append(P[:, 2])
    if bd_piece is not None and bd_xform is not None:
        for U, sub in mesh_list:
            if re.search(bd_re_bytes, sub):
                k = bd_keep(U)
                if k.any():
                    V = bd_xform(U[k])
                    pts.append(V[:, :2])
                    zs.append(V[:, 2])
    P = np.vstack(pts)
    Z = np.concatenate(zs)
    # the road is a hard ceiling (below+1 m under it), spread over erode_m so
    # coarse filler triangles cannot span a road cut from hillside to hillside
    cl = resample(new_cl, 2.0)
    Rp, Rz = cl[:, :2], cl[:, 2] - below - 1.0
    lo = new_cl[:, :2].min(0) - pad
    hi = new_cl[:, :2].max(0) + pad
    gx = np.arange(lo[0], hi[0] + cell, cell)
    gy = np.arange(lo[1], hi[1] + cell, cell)
    ix = np.clip(((P[:, 0] - lo[0]) / cell).round().astype(int), 0, len(gx) - 1)
    iy = np.clip(((P[:, 1] - lo[1]) / cell).round().astype(int), 0, len(gy) - 1)
    # lowest sample per cell (the filler must stay under the detailed terrain)
    G = np.full((len(gx), len(gy)), np.inf)
    np.minimum.at(G, (ix, iy), Z)
    valid = np.isfinite(G)
    _, (nx, ny) = distance_transform_edt(~valid, return_indices=True)
    F = G[nx, ny]
    from scipy.ndimage import minimum_filter

    # the filler mesh is coarse (adaptive, up to ~100 m triangles): keep it under
    # the lowest stage height within erode_m so it does not form shelves in valleys
    E = minimum_filter(F, size=int(2 * erode_m / cell) + 1)
    # (continuing the edge heights outside the detailed terrain was tried: it
    # forms overhanging shelves over valleys, worse than the occasional gap)
    Ht = np.minimum(gaussian_filter(E, sigma=3), E)
    GR = np.full_like(G, np.inf)
    rx = np.clip(((Rp[:, 0] - lo[0]) / cell).round().astype(int), 0, len(gx) - 1)
    ry = np.clip(((Rp[:, 1] - lo[1]) / cell).round().astype(int), 0, len(gy) - 1)
    np.minimum.at(GR, (rx, ry), Rz)
    GR = minimum_filter(GR, size=int(2 * erode_m / cell) + 1)  # +inf away from the road
    H = np.minimum(Ht, GR)
    f = RegularGridInterpolator((gx, gy), H, bounds_error=False, fill_value=None)
    return lambda xy: f(np.atleast_2d(xy)[:, :2])


def _component_transforms_for_mesh(pk: Package, mesh_pkg_path: str) -> list:
    """Every placement of the mesh package mesh_pkg_path by a StaticMeshComponent
    in pk (via imported package names -> import map): [(location cm, yaw deg,
    scale)]. StaticMeshComponent props here: 4 StaticMesh (i32), 42
    OverrideMaterials (i32 count + i32 each), 163 location, 164 rotation
    (pitch, yaw, roll), 165 scale (3 doubles each)."""
    from ..extract.unversioned import Reader
    from ..extract.zen import read_name_batch

    game = "/Game/" + mesh_pkg_path.split("/Content/", 1)[1].rsplit(".", 1)[0]
    names, _ = read_name_batch(pk.d, pk.imported_pkg_names_off)
    if game not in names:
        return []
    k = names.index(game)
    refs = {-(i + 1) for i, v in enumerate(pk.imports) if (v >> 62) == 2 and ((v >> 32) & 0x3FFFFFFF) == k}
    out = []
    for e in pk.exports:
        b = pk.export_bytes(e)
        r = Reader(pk.d, e.data_offset)
        try:
            pres, _ = r.header_full()
        except Exception:
            continue
        q = r.p - e.data_offset
        if not pres or pres[0] != 4 or q + 4 > len(b) or struct.unpack_from("<i", b, q)[0] not in refs:
            continue
        q += 4
        if 42 in pres:
            q += 4 + 4 * struct.unpack_from("<i", b, q)[0]
        loc = np.array(struct.unpack_from("<3d", b, q)) if 163 in pres else np.zeros(3)
        q += 24 if 163 in pres else 0
        yaw = struct.unpack_from("<3d", b, q)[1] if 164 in pres else 0.0
        q += 24 if 164 in pres else 0
        scale = np.array(struct.unpack_from("<3d", b, q)) if 165 in pres else np.ones(3)
        out.append((loc, float(yaw), scale))
    return out


def _occupancy(xy: np.ndarray, tris: np.ndarray, cell: float = 20.0):
    """Raster (origin, cell, bool grid) of where a mesh has triangles (xy in m)."""
    tris = tris[(tris[:, 0] != tris[:, 1]) | (tris[:, 1] != tris[:, 2])]
    lo = xy.min(0) - cell
    shape = np.ceil((xy.max(0) - lo) / cell).astype(int) + 2
    occ = np.zeros(shape, bool)
    gx = lo[0] + (np.arange(shape[0]) + 0.5) * cell
    gy = lo[1] + (np.arange(shape[1]) + 0.5) * cell
    for t in tris:
        a, b, c = xy[t[0]], xy[t[1]], xy[t[2]]
        mn, mx = np.minimum(np.minimum(a, b), c), np.maximum(np.maximum(a, b), c)
        i0, i1 = np.searchsorted(gx, mn[0] - cell), np.searchsorted(gx, mx[0] + cell)
        j0, j1 = np.searchsorted(gy, mn[1] - cell), np.searchsorted(gy, mx[1] + cell)
        if i1 <= i0 or j1 <= j0:
            continue
        X, Y = np.meshgrid(gx[i0:i1], gy[j0:j1], indexing="ij")
        v0, v1 = c - a, b - a
        den = v0[0] * v1[1] - v1[0] * v0[1]
        if den == 0:
            continue
        px, py = X - a[0], Y - a[1]
        u = (px * v1[1] - v1[0] * py) / den
        v = (v0[0] * py - px * v0[1]) / den
        # a cell counts if its centre is in the triangle (with half a cell of slack)
        slack = 0.5 * cell / max(np.linalg.norm(v0), np.linalg.norm(v1), 1e-6)
        occ[i0:i1, j0:j1] |= (u >= -slack) & (v >= -slack) & (u + v <= 1 + slack)
    return lo, cell, occ


def _occ_hit(raster, pts: np.ndarray) -> np.ndarray:
    lo, cell, occ = raster
    ij = np.floor((pts - lo) / cell).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < occ.shape[0]) & (ij[:, 1] < occ.shape[1])
    out = np.zeros(len(pts), bool)
    out[ok] = occ[ij[ok, 0], ij[ok, 1]]
    return out


def pkg_name_of_path_exists(fs, pkg_name: str) -> bool:
    """Is there a package with exactly this name (e.g. an un-numbered base)?"""
    rel = pkg_name[len("/Game/") :] if pkg_name.startswith("/Game/") else None
    return rel is not None and any(f"acr/Content/{rel}{ext}" in fs.index for ext in (".uasset", ".umap"))


_MESH_EXTENT: dict = {}


_MESH_VERTS: dict = {}


def _mesh_lod0_m(fs, mesh_path: str):
    """LOD0 vertex positions (m, mesh-local) of a mesh package, cached."""
    if mesh_path not in _MESH_VERTS:
        P = None
        ap = "acr/Content/" + mesh_path[len("/Game/") :] + ".uasset"
        if ap in fs.index:
            try:
                mp = Package(fs.read(ap))
                b = mp.export_bytes(mp.exports[mesh_export_index(mp)])
                l0 = find_lods(b)[0]
                P = np.frombuffer(b, "<f4", 3 * l0.n, l0.pos_off).reshape(-1, 3).astype(np.float64) / 100
            except Exception:
                P = None
        _MESH_VERTS[mesh_path] = P
    return _MESH_VERTS[mesh_path]


def _giant_mesh_components(fs, pk: Package, limit_m: float) -> dict:
    """{export: mesh path} of pk's static mesh components drawing a mesh wider
    than limit_m (landscape pieces: service park terrain, mountains, towns).
    They are scenery unless the route runs over them (see build_wp_stage)."""
    from ..extract.unversioned import Reader
    from ..extract.zen import read_name_batch

    names, _ = read_name_batch(pk.d, pk.imported_pkg_names_off)
    giant_k = set()
    for k, n in enumerate(names):
        leaf = n.rsplit("/", 1)[-1]
        if not leaf.startswith("SM_"):
            continue
        if n not in _MESH_EXTENT:
            ext = 0.0
            ap = "acr/Content/" + n[len("/Game/") :] + ".uasset"
            if ap in fs.index:
                try:
                    mp = Package(fs.read(ap))
                    wb = mesh_world_bounds(mp.export_bytes(mp.exports[mesh_export_index(mp)]))
                    if wb is not None:
                        ext = float(((wb[1] - wb[0]) / 100)[:2].max())
                except Exception:
                    pass
            _MESH_EXTENT[n] = ext
        if _MESH_EXTENT[n] > limit_m:
            giant_k.add(k)
    if not giant_k:
        return {}
    ref_name = {
        -(i + 1): names[(v >> 32) & 0x3FFFFFFF]
        for i, v in enumerate(pk.imports)
        if (v >> 62) == 2 and ((v >> 32) & 0x3FFFFFFF) in giant_k
    }
    out = {}
    for i, e in enumerate(pk.exports):
        r = Reader(pk.d, e.data_offset)
        try:
            pres, _ = r.header_full()
        except Exception:
            continue
        q = r.p - e.data_offset
        if pres and pres[0] == 4 and q + 4 <= e.serial_size:
            ref = struct.unpack_from("<i", pk.d, e.data_offset + q)[0]
            if ref in ref_name:
                out[i] = ref_name[ref]
    return out


def _background_meshes(fs, main: Package) -> list[str]:
    """Background mesh packages placed by the main level: imported static meshes
    with BG in the name (SM_S4HafrensSouthBG2, SM_ElatiaExtraBG04, SM_BGSet7_new)."""
    import re as _re
    from ..extract.zen import read_name_batch

    names, _ = read_name_batch(main.d, main.imported_pkg_names_off)
    out = []
    for n in names:
        leaf = n.rsplit("/", 1)[-1]
        if n.startswith("/Game/Environments/") and _re.match(r"SM_.*BG", leaf, _re.I):  # BG, Bg
            p = "acr/Content/" + n[len("/Game/") :] + ".uasset"
            if p in fs.index:
                out.append(p)
    return out


def build_wp_stage(
    layout: Layout,
    best: TState,
    cuts: list[Cut],
    grid: TileGrid,
    ws: WPSource,
    paks_dir: str,
    out_dir: str,
    cfg: WPConfig | None = None,
) -> WPReport:
    cfg = cfg or WPConfig()
    rep = WPReport()
    fs = base_game_fs(paks_dir)
    src = ws.stage
    source_station = src.station
    L_src = float(source_station[-1])
    lvl = src.name

    # ---- placement of the new stage
    xs = [grid.x0 + grid.cell * k[1] for k in grid.tiles]
    ys = [grid.y0 + grid.cell * k[0] for k in grid.tiles]
    old_max_x, old_min_y = max(xs) + grid.cell, min(ys)
    cl = layout.centreline
    offset = np.array([old_max_x + cfg.margin - cl[:, 0].min(), old_min_y - cl[:, 1].min(), 0.0])
    if cfg.debug_offset_zero:
        offset = np.zeros(3)
    if cfg.debug_offset is not None:
        offset = np.asarray(cfg.debug_offset, float)
    rep.offset = tuple(np.round(offset, 1))
    states = list(reversed(list(best.chain())))
    js = cfg.join_shears or [[] for _ in states]
    pieces = [
        PieceX(
            cuts[s.rng.a].s,
            cuts[s.rng.b].s,
            rot_z(s.piece.yaw),
            np.asarray(s.piece.translation) + offset,
            float(s.piece.yaw),
            s.rng.tiles,
            getattr(s, "layer", 0),
            [sh.shifted(offset) for sh in js[k]],
        )
        for k, s in enumerate(states)
    ]

    def warp(j, P):
        """Join smoothing of piece j applied to world points (N,2|3), metres."""
        P = np.atleast_2d(np.asarray(P, np.float64))
        for sh in pieces[j].shears:
            P = sh.apply(P)
        return P

    def warp_yaw(j, xy, heading):
        """Yaw delta (rad) of a specified direction through the full shear map.

        Each shear is differentiated at the point produced by all previous
        shears, so the result is the direction of the composed transform.
        """
        P = np.atleast_2d(np.asarray(xy, dtype=np.float64)).copy()
        h = np.broadcast_to(np.asarray(heading, dtype=np.float64), (len(P),)).copy()
        h0 = h.copy()
        for sh in pieces[j].shears:
            h += sh.yaw(P, h)
            P = sh.apply(P)
        return np.arctan2(np.sin(h - h0), np.cos(h - h0))

    def to_new(j, P_src):
        """Source points (N,3) -> new world: piece j's rigid transform + join smoothing."""
        pc = pieces[j]
        return warp(j, np.atleast_2d(P_src) @ pc.R.T + pc.t)

    n_layers = 1 + max(pc.layer for pc in pieces)
    if n_layers > 1 and not cfg.own_level:
        raise ValueError("repeated tiles need an own level (copies of cells); not available with --override")
    assign = _Assign(src.points, source_station, pieces, cfg.range_ext)
    tile_pieces = {(pc.layer, t): j for j, pc in enumerate(pieces) for t in pc.tiles}  # (layer, tile) -> piece
    tile_piece = {t: j for (lay, t), j in tile_pieces.items() if lay == 0}
    new_cl = cl + offset
    new_s = arclength(new_cl)
    new_head = headings(new_cl, new_s)
    src_head = headings(src.points, source_station)

    def lat_ok(s, d, base):
        """Lateral limit; widened at the route ends, where start/finish areas sit off the spline."""
        end = (np.asarray(s) <= 50) | (np.asarray(s) >= L_src - 50)
        return np.asarray(d) <= np.where(end, max(base, cfg.end_radius), base)

    # Voronoi ownership on the *new* layout: after moving, a triangle/prop of
    # piece j is kept only if j's road is (about) the nearest road to it, so
    # no piece reaches over another piece's road.
    piece_new = []
    for j, pc in enumerate(pieces):
        i0 = max(0, int(np.searchsorted(source_station, pc.sa, side="left")) - 1)
        i1 = min(len(src.points) - 1, int(np.searchsorted(source_station, pc.sb, side="right")))
        piece_new.append(to_new(j, src.points[i0 : i1 + 1])[:, :2])
    tree_new_all = cKDTree(np.vstack(piece_new))
    tree_new = [cKDTree(p) for p in piece_new]
    piece_box = [(p.min(0), p.max(0)) for p in piece_new]  # XY bounding box of each piece's road

    def owns(j, src_pts_m, margin: float = 2.0):
        q = np.atleast_2d(src_pts_m)
        q = to_new(j, q if q.shape[1] == 3 else np.c_[q[:, :2], np.zeros(len(q))])[:, :2]
        d_own, _ = tree_new[j].query(q)
        d_all, _ = tree_new_all.query(q)
        return d_own <= d_all + margin

    def rigid_raw(j, p_m, rot):
        """Piece j's rigid transform only (tile and backdrop actors: their meshes
        carry the join smoothing in their vertices)."""
        pc = pieces[j]
        r = (rot if rot is not None else np.zeros(3)) + np.array([0, np.degrees(pc.yaw), 0])
        r[1] = (r[1] + 180.0) % 360.0 - 180.0  # piece yaws add up (repeats, reversals)
        return pc.R @ p_m + pc.t, r

    def rigid(j, p_m, rot):
        """Rigid transform + join smoothing (props, start/finish actors)."""
        loc, r = rigid_raw(j, p_m, rot)
        if pieces[j].shears:
            loc_raw = loc.copy()
            delta_yaw = warp_yaw(j, loc_raw[None], np.radians(r[1]))[0]
            loc = warp(j, loc_raw[None])[0]
            r[1] = (r[1] + np.degrees(delta_yaw) + 180.0) % 360.0 - 180.0
        return loc, r

    def along(p_m, rot, idx):
        """Keep the relative position along the route (fraction of length), facing the same way."""
        s_new = source_station[idx] / L_src * new_s[-1]
        j = min(int(np.searchsorted(new_s, s_new)), len(new_cl) - 1)
        dh = new_head[j] - src_head[idx]
        off = (p_m[:2] - src.points[idx, :2]) @ rot_z(dh)[:2, :2].T
        loc = np.r_[new_cl[j, :2] + off, new_cl[j, 2] + (p_m[2] - src.points[idx, 2])]
        r = (rot if rot is not None else np.zeros(3)) + np.array([0, np.degrees(dh), 0])
        r[1] = (r[1] + 180.0) % 360.0 - 180.0
        return loc, r, s_new

    # ---- route stretches no track tile covers (start areas etc.) live on the
    # coarse backdrop terrain (SM_<Level>TerrainYYXX); it follows the piece
    # that owns most of them, clipped to those stretches.
    bpk = Package(fs.read(ws.baked))
    bf = W.read_baked_full(bpk)
    meshes = bf.meshes
    tile_v = np.vstack(
        [m.vertices[::4, [0, 2, 1]] for m in meshes if re.search(rb"Track\d{4}", m.soft_sub) and len(m.vertices)]
    )
    dcov, _ = cKDTree(tile_v[:, :2]).query(src.points[:, :2])
    uncovered = dcov > 15.0
    # Dilate uncovered samples by a physical +/-40 m station window.
    lo_idx = np.searchsorted(source_station, source_station - 40.0, side="left")
    hi_idx = np.searchsorted(source_station, source_station + 40.0, side="right")
    prefix = np.r_[0, np.cumsum(uncovered.astype(np.int64))]
    unc_ext = (prefix[hi_idx] - prefix[lo_idx]) > 0
    s_all = source_station
    # Approximate each sample's represented station cell for a length estimate.
    sample_span = np.empty(len(source_station), dtype=np.float64)
    sample_span[0] = (source_station[1] - source_station[0]) / 2
    sample_span[-1] = (source_station[-1] - source_station[-2]) / 2
    if len(source_station) > 2:
        sample_span[1:-1] = (source_station[2:] - source_station[:-2]) / 2
    # (the backdrop lives in the main level, which exists once: layer-0 pieces only)
    unc_len = [
        float(sample_span[uncovered & (s_all >= pc.sa) & (s_all <= pc.sb)].sum()) if pc.layer == 0 else 0.0
        for pc in pieces
    ]
    bd_piece = int(np.argmax(unc_len)) if max(unc_len) > 20 else None
    if bd_piece is not None:
        rep.checks.append(
            f"backdrop terrain follows piece {bd_piece} ({unc_len[bd_piece]:.0f} m without track tiles)"
            + "".join(
                f"; piece {j}: {u:.0f} m uncovered (no ground)"
                for j, u in enumerate(unc_len)
                if j != bd_piece and u > 20
            )
        )
    # ---- seam blending: neighbouring pieces' terrain meets at different heights
    # beside the road (one source spot is a hillside, the other a valley). Near
    # each ownership boundary, vertices are pulled towards the mean of both
    # surfaces (weight 1 at the boundary, 0 at seam_width into the own piece).
    s_tree_src = cKDTree(src.points[:, :2])
    surf = {}
    for j, pc in enumerate(pieces):
        pts = []
        for m in meshes:
            mt = re.search(rb"Track(\d\d)(\d\d)", m.soft_sub)
            if not (mt and len(m.vertices)) or (int(mt.group(1)), int(mt.group(2))) not in pc.tiles:
                continue
            U = m.vertices[:, [0, 2, 1]].astype(np.float64)
            _, ix = s_tree_src.query(U[:, :2])
            s_ = source_station[ix]
            U = U[(s_ >= pc.sa - 150) & (s_ <= pc.sb + 150)]
            pts.append(to_new(j, U))
        if pts:
            P = np.vstack(pts)
            surf[j] = (cKDTree(P[:, :2]), P[:, 2])

    # continuous height of each piece's terrain: mean height per 4 m cell, gaps
    # filled from the nearest cell, bilinear in between; plus the (bilinear)
    # distance to real data for the confidence. (IDW over the k nearest
    # vertices jumped when a neighbour swapped: up to 0.7 m steps on the road.)
    from scipy.interpolate import RegularGridInterpolator
    from scipy.ndimage import distance_transform_edt

    surf_grid = {}
    GC = 4.0
    for j, (tr, z) in surf.items():
        XY = tr.data
        lo_ = XY.min(0) - 48.0
        nx, ny = np.ceil((XY.max(0) + 48.0 - lo_) / GC).astype(int) + 1
        ix = np.clip(((XY[:, 0] - lo_[0]) / GC).round().astype(int), 0, nx - 1)
        iy = np.clip(((XY[:, 1] - lo_[1]) / GC).round().astype(int), 0, ny - 1)
        S_ = np.zeros((nx, ny))
        N_ = np.zeros((nx, ny))
        np.add.at(S_, (ix, iy), z)
        np.add.at(N_, (ix, iy), 1.0)
        valid = N_ > 0
        dist, (jx, jy) = distance_transform_edt(~valid, return_indices=True)
        Zg = np.where(valid, S_ / np.maximum(N_, 1.0), 0.0)[jx, jy]
        gx = lo_[0] + GC * np.arange(nx)
        gy = lo_[1] + GC * np.arange(ny)
        surf_grid[j] = (
            RegularGridInterpolator((gx, gy), Zg, bounds_error=False, fill_value=np.nan),
            RegularGridInterpolator((gx, gy), dist * GC, bounds_error=False, fill_value=1e9),
        )

    def surface_z(j, xy):
        """(height, confidence) of piece j's terrain at xy, continuous; confidence
        fades from 1 (data within 12 m) to 0 (none within 40 m)."""
        if j not in surf_grid:
            return np.full(len(xy), np.nan), np.zeros(len(xy))
        hz, hd = surf_grid[j]
        q = np.atleast_2d(xy)[:, :2]
        h = hz(q)
        conf = np.clip((40.0 - hd(q)) / 28.0, 0.0, 1.0)
        return np.where(conf > 0, h, np.nan), conf

    def seam_dz(j, P_new_m):
        """Vertical correction (m) for points of piece j at new-layout positions:
        blended towards every piece whose road is about as near (Voronoi edge).

        Limiting this to the tile edge two consecutive pieces share (tried in
        8f750d9) left terrain of a neighbouring tile 9 m above the road 90-160 m
        before a join (Hafren South, pieces 0-5 9-11 11-14 15-19): undrivable.
        Blending by road proximity pulls that terrain down, as before."""
        P = np.atleast_2d(P_new_m)
        dz = np.zeros(len(P))
        if cfg.seam_width <= 0:
            return dz
        d_j, _ = tree_new[j].query(P[:, :2])
        # w > 0 needs d_b < d_j + 2 seam_width: pieces whose road is farther from
        # every point contribute nothing (skipped), and queries stop at that bound
        # (an infinite distance gives w = 0, as the exact one would)
        reach = float(d_j.max()) + 2.0 * cfg.seam_width
        lo_p, hi_p = P[:, :2].min(0), P[:, :2].max(0)
        for b in range(len(pieces)):
            if b == j:
                continue
            lo_b, hi_b = piece_box[b]
            if np.any(lo_b - reach > hi_p) or np.any(hi_b + reach < lo_p):
                continue
            d_b, _ = tree_new[b].query(P[:, :2], distance_upper_bound=reach)
            w = np.clip(1.0 - (d_b - d_j) / 2.0 / cfg.seam_width, 0.0, 1.0)
            near = w > 0
            if not near.any():
                continue
            w = w * w * (3 - 2 * w)
            zb, conf = surface_z(b, P[near, :2])
            ok = np.isfinite(zb)
            idx = np.flatnonzero(near)[ok]
            dz[idx] += w[idx] * conf[ok] * (zb[ok] - P[idx, 2]) / 2.0
        return dz

    bd_re = re.compile(rf"SM_{lvl}Terrain\d{{4}}$")
    bd_pivot: dict[str, np.ndarray] = {}

    # ---- packages
    pk_paths = [ws.main] + ws.cells
    pkgs = {p: Package(fs.read(p)) for p in pk_paths}
    rules = learn_rules(pkgs.values())
    tile_re = re.compile(rf"SM_{lvl}Track(\d\d)(\d\d)_GEN")
    # filler ground: the coarse backdrop terrain tiles become ground under the
    # whole new stage (reshaped below to its height field). Tiles that carry a
    # route stretch without track tiles (start/finish areas, roads attached to
    # the backdrop) stay rigid with that piece; every other tile is laid out
    # afresh in a grid under the new route (nearest cells first); spare tiles
    # are hidden. Without full coverage, clipped stage terrain floats in the air.
    # footprints from the visual meshes (most backdrop tiles have no collision):
    # LOD0 vertices + the tile actor's location, in source metres
    tile_src: dict = {}
    tile_occ: dict = {}  # name -> occupancy raster (the meshes have holes)
    main_pk = pkgs[ws.main]
    for c in decode_components(main_pk, rules):
        own = main_pk.exports[c.owner].name if c.owner < len(main_pk.exports) else ""
        if not (c.root and c.loc is not None and bd_re.match(own)):
            continue
        mp = next((x for x in fs.index if x.endswith(f"/{own}.uasset") and "/MeshesTerrain/" in x), None)
        if mp is None:
            continue
        mpk_ = Package(fs.read(mp))
        mb_ = mpk_.export_bytes(mpk_.exports[mesh_export_index(mpk_)])
        lods_ = find_lods(mb_)
        if not lods_:
            continue
        P_ = np.frombuffer(mb_, "<f4", 3 * lods_[0].n, lods_[0].pos_off).reshape(-1, 3).astype(np.float64)
        tile_src[own] = (P_ + c.loc) / 100
        tris_ = [
            np.frombuffer(mb_, "<u4" if w_ == 4 else "<u2", cnt_ // w_, off_).reshape(-1, 3)
            for off_, cnt_, w_ in lods_[0].index_bufs
        ]
        tile_occ[own] = _occupancy(tile_src[own][:, :2], np.vstack(tris_) if tris_ else np.empty((0, 3), int))
    filler_plan: dict = {}  # name -> None (rigid with bd_piece) | shift (2,) | "hide"
    if cfg.filler and tile_src:
        rigid_keep = set()
        if bd_piece is not None and uncovered.any():
            ut = cKDTree(src.points[uncovered][:, :2])
            rigid_keep = {n for n, U in tile_src.items() if (ut.query(U[:, :2])[0] < 80.0).any()}
        # full-size tiles only: the map's edge tiles are narrow strips
        ext = {n: np.ptp(U[:, :2], axis=0) for n, U in tile_src.items()}
        size = float(max(e.max() for e in ext.values()))
        full = {n for n, e in ext.items() if e.min() > 0.8 * size}
        # greedy cover on the real tile shapes (occupancy rasters: the backdrop
        # has holes where the original track tiles were): sample points along
        # the route and 200 m either side; each step places the free tile and
        # position that covers the most still-uncovered points
        rq = resample(new_cl, 20.0)
        tq = np.gradient(rq[:, :2], axis=0)
        tq /= np.maximum(np.linalg.norm(tq, axis=1, keepdims=True), 1e-9)
        nq = np.c_[-tq[:, 1], tq[:, 0]]
        Q = np.vstack([rq[:, :2] + nq * off for off in (-200.0, 0.0, 200.0)])
        todo = np.ones(len(Q), bool)
        if rigid_keep:
            pc0 = pieces[bd_piece]
            for n in rigid_keep:  # source xy of the query points for that tile
                Qs = (Q - pc0.t[:2]) @ pc0.R[:2, :2]
                todo &= ~_occ_hit(tile_occ[n], Qs)
        placed: dict = {}
        free_t = sorted(n for n in tile_src if n not in rigid_keep and n in full)
        while todo.any():
            cand = Q[todo][:: max(1, int(todo.sum()) // 120)]
            best = (0, None, None)
            for n in free_t:
                if n in placed:
                    continue
                ctr_n = (tile_src[n][:, :2].min(0) + tile_src[n][:, :2].max(0)) / 2
                for cxy in cand:
                    sh = cxy - ctr_n
                    hit = todo & _occ_hit(tile_occ[n], Q - sh)
                    k_ = int(hit.sum())
                    if k_ > best[0]:
                        best = (k_, n, sh)
            if best[1] is None:
                break
            _, n, sh = best
            placed[n] = sh
            todo &= ~_occ_hit(tile_occ[n], Q - sh)
        cells = list(placed.values())
        free_t = list(placed) + [n for n in free_t if n not in placed]
        for n in tile_src:
            if n not in rigid_keep and n not in full:
                filler_plan[n] = "hide"
        for n in rigid_keep:
            filler_plan[n] = None
        for n, sh in placed.items():
            filler_plan[n] = sh
        for n in free_t[len(cells) :]:
            filler_plan[n] = "hide"
        rep.checks.append(
            f"filler ground: {len(tile_src)} backdrop tiles of ~{size:.0f} m: {len(rigid_keep)} stay with "
            f"piece {bd_piece}, {min(len(free_t), len(cells))} laid out under the route, "
            f"{max(0, len(free_t) - len(cells))} spare (hidden); {len(cells)} cells needed"
        )
    elif cfg.filler:
        cfg.filler = False
        rep.checks.append("filler ground: no backdrop tiles, off")

    payloads: dict[str, dict[int, bytes]] = collections.defaultdict(dict)
    tile_pivot: dict[tuple, np.ndarray] = {}

    def place_trees(W, source_yaw, layer=0):
        """Move foliage and compute its orientation from its actual forward axis."""
        k, d, s, _ = assign(W, layer=layer)
        keep = (k >= 0) & lat_ok(s, d, cfg.terrain_clip)
        NW = W.copy()
        yaw = np.zeros(len(W))
        for kk in np.unique(k[k >= 0]):
            m_ = k == kk
            pc = pieces[int(kk)]
            keep[m_] &= owns(int(kk), W[m_])
            raw_new = W[m_] @ pc.R.T + pc.t
            NW[m_] = warp(int(kk), raw_new) if pc.shears else raw_new
            NW[m_, 2] += seam_dz(int(kk), NW[m_])
            yaw[m_] = pc.yaw
            if pc.shears:
                yaw[m_] += warp_yaw(int(kk), raw_new, source_yaw[m_] + pc.yaw)
            if int(kk) in surf:  # only where the piece's terrain is
                dd, _ = surf[int(kk)][0].query(NW[m_, :2], distance_upper_bound=15.0)
                keep[np.flatnonzero(m_)] &= np.isfinite(dd)
        if keep.any():  # park hidden ones among the kept (tight bounds)
            NW[~keep] = NW[keep].mean(0)
            yaw[~keep] = 0.0
        return keep, NW, yaw

    forest_nobox = [0]
    PARK_M = 5000.0  # copies: actors no piece of their layer wants go this far underground

    def process(path, pk, layer, payl):
        """Move the actors of one package for the pieces of one layer (layer 0:
        the original package; k > 0: a copy of a cell for the k-th use of its
        tiles; there, anything unused is parked underground and gameplay-only
        actors stay in layer 0). Returns the number of moved actors/trees."""
        n_moved = 0
        touched: set = set()
        ex = pk.exports
        route = ws.route_cells.get(path)
        # foliage left out ("hide"): zero-scaled; unmoved it would float at the
        # original location in the void left by the moved tiles
        for i, e in enumerate(ex):
            own = ex[e.outer].name if 0 <= e.outer < len(ex) else ""
            mode = _foliage_mode(cfg, own, e.name)
            if mode == "hide":
                nb, nh = hide_instances(pk.export_bytes(e))
                if nh:
                    payl[i] = nb
                    if layer == 0:
                        rep.moved["foliage instances left out"] += nh
        comps = [c for c in decode_components(pk, rules) if c.root and c.loc is not None]
        pcg_roots: dict[int, np.ndarray] = {}
        pcg_root_comp: dict[int, int] = {}  # foliage owner actor -> its root component export
        giant = _giant_mesh_components(fs, pk, cfg.giant_mesh_m)

        def put(c, loc, rot, kind=None):
            nonlocal n_moved
            payl[c.export] = patched_payload(pk, c, loc * 100, rot)
            touched.add(c.export)
            n_moved += 1
            if kind:
                rep.moved[kind if layer == 0 else f"{kind} (copies)"] += 1

        for c in comps:
            own_ = ex[c.owner].name if c.owner < len(ex) else ""
            if cfg.tapes == "hide" and own_.startswith(("TapePoleActor", "BC_TapeHandleActor")):
                p_m = c.loc / 100
                payl[c.export] = patched_payload(
                    pk, c, (p_m - np.array([0, 0, PARK_M])) * 100, c.rot if c.rot is not None else np.zeros(3)
                )
                touched.add(c.export)
                if layer == 0:
                    rep.moved["tape poles hidden (with their tapes)"] += 1
                continue
            if layer and (c.export in giant or bd_re.match(own_) or c.rule.L == 26 or "PlayerStart" in own_):
                continue  # exists once (layer 0); parked below
            if c.export in giant and not (bd_re.match(own_) or tile_re.match(own_)):  # tiles: own handling
                # scenery stays, unless the route runs over it (e.g. the ground
                # of a start/finish area): then it moves like any prop
                P_ = _mesh_lod0_m(fs, giant[c.export])
                carries = False
                if P_ is not None:
                    S_ = c.scale if getattr(c, "scale", None) is not None else np.ones(3)
                    Wg = P_ * S_ + c.loc / 100  # rotation of such meshes is ~0
                    dg, ig = cKDTree(Wg[:, :2]).query(src.points[:, :2], distance_upper_bound=15.0)
                    ok_ = np.isfinite(dg)
                    carries = bool(ok_.any() and (np.abs(Wg[ig[ok_], 2] - src.points[ok_, 2]) < 3.0).sum() >= 5)
                if not carries:
                    rep.moved["giant scenery mesh left in place"] += 1
                    continue
                rep.checks.append(f"giant mesh {giant[c.export].rsplit('/', 1)[-1]} carries the route: moved")
            owner = ex[c.owner].name if c.owner < len(ex) else ""
            p_m = c.loc / 100
            if c.rule.L == 26:  # runtime virtual texture volume
                lo = np.r_[new_cl[:, :2].min(0) - 300, new_cl[:, 2].min() - 150]
                hi = np.r_[new_cl[:, :2].max(0) + 300, new_cl[:, 2].max() + 300]
                pl = patched_payload(pk, c, lo * 100, c.rot if c.rot is not None else np.zeros(3), (hi - lo) * 100)
                pl, dropped = _drop_streaming_mips(pl, pk.names)
                payl[c.export] = pl
                touched.add(c.export)
                rep.moved["RVT volume"] += 1
                rep.checks.append(
                    "RVT: "
                    + (
                        "baked streaming low mips disabled (rendered live from the moved geometry)"
                        if dropped
                        else "no streaming mips found"
                    )
                )
                continue
            m = tile_re.match(owner)
            if m:
                key = (int(m.group(1)), int(m.group(2)))
                tile_pivot[key] = c.loc.copy()
                if (layer, key) in tile_pieces:
                    loc, rot = rigid_raw(tile_pieces[(layer, key)], p_m, c.rot)
                    put(c, loc, rot, "track tile")
                continue
            if bd_re.match(owner):
                bd_pivot[owner] = c.loc.copy()
                plan = filler_plan.get(owner) if cfg.filler else None
                if isinstance(plan, np.ndarray):
                    loc = p_m + np.r_[plan, 0.0]
                    put(c, loc, c.rot if c.rot is not None else np.zeros(3), "filler ground")
                elif isinstance(plan, str):
                    pass  # spare tile: its mesh is hidden below
                elif bd_piece is not None:
                    loc, rot = rigid_raw(bd_piece, p_m, c.rot)
                    put(c, loc, rot, "backdrop terrain")
                continue
            if c.root and owner.startswith(FOLIAGE_OWNERS):
                pcg_roots[c.owner] = c.loc.copy()
                pcg_root_comp[c.owner] = c.export
            if owner.startswith(SKIP_OWNERS) or any(w in owner for w in SKIP_WORDS):
                continue
            kind = owner.split("_UAID")[0].rstrip("0123456789_")
            if kind in cfg.debug_skip_kinds:
                continue
            if "PlayerStart" in owner:
                # where the car is spawned (sits behind the start line, often
                # far from the route spline): always with the opening piece
                loc, rot = rigid(0, p_m, c.rot)
                if cfg.debug_spawn_s is not None:  # test builds: spawn anywhere on the route
                    j = min(int(np.searchsorted(new_s, cfg.debug_spawn_s)), len(new_cl) - 1)
                    loc = new_cl[j] + np.array([0, 0, 1.0])
                    rot = np.array([0.0, np.degrees(new_head[j]), 0.0])
                    rep.checks.append(f"DEBUG: car spawns at {cfg.debug_spawn_s:.0f} m on the new route")
                put(c, loc, rot, kind)
                continue
            k, d, s, idx = (v[0] for v in assign(p_m[None], layer=layer))
            if route == "cut":
                continue
            if route:
                # start/finish areas travel with the first/last piece; intermediate
                # splits are re-spaced along the new route so they stay in order
                if kind.startswith("RaceSector") and k not in (0, len(pieces) - 1):
                    k = -1
                if cfg.debug_spawn_s is not None and route == "fwd" and "StartTargetPoint" in kind:
                    j = min(int(np.searchsorted(new_s, cfg.debug_spawn_s)), len(new_cl) - 1)
                    put(c, new_cl[j] + np.array([0, 0, 0.5]), np.array([0.0, np.degrees(new_head[j]), 0.0]))
                    rep.checks.append(f"DEBUG: start target point at {cfg.debug_spawn_s:.0f} m")
                    continue
                if k >= 0 and d < 150:
                    loc, rot = rigid(k, p_m, c.rot)
                    put(c, loc, rot, kind)
                else:
                    loc, rot, s_new = along(p_m, c.rot, idx)
                    rep.relocated.append((f"{route}:{kind}", round(s), round(s_new)))
                    put(c, loc, rot)
                continue
            if k >= 0 and lat_ok(s, d, cfg.prop_lateral) and owns(k, p_m[None])[0]:
                loc, rot = rigid(k, p_m, c.rot)
                loc[2] += seam_dz(k, loc[None])[0]
                put(c, loc, rot, kind)

        # PCG forest (HISM trees on PCGPartitionActors, positions relative to the
        # actor root): each tree moves with the piece owning its spot, if that
        # piece's terrain is under it; the rest are hidden. The distant HLOD
        # trees (HLODTrees actors in the main level) are the same trees: root at
        # the origin (stored as default, so not decoded), world positions
        if True:
            for i, e in enumerate(ex):
                own_name = ex[e.outer].name if 0 <= e.outer < len(ex) else ""
                root = pcg_roots.get(e.outer)
                if root is None and own_name.startswith(ROOTLESS_TREES):
                    root = np.zeros(3)
                if root is None or _foliage_mode(cfg, own_name, e.name) != "move":
                    continue
                label = next((v for k, v in TREE_LABELS.items() if own_name.startswith(k)), None)
                try:
                    nb, st = move_hism(
                        pk.export_bytes(e),
                        root,
                        lambda W, source_yaw, layer=layer: place_trees(W, source_yaw, layer),
                    )
                except Exception as ex_:  # layout not understood: hide instead
                    nb, _ = hide_instances(pk.export_bytes(e))
                    rep.checks.append(f"forest {e.name}: hidden ({ex_})")
                    st = {"n": 0}
                if st.get("n"):
                    payl[i] = nb
                    n_moved += st["kept"]
                    if st["kept"] and e.outer in pcg_root_comp:
                        # instances are relative to the owner's root: in a copy
                        # it must stay where it is, not be parked with the unused
                        touched.add(pcg_root_comp[e.outer])
                    if label:
                        rep.moved[label] += st["kept"]
                        rep.moved[f"{label} hidden"] += st["n"] - st["kept"]
                    else:
                        rep.moved["forest trees" if layer == 0 else "forest trees (copies)"] += st["kept"]
                        if layer == 0:
                            rep.moved["forest trees hidden"] += st["n"] - st["kept"]
                    if st.get("boxes", 0) == 0:
                        forest_nobox[0] += 1
        if layer:
            # a copy: whatever its layer does not use is a duplicate of an
            # original actor; park it underground instead of leaving it twice
            for c in comps:
                if c.export not in touched and c.rule.L != 26:
                    p_m = c.loc / 100
                    payl[c.export] = patched_payload(
                        pk, c, (p_m - np.array([0, 0, PARK_M])) * 100, c.rot if c.rot is not None else np.zeros(3)
                    )
        return n_moved

    for k_pkg, (path, pk) in enumerate(pkgs.items()):
        span(0.12, 0.45, k_pkg, len(pkgs), "moving actors and foliage")
        process(path, pk, 0, payloads[path])
    # repeated tiles: every cell with something for a layer-k piece gets a copy
    # (not the main level, not the route/gameplay cells: those exist once)
    dup_payloads: dict[tuple, dict] = {}  # (cell path, layer) -> payloads
    dup_cells_ = [p for p in ws.cells if p not in ws.route_cells]
    for lay in range(1, n_layers):
        for k_pkg, path in enumerate(dup_cells_):
            span(
                0.45,
                0.55,
                (lay - 1) * len(dup_cells_) + k_pkg,
                (n_layers - 1) * len(dup_cells_),
                "copying cells for repeated tiles",
            )
            payl: dict = {}
            if process(path, pkgs[path], lay, payl):
                dup_payloads[(path, lay)] = payl
    if n_layers > 1:
        rep.checks.append(
            f"repeated tiles: {n_layers - 1} extra layer(s), {len(dup_payloads)} cell copies "
            f"({sum(1 for pc in pieces if pc.layer)} of {len(pieces)} pieces repeat tiles)"
        )

    if "move" in (cfg.forest, cfg.grass, cfg.foliage_tool):
        rep.checks.append(
            f"forest: {rep.moved['forest trees']} trees moved, {rep.moved['forest trees hidden']} "
            f"hidden; {forest_nobox[0]} components without cached bounds found"
        )

    # ---- route splines (per variant direction)
    ctrl = resample(new_cl, cfg.ctrl_spacing)
    w_ctrl = np.interp(arclength(ctrl), new_s, layout.widths)
    for path, kind in ws.route_cells.items():
        if kind == "cut":
            continue
        pk = pkgs[path]
        C, Wd = (ctrl, w_ctrl) if kind == "fwd" else (ctrl[::-1].copy(), w_ctrl[::-1].copy())
        t = np.gradient(C[:, :2], axis=0)
        t /= np.linalg.norm(t, axis=1, keepdims=True)
        n_left = np.c_[t[:, 1], -t[:, 0]]
        _, sc_P, _, _ = read_spline_points(pk, "CenterSpline")
        _, co_P, _, _ = read_spline_points(pk, "ContourSpline")
        dc, _ = cKDTree(sc_P[:, :2]).query(co_P[:, :2])
        margin = max(1.0, float(np.median(dc) / 100 - np.median(src.width) / 2))
        left = np.c_[C[:, :2] + n_left * (Wd / 2)[:, None], C[:, 2]]
        right = np.c_[C[:, :2] - n_left * (Wd / 2)[:, None], C[:, 2]]
        cleft = np.c_[C[:, :2] + n_left * (Wd / 2 + margin)[:, None], C[:, 2]]
        cright = np.c_[C[:, :2] - n_left * (Wd / 2 + margin)[:, None], C[:, 2]]
        pts = {
            "CenterSpline": C,
            "IdealSpline": C,
            "LeftBorderSpline": left,
            "RightBorderSpline": right,
            "ContourSpline": np.vstack([cleft, cright[::-1], cleft[:1]]),
        }
        for name in SPLINES:
            i = next((i for i, e in enumerate(pk.exports) if e.name == name), None)
            if i is not None:
                payloads[path][i] = W.spline_payload(pk, name, pts[name] * 100)

    # ---- keep every cell loaded
    main = pkgs[ws.main]
    hi_ = next(i for i, e in enumerate(main.exports) if e.name.startswith("WorldPartitionRuntimeSpatialHash"))
    e = main.exports[hi_]
    b = bytearray(payloads[ws.main].get(hi_, main.export_bytes(e)))
    # grid settings: i32 CellSize (cm) then f32 LoadingRange (cm). Their offset
    # varies (some levels store a grid origin vector in front), so find the
    # first plausible pair after the property header.
    k_cell = next(
        (
            k
            for k in range(2, min(len(b) - 8, 256))
            if 1000 <= struct.unpack_from("<i", b, k)[0] <= 10_000_000
            and struct.unpack_from("<i", b, k)[0] % 100 == 0
            and 1000.0 <= struct.unpack_from("<f", b, k + 4)[0] <= 10_000_000.0
        ),
        None,
    )
    if k_cell is None:
        raise ValueError("streaming grid settings not found in the spatial hash")
    cell_size, rng = struct.unpack_from("<if", b, k_cell)
    struct.pack_into("<f", b, k_cell + 4, cfg.loading_range_cm)
    payloads[ws.main][hi_] = bytes(b)
    if cfg.hlod_visuals:
        # track tiles do not render once moved; keep their cells unloaded so the
        # (moved) HLOD track meshes in the main level are what is drawn
        tile_cell_names = {
            p.rsplit("/", 1)[-1][:-5]
            for p, pk in pkgs.items()
            if p != ws.main and any(tile_re.match(e.name) for e in pk.exports)
        }
        refs = {i + 1 for i, e in enumerate(main.exports) if e.name in tile_cell_names}
        nb2, ndrop = drop_cells(payloads[ws.main][hi_], refs)
        payloads[ws.main][hi_] = nb2
        rep.checks.append(f"streaming: {ndrop} track-tile cells never load (HLOD visuals)")
    if cfg.all_cells_loaded:
        # experimental: register every cell in the top-level cell and widen its
        # record. NOTE: this hides the HLODs, which are the visible track in ACR.
        nb, hst = all_cells_in_top_level(bytes(b))
        payloads[ws.main][hi_] = nb
        for i, e in enumerate(main.exports):
            if e.name.startswith("WorldPartitionRuntimeCellDataSpatialHash"):
                nbp = widen_cell_data(main.export_bytes(e), hst["level"])
                if nbp is not None:
                    payloads[ws.main][i] = nbp
        rep.checks.append(f"streaming: all {hst['cells']} cells in grid level {hst['level']}, records widened")

    # ---- HLOD track meshes: the visible road/terrain in ACR. World-space
    # meshes (one per track tile) in the main level; move + clip them in place.
    moved_hlod = 0
    for i, e in enumerate(main.exports):
        if not e.name.startswith("StaticMesh_"):
            continue
        mb = main.export_bytes(e)
        wb = mesh_world_bounds(mb)
        if wb is None:
            continue
        lo, hi = wb
        if np.any((hi - lo)[:2] > grid.cell * 100 * 1.2):
            continue
        key = tuple(grid.of(((lo + hi) / 2 / 100)[None, :2])[0])
        if cfg.hlod_hide:
            # every cell stays loaded, so the full-detail geometry is always drawn;
            # the HLOD proxies (clipped to the corridor) only showed up as slabs
            # floating in the distance. Empty them (same size, all triangles degenerate).
            try:
                nbm, _ = clip_mesh_payload(mb, lambda cen: np.zeros(len(cen), bool))
            except ValueError:
                continue
            payloads[ws.main][i] = nbm
            moved_hlod += 1
            continue
        if key not in tile_piece:
            continue
        j = tile_piece[key]
        pc = pieces[j]

        def keep(cen_cm, pc=pc, j=j):
            d, idx = assign.tree.query(cen_cm[:, :2] / 100)
            s = source_station[idx]
            return (
                (s >= pc.sa - cfg.range_ext)
                & (s <= pc.sb + cfg.range_ext)
                & lat_ok(s, d, cfg.terrain_clip)
                & owns(j, cen_cm / 100)
            )

        nbm, hs = transform_mesh(mb, pc.R, pc.t * 100, None if cfg.debug_no_clip else keep)
        payloads[ws.main][i] = nbm
        moved_hlod += 1
    rep.moved["HLOD mesh " + ("hidden" if cfg.hlod_hide else "moved")] += moved_hlod
    rep.checks.append(
        f"streaming grid: cell {cell_size / 100:.0f} m, loading range {rng / 100:.0f} m -> "
        f"{cfg.loading_range_cm / 100:.0f} m"
    )

    new_pkgs: dict[str, bytes] = {p: rewrite_package(pkgs[p], pl) for p, pl in payloads.items() if pl}
    dup_pkgs: dict[tuple, bytes] = {key: rewrite_package(pkgs[key[0]], pl) for key, pl in dup_payloads.items()}

    # ---- clip the used tiles' meshes
    def keeper(pivot_cm, j, lim):
        pc = pieces[j]

        def keep(cen_cm):
            w = (cen_cm + pivot_cm) / 100
            d, idx = assign.tree.query(w[:, :2])
            s = source_station[idx]
            return (s >= pc.sa - cfg.range_ext) & (s <= pc.sb + cfg.range_ext) & lat_ok(s, d, lim) & owns(j, w)

        return keep

    gen_dir = f"/{lvl}/GeneratedSplits/"
    seam_max = 0.0
    # every (layer, tile) gets its own clipped meshes: layer 0 in place, layer k
    # as copies (layer_assets; the cell copies are pointed at them)
    layer_assets: dict[tuple, bytes] = {}  # (layer, mesh path) -> package bytes
    clip_items = [] if cfg.debug_no_clip else sorted(tile_pieces.items())
    for k_clip, ((lay, key), j) in enumerate(clip_items):
        span(0.55, 0.75, k_clip, len(clip_items), "clipping and blending terrain")
        if key not in tile_pivot:
            rep.checks.append(f"tile {key} has no actor")
            continue
        stem = f"SM_{lvl}Track{key[0]:02d}{key[1]:02d}_"
        for p in sorted(
            x for x in fs.index if gen_dir in x and x.rsplit("/", 1)[-1].startswith(stem) and x.endswith(".uasset")
        ):
            mpk = Package(fs.read(p))
            mi = mesh_export_index(mpk)
            lim = cfg.road_clip if "_Road_" in p else cfg.terrain_clip
            nb, st = clip_mesh_payload(mpk.export_bytes(mpk.exports[mi]), keeper(tile_pivot[key], j, lim))
            pc_, pv_ = pieces[j], tile_pivot[key]

            def local_fn(P, pc_=pc_, pv_=pv_, j=j):
                # mesh-local cm -> new world (rigid + join smoothing + seam height) -> mesh-local cm
                Wn = ((P + pv_) / 100) @ pc_.R.T + pc_.t
                if pc_.shears:
                    Wn = warp(j, Wn)
                Wn[:, 2] += seam_dz(j, Wn)
                return ((Wn - pc_.t) @ pc_.R) * 100 - pv_

            nb, mx = warp_payload(nb, local_fn)
            seam_max = max(seam_max, mx / 100)
            if lay == 0:
                new_pkgs[p] = rewrite_package(mpk, {mi: nb})
            else:
                layer_assets[(lay, p)] = rewrite_package(mpk, {mi: nb})
            rep.clipped.append(
                (p.rsplit("/", 1)[-1] + (f" (layer {lay})" if lay else ""), st["lods"], st["dropped"], st["tris"])
            )

    # ---- background meshes (satellite-map terrain around the original region,
    # world space): near the new stage they cut through or float above it
    if cfg.bg_clear > 0:
        cl_tree = cKDTree(new_cl[:, :2])
        n_bg = 0
        for p in _background_meshes(fs, main):
            places = _component_transforms_for_mesh(main, p)
            if not places:
                continue
            mpk = Package(fs.read(p))
            mi = mesh_export_index(mpk)
            mb = mpk.export_bytes(mpk.exports[mi])

            def far_everywhere(cen, places=places):
                keep = np.ones(len(cen), bool)
                for L, yaw, S in places:  # hidden if near the route in any placement
                    c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
                    P = cen * S
                    w = np.c_[c * P[:, 0] - s * P[:, 1], s * P[:, 0] + c * P[:, 1]] + L[:2]
                    keep &= cl_tree.query(w / 100)[0] > cfg.bg_clear
                return keep

            try:
                nb, st = clip_mesh_payload(mb, far_everywhere)
            except ValueError:
                continue
            if st["dropped"]:
                new_pkgs[p] = rewrite_package(mpk, {mi: nb})
                n_bg += 1
                rep.checks.append(
                    f"background {p.rsplit('/', 1)[-1]} ({len(places)} placed): "
                    f"{st['dropped']}/{st['tris']} triangles within {cfg.bg_clear:.0f} m hidden"
                )
        if not n_bg:
            rep.checks.append("background: nothing near the new route")

    if cfg.hide_unused_tiles and not cfg.debug_no_clip:
        n_un = 0
        for key in sorted(set(tile_pivot) - set(tile_piece)):
            stem = f"SM_{lvl}Track{key[0]:02d}{key[1]:02d}_"
            for p in sorted(
                x for x in fs.index if gen_dir in x and x.rsplit("/", 1)[-1].startswith(stem) and x.endswith(".uasset")
            ):
                mpk = Package(fs.read(p))
                mi = mesh_export_index(mpk)
                try:
                    nb, _ = clip_mesh_payload(mpk.export_bytes(mpk.exports[mi]), lambda c: np.zeros(len(c), bool))
                except ValueError:
                    continue
                new_pkgs[p] = rewrite_package(mpk, {mi: nb})
                n_un += 1
        rep.checks.append(f"unused tiles: {n_un} meshes hidden ({len(set(tile_pivot) - set(tile_piece))} tiles)")
    n_sh = sum(len(pc.shears) for pc in pieces) // 2
    rep.checks.append(
        f"seams: tile vertices moved up to {seam_max:.1f} m (height blending within "
        f"{cfg.seam_width:.0f} m of piece boundaries" + (f", {n_sh} join(s) smoothed)" if n_sh else ")")
    )

    def bd_keep_world(w_m):
        if cfg.debug_no_backdrop:
            return np.zeros(len(w_m), bool)
        pc = pieces[bd_piece]
        d, idx = assign.tree.query(w_m[:, :2])
        s = source_station[idx]
        return (
            unc_ext[idx] & (s >= pc.sa - 40) & (s <= pc.sb + 40) & lat_ok(s, d, cfg.terrain_clip) & owns(bd_piece, w_m)
        )

    progress(0.76, "shaping the ground")
    unc_s = s_all[unc_ext]

    def bd_weight(src_m):
        """1 where the backdrop *is* the stage (route stretches without track
        tiles, e.g. start areas): keep its exact heights there; fades to 0 (pure
        filler) within 60 m along the route."""
        if bd_piece is None or not len(unc_s):
            return np.zeros(len(src_m))
        d, idx = assign.tree.query(src_m[:, :2])
        s = source_station[idx]
        j = np.clip(np.searchsorted(unc_s, s), 1, len(unc_s) - 1)
        ds = np.minimum(np.abs(s - unc_s[j - 1]), np.abs(s - unc_s[j]))
        pc0 = pieces[bd_piece]
        w = np.clip(1.0 - ds / 60.0, 0.0, 1.0) * ((s >= pc0.sa - 40) & (s <= pc0.sb + 40))
        if w.any():
            w = w * owns(bd_piece, src_m)
        return w

    filler_h = None
    filler_vis = []
    filler_tris = []
    if cfg.filler:
        filler_h = _filler_height_field(
            surf,
            pieces,
            tree_new,
            tree_new_all,
            seam_dz,
            new_cl,
            [(m.vertices[:, [0, 2, 1]].astype(np.float64), m.soft_sub) for m in meshes if len(m.vertices)],
            bd_re_bytes=rb"Terrain\d{4}",
            bd_piece=bd_piece,
            bd_keep=bd_keep_world,
            bd_xform=(lambda U: U @ pieces[bd_piece].R.T + pieces[bd_piece].t) if bd_piece is not None else None,
            below=cfg.filler_below,
        )
        for name, pivot in bd_pivot.items():
            p = next((x for x in fs.index if x.endswith(f"/{name}.uasset") and "/MeshesTerrain/" in x), None)
            if p is None:
                rep.checks.append(f"backdrop mesh for {name} not found")
                continue
            mpk = Package(fs.read(p))
            mi = mesh_export_index(mpk)
            plan = filler_plan.get(name)
            if isinstance(plan, str):  # spare tile
                nb, _ = clip_mesh_payload(mpk.export_bytes(mpk.exports[mi]), lambda c: np.zeros(len(c), bool))
                new_pkgs[p] = rewrite_package(mpk, {mi: nb})
                continue

            def to_h(P, pv=pivot, plan=plan):
                # laid-out tile: world = P + pivot + shift; rigid tile: world = R (P + pivot) + t
                src = (P + pv) / 100
                if isinstance(plan, np.ndarray):
                    return filler_h(src[:, :2] + plan) * 100 - pv[2] - P[:, 2]
                if bd_piece is not None:
                    pc0 = pieces[bd_piece]
                    xy = (src @ pc0.R.T + pc0.t)[:, :2]
                    dz = (filler_h(xy) - pc0.t[2]) * 100 - pv[2] - P[:, 2]
                    return (1.0 - bd_weight(src)) * dz  # start/finish areas keep their ground
                return np.zeros(len(P))

            nb, mx = displace_payload(mpk.export_bytes(mpk.exports[mi]), to_h)
            new_pkgs[p] = rewrite_package(mpk, {mi: nb})
            # visual check: final filler vertices (LOD0) in world space
            lod0 = find_lods(nb)[0]
            Pn = np.frombuffer(nb, "<f4", 3 * lod0.n, lod0.pos_off).reshape(-1, 3).astype(np.float64)
            srcn = (Pn + pivot) / 100
            if isinstance(plan, np.ndarray):
                Wn = srcn.copy()
                Wn[:, :2] += plan
            else:
                Wn = srcn @ pieces[bd_piece].R.T + pieces[bd_piece].t
            filler_vis.append(Wn)
            for off_, cnt_, w_ in lod0.index_bufs:  # every section's triangles
                tri_ = np.frombuffer(nb, "<u4" if w_ == 4 else "<u2", cnt_ // w_, off_).reshape(-1, 3)
                tri_ = tri_[(tri_[:, 0] != tri_[:, 1]) | (tri_[:, 1] != tri_[:, 2])]  # skip hidden ones
                if len(tri_):
                    filler_tris.append(Wn[tri_][:, :, :2])
        rep.checks.append(
            f"filler ground: {len(bd_pivot)} tiles reshaped to the stage height field "
            f"({cfg.filler_below:.1f} m below the stage terrain)"
        )
    if filler_vis:
        Fv = np.vstack(filler_vis)
        ftree = cKDTree(Fv[:, :2])
        rp = resample(new_cl, 10.0)
        rp_s = arclength(rp)
        above = []
        for k_r, p_r in enumerate(rp):
            ii = ftree.query_ball_point(p_r[:2], 6.0)
            if ii and Fv[ii, 2].max() > p_r[2] + 1.0:
                above.append((float(rp_s[k_r]), Fv[ii, 2].max() - p_r[2]))
        if above:
            runs, a0, mx_ = [], above[0][0], 0.0
            for (x0, h0), nxt in zip(above, above[1:] + [(None, 0)]):
                mx_ = max(mx_, h0)
                if nxt[0] is None or nxt[0] - x0 > 10.0:
                    runs.append((a0, x0, mx_))
                    a0, mx_ = nxt[0], 0.0
            rep.checks.append(
                "FILLER ABOVE ROAD (visual): "
                + ", ".join(f"{r0:.0f}-{r1:.0f} m (+{h:.0f} m)" for r0, r1, h in runs[:8])
            )
        else:
            rep.checks.append("filler ground: below the road everywhere")
        # coverage: the filler must reach under the whole stage, or the stage's
        # clipped terrain edges hang in the air (seen from below: floating strips)
        # (inside a filler triangle; the backdrop meshes are coarse, so vertex
        # distance says little)
        far = np.ones(len(rp), bool)
        if filler_tris:
            T = np.vstack(filler_tris)
            cen = T.mean(1)
            rad = np.linalg.norm(T - cen[:, None, :], axis=2).max()
            ctree = cKDTree(cen)
            for k_r, p_r in enumerate(rp[:, :2]):
                for ti in ctree.query_ball_point(p_r, rad):
                    a, b_, c_ = T[ti]
                    v0, v1, v2 = c_ - a, b_ - a, p_r - a
                    den = v0[0] * v1[1] - v1[0] * v0[1]
                    if den == 0:
                        continue
                    u = (v2[0] * v1[1] - v1[0] * v2[1]) / den
                    v = (v0[0] * v2[1] - v2[0] * v0[1]) / den
                    if u >= -1e-6 and v >= -1e-6 and u + v <= 1 + 1e-6:
                        far[k_r] = False
                        break
        if far.any():
            idx = np.flatnonzero(far)
            rep.checks.append(
                f"FILLER COVERAGE: {far.mean():.0%} of the route has no filler ground under it "
                f"(between {rp_s[idx[0]]:.0f} and {rp_s[idx[-1]]:.0f} m)"
            )
        else:
            rep.checks.append("filler coverage: under the whole route")
    if cfg.filler:
        pass
    elif bd_piece is not None and not cfg.debug_no_clip:
        for name, pivot in bd_pivot.items():
            p = next((x for x in fs.index if x.endswith(f"/{name}.uasset") and "/MeshesTerrain/" in x), None)
            if p is None:
                rep.checks.append(f"backdrop mesh for {name} not found")
                continue
            mpk = Package(fs.read(p))
            mi = mesh_export_index(mpk)
            nb, st = clip_mesh_payload(
                mpk.export_bytes(mpk.exports[mi]), lambda cen, pv=pivot: bd_keep_world((cen + pv) / 100)
            )
            new_pkgs[p] = rewrite_package(mpk, {mi: nb})
            rep.clipped.append((p.rsplit("/", 1)[-1], st["lods"], st["dropped"], st["tris"]))

    # ---- baked collision
    progress(0.8, "building collision")
    out_meshes: list[W.MeshOut] = []
    for m in meshes:
        if b"OriginQuad" in m.soft_sub:
            out_meshes.append(m)
            continue
        if len(m.indices) < 3:
            continue
        U = m.vertices[:, [0, 2, 1]]
        tri = m.indices.reshape(-1, 3)
        cen = U[tri].mean(1)
        mt = re.search(rb"Track(\d\d)(\d\d)", m.soft_sub)
        ks = []  # per layer: triangle -> piece (-1: none)
        if mt:
            key = (int(mt.group(1)), int(mt.group(2)))
            d, idx = assign.tree.query(cen[:, :2])
            s = source_station[idx]
            lim = cfg.road_clip if b"Road" in m.soft_sub else cfg.terrain_clip
            for lay in range(n_layers):
                j = tile_pieces.get((lay, key))
                if j is None:
                    continue
                pc = pieces[j]
                keep = (s >= pc.sa - cfg.range_ext) & (s <= pc.sb + cfg.range_ext) & lat_ok(s, d, lim) & owns(j, cen)
                ks.append(np.where(keep, j, -1))
        elif re.search(rb"Terrain\d{4}", m.soft_sub):
            if filler_h is not None:
                mt_ = re.search(rb"Terrain(\d{4})", m.soft_sub)
                plan = filler_plan.get(f"SM_{lvl}Terrain{mt_.group(1).decode()}")
                if isinstance(plan, str):
                    continue  # spare tile: no collision
                V = U.astype(np.float64).copy()
                if isinstance(plan, np.ndarray):
                    V[:, :2] += plan
                    w_bd = np.zeros(len(V))
                else:
                    w_bd = bd_weight(V)
                    V = V @ pieces[bd_piece].R.T + pieces[bd_piece].t
                V[:, 2] = w_bd * V[:, 2] + (1.0 - w_bd) * filler_h(V[:, :2])
                out_meshes.append(
                    W.MeshOut(
                        V[:, [0, 2, 1]],
                        m.indices.astype(np.uint32),
                        m.material_ref,
                        m.soft_pkg,
                        m.soft_asset,
                        m.soft_sub,
                        m.extra,
                        m.group,
                    )
                )
                rep.baked_tris += len(tri)
                continue
            if bd_piece is None:
                continue
            ks.append(np.where(bd_keep_world(cen), bd_piece, -1))
        elif b"PCGGrass" in m.soft_sub:
            continue  # grass/rock instances stay behind (not moved)
        else:
            for lay in range(n_layers):  # props: once per layer that uses their spot
                k, d, s, _ = assign(cen, layer=lay)
                k[~lat_ok(s, d, cfg.prop_lateral)] = -1
                for kk in np.unique(k[k >= 0]):
                    m_ = k == kk
                    k[np.flatnonzero(m_)[~owns(int(kk), cen[m_])]] = -1
                ks.append(k)
        for k in ks:
            for kk in np.unique(k[k >= 0]):
                sel = tri[k == kk]
                used, inv = np.unique(sel, return_inverse=True)
                pc = pieces[int(kk)]
                V = U[used] @ pc.R.T + pc.t
                if pc.shears and not re.search(rb"Terrain\d{4}", m.soft_sub):  # backdrop: rigid like its mesh
                    V = warp(int(kk), V)
                V[:, 2] += seam_dz(int(kk), V)
                V = V[:, [0, 2, 1]]
                out_meshes.append(
                    W.MeshOut(
                        V,
                        inv.ravel().astype(np.uint32),
                        m.material_ref,
                        m.soft_pkg,
                        m.soft_asset,
                        m.soft_sub,
                        m.extra,
                        m.group,
                    )
                )
                rep.baked_tris += len(sel)
    rep.baked_meshes = len(out_meshes)
    # drivability: collision under the route must sit just below the route line
    # (catches lowered/missing ground, e.g. a start area on reshaped terrain)
    gv = np.vstack([mo.vertices[:, [0, 2, 1]] for mo in out_meshes if len(mo.vertices)])
    gtree = cKDTree(gv[:, :2])
    rp = resample(new_cl, 10.0)
    rp_s = arclength(rp)
    bad_s = []
    for k_r, p_r in enumerate(rp):
        ii = gtree.query_ball_point(p_r[:2], 4.0)
        z = gv[ii, 2] if ii else np.empty(0)
        z = z[z <= p_r[2] + 1.5]
        if not len(z) or p_r[2] - z.max() > 2.0:
            bad_s.append(float(rp_s[k_r]))
    if bad_s:
        runs, a0 = [], bad_s[0]
        for x0, x1 in zip(bad_s, bad_s[1:] + [None]):
            if x1 is None or x1 - x0 > 10.0:
                runs.append((a0, x0))
                a0 = x1
        rep.checks.append(
            "DRIVABILITY: no ground at the road surface at "
            + ", ".join(f"{r0:.0f}-{r1:.0f} m" for r0, r1 in runs[:8])
            + (" ..." if len(runs) > 8 else "")
        )
    else:
        rep.checks.append("drivability: ground at the road surface along the whole route")
    steps = _road_steps(out_meshes, new_cl)
    if steps:
        rep.checks.append(
            "STEPS ON THE ROAD (collision jumps > 0.4 m within 1 m): "
            + ", ".join(f"{s_:.0f} m ({lat:+.0f} m: {dz:+.1f} m)" for s_, lat, dz in steps[:8])
            + (" ..." if len(steps) > 8 else "")
        )
    else:
        rep.checks.append("road surface: no steps (checked every 1 m at the centre and 2 m either side)")
    # seams: where terrain of two pieces meets at different heights (shows as
    # floating sheets, the lower side is culled from below)
    seam_pts = collections.defaultdict(list)
    for mo in out_meshes:
        if not (re.search(rb"Track\d{4}", mo.soft_sub) and len(mo.indices)):  # not the filler
            continue
        V = mo.vertices[:, [0, 2, 1]]
        cen = V[mo.indices.reshape(-1, 3)].mean(1)
        jj = int(np.argmin(np.array([np.linalg.norm(tree_new[j].query(cen[:50, :2])[0]) for j in range(len(pieces))])))
        seam_pts[jj].append(cen)
    rep.seams = []
    P = {j: np.vstack(v) for j, v in seam_pts.items()}
    for a in P:
        ta = cKDTree(P[a][:, :2])
        for b_ in P:
            if b_ <= a:
                continue
            d, ii = ta.query(P[b_][:, :2], distance_upper_bound=6.0)
            ok = np.isfinite(d)
            if not ok.any():
                continue
            dz = P[b_][ok, 2] - P[a][ii[ok], 2]
            big = np.abs(dz) > 3.0
            if big.any():
                q = P[b_][ok][big]
                ds, _ = cKDTree(new_cl[:, :2]).query(q[:, :2])
                rep.checks.append(
                    f"seam pieces {a}/{b_}: {int(big.sum())} spots with height step >3 m "
                    f"(max {np.abs(dz).max():.0f} m, {np.median(ds):.0f} m from the road)"
                )
    new_pkgs[ws.baked] = rewrite_package(bpk, {0: W.baked_payload_full(bf, out_meshes)})

    # ---- pacenotes
    progress(0.86, "pacenotes and mod files")
    for key, path in ws.tables.items():
        tpk = Package(fs.read(path))
        if key == "forward":
            notes = [
                W.NoteOut(float(n.s), float(getattr(n, "length", 30.0)), n.text.split())
                for n in layout.pacenotes
                if n.text.strip()
            ]
        else:
            notes = [
                W.NoteOut(float(n.s), 30.0, n.text.split()) for n in generate(new_cl[::-1].copy()) if n.text.strip()
            ]
        pay, names = W.pacenote_payload(tpk, notes)
        new_pkgs[path] = rewrite_package(tpk, {0: pay}, names)
        rep.notes[key] = len(notes)

    # ---- container
    os.makedirs(out_dir, exist_ok=True)
    lvl_container, _ = fs.index[ws.main]
    pak_name = f"pakchunk99-{cfg.own_level or lvl}_{cfg.tag}_9_P"
    cid = name_hash(pak_name)
    hdr_idx = next(i for i, x in enumerate(lvl_container.chunk_ids) if x[11] == 6)
    chunks = []
    if cfg.own_level:
        # the level under its own name: main + every cell (changed or not)
        # renamed, other changed packages as they are; fresh container header
        # every other changed asset (tile meshes, backdrop, backgrounds,
        # collision, pacenotes) gets its own copy under /Game/Stagegen/<name>/,
        # keeping its file name (so object names and export hashes stay valid),
        # so the original stage stays untouched
        lvl_ren = level_renamer(lvl, cfg.own_level)
        level_paths = set([ws.main] + ws.cells)

        def pkg_name_of(path):
            return "/Game/" + path.split("/Content/", 1)[1].rsplit(".", 1)[0]

        asset_map = {
            pkg_name_of(p): f"/Game/Stagegen/{cfg.own_level}/" + pkg_name_of(p)[len("/Game/") :]
            for p in new_pkgs
            if p not in level_paths
        }
        # names are stored as base + number ("..._Terrain" #2 = "..._Terrain_1"):
        # map a base name too, if every numbered package with that base is copied
        by_base = collections.defaultdict(set)
        for p in fs.index:
            if p.endswith((".uasset", ".umap")) and "/Content/" in p:
                nm = pkg_name_of(p)
                m_ = re.match(r"(.*)_(\d+)$", nm)
                if m_:
                    by_base[m_.group(1)].add(nm)
        for base, variants in by_base.items():
            if (
                base not in asset_map
                and variants
                and all(v in asset_map for v in variants)
                and pkg_name_of_path_exists(fs, base) is False
            ):
                asset_map[base] = f"/Game/Stagegen/{cfg.own_level}/" + base[len("/Game/") :]

        def ren(n: str) -> str:
            if n in asset_map:
                return asset_map[n]
            head, dot, tail = n.partition(".")
            if dot and head in asset_map:  # object path pkg.object
                return asset_map[head] + "." + tail
            return lvl_ren(n)

        out_pkgs = {}
        for path in [ws.main] + ws.cells:
            data = new_pkgs.get(path) or fs.read(path)
            new_path = path.replace(f"/Levels/{lvl}.umap", f"/Levels/{cfg.own_level}.umap").replace(
                f"/Levels/{lvl}/", f"/Levels/{cfg.own_level}/"
            )
            out_pkgs[path] = (new_path, rename_package(data, ren))
        for path, data in new_pkgs.items():
            if path not in out_pkgs:
                new_path = path.replace("/Content/", f"/Content/Stagegen/{cfg.own_level}/", 1)
                out_pkgs[path] = (new_path, rename_package(data, ren))
        rep.checks.append(
            f"own level: {sum(1 for p in new_pkgs if p not in level_paths)} changed assets copied to "
            f"/Game/Stagegen/{cfg.own_level}/"
        )
        src_of = {}  # out_pkgs key -> source file (for chunk ids)
        if cfg.debug_dup_cells:
            from .wp_dup import copy_cell_package, duplicate_cells, new_cell_id

            new_lp = f"/Game/Levels/{cfg.own_level}"
            copies = [(src, new_cell_id(src, k)) for k, src in enumerate(cfg.debug_dup_cells)]
            mp, md = out_pkgs[ws.main]
            out_pkgs[ws.main] = (
                mp,
                duplicate_cells(md, new_lp, copies, cfg.debug_dup_grid, cfg.debug_dup_same_pkg, cfg.debug_dup_new_guid),
            )
            for src, new in [] if cfg.debug_dup_same_pkg else copies:
                cp = next(p for p in ws.cells if p.rsplit("/", 1)[1].split(".")[0] == src)
                np_, cd = out_pkgs[cp]
                key = f"{cp}#dup#{new}"
                out_pkgs[key] = (np_.replace(src, new), copy_cell_package(cd, new_lp, src, new))
                src_of[key] = cp
            rep.checks.append(f"debug: duplicated cells {copies}")
        if dup_pkgs:
            # repeated tiles: layer k's clipped tile meshes as copies under
            # /Game/Stagegen/<name>/L<k>/, and the cell copies (new IDs,
            # registered in the main level) pointing at them
            from .wp_dup import copy_cell_package, duplicate_cells, new_cell_id

            new_lp = f"/Game/Levels/{cfg.own_level}"
            ren_of = {}
            for lay in range(1, n_layers):
                lay_map = {
                    pkg_name_of(p): f"/Game/Stagegen/{cfg.own_level}/L{lay}/" + pkg_name_of(p)[len("/Game/") :]
                    for (l_, p) in layer_assets
                    if l_ == lay
                }
                for base, variants in by_base.items():
                    if (
                        base not in lay_map
                        and variants
                        and all(v in lay_map for v in variants)
                        and pkg_name_of_path_exists(fs, base) is False
                    ):
                        lay_map[base] = f"/Game/Stagegen/{cfg.own_level}/L{lay}/" + base[len("/Game/") :]

                def ren_k(n: str, lay_map=lay_map) -> str:
                    if n in lay_map:
                        return lay_map[n]
                    head, dot, tail = n.partition(".")
                    if dot and head in lay_map:
                        return lay_map[head] + "." + tail
                    return ren(n)

                ren_of[lay] = ren_k
            for (lay, p), data in layer_assets.items():
                key = f"{p}#L{lay}"
                out_pkgs[key] = (
                    p.replace("/Content/", f"/Content/Stagegen/{cfg.own_level}/L{lay}/", 1),
                    rename_package(data, ren_of[lay]),
                )
                src_of[key] = p
            copies = []
            for (cp, lay), data in sorted(dup_pkgs.items()):
                src_id = cp.rsplit("/", 1)[1].split(".")[0]
                new_id = new_cell_id(src_id, 100 + lay)
                copies.append((src_id, new_id))
                np_ = out_pkgs[cp][0].replace(src_id, new_id)
                key = f"{cp}#dup#{new_id}"
                out_pkgs[key] = (np_, copy_cell_package(rename_package(data, ren_of[lay]), new_lp, src_id, new_id))
                src_of[key] = cp
            mp, md = out_pkgs[ws.main]
            out_pkgs[ws.main] = (mp, duplicate_cells(md, new_lp, copies))
            rep.checks.append(
                f"repeated tiles: {len(copies)} cell copies registered, "
                f"{len(layer_assets)} tile meshes copied for layers 1..{n_layers - 1}"
            )
        # menu entries: not in this container; installing writes one shared
        # table container for all installed stages from this metadata.
        # Dry run here so a stage that cannot get rows fails at build time.
        import json
        from .tables import STAGE_META, stage_entries

        meta = {
            "level": lvl,
            "name": cfg.own_level,
            "display": cfg.display_name or f"Stagegen {cfg.own_level}",
            "length_km": round(layout.length / 1000, 2),
            "params": cfg.build_params or {},
        }
        _, tnotes = stage_entries(GameFS(paks_dir, include_mods=True), lvl, meta["name"], meta["display"])
        with open(os.path.join(out_dir, pak_name + STAGE_META), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1)
        rep.checks += [f"menu: {n}" for n in tnotes]
        src_headers: dict = {}
        entries = {}
        for path, (new_path, data) in out_pkgs.items():
            c, i = fs.index[src_of.get(path, path)]
            if id(c) not in src_headers:
                h_i = next(k for k, x in enumerate(c.chunk_ids) if x[11] == 6)
                src_headers[id(c)] = CH.parse(c.read_entry(h_i))
            old_id = struct.unpack_from("<Q", c.chunk_ids[i], 0)[0]
            pkg_name = "/Game/" + new_path.split("/Content/", 1)[1].rsplit(".", 1)[0]
            pid = CH.package_id(pkg_name)
            orig = src_headers[id(c)].entries.get(old_id, CH.StoreEntry())
            entries[pid] = CH.StoreEntry(
                [CH.package_id(n) for n in Package(data).imported_package_names()], orig.shader_maps
            )
            chunks.append(ChunkIn(struct.pack("<Q", pid) + c.chunk_ids[i][8:], data, new_path))
        chunks.append(ChunkIn(container_header_chunk_id(cid), CH.build(CH.ContainerHeader(cid, entries))))
        rep.checks.append(
            f"own level: /Game/Levels/{cfg.own_level} ({len(ws.cells)} cells renamed with it); "
            f"open it with the console command 'open {cfg.own_level}'"
        )
    else:
        for path, data in new_pkgs.items():
            c, i = fs.index[path]
            chunks.append(ChunkIn(c.chunk_ids[i], data, path))
        chunks.append(
            ChunkIn(container_header_chunk_id(cid), retarget_container_header(lvl_container.read_entry(hdr_idx), cid))
        )
    with open(os.path.join(os.path.dirname(__file__), "empty_mod.pak"), "rb") as f:
        pak_template = f.read()
    base = os.path.join(out_dir, pak_name)
    write_container(base, cid, chunks, pak_template)
    rep.out_files = [base + ext for ext in (".utoc", ".ucas", ".pak")] + (
        [base + ".stagegen.json"] if cfg.own_level else []
    )
    rep.packages = len(new_pkgs)
    progress(0.93, "checking the result")
    rep.checks += verify_wp(
        base + ".utoc",
        ws,
        new_pkgs,
        payloads,
        rules,
        {p: np_ for p, (np_, _) in out_pkgs.items() if "#" not in p} if cfg.own_level else None,
    )
    return rep


def _road_steps(out_meshes, new_cl, lats=(-2.0, 0.0, 2.0), max_jump=0.4, step=1.0):
    """Steps in the collision under the road: the ground height (highest
    triangle at most 1.5 m above the route) every `step` m along the route at
    the given lateral offsets; jumps larger than max_jump between neighbouring
    samples. Returns [(s m, lateral m, jump m)], one per spot (s rounded)."""
    P = resample(new_cl, step)
    s = arclength(P)
    t = np.gradient(P[:, :2], axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    n = np.c_[-t[:, 1], t[:, 0]]
    route_tree = cKDTree(P[:, :2])
    lat_max = max(abs(x) for x in lats)
    tris = []
    for mo in out_meshes:
        # road and terrain only (track tiles, backdrop/filler): props such as
        # bales or barriers by the road are meant to stand up
        if len(mo.indices) < 3 or not re.search(rb"Track\d{4}|Terrain\d{4}", mo.soft_sub):
            continue
        V = np.asarray(mo.vertices, np.float64)[:, [0, 2, 1]]
        # skip meshes nowhere near the road before expanding them into triangles
        lo, hi = V[:, :2].min(0), V[:, :2].max(0)
        if not route_tree.query_ball_point((lo + hi) / 2, float(np.linalg.norm(hi - lo)) / 2 + lat_max + 1.0):
            continue
        T = V[np.asarray(mo.indices).reshape(-1, 3)]
        c = T[:, :, :2].mean(1)
        r = np.linalg.norm(T[:, :, :2] - c[:, None, :], axis=2).max(1)
        d, _ = route_tree.query(c)
        tris.append(T[d <= lat_max + 1.0 + r])
    if not tris:
        return []
    T = np.vstack(tris)
    del tris
    C = T[:, :, :2].mean(1)
    rad = np.linalg.norm(T[:, :, :2] - C[:, None, :], axis=2).max(1)
    a, b, c3 = T[:, 0], T[:, 1], T[:, 2]
    v0, v1 = c3[:, :2] - a[:, :2], b[:, :2] - a[:, :2]
    den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    # triangles in size classes, each searched with its own radius (one radius
    # from the largest, 50 m background triangles gave thousands of candidates
    # per sample for the 1-2 m road triangles)
    classes = []
    for lo_r, hi_r in ((0.0, 2.0), (2.0, 8.0), (8.0, 32.0), (32.0, np.inf)):
        sel = np.flatnonzero((rad > lo_r) & (rad <= hi_r)) if lo_r else np.flatnonzero(rad <= hi_r)
        if len(sel):
            classes.append((cKDTree(C[sel]), sel, float(rad[sel].max())))
    found = {}
    for lat in lats:
        Q = P[:, :2] + n * lat
        # highest ground at most 1.5 m above the route at each sample, all
        # (sample, candidate triangle) pairs at once, in chunks to bound memory
        zmax = np.full(len(Q), -np.inf)
        for (ctree, sel, R), q0 in ((cl, q0) for cl in classes for q0 in range(0, len(Q), 2000)):
            cands = ctree.query_ball_point(Q[q0 : q0 + 2000], R, workers=-1)
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
            low = zz <= P[qi, 2] + 1.5
            np.maximum.at(zmax, qi[low], zz[low])
        z = np.where(np.isfinite(zmax), zmax, np.nan)
        # a jump between two neighbouring samples that both have ground
        jump = np.abs(np.diff(z)) > max_jump
        for i in np.flatnonzero(jump) + 1:
            key = round(s[i] / 5)
            if key not in found:
                found[key] = (float(s[i]), lat, float(z[i] - z[i - 1]))
    return sorted(found.values())


def _drop_streaming_mips(payload: bytes, names) -> tuple[bytes, bool]:
    """RuntimeVirtualTextureComponent: remove StreamingTexture (12) and
    StreamLowMips (13). The streaming texture is baked for the original map;
    without it the whole RVT is rendered at runtime from the moved writers."""
    from ..extract.unversioned import Reader
    from .writers import _hdr_for

    r = Reader(payload, 0)
    h = r.header()
    if not {12, 13} <= set(h) or [x for x in h if x < 12] != [0, 4]:
        return payload, False
    p = r.p
    p += 16  # soft object path: two FNames
    n = struct.unpack_from("<i", payload, p)[0]
    p += 4 + n  # + sub-path string
    p += 4  # [4] VirtualTexture (object ref)
    body = payload[r.p : p] + payload[p + 8 :]  # skip [12] and [13] (4 bytes each)
    return _hdr_for([x for x in h if x not in (12, 13)]) + body, True


def verify_wp(
    utoc: str, ws: WPSource, new_pkgs: dict, payloads: dict, rules, path_map: dict | None = None
) -> list[str]:
    from ..extract.bakeddata import read_baked
    from ..extract.pacenote_table import read_pacenote_table

    out = []
    c_ = Container(utoc)

    class _C:  # reads by original path (own level: renamed in the container)
        @staticmethod
        def read(p):
            return c_.read((path_map or {}).get(p, p))

    c = _C()
    bad = 0
    n_patched = 0
    bad_names = collections.Counter()
    for path, pl in payloads.items():
        pk = Package(c.read(path))
        comps = {cc.export for cc in decode_components(pk, rules)}
        want = [
            i
            for i in pl
            if pk.exports[i].name not in SPLINES
            and not pk.exports[i].name.startswith(
                ("WorldPartitionRuntimeSpatialHash", "WorldPartitionRuntimeCellData", "StaticMesh_", "HISM_", "ISM_")
            )  # instance data, no transform
            and not (
                0 <= pk.exports[i].outer < len(pk.exports)
                and pk.exports[pk.exports[i].outer].name.startswith(
                    ("InstancedFoliageActor", "PCGGrassContainer", "PCGPartitionActor", "PCGStamp", *ROOTLESS_TREES)
                )
            )
        ]
        n_patched += len(want)
        bad += sum(i not in comps for i in want)
        bad_names.update(
            re.sub(r"(_UAID_\w+|_\d+)+$", "", pk.exports[i].name)
            + " < "
            + re.sub(
                r"(_UAID_\w+|_\d+)+$",
                "",
                pk.exports[pk.exports[i].outer].name if 0 <= pk.exports[i].outer < len(pk.exports) else "-",
            )
            for i in want
            if i not in comps
        )
    out.append(
        f"level: {n_patched} patched components in {len(payloads)} packages, {bad} fail to re-decode"
        + (f" ({', '.join(f'{k} x{v}' for k, v in bad_names.most_common(5))})" if bad else "")
    )
    for path, kind in ws.route_cells.items():
        if kind == "cut":
            continue
        sp = level_splines(Package(c.read(path)))
        out.append(f"route {kind}: " + ", ".join(f"{k}={float(arclength(v)[-1]):.0f}m" for k, v in sp.items()))
    bpk = Package(c.read(ws.baked))
    meshes = read_baked(bpk)
    out.append(
        f"baked: {W.check_baked(bpk)}; triangle indices in range: "
        f"{all((m.indices.max() < len(m.vertices)) if len(m.indices) else True for m in meshes)}"
    )
    for key, path in ws.tables.items():
        out.append(f"pacenotes {key}: {len(read_pacenote_table(Package(c.read(path))))} rows")
    return out
