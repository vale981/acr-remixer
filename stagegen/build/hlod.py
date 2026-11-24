# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Move a world-space HLOD StaticMesh (the visible track in ACR) in place.

HLOD track meshes (`StaticMesh_N` exports in the main level, one per 350 m
track tile) are baked in world coordinates and drawn by components without a
transform, so moving a tile's visuals means rewriting the mesh itself, same
size: positions (float32 cm), packed int8 tangents (rotated about Z) and the
two FBoxSphereBounds copies (doubles) near the start of the export. Triangles
are clipped with the same keep rule as the tile meshes.
"""
from __future__ import annotations

import struct

import numpy as np

from .mesh_clip import clip_mesh_payload, find_lods


def mesh_world_bounds(b: bytes):
    lods = find_lods(b)
    if not lods:
        return None
    P = np.frombuffer(b, "<f4", 3 * lods[0].n, lods[0].pos_off).reshape(-1, 3).astype(np.float64)
    return P.min(0), P.max(0)


def _bounds_offsets(b: bytes, lo, hi, limit: int) -> list[int]:
    """Offsets of FBoxSphereBounds (origin d3, extent d3, radius d) matching lo/hi."""
    c = (lo + hi) / 2
    out = []
    for k in range(0, limit - 56):
        ox, oy = struct.unpack_from("<2d", b, k)
        if abs(ox - c[0]) < 50 and abs(oy - c[1]) < 50:
            ex, ey = struct.unpack_from("<2d", b, k + 24)
            if abs(ex - (hi[0] - lo[0]) / 2) < 200 and abs(ey - (hi[1] - lo[1]) / 2) < 200:
                out.append(k)
    return out


def transform_mesh(b: bytes, R: np.ndarray, t_cm: np.ndarray, keep_fn=None) -> tuple[bytes, dict]:
    """R: 3x3 rotation (about Z), t_cm: translation (cm). keep_fn(centroids_cm_world) -> bool."""
    lods = find_lods(b)
    if not lods or any(len(l.index_bufs) == 0 for l in lods):
        raise ValueError("HLOD mesh layout not understood")
    lo, hi = mesh_world_bounds(b)
    boffs = _bounds_offsets(b, lo, hi, lods[0].pos_off)
    if not boffs:
        raise ValueError("HLOD mesh bounds not found")
    st = {"lods": len(lods), "dropped": 0, "tris": 0}
    if keep_fn is not None:
        b, cs = clip_mesh_payload(b, keep_fn)
        st["dropped"], st["tris"] = cs["dropped"], cs["tris"]
    out = bytearray(b)
    c, s = R[0, 0], R[1, 0]
    for lod in lods:
        P = np.frombuffer(b, "<f4", 3 * lod.n, lod.pos_off).reshape(-1, 3).astype(np.float64)
        out[lod.pos_off: lod.pos_off + 12 * lod.n] = (P @ R.T + t_cm).astype("<f4").tobytes()
        # tangents: bulk (8, n) a few bytes after the positions: TangentX, TangentZ as int8 xyzw
        p = lod.pos_off + 12 * lod.n
        hit = None
        for q in range(p, p + 64):
            if struct.unpack_from("<II", b, q) == (8, lod.n):
                hit = q + 8
                break
        if hit is not None:
            T = np.frombuffer(b, np.int8, 8 * lod.n, hit).reshape(-1, 8).astype(np.float64)
            for x, y in ((0, 1), (4, 5)):
                nx = c * T[:, x] - s * T[:, y]
                ny = s * T[:, x] + c * T[:, y]
                T[:, x], T[:, y] = nx, ny
            out[hit: hit + 8 * lod.n] = np.clip(np.round(T), -127, 127).astype(np.int8).tobytes()
            st["tangents"] = True
    # bounds: new AABB of the transformed box
    ctr = (lo + hi) / 2
    ext = (hi - lo) / 2
    nctr = R @ ctr + t_cm
    next_ = np.abs(R) @ ext
    for k in boffs:
        struct.pack_into("<3d", out, k, *nctr)
        struct.pack_into("<3d", out, k + 24, *next_)
    st["bounds"] = len(boffs)
    return bytes(out), st
