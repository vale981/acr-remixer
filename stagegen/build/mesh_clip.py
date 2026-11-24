# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Hide triangles of a cooked (non-Nanite) UStaticMesh without changing any
buffer size: dropped triangles are rewritten as degenerate (all three indices
equal), which the GPU skips.

Per LOD the cooked render data holds, inline: position buffer (stride 12,
float32 cm) -> tangents -> UVs -> colours -> index buffer -> depth-only index
buffer. Index buffers are serialised as `u32 b32Bit` + bulk array
(`u32 elem_size=1`, `i32 byte_count`, bytes). Every index buffer found after a
LOD's positions is clipped by triangle centroid, so the depth prepass and
shadows match what is drawn.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from ..extract.zen import Package


@dataclass
class Lod:
    pos_off: int  # offset of the vertex data inside the export
    n: int
    index_bufs: list  # [(data_offset, byte_count, width)]


def find_lods(b: bytes) -> list[Lod]:
    pos = []
    i = 0
    while True:
        i = b.find(b"\x0c\x00\x00\x00", i)
        if i < 0 or i + 16 > len(b):
            break
        st, n, es, n2 = struct.unpack_from("<IIII", b, i)
        # LODs shrink but never below a few dozen vertices; tiny matches are
        # coincidences inside tangent/UV data
        if (
            st == 12
            and es == 12
            and n == n2
            and (3 if not pos else 24) <= n < 20_000_000
            and i + 16 + 12 * n <= len(b)
            and (not pos or n <= pos[-1][1])
        ):
            pos.append((i + 16, n))
            i += 16 + 12 * n
            continue
        i += 1
    lods = []
    for k, (p0, n) in enumerate(pos):
        end = pos[k + 1][0] - 16 if k + 1 < len(pos) else len(b)
        j = p0 + 12 * n
        bufs = []
        while True:
            q = b.find(b"\x01\x00\x00\x00", j + 4, end - 8)  # elem_size == 1
            if q < 0:
                break
            j = q - 4
            b32, es, cnt = struct.unpack_from("<IIi", b, j)
            if b32 in (0, 1) and cnt > 0 and j + 12 + cnt <= end:
                w = 4 if b32 else 2
                if cnt % (3 * w) == 0:
                    a = np.frombuffer(b, "<u4" if w == 4 else "<u2", cnt // w, j + 12)
                    if len(a) and int(a.max()) < n:
                        bufs.append((j + 12, cnt, w))
                        j += 12 + cnt - 4
                        continue
            j += 1
        lods.append(Lod(p0, n, bufs))
    return lods


def clip_mesh_payload(b: bytes, keep_fn) -> tuple[bytes, dict]:
    """keep_fn(centroids_cm (T,3) in mesh-local space) -> bool mask (T,)."""
    out = bytearray(b)
    stats = {"lods": 0, "tris": 0, "dropped": 0, "bufs": 0}
    lods = find_lods(b)
    if not lods or any(len(l.index_bufs) == 0 for l in lods):
        raise ValueError(f"mesh layout not understood: LODs {[(l.n, len(l.index_bufs)) for l in lods]}")
    for lod in lods:
        P = np.frombuffer(b, "<f4", 3 * lod.n, lod.pos_off).reshape(-1, 3).astype(np.float64)
        stats["lods"] += 1
        for off, cnt, w in lod.index_bufs:
            dt = "<u4" if w == 4 else "<u2"
            tri = np.frombuffer(b, dt, cnt // w, off).reshape(-1, 3).copy()
            keep = np.asarray(keep_fn(P[tri].mean(1)), bool)
            tri[~keep] = tri[~keep][:, :1]
            out[off : off + cnt] = tri.astype(dt).tobytes()
            stats["bufs"] += 1
            stats["tris"] += len(tri)
            stats["dropped"] += int((~keep).sum())
    return bytes(out), stats


def displace_payload(b: bytes, dz_fn) -> tuple[bytes, float]:
    """Move vertices vertically in every LOD: dz_fn(positions_cm (N,3) mesh-local) -> dz_cm (N,).
    Returns the payload and the largest |dz| applied (cm)."""
    out = bytearray(b)
    mx = 0.0
    for lod in find_lods(b):
        P = np.frombuffer(b, "<f4", 3 * lod.n, lod.pos_off).reshape(-1, 3).astype(np.float64)
        dz = np.asarray(dz_fn(P), np.float64)
        if not np.any(dz):
            continue
        mx = max(mx, float(np.abs(dz).max()))
        P[:, 2] += dz
        out[lod.pos_off : lod.pos_off + 12 * lod.n] = P.astype("<f4").tobytes()
    return bytes(out), mx


def warp_payload(b: bytes, fn) -> tuple[bytes, float]:
    """Move vertices in every LOD: fn(positions_cm (N,3) mesh-local) -> new
    positions (N,3). The mesh bounds (FBoxSphereBounds copies) grow to contain
    the moved vertices. Returns the payload and the largest move (cm)."""
    from .hlod import _bounds_offsets

    out = bytearray(b)
    mx = 0.0
    lods = find_lods(b)
    lo0 = hi0 = lo1 = hi1 = None
    for li, lod in enumerate(lods):
        P = np.frombuffer(b, "<f4", 3 * lod.n, lod.pos_off).reshape(-1, 3).astype(np.float64)
        Q = np.asarray(fn(P), np.float64)
        d = np.linalg.norm(Q - P, axis=1)
        if not len(d) or not np.any(d):
            continue
        mx = max(mx, float(d.max()))
        out[lod.pos_off : lod.pos_off + 12 * lod.n] = Q.astype("<f4").tobytes()
        if li == 0:
            lo0, hi0, lo1, hi1 = P.min(0), P.max(0), Q.min(0), Q.max(0)
    if lo0 is not None and lods:
        nlo, nhi = np.minimum(lo0, lo1), np.maximum(hi0, hi1)
        for k in _bounds_offsets(b, lo0, hi0, lods[0].pos_off):
            ctr, ext = (nlo + nhi) / 2, (nhi - nlo) / 2
            struct.pack_into("<3d", out, k, *ctr)
            struct.pack_into("<3d", out, k + 24, *ext)
            struct.pack_into("<d", out, k + 48, float(np.linalg.norm(ext)))
    return bytes(out), mx


def mesh_export_index(pk: Package) -> int:
    """The UStaticMesh export (the largest one that is not the BodySetup)."""
    cands = [(e.serial_size, i) for i, e in enumerate(pk.exports) if not e.name.startswith("BodySetup")]
    return max(cands)[1]
