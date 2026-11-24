# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-FileCopyrightText: 2026 iceone88
# SPDX-License-Identifier: GPL-3.0-or-later AND MIT

"""Instanced (foliage / PCG grass) components: hide instances in place.

Instanced static mesh components serialise their instances as a bulk array of
FMatrix (16 doubles, 128 bytes each): `u32 128, i32 count, count*128 bytes`.
Hiding = zero scale (rows 0-2 zeroed, translation kept), so the count, the
cluster tree and all bounds stay valid.
"""

from __future__ import annotations

import struct

import numpy as np


def find_instances(b: bytes):
    """(offset of the first matrix, count) or None."""
    k = 0
    pat = struct.pack("<I", 128)
    while True:
        k = b.find(pat, k)
        if k < 0 or k + 8 > len(b):
            return None
        n = struct.unpack_from("<i", b, k + 4)[0]
        if 0 < n < 5_000_000 and k + 8 + 128 * n <= len(b):
            M = np.frombuffer(b, "<f8", 16 * n, k + 8).reshape(n, 4, 4)
            if np.allclose(M[:, 3, 3], 1.0) and np.allclose(M[:, :3, 3], 0.0):
                return k + 8, n
        k += 1


def instance_positions(b: bytes):
    hit = find_instances(b)
    if hit is None:
        return None
    off, n = hit
    return np.frombuffer(b, "<f8", 16 * n, off).reshape(n, 4, 4)[:, 3, :3].copy()


def _cluster_tree(b: bytes, inst_end: int, n: int):
    """(offset of the first node, count) of the HISM cluster tree after the
    instances: bulk `u32 64, i32 count`, nodes of 64 bytes (BoundMin f3,
    FirstChild, BoundMax f3, LastChild, FirstInstance, LastInstance, min/max
    instance scale f3 f3); the root covers instances 0..n-1."""
    k = b.find(struct.pack("<I", 64), inst_end)
    while 0 <= k < inst_end + 4096:
        m = struct.unpack_from("<i", b, k + 4)[0]
        if 0 < m < 1_000_000 and k + 8 + 64 * m <= len(b):
            fi, li = struct.unpack_from("<ii", b, k + 8 + 32)
            if fi == 0 and li == n - 1:
                return k + 8, m
        k = b.find(struct.pack("<I", 64), k + 1)
    return None


def _box_offsets(b: bytes, limit: int, lo, hi, tol: float):
    """Offsets of FBox (min d3, max d3) near lo/hi (cached component bounds)."""
    out = []
    want = np.r_[lo, hi]
    for a in range(8):
        n = (limit - a) // 8
        if n < 6:
            continue
        arr = np.frombuffer(b, "<f8", n, a)
        with np.errstate(invalid="ignore"):
            c0 = np.flatnonzero(np.abs(arr[: n - 5] - want[0]) < tol)
        for i in c0:
            if np.all(np.abs(arr[i : i + 6] - want) < tol):
                out.append(a + 8 * int(i))
    return out


def move_hism(b: bytes, root_cm, place) -> tuple[bytes, dict]:
    """Move/hide the instances of a (H)ISM component attached to a root at root_cm.
    place(world_m (N,3), source_yaw_rad (N,)) ->
    (keep (N,), new_world_m (N,3), yaw_delta_rad (N,)).
    Instances are rewritten in place, hidden ones zero-scaled at their new spot;
    the cluster tree boxes and cached FBox bounds follow the moved instances."""
    hit = find_instances(b)
    if hit is None:
        return b, {"n": 0}
    off, n = hit
    M = np.frombuffer(b, "<f8", 16 * n, off).reshape(n, 4, 4).copy()
    root = np.asarray(root_cm, np.float64)
    W = (M[:, 3, :3] + root) / 100
    # UE instance matrices store their local axes by row; row 0 is local +X.
    source_yaw = np.arctan2(M[:, 0, 1], M[:, 0, 0])
    keep, NW, yaw = place(W, source_yaw)
    keep = np.asarray(keep, bool)
    c, s = np.cos(yaw), np.sin(yaw)
    for r in range(3):  # rows are the scaled axes: A' = A R^T
        x, y = M[:, r, 0].copy(), M[:, r, 1].copy()
        M[:, r, 0] = c * x - s * y
        M[:, r, 1] = s * x + c * y
    oldP = M[:, 3, :3].copy()
    M[:, 3, :3] = NW * 100 - root
    M[~keep, :3, :] = 0.0
    out = bytearray(b)
    out[off : off + 128 * n] = M.astype("<f8").tobytes()
    st = {"n": n, "kept": int(keep.sum()), "nodes": 0, "boxes": 0}
    newP = M[:, 3, :3]
    # instance bounds box (FBox, local) in the properties: contains every
    # instance, a mesh-sized margin around them
    fb = _instance_box(b, off - 8, oldP)
    tree = _cluster_tree(b, off + 128 * n, n)
    O = np.zeros(3)  # cluster boxes are relative to this origin
    if tree is not None and fb is not None:
        t0, _ = tree
        r_lo = np.array(struct.unpack_from("<3f", b, t0), np.float64)
        r_hi = np.array(struct.unpack_from("<3f", b, t0 + 16), np.float64)
        O = (fb[0] + fb[1]) / 2 - (r_lo + r_hi) / 2
        if np.linalg.norm(O) < 50:  # same frame (float noise)
            O = np.zeros(3)
    new_lo = new_hi = None
    if tree is not None:
        t0, m = tree
        for q in range(m):
            p = t0 + 64 * q
            lo = np.array(struct.unpack_from("<3f", b, p), np.float64)
            hi = np.array(struct.unpack_from("<3f", b, p + 16), np.float64)
            fi, li = struct.unpack_from("<ii", b, p + 32)
            if not (0 <= fi <= li < n):
                continue
            o_lo, o_hi = oldP[fi : li + 1].min(0) - O, oldP[fi : li + 1].max(0) - O
            mg = np.maximum(np.maximum(o_lo - lo, hi - o_hi), 0.0)
            mg[:2] = mg[:2].max()  # rotated meshes: symmetric in xy
            n_lo = newP[fi : li + 1].min(0) - O - mg
            n_hi = newP[fi : li + 1].max(0) - O + mg
            struct.pack_into("<3f", out, p, *n_lo)
            struct.pack_into("<3f", out, p + 16, *n_hi)
            st["nodes"] += 1
            if q == 0:
                new_lo, new_hi = n_lo + O, n_hi + O
    if fb is not None:
        if new_lo is None:  # no tree: grow the box with the instances
            mg = np.maximum(np.maximum(oldP.min(0) - fb[0], fb[1] - oldP.max(0)), 0.0)
            mg[:2] = mg[:2].max()
            new_lo, new_hi = newP.min(0) - mg, newP.max(0) + mg
        # every FBox copy, and the box-sphere bounds (origin, extent, radius) in
        # local and world space (world = local + root: the component sits at the root)
        for k in _box_offsets(b, off - 8, fb[0], fb[1], tol=1.0):
            if not _fbox_ok(b, k):
                continue
            struct.pack_into("<3d", out, k, *new_lo)
            struct.pack_into("<3d", out, k + 24, *new_hi)
            st["boxes"] += 1
        oc, oe = (fb[0] + fb[1]) / 2, (fb[1] - fb[0]) / 2
        nc, ne = (new_lo + new_hi) / 2, (new_hi - new_lo) / 2
        for shift in (np.zeros(3), root):
            for k in _box_offsets(b, off - 8, oc + shift, oe, tol=2.0):
                if not _bsb_ok(b, k):
                    continue
                struct.pack_into("<3d", out, k, *(nc + shift))
                struct.pack_into("<3d", out, k + 24, *ne)
                struct.pack_into("<d", out, k + 48, float(np.linalg.norm(ne)))
                st["boxes"] += 1
    return bytes(out), st


def _instance_box(b: bytes, limit: int, P):
    """The FBox (min d3, max d3) in the properties that tightly contains P."""
    lo, hi = P.min(0), P.max(0)
    for a in range(8):
        n = (limit - a) // 8
        if n < 6:
            continue
        arr = np.frombuffer(b, "<f8", n, a)
        with np.errstate(invalid="ignore"):
            c0 = np.flatnonzero((arr[: n - 5] <= lo[0] + 1) & (arr[: n - 5] >= lo[0] - 20000))
        for i in c0:
            v = arr[i : i + 6]
            k = a + 8 * int(i)
            if (
                np.all(v[:3] <= lo + 1)
                and np.all(v[:3] >= lo - 5000)
                and np.all(v[3:] >= hi - 1)
                and np.all(v[3:] <= hi + 5000)
                and k + 48 < len(b)
                and b[k + 48] == 1
            ):  # FBox.IsValid
                return v[:3].copy(), v[3:].copy()
    return None


def _fbox_ok(b: bytes, k: int) -> bool:
    return k + 48 < len(b) and b[k + 48] == 1


def _bsb_ok(b: bytes, k: int) -> bool:
    """Box-sphere bounds: radius equals |extent| (within 5 %)."""
    if k + 56 > len(b):
        return False
    e = np.array(struct.unpack_from("<3d", b, k + 24))
    r = struct.unpack_from("<d", b, k + 48)[0]
    return abs(r - np.linalg.norm(e)) <= 0.05 * max(r, 1.0)


def hide_instances(b: bytes, mask=None) -> tuple[bytes, int]:
    """Zero-scale the instances where mask is True (all if None). Returns (payload, hidden)."""
    hit = find_instances(b)
    if hit is None:
        return b, 0
    off, n = hit
    M = np.frombuffer(b, "<f8", 16 * n, off).reshape(n, 4, 4).copy()
    m = np.ones(n, bool) if mask is None else np.asarray(mask, bool)
    M[m, :3, :] = 0.0
    out = bytearray(b)
    out[off : off + 128 * n] = M.astype("<f8").tobytes()
    return bytes(out), int(m.sum())
