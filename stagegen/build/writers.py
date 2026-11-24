# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Serialisers for the three asset kinds a generated stage rewrites:
pacenote DataTable rows, BakedDataContainer meshes and DMSplineComponent curves.

Each writer re-parses the original export to find the byte ranges it
replaces, so anything it does not understand is carried over verbatim."""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from ..extract import bakeddata as B
from ..extract.unversioned import Reader
from ..extract.zen import Package


def _hdr(n_values: int, skip: int = 0, last: bool = True) -> bytes:
    return struct.pack("<H", (skip & 0x7F) | (int(last) << 8) | (n_values << 9))


def _fname(names: list[str], s: str) -> bytes:
    return struct.pack("<II", names.index(s), 0)


def _fstring(s: str) -> bytes:
    if not s:
        return struct.pack("<i", 0)
    b = s.encode("latin-1") + b"\0"
    return struct.pack("<i", len(b)) + b


# --------------------------------------------------------------- pacenotes
@dataclass
class NoteOut:
    distance: float
    length: float
    tokens: list[str]
    flag: int = 0
    line_id: str = ""


def pacenote_payload(pk: Package, notes: list[NoteOut]) -> tuple[bytes, list[str]]:
    """Returns (export payload, full name list incl. any appended names)."""
    e = pk.exports[0]
    d = pk.d
    r = Reader(d, e.data_offset)
    for _ in r.header():
        r.i32()
    r.i32()
    count_pos = r.p
    n = r.i32()
    # skip original rows to find the suffix
    for _ in range(n):
        r.fname(pk.names)
        for f in r.header():
            if f in (0, 1):
                r.f32()
            elif f == 2:
                for _g in r.header():
                    for _k in range(r.u32()):
                        r.fname(pk.names)
            elif f == 3:
                r.u8()
            elif f == 4:
                r.fstring()
            elif f == 5:
                r.i32()
    prefix = d[e.data_offset : count_pos]
    suffix = d[r.p : e.data_offset + e.serial_size]

    names = list(pk.names)

    def need(s):
        if s not in names:
            names.append(s)

    rows = bytearray(struct.pack("<i", len(notes)))
    for k, note in enumerate(notes):
        rn = f"R{k:04d}"
        need(rn)
        for t in note.tokens:
            need(t)
    for k, note in enumerate(notes):
        rows += _fname(names, f"R{k:04d}")
        rows += _hdr(5)
        rows += struct.pack("<ff", note.distance, note.length)
        rows += _hdr(1) + struct.pack("<I", len(note.tokens))
        for t in note.tokens:
            rows += _fname(names, t)
        rows += struct.pack("<B", note.flag)
        rows += _fstring(note.line_id)
    return bytes(prefix) + bytes(rows) + bytes(suffix), names


# --------------------------------------------------------------- baked data
@dataclass
class MeshOut:
    vertices: np.ndarray  # (N,3) float, baked frame (x, up, y)
    indices: np.ndarray  # (M,) uint32
    material_ref: int  # raw package index (e.g. -4)
    soft_pkg: tuple[int, int]  # raw FName pairs of the soft object path
    soft_asset: tuple[int, int]
    soft_sub: bytes  # raw length-prefixed string bytes
    extra: np.ndarray  # the 4 trailing floats
    group: int = 8  # collision group key (base-game index lists)


def read_baked_raw(pk: Package):
    """Original meshes with raw references + byte ranges for rewriting."""
    e = pk.exports[0]
    d = pk.d
    r = Reader(d, e.data_offset)
    top = r.header()
    assert top[0] == 0
    start_meshes = r.p
    n = r.u32()
    meshes = []
    transform_bytes = None
    for _ in range(n):
        verts = np.zeros((0, 3), np.float32)
        extra = np.zeros(0, np.float32)
        idx = np.zeros(0, np.uint32)
        mat = 0
        sp = (0, 0)
        sa = (0, 0)
        sub = struct.pack("<i", 0)
        for f in r.header():
            if f == 0:
                t0 = r.p
                rot, tr, sc = B._transform(r)
                if not (np.allclose(rot, (0, 0, 0, 1)) and np.allclose(tr, 0) and np.allclose(sc, 1)):
                    raise NotImplementedError("non-identity baked mesh transform")
                transform_bytes = d[t0 : r.p]
            elif f == 1:
                verts = B._array(r, np.float32, 3).astype(np.float64)
            elif f == 2:
                extra = B._array(r, np.float32).copy()
            elif f == 3:
                idx = B._array(r, np.uint32).copy()
            elif f == 4:
                mat = r.i32()
            elif f == 5:
                sp = struct.unpack_from("<II", d, r.p)
                r.p += 8
                sa = struct.unpack_from("<II", d, r.p)
                r.p += 8
                n_sub = r.i32()
                sub = d[r.p - 4 : r.p + n_sub]  # raw length-prefixed bytes (no terminator)
                r.p += n_sub
        meshes.append(MeshOut(verts, idx, mat, sp, sa, sub, extra))
    tail = d[r.p : e.data_offset + e.serial_size]
    k = tail.find(struct.pack("<I", n))
    tail_prefix = tail[:k]
    table_len = 4 + 7 + 10 * (n - 1) if n else 4
    tail_suffix = tail[k + table_len :]
    head = d[e.data_offset : start_meshes]
    return meshes, head, transform_bytes, tail_prefix, tail_suffix


def baked_payload(
    head: bytes, transform_bytes: bytes, meshes: list[MeshOut], tail_prefix: bytes, tail_suffix: bytes
) -> bytes:
    out = bytearray(head)
    out += struct.pack("<I", len(meshes))
    for m in meshes:
        out += _hdr(6)
        out += transform_bytes
        v = np.ascontiguousarray(m.vertices, dtype=np.float32).ravel()
        out += struct.pack("<I", len(v)) + v.tobytes()
        ex = np.ascontiguousarray(m.extra, dtype=np.float32)
        out += struct.pack("<I", len(ex)) + ex.tobytes()
        ix = np.ascontiguousarray(m.indices, dtype=np.uint32).ravel()
        out += struct.pack("<I", ix.size) + ix.tobytes()
        out += struct.pack("<i", m.material_ref)
        out += struct.pack("<II", *m.soft_pkg) + struct.pack("<II", *m.soft_asset) + m.soft_sub
    out += tail_prefix
    out += struct.pack("<I", len(meshes))
    for i in range(len(meshes)):
        out += (b"\x80\x05\x01" + struct.pack("<i", 1)) if i == 0 else (b"\x00\x05" + struct.pack("<ii", i, 1))
    out += tail_suffix
    return bytes(out)


# ------------------------------------------------- baked data, full container
# Base-game BakedDataContainer (top fields 0-3):
#   0 meshes                     TArray<FBakedMesh>
#   1 shape entries              TArray<{TArray<Box>, TArray<Sphere>, TArray<Capsule>, i32 material, SoftPath}>
#   2 (empty array in every level seen)
#   3 index lists                3 x TArray<{i32 key, i32 0, u16 hdr, TSet<{start,count}>}> + u32 0
#     list 0 groups mesh indices, list 2 shape-entry indices, by collision key.
SHAPE_SCHEMAS = {
    0: {0: 24, 1: 24, 2: 4, 3: 4, 4: 4, 5: 4, 6: 8, 7: 1, 8: 1, 9: 1},  # box
    1: {0: 24, 1: 4, 2: 4, 3: 8, 4: 1, 5: 1, 6: 1},  # sphere
    2: {0: 24, 1: 24, 2: 4, 3: 4, 4: 4, 5: 8, 6: 1, 7: 1, 8: 1},  # capsule
}


@dataclass
class BakedFull:
    top: bytes  # top-level unversioned header bytes
    transform_bytes: bytes
    meshes: list  # MeshOut with .group set
    mid: bytes  # fields 1 and 2, verbatim
    list0_hdr: list  # [(key, z1, hdr2)] in original order
    rest: bytes  # lists 1, 2 and trailer, verbatim
    covered: set = None  # mesh indices that list 0 references


def _skip_shapes(r: Reader):
    n = r.u32()
    for _ in range(n):
        for f in r.header():
            if f == 3:
                r.i32()
            elif f in (0, 1, 2):
                sch = SHAPE_SCHEMAS[f]
                for _ in range(r.u32()):
                    for g in r.header():
                        if g not in sch:
                            raise ValueError(f"unknown collision shape field {f}.{g}")
                        r.p += sch[g]
            elif f == 4:
                r.p += 16
                n_ = r.i32()
                r.p += n_
            else:
                raise ValueError(f"unknown shape entry field {f}")


def _read_list(r: Reader):
    groups = []
    for _ in range(r.i32()):
        key, z1 = r.i32(), r.i32()
        hdr = r.raw(2)
        rem, n = r.i32(), r.i32()
        if rem != 0:
            raise ValueError("TSet with removed elements")
        ranges = []
        for _ in range(n):
            v = {0: 0, 1: 0}
            for f in r.header():
                v[f] = r.i32()
            ranges.append((v[0], v[1]))
        groups.append((key, z1, hdr, ranges))
    return groups


def read_baked_full(pk: Package) -> BakedFull:
    e = pk.exports[0]
    d = pk.d
    end = e.data_offset + e.serial_size
    r = Reader(d, e.data_offset)
    top_fields = r.header()
    top = d[e.data_offset : r.p]
    if top_fields != [0, 1, 2, 3]:
        raise ValueError(f"unexpected BakedDataContainer layout {top_fields}")
    meshes, head, transform_bytes, _, _ = read_baked_raw(pk)
    # position after the meshes: re-walk them
    r.u32()
    for _ in range(len(meshes)):
        for f in r.header():
            if f == 0:
                B._transform(r)
            elif f == 1:
                B._array(r, np.float32, 3)
            elif f == 2:
                B._array(r, np.float32)
            elif f == 3:
                B._array(r, np.uint32)
            elif f == 4:
                r.i32()
            elif f == 5:
                r.p += 16
                n_ = r.i32()
                r.p += n_
    mid0 = r.p
    _skip_shapes(r)
    if r.u32() != 0:
        raise ValueError("BakedDataContainer field 2 is not empty")
    mid = d[mid0 : r.p]
    l0 = _read_list(r)
    rest = d[r.p : end]
    key_of = {}
    for key, _, _, ranges in l0:
        for a, b in ranges:
            for i in range(a, a + b):
                key_of.setdefault(i, key)
    for i, m in enumerate(meshes):
        m.group = key_of.get(i, l0[0][0] if l0 else 8)
    return BakedFull(top, transform_bytes, meshes, mid, [(k, z, h) for k, z, h, _ in l0], rest, set(key_of))


def check_baked(pk: Package) -> str:
    """Parse every field strictly and require the parse to end exactly at the
    export's end (the engine asserts on any size mismatch)."""
    bf = read_baked_full(pk)
    r = Reader(bf.rest, 0)
    l1 = _read_list(r)
    l2 = _read_list(r)
    if r.p != len(bf.rest) - 4 or bf.rest[-4:] != bytes(4):
        raise ValueError(f"index lists end at {r.p}, expected {len(bf.rest) - 4}")
    n = len(bf.meshes)
    if bf.covered != set(range(n)):
        raise ValueError(f"mesh index list covers {len(bf.covered)} indices, {n} meshes")
    return f"{n} meshes all grouped {[k for k, _, _ in bf.list0_hdr]}, lists {len(l1)}/{len(l2)}, exact size"


def _runs(idx: list[int]) -> list[tuple[int, int]]:
    out = []
    for i in sorted(idx):
        if out and out[-1][0] + out[-1][1] == i:
            out[-1] = (out[-1][0], out[-1][1] + 1)
        else:
            out.append((i, 1))
    return out


def baked_payload_full(bf: BakedFull, meshes: list[MeshOut]) -> bytes:
    out = bytearray(bf.top)
    out += struct.pack("<I", len(meshes))
    for m in meshes:
        out += _hdr(6)
        out += bf.transform_bytes
        v = np.ascontiguousarray(m.vertices, dtype=np.float32).ravel()
        out += struct.pack("<I", len(v)) + v.tobytes()
        ex = np.ascontiguousarray(m.extra, dtype=np.float32)
        out += struct.pack("<I", len(ex)) + ex.tobytes()
        ix = np.ascontiguousarray(m.indices, dtype=np.uint32).ravel()
        out += struct.pack("<I", ix.size) + ix.tobytes()
        out += struct.pack("<i", m.material_ref)
        out += struct.pack("<II", *m.soft_pkg) + struct.pack("<II", *m.soft_asset) + m.soft_sub
    out += bf.mid
    # one range per run of consecutive meshes from the same actor (one body each)
    by_key: dict[int, list[tuple[int, int]]] = {}
    i = 0
    while i < len(meshes):
        j = i
        while (
            j + 1 < len(meshes)
            and meshes[j + 1].group == meshes[i].group
            and meshes[j + 1].soft_sub == meshes[i].soft_sub
        ):
            j += 1
        by_key.setdefault(meshes[i].group, []).append((i, j - i + 1))
        i = j + 1
    groups = [(k, z, h) for k, z, h in bf.list0_hdr if k in by_key]
    known = {k for k, _, _ in groups}
    groups += [(k, 0, b"\x00\x03") for k in by_key if k not in known]
    out += struct.pack("<i", len(groups))
    for k, z, h in groups:
        runs = by_key[k]
        out += struct.pack("<ii", k, z) + h + struct.pack("<ii", 0, len(runs))
        for a, b in runs:
            if a == 0:
                out += b"\x80\x05\x01" + struct.pack("<i", b)  # start is zero: masked out
            else:
                out += b"\x00\x05" + struct.pack("<ii", a, b)
    out += bf.rest
    return bytes(out)


# --------------------------------------------------------------- splines
def _skip_curve(r: Reader, vw: int, dbl: bool) -> tuple[list[int], bytes]:
    """Skip one FInterpCurve; return (its header indices, bytes of the fields
    after Points: bIsLooped u8 / LoopKeyOffset f32)."""
    h = r.header()
    if not h:
        return h, b""
    if h[0] == 0:
        n = r.u32()
        for _ in range(n):
            for f in r.header():
                if f == 0:
                    r.f32()
                elif f in (1, 2, 3):
                    r.raw(vw * (8 if dbl else 4))
                elif f == 4:
                    r.u8()
    start = r.p
    for f in h:
        if f == 1:
            r.u8()
        elif f == 2:
            r.f32()
        elif f != 0:
            raise ValueError(f"unexpected FInterpCurve field {f}")
    return h, r.d[start : r.p]


def _hdr_for(indices: list[int], zeros=()) -> bytes:
    """Encode an unversioned header for these present indices plus the
    zero-masked ones (explicitly zero; they carry no value bytes)."""
    frags = []
    prev = 0
    i = 0
    idx = sorted(set(indices) | set(zeros or ()))
    while i < len(idx):
        skip = idx[i] - prev
        while skip > 127:
            frags.append([127, 0])
            skip -= 127
        run = 1
        while i + run < len(idx) and idx[i + run] == idx[i] + run and run < 127:
            run += 1
        frags.append([skip, run])
        prev = idx[i] + run
        i += run
    if not frags:
        frags = [[0, 0]]
    zeros = set(zeros or ())
    out = b""
    bits = []  # zero-mask bits, in value order of flagged fragments
    pos = 0
    for k, (skip, run) in enumerate(frags):
        pos += skip
        vals = list(range(pos, pos + run))
        pos += run
        has_zero = any(v in zeros for v in vals)
        if has_zero:
            bits += [v in zeros for v in vals]
        out += struct.pack("<H", skip | (int(has_zero) << 7) | (int(k == len(frags) - 1) << 8) | (run << 9))
    if bits:
        m = sum(1 << i for i, b in enumerate(bits) if b)
        if len(bits) <= 8:
            out += struct.pack("<B", m)
        elif len(bits) <= 16:
            out += struct.pack("<H", m)
        else:
            out += m.to_bytes(4 * ((len(bits) + 31) // 32), "little")
    return out


def _quat_from_dir(t: np.ndarray) -> np.ndarray:
    """UE quaternion (x,y,z,w) rotating +X onto direction t (yaw + pitch, no roll)."""
    yaw = np.arctan2(t[1], t[0])
    pitch = np.arctan2(t[2], np.hypot(t[0], t[1]))
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    cp, sp = np.cos(-pitch / 2), np.sin(-pitch / 2)  # UE pitch: rotation about Y, nose-up positive
    # q = qYaw * qPitch
    return np.array([-sy * sp, cy * sp, sy * cp, cy * cp])


def spline_payload(pk: Package, export_name: str, pts_cm: np.ndarray) -> bytes:
    """Replace the SplineCurves of a DMSplineComponent with control points pts_cm (N,3)."""
    e = next(x for x in pk.exports if x.name == export_name)
    d = pk.d
    r = Reader(d, e.data_offset)
    top = r.header()
    if top and top[0] == 0:
        r.i32()
    curves_start = r.p
    cv = r.header()
    assert cv[:3] == [0, 1, 2] and set(cv) <= {0, 1, 2, 3}, cv
    has_reparam = 3 in cv
    curve_meta = [_skip_curve(r, 3, True), _skip_curve(r, 4, True), _skip_curve(r, 3, True)]
    if has_reparam:
        curve_meta.append(_skip_curve(r, 1, False))
    prefix = d[e.data_offset : curves_start]
    suffix = d[r.p : e.data_offset + e.serial_size]

    P = np.asarray(pts_cm, float)
    n = len(P)
    T = np.zeros_like(P)
    T[1:-1] = (P[2:] - P[:-2]) / 2
    T[0] = P[1] - P[0]
    T[-1] = P[-1] - P[-2]

    out = bytearray(prefix)
    out += _hdr(4 if has_reparam else 3)  # FSplineCurves: Position, Rotation, Scale[, Reparam]

    def curve_head(k):
        h, _ = curve_meta[k]
        return _hdr_for(sorted(set(h) | {0}))

    def curve_tail(k):
        return curve_meta[k][1]

    # Position
    out += curve_head(0) + struct.pack("<I", n)
    for i in range(n):
        out += _hdr(5) + struct.pack("<f", float(i))
        out += struct.pack("<3d", *P[i]) + struct.pack("<3d", *T[i]) + struct.pack("<3d", *T[i]) + b"\x03"
    out += curve_tail(0)
    # Rotation
    out += curve_head(1) + struct.pack("<I", n)
    for i in range(n):
        q = _quat_from_dir(T[i])
        out += _hdr(5) + struct.pack("<f", float(i))
        out += struct.pack("<4d", *q) * 3 + b"\x01"
    out += curve_tail(1)
    # Scale
    out += curve_head(2) + struct.pack("<I", n)
    for i in range(n):
        out += _hdr(5) + struct.pack("<f", float(i))
        out += struct.pack("<3d", 1, 1, 1) + struct.pack("<3d", 0, 0, 0) * 2 + b"\x01"
    out += curve_tail(2)
    if not has_reparam:
        return bytes(out) + suffix
    # ReparamTable: 10 steps per segment, (distance -> input key)
    steps = 10
    entries = []
    dist = 0.0
    for i in range(n - 1):
        p0, p1, m0, m1 = P[i], P[i + 1], T[i], T[i + 1]
        prev = p0
        for k in range(steps):
            t = k / steps
            if k:
                h00 = 2 * t**3 - 3 * t**2 + 1
                h10 = t**3 - 2 * t**2 + t
                h01 = -2 * t**3 + 3 * t**2
                h11 = t**3 - t**2
                cur = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
                dist += float(np.linalg.norm(cur - prev))
                prev = cur
            entries.append((dist, i + t))
        dist += float(np.linalg.norm(p1 - prev))
    entries.append((dist, float(n - 1)))
    out += curve_head(3) + struct.pack("<I", len(entries))
    for dval, key in entries:
        # InVal, OutVal only; tangents/mode zero -> fragment: 2 values then skip
        out += struct.pack("<H", (2 << 9) | (1 << 8)) + struct.pack("<ff", dval, key)
    out += curve_tail(3)
    out += suffix
    return bytes(out)
