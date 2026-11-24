# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Locate scene-component transforms inside a cooked level so they can be
patched in place.

Unversioned property indices are counted most-derived-class first, so the
index of RelativeLocation differs per component class (StaticMeshComponent
163, BoxComponent 125, SceneComponent 5, ...). In every class the three
SceneComponent properties are consecutive: L = RelativeLocation,
L+1 = RelativeRotation, L+2 = RelativeScale3D, and AttachParent is L-5.

For each class we infer L by consensus: scan payloads for a location-like
double triple and see which header index lines up with it. A component is a
root (world-space transform) when AttachParent (L-5) is absent from its header.
"""

from __future__ import annotations

import collections
import struct
from dataclasses import dataclass

import numpy as np

from .unversioned import Reader
from .zen import Package

WORLD = 6.0e5  # |coordinate| bound in cm for this kind of level


@dataclass
class SceneTransform:
    export: int
    owner: int  # actor export index (outer)
    loc_off: int | None  # absolute byte offsets into package data
    rot_off: int | None
    scale_off: int | None
    root: bool
    loc: np.ndarray  # cm
    rot: np.ndarray  # pitch, yaw, roll (deg)
    scale: np.ndarray


def _triple(d, p):
    return struct.unpack_from("<3d", d, p)


def _loc_like(v):
    return (
        all(np.isfinite(v))
        and all(abs(x) < WORLD for x in v)
        and max(abs(x) for x in v) > 50
        and all(x == 0 or abs(x) > 1e-6 for x in v)
    )


def _rot_like(v):
    return all(np.isfinite(v)) and all(abs(x) <= 360.001 for x in v) and all(x == 0 or abs(x) > 1e-9 for x in v)


def _scale_like(v):
    return all(np.isfinite(v)) and all(1e-4 < abs(x) < 1e4 for x in v)


def _candidates(d, start, end, nbefore_min):
    """Offsets p (>= start) where a location-like triple sits."""
    out = []
    for p in range(start, end - 23):
        if _loc_like(_triple(d, p)):
            out.append(p)
    return out


def scene_transforms(pk: Package) -> list[SceneTransform]:
    d, ex = pk.d, pk.exports
    headers = {}
    for i, e in enumerate(ex):
        try:
            r = Reader(d, e.data_offset)
            h = r.header()
            headers[i] = (h, r.p)
        except Exception:  # noqa: BLE001 - not every export is property data
            continue

    # Infer L per class: for exports whose header has exactly one value before
    # a run of >= 2 consecutive indices, try each candidate L and check the
    # layout "loc at the k-th value position" is consistent from the tail.
    by_class = collections.defaultdict(list)
    for i, (h, p) in headers.items():
        by_class[ex[i].class_index].append(i)

    class_L: dict[int, int] = {}
    for cls, ids in by_class.items():
        votes = collections.Counter()
        for i in ids[:400]:
            h, p0 = headers[i]
            e = ex[i]
            end = e.data_offset + e.serial_size
            hs = set(h)
            for L in h:
                if L + 1 not in hs and L + 2 not in hs:
                    continue
                # find a location triple followed by the right number of triples
                for p in range(p0, end - 23):
                    v = _triple(d, p)
                    if not _loc_like(v):
                        continue
                    q = p + 24
                    ok = True
                    if L + 1 in hs:
                        ok &= q + 24 <= end and _rot_like(_triple(d, q))
                        q += 24
                    if ok and L + 2 in hs:
                        ok &= q + 24 <= end and _scale_like(_triple(d, q))
                        q += 24
                    if ok:
                        votes[L] += 1
                        break
        if votes:
            L, n = votes.most_common(1)[0]
            if n >= max(1, len(ids[:400]) // 3):
                class_L[cls] = L

    out = []
    for i, (h, p0) in headers.items():
        e = ex[i]
        L = class_L.get(e.class_index)
        if L is None or not ({L, L + 1, L + 2} & set(h)):
            continue
        hs = set(h)
        end = e.data_offset + e.serial_size
        loc_off = rot_off = scale_off = None
        loc = np.zeros(3)
        rot = np.zeros(3)
        sc = np.ones(3)
        found = False
        for p in range(p0, end - 23):
            q = p
            if L in hs:
                if not _loc_like(_triple(d, q)):
                    continue
                cand_loc = q
                q += 24
            else:
                cand_loc = None
            if L + 1 in hs:
                if q + 24 > end or not _rot_like(_triple(d, q)):
                    if cand_loc is None:
                        continue
                    continue
                cand_rot = q
                q += 24
            else:
                cand_rot = None
            if L + 2 in hs:
                if q + 24 > end or not _scale_like(_triple(d, q)):
                    continue
                cand_sc = q
                q += 24
            else:
                cand_sc = None
            loc_off, rot_off, scale_off = cand_loc, cand_rot, cand_sc
            found = True
            break
        if not found:
            continue
        if loc_off is not None:
            loc = np.array(_triple(d, loc_off))
        if rot_off is not None:
            rot = np.array(_triple(d, rot_off))
        if scale_off is not None:
            sc = np.array(_triple(d, scale_off))
        root = (L - 5) not in hs
        out.append(SceneTransform(i, e.outer, loc_off, rot_off, scale_off, root, loc, rot, sc))
    return out, class_L
