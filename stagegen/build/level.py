# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Exact, per-class decoding and patching of root scene-component transforms
in a cooked level (UE 5.6, unversioned properties).

Unversioned property indices count the most-derived class first, so the
SceneComponent block (RelativeLocation L, RelativeRotation L+1,
RelativeScale3D L+2, AttachParent L-5) sits at a class-specific index. The
rules below were derived from the WRC Generations stage levels and are only
applied to components whose decoded values pass sanity checks.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from ..extract.unversioned import Reader
from ..extract.zen import Package
from .writers import _hdr_for


@dataclass(frozen=True)
class Rule:
    L: int
    trailer: int
    extras: dict  # index -> byte size, for indices > L+2 seen in practice


# keyed by a representative export name of the class
RULES = {
    "StaticMeshComponent": Rule(163, 20, {171: 1, 177: 1}),
    "NewDecalComponent": Rule(15, 8, {}),
    "SceneComponent": Rule(5, 8, {33: 1, 41: 1}),
    "BoxComponent": Rule(125, 8, {}),
    "VirtualTextureComponent": Rule(26, 8, {}),
}
_REPRESENTATIVE = {
    "StaticMeshComponent0": "StaticMeshComponent",
    "NewDecalComponent": "NewDecalComponent",
    "SceneComp": "SceneComponent",
    "DefaultSceneRoot": "SceneComponent",
    "Trigger": "BoxComponent",
    "VirtualTextureComponent": "VirtualTextureComponent",
}


@dataclass
class Comp:
    export: int
    owner: int
    rule: Rule
    present: list
    hdr_end: int
    block_start: int
    block_end: int
    loc: np.ndarray | None
    rot: np.ndarray | None
    scale: np.ndarray | None
    zeros: tuple = ()  # zero-masked property indices (explicitly zero, must be kept)

    @property
    def root(self) -> bool:
        return (self.rule.L - 5) not in self.present


def _class_rules(pk: Package, votes: dict | None = None) -> dict[int, Rule]:
    votes = {} if votes is None else votes
    for e in pk.exports:
        key = _REPRESENTATIVE.get(e.name)
        if key:
            votes.setdefault(e.class_index, {}).setdefault(key, 0)
            votes[e.class_index][key] += 1
    return {cls: RULES[max(v, key=v.get)] for cls, v in votes.items()}


def learn_rules(pkgs) -> dict[int, Rule]:
    """Class rules voted over several packages. Script class import indices are
    global hashes, so a rule learned in one World Partition cell applies to all."""
    votes: dict = {}
    for pk in pkgs:
        _class_rules(pk, votes)
    return {cls: RULES[max(v, key=v.get)] for cls, v in votes.items()}


# Native data serialised after the properties, per rule (keyed by L): seen on
# every ordinary component of the class across whole base-game levels.
TAILS = {163: bytes(12) + b"\x01\0\0\0" + bytes(4)}


def _find_lod_block(d: bytes, lo: int, end: int, n_trip: int) -> int | None:
    """End of the transform block of a StaticMeshComponent whose native tail
    holds LODData. Exactly one candidate must fit, otherwise give up."""
    if d[end - 8 : end] != b"\x01\0\0\0" + bytes(4):
        return None
    hits = []
    for p in range(lo + 24 * n_trip, min(end - 12, lo + 24 * n_trip + 256)):
        if d[p : p + 8] != bytes(8):
            continue
        cnt = struct.unpack_from("<i", d, p + 8)[0]
        if not 1 <= cnt <= 8:
            continue
        v = np.array(struct.unpack_from(f"<{3 * n_trip}d", d, p - 24 * n_trip))
        if np.all(np.isfinite(v)) and np.all(np.abs(v) < 2e6) and np.abs(v[:3]).max() > 1.0:
            hits.append(p)
    return hits[0] if len(hits) == 1 else None


def _sane(loc, rot, scale) -> bool:
    if loc is not None and not (np.all(np.isfinite(loc)) and np.all(np.abs(loc) < 2e6)):
        return False
    if rot is not None and not (np.all(np.isfinite(rot)) and np.all(np.abs(rot) <= 720)):
        return False
    if scale is not None and not (np.all(np.isfinite(scale)) and np.all(np.abs(scale) < 1e7)):
        return False
    return True


def decode_components(pk: Package, rules: dict | None = None) -> list[Comp]:
    rules = _class_rules(pk) if rules is None else rules
    d = pk.d
    out = []
    for i, e in enumerate(pk.exports):
        rule = rules.get(e.class_index)
        if rule is None:
            continue
        r = Reader(d, e.data_offset)
        try:
            h, zeros = r.header_full()
        except Exception:  # noqa: BLE001
            continue
        L = rule.L
        if not ({L, L + 1, L + 2} & set(h)):
            continue
        after = [x for x in h if x > L + 2]
        front_ok = not any(x < L for x in h)
        if any(x not in rule.extras for x in after) and not front_ok:
            continue
        end = e.data_offset + e.serial_size
        n_trip = sum(x in h for x in (L, L + 1, L + 2))
        cands = []
        if not any(x < L for x in h):
            # values are serialised in index order: with nothing before L the
            # transform block starts right after the header, whatever follows
            cands.append(r.p + 24 * n_trip)
        if d[end - rule.trailer : end] == TAILS.get(L, bytes(8)) and all(x in rule.extras for x in after):
            cands.append(end - rule.trailer - sum(rule.extras[x] for x in after))
        if L == 163 and not after:
            # StaticMeshComponent with native LODData (painted vertex colours):
            # props | u64 0 | i32 count>0 | LOD entries | 01 00 00 00 00 00 00 00
            q = _find_lod_block(d, r.p, end, n_trip)
            if q is not None:
                cands.append(q)
        for block_end in cands:
            block_start = block_end - 24 * n_trip
            if block_start < r.p:
                continue
            p = block_start
            loc = rot = sc = None
            if L in h:
                loc = np.array(struct.unpack_from("<3d", d, p))
                p += 24
            if L + 1 in h:
                rot = np.array(struct.unpack_from("<3d", d, p))
                p += 24
            if L + 2 in h:
                sc = np.array(struct.unpack_from("<3d", d, p))
                p += 24
            # unversioned properties are only written when they differ from the default
            # (exactly: a scale of 0.99999998 is stored, e.g. on scaled tape poles)
            if (
                (loc is not None and not loc.any())
                or (rot is not None and not rot.any())
                or (sc is not None and np.all(sc == 1.0))
            ):
                continue
            if not _sane(loc, rot, sc):
                continue
            out.append(Comp(i, e.outer, rule, h, r.p, block_start, block_end, loc, rot, sc, tuple(zeros)))
            break
    return out


def patched_payload(pk: Package, c: Comp, loc: np.ndarray, rot: np.ndarray, scale: np.ndarray | None = None) -> bytes:
    """Payload of component c with RelativeLocation/Rotation (and optionally
    Scale) set; adds the properties to the header if they were default."""
    e = pk.exports[c.export]
    d = pk.d
    L = c.rule.L
    sc = scale if scale is not None else (c.scale if c.scale is not None else np.ones(3))
    rot = np.where(np.abs(rot) < 1e-9, 0.0, rot)
    # unversioned: a value equal to its default is left out of the header
    vals = [(L, loc, np.any(loc)), (L + 1, rot, np.any(rot)), (L + 2, sc, not np.allclose(sc, 1.0))]
    present = (set(c.present) - {L, L + 1, L + 2}) | {i for i, _, keep in vals if keep}
    block = b"".join(struct.pack("<3d", *v) for _, v, keep in vals if keep)
    zeros = [z for z in c.zeros if z not in present and z not in (L, L + 1, L + 2)]
    head = _hdr_for(sorted(present), zeros)
    return head + d[c.hdr_end : c.block_start] + block + d[c.block_end : e.data_offset + e.serial_size]
