# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Append exports to a cooked zen package (UE 5.6), e.g. to give a World
Partition level an extra (duplicated) cell.

Header sections after the export map shift: export bundle entries (u32 local
export index, u32 command 0 = create / 1 = serialize), dependency bundle
headers (per export: i32 first entry, u32 counts[4]) and entries (i32 package
index, export i = i + 1), imported package names, cell maps. Payloads are
found by their serial offset, so new ones go at the end of the data (before
the 4-byte package tag).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

from ..extract.zen import Package
from .zen_write import SUMMARY, rewrite_package

EXPORT_SIZE = 72
DEP_HDR = "<iIIII"


@dataclass
class NewExport:
    template: int  # existing export to copy the map entry and dependencies from
    name: str  # object name (may end in _N: stored as FName number)
    payload: bytes
    outer: int | None = None  # export index (incl. other new ones), None = template's
    dep_remap: dict = field(default_factory=dict)  # export index -> export index, for dependencies


def _fname(names: list[str], s: str) -> tuple[int, int]:
    """Name map index + number (UE splits a trailing _N, N without leading zero)."""
    head, _, tail = s.rpartition("_")
    if head and tail.isdigit() and (tail == "0" or not tail.startswith("0")):
        return names.index(head), int(tail) + 1
    return names.index(s), 0


def needed_names(s: str) -> str:
    head, _, tail = s.rpartition("_")
    return head if head and tail.isdigit() and (tail == "0" or not tail.startswith("0")) else s


def append_exports(
    d: bytes, adds: list[NewExport], extra_names: list[str] = (), referenced_by: dict | None = None
) -> bytes:
    """referenced_by: {existing export: [new export, ...]} - existing exports
    whose payload references new ones; those must be created before the
    existing export is serialized (else the reference resolves to null), so
    they get a create-before-serialize dependency, and all new exports are
    created right after the last create of an existing export they depend on."""
    pk = Package(d)
    names = list(pk.names)
    for n in list(extra_names) + [needed_names(a.name) for a in adds]:
        if n not in names:
            names.append(n)
    if names != pk.names:
        d = rewrite_package(pk, {}, names)
        pk = Package(d)
    n0 = len(pk.exports)
    n = len(adds)
    summ = list(struct.unpack_from(SUMMARY, d, 0))
    hs = pk.header_size

    em_end = pk.export_bundle_off
    be_end = pk.dep_hdr_off
    dh_end = pk.dep_entries_off
    de_end = pk.imported_pkg_names_off

    # dependency entries of the new exports: the template's, remapped
    n_dep_entries = (de_end - dh_end) // 4
    new_hdrs, new_entries = bytearray(), bytearray()
    new_deps = []
    for a in adds:
        first, *cnt = struct.unpack_from(DEP_HDR, d, pk.dep_hdr_off + 20 * a.template)
        k = sum(cnt)
        deps = struct.unpack_from(f"<{k}i", d, pk.dep_entries_off + 4 * first) if k else ()
        deps = [(a.dep_remap.get(x - 1, x - 1) + 1) if x > 0 else x for x in deps]
        new_deps.append(deps)
        new_hdrs += struct.pack(DEP_HDR, n_dep_entries + len(new_entries) // 4, *cnt)
        new_entries += struct.pack(f"<{k}i", *deps)
    # existing exports that reference new ones: their dependency list moves
    # to the end with create-before-serialize entries (group 2) added
    hdr_patch = {}
    for ex_i, refs in (referenced_by or {}).items():
        first, *cnt = struct.unpack_from(DEP_HDR, d, pk.dep_hdr_off + 20 * ex_i)
        k = sum(cnt)
        deps = list(struct.unpack_from(f"<{k}i", d, pk.dep_entries_off + 4 * first)) if k else []
        g2 = cnt[0] + cnt[1] + cnt[2]  # end of group 2 (create before serialize)
        deps[g2:g2] = [r + 1 for r in refs]
        cnt[2] += len(refs)
        hdr_patch[ex_i] = struct.pack(DEP_HDR, n_dep_entries + len(new_entries) // 4, *cnt)
        new_entries += struct.pack(f"<{len(deps)}i", *deps)

    data_end = max(e.serial_offset + e.serial_size for e in pk.exports)
    tag = d[hs + data_end :]
    body = bytearray(d[hs : hs + data_end])

    new_map = bytearray()
    for j, a in enumerate(adds):
        q = pk.export_map_off + EXPORT_SIZE * a.template
        ent = bytearray(d[q : q + EXPORT_SIZE])
        ni, nn = _fname(names, a.name)
        struct.pack_into("<QQII", ent, 0, len(body), len(a.payload), ni, nn)
        if a.outer is not None:
            struct.pack_into("<Q", ent, 24, a.outer)
        struct.pack_into("<Q", ent, 56, 0)  # not public
        new_map += ent
        body += a.payload

    # creates go right after the last create of an existing export the new ones
    # depend on (outers, references); serializes at the end
    be_start = pk.export_bundle_off
    entries = [struct.unpack_from("<II", d, be_start + 8 * k) for k in range((be_end - be_start) // 8)]
    needed = {x - 1 for deps in new_deps for x in deps if 0 < x <= n0}
    for a in adds:
        o = (
            a.outer
            if a.outer is not None
            else struct.unpack_from("<Q", d, pk.export_map_off + EXPORT_SIZE * a.template + 24)[0]
        )
        if o < n0:
            needed.add(o)
    pos = 1 + max((k for k, (i, c) in enumerate(entries) if c == 0 and i in needed), default=-1)
    for ex_i in referenced_by or {}:
        ser = next(k for k, (i, c) in enumerate(entries) if c == 1 and i == ex_i)
        if ser < pos:
            raise ValueError(f"export {ex_i} is serialized before the new exports' dependencies are created")
    entries[pos:pos] = [(n0 + j, 0) for j in range(n)]
    entries += [(n0 + j, 1) for j in range(n)]
    bundle_all = b"".join(struct.pack("<II", i, c) for i, c in entries)

    hdrs_old = bytearray(d[be_end:dh_end])
    for ex_i, h in hdr_patch.items():
        hdrs_old[20 * ex_i : 20 * ex_i + 20] = h

    s1, s2, s3, s4 = len(new_map), 16 * n, len(new_hdrs), len(new_entries)
    out_hdr = (
        d[:em_end] + new_map + bundle_all + bytes(hdrs_old) + new_hdrs + d[dh_end:de_end] + new_entries + d[de_end:hs]
    )

    def shift(off: int) -> int:
        k = 0
        if off >= em_end:
            k += s1
        if off >= be_end:
            k += s2
        if off >= dh_end:
            k += s3
        if off >= de_end:
            k += s4
        return off + k

    # summary offsets: [6] imported hashes ... [14] cell exports
    for i in range(6, 15):
        if summ[i] > 0:
            # a section that *starts* at a boundary belongs after the inserted bytes,
            # except the sections being extended (their start stays)
            summ[i] = summ[i] if summ[i] < em_end else shift(summ[i])
    summ[9] = em_end + s1  # export bundle entries start after the grown export map
    summ[10] = be_end + s1 + s2
    summ[11] = dh_end + s1 + s2 + s3
    summ[1] = len(out_hdr)
    out = bytearray(out_hdr)
    struct.pack_into(SUMMARY, out, 0, *summ)
    return bytes(out) + bytes(body) + tag
