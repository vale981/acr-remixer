# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Duplicate World Partition cells, so a tile can appear twice in a stage.

A cell of a cooked WP level is, in the main level package:
  * WorldPartitionLevelStreaming_<ID> (outer: the world): world soft path and
    the cell package name /Game/Levels/<Level>/_Generated_/<ID> (by name)
  * <ID> (outer: WorldPartitionRuntimeSpatialHash_0): the cell; its first
    property references the streaming object; then a cell GUID
  * WorldPartitionRuntimeCellDataSpatialHash_0 (outer: the cell): position,
    bounds, grid name, debug name
  * a reference in the spatial hash's streaming grid (layer cells)
and the cell package itself, whose content is found by its package name.
A copy gets a new ID, three new exports and a renamed copy of the package.
"""

from __future__ import annotations

import hashlib
import re
import struct

from ..extract.zen import Package
from .rename import rename_package
from .wp_hash import _write_levels, read_grid
from .zen_append import NewExport, append_exports
from .zen_write import rewrite_package

ID_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def new_cell_id(src: str, k: int) -> str:
    """Deterministic 25-character ID like the cooker's; never ends in a digit
    run that FName would split off (the cooker's IDs are plain names too)."""
    h = hashlib.sha256(f"{src}#{k}".encode()).digest()
    s = "".join(ID_CHARS[b % 36] for b in h[:24])
    return s + "Z"


def _find(pk: Package, cell_id: str) -> tuple[int, int, int]:
    ex = pk.exports
    cell = next(i for i, e in enumerate(ex) if e.name == cell_id)
    stream = next(i for i, e in enumerate(ex) if e.name == "WorldPartitionLevelStreaming_" + cell_id)
    data = next(i for i, e in enumerate(ex) if e.outer == cell)
    return stream, cell, data


def duplicate_cells(
    main: bytes,
    level_path: str,
    copies: list[tuple[str, str]],
    grid: bool = True,
    same_package: bool = False,
    new_guid: bool = True,
) -> bytes:
    """Main level with extra cells: copies = [(source cell ID, new ID)].
    Debugging: grid=False leaves the cells out of the streaming grid;
    same_package=True streams the source cell's package (no copy needed)."""
    pk = Package(main)
    n0 = len(pk.exports)
    adds, extra = [], []
    new_refs = {}  # source cell export -> new cell export
    for k, (src, new) in enumerate(copies):
        s, c, dd = _find(pk, src)
        ns, nc, nd = n0 + 3 * k, n0 + 3 * k + 1, n0 + 3 * k + 2
        old_pkg, new_pkg = f"{level_path}/_Generated_/{src}", f"{level_path}/_Generated_/{new}"
        extra += [new_pkg, new]
        remap = {s: ns, c: nc, dd: nd}
        adds += [
            NewExport(s, "WorldPartitionLevelStreaming_" + new, pk.export_bytes(pk.exports[s]), None, remap),
            NewExport(c, new, pk.export_bytes(pk.exports[c]), None, remap),
            NewExport(dd, pk.exports[dd].name, pk.export_bytes(pk.exports[dd]), nc, remap),
        ]
        new_refs.setdefault(c, []).append(nc)
        adds[-3].meta = (old_pkg, new_pkg)
        adds[-2].meta = (s, ns)
    # the spatial hash's grid will reference the new cells: they must exist
    # before it is serialized
    p0 = Package(main)
    hash_i = next(i for i, e in enumerate(p0.exports) if e.name.startswith("WorldPartitionRuntimeSpatialHash"))
    refs = {hash_i: [n0 + 3 * k + 1 for k in range(len(copies))]} if grid else None
    d = append_exports(main, adds, extra, refs)
    pk = Package(d)
    names = pk.names
    payloads = {}
    for k, (src, new) in enumerate(copies):
        ns, nc = n0 + 3 * k, n0 + 3 * k + 1
        old_pkg, new_pkg = adds[3 * k].meta
        # streaming object: cell package name (FName references by index)
        b = bytearray(pk.export_bytes(pk.exports[ns]))
        oi, ni = names.index(old_pkg), names.index(new_pkg)
        hits = 0
        for q in range(0, len(b) - 7):
            if struct.unpack_from("<ii", b, q) == (oi, 0):
                struct.pack_into("<ii", b, q, ni, 0)
                hits += 1
        if hits == 0:
            raise ValueError(f"cell package name not found in the streaming object of {src}")
        if not same_package:
            payloads[ns] = bytes(b)
        # cell: [0] streaming object ref, [2] bool, ([3] data layers), [9] CellGuid,
        # [11] RuntimeCellData ref, then 4 native bytes
        s_old, _ = adds[3 * k + 1].meta
        d_old = _find(Package(main), src)[2]
        b = bytearray(pk.export_bytes(pk.exports[nc]))
        if struct.unpack_from("<i", b, 8)[0] != s_old + 1 or struct.unpack_from("<i", b, len(b) - 8)[0] != d_old + 1:
            raise ValueError(f"cell {src}: layout not understood")
        struct.pack_into("<i", b, 8, ns + 1)
        struct.pack_into("<i", b, len(b) - 8, n0 + 3 * k + 2 + 1)
        if new_guid:
            b[len(b) - 24 : len(b) - 8] = hashlib.md5(new.encode()).digest()
        payloads[nc] = bytes(b)
    # streaming grid: the copy sits in every layer cell its source sits in
    hi = next(i for i, e in enumerate(pk.exports) if e.name.startswith("WorldPartitionRuntimeSpatialHash"))
    hp = pk.export_bytes(pk.exports[hi])
    levels, a, b_, _ = read_grid(hp)
    added = 0
    for lv in levels:
        for refs in lv.layer_cells:
            for r in list(refs):
                for nc in new_refs.get(r - 1, []):
                    refs.append(nc + 1)
                    added += 1
    if added < len(copies):
        raise ValueError("a source cell is not in the streaming grid")
    if grid:
        payloads[hi] = hp[:a] + _write_levels(levels) + hp[b_:]
    return rewrite_package(pk, payloads)


def copy_cell_package(data: bytes, level_path: str, src: str, new: str) -> bytes:
    """The cell package under its new name (paths and the bare ID)."""
    old_pkg = f"{level_path}/_Generated_/{src}"
    new_pkg = f"{level_path}/_Generated_/{new}"
    path_re = re.compile(rf"^{re.escape(old_pkg)}(?=$|[/.:])")

    def ren(n: str) -> str:
        if path_re.match(n):
            return path_re.sub(new_pkg, n)
        return new if n == src else n

    return rename_package(data, ren)
