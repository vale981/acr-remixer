# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Dump the rows of a cooked DataTable: row names, row sizes and the raw bytes
with likely FName references (i32 name index + i32 number) resolved.

    py tools/dt_dump.py acr/Content/Data/Database/Main/TrackSelection/DT_Tracks.uasset [row filter]
"""
from __future__ import annotations

import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from stagegen.extract.iostore import GameFS  # noqa: E402
from stagegen.extract.unversioned import Reader  # noqa: E402
from stagegen.extract.zen import Package  # noqa: E402

PAKS = os.path.join(os.path.dirname(__file__), "..", "..", "acr", "Content", "Paks")


def rows(pk: Package):
    """[(row name, offset, bytes)] using the row names to find row boundaries:
    after the export properties: i32 NumRows, then per row FName + struct."""
    e = pk.exports[0]
    b = pk.export_bytes(e)
    r = Reader(pk.d, e.data_offset)
    r.header_full()
    n_names = len(pk.names)
    # find NumRows: an i32 followed by a valid FName (the first row name)
    for k in range(r.p - e.data_offset, min(len(b) - 12, 64)):
        n, ni, nn = struct.unpack_from("<iii", b, k)
        if 0 < n < 100000 and 0 <= ni < n_names and nn == 0:
            start, count = k + 4, n
            break
    else:
        raise ValueError("row count not found")
    # row boundaries: next row starts with a name index that is a row name;
    # collect candidate row names = names used right at row starts. Rows of one
    # table have similar sizes, so walk greedily: the next FName that parses as
    # a row name and leaves exactly the remaining rows possible.
    offs = [start]
    row_name_idx = {struct.unpack_from("<i", b, start)[0]}
    while len(offs) < count:
        k = offs[-1] + 10
        while k < len(b) - 8:
            ni, nn = struct.unpack_from("<ii", b, k)
            if 0 <= ni < n_names and nn == 0 and b[k + 8: k + 10] == b[offs[0] + 8: offs[0] + 10] \
                    and pk.names[ni] not in ("None",) and not pk.names[ni].startswith("/"):
                offs.append(k)
                break
            k += 1
        else:
            break
    offs.append(len(b))
    out = []
    for a, z in zip(offs, offs[1:]):
        ni = struct.unpack_from("<i", b, a)[0]
        out.append((pk.names[ni], a, b[a:z]))
    return count, out


def show(pk: Package, row: bytes):
    n_names = len(pk.names)
    k = 0
    parts = []
    while k < len(row):
        if k + 8 <= len(row):
            ni, nn = struct.unpack_from("<ii", row, k)
            if 0 <= ni < n_names and 0 <= nn < 8 and (ni > 0 or nn > 0):
                parts.append(f"[{k}] N:{pk.names[ni]}" + (f"#{nn}" if nn else ""))
                k += 8
                continue
        parts.append(f"[{k}] {row[k]:02x}")
        k += 1
    return parts


def main():
    path, flt = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "")
    fs = GameFS(PAKS, include_mods=False)
    pk = Package(fs.read(path))
    count, rs = rows(pk)
    print(f"{path}: {count} rows (found {len(rs)}), {len(pk.names)} names")
    print("sizes:", sorted({len(r) for _, _, r in rs}))
    for name, off, r in rs:
        if flt and flt.lower() not in name.lower():
            continue
        print(f"--- row {name} @ {off}, {len(r)} bytes")
        print("   " + "  ".join(show(pk, r)))


if __name__ == "__main__":
    main()
