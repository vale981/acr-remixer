# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Rename things inside a cooked zen package (UE 5.6), e.g. to duplicate a
level under a new name:

- name map: every name passes through rename(); count and order stay, so all
  references by name index (export names, FName properties, soft object
  paths) follow; name hashes are recomputed
- imported package names (name batch + u32 number each, near the end of the
  header): renamed as well; the header offsets after them shift
- public export hashes: CityHash64 of the lowercase UTF-16 path of the export
  inside the package (outer chain joined by '/', package name not included);
  recomputed for exports whose names changed

Not handled yet: imported public export hashes for exports of *other* renamed
packages (a level's cells import nothing from the main package by hash).
"""

from __future__ import annotations

import re
import struct
from typing import Callable

from ..extract.zen import Package, read_name_batch
from .cityhash import cityhash64
from .zen_write import SUMMARY, _name_batch, _name_batch_end


def export_hash(path_in_package: str) -> int:
    return cityhash64(path_in_package.lower().encode("utf-16-le"))


def _export_rel(pk: Package, names: list[str], i: int) -> str:
    parts = []
    while 0 <= i < len(pk.exports):
        e = pk.exports[i]
        q = pk.export_map_off + 72 * i
        ni, nn = struct.unpack_from("<II", pk.d, q + 16)
        base = names[ni & 0x3FFFFFFF]
        parts.append(base if nn == 0 else f"{base}_{nn - 1}")
        i = e.outer
    return "/".join(reversed(parts))


def rename_package(d: bytes, rename: Callable[[str], str]) -> bytes:
    pk = Package(d)
    summ = list(struct.unpack_from(SUMMARY, d, 0))
    header_size = summ[1]
    nb_start = struct.calcsize(SUMMARY)
    nb_end, hv = _name_batch_end(d, nb_start)
    new_names = [rename(n) for n in pk.names]
    new_nb = _name_batch(new_names, hv or 0xC1640000, nb_start)

    # imported package names: name batch + numbers
    ipn_off = pk.imported_pkg_names_off
    ipn_names, p = read_name_batch(d, ipn_off)
    ipn_end = p + 4 * len(ipn_names)
    nums = struct.unpack_from(f"<{len(ipn_names)}I", d, p)
    _, ipn_hv = _name_batch_end(d, ipn_off)
    d1 = len(new_nb) - (nb_end - nb_start)
    new_ipn = _name_batch([rename(n) for n in ipn_names], ipn_hv or hv or 0xC1640000, ipn_off + d1) + struct.pack(
        f"<{len(nums)}I", *nums
    )
    d2 = len(new_ipn) - (ipn_end - ipn_off)
    for k in range(6, 15):  # offset fields of the summary
        if summ[k] <= 0 and k >= 13:
            continue  # absent cell maps
        summ[k] += d1 + (d2 if summ[k] > ipn_off else 0)
    summ[1] = header_size + d1 + d2

    mid = bytearray(d[nb_end:ipn_off])  # imported hashes, import/export maps, bundles, deps
    tail = d[ipn_end:header_size]  # cell import/export maps
    # public export hashes of renamed exports (export map lives in 'mid')
    em_rel = pk.export_map_off - nb_end
    for i, e in enumerate(pk.exports):
        if not e.public_hash:
            continue
        old_rel = _export_rel(pk, pk.names, i)
        new_rel = _export_rel(pk, new_names, i)
        if new_rel != old_rel:
            struct.pack_into("<Q", mid, em_rel + 72 * i + 56, export_hash(new_rel))
    return struct.pack(SUMMARY, *summ) + new_nb + bytes(mid) + new_ipn + tail + d[header_size:]


def level_renamer(old: str, new: str) -> Callable[[str], str]:
    """Names of a level OLD -> NEW: the level's package paths
    (/Game/Levels/OLD, its _Generated_ cells, object paths below it) and the
    level name as a standalone word (OLD, OLD_C, Default__OLD_C,
    ExecuteUbergraph_OLD) in names that are not paths to other assets."""
    path_re = re.compile(rf"^/Game/Levels/{re.escape(old)}(?=$|[/.:])")
    word_re = re.compile(rf"(?<![A-Za-z0-9]){re.escape(old)}(?![A-Za-z0-9])")

    def rename(n: str) -> str:
        if n.startswith("/"):
            if not path_re.match(n):
                return n  # other assets keep their paths
            n = path_re.sub(f"/Game/Levels/{new}", n)
        return word_re.sub(new, n)

    return rename
