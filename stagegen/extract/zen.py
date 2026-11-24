# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Zen (IoStore) cooked package header parsing for UE 5.6: name map, imports,
export map. Property data is unversioned, so callers decode export payloads
with type-specific readers."""

from __future__ import annotations

import struct
from dataclasses import dataclass


@dataclass
class Export:
    serial_offset: int
    serial_size: int
    name: str
    outer: int
    class_index: int
    super_index: int
    template_index: int
    public_hash: int
    flags: int
    data_offset: int = 0  # absolute offset into the package bytes


def read_name_batch(d: bytes, p: int) -> tuple[list[str], int]:
    (n,) = struct.unpack_from("<I", d, p)
    if n == 0:
        return [], p + 4  # an empty batch is just the count
    p += 8
    p += 8  # hash version
    p += 8 * n  # hashes
    lens = [struct.unpack_from(">H", d, p + 2 * i)[0] for i in range(n)]
    p += 2 * n
    names = []
    for L in lens:
        if L & 0x8000:
            L &= 0x7FFF  # UTF-16, not aligned
            names.append(d[p : p + 2 * L].decode("utf-16-le"))
            p += 2 * L
        else:
            names.append(d[p : p + L].decode("latin-1"))
            p += L
    return names, p


class Package:
    def __init__(self, data: bytes):
        self.d = d = data
        (
            has_ver,
            self.header_size,
            name_idx,
            name_num,
            self.pkg_flags,
            self.cooked_header_size,
            self.imported_hashes_off,
            self.import_map_off,
            self.export_map_off,
            self.export_bundle_off,
            self.dep_hdr_off,
            self.dep_entries_off,
            self.imported_pkg_names_off,
            self.cell_import_off,
            self.cell_export_off,
        ) = struct.unpack_from("<IIIIIIiiiiiiiii", d, 0)
        p = 15 * 4  # UE 5.5+ summary includes cell import/export map offsets
        if has_ver:
            raise NotImplementedError("versioned zen package")
        self.names, p = read_name_batch(d, p)
        self.name = self.names[name_idx] if name_idx < len(self.names) else "?"
        n_imp = (self.export_map_off - self.import_map_off) // 8
        self.imports = list(struct.unpack_from(f"<{n_imp}Q", d, self.import_map_off))
        n_exp = (self.export_bundle_off - self.export_map_off) // 72
        self.exports: list[Export] = []
        for i in range(n_exp):
            q = self.export_map_off + 72 * i
            so, ss, ni, nn, outer, cls, sup, tmpl, h, fl = struct.unpack_from("<QQIIQQQQQI", d, q)
            self.exports.append(Export(so, ss, self.mapped(ni, nn), outer, cls, sup, tmpl, h, fl))
        # Export payloads follow the header; serial offsets are relative to its end.
        for e in self.exports:
            e.data_offset = self.header_size + e.serial_offset

    def mapped(self, idx: int, num: int) -> str:
        idx &= 0x3FFFFFFF
        s = self.names[idx] if idx < len(self.names) else f"<name{idx}>"
        return s if num == 0 else f"{s}_{num - 1}"

    def export_bytes(self, e: Export) -> bytes:
        return self.d[e.data_offset : e.data_offset + e.serial_size]

    def imported_package_names(self) -> list[str]:
        """Full names of the imported packages (name batch + u32 number per
        name; number k > 0 means the suffix _{k-1}). Their package ids, in
        this order, are the package's imports in the container header."""
        names, p = read_name_batch(self.d, self.imported_pkg_names_off)
        nums = struct.unpack_from(f"<{len(names)}I", self.d, p)
        return [n if k == 0 else f"{n}_{k - 1}" for n, k in zip(names, nums)]
