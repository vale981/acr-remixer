# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""IoStore container header (FIoContainerHeader, version 5 as written by
UE 5.6): which packages a container holds and which packages each imports.

Layout:
    u32 'nCoI', u32 version (5), u64 container id
    TArray<u64> package ids (ascending)
    TArray<u8>  store entries: per package 16 bytes (imported packages
                count + offset from the field, shader map hashes count +
                offset), then per package its imported package ids (u64)
                followed by its shader map hashes (20 bytes each)

A package's imports are the package ids of its imported package names
(Package.imported_package_names(), name numbers included), in that order;
verified on 121 base-game level packages.
    TArray<u64> optional segment package ids, TArray<u8> their store entries
    name batch (redirect names), TArray localized packages, TArray package redirects
    soft package references: i64 offset, i64 size, payload

New packages need an entry here, or the game cannot find them.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

from .cityhash import cityhash64

MAGIC = b"nCoI"


def package_id(package_name: str) -> int:
    """FPackageId of a package name such as /Game/Levels/WalesS4HafrenSouth."""
    return cityhash64(package_name.lower().encode("utf-16-le"))


@dataclass
class StoreEntry:
    imports: list[int] = field(default_factory=list)
    shader_maps: list[bytes] = field(default_factory=list)      # 20-byte hashes


@dataclass
class ContainerHeader:
    container_id: int
    entries: dict[int, StoreEntry]          # package id -> entry
    version: int = 5


def _entries(ids, blob):
    out = {}
    for i, pid in enumerate(ids):
        n_imp, off_imp, n_sh, off_sh = struct.unpack_from("<IiIi", blob, 16 * i)
        imps = list(struct.unpack_from(f"<{n_imp}Q", blob, 16 * i + off_imp)) if n_imp else []
        shs = [blob[16 * i + 8 + off_sh + 20 * k: 16 * i + 8 + off_sh + 20 * (k + 1)] for k in range(n_sh)]
        out[pid] = StoreEntry(imps, shs)
    return out


def parse(d: bytes) -> ContainerHeader:
    if d[:4] != MAGIC:
        raise ValueError("not a container header")
    ver, cid = struct.unpack_from("<IQ", d, 4)
    p = 16
    n = struct.unpack_from("<i", d, p)[0]
    ids = struct.unpack_from(f"<{n}Q", d, p + 4)
    p += 4 + 8 * n
    nb = struct.unpack_from("<i", d, p)[0]
    blob = d[p + 4: p + 4 + nb]
    p += 4 + nb
    if struct.unpack_from("<i", d, p)[0] or struct.unpack_from("<i", d, p + 4)[0]:
        raise NotImplementedError("optional segment packages")
    p += 8
    if struct.unpack_from("<I", d, p)[0]:
        raise NotImplementedError("redirect names")
    p += 4
    if struct.unpack_from("<i", d, p)[0] or struct.unpack_from("<i", d, p + 4)[0]:
        raise NotImplementedError("localized packages / package redirects")
    return ContainerHeader(cid, _entries(ids, blob), ver)


def build(h: ContainerHeader) -> bytes:
    ids = sorted(h.entries)
    n = len(ids)
    head = bytearray(16 * n)
    data = bytearray()
    for i, pid in enumerate(ids):                 # per package: its imports, then its shader maps
        e = h.entries[pid]
        if e.imports:
            struct.pack_into("<Ii", head, 16 * i, len(e.imports), 16 * n + len(data) - 16 * i)
            data += struct.pack(f"<{len(e.imports)}Q", *e.imports)
        if e.shader_maps:
            struct.pack_into("<Ii", head, 16 * i + 8, len(e.shader_maps), 16 * n + len(data) - (16 * i + 8))
            data += b"".join(e.shader_maps)
    blob = bytes(head + data)
    out = bytearray(MAGIC + struct.pack("<IQ", h.version, h.container_id))
    out += struct.pack("<i", n) + struct.pack(f"<{n}Q", *ids)
    out += struct.pack("<i", len(blob)) + blob
    out += struct.pack("<ii", 0, 0)            # optional segment: ids, store entries
    out += struct.pack("<I", 0)                # redirect names (empty name batch)
    out += struct.pack("<ii", 0, 0)            # localized packages, package redirects
    payload = struct.pack("<i", 0)             # soft package references: none
    out += struct.pack("<qq", len(out) + 16, len(payload)) + payload
    return bytes(out)
