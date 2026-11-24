# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Rewrite cooked zen packages: replace export payloads and (optionally) append
names to the name map. Everything else in the header is carried over."""

from __future__ import annotations

import struct

from ..extract.zen import Package
from .cityhash import name_hash

SUMMARY = "<IIIIIIiiiiiiiii"  # UE 5.5+/5.6 FZenPackageSummary (60 bytes)
OFFSET_FIELDS = range(6, 15)  # offsets that point past the name batch


def _name_batch(names: list[str], hash_version: int, base: int = 0) -> bytes:
    """Zen name batch. Pure 7-bit names narrow, others UTF-16 (unaligned).
    `base` (the package offset it is written at) is currently unused."""
    enc = []
    for n in names:
        if n.isascii():  # UE writes only 7-bit names narrow
            enc.append((False, n.encode("ascii")))
        else:
            enc.append((True, n.encode("utf-16-le")))
    if not names:
        return struct.pack("<I", 0)  # an empty batch is just the count
    nbytes = sum(len(b) for _, b in enc)
    out = bytearray(struct.pack("<II", len(names), nbytes))
    out += struct.pack("<Q", hash_version)
    for n in names:
        out += struct.pack("<Q", name_hash(n))
    for wide, b in enc:
        L = len(b) // 2 if wide else len(b)
        out += struct.pack(">H", L | (0x8000 if wide else 0))
    for wide, b in enc:
        out += b  # UTF-16 strings are not aligned
    return bytes(out)


def _name_batch_end(d: bytes, p: int) -> tuple[int, int]:
    """Return (end offset, hash_version) of the name batch starting at p."""
    (n,) = struct.unpack_from("<I", d, p)
    if n == 0:
        return p + 4, 0  # an empty batch is just the count
    (hv,) = struct.unpack_from("<Q", d, p + 8)
    q = p + 16 + 8 * n
    lens = [struct.unpack_from(">H", d, q + 2 * i)[0] for i in range(n)]
    q += 2 * n
    for L in lens:
        if L & 0x8000:
            q += 2 * (L & 0x7FFF)  # UTF-16, not aligned
        else:
            q += L
    return q, hv


def rewrite_package(pk: Package, payloads: dict[int, bytes], names: list[str] | None = None) -> bytes:
    d = pk.d
    summ = list(struct.unpack_from(SUMMARY, d, 0))
    header_size = summ[1]
    nb_start = struct.calcsize(SUMMARY)
    nb_end, hv = _name_batch_end(d, nb_start)
    if names is None:
        new_nb = d[nb_start:nb_end]
    else:
        if names[: len(pk.names)] != pk.names:
            raise ValueError("names may only be appended")
        new_nb = _name_batch(names, hv or 0xC1640000, nb_start)
    delta = len(new_nb) - (nb_end - nb_start)
    for i in OFFSET_FIELDS:
        summ[i] += delta
    summ[1] = header_size + delta

    rest = bytearray(d[nb_end:header_size])  # import/export maps etc.
    export_map_rel = pk.export_map_off - nb_end

    # Payloads in serial order; rebuild offsets.
    order = sorted(range(len(pk.exports)), key=lambda i: pk.exports[i].serial_offset)
    data_start = header_size
    last_end = max(e.serial_offset + e.serial_size for e in pk.exports)
    trailer = d[data_start + last_end :]
    body = bytearray()
    for i in order:
        e = pk.exports[i]
        payload = payloads.get(i)
        if payload is None:
            payload = d[e.data_offset : e.data_offset + e.serial_size]
        off = len(body)
        body += payload
        q = export_map_rel + 72 * i
        struct.pack_into("<QQ", rest, q, off, len(payload))
    return struct.pack(SUMMARY, *summ) + new_nb + bytes(rest) + bytes(body) + trailer
