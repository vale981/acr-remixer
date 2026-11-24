# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Helpers for UE unversioned property serialization (cooked assets).

An unversioned struct starts with u16 fragments: bits 0-6 skip count, bit 7
'has zero mask', bit 8 'is last', bits 9-15 value count. If any fragment has
zeroes a bit mask follows (one bit per serialized value, 1 = zero/default and
omitted). Values then follow in schema order."""

from __future__ import annotations

import struct


class Reader:
    def __init__(self, data: bytes, pos: int = 0):
        self.d = data
        self.p = pos

    def u8(self):
        v = self.d[self.p]
        self.p += 1
        return v

    def u16(self):
        (v,) = struct.unpack_from("<H", self.d, self.p)
        self.p += 2
        return v

    def i32(self):
        (v,) = struct.unpack_from("<i", self.d, self.p)
        self.p += 4
        return v

    def u32(self):
        (v,) = struct.unpack_from("<I", self.d, self.p)
        self.p += 4
        return v

    def f32(self):
        (v,) = struct.unpack_from("<f", self.d, self.p)
        self.p += 4
        return v

    def f64(self):
        (v,) = struct.unpack_from("<d", self.d, self.p)
        self.p += 8
        return v

    def raw(self, n):
        v = self.d[self.p : self.p + n]
        self.p += n
        return v

    def fstring(self):
        n = self.i32()
        if n == 0:
            return ""
        if n > 0:
            return self.raw(n).rstrip(b"\0").decode("latin-1")
        return self.raw(-2 * n).decode("utf-16-le").rstrip("\0")

    def fname(self, names):
        i, num = struct.unpack_from("<II", self.d, self.p)
        self.p += 8
        s = names[i] if i < len(names) else f"<{i}>"
        return s if num == 0 else f"{s}_{num - 1}"

    def header_full(self) -> tuple[list[int], list[int]]:
        """(present indices, zero-masked indices). Zero-masked properties are
        explicitly zero/false/empty: NOT the same as absent (class default)."""
        start = self.p
        frags = []
        while True:
            f = self.u16()
            frags.append(f)
            if (f >> 8) & 1:
                break
        self.p = start
        present = self.header()
        idx, listed = 0, []
        for f in frags:
            idx += f & 0x7F
            n = f >> 9
            listed.extend(range(idx, idx + n))
            idx += n
        return present, [i for i in listed if i not in present]

    def header(self) -> list[int]:
        """Return the schema indices of the values that follow (zero-masked
        ones removed)."""
        frags = []
        while True:
            f = self.u16()
            frags.append(f)
            if (f >> 8) & 1:
                break
        # The zero mask has one bit per value of fragments flagged HasAnyZeroes
        # only; values in other fragments are always serialised.
        idx, values = 0, []  # (schema index, covered_by_mask)
        for f in frags:
            idx += f & 0x7F
            n = f >> 9
            z = bool((f >> 7) & 1)
            values.extend((s, z) for s in range(idx, idx + n))
            idx += n
        nbits = sum(z for _, z in values)
        if nbits == 0:
            return [s for s, _ in values]
        if nbits <= 8:
            mask = self.u8()
        elif nbits <= 16:
            mask = self.u16()
        else:
            words = (nbits + 31) // 32
            mask = int.from_bytes(self.raw(4 * words), "little")
        present, bit = [], 0
        for s, z in values:
            if z:
                if not (mask >> bit) & 1:
                    present.append(s)
                bit += 1
            else:
                present.append(s)
        return present
