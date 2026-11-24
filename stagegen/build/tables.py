# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Add entries to the game's track tables (cooked DataTables, unversioned):
a generated stage gets its own rows, made from the source stage's rows.

A DataTable export: properties, i32 row count, then per row an FName row name
and the row struct. Rows reference other tables through row handles, stored
as `00 05 <FName table> <FName row>` (18 bytes; arrays of them are an i32
count followed by the elements). Display names are FTexts referencing a string
table (`u32 flags, i8 history 11, FName table, FString key`); copies get a
plain text instead (`u32 flags, i8 history 0, FString namespace, FString key,
FString source`).
"""

from __future__ import annotations

import struct

from ..extract.unversioned import Reader
from ..extract.zen import Package
from .zen_write import rewrite_package


def _fstring(s: str) -> bytes:
    if s.isascii():
        b = s.encode("ascii") + b"\0"
        return struct.pack("<i", len(b)) + b
    b = s.encode("utf-16-le") + b"\0\0"  # negative length = UTF-16 characters
    return struct.pack("<i", -(len(b) // 2)) + b


class Table:
    def __init__(self, pk: Package):
        self.pk = pk
        self.names = list(pk.names)
        e = pk.exports[0]
        b = pk.export_bytes(e)
        r = Reader(pk.d, e.data_offset)
        r.header_full()
        for k in range(r.p - e.data_offset, min(len(b) - 12, 64)):
            n, ni, nn = struct.unpack_from("<iii", b, k)
            if 0 < n < 100000 and 0 <= ni < len(self.names) and nn == 0:
                self.count_off, count = k, n
                break
        else:
            raise ValueError("row count not found")
        self.head = b[: self.count_off]
        self.rows: list[list] = []  # [name, bytes]
        offs = self._row_offsets(b, self.count_off + 4, count)
        for a, z in zip(offs, offs[1:] + [len(b)]):
            self.rows.append([self.names[struct.unpack_from("<i", b, a)[0]], bytearray(b[a:z])])

    @staticmethod
    def _covered(b, k):
        """Number of properties an unversioned header at k covers (fragments of
        u16: skip 7 bits, has-zeroes 1, is-last 1, values 7), or None."""
        total = 0
        for _ in range(8):
            if k + 2 > len(b):
                return None
            f = struct.unpack_from("<H", b, k)[0]
            total += (f & 0x7F) + (f >> 9)
            k += 2
            if (f >> 8) & 1:
                return total
        return None

    def _row_offsets(self, b, start, count):
        """Rows of one table share the struct, so their headers cover the same
        number of properties (the bytes differ when a zero mask is present)."""
        offs = [start]
        sig = self._covered(b, start + 8)
        while len(offs) < count:
            k = offs[-1] + 10
            while k < len(b) - 10:
                ni, nn = struct.unpack_from("<ii", b, k)
                in_handle = b[k - 2 : k] == b"\x00\x05" or (k >= 10 and b[k - 10 : k - 8] == b"\x00\x05")
                if (
                    0 <= ni < len(self.names)
                    and nn == 0
                    and not in_handle
                    and not self.names[ni].startswith("/")
                    and self.names[ni] != "None"
                    and self._covered(b, k + 8) == sig
                ):
                    offs.append(k)
                    break
                k += 1
            else:
                raise ValueError(f"only {len(offs)} of {count} rows found")
        return offs

    # -- names
    def idx(self, name: str) -> int:
        if name not in self.names:
            self.names.append(name)
        return self.names.index(name)

    def fname(self, name: str) -> bytes:
        return struct.pack("<ii", self.idx(name), 0)

    # -- rows
    def row(self, name: str) -> bytearray:
        for n, r in self.rows:
            if n == name:
                return r
        raise KeyError(name)

    def copy_row(self, template: str, new_name: str, rename: dict[str, str]) -> bytearray:
        """Copy a row; FName references (8-byte index+number at any offset that
        decodes to a name in `rename`) are redirected; the row name is set."""
        src = bytes(self.row(template))
        out = bytearray(src)
        struct.pack_into("<ii", out, 0, self.idx(new_name), 0)
        k = 8
        while k <= len(out) - 8:
            ni, nn = struct.unpack_from("<ii", out, k)
            if 0 <= ni < len(self.names) and nn == 0 and self.names[ni] in rename:
                struct.pack_into("<ii", out, k, self.idx(rename[self.names[ni]]), 0)
                k += 8
                continue
            k += 1
        self.rows.append([new_name, out])
        return out

    def replace_texts(self, row: bytearray, texts: list[tuple[str, str]]) -> bytearray:
        """Replace the row's string-table FTexts, in order, by plain texts
        [(key, text)]. Returns the new row bytes (stored in place)."""
        st = {i for i, n in enumerate(self.names) if "/Localization/" in n}
        out = bytearray()
        k = 0
        t = 0
        while k < len(row):
            if t < len(texts) and k + 17 <= len(row) and row[k + 4] == 11:
                ni, nn = struct.unpack_from("<ii", row, k + 5)
                if ni in st:
                    n_key = struct.unpack_from("<i", row, k + 13)[0]
                    if 0 < n_key < 256:
                        key, text = texts[t]
                        out += row[k : k + 4] + bytes([0]) + _fstring("Stagegen") + _fstring(key) + _fstring(text)
                        k += 13 + 4 + n_key
                        t += 1
                        continue
            out.append(row[k])
            k += 1
        row[:] = out
        return row

    def append_handle(self, row: bytearray, table: str, new_row: str, after_row: str):
        """Append a row handle (table, new_row) to the array that contains the
        handle (table, after_row): count +1, element after its last element."""
        el = bytes([0, 5]) + self.fname(table)
        anchor = el + self.fname(after_row)
        a = bytes(row).find(anchor)
        if a < 0:
            raise ValueError(f"handle {table}/{after_row} not found")
        # walk back over earlier elements of the same table to the array start
        start = a
        while (
            start >= 18
            and bytes(row[start - 18 : start - 16]) == bytes([0, 5])
            and struct.unpack_from("<i", row, start - 16)[0] == self.idx(table)
        ):
            start -= 18
        n = struct.unpack_from("<i", row, start - 4)[0]
        if not (0 < n < 1000) or bytes(row[start + 18 * (n - 1) : start + 18 * (n - 1) + 2]) != bytes([0, 5]):
            raise ValueError("array layout not understood")
        struct.pack_into("<i", row, start - 4, n + 1)
        end = start + 18 * n
        row[end:end] = el + self.fname(new_row)

    def handles(self, row: bytes, table: str) -> list[str]:
        """Row names of all handles into `table` in this row, in order."""
        ti = self.names.index(table) if table in self.names else -1
        out = []
        k = 0
        while k <= len(row) - 18:
            if row[k] == 0 and row[k + 1] == 5 and struct.unpack_from("<ii", row, k + 2) == (ti, 0):
                ri = struct.unpack_from("<i", row, k + 10)[0]
                out.append(self.names[ri])
                k += 18
                continue
            k += 1
        return out

    def keep_handles(self, row: bytearray, table: str, keep: set[str]):
        """Drop handles into `table` whose row is not in `keep` (array counts
        adjusted)."""
        ti = self.idx(table)
        k = 0
        while k <= len(row) - 18:
            if row[k] == 0 and row[k + 1] == 5 and struct.unpack_from("<ii", row, k + 2) == (ti, 0):
                ri = struct.unpack_from("<i", row, k + 10)[0]
                if self.names[ri] in keep:
                    k += 18
                    continue
                # array start: walk back over elements of the same table
                s = k
                while (
                    s >= 18
                    and bytes(row[s - 18 : s - 16]) == b"\x00\x05"
                    and struct.unpack_from("<i", row, s - 16)[0] == ti
                ):
                    s -= 18
                n = struct.unpack_from("<i", row, s - 4)[0]
                if not 0 < n < 1000:
                    raise ValueError("handle is not in an array")
                struct.pack_into("<i", row, s - 4, n - 1)
                del row[k : k + 18]
                continue
            k += 1

    def set_float(self, row: bytearray, old: float, new: float, nth: int = 0):
        pat = struct.pack("<f", old)
        k = -1
        for _ in range(nth + 1):
            k = bytes(row).find(pat, k + 1)
            if k < 0:
                raise ValueError(f"float {old} not found")
        struct.pack_into("<f", row, k, new)

    # -- write
    def build(self) -> bytes:
        body = bytearray(self.head) + struct.pack("<i", len(self.rows))
        for _, r in self.rows:
            body += r
        return rewrite_package(self.pk, {0: bytes(body)}, self.names)


DB = "acr/Content/Data/Database/Main/"
T_LEVELS = DB + "Common/DT_Levels.uasset"
T_TRACKS = DB + "TrackSelection/DT_Tracks.uasset"
T_VARIANTS = DB + "TrackSelection/DT_TracksVariants.uasset"
T_LOCATIONS = DB + "TrackSelection/DT_TracksLocations.uasset"


TABLES_PAK = "pakchunk99-Tables_Stagegen_9_P"  # "_Stagegen_": never read back as source data
STAGE_META = ".stagegen.json"  # next to an own-level container: {level, name, display, ...}


def build_tables(fs, stages: list[dict]) -> tuple[dict, list[str]]:
    """Menu rows for several generated stages ({level, name, display}) on
    top of the tables in `fs` (which should include other mods, e.g. WRCG,
    but not stagegen containers). Returns ({table path: package bytes}, notes)."""
    tabs = {p: Table(Package(fs.read(p))) for p in (T_LEVELS, T_TRACKS, T_VARIANTS, T_LOCATIONS)}
    notes = []
    for s in stages:
        notes += _add_stage(
            tabs[T_LEVELS], tabs[T_TRACKS], tabs[T_VARIANTS], tabs[T_LOCATIONS], s["level"], s["name"], s["display"]
        )
    return {p: t.build() for p, t in tabs.items()}, notes


def stage_entries(fs, level: str, new: str, display: str, length_km: float = 0) -> tuple[dict, list[str]]:
    """Rows for one generated stage (see build_tables)."""
    return build_tables(fs, [{"level": level, "name": new, "display": display}])


def _add_stage(lv: Table, tr: Table, va: Table, lo: Table, level: str, new: str, display: str) -> list[str]:
    """Rows for a generated stage /Game/Levels/<new> made from <level>:
    DT_Levels, DT_Tracks, DT_TracksVariants (full forward + reverse) and the
    location's track list."""
    notes = []
    lvl_path = f"/Game/Levels/{level}"
    lv_key = (
        next((n for n, r in lv.rows if struct.unpack_from("<i", r, 10)[0] == lv.names.index(lvl_path)), None)
        if lvl_path in lv.names
        else None
    )
    if lv_key is None:
        raise ValueError(f"no DT_Levels row for {lvl_path}")
    lv.copy_row(lv_key, new, {lvl_path: f"/Game/Levels/{new}", level: new})

    tr_key = next(n for n, r in tr.rows if lv_key in tr.handles(r, "Levels"))
    src_row = tr.row(tr_key)
    variants = tr.handles(src_row, "TracksVariants")
    fwd = next(v for v in variants if v.endswith("FullForward"))
    rev = next(v for v in variants if v.endswith("FullReverse"))
    loc_key = tr.handles(src_row, "TracksLocations")[0]

    for src_v, suffix, label in ((fwd, "FullForward", "Forward"), (rev, "FullReverse", "Reverse")):
        row = va.copy_row(src_v, new + suffix, {tr_key: new})
        va.replace_texts(
            row,
            [
                (f"STAGEGEN_{new}_{suffix}_SHORT", f"{display} ({label})"),
                (f"STAGEGEN_{new}_{suffix}", f"{display} ({label})"),
            ],
        )
        notes.append(f"variant {new + suffix} from {src_v}")

    row = tr.copy_row(tr_key, new, {lv_key: new, fwd: new + "FullForward", rev: new + "FullReverse"})
    tr.keep_handles(row, "TracksVariants", {new + "FullForward", new + "FullReverse"})
    tr.replace_texts(row, [(f"STAGEGEN_{new}", display)])

    lo.append_handle(lo.row(loc_key), "Tracks", new, tr_key)
    notes.append(f"track {new} ({display}) under location {loc_key}, level key {new}, from {tr_key}")
    return notes


def write_tables_container(paks_dir: str) -> list[str]:
    """(Re)write the shared menu-table container in ~mods from every
    installed own-level stage (its STAGE_META file); removed if none."""
    import glob
    import json
    import os
    from ..extract.iostore import GameFS
    from . import container_header as CH
    from .cityhash import name_hash
    from .iostore_write import ChunkIn, container_header_chunk_id, write_container

    mods = os.path.join(paks_dir, "~mods")
    base = os.path.join(mods, TABLES_PAK)
    for ext in (".utoc", ".ucas", ".pak"):
        if os.path.exists(base + ext):
            os.remove(base + ext)
    stages = []
    for p in sorted(glob.glob(os.path.join(mods, "*" + STAGE_META))):
        if os.path.exists(p[: -len(STAGE_META)] + ".utoc"):
            with open(p, encoding="utf-8") as f:
                stages.append(json.load(f))
    if not stages:
        return ["no generated stages installed: menu tables removed"]
    fs = GameFS(paks_dir, include_mods=True)
    tbl, notes = build_tables(fs, stages)
    cid = name_hash(TABLES_PAK)
    chunks, entries, headers = [], {}, {}
    for path, data in tbl.items():
        c, i = fs.index[path]
        if id(c) not in headers:
            h_i = next(k for k, x in enumerate(c.chunk_ids) if x[11] == 6)
            headers[id(c)] = CH.parse(c.read_entry(h_i))
        pid = struct.unpack_from("<Q", c.chunk_ids[i], 0)[0]
        orig = headers[id(c)].entries.get(pid, CH.StoreEntry())
        entries[pid] = CH.StoreEntry(
            [CH.package_id(n) for n in Package(data).imported_package_names()], orig.shader_maps
        )
        chunks.append(ChunkIn(c.chunk_ids[i], data, path))
    chunks.append(ChunkIn(container_header_chunk_id(cid), CH.build(CH.ContainerHeader(cid, entries))))
    with open(os.path.join(os.path.dirname(__file__), "empty_mod.pak"), "rb") as f:
        write_container(base, cid, chunks, f.read())
    return notes
