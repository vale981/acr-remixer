# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Read DT_Pacenote* DataTables (row struct /Script/acr.PacenoteSplineEntry).

Row layout (unversioned):
    [0] float  distance along the centre spline (m)
    [1] float  length / extent of the call (m)
    [2] struct { [0] TArray<FName> tokens }   e.g. [Left3, KeepIn]
    [3] uint8/bool flag
    [4] FString  audio/line id (e.g. ES2_SS3_PN_41[0])
    [5] (unknown, rarely set; 4 bytes)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .unversioned import Reader
from .zen import Package


@dataclass
class PacenoteRow:
    row: str
    distance: float = 0.0
    length: float = 0.0
    tokens: list[str] = field(default_factory=list)
    flag: int = 0
    line_id: str = ""
    extra: int = 0


def read_pacenote_table(pk: Package) -> list[PacenoteRow]:
    e = pk.exports[0]
    r = Reader(pk.d, e.data_offset)
    obj = r.header()            # UDataTable properties (RowStruct)
    for _ in obj:
        r.i32()
    r.i32()                     # no-guid marker
    n = r.i32()
    rows = []
    for _ in range(n):
        row = PacenoteRow(r.fname(pk.names))
        for f in r.header():
            if f == 0:
                row.distance = r.f32()
            elif f == 1:
                row.length = r.f32()
            elif f == 2:
                for g in r.header():
                    if g != 0:
                        raise ValueError(f"unexpected token struct field {g}")
                    row.tokens = [r.fname(pk.names) for _ in range(r.u32())]
            elif f == 3:
                row.flag = r.u8()
            elif f == 4:
                row.line_id = r.fstring()
            elif f == 5:
                row.extra = r.i32()
            else:
                raise ValueError(f"unexpected pacenote field {f} in {row.row}")
        rows.append(row)
    return rows
