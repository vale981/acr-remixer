# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Share codes: the built stage itself, not the settings that found it.

    R2:WalesS4HafrenSouth:0-318,1563-2805,1563-2805r,5322-5519:FGL:a3f1:K7
    |  |                  |                                    |   |    checksum
    |  |                  |                                    |   source fingerprint
    |  |                  |                                    foliage (Forest, Grass, foLiage tool; - = off)
    |  |                  pieces: metres along the source route, r = driven backwards
    |  source level
    format version

A seed only says how to search; any change to the search gives another stage.
A code lists the pieces by distance along the original route instead, so it
rebuilds the same stage with later versions, and is snapped to the nearest
tile cut when loaded (robust to cut numbering). The fingerprint notices when a
game update changed the source stage; the checksum catches typos.
"""

from __future__ import annotations

import hashlib
import zlib
from dataclasses import dataclass

import numpy as np

VERSION = "R2"
_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # base32 without 0/O, 1/I
SNAP_TOLERANCE_M = 10.0


class ShareCodeError(ValueError):
    """A code that can't be used (with a message for the player)."""


class LegacyShareCode(ShareCodeError):
    """A code from v0.1.x (Stage/Length/Seed/Foliage...)."""


@dataclass(frozen=True)
class ShareCode:
    level: str
    pieces: tuple[tuple[int, int, bool], ...]  # (start m, end m, reversed), start < end
    foliage: str  # e.g. "FGL", "F-L"
    fingerprint: str

    @property
    def forest(self) -> bool:
        return self.foliage[0] == "F"

    @property
    def grass(self) -> bool:
        return self.foliage[1] == "G"

    @property
    def foliage_tool(self) -> bool:
        return self.foliage[2] == "L"


def fingerprint(points: np.ndarray) -> str:
    """4 hex digits from the source centreline (decimetre precision)."""
    q = np.round(np.asarray(points, dtype=np.float64)[:, :3] * 10.0).astype(np.int64)
    return hashlib.sha1(q.tobytes()).hexdigest()[:4]


def _checksum(body: str) -> str:
    c = zlib.crc32(body.encode("ascii"))
    return _ALPHABET[c & 31] + _ALPHABET[(c >> 5) & 31]


def foliage_flags(forest: bool, grass: bool, foliage_tool: bool) -> str:
    return ("F" if forest else "-") + ("G" if grass else "-") + ("L" if foliage_tool else "-")


def encode(level: str, pieces, foliage: str, fp: str) -> str:
    """pieces: iterable of (start m, end m, reversed) along the source route."""
    toks = []
    for a, b, rev in pieces:
        a, b = int(round(a)), int(round(b))
        if not 0 <= a < b:
            raise ValueError(f"bad piece {a}-{b}")
        toks.append(f"{a}-{b}{'r' if rev else ''}")
    if not toks:
        raise ValueError("a stage needs at least one piece")
    body = f"{VERSION}:{level}:{','.join(toks)}:{foliage}:{fp}"
    return f"{body}:{_checksum(body)}"


def decode(code: str) -> ShareCode:
    code = "".join(code.split())  # spaces/line breaks from pasting
    if not code.upper().startswith("R"):
        if "/" in code:
            raise LegacyShareCode(
                "this is a code from ACR Remixer 0.1.x; those describe the search settings, "
                "and newer versions find a different stage with them"
            )
        raise ShareCodeError("not a share code (they start with R2:)")
    parts = code.split(":")
    if parts[0].upper() != VERSION:
        raise ShareCodeError(f"share code version {parts[0]} is not supported by this version (expects {VERSION})")
    if len(parts) != 6:
        raise ShareCodeError("incomplete share code (expected R2:stage:pieces:foliage:fingerprint:check)")
    _, level, toks, foliage, fp, check = parts
    if _checksum(":".join(parts[:5])) != check.upper():
        raise ShareCodeError("the share code has a typo (checksum doesn't match)")
    if len(foliage) != 3 or any(c not in ok for c, ok in zip(foliage, ("F-", "G-", "L-"))):
        raise ShareCodeError(f"bad foliage flags {foliage!r}")
    pieces = []
    for tok in toks.split(","):
        rev = tok.endswith("r")
        a, sep, b = tok.rstrip("r").partition("-")
        if not (sep and a.isdigit() and b.isdigit() and int(a) < int(b)):
            raise ShareCodeError(f"bad piece {tok!r}")
        pieces.append((int(a), int(b), rev))
    return ShareCode(level, tuple(pieces), foliage, fp.lower())


def to_sequence(pieces, cut_stations, tolerance: float = SNAP_TOLERANCE_M) -> list[tuple[int, int, bool]]:
    """Snap pieces (metres) to the nearest cut indices: [(a, b, reversed)]."""
    s = np.asarray(cut_stations, dtype=np.float64)
    out = []
    for a_m, b_m, rev in pieces:
        idx = []
        for m in (a_m, b_m):
            i = int(np.argmin(np.abs(s - m)))
            if abs(s[i] - m) > tolerance:
                raise ShareCodeError(
                    f"no piece boundary near {m} m on this stage (nearest {s[i]:.0f} m); "
                    "the stage data or version differs from where the code was made"
                )
            idx.append(i)
        if idx[0] >= idx[1]:
            raise ShareCodeError(f"piece {a_m}-{b_m} m collapses on this stage")
        out.append((idx[0], idx[1], rev))
    return out


def from_states(level: str, states, cuts, foliage: str, fp: str) -> str:
    """The code of a built layout (states in driving order, as the search returns them)."""
    pieces = [(cuts[s.rng.a].s, cuts[s.rng.b].s, bool(s.piece.reversed)) for s in states]
    return encode(level, pieces, foliage, fp)
