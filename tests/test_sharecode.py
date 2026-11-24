# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Share codes (synthetic)."""

from __future__ import annotations

import numpy as np
import pytest

from stagegen.sharecode import (
    LegacyShareCode,
    ShareCodeError,
    decode,
    encode,
    fingerprint,
    foliage_flags,
    to_sequence,
)

PIECES = [(0, 318, False), (1563, 2805, False), (1563, 2805, True), (5322, 5519, False)]


def test_roundtrip():
    code = encode("WalesS4HafrenSouth", PIECES, "FGL", "a3f1")
    assert code.startswith("R2:WalesS4HafrenSouth:0-318,1563-2805,1563-2805r,5322-5519:FGL:a3f1:")
    sc = decode(code)
    assert sc.level == "WalesS4HafrenSouth" and list(sc.pieces) == PIECES
    assert (sc.forest, sc.grass, sc.foliage_tool) == (True, True, True)


def test_pasting_whitespace_and_case_is_tolerated():
    code = encode("GreeceS3Elatia", PIECES[:2], foliage_flags(True, False, True), "00ff")
    assert decode("  " + code[:20] + "\n" + code[20:].replace(code[-2:], code[-2:].lower())).pieces == tuple(PIECES[:2])


@pytest.mark.parametrize("typo", [("1563", "1536"), ("FGL", "F-L"), ("Wales", "Welsh")])
def test_typos_are_caught(typo):
    code = encode("WalesS4HafrenSouth", PIECES, "FGL", "a3f1")
    with pytest.raises(ShareCodeError, match="typo"):
        decode(code.replace(*typo, 1))


def test_legacy_codes_get_a_clear_message():
    with pytest.raises(LegacyShareCode, match="0.1.x"):
        decode("WalesS4HafrenSouth/4500/1/FGL")


@pytest.mark.parametrize("bad", ["hello", "R3:x:0-1:FGL:0000:AA", "R2:x:0-1:FGL"])
def test_unusable_codes(bad):
    with pytest.raises(ShareCodeError):
        decode(bad)


def test_snapping_to_cuts():
    cuts = [0.0, 318.4, 702.0, 1563.2, 2804.6, 5321.7, 5519.2]
    assert to_sequence(PIECES, cuts) == [(0, 1, False), (3, 4, False), (3, 4, True), (5, 6, False)]
    with pytest.raises(ShareCodeError, match="no piece boundary near 1000"):
        to_sequence([(0, 1000, False)], cuts)


def test_fingerprint_notices_changed_stage_data():
    pts = np.c_[np.arange(100.0) * 2, np.zeros(100), np.linspace(300, 320, 100)]
    fp = fingerprint(pts)
    assert len(fp) == 4 and fingerprint(pts.copy()) == fp
    moved = pts.copy()
    moved[50, 2] += 0.5
    assert fingerprint(moved) != fp
