# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-License-Identifier: MIT

"""Piece sequences: parsing (synthetic) and layouts on the real Hafren South (game)."""
from __future__ import annotations

import numpy as np
import pytest

from stagegen.tiles import parse_sequence


def test_parse_sequence():
    assert parse_sequence("0-1 7-11 7-11r, 18-19") == [(0, 1, False), (7, 11, False), (7, 11, True), (18, 19, False)]
    assert parse_sequence("  ") == []


@pytest.mark.parametrize("bad", ["7", "a-b", "7-11x", "-3"])
def test_parse_sequence_rejects(bad):
    with pytest.raises(ValueError):
        parse_sequence(bad)


def _sequence(hafren, text):
    from stagegen.tiles import TileSearchConfig, make_ranges, sequence_state
    st, grid, cuts = hafren["ws"].stage, hafren["grid"], hafren["cuts"]
    cfg = TileSearchConfig(max_uses=99)
    ranges = make_ranges(st, grid, cuts, 2.0, cfg.min_len, cfg.max_len)
    return sequence_state(st, ranges, cuts, cfg, parse_sequence(text)), cfg


@pytest.mark.game
def test_practice_loop_joins_are_smooth(hafren):
    """Same piece three times in the same direction: two ~42 deg kinks, smoothed."""
    from stagegen.geometry import headings, wrap_angle
    from stagegen.tiles import smooth_layout_points
    (best, warns), cfg = _sequence(hafren, "0-1 7-11 7-11 7-11 7-11r 18-19")
    assert max(abs(k) for k in best.kinks) > 30
    shears, smoothed = smooth_layout_points(best, cfg)
    assert len(smoothed) >= 2
    pts = [s.piece.points for s in reversed(list(best.chain()))]
    for a, b in zip(pts, pts[1:]):
        P = np.vstack([a[-30:], b[1:30]])
        turn = np.degrees(np.abs(wrap_angle(np.diff(headings(P)))).max())
        assert turn < 20, f"sharp turn of {turn:.0f} deg at a join"


@pytest.mark.game
def test_unknown_piece_is_an_error(hafren):
    with pytest.raises(ValueError):
        _sequence(hafren, "0-1 99-100 18-19")


@pytest.mark.game
def test_all_tape_poles_decode(hafren):
    """Every tape pole must be decodable, or it stays behind and stretches its tape."""
    from stagegen.build.level import decode_components, learn_rules
    from stagegen.extract.zen import Package
    fs, ws = hafren["fs"], hafren["ws"]
    pkgs = [Package(fs.read(p)) for p in [ws.main] + ws.cells]
    rules = learn_rules(pkgs)
    poles = dec = 0
    for pk in pkgs:
        ex = pk.exports
        ix = {i for i, e in enumerate(ex) if e.name.startswith(("TapePoleActor", "BC_TapeHandleActor"))
              and ex[e.outer].name == "PersistentLevel"}
        poles += len(ix)
        dec += len({c.owner for c in decode_components(pk, rules) if c.root and c.loc is not None and c.owner in ix})
    assert poles > 0 and dec == poles
