# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-FileCopyrightText: 2026 iceone88
# SPDX-License-Identifier: GPL-3.0-or-later AND MIT

"""Manual sequences: joins over the limits are built anyway and reported as warnings."""

import numpy as np

from stagegen.model import Pacenote, Segment, SourceStage
from stagegen.tiles import Cut, Range, TileSearchConfig, sequence_state


def _sequence_fixture(
    join_line=np.pi / 2, join_heading=np.pi / 2, join_grade=0.0, join_width=7.0, prev_heading=np.pi / 2
):
    points = np.column_stack((np.arange(376) * 2.0, np.zeros(376), np.zeros(376)))
    stage = SourceStage("TestStage", "test", "gravel", points, np.full(len(points), 7.0), [Pacenote(20.0, "3 left")])
    cuts = [
        Cut(0.0, np.array([0.0, 0.0, 0.0]), 0.0, np.pi / 2, 0.0, 7.0, "start"),
        Cut(250.0, np.array([0.0, 250.0, 0.0]), 0.0, prev_heading, 0.0, 7.0, "cross"),
        Cut(500.0, np.array([500.0, 0.0, 0.0]), join_line, join_heading, join_grade, join_width, "cross"),
        Cut(750.0, np.array([500.0, 250.0, 0.0]), 0.0, np.pi / 2, 0.0, 7.0, "finish"),
    ]
    r0 = Range("TestStage#0-1", 0, 1, frozenset({(0, 0)}), Segment("r0", stage, 0, 125, 2.0))
    r1 = Range("TestStage#2-3", 2, 3, frozenset({(0, 1)}), Segment("r1", stage, 250, 375, 2.0))
    cfg = TileSearchConfig(
        target_length=500.0, min_len=200.0, max_len=300.0, max_uses=1, smooth_kink_deg=45.0, smooth_grade=0.12
    )
    return stage, cuts, [r0, r1], cfg


def test_manual_sequence_warns_for_kink_beyond_smoothing_limit():
    stage, cuts, ranges, cfg = _sequence_fixture(join_line=np.pi / 2, join_heading=0.0, prev_heading=np.deg2rad(30))
    best, warns = sequence_state(stage, ranges, cuts, cfg, [(0, 1, False), (2, 3, False)])
    assert len(list(best.chain())) == 2
    assert any("stays sharp" in w for w in warns)


def test_manual_sequence_warns_for_grade_step_above_smoothing_limit():
    stage, cuts, ranges, cfg = _sequence_fixture(join_line=0.0, join_heading=np.pi / 2, join_grade=0.13)
    _, warns = sequence_state(stage, ranges, cuts, cfg, [(0, 1, False), (2, 3, False)])
    assert any("gradient step" in w for w in warns)


def test_manual_sequence_warns_for_road_width_step_above_limit():
    stage, cuts, ranges, cfg = _sequence_fixture(join_line=0.0, join_heading=np.pi / 2, join_width=9.1)
    _, warns = sequence_state(stage, ranges, cuts, cfg, [(0, 1, False), (2, 3, False)])
    assert any("road width step" in w for w in warns)
