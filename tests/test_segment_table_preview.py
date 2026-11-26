# SPDX-FileCopyrightText: 2026 iceone88
# SPDX-License-Identifier: MIT

"""segment_table: the stock map, its cut-to-cut sections and build candidates (synthetic)."""

import json

import numpy as np

from stagegen.model import Segment, SourceStage
from stagegen.tiles import Cut, Range, segment_table


def _segment(stage, cuts, a, b):
    ca, cb = cuts[a], cuts[b]
    i0 = max(0, int(np.searchsorted(stage.station, ca.s, side="right")) - 1)
    i1 = min(len(stage.points) - 1, int(np.searchsorted(stage.station, cb.s, side="left")))
    seg = Segment(
        f"{stage.name}#{a}-{b}",
        stage,
        i0,
        i1,
        2.0,
        start_s=ca.s,
        end_s=cb.s,
        start_point=ca.p,
        end_point=cb.p,
        start_width=ca.width,
        end_width=cb.width,
    )
    return Range(f"{stage.name}#{a}-{b}", a, b, frozenset({(0, 0)}), seg)


def _source_stage_and_cuts():
    x = np.arange(0.0, 402.0, 2.0)
    points = np.column_stack((x, np.zeros_like(x), 100.0 + 0.05 * x))
    stage = SourceStage(
        "WalesS4HafrenSouth",
        "wales",
        "gravel",
        points,
        np.full(len(points), 7.0),
    )
    cuts = [
        Cut(0.0, points[0].copy(), 0.0, 0.0, 0.05, 7.0, "start"),
        Cut(100.0, points[50].copy(), 0.0, 0.0, 0.05, 7.0, "cross"),
        Cut(200.0, points[100].copy(), 0.0, 0.0, 0.05, 7.0, "cross"),
        Cut(float(stage.station[-1]), points[-1].copy(), 0.0, 0.0, 0.05, 7.0, "finish"),
    ]
    return stage, cuts


def test_segment_table_exports_full_route_cut_elevation_and_piece_geometry():
    stage, cuts = _source_stage_and_cuts()
    ranges = [_segment(stage, cuts, a, b) for a, b in ((0, 1), (1, 2), (2, 3))]

    table = segment_table(stage, ranges, cuts)

    # The payload is passed directly from the Python backend to the Rust GUI.
    encoded = json.dumps(table, separators=(",", ":"))
    decoded = json.loads(encoded)
    assert decoded["level"] == stage.name
    assert len(decoded["route_xyz"]) == len(decoded["route_s"])
    assert len(decoded["sections"]) == len(decoded["cuts"]) - 1
    table = decoded
    assert table["route_xyz"][0] == [0.0, 0.0, 100.0]
    assert table["route_xyz"][-1] == [400.0, 0.0, 120.0]
    assert table["cuts"][1]["z"] == 105.0

    piece = next(p for p in table["pieces"] if p["id"] == "1-2")
    assert piece["reversible"] is True
    assert piece["buildable"] is True
    assert piece["station_start"] == 100.0
    assert piece["station_end"] == 200.0
    assert np.isclose(piece["midpoint"][2], 107.5)
    assert np.isclose(piece["elevation_min"], 105.0)
    assert np.isclose(piece["elevation_max"], 110.0)
    assert piece["xyz"][0] == [100.0, 0.0, 105.0]
    assert piece["xyz"][-1] == [200.0, 0.0, 110.0]
    assert len(piece["station"]) == len(piece["xyz"])


def test_map_sections_include_intervals_filtered_out_of_build_candidates():
    stage, cuts = _source_stage_and_cuts()
    # Deliberately omit 1-2 from the candidate ranges. The map still needs to
    # show that physical source section and label it as unavailable to sequence.
    ranges = [_segment(stage, cuts, 0, 1), _segment(stage, cuts, 2, 3)]

    table = segment_table(stage, ranges, cuts)

    assert [section["id"] for section in table["sections"]] == ["0-1", "1-2", "2-3"]
    assert [section["buildable"] for section in table["sections"]] == [True, False, True]
    assert table["sections"][1]["xyz"][0] == [100.0, 0.0, 105.0]
    assert table["sections"][1]["xyz"][-1] == [200.0, 0.0, 110.0]
    assert [piece["id"] for piece in table["pieces"]] == ["0-1", "2-3"]


def test_segment_table_keeps_stock_map_when_no_manual_candidates_exist():
    stage, cuts = _source_stage_and_cuts()

    table = segment_table(stage, [], cuts)

    assert len(table["route_xyz"]) > 1
    assert len(table["sections"]) == 3
    assert table["pieces"] == []
    assert not any(section["buildable"] for section in table["sections"])
