# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-FileCopyrightText: 2026 iceone88
# SPDX-License-Identifier: MIT

import numpy as np

from stagegen.geometry import arclength, grade, curvature, resample
from stagegen.joins import Shear, seam_edge_weights, source_aligned_edge_interval
from stagegen.model import SourceStage
from stagegen.tiles import TileGrid, find_cuts, make_ranges


def test_resample_station_is_not_nominal_index_step():
    src = np.array([[0.0, 0.0, 0.0], [1001.0, 0.0, 0.0]])
    p = resample(src, 2.0)
    s = arclength(p)
    assert len(p) == 501
    assert np.isclose(s[-1], 1001.0)
    assert not np.isclose((len(p) - 1) * 2.0, s[-1])
    assert np.isclose(np.diff(s).mean(), 1001.0 / 500)


def test_geometry_derivatives_use_physical_station():
    s = np.array([0.0, 0.4, 1.6, 3.9, 6.0, 9.5])
    p = np.column_stack([s, np.zeros_like(s), 0.05 * s])
    assert np.allclose(grade(p, s, window=1), 0.05, atol=1e-12)

    theta = np.array([0.0, 0.05, 0.13, 0.21, 0.31, 0.4, 0.53, 0.7])
    radius = 100.0
    circle = np.column_stack([radius * np.sin(theta), radius * (1 - np.cos(theta)), np.zeros_like(theta)])
    sc = arclength(circle)
    k = curvature(circle, sc, window=1)
    assert np.allclose(k[2:-2], 1.0 / radius, atol=2e-4)


def test_fractional_tile_cut_segments_share_exact_endpoint_and_length():
    x = np.linspace(0.0, 120.3, 30)
    p = np.column_stack([x, np.zeros_like(x), 0.02 * x])
    w = 6.0 + 0.01 * x
    stage = SourceStage("synthetic", "test", "gravel", p, w)
    grid = TileGrid(cell=60.0, x0=0.0, y0=-30.0, tiles={(0, 0), (0, 1), (0, 2)})
    cuts = find_cuts(stage, grid, step=2.0, min_gap=10.0)
    cut = next(c for c in cuts if c.kind == "cross")
    assert np.isclose(cut.p[0], 60.0, atol=1e-9)
    ranges = make_ranges(stage, grid, cuts, step=2.0, min_len=10.0, max_len=100.0)
    first = next(r.seg for r in ranges if r.a == 0 and r.b == 1)
    second = next(r.seg for r in ranges if r.a == 1 and r.b == 2)
    p1, p2 = first.points(), second.points()
    assert np.allclose(p1[-1], p2[0], atol=1e-10)
    assert np.isclose(first.length + second.length, stage.station[-1], atol=1e-9)
    assert np.allclose(first.widths()[-1], second.widths()[0], atol=1e-10)
    assert np.isclose(arclength(p1)[-1], first.length, atol=1e-9)
    assert np.isclose(arclength(p2)[-1], second.length, atol=1e-9)


def test_shear_yaw_matches_finite_difference_jacobian():
    sh = Shear(
        p0=np.array([0.0, 0.0]),
        e=np.array([1.0, 0.0]),
        n=np.array([0.0, 1.0]),
        A=0.2,
        Az=0.0,
        D=120.0,
    )
    p = np.array([[10.0, 40.0, 0.0]])
    heading = np.array([np.radians(30.0)])
    delta = sh.yaw(p, heading)[0]
    eps = 1e-5
    v = np.array([np.cos(heading[0]), np.sin(heading[0]), 0.0])
    q = np.vstack([p[0], p[0] + eps * v])
    mapped = sh.apply(q)
    dv = mapped[1, :2] - mapped[0, :2]
    got = np.arctan2(dv[1], dv[0])
    expected = heading[0] + delta
    assert abs((got - expected + np.pi) % (2 * np.pi) - np.pi) < 1e-7


def test_finite_tile_edge_interval_remains_correct_after_180_degree_rotation():
    # Source +X maps to world -X.  The coordinate interval is expressed along
    # that transformed source-positive axis, not the world-positive axis.
    q0, q1 = source_aligned_edge_interval(coordinate=100.0, grid_origin=0.0, cell=350.0)
    assert np.allclose([q0, q1], [-100.0, 250.0])
    p0 = np.array([10.0, 20.0])
    e = np.array([-1.0, 0.0])
    pts = np.array([p0 + q0 * e, p0 + 100.0 * e, p0 + q1 * e,
                    p0 + (q0 - 1.0) * e, p0 + (q1 + 1.0) * e])
    mask, _ = seam_edge_weights(pts, p0, e, q0, q1, width=10.0)
    assert mask.tolist() == [True, True, True, False, False]


def test_seam_weights_are_limited_to_finite_shared_edge_band():
    pts = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 4.0, 0.0],
        [8.0, 0.0, 0.0],
        [0.0, 11.0, 0.0],
        [-6.0, 0.0, 0.0],
    ])
    mask, weights = seam_edge_weights(
        pts, p0=np.array([0.0, 0.0]), e=np.array([1.0, 0.0]),
        along_min=-5.0, along_max=5.0, width=10.0,
    )
    assert mask.tolist() == [True, True, False, False, False]
    assert np.isclose(weights[0], 1.0)
    assert 0.0 < weights[1] < 1.0
    assert np.all(weights[~mask] == 0.0)


def test_tile_sequence_remains_contiguous_after_fractional_stationing():
    from stagegen.tiles import TileSearchConfig, sequence_state
    from stagegen.layout import build_layout

    x = np.linspace(0.0, 1000.0, 501)
    p = np.column_stack([x, np.zeros_like(x), 0.01 * x])
    stage = SourceStage("straight", "test", "gravel", p, np.full(len(p), 7.0))
    grid = TileGrid(
        cell=250.0, x0=0.0, y0=-125.0,
        tiles={(0, 0), (0, 1), (0, 2), (0, 3)},
    )
    cuts = find_cuts(stage, grid, step=2.0, min_gap=20.0)
    ranges = make_ranges(stage, grid, cuts, step=2.0, min_len=200.0, max_len=300.0)
    by_pair = {(r.a, r.b): r for r in ranges}
    seq = [(i, i + 1, False) for i in range(len(cuts) - 1)]
    assert all((a, b) in by_pair for a, b, _ in seq)

    cfg = TileSearchConfig(min_len=200.0, max_len=300.0, target_length=1000.0,
                           clearance=95.0, max_kink_deg=15.0, smooth_kink_deg=45.0)
    state, warns = sequence_state(stage, ranges, cuts, cfg, seq)
    layout = build_layout(state, step=2.0)
    assert not warns
    assert np.isclose(layout.length, stage.station[-1], atol=1e-8)
    assert np.allclose(layout.centreline[0], stage.points[0], atol=1e-9)
    assert np.allclose(layout.centreline[-1], stage.points[-1], atol=1e-9)
    assert np.all(np.diff(arclength(layout.centreline)) > 0)


def test_join_shears_match_tangent_and_grade_at_both_sides():
    from stagegen.joins import join_shears

    p0 = np.zeros(2)
    line = 0.0
    h_prev = np.pi / 2 - np.radians(10.0)
    h_next = np.pi / 2 + np.radians(10.0)
    g_prev, g_next = -0.03, 0.05
    sh_prev, sh_next = join_shears(
        p0, line, h_prev, h_next, g_prev, g_next,
        120.0, 120.0, max_kink_deg=60.0,
    )
    assert sh_prev is not None and sh_next is not None
    target_grade = 0.5 * (g_prev + g_next)

    for sh, heading, source_grade, side in (
        (sh_prev, h_prev, g_prev, -1.0),
        (sh_next, h_next, g_next, 1.0),
    ):
        eps = 1e-4
        p0_side = np.array([0.0, side * eps, 0.0])
        v = np.array([np.cos(heading), np.sin(heading), 0.0])
        p1_side = p0_side + eps * v + np.array([0.0, 0.0, source_grade * eps])
        out = sh.apply(np.vstack([p0_side, p1_side]))
        dv = out[1] - out[0]
        h_out = np.arctan2(dv[1], dv[0])
        g_out = dv[2] / np.linalg.norm(dv[:2])
        assert abs((h_out - np.pi / 2 + np.pi) % (2 * np.pi) - np.pi) < 1e-4
        assert abs(g_out - target_grade) < 1e-4


def test_world_partition_assignment_uses_interpolated_source_station_array():
    from stagegen.build.wp import PieceX, _Assign

    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    station = np.array([0.0, 1.0, 4.0, 10.0])
    piece = PieceX(3.0, 8.0, np.eye(3), np.zeros(3), 0.0, frozenset())
    assign = _Assign(pts, station, [piece], ext=0.0)
    k, d, s, idx = assign(np.array([[4.1, 0.0, 0.0]]))
    assert idx[0] == 2
    assert np.isclose(s[0], 4.0)
    assert k[0] == 0


def test_generic_builder_piece_index_uses_segment_station_bounds():
    from stagegen.build.stage import _Pieces
    from stagegen.layout import Layout
    from stagegen.model import Piece, Segment

    x = np.array([0.0, 1.0, 4.0, 10.0])
    pts = np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])
    stage = SourceStage("s", "test", "gravel", pts, np.full(4, 7.0))
    seg = Segment("s#0", stage, 1, 3, 2.0)
    piece = Piece("segment", seg.points(), seg.widths(), "s#0", False, 0.0,
                  np.zeros(3), [])
    layout = Layout([piece], piece.points, piece.widths, [], [(0.0, 9.0)])
    mapped = _Pieces(layout, {"s#0": seg}, np.zeros(3))
    assert np.array_equal(mapped.index(np.array([0.0, 1.0, 4.0, 10.0]), True, True),
                          np.array([0, 0, 0, 0]))
    assert np.allclose(mapped.items[0][:2], [1.0, 10.0])


def test_hism_callback_passes_instance_yaw_without_overwriting_layer():
    import struct
    from stagegen.build.foliage import move_hism

    # Minimal serialized HISM instance bulk with one matrix and no cached bounds.
    matrix = np.eye(4, dtype=np.float64)
    angle = np.radians(37.0)
    matrix[0, :2] = [np.cos(angle), np.sin(angle)]
    matrix[:3, 3] = 0.0
    matrix[3, 3] = 1.0
    blob = bytearray(struct.pack("<Ii", 128, 1) + matrix.astype("<f8").tobytes())
    seen = {}

    def place(world, source_yaw, layer=0):
        seen["world"] = world.copy()
        seen["source_yaw"] = source_yaw.copy()
        seen["layer"] = layer
        return np.ones(len(world), dtype=bool), world.copy(), np.zeros(len(world))

    # Mirror the callback shape used by build/wp.py's process() loop.
    _, stats = move_hism(bytes(blob), np.zeros(3),
                         lambda W, source_yaw, layer=2: place(W, source_yaw, layer))
    assert stats["n"] == 1
    assert seen["layer"] == 2
    assert np.allclose(seen["source_yaw"], [angle])


def test_shear_yaw_at_seam_uses_inward_one_sided_derivative():
    sh = Shear(
        p0=np.array([0.0, 0.0]), e=np.array([1.0, 0.0]),
        n=np.array([0.0, 1.0]), A=0.2, Az=0.0, D=120.0,
    )
    p = np.array([[0.0, 0.0, 0.0]])  # exactly on the seam
    h = np.array([np.radians(30.0)])  # points into the transformed side
    predicted = h[0] + sh.yaw(p, h)[0]
    eps = 1e-6
    v = np.array([np.cos(h[0]), np.sin(h[0]), 0.0])
    mapped = sh.apply(np.vstack([p[0], p[0] + eps * v]))
    actual_vec = mapped[1, :2] - mapped[0, :2]
    actual = np.arctan2(actual_vec[1], actual_vec[0])
    assert abs((actual - predicted + np.pi) % (2 * np.pi) - np.pi) < 1e-6


def test_shear_yaw_at_seam_uses_directional_one_sided_derivative():
    sh = Shear(
        p0=np.array([0.0, 0.0]), e=np.array([1.0, 0.0]),
        n=np.array([0.0, 1.0]), A=0.2, Az=0.0, D=120.0,
    )
    p = np.array([[0.0, 0.0, 0.0]])
    for degrees, should_shear in ((30.0, True), (-30.0, False)):
        h = np.array([np.radians(degrees)])
        predicted = h[0] + sh.yaw(p, h)[0]
        eps = 1e-6
        v = np.array([np.cos(h[0]), np.sin(h[0]), 0.0])
        mapped = sh.apply(np.vstack([p[0], p[0] + eps * v]))
        actual_vec = mapped[1, :2] - mapped[0, :2]
        actual = np.arctan2(actual_vec[1], actual_vec[0])
        assert abs((actual - predicted + np.pi) % (2 * np.pi) - np.pi) < 1e-6
        if not should_shear:
            assert abs(predicted - h[0]) < 1e-12


def test_join_assessment_offsets_match_closed_form_for_symmetric_kink():
    from stagegen.joins import assess_join

    theta = np.radians(15.0)
    reach = 120.0
    a = assess_join(0.0, np.pi / 2 - theta / 2, np.pi / 2 + theta / 2,
                    0.0, 0.0, reach, reach, max_kink_deg=20.0)
    assert a is not None
    expected_shift = 4.0 * reach * np.sin(theta / 2) / 27.0
    assert np.isclose(a.max_boundary_parallel_shift_m, expected_shift, rtol=1e-10)
    expected_radius = reach * np.cos(theta / 2) / (4.0 * np.tan(theta / 2))
    assert np.isclose(1.0 / (4.0 * abs(a.A_prev) / a.D_prev_normal), expected_radius, rtol=1e-10)
    assert np.isclose(a.max_vertical_shift_m, 0.0)


def test_automatic_tile_search_limits_remain_conservative():
    from stagegen.tiles import TileSearchConfig

    cfg = TileSearchConfig()
    assert cfg.max_kink_deg == 15.0
    assert cfg.max_grade_mismatch == 0.06
    assert cfg.smooth_kink_deg == 45.0
    assert cfg.smooth_grade == 0.12
    assert cfg.clearance == 95.0
