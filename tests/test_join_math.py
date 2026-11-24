# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-FileCopyrightText: 2026 iceone88
# SPDX-License-Identifier: MIT

import numpy as np
import pytest

from stagegen.joins import assess_join, join_shears


def test_join_assessment_limits_normal_projection_and_reports_offsets():
    angle = 0.0
    h_in = np.deg2rad(80)
    h_out = np.deg2rad(100)
    a = assess_join(angle, h_in, h_out, 0.10, 0.0, 100.0, 100.0,
                    max_kink_deg=30, min_normal_component=0.2)
    assert a is not None
    assert abs(a.kink_deg - 20.0) < 1e-8
    assert abs(a.D_prev_normal / a.normal_in - 100.0) < 1e-8
    assert abs(a.D_next_normal / a.normal_out - 100.0) < 1e-8
    assert a.max_boundary_parallel_shift_m >= 0
    assert a.max_vertical_shift_m > 0


def test_nearly_parallel_boundary_crossing_is_rejected_not_clamped():
    # With a horizontal grid line and a road travelling along it, the normal
    # component is zero and the shear equations are ill-conditioned.
    assert assess_join(0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 100.0) is None
    a, b = join_shears([0, 0], 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 100.0)
    assert a is None and b is None


def test_join_shear_maps_both_sides_to_same_boundary_tangent_and_grade():
    line = 0.0
    h_in, h_out = np.deg2rad(80), np.deg2rad(100)
    g_in, g_out = 0.10, 0.0
    prev, nxt = join_shears([0, 0], line, h_in, h_out, g_in, g_out,
                            100.0, 100.0, max_kink_deg=30)
    assert prev is not None and nxt is not None

    eps = 1e-3
    u_in = np.array([np.cos(h_in), np.sin(h_in)])
    u_out = np.array([np.cos(h_out), np.sin(h_out)])
    before = np.array([[-eps * u_in[0], -eps * u_in[1], -eps * g_in], [0, 0, 0]])
    after = np.array([[0, 0, 0], [eps * u_out[0], eps * u_out[1], eps * g_out]])
    mapped_before = prev.apply(before)
    mapped_after = nxt.apply(after)

    t_in = (mapped_before[1] - mapped_before[0]) / eps
    t_out = (mapped_after[1] - mapped_after[0]) / eps
    heading_target = np.deg2rad(90)
    assert abs(np.arctan2(t_in[1], t_in[0]) - heading_target) < 1.5e-3
    assert abs(np.arctan2(t_out[1], t_out[0]) - heading_target) < 1.5e-3
    assert abs(t_in[2] / np.hypot(t_in[0], t_in[1]) - 0.05) < 1.5e-3
    assert abs(t_out[2] / np.hypot(t_out[0], t_out[1]) - 0.05) < 1.5e-3


def test_join_rejects_heading_kink_over_configured_limit():
    assert assess_join(np.pi / 2, np.deg2rad(60), np.deg2rad(120),
                       0, 0, 100, 100, max_kink_deg=30) is None


def test_join_grade_is_matched_after_horizontal_shear_changes_speed():
    # Exercise asymmetric crossing angles: the horizontal shear changes
    # horizontal speed, so checking dz per source-distance is insufficient.
    rng = np.random.default_rng(1847)
    checked = 0
    for _ in range(500):
        line = rng.uniform(-np.pi, np.pi)
        e = np.array([np.cos(line), np.sin(line)])
        n = np.array([-e[1], e[0]])
        if rng.random() < 0.5:
            n = -n
        a_in, a_out = rng.uniform(-1.25, 1.25, size=2)
        u_in = np.cos(a_in) * n + np.sin(a_in) * e
        u_out = np.cos(a_out) * n + np.sin(a_out) * e
        h_in = np.arctan2(u_in[1], u_in[0])
        h_out = np.arctan2(u_out[1], u_out[0])
        g_in, g_out = rng.uniform(-0.18, 0.18, size=2)
        prev, nxt = join_shears([0.0, 0.0], line, h_in, h_out,
                                g_in, g_out, 100.0, 100.0,
                                max_kink_deg=160.0, min_normal_component=0.2)
        if prev is None:
            continue

        target_a = (a_in + a_out) * 0.5
        target_dir = np.sin(target_a) * e + np.cos(target_a) * n
        eps = 1e-5
        before = np.array([[-eps * u_in[0], -eps * u_in[1], -eps * g_in],
                           [0.0, 0.0, 0.0]])
        after = np.array([[0.0, 0.0, 0.0],
                          [eps * u_out[0], eps * u_out[1], eps * g_out]])
        t_in = (prev.apply(before)[1] - prev.apply(before)[0]) / eps
        t_out = (nxt.apply(after)[1] - nxt.apply(after)[0]) / eps

        # Both transformed paths must have the chosen heading and mean grade
        # when grade is measured against transformed horizontal arclength.
        for tangent in (t_in, t_out):
            heading_error = np.arctan2(
                target_dir[0] * tangent[1] - target_dir[1] * tangent[0],
                target_dir @ tangent[:2],
            )
            grade_after = tangent[2] / np.linalg.norm(tangent[:2])
            assert abs(heading_error) < 2e-5
            assert grade_after == pytest.approx((g_in + g_out) * 0.5, abs=2e-5)
        checked += 1
    assert checked >= 100
