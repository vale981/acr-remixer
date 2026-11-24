# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Join smoothing (stagegen/joins.py) on synthetic straight roads."""

from __future__ import annotations

import numpy as np
import pytest

from stagegen.joins import join_shears


def _roads(h_in, h_out, n=401, L=200.0):
    t = np.linspace(-L, 0, n)[:, None]
    prev = np.c_[t * np.cos(h_in), t * np.sin(h_in), 0.05 * t]  # arrives at the origin
    t2 = np.linspace(0, L, n)[:, None]
    nxt = np.c_[t2 * np.cos(h_out), t2 * np.sin(h_out), -0.02 * t2]
    return prev, nxt


def _heading(a, b):
    v = b - a
    return np.degrees(np.arctan2(v[1], v[0]))


@pytest.mark.parametrize("deg_in,deg_out,line", [(80, 120, 0.0), (90, 60, 0.0), (100, 75, 0.3), (45, 85, -0.4)])
def test_both_sides_meet_at_the_mean_heading(deg_in, deg_out, line):
    h_in, h_out = np.radians(deg_in), np.radians(deg_out)
    sp, sn = join_shears([0, 0], line, h_in, h_out, 0.05, -0.02, 100.0, 100.0)
    prev, nxt = _roads(h_in, h_out)
    P, N = sp.apply(prev), sn.apply(nxt)
    mid = (deg_in + deg_out) / 2
    assert abs(_heading(P[-2, :2], P[-1, :2]) - mid) < 1.0
    assert abs(_heading(N[0, :2], N[1, :2]) - mid) < 1.0


def test_points_on_the_line_and_beyond_reach_stay():
    h_in, h_out = np.radians(80), np.radians(120)
    sp, sn = join_shears([10, 5], 0.2, h_in, h_out, 0.0, 0.0, 100.0, 100.0)
    e = np.array([np.cos(0.2), np.sin(0.2)])
    on_line = np.array([[10, 5, 3]]) + np.c_[np.linspace(-50, 50, 11)[:, None] * e, np.zeros(11)]
    for sh in (sp, sn):
        assert np.allclose(sh.apply(on_line), on_line)  # the shared edge: meshes stay seamless
        far = np.array([[10, 5, 0]]) + np.r_[sh.n * 150.0, 0.0]
        assert np.allclose(sh.apply(far), far)  # beyond the reach nothing moves
        behind = np.array([[10, 5, 0]]) - np.r_[sh.n * 30.0, 0.0]
        assert np.allclose(sh.apply(behind), behind)  # the other side is not touched


def test_gradient_step_is_split():
    sp, sn = join_shears([0, 0], 0.0, np.radians(90), np.radians(90), 0.08, -0.04, 100.0, 100.0)
    prev, nxt = _roads(np.radians(90), np.radians(90))
    prev[:, 2] = 0.08 * (prev[:, 1])
    nxt[:, 2] = -0.04 * (nxt[:, 1])
    P, N = sp.apply(prev), sn.apply(nxt)
    g_in = (P[-1, 2] - P[-2, 2]) / np.linalg.norm(P[-1, :2] - P[-2, :2])
    g_out = (N[1, 2] - N[0, 2]) / np.linalg.norm(N[1, :2] - N[0, :2])
    assert abs(g_in - 0.02) < 0.005 and abs(g_out - 0.02) < 0.005


@pytest.mark.parametrize("direction_deg", [0, 35, 90, 140, 250])
def test_yaw_matches_the_local_rotation(direction_deg):
    """An object's heading turns as the shear turns its own direction (not only
    directions across the line: trees and props face any way)."""
    sp, sn = join_shears([0, 0], 0.3, np.radians(80), np.radians(120), 0, 0, 100.0, 100.0)
    h = np.radians(direction_deg)
    for sh in (sp, sn):
        for d in (5.0, 30.0, 70.0):
            p = sh.p0 + sh.n * d + sh.e * 10
            v = np.array([np.cos(h), np.sin(h)]) * 0.01
            w = (sh.apply(np.array([p + v])) - sh.apply(np.array([p])))[0]
            diff = np.arctan2(w[1], w[0]) - h - sh.yaw(p[None], h)[0]
            assert abs((diff + np.pi) % (2 * np.pi) - np.pi) < np.radians(0.5)


def test_too_sharp_is_refused():
    assert join_shears([0, 0], 0.0, np.radians(90), np.radians(160), 0, 0, 100, 100, max_kink_deg=60) == (None, None)
