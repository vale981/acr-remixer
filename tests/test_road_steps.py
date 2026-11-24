# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The build report's step check on synthetic road meshes."""
from __future__ import annotations

import numpy as np

from stagegen.build.wp import _road_steps
from stagegen.build.writers import MeshOut


def _strip(x0, x1, z_fn, sub=b"PersistentLevel.SM_LevelTrack0101_GEN", width=12.0, step=1.0):
    """A road strip along x from x0 to x1, y in [-width/2, width/2], height z_fn(x)."""
    xs = np.arange(x0, x1 + 1e-9, step)
    V = []
    for x in xs:
        for y in (-width / 2, width / 2):
            V.append((x, z_fn(x), y))                        # baked frame: (x, up, y)
    V = np.array(V, np.float32)
    idx = []
    for i in range(len(xs) - 1):
        a, b, c, d = 2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3
        idx += [a, c, b, b, c, d]
    return MeshOut(V, np.array(idx, np.uint32), -1, (0, 0), (0, 0), sub, np.zeros(4, np.float32))


def _route(z_fn, x0=0.0, x1=200.0):
    xs = np.arange(x0, x1 + 1e-9, 2.0)
    return np.c_[xs, np.zeros_like(xs), [z_fn(x) for x in xs]]


def slope(x):
    return 0.05 * x


def test_smooth_road_has_no_steps():
    assert _road_steps([_strip(-5, 205, slope)], _route(slope)) == []


def test_step_between_two_meshes_is_found():
    lo = _strip(-5, 100, slope)
    hi = _strip(100, 205, lambda x: slope(x) + 0.7)
    steps = _road_steps([lo, hi], _route(slope))
    assert steps and abs(steps[0][0] - 100) <= 2 and abs(steps[0][2] - 0.7) < 0.1


def test_small_unevenness_is_not_a_step():
    lo = _strip(-5, 100, slope)
    hi = _strip(100, 205, lambda x: slope(x) + 0.2)
    assert _road_steps([lo, hi], _route(slope)) == []


def test_props_are_ignored():
    road = _strip(-5, 205, slope)
    bale = _strip(90, 110, lambda x: slope(x) + 1.0, sub=b"PersistentLevel.StaticMeshActor_UAID_1", width=4.0)
    assert _road_steps([road, bale], _route(slope)) == []
