# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Offline drive check on synthetic collision: holes, small steps, surfaces, winding."""

from __future__ import annotations

import numpy as np

from stagegen.build.drivecheck import collision_from_meshes, drive_check, material_names
from stagegen.build.writers import MeshOut, baked_up

ROAD, GRASS = -7, -5
NAMES = {ROAD: "PM_SurfaceGravelHard", GRASS: "PM_SurfaceGrassWales"}


def _strip(x0, x1, z_fn, mat=ROAD, y0=-3.0, y1=3.0, step=0.5, sub=b"\x00\x00\x00\x00.SM_LevelTrack0101_GEN", up=True):
    """Collision strip along x from x0 to x1, y in [y0, y1], height z_fn(x),
    wound facing up like the game's ground (or facing down with up=False)."""
    xs = np.arange(x0, x1 + 1e-9, step)
    V = np.array([(x, z_fn(x), y) for x in xs for y in (y0, y1)], np.float32)  # baked frame (x, up, y)
    idx = []
    for i in range(len(xs) - 1):
        a, b, c, d = 2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3
        idx += [a, b, c, b, d, c] if up else [a, c, b, b, c, d]
    return MeshOut(V, np.array(idx, np.uint32), mat, (0, 0), (0, 0), sub, np.zeros(4, np.float32))


def slope(x):
    return 0.04 * x


def _route(x1=300.0, lift=0.06):
    xs = np.arange(0.0, x1 + 1e-9, 2.0)
    return np.c_[xs, np.zeros_like(xs), slope(xs) + lift]  # route line just over the road


def _check(meshes, **kw):
    return drive_check(collision_from_meshes(meshes, NAMES), _route(), **kw)


def test_continuous_road_is_fine():
    rep = _check([_strip(-5, 150, slope), _strip(149, 305, slope)])
    assert rep.ok and not rep.holes and not rep.steps and not rep.off_road
    assert rep.lines[0].startswith("holes: none")


def test_short_gap_at_a_join_is_a_hole():
    # 1.5 m gap where two roads meet: the build's 1 m / 4 m drivability
    # checks do not see it, the car drops into it
    rep = _check([_strip(-5, 150, slope), _strip(151.5, 305, slope)])
    assert len(rep.holes) == 1
    a, b = rep.holes[0]
    assert 149.5 <= a <= 150.5 and 150.5 <= b <= 152.0
    assert rep.lines[0].startswith("HOLES")


def test_pit_is_a_hole():
    # 40 cm deep, 3 m long: the car drops in
    pit = lambda x: slope(x) - (0.4 if 150 <= x <= 153 else 0.0)  # noqa: E731
    rep = _check([_strip(-5, 305, pit)])
    assert len(rep.holes) == 1 and 149 <= rep.holes[0][0] <= 151 and 152 <= rep.holes[0][1] <= 154


def test_smooth_dip_under_the_route_line_is_not_a_hole():
    # the road 50 cm under the route line over 40 m, eased in and out (smoothed
    # joins of real remixes): the car does not notice
    dip = lambda x: slope(x) - 0.5 * np.sin(np.pi * np.clip((x - 130) / 40, 0, 1)) ** 2  # noqa: E731
    rep = _check([_strip(-5, 305, dip, step=0.25)])
    assert rep.ok and 0.5 <= rep.route_above <= 0.62


def test_small_step_is_found():
    # 12 cm: far below the build's 0.4 m step check, enough to hit the car
    rep = _check([_strip(-5, 150, slope), _strip(150, 305, lambda x: slope(x) + 0.12)])
    assert not rep.holes
    assert rep.steps and abs(rep.steps[0][0] - 150) <= 1 and abs(rep.steps[0][2] - 0.12) < 0.02


def test_steady_slope_is_not_a_step():
    steep = lambda x: 0.15 * x  # noqa: E731
    xs = np.arange(0.0, 301.0, 2.0)
    route = np.c_[xs, np.zeros_like(xs), steep(xs) + 0.06]
    rep = drive_check(collision_from_meshes([_strip(-5, 305, steep)], NAMES), route)
    assert rep.ok


def test_wheels_on_grass_are_flagged():
    road = _strip(-5, 305, slope)
    # grass strip covering the left wheel line from 200 to 230 m, slightly above the road
    grass = _strip(200, 230, lambda x: slope(x) + 0.02, mat=GRASS, y0=-1.5, y1=-0.5)
    rep = _check([road, grass])
    assert rep.off_road and 199 <= rep.off_road[0][0] <= 201 and 229 <= rep.off_road[0][1] <= 231
    assert any(line.startswith("WHEELS OFF") for line in rep.lines)


def test_surface_runs_name_the_material_and_mesh():
    rep = _check([_strip(-5, 305, slope)])
    assert [(m, a) for _, _, m, a in rep.surfaces] == [("PM_SurfaceGravelHard", "Track0101")]


def test_props_above_the_road_are_not_ground():
    # a bale 2 m over the road is above the probe window: the road stays the surface
    road = _strip(-5, 305, slope)
    bale = _strip(100, 102, lambda x: slope(x) + 2.0, mat=GRASS)
    assert _check([road, bale]).ok


class _Pkg:
    """Just what material_names reads from a zen package."""

    def __init__(self, imports, pkg_names):
        self.imports, self._names = imports, pkg_names

    def imported_package_names(self):
        return self._names


def test_material_names_from_package_imports():
    pkg = _Pkg([0, (2 << 62) | (1 << 32) | 4], ["/Game/X", "/Game/Data/PhysicalMaterials/PM_SurfaceGravelHard"])
    assert material_names(pkg, [-2, -1]) == {-2: "PM_SurfaceGravelHard", -1: "import0"}


def test_strip_helper_faces_up():
    m = _strip(0, 10, slope)
    assert (baked_up(m.vertices, m.indices) > 0.99).all()


def test_upside_down_ground_is_flagged():
    # right height, wound facing down: the car stops dead where it drives onto it
    rep = _check([_strip(-5, 150, slope), _strip(149, 305, slope, up=False)])
    assert not rep.holes and not rep.steps
    assert rep.upside_down and 148.5 <= rep.upside_down[0][0] <= 150.5 and rep.upside_down[-1][1] >= 299
    assert not rep.ok and any(line.startswith("UPSIDE-DOWN") for line in rep.lines)
