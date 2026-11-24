# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Top-down render of a built container's collision (what the car drives on)."""
from __future__ import annotations

import numpy as np

from ..extract.bakeddata import read_baked
from ..extract.iostore import Container
from ..extract.splines import level_splines
from ..extract.zen import Package


def render_collision(utoc: str, baked_path: str, route_cell: str | None, out_png: str, title: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    c = Container(utoc)
    meshes = read_baked(Package(c.read(baked_path)))
    polys, z, road = [], [], []
    for m in meshes:
        if len(m.indices) < 3:
            continue
        V = m.vertices[:, [0, 2, 1]]
        tri = m.indices.reshape(-1, 3)
        isroad = "_Road" in (m.actor or "") or "Road" in (m.material or "")
        sel = tri[:: 1 if isroad else 3]
        polys.append(V[sel][:, :, :2])
        z.append(V[sel][:, :, 2].mean(1))
        road.append(np.full(len(sel), isroad))
    P = np.concatenate(polys); Z = np.concatenate(z); RD = np.concatenate(road)
    fig, ax = plt.subplots(figsize=(14, 14))
    pc = PolyCollection(P[~RD], array=Z[~RD], cmap="terrain", linewidths=0)
    ax.add_collection(pc)
    ax.add_collection(PolyCollection(P[RD], facecolors="#303030", linewidths=0))
    if route_cell:
        sp = level_splines(Package(c.read(route_cell)))
        C = sp.get("CenterSpline")
        if C is not None:
            ax.plot(C[:, 0], C[:, 1], "r-", lw=0.6)
            ax.plot(*C[0, :2], "go", ms=8)
            ax.plot(*C[-1, :2], "rs", ms=8)
    ax.autoscale(); ax.set_aspect("equal"); ax.set_title(title)
    fig.colorbar(pc, ax=ax, shrink=0.6, label="height (m)")
    fig.savefig(out_png, dpi=90); plt.close(fig)
