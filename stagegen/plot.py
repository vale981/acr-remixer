# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Top-down map + elevation profile of a generated layout."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .geometry import arclength
from .layout import Layout

PALETTE = ["#2a78d6", "#e8743b", "#19a979", "#a35fd0", "#c9a227", "#d64f7e"]
CONNECTOR = "#d62828"


def plot_layout(layout: Layout, path: str, title: str, corridor: float) -> None:
    stages = sorted({p.segment_id.split("#")[0] for p in layout.pieces if p.segment_id})
    colour = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(stages)}

    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1])
    ax, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    for p, (s0, s1) in zip(layout.pieces, layout.piece_ranges):
        xy = p.points[:, :2]
        if p.kind == "connector":
            ax.plot(xy[:, 0], xy[:, 1], color=CONNECTOR, lw=3.2, solid_capstyle="butt", zorder=3)
        else:
            c = colour[p.segment_id.split("#")[0]]
            # faint band = scanned corridor that comes along with the segment
            ax.plot(xy[:, 0], xy[:, 1], color=c, alpha=0.16,
                    solid_capstyle="round", zorder=1,
                    linewidth=corridor * 2 * 72 / _m_per_inch(ax, layout))
            ax.plot(xy[:, 0], xy[:, 1], color=c, lw=2.2, zorder=2)
        s = arclength(p.points) + s0
        ax2.plot(s / 1000, p.points[:, 2],
                 color=CONNECTOR if p.kind == "connector" else colour[p.segment_id.split("#")[0]],
                 lw=2.2 if p.kind == "connector" else 1.6)

    c = layout.centreline
    ax.plot(*c[0, :2], "o", color="black", ms=8, zorder=5)
    ax.plot(*c[-1, :2], "s", color="black", ms=8, zorder=5)
    ax.annotate("start", c[0, :2], xytext=(8, 8), textcoords="offset points")
    ax.annotate("finish", c[-1, :2], xytext=(8, 8), textcoords="offset points")
    ax.set_aspect("equal")
    ax.set_title(title, loc="left", fontsize=13)
    ax.set_xlabel("m"); ax.set_ylabel("m")
    handles = [plt.Line2D([], [], color=colour[s], lw=2.2, label=s) for s in stages]
    handles.append(plt.Line2D([], [], color=CONNECTOR, lw=3, label="generated connector"))
    ax.legend(handles=handles, loc="best", frameon=False, fontsize=9)
    for a in (ax, ax2):
        a.spines[["top", "right"]].set_visible(False)
        a.grid(alpha=0.15)
    ax2.set_xlabel("distance (km)"); ax2.set_ylabel("elevation (m)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _m_per_inch(ax, layout: Layout) -> float:
    """Rough metres-per-inch of the map axes so the corridor band has a true-ish width."""
    c = layout.centreline[:, :2]
    span = max(np.ptp(c[:, 0]), np.ptp(c[:, 1])) * 1.1
    bbox = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
    return span / max(min(bbox.width, bbox.height), 1e-3)
