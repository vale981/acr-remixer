# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Two kinds of tests:

- synthetic (default): build their own data, never touch game files; these
  run in CI, so nothing from Assetto Corsa Rally ends up in logs or artifacts
- @pytest.mark.game: read the local ACR install (ACR_PAKS or the default path
  next to this repo); skipped automatically when it isn't there. Run them
  locally with `pytest`; CI runs `pytest -m "not game"`.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def pytest_configure(config):
    config.addinivalue_line("markers", "game: needs a local Assetto Corsa Rally install (skipped without)")


@pytest.fixture(scope="session")
def paks():
    from stagegen.extract.real import DEFAULT_PAKS
    p = os.environ.get("ACR_PAKS", DEFAULT_PAKS)
    if not os.path.isdir(p) or not any(f.endswith(".utoc") for f in os.listdir(p)):
        pytest.skip("no Assetto Corsa Rally install (set ACR_PAKS)")
    return p


@pytest.fixture(scope="session")
def hafren(paks):
    """Hafren South: source stage, World Partition source and cut grid (slow, shared)."""
    from stagegen.build import writers as W
    from stagegen.extract.iostore import GameFS
    from stagegen.extract.real import load_wp_source
    from stagegen.extract.zen import Package
    from stagegen.tiles import TileGrid, find_cuts
    fs = GameFS(paks, include_mods=False)
    ws = load_wp_source(fs, "WalesS4HafrenSouth")
    meshes, *_ = W.read_baked_raw(Package(fs.read(ws.baked)))
    grid = TileGrid.from_baked(meshes)
    return {"fs": fs, "ws": ws, "grid": grid, "cuts": find_cuts(ws.stage, grid, 2.0)}
