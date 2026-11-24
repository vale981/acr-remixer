# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""acr-stagegen command line.

python -m stagegen.cli generate --fake --length 10000 --seed 3 --out out/
python -m stagegen.cli generate --location WrcgEs --length 8000 --seed 1 --out out/
python -m stagegen.cli list
"""

from __future__ import annotations

import argparse
import os
import time

from .connector import max_abs_grade
from .fake import fake_location, self_clear
from .layout import build_layout, to_json
from .search import SearchConfig, search


def plot_layout(*args, **kw):
    """PNG preview (matplotlib, optional: release builds leave it out; the GUI draws its own)."""
    try:
        from .plot import plot_layout as _p
    except ImportError:
        return
    _p(*args, **kw)


from .segments import CutConfig, cut_stage

# Real roads are twistier than the synthetic ones: accept cuts on gentler
# curves (R > 120 m) and slightly less steady grades.
REAL_CUT = CutConfig(max_curvature=1 / 120, max_grade_change=0.04, window_m=20.0)


def _list(a) -> int:
    from .extract.iostore import GameFS
    from .extract.real import DEFAULT_PAKS, find_levels

    fs = GameFS(a.paks or DEFAULT_PAKS, include_mods=True)
    for lv in find_levels(fs):
        c, _ = fs.index[lv]
        comp = "compressed" if any(b.method for b in c.blocks[:1]) else "uncompressed"
        print(f"{os.path.basename(lv)[:-5]:40s} {os.path.basename(c.utoc):34s} {comp}")
    return 0


# Single-level builds: shorter segments so a 6-7 km stage still recombines,
# with the opening/closing segments anchored to the real start and finish.
BUILD_CUT = CutConfig(
    max_curvature=1 / 90, max_grade_change=0.05, window_m=16.0, min_len=200.0, max_len=700.0, anchor_ends=True
)


def _segments(a) -> int:
    """The level's pieces between tile crossings, for --sequence."""
    import json
    from .build import writers as W
    from .extract.iostore import GameFS
    from .extract.real import DEFAULT_PAKS, load_wp_source
    from .extract.zen import Package
    from .tiles import TileGrid, TileSearchConfig, find_cuts, make_ranges, segment_table

    fs = GameFS(a.paks or DEFAULT_PAKS, include_mods=False)
    ws = load_wp_source(fs, a.level)
    meshes, *_ = W.read_baked_raw(Package(fs.read(ws.baked)))
    grid = TileGrid.from_baked(meshes)
    cuts = find_cuts(ws.stage, grid, 2.0)
    cfg = TileSearchConfig()
    tab = segment_table(ws.stage, make_ranges(ws.stage, grid, cuts, 2.0, cfg.min_len, cfg.max_len), cuts)
    if a.json:
        print(json.dumps(tab))
        return 0
    print(
        f"{a.level}: cuts 0 (start) .. {len(cuts) - 1} (finish); a sequence starts with 0-N and ends with N-"
        f"{len(cuts) - 1}; 'r' drives a piece backwards"
    )
    for p in tab["pieces"]:
        print(f"  {p['id']:>7}  {p['length']:6.0f} m" + ("" if p["reversible"] else "   (start/finish, forwards only)"))
    return 0


def _pretty(level: str) -> str:
    """WalesS4HafrenSouth -> Hafren South"""
    import re

    m = re.match(r"[A-Z][a-z]+(?:[A-Z][a-z]+)?S\d+(.*)$", level)
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", m.group(1) if m and m.group(1) else level)


def _own_name(a) -> str:
    """Level name of a generated stage: unique per source, seed and length; it
    must not end in digits (FNames store a trailing _N as a number) and must
    not contain the source level name (that gets renamed)."""
    import re

    m = re.match(r"[A-Z][a-z]+(?:[A-Z][a-z]+)?S\d+(.*)$", a.level)
    short = m.group(1) if m and m.group(1) else re.sub(r"\d", "", a.level)
    if getattr(a, "sequence", None):
        return f"SGQ{_seq_tag(a)}{short}"
    uses = getattr(a, "max_uses", 1)
    return f"SG{a.seed}L{int(a.length)}" + (f"U{uses}" if uses > 1 else "") + short


def _seq_tag(a) -> str:
    """Short stable tag of a piece sequence (for names and folders): FNV-1a
    32-bit of the normalized text, as hex (the GUI computes the same)."""
    norm = " ".join(a.sequence.replace(",", " ").split())
    h = 0x811C9DC5
    for b in norm.encode():
        h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return f"{h:08X}"


def _out_dir(a) -> str:
    if getattr(a, "override", False):
        return os.path.join(a.out, f"{a.level}_seed{a.seed}")
    if getattr(a, "sequence", None):
        return os.path.join(a.out, f"{a.level}_seq{_seq_tag(a)}")
    uses = getattr(a, "max_uses", 1)
    return os.path.join(a.out, f"{a.level}_seed{a.seed}_{int(a.length)}m" + (f"_x{uses}" if uses > 1 else ""))


def _build_wp(a, paks: str) -> int:
    """Base-game World Partition level: tile-aligned recombination."""
    from .build import writers as W
    from .build.wp import build_wp_stage
    from .extract.iostore import GameFS
    from .extract.real import load_wp_source
    from .extract.zen import Package
    from .progress import progress
    from .tiles import TileGrid, TileSearchConfig, find_cuts, make_ranges, tile_search

    progress(0.0, "reading the game files")
    fs = GameFS(paks, include_mods=False)
    ws = load_wp_source(fs, a.level)
    progress(0.08, "searching a layout")
    st = ws.stage
    meshes, *_ = W.read_baked_raw(Package(fs.read(ws.baked)))
    grid = TileGrid.from_baked(meshes)
    cuts = find_cuts(st, grid, 2.0)
    cfg = TileSearchConfig(target_length=a.length, seed=a.seed, max_uses=max(1, getattr(a, "max_uses", 1)))
    ranges = make_ranges(st, grid, cuts, 2.0, cfg.min_len, cfg.max_len)
    print(
        f"{st.name}: {(len(st.points) - 1) * 2 / 1000:.2f} km, {len(grid.tiles)} tiles of {grid.cell:.0f} m, "
        f"{len(cuts) - 2} usable tile crossings, {len(ranges)} candidate pieces"
    )
    t = time.time()
    if getattr(a, "sequence", None):
        # a given piece sequence instead of the search (e.g. one hard piece many
        # times for practice): as many uses as the sequence needs
        from .tiles import parse_sequence, sequence_state

        cfg.max_uses = 99
        try:
            best, warns = sequence_state(st, ranges, cuts, cfg, parse_sequence(a.sequence))
        except ValueError as e:
            print(f"error: {e}")
            return 1
        for w in warns:
            print("warning:", w)
    else:
        best = tile_search(st, ranges, cuts, cfg)
    if best is None:
        src_km = (len(st.points) - 1) * 2 / 1000
        uses = max(1, getattr(a, "max_uses", 1))
        hint = (
            f"{a.level} has only {src_km:.1f} km of road and every section is used at most {uses}x, "
            f"so stages are limited to about {0.9 * uses * src_km:.1f} km; use a lower --length, a higher "
            f"--max-uses or a bigger level"
            if a.length > 0.9 * uses * src_km * 1000
            else "try another --seed or a slightly different --length"
        )
        print(f"no layout found: {hint}")
        return 1
    # smooth the joins (shear both sides of each kinked join) before the layout
    # is built; the pacenotes around smoothed joins are regenerated
    from .tiles import smooth_layout_points

    shears, smoothed = smooth_layout_points(best, cfg)
    regen = {k: reach + 40.0 for k, reach in smoothed.items()}
    layout = build_layout(best, 2.0, regen)
    n_sm = len(regen)
    if n_sm:
        print(f"smoothed {n_sm} join(s); pacenotes there regenerated from the new road")
    if not getattr(a, "sequence", None) and abs(best.length - a.length) > cfg.length_tolerance * a.length:
        print(
            f"note: {a.length / 1000:.1f} km is not achievable from {a.level}; using the closest: "
            f"{best.length / 1000:.2f} km"
        )
    order = " ".join(p.segment_id.split("#")[1] + ("r" if p.reversed else "") for p in layout.pieces)
    print(
        f"layout {layout.length / 1000:.2f} km in {time.time() - t:.1f}s: pieces {order}, "
        f"kinks {', '.join(f'{k:+.0f}' for k in best.kinks)} deg, {len(layout.pacenotes)} calls"
    )
    out = _out_dir(a)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "layout.json"), "w") as f:
        f.write(to_json(layout, {"seed": a.seed, "source": "game", "level": a.level}))
    plot_layout(
        layout, os.path.join(out, "layout.png"), f"{a.level} seed {a.seed}: {layout.length / 1000:.2f} km ({order})", 40
    )
    import ast
    import dataclasses
    from .build.wp import WPConfig

    cfg = WPConfig(
        debug_spawn_s=a.spawn_at,
        debug_no_clip=a.no_clip,
        debug_offset_zero=a.offset_zero,
        debug_offset=None if a.offset is None else tuple(float(v) for v in a.offset.split(",")),
    )
    if not getattr(a, "override", False):
        # its own level and menu entry next to the original stage
        cfg.own_level = _own_name(a)
        cfg.display_name = (
            f"{_pretty(a.level)} · custom {_seq_tag(a)} · {layout.length / 1000:.1f} km"
            if getattr(a, "sequence", None)
            else f"{_pretty(a.level)} · seed {a.seed} · {layout.length / 1000:.1f} km"
        )
        # how to build it again (stored with the installed stage: regenerate / load settings)
        cfg.build_params = {
            "level": a.level,
            "length": a.length,
            "seed": a.seed,
            "max_uses": getattr(a, "max_uses", 1),
            "sequence": getattr(a, "sequence", None) or "",
            "forest": not getattr(a, "no_forest", False),
            "grass": not getattr(a, "no_grass", False),
            "foliage": not getattr(a, "no_foliage", False),
        }
    if getattr(a, "no_forest", False):
        cfg.forest = "hide"
    if getattr(a, "no_grass", False):
        cfg.grass = "hide"
    if getattr(a, "no_foliage", False):
        cfg.foliage_tool = "hide"
    fields = {f.name for f in dataclasses.fields(WPConfig)}
    for kv in getattr(a, "cfg", []) or []:
        k, _, v = kv.partition("=")
        if k not in fields:
            raise SystemExit(f"--cfg: unknown option {k!r}; known: {', '.join(sorted(fields))}")
        try:
            val = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            val = v
        setattr(cfg, k, val)
        print(f"  option {k} = {val!r}")
    cfg.join_shears = shears
    rep = build_wp_stage(layout, best, cuts, grid, ws, paks, out, cfg)
    progress(0.99, "finishing")
    try:
        from .build.preview import render_collision

        fwd = next(p for p, k in ws.route_cells.items() if k == "fwd")
        render_collision(
            rep.out_files[0],
            ws.baked,
            fwd,
            os.path.join(out, "collision.png"),
            f"{a.level} seed {a.seed}: built collision (start o, finish #)",
        )
    except (ImportError, KeyError):
        pass  # no matplotlib (release build) / own level (renamed paths)
    print(
        f"placed at offset {tuple(float(x) for x in rep.offset)} m; {rep.packages} packages rewritten; moved "
        + ", ".join(f"{k} {v}" for k, v in rep.moved.most_common(12))
    )
    for r in rep.relocated:
        print(f"  relocated {r[0][:40]} {r[1]} m -> {r[2]} m")
    dropped = sum(c[2] for c in rep.clipped)
    total = sum(c[3] for c in rep.clipped)
    print(
        f"tile meshes clipped: {len(rep.clipped)} ({dropped}/{total} triangles hidden); "
        f"baked: {rep.baked_meshes} meshes, {rep.baked_tris} triangles; pacenotes {rep.notes}"
    )
    for c in rep.checks:
        print("  check:", c)
    for f_ in rep.out_files:
        print("  wrote", f_)
    return 0


def _stop_game() -> None:
    """ACR keeps the mod container open; close it before overwriting."""
    import subprocess

    out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq acr.exe"], capture_output=True, text=True).stdout
    if "acr.exe" in out.lower():
        print("closing ACR (it holds the installed container open)")
        subprocess.run(["taskkill", "/F", "/IM", "acr.exe"], capture_output=True)
        time.sleep(3)


def _build(a) -> int:
    rc = _build_impl(a)
    if rc == 0 and (a.install or a.launch):
        from .progress import progress

        progress(1.0, "installing into the game")
        _stop_game()
        a.build_dir = _out_dir(a)
        rc = _install(a)
        if rc == 0 and a.launch:
            os.startfile("steam://rungameid/3917090")
            print("launched ACR")
    return rc


def _build_impl(a) -> int:
    from .build.stage import build_stage
    from .extract.iostore import GameFS
    from .extract.real import DEFAULT_PAKS, load_location

    paks = a.paks or DEFAULT_PAKS
    fs = GameFS(paks, include_mods=False)
    if any(p.startswith(f"acr/Content/Levels/{a.level}/_Generated_/") for p in fs.index):
        return _build_wp(a, paks)
    stages = [s for s in load_location(a.level, paks) if s.name == a.level]
    if not stages:
        print(f"level {a.level!r} not found or not readable (try `list`)")
        return 1
    st = stages[0]
    segs = cut_stage(st, BUILD_CUT)
    print(f"{st.name}: {(len(st.points) - 1) * 2 / 1000:.2f} km, {len(segs)} segments")
    cfg = SearchConfig(
        target_length=a.length,
        seed=a.seed,
        corridor_half_width=a.corridor,
        start_segment=segs[0].id,
        end_segment=segs[-1].id,
        novelty=25,
        variety=25,
    )
    t = time.time()
    best = search(segs, cfg)
    if best is None:
        print("no layout found; try another --seed or --length")
        return 1
    layout = build_layout(best, cfg.step)
    order = " ".join(
        p.segment_id.split("#")[1] + ("r" if p.reversed else "") for p in layout.pieces if p.kind == "segment"
    )
    print(
        f"layout {layout.length / 1000:.2f} km in {time.time() - t:.1f}s: segments {order}, "
        f"{layout.synthetic_share:.1%} connector, {len(layout.pacenotes)} calls"
    )
    out = os.path.join(a.out, f"{a.level}_seed{a.seed}")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "layout.json"), "w") as f:
        f.write(to_json(layout, {"seed": a.seed, "source": "game", "level": a.level}))
    plot_layout(
        layout,
        os.path.join(out, "layout.png"),
        f"{a.level} seed {a.seed}: {layout.length / 1000:.2f} km ({order})",
        a.corridor,
    )
    rep = build_stage(layout, st, {s.id: s for s in segs}, a.level, paks, out)
    print(
        f"placed at offset {tuple(float(x) for x in rep.offset)} m; moved {sum(rep.moved.values())} actors: "
        + ", ".join(f"{k} {v}" for k, v in rep.moved.most_common(10))
        + f" ... RVT volumes {rep.moved.get('RVT volume', 0)}"
    )
    for r in rep.relocated:
        print(f"  relocated {r[0][:40]} {r[1]} m -> {r[2]} m")
    print(
        f"baked: {rep.baked_meshes} meshes, {rep.baked_tris} triangles + {rep.connector_tris} connector triangles; "
        f"{rep.notes} pacenotes"
    )
    for c in rep.checks:
        print("  check:", c)
    for f_ in rep.out_files:
        print("  wrote", f_)
    return 0


def _mods_dir(paks: str | None) -> str:
    from .extract.real import DEFAULT_PAKS

    return os.path.normpath(os.path.join(paks or DEFAULT_PAKS, "~mods"))


def _install(a) -> int:
    import glob
    import shutil

    src = glob.glob(os.path.join(a.build_dir, "*_Stagegen_9_P.utoc"))
    if not src:
        print(f"no built container in {a.build_dir}")
        return 1
    mods = _mods_dir(a.paks)
    base = src[0][:-5]
    name = os.path.basename(base)
    # replace an earlier install of the same stage (same container name)
    for p in glob.glob(os.path.join(mods, name + ".*")):
        os.remove(p)
    for ext in (".utoc", ".ucas", ".pak", ".stagegen.json"):
        if os.path.exists(base + ext):
            shutil.copy2(base + ext, mods)
            print("installed", os.path.join(mods, name + ext))
    # the map of the stage, for the GUI's list of installed stages (the game ignores it)
    lay = os.path.join(a.build_dir, "layout.json")
    if os.path.exists(lay) and os.path.exists(base + ".stagegen.json"):
        shutil.copy2(lay, os.path.join(mods, name + ".layout.json"))
    return _refresh_tables(a)


def _refresh_tables(a) -> int:
    from .build.tables import write_tables_container
    from .extract.real import DEFAULT_PAKS

    for n in write_tables_container(os.path.normpath(a.paks or DEFAULT_PAKS)):
        print("menu:", n)
    return 0


def installed_stages(paks: str | None) -> list[str]:
    """Container names of installed generated stages (without the menu tables)."""
    import glob
    from .build.tables import TABLES_PAK

    return sorted(
        os.path.basename(p)[:-5]
        for p in glob.glob(os.path.join(_mods_dir(paks), "*_Stagegen_9_P.utoc"))
        if os.path.basename(p)[:-5] != TABLES_PAK
    )


def _uninstall(a, quiet: bool = False) -> int:
    """Remove one installed stage (a.stage: level name or container name) or all."""
    import glob
    from .build.tables import TABLES_PAK

    want = getattr(a, "stage", None)
    removed = 0
    for name in installed_stages(a.paks):
        if want and want not in (name, name.removeprefix("pakchunk99-").removesuffix("_Stagegen_9_P")):
            continue
        for p in glob.glob(os.path.join(_mods_dir(a.paks), name + ".*")):
            os.remove(p)
            removed += 1
            if not quiet:
                print("removed", p)
    if not want:
        for p in glob.glob(os.path.join(_mods_dir(a.paks), TABLES_PAK + ".*")):
            os.remove(p)
    if not quiet and not removed:
        print("nothing to remove")
    return _refresh_tables(a) if want else 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="acr-stagegen")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="generate a new stage layout")
    src = g.add_mutually_exclusive_group(required=True)
    src.add_argument("--fake", action="store_true", help="use synthetic source stages")
    src.add_argument("--location", help="level name prefix to recombine, e.g. WrcgEs")
    g.add_argument("--paks", help="path to acr/Content/Paks (default: the install this tool lives in)")
    g.add_argument("--length", type=float, default=10_000)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--no-reverse", action="store_true")
    g.add_argument("--corridor", type=float, default=35.0, help="scanned corridor half-width (m)")
    g.add_argument("--out", default="out")
    ls = sub.add_parser("list", help="list stage levels found in the install")
    ls.add_argument("--paks")
    b = sub.add_parser("build", help="generate a stage from one level and build an installable mod container")
    b.add_argument("--level", default="WrcgEsSS1", help="stage level to recombine and replace")
    b.add_argument("--paks")
    b.add_argument("--length", type=float, default=5500)
    b.add_argument(
        "--max-uses",
        type=int,
        default=1,
        help="how often a tile may appear (>1: repeated tiles for stages longer than the source)",
    )
    b.add_argument(
        "--sequence",
        default=None,
        metavar="PIECES",
        help="build exactly these pieces instead of searching, e.g. '0-1 7-11 7-11 7-11 18-19' "
        "(cut indices from `segments`; 'r' = backwards); repeats allowed",
    )
    sg = sub.add_parser("segments", help="list a level's pieces (for --sequence); --json for tools")
    sg.add_argument("--level", required=True)
    sg.add_argument("--paks")
    sg.add_argument("--json", action="store_true")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--corridor", type=float, default=35.0)
    b.add_argument("--out", default="out")
    b.add_argument(
        "--spawn-at", type=float, default=None, help="test builds: spawn the car this many metres along the route"
    )
    b.add_argument("--install", action="store_true", help="install into ~mods after building (closes ACR if running)")
    b.add_argument("--launch", action="store_true", help="install and start ACR through Steam")
    b.add_argument("--no-clip", action="store_true", help="test builds: leave track meshes unclipped")
    b.add_argument("--offset-zero", action="store_true", help="test builds: build on top of the original map")
    b.add_argument("--offset", default=None, help="test builds: placement offset x,y,z in metres")
    for kind, what in (
        ("forest", "forest trees"),
        ("grass", "grass, branches and rocks"),
        ("foliage", "foliage-tool grass"),
    ):
        b.add_argument(f"--no-{kind}", action="store_true", help=f"leave out the {what} (performance)")
    b.add_argument(
        "--cfg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="set any build option (see WPConfig), e.g. --cfg seam_width=40 --cfg filler=False",
    )
    ins = sub.add_parser("install", help="copy a built stage container into acr/Content/Paks/~mods")
    ins.add_argument("build_dir", help="folder written by `build`, e.g. out/WrcgEsSS1_seed0")
    ins.add_argument("--paks")
    uni = sub.add_parser("uninstall", help="remove installed generated stages (all, or --stage NAME)")
    uni.add_argument("--paks")
    uni.add_argument("--stage", help="level name of one installed stage, e.g. SG1L4500HafrenSouth")
    lsi = sub.add_parser("installed", help="list installed generated stages")
    lsi.add_argument("--paks")
    b.add_argument(
        "--override", action="store_true", help="replace the original stage instead of adding a new one (old behaviour)"
    )
    a = ap.parse_args(argv)
    if a.cmd == "install":
        return _install(a)
    if a.cmd == "uninstall":
        _stop_game()
        return _uninstall(a)
    if a.cmd == "installed":
        import json

        for name in installed_stages(a.paks):
            meta = os.path.join(_mods_dir(a.paks), name + ".stagegen.json")
            info = json.load(open(meta, encoding="utf-8")) if os.path.exists(meta) else {}
            print(
                name.removeprefix("pakchunk99-").removesuffix("_Stagegen_9_P"),
                "|",
                info.get("display", "replaces the original stage"),
            )
        return 0

    if a.cmd == "list":
        return _list(a)
    if a.cmd == "segments":
        return _segments(a)
    if a.cmd == "build":
        return _build(a)

    if a.fake:
        stages, cut, source = fake_location(), CutConfig(), "fake"
    else:
        from .extract.real import DEFAULT_PAKS, load_location

        stages = load_location(a.location, a.paks or DEFAULT_PAKS)
        if not stages:
            ap.error(f"no readable levels start with {a.location!r} (try `list`)")
        cut, source = REAL_CUT, "game"
    segments = [s for st in stages for s in cut_stage(st, cut)]
    print(
        f"{len(stages)} source stages ({', '.join(s.name for s in stages)}), {len(segments)} segments "
        f"({sum(s.length for s in segments) / 1000:.1f} km of reusable road)"
    )

    cfg = SearchConfig(
        target_length=a.length, seed=a.seed, allow_reversed=not a.no_reverse, corridor_half_width=a.corridor
    )
    t = time.time()
    best = search(segments, cfg)
    if best is None:
        print("no layout found; try a shorter --length, smaller --corridor, or another --seed")
        return 1
    layout = build_layout(best, cfg.step)
    print(
        f"found {layout.length / 1000:.2f} km layout in {time.time() - t:.1f}s; "
        f"{sum(p.kind == 'segment' for p in layout.pieces)} segments, "
        f"{layout.synthetic_share:.1%} generated connector road, "
        f"{len(layout.pacenotes)} calls "
        f"({sum(n.origin == 'original' for n in layout.pacenotes)} carried over)"
    )

    clear = self_clear(layout.centreline, 2 * a.corridor * 0.95, cfg.step)
    print(f"check: corridors clear={clear}, max grade={max_abs_grade(layout.centreline[::5]):.1%}")

    os.makedirs(a.out, exist_ok=True)
    tag = "fake" if a.fake else a.location
    base = os.path.join(a.out, f"stage_{tag}_seed{a.seed}")
    with open(base + ".json", "w") as f:
        f.write(
            to_json(
                layout,
                {
                    "seed": a.seed,
                    "source": source,
                    "location": stages[0].location,
                    "surface": stages[0].surface,
                    "source_stages": [s.name for s in stages],
                },
            )
        )
    plot_layout(
        layout,
        base + ".png",
        f"Generated stage ({tag}), seed {a.seed}: {layout.length / 1000:.2f} km, "
        f"{layout.synthetic_share:.0%} connector",
        a.corridor,
    )
    print(f"wrote {base}.json, {base}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
