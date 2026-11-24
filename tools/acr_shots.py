# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Screenshot a built stage at fixed points via the UE4SS StageTest mod.

    py tools/acr_shots.py --build out/WalesS4HafrenSouth_seed1 [--every 600] [--install-mod]

Writes the test points (start, every --every metres, both sides of every
join, finish) for the mod, waits for its "[StageTest] at <label>" log lines
while you (or the harness) start the session, grabs a screenshot 1 s after
each, and saves ONE contact sheet (sheet.jpg) with the labels.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.normpath(os.path.join(HERE, "..", "..", "acr", "Binaries", "Win64"))
MODS = os.path.join(BIN, "ue4ss", "Mods")
LOG = os.path.join(BIN, "ue4ss", "UE4SS.log")


def install_mod():
    dst = os.path.join(MODS, "StageTest")
    shutil.copytree(os.path.join(HERE, "ue4ss", "StageTest"), dst, dirs_exist_ok=True)
    mods_txt = os.path.join(MODS, "mods.txt")
    txt = open(mods_txt, encoding="utf-8", errors="ignore").read()
    if "StageTest" not in txt:
        open(mods_txt, "a").write("\nStageTest : 1\n")
    return dst


def points_for(build_dir: str, every: float):
    sys.path.insert(0, HERE)
    from acr_autopilot import route_from_container
    utoc = glob.glob(os.path.join(build_dir, "*_Stagegen_9_P.utoc"))[0]
    R = route_from_container(utoc)
    S = np.r_[0, np.cumsum(np.linalg.norm(np.diff(R[:, :2], axis=0), axis=1))]
    marks = [("start", 20.0)]
    marks += [(f"{int(s)}m", float(s)) for s in np.arange(every, S[-1] - 100, every)]
    lay = os.path.join(build_dir, "layout.json")
    if os.path.exists(lay):
        for p in json.load(open(lay))["pieces"][1:]:
            j = p["s_start"]
            marks += [(f"join{int(j)}-30", j - 30), (f"join{int(j)}+30", j + 30)]
    marks += [("finish", S[-1] - 30)]
    marks.sort(key=lambda m: m[1])
    out = []
    for lab, s in marks:
        i = int(np.clip(np.searchsorted(S, s), 1, len(R) - 2))
        d = R[i + 1] - R[i - 1]
        out.append((lab, R[i, 0] * 100, R[i, 1] * 100, R[i, 2] * 100, math.degrees(math.atan2(d[1], d[0]))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True)
    ap.add_argument("--every", type=float, default=250)
    ap.add_argument("--out", default=os.path.join(os.environ.get("TEMP", "."), "acrshots", "sheet"))
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--prepare", action="store_true", help="only install the mod and write the points")
    a = ap.parse_args()
    mod = install_mod()
    pts = points_for(a.build, a.every)
    with open(os.path.join(mod, "points.txt"), "w") as f:
        for p in pts:
            f.write("%s %.1f %.1f %.1f %.1f\n" % p)
    print(f"{len(pts)} points written", flush=True)
    go = os.path.join(mod, "go.txt")
    if os.path.exists(go):
        os.remove(go)
    if a.prepare:
        return
    os.makedirs(a.out, exist_ok=True)
    for old in glob.glob(os.path.join(a.out, "*.jpg")):
        os.remove(old)
    sys.path.insert(0, HERE)
    from acr_harness import shot
    pos = os.path.getsize(LOG) if os.path.exists(LOG) else 0
    open(go, "w").close()            # tour starts now (the car must have driven off the start line)
    got, t0 = [], time.time()
    while time.time() - t0 < a.timeout:
        time.sleep(0.5)
        if not os.path.exists(LOG):
            continue
        if os.path.getsize(LOG) < pos:
            pos = 0                                       # log restarted with the game
        with open(LOG, "rb") as f:
            f.seek(pos)
            new = f.read().decode("utf-8", "ignore")
            pos = f.tell()
        for part in new.split("[StageTest] ")[1:]:
            if part.startswith("at "):
                lab = part[3:].split(" ")[0]
                time.sleep(1.0)
                path = os.path.join(a.out, f"{len(got):02d}_{lab}.jpg")
                shot(path, 480)
                got.append((lab, path))
                print("shot", lab, part[3:].split("\n")[0][:60], flush=True)
            elif part.startswith("done"):
                t0 = -1e9
            elif part.startswith("error"):
                print(part[:200], flush=True)
        if t0 < 0:
            break
    if not got:
        print("no shots"); return
    from PIL import Image, ImageDraw
    cols = 5
    w, h = 384, 216
    sheet = Image.new("RGB", (cols * w, ((len(got) + cols - 1) // cols) * h), "black")
    for k, (lab, path) in enumerate(got):
        im = Image.open(path).resize((w, h))
        ImageDraw.Draw(im).text((8, 8), lab, fill=(255, 255, 0))
        sheet.paste(im, ((k % cols) * w, (k // cols) * h))
    sp = os.path.join(a.out, "sheet.jpg")
    sheet.save(sp, quality=70)
    print("sheet", sp)


if __name__ == "__main__":
    main()
