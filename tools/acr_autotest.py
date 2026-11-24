# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Fully automatic in-game test of a built stage: install, launch ACR, drive the
menus with a virtual pad (waiting for each screen change reported by the
StageTest UE4SS mod in ui.txt), let the camera tour run, write one contact sheet.

    py tools/acr_autotest.py --build out/WalesS4HafrenSouth_seed1

Menu path (Free Practice remembers the last stage and car):
    splash --A--> main menu --RB--> Racing tab --DOWN,DOWN,A--> Free Practice
    --DOWN,DOWN,A (Start Session)--> loading --> service park --A--> loading --> car
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
MOD = os.path.normpath(os.path.join(HERE, "..", "..", "acr", "Binaries", "Win64", "ue4ss", "Mods", "StageTest"))
UI = os.path.join(MOD, "ui.txt")


def ui() -> str:
    try:
        return open(UI).read().strip()
    except OSError:
        return ""


def wait_ui(pred, timeout: float, what: str) -> str:
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = ui()
        if pred(s):
            return s
        time.sleep(0.5)
    raise SystemExit(f"timeout waiting for {what}; screen: {ui()!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True)
    ap.add_argument("--every", type=float, default=250)
    ap.add_argument("--drive", action="store_true", help="autopilot drive-through instead of the camera tour")
    ap.add_argument("--stop-at-fp", action="store_true", help="stop on the Free Practice screen (menu mapping)")
    ap.add_argument("--from-fp", action="store_true", help="game already on the Free Practice screen: just run")
    ap.add_argument("--select", default=None, metavar="R,T",
                    help="pick rally #R (Alsace=1 ... Wales=6 with one mod rally first) and stage tile #T first")
    a = ap.parse_args()
    import vgamepad as vg
    from acr_harness import BUTTONS, focus, kill
    py = sys.executable

    # close ACR, install the stage, write the tour points (the mod reads them on entering the car), launch
    if not a.from_fp:
        kill()
        subprocess.run([py, "-m", "stagegen.cli", "install", a.build], cwd=os.path.join(HERE, ".."), check=True)
    subprocess.run([py, os.path.join(HERE, "acr_shots.py"), "--build", a.build, "--every", str(a.every), "--prepare"],
                   check=True)
    if not a.from_fp:
        if os.path.exists(UI):
            os.remove(UI)
        os.startfile("steam://rungameid/3917090")
    pad = vg.VX360Gamepad()
    if a.from_fp:
        time.sleep(3)                                    # the game ignores a freshly connected pad briefly

    def press(name, n=1):
        focus()
        b = getattr(vg.XUSB_BUTTON, BUTTONS[name])
        for _ in range(n):
            pad.press_button(button=b); pad.update()
            time.sleep(0.06 if "DPAD" in BUTTONS[name] else 0.25)   # a longer D-pad hold counts twice in carousels
            pad.release_button(button=b); pad.update(); time.sleep(0.8)

    title = lambda s: "WB_SplashScreen_C" in s.split(",")      # 'press any button' (not the logo intro)
    if a.from_fp:
        if a.select:
            select_stage(a.select, press)
        return after_fp(a, py, press, pad, ui())
    s = None
    for attempt in range(2):                             # the game occasionally dies during the intro
        try:
            # the title can be skipped (a key press, or the game goes straight on): the main menu counts too
            s = wait_ui(lambda s: title(s) or "WB_MainMenu_C" in s, 240, "title screen")
            break
        except SystemExit:
            alive = "acr.exe" in subprocess.run(["tasklist", "/FI", "IMAGENAME eq acr.exe"], capture_output=True,
                                                text=True).stdout.lower()
            if alive or attempt:
                raise
            print("ACR died during the intro; relaunching", flush=True)
            if os.path.exists(UI):
                os.remove(UI)
            os.startfile("steam://rungameid/3917090")
    print("title:", s, flush=True)
    t0 = time.time()
    while title(ui()):
        if time.time() - t0 > 60:
            raise SystemExit("title screen does not react")
        time.sleep(2)
        press("A")
    main_menu = wait_ui(lambda s: "WB_MainMenu_C" in s, 90, "main menu")
    print("main menu:", main_menu, flush=True)
    fp = None
    for attempt in range(4):                             # Racing tab -> Free Practice, verified
        time.sleep(4)
        press("RB"); time.sleep(2)
        # the tab remembers its last item: go to the top first; other counts as fallback
        press("UP", 6)
        press("DOWN", (2, 1, 3, 0)[attempt]); press("A")
        try:
            fp = wait_ui(lambda s: "WB_TimeAttackHubLayout_C" in s, 10, "free practice screen")
            break
        except SystemExit:
            print("not free practice, backing out:", ui(), flush=True)
            for _ in range(3):
                press("B"); time.sleep(1)
            wait_ui(lambda s: "WB_MainMenu_C" in s, 30, "main menu")
    if fp is None:
        raise SystemExit("could not reach free practice")
    print("free practice:", fp, flush=True)
    if a.select:
        select_stage(a.select, press)
        fp = ui()
    if a.stop_at_fp:
        return
    return after_fp(a, py, press, pad, fp)


def select_stage(spec: str, press):
    """Free Practice: Change Rally Stage -> rally #R in the list -> stage tile #T
    -> first variant -> SELECT. spec "R,T" (rally order: [mods], Alsace, Greece,
    Livigno, Monte Carlo, Sweden, Wales; tiles left to right)."""
    r, t = (int(v) for v in spec.split(","))
    time.sleep(2)
    press("UP", 2); press("LEFT"); press("A")
    wait_ui(lambda s: "WB_TrackPage_C" in s, 20, "rally selection")
    time.sleep(2.0)

    def slow(name, n):                       # the carousels drop presses while they scroll
        for _ in range(n):
            press(name); time.sleep(0.7)
    slow("LEFT", 12); slow("RIGHT", r); press("A")
    time.sleep(2.0)
    slow("LEFT", 4); slow("RIGHT", t); press("A")
    time.sleep(1.0)
    press("DOWN"); press("A")
    s = wait_ui(lambda s: "WB_TimeAttackHubLayout_C" in s, 20, "free practice after selecting the stage")
    print(f"selected rally {r}, stage {t}", flush=True)
    return s


def after_fp(a, py, press, pad, fp):
    """From the Free Practice screen (cursor on Change Rally Stage): start the
    session, get into the car, drive off the line, run the tour."""
    from acr_harness import focus
    time.sleep(2)
    press("DOWN", 2); press("A")                         # Start Session
    t0 = time.time()
    while "IN_CAR" not in ui():                          # service park: A starts the stage
        if time.time() - t0 > 300:
            raise SystemExit(f"never got into the car; screen: {ui()!r}")
        time.sleep(8)
        if "IN_CAR" not in ui() and ui() and ui() != fp:
            press("A")
    print("in car:", ui(), flush=True)
    # drive off the start line (the start sequence holds the camera until then)
    import mmap
    import struct
    phys = mmap.mmap(-1, 1024, "Local\\acpmf_physics", access=mmap.ACCESS_READ)
    speed = lambda: struct.unpack_from("<f", phys, 28)[0]          # km/h
    gear = lambda: struct.unpack_from("<i", phys, 16)[0]           # 0 R, 1 N, 2 1st
    # a short fixed burst: the shared-memory telemetry is sometimes stale (speed
    # stays 0), and "throttle until moving" then drove the car into the trees
    time.sleep(6)
    press("A")                                                     # manual gearbox: up into 1st
    focus()
    pad.right_trigger_float(value_float=0.5); pad.update()
    time.sleep(2.0)
    pad.right_trigger_float(value_float=0.0); pad.left_trigger_float(value_float=1.0); pad.update()
    time.sleep(3)
    pad.left_trigger_float(value_float=0.0); pad.update()
    print(f"drove off (telemetry: {speed():.0f} km/h, gear {gear()})", flush=True)
    del pad
    if a.drive:                                          # autopilot drive-through instead of the camera tour
        import glob
        time.sleep(8)
        utoc = glob.glob(os.path.join(a.build, "*_Stagegen_9_P.utoc"))[0]
        out = os.path.join(os.environ.get("TEMP", "."), "acrshots", "drive")
        subprocess.run([py, os.path.join(HERE, "acr_autopilot.py"), "--utoc", utoc, "--build", a.build, "--out", out],
                       check=True)
        return
    subprocess.run([py, os.path.join(HERE, "acr_shots.py"), "--build", a.build, "--every", str(a.every),
                    "--timeout", "400"], check=True)


if __name__ == "__main__":
    main()
