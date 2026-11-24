# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Drive Assetto Corsa Rally with a virtual Xbox pad (ViGEmBus + vgamepad).

    py tools/acr_harness.py --steps "launch,wait45,RB,wait2,A,shot:menu" --out shots

Steps (comma separated):
    launch            kill ACR if running and start it through Steam
    kill              stop ACR
    install:<dir>     copy a built stage container into ~mods (ACR must not run)
    waitN             sleep N seconds
    A B X Y LB RB START BACK UP DOWN LEFT RIGHT   tap a button (also e.g. A*3)
    gas:N             hold the right trigger N seconds
    shot:<name>       save a small JPEG (640x360) to --out/<name>.jpg
Screenshots are small on purpose: one image per test is the budget.
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import time

APP_ID = "3917090"
BUTTONS = {
    "A": "XUSB_GAMEPAD_A", "B": "XUSB_GAMEPAD_B", "X": "XUSB_GAMEPAD_X", "Y": "XUSB_GAMEPAD_Y",
    "LB": "XUSB_GAMEPAD_LEFT_SHOULDER", "RB": "XUSB_GAMEPAD_RIGHT_SHOULDER",
    "START": "XUSB_GAMEPAD_START", "BACK": "XUSB_GAMEPAD_BACK",
    "UP": "XUSB_GAMEPAD_DPAD_UP", "DOWN": "XUSB_GAMEPAD_DPAD_DOWN",
    "LEFT": "XUSB_GAMEPAD_DPAD_LEFT", "RIGHT": "XUSB_GAMEPAD_DPAD_RIGHT",
}


def acr_running() -> bool:
    out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq acr.exe"], capture_output=True, text=True).stdout
    return "acr.exe" in out.lower()


def kill():
    subprocess.run(["taskkill", "/F", "/IM", "acr.exe"], capture_output=True)
    for _ in range(20):
        if not acr_running():
            return
        time.sleep(0.5)


def focus():
    import ctypes
    user32 = ctypes.windll.user32
    hwnd = user32.FindWindowW(None, "Assetto Corsa Rally")
    if hwnd:
        user32.keybd_event(0x12, 0, 0, 0); user32.keybd_event(0x12, 0, 2, 0)   # Alt tap unlocks focus
        user32.ShowWindow(hwnd, 9)
        user32.SetForegroundWindow(hwnd)


def shot(path: str, w: int = 640):
    from PIL import ImageGrab
    img = ImageGrab.grab()
    img.thumbnail((w, w * 9 // 16))
    img.convert("RGB").save(path, quality=70)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", required=True)
    ap.add_argument("--out", default="shots")
    ap.add_argument("--mods", default=os.path.join(os.path.dirname(__file__), "..", "..", "acr", "Content", "Paks",
                                                   "~mods"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import vgamepad as vg
    pad = vg.VX360Gamepad()
    time.sleep(3.0)                  # the game ignores a freshly connected pad briefly
    for step in [s.strip() for s in a.steps.split(",") if s.strip()]:
        name, _, arg = step.partition(":")
        if name == "launch":
            kill()
            os.startfile(f"steam://rungameid/{APP_ID}")
            for _ in range(60):
                if acr_running():
                    break
                time.sleep(1)
        elif name == "kill":
            kill()
        elif name == "install":
            kill()
            for p in glob.glob(os.path.join(a.mods, "*_Stagegen_9_P.*")):
                os.remove(p)
            for p in glob.glob(os.path.join(arg, "*_Stagegen_9_P.*")):
                shutil.copy2(p, a.mods)
        elif name.startswith("wait"):
            time.sleep(float(name[4:] or 1))
        elif name == "gas":
            focus()
            pad.right_trigger(value=255); pad.update()
            time.sleep(float(arg or 1))
            pad.right_trigger(value=0); pad.update()
        elif name == "shot":
            nm, _, w = (arg or "shot").partition(":")
            shot(os.path.join(a.out, nm + ".jpg"), int(w or 640))
        else:
            btn, _, n = name.partition("*")
            if btn not in BUTTONS:
                raise SystemExit(f"unknown step {step}")
            focus()
            for _ in range(int(n or 1)):
                b = getattr(vg.XUSB_BUTTON, BUTTONS[btn])
                pad.press_button(button=b); pad.update(); time.sleep(0.25)
                pad.release_button(button=b); pad.update(); time.sleep(0.6)
        print("done", step, flush=True)


if __name__ == "__main__":
    main()
