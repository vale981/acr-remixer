# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Press a gamepad step sequence and log the active CommonUI screens (ui2.txt
from the StageTest live reporter) after each step. Used to map the menu path."""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import vgamepad as vg
from acr_harness import BUTTONS, focus

UI = os.path.join(
    os.path.dirname(__file__), "..", "..", "acr", "Binaries", "Win64", "ue4ss", "Mods", "StageTest", "ui.txt"
)


def ui():
    try:
        return open(UI).read().strip()
    except OSError:
        return "?"


def main(steps):
    pad = vg.VX360Gamepad()
    time.sleep(3)
    print("start:", ui(), flush=True)
    for step in steps.split(","):
        name, _, n = step.partition("*")
        if name.startswith("wait"):
            time.sleep(float(name[4:]))
        else:
            focus()
            b = getattr(vg.XUSB_BUTTON, BUTTONS[name])
            for _ in range(int(n or 1)):
                pad.press_button(button=b)
                pad.update()
                time.sleep(0.06 if "DPAD" in BUTTONS[name] else 0.25)  # a longer D-pad hold counts twice in carousels
                pad.release_button(button=b)
                pad.update()
                time.sleep(0.7)
            time.sleep(2.0)
        print(f"{step:8s} -> {ui()}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
