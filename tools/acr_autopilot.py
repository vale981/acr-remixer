# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Drive a generated stage in ACR along its own CenterSpline (virtual pad +
Kunos shared-memory telemetry) and report how far the car got.

    py tools/acr_autopilot.py --utoc out/X/pakchunk99-X_Stagegen_9_P.utoc --shots 700,1300 --out shots

Telemetry (AC layout): acpmf_graphics @256 = car position (x, height, y) in
metres = Unreal world / 100; acpmf_physics @28 speed km/h, @32 velocity
(x, height, y) m/s. Output: one log line every 100 m (progress, offset from the
centreline, speed) plus small JPEGs at the requested distances. Stops at the
finish, when stuck, or when the car falls (height drops far below the road).
"""
from __future__ import annotations

import argparse
import math
import mmap
import os
import struct
import time

import numpy as np


def telemetry():
    g = mmap.mmap(-1, 1024, "Local\\acpmf_graphics", access=mmap.ACCESS_READ)
    p = mmap.mmap(-1, 1024, "Local\\acpmf_physics", access=mmap.ACCESS_READ)

    def read():
        x, h, y = struct.unpack_from("<3f", g, 256)
        spd, = struct.unpack_from("<f", p, 28)
        vx, vh, vy = struct.unpack_from("<3f", p, 32)
        head, = struct.unpack_from("<f", p, 208)
        return np.array([x, y, h]), spd / 3.6, np.array([vx, vy]), head
    return read


def route_from_container(utoc: str) -> np.ndarray:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from stagegen.extract.iostore import Container
    from stagegen.extract.splines import level_splines
    from stagegen.extract.zen import Package
    from stagegen.geometry import resample
    c = Container(utoc)
    best = None
    for path in c.files:
        if "/_Generated_/" in path and path.endswith(".umap"):
            d = c.read(path)
            if b"CenterSpline" in d:
                sp = level_splines(Package(d))
                if "CenterSpline" in sp:
                    cand = resample(sp["CenterSpline"], 2.0)
                    best = best if best is not None and len(best) >= len(cand) else cand
                    if path.endswith("22VG0QIP9ATUQI633K86KWPFP.umap") or "fwd" in path:
                        best = cand
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--utoc", required=True)
    ap.add_argument("--route-cell", default=None, help="umap path of the forward route cell")
    ap.add_argument("--shots", default="")
    ap.add_argument("--out", default="shots")
    ap.add_argument("--max-speed", type=float, default=50.0, help="km/h")
    ap.add_argument("--lat-acc", type=float, default=2.5, help="m/s^2 allowed in corners")
    ap.add_argument("--steer-gain", type=float, default=1.1)
    ap.add_argument("--build", default=None, help="build dir: shots every 600 m + around each join, sheet at the end")
    ap.add_argument("--steer-sign", type=float, default=1.0)
    ap.add_argument("--stop-at", type=float, default=1e9)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from acr_harness import focus, shot
    import vgamepad as vg

    R = route_from_container(a.utoc)
    seg = np.linalg.norm(np.diff(R[:, :2], axis=0), axis=1)
    S = np.r_[0, np.cumsum(seg)]
    hd = np.unwrap(np.arctan2(np.gradient(R[:, 1]), np.gradient(R[:, 0])))
    curv = np.abs(np.gradient(hd)) / 2.0                       # 1/m on the 2 m resample
    shots = sorted(float(x) for x in a.shots.split(",") if x)
    if a.build:
        import json
        marks = list(np.arange(300, S[-1] - 100, 600))
        lay = os.path.join(a.build, "layout.json")
        if os.path.exists(lay):
            for p in json.load(open(lay))["pieces"][1:]:
                marks += [p["s_start"] - 25, p["s_start"] + 25]
        shots = sorted(set(round(m) for m in marks if 0 < m < S[-1] - 20))
    taken = []
    read = telemetry()
    pad = vg.VX360Gamepad()
    time.sleep(3)
    focus()

    def apply(steer, gas, brake):
        pad.left_joystick_float(x_value_float=float(np.clip(steer, -1, 1)), y_value_float=0.0)
        pad.right_trigger_float(value_float=float(np.clip(gas, 0, 1)))
        pad.left_trigger_float(value_float=float(np.clip(brake, 0, 1)))
        pad.update()

    phys = mmap.mmap(-1, 1024, "Local\\acpmf_physics", access=mmap.ACCESS_READ)
    gear = lambda: struct.unpack_from("<i", phys, 16)[0]            # 0 = R, 1 = N, 2 = 1st ...
    rpm = lambda: struct.unpack_from("<i", phys, 20)[0]
    UP, DOWN = vg.XUSB_BUTTON.XUSB_GAMEPAD_A, vg.XUSB_BUTTON.XUSB_GAMEPAD_X   # manual gearbox

    def tap(b):
        pad.press_button(button=b); pad.update(); time.sleep(0.15)
        pad.release_button(button=b); pad.update()

    for _ in range(4):                                               # into 1st gear
        if gear() >= 2:
            break
        tap(UP if gear() < 2 else DOWN)
        time.sleep(0.5)
    last_shift = 0.0
    # start procedure: hold the throttle
    apply(0, 1, 0)
    time.sleep(4)
    last_log, last_move_t, last_pos = -1e9, time.time(), None
    i_prev = None
    err_prev, t_prev = 0.0, time.time()
    off_since, recoveries, started = None, 0, False

    def recover(err_side):
        """Game respawn (hold Y puts the car back on the road, upright, in neutral), then 1st gear."""
        apply(0, 0, 0); time.sleep(0.5)
        pad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_Y); pad.update(); time.sleep(4)
        pad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_Y); pad.update(); time.sleep(3)
        for _ in range(6):
            if gear() >= 2:
                break
            tap(UP); time.sleep(0.4)

    while True:
        pos, v, vel, head = read()
        d = np.linalg.norm(R[:, :2] - pos[:2], axis=1)
        if i_prev is not None:                                   # search near the previous index (no shortcuts)
            lo, hi = max(0, i_prev - 30), min(len(R), i_prev + 120)
            i = lo + int(np.argmin(d[lo:hi]))
        else:
            i = int(np.argmin(d))
        i_prev = i
        s, off = S[i], d[i]
        # heading of travel: velocity when moving, telemetry heading (+90 deg) otherwise
        yaw = math.atan2(vel[1], vel[0]) if v > 2 else head + math.pi / 2
        look = 12 + 0.9 * v
        j = min(int(np.searchsorted(S, s + look)), len(R) - 1)
        tgt = R[j, :2] - pos[:2]
        err = (math.atan2(tgt[1], tgt[0]) - yaw + math.pi) % (2 * math.pi) - math.pi
        started = started or v > 5                                   # countdown over, car rolling
        now_t = time.time()
        derr = (err - err_prev) / max(now_t - t_prev, 1e-3)
        err_prev, t_prev = err, now_t
        steer = a.steer_sign * (a.steer_gain * err + 0.15 * derr)
        k_ahead = curv[i: min(len(R), i + int((20 + 2.5 * v) / 2))].max(initial=1e-4)
        v_tgt = min(a.max_speed / 3.6, math.sqrt(a.lat_acc / max(k_ahead, 1e-4)))
        gas = float(np.clip(0.35 + 0.12 * (v_tgt - v), 0.0, 0.85))
        brake = min(1.0, (v - v_tgt) / 5) if v > v_tgt + 2 else 0.0
        # recovery: far off the line or not moving for a while -> back out and retry
        bad = started and (off > 12 or (time.time() - last_move_t > 5))
        off_since = (off_since or time.time()) if bad else None
        if off_since and time.time() - off_since > 3 and recoveries < 15:
            recoveries += 1
            print(f"recovering ({recoveries}) at s={s:.0f} m, off={off:.1f} m", flush=True)
            recover(err)
            off_since, last_move_t = None, time.time()
            continue
        apply(steer, gas, brake)
        now = time.time()
        if now - last_shift > 1.0:                                   # simple automatic gearbox
            g, r = gear(), rpm()
            if g >= 2 and r > 6800 and g < 7:
                tap(UP); last_shift = now
            elif g > 2 and r < 3200:
                tap(DOWN); last_shift = now
            elif g < 2:
                tap(UP); last_shift = now
        if shots and s >= shots[0]:
            p_ = os.path.join(a.out, f"at{int(shots[0]):05d}.jpg")
            shot(p_, 480)
            taken.append((f"{int(shots[0])} m (off {off:.0f} m)", p_))
            print(f"shot at {s:.0f} m", flush=True)
            shots.pop(0)
        if s - last_log >= 100:
            print(f"s={s:6.0f} m  off={off:4.1f} m  v={v*3.6:5.1f} km/h  z={pos[2]:6.1f} (road {R[i,2]:6.1f})",
                  flush=True)
            last_log = s
        if last_pos is None or np.linalg.norm(pos[:2] - last_pos) > 3:
            last_pos, last_move_t = pos[:2].copy(), time.time()
        if time.time() - last_move_t > 12 and recoveries >= 15:
            print(f"STUCK at s={s:.0f} m (offset {off:.1f} m)"); break
        if pos[2] < R[i, 2] - 15:
            print(f"FELL at s={s:.0f} m: car height {pos[2]:.1f}, road {R[i,2]:.1f}"); break
        if s >= min(S[-1] - 5, a.stop_at):
            print(f"REACHED s={s:.0f} m of {S[-1]:.0f} m"); break
        time.sleep(0.05)
    apply(0, 0, 1)
    time.sleep(2)
    endp = os.path.join(a.out, "end.jpg")
    shot(endp, 480)
    taken.append(("end", endp))
    from PIL import Image, ImageDraw
    cols, w, h = 3, 480, 270
    sheet = Image.new("RGB", (cols * w, ((len(taken) + cols - 1) // cols) * h), "black")
    for k, (lab, p_) in enumerate(taken):
        im = Image.open(p_).resize((w, h))
        ImageDraw.Draw(im).text((8, 8), lab, fill=(255, 255, 0))
        sheet.paste(im, ((k % cols) * w, (k // cols) * h))
    sp = os.path.join(a.out, "sheet.jpg")
    sheet.save(sp, quality=70)
    print("sheet", sp, flush=True)


if __name__ == "__main__":
    main()
