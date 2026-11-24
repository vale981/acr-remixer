# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-License-Identifier: GPL-3.0-or-later

# PyInstaller spec: one-file stagegen backend (the GUI embeds this exe).
#   cd acr-stagegen; python -m PyInstaller packaging/backend.spec --distpath packaging/dist --workpath packaging/build
import os

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))

a = Analysis(
    [os.path.join(SPECPATH, "backend_main.py")],
    pathex=[ROOT],
    binaries=[(os.path.join(ROOT, "oodle_shim", "target", "release", "oo2core_shim.dll"), ".")],
    datas=[(os.path.join(ROOT, "stagegen", "build", "empty_mod.pak"), os.path.join("stagegen", "build"))],
    hiddenimports=["scipy.spatial._ckdtree", "scipy.interpolate", "scipy.ndimage"],
    excludes=["matplotlib", "tkinter", "PIL", "IPython", "pandas", "pytest", "vgamepad"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, a.binaries, a.datas, [],
    name="stagegen-backend",
    console=True,
    upx=False,
    strip=False,
)
