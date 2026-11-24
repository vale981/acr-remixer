# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Entry point of the frozen backend (stagegen-backend.exe): the CLI."""
import sys

# frozen apps ignore PYTHONUNBUFFERED: without this the GUI sees no log until
# the build ends (stdout is a pipe there)
for s in (sys.stdout, sys.stderr):
    try:
        s.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

from stagegen.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
