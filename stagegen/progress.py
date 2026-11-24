# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Machine-readable progress for the GUI: lines `@progress <0..1> <phase>` on
stdout (the GUI shows them as a progress bar instead of log lines)."""


def progress(frac: float, phase: str) -> None:
    print(f"@progress {max(0.0, min(1.0, frac)):.3f} {phase}", flush=True)


def span(lo: float, hi: float, i: int, n: int, phase: str) -> None:
    """Progress inside a phase covering lo..hi, at step i of n."""
    progress(lo + (hi - lo) * (i / max(1, n)), phase)
