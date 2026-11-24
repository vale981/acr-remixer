# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-FileCopyrightText: 2026 iceone88
# SPDX-License-Identifier: GPL-3.0-or-later AND MIT

"""Read and evaluate Unreal FInterpCurveVector data from cooked stage levels.

Coordinates in the inspected ACR spline exports are UE world-space centimetres
(X/Y horizontal, Z up). The containing spline component transforms have been
observed to be identity in the levels examined so far; this must be verified for
any new stage before assuming the values are world-space.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .unversioned import Reader
from .zen import Package


# EInterpCurveMode values in Unreal's Math/InterpCurvePoint.h.
CIM_LINEAR = 0
CIM_CURVE_AUTO = 1
CIM_CONSTANT = 2
CIM_CURVE_USER = 3
CIM_CURVE_BREAK = 4
CIM_CURVE_AUTO_CLAMPED = 5
CIM_UNKNOWN = 6


@dataclass(frozen=True)
class SplineCurve:
    """One serialized FInterpCurveVector, in its native input-key domain."""

    input_keys: np.ndarray       # (N,), FInterpCurvePoint::InVal
    positions: np.ndarray        # (N,3), FVector OutVal (cm)
    arrive_tangents: np.ndarray  # (N,3), FVector (per input-key unit)
    leave_tangents: np.ndarray   # (N,3), FVector (per input-key unit)
    interp_modes: np.ndarray     # (N,), EInterpCurveMode byte
    is_looped: bool = False
    loop_key_offset: float = 0.0


def _vec(r: Reader):
    return (r.f64(), r.f64(), r.f64())


def read_spline_curve(pk: Package, export_name: str) -> SplineCurve:
    """Decode the Position FInterpCurveVector of a DMSplineComponent export."""
    e = next(x for x in pk.exports if x.name == export_name)
    r = Reader(pk.d, e.data_offset)
    top = r.header()
    if top and top[0] == 0:      # 4-byte value before SplineCurves (IdealSpline)
        r.i32()
        top = top[1:]
    if not top or top[0] != 2:
        raise ValueError(f"no SplineCurves in {export_name}: {top}")

    curves = r.header()          # FSplineCurves
    if not curves or curves[0] != 0:
        raise ValueError(f"no Position curve in {export_name}: {curves}")

    curve_fields = r.header()    # FInterpCurveVector: Points, bIsLooped, LoopKeyOffset
    if not curve_fields or curve_fields[0] != 0:
        return SplineCurve(np.zeros(0), np.zeros((0, 3)), np.zeros((0, 3)),
                           np.zeros((0, 3)), np.zeros(0, dtype=np.uint8))

    n = r.u32()
    inval = np.zeros(n, dtype=np.float64)
    positions = np.zeros((n, 3), dtype=np.float64)
    arrive = np.zeros((n, 3), dtype=np.float64)
    leave = np.zeros((n, 3), dtype=np.float64)
    # The two-argument FInterpCurvePoint constructor defaults to CIM_Linear.
    modes = np.full(n, CIM_LINEAR, dtype=np.uint8)
    for k in range(n):
        for f in r.header():
            if f == 0:
                inval[k] = r.f32()
            elif f == 1:
                positions[k] = _vec(r)
            elif f == 2:
                arrive[k] = _vec(r)
            elif f == 3:
                leave[k] = _vec(r)
            elif f == 4:
                modes[k] = r.u8()
            else:
                raise ValueError(f"unexpected curve point field {f}")

    is_looped = False
    loop_key_offset = 0.0
    for f in curve_fields:
        if f == 1:
            is_looped = bool(r.u8())
        elif f == 2:
            loop_key_offset = float(r.f32())
        elif f != 0:
            raise ValueError(f"unexpected FInterpCurve field {f}")

    arrays = (inval, positions, arrive, leave)
    if n and not all(np.all(np.isfinite(a)) for a in arrays):
        raise ValueError(f"non-finite spline data in {export_name}")
    if n > 1 and np.any(np.diff(inval) <= 0):
        raise ValueError(f"non-increasing spline input keys in {export_name}")
    if np.any(modes > CIM_UNKNOWN):
        raise ValueError(f"unsupported spline interpolation mode in {export_name}: {modes.max()}")
    # Unreal defines this as the interval from the last input key to the
    # virtual loop point, not as the total first-to-last key span.
    if is_looped and n > 1 and (not np.isfinite(loop_key_offset) or loop_key_offset <= 0):
        raise ValueError(f"invalid LoopKeyOffset in {export_name}: {loop_key_offset}")

    return SplineCurve(inval, positions, arrive, leave, modes, is_looped, loop_key_offset)


def read_spline_points(pk: Package, export_name: str):
    """Backward-compatible reader: (inval, positions, arrive, leave), in UE cm."""
    c = read_spline_curve(pk, export_name)
    return c.input_keys, c.positions, c.arrive_tangents, c.leave_tangents


def _point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    den = float(ab @ ab)
    if den <= 1e-24:
        return float(np.linalg.norm(p - a))
    t = float(np.clip(((p - a) @ ab) / den, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _split_bezier(c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    b0, b1, b2, b3 = c
    b01 = (b0 + b1) * 0.5
    b12 = (b1 + b2) * 0.5
    b23 = (b2 + b3) * 0.5
    b012 = (b01 + b12) * 0.5
    b123 = (b12 + b23) * 0.5
    mid = (b012 + b123) * 0.5
    return (np.stack((b0, b01, b012, mid)),
            np.stack((mid, b123, b23, b3)))


def _append_unique(out: list[np.ndarray], point: np.ndarray, eps: float = 1e-9):
    if not out or np.linalg.norm(point - out[-1]) > eps:
        out.append(np.asarray(point, dtype=np.float64).copy())


def _append_adaptive_bezier(control: np.ndarray, out: list[np.ndarray], step_cm: float,
                            tolerance_cm: float, depth: int, max_depth: int,
                            max_points: int):
    """Append a cubic Bezier endpoint using convex-hull length/flatness bounds."""
    b0, b1, b2, b3 = control
    polygon = (float(np.linalg.norm(b1 - b0)) + float(np.linalg.norm(b2 - b1))
               + float(np.linalg.norm(b3 - b2)))
    flatness = max(_point_segment_distance(b1, b0, b3),
                   _point_segment_distance(b2, b0, b3))
    if polygon <= step_cm and flatness <= tolerance_cm:
        _append_unique(out, b3)
        if len(out) > max_points:
            raise ValueError(f"spline sampling exceeded the {max_points:,} point safety limit")
        return
    if depth >= max_depth:
        raise ValueError("spline adaptive sampling reached max depth; curve is too sharp or malformed")
    left, right = _split_bezier(control)
    _append_adaptive_bezier(left, out, step_cm, tolerance_cm, depth + 1, max_depth, max_points)
    _append_adaptive_bezier(right, out, step_cm, tolerance_cm, depth + 1, max_depth, max_points)


def sample_spline(curve: SplineCurve, step_cm: float = 50.0,
                  tolerance_cm: float = 1.0, max_depth: int = 16,
                  max_points: int = 2_000_000) -> np.ndarray:
    """Sample FInterpCurveVector using Unreal's key-interval and mode semantics.

    Unreal scales endpoint tangents by the input-key interval before cubic
    Hermite interpolation. This implementation converts the resulting segment
    to an equivalent Bezier and subdivides it adaptively. Linear interpolation
    is sampled directly. Constant interpolation is rejected when it would
    create a positional jump, because the downstream road representation is a
    continuous polyline and must not invent a diagonal ramp across that jump.
    """
    if not np.isfinite(step_cm) or step_cm <= 0:
        raise ValueError("step_cm must be finite and positive")
    if not np.isfinite(tolerance_cm) or tolerance_cm < 0:
        raise ValueError("tolerance_cm must be finite and non-negative")
    if not isinstance(max_depth, int) or max_depth < 0:
        raise ValueError("max_depth must be a non-negative integer")
    if not isinstance(max_points, int) or max_points < 2:
        raise ValueError("max_points must be an integer >= 2")

    P = np.asarray(curve.positions, np.float64)
    keys = np.asarray(curve.input_keys, np.float64)
    arrive = np.asarray(curve.arrive_tangents, np.float64)
    leave = np.asarray(curve.leave_tangents, np.float64)
    modes = np.asarray(curve.interp_modes, np.uint8)
    n = len(P)
    if P.shape != (n, 3) or keys.shape != (n,) or arrive.shape != (n, 3)             or leave.shape != (n, 3) or modes.shape != (n,):
        raise ValueError("spline curve arrays have inconsistent shapes")
    if not all(np.all(np.isfinite(a)) for a in (P, keys, arrive, leave)):
        raise ValueError("spline curve contains non-finite values")
    if np.any(modes > CIM_UNKNOWN):
        raise ValueError("spline curve contains an unsupported interpolation mode")
    if n > 1 and np.any(np.diff(keys) <= 0):
        raise ValueError("spline input keys must be strictly increasing")
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if n == 1:
        return P.copy()

    segments = [(i, i + 1, float(keys[i + 1] - keys[i])) for i in range(n - 1)]
    if curve.is_looped:
        # FInterpCurve::Eval uses LoopKeyOffset itself as the final segment Diff.
        if not np.isfinite(curve.loop_key_offset) or curve.loop_key_offset <= 0:
            raise ValueError("looped spline has a non-positive LoopKeyOffset")
        segments.append((n - 1, 0, float(curve.loop_key_offset)))

    out: list[np.ndarray] = [P[0].copy()]
    for i, j, diff in segments:
        if diff <= 0 or not np.isfinite(diff):
            raise ValueError(f"invalid spline input-key interval at segment {i}")
        mode = int(modes[i])
        p0, p1 = P[i], P[j]

        if mode == CIM_CONSTANT:
            if np.linalg.norm(p1 - p0) > 1e-6:
                raise ValueError(f"constant spline segment {i} is discontinuous; cannot use as road geometry")
            _append_unique(out, p1)
            continue

        if mode == CIM_LINEAR:
            length = float(np.linalg.norm(p1 - p0))
            count = max(1, int(np.ceil(length / step_cm)))
            if len(out) + count > max_points:
                raise ValueError(f"spline sampling exceeded the {max_points:,} point safety limit")
            for k in range(1, count + 1):
                _append_unique(out, p0 + (p1 - p0) * (k / count))
            continue

        # As in FInterpCurve::Eval, every non-linear/non-constant mode uses
        # cubic interpolation. CIM_Unknown is documented as invalid, but UE's
        # evaluator takes the cubic branch for it; preserve that behavior.
        m0 = leave[i] * diff
        m1 = arrive[j] * diff
        control = np.stack((p0, p0 + m0 / 3.0, p1 - m1 / 3.0, p1))
        _append_adaptive_bezier(control, out, step_cm, tolerance_cm, 0, max_depth, max_points)

    result = np.asarray(out, np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError("spline sampling produced non-finite coordinates")
    return result


def sample_hermite(P, A, L, step_cm: float = 50.0) -> np.ndarray:
    """Compatibility helper for uniform-key cubic curves."""
    P = np.asarray(P, np.float64)
    A = np.asarray(A, np.float64)
    L = np.asarray(L, np.float64)
    if P.ndim != 2 or P.shape[1:] != (3,) or A.shape != P.shape or L.shape != P.shape:
        raise ValueError("P, A and L must be arrays of shape (N, 3)")
    n = len(P)
    curve = SplineCurve(np.arange(n, dtype=np.float64), P, A, L,
                        np.full(n, CIM_CURVE_USER, dtype=np.uint8))
    return sample_spline(curve, step_cm=step_cm)


def level_splines(pk: Package) -> dict[str, np.ndarray]:
    """All non-empty route/border splines, sampled in metres (UE axes kept)."""
    res = {}
    for name in ("CenterSpline", "IdealSpline", "LeftBorderSpline", "RightBorderSpline"):
        try:
            curve = read_spline_curve(pk, name)
        except StopIteration:
            continue
        if len(curve.positions) > 1:
            res[name] = sample_spline(curve) / 100.0
    return res
