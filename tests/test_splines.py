# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-FileCopyrightText: 2026 iceone88
# SPDX-License-Identifier: MIT

import numpy as np
import pytest

from stagegen.extract.splines import (
    CIM_CONSTANT,
    CIM_CURVE_USER,
    CIM_LINEAR,
    SplineCurve,
    sample_spline,
)


def curve(positions, keys=None, leave=None, arrive=None, modes=None,
          is_looped=False, loop_key_offset=0.0):
    positions = np.asarray(positions, dtype=np.float64)
    n = len(positions)
    return SplineCurve(
        np.arange(n, dtype=np.float64) if keys is None else np.asarray(keys, dtype=np.float64),
        positions,
        np.zeros((n, 3), dtype=np.float64) if arrive is None else np.asarray(arrive, dtype=np.float64),
        np.zeros((n, 3), dtype=np.float64) if leave is None else np.asarray(leave, dtype=np.float64),
        np.full(n, CIM_LINEAR, dtype=np.uint8) if modes is None else np.asarray(modes, dtype=np.uint8),
        is_looped,
        loop_key_offset,
    )


def test_linear_interpolation_ignores_tangents_and_respects_step():
    c = curve([[0, 0, 0], [10, 0, 0]], leave=[[100, 0, 0], [0, 0, 0]])
    sampled = sample_spline(c, step_cm=3)
    assert np.allclose(sampled[[0, -1]], [[0, 0, 0], [10, 0, 0]])
    assert np.all(np.diff(sampled[:, 0]) <= 3.0 + 1e-9)
    assert np.allclose(sampled[:, 1:], 0)


def test_cubic_tangents_are_scaled_by_nonuniform_input_key_delta():
    c = curve(
        [[0, 0, 0], [10, 0, 0]],
        keys=[2, 4],
        leave=[[1, 0, 0], [0, 0, 0]],
        arrive=[[0, 0, 0], [0, 0, 0]],
        modes=[CIM_CURVE_USER, CIM_CURVE_USER],
    )
    sampled = sample_spline(c, step_cm=2, tolerance_cm=0.1)
    # The de Casteljau split includes t=0.5. Hermite with M0=LeaveTangent*2
    # evaluates to 5.25 cm; omitting the input-key delta would give 5.125 cm.
    midpoint = sampled[np.argmin(np.abs(sampled[:, 0] - 5.25))]
    assert np.allclose(midpoint, [5.25, 0, 0], atol=1e-8)


def test_loop_key_offset_is_the_final_segment_interval_not_total_curve_span():
    c = curve(
        [[0, 0, 0], [10, 0, 0]],
        keys=[5, 10],
        modes=[CIM_LINEAR, CIM_LINEAR],
        is_looped=True,
        loop_key_offset=2.0,
    )
    sampled = sample_spline(c, step_cm=2)
    assert np.allclose(sampled[0], sampled[-1])
    assert np.allclose(sampled[-1], [0, 0, 0])


def test_discontinuous_constant_curve_is_rejected_not_turned_into_a_ramp():
    c = curve([[0, 0, 0], [10, 0, 0]], modes=[CIM_CONSTANT, CIM_LINEAR])
    with pytest.raises(ValueError, match="constant spline segment 0 is discontinuous"):
        sample_spline(c)


def test_equal_value_constant_segment_is_safe_and_deduplicated():
    c = curve([[0, 0, 0], [0, 0, 0], [10, 0, 0]], modes=[CIM_CONSTANT, CIM_LINEAR, CIM_LINEAR])
    sampled = sample_spline(c, step_cm=5)
    assert np.allclose(sampled[0], [0, 0, 0])
    assert np.allclose(sampled[-1], [10, 0, 0])
    assert not np.any(np.linalg.norm(np.diff(sampled, axis=0), axis=1) == 0)


def test_bad_shapes_and_nonmonotonic_keys_are_rejected():
    bad = SplineCurve(
        np.array([0, 0]), np.array([[0, 0, 0], [1, 0, 0]]),
        np.zeros((2, 3)), np.zeros((2, 3)), np.array([0, 0], dtype=np.uint8)
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        sample_spline(bad)

    bad_shape = SplineCurve(
        np.array([0]), np.array([[0, 0, 0]]), np.zeros((1, 3)),
        np.zeros((1, 3)), np.array([0, 0], dtype=np.uint8)
    )
    with pytest.raises(ValueError, match="inconsistent shapes"):
        sample_spline(bad_shape)


def _header(indices):
    # Test-only encoder for one contiguous unversioned-property fragment.
    indices = list(indices)
    assert indices == list(range(indices[0], indices[-1] + 1))
    skip = indices[0]
    count = len(indices)
    return __import__("struct").pack("<H", skip | (count << 9) | (1 << 8))


def test_reader_preserves_curve_modes_loop_flag_and_loop_interval():
    import struct
    from types import SimpleNamespace
    from stagegen.extract.splines import read_spline_curve

    data = bytearray()
    data += _header([2])               # DMSplineComponent: SplineCurves
    data += _header([0])               # FSplineCurves: Position
    data += _header([0, 1, 2])         # FInterpCurveVector: Points, bIsLooped, LoopKeyOffset
    data += struct.pack("<I", 2)
    point_values = [
        (1.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), CIM_CURVE_USER),
        (2.0, (10.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), CIM_LINEAR),
    ]
    for key, pos, arrive, leave, mode in point_values:
        data += _header([0, 1, 2, 3, 4])
        data += struct.pack("<f", key)
        for vec in (pos, arrive, leave):
            data += struct.pack("<ddd", *vec)
        data += struct.pack("<B", mode)
    data += struct.pack("<?f", True, 0.5)

    fake = SimpleNamespace(
        d=bytes(data),
        exports=[SimpleNamespace(name="CenterSpline", data_offset=0)],
    )
    decoded = read_spline_curve(fake, "CenterSpline")
    assert decoded.is_looped is True
    assert decoded.loop_key_offset == pytest.approx(0.5)
    assert decoded.interp_modes.tolist() == [CIM_CURVE_USER, CIM_LINEAR]
    assert decoded.input_keys.tolist() == [1.0, 2.0]
