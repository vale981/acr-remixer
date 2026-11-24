# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Reader for DA_<Location>BakedData (class /Script/dmphysics.BakedDataContainer).

Layout reverse-engineered from cooked data (UE 5.6, unversioned):

    BakedDataContainer
      [0] TArray<FBakedMesh> Meshes
      [2] ... (not decoded; trailing data)
    FBakedMesh (6 values)
      [0] FTransform        (Rotation FQuat d, Translation FVector d, Scale3D d)
      [1] TArray<FVector3f>  vertices (metres; X/Z horizontal, Y up)
      [2] TArray<float>      (4 floats, unknown; usually 0,0,0,1)
      [3] TArray<uint32>     triangle indices
      [4] UDMPhysicalMaterial* (package index into imports)
      [5] FSoftObjectPath    source actor, e.g. ...:PersistentLevel.<Level>Road000
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .unversioned import Reader
from .zen import Package


@dataclass
class BakedMesh:
    rotation: tuple
    translation: tuple
    scale: tuple
    vertices: np.ndarray
    extra: np.ndarray
    indices: np.ndarray
    material: str
    actor: str

    def world_vertices(self) -> np.ndarray:
        """Vertices with the stored transform applied (UE quat x,y,z,w)."""
        x, y, z, w = self.rotation
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        return (self.vertices * np.array(self.scale)) @ R.T + np.array(self.translation)


def _transform(r: Reader):
    rot, tr, sc = (0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    for i in r.header():
        if i == 0:
            rot = tuple(r.f64() for _ in range(4))
        elif i == 1:
            tr = tuple(r.f64() for _ in range(3))
        elif i == 2:
            sc = tuple(r.f64() for _ in range(3))
        else:
            raise ValueError(f"unexpected FTransform field {i}")
    return rot, tr, sc


def _array(r: Reader, dtype, width=1):
    """Count is in scalars (vertices are stored as a flat float array)."""
    n = r.u32()
    a = np.frombuffer(r.d, dtype, n, r.p)
    r.p += a.nbytes
    return a.reshape(-1, width) if width > 1 else a


def _material_name(pk: Package, idx: int) -> str:
    # Imports are FPackageObjectIndex hashes; the name map lists material
    # names in import order, so keep the raw index as a fallback label.
    return f"import{-idx - 1}" if idx < 0 else f"export{idx - 1}" if idx > 0 else "none"


def read_baked(pk: Package) -> list[BakedMesh]:
    e = pk.exports[0]
    r = Reader(pk.d, e.data_offset)
    top = r.header()
    if not top or top[0] != 0:
        raise ValueError(f"unexpected BakedDataContainer layout {top}")
    n = r.u32()
    meshes = []
    for _ in range(n):
        rot, tr, sc = (0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
        verts = np.zeros((0, 3), np.float32)
        extra = np.zeros(0, np.float32)
        idx = np.zeros(0, np.uint32)
        mat, actor = "none", ""
        for i in r.header():
            if i == 0:
                rot, tr, sc = _transform(r)
            elif i == 1:
                verts = _array(r, np.float32, 3)
            elif i == 2:
                extra = _array(r, np.float32)
            elif i == 3:
                idx = _array(r, np.uint32)
            elif i == 4:
                mat = _material_name(pk, r.i32())
            elif i == 5:
                pkg = r.fname(pk.names)
                asset = r.fname(pk.names)
                sub = r.fstring()
                actor = sub or f"{pkg}.{asset}"
            else:
                raise ValueError(f"unexpected FBakedMesh field {i} at {r.p}")
        meshes.append(BakedMesh(rot, tr, sc, verts, extra, idx, mat, actor))
    return meshes


def material_table(pk: Package) -> list[str]:
    """Physical material short names in import order (best effort)."""
    return [n for n in pk.names if n.startswith("PM_")]
