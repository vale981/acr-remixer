# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-License-Identifier: MIT

"""IoStore containers: write synthetic ones, read them back with the real reader."""
from __future__ import annotations

import os
import struct

import pytest

from stagegen.build.iostore_write import BLOCK, ChunkIn, write_container
from stagegen.extract.iostore import Container


def _chunks():
    small = b"hello stage" * 10
    big = bytes(range(256)) * ((2 * BLOCK + 1234) // 256)      # spans three blocks
    return [ChunkIn(b"\x01" * 12, small, "Game/A/small.uasset"),
            ChunkIn(b"\x02" * 12, big, "Game/B/big.ubulk"),
            ChunkIn(b"\x03" * 12, b"", None)]


@pytest.mark.parametrize("compression", [None, "zlib", "gzip"])
def test_roundtrip(tmp_path, compression):
    base = str(tmp_path / "c")
    chunks = _chunks()
    write_container(base, 0x1234, chunks, b"pak", compression=compression)
    c = Container(base + ".utoc")
    assert c.container_id == 0x1234
    assert c.methods == (["None", compression.capitalize()] if compression else ["None"])
    assert set(c.files) == {"Game/A/small.uasset", "Game/B/big.ubulk"}
    assert c.read("Game/A/small.uasset") == chunks[0].data
    assert c.read("Game/B/big.ubulk") == chunks[1].data
    if compression:
        assert all(b.method == 1 for b in c.blocks if b.usize >= 1000)
        assert os.path.getsize(base + ".ucas") < len(chunks[1].data)   # actually compressed
        assert c.flags & 0x01                                           # container flag: Compressed
    else:
        assert not c.flags & 0x01


def test_incompressible_blocks_stay_raw_and_metas_are_flagged(tmp_path):
    base = str(tmp_path / "c")
    noise = os.urandom(3000)
    text = b"stage " * 2000
    write_container(base, 1, [ChunkIn(b"\x01" * 12, noise, "Game/noise.bin"),
                              ChunkIn(b"\x02" * 12, text, "Game/text.bin")], b"pak", compression="zlib")
    c = Container(base + ".utoc")
    assert c.read("Game/noise.bin") == noise and c.read("Game/text.bin") == text
    assert [b.method for b in c.blocks] == [0, 1]                       # like UE's packer
    d = open(base + ".utoc", "rb").read()
    metas = d[-24 * 2:]                                                  # 20 B hash, flags, padding
    assert [metas[20], metas[24 + 20]] == [0, 1]


def _patch_first_block(base: str, edit):
    """Rewrite the first block entry of a utoc (offset 5 B, csize 3 B, usize 3 B, method 1 B)."""
    p = base + ".utoc"
    d = bytearray(open(p, "rb").read())
    hsize, n_entries = struct.unpack_from("<2I", d, 20)
    at = hsize + 12 * n_entries + 10 * n_entries
    edit(d, at)
    open(p, "wb").write(bytes(d))


@pytest.mark.parametrize("compression", ["zlib", "gzip"])
def test_corrupt_block_is_reported(tmp_path, compression):
    base = str(tmp_path / "c")
    write_container(base, 1, _chunks(), b"pak", compression=compression)
    with open(base + ".ucas", "r+b") as f:                    # break the compressed stream
        f.seek(2)
        f.write(b"\xff\xff\xff\xff")
    with pytest.raises(RuntimeError, match=f"{compression.capitalize()} decompress failed"):
        Container(base + ".utoc").read("Game/A/small.uasset")


def test_size_mismatch_is_reported(tmp_path):
    base = str(tmp_path / "c")
    write_container(base, 1, _chunks(), b"pak", compression="zlib")

    def claim_bigger(d, at):
        us = int.from_bytes(d[at + 8: at + 11], "little")
        d[at + 8: at + 11] = (us + 1).to_bytes(3, "little")
    _patch_first_block(base, claim_bigger)
    with pytest.raises(RuntimeError, match="size mismatch"):
        Container(base + ".utoc").read("Game/A/small.uasset")


def test_unknown_method_is_reported(tmp_path):
    base = str(tmp_path / "c")
    write_container(base, 1, _chunks(), b"pak", compression="zlib")
    p = base + ".utoc"
    d = open(p, "rb").read().replace(b"Zlib".ljust(32, b"\0"), b"LZ4".ljust(32, b"\0"))
    open(p, "wb").write(d)
    with pytest.raises(RuntimeError, match="unsupported compression LZ4"):
        Container(base + ".utoc").read("Game/A/small.uasset")


@pytest.mark.game
def test_game_containers_read(paks):
    """Oodle and uncompressed blocks of the real install pass the size checks."""
    from stagegen.extract.iostore import GameFS
    fs = GameFS(paks, include_mods=False)
    files = [p for p in fs.index if "WalesS4HafrenSouth" in p][:50]
    assert files
    assert all(len(fs.read(p)) > 0 for p in files)
