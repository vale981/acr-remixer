# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Names and folders of generated stages (the GUI computes the same folders)."""

from argparse import Namespace

from stagegen.cli import _min_piece, _out_dir, _own_name


def _args(**kw):
    base = dict(level="WalesS4HafrenSouth", seed=1, length=4500.0, max_uses=1, out="out", min_piece=None)
    base.update(kw)
    return Namespace(**base)


def test_default_shortest_piece_keeps_names():
    a = _args()
    assert _min_piece(a) == 250.0
    assert _out_dir(a).replace("\\", "/") == "out/WalesS4HafrenSouth_seed1_4500m"
    assert _own_name(a) == "SG1L4500HafrenSouth"
    # the default given explicitly is the same stage
    assert _out_dir(_args(min_piece=250.0)) == _out_dir(a)


def test_shorter_pieces_get_their_own_names():
    a = _args(min_piece=100.0, max_uses=2)
    assert _out_dir(a).replace("\\", "/") == "out/WalesS4HafrenSouth_seed1_4500m_x2_p100"
    assert _own_name(a) == "SG1L4500U2P100HafrenSouth"
    assert not _own_name(a)[-1].isdigit()
