"""random-distinct wiring, private channel blocks, and dense relabelling."""

import torch
import pytest

from torchlogix.connections import (FixedConvConnections, FixedDenseConnections,
                                    repair_duplicate_inputs)


def _leaf_coords(conn):
    """(K, S, L, 3) leaf coordinates (h, w, c) at the FIRST sliding position."""
    idx = conn.indices[0]                       # (L, K, P, S, 3)
    return idx[:, :, 0].permute(1, 2, 0, 3)     # (K, S, L, 3)


def _conv(**kw):
    base = dict(in_dim=(1, 40), channels=8, num_kernels=16, tree_depth=3,
                receptive_field_size=(1, 5), padding=(0, 2))
    base.update(kw)
    return FixedConvConnections(**base)


@pytest.mark.parametrize("g", [None, 2, 4])
def test_conv_random_distinct_has_no_repeated_tap(g):
    torch.manual_seed(0)
    conn = _conv(init_method="random-distinct", channel_group_size=g)
    c = _leaf_coords(conn)
    same = (c[:, :, 0] == c[:, :, 1]).all(-1)
    assert not same.any()


def test_conv_random_still_repeats_sometimes():
    # the historical draw is with replacement; this pins that the default changed
    # NOTHING for configs that keep init_method="random"
    torch.manual_seed(0)
    conn = _conv(init_method="random", channel_group_size=2, num_kernels=256)
    c = _leaf_coords(conn)
    assert (c[:, :, 0] == c[:, :, 1]).all(-1).any()


def test_conv_random_default_is_bit_identical_to_before():
    # channel_groups=None must take exactly the old code path (same RNG consumption)
    torch.manual_seed(3)
    a = _conv(init_method="random", channel_group_size=2)
    torch.manual_seed(3)
    b = _conv(init_method="random", channel_group_size=2, channel_groups=None)
    for x, y in zip(a.indices, b.indices):
        assert torch.equal(x, y)


@pytest.mark.parametrize("g", [None, 2])
def test_conv_distinct_keeps_per_channel_counts(g):
    torch.manual_seed(1)
    conn = _conv(init_method="random-distinct", channel_group_size=g)
    c = _leaf_coords(conn)
    if g is not None:
        # balanced: every channel of the kernel's window gets 2^d / g leaves
        for k in range(c.shape[0]):
            _, counts = c[k, ..., 2].flatten().unique(return_counts=True)
            assert counts.tolist() == [8 // g] * g


@pytest.mark.parametrize("g", [None, 2])
def test_conv_channel_groups_stay_in_block(g):
    torch.manual_seed(2)
    G, c_blk, kpb = 4, 2 if g is None else 3, 4
    conn = _conv(channels=G * c_blk, num_kernels=G * kpb, channel_groups=G,
                 channel_group_size=g, init_method="random-distinct")
    c = _leaf_coords(conn)
    for k in range(c.shape[0]):
        blk = k // kpb
        ch = c[k, ..., 2]
        assert ch.min() >= blk * c_blk and ch.max() < (blk + 1) * c_blk


def test_conv_channel_groups_window_rule_is_local():
    # with a window, kernel k uses start = base + (k_local mod (c_blk - g + 1))
    torch.manual_seed(4)
    G, c_blk, kpb, g = 2, 4, 6, 2
    conn = _conv(channels=G * c_blk, num_kernels=G * kpb, channel_groups=G,
                 channel_group_size=g, init_method="random")
    c = _leaf_coords(conn)
    for k in range(G * kpb):
        base = (k // kpb) * c_blk
        start = base + (k % kpb) % (c_blk - g + 1)
        assert sorted(c[k, ..., 2].unique().tolist()) == list(range(start, start + g))


def test_dense_random_distinct_and_relabel():
    torch.manual_seed(5)
    d = FixedDenseConnections(in_dim=50, out_dim=300, init_method="random-distinct",
                              relabel=True)
    idx = d.indices
    assert not (idx[0] == idx[1]).any()
    counts = torch.bincount(idx.flatten(), minlength=50)
    assert counts.min() >= 600 // 50 and counts.max() <= 600 // 50 + 1


def test_dense_relabel_removes_low_index_bias():
    # 2*n = 140 slots over 60 inputs: 20 inputs read 3x. Without relabelling they are
    # always inputs 0..19; with it they are spread.
    torch.manual_seed(6)
    plain = FixedDenseConnections(in_dim=60, out_dim=70, init_method="random")
    heavy = (torch.bincount(plain.indices.flatten(), minlength=60) == 3).nonzero().flatten()
    assert heavy.tolist() == list(range(20))
    torch.manual_seed(6)
    rel = FixedDenseConnections(in_dim=60, out_dim=70, init_method="random", relabel=True)
    heavy = (torch.bincount(rel.indices.flatten(), minlength=60) == 3).nonzero().flatten()
    assert len(heavy) == 20 and heavy.tolist() != list(range(20))


def test_repair_raises_when_impossible():
    with pytest.raises(ValueError):
        repair_duplicate_inputs(torch.zeros(4, 2, 1, dtype=torch.long))


def test_default_init_method_context():
    from torchlogix.connections import default_init_method
    assert FixedDenseConnections(in_dim=10, out_dim=20).init_method == "random"
    with default_init_method("random-distinct"):
        d = FixedDenseConnections(in_dim=10, out_dim=20)
        c = _conv()
        explicit = FixedDenseConnections(in_dim=10, out_dim=20, init_method="random")
    assert d.init_method == "random-distinct" and c.init_method == "random-distinct"
    assert explicit.init_method == "random"
    assert not (d.indices[0] == d.indices[1]).any()
    assert FixedDenseConnections(in_dim=10, out_dim=20).init_method == "random"


def test_learnable_random_distinct():
    from torchlogix.connections import LearnableDenseConnections
    torch.manual_seed(7)
    l = LearnableDenseConnections(in_dim=40, out_dim=30, num_candidates=4,
                                  init_method="random-distinct")
    idx = l.indices                                   # (cand, L, out)
    assert not (idx[:, 0] == idx[:, 1]).any()
