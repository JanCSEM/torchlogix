"""tree_tap_draw: the 2^d leaf taps of a conv tree drawn without replacement from the tree's
candidate pool (ke x kt x channels it may read); refill only when the pool is smaller."""

from collections import Counter

import pytest
import torch

from torchlogix.connections import (FixedConvConnections, default_tree_tap_draw,
                                    draw_without_replacement)


def _taps(conn):
    idx = conn.indices[0]                                    # (L, K, P, S, 3)
    return idx[:, :, 0].permute(1, 2, 0, 3).reshape(idx.shape[1], -1, 3)   # (K, 2^d, 3)


def _conv(**kw):
    base = dict(in_dim=(5, 20), channels=8, num_kernels=32, tree_depth=3,
                receptive_field_size=(5, 3), padding=(2, 1), init_method="random-distinct",
                channel_group_size=4)
    base.update(kw)
    return FixedConvConnections(**base)


def test_default_is_replace_and_bitwise_the_explicit_replace():
    torch.manual_seed(0)
    a = _conv()
    torch.manual_seed(0)
    b = _conv(tree_tap_draw="replace")
    assert a.tree_tap_draw == "replace" and torch.equal(a.indices[0], b.indices[0])
    with default_tree_tap_draw("no-replace"):
        assert _conv().tree_tap_draw == "no-replace"
    assert _conv().tree_tap_draw == "replace"                  # the context manager restores


@pytest.mark.parametrize("g", [None, 2, 4])
def test_taps_of_a_tree_are_distinct_and_in_the_pool(g):
    torch.manual_seed(1)
    c = _conv(tree_tap_draw="no-replace", channel_group_size=g)
    for t in _taps(c):
        assert len({tuple(x) for x in t.tolist()}) == 8
        assert t[:, 0].max() < 5 and t[:, 1].max() < 3
        if g:
            assert set(Counter(t[:, 2].tolist()).values()) == {8 // g}
            assert t[:, 2].max() - t[:, 2].min() + 1 == g


def test_refill_balances_when_pool_is_smaller_than_the_leaves():
    torch.manual_seed(2)
    c = _conv(in_dim=(1, 7), receptive_field_size=(1, 3), padding=(0, 1), channel_group_size=2,
              tree_tap_draw="no-replace")                       # sub-pool 3, 4 leaves per channel
    assert c.pool_report()["refill"] is True
    for t in _taps(c):
        per = Counter(tuple(x) for x in t.tolist())
        assert len(per) == 6 and set(per.values()) <= {1, 2}


def test_draw_without_replacement_counts():
    for n_pool, n_draw in ((6, 4), (6, 6), (6, 8), (3, 10)):
        c = Counter(draw_without_replacement(n_pool, n_draw).tolist())
        assert sum(c.values()) == n_draw and len(c) == min(n_pool, n_draw)
        assert max(c.values()) - min(c.values()) <= 1 and max(c.values()) <= -(-n_draw // n_pool)


def test_random_unique_is_rejected():
    with pytest.raises(ValueError, match="random-unique"):
        _conv(init_method="random-unique", channel_group_size=None, tree_tap_draw="no-replace")
