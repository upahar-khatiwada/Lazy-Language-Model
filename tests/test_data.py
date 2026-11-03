import numpy as np
import pandas as pd
import pytest

from data import load_corpus, split_ids, make_windows, iterate_batches

IDS = np.array([3, 1, 4, 1, 5, 2, 6, 5, 3, 5])   # N = 10


def test_window_table_matches_hand_example():
    # T = 4, stride 1 -> W = N - T = 6 windows
    x, y = make_windows(IDS, seq_len=4)
    x, y = x.to_numpy(), y.to_numpy()
    assert x.shape == (6, 4) and y.shape == (6, 4)
    assert x[0].tolist() == [3, 1, 4, 1] and y[0].tolist() == [1, 4, 1, 5]
    assert x[5].tolist() == [2, 6, 5, 3] and y[5].tolist() == [6, 5, 3, 5]


def test_targets_are_inputs_shifted_by_one():
    x, y = make_windows(IDS, seq_len=4)
    assert (y.to_numpy()[:, :-1] == x.to_numpy()[:, 1:]).all()


def test_stride_skips_windows():
    x, _ = make_windows(IDS, seq_len=4, stride=2)
    assert x.to_numpy()[:, 0].tolist() == [3, 4, 5]       # starts 0, 2, 4


def test_too_few_tokens_raises():
    with pytest.raises(ValueError):
        make_windows(np.arange(4), seq_len=4)


def test_split_is_contiguous_and_disjoint():
    ids = np.arange(100)
    train, val = split_ids(ids, 0.1)
    assert len(train) == 90 and len(val) == 10
    assert train[-1] + 1 == val[0]                        # cut, no overlap


def test_window_counts_for_n_3000():
    # guide: N = 3000, T = 16, rho = 0.1 -> 2684 train and 284 val windows
    train, val = split_ids(np.arange(3000), 0.1)
    assert len(make_windows(train, 16)[0]) == 2684
    assert len(make_windows(val, 16)[0]) == 284


def test_batches_cover_every_window_once():
    x, y = make_windows(np.arange(50), seq_len=4)         # 46 windows
    rng = np.random.default_rng(0)
    seen, sizes = [], []
    for xb, yb in iterate_batches(x, y, batch_size=16, rng=rng):
        assert xb.shape == yb.shape and xb.shape[1] == 4
        assert (yb[:, :-1] == xb[:, 1:]).all()            # pairs stay aligned
        seen.extend(xb[:, 0].tolist())
        sizes.append(len(xb))
    assert sorted(seen) == list(range(46))
    assert sizes == [16, 16, 14]


def test_no_shuffle_keeps_order():
    x, y = make_windows(np.arange(20), seq_len=4)
    first = next(iterate_batches(x, y, 5, np.random.default_rng(0), shuffle=False))[0]
    assert first[:, 0].tolist() == [0, 1, 2, 3, 4]


def test_load_corpus_joins_rows_and_skips_missing(tmp_path):
    f = tmp_path / "c.csv"
    pd.DataFrame({"text": [" hello ", None, "world"]}).to_csv(f, index=False)
    assert load_corpus(str(f), "text", 10) == "hello\nworld\n"


def test_load_corpus_respects_max_rows(tmp_path):
    f = tmp_path / "c.csv"
    pd.DataFrame({"text": ["a", "b", "c"]}).to_csv(f, index=False)
    assert load_corpus(str(f), "text", 2) == "a\nb\n"


def test_load_corpus_wrong_column_raises(tmp_path):
    f = tmp_path / "c.csv"
    pd.DataFrame({"body": ["a"]}).to_csv(f, index=False)
    with pytest.raises(ValueError):
        load_corpus(str(f), "text", 10)