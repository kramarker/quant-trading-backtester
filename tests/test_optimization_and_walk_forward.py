import pandas as pd
import pytest

from src.optimization import optimize_ma_crossover, train_test_split_df
from src.walk_forward import make_folds, walk_forward_ma

SHORT = [10, 15, 20, 30]
LONG = [50, 100, 150, 200]


def test_train_test_split_is_chronological_and_disjoint(long_df):
    shuffled = long_df.sample(frac=1.0, random_state=0).sort_index()  # sorted input, as in main.py
    train, test = train_test_split_df(shuffled, train_ratio=0.7)
    assert len(train) == int(len(shuffled) * 0.7)
    assert len(train) + len(test) == len(shuffled)
    assert train.index.max() < test.index.min()  # every training date precedes every test date
    assert train.index.is_monotonic_increasing and test.index.is_monotonic_increasing


def test_optimizer_searches_16_pairs_and_ranks_by_sharpe(long_df):
    best, results = optimize_ma_crossover(long_df, "TEST", SHORT, LONG)
    assert len(results) == 16
    assert results["sharpe_ratio"].is_monotonic_decreasing
    assert best["objective_value"] == pytest.approx(results["sharpe_ratio"].iloc[0])
    assert best["short_window"] < best["long_window"]


def test_optimizer_skips_invalid_pairs(long_df):
    _, results = optimize_ma_crossover(long_df, "TEST", [10, 50], [20, 50])
    assert ((results["short_window"] < results["long_window"])).all()
    assert len(results) == 2  # (10,20) and (10,50); (50,20) and (50,50) are skipped


def test_make_folds_never_lets_test_overlap_training():
    folds = make_folds(n_obs=300, train_size=100, test_size=50)
    assert len(folds) == 4
    for train, test in folds:
        assert train.stop == test.start
        assert test.stop - test.start == 50
    with pytest.raises(ValueError):
        make_folds(100, 0, 10)


def test_walk_forward_test_windows_are_contiguous_and_out_of_sample(long_df):
    folds_df, oos, summary = walk_forward_ma(long_df, "TEST", SHORT, LONG, train_size=400, test_size=100)
    assert summary["num_folds"] == len(folds_df) == 5
    assert (folds_df["train_end"] < folds_df["test_start"]).all()
    assert oos.index.is_unique and oos.index.is_monotonic_increasing
    assert len(oos) == 5 * 100
    assert -1.0 < summary["oos_max_drawdown"] <= 0.0


def test_walk_forward_requires_enough_data(long_df):
    with pytest.raises(ValueError):
        walk_forward_ma(long_df.iloc[:100], "TEST", SHORT, LONG, train_size=504, test_size=63)
