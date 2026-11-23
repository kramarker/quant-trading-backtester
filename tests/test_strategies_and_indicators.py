import numpy as np
import pandas as pd
import pytest

from src.indicators import rsi, sma
from src.strategies import (
    bollinger_mean_reversion_strategy,
    buy_and_hold_strategy,
    ma_crossover_strategy,
    rsi_mean_reversion_strategy,
)


def test_sma_matches_manual_mean():
    s = pd.Series([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(sma(s, 2).dropna(), [1.5, 2.5, 3.5])


def test_rsi_bounds_and_extremes():
    noisy = pd.Series(100 + np.random.default_rng(1).normal(0, 1, 200).cumsum())
    r = rsi(noisy, 14).dropna()
    assert r.between(0, 100).all()


def test_ma_crossover_long_only_when_short_above_long():
    df = pd.DataFrame({"Price": [1, 2, 3, 4, 3, 2, 1, 2, 3, 4]}, dtype=float)
    out = ma_crossover_strategy(df, short_window=2, long_window=4)
    expected = (out["SMA_2"] > out["SMA_4"]).astype(int)
    assert out["Signal"].tolist() == expected.tolist()
    assert set(out["Signal"].unique()) <= {0, 1}


def test_rsi_strategy_enters_below_oversold_and_holds_until_exit_level():
    df = pd.DataFrame({"RSI_14": [50, 25, 40, 54, 56, 45, 20, np.nan]})
    out = rsi_mean_reversion_strategy(df, oversold=30, exit_level=55)
    # enter at 25, hold through 40 and 54, exit above 55, stay out at 45, re-enter at 20, NaN -> 0
    assert out["Signal"].tolist() == [0, 1, 1, 1, 0, 0, 1, 0]


def test_rsi_strategy_requires_indicator_column():
    with pytest.raises(ValueError):
        rsi_mean_reversion_strategy(pd.DataFrame({"Price": [1.0, 2.0]}))


def test_bollinger_strategy_enters_below_lower_band_and_exits_above_mid():
    prices = [100.0] * 20 + [90.0, 95.0, 101.0, 100.0]
    out = bollinger_mean_reversion_strategy(pd.DataFrame({"Price": prices}), window=20, num_std=2.0)
    assert out["Signal"].iloc[:20].sum() == 0  # warm-up / flat prices: no entry
    assert out["Signal"].iloc[20] == 1        # 90 falls below the lower band
    assert out["Signal"].iloc[22] == 0        # 101 reverts above the rolling mean


def test_buy_and_hold_always_long():
    out = buy_and_hold_strategy(pd.DataFrame({"Price": [1.0, 2.0, 3.0]}))
    assert out["Signal"].tolist() == [1, 1, 1]
