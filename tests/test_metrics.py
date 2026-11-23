import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    calmar_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    total_return,
    trade_statistics,
)


def test_total_return_and_max_drawdown():
    pv = pd.Series([100.0, 120.0, 90.0, 130.0])
    assert total_return(pv) == pytest.approx(0.30)
    assert max_drawdown(pv) == pytest.approx(-0.25)


def test_sharpe_is_nan_for_zero_volatility_and_annualized_otherwise():
    assert np.isnan(sharpe_ratio(pd.Series([0.001] * 10)))
    r = pd.Series([0.01, -0.005, 0.002, 0.004])
    assert sharpe_ratio(r) == pytest.approx(r.mean() / r.std() * np.sqrt(252))


def test_sortino_only_penalizes_downside():
    r = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01])
    assert sortino_ratio(r) > sharpe_ratio(r)


def test_calmar_uses_absolute_drawdown():
    pv = pd.Series(np.linspace(100, 120, 253))
    pv.iloc[100] = 90
    assert calmar_ratio(pv) > 0


def test_trade_statistics_empty_and_basic():
    assert trade_statistics(pd.DataFrame())["num_trades"] == 0
    trades = pd.DataFrame({"trade_return": [0.1, -0.05], "holding_days": [5, 3]})
    stats = trade_statistics(trades)
    assert stats["num_trades"] == 2
    assert stats["win_rate"] == pytest.approx(0.5)
    assert stats["avg_holding_days"] == pytest.approx(4.0)
