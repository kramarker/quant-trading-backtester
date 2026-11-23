import numpy as np
import pandas as pd
import pytest

from src.backtester import generate_trade_log, run_backtest


def with_signal(df, signal):
    out = df.copy()
    out["Signal"] = signal
    return out


def test_position_uses_previous_days_signal(price_df):
    # A signal raised on day t must only earn the return of day t+1 (no look-ahead).
    signal = [0, 1, 1, 0, 0, 1, 0, 0]
    out, _ = run_backtest(with_signal(price_df, signal), transaction_cost=0.0)
    expected_position = [0, 0, 1, 1, 0, 0, 1, 0]
    assert out["Position"].tolist() == expected_position
    np.testing.assert_allclose(out["Gross_Strategy_Return"], np.array(expected_position) * price_df["Return"])


def test_transaction_cost_charged_on_every_position_change(price_df):
    signal = [0, 1, 1, 0, 0, 1, 0, 0]  # positions change 4 times: in, out, in, out
    out, _ = run_backtest(with_signal(price_df, signal), transaction_cost=0.001)
    assert out["Trade"].sum() == 4
    np.testing.assert_allclose(out["Transaction_Cost"], out["Trade"] * 0.001)
    np.testing.assert_allclose(out["Strategy_Return"], out["Gross_Strategy_Return"] - out["Transaction_Cost"])


def test_costs_reduce_final_value(price_df):
    signal = [1, 0, 1, 0, 1, 0, 1, 0]
    free, _ = run_backtest(with_signal(price_df, signal), transaction_cost=0.0)
    costly, _ = run_backtest(with_signal(price_df, signal), transaction_cost=0.001)
    assert costly["Portfolio_Value"].iloc[-1] < free["Portfolio_Value"].iloc[-1]


def test_always_long_without_costs_matches_buy_and_hold_after_first_day(price_df):
    out, _ = run_backtest(with_signal(price_df, [1] * len(price_df)), transaction_cost=0.0)
    # Day 0 is flat because of the one-day lag; afterwards it compounds exactly like buy-and-hold.
    growth = (1 + price_df["Return"].iloc[1:]).prod()
    assert out["Portfolio_Value"].iloc[-1] == pytest.approx(10_000 * growth)


def test_drawdown_is_never_positive(price_df):
    out, _ = run_backtest(with_signal(price_df, [1, 1, 0, 1, 1, 1, 0, 1]))
    assert (out["Drawdown"] <= 1e-12).all()


def test_missing_signal_column_raises(price_df):
    with pytest.raises(ValueError):
        run_backtest(price_df)


def test_trade_log_records_entries_exits_and_open_trade(price_df):
    df = price_df.copy()
    df["Position"] = [0, 1, 1, 0, 0, 1, 1, 1]
    trades = generate_trade_log(df)
    assert len(trades) == 2
    first = trades.iloc[0]
    assert first["entry_price"] == 102 and first["exit_price"] == 105
    assert first["trade_return"] == pytest.approx(105 / 102 - 1)
    # The open trade is closed on the last available date.
    assert trades.iloc[1]["exit_date"] == df.index[-1]
