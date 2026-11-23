"""Walk-forward (rolling) validation for the moving-average crossover strategy.

Instead of a single 70/30 split, the data is cut into consecutive folds. In each fold the
MA windows are chosen on a trailing training window only, then traded on the next,
unseen test window. Stitching the test windows together gives an out-of-sample equity
curve that covers most of the history, which shows whether one lucky split drove the result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtester import run_backtest
from src.metrics import max_drawdown, sharpe_ratio, total_return
from src.optimization import optimize_ma_crossover
from src.strategies import ma_crossover_strategy


def _add_sma_columns(df: pd.DataFrame, windows: set[int]) -> pd.DataFrame:
    """Compute trailing SMAs on the full series so test folds do not lose a warm-up period.

    Rolling means only use past prices, so computing them before slicing adds no look-ahead.
    """
    out = df.copy()
    for window in sorted(windows):
        col = f"SMA_{window}"
        if col not in out.columns:
            out[col] = out["Price"].rolling(window=window).mean()
    return out


def make_folds(n_obs: int, train_size: int, test_size: int, step: int | None = None) -> list[tuple[slice, slice]]:
    """Return (train, test) index slices for rolling folds. Test windows never overlap their training window."""
    if train_size <= 0 or test_size <= 0:
        raise ValueError("train_size and test_size must be positive.")
    step = test_size if step is None else step
    if step <= 0:
        raise ValueError("step must be positive.")

    folds = []
    start = 0
    while start + train_size + test_size <= n_obs:
        train = slice(start, start + train_size)
        test = slice(start + train_size, start + train_size + test_size)
        folds.append((train, test))
        start += step
    return folds


def walk_forward_ma(
    df: pd.DataFrame,
    ticker: str,
    short_windows: list[int],
    long_windows: list[int],
    train_size: int = 504,
    test_size: int = 63,
    step: int | None = None,
    initial_capital: float = 10000.0,
    transaction_cost: float = 0.001,
    objective: str = "sharpe_ratio",
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Run rolling walk-forward validation for the MA crossover strategy.

    Returns
    -------
    folds_df : one row per fold with the chosen windows and in/out-of-sample results
    oos_returns : daily out-of-sample strategy returns stitched across all test windows
    summary : aggregate out-of-sample statistics vs. buy-and-hold over the same dates
    """
    data = _add_sma_columns(df, set(short_windows) | set(long_windows))
    folds = make_folds(len(data), train_size, test_size, step)
    if not folds:
        raise ValueError("Not enough data for a single train/test fold.")

    rows = []
    oos_chunks = []
    bh_chunks = []

    for i, (train_idx, test_idx) in enumerate(folds):
        train_df = data.iloc[train_idx]
        test_df = data.iloc[test_idx]

        best, _ = optimize_ma_crossover(
            train_df,
            ticker=ticker,
            short_windows=short_windows,
            long_windows=long_windows,
            initial_capital=initial_capital,
            transaction_cost=transaction_cost,
            objective=objective,
        )

        test_signals = ma_crossover_strategy(
            test_df, short_window=best["short_window"], long_window=best["long_window"]
        )
        test_bt, _ = run_backtest(test_signals, initial_capital=initial_capital, transaction_cost=transaction_cost)

        test_returns = test_bt["Strategy_Return"]
        rows.append({
            "fold": i,
            "train_start": train_df.index[0],
            "train_end": train_df.index[-1],
            "test_start": test_df.index[0],
            "test_end": test_df.index[-1],
            "short_window": best["short_window"],
            "long_window": best["long_window"],
            "train_sharpe": best["objective_value"],
            "test_sharpe": sharpe_ratio(test_returns),
            "test_total_return": total_return(test_bt["Portfolio_Value"]),
            "test_buy_hold_return": float((1 + test_df["Return"]).prod() - 1),
        })
        oos_chunks.append(test_returns)
        bh_chunks.append(test_df["Return"])

    folds_df = pd.DataFrame(rows)
    oos_returns = pd.concat(oos_chunks)
    bh_returns = pd.concat(bh_chunks)

    oos_value = initial_capital * (1 + oos_returns).cumprod()
    bh_value = initial_capital * (1 + bh_returns).cumprod()

    summary = {
        "ticker": ticker,
        "num_folds": len(folds_df),
        "oos_start": oos_returns.index[0],
        "oos_end": oos_returns.index[-1],
        "oos_total_return": float(oos_value.iloc[-1] / initial_capital - 1),
        "oos_sharpe": float(sharpe_ratio(oos_returns)),
        "oos_max_drawdown": float(max_drawdown(oos_value)),
        "buy_hold_total_return": float(bh_value.iloc[-1] / initial_capital - 1),
        "buy_hold_sharpe": float(sharpe_ratio(bh_returns)),
        "buy_hold_max_drawdown": float(max_drawdown(bh_value)),
        "pct_folds_beating_buy_hold": float(
            np.mean(folds_df["test_total_return"] > folds_df["test_buy_hold_return"])
        ),
        "distinct_window_pairs_chosen": int(
            folds_df[["short_window", "long_window"]].drop_duplicates().shape[0]
        ),
    }
    return folds_df, oos_returns, summary
