from itertools import product

import pandas as pd

from src.backtester import run_backtest
from src.metrics import summarize_performance
from src.strategies import ma_crossover_strategy


def train_test_split_df(df: pd.DataFrame, train_ratio: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


def optimize_ma_crossover(
    df: pd.DataFrame,
    ticker: str,
    short_windows: list[int],
    long_windows: list[int],
    initial_capital: float = 10000.0,
    transaction_cost: float = 0.001,
    objective: str = "sharpe_ratio"
) -> tuple[dict, pd.DataFrame]:
    results = []

    for short_w, long_w in product(short_windows, long_windows):
        if short_w >= long_w:
            continue

        strategy_df = ma_crossover_strategy(df, short_window=short_w, long_window=long_w)
        backtest_df, trades_df = run_backtest(
            strategy_df,
            initial_capital=initial_capital,
            transaction_cost=transaction_cost,
        )

        summary = summarize_performance(
            strategy_name=f"MA_{short_w}_{long_w}",
            ticker=ticker,
            sample_period="train",
            results_df=backtest_df,
            trades_df=trades_df,
        )
        summary["short_window"] = short_w
        summary["long_window"] = long_w
        results.append(summary)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by=objective, ascending=False).reset_index(drop=True)

    best_params = {
        "short_window": int(results_df.iloc[0]["short_window"]),
        "long_window": int(results_df.iloc[0]["long_window"]),
        "objective_value": float(results_df.iloc[0][objective]),
    }

    return best_params, results_df