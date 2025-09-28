from pathlib import Path
from datetime import datetime

import pandas as pd

from src.backtester import run_backtest
from src.data_loader import get_data
from src.indicators import add_indicators
from src.metrics import summarize_performance
from src.plots import plot_drawdown, plot_equity_curve, plot_price_with_signals
from src.strategies import (
    buy_and_hold_strategy,
    ma_crossover_strategy,
    rsi_mean_reversion_strategy,
)


def ensure_directories() -> None:
    for folder in ["data", "plots", "results"]:
        Path(folder).mkdir(parents=True, exist_ok=True)


def save_outputs(
    ticker: str,
    strategy_name: str,
    results_df: pd.DataFrame,
    trades_df: pd.DataFrame
) -> None:
    safe_strategy = strategy_name.lower().replace(" ", "_")

    results_df.to_csv(f"results/{ticker}_{safe_strategy}_results.csv")
    trades_df.to_csv(f"results/{ticker}_{safe_strategy}_trades.csv", index=False)

    plot_price_with_signals(
        results_df,
        ticker=ticker,
        strategy_name=strategy_name,
        save_path=f"plots/{ticker}_{safe_strategy}_signals.png",
    )

    plot_equity_curve(
        results_df,
        ticker=ticker,
        strategy_name=strategy_name,
        save_path=f"plots/{ticker}_{safe_strategy}_equity.png",
    )

    plot_drawdown(
        results_df,
        ticker=ticker,
        strategy_name=strategy_name,
        save_path=f"plots/{ticker}_{safe_strategy}_drawdown.png",
    )


def run_strategy_pipeline(
    ticker: str,
    strategy_name: str,
    strategy_df: pd.DataFrame,
    initial_capital: float,
    transaction_cost: float
) -> dict:
    results_df, trades_df = run_backtest(
        strategy_df,
        initial_capital=initial_capital,
        transaction_cost=transaction_cost,
    )

    save_outputs(ticker, strategy_name, results_df, trades_df)
    summary = summarize_performance(strategy_name, ticker, results_df, trades_df)

    print(f"\n{'=' * 80}")
    print(f"{ticker} - {strategy_name}")
    print(f"{'=' * 80}")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    return summary


def main() -> None:
    ensure_directories()

    tickers = ["SPY", "QQQ", "AAPL"]
    start = "2020-01-01"
    end = datetime.today().strftime("%Y-%m-%d")

    initial_capital = 10000.0
    transaction_cost = 0.001

    all_summaries = []

    for ticker in tickers:
        print(f"\nLoading data for {ticker}...")
        df = get_data(ticker, start, end)
        df = add_indicators(df)

        # Save cleaned indicator dataset
        df.to_csv(f"data/{ticker}_data.csv")

        # Strategy 1: Moving average crossover
        ma_df = ma_crossover_strategy(df, short_window=20, long_window=50)
        summary_ma = run_strategy_pipeline(
            ticker=ticker,
            strategy_name="MA Crossover",
            strategy_df=ma_df,
            initial_capital=initial_capital,
            transaction_cost=transaction_cost,
        )
        all_summaries.append(summary_ma)

        # Strategy 2: RSI mean reversion
        rsi_df = rsi_mean_reversion_strategy(df, rsi_window=14, oversold=30, exit_level=55)
        summary_rsi = run_strategy_pipeline(
            ticker=ticker,
            strategy_name="RSI Mean Reversion",
            strategy_df=rsi_df,
            initial_capital=initial_capital,
            transaction_cost=transaction_cost,
        )
        all_summaries.append(summary_rsi)

        # Benchmark: Buy and hold
        bh_df = buy_and_hold_strategy(df)
        summary_bh = run_strategy_pipeline(
            ticker=ticker,
            strategy_name="Buy and Hold",
            strategy_df=bh_df,
            initial_capital=initial_capital,
            transaction_cost=0.0,
        )
        all_summaries.append(summary_bh)

    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv("results/summary_metrics.csv", index=False)

    print(f"\n{'#' * 80}")
    print("FINAL SUMMARY")
    print(f"{'#' * 80}")
    print(summary_df)


if __name__ == "__main__":
    main()