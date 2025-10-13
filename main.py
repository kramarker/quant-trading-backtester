from pathlib import Path
from datetime import datetime
import shutil

import pandas as pd

from src.backtester import run_backtest
from src.data_loader import get_data
from src.indicators import add_indicators
from src.metrics import summarize_performance
from src.optimization import optimize_ma_crossover, train_test_split_df
from src.plots import (
    plot_drawdown,
    plot_equity_curve,
    plot_ma_heatmap,
    plot_price_with_signals,
    plot_strategy_comparison,
)
from src.strategies import (
    buy_and_hold_strategy,
    bollinger_mean_reversion_strategy,
    ma_crossover_strategy,
    rsi_mean_reversion_strategy,
)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)


def ensure_directories() -> None:
    for folder in ["data", "plots", "results", "assets"]:
        Path(folder).mkdir(parents=True, exist_ok=True)


def clear_output_directories(clear_old_outputs: bool) -> None:
    if not clear_old_outputs:
        return

    for folder in ["data", "plots", "results"]:
        folder_path = Path(folder)
        if folder_path.exists():
            for item in folder_path.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)


def get_date_range(lookback_years: int = 5) -> tuple[str, str]:
    end_dt = datetime.today()

    try:
        start_dt = end_dt.replace(year=end_dt.year - lookback_years)
    except ValueError:
        # Handles leap-year edge cases like Feb 29
        start_dt = end_dt.replace(month=2, day=28, year=end_dt.year - lookback_years)

    return start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")


def clean_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "win_rate" in out.columns:
        out["win_rate"] = out["win_rate"] * 100

    numeric_cols = out.select_dtypes(include="number").columns
    out[numeric_cols] = out[numeric_cols].round(4)

    preferred_order = [
        "ticker",
        "sample_period",
        "strategy",
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "downside_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "calmar_ratio",
        "num_trades",
        "win_rate",
        "avg_trade_return",
        "best_trade_return",
        "worst_trade_return",
        "avg_holding_days",
    ]

    existing_cols = [col for col in preferred_order if col in out.columns]
    remaining_cols = [col for col in out.columns if col not in existing_cols]
    out = out[existing_cols + remaining_cols]

    out = out.sort_values(
        by=["ticker", "sample_period", "strategy"]
    ).reset_index(drop=True)

    return out


def save_outputs(
    ticker: str,
    strategy_name: str,
    sample_period: str,
    results_df: pd.DataFrame,
    trades_df: pd.DataFrame
) -> None:
    safe_strategy = (
        strategy_name.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "_")
    )
    safe_sample = sample_period.lower().replace(" ", "_")

    results_df.to_csv(f"results/{ticker}_{safe_strategy}_{safe_sample}_results.csv")
    trades_df.to_csv(f"results/{ticker}_{safe_strategy}_{safe_sample}_trades.csv", index=False)

    plot_price_with_signals(
        results_df,
        ticker=ticker,
        strategy_name=f"{strategy_name} ({sample_period})",
        save_path=f"plots/{ticker}_{safe_strategy}_{safe_sample}_signals.png",
    )

    plot_equity_curve(
        results_df,
        ticker=ticker,
        strategy_name=f"{strategy_name} ({sample_period})",
        save_path=f"plots/{ticker}_{safe_strategy}_{safe_sample}_equity.png",
    )

    plot_drawdown(
        results_df,
        ticker=ticker,
        strategy_name=f"{strategy_name} ({sample_period})",
        save_path=f"plots/{ticker}_{safe_strategy}_{safe_sample}_drawdown.png",
    )


def run_strategy_pipeline(
    ticker: str,
    strategy_name: str,
    sample_period: str,
    strategy_df: pd.DataFrame,
    initial_capital: float,
    transaction_cost: float
) -> dict:
    results_df, trades_df = run_backtest(
        strategy_df,
        initial_capital=initial_capital,
        transaction_cost=transaction_cost,
    )

    save_outputs(ticker, strategy_name, sample_period, results_df, trades_df)
    summary = summarize_performance(strategy_name, ticker, sample_period, results_df, trades_df)

    print(f"\n{'=' * 90}")
    print(f"{ticker} - {strategy_name} - {sample_period}")
    print(f"{'=' * 90}")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    return summary


def copy_best_plots_to_assets() -> None:
    preferred_files = [
        ("plots/SPY_ma_heatmap.png", "assets/SPY_ma_heatmap.png"),
        ("plots/strategy_comparison_sharpe.png", "assets/strategy_comparison_sharpe.png"),
    ]

    for src, dst in preferred_files:
        src_path = Path(src)
        if src_path.exists():
            shutil.copy2(src_path, dst)

    candidate_equity_files = [
        "plots/AAPL_ma_crossover_15_50_test_equity.png",
        "plots/AAPL_ma_crossover_15_50_train_equity.png",
        "plots/SPY_ma_crossover_30_200_test_equity.png",
        "plots/QQQ_ma_crossover_15_200_test_equity.png",
        "plots/AAPL_bollinger_mean_reversion_test_equity.png",
        "plots/SPY_bollinger_mean_reversion_test_equity.png",
        "plots/QQQ_bollinger_mean_reversion_test_equity.png",
    ]

    for candidate in candidate_equity_files:
        candidate_path = Path(candidate)
        if candidate_path.exists():
            shutil.copy2(candidate_path, "assets/example_equity_curve.png")
            break


def main() -> None:
    ensure_directories()

    # -----------------------------
    # CONFIG
    # -----------------------------
    tickers = ["SPY", "QQQ", "AAPL"]
    lookback_years = 5
    train_ratio = 0.7
    initial_capital = 10000.0
    transaction_cost = 0.001
    clear_old_outputs = True

    short_windows = [10, 15, 20, 30]
    long_windows = [50, 100, 150, 200]

    start, end = get_date_range(lookback_years=lookback_years)

    clear_output_directories(clear_old_outputs=clear_old_outputs)

    print(f"\nRunning backtests from {start} to {end}")
    print(f"Tickers: {tickers}")
    print(f"Lookback years: {lookback_years}")
    print(f"Train ratio: {train_ratio}")

    all_summaries = []

    for ticker in tickers:
        print(f"\nLoading data for {ticker}...")
        df = get_data(ticker, start, end)
        df = add_indicators(df)

        df.to_csv(f"data/{ticker}_data.csv")

        train_df, test_df = train_test_split_df(df, train_ratio=train_ratio)

        best_params, optimization_df = optimize_ma_crossover(
            train_df,
            ticker=ticker,
            short_windows=short_windows,
            long_windows=long_windows,
            initial_capital=initial_capital,
            transaction_cost=transaction_cost,
            objective="sharpe_ratio",
        )

        optimization_df.to_csv(f"results/{ticker}_ma_optimization.csv", index=False)
        plot_ma_heatmap(
            optimization_df,
            ticker=ticker,
            save_path=f"plots/{ticker}_ma_heatmap.png",
        )

        print(f"\nBest MA params for {ticker}: {best_params}")

        # -----------------------------
        # TRAIN
        # -----------------------------
        train_ma_df = ma_crossover_strategy(
            train_df,
            short_window=best_params["short_window"],
            long_window=best_params["long_window"],
        )
        all_summaries.append(
            run_strategy_pipeline(
                ticker=ticker,
                strategy_name=f"MA Crossover ({best_params['short_window']},{best_params['long_window']})",
                sample_period="Train",
                strategy_df=train_ma_df,
                initial_capital=initial_capital,
                transaction_cost=transaction_cost,
            )
        )

        train_rsi_df = rsi_mean_reversion_strategy(
            train_df,
            rsi_window=14,
            oversold=30,
            exit_level=55,
        )
        all_summaries.append(
            run_strategy_pipeline(
                ticker=ticker,
                strategy_name="RSI Mean Reversion",
                sample_period="Train",
                strategy_df=train_rsi_df,
                initial_capital=initial_capital,
                transaction_cost=transaction_cost,
            )
        )

        train_boll_df = bollinger_mean_reversion_strategy(
            train_df,
            window=20,
            num_std=2.0,
        )
        all_summaries.append(
            run_strategy_pipeline(
                ticker=ticker,
                strategy_name="Bollinger Mean Reversion",
                sample_period="Train",
                strategy_df=train_boll_df,
                initial_capital=initial_capital,
                transaction_cost=transaction_cost,
            )
        )

        train_bh_df = buy_and_hold_strategy(train_df)
        all_summaries.append(
            run_strategy_pipeline(
                ticker=ticker,
                strategy_name="Buy and Hold",
                sample_period="Train",
                strategy_df=train_bh_df,
                initial_capital=initial_capital,
                transaction_cost=0.0,
            )
        )

        # -----------------------------
        # TEST
        # -----------------------------
        test_ma_df = ma_crossover_strategy(
            test_df,
            short_window=best_params["short_window"],
            long_window=best_params["long_window"],
        )
        all_summaries.append(
            run_strategy_pipeline(
                ticker=ticker,
                strategy_name=f"MA Crossover ({best_params['short_window']},{best_params['long_window']})",
                sample_period="Test",
                strategy_df=test_ma_df,
                initial_capital=initial_capital,
                transaction_cost=transaction_cost,
            )
        )

        test_rsi_df = rsi_mean_reversion_strategy(
            test_df,
            rsi_window=14,
            oversold=30,
            exit_level=55,
        )
        all_summaries.append(
            run_strategy_pipeline(
                ticker=ticker,
                strategy_name="RSI Mean Reversion",
                sample_period="Test",
                strategy_df=test_rsi_df,
                initial_capital=initial_capital,
                transaction_cost=transaction_cost,
            )
        )

        test_boll_df = bollinger_mean_reversion_strategy(
            test_df,
            window=20,
            num_std=2.0,
        )
        all_summaries.append(
            run_strategy_pipeline(
                ticker=ticker,
                strategy_name="Bollinger Mean Reversion",
                sample_period="Test",
                strategy_df=test_boll_df,
                initial_capital=initial_capital,
                transaction_cost=transaction_cost,
            )
        )

        test_bh_df = buy_and_hold_strategy(test_df)
        all_summaries.append(
            run_strategy_pipeline(
                ticker=ticker,
                strategy_name="Buy and Hold",
                sample_period="Test",
                strategy_df=test_bh_df,
                initial_capital=initial_capital,
                transaction_cost=0.0,
            )
        )

    summary_df = pd.DataFrame(all_summaries)

    summary_df.to_csv("results/summary_metrics_raw.csv", index=False)

    clean_df = clean_summary(summary_df)
    clean_df.to_csv("results/summary_metrics_clean.csv", index=False)

    plot_strategy_comparison(
        clean_df,
        save_path="plots/strategy_comparison_sharpe.png",
    )

    copy_best_plots_to_assets()

    print(f"\n{'#' * 90}")
    print("FINAL CLEAN SUMMARY")
    print(f"{'#' * 90}")
    print(clean_df.to_string(index=False))


if __name__ == "__main__":
    main()