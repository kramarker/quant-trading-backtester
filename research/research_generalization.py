from pathlib import Path
from datetime import datetime
import shutil
import sys

import matplotlib.pyplot as plt
import pandas as pd

# Allow imports from the project root when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtester import run_backtest
from src.data_loader import get_data
from src.indicators import add_indicators
from src.metrics import summarize_performance
from src.optimization import optimize_ma_crossover, train_test_split_df
from src.strategies import (
    buy_and_hold_strategy,
    bollinger_mean_reversion_strategy,
    ma_crossover_strategy,
    rsi_mean_reversion_strategy,
)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)


# -------------------------------------------------------------------
# PATHS
# -------------------------------------------------------------------
RESEARCH_DIR = PROJECT_ROOT / "research"
RESULTS_DIR = RESEARCH_DIR / "research_results"
PLOTS_DIR = RESEARCH_DIR / "research_plots"
ASSETS_DIR = RESEARCH_DIR / "research_assets"


# -------------------------------------------------------------------
# DIRECTORY HELPERS
# -------------------------------------------------------------------
def ensure_directories() -> None:
    for folder in [RESULTS_DIR, PLOTS_DIR, ASSETS_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def clear_output_directories(clear_old_outputs: bool = True) -> None:
    if not clear_old_outputs:
        return

    for folder in [RESULTS_DIR, PLOTS_DIR, ASSETS_DIR]:
        if folder.exists():
            for item in folder.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)


# -------------------------------------------------------------------
# DATE / CONFIG HELPERS
# -------------------------------------------------------------------
def get_date_range(lookback_years: int = 5) -> tuple[str, str]:
    end_dt = datetime.today()

    try:
        start_dt = end_dt.replace(year=end_dt.year - lookback_years)
    except ValueError:
        # handles leap-year edge cases like Feb 29
        start_dt = end_dt.replace(month=2, day=28, year=end_dt.year - lookback_years)

    return start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")


def clean_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "win_rate" in out.columns:
        out["win_rate"] = out["win_rate"] * 100

    numeric_cols = out.select_dtypes(include="number").columns
    out[numeric_cols] = out[numeric_cols].round(4)

    preferred_order = [
        "training_asset",
        "evaluation_asset",
        "sample_period",
        "strategy",
        "strategy_family",
        "optimized_on",
        "ma_short_window",
        "ma_long_window",
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
        by=["training_asset", "evaluation_asset", "sample_period", "strategy"]
    ).reset_index(drop=True)

    return out


# -------------------------------------------------------------------
# PLOTTING
# -------------------------------------------------------------------
def plot_generalization_bar_chart(summary_df: pd.DataFrame, save_path: Path) -> None:
    """
    Bar chart of Sharpe ratios across strategies / assets for TEST set only.
    """
    plot_df = summary_df.copy()
    plot_df = plot_df[plot_df["sample_period"] == "Test"].copy()

    plot_df["label"] = (
        plot_df["evaluation_asset"]
        + " | "
        + plot_df["strategy"]
    )

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.bar(plot_df["label"], plot_df["sharpe_ratio"])

    ax.set_title("Cross-Asset Generalization (Test Set Sharpe Ratio)")
    ax.set_xlabel("Evaluation Asset | Strategy")
    ax.set_ylabel("Sharpe Ratio")
    ax.tick_params(axis="x", rotation=90)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_ma_generalization_heatmap(summary_df: pd.DataFrame, save_path: Path) -> None:
    """
    Heatmap of MA Sharpe ratios on TEST set, indexed by evaluation asset and training asset.
    Since we usually train on one asset, this still makes the transfer result easy to read.
    """
    plot_df = summary_df.copy()
    plot_df = plot_df[
        (plot_df["sample_period"] == "Test")
        & (plot_df["strategy_family"] == "MA")
    ].copy()

    pivot = plot_df.pivot(
        index="evaluation_asset",
        columns="training_asset",
        values="sharpe_ratio"
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    ax.set_title("MA Strategy Cross-Asset Generalization (Test Sharpe)")
    ax.set_xlabel("Training Asset")
    ax.set_ylabel("Evaluation Asset")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Sharpe Ratio")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_total_return_bar_chart(summary_df: pd.DataFrame, save_path: Path) -> None:
    plot_df = summary_df.copy()
    plot_df = plot_df[plot_df["sample_period"] == "Test"].copy()

    plot_df["label"] = (
        plot_df["evaluation_asset"]
        + " | "
        + plot_df["strategy"]
    )

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.bar(plot_df["label"], plot_df["total_return"])

    ax.set_title("Cross-Asset Generalization (Test Set Total Return)")
    ax.set_xlabel("Evaluation Asset | Strategy")
    ax.set_ylabel("Total Return")
    ax.tick_params(axis="x", rotation=90)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def copy_best_plots_to_assets() -> None:
    preferred_files = [
        (PLOTS_DIR / "generalization_sharpe.png", ASSETS_DIR / "generalization_sharpe.png"),
        (PLOTS_DIR / "generalization_total_return.png", ASSETS_DIR / "generalization_total_return.png"),
        (PLOTS_DIR / "ma_generalization_heatmap.png", ASSETS_DIR / "ma_generalization_heatmap.png"),
    ]

    for src, dst in preferred_files:
        if src.exists():
            shutil.copy2(src, dst)


# -------------------------------------------------------------------
# RESEARCH PIPELINE HELPERS
# -------------------------------------------------------------------
def run_strategy_pipeline(
    training_asset: str,
    evaluation_asset: str,
    strategy_name: str,
    strategy_family: str,
    sample_period: str,
    strategy_df: pd.DataFrame,
    initial_capital: float,
    transaction_cost: float,
    optimized_on: str,
    ma_short_window: int | None = None,
    ma_long_window: int | None = None,
) -> dict:
    results_df, trades_df = run_backtest(
        strategy_df,
        initial_capital=initial_capital,
        transaction_cost=transaction_cost,
    )

    summary = summarize_performance(
        strategy_name=strategy_name,
        ticker=evaluation_asset,
        sample_period=sample_period,
        results_df=results_df,
        trades_df=trades_df,
    )

    summary["training_asset"] = training_asset
    summary["evaluation_asset"] = evaluation_asset
    summary["strategy_family"] = strategy_family
    summary["optimized_on"] = optimized_on
    summary["ma_short_window"] = ma_short_window
    summary["ma_long_window"] = ma_long_window

    return summary


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main() -> None:
    ensure_directories()

    # -----------------------------
    # CONFIG
    # -----------------------------
    training_asset = "SPY"
    evaluation_assets = ["SPY", "QQQ", "AAPL"]

    lookback_years = 5
    train_ratio = 0.7
    initial_capital = 10000.0
    transaction_cost = 0.001
    clear_old_outputs = True

    short_windows = [10, 15, 20, 30]
    long_windows = [50, 100, 150, 200]

    start, end = get_date_range(lookback_years=lookback_years)

    clear_output_directories(clear_old_outputs=clear_old_outputs)

    print(f"\nRunning research experiment from {start} to {end}")
    print(f"Training asset: {training_asset}")
    print(f"Evaluation assets: {evaluation_assets}")
    print(f"Lookback years: {lookback_years}")
    print(f"Train ratio: {train_ratio}")

    # -----------------------------
    # 1) Load and optimize ONLY on the training asset
    # -----------------------------
    print(f"\nLoading training asset data for {training_asset}...")
    train_asset_df = get_data(training_asset, start, end)
    train_asset_df = add_indicators(train_asset_df)

    training_split_df, _ = train_test_split_df(train_asset_df, train_ratio=train_ratio)

    best_params, optimization_df = optimize_ma_crossover(
        training_split_df,
        ticker=training_asset,
        short_windows=short_windows,
        long_windows=long_windows,
        initial_capital=initial_capital,
        transaction_cost=transaction_cost,
        objective="sharpe_ratio",
    )

    optimization_df.to_csv(
        RESULTS_DIR / f"{training_asset}_training_asset_ma_optimization.csv",
        index=False
    )

    print(f"\nBest MA params learned from {training_asset}: {best_params}")

    all_summaries = []

    # -----------------------------
    # 2) Evaluate fixed strategy definitions across all assets
    # -----------------------------
    for eval_asset in evaluation_assets:
        print(f"\nEvaluating transfer performance on {eval_asset}...")
        df = get_data(eval_asset, start, end)
        df = add_indicators(df)

        full_data_path = RESULTS_DIR / f"{eval_asset}_full_dataset.csv"
        df.to_csv(full_data_path)

        train_df, test_df = train_test_split_df(df, train_ratio=train_ratio)

        # ---------- TRAIN SAMPLE ----------
        train_ma_df = ma_crossover_strategy(
            train_df,
            short_window=best_params["short_window"],
            long_window=best_params["long_window"],
        )
        all_summaries.append(
            run_strategy_pipeline(
                training_asset=training_asset,
                evaluation_asset=eval_asset,
                strategy_name=f"MA Crossover ({best_params['short_window']},{best_params['long_window']})",
                strategy_family="MA",
                sample_period="Train",
                strategy_df=train_ma_df,
                initial_capital=initial_capital,
                transaction_cost=transaction_cost,
                optimized_on=f"{training_asset} train split",
                ma_short_window=best_params["short_window"],
                ma_long_window=best_params["long_window"],
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
                training_asset=training_asset,
                evaluation_asset=eval_asset,
                strategy_name="RSI Mean Reversion",
                strategy_family="RSI",
                sample_period="Train",
                strategy_df=train_rsi_df,
                initial_capital=initial_capital,
                transaction_cost=transaction_cost,
                optimized_on="fixed parameters",
            )
        )

        train_boll_df = bollinger_mean_reversion_strategy(
            train_df,
            window=20,
            num_std=2.0,
        )
        all_summaries.append(
            run_strategy_pipeline(
                training_asset=training_asset,
                evaluation_asset=eval_asset,
                strategy_name="Bollinger Mean Reversion",
                strategy_family="Bollinger",
                sample_period="Train",
                strategy_df=train_boll_df,
                initial_capital=initial_capital,
                transaction_cost=transaction_cost,
                optimized_on="fixed parameters",
            )
        )

        train_bh_df = buy_and_hold_strategy(train_df)
        all_summaries.append(
            run_strategy_pipeline(
                training_asset=training_asset,
                evaluation_asset=eval_asset,
                strategy_name="Buy and Hold",
                strategy_family="Benchmark",
                sample_period="Train",
                strategy_df=train_bh_df,
                initial_capital=initial_capital,
                transaction_cost=0.0,
                optimized_on="not applicable",
            )
        )

        # ---------- TEST SAMPLE ----------
        test_ma_df = ma_crossover_strategy(
            test_df,
            short_window=best_params["short_window"],
            long_window=best_params["long_window"],
        )
        all_summaries.append(
            run_strategy_pipeline(
                training_asset=training_asset,
                evaluation_asset=eval_asset,
                strategy_name=f"MA Crossover ({best_params['short_window']},{best_params['long_window']})",
                strategy_family="MA",
                sample_period="Test",
                strategy_df=test_ma_df,
                initial_capital=initial_capital,
                transaction_cost=transaction_cost,
                optimized_on=f"{training_asset} train split",
                ma_short_window=best_params["short_window"],
                ma_long_window=best_params["long_window"],
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
                training_asset=training_asset,
                evaluation_asset=eval_asset,
                strategy_name="RSI Mean Reversion",
                strategy_family="RSI",
                sample_period="Test",
                strategy_df=test_rsi_df,
                initial_capital=initial_capital,
                transaction_cost=transaction_cost,
                optimized_on="fixed parameters",
            )
        )

        test_boll_df = bollinger_mean_reversion_strategy(
            test_df,
            window=20,
            num_std=2.0,
        )
        all_summaries.append(
            run_strategy_pipeline(
                training_asset=training_asset,
                evaluation_asset=eval_asset,
                strategy_name="Bollinger Mean Reversion",
                strategy_family="Bollinger",
                sample_period="Test",
                strategy_df=test_boll_df,
                initial_capital=initial_capital,
                transaction_cost=transaction_cost,
                optimized_on="fixed parameters",
            )
        )

        test_bh_df = buy_and_hold_strategy(test_df)
        all_summaries.append(
            run_strategy_pipeline(
                training_asset=training_asset,
                evaluation_asset=eval_asset,
                strategy_name="Buy and Hold",
                strategy_family="Benchmark",
                sample_period="Test",
                strategy_df=test_bh_df,
                initial_capital=initial_capital,
                transaction_cost=0.0,
                optimized_on="not applicable",
            )
        )

    # -----------------------------
    # 3) Save summaries
    # -----------------------------
    summary_df = pd.DataFrame(all_summaries)
    summary_df = summary_df.drop(columns=["ticker"], errors="ignore")
    summary_df.to_csv(RESULTS_DIR / "generalization_summary_raw.csv", index=False)

    clean_df = clean_summary(summary_df)
    clean_df.to_csv(RESULTS_DIR / "generalization_summary_clean.csv", index=False)

    # -----------------------------
    # 4) Plots
    # -----------------------------
    plot_generalization_bar_chart(
        clean_df,
        PLOTS_DIR / "generalization_sharpe.png"
    )

    plot_total_return_bar_chart(
        clean_df,
        PLOTS_DIR / "generalization_total_return.png"
    )

    plot_ma_generalization_heatmap(
        clean_df,
        PLOTS_DIR / "ma_generalization_heatmap.png"
    )

    copy_best_plots_to_assets()

    # -----------------------------
    # 5) Print final clean summary
    # -----------------------------
    print("\n" + "#" * 100)
    print("FINAL CLEAN RESEARCH SUMMARY")
    print("#" * 100)
    print(clean_df.to_string(index=False))

    print("\nSaved files:")
    print(f"- {RESULTS_DIR / 'generalization_summary_raw.csv'}")
    print(f"- {RESULTS_DIR / 'generalization_summary_clean.csv'}")
    print(f"- {PLOTS_DIR / 'generalization_sharpe.png'}")
    print(f"- {PLOTS_DIR / 'generalization_total_return.png'}")
    print(f"- {PLOTS_DIR / 'ma_generalization_heatmap.png'}")


if __name__ == "__main__":
    main()