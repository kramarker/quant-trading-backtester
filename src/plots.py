from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_price_with_signals(
    df: pd.DataFrame,
    ticker: str,
    strategy_name: str,
    save_path: str
) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df["Price"], label="Price")

    if "SMA_20" in df.columns:
        ax.plot(df.index, df["SMA_20"], label="SMA 20")
    if "SMA_50" in df.columns:
        ax.plot(df.index, df["SMA_50"], label="SMA 50")

    buys = df[(df["Position"] == 1) & (df["Position"].shift(1).fillna(0) == 0)]
    sells = df[(df["Position"] == 0) & (df["Position"].shift(1).fillna(0) == 1)]

    ax.scatter(buys.index, buys["Price"], marker="^", s=80, label="Buy")
    ax.scatter(sells.index, sells["Price"], marker="v", s=80, label="Sell")

    ax.set_title(f"{ticker} Price and Trading Signals - {strategy_name}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_equity_curve(
    df: pd.DataFrame,
    ticker: str,
    strategy_name: str,
    save_path: str
) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df["Portfolio_Value"], label=f"{strategy_name}")
    ax.plot(df.index, df["Buy_Hold_Value"], label="Buy & Hold")

    ax.set_title(f"{ticker} Equity Curve - {strategy_name}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_drawdown(
    df: pd.DataFrame,
    ticker: str,
    strategy_name: str,
    save_path: str
) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df["Drawdown"], label="Drawdown")

    ax.set_title(f"{ticker} Drawdown - {strategy_name}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_strategy_comparison(summary_df: pd.DataFrame, save_path: str) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    plot_df = summary_df.copy()
    plot_df["label"] = (
        plot_df["ticker"] + " | " + plot_df["strategy"] + " | " + plot_df["sample_period"]
    )

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(plot_df["label"], plot_df["sharpe_ratio"])

    ax.set_title("Sharpe Ratio by Ticker / Strategy / Sample")
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Sharpe Ratio")
    ax.tick_params(axis="x", rotation=90)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_ma_heatmap(optimization_df: pd.DataFrame, ticker: str, save_path: str) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    pivot = optimization_df.pivot(
        index="short_window",
        columns="long_window",
        values="sharpe_ratio"
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    ax.set_title(f"{ticker} MA Optimization Heatmap (Sharpe Ratio)")
    ax.set_xlabel("Long Window")
    ax.set_ylabel("Short Window")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Sharpe Ratio")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()