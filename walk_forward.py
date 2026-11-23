"""Run walk-forward validation of the MA crossover strategy on the cached daily data.

Usage:
    python walk_forward.py                 # SPY, QQQ, AAPL with default windows
    python walk_forward.py --train 504 --test 63

Reads data/<TICKER>_data.csv (written by main.py), so it needs no network access.
Writes results/walk_forward/ and plots/walk_forward_<TICKER>.png.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.walk_forward import walk_forward_ma

SHORT_WINDOWS = [10, 15, 20, 30]
LONG_WINDOWS = [50, 100, 150, 200]


def load_cached(ticker: str) -> pd.DataFrame:
    path = Path("data") / f"{ticker}_data.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run main.py first to download data.")
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    required = {"Price", "Return"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} is missing columns: {required - set(df.columns)}")
    return df


def plot_oos(ticker: str, oos_returns: pd.Series, bh_returns: pd.Series, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot((1 + oos_returns).cumprod(), label="MA crossover (walk-forward, out of sample)")
    ax.plot((1 + bh_returns).cumprod(), label="Buy and hold (same dates)")
    ax.set_title(f"{ticker}: stitched out-of-sample growth of $1")
    ax.set_ylabel("Growth of $1")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", nargs="+", default=["SPY", "QQQ", "AAPL"])
    parser.add_argument("--train", type=int, default=504, help="training window in trading days (default 2 years)")
    parser.add_argument("--test", type=int, default=63, help="test window in trading days (default 1 quarter)")
    parser.add_argument("--cost", type=float, default=0.001, help="transaction cost per position change")
    args = parser.parse_args()

    out_dir = Path("results/walk_forward")
    out_dir.mkdir(parents=True, exist_ok=True)
    Path("plots").mkdir(exist_ok=True)

    summaries = []
    for ticker in args.tickers:
        df = load_cached(ticker)
        folds_df, oos_returns, summary = walk_forward_ma(
            df, ticker, SHORT_WINDOWS, LONG_WINDOWS,
            train_size=args.train, test_size=args.test, transaction_cost=args.cost,
        )
        folds_df.to_csv(out_dir / f"{ticker}_folds.csv", index=False)
        bh_returns = df.loc[oos_returns.index, "Return"]
        plot_oos(ticker, oos_returns, bh_returns, Path("plots") / f"walk_forward_{ticker}.png")
        summaries.append(summary)
        print(f"{ticker}: {summary['num_folds']} folds, OOS Sharpe {summary['oos_sharpe']:.2f} "
              f"vs buy-and-hold {summary['buy_hold_sharpe']:.2f}, "
              f"{summary['distinct_window_pairs_chosen']} distinct window pairs chosen")

    summary_df = pd.DataFrame(summaries)
    numeric = summary_df.select_dtypes("number").columns
    summary_df[numeric] = summary_df[numeric].round(4)
    summary_df.to_csv(out_dir / "walk_forward_summary.csv", index=False)
    print()
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
