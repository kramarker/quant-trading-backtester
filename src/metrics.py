import numpy as np
import pandas as pd


TRADING_DAYS = 252


def total_return(portfolio_value: pd.Series) -> float:
    return portfolio_value.iloc[-1] / portfolio_value.iloc[0] - 1


def annualized_return(portfolio_value: pd.Series) -> float:
    n_days = len(portfolio_value)
    if n_days <= 1:
        return np.nan

    total = total_return(portfolio_value)
    return (1 + total) ** (TRADING_DAYS / n_days) - 1


def annualized_volatility(strategy_returns: pd.Series) -> float:
    return strategy_returns.std() * np.sqrt(TRADING_DAYS)


def sharpe_ratio(strategy_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    vol = strategy_returns.std()
    if vol == 0 or pd.isna(vol):
        return np.nan

    daily_rf = risk_free_rate / TRADING_DAYS
    excess_return = strategy_returns.mean() - daily_rf
    return (excess_return / vol) * np.sqrt(TRADING_DAYS)


def max_drawdown(portfolio_value: pd.Series) -> float:
    running_max = portfolio_value.cummax()
    drawdown = portfolio_value / running_max - 1
    return drawdown.min()


def trade_statistics(trades_df: pd.DataFrame) -> dict:
    if trades_df.empty:
        return {
            "num_trades": 0,
            "win_rate": np.nan,
            "avg_trade_return": np.nan,
            "best_trade_return": np.nan,
            "worst_trade_return": np.nan,
            "avg_holding_days": np.nan,
        }

    trade_returns = trades_df["trade_return"]

    return {
        "num_trades": int(len(trades_df)),
        "win_rate": float((trade_returns > 0).mean()),
        "avg_trade_return": float(trade_returns.mean()),
        "best_trade_return": float(trade_returns.max()),
        "worst_trade_return": float(trade_returns.min()),
        "avg_holding_days": float(trades_df["holding_days"].mean()),
    }


def summarize_performance(
    strategy_name: str,
    ticker: str,
    results_df: pd.DataFrame,
    trades_df: pd.DataFrame
) -> dict:
    strategy_returns = results_df["Strategy_Return"].dropna()
    portfolio_value = results_df["Portfolio_Value"].dropna()

    trade_stats = trade_statistics(trades_df)

    summary = {
        "ticker": ticker,
        "strategy": strategy_name,
        "total_return": float(total_return(portfolio_value)),
        "annualized_return": float(annualized_return(portfolio_value)),
        "annualized_volatility": float(annualized_volatility(strategy_returns)),
        "sharpe_ratio": float(sharpe_ratio(strategy_returns)),
        "max_drawdown": float(max_drawdown(portfolio_value)),
    }

    summary.update(trade_stats)
    return summary