import pandas as pd


def generate_trade_log(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a trade log from Position and Price columns.
    Assumes long-only trading with Position in {0,1}.
    """
    trades = []
    in_trade = False
    entry_date = None
    entry_price = None

    for date, row in df.iterrows():
        position = row["Position"]
        price = row["Price"]

        if not in_trade and position == 1:
            in_trade = True
            entry_date = date
            entry_price = price

        elif in_trade and position == 0:
            exit_date = date
            exit_price = price
            trade_return = exit_price / entry_price - 1
            holding_days = (exit_date - entry_date).days

            trades.append({
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "trade_return": trade_return,
                "holding_days": holding_days,
            })

            in_trade = False
            entry_date = None
            entry_price = None

    # Close open trade on last available date
    if in_trade:
        exit_date = df.index[-1]
        exit_price = df["Price"].iloc[-1]
        trade_return = exit_price / entry_price - 1
        holding_days = (exit_date - entry_date).days

        trades.append({
            "entry_date": entry_date,
            "exit_date": exit_date,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "trade_return": trade_return,
            "holding_days": holding_days,
        })

    return pd.DataFrame(trades)


def run_backtest(
    df: pd.DataFrame,
    initial_capital: float = 10000.0,
    transaction_cost: float = 0.001
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run a long-only backtest using the Signal column.
    Signal_t determines exposure for the NEXT day via Position = Signal.shift(1).
    """
    out = df.copy()

    if "Signal" not in out.columns:
        raise ValueError("DataFrame must contain a 'Signal' column.")

    out["Signal"] = out["Signal"].fillna(0).astype(int)

    # Position used on today's return is yesterday's signal
    out["Position"] = out["Signal"].shift(1).fillna(0).astype(int)

    # Detect trades when position changes
    out["Trade"] = out["Position"].diff().abs().fillna(out["Position"]).astype(int)

    # Gross strategy returns
    out["Gross_Strategy_Return"] = out["Position"] * out["Return"]

    # Transaction cost charged when a trade occurs
    out["Transaction_Cost"] = out["Trade"] * transaction_cost

    out["Strategy_Return"] = out["Gross_Strategy_Return"] - out["Transaction_Cost"]

    # Portfolio value
    out["Portfolio_Value"] = initial_capital * (1 + out["Strategy_Return"]).cumprod()

    # Benchmark buy-and-hold
    out["Buy_Hold_Value"] = initial_capital * (1 + out["Return"]).cumprod()

    # Drawdown
    out["Running_Max"] = out["Portfolio_Value"].cummax()
    out["Drawdown"] = out["Portfolio_Value"] / out["Running_Max"] - 1

    trades_df = generate_trade_log(out)

    return out, trades_df