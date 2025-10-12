import pandas as pd


def ma_crossover_strategy(
    df: pd.DataFrame,
    short_window: int = 20,
    long_window: int = 50
) -> pd.DataFrame:
    """
    Long-only strategy:
    Buy when short SMA > long SMA
    Exit when short SMA <= long SMA
    """
    out = df.copy()

    short_col = f"SMA_{short_window}"
    long_col = f"SMA_{long_window}"

    if short_col not in out.columns or long_col not in out.columns:
        raise ValueError(
            f"Required columns {short_col} and/or {long_col} not found in DataFrame."
        )

    out["Signal"] = 0
    out.loc[out[short_col] > out[long_col], "Signal"] = 1
    out["Signal"] = out["Signal"].fillna(0).astype(int)

    return out


def rsi_mean_reversion_strategy(
    df: pd.DataFrame,
    rsi_window: int = 14,
    oversold: float = 30,
    exit_level: float = 55
) -> pd.DataFrame:
    """
    Long-only RSI mean reversion:
    Enter when RSI < oversold
    Exit when RSI > exit_level
    """
    out = df.copy()

    rsi_col = f"RSI_{rsi_window}"
    if rsi_col not in out.columns:
        raise ValueError(f"Required column {rsi_col} not found in DataFrame.")

    signal = []
    in_position = 0

    for value in out[rsi_col]:
        if pd.isna(value):
            signal.append(0)
            continue

        if in_position == 0 and value < oversold:
            in_position = 1
        elif in_position == 1 and value > exit_level:
            in_position = 0

        signal.append(in_position)

    out["Signal"] = signal
    out["Signal"] = out["Signal"].fillna(0).astype(int)

    return out


def buy_and_hold_strategy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Signal"] = 1
    return out