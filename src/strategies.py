import pandas as pd


def ma_crossover_strategy(
    df: pd.DataFrame,
    short_window: int = 20,
    long_window: int = 50
) -> pd.DataFrame:
    """
    Long-only moving average crossover strategy.
    Buy when short SMA > long SMA.
    Exit when short SMA <= long SMA.
    """
    out = df.copy()

    short_col = f"SMA_{short_window}"
    long_col = f"SMA_{long_window}"

    if short_col not in out.columns:
        out[short_col] = out["Price"].rolling(window=short_window).mean()

    if long_col not in out.columns:
        out[long_col] = out["Price"].rolling(window=long_window).mean()

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


def bollinger_mean_reversion_strategy(
    df: pd.DataFrame,
    window: int = 20,
    num_std: float = 2.0
) -> pd.DataFrame:
    """
    Long-only Bollinger Band mean reversion:
    Enter when price falls below lower band.
    Exit when price reverts above rolling mean.
    """
    out = df.copy()

    out["BB_Mid"] = out["Price"].rolling(window).mean()
    out["BB_Std"] = out["Price"].rolling(window).std()
    out["BB_Upper"] = out["BB_Mid"] + num_std * out["BB_Std"]
    out["BB_Lower"] = out["BB_Mid"] - num_std * out["BB_Std"]

    signal = []
    in_position = 0

    for i, row in enumerate(out.itertuples()):
        price = row.Price
        mid = row.BB_Mid
        lower = row.BB_Lower

        if pd.isna(price) or pd.isna(mid) or pd.isna(lower):
            signal.append(0)
            continue

        if in_position == 0 and price < lower:
            in_position = 1
        elif in_position == 1 and price > mid:
            in_position = 0

        signal.append(in_position)

    out["Signal"] = signal
    out["Signal"] = out["Signal"].fillna(0).astype(int)

    return out


def buy_and_hold_strategy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Signal"] = 1
    return out