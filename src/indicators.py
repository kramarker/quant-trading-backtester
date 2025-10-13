import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    out = 100 - (100 / (1 + rs))
    return out


def rolling_volatility(series: pd.Series, window: int = 20) -> pd.Series:
    return series.pct_change().rolling(window=window).std() * (252 ** 0.5)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for window in [10, 15, 20, 30, 50, 100, 150, 200]:
        out[f"SMA_{window}"] = sma(out["Price"], window)

    out["EMA_12"] = ema(out["Price"], 12)
    out["EMA_26"] = ema(out["Price"], 26)
    out["RSI_14"] = rsi(out["Price"], 14)
    out["Vol_20"] = rolling_volatility(out["Price"], 20)

    return out