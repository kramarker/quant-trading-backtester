import pandas as pd
import yfinance as yf


def get_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download and clean historical price data for a single ticker.
    """
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for ticker {ticker}.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for {ticker}: {missing}")

    df = df[required_cols].copy()
    df["Price"] = df["Close"]
    df["Return"] = df["Price"].pct_change()

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.dropna()

    return df