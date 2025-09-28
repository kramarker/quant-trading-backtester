import pandas as pd
import yfinance as yf


def get_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)

    # flatten multiindex if it exists
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # create clean price column
    df["Price"] = df["Close"]

    # returns
    df["Return"] = df["Price"].pct_change()

    df = df.dropna()

    return df