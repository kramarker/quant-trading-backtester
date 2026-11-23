import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def price_df():
    """Small deterministic price series with Price and Return columns, indexed by business days."""
    idx = pd.bdate_range("2024-01-01", periods=8)
    prices = pd.Series([100, 102, 101, 105, 104, 108, 107, 110], index=idx, dtype=float)
    return pd.DataFrame({"Price": prices, "Return": prices.pct_change().fillna(0.0)})


@pytest.fixture
def long_df():
    """Longer synthetic random-walk series for optimizer and walk-forward tests."""
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=900)
    returns = rng.normal(0.0004, 0.01, size=len(idx))
    prices = 100 * np.cumprod(1 + returns)
    df = pd.DataFrame({"Price": prices}, index=idx)
    df["Return"] = df["Price"].pct_change().fillna(0.0)
    return df
