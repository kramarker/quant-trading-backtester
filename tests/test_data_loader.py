import pandas as pd
import pytest

import src.data_loader as data_loader


def fake_download(frame):
    def _download(*args, **kwargs):
        return frame.copy()
    return _download


def ohlcv(n=5):
    idx = pd.bdate_range("2024-01-01", periods=n)
    close = pd.Series(range(100, 100 + n), index=idx, dtype=float)
    return pd.DataFrame({"Open": close, "High": close + 1, "Low": close - 1, "Close": close, "Volume": 1_000}, index=idx)


def test_get_data_builds_price_and_return_without_network(monkeypatch):
    monkeypatch.setattr(data_loader.yf, "download", fake_download(ohlcv()))
    df = data_loader.get_data("SPY", "2024-01-01", "2024-02-01")
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume", "Price", "Return"]
    assert len(df) == 4  # first row has no return and is dropped
    assert df["Return"].iloc[0] == pytest.approx(101 / 100 - 1)


def test_get_data_flattens_multiindex_columns(monkeypatch):
    frame = ohlcv()
    frame.columns = pd.MultiIndex.from_product([frame.columns, ["SPY"]])
    monkeypatch.setattr(data_loader.yf, "download", fake_download(frame))
    assert "Price" in data_loader.get_data("SPY", "2024-01-01", "2024-02-01").columns


def test_get_data_raises_on_empty_download(monkeypatch):
    monkeypatch.setattr(data_loader.yf, "download", fake_download(pd.DataFrame()))
    with pytest.raises(ValueError):
        data_loader.get_data("ZZZZ", "2024-01-01", "2024-02-01")
