import os
import time
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.data.fetcher import StockDataFetcher


def _make_mock_yf_df(tickers: list[str], n_days: int = 240, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    rets = rng.normal(0.0003, 0.012, size=(n_days, len(tickers)))
    prices = 1000.0 * np.exp(np.cumsum(rets, axis=0))
    cols = pd.MultiIndex.from_product([["Close"], sorted(tickers)])
    return pd.DataFrame(prices, index=dates, columns=cols)


def test_fetcher_sorts_tickers_and_caches(tmp_path):
    raw = _make_mock_yf_df(["BBCA.JK", "ASII.JK"])
    with patch("src.data.fetcher.yf.download", return_value=raw) as mock_dl:
        f1 = StockDataFetcher(
            ["BBCA.JK", "ASII.JK"], "2024-01-01", "2024-12-31", cache_dir=tmp_path
        )
        df1 = f1.fetch()
        assert list(df1.columns) == ["ASII.JK", "BBCA.JK"]
        assert mock_dl.call_count == 1

        # Second call with reversed ticker order should hit the same 24h cache.
        f2 = StockDataFetcher(
            ["ASII.JK", "BBCA.JK"], "2024-01-01", "2024-12-31", cache_dir=tmp_path
        )
        df2 = f2.fetch()
        assert mock_dl.call_count == 1
        pd.testing.assert_frame_equal(df2, df1, check_freq=False)


def test_fetcher_expires_cache_after_24h(tmp_path):
    raw = _make_mock_yf_df(["BBCA.JK", "TLKM.JK"])
    with patch("src.data.fetcher.yf.download", return_value=raw) as mock_dl:
        fetcher = StockDataFetcher(
            ["BBCA.JK", "TLKM.JK"], "2024-01-01", "2024-12-31", cache_dir=tmp_path
        )
        fetcher.fetch()
        assert mock_dl.call_count == 1

        cache_file = next(tmp_path.glob("*.csv"))
        stale_ts = time.time() - 25 * 3600
        os.utime(cache_file, (stale_ts, stale_ts))

        fetcher.fetch()
        assert mock_dl.call_count == 2


def test_validate_prices_drops_bad_ticker_and_ffills(tmp_path):
    raw = _make_mock_yf_df(["A.JK", "B.JK", "BAD.JK"], n_days=100)
    # 2-day gap in A.JK should be forward-filled; >5% NaN in BAD.JK should drop BAD.JK.
    raw.iloc[10:12, raw.columns.get_loc(("Close", "A.JK"))] = np.nan
    raw.iloc[5:20, raw.columns.get_loc(("Close", "BAD.JK"))] = np.nan

    with patch("src.data.fetcher.yf.download", return_value=raw):
        fetcher = StockDataFetcher(
            ["A.JK", "B.JK", "BAD.JK"], "2024-01-01", "2024-06-01", cache_dir=tmp_path
        )
        clean = fetcher.fetch()
        assert list(clean.columns) == ["A.JK", "B.JK"]
        assert not clean.isna().any().any()


def test_validate_prices_raises_when_fewer_than_two_valid(tmp_path):
    raw = _make_mock_yf_df(["A.JK", "BAD.JK"], n_days=100)
    raw.iloc[5:25, raw.columns.get_loc(("Close", "BAD.JK"))] = np.nan

    with patch("src.data.fetcher.yf.download", return_value=raw):
        fetcher = StockDataFetcher(
            ["A.JK", "BAD.JK"], "2024-01-01", "2024-06-01", cache_dir=tmp_path
        )
        with pytest.raises(ValueError, match="Fewer than 2 valid tickers"):
            fetcher.fetch()


def test_to_log_returns_validation_and_min_days(tmp_path):
    fetcher = StockDataFetcher(["A.JK", "B.JK"], "2024-01-01", "2024-12-31", cache_dir=tmp_path)
    dates = pd.bdate_range("2024-01-01", periods=240)
    prices = pd.DataFrame(
        {"A.JK": np.linspace(100, 120, 240), "B.JK": np.linspace(200, 210, 240)}, index=dates
    )

    # 1-year lookback (~239 return rows) succeeds with default min_trading_days=200.
    log_rets = fetcher.to_log_returns(prices)
    assert len(log_rets) == 239

    with pytest.raises(ValueError, match="Insufficient trading days"):
        fetcher.to_log_returns(prices, min_trading_days=252)

    # Extreme single-day jump (>= 50% log return) raises ValueError.
    bad_prices = prices.copy()
    bad_prices.iloc[50, 0] = bad_prices.iloc[49, 0] * 3.0
    with pytest.raises(ValueError, match="Suspiciously large daily log returns"):
        fetcher.to_log_returns(bad_prices)
