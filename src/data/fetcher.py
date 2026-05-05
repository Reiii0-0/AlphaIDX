"""
Data fetching and preprocessing for stock prices.
"""

import hashlib
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from src.portfolio.metrics import (
    RISK_FREE_RATE_ANNUAL,
    TRADING_DAYS_PER_YEAR,
    sharpe_ratio,
)

logger = logging.getLogger(__name__)

__all__ = ["StockDataFetcher"]


class StockDataFetcher:
    """
    Fetches and processes stock price data from yfinance.
    """

    def __init__(self, tickers: list[str], start: str, end: str) -> None:
        """
        Initialize the fetcher.

        Args:
            tickers (list[str]): List of stock tickers.
            start (str): Start date (YYYY-MM-DD).
            end (str): End date (YYYY-MM-DD).
        """
        self.tickers = sorted(tickers)
        self.start = start
        self.end = end
        self.cache_dir = Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch(self) -> pd.DataFrame:
        """
        Fetch stock prices from cache or yfinance.

        Returns:
            pd.DataFrame: Adjusted close prices.
        """
        cached_data = self._load_from_cache()
        if cached_data is not None:
            return cached_data

        logger.info(f"Downloading data for {len(self.tickers)} tickers...")
        raw = yf.download(
            tickers=self.tickers,
            start=self.start,
            end=self.end,
            auto_adjust=True,
            progress=False,
            threads=True,
        )

        if raw.empty:
            raise ValueError("No data downloaded from yfinance.")

        # Extract Adjusted Close (yfinance auto_adjust=True makes "Close" adjusted)
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"]
        else:
            prices = raw[["Close"]]
            prices.columns = self.tickers

        # Ensure all requested tickers are columns
        prices = prices[self.tickers]

        prices = self._validate_prices(prices)
        self._save_to_cache(prices)

        return prices

    def to_log_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Convert price data to log returns.

        Args:
            prices (pd.DataFrame): Stock prices.

        Returns:
            pd.DataFrame: Log returns.
        """
        # r_t = ln(P_t / P_{t-1})
        log_returns = np.log(prices / prices.shift(1)).dropna()

        # Validate output
        assert not log_returns.isnull().any().any(), "NaN in log returns"
        assert log_returns.shape[0] > 252, "Less than 1 year of data"
        assert (log_returns.abs() < 0.5).all().all(), "Suspiciously large daily returns (>50%)"

        return log_returns

    def get_summary_stats(self, log_returns: pd.DataFrame) -> pd.DataFrame:
        """
        Compute summary statistics for log returns.

        Args:
            log_returns (pd.DataFrame): Log returns.

        Returns:
            pd.DataFrame: Summary statistics.
        """
        stats = pd.DataFrame(index=log_returns.columns)
        stats["annual_return"] = log_returns.mean() * TRADING_DAYS_PER_YEAR
        stats["annual_volatility"] = log_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        stats["sharpe_ratio"] = stats.apply(
            lambda row: sharpe_ratio(row["annual_return"], row["annual_volatility"]),
            axis=1,
        )
        stats["skewness"] = log_returns.skew()
        stats["kurtosis"] = log_returns.kurtosis()

        return stats

    def _cache_key(self) -> str:
        """Generate a unique MD5 hash for the current request."""
        payload = json.dumps(self.tickers + [self.start, self.end])
        return hashlib.md5(payload.encode()).hexdigest()

    def _load_from_cache(self) -> pd.DataFrame | None:
        """Load data from cache if it exists and is less than 24h old."""
        cache_file = self.cache_dir / f"{self._cache_key()}.csv"
        if cache_file.exists():
            age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if age < timedelta(hours=24):
                logger.info(f"Loading from cache: {cache_file}")
                return pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return None

    def _save_to_cache(self, prices: pd.DataFrame) -> None:
        """Save data to cache in CSV format."""
        cache_file = self.cache_dir / f"{self._cache_key()}.csv"
        prices.to_csv(cache_file)
        logger.info(f"Saved to cache: {cache_file}")

    def _validate_prices(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Apply data quality rules from DATA.md."""
        # Step 1: forward fill gaps of up to 3 consecutive days
        prices = prices.ffill(limit=3)

        # Step 2: drop tickers with >5% NaN
        nan_pct = prices.isna().mean()
        bad_tickers = nan_pct[nan_pct > 0.05].index.tolist()

        if bad_tickers:
            logger.warning(f"Dropping tickers with >5% NaN: {bad_tickers}")
            prices = prices.drop(columns=bad_tickers)

        if prices.shape[1] < 2:
            raise ValueError("Fewer than 2 valid tickers remain after quality check.")

        # Step 3: drop remaining NaN rows
        prices = prices.dropna()
        
        return prices
