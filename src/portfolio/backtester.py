"""
Backtesting engine for Out-of-Sample (OOS) validation.
"""

import logging
import numpy as np
import pandas as pd
from src.portfolio.metrics import portfolio_return, portfolio_volatility, sharpe_ratio, max_drawdown

logger = logging.getLogger(__name__)

__all__ = ["Backtester"]

class Backtester:
    """
    Validates optimized portfolios against realized market data (Out-of-Sample).
    """

    def __init__(self, is_log_returns: pd.DataFrame, oos_log_returns: pd.DataFrame) -> None:
        """
        Initialize the backtester.

        Args:
            is_log_returns (pd.DataFrame): In-Sample (IS) log returns used for optimization.
            oos_log_returns (pd.DataFrame): Out-of-Sample (OOS) realized log returns.
        """
        self.is_data = is_log_returns
        self.oos_data = oos_log_returns
        self.tickers = list(is_log_returns.columns)

    def run_backtest(self, weights_dict: dict, label: str) -> dict:
        """
        Compare In-Sample (Predicted) vs Out-of-Sample (Realized) performance.

        Args:
            weights_dict (dict): Ticker-to-weight mapping.
            label (str): Name of the portfolio (e.g., 'Max Sharpe').

        Returns:
            dict: Performance metrics comparison.
        """
        weights = np.array([weights_dict[t] for t in self.tickers])
        
        # ── 1. Realized Returns (Cumulative) ──
        # Portfolio realized log returns
        realized_log_rets = self.oos_data @ weights
        cumulative_realized = np.exp(realized_log_rets.cumsum()) - 1
        
        # ── 2. Realized Metrics ──
        realized_ann_ret = realized_log_rets.mean() * 252
        realized_ann_vol = realized_log_rets.std() * np.sqrt(252)
        realized_sharpe  = (realized_ann_ret - 0.0625) / realized_ann_vol
        realized_mdd     = max_drawdown(np.exp(realized_log_rets.cumsum()).values)

        # ── 3. Predicted Metrics (Historical IS) ──
        is_log_rets = self.is_data @ weights
        pred_ann_ret = is_log_rets.mean() * 252
        pred_ann_vol = is_log_rets.std() * np.sqrt(252)
        
        logger.info(f"Backtest [{label}] complete. Realized Sharpe: {realized_sharpe:.2f} vs Predicted: {(pred_ann_ret-0.0625)/pred_ann_vol:.2f}")

        return {
            "label": label,
            "cumulative_returns": cumulative_realized,
            "realized_return": realized_ann_ret,
            "realized_volatility": realized_ann_vol,
            "realized_sharpe": realized_sharpe,
            "realized_mdd": realized_mdd,
            "predicted_return": pred_ann_ret,
            "predicted_volatility": pred_ann_vol,
        }

    def get_benchmark_performance(self) -> pd.Series:
        """Equally weighted portfolio performance as a baseline."""
        n = len(self.tickers)
        equal_weights = np.array([1/n] * n)
        realized_log_rets = self.oos_data @ equal_weights
        return np.exp(realized_log_rets.cumsum()) - 1
