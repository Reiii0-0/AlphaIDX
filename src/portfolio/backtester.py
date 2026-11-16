"""
Backtesting engine for Out-of-Sample (OOS) validation.
"""

import logging
from typing import Literal

import numpy as np
import pandas as pd

from src.portfolio.advanced_models import HierarchicalRiskParityOptimizer
from src.portfolio.metrics import (
    RISK_FREE_RATE_ANNUAL,
    TRADING_DAYS_PER_YEAR,
    max_drawdown,
    sharpe_ratio,
)
from src.portfolio.optimizer import EfficientFrontier

logger = logging.getLogger(__name__)

__all__ = ["Backtester"]


class Backtester:
    """
    Validates optimized portfolios against Out-of-Sample (OOS) market data.
    """

    def __init__(self, is_log_returns: pd.DataFrame, oos_log_returns: pd.DataFrame) -> None:
        """
        Initialize the backtester.

        Args:
            is_log_returns (pd.DataFrame): In-Sample (IS) log returns used for optimization.
            oos_log_returns (pd.DataFrame): Out-of-Sample (OOS) realized log returns.
        """
        if is_log_returns.empty or oos_log_returns.empty:
            raise ValueError("is_log_returns and oos_log_returns must not be empty.")
        self.is_data = is_log_returns
        self.oos_data = oos_log_returns[list(is_log_returns.columns)]
        self.tickers = list(is_log_returns.columns)

    @staticmethod
    def _metrics_from_wealth(wealth: pd.Series, rf: float) -> tuple[float, float, float, float]:
        prev = wealth.shift(1).fillna(1.0)
        port_log_rets = np.log(wealth / prev)
        ann_ret = float(port_log_rets.mean() * TRADING_DAYS_PER_YEAR)
        ann_vol = float(port_log_rets.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
        sr = sharpe_ratio(ann_ret, ann_vol, rf)
        mdd = max_drawdown(wealth.to_numpy(dtype=float))
        return ann_ret, ann_vol, sr, mdd

    def run_backtest(
        self,
        weights_dict: dict[str, float],
        label: str,
        risk_free_rate: float = RISK_FREE_RATE_ANNUAL,
    ) -> dict:
        """
        Compare In-Sample (Predicted) vs Out-of-Sample (Realized buy-and-hold) performance.
        """
        weights = np.array([weights_dict[t] for t in self.tickers], dtype=float)

        # Buy-and-hold asset wealth accounts for exact daily weight drift.
        asset_wealth = np.exp(self.oos_data.cumsum())
        port_wealth = pd.Series(
            asset_wealth.to_numpy(dtype=float) @ weights, index=self.oos_data.index
        )
        cumulative_realized = port_wealth - 1.0

        realized_ann_ret, realized_ann_vol, realized_sharpe, realized_mdd = (
            self._metrics_from_wealth(port_wealth, risk_free_rate)
        )

        is_log_rets = self.is_data @ weights
        pred_ann_ret = float(is_log_rets.mean() * TRADING_DAYS_PER_YEAR)
        pred_ann_vol = float(is_log_rets.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
        pred_sharpe = sharpe_ratio(pred_ann_ret, pred_ann_vol, risk_free_rate)

        logger.info(
            f"Backtest [{label}] complete. Realized Sharpe: {realized_sharpe:.2f} vs Predicted: {pred_sharpe:.2f}"
        )

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

    def run_walk_forward(
        self,
        lookback_days: int = TRADING_DAYS_PER_YEAR,
        rebalance_freq: int = 63,
        strategy: Literal["max_sharpe", "min_variance", "min_cvar", "hrp"] = "max_sharpe",
        cov_method: Literal["sample", "ledoit_wolf"] = "ledoit_wolf",
        transaction_cost: float = 0.001,
        allow_short: bool = False,
        max_weight: float = 1.0,
        risk_free_rate: float = RISK_FREE_RATE_ANNUAL,
        label: str | None = None,
    ) -> dict:
        """
        Run a rolling walk-forward backtest with periodic rebalancing and turnover cost.
        """
        if lookback_days < 20 or rebalance_freq < 1:
            raise ValueError("lookback_days must be >= 20 and rebalance_freq >= 1.")

        full_log_rets = pd.concat([self.is_data, self.oos_data])
        oos_start_pos = len(self.is_data)
        oos_simple = np.exp(self.oos_data.to_numpy(dtype=float)) - 1.0
        n_oos = len(self.oos_data)

        wealth_vals = np.empty(n_oos, dtype=float)
        current_wealth = 1.0
        current_weights = np.zeros(len(self.tickers), dtype=float)
        turnovers: list[float] = []

        for t in range(n_oos):
            if t % rebalance_freq == 0:
                hist_end = oos_start_pos + t
                hist_start = max(0, hist_end - lookback_days)
                window = full_log_rets.iloc[hist_start:hist_end]

                if strategy == "hrp":
                    res = HierarchicalRiskParityOptimizer(window).optimize()
                else:
                    ef = EfficientFrontier(
                        window,
                        allow_short=allow_short,
                        risk_free_rate=risk_free_rate,
                        cov_method=cov_method,
                        max_weight=max_weight,
                    )
                    if strategy == "max_sharpe":
                        res = ef.optimize_max_sharpe(
                            current_weights=current_weights if t > 0 else None,
                            transaction_cost=transaction_cost if t > 0 else 0.0,
                        )
                    elif strategy == "min_variance":
                        res = ef.optimize_min_variance()
                    elif strategy == "min_cvar":
                        res = ef.optimize_min_cvar()
                    else:
                        raise ValueError(f"Unsupported strategy: {strategy}")

                target_weights = np.array([res["weights"][tk] for tk in self.tickers], dtype=float)
                turnover = float(np.sum(np.abs(target_weights - current_weights)))
                turnovers.append(turnover)
                current_wealth *= max(0.0, 1.0 - transaction_cost * turnover)
                current_weights = target_weights

            asset_mult = 1.0 + oos_simple[t]
            gross = current_weights * asset_mult
            day_factor = float(np.sum(gross))
            current_wealth *= day_factor
            wealth_vals[t] = current_wealth
            if day_factor > 0.0:
                current_weights = gross / day_factor

        port_wealth = pd.Series(wealth_vals, index=self.oos_data.index)
        ann_ret, ann_vol, sr, mdd = self._metrics_from_wealth(port_wealth, risk_free_rate)

        return {
            "label": label or f"Walk-Forward {strategy} ({cov_method})",
            "cumulative_returns": port_wealth - 1.0,
            "realized_return": ann_ret,
            "realized_volatility": ann_vol,
            "realized_sharpe": sr,
            "realized_mdd": mdd,
            "rebalances": len(turnovers),
            "avg_turnover": float(np.mean(turnovers)) if turnovers else 0.0,
        }

    def get_benchmark_performance(self) -> pd.Series:
        """Equally weighted buy-and-hold portfolio cumulative return as baseline."""
        n = len(self.tickers)
        equal_weights = np.full(n, 1.0 / n)
        asset_wealth = np.exp(self.oos_data.cumsum())
        return pd.Series(
            asset_wealth.to_numpy(dtype=float) @ equal_weights - 1.0,
            index=self.oos_data.index,
        )
