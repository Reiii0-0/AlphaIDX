"""
Efficient Frontier optimization using scipy.optimize.
"""

import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.portfolio.metrics import (
    N_PORTFOLIOS_MONTE_CARLO,
    RISK_FREE_RATE_ANNUAL,
    TRADING_DAYS_PER_YEAR,
    OptimizationError,
    portfolio_return,
    portfolio_volatility,
    sharpe_ratio,
)

logger = logging.getLogger(__name__)

__all__ = ["EfficientFrontier"]


class EfficientFrontier:
    """
    Markowitz Efficient Frontier optimizer.
    """

    def __init__(
        self,
        log_returns: pd.DataFrame,
        allow_short: bool = False,
        risk_free_rate: float = RISK_FREE_RATE_ANNUAL,
    ) -> None:
        """
        Initialize the optimizer.

        Args:
            log_returns (pd.DataFrame): Daily log returns.
            allow_short (bool): Whether to allow short selling.
            risk_free_rate (float): Annualized risk-free rate.
        """
        self.tickers = list(log_returns.columns)
        self.n_assets = len(self.tickers)
        self.rf = risk_free_rate

        # Annualized metrics
        self.mean_returns = log_returns.mean().values * TRADING_DAYS_PER_YEAR
        self.cov_matrix = log_returns.cov().values * TRADING_DAYS_PER_YEAR

        # Bounds: (0, 1) for long-only, (-0.5, 1) for short-allowed (per specs)
        if allow_short:
            self.bounds = tuple((-0.5, 1.0) for _ in range(self.n_assets))
        else:
            self.bounds = tuple((0.0, 1.0) for _ in range(self.n_assets))

        self._frontier_cache: list[dict] | None = None

        logger.info(
            f"Initialized EF with {self.n_assets} assets. "
            f"Date range: {log_returns.index[0].date()} to {log_returns.index[-1].date()}. "
            f"Annualized return range: [{self.mean_returns.min():.2%}, {self.mean_returns.max():.2%}]"
        )

    def _base_constraints(self) -> list[dict]:
        """Weights sum to 1."""
        return [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    def _initial_weights(self) -> np.ndarray:
        """Equal weights initial guess."""
        return np.array([1.0 / self.n_assets] * self.n_assets)

    def optimize_max_sharpe(self) -> dict:
        """
        Maximize Sharpe ratio (minimize negative Sharpe).

        Returns:
            dict: Portfolio metrics and weights.
        """

        def neg_sharpe(w):
            ret = portfolio_return(w, self.mean_returns)
            vol = portfolio_volatility(w, self.cov_matrix)
            return -sharpe_ratio(ret, vol, self.rf)

        result = minimize(
            neg_sharpe,
            self._initial_weights(),
            method="SLSQP",
            bounds=self.bounds,
            constraints=self._base_constraints(),
            options={"ftol": 1e-9, "maxiter": 1000},
        )

        if not result.success:
            raise OptimizationError(f"Max Sharpe optimization failed: {result.message}")

        # Post-process weights
        weights = np.round(result.x, 6)
        weights /= np.sum(weights)  # Normalize to exactly 1.0

        ret = portfolio_return(weights, self.mean_returns)
        vol = portfolio_volatility(weights, self.cov_matrix)

        return {
            "weights": dict(zip(self.tickers, weights)),
            "return": ret,
            "volatility": vol,
            "sharpe": sharpe_ratio(ret, vol, self.rf),
        }

    def optimize_min_variance(self) -> dict:
        """
        Minimize portfolio volatility.

        Returns:
            dict: Portfolio metrics and weights.
        """
        result = minimize(
            portfolio_volatility,
            self._initial_weights(),
            args=(self.cov_matrix,),
            method="SLSQP",
            bounds=self.bounds,
            constraints=self._base_constraints(),
            options={"ftol": 1e-9, "maxiter": 1000},
        )

        if not result.success:
            raise OptimizationError(f"Min Variance optimization failed: {result.message}")

        weights = np.round(result.x, 6)
        weights /= np.sum(weights)

        ret = portfolio_return(weights, self.mean_returns)
        vol = portfolio_volatility(weights, self.cov_matrix)

        return {
            "weights": dict(zip(self.tickers, weights)),
            "return": ret,
            "volatility": vol,
            "sharpe": sharpe_ratio(ret, vol, self.rf),
        }

    def generate_frontier(self, n_points: int = 200) -> list[dict]:
        """
        Generate efficient frontier points.

        Returns:
            list[dict]: List of frontier points.
        """
        if self._frontier_cache is not None:
            return self._frontier_cache

        # Find return range
        min_var = self.optimize_min_variance()
        min_ret = min_var["return"]
        # Max return is 100% in the asset with highest mean return (long-only)
        max_ret = np.max(self.mean_returns)

        target_returns = np.linspace(min_ret * 1.01, max_ret * 0.99, n_points)
        frontier = []

        for target in target_returns:
            constraints = self._base_constraints() + [
                {
                    "type": "eq",
                    "fun": lambda w, t=target: portfolio_return(w, self.mean_returns) - t,
                }
            ]
            res = minimize(
                portfolio_volatility,
                self._initial_weights(),
                args=(self.cov_matrix,),
                method="SLSQP",
                bounds=self.bounds,
                constraints=constraints,
                options={"ftol": 1e-9, "maxiter": 1000},
            )
            if res.success:
                w = res.x
                frontier.append(
                    {
                        "return": portfolio_return(w, self.mean_returns),
                        "volatility": portfolio_volatility(w, self.cov_matrix),
                        "sharpe": sharpe_ratio(
                            portfolio_return(w, self.mean_returns),
                            portfolio_volatility(w, self.cov_matrix),
                            self.rf,
                        ),
                        "weights": dict(zip(self.tickers, w)),
                    }
                )
            else:
                logging.debug(f"Frontier point at target {target:.4f} failed to converge.")

        logger.info(f"Frontier: {len(frontier)}/{n_points} points converged")
        self._frontier_cache = frontier
        return frontier

    def get_random_portfolios(self, n: int = N_PORTFOLIOS_MONTE_CARLO) -> pd.DataFrame:
        """
        Generate random portfolios using Dirichlet sampling.

        Returns:
            pd.DataFrame: Random portfolios with weights and metrics.
        """
        rng = np.random.default_rng(42)
        weights_matrix = rng.dirichlet(np.ones(self.n_assets), n)

        # Vectorized metrics
        rets = weights_matrix @ self.mean_returns
        # σ_p = sqrt(w^T · Σ · w) vectorized
        vols = np.sqrt(np.einsum("ij,jk,ik->i", weights_matrix, self.cov_matrix, weights_matrix))
        sharpes = (rets - self.rf) / vols

        data = np.hstack([weights_matrix, rets[:, None], vols[:, None], sharpes[:, None]])
        columns = self.tickers + ["return", "volatility", "sharpe"]

        return pd.DataFrame(data, columns=columns)
