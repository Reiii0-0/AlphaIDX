"""
Efficient Frontier optimization using scipy.optimize.
"""

import logging
from typing import Literal

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

__all__ = ["EfficientFrontier", "ledoit_wolf_shrinkage"]


def ledoit_wolf_shrinkage(log_returns: pd.DataFrame) -> np.ndarray:
    """
    Compute annualized covariance via analytical Ledoit-Wolf shrinkage toward scaled identity.

    Args:
        log_returns (pd.DataFrame): Daily log returns, shape (T, N).

    Returns:
        np.ndarray: Shrunk annualized covariance matrix, shape (N, N).
    """
    x = log_returns.to_numpy(dtype=float)
    n_obs, n_assets = x.shape
    if n_obs <= 1:
        raise ValueError("Need at least 2 observations for covariance estimation.")

    xc = x - x.mean(axis=0, keepdims=True)
    sample = (xc.T @ xc) / n_obs
    mu = np.trace(sample) / n_assets
    target = mu * np.eye(n_assets)

    gamma = float(np.sum((sample - target) ** 2))
    if gamma == 0.0:
        return sample * TRADING_DAYS_PER_YEAR

    sq_norms = np.sum(xc**2, axis=1)
    quad = np.einsum("ti,ij,tj->t", xc, sample, xc)
    sample_norm_sq = float(np.sum(sample**2))
    pi = float(np.sum(sq_norms**2 - 2.0 * quad + sample_norm_sq)) / (n_obs**2)

    shrinkage = min(1.0, max(0.0, pi / gamma))
    shrunk_daily = shrinkage * target + (1.0 - shrinkage) * sample
    return shrunk_daily * TRADING_DAYS_PER_YEAR


class EfficientFrontier:
    """
    Markowitz Efficient Frontier optimizer.
    """

    def __init__(
        self,
        log_returns: pd.DataFrame,
        allow_short: bool = False,
        risk_free_rate: float = RISK_FREE_RATE_ANNUAL,
        cov_method: Literal["sample", "ledoit_wolf"] = "sample",
        mean_returns: np.ndarray | None = None,
        cov_matrix: np.ndarray | None = None,
        max_weight: float = 1.0,
    ) -> None:
        """
        Initialize the optimizer.

        Args:
            log_returns (pd.DataFrame): Daily log returns.
            allow_short (bool): Whether to allow short selling (-0.5 to max_weight bounds).
            risk_free_rate (float): Annualized risk-free rate.
            cov_method (Literal["sample", "ledoit_wolf"]): Covariance estimator.
            mean_returns (np.ndarray | None): Optional custom annualized mean returns.
            cov_matrix (np.ndarray | None): Optional custom annualized covariance matrix.
            max_weight (float): Maximum weight per asset in (0, 1], with max_weight * n_assets >= 1.
        """
        self.tickers = list(log_returns.columns)
        self.n_assets = len(self.tickers)
        self.rf = risk_free_rate
        self.allow_short = allow_short
        if max_weight <= 0.0 or max_weight > 1.0 or max_weight * self.n_assets < 1.0 - 1e-9:
            raise ValueError(f"max_weight ({max_weight}) must be in [1/{self.n_assets}, 1.0].")
        self.max_weight = float(max_weight)
        self._returns_matrix = np.round(log_returns.to_numpy(dtype=float), 12)

        default_mean = self._returns_matrix.mean(axis=0) * TRADING_DAYS_PER_YEAR
        if cov_method == "ledoit_wolf":
            default_cov = ledoit_wolf_shrinkage(log_returns)
        elif cov_method == "sample":
            default_cov = log_returns.cov().to_numpy(dtype=float) * TRADING_DAYS_PER_YEAR
        else:
            raise ValueError(f"Unsupported cov_method: {cov_method}")

        self.mean_returns = default_mean
        self.cov_matrix = default_cov
        self._frontier_cache: list[dict] | None = None
        self.update_estimates(mean_returns=mean_returns, cov_matrix=cov_matrix)

        lower = -0.5 if allow_short else 0.0
        self.bounds = tuple((lower, self.max_weight) for _ in range(self.n_assets))

        logger.info(
            f"Initialized EF ({cov_method}, max_weight={self.max_weight:.2f}) with {self.n_assets} assets. "
            f"Date range: {log_returns.index[0].date()} to {log_returns.index[-1].date()}. "
            f"Annualized return range: [{self.mean_returns.min():.2%}, {self.mean_returns.max():.2%}]"
        )

    def update_estimates(
        self,
        mean_returns: np.ndarray | None = None,
        cov_matrix: np.ndarray | None = None,
    ) -> None:
        """Replace expected return or covariance estimates and clear cached frontier."""
        if mean_returns is not None:
            arr = np.asarray(mean_returns, dtype=float)
            if arr.shape != (self.n_assets,):
                raise ValueError(
                    f"Expected mean_returns shape {(self.n_assets,)}, got {arr.shape}."
                )
            self.mean_returns = arr
        if cov_matrix is not None:
            mat = np.asarray(cov_matrix, dtype=float)
            if mat.shape != (self.n_assets, self.n_assets):
                raise ValueError(
                    f"Expected cov_matrix shape {(self.n_assets, self.n_assets)}, got {mat.shape}."
                )
            self.cov_matrix = mat
        self._frontier_cache = None

    def _base_constraints(self) -> list[dict]:
        """Weights sum to 1."""
        return [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    def _initial_weights(self) -> np.ndarray:
        """Equal weights initial guess."""
        return np.full(self.n_assets, 1.0 / self.n_assets)

    def _format_result(self, raw_weights: np.ndarray) -> dict:
        weights = np.round(raw_weights, 6)
        weights /= np.sum(weights)
        ret = portfolio_return(weights, self.mean_returns)
        vol = portfolio_volatility(weights, self.cov_matrix)
        return {
            "weights": dict(zip(self.tickers, weights)),
            "return": ret,
            "volatility": vol,
            "sharpe": sharpe_ratio(ret, vol, self.rf),
        }

    def optimize_max_sharpe(
        self,
        current_weights: np.ndarray | None = None,
        transaction_cost: float = 0.0,
    ) -> dict:
        """
        Maximize Sharpe ratio, optionally subtracting L1 turnover transaction costs.
        """

        def neg_sharpe(w: np.ndarray) -> float:
            ret = portfolio_return(w, self.mean_returns)
            if current_weights is not None and transaction_cost > 0:
                turnover = float(np.sum(np.abs(w - current_weights)))
                ret -= transaction_cost * turnover
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

        return self._format_result(result.x)

    def optimize_min_variance(self) -> dict:
        """Minimize portfolio volatility."""
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

        return self._format_result(result.x)

    def optimize_min_cvar(self, confidence: float = 0.95) -> dict:
        """
        Minimize empirical Conditional Value at Risk (Expected Shortfall).
        """
        if not (0.0 < confidence < 1.0):
            raise ValueError(f"confidence ({confidence}) must be in (0, 1).")
        alpha_pct = (1.0 - confidence) * 100.0

        def empirical_cvar(w: np.ndarray) -> float:
            port_rets = self._returns_matrix @ w
            cutoff = np.percentile(port_rets, alpha_pct)
            tail = port_rets[port_rets <= cutoff]
            return float(-tail.mean()) if len(tail) else float(-cutoff)

        result = minimize(
            empirical_cvar,
            self._initial_weights(),
            method="SLSQP",
            bounds=self.bounds,
            constraints=self._base_constraints(),
            options={"ftol": 1e-9, "maxiter": 1000},
        )

        if not result.success:
            raise OptimizationError(f"Min CVaR optimization failed: {result.message}")

        out = self._format_result(result.x)
        w_vec = np.array([out["weights"][t] for t in self.tickers], dtype=float)
        out["cvar_daily"] = float(empirical_cvar(w_vec))
        return out

    def _max_feasible_return(self) -> float:
        if not self.allow_short and self.max_weight >= 1.0:
            return float(np.max(self.mean_returns))
        res = minimize(
            lambda w: -portfolio_return(w, self.mean_returns),
            self._initial_weights(),
            method="SLSQP",
            bounds=self.bounds,
            constraints=self._base_constraints(),
            options={"ftol": 1e-9, "maxiter": 500},
        )
        return (
            float(portfolio_return(res.x, self.mean_returns))
            if res.success
            else float(np.max(self.mean_returns))
        )

    def generate_frontier(self, n_points: int = 200) -> list[dict]:
        """
        Generate efficient frontier points using warm-started SLSQP.
        """
        if self._frontier_cache is not None and len(self._frontier_cache) == n_points:
            return self._frontier_cache

        min_var = self.optimize_min_variance()
        min_ret = min_var["return"]
        max_ret = self._max_feasible_return()

        if max_ret <= min_ret:
            self._frontier_cache = [min_var]
            return self._frontier_cache

        # Additive margin keeps target ordering valid even when min_ret <= 0.
        eps = max(1e-6, (max_ret - min_ret) * 1e-3)
        target_returns = np.linspace(min_ret + eps, max_ret - eps, n_points)
        frontier = []
        x0 = np.array([min_var["weights"][t] for t in self.tickers], dtype=float)

        for target in target_returns:
            constraints = self._base_constraints() + [
                {
                    "type": "eq",
                    "fun": lambda w, t=target: portfolio_return(w, self.mean_returns) - t,
                }
            ]
            res = minimize(
                portfolio_volatility,
                x0,
                args=(self.cov_matrix,),
                method="SLSQP",
                bounds=self.bounds,
                constraints=constraints,
                options={"ftol": 1e-9, "maxiter": 1000},
            )
            if res.success:
                w = res.x
                x0 = w
                ret = portfolio_return(w, self.mean_returns)
                vol = portfolio_volatility(w, self.cov_matrix)
                frontier.append(
                    {
                        "return": ret,
                        "volatility": vol,
                        "sharpe": sharpe_ratio(ret, vol, self.rf),
                        "weights": dict(zip(self.tickers, w)),
                    }
                )
            else:
                logger.debug(f"Frontier point at target {target:.4f} failed to converge.")

        logger.info(f"Frontier: {len(frontier)}/{n_points} points converged")
        self._frontier_cache = frontier
        return frontier

    def get_random_portfolios(self, n: int = N_PORTFOLIOS_MONTE_CARLO) -> pd.DataFrame:
        """
        Generate random long-only portfolios using Dirichlet sampling.
        """
        rng = np.random.default_rng(42)
        weights_matrix = rng.dirichlet(np.ones(self.n_assets), n)

        rets = weights_matrix @ self.mean_returns
        vols = np.sqrt(np.einsum("ij,jk,ik->i", weights_matrix, self.cov_matrix, weights_matrix))
        sharpes = (rets - self.rf) / vols

        data = np.hstack([weights_matrix, rets[:, None], vols[:, None], sharpes[:, None]])
        columns = self.tickers + ["return", "volatility", "sharpe"]

        return pd.DataFrame(data, columns=columns)
