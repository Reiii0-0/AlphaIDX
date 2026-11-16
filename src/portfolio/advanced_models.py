"""
Advanced Portfolio Allocation Models: Risk Parity and Black-Litterman.
"""

import logging

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

from src.portfolio.metrics import OptimizationError, portfolio_volatility
from src.portfolio.optimizer import ledoit_wolf_shrinkage

logger = logging.getLogger(__name__)

__all__ = ["RiskParityOptimizer", "BlackLittermanModel", "HierarchicalRiskParityOptimizer"]


class RiskParityOptimizer:
    """
    Optimizes portfolio weights such that each asset contributes equally to the total portfolio risk.
    """

    def __init__(self, cov_matrix: np.ndarray, tickers: list[str]):
        self.cov_matrix = cov_matrix
        self.tickers = tickers
        self.n_assets = len(tickers)
        self.bounds = tuple((0.0, 1.0) for _ in range(self.n_assets))

    def _risk_contributions(self, weights: np.ndarray) -> np.ndarray:
        """Calculate the marginal risk contribution of each asset."""
        vol = portfolio_volatility(weights, self.cov_matrix)
        if vol == 0.0:
            return np.zeros_like(weights)
        mrc = (self.cov_matrix @ weights) / vol
        return weights * mrc

    def _risk_parity_objective(self, weights: np.ndarray) -> float:
        """Objective function: minimize the squared differences between risk contributions."""
        rc = self._risk_contributions(weights)
        target_rc = rc.sum() / self.n_assets
        return float(np.sum((rc - target_rc) ** 2))

    def optimize(self) -> dict:
        """
        Run the Risk Parity optimization.

        Returns:
            dict: Optimized weights and risk metrics.
        """
        initial_weights = np.full(self.n_assets, 1.0 / self.n_assets)
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        res = minimize(
            self._risk_parity_objective,
            initial_weights,
            method="SLSQP",
            bounds=self.bounds,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 1000},
        )

        if not res.success:
            raise OptimizationError(f"Risk Parity optimization failed: {res.message}")

        weights = np.round(res.x, 6)
        weights /= np.sum(weights)

        vol = portfolio_volatility(weights, self.cov_matrix)
        rc = self._risk_contributions(weights)
        rc_sum = float(np.sum(rc))
        rc_pct = rc / rc_sum if rc_sum > 0 else np.full(self.n_assets, 1.0 / self.n_assets)

        return {
            "weights": dict(zip(self.tickers, weights)),
            "volatility": vol,
            "risk_contributions": dict(zip(self.tickers, rc_pct)),
        }


class HierarchicalRiskParityOptimizer:
    """
    Hierarchical Risk Parity (HRP) allocator via correlation tree clustering and recursive bisection.
    """

    def __init__(
        self,
        log_returns: pd.DataFrame,
        cov_matrix: np.ndarray | None = None,
    ) -> None:
        self.log_returns = log_returns
        self.tickers = list(log_returns.columns)
        self.n_assets = len(self.tickers)
        self.cov_matrix = (
            np.asarray(cov_matrix, dtype=float)
            if cov_matrix is not None
            else ledoit_wolf_shrinkage(log_returns)
        )

    @staticmethod
    def _cluster_var(cov: np.ndarray, indices: list[int]) -> float:
        sub_cov = cov[np.ix_(indices, indices)]
        diag = np.clip(np.diag(sub_cov), 1e-12, None)
        inv_diag = 1.0 / diag
        w = inv_diag / np.sum(inv_diag)
        return float(w @ sub_cov @ w)

    def optimize(self) -> dict:
        """Compute HRP weights without inverting the covariance matrix."""
        corr = np.nan_to_num(self.log_returns.corr().to_numpy(dtype=float), nan=0.0)
        np.fill_diagonal(corr, 1.0)
        dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0))
        np.fill_diagonal(dist, 0.0)

        link = linkage(squareform(dist, checks=False), method="single")
        sort_ix = [int(i) for i in leaves_list(link)]

        weights = pd.Series(1.0, index=sort_ix, dtype=float)
        clusters = [sort_ix]
        while clusters:
            clusters = [
                c[j:k]
                for c in clusters
                for j, k in ((0, len(c) // 2), (len(c) // 2, len(c)))
                if len(c) > 1
            ]
            for i in range(0, len(clusters), 2):
                c0, c1 = clusters[i], clusters[i + 1]
                var0 = self._cluster_var(self.cov_matrix, c0)
                var1 = self._cluster_var(self.cov_matrix, c1)
                denom = var0 + var1
                alpha = 1.0 - var0 / denom if denom > 0 else 0.5
                weights[c0] *= alpha
                weights[c1] *= 1.0 - alpha

        w_arr = np.round(weights.sort_index().to_numpy(dtype=float), 6)
        w_arr /= np.sum(w_arr)
        vol = portfolio_volatility(w_arr, self.cov_matrix)
        return {
            "weights": dict(zip(self.tickers, w_arr)),
            "volatility": vol,
            "cluster_order": [self.tickers[i] for i in sort_ix],
        }


class BlackLittermanModel:
    """
    Black-Litterman model for deriving posterior expected returns combining market equilibrium and subjective views.
    """

    def __init__(
        self,
        cov_matrix: np.ndarray,
        market_weights: np.ndarray,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
    ):
        self.cov = cov_matrix
        self.mkt_weights = market_weights
        self.delta = risk_aversion
        self.tau = tau
        self.n_assets = len(market_weights)

    def implied_equilibrium_returns(self) -> np.ndarray:
        """Pi = delta * Sigma * w_mkt"""
        return self.delta * (self.cov @ self.mkt_weights)

    def posterior_returns(
        self, P: np.ndarray, Q: np.ndarray, Omega: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Calculate posterior expected returns given views using the Woodbury identity.

        Args:
            P (np.ndarray): Pick matrix (K x N).
            Q (np.ndarray): View vector (K,).
            Omega (np.ndarray, optional): Uncertainty matrix of views (K x K).

        Returns:
            np.ndarray: Posterior expected returns (N,).
        """
        pi = self.implied_equilibrium_returns()
        tau_cov = self.tau * self.cov

        if Omega is None:
            Omega = np.diag(np.diag(P @ tau_cov @ P.T))

        # Single KxK solve avoids inverting N x N matrices and handles singular cov safely.
        view_cov = P @ tau_cov @ P.T + Omega
        innovation = Q - P @ pi
        return pi + tau_cov @ P.T @ np.linalg.solve(view_cov, innovation)
