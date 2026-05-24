"""
Advanced Portfolio Allocation Models: Risk Parity and Black-Litterman.
"""

import logging
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from src.portfolio.metrics import TRADING_DAYS_PER_YEAR, portfolio_volatility, OptimizationError

logger = logging.getLogger(__name__)

__all__ = ["RiskParityOptimizer", "BlackLittermanModel"]

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
        # Marginal Risk Contribution (MRC) = (Sigma * w) / vol
        mrc = (self.cov_matrix @ weights) / vol
        # Risk Contribution (RC) = w * MRC
        rc = weights * mrc
        return rc

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
        initial_weights = np.array([1.0 / self.n_assets] * self.n_assets)
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        
        res = minimize(
            self._risk_parity_objective,
            initial_weights,
            method="SLSQP",
            bounds=self.bounds,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 1000}
        )
        
        if not res.success:
            raise OptimizationError(f"Risk Parity optimization failed: {res.message}")
            
        weights = np.round(res.x, 6)
        weights /= np.sum(weights)
        
        vol = portfolio_volatility(weights, self.cov_matrix)
        rc = self._risk_contributions(weights)
        rc_pct = rc / np.sum(rc)
        
        return {
            "weights": dict(zip(self.tickers, weights)),
            "volatility": vol,
            "risk_contributions": dict(zip(self.tickers, rc_pct))
        }

class BlackLittermanModel:
    """
    Black-Litterman model for deriving posterior expected returns combining market equilibrium and subjective views.
    """
    def __init__(self, cov_matrix: np.ndarray, market_weights: np.ndarray, risk_aversion: float = 2.5, tau: float = 0.05):
        self.cov = cov_matrix
        self.mkt_weights = market_weights
        self.delta = risk_aversion
        self.tau = tau
        self.n_assets = len(market_weights)

    def implied_equilibrium_returns(self) -> np.ndarray:
        """Pi = delta * Sigma * w_mkt"""
        return self.delta * (self.cov @ self.mkt_weights)

    def posterior_returns(self, P: np.ndarray, Q: np.ndarray, Omega: np.ndarray = None) -> np.ndarray:
        """
        Calculate posterior expected returns given views.
        
        Args:
            P (np.ndarray): Pick matrix (K x N).
            Q (np.ndarray): View vector (K,).
            Omega (np.ndarray, optional): Uncertainty matrix of views (K x K). If None, proportional to variance.
            
        Returns:
            np.ndarray: Posterior expected returns (N,).
        """
        Pi = self.implied_equilibrium_returns()
        tau_cov = self.tau * self.cov
        
        if Omega is None:
            # Default Omega: Proportional to the variance of the view portfolios
            Omega = np.diag(np.diag(P @ tau_cov @ P.T))
            
        # BL Formula: mu = [(tau*Sigma)^-1 + P^T * Omega^-1 * P]^-1 * [(tau*Sigma)^-1 * Pi + P^T * Omega^-1 * Q]
        tau_cov_inv = np.linalg.inv(tau_cov)
        Omega_inv = np.linalg.inv(Omega)
        
        term1 = np.linalg.inv(tau_cov_inv + P.T @ Omega_inv @ P)
        term2 = tau_cov_inv @ Pi + P.T @ Omega_inv @ Q
        
        mu_bl = term1 @ term2
        return mu_bl
