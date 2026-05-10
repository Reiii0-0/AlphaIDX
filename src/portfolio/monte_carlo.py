"""
Monte Carlo simulation using Geometric Brownian Motion (GBM).
"""

import logging

import numpy as np
import pandas as pd

from src.portfolio.metrics import (
    N_SIMULATION_PATHS,
    SIMULATION_DAYS,
    TRADING_DAYS_PER_YEAR,
    portfolio_return,
    portfolio_volatility,
)

logger = logging.getLogger(__name__)

__all__ = ["MonteCarloSimulator"]


class MonteCarloSimulator:
    """
    Monte Carlo simulator for portfolio performance projection.
    """

    def __init__(
        self,
        log_returns: pd.DataFrame,
        optimal_weights: np.ndarray,
        random_seed: int = 42,
    ) -> None:
        """
        Initialize the simulator.

        Args:
            log_returns (pd.DataFrame): Historical daily log returns.
            optimal_weights (np.ndarray): Portfolio weights (sum to 1.0).
            random_seed (int): Seed for reproducibility.
        """
        self.weights = optimal_weights / np.sum(optimal_weights)
        self.tickers = list(log_returns.columns)

        # Daily metrics for GBM
        self.mu_daily = log_returns.mean().values
        self.cov_daily = log_returns.cov().values

        # Portfolio daily metrics
        self.mu_p_daily = float(np.dot(self.weights, self.mu_daily))
        self.sig_p_daily = float(np.sqrt(self.weights @ self.cov_daily @ self.weights))

        self.rng = np.random.default_rng(random_seed)

    def simulate_price_paths(
        self,
        n_simulations: int = N_SIMULATION_PATHS,
        n_days: int = SIMULATION_DAYS,
        initial_value: float = 100.0,
    ) -> np.ndarray:
        """
        Simulate price paths using GBM for the optimal portfolio.

        Args:
            n_simulations (int): Number of paths to simulate.
            n_days (int): Number of trading days to project.
            initial_value (float): Starting portfolio value.

        Returns:
            np.ndarray: Simulated paths, shape (n_simulations, n_days + 1).
        """
        # drift = mu_p_daily - 0.5 * sig_p_daily² (Itô correction, MANDATORY)
        drift = self.mu_p_daily - 0.5 * (self.sig_p_daily**2)

        # Z ~ N(0, 1)
        Z = self.rng.standard_normal((n_simulations, n_days))

        # log_ret = drift + sig_p_daily * Z
        log_ret = drift + self.sig_p_daily * Z

        # paths = S(0) * exp(cumsum(log_ret))
        paths = initial_value * np.exp(np.cumsum(log_ret, axis=1))

        # Prepend initial value
        paths = np.hstack([np.full((n_simulations, 1), initial_value), paths])

        assert paths.shape == (n_simulations, n_days + 1)
        assert np.allclose(paths[:, 0], initial_value)

        logger.info(
            f"Simulated {n_simulations} paths for {n_days} days. "
            f"Ann. Return: {self.mu_p_daily * TRADING_DAYS_PER_YEAR:.2%}, "
            f"Ann. Vol: {self.sig_p_daily * np.sqrt(TRADING_DAYS_PER_YEAR):.2%}, "
            f"Drift: {drift:.6f}"
        )

        return paths

    def compute_var_cvar(self, paths: np.ndarray, confidence: float = 0.95) -> dict:
        """
        Compute VaR and CVaR from simulated final values.

        Args:
            paths (np.ndarray): Simulated paths.
            confidence (float): Confidence level.

        Returns:
            dict: Risk metrics.
        """
        initial_value = paths[0, 0]
        final_values = paths[:, -1]
        returns_pct = (final_values - initial_value) / initial_value

        # var_pct: loss at (1-confidence) percentile
        # Using negative percentile because VaR is typically expressed as a positive loss
        var_pct = float(-np.percentile(returns_pct, (1 - confidence) * 100))
        
        # cvar_pct: mean of losses beyond VaR
        losses_beyond_var = returns_pct[returns_pct <= -var_pct]
        cvar_pct = float(-losses_beyond_var.mean()) if len(losses_beyond_var) > 0 else var_pct

        return {
            "var_pct": var_pct,
            "cvar_pct": cvar_pct,
            "var_abs": var_pct * initial_value,
            "cvar_abs": cvar_pct * initial_value,
            "median_final": float(np.median(final_values)),
            "mean_final": float(np.mean(final_values)),
            "percentile_5": float(np.percentile(final_values, 5)),
            "percentile_95": float(np.percentile(final_values, 95)),
        }

    def get_path_statistics(self, paths: np.ndarray) -> pd.DataFrame:
        """
        Compute per-step statistics for simulation paths.

        Args:
            paths (np.ndarray): Simulated paths.

        Returns:
            pd.DataFrame: Step-by-step stats.
        """
        # Calculate percentiles across simulations (axis 0)
        mean = np.mean(paths, axis=0)
        median = np.median(paths, axis=0)
        pct_5 = np.percentile(paths, 5, axis=0)
        pct_95 = np.percentile(paths, 95, axis=0)

        return pd.DataFrame(
            {"mean": mean, "median": median, "pct_5": pct_5, "pct_95": pct_95},
            index=range(paths.shape[1]),
        )
