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
        mean_returns_annual: np.ndarray | None = None,
        cov_matrix_annual: np.ndarray | None = None,
    ) -> None:
        """
        Initialize the simulator.

        Args:
            log_returns (pd.DataFrame): Historical daily log returns.
            optimal_weights (np.ndarray): Portfolio weights (sum to 1.0).
            random_seed (int): Seed for reproducibility.
            mean_returns_annual (np.ndarray | None): Optional annualized mean returns override.
            cov_matrix_annual (np.ndarray | None): Optional annualized covariance matrix override.
        """
        weights_arr = np.asarray(optimal_weights, dtype=float)
        weight_sum = float(np.sum(weights_arr))
        if weight_sum == 0.0:
            raise ValueError("optimal_weights sum must not be zero.")
        self.weights = weights_arr / weight_sum
        self.tickers = list(log_returns.columns)
        n_assets = len(self.tickers)
        if self.weights.shape != (n_assets,):
            raise ValueError(f"Expected weights shape {(n_assets,)}, got {self.weights.shape}.")

        if mean_returns_annual is not None:
            mu_ann = np.asarray(mean_returns_annual, dtype=float)
            if mu_ann.shape != (n_assets,):
                raise ValueError(
                    f"Expected mean_returns_annual shape {(n_assets,)}, got {mu_ann.shape}."
                )
            self.mu_daily = mu_ann / TRADING_DAYS_PER_YEAR
        else:
            self.mu_daily = log_returns.mean().to_numpy(dtype=float)

        if cov_matrix_annual is not None:
            cov_ann = np.asarray(cov_matrix_annual, dtype=float)
            if cov_ann.shape != (n_assets, n_assets):
                raise ValueError(
                    f"Expected cov_matrix_annual shape {(n_assets, n_assets)}, got {cov_ann.shape}."
                )
            self.cov_daily = cov_ann / TRADING_DAYS_PER_YEAR
        else:
            self.cov_daily = log_returns.cov().to_numpy(dtype=float)

        self.mu_p_daily = float(np.dot(self.weights, self.mu_daily))
        self.sig_p_daily = float(
            np.sqrt(max(0.0, float(self.weights @ self.cov_daily @ self.weights)))
        )

        self.rng = np.random.default_rng(random_seed)

    def simulate_price_paths(
        self,
        n_simulations: int = N_SIMULATION_PATHS,
        n_days: int = SIMULATION_DAYS,
        initial_value: float = 100.0,
        antithetic: bool = True,
    ) -> np.ndarray:
        """
        Simulate price paths using GBM with Ito correction and optional antithetic variates.

        Args:
            n_simulations (int): Number of paths to simulate.
            n_days (int): Number of trading days to project.
            initial_value (float): Starting portfolio value.
            antithetic (bool): Pair Z and -Z draws to reduce Monte Carlo variance.

        Returns:
            np.ndarray: Simulated paths, shape (n_simulations, n_days + 1).
        """
        if n_simulations < 1 or n_days < 1:
            raise ValueError("n_simulations and n_days must be >= 1.")

        drift = self.mu_p_daily - 0.5 * (self.sig_p_daily**2)

        if antithetic and n_simulations >= 2:
            half = n_simulations // 2
            z_half = self.rng.standard_normal((half, n_days))
            z = np.vstack([z_half, -z_half])
            if n_simulations % 2 == 1:
                z = np.vstack([z, self.rng.standard_normal((1, n_days))])
        else:
            z = self.rng.standard_normal((n_simulations, n_days))

        log_ret = drift + self.sig_p_daily * z
        paths = initial_value * np.exp(np.cumsum(log_ret, axis=1))
        paths = np.hstack([np.full((n_simulations, 1), initial_value), paths])

        logger.info(
            f"Simulated {n_simulations} paths for {n_days} days (antithetic={antithetic}). "
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
