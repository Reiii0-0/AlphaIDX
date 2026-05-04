"""
Portfolio metrics and constants.

All constants follow industry standards:
- TRADING_DAYS_PER_YEAR = 252 (standard finance convention, not actual IDX calendar)
- RISK_FREE_RATE_ANNUAL = 0.0625 (BI Rate ~6.25%, as of 2024)
- Annualization: multiply daily μ by 252, daily σ² by 252
"""

import logging
import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)

__all__ = [
    "TRADING_DAYS_PER_YEAR",
    "RISK_FREE_RATE_ANNUAL",
    "RISK_FREE_RATE_DAILY",
    "DEFAULT_TICKERS",
    "DEFAULT_START_DATE",
    "DEFAULT_END_DATE",
    "N_PORTFOLIOS_MONTE_CARLO",
    "N_SIMULATION_PATHS",
    "SIMULATION_DAYS",
    "portfolio_return",
    "portfolio_volatility",
    "sharpe_ratio",
    "portfolio_var",
    "portfolio_cvar",
    "max_drawdown",
    "OptimizationError",
]

# ── Time ──
TRADING_DAYS_PER_YEAR = 252

# ── Rates ──
RISK_FREE_RATE_ANNUAL = 0.0625  # BI Rate ~6.25% as of 2024 (Indonesia)
RISK_FREE_RATE_DAILY = RISK_FREE_RATE_ANNUAL / TRADING_DAYS_PER_YEAR

# ── Defaults ──
DEFAULT_TICKERS = [
    "BBCA.JK",
    "TLKM.JK",
    "BBRI.JK",
    "BMRI.JK",
    "ASII.JK",
    "UNVR.JK",
    "ICBP.JK",
    "EXCL.JK",
    "KLBF.JK",
    "INDF.JK",
]
DEFAULT_START_DATE = "2019-01-01"
DEFAULT_END_DATE = "2024-12-31"  # 5 years, ~1260 trading days

# ── Simulation ──
N_PORTFOLIOS_MONTE_CARLO = 10_000
N_SIMULATION_PATHS = 1_000  # GBM price path simulations
SIMULATION_DAYS = 252  # 1 year forward projection


def portfolio_return(weights: np.ndarray, mean_returns: np.ndarray) -> float:
    """
    Annualized expected return.

    Args:
        weights (np.ndarray): Portfolio weights, shape (n,).
        mean_returns (np.ndarray): Annualized mean returns, shape (n,).

    Returns:
        float: Expected portfolio return.

    Raises:
        ValueError: If weights and mean_returns shapes are incompatible.
    """
    if weights.shape != mean_returns.shape:
        raise ValueError(
            f"Weights {weights.shape} and mean_returns {mean_returns.shape} shapes incompatible."
        )
    # E[R_p] = w^T · μ
    return float(np.dot(weights, mean_returns))


def portfolio_volatility(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """
    Annualized portfolio volatility.

    Args:
        weights (np.ndarray): Portfolio weights, shape (n,).
        cov_matrix (np.ndarray): Annualized covariance matrix, shape (n, n).

    Returns:
        float: Annualized portfolio volatility.

    Raises:
        ValueError: If weights and cov_matrix shapes are incompatible.
    """
    if weights.shape[0] != cov_matrix.shape[0] or cov_matrix.shape[0] != cov_matrix.shape[1]:
        raise ValueError(
            f"Weights {weights.shape} and cov_matrix {cov_matrix.shape} shapes incompatible."
        )
    # σ_p = sqrt(w^T · Σ · w)
    return float(np.sqrt(weights.T @ cov_matrix @ weights))


def sharpe_ratio(ret: float, vol: float, rf: float = RISK_FREE_RATE_ANNUAL) -> float:
    """
    Compute the Sharpe ratio.

    Args:
        ret (float): Annualized portfolio return.
        vol (float): Annualized portfolio volatility.
        rf (float): Annualized risk-free rate. Defaults to RISK_FREE_RATE_ANNUAL.

    Returns:
        float: Sharpe ratio. Returns 0.0 if vol is 0.
    """
    if vol == 0:
        return 0.0
    # S = (E[R_p] - R_f) / σ_p
    return (ret - rf) / vol


def portfolio_var(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    confidence: float = 0.95,
    horizon_days: int = 1,
) -> float:
    """
    Parametric Value at Risk (VaR) under normal distribution assumption.

    Args:
        weights (np.ndarray): Portfolio weights, shape (n,).
        mean_returns (np.ndarray): Annualized mean returns, shape (n,).
        cov_matrix (np.ndarray): Annualized covariance matrix, shape (n, n).
        confidence (float): Confidence level (e.g., 0.95).
        horizon_days (int): Time horizon in trading days.

    Returns:
        float: Positive number representing the loss at the given confidence level.
    """
    # h = horizon in days as fraction of year
    h = horizon_days / TRADING_DAYS_PER_YEAR
    mu_p = portfolio_return(weights, mean_returns)
    sig_p = portfolio_volatility(weights, cov_matrix)
    # z = α-quantile of standard normal
    z = norm.ppf(1 - confidence)
    # VaR_α = -(μ_p · h + z_α · σ_p · √h)
    var = -(mu_p * h + z * sig_p * np.sqrt(h))
    return float(var)


def portfolio_cvar(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    confidence: float = 0.95,
) -> float:
    """
    Parametric Conditional VaR (Expected Shortfall) under normal assumption.

    Args:
        weights (np.ndarray): Portfolio weights, shape (n,).
        mean_returns (np.ndarray): Annualized mean returns, shape (n,).
        cov_matrix (np.ndarray): Annualized covariance matrix, shape (n, n).
        confidence (float): Confidence level (e.g., 0.95).

    Returns:
        float: Positive number representing the expected loss beyond VaR.
    """
    h = 1 / TRADING_DAYS_PER_YEAR
    mu_p = portfolio_return(weights, mean_returns)
    sig_p = portfolio_volatility(weights, cov_matrix)
    z = norm.ppf(1 - confidence)
    # CVaR_α = -μ_p · h + σ_p · √h · φ(z_α) / (1 - α)
    cvar = -mu_p * h + sig_p * np.sqrt(h) * norm.pdf(z) / (1 - confidence)
    return float(cvar)


def max_drawdown(cumulative_returns: np.ndarray) -> float:
    """
    Compute the Maximum Drawdown.

    Args:
        cumulative_returns (np.ndarray): Cumulative return series (1 + r_cum).

    Returns:
        float: Most negative value representing max drawdown.
    """
    # MDD = max over t of [ max over s≤t of P(s) - P(t) ] / max over s≤t of P(s)
    roll_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = (cumulative_returns - roll_max) / roll_max
    return float(drawdowns.min())


class OptimizationError(ValueError):
    """Raised when scipy.optimize.minimize fails (result.success == False)."""
    pass
