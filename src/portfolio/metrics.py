"""
Portfolio metrics and constants.

All constants follow industry standards:
- TRADING_DAYS_PER_YEAR = 252 (standard finance convention, not actual IDX calendar)
- RISK_FREE_RATE_ANNUAL = 0.0625 (BI Rate ~6.25%, as of 2024)
- Annualization: multiply daily μ by 252, daily σ² by 252
"""

import logging
import os
from pathlib import Path
from typing import Literal

import numpy as np
from dotenv import load_dotenv
from scipy.stats import norm
from scipy.stats import t as student_t

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

__all__ = [
    "PROJECT_ROOT",
    "CACHE_DIR",
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

CACHE_DIR = PROJECT_ROOT / "data" / "cache"

# ── Time ──
TRADING_DAYS_PER_YEAR = 252

# ── Rates ──
RISK_FREE_RATE_ANNUAL = float(os.getenv("RISK_FREE_RATE", "0.0625"))
RISK_FREE_RATE_DAILY = RISK_FREE_RATE_ANNUAL / TRADING_DAYS_PER_YEAR

# ── Defaults ──
_DEFAULT_TICKERS_FALLBACK = [
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
_env_tickers = os.getenv("DEFAULT_TICKERS", "").strip()
DEFAULT_TICKERS = (
    [t.strip() for t in _env_tickers.split(",") if t.strip()]
    if _env_tickers
    else _DEFAULT_TICKERS_FALLBACK
)
DEFAULT_START_DATE = os.getenv("START_DATE", "2019-01-01")
DEFAULT_END_DATE = os.getenv("END_DATE", "2024-12-31")

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
    return float(np.sqrt(weights.T @ cov_matrix @ weights))


def sharpe_ratio(ret: float, vol: float, rf: float = RISK_FREE_RATE_ANNUAL) -> float:
    """
    Compute the Sharpe ratio.
    """
    if vol == 0:
        return 0.0
    return (ret - rf) / vol


def portfolio_var(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    confidence: float = 0.95,
    horizon_days: int = 1,
    dist: Literal["normal", "t"] = "normal",
    df: float = 4.0,
) -> float:
    """
    Parametric Value at Risk (VaR) under normal or fat-tailed Student-t assumption.
    """
    h = horizon_days / TRADING_DAYS_PER_YEAR
    mu_p = portfolio_return(weights, mean_returns)
    sig_p = portfolio_volatility(weights, cov_matrix)
    if dist == "t":
        if df <= 2.0:
            raise ValueError("Student-t degrees of freedom (df) must be > 2.0.")
        scale = sig_p * np.sqrt(h * (df - 2.0) / df)
        q = student_t.ppf(1.0 - confidence, df)
        return float(-(mu_p * h + q * scale))
    if dist != "normal":
        raise ValueError(f"Unsupported distribution: {dist}")
    z = norm.ppf(1.0 - confidence)
    return float(-(mu_p * h + z * sig_p * np.sqrt(h)))


def portfolio_cvar(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    confidence: float = 0.95,
    dist: Literal["normal", "t"] = "normal",
    df: float = 4.0,
) -> float:
    """
    Parametric Conditional VaR (Expected Shortfall) under normal or Student-t assumption.
    """
    h = 1.0 / TRADING_DAYS_PER_YEAR
    mu_p = portfolio_return(weights, mean_returns)
    sig_p = portfolio_volatility(weights, cov_matrix)
    alpha = 1.0 - confidence
    if dist == "t":
        if df <= 2.0:
            raise ValueError("Student-t degrees of freedom (df) must be > 2.0.")
        scale = sig_p * np.sqrt(h * (df - 2.0) / df)
        q = student_t.ppf(alpha, df)
        tail_factor = (student_t.pdf(q, df) / alpha) * ((df + q**2) / (df - 1.0))
        return float(-mu_p * h + scale * tail_factor)
    if dist != "normal":
        raise ValueError(f"Unsupported distribution: {dist}")
    z = norm.ppf(alpha)
    return float(-mu_p * h + sig_p * np.sqrt(h) * norm.pdf(z) / alpha)


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
