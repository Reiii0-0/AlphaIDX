import numpy as np
import pytest
from src.portfolio.metrics import (
    portfolio_return,
    portfolio_volatility,
    sharpe_ratio,
    portfolio_var,
    portfolio_cvar,
    max_drawdown,
    RISK_FREE_RATE_ANNUAL,
)

def test_portfolio_return_equal_weights():
    weights = np.array([0.5, 0.5])
    mean_returns = np.array([0.1, 0.2])
    # expected = 0.5*0.1 + 0.5*0.2 = 0.15
    assert pytest.approx(portfolio_return(weights, mean_returns)) == 0.15

def test_portfolio_return_single_asset():
    weights = np.array([1.0, 0.0])
    mean_returns = np.array([0.15, 0.25])
    assert pytest.approx(portfolio_return(weights, mean_returns)) == 0.15

def test_portfolio_volatility_uncorrelated():
    weights = np.array([0.5, 0.5])
    # Identity cov matrix (vol=0.1, 0.1, corr=0) -> var = 0.01, 0.01
    cov_matrix = np.array([[0.01, 0.0], [0.0, 0.01]])
    # expected_vol = sqrt(0.5^2 * 0.01 + 0.5^2 * 0.01) = sqrt(0.25*0.01 + 0.25*0.01) = sqrt(0.005)
    expected_vol = np.sqrt(0.005)
    assert pytest.approx(portfolio_volatility(weights, cov_matrix)) == expected_vol

def test_portfolio_volatility_perfect_corr():
    weights = np.array([0.5, 0.5])
    # perfect correlation, vol=0.1 each
    # cov = 0.1 * 0.1 * 1 = 0.01
    cov_matrix = np.array([[0.01, 0.01], [0.01, 0.01]])
    # expected_vol = 0.5*0.1 + 0.5*0.1 = 0.1
    assert pytest.approx(portfolio_volatility(weights, cov_matrix)) == 0.1

def test_sharpe_ratio_zero_vol():
    assert sharpe_ratio(0.1, 0.0) == 0.0

def test_sharpe_ratio_known_values():
    ret = 0.15
    vol = 0.10
    rf = 0.05
    # (0.15 - 0.05) / 0.10 = 1.0
    assert pytest.approx(sharpe_ratio(ret, vol, rf)) == 1.0

def test_portfolio_var_positive():
    weights = np.array([1.0])
    mean_returns = np.array([0.1])
    cov_matrix = np.array([[0.04]]) # vol = 0.2
    var = portfolio_var(weights, mean_returns, cov_matrix)
    assert var > 0

def test_portfolio_var_99_gt_95():
    weights = np.array([1.0])
    mean_returns = np.array([0.1])
    cov_matrix = np.array([[0.04]])
    var_95 = portfolio_var(weights, mean_returns, cov_matrix, confidence=0.95)
    var_99 = portfolio_var(weights, mean_returns, cov_matrix, confidence=0.99)
    assert var_99 > var_95

def test_portfolio_cvar_gt_var():
    weights = np.array([1.0])
    mean_returns = np.array([0.1])
    cov_matrix = np.array([[0.04]])
    var_95 = portfolio_var(weights, mean_returns, cov_matrix, confidence=0.95)
    cvar_95 = portfolio_cvar(weights, mean_returns, cov_matrix, confidence=0.95)
    assert cvar_95 >= var_95

def test_max_drawdown_flat():
    cum_rets = np.array([1.0, 1.0, 1.0])
    assert max_drawdown(cum_rets) == 0.0

def test_max_drawdown_known_sequence():
    # [100, 90, 80, 100] -> [1.0, 0.9, 0.8, 1.0]
    # MDD = (0.8 - 1.0) / 1.0 = -0.2
    cum_rets = np.array([1.0, 0.9, 0.8, 1.0])
    assert pytest.approx(max_drawdown(cum_rets)) == -0.2

def test_max_drawdown_recovery():
    # [1.0, 0.8, 1.1, 0.9] 
    # Max peak 1 at t=0, min 0.8 at t=1 -> DD = -0.2
    # New peak 1.1 at t=2, min 0.9 at t=3 -> DD = (0.9-1.1)/1.1 = -0.1818
    # MDD = -0.2
    cum_rets = np.array([1.0, 0.8, 1.1, 0.9])
    assert pytest.approx(max_drawdown(cum_rets)) == -0.2
