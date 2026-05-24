import numpy as np
import pytest
from src.portfolio.advanced_models import RiskParityOptimizer, BlackLittermanModel

def test_risk_parity_equal_volatility():
    # If assets are uncorrelated and have the same volatility, 
    # risk parity should equal equal weights (1/n)
    cov = np.diag([0.04, 0.04, 0.04]) # Volatility = 0.2
    tickers = ["A", "B", "C"]
    
    rp = RiskParityOptimizer(cov, tickers)
    result = rp.optimize()
    
    weights = np.array([result["weights"][t] for t in tickers])
    rc = np.array([result["risk_contributions"][t] for t in tickers])
    
    # Weights should be 1/3
    assert np.allclose(weights, [1/3, 1/3, 1/3])
    # Risk contributions should be equal (1/3 each)
    assert np.allclose(rc, [1/3, 1/3, 1/3])

def test_risk_parity_different_volatility():
    # Asset A is highly volatile, B is low volatility
    # Risk parity should allocate more weight to B
    cov = np.diag([0.09, 0.01]) # Vol: A=0.3, B=0.1
    tickers = ["A", "B"]
    
    rp = RiskParityOptimizer(cov, tickers)
    result = rp.optimize()
    
    weights = np.array([result["weights"][t] for t in tickers])
    rc = np.array([result["risk_contributions"][t] for t in tickers])
    
    # Weight of B should be > Weight of A
    assert weights[1] > weights[0]
    # Risk contributions should be equal (0.5 each)
    assert np.allclose(rc, [0.5, 0.5])

def test_black_litterman_equilibrium():
    cov = np.diag([0.04, 0.04])
    mkt_weights = np.array([0.5, 0.5])
    delta = 2.5
    
    bl = BlackLittermanModel(cov, mkt_weights, delta)
    pi = bl.implied_equilibrium_returns()
    
    # Pi = delta * Cov * w_mkt = 2.5 * [[0.04, 0], [0, 0.04]] @ [0.5, 0.5]
    # = 2.5 * [0.02, 0.02] = [0.05, 0.05]
    assert np.allclose(pi, [0.05, 0.05])

def test_black_litterman_posterior_with_view():
    cov = np.diag([0.04, 0.04])
    mkt_weights = np.array([0.5, 0.5])
    bl = BlackLittermanModel(cov, mkt_weights, risk_aversion=2.5, tau=0.05)
    
    # Equilibrium return is [0.05, 0.05]
    # View: Asset 0 will have return of 10% (0.10)
    P = np.array([[1.0, 0.0]])
    Q = np.array([0.10])
    
    # Low uncertainty on the view
    Omega = np.array([[0.0001]])
    
    post_returns = bl.posterior_returns(P, Q, Omega)
    
    # Posterior for Asset 0 should shift strongly toward 0.10 from 0.05
    assert post_returns[0] > 0.08
    # Posterior for Asset 1 should remain close to equilibrium
    assert np.isclose(post_returns[1], 0.05, atol=1e-3)
