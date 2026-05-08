import numpy as np
import pandas as pd
import pytest
from src.portfolio.optimizer import EfficientFrontier
from src.portfolio.metrics import OptimizationError

@pytest.fixture
def sample_returns():
    # 3 assets, 1000 days of log returns
    # Asset 1: high return, high vol
    # Asset 2: low return, low vol
    # Asset 3: medium return, medium vol
    rng = np.random.default_rng(42)
    n_days = 1000
    means = [0.15/252, 0.05/252, 0.10/252]
    # Simple uncorrelated cov
    vols = [0.30/np.sqrt(252), 0.10/np.sqrt(252), 0.20/np.sqrt(252)]
    
    returns = rng.normal(means, vols, size=(n_days, 3))
    df = pd.DataFrame(returns, columns=["A", "B", "C"])
    df.index = pd.date_range("2020-01-01", periods=n_days)
    return df

def test_optimize_max_sharpe_runs(sample_returns):
    ef = EfficientFrontier(sample_returns)
    result = ef.optimize_max_sharpe()
    assert "weights" in result
    assert "return" in result
    assert "volatility" in result
    assert "sharpe" in result
    assert isinstance(result["weights"], dict)
    assert len(result["weights"]) == 3

def test_optimize_max_sharpe_weights_sum(sample_returns):
    ef = EfficientFrontier(sample_returns)
    result = ef.optimize_max_sharpe()
    assert pytest.approx(sum(result["weights"].values())) == 1.0

def test_optimize_max_sharpe_weights_nonneg(sample_returns):
    ef = EfficientFrontier(sample_returns)
    result = ef.optimize_max_sharpe()
    assert all(w >= -1e-6 for w in result["weights"].values())

def test_optimize_min_variance_lower_vol(sample_returns):
    ef = EfficientFrontier(sample_returns)
    ms = ef.optimize_max_sharpe()
    mv = ef.optimize_min_variance()
    assert mv["volatility"] <= ms["volatility"] + 1e-7

def test_optimize_max_sharpe_gt_min_var_sharpe(sample_returns):
    ef = EfficientFrontier(sample_returns)
    ms = ef.optimize_max_sharpe()
    mv = ef.optimize_min_variance()
    assert ms["sharpe"] >= mv["sharpe"] - 1e-7

def test_generate_frontier_sorted(sample_returns):
    ef = EfficientFrontier(sample_returns)
    frontier = ef.generate_frontier(n_points=20)
    vols = [p["volatility"] for p in frontier]
    # Volatility should generally increase with return on the frontier
    # (though numerical precision and point selection might show slight noise)
    assert all(vols[i] <= vols[i+1] + 1e-5 for i in range(len(vols)-1))

def test_generate_frontier_length(sample_returns):
    ef = EfficientFrontier(sample_returns)
    n = 20
    frontier = ef.generate_frontier(n_points=n)
    assert len(frontier) >= n * 0.8

def test_get_random_portfolios_shape(sample_returns):
    ef = EfficientFrontier(sample_returns)
    n = 100
    df = ef.get_random_portfolios(n=n)
    assert df.shape == (n, 3 + 3) # 3 tickers + ret, vol, sharpe

def test_get_random_portfolios_weights_sum(sample_returns):
    ef = EfficientFrontier(sample_returns)
    n = 100
    df = ef.get_random_portfolios(n=n)
    # Check weight columns sum to 1
    weights = df[["A", "B", "C"]]
    sums = weights.sum(axis=1)
    assert np.allclose(sums, 1.0)

def test_optimization_error_raised():
    # Provide input that makes optimization fail (e.g., highly inconsistent returns/cov)
    df = pd.DataFrame(np.random.randn(100, 2), columns=["A", "B"])
    df.index = pd.date_range("2020-01-01", periods=100)
    ef = EfficientFrontier(df)
    # Manually corrupting the cov_matrix to something non-positive definite or extreme
    ef.cov_matrix = np.array([[1.0, 2.0], [2.0, 1.0]]) # Not positive definite
    # Also set extreme mean returns to force the solver into difficult regions
    ef.mean_returns = np.array([1e10, -1e10])
    
    with pytest.raises(OptimizationError):
        ef.optimize_max_sharpe()

def test_efficient_frontier_dominance(sample_returns):
    # Every frontier portfolio should have max return for its volatility level
    # (or min volatility for its return level)
    ef = EfficientFrontier(sample_returns)
    frontier = ef.generate_frontier(n_points=20)
    random_portfolios = ef.get_random_portfolios(n=1000)
    
    for f_p in frontier:
        # For random portfolios with similar or lower return, 
        # frontier portfolio should have lower or equal volatility (within tolerance)
        better_random = random_portfolios[
            (random_portfolios["return"] >= f_p["return"] - 1e-5) & 
            (random_portfolios["volatility"] < f_p["volatility"] - 1e-3)
        ]
        assert better_random.empty, f"Frontier point {f_p} is dominated by random portfolios."
