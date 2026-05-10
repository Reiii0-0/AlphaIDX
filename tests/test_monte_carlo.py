import numpy as np
import pandas as pd
import pytest
from src.portfolio.monte_carlo import MonteCarloSimulator

@pytest.fixture
def sample_mc_input():
    # 2 assets, 500 days of log returns
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0005, 0.01, size=(500, 2))
    df = pd.DataFrame(returns, columns=["A", "B"])
    weights = np.array([0.6, 0.4])
    return df, weights

def test_simulate_paths_shape(sample_mc_input):
    df, weights = sample_mc_input
    sim = MonteCarloSimulator(df, weights)
    n_sims = 100
    n_days = 50
    paths = sim.simulate_price_paths(n_simulations=n_sims, n_days=n_days)
    assert paths.shape == (n_sims, n_days + 1)

def test_simulate_paths_initial_value(sample_mc_input):
    df, weights = sample_mc_input
    sim = MonteCarloSimulator(df, weights)
    initial_value = 150.0
    paths = sim.simulate_price_paths(initial_value=initial_value)
    assert np.allclose(paths[:, 0], initial_value)

def test_simulate_paths_positive(sample_mc_input):
    df, weights = sample_mc_input
    sim = MonteCarloSimulator(df, weights)
    paths = sim.simulate_price_paths()
    assert np.all(paths > 0)

def test_simulate_paths_reproducible(sample_mc_input):
    df, weights = sample_mc_input
    sim1 = MonteCarloSimulator(df, weights, random_seed=42)
    paths1 = sim1.simulate_price_paths(n_simulations=10)
    
    sim2 = MonteCarloSimulator(df, weights, random_seed=42)
    paths2 = sim2.simulate_price_paths(n_simulations=10)
    
    assert np.allclose(paths1, paths2)

def test_compute_var_cvar_keys(sample_mc_input):
    df, weights = sample_mc_input
    sim = MonteCarloSimulator(df, weights)
    paths = sim.simulate_price_paths(n_simulations=100)
    metrics = sim.compute_var_cvar(paths)
    expected_keys = {
        "var_pct", "cvar_pct", "var_abs", "cvar_abs",
        "median_final", "mean_final", "percentile_5", "percentile_95"
    }
    assert set(metrics.keys()) == expected_keys

def test_compute_var_cvar_cvar_gt_var(sample_mc_input):
    df, weights = sample_mc_input
    sim = MonteCarloSimulator(df, weights)
    paths = sim.simulate_price_paths(n_simulations=500)
    metrics = sim.compute_var_cvar(paths)
    # CVaR is the mean of losses beyond VaR, so it should be >= VaR
    assert metrics["cvar_pct"] >= metrics["var_pct"] - 1e-7

def test_compute_var_cvar_99_gt_95(sample_mc_input):
    df, weights = sample_mc_input
    sim = MonteCarloSimulator(df, weights)
    paths = sim.simulate_price_paths(n_simulations=1000)
    metrics_95 = sim.compute_var_cvar(paths, confidence=0.95)
    metrics_99 = sim.compute_var_cvar(paths, confidence=0.99)
    assert metrics_99["var_pct"] > metrics_95["var_pct"]

def test_get_path_statistics_shape(sample_mc_input):
    df, weights = sample_mc_input
    sim = MonteCarloSimulator(df, weights)
    n_days = 30
    paths = sim.simulate_price_paths(n_days=n_days)
    stats = sim.get_path_statistics(paths)
    assert stats.shape == (n_days + 1, 4)
    assert list(stats.columns) == ["mean", "median", "pct_5", "pct_95"]

def test_get_path_statistics_pct5_lt_median(sample_mc_input):
    df, weights = sample_mc_input
    sim = MonteCarloSimulator(df, weights)
    paths = sim.simulate_price_paths(n_simulations=500)
    stats = sim.get_path_statistics(paths)
    assert np.all(stats["pct_5"] <= stats["median"])
    assert np.all(stats["median"] <= stats["pct_95"])

def test_ito_drift_correction(sample_mc_input):
    # Proving Itô correction is necessary: 
    # With correction (drift = mu - 0.5*sigma^2), log(E[S(T)]) should be mu*T.
    # Actually, E[S(T)] = S0 * exp(mu * T) for GBM.
    # If we simulated without correction (drift = mu), E[S(T)] would be S0 * exp((mu + 0.5*sigma^2)*T)
    df, weights = sample_mc_input
    sim = MonteCarloSimulator(df, weights)
    initial_value = 100.0
    n_days = 252
    n_sims = 5000 # High N for statistical significance
    
    paths = sim.simulate_price_paths(n_simulations=n_sims, n_days=n_days, initial_value=initial_value)
    final_mean = np.mean(paths[:, -1])
    
    # Expected: 100 * exp(mu_p * n_days)
    mu_p = sim.mu_p_daily
    expected_final = initial_value * np.exp(mu_p * n_days)
    
    # Check within 5% tolerance
    assert abs(final_mean / expected_final - 1.0) < 0.05

def test_expected_value_with_ito(sample_mc_input):
    # Statistical test for mean path
    df, weights = sample_mc_input
    sim = MonteCarloSimulator(df, weights)
    n_days = 100
    n_sims = 1000
    paths = sim.simulate_price_paths(n_simulations=n_sims, n_days=n_days)
    final_values = paths[:, -1]
    
    # For a large number of simulations, mean should be close to expectation
    expected_mean = 100.0 * np.exp(sim.mu_p_daily * n_days)
    assert abs(np.mean(final_values) / expected_mean - 1.0) < 0.05
