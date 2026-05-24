import numpy as np
import pandas as pd
import pytest
from src.portfolio.garch_model import GARCHVolatilityModel

def test_garch_volatility_forecasting():
    # Generate synthetic returns with some volatility clustering
    rng = np.random.default_rng(42)
    n_samples = 1000
    
    # Simple simulated GARCH process
    omega = 0.05
    alpha = 0.1
    beta = 0.8
    
    returns = np.zeros(n_samples)
    sigma2 = np.zeros(n_samples)
    sigma2[0] = omega / (1 - alpha - beta)
    
    for t in range(1, n_samples):
        sigma2[t] = omega + alpha * (returns[t-1]**2) + beta * sigma2[t-1]
        returns[t] = rng.standard_normal() * np.sqrt(sigma2[t])
        
    # Scale down to daily decimal returns
    returns = returns / 100.0
    returns_series = pd.Series(returns)
    
    model = GARCHVolatilityModel(returns_series)
    
    with pytest.raises(RuntimeError):
        model.forecast_volatility()
        
    model.fit()
    ann_vol = model.forecast_volatility(horizon=10)
    
    assert ann_vol > 0
    assert 0.05 < ann_vol < 0.50 # Reasonable bound for this synthetic data
