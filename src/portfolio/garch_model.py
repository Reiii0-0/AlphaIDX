"""
GARCH(1,1) Volatility Modeling.
"""

import logging
import pandas as pd
import numpy as np
from arch import arch_model

logger = logging.getLogger(__name__)

__all__ = ["GARCHVolatilityModel"]

class GARCHVolatilityModel:
    """
    Fits a GARCH(1,1) model to portfolio returns to forecast volatility.
    """
    def __init__(self, log_returns: pd.Series):
        """
        Initialize the GARCH model.

        Args:
            log_returns (pd.Series): Historical daily log returns of the portfolio or asset.
        """
        self.returns = log_returns * 100  # arch_model prefers scaled up data (percentages)
        # Using Zero mean, since daily returns are close to zero, and GARCH(1,1) for variance
        self.model = arch_model(self.returns, vol='Garch', p=1, q=1, mean='Zero', dist='Normal')
        self.fit_result = None

    def fit(self) -> None:
        """Fit the GARCH(1,1) model to the data."""
        logger.info("Fitting GARCH(1,1) model to historical returns...")
        # disp='off' suppresses standard output printing from the optimizer
        self.fit_result = self.model.fit(disp='off')
        logger.info(f"GARCH(1,1) fitting complete. AIC: {self.fit_result.aic:.2f}")

    def forecast_volatility(self, horizon: int = 1) -> float:
        """
        Forecast the annualized volatility for a given horizon.

        Args:
            horizon (int): Forecasting horizon in days.

        Returns:
            float: Annualized forecasted volatility (as a decimal, e.g., 0.15 for 15%).
        """
        if self.fit_result is None:
            raise RuntimeError("Model must be fitted before forecasting. Call .fit() first.")

        forecasts = self.fit_result.forecast(horizon=horizon, reindex=False)
        # Variance forecast is for the scaled data (variance in %^2)
        var_forecast = forecasts.variance.values[-1, :]
        
        # Average daily variance over the horizon, scaled back to decimal
        avg_daily_var = np.mean(var_forecast) / 10000.0
        
        # Annualize
        ann_vol = np.sqrt(avg_daily_var * 252)
        return float(ann_vol)
