"""
Live Portfolio Dashboard using Streamlit.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Path hack to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.fetcher import StockDataFetcher
from src.portfolio.optimizer import EfficientFrontier
from src.portfolio.monte_carlo import MonteCarloSimulator
from src.portfolio.advanced_models import BlackLittermanModel, RiskParityOptimizer
from src.portfolio.garch_model import GARCHVolatilityModel
from src.portfolio.metrics import DEFAULT_TICKERS, RISK_FREE_RATE_ANNUAL
from src.visualization.plots import (
    plot_efficient_frontier,
    plot_weights_bar,
    plot_price_paths,
    plot_correlation_heatmap,
    plot_risk_parity_contributions
)

st.set_page_config(page_title="AlphaIDX | Live Portfolio Optimizer", layout="wide")

st.title("🚀 AlphaIDX: Institutional Quant Engine")
st.markdown("""
This dashboard performs **real-time portfolio optimization** utilizing standard **Mean-Variance** and **Advanced Institutional Frameworks** (Risk Parity, Black-Litterman, GARCH, Short Selling, Transaction Costs).
""")

# ── Sidebar Configuration ──
st.sidebar.header("🛠️ Market Data Config")

selected_tickers = st.sidebar.multiselect(
    "Select Tickers",
    options=DEFAULT_TICKERS + ["GOTO.JK", "BUKA.JK", "ADRO.JK", "PTBA.JK"],
    default=DEFAULT_TICKERS[:5]
)

rf_rate = st.sidebar.slider(
    "Annual Risk-Free Rate (%)",
    min_value=0.0,
    max_value=15.0,
    value=float(RISK_FREE_RATE_ANNUAL * 100),
    step=0.25
) / 100

lookback_years = st.sidebar.selectbox(
    "Historical Lookback (Years)",
    options=[1, 2, 3, 5, 10],
    index=3
)

st.sidebar.header("⚙️ Advanced Models Config")
allow_short = st.sidebar.checkbox("Allow Short Selling", value=False)
apply_tc = st.sidebar.checkbox("Apply Transaction Costs (10 bps turnover penalty)", value=False)
tc_rate = 0.001 if apply_tc else 0.0

use_garch = st.sidebar.checkbox("Forecast Volatility with GARCH(1,1)", value=False)
use_bl = st.sidebar.checkbox("Apply Black-Litterman Views", value=False)

if st.sidebar.button("Run Optimization", type="primary"):
    if len(selected_tickers) < 2:
        st.error("Please select at least 2 tickers.")
    else:
        with st.spinner("Fetching latest data and compiling optimization matrices..."):
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=365 * lookback_years)).strftime("%Y-%m-%d")
            
            fetcher = StockDataFetcher(selected_tickers, start_date, end_date)
            try:
                prices = fetcher.fetch()
                log_returns = fetcher.to_log_returns(prices)
                
                # Pre-processing for advanced models
                if use_garch:
                    st.info("Fitting GARCH(1,1) for volatility forecasting...")
                    forecasted_vols = []
                    for ticker in log_returns.columns:
                        garch = GARCHVolatilityModel(log_returns[ticker])
                        garch.fit()
                        # Forecast 1Y horizon vol
                        forecasted_vols.append(garch.forecast_volatility(horizon=252))
                    
                    # Implied Correlation Matrix
                    corr_matrix = log_returns.corr().values
                    # Reconstruct Covariance from Forecasted Vols
                    diag_vol = np.diag(forecasted_vols)
                    cov_matrix = diag_vol @ corr_matrix @ diag_vol
                else:
                    cov_matrix = log_returns.cov().values * 252
                
                mean_returns = log_returns.mean().values * 252
                
                if use_bl:
                    st.info("Applying Black-Litterman Subjective Views...")
                    mkt_weights = np.array([1/len(selected_tickers)] * len(selected_tickers))
                    bl = BlackLittermanModel(cov_matrix, mkt_weights)
                    
                    # Dummy view: asset 0 outperforms asset 1 by 5%
                    P = np.zeros((1, len(selected_tickers)))
                    P[0, 0] = 1.0
                    P[0, 1] = -1.0
                    Q = np.array([0.05])
                    
                    mean_returns = bl.posterior_returns(P, Q)

                # Overwrite EF initialization dynamically
                ef = EfficientFrontier(log_returns, allow_short=allow_short, risk_free_rate=rf_rate)
                ef.cov_matrix = cov_matrix
                ef.mean_returns = mean_returns
                
                col1, col2 = st.columns(2)
                
                initial_cash_weights = np.zeros(len(selected_tickers)) # For turnover calc
                max_sharpe = ef.optimize_max_sharpe(current_weights=initial_cash_weights, transaction_cost=tc_rate)
                min_var = ef.optimize_min_variance()
                random_portfolios = ef.get_random_portfolios(n=5000)
                frontier = ef.generate_frontier(n_points=100)
                
                with col1:
                    st.subheader("📈 Efficient Frontier")
                    fig_ef = plot_efficient_frontier(frontier, random_portfolios, max_sharpe, min_var, selected_tickers)
                    st.plotly_chart(fig_ef, use_container_width=True)
                
                with col2:
                    st.subheader("⚖️ Optimal Allocation Vectors")
                    fig_weights = plot_weights_bar(max_sharpe, min_var, list(prices.columns))
                    st.plotly_chart(fig_weights, use_container_width=True)
                
                st.divider()
                
                col3, col4 = st.columns(2)
                
                with col3:
                    st.subheader("🔗 Risk Parity Marginal Contributions")
                    rp = RiskParityOptimizer(cov_matrix, selected_tickers)
                    res_rp = rp.optimize()
                    fig_rp = plot_risk_parity_contributions(res_rp['risk_contributions'], selected_tickers)
                    st.plotly_chart(fig_rp, use_container_width=True)
                    
                with col4:
                    st.subheader("🎲 Stochastic Projection (GBM, 1Y)")
                    weights_arr = np.array(list(max_sharpe['weights'].values()))
                    
                    # Need an unmodified log_returns or pass explicit cov/mean for simulation
                    # For purity, we use historical log_returns to maintain drift properties
                    sim = MonteCarloSimulator(log_returns, weights_arr)
                    paths = sim.simulate_price_paths(n_simulations=500, n_days=252)
                    stats = sim.get_path_statistics(paths)
                    var_cvar = sim.compute_var_cvar(paths)
                    
                    fig_paths = plot_price_paths(paths, stats, var_cvar)
                    st.plotly_chart(fig_paths, use_container_width=True)
                
                st.subheader("📊 Execution Summary")
                summary_df = pd.DataFrame([max_sharpe, min_var], index=['Max Sharpe', 'Min Variance'])
                summary_df = summary_df[['return', 'volatility', 'sharpe']]
                summary_df['return'] = summary_df['return'].apply(lambda x: f"{x:.2%}")
                summary_df['volatility'] = summary_df['volatility'].apply(lambda x: f"{x:.2%}")
                summary_df['sharpe'] = summary_df['sharpe'].apply(lambda x: f"{x:.2f}")
                st.table(summary_df)
                
            except Exception as e:
                st.error(f"Execution Error: {e}")
else:
    st.info("👈 Configure target market and instantiate models via the sidebar.")
    
st.sidebar.markdown("---")
st.sidebar.caption("Data infrastructure via yfinance.")
st.sidebar.caption("© 2026 AlphaIDX Quantitative Desk")
