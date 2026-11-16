"""
Live Portfolio Dashboard using Streamlit.
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Literal

import numpy as np
import pandas as pd
import streamlit as st

# Fallback path when invoked directly via `streamlit run src/live_dashboard.py` without editable install.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data.fetcher import StockDataFetcher  # noqa: E402
from src.portfolio.advanced_models import (  # noqa: E402
    BlackLittermanModel,
    HierarchicalRiskParityOptimizer,
    RiskParityOptimizer,
)
from src.portfolio.garch_model import GARCHVolatilityModel  # noqa: E402
from src.portfolio.metrics import (  # noqa: E402
    DEFAULT_TICKERS,
    RISK_FREE_RATE_ANNUAL,
    TRADING_DAYS_PER_YEAR,
)
from src.portfolio.monte_carlo import MonteCarloSimulator  # noqa: E402
from src.portfolio.optimizer import EfficientFrontier, ledoit_wolf_shrinkage  # noqa: E402
from src.visualization.plots import (  # noqa: E402
    plot_efficient_frontier,
    plot_price_paths,
    plot_risk_parity_contributions,
    plot_weights_bar,
)


def compute_dashboard_analytics(
    log_returns: pd.DataFrame,
    rf_rate: float = RISK_FREE_RATE_ANNUAL,
    cov_method: Literal["ledoit_wolf", "sample"] = "ledoit_wolf",
    allow_short: bool = False,
    tc_rate: float = 0.0,
    initial_weights_mode: Literal["equal", "cash"] = "equal",
    use_garch: bool = False,
    bl_view: dict | None = None,
    max_weight: float = 1.0,
    dist: Literal["normal", "t"] = "normal",
) -> dict:
    """
    Run the end-to-end dashboard analytics pipeline using consistent sorted ticker alignment.
    """
    active_tickers = list(log_returns.columns)
    n_assets = len(active_tickers)

    if use_garch:
        garch_dist: Literal["Normal", "t"] = "t" if dist == "t" else "Normal"
        forecasted_vols = []
        for ticker in active_tickers:
            garch = GARCHVolatilityModel(log_returns[ticker], dist=garch_dist)
            garch.fit()
            forecasted_vols.append(garch.forecast_volatility(horizon=TRADING_DAYS_PER_YEAR))
        corr_matrix = log_returns.corr().to_numpy(dtype=float)
        diag_vol = np.diag(forecasted_vols)
        cov_matrix = diag_vol @ corr_matrix @ diag_vol
    elif cov_method == "ledoit_wolf":
        cov_matrix = ledoit_wolf_shrinkage(log_returns)
    else:
        cov_matrix = log_returns.cov().to_numpy(dtype=float) * TRADING_DAYS_PER_YEAR

    mean_returns = log_returns.mean().to_numpy(dtype=float) * TRADING_DAYS_PER_YEAR

    if bl_view is not None:
        asset_a = bl_view["asset_a"]
        if asset_a not in active_tickers:
            raise ValueError(f"Black-Litterman ticker {asset_a} not found in active tickers.")
        p_mat = np.zeros((1, n_assets), dtype=float)
        p_mat[0, active_tickers.index(asset_a)] = 1.0
        if bl_view.get("view_type") == "Relative":
            asset_b = bl_view.get("asset_b")
            if not asset_b or asset_b == asset_a or asset_b not in active_tickers:
                raise ValueError(
                    "Relative Black-Litterman view requires a distinct active asset_b."
                )
            p_mat[0, active_tickers.index(asset_b)] = -1.0

        q_vec = np.array([float(bl_view["expected_return"])], dtype=float)
        tau = float(bl_view.get("tau", 0.05))
        mkt_weights = np.full(n_assets, 1.0 / n_assets)
        bl = BlackLittermanModel(cov_matrix, mkt_weights, tau=tau)
        mean_returns = bl.posterior_returns(p_mat, q_vec)

    ef = EfficientFrontier(
        log_returns,
        allow_short=allow_short,
        risk_free_rate=rf_rate,
        cov_method=cov_method,
        mean_returns=mean_returns,
        cov_matrix=cov_matrix,
        max_weight=max_weight,
    )
    init_weights = (
        np.full(n_assets, 1.0 / n_assets) if initial_weights_mode == "equal" else np.zeros(n_assets)
    )
    max_sharpe = ef.optimize_max_sharpe(current_weights=init_weights, transaction_cost=tc_rate)
    min_var = ef.optimize_min_variance()
    min_cvar = ef.optimize_min_cvar()
    random_portfolios = ef.get_random_portfolios(n=5000)
    frontier = ef.generate_frontier(n_points=100)

    rp = RiskParityOptimizer(cov_matrix, active_tickers)
    res_rp = rp.optimize()
    res_hrp = HierarchicalRiskParityOptimizer(log_returns, cov_matrix=cov_matrix).optimize()

    weights_arr = np.array([max_sharpe["weights"][t] for t in active_tickers], dtype=float)
    sim = MonteCarloSimulator(
        log_returns,
        weights_arr,
        mean_returns_annual=mean_returns,
        cov_matrix_annual=cov_matrix,
    )
    paths = sim.simulate_price_paths(
        n_simulations=500,
        n_days=TRADING_DAYS_PER_YEAR,
        antithetic=True,
        dist=dist,
    )
    stats = sim.get_path_statistics(paths)
    var_cvar = sim.compute_var_cvar(paths)

    return {
        "active_tickers": active_tickers,
        "mean_returns": mean_returns,
        "cov_matrix": cov_matrix,
        "max_sharpe": max_sharpe,
        "min_var": min_var,
        "min_cvar": min_cvar,
        "random_portfolios": random_portfolios,
        "frontier": frontier,
        "risk_parity": res_rp,
        "hrp": res_hrp,
        "paths": paths,
        "path_stats": stats,
        "var_cvar": var_cvar,
    }


def main() -> None:
    st.set_page_config(page_title="AlphaIDX | Live Portfolio Optimizer", layout="wide")

    st.title("AlphaIDX: Institutional Quant Engine")
    st.markdown(
        "Real-time IDX portfolio optimization using **Mean-Variance** (Sample or Ledoit-Wolf shrinkage), "
        "**Min CVaR**, **Risk Parity & HRP**, **Black-Litterman**, **GARCH(1,1)**, and **Antithetic GBM Monte Carlo**."
    )

    st.sidebar.header("Market Data Config")
    ticker_pool = list(
        dict.fromkeys(DEFAULT_TICKERS + ["GOTO.JK", "BUKA.JK", "ADRO.JK", "PTBA.JK"])
    )
    selected_tickers = st.sidebar.multiselect(
        "Select Tickers",
        options=ticker_pool,
        default=DEFAULT_TICKERS[:5],
    )

    rf_rate = (
        st.sidebar.slider(
            "Annual Risk-Free Rate (%)",
            min_value=0.0,
            max_value=15.0,
            value=float(RISK_FREE_RATE_ANNUAL * 100),
            step=0.25,
        )
        / 100.0
    )

    lookback_years = st.sidebar.selectbox(
        "Historical Lookback (Years)",
        options=[1, 2, 3, 5, 10],
        index=3,
    )

    st.sidebar.header("Advanced Models Config")
    cov_choice = st.sidebar.selectbox(
        "Covariance Estimator",
        options=["ledoit_wolf", "sample"],
        format_func=lambda m: (
            "Ledoit-Wolf Shrinkage" if m == "ledoit_wolf" else "Sample Covariance"
        ),
    )
    allow_short = st.sidebar.checkbox("Allow Short Selling", value=False)
    max_weight_pct = st.sidebar.slider(
        "Max Weight per Asset (%)",
        min_value=20,
        max_value=100,
        value=100,
        step=5,
        disabled=allow_short,
    )
    max_weight = max_weight_pct / 100.0
    dist_choice: Literal["normal", "t"] = st.sidebar.selectbox(
        "Return Innovation Distribution",
        options=["normal", "t"],
        format_func=lambda d: "Student-t (Fat-Tailed)" if d == "t" else "Gaussian (Normal)",
    )
    apply_tc = st.sidebar.checkbox("Apply Transaction Costs (10 bps turnover)", value=False)
    tc_rate = 0.001 if apply_tc else 0.0
    init_mode: Literal["equal", "cash"] = "equal"
    if apply_tc:
        init_mode = st.sidebar.radio(
            "Rebalancing Baseline",
            options=["equal", "cash"],
            format_func=lambda x: "Equal Weight (Rebalance)" if x == "equal" else "New Cash (0%)",
        )

    use_garch = st.sidebar.checkbox("Forecast Volatility with GARCH(1,1)", value=False)
    use_bl = st.sidebar.checkbox("Apply Black-Litterman View", value=False)

    bl_view: dict | None = None
    if use_bl and len(selected_tickers) >= 2:
        st.sidebar.subheader("Black-Litterman View")
        view_type = st.sidebar.radio("View Type", options=["Relative", "Absolute"])
        asset_a = st.sidebar.selectbox("Target Asset", options=selected_tickers, index=0)
        asset_b = None
        if view_type == "Relative":
            other_tickers = [t for t in selected_tickers if t != asset_a]
            asset_b = st.sidebar.selectbox("Benchmark Asset", options=other_tickers, index=0)
        view_ret_pct = st.sidebar.slider(
            "Expected Return / Outperformance (%)",
            min_value=-20.0,
            max_value=40.0,
            value=5.0 if view_type == "Relative" else 12.0,
            step=0.5,
        )
        bl_tau = st.sidebar.slider("View Uncertainty Scale (tau)", 0.01, 0.20, 0.05, 0.01)
        bl_view = {
            "view_type": view_type,
            "asset_a": asset_a,
            "asset_b": asset_b,
            "expected_return": view_ret_pct / 100.0,
            "tau": bl_tau,
        }

    if st.sidebar.button("Run Optimization", type="primary"):
        if len(selected_tickers) < 2:
            st.error("Please select at least 2 tickers.")
        else:
            with st.spinner("Fetching latest data and running optimization..."):
                end_date = datetime.now().strftime("%Y-%m-%d")
                start_date = (datetime.now() - timedelta(days=365 * lookback_years)).strftime(
                    "%Y-%m-%d"
                )
                fetcher = StockDataFetcher(selected_tickers, start_date, end_date)
                try:
                    prices = fetcher.fetch()
                    min_days = 180 if lookback_years == 1 else 200
                    log_returns = fetcher.to_log_returns(prices, min_trading_days=min_days)
                    results = compute_dashboard_analytics(
                        log_returns=log_returns,
                        rf_rate=rf_rate,
                        cov_method=cov_choice,
                        allow_short=allow_short,
                        tc_rate=tc_rate,
                        initial_weights_mode=init_mode,
                        use_garch=use_garch,
                        bl_view=bl_view,
                        max_weight=max_weight,
                        dist=dist_choice,
                    )

                    active_tickers = results["active_tickers"]
                    max_sharpe = results["max_sharpe"]
                    min_var = results["min_var"]
                    min_cvar = results["min_cvar"]

                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Efficient Frontier")
                        fig_ef = plot_efficient_frontier(
                            results["frontier"],
                            results["random_portfolios"],
                            max_sharpe,
                            min_var,
                            active_tickers,
                        )
                        st.plotly_chart(fig_ef)

                    with col2:
                        st.subheader("Optimal Allocation Vectors")
                        fig_weights = plot_weights_bar(max_sharpe, min_var, active_tickers)
                        st.plotly_chart(fig_weights)

                    st.divider()

                    col3, col4 = st.columns(2)
                    with col3:
                        st.subheader("Risk Parity Marginal Contributions")
                        fig_rp = plot_risk_parity_contributions(
                            results["risk_parity"]["risk_contributions"], active_tickers
                        )
                        st.plotly_chart(fig_rp)

                    with col4:
                        st.subheader("Stochastic Projection (GBM, 1Y)")
                        fig_paths = plot_price_paths(
                            results["paths"], results["path_stats"], results["var_cvar"]
                        )
                        st.plotly_chart(fig_paths)

                    st.subheader("Execution Summary")
                    summary_df = pd.DataFrame(
                        [max_sharpe, min_var, min_cvar],
                        index=["Max Sharpe", "Min Variance", "Min CVaR (95%)"],
                    )[["return", "volatility", "sharpe"]]
                    summary_df["return"] = summary_df["return"].apply(lambda x: f"{x:.2%}")
                    summary_df["volatility"] = summary_df["volatility"].apply(lambda x: f"{x:.2%}")
                    summary_df["sharpe"] = summary_df["sharpe"].apply(lambda x: f"{x:.2f}")
                    st.table(summary_df)
                except Exception as exc:
                    st.error(f"Execution Error: {exc}")
    else:
        st.info("Configure target market and models via the sidebar.")

    st.sidebar.markdown("---")
    st.sidebar.caption("Data infrastructure via yfinance.")
    st.sidebar.caption("© 2026 AlphaIDX Quantitative Desk")


if __name__ == "__main__":
    main()
