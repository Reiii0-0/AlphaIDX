import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from src.portfolio.backtester import Backtester
from src.portfolio.monte_carlo import MonteCarloSimulator
from src.portfolio.optimizer import EfficientFrontier
from src.visualization.plots import (
    plot_backtest_performance,
    plot_correlation_heatmap,
    plot_cumulative_returns,
    plot_efficient_frontier,
    plot_monte_carlo_portfolios,
    plot_price_paths,
    plot_return_distribution,
    plot_risk_parity_contributions,
    plot_weights_bar,
)


@pytest.fixture
def plot_inputs():
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2022-01-03", periods=150)
    tickers = ["ASII.JK", "BBCA.JK", "TLKM.JK"]
    rets = rng.normal([0.0004, 0.0005, 0.0002], 0.012, size=(150, 3))
    log_returns = pd.DataFrame(rets, index=dates, columns=tickers)
    prices = 1000.0 * np.exp(log_returns.cumsum())

    ef = EfficientFrontier(log_returns)
    ms = ef.optimize_max_sharpe()
    mv = ef.optimize_min_variance()
    frontier = ef.generate_frontier(n_points=15)
    rand_df = ef.get_random_portfolios(n=50)

    w_arr = np.array([ms["weights"][t] for t in tickers])
    sim = MonteCarloSimulator(log_returns, w_arr)
    paths = sim.simulate_price_paths(n_simulations=30, n_days=30)
    stats = sim.get_path_statistics(paths)
    var_cvar = sim.compute_var_cvar(paths)

    bt = Backtester(log_returns.iloc[:100], log_returns.iloc[100:])
    bt_res = [
        bt.run_backtest(ms["weights"], "Max Sharpe"),
        bt.run_backtest(mv["weights"], "Min Variance"),
    ]
    bench = bt.get_benchmark_performance()

    return {
        "tickers": tickers,
        "prices": prices,
        "log_returns": log_returns,
        "ms": ms,
        "mv": mv,
        "frontier": frontier,
        "rand_df": rand_df,
        "w_arr": w_arr,
        "paths": paths,
        "stats": stats,
        "var_cvar": var_cvar,
        "bt_res": bt_res,
        "bench": bench,
    }


def test_all_nine_plots_return_valid_figures(plot_inputs):
    d = plot_inputs
    figures = [
        plot_efficient_frontier(d["frontier"], d["rand_df"], d["ms"], d["mv"], d["tickers"]),
        plot_monte_carlo_portfolios(d["rand_df"], d["ms"], d["mv"]),
        plot_correlation_heatmap(d["log_returns"]),
        plot_price_paths(d["paths"], d["stats"], d["var_cvar"]),
        plot_return_distribution(d["log_returns"], d["w_arr"]),
        plot_weights_bar(d["ms"], d["mv"], d["tickers"]),
        plot_cumulative_returns(d["prices"], d["w_arr"], d["tickers"]),
        plot_backtest_performance(d["bt_res"], d["bench"]),
        plot_risk_parity_contributions(
            dict(zip(d["tickers"], [1 / 3, 1 / 3, 1 / 3])), d["tickers"]
        ),
    ]

    for fig in figures:
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1
