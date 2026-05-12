"""
Visualization module for portfolio analysis using Plotly.
"""

import logging
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import gaussian_kde, norm

logger = logging.getLogger(__name__)

__all__ = [
    "plot_efficient_frontier",
    "plot_monte_carlo_portfolios",
    "plot_correlation_heatmap",
    "plot_price_paths",
    "plot_return_distribution",
    "plot_weights_bar",
    "plot_cumulative_returns",
]

# ── Color Constants ──
COLOR_SCATTER = "#94A3B8"  # random portfolios (gray-blue)
COLOR_FRONTIER = "#3B82F6"  # efficient frontier line (blue)
COLOR_MAX_SHARPE = "#10B981"  # max Sharpe portfolio (green)
COLOR_MIN_VAR = "#F59E0B"  # min variance portfolio (amber)
COLOR_PATHS = "rgba(148,163,184,0.15)"  # MC paths (faint)
COLOR_MEDIAN_PATH = "#EF4444"  # median GBM path (red)
TEMPLATE = "plotly_white"


def plot_efficient_frontier(
    frontier: list[dict],
    random_portfolios: pd.DataFrame,
    max_sharpe: dict,
    min_var: dict,
    tickers: list[str],
) -> go.Figure:
    """
    Plot the efficient frontier with random portfolios and optimal points.
    """
    fig = go.Figure()

    # Scatter: random portfolios
    fig.add_trace(
        go.Scatter(
            x=random_portfolios["volatility"] * 100,
            y=random_portfolios["return"] * 100,
            mode="markers",
            marker=dict(
                color=random_portfolios["sharpe"],
                colorscale="RdYlGn",
                showscale=True,
                colorbar=dict(title="Sharpe Ratio"),
                size=5,
                opacity=0.6,
            ),
            name="Random Portfolios",
            hovertemplate="Return: %{y:.2f}%<br>Volatility: %{x:.2f}%<br>Sharpe: %{marker.color:.2f}<extra></extra>",
        )
    )

    # Line: efficient frontier
    f_vols = [p["volatility"] * 100 for p in frontier]
    f_rets = [p["return"] * 100 for p in frontier]
    fig.add_trace(
        go.Scatter(
            x=f_vols,
            y=f_rets,
            mode="lines",
            line=dict(color=COLOR_FRONTIER, width=3),
            name="Efficient Frontier",
        )
    )

    # Marker: Max Sharpe
    fig.add_trace(
        go.Scatter(
            x=[max_sharpe["volatility"] * 100],
            y=[max_sharpe["return"] * 100],
            mode="markers+text",
            marker=dict(color=COLOR_MAX_SHARPE, size=16, symbol="star"),
            name="Max Sharpe",
            text=[f"Max Sharpe ({max_sharpe['sharpe']:.2f})"],
            textposition="top center",
        )
    )

    # Marker: Min Variance
    fig.add_trace(
        go.Scatter(
            x=[min_var["volatility"] * 100],
            y=[min_var["return"] * 100],
            mode="markers+text",
            marker=dict(color=COLOR_MIN_VAR, size=16, symbol="diamond"),
            name="Min Variance",
            text=["Min Variance"],
            textposition="bottom center",
        )
    )

    fig.update_layout(
        title="Efficient Frontier & Portfolio Optimization",
        xaxis_title="Annual Volatility (%)",
        yaxis_title="Annual Return (%)",
        template=TEMPLATE,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    return fig


def plot_monte_carlo_portfolios(
    random_portfolios: pd.DataFrame, max_sharpe: dict, min_var: dict
) -> go.Figure:
    """
    Plot simulated random portfolios colored by Sharpe ratio.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=random_portfolios["volatility"] * 100,
            y=random_portfolios["return"] * 100,
            mode="markers",
            marker=dict(
                color=random_portfolios["sharpe"],
                colorscale="RdYlGn",
                showscale=True,
                colorbar=dict(title="Sharpe Ratio"),
                size=6,
            ),
            name="Simulated Portfolios",
            hovertemplate="Return: %{y:.2f}%<br>Volatility: %{x:.2f}%<br>Sharpe: %{marker.color:.2f}",
        )
    )

    # Optimal points
    fig.add_trace(
        go.Scatter(
            x=[max_sharpe["volatility"] * 100],
            y=[max_sharpe["return"] * 100],
            mode="markers",
            marker=dict(color=COLOR_MAX_SHARPE, size=14, symbol="star", line=dict(width=2, color="black")),
            name="Max Sharpe",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[min_var["volatility"] * 100],
            y=[min_var["return"] * 100],
            mode="markers",
            marker=dict(color=COLOR_MIN_VAR, size=14, symbol="diamond", line=dict(width=2, color="black")),
            name="Min Variance",
        )
    )

    fig.update_layout(
        title="Monte Carlo Portfolio Sampling",
        xaxis_title="Annual Volatility (%)",
        yaxis_title="Annual Return (%)",
        template=TEMPLATE,
        annotations=[
            dict(
                x=0.05, y=0.95, xref="paper", yref="paper",
                text="Color scale represents Sharpe Ratio (Green = Higher)",
                showarrow=False, font=dict(size=12)
            )
        ]
    )

    return fig


def plot_correlation_heatmap(log_returns: pd.DataFrame) -> go.Figure:
    """
    Plot correlation matrix as a heatmap.
    """
    corr = log_returns.corr()
    # Clean labels: strip .JK
    labels = [c.replace(".JK", "") for c in corr.columns]

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=labels,
            y=labels,
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            texttemplate="%{z:.2f}",
            hovertemplate="Ticker 1: %{y}<br>Ticker 2: %{x}<br>Correlation: %{z:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Asset Correlation Heatmap",
        template=TEMPLATE,
        yaxis=dict(scaleanchor="x"),
        width=700,
        height=700,
    )

    return fig


def plot_price_paths(paths: np.ndarray, stats: pd.DataFrame, var_cvar: dict) -> go.Figure:
    """
    Plot Monte Carlo simulation price paths.
    """
    fig = go.Figure()
    n_days = paths.shape[1] - 1
    x = list(range(n_days + 1))

    # Plot first 200 paths
    for i in range(min(200, paths.shape[0])):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=paths[i, :],
                mode="lines",
                line=dict(color=COLOR_PATHS, width=0.5),
                showlegend=False,
            )
        )

    # Shaded band (5th - 95th percentile)
    fig.add_trace(
        go.Scatter(
            x=x + x[::-1],
            y=stats["pct_95"].tolist() + stats["pct_5"].tolist()[::-1],
            fill="toself",
            fillcolor="rgba(59,130,246,0.1)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="5th-95th Percentile",
        )
    )

    # Median path
    fig.add_trace(
        go.Scatter(
            x=x,
            y=stats["median"],
            mode="lines",
            line=dict(color=COLOR_MEDIAN_PATH, width=2.5),
            name="Median Path",
        )
    )

    fig.update_layout(
        title="Monte Carlo Simulation: Portfolio Value Projections",
        xaxis_title="Trading Days",
        yaxis_title="Portfolio Value (Base = 100)",
        template=TEMPLATE,
        annotations=[
            dict(
                x=n_days, y=var_cvar["percentile_5"],
                xref="x", yref="y",
                text=f"5th pct (1yr): {var_cvar['percentile_5']:.1f}",
                showarrow=True, arrowhead=2, ax=-100, ay=30
            )
        ]
    )

    return fig


def plot_return_distribution(log_returns: pd.DataFrame, optimal_weights: np.ndarray) -> go.Figure:
    """
    Plot the distribution of portfolio log returns.
    """
    port_returns = log_returns @ optimal_weights
    
    fig = go.Figure()

    # Histogram
    fig.add_trace(
        go.Histogram(
            x=port_returns,
            histnorm="probability density",
            name="Portfolio Returns",
            marker_color=COLOR_FRONTIER,
            opacity=0.6,
            nbinsx=50
        )
    )

    # KDE
    x_range = np.linspace(port_returns.min(), port_returns.max(), 100)
    kde = gaussian_kde(port_returns)
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=kde(x_range),
            mode="lines",
            line=dict(color="black", width=2),
            name="KDE"
        )
    )

    # Normal Distribution overlay
    mu, std = port_returns.mean(), port_returns.std()
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=norm.pdf(x_range, mu, std),
            mode="lines",
            line=dict(color="gray", width=2, dash="dash"),
            name="Normal Dist."
        )
    )

    # Metrics lines
    fig.add_vline(x=mu, line_color="green", annotation_text="Mean")
    fig.add_vline(x=mu-std, line_dash="dot", line_color="orange", annotation_text="-1σ")
    fig.add_vline(x=mu+std, line_dash="dot", line_color="orange", annotation_text="+1σ")

    fig.update_layout(
        title="Portfolio Return Distribution vs Normal Assumption",
        xaxis_title="Daily Log Return",
        yaxis_title="Density",
        template=TEMPLATE
    )

    return fig


def plot_weights_bar(max_sharpe: dict, min_var: dict, tickers: list[str]) -> go.Figure:
    """
    Plot a bar chart comparing weights of two portfolios.
    """
    clean_tickers = [t.replace(".JK", "") for t in tickers]
    
    # Sort by Max Sharpe weight descending
    ms_weights = [max_sharpe["weights"][t] for t in tickers]
    mv_weights = [min_var["weights"][t] for t in tickers]
    
    sorted_indices = np.argsort(ms_weights)[::-1]
    
    fig = go.Figure(
        data=[
            go.Bar(
                name="Max Sharpe",
                x=[clean_tickers[i] for i in sorted_indices],
                y=[ms_weights[i] * 100 for i in sorted_indices],
                marker_color=COLOR_MAX_SHARPE,
                text=[f"{ms_weights[i]*100:.1f}%" for i in sorted_indices],
                textposition="auto",
            ),
            go.Bar(
                name="Min Variance",
                x=[clean_tickers[i] for i in sorted_indices],
                y=[mv_weights[i] * 100 for i in sorted_indices],
                marker_color=COLOR_MIN_VAR,
                text=[f"{mv_weights[i]*100:.1f}%" for i in sorted_indices],
                textposition="auto",
            ),
        ]
    )

    fig.update_layout(
        title="Asset Allocation: Max Sharpe vs Min Variance",
        xaxis_title="Ticker",
        yaxis_title="Weight (%)",
        yaxis_range=[0, 100],
        barmode="group",
        template=TEMPLATE
    )

    return fig


def plot_cumulative_returns(
    prices: pd.DataFrame, max_sharpe_weights: np.ndarray, tickers: list[str]
) -> go.Figure:
    """
    Plot cumulative returns of the portfolio vs individual assets.
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()
    
    # Portfolio cumulative
    port_log_ret = log_returns @ max_sharpe_weights
    port_cum = np.exp(port_log_ret.cumsum()) - 1
    
    fig = go.Figure()

    # Individual assets
    for ticker in tickers:
        asset_cum = np.exp(log_returns[ticker].cumsum()) - 1
        fig.add_trace(
            go.Scatter(
                x=asset_cum.index,
                y=asset_cum * 100,
                mode="lines",
                line=dict(color="lightgray", width=1),
                name=ticker.replace(".JK", ""),
                legendgroup="Assets",
                showlegend=True
            )
        )

    # Portfolio
    fig.add_trace(
        go.Scatter(
            x=port_cum.index,
            y=port_cum * 100,
            mode="lines",
            line=dict(color=COLOR_FRONTIER, width=3),
            name="Optimal Portfolio (Max Sharpe)"
        )
    )

    fig.update_layout(
        title="Cumulative Performance: Optimal Portfolio vs Individual Assets",
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        template=TEMPLATE,
        hovermode="x unified"
    )
    
    # Add benchmark annotation
    final_ret = port_cum.iloc[-1] * 100
    fig.add_annotation(
        x=port_cum.index[-1], y=final_ret,
        text=f"Total: {final_ret:.1f}%",
        showarrow=True, arrowhead=2, ax=-60, ay=-30
    )

    return fig
