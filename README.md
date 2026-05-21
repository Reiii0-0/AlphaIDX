# Portfolio Optimizer: Markowitz Efficient Frontier & Monte Carlo Simulation

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![scipy](https://img.shields.io/badge/scipy-optimize-orange)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-lightblue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?logo=jupyter)
![Domain](https://img.shields.io/badge/Domain-Quantitative_Finance-darkgreen)
![MPT](https://img.shields.io/badge/Theory-Markowitz_MPT-purple)
![License](https://img.shields.io/badge/License-MIT-green)

## Executive Summary
This project is a high-fidelity quantitative finance engine designed to optimize equity portfolios using **Modern Portfolio Theory (MPT)** and project future performance via **Monte Carlo simulations**. Targeting the Indonesia Stock Exchange (IDX), it fetches real-market data for 10 blue-chip tickers, computes optimal asset allocations (Maximum Sharpe and Minimum Variance), and models the range of future outcomes using **Geometric Brownian Motion (GBM)** with rigorous **Itô's Lemma** drift correction. 

It demonstrates proficiency in numerical optimization, stochastic modeling, and professional software engineering standards suitable for Quantitative and Data Analyst roles at top-tier financial institutions.

## Visual Output Preview

- **Efficient Frontier:** Interactive scatter plot showing the risk-return tradeoff and the location of optimal portfolios.
- **Monte Carlo Price Paths:** "Fan chart" showing 1,000 simulated trajectories with 5th-95th percentile confidence bands.
- **Correlation Heatmap:** Visualizes asset dependencies using Pearson correlation on log returns.
- **Return Distributions:** Compares empirical log returns against the normal distribution assumption to identify tail risks.

## Results

| Metric | Max Sharpe Portfolio | Min Variance Portfolio |
|--------|---------------------|----------------------|
| Annual Return | 12.67% | 2.37% |
| Annual Volatility | 23.69% | 18.17% |
| Sharpe Ratio | 0.27 | -0.21 |
| 1Y VaR (95%) | 25.40% | [Run NB03] |
| Max Drawdown | -21.4% (approx) | [Run NB01] |

*Results based on 2019–2024 IDX data for 10 blue-chip tickers.*

## Mathematical Foundation

### 1. Portfolio Metrics
Expected Return and Variance are computed using vectorized matrix operations:
```
E[R_p] = w^T · μ
σ²_p = w^T · Σ · w
```

### 2. Sharpe Ratio
Measures risk-adjusted return relative to the risk-free rate (BI Rate ~6.25%):
```
S = (E[R_p] - R_f) / σ_p
```

### 3. Efficient Frontier Optimization
The engine solves a constrained quadratic programming problem to minimize variance for a given target return:
```
Minimize: w^T · Σ · w
Subject to: Σ w_i = 1, w_i ≥ 0, w^T · μ = μ_target
```

### 4. Stochastic Projection (GBM)
Future price paths are simulated using Geometric Brownian Motion. We apply the **Itô correction** to ensure the drift is unbiased:
```
ln(S_t/S_0) = (μ - 0.5σ²)·t + σ·W_t
```

## Project Structure
```text
project/
├── notebooks/
│   ├── 01_data_and_eda.ipynb    # Context & Correlations
│   ├── 02_optimization.ipynb    # MPT & Efficient Frontier
│   └── 03_monte_carlo.ipynb     # GBM Simulation
├── src/
│   ├── data/
│   │   └── fetcher.py           # yfinance API & Caching
│   ├── portfolio/
│   │   ├── metrics.py           # Math core & Constants
│   │   ├── optimizer.py         # Scipy SLSQP Engine
│   │   └── monte_carlo.py       # Stochastic Paths
│   └── visualization/
│       └── plots.py             # Plotly Engine
├── tests/                       # 34 Unit Tests
├── requirements.txt             # Pinned Deps
├── .env.example                 # Config
├── LICENSE                      # MIT
└── README.md                    # Documentation
```

## Assumptions & Limitations
- **Normality:** Parametric VaR and CVaR assume log-returns are normally distributed.
- **Stationarity:** The model assumes that historical mean and covariance are representative of the future.
- **GBM Limits:** Geometric Brownian Motion assumes constant volatility and does not capture "fat tails" or volatility clustering (GARCH).
- **Market Impact:** Optimization does not account for transaction costs, slippage, or liquidity constraints.

## Quickstart


| Notebook | Focus | Key Output |
|----------|-------|------------|
| `01_data_and_eda` | Market Context | Correlation Heatmaps & Return Distributions |
| `02_optimization` | Allocation | Efficient Frontier & Optimal Weight Bars |
| `03_monte_carlo` | Risk/Projection | GBM Path Projections & 1Y VaR/CVaR |

## Quickstart

### 1. Clone and Install
```bash
git clone https://github.com/Reiii0-0/AlphaIDX.git
cd AlphaIDX
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Launch Analysis
```bash
jupyter lab
```
*Run notebooks in order: 01 → 02 → 03.*

## Data Sources
The project uses **yfinance** to fetch Adjusted Close prices for 10 IDX blue-chips:
- **Financials:** BBCA.JK, BBRI.JK, BMRI.JK
- **Telecom:** TLKM.JK, EXCL.JK
- **Consumer/Healthcare:** UNVR.JK, ICBP.JK, INDF.JK, KLBF.JK, ASII.JK

## Extending This Project
- **Short Selling:** Set `allow_short=True` in `EfficientFrontier` to explore the unconstrained frontier.
- **Black-Litterman:** Incorporate subjective market views to adjust the equilibrium returns.
- **GARCH Modeling:** Replace constant volatility with GARCH(1,1) for better volatility clustering.
- **Transaction Costs:** Add a penalty function to the optimizer to account for rebalancing friction.
- **Dash Integration:** Wrap `plots.py` in a Streamlit or Dash web app for real-time monitoring.

## Skills Demonstrated
- **Domain:** Quantitative Finance, Modern Portfolio Theory, Stochastic Calculus (GBM), Risk Management (VaR/CVaR).
- **Technical:** Python (SciPy, NumPy, Pandas), Numerical Optimization (SLSQP), Monte Carlo Methods, Interactive Data Viz (Plotly).
- **Engineering:** Unit Testing (pytest), API Integration, Caching, Type Hinting, Google-style Docstrings.
