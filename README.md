# AlphaIDX: Quantitative Portfolio Optimization & Risk Engine

![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue?logo=python)
![scipy](https://img.shields.io/badge/scipy-optimize-orange)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-lightblue)
![Streamlit](https://img.shields.io/badge/Streamlit-Live_Dashboard-FF4B4B?logo=streamlit)
![Tests](https://img.shields.io/badge/Tests-73_Passed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

## Executive Summary
AlphaIDX is a quantitative finance engine built for Indonesia Stock Exchange (IDX) blue-chip equities. It pairs **Modern Portfolio Theory (Markowitz MPT)**, **Ledoit-Wolf covariance shrinkage**, and **Min CVaR (Expected Shortfall)** optimization with **Geometric Brownian Motion (GBM)** Monte Carlo risk projections using Itô drift correction, antithetic variates, and fat-tailed **Student-$t$** innovations.

To address mean-variance estimation sensitivity and emerging-market tail risk, the engine includes **concentration-capped optimization (`max_weight`)**, **Risk Parity**, **Hierarchical Risk Parity (HRP, López de Prado 2016)**, the **Black-Litterman** model (solved via the Woodbury identity), **GARCH(1,1)** conditional volatility forecasting (Gaussian and Student-$t$ MLE), L1 turnover transaction cost penalties, and rolling **walk-forward out-of-sample backtesting**. All capabilities are exposed through a non-interactive CLI, an interactive Streamlit web app, and five research notebooks.

## Visual Output Preview

![Efficient Frontier](assets/efficient_frontier.png)
![Out-of-Sample Backtest](assets/oos_backtest.png)

## Empirical Results (10 IDX Blue-Chips)

### In-Sample Optimization & 1Y Risk Projection (2019-01-01 to 2024-12-31, Sample Covariance)

| Metric | Max Sharpe Portfolio | Min Variance Portfolio |
|--------|----------------------|------------------------|
| Annualized Return | 12.67% | 2.37% |
| Annualized Volatility | 23.68% | 18.17% |
| Sharpe Ratio ($R_f = 6.25\%$) | 0.27 | -0.21 |
| 1Y Monte Carlo VaR (95%) | 25.38% | 25.41% |
| 1Y Monte Carlo CVaR (95%) | 32.23% | 30.75% |
| In-Sample Max Drawdown | -36.15% | -36.01% |

### Out-of-Sample Backtest (2025-01-01 to 2026-05-25, 10 bps Rebalance Cost)

| Strategy | Realized Return | Realized Volatility | Realized Sharpe | Max Drawdown | Avg Turnover | Cumulative Return |
|----------|-----------------|---------------------|-----------------|--------------|--------------|-------------------|
| Static Max Sharpe (Sample) | -31.78% | 28.64% | -1.33 | -34.94% | - | -33.54% |
| Static Min Variance (Sample) | -15.07% | 21.92% | -0.97 | -23.66% | - | -17.62% |
| Quarterly Walk-Forward Max Sharpe (Ledoit-Wolf, Uncapped) | -7.82% | 36.80% | -0.38 | -37.45% | 75.56% | -9.57% |
| **Quarterly Walk-Forward HRP (Correlation Clustering)** | **-0.65%** | **21.67%** | **-0.32** | **-17.61%** | **37.94%** | **-0.83%** |
| **Quarterly Walk-Forward Min Variance (Ledoit-Wolf)** | **+2.55%** | **22.37%** | **-0.17** | **-18.03%** | **48.17%** | **+3.33%** |
| **Quarterly Walk-Forward Max Sharpe (Ledoit-Wolf, Cap 25%)** | **+3.23%** | **24.40%** | **-0.12** | **-21.74%** | **49.19%** | **+4.24%** |
| **Quarterly Walk-Forward Min CVaR (Historical 95%)** | **+5.61%** | **23.61%** | **-0.03** | **-20.85%** | **60.43%** | **+7.48%** |
| Equal-Weight Benchmark | - | - | - | - | - | -6.98% |

*Key empirical findings during the 2025–2026 IDX sector rotation:*
1. **Tail-Risk Minimization (`Min CVaR 95%`, `+7.48%` cumulative return):** Directly penalizing the worst 5% daily empirical losses outperformed variance minimization (`+3.33%`) and beat the equal-weight benchmark (`-6.98%`) by **+14.46 percentage points** because IDX daily returns exhibit severe excess kurtosis (`21.05`, fitted Student-$t$ $\nu = 3.49$).
2. **Concentration Cap (`max_weight=0.25`, `+4.24%` vs `-9.57%` uncapped):** Unconstrained `Max Sharpe` acts as an "error maximizer" (Michaud 1989; Jagannathan & Ma 2003), allocating `>70%` to a single past winner. Capping per-asset weights at `25%` lifted OOS cumulative return by **+13.81 percentage points** and reduced max drawdown from `-37.45%` to `-21.74%`.
3. **Hierarchical Risk Parity (`HRP`, `-17.61%` max drawdown):** Clustering correlated banking stocks (`BBCA.JK`, `BBRI.JK`, `BMRI.JK`, $\rho = 0.56\text{--}0.60$) via single-linkage tree bisection achieved the lowest volatility (`21.67%`), lowest drawdown (`-17.61%`), and lowest quarterly turnover (`37.94%`) among all dynamic strategies.

## Mathematical Foundation

### 1. Portfolio Metrics & Fat-Tailed VaR/CVaR
Expected return, variance, and parametric Normal or Student-$t$ tail risk are computed via vectorized operations:
```text
E[R_p] = w^T · μ
σ²_p   = w^T · Σ · w
S      = (E[R_p] - R_f) / σ_p
```

### 2. Efficient Frontier, Ledoit-Wolf Shrinkage & Min CVaR
The optimizer solves constrained quadratic and tail-risk programs using warm-started `SLSQP` with configurable concentration bounds $w_i \in [0, w_{\max}]$:
```text
Minimize:   w^T · Σ · w   (or Empirical CVaR_α(w))
Subject to: Σ w_i = 1,  0 <= w_i <= w_max,  w^T · μ = μ_target
```

### 3. Stochastic Projection (GBM with Itô Correction, Antithetic Variates & Student-t Innovations)
Price trajectories incorporate the Itô drift adjustment and paired $(Z, -Z)$ antithetic draws under Gaussian or variance-normalized Student-$t$ innovations:
```text
ln(S_t / S_0) = (μ - 0.5σ²) · t + σ · W_t
```

## Project Structure
```text
AlphaIDX/
├── .github/workflows/
│   └── ci.yml                       # Ruff, Black, pytest, and CLI smoke tests (Python 3.11 & 3.12)
├── assets/                          # Empirical chart previews
├── notebooks/
│   ├── 01_data_and_eda.ipynb        # Market context, fat tails & sector correlations
│   ├── 02_optimization.ipynb        # Markowitz MPT & Efficient Frontier
│   ├── 03_monte_carlo.ipynb         # GBM stochastic projection & 1Y VaR/CVaR
│   ├── 04_backtesting.ipynb         # Out-of-Sample validation
│   └── 05_advanced_allocations.ipynb# Risk Parity & Black-Litterman models
├── src/
│   ├── cli.py                       # Non-interactive CLI (`python -m src.cli` / `alphaidx`)
│   ├── live_dashboard.py            # Streamlit interactive dashboard
│   ├── data/
│   │   └── fetcher.py               # yfinance API, quality checks & 24h CSV cache
│   ├── portfolio/
│   │   ├── metrics.py               # Market constants, .env config & Normal/Student-t VaR/CVaR
│   │   ├── optimizer.py             # SLSQP Efficient Frontier, Min CVaR & Ledoit-Wolf shrinkage
│   │   ├── monte_carlo.py           # Antithetic GBM Monte Carlo simulator (Normal & Student-t)
│   │   ├── garch_model.py           # GARCH(1,1) conditional volatility (Normal & Student-t MLE)
│   │   ├── advanced_models.py       # Risk Parity, Hierarchical Risk Parity (HRP) & Black-Litterman
│   │   └── backtester.py            # Buy-and-hold & rolling walk-forward backtester
│   └── visualization/
│       └── plots.py                 # 9 interactive Plotly visualizations
├── tests/                           # 73 unit tests across 10 test modules
├── pyproject.toml                   # Python >=3.11,<3.13 package, pytest, ruff & black config
├── requirements.txt                 # Compatible dependency bounds
├── .env.example                     # Optional environment variable overrides
└── LICENSE                          # MIT
```

## Quickstart

### 1. Clone and Install (Python 3.11 or 3.12)
```bash
git clone https://github.com/Reiii0-0/AlphaIDX.git
cd AlphaIDX
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### 2. Run Non-Interactive CLI Pipeline
```bash
python -m src.cli run --start 2019-01-01 --end 2024-12-31 --max-weight 0.25 --dist t --oos-start 2025-01-01 --oos-end 2026-05-25 --output-dir reports/
python -m src.cli backtest --rebalance-freq 63 --cov-method ledoit_wolf --max-weight 0.25 --format json
```

### 3. Launch Live Dashboard or Research Notebooks
```bash
streamlit run src/live_dashboard.py
jupyter lab
```

### 4. Run Tests and Linters
```bash
pytest -v
ruff check src tests
black --check src tests
```

## Data Sources & Limitations
The project uses **yfinance** to fetch Adjusted Close prices for 10 IDX blue-chip tickers (`BBCA.JK`, `BBRI.JK`, `BMRI.JK`, `TLKM.JK`, `EXCL.JK`, `UNVR.JK`, `ICBP.JK`, `INDF.JK`, `KLBF.JK`, `ASII.JK`).

- **Fat-Tail Dynamics:** Both Gaussian (`dist="normal"`) and variance-normalized Student-$t$ (`dist="t"`, default $\nu = 4.0$) innovations are supported across parametric VaR/CVaR, GARCH(1,1), and GBM Monte Carlo, alongside non-parametric empirical `Min CVaR` optimization.
- **GBM Stationarity:** GBM projects with constant drift and volatility unless paired with GARCH(1,1) covariance updates.
- **Market Friction:** Optimization and walk-forward backtests model proportional L1 turnover costs (`10 bps` default) without modeling order-book slippage or IDX 100-share lot-size rounding.

## License
This project is licensed under the MIT License; see [LICENSE](LICENSE) for details.

