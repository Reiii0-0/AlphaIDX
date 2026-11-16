import numpy as np
import pandas as pd
import pytest

from src.portfolio.backtester import Backtester
from src.portfolio.metrics import TRADING_DAYS_PER_YEAR


@pytest.fixture
def synthetic_is_oos():
    rng = np.random.default_rng(123)
    dates = pd.bdate_range("2022-01-03", periods=450)
    rets = rng.normal([0.0005, 0.0002, 0.0003], [0.012, 0.008, 0.010], size=(450, 3))
    df = pd.DataFrame(rets, index=dates, columns=["A.JK", "B.JK", "C.JK"])
    return df.iloc[:300], df.iloc[300:]


def test_run_backtest_buy_and_hold_drift(synthetic_is_oos):
    is_df, oos_df = synthetic_is_oos
    weights = {"A.JK": 0.5, "B.JK": 0.3, "C.JK": 0.2}
    bt = Backtester(is_df, oos_df)
    res = bt.run_backtest(weights, "Test Portfolio")

    w_vec = np.array([0.5, 0.3, 0.2])
    expected_final = float((np.exp(oos_df.cumsum()).iloc[-1].to_numpy() @ w_vec) - 1.0)
    assert pytest.approx(res["cumulative_returns"].iloc[-1], rel=1e-9) == expected_final
    assert res["realized_mdd"] <= 0.0


def test_run_backtest_respects_risk_free_rate(synthetic_is_oos):
    is_df, oos_df = synthetic_is_oos
    weights = {"A.JK": 0.4, "B.JK": 0.4, "C.JK": 0.2}
    bt = Backtester(is_df, oos_df)

    res_low_rf = bt.run_backtest(weights, "Low RF", risk_free_rate=0.0)
    res_high_rf = bt.run_backtest(weights, "High RF", risk_free_rate=0.10)
    assert res_low_rf["realized_sharpe"] > res_high_rf["realized_sharpe"]


def test_walk_forward_quarterly_and_transaction_cost(synthetic_is_oos):
    is_df, oos_df = synthetic_is_oos
    bt = Backtester(is_df, oos_df)

    wf_zero = bt.run_walk_forward(
        lookback_days=TRADING_DAYS_PER_YEAR,
        rebalance_freq=63,
        strategy="max_sharpe",
        cov_method="ledoit_wolf",
        transaction_cost=0.0,
    )
    wf_cost = bt.run_walk_forward(
        lookback_days=TRADING_DAYS_PER_YEAR,
        rebalance_freq=63,
        strategy="max_sharpe",
        cov_method="ledoit_wolf",
        transaction_cost=0.01,
    )

    assert wf_zero["rebalances"] == 3  # ceil(150 / 63) = 3
    assert len(wf_zero["cumulative_returns"]) == len(oos_df)
    assert wf_zero["avg_turnover"] > 0.0
    assert wf_cost["cumulative_returns"].iloc[-1] < wf_zero["cumulative_returns"].iloc[-1]


def test_benchmark_performance_shape_and_values(synthetic_is_oos):
    is_df, oos_df = synthetic_is_oos
    bt = Backtester(is_df, oos_df)
    bench = bt.get_benchmark_performance()
    assert len(bench) == len(oos_df)
    expected = float((np.exp(oos_df.cumsum()).iloc[-1].mean()) - 1.0)
    assert pytest.approx(bench.iloc[-1], rel=1e-9) == expected
