import numpy as np
import pandas as pd
import pytest

from src.live_dashboard import compute_dashboard_analytics


@pytest.fixture
def dashboard_log_returns():
    rng = np.random.default_rng(77)
    n_days = 300
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    # Columns sorted alphabetically, matching StockDataFetcher output.
    rets = rng.normal([0.0002, 0.0004, 0.0003], [0.011, 0.015, 0.012], size=(n_days, 3))
    return pd.DataFrame(rets, index=dates, columns=["ASII.JK", "BBCA.JK", "TLKM.JK"])


def test_dashboard_pipeline_with_bl_and_ledoit_wolf(dashboard_log_returns):
    bl_view = {
        "view_type": "Relative",
        "asset_a": "TLKM.JK",
        "asset_b": "ASII.JK",
        "expected_return": 0.15,
        "tau": 0.05,
    }
    res = compute_dashboard_analytics(
        log_returns=dashboard_log_returns,
        rf_rate=0.0625,
        cov_method="ledoit_wolf",
        tc_rate=0.001,
        initial_weights_mode="equal",
        use_garch=False,
        bl_view=bl_view,
    )

    assert res["active_tickers"] == ["ASII.JK", "BBCA.JK", "TLKM.JK"]
    # Relative BL view boosts TLKM.JK expected return above ASII.JK.
    assert res["mean_returns"][2] > res["mean_returns"][0]
    rc_vals = np.array(list(res["risk_parity"]["risk_contributions"].values()))
    assert np.allclose(rc_vals, [1 / 3, 1 / 3, 1 / 3], atol=1e-2)
    assert res["paths"].shape == (500, 253)


def test_dashboard_pipeline_with_garch_propagation(dashboard_log_returns):
    res_sample = compute_dashboard_analytics(
        log_returns=dashboard_log_returns,
        cov_method="sample",
        use_garch=False,
    )
    res_garch = compute_dashboard_analytics(
        log_returns=dashboard_log_returns,
        use_garch=True,
        max_weight=0.5,
        dist="t",
    )
    assert res_garch["cov_matrix"].shape == (3, 3)
    assert np.allclose(res_garch["cov_matrix"], res_garch["cov_matrix"].T)
    assert np.all(np.linalg.eigvalsh(res_garch["cov_matrix"]) > 0)
    assert not np.allclose(res_garch["cov_matrix"], res_sample["cov_matrix"])
    assert not np.allclose(res_garch["paths"], res_sample["paths"])
    assert all(w <= 0.5 + 1e-5 for w in res_garch["max_sharpe"]["weights"].values())
    assert "min_cvar" in res_garch
    assert "hrp" in res_garch
    assert res_garch["var_cvar"]["cvar_pct"] >= res_garch["var_cvar"]["var_pct"]
    assert (
        res_garch["var_cvar"]["percentile_5"]
        <= res_garch["var_cvar"]["median_final"]
        <= res_garch["var_cvar"]["percentile_95"]
    )


def test_dashboard_pipeline_rejects_invalid_bl_view(dashboard_log_returns):
    with pytest.raises(ValueError, match="active asset_b"):
        compute_dashboard_analytics(
            log_returns=dashboard_log_returns,
            bl_view={
                "view_type": "Relative",
                "asset_a": "TLKM.JK",
                "asset_b": "TLKM.JK",
                "expected_return": 0.05,
            },
        )
