import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.cli import main


def _mock_yf_download(tickers, start, end, **_kwargs):
    dates = pd.bdate_range(start, end)
    rng = np.random.default_rng(101 if start.startswith("2019") else 202)
    rets = rng.normal(
        [0.0004, 0.0002, 0.0003][: len(tickers)], 0.012, size=(len(dates), len(tickers))
    )
    prices = 1000.0 * np.exp(np.cumsum(rets, axis=0))
    cols = pd.MultiIndex.from_product([["Close"], sorted(tickers)])
    return pd.DataFrame(prices, index=dates, columns=cols)


def test_cli_help_includes_examples(capsys):
    for argv in (["--help"], ["run", "--help"], ["backtest", "--help"]):
        with pytest.raises(SystemExit) as exc:
            main(argv)
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "Examples:" in out


def test_cli_dry_run_json(tmp_path, capsys):
    out_dir = tmp_path / "out"
    code = main(
        [
            "run",
            "--tickers",
            "BBCA.JK,BBRI.JK",
            "--dry-run",
            "--format",
            "json",
            "--output-dir",
            str(out_dir),
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["tickers"] == ["BBCA.JK", "BBRI.JK"]
    assert not out_dir.exists()


@pytest.mark.parametrize(
    "argv",
    [
        ["run", "--tickers", "BBCA.JK"],
        ["run", "--start", "2024-01-01", "--end", "2024-01-01"],
        ["run", "--oos-start", "2025-01-01"],
        ["backtest", "--end", "2024-12-31", "--oos-start", "2024-06-01", "--oos-end", "2025-06-01"],
    ],
)
def test_cli_validation_fails_fast(argv, capsys):
    code = main(argv)
    assert code == 2
    err = capsys.readouterr().err
    assert "Error:" in err


def test_cli_run_end_to_end_and_idempotent(tmp_path, capsys):
    cache_dir = tmp_path / "cache"
    out_dir = tmp_path / "reports"
    argv = [
        "run",
        "--tickers",
        "BBCA.JK,TLKM.JK,ASII.JK",
        "--start",
        "2019-01-01",
        "--end",
        "2021-12-31",
        "--oos-start",
        "2022-01-01",
        "--oos-end",
        "2022-12-31",
        "--n-sims",
        "200",
        "--cache-dir",
        str(cache_dir),
        "--output-dir",
        str(out_dir),
        "--format",
        "json",
    ]

    with patch("src.data.fetcher.yf.download", side_effect=_mock_yf_download):
        code1 = main(argv)
        out1 = capsys.readouterr().out
        assert code1 == 0
        payload1 = json.loads(out1)
        assert payload1["command"] == "run"
        assert "max_sharpe" in payload1
        assert "oos_backtest" in payload1
        assert "walk_forward_min_variance" in payload1["oos_backtest"]
        assert (out_dir / "summary.json").exists()
        assert (out_dir / "summary.md").exists()

        # Running twice overwrites artifacts cleanly with the same schema.
        code2 = main(argv)
        out2 = capsys.readouterr().out
        assert code2 == 0
        payload2 = json.loads(out2)
        assert payload2["max_sharpe"] == payload1["max_sharpe"]
        assert payload2["oos_backtest"] == payload1["oos_backtest"]


def test_cli_backtest_command(tmp_path, capsys):
    cache_dir = tmp_path / "cache"
    argv = [
        "backtest",
        "--tickers",
        "BBCA.JK,TLKM.JK,ASII.JK",
        "--start",
        "2019-01-01",
        "--end",
        "2021-12-31",
        "--oos-start",
        "2022-01-01",
        "--oos-end",
        "2022-12-31",
        "--rebalance-freq",
        "63",
        "--cache-dir",
        str(cache_dir),
        "--format",
        "json",
    ]
    with patch("src.data.fetcher.yf.download", side_effect=_mock_yf_download):
        code = main(argv)
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "backtest"
    assert payload["results"]["walk_forward_max_sharpe"]["rebalances"] >= 3
