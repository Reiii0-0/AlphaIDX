"""
Non-interactive CLI entry point for AlphaIDX quantitative pipelines.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.data.fetcher import StockDataFetcher
from src.portfolio.advanced_models import RiskParityOptimizer
from src.portfolio.backtester import Backtester
from src.portfolio.garch_model import GARCHVolatilityModel
from src.portfolio.metrics import (
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    DEFAULT_TICKERS,
    N_SIMULATION_PATHS,
    RISK_FREE_RATE_ANNUAL,
    TRADING_DAYS_PER_YEAR,
)
from src.portfolio.monte_carlo import MonteCarloSimulator
from src.portfolio.optimizer import EfficientFrontier, ledoit_wolf_shrinkage

logger = logging.getLogger(__name__)


def _parse_tickers(raw: str) -> list[str]:
    tickers = [t.strip() for t in raw.split(",") if t.strip()]
    if len(tickers) < 2:
        raise ValueError(
            "Need at least 2 comma-separated tickers.\n"
            "  Example: python -m src.cli run --tickers BBCA.JK,BBRI.JK,TLKM.JK"
        )
    return tickers


def _parse_iso_date(raw: str, flag_name: str) -> date:
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(
            f"Invalid date for {flag_name}: '{raw}' (expected YYYY-MM-DD).\n"
            "  Example: python -m src.cli run --start 2019-01-01 --end 2024-12-31"
        ) from exc


def _validate_args(args: argparse.Namespace) -> list[str]:
    tickers = _parse_tickers(args.tickers)
    start_d = _parse_iso_date(args.start, "--start")
    end_d = _parse_iso_date(args.end, "--end")
    if start_d >= end_d:
        raise ValueError(
            f"--start ({args.start}) must be earlier than --end ({args.end}).\n"
            "  Example: python -m src.cli run --start 2019-01-01 --end 2024-12-31"
        )
    if args.transaction_cost < 0:
        raise ValueError("--transaction-cost must be >= 0.")
    if args.rf < 0:
        raise ValueError("--rf must be >= 0.")

    if args.command == "run":
        if bool(args.oos_start) != bool(args.oos_end):
            raise ValueError(
                "Provide both --oos-start and --oos-end together for OOS backtesting.\n"
                "  Example: python -m src.cli run --oos-start 2025-01-01 --oos-end 2026-05-25"
            )
        if args.n_sims < 1:
            raise ValueError("--n-sims must be >= 1.")

    if getattr(args, "oos_start", None) and getattr(args, "oos_end", None):
        oos_start_d = _parse_iso_date(args.oos_start, "--oos-start")
        oos_end_d = _parse_iso_date(args.oos_end, "--oos-end")
        if oos_start_d >= oos_end_d:
            raise ValueError(
                f"--oos-start ({args.oos_start}) must be earlier than --oos-end ({args.oos_end})."
            )
        if oos_start_d <= end_d:
            raise ValueError(
                f"Temporal split leak: --oos-start ({args.oos_start}) must be after --end ({args.end}).\n"
                "  Example: python -m src.cli backtest --end 2024-12-31 --oos-start 2025-01-01 --oos-end 2026-05-25"
            )

    if getattr(args, "rebalance_freq", 63) < 1:
        raise ValueError("--rebalance-freq must be >= 1.")
    if getattr(args, "lookback_days", 252) < 20:
        raise ValueError("--lookback-days must be >= 20.")

    return tickers


def _estimate_cov(
    log_returns: pd.DataFrame,
    cov_method: str,
    use_garch: bool,
) -> np.ndarray:
    if use_garch:
        vols = []
        for col in log_returns.columns:
            garch = GARCHVolatilityModel(log_returns[col])
            garch.fit()
            vols.append(garch.forecast_volatility(horizon=TRADING_DAYS_PER_YEAR))
        corr = log_returns.corr().to_numpy(dtype=float)
        diag = np.diag(vols)
        return diag @ corr @ diag
    if cov_method == "ledoit_wolf":
        return ledoit_wolf_shrinkage(log_returns)
    return log_returns.cov().to_numpy(dtype=float) * TRADING_DAYS_PER_YEAR


def _clean_portfolio(res: dict[str, Any]) -> dict[str, Any]:
    return {
        "return": round(float(res["return"]), 6),
        "volatility": round(float(res["volatility"]), 6),
        "sharpe": round(float(res["sharpe"]), 4),
        "weights": {k: round(float(v), 6) for k, v in res["weights"].items()},
    }


def _clean_backtest(res: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "label": res["label"],
        "realized_return": round(float(res["realized_return"]), 6),
        "realized_volatility": round(float(res["realized_volatility"]), 6),
        "realized_sharpe": round(float(res["realized_sharpe"]), 4),
        "realized_mdd": round(float(res["realized_mdd"]), 6),
        "final_cumulative_return": round(float(res["cumulative_returns"].iloc[-1]), 6),
    }
    if "rebalances" in res:
        payload["rebalances"] = int(res["rebalances"])
        payload["avg_turnover"] = round(float(res["avg_turnover"]), 6)
    return payload


def _fetch_is_and_oos(
    tickers: list[str],
    start: str,
    end: str,
    oos_start: str,
    oos_end: str,
    cache_dir: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    is_fetcher = StockDataFetcher(tickers, start, end, cache_dir=cache_dir)
    is_prices = is_fetcher.fetch()
    is_log_rets = is_fetcher.to_log_returns(is_prices, min_trading_days=60)

    oos_fetcher = StockDataFetcher(
        list(is_log_rets.columns), oos_start, oos_end, cache_dir=cache_dir
    )
    oos_prices = oos_fetcher.fetch()
    oos_log_rets = oos_fetcher.to_log_returns(oos_prices, min_trading_days=20)
    # Enforce strict non-overlapping temporal split.
    oos_log_rets = oos_log_rets[oos_log_rets.index > is_log_rets.index.max()]
    if oos_log_rets.empty:
        raise ValueError("No non-overlapping OOS trading days found after IS end date.")
    return is_log_rets, oos_log_rets


def _run_pipeline(args: argparse.Namespace, tickers: list[str]) -> dict[str, Any]:
    t0 = time.perf_counter()
    fetcher = StockDataFetcher(tickers, args.start, args.end, cache_dir=args.cache_dir)
    prices = fetcher.fetch()
    log_rets = fetcher.to_log_returns(prices, min_trading_days=60)
    active_tickers = list(log_rets.columns)

    cov_matrix = _estimate_cov(log_rets, args.cov_method, args.garch)
    mean_returns = log_rets.mean().to_numpy(dtype=float) * TRADING_DAYS_PER_YEAR

    ef = EfficientFrontier(
        log_rets,
        allow_short=args.allow_short,
        risk_free_rate=args.rf,
        cov_method=args.cov_method,
        mean_returns=mean_returns,
        cov_matrix=cov_matrix,
    )
    max_sharpe = ef.optimize_max_sharpe()
    min_var = ef.optimize_min_variance()

    rp = RiskParityOptimizer(cov_matrix, active_tickers).optimize()
    ms_weights = np.array([max_sharpe["weights"][t] for t in active_tickers], dtype=float)
    sim = MonteCarloSimulator(
        log_rets,
        ms_weights,
        mean_returns_annual=mean_returns,
        cov_matrix_annual=cov_matrix,
    )
    paths = sim.simulate_price_paths(
        n_simulations=args.n_sims,
        n_days=TRADING_DAYS_PER_YEAR,
        antithetic=True,
    )
    var_cvar = sim.compute_var_cvar(paths)

    result: dict[str, Any] = {
        "command": "run",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "tickers": active_tickers,
            "start": args.start,
            "end": args.end,
            "risk_free_rate": args.rf,
            "cov_method": args.cov_method,
            "garch": args.garch,
            "allow_short": args.allow_short,
            "n_sims": args.n_sims,
            "trading_days": int(len(log_rets)),
        },
        "max_sharpe": _clean_portfolio(max_sharpe),
        "min_variance": _clean_portfolio(min_var),
        "risk_parity": {
            "volatility": round(float(rp["volatility"]), 6),
            "weights": {k: round(float(v), 6) for k, v in rp["weights"].items()},
        },
        "monte_carlo_1y": {k: round(float(v), 6) for k, v in var_cvar.items()},
    }

    if args.oos_start and args.oos_end:
        _, oos_log_rets = _fetch_is_and_oos(
            active_tickers,
            args.start,
            args.end,
            args.oos_start,
            args.oos_end,
            args.cache_dir,
        )
        bt = Backtester(log_rets, oos_log_rets)
        wf_ms = bt.run_walk_forward(
            rebalance_freq=args.rebalance_freq,
            strategy="max_sharpe",
            cov_method=args.cov_method,
            transaction_cost=args.transaction_cost,
            allow_short=args.allow_short,
            risk_free_rate=args.rf,
        )
        wf_mv = bt.run_walk_forward(
            rebalance_freq=args.rebalance_freq,
            strategy="min_variance",
            cov_method=args.cov_method,
            transaction_cost=args.transaction_cost,
            allow_short=args.allow_short,
            risk_free_rate=args.rf,
        )
        result["oos_backtest"] = {
            "oos_start": args.oos_start,
            "oos_end": args.oos_end,
            "static_max_sharpe": _clean_backtest(
                bt.run_backtest(max_sharpe["weights"], "Static Max Sharpe", args.rf)
            ),
            "static_min_variance": _clean_backtest(
                bt.run_backtest(min_var["weights"], "Static Min Variance", args.rf)
            ),
            "walk_forward_max_sharpe": _clean_backtest(wf_ms),
            "walk_forward_min_variance": _clean_backtest(wf_mv),
            "benchmark_equal_weight_cum_return": round(
                float(bt.get_benchmark_performance().iloc[-1]), 6
            ),
        }

    result["elapsed_seconds"] = round(time.perf_counter() - t0, 3)
    return result


def _run_backtest_command(args: argparse.Namespace, tickers: list[str]) -> dict[str, Any]:
    t0 = time.perf_counter()
    is_log_rets, oos_log_rets = _fetch_is_and_oos(
        tickers,
        args.start,
        args.end,
        args.oos_start,
        args.oos_end,
        args.cache_dir,
    )
    ef = EfficientFrontier(
        is_log_rets,
        allow_short=args.allow_short,
        risk_free_rate=args.rf,
        cov_method=args.cov_method,
    )
    max_sharpe = ef.optimize_max_sharpe()
    min_var = ef.optimize_min_variance()

    bt = Backtester(is_log_rets, oos_log_rets)
    static_ms = bt.run_backtest(max_sharpe["weights"], "Static Max Sharpe", args.rf)
    static_mv = bt.run_backtest(min_var["weights"], "Static Min Variance", args.rf)
    wf_ms = bt.run_walk_forward(
        lookback_days=args.lookback_days,
        rebalance_freq=args.rebalance_freq,
        strategy="max_sharpe",
        cov_method=args.cov_method,
        transaction_cost=args.transaction_cost,
        allow_short=args.allow_short,
        risk_free_rate=args.rf,
    )
    wf_mv = bt.run_walk_forward(
        lookback_days=args.lookback_days,
        rebalance_freq=args.rebalance_freq,
        strategy="min_variance",
        cov_method=args.cov_method,
        transaction_cost=args.transaction_cost,
        allow_short=args.allow_short,
        risk_free_rate=args.rf,
    )
    bench = bt.get_benchmark_performance()

    return {
        "command": "backtest",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "tickers": list(is_log_rets.columns),
            "is_start": args.start,
            "is_end": args.end,
            "oos_start": args.oos_start,
            "oos_end": args.oos_end,
            "lookback_days": args.lookback_days,
            "rebalance_freq": args.rebalance_freq,
            "cov_method": args.cov_method,
            "transaction_cost": args.transaction_cost,
            "risk_free_rate": args.rf,
        },
        "results": {
            "static_max_sharpe": _clean_backtest(static_ms),
            "static_min_variance": _clean_backtest(static_mv),
            "walk_forward_max_sharpe": _clean_backtest(wf_ms),
            "walk_forward_min_variance": _clean_backtest(wf_mv),
            "benchmark_equal_weight_cum_return": round(float(bench.iloc[-1]), 6),
        },
        "elapsed_seconds": round(time.perf_counter() - t0, 3),
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# AlphaIDX Pipeline Summary ({payload['command']})",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Elapsed: `{payload['elapsed_seconds']}s`",
        f"- Tickers ({len(payload['config']['tickers'])}): `{', '.join(payload['config']['tickers'])}`",
        "",
    ]
    if payload["command"] == "run":
        ms = payload["max_sharpe"]
        mv = payload["min_variance"]
        mc = payload["monte_carlo_1y"]
        lines += [
            "## Optimization",
            "",
            "| Portfolio | Return | Volatility | Sharpe |",
            "|---|---|---|---|",
            f"| Max Sharpe | {ms['return']:.2%} | {ms['volatility']:.2%} | {ms['sharpe']:.2f} |",
            f"| Min Variance | {mv['return']:.2%} | {mv['volatility']:.2%} | {mv['sharpe']:.2f} |",
            "",
            "## Monte Carlo (1Y)",
            "",
            f"- Median final (base 100): `{mc['median_final']:.2f}`",
            f"- 95% VaR: `{mc['var_pct']:.2%}` | 95% CVaR: `{mc['cvar_pct']:.2%}`",
            "",
        ]
    if "oos_backtest" in payload or payload["command"] == "backtest":
        bt_block = payload.get("oos_backtest") or payload["results"]
        lines += [
            "## Out-of-Sample Backtest",
            "",
            "| Strategy | Realized Ret | Realized Vol | Sharpe | Max Drawdown | Cum Return |",
            "|---|---|---|---|---|---|",
        ]
        for key, item in bt_block.items():
            if isinstance(item, dict) and "realized_return" in item:
                lines.append(
                    f"| {item['label']} | {item['realized_return']:.2%} | {item['realized_volatility']:.2%} | "
                    f"{item['realized_sharpe']:.2f} | {item['realized_mdd']:.2%} | {item['final_cumulative_return']:.2%} |"
                )
        lines.append(
            f"\n- Equal-weight benchmark cumulative return: `{bt_block['benchmark_equal_weight_cum_return']:.2%}`"
        )
    return "\n".join(lines) + "\n"


def _write_artifacts(output_dir: Path, payload: dict[str, Any]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "summary.json"
    md_path = output_dir / "summary.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(payload), encoding="utf-8")
    return {"summary_json": str(json_path), "summary_md": str(md_path)}


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--tickers",
        default=",".join(DEFAULT_TICKERS),
        help="Comma-separated Yahoo Finance tickers (default: 10 IDX blue-chips).",
    )
    common.add_argument(
        "--start", default=DEFAULT_START_DATE, help="In-sample start date (YYYY-MM-DD)."
    )
    common.add_argument("--end", default=DEFAULT_END_DATE, help="In-sample end date (YYYY-MM-DD).")
    common.add_argument(
        "--rf", type=float, default=RISK_FREE_RATE_ANNUAL, help="Annual risk-free rate."
    )
    common.add_argument(
        "--cov-method",
        choices=["ledoit_wolf", "sample"],
        default="ledoit_wolf",
        help="Covariance estimator (default: ledoit_wolf).",
    )
    common.add_argument(
        "--allow-short", action="store_true", help="Allow short weights (-0.5 to 1.0)."
    )
    common.add_argument(
        "--transaction-cost",
        type=float,
        default=0.001,
        help="Proportional turnover transaction cost (default: 0.001).",
    )
    common.add_argument(
        "--cache-dir", type=Path, default=None, help="Optional CSV cache directory."
    )
    common.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save summary.json and summary.md.",
    )
    common.add_argument(
        "--format", choices=["text", "json"], default="text", help="Stdout output format."
    )
    common.add_argument(
        "--dry-run", action="store_true", help="Validate config and print plan without running."
    )

    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="AlphaIDX non-interactive quantitative portfolio pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m src.cli run --start 2019-01-01 --end 2024-12-31 --output-dir reports/\n"
            "  python -m src.cli run --tickers BBCA.JK,BBRI.JK,TLKM.JK --garch --format json\n"
            "  python -m src.cli backtest --end 2024-12-31 --oos-start 2025-01-01 --oos-end 2026-05-25\n"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_p = subparsers.add_parser(
        "run",
        parents=[common],
        help="Run optimization, Risk Parity, Monte Carlo, and optional OOS backtest.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m src.cli run --dry-run\n"
            "  python -m src.cli run --cov-method ledoit_wolf --n-sims 2000 --format json\n"
            "  python -m src.cli run --oos-start 2025-01-01 --oos-end 2026-05-25 --output-dir reports/\n"
        ),
    )
    run_p.add_argument(
        "--n-sims", type=int, default=N_SIMULATION_PATHS, help="Monte Carlo simulation paths."
    )
    run_p.add_argument(
        "--garch", action="store_true", help="Forecast volatilities with GARCH(1,1)."
    )
    run_p.add_argument("--oos-start", default=None, help="Optional OOS start date (YYYY-MM-DD).")
    run_p.add_argument("--oos-end", default=None, help="Optional OOS end date (YYYY-MM-DD).")
    run_p.add_argument(
        "--rebalance-freq", type=int, default=63, help="Walk-forward rebalance interval in days."
    )

    bt_p = subparsers.add_parser(
        "backtest",
        parents=[common],
        help="Compare static vs rolling walk-forward portfolios out-of-sample.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m src.cli backtest --oos-start 2025-01-01 --oos-end 2026-05-25\n"
            "  python -m src.cli backtest --rebalance-freq 63 --cov-method ledoit_wolf --format json\n"
        ),
    )
    bt_p.add_argument("--oos-start", default="2025-01-01", help="OOS start date (YYYY-MM-DD).")
    bt_p.add_argument("--oos-end", default="2026-05-25", help="OOS end date (YYYY-MM-DD).")
    bt_p.add_argument(
        "--lookback-days", type=int, default=TRADING_DAYS_PER_YEAR, help="Rolling lookback window."
    )
    bt_p.add_argument(
        "--rebalance-freq", type=int, default=63, help="Rebalance interval in trading days."
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        stream=sys.stderr,
        level=logging.WARNING if args.format == "json" else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        tickers = _validate_args(args)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        plan = {
            "dry_run": True,
            "command": args.command,
            "tickers": sorted(tickers),
            "start": args.start,
            "end": args.end,
            "cov_method": args.cov_method,
            "risk_free_rate": args.rf,
            "output_dir": str(args.output_dir) if args.output_dir else None,
        }
        print(json.dumps(plan, indent=2) if args.format == "json" else f"Dry run valid: {plan}")
        return 0

    try:
        payload = (
            _run_pipeline(args, tickers)
            if args.command == "run"
            else _run_backtest_command(args, tickers)
        )
    except Exception as exc:
        print(f"Execution error: {exc}", file=sys.stderr)
        return 1

    if args.output_dir is not None:
        payload["artifacts"] = _write_artifacts(args.output_dir, payload)

    if args.format == "json":
        print(json.dumps(payload, indent=2))
    else:
        print(_render_markdown(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
