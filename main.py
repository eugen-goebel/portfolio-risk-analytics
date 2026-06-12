"""Command line entry point for ingestion and quick metric checks.

Examples:
    uv run main.py ingest SPY AAPL
    uv run main.py ingest --demo demo-a demo-b
    uv run main.py ingest-intraday SPY --interval 1h
    uv run main.py ingest-fx USD GBP
    uv run main.py metrics SPY
    uv run main.py benchmark AAPL --benchmark SPY
    uv run main.py drift SPY
    uv run main.py var-test SPY --window 250 --confidence 0.95
    uv run main.py simulate SPY --horizon 252 --paths 2000 --method bootstrap --seed 7
    uv run main.py report SPY --output spy-factsheet.pdf
"""

import argparse
import sys

from db.database import SessionLocal, init_db


def cmd_ingest(symbols: list[str], demo: bool) -> int:
    from ingestion.base import ProviderError
    from ingestion.demo import generate_demo_bars
    from ingestion.store import store_bars
    from ingestion.yahoo import fetch_daily

    init_db()
    db = SessionLocal()
    try:
        for symbol in symbols:
            if demo:
                bars = generate_demo_bars(symbol)
            else:
                try:
                    bars = fetch_daily(symbol)
                except ProviderError as exc:
                    print(f"{symbol}: {exc}")
                    return 1
            inserted = store_bars(db, symbol, bars)
            print(f"{symbol}: {inserted} new rows ({len(bars)} fetched)")
    finally:
        db.close()
    return 0


def cmd_ingest_intraday(symbols: list[str], interval: str) -> int:
    from ingestion.base import ProviderError
    from ingestion.store import store_intraday_bars
    from ingestion.yahoo import fetch_intraday

    init_db()
    db = SessionLocal()
    try:
        for symbol in symbols:
            try:
                bars = fetch_intraday(symbol, interval)
            except ProviderError as exc:
                print(f"{symbol}: {exc}")
                return 1
            inserted = store_intraday_bars(db, symbol, bars)
            print(f"{symbol}: {inserted} new rows ({len(bars)} fetched)")
    finally:
        db.close()
    return 0


def cmd_ingest_fx(currencies: list[str]) -> int:
    from ingestion.base import ProviderError
    from ingestion.ecb import fetch_daily_fx
    from ingestion.store import store_bars

    init_db()
    db = SessionLocal()
    try:
        for currency in currencies:
            currency = currency.upper()
            try:
                bars = fetch_daily_fx(currency)
            except ProviderError as exc:
                print(f"{currency}: {exc}")
                return 1
            symbol = f"EUR{currency}"
            name = f"ECB reference rate {currency} per EUR"
            inserted = store_bars(db, symbol, bars, name)
            print(f"{symbol}: {inserted} new rows ({len(bars)} fetched)")
    finally:
        db.close()
    return 0


def cmd_metrics(symbol: str, risk_free_rate: float) -> int:
    from analytics.loader import load_close_series
    from analytics.metrics import summarize

    init_db()
    db = SessionLocal()
    try:
        try:
            series = load_close_series(db, symbol)
        except LookupError as exc:
            print(exc)
            return 1
        result = summarize(symbol, series, risk_free_rate)
    finally:
        db.close()

    print(f"Symbol:                {result.symbol}")
    print(f"Observations:          {result.observations}")
    print(f"Total return:          {result.total_return_pct:+.2f}%")
    print(f"Annualized volatility: {result.annualized_volatility_pct:.2f}%")
    print(f"Sharpe ratio:          {result.sharpe_ratio:.3f}")
    print(f"Max drawdown:          {result.max_drawdown_pct:.2f}%")
    return 0


def cmd_benchmark(symbol: str, benchmark: str, risk_free_rate: float) -> int:
    from analytics.loader import load_close_series
    from analytics.metrics import compare_to_benchmark

    init_db()
    db = SessionLocal()
    try:
        try:
            series = load_close_series(db, symbol)
            benchmark_series = load_close_series(db, benchmark)
            report = compare_to_benchmark(
                symbol, series, benchmark, benchmark_series, risk_free_rate
            )
        except (LookupError, ValueError) as exc:
            print(exc)
            return 1
    finally:
        db.close()

    print(f"Symbol:            {report.symbol}")
    print(f"Benchmark:         {report.benchmark}")
    print(f"Observations:      {report.observations}")
    print(f"Beta:              {report.beta:.3f}")
    print(f"Alpha:             {report.alpha_pct:+.2f}%")
    print(f"Tracking error:    {report.tracking_error_pct:.2f}%")
    print(f"Information ratio: {report.information_ratio:.3f}")
    return 0


def cmd_forecast(symbol: str, test_size: int) -> int:
    from analytics.forecast import evaluate_models
    from analytics.loader import load_close_series
    from analytics.metrics import daily_returns

    init_db()
    db = SessionLocal()
    try:
        try:
            series = load_close_series(db, symbol)
            report = evaluate_models(symbol, daily_returns(series), test_size)
        except (LookupError, ValueError) as exc:
            print(exc)
            return 1
    finally:
        db.close()

    print(f"Walk-forward test over the last {report.test_observations} trading days\n")
    print(f"{'model':<10} {'MAE %':>8} {'RMSE %':>8} {'next-day vol (ann.) %':>22}")
    for score in report.scores:
        nd = report.next_day_volatility_pct[score.model]
        print(f"{score.model:<10} {score.mae_pct:>8.4f} {score.rmse_pct:>8.4f} {nd:>22.2f}")
    print(f"\nLowest RMSE: {report.best_model}")
    return 0


def cmd_drift(symbol: str, reference_size: int, recent_size: int) -> int:
    from analytics.drift import evaluate_drift
    from analytics.loader import load_close_series
    from analytics.metrics import daily_returns

    init_db()
    db = SessionLocal()
    try:
        try:
            series = load_close_series(db, symbol)
            report = evaluate_drift(symbol, daily_returns(series), reference_size, recent_size)
        except (LookupError, ValueError) as exc:
            print(exc)
            return 1
    finally:
        db.close()

    print(f"Symbol:           {report.symbol}")
    print(f"Reference window: {report.reference_size} observations")
    print(f"Recent window:    {report.recent_size} observations")
    print(f"PSI:              {report.psi:.4f}")
    print(f"KS statistic:     {report.ks:.4f}")
    print(f"Mean shift:       {report.mean_shift:+.4f}% daily")
    print(f"Volatility ratio: {report.volatility_ratio:.4f}")
    print(f"Drift detected:   {'yes' if report.drift_detected else 'no'}")
    return 0


def cmd_var_test(symbol: str, window: int, confidence: float) -> int:
    from analytics.loader import load_close_series
    from analytics.metrics import daily_returns
    from analytics.var_validation import validate_var

    init_db()
    db = SessionLocal()
    try:
        try:
            series = load_close_series(db, symbol)
            report = validate_var(symbol, daily_returns(series), window, confidence)
        except (LookupError, ValueError) as exc:
            print(exc)
            return 1
    finally:
        db.close()

    print(f"Symbol:            {report.symbol}")
    print(f"Confidence:        {report.confidence * 100:.0f}%")
    print(f"Window:            {report.window} observations")
    print(f"Out-of-sample:     {report.observations} observations")
    print(f"Expected breaches: {report.expected_breaches:.1f}")
    print(f"Actual breaches:   {report.actual_breaches}")
    print(f"Breach rate:       {report.breach_rate_pct:.2f}%")
    print(f"Kupiec LR:         {report.kupiec_statistic:.4f}")
    print(f"p-value:           {report.p_value:.4f}")
    print(f"VaR model rejected at the 5% level: {'yes' if report.model_rejected else 'no'}")
    return 0


def cmd_simulate(symbol: str, horizon: int, paths: int, method: str, seed: int | None) -> int:
    from analytics.loader import load_close_series
    from analytics.metrics import daily_returns
    from analytics.montecarlo import run_monte_carlo

    init_db()
    db = SessionLocal()
    try:
        try:
            series = load_close_series(db, symbol)
            report = run_monte_carlo(symbol, daily_returns(series), horizon, paths, method, seed)
        except (LookupError, ValueError) as exc:
            print(exc)
            return 1
    finally:
        db.close()

    print(f"Symbol:               {report.symbol}")
    print(f"Method:               {report.method}")
    print(f"Horizon:              {report.horizon_days} trading days")
    print(f"Paths:                {report.n_paths}")
    print(f"Start value:          {report.start_value:.2f}")
    for name in ("p5", "p25", "p50", "p75", "p95"):
        label = f"Final value {name}:"
        print(f"{label:<22}{report.percentiles[name]:.2f}")
    print(f"Probability of loss:  {report.prob_loss * 100:.2f}%")
    print(f"Expected final value: {report.expected_final:.2f}")
    return 0


def cmd_report(symbol: str, output: str | None, risk_free_rate: float) -> int:
    from analytics.loader import load_close_series
    from reporting.factsheet import generate_factsheet

    init_db()
    db = SessionLocal()
    try:
        try:
            series = load_close_series(db, symbol)
        except LookupError as exc:
            print(exc)
            return 1
        path = generate_factsheet(
            symbol, series, output or f"{symbol}-factsheet.pdf", risk_free_rate
        )
    finally:
        db.close()

    print(f"Factsheet written to {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Portfolio risk analytics")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Download and store daily prices")
    p_ingest.add_argument("symbols", nargs="+", help="Symbols, e.g. SPY AAPL")
    p_ingest.add_argument("--demo", action="store_true", help="Generate offline demo data")

    p_intraday = sub.add_parser("ingest-intraday", help="Download and store intraday prices")
    p_intraday.add_argument("symbols", nargs="+", help="Symbols, e.g. SPY AAPL")
    p_intraday.add_argument(
        "--interval", default="1h", choices=["15m", "30m", "1h"], help="Bar size, default 1h"
    )

    p_fx = sub.add_parser("ingest-fx", help="Download and store ECB reference exchange rates")
    p_fx.add_argument("currencies", nargs="+", help="Currency codes, e.g. USD GBP")

    p_metrics = sub.add_parser("metrics", help="Show risk metrics for a stored symbol")
    p_metrics.add_argument("symbol")
    p_metrics.add_argument("--risk-free-rate", type=float, default=0.0)

    p_benchmark = sub.add_parser("benchmark", help="Compare a stored symbol against a benchmark")
    p_benchmark.add_argument("symbol")
    p_benchmark.add_argument("--benchmark", default="SPY")
    p_benchmark.add_argument("--risk-free-rate", type=float, default=0.0)

    p_forecast = sub.add_parser("forecast", help="Compare volatility forecasters for a symbol")
    p_forecast.add_argument("symbol")
    p_forecast.add_argument("--test-size", type=int, default=250)

    p_drift = sub.add_parser("drift", help="Check a symbol for return distribution drift")
    p_drift.add_argument("symbol")
    p_drift.add_argument("--reference-size", type=int, default=500)
    p_drift.add_argument("--recent-size", type=int, default=60)

    p_var = sub.add_parser("var-test", help="Backtest the VaR model with the Kupiec test")
    p_var.add_argument("symbol")
    p_var.add_argument("--window", type=int, default=250)
    p_var.add_argument("--confidence", type=float, default=0.95)

    p_simulate = sub.add_parser("simulate", help="Monte Carlo simulation of future value paths")
    p_simulate.add_argument("symbol")
    p_simulate.add_argument("--horizon", type=int, default=252)
    p_simulate.add_argument("--paths", type=int, default=2000)
    p_simulate.add_argument("--method", choices=["bootstrap", "normal"], default="bootstrap")
    p_simulate.add_argument("--seed", type=int, default=None)

    p_report = sub.add_parser("report", help="Write a one-page PDF factsheet for a stored symbol")
    p_report.add_argument("symbol")
    p_report.add_argument(
        "--output", default=None, help="Output path, default {symbol}-factsheet.pdf"
    )
    p_report.add_argument("--risk-free-rate", type=float, default=0.0)

    args = parser.parse_args()
    if args.command == "ingest":
        return cmd_ingest(args.symbols, args.demo)
    if args.command == "ingest-intraday":
        return cmd_ingest_intraday(args.symbols, args.interval)
    if args.command == "ingest-fx":
        return cmd_ingest_fx(args.currencies)
    if args.command == "metrics":
        return cmd_metrics(args.symbol, args.risk_free_rate)
    if args.command == "benchmark":
        return cmd_benchmark(args.symbol, args.benchmark, args.risk_free_rate)
    if args.command == "forecast":
        return cmd_forecast(args.symbol, args.test_size)
    if args.command == "var-test":
        return cmd_var_test(args.symbol, args.window, args.confidence)
    if args.command == "simulate":
        return cmd_simulate(args.symbol, args.horizon, args.paths, args.method, args.seed)
    if args.command == "report":
        return cmd_report(args.symbol, args.output, args.risk_free_rate)
    return cmd_drift(args.symbol, args.reference_size, args.recent_size)


if __name__ == "__main__":
    sys.exit(main())
