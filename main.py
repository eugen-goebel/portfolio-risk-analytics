"""Command line entry point for ingestion and quick metric checks.

Examples:
    uv run main.py ingest SPY AAPL
    uv run main.py ingest --demo demo-a demo-b
    uv run main.py ingest-fx USD GBP
    uv run main.py metrics SPY
    uv run main.py drift SPY
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Portfolio risk analytics")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Download and store daily prices")
    p_ingest.add_argument("symbols", nargs="+", help="Symbols, e.g. SPY AAPL")
    p_ingest.add_argument("--demo", action="store_true", help="Generate offline demo data")

    p_fx = sub.add_parser("ingest-fx", help="Download and store ECB reference exchange rates")
    p_fx.add_argument("currencies", nargs="+", help="Currency codes, e.g. USD GBP")

    p_metrics = sub.add_parser("metrics", help="Show risk metrics for a stored symbol")
    p_metrics.add_argument("symbol")
    p_metrics.add_argument("--risk-free-rate", type=float, default=0.0)

    p_forecast = sub.add_parser("forecast", help="Compare volatility forecasters for a symbol")
    p_forecast.add_argument("symbol")
    p_forecast.add_argument("--test-size", type=int, default=250)

    p_drift = sub.add_parser("drift", help="Check a symbol for return distribution drift")
    p_drift.add_argument("symbol")
    p_drift.add_argument("--reference-size", type=int, default=500)
    p_drift.add_argument("--recent-size", type=int, default=60)

    args = parser.parse_args()
    if args.command == "ingest":
        return cmd_ingest(args.symbols, args.demo)
    if args.command == "ingest-fx":
        return cmd_ingest_fx(args.currencies)
    if args.command == "metrics":
        return cmd_metrics(args.symbol, args.risk_free_rate)
    if args.command == "forecast":
        return cmd_forecast(args.symbol, args.test_size)
    return cmd_drift(args.symbol, args.reference_size, args.recent_size)


if __name__ == "__main__":
    sys.exit(main())
