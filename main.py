"""Command line entry point for ingestion and quick metric checks.

Examples:
    uv run main.py ingest SPY AAPL
    uv run main.py ingest --demo demo-a demo-b
    uv run main.py metrics SPY
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Portfolio risk analytics")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Download and store daily prices")
    p_ingest.add_argument("symbols", nargs="+", help="Stooq symbols, e.g. SPY AAPL")
    p_ingest.add_argument("--demo", action="store_true", help="Generate offline demo data")

    p_metrics = sub.add_parser("metrics", help="Show risk metrics for a stored symbol")
    p_metrics.add_argument("symbol")
    p_metrics.add_argument("--risk-free-rate", type=float, default=0.0)

    args = parser.parse_args()
    if args.command == "ingest":
        return cmd_ingest(args.symbols, args.demo)
    return cmd_metrics(args.symbol, args.risk_free_rate)


if __name__ == "__main__":
    sys.exit(main())
