"""Client for the public Yahoo Finance chart endpoint.

The endpoint serves daily candles as JSON without authentication, for
example https://query1.finance.yahoo.com/v8/finance/chart/SPY with
range and interval parameters. A browser-like User-Agent is required.

When the response carries adjusted closes, those replace the raw
closes: risk metrics should be computed on prices that account for
dividends and splits.
"""

from datetime import UTC, datetime
from typing import Any

import httpx

from ingestion.base import PriceBar, ProviderError

BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/"
USER_AGENT = "Mozilla/5.0 (portfolio-risk-analytics)"


def parse_chart_json(payload: dict[str, Any]) -> list[PriceBar]:
    """Turn one chart API response into price bars.

    Yahoo pads its arrays with null for days without data, those rows
    are skipped.
    """
    chart = payload.get("chart") or {}
    if chart.get("error"):
        raise ProviderError(f"Yahoo returned an error: {chart['error']}")
    results = chart.get("result") or []
    if not results:
        raise ProviderError("Yahoo response contained no result")

    result = results[0]
    timestamps = result.get("timestamp") or []
    indicators = result.get("indicators") or {}
    quotes = (indicators.get("quote") or [{}])[0]
    adjclose_series = (indicators.get("adjclose") or [{}])[0].get("adjclose")

    bars: list[PriceBar] = []
    for i, ts in enumerate(timestamps):
        close = quotes.get("close", [None])[i] if i < len(quotes.get("close", [])) else None
        if adjclose_series and i < len(adjclose_series) and adjclose_series[i] is not None:
            close = adjclose_series[i]
        o = _at(quotes.get("open"), i)
        h = _at(quotes.get("high"), i)
        lo = _at(quotes.get("low"), i)
        v = _at(quotes.get("volume"), i)
        if close is None or o is None or h is None or lo is None:
            continue
        bars.append(
            PriceBar(
                day=datetime.fromtimestamp(ts, tz=UTC).date(),
                open=float(o),
                high=float(h),
                low=float(lo),
                close=float(close),
                volume=float(v or 0.0),
            )
        )
    if not bars:
        raise ProviderError("Yahoo response contained no usable bars")
    return bars


def _at(values: list[Any] | None, index: int) -> Any:
    if not values or index >= len(values):
        return None
    return values[index]


def fetch_daily(symbol: str, lookback: str = "5y", timeout: float = 30.0) -> list[PriceBar]:
    """Download daily history for one symbol, e.g. SPY or AAPL."""
    try:
        response = httpx.get(
            BASE_URL + symbol.upper(),
            params={"range": lookback, "interval": "1d"},
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
            follow_redirects=True,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise ProviderError(f"Request to Yahoo failed for {symbol}: {exc}") from exc
    return parse_chart_json(response.json())
