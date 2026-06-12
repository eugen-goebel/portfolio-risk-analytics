"""Client for the public Yahoo Finance chart endpoint.

The endpoint serves daily and intraday candles as JSON without
authentication, for example
https://query1.finance.yahoo.com/v8/finance/chart/SPY with range and
interval parameters. A browser-like User-Agent is required.

When the response carries adjusted closes, those replace the raw
closes: risk metrics should be computed on prices that account for
dividends and splits. Intraday responses carry no adjusted closes, so
the raw closes are used there.
"""

from datetime import UTC, datetime
from typing import Any

import httpx

from ingestion.base import IntradayBar, PriceBar, ProviderError

BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/"
USER_AGENT = "Mozilla/5.0 (portfolio-risk-analytics)"
INTRADAY_INTERVALS = {"15m", "30m", "1h"}


def _chart_result(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate one chart API response and return its first result."""
    chart = payload.get("chart") or {}
    if chart.get("error"):
        raise ProviderError(f"Yahoo returned an error: {chart['error']}")
    results = chart.get("result") or []
    if not results:
        raise ProviderError("Yahoo response contained no result")
    result: dict[str, Any] = results[0]
    return result


def parse_chart_json(payload: dict[str, Any]) -> list[PriceBar]:
    """Turn one chart API response into price bars.

    Yahoo pads its arrays with null for days without data, those rows
    are skipped.
    """
    result = _chart_result(payload)
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


def parse_intraday_json(payload: dict[str, Any]) -> list[IntradayBar]:
    """Turn one intraday chart API response into intraday bars.

    The JSON shape matches the daily response, but the timestamps are
    intraday epochs and there is no adjclose block, so the raw closes
    are used. Null rows are skipped like in the daily parser.
    """
    result = _chart_result(payload)
    timestamps = result.get("timestamp") or []
    indicators = result.get("indicators") or {}
    quotes = (indicators.get("quote") or [{}])[0]

    bars: list[IntradayBar] = []
    for i, ts in enumerate(timestamps):
        close = _at(quotes.get("close"), i)
        o = _at(quotes.get("open"), i)
        h = _at(quotes.get("high"), i)
        lo = _at(quotes.get("low"), i)
        v = _at(quotes.get("volume"), i)
        if close is None or o is None or h is None or lo is None:
            continue
        bars.append(
            IntradayBar(
                ts=datetime.fromtimestamp(ts, tz=UTC),
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


def _get_chart(symbol: str, lookback: str, interval: str, timeout: float) -> dict[str, Any]:
    """Download one chart API response for a symbol."""
    try:
        response = httpx.get(
            BASE_URL + symbol.upper(),
            params={"range": lookback, "interval": interval},
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
            follow_redirects=True,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise ProviderError(f"Request to Yahoo failed for {symbol}: {exc}") from exc
    payload: dict[str, Any] = response.json()
    return payload


def fetch_daily(symbol: str, lookback: str = "5y", timeout: float = 30.0) -> list[PriceBar]:
    """Download daily history for one symbol, e.g. SPY or AAPL."""
    return parse_chart_json(_get_chart(symbol, lookback, "1d", timeout))


def fetch_intraday(
    symbol: str, interval: str = "1h", lookback: str = "60d", timeout: float = 30.0
) -> list[IntradayBar]:
    """Download intraday history for one symbol, hourly by default."""
    if interval not in INTRADAY_INTERVALS:
        raise ProviderError(
            f"Unsupported intraday interval {interval}, choose one of {sorted(INTRADAY_INTERVALS)}"
        )
    return parse_intraday_json(_get_chart(symbol, lookback, interval, timeout))
