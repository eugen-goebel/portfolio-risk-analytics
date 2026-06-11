"""Client for the official ECB exchange rate data API.

The API serves daily euro foreign exchange reference rates as CSV
without authentication, for example
https://data-api.ecb.europa.eu/service/data/EXR/D.USD.EUR.SP00.A
with format=csvdata. Rates are quoted as units of the foreign
currency per 1 EUR.

Reference rates have exactly one fix per business day, so each bar
carries the rate as open, high, low and close, with volume 0.
"""

import csv
import io
from datetime import date, timedelta

import httpx

from ingestion.base import PriceBar, ProviderError

BASE_URL = "https://data-api.ecb.europa.eu/service/data/EXR/"
DEFAULT_LOOKBACK_DAYS = 5 * 365

REQUIRED_COLUMNS = ("KEY", "TIME_PERIOD", "OBS_VALUE")


def parse_exr_csv(text: str) -> list[PriceBar]:
    """Turn one EXR CSV response into price bars.

    Rows with an empty OBS_VALUE (holidays and weekends) are skipped.
    """
    if not text.strip():
        raise ProviderError("ECB response was empty")
    reader = csv.DictReader(io.StringIO(text))
    missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
    if missing:
        raise ProviderError(f"ECB response is missing columns: {', '.join(missing)}")

    bars: list[PriceBar] = []
    for row in reader:
        value = (row.get("OBS_VALUE") or "").strip()
        if not value:
            continue
        rate = float(value)
        bars.append(
            PriceBar(
                day=date.fromisoformat(row["TIME_PERIOD"]),
                open=rate,
                high=rate,
                low=rate,
                close=rate,
                volume=0.0,
            )
        )
    if not bars:
        raise ProviderError("ECB response contained no usable rates")
    return bars


def fetch_daily_fx(
    currency: str, lookback_days: int = DEFAULT_LOOKBACK_DAYS, timeout: float = 30.0
) -> list[PriceBar]:
    """Download daily EUR reference rates for one currency code, e.g. USD."""
    start = date.today() - timedelta(days=lookback_days)
    try:
        response = httpx.get(
            f"{BASE_URL}D.{currency.upper()}.EUR.SP00.A",
            params={"format": "csvdata", "startPeriod": start.isoformat()},
            timeout=timeout,
            follow_redirects=True,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise ProviderError(f"Request to the ECB failed for {currency}: {exc}") from exc
    return parse_exr_csv(response.text)
