"""Deterministic demo data so the platform works offline.

Generates a reproducible geometric random walk per symbol. The same
symbol always produces the same series, which keeps tests stable and
lets CI run without network access.
"""

import zlib
from datetime import date, timedelta

import numpy as np

from ingestion.base import PriceBar


def generate_demo_bars(symbol: str, days: int = 500, start_price: float = 100.0) -> list[PriceBar]:
    seed = zlib.crc32(symbol.encode())
    rng = np.random.default_rng(seed)
    daily_returns = rng.normal(loc=0.0004, scale=0.012, size=days)

    bars: list[PriceBar] = []
    price = start_price
    day = date.today() - timedelta(days=days)
    for r in daily_returns:
        day += timedelta(days=1)
        if day.weekday() >= 5:
            continue
        close = price * (1 + r)
        high = max(price, close) * (1 + abs(rng.normal(0, 0.003)))
        low = min(price, close) * (1 - abs(rng.normal(0, 0.003)))
        bars.append(
            PriceBar(
                day=day,
                open=round(price, 4),
                high=round(high, 4),
                low=round(low, 4),
                close=round(close, 4),
                volume=float(rng.integers(1_000_000, 50_000_000)),
            )
        )
        price = close
    return bars
