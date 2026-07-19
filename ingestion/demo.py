"""Deterministic demo data so the platform works offline.

Generates a reproducible price series per symbol. The same symbol always
produces the same series, on any day it runs, which keeps tests stable and
lets CI run without network access. Only the bar dates track the calendar; the
returns are fixed by the seeds, so realized volatility and the optimizer output
never drift with the date.

The named demo assets are shaped to behave like the things they stand for:
bonds are calm, equity swings, gold sits in between. Each series is a shared
market factor plus an idiosyncratic part, so the correlation matrix and the
portfolio tab show a real diversification effect instead of three
interchangeable random walks. Because every symbol loads on the same factor,
the correlation between two assets is the product of their betas.
"""

import zlib
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from ingestion.base import PriceBar

# Every symbol loads on the same market factor, so the factor needs a fixed
# seed of its own. The per-symbol seed only drives the idiosyncratic part.
# This particular seed is one whose realized paths land near the profiles
# below over the 750 days the dashboard asks for: equity Sharpe stays inside a
# believable band and the two generic assets both earn a positive return, so
# the optimizer's tangency portfolio is always defined. Other seeds happen to
# draw a two year bull run that reads as a broken demo rather than a good one.
MARKET_SEED = 60
TRADING_DAYS = 252


@dataclass(frozen=True)
class AssetProfile:
    """Annualized drift and volatility, plus the correlation to the market.

    beta is that correlation, so it stays within [-1, 1].
    """

    drift: float
    volatility: float
    beta: float


PROFILES: dict[str, AssetProfile] = {
    "demo-equity": AssetProfile(drift=0.08, volatility=0.18, beta=0.95),
    "demo-bonds": AssetProfile(drift=0.03, volatility=0.045, beta=-0.15),
    "demo-gold": AssetProfile(drift=0.05, volatility=0.14, beta=-0.30),
}
DEFAULT_PROFILE = AssetProfile(drift=0.06, volatility=0.16, beta=0.80)

# The symbols the dashboard seeds when it holds no data of its own. They map to
# the named profiles above; any other symbol falls back to DEFAULT_PROFILE.
DEMO_SYMBOLS = ("demo-equity", "demo-bonds", "demo-gold")

# Bump whenever generate_demo_bars changes its output. A hosted deploy keeps its
# database across redeploys, so without a version stamp a changed generator
# never reaches the live demo. "1" was the original calendar-dependent series;
# "2" is the fixed generator that no longer drifts with the run date.
DEMO_DATA_VERSION = "2"


def _market_factor(days: int) -> np.ndarray:
    """Standardized shocks that every symbol loads on."""
    return np.random.default_rng(MARKET_SEED).normal(size=days)


def _weekday_calendar(count: int, end: date) -> list[date]:
    """The ``count`` most recent weekdays ending on or before ``end``, in order.

    Only the bar dates depend on ``end``; the price path itself is fixed by the
    seeds. Skipping weekends here (instead of dropping their returns in the loop)
    is what keeps the series from shifting with the day it is generated on.
    """
    days: list[date] = []
    cursor = end
    while len(days) < count:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor -= timedelta(days=1)
    days.reverse()
    return days


def generate_demo_bars(symbol: str, days: int = 500, start_price: float = 100.0) -> list[PriceBar]:
    profile = PROFILES.get(symbol, DEFAULT_PROFILE)
    rng = np.random.default_rng(zlib.crc32(symbol.encode()))

    # Weighting the idiosyncratic part with sqrt(1 - beta**2) keeps the
    # combined shock at unit variance, so realized volatility lands on
    # profile.volatility no matter which beta the profile carries.
    shock = profile.beta * _market_factor(days) + np.sqrt(1 - profile.beta**2) * rng.normal(
        size=days
    )
    daily_returns = (
        profile.drift / TRADING_DAYS + profile.volatility / np.sqrt(TRADING_DAYS) * shock
    )

    bars: list[PriceBar] = []
    price = start_price
    calendar = _weekday_calendar(len(daily_returns), date.today())
    for day, r in zip(calendar, daily_returns, strict=True):
        # plain Python floats: psycopg2 rejects numpy scalars
        close = float(price * (1 + r))
        high = float(max(price, close) * (1 + abs(rng.normal(0, 0.003))))
        low = float(min(price, close) * (1 - abs(rng.normal(0, 0.003))))
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
