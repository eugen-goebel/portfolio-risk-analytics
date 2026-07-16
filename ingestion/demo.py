"""Deterministic demo data so the platform works offline.

Generates a reproducible price series per symbol. The same symbol always
produces the same series, which keeps tests stable and lets CI run without
network access.

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
# below over the 750 days the dashboard asks for. Other seeds happen to draw
# a two year bull run that would put the equity Sharpe ratio above two, which
# reads as a broken demo rather than a good one.
MARKET_SEED = 12345
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


def _market_factor(days: int) -> np.ndarray:
    """Standardized shocks that every symbol loads on."""
    return np.random.default_rng(MARKET_SEED).normal(size=days)


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
    day = date.today() - timedelta(days=days)
    for r in daily_returns:
        day += timedelta(days=1)
        if day.weekday() >= 5:
            continue
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
