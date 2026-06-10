"""Shared types for price data providers."""

from dataclasses import dataclass
from datetime import date


class ProviderError(RuntimeError):
    """Raised when a data provider returns no usable data."""


@dataclass(frozen=True)
class PriceBar:
    day: date
    open: float
    high: float
    low: float
    close: float
    volume: float
