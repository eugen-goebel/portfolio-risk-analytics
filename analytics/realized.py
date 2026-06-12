"""Realized volatility from intraday closes.

Squared daily returns are a noisy proxy for the variance of a trading
day. Summing squared intraday returns within the day measures the same
quantity far more precisely, this is the realized volatility the HAR
forecasting model was originally specified on.

Returns are computed within each calendar day only: the first intraday
observation of a day has no predecessor inside that day, so overnight
gaps between the previous close and the next open never enter the sum.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252
MIN_INTRADAY_OBSERVATIONS = 2


def realized_volatility(intraday_closes: pd.Series) -> pd.Series:
    """Daily realized volatility from intraday closes.

    The input is a series of closes indexed by intraday timestamps
    spanning several days. Simple returns are computed between
    consecutive closes of the same calendar day, the realized
    volatility of a day is the square root of its sum of squared
    intraday returns. Days with fewer than two intraday observations
    carry no within-day return and are dropped. The result is indexed
    by day.
    """
    closes = intraday_closes.sort_index()
    days = pd.Index(closes.index.date)
    intraday_returns = closes.groupby(days).pct_change()
    observations = closes.groupby(days).size()
    daily = np.sqrt((intraday_returns**2).groupby(days).sum())
    daily = daily[observations >= MIN_INTRADAY_OBSERVATIONS]
    daily.index = pd.DatetimeIndex(daily.index)
    daily.name = closes.name
    return daily


def annualized_realized_volatility_pct(intraday_closes: pd.Series) -> pd.Series:
    """Annualized realized volatility in percent, for display."""
    return realized_volatility(intraday_closes) * np.sqrt(TRADING_DAYS) * 100
