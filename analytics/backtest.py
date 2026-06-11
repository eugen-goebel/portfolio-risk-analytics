"""Rebalanced portfolio backtesting against buy and hold.

A portfolio that is bought once drifts away from its target weights,
the winners grow into ever larger positions and quietly concentrate
the risk. This module simulates the same target weights twice over a
shared price history.

- buy and hold: shares are bought at the first close and never
  touched, so the weights float with performance
- rebalanced: on the first trading day of each calendar month or
  quarter the shares are reset to the target weights at that day's
  closes, trading discipline against drift

Both value paths start at the same initial value and are summarized
with the usual risk metrics, so the effect of rebalancing discipline
shows up directly in the comparison. Transaction costs are not
modeled.
"""

import numpy as np
import pandas as pd
from pydantic import BaseModel

from analytics.metrics import (
    annualized_volatility,
    daily_returns,
    max_drawdown,
    sharpe_ratio,
    total_return,
)

REBALANCE_FREQUENCIES = ("monthly", "quarterly")
MIN_OBSERVATIONS = 30


class StrategyResult(BaseModel):
    final_value: float
    total_return_pct: float
    annualized_volatility_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float


class BacktestReport(BaseModel):
    symbols: list[str]
    rebalance: str
    observations: int
    rebalanced: StrategyResult
    buy_and_hold: StrategyResult


def _validate_weights(price_frame: pd.DataFrame, weights: dict[str, float]) -> None:
    missing = set(weights) - set(price_frame.columns)
    if missing:
        raise ValueError(f"No price data for: {', '.join(sorted(missing))}")
    if abs(sum(weights.values()) - 1.0) > 1e-6:
        raise ValueError("Portfolio weights must sum to 1")


def rebalance_dates(index: pd.DatetimeIndex, rebalance: str) -> list[pd.Timestamp]:
    """First trading day of each calendar month or quarter in the index.

    The very first index entry is excluded, the portfolio starts at
    the target weights anyway.
    """
    if rebalance not in REBALANCE_FREQUENCIES:
        raise ValueError(f"rebalance must be one of: {', '.join(REBALANCE_FREQUENCIES)}")
    freq = "M" if rebalance == "monthly" else "Q"
    firsts = pd.Series(index, index=index.to_period(freq)).groupby(level=0).first()
    return [day for day in firsts if day != index[0]]


def buy_and_hold(
    price_frame: pd.DataFrame, weights: dict[str, float], initial_value: float = 100.0
) -> pd.Series:
    """Value path of a portfolio bought once at the first close.

    Shares are sized to the target weights at the first row and never
    touched again, so the weights drift with performance.
    """
    _validate_weights(price_frame, weights)
    columns = list(weights)
    prices = price_frame[columns].to_numpy(dtype=float)
    weight_vector = np.array([weights[c] for c in columns])
    shares = initial_value * weight_vector / prices[0]
    return pd.Series(prices @ shares, index=price_frame.index)


def simulate(
    price_frame: pd.DataFrame,
    weights: dict[str, float],
    rebalance: str = "monthly",
    initial_value: float = 100.0,
) -> pd.Series:
    """Value path of the periodically rebalanced portfolio.

    Starts exactly like buy and hold, but on every rebalance date the
    shares are reset to the target weights at that day's closes. The
    reset happens after the day's value is recorded, so it never
    changes the value on the rebalance day itself.
    """
    _validate_weights(price_frame, weights)
    schedule = set(rebalance_dates(price_frame.index, rebalance))
    columns = list(weights)
    prices = price_frame[columns].to_numpy(dtype=float)
    weight_vector = np.array([weights[c] for c in columns])

    shares = initial_value * weight_vector / prices[0]
    values = np.empty(len(prices))
    for i, day in enumerate(price_frame.index):
        value = float(prices[i] @ shares)
        values[i] = value
        if day in schedule:
            shares = value * weight_vector / prices[i]
    return pd.Series(values, index=price_frame.index)


def _summarize(values: pd.Series, risk_free_rate: float) -> StrategyResult:
    returns = daily_returns(values)
    return StrategyResult(
        final_value=round(float(values.iloc[-1]), 2),
        total_return_pct=round(total_return(values) * 100, 2),
        annualized_volatility_pct=round(annualized_volatility(returns) * 100, 2),
        sharpe_ratio=round(sharpe_ratio(returns, risk_free_rate), 3),
        max_drawdown_pct=round(max_drawdown(values) * 100, 2),
    )


def run_backtest(
    price_frame: pd.DataFrame,
    weights: dict[str, float],
    rebalance: str = "monthly",
    risk_free_rate: float = 0.0,
) -> BacktestReport:
    """Run both strategies on the same frame and summarize each path."""
    if len(price_frame) < MIN_OBSERVATIONS:
        raise ValueError(f"Need at least {MIN_OBSERVATIONS} observations, have {len(price_frame)}")
    return BacktestReport(
        symbols=list(weights),
        rebalance=rebalance,
        observations=len(price_frame),
        rebalanced=_summarize(simulate(price_frame, weights, rebalance), risk_free_rate),
        buy_and_hold=_summarize(buy_and_hold(price_frame, weights), risk_free_rate),
    )
