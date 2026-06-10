"""Portfolio risk metrics on top of daily closing prices.

All functions take pandas objects indexed by date. Annualization uses
252 trading days. Returns are simple daily returns, not log returns.
"""

import numpy as np
import pandas as pd
from pydantic import BaseModel

TRADING_DAYS = 252


class MetricsSummary(BaseModel):
    symbol: str
    observations: int
    total_return_pct: float
    annualized_volatility_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float


def daily_returns(prices: pd.Series) -> pd.Series:
    """Simple day-over-day returns, without the first NaN."""
    return prices.pct_change().dropna()


def annualized_volatility(returns: pd.Series) -> float:
    """Sample standard deviation of daily returns, scaled to one year."""
    if len(returns) < 2:
        return 0.0
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualized Sharpe ratio against a yearly risk free rate.

    The risk free rate is given as a yearly decimal, for example 0.02
    for two percent.
    """
    vol = annualized_volatility(returns)
    if vol == 0.0:
        return 0.0
    annual_return = float(returns.mean()) * TRADING_DAYS
    return (annual_return - risk_free_rate) / vol


def drawdown_series(prices: pd.Series) -> pd.Series:
    """Running drawdown relative to the historical peak, as decimals."""
    if prices.empty:
        return prices
    return prices / prices.cummax() - 1.0


def max_drawdown(prices: pd.Series) -> float:
    """Largest peak-to-trough loss as a negative decimal.

    A series that falls from 120 to 90 at some point has a max drawdown
    of -0.25 regardless of what happens afterwards.
    """
    if prices.empty:
        return 0.0
    return float(drawdown_series(prices).min())


def total_return(prices: pd.Series) -> float:
    """Overall return from the first to the last observation."""
    if len(prices) < 2:
        return 0.0
    return float(prices.iloc[-1] / prices.iloc[0] - 1.0)


def correlation_matrix(price_frame: pd.DataFrame) -> pd.DataFrame:
    """Pairwise correlation of daily returns between assets."""
    returns = price_frame.pct_change().dropna(how="all")
    return returns.corr()


def portfolio_returns(price_frame: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Daily returns of a weighted portfolio.

    The columns of the frame are symbols. Weights must cover exactly
    those symbols and sum to one within a small tolerance.
    """
    missing = set(weights) - set(price_frame.columns)
    if missing:
        raise ValueError(f"No price data for: {', '.join(sorted(missing))}")
    if abs(sum(weights.values()) - 1.0) > 1e-6:
        raise ValueError("Portfolio weights must sum to 1")

    returns = price_frame[list(weights)].pct_change().dropna()
    weight_vector = np.array([weights[c] for c in returns.columns])
    return pd.Series(returns.to_numpy() @ weight_vector, index=returns.index)


def summarize(symbol: str, prices: pd.Series, risk_free_rate: float = 0.0) -> MetricsSummary:
    """Bundle the headline metrics for one price series."""
    returns = daily_returns(prices)
    return MetricsSummary(
        symbol=symbol,
        observations=int(len(prices)),
        total_return_pct=round(total_return(prices) * 100, 2),
        annualized_volatility_pct=round(annualized_volatility(returns) * 100, 2),
        sharpe_ratio=round(sharpe_ratio(returns, risk_free_rate), 3),
        max_drawdown_pct=round(max_drawdown(prices) * 100, 2),
    )
