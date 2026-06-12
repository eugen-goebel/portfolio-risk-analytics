"""Closed-form Markowitz mean-variance portfolio optimization.

Two textbook portfolios are computed directly from the annualized
return moments with plain linear algebra, no numerical optimizer.

- minimum variance: w = inv(S) 1 / (1' inv(S) 1), the fully invested
  portfolio with the smallest variance regardless of expected returns
- tangency (maximum Sharpe): w proportional to inv(S) (mu - rf),
  normalized to sum to 1, the portfolio with the highest Sharpe ratio

Annualization is consistent throughout: the expected return vector is
the mean daily return times 252, the covariance matrix is the sample
covariance (ddof=1) of daily returns times 252.

The assumptions are the textbook ones and they are strong. The sample
moments are treated as the true ones, although estimated weights are
notoriously sensitive to errors in the mean vector. The closed forms
are unconstrained, so negative weights, meaning short positions, are
allowed: that is the textbook unconstrained solution, not a tradable
long-only allocation.
"""

import numpy as np
import pandas as pd
from pydantic import BaseModel

from analytics.metrics import TRADING_DAYS

MIN_ASSETS = 2
MIN_OBSERVATIONS = 60


class OptimizedPortfolio(BaseModel):
    weights: dict[str, float]
    expected_return_pct: float
    volatility_pct: float
    sharpe_ratio: float


class OptimizationReport(BaseModel):
    symbols: list[str]
    observations: int
    risk_free_rate: float
    minimum_variance: OptimizedPortfolio
    maximum_sharpe: OptimizedPortfolio


def _annualized_moments(returns: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Annualized mean vector and covariance matrix of daily returns."""
    mu = returns.mean().to_numpy() * TRADING_DAYS
    cov = returns.cov(ddof=1).to_numpy() * TRADING_DAYS
    return mu, cov


def _solve(cov: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve cov @ x = rhs without forming an explicit inverse."""
    try:
        return np.asarray(np.linalg.solve(cov, rhs))
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Covariance matrix is singular, asset returns are linearly dependent"
        ) from exc


def minimum_variance_weights(returns: pd.DataFrame) -> dict[str, float]:
    """Weights of the unconstrained global minimum variance portfolio.

    The closed form w = inv(S) 1 / (1' inv(S) 1), evaluated by solving
    the linear system S x = 1 and normalizing.
    """
    _, cov = _annualized_moments(returns)
    ones = np.ones(len(cov))
    raw = _solve(cov, ones)
    weights = raw / float(ones @ raw)
    return {symbol: float(w) for symbol, w in zip(returns.columns, weights, strict=True)}


def tangency_weights(returns: pd.DataFrame, risk_free_rate: float = 0.0) -> dict[str, float]:
    """Weights of the unconstrained maximum Sharpe (tangency) portfolio.

    w is proportional to inv(S) (mu - rf) and normalized to sum to 1.
    The normalization denominator must be positive, otherwise no asset
    earns a positive excess return and the tangency portfolio is
    undefined on the upper branch of the frontier.
    """
    mu, cov = _annualized_moments(returns)
    raw = _solve(cov, mu - risk_free_rate)
    denominator = float(raw.sum())
    if denominator <= 0.0:
        raise ValueError(
            "Tangency portfolio is undefined, no asset earns a positive excess return "
            "over the risk free rate"
        )
    weights = raw / denominator
    return {symbol: float(w) for symbol, w in zip(returns.columns, weights, strict=True)}


def _summarize(
    returns: pd.DataFrame, weights: dict[str, float], risk_free_rate: float
) -> OptimizedPortfolio:
    """Annualized return, volatility and Sharpe ratio of a weight vector."""
    mu, cov = _annualized_moments(returns)
    w = np.array([weights[symbol] for symbol in returns.columns])
    expected = float(w @ mu)
    volatility = float(np.sqrt(w @ cov @ w))
    sharpe = (expected - risk_free_rate) / volatility if volatility > 0.0 else 0.0
    return OptimizedPortfolio(
        weights={symbol: round(value, 4) for symbol, value in weights.items()},
        expected_return_pct=round(expected * 100, 2),
        volatility_pct=round(volatility * 100, 2),
        sharpe_ratio=round(sharpe, 3),
    )


def optimize_portfolio(
    price_frame: pd.DataFrame, risk_free_rate: float = 0.0
) -> OptimizationReport:
    """Both closed-form portfolios from one shared price history."""
    if price_frame.shape[1] < MIN_ASSETS:
        raise ValueError(f"Need at least {MIN_ASSETS} assets, have {price_frame.shape[1]}")
    returns = price_frame.pct_change().dropna()
    if len(returns) < MIN_OBSERVATIONS:
        raise ValueError(
            f"Need at least {MIN_OBSERVATIONS} return observations, have {len(returns)}"
        )
    return OptimizationReport(
        symbols=list(price_frame.columns),
        observations=len(returns),
        risk_free_rate=risk_free_rate,
        minimum_variance=_summarize(returns, minimum_variance_weights(returns), risk_free_rate),
        maximum_sharpe=_summarize(
            returns, tangency_weights(returns, risk_free_rate), risk_free_rate
        ),
    )
