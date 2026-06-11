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
    var_95_pct: float
    expected_shortfall_95_pct: float


class BenchmarkReport(BaseModel):
    symbol: str
    benchmark: str
    observations: int
    beta: float
    alpha_pct: float
    tracking_error_pct: float
    information_ratio: float


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


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical Value at Risk as a positive decimal.

    The (1 - confidence) empirical quantile of daily returns with
    linear interpolation, sign-flipped: a value of 0.02 means the worst
    five percent of days lost two percent or more. Floored at 0.0 so a
    strictly positive quantile never yields a negative VaR.
    """
    if len(returns) < 2:
        return 0.0
    return max(0.0, float(-returns.quantile(1 - confidence)))


def expected_shortfall(returns: pd.Series, confidence: float = 0.95) -> float:
    """Mean loss beyond the VaR threshold as a positive decimal.

    Average of the daily returns at or below the (1 - confidence)
    quantile, sign-flipped, so it is never smaller than the VaR at the
    same confidence. Floored at 0.0, and 0.0 when no observation falls
    in the tail or the series is too short.
    """
    if len(returns) < 2:
        return 0.0
    tail = returns[returns <= returns.quantile(1 - confidence)]
    if tail.empty:
        return 0.0
    return max(0.0, float(-tail.mean()))


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


def beta(asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Sensitivity of the asset to benchmark moves.

    Sample covariance of the two return series divided by the sample
    variance of the benchmark returns. Both series are assumed already
    aligned. 0.0 when the benchmark never moves or fewer than two
    observations are available.
    """
    if len(benchmark_returns) < 2:
        return 0.0
    benchmark_variance = float(benchmark_returns.var(ddof=1))
    if benchmark_variance == 0.0:
        return 0.0
    return float(asset_returns.cov(benchmark_returns, ddof=1)) / benchmark_variance


def capm_alpha(
    asset_returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate: float = 0.0
) -> float:
    """Annualized CAPM alpha against a yearly risk free rate.

    The return the asset earned beyond what its beta exposure to the
    benchmark explains: asset_annual - (rf + beta * (benchmark_annual - rf)),
    with annual returns as the mean daily return times 252.
    """
    asset_annual = float(asset_returns.mean()) * TRADING_DAYS
    benchmark_annual = float(benchmark_returns.mean()) * TRADING_DAYS
    exposure = beta(asset_returns, benchmark_returns)
    return asset_annual - (risk_free_rate + exposure * (benchmark_annual - risk_free_rate))


def tracking_error(asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Annualized standard deviation of the active returns.

    Active returns are asset minus benchmark daily returns, so an asset
    that tracks its benchmark perfectly has a tracking error of zero.
    0.0 below two observations.
    """
    return annualized_volatility(asset_returns - benchmark_returns)


def information_ratio(asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Annualized mean active return divided by the tracking error.

    0.0 when the tracking error is zero, so a perfect tracker scores
    zero instead of blowing up.
    """
    te = tracking_error(asset_returns, benchmark_returns)
    if te == 0.0:
        return 0.0
    active = asset_returns - benchmark_returns
    return float(active.mean()) * TRADING_DAYS / te


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
        var_95_pct=round(historical_var(returns) * 100, 2),
        expected_shortfall_95_pct=round(expected_shortfall(returns) * 100, 2),
    )


def compare_to_benchmark(
    symbol: str,
    prices: pd.Series,
    benchmark_symbol: str,
    benchmark_prices: pd.Series,
    risk_free_rate: float = 0.0,
) -> BenchmarkReport:
    """Bundle the benchmark-relative metrics for one asset.

    The two price series are aligned on their common dates first, so
    mismatched histories only count where both assets traded. Returns
    are computed on the aligned series.
    """
    aligned = pd.concat([prices, benchmark_prices], axis=1, join="inner").dropna()
    if len(aligned) < 30:
        raise ValueError(
            f"Need at least 30 common observations for {symbol} and {benchmark_symbol}, "
            f"have {len(aligned)}"
        )
    asset_returns = daily_returns(aligned.iloc[:, 0])
    benchmark_returns = daily_returns(aligned.iloc[:, 1])
    return BenchmarkReport(
        symbol=symbol,
        benchmark=benchmark_symbol,
        observations=int(len(asset_returns)),
        beta=round(beta(asset_returns, benchmark_returns), 3),
        alpha_pct=round(capm_alpha(asset_returns, benchmark_returns, risk_free_rate) * 100, 2),
        tracking_error_pct=round(tracking_error(asset_returns, benchmark_returns) * 100, 2),
        information_ratio=round(information_ratio(asset_returns, benchmark_returns), 3),
    )
