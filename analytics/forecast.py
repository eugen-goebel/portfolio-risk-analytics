"""One-day-ahead volatility forecasting on daily returns.

Daily returns themselves are close to unpredictable, their volatility
is not: turbulent days cluster. Three forecasters are compared here.

- rolling: the mean squared return over a trailing window, the naive
  baseline every other model has to beat
- ewma: the RiskMetrics exponentially weighted recursion with a decay
  of 0.94
- har: the heterogeneous autoregressive model (Corsi 2009), a linear
  regression of tomorrow's squared return on the daily, weekly and
  monthly mean squared returns, fit with ordinary least squares

All forecast series follow one convention: the value stored at date t
is the variance forecast for t+1, computed from returns up to and
including t. Evaluation is walk-forward, the HAR coefficients are
refit at every step on an expanding window, so no forecast ever sees
the future.
"""

import numpy as np
import pandas as pd
from pydantic import BaseModel

from analytics.metrics import TRADING_DAYS

ROLLING_WINDOW = 22
HAR_WEEK = 5
HAR_MONTH = 22
EWMA_LAMBDA = 0.94


class ModelScore(BaseModel):
    model: str
    mae_pct: float
    rmse_pct: float


class ForecastReport(BaseModel):
    symbol: str
    test_observations: int
    next_day_volatility_pct: dict[str, float]
    scores: list[ModelScore]
    best_model: str


def rolling_variance_forecast(returns: pd.Series, window: int = ROLLING_WINDOW) -> pd.Series:
    """Trailing mean squared return as the forecast for the next day."""
    return (returns**2).rolling(window).mean()


def ewma_variance_forecast(returns: pd.Series, lam: float = EWMA_LAMBDA) -> pd.Series:
    """RiskMetrics recursion, seeded with the first squared return."""
    r2 = (returns**2).to_numpy()
    out = np.empty(len(r2))
    out[0] = r2[0]
    for i in range(1, len(r2)):
        out[i] = lam * out[i - 1] + (1 - lam) * r2[i]
    return pd.Series(out, index=returns.index)


def har_features(returns: pd.Series) -> pd.DataFrame:
    """Daily, weekly and monthly mean squared returns at each date."""
    r2 = returns**2
    return pd.DataFrame(
        {
            "daily": r2,
            "weekly": r2.rolling(HAR_WEEK).mean(),
            "monthly": r2.rolling(HAR_MONTH).mean(),
        }
    )


def fit_har(returns: pd.Series) -> np.ndarray:
    """Least squares fit of next-day squared returns on the HAR features.

    Returns the coefficients (intercept, daily, weekly, monthly).
    """
    features = har_features(returns)
    target = (returns**2).shift(-1)
    valid = features.notna().all(axis=1) & target.notna()
    if int(valid.sum()) < HAR_MONTH * 2:
        raise ValueError("Not enough observations to fit the HAR model")
    x = features[valid].to_numpy()
    x = np.column_stack([np.ones(len(x)), x])
    y = target[valid].to_numpy()
    coefs, *_ = np.linalg.lstsq(x, y, rcond=None)
    return np.asarray(coefs)


def har_variance_forecast(returns: pd.Series, coefs: np.ndarray) -> float:
    """Variance forecast for the day after the last observation."""
    last = har_features(returns).iloc[-1]
    if last.isna().any():
        raise ValueError("Not enough observations to build the HAR features")
    value = coefs[0] + float(np.dot(coefs[1:], last.to_numpy()))
    return max(float(value), 0.0)


def _annualized_pct(variance: float) -> float:
    return float(np.sqrt(max(variance, 0.0) * TRADING_DAYS) * 100)


def evaluate_models(symbol: str, returns: pd.Series, test_size: int = 250) -> ForecastReport:
    """Walk-forward comparison of the three forecasters.

    Forecast volatility is scored against the absolute next-day return,
    the observable if noisy proxy for true volatility. Errors are
    reported in daily percent.
    """
    min_history = HAR_MONTH * 3
    if len(returns) < min_history + test_size:
        test_size = len(returns) - min_history
    if test_size < 30:
        raise ValueError("Not enough observations for a meaningful walk-forward test")

    rolling_path = rolling_variance_forecast(returns)
    ewma_path = ewma_variance_forecast(returns)

    start = len(returns) - test_size - 1
    targets = np.empty(test_size)
    errors: dict[str, np.ndarray] = {m: np.empty(test_size) for m in ("rolling", "ewma", "har")}

    for step in range(test_size):
        t = start + step
        history = returns.iloc[: t + 1]
        target_vol = abs(float(returns.iloc[t + 1]))
        targets[step] = target_vol

        forecasts = {
            "rolling": float(rolling_path.iloc[t]),
            "ewma": float(ewma_path.iloc[t]),
            "har": har_variance_forecast(history, fit_har(history)),
        }
        for model, variance in forecasts.items():
            errors[model][step] = np.sqrt(max(variance, 0.0)) - target_vol

    scores = [
        ModelScore(
            model=model,
            mae_pct=round(float(np.mean(np.abs(errs))) * 100, 4),
            rmse_pct=round(float(np.sqrt(np.mean(errs**2))) * 100, 4),
        )
        for model, errs in errors.items()
    ]
    best = min(scores, key=lambda s: s.rmse_pct)

    next_day = {
        "rolling": _annualized_pct(float(rolling_path.iloc[-1])),
        "ewma": _annualized_pct(float(ewma_path.iloc[-1])),
        "har": _annualized_pct(har_variance_forecast(returns, fit_har(returns))),
    }

    return ForecastReport(
        symbol=symbol,
        test_observations=test_size,
        next_day_volatility_pct={k: round(v, 2) for k, v in next_day.items()},
        scores=scores,
        best_model=best.model,
    )
