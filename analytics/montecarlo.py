"""Monte Carlo simulation of future value paths from daily returns.

A point forecast hides how wide the range of plausible outcomes is,
so this module simulates thousands of value paths and reports the
distribution of where they end. The core assumption is stated
plainly: resampling history assumes the future resembles the past.
The simulated ranges describe a market that behaves like the stored
sample, they say nothing about events that sample never contained.

Two sampling methods are supported, both drawing iid daily returns.

- bootstrap: draws from the historical sample with replacement,
  keeping the empirical shape of the distribution, fat tails included
- normal: draws from a normal distribution with the sample mean and
  sample standard deviation, the classic textbook simplification

Every path starts at a value of 100 and compounds one drawn return
per day over the horizon.
"""

import numpy as np
import pandas as pd
from pydantic import BaseModel

START_VALUE = 100.0
MIN_HISTORY = 100
MIN_PATHS = 100
PERCENTILE_LEVELS = {"p5": 5, "p25": 25, "p50": 50, "p75": 75, "p95": 95}


class MonteCarloReport(BaseModel):
    symbol: str
    method: str
    horizon_days: int
    n_paths: int
    start_value: float
    percentiles: dict[str, float]
    prob_loss: float
    expected_final: float


def simulate_paths(
    returns: pd.Series,
    horizon: int = 252,
    n_paths: int = 2000,
    method: str = "bootstrap",
    seed: int | None = None,
) -> np.ndarray:
    """Value paths of shape (n_paths, horizon + 1), each starting at 100."""
    rng = np.random.default_rng(seed)
    sample = returns.to_numpy()
    if method == "bootstrap":
        draws = rng.choice(sample, size=(n_paths, horizon), replace=True)
    elif method == "normal":
        mean = float(np.mean(sample))
        std = float(np.std(sample, ddof=1))
        draws = rng.normal(mean, std, size=(n_paths, horizon))
    else:
        raise ValueError(f"Unknown method '{method}', expected 'bootstrap' or 'normal'")
    paths = np.empty((n_paths, horizon + 1))
    paths[:, 0] = START_VALUE
    paths[:, 1:] = START_VALUE * np.cumprod(1.0 + draws, axis=1)
    return paths


def run_monte_carlo(
    symbol: str,
    returns: pd.Series,
    horizon: int = 252,
    n_paths: int = 2000,
    method: str = "bootstrap",
    seed: int | None = None,
) -> MonteCarloReport:
    """Simulate value paths and summarize the distribution of final values."""
    if len(returns) < MIN_HISTORY:
        raise ValueError(f"Need at least {MIN_HISTORY} historical returns, got {len(returns)}")
    if horizon < 1:
        raise ValueError("Horizon must be at least 1 trading day")
    if n_paths < MIN_PATHS:
        raise ValueError(f"Need at least {MIN_PATHS} paths, got {n_paths}")

    paths = simulate_paths(returns, horizon, n_paths, method, seed)
    final = paths[:, -1]
    return MonteCarloReport(
        symbol=symbol,
        method=method,
        horizon_days=horizon,
        n_paths=n_paths,
        start_value=START_VALUE,
        percentiles={
            name: round(float(np.percentile(final, level)), 2)
            for name, level in PERCENTILE_LEVELS.items()
        },
        prob_loss=round(float(np.mean(final < START_VALUE)), 4),
        expected_final=round(float(np.mean(final)), 2),
    )
