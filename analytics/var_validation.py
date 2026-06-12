"""Out-of-sample VaR backtesting with the Kupiec proportion-of-failures test.

A 95% VaR model promises that daily losses exceed the VaR on about one
day in twenty. Whether a model keeps that promise is an empirical
question, so this module backtests it: walk forward over the return
history, estimate the VaR at each day from the trailing window only,
and count the days that lost more than the estimate said they could.

The Kupiec (1995) proportion-of-failures test then asks whether the
observed breach count is consistent with the promised breach
probability. Too many breaches mean the model understates risk, too
few mean it is needlessly conservative, both reject the model. The
backtest has no lookahead, the VaR tested on day t is computed from
returns strictly before t.
"""

import math

import pandas as pd
from pydantic import BaseModel

from analytics.metrics import historical_var

DEFAULT_WINDOW = 250
MIN_OUT_OF_SAMPLE = 50
REJECTION_LEVEL = 0.05


class VarValidationReport(BaseModel):
    symbol: str
    confidence: float
    window: int
    observations: int
    expected_breaches: float
    actual_breaches: int
    breach_rate_pct: float
    kupiec_statistic: float
    p_value: float
    model_rejected: bool


def kupiec_pof(observations: int, breaches: int, confidence: float = 0.95) -> tuple[float, float]:
    """Kupiec proportion-of-failures likelihood ratio and its p-value.

    Under the null hypothesis the VaR model is correct, so each day is
    an independent breach with probability p = 1 - confidence. The
    likelihood ratio compares that null against the observed breach
    frequency x / n:

        LR = -2 * ((n - x) * ln(1 - p) + x * ln(p))
             + 2 * ((n - x) * ln(1 - x/n) + x * ln(x/n))

    For x = 0 and x = n the degenerate terms are dropped, they
    contribute nothing under the 0 * ln(0) = 0 convention.

    Under the null LR is asymptotically chi-square with one degree of
    freedom, the distribution of Z squared for a standard normal Z.
    Hence P(chi2_1 > LR) = P(|Z| > sqrt(LR)) = erfc(sqrt(LR / 2)),
    which gives the exact chi-square(1) survival function from the
    standard library without scipy.
    """
    if observations <= 0:
        raise ValueError("observations must be positive")
    if not 0 <= breaches <= observations:
        raise ValueError("breaches must lie between 0 and observations")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1")

    p = 1.0 - confidence
    n = observations
    x = breaches
    log_null = (n - x) * math.log(1.0 - p) + x * math.log(p)
    log_alt = 0.0
    if x > 0:
        log_alt += x * math.log(x / n)
    if x < n:
        log_alt += (n - x) * math.log(1.0 - x / n)
    # floored at zero: when x/n equals p exactly the statistic is zero
    # in exact arithmetic but can come out at -1e-16 in floating point
    statistic = max(0.0, -2.0 * log_null + 2.0 * log_alt)
    p_value = math.erfc(math.sqrt(statistic / 2.0))
    return statistic, p_value


def rolling_var_breaches(
    returns: pd.Series, window: int = DEFAULT_WINDOW, confidence: float = 0.95
) -> tuple[int, int]:
    """Walk-forward breach count of the historical VaR.

    At each step t >= window the VaR is the historical VaR of the
    window returns strictly before t, so no estimate ever sees the day
    it is tested on. A breach is a day whose return falls below the
    negative of that VaR. Returns the pair (observations, breaches).
    """
    observations = 0
    breaches = 0
    for t in range(window, len(returns)):
        var = historical_var(returns.iloc[t - window : t], confidence)
        observations += 1
        if float(returns.iloc[t]) < -var:
            breaches += 1
    return observations, breaches


def validate_var(
    symbol: str, returns: pd.Series, window: int = DEFAULT_WINDOW, confidence: float = 0.95
) -> VarValidationReport:
    """Backtest the historical VaR and test the breach count with Kupiec.

    The first window returns only seed the estimator, everything after
    them is scored out of sample. The model is rejected when the Kupiec
    p-value falls below the conventional 5% level.
    """
    out_of_sample = len(returns) - window
    if out_of_sample < MIN_OUT_OF_SAMPLE:
        raise ValueError(
            f"Need at least {MIN_OUT_OF_SAMPLE} out-of-sample observations after the "
            f"{window}-day window, have {max(out_of_sample, 0)}"
        )

    observations, breaches = rolling_var_breaches(returns, window, confidence)
    statistic, p_value = kupiec_pof(observations, breaches, confidence)

    return VarValidationReport(
        symbol=symbol,
        confidence=confidence,
        window=window,
        observations=observations,
        expected_breaches=round((1.0 - confidence) * observations, 1),
        actual_breaches=breaches,
        breach_rate_pct=round(breaches / observations * 100, 2),
        kupiec_statistic=round(statistic, 4),
        p_value=round(p_value, 4),
        model_rejected=p_value < REJECTION_LEVEL,
    )
