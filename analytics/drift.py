"""Distribution drift monitoring on daily returns.

A forecast fitted on a long reference window quietly assumes that
recent returns still look like that history. This module checks the
assumption by comparing the recent return distribution against the
reference window with two classic monitoring statistics.

- psi: the population stability index over quantile bins of the
  reference sample. Each bin holds about the same share of the
  reference, so any shift in the recent shares shows up directly.
- ks: the two-sample Kolmogorov-Smirnov statistic, the largest
  vertical distance between the two empirical CDFs.

Drift is flagged at the conventional industry thresholds, a PSI above
0.2 or a KS statistic above 0.15. A flag does not invalidate the
forecasts on its own, it says they were fitted on a history that no
longer matches the present and deserve scrutiny.
"""

import numpy as np
import pandas as pd
from pydantic import BaseModel

PSI_BINS = 10
PSI_THRESHOLD = 0.2
KS_THRESHOLD = 0.15
SHARE_FLOOR = 1e-6
MIN_REFERENCE = 100
MIN_RECENT = 20


class DriftReport(BaseModel):
    symbol: str
    reference_size: int
    recent_size: int
    psi: float
    ks: float
    mean_shift: float
    volatility_ratio: float
    drift_detected: bool


def population_stability_index(
    reference: pd.Series, recent: pd.Series, bins: int = PSI_BINS
) -> float:
    """Population stability index between the recent and the reference sample.

    Bin edges are the quantiles of the reference sample, so each
    reference bin holds about the same share of observations, with
    -inf and +inf as the outer edges. Duplicate quantile edges are
    collapsed and shares are floored at a small constant to keep the
    logarithm finite.
    """
    ref = reference.to_numpy(dtype=float)
    rec = recent.to_numpy(dtype=float)
    inner = np.unique(np.quantile(ref, np.linspace(0.0, 1.0, bins + 1)[1:-1]))
    edges = np.concatenate(([-np.inf], inner, [np.inf]))
    ref_share = np.clip(np.histogram(ref, bins=edges)[0] / len(ref), SHARE_FLOOR, None)
    rec_share = np.clip(np.histogram(rec, bins=edges)[0] / len(rec), SHARE_FLOOR, None)
    return float(np.sum((rec_share - ref_share) * np.log(rec_share / ref_share)))


def ks_statistic(reference: pd.Series, recent: pd.Series) -> float:
    """Two-sample Kolmogorov-Smirnov statistic.

    The maximum absolute distance between the two empirical CDFs,
    evaluated at every pooled observation.
    """
    ref = np.sort(reference.to_numpy(dtype=float))
    rec = np.sort(recent.to_numpy(dtype=float))
    pooled = np.concatenate([ref, rec])
    cdf_ref = np.searchsorted(ref, pooled, side="right") / len(ref)
    cdf_rec = np.searchsorted(rec, pooled, side="right") / len(rec)
    return float(np.max(np.abs(cdf_ref - cdf_rec)))


def evaluate_drift(
    symbol: str, returns: pd.Series, reference_size: int = 500, recent_size: int = 60
) -> DriftReport:
    """Compare the recent return window against the reference history.

    The last recent_size observations form the recent window, the
    reference_size observations before them the reference window. The
    reference shrinks to whatever history is available. Drift is
    flagged at the conventional industry thresholds, a PSI above 0.2
    or a KS statistic above 0.15. The mean shift is reported in daily
    percent, the volatility ratio as recent over reference standard
    deviation.
    """
    split = max(len(returns) - recent_size, 0)
    recent = returns.iloc[split:]
    reference = returns.iloc[:split].tail(reference_size)
    if len(recent) < MIN_RECENT:
        raise ValueError(f"Need at least {MIN_RECENT} recent observations, have {len(recent)}")
    if len(reference) < MIN_REFERENCE:
        raise ValueError(
            f"Need at least {MIN_REFERENCE} reference observations, have {len(reference)}"
        )

    psi = population_stability_index(reference, recent)
    ks = ks_statistic(reference, recent)
    mean_shift = (float(recent.mean()) - float(reference.mean())) * 100
    volatility_ratio = float(recent.std(ddof=1)) / float(reference.std(ddof=1))

    return DriftReport(
        symbol=symbol,
        reference_size=len(reference),
        recent_size=len(recent),
        psi=round(psi, 4),
        ks=round(ks, 4),
        mean_shift=round(mean_shift, 4),
        volatility_ratio=round(volatility_ratio, 4),
        drift_detected=psi > PSI_THRESHOLD or ks > KS_THRESHOLD,
    )
