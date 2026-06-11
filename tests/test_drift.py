"""Drift statistics checked against hand-computed values."""

import numpy as np
import pandas as pd
import pytest

from analytics.drift import evaluate_drift, ks_statistic, population_stability_index


def series(values) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=len(values), freq="B")
    return pd.Series(values, index=idx)


def normal_series(mean: float, std: float, size: int, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    return series(rng.normal(mean, std, size=size))


def regime_returns(
    reference_std: float, recent_std: float, reference_size: int = 500, recent_size: int = 60
) -> pd.Series:
    rng = np.random.default_rng(9)
    values = np.concatenate(
        [
            rng.normal(0.0, reference_std, size=reference_size),
            rng.normal(0.0, recent_std, size=recent_size),
        ]
    )
    return series(values)


class TestPsi:
    def test_sample_against_itself_is_near_zero(self):
        sample = normal_series(0.0, 0.01, 500, seed=1)
        assert population_stability_index(sample, sample) == pytest.approx(0.0, abs=1e-9)

    def test_shifted_sample_is_large(self):
        reference = normal_series(0.0, 0.01, 500, seed=1)
        recent = normal_series(0.02, 0.01, 200, seed=2)
        assert population_stability_index(reference, recent) > 0.2

    def test_duplicate_quantile_edges_are_collapsed(self):
        # a heavily tied reference produces duplicate quantiles
        reference = series([0.0] * 90 + [0.01] * 10)
        recent = series([0.0] * 50 + [0.01] * 50)
        value = population_stability_index(reference, recent)
        assert np.isfinite(value)
        assert value > 0


class TestKs:
    def test_hand_computed(self):
        # ECDF distances at the pooled points:
        # x=1: |1/4 - 0/4| = 0.25, x=2: |2/4 - 0/4| = 0.50,
        # x=3: |3/4 - 1/4| = 0.50, x=4: |4/4 - 2/4| = 0.50,
        # x=5: |4/4 - 3/4| = 0.25, x=6: |4/4 - 4/4| = 0.00
        # so the maximum is 0.5
        assert ks_statistic(series([1, 2, 3, 4]), series([3, 4, 5, 6])) == pytest.approx(0.5)

    def test_sample_against_itself_is_zero(self):
        sample = normal_series(0.0, 0.01, 300, seed=3)
        assert ks_statistic(sample, sample) == 0.0

    def test_disjoint_samples_reach_one(self):
        assert ks_statistic(series([1, 2, 3]), series([10, 11, 12])) == pytest.approx(1.0)


class TestEvaluateDrift:
    def test_report_structure(self):
        report = evaluate_drift("demo", regime_returns(0.01, 0.01))
        assert report.symbol == "demo"
        assert report.reference_size == 500
        assert report.recent_size == 60
        assert report.psi >= 0
        assert 0 <= report.ks <= 1
        assert report.volatility_ratio == pytest.approx(1.0, abs=0.3)

    def test_regime_change_is_detected(self):
        report = evaluate_drift("demo", regime_returns(0.01, 0.03))
        assert report.drift_detected
        assert report.volatility_ratio > 2.0

    def test_identical_regime_is_not_flagged(self):
        report = evaluate_drift("demo", regime_returns(0.01, 0.01))
        assert not report.drift_detected

    def test_reference_shrinks_to_available_history(self):
        report = evaluate_drift("demo", regime_returns(0.01, 0.01, reference_size=150))
        assert report.reference_size == 150
        assert report.recent_size == 60

    def test_too_few_reference_observations_raise(self):
        with pytest.raises(ValueError, match="reference"):
            evaluate_drift("demo", normal_series(0.0, 0.01, 100, seed=4))

    def test_too_few_recent_observations_raise(self):
        with pytest.raises(ValueError, match="recent"):
            evaluate_drift("demo", normal_series(0.0, 0.01, 700, seed=4), recent_size=10)
