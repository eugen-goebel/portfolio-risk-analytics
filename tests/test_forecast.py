"""Volatility forecasters checked against hand-computed values."""

import numpy as np
import pandas as pd
import pytest

from analytics.forecast import (
    evaluate_models,
    ewma_variance_forecast,
    fit_har,
    har_variance_forecast,
    rolling_variance_forecast,
)


def series(values: list[float]) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=len(values), freq="B")
    return pd.Series(values, index=idx)


def demo_returns(days: int = 600) -> pd.Series:
    rng = np.random.default_rng(7)
    return pd.Series(
        rng.normal(0.0003, 0.011, size=days),
        index=pd.date_range("2022-01-03", periods=days, freq="B"),
    )


class TestEwma:
    def test_hand_computed_recursion(self):
        # seed 0.01^2, then 0.9 * prev + 0.1 * r2
        path = ewma_variance_forecast(series([0.01, 0.02, -0.01]), lam=0.9)
        assert path.iloc[0] == pytest.approx(1e-4)
        assert path.iloc[1] == pytest.approx(0.9 * 1e-4 + 0.1 * 4e-4)
        assert path.iloc[2] == pytest.approx(0.9 * 1.3e-4 + 0.1 * 1e-4)

    def test_no_lookahead(self):
        full = demo_returns(100)
        prefix = full.iloc[:60]
        full_path = ewma_variance_forecast(full)
        prefix_path = ewma_variance_forecast(prefix)
        pd.testing.assert_series_equal(full_path.iloc[:60], prefix_path)


class TestRollingBaseline:
    def test_hand_computed(self):
        path = rolling_variance_forecast(series([0.01, 0.03]), window=2)
        assert np.isnan(path.iloc[0])
        assert path.iloc[1] == pytest.approx((1e-4 + 9e-4) / 2)

    def test_no_lookahead(self):
        full = demo_returns(100)
        full_path = rolling_variance_forecast(full)
        prefix_path = rolling_variance_forecast(full.iloc[:60])
        pd.testing.assert_series_equal(full_path.iloc[:60], prefix_path)


class TestHar:
    def test_constant_variance_is_recovered(self):
        # on white noise the forecast has to land near the true variance
        returns = demo_returns(800)
        coefs = fit_har(returns)
        forecast = har_variance_forecast(returns, coefs)
        true_var = float(returns.var())
        assert forecast == pytest.approx(true_var, rel=0.5)

    def test_forecast_is_never_negative(self):
        returns = demo_returns(200)
        coefs = np.array([-1.0, 0.0, 0.0, 0.0])
        assert har_variance_forecast(returns, coefs) == 0.0

    def test_too_little_data_raises(self):
        with pytest.raises(ValueError, match="Not enough observations"):
            fit_har(series([0.01] * 30))


class TestEvaluateModels:
    def test_report_structure(self):
        report = evaluate_models("demo", demo_returns(500), test_size=100)
        assert report.test_observations == 100
        assert {s.model for s in report.scores} == {"rolling", "ewma", "har"}
        assert report.best_model in {"rolling", "ewma", "har"}
        for score in report.scores:
            assert np.isfinite(score.mae_pct) and score.mae_pct > 0
            assert score.rmse_pct >= score.mae_pct * 0.5
        for vol in report.next_day_volatility_pct.values():
            assert 0 < vol < 200

    def test_short_series_raises(self):
        with pytest.raises(ValueError, match="walk-forward"):
            evaluate_models("demo", demo_returns(80), test_size=250)

    def test_test_size_shrinks_to_available_data(self):
        report = evaluate_models("demo", demo_returns(150), test_size=10_000)
        assert report.test_observations == 150 - 66
