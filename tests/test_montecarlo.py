"""Monte Carlo simulation checked against hand-computed values."""

import numpy as np
import pandas as pd
import pytest

from analytics.montecarlo import run_monte_carlo, simulate_paths


def series(values: list[float]) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=len(values), freq="B")
    return pd.Series(values, index=idx)


def demo_returns(days: int = 300) -> pd.Series:
    rng = np.random.default_rng(7)
    return pd.Series(
        rng.normal(0.0003, 0.011, size=days),
        index=pd.date_range("2022-01-03", periods=days, freq="B"),
    )


class TestSimulatePaths:
    def test_shape_and_start_value(self):
        paths = simulate_paths(demo_returns(), horizon=10, n_paths=150, seed=1)
        assert paths.shape == (150, 11)
        assert np.all(paths[:, 0] == 100.0)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            simulate_paths(demo_returns(), method="garch")


class TestConstantReturns:
    def test_every_path_compounds_to_the_same_final_value(self):
        # the sample contains only 0.01, so every bootstrap draw is 0.01
        # and each path ends at 100 * 1.01^3 = 100 * 1.030301 = 103.0301
        paths = simulate_paths(series([0.01] * 120), horizon=3, n_paths=200, seed=3)
        assert paths[:, -1] == pytest.approx(np.full(200, 103.0301))

    def test_percentiles_collapse_and_no_loss(self):
        report = run_monte_carlo("const", series([0.01] * 120), horizon=3, n_paths=200, seed=3)
        # 103.0301 rounded to two decimals
        assert all(value == 103.03 for value in report.percentiles.values())
        assert report.prob_loss == 0.0


class TestZeroReturns:
    def test_paths_stay_flat_at_the_start_value(self):
        paths = simulate_paths(series([0.0] * 150), horizon=5, n_paths=120, seed=2)
        assert np.all(paths == 100.0)
        report = run_monte_carlo("flat", series([0.0] * 150), horizon=5, n_paths=120, seed=2)
        assert report.prob_loss == 0.0
        assert report.expected_final == 100.0


class TestRunMonteCarlo:
    @pytest.mark.parametrize("method", ["bootstrap", "normal"])
    def test_percentiles_are_ordered(self, method):
        report = run_monte_carlo(
            "demo", demo_returns(), horizon=60, n_paths=500, method=method, seed=11
        )
        p = report.percentiles
        assert p["p5"] <= p["p25"] <= p["p50"] <= p["p75"] <= p["p95"]
        assert 0.0 <= report.prob_loss <= 1.0

    def test_same_seed_gives_an_identical_report(self):
        first = run_monte_carlo("demo", demo_returns(), horizon=30, n_paths=300, seed=42)
        second = run_monte_carlo("demo", demo_returns(), horizon=30, n_paths=300, seed=42)
        assert first == second

    def test_different_seeds_differ(self):
        first = run_monte_carlo("demo", demo_returns(), horizon=30, n_paths=300, seed=1)
        second = run_monte_carlo("demo", demo_returns(), horizon=30, n_paths=300, seed=2)
        assert first != second


class TestGuards:
    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            run_monte_carlo("demo", demo_returns(), method="garch")

    def test_short_history_raises(self):
        with pytest.raises(ValueError, match="100 historical returns"):
            run_monte_carlo("demo", series([0.01] * 99))

    def test_zero_horizon_raises(self):
        with pytest.raises(ValueError, match="Horizon"):
            run_monte_carlo("demo", demo_returns(), horizon=0)

    def test_too_few_paths_raises(self):
        with pytest.raises(ValueError, match="100 paths"):
            run_monte_carlo("demo", demo_returns(), n_paths=50)
