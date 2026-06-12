"""Closed-form optimization checked against hand-computed weights."""

import numpy as np
import pandas as pd
import pytest

from analytics.optimize import (
    minimum_variance_weights,
    optimize_portfolio,
    tangency_weights,
)


def orthogonal_returns() -> pd.DataFrame:
    """Two return series whose sample covariance is exactly zero.

    Each series is built as mean + deviation per element. The deviation
    vectors are orthogonal by construction, their dot product is
    0.02 * 0.01 - 0.02 * 0.01 - 0.02 * 0.01 + 0.02 * 0.01
    = 0.0002 - 0.0002 - 0.0002 + 0.0002 = 0 exactly, so the sample
    covariance matrix is diagonal and the closed forms reduce to
    inverse-variance weighting.

    Sample variances with ddof=1 (n = 4 observations):
    var_A = 4 * 0.02^2 / 3 = 0.0016 / 3
    var_B = 4 * 0.01^2 / 3 = 0.0004 / 3
    """
    return pd.DataFrame(
        {
            "A": 0.01 + np.array([0.02, -0.02, 0.02, -0.02]),
            "B": 0.005 + np.array([0.01, 0.01, -0.01, -0.01]),
        }
    )


def random_returns(assets: int = 3, days: int = 120) -> pd.DataFrame:
    # seed 42 keeps every annualized sample mean positive, so the
    # tangency denominator is positive and the portfolio is defined
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        rng.normal(0.0005, 0.01, size=(days, assets)),
        columns=[f"S{i}" for i in range(assets)],
    )


def random_prices(assets: int = 3, days: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    return pd.DataFrame(
        100.0 * np.cumprod(1 + rng.normal(0.0005, 0.01, size=(days, assets)), axis=0),
        columns=[f"S{i}" for i in range(assets)],
        index=pd.date_range("2024-01-02", periods=days, freq="B"),
    )


class TestMinimumVariance:
    def test_hand_computed_inverse_variance_weights(self):
        # diagonal covariance, so w_A = var_B / (var_A + var_B). The
        # 4/3 from ddof=1 and the 252 annualization cancel in the
        # ratio: w_A = 0.0001 / (0.0004 + 0.0001) = 0.2
        weights = minimum_variance_weights(orthogonal_returns())
        assert weights["A"] == pytest.approx(0.2)
        assert weights["B"] == pytest.approx(0.8)

    def test_weights_sum_to_one(self):
        weights = minimum_variance_weights(random_returns())
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_singular_covariance_rejected(self):
        # B is an exact multiple of A, so the covariance matrix has
        # rank 1 and the linear system has no unique solution
        base = random_returns(assets=1)["S0"]
        frame = pd.DataFrame({"A": base, "B": 2.0 * base})
        with pytest.raises(ValueError, match="singular"):
            minimum_variance_weights(frame)


class TestTangency:
    def test_hand_computed_weights(self):
        # diagonal covariance and rf = 0, so the raw weights are
        # mu_i / var_i (the 252 annualization cancels between mu and
        # var): A: 0.01 / (0.0016 / 3) = 18.75 and
        # B: 0.005 / (0.0004 / 3) = 37.5, normalized by the sum 56.25
        # to w_A = 18.75 / 56.25 = 1/3 and w_B = 37.5 / 56.25 = 2/3
        weights = tangency_weights(orthogonal_returns())
        assert weights["A"] == pytest.approx(1 / 3)
        assert weights["B"] == pytest.approx(2 / 3)

    def test_weights_sum_to_one(self):
        weights = tangency_weights(random_returns())
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_all_negative_excess_returns_rejected(self):
        # flip the signs of the means, every excess return is negative
        # and the normalization denominator turns negative
        frame = -orthogonal_returns()
        with pytest.raises(ValueError, match="excess return"):
            tangency_weights(frame)


class TestOptimizePortfolio:
    def test_report_structure(self):
        report = optimize_portfolio(random_prices(), risk_free_rate=0.02)
        assert report.symbols == ["S0", "S1", "S2"]
        assert report.observations == 119
        assert report.risk_free_rate == 0.02
        for portfolio in (report.minimum_variance, report.maximum_sharpe):
            assert sum(portfolio.weights.values()) == pytest.approx(1.0, abs=1e-3)
            assert portfolio.volatility_pct > 0
        assert report.minimum_variance.volatility_pct <= report.maximum_sharpe.volatility_pct

    def test_single_asset_rejected(self):
        with pytest.raises(ValueError, match="at least 2 assets"):
            optimize_portfolio(random_prices(assets=1))

    def test_short_history_rejected(self):
        with pytest.raises(ValueError, match="at least 60"):
            optimize_portfolio(random_prices(days=50))
