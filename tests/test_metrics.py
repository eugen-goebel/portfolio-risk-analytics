"""Metric functions checked against hand-computed values."""

import math

import pandas as pd
import pytest

from analytics.metrics import (
    annualized_volatility,
    beta,
    capm_alpha,
    compare_to_benchmark,
    correlation_matrix,
    daily_returns,
    drawdown_series,
    expected_shortfall,
    historical_var,
    information_ratio,
    max_drawdown,
    portfolio_returns,
    sharpe_ratio,
    summarize,
    total_return,
    tracking_error,
)


def series(values: list[float]) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=len(values), freq="B")
    return pd.Series(values, index=idx)


class TestDailyReturns:
    def test_simple_returns(self):
        r = daily_returns(series([100.0, 110.0, 99.0]))
        assert r.iloc[0] == pytest.approx(0.10)
        assert r.iloc[1] == pytest.approx(-0.10)

    def test_first_nan_dropped(self):
        assert len(daily_returns(series([100.0, 101.0, 102.0]))) == 2


class TestVolatility:
    def test_hand_computed(self):
        # returns alternate +1% and -1%: mean 0, sample std 0.01 * sqrt(4/3)
        r = pd.Series([0.01, -0.01, 0.01, -0.01])
        expected = 0.01 * math.sqrt(4 / 3) * math.sqrt(252)
        assert annualized_volatility(r) == pytest.approx(expected)

    def test_single_observation_is_zero(self):
        assert annualized_volatility(pd.Series([0.01])) == 0.0


class TestSharpe:
    def test_zero_volatility_guard(self):
        assert sharpe_ratio(pd.Series([0.001, 0.001, 0.001])) == 0.0

    def test_risk_free_rate_lowers_sharpe(self):
        r = daily_returns(series([100, 101, 100.5, 102, 103, 102.5, 104]))
        assert sharpe_ratio(r, risk_free_rate=0.05) < sharpe_ratio(r, risk_free_rate=0.0)


class TestDrawdownSeries:
    def test_running_drawdown(self):
        dd = drawdown_series(series([100.0, 120.0, 90.0, 100.0]))
        assert dd.iloc[0] == pytest.approx(0.0)
        assert dd.iloc[2] == pytest.approx(-0.25)
        # partial recovery: 100/120 - 1
        assert dd.iloc[3] == pytest.approx(-1 / 6)


class TestMaxDrawdown:
    def test_peak_to_trough(self):
        # peak 120, trough 90: drawdown 90/120 - 1 = -25%
        assert max_drawdown(series([100.0, 120.0, 90.0, 100.0])) == pytest.approx(-0.25)

    def test_monotonic_rise_has_no_drawdown(self):
        assert max_drawdown(series([100.0, 105.0, 110.0])) == pytest.approx(0.0)

    def test_empty_series(self):
        assert max_drawdown(pd.Series(dtype=float)) == 0.0


class TestTotalReturn:
    def test_ten_percent(self):
        assert total_return(series([100.0, 104.0, 110.0])) == pytest.approx(0.10)


class TestCorrelation:
    def test_perfectly_correlated(self):
        a = series([100.0, 110.0, 105.0, 115.0])
        frame = pd.DataFrame({"a": a, "b": a * 2})
        corr = correlation_matrix(frame)
        assert corr.loc["a", "b"] == pytest.approx(1.0)


class TestPortfolioReturns:
    def test_offsetting_positions(self):
        frame = pd.DataFrame({"a": series([100.0, 110.0]), "b": series([100.0, 90.0])})
        r = portfolio_returns(frame, {"a": 0.5, "b": 0.5})
        assert r.iloc[0] == pytest.approx(0.0)

    def test_weights_must_sum_to_one(self):
        frame = pd.DataFrame({"a": series([100.0, 110.0])})
        with pytest.raises(ValueError, match="sum to 1"):
            portfolio_returns(frame, {"a": 0.5})

    def test_unknown_symbol_rejected(self):
        frame = pd.DataFrame({"a": series([100.0, 110.0])})
        with pytest.raises(ValueError, match="No price data"):
            portfolio_returns(frame, {"a": 0.5, "zzz": 0.5})


class TestSummarize:
    def test_summary_fields(self):
        s = summarize("test", series([100.0, 120.0, 90.0, 100.0]))
        assert s.symbol == "test"
        assert s.observations == 4
        assert s.total_return_pct == pytest.approx(0.0)
        assert s.max_drawdown_pct == pytest.approx(-25.0)
        # returns are 0.2, -0.25 and 1/9. The 5% quantile interpolates a
        # tenth of the way from -0.25 toward 1/9: -0.25 + 0.1 * (1/9 + 0.25)
        assert s.var_95_pct == pytest.approx(21.39)
        # only -0.25 lies at or below that quantile
        assert s.expected_shortfall_95_pct == pytest.approx(25.0)


class TestTailRisk:
    # 20 returns whose two smallest values are -0.05 and -0.03. With linear
    # interpolation the 5% quantile sits at sorted position 0.05 * 19 = 0.95,
    # between those two: -0.05 + 0.95 * 0.02 = -0.031.
    TAIL_RETURNS = [
        0.010, -0.050, 0.002, 0.015, -0.010, 0.007, 0.020, -0.030, 0.005, 0.012,
        0.003, -0.008, 0.018, 0.001, 0.009, -0.004, 0.011, 0.006, 0.013, 0.004,
    ]  # fmt: skip

    def test_var_hand_computed(self):
        assert historical_var(pd.Series(self.TAIL_RETURNS)) == pytest.approx(0.031)

    def test_expected_shortfall_hand_computed(self):
        # only -0.05 lies at or below the -0.031 quantile
        assert expected_shortfall(pd.Series(self.TAIL_RETURNS)) == pytest.approx(0.05)

    def test_shortfall_never_below_var(self):
        r = pd.Series(self.TAIL_RETURNS)
        assert expected_shortfall(r) >= historical_var(r)

    def test_all_positive_returns_floor_at_zero(self):
        # a strictly positive quantile flips to a negative loss, floored to 0.0
        r = pd.Series([0.010, 0.020, 0.015, 0.005, 0.012])
        assert historical_var(r) == 0.0
        assert expected_shortfall(r) == 0.0

    def test_single_observation_is_zero(self):
        assert historical_var(pd.Series([-0.02])) == 0.0
        assert expected_shortfall(pd.Series([-0.02])) == 0.0


class TestBenchmark:
    BENCH_RETURNS = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01])

    def test_doubled_returns_have_beta_two(self):
        # cov(2x, x) = 2 var(x), so beta is exactly 2
        assert beta(self.BENCH_RETURNS * 2, self.BENCH_RETURNS) == 2.0

    def test_pure_beta_exposure_has_zero_alpha(self):
        # asset_annual = 2 * bench_annual, so alpha = 2b - 2b = 0
        assert capm_alpha(self.BENCH_RETURNS * 2, self.BENCH_RETURNS) == pytest.approx(0.0)

    def test_identical_series(self):
        # a perfect tracker: beta 1, no active risk, ratio guarded to 0
        assert beta(self.BENCH_RETURNS, self.BENCH_RETURNS) == pytest.approx(1.0)
        assert tracking_error(self.BENCH_RETURNS, self.BENCH_RETURNS) == 0.0
        assert information_ratio(self.BENCH_RETURNS, self.BENCH_RETURNS) == 0.0

    def test_flat_benchmark_has_zero_beta(self):
        flat = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0])
        assert beta(self.BENCH_RETURNS, flat) == 0.0

    def test_compare_aligns_mismatched_date_ranges(self):
        idx = pd.date_range("2024-01-01", periods=70, freq="B")
        asset = pd.Series([100.0 + i for i in range(60)], index=idx[:60])
        bench = pd.Series([50.0 + i for i in range(60)], index=idx[10:70])
        report = compare_to_benchmark("asset", asset, "bench", bench)
        # 50 common dates leave 49 return observations
        assert report.observations == 49
        assert report.benchmark == "bench"

    def test_compare_rejects_short_overlap(self):
        idx = pd.date_range("2024-01-01", periods=50, freq="B")
        asset = pd.Series([100.0 + i for i in range(35)], index=idx[:35])
        bench = pd.Series([100.0 + i for i in range(35)], index=idx[15:50])
        # only 20 common dates, below the 30 required
        with pytest.raises(ValueError, match="common observations"):
            compare_to_benchmark("asset", asset, "bench", bench)
