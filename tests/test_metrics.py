"""Metric functions checked against hand-computed values."""

import math

import pandas as pd
import pytest

from analytics.metrics import (
    annualized_volatility,
    correlation_matrix,
    daily_returns,
    max_drawdown,
    portfolio_returns,
    sharpe_ratio,
    summarize,
    total_return,
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
