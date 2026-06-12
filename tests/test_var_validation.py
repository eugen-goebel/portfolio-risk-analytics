"""Kupiec test and rolling VaR backtest checked against hand-computed values."""

import numpy as np
import pandas as pd
import pytest

from analytics.var_validation import kupiec_pof, rolling_var_breaches, validate_var


def series(values) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=len(values), freq="B")
    return pd.Series(values, index=idx)


def normal_returns(size: int, seed: int, std: float = 0.01) -> pd.Series:
    rng = np.random.default_rng(seed)
    return series(rng.normal(0.0, std, size=size))


class TestKupiecPof:
    def test_hand_computed_near_expected_count(self):
        # n = 250, x = 13, p = 0.05, expected breaches 12.5, so the
        # statistic is tiny:
        # null:  -2 * (237 * ln(0.95) + 13 * ln(0.05))
        #      = -2 * (237 * -0.0512933 + 13 * -2.9957323) = 102.2020607
        # alt:    2 * (237 * ln(237/250) + 13 * ln(13/250))
        #      =  2 * (237 * -0.0534008 + 13 * -2.9565116) = -102.1812687
        # LR = 102.2020607 - 102.1812687 = 0.0207919
        statistic, p_value = kupiec_pof(250, 13)
        assert statistic == pytest.approx(0.0207919, rel=1e-3)
        assert p_value > 0.5

    def test_severe_underestimation_is_rejected(self):
        # 40 breaches against 12.5 expected: the model badly understates
        # risk, the statistic is far beyond any reasonable threshold
        statistic, p_value = kupiec_pof(250, 40)
        assert statistic > 30
        assert p_value < 0.001
        # validate_var flags model_rejected at p_value < 0.05, which
        # such counts clear by many orders of magnitude
        assert p_value < 0.05

    def test_zero_breaches_edge_case(self):
        # x = 0 drops the x * ln(x/n) term, so
        # LR = -2 * n * ln(1 - p) = -2 * 100 * ln(0.95) = 10.2586589
        statistic, p_value = kupiec_pof(100, 0)
        assert statistic == pytest.approx(10.2586589, rel=1e-3)
        # zero breaches in 100 days is suspiciously conservative
        assert p_value < 0.05

    def test_all_breaches_edge_case(self):
        # x = n drops the (n - x) terms, so
        # LR = -2 * n * ln(p) = -2 * 100 * ln(0.05) = 599.1464547
        statistic, p_value = kupiec_pof(100, 100)
        assert statistic == pytest.approx(599.1464547, rel=1e-3)
        assert p_value < 1e-100

    def test_breach_rate_equal_to_p_scores_zero(self):
        statistic, p_value = kupiec_pof(100, 5)
        assert statistic == pytest.approx(0.0, abs=1e-9)
        assert p_value == pytest.approx(1.0)

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            kupiec_pof(0, 0)
        with pytest.raises(ValueError):
            kupiec_pof(100, 101)
        with pytest.raises(ValueError):
            kupiec_pof(100, 5, confidence=1.0)


class TestRollingVarBreaches:
    def test_no_lookahead_last_day_crash_adds_exactly_one_breach(self):
        # the VaR tested on the last day is estimated from the window
        # strictly before it, so changing only the last return changes
        # only whether that one day breaches
        base = normal_returns(120, seed=7).to_numpy()
        mild = base.copy()
        mild[-1] = 0.0
        crash = base.copy()
        crash[-1] = -0.5
        obs_mild, breaches_mild = rolling_var_breaches(series(mild), window=100)
        obs_crash, breaches_crash = rolling_var_breaches(series(crash), window=100)
        assert obs_mild == obs_crash == 20
        assert breaches_crash == breaches_mild + 1

    def test_breach_rate_in_sane_band_on_normal_data(self):
        returns = normal_returns(1000, seed=11)
        observations, breaches = rolling_var_breaches(returns, window=250)
        assert observations == 750
        rate = breaches / observations * 100
        assert 1 <= rate <= 12


class TestValidateVar:
    def test_report_structure(self):
        returns = normal_returns(1000, seed=11)
        report = validate_var("demo", returns, window=250)
        assert report.symbol == "demo"
        assert report.confidence == 0.95
        assert report.window == 250
        assert report.observations == 750
        assert report.expected_breaches == pytest.approx(37.5)
        assert report.actual_breaches >= 0
        assert report.breach_rate_pct == pytest.approx(
            report.actual_breaches / report.observations * 100, abs=0.01
        )
        assert report.kupiec_statistic >= 0
        assert 0 <= report.p_value <= 1
        assert isinstance(report.model_rejected, bool)

    def test_short_series_raises(self):
        returns = normal_returns(299, seed=3)
        with pytest.raises(ValueError, match="out-of-sample"):
            validate_var("demo", returns, window=250)
