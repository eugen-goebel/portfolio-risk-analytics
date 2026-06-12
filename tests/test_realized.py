"""Realized volatility checked against hand-computed values."""

import numpy as np
import pandas as pd
import pytest

from analytics.realized import annualized_realized_volatility_pct, realized_volatility


def closes(points: dict[str, float]) -> pd.Series:
    idx = pd.DatetimeIndex([pd.Timestamp(t, tz="UTC") for t in points])
    return pd.Series(list(points.values()), index=idx)


SINGLE_DAY = {
    "2024-01-02 10:00": 100.0,
    "2024-01-02 11:00": 101.0,
    "2024-01-02 12:00": 99.99,
}


class TestRealizedVolatility:
    def test_hand_computed_single_day(self):
        # closes 100 -> 101 -> 99.99: returns 101/100 - 1 = 0.01 and
        # 99.99/101 - 1 = -0.01, so RV = sqrt(0.01^2 + (-0.01)^2)
        #                              = sqrt(0.0002) = 0.0141421
        rv = realized_volatility(closes(SINGLE_DAY))
        assert len(rv) == 1
        assert rv.index[0] == pd.Timestamp("2024-01-02")
        assert rv.iloc[0] == pytest.approx(0.0141421, abs=1e-7)

    def test_overnight_gap_is_excluded(self):
        # day one ends at 101, day two opens at 200: the overnight jump
        # 200/101 - 1 must not enter. Day two carries only the return
        # 202/200 - 1 = 0.01, so its RV is sqrt(0.01^2) = 0.01 exactly.
        rv = realized_volatility(
            closes(
                {
                    "2024-01-02 10:00": 100.0,
                    "2024-01-02 11:00": 101.0,
                    "2024-01-03 10:00": 200.0,
                    "2024-01-03 11:00": 202.0,
                }
            )
        )
        assert rv.loc[pd.Timestamp("2024-01-03")] == pytest.approx(0.01)

    def test_single_observation_days_are_dropped(self):
        rv = realized_volatility(
            closes(
                {
                    "2024-01-02 10:00": 100.0,
                    "2024-01-03 10:00": 100.0,
                    "2024-01-03 11:00": 101.0,
                }
            )
        )
        assert list(rv.index) == [pd.Timestamp("2024-01-03")]

    def test_unsorted_input_is_handled(self):
        shuffled = closes(SINGLE_DAY).sort_values()
        rv = realized_volatility(shuffled)
        assert rv.iloc[0] == pytest.approx(0.0141421, abs=1e-7)


class TestAnnualizedRealizedVolatilityPct:
    def test_scales_daily_series(self):
        # 0.0141421 * sqrt(252) * 100 = 22.4497
        annualized = annualized_realized_volatility_pct(closes(SINGLE_DAY))
        expected = np.sqrt(0.0002) * np.sqrt(252) * 100
        assert annualized.iloc[0] == pytest.approx(expected)
        assert annualized.iloc[0] == pytest.approx(22.4497, abs=1e-3)
